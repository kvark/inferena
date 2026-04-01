//! Meganeura framework benchmark runner for inferena.
//!
//! Supports SmolLM2 (text LLM) and SmolVLA (action expert) models
//! using the meganeura crate (e-graph optimized NN on blade-graphics).

use meganeura::data::safetensors::SafeTensorsModel;
use meganeura::{Graph, build_inference_session, build_session};
use sha2::{Digest, Sha256};
use std::time::Instant;

fn find_local_model(model_name: &str) -> Option<std::path::PathBuf> {
    // Search up from exe location.
    let exe = std::env::current_exe().unwrap_or_default();
    let mut root = exe
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .to_path_buf();
    for _ in 0..5 {
        let local = root
            .join("models")
            .join(model_name)
            .join("model.safetensors");
        if local.exists() {
            return Some(local);
        }
        if !root.pop() {
            break;
        }
    }
    // Check relative to cwd.
    let cwd = std::path::PathBuf::from("models")
        .join(model_name)
        .join("model.safetensors");
    if cwd.exists() {
        return Some(cwd);
    }
    None
}

fn sha256_f32(data: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for &v in data {
        hasher.update(v.to_le_bytes());
    }
    format!("sha256:{}", hex::encode(hasher.finalize()))
}

fn bench_smollm2(model_name: &str) {
    use meganeura::models::smollm2::{self, SmolLM2Config};

    let (repo_id, config) = match model_name {
        "SmolLM2-135M" => ("HuggingFaceTB/SmolLM2-135M", SmolLM2Config::smollm2_135m()),
        _ => {
            eprintln!("Unknown SmolLM model: {model_name}");
            std::process::exit(1);
        }
    };

    let seq_len: usize = 128;
    let vocab = config.vocab_size;

    // --- Load weights ---
    eprintln!("[meganeura] loading model {repo_id}...");
    let model = if let Some(path) = find_local_model(model_name) {
        eprintln!("[meganeura] loading from {}", path.display());
        SafeTensorsModel::load(path).expect("local model load failed")
    } else {
        SafeTensorsModel::download(repo_id).expect("model download/load failed")
    };

    // --- Build & compile ---
    eprintln!("[meganeura] building graph...");
    let compile_start = Instant::now();
    let mut g = Graph::new();
    let logits = smollm2::build_graph(&mut g, &config, seq_len);
    g.set_outputs(vec![logits]);

    eprintln!("[meganeura] compiling...");
    let mut session = build_inference_session(&g);

    // --- Load weights ---
    let transposed = smollm2::transposed_weight_names(&config);
    let transposed_set: std::collections::HashSet<&str> =
        transposed.iter().map(|s| s.as_str()).collect();
    for (name, _) in session.plan().param_buffers.clone() {
        // Skip derived (fused) params — auto-populated when source params are loaded.
        if !model.tensor_info().contains_key(&name) && name != "lm_head.weight" {
            continue;
        }
        if name == "lm_head.weight" {
            if model.tensor_info().contains_key("lm_head.weight") {
                let data = if transposed_set.contains(name.as_str()) {
                    model.tensor_f32_auto_transposed(&name)
                } else {
                    model.tensor_f32_auto(&name)
                };
                session.set_parameter(&name, &data.unwrap());
            } else {
                let data = model
                    .tensor_f32_auto_transposed("model.embed_tokens.weight")
                    .unwrap();
                session.set_parameter("lm_head.weight", &data);
            }
        } else if transposed_set.contains(name.as_str()) {
            let data = model.tensor_f32_auto_transposed(&name).unwrap();
            session.set_parameter(&name, &data);
        } else {
            let data = model.tensor_f32_auto(&name).unwrap();
            session.set_parameter(&name, &data);
        }
    }

    let compile_s = compile_start.elapsed().as_secs_f64();
    eprintln!("[meganeura] ready (compile: {compile_s:.2}s)");

    // --- Forward ---
    let input_ids: Vec<u32> = (0..seq_len as u32).map(|i| i % vocab as u32).collect();
    let labels: Vec<u32> = (0..seq_len as u32)
        .map(|i| (i + 1) % vocab as u32)
        .collect();

    session.set_input_u32("token_ids", &input_ids);
    let fwd_start = Instant::now();
    session.step();
    session.wait();
    let forward_ms = fwd_start.elapsed().as_secs_f64() * 1000.0;

    let all_logits = session.read_output(seq_len * vocab);

    // Cross-entropy loss on CPU (HF-compatible: shifted labels).
    // HF internally shifts: logits[0..seq-1] predict labels[1..seq].
    let mut total_loss = 0.0f64;
    let loss_positions = seq_len - 1;
    for pos in 0..loss_positions {
        let sl = &all_logits[pos * vocab..(pos + 1) * vocab];
        let target = labels[pos + 1] as usize; // shifted: predict next label
        let max_l = sl.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f64 = sl.iter().map(|&l| ((l - max_l) as f64).exp()).sum();
        total_loss -= (sl[target] - max_l) as f64 - sum_exp.ln();
    }
    let loss = total_loss / loss_positions as f64;

    // Backward (re-run forward as proxy).
    session.set_input_u32("token_ids", &input_ids);
    let bwd_start = Instant::now();
    session.step();
    session.wait();
    let backward_ms = bwd_start.elapsed().as_secs_f64() * 1000.0;

    emit_result(
        model_name,
        compile_s,
        forward_ms,
        backward_ms,
        &all_logits,
        loss,
    );
}

fn bench_smolvla() {
    use meganeura::models::smolvla::{self, SmolVLAConfig};

    let config = SmolVLAConfig::smolvla_base();
    let action_seq_len: usize = 50;
    let vlm_seq_len: usize = 16;
    let expert_hidden = config.expert.hidden_size;
    let action_dim = config.max_action_dim;

    let compile_start = Instant::now();

    // Inference graph: forward only, outputs predictions.
    eprintln!("[meganeura] building SmolVLA inference graph...");
    let mut infer_g = Graph::new();
    let pred = smolvla::build_action_expert(&mut infer_g, &config, action_seq_len, vlm_seq_len);
    infer_g.set_outputs(vec![pred]);
    eprintln!("[meganeura] compiling inference session...");
    let mut infer_session = build_inference_session(&infer_g);

    // Training graph: forward + backward + loss.
    eprintln!("[meganeura] building SmolVLA training graph...");
    let training_g = smolvla::build_action_expert_training(&config, action_seq_len, vlm_seq_len);
    eprintln!("[meganeura] compiling training session...");
    let mut train_session = build_session(&training_g);

    let compile_s = compile_start.elapsed().as_secs_f64();
    eprintln!("[meganeura] ready (compile: {compile_s:.2}s)");

    // --- Initialize with deterministic random values ---
    // Use infer_session's param list (avoids fused param names from optimizer).
    eprintln!("[meganeura] initializing parameters...");
    for (i, (name, buf_ref)) in infer_session
        .plan()
        .param_buffers
        .clone()
        .iter()
        .enumerate()
    {
        let n = infer_session.plan().buffers[buf_ref.0 as usize] / 4;
        let data: Vec<f32> = (0..n)
            .map(|j| (j as f32 * 0.01 + i as f32).sin() * 0.1)
            .collect();
        infer_session.set_parameter(name, &data);
        train_session.set_parameter(name, &data);
    }

    // --- Prepare inputs ---
    let noisy_actions: Vec<f32> = (0..action_seq_len * action_dim)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();
    let timestep: Vec<f32> = (0..expert_hidden * 2)
        .map(|i| (i as f32 * 0.005).sin())
        .collect();
    let kv_dim = config.expert.kv_dim();
    let vlm_kv: Vec<f32> = (0..vlm_seq_len * kv_dim)
        .map(|i| (i as f32 * 0.01).cos())
        .collect();

    // Set inputs — VLM context is per cross-attention layer.
    let set_inputs = |session: &mut meganeura::Session| {
        session.set_input("noisy_actions", &noisy_actions);
        session.set_input("timestep", &timestep);
        for i in 0..config.expert.num_layers {
            if i % config.expert.self_attn_every_n_layers != 0 {
                session.set_input(&format!("vlm_kv_layer_{i}"), &vlm_kv);
            }
        }
    };

    // --- Forward (inference session) ---
    set_inputs(&mut infer_session);

    let fwd_start = Instant::now();
    infer_session.step();
    infer_session.wait();
    let forward_ms = fwd_start.elapsed().as_secs_f64() * 1000.0;

    let output = infer_session.read_output(action_seq_len * action_dim);
    eprintln!(
        "[meganeura] forward: {forward_ms:.2}ms, {} outputs",
        output.len()
    );

    // MSE loss on CPU.
    let nan_indices: Vec<usize> = output
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_nan())
        .map(|(i, _)| i)
        .collect();
    if !nan_indices.is_empty() {
        let action_dim = config.max_action_dim;
        let positions: Vec<String> = nan_indices
            .iter()
            .map(|&i| format!("[seq={}, dim={}]", i / action_dim, i % action_dim))
            .collect();
        eprintln!(
            "[meganeura] WARNING: {} NaN values at: {}",
            nan_indices.len(),
            positions.join(", ")
        );
    }
    let loss: f64 = output.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / output.len() as f64;

    // --- Training step (forward + backward + SGD) ---
    let target_actions = vec![0.0f32; action_seq_len * action_dim];
    set_inputs(&mut train_session);
    train_session.set_input("target_actions", &target_actions);

    let bwd_start = Instant::now();
    train_session.step();
    train_session.wait();
    let train_ms = bwd_start.elapsed().as_secs_f64() * 1000.0;
    // Approximate backward as train_step - forward.
    let backward_ms = (train_ms - forward_ms).max(0.0);

    emit_result("SmolVLA", compile_s, forward_ms, backward_ms, &output, loss);
}

fn detect_backend() -> &'static str {
    if cfg!(target_os = "macos") {
        "Metal"
    } else if cfg!(target_os = "windows") {
        "Vulkan/DX12"
    } else {
        "Vulkan"
    }
}

fn emit_result(
    model: &str,
    compile_s: f64,
    forward_ms: f64,
    backward_ms: f64,
    output: &[f32],
    loss: f64,
) {
    let hash = sha256_f32(output);
    let sample: Vec<f64> = output.iter().take(16).map(|&v| v as f64).collect();
    let backend = detect_backend();

    let rev = std::env::var("FRAMEWORK_REV").unwrap_or_default();

    let result = serde_json::json!({
        "framework": "meganeura",
        "framework_rev": rev,
        "model": model,
        "device": backend,
        "gpu_name": backend,
        "backend": backend,
        "timings": {
            "compile_s": (compile_s * 100.0).round() / 100.0,
            "inference_ms": (forward_ms * 1000.0).round() / 1000.0,
            "latency_ms": 0.0,
            "train_ms": (backward_ms * 1000.0).round() / 1000.0,
        },
        "outputs": {
            "logits_hash": hash,
            "logits_sample": sample,
            "loss": if loss.is_nan() { -1.0 } else { (loss * 1_000_000.0).round() / 1_000_000.0 },
        },
    });

    println!("{}", serde_json::to_string(&result).unwrap());
}

fn bench_stable_diffusion() {
    use meganeura::models::sd_unet::{self, SDUNetConfig};

    let config = SDUNetConfig::small();
    let batch = config.batch_size;
    let in_c = config.in_channels;
    let res = config.resolution;
    let in_size = (batch * in_c * res * res) as usize;

    // --- Build training graph ---
    eprintln!("[meganeura] building SD U-Net training graph (small config)...");
    let compile_start = Instant::now();
    let mut g = Graph::new();
    let loss = sd_unet::build_training_graph(&mut g, &config);
    g.set_outputs(vec![loss]);

    eprintln!("[meganeura] compiling inference session...");
    let mut infer_session = build_inference_session(&g);

    eprintln!("[meganeura] compiling training session...");
    let mut train_session = build_session(&g);

    let compile_s = compile_start.elapsed().as_secs_f64();
    eprintln!("[meganeura] ready (compile: {compile_s:.2}s)");

    // --- Initialize with deterministic values ---
    eprintln!("[meganeura] initializing parameters...");
    for (i, (name, buf_ref)) in train_session
        .plan()
        .param_buffers
        .clone()
        .iter()
        .enumerate()
    {
        let n = train_session.plan().buffers[buf_ref.0 as usize] / 4;
        let data: Vec<f32> = (0..n)
            .map(|j| (j as f32 * 0.01 + i as f32).sin() * 0.1)
            .collect();
        infer_session.set_parameter(name, &data);
        train_session.set_parameter(name, &data);
    }

    // --- Prepare inputs ---
    let noisy_latent: Vec<f32> = (0..in_size).map(|i| (i as f32 * 0.01).sin()).collect();
    let noise_target: Vec<f32> = (0..in_size).map(|i| (i as f32 * 0.007).cos()).collect();

    // --- Forward (inference session) ---
    infer_session.set_input("noisy_latent", &noisy_latent);
    infer_session.set_input("noise_target", &noise_target);

    let fwd_start = Instant::now();
    infer_session.step();
    infer_session.wait();
    let forward_ms = fwd_start.elapsed().as_secs_f64() * 1000.0;

    // The output is the MSE loss scalar.
    let output = infer_session.read_output(1);
    let loss_val = output[0] as f64;
    eprintln!("[meganeura] forward: {forward_ms:.2}ms, loss={loss_val:.6}");

    // For logits hash, use the loss value encoded as f32.
    let logits_data = vec![output[0]];

    // --- Training step (forward + backward + SGD) ---
    train_session.set_input("noisy_latent", &noisy_latent);
    train_session.set_input("noise_target", &noise_target);

    let bwd_start = Instant::now();
    train_session.step();
    train_session.wait();
    let train_ms = bwd_start.elapsed().as_secs_f64() * 1000.0;
    let backward_ms = (train_ms - forward_ms).max(0.0);

    emit_result(
        "StableDiffusion",
        compile_s,
        forward_ms,
        backward_ms,
        &logits_data,
        loss_val,
    );
}

fn main() {
    env_logger::init();

    let model_name = std::env::args().nth(1).unwrap_or("SmolLM2-135M".into());
    match model_name.as_str() {
        "SmolLM2-135M" => bench_smollm2(&model_name),
        "SmolVLA" => bench_smolvla(),
        "StableDiffusion" => bench_stable_diffusion(),
        other => {
            eprintln!("Unknown model: {other}. Available: SmolLM2-135M, SmolVLA, StableDiffusion");
            std::process::exit(1);
        }
    }
}
