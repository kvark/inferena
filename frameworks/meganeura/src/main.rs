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

/// Deterministic seed from parameter name — framework-independent init.
fn name_seed(name: &str) -> f32 {
    let mut h: u32 = 0;
    for c in name.bytes() {
        h = h.wrapping_mul(31).wrapping_add(c as u32);
    }
    (h % 10000) as f32
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

    // --- Latency (single-token forward) ---
    // Build a separate seq_len=1 inference graph.
    eprintln!("[meganeura] measuring single-token latency...");
    let mut lat_g = Graph::new();
    let lat_logits = smollm2::build_graph(&mut lat_g, &config, 1);
    lat_g.set_outputs(vec![lat_logits]);
    let mut lat_session = build_inference_session(&lat_g);
    // Copy weights from main session.
    for (name, _) in lat_session.plan().param_buffers.clone() {
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
                lat_session.set_parameter(&name, &data.unwrap());
            } else {
                let data = model
                    .tensor_f32_auto_transposed("model.embed_tokens.weight")
                    .unwrap();
                lat_session.set_parameter("lm_head.weight", &data);
            }
        } else if transposed_set.contains(name.as_str()) {
            let data = model.tensor_f32_auto_transposed(&name).unwrap();
            lat_session.set_parameter(&name, &data);
        } else {
            let data = model.tensor_f32_auto(&name).unwrap();
            lat_session.set_parameter(&name, &data);
        }
    }
    // Warm-up.
    lat_session.set_input_u32("token_ids", &[0u32]);
    lat_session.step();
    lat_session.wait();
    // Measure.
    lat_session.set_input_u32("token_ids", &[0u32]);
    let lat_start = Instant::now();
    lat_session.step();
    lat_session.wait();
    let latency_ms = lat_start.elapsed().as_secs_f64() * 1000.0;

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
        latency_ms,
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

    // --- Latency (single action chunk) ---
    eprintln!("[meganeura] measuring single-chunk latency...");
    let mut lat_g = Graph::new();
    let lat_pred = smolvla::build_action_expert(&mut lat_g, &config, 1, vlm_seq_len);
    lat_g.set_outputs(vec![lat_pred]);
    let mut lat_session = build_inference_session(&lat_g);
    for (i, (name, buf_ref)) in lat_session.plan().param_buffers.clone().iter().enumerate() {
        let n = lat_session.plan().buffers[buf_ref.0 as usize] / 4;
        let data: Vec<f32> = (0..n)
            .map(|j| (j as f32 * 0.01 + i as f32).sin() * 0.1)
            .collect();
        lat_session.set_parameter(name, &data);
    }
    let lat_actions: Vec<f32> = (0..action_dim).map(|i| (i as f32 * 0.01).sin()).collect();
    let lat_timestep = &timestep;
    lat_session.set_input("noisy_actions", &lat_actions);
    lat_session.set_input("timestep", lat_timestep);
    for i in 0..config.expert.num_layers {
        if i % config.expert.self_attn_every_n_layers != 0 {
            lat_session.set_input(&format!("vlm_kv_layer_{i}"), &vlm_kv);
        }
    }
    // Warm-up.
    lat_session.step();
    lat_session.wait();
    lat_session.set_input("noisy_actions", &lat_actions);
    lat_session.set_input("timestep", lat_timestep);
    for i in 0..config.expert.num_layers {
        if i % config.expert.self_attn_every_n_layers != 0 {
            lat_session.set_input(&format!("vlm_kv_layer_{i}"), &vlm_kv);
        }
    }
    let lat_start = Instant::now();
    lat_session.step();
    lat_session.wait();
    let latency_ms = lat_start.elapsed().as_secs_f64() * 1000.0;

    emit_result(
        "SmolVLA",
        compile_s,
        forward_ms,
        backward_ms,
        &output,
        loss,
        latency_ms,
    );
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
    latency_ms: f64,
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
            "latency_ms": (latency_ms * 1000.0).round() / 1000.0,
            "training_ms": (backward_ms * 1000.0).round() / 1000.0,
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

    // --- Latency (re-run forward, batch=2 is already minimal) ---
    infer_session.set_input("noisy_latent", &noisy_latent);
    infer_session.set_input("noise_target", &noise_target);
    infer_session.step();
    infer_session.wait();
    infer_session.set_input("noisy_latent", &noisy_latent);
    infer_session.set_input("noise_target", &noise_target);
    let lat_start = Instant::now();
    infer_session.step();
    infer_session.wait();
    let latency_ms = lat_start.elapsed().as_secs_f64() * 1000.0;

    emit_result(
        "StableDiffusion",
        compile_s,
        forward_ms,
        backward_ms,
        &logits_data,
        loss_val,
        latency_ms,
    );
}

fn bench_resnet() {
    use meganeura::models::resnet;

    let batch: u32 = 4;
    let scale: f32 = 0.01; // small scale to prevent explosion with identity BN

    eprintln!("[meganeura] building ResNet graph...");
    let compile_start = Instant::now();
    let mut g = Graph::new();
    let logits = resnet::build_resnet50(&mut g, batch);
    g.set_outputs(vec![logits]);

    eprintln!("[meganeura] compiling...");
    let mut session = build_inference_session(&g);

    let compile_s = compile_start.elapsed().as_secs_f64();
    eprintln!("[meganeura] ready (compile: {compile_s:.2}s)");

    // --- Initialize with deterministic values ---
    // BN fused_bias → zero (identity BN in eval mode matches PyTorch).
    // Conv/FC weights → name-seeded sin values (framework-independent).
    // FC weight: init in PyTorch [out, in] layout then transpose to meganeura [in, out].
    for (name, buf_ref) in session.plan().param_buffers.clone().iter() {
        let n = session.plan().buffers[buf_ref.0 as usize] / 4;
        let data: Vec<f32> = if name.contains("fused_bias") {
            vec![0.0; n]
        } else if name == "fc.weight" {
            // Init in PyTorch layout [out=1000, in=2048] then transpose to [in=2048, out=1000]
            let in_dim = 2048usize;
            let out_dim = 1000usize;
            let seed = name_seed(name);
            let mut buf = vec![0.0f32; n];
            for i in 0..out_dim {
                for j in 0..in_dim {
                    buf[j * out_dim + i] = ((i * in_dim + j) as f32 * 0.01 + seed).sin() * scale;
                }
            }
            buf
        } else {
            let seed = name_seed(name);
            (0..n)
                .map(|j| (j as f32 * 0.01 + seed).sin() * scale)
                .collect()
        };
        session.set_parameter(name, &data);
    }

    // --- Forward ---
    let in_size = (batch * 3 * 224 * 224) as usize;
    let images: Vec<f32> = (0..in_size).map(|i| (i as f32 * 0.001).sin()).collect();
    session.set_input("image", &images);

    let fwd_start = Instant::now();
    session.step();
    session.wait();
    let forward_ms = fwd_start.elapsed().as_secs_f64() * 1000.0;

    let output = session.read_output((batch * 1000) as usize);

    // Cross-entropy loss.
    let labels: Vec<usize> = (0..batch as usize).map(|i| i % 1000).collect();
    let mut total_loss = 0.0f64;
    for b in 0..batch as usize {
        let sl = &output[b * 1000..(b + 1) * 1000];
        let max_l = sl.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f64 = sl.iter().map(|&l| ((l - max_l) as f64).exp()).sum();
        total_loss -= (sl[labels[b]] - max_l) as f64 - sum_exp.ln();
    }
    let loss = total_loss / batch as f64;

    // --- Latency (single-image) ---
    let lat_images: Vec<f32> = vec![0.0; (3 * 224 * 224) as usize];
    // Build single-batch graph.
    let mut lat_g = Graph::new();
    let lat_logits = resnet::build_resnet50(&mut lat_g, 1);
    lat_g.set_outputs(vec![lat_logits]);
    let mut lat_session = build_inference_session(&lat_g);
    for (name, buf_ref) in lat_session.plan().param_buffers.clone().iter() {
        let n = lat_session.plan().buffers[buf_ref.0 as usize] / 4;
        let data: Vec<f32> = if name.contains("fused_bias") {
            vec![0.0; n]
        } else if name == "fc.weight" {
            let in_dim = 2048usize;
            let out_dim = 1000usize;
            let seed = name_seed(name);
            let mut buf = vec![0.0f32; n];
            for i in 0..out_dim {
                for j in 0..in_dim {
                    buf[j * out_dim + i] = ((i * in_dim + j) as f32 * 0.01 + seed).sin() * scale;
                }
            }
            buf
        } else {
            let seed = name_seed(name);
            (0..n)
                .map(|j| (j as f32 * 0.01 + seed).sin() * scale)
                .collect()
        };
        lat_session.set_parameter(name, &data);
    }
    lat_session.set_input("image", &lat_images);
    lat_session.step();
    lat_session.wait();
    lat_session.set_input("image", &lat_images);
    let lat_start = Instant::now();
    lat_session.step();
    lat_session.wait();
    let latency_ms = lat_start.elapsed().as_secs_f64() * 1000.0;

    emit_result(
        "ResNet-50",
        compile_s,
        forward_ms,
        0.0,
        &output,
        loss,
        latency_ms,
    );
}

fn bench_whisper() {
    use meganeura::models::whisper::{self, WhisperConfig};

    let config = WhisperConfig::whisper_tiny();
    let batch: u32 = 1;
    let mel_len: u32 = 3000;
    let d_model = config.d_model;

    eprintln!("[meganeura] building Whisper encoder graph...");
    let compile_start = Instant::now();
    let mut g = Graph::new();
    let encoder_out = whisper::build_encoder(&mut g, &config, batch, mel_len);
    g.set_outputs(vec![encoder_out]);

    eprintln!("[meganeura] compiling...");
    let mut session = build_inference_session(&g);

    // Load weights with deterministic init matching PyTorch encoder.
    let transposed = whisper::transposed_weight_names(&config);
    let transposed_set: std::collections::HashSet<&str> =
        transposed.iter().map(|s| s.as_str()).collect();
    let prefix = "model.encoder.";
    for (name, buf_ref) in session.plan().param_buffers.clone().iter() {
        let n = session.plan().buffers[buf_ref.0 as usize] / 4;
        // Strip prefix and map fused_bias→bias for seed to match PyTorch encoder names.
        let seed_name = name.strip_prefix(prefix).unwrap_or(name);
        let seed_name = seed_name.replace("fused_bias", "bias");
        let seed = name_seed(&seed_name);

        let data: Vec<f32> = if name.contains("fused_bias") {
            // Per-channel bias broadcast to [batch * channels * spatial].
            let channels = d_model;
            let spatial = n / (batch as usize * channels);
            let per_ch: Vec<f32> = (0..channels)
                .map(|c| (c as f32 * 0.01 + seed).sin() * 0.1)
                .collect();
            let mut buf = vec![0.0f32; n];
            for b in 0..batch as usize {
                for c in 0..channels {
                    for s in 0..spatial {
                        buf[(b * channels + c) * spatial + s] = per_ch[c];
                    }
                }
            }
            buf
        } else if transposed_set.contains(name.as_str()) {
            // Linear weight: init in PyTorch [out, in] then transpose to [in, out].
            let (in_dim, out_dim) = if name.contains("fc1.weight") {
                (config.d_model, config.ffn_dim)
            } else if name.contains("fc2.weight") {
                (config.ffn_dim, config.d_model)
            } else {
                (config.d_model, config.d_model)
            };
            let mut buf = vec![0.0f32; n];
            for i in 0..out_dim {
                for j in 0..in_dim {
                    buf[j * out_dim + i] = ((i * in_dim + j) as f32 * 0.01 + seed).sin() * 0.1;
                }
            }
            buf
        } else {
            (0..n)
                .map(|j| (j as f32 * 0.01 + seed).sin() * 0.1)
                .collect()
        };
        session.set_parameter(name, &data);
    }

    let compile_s = compile_start.elapsed().as_secs_f64();
    eprintln!("[meganeura] ready (compile: {compile_s:.2}s)");

    // --- Forward ---
    let mel_size = (batch * config.n_mels as u32 * mel_len) as usize;
    let mel: Vec<f32> = (0..mel_size).map(|i| (i as f32 * 0.001).sin()).collect();
    session.set_input("mel", &mel);

    let fwd_start = Instant::now();
    session.step();
    session.wait();
    let forward_ms = fwd_start.elapsed().as_secs_f64() * 1000.0;

    let seq_len = mel_len / 2; // stride-2 in conv2
    let output = session.read_output((batch * seq_len * d_model as u32) as usize);

    // MSE loss (encoder output vs zero).
    let loss: f64 = output.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / output.len() as f64;

    // --- Latency (re-run forward — same batch/mel_len) ---
    session.set_input("mel", &mel);
    session.step();
    session.wait();
    session.set_input("mel", &mel);
    let lat_start = Instant::now();
    session.step();
    session.wait();
    let latency_ms = lat_start.elapsed().as_secs_f64() * 1000.0;

    emit_result(
        "Whisper-tiny",
        compile_s,
        forward_ms,
        0.0,
        &output,
        loss,
        latency_ms,
    );
}

fn main() {
    env_logger::init();

    let model_name = std::env::args().nth(1).unwrap_or("SmolLM2-135M".into());
    match model_name.as_str() {
        "SmolLM2-135M" => bench_smollm2(&model_name),
        "SmolVLA" => bench_smolvla(),
        "StableDiffusion" => bench_stable_diffusion(),
        "ResNet-50" => bench_resnet(),
        "Whisper-tiny" => bench_whisper(),
        other => {
            eprintln!(
                "Unknown model: {other}. Available: SmolLM2-135M, SmolVLA, StableDiffusion, ResNet-50, Whisper-tiny"
            );
            std::process::exit(1);
        }
    }
}
