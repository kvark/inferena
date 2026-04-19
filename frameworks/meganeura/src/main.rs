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

/// Initialize all session parameters with deterministic name-seeded values.
///
/// Uses `sin(j * 0.01 + name_seed(name)) * 0.02` — the 0.02 scale matches
/// standard transformer init (GPT-2/LLaMA convention) and produces
/// realistic activation magnitudes through deep networks.
fn init_params(session: &mut meganeura::Session) {
    for (name, buf_ref) in session.plan().param_buffers.clone() {
        let n = session.plan().buffers[buf_ref.0 as usize] / 4;
        let seed = name_seed(&name);
        let data: Vec<f32> = (0..n)
            .map(|j| (j as f32 * 0.01 + seed).sin() * 0.02)
            .collect();
        session.set_parameter(&name, &data);
    }
}

fn load_weights(
    session: &mut meganeura::Session,
    model: &SafeTensorsModel,
    transposed_set: &std::collections::HashSet<&str>,
) {
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
}

fn compute_grad_norm(session: &meganeura::Session) -> f64 {
    let plan = session.plan();
    let num_buffers = plan.buffers.len();
    let mut norm_sq = 0.0f64;
    let mut total_params = 0usize;
    let mut param_norms: Vec<(String, f64, usize)> = Vec::new();
    for (name, buf_ref) in plan.param_buffers.iter() {
        let grad_pair = plan.param_grad_pairs.iter().find(|&&(p, _)| p == *buf_ref);
        let grad_buf = match grad_pair {
            Some(&(_, g)) => g,
            None => continue,
        };
        if buf_ref.0 as usize >= num_buffers || grad_buf.0 as usize >= num_buffers {
            continue;
        }
        let n = plan.buffers[buf_ref.0 as usize] / 4;
        let grad_size = plan.buffers[grad_buf.0 as usize] / 4;
        if n != grad_size {
            continue;
        }
        let mut grad = vec![0.0f32; n];
        session.read_param_grad(name, &mut grad);
        let param_sq: f64 = grad.iter().map(|&v| (v as f64) * (v as f64)).sum();
        norm_sq += param_sq;
        total_params += 1;
        param_norms.push((name.clone(), param_sq.sqrt(), n));
    }
    let grad_norm = norm_sq.sqrt();
    eprintln!("[meganeura] grad_norm={grad_norm:.6} ({total_params} params with gradients)");
    param_norms.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    eprintln!("[meganeura] top gradient norms:");
    for (name, norm, n) in param_norms.iter().take(10) {
        eprintln!("  {name}: {norm:.6} ({n} params)");
    }
    grad_norm
}

/// Warm up a session (3 runs) then return the best of 5 timed runs in ms.
fn bench_session(
    session: &mut meganeura::Session,
    set_inputs: &dyn Fn(&mut meganeura::Session),
) -> f64 {
    for _ in 0..3 {
        set_inputs(session);
        session.step();
        session.wait();
    }
    let mut best = f64::MAX;
    for _ in 0..5 {
        set_inputs(session);
        let t0 = Instant::now();
        session.step();
        session.wait();
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        if ms < best {
            best = ms;
        }
    }
    best
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
    load_weights(&mut session, &model, &transposed_set);

    let compile_s = compile_start.elapsed().as_secs_f64();
    eprintln!("[meganeura] ready (compile: {compile_s:.2}s)");

    // --- Forward ---
    let input_ids: Vec<u32> = (0..seq_len as u32).map(|i| i % vocab as u32).collect();
    let labels: Vec<u32> = (0..seq_len as u32)
        .map(|i| (i + 1) % vocab as u32)
        .collect();

    // Warm-up + timed: 3 warmup runs + best of 5.
    let forward_ms = bench_session(&mut session, &|s| {
        s.set_input_u32("token_ids", &input_ids);
    });

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
    load_weights(&mut lat_session, &model, &transposed_set);
    // Warm-up + timed: 3 warmup runs + best of 5.
    let latency_ms = bench_session(&mut lat_session, &|s| {
        s.set_input_u32("token_ids", &[0u32]);
    });

    // Drop inference sessions to free GPU memory before training.
    drop(session);
    drop(lat_session);

    // --- Training step (forward + backward) ---
    eprintln!("[meganeura] building training graph...");
    let training_g = smollm2::build_training_graph(&config, seq_len);
    eprintln!("[meganeura] compiling training session...");
    let mut train_session = build_session(&training_g);
    load_weights(&mut train_session, &model, &transposed_set);

    // HF-shifted labels: position p predicts labels[p+1] (the next token).
    // Last position has no target (zero label row).
    // Scale by seq_len/(seq_len-1) to compensate for meganeura dividing by seq_len
    // while HF divides by seq_len-1.
    let label_scale = seq_len as f32 / (seq_len - 1) as f32;
    let mut one_hot_labels = vec![0.0f32; seq_len * vocab];
    for pos in 0..seq_len - 1 {
        let target = labels[pos + 1] as usize;
        one_hot_labels[pos * vocab + target] = label_scale;
    }

    // Warm-up training: 3 runs + best of 5.
    let train_ms = bench_session(&mut train_session, &|s| {
        s.set_input_u32("token_ids", &input_ids);
        s.set_input("labels", &one_hot_labels);
    });
    let backward_ms = (train_ms - forward_ms).max(0.0);

    let grad_norm = compute_grad_norm(&train_session);
    if !grad_norm.is_finite() || grad_norm > 1e6 {
        eprintln!(
            "[meganeura] WARNING: grad_norm={grad_norm:.1} is suspiciously large — \
             possible GPU driver issue (see https://github.com/kvark/meganeura/issues/TBD)"
        );
    }

    emit_result(
        model_name,
        compile_s,
        forward_ms,
        backward_ms,
        &all_logits,
        loss,
        latency_ms,
        grad_norm,
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

    let compile_s = compile_start.elapsed().as_secs_f64();
    eprintln!("[meganeura] inference ready (compile: {compile_s:.2}s)");

    // --- Initialize with deterministic random values ---
    eprintln!("[meganeura] initializing parameters...");
    init_params(&mut infer_session);

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
    // Warm-up: 3 runs to stabilize pipeline caches and GPU clocks.
    for _ in 0..3 {
        set_inputs(&mut infer_session);
        infer_session.step();
        infer_session.wait();
    }

    // Timed: best of 5 runs.
    let mut best_ms = f64::MAX;
    for _ in 0..5 {
        set_inputs(&mut infer_session);
        let t0 = Instant::now();
        infer_session.step();
        infer_session.wait();
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        if ms < best_ms {
            best_ms = ms;
        }
    }
    let forward_ms = best_ms;

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
    // Drop inference session to free GPU memory before training.
    drop(infer_session);

    eprintln!("[meganeura] building SmolVLA training graph...");
    let training_g = smolvla::build_action_expert_training(&config, action_seq_len, vlm_seq_len);
    eprintln!("[meganeura] compiling training session...");
    let mut train_session = build_session(&training_g);

    init_params(&mut train_session);

    let target_actions = vec![0.0f32; action_seq_len * action_dim];

    // Warm-up training steps.
    for _ in 0..3 {
        set_inputs(&mut train_session);
        train_session.set_input("target_actions", &target_actions);
        train_session.set_learning_rate(0.0);
        train_session.step();
        train_session.wait();
    }

    // Timed: best of 5 runs.
    let mut best_train = f64::MAX;
    for _ in 0..5 {
        set_inputs(&mut train_session);
        train_session.set_input("target_actions", &target_actions);
        train_session.set_learning_rate(0.0);
        let t0 = Instant::now();
        train_session.step();
        train_session.wait();
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        if ms < best_train {
            best_train = ms;
        }
    }
    let train_ms = best_train;
    // Approximate backward as train_step - forward.
    let backward_ms = (train_ms - forward_ms).max(0.0);

    // --- Latency (single action chunk) ---
    eprintln!("[meganeura] measuring single-chunk latency...");
    let mut lat_g = Graph::new();
    let lat_pred = smolvla::build_action_expert(&mut lat_g, &config, 1, vlm_seq_len);
    lat_g.set_outputs(vec![lat_pred]);
    let mut lat_session = build_inference_session(&lat_g);
    init_params(&mut lat_session);
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
    for _ in 0..3 {
        lat_session.set_input("noisy_actions", &lat_actions);
        lat_session.set_input("timestep", lat_timestep);
        for i in 0..config.expert.num_layers {
            if i % config.expert.self_attn_every_n_layers != 0 {
                lat_session.set_input(&format!("vlm_kv_layer_{i}"), &vlm_kv);
            }
        }
        lat_session.step();
        lat_session.wait();
    }
    // Timed: best of 5.
    let mut best_lat = f64::MAX;
    for _ in 0..5 {
        lat_session.set_input("noisy_actions", &lat_actions);
        lat_session.set_input("timestep", lat_timestep);
        for i in 0..config.expert.num_layers {
            if i % config.expert.self_attn_every_n_layers != 0 {
                lat_session.set_input(&format!("vlm_kv_layer_{i}"), &vlm_kv);
            }
        }
        let t0 = Instant::now();
        lat_session.step();
        lat_session.wait();
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        if ms < best_lat {
            best_lat = ms;
        }
    }
    let latency_ms = best_lat;

    let grad_norm = compute_grad_norm(&train_session);

    emit_result(
        "SmolVLA",
        compile_s,
        forward_ms,
        backward_ms,
        &output,
        loss,
        latency_ms,
        grad_norm,
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
    grad_norm: f64,
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
            "grad_norm": if grad_norm.is_nan() { -1.0 } else { (grad_norm * 1_000_000.0).round() / 1_000_000.0 },
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

    eprintln!("[meganeura] building SD U-Net inference graph (small config)...");
    let compile_start = Instant::now();
    let mut infer_g = Graph::new();
    let pred = sd_unet::build_unet(&mut infer_g, &config);
    infer_g.set_outputs(vec![pred]);
    let mut infer_session = build_inference_session(&infer_g);

    let compile_s = compile_start.elapsed().as_secs_f64();
    eprintln!("[meganeura] ready (compile: {compile_s:.2}s)");

    // --- Initialize with deterministic values ---
    // Use name-seeded init so PyTorch's _name_seeded_init can match exactly
    // (parameter ordering between frameworks is unstable; name seeding isn't).
    eprintln!("[meganeura] initializing parameters...");
    let init_params = |session: &mut meganeura::Session| {
        for (name, buf_ref) in session.plan().param_buffers.clone().iter() {
            let n = session.plan().buffers[buf_ref.0 as usize] / 4;
            let seed = name_seed(name);
            let data: Vec<f32> = (0..n)
                .map(|j| (j as f32 * 0.01 + seed).sin() * 0.1)
                .collect();
            session.set_parameter(name, &data);
        }
    };
    init_params(&mut infer_session);

    // --- Prepare inputs ---
    let noisy_latent: Vec<f32> = (0..in_size).map(|i| (i as f32 * 0.01).sin()).collect();
    let noise_target: Vec<f32> = (0..in_size).map(|i| (i as f32 * 0.007).cos()).collect();

    // --- Forward (inference graph: returns noise prediction) ---
    // Warm-up + timed: 3 warmup runs + best of 5.
    let forward_ms = bench_session(&mut infer_session, &|s| {
        s.set_input("noisy_latent", &noisy_latent);
    });

    let output = infer_session.read_output(in_size);
    // MSE loss on CPU: mean((pred - target)^2) — matches PyTorch's F.mse_loss.
    let loss_val: f64 = output
        .iter()
        .zip(noise_target.iter())
        .map(|(&p, &t)| ((p - t) as f64).powi(2))
        .sum::<f64>()
        / output.len() as f64;
    eprintln!("[meganeura] forward: {forward_ms:.2}ms, loss={loss_val:.6}");

    // --- Latency (re-run forward, batch=2 is already minimal) ---
    // Already warmed up from forward benchmarking; 3 warmup + best of 5.
    let latency_ms = bench_session(&mut infer_session, &|s| {
        s.set_input("noisy_latent", &noisy_latent);
    });

    // Drop inference session to free GPU memory before training.
    drop(infer_session);

    // --- Training step (separate graph for backward timing) ---
    eprintln!("[meganeura] building SD U-Net training graph...");
    let mut train_g = Graph::new();
    let loss = sd_unet::build_training_graph(&mut train_g, &config);
    train_g.set_outputs(vec![loss]);
    let mut train_session = build_session(&train_g);
    init_params(&mut train_session);

    // Warm-up training: 3 runs + best of 5.
    let train_ms = bench_session(&mut train_session, &|s| {
        s.set_input("noisy_latent", &noisy_latent);
        s.set_input("noise_target", &noise_target);
    });
    let backward_ms = (train_ms - forward_ms).max(0.0);

    // GPU profiling for training breakdown.
    if std::env::var("MEGANEURA_PROFILE").is_ok() {
        train_session.set_profiling(true);
        train_session.set_input("noisy_latent", &noisy_latent);
        train_session.set_input("noise_target", &noise_target);
        train_session.step();
        train_session.wait();
        train_session.set_profiling(false);
        for _ in 0..2 {
            train_session.set_input("noisy_latent", &noisy_latent);
            train_session.set_input("noise_target", &noise_target);
            train_session.step();
            train_session.wait();
        }
        train_session.dump_gpu_timings();
    }

    let grad_norm = compute_grad_norm(&train_session);

    emit_result(
        "StableDiffusion",
        compile_s,
        forward_ms,
        backward_ms,
        &output,
        loss_val,
        latency_ms,
        grad_norm,
    );
}

fn bench_resnet() {
    use meganeura::models::resnet;

    let batch: u32 = 4;
    let scale: f32 = 0.01; // small scale to prevent explosion with identity BN

    eprintln!("[meganeura] building ResNet inference graph...");
    let compile_start = Instant::now();
    let mut infer_g = Graph::new();
    let logits_node = resnet::build_resnet50(&mut infer_g, batch);
    infer_g.set_outputs(vec![logits_node]);
    let mut infer_session = build_inference_session(&infer_g);

    let compile_s = compile_start.elapsed().as_secs_f64();
    eprintln!("[meganeura] ready (compile: {compile_s:.2}s)");

    // Helper to init parameters (shared between sessions).
    // Note: matches PyTorch's _resnet_init — fused_bias=0 (BN identity),
    // fc.weight transposed, others name-seeded sin*scale.
    let init_params = |session: &mut meganeura::Session| {
        for (name, buf_ref) in session.plan().param_buffers.clone().iter() {
            let n = session.plan().buffers[buf_ref.0 as usize] / 4;
            let data: Vec<f32> = if name.contains("fused_bias") {
                vec![0.0; n]
            } else if name == "fc.weight" {
                let in_dim = 2048usize;
                let out_dim = 1000usize;
                let seed = name_seed(name);
                let mut buf = vec![0.0f32; n];
                for i in 0..out_dim {
                    for j in 0..in_dim {
                        buf[j * out_dim + i] =
                            ((i * in_dim + j) as f32 * 0.01 + seed).sin() * scale;
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
    };
    init_params(&mut infer_session);

    // --- Inputs ---
    let in_size = (batch * 3 * 224 * 224) as usize;
    let images: Vec<f32> = (0..in_size).map(|i| (i as f32 * 0.001).sin()).collect();
    let labels_idx: Vec<usize> = (0..batch as usize).map(|i| i % 1000).collect();
    let mut one_hot_labels = vec![0.0f32; (batch * 1000) as usize];
    for (b, &l) in labels_idx.iter().enumerate() {
        one_hot_labels[b * 1000 + l] = 1.0;
    }

    // --- Forward (inference graph: returns logits) ---
    // Warm-up + timed: 3 warmup runs + best of 5.
    let forward_ms = bench_session(&mut infer_session, &|s| {
        s.set_input("image", &images);
    });

    let logits = infer_session.read_output((batch * 1000) as usize);
    // Cross-entropy on CPU with the same one-hot labels (matches PyTorch's
    // F.cross_entropy(logits, labels), which defaults to mean reduction).
    let mut total_loss = 0.0f64;
    for b in 0..batch as usize {
        let row = &logits[b * 1000..(b + 1) * 1000];
        let target = labels_idx[b];
        let max_l = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f64 = row.iter().map(|&l| ((l - max_l) as f64).exp()).sum();
        total_loss -= (row[target] - max_l) as f64 - sum_exp.ln();
    }
    let loss = total_loss / batch as f64;
    eprintln!("[meganeura] forward: {forward_ms:.2}ms, loss={loss:.6}");

    // --- Latency (single-image) ---
    let lat_images: Vec<f32> = vec![0.0; (3 * 224 * 224) as usize];
    let mut lat_g = Graph::new();
    let lat_logits = resnet::build_resnet50(&mut lat_g, 1);
    lat_g.set_outputs(vec![lat_logits]);
    let mut lat_session = build_inference_session(&lat_g);
    init_params(&mut lat_session);
    // Warm-up + timed: 3 warmup runs + best of 5.
    let latency_ms = bench_session(&mut lat_session, &|s| {
        s.set_input("image", &lat_images);
    });

    // Drop inference sessions to free GPU memory before training.
    drop(infer_session);
    drop(lat_session);

    // --- Training step (separate graph for backward timing) ---
    eprintln!("[meganeura] building ResNet training graph...");
    let training_g = resnet::build_resnet50_training(batch);
    let mut train_session = build_session(&training_g);
    init_params(&mut train_session);

    // Warm-up training: 3 runs + best of 5.
    let train_ms = bench_session(&mut train_session, &|s| {
        s.set_input("image", &images);
        s.set_input("labels", &one_hot_labels);
    });
    let backward_ms = (train_ms - forward_ms).max(0.0);

    if std::env::var("MEGANEURA_PROFILE").is_ok() {
        train_session.set_profiling(true);
        train_session.set_input("image", &images);
        train_session.set_input("labels", &one_hot_labels);
        train_session.step();
        train_session.wait();
        train_session.set_profiling(false);
        for _ in 0..2 {
            train_session.set_input("image", &images);
            train_session.set_input("labels", &one_hot_labels);
            train_session.step();
            train_session.wait();
        }
        train_session.dump_gpu_timings();
    }

    let grad_norm = compute_grad_norm(&train_session);

    emit_result(
        "ResNet-50",
        compile_s,
        forward_ms,
        backward_ms,
        &logits,
        loss,
        latency_ms,
        grad_norm,
    );
}

fn bench_whisper() {
    use meganeura::models::whisper::{self, WhisperConfig};

    let config = WhisperConfig::whisper_tiny();
    let batch: u32 = 1;
    let mel_len: u32 = 3000;
    let d_model = config.d_model;
    let seq_len = mel_len / 2; // stride-2 in conv2
    let num_classes: usize = 64;

    eprintln!("[meganeura] building Whisper encoder graph...");
    let compile_start = Instant::now();
    let mut infer_g = Graph::new();
    let encoder_out = whisper::build_encoder(&mut infer_g, &config, batch, mel_len);
    infer_g.set_outputs(vec![encoder_out]);

    eprintln!("[meganeura] compiling inference session...");
    let mut session = build_inference_session(&infer_g);

    // Load weights with deterministic init matching PyTorch encoder.
    let transposed = whisper::transposed_weight_names(&config);
    let transposed_set: std::collections::HashSet<&str> =
        transposed.iter().map(|s| s.as_str()).collect();
    let prefix = "model.encoder.";
    let init_params = |session: &mut meganeura::Session| {
        for (name, buf_ref) in session.plan().param_buffers.clone().iter() {
            let n = session.plan().buffers[buf_ref.0 as usize] / 4;
            let seed_name = name.strip_prefix(prefix).unwrap_or(name);
            let seed_name = seed_name.replace("fused_bias", "bias");
            let seed = name_seed(&seed_name);

            let data: Vec<f32> = if name.contains("fused_bias") {
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
    };
    init_params(&mut session);

    let compile_s = compile_start.elapsed().as_secs_f64();
    eprintln!("[meganeura] ready (compile: {compile_s:.2}s)");

    // --- Forward ---
    let mel_size = (batch * config.n_mels as u32 * mel_len) as usize;
    let mel: Vec<f32> = (0..mel_size).map(|i| (i as f32 * 0.001).sin()).collect();
    let mut one_hot_labels = vec![0.0f32; seq_len as usize * num_classes];
    for t in 0..seq_len as usize {
        one_hot_labels[t * num_classes + (t % num_classes)] = 1.0;
    }

    // Warm-up + timed: 3 warmup runs + best of 5.
    let forward_ms = bench_session(&mut session, &|s| {
        s.set_input("mel", &mel);
    });

    let output = session.read_output((batch * seq_len * d_model as u32) as usize);
    // MSE loss (encoder output vs zero) — matches PyTorch ground truth.
    let loss: f64 = output.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / output.len() as f64;
    eprintln!("[meganeura] forward: {forward_ms:.2}ms, loss={loss:.6}");

    // --- Latency ---
    // Already warmed up from forward benchmarking; 3 warmup + best of 5.
    let latency_ms = bench_session(&mut session, &|s| {
        s.set_input("mel", &mel);
    });

    // Drop inference session to free GPU memory before training.
    drop(session);

    // --- Training step ---
    // NOTE: on AMD/RADV the backward kernels (GELU at [576000]) can cause a GPU
    // context loss; the main harness forces the NVIDIA ICD via VK_ICD_FILENAMES.
    eprintln!("[meganeura] building + compiling Whisper training graph...");
    let train_g = whisper::build_training_graph(&config, batch, mel_len);
    let mut train_session = build_session(&train_g);
    init_params(&mut train_session);

    // Warm-up training: 3 runs + best of 5.
    let train_ms = bench_session(&mut train_session, &|s| {
        s.set_input("mel", &mel);
        s.set_input("labels", &one_hot_labels);
    });
    let backward_ms = (train_ms - forward_ms).max(0.0);

    // GPU profiling: capture per-dispatch timings to identify hotspots.
    if std::env::var("MEGANEURA_PROFILE").is_ok() {
        train_session.set_profiling(true);
        let set = |s: &mut meganeura::Session| {
            s.set_input("mel", &mel);
            s.set_input("labels", &one_hot_labels);
        };
        set(&mut train_session);
        train_session.step();
        train_session.wait();
        train_session.set_profiling(false);
        // Two more steps to rotate back to the profiled buffer's timestamps.
        set(&mut train_session);
        train_session.step();
        train_session.wait();
        set(&mut train_session);
        train_session.step();
        train_session.wait();
        train_session.dump_gpu_timings();
    }

    let grad_norm = compute_grad_norm(&train_session);

    emit_result(
        "Whisper-tiny",
        compile_s,
        forward_ms,
        backward_ms,
        &output,
        loss,
        latency_ms,
        grad_norm,
    );
}

fn main() {
    env_logger::init();

    let model_name = std::env::args().nth(1).unwrap_or("SmolLM2-135M".into());
    let all_models = [
        "SmolLM2-135M",
        "SmolVLA",
        "StableDiffusion",
        "ResNet-50",
        "Whisper-tiny",
    ];

    if !all_models.contains(&model_name.as_str()) {
        eprintln!(
            "Unknown model: {model_name}. Available: {}",
            all_models.join(", ")
        );
        std::process::exit(1);
    }

    if std::env::var("INFERENA_DRY_RUN").as_deref() == Ok("1") {
        eprintln!("[meganeura] dry-run OK: {model_name}");
        return;
    }

    // Pipeline-stats-driven auto-tune: spin up a temporary GPU context,
    // measure register counts for each flash kernel × candidate EPT and
    // for the fused ops the e-graph cost model recognizes, then install
    // the result in process globals. Subsequent Sessions and graph
    // optimizations pick it up automatically.
    //
    // Skipped for ResNet-50 (no flash attention) and skip-able
    // explicitly via INFERENA_MEGANEURA_SKIP_AUTOTUNE=1.
    if model_name != "ResNet-50"
        && std::env::var("INFERENA_MEGANEURA_SKIP_AUTOTUNE").as_deref() != Ok("1")
    {
        let gpu = meganeura::runtime::init_gpu_context()
            .expect("[meganeura] failed to init GPU for auto-tune");
        let auto_start = Instant::now();
        let result = meganeura::runtime::auto_tune(&gpu, 64);
        eprintln!(
            "[meganeura] auto-tune ({:.2}s): forward={} grad_q={} grad_kv={} grad_k={} grad_v={}",
            auto_start.elapsed().as_secs_f64(),
            result.flash_ept.forward_cap,
            result.flash_ept.grad_q_cap,
            result.flash_ept.grad_kv_cap,
            result.flash_ept.grad_k_cap,
            result.flash_ept.grad_v_cap,
        );
        meganeura::runtime::install_auto_tune(result);
    }

    match model_name.as_str() {
        "SmolLM2-135M" => bench_smollm2(&model_name),
        "SmolVLA" => bench_smolvla(),
        "StableDiffusion" => bench_stable_diffusion(),
        "ResNet-50" => bench_resnet(),
        "Whisper-tiny" => bench_whisper(),
        _ => unreachable!(),
    }
}
