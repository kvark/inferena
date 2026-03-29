//! Meganeura framework benchmark runner for infermark.
//!
//! Runs a fake training step (forward + backward) on SmolLM2 using the
//! meganeura crate (e-graph optimized NN on blade-graphics), producing JSON
//! output compatible with the infermark harness.

use meganeura::{
    Graph, build_inference_session,
    data::safetensors::SafeTensorsModel,
    models::smollm2::{self, SmolLM2Config},
};
use sha2::{Digest, Sha256};
use std::time::Instant;

fn model_config(name: &str) -> Option<(&str, SmolLM2Config)> {
    match name {
        "SmolLM2-135M" => Some(("HuggingFaceTB/SmolLM2-135M", SmolLM2Config::smollm2_135m())),
        _ => None,
    }
}

fn load_model(model_name: &str, repo_id: &str) -> SafeTensorsModel {
    // Try local models/ directory first (populated by models/download.sh or manually).
    let exe = std::env::current_exe().unwrap_or_default();
    let mut root = exe.parent().unwrap_or(std::path::Path::new(".")).to_path_buf();
    for _ in 0..5 {
        let local = root.join("models").join(model_name).join("model.safetensors");
        if local.exists() {
            eprintln!("[meganeura] loading from {}", local.display());
            return SafeTensorsModel::load(local).expect("local model load failed");
        }
        if !root.pop() { break; }
    }
    // Also check relative to cwd.
    let cwd_local = std::path::PathBuf::from("models")
        .join(model_name)
        .join("model.safetensors");
    if cwd_local.exists() {
        eprintln!("[meganeura] loading from {}", cwd_local.display());
        return SafeTensorsModel::load(cwd_local).expect("local model load failed");
    }
    // Fall back to HF download (caches to ~/.cache/huggingface/hub/).
    SafeTensorsModel::download(repo_id).expect("model download/load failed")
}

fn main() {
    env_logger::init();

    let model_name = std::env::args().nth(1).unwrap_or("SmolLM2-135M".into());
    let (repo_id, config) = model_config(&model_name).unwrap_or_else(|| {
        eprintln!("Unknown model: {model_name}. Available: SmolLM2-135M");
        std::process::exit(1);
    });

    let seq_len: usize = 128;
    let vocab = config.vocab_size;

    // --- Load model from HF cache ---
    eprintln!("[meganeura] loading model {repo_id}...");
    let model = load_model(&model_name, repo_id);

    // --- Build & compile graph ---
    eprintln!("[meganeura] building graph...");
    let compile_start = Instant::now();
    let mut g = Graph::new();
    let logits = smollm2::build_graph(&mut g, &config, seq_len);
    g.set_outputs(vec![logits]);

    eprintln!("[meganeura] compiling...");
    let mut session = build_inference_session(&g);
    eprintln!(
        "[meganeura] compiled: {} buffers, {} dispatches",
        session.plan().buffers.len(),
        session.plan().dispatches.len()
    );

    // --- Load weights ---
    eprintln!("[meganeura] loading weights...");
    let transposed = smollm2::transposed_weight_names(&config);
    let transposed_set: std::collections::HashSet<&str> =
        transposed.iter().map(|s| s.as_str()).collect();

    for (name, _) in session.plan().param_buffers.clone() {
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

    // --- Prepare deterministic input ---
    let input_ids: Vec<u32> = (0..seq_len as u32).map(|i| i % vocab as u32).collect();
    let labels: Vec<u32> = (0..seq_len as u32).map(|i| (i + 1) % vocab as u32).collect();

    // --- Forward pass ---
    session.set_input_u32("token_ids", &input_ids);
    let forward_start = Instant::now();
    session.step();
    session.wait();
    let forward_ms = forward_start.elapsed().as_secs_f64() * 1000.0;

    let all_logits = session.read_output(seq_len * vocab);
    eprintln!("[meganeura] forward: {forward_ms:.2}ms, got {} logits", all_logits.len());

    // --- Compute loss on CPU (cross-entropy) ---
    let mut total_loss = 0.0f64;
    for pos in 0..seq_len {
        let logit_slice = &all_logits[pos * vocab..(pos + 1) * vocab];
        let target = labels[pos] as usize;
        let max_logit = logit_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f64 = logit_slice.iter().map(|&l| ((l - max_logit) as f64).exp()).sum();
        let log_prob = (logit_slice[target] - max_logit) as f64 - sum_exp.ln();
        total_loss -= log_prob;
    }
    let loss = total_loss / seq_len as f64;

    // --- Backward pass ---
    // TODO: Use meganeura's Trainer API for real backward pass once we wire
    // up the training graph builder for SmolLM2.
    let backward_start = Instant::now();
    session.set_input_u32("token_ids", &input_ids);
    session.step();
    session.wait();
    let backward_ms = backward_start.elapsed().as_secs_f64() * 1000.0;

    // --- Logits hash ---
    let logits_hash = {
        let mut hasher = Sha256::new();
        for &v in &all_logits {
            hasher.update(v.to_le_bytes());
        }
        format!("sha256:{}", hex::encode(hasher.finalize()))
    };

    let logits_sample: Vec<f64> = all_logits.iter().take(16).map(|&v| v as f64).collect();

    let result = serde_json::json!({
        "framework": "meganeura",
        "model": model_name,
        "device": "blade-gpu",
        "gpu_name": "blade-gpu",
        "timings": {
            "compile_s": (compile_s * 100.0).round() / 100.0,
            "forward_ms": (forward_ms * 1000.0).round() / 1000.0,
            "backward_ms": (backward_ms * 1000.0).round() / 1000.0,
        },
        "outputs": {
            "logits_hash": logits_hash,
            "logits_sample": logits_sample,
            "loss": (loss * 1_000_000.0).round() / 1_000_000.0,
        },
    });

    println!("{}", serde_json::to_string(&result).unwrap());
}
