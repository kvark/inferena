//! Candle framework benchmark runner for infermark.
//!
//! Uses candle-transformers' LLaMA implementation with random-init weights
//! on CPU. Candle supports CUDA and Metal but we default to CPU here.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama as llama_model;
use sha2::{Digest, Sha256};
use std::time::Instant;

fn smollm2_config() -> llama_model::Config {
    llama_model::Config {
        hidden_size: 576,
        intermediate_size: 1536,
        vocab_size: 49152,
        num_hidden_layers: 30,
        num_attention_heads: 9,
        num_key_value_heads: 3,
        use_flash_attn: false,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        bos_token_id: Some(1),
        eos_token_id: None,
        rope_scaling: None,
        max_position_embeddings: 2048,
        tie_word_embeddings: true,
    }
}

fn sha256_f32(data: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for &v in data {
        hasher.update(v.to_le_bytes());
    }
    format!("sha256:{}", hex::encode(hasher.finalize()))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_name = std::env::args().nth(1).unwrap_or("SmolLM2-135M".into());
    let config = match model_name.as_str() {
        "SmolLM2-135M" => smollm2_config(),
        other => {
            eprintln!("Unknown model: {other}. Available: SmolLM2-135M");
            std::process::exit(1);
        }
    };

    let device = Device::Cpu;
    let dtype = DType::F32;
    let seq_len: usize = 128;

    // --- Build & load model with random weights ---
    eprintln!("[candle] building model...");
    let compile_start = Instant::now();

    // Use random-init weights via VarBuilder.
    let vb = VarBuilder::zeros(dtype, &device);
    let model = llama_model::Llama::load(vb, &config)?;
    let mut cache = llama_model::Cache::new(false, dtype, &config, &device)?;

    let compile_s = compile_start.elapsed().as_secs_f64();
    eprintln!("[candle] built in {compile_s:.2}s");

    // --- Prepare deterministic input ---
    let input_ids: Vec<u32> = (0..seq_len as u32).collect();
    let input = Tensor::new(&input_ids[..], &device)?.unsqueeze(0)?;

    // --- Forward ---
    let fwd_start = Instant::now();
    let logits = model.forward(&input, 0, &mut cache)?;
    let forward_ms = fwd_start.elapsed().as_secs_f64() * 1000.0;

    // logits shape: [1, vocab_size] (last position only)
    let logits_data: Vec<f32> = logits.squeeze(0)?.to_vec1()?;
    eprintln!("[candle] forward: {forward_ms:.2}ms, {} logits", logits_data.len());

    // --- Loss: cross-entropy on last position ---
    let vocab = config.vocab_size;
    let target = seq_len % vocab;
    let max_logit = logits_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f64 = logits_data.iter().map(|&l| ((l - max_logit) as f64).exp()).sum();
    let loss = -((logits_data[target] - max_logit) as f64 - sum_exp.ln());

    // --- Backward ---
    // Candle has experimental autograd via Var, but it's not straightforward
    // with the pretrained model loading path. Report 0 for now.
    // TODO: Wire up candle-nn's backward pass.
    let backward_ms: f64 = 0.0;

    // --- Output ---
    let logits_hash = sha256_f32(&logits_data);
    let logits_sample: Vec<f64> = logits_data.iter().take(16).map(|&v| v as f64).collect();

    let result = serde_json::json!({
        "framework": "candle",
        "model": model_name,
        "device": format!("{:?}", device),
        "gpu_name": "cpu",
        "timings": {
            "compile_s": (compile_s * 100.0).round() / 100.0,
            "forward_ms": (forward_ms * 1000.0).round() / 1000.0,
            "backward_ms": backward_ms,
        },
        "outputs": {
            "logits_hash": logits_hash,
            "logits_sample": logits_sample,
            "loss": (loss * 1_000_000.0).round() / 1_000_000.0,
        },
    });

    println!("{}", serde_json::to_string(&result).unwrap());
    Ok(())
}
