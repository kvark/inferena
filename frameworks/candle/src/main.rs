//! Candle framework benchmark runner for inferena.
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

fn find_model_path(model_name: &str) -> Option<std::path::PathBuf> {
    let exe = std::env::current_exe().unwrap_or_default();
    let mut root = exe
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .to_path_buf();
    for _ in 0..5 {
        let p = root
            .join("models")
            .join(model_name)
            .join("model.safetensors");
        if p.exists() {
            return Some(p);
        }
        if !root.pop() {
            break;
        }
    }
    let p = std::path::PathBuf::from("models")
        .join(model_name)
        .join("model.safetensors");
    if p.exists() {
        return Some(p);
    }
    None
}

fn emit_result(
    model_name: &str,
    compile_s: f64,
    forward_ms: f64,
    backward_ms: f64,
    logits_data: &[f32],
    loss: f64,
) {
    let logits_hash = sha256_f32(logits_data);
    let logits_sample: Vec<f64> = logits_data.iter().take(16).map(|&v| v as f64).collect();

    let result = serde_json::json!({
        "framework": "candle",
        "framework_rev": std::env::var("FRAMEWORK_REV").unwrap_or_default(),
        "model": model_name,
        "device": "Cpu",
        "gpu_name": "cpu",
        "timings": {
            "compile_s": (compile_s * 100.0).round() / 100.0,
            "inference_ms": (forward_ms * 1000.0).round() / 1000.0,
            "latency_ms": 0.0,
            "train_ms": backward_ms,
        },
        "outputs": {
            "logits_hash": logits_hash,
            "logits_sample": logits_sample,
            "loss": (loss * 1_000_000.0).round() / 1_000_000.0,
        },
    });

    println!("{}", serde_json::to_string(&result).unwrap());
}

fn bench_smollm2(model_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let config = smollm2_config();
    let device = Device::Cpu;
    let dtype = DType::F32;
    let seq_len: usize = 128;

    eprintln!("[candle] building model...");
    let compile_start = Instant::now();

    let vb = if let Some(ref path) = find_model_path(model_name) {
        eprintln!("[candle] loading weights from {}", path.display());
        unsafe { VarBuilder::from_mmaped_safetensors(std::slice::from_ref(path), dtype, &device)? }
    } else {
        eprintln!("[candle] no local weights, using zeros");
        VarBuilder::zeros(dtype, &device)
    };
    let model = llama_model::Llama::load(vb, &config)?;
    let mut cache = llama_model::Cache::new(false, dtype, &config, &device)?;

    let compile_s = compile_start.elapsed().as_secs_f64();
    eprintln!("[candle] built in {compile_s:.2}s");

    let input_ids: Vec<u32> = (0..seq_len as u32).collect();
    let input = Tensor::new(&input_ids[..], &device)?.unsqueeze(0)?;

    let fwd_start = Instant::now();
    let logits = model.forward(&input, 0, &mut cache)?;
    let forward_ms = fwd_start.elapsed().as_secs_f64() * 1000.0;

    let logits_data: Vec<f32> = logits.squeeze(0)?.to_vec1()?;
    eprintln!(
        "[candle] forward: {forward_ms:.2}ms, {} logits (last position)",
        logits_data.len()
    );

    let vocab = config.vocab_size;
    let target = seq_len % vocab;
    let max_logit = logits_data
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f64 = logits_data
        .iter()
        .map(|&l| ((l - max_logit) as f64).exp())
        .sum();
    let loss = -((logits_data[target] - max_logit) as f64 - sum_exp.ln());

    emit_result(model_name, compile_s, forward_ms, 0.0, &logits_data, loss);
    Ok(())
}

fn bench_stable_diffusion() -> Result<(), Box<dyn std::error::Error>> {
    use candle_transformers::models::stable_diffusion::unet_2d::{
        BlockConfig, UNet2DConditionModel, UNet2DConditionModelConfig,
    };

    let device = Device::Cpu;
    let dtype = DType::F32;

    // SD 1.5 UNet configuration (matches HF runwayml/stable-diffusion-v1-5).
    let unet_config = UNet2DConditionModelConfig {
        blocks: vec![
            BlockConfig {
                out_channels: 320,
                use_cross_attn: Some(1),
                attention_head_dim: 8,
            },
            BlockConfig {
                out_channels: 640,
                use_cross_attn: Some(1),
                attention_head_dim: 8,
            },
            BlockConfig {
                out_channels: 1280,
                use_cross_attn: Some(1),
                attention_head_dim: 8,
            },
            BlockConfig {
                out_channels: 1280,
                use_cross_attn: None,
                attention_head_dim: 8,
            },
        ],
        center_input_sample: false,
        cross_attention_dim: 768,
        downsample_padding: 1,
        flip_sin_to_cos: true,
        freq_shift: 0.,
        layers_per_block: 2,
        mid_block_scale_factor: 1.,
        norm_eps: 1e-5,
        norm_num_groups: 32,
        sliced_attention_size: None,
        use_linear_projection: false,
    };

    eprintln!("[candle] building SD 1.5 UNet (zeros init)...");
    let compile_start = Instant::now();

    // Build with zeros — no real weights needed for benchmarking.
    let vb = VarBuilder::zeros(dtype, &device);
    let unet = UNet2DConditionModel::new(vb, 4, 4, false, unet_config)?;

    let compile_s = compile_start.elapsed().as_secs_f64();
    eprintln!("[candle] built in {compile_s:.2}s");

    // Prepare deterministic inputs.
    // Latent: [1, 4, 64, 64] for 512×512 image.
    let latent_h = 64usize;
    let latent_w = 64usize;
    let in_channels = 4usize;
    let latent_size = in_channels * latent_h * latent_w;
    let latent_data: Vec<f32> = (0..latent_size).map(|i| (i as f32 * 0.01).sin()).collect();
    let latent =
        Tensor::new(&latent_data[..], &device)?.reshape((1, in_channels, latent_h, latent_w))?;

    // Text embedding: [1, 77, 768] (random deterministic).
    let text_seq = 77usize;
    let text_dim = 768usize;
    let text_size = text_seq * text_dim;
    let text_data: Vec<f32> = (0..text_size).map(|i| (i as f32 * 0.005).cos()).collect();
    let text_emb = Tensor::new(&text_data[..], &device)?.reshape((1, text_seq, text_dim))?;

    // Timestep (single denoising step at t=500).
    let timestep: f64 = 500.0;

    // --- Forward ---
    eprintln!("[candle] running forward pass...");
    let fwd_start = Instant::now();
    let output = unet.forward(&latent, timestep, &text_emb)?;
    let forward_ms = fwd_start.elapsed().as_secs_f64() * 1000.0;

    let output_flat: Vec<f32> = output.flatten_all()?.to_vec1()?;
    eprintln!(
        "[candle] forward: {forward_ms:.2}ms, {} outputs",
        output_flat.len()
    );

    // MSE loss against zeros (noise prediction target).
    let loss: f64 =
        output_flat.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / output_flat.len() as f64;

    emit_result(
        "StableDiffusion",
        compile_s,
        forward_ms,
        0.0,
        &output_flat,
        loss,
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_name = std::env::args().nth(1).unwrap_or("SmolLM2-135M".into());
    match model_name.as_str() {
        "SmolLM2-135M" => bench_smollm2(&model_name)?,
        "StableDiffusion" => bench_stable_diffusion()?,
        other => {
            eprintln!("Unknown model: {other}. Available: SmolLM2-135M, StableDiffusion");
            std::process::exit(1);
        }
    }
    Ok(())
}
