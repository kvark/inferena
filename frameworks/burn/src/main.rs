//! Burn framework benchmark runner for inferena.
//!
//! Implements a minimal LLaMA-style transformer (matching SmolLM2 architecture)
//! and runs a fake training step (forward + backward), producing JSON output
//! compatible with the inferena harness.
//!
//! This is a scaffold — the model architecture is simplified and weights are
//! randomly initialized. The goal is to match the computational graph so that
//! timing comparisons are meaningful.

use burn::backend::Autodiff;
use burn::backend::wgpu::{
    Wgpu, WgpuDevice,
    graphics::{AutoGraphicsApi, GraphicsApi},
};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::softmax;
use sha2::{Digest, Sha256};
use std::time::Instant;

/// Simplified LLaMA-style transformer block.
#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    attn_q: Linear<B>,
    attn_k: Linear<B>,
    attn_v: Linear<B>,
    attn_out: Linear<B>,
    ffn_up: Linear<B>,
    ffn_down: Linear<B>,
}

impl<B: Backend> TransformerBlock<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_batch, _seq, dim] = x.dims();

        // Self-attention (single-head for simplicity).
        let q = self.attn_q.forward(x.clone());
        let k = self.attn_k.forward(x.clone());
        let v = self.attn_v.forward(x.clone());

        // scores = Q @ K^T / sqrt(dim)
        let scale = (dim as f64).sqrt();
        let scores = q.matmul(k.transpose()) / scale;
        let weights = softmax(scores, 2);
        let attn_out = weights.matmul(v);
        let attn_out = self.attn_out.forward(attn_out);

        // Residual connection.
        let x = x + attn_out;

        // Feed-forward network with SiLU activation.
        let ff = self.ffn_up.forward(x.clone());
        let ff = burn::tensor::activation::silu(ff);
        let ff = self.ffn_down.forward(ff);

        x + ff
    }
}

/// Minimal SmolLM2-style model.
#[derive(Module, Debug)]
pub struct SmolModel<B: Backend> {
    embedding: Embedding<B>,
    blocks: Vec<TransformerBlock<B>>,
    lm_head: Linear<B>,
}

/// Model hyperparameters matching SmolLM2-135M.
struct ModelConfig {
    vocab_size: usize,
    dim: usize,
    n_layers: usize,
    intermediate_size: usize,
}

impl ModelConfig {
    fn smol_135m() -> Self {
        ModelConfig {
            vocab_size: 49152,
            dim: 576,
            n_layers: 30,
            intermediate_size: 1536,
        }
    }

    fn smol_360m() -> Self {
        ModelConfig {
            vocab_size: 49152,
            dim: 960,
            n_layers: 32,
            intermediate_size: 2560,
        }
    }

    fn smol_1_7b() -> Self {
        ModelConfig {
            vocab_size: 49152,
            dim: 2048,
            n_layers: 24,
            intermediate_size: 8192,
        }
    }

    fn from_name(name: &str) -> Option<Self> {
        match name {
            "SmolLM2-135M" => Some(Self::smol_135m()),
            "SmolLM2-360M" => Some(Self::smol_360m()),
            "SmolLM2-1.7B" => Some(Self::smol_1_7b()),
            _ => None,
        }
    }
}

fn init_model<B: Backend>(cfg: &ModelConfig, device: &B::Device) -> SmolModel<B> {
    let embedding = EmbeddingConfig::new(cfg.vocab_size, cfg.dim).init(device);

    let mut blocks = Vec::new();
    for _ in 0..cfg.n_layers {
        let block = TransformerBlock {
            attn_q: LinearConfig::new(cfg.dim, cfg.dim).init(device),
            attn_k: LinearConfig::new(cfg.dim, cfg.dim).init(device),
            attn_v: LinearConfig::new(cfg.dim, cfg.dim).init(device),
            attn_out: LinearConfig::new(cfg.dim, cfg.dim).init(device),
            ffn_up: LinearConfig::new(cfg.dim, cfg.intermediate_size).init(device),
            ffn_down: LinearConfig::new(cfg.intermediate_size, cfg.dim).init(device),
        };
        blocks.push(block);
    }

    let lm_head = LinearConfig::new(cfg.dim, cfg.vocab_size).init(device);

    SmolModel {
        embedding,
        blocks,
        lm_head,
    }
}

fn forward<B: Backend>(model: &SmolModel<B>, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
    let mut x = model.embedding.forward(input_ids);
    for block in &model.blocks {
        x = block.forward(x);
    }
    model.lm_head.forward(x)
}

fn cross_entropy_loss<B: Backend>(
    logits: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
) -> Tensor<B, 1> {
    let [batch, seq, vocab] = logits.dims();
    let logits_2d = logits.reshape([batch * seq, vocab]);
    let targets_1d = targets.reshape([batch * seq]);

    // Manual cross-entropy: -log(softmax(logits))[target]
    let log_probs = burn::tensor::activation::log_softmax(logits_2d, 1);
    // Gather the log-prob at each target index.
    let target_log_probs: Tensor<B, 2> = log_probs.gather(1, targets_1d.unsqueeze_dim(1));
    // Mean negative log-likelihood.
    target_log_probs.neg().mean()
}

fn sha256_f32(data: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for &v in data {
        hasher.update(v.to_le_bytes());
    }
    format!("sha256:{}", hex::encode(hasher.finalize()))
}

fn main() {
    let model_name = std::env::args().nth(1).unwrap_or("SmolLM2-135M".into());
    let cfg = ModelConfig::from_name(&model_name).unwrap_or_else(|| {
        eprintln!(
            "Unknown model: {model_name}. Available: SmolLM2-135M, SmolLM2-360M, SmolLM2-1.7B"
        );
        std::process::exit(1);
    });

    type MyBackend = Autodiff<Wgpu>;
    let device = WgpuDevice::default();
    let seq_len = 128;

    // --- Compile / init ---
    let t0 = Instant::now();
    let model = init_model::<MyBackend>(&cfg, &device);
    let compile_s = t0.elapsed().as_secs_f64();

    // --- Prepare dummy input (deterministic) ---
    // Use fixed values for reproducibility.
    let input_data: Vec<i64> = (0..seq_len as i64)
        .map(|i| i % cfg.vocab_size as i64)
        .collect();
    let label_data: Vec<i64> = (0..seq_len as i64)
        .map(|i| (i + 1) % cfg.vocab_size as i64)
        .collect();
    let input_ids = Tensor::<MyBackend, 1, Int>::from_data(
        burn::tensor::TensorData::new(input_data, [seq_len]),
        &device,
    )
    .unsqueeze::<2>();
    let labels = Tensor::<MyBackend, 1, Int>::from_data(
        burn::tensor::TensorData::new(label_data, [seq_len]),
        &device,
    )
    .unsqueeze::<2>();

    // --- Forward ---
    let t0 = Instant::now();
    let logits = forward(&model, input_ids.clone());
    // Force computation to complete by reading a value.
    let logits_data: Vec<f32> = logits.to_data().to_vec().unwrap();
    let forward_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // --- Loss + Backward ---
    let logits2 = forward(&model, input_ids);
    let loss = cross_entropy_loss(logits2, labels);
    let t0 = Instant::now();
    let grads = loss.backward();
    // Force gradient computation.
    let _ = grads;
    let backward_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let loss_val: f64 = loss.inner().into_scalar().elem();

    // --- Collect outputs ---
    let logits_hash = sha256_f32(&logits_data);
    let logits_sample: Vec<f64> = logits_data.iter().take(16).map(|&v| v as f64).collect();

    let wgpu_backend = AutoGraphicsApi::backend();

    let result = serde_json::json!({
        "framework": "burn",
        "framework_rev": std::env::var("FRAMEWORK_REV").unwrap_or_default(),
        "model": model_name,
        "device": format!("{:?}", device),
        "gpu_name": format!("{:?}", device),
        "backend": format!("wgpu/{wgpu_backend}"),
        "timings": {
            "compile_s": (compile_s * 100.0).round() / 100.0,
            "inference_ms": (forward_ms * 1000.0).round() / 1000.0,
            "latency_ms": 0.0,
            "train_ms": (backward_ms * 1000.0).round() / 1000.0,
        },
        "outputs": {
            "logits_hash": logits_hash,
            "logits_sample": logits_sample,
            "loss": (loss_val * 1_000_000.0).round() / 1_000_000.0,
        },
    });

    println!("{}", serde_json::to_string(&result).unwrap());
}
