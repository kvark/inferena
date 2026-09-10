//! Meganeura framework benchmark runner for inferena.
//!
//! Supports SmolLM2 (text LLM) and SmolVLA (action expert) models
//! using the meganeura crate (e-graph optimized NN on blade-graphics).

use meganeura::{CoopPolicy, Graph, Mode, Session, SessionConfig};
use sha2::{Digest, Sha256};
use std::time::Instant;

mod compilation;
mod graphics;
mod stream_weights;

thread_local! {
    static CAPTURE_GPU: std::cell::RefCell<Option<std::sync::Arc<blade_graphics::Context>>> =
        const { std::cell::RefCell::new(None) };
}

struct CaptureGpuOwner;

impl Drop for CaptureGpuOwner {
    fn drop(&mut self) {
        CAPTURE_GPU.with(|gpu| drop(gpu.borrow_mut().take()));
    }
}

fn shared_capture_gpu() -> bool {
    std::env::var("INFERENA_SHARED_CAPTURE_GPU").as_deref() == Ok("1")
}

fn build_inference_session(graph: &Graph) -> Session {
    build_session_for(graph, Mode::Inference)
}

fn build_session(graph: &Graph) -> Session {
    build_session_for(graph, Mode::Training)
}

fn build_session_for(graph: &Graph, mode: Mode) -> Session {
    let _span = tracing::info_span!("inferena_build_session", mode = ?mode).entered();
    let mut config = session_config();
    config.mode = mode;
    // The library's process-global default is never destroyed. Own the context
    // so dropping a session releases its device and flushes vendor trace data.
    config.gpu.get_or_insert_with(|| {
        let _span = tracing::info_span!("create_gpu_context").entered();
        std::sync::Arc::new(
            meganeura::runtime::init_gpu_context().expect("GPU initialization failed"),
        )
    });
    let report_dir = std::env::var_os("INFERENA_TUNE_REPORT");
    let report_tuning = report_dir.is_some();
    if report_tuning {
        assert!(config.tune, "tuning reports require MEGANEURA_TUNE=1");
        config.tune = false;
    }
    let mut session = meganeura::build(graph, config).0;
    if let Some(directory) = report_dir {
        let options = std::env::var_os("INFERENA_TUNE_OPTIONS").map_or_else(
            meganeura::TuneOptions::default,
            |path| {
                serde_json::from_reader(std::fs::File::open(path).unwrap())
                    .expect("invalid tuning options")
            },
        );
        let _span = tracing::info_span!("tune").entered();
        let report = session.tune_with(options).expect("tuning failed");
        static NEXT_REPORT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let index = NEXT_REPORT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::path::PathBuf::from(directory).join(format!("{index}-{mode:?}.json"));
        let file = std::fs::File::create_new(path).expect("tuning report must be a new file");
        serde_json::to_writer_pretty(file, &report).unwrap();
    }
    session
}

fn session_config() -> SessionConfig<'static> {
    let strict = std::env::var("INFERENA_STRICT").as_deref() == Ok("1");
    let mut config = if shared_capture_gpu() {
        assert!(
            std::env::var_os("INFERENA_NSYS").is_some(),
            "shared capture context is diagnostic only"
        );
        if let Some(directory) = meganeura::config::DUMP_WGSL.text() {
            meganeura::codegen::set_wgsl_dump_dir(directory);
        }
        let gpu = CAPTURE_GPU.with(|slot| {
            slot.borrow_mut()
                .get_or_insert_with(|| {
                    std::sync::Arc::new(
                        meganeura::init_gpu_context_with(meganeura::GpuOptions::from_env())
                            .expect("capture GPU initialization failed"),
                    )
                })
                .clone()
        });
        SessionConfig {
            gpu: Some(gpu),
            options: meganeura::CompileOptions::from_env(),
            optimize: meganeura::OptimizeConfig::from_env(),
            runtime: meganeura::SessionOptions::from_env(),
            tune: meganeura::config::TUNE.bool_or(false),
            ..Default::default()
        }
    } else {
        SessionConfig::from_env()
    };
    config.runtime.coop = if strict {
        CoopPolicy::Disabled
    } else {
        CoopPolicy::Auto
    };
    config.options.flash_forward_coop = !strict;
    config.options.flash_backward_coop = false;
    config
}

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

fn validation_sample(data: &[f32], count: usize) -> Vec<f64> {
    if data.len() <= count {
        return data.iter().map(|&value| value as f64).collect();
    }
    (0..count)
        .map(|index| {
            let source_index = index * (data.len() - 1) / (count - 1);
            data[source_index] as f64
        })
        .collect()
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
    let _span = tracing::info_span!("parameter_initialization").entered();
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
    model: &mut stream_weights::Weights,
    transposed_set: &std::collections::HashSet<&str>,
) {
    let _span = tracing::info_span!("parameter_preparation").entered();
    let _range = nsys_range("meganeura/parameter_preparation");
    for (name, _) in session.plan().param_buffers.clone() {
        // Skip derived (fused) params — auto-populated when source params are loaded.
        if !model.contains(&name) && name != "lm_head.weight" {
            continue;
        }
        let tied_head = name == "lm_head.weight" && !model.contains(&name);
        let source = if tied_head {
            "model.embed_tokens.weight"
        } else {
            &name
        };
        let data = {
            let _span = tracing::info_span!("tensor_preparation", name).entered();
            let _range = nsys_range("meganeura/tensor_preparation");
            model.tensor_f32(source, tied_head || transposed_set.contains(name.as_str()))
        };
        let _span = tracing::info_span!("parameter_upload", name).entered();
        let _range = nsys_range("meganeura/parameter_upload");
        session.set_parameter(&name, &data);
    }
}

fn compute_grad_norm(
    session: &meganeura::Session,
) -> (f64, std::collections::BTreeMap<String, f64>) {
    let plan = session.plan();
    let num_buffers = plan.buffers.len();
    let mut norm_sq = 0.0f64;
    let mut total_params = 0usize;
    let mut param_norms: Vec<(String, f64, usize)> = Vec::new();
    let mut gradient_norms = std::collections::BTreeMap::new();
    for (name, buf_ref) in plan.param_buffers.iter() {
        let grad_pair = plan.param_grad_pairs.iter().find(|&&(p, _)| p == *buf_ref);
        let grad_buf = match grad_pair {
            Some(&(_, g)) => g,
            None => continue,
        };
        if buf_ref.0 as usize >= num_buffers || grad_buf.0 as usize >= num_buffers {
            continue;
        }
        let parameter_size = plan.buffers[buf_ref.0 as usize] / 4;
        // Cooperative kernels pad gradient output buffers to whole tiles.
        // Read only the logical parameter extent. A one-element gradient for
        // a larger parameter is the autodiff placeholder for a dead/fused
        // logical parameter and must not appear in the canonical map.
        let grad_size = plan.buffers[grad_buf.0 as usize] / 4;
        if grad_size == 1 && parameter_size > 1 {
            continue;
        }
        assert!(
            grad_size >= parameter_size,
            "gradient buffer for {name} is smaller than its parameter"
        );
        let mut grad = vec![0.0f32; parameter_size];
        session.read_param_grad(name, &mut grad);
        let param_sq: f64 = grad.iter().map(|&v| (v as f64) * (v as f64)).sum();
        norm_sq += param_sq;
        total_params += 1;
        let param_norm = param_sq.sqrt();
        gradient_norms.insert(name.clone(), param_norm);
        param_norms.push((name.clone(), param_norm, parameter_size));
    }
    let grad_norm = norm_sq.sqrt();
    eprintln!("[meganeura] grad_norm={grad_norm:.6} ({total_params} params with gradients)");
    param_norms.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    eprintln!("[meganeura] top gradient norms:");
    for (name, norm, n) in param_norms.iter().take(10) {
        eprintln!("  {name}: {norm:.6} ({n} params)");
    }
    (grad_norm, gradient_norms)
}

#[derive(Clone, Debug)]
struct BenchStats {
    samples_ms: Vec<f64>,
    median_ms: f64,
    p25_ms: f64,
    p75_ms: f64,
}

fn benchmark_counts() -> (usize, usize) {
    let warmups = std::env::var("INFERENA_WARMUP_RUNS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(5);
    let samples = std::env::var("INFERENA_MEASUREMENT_RUNS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(20);
    assert!(samples > 0, "INFERENA_MEASUREMENT_RUNS must be positive");
    (warmups, samples)
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.len() == 1 {
        return sorted[0];
    }
    let position = (sorted.len() - 1) as f64 * q;
    let low = position.floor() as usize;
    let high = (low + 1).min(sorted.len() - 1);
    let fraction = position - low as f64;
    sorted[low] * (1.0 - fraction) + sorted[high] * fraction
}

impl BenchStats {
    fn from_samples(samples_ms: Vec<f64>) -> Self {
        let mut sorted = samples_ms.clone();
        sorted.sort_by(f64::total_cmp);
        Self {
            median_ms: quantile(&sorted, 0.5),
            p25_ms: quantile(&sorted, 0.25),
            p75_ms: quantile(&sorted, 0.75),
            samples_ms,
        }
    }
}

/// What Meganeura's per-phase `allocated_bytes` counts. Recorded with every
/// figure so it can never be tabulated against PyTorch's allocator
/// high-water mark as if the two measured the same thing.
const PLAN_MEMORY_BASIS: &str = "execution-plan physical allocation after lifetime-based aliasing";

/// GPU memory collected per phase.
///
/// Each phase must be recorded while its session is alive: the plan's
/// allocation is released when the session drops, and these runners drop
/// inference sessions before building the training graph.
#[derive(Default)]
struct MemoryCollector {
    phases: serde_json::Map<String, serde_json::Value>,
    /// Largest per-process device usage seen at any phase boundary.
    peak_process_bytes: Option<u64>,
    budget_bytes: Option<u64>,
}

impl MemoryCollector {
    fn record(&mut self, phase: &str, session: &meganeura::Session) {
        let summary = session.memory_summary();
        self.phases.insert(
            phase.to_string(),
            serde_json::json!({
                "allocated_bytes": summary.allocated_buffer_bytes,
                "basis": PLAN_MEMORY_BASIS,
                // Same plan without lifetime reuse — the aliasing saving is
                // the difference between this and `allocated_bytes`.
                "plan_logical_bytes": summary.total_buffer_bytes,
                "device_local_bytes": summary.device_local_bytes,
                "largest_buffer_bytes": summary.largest_buffer_bytes,
                "optimizer_state_bytes": summary.adam_state_bytes,
                "buffer_count": summary.num_buffers,
                "allocation_count": summary.num_allocations,
            }),
        );
        if let Some(stats) = session.device_memory_stats() {
            self.peak_process_bytes = Some(
                self.peak_process_bytes
                    .map_or(stats.usage_bytes, |peak| peak.max(stats.usage_bytes)),
            );
            self.budget_bytes = Some(stats.budget_bytes);
        }
    }

    fn to_json(&self) -> serde_json::Value {
        let device = self.peak_process_bytes.map(|process_bytes| {
            serde_json::json!({
                "process_bytes": process_bytes,
                "budget_bytes": self.budget_bytes,
                "process_source": if cfg!(target_vendor = "apple") {
                    "metal-current-allocated"
                } else {
                    "vulkan-memory-budget"
                },
                // Sampled between phases, not continuously during a step,
                // so this bounds residency from below.
                "sampled_at": "maximum over phase boundaries after warmup",
            })
        });
        serde_json::json!({ "device": device, "phases": self.phases })
    }
}

struct NsysRange;

impl Drop for NsysRange {
    fn drop(&mut self) {
        nvtx::range_pop!();
    }
}

fn nsys_range(name: &str) -> Option<NsysRange> {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    ENABLED
        .get_or_init(|| std::env::var_os("INFERENA_NSYS").is_some())
        .then(|| {
            nvtx::range_push!("{name}");
            NsysRange
        })
}

/// Warm up a session, retain every timed sample, and summarize by median.
fn bench_session(
    phase: &str,
    session: &mut meganeura::Session,
    set_inputs: &dyn Fn(&mut meganeura::Session),
) -> BenchStats {
    let _phase_span = tracing::info_span!("inferena_bench_phase", phase).entered();
    let (warmups, samples) = benchmark_counts();
    let warmup_range = nsys_range(&format!("meganeura/{phase}/warmup"));
    for _ in 0..warmups {
        set_inputs(session);
        session.step();
        session.wait();
    }
    drop(warmup_range);
    graphics::start_phase(phase);
    let _measure_range = nsys_range(&format!("meganeura/{phase}/measure"));
    let sample_label = format!("meganeura/{phase}/sample");
    let mut samples_ms = Vec::with_capacity(samples);
    for _ in 0..samples {
        let _sample_range = nsys_range(&sample_label);
        set_inputs(session);
        let t0 = Instant::now();
        let step_range = nsys_range("meganeura/step");
        session.step();
        drop(step_range);
        let wait_range = nsys_range("meganeura/wait");
        session.wait();
        drop(wait_range);
        samples_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    BenchStats::from_samples(samples_ms)
}

fn capture_gap_profile(
    model: &str,
    mode: &str,
    session: &mut meganeura::Session,
    benchmark: &BenchStats,
    set_inputs: &dyn Fn(&mut meganeura::Session),
) -> Option<String> {
    let profile_dir = std::env::var_os("INFERENA_PROFILE_DIR")?;
    let sample_count = std::env::var("INFERENA_PROFILE_SAMPLES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(3);
    assert!(
        sample_count > 0,
        "INFERENA_PROFILE_SAMPLES must be positive"
    );

    let profile_dir = std::path::PathBuf::from(profile_dir);
    std::fs::create_dir_all(&profile_dir).unwrap_or_else(|error| {
        panic!(
            "failed to create profile directory {}: {error}",
            profile_dir.display()
        )
    });
    let strict = std::env::var("INFERENA_STRICT").as_deref() == Ok("1");
    let precision = if strict {
        "strict-f32"
    } else {
        "accelerated-f32"
    };
    let slug: String = model
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    let path = profile_dir.join(format!("{slug}_{mode}_{precision}.json"));

    eprintln!("[meganeura] profiling {model} {mode}: {sample_count} retained GPU samples");
    let profile = meganeura::profiler::capture_session_profile(
        session,
        |session| set_inputs(session),
        meganeura::profiler::CaptureOptions {
            samples: sample_count,
            unprofiled_median_ms: Some(benchmark.median_ms),
            include_pipeline_statistics: true,
        },
    )
    .unwrap_or_else(|error| panic!("failed to profile {model} {mode}: {error}"));

    let artifact = serde_json::json!({
        "schema_version": 1,
        "artifact": "inferena-meganeura-gap-profile",
        "model": model,
        "mode": mode,
        "precision": precision,
        "framework_rev": std::env::var("FRAMEWORK_REV").unwrap_or_default(),
        "environment": {
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
        },
        "graph_optimizer": {
            "mode": std::env::var("MEGANEURA_OPTIMIZER").unwrap_or_else(|_| "greedy".to_string()),
            "extraction_cost": std::env::var("MEGANEURA_EGRAPH_COST").unwrap_or_else(|_| "tensor-traffic".to_string()),
        },
        "benchmark_protocol": "inferena-paper-v1",
        "normal_benchmark": {
            "samples_ms": &benchmark.samples_ms,
            "median_ms": benchmark.median_ms,
            "p25_ms": benchmark.p25_ms,
            "p75_ms": benchmark.p75_ms,
        },
        "profile": profile,
    });
    let file = std::fs::File::create(&path)
        .unwrap_or_else(|error| panic!("failed to create {}: {error}", path.display()));
    serde_json::to_writer_pretty(std::io::BufWriter::new(file), &artifact)
        .unwrap_or_else(|error| panic!("failed to write {}: {error}", path.display()));
    eprintln!("[meganeura] wrote {}", path.display());
    let directory_name = profile_dir
        .file_name()
        .and_then(std::ffi::OsStr::to_str)
        .unwrap_or("profiles");
    let file_name = path
        .file_name()
        .and_then(std::ffi::OsStr::to_str)
        .expect("profile filename is UTF-8");
    Some(format!("{directory_name}/{file_name}"))
}

fn bench_smollm2(model_name: &str) {
    use meganeura::models::smollm2::{self, SmolLM2Config};

    let mut profile_artifacts = std::collections::BTreeMap::new();
    let mut memory = MemoryCollector::default();
    let path = find_local_model(model_name)
        .expect("prepare pinned weights first: python scripts/prepare_models.py <model>");
    let source: serde_json::Value = serde_json::from_reader(
        std::fs::File::open(path.with_file_name("config.json")).expect("model config missing"),
    )
    .expect("invalid model config");
    assert_eq!(source["model_type"], "llama");
    assert_eq!(source["hidden_act"], "silu");
    assert_eq!(source["attention_bias"], false);
    assert!(
        source["rope_scaling"].is_null(),
        "scaled RoPE is unsupported"
    );
    assert!(source["mlp_bias"].is_null() || source["mlp_bias"] == false);
    let size = |name: &str| usize::try_from(source[name].as_u64().expect(name)).unwrap();
    let scalar = |name: &str| source[name].as_f64().expect(name) as f32;
    let config = SmolLM2Config {
        vocab_size: size("vocab_size"),
        hidden_size: size("hidden_size"),
        num_hidden_layers: size("num_hidden_layers"),
        num_attention_heads: size("num_attention_heads").try_into().unwrap(),
        num_key_value_heads: size("num_key_value_heads").try_into().unwrap(),
        intermediate_size: size("intermediate_size"),
        rms_norm_eps: scalar("rms_norm_eps"),
        rope_theta: scalar("rope_theta"),
        tie_word_embeddings: source["tie_word_embeddings"].as_bool().unwrap(),
    };

    let seq_len: usize = 128;
    let vocab = config.vocab_size;

    // --- Load weights ---
    eprintln!("[meganeura] loading from {}", path.display());
    let mut model = {
        let _span = tracing::info_span!("checkpoint_file_load").entered();
        stream_weights::Weights::open(
            path,
            std::env::var("INFERENA_STREAM_WEIGHTS").as_deref() == Ok("1"),
        )
    };

    // --- Build & compile ---
    eprintln!("[meganeura] building graph...");
    let compile_start = Instant::now();
    let mut g = Graph::new();
    let logits = smollm2::build_graph(&mut g, &config, seq_len);
    g.set_outputs(vec![logits]);

    eprintln!("[meganeura] compiling...");
    let mut session = build_inference_session(&g);
    let mut compile_s = compile_start.elapsed().as_secs_f64();

    // --- Load weights ---
    let transposed = smollm2::transposed_weight_names(&config);
    let transposed_set: std::collections::HashSet<&str> =
        transposed.iter().map(|s| s.as_str()).collect();
    load_weights(&mut session, &mut model, &transposed_set);

    eprintln!("[meganeura] ready (compile: {compile_s:.2}s)");

    // --- Forward ---
    let input_ids: Vec<u32> = (0..seq_len as u32).map(|i| i % vocab as u32).collect();
    let labels: Vec<u32> = (0..seq_len as u32)
        .map(|i| (i + 1) % vocab as u32)
        .collect();

    // Identical warmup/sample counts to the reference engine.
    let forward = bench_session("inference", &mut session, &|s| {
        s.set_input_u32("token_ids", &input_ids);
    });
    if let Some(path) = capture_gap_profile(model_name, "inference", &mut session, &forward, &|s| {
        s.set_input_u32("token_ids", &input_ids)
    }) {
        profile_artifacts.insert("inference", path);
    }

    let all_logits = session.read_output(seq_len * vocab);

    // Cross-entropy loss on CPU. `labels[pos]` is already the target token
    // for position `pos` (labels = (i+1) % vocab, i.e. pre-shifted relative
    // to input_ids), so this compares logits at `pos` directly against
    // `labels[pos]` — no additional shift. (Earlier code shifted an extra
    // position, assuming `labels` was input_ids-aligned like HF's raw
    // `labels` kwarg; that double-shift silently compared each position
    // against the token two steps ahead and inflated the loss.)
    let mut total_loss = 0.0f64;
    for pos in 0..seq_len {
        let sl = &all_logits[pos * vocab..(pos + 1) * vocab];
        let target = labels[pos] as usize;
        let max_l = sl.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f64 = sl.iter().map(|&l| ((l - max_l) as f64).exp()).sum();
        total_loss -= (sl[target] - max_l) as f64 - sum_exp.ln();
    }
    let loss = total_loss / seq_len as f64;

    let gpu_name = session.device_information().device_name.clone();
    let mut environment = environment_json(&session);
    memory.record("inference", &session);
    // The next shape gets its own plan, but never a second resident copy of weights.
    drop(session);

    // --- Latency (single-token forward) ---
    // Build a separate seq_len=1 inference graph.
    eprintln!("[meganeura] measuring single-token latency...");
    let lat_compile_start = Instant::now();
    let mut lat_g = Graph::new();
    let lat_logits = smollm2::build_graph(&mut lat_g, &config, 1);
    lat_g.set_outputs(vec![lat_logits]);
    let mut lat_session = build_inference_session(&lat_g);
    compile_s += lat_compile_start.elapsed().as_secs_f64();
    // Load the same checkpoint for the single-token shape.
    load_weights(&mut lat_session, &mut model, &transposed_set);
    let latency = bench_session("latency", &mut lat_session, &|s| {
        s.set_input_u32("token_ids", &[0u32]);
    });
    if let Some(path) =
        capture_gap_profile(model_name, "latency", &mut lat_session, &latency, &|s| {
            s.set_input_u32("token_ids", &[0u32])
        })
    {
        profile_artifacts.insert("latency", path);
    }

    // Record memory while the plans are still resident — dropping a session
    // releases the allocation this measures.
    memory.record("latency", &lat_session);

    let token_output = lat_session.read_output(vocab);
    assert!(token_output.iter().all(|value| value.is_finite()));
    let reference = &all_logits[..vocab];
    let squared_norm: f64 = reference.iter().map(|&value| (value as f64).powi(2)).sum();
    let squared_error: f64 = reference
        .iter()
        .zip(&token_output)
        .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
        .sum();
    let relative_l2 = if squared_norm > 0.0 {
        (squared_error / squared_norm).sqrt()
    } else if squared_error == 0.0 {
        0.0
    } else {
        f64::INFINITY
    };
    assert!(
        relative_l2 < 0.01,
        "stateless output disagrees with the causal prefill prefix: {relative_l2}"
    );
    environment["stateless_validation"] = serde_json::json!({
        "reference": "first causal prefill position, full vocabulary",
        "output_shape": [1, 1, vocab], "logits_hash": sha256_f32(&token_output),
        "prefill_prefix_relative_l2": relative_l2,
    });

    drop(lat_session);

    if std::env::var("INFERENA_INFERENCE_ONLY").as_deref() == Ok("1") {
        emit_result(
            model_name,
            compile_s,
            &forward,
            None,
            &all_logits,
            &[1, seq_len, vocab],
            loss,
            &latency,
            f64::NAN,
            &std::collections::BTreeMap::new(),
            &gpu_name,
            &profile_artifacts,
            &memory,
            &environment,
        );
        return;
    }

    // --- Training step (forward + backward) ---
    eprintln!("[meganeura] building training graph...");
    let train_compile_start = Instant::now();
    let training_g = smollm2::build_training_graph(&config, seq_len);
    eprintln!("[meganeura] compiling training session...");
    let mut train_session = build_session(&training_g);
    compile_s += train_compile_start.elapsed().as_secs_f64();
    load_weights(&mut train_session, &mut model, &transposed_set);

    // `labels[pos]` is already the target for position `pos` (see the
    // inference loss above) — every position has a target, so no scale
    // compensation for an excluded last position is needed anymore.
    let mut one_hot_labels = vec![0.0f32; seq_len * vocab];
    for pos in 0..seq_len {
        let target = labels[pos] as usize;
        one_hot_labels[pos * vocab + target] = 1.0;
    }

    let training = bench_session("training", &mut train_session, &|s| {
        s.set_input_u32("token_ids", &input_ids);
        s.set_input("labels", &one_hot_labels);
    });
    if let Some(path) = capture_gap_profile(
        model_name,
        "training",
        &mut train_session,
        &training,
        &|s| {
            s.set_input_u32("token_ids", &input_ids);
            s.set_input("labels", &one_hot_labels);
        },
    ) {
        profile_artifacts.insert("training", path);
    }

    memory.record("training", &train_session);
    let (grad_norm, gradient_norms) = compute_grad_norm(&train_session);
    if !grad_norm.is_finite() || grad_norm > 1e6 {
        eprintln!(
            "[meganeura] WARNING: grad_norm={grad_norm:.1} is suspiciously large — \
             possible GPU driver issue (see https://github.com/kvark/meganeura/issues/TBD)"
        );
    }
    let gpu_name = train_session.device_information().device_name.clone();
    let environment = environment_json(&train_session);

    emit_result(
        model_name,
        compile_s,
        &forward,
        Some(&training),
        &all_logits,
        &[1, seq_len, vocab],
        loss,
        &latency,
        grad_norm,
        &gradient_norms,
        &gpu_name,
        &profile_artifacts,
        &memory,
        &environment,
    );
}

fn bench_smolvla() {
    use meganeura::models::smolvla::{self, SmolVLAConfig};

    let mut profile_artifacts = std::collections::BTreeMap::new();
    let mut memory = MemoryCollector::default();
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

    let mut compile_s = compile_start.elapsed().as_secs_f64();
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
    let forward = bench_session("inference", &mut infer_session, &set_inputs);

    let output = infer_session.read_output(action_seq_len * action_dim);
    eprintln!(
        "[meganeura] forward: {:.2}ms, {} outputs",
        forward.median_ms,
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
    if let Some(path) = capture_gap_profile(
        "SmolVLA",
        "inference",
        &mut infer_session,
        &forward,
        &set_inputs,
    ) {
        profile_artifacts.insert("inference", path);
    }

    // --- Training step (forward + loss + backward; no optimizer update) ---
    memory.record("inference", &infer_session);
    // Drop inference session to free GPU memory before training.
    drop(infer_session);

    eprintln!("[meganeura] building SmolVLA training graph...");
    let train_compile_start = Instant::now();
    let training_g = smolvla::build_action_expert_training(&config, action_seq_len, vlm_seq_len);
    eprintln!("[meganeura] compiling training session...");
    let mut train_session = build_session(&training_g);
    compile_s += train_compile_start.elapsed().as_secs_f64();

    init_params(&mut train_session);

    let target_actions = vec![0.0f32; action_seq_len * action_dim];

    let training = bench_session("training", &mut train_session, &|session| {
        set_inputs(session);
        session.set_input("target_actions", &target_actions);
    });
    if let Some(path) = capture_gap_profile(
        "SmolVLA",
        "training",
        &mut train_session,
        &training,
        &|session| {
            set_inputs(session);
            session.set_input("target_actions", &target_actions);
        },
    ) {
        profile_artifacts.insert("training", path);
    }

    // --- Latency (single action chunk) ---
    eprintln!("[meganeura] measuring single-chunk latency...");
    let lat_compile_start = Instant::now();
    let mut lat_g = Graph::new();
    let lat_pred = smolvla::build_action_expert(&mut lat_g, &config, 1, vlm_seq_len);
    lat_g.set_outputs(vec![lat_pred]);
    let mut lat_session = build_inference_session(&lat_g);
    compile_s += lat_compile_start.elapsed().as_secs_f64();
    init_params(&mut lat_session);
    let lat_actions: Vec<f32> = (0..action_dim).map(|i| (i as f32 * 0.01).sin()).collect();
    let lat_timestep = &timestep;
    let latency = bench_session("latency", &mut lat_session, &|session| {
        session.set_input("noisy_actions", &lat_actions);
        session.set_input("timestep", lat_timestep);
        for i in 0..config.expert.num_layers {
            if i % config.expert.self_attn_every_n_layers != 0 {
                session.set_input(&format!("vlm_kv_layer_{i}"), &vlm_kv);
            }
        }
    });
    if let Some(path) = capture_gap_profile(
        "SmolVLA",
        "latency",
        &mut lat_session,
        &latency,
        &|session| {
            session.set_input("noisy_actions", &lat_actions);
            session.set_input("timestep", lat_timestep);
            for i in 0..config.expert.num_layers {
                if i % config.expert.self_attn_every_n_layers != 0 {
                    session.set_input(&format!("vlm_kv_layer_{i}"), &vlm_kv);
                }
            }
        },
    ) {
        profile_artifacts.insert("latency", path);
    }

    memory.record("training", &train_session);
    let (grad_norm, gradient_norms) = compute_grad_norm(&train_session);
    let gpu_name = train_session.device_information().device_name.clone();
    let environment = environment_json(&train_session);

    emit_result(
        "SmolVLA",
        compile_s,
        &forward,
        Some(&training),
        &output,
        &[1, action_seq_len, action_dim],
        loss,
        &latency,
        grad_norm,
        &gradient_norms,
        &gpu_name,
        &profile_artifacts,
        &memory,
        &environment,
    );
}

/// Kernel release string on unix, absent elsewhere.
fn kernel_release() -> Option<String> {
    if !cfg!(unix) {
        return None;
    }
    std::process::Command::new("uname")
        .arg("-r")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .filter(|release| !release.is_empty())
}

/// Environment metadata the paper protocol requires for a device row.
///
/// Driver identity is the primary confound across a multi-vendor matrix and
/// cannot be reconstructed once a run is archived, so it is recorded in the
/// result itself rather than left to the operator's notes.
fn environment_json(session: &meganeura::Session) -> serde_json::Value {
    let information = session.device_information();
    // Match the enumerated device against the one this context selected, so
    // the row carries a device identifier as well as a marketing name.
    let device_id = session
        .context()
        .enumerate_devices()
        .into_iter()
        .find(|report| report.information.device_name == information.device_name)
        .map(|report| report.device_id);
    serde_json::json!({
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "kernel": kernel_release(),
        "gpu_device_name": information.device_name,
        "gpu_driver_name": information.driver_name,
        "gpu_driver_info": information.driver_info,
        "gpu_software_emulated": information.is_software_emulated,
        "gpu_device_id": device_id,
        "gpu_device_local_budget_bytes": session
            .device_memory_stats()
            .map(|stats| stats.budget_bytes),
    })
}

fn detect_backend() -> &'static str {
    if cfg!(target_os = "macos") {
        "Metal"
    } else {
        "Vulkan"
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_result(
    model: &str,
    compile_s: f64,
    forward: &BenchStats,
    training: Option<&BenchStats>,
    output: &[f32],
    output_shape: &[usize],
    loss: f64,
    latency: &BenchStats,
    grad_norm: f64,
    gradient_norms: &std::collections::BTreeMap<String, f64>,
    gpu_name: &str,
    profile_artifacts: &std::collections::BTreeMap<&'static str, String>,
    memory: &MemoryCollector,
    environment: &serde_json::Value,
) {
    assert_eq!(output_shape.iter().product::<usize>(), output.len());
    let hash = sha256_f32(output);
    let sample = validation_sample(output, 256);
    let backend = detect_backend();
    let (warmup_runs, measurement_runs) = benchmark_counts();
    let strict = std::env::var("INFERENA_STRICT").as_deref() == Ok("1");
    let accelerated = !strict;
    let precision = if accelerated {
        serde_json::json!({
            "comparison_class": "reduced-input-f32-accumulate",
            "cooperative_matrix_policy": "Auto: protect full-precision derivative regions",
            "tensor_storage": "f32",
            "matmul_inputs": "forward: f16 for eligible cooperative-matrix kernels; f32 otherwise",
            "attention_inputs": "forward: f16 for eligible cooperative-matrix kernels; f32 otherwise",
            "convolution_inputs": "forward: f16 for eligible cooperative-matrix kernels; f32 otherwise",
            "backward_matmul_inputs": "f32",
            "backward_convolution_inputs": "f32",
            "backward_attention_inputs": "f32",
            "accumulation": "f32",
            "output": "f32",
            "reduced_precision_allowed": true,
            "f16_cooperative_matrix_permitted": true,
            "persistent_f16_tensors": false,
        })
    } else {
        serde_json::json!({
            "comparison_class": "strict-f32",
            "cooperative_matrix_policy": "Disabled: includes native-f32 cooperative tiles",
            "tensor_storage": "f32",
            "matmul_inputs": "f32",
            "attention_inputs": "f32",
            "convolution_inputs": "f32",
            "accumulation": "f32",
            "output": "f32",
            "reduced_precision_allowed": false,
            "f16_cooperative_matrix_permitted": false,
            "persistent_f16_tensors": false,
        })
    };

    let rev = std::env::var("FRAMEWORK_REV").unwrap_or_default();
    let workload_metrics = model.starts_with("SmolLM").then(|| {
        serde_json::json!({
            "prefill_ms": (forward.median_ms * 1000.0).round() / 1000.0,
            "prefill_tokens": 128,
            "stateless_one_token_ms": (latency.median_ms * 1000.0).round() / 1000.0,
            "has_kv_cache": false,
            "decode_ms": serde_json::Value::Null,
        })
    });

    let result = serde_json::json!({
        "framework": "meganeura",
        "framework_rev": rev,
        "model": model,
        "device": backend,
        "gpu_name": gpu_name,
        "backend": backend,
        "environment": environment,
        "protocol": {
            "name": "inferena-paper-v1",
            "warmup_runs": warmup_runs,
            "measurement_runs": measurement_runs,
            "statistic": "median",
            "training_requested": training.is_some(),
            "training_scope": training.map(|_| "forward + loss + backward; no optimizer update"),
            "compile_scope": "graph construction, optimization, and GPU pipeline creation for requested sessions",
            "diagnostic": std::env::var_os("INFERENA_NSYS").is_some(),
            "context_lifetime": if shared_capture_gpu() {
                "diagnostic shared context; destroyed after all sessions"
            } else {
                "owned per session; destroyed after use"
            },
        },
        "precision": precision,
        "optimizer": {
            "mode": std::env::var("MEGANEURA_OPTIMIZER").unwrap_or_else(|_| "greedy".to_string()),
            "extraction_cost": std::env::var("MEGANEURA_EGRAPH_COST").unwrap_or_else(|_| "tensor-traffic".to_string()),
            "measured_kernel_search": meganeura::config::TUNE.bool_or(false),
        },
        "timings": {
            "compile_s": (compile_s * 1000.0).round() / 1000.0,
            "inference_ms": (forward.median_ms * 1000.0).round() / 1000.0,
            "latency_ms": (latency.median_ms * 1000.0).round() / 1000.0,
            "training_ms": training.map(|stats| (stats.median_ms * 1000.0).round() / 1000.0),
        },
        "timing_samples_ms": {
            "inference": forward.samples_ms,
            "latency": latency.samples_ms,
            "training": training.map(|stats| &stats.samples_ms),
        },
        "timing_summary_ms": {
            "inference": {
                "median": forward.median_ms,
                "p25": forward.p25_ms,
                "p75": forward.p75_ms,
                "min": forward.samples_ms.iter().copied().fold(f64::INFINITY, f64::min),
                "max": forward.samples_ms.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            },
            "latency": {
                "median": latency.median_ms,
                "p25": latency.p25_ms,
                "p75": latency.p75_ms,
                "min": latency.samples_ms.iter().copied().fold(f64::INFINITY, f64::min),
                "max": latency.samples_ms.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            },
            "training": training.map(|training| serde_json::json!({
                "median": training.median_ms,
                "p25": training.p25_ms,
                "p75": training.p75_ms,
                "min": training.samples_ms.iter().copied().fold(f64::INFINITY, f64::min),
                "max": training.samples_ms.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            })),
        },
        "workload_metrics": workload_metrics,
        "profile_artifacts": profile_artifacts,
        "memory": memory.to_json(),
        "outputs": {
            "logits_hash": hash,
            "output_shape": output_shape,
            "logits_sample": sample,
            "loss": if loss.is_nan() { -1.0 } else { (loss * 1_000_000.0).round() / 1_000_000.0 },
            "grad_norm": training.map(|_| if grad_norm.is_nan() { -1.0 } else { (grad_norm * 1_000_000.0).round() / 1_000_000.0 }),
            "gradient_norms": gradient_norms,
        },
    });

    println!("{}", serde_json::to_string(&result).unwrap());
}

fn bench_stable_diffusion() {
    use meganeura::models::sd_unet::{self, SDUNetConfig};

    let mut profile_artifacts = std::collections::BTreeMap::new();
    let mut memory = MemoryCollector::default();
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

    let mut compile_s = compile_start.elapsed().as_secs_f64();
    eprintln!("[meganeura] ready (compile: {compile_s:.2}s)");

    // --- Initialize with deterministic values ---
    // Use name-seeded init so PyTorch can match exactly (parameter ordering
    // between frameworks is unstable; canonical names are not). Normalization
    // layers use their conventional identity initialization so signals and
    // gradients are not artificially attenuated through the deep U-Net.
    eprintln!("[meganeura] initializing parameters...");
    let init_params = |session: &mut meganeura::Session| {
        for (name, buf_ref) in session.plan().param_buffers.clone().iter() {
            let n = session.plan().buffers[buf_ref.0 as usize] / 4;
            let data = if name.contains(".norm") && name.ends_with(".weight") {
                vec![1.0; n]
            } else if name.contains(".norm") && name.ends_with(".bias") {
                vec![0.0; n]
            } else {
                let seed = name_seed(name);
                (0..n)
                    .map(|j| (j as f32 * 0.01 + seed).sin() * 0.02)
                    .collect()
            };
            session.set_parameter(name, &data);
        }
    };
    init_params(&mut infer_session);

    // --- Prepare inputs ---
    let noisy_latent: Vec<f32> = (0..in_size).map(|i| (i as f32 * 0.01).sin()).collect();
    let noise_target: Vec<f32> = (0..in_size).map(|i| (i as f32 * 0.007).cos()).collect();
    let timestep_embedding: Vec<f32> = (0..(batch * config.time_input_dim) as usize)
        .map(|i| (i as f32 * 0.005).sin())
        .collect();
    let text_context: Vec<f32> = (0..(config.context_len * config.context_dim) as usize)
        .map(|i| (i as f32 * 0.003).cos() * 0.1)
        .collect();

    // --- Forward (inference graph: returns noise prediction) ---
    let forward = bench_session("inference", &mut infer_session, &|s| {
        s.set_input("noisy_latent", &noisy_latent);
        s.set_input("timestep_embedding", &timestep_embedding);
        s.set_input("text_context", &text_context);
    });

    let output = infer_session.read_output(in_size);
    // MSE loss on CPU: mean((pred - target)^2) — matches PyTorch's F.mse_loss.
    let loss_val: f64 = output
        .iter()
        .zip(noise_target.iter())
        .map(|(&p, &t)| ((p - t) as f64).powi(2))
        .sum::<f64>()
        / output.len() as f64;
    eprintln!(
        "[meganeura] forward: {:.2}ms, loss={loss_val:.6}",
        forward.median_ms
    );
    if let Some(path) = capture_gap_profile(
        "StableDiffusion",
        "inference",
        &mut infer_session,
        &forward,
        &|s| {
            s.set_input("noisy_latent", &noisy_latent);
            s.set_input("timestep_embedding", &timestep_embedding);
            s.set_input("text_context", &text_context);
        },
    ) {
        profile_artifacts.insert("inference", path);
    }

    // --- Latency (batch=1, matching the PyTorch latency workload) ---
    let latency_compile_start = Instant::now();
    let mut latency_config = SDUNetConfig::small();
    latency_config.batch_size = 1;
    let mut latency_g = Graph::new();
    let latency_pred = sd_unet::build_unet(&mut latency_g, &latency_config);
    latency_g.set_outputs(vec![latency_pred]);
    let mut latency_session = build_inference_session(&latency_g);
    compile_s += latency_compile_start.elapsed().as_secs_f64();
    init_params(&mut latency_session);
    let latency_input_len = (in_c * res * res) as usize;
    let latency = bench_session("latency", &mut latency_session, &|s| {
        s.set_input("noisy_latent", &noisy_latent[..latency_input_len]);
        s.set_input(
            "timestep_embedding",
            &timestep_embedding[..latency_config.time_input_dim as usize],
        );
        s.set_input("text_context", &text_context);
    });
    if let Some(path) = capture_gap_profile(
        "StableDiffusion",
        "latency",
        &mut latency_session,
        &latency,
        &|s| {
            s.set_input("noisy_latent", &noisy_latent[..latency_input_len]);
            s.set_input(
                "timestep_embedding",
                &timestep_embedding[..latency_config.time_input_dim as usize],
            );
            s.set_input("text_context", &text_context);
        },
    ) {
        profile_artifacts.insert("latency", path);
    }

    memory.record("inference", &infer_session);
    memory.record("latency", &latency_session);

    // Drop inference session to free GPU memory before training.
    drop(infer_session);
    drop(latency_session);

    // --- Training step (forward + loss + backward) ---
    eprintln!("[meganeura] building SD U-Net training graph...");
    let train_compile_start = Instant::now();
    let mut train_g = Graph::new();
    let loss = sd_unet::build_training_graph(&mut train_g, &config);
    train_g.set_outputs(vec![loss]);
    let mut train_session = build_session(&train_g);
    compile_s += train_compile_start.elapsed().as_secs_f64();
    init_params(&mut train_session);

    let training = bench_session("training", &mut train_session, &|s| {
        s.set_input("noisy_latent", &noisy_latent);
        s.set_input("timestep_embedding", &timestep_embedding);
        s.set_input("text_context", &text_context);
        s.set_input("noise_target", &noise_target);
    });
    if let Some(path) = capture_gap_profile(
        "StableDiffusion",
        "training",
        &mut train_session,
        &training,
        &|s| {
            s.set_input("noisy_latent", &noisy_latent);
            s.set_input("timestep_embedding", &timestep_embedding);
            s.set_input("text_context", &text_context);
            s.set_input("noise_target", &noise_target);
        },
    ) {
        profile_artifacts.insert("training", path);
    }

    memory.record("training", &train_session);
    let (grad_norm, gradient_norms) = compute_grad_norm(&train_session);
    let gpu_name = train_session.device_information().device_name.clone();
    let environment = environment_json(&train_session);

    emit_result(
        "StableDiffusion",
        compile_s,
        &forward,
        Some(&training),
        &output,
        &[batch as usize, in_c as usize, res as usize, res as usize],
        loss_val,
        &latency,
        grad_norm,
        &gradient_norms,
        &gpu_name,
        &profile_artifacts,
        &memory,
        &environment,
    );
}

fn bench_resnet() {
    use meganeura::models::resnet;

    let mut profile_artifacts = std::collections::BTreeMap::new();
    let mut memory = MemoryCollector::default();
    let batch: u32 = 4;
    // Small enough to prevent explosion with synthetic identity BN while
    // preserving a non-trivial residual path.
    let scale: f32 = 0.01;

    eprintln!("[meganeura] building ResNet inference graph...");
    let compile_start = Instant::now();
    let mut infer_g = Graph::new();
    let logits_node = resnet::build_resnet50(&mut infer_g, batch);
    infer_g.set_outputs(vec![logits_node]);
    let mut infer_session = build_inference_session(&infer_g);

    let mut compile_s = compile_start.elapsed().as_secs_f64();
    eprintln!("[meganeura] ready (compile: {compile_s:.2}s)");

    // Helper to init parameters (shared between sessions).
    // Note: matches PyTorch's _resnet_init — fused_bias=0 (BN identity)
    // and every Meganeura-native parameter buffer is name-seeded in its
    // native layout. PyTorch transposes its FC storage when initializing.
    let init_params = |session: &mut meganeura::Session| {
        for (name, buf_ref) in session.plan().param_buffers.clone().iter() {
            let n = session.plan().buffers[buf_ref.0 as usize] / 4;
            let data: Vec<f32> = if name.contains("fused_bias") {
                vec![0.0; n]
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
    let forward = bench_session("inference", &mut infer_session, &|s| {
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
    eprintln!(
        "[meganeura] forward: {:.2}ms, loss={loss:.6}",
        forward.median_ms
    );
    if let Some(path) = capture_gap_profile(
        "ResNet-50",
        "inference",
        &mut infer_session,
        &forward,
        &|s| s.set_input("image", &images),
    ) {
        profile_artifacts.insert("inference", path);
    }

    // --- Latency (single-image) ---
    let lat_compile_start = Instant::now();
    let lat_images: Vec<f32> = vec![0.0; (3 * 224 * 224) as usize];
    let mut lat_g = Graph::new();
    let lat_logits = resnet::build_resnet50(&mut lat_g, 1);
    lat_g.set_outputs(vec![lat_logits]);
    let mut lat_session = build_inference_session(&lat_g);
    compile_s += lat_compile_start.elapsed().as_secs_f64();
    init_params(&mut lat_session);
    let latency = bench_session("latency", &mut lat_session, &|s| {
        s.set_input("image", &lat_images);
    });
    if let Some(path) =
        capture_gap_profile("ResNet-50", "latency", &mut lat_session, &latency, &|s| {
            s.set_input("image", &lat_images)
        })
    {
        profile_artifacts.insert("latency", path);
    }

    memory.record("inference", &infer_session);
    memory.record("latency", &lat_session);

    // Drop inference sessions to free GPU memory before training.
    drop(infer_session);
    drop(lat_session);

    // --- Training step (forward + loss + backward) ---
    eprintln!("[meganeura] building ResNet training graph...");
    let train_compile_start = Instant::now();
    let training_g = resnet::build_resnet50_training(batch);
    let mut train_session = build_session(&training_g);
    compile_s += train_compile_start.elapsed().as_secs_f64();
    init_params(&mut train_session);

    let training = bench_session("training", &mut train_session, &|s| {
        s.set_input("image", &images);
        s.set_input("labels", &one_hot_labels);
    });
    if let Some(path) = capture_gap_profile(
        "ResNet-50",
        "training",
        &mut train_session,
        &training,
        &|s| {
            s.set_input("image", &images);
            s.set_input("labels", &one_hot_labels);
        },
    ) {
        profile_artifacts.insert("training", path);
    }

    memory.record("training", &train_session);
    let (grad_norm, gradient_norms) = compute_grad_norm(&train_session);
    let gpu_name = train_session.device_information().device_name.clone();
    let environment = environment_json(&train_session);

    emit_result(
        "ResNet-50",
        compile_s,
        &forward,
        Some(&training),
        &logits,
        &[batch as usize, 1000],
        loss,
        &latency,
        grad_norm,
        &gradient_norms,
        &gpu_name,
        &profile_artifacts,
        &memory,
        &environment,
    );
}

fn bench_whisper() {
    use meganeura::models::whisper::{self, WhisperConfig};

    let mut profile_artifacts = std::collections::BTreeMap::new();
    let mut memory = MemoryCollector::default();
    let config = WhisperConfig::whisper_tiny();
    let batch: u32 = 1;
    let mel_len: u32 = 3000;
    let d_model = config.d_model;
    let seq_len = mel_len / 2; // stride-2 in conv2

    eprintln!("[meganeura] building Whisper encoder graph...");
    let compile_start = Instant::now();
    let mut infer_g = Graph::new();
    let encoder_out = whisper::build_encoder(&mut infer_g, &config, batch, mel_len);
    infer_g.set_outputs(vec![encoder_out]);

    eprintln!("[meganeura] compiling inference session...");
    let mut session = build_inference_session(&infer_g);

    // Load weights with deterministic init matching PyTorch encoder.
    let prefix = "model.encoder.";
    let init_params = |session: &mut meganeura::Session| {
        for (name, buf_ref) in session.plan().param_buffers.clone().iter() {
            let n = session.plan().buffers[buf_ref.0 as usize] / 4;
            let seed_name = name.strip_prefix(prefix).unwrap_or(name);
            let seed_name = seed_name.replace("fused_bias", "bias");
            let seed = name_seed(&seed_name);

            // All Meganeura buffers, including [in, out] linear weights
            // and per-channel convolution biases, use this native-layout
            // sequence. The PyTorch runner transposes only its [out, in]
            // linear storage.
            let data: Vec<f32> = (0..n)
                .map(|j| (j as f32 * 0.01 + seed).sin() * 0.02)
                .collect();
            session.set_parameter(name, &data);
        }
    };
    init_params(&mut session);

    let mut compile_s = compile_start.elapsed().as_secs_f64();
    eprintln!("[meganeura] ready (compile: {compile_s:.2}s)");

    // --- Forward ---
    let mel_size = (batch * config.n_mels as u32 * mel_len) as usize;
    let mel: Vec<f32> = (0..mel_size).map(|i| (i as f32 * 0.001).sin()).collect();

    let forward = bench_session("inference", &mut session, &|s| {
        s.set_input("mel", &mel);
    });

    let output = session.read_output((batch * seq_len * d_model as u32) as usize);
    // MSE loss (encoder output vs zero) — matches PyTorch ground truth.
    let loss: f64 = output.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / output.len() as f64;
    eprintln!(
        "[meganeura] forward: {:.2}ms, loss={loss:.6}",
        forward.median_ms
    );

    // --- Latency ---
    let latency = bench_session("latency", &mut session, &|s| {
        s.set_input("mel", &mel);
    });
    if let Some(path) =
        capture_gap_profile("Whisper-tiny", "inference", &mut session, &forward, &|s| {
            s.set_input("mel", &mel)
        })
    {
        // Whisper's current latency workload is the same full encoder graph.
        profile_artifacts.insert("inference", path.clone());
        profile_artifacts.insert("latency", path);
    }

    // Whisper's latency workload is the same full encoder graph on the same
    // session, so both phases report the same plan allocation.
    memory.record("inference", &session);
    memory.record("latency", &session);

    // Drop inference session to free GPU memory before training.
    drop(session);

    // --- Training step ---
    // NOTE: on AMD/RADV the backward kernels (GELU at [576000]) can cause a GPU
    // context loss; the main harness forces the NVIDIA ICD via VK_ICD_FILENAMES.
    eprintln!("[meganeura] building + compiling Whisper training graph...");
    let train_compile_start = Instant::now();
    let train_g = whisper::build_training_graph(&config, batch, mel_len);
    let mut train_session = build_session(&train_g);
    compile_s += train_compile_start.elapsed().as_secs_f64();
    init_params(&mut train_session);

    let training = bench_session("training", &mut train_session, &|s| {
        s.set_input("mel", &mel);
    });
    if let Some(path) = capture_gap_profile(
        "Whisper-tiny",
        "training",
        &mut train_session,
        &training,
        &|s| s.set_input("mel", &mel),
    ) {
        profile_artifacts.insert("training", path);
    }

    memory.record("training", &train_session);
    let (grad_norm, gradient_norms) = compute_grad_norm(&train_session);
    let gpu_name = train_session.device_information().device_name.clone();
    let environment = environment_json(&train_session);

    emit_result(
        "Whisper-tiny",
        compile_s,
        &forward,
        Some(&training),
        &output,
        &[batch as usize, seq_len as usize, d_model],
        loss,
        &latency,
        grad_norm,
        &gradient_norms,
        &gpu_name,
        &profile_artifacts,
        &memory,
        &environment,
    );
}

fn main() {
    env_logger::init();
    let _compile_trace = compilation::init();
    let _capture_gpu_owner = CaptureGpuOwner;

    let model_name = std::env::args().nth(1).unwrap_or("SmolLM2-135M".into());
    if model_name == "--list-devices" {
        let devices: Vec<_> = blade_graphics::Context::enumerate()
            .expect("native device enumeration failed")
            .into_iter()
            .map(|report| {
                serde_json::json!({
                    "device_id": report.device_id,
                    "name": report.information.device_name,
                    "driver_name": report.information.driver_name,
                    "driver_info": report.information.driver_info,
                    "software_emulated": report.information.is_software_emulated,
                    "available": matches!(report.status, blade_graphics::DeviceReportStatus::Available { .. }),
                    "status": format!("{:?}", report.status),
                })
            })
            .collect();
        println!("{}", serde_json::to_string(&devices).unwrap());
        return;
    }
    let all_models = [
        "SmolLM2-135M",
        "SmolLM2-360M",
        "SmolLM2-1.7B",
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

    assert!(
        std::env::var("INFERENA_INFERENCE_ONLY").as_deref() != Ok("1")
            || model_name.starts_with("SmolLM2-"),
        "inference-only currently supports SmolLM2 workloads"
    );

    match model_name.as_str() {
        "SmolLM2-135M" | "SmolLM2-360M" | "SmolLM2-1.7B" => bench_smollm2(&model_name),
        "SmolVLA" => bench_smolvla(),
        "StableDiffusion" => bench_stable_diffusion(),
        "ResNet-50" => bench_resnet(),
        "Whisper-tiny" => bench_whisper(),
        _ => unreachable!(),
    }
}
