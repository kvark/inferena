#![allow(clippy::print_literal)]

use clap::Parser;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Frozen workload/validation contract; this branch explicitly versions the
/// PyTorch execution experiment below without relabelling Meganeura's runner.
pub const PAPER_PROTOCOL: &str = "inferena-paper-v1";

/// Result produced by each framework benchmark runner.
/// Every runner must print exactly one JSON object matching this schema to stdout.
/// Extra framework-specific fields (e.g. torch_version) are preserved in `extra`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchResult {
    pub framework: String,
    pub model: String,
    pub device: String,
    pub gpu_name: String,
    pub timings: Timings,
    pub outputs: Outputs,
    /// GPU memory usage, when the runner can report it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory: Option<MemoryReport>,
    /// Framework-specific extra fields (torch_version, driver_name, etc.).
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Timings {
    /// Time to compile/optimize the model (seconds).
    pub compile_s: f64,
    /// Inference (full forward pass) time (milliseconds).
    pub inference_ms: f64,
    /// Single-token / minimal-input latency (milliseconds).
    #[serde(default)]
    pub latency_ms: f64,
    /// Training backward pass time (milliseconds).
    pub training_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Outputs {
    /// SHA-256 hash of the full logits tensor (flattened, f32 little-endian bytes).
    pub logits_hash: String,
    /// Canonical logical shape of the output tensor.
    #[serde(default)]
    pub output_shape: Vec<usize>,
    /// Deterministic sample of output values for numerical comparison.
    pub logits_sample: Vec<f64>,
    /// Scalar loss value from the fake training step.
    pub loss: f64,
    /// L2 norm of all parameter gradients after the measured training step.
    #[serde(default)]
    pub grad_norm: Option<f64>,
    /// Per-parameter gradient L2 norms keyed by canonical parameter name.
    #[serde(default)]
    pub gradient_norms: std::collections::BTreeMap<String, f64>,
}

/// GPU memory usage for one framework run.
///
/// The two engines account for memory differently and their per-phase
/// figures must not be placed in a single table column without their
/// `basis` label: Meganeura reports what its execution plan allocates,
/// while PyTorch reports a caching-allocator high-water mark.
///
/// `device.process_bytes` is the cross-engine comparable number. It is the
/// driver's own per-process figure, so it includes context, pipeline, and
/// driver overhead that neither engine's internal accounting sees, and it
/// is not disturbed by an unrelated workload sharing the GPU.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryReport {
    #[serde(default)]
    pub device: Option<DeviceMemory>,
    /// Per-phase engine accounting, keyed by `inference`/`latency`/`training`.
    #[serde(default)]
    pub phases: std::collections::BTreeMap<String, PhaseMemory>,
    /// Caveats a reader needs to interpret the figures, such as residency
    /// carried over from an earlier phase.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceMemory {
    /// Total physical device memory, in bytes, where the engine can report
    /// it. Vulkan reports a per-process budget instead; see `budget_bytes`.
    #[serde(default)]
    pub total_bytes: Option<u64>,
    /// Device-local memory this process may use, in bytes. Not the same
    /// quantity as `total_bytes` and not interchangeable with it.
    #[serde(default)]
    pub budget_bytes: Option<u64>,
    /// Device memory attributed to this process, in bytes.
    #[serde(default)]
    pub process_bytes: Option<u64>,
    /// How `process_bytes` was obtained, e.g. `vulkan-memory-budget`,
    /// `metal-current-allocated`, `nvml-per-process`. Absent when the
    /// platform could not report a per-process figure — which is recorded
    /// as missing rather than as zero.
    #[serde(default)]
    pub process_source: Option<String>,
    /// When the sample was taken, e.g. `after-training`.
    #[serde(default)]
    pub sampled_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseMemory {
    /// Bytes this engine accounts for in this phase.
    pub allocated_bytes: u64,
    /// What `allocated_bytes` counts. Required, so that no cell can be
    /// published without saying what it measured.
    pub basis: String,
    /// Engine-specific detail (plan buffer counts, reserved bytes, ...).
    #[serde(flatten)]
    pub detail: serde_json::Map<String, serde_json::Value>,
}

/// Outcome for a framework: either a result or a failure reason.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status")]
#[allow(clippy::large_enum_variant)]
pub enum FrameworkOutcome {
    #[serde(rename = "ok")]
    Ok(BenchResult),
    #[serde(rename = "error")]
    Error {
        framework: String,
        model: String,
        error: String,
    },
    #[serde(rename = "skipped")]
    Skipped {
        framework: String,
        model: String,
        reason: String,
    },
}

#[derive(Parser)]
#[command(name = "inferena", about = "Inference Arena")]
struct Cli {
    /// Which frameworks to benchmark (omit for all).
    #[arg(short, long, value_delimiter = ',')]
    frameworks: Option<Vec<String>>,

    /// Model to benchmark.
    #[arg(short, long, default_value = "SmolLM2-135M")]
    model: String,

    /// Path to the project root (auto-detected if omitted).
    #[arg(long)]
    root: Option<PathBuf>,

    /// Output results as JSON array instead of a human-readable table.
    #[arg(long)]
    json: bool,

    /// Directory for per-framework JSON artifacts.
    #[arg(long)]
    results_dir: Option<PathBuf>,

    /// Dry-run: validate framework+model support without running benchmarks.
    #[arg(long)]
    dry_run: bool,

    /// Disable the default discrete-GPU preference (lets backends pick an
    /// integrated GPU / APU if that's what their default selection returns).
    #[arg(long)]
    allow_integrated_gpu: bool,

    /// Disable reduced-input hardware paths for a controlled f32 comparison.
    /// The practical hardware-accelerated configuration is the default.
    #[arg(long)]
    strict: bool,

    /// Untimed executions before each measurement series.
    #[arg(long, default_value_t = 5)]
    warmup_runs: usize,

    /// Timed executions retained for each measurement series.
    #[arg(long, default_value_t = 20)]
    measurement_runs: usize,

    /// Collect each supported runner's diagnostic profile after each
    /// normal benchmark series.
    #[arg(long)]
    profile: bool,

    /// Retained hardware-timestamp samples per structured profile.
    #[arg(long, default_value_t = 3)]
    profile_samples: usize,
}

fn all_frameworks() -> Vec<&'static str> {
    let mut v = vec![
        "pytorch",
        "candle",
        "burn",
        "inferi",
        "luminal",
        "meganeura",
        "ggml",
        "onnxruntime",
        "max",
        "jax",
    ];
    if cfg!(target_os = "macos") {
        v.insert(1, "mlx"); // after pytorch
    }
    v
}

/// Framework metadata: (display_name, repo_url).
fn framework_meta(name: &str) -> (&'static str, &'static str) {
    match name {
        "pytorch" => ("PyTorch", "https://github.com/pytorch/pytorch"),
        "mlx" => ("MLX", "https://github.com/ml-explore/mlx"),
        "candle" => ("Candle", "https://github.com/huggingface/candle"),
        "burn" => ("Burn", "https://github.com/tracel-ai/burn"),
        "luminal" => ("Luminal", "https://github.com/luminal-ai/luminal"),
        "meganeura" => ("Meganeura", "https://github.com/kvark/meganeura"),
        "inferi" => ("Inferi", "https://github.com/dimforge/inferi"),
        "ggml" => ("GGML", "https://github.com/ggerganov/ggml"),
        "onnxruntime" => ("ONNX Runtime", "https://github.com/microsoft/onnxruntime"),
        "max" => ("MAX", "https://github.com/modular/modular"),
        "jax" => ("JAX", "https://github.com/jax-ml/jax"),
        _ => ("unknown", ""),
    }
}

/// Format framework name as a markdown link with backend.
/// Revision comes from the runner's JSON output ("framework_rev" field),
/// which each run.sh extracts from Cargo.lock. Python frameworks instead
/// report their installed package version as a "<name>_version" field
/// (e.g. "onnxruntime_version", "jax_version", "mlx_version") — pytorch is
/// the one special case that links straight to its GitHub release tag.
fn framework_md_link(name: &str, extra: &serde_json::Map<String, serde_json::Value>) -> String {
    let (display, url) = framework_meta(name);
    if url.is_empty() {
        return display.to_string();
    }

    let rev = extra
        .get("framework_rev")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let version_field = format!("{name}_version");
    let version = extra
        .get(&version_field)
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let link = if name == "pytorch" {
        let ver = extra
            .get("torch_version")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if !ver.is_empty() {
            let base_ver = ver.split('+').next().unwrap_or(ver);
            format!("[{display} {ver}]({url}/releases/tag/v{base_ver})")
        } else {
            format!("[{display}]({url})")
        }
    } else if !version.is_empty() {
        format!("[{display} {version}]({url})")
    } else if rev.is_empty() {
        format!("[{display}]({url})")
    } else {
        format!("[{display}]({url}/tree/{rev})")
    };

    // Append backend in parens if available.
    let backend = extra.get("backend").and_then(|v| v.as_str()).unwrap_or("");
    if !backend.is_empty() {
        format!("{link} ({backend})")
    } else {
        // For Rust frameworks, infer backend from framework name.
        let inferred = match name {
            "candle" => "CPU",
            "burn" => "wgpu",
            "luminal" => "CPU",
            "meganeura" => {
                if cfg!(target_os = "macos") {
                    "Metal"
                } else {
                    "Vulkan"
                }
            }
            "inferi" => {
                if cfg!(target_os = "macos") {
                    "Metal"
                } else {
                    "Vulkan"
                }
            }
            "mlx" => "MLX",
            "ggml" => "CPU",
            "onnxruntime" => "CPU",
            "max" => "CPU",
            "jax" => "CPU",
            _ => "",
        };
        if inferred.is_empty() {
            link
        } else {
            format!("{link} ({inferred})")
        }
    }
}

fn project_root(cli_root: Option<&Path>) -> PathBuf {
    if let Some(r) = cli_root {
        // Don't canonicalize — on Windows, std::fs::canonicalize returns paths
        // with the `\\?\` extended-length prefix, which git-bash's bash cannot
        // open. Callers already pass an absolute path.
        return r.to_path_buf();
    }
    let exe = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("."));
    let mut dir = exe.parent().unwrap_or(Path::new(".")).to_path_buf();
    for _ in 0..5 {
        if dir.join("Cargo.toml").exists() && dir.join("frameworks").exists() {
            return dir;
        }
        if !dir.pop() {
            break;
        }
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

/// On Linux/Mesa, parse `vulkaninfo --summary` for the first DISCRETE_GPU and
/// return its `vendorID:deviceID` pair (lowercase hex, no `0x`), shaped for
/// `MESA_VK_DEVICE_SELECT`. Returns None if vulkaninfo is missing or no
/// discrete device is visible — callers should then leave the env alone.
fn discover_discrete_gpu_pci_id() -> Option<String> {
    let output = Command::new("vulkaninfo").arg("--summary").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);

    let extract_hex = |line: &str| -> Option<String> {
        line.split('=')
            .nth(1)?
            .trim()
            .strip_prefix("0x")
            .map(|s| s.to_lowercase())
    };

    let mut vendor: Option<String> = None;
    let mut device: Option<String> = None;
    for line in text.lines() {
        let t = line.trim();
        if t.starts_with("GPU") && t.ends_with(':') {
            vendor = None;
            device = None;
        } else if t.starts_with("vendorID") {
            vendor = extract_hex(t);
        } else if t.starts_with("deviceID") {
            device = extract_hex(t);
        } else if t.starts_with("deviceType")
            && t.contains("DISCRETE_GPU")
            && let (Some(v), Some(d)) = (vendor.as_ref(), device.as_ref())
        {
            return Some(format!("{v}:{d}"));
        }
    }
    None
}

/// Validate the arithmetic contract reported by an instrumented runner.
///
/// A comparison-class label alone is insufficient evidence: a CUDA
/// environment variable or backend flag can silently re-enable TF32. Keep
/// this independent of numerical agreement so a close answer obtained under
/// the wrong arithmetic policy is still rejected.
fn precision_contract_error(result: &BenchResult, requested: &str) -> Option<String> {
    let Some(precision) = result
        .extra
        .get("precision")
        .and_then(serde_json::Value::as_object)
    else {
        return Some("missing precision object".to_string());
    };
    let text = |key: &str| precision.get(key).and_then(serde_json::Value::as_str);
    let flag = |key: &str| precision.get(key).and_then(serde_json::Value::as_bool);

    for key in ["tensor_storage", "accumulation", "output"] {
        if text(key) != Some("f32") {
            return Some(format!(
                "precision.{key}={:?}; Inferena requires f32 storage, accumulation, and output",
                text(key)
            ));
        }
    }

    let accelerated = requested == "accelerated-f32";
    if flag("reduced_precision_allowed") != Some(accelerated) {
        return Some(format!(
            "precision.reduced_precision_allowed={:?}; expected {accelerated}",
            flag("reduced_precision_allowed")
        ));
    }

    let backend = result
        .extra
        .get("backend")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    match (result.framework.as_str(), requested) {
        ("pytorch", "strict-f32") if backend.starts_with("CUDA") => {
            for key in ["cuda_matmul_allow_tf32", "cudnn_allow_tf32"] {
                if flag(key) != Some(false) {
                    return Some(format!(
                        "precision.{key}={:?}; strict CUDA requires false",
                        flag(key)
                    ));
                }
            }
            if text("nvidia_tf32_override") != Some("0") {
                return Some(format!(
                    "precision.nvidia_tf32_override={:?}; strict CUDA requires 0",
                    text("nvidia_tf32_override")
                ));
            }
        }
        ("pytorch", "accelerated-f32") if backend.starts_with("CUDA") => {
            for key in ["cuda_matmul_allow_tf32", "cudnn_allow_tf32"] {
                if flag(key) != Some(true) {
                    return Some(format!(
                        "precision.{key}={:?}; accelerated CUDA requires true",
                        flag(key)
                    ));
                }
            }
            if text("nvidia_tf32_override") == Some("0") {
                return Some(
                    "precision.nvidia_tf32_override=0 disables the requested accelerated path"
                        .to_string(),
                );
            }
        }
        ("meganeura", "strict-f32") if flag("f16_cooperative_matrix_permitted") != Some(false) => {
            return Some(format!(
                "precision.f16_cooperative_matrix_permitted={:?}; strict mode requires false",
                flag("f16_cooperative_matrix_permitted")
            ));
        }
        ("meganeura", "accelerated-f32")
            if flag("f16_cooperative_matrix_permitted") != Some(true) =>
        {
            return Some(format!(
                "precision.f16_cooperative_matrix_permitted={:?}; accelerated mode requires true",
                flag("f16_cooperative_matrix_permitted")
            ));
        }
        _ => {}
    }

    if result.framework == "meganeura" && flag("persistent_f16_tensors") != Some(false) {
        return Some(format!(
            "precision.persistent_f16_tensors={:?}; audited workloads require false",
            flag("persistent_f16_tensors")
        ));
    }

    None
}

#[allow(clippy::too_many_arguments)]
fn run_framework(
    root: &Path,
    framework: &str,
    model: &str,
    dry_run: bool,
    prefer_discrete_gpu: bool,
    discrete_gpu_pci_id: Option<&str>,
    strict: bool,
    warmup_runs: usize,
    measurement_runs: usize,
    profile_dir: Option<&Path>,
    profile_samples: usize,
) -> FrameworkOutcome {
    let precision = if strict {
        "strict-f32"
    } else {
        "accelerated-f32"
    };
    let fw_dir = root.join("frameworks").join(framework);
    let run_script = fw_dir.join("run.sh");

    if !run_script.exists() {
        return FrameworkOutcome::Skipped {
            framework: framework.to_string(),
            model: model.to_string(),
            reason: format!("run.sh not found at {}", run_script.display()),
        };
    }

    if dry_run {
        eprintln!("[{framework}] dry-run for {model} ...");
    } else {
        eprintln!("[{framework}] running benchmark for {model} ...");
    }

    // Always use bash (Git Bash on Windows).
    // Inherits environment so WGPU_BACKEND, HSA_OVERRIDE_GFX_VERSION, etc. propagate.
    let mut cmd = Command::new("bash");
    cmd.arg(&run_script).arg(model).current_dir(&fw_dir);
    cmd.env("INFERENA_STRICT", if strict { "1" } else { "0" })
        .env("INFERENA_WARMUP_RUNS", warmup_runs.to_string())
        .env("INFERENA_MEASUREMENT_RUNS", measurement_runs.to_string());
    if let Some(profile_dir) = profile_dir {
        cmd.env("INFERENA_PROFILE_DIR", profile_dir)
            .env("INFERENA_PROFILE_SAMPLES", profile_samples.to_string());
        if framework == "meganeura" {
            cmd.env("MEGANEURA_GPU_TIMING", "1");
        }
    }
    if dry_run {
        cmd.env("INFERENA_DRY_RUN", "1");
    }
    // Steer adapter selection at the discrete GPU on hybrid systems. wgpu's
    // env-driven power preference is a soft hint; on Linux/Mesa we also set
    // MESA_VK_DEVICE_SELECT, which filters adapters at the Vulkan loader so
    // even frameworks that don't plumb env into request_adapter get the
    // discrete device. Caller overrides win.
    if prefer_discrete_gpu {
        if std::env::var_os("WGPU_POWER_PREFERENCE").is_none() {
            cmd.env("WGPU_POWER_PREFERENCE", "high");
        }
        if let Some(pci) = discrete_gpu_pci_id
            && std::env::var_os("MESA_VK_DEVICE_SELECT").is_none()
        {
            cmd.env("MESA_VK_DEVICE_SELECT", pci);
        }
    }
    let output = cmd.output();

    let output = match output {
        Ok(o) => o,
        Err(e) => {
            return FrameworkOutcome::Error {
                framework: framework.to_string(),
                model: model.to_string(),
                error: format!("failed to execute run.sh: {e}"),
            };
        }
    };

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let combined = format!("{stderr}\n{stdout}");
        // Truncate long error output for readability.
        let stderr_short: String = stderr.lines().take(20).collect::<Vec<_>>().join("\n");

        // "Unknown model" / "unsupported" → skip, not error.
        let lower = combined.to_lowercase();
        if lower.contains("unknown model") || lower.contains("unsupported") {
            return FrameworkOutcome::Skipped {
                framework: framework.to_string(),
                model: model.to_string(),
                reason: format!("model not supported by {framework}"),
            };
        }

        return FrameworkOutcome::Error {
            framework: framework.to_string(),
            model: model.to_string(),
            error: format!("{}: {}", output.status, stderr_short),
        };
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let json_str = match stdout
        .lines()
        .rev()
        .find(|l| l.trim_start().starts_with('{'))
    {
        Some(s) => s,
        None => {
            // In dry-run, no JSON output is fine — the framework validated OK.
            if dry_run {
                return FrameworkOutcome::Skipped {
                    framework: framework.to_string(),
                    model: model.to_string(),
                    reason: "dry-run OK (no benchmark data)".to_string(),
                };
            }
            return FrameworkOutcome::Error {
                framework: framework.to_string(),
                model: model.to_string(),
                error: "no JSON found in stdout".to_string(),
            };
        }
    };

    match serde_json::from_str::<BenchResult>(json_str) {
        Ok(r) => {
            if matches!(framework, "pytorch" | "meganeura") {
                let expected_protocol = if framework == "pytorch" {
                    "inferena-cuda-graphs-v2"
                } else {
                    PAPER_PROTOCOL
                };
                let reported_protocol = r
                    .extra
                    .get("protocol")
                    .and_then(|value| value.get("name"))
                    .and_then(|value| value.as_str());
                let expected_class = if precision == "strict-f32" {
                    "strict-f32"
                } else {
                    "reduced-input-f32-accumulate"
                };
                let reported_class = r
                    .extra
                    .get("precision")
                    .and_then(|value| value.get("comparison_class"))
                    .and_then(|value| value.as_str());
                if reported_protocol != Some(expected_protocol)
                    || reported_class != Some(expected_class)
                {
                    return FrameworkOutcome::Error {
                        framework: framework.to_string(),
                        model: model.to_string(),
                        error: format!(
                            "runner reported protocol={reported_protocol:?}, \
                             precision class={reported_class:?}; expected \
                             {expected_protocol}/{expected_class}"
                        ),
                    };
                }
                if let Some(error) = precision_contract_error(&r, precision) {
                    return FrameworkOutcome::Error {
                        framework: framework.to_string(),
                        model: model.to_string(),
                        error: format!("precision contract failed: {error}"),
                    };
                }
                if r.outputs.output_shape.is_empty()
                    || r.outputs.logits_sample.len() != 256
                    || r.outputs.grad_norm.is_none()
                    || r.outputs.gradient_norms.is_empty()
                {
                    return FrameworkOutcome::Error {
                        framework: framework.to_string(),
                        model: model.to_string(),
                        error: format!(
                            "instrumented result is incomplete: output_shape={:?}, \
                             output_samples={}, grad_norm={}, parameter_gradients={}",
                            r.outputs.output_shape,
                            r.outputs.logits_sample.len(),
                            r.outputs.grad_norm.is_some(),
                            r.outputs.gradient_norms.len()
                        ),
                    };
                }
            }
            FrameworkOutcome::Ok(r)
        }
        Err(e) => FrameworkOutcome::Error {
            framework: framework.to_string(),
            model: model.to_string(),
            error: format!("JSON parse error: {e}"),
        },
    }
}

/// Save each framework's result as a separate JSON file in results/.
fn save_results(results_dir: &Path, model: &str, outcomes: &[FrameworkOutcome]) {
    std::fs::create_dir_all(results_dir).ok();

    for outcome in outcomes {
        let (fw, content) = match outcome {
            FrameworkOutcome::Ok(r) => {
                (&r.framework, serde_json::to_string_pretty(outcome).unwrap())
            }
            FrameworkOutcome::Error { framework, .. } => {
                (framework, serde_json::to_string_pretty(outcome).unwrap())
            }
            FrameworkOutcome::Skipped { framework, .. } => {
                (framework, serde_json::to_string_pretty(outcome).unwrap())
            }
        };
        let path = results_dir.join(format!("{model}_{fw}.json"));
        if let Err(e) = std::fs::write(&path, &content) {
            eprintln!("Warning: failed to save {}: {e}", path.display());
        }
    }

    // Also save a combined summary.
    let summary_path = results_dir.join(format!("{model}_summary.json"));
    let summary = serde_json::to_string_pretty(outcomes).unwrap();
    if let Err(e) = std::fs::write(&summary_path, &summary) {
        eprintln!("Warning: failed to save summary: {e}");
    }

    eprintln!();
    eprintln!("Results saved to {}/", results_dir.display());
}

/// Compute error metrics between two logit sample vectors.
fn compute_errors(a: &[f64], b: &[f64]) -> (f64, f64, f64, f64) {
    let n = a.len().min(b.len());
    if n == 0 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let mut max_err = 0.0f64;
    let mut sum_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut sum_ref_sq = 0.0f64;
    for i in 0..n {
        let diff = (a[i] - b[i]).abs();
        max_err = max_err.max(diff);
        sum_abs += diff;
        sum_sq += diff * diff;
        sum_ref_sq += a[i] * a[i];
    }
    let mae = sum_abs / n as f64;
    let rmse = (sum_sq / n as f64).sqrt();
    let rel = if sum_ref_sq > 0.0 {
        (sum_sq / sum_ref_sq).sqrt()
    } else if sum_sq > 0.0 {
        f64::INFINITY
    } else {
        0.0
    };
    (max_err, mae, rmse, rel)
}

fn relative_scalar_error(reference: f64, other: f64) -> f64 {
    (reference - other).abs() / reference.abs().max(other.abs()).max(1e-12)
}

fn compare_gradient_norms(
    reference: &std::collections::BTreeMap<String, f64>,
    other: &std::collections::BTreeMap<String, f64>,
) -> f64 {
    if reference.is_empty() && other.is_empty() {
        return f64::NAN;
    }
    if reference.keys().ne(other.keys()) {
        return f64::INFINITY;
    }
    let mut difference_sq = 0.0;
    let mut reference_sq = 0.0;
    for (name, reference_norm) in reference {
        let difference = reference_norm - other[name];
        difference_sq += difference * difference;
        reference_sq += reference_norm * reference_norm;
    }
    if reference_sq > 0.0 {
        (difference_sq / reference_sq).sqrt()
    } else if difference_sq > 0.0 {
        f64::INFINITY
    } else {
        0.0
    }
}

struct ComparisonMetrics {
    hash_match: bool,
    output_shape_match: bool,
    output_sample_count_match: bool,
    loss_diff: f64,
    loss_relative_error: f64,
    max_error: f64,
    mae: f64,
    rmse: f64,
    output_relative_l2_error: f64,
    total_gradient_relative_error: f64,
    parameter_gradient_relative_l2_error: f64,
    gradients_available: bool,
    forward_valid: bool,
    training_valid: bool,
    status: &'static str,
}

fn compare_result(reference: &BenchResult, other: &BenchResult) -> ComparisonMetrics {
    let hash_match = reference.outputs.logits_hash == other.outputs.logits_hash;
    let output_shape_match = reference.outputs.output_shape == other.outputs.output_shape;
    let output_sample_count_match =
        reference.outputs.logits_sample.len() == other.outputs.logits_sample.len();
    let loss_diff = (reference.outputs.loss - other.outputs.loss).abs();
    let loss_relative_error = relative_scalar_error(reference.outputs.loss, other.outputs.loss);
    let (max_error, mae, rmse, output_relative_l2_error) = if output_sample_count_match {
        compute_errors(
            &reference.outputs.logits_sample,
            &other.outputs.logits_sample,
        )
    } else {
        (f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::INFINITY)
    };
    let total_gradient_relative_error = match (reference.outputs.grad_norm, other.outputs.grad_norm)
    {
        (Some(reference_grad), Some(other_grad)) => {
            (reference_grad - other_grad).abs() / reference_grad.abs().max(1e-12)
        }
        _ => f64::NAN,
    };
    let parameter_gradient_relative_l2_error = compare_gradient_norms(
        &reference.outputs.gradient_norms,
        &other.outputs.gradient_norms,
    );
    let gradients_available = total_gradient_relative_error.is_finite()
        && parameter_gradient_relative_l2_error.is_finite();
    let gradients_pass = gradients_available
        && total_gradient_relative_error < 0.05
        && parameter_gradient_relative_l2_error < 0.05;
    let gradients_close = gradients_available
        && total_gradient_relative_error < 0.10
        && parameter_gradient_relative_l2_error < 0.10;
    let forward_valid = output_shape_match
        && output_sample_count_match
        && loss_relative_error < 0.01
        && output_relative_l2_error < 0.01;
    let forward_close = output_shape_match
        && output_sample_count_match
        && loss_relative_error < 0.05
        && output_relative_l2_error < 0.10;
    let training_valid = forward_valid && gradients_pass;
    let status = if hash_match && output_shape_match && loss_relative_error < 0.01 && gradients_pass
    {
        "EXACT MATCH"
    } else if training_valid {
        "PASS (<1% forward, <5% gradient)"
    } else if forward_valid {
        "INFERENCE PASS; TRAINING FAIL"
    } else if forward_close && gradients_close {
        "CLOSE"
    } else {
        "DIFFERENT MODEL"
    };

    ComparisonMetrics {
        hash_match,
        output_shape_match,
        output_sample_count_match,
        loss_diff,
        loss_relative_error,
        max_error,
        mae,
        rmse,
        output_relative_l2_error,
        total_gradient_relative_error,
        parameter_gradient_relative_l2_error,
        gradients_available,
        forward_valid,
        training_valid,
        status,
    }
}

fn compare_outputs(results: &[&BenchResult]) {
    if results.len() < 2 {
        return;
    }
    let reference = results[0];
    // Skip detailed comparison if PyTorch (ground truth) isn't the reference.
    if reference.framework != "pytorch" {
        eprintln!();
        eprintln!(
            "=== Output comparison skipped (no PyTorch ground truth, reference: {}) ===",
            reference.framework
        );
        return;
    }
    eprintln!();
    eprintln!(
        "=== Output comparison (reference: {}) ===",
        reference.framework
    );
    eprintln!(
        "  {:<12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}  {}",
        "Framework",
        "Loss Diff",
        "Max Error",
        "MAE",
        "RMSE",
        "Rel Error",
        "Grad Rel",
        "Param Grad",
        "Status"
    );
    eprintln!("  {}", "-".repeat(118));
    for other in results
        .iter()
        .copied()
        .filter(|r| r.framework != reference.framework)
    {
        let comparison = compare_result(reference, other);
        eprintln!(
            "  {:<12} {:>12.6e} {:>12.6e} {:>12.6e} {:>12.6e} {:>12.6e} {:>12.6e} {:>12.6e}  {}",
            other.framework,
            comparison.loss_diff,
            comparison.max_error,
            comparison.mae,
            comparison.rmse,
            comparison.output_relative_l2_error,
            comparison.total_gradient_relative_error,
            comparison.parameter_gradient_relative_l2_error,
            comparison.status
        );
    }
}

fn mib(bytes: u64) -> String {
    format!("{:.1}", bytes as f64 / (1024.0 * 1024.0))
}

/// Print GPU memory to stderr.
///
/// Deliberately not part of the markdown table: the per-phase figures come
/// from different accounting in each engine, so they are shown with their
/// `basis` label rather than as a comparable column. The JSON artifact
/// carries the full detail.
fn print_memory(results: &[&BenchResult]) {
    if !results.iter().any(|r| r.memory.is_some()) {
        return;
    }
    eprintln!();
    eprintln!("=== GPU memory (MiB) ===");
    eprintln!(
        "  {:<12} {:>12} {:>12} {:>12} {:>12}  {}",
        "Framework", "Inference", "Latency", "Training", "Process", "Accounting basis"
    );
    eprintln!("  {}", "-".repeat(94));
    for result in results {
        let Some(memory) = &result.memory else {
            continue;
        };
        let phase = |name: &str| {
            memory
                .phases
                .get(name)
                .map(|p| mib(p.allocated_bytes))
                .unwrap_or_else(|| "—".to_string())
        };
        let process = memory
            .device
            .as_ref()
            .and_then(|d| d.process_bytes)
            .map(mib)
            .unwrap_or_else(|| "—".to_string());
        // Every phase of one engine shares the same basis; show it once.
        let basis = memory
            .phases
            .values()
            .next()
            .map(|p| p.basis.as_str())
            .unwrap_or("—");
        eprintln!(
            "  {:<12} {:>12} {:>12} {:>12} {:>12}  {}",
            result.framework,
            phase("inference"),
            phase("latency"),
            phase("training"),
            process,
            basis
        );
    }
    let sources: std::collections::BTreeSet<&str> = results
        .iter()
        .filter_map(|r| r.memory.as_ref())
        .filter_map(|m| m.device.as_ref())
        .filter_map(|d| d.process_source.as_deref())
        .collect();
    if !sources.is_empty() {
        eprintln!(
            "  Process column is per-process driver accounting via {}.",
            sources.into_iter().collect::<Vec<_>>().join(", ")
        );
    }
}

/// Determine which frameworks "match" the reference (first successful result).
/// Returns a set of framework names that passed correctness checks.
/// Uses PyTorch as the reference (ground truth). If PyTorch isn't present,
/// all frameworks are considered matching (caller handles this case).
fn matching_frameworks(
    successes: &[&BenchResult],
    require_gradients: bool,
) -> std::collections::HashSet<String> {
    let mut matching = std::collections::HashSet::new();
    if successes.is_empty() {
        return matching;
    }
    // Find PyTorch as ground truth; fall back to first framework.
    let reference = *successes
        .iter()
        .find(|r| r.framework == "pytorch")
        .unwrap_or(&successes[0]);
    matching.insert(reference.framework.clone());
    for other in successes
        .iter()
        .copied()
        .filter(|r| r.framework != reference.framework)
    {
        let comparison = compare_result(reference, other);
        if comparison.forward_valid && (!require_gradients || comparison.training_valid) {
            matching.insert(other.framework.clone());
        }
    }
    matching
}

fn finite_json(value: f64) -> serde_json::Value {
    if value.is_finite() {
        serde_json::json!(value)
    } else {
        serde_json::Value::Null
    }
}

fn git_revision(path: &Path) -> String {
    let revision = Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .current_dir(path)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .filter(|revision| !revision.is_empty())
        .unwrap_or_else(|| "unknown".to_string());
    let dirty = Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=normal"])
        .current_dir(path)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .is_some_and(|output| !output.stdout.is_empty());
    if dirty {
        format!("{revision}-dirty")
    } else {
        revision
    }
}

fn annotate_validation(outcomes: &mut [FrameworkOutcome], benchmark_revision: &str) {
    let reference = outcomes.iter().find_map(|outcome| match outcome {
        FrameworkOutcome::Ok(result) if result.framework == "pytorch" => Some(result.clone()),
        _ => None,
    });

    for outcome in outcomes {
        let FrameworkOutcome::Ok(result) = outcome else {
            continue;
        };
        result.extra.insert(
            "benchmark_rev".to_string(),
            serde_json::json!(benchmark_revision),
        );
        let validation = match &reference {
            None => serde_json::json!({
                "comparison_performed": false,
                "reference_framework": null,
                "forward_valid": null,
                "training_valid": null,
                "status": "NO REFERENCE",
            }),
            Some(reference) if result.framework == reference.framework => serde_json::json!({
                "comparison_performed": true,
                "reference_framework": reference.framework,
                "forward_valid": true,
                "training_valid": true,
                "status": "REFERENCE",
            }),
            Some(reference) => {
                let comparison = compare_result(reference, result);
                serde_json::json!({
                    "comparison_performed": true,
                    "reference_framework": reference.framework,
                    "forward_valid": comparison.forward_valid,
                    "training_valid": comparison.training_valid,
                    "status": comparison.status,
                    "full_output_hash_match": comparison.hash_match,
                    "output_shape_match": comparison.output_shape_match,
                    "output_sample_count_match": comparison.output_sample_count_match,
                    "gradients_available": comparison.gradients_available,
                    "loss_absolute_error": finite_json(comparison.loss_diff),
                    "loss_relative_error": finite_json(comparison.loss_relative_error),
                    "output_max_absolute_error": finite_json(comparison.max_error),
                    "output_mae": finite_json(comparison.mae),
                    "output_rmse": finite_json(comparison.rmse),
                    "output_relative_l2_error": finite_json(
                        comparison.output_relative_l2_error
                    ),
                    "total_gradient_relative_error": finite_json(
                        comparison.total_gradient_relative_error
                    ),
                    "parameter_gradient_relative_l2_error": finite_json(
                        comparison.parameter_gradient_relative_l2_error
                    ),
                })
            }
        };
        result.extra.insert("validation".to_string(), validation);
    }
}

/// Heuristic: does this backend string look like a CPU backend?
/// Matches "CPU", "CPUExecutionProvider", "faster-whisper (CTranslate2, CPU)".
fn is_cpu_backend(backend: &str) -> bool {
    backend.to_uppercase().contains("CPU")
}

fn result_backend(r: &BenchResult) -> &str {
    r.extra
        .get("backend")
        .and_then(|v| v.as_str())
        .unwrap_or("")
}

fn print_table(outcomes: &[FrameworkOutcome], successes: &[&BenchResult]) {
    // Check if PyTorch (ground truth) ran successfully.
    let has_pytorch = successes.iter().any(|r| r.framework == "pytorch");

    // If PyTorch ran on a non-CPU backend, skip CPU-only rows from other
    // frameworks — a CPU-vs-GPU comparison isn't meaningful. The reverse
    // (PyTorch on CPU, others on GPU) is fine and stays as-is.
    let pytorch_on_gpu = successes
        .iter()
        .find(|r| r.framework == "pytorch")
        .is_some_and(|r| !is_cpu_backend(result_backend(r)));
    if !has_pytorch && !successes.is_empty() {
        eprintln!();
        eprintln!("⚠ WARNING: PyTorch (ground truth) did not run successfully.");
        eprintln!("  Results are shown but NOT validated against a reference implementation.");
        eprintln!("  Loss-based correctness checks are disabled.");
        eprintln!();
    }

    // Forward results can remain valid even when a reduced-precision
    // backward pass fails the gradient gate. Track the two scopes
    // independently so a training failure does not erase valid inference.
    let matching_forward = if has_pytorch {
        matching_frameworks(successes, false)
    } else {
        successes.iter().map(|r| r.framework.clone()).collect()
    };
    let matching_training = if has_pytorch {
        matching_frameworks(successes, true)
    } else {
        successes.iter().map(|r| r.framework.clone()).collect()
    };

    // Find best values only among frameworks validated for each scope.
    let mut best_compile = f64::MAX;
    let mut best_inference = f64::MAX;
    let mut best_latency = f64::MAX;
    let mut best_training = f64::MAX;
    for o in outcomes {
        if let FrameworkOutcome::Ok(r) = o {
            let forward_valid = matching_forward.contains(&r.framework);
            let training_valid = matching_training.contains(&r.framework);
            if forward_valid && r.timings.compile_s < best_compile {
                best_compile = r.timings.compile_s;
            }
            if forward_valid && r.timings.inference_ms < best_inference {
                best_inference = r.timings.inference_ms;
            }
            if forward_valid && r.timings.latency_ms > 0.0 && r.timings.latency_ms < best_latency {
                best_latency = r.timings.latency_ms;
            }
            if training_valid
                && r.timings.training_ms > 0.0
                && r.timings.training_ms < best_training
            {
                best_training = r.timings.training_ms;
            }
        }
    }

    let causal_lm = successes
        .iter()
        .any(|result| result.model.starts_with("SmolLM"));
    let (forward_heading, latency_heading) = if causal_lm {
        ("Prefill (ms)", "Stateless 1-token (ms)")
    } else {
        ("Inference (ms)", "Latency (ms)")
    };
    println!(
        "| Platform | Framework | Compile (s) | {forward_heading} | {latency_heading} | Training (ms) | Loss |"
    );
    println!(
        "|----------|-----------|:-----------:|:--------------:|:------------:|:-------------:|:----:|"
    );

    let mut platform_shown = false;
    for outcome in outcomes {
        // Skip CPU-only rows when PyTorch is on GPU.
        if pytorch_on_gpu
            && let FrameworkOutcome::Ok(r) = outcome
            && is_cpu_backend(result_backend(r))
        {
            continue;
        }

        // Show platform (device name) only on the first row.
        let platform = if !platform_shown {
            if let FrameworkOutcome::Ok(r) = outcome {
                platform_shown = true;
                if cfg!(target_os = "windows") {
                    format!("{} (Windows)", r.gpu_name)
                } else {
                    r.gpu_name.clone()
                }
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        match outcome {
            FrameworkOutcome::Ok(r) => {
                let link = framework_md_link(&r.framework, &r.extra);
                let forward_valid = matching_forward.contains(&r.framework);
                let training_valid = matching_training.contains(&r.framework);

                let fmt_val = |val: f64, best: f64, is_time: bool, valid: bool| -> String {
                    let s = if is_time && val == 0.0 {
                        "—".to_string()
                    } else if is_time && val < 10.0 {
                        // Sub-10ms values need a decimal to distinguish e.g.
                        // 2.5ms inference from 3.4ms training — otherwise both
                        // round to "3" and the visual extension vanishes.
                        format!("{:.1}", val)
                    } else if is_time {
                        format!("{:.0}", val)
                    } else {
                        format!("{:.2}", val)
                    };
                    if !valid {
                        format!("~~{s}~~")
                    } else if val > 0.0 && (val - best).abs() < 0.01 {
                        format!("**{s}**")
                    } else {
                        s
                    }
                };

                let compile = fmt_val(r.timings.compile_s, best_compile, false, forward_valid);
                let inference =
                    fmt_val(r.timings.inference_ms, best_inference, true, forward_valid);
                let latency = fmt_val(r.timings.latency_ms, best_latency, true, forward_valid);
                let training = fmt_val(r.timings.training_ms, best_training, true, training_valid);
                let loss = if forward_valid {
                    format!("{:.2}", r.outputs.loss)
                } else {
                    format!("~~{:.2}~~", r.outputs.loss)
                };

                println!(
                    "| {platform} | {link} | {compile} | {inference} | {latency} | {training} | {loss} |"
                );
            }
            FrameworkOutcome::Error { framework, .. } => {
                let (display, url) = framework_meta(framework);
                let link = if url.is_empty() {
                    display.to_string()
                } else {
                    format!("[{display}]({url})")
                };
                println!("| {platform} | {link} | ✗ | ✗ | ✗ | ✗ | |");
            }
            FrameworkOutcome::Skipped { framework, .. } => {
                let (display, url) = framework_meta(framework);
                let link = if url.is_empty() {
                    display.to_string()
                } else {
                    format!("[{display}]({url})")
                };
                println!("| {platform} | {link} | — | — | — | — | |");
            }
        }
    }
}

fn main() {
    let cli = Cli::parse();
    if cli.measurement_runs == 0 {
        eprintln!("--measurement-runs must be at least 1");
        std::process::exit(2);
    }
    if cli.profile && cli.profile_samples == 0 {
        eprintln!("--profile-samples must be at least 1");
        std::process::exit(2);
    }
    let root = project_root(cli.root.as_deref());
    let results_dir = cli
        .results_dir
        .clone()
        .unwrap_or_else(|| root.join("results"));
    let profile_dir = cli.profile.then(|| {
        let directory = results_dir.join("profiles");
        if directory.is_absolute() {
            directory
        } else {
            std::env::current_dir()
                .expect("resolve current directory for profile output")
                .join(directory)
        }
    });

    let frameworks: Vec<&str> = match &cli.frameworks {
        Some(list) => list.iter().map(String::as_str).collect(),
        None => all_frameworks(),
    };

    let prefer_discrete_gpu = !cli.allow_integrated_gpu;
    let discrete_gpu_pci_id = if prefer_discrete_gpu && cfg!(target_os = "linux") {
        discover_discrete_gpu_pci_id()
    } else {
        None
    };
    if let Some(pci) = &discrete_gpu_pci_id {
        eprintln!("[inferena] pinning Vulkan to discrete GPU (MESA_VK_DEVICE_SELECT={pci})");
    }
    let mut outcomes = Vec::new();
    for fw in &frameworks {
        outcomes.push(run_framework(
            &root,
            fw,
            &cli.model,
            cli.dry_run,
            prefer_discrete_gpu,
            discrete_gpu_pci_id.as_deref(),
            cli.strict,
            cli.warmup_runs,
            cli.measurement_runs,
            profile_dir.as_deref(),
            cli.profile_samples,
        ));
    }
    let benchmark_revision = git_revision(&root);
    annotate_validation(&mut outcomes, &benchmark_revision);

    // Collect successful results for comparison.
    let successes: Vec<&BenchResult> = outcomes
        .iter()
        .filter_map(|o| match o {
            FrameworkOutcome::Ok(r) => Some(r),
            _ => None,
        })
        .collect();

    if cli.dry_run {
        // Dry-run: just show support matrix, no table/comparison/save.
        let model = &cli.model;
        eprintln!();
        eprintln!("=== Dry-run: {model} ===");
        for outcome in &outcomes {
            match outcome {
                FrameworkOutcome::Ok(r) => {
                    eprintln!("  ✓ {}", r.framework);
                }
                FrameworkOutcome::Error {
                    framework, error, ..
                } => {
                    let short = error
                        .lines()
                        .next()
                        .unwrap_or(error)
                        .chars()
                        .take(70)
                        .collect::<String>();
                    eprintln!("  ✗ {framework}: {short}");
                }
                FrameworkOutcome::Skipped {
                    framework, reason, ..
                } => {
                    eprintln!("  — {framework}: {reason}");
                }
            }
        }
        return;
    }

    if cli.json {
        println!("{}", serde_json::to_string_pretty(&outcomes).unwrap());
    } else {
        print_table(&outcomes, &successes);
        // Dump each framework error to stderr so users can see what failed
        // instead of staring at a wall of ✗.
        let mut printed_header = false;
        for outcome in &outcomes {
            if let FrameworkOutcome::Error {
                framework, error, ..
            } = outcome
            {
                if !printed_header {
                    eprintln!();
                    eprintln!("=== Framework errors ===");
                    printed_header = true;
                }
                eprintln!("[{framework}] {error}");
            }
        }
    }

    // Save results even when everything fails — the per-framework JSON captures
    // the stderr excerpt, which is often the only record of what went wrong.
    save_results(&results_dir, &cli.model, &outcomes);

    if successes.is_empty() {
        eprintln!("No successful benchmark results.");
        std::process::exit(1);
    }

    compare_outputs(&successes);
    print_memory(&successes);
}

#[cfg(test)]
mod tests {
    use super::{
        BenchResult, MemoryReport, Outputs, Timings, compare_gradient_norms, compare_result,
        precision_contract_error, relative_scalar_error,
    };
    use std::collections::BTreeMap;

    /// Archived results predate the memory report and must stay readable —
    /// the paper's analysis reads them alongside new runs.
    #[test]
    fn result_without_memory_still_parses() {
        let json = serde_json::json!({
            "framework": "meganeura",
            "model": "ResNet-50",
            "device": "Vulkan",
            "gpu_name": "gpu",
            "timings": {
                "compile_s": 0.671, "inference_ms": 5.271,
                "latency_ms": 3.657, "training_ms": 31.935,
            },
            "outputs": {
                "logits_hash": "sha256:0", "output_shape": [4, 1000],
                "logits_sample": [0.0], "loss": 6.9,
            },
        });
        let parsed: BenchResult = serde_json::from_value(json).unwrap();
        assert!(parsed.memory.is_none());
        // An absent report must not be serialized back as a null field.
        let reserialized = serde_json::to_value(&parsed).unwrap();
        assert!(!reserialized.as_object().unwrap().contains_key("memory"));
        let mut with_execution = reserialized;
        with_execution["execution"] = serde_json::json!({
            "compiled": true,
            "cuda_graphs": {"phases": {"training": {"status": "captured-and-validated"}}}
        });
        let parsed: BenchResult = serde_json::from_value(with_execution.clone()).unwrap();
        assert_eq!(
            serde_json::to_value(parsed).unwrap()["execution"],
            with_execution["execution"]
        );
    }

    #[test]
    fn memory_report_keeps_basis_and_engine_detail() {
        let json = serde_json::json!({
            "device": {
                "total_bytes": 17_170_956_288u64,
                "process_bytes": 1_234_567u64,
                "process_source": "vulkan-memory-budget",
                "sampled_at": "after-training",
            },
            "phases": {
                "training": {
                    "allocated_bytes": 900u64,
                    "basis": "execution-plan physical allocation after lifetime aliasing",
                    "plan_logical_bytes": 1500u64,
                },
            },
        });
        let report: MemoryReport = serde_json::from_value(json).unwrap();
        let training = &report.phases["training"];
        assert_eq!(training.allocated_bytes, 900);
        assert!(training.basis.contains("aliasing"));
        // Engine-specific keys survive the round trip through `detail`.
        assert_eq!(training.detail["plan_logical_bytes"], 1500);
        assert_eq!(
            report.device.unwrap().process_source.unwrap(),
            "vulkan-memory-budget"
        );
    }

    /// A phase figure is meaningless without knowing what it counted, so
    /// `basis` is required rather than defaulted.
    #[test]
    fn phase_memory_requires_a_basis() {
        let json = serde_json::json!({
            "phases": { "inference": { "allocated_bytes": 100u64 } },
        });
        assert!(serde_json::from_value::<MemoryReport>(json).is_err());
    }

    fn result(shape: Vec<usize>, gradients: bool) -> BenchResult {
        BenchResult {
            framework: "test".to_string(),
            model: "model".to_string(),
            device: "device".to_string(),
            gpu_name: "gpu".to_string(),
            timings: Timings {
                compile_s: 0.0,
                inference_ms: 0.0,
                latency_ms: 0.0,
                training_ms: 0.0,
            },
            outputs: Outputs {
                logits_hash: "hash".to_string(),
                output_shape: shape,
                logits_sample: vec![1.0; 256],
                loss: 1.0,
                grad_norm: gradients.then_some(1.0),
                gradient_norms: if gradients {
                    BTreeMap::from([("weight".to_string(), 1.0)])
                } else {
                    BTreeMap::new()
                },
            },
            memory: None,
            extra: serde_json::Map::new(),
        }
    }

    #[test]
    fn relative_loss_error_is_not_diluted_for_small_losses() {
        assert!((relative_scalar_error(0.0003, 0.0006) - 0.5).abs() < 1e-12);
        assert_eq!(relative_scalar_error(0.0, 0.0), 0.0);
    }

    #[test]
    fn gradient_comparison_rejects_different_parameter_sets() {
        let reference = BTreeMap::from([("weight".to_string(), 1.0)]);
        let other = BTreeMap::from([("bias".to_string(), 1.0)]);
        assert_eq!(compare_gradient_norms(&reference, &other), f64::INFINITY);
    }

    #[test]
    fn comparison_rejects_shape_mismatch() {
        let reference = result(vec![1, 256], true);
        let other = result(vec![2, 128], true);
        let comparison = compare_result(&reference, &other);
        assert!(!comparison.forward_valid);
        assert!(!comparison.training_valid);
    }

    #[test]
    fn comparison_requires_gradients_only_for_training() {
        let reference = result(vec![1, 256], true);
        let other = result(vec![1, 256], false);
        let comparison = compare_result(&reference, &other);
        assert!(comparison.forward_valid);
        assert!(!comparison.training_valid);
    }

    #[test]
    fn strict_cuda_contract_rejects_tf32_even_when_labeled_strict() {
        let mut candidate = result(vec![1, 256], true);
        candidate.framework = "pytorch".to_string();
        candidate
            .extra
            .insert("backend".to_string(), serde_json::json!("CUDA 13.0"));
        candidate.extra.insert(
            "precision".to_string(),
            serde_json::json!({
                "tensor_storage": "f32",
                "accumulation": "f32",
                "output": "f32",
                "reduced_precision_allowed": false,
                "cuda_matmul_allow_tf32": true,
                "cudnn_allow_tf32": false,
                "nvidia_tf32_override": "0",
            }),
        );
        assert!(precision_contract_error(&candidate, "strict-f32").is_some());
    }

    #[test]
    fn strict_cuda_contract_accepts_explicit_f32_controls() {
        let mut candidate = result(vec![1, 256], true);
        candidate.framework = "pytorch".to_string();
        candidate
            .extra
            .insert("backend".to_string(), serde_json::json!("CUDA 13.0"));
        candidate.extra.insert(
            "precision".to_string(),
            serde_json::json!({
                "tensor_storage": "f32",
                "accumulation": "f32",
                "output": "f32",
                "reduced_precision_allowed": false,
                "cuda_matmul_allow_tf32": false,
                "cudnn_allow_tf32": false,
                "nvidia_tf32_override": "0",
            }),
        );
        assert_eq!(precision_contract_error(&candidate, "strict-f32"), None);
    }
}
