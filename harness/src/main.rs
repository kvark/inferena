#![allow(clippy::print_literal)]

use clap::Parser;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Command;

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
    /// Framework-specific extra fields (torch_version, driver_name, etc.).
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Timings {
    /// Time to compile/optimize the model (seconds).
    pub compile_s: f64,
    /// Forward pass time (milliseconds).
    pub forward_ms: f64,
    /// Backward pass time (milliseconds).
    pub backward_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Outputs {
    /// SHA-256 hash of the full logits tensor (flattened, f32 little-endian bytes).
    pub logits_hash: String,
    /// First 16 logit values for quick human inspection.
    pub logits_sample: Vec<f64>,
    /// Scalar loss value from the fake training step.
    pub loss: f64,
}

/// Outcome for a framework: either a result or a failure reason.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status")]
pub enum FrameworkOutcome {
    #[serde(rename = "ok")]
    Ok(BenchResult),
    #[serde(rename = "error")]
    Error { framework: String, model: String, error: String },
    #[serde(rename = "skipped")]
    Skipped { framework: String, model: String, reason: String },
}

#[derive(Parser)]
#[command(name = "infermark", about = "ML framework inference benchmark")]
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
}

const ALL_FRAMEWORKS: &[&str] = &["pytorch", "burn", "luminal", "meganeura"];

/// Framework metadata: (display_name, repo_url, git_rev).
/// git_rev must match the pinned revision in the workspace Cargo.toml.
fn framework_meta(name: &str) -> (&'static str, &'static str, &'static str) {
    match name {
        "pytorch" => ("PyTorch", "https://github.com/pytorch/pytorch", ""),
        "burn" => ("Burn", "https://github.com/tracel-ai/burn", "ed72d2b"),
        "luminal" => ("Luminal", "https://github.com/luminal-ai/luminal", "f32161d"),
        "meganeura" => ("Meganeura", "https://github.com/kvark/meganeura", "550bb6c"),
        _ => ("unknown", "", ""),
    }
}

/// Format framework name as a markdown link to the repo at the pinned revision.
fn framework_md_link(name: &str, extra: &serde_json::Map<String, serde_json::Value>) -> String {
    let (display, url, rev) = framework_meta(name);
    if url.is_empty() {
        return display.to_string();
    }
    // For PyTorch, include version from the runner's extra fields.
    if name == "pytorch" {
        let ver = extra
            .get("torch_version")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if !ver.is_empty() {
            return format!("[{display} {ver}]({url})");
        }
    }
    if rev.is_empty() {
        return format!("[{display}]({url})");
    }
    format!("[{display}]({url}/tree/{rev})")
}

fn project_root(cli_root: Option<&Path>) -> PathBuf {
    if let Some(r) = cli_root {
        return std::fs::canonicalize(r).unwrap_or_else(|_| r.to_path_buf());
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

fn run_framework(root: &Path, framework: &str, model: &str) -> FrameworkOutcome {
    let fw_dir = root.join("frameworks").join(framework);
    let run_script = fw_dir.join("run.sh");

    if !run_script.exists() {
        return FrameworkOutcome::Skipped {
            framework: framework.to_string(),
            model: model.to_string(),
            reason: format!("run.sh not found at {}", run_script.display()),
        };
    }

    eprintln!("[{framework}] running benchmark for {model} ...");

    // Always use bash (Git Bash on Windows).
    // Inherits environment so WGPU_BACKEND, HSA_OVERRIDE_GFX_VERSION, etc. propagate.
    let output = Command::new("bash")
        .arg(&run_script)
        .arg(model)
        .current_dir(&fw_dir)
        .output();

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
        // Truncate long error output for readability.
        let stderr_short: String = stderr.lines().take(20).collect::<Vec<_>>().join("\n");
        return FrameworkOutcome::Error {
            framework: framework.to_string(),
            model: model.to_string(),
            error: format!("exit {}: {}", output.status, stderr_short),
        };
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let json_str = match stdout.lines().rev().find(|l| l.trim_start().starts_with('{')) {
        Some(s) => s,
        None => {
            return FrameworkOutcome::Error {
                framework: framework.to_string(),
                model: model.to_string(),
                error: "no JSON found in stdout".to_string(),
            };
        }
    };

    match serde_json::from_str::<BenchResult>(json_str) {
        Ok(r) => FrameworkOutcome::Ok(r),
        Err(e) => FrameworkOutcome::Error {
            framework: framework.to_string(),
            model: model.to_string(),
            error: format!("JSON parse error: {e}"),
        },
    }
}

/// Save each framework's result as a separate JSON file in results/.
fn save_results(root: &Path, model: &str, outcomes: &[FrameworkOutcome]) {
    let results_dir = root.join("results");
    std::fs::create_dir_all(&results_dir).ok();

    for outcome in outcomes {
        let (fw, content) = match outcome {
            FrameworkOutcome::Ok(r) => (&r.framework, serde_json::to_string_pretty(outcome).unwrap()),
            FrameworkOutcome::Error { framework, .. } => (framework, serde_json::to_string_pretty(outcome).unwrap()),
            FrameworkOutcome::Skipped { framework, .. } => (framework, serde_json::to_string_pretty(outcome).unwrap()),
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
    } else {
        0.0
    };
    (max_err, mae, rmse, rel)
}

fn compare_outputs(results: &[&BenchResult]) {
    if results.len() < 2 {
        return;
    }
    let reference = results[0];
    eprintln!();
    eprintln!("=== Output comparison (reference: {}) ===", reference.framework);
    eprintln!(
        "  {:<12} {:>12} {:>12} {:>12} {:>12} {:>12}  {}",
        "Framework", "Loss Diff", "Max Error", "MAE", "RMSE", "Rel Error", "Status"
    );
    eprintln!("  {}", "-".repeat(90));
    for other in &results[1..] {
        let hash_match = reference.outputs.logits_hash == other.outputs.logits_hash;
        let loss_diff = (reference.outputs.loss - other.outputs.loss).abs();
        let (max_err, mae, rmse, rel) = compute_errors(
            &reference.outputs.logits_sample,
            &other.outputs.logits_sample,
        );
        let status = if hash_match {
            "EXACT MATCH"
        } else if loss_diff < 1e-3 && rel < 0.01 {
            "PASS (<1% rel)"
        } else if loss_diff < 0.1 && rel < 0.1 {
            "CLOSE"
        } else {
            "DIFFERENT MODEL"
        };
        eprintln!(
            "  {:<12} {:>12.6e} {:>12.6e} {:>12.6e} {:>12.6e} {:>12.6e}  {}",
            other.framework, loss_diff, max_err, mae, rmse, rel, status
        );
    }
}

/// Determine which frameworks "match" the reference (first successful result).
/// Returns a set of framework names that passed correctness checks.
fn matching_frameworks(successes: &[&BenchResult]) -> std::collections::HashSet<String> {
    let mut matching = std::collections::HashSet::new();
    if successes.is_empty() {
        return matching;
    }
    let reference = successes[0];
    matching.insert(reference.framework.clone());
    for other in &successes[1..] {
        let loss_diff = (reference.outputs.loss - other.outputs.loss).abs();
        let (_, _, _, rel) = compute_errors(
            &reference.outputs.logits_sample,
            &other.outputs.logits_sample,
        );
        if loss_diff < 0.1 && rel < 0.1 {
            matching.insert(other.framework.clone());
        }
    }
    matching
}

fn print_table(outcomes: &[FrameworkOutcome], successes: &[&BenchResult]) {
    let matching = matching_frameworks(successes);

    // Find best compile/forward/backward among matching frameworks.
    let mut best_compile = f64::MAX;
    let mut best_forward = f64::MAX;
    let mut best_backward = f64::MAX;
    for o in outcomes {
        if let FrameworkOutcome::Ok(r) = o && matching.contains(&r.framework) {
            if r.timings.compile_s < best_compile { best_compile = r.timings.compile_s; }
            if r.timings.forward_ms < best_forward { best_forward = r.timings.forward_ms; }
            if r.timings.backward_ms > 0.0 && r.timings.backward_ms < best_backward {
                best_backward = r.timings.backward_ms;
            }
        }
    }

    println!("| Framework | Compile (s) | Forward (ms) | Backward (ms) | Loss |");
    println!("|-----------|:-----------:|:------------:|:--------------:|:----:|");

    for outcome in outcomes {
        match outcome {
            FrameworkOutcome::Ok(r) => {
                let link = framework_md_link(&r.framework, &r.extra);
                let is_matching = matching.contains(&r.framework);

                let fmt_val = |val: f64, best: f64, is_time: bool| -> String {
                    let s = if is_time && val == 0.0 {
                        "—".to_string()
                    } else if is_time {
                        format!("{:.0}", val)
                    } else {
                        format!("{:.2}", val)
                    };
                    if !is_matching {
                        format!("~~{s}~~")
                    } else if val > 0.0 && (val - best).abs() < 0.01 {
                        format!("**{s}**")
                    } else {
                        s
                    }
                };

                let compile = fmt_val(r.timings.compile_s, best_compile, false);
                let forward = fmt_val(r.timings.forward_ms, best_forward, true);
                let backward = fmt_val(r.timings.backward_ms, best_backward, true);
                let loss = if is_matching {
                    format!("{:.2}", r.outputs.loss)
                } else {
                    format!("~~{:.2}~~", r.outputs.loss)
                };

                println!("| {link} | {compile} | {forward} | {backward} | {loss} |");
            }
            FrameworkOutcome::Error { framework, .. } => {
                let (display, url, rev) = framework_meta(framework);
                let link = if !url.is_empty() && !rev.is_empty() {
                    format!("[{display}]({url}/tree/{rev})")
                } else {
                    display.to_string()
                };
                println!("| {link} | ✗ | ✗ | ✗ | |");
            }
            FrameworkOutcome::Skipped { framework, .. } => {
                let (display, ..) = framework_meta(framework);
                println!("| {display} | — | — | — | |");
            }
        }
    }
}

fn main() {
    let cli = Cli::parse();
    let root = project_root(cli.root.as_deref());

    let frameworks: Vec<&str> = match &cli.frameworks {
        Some(list) => list.iter().map(String::as_str).collect(),
        None => ALL_FRAMEWORKS.to_vec(),
    };

    let mut outcomes = Vec::new();
    for fw in &frameworks {
        outcomes.push(run_framework(&root, fw, &cli.model));
    }

    // Collect successful results for comparison.
    let successes: Vec<&BenchResult> = outcomes
        .iter()
        .filter_map(|o| match o {
            FrameworkOutcome::Ok(r) => Some(r),
            _ => None,
        })
        .collect();

    if cli.json {
        println!("{}", serde_json::to_string_pretty(&outcomes).unwrap());
    } else {
        print_table(&outcomes, &successes);
    }

    if successes.is_empty() {
        eprintln!("No successful benchmark results.");
        std::process::exit(1);
    }

    compare_outputs(&successes);
    save_results(&root, &cli.model, &outcomes);
}
