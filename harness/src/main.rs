use clap::Parser;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Result produced by each framework benchmark runner.
/// Every runner must print exactly one JSON object matching this schema to stdout.
#[derive(Debug, Serialize, Deserialize)]
pub struct BenchResult {
    pub framework: String,
    pub model: String,
    pub device: String,
    pub gpu_name: String,
    pub timings: Timings,
    pub outputs: Outputs,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Timings {
    /// Time to compile/optimize the model (milliseconds).
    pub compile_ms: f64,
    /// Forward pass time (milliseconds).
    pub forward_ms: f64,
    /// Backward pass time (milliseconds).
    pub backward_ms: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Outputs {
    /// SHA-256 hash of the full logits tensor (flattened, f32 little-endian bytes).
    pub logits_hash: String,
    /// First 16 logit values for quick human inspection.
    pub logits_sample: Vec<f64>,
    /// Scalar loss value from the fake training step.
    pub loss: f64,
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

fn project_root(cli_root: Option<&Path>) -> PathBuf {
    if let Some(r) = cli_root {
        return r.to_path_buf();
    }
    // Walk up from the binary location to find Cargo.toml workspace root.
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
    // Fallback: current working directory.
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

fn run_framework(root: &Path, framework: &str, model: &str) -> Option<BenchResult> {
    let fw_dir = root.join("frameworks").join(framework);
    let run_script = fw_dir.join("run.sh");

    if !run_script.exists() {
        eprintln!("[{framework}] run.sh not found at {}, skipping", run_script.display());
        return None;
    }

    eprintln!("[{framework}] running benchmark for {model} ...");

    let output = Command::new("bash")
        .arg(&run_script)
        .arg(model)
        .current_dir(&fw_dir)
        .output();

    let output = match output {
        Ok(o) => o,
        Err(e) => {
            eprintln!("[{framework}] failed to execute run.sh: {e}");
            return None;
        }
    };

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!("[{framework}] run.sh exited with {}: {stderr}", output.status);
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    // The runner may print log lines before the JSON. Find the last line that
    // looks like a JSON object.
    let json_str = stdout
        .lines()
        .rev()
        .find(|l| l.trim_start().starts_with('{'))?;

    match serde_json::from_str::<BenchResult>(json_str) {
        Ok(r) => Some(r),
        Err(e) => {
            eprintln!("[{framework}] failed to parse result JSON: {e}");
            eprintln!("[{framework}] raw output: {json_str}");
            None
        }
    }
}

fn compare_outputs(results: &[BenchResult]) {
    if results.len() < 2 {
        return;
    }
    let reference = &results[0];
    eprintln!();
    eprintln!("=== Output comparison (reference: {}) ===", reference.framework);
    for other in &results[1..] {
        let hash_match = reference.outputs.logits_hash == other.outputs.logits_hash;
        let loss_diff = (reference.outputs.loss - other.outputs.loss).abs();
        let status = if hash_match {
            "EXACT MATCH"
        } else if loss_diff < 1e-4 {
            "CLOSE (loss <1e-4)"
        } else {
            "MISMATCH"
        };
        eprintln!(
            "  {} vs {}: {} (loss diff: {:.6e})",
            reference.framework, other.framework, status, loss_diff
        );
    }
}

fn print_table(results: &[BenchResult]) {
    println!(
        "{:<12} {:<16} {:>12} {:>12} {:>12}  {:>10}  {}",
        "Framework", "Model", "Compile(ms)", "Forward(ms)", "Backward(ms)", "Loss", "GPU"
    );
    println!("{}", "-".repeat(100));
    for r in results {
        println!(
            "{:<12} {:<16} {:>12.2} {:>12.2} {:>12.2}  {:>10.6}  {}",
            r.framework,
            r.model,
            r.timings.compile_ms,
            r.timings.forward_ms,
            r.timings.backward_ms,
            r.outputs.loss,
            r.gpu_name
        );
    }
}

fn main() {
    let cli = Cli::parse();
    let root = project_root(cli.root.as_deref());

    let frameworks: Vec<&str> = match &cli.frameworks {
        Some(list) => list.iter().map(String::as_str).collect(),
        None => ALL_FRAMEWORKS.to_vec(),
    };

    let mut results = Vec::new();
    for fw in &frameworks {
        if let Some(r) = run_framework(&root, fw, &cli.model) {
            results.push(r);
        }
    }

    if results.is_empty() {
        eprintln!("No benchmark results collected.");
        std::process::exit(1);
    }

    if cli.json {
        println!("{}", serde_json::to_string_pretty(&results).unwrap());
    } else {
        print_table(&results);
    }

    compare_outputs(&results);
}
