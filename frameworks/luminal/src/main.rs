//! Luminal framework benchmark runner for infermark.
//!
//! Luminal is a graph-based Rust ML framework that compiles computation graphs
//! to optimized GPU kernels. This is a scaffold — full implementation will be
//! added once the luminal crate dependency is stabilized.
//!
//! See: https://github.com/jafioti/luminal

fn main() {
    let model_name = std::env::args().nth(1).unwrap_or("SmolLM2-135M".into());
    eprintln!("[luminal] Benchmark for {model_name} is not yet implemented.");
    eprintln!("[luminal] Luminal uses graph-based compilation — implementation pending.");
    std::process::exit(1);
}
