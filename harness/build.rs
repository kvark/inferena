//! Extract framework git revisions from the workspace Cargo.toml at build time.
//! This ensures the harness always uses the same revisions as the actual dependencies.

use std::path::Path;

fn main() {
    let workspace_toml = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("Cargo.toml");

    println!("cargo::rerun-if-changed={}", workspace_toml.display());

    let content = std::fs::read_to_string(&workspace_toml)
        .expect("failed to read workspace Cargo.toml");
    let doc: toml::Value = content.parse().expect("failed to parse workspace Cargo.toml");

    let deps = doc
        .get("workspace")
        .and_then(|w| w.get("dependencies"))
        .expect("no [workspace.dependencies]");

    // Extract short rev (first 7 chars) for each framework.
    for (env_name, dep_key) in [
        ("BURN_REV", "burn"),
        ("CANDLE_REV", "candle-core"),
        ("LUMINAL_REV", "luminal"),
        ("MEGANEURA_REV", "meganeura"),
    ] {
        let rev = deps
            .get(dep_key)
            .and_then(|d| d.get("rev"))
            .and_then(|r| r.as_str())
            .unwrap_or("");
        let short = if rev.len() >= 7 { &rev[..7] } else { rev };
        println!("cargo::rustc-env={env_name}={short}");
    }
}
