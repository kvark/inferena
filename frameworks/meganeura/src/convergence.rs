//! Real text, actual SGD updates, and durable loss trajectories. No timing claims.

use super::{Graph, Mode, SafeTensorsModel, Session, build_session, load_weights};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs::File, io::Write, path::Path, time::Instant};

#[derive(Deserialize)]
struct Spec {
    protocol: String,
    model: String,
    sequence_length: usize,
    steps: usize,
    learning_rate: f32,
    checkpoint_every: usize,
    train: Vec<Vec<u32>>,
    probe: Vec<Vec<u32>>,
    heldout: Vec<Vec<u32>>,
    input_sha256: BTreeMap<String, String>,
}

fn save(path: &Path, value: &serde_json::Value) {
    let temporary = path.with_extension("tmp");
    let mut file = File::create(&temporary).expect("cannot create convergence receipt");
    serde_json::to_writer_pretty(&mut file, value).unwrap();
    file.write_all(b"\n").unwrap();
    file.sync_all().unwrap();
    std::fs::rename(temporary, path).expect("cannot replace convergence receipt");
    #[cfg(unix)]
    File::open(path.parent().unwrap())
        .unwrap()
        .sync_all()
        .unwrap();
}

fn bind(session: &mut Session, tokens: &[u32]) {
    session.set_input_u32("token_ids", &tokens[..tokens.len() - 1]);
    session.set_input_u32("labels", &tokens[1..]);
}

fn evaluate(session: &mut Session, batches: &[Vec<u32>]) -> Vec<f32> {
    session.clear_optimizer();
    batches
        .iter()
        .map(|tokens| {
            bind(session, tokens);
            session.step();
            session.wait();
            let loss = session.read_loss();
            assert!(loss.is_finite(), "non-finite evaluation loss");
            loss
        })
        .collect()
}

pub fn run(spec_path: &Path, output: &Path) {
    let bytes = std::fs::read(spec_path).expect("missing convergence spec");
    let spec: Spec = serde_json::from_slice(&bytes).expect("invalid convergence spec");
    assert_eq!(spec.protocol, "smollm2-text-v3");
    assert!(matches!(
        spec.model.as_str(),
        "SmolLM2-135M" | "SmolLM2-360M" | "SmolLM2-1.7B"
    ));
    assert_eq!(spec.steps, spec.train.len());
    assert!(spec.steps > 0 && spec.checkpoint_every > 0);
    assert!(spec.learning_rate.is_finite() && spec.learning_rate > 0.0);
    assert!(!spec.probe.is_empty() && !spec.heldout.is_empty());
    for (path, expected) in &spec.input_sha256 {
        let mut file = File::open(path).expect("missing checkpoint file");
        let mut hash = Sha256::new();
        std::io::copy(&mut file, &mut hash).unwrap();
        assert_eq!(hex::encode(hash.finalize()), *expected, "changed {path}");
    }
    let path = super::find_local_model(&spec.model).expect("prepare pinned SmolLM2 weights first");
    let config = super::smollm2_config(&path);
    for tokens in spec.train.iter().chain(&spec.probe).chain(&spec.heldout) {
        assert_eq!(tokens.len(), spec.sequence_length + 1);
        assert!(
            tokens
                .iter()
                .all(|&token| (token as usize) < config.vocab_size)
        );
    }
    let model = SafeTensorsModel::load(path).unwrap();
    let transposed = meganeura::models::smollm2::transposed_weight_names(&config);
    let transposed_set = transposed.iter().map(String::as_str).collect();
    let mut graph = Graph::new();
    let logits = meganeura::models::smollm2::build_graph(&mut graph, &config, spec.sequence_length);
    let labels = graph.input_u32("labels", &[spec.sequence_length]);
    let loss = graph.cross_entropy_loss_indices(logits, labels);
    graph.set_outputs(vec![loss]);
    let started = Instant::now();
    let mut session = build_session(
        &graph,
        Mode::Training,
        |session, donor| load_weights(session, &model, &transposed_set, donor),
        |session| bind(session, &spec.train[0]),
    );
    // Search is untimed preparation. Restore the checkpoint explicitly before
    // the first evaluation or optimizer update.
    load_weights(&mut session, &model, &transposed_set, None);
    let parameter = "model.norm.weight";
    let mut initial = vec![0.0; config.hidden_size];
    session.read_param(parameter, &mut initial);
    let strict = std::env::var("INFERENA_STRICT").as_deref() == Ok("1");
    let preparation_path = output.with_file_name("native-preparation.json");
    save(
        &preparation_path,
        &super::SESSION_PREPARATION.with_borrow(|reports| json!(reports)),
    );
    let mut record = json!({
        "protocol": spec.protocol, "status": "running", "framework": "meganeura",
        "model": spec.model, "precision": if strict { "strict-f32" } else { "accelerated" },
        "spec_sha256": hex::encode(Sha256::digest(&bytes)),
        "environment": super::environment_json(&session),
        "arithmetic": super::precision::Policy::from_env().metadata(),
        "preparation_seconds": started.elapsed().as_secs_f64(),
        "preparation": "native-preparation.json",
        "optimizer": {"name": "SGD", "learning_rate": spec.learning_rate, "momentum": 0, "weight_decay": 0},
        "training_scope": "forward + f32 loss + backward + parameter update",
        "losses": [], "checkpoints": [], "completed_updates": 0, "skipped_updates": 0,
        "parameter_check": {"name": parameter, "initial": initial},
    });
    save(output, &record);
    for step in 0..=spec.steps {
        if step % spec.checkpoint_every == 0 || step == spec.steps {
            let probe = evaluate(&mut session, &spec.probe);
            let heldout = evaluate(&mut session, &spec.heldout);
            record["checkpoints"].as_array_mut().unwrap().push(json!({
                "step": step, "probe_losses": probe, "heldout_losses": heldout,
            }));
            save(output, &record);
        }
        if step == spec.steps {
            break;
        }
        session.set_learning_rate(spec.learning_rate);
        bind(&mut session, &spec.train[step]);
        session.step();
        session.wait();
        let loss = session.read_loss();
        assert!(loss.is_finite(), "non-finite training loss at step {step}");
        record["losses"].as_array_mut().unwrap().push(json!(loss));
        record["completed_updates"] = json!(step + 1);
        save(output, &record);
        if (step + 1) % spec.checkpoint_every == 0 {
            eprintln!("[meganeura] SGD step {}: loss={loss:.6}", step + 1);
        }
    }
    let mut final_values = vec![0.0; initial.len()];
    session.read_param(parameter, &mut final_values);
    assert!(final_values.iter().all(|value| value.is_finite()));
    assert_ne!(
        initial, final_values,
        "SGD did not update the checked parameter"
    );
    record["parameter_check"]["final"] = json!(final_values);
    record["parameter_check"]["changed"] = json!(true);
    record["status"] = json!("complete");
    record["elapsed_seconds"] = json!(started.elapsed().as_secs_f64());
    save(output, &record);
    println!("{record}");
}
