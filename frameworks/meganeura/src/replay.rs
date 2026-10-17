//! Temporary diagnostic harness; retained with experiment artifacts, not shipped.
use super::*;
use std::{collections::BTreeMap, io::BufRead, path::Path};

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Frozen {
    pub plan: meganeura::compile::ExecutionPlan,
    pub pipelines: Vec<String>,
    pub groups: usize,
    pub chunks: usize,
    pub hashes: BTreeMap<String, (usize, String)>,
}

pub fn hashes(graph: &Graph, session: &Session) -> BTreeMap<String, (usize, String)> {
    let mut tensors = qualification::gradients(session).unwrap();
    for (i, (id, mut values)) in graph
        .outputs()
        .iter()
        .zip(session.read_buffers(&session.plan().output_buffers))
        .enumerate()
    {
        values.truncate(graph.node(*id).ty.num_elements());
        tensors.insert(format!("output:{i}"), values);
    }
    tensors
        .into_iter()
        .map(|(name, values)| {
            assert!(values.iter().all(|x| x.is_finite()), "nonfinite {name}");
            (name, (values.len(), sha256_f32(&values)))
        })
        .collect()
}

pub fn read(path: &Path) -> Frozen {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

pub fn save(path: &Path, graph: &Graph, session: &Session, chunks: usize) {
    let frozen = Frozen {
        plan: session.plan().clone(),
        pipelines: session.dispatch_pipeline_keys(),
        groups: session.num_groups(),
        chunks,
        hashes: hashes(graph, session),
    };
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, serde_json::to_vec(&frozen).unwrap()).unwrap();
    std::fs::rename(tmp, path).unwrap();
}

pub fn interactive(
    phase: &str,
    session: &mut Session,
    set_inputs: &dyn Fn(&mut Session),
) -> BenchStats {
    eprintln!(
        "REPLAY {}",
        serde_json::json!({"event":"ready", "phase":phase})
    );
    let mut times = Vec::new();
    let mut warmup_runs = 0;
    let mut warmup_seconds = 0.0;
    let stdin = std::io::stdin();
    let mut input = stdin.lock();
    loop {
        let mut line = String::new();
        assert!(
            input.read_line(&mut line).unwrap() > 0,
            "controller disconnected"
        );
        let req: serde_json::Value = serde_json::from_str(&line).unwrap();
        match req["command"].as_str().unwrap() {
            "warmup" => {
                let start = Instant::now();
                let mut runs = 0;
                while start.elapsed().as_secs_f64() < req["seconds"].as_f64().unwrap() {
                    set_inputs(session);
                    session.step();
                    session.wait();
                    runs += 1;
                }
                warmup_runs += runs;
                warmup_seconds += start.elapsed().as_secs_f64();
                eprintln!(
                    "REPLAY {}",
                    serde_json::json!({"event":"warm", "phase":phase, "runs":runs})
                );
            }
            "samples" => {
                let mut elapsed = Vec::new();
                let mut encode = Vec::new();
                let mut wait = Vec::new();
                let mut gpu = Vec::new();
                for _ in 0..req["runs"].as_u64().unwrap() {
                    set_inputs(session);
                    let start = Instant::now();
                    session.step();
                    let encoded = start.elapsed();
                    session.wait();
                    let completed = start.elapsed();
                    elapsed.push(completed.as_secs_f64() * 1000.0);
                    encode.push(encoded.as_secs_f64() * 1000.0);
                    wait.push((completed - encoded).as_secs_f64() * 1000.0);
                    gpu.push(
                        session
                            .gpu_timings()
                            .iter()
                            .map(|(_, t)| t.as_secs_f64() * 1000.0)
                            .sum::<f64>(),
                    );
                }
                times.extend_from_slice(&elapsed);
                eprintln!(
                    "REPLAY {}",
                    serde_json::json!({"event":"samples", "phase":phase,"samples_ms":elapsed,"encode_ms":encode,"wait_ms":wait,"gpu_ms":gpu})
                );
            }
            "finish" => break,
            x => panic!("unknown command {x}"),
        }
    }
    BenchStats::from_samples(
        times,
        serde_json::json!({"runs":warmup_runs,"seconds":warmup_seconds,"interleaved":true}),
    )
}
