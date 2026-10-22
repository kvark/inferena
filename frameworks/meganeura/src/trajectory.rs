//! Opt-in accelerated training experiment using the shared PyTorch SGD gate.
use meganeura::{Mode, Session};
use serde_json::{Value, json};
use std::{
    io::Write,
    process::{Command, Stdio},
    time::Instant,
};

pub struct Workload {
    description: Value,
    train: Vec<Vec<u32>>,
    probe: Vec<Vec<u32>>,
    heldout: Vec<Vec<u32>>,
}

fn shared_gate(request: Value) -> Value {
    let script =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/training_gate.py");
    let mut child = Command::new("python3")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("cannot start shared training gate");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(request.to_string().as_bytes())
        .unwrap();
    let output = child
        .wait_with_output()
        .expect("training gate did not finish");
    assert!(output.status.success(), "shared training gate failed");
    serde_json::from_slice(&output.stdout).expect("invalid shared training gate response")
}

impl Workload {
    pub fn from_env(mode: Mode) -> Option<Self> {
        let policy = std::env::var("INFERENA_MEGANEURA_TRAINING_GATE")
            .unwrap_or_else(|_| "full-gradient".into());
        assert!(
            matches!(policy.as_str(), "full-gradient" | "trajectory"),
            "INFERENA_MEGANEURA_TRAINING_GATE must be full-gradient or trajectory"
        );
        if policy == "full-gradient" {
            return None;
        }
        assert!(
            !crate::precision::strict(),
            "strict mode forbids a relaxed training gate"
        );
        if mode != Mode::Training {
            return None;
        }
        let model = std::env::args()
            .nth(1)
            .unwrap_or_else(|| "SmolLM2-135M".into());
        assert!(
            !model.starts_with("--"),
            "trajectory gate requires a benchmark model"
        );
        let plan = shared_gate(json!({"operation": "workload", "model": model}));
        let batches = |key| {
            if plan["fixture"].is_null() {
                Vec::new()
            } else {
                serde_json::from_value(plan["fixture"][key].clone())
                    .expect("invalid frozen token batches")
            }
        };
        Some(Self {
            description: plan["workload"].clone(),
            train: batches("train"),
            probe: batches("probe"),
            heldout: batches("heldout"),
        })
    }

    fn bind(session: &mut Session, tokens: &[u32]) {
        session.set_input_u32("token_ids", &tokens[..tokens.len() - 1]);
        session.set_input_u32("labels", &tokens[1..]);
    }

    fn evaluate(session: &mut Session, batches: &[Vec<u32>]) -> Vec<f64> {
        session.clear_optimizer();
        batches
            .iter()
            .map(|tokens| {
                Self::bind(session, tokens);
                session.step();
                session.wait();
                f64::from(session.read_loss())
            })
            .collect()
    }

    pub fn run(
        &self,
        session: &mut Session,
        initialize_inputs: &impl Fn(&mut Session),
        label: &str,
    ) -> Value {
        let started = Instant::now();
        let steps = self.description["steps"].as_u64().unwrap() as usize;
        let learning_rate = self.description["learning_rate"].as_f64().unwrap();
        let real_text = !self.train.is_empty();
        let checkpoint_every = self.description["checkpoint_every"].as_u64().unwrap_or(32) as usize;
        if real_text {
            assert_eq!(self.train.len(), steps);
            assert_eq!(self.probe.len(), 4);
            assert_eq!(self.heldout.len(), 4);
            assert!(
                self.train
                    .iter()
                    .chain(&self.probe)
                    .chain(&self.heldout)
                    .all(|tokens| tokens.len() == 129)
            );
        }
        let mut initial =
            vec![0.0; self.description["parameter_elements"].as_u64().unwrap_or(0) as usize];
        if real_text {
            session.read_param("model.norm.weight", &mut initial);
        }
        let mut losses = Vec::new();
        let mut attempts = Vec::new();
        let mut checkpoints = Vec::new();
        for step in 0..=steps {
            if real_text && step % checkpoint_every == 0 {
                checkpoints.push(json!({"step": step,
                    "probe_losses": Self::evaluate(session, &self.probe),
                    "heldout_losses": Self::evaluate(session, &self.heldout)}));
            }
            if step % checkpoint_every == 0 {
                eprintln!("[meganeura] {label} SGD trajectory: {step}/{steps}");
            }
            if step == steps {
                break;
            }
            initialize_inputs(session);
            if real_text {
                Self::bind(session, &self.train[step]);
            }
            session.set_learning_rate(learning_rate as f32);
            session.step();
            session.wait();
            let loss = f64::from(session.read_loss());
            // Non-finite values are rejected by the common evaluator; preserve
            // the receipt rather than reporting timings from a partial run.
            losses.push(loss);
            attempts.push(json!([{"scale": 1.0, "loss": loss, "updated": true}]));
            if !loss.is_finite() {
                break;
            }
        }
        session.clear_optimizer();
        let mut record = json!({
            "optimizer": "SGD", "learning_rate": learning_rate,
            "completed_updates": losses.len(), "skipped_steps": 0,
            "losses": losses, "scale_attempts": attempts, "overflow_attempts": 0,
            "elapsed_seconds": started.elapsed().as_secs_f64(),
        });
        if real_text {
            let mut final_values = vec![0.0; initial.len()];
            session.read_param("model.norm.weight", &mut final_values);
            record["checkpoints"] = json!(checkpoints);
            record["parameter_check"] = json!({"name": "model.norm.weight",
                "changed": initial != final_values, "initial": initial, "final": final_values});
        }
        record
    }

    pub fn assess(&self, actual: Value, reference: Value) -> Value {
        let request = json!({"operation": "assess", "actual": actual,
            "reference": reference, "workload": self.description});
        // Retain rejected evidence too. The normal benchmark only emits timings
        // after a successful assessment and restoration of the initial weights.
        let receipt =
            std::env::var_os("INFERENA_TRAINING_GATE_RECEIPT").map(std::path::PathBuf::from);
        if let Some(path) = &receipt {
            std::fs::write(path, serde_json::to_vec_pretty(&request).unwrap())
                .expect("cannot save SGD evidence");
        }
        let assessment = shared_gate(request.clone());
        let record = json!({"policy": "native-f32-sgd-trajectory-v1", "assessment": assessment,
            "workload": self.description, "actual": request["actual"], "reference": request["reference"]});
        if let Some(path) = &receipt {
            std::fs::write(path, serde_json::to_vec_pretty(&record).unwrap())
                .expect("cannot save SGD assessment");
        }
        assert_eq!(
            record["assessment"]["status"], "pass",
            "selected plan failed shared SGD trajectory gate"
        );
        record
    }
}
