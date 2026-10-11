//! Campaign-local kernel decisions, never saved tensors or measured programs.
use meganeura::{runtime::KernelMemo, train::BuildSearchReport};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    io,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

pub const POLICY: &str = "campaign-kernel-memo-v1";

pub struct Cache {
    pub memo: Arc<Mutex<KernelMemo>>,
    path: Option<PathBuf>,
    key: String,
    loaded: &'static str,
    loaded_classes: usize,
}

impl Cache {
    pub fn open(directory: Option<&Path>, identity: Value) -> io::Result<Self> {
        // The library checks device/driver/decision policy on use. Also isolate
        // dependency revisions: a memo must never survive a changed compiler.
        let identity = json!({
            "policy": POLICY,
            "dependencies": hex::encode(Sha256::digest(include_bytes!("../../../Cargo.lock"))),
            "target": identity,
        });
        let key = hex::encode(Sha256::digest(serde_json::to_vec(&identity)?));
        let path = directory.map(|dir| dir.join(format!("{key}.json")));
        let (memo, loaded) = match path.as_ref().map(KernelMemo::load) {
            None => (KernelMemo::default(), "disabled"),
            Some(Ok(memo)) => (memo, "hit"),
            Some(Err(error)) if error.kind() == io::ErrorKind::NotFound => {
                (KernelMemo::default(), "miss")
            }
            Some(Err(error))
                if matches!(
                    error.kind(),
                    io::ErrorKind::InvalidData | io::ErrorKind::UnexpectedEof
                ) =>
            {
                eprintln!("[meganeura] ignoring invalid kernel memo: {error}");
                (KernelMemo::default(), "invalid")
            }
            Some(Err(error)) => return Err(error),
        };
        let loaded_classes = memo.len();
        Ok(Self {
            memo: Arc::new(Mutex::new(memo)),
            path,
            key,
            loaded,
            loaded_classes,
        })
    }

    /// Publish only after construction and full-tensor qualification succeed.
    pub fn finish(&self, report: &BuildSearchReport) -> io::Result<Value> {
        let memo = self.memo.lock().unwrap();
        if let Some(ref path) = self.path {
            std::fs::create_dir_all(path.parent().unwrap())?;
            memo.save(path)?;
        }
        let tuning = || {
            report
                .trials
                .iter()
                .filter_map(|trial| trial.kernel_tuning.as_ref())
        };
        Ok(json!({
            "policy": POLICY,
            "enabled": self.path.is_some(),
            "key": self.path.as_ref().map(|_| &self.key),
            "load": self.loaded,
            "loaded_classes": self.loaded_classes,
            "saved_classes": self.path.as_ref().map(|_| memo.len()),
            // Includes reuse between graph forms within this search, not just
            // decisions loaded from an earlier process.
            "reused_class_visits": tuning().map(|r| r.reused_classes.len()).sum::<usize>(),
            "new_comparisons": tuning().map(|r| r.outcomes.len()).sum::<usize>(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persists_only_after_success_and_scopes_the_identity() {
        let directory = tempfile::tempdir().unwrap();
        let target = json!({"device": "test GPU", "coop": "Auto", "policy": 1});
        let cache = Cache::open(Some(directory.path()), target.clone()).unwrap();
        assert_eq!(cache.loaded, "miss");
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
        let receipt = cache.finish(&BuildSearchReport::default()).unwrap();
        assert_eq!(receipt["saved_classes"], 0);
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
        let restored = Cache::open(Some(directory.path()), target.clone()).unwrap();
        assert_eq!(restored.loaded, "hit");
        assert!(restored.memo.lock().unwrap().is_empty());
        restored.finish(&BuildSearchReport::default()).unwrap();
        for identity in [
            json!({"device": "another GPU"}),
            json!({"coop": "NativeF32"}),
        ] {
            let different = Cache::open(Some(directory.path()), identity).unwrap();
            assert_eq!(different.loaded, "miss");
            assert_ne!(different.key, cache.key);
        }
        let disabled = Cache::open(None, target).unwrap();
        let receipt = disabled.finish(&BuildSearchReport::default()).unwrap();
        assert_eq!(receipt["load"], "disabled");
        assert!(receipt["key"].is_null() && receipt["saved_classes"].is_null());
    }

    #[test]
    fn corrupt_memo_is_a_miss_without_removing_the_old_file() {
        let directory = tempfile::tempdir().unwrap();
        let target = json!({"device": "test"});
        let cache = Cache::open(Some(directory.path()), target.clone()).unwrap();
        let path = cache.path.as_ref().unwrap();
        for invalid in ["not JSON", "{\"scope\":"] {
            std::fs::write(path, invalid).unwrap();
            let restored = Cache::open(Some(directory.path()), target.clone()).unwrap();
            assert_eq!(restored.loaded, "invalid");
            assert_eq!(std::fs::read_to_string(path).unwrap(), invalid);
            restored.finish(&BuildSearchReport::default()).unwrap();
        }
        assert_eq!(
            Cache::open(Some(directory.path()), target).unwrap().loaded,
            "hit"
        );
    }

    #[test]
    #[ignore = "GPU regression: three fresh sessions, saved kernel decisions and fresh data"]
    fn saved_decisions_reuse_kernels_but_recheck_fresh_data() {
        use meganeura::{CoopPolicy, Graph, Mode, SessionConfig, train, tune::TuneOptions};
        use std::time::Duration;

        let directory = tempfile::tempdir().unwrap();
        let gpu = Arc::new(meganeura::runtime::init_gpu_context().unwrap());
        let mut graph = Graph::new();
        let x = graph.input("x", &[33, 17]);
        let w = graph.parameter("w", &[17, 65]);
        let y = graph.matmul(x, w);
        let loss = graph.mean_all(y);
        graph.set_outputs(vec![loss]);
        let options = train::BuildSearchOptions {
            max_graphs: 1,
            // The ordinary fallback and its tuned challenger each need a slot.
            max_programs: 2,
            min_improvement: 0.99,
            max_time: Duration::from_secs(30),
            tuning: TuneOptions {
                max_time: Duration::from_secs(30),
                sample_pairs: 4,
                dispatches_per_sample: 16,
                target_sample_time: Some(Duration::from_nanos(1)),
                ..Default::default()
            },
            ..Default::default()
        };
        for mode in [Mode::Inference, Mode::Training] {
            let identity = json!({"device": gpu.device_information().device_name, "mode": format!("{mode:?}")});
            for run in 0..3 {
                let cache = Cache::open(Some(directory.path()), identity.clone()).unwrap();
                let mut config = SessionConfig {
                    mode,
                    gpu: Some(gpu.clone()),
                    ..Default::default()
                };
                config.runtime.coop = CoopPolicy::Disabled;
                let mut qualification = crate::qualification::Qualification::new(&graph, mode);
                // Changing both tensors proves the memo restores decisions,
                // never old weights, inputs or a qualification result.
                let input = (run + 1) as f32 * 0.25;
                let weight = (run + 1) as f32 * 0.125;
                let (session, report) = train::build_measured(
                    &graph,
                    config,
                    train::BuildSearchOptions {
                        kernel_memo: Some(cache.memo.clone()),
                        ..options.clone()
                    },
                    |session, _| {
                        session.set_input("x", &[input; 33 * 17]);
                        session.set_parameter("w", &[weight; 17 * 65]);
                        Ok(())
                    },
                    |session| {
                        qualification.check(session)?;
                        assert!((session.read_output(1)[0] - 17.0 * input * weight).abs() < 1e-6);
                        if mode == Mode::Training {
                            let mut gradient = [0.0; 17 * 65];
                            session.read_param_grad("w", &mut gradient);
                            assert!(gradient.iter().all(|g| (*g - input / 65.0).abs() < 1e-6));
                        }
                        Ok(())
                    },
                )
                .unwrap();
                let receipt = cache.finish(&report).unwrap();
                assert!(!cache.memo.lock().unwrap().is_empty());
                assert_eq!(receipt["load"], if run == 0 { "miss" } else { "hit" });
                // A memo contains private decisions, even when whole-program
                // timing retains the untuned fallback instead of installing them.
                let tuning = report
                    .trials
                    .iter()
                    .find_map(|trial| trial.kernel_tuning.as_ref())
                    .unwrap();
                assert!(!tuning.time_budget_exhausted);
                if run == 0 {
                    assert!(!tuning.outcomes.is_empty());
                } else {
                    assert!(tuning.outcomes.is_empty());
                    assert!(!tuning.reused_classes.is_empty());
                    assert_eq!(receipt["new_comparisons"], 0);
                }
                assert!(report.trials.iter().all(|trial| trial.outcome.qualified));
                assert_eq!(
                    qualification.report()["qualified_calls"],
                    report
                        .trials
                        .iter()
                        .map(|trial| trial.qualifications)
                        .sum::<u32>()
                        + u32::from(!report.final_qualification_time.is_zero())
                );
                eprintln!("{mode:?} run {}: {receipt}", run + 1);
                drop(session);
            }
        }
    }
}
