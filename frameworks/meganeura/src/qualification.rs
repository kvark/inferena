//! Independent F32 correctness and same-program repeatability checks.
//! The outer harness remains the independent PyTorch implementation check.
use meganeura::{Graph, Mode, Session};
use std::collections::BTreeMap;

pub use crate::parameters::{gradients, source_parameters};

#[derive(Default)]
struct Error {
    elements: usize,
    max_error: f64,
    max_reference: f64,
    square_error: f64,
    square_reference: f64,
}

impl Error {
    fn merge(&mut self, other: &Self) {
        self.elements += other.elements;
        self.max_error = self.max_error.max(other.max_error);
        self.max_reference = self.max_reference.max(other.max_reference);
        self.square_error += other.square_error;
        self.square_reference += other.square_reference;
    }

    fn include(&mut self, actual: &[f32], reference: &[f32]) -> Result<(), String> {
        if actual.is_empty() || actual.len() != reference.len() {
            return Err("tensor shape changed or empty".into());
        }
        // Independent f64 accumulators let the CPU vectorize this full-tensor scan.
        let mut lanes: [Self; 8] = std::array::from_fn(|_| Self::default());
        let mut finite = [true; 8];
        let (actual_chunks, actual_tail) = actual.as_chunks::<8>();
        let (reference_chunks, reference_tail) = reference.as_chunks::<8>();
        for (actual, reference) in actual_chunks.iter().zip(reference_chunks) {
            for i in 0..8 {
                let (a, b) = (f64::from(actual[i]), f64::from(reference[i]));
                finite[i] &= a.is_finite() & b.is_finite();
                lanes[i].max_error = lanes[i].max_error.max((a - b).abs());
                lanes[i].max_reference = lanes[i].max_reference.max(b.abs());
                lanes[i].square_error += (a - b).powi(2);
                lanes[i].square_reference += b * b;
            }
        }
        for (lane, finite) in lanes.iter().zip(finite) {
            if !finite {
                return Err("non-finite tensor".into());
            }
            self.merge(lane);
        }
        for (&a, &b) in actual_tail.iter().zip(reference_tail) {
            if !a.is_finite() || !b.is_finite() {
                return Err("non-finite tensor".into());
            }
            let (a, b) = (f64::from(a), f64::from(b));
            self.max_error = self.max_error.max((a - b).abs());
            self.max_reference = self.max_reference.max(b.abs());
            self.square_error += (a - b).powi(2);
            self.square_reference += b * b;
        }
        self.elements += actual.len();
        Ok(())
    }

    fn check(&self, rtol: f64) -> Result<(), String> {
        let rms_error = (self.square_error / self.elements as f64).sqrt();
        let rms_reference = (self.square_reference / self.elements as f64).sqrt();
        if self.elements == 0
            || self.max_error > 1e-6 + rtol * self.max_reference
            || rms_error > 1e-6 + rtol * rms_reference
        {
            return Err(format!(
                "fixed tensor bounds exceeded: max={:.3e} / {:.3e}, RMS={:.3e} / {:.3e}",
                self.max_error,
                1e-6 + rtol * self.max_reference,
                rms_error,
                1e-6 + rtol * rms_reference
            ));
        }
        Ok(())
    }
}

#[derive(PartialEq)]
struct Snapshot {
    outputs: Vec<Vec<f32>>,
    gradients: BTreeMap<String, Vec<f32>>,
}

struct CheckedSnapshot {
    tensors: Snapshot,
    quality: Option<serde_json::Value>,
    raw_logits: Option<serde_json::Value>,
}

impl Snapshot {
    fn read(session: &Session, output_lengths: &[usize], mode: Mode) -> Result<Self, String> {
        if session.num_outputs() != output_lengths.len() {
            return Err("output inventory changed".into());
        }
        let mut outputs = session.read_buffers(&session.plan().output_buffers);
        for (values, &len) in outputs.iter_mut().zip(output_lengths) {
            if values.len() < len {
                return Err("output buffer is smaller than its logical tensor".into());
            }
            values.truncate(len);
        }
        let gradients = if mode == Mode::Training {
            gradients(session)?
        } else {
            BTreeMap::new()
        };
        Ok(Self { outputs, gradients })
    }

    fn compare(
        &self,
        reference: &Self,
        mode: Mode,
        output_rtol: f64,
        accelerated: bool,
    ) -> Result<(), String> {
        if self.outputs.len() != reference.outputs.len()
            || self.gradients.keys().ne(reference.gradients.keys())
        {
            return Err("tensor inventory changed".into());
        }
        for (index, (actual, expected)) in self.outputs.iter().zip(&reference.outputs).enumerate() {
            let mut error = Error::default();
            error.include(actual, expected)?;
            error
                .check(output_rtol)
                .map_err(|error| format!("output {index}: {error}"))?;
        }
        let mut total = Error::default();
        for (name, actual) in &self.gradients {
            let mut error = Error::default();
            error.include(actual, &reference.gradients[name])?;
            if !accelerated {
                error
                    .check(1e-4)
                    .map_err(|error| format!("gradient {name}: {error}"))?;
            }
            total.merge(&error);
        }
        if mode == Mode::Training {
            total
                .check(if accelerated { 0.01 } else { 1e-4 })
                .map_err(|error| format!("full gradient: {error}"))?;
        }
        Ok(())
    }
}

pub struct Qualification {
    output_lengths: Vec<usize>,
    mode: Mode,
    accelerated: bool,
    f32: Snapshot,
    reference_seconds: f64,
    anchor: Option<CheckedSnapshot>,
    anchor_plan: Option<meganeura::compile::ExecutionPlan>,
    program_checks: usize,
    repeatability_checks: usize,
    language_model: Option<crate::quality::LanguageModel>,
    last_quality: Option<serde_json::Value>,
    last_raw_logits: Option<serde_json::Value>,
    calls: usize,
    rejected_calls: usize,
}

impl Qualification {
    /// The F32 reference is built from the original graph, before any precision
    /// conversion or search. Never let a reduced candidate define this oracle.
    pub fn new(graph: &Graph, mode: Mode, reference: &Session) -> Result<Self, String> {
        let output_lengths: Vec<_> = graph
            .outputs()
            .iter()
            .map(|&id| graph.node(id).ty.num_elements())
            .collect();
        let f32 = Snapshot::read(reference, &output_lengths, mode)?;
        f32.compare(&f32, mode, 1e-4, false)?;
        Ok(Self {
            output_lengths,
            mode,
            accelerated: !crate::precision::strict(),
            f32,
            reference_seconds: 0.0,
            anchor: None,
            anchor_plan: None,
            program_checks: 0,
            repeatability_checks: 0,
            language_model: None,
            last_quality: None,
            last_raw_logits: None,
            calls: 0,
            rejected_calls: 0,
        })
    }

    pub fn set_reference_seconds(&mut self, seconds: f64) {
        self.reference_seconds = seconds;
    }

    pub fn set_language_model(&mut self, vocabulary: usize, targets: &[u32]) -> Result<(), String> {
        if self.anchor.is_some() {
            return Err("language-model policy must be set before qualification".into());
        }
        if self.mode != Mode::Inference || self.f32.outputs.len() != 1 {
            return Err("language-model quality requires one inference output".into());
        }
        if self.accelerated {
            self.language_model = Some(crate::quality::LanguageModel::new(
                &self.f32.outputs[0],
                vocabulary,
                targets,
            )?);
        }
        Ok(())
    }

    pub fn check(&mut self, session: &Session) -> Result<(), String> {
        self.begin_program(session.plan());
        self.check_snapshot(Snapshot::read(session, &self.output_lengths, self.mode)?)
    }

    fn begin_program(&mut self, plan: &meganeura::compile::ExecutionPlan) {
        if self.anchor_plan.as_ref() != Some(plan) {
            self.anchor_plan = Some(plan.clone());
            self.anchor = None;
            self.program_checks = 0;
        }
    }

    fn check_snapshot(&mut self, actual: Snapshot) -> Result<(), String> {
        let result = (|| {
            // An identical snapshot already passed the independent F32 gate.
            // Compare every output and gradient element, but avoid repeating
            // the expensive error reductions and language-model softmax.
            if let Some(anchor) = self.anchor.as_ref().filter(|a| a.tensors == actual) {
                self.last_quality.clone_from(&anchor.quality);
                self.last_raw_logits.clone_from(&anchor.raw_logits);
                self.repeatability_checks += 1;
                self.program_checks += 1;
                return Ok(());
            }
            let mut raw_logits = None;
            let quality = if let Some(language_model) = &self.language_model {
                if actual.outputs.len() != 1 {
                    return Err("language-model output inventory changed".into());
                }
                let report = language_model.evaluate(&actual.outputs[0])?;
                if report["status"] != "pass" {
                    return Err(format!(
                        "language-model quality: mean KL={}, max KL={}, loss error={}",
                        report["mean_kl"], report["max_kl"], report["loss_error"]
                    ));
                }
                let mut error = Error::default();
                error.include(&actual.outputs[0], &self.f32.outputs[0])?;
                raw_logits = Some(serde_json::json!({
                    "role": "diagnostic", "elements": error.elements,
                    "max_abs_error": error.max_error,
                    "max_abs_reference": error.max_reference,
                    "rms_error": (error.square_error / error.elements as f64).sqrt(),
                    "rms_reference": (error.square_reference / error.elements as f64).sqrt(),
                }));
                Some(report)
            } else {
                actual
                    .compare(
                        &self.f32,
                        self.mode,
                        if self.accelerated { 0.02 } else { 1e-4 },
                        self.accelerated,
                    )
                    .map_err(|error| format!("strict-f32 reference: {error}"))?;
                None
            };
            // Arithmetic and fusion choices may differ between programs. The
            // tight repeatability bound applies to executions of this plan;
            // every plan must independently satisfy the F32 correctness gate.
            if let Some(anchor) = &self.anchor {
                actual
                    .compare(&anchor.tensors, self.mode, 1e-4, self.accelerated)
                    .map_err(|error| format!("program repeatability: {error}"))?;
                self.repeatability_checks += 1;
            } else {
                self.anchor = Some(CheckedSnapshot {
                    tensors: actual,
                    quality: quality.clone(),
                    raw_logits: raw_logits.clone(),
                });
            }
            self.program_checks += 1;
            self.last_quality = quality;
            self.last_raw_logits = raw_logits;
            Ok(())
        })();
        if let Err(error) = &result {
            self.rejected_calls += 1;
            eprintln!(
                "[qualification] rejected {:?} candidate: {error}",
                self.mode
            );
        } else {
            self.calls += 1;
        }
        result
    }

    pub fn report(&self) -> serde_json::Value {
        assert!(self.anchor.is_some(), "qualified anchor");
        let gradients: BTreeMap<_, _> = self
            .f32
            .gradients
            .iter()
            .map(|(name, values)| {
                (
                    name,
                    values
                        .iter()
                        .map(|&x| f64::from(x).powi(2))
                        .sum::<f64>()
                        .sqrt(),
                )
            })
            .collect();
        let output = &self.f32.outputs[0];
        let reference_loss = if let Some(language_model) = &self.language_model {
            Some(language_model.reference_loss())
        } else if self.mode == Mode::Training && output.len() == 1 {
            Some(f64::from(output[0]))
        } else {
            None
        };
        serde_json::json!({
            "policy": "selected-program-repeatability-v2", "reference": "same execution plan",
            "rtol": 1e-4, "atol": 1e-6, "accelerated_gradient_rtol": 0.01,
            "program_checks": self.program_checks,
            "repeatability_checks": self.repeatability_checks,
            "qualified_calls": self.calls,
            "rejected_calls": self.rejected_calls,
            "output_elements": self.f32.outputs.iter().map(Vec::len).sum::<usize>(),
            "gradient_tensors": self.f32.gradients.len(),
            "gradient_elements": self.f32.gradients.values().map(Vec::len).sum::<usize>(),
            "arithmetic_reference": {
                "policy": "native-f32-full-tensor-v2", "status": "pass",
                "reference": "original graph; untuned native strict-f32",
                "prepared_before_precision_conversion": true,
                "output_policy": if self.language_model.is_some() { crate::quality::POLICY } else { "tensor-fidelity-v1" },
                "output_rtol": if self.language_model.is_some() { None } else { Some(if self.accelerated { 0.02 } else { 1e-4 }) },
                "language_model": self.last_quality,
                "raw_logits": self.last_raw_logits,
                "gradient_rtol": if self.accelerated { 0.01 } else { 1e-4 },
                "atol": 1e-6, "preparation_seconds": self.reference_seconds,
                "checked_calls": self.calls, "rejected_calls": self.rejected_calls,
                "reference_outputs": {
                    "logits_hash": crate::sha256_f32(output), "logits_sample": crate::validation_sample(output, 256),
                    "loss": reference_loss,
                    "grad_norm": if self.mode == Mode::Training { Some(gradients.values().map(|x| x*x).sum::<f64>().sqrt()) } else { None },
                    "gradient_norms": gradients,
                },
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{Error, Qualification, Snapshot};
    use meganeura::Mode;

    fn inference_gate(reference: Vec<f32>) -> Qualification {
        Qualification {
            output_lengths: vec![reference.len()],
            mode: Mode::Inference,
            accelerated: true,
            f32: Snapshot {
                outputs: vec![reference],
                gradients: Default::default(),
            },
            reference_seconds: 0.0,
            anchor: None,
            anchor_plan: None,
            program_checks: 0,
            repeatability_checks: 0,
            language_model: None,
            last_quality: None,
            last_raw_logits: None,
            calls: 0,
            rejected_calls: 0,
        }
    }

    #[test]
    fn identical_snapshots_reuse_their_own_quality_evidence() {
        let reference = vec![0.9_f32.ln(), 0.1_f32.ln()];
        let mut gate = inference_gate(reference.clone());
        gate.set_language_model(2, &[0]).unwrap();
        let snapshot = |values| Snapshot {
            outputs: vec![values],
            gradients: Default::default(),
        };
        gate.check_snapshot(snapshot(reference.clone())).unwrap();
        let anchor_quality = gate.last_quality.clone();
        let anchor_raw = gate.last_raw_logits.clone();
        let mut close = reference.clone();
        close[0] += 0.00001;
        gate.check_snapshot(snapshot(close)).unwrap();
        assert_ne!(gate.last_quality, anchor_quality);
        gate.check_snapshot(snapshot(reference)).unwrap();
        assert_eq!(gate.last_quality, anchor_quality);
        assert_eq!(gate.last_raw_logits, anchor_raw);
        assert_eq!(gate.repeatability_checks, 2);
        assert!(gate.set_language_model(2, &[1]).is_err());
    }

    #[test]
    fn arithmetic_choices_are_independent_but_each_program_must_repeat() {
        let snapshot = |value| Snapshot {
            outputs: vec![vec![value]],
            gradients: Default::default(),
        };
        let mut gate = inference_gate(vec![1.0]);
        let mut graph = meganeura::Graph::new();
        let x = graph.input("x", &[1]);
        let y = graph.neg(x);
        graph.set_outputs(vec![y]);
        let first = meganeura::compile::compile(&graph);
        let mut second = first.clone();
        // Code-generation knobs matter even when dispatch geometry is equal.
        second.knobs.flash.interleave = !first.knobs.flash.interleave;

        gate.begin_program(&first);
        gate.check_snapshot(snapshot(1.01)).unwrap();
        gate.begin_program(&first);
        gate.check_snapshot(snapshot(1.01)).unwrap();
        assert_eq!(gate.program_checks, 2);

        // This more accurate program must not reproduce the first program's
        // rounding error. Both still obey the unchanged independent F32 gate.
        gate.begin_program(&second);
        gate.check_snapshot(snapshot(1.0)).unwrap();
        gate.check_snapshot(snapshot(1.0)).unwrap();
        assert_eq!(gate.program_checks, 2);
        assert!(
            gate.check_snapshot(snapshot(1.001))
                .unwrap_err()
                .contains("repeatability")
        );
        assert!(
            gate.check_snapshot(snapshot(1.1))
                .unwrap_err()
                .contains("strict-f32")
        );

        // Revisiting an incumbent starts its own fresh repeatability check;
        // neither an intervening candidate nor its failures become its oracle.
        gate.begin_program(&first);
        assert_eq!(gate.program_checks, 0);
        gate.check_snapshot(snapshot(1.01)).unwrap();
        gate.check_snapshot(snapshot(1.01)).unwrap();
        assert_eq!(gate.repeatability_checks, 3);
    }

    #[test]
    fn first_candidate_cannot_define_its_own_precision_reference() {
        let snapshot = |gradient| Snapshot {
            outputs: vec![vec![1.0]],
            gradients: [("weight".into(), gradient)].into(),
        };
        let mut gate = Qualification {
            output_lengths: vec![1],
            mode: Mode::Training,
            accelerated: true,
            f32: snapshot(vec![0.25, -0.5]),
            reference_seconds: 0.01,
            anchor: None,
            anchor_plan: None,
            program_checks: 0,
            repeatability_checks: 0,
            language_model: None,
            last_quality: None,
            last_raw_logits: None,
            calls: 0,
            rejected_calls: 0,
        };
        // Identical norms conceal reversed derivatives; the full gate rejects
        // them even on the very first program, before an anchor exists.
        assert!(gate.check_snapshot(snapshot(vec![-0.25, 0.5])).is_err());
        assert!(gate.anchor.is_none());
        assert!(gate.check_snapshot(snapshot(vec![0.0, 0.0])).is_err());
        assert!(gate.anchor.is_none());
        gate.check_snapshot(snapshot(vec![0.25, -0.5])).unwrap();
        assert_eq!(gate.report()["arithmetic_reference"]["rejected_calls"], 2);
        assert_eq!(gate.report()["arithmetic_reference"]["checked_calls"], 1);
        assert!(gate.check_snapshot(snapshot(vec![f32::NAN, -0.5])).is_err());
    }

    #[test]
    fn full_tensor_bounds_reject_shape_nonfinite_sparse_and_diffuse_errors() {
        assert!(Error::default().include(&[0.0], &[0.0, 1.0]).is_err());
        assert!(Error::default().include(&[f32::NAN], &[0.0]).is_err());
        assert!(Error::default().check(1e-4).is_err());
        let reference: Vec<_> = (0..19).map(|i| i as f32 * 0.125).collect();
        let mut error = Error::default();
        error.include(&reference, &reference).unwrap();
        assert_eq!(error.elements, reference.len());
        assert_eq!(error.max_reference, 2.25);
        assert_eq!(
            error.square_reference,
            reference.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>()
        );
        for index in [0, 7, 8, 15, 18] {
            let mut corrupted = reference.clone();
            corrupted[index] += 1.0;
            let mut error = Error::default();
            error.include(&corrupted, &reference).unwrap();
            assert_eq!(error.square_error, 1.0);
            assert!(error.check(1e-4).is_err());
            for nonfinite in [f32::NAN, f32::INFINITY] {
                corrupted[index] = nonfinite;
                assert!(Error::default().include(&corrupted, &reference).is_err());
                assert!(Error::default().include(&reference, &corrupted).is_err());
            }
        }
        for actual in [vec![0.0, 1.0], vec![0.01, 1.0], vec![0.0, 1.00001]] {
            let mut error = Error::default();
            error.include(&actual, &[0.0, 1.0]).unwrap();
            assert_eq!(error.check(1e-4).is_ok(), actual[0] == 0.0);
        }
        let mut error = Error::default();
        error
            .include(&[0.00001, 0.00001, 0.00001, 1.0], &[0.0, 0.0, 0.0, 1.0])
            .unwrap();
        assert!(error.check(0.0).is_err());
        let mut total = Error::default();
        total.merge(&error);
        total.merge(&error);
        assert_eq!(total.elements, 2 * error.elements);
        assert_eq!(total.check(1e-4).is_ok(), error.check(1e-4).is_ok());
        assert!(total.check(0.0).is_err());
    }
}
