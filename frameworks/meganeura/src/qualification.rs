//! Full-tensor construction checks against both F32 arithmetic and the search anchor.
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

struct Snapshot {
    outputs: Vec<Vec<f32>>,
    gradients: BTreeMap<String, Vec<f32>>,
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
    anchor: Option<Snapshot>,
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
            calls: 0,
            rejected_calls: 0,
        })
    }

    pub fn set_reference_seconds(&mut self, seconds: f64) {
        self.reference_seconds = seconds;
    }

    pub fn check(&mut self, session: &Session) -> Result<(), String> {
        self.check_snapshot(Snapshot::read(session, &self.output_lengths, self.mode)?)
    }

    fn check_snapshot(&mut self, actual: Snapshot) -> Result<(), String> {
        let result = (|| {
            actual
                .compare(
                    &self.f32,
                    self.mode,
                    if self.accelerated { 0.02 } else { 1e-4 },
                    self.accelerated,
                )
                .map_err(|error| format!("strict-f32 reference: {error}"))?;
            // Preserve the tighter within-search repeatability check as well.
            actual
                .compare(
                    self.anchor.as_ref().unwrap_or(&actual),
                    self.mode,
                    1e-4,
                    self.accelerated,
                )
                .map_err(|error| format!("search anchor: {error}"))?;
            if self.anchor.is_none() {
                self.anchor = Some(actual);
            }
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
        serde_json::json!({
            "policy": "fixed-full-tensor-v4", "reference": "first checked program",
            "rtol": 1e-4, "atol": 1e-6, "accelerated_gradient_rtol": 0.01,
            "qualified_calls": self.calls,
            "rejected_calls": self.rejected_calls,
            "output_elements": self.f32.outputs.iter().map(Vec::len).sum::<usize>(),
            "gradient_tensors": self.f32.gradients.len(),
            "gradient_elements": self.f32.gradients.values().map(Vec::len).sum::<usize>(),
            "arithmetic_reference": {
                "policy": "native-f32-full-tensor-v1", "status": "pass",
                "reference": "original graph; untuned native strict-f32",
                "prepared_before_precision_conversion": true,
                "output_rtol": if self.accelerated { 0.02 } else { 1e-4 },
                "gradient_rtol": if self.accelerated { 0.01 } else { 1e-4 },
                "atol": 1e-6, "preparation_seconds": self.reference_seconds,
                "checked_calls": self.calls, "rejected_calls": self.rejected_calls,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{Error, Qualification, Snapshot};
    use meganeura::Mode;

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
