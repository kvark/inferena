//! Full-tensor construction checks. The outer harness remains the PyTorch oracle.
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

pub struct Qualification {
    output_lengths: Vec<usize>,
    mode: Mode,
    reference: Option<Snapshot>,
    calls: usize,
}

impl Qualification {
    pub fn new(graph: &Graph, mode: Mode) -> Self {
        Self {
            output_lengths: graph
                .outputs()
                .iter()
                .map(|&id| graph.node(id).ty.num_elements())
                .collect(),
            mode,
            reference: None,
            calls: 0,
        }
    }

    pub fn check(&mut self, session: &Session) -> Result<(), String> {
        if session.num_outputs() != self.output_lengths.len() {
            return Err("output inventory changed".into());
        }
        // Candidate allocations are short-lived. A staged batch avoids paying
        // the mapped-read calibration cost again for every new candidate.
        let mut outputs = session.read_buffers(&session.plan().output_buffers);
        for (values, &len) in outputs.iter_mut().zip(&self.output_lengths) {
            if values.len() < len {
                return Err("output buffer is smaller than its logical tensor".into());
            }
            values.truncate(len);
        }
        let gradients = if self.mode == Mode::Training {
            gradients(session)?
        } else {
            BTreeMap::new()
        };
        let (reference_outputs, reference_gradients) = self
            .reference
            .as_ref()
            .map(|snapshot| (&snapshot.outputs, &snapshot.gradients))
            .unwrap_or((&outputs, &gradients));
        if gradients.keys().ne(reference_gradients.keys()) {
            return Err("gradient inventory changed".into());
        }
        for (index, (actual, reference)) in outputs.iter().zip(reference_outputs).enumerate() {
            let mut error = Error::default();
            error.include(actual, reference)?;
            error
                .check(1e-4)
                .map_err(|error| format!("output {index}: {error}"))?;
        }
        let accelerated = std::env::var("INFERENA_STRICT").as_deref() != Ok("1");
        let mut total = Error::default();
        for (name, actual) in &gradients {
            let reference = &reference_gradients[name];
            let mut error = Error::default();
            error.include(actual, reference)?;
            if !accelerated {
                error
                    .check(1e-4)
                    .map_err(|error| format!("gradient {name}: {error}"))?;
            }
            total.merge(&error);
        }
        if self.mode == Mode::Training {
            total.check(if accelerated { 0.01 } else { 1e-4 })?;
        }
        if self.reference.is_none() {
            self.reference = Some(Snapshot { outputs, gradients });
        }
        self.calls += 1;
        Ok(())
    }

    pub fn report(&self) -> serde_json::Value {
        let reference = self.reference.as_ref().expect("qualified anchor");
        serde_json::json!({
            // With a completed memo, the first check is already kernel-tuned.
            // Cross-engine correctness is checked separately from this construction gate.
            "policy": "fixed-full-tensor-v4", "reference": "first checked program",
            "rtol": 1e-4, "atol": 1e-6, "accelerated_gradient_rtol": 0.01,
            "qualified_calls": self.calls,
            "output_elements": reference.outputs.iter().map(Vec::len).sum::<usize>(),
            "gradient_tensors": reference.gradients.len(),
            "gradient_elements": reference.gradients.values().map(Vec::len).sum::<usize>(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::Error;

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
