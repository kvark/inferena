//! Full-tensor construction checks. The outer harness remains the PyTorch oracle.
use meganeura::{Graph, Mode, Session, graph::ParamTransform};
use std::collections::BTreeMap;

pub fn source_parameters(session: &Session) -> Vec<(String, usize)> {
    let plan = session.plan();
    plan.param_buffers
        .iter()
        .filter(|entry| {
            !plan
                .derived_params
                .iter()
                .any(|derived| derived.0 == entry.1)
        })
        .map(|entry| (entry.0.clone(), session.param_size(&entry.0).unwrap()))
        .collect()
}

/// Restore checkpoint coordinates, including gradients of packed projections.
pub fn gradients(session: &Session) -> Result<BTreeMap<String, Vec<f32>>, String> {
    let plan = session.plan();
    let mut gradients = BTreeMap::<String, Vec<f32>>::new();
    let mut add = |name: &str, values: Vec<f32>| -> Result<(), String> {
        if let Some(previous) = gradients.get_mut(name) {
            if previous.len() != values.len() {
                return Err(format!("gradient shape changed: {name}"));
            }
            for (sum, value) in previous.iter_mut().zip(values) {
                *sum += value;
            }
        } else {
            gradients.insert(name.to_owned(), values);
        }
        Ok(())
    };
    let requests: Vec<_> = plan
        .param_buffers
        .iter()
        .filter_map(|(name, parameter)| {
            plan.param_grad_pairs
                .iter()
                .find(|entry| entry.0 == *parameter)
                .map(|&(_, gradient)| (name, *parameter, gradient))
        })
        .collect();
    let mut pending = requests.as_slice();
    while !pending.is_empty() {
        // Batch staging avoids a submission per gradient and repeated mapped-read
        // probes. Bound temporary storage to 64 MiB or one oversized tensor.
        let mut bytes = 0;
        let count = pending
            .iter()
            .take_while(|request| {
                let size = plan.buffers[request.2.0 as usize];
                if bytes != 0 && bytes + size > 64 * 1024 * 1024 {
                    return false;
                }
                bytes += size;
                true
            })
            .count();
        let (batch, rest) = pending.split_at(count);
        pending = rest;
        let buffers: Vec<_> = batch.iter().map(|request| request.2).collect();
        for ((name, parameter, _), mut values) in batch.iter().zip(session.read_buffers(&buffers)) {
            let len = session.param_size(name).unwrap();
            if values.len() < len {
                return Err(format!(
                    "gradient buffer is smaller than its tensor: {name}"
                ));
            }
            values.truncate(len);
            if let Some((_, sources, transform)) = plan
                .derived_params
                .iter()
                .find(|entry| entry.0 == *parameter)
            {
                let width = plan.param_types[parameter].shape[1];
                let mut offset = 0;
                for &(ref source, extent) in sources {
                    let unpacked = match *transform {
                        ParamTransform::HorizontalConcat => values
                            .chunks_exact(width)
                            .flat_map(|row| row[offset..offset + extent].iter().copied())
                            .collect(),
                        ParamTransform::VerticalConcat => {
                            values[offset * width..(offset + extent) * width].to_vec()
                        }
                    };
                    add(source, unpacked)?;
                    offset += extent;
                }
            } else {
                add(name, values)?;
            }
        }
    }
    Ok(gradients)
}

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
            "policy": "fixed-full-tensor-v4", "reference": "ordinary untuned construction",
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
