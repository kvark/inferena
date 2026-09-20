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
    for &(ref name, parameter) in &plan.param_buffers {
        let Some(&(_, gradient)) = plan
            .param_grad_pairs
            .iter()
            .find(|entry| entry.0 == parameter)
        else {
            continue;
        };
        let mut values = vec![0.0; session.param_size(name).unwrap()];
        session.read_buffer(gradient, &mut values);
        if let Some((_, sources, transform)) = plan
            .derived_params
            .iter()
            .find(|entry| entry.0 == parameter)
        {
            let width = plan.param_types[&parameter].shape[1];
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
                    _ => return Err(format!("unsupported gradient transform: {transform:?}")),
                };
                add(source, unpacked)?;
                offset += extent;
            }
        } else {
            add(name, values)?;
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
    fn include(&mut self, actual: &[f32], reference: &[f32]) -> Result<(), String> {
        if actual.is_empty() || actual.len() != reference.len() {
            return Err("tensor shape changed or empty".into());
        }
        for (&a, &b) in actual.iter().zip(reference) {
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
        let outputs: Vec<_> = self
            .output_lengths
            .iter()
            .enumerate()
            .map(|(index, &len)| {
                let mut values = vec![0.0; len];
                session.read_output_by_index(index, &mut values);
                values
            })
            .collect();
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
            total.include(actual, reference)?;
            if !accelerated {
                let mut error = Error::default();
                error.include(actual, reference)?;
                error
                    .check(1e-4)
                    .map_err(|error| format!("gradient {name}: {error}"))?;
            }
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
    }
}
