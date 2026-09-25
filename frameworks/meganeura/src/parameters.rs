//! Parameter initialization and gradient readback in original model coordinates.
use meganeura::{Session, graph::ParamTransform};
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
