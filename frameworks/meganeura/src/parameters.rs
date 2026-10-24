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

/// Only skip a source when all of its derived representations are inherited
/// too. A shared source with a new packed/fused representation must be uploaded
/// again to initialize that representation. References and standalone builds
/// have no donor and always initialize every source from fresh data.
pub fn initialization_parameters(
    session: &Session,
    donor: Option<&Session>,
) -> Vec<(String, usize)> {
    let mut parameters = source_parameters(session);
    parameters
        .retain(|(name, _)| !donor.is_some_and(|donor| session.inherits_parameter(donor, name)));
    parameters
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
