//! Immutable search parameters and the synthetic initializer shared with PyTorch.

/// Reuse only a complete matching parameter representation. Packed parameters
/// must match too: skipping source uploads would otherwise leave them unfilled.
pub fn reuse_parameters(
    session: &mut meganeura::Session,
    source: &mut meganeura::Session,
) -> Result<bool, String> {
    let target = session.plan();
    let origin = source.plan();
    let compatible = target.param_buffers.len() == origin.param_buffers.len()
        && target.param_buffers.iter().all(|(name, buffer)| {
            let Some((_, original)) = origin.param_buffers.iter().find(|entry| entry.0 == *name)
            else {
                return false;
            };
            target.param_types.get(buffer).is_some_and(|ty| {
                origin.param_types.get(original) == Some(ty)
                    && target
                        .derived_params
                        .iter()
                        .find(|entry| entry.0 == *buffer)
                        .map(|entry| (&entry.1, &entry.2))
                        == origin
                            .derived_params
                            .iter()
                            .find(|entry| entry.0 == *original)
                            .map(|entry| (&entry.1, &entry.2))
            })
        });
    if compatible {
        for (name, _) in target.param_buffers.clone() {
            session
                .share_parameter_from(source, &name)
                .map_err(|error| error.to_string())?;
        }
    }
    Ok(compatible)
}

pub const POLICY: &str = "name-index-uniform-v1";
// Uniform variance is amplitude²/3; preserve the old 0.02 sinusoid's RMS.
pub const AMPLITUDE: f32 = 0.024494898;

pub fn parameter_values(name: &str, count: usize, amplitude: f32) -> Vec<f32> {
    let seed = name.bytes().fold(0u32, |hash, byte| {
        hash.wrapping_mul(31).wrapping_add(u32::from(byte))
    });
    (0..count)
        .map(|index| {
            let mut hash = index as u32 ^ seed;
            hash = (hash ^ (hash >> 16)).wrapping_mul(0x7feb352d);
            hash = (hash ^ (hash >> 15)).wrapping_mul(0x846ca68b);
            hash ^= hash >> 16;
            // The upper 24 bits convert exactly to f32. No platform RNG or sin.
            ((hash >> 8) as f32 * (1.0 / 8388608.0) - 1.0) * amplitude
        })
        .collect()
}

#[test]
fn synthetic_parameter_bits() {
    // The Python test carries the same cross-language reference vector.
    let values = parameter_values("time_embed.0.weight", 8, AMPLITUDE);
    assert_eq!(
        values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        [
            0x3b10bfaf, 0xbb65c670, 0x3adfd0cc, 0xbb06e742, 0x3b887a65, 0x3c9f5be6, 0xbcb7f94b,
            0x3cafdd01,
        ]
    );
    assert_eq!(
        parameter_values("time_embed.0.weight", 8, AMPLITUDE * 0.5),
        values.iter().map(|value| value * 0.5).collect::<Vec<_>>()
    );
    assert_ne!(
        values,
        parameter_values("time_embed.1.weight", 8, AMPLITUDE)
    );
    assert!(parameter_values("empty", 0, AMPLITUDE).is_empty());
}
