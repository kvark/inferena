//! The same language-model distribution/loss contract as scripts/quality.py.
pub const POLICY: &str = "language-model-distribution-v1";

pub struct LanguageModel {
    vocabulary: usize,
    targets: Vec<u32>,
    log_probabilities: Vec<f64>,
    probabilities: Vec<f64>,
    losses: Vec<f64>,
}

fn log_softmax(values: &[f32]) -> Result<Vec<f64>, String> {
    if values.is_empty() || values.iter().any(|x| !x.is_finite()) {
        return Err("empty or non-finite language-model logits".into());
    }
    let peak = values.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum = values.iter().map(|&x| (x as f64 - peak).exp()).sum::<f64>();
    let offset = sum.ln();
    Ok(values.iter().map(|&x| (x as f64 - peak) - offset).collect())
}

impl LanguageModel {
    pub fn new(reference: &[f32], vocabulary: usize, targets: &[u32]) -> Result<Self, String> {
        if vocabulary < 2
            || targets.is_empty()
            || reference.len() != vocabulary * targets.len()
            || targets.iter().any(|&x| x as usize >= vocabulary)
        {
            return Err("invalid language-model shape or targets".into());
        }
        let mut log_probabilities = Vec::with_capacity(reference.len());
        let mut losses = Vec::with_capacity(targets.len());
        for (row, &target) in reference.chunks_exact(vocabulary).zip(targets) {
            let log = log_softmax(row)?;
            losses.push(-log[target as usize]);
            log_probabilities.extend(log);
        }
        let probabilities = log_probabilities.iter().map(|x| x.exp()).collect();
        Ok(Self {
            vocabulary,
            targets: targets.to_vec(),
            log_probabilities,
            probabilities,
            losses,
        })
    }

    pub fn reference_loss(&self) -> f64 {
        self.losses.iter().sum::<f64>() / self.losses.len() as f64
    }

    pub fn evaluate(&self, actual: &[f32]) -> Result<serde_json::Value, String> {
        if actual.len() != self.log_probabilities.len() {
            return Err("language-model output shape changed".into());
        }
        let mut kl = Vec::with_capacity(self.targets.len());
        let mut losses = Vec::with_capacity(self.targets.len());
        for (index, row) in actual.chunks_exact(self.vocabulary).enumerate() {
            let log = log_softmax(row)?;
            let offset = index * self.vocabulary;
            let divergence = log
                .iter()
                .enumerate()
                .map(|(i, &a)| {
                    self.probabilities[offset + i] * (self.log_probabilities[offset + i] - a)
                })
                .sum::<f64>();
            kl.push(divergence.max(0.0));
            losses.push(-log[self.targets[index] as usize]);
        }
        let mean_kl = kl.iter().sum::<f64>() / kl.len() as f64;
        let max_kl = kl.iter().copied().fold(0.0, f64::max);
        let loss = losses.iter().sum::<f64>() / losses.len() as f64;
        let reference_loss = self.reference_loss();
        let loss_error = (loss - reference_loss).abs();
        let loss_bound = 1e-6 + 0.02 * reference_loss.abs();
        let mean_limit = 1.01_f64.ln();
        let token_limit = 1.10_f64.ln();
        Ok(serde_json::json!({
            "policy": POLICY,
            "status": if mean_kl <= mean_limit && max_kl <= token_limit && loss_error <= loss_bound { "pass" } else { "failed" },
            "tokens": kl.len(), "vocabulary": self.vocabulary,
            "mean_kl_limit": mean_limit, "token_kl_limit": token_limit,
            "loss_rtol": 0.02, "loss_atol": 1e-6,
            "mean_kl": mean_kl, "max_kl": max_kl,
            "loss": loss, "reference_loss": reference_loss, "loss_error": loss_error, "loss_bound": loss_bound,
            "kl_per_token": kl, "losses": losses, "reference_losses": self.losses,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::LanguageModel;

    #[test]
    fn probability_invariance_and_corruption() {
        let reference = [0.9_f32.ln(), 0.1_f32.ln()].repeat(512);
        let gate = LanguageModel::new(&reference, 2, &[0; 512]).unwrap();
        let shift: Vec<_> = reference.iter().map(|x| x + 1.0).collect();
        assert_eq!(gate.evaluate(&shift).unwrap()["status"], "pass");
        for wrong in [
            vec![0.0; reference.len()],
            reference.iter().map(|x| -*x).collect(),
        ] {
            assert_eq!(gate.evaluate(&wrong).unwrap()["status"], "failed");
        }
        let mut sparse = reference.clone();
        sparse[..2].fill(0.0);
        let report = gate.evaluate(&sparse).unwrap();
        assert!(report["mean_kl"].as_f64().unwrap() < 1.01_f64.ln());
        assert!(report["loss_error"].as_f64().unwrap() < report["loss_bound"].as_f64().unwrap());
        assert_eq!(report["status"], "failed");
        sparse[0] = f32::NAN;
        assert!(gate.evaluate(&sparse).is_err());
        assert!(LanguageModel::new(&reference, 2, &[2; 512]).is_err());
    }
}
