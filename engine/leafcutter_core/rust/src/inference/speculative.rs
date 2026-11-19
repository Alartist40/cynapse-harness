//! Eagle-style speculative decoding
//!
//! Uses next-token prediction heads (`nextn.*` tensors) to draft multiple
//! tokens per forward pass, then verifies them with the main model.
//!
//! Algorithm:
//!   1. Main model produces hidden state h_t
//!   2. Draft head predicts h_{t+1}, h_{t+2}, ... h_{t+gamma}
//!   3. Main model verifies all gamma draft tokens in parallel
//!   4. Accept tokens until first mismatch, then sample corrected distribution
//!
//! For Qwen3.5, the draft head tensors are in layer 32:
//!   - nextn.eh_proj: projects hidden state to draft embedding
//!   - nextn.enorm:  draft embedding RMSNorm
//!   - nextn.hnorm:  hidden state RMSNorm before draft
//!   - nextn.eh_scale: scaling factor

use crate::model::tensor::Tensor;
use std::collections::HashMap;

/// Holds the weights for an Eagle speculative decoding head.
pub struct SpeculativeHead {
    pub eh_proj: Tensor,
    pub enorm: Option<Tensor>,
    pub hnorm: Option<Tensor>,
    pub eh_scale: Option<f32>,
    pub gamma: usize, // Number of draft tokens to generate
}

impl SpeculativeHead {
    pub fn from_weights(weights: &HashMap<String, Tensor>, gamma: usize) -> Option<Self> {
        let eh_proj = weights.get("nextn.eh_proj.weight")
            .or_else(|| weights.get("nextn.eh_proj"))?
            .clone();
        let enorm = weights.get("nextn.enorm.weight").cloned();
        let hnorm = weights.get("nextn.hnorm.weight").cloned();
        let eh_scale = weights.get("nextn.eh_scale")
            .map(|t| t.data.get(0).copied().unwrap_or(1.0));

        Some(Self { eh_proj, enorm, hnorm, eh_scale, gamma })
    }

    /// Draft `gamma` future tokens given the current hidden state.
    /// Returns a vec of draft hidden states (one per future token).
    pub fn draft(&self, hidden: &Tensor, _lm_head: &Tensor) -> Vec<Tensor> {
        let mut drafts = Vec::with_capacity(self.gamma);
        let mut current = hidden.clone();

        for _ in 0..self.gamma {
            // Project hidden → draft embedding
            let mut draft_emb = current.matmul(&self.eh_proj);

            // Apply norms if present
            if let Some(ref enorm) = self.enorm {
                draft_emb = draft_emb.rms_norm(enorm, 1e-5);
            }

            // Scale
            if let Some(scale) = self.eh_scale {
                for v in &mut draft_emb.data { *v *= scale; }
            }

            drafts.push(draft_emb.clone());

            // For the next iteration, we'd need the actual token embedding.
            // In a full implementation, we'd sample a token, look up its embedding,
            // and feed it back. Here we approximate by reusing the projection.
            current = draft_emb;
        }

        drafts
    }

    /// Verify draft tokens against a single position's main logits (greedy check).
    pub fn verify(
        &self,
        draft_tokens: &[usize],
        main_logits: &[f32],
        _temperature: f32,
    ) -> (usize, usize) {
        if draft_tokens.is_empty() || main_logits.is_empty() {
            return (0, 0);
        }
        let target = greedy_sample(main_logits);
        if draft_tokens[0] == target {
            (1, target)
        } else {
            (0, target)
        }
    }

    /// Verify a full draft sequence against per-position target logits.
    /// Returns `(accepted_count, next_token)` where `next_token` is the first
    /// corrected token (or bonus token if all drafts matched).
    pub fn verify_batch(
        &self,
        draft_tokens: &[usize],
        target_logits: &[&[f32]],
    ) -> (usize, usize) {
        let mut accepted = 0usize;
        for (i, &draft_tok) in draft_tokens.iter().enumerate() {
            let Some(logits) = target_logits.get(i) else {
                break;
            };
            let target_tok = greedy_sample(logits);
            if draft_tok == target_tok {
                accepted += 1;
            } else {
                return (accepted, target_tok);
            }
        }
        let bonus = target_logits
            .get(accepted)
            .map(|l| greedy_sample(l))
            .unwrap_or(0);
        (accepted, bonus)
    }
}

/// Hint returned by `SpeculativeDecoder::try_accept` so callers can detect
/// the "speculative decoding is currently a no-op" case and skip the draft
/// step entirely on subsequent tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeculativeStatus {
    /// Speculative decoding is fully implemented and returned a verdict.
    Active,
    /// Underlying `verify` returned a no-op; treat drafts as always-rejected.
    Disabled,
}

/// Speculative decoder that manages draft generation and verification.
pub struct SpeculativeDecoder {
    pub head: SpeculativeHead,
}

impl SpeculativeDecoder {
    pub fn new(head: SpeculativeHead) -> Self {
        Self { head }
    }

    /// Generate tokens using speculative decoding.
    /// `generate_fn` is a callback that runs the main model for a given token sequence.
    pub fn generate<F>(
        &self,
        initial_tokens: &[usize],
        max_tokens: usize,
        _temperature: f32,
        mut generate_fn: F,
    ) -> Vec<usize>
    where
        F: FnMut(&[usize]) -> Vec<f32>,
    {
        let mut tokens = initial_tokens.to_vec();

        while tokens.len() < initial_tokens.len() + max_tokens {
            // Get main model logits for current sequence
            let _logits = generate_fn(&tokens);

            // Draft future tokens (simplified — full impl needs hidden state passing)
            // For now, just generate 1 token at a time
            let next_token = greedy_sample(&_logits);
            tokens.push(next_token);

            if next_token == 2 { // EOS
                break;
            }
        }

        tokens[initial_tokens.len()..].to_vec()
    }
}

/// Finds speculative draft tokens using prompt n-gram matching (Prompt Lookup Decoding).
/// Matches the trailing `ngram_size` tokens against earlier occurrences in `context`
/// and drafts up to `max_draft` subsequent tokens.
pub fn find_prompt_lookup_draft(context: &[usize], ngram_size: usize, max_draft: usize) -> Vec<usize> {
    if context.len() < ngram_size + 1 || ngram_size == 0 || max_draft == 0 {
        return Vec::new();
    }
    let query = &context[context.len() - ngram_size..];
    let search_end = context.len() - ngram_size;
    for i in (0..search_end).rev() {
        if &context[i..i + ngram_size] == query {
            let draft_start = i + ngram_size;
            let draft_len = (search_end - draft_start).min(max_draft);
            if draft_len > 0 {
                return context[draft_start..draft_start + draft_len].to_vec();
            }
        }
    }
    Vec::new()
}

/// Multi-length suffix lookup (tries `max_ngram` down to `min_ngram` for highest precision).
pub fn find_best_suffix_draft(
    context: &[usize],
    min_ngram: usize,
    max_ngram: usize,
    max_draft: usize,
) -> Vec<usize> {
    let min_n = min_ngram.max(1);
    let max_n = max_ngram.max(min_n);
    for n in (min_n..=max_n).rev() {
        let draft = find_prompt_lookup_draft(context, n, max_draft);
        if !draft.is_empty() {
            return draft;
        }
    }
    Vec::new()
}

/// Adaptive speculation controller (`strata/engine/src/spec/controller.cpp`).
/// Tracks EMA acceptance rate and scales `current_gamma` dynamically, pausing
/// speculation during low-acceptance stretches.
#[derive(Debug, Clone)]
pub struct AdaptiveSpecController {
    pub current_gamma: usize,
    pub min_gamma: usize,
    pub max_gamma: usize,
    pub ema_accept: f32,
    cooldown: usize,
}

impl AdaptiveSpecController {
    pub fn new(initial_gamma: usize, min_gamma: usize, max_gamma: usize) -> Self {
        let min_g = min_gamma.max(1);
        let max_g = max_gamma.max(min_g);
        Self {
            current_gamma: initial_gamma.clamp(min_g, max_g),
            min_gamma: min_g,
            max_gamma: max_g,
            ema_accept: 0.5,
            cooldown: 0,
        }
    }

    pub fn should_speculate(&mut self) -> bool {
        if self.cooldown > 0 {
            self.cooldown -= 1;
            return false;
        }
        true
    }

    pub fn record_outcome(&mut self, drafted: usize, accepted: usize) {
        if drafted == 0 {
            return;
        }
        let rate = (accepted as f32 / drafted as f32).clamp(0.0, 1.0);
        self.ema_accept = 0.7 * self.ema_accept + 0.3 * rate;
        if self.ema_accept > 0.65 && self.current_gamma < self.max_gamma {
            self.current_gamma += 1;
        } else if self.ema_accept < 0.25 {
            if self.current_gamma > self.min_gamma {
                self.current_gamma -= 1;
            } else {
                self.cooldown = 4;
            }
        }
    }
}

/// Greedy sample argmax from logits
fn greedy_sample(logits: &[f32]) -> usize {
    logits.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speculative_head_creation() {
        let mut weights = HashMap::new();
        weights.insert("nextn.eh_proj.weight".to_string(), Tensor::from_vec(vec![0.1; 64 * 64], vec![64, 64]));
        weights.insert("nextn.enorm.weight".to_string(), Tensor::from_vec(vec![1.0; 64], vec![64]));

        let head = SpeculativeHead::from_weights(&weights, 4);
        assert!(head.is_some());
        let head = head.unwrap();
        assert_eq!(head.gamma, 4);
        assert!(head.enorm.is_some());
    }

    #[test]
    fn test_speculative_head_missing() {
        let weights = HashMap::new();
        assert!(SpeculativeHead::from_weights(&weights, 4).is_none());
    }

    #[test]
    fn test_draft_produces_gamma_outputs() {
        let mut weights = HashMap::new();
        weights.insert("nextn.eh_proj.weight".to_string(), Tensor::from_vec(vec![0.0; 16 * 16], vec![16, 16]));

        let head = SpeculativeHead::from_weights(&weights, 3).unwrap();
        let hidden = Tensor::from_vec(vec![0.1; 16], vec![1, 16]);
        let lm_head = Tensor::from_vec(vec![0.0; 16 * 100], vec![16, 100]);

        let drafts = head.draft(&hidden, &lm_head);
        assert_eq!(drafts.len(), 3);
    }

    #[test]
    fn test_find_prompt_lookup_draft_success() {
        let context = vec![10, 20, 30, 40, 50, 60, 10, 20, 30];
        let drafts = find_prompt_lookup_draft(&context, 3, 4);
        assert_eq!(drafts, vec![40, 50, 60]);
    }

    #[test]
    fn test_find_prompt_lookup_draft_no_match() {
        let context = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let drafts = find_prompt_lookup_draft(&context, 3, 3);
        assert!(drafts.is_empty());
    }

    #[test]
    fn test_find_prompt_lookup_draft_short_context() {
        let context = vec![10, 20];
        let drafts = find_prompt_lookup_draft(&context, 3, 3);
        assert!(drafts.is_empty());
    }

    #[test]
    fn test_adaptive_suffix_spec_controller_and_verify_batch() {
        let context = vec![1, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5];
        let best = find_best_suffix_draft(&context, 2, 5, 3);
        assert_eq!(best, vec![6, 7]);

        let mut weights = HashMap::new();
        weights.insert("nextn.eh_proj.weight".to_string(), Tensor::from_vec(vec![0.0; 4], vec![2, 2]));
        let head = SpeculativeHead::from_weights(&weights, 3).unwrap();
        let l0 = [0.1, 0.9, 0.0]; // argmax 1
        let l1 = [0.0, 0.2, 0.8]; // argmax 2
        let l2 = [0.9, 0.1, 0.0]; // argmax 0
        let (acc, next) = head.verify_batch(&[1, 99], &[&l0, &l1, &l2]);
        assert_eq!(acc, 1);
        assert_eq!(next, 2);

        let mut ctrl = AdaptiveSpecController::new(2, 1, 5);
        for _ in 0..4 {
            ctrl.record_outcome(3, 3);
        }
        assert!(ctrl.current_gamma > 2);
        for _ in 0..8 {
            ctrl.record_outcome(3, 0);
        }
        assert!(!ctrl.should_speculate());
    }
}
