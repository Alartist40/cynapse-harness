//! Multi-snapshot prefix pool (`colibri/c/pin_pool.h`).
//!
//! Keeps 4–8 pinned `(token_ids, last_logits, recurrent_state)` snapshots and
//! rewinds to the deepest strict prefix of a new prompt, avoiding re-prefilling
//! shared system instructions and recurrent DeltaNet/SSM states.

#[derive(Clone, Debug)]
pub struct PinSnapshot {
    pub ids: Vec<usize>,
    pub logits: Vec<f32>,
    pub recurrent_state: Vec<u8>,
    pub used_tick: u64,
}

pub struct PinPool {
    slots: Vec<PinSnapshot>,
    max_slots: usize,
    clock: u64,
}

impl PinPool {
    pub fn new(max_slots: usize) -> Self {
        Self {
            slots: Vec::new(),
            max_slots: max_slots.clamp(1, 8),
            clock: 0,
        }
    }

    pub fn save_snapshot(&mut self, ids: Vec<usize>, logits: Vec<f32>, recurrent_state: Vec<u8>) {
        if ids.is_empty() {
            return;
        }
        self.clock += 1;
        if let Some(existing) = self.slots.iter_mut().find(|s| s.ids == ids) {
            existing.logits = logits;
            existing.recurrent_state = recurrent_state;
            existing.used_tick = self.clock;
            return;
        }
        if self.slots.len() >= self.max_slots {
            if let Some((evict_idx, _)) = self
                .slots
                .iter()
                .enumerate()
                .min_by_key(|(_, s)| s.used_tick)
            {
                self.slots.swap_remove(evict_idx);
            }
        }
        self.slots.push(PinSnapshot {
            ids,
            logits,
            recurrent_state,
            used_tick: self.clock,
        });
    }

    /// Return the deepest snapshot that is a strict prefix of `prompt_ids`
    /// (`snap.ids.len() < prompt_ids.len()`), updating its LRU timestamp.
    pub fn best_strict_prefix(&mut self, prompt_ids: &[usize]) -> Option<&PinSnapshot> {
        let best_idx = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.ids.len() < prompt_ids.len() && prompt_ids.starts_with(&s.ids))
            .max_by_key(|(_, s)| s.ids.len())
            .map(|(i, _)| i)?;

        self.clock += 1;
        self.slots[best_idx].used_tick = self.clock;
        Some(&self.slots[best_idx])
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_deepest_strict_prefix_and_evicts_lru() {
        let mut pool = PinPool::new(2);
        pool.save_snapshot(vec![1, 2], vec![0.1], vec![10]);
        pool.save_snapshot(vec![1, 2, 3, 4], vec![0.4], vec![20]);

        let best = pool.best_strict_prefix(&[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(best.ids, vec![1, 2, 3, 4]);
        assert_eq!(best.recurrent_state, vec![20]);

        // Exact match is not a strict prefix; falls back to [1, 2].
        let fallback = pool.best_strict_prefix(&[1, 2, 3, 4]).unwrap();
        assert_eq!(fallback.ids, vec![1, 2]);

        // Inserting a 3rd snapshot evicts the LRU one ([1, 2, 3, 4] was older than [1, 2]).
        pool.save_snapshot(vec![9, 9], vec![0.9], vec![30]);
        assert_eq!(pool.len(), 2);
        assert!(pool.best_strict_prefix(&[1, 2, 9]).is_some());
    }
}
