//! Largest-first trunk pinning and 2-slot async streaming ring.
//!
//! Inspired by `kimi-k3-in-c` (`k3_trunk.h`):
//! - Pinning the fattest dense layers first shrinks the uniform streaming ring
//!   slot size, freeing budget to pin additional layers via fixed-point iteration.
//! - Uses a 2-slot ring when budget allows so layer `L+1` can be prefetched
//!   while layer `L` is computed.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrunkPlan {
    pub pinned_layers: Vec<usize>,
    pub streaming_layers: Vec<usize>,
    pub slot_bytes: usize,
    pub ring_slots: usize,
    pub pinned_bytes: usize,
}

/// Fixed-point solver: pins largest layers first so the uniform ring slot
/// shrinks to the largest remaining unpinned layer.
pub fn solve_pinned_layers(layer_sizes: &[usize], budget_bytes: usize) -> TrunkPlan {
    if layer_sizes.is_empty() {
        return TrunkPlan {
            pinned_layers: vec![],
            streaming_layers: vec![],
            slot_bytes: 0,
            ring_slots: 0,
            pinned_bytes: 0,
        };
    }

    let mut order: Vec<(usize, usize)> = layer_sizes.iter().copied().enumerate().collect();
    order.sort_by(|a, b| b.1.cmp(&a.1));

    let mut is_pinned = vec![false; layer_sizes.len()];
    let mut pinned_bytes = 0usize;

    for &(idx, sz) in &order {
        let next_max_unpinned = order
            .iter()
            .filter(|&&(i, _)| !is_pinned[i] && i != idx)
            .map(|&(_, s)| s)
            .max()
            .unwrap_or(0);
        // Keep at least 1 ring slot if any layers remain unpinned.
        if pinned_bytes + sz + next_max_unpinned <= budget_bytes {
            is_pinned[idx] = true;
            pinned_bytes += sz;
        }
    }

    let pinned_layers: Vec<usize> = (0..layer_sizes.len()).filter(|&i| is_pinned[i]).collect();
    let streaming_layers: Vec<usize> = (0..layer_sizes.len()).filter(|&i| !is_pinned[i]).collect();
    let slot_bytes = streaming_layers.iter().map(|&i| layer_sizes[i]).max().unwrap_or(0);
    let remaining = budget_bytes.saturating_sub(pinned_bytes);
    let ring_slots = if slot_bytes == 0 {
        0
    } else if remaining >= slot_bytes.saturating_mul(2) {
        2
    } else {
        1
    };

    TrunkPlan {
        pinned_layers,
        streaming_layers,
        slot_bytes,
        ring_slots,
        pinned_bytes,
    }
}

pub struct TrunkRing {
    slots: Vec<Vec<u8>>,
    slot_layer: Vec<Option<usize>>,
    cursor: usize,
}

impl TrunkRing {
    pub fn new(ring_slots: usize, slot_bytes: usize) -> Self {
        let n = ring_slots.clamp(1, 2);
        Self {
            slots: vec![vec![0u8; slot_bytes]; n],
            slot_layer: vec![None; n],
            cursor: 0,
        }
    }

    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Prefetch `layer_idx` from `file` into the next ring slot.
    pub fn prefetch_layer(&mut self, file: &mut File, layer_idx: usize, offset: u64, len: usize) -> io::Result<usize> {
        let slot_idx = self.cursor % self.slots.len();
        if self.slot_layer[slot_idx] == Some(layer_idx) {
            return Ok(slot_idx);
        }
        if self.slots[slot_idx].len() < len {
            self.slots[slot_idx].resize(len, 0);
        }
        file.seek(SeekFrom::Start(offset))?;
        file.read_exact(&mut self.slots[slot_idx][..len])?;
        self.slot_layer[slot_idx] = Some(layer_idx);
        self.cursor = (slot_idx + 1) % self.slots.len();
        Ok(slot_idx)
    }

    pub fn get_layer(&self, layer_idx: usize, len: usize) -> Option<&[u8]> {
        self.slot_layer
            .iter()
            .position(|&l| l == Some(layer_idx))
            .map(|idx| &self.slots[idx][..len.min(self.slots[idx].len())])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn largest_first_pinning_shrinks_ring_slot_size() {
        // Fat layer 0 (200) + three small layers (100 each) with budget 400:
        // Pinning layer 0 first shrinks slot_bytes from 200 -> 100, allowing layer 1 to pin too.
        let plan = solve_pinned_layers(&[200, 100, 100, 100], 400);
        assert_eq!(plan.pinned_layers, vec![0, 1]);
        assert_eq!(plan.slot_bytes, 100);
        assert_eq!(plan.ring_slots, 1);
    }

    #[test]
    fn two_slot_ring_holds_current_and_prefetched_layer() {
        let path = std::env::temp_dir().join(format!("leafcutter_trunk_{}.bin", std::process::id()));
        {
            let mut f = File::create(&path).unwrap();
            f.write_all(b"LAYER_00LAYER_01LAYER_02").unwrap();
        }
        let mut f = File::open(&path).unwrap();
        let mut ring = TrunkRing::new(2, 8);
        ring.prefetch_layer(&mut f, 0, 0, 8).unwrap();
        ring.prefetch_layer(&mut f, 1, 8, 8).unwrap();
        assert_eq!(ring.get_layer(0, 8).unwrap(), b"LAYER_00");
        assert_eq!(ring.get_layer(1, 8).unwrap(), b"LAYER_01");
        let _ = std::fs::remove_file(&path);
    }
}
