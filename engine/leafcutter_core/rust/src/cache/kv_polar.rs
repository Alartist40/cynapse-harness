//! PolarQuant data-oblivious KV cache quantization (inspired by `colibri/c/kv_tq.h`).
//!
//! 1) Randomized Hadamard rotation (deterministic sign flips + normalized self-inverse FWHT)
//!    makes coordinates near-Gaussian without calibration.
//! 2) Recursive pairwise polar transform packs `(radius, n - 1 angles)` into compact bytes,
//!    reducing KV cache memory by 4–8x for constrained-RAM targets.

use std::f32::consts::PI;

pub const DEFAULT_POLAR_SEED: u32 = 0x9E37_79B9;

#[inline]
fn splitmix32(mut x: u32) -> u32 {
    x = x.wrapping_add(0x9E37_79B9);
    x = (x ^ (x >> 16)).wrapping_mul(0x85EB_CA6B);
    x = (x ^ (x >> 13)).wrapping_mul(0xC2B2_AE35);
    x ^ (x >> 16)
}

#[inline]
fn hadamard_sign(idx: usize, seed: u32) -> f32 {
    if (splitmix32(seed ^ (idx as u32)) & 1) != 0 { -1.0 } else { 1.0 }
}

/// Normalized Fast Walsh-Hadamard Transform in place (`n` must be a power of 2).
pub fn fwht_inplace(a: &mut [f32]) {
    let n = a.len();
    debug_assert!(n.is_power_of_two());
    let mut len = 1;
    while len < n {
        let step = len << 1;
        for i in (0..n).step_by(step) {
            for j in i..(i + len) {
                let u = a[j];
                let v = a[j + len];
                a[j] = u + v;
                a[j + len] = u - v;
            }
        }
        len <<= 1;
    }
    let inv = 1.0 / (n as f32).sqrt();
    for v in a.iter_mut() {
        *v *= inv;
    }
}

pub fn rotate_hadamard(x: &[f32], seed: u32) -> Vec<f32> {
    let mut y: Vec<f32> = x.iter().enumerate().map(|(i, &v)| v * hadamard_sign(i, seed)).collect();
    fwht_inplace(&mut y);
    y
}

pub fn unrotate_hadamard(y: &mut [f32], seed: u32) {
    fwht_inplace(y);
    for (i, v) in y.iter_mut().enumerate() {
        *v *= hadamard_sign(i, seed);
    }
}

#[derive(Clone, Debug)]
pub struct PolarRow {
    pub n: usize,
    pub bits: u8,
    pub radius: f32,
    pub packed_angles: Vec<u8>,
}

impl PolarRow {
    pub fn memory_bytes(&self) -> usize {
        std::mem::size_of::<f32>() + self.packed_angles.len()
    }
}

fn push_bits(buf: &mut Vec<u8>, bit_pos: &mut usize, val: u32, bits: u8) {
    for b in 0..(bits as usize) {
        let byte_idx = *bit_pos / 8;
        let bit_off = *bit_pos % 8;
        if byte_idx >= buf.len() {
            buf.push(0);
        }
        if ((val >> b) & 1) != 0 {
            buf[byte_idx] |= 1 << bit_off;
        }
        *bit_pos += 1;
    }
}

fn read_bits(buf: &[u8], bit_pos: &mut usize, bits: u8) -> u32 {
    let mut out = 0u32;
    for b in 0..(bits as usize) {
        let byte_idx = *bit_pos / 8;
        let bit_off = *bit_pos % 8;
        if byte_idx < buf.len() && ((buf[byte_idx] >> bit_off) & 1) != 0 {
            out |= 1 << b;
        }
        *bit_pos += 1;
    }
    out
}

pub fn quantize_polar_row(row: &[f32], bits: u8, seed: u32) -> PolarRow {
    let n = row.len();
    assert!(n.is_power_of_two() && n >= 2, "PolarQuant requires power-of-two width >= 2");
    let bits = bits.clamp(2, 8);
    let mut cur = rotate_hadamard(row, seed);
    let mut packed = Vec::with_capacity(n);
    let mut bit_pos = 0usize;

    let mut level = 1usize;
    let mut width = n;
    while width > 1 {
        let half = width / 2;
        let mut next = Vec::with_capacity(half);
        let b = if level == 1 { bits } else { bits.saturating_sub(1).max(1) };
        let max_q = (1u32 << b) - 1;

        for i in 0..half {
            let x = cur[2 * i];
            let y = cur[2 * i + 1];
            let r = x.hypot(y);
            let ang = y.atan2(x);
            let u = if level == 1 {
                ((ang + PI) / (2.0 * PI)).clamp(0.0, 0.999_999)
            } else {
                // Deeper levels lie in [0, pi/2], centered around pi/4.
                (ang / (PI * 0.5)).clamp(0.0, 0.999_999)
            };
            let q = (u * max_q as f32).round() as u32;
            push_bits(&mut packed, &mut bit_pos, q.min(max_q), b);
            next.push(r);
        }
        cur = next;
        width = half;
        level += 1;
    }

    PolarRow {
        n,
        bits,
        radius: cur[0],
        packed_angles: packed,
    }
}

pub fn dequantize_polar_row(q: &PolarRow, seed: u32) -> Vec<f32> {
    let mut levels: Vec<(usize, u8, Vec<f32>)> = Vec::new();
    let mut bit_pos = 0usize;
    let mut width = q.n;
    let mut level = 1usize;

    while width > 1 {
        let half = width / 2;
        let b = if level == 1 { q.bits } else { q.bits.saturating_sub(1).max(1) };
        let max_q = ((1u32 << b) - 1) as f32;
        let mut angs = Vec::with_capacity(half);
        for _ in 0..half {
            let code = read_bits(&q.packed_angles, &mut bit_pos, b) as f32;
            let u = code / max_q;
            let ang = if level == 1 {
                u * (2.0 * PI) - PI
            } else {
                u * (PI * 0.5)
            };
            angs.push(ang);
        }
        levels.push((half, b, angs));
        width = half;
        level += 1;
    }

    let mut cur = vec![q.radius];
    for (_, _, angs) in levels.into_iter().rev() {
        let mut next = Vec::with_capacity(angs.len() * 2);
        for (r, ang) in cur.into_iter().zip(angs.into_iter()) {
            next.push(r * ang.cos());
            next.push(r * ang.sin());
        }
        cur = next;
    }

    unrotate_hadamard(&mut cur, seed);
    cur
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fwht_rotation_is_self_inverse() {
        let x: Vec<f32> = (0..64).map(|i| (i as f32 * 0.13).sin()).collect();
        let mut y = rotate_hadamard(&x, DEFAULT_POLAR_SEED);
        unrotate_hadamard(&mut y, DEFAULT_POLAR_SEED);
        for (a, b) in x.iter().zip(y.iter()) {
            assert!((a - b).abs() < 1e-4, "mismatch {a} vs {b}");
        }
    }

    #[test]
    fn polar_quant_4bit_preserves_cosine_similarity_and_compresses() {
        let x: Vec<f32> = (0..128).map(|i| ((i as f32) * 0.37).cos() * 1.7).collect();
        let q = quantize_polar_row(&x, 4, DEFAULT_POLAR_SEED);
        let recon = dequantize_polar_row(&q, DEFAULT_POLAR_SEED);

        let dot: f32 = x.iter().zip(recon.iter()).map(|(a, b)| a * b).sum();
        let nx: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let nr: f32 = recon.iter().map(|v| v * v).sum::<f32>().sqrt();
        let cos = dot / (nx * nr);

        assert!(cos > 0.95, "cosine similarity too low: {cos}");
        assert!(q.memory_bytes() < x.len() * 4 / 4, "expected >4x compression");
    }
}
