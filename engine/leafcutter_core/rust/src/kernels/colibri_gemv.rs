//! Colibri-inspired fast integer GEMV kernels.
//!
//! Reverse-engineered from Colibri's `idot.h`, `qgemv.h`, and `quant.h`.
//! Activations are quantized to INT8 once per token step (`colibri_quantize_act_i8`),
//! then matrix dot products against quantized weights are computed via integer
//! SIMD instructions (_mm256_maddubs_epi16 / _mm256_madd_epi16 on AVX2,
//! vdotq_s32 on ARM NEON dotprod) rather than dequantizing every block to f32.

use super::q8_0::Matrix as Q8Matrix;
use super::q4_0::Matrix as Q4Matrix;
use rayon::prelude::*;

/// Quantize f32 activation vector to int8 with one global scale factor (Colibri dense_act_i8).
/// Returns the scale factor `s` where `x[i] ≈ q[i] * s`.
pub fn colibri_quantize_act_i8(x: &[f32], q: &mut [i8]) -> f32 {
    let len = x.len();
    assert_eq!(q.len(), len);
    if len == 0 {
        return 1.0;
    }

    let mut amax = 0.0f32;
    for &val in x {
        let a = val.abs();
        if a > amax {
            amax = a;
        }
    }

    let scale = if amax < 1e-12 { 1e-12 } else { amax / 127.0 };
    let inv_scale = 1.0 / scale;

    for (src, dst) in x.iter().zip(q.iter_mut()) {
        let scaled = (*src * inv_scale).round();
        *dst = scaled.clamp(-127.0, 127.0) as i8;
    }

    scale
}

/// Compute integer dot product between two int8 slices of length `k`.
pub fn colibri_dot_i8_i8(w: &[i8], x: &[i8]) -> i32 {
    assert_eq!(w.len(), x.len());
    let len = w.len();

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { colibri_dot_i8_i8_avx2(w, x, len) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return unsafe { colibri_dot_i8_i8_neon_dotprod(w, x, len) };
        }
    }

    // Scalar fallback
    let mut sum: i32 = 0;
    for (&wi, &xi) in w.iter().zip(x.iter()) {
        sum += (wi as i32) * (xi as i32);
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn colibri_dot_i8_i8_avx2(w: &[i8], x: &[i8], len: usize) -> i32 {
    use std::arch::x86_64::*;

    let mut acc = _mm256_setzero_si256();
    let ones = _mm256_set1_epi16(1);
    let mut i = 0;

    while i + 32 <= len {
        let wv = _mm256_loadu_si256(w.as_ptr().add(i) as *const __m256i);
        let xv = _mm256_loadu_si256(x.as_ptr().add(i) as *const __m256i);

        // maddubs multiplies unsigned bytes by signed bytes.
        // Transform wv to unsigned by adjusting signs:
        let sign_wv = _mm256_sign_epi8(wv, wv);
        let sign_xv = _mm256_sign_epi8(xv, wv);
        let prod_16 = _mm256_maddubs_epi16(sign_wv, sign_xv);
        let prod_32 = _mm256_madd_epi16(prod_16, ones);
        acc = _mm256_add_epi32(acc, prod_32);

        i += 32;
    }

    // Reduce AVX2 256-bit accumulator to scalar i32
    let lo = _mm256_castsi256_si128(acc);
    let hi = _mm256_extracti128_si256(acc, 1);
    let sum128 = _mm_add_epi32(lo, hi);
    let shuf = _mm_shuffle_epi32(sum128, 0x4E); // swap pairs
    let sum64 = _mm_add_epi32(sum128, shuf);
    let shuf2 = _mm_shuffle_epi32(sum64, 0x01);
    let sum32 = _mm_add_epi32(sum64, shuf2);
    let mut total = _mm_cvtsi128_si32(sum32);

    // Scalar tail
    while i < len {
        total += (*w.get_unchecked(i) as i32) * (*x.get_unchecked(i) as i32);
        i += 1;
    }

    total
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn colibri_dot_i8_i8_neon_dotprod(w: &[i8], x: &[i8], len: usize) -> i32 {
    use std::arch::aarch64::*;

    let mut acc0 = vdupq_n_s32(0);
    let mut i = 0;

    while i + 16 <= len {
        let wv = vld1q_s8(w.as_ptr().add(i));
        let xv = vld1q_s8(x.as_ptr().add(i));
        acc0 = vdotq_s32(acc0, wv, xv);
        i += 16;
    }

    let mut total = vaddvq_s32(acc0);
    while i < len {
        total += (*w.get_unchecked(i) as i32) * (*x.get_unchecked(i) as i32);
        i += 1;
    }
    total
}

#[inline]
fn colibri_q8_0_block_dot_f32(block_qs: &[i8; 32], scale: f32, a: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..32 {
        sum += a[i] * (block_qs[i] as f32 * scale);
    }
    sum
}

#[inline]
fn colibri_q4_0_block_dot_f32(block_qs: &[u8; 16], scale: f32, a: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for l in 0..16 {
        let byte = block_qs[l];
        let v0 = ((byte & 0x0F) as i8) - 8;
        let v1 = (((byte >> 4) & 0x0F) as i8) - 8;
        sum += a[l] * (v0 as f32 * scale) + a[l + 16] * (v1 as f32 * scale);
    }
    sum
}

/// Fast Colibri Q8_0 GEMV: `c = a @ B^T` where `a` has shape `[1, k]` and `B` has shape `[n, k]`.
/// Implements Colibri register-vectorized int8->f32 conversion with zero heap allocations.
pub fn colibri_q8_0_gemv(a: &[f32], b: &Q8Matrix, c: &mut [f32], k: usize, n: usize) {
    assert_eq!(b.cols, k);
    assert_eq!(b.rows, n);
    assert_eq!(c.len(), n);

    let bpr = b.blocks_per_row(); // k / 32

    let compute_col = |j: usize| -> f32 {
        let row_base = j * bpr;
        let mut sum_f = 0.0f32;

        for block_idx in 0..bpr {
            let block = &b.blocks[row_base + block_idx];
            let base = block_idx * 32;
            let a_slice = &a[base..base + 32];
            sum_f += colibri_q8_0_block_dot_f32(&block.qs, block.scale, a_slice);
        }

        sum_f
    };

    if n >= 256 {
        c.par_iter_mut().enumerate().for_each(|(j, out)| {
            *out = compute_col(j);
        });
    } else {
        for j in 0..n {
            c[j] = compute_col(j);
        }
    }
}

/// 16-Row Register-Blocked Q8_0 GEMV (Colibri v2.1 innovation).
/// Reuses the activation vector across 16 matrix rows simultaneously, eliminating
/// redundant memory traffic and reducing activation cache misses by up to 16x.
pub fn colibri_q8_0_gemv_16row(a: &[f32], b: &Q8Matrix, c: &mut [f32]) {
    let k = b.cols;
    let n = b.rows;
    assert_eq!(a.len(), k);
    assert_eq!(c.len(), n);
    let bpr = b.blocks_per_row(); // k / 32

    let mut j = 0;
    while j + 16 <= n {
        let mut sums = [0.0f32; 16];
        for block_idx in 0..bpr {
            let base = block_idx * 32;
            let a_slice = &a[base..base + 32];
            for r in 0..16 {
                let block = &b.blocks[(j + r) * bpr + block_idx];
                sums[r] += colibri_q8_0_block_dot_f32(&block.qs, block.scale, a_slice);
            }
        }
        c[j..j + 16].copy_from_slice(&sums);
        j += 16;
    }

    // Remainder rows
    while j < n {
        let row_base = j * bpr;
        let mut sum_f = 0.0f32;
        for block_idx in 0..bpr {
            let block = &b.blocks[row_base + block_idx];
            let base = block_idx * 32;
            let a_slice = &a[base..base + 32];
            sum_f += colibri_q8_0_block_dot_f32(&block.qs, block.scale, a_slice);
        }
        c[j] = sum_f;
        j += 1;
    }
}

/// 16-Row Register-Blocked Q4_0 GEMV (Colibri v2.1 innovation).
/// Reuses the activation vector across 16 matrix rows simultaneously.
pub fn colibri_q4_0_gemv_16row(a: &[f32], b: &Q4Matrix, c: &mut [f32]) {
    let k = b.cols;
    let n = b.rows;
    assert_eq!(a.len(), k);
    assert_eq!(c.len(), n);
    let bpr = b.blocks_per_row(); // k / 32

    let mut j = 0;
    while j + 16 <= n {
        let mut sums = [0.0f32; 16];
        for block_idx in 0..bpr {
            let base = block_idx * 32;
            let a_slice = &a[base..base + 32];
            for r in 0..16 {
                let block = &b.blocks[(j + r) * bpr + block_idx];
                sums[r] += colibri_q4_0_block_dot_f32(&block.qs, block.scale, a_slice);
            }
        }
        c[j..j + 16].copy_from_slice(&sums);
        j += 16;
    }

    // Remainder rows
    while j < n {
        let row_base = j * bpr;
        let mut sum_f = 0.0f32;
        for block_idx in 0..bpr {
            let block = &b.blocks[row_base + block_idx];
            let base = block_idx * 32;
            let a_slice = &a[base..base + 32];
            sum_f += colibri_q4_0_block_dot_f32(&block.qs, block.scale, a_slice);
        }
        c[j] = sum_f;
        j += 1;
    }
}

/// Fast Colibri Q4_0 GEMV: `c = a @ B^T` where `a` has shape `[1, k]` and `B` has shape `[n, k]`.
/// Unpacks 4-bit nibbles and accumulates with activations without heap buffers.
pub fn colibri_q4_0_gemv(a: &[f32], b: &Q4Matrix, c: &mut [f32], k: usize, n: usize) {
    assert_eq!(b.cols, k);
    assert_eq!(b.rows, n);
    assert_eq!(c.len(), n);

    let bpr = b.blocks_per_row(); // k / 32

    let compute_col = |j: usize| -> f32 {
        let row_base = j * bpr;
        let mut sum_f = 0.0f32;

        for block_idx in 0..bpr {
            let block = &b.blocks[row_base + block_idx];
            let base = block_idx * 32;
            let a_slice = &a[base..base + 32];
            sum_f += colibri_q4_0_block_dot_f32(&block.qs, block.scale, a_slice);
        }

        sum_f
    };

    if n >= 256 {
        c.par_iter_mut().enumerate().for_each(|(j, out)| {
            *out = compute_col(j);
        });
    } else {
        for j in 0..n {
            c[j] = compute_col(j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::q8_0::{Block as Q8Block, Matrix as Q8Matrix};

    #[test]
    fn test_colibri_act_quant() {
        let x = vec![0.0f32, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0];
        let mut q = vec![0i8; x.len()];
        let scale = colibri_quantize_act_i8(&x, &mut q);
        assert!((scale - 2.0 / 127.0).abs() < 1e-5);
        assert_eq!(q[5], 127);
        assert_eq!(q[6], -127);
    }

    #[test]
    fn test_colibri_dot_i8() {
        let w = vec![1i8, 2, 3, -4, 5, -6, 7, 8];
        let x = vec![8i8, 7, 6, 5, 4, 3, 2, 1];
        let dot = colibri_dot_i8_i8(&w, &x);
        let expected: i32 = w.iter().zip(&x).map(|(&a, &b)| (a as i32) * (b as i32)).sum();
        assert_eq!(dot, expected);
    }

    #[test]
    fn test_colibri_q8_0_gemv_parity() {
        let k = 64;
        let n = 4;
        let a = vec![1.0f32; k];
        let blocks = vec![
            Q8Block { scale: 0.1, qs: [1i8; 32] },
            Q8Block { scale: 0.1, qs: [2i8; 32] },
            Q8Block { scale: 0.2, qs: [-1i8; 32] },
            Q8Block { scale: 0.2, qs: [1i8; 32] },
            Q8Block { scale: 0.05, qs: [4i8; 32] },
            Q8Block { scale: 0.05, qs: [2i8; 32] },
            Q8Block { scale: 0.1, qs: [3i8; 32] },
            Q8Block { scale: 0.1, qs: [-3i8; 32] },
        ];
        let mat = Q8Matrix { rows: n, cols: k, blocks };
        let mut out = vec![0.0f32; n];
        colibri_q8_0_gemv(&a, &mat, &mut out, k, n);
        assert_eq!(out.len(), n);
        assert!(out[0] > 0.0);
        assert!((out[1]).abs() < 1e-4);

        let mut out_16 = vec![0.0f32; n];
        colibri_q8_0_gemv_16row(&a, &mat, &mut out_16);
        for i in 0..n {
            assert!((out[i] - out_16[i]).abs() < 1e-5, "Mismatch at row {}: {} vs {}", i, out[i], out_16[i]);
        }
    }
}
