//! Pure-Rust Zero-Backprop Neuron Surgery & Control Vector Synthesizer.
//!
//! Reconstructed from `gguf-neuron-explorer` (`gne`) with zero Python or external dependencies:
//! 1. Contrast direction extraction & equal-norm random-direction null controls.
//! 2. CHiPS (Causal Head-isolated Projection Steering) JS-divergence & cosine ranking.
//! 3. Directional ablation (`W' = (I - s * r * r^T) W`) & exact Rank-1 LoRA (`A = r^T W`, `B = -s * r`).
//! 4. Native GGUF v3 `.cvec.gguf` writer compatible with `llama-server --control-vector`.

use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_VERSION: u32 = 3;
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_STRING: u32 = 8;
const GGML_TYPE_F32: u32 = 0;

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[inline]
fn norm_l2(v: &[f32]) -> f32 {
    dot(v, v).sqrt()
}

#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn random_unit_vec(dim: usize, seed: &mut u64) -> Vec<f32> {
    let mut out = Vec::with_capacity(dim);
    while out.len() < dim {
        let u1 = ((splitmix64(seed) >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0);
        let u2 = ((splitmix64(seed) >> 11) as f64) / ((1u64 << 53) as f64);
        let mag = (-2.0 * u1.ln()).sqrt();
        let angle = 2.0 * std::f64::consts::PI * u2;
        out.push((mag * angle.cos()) as f32);
        if out.len() < dim {
            out.push((mag * angle.sin()) as f32);
        }
    }
    let n = norm_l2(&out).max(1e-12);
    for v in &mut out {
        *v /= n;
    }
    out
}

/// Extract unit contrast direction `(r, norm)` from positive vs negative residual activations.
pub fn compute_contrast_direction(pos_acts: &[&[f32]], neg_acts: &[&[f32]]) -> (Vec<f32>, f32) {
    let Some(first) = pos_acts.first().or_else(|| neg_acts.first()) else {
        return (vec![], 0.0);
    };
    let dim = first.len();
    let mut diff = vec![0.0f32; dim];
    if !pos_acts.is_empty() {
        let inv = 1.0 / pos_acts.len() as f32;
        for row in pos_acts {
            for (i, &v) in row.iter().enumerate().take(dim) {
                diff[i] += v * inv;
            }
        }
    }
    if !neg_acts.is_empty() {
        let inv = 1.0 / neg_acts.len() as f32;
        for row in neg_acts {
            for (i, &v) in row.iter().enumerate().take(dim) {
                diff[i] -= v * inv;
            }
        }
    }
    let n = norm_l2(&diff);
    if n < 1e-12 {
        return (vec![0.0; dim], 0.0);
    }
    for v in &mut diff {
        *v /= n;
    }
    (diff, n)
}

/// Fraction of `n_controls` equal-norm random directions beaten by `direction` on `probe_delta`.
pub fn null_control_percentile(
    direction: &[f32],
    probe_delta: &[f32],
    n_controls: usize,
    seed: u64,
) -> f32 {
    let d_norm = norm_l2(direction);
    if d_norm < 1e-12 || direction.len() != probe_delta.len() {
        return 0.0;
    }
    let target = (dot(direction, probe_delta) / d_norm).abs();
    let mut rng_state = seed;
    let total = n_controls.max(1);
    let mut beaten = 0usize;
    for _ in 0..total {
        let rand_dir = random_unit_vec(direction.len(), &mut rng_state);
        if target > dot(&rand_dir, probe_delta).abs() {
            beaten += 1;
        }
    }
    beaten as f32 / total as f32
}

/// Row-wise Jensen-Shannon divergence in bits `[0, 1]` for CHiPS `pattern_shift`.
pub fn js_divergence_bits(p: &[f32], q: &[f32]) -> f32 {
    if p.is_empty() || p.len() != q.len() {
        return 0.0;
    }
    let sum_p: f64 = p.iter().map(|&x| (x as f64).max(1e-30)).sum();
    let sum_q: f64 = q.iter().map(|&x| (x as f64).max(1e-30)).sum();
    let mut js = 0.0f64;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        let a = (pi as f64).max(1e-30) / sum_p;
        let b = (qi as f64).max(1e-30) / sum_q;
        let m = 0.5 * (a + b);
        js += 0.5 * (a * (a / m).log2() + b * (b / m).log2());
    }
    js.clamp(0.0, 1.0) as f32
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChipsHeadScore {
    pub head: usize,
    pub pattern_shift: f32,
    pub out_shift: f32,
    pub dir_cos: f32,
    pub chips_score: f32,
}

/// Rank attention heads using CHiPS (`pattern_shift + out_shift + dir_cos`).
pub fn score_chips_heads(
    attn_a: &[&[f32]],
    attn_b: &[&[f32]],
    out_a: &[&[f32]],
    out_b: &[&[f32]],
    direction: &[f32],
) -> Vec<ChipsHeadScore> {
    let r_norm = norm_l2(direction).max(1e-12);
    let mut out = Vec::new();
    for (head, (((&pa, &pb), &oa), &ob)) in attn_a
        .iter()
        .zip(attn_b.iter())
        .zip(out_a.iter())
        .zip(out_b.iter())
        .enumerate()
    {
        let pattern_shift = js_divergence_bits(pa, pb);
        let delta: Vec<f32> = oa.iter().zip(ob.iter()).map(|(a, b)| a - b).collect();
        let base_norm = norm_l2(ob).max(1e-8);
        let delta_norm = norm_l2(&delta);
        let out_shift = delta_norm / base_norm;
        let dir_cos = if delta_norm > 1e-12 && delta.len() == direction.len() {
            (dot(&delta, direction) / (delta_norm * r_norm)).abs()
        } else {
            0.0
        };
        let chips_score = pattern_shift + out_shift + dir_cos;
        out.push(ChipsHeadScore {
            head,
            pattern_shift,
            out_shift,
            dir_cos,
            chips_score,
        });
    }
    out.sort_by(|a, b| b.chips_score.partial_cmp(&a.chips_score).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// Directional ablation: `W' = W - scale * r * (r^T W)` for row-major `W[rows][cols]`.
pub fn project_out_direction(weight: &[f32], rows: usize, cols: usize, direction: &[f32], scale: f32) -> Vec<f32> {
    assert_eq!(weight.len(), rows * cols);
    assert_eq!(direction.len(), rows);
    let r_norm = norm_l2(direction).max(1e-12);
    let r: Vec<f32> = direction.iter().map(|&v| v / r_norm).collect();
    let mut r_t_w = vec![0.0f32; cols];
    for row in 0..rows {
        let ri = r[row];
        let base = row * cols;
        for col in 0..cols {
            r_t_w[col] += ri * weight[base + col];
        }
    }
    let mut out = weight.to_vec();
    for row in 0..rows {
        let sr = scale * r[row];
        let base = row * cols;
        for col in 0..cols {
            out[base + col] -= sr * r_t_w[col];
        }
    }
    out
}

/// Exact Rank-1 LoRA decomposition of directional ablation: `A = r^T W` `(1, cols)`, `B = -scale * r` `(rows, 1)`.
pub fn rank1_ablate_lora(weight: &[f32], rows: usize, cols: usize, direction: &[f32], scale: f32) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(weight.len(), rows * cols);
    assert_eq!(direction.len(), rows);
    let r_norm = norm_l2(direction).max(1e-12);
    let r: Vec<f32> = direction.iter().map(|&v| v / r_norm).collect();
    let mut lora_a = vec![0.0f32; cols];
    for row in 0..rows {
        let ri = r[row];
        let base = row * cols;
        for col in 0..cols {
            lora_a[col] += ri * weight[base + col];
        }
    }
    let lora_b: Vec<f32> = r.iter().map(|&ri| -scale * ri).collect();
    (lora_a, lora_b)
}

fn push_gguf_str(buf: &mut Vec<u8>, s: &str) {
    let b = s.as_bytes();
    buf.extend_from_slice(&(b.len() as u64).to_le_bytes());
    buf.extend_from_slice(b);
}

#[inline]
fn align_up(n: usize, align: usize) -> usize {
    ((n + align - 1) / align) * align
}

/// Export layer steering vectors as a GGUF v3 `.cvec.gguf` file for `llama-server --control-vector`.
pub fn export_cvec_gguf(out_path: &Path, arch: &str, layer_vectors: &[(usize, Vec<f32>)]) -> io::Result<()> {
    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut hdr = Vec::new();
    hdr.extend_from_slice(GGUF_MAGIC);
    hdr.extend_from_slice(&GGUF_VERSION.to_le_bytes());
    hdr.extend_from_slice(&(layer_vectors.len() as u64).to_le_bytes());
    hdr.extend_from_slice(&2u64.to_le_bytes()); // 2 KV entries

    push_gguf_str(&mut hdr, "general.architecture");
    hdr.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
    push_gguf_str(&mut hdr, arch);

    push_gguf_str(&mut hdr, "control_vector.layer_count");
    hdr.extend_from_slice(&GGUF_TYPE_UINT32.to_le_bytes());
    hdr.extend_from_slice(&(layer_vectors.len() as u32).to_le_bytes());

    let mut data_offset = 0u64;
    let mut blobs = Vec::with_capacity(layer_vectors.len());
    for (layer, vec) in layer_vectors {
        push_gguf_str(&mut hdr, &format!("direction.{layer}"));
        hdr.extend_from_slice(&1u32.to_le_bytes()); // n_dims
        hdr.extend_from_slice(&(vec.len() as u64).to_le_bytes());
        hdr.extend_from_slice(&GGML_TYPE_F32.to_le_bytes());
        hdr.extend_from_slice(&data_offset.to_le_bytes());

        let mut raw = Vec::with_capacity(vec.len() * 4);
        for &f in vec {
            raw.extend_from_slice(&f.to_le_bytes());
        }
        let padded = align_up(raw.len(), 32);
        raw.resize(padded, 0);
        data_offset += padded as u64;
        blobs.push(raw);
    }

    let hdr_padded = align_up(hdr.len(), 32);
    hdr.resize(hdr_padded, 0);

    let mut file = File::create(out_path)?;
    file.write_all(&hdr)?;
    for blob in blobs {
        file.write_all(&blob)?;
    }
    Ok(())
}

/// Internal weight summary for Teacher-guided inspection (`2611.21885v2` Step 1).
#[derive(Debug, Clone)]
pub struct GgufWeightDiagnostic {
    pub arch: String,
    pub tensor_count: usize,
    pub residual_writers: Vec<String>,
}

/// Inspect a local GGUF file's internal tensor table without loading weights into RAM.
pub fn inspect_gguf_weights(path: &Path) -> Result<GgufWeightDiagnostic, String> {
    let gguf = crate::model::gguf::GGUFile::open(path).map_err(|e| format!("GGUF open failed: {e}"))?;
    let arch = match gguf.metadata.get("general.architecture") {
        Some(crate::model::gguf::GGUFValue::String(s)) => s.clone(),
        _ => "unknown".to_string(),
    };
    let residual_writers = gguf
        .tensors
        .iter()
        .filter(|t| {
            t.name.ends_with("attn_output.weight")
                || t.name.ends_with("ffn_down.weight")
                || t.name.ends_with("ssm_out.weight")
                || t.name.ends_with("out_proj.weight")
        })
        .map(|t| t.name.clone())
        .collect();
    Ok(GgufWeightDiagnostic {
        arch,
        tensor_count: gguf.tensors.len(),
        residual_writers,
    })
}

/// Outcome of the held-out validation gate (`2611.21885v2`: keep only edits that pass tests).
#[derive(Debug, Clone, PartialEq)]
pub enum EditGateOutcome {
    Committed { null_pct: f32 },
    Withdrawn { reason: String },
}

/// Auto-withdraw gate: writes `.cvec.gguf` ONLY when `null_pct >= min_null_pct` and `val_after >= val_before`.
pub fn evaluate_and_commit_cvec(
    out_path: &Path,
    arch: &str,
    layer_vectors: &[(usize, Vec<f32>)],
    null_pct: f32,
    min_null_pct: f32,
    val_before: f32,
    val_after: f32,
) -> io::Result<EditGateOutcome> {
    if null_pct < min_null_pct {
        return Ok(EditGateOutcome::Withdrawn {
            reason: format!("null_pct {:.0}% < {:.0}% threshold", null_pct * 100.0, min_null_pct * 100.0),
        });
    }
    if val_after < val_before {
        return Ok(EditGateOutcome::Withdrawn {
            reason: format!("held-out regression ({:.3} < {:.3})", val_after, val_before),
        });
    }
    export_cvec_gguf(out_path, arch, layer_vectors)?;
    Ok(EditGateOutcome::Committed { null_pct })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank1_ablation_lora_matches_projection_and_zeroes_direction() {
        let rows = 8;
        let cols = 6;
        let w: Vec<f32> = (0..(rows * cols)).map(|i| ((i as f32) * 0.31).sin()).collect();
        let pos = [1.0f32; 8];
        let neg = [-1.0f32; 8];
        let (r, norm) = compute_contrast_direction(&[&pos], &[&neg]);
        assert!(norm > 1.0);

        let w_proj = project_out_direction(&w, rows, cols, &r, 1.0);
        let (lora_a, lora_b) = rank1_ablate_lora(&w, rows, cols, &r, 1.0);
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 + 1.0) * 0.5).collect();

        let mut out_proj = vec![0.0f32; rows];
        let mut out_lora = vec![0.0f32; rows];
        let a_dot_x = dot(&lora_a, &x);
        for row in 0..rows {
            out_proj[row] = dot(&w_proj[row * cols..(row + 1) * cols], &x);
            out_lora[row] = dot(&w[row * cols..(row + 1) * cols], &x) + lora_b[row] * a_dot_x;
            assert!((out_proj[row] - out_lora[row]).abs() < 1e-5);
        }
        assert!(dot(&out_proj, &r).abs() < 1e-5);
    }

    #[test]
    fn chips_scoring_null_control_and_cvec_export() {
        let dir = [1.0f32, 0.0, 0.0, 0.0];
        let delta = [2.0f32, 0.0, 0.0, 0.0];
        assert!(null_control_percentile(&dir, &delta, 16, 7919) >= 0.9);

        let heads = score_chips_heads(
            &[&[0.9, 0.1], &[0.5, 0.5]],
            &[&[0.1, 0.9], &[0.5, 0.5]],
            &[&[2.0, 0.0, 0.0, 0.0], &[0.1, 0.1, 0.1, 0.1]],
            &[&[0.1, 0.0, 0.0, 0.0], &[0.1, 0.1, 0.1, 0.1]],
            &dir,
        );
        assert_eq!(heads[0].head, 0);

        let path = std::env::temp_dir().join(format!("leafcutter_test_{}.cvec.gguf", std::process::id()));
        let res = evaluate_and_commit_cvec(&path, "qwen3", &[(14, dir.to_vec()), (22, dir.to_vec())], 0.95, 0.90, 0.50, 0.62).unwrap();
        assert_eq!(res, EditGateOutcome::Committed { null_pct: 0.95 });
        let diag = inspect_gguf_weights(&path).unwrap();
        assert_eq!(diag.arch, "qwen3");
        assert_eq!(diag.tensor_count, 2);
        let _ = fs::remove_file(&path);

        let rej_path = std::env::temp_dir().join(format!("leafcutter_rej_{}.cvec.gguf", std::process::id()));
        let rej = evaluate_and_commit_cvec(&rej_path, "qwen3", &[(14, dir.to_vec())], 0.95, 0.90, 0.70, 0.60).unwrap();
        assert!(matches!(rej, EditGateOutcome::Withdrawn { .. }));
        assert!(!rej_path.exists());
    }
}
