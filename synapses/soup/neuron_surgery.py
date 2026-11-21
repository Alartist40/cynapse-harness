#!/usr/bin/env python3
"""Zero-backprop Neuron Surgery & Control Vector Synthesizer for Cynapse Soup.

Reconstructed from gguf-neuron-explorer (gne) core mechanisms:
1. Contrast direction extraction & equal-norm random-direction null controls.
2. CHiPS (Causal Head-isolated Projection Steering) JS-divergence & cosine ranking.
3. Directional ablation (`W' = (I - s * r * r^T) W`) & Rank-1 LoRA synthesis (`A = r^T W`, `B = -s * r`).
4. Native `.cvec.gguf` control vector writer compatible with `llama-server --control-vector`.
"""

from __future__ import annotations

import math
import os
import struct
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_STRING = 8
GGML_TYPE_F32 = 0


def compute_contrast_direction(pos_acts: np.ndarray, neg_acts: np.ndarray) -> Tuple[np.ndarray, float]:
    """Return unit contrast direction `(r, norm)` from `(N, d)` positive/negative residual activations."""
    diff = np.asarray(pos_acts, dtype=np.float32).mean(axis=0) - np.asarray(neg_acts, dtype=np.float32).mean(axis=0)
    norm = float(np.linalg.norm(diff))
    if norm < 1e-12:
        return np.zeros_like(diff, dtype=np.float32), 0.0
    return (diff / norm).astype(np.float32), norm


def null_control_percentile(
    direction: np.ndarray,
    probe_delta: np.ndarray,
    n_controls: int = 16,
    seed: int = 7919,
) -> float:
    """Fraction of equal-norm random directions beaten by `direction` on `probe_delta`."""
    rng = np.random.default_rng(seed)
    unit = np.asarray(direction, dtype=np.float32)
    u_norm = float(np.linalg.norm(unit))
    if u_norm < 1e-12:
        return 0.0
    unit = unit / u_norm
    target_score = abs(float(np.dot(unit, probe_delta)))
    beaten = 0
    for _ in range(max(1, n_controls)):
        rand_v = rng.normal(size=unit.shape).astype(np.float32)
        rand_v /= max(float(np.linalg.norm(rand_v)), 1e-12)
        if target_score > abs(float(np.dot(rand_v, probe_delta))):
            beaten += 1
    return beaten / float(max(1, n_controls))


def js_divergence_bits(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Row-wise Jensen-Shannon divergence in bits [0, 1] for CHiPS pattern_shift."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-30, 1.0)
    q = np.clip(np.asarray(q, dtype=np.float64), 1e-30, 1.0)
    p /= p.sum(axis=-1, keepdims=True)
    q /= q.sum(axis=-1, keepdims=True)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log2(p) - np.log2(m)), axis=-1)
    kl_qm = np.sum(q * (np.log2(q) - np.log2(m)), axis=-1)
    return np.clip(0.5 * (kl_pm + kl_qm), 0.0, 1.0)


def score_chips_heads(
    attn_a: Sequence[np.ndarray],
    attn_b: Sequence[np.ndarray],
    out_a: Sequence[np.ndarray],
    out_b: Sequence[np.ndarray],
    direction: np.ndarray,
) -> List[Dict[str, float]]:
    """Rank attention heads by CHiPS metrics: pattern_shift, out_shift, and dir_cos."""
    r = np.asarray(direction, dtype=np.float32)
    r_norm = max(float(np.linalg.norm(r)), 1e-12)
    r_unit = r / r_norm

    scored: List[Dict[str, float]] = []
    for h_idx, (pa, pb, oa, ob) in enumerate(zip(attn_a, attn_b, out_a, out_b)):
        pattern_shift = float(np.mean(js_divergence_bits(pa, pb)))
        delta_out = np.asarray(oa, dtype=np.float32) - np.asarray(ob, dtype=np.float32)
        base_norm = max(float(np.mean(np.linalg.norm(ob, axis=-1))), 1e-8)
        out_shift = float(np.linalg.norm(delta_out.mean(axis=0))) / base_norm
        mean_delta = delta_out.mean(axis=0)
        d_norm = max(float(np.linalg.norm(mean_delta)), 1e-12)
        dir_cos = abs(float(np.dot(mean_delta / d_norm, r_unit)))
        chips_score = pattern_shift + out_shift + dir_cos
        scored.append({
            "head": float(h_idx),
            "pattern_shift": pattern_shift,
            "out_shift": out_shift,
            "dir_cos": dir_cos,
            "chips_score": chips_score,
        })
    scored.sort(key=lambda item: item["chips_score"], reverse=True)
    return scored


def project_out_direction(weight: np.ndarray, direction: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Directional ablation: W' = W - s * r * (r^T W) for W of shape (n_embd, in_dim)."""
    w = np.asarray(weight, dtype=np.float32)
    r = np.asarray(direction, dtype=np.float32)
    r = r / max(float(np.linalg.norm(r)), 1e-12)
    proj = np.outer(r, r @ w)
    return (w - float(scale) * proj).astype(np.float32)


def rank1_ablate_lora(weight: np.ndarray, direction: np.ndarray, scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Exact rank-1 LoRA decomposition of directional ablation: A = r^T W (1, in), B = -s * r (out, 1)."""
    w = np.asarray(weight, dtype=np.float32)
    r = np.asarray(direction, dtype=np.float32)
    r = r / max(float(np.linalg.norm(r)), 1e-12)
    lora_a = (r @ w).reshape(1, -1).astype(np.float32)
    lora_b = (-float(scale) * r).reshape(-1, 1).astype(np.float32)
    return lora_a, lora_b


def _pack_gguf_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _align_up(n: int, align: int = 32) -> int:
    return ((n + align - 1) // align) * align


def export_cvec_gguf(out_path: str, arch: str, layer_vectors: Dict[int, np.ndarray]) -> str:
    """Write a minimal GGUF v3 control vector file (`direction.{layer}` F32) for `llama-server --control-vector`."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    sorted_layers = sorted(layer_vectors.keys())
    kv_entries = [
        ("general.architecture", GGUF_TYPE_STRING, _pack_gguf_str(arch)),
        ("control_vector.layer_count", GGUF_TYPE_UINT32, struct.pack("<I", len(sorted_layers))),
    ]

    header = bytearray()
    header += GGUF_MAGIC
    header += struct.pack("<IQQ", GGUF_VERSION, len(sorted_layers), len(kv_entries))
    for k, vtype, vbytes in kv_entries:
        header += _pack_gguf_str(k) + struct.pack("<I", vtype) + vbytes

    data_offset = 0
    tensor_blobs: List[bytes] = []
    for layer in sorted_layers:
        vec = np.asarray(layer_vectors[layer], dtype=np.float32).reshape(-1)
        name = f"direction.{layer}"
        header += _pack_gguf_str(name)
        header += struct.pack("<IQIQ", 1, vec.shape[0], GGML_TYPE_F32, data_offset)
        raw = vec.tobytes()
        padded_len = _align_up(len(raw), 32)
        tensor_blobs.append(raw + b"\x00" * (padded_len - len(raw)))
        data_offset += padded_len

    # Pad header to 32-byte alignment before tensor data section.
    pad_hdr = _align_up(len(header), 32) - len(header)
    header += b"\x00" * pad_hdr

    with open(out_path, "wb") as f:
        f.write(header)
        for blob in tensor_blobs:
            f.write(blob)
    return out_path


def run_selftest() -> dict:
    rng = np.random.default_rng(42)
    w = rng.normal(size=(16, 12)).astype(np.float32)
    pos = rng.normal(loc=1.5, size=(8, 16)).astype(np.float32)
    neg = rng.normal(loc=-1.5, size=(8, 16)).astype(np.float32)
    r, norm = compute_contrast_direction(pos, neg)
    assert norm > 0.0

    # Verify W' x == W x + B (A x)
    x = rng.normal(size=(12,)).astype(np.float32)
    w_proj = project_out_direction(w, r, scale=1.0)
    lora_a, lora_b = rank1_ablate_lora(w, r, scale=1.0)
    err = float(np.max(np.abs((w_proj @ x) - (w @ x + lora_b @ (lora_a @ x)))))
    assert err < 1e-5, f"LoRA ablation mismatch: {err}"

    # Verify component along r is zeroed
    residual_along_r = abs(float(np.dot(r, w_proj @ x)))
    assert residual_along_r < 1e-5, f"residual not ablated: {residual_along_r}"

    # Verify CHiPS scoring and null control
    null_pct = null_control_percentile(r, pos.mean(axis=0) - neg.mean(axis=0), n_controls=16)
    assert null_pct >= 0.9

    attn_a = [np.array([[0.8, 0.2], [0.7, 0.3]]), np.array([[0.5, 0.5], [0.5, 0.5]])]
    attn_b = [np.array([[0.2, 0.8], [0.3, 0.7]]), np.array([[0.5, 0.5], [0.5, 0.5]])]
    out_a = [np.tile(r * 2.0, (4, 1)), np.zeros((4, 16), dtype=np.float32)]
    out_b = [np.zeros((4, 16), dtype=np.float32), np.zeros((4, 16), dtype=np.float32)]
    heads = score_chips_heads(attn_a, attn_b, out_a, out_b, r)
    assert int(heads[0]["head"]) == 0

    tmp_cvec = "/tmp/cynapse_soup_selftest.cvec.gguf"
    export_cvec_gguf(tmp_cvec, "qwen3", {14: r, 22: r})
    with open(tmp_cvec, "rb") as f:
        magic = f.read(4)
    os.remove(tmp_cvec)
    assert magic == GGUF_MAGIC

    return {
        "status": "PASS",
        "lora_ablation_max_err": err,
        "residual_along_r": residual_along_r,
        "null_pct": null_pct,
        "top_chips_head": int(heads[0]["head"]),
    }


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        print(json.dumps(run_selftest(), indent=2))
