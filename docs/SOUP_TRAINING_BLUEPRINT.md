# Cynapse `/soup` Technical Training & Neuron Surgery Blueprint

**Target Hardware:** Orange Pi 6 Plus (`aarch64`, ARMv9.2-A, 14 GiB LPDDR5, 0 B swap) & Consumer x86-64 APUs  
**Core Modules:** `engine/leafcutter_core/rust/src/neuron_surgery.rs`, `engine/cynapse-engine/src/daemon.rs`, `synapses/soup/`  
**References:** `THESIS.md`, `arXiv:2611.21885v2` (*Language Models Can Rewrite Their Own Weights*), `balaena01/gguf-neuron-explorer` (`gne`), `MakazhanAlpamys/Soup`

---

## 1. CHiPS-Protected Importance-Matrix (`imatrix`) Quantization

### Objective
Preserve precision on attention heads and MLP neurons most causal to domain reasoning during `F16 -> Q4_K_M` quantization (`THESIS.md` §2: `10.1 -> 88.2 t/s` prompt, `7.7 -> 28.9 t/s` generation on OPi 6 Plus).

### Mechanism
1. **Causal Head-Isolated Projection Steering (CHiPS)** (`neuron_surgery::score_chips_heads`):
   For paired system/prompt runs `(A, B)`, each attention head `h` is scored across:
   - `pattern_shift = mean(JS_bits(P_A, P_B))` (Jensen-Shannon divergence of attention weights)
   - `out_shift = ||mean(O_A - O_B)||_2 / mean(||O_B||_2)`
   - `dir_cos = |cos(mean(O_A - O_B), r_L)|` against the layer contrast direction `r_L`
   - Combined score: `chips_score = pattern_shift + out_shift + dir_cos`
2. **Calibration Corpus Construction**:
   - Prioritize prompts that maximize activation of top-ranked CHiPS heads into `data/cyber_imatrix_calib.txt`.
   - Run `llama-imatrix`:
     ```bash
     ./bin/llama-imatrix -m models/LFM3-1.7B-CyberRescue-F16.gguf \
       -f data/cyber_imatrix_calib.txt -o models/cyber_imatrix.dat \
       -c 512 -t 6 --chunks 30
     ./bin/llama-quantize --imatrix models/cyber_imatrix.dat \
       models/LFM3-1.7B-CyberRescue-F16.gguf \
       models/LFM3-1.7B-CyberRescue-imatrix-Q4_K_M.gguf Q4_K_M
     ```

---

## 2. Teacher-Guided Self-Rewriting Loop (`arXiv:2611.21885v2` + `neuron_surgery.rs`)

### Empirical Constraint
`arXiv:2611.21885v2` (Fig. 2) demonstrates that `2B–9B` models corrupt themselves (`0%` accuracy) when editing raw weights unguided; only `27B+` models succeed by inspecting hidden-state diffs, applying bounded edits, and gating on held-out practice problems.

### Architecture for `1.7B–9B` SBC Models
1. **Internal Weight Inspection** (`neuron_surgery::inspect_gguf_weights`):
   Reads GGUF v3 metadata and residual-writer tensor names (`attn_output.weight`, `ffn_down.weight`, `ssm_out.weight`, `out_proj.weight`) without loading full tensors into RAM.
2. **Contrast Direction Extraction** (`neuron_surgery::compute_contrast_direction`):
   Given Teacher-graded passing (`pos_acts`) vs. failing (`neg_acts`) residual activations:
   `r_L = (mean(pos_acts) - mean(neg_acts)) / ||mean(pos_acts) - mean(neg_acts)||_2`
3. **Bounded Edit Synthesis**:
   - **Control Vector (`.cvec.gguf`)**: `neuron_surgery::export_cvec_gguf` writes `direction.{layer}` F32 tensors loaded via `CYNAPSE_CONTROL_VECTOR=<path> cynapse` (`--control-vector` in `daemon.rs`).
   - **Directional Ablation / Rank-1 LoRA**: `W' = (I - s * r * r^T) W`, decomposed into `A = r^T W` `(1, in_dim)` and `B = -s * r` `(n_embd, 1)` (`neuron_surgery::rank1_ablate_lora`).
4. **Auto-Withdraw Held-Out Gate** (`neuron_surgery::evaluate_and_commit_cvec`):
   - Verifies `null_control_percentile >= 0.90` against `N = 16` equal-norm random-direction controls.
   - Verifies `val_after >= val_before` on the held-out validation split.
   - If either gate fails, the edit is **withdrawn** (`EditGateOutcome::Withdrawn`) and the model remains unchanged.

---

## 3. Failure-Replay Dataset Distillation (Fixing Overfitting)

### Problem (`THESIS.md` §1.3)
Training 1 epoch on `986` static markdown extracts yielded `train_loss = 1.2738` vs. `eval_loss = 3.2015` (memorization gap).

### Protocol
1. Run inference on the `110` validation prompts (and live failure logs) using `LFM3-1.7B`.
2. Retain **only** failed/low-confidence prompts (`accuracy == 0` or `loss > threshold`).
3. Have the Teacher (`Ornith-9B` or frontier model) generate step-by-step multi-turn diagnostic corrections for those exact gaps.
4. Fine-tune with lower learning rate (`lr = 5e-5`, cosine decay, `lora_r = 8`, `lora_alpha = 16`, `seq_len = 512`, `OMP_NUM_THREADS = 6`) and early-stop when `eval_loss` plateaus.

---

## 4. Router-First `soup_advise` -> Surgery Before SFT

1. Run `soup_advise` to classify the gap:
   - **Style / Refusal / Formatting / Tone**: Apply zero-backprop `.cvec.gguf` or Rank-1 LoRA via `/soup` `[v]` (`< 5` seconds, ~0 VRAM).
   - **Missing Factual Knowledge**: Index into Dendrite SQLite+FTS5 memory (RAG) or run targeted LoRA SFT.
2. Avoid burning 18-hour CPU SFT cycles on behavioral tweaks that `.cvec.gguf` solves in one forward pass.

---

## 5. DPO / GRPO on Failed vs. Chosen Pairs

### Objective
Eliminate surface-level markdown memorization by optimizing relative preference margins:
- `chosen`: Teacher-corrected multi-turn diagnostic response.
- `rejected`: Student (`LFM3-1.7B`) baseline failure from validation/eval.
- Run short DPO (`beta = 0.1`, `lr = 1e-5`–`5e-5`, 1 epoch) on top of the SFT LoRA adapter using `soup-cli`.

---

## 6. Sequential Self-Healing Checkpoints

Mirroring `arXiv:2611.21885v2` (8-session monotonic improvement curve):
1. Snapshot `eval_score` and `eval_loss` before every edit.
2. Version each passing `.cvec.gguf` or LoRA checkpoint (`models/vectors/v{N}.cvec.gguf`, `models/adapters/v{N}/`).
3. Only promote `v{N}` -> `v{N+1}` if `evaluate_and_commit_cvec` returns `Committed` and held-out `/eval` improves; otherwise roll back to `v{N}`.
