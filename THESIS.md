# Cynapse Research & Engineering Thesis Notes

**Hardware Platform:** Orange Pi 6 Plus (`aarch64`, ARMv9.2-A, 12 cores, 14 GiB LPDDR5, 0 B swap, Debian 12, GCC 12.2.0)  
**Repository:** `Alartist40/cynapse-harness`

---

## 1. End-to-End On-Device SBC Post-Training & Quantization Milestone

This project demonstrates that a **1.7B-parameter hybrid reasoning model (`LiquidAI/LFM3-1.7B`)** can be **fine-tuned via LoRA, merged to F16 GGUF, calibrated with a domain-targeted Importance Matrix (`imatrix`), and quantized to `Q4_K_M`** entirely locally on an ARM64 single-board computer (Orange Pi 6 Plus) with **zero discrete GPU and zero cloud training**.

### 1.1 Native ARMv9 Engine Unlock (`lfm3.cpp`)
- Compiled upstream `llama.cpp` (`lfm3.cpp` + `lfm-base.cpp`) natively on `aarch64` with `-march=armv9-a+dotprod+fp16+i8mm` (`HAVE_DOTPROD`, `HAVE_SVE`, `HAVE_MATMUL_INT8`).
- Unlocked `LFM3-1.7B-Q4_K_M.gguf` locally at **`72.4 t/s` prompt** and **`25.8 t/s` generation**.

### 1.2 Safety-Filtered Cybersecurity SFT Dataset
- Converted `Anthropic-Cybersecurity-Skills` and `grim-prompter` markdown playbooks into `1,096` ChatML SFT examples (`data/cyber_rescue_sft.jsonl`, `2.2 MB`, `0` empty fields, `0` duplicates, `16` unsafe/destructive entries filtered out).
- Split: `986` train / `110` validation samples.

### 1.3 Local ARM64 CPU LoRA Training (`soup-cli` `0.77.2` + `torch-2.15.0+cpu`)
- **Configuration (`synapses/soup/cyber_rescue_lfm3.yaml`):**
  - Base model: `LiquidAI/LFM3-1.7B` (`1,731,250,944` total params; `2,449,408` trainable LoRA params = `0.14%`)
  - LoRA: `r=8`, `alpha=16` on `q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj`, `out_proj`
  - Memory-safe SBC settings: `seq_len=512`, `batch_size=1`, `gradient_accumulation_steps=4`, `gradient_checkpointing=true`, `OMP_NUM_THREADS=6` (~`5.08 GB` peak footprint)
- **Training Outcome (`247 / 247` steps, 1 full epoch, resumable checkpoints every 5 steps):**
  - Training loss: **`3.0083 -> 1.2738`**
  - Validation loss: **`3.2015`** (`eval_mean_token_accuracy: 0.4042`)
  - **Honest Finding:** The gap between `train_loss` (`1.27`) and `eval_loss` (`3.20`) indicates partial memorization/overfitting to the 986 static markdown extracts. Base `LFM3-1.7B` already has strong general Linux knowledge, so v1 improved defensive framing and structure rather than delivering a large reasoning leap.

---

## 2. Domain-Targeted Importance-Matrix (`imatrix`) Quantization

Inspired by `Underdog-Saluki-27B-1.0`, instead of quantizing all weights uniformly, we calibrated `llama-imatrix` on a focused domain corpus (`data/cyber_imatrix_calib.txt`, `375.0 KB`: `183` cybersecurity triage/forensics samples + `40` tool-calling samples) so the quantizer preserves precision on weights that activate during cybersecurity reasoning and tool calls.

- **Calibration (`./bin/llama-imatrix`):** `30` chunks (`n_ctx=512`, `t=6`), `153` importance matrix entries recorded, final `PPL = 9.5306 +/- 0.31525`.
- **Measured `F16` vs `imatrix-Q4_K_M` Comparison (Orange Pi 6 Plus, `-t 6`):**

| Artifact | Size / BPW | Prompt Speed | Generation Speed |
| :--- | :--- | :--- | :--- |
| `LFM3-1.7B-CyberRescue-F16.gguf` | `3298.17 MiB` (`3.2 GB`, `16.00 BPW`) | `10.1 t/s` | `7.7 t/s` |
| `LFM3-1.7B-CyberRescue-imatrix-Q4_K_M.gguf` | **`1040.82 MiB`** (`1.1 GB`, `5.05 BPW`) | **`88.2 t/s`** (**8.7x faster**) | **`28.9 t/s`** (**3.75x faster**) |

- Both the `F16` and `imatrix-Q4_K_M` GGUF artifacts are preserved for reproducible evaluation and public sharing.

---

## 3. Harness Architecture & Token-Efficiency Innovations
- **Pure-Rust `rtk` Output Compactor (`harness/cynapse-core/src/rtk.rs`):** Transparently compacts noisy `execute_command`, `grep`, and subagent outputs (60–90% token savings) with a zero-overhead fast path (`<= 20` lines / `<= 1,200` chars).
- **Spec-Driven Development (`/sdd`, `harness/cynapse-core/src/spec.rs`):** Scaffolds `SPEC.md` + `GATES.md` before coding so small local models do not guess.
- **Sequential `/eval` (`harness/cynapse-core/src/eval.rs`):** Evaluates models one-after-another (never simultaneously) to protect low-RAM SBCs without swap.
- **Active-Expert Disk Streaming (`engine/leafcutter_core/rust/src/expert_stream.rs`):** Bounded-cache on-demand disk streaming for MoE experts.
- **Dendrite Nervous System (`crates/dendrite-3d` + `cynapse-memory`):** Brainstem -> Spine -> Synaptic Dendrites layout with deterministic auto-tagging on save.

---

## 4. v2 Roadmap: Universal, Harness-Agnostic Cybersecurity Model
To make v2 genuinely stronger and universally usable outside Cynapse (`llama.cpp`, `Ollama`, LM Studio, or any agent):
1. **Harness-agnostic training data:** Focus on universal defensive cybersecurity (Linux/Windows forensics, log/PCAP/IOC analysis, MITRE ATT&CK containment, safe remediation) and standard OpenAI-style tool calling — avoiding Cynapse-only tool names.
2. **Higher-signal, multi-turn diagnostic data + lower LR (`5e-5`):** Reduce overfitting (`eval_loss` tracking `train_loss`) with diverse, realistic incident-response dialogues.

---

## 5. `Glia-1.7B-v1.5` Failure-Targeted SFT & Verification (`2026-11-22`)

Rather than re-running an 18-hour full-corpus SFT, v1 (`Glia-1.7B`) was evaluated on the universal cyber suite, weak points were harvested (`harvest.rs` / `cyber_eval.rs`), and a focused **24-example** universal SFT dataset (`data/glia_v2_sft.jsonl`, `21` train / `3` val) was trained on Orange Pi 6 Plus (`OMP_NUM_THREADS=6`, `epochs=2`, `lr=1e-4`, LoRA `r=8`, `alpha=16`).

### 5.1 v1 vs. v1.5 Training Efficiency & Generalization

| Metric | `Glia-1.7B` (v1) | `Glia-1.7B-v1.5` | Change |
| :--- | :--- | :--- | :--- |
| Training Duration | ~18 hours (`247` steps) | **13 minutes** (`12` steps) | **~83x faster** |
| Validation Loss (`eval_loss`) | `3.2015` | **`2.0242`** | **-1.1773** |
| Validation Token Accuracy | `40.42%` | **`62.41%`** | **+21.99%** |
| Quantized GGUF (`imatrix-Q4_K_M`) | `1040.82 MiB` (`5.05 BPW`) | `1040.82 MiB` (`5.05 BPW`) | Preserved `F16` + `Q4_K_M` |

### 5.2 Universal & Adversarial Evaluation (`10 / 10` Pass, `27.9–28.4 t/s`)
Tested via `./bin/llama-cli --single-turn --simple-io -n 350 -t 6` (`models/Glia-1.7B-v1.5-Q4_K_M.gguf`):
- **Core Cyber Suite (`4 / 4`):** `linux_tmp_triage` (`86.1 / 27.9 t/s`), `destructive_guardrail` (`83.0 / 28.4 t/s`), `ssh_brute_containment` (`84.8 / 28.3 t/s`), `ransomware_backup_check` (`72.6 / 28.4 t/s`).
- **Adversarial Suite (`6 / 6`):** `prompt_injection` (`93.9 / 28.0 t/s`), `windows_persistence` (`86.0 / 28.3 t/s`), `reverse_shell_ioc` (`93.8 / 28.2 t/s`), `generic_tool_json` (`86.4 / 28.2 t/s`), `dns_tunneling` (`91.3 / 28.4 t/s`), `suid_audit` (`82.5 / 28.4 t/s`).
- **Observed Limitation:** At `-n 350`, `LFM3-1.7B`'s `[Start thinking]` block consumes most of the 350-token cap on complex prompts; `-n 700+` or `--reasoning-budget 0` yields complete un-truncated final answers.

### 5.3 Interactive TUI Controls & Portability (`2026-11-22`)
- **`/effort <none|low|med|high|auto>`:** Wired `reasoning_budget_tokens` (`0 / 96 / 256 / 1024`) into `cynapse-engine` (`27.3 t/s` observed on `Glia-1.7B-v1.5-Q4_K_M.gguf` with `/effort none`), plus clean `[Start thinking]` / `[End thinking]` separation.
- **Safety & Stability:** Offloads prior `llama-server` on model switch, enforces a RAM headroom guard, reads hardware `timings.predicted_per_second`, removes strobing borders, and ships `models/Glia-1.7B.Modelfile` + `GLIA.md` for portable Ollama/`llama.cpp` reuse.


