# Cynapse

Local-first AI agent harness written in Rust. Runs offline on SBCs (Orange Pi 6 Plus) and desktops with no cloud dependencies.

## Core Subsystems

- **Leafcutter (`engine/leafcutter_core/rust`):** 3-tier inference router (managed `llama-server` daemon on `:38265`, fallback providers, and pure-Rust GGUF/Safetensors streaming with MoE active-expert disk streaming and `lfm3` support).
- **Dendrite (`memory/cynapse-memory`, `crates/dendrite-3d`):** SQLite + FTS5/BM25 local memory graph with automatic tag synthesis and a Brainstem -> Spine -> Synaptic Dendrites terminal visualizer.
- **Harness (`harness/cynapse-core`, `harness/cynapse-tui`):** Terminal UI with built-in pure-Rust `rtk` command-output compaction, `bwrap` sandboxing, Spec-Driven Development (`/sdd`), sequential A/B model evaluation (`/eval`), Zero-Breach local forensics (`/security`), and optional Arch-style `/soup` training extension (`/soup install`, `/soup harvest`).

## Build & Setup

```bash
git clone https://github.com/Alartist40/cynapse-harness.git
cd cynapse-harness

# Build release binary (ARM64/SBC-safe: .cargo/config.toml sets jobs = 2)
cargo build --release -p cynapse

# Optional: build native llama-server, llama-cli, llama-quantize, llama-imatrix
./scripts/build_llama_server.sh

# Run diagnostics and launch
./target/release/cynapse doctor --fix
./target/release/cynapse
```

## Launch Modes

- `cynapse` — Ratatui terminal interface (default)
- `cynapse --cli` — Line-by-line CLI mode
- `cynapse --resume <session_id>` — Resume a saved session
- `cynapse run <model>` — Run a model directly

## Slash Commands

| Command | Purpose |
| :--- | :--- |
| `/model`, `/pull` | Select local GGUF or download from Hugging Face |
| `/sdd <goal>` | Scaffold `SPEC.md` and `GATES.md` before coding |
| `/eval <a.gguf> <b.gguf>` | Sequential A/B speed and response benchmark (`data/eval_history.json`) |
| `/security` | Offline Zero-Breach forensics scan + `[a]` CyberRescue AI triage |
| `/soup` | Neuron surgery & LoRA studio (`/soup install`, `/soup remove`, `/soup harvest`) |
| `/dendrite`, `/drawer` | Brainstem-Spine-Dendrite memory atlas and node inspector |
| `/persona`, `/bots` | Manage specialist personas and delegated subagents |
| `/doctor` | Hardware, SIMD, database, and daemon diagnostics |
| `/spec`, `/thinking` | Toggle speculative prompt lookup (`F6`) and thinking blocks |
| `/undo`, `/rewind`, `/fork` | Revert file edits, rewind turns, or branch sessions |
| `/clear`, `/exit` | Reset conversation or quit |

## Shortcuts

- `Ctrl+P` — Command palette
- `Ctrl+D` — Context drawer
- `Ctrl+O` — Collapse/expand tool pills
- `F6` — Toggle speculative decoding
