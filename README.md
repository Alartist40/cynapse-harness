# CYNAPSE — Offline-First Agent System & Synaptic Knowledge Memory

**Cynapse** is a high-performance, offline-first AI agent platform engineered entirely in 100% pure Rust. Designed for total privacy, zero external runtime dependencies, and instant local execution, it bridges host hardware telemetry with an intelligent multi-engine router, a dynamic synaptic knowledge graph, and an Atomic-Agent-inspired architecture elevation.

At the heart of Cynapse lies a unified architecture where three sovereign systems work in concert: **Dendrite**, a force-directed 3D graph memory system with Coulomb/Hooke physics, SQLite FTS5 full-text indexing, and BM25 relevance ranking; **Leafcutter**, a custom pure-Rust GGUF and Safetensors execution core providing local tensor streaming with KV cache trunk reservation; and **sovereign llama-server daemon / in-process Leafcutter engine**, acting as a high-speed Tier-1 inference runner with Vulkan/AVX2 acceleration, speculative decoding (`needle3` / `qwen2.5-0.5b`), and multi-provider fallback. Together, these subsystems provide a complete local agent harness capable of autonomous tool execution, strict GBNF grammar constrained outputs, real-time memory synthesis, 3D orbital galaxy memory visualization, and self-healing diagnostics—all running strictly locally on your hardware without a single byte leaving your machine.

---

## Installation & Setup

### Build from Source
```bash
git clone https://github.com/Alartist40/cynapse-harness.git
cd cynapse-harness
cargo build --release
./target/release/cynapse
```

### System Health Verification
After compilation, run the self-healing diagnostic to verify hardware, SIMD acceleration, storage, and database integrity:
```bash
./target/release/cynapse doctor --fix
```

`cynapse doctor` performs 22 comprehensive diagnostics across CPU, GPU, AVX2 SIMD, GBNF grammar compiler, SQLite FTS5 database, model integrity, and inference daemon reachability (port 38265).

---

## Launch Modes

- **Visual TUI Dashboard (Default)**: Launch the full visual Ratatui interactive interface:
  ```bash
  cynapse
  ```
- **CLI REPL Mode**: Run in line-by-line terminal mode:
  ```bash
  cynapse --cli
  ```
- **Resume Past Session**: Reopen any saved session transcript by ID:
  ```bash
  cynapse --resume <session_id>
  ```
- **Run Model Directly**:
  ```bash
  cynapse run <model_name_or_number>
  ```

---

## Interactive Slash Commands

Type `/` in the prompt bar to trigger the floating command menu:

| Command | Description |
|---------|-------------|
| `/help` | Display interactive keyboard shortcuts & help menu |
| `/model` | Open interactive model selector & scanner |
| `/pull` | Download GGUF models from HuggingFace (curated catalog & custom URLs) |
| `/persona` | Manage agent personality markdown files (`IDENTITY`, `SOUL`, `USER`, custom `.md`) |
| `/bots` | Manage bot profiles; `/bots spawn <slug> <task>`, `/bots status [id]`, `/bots cancel <id\|all>` |
| `/doctor` | Launch Cynapse Doctor self-healing diagnostic dashboard (25 comprehensive checks) |
| `/security` | Launch Zero-Breach air-gapped digital forensics auditor, IOC search & NetGuard probe |
| `/soup` | Launch Soup Low-VRAM Fine-Tuning Studio (layer streaming, distillation & loss curve) |
| `/dendrite` | Open interactive Dendrite 3D Planetary Memory Graph |
| `/memory` | Open 3D Orbital Planetary Memory Atlas visualizer |
| `/drawer` | Open interactive Dendrite Memory drawer inspector |
| `/synapse` | Manage modular Synapse tool extensions (`zero_breach`, `soup`) |
| `/spec` | Toggle n-gram speculative prompt lookup decoding acceleration (`F6`) |
| `/thinking` | Toggle collapsible model reasoning/thinking stream blocks |
| `/theme` | Cycle visual color themes (Tokyo Night, Emerald Matrix, Amber CRT, Dark Slate, etc.) |
| `/update` | Check, pull, and compile latest Cynapse updates with non-blocking live modal |
| `/undo` | Roll back latest file modifications made by autonomous agent tools |
| `/session` | Manage and resume saved conversation sessions |
| `/clear` | Reset conversation view and restore brand banner |
| `/exit` | Quit Cynapse TUI |

### Terminal Shortcuts

| Key | Action |
|-----|--------|
| `Tab` | Open Dendrite Memory Drawer inspector from anywhere |
| `Up` / `Down` | Scroll conversation viewport line-by-line |
| `Ctrl+Backspace` / `Ctrl+W` | Delete word backward in prompt input |
| `Ctrl+T` | Toggle model thinking/reasoning blocks |
| `F6` | Toggle n-gram speculative prompt lookup acceleration |
| `Ctrl+A` / `Ctrl+E` | Move input cursor to start / end of line |
| `Ctrl+U` | Clear input line |
| `Spacebar` / `s` (in `/memory`) | Toggle 3D Galaxy auto-rotation |
| `r` / `F5` (in `/doctor`) | Re-run self-healing diagnostics with auto-repair |

---

## Architecture

Cynapse is structured as a decoupled multi-crate Rust workspace. See [ARCHITECTURE.md](ARCHITECTURE.md) for the full technical breakdown.

```
cynapse-mini/
├── harness/
│   ├── src/main.rs             # CLI & TUI entrypoint dispatcher
│   ├── cynapse-tui/            # Ratatui visual interface, modals, 3D visualizer
│   └── cynapse-core/           # Tool execution, forensics engine, GBNF parser, bot registry, doctor
├── crates/
│   └── dendrite-3d/            # 3D Celestial Galaxy Visualization & Hierarchical Orbit Engine
├── memory/
│   └── cynapse-memory/         # Dendrite graph engine, SQLite FTS5 store, BM25 ranker
├── engine/
│   ├── cynapse-engine/         # Semantic hardware router, Tier-1 LLM client, provider fallback
│   └── leafcutter_core/        # Pure Rust GGUF & Safetensors tensor kernels
├── synapses/                   # Modular Synapse tool extension packs (zero_breach, soup)
└── install.sh                  # Single-line automated pure Rust installer & hardware detector
```

### Research & Speculative Decoding Thesis Report
For empirical evaluations, roofline model analysis, and mathematical proofs of distribution equivalence, see **[docs/SPECULATIVE_DECODING.md](docs/SPECULATIVE_DECODING.md)**.
- **Roofline Model & DRAM Wall**: Mathematical breakdown of why 35B parameter models are memory-bandwidth bound on edge DDR5, and how L3-cache resident draft models achieve 1.8×–2.5× throughput acceleration.
- **Cactus-Compute/needle3 vs. Qwen 2.5 0.5B**: In-depth empirical study on why Monarch-Hadamard layer-sliced edge models eliminate JSON hallucination and bracket errors for autonomous agent execution.

---

## Capabilities

### Synaptic Memory & 3D Galaxy Memory Atlas
Memory in Cynapse is modeled after biological neural networks with a force-directed 3D physics engine. Ideas, facts, and conversation turn logs form nodes connected by weighted synaptic links that strengthen with use and decay with disuse over time.
- **Force-Directed 3D Galaxy Physics**: Real Coulomb repulsion, Hooke spring attraction, central gravity toward the supermassive core, velocity damping (0.88), collision avoidance, and force clamping.
- **3D Orbital Galaxy Visualizer (`/memory`)**: Visualizes memory nodes as stars organized into galactic clusters revolving around a supermassive central core.
- **Two-Tier Hybrid Memory Recall**: Merges SQLite FTS5 keyword matching with BM25 scoring, specialization index weighting, and exponential temporal decay.
- **Engram & DeepSeek Hardened Architecture**: Generation-fenced SQLite transactions, calibrated BM25 score bounds, topic-key in-place upserting (`family/slug`), contradiction and supersede tracking, tool-result middle-pruning, and speculative memory prefetching.
- **Structured Memory Micro-Channels**: Session facts, recalled knowledge, and memory index injected as structured prompt sections.

### Atomic-Agent Architecture Elevation
Six architecture phases inspired by the Atomic-Agent reference implementation:
- **Stable Zone A Prefix**: Byte-stable system prompt cached across multi-turn tool loops for KV-cache reuse.
- **Two-Phase LoopGuard**: Prospective loop detection with no-progress streak tracking, wandering detection, and circuit breaker.
- **Batch Tool Dispatch**: Array tool call parsing `[{"tool": ...}, ...]` with sequential per-call guard evaluation.
- **Notice Injection**: `=== NOTICE ===` prompt tail for loop detector and steering redirections.
- **Provider Fallback Chain**: Sticky multi-provider fallback (llama-server -> Ollama -> NativeLeafcutter) with escalating cooldowns (30s -> 60s -> 300s) and lazy probe recovery.

### Agent Loop Hardening (Stages 13-17)
Five hardening stages layered on the Atomic-Agent elevation:
- **Verified Read Receipts**: Re-reading an unchanged file within the 6-message context window returns `[File unchanged since read #N]` instead of re-sending content — verified against a content hash *and* the anchor's presence in actual model context.
- **Forced Finalization**: On step exhaustion (`MAX_AGENT_STEPS = 14`), one final no-tools turn demands an honest status report (completed / outstanding / errors) instead of an abrupt stop; a `tools_allowed` guard hard-blocks tool execution on that reply.
- **Memory Injection Sanitization**: Recalled memory is fenced with an `[UNTRUSTED MEMORY]` banner; code fences (``` -> `` ` ``) and channel headers (`===` -> `= = =`) inside stored nodes are neutralized so they cannot spoof prompt channels.
- **Concurrency-Safe Batch Partitioning**: Tools carry `concurrency_safe` metadata (`read_file`/`grep` safe, `write_file`/`execute_command` exclusive); batches partition into contiguous safe runs and singletons as the seam for future parallel dispatch.
- **Verified Reflection Admission**: Reflection distills only clean turns — tool errors, LoopGuard vetoes, step exhaustion, and stream failures set `turn_verified = false` and skip distillation.

### Agency Capabilities (Task 2, Stages 18-21)
Delegation to scoped specialist bots with fail-closed permission enforcement:
- **Bot Profiles & Registry**: TOML profiles at `~/.cynapse/bots/` (`tools_allow`/`tools_deny`/`tools_ask`, workspace restriction, persona, concurrency cap); deny beats allow, unknown tools fail closed; seeds `@coder`, `@researcher`, `@auditor`.
- **Autonomous Subagent Loop**: `spawn_subagent` runs a budgeted loop (5 steps + forced finalization) with GBNF tool parsing, LoopGuard cycle detection, cancellation tokens, and a concurrency semaphore (2); results announce back into the conversation.
- **Interactive Approval Gates**: `tools_ask` tools in background subagents pause for a `y/n` approval modal (`[tool denied by user]` on deny); inline subagents and unknown bot slugs fail closed.
- **Orchestration UX**: `@slug <task>` input routing, `/bots spawn|status|cancel` commands, 120s stall auto-cancel with `=== NOTICE ===` steering, sidebar agency telemetry, and a doctor registry check.

### Offline Agent Capabilities & GBNF Grammar Engine
- **NDJSON Streaming**: Polymorphic streaming parser supporting multiple API formats (Ollama, vLLM, generic).
- **GBNF Grammar Constraints**: Enforces valid JSON tool-call schema syntax, preventing output formatting panics offline.
- **KV-Cache Slot Affinity**: Slot 0 pinned for chat, slot 1 for reflection, with `cache_prompt: true` for warm cache reuse.
- **Max Step Safeguards**: Configurable step limits (`MAX_AGENT_STEPS = 14`) with forced finalization — one no-tools status-report turn on exhaustion prevents runaway recursive tool execution.

### Async Background Reflection
Fire-and-forget reflection worker distills conversation turns into atomic facts, procedures, and lessons without blocking the UI. Runs on a dedicated background thread after each turn completion, gated by verified-run admission: turns with tool failures, vetoes, or exhaustion are not distilled.

### Offline Model Downloader & Model Management (`/pull`)
- **Curated Recommendations**: Probes system RAM and GPU VRAM to tag optimal model sizes.
- **Custom HuggingFace Downloads**: Download any GGUF model from HuggingFace with real-time progress.
- **Background Async Progress**: Non-blocking Tokio downloader with real-time speed and progress bars.

### System Persona Manager (`/persona`)
- **Interactive TUI Modal & Live Editor**: Inspect, edit, and switch persona `.md` files in `~/.cynapse/persona/`.
- **Dynamic Prompt Compiler**: Injects direct system instructions into every model prompt without generic LLM headers.

### Cynapse Doctor Self-Healing Engine (`/doctor`)
- **13-Subsystem Auditing**: Host RAM headroom, GPU & hardware acceleration, SIMD (AVX2), model storage directory, GGUF file integrity, SQLite connection, SQLite table/index health, GBNF grammar compiler, local host tools, Tokio scheduler, Tier-1 endpoint & model catalog, persona catalog, bot registry & profile catalog.
- **Auto-Fix Mode (`--fix`)**: Automatically recreates missing directories, repairs database indexes, clears stale files.

### Visual Themes & UI Experience
- **4 Visual Color Presets**: Dark Slate, Neon Cyber, Amber CRT, Emerald Matrix.
- **Dynamic Input Box**: Expands from 3 to 8 lines as text grows.
- **Collapsible Reasoning Blocks (`Ctrl+T`)**: Expand or collapse internal model thinking streams.
- **Rich Markdown Formatting**: In-terminal syntax-highlighted code blocks, blockquotes, headers, and bullet lists.

### Universal Skill & Low-Parameter Harness Elevation (Stage 33)
Six architectural pillars transforming small local models (0.5B to 3B) into reliable agents:
- **Tolerant Argument Coercion**: Normalizes common model key aliases (`filepath` $\to$ `path`, `cmd` $\to$ `command`) and un-wraps stringified types automatically.
- **Actionable Levenshtein Hints**: Distance $\le 2$ suggestions for unknown arguments (`"Unknown argument 'filenam'. Did you mean 'path'?"`).
- **Surgical File Patching (`patch_file`)**: Exact block matching with anchor checks to eliminate whole-file drop catastrophes.
- **Deterministic Acceptance Gates (`run_gates`)**: Native runner executing `GATES.md` (`CHECK:`, `EXPECT:`) and recording verified exit codes.
- **Dynamic Micro-Skill Intent Slicing**: Slices the 30KB Universal Skill into zero-token in-memory Rust structures, holding prompt context under 650 tokens.
- **Autonomous Security Shield**: Pre-flight AST/heuristic scanner catching command injection, path traversal, and destructive actions.

---

## Verification

| Metric | Value |
|--------|-------|
| Tests passed | 328+ |
| Tests failed | 0 |
| Build | `cargo build --release` succeeds (9.6 MB standalone binary) |
| System doctor | 25 checks, 100% health (25 Pass, 0 Warning, 0 Failed) |
| Acceptance gates | G1-G42 all PASSED |

---

## License
MIT License. Free and open-source.
