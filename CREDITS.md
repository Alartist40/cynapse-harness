# Open Source Credits, Acknowledgments & Inspirations

Cynapse is engineered as a sovereign, offline-first agent system written in pure safe Rust. It builds upon ideas, research, and pioneering open-source architectures created by the broader AI, systems programming, and digital forensics communities.

We extend our deep gratitude and legal attribution to the original authors and maintainers of the following projects whose designs, architectures, and breakthroughs inspired and shaped Cynapse:

---

## 1. Inference Engines & High-Throughput Kernel Compute

- **[llama.cpp / ggml](https://github.com/ggerganov/llama.cpp)** — *Georgi Gerganov & the llama.cpp open-source community*
  - **Inspiration**: The foundational GGUF binary container format specification, standard quantized block layout definitions (`Q4_0`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, `Q8_K`), and AVX2/NEON block SIMD matrix-vector dot product mathematics.
  - **License**: MIT License.

- **[Strata](https://github.com/interplains/strata)** — *Strata Engine Project*
  - **Inspiration**: High-throughput layer-major chunked prefill mechanics, asynchronous disk/memory prefetching pipelines, and pre-transposed zero-clone weight buffers for sovereign model streaming.
  - **License**: MIT / Apache 2.0.

- **[AirLLM](https://github.com/lyogavin/Anima/tree/main/air_llm)** — *Gavin Li & the Anima Lab team*
  - **Inspiration**: Pioneered layered sequential transformer execution directly from NVMe/SSD storage, enabling 70B+ large language models to run on resource-constrained consumer hardware without multi-GPU clusters.
  - **License**: Apache 2.0.

- **[Colibri](https://github.com/colibri-core)** — *Colibri Research Contributors*
  - **Inspiration**: Micro-GEMV vectorization routines, low-latency weight streaming patterns, and memory-bounded quantization arithmetic for edge hardware.
  - **License**: MIT License.

---

## 2. Agent Harness Engineering & UI Architecture

- **[Atomic-Agent](https://github.com/Brain-Bakers/atomic-agents)** — *Brain Bakers & contributors*
  - **Inspiration**: Single-responsibility atomic tool design, strict JSON Schema parameter coercion, surgical exact-anchor patching (`patch_file_at`), and isolated subagent task boundaries.
  - **License**: MIT License.

- **[jcode](https://github.com/jcode-sh/jcode)** — *jcode Contributors*
  - **Inspiration**: Terminal User Interface (TUI) design paradigms, responsive split-screen viewport layout, top header telemetry pills, and developer-first keyboard ergonomics.
  - **License**: MIT License.

- **[DeepSeek Harness & Cordis Architecture](https://github.com/deepseek-ai)** — *DeepSeek AI Team*
  - **Inspiration**: Decoupled agent lifecycle hooks (`before_tool_exec`, `after_tool_exec`), dynamic micro-skill intent slicing, and bounded KV-cache preservation prompts.
  - **License**: MIT / Apache 2.0.

---

## 3. Knowledge Graph, Memory Architecture & Visualization

- **[Engram](https://github.com/engram-memory)** — *Engram Memory Project*
  - **Inspiration**: Epistemic memory confidence decay curves, topic-key structured upsert mechanics, semantic supersede & contradiction detection, and dead-memory garbage collection.
  - **License**: Apache 2.0 / MIT.

- **[Dendrite Memory System]**
  - **Inspiration**: Force-directed 3D knowledge graph dynamics utilizing Hooke's elastic springs and Coulomb's electrostatic repulsion physics, integrated with embedded SQLite FTS5 full-text indexing and BM25 relevance ranking.

- **[Dendrite-3D Celestial Engine](https://github.com/Alartist40/cynapse-harness/tree/main/crates/dendrite-3d)** — *3D Galaxy Perspective & Orbital Physics Engine for Dendrite Memory*
  - **Inspiration**: 3D orbital projection math, Euler yaw/pitch camera matrix transformations, and multi-tier galaxy atlas hierarchy rendered inside pure ASCII/Ratatui canvas buffers.
  - **License**: MIT License.

---

## 4. Digital Forensics & Synapse Extension Ecosystem

- **[Zero-Breach / AdFection](https://github.com/interplains/zero-breach)** — *Interplains & citizen forensics researchers*
  - **Inspiration**: Local digital forensics methodology, Doctrine D2 strict zero-network air-gapped matching, read-only multi-browser SQLite history snapshot isolation, and curated indicators of compromise (IOCs) originating from Citizen Lab and Recorded Future threat disclosures.
  - **License**: Apache 2.0 / CC-BY-SA 4.0.

- **[Soup](https://github.com/MakazhanAlpamys/soup)** — *MakazhanAlpamys*
  - **Inspiration**: Low-VRAM layer streaming post-training profiler, dataset token length distribution inspection, and chat-template compatibility diagnostics.
  - **License**: Apache 2.0.

- **[Cactus-Compute Needle3](https://huggingface.co/Cactus-Compute/needle3)** — *Cactus Compute Team*
  - **Inspiration**: Curated low-parameter (`0.5B`) high-speed draft model architecture for prompt lookup speculative decoding and zero-hallucination JSON extraction.

---

## 5. Rust Ecosystem & Open-Source Libraries

Cynapse is built upon world-class Rust open-source libraries:
- **[Ratatui](https://github.com/ratatui/ratatui)** & **[Crossterm](https://github.com/crossterm-rs/crossterm)** — Interactive terminal rendering and raw cross-platform terminal events.
- **[rusqlite](https://github.com/rusqlite/rusqlite)** — Embedded, safe, bundled SQLite connection interface.
- **[tokio](https://github.com/tokio-rs/tokio)** — Asynchronous multi-threaded runtime, channels, and task orchestration.
- **[serde](https://github.com/serde-rs/serde)** & **[serde_json](https://github.com/serde-rs/json)** — High-performance zero-copy serialization/deserialization framework.
- **[aes-gcm](https://github.com/RustCrypto/AEADs)** & **[pbkdf2](https://github.com/RustCrypto/password-hashes)** — Authenticated AES-256-GCM memory encryption vaults.
- **[sha2](https://github.com/RustCrypto/hashes)** — Pure Rust cryptographic hashing kernels.
- **[wgpu](https://github.com/gfx-rs/wgpu)** — Cross-platform Vulkan, Metal, and DirectX compute pipeline bindings.

---

*All trademarks, logos, and brand names are the property of their respective owners. Their mention in this file is solely for fair-use open-source attribution, technical credit, and educational acknowledgment.*
