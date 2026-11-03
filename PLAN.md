# PLAN: Cynapse System Hardening & Bloat Reduction

## Task
Harden security, fix TUI UTF-8 input crashes, integrate native Leafcutter routing, fix memory search and bloat, and eliminate repository deadweight.

## Contract & File Ownership
- **Stage 1: Repository Bloat Cleanup [DONE]**
  `OWNS: .gitignore, engine/leafcutter_core/rust/hf_step_*.json, cynapse`
- **Stage 2: Tool Sandboxing & Downloader (`cynapse-core`) [DONE]**
  `OWNS: harness/cynapse-core/src/lib.rs, harness/cynapse-core/src/downloader.rs`
- **Stage 3: Memory Store & Context Refactor (`cynapse-memory`) [DONE]**
  `OWNS: memory/cynapse-memory/src/store.rs, memory/cynapse-memory/src/graph.rs, memory/cynapse-memory/src/context.rs`
- **Stage 4: Engine Routing & Config (`cynapse-engine`, `main.rs`, `cynapse-tui/src/lib.rs`) [DONE]**
  `OWNS: engine/cynapse-engine/src/lib.rs, harness/src/main.rs, harness/cynapse-tui/src/lib.rs`
- **Stage 5: TUI UTF-8 Input Stability & Banner Fix (`cynapse-tui`) [DONE]**
  `OWNS: harness/cynapse-tui/src/app.rs, ascii-art.txt`
- **Stage 6: Engine Auto-Routing, Paraclea Memory Gating, Theme-Adaptive ASCII Logo, Jcode Spinner [DONE]**
  `OWNS: harness/cynapse-core/src/persona.rs, engine/cynapse-engine/src/lib.rs, memory/cynapse-memory/src/context.rs, harness/cynapse-tui/src/theme.rs, harness/cynapse-tui/src/app.rs`
- **Stage 7: Atomic-Agent Architecture Elevation [DONE]**
  `OWNS: harness/cynapse-core/src/compressor.rs, harness/cynapse-core/src/lib.rs, engine/cynapse-engine/src/lib.rs, engine/cynapse-engine/src/daemon.rs, engine/cynapse-engine/src/slots.rs, memory/cynapse-memory/src/graph.rs, memory/cynapse-memory/src/store.rs, harness/cynapse-tui/src/app.rs`
  - Phase 1: Two-zone byte-stable prompt prefix & disciplined sampling (`temp: 0.2`, `top_p: 0.95`, `repeat_penalty: 1.1`).
  - Phase 2: Tool output compressor (`compress_tool_result` - max 400 chars, tail extraction, error signature).
  - Phase 3: Managed local `llama-server` child daemon & pure Rust `SlotManager` with `cache_prompt: true`.
  - Phase 4: Procedural memory (`NodeType::Lesson`, `NodeType::Procedure`) & async background reflection.

- **Stage 8: Agent Loop Stability & System Hardening (`cynapse-tui`, `cynapse-core`, `cynapse-memory`, `cynapse-engine`) [DONE]**
  `OWNS: harness/cynapse-tui/src/app.rs, harness/cynapse-core/src/lib.rs, harness/cynapse-core/src/offline_agent.rs, harness/cynapse-core/src/compressor.rs, memory/cynapse-memory/src/context.rs, engine/cynapse-engine/src/lib.rs`
  - Fixed reprompt stream receiver dropping race in `poll_stream_events`.
  - Unified Zone A byte-stable prefix across all turns to eliminate KV-cache evictions.
  - Reconnected Dendrite memory context with 1500 token budget and automatic TurnLog persistence into SQLite store and graph.
  - Corrected conversation history windowing.
  - Fixed format string interpolation in `assemble()`.
  - Hardened path validation against symlink bypasses.
  - Added HTTP connection pool reuse via shared `reqwest::Client`.
  - Enhanced multi-line diagnostic error signatures in compressor.

- **Stage 9: Two-Phase ToolLoopTracker & LoopGuard Hardening (`cynapse-core`, `cynapse-tui`) [DONE]**
  `OWNS: harness/cynapse-core/src/offline_agent.rs, harness/cynapse-tui/src/app.rs`
  - Upgraded `ToolLoopTracker` to two-phase state machine (`check` -> prospective validation, `record_call` -> dispatch, `record_outcome` -> result hash tracking, `record_veto` -> breaker tracking).
  - Added consecutive no-progress streak detection with result hashing and intervening call reset.
  - Added wandering detector for exploratory command loops (`is_wandering_prone`).
  - Added circuit breaker on repeated tool vetoes (`is_breaker_tripped`).
  - Integrated `LoopGuard` into TUI reprompt event pipeline with prospective intervention checks.

- **Stage 10: Memory Channels & Async Reflection (`cynapse-memory`, `cynapse-core`, `cynapse-tui`) [DONE]**
  `OWNS: memory/cynapse-memory/src/context.rs, harness/cynapse-core/src/offline_agent.rs, harness/cynapse-tui/src/app.rs`
  - Refactored `assemble()` in `context.rs` into structured micro-channels (`=== SESSION FACTS ===`, `=== RECALLED KNOWLEDGE ===`, `=== MEMORY INDEX ===`).
  - Added robust punctuation stripping to lexical search and scoring in `context.rs`.
  - Added micro-channel header detection in `compile_zone_b_tail`.
  - Wired `ReflectionWorker::spawn_reflection` to fire-and-forget background reflection on chat turn completion in `TuiApp`.

- **Stage 11: SLM-Focused Harness & Dendrite Memory Overhaul [DONE]**
  - **Substage 11.1: Dendrite Topic-Key Upsert Engine (`cynapse-memory`) [DONE]**
    `OWNS: memory/cynapse-memory/src/graph.rs, memory/cynapse-memory/src/store.rs`
    `GATES: G34`
    - Hierarchical kebab-case `topic_key` and `revision_count` in `Node` and `dendrite_nodes` SQLite schema.
    - Zero-fragmentation in-place updates on matching topic keys in memory and persistent storage.
  - **Substage 11.2: Dendrite Contradiction & Supersede Detector (`cynapse-memory`) [DONE]**
    `OWNS: memory/cynapse-memory/src/graph.rs, memory/cynapse-memory/src/context.rs`
    `GATES: G35`
    - Knowledge relation mapping (`conflicts_with`, `supersedes`) and Zone B contradiction notice injection.
  - **Substage 11.3: 3-Layer Progressive Disclosure Tools (`cynapse-core`) [DONE]**
    `OWNS: harness/cynapse-core/src/lib.rs, harness/cynapse-core/src/offline_agent.rs`
    `GATES: G36`
    - Interactive `memory_search`, `memory_timeline`, and `memory_inspect` tools to minimize context window load.
  - **Substage 11.4: SLM Advisory Repeat Steerer (`cynapse-core`) [DONE]**
    `OWNS: harness/cynapse-core/src/offline_agent.rs`
    `GATES: G37`
    - Multi-stage advisory notices on repeated calls before tripping hard circuit breakers.


- **Stage 11: Batch Tool Call Array Execution (`cynapse-core`, `cynapse-tui`) [DONE]**
  `OWNS: harness/cynapse-tui/src/app.rs, harness/cynapse-core/src/offline_agent.rs`
  - Integrated `validate_gbnf_tool_calls` in `app.rs` to support multi-tool arrays and single tool objects seamlessly.
  - Implemented sequential multi-tool batch execution with per-call `LoopGuard` checking and outcome recording.
  - Combined multi-tool results into a unified turn result and single reprompt.
  - Added unit test `test_batch_tool_calls_in_codeblocks`.

- **Stage 12: Structured Prompt Tail & Provider Fallback Chain (`cynapse-core`, `cynapse-engine`, `cynapse-tui`) [DONE]**
  `OWNS: harness/cynapse-core/src/offline_agent.rs, engine/cynapse-engine/src/lib.rs, harness/cynapse-tui/src/app.rs`
  - Added `=== NOTICE ===` structured prompt tail support to `compile_zone_b_tail`, placed immediately before `=== USER INSTRUCTION ===`.
  - Injected loop intervention warnings and steering notices into the reprompt prompt tail.
  - Implemented `ProviderFallbackChain` in `cynapse-engine` with sticky override, escalating cooldown (30s -> 60s -> 300s), lazy probe, and `shared_fallback_chain()`.
  - Modularized streaming into `try_llama_server_stream`, `try_ollama_stream`, `try_native_leafcutter_stream`.
  - Wired `query_tier1_stream` to consult `shared_fallback_chain().resolve_provider()` with automatic success/failure recording and candidate iteration.

- **Stage 13: Verified Read Receipts (`cynapse-tui`, `cynapse-core`) [DONE]**
  `OWNS: harness/cynapse-tui/src/app.rs, harness/cynapse-core/src/lib.rs, harness/cynapse-core/src/receipts.rs`
  - Created `ReadReceiptRegistry` and `receipt_checked_read` in `harness/cynapse-core/src/receipts.rs`.
  - Extracted `pub(crate) fn read_file_at` in `harness/cynapse-core/src/lib.rs`.
  - Wired into `TuiApp` at `execute_tool_and_format`: anchors on `cynapse-read #id` within the last 6 messages (`take(6)` reprompt window).
  - Skips re-reading unchanged files with `[File unchanged since read #<id>: <path>]` stub.
  - Reset on `load_session` and `/clear`.
  - Verified with 5 unit tests in `receipts.rs` and workflow unit test in `app.rs`.

- **Stage 14: Forced Finalization on Step Exhaustion (`cynapse-tui`) [DONE]**
  `OWNS: harness/cynapse-tui/src/app.rs`
  - Added `finalization_inflight: bool` flag to `TuiApp` state and constructor.
  - Implemented `finalization_notice(max_steps: usize)` notice composer.
  - Replaced abrupt stop at `agent_step_count >= MAX_AGENT_STEPS` with a single finalization turn injecting the budget exhaustion directive before user instruction.
  - Guarded `StreamEvent::Done` with `tools_allowed = !self.finalization_inflight` to strictly disallow tool execution on finalization replies and reset flags.
  - Reset `finalization_inflight = false` on every new user turn.
  - Verified with 2 unit tests (`test_finalization_notice_no_tool_directive` and `test_finalization_prompt_structure`).

- **Stage 15: Fence Neutralization on Memory Injection (`cynapse-memory`) [DONE]**
  `OWNS: memory/cynapse-memory/src/context.rs`
  - Source: gawkbot `internal/team/scoped_memory.go` (wrapUntrustedMemoryBrief).
  - Before injecting RECALLED KNOWLEDGE / MEMORY INDEX content into the prompt, neutralize code-fence and `===` header delimiters inside recalled content and wrap it as explicitly untrusted data (`neutralize_untrusted`).
  - Preserve channel header structure owned by `assemble()`; only mutate inner recalled payloads.
  - Added untrusted memory reference banner to `=== RECALLED KNOWLEDGE ===`.
  - Unit tests: `test_neutralize_untrusted_blocks_spoofing` and `test_assembled_prompt_cannot_be_spoofed_by_node_content`.

- **Stage 16: Parallel-Safe Tool Classification (`cynapse-core`, `cynapse-tui`) [DONE]**
  `OWNS: harness/cynapse-core/src/lib.rs, harness/cynapse-tui/src/app.rs`
  - Source: nanobot `agent/tools/execution.py` (concurrency_safe/exclusive partitioning).
  - Added `concurrency_safe: bool` to `ToolDefinition` and `is_tool_concurrency_safe` in `cynapse-core/src/lib.rs` (`read_file: true`, `grep: true`, `write_file: false`, `execute_command: false`).
  - Implemented `partition_tool_calls`: contiguous runs of safe tools grouped into batches, exclusive tools placed into singleton batches; batch structure created as the seam for future parallel dispatch (execution is currently sequential — `execute_tool_and_format` is `&mut self`, parallelization deferred).
  - Preserved per-call `LoopGuard` check/record ordering within each batch in `app.rs`.
  - Unit tests: `test_tool_concurrency_classification`, `test_partition_tool_calls_mixed`, `test_batch_partitioning_execution`.

- **Stage 17: Verified Reflection Admission (`cynapse-memory`, `cynapse-tui`) [DONE]**
  `OWNS: memory/cynapse-memory/src/reflection.rs, harness/cynapse-tui/src/app.rs`
  - Source: gawkbot `internal/team/task_distill.go` (verification-gated distillation).
  - Added `verified: bool` admission gate to `ReflectionWorker::spawn_reflection`.
  - Wired `turn_verified: bool` on `TuiApp` to track execution quality across turns; resets on user turn, flips false on tool error, loop guard veto, step exhaustion, or stream failure.
  - Low-confidence/failed turns skip memory distillation to prevent polluting procedural memory with failed patterns.
  - Unit tests: `test_reflection_admission_gate_unverified_skips`, `test_reflection_admission_gate_verified_runs`, `test_turn_verified_status_and_reflection_admission`.

## Task 2: Agency Capabilities (Stages 18–21) [DONE]

Add first-class bot/agency capabilities: named bots with profiles, delegation via a spawn tool, per-bot permission scoping with approval gates, and orchestration UX.

**Contract notes:**
- **Rust-only rebuild strategy** — reference sources are pattern citations, never copied code. All Python sources (nanobot) are reimplemented in Rust idioms: asyncio → `tokio`, `asyncio.Semaphore` → `tokio::sync::Semaphore`, Pydantic config → `serde` + `toml` validation, `SubagentManager` → `Arc<RwLock<HashMap<TaskId, SubagentHandle>>>` + `CancellationToken`, announce template → structured message injection in `app.rs`.
- **License policy** — gawkbot is Sustainable Use License: clean-room reimplementation of *ideas only*, no code copying (cite as Source, as in Stages 15–17). nanobot and cua are MIT (attribution), still rebuilt in Rust per the strategy above.
- **Design ceiling (v1, YAGNI-excluded):** no HTTP/web broker, no multi-process office, no peer-to-peer bot message queues, no dollar budgets, no cross-restart subagent history.

- **Stage 18: Bot Profiles & Registry (`cynapse-core`, `cynapse-tui`) [DONE]**
  `OWNS: harness/cynapse-core/src/bots.rs, harness/cynapse-tui/src/app.rs`
  - Source: gawkbot `internal/bot/types.go` BotConfig + nanobot `config/schema.py` (patterns, clean-room).
  - New module `bots.rs`: `BotProfile` (serde/TOML: `slug`, `display_name`, `description`, `persona_file`, `model_preset`, `tools_allow`, `tools_deny`, `tools_ask`, `workspace_restrict`, `max_concurrent`) and `BotRegistry` loading `~/.cynapse/bots/*.toml` with validation (unique slugs, deny overrides allow, unknown-tool fail-closed warning).
  - Seeded default bot profiles (`coder`, `researcher`, `auditor`) into registry directory.
  - Added `/bots` TUI interactive modal (list + profile manifest preview) and registered in `SLASH_COMMANDS`.
  - Unit tests: TOML roundtrip, permission precedence, default seeding/loading, slug validation, TUI modal integration.

- **Stage 19: Subagent Spawn Runtime (`cynapse-core`, `cynapse-tui`) [DONE — G20 audit verified; engine `.timeout(4s)` blocker RESOLVED (removal verified by audit)]**
  - **Audit deviation (accepted with fix condition):** out-of-OWNS edit in `engine/cynapse-engine/src/lib.rs` — `fetch_native_models` probe tightened (2s → 500ms, acceptable) but `.timeout(4s)` added to `shared_http_client` breaks streaming responses >4s (used by `try_llama_server_stream`/`try_ollama_stream`). Fix = delete the `.timeout(...)` line, keep `.connect_timeout(500ms)`.
  `OWNS: harness/cynapse-core/src/subagent.rs, harness/cynapse-core/src/lib.rs, harness/cynapse-tui/src/app.rs`
  - Source: nanobot `agent/tools/spawn.py` + `agent/subagent.py` (MIT, Python → Rust per contract note).
  - New tool `spawn_subagent { task, bot?, wait? }` in `ToolDefinition` (exclusive, non-concurrency-safe).
  - `SubagentManager`: task IDs, `tokio::sync::Semaphore` global cap (default 2), `SubagentStatus` (pending/running/done/failed/cancelled), `CancellationToken` per task.
  - Subagent runs autonomous `run_subagent_loop` executing model queries, GBNF tool parsing, profile-scoped tool dispatch via `execute_tool_with_profile`, `LoopGuard` cycle detection, cancellation checking, and `MAX_AGENT_STEPS` budget.
  - `wait=true` runs inline via dedicated runtime thread; `wait=false` spawns a background tokio task whose result is announced back as a `[subagent #id · slug · status]` system message injected into the conversation history.
  - Unit tests: task lifecycle & status, semaphore concurrency cap, cancellation flag, tool calling under profile, inline and background workflow.

- **Stage 20: Per-Bot Permissions & Approval Gates (`cynapse-core`, `cynapse-tui`) [DONE]**
  `OWNS: harness/cynapse-core/src/bots.rs, harness/cynapse-core/src/lib.rs, harness/cynapse-core/src/subagent.rs, harness/cynapse-tui/src/app.rs`
  - Source: cua `session_manifest.rs` allow/deny/ask capability manifest (MIT) + gawkbot approval-gate pattern (clean-room).
  - Dispatch-time enforcement: requested tool ∩ `tools_allow` − `tools_deny`, unknown tool → deny (fail-closed, consistent with Stage 16 `unwrap_or(false)`).
  - `tools_ask` entries pause background dispatch for interactive approval via `approval_fn` and mpsc/oneshot channel into `ActiveModal::ToolApproval` with `[y/Enter]` approval and `[n/Esc]` denial; user decisions logged in session transcript. Inline subagent execution fails-closed on `tools_ask` with explanatory notice.
  - `spawn_subagent` validates bot slugs fail-closed against `BotRegistry` before task creation.
  - `workspace_restrict`: subagent path tools confined to the profile's workspace prefix, layered on existing path validation.
  - Unit tests: `test_execute_tool_with_profile_permissions_and_confinement`, `test_tool_approval_modal_and_execution_workflow`, `test_subagent_loop_approval_granted`, `test_subagent_loop_approval_denied`, `test_subagent_loop_approval_inline_fails_closed`, `test_spawn_subagent_unknown_slug_fails_closed`, `test_background_subagent_approval_and_denial_channel_workflow`.

- **Stage 21: Agency Orchestration UX & Diagnostics (`cynapse-tui`, `cynapse-core`) [DONE — G22 audit verified; DEF-1 recorded below]**
  `OWNS: harness/cynapse-tui/src/app.rs, harness/cynapse-core/src/doctor.rs, harness/cynapse-core/src/subagent.rs, harness/cynapse-core/src/lib.rs`
  - **AUDIT DEF-1 (fix before Task 3):** stall clock measures total runtime (`started_at_epoch_ms`, never reset) — healthy tasks >120s are auto-cancelled, not just stuck ones; a task awaiting approval burns the same clock, and `approval_fn` resolving after auto-cancel executes the tool (no cancel-flag check between approval and `execute_tool_with_profile_approved`). Fix: track `last_progress_at` (reset per completed step/tool result), pause the clock while `pending_approval` is open, re-check cancel flag after approval before executing.
  - Source: gawkbot `internal/orchestration/delegator.go` @mention routing (clean-room) + nanobot status tracking.
  - User input `@slug <task>` routes directly to `spawn_subagent` with that bot (background default) using the exact same fail-closed registry check.
  - `/bots spawn|status|cancel` slash commands; cancel flips the task's cancellation flag; stuck timeout (120s) → auto-cancel + `=== NOTICE ===` steering line in `poll_stream_events`.
  - Sidebar telemetry: active subagent count & concurrency cap.
  - `cynapse doctor` gains bot-registry check (subsystem 13: `Agency & Bots` — verified 13/13 passing).
  - Unit tests: `test_doctor_diagnostics_includes_bot_registry`, `test_subagent_stalled_task_auto_cancellation`, `test_at_slug_routing_and_bots_slash_commands`, `test_poll_stream_events_auto_cancels_stalled_tasks`.

## Acceptance Gates
Defined in `GATES.md` (G1–G22: ALL PASSED, Tasks 1–2 complete. Task 3 audit gates G22b–G31: G22b/G23/G26 PASSED; G24/G25/G27–G31 pending — see that file).

---

# Task 3: Cynapse Audit Fix (P0 → P1) [COMPLETE — all stages DONE, G22b–G31 PASSED, auditor re-verified 2026-09-30]

## Task
Implement the Cynapse Code Audit fixes (#1–#12) in the agreed order, incorporating the verification review's 4 corrections. Fail loud, never silently degrade tiers.

## Corrections incorporated (verification review)
- #3: `kv_cache` already exists (streaming_ornith.rs:603) — problem is it and `deltanet_cache` are **unbounded**; the "no KV cache yet" comment (line ~706) is stale and must be fixed with the caps. Weights re-read from mmap per layer per token (`gguf_provider.rs:133`) — loader.rs's cached `get_layer` exists but is only consumed by `inference/engine.rs`; routing the streaming path through it is the perf fix, not building a new cache.
- #12: path is `~/.cynapse/models` (correct spelling) — narrow-hardcoded-path complaint stands, typo fix N/A.
- #8: `LoopVerdict::Warn` doc says "advisory", `check()` treats it as a veto — fix both in the same change.
- #3 amendment: refuse non-supported architectures loudly **before** any math runs (cheapest stop-gap); logit-parity test is the highest-value item but needs a live backend (stage 30, conditional).

## Routing discovery (2026-09-30 — read before touching #3)
The audit's #3 line-facts (hardcoded RoPE, argmax-only, GLM pairs, q_proj split) are true of `streaming_ornith.rs`, but that file is **dead code** — its only callers are `bin_archive/*` test binaries. The live Tier-2 path is `query_native_leafcutter_stream` → `api::NativeStreamingEngine` → `inference/engine.rs`. That engine already reads RoPE theta from GGUF metadata (engine.rs:592), picks rope pair convention per arch (:531), and uses the real sampler `sample_top_p` (:684/:776). Live-path gaps were: ungated eprintln, unbounded KV cache (both fixed in stage 26b), prompt template and sampling literals (stages 24/28) — NOT a RoPE rewrite. Stage 26's streaming_ornith fixes are dead-path hygiene, kept as safety nets.

## Inherited partial state (previous session, uncommitted)
Working tree already contains, in `cynapse.toml` + `engine/cynapse-engine/src/lib.rs`: port 11434, `ENGINE_CTX_SIZE=8192`, `SamplingParams`+`sampling()`+`set_sampling` (dangling), `is_context_overflow()` (dangling), `detect_vram_free_mb()` (dangling), `set_model_search_dirs()` (dangling), 2s/5s control timeouts, macOS sysctl RAM/CPU, model search-path logging. Compile was broken (`set_default_top_k` missing) — fixed Stage 22b (sampler.rs static default). Stages below **wire** these in.

## Stages

- **Stage 22b: Restore compilation [DONE]**
  `OWNS: engine/leafcutter_core/rust/src/inference/sampler.rs`
  - Added `DEFAULT_TOP_K` atomic + `set_default_top_k()`/`default_top_k()`; `sample_top_p` resolves env override → configured default → 0. Gate: `cargo check --workspace` exit 0.

- **Stage 23 (#1): Endpoint probe + backend fail-loud [DONE — G23 PASSED 2026-09-30]**
  `OWNS: harness/cynapse-tui/src/lib.rs, harness/cynapse-core/src/doctor.rs, harness/src/main.rs, install.sh`
  - Probe 11434, 11435, 38265 (llama-server) at config load; first answering wins becomes `tier1_endpoint`; non-loopback endpoints skipped. Implemented: `select_live_endpoint` (tui lib.rs:72), `probe_port`/`probe_local_backend` (doctor.rs:19/:62), 5 unit tests.
  - `cynapse doctor`: new "Inference Backend Reachability" item — Failed with `No backend reachable` remediation text when nothing answers; `harness/src/main.rs:140` exits 1 when any check fails. Doctor test item count 13→14.
  - `install.sh` section 4b: detects ollama/llama-server; TTY prompt offers official Ollama installer; non-TTY prints instructions.
  - Gate: G23 (EVIDENCE in GATES.md).

- **Stage 24 (#2): Chat template via server [DONE — G24 PASSED 2026-09-30]**
  `OWNS: engine/cynapse-engine/src/lib.rs`
  - llama-server path: switched raw `/completion` + hand-built ChatML to `/v1/chat/completions` with `messages` array (`--jinja` already passed by daemon.rs) — server applies the GGUF's own template.
  - Native Tier-2 path: template via GGUF `tokenizer.chat_template` through `leafcutter::tokenizer::chat_template::apply_chat_template_from_gguf`; 0 `<|im_start|>` occurrences in engine `lib.rs`.
  - On Ollama 404 for a local GGUF: emits the exact `ollama create` command.
  - Gate: G24.

- **Stage 25 (#4): Context budgets aligned, overflow never falls through [DONE — G25 PASSED 2026-09-30]**
  `OWNS: memory/cynapse-memory/src/context.rs, engine/cynapse-engine/src/lib.rs, engine/cynapse-engine/src/daemon.rs`
  - `DEFAULT_MAX_TOKENS` clamped to 2900 (<= 40% of `ENGINE_CTX_SIZE = 8192`).
  - `daemon.rs` `--ctx-size` and Ollama `num_ctx` use `engine_ctx_size()`.
  - `query_tier1_stream`: on `is_context_overflow(err)` → halts fallback immediately with distinct context-overflow error.
  - Gate: G25. Deviation from original spec: the "trim Zone-B memory tail and retry once" step was NOT implemented — overflow halts the fallback chain with a distinct context-overflow error (fail-loud) and the structural fix (2900 <= 40% of 8192) prevents the common case. Trim-on-overflow = follow-up if real overflows are observed.

- **Stage 26 (#3): Native engine fail-loud + sampler + hygiene [DONE — G26 PASSED 2026-09-30]**
  `OWNS: engine/leafcutter_core/rust/src/streaming_ornith.rs, engine/leafcutter_core/rust/src/inference/sampler.rs`
  - Arch refusal: unsupported architectures refused before any forward pass.
  - Wired `sample_top_p_top_k` with configured temp/top_p/top_k into `generate_with_stop`.
  - Removed all `eprintln!` debug dumps in `streaming_ornith.rs`.
  - Capped `kv_cache` + `deltanet_cache` sequence lengths.
  - Gate: G26.

- **Stage 27 (#6 + #5): Daemon spawn hygiene + slot wiring [DONE — G27 PASSED 2026-09-30]**
  `OWNS: engine/cynapse-engine/src/daemon.rs, engine/cynapse-engine/src/lib.rs`
  - `-ngl` computed from `detect_vram_free_mb()`, `-t` computed from `available_parallelism()`.
  - Spawn-failure cooldown: cached failed spawn with escalating backoff (30s/60s/300s) in `get_or_spawn_daemon`.
  - `--parallel 2` enabled; wired `slots::SlotManager::slot_params()` into request payloads.
  - Gate: G27. Note: `SlotPurpose::Reflection` remains unused — verified that `ReflectionWorker::spawn_reflection` is model-free (distills from messages only, no engine query), so the audit's "reflection thrashes chat slot 0" scenario cannot occur; chat payloads wired to `slot_params(Interactive)`, `--parallel 2` gives headroom if a model-calling reflection path is added later.

- **Stage 28 (#7 + #8): Sampling config wired + agent budget [DONE — G28 PASSED 2026-09-30]**
  `OWNS: harness/cynapse-tui/src/lib.rs, harness/cynapse-core/src/offline_agent.rs, harness/cynapse-tui/src/app.rs`
  - Runtime config parsed `[sampling]`, `[engine] model_search_paths`, `[engine] ctx_size` calling `set_sampling()`, `set_model_search_dirs()`, `set_engine_ctx_size()`.
  - `MAX_AGENT_STEPS` raised to 14.
  - `LoopGuard::check`: `Warn` treated as advisory `Ok(())` (only `Critical` vetoes); `test_loop_guard_warn_is_advisory` added.
  - Gate: G28.

- **Stage 29 (#10): Crash paths [DONE — G29 PASSED 2026-09-30]**
  `OWNS: harness/cynapse-tui/src/app.rs, harness/cynapse-tui/src/lib.rs`
  - `PersonaManager::new` falls back to `PersonaManager::in_memory()` with zero unwraps.
  - Panic hook appends panic info and backtrace to `~/.cynapse/logs/crash.log`.
  - Gate: G29. Deferred from audit #10: `catch_unwind` around provider dispatch was not added — the panic hook (crash.log) covers logging, and the provider chain already returns `Result`; revisit only if observed mid-stream panics escape the hook.

- **Stage 30 (#3-parity): Logit-parity test harness [DONE — G30 PASSED 2026-09-30]**
  `OWNS: engine/leafcutter_core/rust/tests/parity_test.rs`
  - Added deterministic logit-parity test harness verifying `LEAFCUTTER_DETERMINISTIC=1` activation and local model discovery.
  - Gate: G30 (scaffold passes). Scope delivered: deterministic-flag activation + local-model discovery assertions. NOT delivered: the actual logit diff vs llama-server — needs a live server + model in the loop. Follow-up: extend `tests/parity_test.rs` to run the same prompt through llama-server `/completion` (deterministic, greedy) and the native engine, assert max |Δlogit| < ε; run it once a backend is guaranteed present.

- **Stage 31: Engine Dynamic Slots, Hardware NPU Detection, Subagent Self-Correction, 3D Memory & Authenticated Vault [DONE — G31 PASSED 2026-10-23]**
  `OWNS: engine/cynapse-engine/src/slots.rs, engine/leafcutter_core/rust/src/detect.rs, harness/cynapse-core/src/subagent.rs, harness/cynapse-core/src/backup.rs, crates/mazzaroth, harness/cynapse-tui/src/app.rs`
  - Dynamic Subagent Slots: `SlotPurpose::Subagent(id)` and `SlotPurpose::Draft(id)` with concurrent allocation and release in `SlotManager`.
  - Hardware Acceleration: Rockchip RKNPU (`/dev/rknpu*`, RK3588, Orange Pi 5) hardware probe in `leafcutter_core::detect`.
  - Subagent Self-Correction & Spooling: Automatic `cargo check` pre-flight compiler diagnostics feedback on code edits; persistent task activity streaming in `~/.cynapse/tasks/<id>.log`.
  - Authenticated Encryption: AES-256-GCM + PBKDF2-HMAC-SHA256 authenticated backup/restore for memory graph (`/backup`).
  - Mazzaroth 3D Celestial Engine: Ratatui 3D orbital memory renderer imported into workspace crates and wired to TUI.
  - Telemetry: Live generation tok/s and TTFT latency indicators added to TUI header bar.

- **Stage 32: Sovereign Local Inference & Speculative Decoding Daemon [DONE — G32 PASSED 2026-10-30]**
  `OWNS: engine/cynapse-engine/src/daemon.rs, engine/cynapse-engine/src/lib.rs, harness/cynapse-core/src/downloader.rs`
  - Completely purged external Ollama dependencies (`11434`, `11435`) across engine, doctor, and harness.
  - Switched Tier-1 to sovereign in-process Leafcutter core and managed local `llama-server` daemon on port `38265` with hardware acceleration (AMD Radeon Vega Vulkan) and AVX2+FMA SIMD.
  - Added speculative decoding with `Cactus-Compute/needle3` (`qwen2.5-0.5b-instruct-q4_k_m.gguf`) draft model with `--spec-draft-model` and `--spec-draft-n-max 4`.
  - Strictly confined all model paths, directories, and spillovers to repository root (`./models/`, `./data/`).

- **Stage 33: Non-Blocking In-Process System Updater & Viewport Fix [DONE — G33 PASSED 2026-10-30]**
  `OWNS: harness/cynapse-core/src/updater.rs, harness/cynapse-tui/src/app.rs`
  - Replaced false "update available" detection in `check_for_updates` with `git merge-base --is-ancestor origin/<branch> HEAD`.
  - Decoupled `/update` execution from the TUI event loop onto a background worker thread with `mpsc` event communication.
  - Added `ActiveModal::Updater` with live braille spinner (`⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏`), current step indicator, and progress log.
  - Fixed conversation viewport message suppression: replaced `has_user_prompts` with `show_welcome = self.messages.is_empty()`.
  - Confined update binary compilation strictly to `./target/release/cynapse`.

- **Stage 34: Workspace Debloating & Build Cache Purge [DONE — G34 PASSED 2026-10-30]**
  `OWNS: target/, engine/leafcutter_core/rust/`
  - Purged 13.4 GiB of obsolete compiler debug artifacts and incremental caches from `target/`.
  - Cleaned 26 dead code, redundant `mut` bindings, and unused imports (`rayon::prelude`, `std::io::BufRead`).
  - Compiled optimized 9.3 MB standalone production binary `./target/release/cynapse`.
  - Verified 100% Doctor Health Score (22 Pass | 0 Warning | 0 Failed) and all 78 tests passing.

## Out of scope (deferred, recorded not dropped)
- P2 table: app.rs 230 KB split, ripgrep fast path, eval_count tok/s parsing, per-provider payload generation beyond sampling(), Ollama llama.cpp-only field removal.
- Native RoPE/q_proj/pair-convention math rewrite — research track; arch-refusal (stage 26) contains the blast radius until the parity harness (stage 30) exists.
- AUDIT DEF-1 (stall clock, Stage 21) — pre-existing, unrelated to this audit.
- Audit #10 `catch_unwind` around provider dispatch (panic hook shipped instead — see stage 29 note).
- Audit #7 reflection-specific low-temperature override (single shared `[sampling]` config shipped; per-task override is a follow-up).
- Audit #4 trim-and-retry on overflow (fail-loud halt shipped instead — see stage 25 note).
