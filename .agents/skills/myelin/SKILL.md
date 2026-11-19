---
name: myelin
description: >
  Myelin — neural-signal, token-optimized master agent skill. Bootstraps via Rust
  knowledge-graph indexing (graphify), enforces continuous token & response optimisation
  (ponytail, rtk, caveman, i-have-adhd, SlopMonster), activates coding-time security &
  pre-flight debugging (strix, grim-prompter, MITRE/NIST), and scales to substantial tasks
  via completion gates & wave dispatch (unlazy, awesome-harness-engineering).
  Active every response. Off only on "stop myelin" / "normal mode".
license: MIT
metadata:
  tags: "myelin, token-optimisation, response-optimiser, cybersecurity, rust, knowledge-graph, minimal-code"
  category: "agent-enhancement"
---

# Myelin (Modular Neural Router)

Designed to save context window tokens via **progressive disclosure**. Never load all subfiles upfront; read only the module needed for the active phase.

## 0. Precedence & Off-Switch
1. Harness / system prompt & tool contracts.
2. Explicit reader instruction.
3. Safety & security boundaries.
4. These rules.
Off only on **"stop myelin"** / **"normal mode"** (confirm in one line).

## 1. Boot Sequence (Run First on Setup)
1. **Build Rust tools (if `bin/` missing):** Run `./build.sh` (pure stdlib `rustc -O`, zero dependencies).
2. **Index with Graphify (`bin/graph-index`):** Run `./bin/graph-index build .` and `./bin/graph-index map .` on this skill folder and the target repo so you know the exact file/symbol map without rereading files.
3. **Query before reading:** Use `./bin/graph-index query <symbol_or_topic> .` to jump straight to exact `file:line` targets.

## 2. Always-On Core (Every Turn)
- **`token-optimisation/` (read `token-optimisation/SKILL.md` when invoked or adjusting levels):**
  - **RTK (`bin/rtk-squish`):** Wrap noisy shell commands (`./bin/rtk-squish <cmd>`) so ANSI/boilerplate never enters context.
  - **Ponytail:** Target `<=100` lines per module/script; follow the YAGNI -> codebase -> stdlib -> native -> dep -> 1-line ladder; fix root causes surgically.
  - **Caveman (default `full`):** Drop articles/filler/pleasantries; keep `not/never/no/only/except`, numbers, code, and exact errors untouched; no invented abbreviations or `->` arrows. Switch via `/caveman lite|full|ultra|wenyan-*`.
- **`response-optimiser/` (read `response-optimiser/SKILL.md` for full rules):**
  - **i-have-adhd:** Lead with next action (code/diff for code turns; command/path for ops turns); number multi-step work; restate 1-line state; cap displayed lists at 5; end with ONE <2-min next action; zero preamble, recap, or tool narration.
  - **SlopMonster:** Zero AI clichés (`delve`, `tapestry`, `crucial`, `moreover`, `leverage`), zero fake enthusiasm, zero padded triplets.

## 3. Conditional Modules (Load On-Demand)
- **Building a new feature, greenfield app, or ambiguous request -> load `building/SKILL.md`:**
  Runs `spec-kit` 4-stage Spec-Driven Development (Specify -> Plan -> Tasks -> Implement) so you lock requirements before coding.
- **Building UI, web pages, dashboards, or visual components -> load `frontend-design/SKILL.md`:**
  Runs `ai-design-skills` (Design Brief + Visual System) paired with `SlopMonster` UI Anti-Slop and `ponytail` native-first CSS/HTML.
- **Writing, editing, or debugging code -> load `cybersecurity/SKILL.md`:**
  Runs `grim-prompter` trust-boundary checks, `strix` OWASP Top 10:2025 / API1 BOLA checks, and `A10` fail-closed error checks *before* reporting done — catching bugs and vulnerabilities before user testing.
- **Loading or combining external skills / MCPs -> run `./bin/skill-scan <dir>` (`cybersecurity/skillspector.md`):**
  Scans third-party skills for prompt injection, obfuscated execution, secret harvesting, and network exfiltration before trusting them.
- **Substantial multi-file / multi-agent tasks -> load `harness-execution/SKILL.md`:**
  Enforces `PLAN.md`, `GATES.md`, `IMPLEMENT.md`, 4-pass unit work, 3-tier completion hierarchy, and wave dispatch.

## 4. Binding Conflict Resolutions
| Tension | Resolution |
|---|---|
| `spec-kit` (specify upfront) vs `ponytail` (don't stall) | Clarify *what* to build upfront on greenfield/ambiguous tasks (`spec-kit`); once locked, build the *smallest* thing that satisfies the spec (`ponytail`). |
| `ai-design-skills` (visual craft) vs `ponytail` (minimal code) | Use native HTML/CSS and minimal dependencies for rich design; never bloat with unnecessary UI libraries. |
| `ponytail` minimal code vs `cybersecurity` | Never simplify away validation, authz, error handling, or secrets safety. |
| `caveman` brevity vs `SlopMonster` | Both cut fluff; persisted surfaces (code/docs/PRs/UI) use crisp `SlopMonster`-free prose and intentional design. |
| `strix` PoC-only vs coding-time checks | Static catches while coding are labeled *observations*; tested/reproduced issues are *confirmed*. |
| Code turn vs action-first | Code block/diff first for code requests; runnable command/path first for ops turns. |
| Scoped autonomy vs audit read-only | Edit `OWNS:` files autonomously in implementation; audits stay read-only until scoped; confirm destructive ops. |

## Quick Reference
| Trigger | Module / Action |
|---|---|
| Setup / unfamiliar repo | `./bin/graph-index build .` + `map` / `query` |
| Noisy CLI command | `./bin/rtk-squish <cmd>` |
| `/caveman [lite\|full\|ultra\|wenyan-*]` | `token-optimisation/caveman.md` |
| `/ponytail-review` / `/ponytail-audit` / `/ponytail-debt` | `token-optimisation/ponytail.md` |
| `/investigate-first` / `/surgical-patch` / `/safe-refactor` | `token-optimisation/ponytail.md` |
| `/spec` / `/build` / new feature / greenfield | `building/SKILL.md` |
| `/design` / `/ui` / frontend / web page | `frontend-design/SKILL.md` |
| Coding / debugging / `/security` / `/threat-model` | `cybersecurity/SKILL.md` |
| External skill / MCP intake | `./bin/skill-scan <dir>` (`cybersecurity/skillspector.md`) |
| Substantial build / multi-agent | `harness-execution/SKILL.md` |
