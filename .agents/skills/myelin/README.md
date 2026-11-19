# Myelin

A modular, token-optimized AI agent skill suite designed to insulate model reasoning—cutting wasted tokens, eliminating AI slop, enforcing minimal code, and catching security flaws before testing.

## Quick Setup
```bash
./build.sh
./bin/graph-index build .
./bin/graph-index map .
```
Requires only `rustc` (zero Cargo or Python dependencies).

## Structure
- `SKILL.md` — Lean progressive-disclosure router.
- `src/` — Pure stdlib Rust tools (`graph_index.rs`, `rtk_squish.rs`, `skill_scan.rs`).
- `token-optimisation/` — `graphify`, `rtk`, `ponytail` (<=100-line minimal code), `caveman` (leveled compression).
- `response-optimiser/` — `i-have-adhd` (action-first formatting), `SlopMonster` (prose & UI anti-slop rules).
- `building/` — `spec-kit` (4-stage Spec-Driven Development: Specify, Plan, Tasks, Implement).
- `frontend-design/` — `ai-design-skills` (Design Brief, Visual System, native-first UI craft).
- `cybersecurity/` — `skillspector` (external skill/MCP intake scanner), `grim-prompter` (threat modeling), `strix` (OWASP Top 10:2025 / PoC review), MITRE & NIST mappings.
- `harness-execution/` — `unlazy` completion gates, 3-tier hierarchy, wave dispatch, and drop-in templates.
