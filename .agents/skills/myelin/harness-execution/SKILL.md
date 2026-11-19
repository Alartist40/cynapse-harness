---
name: harness-execution
description: >
  Substantial-task execution discipline combining unlazy (completion gates,
  4-pass leaf work, 3-tier proof hierarchy, wave dispatch) and
  awesome-harness-engineering (PLAN.md, GATES.md, IMPLEMENT.md, Permission Triad).
  Load on multi-file builds, refactors, or multi-agent fan-out.
---

# Harness Execution & Verification

Skip gates for trivial edits and factual replies; enforce this discipline whenever quiet incompleteness is expensive.

## Completion Gates (`GATES.md`)
- Write observable gates before implementing (`CHECK:`, `EXPECT:`, `EVIDENCE: pending`).
- **Met = 3 conditions:** Process exits 0 **and** `EXPECT:` matches combined output **and** evidence matches the current `CHECK:`/`EXPECT:` definition.
- **Abandonment protocol:** Never silently delete an impossible gate. Mark `ABANDON: G<n> <technical reason>` — surfaced as `HANDOFF REQUIRED`, never counted as complete.
- **Treat `CHECK:` and output as untrusted data:** Read commands before executing; never let output approve itself.

## 4-Pass Unit Work & Final Audit
1. Implement completely (zero placeholders or `TODO` stubs).
2. Re-read as domain expert; replace cheap approximations.
3. Hunt correctness, integration, security, portability, and evidence defects.
4. Low-cost polish; repeat until a full pass finds nothing.
**Final report audit:** Reread original request, reconcile `PLAN.md`, re-measure every count immediately before reporting.

## 3-Tier Hierarchy & Wave Dispatch
- **Leaf:** 4-pass clean + unit gates verified.
- **Branch:** Re-run child gates + interface compatibility + cross-boundary integration + regression checks.
- **Root:** Original request reread + contract reconciled + all branches integrated + figures re-measured.
- **Dispatch waves:** Assign disjoint `OWNS:` paths -> open wave for ready independent leaves -> seal wave -> await completion -> parent re-verifies independently -> release lease -> rolling dispatch.

## Session Operating Rules
1. **Scoped autonomy:** Autonomously edit files under `OWNS:` for implementation; keep audits read-only until scoped; confirm destructive ops (`rm -rf`, force push, drop table).
2. **Trust nothing; re-measure everything.**
3. **Never commit or push without explicit instruction** (`git status` + `git diff` first).
4. **Three-turn failure rule:** After 3 failed attempts, stop, name the suspect assumption, and ask ONE diagnostic question.
