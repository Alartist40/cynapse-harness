---
name: building
description: >
  Spec-Driven Development (SDD) workflow adapted from github/spec-kit.
  Auto-activates when building new features, greenfield projects, or handling
  ambiguous implementation requests so the model never builds the wrong thing
  from assumptions.
---

# Building & Spec-Driven Development

**Trigger:** Activate when starting a new build, greenfield project, or ambiguous feature request. Skip for small, unambiguous bug fixes (`surgical-patch`).

## 4-Stage Build Flow (`spec-kit.md`)
1. **Specify:** Lock what the user wants built, who uses it, explicit `In-Scope` / `Out-of-Scope` boundaries, and testable acceptance criteria *before* writing code.
2. **Plan:** Define minimal architecture, data flow, and tech stack (`ponytail` rules apply: reuse existing patterns, stdlib/native first).
3. **Tasks:** Break into atomic stages with disjoint `OWNS:` paths and `GATES:` (hand off to `harness-execution/` for multi-file builds).
4. **Implement:** Build stage-by-stage with `ponytail` (`<=100`-line modules) and `cybersecurity/` pre-flight checks.
