# Spec-Kit — Spec-Driven Development Workflow

Prevent assumption-driven rewrites by locking the specification before writing product code.

## Stage 1: Specify (What & Why — Zero Code)
- Restate the core problem and target user in 2-3 lines.
- List explicit **In-Scope** and **Out-of-Scope** boundaries.
- Define concrete, observable **Acceptance Criteria** (what must be true when done).
- **Ambiguity rule:** If a core requirement, data model, or UX behavior has multiple valid interpretations, ask 1-3 numbered clarifying questions *before* coding. Never guess past an unclear requirement.

## Stage 2: Plan (How — Minimal Architecture)
- Choose the simplest architecture that satisfies Stage 1 (`ponytail` ladder).
- Map data flow, interfaces, and failure/security boundaries (`grim-prompter`).
- Name exact files to create or modify.

## Stage 3: Tasks (Atomic Execution Units)
- Split into numbered, verifiable stages (`PLAN.md` + `GATES.md` when multi-file).
- Each stage has one clear deliverable, exact `OWNS:` paths, and a runnable check.

## Stage 4: Implement & Verify
- Execute one stage at a time (`<=100`-line modules where possible).
- Run `cybersecurity/` pre-flight check and `harness-execution/` verification gates before reporting done.
