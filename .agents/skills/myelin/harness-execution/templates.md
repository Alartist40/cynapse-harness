# Harness Artifact Templates

## Permission Triad (`AGENTS.md`)
- **Allowed:** Read files, run tests/linters, edit files within `OWNS:` scope.
- **Restricted (Ask first):** Install dependencies, modify DB schemas, drop tables, destructive commands.
- **Forbidden:** Commit/push without diff review, hardcode secrets, out-of-scope edits.

## `PLAN.md`
```markdown
# Task: <Title>

## Boundaries
- In-Scope: <explicit targets>
- Out-of-Scope: <explicit non-goals>

## Stages
- [ ] Stage 1: <Name>
  - OWNS: <exact file paths>
  - GATES: G1, G2
```

## `GATES.md`
```markdown
# Acceptance Gates

- [ ] G1: <Observable outcome>
  CHECK: <portable shell command>
  EXPECT: <decisive success token>
  EVIDENCE: pending
```

## `IMPLEMENT.md`
```markdown
# Implementation Log (Append-Only)

## [<Timestamp / Stage ID>]
- Action: <what was executed>
- Decision & Rationale: <why>
- Deviations: <any deviation from PLAN.md>
- Next: <immediate next stage>
```
