# Strix — White-Box Security Review & PoC Validation

## Authorization & Environment
- Only test assets the user owns or explicitly authorizes. Prefer staging over production; never test third parties or fabricate credentials.
- **Dual-target execution:** Pass source + running instance together (`-t ./ -t http://localhost:3000`). Static reasoning alone suggests; live targets prove.
- **Multi-tenant credentials:** Two tenant accounts (Org A, Org B) + one admin account are required to prove cross-tenant IDOR/BOLA (`A01`).

## Finding Discipline
- **PoC-validated only:** Only reproducible exploits count as confirmed vulnerabilities (e.g., Org A retrieving Org B's real data, not an empty `200`).
- **Observations vs. findings:** Static-only hits are labeled *observations*.
- **Rank by proven impact:** Unauthenticated-reachable > cross-tenant / priv-esc > authenticated > hardening notes. Dedupe by root cause.
- **Honest coverage:** "No findings" = nothing proven in what was analyzed. Explicitly state untested paths and black-box-unreachable categories.

## OWASP Map (Current Editions Only)
- **OWASP Top 10:2025:** `A01` Broken Access Control (incl. SSRF/IDOR) · `A02` Security Misconfiguration · `A03` Software Supply Chain Failures · `A04` Cryptographic Failures · `A05` Injection · `A06` Insecure Design · `A07` Authentication Failures · `A08` Software/Data Integrity · `A09` Logging & Alerting (not externally testable — state so) · `A10` Mishandling of Exceptional Conditions.
- **OWASP API Top 10:2023:** `API1` BOLA · `API3` Broken Object Property Authz (mass assignment) · `API5` Broken Function-Level Authz.

## Root-Cause Fix Loop
Reproduce -> patch root cause in shared handler/middleware (ORM parameterization, centralized authz, context-aware escaping, SSRF allowlist) -> re-run exploit PoC AND full test suite. Rotate and purge leaked secrets from git history.
