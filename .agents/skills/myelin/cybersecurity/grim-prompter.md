# Grim-Prompter — Evidence-Traced Threat Modeling

Every threat claim must trace to exact `file:line` evidence in the codebase. Never invent hypothetical architecture.

## 5-Step Threat Model Workflow
1. **System & Data-Flow Map:** Enumerate entry points (routes, CLI args, queues, webhooks), data stores, and external calls with `file:line` anchors.
2. **Trust Boundaries:** Mark every boundary where privilege level or data trustworthiness changes (anonymous -> authenticated, user -> admin, tenant A -> tenant B, external -> internal).
3. **Threat Identification:** At each boundary, check:
   - Spoofing / auth bypass
   - Tampering / mass assignment / injection
   - Information disclosure / error leaks / PII in logs
   - Elevation of privilege / IDOR / BOLA / SSRF
   - Denial of service / unbounded loops or allocations
4. **Existing Controls Analysis:** Inspect the actual middleware/guard code (`file:line`) — verify it cannot be bypassed by alternate routes, missing middleware, or race conditions.
5. **Actionable Mitigations:** Propose the smallest root-cause fix at the shared boundary, with verification criteria.
