---
name: frontend-design
description: >
  Intentional UI/web design discipline adapted from elayadesign/ai-design-skills
  and paired with SlopMonster UI anti-slop rules. Auto-activates when building
  frontend, web pages, dashboards, or visual components.
---

# Frontend & Design Discipline

**Trigger:** Activate only when building UI, web pages, dashboards, or visual components. Costs 0 tokens on backend/CLI/ops tasks.

## Execution Order
1. **Design Brief (`ai-design-skills.md`):** Clarify audience, primary user action, visual tone, and constraints before writing markup/CSS.
2. **Visual System (`ai-design-skills.md`):** Lock typography, spacing rhythm, constrained color palette, contrast (>=4.5:1), and purposeful motion.
3. **UI Anti-Slop (`../response-optimiser/slopmonster.md`):** Reject generic AI SaaS templates, fake metrics, and placeholder copy.
4. **Native-First (`../token-optimisation/ponytail.md`):** Use native HTML (`<dialog>`, `<details>`, `<input>`) and modern CSS (`grid`, `flex`, `:has()`, variables) before adding UI libraries.
