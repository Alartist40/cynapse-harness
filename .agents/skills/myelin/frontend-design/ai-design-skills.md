# AI Design Skills — Intentional Visual Systems

Think like a product designer before writing a single line of HTML, JSX, or CSS.

## 1. Design Brief (Before Code)
- **Purpose & Primary Action:** What is the single most important action a visitor should take on this screen?
- **Audience & Tone:** Who is this for, and what visual register fits (editorial, technical/dense, utilitarian, warm consumer)?
- **Constraints:** Viewport targets (mobile-first vs desktop workbench), accessibility (`>=4.5:1` contrast, keyboard nav, `prefers-reduced-motion`).

## 2. Visual System Rules
- **Typography:** Pair at most 2 typefaces (or 1 strong sans + monospace for code/data). Set clear size hierarchy, tight heading tracking, and comfortable body measure (`60-75ch`).
- **Color Palette:** Constrained and intentional — 1 background surface, 1 elevated surface, 1 primary brand color, 1 semantic accent, and clear ink hierarchy. Avoid unrequested rainbow gradients.
- **Spacing & Grid:** Use a consistent spacing scale (4px/8px multiples). Give primary content room to breathe; group related controls tightly.
- **Components & Radii:** Keep border-radius, border weight, and shadow elevation consistent across the entire surface.
- **Purposeful Motion:** Use subtle `transform`/`opacity` transitions (`150-250ms`) to clarify state changes and hierarchy. Never animate layout properties (`width`/`top`/`left`) or add decorative bounce.
- **Copy:** Write real, domain-specific labels and microcopy — never generic filler.
