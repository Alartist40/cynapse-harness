# SlopMonster — Anti-AI-Slop Discipline

Eliminate predictable LLM writing tics, hollow filler, and synthetic enthusiasm across all prose, docs, comments, and explanations.

## Banned Patterns
1. **Slop vocabulary:** `delve`, `tapestry`, `testament`, `crucial`, `pivotal`, `paramount`, `landscape`, `realm`, `beacon`, `synergy`, `elevate`, `foster`, `nuanced`, `comprehensive`, `leverage` (use `use`), `utilize` (use `use`), `robust`, `seamless`.
2. **Filler transitions:** `Moreover,`, `Furthermore,`, `Additionally,`, `It's worth noting that`, `Importantly,`, `In conclusion,`, `To summarize,`, `At the end of the day`.
3. **Fake enthusiasm & throat-clearing:** `Great question!`, `Certainly!`, `I'd be happy to help`, `Let's dive in`, `Here's a breakdown:`.
4. **Structural tells:**
   - **Padded triplets:** Do not force every list or clause into three parallel adjectives/nouns just for rhythm.
   - **Em-dash overuse:** Limit `—` to genuine parentheticals; use periods or commas instead.
   - **Bold-lead bullet spam:** Avoid `**Concept:** explanation` on every bullet when plain direct sentences are clearer.
   - **Self-answering rhetorical questions:** State the fact directly.

## Cross-Check Rule
Before finalizing human-facing prose (docs, comments, PR descriptions, reports), re-read as a skeptical human editor: cut any sentence that states the obvious, replace abstract nouns with concrete verbs, and verify every claim has a specific file, symbol, or measurement.

## UI & Web Design Anti-Slop
When building interfaces (`frontend-design/`), reject generic AI-generated frontend patterns:
- **Banned visual clichés:** Unrequested dark-mode purple/indigo SaaS gradients, gratuitous glassmorphism (`backdrop-blur` on every card), glowing neon borders, and rainbow pill badges.
- **Banned layout clichés:** Repetitive 3-column icon-above-title feature grids, fake metric counters (`99.9% uptime`, `10k+ users` unless backed by real data), and generic hero+3-cards boilerplate.
- **Banned placeholder copy:** Never ship `Acme Corp`, `Elevate your workflow`, `Seamless experience`, `Next-gen`, or `Lorem ipsum` — write real, domain-specific copy.
- **Craft check:** Vary section rhythm to fit the actual content, verify `>=4.5:1` contrast, and test mobile + desktop layouts before reporting done.
