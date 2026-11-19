---
name: token-optimisation
description: >
  Core token-saving suite combining graphify (Rust knowledge-graph indexing),
  rtk (Rust terminal noise squishing), ponytail (<=100-line minimal code &
  complexity review), and caveman (leveled terse output compression).
---

# Token Optimisation Suite

Active continuously. Four layers work together without overlap:

1. **Input / Context (`graphify.md`):** Index the repo once with `./bin/graph-index build .` and query symbols instead of rereading whole files.
2. **Tool Output (`rtk.md`):** Wrap noisy commands with `./bin/rtk-squish <cmd>` (or native `rtk`) before output enters context.
3. **Code Output (`ponytail.md`):** Target <=100 lines per module/script. Delete before adding; reuse stdlib/native before custom code.
4. **Prose Output (`caveman.md`):** Compress chat replies (`lite` / `full` / `ultra` / `wenyan-*`) while keeping technical facts exact.
