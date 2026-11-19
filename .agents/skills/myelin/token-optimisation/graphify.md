# Graphify — Rust Knowledge-Graph Indexing

Use `./bin/graph-index` (pure stdlib Rust binary) at session start and before exploring unfamiliar directories so you never waste tokens rereading files.

## Commands
- **Index directory:** `./bin/graph-index build [dir]` (writes `.myelin-graph.tsv`, skips `.git`, `node_modules`, `target`, `dist`, `build`).
- **Compact map:** `./bin/graph-index map [dir]` (1 line per file with line count and top symbols + line numbers).
- **Search symbols/files:** `./bin/graph-index query <term> [dir]` (returns only matching files and exact `kind symbol @ L<line>` anchors).

## Workflow
1. On setup or new repo: run `build` + `map`.
2. Before editing or debugging: `query <symbol>` to jump straight to the exact file and line range, then read only those lines.
3. Re-run `build` after creating or heavily restructuring files.
