# RTK — Rust Terminal Output Squisher

Use `./bin/rtk-squish` (pure stdlib Rust binary, or native `rtk` if installed) on noisy shell commands so terminal boilerplate never floods the context window.

## Usage
- **Wrap command:** `./bin/rtk-squish <command> [args...]`
- **Pipe stdin:** `<command> | ./bin/rtk-squish`

## What It Does
1. Strips ANSI color/cursor codes and `\r` spinner frames.
2. Drops blank lines and collapses consecutive identical lines into `<line> [xN]`.
3. Preserves exit codes and high-signal lines (`error`, `fail`, `exception`, `traceback`, `panic`, `warn`, `diff --git`, `@@`, `file:line`).
4. Compresses outputs >60 lines to head (15L) + mid signals (<=20L) + tail (25L).

## When to Use
Always wrap: `git diff`, `git log`, `cargo test`, `pytest`, `npm test`, `docker logs`, `find`, or any build/lint run.
