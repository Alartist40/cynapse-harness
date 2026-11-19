#!/usr/bin/env bash
set -euo pipefail
mkdir -p bin
rustc -O src/graph_index.rs -o bin/graph-index
rustc -O src/rtk_squish.rs -o bin/rtk-squish
rustc -O src/skill_scan.rs -o bin/skill-scan
echo "Built bin/graph-index, bin/rtk-squish, and bin/skill-scan"
