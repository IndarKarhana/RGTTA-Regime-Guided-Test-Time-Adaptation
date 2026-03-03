#!/bin/bash
# Full benchmark: 7 policies × 4 models × 14 datasets × 3 horizons × 5 seeds
# Launched: $(date)
cd "$(dirname "$0")/.."
export PYTHONPATH=".:src:benchmarks:benchmarks/data_loaders"
mkdir -p benchmarks/results/unified
exec .venv/bin/python benchmarks/run_unified_benchmark.py \
    --seeds 5 \
    --datasets all \
    --workers 4
