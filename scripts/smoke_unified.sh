#!/bin/bash
cd "/Users/indarkumar/Documents/Incremental_learning research"
export PYTHONPATH=".:src:benchmarks:benchmarks/data_loaders"
exec .venv/bin/python benchmarks/run_unified_benchmark.py \
  --seeds 1 \
  --horizons 720 \
  --models gru_small lstm gru_large transformer dlinear \
  2>&1 | tee benchmarks/results/smoke_unified_full.log
