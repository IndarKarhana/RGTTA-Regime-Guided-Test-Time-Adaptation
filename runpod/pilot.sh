#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# This bounds job execution, not Pod billing. Stop the Pod through RunPod afterwards.
timeout --signal=TERM --kill-after=30s 3600 \
  .venv-runpod/bin/python benchmarks/run_unified_benchmark.py \
  --device cuda --workers 1 --seeds 1 --horizons 96 \
  --models itransformer patchtst --datasets synth_recurring \
  --policies tta rgtta --results-dir benchmarks/results/runpod_gpu_pilot
