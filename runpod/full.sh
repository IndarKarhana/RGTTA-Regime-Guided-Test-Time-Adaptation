#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv-runpod/bin/python -u runpod/run_matrix.py \
  --results-dir benchmarks/results/full_matrix --device cuda --models itransformer patchtst dlinear gru_small > logs/matrix.log 2>&1
matrix_status=$?
printf '%s\n' "$matrix_status" > logs/matrix.exit
.venv-runpod/bin/python -u runpod/run_matrix.py \
  --results-dir benchmarks/results/full_retrain --device cuda --models itransformer patchtst dlinear gru_small --policies retrain > logs/retrain.log 2>&1
retrain_status=$?
printf '%s\n' "$retrain_status" > logs/retrain.exit
if (( matrix_status != 0 || retrain_status != 0 )); then exit 1; fi
