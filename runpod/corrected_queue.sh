#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
while ! test -f logs/full.exit; do sleep 5; done
.venv-runpod/bin/python -u runpod/run_matrix.py --device cuda \
  --models itransformer patchtst dlinear gru_small \
  --policies rgtta rgtta_ewc rgtta_dynatta \
  --results-dir benchmarks/results/corrected_rg > logs/corrected_rg.log 2>&1
corrected_status=$?
printf '%s\n' "$corrected_status" > logs/corrected_rg.exit
while ! test -f logs/official.ready; do sleep 10; done
bash runpod/official.sh > logs/official.log 2>&1
official_status=$?
printf '%s\n' "$official_status" > logs/official.exit
if (( corrected_status != 0 || official_status != 0 )); then exit 1; fi
