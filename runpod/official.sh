#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
# Separate dependencies from the corrected benchmark environment.
python3 -m venv --system-site-packages .venv-official
.venv-official/bin/python -m pip install -r runpod/requirements.txt -r runpod/official-requirements.txt || exit 1
.venv-official/bin/python -m pip freeze > benchmarks/results/official-environment.txt
failed=0
for seed in 0 1 2; do
  for dataset in ETTh1 ETTh2; do
    output="benchmarks/results/official_dynatta/${dataset}_96_seed${seed}"
    mkdir -p "$output"
    .venv-official/bin/python -u official/official_reproduction.py \
      --framework official/framework --results-dir "$output" \
      --dataset "$dataset" --seed "$seed" > "$output/run.log" 2>&1
    status=$?
    printf '%s\n' "$status" > "$output/exit.txt"
    if (( status != 0 )); then failed=1; fi
  done
done
exit "$failed"
