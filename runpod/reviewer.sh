#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
bash runpod/setup.sh > logs/setup.log 2>&1
.venv-runpod/bin/python -m pytest tests/test_reviewer_studies.py tests/test_run_matrix.py -q > logs/reviewer-tests.log 2>&1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv-runpod/bin/python -u runpod/reviewer_studies.py > logs/reviewer.log 2>&1
