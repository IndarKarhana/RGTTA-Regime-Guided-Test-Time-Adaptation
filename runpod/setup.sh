#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.__version__, torch.cuda.get_device_name(0))'
python3 -m venv --system-site-packages .venv-runpod
.venv-runpod/bin/python -m pip install -r runpod/requirements.txt
.venv-runpod/bin/python -m pytest tests/test_streaming_causality.py tests/test_tta_baseline.py tests/test_benchmark_device.py -q
.venv-runpod/bin/python -m pip freeze > runpod/environment-resolved.txt
