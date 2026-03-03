#!/bin/bash
# Setup VM2 (rgtta-sliding) for sliding window benchmark
# Run this ONCE after VM creation

set -e

VM_IP="104.154.221.163"
SSH="ssh -i ~/.ssh/google_compute_engine -o StrictHostKeyChecking=no indarkumar@$VM_IP"
SCP="scp -i ~/.ssh/google_compute_engine -o StrictHostKeyChecking=no"
LOCAL_ROOT="/Users/indarkumar/Documents/Incremental_learning research"

echo "=== Step 1: Install Python 3.11, pip, venv ==="
gcloud compute ssh rgtta-sliding --zone=us-central1-a --project=project-research-488102 --command='
sudo apt-get update -qq
sudo apt-get install -y -qq python3.11 python3.11-venv python3-pip git
echo "PYTHON_DONE"
'

echo "=== Step 2: Create project dir and venv ==="
gcloud compute ssh rgtta-sliding --zone=us-central1-a --project=project-research-488102 --command='
mkdir -p ~/rgtta/benchmarks/data_loaders ~/rgtta/benchmarks/results/sliding ~/rgtta/src ~/rgtta/data/benchmarks
python3.11 -m venv ~/rgtta/.venv
echo "VENV_DONE"
'

echo "=== Step 3: Install Python deps ==="
gcloud compute ssh rgtta-sliding --zone=us-central1-a --project=project-research-488102 --command='
~/rgtta/.venv/bin/pip install -q torch numpy pandas scipy scikit-learn statsmodels huggingface_hub datasets
echo "DEPS_DONE"
'

echo "=== Step 4: Rsync code ==="
rsync -avz --delete \
  -e "ssh -i ~/.ssh/google_compute_engine -o StrictHostKeyChecking=no" \
  "$LOCAL_ROOT/src/" \
  indarkumar@$VM_IP:~/rgtta/src/

rsync -avz \
  -e "ssh -i ~/.ssh/google_compute_engine -o StrictHostKeyChecking=no" \
  --exclude='results/' --exclude='checkpoints*/' --exclude='data/' --exclude='__pycache__/' \
  "$LOCAL_ROOT/benchmarks/" \
  indarkumar@$VM_IP:~/rgtta/benchmarks/

echo "=== Step 5: Test imports ==="
gcloud compute ssh rgtta-sliding --zone=us-central1-a --project=project-research-488102 --command='
cd ~/rgtta && PYTHONPATH=".:src:benchmarks:benchmarks/data_loaders" .venv/bin/python -c "
from regime_forecasting.models.itransformer_model import iTransformerForecaster
from regime_forecasting.models.patchtst_model import PatchTSTForecaster
from data_loaders.standard_benchmarks import StandardBenchmarkLoader
print(\"ALL_IMPORTS_OK\")
"
'

echo "=== SETUP COMPLETE ==="
