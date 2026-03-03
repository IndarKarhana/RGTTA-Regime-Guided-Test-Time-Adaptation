#!/bin/bash
# ==========================================================================
# Cloud Benchmark Setup — RGTTA Full Benchmark on AWS EC2
# ==========================================================================
#
# Target: c7a.24xlarge (96 vCPUs, 192 GB RAM, ~$3.60/hr)
# Expected: 504 experiments (3 seeds) in ~1.5-2 hours ≈ $5-8
#           840 experiments (5 seeds) in ~2-3 hours ≈ $8-12
#
# STEP 1: Launch EC2 instance
# ---------------------------
# From your local machine (AWS CLI must be configured):
#
#   aws ec2 run-instances \
#     --image-id ami-0c7217cdde317cfec \
#     --instance-type c7a.24xlarge \
#     --key-name YOUR_KEY_NAME \
#     --security-group-ids sg-XXXXXXXX \
#     --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":50}}]' \
#     --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=rgtta-benchmark}]' \
#     --query 'Instances[0].InstanceId' \
#     --output text
#
# Wait for it to start, then get the public IP:
#   aws ec2 describe-instances --instance-ids i-XXXXX \
#     --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
#
# SSH in:
#   ssh -i ~/.ssh/YOUR_KEY.pem ubuntu@<PUBLIC_IP>
#
# ==========================================================================

set -euo pipefail

echo "=============================================="
echo "  RGTTA Cloud Benchmark Setup"
echo "=============================================="

# STEP 2: System dependencies
echo "[1/5] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3.11 python3.11-venv python3-pip git

# STEP 3: Clone repo
echo "[2/5] Cloning repository..."
cd /home/ubuntu
if [ ! -d "rgtta-benchmark" ]; then
    # Option A: Clone from git (if repo is on GitHub)
    # git clone https://github.com/YOUR_USER/incremental-learning-research.git rgtta-benchmark

    # Option B: If not on GitHub, you'll scp the project first (see below)
    echo "⚠️  Please upload your project first. See instructions below."
    echo ""
    echo "From your LOCAL machine, run:"
    echo "  tar czf /tmp/rgtta-project.tar.gz -C '/Users/indarkumar/Documents/Incremental_learning research' \\"
    echo "    benchmarks/ src/ data/ pyproject.toml"
    echo "  scp -i ~/.ssh/YOUR_KEY.pem /tmp/rgtta-project.tar.gz ubuntu@<IP>:/home/ubuntu/"
    echo ""
    echo "Then on the EC2 instance:"
    echo "  mkdir -p rgtta-benchmark && tar xzf rgtta-project.tar.gz -C rgtta-benchmark"
    echo ""
    exit 1
fi

cd rgtta-benchmark

# STEP 4: Python environment
echo "[3/5] Setting up Python environment..."
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
pip install -q torch --index-url https://download.pytorch.org/whl/cpu
pip install -q numpy pandas scipy scikit-learn statsmodels

# STEP 5: Verify setup
echo "[4/5] Verifying setup..."
export PYTHONPATH=".:src:benchmarks:benchmarks/data_loaders"
python -c "
from run_unified_benchmark import UnifiedBenchmark
print('✅ Imports OK')
import multiprocessing as mp
print(f'✅ CPUs available: {mp.cpu_count()}')
"

# STEP 6: Run benchmark
echo "[5/5] Launching benchmark..."
echo ""
echo "=============================================="
echo "  Ready to run! Choose your configuration:"
echo "=============================================="
echo ""
echo "  # 3 seeds (recommended for arXiv, ~1.5 hrs):"
echo "  nohup python benchmarks/run_unified_benchmark.py \\"
echo "    --seeds 3 --datasets all --workers 48 \\"
echo "    > /tmp/benchmark_cloud.log 2>&1 &"
echo ""
echo "  # 5 seeds (gold standard, ~2.5 hrs):"
echo "  nohup python benchmarks/run_unified_benchmark.py \\"
echo "    --seeds 5 --datasets all --workers 48 \\"
echo "    > /tmp/benchmark_cloud.log 2>&1 &"
echo ""
echo "  # Monitor progress:"
echo "  tail -f /tmp/benchmark_cloud.log | grep -E '✅|COMPLETE|FAILED'"
echo ""
echo "  # When done, copy results back to your Mac:"
echo "  # (from your LOCAL machine)"
echo "  scp -i ~/.ssh/YOUR_KEY.pem ubuntu@<IP>:/home/ubuntu/rgtta-benchmark/benchmarks/results/unified/unified_results.json ."
echo "  scp -i ~/.ssh/YOUR_KEY.pem ubuntu@<IP>:/home/ubuntu/rgtta-benchmark/benchmarks/results/unified/unified_report.md ."
echo ""
echo "  # IMPORTANT: Terminate instance when done!"
echo "  # aws ec2 terminate-instances --instance-ids i-XXXXX"
echo "=============================================="
