#!/bin/bash
# Launch streaming benchmark on VM1 with ALL 12 cores
gcloud compute ssh rgtta-benchmark --zone=us-central1-a --command='
cat > /tmp/run_streaming.sh << "SCRIPT"
#!/bin/bash
cd ~/rgtta
export OMP_NUM_THREADS=1
export PYTHONPATH=".:src:benchmarks:benchmarks/data_loaders"
rm -rf benchmarks/results/unified/ckpt_* 2>/dev/null
.venv/bin/python benchmarks/run_unified_benchmark.py \
  --seeds 5 \
  --horizons 96 192 336 720 \
  --models gru_small itransformer gru_large patchtst dlinear \
  --workers 12
SCRIPT
chmod +x /tmp/run_streaming.sh
nohup /tmp/run_streaming.sh > ~/rgtta/benchmark_streaming_v3.log 2>&1 &
echo "PID=$!"
'
