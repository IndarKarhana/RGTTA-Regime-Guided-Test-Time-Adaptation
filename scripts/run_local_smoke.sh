#!/bin/bash
# Local smoke tests: all 5 models, horizons 96+720, 1 seed
cd "/Users/indarkumar/Documents/Incremental_learning research"
export PYTHONPATH=".:src:benchmarks:benchmarks/data_loaders"

echo "=== Launching sliding smoke test ==="
nohup .venv/bin/python benchmarks/run_sliding_window_benchmark.py \
  --seeds 1 --horizons 96 720 \
  --models gru_small itransformer gru_large patchtst dlinear \
  --max-windows 50 \
  > benchmarks/results/smoke_sliding_v3.log 2>&1 &
SLIDE_PID=$!
echo "Sliding PID: $SLIDE_PID"

echo "=== Launching streaming smoke test ==="
nohup .venv/bin/python benchmarks/run_unified_benchmark.py \
  --seeds 1 --horizons 96 720 \
  --models gru_small itransformer gru_large patchtst dlinear \
  > benchmarks/results/smoke_streaming_v3.log 2>&1 &
STREAM_PID=$!
echo "Streaming PID: $STREAM_PID"

echo "Both launched. Sliding=$SLIDE_PID, Streaming=$STREAM_PID"
