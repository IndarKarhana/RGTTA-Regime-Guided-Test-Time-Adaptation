# Ablation Study Results

**Date**: 2026-02-17T19:12:36.592897

## Similarity Threshold Ablation

This tests how different similarity thresholds affect the accuracy-efficiency trade-off.

### ETTm1

| Threshold | MAPE | Match Rate | Time (s) |
|-----------|------|------------|----------|
| 0.5 | 11.21% ± 6.07% | 100.0% | 0.0 |
| 0.6 | 8.87% ± 5.28% | 100.0% | 0.1 |
| 0.7 | 8.70% ± 4.85% | 100.0% | 0.0 |
| 0.75 | 9.88% ± 5.31% | 100.0% | 0.0 |
| 0.8 | 9.64% ± 4.80% | 88.9% | 14.7 |
| 0.85 | 8.95% ± 2.80% | 77.8% | 45.8 |
| 0.9 | 7.70% ± 2.48% | 55.6% | 82.4 |

### ETTh1

| Threshold | MAPE | Match Rate | Time (s) |
|-----------|------|------------|----------|
| 0.5 | 59.19% ± 27.30% | 88.9% | 17.7 |
| 0.6 | 24.73% ± 15.41% | 77.8% | 39.8 |
| 0.7 | 26.24% ± 14.92% | 55.6% | 79.5 |
| 0.75 | 39.01% ± 24.90% | 55.6% | 81.3 |
| 0.8 | 25.74% ± 20.82% | 44.4% | 108.7 |
| 0.85 | 20.59% ± 13.53% | 22.2% | 146.2 |
| 0.9 | 31.01% ± 20.06% | 0.0% | 178.3 |

## Model Selection Strategy Ablation

This tests different training strategies when retraining is needed.

### ETTm1

| Strategy | MAPE | Time (s) |
|----------|------|----------|
| full | 10.63% ± 4.25% | 258.2 |
| partial | 6.15% ± 4.62% | 280.1 |
| adaptive | 9.14% ± 5.19% | 393.6 |

### ETTh1

| Strategy | MAPE | Time (s) |
|----------|------|----------|
| full | 31.26% ± 19.86% | 226.4 |
| partial | 16.50% ± 10.82% | 216.9 |
| adaptive | 19.07% ± 14.26% | 221.5 |
