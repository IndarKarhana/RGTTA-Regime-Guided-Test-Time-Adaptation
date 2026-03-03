"""
Benchmarking module for RGTTA incremental forecasting experiments.

7 update policies × 4 model architectures × ETT datasets.
Entry point: run_unified_benchmark.py
"""

from .baseline_forecaster import BaselineForecaster

__all__ = ["BaselineForecaster"]
