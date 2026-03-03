"""
Smoke tests for TTA baseline (same base model, different update policy).
Ensures TTA forecaster fits, updates, and predicts without affecting core forecaster.
"""
import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))

from regime_forecasting.models.transformer import TimeSeriesTransformer
from baseline_forecaster import BaselineForecaster
from tta_forecaster import TTAForecaster


@pytest.fixture
def sample_data():
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=80, freq="ME")
    y = 100 + 0.5 * np.arange(80) + 10 * np.sin(2 * np.pi * np.arange(80) / 12) + np.random.normal(0, 5, 80)
    return pd.DataFrame({"unique_id": "ts_001", "ds": dates, "y": y})


def test_tta_fit_and_predict(sample_data):
    """TTA forecaster fits on initial data and predicts."""
    tta = TTAForecaster(season_length=12, forecast_horizon=6, sequence_length=24, tta_steps=5)
    res = tta.fit(sample_data.head(60), epochs=3)
    assert res["status"] == "completed"
    ctx = sample_data.head(50)
    pred = tta.predict(ctx, steps_ahead=6)
    assert pred is not None
    assert len(pred) == 6
    assert "y_pred" in pred.columns


def test_tta_update_with_new_data(sample_data):
    """TTA update_with_new_data runs without error (adapt on new batch only)."""
    tta = TTAForecaster(season_length=12, forecast_horizon=6, sequence_length=24, tta_steps=10)
    tta.fit(sample_data.head(50), epochs=3)
    new_batch = sample_data.iloc[50:62]
    out = tta.update_with_new_data(new_batch)
    assert out["status"] in ("completed", "skipped")
    if out["status"] == "completed":
        assert "tta_time" in out


def test_tta_same_base_model_as_baseline(sample_data):
    """TTA and Baseline both use TimeSeriesTransformer (same architecture)."""
    baseline = BaselineForecaster(season_length=12, forecast_horizon=6, sequence_length=24, hidden_dim=32, num_layers=1)
    baseline.fit(sample_data.head(50), epochs=2)
    tta = TTAForecaster(season_length=12, forecast_horizon=6, sequence_length=24, hidden_dim=32, num_layers=1, tta_steps=5)
    tta.fit(sample_data.head(50), epochs=2)
    assert isinstance(baseline.model, TimeSeriesTransformer)
    assert isinstance(tta.model, TimeSeriesTransformer)
    assert baseline.model.hidden_dim == tta.model.hidden_dim
    assert baseline.model.forecast_horizon == tta.model.forecast_horizon
