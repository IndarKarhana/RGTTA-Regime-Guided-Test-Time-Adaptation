"""Regression checks for forecast origins and delayed target availability."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from baseline_forecaster import BaselineForecaster
from tta_forecaster import TTAForecaster
from ewc_forecaster import EWCForecaster
from dynatta_forecaster import DynaTTAForecaster
from rgtta_forecaster import RGTTAForecaster
from rgtta_dynatta_forecaster import RGTTADynaTTAForecaster
from run_unified_benchmark import UnifiedBenchmark
from regime_forecasting.core.forecaster import CorrectedRegimeForecaster


@pytest.mark.parametrize("forecaster_class", [
    BaselineForecaster, TTAForecaster, EWCForecaster, DynaTTAForecaster,
    RGTTAForecaster, RGTTADynaTTAForecaster, CorrectedRegimeForecaster,
])
@pytest.mark.parametrize("multivariate", [False, True])
def test_prediction_uses_latest_observations(forecaster_class, multivariate, tmp_path):
    df = pd.DataFrame({
        "unique_id": "series", "ds": pd.date_range("2020-01-01", periods=30),
        "y": np.arange(30, dtype=float), "covariate": np.arange(30, dtype=float)[::-1],
    })
    kwargs = {"storage_path": str(tmp_path)} if forecaster_class is CorrectedRegimeForecaster else {}
    fc = forecaster_class(sequence_length=8, forecast_horizon=4, **kwargs)
    fc.exog_cols = []
    features = ["covariate"] if multivariate else None
    scaled, _ = fc.preprocessor.fit_transform(df, feature_cols=features)
    fc._scaled_feature_cols = ["y_scaled", "covariate_scaled"] if multivariate else None

    class CaptureInput(torch.nn.Module):
        def forward(self, x, exog=None):
            self.observed = x.detach().cpu().numpy()
            return torch.zeros((len(x), 4), device=x.device)

    fc.model = CaptureInput()
    fc.predict(df, steps_ahead=4)
    columns = fc._scaled_feature_cols or ["y_scaled"]
    np.testing.assert_allclose(fc.model.observed[0], scaled[columns].values[-8:], atol=1e-7)


def test_benchmark_does_not_reveal_scored_targets(tmp_path, monkeypatch):
    df = pd.DataFrame({
        "unique_id": "series", "ds": pd.date_range("2020-01-01", periods=20),
        "y": np.arange(20, dtype=float),
    })
    batches = [df.iloc[i:i + 4].copy() for i in (8, 12, 16)]

    class Loader:
        DATASET_CONFIGS = {"test": {"synthetic": True}}

        def prepare_incremental_batches(self, *args, **kwargs):
            return df.iloc[:8].copy(), batches

    initialization_draws = []

    class Forecaster:
        def fit(self, history, **kwargs):
            self.seen = history.copy()
            initialization_draws.append((np.random.random(), torch.rand(1).item()))

        def update_with_new_data(self, batch):
            self.seen = pd.concat([self.seen, batch])
            return {"status": "completed"}

        def predict(self, context, steps_ahead):
            pd.testing.assert_frame_equal(context.reset_index(drop=True), self.seen.reset_index(drop=True))
            # Perfect extrapolation only if scoring starts AFTER observed data.
            return pd.DataFrame({"y_pred": self.seen.y.iloc[-1] + np.arange(1, steps_ahead + 1)})

    monkeypatch.setattr("run_unified_benchmark.StandardBenchmarkLoader", Loader)
    bench = UnifiedBenchmark(results_dir=str(tmp_path), policies=["tta", "ewc"])
    monkeypatch.setattr(bench, "_build_forecasters", lambda *args, **kwargs: {
        "tta": Forecaster(), "ewc": Forecaster(),
    })
    result = bench.run_single("gru_small", "test", 2, 42)
    assert initialization_draws[0] == initialization_draws[1]
    assert result["tta"]["mse_mean"] == 0
    assert result["tta"]["n_batches_evaluated"] == 2
    for batch in result["tta"]["batch_metrics"]:
        assert batch["forecast_origin"] < batch["target_start"] <= batch["target_end"]


def test_failed_experiments_cannot_report_success(tmp_path, monkeypatch):
    bench = UnifiedBenchmark(results_dir=str(tmp_path), n_seeds=1,
                             model_keys=["gru_small"], forecast_horizons=[2], policies=["tta"])
    monkeypatch.setattr(bench, "run_single", lambda *args: None)
    with pytest.raises(RuntimeError, match="Incomplete benchmark: 0/1"):
        bench.run_full(datasets=["synth_recurring"])
