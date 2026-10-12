"""Retraining must optimize the layers created by DLinear's first forward."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from regime_forecasting.core.forecaster import CorrectedRegimeForecaster
from regime_forecasting.models.dlinear_model import DLinearForecaster


def test_retrain_initializes_and_updates_lazy_parameters(tmp_path, monkeypatch):
    fc = CorrectedRegimeForecaster(sequence_length=8, forecast_horizon=4,
                                  storage_path=str(tmp_path), model_class=DLinearForecaster)
    fc.model = DLinearForecaster(forecast_horizon=4, season_length=3)
    assert list(fc.model.parameters()) == []
    original_adam = torch.optim.Adam
    before = []

    def capture_adam(parameters, **kwargs):
        parameters = list(parameters)
        assert parameters
        before.extend((p, p.detach().clone()) for p in parameters)
        return original_adam(parameters, **kwargs)

    monkeypatch.setattr(torch.optim, 'Adam', capture_adam)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(12, 8, 1)).astype(np.float32)
    y = rng.normal(size=(12, 4)).astype(np.float32)
    fc._run_training_loop(x, None, y, epochs=1, batch_size=4,
                          learning_rate=0.0005, validation_split=0.2)
    assert any(not torch.equal(p.detach(), initial) for p, initial in before)
