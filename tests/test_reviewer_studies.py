import numpy as np
import pandas as pd
import pytest
import torch
from runpod.run_matrix import ROOT
from benchmarks.rgtta_forecaster import RGTTAForecaster
from benchmarks.rgtta_dynatta_forecaster import RGTTADynaTTAForecaster


@pytest.mark.parametrize('cls', [RGTTAForecaster, RGTTADynaTTAForecaster])
def test_rejected_checkpoint_restores_actual_weights(cls):
    frame = pd.DataFrame({'ds': pd.date_range('2020-01-01', periods=48, freq='h'),
                          'y': np.sin(np.arange(48) / 3), 'unique_id': 'test'})
    fc = cls(sequence_length=8, forecast_horizon=4, hidden_dim=8, num_layers=1,
             season_length=4, ckpt_gate=0.0)
    fc.fit(frame.iloc[:32], epochs=1)
    before = {k: v.clone() for k, v in fc.model.state_dict().items()}
    candidate = {k: v + 0.5 for k, v in before.items()}
    fc._memory.query = lambda *a, **kw: (1.0, candidate)
    fc.max_steps = 0
    fc.update_with_new_data(frame.iloc[32:])
    for name, value in fc.model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


@pytest.mark.parametrize('key', ['itransformer', 'patchtst'])
def test_adapter_device_and_backward(key):
    from benchmarks.run_unified_benchmark import MODEL_REGISTRY
    from regime_forecasting.models.adapter import inject_adapters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MODEL_REGISTRY[key]['class'](input_dim=2, hidden_dim=16, num_layers=1,
              num_heads=2, forecast_horizon=4).to(device)
    inject_adapters(model, key)
    output = model(torch.randn(2, 96, 2, device=device))
    output.sum().backward()
    assert all(p.device.type == device for p in model.adapters.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.adapters.parameters())


def test_dependence_signature_detects_shift_with_unchanged_marginals():
    from benchmarks.regime_features import dependence_signature
    from benchmarks.rgtta_forecaster import _multivariate_distribution_features
    rng = np.random.default_rng(3)
    y = rng.normal(size=1000)
    a = pd.DataFrame({'y': y, 'x': y})
    b = pd.DataFrame({'y': y, 'x': y[::-1]})
    ma = _multivariate_distribution_features(a, ['x'])
    mb = _multivariate_distribution_features(b, ['x'])
    np.testing.assert_allclose(ma, mb)
    sa = dependence_signature(a, ['x'], ma)
    sb = dependence_signature(b, ['x'], mb)
    assert np.linalg.norm(sa-sb) > 0.5
    constant = pd.DataFrame({'y': [1.0]*30, 'x': [float('nan')]*30})
    assert np.isfinite(dependence_signature(constant, ['x'], np.zeros(7))).all()
