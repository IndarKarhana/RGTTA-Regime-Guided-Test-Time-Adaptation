import json
import pytest
from runpod.run_matrix import save, validate


def test_checkpoint_atomic_and_invalid_results(tmp_path):
    path = tmp_path / 'job.json'
    valid = {'tta': {'n_batches_evaluated': 9, 'mse_mean': 1.0, 'mae_mean': 0.5, 'total_time': 2.0}}
    validate(valid, ['tta'])
    save(path, valid)
    with pytest.raises(ValueError):
        save(path, {'bad': float('nan')})
    assert json.loads(path.read_text()) == valid
    for field, value in [('n_batches_evaluated', 0), ('mse_mean', float('nan'))]:
        bad = {'tta': {**valid['tta'], field: value}}
        with pytest.raises(ValueError):
            validate(bad, ['tta'])
    with pytest.raises(KeyError):
        validate(valid, ['tta', 'ewc'])


@pytest.mark.parametrize('name', ['ewc', 'rgtta'])
def test_gru_fisher_backward_in_eval_mode(name):
    import torch
    from benchmarks.ewc_forecaster import EWCForecaster
    from benchmarks.rgtta_forecaster import RGTTAForecaster
    from regime_forecasting.models.transformer import TimeSeriesTransformer
    cls = EWCForecaster if name == 'ewc' else RGTTAForecaster
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fc = cls(device=device)
    fc.model = TimeSeriesTransformer(hidden_dim=8, num_layers=1, forecast_horizon=2).to(device)
    enabled = torch.backends.cudnn.enabled
    fisher = fc._compute_fisher(torch.randn(2, 8, 1, device=device), None,
                                torch.randn(2, 2, device=device), n_samples=2)
    assert not fc.model.training
    assert torch.backends.cudnn.enabled == enabled
    assert fisher and all(torch.isfinite(value).all() for value in fisher.values())
    assert sum(value.sum().item() for value in fisher.values()) > 0


def test_rnn_grad_preserves_eval_output_and_restores_on_error():
    import torch
    from regime_forecasting.utils.rnn_grad import evaluation_rnn_grad
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.nn.GRU(3, 4, num_layers=2, dropout=0.5, batch_first=True).to(device).eval()
    inputs = torch.randn(2, 8, 3, device=device)
    expected = model(inputs)[0].detach()
    with pytest.raises(RuntimeError, match='intentional'):
        with evaluation_rnn_grad(model):
            actual = model(inputs)[0]
            torch.testing.assert_close(actual.detach(), expected)
            actual.sum().backward()
            raise RuntimeError('intentional')
    assert not model.training and model.dropout == 0.5
