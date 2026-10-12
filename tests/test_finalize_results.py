import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runpod.finalize_results import summarize, validate_metric


def valid_metric():
    return {'n_batches_evaluated': 1, 'mse_mean': 2.0, 'mae_mean': 1.0, 'total_time': 1.0,
            'batch_metrics': [{'forecast_origin': '2020-01-01', 'target_start': '2020-01-02',
                               'target_end': '2020-01-03', 'mse': 2.0, 'mae': 1.0,
                               'batch_time': 0.5, 'per_step_mse': [2., 2.], 'per_step_mae': [1., 1.]}]}


def test_validates_observed_then_future_errors():
    assert len(validate_metric(valid_metric(), 2, 1)) == 1


@pytest.mark.parametrize('damage', ['future', 'nonfinite', 'aggregation', 'horizon'])
def test_rejects_invalid_results(damage):
    m=copy.deepcopy(valid_metric())
    if damage=='future': m['batch_metrics'][0]['forecast_origin']='2020-01-04'
    if damage=='nonfinite': m['batch_metrics'][0]['mse']=float('nan')
    if damage=='aggregation': m['mse_mean']=3.
    if damage=='horizon': m['batch_metrics'][0]['per_step_mse']=[2.]
    with pytest.raises(ValueError): validate_metric(m, 2, 1)


def test_seed_summary_rejects_pseudoreplication():
    rows=[dict(model='x', seed=s, mse=1.) for s in [0, 0, 2]]
    with pytest.raises(ValueError): summarize(rows, ['model'], ['mse'])
