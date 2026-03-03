#!/usr/bin/env python3
"""
Phase B1: Ablation sweep runner for RG-TTA hyperparameters.

Runs RG-TTA with varied hyperparameters + a TTA baseline for reference.
Each ablation varies ONE parameter while holding others at v2 defaults.

Usage:
  python paper/run_ablation_sweeps.py --ablation loss_gate
  python paper/run_ablation_sweeps.py --ablation all
  python paper/run_ablation_sweeps.py --ablation lr_sim_scale --datasets ETTh1 synth_recurring --seeds 0 1

Results saved to: benchmarks/results/ablation/
"""

import argparse
import json
import sys
import os
import time
import logging
from pathlib import Path

# Add project paths
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "benchmarks" / "data_loaders"))

import numpy as np
import pandas as pd
import torch

from standard_benchmarks import StandardBenchmarkLoader
from rgtta_forecaster import RGTTAForecaster
from tta_forecaster import TTAForecaster

from regime_forecasting.models.transformer import TimeSeriesTransformer
from regime_forecasting.models.itransformer_model import iTransformerForecaster
from regime_forecasting.models.patchtst_model import PatchTSTForecaster
from regime_forecasting.models.dlinear_model import DLinearForecaster
from regime_forecasting.utils.evaluation import weighted_mape, symmetric_mape

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


# ── Ablation configurations ──────────────────────────────────────────
ABLATIONS = {
    'loss_gate': {
        'param': 'ckpt_gate',
        'values': [0.50, 0.60, 0.70, 0.80, 0.90],
        'default': 0.70,
        'description': 'Checkpoint loss gate (ckpt_gate)',
    },
    'lr_sim_scale': {
        'param': 'lr_sim_scale',
        'values': [0.0, 0.33, 0.67, 1.0, 1.5],
        'default': 0.67,
        'description': 'LR similarity scale factor (γ)',
    },
    'memory_cap': {
        'param': 'memory_cap',
        'values': [1, 3, 5, 10, 20],
        'default': 5,
        'description': 'Checkpoint memory capacity (M)',
    },
    'early_stopping': {
        'param': 'early_stopping',
        'values': ['fixed_20', 'loss_driven'],
        'default': 'loss_driven',
        'description': 'Fixed K=20 vs loss-driven early stopping',
    },
    'ckpt_sim_threshold': {
        'param': 'ckpt_sim_threshold',
        'values': [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
        'default': 0.75,
        'description': 'Checkpoint similarity threshold (τ)',
    },
}

# v2 defaults (must match copilot-instructions §4)
V2_DEFAULTS = {
    'lr_base': 3e-4,
    'max_steps': 25,
    'min_steps': 5,
    'patience': 3,
    'epsilon': 0.005,
    'ckpt_gate': 0.70,
    'lr_sim_scale': 0.67,
    'ckpt_sim_threshold': 0.75,
    'memory_cap': 5,
}

# ── Benchmark protocol (matches run_unified_benchmark.py) ───────────
BATCH_SIZE = 750
BASE_INITIAL_TRAIN_SIZE = 720
MAX_BATCHES = 10
INITIAL_EPOCHS = 15
SEQUENCE_LENGTH = 96
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_HORIZONS = [96, 192]
DEFAULT_DATASETS = ['ETTh1', 'ETTm1', 'synth_recurring', 'synth_shock_recovery']
DEFAULT_MODEL = 'gru_small'

MODEL_REGISTRY = {
    'gru_small': {'class': TimeSeriesTransformer, 'kwargs': {'hidden_dim': 64, 'num_layers': 2}},
    'itransformer': {'class': iTransformerForecaster, 'kwargs': {'hidden_dim': 64, 'num_layers': 2, 'num_heads': 2}},
    'patchtst': {'class': PatchTSTForecaster, 'kwargs': {'hidden_dim': 64, 'num_layers': 2, 'num_heads': 2, 'patch_len': 16, 'stride': 8}},
    'dlinear': {'class': DLinearForecaster, 'kwargs': {'hidden_dim': 64, 'num_layers': 1}},
}

DATASET_SEASON_LENGTH = {
    'ETTh1': 24, 'ETTh2': 24, 'ETTm1': 96, 'ETTm2': 96,
    'Weather': 144, 'Exchange': 5,
    'synth_stable': 24, 'synth_trend_break': 24, 'synth_slow_drift': 24,
    'synth_fast_switch': 24, 'synth_recurring': 24, 'synth_volatility': 24,
    'synth_shock_recovery': 24, 'synth_multi_regime': 24,
}


def get_initial_train_size(forecast_horizon: int) -> int:
    """Same logic as run_unified_benchmark.py."""
    min_needed = SEQUENCE_LENGTH + forecast_horizon + 100
    return max(BASE_INITIAL_TRAIN_SIZE, min_needed)


def _build_rgtta(model_key, dataset_name, forecast_horizon, input_dim,
                 feature_cols, params):
    """Build an RGTTAForecaster with the given params."""
    model_info = MODEL_REGISTRY[model_key]
    model_cls = model_info['class']
    model_kw = model_info['kwargs']
    season_len = DATASET_SEASON_LENGTH.get(dataset_name, 24)

    forecaster = RGTTAForecaster(
        season_length=season_len,
        forecast_horizon=forecast_horizon,
        sequence_length=SEQUENCE_LENGTH,
        hidden_dim=model_kw.get('hidden_dim', 64),
        num_layers=model_kw.get('num_layers', 2),
        num_heads=model_kw.get('num_heads', 4),
        model_class=model_cls,
        input_dim=input_dim,
        feature_cols=feature_cols,
        lr_base=params['lr_base'],
        max_steps=params['max_steps'],
        min_steps=params['min_steps'],
        patience=params['patience'],
        epsilon=params['epsilon'],
        ckpt_gate=params['ckpt_gate'],
        lr_sim_scale=params['lr_sim_scale'],
        ckpt_sim_threshold=params.get('ckpt_sim_threshold', 0.75),
        use_ewc=False,
    )
    # Override memory capacity if different from default
    if params.get('memory_cap', 5) != 5:
        forecaster._memory.max_entries = params['memory_cap']

    return forecaster


def _build_tta(model_key, dataset_name, forecast_horizon, input_dim, feature_cols):
    """Build a TTA baseline for comparison."""
    model_info = MODEL_REGISTRY[model_key]
    model_cls = model_info['class']
    model_kw = model_info['kwargs']
    season_len = DATASET_SEASON_LENGTH.get(dataset_name, 24)

    return TTAForecaster(
        season_length=season_len,
        forecast_horizon=forecast_horizon,
        sequence_length=SEQUENCE_LENGTH,
        hidden_dim=model_kw.get('hidden_dim', 64),
        num_layers=model_kw.get('num_layers', 2),
        num_heads=model_kw.get('num_heads', 4),
        model_class=model_cls,
        input_dim=input_dim,
        feature_cols=feature_cols,
        tta_steps=20,
        tta_lr=3e-4,
    )


def run_single_experiment(loader, dataset_name, horizon, seed, model_key, params,
                          run_tta_baseline=False):
    """Run one experiment. Returns dict of metrics (or None on failure)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data (same protocol as unified benchmark)
    dataset_config = loader.DATASET_CONFIGS.get(dataset_name, {})
    is_synthetic = dataset_config.get('synthetic', False)
    use_multivariate = not is_synthetic

    initial_train_size = get_initial_train_size(horizon)
    try:
        initial_df, batches = loader.prepare_incremental_batches(
            dataset_name,
            batch_size=BATCH_SIZE,
            initial_train_size=initial_train_size,
            forecast_horizon=horizon,
            multivariate=use_multivariate,
        )
    except Exception as e:
        return None

    batches = batches[:MAX_BATCHES]
    if len(batches) < 3:
        return None

    # Determine feature columns (same as unified benchmark)
    feature_cols = None
    if use_multivariate:
        raw_target = dataset_config.get('target', 'OT')
        exclude = {'unique_id', 'ds', 'y', raw_target}
        numeric_cols = initial_df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in exclude]
        if not feature_cols:
            feature_cols = None
    input_dim = len(feature_cols) + 1 if feature_cols else 1

    # Build forecasters
    rgtta = _build_rgtta(model_key, dataset_name, horizon, input_dim,
                         feature_cols, params)
    forecasters = {'rgtta': rgtta}

    if run_tta_baseline:
        tta = _build_tta(model_key, dataset_name, horizon, input_dim, feature_cols)
        forecasters['tta'] = tta

    # Initial training (same seed, same data)
    for name, fc in forecasters.items():
        try:
            fc.fit(initial_df, epochs=INITIAL_EPOCHS)
        except Exception as e:
            return None

    # Streaming evaluation
    full_history = initial_df.copy()
    batch_mses = {name: [] for name in forecasters}
    batch_times = {name: [] for name in forecasters}

    for i, batch in enumerate(batches):
        full_history = pd.concat([full_history, batch], ignore_index=True)
        if len(batch) < horizon:
            continue

        test_context = full_history.iloc[:-horizon].copy()
        ground_truth = batch['y'].values[-horizon:]

        for name, fc in forecasters.items():
            t0 = time.time()
            try:
                fc.update_with_new_data(batch)
                pred = fc.predict(test_context, steps_ahead=horizon)
                if pred is not None:
                    if hasattr(pred, 'values'):
                        pred = pred.values.flatten()
                    if len(pred) == horizon:
                        gt = ground_truth.astype(float)
                        pr = np.array(pred, dtype=float)
                        mse = float(np.mean((gt - pr) ** 2))
                        batch_mses[name].append(mse)
                batch_times[name].append(time.time() - t0)
            except Exception as e:
                batch_times[name].append(time.time() - t0)

    if not batch_mses['rgtta']:
        return None

    result = {
        'rgtta_mse': np.mean(batch_mses['rgtta']),
        'rgtta_mse_std': np.std(batch_mses['rgtta']),
        'rgtta_time': np.mean(batch_times['rgtta']),
        'n_batches': len(batch_mses['rgtta']),
    }

    if run_tta_baseline and batch_mses.get('tta'):
        result['tta_mse'] = np.mean(batch_mses['tta'])
        result['tta_time'] = np.mean(batch_times['tta'])

    return result


def run_ablation(loader, ablation_name, datasets, horizons, model_key, seeds,
                 run_tta_baseline=True):
    """Run a complete ablation sweep."""
    config = ABLATIONS[ablation_name]
    param_name = config['param']
    values = config['values']

    print(f"\n{'='*60}")
    print(f"ABLATION: {config['description']}")
    print(f"Parameter: {param_name}, Values: {values}")
    print(f"Datasets: {datasets}, Horizons: {horizons}")
    print(f"Model: {model_key}, Seeds: {seeds}")
    n_rgtta = len(values) * len(datasets) * len(horizons) * len(seeds)
    n_tta = len(datasets) * len(horizons) * len(seeds) if run_tta_baseline else 0
    n_total = n_rgtta + n_tta
    print(f"Total experiments: {n_total} ({n_rgtta} RGTTA + {n_tta} TTA)")
    print(f"{'='*60}")

    results = []
    done = 0
    t_start = time.time()

    # Run TTA baseline once per (dataset, horizon, seed)
    if run_tta_baseline:
        for ds in datasets:
            for h in horizons:
                for seed in seeds:
                    done += 1
                    print(f"  [{done}/{n_total}] TTA baseline {ds}/H={h}/s={seed}...",
                          end='', flush=True)
                    metrics = run_single_experiment(
                        loader, ds, h, seed, model_key, V2_DEFAULTS,
                        run_tta_baseline=True
                    )
                    if metrics and 'tta_mse' in metrics:
                        results.append({
                            'ablation': ablation_name,
                            'param': param_name,
                            'value': 'TTA_baseline',
                            'dataset': ds, 'horizon': h, 'seed': seed,
                            'model': model_key,
                            'mse_mean': metrics['tta_mse'],
                            'mse_std': 0.0,
                            'time_mean': metrics['tta_time'],
                            'n_batches': metrics['n_batches'],
                        })
                        print(f" MSE={metrics['tta_mse']:.4f}")
                    else:
                        print(f" SKIP")

    # Run RG-TTA with each param value
    for val in values:
        for ds in datasets:
            for h in horizons:
                for seed in seeds:
                    done += 1
                    params = V2_DEFAULTS.copy()

                    # Handle special cases
                    if ablation_name == 'early_stopping':
                        if val == 'fixed_20':
                            params['max_steps'] = 20
                            params['min_steps'] = 20
                            params['patience'] = 999  # disable early stopping
                        # else: use defaults (loss_driven)
                    else:
                        params[param_name] = val

                    print(f"  [{done}/{n_total}] {ds}/H={h}/s={seed}/{param_name}={val}...",
                          end='', flush=True)

                    try:
                        metrics = run_single_experiment(
                            loader, ds, h, seed, model_key, params,
                            run_tta_baseline=False
                        )
                        if metrics:
                            results.append({
                                'ablation': ablation_name,
                                'param': param_name,
                                'value': val,
                                'dataset': ds, 'horizon': h, 'seed': seed,
                                'model': model_key,
                                'mse_mean': metrics['rgtta_mse'],
                                'mse_std': metrics['rgtta_mse_std'],
                                'time_mean': metrics['rgtta_time'],
                                'n_batches': metrics['n_batches'],
                            })
                            print(f" MSE={metrics['rgtta_mse']:.4f} ({time.time()-t_start:.0f}s)")
                        else:
                            print(f" SKIP (no batches)")
                    except Exception as e:
                        print(f" ERROR: {e}")

    elapsed = time.time() - t_start
    print(f"\n⏱ Ablation '{ablation_name}' done in {elapsed/60:.1f} min ({len(results)} results)")
    return pd.DataFrame(results)


def summarize_ablation(df, ablation_name):
    """Print summary table and return overall averages."""
    config = ABLATIONS[ablation_name]
    print(f"\n{'─'*60}")
    print(f"SUMMARY: {config['description']}")
    print(f"{'─'*60}")

    # Separate TTA baseline
    tta_rows = df[df['value'] == 'TTA_baseline']
    rgtta_rows = df[df['value'] != 'TTA_baseline']

    # Average over seeds first, then over datasets/horizons
    overall = rgtta_rows.groupby('value').agg(
        mse_mean=('mse_mean', 'mean'),
        mse_se=('mse_mean', lambda x: x.std() / np.sqrt(max(1, len(x)))),
        time_mean=('time_mean', 'mean'),
    ).reset_index()

    default_val = config['default']
    default_mse = None
    for _, row in overall.iterrows():
        if str(row['value']) == str(default_val):
            default_mse = row['mse_mean']
    if default_mse is None:
        default_mse = overall['mse_mean'].mean()

    # TTA baseline average
    tta_mse = tta_rows['mse_mean'].mean() if len(tta_rows) > 0 else None

    print(f"\n{'Value':>15} | {'MSE':>12} | {'±SE':>10} | {'Δ vs default':>12} | {'Time (s)':>8}")
    print(f"{'-'*15}-+-{'-'*12}-+-{'-'*10}-+-{'-'*12}-+-{'-'*8}")
    if tta_mse is not None:
        delta_tta = (tta_mse - default_mse) / default_mse * 100
        print(f"{'TTA baseline':>15} | {tta_mse:>12.4f} | {'—':>10} | {delta_tta:>+11.1f}% | {'—':>8}")
        print(f"{'-'*15}-+-{'-'*12}-+-{'-'*10}-+-{'-'*12}-+-{'-'*8}")

    for _, row in overall.sort_values('value').iterrows():
        delta = (row['mse_mean'] - default_mse) / default_mse * 100
        marker = " ← default" if str(row['value']) == str(default_val) else ""
        print(f"{str(row['value']):>15} | {row['mse_mean']:>12.4f} | {row['mse_se']:>10.4f} | "
              f"{delta:>+11.1f}%{marker} | {row['time_mean']:>8.2f}")

    return overall


def main():
    parser = argparse.ArgumentParser(description='Run RGTTA ablation sweeps')
    parser.add_argument('--ablation', type=str, default='all',
                       choices=list(ABLATIONS.keys()) + ['all'],
                       help='Which ablation to run (default: all)')
    parser.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS)
    parser.add_argument('--horizons', nargs='+', type=int, default=DEFAULT_HORIZONS)
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                       choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS)
    parser.add_argument('--results-dir', type=str,
                       default=str(ROOT / 'benchmarks' / 'results' / 'ablation'))
    parser.add_argument('--no-tta-baseline', action='store_true',
                       help='Skip TTA baseline (faster, but no comparison)')
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # Initialize data loader
    loader = StandardBenchmarkLoader()

    ablations_to_run = list(ABLATIONS.keys()) if args.ablation == 'all' else [args.ablation]

    all_summaries = {}
    total_start = time.time()

    for abl_name in ablations_to_run:
        print(f"\n{'#'*60}")
        print(f"# Running ablation: {abl_name}")
        print(f"{'#'*60}")

        df = run_ablation(
            loader, abl_name, args.datasets, args.horizons,
            args.model, args.seeds,
            run_tta_baseline=not args.no_tta_baseline,
        )

        if len(df) > 0:
            # Save raw results
            out_path = Path(args.results_dir) / f"ablation_{abl_name}.csv"
            df.to_csv(out_path, index=False)
            print(f"\n✅ Raw results saved to {out_path}")

            # Summary
            overall = summarize_ablation(df, abl_name)
            summary_path = Path(args.results_dir) / f"ablation_{abl_name}_summary.csv"
            overall.to_csv(summary_path, index=False)
            print(f"✅ Summary saved to {summary_path}")
            all_summaries[abl_name] = overall
        else:
            print(f"⚠️ No results for ablation {abl_name}")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"✅ All ablations complete in {total_elapsed/60:.1f} min")
    print(f"Results in: {args.results_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
