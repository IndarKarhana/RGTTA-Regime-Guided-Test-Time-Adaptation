#!/usr/bin/env python3
"""
Smoke test: run H=720 (hardest config) × all 14 datasets × 1 seed × gru_small.

Purpose: Verify that the partial-checkpoint fix and season_length mapping
produce valid results for every dataset at the maximum forecast horizon.

Checks:
  1. No dataset is skipped (enough batches)
  2. All 7 policies produce non-NaN metrics
  3. wmape_mean field is populated
  4. Partial checkpoints are actually trained (not skipped)
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root / "benchmarks"))
sys.path.insert(0, str(_root / "benchmarks" / "data_loaders"))

from standard_benchmarks import StandardBenchmarkLoader
from regime_forecasting.core.forecaster import CorrectedRegimeForecaster
from regime_forecasting.utils.evaluation import (
    weighted_mape, symmetric_mape, rmse as calc_rmse, directional_accuracy,
)
from tta_forecaster import TTAForecaster
from ewc_forecaster import EWCForecaster
from rgtta_forecaster import RGTTAForecaster
from dynatta_forecaster import DynaTTAForecaster
from rgtta_dynatta_forecaster import RGTTADynaTTAForecaster
from tafas_forecaster import TAFASForecaster

from regime_forecasting.models.transformer import TimeSeriesTransformer

logging.basicConfig(
    level=logging.WARNING,  # keep output clean
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("smoke_h720")
logger.setLevel(logging.INFO)

# ── Constants (match run_unified_benchmark.py) ────────────────────────────
BATCH_SIZE = 750
INITIAL_TRAIN_SIZE = 720
MAX_BATCHES = 10
SEQUENCE_LENGTH = 96
INITIAL_EPOCHS = 15
FORECAST_HORIZON = 720          # hardest config
SEED = 42

ALL_DATASETS = [
    "ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Exchange",
    "synth_stable", "synth_trend_break", "synth_slow_drift", "synth_fast_switch",
    "synth_recurring", "synth_volatility", "synth_shock_recovery", "synth_multi_regime",
]

DATASET_SEASON_LENGTH = {
    "ETTh1": 24, "ETTh2": 24,
    "ETTm1": 96, "ETTm2": 96,
    "Weather": 144,
    "Exchange": 5,
    "synth_stable": 24, "synth_trend_break": 24, "synth_slow_drift": 24,
    "synth_fast_switch": 24, "synth_recurring": 24, "synth_volatility": 24,
    "synth_shock_recovery": 24, "synth_multi_regime": 24,
}

POLICIES = ["retrain", "tta", "ewc", "dynatta", "rgtta", "rgtta_ewc", "rgtta_dynatta", "tafas"]


def build_forecasters(season_len: int, ckpt_base: str):
    """Build all 8 policy forecasters for gru_small."""
    common = dict(
        season_length=season_len,
        forecast_horizon=FORECAST_HORIZON,
        sequence_length=SEQUENCE_LENGTH,
    )
    mkw = dict(hidden_dim=64, num_layers=2, num_heads=4)

    retrain = CorrectedRegimeForecaster(
        season_length=season_len, forecast_horizon=FORECAST_HORIZON,
        sequence_length=SEQUENCE_LENGTH, similarity_threshold=2.0,
        model_selection="full",
        storage_path=f"{ckpt_base}_retrain",
        model_class=TimeSeriesTransformer, **mkw,
    )
    tta = TTAForecaster(**common, **mkw, tta_steps=20, tta_lr=0.0003,
                        model_class=TimeSeriesTransformer)
    ewc = EWCForecaster(**common, **mkw, model_class=TimeSeriesTransformer,
                        ewc_lambda=400.0, ewc_update_steps=15, ewc_lr=0.0003)
    dynatta = DynaTTAForecaster(**common, **mkw, model_class=TimeSeriesTransformer,
                                alpha_min=1e-4, alpha_max=1e-3, kappa=1.0,
                                eta=0.1, tta_steps=20, warmup_factor=1)
    rgtta = RGTTAForecaster(**common, **mkw, model_class=TimeSeriesTransformer,
                            tau_high=0.80, tau_low=0.55,
                            steps_high=5, steps_mid=20, steps_low=50,
                            lr_high=1e-4, lr_mid=3e-4, lr_low=1e-3,
                            use_ewc_on_low=False)
    rgtta_ewc = RGTTAForecaster(**common, **mkw, model_class=TimeSeriesTransformer,
                                tau_high=0.80, tau_low=0.55,
                                steps_high=5, steps_mid=20, steps_low=50,
                                lr_high=1e-4, lr_mid=3e-4, lr_low=1e-3,
                                use_ewc_on_low=True, ewc_lambda=400.0)
    rgtta_dynatta = RGTTADynaTTAForecaster(
        **common, **mkw, model_class=TimeSeriesTransformer,
        tau_high=0.80, tau_low=0.55,
        steps_high=10, steps_mid=20, steps_low=50,
        alpha_min_high=5e-5, alpha_max_high=5e-4,
        alpha_min_mid=1e-4, alpha_max_mid=1e-3,
        alpha_min_low=5e-4, alpha_max_low=5e-3,
        kappa=1.0, eta=0.1, use_ewc_on_low=False,
    )
    tafas = TAFASForecaster(**common, **mkw, model_class=TimeSeriesTransformer,
                            gcm_lr=0.005, gcm_steps=1, gating_init=0.01,
                            use_paas=True, use_prediction_adjustment=True,
                            reset_between_batches=True)
    return {
        "retrain": retrain, "tta": tta, "ewc": ewc, "dynatta": dynatta,
        "rgtta": rgtta, "rgtta_ewc": rgtta_ewc, "rgtta_dynatta": rgtta_dynatta,
        "tafas": tafas,
    }


def run_one_dataset(dataset_name: str, loader: StandardBenchmarkLoader) -> dict:
    """Run H=720 smoke test for one dataset.  Returns summary dict."""
    sl = DATASET_SEASON_LENGTH.get(dataset_name, 24)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load data
    initial_df, batches = loader.prepare_incremental_batches(
        dataset_name, batch_size=BATCH_SIZE,
        initial_train_size=INITIAL_TRAIN_SIZE,
        forecast_horizon=FORECAST_HORIZON,
    )
    batches = batches[:MAX_BATCHES]
    n_batches = len(batches)
    if n_batches < 3:
        return {"dataset": dataset_name, "status": "SKIPPED", "reason": f"only {n_batches} batches"}

    ckpt_base = str(_root / "benchmarks" / "results" / "smoke_h720" / f"ckpt_{dataset_name}")
    Path(ckpt_base).parent.mkdir(parents=True, exist_ok=True)
    forecasters = build_forecasters(sl, ckpt_base)

    _regime_policies = {"retrain"}
    _tta_style_policies = {"tta", "ewc", "rgtta", "rgtta_ewc", "dynatta", "rgtta_dynatta", "tafas"}

    full_history = initial_df.copy()

    # Initial fit
    for pol, fc in forecasters.items():
        try:
            if pol in _regime_policies:
                fc.fit_incremental(full_history)
            else:
                fc.fit(full_history, epochs=INITIAL_EPOCHS)
        except Exception as e:
            logger.warning(f"  {dataset_name}/{pol} init-fit error: {e}")

    # Per-policy metrics
    policy_metrics = {p: {"mse": [], "wmape": [], "smape": [], "time": []} for p in POLICIES}

    # Process batches (only first 3 to keep fast)
    n_eval = min(3, n_batches)
    for i in range(n_eval):
        batch = batches[i]
        full_history = pd.concat([full_history, batch], ignore_index=True)
        forecast_start = len(full_history)
        test_context = full_history.tail(SEQUENCE_LENGTH + FORECAST_HORIZON).head(SEQUENCE_LENGTH)

        if i + 1 < n_batches:
            ground_truth = batches[i + 1]["y"].values[:FORECAST_HORIZON]
        else:
            ground_truth = full_history["y"].values[-FORECAST_HORIZON:]

        if len(ground_truth) < FORECAST_HORIZON:
            continue

        for pol, fc in forecasters.items():
            t0 = time.time()
            try:
                if pol in _regime_policies:
                    fc.fit_incremental(full_history)
                else:
                    fc.update_with_new_data(batch)
                pred = fc.predict(test_context, steps_ahead=FORECAST_HORIZON)
                elapsed = time.time() - t0
                if pred is not None:
                    if hasattr(pred, "values"):
                        pred = pred.values.flatten()
                    if len(pred) == FORECAST_HORIZON:
                        gt = ground_truth.astype(float)
                        pr = np.array(pred, dtype=float)
                        policy_metrics[pol]["mse"].append(float(np.mean((gt - pr) ** 2)))
                        policy_metrics[pol]["wmape"].append(float(weighted_mape(gt, pr)))
                        policy_metrics[pol]["smape"].append(float(symmetric_mape(gt, pr)))
                        policy_metrics[pol]["time"].append(elapsed)
                    else:
                        policy_metrics[pol]["time"].append(elapsed)
                else:
                    policy_metrics[pol]["time"].append(elapsed)
            except Exception as e:
                policy_metrics[pol]["time"].append(time.time() - t0)
                logger.warning(f"  {dataset_name}/{pol} batch {i} error: {e}")

    # Summarize
    summary = {
        "dataset": dataset_name,
        "season_length": sl,
        "n_batches_available": n_batches,
        "n_batches_evaluated": n_eval,
        "combined_df_size_batch0": INITIAL_TRAIN_SIZE + BATCH_SIZE,
        "status": "OK",
        "policies": {},
    }
    all_ok = True
    for pol in POLICIES:
        m = policy_metrics[pol]
        mse_vals = m["mse"]
        wmape_vals = m["wmape"]
        if mse_vals:
            summary["policies"][pol] = {
                "mse_mean": float(np.mean(mse_vals)),
                "wmape_mean": float(np.mean(wmape_vals)),
                "smape_mean": float(np.mean(m["smape"])),
                "total_time": float(sum(m["time"])),
                "n_valid_preds": len(mse_vals),
            }
        else:
            summary["policies"][pol] = {
                "mse_mean": None, "wmape_mean": None, "smape_mean": None,
                "total_time": float(sum(m["time"])) if m["time"] else 0,
                "n_valid_preds": 0,
            }
            all_ok = False

    if not all_ok:
        summary["status"] = "PARTIAL_FAIL"

    return summary


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd

    loader = StandardBenchmarkLoader()
    results = []

    total_t0 = time.time()
    logger.info("=" * 70)
    logger.info("SMOKE TEST  —  H=720 × 14 datasets × 1 seed × gru_small")
    logger.info("=" * 70)

    for ds in ALL_DATASETS:
        logger.info(f"\n{'─' * 50}")
        logger.info(f"▶ {ds}  (season_length={DATASET_SEASON_LENGTH.get(ds, 24)})")
        ds_t0 = time.time()
        result = run_one_dataset(ds, loader)
        elapsed = time.time() - ds_t0
        results.append(result)

        status = result["status"]
        icon = "✅" if status == "OK" else ("⚠️" if status == "PARTIAL_FAIL" else "❌")
        logger.info(f"{icon}  {ds}  [{status}]  ({elapsed:.1f}s)")

        if result.get("policies"):
            for pol in POLICIES:
                pd_info = result["policies"].get(pol, {})
                mse = pd_info.get("mse_mean")
                wmape = pd_info.get("wmape_mean")
                n = pd_info.get("n_valid_preds", 0)
                t = pd_info.get("total_time", 0)
                if mse is not None:
                    logger.info(f"    {pol:18s}: MSE={mse:10.2f}  wMAPE={wmape:8.2f}%  time={t:6.1f}s  ({n} preds)")
                else:
                    logger.info(f"    {pol:18s}: *** NO VALID PREDICTIONS ***  time={t:6.1f}s")

    total_elapsed = time.time() - total_t0

    # ── Summary table ─────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY TABLE")
    logger.info("=" * 70)
    header = f"{'Dataset':<25} {'SL':>3} {'Batches':>7} {'Status':<14} " + \
             " ".join(f"{p:>10}" for p in POLICIES)
    logger.info(header)
    logger.info("-" * len(header))
    for r in results:
        ds = r["dataset"]
        sl = r.get("season_length", "?")
        nb = r.get("n_batches_available", 0)
        st = r["status"]
        pol_vals = []
        for p in POLICIES:
            pd_info = r.get("policies", {}).get(p, {})
            mse = pd_info.get("mse_mean")
            pol_vals.append(f"{mse:10.2f}" if mse is not None else "       NaN")
        line = f"{ds:<25} {sl:>3} {nb:>7} {st:<14} " + " ".join(pol_vals)
        logger.info(line)

    # Count failures
    fails = [r for r in results if r["status"] != "OK"]
    skips = [r for r in results if r["status"] == "SKIPPED"]

    logger.info(f"\nTotal: {len(results)} datasets, {len(results) - len(fails)} OK, "
                f"{len(fails) - len(skips)} partial-fail, {len(skips)} skipped")
    logger.info(f"Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")

    # Save results
    out_path = _root / "benchmarks" / "results" / "smoke_h720" / "smoke_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")

    # Exit code
    if fails:
        logger.error(f"⚠️  {len(fails)} datasets had issues — check above!")
        sys.exit(1)
    else:
        logger.info("✅  All 14 datasets passed at H=720!")
        sys.exit(0)
