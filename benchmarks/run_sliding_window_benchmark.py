"""
Standard Sliding-Window Benchmark Runner  (Level 2)
=====================================================

Evaluates the same 8 update policies using the **standard protocol** from
TAFAS (AAAI 2025) and DynaTTA (ICML 2025):

    1.  Train on the first 60 % of each dataset.
    2.  Slide a (lookback=L, horizon=H) window across the remaining 40 %
        test split, producing thousands of evaluation windows.
    3.  At each window the model may adapt (TTA / EWC / regime-guided) and
        then forecast.  Metric is the mean across all windows.

Key differences from the streaming benchmark (run_unified_benchmark.py):
    - Many more evaluation windows (hundreds–thousands vs 10 batches).
    - No "data update batch" concept — each window is a fresh adapt+predict.
    - Train split is fixed (no accumulating history).

Usage:
    python benchmarks/run_sliding_window_benchmark.py                 # full
    python benchmarks/run_sliding_window_benchmark.py --quick         # smoke
    python benchmarks/run_sliding_window_benchmark.py --datasets ETTh1 Weather --horizons 96 336
"""

import argparse
import copy
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_benchmark_dir = Path(__file__).parent
sys.path.insert(0, str(_benchmark_dir.parent / "src"))
sys.path.insert(0, str(_benchmark_dir))
sys.path.insert(0, str(_benchmark_dir / "data_loaders"))

from standard_benchmarks import StandardBenchmarkLoader
from regime_forecasting.core.forecaster import CorrectedRegimeForecaster
from regime_forecasting.utils.evaluation import (
    weighted_mape, symmetric_mape, rmse as calc_rmse,
    directional_accuracy,
)
from tta_forecaster import TTAForecaster
from ewc_forecaster import EWCForecaster
from rgtta_forecaster import RGTTAForecaster
from dynatta_forecaster import DynaTTAForecaster
from rgtta_dynatta_forecaster import RGTTADynaTTAForecaster
from tafas_forecaster import TAFASForecaster

from regime_forecasting.models.transformer import TimeSeriesTransformer
from regime_forecasting.models.large_gru_model import LargeGRUForecaster
from regime_forecasting.models.dlinear_model import DLinearForecaster
from regime_forecasting.models.itransformer_model import iTransformerForecaster
from regime_forecasting.models.patchtst_model import PatchTSTForecaster

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry  (identical to run_unified_benchmark.py)
# ---------------------------------------------------------------------------
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "gru_small": {
        "class": TimeSeriesTransformer,
        "kwargs": {"hidden_dim": 64, "num_layers": 2},
        "label": "GRU-Small",
    },
    "itransformer": {
        "class": iTransformerForecaster,
        "kwargs": {"hidden_dim": 64, "num_layers": 2, "num_heads": 2},
        "label": "iTransformer",
    },
    "gru_large": {
        "class": LargeGRUForecaster,
        "kwargs": {"hidden_dim": 128, "num_layers": 3},
        "label": "GRU-Large",
    },
    "patchtst": {
        "class": PatchTSTForecaster,
        "kwargs": {"hidden_dim": 64, "num_layers": 2, "num_heads": 2, "patch_len": 16, "stride": 8},
        "label": "PatchTST",
    },
    "dlinear": {
        "class": DLinearForecaster,
        "kwargs": {"hidden_dim": 64, "num_layers": 1},
        "label": "DLinear",
    },
}

POLICY_LABELS = {
    "retrain": "Source",       # No adaptation — standard baseline in TAFAS/DynaTTA
    "tta": "TTA",
    "ewc": "EWC",
    "dynatta": "DynaTTA",
    "rgtta": "RGTTA",
    "rgtta_ewc": "RGTTA+EWC",
    "rgtta_dynatta": "RGTTA+DynaTTA",
    "tafas": "TAFAS",
}

ALL_POLICIES = list(POLICY_LABELS.keys())

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_HORIZONS = [96, 192, 336, 720]
# Only real-world datasets — sliding-window protocol doesn't apply to
# our short synthetic series (8,000 rows each, already covered by streaming).
SLIDING_DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Exchange"]
TRAIN_RATIO = 0.6           # 60 % train, 40 % test  (TAFAS & DynaTTA use ~70/15/15 or 60/20/20)
SEQUENCE_LENGTH = 96         # Lookback window L
WINDOW_STRIDE = 1            # Stride across test set; 1 = every position
MAX_WINDOWS = 500            # Cap windows per experiment to keep runtime feasible
INITIAL_EPOCHS = 15          # Training epochs on train split
# TAFAS paper uses STEPS=1 per window (see kimanki/TAFAS config.py line 83).
# Using 1 step matches standard protocol — heavier adaptation is for streaming.
ADAPT_STEPS = 1              # TTA/EWC adapt steps per window (TAFAS default=1)

# Correct season length per dataset — must match run_unified_benchmark.py
DATASET_SEASON_LENGTH: Dict[str, int] = {
    "ETTh1": 24, "ETTh2": 24,          # Hourly → 24 = one daily cycle
    "ETTm1": 96, "ETTm2": 96,          # 15-min → 96 = one daily cycle (4 × 24)
    "Weather": 144,                      # 10-min → 144 = one daily cycle (6 × 24)
    "Exchange": 5,                       # Daily → 5 = one business week
    "synth_stable": 24, "synth_trend_break": 24, "synth_slow_drift": 24,
    "synth_fast_switch": 24, "synth_recurring": 24, "synth_volatility": 24,
    "synth_shock_recovery": 24, "synth_multi_regime": 24,
}


# ============================================================================
# Sliding-window benchmark class
# ============================================================================
class SlidingWindowBenchmark:
    """Sliding-window evaluation: train once, slide through test split."""

    def __init__(
        self,
        results_dir: Optional[str] = None,
        n_seeds: int = 3,
        forecast_horizons: Optional[List[int]] = None,
        model_keys: Optional[List[str]] = None,
        max_windows: int = MAX_WINDOWS,
        stride: int = WINDOW_STRIDE,
        policies: Optional[List[str]] = None,
    ):
        if results_dir is None:
            results_dir = str(_benchmark_dir / "results" / "sliding_window")
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.n_seeds = n_seeds
        self.forecast_horizons = forecast_horizons or DEFAULT_HORIZONS
        self.model_keys = model_keys or list(MODEL_REGISTRY.keys())
        self.max_windows = max_windows
        self.stride = stride
        self.policies = policies or ALL_POLICIES  # Filter which policies to run
        self.loader = StandardBenchmarkLoader()

        self.results: Dict[str, Any] = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "protocol": "sliding_window",
                "n_seeds": n_seeds,
                "forecast_horizons": self.forecast_horizons,
                "models": self.model_keys,
                "train_ratio": TRAIN_RATIO,
                "sequence_length": SEQUENCE_LENGTH,
                "max_windows": max_windows,
                "stride": stride,
            },
            "experiments": [],
        }

    # -----------------------------------------------------------------------
    def _build_forecasters(
        self,
        model_key: str,
        dataset_name: str,
        forecast_horizon: int,
        seed: int,
        input_dim: int = 1,
        feature_cols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Build all 8 policy forecasters."""
        model_info = MODEL_REGISTRY[model_key]
        model_cls = model_info["class"]
        model_kw = model_info["kwargs"]
        season_len = DATASET_SEASON_LENGTH.get(dataset_name, 24)

        common = dict(
            season_length=season_len,
            forecast_horizon=forecast_horizon,
            sequence_length=SEQUENCE_LENGTH,
        )
        model_common = dict(
            hidden_dim=model_kw.get("hidden_dim", 64),
            num_layers=model_kw.get("num_layers", 2),
            num_heads=model_kw.get("num_heads", 4),
        )
        mv_common = dict(
            input_dim=input_dim,
            feature_cols=feature_cols,
        )

        ckpt_dir = str(self.results_dir / f"ckpt_{model_key}_{dataset_name}_{seed}")

        retrain = CorrectedRegimeForecaster(
            season_length=season_len,
            forecast_horizon=forecast_horizon,
            sequence_length=SEQUENCE_LENGTH,
            similarity_threshold=2.0,
            model_selection="full",
            storage_path=ckpt_dir,
            model_class=model_cls,
            input_dim=input_dim,
            feature_cols=feature_cols,
            **model_common,
        )

        tta = TTAForecaster(
            **common, **model_common, **mv_common,
            tta_steps=ADAPT_STEPS, tta_lr=0.0003,
            model_class=model_cls,
        )

        ewc = EWCForecaster(
            **common, **model_common, **mv_common,
            model_class=model_cls,
            ewc_lambda=400.0, ewc_update_steps=ADAPT_STEPS, ewc_lr=0.0003,
        )

        dynatta = DynaTTAForecaster(
            **common, **model_common, **mv_common,
            model_class=model_cls,
            alpha_min=1e-4, alpha_max=1e-3,
            kappa=1.0, eta=0.1,
            tta_steps=ADAPT_STEPS, warmup_factor=1,
        )

        rgtta = RGTTAForecaster(
            **common, **model_common, **mv_common,
            model_class=model_cls,
            tau_high=0.85, tau_low=0.55,
            # Sliding window: 1 step per window (matching TAFAS protocol).
            # Streaming benchmark uses 5/15/30 steps for larger batches.
            steps_high=1, steps_mid=1, steps_low=1,
            lr_high=1e-4, lr_mid=3e-4, lr_low=1e-3,
            use_ewc_on_low=False,
        )

        rgtta_ewc = RGTTAForecaster(
            **common, **model_common, **mv_common,
            model_class=model_cls,
            tau_high=0.85, tau_low=0.55,
            steps_high=1, steps_mid=1, steps_low=1,
            lr_high=1e-4, lr_mid=3e-4, lr_low=1e-3,
            use_ewc_on_low=True, ewc_lambda=400.0,
        )

        rgtta_dynatta = RGTTADynaTTAForecaster(
            **common, **model_common, **mv_common,
            model_class=model_cls,
            tau_high=0.85, tau_low=0.55,
            steps_high=1, steps_mid=1, steps_low=1,
            alpha_min_high=5e-5, alpha_max_high=5e-4,
            alpha_min_mid=1e-4, alpha_max_mid=1e-3,
            alpha_min_low=5e-4, alpha_max_low=5e-3,
            kappa=1.0, eta=0.1, use_ewc_on_low=False,
        )

        tafas = TAFASForecaster(
            **common, **model_common, **mv_common,
            model_class=model_cls,
            gcm_lr=0.005, gcm_steps=1,
            gating_init=0.01, use_paas=True,
            use_prediction_adjustment=True,
            reset_between_batches=True,
        )

        all_forecasters = {
            "retrain": retrain,
            "tta": tta,
            "ewc": ewc,
            "dynatta": dynatta,
            "rgtta": rgtta,
            "rgtta_ewc": rgtta_ewc,
            "rgtta_dynatta": rgtta_dynatta,
            "tafas": tafas,
        }
        # Filter by selected policies
        return {k: v for k, v in all_forecasters.items() if k in self.policies}

    # -----------------------------------------------------------------------
    @staticmethod
    def _safe_stat(vals: List[float]) -> Tuple[float, float]:
        if not vals:
            return float("nan"), float("nan")
        return float(np.mean(vals)), float(np.std(vals))

    # -----------------------------------------------------------------------
    def run_single(
        self,
        model_key: str,
        dataset_name: str,
        forecast_horizon: int,
        seed: int,
    ) -> Optional[Dict[str, Any]]:
        """Run one (model, dataset, horizon, seed) sliding-window experiment."""

        np.random.seed(seed)
        torch.manual_seed(seed)

        # Load full dataset (multivariate for real-world, univariate for synthetic)
        dataset_config = self.loader.DATASET_CONFIGS.get(dataset_name, {})
        is_synthetic = dataset_config.get("synthetic", False)
        use_multivariate = not is_synthetic

        try:
            full_df = self.loader.load_dataset(dataset_name, multivariate=use_multivariate)
        except Exception as e:
            logger.error(f"Data load error {dataset_name}: {e}")
            return None

        # Determine multivariate feature columns
        feature_cols = None
        if use_multivariate:
            raw_target = dataset_config.get("target", "OT")
            exclude = {"unique_id", "ds", "y", raw_target}
            numeric_cols = full_df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in numeric_cols if c not in exclude]
            if not feature_cols:
                feature_cols = None
        input_dim = len(feature_cols) + 1 if feature_cols else 1

        n_total = len(full_df)
        n_train = int(n_total * TRAIN_RATIO)
        train_df = full_df.iloc[:n_train].copy()
        test_df = full_df.iloc[n_train:].copy()

        # How many sliding windows fit?
        L = SEQUENCE_LENGTH
        H = forecast_horizon
        n_test = len(test_df)
        n_windows = (n_test - L - H) // self.stride + 1

        if n_windows < 3:
            logger.warning(
                f"Skipping {dataset_name} h={H}: only {n_windows} windows "
                f"(test={n_test}, need L+H={L + H})"
            )
            return None

        # Cap windows for runtime
        if n_windows > self.max_windows:
            # Sample evenly spaced window start indices
            indices = np.linspace(0, n_windows - 1, self.max_windows, dtype=int)
        else:
            indices = np.arange(n_windows)

        actual_windows = len(indices)
        logger.info(
            f"  {dataset_name} h={H}: train={n_train}, test={n_test}, "
            f"total_windows={n_windows}, evaluating={actual_windows}"
        )

        # Build forecasters
        forecasters = self._build_forecasters(
            model_key, dataset_name, forecast_horizon, seed,
            input_dim=input_dim, feature_cols=feature_cols,
        )

        # --- Initial fit on train split ---
        _regime_policies = {"retrain"}
        _tta_style_policies = {"tta", "ewc", "rgtta", "rgtta_ewc", "dynatta", "rgtta_dynatta"}

        for policy, fc in forecasters.items():
            try:
                if policy in _regime_policies:
                    fc.fit_incremental(train_df)
                else:
                    fc.fit(train_df, epochs=INITIAL_EPOCHS)
            except Exception as e:
                logger.debug(f"Init-fit error {policy}: {e}")

        # No per-window reset: TAFAS / DynaTTA papers adapt cumulatively
        # across the test set (each 1-step update builds on previous).
        # Snapshots are NOT taken; model state evolves through the test split.

        # Metrics per policy
        metrics = {
            policy: {
                "mse": [], "mae": [], "rmse": [], "mape": [],
                "smape": [], "direction_acc": [], "time": [],
            }
            for policy in forecasters
        }

        test_values = test_df.reset_index(drop=True)

        # --- Slide through test split ---
        # Suppress noisy per-sequence logging during the window loop.
        # Restores original level after the loop.
        _inner_loggers = [
            logging.getLogger("regime_forecasting.core.forecaster"),
            logging.getLogger("regime_forecasting.utils.data_utils"),
        ]
        _saved_levels = [lg.level for lg in _inner_loggers]
        for lg in _inner_loggers:
            lg.setLevel(logging.WARNING)

        for win_idx, start_offset in enumerate(indices):
            start = start_offset * self.stride
            context_end = start + L
            target_end = context_end + H

            if target_end > n_test:
                break

            # Build context dataframe (L rows before the forecast)
            context_df = test_values.iloc[start:context_end].copy()
            ground_truth = test_values.iloc[context_end:target_end]["y"].values.astype(float)

            if len(ground_truth) < H:
                continue

            # Build a "batch" for the adapt step (the context window itself)
            adapt_batch = context_df.copy()

            for policy, fc in list(forecasters.items()):
                t0 = time.time()
                try:
                    # NO PER-WINDOW RESET — matches TAFAS / DynaTTA protocol.
                    # Both papers adapt cumulatively: each 1-step update builds
                    # on the previous one across the entire test set.
                    # See kimanki/TAFAS tta/tafas.py adapt_tafas() and
                    # shivam-grover/DynaTTA DynaTTA/DynaTTA.py adapt_tafas().

                    # Adapt on the context window (1 step, cumulative)
                    # "retrain" in sliding-window acts as the "Source" (no-adapt)
                    # baseline — standard in TAFAS/DynaTTA papers.
                    if policy not in _regime_policies:
                        fc.update_with_new_data(adapt_batch)

                    # Predict
                    pred = fc.predict(context_df, steps_ahead=H)
                    if pred is not None:
                        if hasattr(pred, "values"):
                            pred = pred.values.flatten()
                        if len(pred) == H:
                            gt = ground_truth
                            pr = np.array(pred, dtype=float)
                            metrics[policy]["mse"].append(float(np.mean((gt - pr) ** 2)))
                            metrics[policy]["mae"].append(float(np.mean(np.abs(gt - pr))))
                            metrics[policy]["rmse"].append(float(calc_rmse(gt, pr)))
                            metrics[policy]["mape"].append(float(weighted_mape(gt, pr)))
                            metrics[policy]["smape"].append(float(symmetric_mape(gt, pr)))
                            da = float(directional_accuracy(gt, pr))
                            if not np.isnan(da):
                                metrics[policy]["direction_acc"].append(da)
                except Exception as e:
                    logger.debug(f"Win {win_idx} {policy} error: {e}")
                metrics[policy]["time"].append(time.time() - t0)

            # Progress every 50 windows
            if (win_idx + 1) % 50 == 0:
                logger.info(f"    … window {win_idx + 1}/{actual_windows}")

        # Restore inner-loop logger levels
        for lg, lvl in zip(_inner_loggers, _saved_levels):
            lg.setLevel(lvl)

        # --- Assemble result ---
        result: Dict[str, Any] = {
            "model": model_key,
            "model_label": MODEL_REGISTRY[model_key]["label"],
            "dataset": dataset_name,
            "forecast_horizon": forecast_horizon,
            "seed": seed,
            "n_windows_total": int(n_windows),
            "n_windows_evaluated": int(actual_windows),
            "n_train": n_train,
            "n_test": n_test,
        }

        for policy in forecasters:
            m = metrics[policy]
            mu_mse, sd_mse = self._safe_stat(m["mse"])
            mu_mae, sd_mae = self._safe_stat(m["mae"])
            mu_rmse, sd_rmse = self._safe_stat(m["rmse"])
            mu_mape, sd_mape = self._safe_stat(m["mape"])
            mu_smape, sd_smape = self._safe_stat(m["smape"])
            mu_da, sd_da = self._safe_stat(m["direction_acc"])
            result[policy] = {
                "mse_mean": mu_mse, "mse_std": sd_mse,
                "mae_mean": mu_mae, "mae_std": sd_mae,
                "rmse_mean": mu_rmse, "rmse_std": sd_rmse,
                "mape_mean": mu_mape, "mape_std": sd_mape,
                "smape_mean": mu_smape, "smape_std": sd_smape,
                "direction_acc_mean": mu_da, "direction_acc_std": sd_da,
                "total_time": sum(m["time"]),
                "n_windows_evaluated": len(m["mse"]),
            }
        return result

    # -----------------------------------------------------------------------
    def run_full(self, datasets: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run the entire sliding-window benchmark matrix."""
        datasets = datasets or SLIDING_DATASETS

        jobs = []
        for model_key in self.model_keys:
            for ds in datasets:
                for h in self.forecast_horizons:
                    for seed in range(self.n_seeds):
                        jobs.append((model_key, ds, h, seed))

        total = len(jobs)
        t_start = time.time()

        logger.info("=" * 70)
        logger.info("  SLIDING-WINDOW BENCHMARK  (Level 2 — standard protocol)")
        logger.info("=" * 70)
        logger.info(f"  Models:     {self.model_keys}")
        logger.info(f"  Datasets:   {datasets}")
        logger.info(f"  Horizons:   {self.forecast_horizons}")
        logger.info(f"  Seeds:      {self.n_seeds}")
        logger.info(f"  Lookback L: {SEQUENCE_LENGTH}")
        logger.info(f"  Max windows: {self.max_windows}")
        logger.info(f"  Total experiments: {total}")
        logger.info("=" * 70)

        for done, (model_key, ds, h, seed) in enumerate(jobs, 1):
            tag = f"[{done}/{total}] {MODEL_REGISTRY[model_key]['label']} | {ds} | h={h} | s={seed}"
            logger.info(f"▶ {tag}")
            try:
                res = self.run_single(model_key, ds, h, seed)
                if res is not None:
                    self.results["experiments"].append(res)
            except Exception as e:
                logger.error(f"  FAILED: {e}")
            if done % 12 == 0:
                self._save_results()

        elapsed = time.time() - t_start
        self.results["metadata"]["total_time_seconds"] = elapsed
        self._save_results()
        self._generate_report()
        self._export_csv()

        logger.info("=" * 70)
        logger.info(f"✅  SLIDING-WINDOW BENCHMARK COMPLETE  ({elapsed / 60:.1f} min)")
        logger.info(f"    Results: {self.results_dir / 'sliding_window_results.json'}")
        logger.info(f"    Report:  {self.results_dir / 'sliding_window_report.md'}")
        logger.info(f"    CSV:     {self.results_dir / 'sliding_window_results.csv'}")
        logger.info("=" * 70)
        return self.results

    # -----------------------------------------------------------------------
    def _save_results(self):
        def _convert(obj):
            if isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        path = self.results_dir / "sliding_window_results.json"
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2, default=_convert)

    # -----------------------------------------------------------------------
    def _export_csv(self):
        """Export flat CSV of all results."""
        rows = []
        for exp in self.results["experiments"]:
            for policy in POLICY_LABELS:
                if policy not in exp:
                    continue
                p = exp[policy]
                rows.append({
                    "model": exp["model"],
                    "dataset": exp["dataset"],
                    "horizon": exp["forecast_horizon"],
                    "seed": exp["seed"],
                    "policy": policy,
                    "mse": p.get("mse_mean", float("nan")),
                    "mae": p.get("mae_mean", float("nan")),
                    "rmse": p.get("rmse_mean", float("nan")),
                    "mape": p.get("mape_mean", float("nan")),
                    "smape": p.get("smape_mean", float("nan")),
                    "direction_acc": p.get("direction_acc_mean", float("nan")),
                    "time_s": p.get("total_time", 0),
                    "n_windows": p.get("n_windows_evaluated", 0),
                })
        if rows:
            df = pd.DataFrame(rows)
            path = self.results_dir / "sliding_window_results.csv"
            df.to_csv(path, index=False)
            logger.info(f"📊 CSV → {path}")

    # -----------------------------------------------------------------------
    def _generate_report(self):
        """Generate a Markdown summary report."""
        experiments = self.results["experiments"]
        if not experiments:
            return

        report = ["# Sliding-Window Benchmark Report", ""]
        report.append(f"**Generated:** {datetime.now().isoformat()}")
        report.append(f"**Protocol:** Train on {TRAIN_RATIO*100:.0f}%, "
                      f"slide (L={SEQUENCE_LENGTH}, H) windows through test split.")
        report.append(f"**Max windows per experiment:** {self.max_windows}")
        report.append("")

        df = pd.DataFrame(experiments)

        # Overall comparison: MSE mean across all datasets & horizons
        report.extend(["## Overall MSE Comparison (lower is better)", ""])
        policies = list(POLICY_LABELS.keys())
        available = [p for p in policies if any(p in exp for exp in experiments)]

        hdr = ["Model"] + [POLICY_LABELS[p] for p in available]
        report.append("| " + " | ".join(hdr) + " |")
        report.append("|" + "|".join(["---"] * len(hdr)) + "|")

        for mk in self.model_keys:
            mdf = df[df["model"] == mk]
            if mdf.empty:
                continue
            cells = [MODEL_REGISTRY[mk]["label"]]
            for pol in available:
                vals = mdf[pol].apply(
                    lambda x: x.get("mse_mean", float("nan")) if isinstance(x, dict) else float("nan")
                ).dropna()
                cells.append(f"{vals.mean():.4f}" if len(vals) > 0 else "—")
            report.append("| " + " | ".join(cells) + " |")
        report.append("")

        # Per-horizon table
        report.extend(["## MSE by Horizon", ""])
        for h in self.forecast_horizons:
            report.append(f"### H = {h}")
            hdr2 = ["Model"] + [POLICY_LABELS[p] for p in available] + ["Best"]
            report.append("| " + " | ".join(hdr2) + " |")
            report.append("|" + "|".join(["---"] * len(hdr2)) + "|")
            for mk in self.model_keys:
                sub = df[(df["model"] == mk) & (df["forecast_horizon"] == h)]
                if sub.empty:
                    continue
                cells = [MODEL_REGISTRY[mk]["label"]]
                row_vals = {}
                for pol in available:
                    vals = sub[pol].apply(
                        lambda x: x.get("mse_mean", float("nan")) if isinstance(x, dict) else float("nan")
                    ).dropna()
                    v = vals.mean() if len(vals) > 0 else float("nan")
                    row_vals[pol] = v
                    cells.append(f"{v:.4f}" if not np.isnan(v) else "—")
                best = min(row_vals, key=lambda k: row_vals[k] if not np.isnan(row_vals[k]) else 1e9)
                cells.append(POLICY_LABELS[best])
                report.append("| " + " | ".join(cells) + " |")
            report.append("")

        # Speed table
        report.extend(["## Total Adaptation Time (seconds) — lower is better", ""])
        hdr3 = ["Model"] + [POLICY_LABELS[p] for p in available]
        report.append("| " + " | ".join(hdr3) + " |")
        report.append("|" + "|".join(["---"] * len(hdr3)) + "|")
        for mk in self.model_keys:
            mdf = df[df["model"] == mk]
            if mdf.empty:
                continue
            cells = [MODEL_REGISTRY[mk]["label"]]
            for pol in available:
                vals = mdf[pol].apply(
                    lambda x: x.get("total_time", 0) if isinstance(x, dict) else 0
                )
                cells.append(f"{vals.mean():.1f}")
            report.append("| " + " | ".join(cells) + " |")
        report.append("")

        path = self.results_dir / "sliding_window_report.md"
        with open(path, "w") as f:
            f.write("\n".join(report))
        logger.info(f"📄 Report → {path}")


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Sliding-window benchmark (Level 2)")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS)
    parser.add_argument("--datasets", type=str, nargs="+", default=None)
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--max-windows", type=int, default=MAX_WINDOWS)
    parser.add_argument("--stride", type=int, default=WINDOW_STRIDE)
    parser.add_argument("--quick", action="store_true",
                        help="1 seed, 1 horizon, 1 model, 50 windows for smoke testing")
    parser.add_argument("--policies", type=str, nargs="+", default=None,
                        choices=ALL_POLICIES,
                        help="Which policies to run. Default: all 8. "
                             "E.g., --policies retrain  OR  --policies tta ewc rgtta")
    args = parser.parse_args()

    if args.quick:
        args.seeds = 1
        args.horizons = [96]
        if args.models is None:
            args.models = ["dlinear"]
        args.max_windows = 50

    bench = SlidingWindowBenchmark(
        n_seeds=args.seeds,
        forecast_horizons=args.horizons,
        model_keys=args.models or list(MODEL_REGISTRY.keys()),
        max_windows=args.max_windows,
        stride=args.stride,
        policies=args.policies,
    )
    logger.info(f"Running policies: {bench.policies}")
    bench.run_full(datasets=args.datasets or SLIDING_DATASETS)


if __name__ == "__main__":
    main()
