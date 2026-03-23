"""
Unified Multi-Model Benchmark Runner
=====================================

Runs the full benchmark matrix:
    {4 models} × {7 policies} × {MSE, MAE, MAPE} × {horizons} × {ETT datasets} × {seeds}

Models:   GRU-Small (60K), iTransformer (114K), PatchTST (123K), DLinear (19K–1.2M)
Policies: Always-retrain, TTA, EWC, DynaTTA, RGTTA, RGTTA+EWC, RGTTA+DynaTTA

Produces a single JSON with all results, from which paper tables are generated.

Usage:
    python benchmarks/run_unified_benchmark.py                    # full run (3 seeds)
    python benchmarks/run_unified_benchmark.py --quick             # smoke test (1 seed, 1 horizon)
    python benchmarks/run_unified_benchmark.py --seeds 5 --models gru_small lstm
"""

import argparse
import json
import logging
import multiprocessing as mp
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
    directional_accuracy, calculate_all_metrics,
)
from tta_forecaster import TTAForecaster
from ewc_forecaster import EWCForecaster
from rgtta_forecaster import RGTTAForecaster
from dynatta_forecaster import DynaTTAForecaster
from rgtta_dynatta_forecaster import RGTTADynaTTAForecaster
try:
    from tafas_forecaster import TAFASForecaster
    from rgtta_tafas_forecaster import RGTTATAFASForecaster
    _TAFAS_AVAILABLE = True
except ImportError:
    _TAFAS_AVAILABLE = False

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
# Model registry
# ---------------------------------------------------------------------------
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "gru_small": {
        "class": TimeSeriesTransformer,
        "kwargs": {"hidden_dim": 64, "num_layers": 2},
        "label": "GRU-Small",
        "approx_params": "60K",
    },
    "itransformer": {
        "class": iTransformerForecaster,
        "kwargs": {"hidden_dim": 64, "num_layers": 2, "num_heads": 2},
        "label": "iTransformer",
        "approx_params": "~150K",
    },
    # gru_large (~330K params) excluded from study: HIGH-tier 12-step budget
    # insufficient for 330K-param convergence, causing cascading MSE spikes.
    # RGTTA is designed for compact models where regime-guided light adaptation
    # is meaningful.
    "patchtst": {
        "class": PatchTSTForecaster,
        "kwargs": {"hidden_dim": 64, "num_layers": 2, "num_heads": 2, "patch_len": 16, "stride": 8},
        "label": "PatchTST",
        "approx_params": "~120K",
    },
    "dlinear": {
        "class": DLinearForecaster,
        "kwargs": {"hidden_dim": 64, "num_layers": 1},
        "label": "DLinear",
        "approx_params": "~19K",
    },
}

# Model-size-aware HIGH-tier step budget.
# Larger models need more steps per batch to converge to a local minimum.
# The speed advantage on HIGH tier comes from lr_high < lr_mid, not fewer steps.
# gru_large is excluded from the study (see comment above).
STEPS_HIGH_BY_MODEL: Dict[str, int] = {
    "gru_small":    12,   # ~60K params  — converges in 12 steps
    "itransformer": 16,   # ~150K params — needs slightly more
    "patchtst":     15,   # ~120K params
    "dlinear":      12,   # ~19K params  — very fast convergence
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_HORIZONS = [96, 192, 336, 720]
DEFAULT_DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Exchange"]
SYNTHETIC_DATASETS = [
    "synth_stable", "synth_trend_break", "synth_slow_drift", "synth_fast_switch",
    "synth_recurring", "synth_volatility", "synth_shock_recovery", "synth_multi_regime",
]
ALL_DATASETS = DEFAULT_DATASETS + SYNTHETIC_DATASETS
BATCH_SIZE = 750
BASE_INITIAL_TRAIN_SIZE = 720
MAX_BATCHES = 10
INITIAL_EPOCHS = 15
SEQUENCE_LENGTH = 96   # Lookback window (L=96 to match DynaTTA protocol)


def get_initial_train_size(forecast_horizon: int) -> int:
    """Ensure enough samples for at least a few training sequences.

    Need seq_len + forecast_horizon samples for 1 sequence, so we use
    max(720, seq_len + forecast_horizon + 100) to guarantee a usable
    initial training set even at H=720.
    """
    min_needed = SEQUENCE_LENGTH + forecast_horizon + 100
    return max(BASE_INITIAL_TRAIN_SIZE, min_needed)

# Season length per dataset — controls lag features, distribution windows,
# and partial-checkpoint sizing.  Derived from each dataset's native frequency.
DATASET_SEASON_LENGTH: Dict[str, int] = {
    # Hourly → 24 = one daily cycle
    "ETTh1": 24, "ETTh2": 24,
    # 15-min → 96 = one daily cycle (4 × 24)
    "ETTm1": 96, "ETTm2": 96,
    # 10-min → 144 = one daily cycle (6 × 24)
    "Weather": 144,
    # Hourly → 24 (321 clients)
    "Electricity": 24,
    # Hourly → 24 (862 sensors)
    "Traffic": 24,
    # Daily → 5 = one business week
    "Exchange": 5,
    # Weekly → 52 = one year
    "ILI": 52,
    # Synthetic (hourly) → 24
    "synth_stable": 24, "synth_trend_break": 24, "synth_slow_drift": 24,
    "synth_fast_switch": 24, "synth_recurring": 24, "synth_volatility": 24,
    "synth_shock_recovery": 24, "synth_multi_regime": 24,
}

# Primary 6 policies + retrain baseline (TAFAS excluded from study, added only if available)
ALL_POLICIES = [
    "retrain", "tta", "ewc", "dynatta",
    "rgtta", "rgtta_ewc", "rgtta_dynatta",
]
if _TAFAS_AVAILABLE:
    ALL_POLICIES.extend(["tafas", "rgtta_tafas"])


# ============================================================================
# Top-level worker function (pickle-safe for multiprocessing)
# ============================================================================
def _run_single_worker(args_tuple):
    """Worker function for multiprocessing. Must be top-level for pickle."""
    results_dir, model_key, dataset_name, forecast_horizon, seed, model_keys, policies, use_adapter = args_tuple
    # Suppress noisy logging in workers
    logging.getLogger().setLevel(logging.WARNING)
    bench = UnifiedBenchmark(
        results_dir=results_dir,
        n_seeds=1,  # not used, just for init
        forecast_horizons=[forecast_horizon],
        model_keys=model_keys,
        policies=policies,
        use_adapter=use_adapter,
    )
    try:
        res = bench.run_single(model_key, dataset_name, forecast_horizon, seed)
        return res
    except Exception as e:
        return {"error": str(e), "model": model_key, "dataset": dataset_name,
                "forecast_horizon": forecast_horizon, "seed": seed}


# ============================================================================
# Unified benchmark class
# ============================================================================
class UnifiedBenchmark:
    """Run the full model × policy × dataset × horizon × seed matrix."""

    def __init__(
        self,
        results_dir: Optional[str] = None,
        n_seeds: int = 3,
        forecast_horizons: Optional[List[int]] = None,
        model_keys: Optional[List[str]] = None,
        n_workers: int = 1,
        policies: Optional[List[str]] = None,
        use_adapter: bool = False,
    ):
        if results_dir is None:
            results_dir = str(_benchmark_dir / "results" / "unified")
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.n_seeds = n_seeds
        self.forecast_horizons = forecast_horizons or DEFAULT_HORIZONS
        self.model_keys = model_keys or list(MODEL_REGISTRY.keys())
        self.n_workers = n_workers
        self.policies = policies or ALL_POLICIES  # Filter which policies to run
        self.use_adapter = use_adapter
        self.loader = StandardBenchmarkLoader()

        # Master results dict
        self.results: Dict[str, Any] = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "n_seeds": n_seeds,
                "forecast_horizons": self.forecast_horizons,
                "models": self.model_keys,
                "batch_size": BATCH_SIZE,
                "initial_train_size": BASE_INITIAL_TRAIN_SIZE,
                "max_batches": MAX_BATCHES,
                "sequence_length": SEQUENCE_LENGTH,
                "use_adapter": use_adapter,
            },
            "experiments": [],  # flat list of per-run dicts
        }

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------
    @staticmethod
    def _safe_stat(vals: List[float]) -> Tuple[float, float]:
        """Mean and std, returning NaN for empty lists."""
        if not vals:
            return float("nan"), float("nan")
        return float(np.mean(vals)), float(np.std(vals))

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
        """
        Build all 8 policy forecasters for a given model.

        Returns dict  policy_name -> forecaster_instance.
        """
        model_info = MODEL_REGISTRY[model_key]
        model_cls = model_info["class"]
        model_kw = model_info["kwargs"]
        season_len = DATASET_SEASON_LENGTH.get(dataset_name, 24)

        # Common model constructor kwargs (minus model_class-specific ones)
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
        # Multivariate kwargs — passed to all TTA-style forecasters
        mv_common = dict(
            input_dim=input_dim,
            feature_cols=feature_cols,
        )
        # Adapter kwargs — passed to all TTA-style forecasters (D3 experiment)
        adapter_common = dict(
            use_adapter=self.use_adapter,
            model_key=model_key,
        )

        # 1. Always-retrain  (regime forecaster with impossible threshold)
        ckpt_dir_bl = str(self.results_dir / f"ckpt_retrain_{model_key}_{dataset_name}_{seed}")
        retrain = CorrectedRegimeForecaster(
            season_length=season_len,
            forecast_horizon=forecast_horizon,
            sequence_length=SEQUENCE_LENGTH,
            similarity_threshold=2.0,  # never match
            model_selection="full",
            storage_path=ckpt_dir_bl,
            model_class=model_cls,
            input_dim=input_dim,
            feature_cols=feature_cols,
            **model_common,
        )

        # 2. TTA
        tta = TTAForecaster(
            **common,
            **model_common,
            **mv_common,
            **adapter_common,
            tta_steps=20,
            tta_lr=0.0003,
            model_class=model_cls,
        )

        # 3. EWC
        ewc = EWCForecaster(
            **common,
            **model_common,
            **mv_common,
            **adapter_common,
            model_class=model_cls,
            ewc_lambda=400.0,
            ewc_update_steps=15,
            ewc_lr=0.0003,
        )

        # 4. DynaTTA  (Grover & Etemad, ICML 2025 — dynamic LR baseline)
        dynatta = DynaTTAForecaster(
            **common,
            **model_common,
            **mv_common,
            **adapter_common,
            model_class=model_cls,
            alpha_min=1e-4,
            alpha_max=1e-3,
            kappa=1.0,
            eta=0.1,
            tta_steps=20,
            warmup_factor=1,
        )

        # 6. RGTTA v2 (regime-guided TTA — our core contribution)
        rgtta = RGTTAForecaster(
            **common,
            **model_common,
            **mv_common,
            **adapter_common,
            model_class=model_cls,
            lr_base=3e-4,
            max_steps=25,
            min_steps=5,
            patience=3,
            epsilon=0.005,
            ckpt_gate=0.70,
            lr_sim_scale=0.67,
            use_ewc=False,
        )

        # 7. RGTTA+EWC v2 (regime-guided TTA + EWC)
        rgtta_ewc = RGTTAForecaster(
            **common,
            **model_common,
            **mv_common,
            **adapter_common,
            model_class=model_cls,
            lr_base=3e-4,
            max_steps=25,
            min_steps=5,
            patience=3,
            epsilon=0.005,
            ckpt_gate=0.70,
            lr_sim_scale=0.67,
            use_ewc=True,
            ewc_lambda=400.0,
        )

        # 8. RGTTA+DynaTTA v2 (regime checkpoint gate + DynaTTA dynamic LR + early stopping)
        rgtta_dynatta = RGTTADynaTTAForecaster(
            **common,
            **model_common,
            **mv_common,
            **adapter_common,
            model_class=model_cls,
            alpha_min=1e-4,
            alpha_max=1e-3,
            kappa=1.0,
            eta=0.1,
            max_steps=25,
            min_steps=5,
            patience=3,
            epsilon=0.005,
            ckpt_gate=0.70,
            use_ewc=False,
        )

        all_forecasters = {
            "retrain": retrain,
            "tta": tta,
            "ewc": ewc,
            "dynatta": dynatta,
            "rgtta": rgtta,
            "rgtta_ewc": rgtta_ewc,
            "rgtta_dynatta": rgtta_dynatta,
        }

        # TAFAS excluded from primary 6-policy comparison; add only if available
        if _TAFAS_AVAILABLE:
            tafas = TAFASForecaster(
                **common, **model_common, **mv_common, model_class=model_cls,
            )
            rgtta_tafas = RGTTATAFASForecaster(
                **common, **model_common, **mv_common, model_class=model_cls,
            )
            all_forecasters["tafas"] = tafas
            all_forecasters["rgtta_tafas"] = rgtta_tafas
        # Filter by selected policies
        return {k: v for k, v in all_forecasters.items() if k in self.policies}

    # -----------------------------------------------------------------------
    def run_single(
        self,
        model_key: str,
        dataset_name: str,
        forecast_horizon: int,
        seed: int,
    ) -> Optional[Dict[str, Any]]:
        """Run one (model, dataset, horizon, seed) experiment across all 8 policies."""

        np.random.seed(seed)
        torch.manual_seed(seed)

        # Load data (multivariate for real-world datasets, univariate for synthetic)
        dataset_config = self.loader.DATASET_CONFIGS.get(dataset_name, {})
        is_synthetic = dataset_config.get("synthetic", False)
        use_multivariate = not is_synthetic

        initial_train_size = get_initial_train_size(forecast_horizon)
        try:
            initial_df, batches = self.loader.prepare_incremental_batches(
                dataset_name,
                batch_size=BATCH_SIZE,
                initial_train_size=initial_train_size,
                forecast_horizon=forecast_horizon,
                multivariate=use_multivariate,
            )
        except Exception as e:
            logger.error(f"Data load error {dataset_name}: {e}")
            return None

        batches = batches[:MAX_BATCHES]
        if len(batches) < 3:
            logger.warning(f"Skipping {dataset_name} h={forecast_horizon}: only {len(batches)} batches")
            return None

        # Determine multivariate feature columns from the loaded data
        feature_cols = None
        if use_multivariate:
            # Feature columns = all numeric columns except y, unique_id, ds
            # Also exclude the raw target column (e.g. 'OT' for ETT datasets)
            # since it's already represented by 'y'
            raw_target = dataset_config.get("target", "OT")
            exclude = {"unique_id", "ds", "y", raw_target}
            numeric_cols = initial_df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in numeric_cols if c not in exclude]
            if feature_cols:
                logger.info(f"Multivariate mode: {len(feature_cols)} feature columns + target")
            else:
                feature_cols = None  # Fall back to univariate
        input_dim = len(feature_cols) + 1 if feature_cols else 1  # +1 for target 'y'

        # Build forecasters
        forecasters = self._build_forecasters(
            model_key, dataset_name, forecast_horizon, seed,
            input_dim=input_dim, feature_cols=feature_cols,
        )

        # Metrics per policy — track ALL metrics for EVERY policy
        metrics = {
            policy: {
                "mape": [], "mse": [], "mae": [], "rmse": [],
                "smape": [], "direction_acc": [],
                "time": [], "matches": 0,
                # Per-batch detail (ALL policies)
                "batch_metrics": [],      # list of dicts, one per batch
                # Per-batch tier tracking (RGTTA variants only)
                "batch_tiers": [],        # tier label per batch: "high"/"mid"/"low"
                "batch_similarities": [], # similarity score per batch
                "batch_dynamic_lr": [],   # dynamic LR (DynaTTA variants)
            }
            for policy in forecasters
        }

        full_history = initial_df.copy()

        # Policies that use CorrectedRegimeForecaster (fit_incremental / update)
        _regime_policies = {"retrain"}
        # Policies that use the standard fit() / update_with_new_data() interface
        _tta_style_policies = {
            "tta", "ewc", "rgtta", "rgtta_ewc", "dynatta",
            "rgtta_dynatta",
        }
        if _TAFAS_AVAILABLE:
            _tta_style_policies.update({"tafas", "rgtta_tafas"})

        # --- Initial fit ---------------------------------------------------
        for policy, fc in forecasters.items():
            t0 = time.time()
            try:
                if policy in _regime_policies:
                    fc.fit_incremental(full_history)
                else:
                    fc.fit(full_history, epochs=INITIAL_EPOCHS)
            except Exception as e:
                logger.debug(f"Init-fit error {policy}: {e}")
            metrics[policy]["time"].append(time.time() - t0)

        # --- Process batches -----------------------------------------------
        for i, batch in enumerate(batches):
            full_history = pd.concat([full_history, batch], ignore_index=True)

            if len(batch) < forecast_horizon:
                continue

            test_context = full_history.iloc[:-forecast_horizon].copy()
            ground_truth = batch["y"].values[-forecast_horizon:]

            for policy, fc in forecasters.items():
                t0 = time.time()
                try:
                    # Update
                    if policy in _regime_policies:
                        res = fc.update_with_new_data(batch)
                    else:
                        res = fc.update_with_new_data(batch)
                        # Track RGTTA regime tiers, similarity, and dynamic LR
                        if policy in ("rgtta", "rgtta_ewc", "rgtta_dynatta", "rgtta_tafas"):
                            if isinstance(res, dict):
                                tier = res.get("tier", "unknown")
                                sim = res.get("similarity", 0.0)
                                metrics[policy]["batch_tiers"].append(tier)
                                metrics[policy]["batch_similarities"].append(float(sim))
                                if tier in ("high", "ckpt"):
                                    metrics[policy]["matches"] += 1
                                # DynaTTA variants also report dynamic LR
                                if "dynamic_lr" in res:
                                    metrics[policy]["batch_dynamic_lr"].append(float(res["dynamic_lr"]))

                    # Predict
                    pred = fc.predict(test_context, steps_ahead=forecast_horizon)
                    if pred is not None:
                        if hasattr(pred, "values"):
                            pred = pred.values.flatten()
                        if len(pred) == forecast_horizon:
                            gt = ground_truth.astype(float)
                            pr = np.array(pred, dtype=float)
                            batch_mape = float(weighted_mape(gt, pr))
                            batch_mse = float(np.mean((gt - pr) ** 2))
                            batch_mae = float(np.mean(np.abs(gt - pr)))
                            batch_rmse = float(calc_rmse(gt, pr))
                            batch_smape = float(symmetric_mape(gt, pr))
                            batch_da = float(directional_accuracy(gt, pr))
                            metrics[policy]["mape"].append(batch_mape)
                            metrics[policy]["mse"].append(batch_mse)
                            metrics[policy]["mae"].append(batch_mae)
                            metrics[policy]["rmse"].append(batch_rmse)
                            metrics[policy]["smape"].append(batch_smape)
                            if not np.isnan(batch_da):
                                metrics[policy]["direction_acc"].append(batch_da)

                            # --- Per-step-ahead error (granular horizon analysis) ---
                            per_step_mse = ((gt - pr) ** 2).tolist()
                            per_step_mae = (np.abs(gt - pr)).tolist()

                            # Per-batch detail dict (ALL policies)
                            batch_time = time.time() - t0
                            batch_detail = {
                                "batch": i,
                                "mse": batch_mse,
                                "mae": batch_mae,
                                "rmse": batch_rmse,
                                "mape": batch_mape,
                                "smape": batch_smape,
                                "direction_acc": batch_da if not np.isnan(batch_da) else None,
                                "batch_time": batch_time,
                                "per_step_mse": per_step_mse,
                                "per_step_mae": per_step_mae,
                            }
                            # Add tier info for RGTTA variants
                            if policy in ("rgtta", "rgtta_ewc", "rgtta_dynatta", "rgtta_tafas") and metrics[policy]["batch_tiers"]:
                                batch_detail["tier"] = metrics[policy]["batch_tiers"][-1]
                                batch_detail["similarity"] = metrics[policy]["batch_similarities"][-1]
                            # Add v2 step/lr diagnostics for RGTTA variants
                            if policy in ("rgtta", "rgtta_ewc", "rgtta_dynatta") and isinstance(res, dict):
                                batch_detail["steps_used"] = res.get("steps_used", 0)
                                batch_detail["lr_used"] = res.get("lr_used", res.get("dynamic_lr", 0))
                                batch_detail["loaded_checkpoint"] = res.get("loaded_checkpoint", False)
                            # Add DynaTTA diagnostics
                            if policy == "dynatta" and isinstance(res, dict):
                                batch_detail["alpha_t"] = res.get("alpha_t")
                                batch_detail["z_score"] = res.get("z_score")
                                batch_detail["dist_rtab"] = res.get("dist_rtab")
                                batch_detail["dist_rdb"] = res.get("dist_rdb")
                            # Add TAFAS diagnostics
                            if policy == "tafas" and isinstance(res, dict):
                                batch_detail["avg_loss"] = res.get("avg_loss")
                                batch_detail["n_adapted"] = res.get("n_adapted")
                                batch_detail["n_subwindows"] = res.get("n_subwindows")
                                batch_detail["n_full_gt"] = res.get("n_full_gt", 0)
                            # Add RGTTA+TAFAS diagnostics
                            if policy == "rgtta_tafas" and isinstance(res, dict):
                                batch_detail["avg_loss"] = res.get("avg_loss")
                                batch_detail["n_adapted"] = res.get("n_adapted")
                                batch_detail["n_subwindows"] = res.get("n_subwindows")
                                batch_detail["n_full_gt"] = res.get("n_full_gt", 0)
                                batch_detail["gcm_loaded"] = res.get("gcm_loaded", False)
                            metrics[policy]["batch_metrics"].append(batch_detail)
                except Exception as e:
                    logger.debug(f"Batch {i} {policy} error: {e}")
                metrics[policy]["time"].append(time.time() - t0)

        # --- Assemble result -----------------------------------------------
        result: Dict[str, Any] = {
            "model": model_key,
            "model_label": MODEL_REGISTRY[model_key]["label"],
            "dataset": dataset_name,
            "forecast_horizon": forecast_horizon,
            "seed": seed,
            "n_batches": len(batches),
        }
        for policy in forecasters:
            m = metrics[policy]
            mu_mape, sd_mape = self._safe_stat(m["mape"])
            mu_mse, sd_mse = self._safe_stat(m["mse"])
            mu_mae, sd_mae = self._safe_stat(m["mae"])
            mu_rmse, sd_rmse = self._safe_stat(m["rmse"])
            mu_smape, sd_smape = self._safe_stat(m["smape"])
            mu_da, sd_da = self._safe_stat(m["direction_acc"])
            pol_result = {
                "mse_mean": mu_mse,
                "mse_std": sd_mse,
                "mae_mean": mu_mae,
                "mae_std": sd_mae,
                "rmse_mean": mu_rmse,
                "rmse_std": sd_rmse,
                "mape_mean": mu_mape,
                "mape_std": sd_mape,
                "wmape_mean": mu_mape,   # explicit alias (weighted_mape)
                "wmape_std": sd_mape,
                "smape_mean": mu_smape,
                "smape_std": sd_smape,
                "direction_acc_mean": mu_da,
                "direction_acc_std": sd_da,
                "total_time": sum(m["time"]),
                "n_batches_evaluated": len(m["mse"]),
                "matches": m.get("matches", 0),
                "match_rate": m["matches"] / len(batches) if len(batches) > 0 else 0,
                "batch_metrics": m["batch_metrics"],  # full per-batch detail
            }

            # --- Per-step-ahead error aggregation (ALL policies) ---
            # Average MSE/MAE at each forecast step across all batches
            bm_with_steps = [b for b in m["batch_metrics"] if "per_step_mse" in b]
            if bm_with_steps:
                n_steps_list = [len(b["per_step_mse"]) for b in bm_with_steps]
                if len(set(n_steps_list)) == 1:  # all same length
                    H = n_steps_list[0]
                    step_mse_arr = np.array([b["per_step_mse"] for b in bm_with_steps])
                    step_mae_arr = np.array([b["per_step_mae"] for b in bm_with_steps])
                    pol_result["per_step_mse_mean"] = step_mse_arr.mean(axis=0).tolist()
                    pol_result["per_step_mae_mean"] = step_mae_arr.mean(axis=0).tolist()
                    # Quartile summaries (step 1, H/4, H/2, 3H/4, H)
                    quartile_idx = [0, max(0, H//4-1), max(0, H//2-1), max(0, 3*H//4-1), H-1]
                    quartile_labels = ["step_1", f"step_{H//4}", f"step_{H//2}", f"step_{3*H//4}", f"step_{H}"]
                    pol_result["horizon_quartile_mse"] = {
                        label: float(step_mse_arr.mean(axis=0)[idx])
                        for label, idx in zip(quartile_labels, quartile_idx)
                    }

            # --- Per-tier detailed tracking (RGTTA variants) ---
            if policy in ("rgtta", "rgtta_ewc", "rgtta_dynatta", "rgtta_tafas") and m["batch_tiers"]:
                tiers = m["batch_tiers"]
                sims = m["batch_similarities"]
                bm_list = m["batch_metrics"]  # per-batch detail dicts

                # Tier counts
                tier_counts = {"high": 0, "mid": 0, "low": 0}
                for t in tiers:
                    if t in tier_counts:
                        tier_counts[t] += 1
                pol_result["tier_counts"] = tier_counts

                # Per-tier accuracy breakdown (from batch_metrics)
                tier_accuracy = {}
                for tier_name in ["high", "mid", "low"]:
                    tier_items = [b for b in bm_list if b.get("tier") == tier_name]
                    if tier_items:
                        tier_mses = [b["mse"] for b in tier_items]
                        tier_mapes = [b["mape"] for b in tier_items]
                        tier_maes = [b["mae"] for b in tier_items]
                        tier_rmses = [b["rmse"] for b in tier_items]
                        tier_smapes = [b["smape"] for b in tier_items]
                        tier_das = [b["direction_acc"] for b in tier_items if b.get("direction_acc") is not None]
                        tier_times = [b["batch_time"] for b in tier_items if "batch_time" in b]
                        tier_accuracy[tier_name] = {
                            "count": len(tier_items),
                            "mse_mean": float(np.mean(tier_mses)),
                            "mse_std": float(np.std(tier_mses)),
                            "mae_mean": float(np.mean(tier_maes)),
                            "mae_std": float(np.std(tier_maes)),
                            "mape_mean": float(np.mean(tier_mapes)),
                            "mape_std": float(np.std(tier_mapes)),
                            "rmse_mean": float(np.mean(tier_rmses)),
                            "smape_mean": float(np.mean(tier_smapes)),
                            "direction_acc_mean": float(np.mean(tier_das)) if tier_das else None,
                            "avg_batch_time": float(np.mean(tier_times)) if tier_times else None,
                        }
                        # Per-tier per-step-ahead MSE (if available)
                        tier_step_items = [b for b in tier_items if "per_step_mse" in b]
                        if tier_step_items:
                            tier_step_arr = np.array([b["per_step_mse"] for b in tier_step_items])
                            tier_accuracy[tier_name]["per_step_mse_mean"] = tier_step_arr.mean(axis=0).tolist()
                pol_result["tier_accuracy"] = tier_accuracy

                # Similarity statistics
                pol_result["similarity_stats"] = {
                    "mean": float(np.mean(sims)),
                    "std": float(np.std(sims)),
                    "min": float(np.min(sims)),
                    "max": float(np.max(sims)),
                }

                # Dynamic LR info (RGTTA+DynaTTA only)
                if m["batch_dynamic_lr"]:
                    pol_result["dynamic_lr_stats"] = {
                        "mean": float(np.mean(m["batch_dynamic_lr"])),
                        "std": float(np.std(m["batch_dynamic_lr"])),
                        "min": float(np.min(m["batch_dynamic_lr"])),
                        "max": float(np.max(m["batch_dynamic_lr"])),
                    }

            # Strip per_step arrays from batch_metrics to keep JSON size manageable.
            # The aggregated per_step_*_mean at the policy level and per-tier level
            # are sufficient for analysis. Raw per-step arrays can be >100KB per
            # experiment at H=720.
            for bm in pol_result.get("batch_metrics", []):
                bm.pop("per_step_mse", None)
                bm.pop("per_step_mae", None)

            result[policy] = pol_result

        return result

    # -----------------------------------------------------------------------
    def run_full(self, datasets: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run the entire benchmark matrix (sequential or parallel)."""
        datasets = datasets or ALL_DATASETS

        # Build job list
        jobs = []
        for model_key in self.model_keys:
            for ds in datasets:
                for h in self.forecast_horizons:
                    for seed in range(self.n_seeds):
                        jobs.append((model_key, ds, h, seed))

        total = len(jobs)
        t_start = time.time()

        logger.info("=" * 70)
        logger.info("  UNIFIED MULTI-MODEL BENCHMARK")
        logger.info("=" * 70)
        logger.info(f"  Models:   {self.model_keys}")
        logger.info(f"  Datasets: {datasets}")
        logger.info(f"  Horizons: {self.forecast_horizons}")
        logger.info(f"  Seeds:    {self.n_seeds}")
        logger.info(f"  Workers:  {self.n_workers}")
        logger.info(f"  Total experiments: {total}")
        logger.info("=" * 70)

        if self.n_workers > 1:
            self._run_parallel(jobs, total)
        else:
            self._run_sequential(jobs, total)

        elapsed = time.time() - t_start
        self.results["metadata"]["total_time_seconds"] = elapsed
        self.results["metadata"]["n_workers"] = self.n_workers
        self._save_results()
        self._generate_report()
        self._export_csv()

        logger.info("=" * 70)
        logger.info(f"✅  BENCHMARK COMPLETE  ({elapsed / 60:.1f} min, {self.n_workers} workers)")
        logger.info(f"    Results JSON: {self.results_dir / 'unified_results.json'}")
        logger.info(f"    Report MD:   {self.results_dir / 'unified_report.md'}")
        logger.info(f"    Full CSV:    {self.results_dir / 'unified_results_full.csv'}")
        logger.info(f"    Batch CSV:   {self.results_dir / 'unified_batch_detail.csv'}")
        logger.info("=" * 70)
        return self.results

    def _run_sequential(self, jobs: list, total: int):
        """Original sequential execution."""
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

    def _run_parallel(self, jobs: list, total: int):
        """Parallel execution using multiprocessing.Pool."""
        # Build worker args — each worker creates its own UnifiedBenchmark
        worker_args = [
            (str(self.results_dir), mk, ds, h, seed, self.model_keys, self.policies, self.use_adapter)
            for mk, ds, h, seed in jobs
        ]

        logger.info(f"🚀 Launching {self.n_workers} parallel workers...")
        done = 0
        # Use 'fork' context to avoid spawn overhead and pipe size limits on macOS
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=self.n_workers) as pool:
            for res in pool.imap_unordered(_run_single_worker, worker_args):
                done += 1
                if res is None:
                    continue
                if "error" in res:
                    logger.error(
                        f"  [{done}/{total}] FAILED: {res['model']} | "
                        f"{res['dataset']} | h={res['forecast_horizon']} | "
                        f"s={res['seed']}: {res['error']}"
                    )
                    continue
                self.results["experiments"].append(res)
                tag = (
                    f"[{done}/{total}] "
                    f"{MODEL_REGISTRY[res['model']]['label']} | "
                    f"{res['dataset']} | h={res['forecast_horizon']} | "
                    f"s={res['seed']}"
                )
                logger.info(f"✅ {tag}")
                if done % 12 == 0:
                    self._save_results()

    # -----------------------------------------------------------------------
    # Save / report
    # -----------------------------------------------------------------------
    def _save_results(self):
        """Write results JSON (numpy-safe)."""
        def _convert(obj):
            if isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_convert(v) for v in obj]
            return obj

        path = self.results_dir / "unified_results.json"
        with open(path, "w") as f:
            json.dump(_convert(self.results), f, indent=2)

    def _export_csv(self):
        """Export two CSVs: experiment-level summary and per-batch detail."""
        exps = self.results["experiments"]
        if not exps:
            return

        ALL_POLICIES = [
            "retrain", "tta", "ewc", "dynatta",
            "rgtta", "rgtta_ewc", "rgtta_dynatta",
        ]
        if _TAFAS_AVAILABLE:
            ALL_POLICIES.extend(["tafas", "rgtta_tafas"])
        METRIC_COLS = [
            "mse_mean", "mse_std", "mae_mean", "mae_std",
            "rmse_mean", "rmse_std", "mape_mean", "mape_std",
            "wmape_mean", "wmape_std",
            "smape_mean", "smape_std", "direction_acc_mean", "direction_acc_std",
            "total_time", "n_batches_evaluated", "matches", "match_rate",
        ]

        # --- 1. Experiment-level CSV ---
        rows = []
        for e in exps:
            base = {
                "model": e["model"],
                "model_label": e["model_label"],
                "dataset": e["dataset"],
                "horizon": e["forecast_horizon"],
                "seed": e["seed"],
                "n_batches": e["n_batches"],
            }
            for pol in ALL_POLICIES:
                if pol not in e:
                    continue
                d = e[pol]
                row = {**base, "policy": pol}
                for col in METRIC_COLS:
                    row[col] = d.get(col, None)
                # RGTTA tier distribution
                tc = d.get("tier_counts", {})
                row["tier_high"] = tc.get("high", None)
                row["tier_mid"] = tc.get("mid", None)
                row["tier_low"] = tc.get("low", None)
                # Similarity stats
                ss = d.get("similarity_stats", {})
                row["sim_mean"] = ss.get("mean", None)
                row["sim_std"] = ss.get("std", None)
                row["sim_min"] = ss.get("min", None)
                row["sim_max"] = ss.get("max", None)
                rows.append(row)

        df_exp = pd.DataFrame(rows)
        csv_path = self.results_dir / "unified_results_full.csv"
        df_exp.to_csv(csv_path, index=False, float_format="%.6f")
        logger.info(f"📊 Experiment CSV → {csv_path}  ({len(df_exp)} rows)")

        # --- 2. Per-batch detail CSV ---
        batch_rows = []
        for e in exps:
            base = {
                "model": e["model"],
                "dataset": e["dataset"],
                "horizon": e["forecast_horizon"],
                "seed": e["seed"],
            }
            for pol in ALL_POLICIES:
                if pol not in e:
                    continue
                d = e[pol]
                for bm in d.get("batch_metrics", []):
                    row = {**base, "policy": pol}
                    row["batch"] = bm.get("batch")
                    row["mse"] = bm.get("mse")
                    row["mae"] = bm.get("mae")
                    row["rmse"] = bm.get("rmse")
                    row["mape"] = bm.get("mape")
                    row["smape"] = bm.get("smape")
                    row["direction_acc"] = bm.get("direction_acc")
                    row["tier"] = bm.get("tier")
                    row["similarity"] = bm.get("similarity")
                    batch_rows.append(row)

        if batch_rows:
            df_batch = pd.DataFrame(batch_rows)
            batch_path = self.results_dir / "unified_batch_detail.csv"
            df_batch.to_csv(batch_path, index=False, float_format="%.6f")
            logger.info(f"📊 Batch CSV   → {batch_path}  ({len(df_batch)} rows)")

    def _generate_report(self):
        """Generate a Markdown summary report from the results."""
        exps = self.results["experiments"]
        if not exps:
            return

        df = pd.DataFrame(exps)
        report_lines = [
            "# Unified Multi-Model Benchmark Report",
            "",
            f"**Date:** {self.results['metadata']['timestamp']}",
            f"**Models:** {', '.join(self.model_keys)}",
            f"**Datasets:** {', '.join(DEFAULT_DATASETS)}",
            f"**Horizons:** {self.forecast_horizons}",
            f"**Seeds:** {self.n_seeds}",
            "",
        ]

        ALL_POLICIES = [
            "retrain", "tta", "ewc", "dynatta",
            "rgtta", "rgtta_ewc", "rgtta_dynatta",
        ]
        if _TAFAS_AVAILABLE:
            ALL_POLICIES.extend(["tafas", "rgtta_tafas"])
        POLICY_LABELS = {
            "retrain": "Retrain", "tta": "TTA",
            "ewc": "EWC", "dynatta": "DynaTTA",
            "rgtta": "RGTTA", "rgtta_ewc": "RGTTA+EWC",
            "rgtta_dynatta": "RGTTA+DynaTTA",
        }
        if _TAFAS_AVAILABLE:
            POLICY_LABELS.update({"tafas": "TAFAS", "rgtta_tafas": "RGTTA+TAFAS"})
        # Only include policies present in the results
        available_policies = [p for p in ALL_POLICIES if p in df.columns]

        # --- Summary table per model: avg MSE + Time across policies --------
        header_cells = ["Model"] + [f"{POLICY_LABELS[p]} MSE" for p in available_policies] + ["Best"]
        time_header = ["Model"] + [f"{POLICY_LABELS[p]} Time(s)" for p in available_policies] + ["Fastest"]
        report_lines.extend([
            "## Summary: Average MSE by Model × Policy",
            "",
            "| " + " | ".join(header_cells) + " |",
            "|" + "|".join(["---"] * len(header_cells)) + "|",
        ])

        time_rows = []
        for mk in self.model_keys:
            mdf = df[df["model"] == mk]
            if mdf.empty:
                continue
            label = MODEL_REGISTRY[mk]["label"]
            row = [label]
            trow = [label]
            vals = {}
            tvals = {}
            for pol in available_policies:
                mses = mdf[pol].apply(lambda x: x["mse_mean"]).dropna()
                avg = mses.mean() if len(mses) > 0 else float("nan")
                vals[pol] = avg
                row.append(f"{avg:.4f}" if not np.isnan(avg) else "—")
                times = mdf[pol].apply(lambda x: x.get("total_time", 0)).dropna()
                tavg = times.mean() if len(times) > 0 else float("nan")
                tvals[pol] = tavg
                trow.append(f"{tavg:.1f}" if not np.isnan(tavg) else "—")
            # Best MSE
            if vals:
                best = min(vals, key=lambda k: vals[k] if not np.isnan(vals[k]) else 1e9)
                row.append(best)
            else:
                row.append("—")
            # Fastest
            if tvals:
                fastest = min(tvals, key=lambda k: tvals[k] if not np.isnan(tvals[k]) else 1e9)
                trow.append(fastest)
            else:
                trow.append("—")
            report_lines.append("| " + " | ".join(row) + " |")
            time_rows.append(trow)

        report_lines.extend([
            "",
            "## Summary: Average Time (seconds) by Model × Policy",
            "",
            "| " + " | ".join(time_header) + " |",
            "|" + "|".join(["---"] * len(time_header)) + "|",
        ])
        for trow in time_rows:
            report_lines.append("| " + " | ".join(trow) + " |")

        # --- Per-dataset per-horizon breakdown --------------------------------
        all_report_datasets = sorted(set(df["dataset"].unique()))

        # MSE table
        report_lines.extend(["", "## Detailed: MSE per Dataset × Horizon (avg over seeds)", ""])
        for mk in self.model_keys:
            label = MODEL_REGISTRY[mk]["label"]
            report_lines.extend([f"### {label}", ""])
            detail_hdr = ["Dataset", "H"] + [POLICY_LABELS[p] for p in available_policies] + ["Best"]
            report_lines.append("| " + " | ".join(detail_hdr) + " |")
            report_lines.append("|" + "|".join(["---"] * len(detail_hdr)) + "|")
            mdf = df[df["model"] == mk]
            for ds in all_report_datasets:
                for h in self.forecast_horizons:
                    sub = mdf[(mdf["dataset"] == ds) & (mdf["forecast_horizon"] == h)]
                    if sub.empty:
                        continue
                    cells = [ds, str(h)]
                    row_vals = {}
                    for pol in available_policies:
                        mse = sub[pol].apply(lambda x: x["mse_mean"]).mean()
                        row_vals[pol] = mse
                        cells.append(f"{mse:.4f}" if not np.isnan(mse) else "—")
                    best_pol = min(row_vals, key=lambda k: row_vals[k] if not np.isnan(row_vals[k]) else 1e9)
                    cells.append(best_pol)
                    report_lines.append("| " + " | ".join(cells) + " |")
            report_lines.append("")

        # MAPE table
        report_lines.extend(["## Detailed: MAPE% per Dataset × Horizon (avg over seeds)", ""])
        for mk in self.model_keys:
            label = MODEL_REGISTRY[mk]["label"]
            report_lines.extend([f"### {label}", ""])
            detail_hdr = ["Dataset", "H"] + [POLICY_LABELS[p] for p in available_policies] + ["Best"]
            report_lines.append("| " + " | ".join(detail_hdr) + " |")
            report_lines.append("|" + "|".join(["---"] * len(detail_hdr)) + "|")
            mdf = df[df["model"] == mk]
            for ds in all_report_datasets:
                for h in self.forecast_horizons:
                    sub = mdf[(mdf["dataset"] == ds) & (mdf["forecast_horizon"] == h)]
                    if sub.empty:
                        continue
                    cells = [ds, str(h)]
                    row_vals = {}
                    for pol in available_policies:
                        mape = sub[pol].apply(lambda x: x["mape_mean"]).mean()
                        row_vals[pol] = mape
                        cells.append(f"{mape:.2f}" if not np.isnan(mape) else "—")
                    best_pol = min(row_vals, key=lambda k: row_vals[k] if not np.isnan(row_vals[k]) else 1e9)
                    cells.append(best_pol)
                    report_lines.append("| " + " | ".join(cells) + " |")
            report_lines.append("")

        # RMSE table
        report_lines.extend(["## Detailed: RMSE per Dataset × Horizon (avg over seeds)", ""])
        for mk in self.model_keys:
            label = MODEL_REGISTRY[mk]["label"]
            report_lines.extend([f"### {label}", ""])
            detail_hdr = ["Dataset", "H"] + [POLICY_LABELS[p] for p in available_policies] + ["Best"]
            report_lines.append("| " + " | ".join(detail_hdr) + " |")
            report_lines.append("|" + "|".join(["---"] * len(detail_hdr)) + "|")
            mdf = df[df["model"] == mk]
            for ds in all_report_datasets:
                for h in self.forecast_horizons:
                    sub = mdf[(mdf["dataset"] == ds) & (mdf["forecast_horizon"] == h)]
                    if sub.empty:
                        continue
                    cells = [ds, str(h)]
                    row_vals = {}
                    for pol in available_policies:
                        val = sub[pol].apply(lambda x: x.get("rmse_mean", float("nan"))).mean()
                        row_vals[pol] = val
                        cells.append(f"{val:.4f}" if not np.isnan(val) else "—")
                    best_pol = min(row_vals, key=lambda k: row_vals[k] if not np.isnan(row_vals[k]) else 1e9)
                    cells.append(best_pol)
                    report_lines.append("| " + " | ".join(cells) + " |")
            report_lines.append("")

        # sMAPE table
        report_lines.extend(["## Detailed: sMAPE% per Dataset × Horizon (avg over seeds)", ""])
        for mk in self.model_keys:
            label = MODEL_REGISTRY[mk]["label"]
            report_lines.extend([f"### {label}", ""])
            detail_hdr = ["Dataset", "H"] + [POLICY_LABELS[p] for p in available_policies] + ["Best"]
            report_lines.append("| " + " | ".join(detail_hdr) + " |")
            report_lines.append("|" + "|".join(["---"] * len(detail_hdr)) + "|")
            mdf = df[df["model"] == mk]
            for ds in all_report_datasets:
                for h in self.forecast_horizons:
                    sub = mdf[(mdf["dataset"] == ds) & (mdf["forecast_horizon"] == h)]
                    if sub.empty:
                        continue
                    cells = [ds, str(h)]
                    row_vals = {}
                    for pol in available_policies:
                        val = sub[pol].apply(lambda x: x.get("smape_mean", float("nan"))).mean()
                        row_vals[pol] = val
                        cells.append(f"{val:.2f}" if not np.isnan(val) else "—")
                    best_pol = min(row_vals, key=lambda k: row_vals[k] if not np.isnan(row_vals[k]) else 1e9)
                    cells.append(best_pol)
                    report_lines.append("| " + " | ".join(cells) + " |")
            report_lines.append("")

        # Direction accuracy table
        report_lines.extend(["## Detailed: Direction Accuracy% per Dataset × Horizon (avg over seeds)", ""])
        for mk in self.model_keys:
            label = MODEL_REGISTRY[mk]["label"]
            report_lines.extend([f"### {label}", ""])
            detail_hdr = ["Dataset", "H"] + [POLICY_LABELS[p] for p in available_policies] + ["Best"]
            report_lines.append("| " + " | ".join(detail_hdr) + " |")
            report_lines.append("|" + "|".join(["---"] * len(detail_hdr)) + "|")
            mdf = df[df["model"] == mk]
            for ds in all_report_datasets:
                for h in self.forecast_horizons:
                    sub = mdf[(mdf["dataset"] == ds) & (mdf["forecast_horizon"] == h)]
                    if sub.empty:
                        continue
                    cells = [ds, str(h)]
                    row_vals = {}
                    for pol in available_policies:
                        val = sub[pol].apply(lambda x: x.get("direction_acc_mean", float("nan"))).mean()
                        row_vals[pol] = val
                        cells.append(f"{val:.1f}" if not np.isnan(val) else "—")
                    best_pol = max(row_vals, key=lambda k: row_vals[k] if not np.isnan(row_vals[k]) else -1)
                    cells.append(best_pol)
                    report_lines.append("| " + " | ".join(cells) + " |")
            report_lines.append("")

        # Time table
        report_lines.extend(["## Detailed: Time(s) per Dataset × Horizon (avg over seeds)", ""])
        for mk in self.model_keys:
            label = MODEL_REGISTRY[mk]["label"]
            report_lines.extend([f"### {label}", ""])
            detail_hdr = ["Dataset", "H"] + [POLICY_LABELS[p] for p in available_policies] + ["Fastest"]
            report_lines.append("| " + " | ".join(detail_hdr) + " |")
            report_lines.append("|" + "|".join(["---"] * len(detail_hdr)) + "|")
            mdf = df[df["model"] == mk]
            for ds in all_report_datasets:
                for h in self.forecast_horizons:
                    sub = mdf[(mdf["dataset"] == ds) & (mdf["forecast_horizon"] == h)]
                    if sub.empty:
                        continue
                    cells = [ds, str(h)]
                    row_vals = {}
                    for pol in available_policies:
                        val = sub[pol].apply(lambda x: x.get("total_time", 0)).mean()
                        row_vals[pol] = val
                        cells.append(f"{val:.1f}")
                    fastest = min(row_vals, key=lambda k: row_vals[k] if not np.isnan(row_vals[k]) else 1e9)
                    cells.append(fastest)
                    report_lines.append("| " + " | ".join(cells) + " |")
            report_lines.append("")

        # --- Regime Tier Analysis (RGTTA variants) ----------------------
        _rgtta_policies = [
            p for p in [
                "rgtta", "rgtta_ewc", "rgtta_dynatta", "rgtta_tafas"
            ] if p in available_policies
        ]
        if _rgtta_policies:
            report_lines.extend([
                "## Regime Tier Analysis",
                "",
                "Shows how often each RGTTA variant hits each tier, and the accuracy "
                "achieved per tier. HIGH-tier batches use checkpoint reuse; LOW-tier batches "
                "use aggressive adaptation.",
                "",
            ])

            # Aggregate tier counts and per-tier accuracy across all experiments
            for pol in _rgtta_policies:
                label = POLICY_LABELS[pol]
                report_lines.extend([f"### {label} — Tier Distribution & Per-Tier Accuracy", ""])

                # Collect across all experiments
                total_high, total_mid, total_low = 0, 0, 0
                tier_mses = {"high": [], "mid": [], "low": []}
                tier_mapes = {"high": [], "mid": [], "low": []}

                for _, row in df.iterrows():
                    pol_data = row.get(pol)
                    if not isinstance(pol_data, dict):
                        continue
                    tc = pol_data.get("tier_counts", {})
                    total_high += tc.get("high", 0)
                    total_mid += tc.get("mid", 0)
                    total_low += tc.get("low", 0)
                    ta = pol_data.get("tier_accuracy", {})
                    for tier_name in ["high", "mid", "low"]:
                        if tier_name in ta:
                            tier_mses[tier_name].append(ta[tier_name]["mse_mean"])
                            tier_mapes[tier_name].append(ta[tier_name]["mape_mean"])

                total_batches = total_high + total_mid + total_low
                if total_batches == 0:
                    report_lines.append("_No tier data available._\n")
                    continue

                report_lines.extend([
                    "| Tier | Count | % of Batches | Avg MSE | Avg MAPE |",
                    "|------|-------|-------------|---------|----------|",
                ])
                for tier_name, count in [("HIGH", total_high), ("MID", total_mid), ("LOW", total_low)]:
                    pct = 100 * count / total_batches if total_batches > 0 else 0
                    tkey = tier_name.lower()
                    mse_avg = f"{np.mean(tier_mses[tkey]):.4f}" if tier_mses[tkey] else "—"
                    mape_avg = f"{np.mean(tier_mapes[tkey]):.2f}" if tier_mapes[tkey] else "—"
                    report_lines.append(f"| {tier_name} | {count} | {pct:.1f}% | {mse_avg} | {mape_avg} |")

                report_lines.append(f"\n**Total batches:** {total_batches}\n")

                # HIGH-tier impact analysis: compare HIGH-tier MSE vs MID-tier MSE
                if tier_mses["high"] and tier_mses["mid"]:
                    high_avg = np.mean(tier_mses["high"])
                    mid_avg = np.mean(tier_mses["mid"])
                    improvement = ((mid_avg - high_avg) / mid_avg) * 100 if mid_avg > 0 else 0
                    if improvement > 0:
                        report_lines.append(
                            f"**HIGH-tier impact:** Checkpoint reuse (HIGH) achieves "
                            f"{improvement:.1f}% lower MSE than standard adaptation (MID), "
                            f"confirming that loading a specialist checkpoint is beneficial."
                        )
                    else:
                        report_lines.append(
                            f"**HIGH-tier impact:** HIGH-tier MSE ({high_avg:.4f}) vs "
                            f"MID-tier MSE ({mid_avg:.4f}). Checkpoint reuse did not "
                            f"outperform standard adaptation on average."
                        )
                report_lines.append("")

        path = self.results_dir / "unified_report.md"
        with open(path, "w") as f:
            f.write("\n".join(report_lines))
        logger.info(f"📄 Report → {path}")


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Unified multi-model benchmark")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS)
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                        help="Dataset names. Use 'all' for ETT + synthetic, "
                             "'synthetic' for synthetic only, or list specific names.")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Which models to benchmark")
    parser.add_argument("--quick", action="store_true",
                        help="1 seed, 1 horizon, 1 model for smoke testing")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers. Use 1 for sequential (default). "
                             "Recommended: CPU_cores // 2 (e.g., 4 for 8-core, 48 for 96-core).")
    parser.add_argument("--policies", type=str, nargs="+", default=None,
                        choices=ALL_POLICIES,
                        help="Which policies to run. Default: all 8. "
                             "E.g., --policies retrain  OR  --policies tta ewc rgtta")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Directory to save results. Default: benchmarks/results/unified")
    parser.add_argument("--use-adapters", action="store_true",
                        help="Inject bottleneck adapters into iTransformer/PatchTST models (D3 experiment). "
                             "GRU and DLinear models are unaffected.")
    args = parser.parse_args()

    # Resolve dataset aliases
    if args.datasets:
        if args.datasets == ["all"]:
            args.datasets = ALL_DATASETS
        elif args.datasets == ["synthetic"]:
            args.datasets = SYNTHETIC_DATASETS
        elif args.datasets == ["ett"]:
            args.datasets = DEFAULT_DATASETS

    if args.quick:
        args.seeds = 1
        args.horizons = [96]
        if args.models is None:
            args.models = ["gru_small"]

    bench = UnifiedBenchmark(
        n_seeds=args.seeds,
        forecast_horizons=args.horizons,
        model_keys=args.models or list(MODEL_REGISTRY.keys()),
        n_workers=args.workers,
        policies=args.policies,
        results_dir=args.results_dir,
        use_adapter=args.use_adapters,
    )
    logger.info(f"Running policies: {bench.policies}")
    if args.use_adapters:
        logger.info("🔌 Adapter injection ENABLED (D3 experiment)")
    bench.run_full(datasets=args.datasets)


if __name__ == "__main__":
    main()
