"""
TTA (Test-Time Adaptation) Forecaster: same base model as baseline and regime-aware,
but on each new batch runs K gradient steps on the new/recent data only (no full retrain).

Used as a baseline for publication: policy-only comparison (same TimeSeriesTransformer).
No changes to src/regime_forecasting; additive baseline only.
"""

import logging
import os
import sys
import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from regime_forecasting.models.transformer import TimeSeriesTransformer, regime_aware_loss
from regime_forecasting.utils.data_utils import (
    DataPreprocessor,
    create_lagged_features,
    prepare_sequences,
)

logger = logging.getLogger(__name__)


class TTAForecaster:
    """
    Test-time adaptation forecaster: same base model (TimeSeriesTransformer) as
    BaselineForecaster and CorrectedRegimeForecaster. On each new batch, runs
    tta_steps gradient steps on the new/recent data only (no full retrain on
    all accumulated data). No checkpoint memory.
    """

    def __init__(
        self,
        season_length: int = 12,
        forecast_horizon: int = 6,
        sequence_length: int = 24,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        device: str = "cpu",
        tta_steps: int = 20,
        tta_lr: float = 0.0003,
        model_class: type = None,
        model_kwargs: Optional[Dict] = None,
        input_dim: int = 1,
        feature_cols: Optional[list] = None,
        freeze_backbone: bool = True,
        # Adapter injection (D3 experiment)
        use_adapter: bool = False,
        adapter_bottleneck: int = 16,
        model_key: str = "",
    ):
        self.season_length = season_length
        self.forecast_horizon = forecast_horizon
        self.sequence_length = sequence_length
        self.device = torch.device(device)
        self.tta_steps = tta_steps
        self.tta_lr = tta_lr
        self.freeze_backbone = freeze_backbone

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_dim = input_dim
        self.feature_cols = feature_cols

        self._model_class = model_class or TimeSeriesTransformer
        self._model_kwargs = model_kwargs or {}
        self.use_adapter = use_adapter
        self.adapter_bottleneck = adapter_bottleneck
        self.model_key = model_key

        self.model = None
        self.preprocessor = DataPreprocessor()
        self.exog_cols = []
        self.accumulated_data: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Freeze/unfreeze helpers
    # ------------------------------------------------------------------
    def _freeze_backbone(self) -> int:
        """Freeze all layers except the output projection.

        Supports multiple model architectures:
          - GRU / iTransformer / LargeGRU: ``output_projection``
          - PatchTST: ``_head`` (nn.Sequential containing nn.Linear)
          - DLinear: ``_linear_seasonal``, ``_linear_trend``

        Returns trainable param count.
        """
        if self.model is None:
            return 0
        # Top-level module names that are "output" layers across architectures
        output_layers = {
            "output_projection",  # GRU-Small, iTransformer, LargeGRU
            "_head",  # PatchTST
            "_linear_seasonal",  # DLinear
            "_linear_trend",  # DLinear
            "adapters",  # Bottleneck adapters (D3)
        }
        trainable = 0
        for name, param in self.model.named_parameters():
            top_level = name.split(".")[0]
            if top_level in output_layers:
                param.requires_grad = True
                trainable += param.numel()
            else:
                param.requires_grad = False
        if trainable == 0:
            logger.warning(
                "⚠️ _freeze_backbone: no output-layer params matched "
                "(model type may be unsupported) — unfreezing ALL parameters"
            )
            for param in self.model.parameters():
                param.requires_grad = True
                trainable += param.numel()
        return trainable

    def _unfreeze_all(self) -> int:
        """Unfreeze all parameters. Returns total count."""
        if self.model is None:
            return 0
        total = 0
        for param in self.model.parameters():
            param.requires_grad = True
            total += param.numel()
        return total

    def fit(
        self,
        df: pd.DataFrame,
        epochs: int = 30,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Initial training from scratch (same as baseline). Called once at start.
        """
        start_time = time.time()

        df = df.copy()
        if "unique_id" not in df.columns:
            df["unique_id"] = "ts_001"
        df = df.sort_values("ds").reset_index(drop=True)

        df = create_lagged_features(df, lags=[1, self.season_length])
        self.exog_cols = ["lag_1", f"lag_{self.season_length}"]

        self.accumulated_data = df.copy()

        data_scaled, _ = self.preprocessor.fit_transform(df, "y", self.exog_cols, feature_cols=self.feature_cols)

        # Build list of scaled feature column names for multivariate input
        self._scaled_feature_cols = None
        if self.feature_cols:
            self._scaled_feature_cols = [
                f"{c}_scaled" for c in self.feature_cols if f"{c}_scaled" in data_scaled.columns
            ]
            # Always include y_scaled as first channel
            if "y_scaled" not in self._scaled_feature_cols:
                self._scaled_feature_cols = ["y_scaled"] + self._scaled_feature_cols

        X_target, X_exog, y = prepare_sequences(
            data_scaled,
            sequence_length=self.sequence_length,
            forecast_horizon=self.forecast_horizon,
            exog_cols=self.exog_cols,
            feature_cols=self._scaled_feature_cols,
        )

        n_sequences = len(X_target)
        if n_sequences < 2:
            return {
                "status": "skipped",
                "reason": "insufficient_sequences",
                "training_time": time.time() - start_time,
            }

        n_exog = X_exog.shape[2] if X_exog is not None else 0
        actual_input_dim = X_target.shape[2] if X_target.ndim == 3 else self.input_dim
        self.model = self._model_class(
            input_dim=actual_input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
            forecast_horizon=self.forecast_horizon,
            season_length=self.season_length,
            exog_dim=n_exog,
            **self._model_kwargs,
        ).to(self.device)

        # DLinear uses lazy layer init — trigger it with a dummy forward
        if sum(p.numel() for p in self.model.parameters()) == 0:
            with torch.no_grad():
                dummy_x = torch.zeros(1, self.sequence_length, actual_input_dim).to(self.device)
                self.model(dummy_x)

        # Inject adapters if requested (D3 experiment)
        if self.use_adapter and self.model_key:
            from regime_forecasting.models.adapter import inject_adapters

            inject_adapters(self.model, self.model_key, self.adapter_bottleneck)

        n_train = max(1, int(n_sequences * (1 - validation_split)))
        train_idx = list(range(n_train))
        val_idx = list(range(n_train, n_sequences)) if n_train < n_sequences else []

        X_target = np.clip(X_target, -5, 5)
        y = np.clip(y, -5, 5)
        if X_exog is not None:
            X_exog = np.clip(X_exog, -5, 5)

        X_target_tensor = torch.FloatTensor(X_target).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        X_exog_tensor = torch.FloatTensor(X_exog).to(self.device) if X_exog is not None else None
        X_target_tensor = torch.nan_to_num(X_target_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        y_tensor = torch.nan_to_num(y_tensor, nan=0.0, posinf=0.0, neginf=0.0)

        actual_lr = min(learning_rate, 0.0005)
        optimizer = optim.Adam(self.model.parameters(), lr=actual_lr, weight_decay=1e-5, eps=1e-8)
        best_val_loss = float("inf")
        best_model_state = None
        nan_count = 0

        self.model.train()
        for epoch in range(epochs):
            np.random.shuffle(train_idx)
            for i in range(0, len(train_idx), batch_size):
                batch_idx = train_idx[i : i + batch_size]
                batch_target = X_target_tensor[batch_idx]
                batch_y = y_tensor[batch_idx]
                batch_exog = X_exog_tensor[batch_idx] if X_exog_tensor is not None else None

                optimizer.zero_grad()
                predictions = self.model(batch_target, batch_exog)
                loss = regime_aware_loss(batch_y, predictions)
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_count += 1
                    if nan_count > 10:
                        for p in self.model.parameters():
                            if p.dim() > 1:
                                torch.nn.init.xavier_uniform_(p, gain=0.1)
                            else:
                                torch.nn.init.zeros_(p)
                        nan_count = 0
                    continue
                nan_count = 0
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

            if val_idx:
                self.model.eval()
                with torch.no_grad():
                    val_target = X_target_tensor[val_idx]
                    val_y = y_tensor[val_idx]
                    val_exog = X_exog_tensor[val_idx] if X_exog_tensor is not None else None
                    val_pred = self.model(val_target, val_exog)
                    val_loss = regime_aware_loss(val_y, val_pred).item()
                model_has_nan = any(torch.isnan(p).any() for p in self.model.parameters())
                if val_loss < best_val_loss and not np.isnan(val_loss) and not model_has_nan:
                    best_val_loss = val_loss
                    best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.model.train()

        if best_model_state is not None:
            has_nan = any(torch.isnan(v).any() for v in best_model_state.values())
            if not has_nan:
                self.model.load_state_dict(best_model_state)
            else:
                logger.warning("TTA: best model state had NaN, using final weights")

        training_time = time.time() - start_time
        return {
            "status": "completed",
            "training_time": training_time,
            "trained_from_scratch": True,
            "n_sequences": n_sequences,
        }

    def update_with_new_data(
        self,
        new_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        TTA: run tta_steps gradient steps on new/recent data only (no full retrain).
        Same base model; only the update policy differs from baseline and ours.
        """
        if self.model is None:
            return {"status": "skipped", "reason": "no_model", "tta_time": 0.0}

        start_time = time.time()

        new_df = new_df.copy()
        if "unique_id" not in new_df.columns:
            new_df["unique_id"] = "ts_001"
        new_df = new_df.sort_values("ds").reset_index(drop=True)
        new_df = create_lagged_features(new_df, lags=[1, self.season_length])

        self.accumulated_data = (
            pd.concat([self.accumulated_data, new_df], ignore_index=True)
            .drop_duplicates(subset=["ds"])
            .sort_values("ds")
            .reset_index(drop=True)
        )

        # Use most recent window that includes new data (min one sequence)
        min_len = self.sequence_length + self.forecast_horizon
        if len(self.accumulated_data) < min_len:
            tta_time = time.time() - start_time
            return {"status": "skipped", "reason": "insufficient_data", "tta_time": tta_time}

        window = self.accumulated_data.tail(max(min_len + 10, len(new_df) + min_len)).copy()

        if not self.preprocessor.is_fitted:
            tta_time = time.time() - start_time
            return {"status": "skipped", "reason": "preprocessor_not_fitted", "tta_time": tta_time}

        # Incrementally expand scaler range if new data exceeds fitted bounds
        self.preprocessor.update_scaler_range(new_df, "y", self.exog_cols)

        data_scaled = self.preprocessor.transform(window, "y", self.exog_cols)

        X_target, X_exog, y = prepare_sequences(
            data_scaled,
            sequence_length=self.sequence_length,
            forecast_horizon=self.forecast_horizon,
            exog_cols=self.exog_cols,
            feature_cols=getattr(self, "_scaled_feature_cols", None),
        )

        if len(X_target) < 1:
            tta_time = time.time() - start_time
            return {"status": "skipped", "reason": "no_sequences", "tta_time": tta_time}

        X_target = np.clip(X_target, -5, 5)
        y = np.clip(y, -5, 5)
        if X_exog is not None:
            X_exog = np.clip(X_exog, -5, 5)

        X_target_tensor = torch.FloatTensor(X_target).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        X_exog_tensor = torch.FloatTensor(X_exog).to(self.device) if X_exog is not None else None
        X_target_tensor = torch.nan_to_num(X_target_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        y_tensor = torch.nan_to_num(y_tensor, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply frozen backbone if enabled
        if self.freeze_backbone:
            self._freeze_backbone()

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.tta_lr, weight_decay=1e-5, eps=1e-8
        )
        self.model.train()
        for _ in range(self.tta_steps):
            optimizer.zero_grad()
            predictions = self.model(X_target_tensor, X_exog_tensor)
            loss = regime_aware_loss(y_tensor, predictions)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

        # Unfreeze for next operation (prediction/etc)
        if self.freeze_backbone:
            self._unfreeze_all()

        tta_time = time.time() - start_time
        return {
            "status": "completed",
            "tta_time": tta_time,
            "n_sequences": len(X_target),
            "frozen_backbone": self.freeze_backbone,
        }

    def predict(self, context_df: pd.DataFrame, steps_ahead: int) -> pd.DataFrame:
        """Direct multi-horizon prediction (single forward pass, not autoregressive)."""
        if self.model is None:
            raise ValueError("Model not trained.")

        self.model.eval()

        context_df = context_df.copy()
        if "unique_id" not in context_df.columns:
            context_df["unique_id"] = "ts_001"
        context_df["y"] = pd.to_numeric(context_df["y"], errors="coerce").astype(np.float64)
        context_df = create_lagged_features(context_df, lags=[1, self.season_length])
        for col in self.exog_cols:
            if col in context_df.columns:
                context_df[col] = pd.to_numeric(context_df[col], errors="coerce").fillna(0).astype(np.float64)

        if not self.preprocessor.is_fitted:
            context_df, _ = self.preprocessor.fit_transform(context_df, "y", self.exog_cols)
        else:
            context_df = self.preprocessor.transform(context_df, "y", self.exog_cols)

        # Build a single sequence from the most recent data
        X_target_seq, X_exog_seq, _ = prepare_sequences(
            context_df.tail(self.sequence_length + self.forecast_horizon),
            sequence_length=self.sequence_length,
            forecast_horizon=self.forecast_horizon,
            exog_cols=self.exog_cols,
            feature_cols=getattr(self, "_scaled_feature_cols", None),
        )

        if len(X_target_seq) == 0:
            # Fallback: build manual sequence
            vals = context_df["y_scaled"].values if "y_scaled" in context_df.columns else context_df["y"].values
            vals = np.array(vals, dtype=np.float64)
            if len(vals) == 0:
                return pd.DataFrame({"y_pred": [0.0] * steps_ahead})
            if len(vals) >= self.sequence_length:
                seq = vals[-self.sequence_length :]
            else:
                pad_val = float(vals[0]) if len(vals) > 0 else 0.0
                pad = np.full(self.sequence_length - len(vals), pad_val, dtype=np.float64)
                seq = np.concatenate([pad, vals])
            input_dim = self.input_dim if hasattr(self, "input_dim") else 1
            X_target_seq = np.array([seq.reshape(-1, 1)], dtype=np.float64)
            if input_dim > 1:
                X_target_seq = np.concatenate(
                    [
                        X_target_seq,
                        np.zeros((1, self.sequence_length, input_dim - 1), dtype=np.float64),
                    ],
                    axis=-1,
                )
            X_exog_seq = None

        X_target_seq = np.clip(X_target_seq, -5, 5)
        if X_exog_seq is not None:
            X_exog_seq = np.clip(X_exog_seq, -5, 5)

        Xt = torch.FloatTensor(X_target_seq[-1:]).to(self.device)
        Xe = torch.FloatTensor(X_exog_seq[-1:]).to(self.device) if X_exog_seq is not None else None
        Xt = torch.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)

        with torch.no_grad():
            pred = self.model(Xt, Xe)  # [1, forecast_horizon]
            pred_scaled = pred[0].cpu().numpy()  # [forecast_horizon]

        # Inverse transform to original scale
        predictions = self.preprocessor.inverse_transform_target(pred_scaled)
        predictions = np.array(predictions, dtype=np.float64).flatten()

        # Handle NaN/Inf
        if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
            context_vals = context_df["y"].values[-self.sequence_length :]
            fallback = float(np.nanmean(context_vals)) if len(context_vals) > 0 else 0.0
            predictions = np.where(
                np.isnan(predictions) | np.isinf(predictions),
                fallback,
                predictions,
            )

        # Truncate or pad to requested steps
        if len(predictions) >= steps_ahead:
            predictions = predictions[:steps_ahead]
        else:
            last_val = predictions[-1] if len(predictions) > 0 else 0.0
            pad = np.full(steps_ahead - len(predictions), last_val)
            predictions = np.concatenate([predictions, pad])

        return pd.DataFrame({"y_pred": predictions})
