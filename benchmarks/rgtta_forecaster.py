"""
Regime-Guided Test-Time Adaptation (RGTTA) Forecaster — v2
==========================================================

Core contribution: uses distributional regime detection as a meta-controller
for adaptation intensity. Unlike fixed-strategy TTA (same K steps every batch)
or pure regime reuse (frozen checkpoint), RGTTA modulates:
  - learning rate (smooth function of similarity)
  - step budget (loss-convergence early stopping)
  - checkpoint reuse (strict loss gate)
  - optional EWC regularization strength (for rgtta_ewc variant)

v2 design (replaces fixed 3-tier system):
  - Similarity modulates LR smoothly: lr = lr_base * (1 + scale * (1 - sim)).
  - Step count is loss-driven via early stopping (patience/epsilon).
  - Checkpoint loading requires both high similarity AND >30% loss improvement.
  - On easy batches, stops at 5-10 steps (FASTER than TTA).
  - On hard batches, runs up to max_steps (BETTER than TTA).
  - EWC regularization optionally applied (for rgtta_ewc variant).

Interface matches TTAForecaster / EWCForecaster / BaselineForecaster for the
unified benchmark runner.
"""

import copy
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import ks_2samp
from scipy.stats import wasserstein_distance

from regime_forecasting.models.transformer import TimeSeriesTransformer, regime_aware_loss
from regime_forecasting.utils.data_utils import (
    DataPreprocessor,
    create_lagged_features,
    prepare_sequences,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight regime memory (no pickle files, in-process only)
# ---------------------------------------------------------------------------
class _RegimeMemory:
    """Stores (features, raw_values, model_state) for regime-guided checkpoint retrieval.

    Uses an ensemble of 4 similarity methods:
      1. KS statistic (CDF shape distance)           — w=0.3
      2. Wasserstein-1 distance (earth-mover's dist)  — w=0.3
      3. Feature vector distance (summary statistics)  — w=0.2
      4. Variance ratio (std_min / std_max)            — w=0.2
    Final similarity is a weighted combination.

    The variance ratio (added v2) specifically catches volatility regime
    shifts where the mean is unchanged but spread differs significantly.

    Design rationale documented in the paper (see paper/main.tex).
    """

    def __init__(
        self,
        w_ks: float = 0.3,
        w_wass: float = 0.3,
        w_feat: float = 0.2,
        w_var: float = 0.2,
        max_entries: int = 5,
    ) -> None:
        self._entries: List[Dict[str, Any]] = []
        self.w_ks = w_ks
        self.w_wass = w_wass
        self.w_feat = w_feat
        self.w_var = w_var
        self.max_entries = max_entries

    def store(
        self,
        features: np.ndarray,
        state_dict: Dict[str, Any],
        raw_values: Optional[np.ndarray] = None,
    ) -> None:
        """Save a snapshot of the model associated with a distributional signature.

        When the memory exceeds *max_entries*, the oldest entry is evicted
        (FIFO).  This prevents stale checkpoints from confusing similarity
        queries and keeps memory usage bounded.
        """
        self._entries.append({
            "features": features.flatten().copy(),
            "state_dict": {k: v.cpu().clone() for k, v in state_dict.items()},
            "raw_values": raw_values.copy() if raw_values is not None else None,
        })
        # FIFO eviction: keep only the most recent entries
        while len(self._entries) > self.max_entries:
            self._entries.pop(0)

    def _compute_similarity(
        self,
        q_feat: np.ndarray,
        s_feat: np.ndarray,
        q_raw: Optional[np.ndarray],
        s_raw: Optional[np.ndarray],
    ) -> float:
        """Multi-method similarity: KS + Wasserstein + feature distance."""
        # --- 1. Feature distance (original method) ---
        dist = float(np.linalg.norm(q_feat - s_feat))
        norm = (np.linalg.norm(q_feat) + np.linalg.norm(s_feat)) / 2.0 + 1e-8
        feat_sim = 1.0 / (1.0 + dist / norm)

        # If raw values unavailable, fall back to feature-only
        if q_raw is None or s_raw is None or len(q_raw) < 5 or len(s_raw) < 5:
            return feat_sim

        # --- 2. KS statistic (1 - D_n, so higher = more similar) ---
        try:
            ks_stat, _ = ks_2samp(q_raw, s_raw)
            ks_sim = 1.0 - ks_stat  # D_n ∈ [0,1], so ks_sim ∈ [0,1]
        except Exception:
            ks_sim = feat_sim  # fallback

        # --- 3. Wasserstein-1 distance → similarity ---
        try:
            w_dist = wasserstein_distance(q_raw, s_raw)
            # Normalize by the combined range of both samples
            combined_range = max(
                np.ptp(q_raw), np.ptp(s_raw), 1e-8
            )
            w_sim = 1.0 / (1.0 + w_dist / combined_range)
        except Exception:
            w_sim = feat_sim  # fallback

        # --- 4. Variance ratio (catches volatility regime shifts) ---
        try:
            std_q = float(np.std(q_raw))
            std_s = float(np.std(s_raw))
            var_sim = min(std_q, std_s) / (max(std_q, std_s) + 1e-8)
        except Exception:
            var_sim = 1.0  # fallback: assume same variance

        # --- Weighted combination ---
        sim = (self.w_ks * ks_sim + self.w_wass * w_sim
               + self.w_feat * feat_sim + self.w_var * var_sim)
        return float(sim)

    def query(
        self,
        features: np.ndarray,
        raw_values: Optional[np.ndarray] = None,
    ) -> Tuple[float, Optional[Dict[str, Any]]]:
        """Return (best_similarity, state_dict | None)."""
        if not self._entries:
            return 0.0, None
        q = features.flatten()
        best_sim = -1.0
        best_state = None
        for entry in self._entries:
            sim = self._compute_similarity(
                q, entry["features"],
                raw_values, entry.get("raw_values"),
            )
            if sim > best_sim:
                best_sim = sim
                best_state = entry["state_dict"]
        return best_sim, best_state

    def __len__(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# Distribution feature extraction (same 5-d vector used by the regime forecaster)
# ---------------------------------------------------------------------------
def _distribution_features(values: np.ndarray) -> np.ndarray:
    """Compute [mean, std, skew, kurtosis, autocorr-1] from a 1-d array."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) < 5:
        return np.zeros(5, dtype=np.float64)
    mu = np.mean(v)
    sd = np.std(v) + 1e-8
    skew = float(np.mean(((v - mu) / sd) ** 3))
    kurt = float(np.mean(((v - mu) / sd) ** 4)) - 3.0
    if len(v) > 1:
        ac = float(np.corrcoef(v[:-1], v[1:])[0, 1])
        if np.isnan(ac):
            ac = 0.0
    else:
        ac = 0.0
    return np.array([mu, sd, skew, kurt, ac], dtype=np.float64)


def _multivariate_distribution_features(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> np.ndarray:
    """Compute distribution features using ALL available columns.

    For multivariate datasets, regime changes can manifest in covariates
    (e.g. temperature drop in Weather dataset) that are invisible to
    univariate y-only features.  This function concatenates per-column
    [mean, std] for every numeric feature to produce a richer signature.

    Returns a feature vector of length 5 + 2*n_features:
      [y_mean, y_std, y_skew, y_kurt, y_ac1, col1_mean, col1_std, ...]
    """
    # Always include y's 5-d features as base
    y_vals = df["y"].dropna().values.astype(np.float64)
    base = _distribution_features(y_vals)

    if not feature_cols:
        return base

    extras = []
    for col in feature_cols:
        if col in df.columns and col != "y":
            vals = pd.to_numeric(df[col], errors="coerce").dropna().values.astype(np.float64)
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                extras.append(float(np.mean(vals)))
                extras.append(float(np.std(vals)))
            else:
                extras.extend([0.0, 0.0])
    if extras:
        return np.concatenate([base, np.array(extras, dtype=np.float64)])
    return base


# ===========================================================================
# RGTTA Forecaster
# ===========================================================================
class RGTTAForecaster:
    """
    Regime-Guided TTA (v2): loss-driven early stopping + smooth LR + checkpoint gate.

    Parameters
    ----------
    lr_base : float
        Base learning rate (same as TTA default: 3e-4).
    max_steps : int
        Maximum gradient steps per batch (default 25, slightly above TTA's 20).
    min_steps : int
        Minimum gradient steps before early stopping can trigger (default 5).
    patience : int
        Stop after this many steps with < epsilon relative improvement.
    epsilon : float
        Minimum relative loss improvement to reset patience counter.
    ckpt_gate : float
        Checkpoint must beat current loss by this factor (0.70 = 30% better).
    lr_sim_scale : float
        How much similarity modulates LR: lr = lr_base * (1 + scale * (1 - sim)).
    use_ewc : bool
        If True, add EWC regularisation during adaptation.
    ewc_lambda : float
        EWC penalty weight (only used when use_ewc=True).
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
        # v2 core parameters
        lr_base: float = 3e-4,
        max_steps: int = 25,
        min_steps: int = 5,
        patience: int = 3,
        epsilon: float = 0.005,
        ckpt_gate: float = 0.70,
        lr_sim_scale: float = 0.67,
        ckpt_sim_threshold: float = 0.75,
        # Optional EWC (for rgtta_ewc variant)
        use_ewc: bool = False,
        ewc_lambda: float = 400.0,
        # Model selection
        model_class: type = None,
        model_kwargs: Optional[Dict] = None,
        # Multivariate support
        input_dim: int = 1,
        feature_cols: Optional[list] = None,
        # Frozen backbone (faster, prevents overfitting)
        freeze_backbone: bool = True,
        # Adapter injection (D3 experiment)
        use_adapter: bool = False,
        adapter_bottleneck: int = 16,
        model_key: str = "",
        # Legacy compatibility (ignored)
        **kwargs,
    ):
        self.season_length = season_length
        self.forecast_horizon = forecast_horizon
        self.sequence_length = sequence_length
        self.device = torch.device(device)

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        self._model_class = model_class or TimeSeriesTransformer
        self._model_kwargs = model_kwargs or {}
        self.input_dim = input_dim
        self.feature_cols = feature_cols
        self.freeze_backbone = freeze_backbone
        self.use_adapter = use_adapter
        self.adapter_bottleneck = adapter_bottleneck
        self.model_key = model_key

        # v2 parameters
        self.lr_base = lr_base
        self.max_steps = max_steps
        self.min_steps = min_steps
        self.patience = patience
        self.epsilon = epsilon
        self.ckpt_gate = ckpt_gate
        self.lr_sim_scale = lr_sim_scale
        self.ckpt_sim_threshold = ckpt_sim_threshold

        # EWC option
        self.use_ewc = use_ewc
        self.ewc_lambda = ewc_lambda

        # State
        self.model: Optional[nn.Module] = None
        self.preprocessor = DataPreprocessor()
        self.exog_cols: List[str] = []
        self.accumulated_data: Optional[pd.DataFrame] = None
        self._memory = _RegimeMemory()

        # EWC state (only if needed)
        self._fisher: Optional[Dict[str, torch.Tensor]] = None
        self._anchor_params: Optional[Dict[str, torch.Tensor]] = None

        # Diagnostics
        self._last_similarity: float = 0.0
        self._last_steps_used: int = 0
        self._last_lr: float = 0.0
        self._last_loaded_ckpt: bool = False
        self._total_steps_all_batches: int = 0
        self._batch_count: int = 0

    # ------------------------------------------------------------------
    # Freeze/unfreeze helpers
    # ------------------------------------------------------------------
    def _freeze_backbone(self) -> int:
        """Freeze all layers except the output projection and adapters.

        Supports multiple model architectures:
          - GRU / iTransformer / LargeGRU: ``output_projection``
          - PatchTST: ``_head`` (nn.Sequential containing nn.Linear)
          - DLinear: ``_linear_seasonal``, ``_linear_trend``
          - Adapters: ``adapters`` (BottleneckAdapter modules, if injected)

        Returns trainable param count.
        """
        if self.model is None:
            return 0
        output_layers = {
            "output_projection",  # GRU-Small, iTransformer, LargeGRU
            "_head",              # PatchTST
            "_linear_seasonal",   # DLinear
            "_linear_trend",      # DLinear
            "adapters",           # Bottleneck adapters (D3)
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

    # ------------------------------------------------------------------
    # EWC helpers
    # ------------------------------------------------------------------
    def _compute_fisher(
        self,
        Xt: torch.Tensor,
        Xe: Optional[torch.Tensor],
        yt: torch.Tensor,
        n_samples: int = 200,
    ) -> Dict[str, torch.Tensor]:
        """Estimate diagonal Fisher information."""
        self.model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        n_used = min(n_samples, len(Xt))
        indices = np.random.choice(len(Xt), n_used, replace=False)
        for idx in indices:
            self.model.zero_grad()
            xt = Xt[idx : idx + 1]
            xe = Xe[idx : idx + 1] if Xe is not None else None
            y1 = yt[idx : idx + 1]
            pred = self.model(xt, xe)
            loss = regime_aware_loss(y1, pred)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data.clone() ** 2
        for n in fisher:
            fisher[n] /= max(n_used, 1)
            fisher[n] = torch.clamp(fisher[n], 0, 1e4)
        return fisher

    def _ewc_penalty(self) -> torch.Tensor:
        if self._fisher is None or self._anchor_params is None:
            return torch.tensor(0.0, device=self.device)
        penalty = torch.tensor(0.0, device=self.device)
        for n, p in self.model.named_parameters():
            if n in self._fisher and n in self._anchor_params:
                penalty += (self._fisher[n] * (p - self._anchor_params[n]) ** 2).sum()
        return penalty

    # ------------------------------------------------------------------
    # fit  (initial training — same as TTA / EWC)
    # ------------------------------------------------------------------
    def fit(
        self,
        df: pd.DataFrame,
        epochs: int = 30,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
    ) -> Dict[str, Any]:
        """Train model from scratch on initial data."""
        start_time = time.time()

        df = df.copy()
        if "unique_id" not in df.columns:
            df["unique_id"] = "ts_001"
        df = df.sort_values("ds").reset_index(drop=True)
        df = create_lagged_features(df, lags=[1, self.season_length])
        self.exog_cols = ["lag_1", f"lag_{self.season_length}"]
        self.accumulated_data = df.copy()

        data_scaled, _ = self.preprocessor.fit_transform(df, "y", self.exog_cols, feature_cols=self.feature_cols)

        # Build scaled feature column names for multivariate input
        self._scaled_feature_cols = None
        if self.feature_cols:
            self._scaled_feature_cols = [f"{c}_scaled" for c in self.feature_cols
                                         if f"{c}_scaled" in data_scaled.columns]
            if "y_scaled" not in self._scaled_feature_cols:
                self._scaled_feature_cols = ["y_scaled"] + self._scaled_feature_cols

        X_target, X_exog, y = prepare_sequences(
            data_scaled,
            sequence_length=self.sequence_length,
            forecast_horizon=self.forecast_horizon,
            exog_cols=self.exog_cols,
            feature_cols=self._scaled_feature_cols,
        )
        n_seq = len(X_target)
        if n_seq < 2:
            return {"status": "skipped", "reason": "insufficient_sequences",
                    "training_time": time.time() - start_time}

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

        n_train = max(1, int(n_seq * (1 - validation_split)))
        train_idx = list(range(n_train))
        val_idx = list(range(n_train, n_seq)) if n_train < n_seq else []

        X_target = np.clip(X_target, -5, 5)
        y = np.clip(y, -5, 5)
        if X_exog is not None:
            X_exog = np.clip(X_exog, -5, 5)

        Xt = torch.FloatTensor(X_target).to(self.device)
        yt = torch.FloatTensor(y).to(self.device)
        Xe = torch.FloatTensor(X_exog).to(self.device) if X_exog is not None else None
        Xt = torch.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)
        yt = torch.nan_to_num(yt, nan=0.0, posinf=0.0, neginf=0.0)

        actual_lr = min(learning_rate, 0.0005)
        optimizer = optim.Adam(self.model.parameters(), lr=actual_lr, weight_decay=1e-5, eps=1e-8)
        best_val_loss = float("inf")
        best_state = None
        nan_count = 0

        self.model.train()
        for epoch in range(epochs):
            np.random.shuffle(train_idx)
            for i in range(0, len(train_idx), batch_size):
                idx = train_idx[i : i + batch_size]
                optimizer.zero_grad()
                pred = self.model(Xt[idx], Xe[idx] if Xe is not None else None)
                loss = regime_aware_loss(yt[idx], pred)
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_count += 1
                    if nan_count > 10:
                        for p in self.model.parameters():
                            if p.dim() > 1:
                                nn.init.xavier_uniform_(p, gain=0.1)
                            else:
                                nn.init.zeros_(p)
                        nan_count = 0
                    continue
                nan_count = 0
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

            if val_idx:
                self.model.eval()
                with torch.no_grad():
                    vp = self.model(Xt[val_idx], Xe[val_idx] if Xe is not None else None)
                    vl = regime_aware_loss(yt[val_idx], vp).item()
                has_nan = any(torch.isnan(p).any() for p in self.model.parameters())
                if vl < best_val_loss and not np.isnan(vl) and not has_nan:
                    best_val_loss = vl
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.model.train()

        if best_state is not None and not any(torch.isnan(v).any() for v in best_state.values()):
            self.model.load_state_dict(best_state)

        # Store initial regime in memory (with raw values for KS/Wasserstein)
        raw_vals = df["y"].dropna().values.astype(np.float64)
        init_features = _multivariate_distribution_features(df, self.feature_cols)
        self._memory.store(init_features, self.model.state_dict(), raw_values=raw_vals)

        # Compute initial Fisher + anchor only when EWC is enabled.
        # Fisher costs 200 forward+backward passes — skip for plain RGTTA.
        if self.use_ewc:
            self.model.eval()
            self._fisher = self._compute_fisher(Xt, Xe, yt)
            self._anchor_params = {
                n: p.data.clone() for n, p in self.model.named_parameters() if p.requires_grad
            }

        return {
            "status": "completed",
            "training_time": time.time() - start_time,
            "trained_from_scratch": True,
            "n_sequences": n_seq,
        }

    # ------------------------------------------------------------------
    # update_with_new_data  (v2: early stopping + smooth LR + checkpoint gate)
    # ------------------------------------------------------------------
    def update_with_new_data(self, new_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Loss-driven regime-guided adaptation (v2):
        1. Compute similarity + model's current loss on new batch.
        2. Optionally load checkpoint if loss-gate passes.
        3. Set LR using similarity as smooth multiplier.
        4. Run gradient steps with early stopping on loss convergence.
        5. Store updated model in memory.
        """
        if self.model is None:
            return {"status": "skipped", "reason": "no_model", "rgtta_time": 0.0}

        start_time = time.time()

        # --- Data prep ---
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

        min_len = self.sequence_length + self.forecast_horizon
        if len(self.accumulated_data) < min_len:
            return {"status": "skipped", "reason": "insufficient_data",
                    "rgtta_time": time.time() - start_time}

        # --- Incremental scaler update ---
        self.preprocessor.update_scaler_range(new_df, "y", self.exog_cols)

        # --- Distribution features (for memory + similarity) ---
        raw_vals = new_df["y"].dropna().values.astype(np.float64)
        new_features = _multivariate_distribution_features(new_df, self.feature_cols)
        best_sim, best_state = self._memory.query(new_features, raw_values=raw_vals)
        self._last_similarity = best_sim

        # --- Prepare sequences ---
        window = self.accumulated_data.tail(
            max(min_len + 10, len(new_df) + min_len)
        ).copy()

        if not self.preprocessor.is_fitted:
            return {"status": "skipped", "reason": "preprocessor_not_fitted",
                    "rgtta_time": time.time() - start_time}

        data_scaled = self.preprocessor.transform(window, "y", self.exog_cols)
        X_target, X_exog, y = prepare_sequences(
            data_scaled,
            sequence_length=self.sequence_length,
            forecast_horizon=self.forecast_horizon,
            exog_cols=self.exog_cols,
            feature_cols=getattr(self, '_scaled_feature_cols', None),
        )
        if len(X_target) < 1:
            return {"status": "skipped", "reason": "no_sequences",
                    "rgtta_time": time.time() - start_time}

        X_target = np.clip(X_target, -5, 5)
        y = np.clip(y, -5, 5)
        if X_exog is not None:
            X_exog = np.clip(X_exog, -5, 5)

        Xt = torch.FloatTensor(X_target).to(self.device)
        yt = torch.FloatTensor(y).to(self.device)
        Xe = torch.FloatTensor(X_exog).to(self.device) if X_exog is not None else None
        Xt = torch.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)
        yt = torch.nan_to_num(yt, nan=0.0, posinf=0.0, neginf=0.0)

        # =================================================================
        # STEP 1: Measure current model's loss on new batch
        # =================================================================
        self.model.eval()
        with torch.no_grad():
            current_pred = self.model(Xt, Xe)
            current_loss = regime_aware_loss(yt, current_pred).item()

        # =================================================================
        # STEP 2: Checkpoint loading with strict loss gate
        # =================================================================
        loaded_checkpoint = False
        if best_state is not None and best_sim >= self.ckpt_sim_threshold:
            try:
                saved_state = self.model.state_dict()
                self.model.load_state_dict(
                    {k: v.to(self.device) for k, v in best_state.items()}
                )
                with torch.no_grad():
                    ckpt_pred = self.model(Xt, Xe)
                    ckpt_loss = regime_aware_loss(yt, ckpt_pred).item()

                if (not math.isnan(ckpt_loss) and not math.isnan(current_loss)
                        and ckpt_loss < current_loss * self.ckpt_gate):
                    loaded_checkpoint = True
                    current_loss = ckpt_loss  # Update baseline for early stopping
                    logger.info(
                        f"🔄 RGTTA: checkpoint LOADED "
                        f"(ckpt={ckpt_loss:.4f} < {current_loss:.4f}*{self.ckpt_gate}, "
                        f"sim={best_sim:.3f})")
                    if self.use_ewc:
                        self._anchor_params = {
                            n: p.data.clone()
                            for n, p in self.model.named_parameters()
                            if p.requires_grad
                        }
                else:
                    self.model.load_state_dict(saved_state)
            except Exception as e:
                logger.debug(f"Checkpoint eval failed: {e}")

        self._last_loaded_ckpt = loaded_checkpoint

        # =================================================================
        # STEP 3: Compute LR using similarity as smooth multiplier
        # =================================================================
        # lr = lr_base * (1 + lr_sim_scale * (1 - sim))
        # sim=1.0 → lr = lr_base  (conservative)
        # sim=0.5 → lr = lr_base * 1.335  (more aggressive)
        # sim=0.0 → lr = lr_base * 1.67  (most aggressive)
        lr = self.lr_base * (1.0 + self.lr_sim_scale * (1.0 - best_sim))
        self._last_lr = lr

        # =================================================================
        # STEP 4: Gradient steps with early stopping
        # =================================================================
        if self.freeze_backbone:
            self._freeze_backbone()

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr, weight_decay=1e-5, eps=1e-8,
        )

        self.model.train()
        prev_loss = current_loss
        patience_counter = 0
        steps_used = 0

        for step in range(self.max_steps):
            optimizer.zero_grad()
            pred = self.model(Xt, Xe)
            task_loss = regime_aware_loss(yt, pred)

            if self.use_ewc:
                ewc_pen = self._ewc_penalty()
                total_loss = task_loss + (self.ewc_lambda / 2.0) * ewc_pen
            else:
                total_loss = task_loss

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                continue

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            steps_used += 1

            step_loss = total_loss.item()

            # Early stopping check (only after min_steps)
            if steps_used >= self.min_steps:
                if prev_loss > 0:
                    relative_improvement = (prev_loss - step_loss) / (abs(prev_loss) + 1e-8)
                else:
                    relative_improvement = 0.0

                if relative_improvement < self.epsilon:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info(
                            f"⏱️ RGTTA early stop at step {steps_used}/{self.max_steps} "
                            f"(loss={step_loss:.6f}, sim={best_sim:.3f}, lr={lr:.6f})")
                        break
                else:
                    patience_counter = 0

            prev_loss = step_loss

        self._last_steps_used = steps_used
        self._total_steps_all_batches += steps_used
        self._batch_count += 1

        # --- Unfreeze after adaptation ---
        if self.freeze_backbone:
            self._unfreeze_all()

        # --- Post-update bookkeeping ---
        self._memory.store(new_features, self.model.state_dict(), raw_values=raw_vals)

        # Refresh Fisher + anchor only when EWC is enabled.
        if self.use_ewc:
            self.model.eval()
            new_fisher = self._compute_fisher(Xt, Xe, yt)
            if self._fisher is not None:
                for n in self._fisher:
                    if n in new_fisher:
                        self._fisher[n] = 0.5 * self._fisher[n] + 0.5 * new_fisher[n]
            else:
                self._fisher = new_fisher
            self._anchor_params = {
                n: p.data.clone() for n, p in self.model.named_parameters() if p.requires_grad
            }

        rgtta_time = time.time() - start_time

        # Tier label for diagnostics / compatibility
        if loaded_checkpoint:
            tier_label = "ckpt"
        elif steps_used <= self.min_steps + self.patience:
            tier_label = "easy"  # converged fast
        elif steps_used >= self.max_steps - 1:
            tier_label = "hard"  # used full budget
        else:
            tier_label = "mid"

        return {
            "status": "completed",
            "rgtta_time": rgtta_time,
            "n_sequences": len(X_target),
            "tier": tier_label,
            "similarity": best_sim,
            "steps_used": steps_used,
            "lr_used": lr,
            "loaded_checkpoint": loaded_checkpoint,
            "final_loss": prev_loss,
            "frozen_backbone": self.freeze_backbone,
        }

    # ------------------------------------------------------------------
    # predict  (same interface as TTA / EWC)
    # ------------------------------------------------------------------
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
                context_df[col] = (
                    pd.to_numeric(context_df[col], errors="coerce").fillna(0).astype(np.float64)
                )

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
            feature_cols=getattr(self, '_scaled_feature_cols', None),
        )

        if len(X_target_seq) == 0:
            # Fallback: build manual sequence
            vals = (
                context_df["y_scaled"].values
                if "y_scaled" in context_df.columns
                else context_df["y"].values
            )
            vals = np.array(vals, dtype=np.float64)
            if len(vals) == 0:
                return pd.DataFrame({"y_pred": [0.0] * steps_ahead})
            if len(vals) >= self.sequence_length:
                seq = vals[-self.sequence_length:]
            else:
                pad_val = float(vals[0]) if len(vals) > 0 else 0.0
                pad = np.full(self.sequence_length - len(vals), pad_val, dtype=np.float64)
                seq = np.concatenate([pad, vals])
            X_target_seq = np.array([seq.reshape(-1, 1)], dtype=np.float64)
            X_exog_seq = None

        X_target_seq = np.clip(X_target_seq, -5, 5)
        if X_exog_seq is not None:
            X_exog_seq = np.clip(X_exog_seq, -5, 5)

        Xt = torch.FloatTensor(X_target_seq[-1:]).to(self.device)
        Xe = (
            torch.FloatTensor(X_exog_seq[-1:]).to(self.device)
            if X_exog_seq is not None
            else None
        )
        Xt = torch.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)

        with torch.no_grad():
            pred = self.model(Xt, Xe)  # [1, forecast_horizon]
            pred_scaled = pred[0].cpu().numpy()  # [forecast_horizon]

        # Inverse transform to original scale
        predictions = self.preprocessor.inverse_transform_target(pred_scaled)
        predictions = np.array(predictions, dtype=np.float64).flatten()

        # Handle NaN/Inf
        if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
            context_vals = context_df["y"].values[-self.sequence_length:]
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

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def get_tier_stats(self) -> Dict[str, Any]:
        """Return v2 adaptation statistics."""
        return {
            "avg_steps": self._total_steps_all_batches / max(1, self._batch_count),
            "total_batches": self._batch_count,
            "total_steps": self._total_steps_all_batches,
        }
