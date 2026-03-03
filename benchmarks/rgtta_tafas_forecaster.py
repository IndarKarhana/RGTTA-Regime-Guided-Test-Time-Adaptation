"""
RGTTA+TAFAS Forecaster — Regime-Guided GCM Adaptation
======================================================

Combines RGTTA's *proactive* regime detection with TAFAS's frozen-model +
GCM calibration paradigm.  This is our 9th update policy.

Core insight:  Vanilla TAFAS resets GCMs every batch and always uses the
same adaptation effort.  By adding regime-guidance, we can:

  HIGH tier (recurring regime):
      → **Load saved GCM weights** from the matched regime checkpoint instead
        of resetting.  The GCM is already calibrated for this distribution.
        Light fine-tuning (10 sub-windows, low LR).

  MID tier (moderate novelty):
      → Reset GCMs + standard TAFAS adaptation (default sub-windows, standard LR).
        Same as vanilla TAFAS.

  LOW tier (distribution shock):
      → Reset GCMs + aggressive adaptation (more sub-windows, higher LR) to
        recalibrate fast.

GCM weights are tiny (a few hundred parameters), so checkpoint storage is
nearly free.  The base model stays fully frozen throughout (same as TAFAS).

Design rationale: documented in docs/RGTTA_DESIGN_DECISIONS.md.

Reference:
    Kim, H.G., Kim, S., Mok, J. & Yoon, S. (2025). "Battling the
    Non-stationarity in Time Series Forecasting via Test-time Adaptation."
    AAAI-25. https://github.com/kimanki/TAFAS
"""

import copy
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import ks_2samp, wasserstein_distance

from regime_forecasting.models.transformer import TimeSeriesTransformer, regime_aware_loss
from regime_forecasting.utils.data_utils import (
    DataPreprocessor,
    create_lagged_features,
    prepare_sequences,
)

# Reuse GCM from tafas_forecaster (single source of truth)
from tafas_forecaster import GatedCalibrationModule

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight GCM-regime memory
# ---------------------------------------------------------------------------
class _GCMRegimeMemory:
    """Stores (features, raw_values, gcm_state) for regime-guided GCM retrieval.

    Unlike RGTTA's memory which stores full model weights (~60K–330K params),
    this stores only GCM weights (~few hundred params) — nearly free.

    Uses the same 4-method similarity ensemble as RGTTA:
      KS(0.3) + Wasserstein(0.3) + Feature(0.2) + VarRatio(0.2)
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
        gcm_state: Dict[str, Any],
        raw_values: Optional[np.ndarray] = None,
    ) -> None:
        """Save GCM state associated with a distributional signature."""
        self._entries.append({
            "features": features.flatten().copy(),
            "gcm_state": {k: v.cpu().clone() for k, v in gcm_state.items()},
            "raw_values": raw_values.copy() if raw_values is not None else None,
        })
        while len(self._entries) > self.max_entries:
            self._entries.pop(0)

    def _compute_similarity(
        self,
        q_feat: np.ndarray,
        s_feat: np.ndarray,
        q_raw: Optional[np.ndarray],
        s_raw: Optional[np.ndarray],
    ) -> float:
        """Multi-method similarity: KS + Wasserstein + feature distance + VarRatio."""
        # 1. Feature distance
        dist = float(np.linalg.norm(q_feat - s_feat))
        norm = (np.linalg.norm(q_feat) + np.linalg.norm(s_feat)) / 2.0 + 1e-8
        feat_sim = 1.0 / (1.0 + dist / norm)

        if q_raw is None or s_raw is None or len(q_raw) < 5 or len(s_raw) < 5:
            return feat_sim

        # 2. KS statistic
        try:
            ks_stat, _ = ks_2samp(q_raw, s_raw)
            ks_sim = 1.0 - ks_stat
        except Exception:
            ks_sim = feat_sim

        # 3. Wasserstein-1 distance → similarity
        try:
            w_dist = wasserstein_distance(q_raw, s_raw)
            combined_range = max(np.ptp(q_raw), np.ptp(s_raw), 1e-8)
            w_sim = 1.0 / (1.0 + w_dist / combined_range)
        except Exception:
            w_sim = feat_sim

        # 4. Variance ratio
        try:
            std_q = float(np.std(q_raw))
            std_s = float(np.std(s_raw))
            var_sim = min(std_q, std_s) / (max(std_q, std_s) + 1e-8)
        except Exception:
            var_sim = 1.0

        return float(
            self.w_ks * ks_sim + self.w_wass * w_sim
            + self.w_feat * feat_sim + self.w_var * var_sim
        )

    def query(
        self,
        features: np.ndarray,
        raw_values: Optional[np.ndarray] = None,
    ) -> Tuple[float, Optional[Dict[str, Any]]]:
        """Return (best_similarity, gcm_state | None)."""
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
                best_state = entry["gcm_state"]
        return best_sim, best_state

    def __len__(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# Distribution feature extraction (same as RGTTA)
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
    """Compute distribution features using ALL available columns."""
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
# RGTTA+TAFAS Forecaster
# ===========================================================================
class RGTTATAFASForecaster:
    """
    RGTTA+TAFAS: Regime-Guided GCM Adaptation.

    Combines TAFAS's frozen-model + GCM calibration with RGTTA's regime
    detection.  The base model is fully frozen; only GCMs are adapted.
    Regime memory stores GCM weights (not model weights) for reuse.

    Three-tier adaptation:
      HIGH (sim >= tau_high): Load matched GCM checkpoint + light tuning
      MID  (tau_low <= sim < tau_high): Reset GCMs + standard TAFAS
      LOW  (sim < tau_low): Reset GCMs + aggressive GCM adaptation
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
        # Regime thresholds (same as RGTTA)
        tau_high: float = 0.90,
        tau_low: float = 0.55,
        # Tier-specific GCM adaptation
        subwindows_high: int = 25,      # Tuning after GCM load (v2: raised from 10)
        subwindows_mid: int = 50,       # Standard TAFAS
        subwindows_low: int = 80,       # Aggressive recalibration
        lr_high: float = 0.002,         # Conservative LR for loaded GCMs
        lr_mid: float = 0.005,          # Standard TAFAS LR
        lr_low: float = 0.01,           # Aggressive LR for novel dist
        # TAFAS hyperparams
        gating_init: float = 0.01,
        use_paas: bool = True,
        fixed_pogt_ratio: float = 0.25,
        weight_decay: float = 0.0001,
        # Model selection
        model_class: type = None,
        model_kwargs: Optional[Dict] = None,
        input_dim: int = 1,
        feature_cols: Optional[list] = None,
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

        # Regime thresholds
        self.tau_high = tau_high
        self.tau_low = tau_low

        # Tier configs
        self.subwindows_high = subwindows_high
        self.subwindows_mid = subwindows_mid
        self.subwindows_low = subwindows_low
        self.lr_high = lr_high
        self.lr_mid = lr_mid
        self.lr_low = lr_low

        # TAFAS hyperparams
        self.gating_init = gating_init
        self.use_paas = use_paas
        self.fixed_pogt_ratio = fixed_pogt_ratio
        self.weight_decay = weight_decay

        # State
        self.model: Optional[nn.Module] = None
        self.preprocessor = DataPreprocessor()
        self.exog_cols: List[str] = []
        self.accumulated_data: Optional[pd.DataFrame] = None

        # GCMs
        self.input_gcm: Optional[GatedCalibrationModule] = None
        self.output_gcm: Optional[GatedCalibrationModule] = None
        self._gcm_optimizer: Optional[optim.Optimizer] = None
        self._initial_gcm_state: Optional[Dict] = None
        self._initial_optimizer_state: Optional[Dict] = None
        self._n_adapt: int = 0
        self._pogt_history: List[int] = []

        # Full-GT delayed adaptation (from TAFAS)
        self._stored_full_gt: Optional[tuple] = None

        # Regime memory (stores GCM weights, not model weights)
        self._memory = _GCMRegimeMemory()

        # Diagnostics
        self._last_tier: Optional[str] = None
        self._last_similarity: float = 0.0
        self._tier_counts = {"high": 0, "mid": 0, "low": 0}
        self._gcm_loaded: bool = False

    # ------------------------------------------------------------------
    # PAAS: Periodicity-Aware Adaptation Scheduling (from TAFAS)
    # ------------------------------------------------------------------
    def _compute_pogt_length(self, lookback: np.ndarray) -> int:
        """FFT-based dominant period detection for POGT length."""
        if not self.use_paas:
            return max(4, int(self.forecast_horizon * self.fixed_pogt_ratio))

        v = np.asarray(lookback, dtype=np.float64).flatten()
        if len(v) < 8:
            return max(4, int(self.forecast_horizon * self.fixed_pogt_ratio))

        v = v - v.mean()
        fft_result = np.fft.rfft(v)
        amplitude = np.abs(fft_result)

        if len(amplitude) < 2:
            return max(4, self.forecast_horizon // 4)

        try:
            f_star = int(np.argmax(amplitude[1:])) + 1
            if f_star == 0:
                return max(4, self.forecast_horizon // 4)
            period = max(1, len(v) // f_star)
        except Exception:
            period = 24

        period = max(4, min(period, self.forecast_horizon))
        self._pogt_history.append(period)
        return period

    # ------------------------------------------------------------------
    # GCM management
    # ------------------------------------------------------------------
    def _init_gcms(self, n_input_var: int = 1) -> None:
        """Initialize input and output GCMs and save initial state."""
        self.input_gcm = GatedCalibrationModule(
            window_len=self.sequence_length,
            n_var=n_input_var,
            gating_init=self.gating_init,
        ).to(self.device)

        self.output_gcm = GatedCalibrationModule(
            window_len=self.forecast_horizon,
            n_var=1,
            gating_init=self.gating_init,
        ).to(self.device)

        self._create_gcm_optimizer(self.lr_mid)  # default LR

        # Save initial state for MID/LOW reset
        self._initial_gcm_state = {
            "input_gcm": copy.deepcopy(self.input_gcm.state_dict()),
            "output_gcm": copy.deepcopy(self.output_gcm.state_dict()),
        }
        self._initial_optimizer_state = copy.deepcopy(self._gcm_optimizer.state_dict())

    def _create_gcm_optimizer(self, lr: float) -> None:
        """Create a fresh optimizer with the specified learning rate."""
        gcm_params = list(self.input_gcm.parameters()) + list(self.output_gcm.parameters())
        self._gcm_optimizer = optim.Adam(
            gcm_params, lr=lr, weight_decay=self.weight_decay
        )

    def _reset_gcms(self) -> None:
        """Reset GCMs to initial (identity) state."""
        if self._initial_gcm_state is None:
            return
        self.input_gcm.load_state_dict(
            copy.deepcopy(self._initial_gcm_state["input_gcm"])
        )
        self.output_gcm.load_state_dict(
            copy.deepcopy(self._initial_gcm_state["output_gcm"])
        )

    def _load_gcm_state(self, gcm_state: Dict[str, Any]) -> None:
        """Load GCM weights from a regime checkpoint."""
        input_state = {k: v.clone().to(self.device) for k, v in gcm_state.items()
                       if k.startswith("input_gcm.")}
        output_state = {k: v.clone().to(self.device) for k, v in gcm_state.items()
                        if k.startswith("output_gcm.")}
        # Strip prefix for load_state_dict
        input_state = {k.replace("input_gcm.", ""): v for k, v in input_state.items()}
        output_state = {k.replace("output_gcm.", ""): v for k, v in output_state.items()}
        if input_state:
            self.input_gcm.load_state_dict(input_state)
        if output_state:
            self.output_gcm.load_state_dict(output_state)

    def _get_gcm_state(self) -> Dict[str, Any]:
        """Get combined GCM state dict for storage."""
        state = {}
        for k, v in self.input_gcm.state_dict().items():
            state[f"input_gcm.{k}"] = v.cpu().clone()
        for k, v in self.output_gcm.state_dict().items():
            state[f"output_gcm.{k}"] = v.cpu().clone()
        return state

    # ------------------------------------------------------------------
    # Calibrated forward pass
    # ------------------------------------------------------------------
    def _calibrated_forward(
        self, Xt: torch.Tensor, Xe: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Frozen source model + input/output GCM calibration."""
        Xt_cal = self.input_gcm(Xt)
        pred = self.model(Xt_cal, Xe)
        if pred.dim() == 2:
            pred = pred.unsqueeze(-1)
        pred_cal = self.output_gcm(pred)
        return pred_cal.squeeze(-1)

    # ------------------------------------------------------------------
    # Initial training (same as TAFAS — train then freeze)
    # ------------------------------------------------------------------
    def fit(
        self,
        df: pd.DataFrame,
        epochs: int = 30,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
    ) -> Dict[str, Any]:
        """Train source model, freeze it, init GCMs."""
        start_time = time.time()

        df = df.copy()
        if "unique_id" not in df.columns:
            df["unique_id"] = "ts_001"
        df = df.sort_values("ds").reset_index(drop=True)
        df = create_lagged_features(df, lags=[1, self.season_length])
        self.exog_cols = ["lag_1", f"lag_{self.season_length}"]
        self.accumulated_data = df.copy()

        data_scaled, _ = self.preprocessor.fit_transform(
            df, "y", self.exog_cols, feature_cols=self.feature_cols
        )

        self._scaled_feature_cols = None
        if self.feature_cols:
            self._scaled_feature_cols = [
                f"{c}_scaled" for c in self.feature_cols
                if f"{c}_scaled" in data_scaled.columns
            ]
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
        if n_seq < 4:
            return {"status": "insufficient_data", "training_time": time.time() - start_time}

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

        # DLinear lazy init
        if sum(p.numel() for p in self.model.parameters()) == 0:
            with torch.no_grad():
                dummy_x = torch.zeros(1, self.sequence_length, actual_input_dim).to(self.device)
                self.model(dummy_x)

        n_train = max(1, int(n_seq * (1 - validation_split)))

        X_target = np.clip(X_target, -5, 5)
        y = np.clip(y, -5, 5)
        if X_exog is not None:
            X_exog = np.clip(X_exog, -5, 5)

        Xt = torch.FloatTensor(X_target).to(self.device)
        Xe = torch.FloatTensor(X_exog).to(self.device) if X_exog is not None else None
        yt = torch.FloatTensor(y).to(self.device)
        Xt = torch.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)
        yt = torch.nan_to_num(yt, nan=0.0, posinf=0.0, neginf=0.0)

        Xt_tr, Xt_val = Xt[:n_train], Xt[n_train:]
        Xe_tr = Xe[:n_train] if Xe is not None else None
        Xe_val = Xe[n_train:] if Xe is not None else None
        yt_tr, yt_val = yt[:n_train], yt[n_train:]

        actual_lr = min(learning_rate, 0.0005)
        optimizer = optim.Adam(self.model.parameters(), lr=actual_lr, weight_decay=1e-5, eps=1e-8)
        best_val = float("inf")
        best_state = None
        patience, wait = 5, 0

        self.model.train()
        for epoch in range(epochs):
            idx = torch.randperm(n_train)
            for start in range(0, n_train, batch_size):
                end = min(start + batch_size, n_train)
                bi = idx[start:end]
                xb = Xt_tr[bi]
                eb = Xe_tr[bi] if Xe_tr is not None else None
                yb = yt_tr[bi]

                pred = self.model(xb, eb)
                loss = regime_aware_loss(yb, pred)
                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

            if len(Xt_val) > 0:
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(Xt_val, Xe_val)
                    val_loss = F.mse_loss(val_pred, yt_val).item()
                self.model.train()

                if val_loss < best_val:
                    best_val = val_loss
                    best_state = copy.deepcopy(self.model.state_dict())
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        # --- Freeze ALL source model parameters ---
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        # --- Initialize GCMs ---
        self._init_gcms(n_input_var=actual_input_dim)

        logger.info(
            f"✅ RGTTA+TAFAS source trained (val_loss: {best_val:.4f}), "
            f"model frozen, GCMs initialized"
        )

        return {
            "status": "ok",
            "val_loss": best_val,
            "epochs": epochs,
            "training_time": time.time() - start_time,
        }

    # ------------------------------------------------------------------
    # Update with new data — regime-guided GCM adaptation
    # ------------------------------------------------------------------
    def update_with_new_data(self, new_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Regime-guided TAFAS adaptation.

        1. Compute distributional features of new batch.
        2. Query regime memory for best match.
        3. Route to HIGH/MID/LOW tier:
           - HIGH: Load saved GCM → light sub-window tuning
           - MID:  Reset GCMs  → standard TAFAS sub-window adaptation
           - LOW:  Reset GCMs  → aggressive sub-window adaptation
        4. Run sub-window sliding with tier-specific effort.
        5. Store post-adaptation GCM state in regime memory.
        """
        if self.model is None:
            return {"status": "no_model"}

        start_time = time.time()

        # --- Regime detection ---
        raw_y = new_df["y"].dropna().values.astype(np.float64)
        if self.feature_cols:
            features = _multivariate_distribution_features(new_df, self.feature_cols)
        else:
            features = _distribution_features(raw_y)

        sim, matched_gcm_state = self._memory.query(features, raw_y)
        self._last_similarity = sim

        # --- Tier classification ---
        self._gcm_loaded = False
        if sim >= self.tau_high and matched_gcm_state is not None:
            tier = "high"
            max_subwindows = self.subwindows_high
            gcm_lr = self.lr_high
            # v2 loss-gate: only load GCM checkpoint if it beats reset state
            self._load_gcm_state(matched_gcm_state)
            self._gcm_loaded = True  # tentative — may be reverted by loss-gate
            logger.info(
                f"🔧 RGTTA+TAFAS HIGH: loaded GCM checkpoint (sim={sim:.3f}), "
                f"{max_subwindows} sub-windows @ lr={gcm_lr}"
            )
        elif sim >= self.tau_low:
            tier = "mid"
            max_subwindows = self.subwindows_mid
            gcm_lr = self.lr_mid
            # Reset to initial GCM state (standard TAFAS)
            self._reset_gcms()
            logger.info(
                f"🔧 RGTTA+TAFAS MID: reset GCMs (sim={sim:.3f}), "
                f"{max_subwindows} sub-windows @ lr={gcm_lr}"
            )
        else:
            tier = "low"
            max_subwindows = self.subwindows_low
            gcm_lr = self.lr_low
            # Reset to initial GCM state + aggressive recalibration
            self._reset_gcms()
            logger.info(
                f"🔧 RGTTA+TAFAS LOW: reset GCMs (sim={sim:.3f}), "
                f"{max_subwindows} sub-windows @ lr={gcm_lr}"
            )

        self._last_tier = tier
        self._tier_counts[tier] += 1

        # Create optimizer with tier-specific LR
        self._create_gcm_optimizer(gcm_lr)

        # --- Data preparation ---
        new_df = new_df.copy()
        if "unique_id" not in new_df.columns:
            new_df["unique_id"] = "ts_001"
        new_df = new_df.sort_values("ds").reset_index(drop=True)
        new_df = create_lagged_features(new_df, lags=[1, self.season_length])

        self.accumulated_data = pd.concat(
            [self.accumulated_data, new_df], ignore_index=True
        ).drop_duplicates(subset=["ds"]).sort_values("ds").reset_index(drop=True)

        min_len = self.sequence_length + self.forecast_horizon
        if len(self.accumulated_data) < min_len:
            return {"status": "insufficient_data", "adapt_time": time.time() - start_time}

        window = self.accumulated_data.tail(
            max(min_len + 10, len(new_df) + min_len)
        ).copy()

        if not self.preprocessor.is_fitted:
            return {"status": "preprocessor_not_fitted", "adapt_time": time.time() - start_time}

        self.preprocessor.update_scaler_range(new_df, "y", self.exog_cols)
        data_scaled = self.preprocessor.transform(window, "y", self.exog_cols)

        X_target, X_exog, y = prepare_sequences(
            data_scaled,
            sequence_length=self.sequence_length,
            forecast_horizon=self.forecast_horizon,
            exog_cols=self.exog_cols,
            feature_cols=getattr(self, '_scaled_feature_cols', None),
        )

        if len(X_target) < 1:
            return {"status": "no_sequences", "adapt_time": time.time() - start_time}

        X_target = np.clip(X_target, -5, 5)
        y = np.clip(y, -5, 5)
        if X_exog is not None:
            X_exog = np.clip(X_exog, -5, 5)

        n_sequences = len(X_target)

        Xt = torch.FloatTensor(X_target).to(self.device)
        Xe = torch.FloatTensor(X_exog).to(self.device) if X_exog is not None else None
        yt = torch.FloatTensor(y).to(self.device)
        Xt = torch.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)
        yt = torch.nan_to_num(yt, nan=0.0, posinf=0.0, neginf=0.0)

        # --- v2 Loss-gate: verify loaded GCM actually helps ---
        if self._gcm_loaded and tier == "high":
            with torch.no_grad():
                self.input_gcm.eval()
                self.output_gcm.eval()
                n_gate = min(20, Xt.shape[0])
                gate_idx = torch.randperm(Xt.shape[0])[:n_gate]
                Xt_gate = Xt[gate_idx]
                Xe_gate = Xe[gate_idx] if Xe is not None else None
                yt_gate = yt[gate_idx]

                # Loss with loaded GCM
                pred_loaded = self._calibrated_forward(Xt_gate, Xe_gate)
                loaded_loss = F.mse_loss(pred_loaded, yt_gate).item()

                # Save loaded state, reset, measure reset loss
                loaded_state = self._get_gcm_state()
                self._reset_gcms()
                pred_reset = self._calibrated_forward(Xt_gate, Xe_gate)
                reset_loss = F.mse_loss(pred_reset, yt_gate).item()

            # Gate: loaded GCM must beat reset by 20%
            if loaded_loss < reset_loss * 0.80:
                # Checkpoint wins — reload it
                self._load_gcm_state(loaded_state)
                logger.info(
                    f"✅ Loss-gate PASSED: loaded={loaded_loss:.4f} < "
                    f"reset={reset_loss:.4f}*0.80={reset_loss*0.80:.4f}"
                )
            else:
                # Checkpoint rejected — stay with reset, fall back to MID budget
                self._gcm_loaded = False
                tier = "mid"
                max_subwindows = self.subwindows_mid
                gcm_lr = self.lr_mid
                self._tier_counts["high"] -= 1
                self._tier_counts["mid"] += 1
                self._last_tier = "mid"
                logger.info(
                    f"❌ Loss-gate FAILED: loaded={loaded_loss:.4f} >= "
                    f"reset={reset_loss:.4f}*0.80={reset_loss*0.80:.4f} → MID fallback"
                )
            # Recreate optimizer with (possibly changed) LR
            self._create_gcm_optimizer(gcm_lr)

        # --- Full-GT adaptation on previous batch's stored sequences ---
        n_full_gt = 0
        if self._stored_full_gt is not None:
            s_Xt, s_Xe, s_yt = self._stored_full_gt
            self.input_gcm.train()
            self.output_gcm.train()
            max_seqs = min(50, s_Xt.shape[0])
            idx_fg = torch.randperm(s_Xt.shape[0])[:max_seqs]
            s_Xt_sub = s_Xt[idx_fg]
            s_Xe_sub = s_Xe[idx_fg] if s_Xe is not None else None
            s_yt_sub = s_yt[idx_fg]
            self._n_adapt += 1
            pred_cal = self._calibrated_forward(s_Xt_sub, s_Xe_sub)
            loss = F.mse_loss(pred_cal, s_yt_sub)
            if not (torch.isnan(loss) or torch.isinf(loss)):
                self._gcm_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.input_gcm.parameters()) + list(self.output_gcm.parameters()),
                    max_norm=1.0,
                )
                self._gcm_optimizer.step()
                n_full_gt += 1
            self._stored_full_gt = None

        # --- Sub-window sliding (tier-modulated) ---
        self.input_gcm.train()
        self.output_gcm.train()

        if n_sequences <= max_subwindows:
            window_indices = list(range(n_sequences))
        else:
            window_indices = np.linspace(
                0, n_sequences - 1, max_subwindows, dtype=int
            ).tolist()

        n_windows = len(window_indices)
        total_loss = 0.0
        n_adapted = 0

        for wi in window_indices:
            Xt_i = Xt[wi:wi+1]
            yt_i = yt[wi:wi+1]
            Xe_i = Xe[wi:wi+1] if Xe is not None else None

            lookback_vals = X_target[wi, :, 0]
            pogt_len = self._compute_pogt_length(lookback_vals)
            pogt_len = min(pogt_len, self.forecast_horizon)

            self._n_adapt += 1
            pred_cal = self._calibrated_forward(Xt_i, Xe_i)
            loss = F.mse_loss(pred_cal[:, :pogt_len], yt_i[:, :pogt_len])

            if not (torch.isnan(loss) or torch.isinf(loss)):
                self._gcm_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.input_gcm.parameters()) + list(self.output_gcm.parameters()),
                    max_norm=1.0,
                )
                self._gcm_optimizer.step()
                total_loss += float(loss.item())
                n_adapted += 1

        self.input_gcm.eval()
        self.output_gcm.eval()

        # --- Store current batch for full-GT next time ---
        self._stored_full_gt = (
            Xt.detach().clone(),
            Xe.detach().clone() if Xe is not None else None,
            yt.detach().clone(),
        )

        # --- Store GCM state in regime memory ---
        gcm_state = self._get_gcm_state()
        self._memory.store(features, gcm_state, raw_y)

        adapt_time = time.time() - start_time
        avg_loss = total_loss / max(n_adapted, 1)

        return {
            "status": "adapted",
            "tier": tier,
            "similarity": float(sim),
            "gcm_loaded": self._gcm_loaded,
            "n_subwindows": n_windows,
            "max_subwindows": max_subwindows,
            "gcm_lr": gcm_lr,
            "n_adapted": n_adapted,
            "n_adapt_total": self._n_adapt,
            "avg_loss": avg_loss,
            "n_full_gt": n_full_gt,
            "memory_size": len(self._memory),
            "adapt_time": adapt_time,
        }

    # ------------------------------------------------------------------
    # Prediction (calibrated forward, same as TAFAS)
    # ------------------------------------------------------------------
    def predict(
        self, context_df: pd.DataFrame, steps_ahead: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """Generate predictions using frozen model + calibrated GCMs."""
        if self.model is None:
            return None

        steps = steps_ahead or self.forecast_horizon

        self.model.eval()
        self.input_gcm.eval()
        self.output_gcm.eval()

        context_df = context_df.copy()
        if "unique_id" not in context_df.columns:
            context_df["unique_id"] = "ts_001"
        context_df["y"] = pd.to_numeric(context_df["y"], errors="coerce").astype(np.float64)
        context_df = create_lagged_features(context_df, lags=[1, self.season_length])
        for col in self.exog_cols:
            if col in context_df.columns:
                context_df[col] = (
                    pd.to_numeric(context_df[col], errors="coerce")
                    .fillna(0)
                    .astype(np.float64)
                )

        if not self.preprocessor.is_fitted:
            context_df, _ = self.preprocessor.fit_transform(context_df, "y", self.exog_cols)
        else:
            context_df = self.preprocessor.transform(context_df, "y", self.exog_cols)

        X_target_seq, X_exog_seq, _ = prepare_sequences(
            context_df.tail(self.sequence_length + self.forecast_horizon),
            sequence_length=self.sequence_length,
            forecast_horizon=self.forecast_horizon,
            exog_cols=self.exog_cols,
            feature_cols=getattr(self, '_scaled_feature_cols', None),
        )

        if len(X_target_seq) == 0:
            vals = (
                context_df["y_scaled"].values
                if "y_scaled" in context_df.columns
                else context_df["y"].values
            )
            vals = np.array(vals, dtype=np.float64)
            if len(vals) == 0:
                return None
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
            pred = self._calibrated_forward(Xt, Xe)
            pred_scaled = pred[0].cpu().numpy()

        predictions = self.preprocessor.inverse_transform_target(pred_scaled)
        predictions = np.array(predictions, dtype=np.float64).flatten()

        if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
            context_vals = context_df["y"].values[-self.sequence_length:]
            fallback = float(np.nanmean(context_vals)) if len(context_vals) > 0 else 0.0
            predictions = np.where(
                np.isnan(predictions) | np.isinf(predictions),
                fallback,
                predictions,
            )

        if len(predictions) >= steps:
            return pd.DataFrame({"y_pred": predictions[:steps]})
        else:
            last_val = predictions[-1] if len(predictions) > 0 else 0.0
            pad = np.full(steps - len(predictions), last_val)
            return pd.DataFrame({"y_pred": np.concatenate([predictions, pad])})
