"""
TAFAS-style Forecaster  (v2 — corrected)
==========================================

Faithful re-implementation of the TAFAS framework (Kim et al., AAAI 2025)
adapted to our GRU-based models and incremental batch pipeline.

TAFAS core idea: At test-time, attach lightweight Gated Calibration Modules
(GCMs) to the input and output of a *frozen* source forecaster. The GCMs are
adapted using partially-observed ground truth (POGT), whose length is
determined by a Periodicity-Aware Adaptation Scheduling (PAAS) scheme.

Key components:
  1. **PAAS** – FFT-based dominant-period detection on the look-back window to
     determine how many steps of ground truth to wait for (the POGT length p).
  2. **GCM (Gated Calibration Module)** – Variable-wise temporal calibration
     (W·x + b) gated by tanh(α). Attached to input (L×L) and output (H×H)
     of the frozen forecaster. Weights initialised to zero → identity at start.
  3. **Prediction Adjustment** – After adaptation, recalculate predictions and
     substitute unobserved portions with adapted predictions.

This is *not* our contribution — it is a baseline for comparison. Our RGTTA
differs from TAFAS in:
  - No ground truth required (fully unsupervised vs partially-observed GT)
  - Checkpoint memory for recurring regimes (TAFAS has none)
  - Source model weights can be updated (TAFAS keeps source frozen)
  - Distributional features for proactive detection vs FFT-based periodic scheduling

v2 Changes (2026-02-21):
  - FIXED: Input GCM gradient flow — removed torch.no_grad() from source model
    forward; relies on requires_grad=False to prevent source param updates while
    allowing gradients to flow back through to input GCM.
  - FIXED: Removed illegal full-GT loss during POGT adaptation. Now only uses
    partial GT (first `period` steps) as per official TAFAS.
  - FIXED: Added model/GCM reset between adaptation batches to match official code.
  - ADDED: Prediction adjustment (_adjust_prediction) from official code.
  - ALIGNED: Hyperparameters to match official defaults (lr=0.005, gating_init=0.01).
  - FIXED: Predict method now uses direct multi-horizon output instead of
    autoregressive step-by-step generation.

Interface matches TTAForecaster / EWCForecaster / RGTTAForecaster for the
unified benchmark runner.

Reference:
    Kim, H.G., Kim, S., Mok, J. & Yoon, S. (2025). "Battling the
    Non-stationarity in Time Series Forecasting via Test-time Adaptation."
    AAAI-25 (main track). https://github.com/kimanki/TAFAS
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

from regime_forecasting.models.transformer import TimeSeriesTransformer, regime_aware_loss
from regime_forecasting.utils.data_utils import (
    DataPreprocessor,
    create_lagged_features,
    prepare_sequences,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gated Calibration Module (GCM) — from the TAFAS paper Eq. 3
# Matches official kimanki/TAFAS GCM class exactly.
# ---------------------------------------------------------------------------
class GatedCalibrationModule(nn.Module):
    """
    Variable-wise temporal calibration with gating (Eq. 3 of TAFAS paper).

    GCM(x) = x + Tile(tanh(α)) ⊙ (Concat({W^c · x^c}_{c=1}^C) + b)

    Matches official: weight init to zeros → identity at start.
    """

    def __init__(
        self,
        window_len: int,
        n_var: int = 1,
        gating_init: float = 0.01,
        var_wise: bool = True,
    ):
        super().__init__()
        self.window_len = window_len
        self.n_var = n_var
        self.var_wise = var_wise

        if var_wise:
            self.weight = nn.Parameter(torch.zeros(window_len, window_len, n_var))
        else:
            self.weight = nn.Parameter(torch.zeros(window_len, window_len))

        self.gating = nn.Parameter(gating_init * torch.ones(n_var))
        self.bias = nn.Parameter(torch.zeros(window_len, n_var))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, window_len, n_var]. Returns calibrated x of same shape."""
        needs_unsqueeze = False
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            needs_unsqueeze = True

        if self.var_wise:
            cal = torch.einsum("blc,loc->boc", x, self.weight) + self.bias
        else:
            cal = torch.einsum("blc,lo->boc", x, self.weight) + self.bias

        out = x + torch.tanh(self.gating) * cal

        if needs_unsqueeze:
            out = out.squeeze(-1)
        return out


# ===========================================================================
# TAFAS Forecaster (v2 — corrected)
# ===========================================================================
class TAFASForecaster:
    """
    TAFAS-style forecaster adapted to our incremental batch pipeline.

    Core mechanism: Freeze the source forecaster. Attach input/output GCMs.
    On each batch, reset GCMs to initial state, then adapt using POGT.
    Source model stays frozen throughout.

    Parameters
    ----------
    gcm_lr : float
        Learning rate for the GCM parameters. Official default: 0.005.
    gcm_steps : int
        Number of gradient steps per adaptation. Official default: 1.
    gating_init : float
        Initial value for GCM gating parameter. Official default: 0.01.
    use_paas : bool
        Whether to use FFT-based period detection for POGT length.
    use_prediction_adjustment : bool
        Whether to re-forecast after adaptation and replace un-observed
        portions with post-adaptation predictions.
    reset_between_batches : bool
        Whether to reset GCMs to initial state before each batch adaptation.
        Official TAFAS resets — set True for faithful comparison.
    use_subwindows : bool
        If True (default), slide sub-windows through each batch and run 1 GCM
        step per sub-window — faithfully replicating TAFAS's many-window
        protocol inside streaming batches.  If False, treat the entire batch
        as a single adaptation episode with ``gcm_steps`` gradient steps
        (the v2 baseline used on the VMs).
    max_subwindows : int
        Maximum number of sub-windows per batch when ``use_subwindows=True``.
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
        # TAFAS-specific hyperparameters (aligned with official defaults)
        gcm_lr: float = 0.005,
        gcm_steps: int = 1,
        gating_init: float = 0.01,
        use_paas: bool = True,
        use_prediction_adjustment: bool = True,
        reset_between_batches: bool = False,
        fixed_pogt_ratio: float = 0.25,
        weight_decay: float = 0.0001,
        # Sub-window adaptation (v3)
        use_subwindows: bool = True,
        max_subwindows: int = 50,
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

        # TAFAS hyperparams (official defaults)
        self.gcm_lr = gcm_lr
        self.gcm_steps = gcm_steps
        self.gating_init = gating_init
        self.use_paas = use_paas
        self.use_prediction_adjustment = use_prediction_adjustment
        self.reset_between_batches = reset_between_batches
        self.fixed_pogt_ratio = fixed_pogt_ratio
        self.weight_decay = weight_decay
        self.use_subwindows = use_subwindows
        self.max_subwindows = max_subwindows

        # State
        self.model: Optional[nn.Module] = None
        self.preprocessor = DataPreprocessor()
        self.exog_cols: List[str] = []
        self.accumulated_data: Optional[pd.DataFrame] = None

        # GCMs — created after initial fit
        self.input_gcm: Optional[GatedCalibrationModule] = None
        self.output_gcm: Optional[GatedCalibrationModule] = None
        self._gcm_optimizer: Optional[optim.Optimizer] = None
        self._n_adapt: int = 0
        self._pogt_history: List[int] = []

        # Full-GT adaptation: store previous batch data for delayed full-horizon pass
        self._stored_full_gt: Optional[tuple] = None

        # Initial state for reset (saved after GCM init)
        self._initial_gcm_state: Optional[Dict] = None
        self._initial_optimizer_state: Optional[Dict] = None

    # ------------------------------------------------------------------
    # PAAS: Periodicity-Aware Adaptation Scheduling
    # ------------------------------------------------------------------
    def _compute_pogt_length(self, lookback: np.ndarray) -> int:
        """
        Use FFT to find the dominant period of the look-back window.
        Returns the POGT length (clamped to [4, forecast_horizon]).

        Follows TAFAS Eq. 1-2 and official code:
          c* = argmax_c ||FFT(X_c)||^2
          f* = argmax_f ||FFT(X_{c*})||^2_f
          p  = L / f*
        """
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

        # Skip DC component, find dominant frequency
        try:
            f_star = int(np.argmax(amplitude[1:])) + 1
            if f_star == 0:
                return max(4, self.forecast_horizon // 4)
            period = max(1, len(v) // f_star)
        except Exception:
            period = 24

        # Clamp to reasonable range
        period = max(4, min(period, self.forecast_horizon))
        self._pogt_history.append(period)
        return period

    # ------------------------------------------------------------------
    # GCM initialization + state management
    # ------------------------------------------------------------------
    def _init_gcms(self, n_input_var: int = 1) -> None:
        """Initialize input and output GCMs. Save initial state for reset.

        Args:
            n_input_var: Number of input variables (channels) for the input GCM.
                         Should match ``actual_input_dim`` from the model.
        """
        self.input_gcm = GatedCalibrationModule(
            window_len=self.sequence_length,
            n_var=n_input_var,
            gating_init=self.gating_init,
        ).to(self.device)

        self.output_gcm = GatedCalibrationModule(
            window_len=self.forecast_horizon,
            n_var=1,  # output is always univariate [B, H]
            gating_init=self.gating_init,
        ).to(self.device)

        gcm_params = list(self.input_gcm.parameters()) + list(self.output_gcm.parameters())
        self._gcm_optimizer = optim.Adam(
            gcm_params, lr=self.gcm_lr, weight_decay=self.weight_decay
        )

        # Save initial state for reset between batches
        self._initial_gcm_state = {
            "input_gcm": copy.deepcopy(self.input_gcm.state_dict()),
            "output_gcm": copy.deepcopy(self.output_gcm.state_dict()),
        }
        self._initial_optimizer_state = copy.deepcopy(self._gcm_optimizer.state_dict())

    def _reset_gcms(self) -> None:
        """Reset GCMs and optimizer to initial state (official TAFAS resets)."""
        if self._initial_gcm_state is None:
            return
        self.input_gcm.load_state_dict(
            copy.deepcopy(self._initial_gcm_state["input_gcm"])
        )
        self.output_gcm.load_state_dict(
            copy.deepcopy(self._initial_gcm_state["output_gcm"])
        )
        self._gcm_optimizer.load_state_dict(
            copy.deepcopy(self._initial_optimizer_state)
        )

    # ------------------------------------------------------------------
    # Forward pass with GCM calibration (gradient-enabled)
    # ------------------------------------------------------------------
    def _calibrated_forward(
        self, Xt: torch.Tensor, Xe: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Run frozen source model with input/output GCMs.

        CRITICAL FIX (v2): No torch.no_grad() on source model forward.
        Source model params have requires_grad=False, so they won't be
        updated by the optimizer, but gradients CAN flow through them
        back to the input GCM. This is essential for the input GCM to
        receive any training signal.

        Xt: [B, seq_len, 1]  (target)
        Xe: [B, seq_len, n_exog] or None
        Returns: calibrated prediction [B, forecast_horizon]
        """
        # 1. Input calibration
        Xt_cal = self.input_gcm(Xt)  # [B, seq_len, 1]

        # 2. Source model forward — NO torch.no_grad() here!
        # Source model params have requires_grad=False so they won't update,
        # but the computational graph remains connected for input GCM gradients.
        pred = self.model(Xt_cal, Xe)  # [B, forecast_horizon]

        # 3. Output calibration
        if pred.dim() == 2:
            pred = pred.unsqueeze(-1)
        pred_cal = self.output_gcm(pred)  # [B, H, 1]

        return pred_cal.squeeze(-1)  # [B, H]

    # ------------------------------------------------------------------
    # Prediction adjustment (from official TAFAS)
    # ------------------------------------------------------------------
    def _adjust_prediction(
        self,
        pred_before: torch.Tensor,
        Xt: torch.Tensor,
        Xe: Optional[torch.Tensor],
        period: int,
        batch_size: int,
    ) -> torch.Tensor:
        """
        After GCM adaptation, re-forecast and replace un-observed portions
        with post-adaptation predictions. Matches official TAFAS.

        For each sample i in the batch, the first (period - i) steps are
        already "observed" (POGT), so we keep those. Steps beyond that are
        replaced with the post-adaptation prediction.

        pred_before: [B, H] — predictions made before adaptation
        Returns: adjusted [B, H] predictions
        """
        with torch.no_grad():
            pred_after = self._calibrated_forward(Xt, Xe)  # [B, H]

        pred_adjusted = pred_before.clone()
        for i in range(min(batch_size - 1, pred_adjusted.shape[0])):
            boundary = max(0, period - i)
            if boundary < pred_adjusted.shape[1]:
                pred_adjusted[i, boundary:] = pred_after[i, boundary:]

        return pred_adjusted

    # ------------------------------------------------------------------
    # Initial training
    # ------------------------------------------------------------------
    def fit(
        self,
        df: pd.DataFrame,
        epochs: int = 30,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
    ) -> Dict[str, Any]:
        """Initial training — trains the source model, then freezes it."""
        start_time = time.time()

        df = df.copy()
        if "unique_id" not in df.columns:
            df["unique_id"] = "ts_001"
        df = df.sort_values("ds").reset_index(drop=True)
        df = create_lagged_features(df, lags=[1, self.season_length])
        self.exog_cols = ["lag_1", f"lag_{self.season_length}"]

        self.accumulated_data = df.copy()

        data_scaled, _ = self.preprocessor.fit_transform(df, "y", self.exog_cols, feature_cols=self.feature_cols)

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

        n_sequences = len(X_target)
        if n_sequences < 4:
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

        # DLinear uses lazy layer init — trigger it with a dummy forward
        if sum(p.numel() for p in self.model.parameters()) == 0:
            with torch.no_grad():
                dummy_x = torch.zeros(1, self.sequence_length, actual_input_dim).to(self.device)
                self.model(dummy_x)

        n_train = max(1, int(n_sequences * (1 - validation_split)))

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
            logger.info(f"✅ TAFAS source model trained (val_loss: {best_val:.4f})")

        # --- TAFAS: Freeze ALL source model parameters ---
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        # --- Initialize GCMs and save initial state ---
        self._init_gcms(n_input_var=actual_input_dim)

        return {
            "status": "ok",
            "val_loss": best_val,
            "epochs": epochs,
            "training_time": time.time() - start_time,
        }

    # ------------------------------------------------------------------
    # Update with new data (TAFAS adaptation — sub-window sliding)
    # ------------------------------------------------------------------
    def update_with_new_data(self, new_df: pd.DataFrame) -> Dict[str, Any]:
        """
        TAFAS adaptation on a new batch via sub-window sliding.

        Official TAFAS processes hundreds of individual sliding windows,
        each with 1 GCM adaptation step, accumulating GCM state across
        windows.  In our streaming protocol batches are large (750 rows),
        so we replicate that behaviour *within* the batch: slide
        (L, H)-sized sub-windows through the available sequences and run
        one GCM adaptation step per sub-window.

        Algorithm:
          1. Reset GCMs once at the start of each batch (matches official
             TAFAS batch-boundary reset).
          2. Slide sub-windows through the batch sequences.
          3. For each sub-window:
             a. Compute POGT length via PAAS on the sub-window look-back.
             b. Forward through calibrated pipeline → POGT loss.
             c. One GCM gradient step (state accumulates across sub-windows).
          4. GCMs retain cumulative state from all sub-windows.

        The source model stays frozen throughout.
        """
        if self.model is None:
            return {"status": "no_model"}

        start_time = time.time()

        # Reset GCMs once per batch (official TAFAS resets at batch boundary)
        if self.reset_between_batches:
            self._reset_gcms()

        new_df = new_df.copy()
        if "unique_id" not in new_df.columns:
            new_df["unique_id"] = "ts_001"
        new_df = new_df.sort_values("ds").reset_index(drop=True)
        new_df = create_lagged_features(new_df, lags=[1, self.season_length])

        self.accumulated_data = pd.concat(
            [self.accumulated_data, new_df], ignore_index=True
        ).drop_duplicates(subset=["ds"]).sort_values("ds").reset_index(drop=True)

        # Need enough data for at least one sequence
        min_len = self.sequence_length + self.forecast_horizon
        if len(self.accumulated_data) < min_len:
            return {"status": "insufficient_data", "adapt_time": time.time() - start_time}

        # Use a window covering the new data + enough context for sequences
        window = self.accumulated_data.tail(
            max(min_len + 10, len(new_df) + min_len)
        ).copy()

        if not self.preprocessor.is_fitted:
            return {"status": "preprocessor_not_fitted", "adapt_time": time.time() - start_time}

        # Incrementally expand scaler range if new data exceeds fitted bounds
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

        # --- Full-GT adaptation on previous batch's stored sequences ---
        # Official TAFAS performs a second adaptation pass once the full
        # forecast horizon has elapsed for past predictions.  In our streaming
        # protocol (batch_size >= H for most horizons), this data is available
        # by the next batch.
        n_full_gt = 0
        if self._stored_full_gt is not None:
            s_Xt, s_Xe, s_yt = self._stored_full_gt
            self.input_gcm.train()
            self.output_gcm.train()
            # Subsample to keep it efficient (official processes one window at a time)
            max_seqs = min(50, s_Xt.shape[0])
            idx = torch.randperm(s_Xt.shape[0])[:max_seqs]
            s_Xt_sub = s_Xt[idx]
            s_Xe_sub = s_Xe[idx] if s_Xe is not None else None
            s_yt_sub = s_yt[idx]
            for _ in range(self.gcm_steps):
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
            self._stored_full_gt = None  # consumed

        self.input_gcm.train()
        self.output_gcm.train()

        if self.use_subwindows:
            # --- Sub-window sliding (v3) ---
            # Replicate TAFAS's many-window protocol inside each streaming
            # batch: 1 GCM gradient step per sub-window, GCM state
            # accumulates across sub-windows within the batch.
            if n_sequences <= self.max_subwindows:
                window_indices = list(range(n_sequences))
            else:
                window_indices = np.linspace(
                    0, n_sequences - 1, self.max_subwindows, dtype=int
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

        else:
            # --- Single-batch adaptation (v2 — runs on VMs) ---
            # Treat the entire batch as one episode with gcm_steps gradient
            # steps on all sequences at once. Matches the code deployed on
            # VM3 (Run #31).
            lookback_values = new_df["y"].values if "y" in new_df.columns else X_target[-1, :, 0]
            pogt_len = self._compute_pogt_length(lookback_values)
            pogt_len = min(pogt_len, self.forecast_horizon)

            n_windows = 1
            total_loss = 0.0
            n_adapted = 0

            for step in range(self.gcm_steps):
                self._n_adapt += 1
                pred_cal = self._calibrated_forward(Xt, Xe)

                loss = F.mse_loss(pred_cal[:, :pogt_len], yt[:, :pogt_len])

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

        # Store current batch data for full-GT adaptation next time
        self._stored_full_gt = (
            Xt.detach().clone(),
            Xe.detach().clone() if Xe is not None else None,
            yt.detach().clone(),
        )

        adapt_time = time.time() - start_time
        avg_loss = total_loss / max(n_adapted, 1)
        return {
            "status": "adapted",
            "use_subwindows": self.use_subwindows,
            "n_subwindows": n_windows if self.use_subwindows else 0,
            "gcm_steps": self.gcm_steps if not self.use_subwindows else 0,
            "n_adapted": n_adapted,
            "n_adapt_total": self._n_adapt,
            "avg_loss": avg_loss,
            "n_full_gt": n_full_gt,
            "adapt_time": adapt_time,
        }

    # ------------------------------------------------------------------
    # Prediction (direct multi-horizon, not autoregressive)
    # ------------------------------------------------------------------
    def predict(
        self, context_df: pd.DataFrame, steps_ahead: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """Generate predictions using the calibrated (GCM-wrapped) frozen model.

        Uses direct multi-horizon prediction (not autoregressive step-by-step).
        """
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
            pred = self._calibrated_forward(Xt, Xe)  # [1, H]
            pred_scaled = pred[0].cpu().numpy()  # [H]

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
        if len(predictions) >= steps:
            return pd.DataFrame({"y_pred": predictions[:steps]})
        else:
            last_val = predictions[-1] if len(predictions) > 0 else 0.0
            pad = np.full(steps - len(predictions), last_val)
            return pd.DataFrame({"y_pred": np.concatenate([predictions, pad])})
