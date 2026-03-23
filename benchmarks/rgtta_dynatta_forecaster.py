"""
Regime-Guided DynaTTA (RGTTA+DynaTTA) Forecaster — v2
======================================================

Combines RGTTA's proactive regime detection with DynaTTA's reactive dynamic
learning-rate mechanism:

  - **RGTTA v2** (proactive): Distributional similarity → checkpoint gate +
    loss-driven early stopping.
  - **DynaTTA** (reactive): Prediction-error z-score + embedding drift →
    sigmoid → continuous LR within [alpha_min, alpha_max].

v2 design:
  - Checkpoint loading via strict loss gate (sim >= 0.75 AND ckpt_loss < current * 0.70).
  - DynaTTA's sigmoid LR replaces similarity-based smooth LR (reactive > proactive for LR).
  - Step count determined by loss-convergence early stopping (patience/epsilon).
  - EWC regularization optionally applied.

This is *our* enhanced contribution — it shows regime-awareness is a composable
meta-controller that improves *any* TTA strategy (fixed-LR, EWC, or DynaTTA).

Interface matches TTAForecaster / EWCForecaster / RGTTAForecaster / DynaTTAForecaster
for the unified benchmark runner.
"""

import logging
import math
import time
from collections import deque
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regime memory  (same as RGTTA — stores (features, state_dict) pairs)
# ---------------------------------------------------------------------------
class _RegimeMemory:
    """Stores (features, raw_values, model_state) for regime-guided checkpoint retrieval.

    Uses ensemble of 4 similarity methods:
      1. KS statistic (CDF shape distance)           — w=0.3
      2. Wasserstein-1 distance (earth-mover's dist)  — w=0.3
      3. Feature vector distance (summary statistics)  — w=0.2
      4. Variance ratio (std_min / std_max)            — w=0.2
    See paper/main.tex for design rationale.
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
        self._entries.append(
            {
                "features": features.flatten().copy(),
                "state_dict": {k: v.cpu().clone() for k, v in state_dict.items()},
                "raw_values": raw_values.copy() if raw_values is not None else None,
            }
        )
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

        if q_raw is None or s_raw is None or len(q_raw) < 5 or len(s_raw) < 5:
            return feat_sim

        # --- 2. KS statistic ---
        try:
            ks_stat, _ = ks_2samp(q_raw, s_raw)
            ks_sim = 1.0 - ks_stat
        except Exception:
            ks_sim = feat_sim

        # --- 3. Wasserstein-1 distance ---
        try:
            w_dist = wasserstein_distance(q_raw, s_raw)
            combined_range = max(np.ptp(q_raw), np.ptp(s_raw), 1e-8)
            w_sim = 1.0 / (1.0 + w_dist / combined_range)
        except Exception:
            w_sim = feat_sim

        # --- 4. Variance ratio (catches volatility regime shifts) ---
        try:
            std_q = float(np.std(q_raw))
            std_s = float(np.std(s_raw))
            var_sim = min(std_q, std_s) / (max(std_q, std_s) + 1e-8)
        except Exception:
            var_sim = 1.0  # fallback: assume same variance

        return float(self.w_ks * ks_sim + self.w_wass * w_sim + self.w_feat * feat_sim + self.w_var * var_sim)

    def query(
        self,
        features: np.ndarray,
        raw_values: Optional[np.ndarray] = None,
    ) -> Tuple[float, Optional[Dict[str, Any]]]:
        if not self._entries:
            return 0.0, None
        q = features.flatten()
        best_sim = -1.0
        best_state = None
        for entry in self._entries:
            sim = self._compute_similarity(
                q,
                entry["features"],
                raw_values,
                entry.get("raw_values"),
            )
            if sim > best_sim:
                best_sim = sim
                best_state = entry["state_dict"]
        return best_sim, best_state

    def __len__(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# Distribution features  (same 5-d vector as RGTTA)
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
# RGTTA+DynaTTA Forecaster
# ===========================================================================
class RGTTADynaTTAForecaster:
    """
    RGTTA+DynaTTA v2: regime-guided checkpoint gate + DynaTTA dynamic LR + early stopping.

    Parameters
    ----------
    alpha_min / alpha_max : float
        DynaTTA learning rate range.
    kappa : float
        Sigmoid steepness for DynaTTA LR mapping.
    eta : float
        EMA smoothing for DynaTTA LR update.
    max_steps : int
        Maximum gradient steps per batch.
    min_steps : int
        Minimum steps before early stopping.
    patience : int
        Early stopping patience.
    epsilon : float
        Minimum relative improvement threshold.
    ckpt_gate : float
        Checkpoint loss gate factor (0.70 = must be 30% better).
    use_ewc : bool
        Add EWC penalty during adaptation.
    ewc_lambda : float
        EWC penalty weight.
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
        # DynaTTA LR range (single range, not per-tier)
        alpha_min: float = 1e-4,
        alpha_max: float = 1e-3,
        # DynaTTA sigmoid parameters
        kappa: float = 1.0,
        eta: float = 0.1,
        eps: float = 1e-6,
        mse_buffer_size: int = 256,
        metric_history_size: int = 256,
        warmup_factor: int = 1,
        # v2 early stopping parameters
        max_steps: int = 25,
        min_steps: int = 5,
        patience: int = 3,
        epsilon: float = 0.005,
        ckpt_gate: float = 0.70,
        # Optional EWC
        use_ewc: bool = False,
        ewc_lambda: float = 400.0,
        # Model selection
        model_class: type = None,
        model_kwargs: Optional[Dict] = None,
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

        # DynaTTA LR range
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        # DynaTTA parameters
        self.kappa = kappa
        self.eta = eta
        self.eps = eps
        self.warmup_factor = warmup_factor

        # v2 early stopping
        self.max_steps = max_steps
        self.min_steps = min_steps
        self.patience = patience
        self.epsilon = epsilon
        self.ckpt_gate = ckpt_gate

        # EWC option
        self.use_ewc = use_ewc
        self.ewc_lambda = ewc_lambda

        # State
        self.model: Optional[nn.Module] = None
        self.preprocessor = DataPreprocessor()
        self.exog_cols: List[str] = []
        self.accumulated_data: Optional[pd.DataFrame] = None
        self._memory = _RegimeMemory()

        # DynaTTA buffers
        self._mse_buffer: deque = deque(maxlen=mse_buffer_size)
        self._metric_hist: List[deque] = [deque(maxlen=metric_history_size) for _ in range(3)]
        self._alpha_t: float = alpha_min  # start conservative
        self._n_adapt: int = 0
        self._warmup_steps: int = 0
        self._lr_history: List[float] = []
        # RTAB / RDB for proper embedding-distance metrics
        self._rtab: Dict[int, List] = {}  # sid -> [emb, mse, alpha]
        self._rdb: Dict[int, List] = {}  # sid -> [emb, mse]
        self._rtab_size: int = 360
        self._rdb_size: int = 100
        self._sample_counter: int = 0

        # EWC state
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
        """Freeze all layers except the output projection.

        Supports multiple model architectures:
          - GRU / iTransformer / LargeGRU: ``output_projection``
          - PatchTST: ``_head`` (nn.Sequential containing nn.Linear)
          - DLinear: ``_linear_seasonal``, ``_linear_trend``

        Returns trainable param count.
        """
        if self.model is None:
            return 0
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

    # ------------------------------------------------------------------
    # Embedding extraction (for DynaTTA shift metrics)
    # ------------------------------------------------------------------
    def _extract_embedding(self, Xt: torch.Tensor, Xe: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract hidden-state embedding [B, embed_dim] from the model.

        Handles all 5 architectures: GRU, PatchTST, iTransformer, DLinear.
        """
        self.model.eval()
        with torch.no_grad():
            if Xe is not None:
                x = torch.cat([Xt, Xe], dim=-1)
            else:
                x = Xt

            # --- PatchTST: channel-independent patched Transformer ---
            if hasattr(self.model, "patch_embed"):
                batch_size, seq_len, n_vars = x.shape
                patch_len = getattr(self.model, "patch_len", 16)
                if seq_len < patch_len:
                    x = F.pad(x, (0, 0, patch_len - seq_len, 0), value=0.0)
                    seq_len = x.shape[1]
                x_flat = x.permute(0, 2, 1).reshape(batch_size * n_vars, seq_len)
                x_patched = self.model.patch_embed(x_flat)
                if hasattr(self.model, "encoder"):
                    x_enc = self.model.encoder(x_patched)
                else:
                    x_enc = x_patched
                pooled = x_enc.mean(dim=1)
                pooled = pooled.reshape(batch_size, n_vars, -1).mean(dim=1)
                return pooled.detach()

            # --- GRU / LSTM: project then encode ---
            if hasattr(self.model, "input_projection"):
                x = self.model.input_projection(x)
            elif hasattr(self.model, "input_proj"):
                x = self.model.input_proj(x)
            if hasattr(self.model, "gru"):
                out, _ = self.model.gru(x)
            elif hasattr(self.model, "lstm"):
                out, _ = self.model.lstm(x)
            elif hasattr(self.model, "encoder"):
                out = self.model.encoder(x)
                if isinstance(out, tuple):
                    out = out[0]
            else:
                out = x
            return out[:, -1, :].detach()

    # ------------------------------------------------------------------
    # RTAB / RDB buffer management (same as vanilla DynaTTA)
    # ------------------------------------------------------------------
    def _dist_rtab(self, Xt: torch.Tensor, Xe: Optional[torch.Tensor]) -> float:
        """L2 distance between current embedding and weighted avg of RTAB."""
        if not self._rtab:
            return 0.0
        embs, mses, alps = [], [], []
        for entry in self._rtab.values():
            embs.append(entry[0])
            mses.append(entry[1])
            alps.append(entry[2])
        inv = np.array([alp / (m + self.eps) for m, alp in zip(mses, alps)], dtype=float)
        w = inv / (inv.sum() + self.eps)
        stack = torch.stack(embs, 0).to(self.device)
        w_tensor = torch.from_numpy(w).float().to(self.device).unsqueeze(-1)
        avg = (stack * w_tensor).sum(0)
        cur = self._extract_embedding(Xt, Xe).mean(0)
        return float(torch.norm(cur - avg, p=2).item())

    def _dist_rdb(self, Xt: torch.Tensor, Xe: Optional[torch.Tensor]) -> float:
        """L2 distance between current embedding and weighted avg of RDB."""
        if not self._rdb:
            return 0.0
        embs, mses = [], []
        for entry in self._rdb.values():
            embs.append(entry[0])
            mses.append(entry[1])
        inv = np.array([1.0 / (m + self.eps) for m in mses], dtype=float)
        w = inv / (inv.sum() + self.eps)
        stack = torch.stack(embs, 0).to(self.device)
        w_tensor = torch.from_numpy(w).float().to(self.device).unsqueeze(-1)
        avg = (stack * w_tensor).sum(0)
        cur = self._extract_embedding(Xt, Xe).mean(0)
        return float(torch.norm(cur - avg, p=2).item())

    def _update_rtab(self, sid: int, emb: torch.Tensor, mse: float, alpha: float = 1.0) -> None:
        self._rtab[sid] = [emb.detach().cpu(), mse, alpha]
        if len(self._rtab) > self._rtab_size:
            oldest = min(self._rtab.keys())
            del self._rtab[oldest]

    def _update_rdb(self, sid: int, emb: torch.Tensor, mse: float) -> None:
        if sid in self._rdb:
            if mse < self._rdb[sid][1]:
                self._rdb[sid] = [emb.detach().cpu(), mse]
        else:
            if len(self._rdb) < self._rdb_size:
                self._rdb[sid] = [emb.detach().cpu(), mse]
            else:
                worst = max(self._rdb.items(), key=lambda x: x[1][1])[0]
                if mse < self._rdb[worst][1]:
                    del self._rdb[worst]
                    self._rdb[sid] = [emb.detach().cpu(), mse]

    def _update_buffers(self, Xt: torch.Tensor, Xe: Optional[torch.Tensor], yt: torch.Tensor) -> None:
        """Update RTAB and RDB with current batch embeddings and errors."""
        self.model.eval()
        with torch.no_grad():
            pred = self.model(Xt, Xe)
            batch_mse = F.mse_loss(pred, yt).item()
        emb = self._extract_embedding(Xt, Xe).mean(0)
        sid = self._sample_counter
        self._update_rtab(sid, emb, batch_mse, alpha=float(self._alpha_t))
        self._update_rdb(sid, emb, batch_mse)

    # ------------------------------------------------------------------
    # DynaTTA shift metrics (proper 3-metric computation)
    # ------------------------------------------------------------------
    def _compute_shift_metrics(
        self, Xt: torch.Tensor, Xe: Optional[torch.Tensor], yt: torch.Tensor
    ) -> Tuple[float, float, float]:
        """Prediction-error z-score + RTAB/RDB embedding distances."""
        self.model.eval()
        with torch.no_grad():
            pred = self.model(Xt, Xe)
            batch_mse = F.mse_loss(pred, yt).item()

        self._mse_buffer.append(batch_mse)
        if len(self._mse_buffer) < 2:
            z = 0.0
        else:
            mu = np.mean(self._mse_buffer)
            sigma = np.std(self._mse_buffer)
            z = (batch_mse - mu) / (sigma + self.eps)

        # Proper RTAB/RDB distances (same as vanilla DynaTTA)
        dr = self._dist_rtab(Xt, Xe)
        dp = self._dist_rdb(Xt, Xe)

        return z, dr, dp

    # ------------------------------------------------------------------
    # DynaTTA dynamic LR
    # ------------------------------------------------------------------
    def _compute_dynamic_lr(self, z: float, dr: float, dp: float) -> float:
        """Compute DynaTTA sigmoid LR within [alpha_min, alpha_max]."""
        alpha_min, alpha_max = self.alpha_min, self.alpha_max

        # Z-normalise each metric
        norms = []
        for i, m in enumerate([z, dr, dp]):
            hist = self._metric_hist[i]
            hist.append(m)
            mu = np.mean(hist)
            sd = np.std(hist)
            norms.append((m - mu) / (sd + self.eps))
        S = sum(norms)

        # Sigmoid mapping to [alpha_min, alpha_max] (same formula as vanilla DynaTTA)
        lam = 1 + (alpha_max / (alpha_min + self.eps) - 1) / (1 + math.exp(-self.kappa * S))
        gamma = min(1.0, self._n_adapt / (self._warmup_steps + self.eps))
        alpha_tgt = alpha_min * (1 + gamma * (lam - 1))

        # EMA smoothing
        self._alpha_t += self.eta * (alpha_tgt - self._alpha_t)
        # Clamp to range
        self._alpha_t = max(alpha_min, min(alpha_max, self._alpha_t))
        self._lr_history.append(float(self._alpha_t))

        return float(self._alpha_t)

    # ------------------------------------------------------------------
    # EWC helpers (same as RGTTA)
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
    # fit  (initial training — same as all other forecasters)
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

        self._scaled_feature_cols = None
        if self.feature_cols:
            self._scaled_feature_cols = [
                f"{c}_scaled" for c in self.feature_cols if f"{c}_scaled" in data_scaled.columns
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
        if n_seq < 2:
            return {"status": "skipped", "reason": "insufficient_sequences", "training_time": time.time() - start_time}

        actual_input_dim = X_target.shape[2] if X_target.ndim == 3 else self.input_dim
        n_exog = X_exog.shape[2] if X_exog is not None else 0
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
        init_raw = df["y"].dropna().values.astype(np.float64)
        init_features = _multivariate_distribution_features(df, self.feature_cols)
        self._memory.store(init_features, self.model.state_dict(), raw_values=init_raw)

        # Initialise DynaTTA state
        # Warmup based on adaptation budget, not forecast horizon.
        # 3 × max_steps ≈ 3 batches of warmup — enough for metric history
        # to stabilise, regardless of horizon.
        self._warmup_steps = self.warmup_factor * self.max_steps * 3
        self._alpha_t = self.alpha_min  # start conservative
        self._n_adapt = 0
        self._sample_counter = 0

        # Seed the MSE buffer
        self.model.eval()
        with torch.no_grad():
            train_pred = self.model(Xt[:n_train], Xe[:n_train] if Xe is not None else None)
            train_mse = F.mse_loss(train_pred, yt[:n_train]).item()
        self._mse_buffer.append(train_mse)

        # Seed RTAB/RDB with initial training embedding
        emb = self._extract_embedding(Xt[:n_train], Xe[:n_train] if Xe is not None else None).mean(0)
        self._update_rtab(0, emb, train_mse)
        self._update_rdb(0, emb, train_mse)

        # Compute initial Fisher + anchor only when EWC is enabled.
        # Fisher costs 200 forward+backward passes — skip for plain RGTTA+DynaTTA.
        if self.use_ewc:
            self.model.eval()
            self._fisher = self._compute_fisher(Xt, Xe, yt)
            self._anchor_params = {n: p.data.clone() for n, p in self.model.named_parameters() if p.requires_grad}

        return {
            "status": "completed",
            "training_time": time.time() - start_time,
            "trained_from_scratch": True,
            "n_sequences": n_seq,
        }

    # ------------------------------------------------------------------
    # update_with_new_data  (v2: checkpoint gate + DynaTTA LR + early stopping)
    # ------------------------------------------------------------------
    def update_with_new_data(self, new_df: pd.DataFrame) -> Dict[str, Any]:
        """
        RGTTA+DynaTTA v2 update:
        1. Measure current model loss on new batch.
        2. Checkpoint gate: load if sim >= 0.75 AND ckpt_loss < current * ckpt_gate.
        3. Compute DynaTTA shift metrics → sigmoid dynamic LR.
        4. Early-stopping gradient loop (min_steps..max_steps, patience).
        5. Optional EWC penalty.
        6. Store updated model in regime memory.
        """
        if self.model is None:
            return {"status": "skipped", "reason": "no_model", "rgtta_dynatta_time": 0.0}

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
            return {"status": "skipped", "reason": "insufficient_data", "rgtta_dynatta_time": time.time() - start_time}

        # --- Regime detection ---
        raw_vals = new_df["y"].dropna().values.astype(np.float64)
        new_features = _multivariate_distribution_features(new_df, self.feature_cols)
        best_sim, best_state = self._memory.query(new_features, raw_values=raw_vals)
        self._last_similarity = best_sim

        # --- Prepare sequences ---
        window = self.accumulated_data.tail(max(min_len + 10, len(new_df) + min_len)).copy()

        if not self.preprocessor.is_fitted:
            return {
                "status": "skipped",
                "reason": "preprocessor_not_fitted",
                "rgtta_dynatta_time": time.time() - start_time,
            }

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
            return {"status": "skipped", "reason": "no_sequences", "rgtta_dynatta_time": time.time() - start_time}

        X_target = np.clip(X_target, -5, 5)
        y = np.clip(y, -5, 5)
        if X_exog is not None:
            X_exog = np.clip(X_exog, -5, 5)

        Xt = torch.FloatTensor(X_target).to(self.device)
        yt = torch.FloatTensor(y).to(self.device)
        Xe = torch.FloatTensor(X_exog).to(self.device) if X_exog is not None else None
        Xt = torch.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)
        yt = torch.nan_to_num(yt, nan=0.0, posinf=0.0, neginf=0.0)

        self._sample_counter += 1

        # --- Measure current model loss ---
        self.model.eval()
        with torch.no_grad():
            current_pred = self.model(Xt, Xe)
            current_loss = regime_aware_loss(yt, current_pred).item()

        # --- Checkpoint gate (v2: sim >= 0.75 AND loss improvement > ckpt_gate) ---
        loaded_checkpoint = False
        tier_label = "mid"  # default diagnostic label
        if best_sim >= 0.75 and best_state is not None:
            try:
                saved_state = self.model.state_dict()
                self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
                with torch.no_grad():
                    ckpt_pred = self.model(Xt, Xe)
                    ckpt_loss = regime_aware_loss(yt, ckpt_pred).item()

                if (
                    not math.isnan(ckpt_loss)
                    and not math.isnan(current_loss)
                    and ckpt_loss < current_loss * self.ckpt_gate
                ):
                    loaded_checkpoint = True
                    tier_label = "ckpt"
                    current_loss = ckpt_loss  # update baseline for early stopping
                    logger.info(
                        f"🔄 RGTTA+DynaTTA: checkpoint LOADED "
                        f"(ckpt={ckpt_loss:.4f} < current*{self.ckpt_gate}={current_loss:.4f}, "
                        f"sim={best_sim:.3f})"
                    )
                    if self.use_ewc:
                        self._anchor_params = {
                            n: p.data.clone() for n, p in self.model.named_parameters() if p.requires_grad
                        }
                else:
                    self.model.load_state_dict(saved_state)
            except Exception as e:
                logger.debug(f"Checkpoint gate failed: {e}")

        if not loaded_checkpoint:
            # Classify for diagnostics only
            if best_sim >= 0.75:
                tier_label = "easy"
            elif best_sim >= 0.55:
                tier_label = "mid"
            else:
                tier_label = "hard"

        self._last_loaded_ckpt = loaded_checkpoint

        # --- Compute DynaTTA shift metrics → dynamic LR ---
        z, dr, dp = self._compute_shift_metrics(Xt, Xe, yt)
        dynamic_lr = self._compute_dynamic_lr(z, dr, dp)
        self._last_lr = dynamic_lr

        # --- Apply frozen backbone if enabled ---
        if self.freeze_backbone:
            self._freeze_backbone()

        # --- Early-stopping gradient loop ---
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=dynamic_lr, weight_decay=1e-5, eps=1e-8
        )
        self.model.train()

        best_loss = current_loss
        no_improve = 0
        steps_done = 0

        for step in range(self.max_steps):
            self._n_adapt += 1
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
            steps_done += 1

            # Early stopping check (after min_steps)
            step_loss = task_loss.item()
            if step >= self.min_steps:
                if best_loss > 0:
                    rel_improve = (best_loss - step_loss) / (abs(best_loss) + 1e-8)
                else:
                    rel_improve = 0.0
                if step_loss < best_loss:
                    best_loss = step_loss
                    no_improve = 0
                elif rel_improve < self.epsilon:
                    no_improve += 1
                    if no_improve >= self.patience:
                        break
                else:
                    no_improve = 0
            else:
                if step_loss < best_loss:
                    best_loss = step_loss

        self._last_steps_used = steps_done
        self._total_steps_all_batches += steps_done
        self._batch_count += 1

        # --- Unfreeze after adaptation ---
        if self.freeze_backbone:
            self._unfreeze_all()

        # --- Update RTAB/RDB with post-adaptation embeddings ---
        self._update_buffers(Xt, Xe, yt)

        # --- Store updated model in regime memory ---
        self._memory.store(new_features, self.model.state_dict(), raw_values=raw_vals)

        # Refresh Fisher + anchor when EWC is enabled
        if self.use_ewc:
            self.model.eval()
            new_fisher = self._compute_fisher(Xt, Xe, yt)
            if self._fisher is not None:
                for n in self._fisher:
                    if n in new_fisher:
                        self._fisher[n] = 0.5 * self._fisher[n] + 0.5 * new_fisher[n]
            else:
                self._fisher = new_fisher
            self._anchor_params = {n: p.data.clone() for n, p in self.model.named_parameters() if p.requires_grad}

        rgtta_dynatta_time = time.time() - start_time
        logger.info(
            f"{'🔄' if loaded_checkpoint else '🔧'} RGTTA+DynaTTA: "
            f"tier={tier_label}, sim={best_sim:.3f}, lr={dynamic_lr:.6f}, "
            f"steps={steps_done}/{self.max_steps}, ckpt={'yes' if loaded_checkpoint else 'no'}, "
            f"ewc={self.use_ewc}"
        )
        return {
            "status": "completed",
            "rgtta_dynatta_time": rgtta_dynatta_time,
            "n_sequences": len(X_target),
            "tier": tier_label,
            "similarity": best_sim,
            "dynamic_lr": dynamic_lr,
            "z_score": z,
            "steps_used": steps_done,
            "loaded_checkpoint": loaded_checkpoint,
            "frozen_backbone": self.freeze_backbone,
        }

    # ------------------------------------------------------------------
    # predict (same interface as all other forecasters)
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
            seq = (
                vals[-self.sequence_length :]
                if len(vals) >= self.sequence_length
                else np.concatenate(
                    [
                        np.full(self.sequence_length - len(vals), float(vals[0]) if len(vals) > 0 else 0.0),
                        vals,
                    ]
                )
            )
            X_target_seq = np.array([seq.reshape(-1, 1)], dtype=np.float64)
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

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def get_tier_stats(self) -> Dict[str, Any]:
        """Return v2 diagnostics."""
        return {
            "total_batches": self._batch_count,
            "total_steps": self._total_steps_all_batches,
            "avg_steps_per_batch": (self._total_steps_all_batches / max(1, self._batch_count)),
            "last_similarity": self._last_similarity,
            "last_steps": self._last_steps_used,
            "last_lr": self._last_lr,
            "last_loaded_ckpt": self._last_loaded_ckpt,
        }

    def get_executed_tier_stats(self) -> Dict[str, Any]:
        """Alias for backward compat — same as get_tier_stats in v2."""
        return self.get_tier_stats()

    def get_lr_history(self) -> List[float]:
        """Return the history of dynamic learning rates."""
        return list(self._lr_history)
