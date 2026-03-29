"""
DynaTTA-style Forecaster
========================

Faithful re-implementation of the DynaTTA dynamic adaptation rate mechanism
(Grover & Etemad, ICML 2025 Workshop) adapted to our GRU-based models and
incremental batch pipeline.

DynaTTA's core idea: instead of fixed-K steps at a fixed LR, it dynamically
adjusts the learning rate based on three shift signals:
  1. Prediction error z-score (how surprising is the current batch?)
  2. Embedding distance to a Recent Table (RTAB) of past embeddings
  3. Embedding distance to a Representative Distribution Buffer (RDB)

These three metrics are z-normalised, summed into a composite shift score S,
and then fed through a sigmoid to compute a dynamic adaptation rate α_t:

    λ = 1 + (α_max/α_min − 1) / (1 + exp(−κ·S))
    γ = min(1, n_adapt / warmup_steps)          # warmup factor
    α_target = α_min · (1 + γ·(λ − 1))
    α_t ← α_t + η · (α_target − α_t)           # EMA smoothing

This is *not* our contribution — it is a baseline for comparison. Our RGTTA
differs from DynaTTA in:
  - Distributional feature matching (proactive) vs prediction-error (reactive)
  - Checkpoint memory with tiered reuse vs no checkpoint memory
  - Discrete tier selection vs continuous LR modulation
  - Optional EWC regularisation vs none

Interface matches TTAForecaster / EWCForecaster / RGTTAForecaster for the
unified benchmark runner.

Reference:
    Grover, S. & Etemad, A. (2025). "Shift-Aware Test Time Adaptation and
    Benchmarking for Time-Series Forecasting." ICML 2025 Workshop (Oral).
"""

import copy
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

from regime_forecasting.models.transformer import TimeSeriesTransformer, regime_aware_loss
from regime_forecasting.utils.data_utils import (
    DataPreprocessor,
    create_lagged_features,
    prepare_sequences,
)

logger = logging.getLogger(__name__)


class DynaTTAForecaster:
    """
    DynaTTA-style dynamic adaptation rate for test-time adaptation.

    Tracks prediction error z-scores and embedding drift to continuously
    adjust the learning rate between [alpha_min, alpha_max].

    Parameters
    ----------
    alpha_min / alpha_max : float
        Learning rate range for the dynamic adaptation rate.
    kappa : float
        Sensitivity scale for the sigmoid mapping (higher = more reactive).
    eta : float
        EMA smoothing factor for the adaptation rate update.
    mse_buffer_size : int
        Size of the rolling MSE buffer for z-score computation.
    metric_history_size : int
        Size of the per-metric history for z-normalisation.
    warmup_factor : int
        Warmup multiplied by tta_steps * 3 to get warmup_steps (~3 batches).
    tta_steps : int
        Number of gradient steps per batch (fixed, unlike LR which is dynamic).
    rtab_size : int
        Max entries in the Recent Table (RTAB).
    rdb_size : int
        Max entries in the Representative Distribution Buffer (RDB).
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
        # DynaTTA-specific hyperparameters (from their paper defaults)
        alpha_min: float = 1e-4,
        alpha_max: float = 1e-3,
        kappa: float = 1.0,
        eta: float = 0.1,
        eps: float = 1e-6,
        mse_buffer_size: int = 256,
        metric_history_size: int = 256,
        warmup_factor: int = 1,
        tta_steps: int = 20,
        rtab_size: int = 360,
        rdb_size: int = 100,
        # Model selection
        model_class: type = None,
        model_kwargs: Optional[Dict] = None,
        # Multivariate support
        input_dim: int = 1,
        feature_cols: Optional[list] = None,
        # Frozen backbone (faster, prevents overfitting)
        freeze_backbone: bool = True,
        # Streaming-mode tuning: fast EMA + dense buffer seeding from training
        # data. Designed for the 10-batch streaming protocol where the standard
        # sliding-window defaults (eta=0.1, 1 seed MSE) leave the dynamic LR
        # 3× below TTA for the first 5 critical batches.
        streaming_mode: bool = False,
        # Adapter injection (D3 experiment)
        use_adapter: bool = False,
        adapter_bottleneck: int = 16,
        model_key: str = "",
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
        self.streaming_mode = streaming_mode
        self.use_adapter = use_adapter
        self.adapter_bottleneck = adapter_bottleneck
        self.model_key = model_key

        # DynaTTA dynamic LR parameters
        # streaming_mode overrides eta to 0.7: converges to alpha_target in ~3
        # batches instead of ~22 (needed when run has only 10 total batches).
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.kappa = kappa
        self.eta = 0.7 if streaming_mode else eta
        self.eps = eps
        self.tta_steps = tta_steps
        self.warmup_factor = warmup_factor

        # Buffer sizes
        self._mse_buffer_size = mse_buffer_size
        self._metric_history_size = metric_history_size
        self._rtab_size = rtab_size
        self._rdb_size = rdb_size

        # State
        self.model: Optional[nn.Module] = None
        self.preprocessor = DataPreprocessor()
        self.exog_cols: List[str] = []
        self.accumulated_data: Optional[pd.DataFrame] = None

        # DynaTTA buffers (initialised after fit)
        self._mse_buffer: deque = deque(maxlen=mse_buffer_size)
        # metric_hist[i] for i in {0: z-score, 1: dist_rtab, 2: dist_rdb}
        self._metric_hist: List[deque] = [
            deque(maxlen=metric_history_size) for _ in range(3)
        ]
        # RTAB: sample_id -> (embedding_tensor, mse, alpha)
        self._rtab: Dict[int, List] = {}
        # RDB:  sample_id -> (embedding_tensor, mse)
        self._rdb: Dict[int, List] = {}
        # Current adaptation rate
        self._alpha_t: float = alpha_min
        self._n_adapt: int = 0
        self._warmup_steps: int = 0  # set in fit()
        self._sample_counter: int = 0
        self._lr_history: List[float] = []

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
    # Embedding extraction (adapted for our GRU model)
    # ------------------------------------------------------------------
    def _extract_embedding(
        self, Xt: torch.Tensor, Xe: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Extract hidden-state embedding from the model's encoder.

        Handles all 5 model architectures:
          - GRU (gru_small, gru_large): input_projection → GRU → last hidden
          - PatchTST: patch_embed → encoder → pool across patches & variates
          - iTransformer / DLinear: fallback to pooled raw input

        Returns a tensor of shape [B, embed_dim].
        """
        self.model.eval()
        with torch.no_grad():
            if Xe is not None:
                x = torch.cat([Xt, Xe], dim=-1)  # [B, seq_len, total_dim]
            else:
                x = Xt  # [B, seq_len, input_dim]

            # --- PatchTST: channel-independent patched Transformer ---
            if hasattr(self.model, "patch_embed"):
                batch_size, seq_len, n_vars = x.shape
                patch_len = getattr(self.model, "patch_len", 16)
                if seq_len < patch_len:
                    x = F.pad(x, (0, 0, patch_len - seq_len, 0), value=0.0)
                    seq_len = x.shape[1]
                # Channel-independent reshape: [B, L, N] → [B*N, L]
                x_flat = x.permute(0, 2, 1).reshape(batch_size * n_vars, seq_len)
                x_patched = self.model.patch_embed(x_flat)  # [B*N, n_patches, D]
                if hasattr(self.model, "encoder"):
                    x_enc = self.model.encoder(x_patched)  # [B*N, n_patches, D]
                else:
                    x_enc = x_patched
                # Pool: [B*N, n_patches, D] → [B*N, D] → [B, D]
                pooled = x_enc.mean(dim=1)  # [B*N, D]
                pooled = pooled.reshape(batch_size, n_vars, -1).mean(dim=1)
                return pooled.detach()

            # --- GRU / LSTM models: project then encode ---
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
                # Fallback (iTransformer, DLinear): pool raw input
                out = x

            # Return last time-step: [B, embed_dim]
            return out[:, -1, :].detach()

    # ------------------------------------------------------------------
    # DynaTTA buffer management
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
        inv = np.array(
            [alp / (m + self.eps) for m, alp in zip(mses, alps)], dtype=float
        )
        w = inv / (inv.sum() + self.eps)
        stack = torch.stack(embs, 0).to(self.device)  # [N, hidden_dim]
        w_tensor = torch.from_numpy(w).float().to(self.device).unsqueeze(-1)  # [N, 1]
        avg = (stack * w_tensor).sum(0)  # [hidden_dim]
        cur = self._extract_embedding(Xt, Xe).mean(0)  # [hidden_dim]
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
        stack = torch.stack(embs, 0).to(self.device)  # [N, hidden_dim]
        w_tensor = torch.from_numpy(w).float().to(self.device).unsqueeze(-1)  # [N, 1]
        avg = (stack * w_tensor).sum(0)  # [hidden_dim]
        cur = self._extract_embedding(Xt, Xe).mean(0)  # [hidden_dim]
        return float(torch.norm(cur - avg, p=2).item())

    def _update_rtab(
        self, sid: int, emb: torch.Tensor, mse: float, alpha: float = 1.0
    ) -> None:
        """Update the Recent Table with a new entry."""
        self._rtab[sid] = [emb.detach().cpu(), mse, alpha]
        if len(self._rtab) > self._rtab_size:
            oldest = min(self._rtab.keys())
            del self._rtab[oldest]

    def _update_rdb(self, sid: int, emb: torch.Tensor, mse: float) -> None:
        """Update the Representative Distribution Buffer."""
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

    # ------------------------------------------------------------------
    # Dynamic adaptation rate (the core DynaTTA formula)
    # ------------------------------------------------------------------
    def _update_adaptation_rate(self, z: float, dr: float, dp: float) -> None:
        """Compute the dynamic adaptation rate α_t from the three shift metrics."""
        norms = []
        for i, m in enumerate([z, dr, dp]):
            hist = self._metric_hist[i]
            hist.append(m)
            mu = np.mean(hist)
            sd = np.std(hist)
            norms.append((m - mu) / (sd + self.eps))
        S = sum(norms)
        lam = 1 + (self.alpha_max / self.alpha_min - 1) / (
            1 + math.exp(-self.kappa * S)
        )
        # Warmup
        gamma = min(1.0, self._n_adapt / (self._warmup_steps + self.eps))
        alpha_tgt = self.alpha_min * (1 + gamma * (lam - 1))
        # EMA smoothing
        self._alpha_t += self.eta * (alpha_tgt - self._alpha_t)
        self._lr_history.append(float(self._alpha_t))

    def _compute_shift_metrics(
        self, Xt: torch.Tensor, Xe: Optional[torch.Tensor], yt: torch.Tensor
    ) -> Tuple[float, float, float]:
        """Compute the 3 shift metrics: prediction-error z-score + 2 embedding distances."""
        # 1. Prediction error z-score
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

        # 2. RTAB embedding distance
        dr = self._dist_rtab(Xt, Xe)

        # 3. RDB embedding distance
        dp = self._dist_rdb(Xt, Xe)

        return z, dr, dp

    def _update_buffers(
        self, Xt: torch.Tensor, Xe: Optional[torch.Tensor], yt: torch.Tensor
    ) -> None:
        """Update RTAB and RDB with current batch embeddings and errors."""
        self.model.eval()
        with torch.no_grad():
            pred = self.model(Xt, Xe)
            batch_mse = F.mse_loss(pred, yt).item()

        emb = self._extract_embedding(Xt, Xe).mean(0)  # [hidden_dim]
        sid = self._sample_counter
        self._update_rtab(sid, emb, batch_mse, alpha=1.0)
        self._update_rdb(sid, emb, batch_mse)

    # ------------------------------------------------------------------
    # fit (initial training — same interface as TTA/EWC/RGTTA)
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

        n_train = max(1, int(n_seq * (1 - validation_split)))
        train_idx = list(range(n_train))
        val_idx = list(range(n_train, n_seq)) if n_train < n_seq else []

        X_target = np.clip(X_target, -5, 5)
        y = np.clip(y, -5, 5)
        if X_exog is not None:
            X_exog = np.clip(X_exog, -5, 5)

        Xt = torch.FloatTensor(X_target).to(self.device)
        yt = torch.FloatTensor(y).to(self.device)
        Xe = (
            torch.FloatTensor(X_exog).to(self.device)
            if X_exog is not None
            else None
        )
        Xt = torch.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)
        yt = torch.nan_to_num(yt, nan=0.0, posinf=0.0, neginf=0.0)

        actual_lr = min(learning_rate, 0.0005)
        optimizer = optim.Adam(
            self.model.parameters(), lr=actual_lr, weight_decay=1e-5, eps=1e-8
        )
        best_val_loss = float("inf")
        best_state = None
        nan_count = 0

        self.model.train()
        for epoch in range(epochs):
            np.random.shuffle(train_idx)
            for i in range(0, len(train_idx), batch_size):
                idx = train_idx[i : i + batch_size]
                optimizer.zero_grad()
                pred = self.model(
                    Xt[idx], Xe[idx] if Xe is not None else None
                )
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
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )
                optimizer.step()

            if val_idx:
                self.model.eval()
                with torch.no_grad():
                    vp = self.model(
                        Xt[val_idx], Xe[val_idx] if Xe is not None else None
                    )
                    vl = regime_aware_loss(yt[val_idx], vp).item()
                has_nan = any(
                    torch.isnan(p).any() for p in self.model.parameters()
                )
                if vl < best_val_loss and not np.isnan(vl) and not has_nan:
                    best_val_loss = vl
                    best_state = {
                        k: v.cpu().clone()
                        for k, v in self.model.state_dict().items()
                    }
                self.model.train()

        if best_state is not None and not any(
            torch.isnan(v).any() for v in best_state.values()
        ):
            self.model.load_state_dict(best_state)

        # Initialise DynaTTA state after training
        # Batch protocol: n_adapt increments by tta_steps per batch (not per-sample
        # like sliding-window). Scale warmup to ~3 batches so the dynamic LR
        # becomes fully responsive early enough.
        self._warmup_steps = self.warmup_factor * self.tta_steps * 3
        self._alpha_t = self.alpha_min
        self._n_adapt = 0
        self._sample_counter = 0

        # Seed MSE buffer and RTAB/RDB with training data statistics.
        # streaming_mode: seed with per-sequence MSEs (mini-batch) so the
        # z-score normalisation at batch 1 uses meaningful statistics rather
        # than the degenerate 2-sample case that forces z=+1.414 always.
        self.model.eval()
        with torch.no_grad():
            if self.streaming_mode:
                # Dense seeding: per-sequence MSEs in mini-batches of 64
                _seed_bs = 64
                _all_mses: List[float] = []
                for _s in range(0, n_train, _seed_bs):
                    _e = min(_s + _seed_bs, n_train)
                    _bp = self.model(Xt[_s:_e], Xe[_s:_e] if Xe is not None else None)
                    # reduction='none' → [B, H]; mean over H → per-seq MSE [B]
                    _bmses = F.mse_loss(_bp, yt[_s:_e], reduction='none').mean(dim=1)
                    _all_mses.extend(_bmses.tolist())
                for _v in _all_mses:
                    self._mse_buffer.append(_v)
                    self._metric_hist[0].append(_v)  # seed z-score history
                train_mse = float(np.mean(_all_mses)) if _all_mses else 0.0
            else:
                train_pred = self.model(Xt[:n_train], Xe[:n_train] if Xe is not None else None)
                train_mse = F.mse_loss(train_pred, yt[:n_train]).item()
                self._mse_buffer.append(train_mse)

        # Store initial embedding in RTAB/RDB
        emb = self._extract_embedding(Xt[:n_train], Xe[:n_train] if Xe is not None else None).mean(0)  # [hidden_dim]
        self._update_rtab(0, emb, train_mse)
        self._update_rdb(0, emb, train_mse)

        return {
            "status": "completed",
            "training_time": time.time() - start_time,
            "trained_from_scratch": True,
            "n_sequences": n_seq,
        }

    # ------------------------------------------------------------------
    # update_with_new_data (DynaTTA core: dynamic LR adaptation)
    # ------------------------------------------------------------------
    def update_with_new_data(self, new_df: pd.DataFrame) -> Dict[str, Any]:
        """
        DynaTTA-style update:
        1. Compute prediction error on new batch → MSE z-score.
        2. Compute embedding distances to RTAB and RDB.
        3. Update dynamic adaptation rate α_t.
        4. Run TTA gradient steps at the computed α_t.
        5. Update RTAB/RDB with new embeddings.
        """
        if self.model is None:
            return {"status": "skipped", "reason": "no_model", "dynatta_time": 0.0}

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
            return {
                "status": "skipped",
                "reason": "insufficient_data",
                "dynatta_time": time.time() - start_time,
            }

        # --- Prepare sequences ---
        window = self.accumulated_data.tail(
            max(min_len + 10, len(new_df) + min_len)
        ).copy()

        if not self.preprocessor.is_fitted:
            return {
                "status": "skipped",
                "reason": "preprocessor_not_fitted",
                "dynatta_time": time.time() - start_time,
            }

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
            return {
                "status": "skipped",
                "reason": "no_sequences",
                "dynatta_time": time.time() - start_time,
            }

        X_target = np.clip(X_target, -5, 5)
        y = np.clip(y, -5, 5)
        if X_exog is not None:
            X_exog = np.clip(X_exog, -5, 5)

        Xt = torch.FloatTensor(X_target).to(self.device)
        yt = torch.FloatTensor(y).to(self.device)
        Xe = (
            torch.FloatTensor(X_exog).to(self.device)
            if X_exog is not None
            else None
        )
        Xt = torch.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)
        yt = torch.nan_to_num(yt, nan=0.0, posinf=0.0, neginf=0.0)

        self._sample_counter += 1

        # --- Step 1-3: Compute shift metrics → update adaptation rate ---
        z, dr, dp = self._compute_shift_metrics(Xt, Xe, yt)
        self._update_adaptation_rate(z, dr, dp)

        # --- Apply frozen backbone if enabled ---
        if self.freeze_backbone:
            self._freeze_backbone()

        # --- Step 4: Run TTA gradient steps at dynamic α_t ---
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=float(self._alpha_t),
            weight_decay=1e-5,
            eps=1e-8,
        )
        self.model.train()

        for _ in range(self.tta_steps):
            self._n_adapt += 1
            optimizer.zero_grad()
            pred = self.model(Xt, Xe)
            loss = regime_aware_loss(yt, pred)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )
            optimizer.step()

        # --- Unfreeze after adaptation ---
        if self.freeze_backbone:
            self._unfreeze_all()

        # --- Step 5: Update RTAB/RDB ---
        self._update_buffers(Xt, Xe, yt)

        dynatta_time = time.time() - start_time
        return {
            "status": "completed",
            "dynatta_time": dynatta_time,
            "n_sequences": len(X_target),
            "alpha_t": float(self._alpha_t),
            "z_score": z,
            "dist_rtab": dr,
            "dist_rdb": dp,
            "frozen_backbone": self.freeze_backbone,
        }

    # ------------------------------------------------------------------
    # predict (same interface as TTA/EWC/RGTTA)
    # ------------------------------------------------------------------
    def predict(self, context_df: pd.DataFrame, steps_ahead: int) -> pd.DataFrame:
        """Direct multi-horizon prediction (single forward pass, not autoregressive)."""
        if self.model is None:
            raise ValueError("Model not trained.")

        self.model.eval()
        context_df = context_df.copy()
        if "unique_id" not in context_df.columns:
            context_df["unique_id"] = "ts_001"
        context_df["y"] = (
            pd.to_numeric(context_df["y"], errors="coerce").astype(np.float64)
        )
        context_df = create_lagged_features(
            context_df, lags=[1, self.season_length]
        )
        for col in self.exog_cols:
            if col in context_df.columns:
                context_df[col] = (
                    pd.to_numeric(context_df[col], errors="coerce")
                    .fillna(0)
                    .astype(np.float64)
                )

        if not self.preprocessor.is_fitted:
            context_df, _ = self.preprocessor.fit_transform(
                context_df, "y", self.exog_cols
            )
        else:
            context_df = self.preprocessor.transform(
                context_df, "y", self.exog_cols
            )

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
            seq = (
                vals[-self.sequence_length:]
                if len(vals) >= self.sequence_length
                else np.concatenate([
                    np.full(self.sequence_length - len(vals),
                            float(vals[0]) if len(vals) > 0 else 0.0),
                    vals,
                ])
            )
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
    def get_lr_history(self) -> List[float]:
        """Return the history of dynamic learning rates for analysis."""
        return list(self._lr_history)
