"""
PatchTST Time Series Forecasting Model
========================================

Re-implementation of PatchTST from:
    Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting
    with Transformers" (ICLR 2023).

Core idea: Divide each univariate channel into fixed-length patches, embed
each patch as a token, and apply a standard Transformer encoder.  Operates
in a **channel-independent** manner — each variate is processed separately
through the same Transformer, then the outputs are combined.

Key design choices:
  1. **Patching**: Reduces the effective sequence length (L/P tokens instead
     of L), lowering quadratic attention cost and capturing local semantics.
  2. **Channel-independence**: Shared Transformer weights across all variates.
     Avoids quadratic growth with number of variates.
  3. **Instance normalisation**: RevIN-style normalisation per sample for
     non-stationary data (optional, enabled by default).

For univariate input (input_dim=1), PatchTST is equivalent to a patched
Transformer on the single channel.

Same forward() interface as TimeSeriesTransformer / DLinear / iTransformer
so all update policies work unchanged.

Typical parameter count with hidden_dim=64, patch_len=16, 2 layers, 2 heads:
    ~120K parameters (independent of input_dim due to channel-independence)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class _RevIN(nn.Module):
    """Reversible Instance Normalisation (Kim et al., ICLR 2022).

    Normalises each sample and channel independently at input, and
    de-normalises the output. Helps with non-stationary distributions.
    """

    def __init__(self, n_vars: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(n_vars))
            self.bias = nn.Parameter(torch.zeros(n_vars))

    def forward(self, x: torch.Tensor, mode: str = "norm") -> torch.Tensor:
        """x: (B, L, N).  mode='norm' or 'denorm'."""
        if mode == "norm":
            self._mean = x.mean(dim=1, keepdim=True).detach()  # (B, 1, N)
            self._std = (x.std(dim=1, keepdim=True) + self.eps).detach()
            x = (x - self._mean) / self._std
            if self.affine:
                x = x * self.weight + self.bias
        elif mode == "denorm":
            if self.affine:
                x = (x - self.bias) / (self.weight + self.eps)
            x = x * self._std + self._mean
        return x


class _PatchEmbedding(nn.Module):
    """Split a 1-D channel into patches and embed each patch."""

    def __init__(self, patch_len: int, stride: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.projection = nn.Linear(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B*N, L) -> (B*N, n_patches, D)."""
        # Unfold into patches: (B*N, L) -> (B*N, n_patches, patch_len)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = self.projection(x)
        return self.dropout(x)


class _PatchTSTEncoder(nn.Module):
    """Transformer encoder operating on patch tokens (shared across variates)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float = 0.1,
        max_patches: int = 128,
    ):
        super().__init__()
        # Learnable positional encoding for patch positions
        self.pos_embedding = nn.Parameter(torch.randn(1, max_patches, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(d_model, eps=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B*N, n_patches, D) -> (B*N, n_patches, D)."""
        n_patches = x.shape[1]
        x = x + self.pos_embedding[:, :n_patches, :]
        x = self.encoder(x)
        return self.final_norm(x)


class PatchTSTForecaster(nn.Module):
    """
    PatchTST forecaster — channel-independent patched Transformer.

    Same interface as TimeSeriesTransformer / LSTMForecaster / etc.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.1,
        forecast_horizon: int = 96,
        season_length: int = 12,
        exog_dim: int = 0,
        max_seq_len: int = 512,
        patch_len: int = 16,
        stride: int = 8,
        use_revin: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon
        self.season_length = season_length
        self.exog_dim = exog_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.patch_len = patch_len
        self.stride = stride
        self.use_revin = use_revin

        # Total variates
        self.n_vars = input_dim + exog_dim

        # RevIN (optional instance normalisation)
        if use_revin:
            self.revin = _RevIN(self.n_vars, affine=True)
        else:
            self.revin = None

        # Patch embedding (shared across variates)
        self.patch_embed = _PatchEmbedding(patch_len, stride, hidden_dim, dropout)

        # Transformer encoder (shared across variates)
        max_patches = (max_seq_len - patch_len) // stride + 2
        self.encoder = _PatchTSTEncoder(
            d_model=hidden_dim,
            n_heads=num_heads,
            n_layers=num_layers,
            d_ff=hidden_dim * 4,
            dropout=dropout,
            max_patches=max_patches,
        )

        # Output head: flatten patch tokens -> forecast horizon
        # Built lazily since n_patches depends on actual seq_len
        self._n_patches: int = 0
        self._head: nn.Module = None  # type: ignore[assignment]

        # Component heads (for compatibility with regime_aware_loss)
        self.trend_head = nn.Identity()
        self.seasonal_head = nn.Identity()

        self._init_weights()

    # ------------------------------------------------------------------
    def _build_head(self, n_patches: int, device: torch.device):
        """Lazily build the output head once n_patches is known."""
        self._n_patches = n_patches
        flatten_dim = n_patches * self.hidden_dim
        self._head = nn.Sequential(
            nn.Flatten(start_dim=-2),        # (B*N, n_patches*D)
            nn.Linear(flatten_dim, self.forecast_horizon),
        ).to(device)

        # Component heads with correct dims
        self.trend_head = nn.Sequential(
            nn.Linear(self.forecast_horizon, self.forecast_horizon),
        ).to(device)
        self.seasonal_head = nn.Sequential(
            nn.Linear(self.forecast_horizon, self.forecast_horizon),
        ).to(device)

        # Init the new layers
        for name, param in self._head.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_uniform_(param, gain=0.5)
            elif "bias" in name:
                nn.init.zeros_(param)

    # ------------------------------------------------------------------
    def _init_weights(self):
        """Initialize weights for stability."""
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_uniform_(param, gain=0.5)
            elif "bias" in name:
                nn.init.zeros_(param)

    def clamp_weights(self, max_val: float = 3.0):
        """Clamp all weights to prevent explosion."""
        with torch.no_grad():
            for param in self.parameters():
                param.data.clamp_(-max_val, max_val)

    def initialize_weights(self):
        """Re-initialize model weights."""
        self._init_weights()

    # ------------------------------------------------------------------
    def forward(
        self,
        target_seq: torch.Tensor,
        exog_seq: torch.Tensor = None,
        return_components: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with channel-independent patched Transformer.

        Args:
            target_seq: (batch, seq_len, input_dim) — target variates
            exog_seq:   (batch, seq_len, exog_dim) or None
            return_components: whether to return trend/seasonal heads

        Returns:
            forecast: (batch, forecast_horizon) — averaged over variates
        """
        target_seq = self._validate_input(target_seq)
        batch_size, seq_len, _ = target_seq.shape

        # Combine with exogenous if present
        if exog_seq is not None and self.exog_dim > 0:
            exog_seq = self._validate_input(exog_seq)
            if exog_seq.shape[-1] == self.exog_dim:
                combined = torch.cat([target_seq, exog_seq], dim=-1)
            else:
                combined = target_seq
        else:
            combined = target_seq  # (B, L, input_dim)

        # Pad/truncate variate dimension
        actual_vars = combined.shape[-1]
        expected_vars = self.n_vars
        if actual_vars < expected_vars:
            pad = torch.zeros(
                batch_size, seq_len, expected_vars - actual_vars,
                device=combined.device, dtype=combined.dtype,
            )
            combined = torch.cat([combined, pad], dim=-1)
        elif actual_vars > expected_vars:
            combined = combined[:, :, :expected_vars]

        n_vars = combined.shape[-1]

        # Pad sequence length to be compatible with patch_len
        if seq_len < self.patch_len:
            pad_len = self.patch_len - seq_len
            combined = F.pad(combined, (0, 0, pad_len, 0), mode="constant", value=0.0)
            seq_len = combined.shape[1]

        # RevIN normalisation
        if self.revin is not None:
            combined = self.revin(combined, mode="norm")

        # Channel-independent: reshape (B, L, N) -> (B*N, L)
        x = combined.permute(0, 2, 1).reshape(batch_size * n_vars, seq_len)

        # Patch embedding: (B*N, L) -> (B*N, n_patches, D)
        x = self.patch_embed(x)
        n_patches = x.shape[1]

        # Encoder: (B*N, n_patches, D) -> (B*N, n_patches, D)
        x = self.encoder(x)

        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        x = torch.clamp(x, -10.0, 10.0)

        # Lazily build head
        if self._head is None or n_patches != self._n_patches:
            self._build_head(n_patches, x.device)

        # Head: (B*N, n_patches, D) -> (B*N, H)
        forecasts = self._head(x)

        # Reshape: (B*N, H) -> (B, N, H)
        forecasts = forecasts.reshape(batch_size, n_vars, self.forecast_horizon)

        # RevIN de-normalisation (on forecast dimension)
        if self.revin is not None:
            # denorm expects (B, L, N) — transpose, denorm, transpose back
            forecasts_t = forecasts.permute(0, 2, 1)  # (B, H, N)
            forecasts_t = self.revin(forecasts_t, mode="denorm")
            forecasts = forecasts_t.permute(0, 2, 1)  # (B, N, H)

        # Average across variates: (B, N, H) -> (B, H)
        forecast = forecasts.mean(dim=1)
        forecast = torch.clamp(forecast, -5.0, 5.0)

        if torch.isnan(forecast).any():
            forecast = torch.nan_to_num(forecast, nan=0.0)

        if return_components:
            trend = torch.clamp(self.trend_head(forecast), -5.0, 5.0)
            seasonal = torch.clamp(self.seasonal_head(forecast), -5.0, 5.0)
            return {
                "forecast": forecast,
                "trend": trend,
                "seasonal": seasonal,
                "residual": forecast - trend - seasonal,
            }
        return forecast

    # ------------------------------------------------------------------
    def _validate_input(self, x: torch.Tensor) -> torch.Tensor:
        """Validate and clean input tensor."""
        x = torch.clamp(x, -100.0, 100.0)
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        return x

    def get_attention_weights(self, target_seq, exog_seq=None):
        """Placeholder for interface compatibility."""
        return None
