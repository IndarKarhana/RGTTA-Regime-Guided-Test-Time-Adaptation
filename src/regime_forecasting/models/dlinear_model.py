"""
DLinear Time Series Forecasting Model
======================================

Re-implementation of DLinear from:
    Zeng et al., "Are Transformers Effective for Time Series Forecasting?"
    (AAAI 2023).

DLinear decomposes the input series into trend and seasonal (remainder)
components using a simple moving-average kernel, then applies a separate
linear layer to each component.  The forecast is the sum of both outputs.

This is the *standard baseline* used in TAFAS (AAAI 2025) and DynaTTA
(ICML 2025).  Same forward() interface as our other models so all update
policies work unchanged.

Typical parameter count with seq_len=96, pred_len=96:
    2 × (96 × 96 + 96) = 18,624 parameters  (~19K)
"""
import torch
import torch.nn as nn
import numpy as np


class _MovingAvgBlock(nn.Module):
    """Moving-average block for trend extraction."""

    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, channels)
        Returns:
            Trend component: (batch, seq_len, channels)
        """
        # Pad symmetrically so output length == input length
        front = x[:, :1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x_padded = torch.cat([front, x, end], dim=1)
        # AvgPool1d expects (batch, channels, length)
        x_t = x_padded.permute(0, 2, 1)
        trend = self.avg(x_t).permute(0, 2, 1)
        return trend


class _SeriesDecomposition(nn.Module):
    """Decompose a series into trend + remainder (seasonal)."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = _MovingAvgBlock(kernel_size)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, seq_len, channels)
        Returns:
            (seasonal, trend)  each (batch, seq_len, channels)
        """
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class DLinearForecaster(nn.Module):
    """
    DLinear forecaster — decomposition + two linear projections.

    Same interface as TimeSeriesTransformer / LSTMForecaster / etc.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,       # ignored — kept for interface compat
        num_layers: int = 2,         # ignored
        num_heads: int = 4,          # ignored
        dropout: float = 0.1,        # ignored (model is linear)
        forecast_horizon: int = 96,
        season_length: int = 12,
        exog_dim: int = 0,
        max_seq_len: int = 512,      # ignored — for interface compat
    ):
        super().__init__()
        self.hidden_dim = hidden_dim  # stored for interface compat
        self.forecast_horizon = forecast_horizon
        self.season_length = season_length
        self.exog_dim = exog_dim
        self.num_layers = num_layers

        # Decomposition kernel: use season_length if odd, else season_length + 1
        kernel = season_length if season_length % 2 == 1 else season_length + 1
        self.decomposition = _SeriesDecomposition(kernel)

        # We don't know seq_len at init time — build lazy on first forward.
        # The linear layers map (seq_len,) -> (forecast_horizon,) per channel.
        self._seq_len: int = 0
        self._linear_seasonal: nn.Linear = None  # type: ignore[assignment]
        self._linear_trend: nn.Linear = None      # type: ignore[assignment]

        # Effective input channels (for the linear projections)
        self._channels = input_dim + exog_dim

        # Component heads (for compatibility with regime_aware_loss)
        # These are simple pass-through placeholders that get replaced lazily.
        self.trend_head = nn.Identity()
        self.seasonal_head = nn.Identity()

    # ------------------------------------------------------------------
    def _build_linear_layers(self, seq_len: int, device: torch.device):
        """Lazily construct the projection layers once seq_len is known."""
        self._seq_len = seq_len
        self._linear_seasonal = nn.Linear(seq_len, self.forecast_horizon).to(device)
        self._linear_trend = nn.Linear(seq_len, self.forecast_horizon).to(device)

        # Re-build component heads with correct dims
        self.trend_head = nn.Sequential(
            nn.Linear(self.forecast_horizon, self.forecast_horizon),
        ).to(device)
        self.seasonal_head = nn.Sequential(
            nn.Linear(self.forecast_horizon, self.forecast_horizon),
        ).to(device)

        self._init_weights()

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
        Forward pass.

        Args:
            target_seq: (batch, seq_len, 1)
            exog_seq:   (batch, seq_len, exog_dim) or None
            return_components: whether to return trend/seasonal separately

        Returns:
            forecast: (batch, forecast_horizon)
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
            combined = target_seq  # (batch, seq_len, 1)

        # Lazy build linear layers on first call (or if seq_len changes)
        if self._linear_seasonal is None or seq_len != self._seq_len:
            self._build_linear_layers(seq_len, combined.device)

        # Decompose
        seasonal, trend = self.decomposition(combined)
        # seasonal, trend: (batch, seq_len, channels)

        # Project each component: (batch, channels, seq_len) -> (batch, channels, H)
        seasonal_t = seasonal.permute(0, 2, 1)  # (batch, channels, seq_len)
        trend_t = trend.permute(0, 2, 1)

        seasonal_out = self._linear_seasonal(seasonal_t)  # (batch, channels, H)
        trend_out = self._linear_trend(trend_t)

        # Sum components and average over channels -> (batch, H)
        forecast_full = seasonal_out + trend_out  # (batch, channels, H)
        forecast = forecast_full.mean(dim=1)      # (batch, H)

        forecast = torch.clamp(forecast, -5.0, 5.0)
        if torch.isnan(forecast).any():
            forecast = torch.nan_to_num(forecast, nan=0.0)

        if return_components:
            trend_fc = trend_out.mean(dim=1)
            seasonal_fc = seasonal_out.mean(dim=1)
            return {
                "forecast": forecast,
                "trend": torch.clamp(trend_fc, -5.0, 5.0),
                "seasonal": torch.clamp(seasonal_fc, -5.0, 5.0),
                "residual": forecast - trend_fc - seasonal_fc,
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
