"""
Large GRU Time Series Forecasting Model
========================================

Larger variant of the GRU model (hidden_dim=128, 3 layers).
Same interface as TimeSeriesTransformer. ~180K parameters.

Used to test whether the update-policy findings scale with model size.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LargeGRUForecaster(nn.Module):
    """
    Larger GRU forecaster (hidden_dim=128, 3 layers, ~180K params).

    Same forward() signature as TimeSeriesTransformer.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,      # ignored
        dropout: float = 0.1,
        forecast_horizon: int = 6,
        season_length: int = 12,
        exog_dim: int = 0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon
        self.season_length = season_length
        self.exog_dim = exog_dim
        self.num_layers = num_layers

        effective_input_dim = input_dim + exog_dim

        self.input_projection = nn.Linear(effective_input_dim, hidden_dim)

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
        )

        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-5)

        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, forecast_horizon),
        )

        self.trend_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, forecast_horizon),
        )
        self.seasonal_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, forecast_horizon),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        """Initialize weights for stability."""
        for name, param in self.named_parameters():
            if "weight" in name:
                if "gru" in name:
                    if param.dim() >= 2:
                        nn.init.orthogonal_(param, gain=0.5)
                elif param.dim() > 1:
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
        """Forward pass — identical to TimeSeriesTransformer."""
        batch_size, seq_len, _ = target_seq.shape
        target_seq = self._validate_input(target_seq)

        if exog_seq is not None and self.exog_dim > 0:
            exog_seq = self._validate_input(exog_seq)
            if exog_seq.shape[-1] == self.exog_dim:
                combined = torch.cat([target_seq, exog_seq], dim=-1)
            else:
                combined = target_seq
        else:
            combined = target_seq

        actual = combined.shape[-1]
        expected = self.input_projection.in_features
        if actual < expected:
            pad = torch.zeros(batch_size, seq_len, expected - actual,
                              device=combined.device, dtype=combined.dtype)
            combined = torch.cat([combined, pad], dim=-1)
        elif actual > expected:
            combined = combined[:, :, :expected]

        hidden = self.input_projection(combined)
        hidden = torch.clamp(hidden, -10.0, 10.0)

        gru_out, _ = self.gru(hidden)
        if torch.isnan(gru_out).any():
            gru_out = torch.nan_to_num(gru_out, nan=0.0)
        gru_out = torch.clamp(gru_out, -10.0, 10.0)

        gru_out = self.layer_norm(gru_out)
        last_hidden = gru_out[:, -1, :]

        forecast = self.output_projection(last_hidden)
        forecast = torch.clamp(forecast, -5.0, 5.0)
        if torch.isnan(forecast).any():
            forecast = torch.nan_to_num(forecast, nan=0.0)

        if return_components:
            trend = torch.clamp(self.trend_head(last_hidden), -5.0, 5.0)
            seasonal = torch.clamp(self.seasonal_head(last_hidden), -5.0, 5.0)
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
