"""
iTransformer Time Series Forecasting Model
============================================

Re-implementation of the iTransformer from:
    Liu et al., "iTransformer: Inverted Transformers Are Effective for
    Time Series Forecasting" (ICLR 2024).

Core idea: Instead of applying attention across the temporal dimension
(like vanilla Transformers), iTransformer *inverts* the dimensions:
  - Each variate (channel) is treated as a token
  - Attention is applied across variates (channel-mixing)
  - The temporal dimension is embedded via a linear projection per variate

This makes iTransformer natively multivariate — it learns inter-variate
dependencies through attention, while a per-variate MLP captures temporal
patterns.

For univariate input (input_dim=1), the model degenerates to a linear
encoder-decoder with self-attention over a single token, which is still
functional but less interesting.

Same forward() interface as TimeSeriesTransformer / DLinear / etc. so all
update policies work unchanged.

Typical parameter count with hidden_dim=64, 2 layers, 2 heads:
    ~150K parameters (varies with input_dim)
"""

import torch
import torch.nn as nn


class _InvertedTokenEmbedding(nn.Module):
    """Embed each variate's temporal window into a d_model-dim token.

    Input:  (batch, seq_len, n_vars)  – standard time-series layout
    Output: (batch, n_vars, d_model)  – one token per variate
    """

    def __init__(self, seq_len: int, d_model: int):
        super().__init__()
        self.linear = nn.Linear(seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, N) -> transpose to (B, N, L) -> project -> (B, N, D)
        return self.linear(x.permute(0, 2, 1))


class _InvertedTransformerEncoderLayer(nn.Module):
    """Transformer encoder layer operating on variate tokens.

    Attention is across variates (channel-mixing), not across time.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm style
        x2 = self.norm1(x)
        attn_out, _ = self.self_attn(x2, x2, x2)
        x = x + self.dropout(attn_out)

        x2 = self.norm2(x)
        x = x + self.ff(x2)
        return x


class iTransformerForecaster(nn.Module):
    """
    iTransformer forecaster — inverted attention across variates.

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
        max_seq_len: int = 512,  # interface compat
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon
        self.season_length = season_length
        self.exog_dim = exog_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.input_dim = input_dim

        # Total number of variates the model sees
        self.n_vars = input_dim + exog_dim

        # Per-variate temporal embedding: (B, L, N) -> (B, N, D)
        # Built lazily on first forward because seq_len may not be known
        self._seq_len: int = 0
        self._token_embed: nn.Module = None  # type: ignore[assignment]

        # Encoder layers (attention across variates)
        self.encoder_layers = nn.ModuleList(
            [
                _InvertedTransformerEncoderLayer(
                    d_model=hidden_dim,
                    n_heads=num_heads,
                    d_ff=hidden_dim * 4,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Final layer norm
        self.final_norm = nn.LayerNorm(hidden_dim, eps=1e-5)

        # Per-variate projection back to forecast horizon: (B, N, D) -> (B, N, H)
        self.output_projection = nn.Linear(hidden_dim, forecast_horizon)

        # Component heads (for compatibility with regime_aware_loss)
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
    def _build_token_embed(self, seq_len: int, device: torch.device):
        """Lazily build the temporal embedding once seq_len is known."""
        self._seq_len = seq_len
        self._token_embed = _InvertedTokenEmbedding(seq_len, self.hidden_dim).to(device)
        # Re-init weights for the new layer
        for name, param in self._token_embed.named_parameters():
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
        Forward pass with inverted (variate-wise) attention.

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

        # Pad/truncate variate dimension if needed
        actual_vars = combined.shape[-1]
        expected_vars = self.n_vars
        if actual_vars < expected_vars:
            pad = torch.zeros(
                batch_size,
                seq_len,
                expected_vars - actual_vars,
                device=combined.device,
                dtype=combined.dtype,
            )
            combined = torch.cat([combined, pad], dim=-1)
        elif actual_vars > expected_vars:
            combined = combined[:, :, :expected_vars]

        # Lazy-build temporal embedding
        if self._token_embed is None or seq_len != self._seq_len:
            self._build_token_embed(seq_len, combined.device)

        # Embed: (B, L, N) -> (B, N, D)  — one token per variate
        tokens = self._token_embed(combined)
        tokens = torch.clamp(tokens, -10.0, 10.0)

        # Encoder: attention across variates
        for layer in self.encoder_layers:
            tokens = layer(tokens)

        if torch.isnan(tokens).any():
            tokens = torch.nan_to_num(tokens, nan=0.0)
        tokens = torch.clamp(tokens, -10.0, 10.0)

        tokens = self.final_norm(tokens)  # (B, N, D)

        # Project each variate to forecast horizon: (B, N, D) -> (B, N, H)
        var_forecasts = self.output_projection(tokens)

        # Average across variates to get final forecast: (B, H)
        forecast = var_forecasts.mean(dim=1)
        forecast = torch.clamp(forecast, -5.0, 5.0)

        if torch.isnan(forecast).any():
            forecast = torch.nan_to_num(forecast, nan=0.0)

        if return_components:
            # Use mean variate token for component heads
            mean_token = tokens.mean(dim=1)  # (B, D)
            trend = torch.clamp(self.trend_head(mean_token), -5.0, 5.0)
            seasonal = torch.clamp(self.seasonal_head(mean_token), -5.0, 5.0)
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
