"""
Stable Time Series Forecasting Models
=====================================

This module provides numerically stable neural network architectures for time series forecasting.
The primary model is GRU-based for stability, with optional attention mechanism.

Key stability features:
- GRU instead of full Transformer (more stable for small datasets)
- Weight clamping and gradient clipping
- Input/output clamping
- Robust initialization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class TimeSeriesTransformer(nn.Module):
    """
    A stable GRU-based time series forecasting model.
    
    Uses GRU (Gated Recurrent Unit) instead of full Transformer for:
    1. Better numerical stability
    2. Fewer parameters (less overfitting on small datasets)
    3. No attention softmax overflow issues
    
    The model maintains the same interface as the original Transformer.
    """
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,  # Not used, kept for compatibility
        dropout: float = 0.1,
        forecast_horizon: int = 6,
        season_length: int = 12,
        exog_dim: int = 0
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon
        self.season_length = season_length
        self.exog_dim = exog_dim
        self.num_layers = num_layers
        
        # Calculate effective input dimension
        effective_input_dim = input_dim + exog_dim
        
        # Input projection with proper initialization
        self.input_projection = nn.Linear(effective_input_dim, hidden_dim)
        
        # GRU layers (more stable than Transformer for small datasets)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-5)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, forecast_horizon)
        )
        
        # Component heads (for compatibility)
        self.trend_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, forecast_horizon)
        )
        
        self.seasonal_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, forecast_horizon)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with small values for stability."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'gru' in name:
                    # GRU uses orthogonal initialization
                    if param.dim() >= 2:
                        nn.init.orthogonal_(param, gain=0.5)
                elif param.dim() > 1:
                    nn.init.xavier_uniform_(param, gain=0.5)
            elif 'bias' in name:
                nn.init.zeros_(param)
                
    def clamp_weights(self, max_val: float = 3.0):
        """Clamp all weights to prevent explosion."""
        with torch.no_grad():
            for param in self.parameters():
                param.data.clamp_(-max_val, max_val)
        
    def forward(
        self, 
        target_seq: torch.Tensor, 
        exog_seq: torch.Tensor = None, 
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Forward pass for time series forecasting.
        
        Args:
            target_seq: (batch_size, seq_len, 1) - target time series
            exog_seq: (batch_size, seq_len, exog_dim) - exogenous variables (optional)
            return_components: bool - whether to return trend/seasonal components
            
        Returns:
            forecast: (batch_size, forecast_horizon) - predicted values
        """
        batch_size, seq_len, input_features = target_seq.shape
        
        # Input validation and clamping
        target_seq = self._validate_input(target_seq)
        
        # Handle exogenous features - combine with target if provided
        if exog_seq is not None and self.exog_dim > 0:
            exog_seq = self._validate_input(exog_seq)
            # Make sure dimensions match
            if exog_seq.shape[-1] == self.exog_dim:
                combined = torch.cat([target_seq, exog_seq], dim=-1)
            else:
                combined = target_seq
        else:
            combined = target_seq
        
        # Verify input dimension matches projection layer
        actual_input_dim = combined.shape[-1]
        expected_input_dim = self.input_projection.in_features
        
        if actual_input_dim != expected_input_dim:
            # Dimension mismatch - pad or truncate as needed
            if actual_input_dim < expected_input_dim:
                # Pad with zeros
                padding = torch.zeros(
                    batch_size, seq_len, expected_input_dim - actual_input_dim,
                    device=combined.device, dtype=combined.dtype
                )
                combined = torch.cat([combined, padding], dim=-1)
            else:
                # Truncate to expected dimension
                combined = combined[:, :, :expected_input_dim]
        
        # Project to hidden dimension
        hidden = self.input_projection(combined)  # (batch, seq_len, hidden)
        hidden = torch.clamp(hidden, -10.0, 10.0)
        
        # GRU encoding
        gru_out, _ = self.gru(hidden)  # (batch, seq_len, hidden)
        
        # Check for NaN and handle
        if torch.isnan(gru_out).any():
            gru_out = torch.nan_to_num(gru_out, nan=0.0)
        gru_out = torch.clamp(gru_out, -10.0, 10.0)
        
        # Apply layer normalization
        gru_out = self.layer_norm(gru_out)
        
        # Use last timestep for forecasting
        last_hidden = gru_out[:, -1, :]  # (batch, hidden)
        
        # Generate forecast
        forecast = self.output_projection(last_hidden)  # (batch, forecast_horizon)
        forecast = torch.clamp(forecast, -5.0, 5.0)
        
        # Final NaN check
        if torch.isnan(forecast).any():
            forecast = torch.nan_to_num(forecast, nan=0.0)
        
        if return_components:
            trend = self.trend_head(last_hidden)
            seasonal = self.seasonal_head(last_hidden)
            trend = torch.clamp(trend, -5.0, 5.0)
            seasonal = torch.clamp(seasonal, -5.0, 5.0)
            return {
                'forecast': forecast,
                'trend': trend,
                'seasonal': seasonal,
                'residual': forecast - trend - seasonal
            }
        
        return forecast
    
    def _validate_input(self, x: torch.Tensor) -> torch.Tensor:
        """Validate and clean input tensor."""
        # Clamp to reasonable range
        x = torch.clamp(x, -100.0, 100.0)
        # Replace NaN/Inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        return x
    
    def get_attention_weights(self, target_seq, exog_seq=None):
        """Placeholder for compatibility."""
        return None
    
    def initialize_weights(self):
        """Re-initialize model weights."""
        self._init_weights()


def regime_aware_loss(
    y_true: torch.Tensor, 
    y_pred: torch.Tensor, 
    components: dict = None, 
    alpha: float = 0.3
) -> torch.Tensor:
    """
    Numerically stable loss function for regime-aware forecasting.
    
    Uses Smooth L1 (Huber) loss which is more robust to outliers
    and numerically stable than MSE.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values  
        components: Dict with 'trend', 'seasonal' components if available
        alpha: Weight for seasonal consistency loss
        
    Returns:
        Total loss value
    """
    # Clamp predictions to prevent extreme values
    y_pred = torch.clamp(y_pred, -5.0, 5.0)
    y_true = torch.clamp(y_true, -5.0, 5.0)
    
    # Replace NaN/Inf
    y_pred = torch.nan_to_num(y_pred, nan=0.0, posinf=1.0, neginf=-1.0)
    y_true = torch.nan_to_num(y_true, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Primary loss - Smooth L1 (Huber)
    forecast_loss = F.smooth_l1_loss(y_pred, y_true, reduction='mean', beta=1.0)
    
    # Fallback if loss is invalid
    if torch.isnan(forecast_loss) or torch.isinf(forecast_loss):
        diff = torch.clamp(y_pred - y_true, -2.0, 2.0)
        forecast_loss = torch.mean(diff ** 2)
    
    if components is not None:
        # Seasonal consistency loss
        seasonal_target = y_true - components['trend']
        seasonal_target = torch.clamp(seasonal_target, -5.0, 5.0)
        seasonal_loss = F.smooth_l1_loss(
            components['seasonal'], seasonal_target, reduction='mean', beta=1.0
        )
        
        if torch.isnan(seasonal_loss) or torch.isinf(seasonal_loss):
            seasonal_loss = torch.tensor(0.0, device=y_pred.device)
            
        total_loss = forecast_loss + alpha * seasonal_loss
    else:
        total_loss = forecast_loss
    
    return total_loss
