"""
Bottleneck Adapter Module for Test-Time Adaptation
====================================================

Implements the Houlsby et al. (ICML 2019) bottleneck adapter:
    down-project → nonlinearity → up-project + residual

Inserted after the FFN in each Transformer encoder layer. During frozen-backbone
TTA, only adapter parameters are trainable (along with the output head).

Designed for iTransformer and PatchTST architectures in the RGTTA framework.
"""

import torch
import torch.nn as nn


class BottleneckAdapter(nn.Module):
    """Bottleneck adapter layer (Houlsby et al., ICML 2019).

    Architecture: x → LayerNorm → Linear(d_model, bottleneck) → GELU
                    → Linear(bottleneck, d_model) → Dropout → + x (residual)

    With bottleneck_dim=16 and d_model=64:
        Parameters: 64*16 + 16 + 16*64 + 64 + 64 + 64 = 2,256
        (~2.3K per adapter, ~4.5K per layer with 2 adapters)

    Args:
        d_model: Dimension of the Transformer hidden state.
        bottleneck_dim: Dimension of the bottleneck projection.
        dropout: Dropout rate applied after up-projection.
    """

    def __init__(self, d_model: int, bottleneck_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=1e-5)
        self.down = nn.Linear(d_model, bottleneck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, d_model)
        self.dropout = nn.Dropout(dropout)

        # Near-identity init: small weights so adapter starts as identity
        nn.init.normal_(self.down.weight, std=0.01)
        nn.init.zeros_(self.down.bias)
        nn.init.normal_(self.up.weight, std=0.01)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., d_model) -> (..., d_model)."""
        residual = x
        h = self.norm(x)
        h = self.down(h)
        h = self.act(h)
        h = self.up(h)
        h = self.dropout(h)
        return residual + h


def inject_adapters_itransformer(
    model: nn.Module,
    bottleneck_dim: int = 16,
    dropout: float = 0.1,
) -> int:
    """Inject bottleneck adapters into an iTransformerForecaster.

    Adds one adapter after each encoder layer's FFN output.
    Returns total adapter parameter count.

    Args:
        model: An iTransformerForecaster instance.
        bottleneck_dim: Bottleneck dimension for adapters.
        dropout: Adapter dropout rate.

    Returns:
        Number of adapter parameters added.
    """
    d_model = model.hidden_dim
    total_params = 0

    # Create adapter ModuleList (one per encoder layer)
    adapters = nn.ModuleList()
    for _ in model.encoder_layers:
        adapter = BottleneckAdapter(d_model, bottleneck_dim, dropout)
        total_params += sum(p.numel() for p in adapter.parameters())
        adapters.append(adapter)

    # Store adapters on the model so they're part of state_dict
    model.adapters = adapters

    # Monkey-patch each encoder layer's forward to include adapter
    for i, layer in enumerate(model.encoder_layers):
        adapter = adapters[i]
        original_forward = layer.forward

        def make_patched_forward(orig_fn, adpt):
            def patched_forward(x):
                x = orig_fn(x)
                x = adpt(x)
                return x

            return patched_forward

        layer.forward = make_patched_forward(original_forward, adapter)

    return total_params


def inject_adapters_patchtst(
    model: nn.Module,
    bottleneck_dim: int = 16,
    dropout: float = 0.1,
) -> int:
    """Inject bottleneck adapters into a PatchTSTForecaster.

    PatchTST uses nn.TransformerEncoder with nn.TransformerEncoderLayer.
    We add one adapter after each encoder layer.
    Returns total adapter parameter count.

    Args:
        model: A PatchTSTForecaster instance.
        bottleneck_dim: Bottleneck dimension for adapters.
        dropout: Adapter dropout rate.

    Returns:
        Number of adapter parameters added.
    """
    d_model = model.hidden_dim
    encoder = model.encoder  # _PatchTSTEncoder
    total_params = 0

    # Access the inner nn.TransformerEncoder's layers
    inner_encoder = encoder.encoder  # nn.TransformerEncoder
    n_layers = len(inner_encoder.layers)

    adapters = nn.ModuleList()
    for _ in range(n_layers):
        adapter = BottleneckAdapter(d_model, bottleneck_dim, dropout)
        total_params += sum(p.numel() for p in adapter.parameters())
        adapters.append(adapter)

    # Store adapters on the model so they're part of state_dict
    model.adapters = adapters

    # Monkey-patch each TransformerEncoderLayer's forward
    for i, layer in enumerate(inner_encoder.layers):
        adapter = adapters[i]
        original_forward = layer.forward

        def make_patched_forward(orig_fn, adpt):
            def patched_forward(src, *args, **kwargs):
                out = orig_fn(src, *args, **kwargs)
                out = adpt(out)
                return out

            return patched_forward

        layer.forward = make_patched_forward(original_forward, adapter)

    return total_params


def inject_adapters(model: nn.Module, model_key: str, bottleneck_dim: int = 16, dropout: float = 0.1) -> int:
    """Inject adapters into a model based on its type.

    Only injects into attention-based architectures (iTransformer, PatchTST).
    Returns 0 for GRU/DLinear (no adapters needed — backbone already compact).

    Args:
        model: The model instance.
        model_key: One of 'itransformer', 'patchtst', 'gru_small', 'dlinear'.
        bottleneck_dim: Bottleneck dimension.
        dropout: Adapter dropout.

    Returns:
        Number of adapter parameters added (0 for non-attention models).
    """
    if model_key == "itransformer":
        return inject_adapters_itransformer(model, bottleneck_dim, dropout)
    elif model_key == "patchtst":
        return inject_adapters_patchtst(model, bottleneck_dim, dropout)
    else:
        return 0  # GRU, DLinear — no adapters
