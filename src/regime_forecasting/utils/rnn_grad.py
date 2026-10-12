"""Allow cuDNN RNN gradients while retaining evaluation-time dropout behavior."""
from contextlib import contextmanager
import torch


@contextmanager
def evaluation_rnn_grad(model):
    saved = []
    try:
        for layer in model.modules():
            if isinstance(layer, torch.nn.RNNBase):
                saved.append((layer, layer.training, layer.dropout))
                # cuDNN requires training mode for its backward reserve buffer.
                # Zero dropout preserves the evaluation-mode forward function.
                layer.training = True
                layer.dropout = 0.0
        yield
    finally:
        for layer, training, dropout in saved:
            layer.training = training
            layer.dropout = dropout
