"""Observed-window dependence signature for the reviewer feature ablation."""
import numpy as np
import pandas as pd


def dependence_signature(df, feature_cols, marginal):
    cols = ['y'] + [c for c in (feature_cols or []) if c != 'y']
    x = df.reindex(columns=cols).apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    x[~np.isfinite(x)] = np.nan
    for col in range(x.shape[1]):
        valid = np.isfinite(x[:, col])
        x[~valid, col] = x[valid, col].mean() if valid.any() else 0.0
    z = (x - x.mean(axis=0)) / np.maximum(x.std(axis=0), 1e-8)
    corr = z.T @ z / max(len(z), 1)
    contemporaneous = corr[np.triu_indices(z.shape[1], k=1)]
    temporal = []
    for lag in (1, 6, 24):
        temporal.extend(np.mean(z[lag:] * z[:-lag], axis=0) if len(z) > lag else np.zeros(z.shape[1]))
    cross_lag = (z[1:].T @ z[:-1] / max(len(z)-1, 1)).ravel()
    moments = np.sign(marginal) * np.log1p(np.abs(marginal))
    blocks = [moments, contemporaneous, np.asarray(temporal), cross_lag]
    return np.concatenate([np.nan_to_num(b) / np.sqrt(max(len(b), 1)) for b in blocks])
