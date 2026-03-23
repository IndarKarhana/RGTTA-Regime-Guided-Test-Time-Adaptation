"""
Distribution change detection using backward non-overlapping seasonal windows
"""

import numpy as np
import pandas as pd


def flag_intuitive_erratic_seasons(
    df: pd.DataFrame, season_length: int = 12, regime_threshold: float = 0.7
) -> pd.DataFrame:
    """
    Flags each observation in a time series as 'intuitively erratic' (1) or not (0)
    for each backward non-overlapping seasonal window.

    Parameters:
        df : pd.DataFrame with columns ['unique_id', 'ds', 'y']
        season_length : int, length of one season (e.g., 12 for monthly data)
        regime_threshold : float, threshold for recent dip detection (default: 0.7 for 30%)

    Returns:
        pd.DataFrame with original columns plus ['erratic_flag'] for each observation.
    """
    results = []
    for uid, g in df.groupby("unique_id"):
        g = g.sort_values("ds").reset_index(drop=True)
        y = g["y"].values
        n = len(y)
        erratic_flag = np.zeros(n, dtype=int)

        # Split series into non-overlapping backward windows
        num_windows = n // season_length
        for i in range(num_windows):
            start = n - (i + 1) * season_length
            end = n - i * season_length
            if start < 0:
                break
            current_window = y[start:end]
            prior = y[:start]
            if len(current_window) == season_length and len(prior) >= season_length:
                current_mean = np.mean(current_window)
                prior_mean = np.mean(prior)
                if prior_mean > 0 and current_mean < regime_threshold * prior_mean:
                    erratic_flag[start:end] = 1

        # Build result DataFrame for this unique_id
        g["erratic_flag"] = erratic_flag
        results.append(g)

    return pd.concat(results, ignore_index=True)
