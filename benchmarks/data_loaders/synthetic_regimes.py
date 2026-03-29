#!/usr/bin/env python3
"""
Synthetic Regime Dataset Generator
====================================

Generates benchmark-scale synthetic time series with *known* regime shifts
for evaluating RGTTA and other incremental forecasting policies.

Each dataset is generated at hourly frequency with enough data for:
  720 (initial train) + 48 × 500 (batches) = 24720 rows minimum.
  This matches the scale of ETTh1/h2 (~17K rows, 33 batches).

Datasets (8 total):
  1. synth_stable        – Single stable regime (no shifts). Baseline control.
  2. synth_trend_break    – One sudden trend reversal mid-series.
  3. synth_slow_drift     – Gradual mean drift over time (no abrupt break).
  4. synth_fast_switch    – Rapid regime switching every ~600 points.
  5. synth_recurring      – 3 regimes that repeat (A-B-C-A-B-C). Best test for
                            checkpoint reuse: later segments should match earlier
                            ones and trigger HIGH tier.
  6. synth_volatility     – Alternating low/high volatility regimes.
  7. synth_shock_recovery – Stable → shock → recovery → new stable.
  8. synth_multi_regime   – Kitchen-sink: 5 distinct regimes, varying lengths,
                            some recurring. Closest to real-world complexity.

All generated CSVs have columns: [date, unique_id, ds, y, regime, regime_change]
and are saved to data/benchmarks/ alongside the ETT files.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
import logging
import json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEASON_LENGTH = 24  # hourly seasonality (daily cycle)
TARGET_ROWS = 25200  # 720 train + 48*500 batches + 520 buffer (matches ETTh scale)
RNG_BASE_SEED = 2026

# ---------------------------------------------------------------------------
# Regime specification library
# ---------------------------------------------------------------------------
REGIME_LIBRARY: Dict[str, dict] = {
    "stable_growth":     {"base": 500,  "season_amp": 30,  "noise": 12, "trend": 0.005},
    "volatile_expansion":{"base": 900,  "season_amp": 90,  "noise": 45, "trend": 0.010},
    "stagnation":        {"base": 600,  "season_amp": 15,  "noise":  8, "trend": 0.001},
    "decline":           {"base": 800,  "season_amp": 40,  "noise": 20, "trend":-0.008},
    "explosive_growth":  {"base":1500,  "season_amp":150,  "noise": 60, "trend": 0.020},
    "high_volatility":   {"base": 700,  "season_amp": 50,  "noise": 80, "trend": 0.002},
    "low_volatility":    {"base": 700,  "season_amp": 50,  "noise":  5, "trend": 0.002},
    "shock":             {"base":2000,  "season_amp":200,  "noise":100, "trend":-0.015},
    "recovery":          {"base":1200,  "season_amp": 70,  "noise": 30, "trend": 0.012},
}


def _synthesize_segment(
    spec: dict,
    length: int,
    rng: np.random.Generator,
    start_value: float = None,
) -> np.ndarray:
    """Generate a time series segment from a regime specification.

    Args:
        spec: Dict with keys base, season_amp, noise, trend.
        length: Number of time steps.
        rng: Numpy random generator.
        start_value: If given, shift the segment so it starts near this value
                     (smooth transitions between regimes).

    Returns:
        1-D array of length `length`.
    """
    t = np.arange(length, dtype=float)
    seasonal = spec["season_amp"] * np.sin(2 * np.pi * t / SEASON_LENGTH)
    trend_component = spec["trend"] * t
    level = spec["base"] + trend_component + seasonal
    noise = rng.normal(0, spec["noise"], length)
    y = level + noise

    # Optionally smooth-start from previous segment's last value
    if start_value is not None:
        offset = start_value - y[0]
        # Decay the offset over ~50 steps so transition isn't perfectly abrupt
        decay = np.exp(-np.arange(length) / 50.0)
        y += offset * decay

    return y


def _build_series_from_schedule(
    schedule: List[Tuple[str, int]],
    seed: int,
    smooth_transitions: bool = True,
) -> pd.DataFrame:
    """Build a full series from a list of (regime_name, length) pairs.

    Args:
        schedule: List of (regime_name, length) tuples.
        seed: Random seed.
        smooth_transitions: If True, each segment starts near the previous
            segment's last value to avoid unrealistic jumps.

    Returns:
        DataFrame with [unique_id, ds, y, regime, regime_change].
    """
    rng = np.random.default_rng(seed)
    segments_y: List[np.ndarray] = []
    regimes: List[str] = []
    regime_changes: List[bool] = []

    last_val = None
    for i, (regime_name, length) in enumerate(schedule):
        spec = REGIME_LIBRARY[regime_name]
        start_val = last_val if (smooth_transitions and last_val is not None) else None
        seg = _synthesize_segment(spec, length, rng, start_value=start_val)
        segments_y.append(seg)
        last_val = float(seg[-1])

        regimes.extend([regime_name] * length)
        regime_changes.extend([i > 0] + [False] * (length - 1))

    y_all = np.concatenate(segments_y)
    total = len(y_all)
    dates = pd.date_range("2020-01-01", periods=total, freq="h")

    df = pd.DataFrame({
        "unique_id": "synth",
        "ds": dates,
        "y": y_all,
        "regime": regimes,
        "regime_change": regime_changes,
    })
    return df


# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------

def generate_synth_stable(seed: int = RNG_BASE_SEED) -> pd.DataFrame:
    """Single stable regime — control baseline (no regime changes)."""
    schedule = [("stable_growth", TARGET_ROWS)]
    return _build_series_from_schedule(schedule, seed)


def generate_synth_trend_break(seed: int = RNG_BASE_SEED + 1) -> pd.DataFrame:
    """One abrupt trend reversal early in the series: growth → decline.

    Break at row 3000 so the streaming benchmark (10 batches starting at
    row 720) sees both regimes within its window.
    """
    break_at = 3000
    schedule = [
        ("stable_growth", break_at),
        ("decline", TARGET_ROWS - break_at),
    ]
    return _build_series_from_schedule(schedule, seed, smooth_transitions=True)


def generate_synth_slow_drift(seed: int = RNG_BASE_SEED + 2) -> pd.DataFrame:
    """Gradual mean drift — no sharp boundary, just slow shift.

    Implemented as many micro-segments with slowly changing base.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(TARGET_ROWS, dtype=float)
    # Base slowly migrates from 500 to 1200
    base = 500 + 700 * (t / TARGET_ROWS)
    seasonal = 40 * np.sin(2 * np.pi * t / SEASON_LENGTH)
    trend = 0.003 * t
    noise = rng.normal(0, 15, TARGET_ROWS)
    y = base + seasonal + trend + noise

    dates = pd.date_range("2020-01-01", periods=TARGET_ROWS, freq="h")
    # Mark "regime change" at every 1000-point boundary as approximate indicator
    regime_labels = [f"drift_phase_{int(i // 1000)}" for i in range(TARGET_ROWS)]
    regime_change = [False] * TARGET_ROWS
    for i in range(1000, TARGET_ROWS, 1000):
        regime_change[i] = True

    return pd.DataFrame({
        "unique_id": "synth",
        "ds": dates,
        "y": y,
        "regime": regime_labels,
        "regime_change": regime_change,
    })


def generate_synth_fast_switch(seed: int = RNG_BASE_SEED + 3) -> pd.DataFrame:
    """Rapid regime switching every ~2100 points (≈88 days at hourly freq).

    Alternates between 3 very different regimes.
    """
    seg_len = 2100
    regimes_cycle = ["stable_growth", "volatile_expansion", "decline"]
    schedule = []
    total = 0
    i = 0
    while total < TARGET_ROWS:
        r = regimes_cycle[i % len(regimes_cycle)]
        this_len = min(seg_len, TARGET_ROWS - total)
        schedule.append((r, this_len))
        total += this_len
        i += 1
    return _build_series_from_schedule(schedule, seed)


def generate_synth_recurring(seed: int = RNG_BASE_SEED + 4) -> pd.DataFrame:
    """Recurring regimes: A → B → C → A → B → C.

    This is the *ideal* test for checkpoint reuse: when regime A repeats,
    RGTTA should find the earlier A-checkpoint and fire HIGH tier.
    Segments shortened to 2000 so the full A→B→C→A cycle is visible
    within the 10-batch streaming window.
    """
    seg_len = 2000  # each segment ~83 days
    pattern = ["stable_growth", "volatile_expansion", "stagnation"]
    schedule = []
    total = 0
    i = 0
    while total < TARGET_ROWS:
        r = pattern[i % len(pattern)]
        this_len = min(seg_len, TARGET_ROWS - total)
        schedule.append((r, this_len))
        total += this_len
        i += 1
    return _build_series_from_schedule(schedule, seed)


def generate_synth_volatility(seed: int = RNG_BASE_SEED + 5) -> pd.DataFrame:
    """Alternating low-volatility and high-volatility regimes.

    Same base level and trend — only noise amplitude changes.
    Tests whether regime detection picks up variance shifts (not just mean).
    Segments shortened to 1500 so multiple transitions occur within
    the 10-batch streaming window.
    """
    seg_len = 1500
    pattern = ["low_volatility", "high_volatility"]
    schedule = []
    total = 0
    i = 0
    while total < TARGET_ROWS:
        r = pattern[i % len(pattern)]
        this_len = min(seg_len, TARGET_ROWS - total)
        schedule.append((r, this_len))
        total += this_len
        i += 1
    return _build_series_from_schedule(schedule, seed)


def generate_synth_shock_recovery(seed: int = RNG_BASE_SEED + 6) -> pd.DataFrame:
    """Stable → shock → recovery → new stable.

    Models a COVID-like event: stable period, sudden shock,
    gradual recovery, then new equilibrium.  Phases are sized so
    all 4 appear within the 10-batch streaming window (rows 720-8220).
    """
    schedule = [
        ("stable_growth",  2500),  # ~B1-B2 stable
        ("shock",          2000),  # ~B3-B5 shock
        ("recovery",       2000),  # ~B5-B7 recovery
        ("stagnation",     TARGET_ROWS - 6500),  # B8+ new equilibrium
    ]
    return _build_series_from_schedule(schedule, seed, smooth_transitions=True)


def generate_synth_multi_regime(seed: int = RNG_BASE_SEED + 7) -> pd.DataFrame:
    """Kitchen-sink: 5 distinct regimes with varying lengths and some recurrence.

    Closest to real-world complexity. Tests the full RGTTA machinery.
    """
    schedule = [
        ("stable_growth",      2000),
        ("volatile_expansion",  1500),
        ("stagnation",         1500),
        ("explosive_growth",    1500),
        ("decline",             1500),
        ("stable_growth",       2000),  # recurrence of first regime
        ("shock",               1500),
        ("recovery",           TARGET_ROWS - 11500),  # fill remainder
    ]
    return _build_series_from_schedule(schedule, seed, smooth_transitions=True)


# ---------------------------------------------------------------------------
# Registry: name → generator function
# ---------------------------------------------------------------------------
SYNTHETIC_DATASETS: Dict[str, callable] = {
    "synth_stable":          generate_synth_stable,
    "synth_trend_break":     generate_synth_trend_break,
    "synth_slow_drift":      generate_synth_slow_drift,
    "synth_fast_switch":     generate_synth_fast_switch,
    "synth_recurring":       generate_synth_recurring,
    "synth_volatility":      generate_synth_volatility,
    "synth_shock_recovery":  generate_synth_shock_recovery,
    "synth_multi_regime":    generate_synth_multi_regime,
}


def generate_all(
    output_dir: Path = None,
    overwrite: bool = False,
) -> Dict[str, Path]:
    """Generate all synthetic datasets and save as CSV.

    Args:
        output_dir: Where to write CSVs. Defaults to data/benchmarks/.
        overwrite: If True, regenerate even if file exists.

    Returns:
        Dict mapping dataset name to file path.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "data" / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {}
    metadata = {}

    for name, gen_fn in SYNTHETIC_DATASETS.items():
        filepath = output_dir / f"{name}.csv"
        if filepath.exists() and not overwrite:
            logger.info(f"  {name}: already exists ({filepath})")
            paths[name] = filepath
            continue

        logger.info(f"  Generating {name}...")
        df = gen_fn()
        df.to_csv(filepath, index=False)
        paths[name] = filepath

        # Metadata
        n_regimes = df["regime"].nunique()
        n_changes = int(df["regime_change"].sum())
        avail = len(df) - 720
        batches_possible = avail // 500
        metadata[name] = {
            "rows": len(df),
            "regimes": n_regimes,
            "regime_changes": n_changes,
            "batches_at_500": batches_possible,
            "regime_sequence": (
                df.groupby((df["regime"] != df["regime"].shift()).cumsum())["regime"]
                .first()
                .tolist()
            ),
        }
        logger.info(
            f"  {name}: {len(df)} rows, {n_regimes} regimes, "
            f"{n_changes} changes, {batches_possible} batches@500"
        )

    # Save metadata
    meta_path = output_dir / "synthetic_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"  Metadata saved to {meta_path}")

    return paths


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("=" * 60)
    print("  Generating Synthetic Regime Datasets")
    print("=" * 60)
    paths = generate_all(overwrite=True)
    print(f"\nGenerated {len(paths)} datasets:")
    for name, path in paths.items():
        df = pd.read_csv(path)
        avail = len(df) - 720
        print(f"  {name:25s}: {len(df):5d} rows, {avail // 500:2d} batches@500")
    print("\nDone.")
