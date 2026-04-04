"""Summarize Run #80 adaptive-gate ablation against fixed-gate baseline.

Compares:
- Adaptive run: benchmarks/results/ablation/vol_gate_adaptive/unified_results_full.csv
- Fixed baseline: benchmarks/results/unified_v2_8pol/unified_results.json

Matrix (matched cells):
  policies={rgtta, rgtta_ewc}
  model=gru_small
  datasets={ETTh1, ETTm1, synth_recurring, synth_shock_recovery}
  horizons={96,192}
  seeds={0,1,2}

Outputs:
- benchmarks/adaptive_gate_run80_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ADAPTIVE_CSV = ROOT / "benchmarks/results/ablation/vol_gate_adaptive/unified_results_full.csv"
FIXED_JSON = ROOT / "benchmarks/results/unified_v2_8pol/unified_results.json"
OUT_JSON = ROOT / "benchmarks/adaptive_gate_run80_summary.json"

DATASETS = {"ETTh1", "ETTm1", "synth_recurring", "synth_shock_recovery"}
HORIZONS = {96, 192}
SEEDS = {0, 1, 2}
POLICIES = {"rgtta", "rgtta_ewc"}
MODEL = "gru_small"


def load_fixed_subset() -> pd.DataFrame:
    obj = json.loads(FIXED_JSON.read_text())
    rows = []
    for e in obj.get("experiments", []):
        if (
            e.get("model") != MODEL
            or e.get("dataset") not in DATASETS
            or e.get("forecast_horizon") not in HORIZONS
            or e.get("seed") not in SEEDS
        ):
            continue

        for pol in POLICIES:
            d = e.get(pol)
            if not isinstance(d, dict):
                continue
            tc = d.get("tier_counts") or {}
            rows.append(
                {
                    "model": e["model"],
                    "dataset": e["dataset"],
                    "horizon": e["forecast_horizon"],
                    "seed": e["seed"],
                    "policy": pol,
                    "mse_mean": float(d.get("mse_mean", float("nan"))),
                    "total_time": float(d.get("total_time", float("nan"))),
                    "tier_high": int(tc.get("high", 0)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    adaptive = pd.read_csv(ADAPTIVE_CSV)
    adaptive = adaptive[
        (adaptive["model"] == MODEL)
        & (adaptive["dataset"].isin(DATASETS))
        & (adaptive["horizon"].isin(HORIZONS))
        & (adaptive["seed"].isin(SEEDS))
        & (adaptive["policy"].isin(POLICIES))
    ].copy()

    fixed = load_fixed_subset()

    keys = ["model", "dataset", "horizon", "seed", "policy"]
    merged = adaptive.merge(fixed, on=keys, suffixes=("_adaptive", "_fixed"))

    by_policy = (
        merged.groupby("policy", as_index=False)
        .agg(
            mse_adaptive=("mse_mean_adaptive", "mean"),
            mse_fixed=("mse_mean_fixed", "mean"),
            time_adaptive=("total_time_adaptive", "mean"),
            time_fixed=("total_time_fixed", "mean"),
            high_adaptive=("tier_high_adaptive", "sum"),
            high_fixed=("tier_high_fixed", "sum"),
        )
    )
    by_policy["mse_delta_pct"] = (by_policy["mse_adaptive"] / by_policy["mse_fixed"] - 1.0) * 100.0
    by_policy["time_delta_pct"] = (by_policy["time_adaptive"] / by_policy["time_fixed"] - 1.0) * 100.0

    overall = {
        "rows_compared": int(len(merged)),
        "mse_adaptive": float(merged["mse_mean_adaptive"].mean()),
        "mse_fixed": float(merged["mse_mean_fixed"].mean()),
        "time_adaptive": float(merged["total_time_adaptive"].mean()),
        "time_fixed": float(merged["total_time_fixed"].mean()),
        "high_adaptive": int(merged["tier_high_adaptive"].sum()),
        "high_fixed": int(merged["tier_high_fixed"].sum()),
        "mse_delta_pct": float((merged["mse_mean_adaptive"].mean() / merged["mse_mean_fixed"].mean() - 1.0) * 100.0),
        "time_delta_pct": float((merged["total_time_adaptive"].mean() / merged["total_time_fixed"].mean() - 1.0) * 100.0),
        "adaptive_better_cells": int((merged["mse_mean_adaptive"] < merged["mse_mean_fixed"]).sum()),
        "total_cells": int(len(merged)),
    }

    per_dataset = (
        merged.groupby(["policy", "dataset"], as_index=False)
        .agg(mse_adaptive=("mse_mean_adaptive", "mean"), mse_fixed=("mse_mean_fixed", "mean"))
    )
    per_dataset["mse_delta_pct"] = (per_dataset["mse_adaptive"] / per_dataset["mse_fixed"] - 1.0) * 100.0

    payload = {
        "run": "80",
        "matrix": {
            "model": MODEL,
            "policies": sorted(POLICIES),
            "datasets": sorted(DATASETS),
            "horizons": sorted(HORIZONS),
            "seeds": sorted(SEEDS),
        },
        "overall": overall,
        "by_policy": by_policy.to_dict(orient="records"),
        "per_dataset": per_dataset.to_dict(orient="records"),
    }

    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
