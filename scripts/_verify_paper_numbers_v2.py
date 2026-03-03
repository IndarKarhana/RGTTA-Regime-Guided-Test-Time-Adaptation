#!/usr/bin/env python3
"""
Verify every numerical claim in paper/main.tex against actual Run #72 benchmark data.
Structure: experiments[] → each has model, dataset, forecast_horizon, seed, and nested policy dicts.
"""
import json
import os
import math
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(BASE, "benchmarks", "results", "unified_v2_8pol", "unified_results.json")

with open(JSON_PATH) as f:
    data = json.load(f)

exps = data["experiments"]
POLICIES = ["tta", "ewc", "dynatta", "rgtta", "rgtta_ewc", "rgtta_dynatta"]

print("=" * 80)
print("PAPER NUMBER VERIFICATION — Run #72")
print("=" * 80)
print(f"Loaded {len(exps)} experiment records")

# ─────────────────────────────────────────────────────────
# Build seed-averaged MSE: key=(policy, model, dataset, horizon) → avg MSE
# ─────────────────────────────────────────────────────────
raw = defaultdict(list)  # (policy, model, dataset, horizon) → [mse_per_seed]
time_raw = defaultdict(list)

for e in exps:
    m = e["model"]
    d = e["dataset"]
    h = e["forecast_horizon"]
    s = e["seed"]
    for p in POLICIES:
        if p in e:
            raw[(p, m, d, h)].append(e[p]["mse_mean"])
            time_raw[(p, m, d, h)].append(e[p]["total_time"])

seed_avg = {k: sum(v) / len(v) for k, v in raw.items()}
time_avg = {k: sum(v) / len(v) for k, v in time_raw.items()}

# Unique combos (model, dataset, horizon)
combos = set()
for (p, m, d, h) in seed_avg:
    combos.add((m, d, h))

n_combos = len(combos)
print(f"Unique (model,dataset,horizon) combos: {n_combos}")
print(f"Seed-averaged entries per policy: {len(seed_avg) // len(POLICIES)}")

# ─────────────────────────────────────────────────────────
# CLAIM 1: 672 experiments total
# ─────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"CLAIM: 672 experiments total")
print(f"ACTUAL: {len(exps)}")
print(f"{'✅' if len(exps) == 672 else '❌'}")

print(f"\nCLAIM: 224 seed-averaged experiments")
print(f"ACTUAL: {n_combos}")
print(f"{'✅' if n_combos == 224 else '❌'}")

# ─────────────────────────────────────────────────────────
# TABLE 1: Win counts
# ─────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("TABLE 1: Win counts")

wins = defaultdict(int)
for combo in combos:
    m, d, h = combo
    best_p, best_mse = None, float("inf")
    for p in POLICIES:
        mse = seed_avg.get((p, m, d, h), float("inf"))
        if mse < best_mse:
            best_mse = mse
            best_p = p
    wins[best_p] += 1

claimed_wins = {"tta": 46, "ewc": 13, "dynatta": 9, "rgtta": 65, "rgtta_ewc": 68, "rgtta_dynatta": 23}
print(f"{'Policy':<20} {'Claimed':>8} {'Actual':>8} {'Match':>6}")
print("-" * 45)
for p in POLICIES:
    c = claimed_wins[p]
    a = wins[p]
    print(f"{p:<20} {c:>8} {a:>8} {'✅' if a == c else '❌'}")

rg_total = wins["rgtta"] + wins["rgtta_ewc"] + wins["rgtta_dynatta"]
rg_pct = rg_total / n_combos * 100
print(f"\nRG total: claimed=156 (69.6%)  actual={rg_total} ({rg_pct:.1f}%) {'✅' if rg_total == 156 else '❌'}")

# ─────────────────────────────────────────────────────────
# TABLE 2: Pairwise comparisons
# ─────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("TABLE 2: Pairwise regime-guidance effect")

def pairwise(base_p, rg_p):
    diffs = []
    rg_wins = 0
    for combo in combos:
        m, d, h = combo
        bk = (base_p, m, d, h)
        rk = (rg_p, m, d, h)
        if bk in seed_avg and rk in seed_avg:
            b_mse = seed_avg[bk]
            r_mse = seed_avg[rk]
            if b_mse > 0:
                diffs.append((r_mse - b_mse) / b_mse)
                if r_mse < b_mse:
                    rg_wins += 1
    avg_d = sum(diffs) / len(diffs) * 100
    sorted_d = sorted(diffs)
    n = len(sorted_d)
    med_d = sorted_d[n // 2] * 100 if n % 2 == 1 else (sorted_d[n//2 - 1] + sorted_d[n//2]) / 2 * 100
    return avg_d, med_d, rg_wins, len(diffs)

pairs = [
    ("tta", "rgtta", -5.7, -5.1, 150, 67.0),
    ("ewc", "rgtta_ewc", -14.1, -10.0, 169, 75.4),
    ("dynatta", "rgtta_dynatta", 0.5, -3.8, 139, 62.1),
]

for base_p, rg_p, c_avg, c_med, c_wins, c_pct in pairs:
    avg, med, w, total = pairwise(base_p, rg_p)
    print(f"\n{base_p} → {rg_p}:")
    print(f"  ΔMSEavg: claimed={c_avg:+.1f}%  actual={avg:+.1f}%  {'✅' if abs(avg - c_avg) < 0.5 else '❌'}")
    print(f"  ΔMSEmed: claimed={c_med:+.1f}%  actual={med:+.1f}%  {'✅' if abs(med - c_med) < 0.5 else '❌'}")
    print(f"  RG wins: claimed={c_wins}/{total} ({c_pct}%)  actual={w}/{total} ({w/total*100:.1f}%)  {'✅' if w == c_wins else '❌'}")

# ─────────────────────────────────────────────────────────
# TABLE 3: MSE by model architecture
# ─────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("TABLE 3: Average MSE by policy × model (all 14 ds × 4 horizons)")

model_mse = defaultdict(lambda: defaultdict(list))
for (p, m, d, h), mse in seed_avg.items():
    model_mse[p][m].append(mse)

claimed_t3 = {
    ("tta", "gru_small"): 15458, ("tta", "itransformer"): 21075, ("tta", "patchtst"): 717298, ("tta", "dlinear"): 17855,
    ("ewc", "gru_small"): 16794, ("ewc", "itransformer"): 24141, ("ewc", "patchtst"): 813314, ("ewc", "dlinear"): 18258,
    ("dynatta", "gru_small"): 16076, ("dynatta", "itransformer"): 20808, ("dynatta", "patchtst"): 874257, ("dynatta", "dlinear"): 17185,
    ("rgtta", "gru_small"): 14092, ("rgtta", "itransformer"): 19202, ("rgtta", "patchtst"): 718642, ("rgtta", "dlinear"): 18782,
    ("rgtta_ewc", "gru_small"): 14047, ("rgtta_ewc", "itransformer"): 18851, ("rgtta_ewc", "patchtst"): 721305, ("rgtta_ewc", "dlinear"): 18752,
    ("rgtta_dynatta", "gru_small"): 16787, ("rgtta_dynatta", "itransformer"): 22619, ("rgtta_dynatta", "patchtst"): 868742, ("rgtta_dynatta", "dlinear"): 16739,
}

print(f"{'Policy':<18} {'Model':<15} {'Claimed':>10} {'Actual':>10} {'Diff%':>8} {'Match':>6}")
print("-" * 70)
for p in POLICIES:
    for m in ["gru_small", "itransformer", "patchtst", "dlinear"]:
        c = claimed_t3.get((p, m), 0)
        vals = model_mse[p][m]
        a = sum(vals) / len(vals) if vals else 0
        d_pct = (a - c) / c * 100 if c > 0 else 0
        print(f"{p:<18} {m:<15} {c:>10,} {a:>10,.0f} {d_pct:>7.1f}% {'✅' if abs(d_pct) < 2.0 else '❌'}")

# Architecture improvement percentages
print(f"\nArchitecture improvement claims:")
for model, rgp, basep, c_pct in [
    ("gru_small", "rgtta", "tta", -8.8),
    ("gru_small", "rgtta_ewc", "tta", -9.1),
    ("itransformer", "rgtta", "tta", -8.9),
    ("itransformer", "rgtta_ewc", "tta", -10.6),
    ("dlinear", "rgtta_dynatta", "dynatta", -2.6),
]:
    b = sum(model_mse[basep][model]) / len(model_mse[basep][model])
    r = sum(model_mse[rgp][model]) / len(model_mse[rgp][model])
    a_pct = (r - b) / b * 100
    print(f"  {rgp} vs {basep} on {model}: claimed={c_pct:+.1f}% actual={a_pct:+.1f}% {'✅' if abs(a_pct - c_pct) < 0.5 else '❌'}")

# PatchTST claim: "TTA edges RG-TTA by <0.2%"
tta_pt = sum(model_mse["tta"]["patchtst"]) / len(model_mse["tta"]["patchtst"])
rgtta_pt = sum(model_mse["rgtta"]["patchtst"]) / len(model_mse["rgtta"]["patchtst"])
pct_diff_pt = (rgtta_pt - tta_pt) / tta_pt * 100
print(f"  PatchTST TTA vs RG-TTA: diff={pct_diff_pt:+.2f}% (claimed <0.2%) {'✅' if abs(pct_diff_pt) < 0.3 else '❌'}")

# ─────────────────────────────────────────────────────────
# Statistical ranking (Friedman)
# ─────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("STATISTICAL RANKING (Friedman)")

from scipy.stats import rankdata
import numpy as np

rank_sums = defaultdict(float)
rank_count = 0

for combo in combos:
    m, d, h = combo
    vals = {}
    for p in POLICIES:
        k = (p, m, d, h)
        if k in seed_avg:
            vals[p] = seed_avg[k]
    if len(vals) == 6:
        # Rank: lower MSE = better (rank 1)
        policies_sorted = sorted(vals.keys(), key=lambda x: vals[x])
        for rank_idx, p in enumerate(policies_sorted, 1):
            rank_sums[p] += rank_idx
        rank_count += 1

claimed_ranks = {"rgtta": 2.46, "rgtta_ewc": 2.51, "tta": 2.98, "ewc": 4.02, "rgtta_dynatta": 4.34, "dynatta": 4.69}
print(f"\nAvg ranks (N={rank_count}):")
for p in ["rgtta", "rgtta_ewc", "tta", "ewc", "rgtta_dynatta", "dynatta"]:
    a = rank_sums[p] / rank_count
    c = claimed_ranks[p]
    print(f"  {p:<18} claimed={c:.2f}  actual={a:.2f}  {'✅' if abs(a - c) < 0.05 else '❌'}")

N = rank_count
k = 6
rank_avgs = [rank_sums[p] / N for p in POLICIES]
chi2 = 12 * N / (k * (k + 1)) * sum((r - (k + 1) / 2) ** 2 for r in rank_avgs)
print(f"\nFriedman χ²: claimed=301.95  actual={chi2:.2f}  {'✅' if abs(chi2 - 301.95) < 5 else '❌'}")

q_alpha = 2.850  # Nemenyi q_alpha for k=6, alpha=0.05
CD = q_alpha * math.sqrt(k * (k + 1) / (6 * N))
print(f"Nemenyi CD: claimed=0.50  actual={CD:.2f}  {'✅' if abs(CD - 0.50) < 0.05 else '❌'}")

# ─────────────────────────────────────────────────────────
# TABLE 5: Real-world MSE
# ─────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("TABLE 5: Real-world MSE")

real_ds = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Exchange"]

rw_mse = defaultdict(lambda: defaultdict(list))
for (p, m, d, h), mse in seed_avg.items():
    if d in real_ds:
        rw_mse[p][d].append(mse)

claimed_rw = {
    ("tta", "ETTh1"): 56.15, ("tta", "ETTh2"): 92.54, ("tta", "ETTm1"): 20.88,
    ("tta", "ETTm2"): 40.53, ("tta", "Weather"): 138.99, ("tta", "Exchange"): 0.01,
    ("ewc", "ETTh1"): 63.80, ("ewc", "ETTh2"): 102.21, ("ewc", "ETTm1"): 23.40,
    ("ewc", "ETTm2"): 42.89, ("ewc", "Weather"): 147.67, ("ewc", "Exchange"): 0.01,
    ("dynatta", "ETTh1"): 89.29, ("dynatta", "ETTh2"): 130.18, ("dynatta", "ETTm1"): 24.62,
    ("dynatta", "ETTm2"): 45.49, ("dynatta", "Weather"): 164.37, ("dynatta", "Exchange"): 0.01,
    ("rgtta", "ETTh1"): 53.74, ("rgtta", "ETTh2"): 78.99, ("rgtta", "ETTm1"): 18.05,
    ("rgtta", "ETTm2"): 37.69, ("rgtta", "Weather"): 145.21, ("rgtta", "Exchange"): 0.01,
    ("rgtta_ewc", "ETTh1"): 52.33, ("rgtta_ewc", "ETTh2"): 78.41, ("rgtta_ewc", "ETTm1"): 18.14,
    ("rgtta_ewc", "ETTm2"): 37.10, ("rgtta_ewc", "Weather"): 151.39, ("rgtta_ewc", "Exchange"): 0.01,
    ("rgtta_dynatta", "ETTh1"): 79.90, ("rgtta_dynatta", "ETTh2"): 127.21, ("rgtta_dynatta", "ETTm1"): 25.22,
    ("rgtta_dynatta", "ETTm2"): 41.79, ("rgtta_dynatta", "Weather"): 175.31, ("rgtta_dynatta", "Exchange"): 0.01,
}

print(f"{'Policy':<18} {'Dataset':<12} {'Claimed':>10} {'Actual':>10} {'Match':>6}")
print("-" * 60)
for p in POLICIES:
    for d in real_ds:
        c = claimed_rw.get((p, d), 0)
        vals = rw_mse[p][d]
        a = sum(vals) / len(vals) if vals else 0
        if c > 0.05:
            diff = abs(a - c) / c * 100
        else:
            diff = abs(a - c) * 10000  # scale for tiny values
        match = "✅" if diff < 2.0 or abs(a - c) < 0.005 else "❌"
        print(f"{p:<18} {d:<12} {c:>10.2f} {a:>10.2f} {match:>6}")

# ─────────────────────────────────────────────────────────
# TABLE 6: Category analysis
# ─────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("TABLE 6: Category MSE")

ett_ds = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
weather_exch = ["Weather", "Exchange"]
synth_recurring_ds = ["synth_recurring", "synth_fast_switch", "synth_multi_regime"]
synth_shock_ds = ["synth_stable", "synth_trend_break", "synth_slow_drift", "synth_volatility", "synth_shock_recovery"]

categories = {
    "ETT (4)": ett_ds,
    "Weather+Exch. (2)": weather_exch,
    "Synth-Recurring (3)": synth_recurring_ds,
    "Synth-Shock (5)": synth_shock_ds,
}

claimed_cat = {
    ("tta", "ETT (4)"): 52.53, ("ewc", "ETT (4)"): 58.08, ("dynatta", "ETT (4)"): 72.40,
    ("rgtta", "ETT (4)"): 47.12, ("rgtta_ewc", "ETT (4)"): 46.50, ("rgtta_dynatta", "ETT (4)"): 68.53,
    ("tta", "Weather+Exch. (2)"): 69.50, ("ewc", "Weather+Exch. (2)"): 73.84,
    ("dynatta", "Weather+Exch. (2)"): 82.19,
    ("rgtta", "Weather+Exch. (2)"): 72.61, ("rgtta_ewc", "Weather+Exch. (2)"): 75.70,
    ("rgtta_dynatta", "Weather+Exch. (2)"): 87.66,
    ("tta", "Synth-Recurring (3)"): 292818, ("ewc", "Synth-Recurring (3)"): 338101,
    ("dynatta", "Synth-Recurring (3)"): 366484,
    ("rgtta", "Synth-Recurring (3)"): 300952, ("rgtta_ewc", "Synth-Recurring (3)"): 303245,
    ("rgtta_dynatta", "Synth-Recurring (3)"): 357825,
    ("tta", "Synth-Shock (5)"): 364420, ("ewc", "Synth-Shock (5)"): 407818,
    ("dynatta", "Synth-Shock (5)"): 429847,
    ("rgtta", "Synth-Shock (5)"): 358865, ("rgtta_ewc", "Synth-Shock (5)"): 359054,
    ("rgtta_dynatta", "Synth-Shock (5)"): 432636,
}

cat_mse = defaultdict(lambda: defaultdict(list))
for (p, m, d, h), mse in seed_avg.items():
    for cn, cds in categories.items():
        if d in cds:
            cat_mse[p][cn].append(mse)

print(f"{'Policy':<18} {'Category':<22} {'Claimed':>12} {'Actual':>12} {'Diff%':>8} {'Match':>6}")
print("-" * 80)
for p in POLICIES:
    for cn in categories:
        c = claimed_cat.get((p, cn), 0)
        vals = cat_mse[p][cn]
        a = sum(vals) / len(vals) if vals else 0
        d_pct = (a - c) / c * 100 if c > 0 else 0
        print(f"{p:<18} {cn:<22} {c:>12,.0f} {a:>12,.0f} {d_pct:>7.1f}% {'✅' if abs(d_pct) < 2 else '❌'}")

# ETT improvement claims
tta_ett = sum(cat_mse["tta"]["ETT (4)"]) / len(cat_mse["tta"]["ETT (4)"])
rgtta_ett = sum(cat_mse["rgtta"]["ETT (4)"]) / len(cat_mse["rgtta"]["ETT (4)"])
rgewc_ett = sum(cat_mse["rgtta_ewc"]["ETT (4)"]) / len(cat_mse["rgtta_ewc"]["ETT (4)"])
print(f"\nETT improvements vs TTA:")
print(f"  RG-TTA: claimed=-10.3% actual={(rgtta_ett-tta_ett)/tta_ett*100:+.1f}%")
print(f"  RG-EWC: claimed=-11.5% actual={(rgewc_ett-tta_ett)/tta_ett*100:+.1f}%")

# ─────────────────────────────────────────────────────────
# Per-dataset win rates
# ─────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("PER-DATASET WIN RATES")

rg_set = {"rgtta", "rgtta_ewc", "rgtta_dynatta"}
ds_wins = defaultdict(lambda: {"rg": 0, "total": 0})

for combo in combos:
    m, d, h = combo
    best_p, best_mse = None, float("inf")
    for p in POLICIES:
        mse = seed_avg.get((p, m, d, h), float("inf"))
        if mse < best_mse:
            best_mse = mse
            best_p = p
    ds_wins[d]["total"] += 1
    if best_p in rg_set:
        ds_wins[d]["rg"] += 1

claimed_dswins = {
    "synth_recurring": 100, "ETTh2": 88, "synth_trend_break": 88, "ETTh1": 81,
    "Exchange": 81, "synth_shock_recovery": 81, "synth_slow_drift": 81,
    "synth_fast_switch": 75, "ETTm2": 69, "ETTm1": 62, "synth_multi_regime": 62,
    "synth_stable": 62, "Weather": 38, "synth_volatility": 6,
}

print(f"{'Dataset':<25} {'Claimed':>8} {'Actual':>8} {'Match':>6}")
print("-" * 50)
for d in sorted(claimed_dswins, key=lambda x: -claimed_dswins[x]):
    c = claimed_dswins[d]
    info = ds_wins[d]
    a = info["rg"] / info["total"] * 100 if info["total"] > 0 else 0
    print(f"{d:<25} {c:>7}% {a:>7.0f}% {'✅' if abs(a - c) < 2 else '❌'}")

# Real-world vs synthetic split
rw_set = set(real_ds)
synth_set = set(d for d in ds_wins if d.startswith("synth_"))

rw_rg, rw_tot = 0, 0
sy_rg, sy_tot = 0, 0
for combo in combos:
    m, d, h = combo
    best_p, best_mse = None, float("inf")
    for p in POLICIES:
        mse = seed_avg.get((p, m, d, h), float("inf"))
        if mse < best_mse:
            best_mse = mse
            best_p = p
    if d in rw_set:
        rw_tot += 1
        if best_p in rg_set:
            rw_rg += 1
    elif d in synth_set:
        sy_tot += 1
        if best_p in rg_set:
            sy_rg += 1

print(f"\nReal-world: claimed=67/96 (69.8%) actual={rw_rg}/{rw_tot} ({rw_rg/rw_tot*100:.1f}%)")
print(f"Synthetic:  claimed=89/128 (69.5%) actual={sy_rg}/{sy_tot} ({sy_rg/sy_tot*100:.1f}%)")

# ─────────────────────────────────────────────────────────
# TABLE 7: Computational cost
# ─────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("TABLE 7: Computational cost (avg total_time per experiment)")

# Seed-averaged time by (policy, model)
time_pm = defaultdict(lambda: defaultdict(list))
time_all = defaultdict(list)
for (p, m, d, h), t in time_avg.items():
    time_pm[p][m].append(t)
    time_all[p].append(t)

claimed_time = {
    ("tta", "gru_small"): 106.3, ("tta", "itransformer"): 33.5, ("tta", "patchtst"): 381.1, ("tta", "dlinear"): 16.1,
    ("ewc", "gru_small"): 253.2, ("ewc", "itransformer"): 42.3, ("ewc", "patchtst"): 312.8, ("ewc", "dlinear"): 17.0,
    ("dynatta", "gru_small"): 121.9, ("dynatta", "itransformer"): 34.7, ("dynatta", "patchtst"): 411.8, ("dynatta", "dlinear"): 17.2,
    ("rgtta", "gru_small"): 118.7, ("rgtta", "itransformer"): 38.7, ("rgtta", "patchtst"): 335.7, ("rgtta", "dlinear"): 14.5,
    ("rgtta_ewc", "gru_small"): 286.7, ("rgtta_ewc", "itransformer"): 55.0, ("rgtta_ewc", "patchtst"): 360.2, ("rgtta_ewc", "dlinear"): 19.1,
    ("rgtta_dynatta", "gru_small"): 129.0, ("rgtta_dynatta", "itransformer"): 36.9, ("rgtta_dynatta", "patchtst"): 421.2, ("rgtta_dynatta", "dlinear"): 17.0,
}
claimed_overall = {"tta": 134.3, "ewc": 156.4, "dynatta": 146.4, "rgtta": 126.9, "rgtta_ewc": 180.3, "rgtta_dynatta": 151.0}

print(f"{'Policy':<18} {'Model':<15} {'Claimed':>10} {'Actual':>10} {'Diff%':>8} {'Match':>6}")
print("-" * 70)
for p in POLICIES:
    for m in ["gru_small", "itransformer", "patchtst", "dlinear"]:
        c = claimed_time.get((p, m), 0)
        vals = time_pm[p][m]
        a = sum(vals) / len(vals) if vals else 0
        d_pct = (a - c) / c * 100 if c > 0 else 0
        print(f"{p:<18} {m:<15} {c:>10.1f} {a:>10.1f} {d_pct:>7.1f}% {'✅' if abs(d_pct) < 5.0 else '❌'}")
    # Overall
    oc = claimed_overall[p]
    oa = sum(time_all[p]) / len(time_all[p]) if time_all[p] else 0
    od = (oa - oc) / oc * 100
    print(f"  {'→ OVERALL':<15}          {oc:>10.1f} {oa:>10.1f} {od:>7.1f}% {'✅' if abs(od) < 5.0 else '❌'}")

# Speed claims
tta_t = sum(time_all["tta"]) / len(time_all["tta"])
rgtta_t = sum(time_all["rgtta"]) / len(time_all["rgtta"])
ewc_t = sum(time_all["ewc"]) / len(time_all["ewc"])
rgewc_t = sum(time_all["rgtta_ewc"]) / len(time_all["rgtta_ewc"])
print(f"\nRG-TTA vs TTA speed: claimed=-5.5% actual={(rgtta_t-tta_t)/tta_t*100:+.1f}%")
print(f"RG-EWC vs EWC speed: claimed=+15.3% actual={(rgewc_t-ewc_t)/ewc_t*100:+.1f}%")

# ─────────────────────────────────────────────────────────
# PER-HORIZON (Appendix Table 11)
# ─────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("TABLE 11 (Appendix): Per-horizon MSE")

hz_mse = defaultdict(lambda: defaultdict(list))
for (p, m, d, h), mse in seed_avg.items():
    hz_mse[p][h].append(mse)

claimed_hz = {
    ("tta", 96): 176830, ("tta", 192): 187295, ("tta", 336): 193082, ("tta", 720): 214479,
    ("ewc", 96): 203296, ("ewc", 192): 211448, ("ewc", 336): 218073, ("ewc", 720): 239690,
    ("dynatta", 96): 216969, ("dynatta", 192): 227041, ("dynatta", 336): 232179, ("dynatta", 720): 252137,
    ("rgtta", 96): 170795, ("rgtta", 192): 184887, ("rgtta", 336): 191360, ("rgtta", 720): 223676,
    ("rgtta_ewc", 96): 168993, ("rgtta_ewc", 192): 184932, ("rgtta_ewc", 336): 192007, ("rgtta_ewc", 720): 227024,
    ("rgtta_dynatta", 96): 212964, ("rgtta_dynatta", 192): 224449, ("rgtta_dynatta", 336): 234977, ("rgtta_dynatta", 720): 252499,
}

print(f"{'Policy':<18} {'H':>5} {'Claimed':>12} {'Actual':>12} {'Diff%':>8} {'Match':>6}")
print("-" * 65)
for p in POLICIES:
    for h in [96, 192, 336, 720]:
        c = claimed_hz.get((p, h), 0)
        vals = hz_mse[p][h]
        a = sum(vals) / len(vals) if vals else 0
        d_pct = (a - c) / c * 100 if c > 0 else 0
        print(f"{p:<18} {h:>5} {c:>12,} {a:>12,.0f} {d_pct:>7.1f}% {'✅' if abs(d_pct) < 2 else '❌'}")

# ─────────────────────────────────────────────────────────
# ABSTRACT: "RG-TTA reduces MSE by 4.4% vs TTA on real-world"
# ─────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("ABSTRACT CLAIMS")

tta_rw_vals = []
rgtta_rw_vals = []
for (p, m, d, h), mse in seed_avg.items():
    if d in rw_set:
        if p == "tta":
            tta_rw_vals.append(mse)
        elif p == "rgtta":
            rgtta_rw_vals.append(mse)

tta_rw_avg = sum(tta_rw_vals) / len(tta_rw_vals)
rgtta_rw_avg = sum(rgtta_rw_vals) / len(rgtta_rw_vals)
rw_impr = (rgtta_rw_avg - tta_rw_avg) / tta_rw_avg * 100
print(f"RG-TTA vs TTA on real-world (flat avg): claimed=-4.4% actual={rw_impr:+.1f}%")

# Maybe it's the per-dataset relative improvement averaged?
per_ds_rel = []
for d in real_ds:
    t_vals = [seed_avg[(p, m, d, h)] for (p, m, d2, h) in seed_avg if p == "tta" and d2 == d]
    r_vals = [seed_avg[(p, m, d, h)] for (p, m, d2, h) in seed_avg if p == "rgtta" and d2 == d]
    if t_vals and r_vals:
        t_avg = sum(t_vals) / len(t_vals)
        r_avg = sum(r_vals) / len(r_vals)
        per_ds_rel.append((r_avg - t_avg) / t_avg * 100)
avg_rel = sum(per_ds_rel) / len(per_ds_rel) if per_ds_rel else 0
print(f"RG-TTA vs TTA on real-world (per-ds avg relative): {avg_rel:+.1f}%")

# ─────────────────────────────────────────────────────────
# Weather claim: "TTA outperforms RG-TTA by 4.5%"
# ─────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("SPECIFIC CLAIMS")

tta_w = sum(rw_mse["tta"]["Weather"]) / len(rw_mse["tta"]["Weather"])
rgtta_w = sum(rw_mse["rgtta"]["Weather"]) / len(rw_mse["rgtta"]["Weather"])
w_diff = (rgtta_w - tta_w) / tta_w * 100
print(f"Weather: TTA outperforms RG-TTA by claimed=4.5% actual={w_diff:+.1f}% {'✅' if abs(w_diff - 4.5) < 1.0 else '❌'}")

# synth_volatility: "RG-TTA wins only 6%"
v_info = ds_wins.get("synth_volatility", {"rg": 0, "total": 0})
v_pct = v_info["rg"] / v_info["total"] * 100 if v_info["total"] > 0 else 0
print(f"synth_volatility RG wins: claimed=6% actual={v_pct:.0f}% {'✅' if abs(v_pct - 6) < 2 else '❌'}")

# ─────────────────────────────────────────────────────────
# Dataset row counts
# ─────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("DATASET ROW COUNTS")
import pandas as pd
data_dir = os.path.join(BASE, "data", "benchmarks")
ds_rows = {"ETTh1": 17420, "ETTh2": 17420, "ETTm1": 69680, "ETTm2": 69680, "Weather": 52696, "Exchange": 7588}

for name, claimed in ds_rows.items():
    csv = name + ".csv" if name != "Exchange" else "exchange_rate.csv"
    path = os.path.join(data_dir, csv)
    if os.path.exists(path):
        actual = len(pd.read_csv(path))
        print(f"  {name}: claimed={claimed} actual={actual} {'✅' if actual == claimed else '❌'}")
    else:
        print(f"  {name}: FILE NOT FOUND at {path}")

# ─────────────────────────────────────────────────────────
# CONCLUSION: match the repeated claims
# ─────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("CONCLUSION CLAIMS (should match above)")
print(f"  RG-TTA -5.7% MSE overall (Table 2 avg): see pairwise above")
print(f"  RG-TTA -5.5% time (Table 7): see speed above")
print(f"  TTA 67% wins: {pairwise('tta', 'rgtta')[2]}")
print(f"  EWC 75% wins: {pairwise('ewc', 'rgtta_ewc')[2]}")
print(f"  DynaTTA 62% wins: {pairwise('dynatta', 'rgtta_dynatta')[2]}")

print(f"\n{'='*80}")
print("VERIFICATION COMPLETE — review all ❌ marks above")
print("=" * 80)
