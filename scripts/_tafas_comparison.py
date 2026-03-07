"""Quick analysis: TAFAS vs RG-TAFAS from Run #72."""
import json
import numpy as np
from collections import defaultdict

with open("benchmarks/results/unified_v2_8pol/unified_results.json") as f:
    data = json.load(f)

tafas_wins = 0
rgtta_tafas_wins = 0
ties = 0
total = 0
details = []

for e in data["experiments"]:
    if "tafas" in e and "rgtta_tafas" in e:
        t_mse = e["tafas"].get("mse_mean", float("inf"))
        r_mse = e["rgtta_tafas"].get("mse_mean", float("inf"))
        if t_mse == 0 and r_mse == 0:
            continue
        total += 1
        model = e.get("model", "?")
        ds = e.get("dataset", "?")
        h = e.get("forecast_horizon", "?")
        s = e.get("seed", "?")
        pct = (r_mse - t_mse) / t_mse * 100 if t_mse > 0 else 0
        if r_mse < t_mse:
            rgtta_tafas_wins += 1
        elif t_mse < r_mse:
            tafas_wins += 1
        else:
            ties += 1
        details.append((model, ds, h, s, t_mse, r_mse, pct))

print(f"=== TAFAS vs RG-TAFAS (Run #72, {total} head-to-head) ===")
print(f"TAFAS wins:      {tafas_wins} ({tafas_wins/total*100:.1f}%)")
print(f"RG-TAFAS wins:   {rgtta_tafas_wins} ({rgtta_tafas_wins/total*100:.1f}%)")
print(f"Ties:            {ties}")

# By model
by_model = defaultdict(lambda: {"tafas": 0, "rgtta_tafas": 0, "total": 0, "pcts": []})
for model, ds, h, s, t, r, pct in details:
    by_model[model]["total"] += 1
    by_model[model]["pcts"].append(pct)
    if r < t:
        by_model[model]["rgtta_tafas"] += 1
    elif t < r:
        by_model[model]["tafas"] += 1

print(f"\n=== By Model ===")
for m in sorted(by_model):
    d = by_model[m]
    avg_pct = np.mean(d["pcts"])
    print(f"  {m:>15}: RG-TAFAS {d['rgtta_tafas']}/{d['total']} ({d['rgtta_tafas']/d['total']*100:.0f}%), "
          f"TAFAS {d['tafas']}/{d['total']} ({d['tafas']/d['total']*100:.0f}%), avg MSE Δ: {avg_pct:+.1f}%")

# By horizon
by_h = defaultdict(lambda: {"tafas": 0, "rgtta_tafas": 0, "total": 0, "pcts": []})
for model, ds, h, s, t, r, pct in details:
    by_h[h]["total"] += 1
    by_h[h]["pcts"].append(pct)
    if r < t:
        by_h[h]["rgtta_tafas"] += 1
    elif t < r:
        by_h[h]["tafas"] += 1

print(f"\n=== By Horizon ===")
for h in sorted(by_h):
    d = by_h[h]
    avg_pct = np.mean(d["pcts"])
    print(f"  H={h:>3}: RG-TAFAS {d['rgtta_tafas']}/{d['total']} ({d['rgtta_tafas']/d['total']*100:.0f}%), "
          f"TAFAS {d['tafas']}/{d['total']} ({d['tafas']/d['total']*100:.0f}%), avg MSE Δ: {avg_pct:+.1f}%")

# By dataset category
by_cat = defaultdict(lambda: {"tafas": 0, "rgtta_tafas": 0, "total": 0, "pcts": []})
for model, ds, h, s, t, r, pct in details:
    cat = "synthetic" if ds.startswith("synth_") else "real-world"
    by_cat[cat]["total"] += 1
    by_cat[cat]["pcts"].append(pct)
    if r < t:
        by_cat[cat]["rgtta_tafas"] += 1
    elif t < r:
        by_cat[cat]["tafas"] += 1

print(f"\n=== By Category ===")
for cat in sorted(by_cat):
    d = by_cat[cat]
    avg_pct = np.mean(d["pcts"])
    print(f"  {cat:>12}: RG-TAFAS {d['rgtta_tafas']}/{d['total']} ({d['rgtta_tafas']/d['total']*100:.0f}%), "
          f"TAFAS {d['tafas']}/{d['total']} ({d['tafas']/d['total']*100:.0f}%), avg MSE Δ: {avg_pct:+.1f}%")

# Speed
tafas_times = []
rgtta_tafas_times = []
for e in data["experiments"]:
    if "tafas" in e and "rgtta_tafas" in e:
        tt = e["tafas"].get("total_time", 0)
        rt = e["rgtta_tafas"].get("total_time", 0)
        if tt > 0 and rt > 0:
            tafas_times.append(tt)
            rgtta_tafas_times.append(rt)

print(f"\n=== Speed ===")
print(f"  TAFAS avg time:      {np.mean(tafas_times):.1f}s")
print(f"  RG-TAFAS avg time:   {np.mean(rgtta_tafas_times):.1f}s")
print(f"  Speedup:             {np.mean(tafas_times)/np.mean(rgtta_tafas_times):.2f}x")
