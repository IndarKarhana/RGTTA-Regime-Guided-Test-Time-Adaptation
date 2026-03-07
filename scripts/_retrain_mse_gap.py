"""Deep MSE gap analysis: retrain vs RG policies across all datasets, models, horizons."""
import json, numpy as np, math
from collections import defaultdict

with open('benchmarks/results/unified_retrain_v6/unified_results.json') as f:
    rd = json.load(f)
with open('benchmarks/results/unified_v2_8pol/unified_results.json') as f:
    td = json.load(f)

POL_TTA = ['tta', 'ewc', 'dynatta', 'rgtta', 'rgtta_ewc', 'rgtta_dynatta']
ALL = ['retrain'] + POL_TTA
RG = ['rgtta', 'rgtta_ewc', 'rgtta_dynatta']

R = {}
for e in rd['experiments']:
    mse = e['retrain']['mse_mean']
    if mse is None or (isinstance(mse, float) and math.isnan(mse)):
        continue
    k = (e['model'], e['dataset'], int(e['forecast_horizon']), int(e['seed']))
    R.setdefault(k, {})
    R[k]['retrain'] = {'mse': mse, 'time': e['retrain']['total_time']}

for e in td['experiments']:
    k = (e['model'], e['dataset'], int(e['forecast_horizon']), int(e['seed']))
    R.setdefault(k, {})
    for p in POL_TTA:
        if p in e and isinstance(e[p], dict):
            R[k][p] = {'mse': e[p]['mse_mean'], 'time': e[p]['total_time']}

F = {k: v for k, v in R.items() if len(v) == 7}

# Seed-average everything
G = defaultdict(lambda: defaultdict(list))
for (m, d, h, s), v in F.items():
    for p in ALL:
        G[(m, d, h)][p].append(v[p]['mse'])

SA = {}  # seed-averaged: (model, dataset, horizon) -> {policy: mean_mse}
for gk in G:
    SA[gk] = {p: np.mean(G[gk][p]) for p in ALL}

# =====================================================================
# 1. FULL TABLE: Per-dataset, per-model MSE and % gap
# =====================================================================
print("=" * 120)
print("FULL MSE TABLE: retrain vs best-RG  (seed-averaged, 3 seeds)")
print("  %gap = (best_RG - retrain) / retrain × 100.  Negative = RG better.")
print("=" * 120)

DATASETS_ORDER = ['ETTh1','ETTh2','ETTm1','ETTm2','Weather','Exchange',
                  'synth_stable','synth_trend_break','synth_slow_drift','synth_fast_switch',
                  'synth_recurring','synth_volatility','synth_shock_recovery','synth_multi_regime']
MODELS = ['gru_small', 'itransformer', 'patchtst']
HORIZONS = [96, 192, 336, 720]

for ds in DATASETS_ORDER:
    print(f"\n{'─'*120}")
    print(f"  {ds}")
    print(f"{'─'*120}")
    print(f"  {'Model':<14} {'H':>4}  {'Retrain':>12}  {'RG-TTA':>12}  {'RG-EWC':>12}  {'RG-Dyn':>12}  {'BestRG':>12}  {'TTA':>12}  {'%gap':>8}  {'Winner':>10}  {'Speedup':>8}")
    
    for model in MODELS:
        for h in HORIZONS:
            gk = (model, ds, h)
            if gk not in SA:
                continue
            a = SA[gk]
            ret = a['retrain']
            rgtta = a['rgtta']
            rgewc = a['rgtta_ewc']
            rgdyn = a['rgtta_dynatta']
            tta = a['tta']
            best_rg_val = min(rgtta, rgewc, rgdyn)
            best_rg_name = ['RG-TTA', 'RG-EWC', 'RG-Dyn'][[rgtta, rgewc, rgdyn].index(best_rg_val)]
            
            pct = 100 * (best_rg_val - ret) / ret if ret > 0 else 0
            winner = best_rg_name if best_rg_val < ret else 'Retrain'
            
            # Speedup
            sp_key_map = {'RG-TTA': 'rgtta', 'RG-EWC': 'rgtta_ewc', 'RG-Dyn': 'rgtta_dynatta'}
            best_rg_code = sp_key_map[best_rg_name]
            times = [(F[k]['retrain']['time'] / F[k][best_rg_code]['time']) 
                     for k in F if k[0]==model and k[1]==ds and k[2]==h and F[k][best_rg_code]['time']>0]
            sp = f"{np.median(times):.0f}x" if times else "?"
            
            marker = "✅" if pct < 0 else "❌" if pct > 10 else "≈"
            print(f"  {model:<14} {h:>4}  {ret:>12.2f}  {rgtta:>12.2f}  {rgewc:>12.2f}  {rgdyn:>12.2f}  {best_rg_val:>12.2f}  {tta:>12.2f}  {pct:>+7.1f}%  {marker} {winner:<8}  {sp:>8}")

# =====================================================================
# 2. AGGREGATE MSE GAP STATISTICS
# =====================================================================
print(f"\n\n{'='*100}")
print("AGGREGATE MSE GAP: best-RG vs retrain (seed-averaged)")
print("  Negative = RG better.  Grouped by model, dataset type, horizon.")
print("="*100)

all_gaps = []
gaps_by_model = defaultdict(list)
gaps_by_dstype = defaultdict(list)
gaps_by_horizon = defaultdict(list)
gaps_by_ds = defaultdict(list)

SYNTH = set(['synth_stable','synth_trend_break','synth_slow_drift','synth_fast_switch',
             'synth_recurring','synth_volatility','synth_shock_recovery','synth_multi_regime'])
REAL = set(['ETTh1','ETTh2','ETTm1','ETTm2','Weather','Exchange'])

for gk, a in SA.items():
    m, d, h = gk
    ret = a['retrain']
    best_rg = min(a['rgtta'], a['rgtta_ewc'], a['rgtta_dynatta'])
    if ret > 0:
        pct = 100 * (best_rg - ret) / ret
        all_gaps.append(pct)
        gaps_by_model[m].append(pct)
        gaps_by_dstype['Real' if d in REAL else 'Synth'].append(pct)
        gaps_by_horizon[h].append(pct)
        gaps_by_ds[d].append(pct)

def report(label, vals):
    wins = sum(1 for v in vals if v < 0)
    med = np.median(vals)
    mn = np.mean(vals)
    p25, p75 = np.percentile(vals, [25, 75])
    print(f"  {label:<25}: n={len(vals):>3}  median={med:>+7.1f}%  mean={mn:>+7.1f}%  Q1={p25:>+7.1f}%  Q3={p75:>+7.1f}%  RG-wins={wins}/{len(vals)}({100*wins/len(vals):.0f}%)")

print("\nBY MODEL:")
for m in MODELS:
    report(m, gaps_by_model[m])
report("ALL", all_gaps)

print("\nBY DATASET TYPE:")
for dt in ['Real', 'Synth']:
    report(dt, gaps_by_dstype[dt])

print("\nBY HORIZON:")
for h in HORIZONS:
    report(f"H={h}", gaps_by_horizon[h])

print("\nBY DATASET:")
for ds in DATASETS_ORDER:
    if ds in gaps_by_ds:
        report(ds, gaps_by_ds[ds])

# =====================================================================
# 3. SAME ANALYSIS FOR retrain vs EACH RG policy individually
# =====================================================================
print(f"\n\n{'='*100}")
print("INDIVIDUAL RG POLICY GAP vs RETRAIN (seed-averaged)")
print("="*100)

for rg_pol, rg_label in [('rgtta', 'RG-TTA'), ('rgtta_ewc', 'RG-EWC'), ('rgtta_dynatta', 'RG-DynaTTA')]:
    gaps = []
    for gk, a in SA.items():
        ret = a['retrain']
        rg = a[rg_pol]
        if ret > 0:
            gaps.append(100 * (rg - ret) / ret)
    wins = sum(1 for v in gaps if v < 0)
    print(f"\n  {rg_label}:")
    print(f"    median gap: {np.median(gaps):+.1f}%  mean gap: {np.mean(gaps):+.1f}%")
    print(f"    beats retrain: {wins}/{len(gaps)} ({100*wins/len(gaps):.0f}%)")
    print(f"    Q1={np.percentile(gaps,25):+.1f}%  Q3={np.percentile(gaps,75):+.1f}%")
    
    # By model
    for m in MODELS:
        mg = [100*(SA[gk][rg_pol]-SA[gk]['retrain'])/SA[gk]['retrain'] for gk in SA if gk[0]==m and SA[gk]['retrain']>0]
        mw = sum(1 for v in mg if v < 0)
        print(f"      {m:<14}: median={np.median(mg):+.1f}%  wins={mw}/{len(mg)}")

# =====================================================================
# 4. ABSOLUTE MSE COMPARISON TABLE (averaged across seeds and horizons)
# =====================================================================
print(f"\n\n{'='*100}")
print("MEAN MSE BY DATASET (averaged across all horizons & seeds)")
print("="*100)
print(f"  {'Dataset':<22} {'Model':<14} {'Retrain':>10} {'RG-TTA':>10} {'RG-EWC':>10} {'TTA':>10} {'Best':>10} {'Best-is':>10}")

for ds in DATASETS_ORDER:
    for model in MODELS:
        mse_by_pol = defaultdict(list)
        for gk, a in SA.items():
            if gk[0] == model and gk[1] == ds:
                for p in ALL:
                    mse_by_pol[p].append(a[p])
        if not mse_by_pol:
            continue
        avg = {p: np.mean(mse_by_pol[p]) for p in ALL}
        best_p = min(ALL, key=lambda p: avg[p])
        print(f"  {ds:<22} {model:<14} {avg['retrain']:>10.2f} {avg['rgtta']:>10.2f} {avg['rgtta_ewc']:>10.2f} {avg['tta']:>10.2f} {avg[best_p]:>10.2f} {best_p:>10}")
