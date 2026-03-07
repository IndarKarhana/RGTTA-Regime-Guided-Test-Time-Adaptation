"""Retrain vs TTA policies analysis."""
import json
import numpy as np
from collections import defaultdict

with open('benchmarks/results/unified_retrain_v6/unified_results.json') as f:
    rd = json.load(f)
with open('benchmarks/results/unified_v2_8pol/unified_results.json') as f:
    td = json.load(f)

POL_TTA = ['tta', 'ewc', 'dynatta', 'rgtta', 'rgtta_ewc', 'rgtta_dynatta']
ALL = ['retrain'] + POL_TTA
R = {}

for e in rd['experiments']:
    k = (e['model'], e['dataset'], int(e['forecast_horizon']), int(e['seed']))
    R.setdefault(k, {})
    R[k]['retrain'] = {'mse': e['retrain']['mse_mean'], 'time': e['retrain']['total_time']}

for e in td['experiments']:
    k = (e['model'], e['dataset'], int(e['forecast_horizon']), int(e['seed']))
    R.setdefault(k, {})
    for p in POL_TTA:
        if p in e and isinstance(e[p], dict):
            R[k][p] = {'mse': e[p]['mse_mean'], 'time': e[p]['total_time']}

F = {k: v for k, v in R.items() if len(v) == 7}
print(f"Full experiments (all 7 policies): {len(F)}")

# ====== SPEEDUP ======
print(f"\n{'='*65}")
print("SPEEDUP vs RETRAIN (retrain_time / policy_time)")
print('='*65)
for p in ALL:
    sp = [F[k]['retrain']['time'] / F[k][p]['time'] for k in F if F[k][p]['time'] > 0]
    if sp:
        print(f"  {p:>16}: median {np.median(sp):6.1f}x  mean {np.mean(sp):6.1f}x  [{np.min(sp):.1f}x – {np.max(sp):.1f}x]")

# ====== HEAD-TO-HEAD: retrain vs rgtta_ewc ======
print(f"\n{'='*65}")
print("HEAD-TO-HEAD: retrain vs RG-EWC — MSE % change (seed-averaged)")
print("  Negative = RG-EWC wins, Positive = Retrain wins")
print('='*65)
G = defaultdict(lambda: defaultdict(list))
for (m, d, h, s), v in F.items():
    for p in ['retrain', 'rgtta_ewc']:
        G[(m, d, h)][p].append(v[p]['mse'])

PC = defaultdict(list)
for gk in G:
    rm = np.mean(G[gk]['retrain'])
    rgm = np.mean(G[gk]['rgtta_ewc'])
    pct = 100 * (rgm - rm) / rm if rm > 0 else 0
    PC[gk[0]].append(pct)
    PC['all'].append(pct)

for m in ['gru_small', 'itransformer', 'patchtst', 'dlinear', 'all']:
    v = PC[m]
    b = sum(1 for x in v if x < 0)
    print(f"  {m:>14}: median={np.median(v):+6.1f}%  mean={np.mean(v):+6.1f}%  RG-EWC wins {b}/{len(v)}")

# ====== HEAD-TO-HEAD: retrain vs rgtta ======
print(f"\n{'='*65}")
print("HEAD-TO-HEAD: retrain vs RG-TTA — MSE % change (seed-averaged)")
print('='*65)
G2 = defaultdict(lambda: defaultdict(list))
for (m, d, h, s), v in F.items():
    for p in ['retrain', 'rgtta']:
        G2[(m, d, h)][p].append(v[p]['mse'])

PC2 = defaultdict(list)
for gk in G2:
    rm = np.mean(G2[gk]['retrain'])
    rgm = np.mean(G2[gk]['rgtta'])
    pct = 100 * (rgm - rm) / rm if rm > 0 else 0
    PC2[gk[0]].append(pct)
    PC2['all'].append(pct)

for m in ['gru_small', 'itransformer', 'patchtst', 'dlinear', 'all']:
    v = PC2[m]
    b = sum(1 for x in v if x < 0)
    print(f"  {m:>14}: median={np.median(v):+6.1f}%  mean={np.mean(v):+6.1f}%  RG-TTA wins {b}/{len(v)}")

# ====== PER-DATASET ======
print(f"\n{'='*65}")
print("PER-DATASET WIN RATE (seed-averaged, all models)")
print('='*65)
GD = defaultdict(lambda: defaultdict(list))
for (m, d, h, s), v in F.items():
    for p in ALL:
        GD[(d, m, h)][p].append(v[p]['mse'])

dw = defaultdict(lambda: defaultdict(int))
dt = defaultdict(int)
for gk in GD:
    a = {p: np.mean(GD[gk][p]) for p in ALL if p in GD[gk]}
    if len(a) == 7:
        best = min(a, key=a.get)
        dw[gk[0]][best] += 1
        dt[gk[0]] += 1

for ds in sorted(dt):
    rw = dw[ds].get('retrain', 0)
    rg = sum(dw[ds].get(p, 0) for p in ['rgtta', 'rgtta_ewc', 'rgtta_dynatta'])
    oth = dt[ds] - rw - rg
    mk = 'RET' if rw > rg else ' RG' if rg > rw else 'TIE'
    print(f"  [{mk}] {ds:>22}: retrain={rw:>2}/{dt[ds]}({100*rw/dt[ds]:5.1f}%)  RG={rg:>2}/{dt[ds]}({100*rg/dt[ds]:5.1f}%)  other={oth}")

# ====== WHERE RG WINS DESPITE RETRAIN HAVING MORE DATA ======
print(f"\n{'='*65}")
print("CASES WHERE RG-EWC BEATS RETRAIN (by how much?)")
print('='*65)
rg_better = []
ret_better = []
for gk in G:
    rm = np.mean(G[gk]['retrain'])
    rgm = np.mean(G[gk]['rgtta_ewc'])
    pct = 100 * (rgm - rm) / rm if rm > 0 else 0
    if pct < 0:
        rg_better.append((gk, pct, rm, rgm))
    else:
        ret_better.append((gk, pct, rm, rgm))

print(f"  RG-EWC better: {len(rg_better)}/{len(rg_better)+len(ret_better)} configs")
print(f"  When RG-EWC wins: median improvement = {np.median([x[1] for x in rg_better]):+.1f}%")
print(f"  When Retrain wins: median gap = {np.median([x[1] for x in ret_better]):+.1f}%")

# ====== BY HORIZON ======
print(f"\n{'='*65}")
print("BY HORIZON (seed-averaged)")
print('='*65)
for h in [96, 192, 336, 720]:
    GH = defaultdict(lambda: defaultdict(list))
    for (m, d, hr, s), v in F.items():
        if hr == h:
            for p in ALL: GH[(m, d)][p].append(v[p]['mse'])
    wins = defaultdict(int); tot = 0
    for gk in GH:
        a = {p: np.mean(GH[gk][p]) for p in ALL if p in GH[gk]}
        if len(a) == 7:
            wins[min(a, key=a.get)] += 1; tot += 1
    rg = wins['rgtta'] + wins['rgtta_ewc'] + wins['rgtta_dynatta']
    oth = tot - wins['retrain'] - rg
    print(f"  H={h:>3}: retrain={wins['retrain']:>2}/{tot}({100*wins['retrain']/tot:5.1f}%)  RG={rg:>2}/{tot}({100*rg/tot:5.1f}%)  other={oth}")

    # Also show timing
    sp = [F[k]['retrain']['time'] / F[k]['rgtta_ewc']['time'] for k in F if k[2] == h and F[k]['rgtta_ewc']['time'] > 0]
    print(f"         RG-EWC speedup vs retrain: median {np.median(sp):.1f}x")
