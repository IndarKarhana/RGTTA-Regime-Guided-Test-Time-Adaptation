"""Analyse checkpoint loading frequency and effectiveness from Run #72 results."""
import json
import numpy as np
from collections import defaultdict

with open('benchmarks/results/unified_v2_8pol/unified_results.json') as f:
    data = json.load(f)

rg_policies = ['rgtta', 'rgtta_ewc', 'rgtta_dynatta']

for pol in rg_policies:
    total_batches = 0
    ckpt_loaded = 0
    ckpt_loaded_beat_tta = 0
    ckpt_loaded_lost_tta = 0
    ckpt_not_loaded = 0
    no_ckpt_beat_tta = 0
    no_ckpt_lost_tta = 0

    by_dataset = defaultdict(lambda: {'loaded': 0, 'total': 0})
    by_model = defaultdict(lambda: {'loaded': 0, 'total': 0})
    by_tier = defaultdict(lambda: {'loaded': 0, 'total': 0})
    by_sim_bucket = defaultdict(lambda: {'loaded': 0, 'total': 0})

    loaded_imps = []
    not_loaded_imps = []

    # Per-dataset loaded improvements
    ds_loaded_imps = defaultdict(list)
    ds_not_loaded_imps = defaultdict(list)

    for exp in data['experiments']:
        if pol not in exp or 'tta' not in exp:
            continue

        rg_batches = exp[pol].get('batch_metrics', [])
        tta_batches = exp['tta'].get('batch_metrics', [])
        tta_mse = {b['batch']: b['mse'] for b in tta_batches}

        ds = exp['dataset']
        model = exp['model']

        for b in rg_batches:
            total_batches += 1
            loaded = b.get('loaded_checkpoint', False)
            tier = b.get('tier', 'unknown')
            sim = b.get('similarity', 0)
            rg_mse = b['mse']
            t_mse = tta_mse.get(b['batch'])

            if sim >= 0.85:
                sbucket = '>=0.85'
            elif sim >= 0.55:
                sbucket = '0.55-0.85'
            else:
                sbucket = '<0.55'

            by_dataset[ds]['total'] += 1
            by_model[model]['total'] += 1
            by_tier[tier]['total'] += 1
            by_sim_bucket[sbucket]['total'] += 1

            if loaded:
                ckpt_loaded += 1
                by_dataset[ds]['loaded'] += 1
                by_model[model]['loaded'] += 1
                by_tier[tier]['loaded'] += 1
                by_sim_bucket[sbucket]['loaded'] += 1

                if t_mse is not None and t_mse > 0:
                    imp = (t_mse - rg_mse) / t_mse * 100
                    loaded_imps.append(imp)
                    ds_loaded_imps[ds].append(imp)
                    if rg_mse < t_mse:
                        ckpt_loaded_beat_tta += 1
                    else:
                        ckpt_loaded_lost_tta += 1
            else:
                ckpt_not_loaded += 1
                if t_mse is not None and t_mse > 0:
                    imp = (t_mse - rg_mse) / t_mse * 100
                    not_loaded_imps.append(imp)
                    ds_not_loaded_imps[ds].append(imp)
                    if rg_mse < t_mse:
                        no_ckpt_beat_tta += 1
                    else:
                        no_ckpt_lost_tta += 1

    pct = ckpt_loaded / total_batches * 100 if total_batches else 0

    print(f'\n{"="*65}')
    print(f' {pol.upper()} — Checkpoint Loading Analysis')
    print(f'{"="*65}')
    print(f' Total batches:          {total_batches}')
    print(f' Checkpoint loaded:      {ckpt_loaded} ({pct:.1f}%)')
    print(f' Checkpoint NOT loaded:  {ckpt_not_loaded} ({100-pct:.1f}%)')

    print(f'\n WHEN CHECKPOINT LOADED ({ckpt_loaded} batches):')
    if ckpt_loaded > 0:
        total_compared = ckpt_loaded_beat_tta + ckpt_loaded_lost_tta
        win_rate = ckpt_loaded_beat_tta / total_compared * 100 if total_compared else 0
        print(f'   Beat TTA:    {ckpt_loaded_beat_tta} ({win_rate:.1f}%)')
        print(f'   Lost to TTA: {ckpt_loaded_lost_tta} ({100-win_rate:.1f}%)')
        if loaded_imps:
            print(f'   Avg MSE improvement over TTA: {np.mean(loaded_imps):+.1f}%')
            print(f'   Median improvement:           {np.median(loaded_imps):+.1f}%')
    else:
        print(f'   (no checkpoints were ever loaded)')

    print(f'\n WHEN CHECKPOINT NOT LOADED ({ckpt_not_loaded} batches):')
    total_nc = no_ckpt_beat_tta + no_ckpt_lost_tta
    win_nc = no_ckpt_beat_tta / total_nc * 100 if total_nc else 0
    print(f'   Beat TTA:    {no_ckpt_beat_tta} ({win_nc:.1f}%)')
    print(f'   Lost to TTA: {no_ckpt_lost_tta} ({100-win_nc:.1f}%)')
    if not_loaded_imps:
        print(f'   Avg MSE improvement over TTA: {np.mean(not_loaded_imps):+.1f}%')
        print(f'   Median improvement:           {np.median(not_loaded_imps):+.1f}%')

    print(f'\n BY DATASET:')
    for ds in sorted(by_dataset.keys()):
        d = by_dataset[ds]
        pct_d = d['loaded'] / d['total'] * 100 if d['total'] else 0
        extra = ''
        if ds in ds_loaded_imps and ds_loaded_imps[ds]:
            extra = f'  (loaded avg imp: {np.mean(ds_loaded_imps[ds]):+.1f}%)'
        print(f'   {ds:25s}: {d["loaded"]:3d}/{d["total"]:4d} loaded ({pct_d:5.1f}%){extra}')

    print(f'\n BY MODEL:')
    for m in sorted(by_model.keys()):
        d = by_model[m]
        pct_m = d['loaded'] / d['total'] * 100 if d['total'] else 0
        print(f'   {m:15s}: {d["loaded"]:3d}/{d["total"]:4d} loaded ({pct_m:5.1f}%)')

    print(f'\n BY SIMILARITY BUCKET:')
    for s in ['>=0.85', '0.55-0.85', '<0.55']:
        if s in by_sim_bucket:
            d = by_sim_bucket[s]
            pct_s = d['loaded'] / d['total'] * 100 if d['total'] else 0
            print(f'   sim {s:10s}: {d["loaded"]:3d}/{d["total"]:4d} loaded ({pct_s:5.1f}%)')

    print(f'\n BY TIER:')
    for t in sorted(by_tier.keys()):
        d = by_tier[t]
        pct_t = d['loaded'] / d['total'] * 100 if d['total'] else 0
        print(f'   {t:10s}: {d["loaded"]:3d}/{d["total"]:4d} loaded ({pct_t:5.1f}%)')
