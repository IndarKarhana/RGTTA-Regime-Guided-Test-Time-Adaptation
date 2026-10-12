"""Validate archived runs and generate evidence tables without rerunning training."""
import csv
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runpod.merge_corrected import merged_results, job_key

POLICIES = ['tta', 'ewc', 'dynatta', 'rgtta', 'rgtta_ewc', 'rgtta_dynatta', 'retrain']
MODELS = ['itransformer', 'patchtst', 'dlinear', 'gru_small']
REAL = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Exchange']
SYNTH = ['synth_stable','synth_trend_break','synth_slow_drift','synth_fast_switch',
         'synth_recurring','synth_volatility','synth_shock_recovery','synth_multi_regime']
OUT = ROOT/'benchmarks/results/final_validated'
TABLES = ROOT/'paper/generated'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_archive(session):
    path = ROOT/'benchmarks/results'/session/'results.tar.gz'
    with path.open('rb') as f:
        checksum = hashlib.file_digest(f, 'sha256').hexdigest()
    expected = path.with_name('archive.sha256').read_text().split()[0]
    require(checksum == expected, f'Archive checksum mismatch: {session}')
    values = {}
    with tarfile.open(path) as archive:
        for member in archive:
            name = member.name
            if name.endswith('.json') and ('/jobs/' in name or name.endswith(('manifest.json', 'unified_results.json', '/results.json', 'failures.json', 'status.json', 'training_complete.json'))):
                values[name] = json.load(archive.extractfile(member))
            elif '/official_dynatta/' in name and name.endswith('/checkpoint_best.pth'):
                values[name] = hashlib.file_digest(archive.extractfile(member), 'sha256').hexdigest()
            elif '/official_dynatta/' in name and name.endswith('_per_window.npz'):
                with np.load(io.BytesIO(archive.extractfile(member).read())) as arrays:
                    values[name] = {k: arrays[k].copy() for k in arrays.files}
    return values, checksum


def jobs(archive, folder):
    prefix = f'benchmarks/results/{folder}/jobs/'
    values = [v for k, v in archive.items() if k.startswith(prefix) and k.endswith('.json')]
    result = {job_key(v): v for v in values}
    require(len(result) == len(values), f'Duplicate jobs: {folder}')
    return result


def validate_metric(metric, horizon, expected_origins=9):
    batches = metric['batch_metrics']
    require(metric['n_batches_evaluated'] == len(batches) == expected_origins, 'Wrong forecast-origin count')
    origins = []
    for b in batches:
        require(b['forecast_origin'] < b['target_start'] <= b['target_end'], 'Noncausal forecast origin')
        for name in ('mse', 'mae', 'batch_time'):
            require(math.isfinite(b[name]) and b[name] >= 0, f'Invalid {name}')
        for name in ('mse', 'mae'):
            seq = b.get('per_step_' + name)
            if seq is not None:
                require(len(seq) == horizon and np.isfinite(seq).all(), 'Invalid horizon errors')
                require(math.isclose(statistics.mean(seq), b[name], rel_tol=1e-6, abs_tol=1e-8), 'Per-step aggregation mismatch')
        origins.append(tuple(b[k] for k in ('forecast_origin','target_start','target_end')))
    for name in ('mse', 'mae'):
        require(math.isclose(statistics.mean(b[name] for b in batches), metric[name+'_mean'], rel_tol=1e-8, abs_tol=1e-9), 'Mean aggregation mismatch')
    require(math.isfinite(metric['total_time']) and metric['total_time'] > 0, 'Invalid timing')
    require(len(set(origins)) == expected_origins, 'Repeated forecast origins')
    return origins


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def write_csv(name, rows):
    require(bool(rows), f'Empty output: {name}')
    with (OUT/name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def summarize(rows, keys, metrics):
    grouped = {}
    for row in rows:
        grouped.setdefault(tuple(row[k] for k in keys), []).append(row)
    output = []
    for key, group in sorted(grouped.items()):
        require(sorted(r['seed'] for r in group) == [0, 1, 2], 'Missing or duplicate seeds')
        entry = dict(zip(keys, key)); entry['seeds'] = 3
        for metric in metrics:
            vals = [r[metric] for r in group]
            entry[metric+'_mean'] = statistics.mean(vals)
            entry[metric+'_sd'] = statistics.stdev(vals)
        output.append(entry)
    return output


def main():
    OUT.mkdir(exist_ok=True);TABLES.mkdir(exist_ok=True)
    legacy, lh = read_archive('runpod_full_session_20261010')
    resume, rh = read_archive('runpod_resume_session_20261010')
    reviewer, vh = read_archive('runpod_reviewer_session_20261010')
    base = 'benchmarks/results/'
    lm = legacy[base+'full_matrix/manifest.json']; cm = resume[base+'corrected_rg/manifest.json']
    combined = merged_results(legacy[base+'full_matrix/unified_results.json'],
                              resume[base+'corrected_rg/unified_results.json'], lm, cm)
    retrain = jobs(resume, 'full_retrain'); fixed = jobs(resume, 'full_retrain_dlinear_fixed')
    require(len(retrain) == 504 and len(fixed) == 168, 'Incomplete retraining')
    require(not (retrain.keys() & fixed.keys()), 'Overlapping retraining sources')
    failures = resume[base+'full_retrain/failures.json']
    require(len(failures) == 168 and set(failures) == {'_'.join(map(str,k)) for k in fixed}, 'Unresolved original failures')
    require(resume[base+'full_retrain_dlinear_fixed/failures.json'] == {}, 'Retry failures')
    rm=resume[base+'full_retrain/manifest.json']; fm=resume[base+'full_retrain_dlinear_fixed/manifest.json']
    require(lm['data_sha256'] == rm['data_sha256'] == fm['data_sha256'], 'Retraining data mismatch')
    changed = [k for k in rm['source_sha256'] if rm['source_sha256'][k] != fm['source_sha256'].get(k)]
    require(changed == ['src/regime_forecasting/core/forecaster.py'], 'Unexpected retry source changes')
    retrain.update(fixed)
    expected = set(itertools.product(MODELS, REAL+SYNTH, [96,192,336,720], range(3)))
    require(set(map(job_key,combined['experiments'])) == set(retrain) == expected, 'Incomplete matrix keys')
    rows=[]
    for e in combined['experiments']:
        e['retrain']=retrain[job_key(e)]['retrain']
        reference=validate_metric(e['tta'], e['forecast_horizon'], e['n_batches']-1)
        for p in POLICIES:
            require(validate_metric(e[p], e['forecast_horizon'], e['n_batches']-1) == reference, 'Policy forecast origins differ')
            rows.append(dict(model=e['model'], dataset=e['dataset'], horizon=e['forecast_horizon'],seed=e['seed'],policy=p,
                             mse=e[p]['mse_mean'],mae=e[p]['mae_mean'],seconds=e[p]['total_time']))
    combined['metadata'].update(policies=POLICIES, retrain_provenance={'original':rm,'dlinear_fixed':fm},
                               validation='finalize_results.py', complete=True)
    write_json(OUT/'unified_results.json', combined)
    write_csv('per_seed.csv', rows)
    means=summarize(rows,['model','dataset','horizon','policy'],['mse','mae','seconds']);write_csv('main_tables.csv',means)
    lookup={(r['model'],r['dataset'],r['horizon'],r['policy']):r for r in means}
    paired=[]
    for r in means:
        if r['policy'] not in ['rgtta','rgtta_ewc','rgtta_dynatta']:continue
        baseline={'rgtta':'tta','rgtta_ewc':'ewc','rgtta_dynatta':'dynatta'}[r['policy']]
        b=lookup[(r['model'],r['dataset'],r['horizon'],baseline)]
        paired.append(dict(model=r['model'],dataset=r['dataset'],horizon=r['horizon'],policy=r['policy'],baseline=baseline,
                           mse_ratio=r['mse_mean']/b['mse_mean'],time_ratio=r['seconds_mean']/b['seconds_mean']))
    write_csv('paired_comparisons.csv',paired)
    aggregate=[]
    for scope,datasets in [('real',REAL),('synthetic',SYNTH),('all',REAL+SYNTH)]:
        for policy in ['rgtta','rgtta_ewc','rgtta_dynatta']:
            selected=[r for r in paired if r['policy']==policy and r['dataset'] in datasets]
            aggregate.append(dict(scope=scope,policy=policy,settings=len(selected),wins=sum(r['mse_ratio']<1 for r in selected),
                                  geometric_mse_ratio=math.exp(statistics.mean(math.log(r['mse_ratio']) for r in selected)),
                                  median_mse_ratio=statistics.median(r['mse_ratio'] for r in selected),
                                  geometric_time_ratio=math.exp(statistics.mean(math.log(r['time_ratio']) for r in selected))))
    write_csv('aggregate.csv',aggregate)
    studies=[v for k,v in reviewer.items() if k.startswith(base+'reviewer_studies/jobs/')]
    require(len(studies)==93 and reviewer[base+'reviewer_studies/failures.json']=={}, 'Incomplete reviewer studies')
    study_rows=[]
    for e in studies:
        c=e['study_config']
        for p in ('tta','rgtta','dynatta'):
            if p not in e:continue
            validate_metric(e[p],96)
            counts=e['parameter_counts'][p]
            study_rows.append(dict(study=c['study'],variant=c['variant'],model=e['model'],dataset=e['dataset'],seed=e['seed'],policy=p,
                                   mse=e[p]['mse_mean'],mae=e[p]['mae_mean'],seconds=e[p]['total_time'],
                                   parameters=counts['total'],adapted=counts['adapted'],peak_mib=e['peak_cuda_bytes']/2**20))
    write_csv('reviewer_per_seed.csv',study_rows)
    study_means=summarize(study_rows,['study','variant','model','dataset','policy'],['mse','mae','seconds','parameters','adapted','peak_mib'])
    write_csv('reviewer_tables.csv',study_means)
    official=[]
    for name,e in resume.items():
        if '/official_dynatta/' not in name or not name.endswith('/results.json'):continue
        require(e.get('complete') and not e['smoke_only'] and not e['paper_numbers_reproduced'], 'Invalid official scope')
        parent=name.rsplit('/',1)[0]
        require(resume[parent+'/training_complete.json']['checkpoint_sha256']==e['checkpoint_sha256'], 'Source checkpoint mismatch')
        require(resume[parent+'/checkpoint/checkpoint_best.pth']==e['checkpoint_sha256'], 'Archived checkpoint bytes differ')
        windows=[]
        for policy in ['frozen','tafas','dynatta']:
            metric=e['policies'][policy];a=resume[parent+'/'+policy+'_per_window.npz']
            for err in ('mse','mae'):
                require(len(a[err])==metric['windows'] and np.isfinite(a[err]).all() and (a[err]>=0).all(), 'Invalid official per-window errors')
                require(math.isclose(float(np.mean(a[err])),metric[err],rel_tol=1e-8), 'Official error mismatch')
            windows.append(metric['windows'])
            official.append(dict(dataset=e['dataset'],seed=e['seed'],policy=policy,mse=metric['mse'],mae=metric['mae'],
                                 seconds=metric['wall_seconds'],windows=metric['windows'],parameters=e['model_parameters'],adapted=metric['adapted_parameters']))
        require(len(set(windows))==1, 'Official window counts differ')
    require(len(official)==18, 'Missing official policy results')
    write_csv('official_per_seed.csv',official)
    official_means=summarize(official,['dataset','policy'],['mse','mae','seconds','windows','parameters','adapted'])
    write_csv('official_tables.csv',official_means)
    status=dict(validated=True,main_experiments=672,main_policy_results=len(rows),reviewer_experiments=93,
                reviewer_policy_results=len(study_rows),official_experiments=6,official_policy_results=18,
                failed_dlinear_runs_replaced=168,archive_sha256={'legacy':lh,'resume':rh,'reviewer':vh},
                caveats=['Released-code official results use a separate protocol; no RG results in that protocol.',
                         'Three seeds quantify initialization variability, not temporal independence.',
                         'Retraining has a different training recipe and compute budget.',
                         'Larger-model and feature studies are focused H96 experiments, not foundation-model evidence.'])
    write_json(OUT/'validation.json',status)
    print(json.dumps(status,indent=2))


if __name__=='__main__':main()
