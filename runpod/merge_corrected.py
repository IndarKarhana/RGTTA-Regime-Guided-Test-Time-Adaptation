"""Replace only affected RG policies after both complete matrices are backed up."""
import argparse
import copy
import csv
import math
import json
from pathlib import Path
import statistics
import tarfile
import time

AFFECTED = ('rgtta', 'rgtta_ewc', 'rgtta_dynatta')
UNCHANGED = ('tta', 'ewc', 'dynatta')
PROTOCOL = 'observed_batch_to_next_batch_v1'


def job_key(result):
    return tuple(result[k] for k in ('model','dataset','forecast_horizon','seed'))


def merged_results(legacy, corrected, legacy_manifest, corrected_manifest):
    for data in (legacy, corrected):
        meta=data['metadata']
        if meta.get('evaluation_protocol') != PROTOCOL or not meta.get('complete'):
            raise ValueError('Incomplete matrix or incompatible evaluation protocol')
        if len(data['experiments']) != meta['expected_experiments']:
            raise ValueError('Experiment count does not match the declared matrix')
    if legacy_manifest['data_sha256'] != corrected_manifest['data_sha256']:
        raise ValueError('Dataset hashes differ')
    old={job_key(e):e for e in legacy['experiments']}
    new={job_key(e):e for e in corrected['experiments']}
    if len(old)!=len(legacy['experiments']) or len(new)!=len(corrected['experiments']):
        raise ValueError('Duplicate experiment keys')
    if old.keys()!=new.keys(): raise ValueError('Experiment matrices differ')
    output=copy.deepcopy(legacy)
    for experiment in output['experiments']:
        corrected_job=new[job_key(experiment)]
        for policy in AFFECTED:
            metrics=corrected_job[policy]
            if metrics['n_batches_evaluated']!=experiment['tta']['n_batches_evaluated']:
                raise ValueError('Evaluated batch counts differ')
            old_origins=[tuple(b[k] for k in ('forecast_origin','target_start','target_end'))
                         for b in experiment['tta']['batch_metrics']]
            new_origins=[tuple(b[k] for k in ('forecast_origin','target_start','target_end'))
                         for b in metrics['batch_metrics']]
            if old_origins!=new_origins: raise ValueError('Forecast origins differ')
            if not all(math.isfinite(metrics[k]) for k in ('mse_mean','mae_mean','total_time')):
                raise ValueError('Nonfinite corrected metrics')
            experiment[policy]=copy.deepcopy(metrics)
        experiment.pop('wall_seconds',None)
    # The merged table has two execution sources, not one meaningful wall duration.
    output['metadata'].pop('total_time_seconds',None)
    output['metadata']['corrected_policies']=list(AFFECTED)
    output['metadata']['unchanged_policies']=list(UNCHANGED)
    output['metadata']['provenance']={'legacy':legacy_manifest,'corrected':corrected_manifest}
    output['metadata']['timing_note']='Policies ran on separate RTX 4090 hosts; preserve per-source timing provenance.'
    return output


def archive_json(path, member):
    with tarfile.open(path,'r:gz') as tar:
        return json.load(tar.extractfile(member))


def atomic(path, data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False));tmp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy',type=Path,required=True)
    parser.add_argument('--corrected',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    deadline=time.time()+6*3600
    while True:
        try:
            legacy=archive_json(args.legacy,'benchmarks/results/full_matrix/unified_results.json')
            lm=archive_json(args.legacy,'benchmarks/results/full_matrix/manifest.json')
            corrected=archive_json(args.corrected,'benchmarks/results/corrected_rg/unified_results.json')
            cm=archive_json(args.corrected,'benchmarks/results/corrected_rg/manifest.json')
            output=merged_results(legacy,corrected,lm,cm)
            atomic(args.output/'unified_results.json',output)
            grouped={}
            for e in output['experiments']:
                for policy in UNCHANGED+AFFECTED:
                    key=(e['model'],e['dataset'],e['forecast_horizon'],policy)
                    grouped.setdefault(key,[]).append(e[policy])
            with (args.output/'tables.csv').open('w',newline='') as handle:
                writer=csv.writer(handle)
                writer.writerow(['model','dataset','horizon','policy','seeds','mse_mean','mse_std','mae_mean','mae_std','time_mean'])
                for key,metrics in sorted(grouped.items()):
                    mse=[m['mse_mean'] for m in metrics];mae=[m['mae_mean'] for m in metrics]
                    writer.writerow([*key,len(metrics),statistics.mean(mse),statistics.stdev(mse) if len(mse)>1 else 0,
                       statistics.mean(mae),statistics.stdev(mae) if len(mae)>1 else 0,
                       statistics.mean(m['total_time'] for m in metrics)])
            pairs={'rgtta':'tta','rgtta_ewc':'ewc','rgtta_dynatta':'dynatta'}
            with (args.output/'paired_comparisons.csv').open('w',newline='') as handle:
                writer=csv.writer(handle)
                writer.writerow(['model','dataset','horizon','rg_policy','baseline','rg_mse','baseline_mse','relative_mse_change_pct'])
                for (model,dataset,horizon,policy),metrics in sorted(grouped.items()):
                    if policy not in pairs: continue
                    baseline=pairs[policy]
                    rg_mse=statistics.mean(m['mse_mean'] for m in metrics)
                    base_mse=statistics.mean(m['mse_mean'] for m in grouped[(model,dataset,horizon,baseline)])
                    writer.writerow([model,dataset,horizon,policy,baseline,rg_mse,base_mse,
                                     100*(rg_mse/base_mse-1) if base_mse else ''])
            atomic(args.output/'status.json',{'complete':True,'experiments':len(output['experiments']),
                'policy_results':len(output['experiments'])*6,'corrected_policy_results':len(output['experiments'])*3})
            print('Corrected tables generated',args.output,flush=True)
            return
        except (FileNotFoundError,KeyError,ValueError,tarfile.TarError) as error:
            atomic(args.output/'status.json',{'complete':False,'reason':str(error),'last_check_utc_epoch':time.time()})
            if not args.watch or time.time()>deadline: raise
            time.sleep(60)


if __name__=='__main__':main()
