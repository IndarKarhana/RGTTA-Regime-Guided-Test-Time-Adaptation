"""Checkpoint each experiment; resume only with identical code, data and settings."""
import argparse
import gc
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.run_unified_benchmark import (
    UnifiedBenchmark, ALL_DATASETS, DEFAULT_HORIZONS, MODEL_REGISTRY,
)

POLICIES = ['tta', 'ewc', 'dynatta', 'rgtta', 'rgtta_ewc', 'rgtta_dynatta']


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def validate(result, policies):
    if not isinstance(result, dict):
        raise ValueError('Experiment returned no results')
    for policy in policies:
        metrics = result[policy]
        if metrics['n_batches_evaluated'] <= 0:
            raise ValueError(f'{policy}: no evaluated batches')
        for metric in ['mse_mean', 'mae_mean', 'total_time']:
            if not math.isfinite(metrics[metric]):
                raise ValueError(f'{policy}: invalid {metric}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--models', nargs='+', default=list(MODEL_REGISTRY))
    parser.add_argument('--datasets', nargs='+', default=ALL_DATASETS)
    parser.add_argument('--horizons', nargs='+', type=int, default=DEFAULT_HORIZONS)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--policies', nargs='+', default=POLICIES)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error('--seeds must be positive')
    bench = UnifiedBenchmark(results_dir=args.results_dir, device=args.device,
                             n_seeds=args.seeds, forecast_horizons=args.horizons,
                             model_keys=args.models, policies=args.policies)
    directory = bench.results_dir
    (directory / 'jobs').mkdir(exist_ok=True)
    sources = sorted(p for folder in ['src', 'benchmarks', 'runpod']
                     for p in (ROOT / folder).rglob('*.py')
                     if 'results' not in p.parts and '__pycache__' not in p.parts)
    provenance = {'settings': vars(args),
                  'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sources}, 'data_sha256': {}}
    for dataset in args.datasets:
        path = bench.loader.download_dataset(dataset)
        provenance['data_sha256'][dataset] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = directory / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != provenance:
        raise RuntimeError('Resume refused: code, data or settings changed; use a new results directory')
    save(manifest, provenance)
    jobs = list(itertools.product(args.models, args.datasets, args.horizons, range(args.seeds)))
    started = time.monotonic()
    failures = {}
    for index, job in enumerate(jobs, 1):
        key = '_'.join(map(str, job))
        path = directory / 'jobs' / (key + '.json')
        try:
            if path.exists():
                result = json.loads(path.read_text())
                validate(result, args.policies)
            else:
                tick = time.monotonic()
                result = bench.run_single(*job)
                validate(result, args.policies)
                result['wall_seconds'] = time.monotonic() - tick
                save(path, result)
            bench.results['experiments'].append(result)
            print(f'PROGRESS {index}/{len(jobs)} OK {key} wall={result.get("wall_seconds", 0):.1f}s', flush=True)
        except Exception:
            failures[key] = traceback.format_exc()
            print(f'PROGRESS {index}/{len(jobs)} FAILED {key}\n{failures[key]}', flush=True)
        finally:
            gc.collect()
            if args.device == 'cuda':
                import torch
                torch.cuda.empty_cache()
        save(directory / 'failures.json', failures)
        bench.results['metadata'].update(expected_experiments=len(jobs),
            completed_experiments=len(bench.results['experiments']),
            complete=len(bench.results['experiments']) == len(jobs),
            total_time_seconds=time.monotonic() - started, policies=args.policies,
            datasets=args.datasets)
        bench._save_results()
    bench._export_csv()
    if failures:
        raise RuntimeError(f'{len(failures)} experiments failed; see failures.json; rerun to retry')
    print(f'COMPLETE {len(jobs)} experiments, {len(jobs) * len(args.policies)} policy results', flush=True)


if __name__ == '__main__':
    main()
