"""Focused reviewer studies; independent results and explicit provenance."""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from runpod.run_matrix import save, validate
from benchmarks.run_unified_benchmark import UnifiedBenchmark, MODEL_REGISTRY, StandardBenchmarkLoader


def plan():
    jobs = []
    def add(study, variant, model, dataset, seed, **settings):
        jobs.append(dict(study=study, variant=variant, model=model, dataset=dataset,
                         seed=seed, horizon=96, **settings))
    # Each seed cycles through studies so a bounded run covers every concern early.
    for seed in range(3):
        for model in ['itransformer', 'patchtst']:
            for variant, large, adaptation in [('small_head', False, 'head'),
                    ('large_head', True, 'head'), ('large_adapter', True, 'adapter'),
                    ('large_full', True, 'full')]:
                add('scale', variant, model, 'ETTh1', seed, large=large, adaptation=adaptation)
        for mode in ['target', 'marginal', 'dependence']:
            add('dependence_shift', mode, 'itransformer', 'CovShift', seed, feature_mode=mode)
        for dataset in ['ETTh1', 'Weather']:
            for mode in ['target', 'marginal', 'dependence']:
                add('features', mode, 'itransformer', dataset, seed, feature_mode=mode)
            for variant in ['no_reuse', 'no_loss_gate', 'no_regime_lr', 'frozen']:
                add('ablation', variant, 'itransformer', dataset, seed)
            for variant, eta, warmup in [('current', 0.1, 'batch'),
                    ('fast_ema', 0.7, 'batch'), ('paper_warmup', 0.1, 'horizon')]:
                add('dynatta_sensitivity', variant, 'itransformer', dataset, seed,
                    eta=eta, warmup=warmup)
    return jobs


class StudyBenchmark(UnifiedBenchmark):
    def __init__(self, config, **kwargs):
        self.config = config
        self.parameter_counts = {}
        policies = ['tta', 'dynatta', 'rgtta'] if config['study'] == 'dynatta_sensitivity' else ['tta', 'rgtta']
        super().__init__(policies=policies, use_adapter=config.get('adaptation') == 'adapter', **kwargs)

    def _build_forecasters(self, *args, **kwargs):
        forecasters = super()._build_forecasters(*args, **kwargs)
        config = self.config
        for policy, fc in forecasters.items():
            fc.freeze_backbone = config.get('adaptation') != 'full'
            if policy == 'rgtta':
                fc.feature_mode = config.get('feature_mode', 'marginal')
                if config['variant'] == 'no_reuse': fc.ckpt_sim_threshold = 2.0
                if config['variant'] == 'no_loss_gate': fc.use_loss_gate = False
                if config['variant'] == 'no_regime_lr': fc.lr_sim_scale = 0.0
            if policy == 'dynatta': fc.eta = config['eta']
            original_fit = fc.fit
            def fit(*fit_args, _fc=fc, _policy=policy, _fit=original_fit, **fit_kwargs):
                outcome = _fit(*fit_args, **fit_kwargs)
                total = sum(p.numel() for p in _fc.model.parameters())
                if config.get('large') and total < 1_000_000:
                    raise RuntimeError('Large-model configuration has fewer than one million parameters')
                if _fc.freeze_backbone:
                    _fc._freeze_backbone()
                    adapted = sum(p.numel() for p in _fc.model.parameters() if p.requires_grad)
                    _fc._unfreeze_all()
                else:
                    adapted = total
                self.parameter_counts[_policy] = {'total': total, 'adapted': adapted}
                if _policy == 'dynatta' and config.get('warmup') == 'horizon':
                    _fc._warmup_steps = _fc.warmup_factor * _fc.forecast_horizon
                return outcome
            fc.fit = fit
            if config['variant'] == 'frozen':
                fc.update_with_new_data = lambda frame: {'status': 'skipped', 'reason': 'frozen_control'}
        return forecasters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--results-dir', default='benchmarks/results/reviewer_studies')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    StandardBenchmarkLoader.DATASET_CONFIGS['CovShift'] = {
        'file': 'covariance_shift.csv', 'freq': 'h', 'target': 'OT',
        'features': ['x', 'z', 'OT'], 'description': 'Controlled cross-variable sign shifts'}
    directory = Path(args.results_dir)
    (directory/'jobs').mkdir(parents=True, exist_ok=True)
    jobs = plan()
    if args.limit: jobs = jobs[:args.limit]
    manifest = {'jobs': jobs, 'device': args.device, 'protocol': 'observed_batch_to_next_batch_v1',
                'scope': 'Focused H96 study; DynaTTA settings sensitivity, not an original-protocol reproduction',
                'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for folder in ['src', 'benchmarks', 'runpod'] for p in (ROOT/folder).rglob('*.py')
                   if 'results' not in p.parts},
                'data_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (ROOT/'data/benchmarks').glob('*.csv')}}
    path = directory/'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise RuntimeError('Resume refused: changed study, source or data')
    save(path, manifest)
    failures = {}
    started = time.monotonic()
    for index, config in enumerate(jobs, 1):
        key = '_'.join(str(config[k]) for k in ['study', 'variant', 'model', 'dataset', 'seed'])
        output = directory/'jobs'/f'{key}.json'
        original = copy.deepcopy(MODEL_REGISTRY[config['model']])
        try:
            if config.get('large'):
                MODEL_REGISTRY[config['model']]['kwargs'].update(hidden_dim=256, num_layers=4, num_heads=8)
            bench = StudyBenchmark(config, results_dir=str(directory/'work'/key), device=args.device)
            if output.exists():
                validate(json.loads(output.read_text()), bench.policies)
                continue
            if args.device == 'cuda': torch.cuda.reset_peak_memory_stats()
            tick = time.monotonic()
            result = bench.run_single(config['model'], config['dataset'], config['horizon'], config['seed'])
            validate(result, bench.policies)
            result.update(study_config=config, parameter_counts=bench.parameter_counts,
                          wall_seconds=time.monotonic()-tick,
                          peak_cuda_bytes=torch.cuda.max_memory_allocated() if args.device == 'cuda' else None)
            if config['variant'] == 'frozen':
                for counts in result['parameter_counts'].values(): counts['adapted'] = 0
            save(output, result)
            print(f'PROGRESS {index}/{len(jobs)} OK {key} wall={result["wall_seconds"]:.1f}s', flush=True)
        except Exception:
            failures[key] = traceback.format_exc()
            print(f'FAILED {key}\n{failures[key]}', flush=True)
        finally:
            MODEL_REGISTRY[config['model']] = original
            gc.collect()
            if args.device == 'cuda': torch.cuda.empty_cache()
        save(directory/'failures.json', failures)
        save(directory/'status.json', {'expected':len(jobs), 'completed':len(list((directory/'jobs').glob('*.json'))),
             'failures':len(failures), 'elapsed_seconds':time.monotonic()-started})
    if failures: raise RuntimeError(f'{len(failures)} study jobs failed')
    print(f'COMPLETE {len(jobs)} reviewer experiments', flush=True)


if __name__ == '__main__': main()
