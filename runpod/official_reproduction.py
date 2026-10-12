"""Run the released DynaTTA adapter with the restored upstream framework."""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def write(path, value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--framework',type=Path,required=True)
    parser.add_argument('--results-dir',type=Path,required=True)
    parser.add_argument('--dataset',choices=['ETTh1','ETTh2'],default='ETTh1')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    framework=args.framework.resolve(); output=args.results_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(framework));os.chdir(output);Path('plots').mkdir(exist_ok=True)
    import numpy as np
    import torch
    from config import get_cfg_defaults
    from datasets.build import update_cfg_from_dataset
    from datasets.loader import get_test_dataloader
    from models.build import build_model
    from models.forecast import forecast
    from trainer import build_trainer
    from utils.misc import set_seeds
    from tta.dynatta import build_adapter as build_dynatta
    from tta.tafas import build_adapter as build_tafas
    cfg=get_cfg_defaults();cfg.SEED=args.seed;cfg.DATA.NAME=args.dataset
    cfg.DATA.BASE_DIR=str(framework/'data');cfg.DATA.PRED_LEN=96;cfg.MODEL.pred_len=96
    cfg.MODEL.NAME='iTransformer';update_cfg_from_dataset(cfg,args.dataset)
    cfg.NUM_GPUS=1;cfg.VISIBLE_DEVICES=0;cfg.DATA_LOADER.NUM_WORKERS=0
    cfg.TRAIN.ENABLE=True;cfg.TTA.ENABLE=True;cfg.WANDB.ENABLE=False
    cfg.RESULT_DIR=str(output);cfg.TRAIN.CHECKPOINT_DIR=str(output/'checkpoint')
    Path(cfg.TRAIN.CHECKPOINT_DIR).mkdir(exist_ok=True)
    cfg.TTA.SOLVER.BASE_LR=0.001;cfg.TTA.SOLVER.WEIGHT_DECAY=0.0
    cfg.TTA.TAFAS.GATING_INIT=0.01
    if args.smoke:
        cfg.MODEL.d_model=32;cfg.MODEL.d_ff=32;cfg.MODEL.e_layers=1
        cfg.SOLVER.MAX_EPOCH=1;cfg.TRAIN.EVAL_PERIOD=1
        cfg.DATA.TEST_RATIO=0.02
    elif not torch.cuda.is_available():
        raise RuntimeError('Production reproduction requires CUDA')
    provenance={'upstream':json.loads((framework/'upstream-manifest.json').read_text()),
                'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'config':cfg.dump(), 'smoke_only':args.smoke, 'torch':torch.__version__}
    manifest=output/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != provenance:
        raise RuntimeError('Changed reproduction configuration/source; refusing resume')
    write(manifest,provenance);(output/'config.yaml').write_text(cfg.dump())
    set_seeds(args.seed);model=build_model(cfg)
    checkpoint=Path(cfg.TRAIN.CHECKPOINT_DIR)/'checkpoint_best.pth'
    training_complete=output/'training_complete.json'
    if not (checkpoint.exists() and training_complete.exists()):
        build_trainer(cfg,model).train()
        if not checkpoint.exists(): raise RuntimeError('Training did not save a checkpoint')
        write(training_complete, {'epochs':cfg.SOLVER.MAX_EPOCH,
              'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
    if json.loads(training_complete.read_text())['checkpoint_sha256'] != hashlib.sha256(checkpoint.read_bytes()).hexdigest():
        raise RuntimeError('Checkpoint changed after training completed')
    if not checkpoint.exists(): raise RuntimeError('Training produced no validated checkpoint')
    # This checkpoint was trained by this harness, not downloaded from an untrusted source.
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model_state'],strict=True)
    checkpoint_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    source_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    results={'checkpoint_sha256':checkpoint_hash,'model_parameters':sum(p.numel() for p in model.parameters()),
             'dataset':args.dataset,'horizon':96,'seed':args.seed,'smoke_only':args.smoke,
             'evaluation':'released_DynaTTA_TAFAS_sliding_window_multivariate_standardized',
             'paper_numbers_reproduced':False,'policies':{}}
    for policy in ['frozen','tafas','dynatta']:
        set_seeds(args.seed);model.load_state_dict(source_state,strict=True);model.requires_grad_(True).eval()
        policy_cfg=copy.deepcopy(cfg)
        if torch.cuda.is_available(): torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        tick=time.perf_counter()
        if policy=='frozen':
            errors=[];absolute=[]
            with torch.no_grad():
                for batch in get_test_dataloader(policy_cfg):
                    pred,target=forecast(policy_cfg,batch,model)
                    errors.extend(((pred-target)**2).mean(dim=(-2,-1)).cpu().tolist())
                    absolute.extend((pred-target).abs().mean(dim=(-2,-1)).cpu().tolist())
            n_adapt=0;adapted=0
        else:
            adapter=(build_dynatta if policy=='dynatta' else build_tafas)(policy_cfg,model)
            adapter.adapt()
            errors=np.asarray(adapter.mse_all).tolist();absolute=np.asarray(adapter.mae_all).tolist()
            n_adapt=adapter.n_adapt
            adapted=sum(p.numel() for p in adapter.named_params_to_adapt.values())
            (output/f'{policy}_resolved_config.yaml').write_text(policy_cfg.dump())
            del adapter
        if torch.cuda.is_available(): torch.cuda.synchronize()
        if len(errors)!=len(get_test_dataloader(cfg).dataset) or not np.isfinite(errors).all():
            raise RuntimeError(f'{policy}: incomplete or nonfinite evaluation')
        metrics={'mse':float(np.mean(errors)),'mae':float(np.mean(absolute)),
                 'windows':len(errors),'adaptation_steps':n_adapt,'adapted_parameters':adapted,
                 'wall_seconds':time.perf_counter()-tick,
                 'peak_cuda_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None}
        results['policies'][policy]=metrics
        np.savez_compressed(output/f'{policy}_per_window.npz',mse=errors,mae=absolute)
        write(output/'results.json',results)
        print('POLICY_COMPLETE',policy,json.dumps(metrics),flush=True)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    results['complete']=True;write(output/'results.json',results)
    print('OFFICIAL_PIPELINE_COMPLETE',args.dataset,args.seed,flush=True)


if __name__=='__main__': main()
