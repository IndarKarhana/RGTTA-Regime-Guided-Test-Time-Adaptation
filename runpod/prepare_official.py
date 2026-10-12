"""Assemble pinned public DynaTTA code with its upstream TAFAS framework."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

TAFAS_COMMIT = '139bf980671da4daad728a0fc21d8df508b9203d'
DYNATTA_COMMIT = '03c3cd9568ee9affa5617efa3c8dbc0224bf9a06'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tafas', type=Path, required=True)
    p.add_argument('--dynatta', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    args = p.parse_args()
    for folder, expected in [(args.tafas, TAFAS_COMMIT), (args.dynatta, DYNATTA_COMMIT)]:
        actual = subprocess.check_output(['git', '-C', str(folder), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != expected: raise RuntimeError(f'Unexpected source commit {actual}')
    shutil.copytree(args.tafas, args.destination, ignore=shutil.ignore_patterns('.git', '__pycache__', 'data', 'wandb'))
    shutil.copy2(args.dynatta/'DynaTTA/DynaTTA.py', args.destination/'tta/dynatta.py')
    patches = []
    # Device portability only: CUDA behavior is unchanged; CPU smoke tests become possible.
    replacements = {
        'utils/misc.py': [('return inputs.float().cuda()', 'return inputs.float().to("cuda" if torch.cuda.is_available() else "cpu")')],
        'models/forecast.py': [('.float().cuda()', '.float().to(enc_window.device)')],
        'tta/tafas.py': [('Calibration(cfg).cuda()', 'Calibration(cfg).to(next(model.parameters()).device)')],
    }
    for name, pairs in replacements.items():
        path = args.destination/name
        before = path.read_text(); after = before
        for old, new in pairs:
            if old not in after: raise RuntimeError(f'Missing patch context: {name}')
            after = after.replace(old, new)
        path.write_text(after)
        patches.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile=name, tofile=name))
    (args.destination/'compatibility.patch').write_text(''.join(patches))
    for dataset in ['ETTh1', 'ETTh2']:
        target=args.destination/'data'/dataset
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.data/f'{dataset}.csv', target/f'{dataset}.csv')
    manifest = {'tafas_commit':TAFAS_COMMIT, 'dynatta_commit':DYNATTA_COMMIT,
       'dynatta_source_unmodified':True,
       'files': {str(f.relative_to(args.destination)):hashlib.sha256(f.read_bytes()).hexdigest()
                 for f in args.destination.rglob('*') if f.is_file() and f.suffix in ['.py','.csv','.patch']}}
    (args.destination/'upstream-manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__ == '__main__': main()
