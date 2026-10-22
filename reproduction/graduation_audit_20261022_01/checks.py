"""Bounded graduation-readiness checks; no training or experiment reruns.

Only writes checks.json in this new audit directory. Existing research assets
are read without changing their contents. Run from the repository root.
"""
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import h5py
    import numpy as np
    import torch
    from common import OBS_KEYS, load_policy
    from world_model import WorldModel, rank_metrics

    torch.set_num_threads(2)
    tracked = subprocess.check_output(
        ['git', 'ls-files', '-z'], cwd=ROOT
    ).decode('utf-8').split('\0')
    failures = []
    count = 0
    for relative in tracked:
        path = ROOT / relative
        if path.suffix != '.py' or not path.is_file():
            continue
        count += 1
        try:
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=relative)
        except (SyntaxError, UnicodeError) as error:
            failures.append({'path': relative, 'error': str(error)})

    splits = json.loads((ROOT / 'data/splits.json').read_text())
    duplicate_names = {
        split: len(names) - len(set(names)) for split, names in splits.items()
    }
    overlaps = {}
    keys = list(splits)
    for index, left in enumerate(keys):
        for right in keys[index + 1:]:
            overlaps[left + '/' + right] = sorted(set(splits[left]) & set(splits[right]))
    dataset = ROOT / 'data/can_ph_lowdim_v15.hdf5'
    manifest = json.loads(dataset.with_suffix('.manifest.json').read_text())
    with dataset.open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    dataset_sha = digest.hexdigest()
    with h5py.File(dataset, 'r') as data:
        missing = sorted({name for names in splits.values() for name in names} - set(data['data']))
        demo = data['data'][splits['dev'][0]]
        x = np.concatenate([demo['obs'][key][0] for key in OBS_KEYS]).astype(np.float32)
    policy = load_policy(ROOT / 'policy/bc_chunk_dev_v4/policy_15000.pt')
    with torch.inference_mode():
        chunk = policy.best(torch.from_numpy(x[None]))
    forwards = []
    for method in ('full', 'plain'):
        for seed in (17, 23, 41):
            relative = f'models/multiseed_20261012_01_{method}_s{seed}/model.pt'
            checkpoint = torch.load(ROOT / relative, map_location='cpu', weights_only=True)
            cfg = checkpoint['config']
            model = WorldModel(cfg['dim'], cfg['width'], cfg.get('future_only', False))
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            with torch.inference_mode():
                states, scores = model(torch.from_numpy(x[None]), chunk)
            forwards.append({
                'path': relative, 'states_shape': list(states.shape),
                'scores_shape': list(scores.shape),
                'finite': bool(torch.isfinite(states).all() and torch.isfinite(scores).all()),
            })
    jobs = json.loads((ROOT / 'reproduction/improvement_v21_20261019_01/smoke_a/jobs.json').read_text())
    contracts = sorted({
        (job.get('steps', 400), job.get('config', {}).get('horizon_steps', 400),
         job.get('config', {}).get('delay', 2)) for job in jobs
    })
    result = {
        'date_local': '2026-10-22', 'python': sys.version,
        'packages': {name: importlib.metadata.version(name) for name in
                     ('torch', 'numpy', 'robosuite', 'mujoco', 'h5py', 'scipy')},
        'syntax': {'tracked_python_files': count, 'failures': failures},
        'splits': {'sizes': {key: len(value) for key, value in splits.items()},
                   'duplicates': duplicate_names, 'overlaps': overlaps, 'missing_demos': missing},
        'dataset': {'sha256': dataset_sha, 'manifest_matches': dataset_sha == manifest['sha256']},
        'model_inference': {'bc_shape': list(chunk.shape), 'bc_finite': bool(torch.isfinite(chunk).all()),
                            'world_models': forwards,
                            'scope': 'One saved observation, CPU forward only; not closed-loop performance.'},
        'v21_registered_job_contracts_steps_horizon_delay': contracts,
        'v21_registered_episodes': sum(len(job['demos']) for job in jobs),
        'rank_metric_tie_probe': {
            'predictions': [0, 0], 'targets': [0, 1],
            'actual': rank_metrics([0, 0], [0, 1], 1, 2),
            'interpretation': 'wrong_pair_rate counts strict reversals only; a tied prediction has zero reversal rate despite regret=1. Add a tie metric before interpreting this as ranking accuracy.',
        },
        'scope': 'Static syntax, split identities, data hash, seven model forwards and saved job configurations. No training, new evaluation episodes, or target-device reproduction.',
    }
    result['passed'] = bool(
        not failures and not any(duplicate_names.values()) and not any(overlaps.values())
        and not missing and result['dataset']['manifest_matches']
        and result['model_inference']['bc_finite'] and all(row['finite'] for row in forwards)
        and contracts == [(400, 400, 2)]
    )
    (RUN / 'checks.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
