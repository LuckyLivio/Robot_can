"""Prospective jobs, immutable stage locks and independent native starts."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import h5py
import numpy as np
from common import ROOT, DATA, make_env, pack_obs, restore_demo

RUN = ROOT / 'reproduction/graduation_repairs_20261022_01'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_new(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, allow_nan=False)


def sources():
    paths = list((ROOT / 'robot_can_v2').glob('*.py'))
    paths += list((ROOT / 'reproduction/recovery_v5_20261012_01').glob('*.py'))
    paths += [DATA] + [ROOT / p for p in ('common.py', 'world_model.py', 'data/splits.json', 'policy/bc_chunk_dev_v4/policy_15000.pt')]
    paths += [ROOT / f'models/multiseed_20261012_01_full_s{s}/model.pt' for s in (17, 23, 41)]
    return sorted(set(paths))


def source_hashes():
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sources()}


def prepare_development():
    names = json.loads((ROOT / 'data/splits.json').read_text())['dev']
    observation_jobs, budget_jobs = [], []
    for semantics in ('legacy', 'canonical'):
        for noise in (0., .05):
            observation_jobs.append({'id': f'{semantics}_{noise:g}', 'method': 'bc',
                'controller': 'none', 'semantics': semantics, 'seed': None, 'noise': noise,
                'demos': names, 'steps': 400, 'delay': 2,
                'policy': (RUN / f'policies/{semantics}_s17/final.pt').relative_to(ROOT).as_posix()})
    for controller in ('v5', 'queue_budget'):
        for method, seed in (('bc', None), ('full', 17)):
            for noise in (0., .05):
                budget_jobs.append({'id': f'{controller}_{method}_{noise:g}', 'method': method,
                    'seed': seed, 'noise': noise, 'controller': controller, 'semantics': 'legacy',
                    'demos': names, 'steps': 400, 'delay': 2})
    write_new(RUN / 'development_observation_jobs.json', observation_jobs)
    write_new(RUN / 'development_budget_jobs.json', budget_jobs)
    print(json.dumps({'observation_episodes': 120, 'budget_episodes': 240}))


def run_batch(job_file, stage, workers=4):
    jobs = json.loads(Path(job_file).read_text())
    confirmation_dataset = (RUN / 'confirmation_starts.hdf5').relative_to(ROOT).as_posix()
    if any(job.get('dataset') == confirmation_dataset for job in jobs):
        lock = json.loads((RUN / 'confirmation_lock.json').read_text())
        if sha(job_file) != lock['source_sha256'][(RUN / 'confirmation_jobs.json').relative_to(ROOT).as_posix()]:
            raise ValueError('Confirmation jobs drifted after prospective lock')
        for relative, expected in lock['source_sha256'].items():
            if not (ROOT / relative).is_file() or sha(ROOT / relative) != expected:
                raise ValueError(f'Confirmation source/input drift after freeze: {relative}')
    if len({j['id'] for j in jobs}) != len(jobs):
        raise ValueError('Duplicate job IDs')
    from robot_can_v2.evaluate import validate_job
    for job in jobs:
        validate_job(job)
    stage = Path(stage); stage.mkdir(parents=True, exist_ok=False)
    current = source_hashes()
    policies = {j.get('policy') for j in jobs if j.get('policy')}
    datasets = {j.get('dataset') for j in jobs if j.get('dataset')}
    current.update({p: sha(ROOT / p) for p in policies | datasets})
    for relative in current:
        if relative.endswith('.py'):
            destination = stage / 'source' / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((ROOT / relative).read_bytes())
    write_new(stage / 'jobs.json', jobs)
    write_new(stage / 'started.json', {'started_utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256': current, 'jobs_sha256': sha(stage / 'jobs.json'),
        'workers': workers, 'registered_episodes': sum(len(j['demos']) for j in jobs),
        'packages': {p: importlib.metadata.version(p) for p in ('torch', 'numpy', 'robosuite', 'mujoco', 'h5py', 'scipy')}})
    def execute(job):
        log = stage / (job['id'] + '.log')
        command = [sys.executable, '-B', '-X', 'utf8', '-m', 'robot_can_v2.evaluate',
                   '--job-file', str(stage / 'jobs.json'), '--job-id', job['id'],
                   '--output-dir', str(stage / job['id'])]
        with log.open('x', encoding='utf-8') as f:
            process = subprocess.run(command, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        return {'id': job['id'], 'returncode': process.returncode}
    started = time.perf_counter(); outcomes = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(execute, job) for job in jobs]):
            item = future.result(); outcomes.append(item)
            print(json.dumps({'finished_jobs': len(outcomes), 'total_jobs': len(jobs), **item}), flush=True)
    unchanged = all((ROOT / p).is_file() and sha(ROOT / p) == digest for p, digest in current.items())
    episodes = 0; complete_jobs = True
    for job in jobs:
        path = stage / job['id'] / 'result.json'
        if path.exists():
            result = json.loads(path.read_text())
            episodes += len(result['records'])
            complete_jobs &= (result['status'] == 'complete' and result['job'] == job
                              and [r['demo'] for r in result['records']] == job['demos'])
        else:
            complete_jobs = False
    success = all(o['returncode'] == 0 for o in outcomes) and unchanged and complete_jobs and episodes == sum(len(j['demos']) for j in jobs)
    receipt = {'passed': success, 'finished_utc': datetime.now(timezone.utc).isoformat(),
               'source_unchanged': unchanged, 'episodes': episodes, 'jobs': outcomes,
               'elapsed_s': time.perf_counter() - started}
    write_new(stage / 'completed.json', receipt)
    print(json.dumps(receipt), flush=True)
    if not success:
        raise RuntimeError('Batch failed; partial records retained and not reported as complete')


def prepare_confirmation(include_candidate=False):
    design_file = RUN / 'confirmation_design.json'
    seeds = list(range(2026102201, 2026102261))
    names = [f'demo_{22000000 + i}' for i in range(60)]
    locked = source_hashes()
    design = {'frozen_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': locked,
              'native_initialization_seeds': seeds, 'state_names': names,
              'conditions': ['clean', 'noise005'], 'world_model_seeds': [17, 23, 41],
              'primary_episodes': 960, 'additional_queue_candidate_episodes': 480 if include_candidate else 0,
              'observation_confirmation_episodes': 240,
              'bootstrap_unit': '60 paired independent starts; fixed WM seeds averaged within start',
              'bootstrap_draws': 20000, 'bootstrap_seed': 20261022,
              'selection': 'All 60 registered seeds retained without policy outcome filtering.',
              'include_candidate': include_candidate}
    for semantics in ('legacy', 'canonical'):
        path = RUN / f'policies/{semantics}_s17/final.pt'
        locked[path.relative_to(ROOT).as_posix()] = sha(path)
    locked[(RUN / 'CONFIRMATION_AMENDMENT.md').relative_to(ROOT).as_posix()] = sha(RUN / 'CONFIRMATION_AMENDMENT.md')
    write_new(design_file, design)
    with h5py.File(DATA, 'r') as old:
        previous = [d['states'][0] for d in old['data'].values()]
        meta = old['data'].attrs['env_args']
    old_files = sorted(p for p in (ROOT / 'reproduction').rglob('*starts*.hdf5') if RUN not in p.parents)
    for p in old_files:
        with h5py.File(p, 'r') as f:
            previous.extend(d['states'][0] for d in f['data'].values())
    previous = np.stack(previous)
    descriptions, regeneration = [], []
    env, original, restored = make_env(False), make_env(False), make_env(False)
    def native_reset(e, seed):
        random.seed(seed); np.random.seed(seed); e.reset()
        e._obs_cache.clear(); e._get_observations(force_update=True)
        obs = e._get_observations(force_update=True)
        return e.sim.get_state().flatten().copy(), pack_obs(obs)
    destination = RUN / 'confirmation_starts.hdf5'
    try:
        with h5py.File(destination, 'x') as f:
            g = f.create_group('data'); g.attrs['env_args'] = meta
            for seed, name in zip(seeds, names):
                state, obs = native_reset(env, seed)
                nearest = float(np.max(np.abs(previous - state), axis=1).min())
                if nearest <= 1e-8 or not np.isfinite(state).all() or env._check_success():
                    raise ValueError('Invalid or duplicate registered start; do not replace seed')
                d = g.create_group(name); d.attrs['model_file'] = env.sim.model.get_xml()
                d.attrs['initialization_seed'] = seed
                d.create_dataset('states', data=state[None]); d.create_dataset('initial_observation', data=obs)
                descriptions.append({'demo': name, 'seed': seed, 'state_sha256': hashlib.sha256(state.tobytes()).hexdigest(),
                                     'nearest_old_max_abs': nearest})
        if len({d['state_sha256'] for d in descriptions}) != 60:
            raise ValueError('Registered starts are not unique')
        with h5py.File(destination, 'r') as f:
            for item in reversed(descriptions):
                state, obs = native_reset(original, item['seed']); d = f['data'][item['demo']]
                np.testing.assert_array_equal(state, d['states'][0])
                np.testing.assert_array_equal(obs, d['initial_observation'][:])
                restored_obs = pack_obs(restore_demo(restored, d))
                error = float(np.max(np.abs(restored_obs - obs)))
                if error > 1e-5:
                    raise ValueError('Saved XML observation restoration mismatch')
                regeneration.append({'demo': item['demo'], 'native_exact': True, 'restored_error': error})
    finally:
        env.close(); original.close(); restored.close()
    jobs = []
    controllers = ['none', 'v5'] + (['queue_budget'] if include_candidate else [])
    for controller in controllers:
        for method, seed in [('bc', None), ('full', 17), ('full', 23), ('full', 41)]:
            for noise in (0., .05):
                jobs.append({'id': f'{controller}_{method}_s{seed}_{noise:g}', 'controller': controller,
                    'method': method, 'seed': seed, 'noise': noise, 'demos': names, 'study_arm': 'recovery_factorial',
                    'steps': 400, 'delay': 2, 'semantics': 'legacy',
                    'dataset': destination.relative_to(ROOT).as_posix()})
    for semantics in ('legacy', 'canonical'):
        for noise in (0., .05):
            jobs.append({'id': f'observation_{semantics}_{noise:g}', 'study_arm': 'observation_confirmation',
                'controller': 'none', 'method': 'bc', 'seed': None, 'noise': noise, 'demos': names,
                'steps': 400, 'delay': 2, 'semantics': semantics,
                'policy': (RUN / f'policies/{semantics}_s17/final.pt').relative_to(ROOT).as_posix(),
                'dataset': destination.relative_to(ROOT).as_posix()})
    write_new(RUN / 'confirmation_jobs.json', jobs)
    write_new(RUN / 'confirmation_state_validation.json', {'passed': True, 'old_states_compared': len(previous),
        'old_start_files': [p.relative_to(ROOT).as_posix() for p in old_files],
        'states': descriptions, 'regeneration': regeneration, 'native_regeneration_exact': True,
        'policy_outcome_filtering': False, 'source_frozen_before_states': True})
    locked.update({p.relative_to(ROOT).as_posix(): sha(p) for p in
                   (design_file, destination, RUN / 'confirmation_jobs.json')})
    for relative, expected in design['source_sha256'].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f'Source/input changed during start generation: {relative}')
    write_new(RUN / 'confirmation_lock.json', {'source_sha256': locked,
        'episodes': sum(len(j['demos']) for j in jobs), 'independent_starts': 60})
    print(json.dumps({'starts': 60, 'episodes': sum(len(j['demos']) for j in jobs), 'native_exact': True}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare-dev')
    c = sub.add_parser('prepare-confirmation'); c.add_argument('--include-candidate', action='store_true')
    b = sub.add_parser('run'); b.add_argument('--jobs', type=Path, required=True)
    b.add_argument('--stage', type=Path, required=True); b.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.command == 'prepare-dev':
        prepare_development()
    elif args.command == 'prepare-confirmation':
        prepare_confirmation(args.include_candidate)
    else:
        run_batch(args.jobs, args.stage, args.workers)
