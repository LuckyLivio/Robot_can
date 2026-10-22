"""Replay four prospectively selected confirmation episodes in actual MuJoCo.

Selection is fixed from job metadata and registration order, without reading
outcomes: clean none/BC and clean v5/Full(seed 17), first two registered starts
in each. These remain four EXISTING episodes, not new independent samples.
Requires a completed stage; never writes to any benchmark or frozen module.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
import types

import h5py
import numpy as np

from verify_traces import (
    DEFAULT_DATA, ROOT, RUN, VerificationError, array, exact, read_json,
    relative_file, require, sha,
)


STATE_TOLERANCE = 1e-5
ACTION_TOLERANCE = 1e-6
REGISTERED_START_COUNT = 2
SELECTION_RULE = (
    'Without inspecting outcomes: recovery_factorial, noise=0, none/BC and '
    'v5/Full(seed=17), first two registered demos in each.'
)


def selection(jobs):
    selected = []
    for controller, method, seed in [('none', 'bc', None), ('v5', 'full', 17)]:
        matches = [job for job in jobs
                   if job.get('study_arm') == 'recovery_factorial'
                   and job.get('controller') == controller and job.get('method') == method
                   and job.get('seed') == seed and job.get('noise', 0.) == 0.]
        require(len(matches) == 1, f'Selection is ambiguous/missing: {controller}/{method}/{seed}')
        job = matches[0]
        require(len(job['demos']) >= REGISTERED_START_COUNT, 'Not enough registered starts')
        selected.append((job, job['demos'][:REGISTERED_START_COUNT]))
    return selected


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def frozen_modules(stage, hashes):
    """Load the actual archived code, redirecting only its project data root."""
    needed = ['common.py', 'robot_can_v2/observations.py',
              'reproduction/recovery_v5_20261012_01/recovery_controller.py',
              'reproduction/recovery_v5_20261012_01/clearance.py']
    paths = {}
    for relative in needed:
        path = relative_file(stage / 'source', relative)
        require(sha(path) == hashes.get(relative), f'Frozen source hash mismatch: {relative}')
        paths[relative] = path
    common = load_module('_physical_replay_common', paths['common.py'])
    common.ROOT, common.DATA = ROOT, ROOT / DEFAULT_DATA
    previous = sys.modules.get('common')
    try:
        sys.modules['common'] = common
        observations = load_module('_physical_replay_observations', paths['robot_can_v2/observations.py'])
    finally:
        if previous is None:
            sys.modules.pop('common', None)
        else:
            sys.modules['common'] = previous
    package = types.ModuleType('_physical_replay_v5')
    package.__path__ = [str(paths['reproduction/recovery_v5_20261012_01/recovery_controller.py'].parent)]
    sys.modules[package.__name__] = package
    controller = load_module('_physical_replay_v5.recovery_controller',
                             paths['reproduction/recovery_v5_20261012_01/recovery_controller.py'])
    return common, observations, controller, {relative: sha(path) for relative, path in paths.items()}


def selected_rows(directory, records, demos):
    """Read only the two fixed first episodes, without scanning other outcomes."""
    counts = {name: next(record['steps'] for record in records if record['demo'] == name) for name in demos}
    rows = {name: [] for name in demos}
    with (directory / 'trace.jsonl').open(encoding='utf-8') as stream:
        for name in demos:
            for step in range(1, counts[name]+1):
                line = stream.readline()
                require(bool(line), f'{directory.name}/{name}: trace ended early')
                row = json.loads(line)
                require(row.get('demo') == name and row.get('step') == step, 'Replay trace registration order mismatch')
                rows[name].append(row)
    return rows


def state_error(actual, expected, context):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    require(actual.shape == expected.shape == (23,) and np.isfinite(actual).all()
            and np.isfinite(expected).all(), f'{context}: invalid replay state')
    per_entry = np.abs(actual-expected)
    maximum = float(per_entry.max())
    require(maximum <= STATE_TOLERANCE, f'{context}: state error {maximum} exceeds {STATE_TOLERANCE}')
    return maximum, per_entry


def replay_episode(common, observations, controllers, env, dataset, job, record, rows):
    name = record['demo']
    context = f'{job["id"]}/{name}'
    obs = common.restore_demo(env, dataset['data'][name], xml=True)
    semantics = job.get('semantics', 'legacy')
    first = observations.pack_observation(obs, semantics)
    target = env.target_bin_placements[env.object_to_id['can']].copy()
    supervisor = controllers.RecoveryController(target, env.bin2_pos[2], first,
                                                job['controller'] != 'none',
                                                controllers.Config(**job.get('config', {})))
    previous = np.zeros(7, np.float32)
    maximum = 0.; per_entry_max = np.zeros(23); action_max = 0.
    first_success = None
    for step, row in enumerate(rows, 1):
        where = f'{context}/step={step}'
        actual_before = observations.pack_observation(obs, semantics)
        error, entries = state_error(actual_before, array(row, 'state_before', (23,), where), where+' before')
        maximum = max(maximum, error); per_entry_max = np.maximum(per_entry_max, entries)
        # Recompute supervisor decisions from actual replay observations and
        # the logged policy/WM proposal. Policy/WM inference itself is outside
        # this replay; its weights and JSON selection are checked separately.
        proposal = array(row, 'proposal', (7,), where).astype(np.float32)
        submitted, diagnostic = supervisor.choose(actual_before, proposal, previous)
        submitted_error = float(np.abs(submitted-array(row, 'submitted_action', (7,), where)).max())
        action_max = max(action_max, submitted_error)
        require(submitted_error <= ACTION_TOLERANCE, f'{where}: supervisor action differs by {submitted_error}')
        require(diagnostic['mode'] == row['recovery']['mode'], f'{where}: supervisor phase differs')
        applied = array(row, 'executed_action', (7,), where).astype(np.float32)
        require((np.abs(applied) <= 1).all(), f'{where}: applied action outside OSC bounds')
        obs, *_ = env.step(applied)
        previous = applied.copy()
        after = observations.pack_observation(obs, semantics)
        error, entries = state_error(after, array(row, 'state_after', (23,), where), where+' after')
        maximum = max(maximum, error); per_entry_max = np.maximum(per_entry_max, entries)
        success = bool(env._check_success())
        if success and first_success is None:
            first_success = step
        require(success == row.get('success_after'), f'{where}: actual simulator success flag differs')
    require(len(rows) == record['steps'], f'{context}: replay control count differs')
    expected_first = record['steps'] if record['success'] else None
    require(first_success == expected_first, f'{context}: first-success control differs')
    events = lambda values: [(e['step'], e['from'], e['to'], e['reason'], e['attempt']) for e in values]
    require(events(supervisor.events) == events(record['events']), f'{context}: recovery event sequence differs')
    return {
        'job': job['id'], 'demo': name, 'passed': True, 'semantics': semantics,
        'replayed_controls': len(rows), 'original_success': record['success'],
        'first_success_control': first_success, 'max_state_absolute_error': maximum,
        'max_state_absolute_error_per_entry': per_entry_max.tolist(),
        'max_supervisor_action_error': action_max,
    }


def replay(stage):
    stage = Path(stage).resolve()
    destination = RUN / 'analysis' / f'{stage.name}_physics_replay.json'
    require(not destination.exists(), f'Refusing to overwrite replay receipt: {destination}')
    receipt = {
        'stage': str(stage), 'started_utc': datetime.now(timezone.utc).isoformat(),
        'script_sha256': sha(__file__), 'passed': False, 'episodes': [],
        'selection_rule': SELECTION_RULE, 'state_tolerance': STATE_TOLERANCE,
        'supervisor_action_tolerance': ACTION_TOLERANCE, 'additional_independent_samples': 0,
        'scope': 'Actual MuJoCo/OSC action replay and v5 supervisor regeneration conditioned on saved policy/WM proposals; no retraining or independent policy/WM inference.',
    }
    try:
        # The prospective selection is derived only from frozen jobs.json.
        jobs, started = read_json(stage / 'jobs.json'), read_json(stage / 'started.json')
        require(sha(stage / 'jobs.json') == started['jobs_sha256'], 'Frozen jobs hash mismatch')
        chosen = selection(jobs)
        receipt['selected_episodes'] = [{'job': job['id'], 'demos': demos} for job, demos in chosen]
        completed = read_json(stage / 'completed.json')
        require(completed.get('passed') is True, 'Physics replay requires a completed passing stage')
        trace_receipt_path = RUN / 'analysis' / f'{stage.name}_trace_verification.json'
        trace_receipt = read_json(trace_receipt_path)
        require(trace_receipt.get('passed') is True and trace_receipt.get('jobs_sha256') == started['jobs_sha256'],
                'Run the complete trace/hash verifier before physics replay')
        receipt['trace_verification_sha256'] = sha(trace_receipt_path)
        for package, version in started.get('packages', {}).items():
            actual = importlib.metadata.version(package)
            require(actual == version, f'Package version differs from stage: {package} {actual} != {version}')
        hashes = started['source_sha256']
        needed_inputs = {DEFAULT_DATA} | {job.get('dataset') or DEFAULT_DATA for job, _ in chosen}
        receipt['input_sha256'] = {}
        for relative in needed_inputs:
            digest = sha(relative_file(ROOT, relative))
            require(digest == hashes.get(relative), f'Replay dataset hash differs from stage: {relative}')
            receipt['input_sha256'][relative] = digest
        common, observations, controllers, sources = frozen_modules(stage, hashes)
        receipt['source_snapshot_sha256'] = sources
        import torch
        torch.set_num_threads(2)
        for job, demos in chosen:
            directory = stage / job['id']
            result = read_json(directory / 'result.json')
            require(result.get('status') == 'complete' and result.get('job') == job, 'Replay job is incomplete/changed')
            verified_job = next(item for item in trace_receipt['jobs'] if item['id'] == job['id'])
            for artifact, digest in verified_job['artifact_sha256'].items():
                require(sha(directory / artifact) == digest, f'Artifact changed since trace verification: {job["id"]}/{artifact}')
            rows = selected_rows(directory, result['records'], demos)
            with h5py.File(ROOT / (job.get('dataset') or DEFAULT_DATA), 'r') as dataset:
                env = common.make_env(False)
                try:
                    arm = env.robots[0].part_controllers['right']
                    require(arm.input_type == 'delta' and arm.input_ref_frame == 'world' and env.control_freq == 20,
                            'Unexpected OSC coordinate/control-frequency contract')
                    exact(arm.output_max, [.05, .05, .05, .5, .5, .5], 'OSC output scale')
                    for name in demos:
                        record = next(r for r in result['records'] if r['demo'] == name)
                        receipt['episodes'].append(replay_episode(common, observations, controllers, env, dataset,
                                                                  job, record, rows[name]))
                finally:
                    env.close()
        receipt['passed'] = True
        receipt['replayed_existing_episodes'] = len(receipt['episodes'])
        receipt['replayed_controls'] = sum(item['replayed_controls'] for item in receipt['episodes'])
    except Exception as error:
        receipt['error'] = f'{type(error).__name__}: {error}'
    receipt['finished_utc'] = datetime.now(timezone.utc).isoformat()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    return receipt, destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, default=RUN / 'confirmation')
    arguments = parser.parse_args()
    result, destination = replay(arguments.stage)
    print(json.dumps({'passed': result['passed'], 'receipt': str(destination),
                      'error': result.get('error'), 'episodes': len(result['episodes'])}, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
