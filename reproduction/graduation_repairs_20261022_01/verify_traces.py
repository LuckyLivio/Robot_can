"""Read-only verification of frozen v2 study outputs; no physics/model execution.

Usage: python reproduction/graduation_repairs_20261022_01/verify_traces.py --stage PATH
Writes a new analysis/<stage-name>_trace_verification.json; never overwrites it.
Python source provenance is checked against the stage's saved source snapshot,
so intentional later edits to the active implementation do not invalidate it.
"""
import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
DEFAULT_DATA = 'data/can_ph_lowdim_v15.hdf5'
DEFAULT_POLICY = 'policy/bc_chunk_dev_v4/policy_15000.pt'


class VerificationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def finite_float(value):
    result = float(value)
    require(np.isfinite(result), 'Nonfinite JSON numeric value')
    return result


def forbidden_constant(value):
    raise VerificationError(f'Nonstandard JSON constant: {value}')


def parse_json(text):
    return json.loads(text, parse_float=finite_float, parse_constant=forbidden_constant)


def read_json(path):
    return parse_json(Path(path).read_text(encoding='utf-8'))


def relative_file(base, relative):
    relative = Path(relative)
    require(not relative.is_absolute(), f'Expected project-relative file: {relative}')
    result = (base / relative).resolve()
    require(result.is_relative_to(base.resolve()), f'Path escapes registered base: {relative}')
    require(result.is_file(), f'Missing registered file: {result}')
    return result


def array(row, key, shape, context):
    result = np.asarray(row.get(key), dtype=float)
    require(result.shape == shape, f'{context}: {key} shape {result.shape}, expected {shape}')
    require(np.isfinite(result).all(), f'{context}: {key} contains nonfinite values')
    return result


def exact(actual, expected, context):
    require(np.array_equal(actual, expected), f'{context}: arrays differ')


def close(actual, expected, context, atol=1e-8):
    require(np.isfinite(actual) and np.isclose(actual, expected, rtol=1e-9, atol=atol),
            f'{context}: {actual!r} != {expected!r}')


def next_row(stream, context):
    line = stream.readline()
    require(bool(line), f'{context}: trace ended early')
    require(bool(line.strip()), f'{context}: empty trace row')
    return parse_json(line)


def timing_check(stored, values, context):
    values = np.asarray(values, dtype=float)
    require(stored.get('n') == len(values), f'{context}: timing sample count is not benchmark steps')
    expected = {
        'p50_ms': np.percentile(values, 50), 'p95_ms': np.percentile(values, 95),
        'p99_ms': np.percentile(values, 99), 'max_ms': values.max(),
        'over_50ms_fraction': np.mean(values > 50),
    }
    for key, value in expected.items():
        close(stored.get(key), float(value), f'{context}: timing {key}')
    require(stored.get('over_50ms') == int((values > 50).sum()), f'{context}: timing threshold count mismatch')


def verify_job(job, directory):
    """Verify one complete job; usable independently for a completed preflight."""
    directory = Path(directory)
    context = str(job['id'])
    result = read_json(directory / 'result.json')
    require(result.get('status') == 'complete', f'{context}: job is not complete')
    require(result.get('job') == job, f'{context}: saved job differs from frozen jobs.json')
    require(job.get('steps', 400) == 400 and job.get('delay', 2) == 2, f'{context}: study is not D2 / 400 steps')
    require(job.get('settle_steps', 20) == 20, f'{context}: diagnostic length differs from 20')
    noise = finite_float(job.get('noise', 0.))
    require(noise >= 0, f'{context}: negative noise')
    names = job['demos']
    require(names and len(set(names)) == len(names), f'{context}: duplicate/empty demo registration')
    records = result.get('records', [])
    require([r.get('demo') for r in records] == names, f'{context}: record demo ordering differs from registration')
    config = result.get('controller_config', {})
    for key, value in [('delay', 2), ('horizon_steps', 400), ('max_attempts', 2), ('stable_frames', 5)]:
        require(config.get(key) == value, f'{context}: controller {key} differs from frozen contract')
    benchmark_steps = compute_samples = successes = settling_rows = 0
    with (directory / 'trace.jsonl').open(encoding='utf-8') as trace, \
            (directory / 'settling_trace.jsonl').open(encoding='utf-8') as settling:
        for record in records:
            name = record['demo']
            label = f'{context}/{name}'
            steps = record.get('steps')
            require(isinstance(steps, int) and 1 <= steps <= 400, f'{label}: invalid benchmark steps')
            success = record.get('success')
            require(isinstance(success, bool), f'{label}: success must be Boolean')
            require(success or steps == 400, f'{label}: failed episode stopped before 400 controls')
            require(0 <= record.get('attempts', -1) <= 2, f'{label}: attempt budget exceeded')
            pending = deque([np.zeros(7, np.float32), np.zeros(7, np.float32)])
            rng = np.random.default_rng(70000 + int(name.split('_')[-1]))
            previous = np.zeros(7, np.float32)
            last_after = None
            timings, phases, selections = [], Counter(), Counter()
            interventions = 0
            for step in range(1, steps + 1):
                where = f'{label}/step={step}'
                row = next_row(trace, where)
                require(row.get('demo') == name and row.get('step') == step, f'{where}: trace demo/step ordering mismatch')
                before = array(row, 'state_before', (23,), where)
                after = array(row, 'state_after', (23,), where)
                if last_after is not None:
                    exact(before, last_after, f'{where}: state continuity')
                last_after = after
                queued = array(row, 'queue_before', (2, 7), where)
                exact(queued, np.asarray(pending), f'{where}: D2 queue recurrence/reset')
                base = array(row, 'base_chunk', (8, 7), where)
                proposal = array(row, 'proposal', (7,), where)
                selected = row.get('selected')
                require(isinstance(selected, int) and 0 <= selected < 4, f'{where}: invalid selected action')
                exact(proposal, base[selected], f'{where}: proposal is not selected BC chunk action')
                if job['method'] == 'bc':
                    require(selected == 2 and row.get('scores') is None, f'{where}: BC selection changed')
                else:
                    scores = array(row, 'scores', (4,), where).astype(np.float32)
                    raw = int(scores.argmax())
                    expected_selected = 2 if raw != 2 and float(scores[raw]-scores[2]) < .02 else raw
                    require(selected == expected_selected, f'{where}: WM selection margin mismatch')
                submitted = array(row, 'submitted_action', (7,), where)
                require((np.abs(submitted) <= 1).all(), f'{where}: submitted action outside OSC input bounds')
                for previous_key in ('previous_action', 'previous_applied_action', 'previous_executed_action'):
                    if previous_key in row:
                        exact(array(row, previous_key, (7,), where), previous, f'{where}: {previous_key}')
                pending.append(submitted.astype(np.float32))
                executed = pending.popleft().copy()
                exact(array(row, 'executed_action_before_noise', (7,), where), executed, f'{where}: FIFO execution before noise')
                if noise:
                    # Replay precisely the evaluator's float32 in-place add.
                    executed[:3] += rng.normal(0, noise, 3)
                applied = np.clip(executed, -1, 1)
                exact(array(row, 'executed_action', (7,), where), applied, f'{where}: seeded noise / clipping')
                previous = applied.copy()
                flag = row.get('success_after')
                require(isinstance(flag, bool), f'{where}: success_after must be Boolean')
                require(flag == (success and step == steps), f'{where}: first-success termination mismatch')
                elapsed = finite_float(row.get('compute_ms'))
                require(elapsed >= 0, f'{where}: negative compute duration')
                timings.append(elapsed)
                diagnostic = row.get('recovery', {})
                phases[diagnostic.get('mode')] += 1
                selections[selected] += 1
                interventions += int(diagnostic.get('intervened', False))
                require(0 <= diagnostic.get('attempts', -1) <= 2, f'{where}: debug attempts exceed budget')
            timing_check(record.get('compute_timing', {}), timings, label)
            require(record.get('phases') == dict(phases), f'{label}: phase counts differ from trace')
            require(record.get('selection_counts') == {str(k): v for k, v in selections.items()}, f'{label}: selection counts differ from trace')
            require(record.get('intervention_steps') == interventions, f'{label}: intervention count differs from trace')
            for event in record.get('events', []):
                require(1 <= event.get('step', 0) <= steps, f'{label}: event outside benchmark budget')
                require(0 <= event.get('attempt', -1) <= 2, f'{label}: event exceeds attempt budget')
            stability = record.get('stability', {})
            if success:
                require(stability.get('observed') is True and stability.get('diagnostic_controls') == 20,
                        f'{label}: successful episode missing 20 separate diagnostics')
                positions = last_after[16:19].astype(np.float32)
                goal_flags, speeds = [], []
                for k in range(1, 21):
                    where = f'{label}/diagnostic={k}'
                    row = next_row(settling, where)
                    require(row.get('demo') == name and row.get('diagnostic_step') == k,
                            f'{where}: settling ordering mismatch')
                    position = array(row, 'object_position', (3,), where).astype(np.float32)
                    speed = float(np.linalg.norm(position-positions)*20.)
                    close(row.get('speed_m_s'), speed, f'{where}: observed object speed', atol=1e-7)
                    positions = position
                    require(isinstance(row.get('benchmark_success'), bool), f'{where}: goal flag must be Boolean')
                    goal_flags.append(row['benchmark_success']); speeds.append(speed)
                require(stability.get('all_in_goal') == all(goal_flags), f'{label}: diagnostic goal aggregate mismatch')
                close(stability.get('last10_max_speed_m_s'), max(speeds[-10:]), f'{label}: diagnostic final speed', atol=1e-7)
                require(stability.get('stable') == bool(all(goal_flags) and max(speeds[-10:]) < .02),
                        f'{label}: diagnostic stable flag mismatch')
                successes += 1; settling_rows += 20
            else:
                require(stability.get('observed') is False and stability.get('stable') is False,
                        f'{label}: failed episode incorrectly counted as observed stable')
            benchmark_steps += steps
            compute_samples += record['compute_timing']['n']
        require(not trace.readline(), f'{context}: extra benchmark trace rows after registered episodes')
        require(not settling.readline(), f'{context}: extra settling rows or failed episodes in settling trace')
    require(compute_samples == benchmark_steps, f'{context}: diagnostics contaminated compute denominator')
    for key, expected in [('total_compute_samples', compute_samples), ('benchmark_steps', benchmark_steps)]:
        if key in result:
            require(result[key] == expected, f'{context}: {key} total mismatch')
    return {
        'id': job['id'], 'passed': True, 'episodes': len(records), 'benchmark_successes': successes,
        'benchmark_steps': benchmark_steps, 'compute_samples': compute_samples,
        'settling_rows': settling_rows, 'settling_independent_samples': 0,
        'artifact_sha256': {name: sha(directory / name) for name in ('result.json', 'trace.jsonl', 'settling_trace.jsonl')},
    }


def verify_stage(stage):
    """Verify a finished stage and write a fresh success/failure audit receipt."""
    stage = Path(stage).resolve()
    destination = RUN / 'analysis' / f'{stage.name}_trace_verification.json'
    require(not destination.exists(), f'Refusing to overwrite verification receipt: {destination}')
    receipt = {
        'stage': str(stage), 'verified_utc': datetime.now(timezone.utc).isoformat(),
        'verifier_sha256': sha(__file__), 'passed': False, 'jobs': [],
        'scope': 'Read-only JSON/hash verification; no physics replay or independent success-oracle validation.',
        'settling_action_scope': 'Settling trace stores positions/success/speed, not actions; only benchmark FIFO/noise can be replayed from logged commands.',
    }
    try:
        jobs_path, started_path, completed_path = (stage / name for name in ('jobs.json', 'started.json', 'completed.json'))
        jobs, started, completed = read_json(jobs_path), read_json(started_path), read_json(completed_path)
        require(jobs and len({j['id'] for j in jobs}) == len(jobs), 'Empty/duplicate stage jobs')
        exact_ids = {j['id'] for j in jobs}
        require(completed.get('passed') is True, 'Stage completion receipt did not pass')
        outcomes = completed.get('jobs', [])
        require(len(outcomes) == len(jobs) and {o['id'] for o in outcomes} == exact_ids
                and all(o.get('returncode') == 0 for o in outcomes), 'Stage process outcomes differ from registered jobs')
        require(sha(jobs_path) == started.get('jobs_sha256'), 'Stage jobs.json hash differs from startup lock')
        expected_episodes = sum(len(j['demos']) for j in jobs)
        require(started.get('registered_episodes') == expected_episodes and completed.get('episodes') == expected_episodes,
                'Stage episode denominator differs from registration')
        hashes = started.get('source_sha256', {})
        require(hashes, 'Missing startup source/input hashes')
        verified_sources, verified_inputs = {}, {}
        for relative, digest in hashes.items():
            if relative.endswith('.py'):
                path = relative_file(stage / 'source', relative)
                require(sha(path) == digest, f'Saved Python source hash mismatch: {relative}')
                verified_sources[relative] = digest
            else:
                path = relative_file(ROOT, relative)
                require(sha(path) == digest, f'Input hash mismatch: {relative}')
                verified_inputs[relative] = {'sha256': digest, 'provenance': 'stage started.json'}
        # Confirmation has an additional prospective lock containing its
        # amendment, design, native starts and registration. Validate Python
        # entries against the archived snapshot; intentional later active-code
        # changes are outside the frozen experiment.
        confirmation_lock = stage.parent / 'confirmation_lock.json'
        if stage.name == 'confirmation':
            lock = read_json(confirmation_lock)
            require(lock.get('episodes') == expected_episodes, 'Confirmation lock episode count mismatch')
            independent = len({name for job in jobs for name in job['demos']})
            require(lock.get('independent_starts') == independent, 'Confirmation lock start count mismatch')
            for relative, digest in lock.get('source_sha256', {}).items():
                if relative.endswith('.py'):
                    require(verified_sources.get(relative) == digest, f'Confirmation Python snapshot lock mismatch: {relative}')
                elif relative in verified_inputs:
                    require(verified_inputs[relative]['sha256'] == digest, f'Confirmation input lock mismatch: {relative}')
                else:
                    require(sha(relative_file(ROOT, relative)) == digest, f'Confirmation metadata/input hash mismatch: {relative}')
                    verified_inputs[relative] = {'sha256': digest, 'provenance': 'confirmation_lock.json'}
            receipt['confirmation_lock_sha256'] = sha(confirmation_lock)
        # Development startup locks predate explicit inclusion of the default
        # HDF5. Its pre-edit preservation manifest supplies an honest fallback;
        # do not pretend that this hash was recorded in the stage startup lock.
        for job in jobs:
            policy = job.get('policy', DEFAULT_POLICY)
            require(policy in verified_inputs, f'Policy lacks startup hash: {policy}')
            result = read_json(stage / job['id'] / 'result.json')
            require(result.get('policy_sha256') == verified_inputs[policy]['sha256'], f'{job["id"]}: loaded policy hash mismatch')
            if job['method'] == 'full':
                model = f'models/multiseed_20261012_01_full_s{job["seed"]}/model.pt'
                require(model in verified_inputs, f'World model lacks startup hash: {model}')
            dataset = job.get('dataset') or DEFAULT_DATA
            if dataset not in verified_inputs:
                require(dataset == DEFAULT_DATA, f'Dataset lacks startup hash: {dataset}')
                manifest = read_json(RUN / 'before_manifest.json')
                digest = manifest.get('files', {}).get(dataset)
                require(digest and sha(relative_file(ROOT, dataset)) == digest, f'Default dataset differs from pre-edit manifest: {dataset}')
                verified_inputs[dataset] = {'sha256': digest, 'provenance': 'before_manifest.json (default dataset absent from startup lock)'}
            receipt['jobs'].append(verify_job(job, stage / job['id']))
        steps = sum(j['benchmark_steps'] for j in receipt['jobs'])
        compute = sum(j['compute_samples'] for j in receipt['jobs'])
        require(steps == compute, 'Total compute samples differ from benchmark controls')
        receipt.update(
            passed=True, benchmark_episodes=expected_episodes,
            independent_start_ids=len({name for job in jobs for name in job['demos']}),
            benchmark_steps=steps, compute_samples=compute,
            settling_rows=sum(j['settling_rows'] for j in receipt['jobs']), settling_independent_samples=0,
            source_snapshot_sha256=verified_sources, verified_input_sha256=verified_inputs,
            jobs_sha256=sha(jobs_path), started_sha256=sha(started_path), completed_sha256=sha(completed_path),
        )
    except Exception as error:
        receipt['error'] = f'{type(error).__name__}: {error}'
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    return receipt, destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, type=Path)
    arguments = parser.parse_args()
    receipt, destination = verify_stage(arguments.stage)
    print(json.dumps({'passed': receipt['passed'], 'receipt': str(destination),
                      'error': receipt.get('error'), 'benchmark_steps': receipt.get('benchmark_steps')}, indent=2))
    return 0 if receipt['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
