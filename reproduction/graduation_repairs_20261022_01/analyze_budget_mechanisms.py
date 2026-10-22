"""Trace-only, paired mechanism audit of queue_budget versus frozen v5.

No outcomes are filtered. Actual interventions require saved transition events;
pressure-estimate diagnostics alone never count as a recovery intervention.
Requires a completed stage and writes a fresh analysis/budget_mechanisms.json.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from verify_traces import RUN, array, read_json, require, sha


TIMING_REASONS = {'early_stall_preserve_budget', 'descent_stalled_preserve_budget'}
EVENT_FIELDS = ('step', 'from', 'to', 'reason', 'attempt')


def record_hash(record):
    serialized = json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


def matching_key(job):
    return json.dumps({key: value for key, value in job.items() if key not in ('id', 'controller')},
                      sort_keys=True, separators=(',', ':'))


def registered_pairs(jobs):
    parents = {}
    for job in jobs:
        if job.get('controller') == 'v5':
            key = matching_key(job)
            require(key not in parents, 'Ambiguous v5 parent registration')
            parents[key] = job
    pairs = []
    for job in jobs:
        if job.get('controller') == 'queue_budget':
            key = matching_key(job)
            require(key in parents, f'No identical v5 registration for {job["id"]}')
            pairs.append((parents[key], job))
    require(bool(pairs), 'No registered queue_budget/v5 pairs')
    return pairs


def event_key(event):
    # Position fields respond to already-different physics; diagnostic payloads
    # are not behavioral differences. Compare actual transition identities.
    return tuple(event.get(key) for key in EVENT_FIELDS)


def event_summary(event):
    return None if event is None else {key: event.get(key) for key in EVENT_FIELDS}


def first_event_difference(parent, candidate):
    for index in range(max(len(parent), len(candidate))):
        old = parent[index] if index < len(parent) else None
        new = candidate[index] if index < len(candidate) else None
        if old is None or new is None or event_key(old) != event_key(new):
            steps = [event['step'] for event in (old, new) if event is not None]
            return {'event_index': index, 'earliest_control': min(steps),
                    'parent': event_summary(old), 'candidate': event_summary(new)}
    return None


def estimate_summary(budget):
    if not budget or not budget.get('enabled'):
        return None
    remaining = budget['remaining_horizon_steps']
    estimate = budget['estimated_steps']
    base = budget['base_steps']
    bounded = budget['phase_steps_bounded']
    motion = float(budget['queue_motion_steps'])
    wait = float(budget['queue_wait_steps'])
    candidate_pressure = remaining < estimate
    noqueue_pressure = remaining < math.ceil(bounded)
    return {
        'remaining_controls': remaining, 'v5_base_steps': base, 'estimated_steps': estimate,
        'phase_steps_before_queue': bounded, 'v5_base_pressure_on_same_observation': remaining < base,
        'candidate_pressure': candidate_pressure, 'pressure_without_queue_term': noqueue_pressure,
        'queue_term_flips_pressure': bool(candidate_pressure != noqueue_pressure),
        'queue_available': budget.get('queue_available', False),
        'queue_motion_steps': motion, 'queue_wait_steps': wait, 'net_queue_steps': motion+wait,
        'queue_motion_direction': ('toward_next_waypoint' if motion < -1e-9 else
                                   'away_from_next_waypoint' if motion > 1e-9 else 'neutral'),
        'queue_projection_m': budget.get('queue_projection_m'),
        'speed_m_per_step': budget['speed_m_per_step'], 'speed_sample_count': budget['speed_sample_count'],
        'observed_progress_m_per_step': budget['observed_progress_m_per_step'],
    }


def read_episode(stream, record, job_id):
    rows = []
    for step in range(1, record['steps']+1):
        line = stream.readline()
        require(bool(line), f'{job_id}/{record["demo"]}: incomplete trace')
        row = json.loads(line)
        require(row.get('demo') == record['demo'] and row.get('step') == step,
                f'{job_id}/{record["demo"]}: trace ordering mismatch')
        rows.append(row)
    return rows


def first_array_difference(parent_rows, candidate_rows, key, shape):
    for old, new in zip(parent_rows, candidate_rows):
        step = old['step']
        a = array(old, key, shape, f'parent/step={step}')
        b = array(new, key, shape, f'candidate/step={step}')
        if not np.array_equal(a, b):
            result = {'step': step, 'max_absolute_difference': float(np.abs(b-a).max())}
            if shape == (7,):
                result.update(parent=a.tolist(), candidate=b.tolist())
            return result
    return None


def analyze_episode(parent_job, candidate_job, parent, candidate, old_rows, new_rows):
    require(parent['demo'] == candidate['demo'], 'Paired record demo mismatch')
    old_timing = [event for event in parent['events'] if event['reason'] in TIMING_REASONS]
    new_timing = [event for event in candidate['events'] if event['reason'] in TIMING_REASONS]
    old_keys, new_keys = {event_key(e) for e in old_timing}, {event_key(e) for e in new_timing}
    novel = []
    for event in new_timing:
        if event_key(event) in old_keys:
            continue
        step = event['step']
        require(1 <= step <= len(new_rows), 'Timing event is outside its saved benchmark trace')
        actual = new_rows[step-1]['recovery']
        require(actual.get('intervened') is True and actual.get('mode') == event['to'],
                'Timing transition is not backed by an actual saved control intervention')
        budget = estimate_summary(event.get('queue_budget'))
        require(budget is not None and budget['candidate_pressure'], 'Actual timing event lacks an active pressure estimate')
        novel.append({**event_summary(event), 'budget': budget,
                      'object_position': event.get('object_position')})
    removed = [event_summary(e) for e in old_timing if event_key(e) not in new_keys]
    mode_difference = None
    for old, new in zip(old_rows, new_rows):
        a, b = old['recovery'], new['recovery']
        fields = ('mode_before', 'mode', 'next_mode', 'intervened', 'attempts')
        if any(a.get(key) != b.get(key) for key in fields):
            mode_difference = {'step': old['step'], 'parent': {k: a.get(k) for k in fields},
                               'candidate': {k: b.get(k) for k in fields}}
            break
    submitted = first_array_difference(old_rows, new_rows, 'submitted_action', (7,))
    applied = first_array_difference(old_rows, new_rows, 'executed_action', (7,))
    states = first_array_difference(old_rows, new_rows, 'state_before', (23,))
    proposals = first_array_difference(old_rows, new_rows, 'proposal', (7,))
    diagnostics = Counter()
    for row in new_rows:
        estimate = estimate_summary(row['recovery'].get('queue_budget'))
        diagnostics['frames_with_estimate'] += int(estimate is not None)
        if estimate:
            diagnostics['pressure_frames'] += int(estimate['candidate_pressure'])
            diagnostics['pressure_disagrees_with_v5_base_frames'] += int(
                estimate['candidate_pressure'] != estimate['v5_base_pressure_on_same_observation'])
            diagnostics['queue_term_flips_pressure_frames'] += int(estimate['queue_term_flips_pressure'])
    divergence_step = submitted['step'] if submitted else None
    same_state_prefix = (states is None or (divergence_step is not None and states['step'] > divergence_step))
    both_success = parent['success'] and candidate['success']
    return {
        'parent_job': parent_job['id'], 'candidate_job': candidate_job['id'], 'demo': parent['demo'],
        'method': candidate_job['method'], 'world_model_seed': candidate_job.get('seed'),
        'noise': candidate_job.get('noise', 0.),
        'parent_success': parent['success'], 'candidate_success': candidate['success'],
        'parent_steps': parent['steps'], 'candidate_steps': candidate['steps'],
        'success_delta': int(candidate['success'])-int(parent['success']),
        'steps_delta_candidate_minus_parent': candidate['steps']-parent['steps'],
        'both_succeeded': bool(both_success),
        'first_submitted_action_divergence': submitted, 'first_applied_action_divergence': applied,
        'first_mode_divergence': mode_difference,
        'first_event_divergence': first_event_difference(parent['events'], candidate['events']),
        'first_observed_state_divergence': states, 'first_policy_proposal_divergence': proposals,
        'same_observations_through_first_submitted_action_divergence': same_state_prefix,
        'different_termination_length': parent['steps'] != candidate['steps'],
        'parent_actual_timing_trigger_count': len(old_timing),
        'candidate_actual_timing_trigger_count': len(new_timing),
        'candidate_novel_timing_triggers': novel, 'parent_timing_triggers_not_repeated': removed,
        'novel_timing_trigger_at_first_submitted_divergence': bool(
            divergence_step is not None and any(e['step'] == divergence_step for e in novel)),
        'estimate_diagnostics_only': dict(diagnostics),
        'parent_record_sha256': record_hash(parent), 'candidate_record_sha256': record_hash(candidate),
    }


def summarize(pairs):
    outcomes = Counter()
    diagnostics = Counter()
    for pair in pairs:
        key = ('candidate_win' if pair['success_delta'] > 0 else 'candidate_regression'
               if pair['success_delta'] < 0 else 'both_success' if pair['both_succeeded'] else 'both_failure')
        outcomes[key] += 1
        diagnostics.update(pair['estimate_diagnostics_only'])
    novel = [event for pair in pairs for event in pair['candidate_novel_timing_triggers']]
    deltas = np.asarray([p['steps_delta_candidate_minus_parent'] for p in pairs if p['both_succeeded']], dtype=float)
    return {
        'paired_registered_episodes': len(pairs), 'registered_start_ids': len({p['demo'] for p in pairs}),
        'outcome_counts': dict(outcomes), 'net_success_delta': sum(p['success_delta'] for p in pairs),
        'episodes_with_submitted_action_divergence': sum(p['first_submitted_action_divergence'] is not None for p in pairs),
        'episodes_with_applied_action_divergence': sum(p['first_applied_action_divergence'] is not None for p in pairs),
        'episodes_with_mode_divergence': sum(p['first_mode_divergence'] is not None for p in pairs),
        'episodes_with_event_divergence': sum(p['first_event_divergence'] is not None for p in pairs),
        'diverged_episodes_with_identical_observations_through_first_action_difference': sum(
            p['first_submitted_action_divergence'] is not None
            and p['same_observations_through_first_submitted_action_divergence'] for p in pairs),
        'pairs_with_identical_shared_observation_trace': sum(p['first_observed_state_divergence'] is None for p in pairs),
        'pairs_without_submitted_action_difference': sum(p['first_submitted_action_divergence'] is None for p in pairs),
        'parent_actual_timing_triggers': sum(p['parent_actual_timing_trigger_count'] for p in pairs),
        'candidate_actual_timing_triggers': sum(p['candidate_actual_timing_trigger_count'] for p in pairs),
        'candidate_novel_actual_timing_triggers': len(novel),
        'novel_actual_triggers_where_v5_base_pressure_was_false': sum(
            not e['budget']['v5_base_pressure_on_same_observation'] for e in novel),
        'novel_actual_triggers_where_queue_term_flipped_pressure': sum(e['budget']['queue_term_flips_pressure'] for e in novel),
        'novel_trigger_at_first_submitted_divergence_episodes': sum(
            p['novel_timing_trigger_at_first_submitted_divergence'] for p in pairs),
        'diagnostic_estimate_frames_not_interventions': dict(diagnostics),
        'both_succeeded_steps': {
            'pairs': len(deltas), 'candidate_faster': int((deltas < 0).sum()),
            'equal': int((deltas == 0).sum()), 'candidate_slower': int((deltas > 0).sum()),
            'mean_delta_candidate_minus_parent': float(deltas.mean()) if len(deltas) else None,
            'median_delta_candidate_minus_parent': float(np.median(deltas)) if len(deltas) else None,
            'min_delta': int(deltas.min()) if len(deltas) else None,
            'max_delta': int(deltas.max()) if len(deltas) else None,
        },
        'candidate_win_pairs': [{'job': p['candidate_job'], 'demo': p['demo']} for p in pairs if p['success_delta'] > 0],
        'candidate_regression_pairs': [{'job': p['candidate_job'], 'demo': p['demo']} for p in pairs if p['success_delta'] < 0],
    }


def analyze(stage):
    stage = Path(stage).resolve()
    destination = RUN / 'analysis' / 'budget_mechanisms.json'
    require(not destination.exists(), f'Refusing to overwrite mechanism receipt: {destination}')
    receipt = {
        'stage': str(stage), 'created_utc': datetime.now(timezone.utc).isoformat(),
        'script_sha256': sha(__file__), 'passed': False, 'pairs': [], 'job_pairs': [],
        'scope': 'Saved-trace causal sequence audit; no simulation, no new samples, no outcome filtering. Differences after an initial intervention may be downstream consequences, not independent causes.',
        'event_comparison': 'Compare step/from/to/reason/attempt; diagnostic payloads and object positions alone are not event divergence.',
        'pressure_scope': 'Same-observation pressure comparisons concern the arithmetic estimate; actual interventions require saved timing transition events.',
        'independence': 'Conditions and world-model seeds repeat the same registered starts; episode pairs are not independent-start samples.',
    }
    try:
        completed = read_json(stage / 'completed.json')
        require(completed.get('passed') is True, 'Mechanism analysis requires a completed passing stage')
        jobs, started = read_json(stage / 'jobs.json'), read_json(stage / 'started.json')
        require(sha(stage / 'jobs.json') == started['jobs_sha256'], 'Frozen job hash differs')
        receipt.update(jobs_sha256=sha(stage / 'jobs.json'), started_sha256=sha(stage / 'started.json'),
                       completed_sha256=sha(stage / 'completed.json'))
        relevant = ['common.py', 'robot_can_v2/budget.py', 'robot_can_v2/evaluate.py',
                    'reproduction/recovery_v5_20261012_01/recovery_controller.py']
        receipt['source_snapshot_sha256'] = {}
        for relative in relevant:
            digest = sha(stage / 'source' / relative)
            require(digest == started['source_sha256'].get(relative), f'Source snapshot mismatch: {relative}')
            receipt['source_snapshot_sha256'][relative] = digest
        trace_audit_path = RUN / 'analysis' / f'{stage.name}_trace_verification.json'
        audit_jobs = {}
        if trace_audit_path.is_file():
            audit = read_json(trace_audit_path)
            require(audit.get('passed') is True and audit.get('jobs_sha256') == started['jobs_sha256'], 'Existing trace audit did not pass this registration')
            receipt['trace_verification_sha256'] = sha(trace_audit_path)
            audit_jobs = {job['id']: job for job in audit['jobs']}
        for parent_job, candidate_job in registered_pairs(jobs):
            parent_dir, candidate_dir = stage / parent_job['id'], stage / candidate_job['id']
            parent, candidate = read_json(parent_dir / 'result.json'), read_json(candidate_dir / 'result.json')
            for job, result in [(parent_job, parent), (candidate_job, candidate)]:
                require(result.get('status') == 'complete' and result.get('job') == job, f'{job["id"]}: incomplete/changed result')
                require([r['demo'] for r in result['records']] == job['demos'], f'{job["id"]}: result ordering mismatch')
            artifacts = {job['id']: {name: sha(stage / job['id'] / name) for name in ('result.json', 'trace.jsonl')}
                         for job in (parent_job, candidate_job)}
            for job_id, hashes in artifacts.items():
                if audit_jobs:
                    for name, digest in hashes.items():
                        require(audit_jobs[job_id]['artifact_sha256'][name] == digest, f'{job_id}: changed since trace audit')
            offset = len(receipt['pairs'])
            with (parent_dir / 'trace.jsonl').open(encoding='utf-8') as old_trace, \
                    (candidate_dir / 'trace.jsonl').open(encoding='utf-8') as new_trace:
                for old_record, new_record in zip(parent['records'], candidate['records']):
                    old_rows = read_episode(old_trace, old_record, parent_job['id'])
                    new_rows = read_episode(new_trace, new_record, candidate_job['id'])
                    receipt['pairs'].append(analyze_episode(parent_job, candidate_job, old_record, new_record,
                                                            old_rows, new_rows))
                require(not old_trace.readline() and not new_trace.readline(), 'Unregistered extra trace rows')
            job_pairs = receipt['pairs'][offset:]
            receipt['job_pairs'].append({'parent': parent_job['id'], 'candidate': candidate_job['id'],
                                         'artifact_sha256': artifacts, 'summary': summarize(job_pairs)})
        receipt['summary'] = summarize(receipt['pairs'])
        receipt['passed'] = True
    except Exception as error:
        receipt['error'] = f'{type(error).__name__}: {error}'
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    return receipt, destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, default=RUN / 'confirmation')
    arguments = parser.parse_args()
    receipt, destination = analyze(arguments.stage)
    print(json.dumps({'passed': receipt['passed'], 'receipt': str(destination),
                      'error': receipt.get('error'), 'summary': receipt.get('summary')}, indent=2))
    return 0 if receipt['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
