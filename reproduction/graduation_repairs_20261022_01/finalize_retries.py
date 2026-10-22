"""Validate all retried artifacts before issuing a recovered completion receipt."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
FAILED = ('queue_budget_full_s17_0', 'observation_legacy_0.05')


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with path.open(encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def main():
    stage = RUN / 'confirmation'
    completed = read(stage / 'completed.json')
    if completed['passed']:
        raise ValueError('No failed batch to recover')
    jobs = read(stage / 'jobs.json')
    frozen = read(RUN / 'confirmation_lock.json')['source_sha256']
    for relative, expected in frozen.items():
        if sha(ROOT / relative) != expected:
            raise ValueError('Frozen input/source drift: ' + relative)
    for job in jobs:
        result = read(stage / job['id'] / 'result.json')
        if (result['status'] != 'complete' or result['job'] != job
                or [r['demo'] for r in result['records']] != job['demos']):
            raise ValueError('Not a complete exact registered job: ' + job['id'])
    prefixes = []
    for identifier in FAILED:
        archived = RUN / 'failed_attempts' / identifier
        old = read(archived / 'result.json')
        new = read(stage / identifier / 'result.json')
        for a, b in zip(old['records'], new['records']):
            for key in ('demo', 'success', 'steps', 'events', 'phases', 'selection_counts', 'attempts', 'stability'):
                if a[key] != b[key]:
                    raise ValueError(f'Retry prefix record differs: {identifier}/{a["demo"]}/{key}')
        replay = iter(rows(stage / identifier / 'trace.jsonl'))
        count, last_rows = 0, {}
        for old_row in rows(archived / 'trace.jsonl'):
            new_row = next(replay)
            a, b = dict(old_row), dict(new_row)
            a.pop('compute_ms'); b.pop('compute_ms')
            if a != b:
                raise ValueError(f'Retry prefix control differs: {identifier}/{a["demo"]}/{a["step"]}')
            count += 1; last_rows[a['demo']] = a
        complete_prefix = sum(r['success_after'] or r['step'] == 400 for r in last_rows.values())
        prefixes.append({'job_id': identifier, 'saved_result_records': len(old['records']),
            'matched_control_prefix_steps': count, 'trace_complete_duplicate_episodes': complete_prefix,
            'partial_trace_episodes': len(last_rows) - complete_prefix,
            'saved_prefix_decisions_exact': True, 'timing_excluded_from_exact_comparison': True,
            'archived_result_sha256': sha(archived / 'result.json'),
            'archived_trace_sha256': sha(archived / 'trace.jsonl'),
            'completed_retry_result_sha256': sha(stage / identifier / 'result.json')})
    second_receipt = read(RUN / 'failed_attempts/observation_legacy_0.05_retry_receipt.json')
    if second_receipt['returncode'] != 0 or second_receipt['records'] != 60:
        raise ValueError('Atomic-I/O retry did not finish')
    recovery = {'passed': True, 'frozen_control_model_data_unchanged': True,
                'partial_failures_preserved': list(FAILED), 'prefix_comparisons': prefixes,
                'extra_complete_duplicate_episodes': sum(p['trace_complete_duplicate_episodes'] for p in prefixes),
                'extra_partial_trace_episodes': sum(p['partial_trace_episodes'] for p in prefixes),
                'registered_final_executions': 1680, 'independent_starts': 60,
                'io_adapter_scope': 'Atomic and retried persistence only; no decision/randomness/physics code changes.',
                'finished_utc': datetime.now(timezone.utc).isoformat()}
    destination = RUN / 'analysis/io_recovery_verification.json'
    with destination.open('x', encoding='utf-8') as f:
        json.dump(recovery, f, indent=2)
    initial_path = RUN / 'failed_attempts/batch_completed_initial.json'
    with initial_path.open('xb') as f:
        f.write((stage / 'completed.json').read_bytes())
    updated = dict(completed)
    updated.update(passed=True, source_unchanged=True, episodes=1680,
        jobs=[{'id': job['id'], 'returncode': 0, 'attempts': 2 if job['id'] in FAILED else 1} for job in jobs],
        finished_utc=datetime.now(timezone.utc).isoformat(),
        initial_failed_receipt=initial_path.relative_to(ROOT).as_posix(),
        io_recovery_receipt=destination.relative_to(ROOT).as_posix(),
        extra_complete_duplicate_episodes=recovery['extra_complete_duplicate_episodes'],
        extra_partial_trace_episodes=recovery['extra_partial_trace_episodes'],
        elapsed_scope='elapsed_s retains original failed batch wall time; rerun overhead is separate')
    (stage / 'completed.json').write_text(json.dumps(updated, indent=2), encoding='utf-8')
    print(json.dumps(recovery))


if __name__ == '__main__':
    main()
