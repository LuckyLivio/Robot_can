"""Rerun a frozen registered job with atomic/retried result-file I/O only.

This adapter changes no controller/model/action code. It affects write_text
on this fresh job's result.json only, after a completed benchmark episode.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(job_id, output):
    from robot_can_v2 import evaluate
    jobs_path = RUN / 'confirmation_jobs.json'
    jobs = json.loads(jobs_path.read_text())
    job = next(j for j in jobs if j['id'] == job_id)
    lock = json.loads((RUN / 'confirmation_lock.json').read_text())
    for relative, expected in lock['source_sha256'].items():
        if sha(ROOT / relative) != expected:
            raise ValueError('Frozen input drift: ' + relative)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if not output.is_relative_to(RUN.resolve()):
        raise ValueError('Retry output must remain in this study')
    result_path = output / 'result.json'
    original_write = Path.write_text
    io_events = []
    def reliable_write(path, data, encoding=None, errors=None, newline=None):
        if path.resolve() != result_path:
            return original_write(path, data, encoding=encoding, errors=errors, newline=newline)
        temporary = path.with_name(f'.result-{os.getpid()}.io.tmp')
        for attempt in range(20):
            try:
                with temporary.open('w', encoding=encoding or 'utf-8', errors=errors, newline=newline) as f:
                    count = f.write(data); f.flush(); os.fsync(f.fileno())
                os.replace(temporary, path)
                return count
            except OSError as error:
                if error.errno not in (13, 22) and getattr(error, 'winerror', None) not in (5, 32, 33):
                    raise
                io_events.append({'attempt': attempt + 1, 'error': repr(error)})
                if attempt == 19:
                    raise
                time.sleep(.05 * (attempt + 1))
    receipt = {'job_id': job_id, 'job': job, 'jobs_sha256': sha(jobs_path),
               'adapter_sha256': sha(__file__), 'scope': 'Result-file persistence only; frozen model/controller/data/config unchanged.',
               'returncode': 1, 'io_retry_events': io_events}
    try:
        Path.write_text = reliable_write
        result = evaluate.run(job, output)
        receipt.update(returncode=0, records=len(result['records']), status=result['status'])
    finally:
        Path.write_text = original_write
        destination = RUN / 'failed_attempts' / f'{job_id}_retry_receipt.json'
        with destination.open('x', encoding='utf-8') as f:
            json.dump(receipt, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    run(args.job_id, args.output_dir)
