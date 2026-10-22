"""Freeze or verify pre-existing tracked assets; no historical mutations."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
DOCUMENTS = ['README.md', 'STATUS.md', 'PROJECT_NAVIGATION.md', 'PROGRAM_GUIDE.md', 'RESULTS.md', 'EVIDENCE_INDEX.md', 'DEMO_GUIDE.md', 'PROJECT_AUDIT.md', 'ENVIRONMENT_GUIDE.md', 'PROTOCOL.md', 'index.html']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def freeze():
    target = RUN / 'before_manifest.json'
    if target.exists():
        raise FileExistsError(target)
    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    names = [n for n in names if n and not n.startswith(('robot_can_v2/', RUN.relative_to(ROOT).as_posix() + '/'))]
    with ThreadPoolExecutor(max_workers=4) as pool:
        hashes = dict(zip(names, pool.map(lambda n: sha(ROOT / n), names)))
    before = RUN / 'before'
    before.mkdir(exist_ok=False)
    for name in DOCUMENTS:
        (before / name).write_bytes((ROOT / name).read_bytes())
    with target.open('x', encoding='utf-8') as f:
        json.dump({'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
                   'files': hashes, 'mutable_navigation_documents_backed_up': DOCUMENTS}, f, indent=2)
    print(json.dumps({'frozen_files': len(hashes), 'navigation_backups': len(DOCUMENTS)}))


def verify():
    d = json.loads((RUN / 'before_manifest.json').read_text())
    def check(item):
        name, digest = item
        path = RUN / 'before' / name if name in DOCUMENTS else ROOT / name
        return name if not path.is_file() or sha(path) != digest else None
    with ThreadPoolExecutor(max_workers=4) as pool:
        errors = [n for n in pool.map(check, d['files'].items()) if n]
    result = {'passed': not errors, 'verified_original_files': len(d['files']), 'errors': errors,
              'scope': 'Old source, weights, data, results and protocols unchanged; original navigation bytes retained in before/.'}
    (RUN / 'preservation_check.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    freeze() if '--freeze' in sys.argv else verify()
