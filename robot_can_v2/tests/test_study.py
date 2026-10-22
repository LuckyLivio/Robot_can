"""A prospective lock cannot be silently replaced by a new execution lock."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from robot_can_v2 import study, evaluate


class StudyTests(unittest.TestCase):
    def fixture(self, root):
        run = root / 'run'; run.mkdir()
        jobs = [{'id': 'fixture', 'method': 'bc', 'controller': 'none',
                 'dataset': 'run/confirmation_starts.hdf5', 'demos': ['demo_0']}]
        path = run / 'confirmation_jobs.json'
        path.write_text(json.dumps(jobs))
        return run, path

    def test_changed_confirmation_job_refused_before_stage_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); run, jobs = self.fixture(root)
            (run / 'confirmation_lock.json').write_text(json.dumps({'source_sha256': {
                'run/confirmation_jobs.json': 'not-the-registered-hash'}}))
            with mock.patch.object(study, 'ROOT', root), mock.patch.object(study, 'RUN', run):
                with self.assertRaisesRegex(ValueError, 'jobs drifted'):
                    study.run_batch(jobs, run / 'stage')
            self.assertFalse((run / 'stage').exists())

    def test_changed_confirmation_input_refused_before_stage_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); run, jobs = self.fixture(root)
            item = root / 'frozen.py'; item.write_text('original')
            hashes = {'run/confirmation_jobs.json': study.sha(jobs), 'frozen.py': study.sha(item)}
            (run / 'confirmation_lock.json').write_text(json.dumps({'source_sha256': hashes}))
            item.write_text('changed')
            with mock.patch.object(study, 'ROOT', root), mock.patch.object(study, 'RUN', run):
                with self.assertRaisesRegex(ValueError, 'drift after freeze'):
                    study.run_batch(jobs, run / 'stage')
            self.assertFalse((run / 'stage').exists())

    def test_policy_shape_rejected_before_dataset_or_simulator(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(evaluate.torch, 'load', return_value={
                'config': {'dim': 23, 'horizon': 2}}), mock.patch.object(evaluate.h5py, 'File') as dataset, \
                mock.patch.object(evaluate, 'make_env') as env:
            with self.assertRaisesRegex(ValueError, 'dim=23 and horizon=8'):
                evaluate.run({'id': 'x', 'method': 'bc', 'controller': 'none', 'demos': ['demo_0']}, Path(tmp) / 'new')
            dataset.assert_not_called(); env.assert_not_called()

    def test_wrong_full_seed_rejected_before_model_io(self):
        with mock.patch.object(evaluate.torch, 'load') as loader:
            with self.assertRaisesRegex(ValueError, 'registered frozen'):
                evaluate.run({'id': 'x', 'method': 'full', 'seed': 5, 'controller': 'none', 'demos': ['demo_0']}, 'unused')
            loader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
