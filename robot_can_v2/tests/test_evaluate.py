"""Evaluation contracts and benchmark/diagnostic boundaries without MuJoCo."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from robot_can_v2 import evaluate


def valid_job(**changes):
    return {"id": "fixture", "method": "bc", "controller": "v5", "semantics": "legacy",
            "delay": 2, "steps": 400, "noise": 0., "demos": ["demo_0"], **changes}


class FakePolicy:
    def __init__(self):
        self.calls = 0

    def best(self, x):
        self.calls += 1
        chunk = np.repeat((np.arange(8, dtype=np.float32) / 10.)[:, None], 7, axis=1)
        return torch.from_numpy(chunk[None])


class FakeEnvironment:
    """Succeed at control 3, then stay stationary during 20 settling controls."""
    def __init__(self, success_at=3):
        controller = SimpleNamespace(input_type="delta", input_ref_frame="world",
                                     output_max=np.array([.05, .05, .05, .5, .5, .5]),
                                     output_min=np.array([-.05, -.05, -.05, -.5, -.5, -.5]))
        self.robots = [SimpleNamespace(part_controllers={"right": controller})]
        self.control_freq = 20
        self.object_to_id = {"can": 0}
        self.target_bin_placements = np.array([[.2, .2, .8]])
        self.bin2_pos = np.array([.1, .3, .8])
        self.controls, self.closed, self.success_at = [], False, success_at
        self.obs = {"robot0_eef_pos": np.array([0., -.2, 1.]),
                    "robot0_eef_quat": np.array([1., 0., 0., 0.]),
                    "robot0_gripper_qpos": np.array([.02, -.02]),
                    "object": np.array([0., 0., 0., 0., 0., 0., 1.,
                                        0., -.2, .86, 0., 0., 0., 1.])}

    def step(self, action):
        self.controls.append(action.copy())
        return self.obs, 0., False, {}

    def _check_success(self):
        return len(self.controls) >= self.success_at

    def close(self):
        self.closed = True


class EvaluationTests(unittest.TestCase):
    def test_wrong_budget_or_delay_rejected_before_model_or_simulator(self):
        bad_jobs = [valid_job(delay=0), valid_job(delay=3), valid_job(steps=500),
                    valid_job(config={"delay": 1}), valid_job(config={"horizon_steps": 500}),
                    valid_job(config={"max_attempts": 3}), valid_job(config={"stable_frames": 4})]
        with mock.patch.object(evaluate, "make_env") as simulator, \
                mock.patch.object(evaluate.torch, "load") as model_loader:
            for job in bad_jobs:
                with self.subTest(job=job), self.assertRaises(ValueError):
                    evaluate.run(job, "unused-contract-output")
            simulator.assert_not_called()
            model_loader.assert_not_called()

    def test_historical_world_model_cannot_take_canonical_observations(self):
        with self.assertRaisesRegex(ValueError, "Historical WM"):
            evaluate.validate_job(valid_job(method="full", semantics="canonical"))
        evaluate.validate_job(valid_job(method="full", seed=17, semantics="legacy"))

    def test_corrected_policy_semantics_mismatch_rejected_before_data_or_sim(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(evaluate.torch, "load", return_value={
                    "config": {"observation_semantics": "canonical"}}), \
                mock.patch.object(evaluate.h5py, "File") as dataset, \
                mock.patch.object(evaluate, "make_env") as simulator:
            with self.assertRaisesRegex(ValueError, "semantics mismatch"):
                evaluate.run(valid_job(), Path(tmp) / "new")
            dataset.assert_not_called()
            simulator.assert_not_called()
            self.assertFalse((Path(tmp) / "new").exists())

    def test_fifo_candidates_preserve_both_actual_pending_commands(self):
        pending = np.arange(14, dtype=np.float32).reshape(2, 7) / 20.
        commands = np.arange(28, dtype=np.float32).reshape(4, 7) / 30.
        result = evaluate.pending_sequences(pending, commands)
        self.assertEqual(result.shape, (4, 8, 7))
        np.testing.assert_array_equal(result[:, :2], np.broadcast_to(pending, (4, 2, 7)))
        for index, command in enumerate(commands):
            np.testing.assert_array_equal(result[index, 2:], np.broadcast_to(command, (6, 7)))
        with self.assertRaisesRegex(ValueError, "actual D2"):
            evaluate.pending_sequences(pending[:1], commands)

    def test_full_proposal_scores_actual_fifo_and_retains_registered_margin(self):
        pending = np.ones((2, 7), dtype=np.float32) * -.5
        base = np.repeat((np.arange(8, dtype=np.float32) / 10.)[:, None], 7, axis=1)
        model = mock.Mock(return_value=(None, torch.zeros((4, 8, 2))))
        with mock.patch.object(evaluate, "score", return_value=torch.tensor([.51, .1, .5, .2])):
            action, selected, _ = evaluate.propose(model, np.zeros(23, np.float32), base, pending)
        self.assertEqual(selected, 2)
        np.testing.assert_array_equal(action, base[2])
        actual_sequences = model.call_args.args[1].numpy()
        np.testing.assert_array_equal(actual_sequences[:, :2], np.broadcast_to(pending, (4, 2, 7)))
        with mock.patch.object(evaluate, "score", return_value=torch.tensor([.53, .1, .5, .2])):
            action, selected, _ = evaluate.propose(model, np.zeros(23, np.float32), base, pending)
        self.assertEqual(selected, 0)
        np.testing.assert_array_equal(action, base[0])

    def test_timing_threshold_and_denominator(self):
        result = evaluate.timing_summary([1., 50., 51., 100.])
        self.assertEqual(result["n"], 4)
        self.assertEqual(result["over_50ms"], 2)
        self.assertEqual(result["over_50ms_fraction"], .5)
        self.assertEqual(result["max_ms"], 100.)

    def test_original_v5_enabled_disabled_and_settling_denominators(self):
        original_controller = evaluate.RecoveryController
        created = []

        def controller_factory(*args):
            controller = original_controller(*args)
            created.append(controller)
            return controller

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "policy/bc_chunk_dev_v4/policy_15000.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"fixture checkpoint hash only")
            for controller_name, enabled in (("none", False), ("v5", True)):
                with self.subTest(controller=controller_name), ExitStack() as stack:
                    env, policy = FakeEnvironment(), FakePolicy()
                    stack.enter_context(mock.patch.object(evaluate, "ROOT", root))
                    stack.enter_context(mock.patch.object(evaluate, "DATA", root / "fixture.hdf5"))
                    stack.enter_context(mock.patch.object(evaluate.torch, "load", return_value={"config": {"dim": 23, "horizon": 8}}))
                    h5 = stack.enter_context(mock.patch.object(evaluate.h5py, "File"))
                    h5.return_value.__enter__.return_value = {"data": {"demo_0": object()}}
                    stack.enter_context(mock.patch.object(evaluate, "load_policy", return_value=policy))
                    world = stack.enter_context(mock.patch.object(evaluate, "load_world"))
                    stack.enter_context(mock.patch.object(evaluate, "make_env", return_value=env))
                    stack.enter_context(mock.patch.object(evaluate, "restore_demo", return_value=env.obs))
                    stack.enter_context(mock.patch.object(evaluate, "RecoveryController", side_effect=controller_factory))
                    stack.enter_context(mock.patch("builtins.print"))
                    output = root / controller_name
                    result = evaluate.run(valid_job(controller=controller_name), output)
                    record = result["records"][0]
                    self.assertEqual(result["status"], "complete")
                    self.assertEqual(created[-1].enabled, enabled)
                    self.assertEqual(record["steps"], 3)
                    self.assertEqual(record["compute_timing"]["n"], 3)
                    self.assertEqual(policy.calls, 3)
                    self.assertEqual(len(env.controls), 23)
                    self.assertTrue(env.closed)
                    self.assertTrue(record["success"])
                    self.assertEqual(record["stability"]["diagnostic_controls"], 20)
                    self.assertTrue(record["stability"]["stable"])
                    trace = [json.loads(row) for row in (output / "trace.jsonl").read_text().splitlines()]
                    settling = [json.loads(row) for row in (output / "settling_trace.jsonl").read_text().splitlines()]
                    self.assertEqual(len(trace), 3)
                    self.assertEqual(len(settling), 20)
                    np.testing.assert_array_equal(env.controls[0], np.zeros(7))
                    np.testing.assert_array_equal(env.controls[1], np.zeros(7))
                    np.testing.assert_allclose(env.controls[2], .2)
                    # Two in-flight benchmark commands survive into diagnostics;
                    # the first diagnostic open-gripper command executes third.
                    np.testing.assert_allclose(env.controls[3:5], .2)
                    np.testing.assert_array_equal(env.controls[5], [0., 0., 0., 0., 0., 0., -1.])
                    world.assert_not_called()
                    self.assertEqual(len(created), 1 if controller_name == "none" else 2)


if __name__ == "__main__":
    unittest.main()
