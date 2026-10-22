"""Mechanism/regression checks; no simulator rollouts or success claims."""
import json
from pathlib import Path
import unittest

import numpy as np

from robot_can_v2.budget import QueueBudgetConfig, QueueBudgetController, V5, V5Config


def state():
    x = np.zeros(23, dtype=np.float32)
    x[:3] = [.1, -.1, 1.1]
    x[3:7] = [1., 0., 0., 0.]
    x[7:9] = [.04, -.04]
    x[16:19] = [.1, -.25, .86]
    x[19:23] = [0., 0., 0., 1.]
    return x


def make(config=None, enabled=True):
    return QueueBudgetController([.1975, .4025, .8], .8, state(), enabled, config)


class BudgetTests(unittest.TestCase):
    def test_queue_toward_and_away_have_opposite_effect(self):
        controller = make()
        x = state()
        d = controller.budget_diagnostics(x)
        direction = np.asarray(d['next_waypoint_m']) - x[:3]
        direction /= np.linalg.norm(direction)
        toward = np.zeros((2, 7)); toward[:, :3] = direction
        away = -toward
        controller.set_pending_actions(toward)
        good = controller.budget_diagnostics(x)
        controller.set_pending_actions(away)
        bad = controller.budget_diagnostics(x)
        self.assertLess(good['queue_motion_steps'], 0)
        self.assertGreater(bad['queue_motion_steps'], 0)
        self.assertLess(good['estimated_steps'], bad['estimated_steps'])

    def test_queue_is_copied_and_bounded(self):
        controller = make()
        queue = np.full((2, 7), 1e100)
        controller.set_pending_actions(queue)
        queue[:] = -1
        returned = controller.pending_actions
        returned[:] = -1
        np.testing.assert_array_equal(controller.pending_actions, np.ones((2, 7)))
        self.assertLessEqual(np.linalg.norm(controller.budget_diagnostics(state())['queue_projection_m']), .04 + 1e-9)

    def test_queue_shape_and_finiteness_are_strict(self):
        controller = make()
        for invalid in [np.zeros(7), np.zeros((3, 7)), np.full((2, 7), np.nan), np.full((2, 7), np.inf)]:
            with self.assertRaises(ValueError):
                controller.set_pending_actions(invalid)

    def test_observed_progress_is_clamped_and_directional(self):
        controller = make()
        x = state()
        goal = np.asarray(controller.budget_diagnostics(x)['next_waypoint_m'])
        direction = (goal-x[:3])/np.linalg.norm(goal-x[:3])
        for i in range(9):
            controller._eef_history.append(('NORMAL', x[:3] + (i-8)*.01*direction))
        advancing = controller.budget_diagnostics(x)
        controller._eef_history.clear()
        for i in range(9):
            controller._eef_history.append(('NORMAL', x[:3] - (i-8)*.01*direction))
        retreating = controller.budget_diagnostics(x)
        self.assertGreater(advancing['speed_m_per_step'], retreating['speed_m_per_step'])
        controller._eef_history.clear()
        for i in range(9):
            controller._eef_history.append(('NORMAL', np.array([1e100, 0., 0.]) * (i % 2)))
        estimate = controller.budget_diagnostics(x)
        self.assertTrue(np.isfinite(estimate['speed_m_per_step']))
        self.assertGreaterEqual(estimate['speed_m_per_step'], .006)
        self.assertLessEqual(estimate['speed_m_per_step'], .020)
        self.assertTrue(0 <= estimate['estimated_steps'] <= 400)

    def test_dwell_retains_close_lift_release_latency(self):
        controller = make()
        x = state()
        controller.mode = 'CLOSE'
        diagnostic = controller.budget_diagnostics(x)
        self.assertGreaterEqual(diagnostic['mandatory_phase_dwell_steps'], 18 + 7 + 16)
        controller.age = 10
        self.assertEqual(controller.budget_diagnostics(x)['mandatory_phase_dwell_steps'], diagnostic['mandatory_phase_dwell_steps'] - 10)

    def test_budget_does_not_change_transit_action(self):
        controller = make()
        old = V5([.1975, .4025, .8], .8, state())
        for mode in ['RETRACT', 'ALIGN', 'LIFT', 'CARRY']:
            controller.mode = old.mode = mode
            controller.clock = old.clock = 180
            controller.stable = old.stable = mode in ('LIFT', 'CARRY')
            for dx in [.01, .3]:
                goal = state()[:3] + [dx, 0., 0.]
                np.testing.assert_array_equal(controller.pose_action(state(), goal, 1), old.pose_action(state(), goal, 1))

    def test_contract_fixed(self):
        for change in [dict(delay=0), dict(horizon_steps=800), dict(max_attempts=3)]:
            with self.assertRaises(ValueError):
                make(QueueBudgetConfig(**change))
        controller = make(V5Config())
        self.assertEqual((controller.cfg.delay, controller.cfg.horizon_steps, controller.cfg.max_attempts), (2, 400, 2))

    def test_timing_changes_only_eligible_stall_trigger(self):
        new = make()
        old = V5([.1975, .4025, .8], .8, state())
        for controller in [new, old]:
            controller.clock = 70
            controller.progress_step = 70
        proposal = np.zeros(7, dtype=np.float32)
        for _ in range(25):
            new.set_pending_actions(np.zeros((2, 7)))
            new.choose(state(), proposal, proposal)
            old.choose(state(), proposal, proposal)
        self.assertEqual(old.mode, 'NORMAL')
        self.assertTrue(new.events)
        self.assertEqual(new.events[0]['reason'], 'early_stall_preserve_budget')
        self.assertTrue(new.events[0]['queue_budget']['pressure'])
        # Original airborne/outside eligibility remains the authority even if
        # an estimate on its own would suggest that little time remains.
        outside = state(); outside[16:19] = [.5, .6, 1.2]
        new.clock = 390
        self.assertFalse(new.time_pressure(outside))

    def test_disabled_exact_synthetic_trace(self):
        for supervisor_enabled, queue_enabled in [(True, False), (False, True)]:
            config = QueueBudgetConfig(queue_budget=queue_enabled)
            new = make(config, enabled=supervisor_enabled)
            old = V5([.1975, .4025, .8], .8, state(), enabled=supervisor_enabled)
            previous = np.zeros(7, dtype=np.float32)
            for step in range(240):
                x = state()
                x[:3] += np.array([.001 * np.sin(step), 0., 0.])
                proposal = np.array([0., -.1, 0., 0., 0., 0., -1.], dtype=np.float32)
                new.set_pending_actions(np.zeros((2, 7)))
                expected, expected_debug = old.choose(x, proposal, previous)
                actual, actual_debug = new.choose(x, proposal, previous)
                np.testing.assert_array_equal(expected, actual)
                self.assertEqual(expected_debug, actual_debug)
                self.assertEqual(old.events, new.events)
                previous = actual

    def test_disabled_exact_saved_inputs(self):
        root = Path(__file__).resolve().parents[2]
        trace = root / 'reproduction/recovery_v5_20261012_01/pilot_b/side_only_known_budget/trace.jsonl'
        if not trace.is_file():
            self.skipTest('Historical input trace unavailable')
        current = None
        checked = 0
        with trace.open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                x = np.asarray(row['state_before'], dtype=np.float32)
                if row['demo'] != current:
                    current = row['demo']
                    old = V5([.1975, .4025, .8], .8, x)
                    new = QueueBudgetController([.1975, .4025, .8], .8, x, config=QueueBudgetConfig(queue_budget=False))
                    previous = np.zeros(7, dtype=np.float32)
                proposal = np.asarray(row['original_proposal'], dtype=np.float32)
                expected, expected_debug = old.choose(x, proposal, previous)
                actual, actual_debug = new.choose(x, proposal, previous)
                np.testing.assert_array_equal(expected, actual)
                self.assertEqual(expected_debug, actual_debug)
                self.assertEqual(old.events, new.events)
                previous = np.asarray(row['executed_action'], dtype=np.float32)
                checked += 1
        self.assertGreater(checked, 0)


if __name__ == '__main__':
    unittest.main()
