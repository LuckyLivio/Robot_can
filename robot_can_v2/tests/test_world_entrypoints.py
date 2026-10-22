"""Contract checks that do not require MuJoCo or historical training reruns."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from robot_can_v2 import evaluate_delay, train_world
from robot_can_v2.metrics import ranking_metrics
from robot_can_v2.observations import canonicalize_state
from world_model import DirectScore, WorldModel


def small_data(groups=4, horizon=3):
    generator = np.random.default_rng(37)
    x = generator.normal(0, .1, (groups, 23)).astype(np.float32)
    a = generator.uniform(-.3, .3, (groups, 2, horizon, 7)).astype(np.float32)
    y = np.repeat(np.repeat(x[:, None, None, :], 2, 1), horizon, 2)
    labels = generator.uniform(0, 1, (groups, 2, horizon, 2)).astype(np.float32)
    return dict(x=x, a=a, y=y, labels=labels)


class RankingMetricsTests(unittest.TestCase):
    def test_prediction_tie_is_half_error(self):
        result = ranking_metrics([0, 0], [0, 1], 1, 2)
        self.assertEqual(result['mean_regret'], 1.)
        self.assertEqual(result['legacy_wrong_pair_rate'], 0.)
        self.assertEqual(result['strict_reversal_rate'], 0.)
        self.assertEqual(result['tie_rate'], 1.)
        self.assertEqual(result['pair_error_half_ties'], .5)
        self.assertEqual(result['effective_pairs'], 1)
        self.assertEqual(result['decided_pairs'], 0)

    def test_strict_ordering_and_noninformative_truth(self):
        reversed_result = ranking_metrics([1, 0], [0, 1], 1, 2)
        self.assertEqual(reversed_result['pair_error_half_ties'], 1.)
        self.assertEqual(reversed_result['strict_reversal_rate'], 1.)
        correct = ranking_metrics([0, 1], [0, 1], 1, 2)
        self.assertEqual(correct['mean_regret'], 0.)
        self.assertEqual(correct['pair_error_half_ties'], 0.)
        uninformative = ranking_metrics([1, 0], [1, 1.01], 1, 2)
        self.assertIsNone(uninformative['tie_rate'])
        self.assertEqual(uninformative['effective_pairs'], 0)

    def test_nan_is_rejected(self):
        with self.assertRaises(ValueError):
            ranking_metrics([np.nan, 0], [0, 1], 1, 2)


class PredictionOnlyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)
        self.model = WorldModel(23, 8, True)
        self.batch = {key: torch.from_numpy(value) for key, value in small_data().items()}

    def test_native_recurrence_matches_without_calling_head(self):
        x, actions = self.batch['x'], self.batch['a'][:, 0]
        expected, _ = self.model(x, actions)
        with mock.patch.object(self.model.labels, 'forward', side_effect=AssertionError('untrained head called')):
            actual, logits = train_world.predict(self.model, x, actions, 'prediction_only')
        torch.testing.assert_close(actual, expected)
        self.assertIsNone(logits)

    def test_dev_evaluation_does_not_score_random_head(self):
        with mock.patch.object(self.model.labels, 'forward', side_effect=AssertionError('head called')), \
             mock.patch.object(train_world, 'score', side_effect=AssertionError('scorer called')):
            result = train_world.evaluate(self.model, self.batch, method='prediction_only')
        self.assertIsNone(result['mean_regret'])
        self.assertGreaterEqual(result['normalized_state_mse'], 0.)

    def test_deployment_uses_physical_cost_without_scores(self):
        x, actions = self.batch['x'], self.batch['a'][:, 0]
        with mock.patch.object(self.model.labels, 'forward', side_effect=AssertionError('head called')), \
             mock.patch.object(evaluate_delay, 'score', side_effect=AssertionError('scorer called')):
            future, logits, learned, costs = evaluate_delay.score_candidates(
                self.model, x, actions, {'method': 'prediction_only'}, [0, 0, .85], False, 2, readout_weight=0.)
        expected = evaluate_delay.physical_cost(future[:, 2:], [0, 0, .85], False)
        torch.testing.assert_close(costs, expected)
        self.assertIsNone(logits)
        self.assertIsNone(learned)

    def test_prediction_only_rejects_score_only(self):
        with self.assertRaisesRegex(ValueError, 'physical cost'):
            evaluate_delay.score_candidates(self.model, self.batch['x'], self.batch['a'][:, 0],
                {'method': 'prediction_only'}, [0, 0, .85], False, 2, score_only=True, readout_weight=0.)

    def test_selection_criterion_uses_state_mse(self):
        metrics = {'mean_regret': .01, 'normalized_state_mse': .7}
        self.assertEqual(train_world.selection_criterion('prediction_only', metrics), .7)
        self.assertEqual(train_world.selection_criterion('full', metrics), .01)

    def test_training_freezes_head_and_selects_best_state_mse(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            train_path, dev_path = directory / 'train.npz', directory / 'dev.npz'
            np.savez(train_path, **small_data())
            np.savez(dev_path, **small_data())
            out = directory / 'run'
            args = train_world.build_parser().parse_args(['--method', 'prediction_only', '--output-dir', str(out),
                '--train-data', str(train_path), '--dev-data', str(dev_path), '--updates', '2', '--eval-every', '1',
                '--batch', '2', '--threads', '1', '--device', 'cpu'])
            before = {key: value.clone() for key, value in self.model.labels.state_dict().items()}
            dev_results = [{'normalized_state_mse': .7, 'mean_regret': .01},
                           {'normalized_state_mse': .3, 'mean_regret': .9}]
            with mock.patch.object(train_world, 'construct_model', return_value=self.model), \
                 mock.patch.object(train_world, 'evaluate', side_effect=dev_results), \
                 mock.patch.object(self.model.labels, 'forward', side_effect=AssertionError('head called')), \
                 mock.patch.object(train_world, 'score', side_effect=AssertionError('scorer called')), \
                 contextlib.redirect_stdout(io.StringIO()):
                train_world.main(args)
            checkpoint = torch.load(out / 'model.pt', map_location='cpu', weights_only=True)
            self.assertEqual(checkpoint['selected_step'], 2)
            self.assertIn('state MSE', checkpoint['config']['selection'])
            self.assertEqual(checkpoint['config']['score_head_usage'], 'frozen_and_never_called')
            for key, value in before.items():
                torch.testing.assert_close(checkpoint['state_dict']['labels.' + key], value, rtol=0, atol=0)
            self.assertTrue(all(not param.requires_grad for param in self.model.labels.parameters()))


class InputAndOutputTests(unittest.TestCase):
    def test_training_refuses_existing_output_before_loading_data_or_model(self):
        with tempfile.TemporaryDirectory() as directory:
            args = train_world.build_parser().parse_args(['--output-dir', directory, '--dataset', 'does_not_exist'])
            with mock.patch.object(train_world, 'load_data', side_effect=AssertionError('data loaded')), \
                 mock.patch.object(train_world.torch, 'set_num_threads', side_effect=AssertionError('compute configured')):
                with self.assertRaises(FileExistsError):
                    train_world.main(args)

    def test_evaluation_refuses_existing_output_before_loading_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            args = evaluate_delay.build_parser().parse_args(['--output-dir', directory, '--model', 'does_not_exist'])
            with mock.patch.object(evaluate_delay.torch, 'load', side_effect=AssertionError('checkpoint loaded')), \
                 mock.patch.object(evaluate_delay, 'make_env', side_effect=AssertionError('env created')):
                with self.assertRaises(FileExistsError):
                    evaluate_delay.run(args)

    def test_metadata_mismatch_is_rejected_and_legacy_default_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            old = Path(directory) / 'old.npz'
            canonical = Path(directory) / 'canonical.npz'
            np.savez(old, **small_data())
            cleaned = small_data()
            for key in ('x', 'y'):
                cleaned[key] = canonicalize_state(cleaned[key])
            np.savez(canonical, **cleaned, observation_semantics=np.asarray('canonical'))
            _, meta = train_world.load_data(old, 'cpu', 'legacy')
            self.assertEqual(meta['semantics_declaration'], 'historical_missing_metadata')
            with self.assertRaisesRegex(ValueError, 'data semantics'):
                train_world.load_data(old, 'cpu', 'canonical')
            with self.assertRaisesRegex(ValueError, 'data semantics'):
                train_world.load_data(canonical, 'cpu', 'legacy')
            _, meta = train_world.load_data(canonical, 'cpu', 'canonical')
            self.assertEqual(meta['observation_semantics'], 'canonical')

    def test_false_canonical_declaration_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'false_canonical.npz'
            np.savez(path, **small_data(), observation_semantics=np.asarray('canonical'))
            with self.assertRaisesRegex(ValueError, 'same-frame geometry'):
                train_world.load_data(path, 'cpu', 'canonical')

    def test_checkpoint_native_config_validates_horizon_and_semantics(self):
        cfg = {'method': 'direct', 'dim': 23, 'width': 8, 'horizon': 3, 'observation_semantics': 'legacy'}
        self.assertIsInstance(train_world.construct_model(cfg), DirectScore)
        with self.assertRaisesRegex(ValueError, 'horizon'):
            train_world.validate_model_config(cfg, horizon=8)
        with self.assertRaisesRegex(ValueError, 'semantics'):
            train_world.validate_model_config(cfg, observation_semantics='canonical')
        with self.assertRaisesRegex(ValueError, 'width'):
            train_world.construct_model({**cfg, 'width': True})

    def test_plain_full_direct_real_forward_evaluation(self):
        data = {key: torch.from_numpy(value) for key, value in small_data().items()}
        for method in ('plain', 'full', 'direct'):
            with self.subTest(method=method):
                model = train_world.construct_model({'method': method, 'dim': 23, 'width': 8, 'horizon': 3})
                result = train_world.evaluate(model, data, method=method)
                self.assertIsInstance(result['mean_regret'], float)
                self.assertIn('tie_rate', result)
                self.assertEqual(result['normalized_state_mse'] is None, method == 'direct')

    def test_cli_help_for_both_entrypoints(self):
        for module in ('robot_can_v2.train_world', 'robot_can_v2.evaluate_delay'):
            result = subprocess.run([sys.executable, '-B', '-m', module, '--help'], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--output-dir', result.stdout)
            self.assertIn('--observation-semantics', result.stdout)


class QueuePrefixTests(unittest.TestCase):
    def test_exact_fifo_prefix_and_committed_first_action(self):
        chunks = np.zeros((2, 4, 7), np.float32)
        chunks[0, 0], chunks[1, 0] = .2, -.4
        chunks[:, 1:] = .9  # Later chunk commands are never committed.
        pending = np.asarray([[.1] * 7, [-.1] * 7], np.float32)
        seqs = evaluate_delay.build_delayed_sequences(chunks, pending, 5, 2)
        np.testing.assert_array_equal(seqs[:, :2], np.repeat(pending[None], 2, 0))
        np.testing.assert_array_equal(seqs[:, 2:], np.repeat(chunks[:, :1], 3, 1))
        queue = list(pending.copy())
        actual = []
        for _ in range(5):
            queue.append(chunks[1, 0].copy())
            actual.append(queue.pop(0))
        np.testing.assert_array_equal(seqs[1], actual)

    def test_zero_delay_executes_new_action_immediately(self):
        chunks = np.ones((2, 3, 7), np.float32)
        seqs = evaluate_delay.build_delayed_sequences(chunks, [], 3, 0)
        np.testing.assert_array_equal(seqs, chunks)

    def test_wrong_queue_length_or_invisible_candidate_is_rejected(self):
        chunks = np.ones((2, 3, 7), np.float32)
        with self.assertRaisesRegex(ValueError, 'FIFO length'):
            evaluate_delay.build_delayed_sequences(chunks, np.zeros((1, 7)), 3, 2)
        with self.assertRaisesRegex(ValueError, 'exceed delay'):
            evaluate_delay.build_delayed_sequences(chunks, np.zeros((2, 7)), 2, 2)


if __name__ == '__main__':
    unittest.main()
