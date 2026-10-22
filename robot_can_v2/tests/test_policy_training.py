"""Small in-memory training checks; never open the real demonstration file."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from robot_can_v2 import train_policy


class TrainingTests(unittest.TestCase):
    def test_default_budget_matches_frozen_architecture(self):
        args = train_policy.build_parser().parse_args(
            ["--semantics", "canonical", "--output-dir", "unused"])
        self.assertEqual((args.width, args.horizon, args.mixtures), (512, 8, 5))
        self.assertEqual((args.updates, args.seed, args.batch, args.lr), (15000, 17, 256, 3e-4))

    def test_existing_output_is_rejected_before_data_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "old-run"
            output.mkdir()
            marker = output / "best.pt"
            marker.write_bytes(b"original historical model")
            args = train_policy.build_parser().parse_args(
                ["--semantics", "canonical", "--output-dir", str(output), "--updates", "1"])
            with mock.patch.object(train_policy, "read_episodes") as reader:
                with self.assertRaises(FileExistsError):
                    train_policy.main(args)
                reader.assert_not_called()
            self.assertEqual(marker.read_bytes(), b"original historical model")

    def test_batch_stream_does_not_depend_on_global_model_rng(self):
        first = train_policy.make_batch_generator(17)
        second = train_policy.make_batch_generator(17)
        for _ in range(3):
            torch.randn(130)  # unrelated model/global random draws
            np.testing.assert_array_equal(torch.randint(1000, (31,), generator=first).numpy(),
                                          torch.randint(1000, (31,), generator=second).numpy())

    def test_tiny_paired_runs_share_initialization_and_minibatches(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(12, 23)).astype(np.float32)
        actions = np.tanh(rng.normal(size=(12, 7))).astype(np.float32)
        calls = []

        def fixture_reader(split, semantics):
            calls.append(split)
            self.assertIn(split, ("train", "dev"))
            shifted = x.copy()
            if semantics == "canonical":
                shifted[:, 9:16] = np.sin(shifted[:, 9:16])
            return [{"name": f"{split}_demo", "x": shifted, "a": actions.copy()}]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            dataset = root / "data/fixture.hdf5"
            dataset.write_bytes(b"test-only dataset fingerprint")
            (root / "data/splits.json").write_text(json.dumps(
                {"train": ["train_demo"], "dev": ["dev_demo"], "sealed_test": ["never_read"]}))
            summaries, configurations = [], []
            with mock.patch.object(train_policy.common, "ROOT", root), \
                    mock.patch.object(train_policy.common, "DATA", dataset), \
                    mock.patch.object(train_policy, "read_episodes", side_effect=fixture_reader), \
                    mock.patch("builtins.print"):
                for semantics in ("legacy", "canonical"):
                    output = root / semantics
                    args = train_policy.build_parser().parse_args(
                        ["--semantics", semantics, "--output-dir", str(output), "--updates", "3",
                         "--width", "8", "--horizon", "2", "--mixtures", "2", "--batch", "4",
                         "--eval-every", "1", "--device", "cpu", "--threads", "1"])
                    summaries.append(train_policy.main(args))
                    config = json.loads((output / "config.json").read_text())
                    configurations.append(config)
                    rows = [json.loads(row) for row in (output / "log.jsonl").read_text().splitlines()]
                    self.assertEqual(len(rows), 3)
                    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
                    best_row = min(rows, key=lambda row: row["dev_nll"])
                    self.assertEqual(checkpoint["selected_update"], best_row["update"])
                    self.assertEqual(checkpoint["config"]["observation_semantics"], semantics)
                    self.assertEqual(set(config["data_split"]), {"train", "dev"})
                    self.assertTrue((output / "final.pt").exists())
                    self.assertEqual(len(list(output.glob("*.pt"))), 2)
                    for name, expected_hash in summaries[-1]["artifacts_sha256"].items():
                        self.assertEqual(train_policy.file_sha256(output / name), expected_hash)
            self.assertEqual(calls, ["train", "dev", "train", "dev"])
            self.assertEqual(summaries[0]["batch_sha256"], summaries[1]["batch_sha256"])
            self.assertEqual(configurations[0]["initial_parameter_sha256"],
                             configurations[1]["initial_parameter_sha256"])


if __name__ == "__main__":
    unittest.main()
