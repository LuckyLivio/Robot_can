"""Tests for paired units, interaction contrasts, gates, and safe stage reads."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from robot_can_v2 import analyze


def make_stage(folder, kind="observation", with_observation_arm=False):
    folder.mkdir()
    starts = [f"demo_{index}" for index in range(60 if kind == "confirmation" else 30)]
    jobs = []
    if kind == "observation":
        for semantics in ("legacy", "canonical"):
            for noise in (0., .05):
                jobs.append({"id": f"{semantics}_{noise}", "method": "bc", "seed": None,
                             "controller": "none", "semantics": semantics, "noise": noise,
                             "demos": starts, "steps": 400, "delay": 2})
    else:
        for controller in ("none", "v5"):
            for method, seed in (("bc", None), ("full", 17), ("full", 23), ("full", 41)):
                for noise in (0., .05):
                    jobs.append({"id": f"{controller}_{method}_{seed}_{noise}", "method": method,
                                 "seed": seed, "controller": controller, "semantics": "legacy",
                                 "noise": noise, "demos": starts, "steps": 400, "delay": 2,
                                 "study_arm": "recovery_factorial"})
        if with_observation_arm:
            for semantics in ("legacy", "canonical"):
                for noise in (0., .05):
                    jobs.append({"id": f"observation_{semantics}_{noise}", "method": "bc", "seed": None,
                                 "controller": "none", "semantics": semantics, "noise": noise,
                                 "demos": starts, "steps": 400, "delay": 2,
                                 "study_arm": "observation_confirmation"})
    for job in jobs:
        out = folder / job["id"]
        out.mkdir()
        if kind == "confirmation" and job["study_arm"] == "recovery_factorial":
            success = True  # Historical BC succeeds; newly retrained legacy fails.
        else:
            success = job["semantics"] == "canonical"
        steps = 10 if success else 400
        records = [{"demo": demo, "success": success, "steps": steps,
                    "compute_timing": {"n": steps, "over_50ms": 1},
                    "stability": ({"observed": True, "stable": True, "diagnostic_controls": 20}
                                  if success else {"observed": False, "stable": False})}
                   for demo in starts]
        (out / "result.json").write_text(json.dumps({"status": "complete", "job": job, "records": records}))
    (folder / "jobs.json").write_text(json.dumps(jobs))
    count = len(jobs) * len(starts)
    (folder / "started.json").write_text(json.dumps({"registered_episodes": count,
        "jobs_sha256": hashlib.sha256((folder / "jobs.json").read_bytes()).hexdigest()}))
    (folder / "completed.json").write_text(json.dumps({"passed": True, "source_unchanged": True,
        "episodes": count, "jobs": [{"id": job["id"], "returncode": 0} for job in jobs]}))
    return starts, jobs


class AnalysisTests(unittest.TestCase):
    def test_paired_bootstrap_uses_start_effects(self):
        result = analyze.paired_bootstrap(np.array([-1., 1.]))
        self.assertEqual(result["independent_starts"], 2)
        self.assertEqual(result["bootstrap_draws"], 20000)
        self.assertEqual(result["bootstrap_seed"], 20261022)
        self.assertEqual(result["effect"], 0.)
        self.assertEqual((result["ci95_low"], result["ci95_high"]), (-1., 1.))
        self.assertEqual(result, analyze.paired_bootstrap([-1., 1.]))

    def test_fixed_world_seeds_and_noise_average_inside_start(self):
        rows = []
        for demo in ("A", "B"):
            for seed in (17, 23, 41):
                for noise in (0., .05):
                    rows.append({"demo": demo, "method": "full", "seed": seed, "noise": noise,
                                 "controller": "v5", "semantics": "legacy",
                                 "success": demo == "A" and seed != 41})
        values = analyze.grouped_cell_values(rows, ["A", "B"], method="full", controller="v5")
        np.testing.assert_allclose(values, [2 / 3, 0.])
        result = analyze.paired_bootstrap(values)
        self.assertEqual(result["independent_starts"], 2)  # Not12 executions.
        self.assertAlmostEqual(result["effect"], 1 / 3)

    def test_known_two_by_two_interaction(self):
        contrasts = analyze.factorial_effects([0., 1.], [1., 1.], [0., 0.], [1., 1.])
        np.testing.assert_array_equal(contrasts["bc_recovery"], [1., 0.])
        np.testing.assert_array_equal(contrasts["full_recovery"], [1., 1.])
        np.testing.assert_array_equal(contrasts["interaction"], [0., 1.])
        self.assertEqual(analyze.paired_bootstrap(contrasts["interaction"])["effect"], .5)

    def test_noop_gate_is_passed_but_not_positive_evidence(self):
        rows = []
        for method, seed in (("bc", None), ("full", 17)):
            for noise in (0., .05):
                for controller in ("v5", "queue_budget"):
                    for demo in ("A", "B"):
                        rows.append({"method": method, "seed": seed, "noise": noise,
                                     "controller": controller, "semantics": "legacy", "demo": demo,
                                     "success": demo == "A"})
        gate = analyze._budget_gate(rows, ["A", "B"])
        self.assertTrue(gate["passed"])
        self.assertFalse(gate["has_positive_evidence"])
        self.assertEqual(gate["paired_regression_executions"], 0)
        # Rescue the noisy cell while losing the clean cell: equal overall
        # counts do not excuse regression within a preregistered condition.
        for row in rows:
            if row["controller"] == "queue_budget" and row["method"] == "bc":
                if row["noise"] == 0. and row["demo"] == "A":
                    row["success"] = False
                if row["noise"] == .05 and row["demo"] == "B":
                    row["success"] = True
        gate = analyze._budget_gate(rows, ["A", "B"])
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["candidate_successes"], gate["parent_successes"])
        self.assertTrue(gate["has_positive_evidence"])

    def test_verified_stage_writes_only_three_new_artifacts_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_stage(root / "stage")
            output = root / "analysis"
            summary = analyze.analyze_stage(root / "stage", "observation", output)
            self.assertEqual({path.name for path in output.iterdir()},
                             {"summary.json", "cells.csv", "paired_effects.csv"})
            self.assertEqual(summary["independent_starts"], 30)
            self.assertEqual(summary["executions"], 120)
            self.assertEqual(summary["stability"]["all_evaluation_denominator"], 120)
            self.assertEqual(summary["stability"]["successful_evaluation_denominator"], 60)
            self.assertEqual(summary["stability"]["rate_successful_evaluations"], 1.)
            self.assertEqual(summary["compute_timing"]["n"], 60 * 400 + 60 * 10)
            self.assertEqual(summary["compute_timing"]["over_50ms"], 120)
            self.assertEqual(len(summary["paired_effects"]), 9)
            with self.assertRaises(FileExistsError):
                analyze.analyze_stage(root / "missing-stage", "observation", output)

    def test_incomplete_receipt_and_duplicate_demos_leave_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, jobs = make_stage(root / "stage")
            receipt_path = root / "stage/completed.json"
            original_receipt = receipt_path.read_text()
            receipt = json.loads(original_receipt)
            receipt["passed"] = False
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "passed completion"):
                analyze.analyze_stage(root / "stage", "observation", root / "analysis")
            self.assertFalse((root / "analysis").exists())
            receipt_path.write_text(original_receipt)
            result_path = root / "stage" / jobs[0]["id"] / "result.json"
            result = json.loads(result_path.read_text())
            result["records"][1]["demo"] = result["records"][0]["demo"]
            result_path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "demos do not exactly"):
                analyze.analyze_stage(root / "stage", "observation", root / "analysis")
            self.assertFalse((root / "analysis").exists())

    def test_confirmation_never_pools_retrained_legacy_with_historical_bc(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_stage(root / "stage", "confirmation", with_observation_arm=True)
            summary = analyze.analyze_stage(root / "stage", "confirmation", root / "analysis")
            self.assertEqual(summary["independent_starts"], 60)
            self.assertEqual(summary["executions"], 1200)
            self.assertEqual(summary["recovery_factorial"]["executions"], 960)
            self.assertEqual(summary["observation_confirmation"]["executions"], 240)
            self.assertEqual(summary["recovery_factorial"]["benchmark"]["successes"], 960)
            self.assertEqual(summary["observation_confirmation"]["benchmark"]["successes"], 120)
            self.assertTrue(summary["observation_confirmation"]["not_pooled_with_historical_bc"])
            semantic = [effect for effect in summary["observation_confirmation"]["paired_effects"]
                        if effect["metric"] == "success"]
            self.assertEqual(len(semantic), 3)
            self.assertTrue(all(effect["effect"] == 100. for effect in semantic))
            self.assertTrue(all(effect["independent_starts"] == 60 for effect in semantic))
            recovery = [effect for effect in summary["recovery_factorial"]["paired_effects"]
                        if effect["contrast"] in ("recovery_effect", "recovery_interaction")]
            self.assertTrue(all(effect["effect"] == 0. for effect in recovery))


if __name__ == "__main__":
    unittest.main()
