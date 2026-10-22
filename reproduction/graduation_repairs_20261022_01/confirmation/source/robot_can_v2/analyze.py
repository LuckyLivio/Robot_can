"""Paired, start-level analysis of completed versioned experiment stages.

No simulator, model selection, plotting, or asset mutation is performed.
The three new output files are written only after all stage checks pass.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


BOOTSTRAP_DRAWS = 20000
BOOTSTRAP_SEED = 20261022
NOISES = (0., .05)
KINDS = ("observation", "budget", "confirmation")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_bootstrap(effect_by_demo, *, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED,
                     indices=None) -> dict:
    """Bootstrap already-paired differences, with one entry per physical start.

    Any WM seeds and noise conditions must be averaged inside each start
    before calling this function.  An index draw resamples whole starts.
    """
    effects = np.asarray(effect_by_demo, dtype=float)
    if effects.ndim != 1 or not len(effects) or not np.isfinite(effects).all():
        raise ValueError("bootstrap requires a finite, nonempty vector of start-level effects")
    if draws < 1:
        raise ValueError("bootstrap draw count must be positive")
    if indices is None:
        indices = np.random.default_rng(seed).integers(0, len(effects), size=(draws, len(effects)))
    elif np.shape(indices) != (draws, len(effects)):
        raise ValueError("bootstrap indices must resample the same number of starts")
    estimates = effects[indices].mean(axis=1)
    low, high = np.percentile(estimates, (2.5, 97.5))
    return {"effect": float(effects.mean()), "ci95_low": float(low), "ci95_high": float(high),
            "independent_starts": len(effects), "bootstrap_draws": draws, "bootstrap_seed": seed,
            "ci_method": "paired-start percentile bootstrap"}


def factorial_effects(bc_none, bc_recovery, full_none, full_recovery) -> dict:
    """Return within-start recovery contrasts and their 2x2 interaction."""
    cells = [np.asarray(values, float) for values in (bc_none, bc_recovery, full_none, full_recovery)]
    if (not cells[0].ndim == 1 or not len(cells[0])
            or any(cell.shape != cells[0].shape or not np.isfinite(cell).all() for cell in cells)):
        raise ValueError("factorial cells must be finite equal-length start vectors")
    bc_delta, full_delta = cells[1] - cells[0], cells[3] - cells[2]
    return {"bc_recovery": bc_delta, "full_recovery": full_delta,
            "interaction": full_delta - bc_delta}


def grouped_cell_values(rows, starts, *, method, controller, semantics="legacy", noise=None,
                        metric="success", world_model_seeds=(17, 23, 41),
                        study_arm="recovery_factorial") -> np.ndarray:
    """Average fixed WM seeds and, optionally, both noises inside each start."""
    noises = NOISES if noise is None else (noise,)
    seeds = (None,) if method == "bc" else tuple(world_model_seeds)
    selected = {}
    for row in rows:
        if (row["method"] == method and row["controller"] == controller
                and row["semantics"] == semantics and row["noise"] in noises and row["seed"] in seeds
                and row.get("study_arm", "recovery_factorial") == study_arm):
            key = (row["demo"], row["seed"], row["noise"])
            if key in selected:
                raise ValueError("duplicate execution in grouped analysis")
            selected[key] = float(row[metric])
    values = []
    for demo in starts:
        keys = [(demo, seed, level) for seed in seeds for level in noises]
        if any(key not in selected for key in keys):
            raise ValueError(f"incomplete paired cell for {demo}")
        values.append(np.mean([selected[key] for key in keys]))
    return np.asarray(values, dtype=float)


def _condition(job) -> tuple:
    return (job["method"], job.get("seed"), job["controller"],
            job.get("semantics", "legacy"), float(job.get("noise", 0.)))


def _expected_conditions(kind, controllers) -> set:
    if kind == "observation":
        return {("bc", None, "none", semantics, noise)
                for semantics in ("legacy", "canonical") for noise in NOISES}
    if kind == "budget":
        families = (("bc", None), ("full", 17))
        controllers = ("v5", "queue_budget")
    else:
        families = (("bc", None), ("full", 17), ("full", 23), ("full", 41))
        if set(controllers) not in ({"none", "v5"}, {"none", "v5", "queue_budget"}):
            raise ValueError("confirmation requires none/v5 and only the optional queue_budget candidate")
    return {(method, seed, controller, "legacy", noise)
            for method, seed in families for controller in controllers for noise in NOISES}


def load_complete_stage(stage, kind) -> tuple[list, list, list, dict]:
    """Validate the completion receipt, exact registered matrix, and each demo."""
    if kind not in KINDS:
        raise ValueError(f"unknown analysis kind: {kind}")
    stage = Path(stage).resolve()
    jobs_path, receipt_path, started_path = [stage / name for name in
                                             ("jobs.json", "completed.json", "started.json")]
    jobs = json.loads(jobs_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    started = json.loads(started_path.read_text(encoding="utf-8"))
    if receipt.get("passed") is not True or receipt.get("source_unchanged") is not True:
        raise ValueError("stage must have a passed completion receipt with unchanged sources")
    if started.get("jobs_sha256") != _sha(jobs_path):
        raise ValueError("registered jobs hash differs from stage start receipt")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("stage jobs must be nonempty")
    ids = [job["id"] for job in jobs]
    if (len(set(ids)) != len(ids) or any(not isinstance(name, str) or Path(name).name != name
                                      or name in (".", "..") for name in ids)):
        raise ValueError("job IDs must be unique safe directory names")
    outcomes = receipt.get("jobs", [])
    if (len(outcomes) != len(ids) or {row.get("id") for row in outcomes} != set(ids)
            or any(row.get("returncode") != 0 for row in outcomes)):
        raise ValueError("completion receipt must contain one successful outcome per registered job")
    starts = jobs[0].get("demos", [])
    expected_n = 60 if kind == "confirmation" else 30
    if len(starts) != expected_n or len(set(starts)) != expected_n:
        raise ValueError(f"{kind} stage requires exactly {expected_n} distinct physical start IDs")
    conditions = [(job.get("study_arm", "recovery_factorial"), _condition(job)) for job in jobs]
    factorial_jobs = [job for job in jobs if job.get("study_arm", "recovery_factorial") == "recovery_factorial"]
    expected = {("recovery_factorial", condition) for condition in
                _expected_conditions(kind, {job["controller"] for job in factorial_jobs})}
    observation_jobs = [job for job in jobs if job.get("study_arm") == "observation_confirmation"]
    if kind == "confirmation" and observation_jobs:
        expected.update({("observation_confirmation", condition) for condition in
                         _expected_conditions("observation", {"none"})})
    if len(set(conditions)) != len(conditions) or set(conditions) != expected:
        raise ValueError("registered jobs do not match the complete expected factorial matrix")
    rows, cells = [], []
    source_hashes = {name: _sha(stage / name) for name in ("jobs.json", "completed.json", "started.json")}
    for job in jobs:
        if job.get("demos") != starts or job.get("steps", 400) != 400 or job.get("delay", 2) != 2:
            raise ValueError("jobs must use the same ordered starts, 400 controls, and D2")
        path = stage / job["id"] / "result.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        source_hashes[path.relative_to(stage).as_posix()] = _sha(path)
        if result.get("status") != "complete" or result.get("job") != job:
            raise ValueError(f"result must complete the exact registered job: {job['id']}")
        records = result.get("records", [])
        names = [record.get("demo") for record in records]
        if names != starts or len(set(names)) != len(names):
            raise ValueError(f"result demos do not exactly match registration: {job['id']}")
        for record in records:
            success, steps = record.get("success"), record.get("steps")
            if (type(success) is not bool or type(steps) is not int or not 1 <= steps <= 400
                    or (not success and steps != 400)):
                raise ValueError("each benchmark must end at first success or the 400-step failure cap")
            timing, stable = record.get("compute_timing", {}), record.get("stability", {})
            if (timing.get("n") != steps or type(timing.get("over_50ms")) is not int
                    or not 0 <= timing["over_50ms"] <= steps):
                raise ValueError("compute timing counts must use only the benchmark controls")
            if (type(stable.get("observed")) is not bool or type(stable.get("stable")) is not bool
                    or stable["observed"] != success or (stable["stable"] and not success)
                    or (success and stable.get("diagnostic_controls") != 20)):
                raise ValueError("settling must be 20 separate controls following benchmark success only")
            rows.append({"job_id": job["id"], "method": job["method"], "seed": job.get("seed"),
                         "study_arm": job.get("study_arm", "recovery_factorial"),
                         "controller": job["controller"], "semantics": job.get("semantics", "legacy"),
                         "noise": float(job.get("noise", 0.)), "demo": record["demo"],
                         "success": success, "steps": steps, "stable": stable["stable"],
                         "record": record})
        cells.append(_summarize_cell(job, records))
    if receipt.get("episodes") != len(rows) or started.get("registered_episodes") != len(rows):
        raise ValueError("stage episode totals do not match complete registered records")
    return starts, rows, cells, {"stage": str(stage), "stage_files_sha256": source_hashes,
                                "completed_passed": True, "source_unchanged": True}


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _summarize_cell(job, records):
    successes = sum(record["success"] for record in records)
    stable_count = sum(record["stability"]["stable"] for record in records)
    timing_n = sum(record["compute_timing"]["n"] for record in records)
    timing_over = sum(record["compute_timing"]["over_50ms"] for record in records)
    return {"job_id": job["id"], "method": job["method"], "world_model_seed": job.get("seed"),
            "study_arm": job.get("study_arm", "recovery_factorial"),
            "controller": job["controller"], "observation_semantics": job.get("semantics", "legacy"),
            "noise": float(job.get("noise", 0.)), "independent_starts": len(records),
            "executions": len(records), "successes": successes,
            "success_rate": _ratio(successes, len(records)),
            "mean_capped_steps": float(np.mean([record["steps"] for record in records])),
            "mean_steps_successful": (float(np.mean([record["steps"] for record in records if record["success"]]))
                                      if successes else None),
            "stable_diagnostic_observed": sum(record["stability"]["observed"] for record in records),
            "stable_successes": stable_count, "stable_rate_all_evaluations": _ratio(stable_count, len(records)),
            "stable_rate_successful_evaluations": _ratio(stable_count, successes),
            "compute_samples": timing_n, "compute_over_50ms": timing_over,
            "compute_over_50ms_fraction": _ratio(timing_over, timing_n)}


def _budget_gate(rows, starts):
    by_condition, rescue_starts, regression_starts = [], set(), set()
    total_parent = total_candidate = regressions = rescues = 0
    for method, seed in (("bc", None), ("full", 17)):
        for noise in NOISES:
            parent = grouped_cell_values(rows, starts, method=method, controller="v5", noise=noise,
                                         world_model_seeds=(17,))
            candidate = grouped_cell_values(rows, starts, method=method, controller="queue_budget", noise=noise,
                                            world_model_seeds=(17,))
            lost, rescued = (candidate < parent), (candidate > parent)
            parent_count, candidate_count = int(parent.sum()), int(candidate.sum())
            n_lost, n_rescued = int(lost.sum()), int(rescued.sum())
            total_parent += parent_count; total_candidate += candidate_count
            regressions += n_lost; rescues += n_rescued
            regression_starts.update(demo for demo, flag in zip(starts, lost) if flag)
            rescue_starts.update(demo for demo, flag in zip(starts, rescued) if flag)
            by_condition.append({"method": method, "world_model_seed": seed, "noise": noise,
                                 "parent_successes": parent_count, "candidate_successes": candidate_count,
                                 "paired_regressions": n_lost, "paired_rescues": n_rescued,
                                 "noninferior_success_count": candidate_count >= parent_count})
    conditions_noninferior = all(cell["noninferior_success_count"] for cell in by_condition)
    passed = total_candidate >= total_parent and conditions_noninferior and regressions <= 2
    return {"passed": passed, "criteria": {
                "total_successes_not_below_parent": total_candidate >= total_parent,
                "all_four_condition_success_counts_not_below_parent": conditions_noninferior,
                "paired_regression_executions_at_most_two": regressions <= 2},
            "parent_successes": total_parent, "candidate_successes": total_candidate,
            "paired_regression_executions": regressions, "paired_rescue_executions": rescues,
            "distinct_regression_starts": len(regression_starts), "distinct_rescue_starts": len(rescue_starts),
            "regression_start_ids": sorted(regression_starts), "rescue_start_ids": sorted(rescue_starts),
            "conditions": by_condition, "has_positive_evidence": bool(rescue_starts),
            "positive_evidence_scope": "At least one distinct rescued start is an additional anti-noop diagnostic; it is not part of the stated gate.",
            "scope": "Development gate over 30 reused starts and four paired conditions; not independent confirmation."}


def build_analysis(kind, starts, rows, cells, provenance) -> tuple[dict, list]:
    all_rows, all_cells = rows, cells
    if kind == "confirmation":
        # Newly retrained legacy BC and historical frozen BC are different
        # policies.  Select the arm before grouping any paired contrasts.
        rows = [row for row in all_rows if row.get("study_arm", "recovery_factorial") == "recovery_factorial"]
        cells = [cell for cell in all_cells if cell.get("study_arm", "recovery_factorial") == "recovery_factorial"]
    observation_rows = [row for row in all_rows if row.get("study_arm") == "observation_confirmation"]
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(starts),
                                                          size=(BOOTSTRAP_DRAWS, len(starts)))
    seeds = (17,) if kind == "budget" else (17, 23, 41)
    effects = []

    def values(method, controller, noise, metric, semantics="legacy"):
        return grouped_cell_values(rows, starts, method=method, controller=controller,
                                   noise=noise, metric=metric, semantics=semantics, world_model_seeds=seeds)

    def add(label, method, noise, metric, before, after, comparison, study_arm="recovery_factorial"):
        scale, unit = (1., "controls") if metric == "steps" else (100., "percentage_points")
        effect = paired_bootstrap((after - before) * scale, indices=indices)
        effects.append({"contrast": label, "comparison": comparison, "method": method,
                        "study_arm": study_arm,
                        "noise": "equal_noise_joint" if noise is None else noise,
                        "metric": metric, "unit": unit, "before_mean": float(before.mean() * scale),
                        "after_mean": float(after.mean() * scale), **effect,
                        "net_positive_starts": int((after > before).sum()),
                        "net_negative_starts": int((after < before).sum()),
                        "interpretation": "negative favors candidate" if metric == "steps" else "positive favors candidate"})

    for noise in (*NOISES, None):
        for metric in ("success", "stable", "steps"):
            if kind == "observation":
                add("canonical_minus_legacy", "bc", noise, metric,
                    values("bc", "none", noise, metric, "legacy"),
                    values("bc", "none", noise, metric, "canonical"), "canonical - legacy")
            elif kind == "budget":
                for method in ("bc", "full"):
                    add("queue_budget_minus_v5", method, noise, metric,
                        values(method, "v5", noise, metric), values(method, "queue_budget", noise, metric),
                        "queue_budget - v5")
            else:
                controllers = sorted({row["controller"] for row in rows})
                comparisons = [("none", "v5")]
                if "queue_budget" in controllers:
                    comparisons += [("v5", "queue_budget"), ("none", "queue_budget")]
                for parent, candidate in comparisons:
                    bc_parent, bc_candidate = [values("bc", controller, noise, metric)
                                               for controller in (parent, candidate)]
                    full_parent, full_candidate = [values("full", controller, noise, metric)
                                                   for controller in (parent, candidate)]
                    for method, before, after in (("bc", bc_parent, bc_candidate),
                                                  ("full", full_parent, full_candidate)):
                        add("recovery_effect", method, noise, metric, before, after, f"{candidate} - {parent}")
                    paired = factorial_effects(bc_parent, bc_candidate, full_parent, full_candidate)
                    add("recovery_interaction", "full_minus_bc", noise, metric,
                        paired["bc_recovery"], paired["full_recovery"],
                        f"(Full {candidate}-{parent}) - (BC {candidate}-{parent})")
                for controller in controllers:
                    add("world_model_effect", controller, noise, metric,
                        values("bc", controller, noise, metric), values("full", controller, noise, metric),
                        f"Full - BC with {controller}")
                if observation_rows:
                    legacy, canonical = [grouped_cell_values(
                        observation_rows, starts, method="bc", controller="none", semantics=semantics,
                        noise=noise, metric=metric, study_arm="observation_confirmation")
                        for semantics in ("legacy", "canonical")]
                    add("observation_confirmation_canonical_minus_legacy", "bc", noise, metric,
                        legacy, canonical, "matched newly trained canonical - matched newly trained legacy",
                        study_arm="observation_confirmation")

    successes = sum(row["success"] for row in rows)
    stable_count = sum(row["stable"] for row in rows)
    timing_n = sum(cell["compute_samples"] for cell in cells)
    timing_over = sum(cell["compute_over_50ms"] for cell in cells)
    summary = {"kind": kind, "status": "complete", "independent_starts": len(starts),
               "executions": len(all_rows), "registered_jobs": len(all_cells), "independent_start_ids": list(starts),
               "sample_scope": ("60 prospectively registered independent native starts; executions share starts across conditions."
                                if kind == "confirmation" else
                                "30 reused development demonstration starts; development comparison, not independent confirmation."),
               "bootstrap": {"unit": "physical start/demo", "draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED,
                             "confidence": .95, "method": "percentile", "same_resamples_for_all_contrasts": True,
                             "fixed_world_model_seeds": list(seeds) if kind != "observation" else [],
                             "world_model_seed_aggregation": "mean within each start before resampling",
                             "noise_aggregation": "joint contrasts weight noise0 and0.05 equally within each start",
                             "multiple_comparison_correction": "none; intervals are pointwise",
                             "uncertainty_scope": "Variation over starts conditional on the fixed trained policies and fixed WM seeds; no retraining uncertainty."},
               "benchmark": {"max_controls": 400, "delay": 2, "successes": successes,
                             "study_arm": "recovery_factorial",
                             "success_denominator_executions": len(rows),
                             "mean_capped_steps": float(np.mean([row["steps"] for row in rows]))},
               "stability": {"stable_successes": stable_count, "observed_after_benchmark_successes": successes,
                             "all_evaluation_denominator": len(rows), "successful_evaluation_denominator": successes,
                             "rate_all_evaluations": _ratio(stable_count, len(rows)),
                             "rate_successful_evaluations": _ratio(stable_count, successes),
                             "diagnostic_controls_per_success": 20,
                             "scope": "Separate post-termination diagnostic, excluded from benchmark steps, timing samples, and independent-start count."},
               "compute_timing": {"n": timing_n, "over_50ms": timing_over,
                                  "over_50ms_fraction": _ratio(timing_over, timing_n),
                                  "scope": "Summed benchmark control counts only; episode quantiles are not aggregated. Parallel evaluation load affects timings."},
               "cells": all_cells, "paired_effects": effects, "provenance": provenance}
    if kind == "budget":
        summary["development_gate"] = _budget_gate(rows, starts)
    if kind == "confirmation":
        summary["recovery_factorial"] = {"independent_starts": len(starts), "executions": len(rows),
            "benchmark": summary["benchmark"], "stability": summary["stability"],
            "compute_timing": summary["compute_timing"], "cells": cells,
            "paired_effects": [effect for effect in effects if effect["study_arm"] == "recovery_factorial"],
            "scope": "Historical frozen BC and fixed WM seeds only; newly retrained BC policies are excluded."}
        if observation_rows:
            observation_cells = [cell for cell in all_cells if cell["study_arm"] == "observation_confirmation"]
            n_success = sum(row["success"] for row in observation_rows)
            n_stable = sum(row["stable"] for row in observation_rows)
            n_compute = sum(cell["compute_samples"] for cell in observation_cells)
            n_over = sum(cell["compute_over_50ms"] for cell in observation_cells)
            summary["observation_confirmation"] = {
                "independent_starts": len(starts), "executions": len(observation_rows),
                "sample_scope": "Prospectively frozen matched newly trained legacy/canonical policies evaluated on the same 60 fresh starts; separate from the historical-BC factorial.",
                "benchmark": {"successes": n_success, "success_denominator_executions": len(observation_rows),
                              "max_controls": 400, "delay": 2,
                              "mean_capped_steps": float(np.mean([row["steps"] for row in observation_rows]))},
                "stability": {"stable_successes": n_stable, "all_evaluation_denominator": len(observation_rows),
                              "successful_evaluation_denominator": n_success,
                              "rate_all_evaluations": _ratio(n_stable, len(observation_rows)),
                              "rate_successful_evaluations": _ratio(n_stable, n_success),
                              "observed_after_benchmark_successes": n_success,
                              "diagnostic_controls_per_success": 20},
                "compute_timing": {"n": n_compute, "over_50ms": n_over,
                                   "over_50ms_fraction": _ratio(n_over, n_compute)},
                "cells": observation_cells,
                "paired_effects": [effect for effect in effects if effect["study_arm"] == "observation_confirmation"],
                "not_pooled_with_historical_bc": True}
    return summary, effects


def analyze_stage(stage, kind, output_dir) -> dict:
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing existing analysis directory: {output}")
    starts, rows, cells, provenance = load_complete_stage(stage, kind)
    summary, effects = build_analysis(kind, starts, rows, cells, provenance)
    # Invalid or incomplete stages leave no analysis output directory behind.
    output.mkdir(parents=True, exist_ok=False)
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for name, records in (("cells.csv", cells), ("paired_effects.csv", effects)):
        with (output / name).open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--kind", choices=KINDS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_stage(args.stage, args.kind, args.output_dir)
    print(json.dumps({"kind": args.kind, "independent_starts": result["independent_starts"],
                      "executions": result["executions"], "output_dir": str(args.output_dir)}))
