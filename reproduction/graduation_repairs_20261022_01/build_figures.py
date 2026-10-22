"""Plot completed, verified analyses without pooling unrelated policy weights.

Run only after the confirmation analysis exists. PNGs use 300 dpi; SVG/PDF
exports retain vector elements. Every plotted number is exported as CSV.
Existing figure directories are refused; --validate-only never writes files.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np


RUN = Path(__file__).resolve().parent
NOISES = (0., .05)
NOISE_KEYS = ("0.0", "0.05", "equal_noise_joint")
NOISE_LABELS = ("Clean", "Noise 0.05", "Equal-noise mean")
NOISE_COLORS = ("#365F83", "#D7813D", "#478575")
FIXED_EXPORT_DATE = datetime(2026, 10, 22, tzinfo=timezone.utc)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def csv_matches_json(records, expected):
    """The analyzer's CSVs and JSON must describe exactly the same rows."""
    normalized = [{key: "" if value is None else str(value) for key, value in row.items()}
                  for row in expected]
    return records == normalized


def load_analysis(folder, kind):
    folder = Path(folder).resolve()
    inputs = {folder / name: sha(folder / name) for name in
              ("summary.json", "cells.csv", "paired_effects.csv")}
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    n = 60 if kind == "confirmation" else 30
    if summary.get("status") != "complete" or summary.get("kind") != kind:
        raise ValueError(f"{folder}: require completed {kind} analysis")
    starts = summary.get("independent_start_ids", [])
    if summary.get("independent_starts") != n or len(starts) != n or len(set(starts)) != n:
        raise ValueError(f"{folder}: expected exactly {n} distinct physical starts")
    bootstrap = summary.get("bootstrap", {})
    if (bootstrap.get("unit") != "physical start/demo" or bootstrap.get("draws") != 20000
            or bootstrap.get("confidence") != .95):
        raise ValueError("Figures require the registered paired-start 20,000-draw 95% analysis")
    provenance = summary.get("provenance", {})
    if provenance.get("completed_passed") is not True or provenance.get("source_unchanged") is not True:
        raise ValueError(f"{folder}: analysis must retain verified stage provenance")
    stage = Path(provenance["stage"])
    for relative, expected in provenance["stage_files_sha256"].items():
        path = stage / relative
        if sha(path) != expected:
            raise ValueError(f"Analysis source changed: {path}")
        inputs[path.resolve()] = expected
    if not csv_matches_json(read_csv(folder / "cells.csv"), summary["cells"]):
        raise ValueError(f"{folder}: cells.csv differs from summary.json")
    if not csv_matches_json(read_csv(folder / "paired_effects.csv"), summary["paired_effects"]):
        raise ValueError(f"{folder}: paired_effects.csv differs from summary.json")
    for cell in summary["cells"]:
        if (cell["independent_starts"] != n or cell["executions"] != n
                or not 0 <= cell["successes"] <= n
                or not math.isclose(cell["success_rate"], cell["successes"] / n, abs_tol=1e-12)):
            raise ValueError("Cell count/rate mismatch")
    if sum(cell["executions"] for cell in summary["cells"]) != summary["executions"]:
        raise ValueError("Summary execution total differs from its cells")
    for effect in summary["paired_effects"]:
        if (effect["independent_starts"] != n or effect["bootstrap_draws"] != 20000
                or effect["ci_method"] != "paired-start percentile bootstrap"
                or not np.isfinite([effect[key] for key in ("effect", "ci95_low", "ci95_high")]).all()
                or effect["ci95_low"] > effect["ci95_high"]):
            raise ValueError("Invalid paired effect or interval")
    return {"folder": folder, "kind": kind, "summary": summary, "inputs": inputs}


def confirmation_counts(bundle):
    cells = bundle["summary"]["cells"]
    counts = {
        "recovery_factorial": sum(c["executions"] for c in cells if c["study_arm"] == "recovery_factorial"
                                 and c["controller"] in ("none", "v5")),
        "additional_queue_candidate": sum(c["executions"] for c in cells if c["study_arm"] == "recovery_factorial"
                                         and c["controller"] == "queue_budget"),
        "observation_confirmation": sum(c["executions"] for c in cells if c["study_arm"] == "observation_confirmation"),
    }
    if (counts["recovery_factorial"] != 960 or counts["additional_queue_candidate"] not in (0, 480)
            or counts["observation_confirmation"] != 240
            or sum(counts.values()) != bundle["summary"]["executions"]):
        raise ValueError("Require 960 factorial + optional 480 candidate + 240 distinct semantic executions")
    if bundle["summary"].get("observation_confirmation", {}).get("not_pooled_with_historical_bc") is not True:
        raise ValueError("Matched new BC policies must remain separate from the historical BC factorial")
    return counts


def aggregate_success(bundle, *, arm, groups):
    """Only combine the fixed WM seeds within an explicitly named study arm."""
    rows = []
    expected_seeds = (17,) if bundle["kind"] == "budget" else (17, 23, 41)
    for noise in NOISES:
        for label, method, controller, semantics, color in groups:
            cells = [cell for cell in bundle["summary"]["cells"] if
                     cell["study_arm"] == arm and cell["method"] == method
                     and cell["controller"] == controller and cell["observation_semantics"] == semantics
                     and cell["noise"] == noise]
            seeds = (None,) if method == "bc" else expected_seeds
            if len(cells) != len(seeds) or {cell["world_model_seed"] for cell in cells} != set(seeds):
                raise ValueError(f"Missing/duplicated explicit cell: {arm}, {label}, noise={noise}")
            executions = sum(cell["executions"] for cell in cells)
            successes = sum(cell["successes"] for cell in cells)
            rows.append({"label": label, "method": method, "controller": controller,
                         "observation_semantics": semantics, "study_arm": arm, "noise": noise,
                         "independent_starts": bundle["summary"]["independent_starts"],
                         "fixed_world_model_seeds": ";".join(str(seed) for seed in seeds if seed is not None),
                         "executions": executions, "successes": successes,
                         "success_percent": 100. * float(np.mean([cell["success_rate"] for cell in cells])),
                         "color": color})
    return rows


def select_effects(bundle, definitions):
    """Definitions specify label, contrast, method, comparison, and study arm."""
    rows = []
    for label, contrast, method, comparison, arm in definitions:
        for metric in ("success", "steps"):
            for noise in NOISE_KEYS:
                candidates = [effect for effect in bundle["summary"]["paired_effects"]
                              if effect["contrast"] == contrast and effect["method"] == method
                              and effect["comparison"] == comparison and effect["study_arm"] == arm
                              and effect["metric"] == metric and str(effect["noise"]) == noise]
                if len(candidates) != 1:
                    raise ValueError(f"Expected one effect: {contrast}/{method}/{comparison}/{noise}/{metric}")
                expected_unit = "percentage_points" if metric == "success" else "controls"
                if candidates[0]["unit"] != expected_unit:
                    raise ValueError("Unexpected effect units")
                rows.append({"figure_label": label, **candidates[0]})
    return rows


def write_rows(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def configure_plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#B0B8BE", "axes.labelcolor": "#26343E", "text.color": "#26343E",
        "xtick.color": "#42525E", "ytick.color": "#42525E", "savefig.facecolor": "white",
        "figure.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.hashsalt": "robot_can_graduation_repairs_20261022_01",
    })
    return plt


def export_figure(fig, output, name, title, caption):
    target = output / name
    fig.savefig(target.with_suffix(".png"), dpi=300, metadata={"Software": "robot_can build_figures.py"})
    fig.savefig(target.with_suffix(".svg"), metadata={"Title": title, "Description": caption,
                "Creator": "robot_can build_figures.py", "Date": FIXED_EXPORT_DATE.isoformat()})
    fig.savefig(target.with_suffix(".pdf"), metadata={"Title": title, "Subject": caption,
                "Creator": "robot_can build_figures.py", "CreationDate": FIXED_EXPORT_DATE,
                "ModDate": FIXED_EXPORT_DATE})


def plot_success(plt, output, name, title, caption, rows):
    labels = list(dict.fromkeys(row["label"] for row in rows))
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.4), sharey=True)
    for axis, noise, condition in zip(axes, NOISES, NOISE_LABELS):
        selected = [row for row in rows if row["noise"] == noise]
        x = np.arange(len(labels))
        rates = [row["success_percent"] for row in selected]
        axis.bar(x, rates, color=[row["color"] for row in selected], width=.65, zorder=3)
        for i, row in enumerate(selected):
            high = row["success_percent"] > 89
            axis.text(i, row["success_percent"] - 4 if high else row["success_percent"] + 2.2,
                      f'{row["success_percent"]:.1f}%', ha="center", va="top" if high else "bottom",
                      fontsize=10, color="white" if high else "#26343E", fontweight="bold")
            axis.text(i, 3., f'{row["successes"]}/{row["executions"]} exec.',
                      ha="center", va="bottom", fontsize=8, color="white" if row["success_percent"] > 15 else "#26343E")
        axis.set_xticks(x, labels)
        axis.set_ylim(0, 100)
        axis.set_yticks(np.arange(0, 101, 20))
        axis.set_title(condition, pad=12)
        axis.grid(axis="y", color="#E5E9EC", linewidth=.8, zorder=0)
    axes[0].set_ylabel("Benchmark success (%)")
    fig.suptitle(title, x=.07, y=.96, ha="left", fontsize=15, fontweight="bold")
    fig.subplots_adjust(left=.07, right=.98, top=.80, bottom=.28, wspace=.15)
    fig.text(.07, .035, caption, ha="left", va="bottom", fontsize=8.2, linespacing=1.6)
    export_figure(fig, output, name, title, caption)
    plt.close(fig)
    write_rows(output / f"{name}_data.csv", rows)


def plot_effects(plt, output, name, title, caption, rows):
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    labels = list(dict.fromkeys(row["figure_label"] for row in rows))
    height = max(4.8, .60 * len(labels) + 2.6)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, height), sharey=True)
    for axis, metric in zip(axes, ("success", "steps")):
        selected = [row for row in rows if row["metric"] == metric]
        for i, label in enumerate(labels):
            for noise_index, noise in enumerate(NOISE_KEYS):
                row = next(row for row in selected if row["figure_label"] == label and str(row["noise"]) == noise)
                y = i + (noise_index - 1) * .19
                axis.hlines(y, row["ci95_low"], row["ci95_high"], color=NOISE_COLORS[noise_index], linewidth=2, zorder=3)
                axis.plot([row["ci95_low"], row["ci95_high"]], [y, y], "|", color=NOISE_COLORS[noise_index], markersize=6, zorder=3)
                axis.scatter(row["effect"], y, color=NOISE_COLORS[noise_index], s=27, zorder=4)
        low = min([0.] + [row["ci95_low"] for row in selected] + [row["effect"] for row in selected])
        high = max([0.] + [row["ci95_high"] for row in selected] + [row["effect"] for row in selected])
        padding = max(1., .12 * (high - low))
        axis.set_xlim(low - padding, high + padding)
        axis.axvline(0, color="#68757D", linestyle="--", linewidth=1)
        axis.set_ylim(len(labels) - .45, -.55)
        axis.set_yticks(np.arange(len(labels)), labels)
        axis.grid(axis="x", color="#E5E9EC", linewidth=.8, zorder=0)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.set_title("Success difference" if metric == "success" else "Capped completion controls", pad=13)
        axis.set_xlabel("Percentage points" if metric == "success" else "Controls (400-step failure cap)")
    axes[1].tick_params(labelleft=False)
    legend = [Line2D([0], [0], marker="o", color=color, markersize=5, linewidth=1.5, label=label)
              for color, label in zip(NOISE_COLORS, NOISE_LABELS)]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.60, .91), ncol=3, frameon=False, fontsize=9)
    fig.suptitle(title, x=.025, y=.98, ha="left", fontsize=15, fontweight="bold")
    fig.subplots_adjust(left=.29, right=.98, top=.77, bottom=.29, wspace=.25)
    fig.text(.025, .025, caption, ha="left", va="bottom", fontsize=8.2, linespacing=1.6)
    export_figure(fig, output, name, title, caption)
    plt.close(fig)
    write_rows(output / f"{name}_data.csv", rows)


def build_plan(confirmation, development_observations=None, development_budget=None):
    counts = confirmation_counts(confirmation)
    plan = []
    frozen = [("BC\nNo recovery", "bc", "none", "legacy", "#8DB0C9"),
              ("BC\nv5 recovery", "bc", "v5", "legacy", "#366E96"),
              ("Full\nNo recovery", "full", "none", "legacy", "#E4B887"),
              ("Full\nv5 recovery", "full", "v5", "legacy", "#B97236")]
    scope = ("60 paired independent starts; 960 factorial executions. Historical frozen BC is shared across the four groups.\n"
             "Full averages three fixed WM seeds; execution counts are not independent sample counts. D2; at most 400 controls.")
    paired = ("60 paired independent starts; three fixed WM seeds averaged within start before bootstrap; two noises equally weighted in joint effects.\n"
              "Dots: paired means. Lines: pointwise 95% percentile intervals from 20,000 start resamples; no multiplicity correction.\n"
              "Positive success differences follow the named contrast. Negative capped-control differences mean fewer controls; failures count as 400.")
    factorial_definitions = [
        ("BC: v5 - none", "recovery_effect", "bc", "v5 - none", "recovery_factorial"),
        ("Full: v5 - none", "recovery_effect", "full", "v5 - none", "recovery_factorial"),
        ("Interaction: Full gain - BC gain", "recovery_interaction", "full_minus_bc", "(Full v5-none) - (BC v5-none)", "recovery_factorial"),
        ("Full - BC: no recovery", "world_model_effect", "none", "Full - BC with none", "recovery_factorial"),
        ("Full - BC: v5 recovery", "world_model_effect", "v5", "Full - BC with v5", "recovery_factorial"),
    ]
    plan.append(("success", "recovery_factorial_success", "Frozen recovery factorial: independent confirmation", scope,
                 aggregate_success(confirmation, arm="recovery_factorial", groups=frozen)))
    plan.append(("effects", "recovery_factorial_effects", "Recovery and WM effects: independent confirmation", paired,
                 select_effects(confirmation, factorial_definitions)))
    candidate_groups = [("BC\nv5", "bc", "v5", "legacy", "#8DB0C9"),
                        ("BC\nQueue budget", "bc", "queue_budget", "legacy", "#366E96"),
                        ("Full\nv5", "full", "v5", "legacy", "#E4B887"),
                        ("Full\nQueue budget", "full", "queue_budget", "legacy", "#B97236")]
    if counts["additional_queue_candidate"]:
        candidate_scope = ("60 paired independent starts; 480 additional candidate executions plus the 480 v5 reference executions.\n"
                           "The v5 reference is reused from the factorial; Full averages three fixed WM seeds. This is a separate candidate comparison.")
        candidate_definitions = [(f"{method.upper() if method == 'bc' else 'Full'}: queue budget - v5", "recovery_effect", method,
                                  "queue_budget - v5", "recovery_factorial") for method in ("bc", "full")]
        plan.append(("success", "queue_budget_candidate_success", "Queue budget candidate: independent confirmation", candidate_scope,
                     aggregate_success(confirmation, arm="recovery_factorial", groups=candidate_groups)))
        plan.append(("effects", "queue_budget_candidate_effects", "Queue budget candidate: paired effects", paired,
                     select_effects(confirmation, candidate_definitions)))
    semantic_groups = [("New legacy BC", "bc", "none", "legacy", "#527EAA"),
                       ("Canonical BC", "bc", "none", "canonical", "#448B78")]
    semantic_scope = ("60 paired independent starts; 240 semantic executions. Matched newly trained final checkpoints; both use no recovery.\n"
                      "This arm is separate from the historical-BC factorial. Each policy uses its matching online observation semantics.")
    semantic_definitions = [("Canonical - new legacy", "observation_confirmation_canonical_minus_legacy", "bc",
                             "matched newly trained canonical - matched newly trained legacy", "observation_confirmation")]
    plan.append(("success", "observation_confirmation_success", "Matched observation semantics: independent confirmation", semantic_scope,
                 aggregate_success(confirmation, arm="observation_confirmation", groups=semantic_groups)))
    plan.append(("effects", "observation_confirmation_effects", "Matched observation semantics: paired effects",
                 "60 paired independent starts; newly trained BC policies only; never pooled with historical frozen BC.\n"
                 "Dots: paired means; lines: pointwise 95% paired-start bootstrap intervals (20,000 draws). Joint effects equally weight both noises.\n"
                 "Positive success differences favor canonical; negative capped-control differences mean fewer controls. No multiplicity correction.",
                 select_effects(confirmation, semantic_definitions)))
    if development_observations:
        dev_scope = ("DEVELOPMENT ONLY: 30 reused demonstration starts; 120 executions. Matched newly trained final BC checkpoints; no recovery.\n"
                     "These starts were available during development and are not additional independent confirmation samples.")
        dev_definitions = [("Canonical - new legacy", "canonical_minus_legacy", "bc", "canonical - legacy", "recovery_factorial")]
        plan.append(("success", "development_observation_success", "Observation semantics: development comparison (30 reused starts)", dev_scope,
                     aggregate_success(development_observations, arm="recovery_factorial", groups=semantic_groups)))
        plan.append(("effects", "development_observation_effects", "Observation semantics: development effects (30 reused starts)",
                     dev_scope + "\nPointwise 95% paired-start bootstrap intervals, 20,000 draws; noise conditions averaged within start for joint effects.",
                     select_effects(development_observations, dev_definitions)))
    if development_budget:
        dev_scope = ("DEVELOPMENT ONLY: 30 reused demonstration starts; 240 executions; Full uses fixed WM seed 17.\n"
                     "This comparison was used before confirmation. A passed noninferiority gate does not establish improvement.")
        dev_definitions = [(f"{method.upper() if method == 'bc' else 'Full'}: queue budget - v5", "queue_budget_minus_v5", method,
                            "queue_budget - v5", "recovery_factorial") for method in ("bc", "full")]
        plan.append(("success", "development_budget_success", "Queue budget: development comparison (30 reused starts)", dev_scope,
                     aggregate_success(development_budget, arm="recovery_factorial", groups=candidate_groups)))
        plan.append(("effects", "development_budget_effects", "Queue budget: development effects (30 reused starts)",
                     dev_scope + "\nPointwise 95% paired-start bootstrap intervals, 20,000 draws; negative capped-control differences mean fewer controls.",
                     select_effects(development_budget, dev_definitions)))
    return plan, counts


def validate_existing(output, inputs):
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("script_sha256") != sha(__file__):
        raise ValueError("Existing figure manifest differs from current generator")
    if manifest.get("inputs_sha256") != {str(path): digest for path, digest in sorted(inputs.items())}:
        raise ValueError("Existing figure manifest differs from current inputs")
    for name, expected in manifest["outputs_sha256"].items():
        path = (output / name).resolve()
        if not path.is_relative_to(output) or sha(path) != expected:
            raise ValueError(f"Existing figure artifact changed: {name}")
    actual = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    if actual != set(manifest["outputs_sha256"]) | {"manifest.json"}:
        raise ValueError("Unexpected/missing files in existing figure directory")
    return manifest


def main(args):
    output = args.output_dir.expanduser().resolve()
    if output.exists() and not args.validate_only:
        raise FileExistsError(f"Refusing existing figure directory: {output}")
    confirmation = load_analysis(args.analysis_dir, "confirmation")
    development_observations = load_analysis(args.dev_observations, "observation") if args.dev_observations else None
    development_budget = load_analysis(args.dev_budget, "budget") if args.dev_budget else None
    bundles = [bundle for bundle in (confirmation, development_observations, development_budget) if bundle]
    plan, counts = build_plan(confirmation, development_observations, development_budget)
    inputs = {path: digest for bundle in bundles for path, digest in bundle["inputs"].items()}
    if args.validate_only:
        if output.exists():
            validate_existing(output, inputs)
        result = {"validated": True, "written": False, "planned_figures": len(plan), "confirmation_execution_counts": counts}
        print(json.dumps(result))
        return result
    output.mkdir(parents=True, exist_ok=False)
    plt = configure_plotting()
    for bundle in bundles:
        for name in ("cells.csv", "paired_effects.csv"):
            shutil.copyfile(bundle["folder"] / name, output / f'source_{bundle["kind"]}_{name}')
    for kind, name, title, caption, rows in plan:
        function = plot_success if kind == "success" else plot_effects
        function(plt, output, name, title, caption, rows)
    # Refuse a successful receipt if any analysis/stage input changed mid-export.
    if any(sha(path) != digest for path, digest in inputs.items()):
        raise RuntimeError("An input changed during figure export; outputs retained without a completion manifest")
    manifest = {"schema_version": 1, "status": "complete", "script_sha256": sha(__file__),
                "inputs_sha256": {str(path): digest for path, digest in sorted(inputs.items())},
                "outputs_sha256": {path.relative_to(output).as_posix(): sha(path) for path in sorted(output.rglob("*")) if path.is_file()},
                "figures": [{"name": name, "title": title, "caption": caption} for _, name, title, caption, _ in plan],
                "confirmation_execution_counts": counts, "independent_confirmation_starts": 60,
                "policy_arms_pooled": False, "exports": ["PNG 300 dpi", "SVG", "PDF", "plotted-data CSV", "source CSV"],
                "intervals": "Pointwise 95% paired-start percentile bootstrap; no multiple-comparison correction",
                "deterministic_export_metadata_date": FIXED_EXPORT_DATE.isoformat(),
                "runtime": {"python": sys.version, "numpy": np.__version__, "matplotlib": importlib.metadata.version("matplotlib")}}
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "complete", "figures": len(plan), "output_dir": str(output)}))
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=RUN / "analysis/confirmation")
    parser.add_argument("--output-dir", type=Path, default=RUN / "figures")
    parser.add_argument("--dev-observations", type=Path, help="Optional completed development observation analysis directory")
    parser.add_argument("--dev-budget", type=Path, help="Optional completed development budget analysis directory")
    parser.add_argument("--validate-only", action="store_true", help="Validate complete inputs and, if present, existing outputs without writing")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
