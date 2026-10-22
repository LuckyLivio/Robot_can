"""Read-only observation audit; no training or closed-loop performance estimate.

Run from the project root with the existing virtual environment and -B.
The default JSON is created exclusively. Use --stdout-only to repeat the audit
without changing the saved receipt. Five demonstration actions are executed in
memory only to inspect online sensor semantics; this is not a success-rate test.
"""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from common import DATA, OBS_KEYS, load_policy, make_env, pack_obs, restore_demo
from robosuite.utils import transform_utils as T
from smoke_env import geometry_error


def summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "n": int(values.size),
        "above_1e_minus_3": int(np.count_nonzero(values > 1e-3)),
        "max": float(values.max()),
        "median": float(np.median(values)),
    }


def observation_errors(x, previous_object=False):
    current = x[1:] if previous_object else x
    objects = x[:-1] if previous_object else x
    eef = Rotation.from_quat(current[:, 3:7])
    obj = Rotation.from_quat(objects[:, 19:23])
    relative = Rotation.from_quat(current[:, 12:16])
    expected_position = eef.inv().apply(objects[:, 16:19] - current[:, :3])
    position_error = np.max(np.abs(expected_position - current[:, 9:12]), axis=1)
    rotation_error = np.max(
        np.abs((eef.inv() * obj).as_matrix() - relative.as_matrix()), axis=(1, 2)
    )
    return position_error, rotation_error


def corrected_first_frame(x):
    corrected = x.copy()
    pose = T.pose_in_A_to_pose_in_B(
        T.pose2mat((x[16:19], x[19:23])),
        T.pose_inv(T.pose2mat((x[:3], x[3:7]))),
    )
    corrected[9:12], corrected[12:16] = T.mat2pose(pose)
    return corrected


def run():
    torch.set_num_threads(1)
    splits = json.loads((ROOT / "data/splits.json").read_text(encoding="utf-8"))
    checkpoint = ROOT / "policy/bc_chunk_dev_v4/policy_15000.pt"
    policy = load_policy(checkpoint)
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "Observation geometry and frozen BC forward sensitivity only; not closed-loop performance.",
        "definitions": {
            "relative_features": "x[9:12] position and x[12:16] xyzw quaternion",
            "absolute_features": "EEF x[0:7], object x[16:23]",
            "rotation_error": "Maximum absolute rotation-matrix entry difference; invariant to quaternion sign.",
            "same_time": "inverse(EEF_t) * object_t",
            "previous_object": "inverse(EEF_t) * object_(t-1), for t > 0",
            "threshold": "1e-3 m for position and 1e-3 dimensionless for matrix entries; descriptive thresholds only.",
            "bc_sensitivity": "Change only first-frame relative features; compare frozen policy best() chunk index k=2.",
        },
        "splits": {},
    }
    raw_training_windows = []
    corrected_training_windows = []
    with h5py.File(DATA, "r") as data:
        for split in ("train", "dev", "sealed_test"):
            first_p, first_r, post_p, post_r, lag_p, lag_r = [], [], [], [], [], []
            first_raw, first_corrected = [], []
            for name in splits[split]:
                x = np.concatenate([data["data"][name]["obs"][key][:] for key in OBS_KEYS], axis=-1).astype(np.float32)
                p, r = observation_errors(x)
                lp, lr = observation_errors(x, previous_object=True)
                first_p.append(p[0]); first_r.append(r[0])
                post_p.extend(p[1:]); post_r.extend(r[1:])
                lag_p.extend(lp); lag_r.extend(lr)
                corrected = corrected_first_frame(x[0])
                first_raw.append(x[0]); first_corrected.append(corrected)
                if split == "train":
                    n = len(x) - policy.horizon + 1
                    window = x[:n].copy()
                    raw_training_windows.append(window)
                    repaired = window.copy(); repaired[0] = corrected
                    corrected_training_windows.append(repaired)
            entry = {
                "parent_count": len(splits[split]),
                "same_time_first_frame": {"position": summary(first_p), "rotation_matrix": summary(first_r)},
                "same_time_post_first_frame": {"position": summary(post_p), "rotation_matrix": summary(post_r)},
                "previous_object_post_first_frame": {"position": summary(lag_p), "rotation_matrix": summary(lag_r)},
            }
            if split in ("train", "dev"):
                raw_action = policy.best(torch.from_numpy(np.asarray(first_raw))).numpy()[:, 2]
                corrected_action = policy.best(torch.from_numpy(np.asarray(first_corrected))).numpy()[:, 2]
                difference = np.abs(raw_action - corrected_action)
                entry["frozen_bc_first_frame_k2_forward_sensitivity"] = {
                    "mean_absolute_action_difference": float(difference.mean()),
                    "maximum_absolute_action_difference": float(difference.max()),
                    "parents_with_any_action_dimension_difference_above_0_1": int(np.count_nonzero(difference.max(axis=1) > .1)),
                    "parent_count": len(first_raw),
                    "not_a_performance_test": True,
                }
            result["splits"][split] = entry

        env = make_env(False)
        try:
            demo = data["data/demo_1"]
            obs = restore_demo(env, demo)
            online = {"demo": "demo_1", "restored_same_time_geometry_error": geometry_error(obs), "steps": []}
            for step in range(5):
                obs, *_ = env.step(demo["actions"][step])
                online["steps"].append({"step": step + 1, "same_time_geometry_error": geometry_error(obs)})
            result["online_five_demo_actions"] = online
        finally:
            env.close()

    raw = np.concatenate(raw_training_windows)
    repaired = np.concatenate(corrected_training_windows)
    raw_mean = torch.from_numpy(raw).mean(0).numpy()
    result["checkpoint_normalization_consistency"] = {
        "training_windows": len(raw),
        "checkpoint_mean_vs_raw_windows_max_absolute_error": float(np.max(np.abs(policy.mean.numpy() - raw_mean))),
        "checkpoint_relative_position_mean": policy.mean.numpy()[9:12].tolist(),
        "raw_windows_relative_position_mean": raw_mean[9:12].tolist(),
        "only_initial_relative_features_corrected_position_mean": repaired[:, 9:12].mean(0).tolist(),
        "interpretation": "Checkpoint normalization agrees with current raw-data training windows. This is evidence for input preprocessing, not proof of every historical training command.",
    }
    paths = [DATA, ROOT / "data/splits.json", checkpoint, ROOT / "common.py", ROOT / "train_policy.py", ROOT / "smoke_env.py", Path(__file__)]
    result["sha256"] = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    result["packages"] = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "torch", "h5py", "robosuite", "mujoco")}
    result["interpretation"] = [
        "Every training trajectory has a stale initial relative pose; the error is not solely a q versus -q quaternion sign change.",
        "For later logged frames, relative observations match current EEF with the previous frame's object pose to numerical precision.",
        "restore_demo refreshes the initial observation twice; ordinary env.step observations retain a small sensor-order lag in this five-step check.",
        "The fixed BC is sensitive to replacing initial relative features with geometrically consistent ones. This check does not determine whether replacement improves or harms task success.",
        "Methods in the existing paired evaluations share the same frozen BC. These findings do not by themselves invalidate the historical paired outcome records.",
    ]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stdout-only", action="store_true", help="Repeat without creating or changing the saved JSON.")
    args = parser.parse_args()
    destination = Path(__file__).with_name("observation_semantics.json")
    if not args.stdout_only and destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    payload = run()
    if args.stdout_only:
        print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        print(json.dumps({"receipt": str(destination), "splits": payload["splits"], "online_five_demo_actions": payload["online_five_demo_actions"]}, ensure_ascii=False, allow_nan=False))
