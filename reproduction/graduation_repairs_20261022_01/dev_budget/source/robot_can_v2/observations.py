"""Versioned observation semantics for new experiments.

The public demonstration has 23 entries: EEF world position (0:3), EEF
world quaternion in xyzw order (3:7), gripper joints (7:9), object pose in
EEF coordinates (9:16), and object world pose (16:23).  Relative sensors
can be stale, especially at reset.  Canonical observations therefore use
only the absolute poses from the current frame to rebuild entries 9:16.
Historical code, HDF5 data, and checkpoints remain untouched.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

import common


STATE_DIM = 23
SEMANTICS = ("canonical", "legacy")
OBSERVATION_SCHEMA = "eef_world_xyzw_grip_object_relative_xyzw_object_world_xyzw_v1"
CANONICAL_CONTRACT = (
    "same-frame absolute poses; relative xyz=R_eef.T@(p_object-p_eef); "
    "relative xyzw=conjugate(q_eef)*q_object; normalized absolute and relative "
    "quaternions; relative sign w-positive, xyz lexicographic at w=0"
)


def _check_semantics(semantics: str) -> None:
    if semantics not in SEMANTICS:
        raise ValueError(f"observation semantics must be one of {SEMANTICS}, got {semantics!r}")


def _normalized_quaternion(q: np.ndarray, name: str) -> np.ndarray:
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError(f"{name} must have nonzero norm")
    return q / norm


def _canonical_quaternion_sign(q: np.ndarray) -> np.ndarray:
    # q and -q encode the same rotation.  Select a unique representation,
    # including the exact 180-degree case whose scalar component is zero.
    pivot = q[..., 3].copy()
    for index in (0, 1, 2):
        pivot = np.where(pivot == 0, q[..., index], pivot)
    return q * np.where(pivot < 0, -1.0, 1.0)[..., None]


def canonicalize_state(state: np.ndarray) -> np.ndarray:
    """Return a new float32 array with the same ``(..., 23)`` shape.

    Absolute quaternion signs are retained after normalization.  The
    relative quaternion has a deterministic sign, so flipping either
    absolute quaternion does not change the rebuilt relative features.
    All entries must be finite; only the two absolute quaternion norms
    need be nonzero because the original relative pose is discarded.
    """
    value = np.asarray(state)
    if value.ndim < 1 or value.shape[-1] != STATE_DIM:
        raise ValueError(f"state must have shape (..., {STATE_DIM}), got {value.shape}")
    if not np.issubdtype(value.dtype, np.number) or np.iscomplexobj(value):
        raise ValueError("state must contain real numbers")
    value = value.astype(np.float64, copy=True)
    if not np.isfinite(value).all():
        raise ValueError("state must contain only finite values")
    eef_q = _normalized_quaternion(value[..., 3:7], "EEF quaternion")
    obj_q = _normalized_quaternion(value[..., 19:23], "object quaternion")
    value[..., 3:7] = eef_q
    value[..., 19:23] = obj_q

    # Rotate the world displacement by inverse(q_eef), using xyzw order.
    vector = value[..., 16:19] - value[..., 0:3]
    conjugate_xyz = -eef_q[..., :3]
    crossed = np.cross(conjugate_xyz, vector)
    value[..., 9:12] = vector + 2 * (
        eef_q[..., 3:4] * crossed + np.cross(conjugate_xyz, crossed)
    )

    # Hamilton product inverse(q_eef) * q_object.
    obj_xyz, obj_w = obj_q[..., :3], obj_q[..., 3:4]
    eef_w = eef_q[..., 3:4]
    relative = np.concatenate(
        (
            eef_w * obj_xyz + obj_w * conjugate_xyz + np.cross(conjugate_xyz, obj_xyz),
            eef_w * obj_w - np.sum(conjugate_xyz * obj_xyz, axis=-1, keepdims=True),
        ),
        axis=-1,
    )
    value[..., 12:16] = _canonical_quaternion_sign(
        _normalized_quaternion(relative, "relative quaternion")
    )
    result = value.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("state values exceed the finite float32 range")
    return result


def pack_observation(obs: Mapping[str, np.ndarray], semantics: str = "canonical") -> np.ndarray:
    """Pack an online frame with exactly the same contract as offline data.

    ``legacy`` delegates directly to the historical packer, preserving old
    checkpoint behavior and its original sensor-cache semantics.
    """
    _check_semantics(semantics)
    packed = common.pack_obs(obs)
    return canonicalize_state(packed) if semantics == "canonical" else packed


def read_episodes(split: str = "train", semantics: str = "canonical") -> list[dict]:
    """Read only train/dev episodes; return read-only in-memory arrays.

    The original reader opens HDF5 in mode ``r``.  This wrapper never
    exposes the sealed test set to training or checkpoint selection.
    """
    _check_semantics(semantics)
    if split not in ("train", "dev"):
        raise ValueError("training episode reader permits only 'train' or 'dev'; sealed test is excluded")
    episodes = common.read_episodes(split)
    for episode in episodes:
        for key in ("x", "xn"):
            if semantics == "canonical":
                episode[key] = canonicalize_state(episode[key])
            elif episode[key].shape[-1] != STATE_DIM or not np.isfinite(episode[key]).all():
                raise ValueError(f"invalid legacy {key} observations in {episode['name']}")
        for key in ("x", "xn", "a", "reward"):
            if not np.isfinite(episode[key]).all():
                raise ValueError(f"nonfinite {key} in {episode['name']}")
            episode[key].setflags(write=False)
    return episodes
