"""Geometry and data-boundary tests for the new observation contract."""
import unittest
from unittest import mock

import numpy as np

from robot_can_v2 import observations


def state(eef_position=(0., 0., 0.), eef_quaternion=(0., 0., 0., 1.),
          object_position=(1., 2., 3.), object_quaternion=(0., 0., 0., 1.)):
    result = np.zeros(23, dtype=np.float64)
    result[:3] = eef_position
    result[3:7] = eef_quaternion
    result[7:9] = (.01, -.01)
    result[9:16] = 99.  # Intentionally stale, impossible relative pose.
    result[16:19] = object_position
    result[19:23] = object_quaternion
    return result


class ObservationTests(unittest.TestCase):
    def test_identity_translation_and_input_unchanged(self):
        raw = state(eef_position=(2., 1., -1.), object_position=(3., 4., 1.))
        before = raw.copy()
        result = observations.canonicalize_state(raw)
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(raw, before)
        np.testing.assert_allclose(result[9:12], (1., 3., 2.))
        np.testing.assert_allclose(result[12:16], (0., 0., 0., 1.))
        np.testing.assert_array_equal(result[7:9], raw[7:9].astype(np.float32))

    def test_rotated_reference_frame_and_quaternion_product(self):
        half = np.sqrt(.5)
        raw = state(eef_position=(1., 1., 0.),
                    eef_quaternion=(0., 0., 2 * half, 2 * half),
                    object_position=(1., 2., 0.))
        result = observations.canonicalize_state(raw)
        np.testing.assert_allclose(result[9:12], (1., 0., 0.), atol=1e-7)
        np.testing.assert_allclose(result[12:16], (0., 0., -half, half), atol=1e-7)
        self.assertAlmostEqual(np.linalg.norm(result[3:7]), 1., places=6)

    def test_batch_shape_and_relative_quaternion_sign_equivalence(self):
        raw = np.broadcast_to(state(object_quaternion=(0., 2., 0., 0.)), (2, 3, 23)).copy()
        raw[0, 1, 3:7] *= -1
        raw[1, 0, 19:23] *= -1
        raw[1, 1, 3:7] *= -1
        raw[1, 1, 19:23] *= -1
        result = observations.canonicalize_state(raw)
        self.assertEqual(result.shape, (2, 3, 23))
        np.testing.assert_allclose(result[..., 9:16], np.broadcast_to(
            np.array([1., 2., 3., 0., 1., 0., 0.]), (2, 3, 7)))

    def test_invalid_shapes_nonfinite_and_quaternion_rejected(self):
        for shape in ((), (22,), (2, 24)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                observations.canonicalize_state(np.zeros(shape))
        for index, bad in ((1, np.nan), (14, np.inf), (20, -np.inf)):
            raw = state()
            raw[index] = bad
            with self.subTest(index=index), self.assertRaises(ValueError):
                observations.canonicalize_state(raw)
        for sl in (slice(3, 7), slice(19, 23)):
            raw = state()
            raw[sl] = 0
            with self.assertRaisesRegex(ValueError, "nonzero"):
                observations.canonicalize_state(raw)

    def test_online_offline_identical_and_legacy_delegates(self):
        raw = state(eef_position=(.3, -.2, .4))
        frame = {"robot0_eef_pos": raw[:3], "robot0_eef_quat": raw[3:7],
                 "robot0_gripper_qpos": raw[7:9], "object-state": raw[9:]}
        np.testing.assert_array_equal(observations.pack_observation(frame),
                                      observations.canonicalize_state(raw.astype(np.float32)))
        with mock.patch.object(observations.common, "pack_obs", wraps=observations.common.pack_obs) as packer:
            legacy = observations.pack_observation(frame, "legacy")
            packer.assert_called_once_with(frame)
        np.testing.assert_array_equal(legacy, raw.astype(np.float32))

    def test_training_reader_blocks_other_splits_before_io(self):
        with mock.patch.object(observations.common, "read_episodes") as reader:
            for split in ("sealed_test", "test", "all", "train+dev"):
                with self.subTest(split=split), self.assertRaises(ValueError):
                    observations.read_episodes(split)
            reader.assert_not_called()

    def test_reader_canonicalizes_both_frames_and_makes_arrays_readonly(self):
        episode = {"name": "fixture", "x": state()[None], "xn": state()[None],
                   "a": np.zeros((1, 7), dtype=np.float32), "reward": np.ones(1, dtype=np.float32)}
        with mock.patch.object(observations.common, "read_episodes", return_value=[episode]):
            result = observations.read_episodes("dev")[0]
        for key in ("x", "xn", "a", "reward"):
            self.assertFalse(result[key].flags.writeable)
        np.testing.assert_allclose(result["xn"][0, 9:12], (1., 2., 3.))


if __name__ == "__main__":
    unittest.main()
