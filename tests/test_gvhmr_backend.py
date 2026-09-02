from types import SimpleNamespace

import numpy as np
import pytest

from robot_keyframe_kit.video.gvhmr_backend import (
    GVHMRBackendError,
    SMPL_24_JOINT_NAMES,
    motion_from_gvhmr_result,
)


def result_fixture(frames=3):
    return SimpleNamespace(
        joints_world=np.zeros((frames, 24, 3), dtype=np.float32),
        smpl_params_world={
            "global_orient": np.zeros((frames, 3), dtype=np.float32),
            "transl": np.zeros((frames, 3), dtype=np.float32),
        },
        joint_confidence=np.ones((frames, 24), dtype=np.float32),
        fps=25.0,
    )


def test_result_is_normalized_to_human_motion_contract():
    motion = motion_from_gvhmr_result(
        result_fixture(),
        source_video_sha256="b" * 64,
        backend_version="test-version",
    )

    assert motion.joint_names == SMPL_24_JOINT_NAMES
    np.testing.assert_allclose(motion.timestamps, [0.0, 0.04, 0.08])
    np.testing.assert_array_equal(motion.root_quaternion_wxyz[:, 0], np.ones(3))
    motion.validate()


def test_result_rejects_multiple_people_instead_of_selecting_one_silently():
    result = result_fixture()
    result.joints_world = np.zeros((2, 3, 24, 3), dtype=np.float32)

    with pytest.raises(GVHMRBackendError, match="exactly one person"):
        motion_from_gvhmr_result(
            result,
            source_video_sha256="b" * 64,
            backend_version="test-version",
        )
