import json

import numpy as np
import pytest

from robot_keyframe_kit.video.human_motion import HumanMotionV1, HumanMotionValidationError


def valid_motion() -> HumanMotionV1:
    return HumanMotionV1(
        timestamps=np.array([0.0, 0.02, 0.04], dtype=np.float32),
        joints_world=np.zeros((3, 24, 3), dtype=np.float32),
        joint_confidence=np.ones((3, 24), dtype=np.float32),
        root_position=np.zeros((3, 3), dtype=np.float32),
        root_quaternion_wxyz=np.tile(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (3, 1)
        ),
        joint_names=tuple(f"joint_{index}" for index in range(24)),
        source_fps=50.0,
        source_video_sha256="a" * 64,
        backend="gvhmr",
        backend_version="1.0",
    )


def test_round_trip_preserves_versioned_contract(tmp_path):
    path = tmp_path / "human-motion.npz"
    valid_motion().save(path)

    loaded = HumanMotionV1.load(path)

    assert loaded.schema_version == 1
    assert loaded.backend == "gvhmr"
    assert loaded.joint_names[3] == "joint_3"
    np.testing.assert_allclose(loaded.timestamps, [0.0, 0.02, 0.04], rtol=0.0, atol=1e-8)
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {
            "timestamps",
            "joints_world",
            "joint_confidence",
            "root_position",
            "root_quaternion_wxyz",
            "joint_names",
            "source_fps",
            "source_video_sha256",
            "backend",
            "backend_version",
            "schema_version",
        }


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("timestamps", np.array([0.0, 0.02, 0.01], dtype=np.float32), "strictly increasing"),
        ("joint_confidence", np.full((3, 24), 1.1, dtype=np.float32), r"within \[0, 1\]"),
        ("root_position", np.full((3, 3), np.nan, dtype=np.float32), "finite"),
        (
            "root_quaternion_wxyz",
            np.zeros((3, 4), dtype=np.float32),
            "unit quaternions",
        ),
    ],
)
def test_validation_rejects_invalid_motion(field, replacement, message):
    motion = valid_motion()
    object.__setattr__(motion, field, replacement)

    with pytest.raises(HumanMotionValidationError, match=message):
        motion.validate()


def test_validation_rejects_long_low_confidence_joint_run():
    motion = valid_motion()
    confidence = motion.joint_confidence.copy()
    confidence[:, 7] = 0.0
    object.__setattr__(motion, "joint_confidence", confidence)

    with pytest.raises(HumanMotionValidationError, match="joint_7.*low confidence"):
        motion.validate(min_confidence=0.25, max_low_confidence_fraction=0.5)


def test_metadata_json_is_stable_and_contains_no_arrays():
    metadata = json.loads(valid_motion().metadata_json())

    assert metadata == {
        "backend": "gvhmr",
        "backend_version": "1.0",
        "frames": 3,
        "joints": 24,
        "schema_version": 1,
        "source_fps": 50.0,
        "source_video_sha256": "a" * 64,
    }
