import numpy as np

from robot_keyframe_kit.video.microduck import (
    MICRODUCK_JOINT_NAMES,
    build_squat_targets,
    inspect_motion_data,
)


def test_squat_targets_preserve_stand_endpoints_and_joint_limits():
    stand = np.array(
        [0.0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491,
         0.0, 0.0, 0.0, 0.0873, 0.4579, 0.0049, -0.4530]
    )
    squat = stand.copy()
    squat[[3, 12]] = [1.0, -1.0]
    height = np.array([1.0, 0.8, 0.5, 0.8, 1.0])
    lower = np.full(14, -1.2)
    upper = np.full(14, 1.2)

    targets, phase = build_squat_targets(height, stand, squat, lower, upper)

    np.testing.assert_allclose(targets[[0, -1]], np.stack([stand, stand]), atol=1e-6)
    np.testing.assert_allclose(targets[2], squat, atol=1e-6)
    np.testing.assert_allclose(phase, [0.0, 0.4, 1.0, 0.4, 0.0])
    assert targets.shape == (5, len(MICRODUCK_JOINT_NAMES))
    assert np.all(targets >= lower) and np.all(targets <= upper)


def test_squat_targets_remain_inside_limits_after_float32_serialization():
    upper_value = 1.5707963267948957
    stand = np.zeros(14)
    squat = np.full(14, upper_value)
    lower = np.full(14, -upper_value)
    upper = np.full(14, upper_value)

    targets, _ = build_squat_targets(
        np.array([1.0, 0.5, 1.0]), stand, squat, lower, upper
    )

    assert np.all(targets.astype(np.float32) <= upper)


def test_inspection_reports_joint_velocity_and_asymmetry_violations():
    motion = {
        "time": np.array([0.0, 0.02]),
        "action": np.zeros((2, 14)),
        "joint_vel": np.vstack([np.zeros(14), np.full(14, 20.0)]),
        "joint_limit_violation": np.array([False, True]),
        "left_foot_pos": np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]]),
        "right_foot_pos": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        "contact": np.ones((2, 2), dtype=bool),
        "root_height": np.array([0.12, 0.01]),
        "symmetry_error": np.array([0.0, 0.2]),
        "illegal_contact_count": np.array([0, 1]),
    }

    report = inspect_motion_data(motion, velocity_limit=10.0)

    assert report["joint_limit_violation_frames"] == [1]
    assert report["joint_velocity_violation_frames"] == [1]
    assert report["max_support_foot_slide_m"] == 0.02
    assert report["minimum_root_height_m"] == 0.01
    assert report["maximum_symmetry_error_rad"] == 0.2
    assert report["illegal_contact_frames"] == [1]
