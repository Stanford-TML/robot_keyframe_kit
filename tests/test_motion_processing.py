import numpy as np

from robot_keyframe_kit.video.motion_processing import (
    detect_squat_keyframes,
    finite_difference,
    resample_motion,
)


def test_detect_squat_keyframes_finds_ordered_semantic_phases():
    height = np.array([1.0, 1.0, 0.9, 0.7, 0.55, 0.55, 0.7, 0.9, 1.0, 1.0])

    keyframes = detect_squat_keyframes(height, max_keyframes=10)

    assert [item.name for item in keyframes] == [
        "standing",
        "descending",
        "bottom",
        "ascending",
        "stable",
    ]
    assert [item.frame for item in keyframes] == [0, 3, 4, 6, 9]


def test_resample_motion_uses_monotonic_50_hz_timeline():
    timestamps = np.array([0.0, 0.04, 0.08])
    values = np.array([[0.0], [2.0], [4.0]])

    time_50hz, values_50hz = resample_motion(timestamps, values, fps=50)

    np.testing.assert_allclose(time_50hz, [0.0, 0.02, 0.04, 0.06, 0.08])
    np.testing.assert_allclose(values_50hz[:, 0], [0.0, 1.0, 2.0, 3.0, 4.0])


def test_finite_difference_has_zero_endpoint_velocity_for_constant_signal():
    velocity = finite_difference(np.ones((4, 2)), dt=0.02)

    np.testing.assert_array_equal(velocity, np.zeros((4, 2)))
