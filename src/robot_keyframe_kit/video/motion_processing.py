"""Deterministic signal processing shared by video adapters and retargeters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SemanticKeyframe:
    name: str
    frame: int


def finite_difference(values: np.ndarray, *, dt: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 1 or values.shape[0] < 2:
        raise ValueError("values must contain at least two frames")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    return np.gradient(values, dt, axis=0, edge_order=1)


def resample_motion(
    timestamps: np.ndarray,
    values: np.ndarray,
    *,
    fps: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if timestamps.ndim != 1 or values.shape[0] != timestamps.size:
        raise ValueError("timestamps and values must share their frame dimension")
    if timestamps.size < 2 or np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    if fps <= 0:
        raise ValueError("fps must be positive")
    duration = float(timestamps[-1] - timestamps[0])
    frame_count = int(round(duration * fps)) + 1
    target = timestamps[0] + np.arange(frame_count, dtype=np.float64) / fps
    target[-1] = timestamps[-1]
    flat = values.reshape(values.shape[0], -1)
    sampled = np.column_stack(
        [np.interp(target, timestamps, flat[:, column]) for column in range(flat.shape[1])]
    )
    return target, sampled.reshape((frame_count,) + values.shape[1:])


def _closest_between(values: np.ndarray, start: int, stop: int, target: float) -> int:
    candidates = np.arange(start, stop + 1)
    return int(candidates[np.argmin(np.abs(values[candidates] - target))])


def detect_squat_keyframes(
    root_height: np.ndarray,
    *,
    max_keyframes: int = 15,
) -> tuple[SemanticKeyframe, ...]:
    """Find the five review phases of one standing-start squat cycle."""
    height = np.asarray(root_height, dtype=np.float64)
    if height.ndim != 1 or height.size < 5 or not np.all(np.isfinite(height)):
        raise ValueError("root_height must contain at least five finite frames")
    if max_keyframes < 5:
        raise ValueError("max_keyframes must allow the five semantic phases")
    bottom = int(np.argmin(height))
    if bottom == 0 or bottom >= height.size - 2:
        raise ValueError("motion must start and finish above an interior squat bottom")
    start = 0
    stable = height.size - 1
    descending_target = 0.5 * (height[start] + height[bottom])
    ascending_target = 0.5 * (height[bottom] + height[stable])
    descending = _closest_between(height, start + 1, bottom, descending_target)
    ascending = _closest_between(height, bottom + 1, stable - 1, ascending_target)
    return (
        SemanticKeyframe("standing", start),
        SemanticKeyframe("descending", descending),
        SemanticKeyframe("bottom", bottom),
        SemanticKeyframe("ascending", ascending),
        SemanticKeyframe("stable", stable),
    )
