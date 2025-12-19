"""Mathematical utilities for interpolation and signal processing.

Provides functions for trajectory interpolation used in keyframe animation.
"""

from __future__ import annotations

from typing import Union

import numpy as np
from numpy.typing import ArrayLike


def interpolate(
    p_start: Union[ArrayLike, float],
    p_end: Union[ArrayLike, float],
    duration: Union[ArrayLike, float],
    t: Union[ArrayLike, float],
    interp_type: str = "linear",
) -> Union[np.ndarray, float]:
    """Interpolate position at time t using specified interpolation type.

    Args:
        p_start: Initial position.
        p_end: Desired end position.
        duration: Total duration from start to end.
        t: Current time (within 0 to duration).
        interp_type: Type of interpolation ('linear', 'quadratic', 'cubic').

    Returns:
        Position at time t.
    """
    if t <= 0:
        return p_start

    if t >= duration:
        return p_end

    if interp_type == "linear":
        return p_start + (p_end - p_start) * (t / duration)
    elif interp_type == "quadratic":
        a = (-p_end + p_start) / duration**2
        b = (2 * p_end - 2 * p_start) / duration
        return a * t**2 + b * t + p_start
    elif interp_type == "cubic":
        a = (2 * p_start - 2 * p_end) / duration**3
        b = (3 * p_end - 3 * p_start) / duration**2
        return a * t**3 + b * t**2 + p_start
    else:
        raise ValueError("Unsupported interpolation type: {}".format(interp_type))


def binary_search(arr: ArrayLike, t: Union[ArrayLike, float]) -> int:
    """Performs a binary search on a sorted array to find the index of a target value.

    Args:
        arr: A sorted array of numbers.
        t: The target value to search for.

    Returns:
        The index of the target value if found; otherwise, the index of the
        largest element less than the target.
    """
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] < t:
            low = mid + 1
        elif arr[mid] > t:
            high = mid - 1
        else:
            return mid
    return low - 1


def interpolate_action(
    t: Union[ArrayLike, float],
    time_arr: ArrayLike,
    action_arr: ArrayLike,
    interp_type: str = "linear",
) -> np.ndarray:
    """Interpolates an action value at a given time using specified interpolation method.

    Args:
        t: The time at which to interpolate the action.
        time_arr: An array of time points corresponding to the action values.
        action_arr: An array of action values corresponding to the time points.
        interp_type: The type of interpolation to use. Defaults to "linear".

    Returns:
        The interpolated action value at time `t`.
    """
    if t <= time_arr[0]:
        return action_arr[0]
    elif t >= time_arr[-1]:
        return action_arr[-1]

    # Use binary search to find the segment containing current_time
    idx = binary_search(time_arr, t)
    idx = max(0, min(idx, len(time_arr) - 2))  # Ensure idx is within valid range

    p_start = action_arr[idx]
    p_end = action_arr[idx + 1]
    duration = time_arr[idx + 1] - time_arr[idx]
    return interpolate(p_start, p_end, duration, t - time_arr[idx], interp_type)



