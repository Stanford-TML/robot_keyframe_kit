"""Mathematical utilities for interpolation and signal processing.

Provides functions for trajectory interpolation used in keyframe animation.
"""

from __future__ import annotations

from typing import Optional, Union

import mujoco
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


def solve_equality_constraints(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    locked_joint_names: Optional[list[str]] = None,
    max_iter: int = 50,
    tol: float = 1e-6,
    alpha: float = 0.5,
) -> None:
    """Project qpos onto the equality-constraint manifold without physics.

    After ``mj_forward``, equality constraints (e.g. ``mjEQ_CONNECT`` for
    closed-loop linkages) may still be violated at the position level because
    ``mj_forward`` only computes what constraint *forces* would be needed — it
    does not move ``qpos``.

    This function iteratively computes a minimum-norm joint correction from the
    constraint Jacobian (``efc_J``) and position error (``efc_pos``), then
    integrates via ``mj_integratePos`` (quaternion-safe).  No ``mj_step`` is
    performed, so gravity never acts and the robot cannot fall.

    Args:
        model: MuJoCo model.
        data: MuJoCo data (modified in-place).
        locked_joint_names: Joint names whose DOFs must not be adjusted.
            Typically the user-facing slider joints so the solver only
            adjusts internal mechanism joints (motor, rod).
        max_iter: Maximum Newton iterations.
        tol: Convergence tolerance on max absolute position error.
        alpha: Damped Newton step size.
    """
    if model.neq == 0:
        return

    # Build set of DOF indices to lock.
    locked_dofs: list[int] = []

    # Always lock free-joint DOFs (6 per free joint) so the root doesn't move.
    for jnt_id in range(model.njnt):
        if model.jnt_type[jnt_id] == mujoco.mjtJoint.mjJNT_FREE:
            dof_adr = model.jnt_dofadr[jnt_id]
            locked_dofs.extend(range(dof_adr, dof_adr + 6))

    # Lock user-specified joints (slider joints the user is actively controlling).
    if locked_joint_names:
        for name in locked_joint_names:
            jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jnt_id >= 0:
                dof_adr = model.jnt_dofadr[jnt_id]
                jnt_type = model.jnt_type[jnt_id]
                if jnt_type == mujoco.mjtJoint.mjJNT_BALL:
                    locked_dofs.extend(range(dof_adr, dof_adr + 3))
                elif jnt_type in (
                    mujoco.mjtJoint.mjJNT_SLIDE,
                    mujoco.mjtJoint.mjJNT_HINGE,
                ):
                    locked_dofs.append(dof_adr)

    for _ in range(max_iter):
        nefc = data.nefc
        if nefc == 0:
            break

        # Identify equality constraint rows (mjCNSTR_EQUALITY == 0).
        efc_type = data.efc_type[:nefc]
        eq_mask = efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY

        if not np.any(eq_mask):
            break

        pos_err = data.efc_pos[:nefc][eq_mask]

        if np.max(np.abs(pos_err)) < tol:
            break

        J = data.efc_J.reshape(nefc, model.nv)[eq_mask].copy()

        # Zero out locked columns so the solver cannot move those DOFs.
        if locked_dofs:
            J[:, locked_dofs] = 0.0

        # Minimum-norm correction: dq = J^+ @ (-alpha * pos_err)
        dq, _, _, _ = np.linalg.lstsq(J, -alpha * pos_err, rcond=None)

        # Quaternion-safe position integration
        mujoco.mj_integratePos(model, data.qpos, dq, 1.0)

        # Recompute kinematics for next iteration
        mujoco.mj_forward(model, data)


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
