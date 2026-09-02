"""MicroDuck-specific squat retargeting and physical preflight reporting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

from .human_motion import HumanMotionV1
from .motion_processing import detect_squat_keyframes, finite_difference, resample_motion


MICRODUCK_JOINT_NAMES = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)


class MicroduckRetargetError(ValueError):
    """The input motion or MuJoCo model cannot satisfy the MicroDuck contract."""


def build_squat_targets(
    root_height: np.ndarray,
    stand: np.ndarray,
    squat: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    height = np.asarray(root_height, dtype=np.float64)
    high = max(float(height[0]), float(height[-1]))
    low = float(np.min(height))
    if high - low < 1e-4:
        raise MicroduckRetargetError("video does not contain a measurable squat")
    phase = np.clip((high - height) / (high - low), 0.0, 1.0)
    targets = stand[None, :] + phase[:, None] * (squat - stand)[None, :]
    safe_lower = np.nextafter(lower.astype(np.float32), np.float32(np.inf)).astype(np.float64)
    safe_upper = np.nextafter(upper.astype(np.float32), np.float32(-np.inf)).astype(np.float64)
    return np.clip(targets, safe_lower, safe_upper), phase


def _named_keyframe(model: mujoco.MjModel, name: str) -> np.ndarray:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
    if key_id < 0:
        raise MicroduckRetargetError(f"MuJoCo scene must define the {name!r} keyframe")
    return model.key_qpos[key_id].copy()


def _joint_contract(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(1, model.njnt)
    )
    if names != MICRODUCK_JOINT_NAMES:
        raise MicroduckRetargetError(f"unexpected MicroDuck joint order: {names!r}")
    qpos_addresses = np.asarray(model.jnt_qposadr[1:], dtype=np.int32)
    return qpos_addresses, model.jnt_range[1:, 0].copy(), model.jnt_range[1:, 1].copy()


def _bounded_temporal_solve(
    targets: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    smoothing_weight: float = 0.03,
) -> np.ndarray:
    solved = np.empty_like(targets)
    previous = np.clip(targets[0], lower, upper)
    for frame, target in enumerate(targets):
        def residual(value: np.ndarray) -> np.ndarray:
            return np.concatenate(
                (value - target, np.sqrt(smoothing_weight) * (value - previous))
            )

        result = least_squares(residual, previous, bounds=(lower, upper), method="trf")
        solved[frame] = result.x
        previous = result.x
    return solved


def _limit_rate(values: np.ndarray, *, dt: float, max_velocity: float) -> np.ndarray:
    limited = values.copy()
    max_step = max_velocity * dt
    for frame in range(1, limited.shape[0]):
        delta = np.clip(limited[frame] - limited[frame - 1], -max_step, max_step)
        limited[frame] = limited[frame - 1] + delta
    return limited


def _body_id(model: mujoco.MjModel, name: str) -> int:
    value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if value < 0:
        raise MicroduckRetargetError(f"MicroDuck model is missing body {name!r}")
    return value


def retarget_microduck(
    motion: HumanMotionV1,
    xml_path: str | Path,
    *,
    output_path: str | Path | None = None,
    config_bytes: bytes = b"",
    max_velocity: float = 3.0 * np.pi,
) -> dict[str, Any]:
    motion.validate()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    qpos_addresses, lower, upper = _joint_contract(model)
    stand_qpos = _named_keyframe(model, "STAND")
    squat_qpos = _named_keyframe(model, "SIT")
    joint_index = {name: index for index, name in enumerate(motion.joint_names)}
    if "pelvis" not in joint_index:
        raise MicroduckRetargetError("human motion must contain the pelvis joint")
    source_height = motion.joints_world[:, joint_index["pelvis"], 2]
    time, height = resample_motion(motion.timestamps, source_height[:, None], fps=50)
    _, root_position = resample_motion(motion.timestamps, motion.root_position, fps=50)
    stand = stand_qpos[qpos_addresses]
    squat = squat_qpos[qpos_addresses]
    targets, phase = build_squat_targets(height[:, 0], stand, squat, lower, upper)
    if phase.size >= 7:
        window = min(11, phase.size if phase.size % 2 else phase.size - 1)
        phase = np.clip(savgol_filter(phase, window, 3), 0.0, 1.0)
        targets = stand[None, :] + phase[:, None] * (squat - stand)[None, :]
    actions = _bounded_temporal_solve(targets, lower, upper)
    actions = _limit_rate(actions, dt=0.02, max_velocity=max_velocity)
    safe_lower = np.nextafter(lower.astype(np.float32), np.float32(np.inf)).astype(np.float64)
    safe_upper = np.nextafter(upper.astype(np.float32), np.float32(-np.inf)).astype(np.float64)
    actions = np.clip(actions, safe_lower, safe_upper)
    joint_vel = finite_difference(actions, dt=0.02)

    qpos = np.tile(stand_qpos, (time.size, 1))
    qpos[:, qpos_addresses] = actions
    root_delta = root_position - root_position[0]
    qpos[:, :2] = root_delta[:, :2]
    qpos[:, 2] = stand_qpos[2] + phase * (squat_qpos[2] - stand_qpos[2])
    body_pos = np.empty((time.size, model.nbody - 1, 3), dtype=np.float32)
    body_quat = np.empty((time.size, model.nbody - 1, 4), dtype=np.float32)
    left_foot = np.empty((time.size, 3), dtype=np.float32)
    right_foot = np.empty((time.size, 3), dtype=np.float32)
    illegal = np.zeros(time.size, dtype=np.int32)
    left_id = _body_id(model, "ankle_left")
    right_id = _body_id(model, "ankle_right")
    allowed_floor_bodies = {left_id, right_id}
    for frame in range(time.size):
        data.qpos[:] = qpos[frame]
        mujoco.mj_forward(model, data)
        body_pos[frame] = data.xpos[1:]
        body_quat[frame] = data.xquat[1:]
        left_foot[frame] = data.xpos[left_id]
        right_foot[frame] = data.xpos[right_id]
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            bodies = {
                int(model.geom_bodyid[contact.geom1]),
                int(model.geom_bodyid[contact.geom2]),
            }
            robot_bodies = bodies - {0}
            if 0 in bodies and robot_bodies and not robot_bodies <= allowed_floor_bodies:
                illegal[frame] += 1
    body_lin_vel = finite_difference(body_pos, dt=0.02).astype(np.float32)
    body_ang_vel = np.zeros_like(body_lin_vel)
    contact = np.column_stack((left_foot[:, 2] <= 0.01, right_foot[:, 2] <= 0.01))
    symmetry = np.max(
        np.abs(
            np.column_stack(
                (
                    actions[:, 0] - actions[:, 9],
                    actions[:, 1] + actions[:, 10],
                    actions[:, 2] + actions[:, 11],
                    actions[:, 3] + actions[:, 12],
                    actions[:, 4] + actions[:, 13],
                )
            )
        ),
        axis=1,
    )
    semantic = detect_squat_keyframes(qpos[:, 2])
    keyframes = [
        {
            "name": item.name,
            "motor_pos": actions[item.frame].astype(np.float32),
            "joint_pos": actions[item.frame].astype(np.float32),
            "qpos": qpos[item.frame].astype(np.float64),
            "source_frame": int(item.frame),
        }
        for item in semantic
    ]
    result = {
        "time": time.astype(np.float64),
        "qpos": qpos.astype(np.float32),
        "action": actions.astype(np.float32),
        "joint_vel": joint_vel.astype(np.float32),
        "body_pos": body_pos,
        "body_quat": body_quat,
        "body_lin_vel": body_lin_vel,
        "body_ang_vel": body_ang_vel,
        "keyframes": keyframes,
        "timed_sequence": [(item.name, float(time[item.frame])) for item in semantic],
        "is_robot_relative_frame": False,
        "source_video_sha256": motion.source_video_sha256,
        "retarget_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "frame_error": np.linalg.norm(actions - targets, axis=1).astype(np.float32),
        "contact": contact,
        "left_foot_pos": left_foot,
        "right_foot_pos": right_foot,
        "root_height": qpos[:, 2].astype(np.float32),
        "symmetry_error": symmetry.astype(np.float32),
        "joint_limit_violation": np.any((actions < lower) | (actions > upper), axis=1),
        "illegal_contact_count": illegal,
        "joint_names": MICRODUCK_JOINT_NAMES,
    }
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(result, output_path, compress="lz4")
    return result


def inspect_motion_data(motion: dict[str, Any], *, velocity_limit: float = 3.0 * np.pi) -> dict[str, Any]:
    joint_vel = np.asarray(motion["joint_vel"])
    joint_limit = np.asarray(motion["joint_limit_violation"], dtype=bool)
    velocity_violation = np.any(np.abs(joint_vel) > velocity_limit, axis=1)
    contact = np.asarray(motion["contact"], dtype=bool)
    left = np.asarray(motion["left_foot_pos"], dtype=np.float64)
    right = np.asarray(motion["right_foot_pos"], dtype=np.float64)
    slide = []
    for positions, supported in ((left, contact[:, 0]), (right, contact[:, 1])):
        delta = np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1)
        active = supported[:-1] & supported[1:]
        slide.append(float(delta[active].sum()) if np.any(active) else 0.0)
    illegal = np.asarray(motion["illegal_contact_count"]) > 0
    return {
        "frames": int(np.asarray(motion["time"]).size),
        "joint_limit_violation_frames": np.flatnonzero(joint_limit).tolist(),
        "joint_velocity_violation_frames": np.flatnonzero(velocity_violation).tolist(),
        "max_support_foot_slide_m": float(max(slide)),
        "minimum_root_height_m": float(np.min(motion["root_height"])),
        "maximum_symmetry_error_rad": float(np.max(motion["symmetry_error"])),
        "illegal_contact_frames": np.flatnonzero(illegal).tolist(),
    }


def inspect_motion_file(path: str | Path, *, report_path: str | Path | None = None) -> dict[str, Any]:
    report = inspect_motion_data(joblib.load(path))
    if report_path is not None:
        Path(report_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
