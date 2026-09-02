"""Adapter for a separately licensed GVHMR environment."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .human_motion import HumanMotionV1


SMPL_24_JOINT_NAMES = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
)


class GVHMRBackendError(RuntimeError):
    """GVHMR is missing or returned an ambiguous result."""


def _value(source: Any, name: str) -> Any:
    if isinstance(source, dict):
        return source.get(name)
    return getattr(source, name, None)


def motion_from_gvhmr_result(
    result: Any,
    *,
    source_video_sha256: str,
    backend_version: str,
) -> HumanMotionV1:
    joints = np.asarray(_value(result, "joints_world"), dtype=np.float32)
    if joints.ndim == 4:
        raise GVHMRBackendError("GVHMR input must contain exactly one person")
    if joints.ndim != 3 or joints.shape[1:] != (24, 3):
        raise GVHMRBackendError(
            f"GVHMR joints_world must have shape [T, 24, 3], got {joints.shape}"
        )
    params = _value(result, "smpl_params_world")
    if not isinstance(params, dict):
        raise GVHMRBackendError("GVHMR result is missing smpl_params_world")
    axis_angle = np.asarray(params.get("global_orient"), dtype=np.float64).reshape(-1, 3)
    if axis_angle.shape[0] != joints.shape[0]:
        raise GVHMRBackendError("global_orient frame count does not match joints_world")
    xyzw = Rotation.from_rotvec(axis_angle).as_quat()
    wxyz = np.column_stack((xyzw[:, 3], xyzw[:, :3])).astype(np.float32)
    translation = np.asarray(params.get("transl", joints[:, 0]), dtype=np.float32).reshape(-1, 3)
    if translation.shape[0] != joints.shape[0]:
        raise GVHMRBackendError("root translation frame count does not match joints_world")
    confidence_value = _value(result, "joint_confidence")
    if confidence_value is None:
        raise GVHMRBackendError(
            "GVHMR adapter requires per-frame joint_confidence to detect subject loss"
        )
    confidence = np.asarray(confidence_value, dtype=np.float32)
    fps_value = _value(result, "fps")
    fps = float(fps_value if fps_value is not None else 30.0)
    motion = HumanMotionV1(
        timestamps=np.arange(joints.shape[0], dtype=np.float32) / fps,
        joints_world=joints,
        joint_confidence=confidence,
        root_position=translation,
        root_quaternion_wxyz=wxyz,
        joint_names=SMPL_24_JOINT_NAMES,
        source_fps=fps,
        source_video_sha256=source_video_sha256,
        backend="gvhmr",
        backend_version=backend_version,
    )
    motion.validate()
    return motion


class GVHMRBackend:
    """Run GVHMR without making it a redistributable package dependency."""

    def extract(
        self,
        video: str | Path,
        *,
        static_camera: bool,
        overlay_output: str | Path | None = None,
    ) -> HumanMotionV1:
        try:
            import gvhmr
        except ImportError as exc:
            raise GVHMRBackendError(
                "GVHMR is not installed in this licensed research environment"
            ) from exc
        from .cli import file_sha256

        pipeline = gvhmr.pipeline("human-motion-recovery", device="cuda")
        result = pipeline(str(video), static_camera=static_camera, flip_test=True)
        if overlay_output is not None:
            result.render(str(overlay_output))
        try:
            version = importlib.metadata.version("gvhmr")
        except importlib.metadata.PackageNotFoundError:
            version = str(getattr(gvhmr, "__version__", "unknown"))
        return motion_from_gvhmr_result(
            result,
            source_video_sha256=file_sha256(video),
            backend_version=version,
        )
