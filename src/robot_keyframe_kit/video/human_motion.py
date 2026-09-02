"""Versioned, backend-neutral human motion archive."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class HumanMotionValidationError(ValueError):
    """Raised when a human-motion archive violates the v1 contract."""


@dataclass(frozen=True)
class HumanMotionV1:
    timestamps: np.ndarray
    joints_world: np.ndarray
    joint_confidence: np.ndarray
    root_position: np.ndarray
    root_quaternion_wxyz: np.ndarray
    joint_names: tuple[str, ...]
    source_fps: float
    source_video_sha256: str
    backend: str
    backend_version: str
    schema_version: int = 1

    _ARCHIVE_KEYS = frozenset(
        {
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
    )

    def validate(
        self,
        *,
        min_confidence: float = 0.25,
        max_low_confidence_fraction: float = 0.5,
    ) -> None:
        timestamps = np.asarray(self.timestamps)
        if timestamps.ndim != 1 or timestamps.size < 2:
            raise HumanMotionValidationError("timestamps must contain at least two frames")
        if not np.all(np.isfinite(timestamps)):
            raise HumanMotionValidationError("timestamps must be finite")
        if np.any(np.diff(timestamps) <= 0):
            raise HumanMotionValidationError("timestamps must be strictly increasing")

        frames = timestamps.size
        joints = np.asarray(self.joints_world)
        confidence = np.asarray(self.joint_confidence)
        if joints.ndim != 3 or joints.shape[0] != frames or joints.shape[2] != 3:
            raise HumanMotionValidationError("joints_world must have shape [T, J, 3]")
        if confidence.shape != joints.shape[:2]:
            raise HumanMotionValidationError("joint_confidence must have shape [T, J]")
        if len(self.joint_names) != joints.shape[1] or len(set(self.joint_names)) != len(self.joint_names):
            raise HumanMotionValidationError("joint_names must be unique and match J")

        root_position = np.asarray(self.root_position)
        quaternion = np.asarray(self.root_quaternion_wxyz)
        if root_position.shape != (frames, 3):
            raise HumanMotionValidationError("root_position must have shape [T, 3]")
        if quaternion.shape != (frames, 4):
            raise HumanMotionValidationError("root_quaternion_wxyz must have shape [T, 4]")
        for name, value in (
            ("joints_world", joints),
            ("joint_confidence", confidence),
            ("root_position", root_position),
            ("root_quaternion_wxyz", quaternion),
        ):
            if not np.all(np.isfinite(value)):
                raise HumanMotionValidationError(f"{name} must be finite")
        if np.any(confidence < 0.0) or np.any(confidence > 1.0):
            raise HumanMotionValidationError("joint_confidence must be within [0, 1]")
        norms = np.linalg.norm(quaternion.astype(np.float64), axis=1)
        if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-3):
            raise HumanMotionValidationError("root rotations must be unit quaternions")

        low_fraction = np.mean(confidence < min_confidence, axis=0)
        offenders = np.flatnonzero(low_fraction > max_low_confidence_fraction)
        if offenders.size:
            joint_name = self.joint_names[int(offenders[0])]
            raise HumanMotionValidationError(f"{joint_name} has a long low confidence run")

        if self.schema_version != 1:
            raise HumanMotionValidationError("schema_version must be 1")
        if not np.isfinite(self.source_fps) or self.source_fps <= 0:
            raise HumanMotionValidationError("source_fps must be positive and finite")
        digest = self.source_video_sha256.lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise HumanMotionValidationError("source_video_sha256 must be a lowercase SHA-256")
        if not self.backend or not self.backend_version:
            raise HumanMotionValidationError("backend and backend_version are required")

    def save(self, path: str | Path) -> None:
        self.validate()
        np.savez_compressed(
            path,
            timestamps=np.asarray(self.timestamps, dtype=np.float32),
            joints_world=np.asarray(self.joints_world, dtype=np.float32),
            joint_confidence=np.asarray(self.joint_confidence, dtype=np.float32),
            root_position=np.asarray(self.root_position, dtype=np.float32),
            root_quaternion_wxyz=np.asarray(self.root_quaternion_wxyz, dtype=np.float32),
            joint_names=np.asarray(self.joint_names, dtype=np.str_),
            source_fps=np.asarray([self.source_fps], dtype=np.float32),
            source_video_sha256=np.asarray([self.source_video_sha256], dtype=np.str_),
            backend=np.asarray([self.backend], dtype=np.str_),
            backend_version=np.asarray([self.backend_version], dtype=np.str_),
            schema_version=np.asarray([self.schema_version], dtype=np.int32),
        )

    @classmethod
    def load(cls, path: str | Path) -> "HumanMotionV1":
        with np.load(path, allow_pickle=False) as archive:
            actual = set(archive.files)
            if actual != cls._ARCHIVE_KEYS:
                missing = sorted(cls._ARCHIVE_KEYS - actual)
                extra = sorted(actual - cls._ARCHIVE_KEYS)
                raise HumanMotionValidationError(
                    f"archive keys do not match HumanMotionV1: missing={missing}, extra={extra}"
                )
            motion = cls(
                timestamps=np.asarray(archive["timestamps"], dtype=np.float32),
                joints_world=np.asarray(archive["joints_world"], dtype=np.float32),
                joint_confidence=np.asarray(archive["joint_confidence"], dtype=np.float32),
                root_position=np.asarray(archive["root_position"], dtype=np.float32),
                root_quaternion_wxyz=np.asarray(archive["root_quaternion_wxyz"], dtype=np.float32),
                joint_names=tuple(str(name) for name in archive["joint_names"]),
                source_fps=float(np.asarray(archive["source_fps"]).reshape(-1)[0]),
                source_video_sha256=str(np.asarray(archive["source_video_sha256"]).reshape(-1)[0]),
                backend=str(np.asarray(archive["backend"]).reshape(-1)[0]),
                backend_version=str(np.asarray(archive["backend_version"]).reshape(-1)[0]),
                schema_version=int(np.asarray(archive["schema_version"]).reshape(-1)[0]),
            )
        motion.validate()
        return motion

    def metadata_json(self) -> str:
        self.validate()
        return json.dumps(
            {
                "backend": self.backend,
                "backend_version": self.backend_version,
                "frames": int(np.asarray(self.timestamps).size),
                "joints": int(np.asarray(self.joints_world).shape[1]),
                "schema_version": self.schema_version,
                "source_fps": float(self.source_fps),
                "source_video_sha256": self.source_video_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
