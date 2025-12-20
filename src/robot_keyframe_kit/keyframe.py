"""Keyframe dataclass for storing robot poses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Keyframe:
    """A single keyframe representing a robot pose.
    
    Attributes:
        name: Human-readable name for this keyframe.
        motor_pos: Array of motor/actuator positions.
        joint_pos: Array of joint positions (may differ from motor_pos if there are transmissions).
        qpos: Full MuJoCo qpos array including base pose (for floating-base robots).
    """
    name: str
    motor_pos: np.ndarray
    joint_pos: Optional[np.ndarray] = None
    qpos: Optional[np.ndarray] = None






