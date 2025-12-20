"""Robot Keyframe Kit - A generalizable Viser-based keyframe editor for MuJoCo robots."""

from .config import EditorConfig
from .editor import ViserKeyframeEditor
from .keyframe import Keyframe

__version__ = "0.1.0"
__all__ = ["ViserKeyframeEditor", "EditorConfig", "Keyframe"]






