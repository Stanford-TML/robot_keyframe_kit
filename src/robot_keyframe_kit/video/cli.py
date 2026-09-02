"""Command-line entry point for the video-to-keyframe research pipeline."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from .gvhmr_backend import GVHMRBackend, GVHMRBackendError
from .human_motion import HumanMotionV1
from .microduck import inspect_motion_file, retarget_microduck


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Video-to-robot keyframe research tools")
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract", help="recover one human motion from video")
    extract.add_argument("video", type=Path)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--backend", choices=("gvhmr",), default="gvhmr")
    extract.add_argument("--static-camera", action="store_true")
    extract.add_argument("--overlay-output", type=Path)
    retarget = commands.add_parser("retarget", help="retarget HumanMotionV1 to MicroDuck")
    retarget.add_argument("human_motion", type=Path)
    retarget.add_argument("xml", type=Path)
    retarget.add_argument("--config", type=Path)
    retarget.add_argument("--output", type=Path, required=True)
    inspect = commands.add_parser("inspect", help="write a physical preflight report")
    inspect.add_argument("motion", type=Path)
    inspect.add_argument("--xml", type=Path, required=True)
    inspect.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "extract":
            if not args.video.is_file():
                raise GVHMRBackendError(f"video does not exist: {args.video}")
            motion = GVHMRBackend().extract(
                args.video,
                static_camera=args.static_camera,
                overlay_output=args.overlay_output,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            motion.save(args.output)
            print(motion.metadata_json())
            return 0
        if args.command == "retarget":
            config_bytes = args.config.read_bytes() if args.config is not None else b""
            result = retarget_microduck(
                HumanMotionV1.load(args.human_motion),
                args.xml,
                output_path=args.output,
                config_bytes=config_bytes,
            )
            print(f"saved {args.output} ({len(result['time'])} frames)")
            return 0
        if args.command == "inspect":
            if not args.xml.is_file():
                raise ValueError(f"MuJoCo XML does not exist: {args.xml}")
            report = inspect_motion_file(args.motion, report_path=args.report)
            print(f"saved {args.report} ({report['frames']} frames)")
            return 0
    except (GVHMRBackendError, OSError, ValueError) as exc:
        print(f"keyframe-video failed: {exc}")
        return 2
    return 2
