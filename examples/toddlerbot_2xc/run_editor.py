#!/usr/bin/env python3
"""Example script to run the keyframe editor for Toddlerbot 2XC.

This demonstrates how to use the Python API to customize the editor
with robot-specific configuration.

Usage:
    python run_editor.py
    python run_editor.py --data keyframes/toddlerbot_2xc/my_motion.lz4
"""

from __future__ import annotations

import argparse
import os
import time

# Add the src directory to path for development
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from robot_keyframe_kit import EditorConfig, ViserKeyframeEditor


def main():
    parser = argparse.ArgumentParser(description="Toddlerbot 2XC Keyframe Editor")
    parser.add_argument(
        "--data",
        type=str,
        default="",
        help="Path to load existing keyframe data from (optional)",
    )
    args = parser.parse_args()

    # Get paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(script_dir, "scene.xml")
    config_path = os.path.join(script_dir, "toddlerbot_config.yaml")

    # Load robot-specific configuration
    print(f"Loading configuration from {config_path}")
    config = EditorConfig.from_yaml(config_path)

    # Create the editor
    print(f"Loading robot from {xml_path}")
    editor = ViserKeyframeEditor(
        xml_path,
        config=config,
        data_path=args.data,
    )

    # Get actual port from viser server
    try:
        port = editor.server.get_port()
    except Exception:
        port = 8080

    print(f"\n🚀 Toddlerbot Keyframe Editor running!")
    print(f"   Open http://localhost:{port} in your browser\n")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()

