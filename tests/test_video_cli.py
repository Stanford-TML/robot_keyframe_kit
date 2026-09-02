from pathlib import Path

import joblib
import numpy as np

from robot_keyframe_kit.video.cli import file_sha256, main


def test_file_sha256_reads_binary_video_without_normalizing_it(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video\r\nbytes\x00")

    assert file_sha256(video) == "0b7a54c474dc174125d9197a94221387c71a470b6d219db1fada3908b27f0301"


def test_inspect_command_writes_machine_readable_report(tmp_path: Path):
    motion = tmp_path / "motion.lz4"
    xml = tmp_path / "scene.xml"
    report = tmp_path / "report.json"
    xml.write_text("<mujoco/>")
    joblib.dump(
        {
            "time": np.array([0.0, 0.02]),
            "joint_vel": np.zeros((2, 14)),
            "joint_limit_violation": np.zeros(2, dtype=bool),
            "left_foot_pos": np.zeros((2, 3)),
            "right_foot_pos": np.zeros((2, 3)),
            "contact": np.ones((2, 2), dtype=bool),
            "root_height": np.array([0.12, 0.12]),
            "symmetry_error": np.zeros(2),
            "illegal_contact_count": np.zeros(2, dtype=int),
        },
        motion,
        compress="lz4",
    )

    status = main(["inspect", str(motion), "--xml", str(xml), "--report", str(report)])

    assert status == 0
    assert '"frames": 2' in report.read_text()
