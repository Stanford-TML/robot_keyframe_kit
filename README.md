# Robot Keyframe Kit

[![PyPI version](https://badge.fury.io/py/robot-keyframe-kit.svg)](https://pypi.org/project/robot-keyframe-kit/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A web-based keyframe editor for **any MuJoCo robot**. Design robot motions through an intuitive 3D interface—no robot-specific code required.

https://github.com/user-attachments/assets/demo-video-placeholder

## ✨ Features

- **Universal Compatibility** — Works with any MuJoCo XML model out of the box
- **Zero Configuration** — Auto-detects joints, actuators, end-effectors, and mirror pairs
- **Web-Based Interface** — 3D visualization powered by [Viser](https://github.com/nerfstudio-project/viser)
- **Mirror Mode** — Automatically synchronize left/right joint movements
- **Physics Simulation** — Test keyframes with full MuJoCo physics
- **Keyframe Sequencing** — Build timed motion sequences
- **Trajectory Recording** — Record and export motion data
- **YAML Configuration** — Optional per-robot config files for advanced customization

## 🚀 Installation

```bash
pip install robot-keyframe-kit
```

Or install from source:

```bash
git clone https://github.com/Stanford-TML/robot_keyframe_kit.git
cd robot_keyframe_kit
pip install -e .
```

## 📖 Quick Start

### Command Line

```bash
# Just provide your robot's MuJoCo XML file
keyframe-editor path/to/robot.xml

# With a custom name and save directory
keyframe-editor path/to/robot.xml --name my_robot --save-dir ./keyframes

# Using a YAML configuration file
keyframe-editor path/to/robot.xml --config robot_config.yaml

# Generate a config template for your robot
keyframe-editor path/to/robot.xml --generate-config
```

Then open **http://localhost:8081** in your browser.

### Python API

```python
from robot_keyframe_kit import ViserKeyframeEditor, EditorConfig

# Minimal usage — just provide the XML path
editor = ViserKeyframeEditor("path/to/robot.xml")

# With configuration
config = EditorConfig(
    name="my_robot",
    root_body="base_link",
    save_dir="my_keyframes",
)
editor = ViserKeyframeEditor("path/to/robot.xml", config=config)

# Keep the server running
import time
while True:
    time.sleep(1.0)
```

### Loading from YAML Config

```python
from robot_keyframe_kit import ViserKeyframeEditor, EditorConfig

# Load configuration from YAML
config = EditorConfig.from_yaml("robot_config.yaml")
editor = ViserKeyframeEditor("path/to/robot.xml", config=config)
```

## ⚙️ Configuration

### YAML Configuration File

Create a `robot_config.yaml` for robot-specific settings:

```yaml
name: my_robot
root_body: torso

# End-effector sites for pose tracking
end_effectors:
  - left_hand
  - right_hand
  - left_foot
  - right_foot

# Joint mirror pairs (left: right)
mirror_pairs:
  left_shoulder: right_shoulder
  left_elbow: right_elbow
  left_hip: right_hip
  left_knee: right_knee

# Mirror sign corrections (-1 to flip, 1 to keep same)
mirror_signs:
  left_shoulder: 1
  left_elbow: -1
  left_hip: 1
  left_knee: 1

# Output settings
save_dir: keyframes
```

### EditorConfig Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `name` | str | `"robot"` | Project name for saved files |
| `root_body` | str | auto-detect | Body used for ground alignment |
| `end_effector_sites` | list | auto-detect | Sites for end-effector tracking |
| `mirror_pairs` | dict | auto-detect | Left-to-right joint mapping |
| `mirror_signs` | dict | auto-detect | Sign corrections for mirroring |
| `save_dir` | str | `"keyframes"` | Directory for saved keyframes |
| `dt` | float | `0.02` | Trajectory timestep (50 Hz) |
| `n_frames` | int | `20` | Physics substeps per control step |
| `physics_dt` | float | `0.001` | Physics simulation timestep |
| `show_com` | bool | `True` | Show center of mass marker |
| `show_grid` | bool | `True` | Show ground grid |

## 🎮 UI Guide

The editor interface has three main sections:

### Left Panel — Keyframe Controls
- **Save Motion** — Export keyframes to compressed `.lz4` files
- **Keyframe List** — Select, add, remove, and reorder keyframes
- **Keyframe Operations** — Update, Test (with physics), Ground (place on floor)
- **Sequence Builder** — Create timed motion sequences

### Center Panel — Left-Side Joints
- Joint sliders for left-side actuators
- **End Effector Poses** — Save and restore end-effector positions

### Right Panel — Right-Side Joints & Settings
- Joint sliders for right-side actuators
- **Mirror Mode** — Sync left/right movements
- **Reverse Mirror** — Invert the mirror direction
- **Physics Toggle** — Enable/disable simulation
- **Visualization Options** — Grid, COM marker, etc.

### Camera Controls
- **Scroll** — Zoom in/out
- **Left-click + Drag** — Rotate view
- **Right-click + Drag** — Pan view

## 📁 Output Format

Keyframe data is saved as compressed `.lz4` files:

```python
{
    "keyframes": [
        {
            "name": "stand",
            "motor_pos": [...],      # Actuator positions
            "joint_pos": [...],      # Joint positions
            "qpos": [...],           # Full MuJoCo qpos
        },
        ...
    ],
    "timed_sequence": [
        ("stand", 0.0),
        ("crouch", 0.5),
        ("jump", 1.0),
    ],
    "time": [...],           # Trajectory timestamps
    "qpos": [...],           # Full qpos trajectory
    "body_pos": [...],       # Body positions over time
    "body_quat": [...],      # Body orientations over time
}
```

### Loading Saved Keyframes

```python
import lz4.frame
import pickle

with lz4.frame.open("keyframes/my_robot/motion.lz4", "rb") as f:
    data = pickle.load(f)

print(data["keyframes"][0]["name"])  # First keyframe name
print(data["timed_sequence"])        # Motion sequence
```

## 🤖 Tested Robots

Works with robots from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) and custom models:

- **Humanoids** — Unitree G1, H1, ToddlerBot, OP3
- **Quadrupeds** — Unitree A1, Go1, Boston Dynamics Spot
- **Arms** — Franka Panda, UR5, xArm
- **Hands** — Leap Hand, Shadow Hand
- **Custom Models** — Any valid MuJoCo XML

## 📋 Requirements

- Python ≥ 3.9
- MuJoCo ≥ 3.0
- Modern web browser (Chrome, Firefox, Safari, Edge)

## 🛠️ Troubleshooting

### Port Already in Use
```bash
# Use a different port
keyframe-editor robot.xml --port 8082
```

### Mirror Mode Not Working Correctly
Generate a config file and manually adjust `mirror_signs`:
```bash
keyframe-editor robot.xml --generate-config
# Edit the generated YAML, then:
keyframe-editor robot.xml --config robot_config.yaml
```

### Robot Floating in Air
The editor auto-detects `root_body` for grounding. If incorrect, specify it:
```bash
keyframe-editor robot.xml --root-body base_link
```

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions welcome! Please open an issue or pull request on GitHub.

## 📚 Citation

If you use this tool in your research, please cite:

```bibtex
@software{robot_keyframe_kit,
  title = {Robot Keyframe Kit},
  author = {Stanford TML},
  year = {2024},
  url = {https://github.com/Stanford-TML/robot_keyframe_kit}
}
```
