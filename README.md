# robot-keyframe-kit

A generalizable MuJoCo keyframe editor for creating and editing robot motion sequences. Works with any MuJoCo-compatible robot model.

![](demo.gif)

## 🎬 Video Tutorial

[![Watch the Tutorial](screenshot_unitree.png)](https://www.youtube.com/watch?v=ZoRK3STKsd0)

**▶️ [Click to watch the full tutorial on YouTube](https://www.youtube.com/watch?v=ZoRK3STKsd0)**


## Installation

```bash
pip install robot-keyframe-kit
```

## Quick Start

### Using Scene XML (Recommended)

For best results, use a `scene.xml` file that includes your robot model along with proper floor setup.

```bash
keyframe-editor /path/to/scene.xml --name my_robot
```

Most robot models from [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) include a `scene.xml` file. For example:

```bash
keyframe-editor /path/to/mujoco_menagerie/unitree_g1/scene.xml --name g1
```

### Using Robot-Only XML

If you only have a robot XML file (without scene setup), the editor will automatically:

1. Detect that no floor plane exists
2. Check for a `scene.xml` in the same directory and suggest using it
3. Auto-generate a scene wrapper with floor plane for physics collision

```bash
keyframe-editor /path/to/robot.xml --name my_robot
```

**Note:** For better visualization and physics, create or use a proper `scene.xml`.

### Creating a Scene XML

If your robot model doesn't have a `scene.xml`, you can create one:

```xml
<mujoco model="my_robot_scene">
  <!-- Include your robot model -->
  <include file="robot.xml"/>

  <!-- Add floor and lighting -->
  <asset>
    <texture type="2d" name="groundplane" builtin="checker" 
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
             markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" 
              texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 3.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="10 10 0.05" material="groundplane"/>
  </worldbody>
</mujoco>
```

## Command-Line Options

```bash
keyframe-editor <xml_path> [OPTIONS]
```

### Required Arguments

- `xml_path`: Path to the MuJoCo XML file (scene.xml recommended, or robot.xml)

### Optional Arguments

- `--name <name>`: Name for this project (used in save filenames). Default: `robot`
- `--root-body <body_name>`: Name of the root body for ground alignment. Auto-detected if not specified.
- `--config <path>`: Path to YAML configuration file. Overrides auto-detection.
- `--generate-config <path>`: Generate a YAML configuration file from the model and exit.
- `--data <path>`: Path to load existing keyframe data from.
- `--save-dir <dir>`: Directory to save keyframe data. Default: `keyframes`
- `--no-auto-floor`: Disable automatic floor injection for robot-only XML files.

## Configuration Files

You can create a YAML configuration file to customize robot-specific settings:

```yaml
name: my_robot
root_body: base_link
end_effector_sites:
  - left_foot
  - right_foot
mirror_pairs:
  left_hip_pitch: right_hip_pitch
  left_knee: right_knee
mirror_signs:
  left_hip_pitch: -1
  left_knee: -1
dt: 0.02
save_dir: keyframes
scene:
  auto_inject_floor: true
  show_floor: true
```

Generate a default config from your model:

```bash
keyframe-editor robot.xml --generate-config config.yaml
```

## Features

- **Visual-Centric Joint Control**: Control motion joints intuitively
- **Automatic Mechanism Detection**: Handles differential drives, gear couplings, and parallel linkages
- **Mirror Mode**: Automatically mirrors joint movements with correct sign conventions
- **Physics Simulation**: Test keyframes and trajectories with full MuJoCo physics
- **Ground Placement**: Automatically places robot on ground based on lowest collision geometry
- **End-Effector Tracking**: Auto-detects and tracks end-effector sites/bodies


## Python API

You can also use the editor programmatically:

```python
from robot_keyframe_kit import ViserKeyframeEditor, EditorConfig

# Load config from file
config = EditorConfig.from_yaml("config.yaml")

# Or create config programmatically
config = EditorConfig(
    name="my_robot",
    root_body="base_link",
    auto_inject_floor=True,
    show_floor=True,
)

# Create editor
editor = ViserKeyframeEditor(
    "scene.xml",
    config=config,
)

# Editor runs until interrupted
```

## Save File Format

Motion data is saved as LZ4-compressed pickle files (`.lz4`) using `joblib`. Files are saved to `{save_dir}/{name}/{motion_name}.lz4`.

### Loading Save Files

```python
import joblib

data = joblib.load("keyframes/my_robot/walk.lz4")
```

### File Structure

| Key | Type | Description |
|-----|------|-------------|
| `keyframes` | `List[dict]` | List of keyframe dictionaries (see below) |
| `timed_sequence` | `List[Tuple[str, float]]` | Sequence of `(keyframe_name, duration_sec)` pairs |
| `time` | `ndarray (T,)` | Timestamps for each trajectory frame |
| `qpos` | `ndarray (T, nq)` | Full MuJoCo qpos at each frame |
| `motor_vel` | `ndarray (T, n_motors)` | Motor/actuator joint velocities (rad/s) |
| `joint_vel` | `ndarray (T, n_joints)` | UI-visible joint velocities (rad/s) |
| `action` | `ndarray (T, nu)` or `None` | Motor commands (if action trajectory was played) |
| `body_pos` | `ndarray (T, nbody, 3)` | Body positions (world or relative frame) |
| `body_quat` | `ndarray (T, nbody, 4)` | Body orientations as quaternions (w, x, y, z) |
| `body_lin_vel` | `ndarray (T, nbody, 3)` | Body linear velocities |
| `body_ang_vel` | `ndarray (T, nbody, 3)` | Body angular velocities |
| `site_pos` | `ndarray (T, n_sites, 3)` | End-effector site positions |
| `site_quat` | `ndarray (T, n_sites, 4)` | End-effector site orientations |
| `is_robot_relative_frame` | `bool` | Whether poses are in robot-relative frame |

### Keyframe Dictionary Structure

Each keyframe in the `keyframes` list contains:

| Key | Type | Description |
|-----|------|-------------|
| `name` | `str` | Human-readable keyframe name |
| `motor_pos` | `ndarray (nu,)` | Motor/actuator positions |
| `joint_pos` | `ndarray (nj,)` or `None` | Joint positions (may differ from motor_pos with transmissions) |
| `qpos` | `ndarray (nq,)` or `None` | Full MuJoCo qpos including base pose |

### Example Usage

```python
import joblib
import numpy as np

# Load motion data
data = joblib.load("keyframes/toddlerbot/wave.lz4")

# Get keyframes
for kf in data["keyframes"]:
    print(f"Keyframe: {kf['name']}, motor_pos shape: {kf['motor_pos'].shape}")

# Get trajectory
times = data["time"]  # (T,)
qpos = data["qpos"]   # (T, nq)
print(f"Trajectory: {len(times)} frames, {times[-1]:.2f}s duration")

# Get timed sequence for playback
for keyframe_name, duration in data["timed_sequence"]:
    print(f"  {keyframe_name}: {duration}s")
```

## Troubleshooting

### Robot Falls Through Floor

- **Check**: Does your XML have a floor plane? Use `scene.xml` if available.
- **Solution**: The editor auto-injects a floor, but collision filtering may need adjustment.
- **Best Fix**: Use a proper `scene.xml` with correct collision settings.

### Slow Physics Simulation

- **Check**: Model timestep settings (`model.opt.timestep`)
- **Solution**: Editor uses `n_frames=20` substeps per control step for stability.

### Wrong Joint Selection

- **Check**: Are you seeing motor joints instead of motion joints?
- **Solution**: The editor auto-detects differential drives and gear mechanisms. Check your config for manual overrides.

## License

MIT License
