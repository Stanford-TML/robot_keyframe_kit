"""Main ViserKeyframeEditor class for the robot keyframe editor.

This module provides a web-based keyframe editor using Viser for visualization
and MuJoCo for physics simulation. It works directly with MuJoCo XML files
without requiring any robot-specific wrapper classes.

NOTE: This file is adapted from the toddlerbot implementation.
The sim/robot parameters have been replaced with direct MuJoCo usage.
"""

from __future__ import annotations

import argparse
import os
import shutil
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict
from typing import Callable, Dict, List, Optional, Tuple

import joblib
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    import viser
    from viser import GuiEvent
except Exception as exc:
    raise ImportError(
        "Viser is required for the keyframe editor.\n"
        "Install with: pip install viser\n"
        f"Original import error: {exc}"
    ) from exc

try:
    import trimesh
except ImportError:
    trimesh = None

from .config import EditorConfig
from .keyframe import Keyframe
from .math_utils import interpolate_action
from .sim_worker import SimWorker


class ViserKeyframeEditor:
    """Controller that manages the Viser server and MuJoCo geometry.

    This class provides a web-based keyframe editor that works directly with
    MuJoCo XML files. Joint names, actuator names, and limits are automatically
    discovered from the model.
    
    Example:
        >>> from robot_keyframe_kit import ViserKeyframeEditor, EditorConfig
        >>> editor = ViserKeyframeEditor("path/to/robot.xml")
        >>> # Open browser to http://localhost:8080
    """

    def __init__(
        self,
        xml_path: str,
        config: Optional[EditorConfig] = None,
        *,
        data_path: str = "",
    ) -> None:
        """Initialize the keyframe editor.
        
        Args:
            xml_path: Path to the MuJoCo XML scene file.
            config: Optional configuration object. Uses defaults if not provided.
            data_path: Optional path to load existing keyframe data from.
        """
        if config is None:
            config = EditorConfig()
        
        self.config = config
        self.xml_path = os.path.abspath(xml_path)
        self.dt = config.dt
        
        # Load MuJoCo model directly
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Auto-detect root body if not specified
        if config.root_body is None:
            config.root_body = self._auto_detect_root_body()
            print(f"[Viser] Auto-detected root body: {config.root_body}", flush=True)
        
        # Auto-discover joints and actuators from model
        self.joint_names: List[str] = []
        self.actuator_names: List[str] = []
        self.joint_limits: Dict[str, Tuple[float, float]] = {}
        self.default_positions: Dict[str, float] = {}
        
        # Maps motion joint -> tendon info for inverse computation
        # Structure: motion_joint -> (tendon_name, [(motor_joint, motor_coef), ...], motion_coef)
        self.differential_drives: Dict[str, Tuple[str, List[Tuple[str, float]], float]] = {}
        
        # Initialize coupling dictionaries
        self.joint_couplings: Dict[str, Tuple[str, float, float]] = {}
        self.joint_couplings_inverse: Dict[str, Tuple[str, float, float]] = {}
        
        # Maps motor_joint -> (motion_joint, approximate_ratio) for parallel linkages
        self.parallel_linkages: Dict[str, Tuple[str, float]] = {}
        # Inverse mapping: motion_joint -> (motor_joint, approximate_ratio)
        self.parallel_linkages_inverse: Dict[str, Tuple[str, float]] = {}
        
        self._discover_joint_couplings()  # Must discover couplings first to know which are drive joints
        self._discover_parallel_linkages()  # Discover parallel linkages before joint discovery
        self._discover_joints_and_actuators()
        self._discover_differential_drives()
        
        # Try to get home pose from model keyframe or use qpos0
        try:
            self.home_qpos = np.array(
                self.model.keyframe("home").qpos, dtype=np.float32
            )
        except (KeyError, Exception):
            self.home_qpos = self.model.qpos0.copy()
        
        # Set up save directory: {save_dir}/{name}/
        # Files will be saved as {motion_name}_{timestamp}.lz4
        self.result_dir = os.path.join(config.save_dir, config.name)
        os.makedirs(self.result_dir, exist_ok=True)
        self.data_path = data_path
        
        # Mirror configuration - can be customized via config, then auto-computed
        self.mirror_joint_signs = config.mirror_signs.copy() if config.mirror_signs else {}
        self._compute_mirror_signs()  # Auto-compute based on joint axis orientations
        
        # State
        self.keyframes: List[Keyframe] = []
        self.sequence_list: List[Tuple[str, float]] = []
        self.selected_keyframe: Optional[int] = None
        self.selected_sequence: Optional[int] = None
        self.traj_times: List[float] = []
        self.action_traj: Optional[List[np.ndarray]] = None
        self.is_qpos_traj = False
        self.is_relative_frame = True
        self.qpos_replay: List[np.ndarray] = []
        self.body_pos_replay: List[np.ndarray] = []
        self.body_quat_replay: List[np.ndarray] = []
        self.body_lin_vel_replay: List[np.ndarray] = []
        self.body_ang_vel_replay: List[np.ndarray] = []
        self.site_pos_replay: List[np.ndarray] = []
        self.site_quat_replay: List[np.ndarray] = []

        self.saved_ee_poses: Dict[str, np.ndarray] = {}

        # Scene bookkeeping
        self._geom_handles: Dict[int, object] = {}
        self._geom_groups: Dict[int, int] = {}
        self._geom_base_rgba: Dict[int, Tuple[float, float, float, float]] = {}
        self._scene_handles: Dict[str, object] = {}
        self._mesh_file_map: Dict[str, str] = {}
        self._mesh_scale_map: Dict[str, Tuple[float, float, float]] = {}
        self._mesh_quat_map: Dict[str, Tuple[float, float, float, float]] = {}
        self._com_sphere: Optional[object] = None
        self._scene_updater: Optional[threading.Thread] = None

        # GUI bookkeeping
        self.slider_widgets: Dict[str, viser.GuiSliderHandle] = {}
        self.collision_geom_checked: Optional[viser.GuiCheckboxHandle] = None
        self.show_all_geoms: Optional[viser.GuiCheckboxHandle] = None
        self.motion_name_input: Optional[viser.GuiTextHandle] = None
        self.keyframes_summary: Optional[viser.GuiHtmlHandle] = None
        self.keyframe_index_input: Optional[viser.GuiTextHandle] = None
        self.keyframe_name_input: Optional[viser.GuiTextHandle] = None
        self.sequence_summary: Optional[viser.GuiHtmlHandle] = None
        self.sequence_index_input: Optional[viser.GuiTextHandle] = None
        self.sequence_time_input: Optional[viser.GuiTextHandle] = None
        self.mirror_checked: Optional[viser.GuiCheckboxHandle] = None
        self.rev_mirror_checked: Optional[viser.GuiCheckboxHandle] = None
        self.physics_enabled: Optional[viser.GuiCheckboxHandle] = None
        self.relative_frame_checked: Optional[viser.GuiCheckboxHandle] = None

        self._updating_handles: set[int] = set()
        self.normalized_range = (-2000.0, 2000.0)

        # Lock shared between geometry updates and worker callbacks
        self.worker_lock = threading.Lock()

        # The Viser server hosts the UI + WebGL viewport
        self.server = viser.ViserServer(label="Keyframe Editor")
        try:
            self.server.gui.configure_theme(
                control_layout="fixed",
                control_width="large",
            )
        except Exception as exc:
            print(f"[Viser] configure_theme failed: {exc}", flush=True)
        
        if config.show_grid:
            self.server.scene.add_grid("/grid", width=20, height=20, infinite_grid=True)

        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            """Apply a sensible default orbit camera when a client connects."""
            camera_pos = (1.5, -0.5, 0.55)
            look_at_pos = (0.0, 0.0, 0.25)
            try:
                client.camera.position = camera_pos
                client.camera.look_at = look_at_pos
                if hasattr(client.camera, "up"):
                    client.camera.up = (0.0, 0.0, 1.0)
                if hasattr(client.camera, "vertical_fov"):
                    client.camera.vertical_fov = 55.0
            except Exception:
                pass

        try:
            mujoco.mj_forward(self.model, self.data)
        except Exception:
            pass

        self._build_ui()

        # Background worker for physics interactions
        self.worker = SimWorker(
            self.model,
            self.data,
            self.config,
            self.worker_lock,
            joint_names=self.joint_names,
            actuator_names=self.actuator_names,
            default_joint_angles=self.default_positions,
            on_state=self._on_state,
            on_traj=self._on_traj,
        )
        self.worker.start()

        self._build_robot_meshes()

        if not self._geom_handles:
            try:
                self._scene_handles["torso_frame"] = self.server.scene.add_frame(
                    "/robot/root",
                    wxyz=(1.0, 0.0, 0.0, 0.0),
                    position=(0.0, 0.0, 0.0),
                )
                for body_index in range(self.model.nbody):
                    name = (
                        mujoco.mj_id2name(
                            self.model, mujoco.mjtObj.mjOBJ_BODY, body_index
                        )
                        or f"body_{body_index}"
                    )
                    try:
                        self._scene_handles[f"body_{body_index}"] = (
                            self.server.scene.add_icosphere(
                                f"/robot/bodies/{body_index:04d}_{name}",
                                radius=0.02,
                                position=(0.0, 0.0, 0.0),
                                color=(0.7, 0.7, 0.7),
                            )
                        )
                    except Exception:
                        continue
            except Exception:
                self._scene_handles.clear()

        self._apply_geom_visibility()

        if config.show_com:
            try:
                self._com_sphere = self.server.scene.add_icosphere(
                    "/robot/com",
                    radius=0.03,
                    position=(0.0, 0.0, 0.0),
                    color=(1.0, 0.0, 0.0),
                )
                print("[Viser] Center of mass sphere added (red ball)", flush=True)
            except Exception as exc:
                print(f"[Viser] Failed to add CoM sphere: {exc}", flush=True)
                self._com_sphere = None

        self._start_scene_updater()
        data_loaded = self._load_data()
        
        # Auto-ground only if no data was loaded (fresh start)
        # When loading saved data, keep the saved positions
        if not data_loaded:
            self.worker.request_on_ground()
            # Wait for grounding to complete and update the default keyframe
            self._update_default_keyframe_after_ground()

    def _discover_joints_and_actuators(self) -> None:
        """Auto-discover joints and actuators from the MuJoCo model.
        
        Only discovers actuated joints (joints that have motors attached).
        This prevents creating sliders for passive joints in closed-loop mechanisms
        that cannot be controlled independently.
        """
        # First, find which joints have actuators attached
        actuated_joint_ids = set()
        for i in range(self.model.nu):
            trnid = self.model.actuator_trnid[i]
            if trnid[0] >= 0:
                actuated_joint_ids.add(trnid[0])
        
        # Track which motor joints are part of differential drives (to be replaced)
        motor_joints_in_differential = set()
        
        # First pass: discover differential drives to identify motor joints to replace
        for tendon_id in range(self.model.ntendon):
            adr = self.model.tendon_adr[tendon_id]
            num = self.model.tendon_num[tendon_id]
            
            joints_in_tendon = []
            for i in range(num):
                wrap_idx = adr + i
                obj_id = self.model.wrap_objid[wrap_idx]
                coef = float(self.model.wrap_prm[wrap_idx])
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, obj_id)
                if joint_name:
                    joints_in_tendon.append((joint_name, coef, obj_id))
            
            if len(joints_in_tendon) < 2:
                continue
            
            motor_joints = [(n, c, jid) for n, c, jid in joints_in_tendon if jid in actuated_joint_ids]
            motion_joints = [(n, c, jid) for n, c, jid in joints_in_tendon if jid not in actuated_joint_ids]
            
            # Differential drive: multiple motors -> single motion
            if len(motor_joints) >= 2 and len(motion_joints) == 1:
                for motor_name, _, motor_jid in motor_joints:
                    motor_joints_in_differential.add(motor_name)
        
        # Discover joints: use motion joints for differential drives, motor joints otherwise
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name is None:
                continue
            jnt_type = self.model.jnt_type[i]
            # Skip free joints (type 0)
            if jnt_type == 0:
                continue
            
            # For differential drives: skip motor joints, add motion joints instead
            if name in motor_joints_in_differential:
                continue  # Skip motor joint, will add motion joint below
            
            # For gear mechanisms: skip drive joints, will add driven joints below
            # Check if this is a drive joint that has a driven partner
            if name in self.joint_couplings:
                continue  # Skip drive joint, will add driven joint below
            
            # For parallel linkages: skip motor joints, will add motion joints below
            if name in self.parallel_linkages:
                continue  # Skip motor joint
            
            # Add actuated joints (motors) that are not drive joints
            if i in actuated_joint_ids:
                self.joint_names.append(name)
                # Get joint limits
                if self.model.jnt_limited[i]:
                    low = float(self.model.jnt_range[i, 0])
                    high = float(self.model.jnt_range[i, 1])
                    self.joint_limits[name] = (low, high)
                else:
                    self.joint_limits[name] = (-3.14159, 3.14159)
                # Get default position from qpos0
                qpos_adr = self.model.jnt_qposadr[i]
                self.default_positions[name] = float(self.model.qpos0[qpos_adr])
        
        # Second pass: add motion joints for differential drives
        for tendon_id in range(self.model.ntendon):
            adr = self.model.tendon_adr[tendon_id]
            num = self.model.tendon_num[tendon_id]
            
            joints_in_tendon = []
            for i in range(num):
                wrap_idx = adr + i
                obj_id = self.model.wrap_objid[wrap_idx]
                coef = float(self.model.wrap_prm[wrap_idx])
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, obj_id)
                if joint_name:
                    joints_in_tendon.append((joint_name, coef, obj_id))
            
            motor_joints = [(n, c, jid) for n, c, jid in joints_in_tendon if jid in actuated_joint_ids]
            motion_joints = [(n, c, jid) for n, c, jid in joints_in_tendon if jid not in actuated_joint_ids]
            
            # Add motion joints for differential drives
            if len(motor_joints) >= 2 and len(motion_joints) == 1:
                motion_name, motion_coef, motion_jid = motion_joints[0]
                if motion_name not in self.joint_names:
                    self.joint_names.append(motion_name)
                    # Get joint limits from the motion joint
                    if self.model.jnt_limited[motion_jid]:
                        low = float(self.model.jnt_range[motion_jid, 0])
                        high = float(self.model.jnt_range[motion_jid, 1])
                        self.joint_limits[motion_name] = (low, high)
                    else:
                        self.joint_limits[motion_name] = (-3.14159, 3.14159)
                    # Get default position
                    qpos_adr = self.model.jnt_qposadr[motion_jid]
                    self.default_positions[motion_name] = float(self.model.qpos0[qpos_adr])
        
        # Third pass: add driven joints for gear mechanisms (replace drive joints)
        for driven_name, (drive_name, offset, ratio) in self.joint_couplings_inverse.items():
            if driven_name not in self.joint_names:
                driven_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, driven_name)
                if driven_id >= 0:
                    self.joint_names.append(driven_name)
                    # Get joint limits from the driven joint
                    if self.model.jnt_limited[driven_id]:
                        low = float(self.model.jnt_range[driven_id, 0])
                        high = float(self.model.jnt_range[driven_id, 1])
                        self.joint_limits[driven_name] = (low, high)
                    else:
                        self.joint_limits[driven_name] = (-3.14159, 3.14159)
                    # Get default position
                    qpos_adr = self.model.jnt_qposadr[driven_id]
                    self.default_positions[driven_name] = float(self.model.qpos0[qpos_adr])
        
        # Fourth pass: add motion joints for parallel linkages (replace motor joints)
        for motion_name, (motor_name, ratio) in self.parallel_linkages_inverse.items():
            if motion_name not in self.joint_names:
                motion_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, motion_name)
                if motion_id >= 0:
                    self.joint_names.append(motion_name)
                    # Get joint limits from the motion joint
                    if self.model.jnt_limited[motion_id]:
                        low = float(self.model.jnt_range[motion_id, 0])
                        high = float(self.model.jnt_range[motion_id, 1])
                        self.joint_limits[motion_name] = (low, high)
                    else:
                        self.joint_limits[motion_name] = (-3.14159, 3.14159)
                    # Get default position
                    qpos_adr = self.model.jnt_qposadr[motion_id]
                    self.default_positions[motion_name] = float(self.model.qpos0[qpos_adr])
        
        # Discover actuators
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                self.actuator_names.append(name)
        
        print(f"[Viser] Discovered {len(self.joint_names)} actuated joints, {len(self.actuator_names)} actuators", flush=True)

    def _discover_joint_couplings(self) -> None:
        """Discover joint equality constraints (gear couplings) from the model.
        
        For robots with gear mechanisms, moving a 'drive' joint should also move
        its coupled 'driven' joint according to the gear ratio. This method builds
        a mapping from drive joints to their driven joints with the gear ratio.
        
        The constraint is: driven = polycoef[0] + polycoef[1] * drive
        """
        # Maps drive_joint -> (driven_joint, offset, ratio) for forward computation
        # Maps driven_joint -> (drive_joint, offset, ratio) for inverse computation
        self.joint_couplings: Dict[str, Tuple[str, float, float]] = {}
        self.joint_couplings_inverse: Dict[str, Tuple[str, float, float]] = {}
        
        for eq_id in range(self.model.neq):
            if self.model.eq_type[eq_id] == mujoco.mjtEq.mjEQ_JOINT:
                # For JOINT constraints, obj1id is the driven joint, obj2id is the drive joint
                driven_id = self.model.eq_obj1id[eq_id]
                drive_id = self.model.eq_obj2id[eq_id]
                
                if drive_id < 0:
                    continue
                
                driven_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, driven_id)
                drive_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, drive_id)
                
                if driven_name and drive_name:
                    # eq_data: [polycoef[0], polycoef[1], ...] where driven = polycoef[0] + polycoef[1] * drive
                    eq_data = self.model.eq_data[eq_id]
                    offset = float(eq_data[0])
                    ratio = float(eq_data[1])
                    
                    # Forward: drive -> driven
                    self.joint_couplings[drive_name] = (driven_name, offset, ratio)
                    # Inverse: driven -> drive (drive = (driven - offset) / ratio)
                    if abs(ratio) > 1e-6:  # Avoid division by zero
                        self.joint_couplings_inverse[driven_name] = (drive_name, offset, ratio)
        
        if self.joint_couplings:
            print(f"[Viser] Discovered {len(self.joint_couplings)} joint couplings (gear constraints)", flush=True)

    def _discover_differential_drives(self) -> None:
        """Discover differential drive mechanisms (tendon couplings).
        
        For differential drives like the waist mechanism:
        - Multiple motor joints (waist_act_1, waist_act_2) control
        - Single motion DOF (waist_yaw, waist_roll)
        
        This method identifies these patterns and stores the inverse mapping
        so users can control the intuitive motion joints instead of motor joints.
        """
        # First, find which joints have actuators
        actuated_joint_ids = set()
        for i in range(self.model.nu):
            trnid = self.model.actuator_trnid[i]
            if trnid[0] >= 0:
                actuated_joint_ids.add(trnid[0])
        
        # Analyze each tendon to find differential drives
        for tendon_id in range(self.model.ntendon):
            tendon_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_TENDON, tendon_id)
            if tendon_name is None:
                continue
            
            adr = self.model.tendon_adr[tendon_id]
            num = self.model.tendon_num[tendon_id]
            
            # Collect all joints in this tendon with their coefficients
            joints_in_tendon = []
            for i in range(num):
                wrap_idx = adr + i
                wrap_type = self.model.wrap_type[wrap_idx]
                obj_id = self.model.wrap_objid[wrap_idx]
                coef = float(self.model.wrap_prm[wrap_idx])
                
                # For fixed tendons, obj_id refers to joint ID (even if wrap_type is pulley)
                joint_id = obj_id
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                
                if joint_name:
                    joints_in_tendon.append((joint_name, coef, joint_id))
            
            if len(joints_in_tendon) < 2:
                continue
            
            # Separate into actuated (motor) and passive (motion) joints
            motor_joints = []
            motion_joints = []
            
            for joint_name, coef, joint_id in joints_in_tendon:
                if joint_id in actuated_joint_ids:
                    motor_joints.append((joint_name, coef))
                else:
                    motion_joints.append((joint_name, coef))
            
            # If we have multiple motors controlling a single motion DOF, it's a differential drive
            if len(motor_joints) >= 2 and len(motion_joints) == 1:
                motion_joint, motion_coef = motion_joints[0]
                motor_list = [(mname, mcoef) for mname, mcoef in motor_joints]
                self.differential_drives[motion_joint] = (tendon_name, motor_list, motion_coef)
        
        if self.differential_drives:
            print(f"[Viser] Discovered {len(self.differential_drives)} differential drive motion joints", flush=True)

    def _discover_parallel_linkages(self) -> None:
        """Discover parallel linkage mechanisms.
        
        For mechanisms like neck_pitch:
        - motor joint (neck_pitch_act) is on a child body (neck_pitch_plate)
        - passive joint (neck_pitch) is on the parent body (head)
        - CONNECT constraints create a closed loop
        
        We want to show the passive joint (intuitive) instead of the motor joint.
        """
        # Find actuated joints
        actuated_joint_ids = set()
        for act_id in range(self.model.nu):
            trnid = self.model.actuator_trnid[act_id]
            if trnid[0] >= 0:
                actuated_joint_ids.add(trnid[0])
        
        # Check for CONNECT constraints (indicate closed-loop mechanisms)
        has_connect = any(
            self.model.eq_type[eq_id] == mujoco.mjtEq.mjEQ_CONNECT
            for eq_id in range(self.model.neq)
        )
        
        if not has_connect:
            return
        
        # Maps motor_joint -> (motion_joint, approximate_ratio)
        self.parallel_linkages: Dict[str, Tuple[str, float]] = {}
        
        # For each actuated joint, check if its body's parent has a passive joint
        for jnt_id in range(self.model.njnt):
            if jnt_id not in actuated_joint_ids:
                continue
            
            motor_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
            motor_body_id = self.model.jnt_bodyid[jnt_id]
            parent_body_id = self.model.body_parentid[motor_body_id]
            
            if parent_body_id <= 0:  # Skip if parent is world
                continue
            
            # Find passive joints on the parent body
            for other_jnt_id in range(self.model.njnt):
                if other_jnt_id in actuated_joint_ids:
                    continue
                if self.model.jnt_bodyid[other_jnt_id] != parent_body_id:
                    continue
                
                motion_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, other_jnt_id)
                
                # Check if they have similar names (e.g., neck_pitch_act and neck_pitch)
                # Motor name should contain motion name or vice versa
                motor_base = motor_name.replace('_act', '').replace('_drive', '').replace('_motor', '')
                motion_base = motion_name
                
                if motor_base == motion_base or motion_base in motor_name:
                    # Found a parallel linkage pair!
                    # Approximate ratio: typically close to -1.0 for parallel linkages
                    self.parallel_linkages[motor_name] = (motion_name, -1.0)
                    self.parallel_linkages_inverse[motion_name] = (motor_name, -1.0)
        
        if self.parallel_linkages:
            print(f"[Viser] Discovered {len(self.parallel_linkages)} parallel linkage mechanisms", flush=True)
            for motor, (motion, ratio) in self.parallel_linkages.items():
                print(f"  {motor} -> {motion} (approx ratio: {ratio:.2f})", flush=True)

    def _compute_mirror_signs(self) -> None:
        """Compute mirror signs based on joint axis orientations.
        
        For geometric mirroring across the sagittal (left-right) plane:
        
        The Y-axis is perpendicular to the sagittal plane, so Y-axis rotations
        produce symmetric motion with the SAME angle values.
        
        X and Z axes are parallel to the sagittal plane, so rotations around
        them require OPPOSITE angles for visual mirror symmetry.
        
        Additionally, if left/right axes point in opposite directions, the
        axis flip already accounts for part of the mirroring.
        
        Rules:
        - Same axes, Y-axis: sign = +1 (same angles for mirror)
        - Same axes, X/Z-axis: sign = -1 (opposite angles for mirror)
        - Opposite axes, X-axis (roll): sign = +1 (axis flip + same angle = mirror)
        - Opposite axes, Y/Z-axis: sign = -1 (opposite angles for mirror)
        """
        # Common left/right naming patterns for partner detection
        patterns = [
            ("left", "right"),
            ("_l_", "_r_"),
            ("_l", "_r"),
            ("l_", "r_"),
            ("_left_", "_right_"),
            ("Left", "Right"),
            ("L_", "R_"),
        ]
        
        processed = set()
        for joint_name in self.joint_names:
            if joint_name in processed:
                continue
            
            # Find partner
            partner = None
            for left_pat, right_pat in patterns:
                if left_pat in joint_name:
                    candidate = joint_name.replace(left_pat, right_pat, 1)
                    if candidate in self.joint_names and candidate != joint_name:
                        partner = candidate
                        break
                elif right_pat in joint_name:
                    candidate = joint_name.replace(right_pat, left_pat, 1)
                    if candidate in self.joint_names and candidate != joint_name:
                        partner = candidate
                        break
            
            if partner is None:
                continue
            
            # Get joint IDs
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            partner_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, partner)
            
            if jnt_id < 0 or partner_id < 0:
                continue
            
            # Get axes
            axis1 = self.model.jnt_axis[jnt_id]
            axis2 = self.model.jnt_axis[partner_id]
            
            # Check if axes are same or opposite direction
            dot = float(np.dot(axis1, axis2))
            axes_same_direction = dot > 0.5
            
            # Determine primary axis type (X, Y, or Z)
            abs_axis = np.abs(axis1)
            is_x_axis = abs_axis[0] > 0.5
            is_y_axis = abs_axis[1] > 0.5
            
            # Compute mirror sign based on axis type and direction relationship
            if axes_same_direction:
                if is_y_axis:
                    # Y-axis with same directions: same angles for mirror
                    mirror_sign = 1.0
                else:
                    # X/Z-axis with same directions: opposite angles for mirror
                    mirror_sign = -1.0
            else:
                # Opposite direction axes
                if is_x_axis:
                    # X-axis (roll): axis flip creates mirror, use same angles
                    mirror_sign = 1.0
                else:
                    # Y/Z-axis: still need opposite angles
                    mirror_sign = -1.0
            
            # Only override if not already specified in config
            if joint_name not in self.mirror_joint_signs:
                self.mirror_joint_signs[joint_name] = mirror_sign
            if partner not in self.mirror_joint_signs:
                self.mirror_joint_signs[partner] = mirror_sign
            
            processed.add(joint_name)
            processed.add(partner)

    def _auto_detect_root_body(self) -> str:
        """Auto-detect the root body as the first non-world body that is a direct child of world.
        
        Returns:
            Name of the root body.
        
        Raises:
            ValueError: If no root body could be detected.
        """
        for body_id in range(self.model.nbody):
            if body_id == 0:  # Skip world body (body 0)
                continue
            if self.model.body_parentid[body_id] == 0:  # Parent is world
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                if name and name != "world":
                    return name
        raise ValueError("Could not auto-detect root body: no non-world body found as direct child of world")

    # ----- Helper methods for raw MuJoCo -----
    def _get_body_transform(self, body_name: str) -> np.ndarray:
        """Get 4x4 transformation matrix for a body."""
        transformation = np.eye(4)
        body_pos = self.data.body(body_name).xpos.copy()
        body_mat = self.data.body(body_name).xmat.reshape(3, 3).copy()
        transformation[:3, :3] = body_mat
        transformation[:3, 3] = body_pos
        return transformation

    def _get_site_transform(self, name: str) -> np.ndarray:
        """Get 4x4 transformation matrix for a site or body.
        
        First tries to find a site with the given name, then falls back to body.
        """
        transformation = np.eye(4)
        
        # Try site first
        try:
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            if site_id >= 0:
                site_pos = self.data.site(name).xpos.copy()
                site_mat = self.data.site(name).xmat.reshape(3, 3).copy()
                transformation[:3, :3] = site_mat
                transformation[:3, 3] = site_pos
                return transformation
        except Exception:
            pass
        
        # Fall back to body
        try:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                body_pos = self.data.xpos[body_id].copy()
                body_mat = self.data.xmat[body_id].reshape(3, 3).copy()
                transformation[:3, :3] = body_mat
                transformation[:3, 3] = body_pos
                return transformation
        except Exception:
            pass
        
        raise ValueError(f"Could not find site or body named '{name}'")

    def _forward(self) -> None:
        """Run forward kinematics."""
        mujoco.mj_forward(self.model, self.data)

    # -- GUI helpers --

    def _build_ui(self) -> None:
        """Construct the three-column GUI layout (controls + joint sliders)."""
        left_joints, right_joints = self._split_joint_lists()
        columns_handle: Optional[viser.GuiColumnsHandle]
        try:
            columns_handle = self.server.gui.add_columns(
                3,
                widths=(0.25, 0.25, 0.25),
            )
            print(
                "[Viser] Using gui.add_columns(3) for controls + joint sliders.",
                flush=True,
            )
        except Exception as exc:
            columns_handle = None
            print(
                f"[Viser] add_columns(3) unavailable, falling back to single column: {exc}",
                flush=True,
            )

        if columns_handle is not None:
            controls_col, left_col, right_col = columns_handle
            with controls_col:
                self._build_controls_panel()
                self._build_keyframe_sequence_panels()
            with left_col:
                self._build_joint_slider_column(left_joints, "Joint Sliders (L)")
                self._build_ee_panel()
            with right_col:
                self._build_joint_slider_column(right_joints, "Joint Sliders (R)")
                self._build_settings_panel()
            return

        # Fallback layout if multi-column support is unavailable
        self._build_controls_panel()
        self._build_keyframe_sequence_panels()
        self._build_joint_slider_column(left_joints, "Joint Sliders (L)")
        self._build_joint_slider_column(right_joints, "Joint Sliders (R)")
        self._build_ee_panel()
        self._build_settings_panel()

    def _build_controls_panel(self) -> None:
        """Populate the controls column with visualization toggles."""
        if self.motion_name_input is not None:
            return

        self.motion_name_input = self.server.gui.add_text(
            "Motion Name",
            self.config.name,
        )
        save_btn = self.server.gui.add_button("💾 Save Motion")

        @save_btn.on_click
        def _(_e: GuiEvent) -> None:
            self._save_data()

        with self.server.gui.add_folder("🔑 Keyframe Operations"):
            keyframe_ops_row1 = self.server.gui.add_button_group(
                "Actions",
                ["Add", "Remove", "Update"],
            )

            @keyframe_ops_row1.on_click
            def _(_e: GuiEvent) -> None:
                val = keyframe_ops_row1.value
                if val == "Add":
                    self._add_keyframe()
                elif val == "Remove":
                    self._remove_keyframe()
                elif val == "Update":
                    self.worker.request_state_data()

            keyframe_ops_row2 = self.server.gui.add_button_group(
                "Actions",
                ["Test", "Ground"],
            )

            @keyframe_ops_row2.on_click
            def _(_e: GuiEvent) -> None:
                val = keyframe_ops_row2.value
                if val == "Test":
                    self._test_keyframe()
                elif val == "Ground":
                    self.worker.request_on_ground()

        with self.server.gui.add_folder("🎬 Sequence Operations"):
            seq_ops_row1 = self.server.gui.add_button_group(
                "Sequence",
                ["Add to Seq", "Remove from Seq"],
            )

            @seq_ops_row1.on_click
            def _(_e: GuiEvent) -> None:
                val = seq_ops_row1.value
                if val == "Add to Seq":
                    self._add_to_sequence()
                elif val == "Remove from Seq":
                    self._remove_from_sequence()

            seq_ops_row2 = self.server.gui.add_button_group(
                "Sequence",
                ["Play Traj", "Play Qpos", "Stop"],
            )

            @seq_ops_row2.on_click
            def _(_e: GuiEvent) -> None:
                val = seq_ops_row2.value
                if val == "Play Traj":
                    self._test_trajectory()
                elif val == "Play Qpos":
                    self._test_qpos_trajectory()
                elif val == "Stop":
                    with self.worker.lock:
                        self.worker.is_testing = False

    def _build_keyframe_sequence_panels(self) -> None:
        if self.keyframes_summary is None:
            with self.server.gui.add_folder("Keyframes List"):
                self.keyframes_summary = self.server.gui.add_html("No keyframes")
                self.keyframes_summary.scroll = {"enable": True, "max_height": 200}

            with self.server.gui.add_folder("Keyframe Controls"):
                self.keyframe_index_input = self.server.gui.add_text(
                    "Selected Keyframe Index",
                    "-1",
                )
                self.keyframe_name_input = self.server.gui.add_text(
                    "Keyframe Name",
                    self.config.name,
                )
                keyframe_actions = self.server.gui.add_button_group(
                    "Actions",
                    ["Copy", "Move Up", "Move Down"],
                )

                @keyframe_actions.on_click
                def _(_e: GuiEvent) -> None:
                    val = keyframe_actions.value
                    if val == "Copy":
                        self._duplicate_selected_keyframe()
                    elif val == "Move Up":
                        self._move_keyframe(-1)
                    elif val == "Move Down":
                        self._move_keyframe(1)

                assert self.keyframe_index_input is not None
                assert self.keyframe_name_input is not None

                @self.keyframe_index_input.on_update
                def _(ev: GuiEvent) -> None:
                    if id(ev.target) in self._updating_handles:
                        return
                    try:
                        idx = int(str(self.keyframe_index_input.value))
                    except Exception:
                        return
                    if 0 <= idx < len(self.keyframes):
                        self.selected_keyframe = idx
                        self.keyframe_name_input.value = self.keyframes[idx].name
                        self._load_keyframe_to_ui(idx)

                @self.keyframe_name_input.on_update
                def _(ev: GuiEvent) -> None:
                    if (
                        id(ev.target) in self._updating_handles
                        or self.selected_keyframe is None
                    ):
                        return
                    new_name = str(self.keyframe_name_input.value).strip()
                    if not new_name:
                        return
                    for i, kf in enumerate(self.keyframes):
                        if i != self.selected_keyframe and kf.name == new_name:
                            return
                    old_name = self.keyframes[self.selected_keyframe].name
                    self.keyframes[self.selected_keyframe].name = new_name
                    for i, (n, t) in enumerate(self.sequence_list):
                        if n == old_name:
                            self.sequence_list[i] = (new_name, t)
                    self._refresh_keyframes_summary()
                    self._refresh_sequence_summary()

        if self.sequence_summary is None:
            with self.server.gui.add_folder("Sequence List"):
                self.sequence_summary = self.server.gui.add_html("No sequence")
                self.sequence_summary.scroll = {"enable": True, "max_height": 200}

            with self.server.gui.add_folder("Sequence Controls"):
                self.sequence_index_input = self.server.gui.add_text(
                    "Selected Sequence Index",
                    "-1",
                )
                self.sequence_time_input = self.server.gui.add_text(
                    "Arrival Time (t)",
                    "0.0",
                )
                sequence_actions = self.server.gui.add_button_group(
                    "Actions",
                    ["Move Up", "Move Down"],
                )

                @sequence_actions.on_click
                def _(_e: GuiEvent) -> None:
                    val = sequence_actions.value
                    if val == "Move Up":
                        self._move_sequence(-1)
                    elif val == "Move Down":
                        self._move_sequence(1)

                assert self.sequence_index_input is not None
                assert self.sequence_time_input is not None

                @self.sequence_index_input.on_update
                def _(ev: GuiEvent) -> None:
                    if id(ev.target) in self._updating_handles:
                        return
                    try:
                        idx = int(str(self.sequence_index_input.value))
                    except Exception:
                        return
                    if 0 <= idx < len(self.sequence_list):
                        self.selected_sequence = idx
                        name, t = self.sequence_list[idx]
                        self._set_handle_value(self.sequence_time_input, f"{float(t)}")

                @self.sequence_time_input.on_update
                def _(ev: GuiEvent) -> None:
                    if (
                        id(ev.target) in self._updating_handles
                        or self.selected_sequence is None
                    ):
                        return
                    try:
                        new_t = float(str(self.sequence_time_input.value))
                    except Exception:
                        return
                    self._edit_sequence_time(self.selected_sequence, new_t)

    def _get_mirror_partner(self, joint_name: str) -> Optional[str]:
        """Get the mirror partner for a joint name using various naming conventions."""
        # Common left/right naming patterns
        patterns = [
            ("left", "right"),
            ("_l_", "_r_"),
            ("_l", "_r"),  # suffix pattern
            ("l_", "r_"),  # prefix pattern
            ("_left_", "_right_"),
            ("Left", "Right"),
            ("L_", "R_"),
        ]
        
        for left_pat, right_pat in patterns:
            if left_pat in joint_name:
                partner = joint_name.replace(left_pat, right_pat, 1)
                if partner in self.joint_names and partner != joint_name:
                    return partner
            elif right_pat in joint_name:
                partner = joint_name.replace(right_pat, left_pat, 1)
                if partner in self.joint_names and partner != joint_name:
                    return partner
        return None
    
    def _is_left_side(self, joint_name: str) -> bool:
        """Check if joint is on the left side."""
        left_patterns = ["left", "_l_", "_l", "l_", "_left_", "Left", "L_"]
        return any(pat in joint_name for pat in left_patterns)

    def _split_joint_lists(self) -> Tuple[List[str], List[str]]:
        """Split joints into two buckets for left/right slider columns."""
        left_column: List[str] = []
        right_column: List[str] = []
        processed: set[str] = set()

        for joint in self.joint_names:
            if joint in processed:
                continue
            partner = self._get_mirror_partner(joint)
            if partner:
                if self._is_left_side(joint):
                    left_column.append(joint)
                    right_column.append(partner)
                else:
                    left_column.append(partner)
                    right_column.append(joint)
                processed.add(joint)
                processed.add(partner)

        unpaired = [
            joint for joint in self.joint_names if joint not in processed
        ]
        midpoint = (len(unpaired) + 1) // 2
        left_column.extend(unpaired[:midpoint])
        right_column.extend(unpaired[midpoint:])

        return left_column, right_column

    def _build_joint_slider_column(self, joints: List[str], title: str) -> None:
        """Create a folder populated with sliders for the provided joints."""
        if not joints:
            with self.server.gui.add_folder(title):
                self.server.gui.add_markdown("_No joints available._")
            return

        with self.server.gui.add_folder(title):
            for joint_name in joints:
                self._create_joint_slider(joint_name)

    def _create_joint_slider(self, joint_name: str) -> Optional[viser.GuiSliderHandle]:
        """Create a GUI slider for a single joint."""
        if joint_name in self.slider_widgets:
            return self.slider_widgets[joint_name]

        limits = self.joint_limits.get(joint_name)
        if limits is None:
            print(
                f"[Viser] Missing joint limits for '{joint_name}', skipping slider.",
                flush=True,
            )
            return None

        jmin, jmax = float(limits[0]), float(limits[1])
        rounded_min = round(jmin, 2)
        rounded_max = round(jmax, 2)
        span = max(rounded_max - rounded_min, 1e-6)
        step = max(span / 4000.0, 1e-4)
        default_val = float(
            self.default_positions.get(joint_name, (jmin + jmax) * 0.5)
        )
        default_val = min(max(default_val, rounded_min), rounded_max)

        label = self._format_joint_label(joint_name)
        slider = self.server.gui.add_slider(
            label,
            min=rounded_min,
            max=rounded_max,
            step=step,
            initial_value=default_val,
        )
        self.slider_widgets[joint_name] = slider
        slider.precision = 4
        slider.value = round(float(slider.value), 4)

        @slider.on_update
        def _(_event: GuiEvent, jname=joint_name, sld=slider) -> None:
            if id(sld) in self._updating_handles:
                return
            try:
                value = round(float(sld.value), 4)
                if sld.value != value:
                    self._set_handle_value(sld, value)
            except Exception:
                return
            updates = self._update_joint_pos(jname, value)
            for name, angle in updates.items():
                if name == jname:
                    continue
                other = self.slider_widgets.get(name)
                if other is not None:
                    self._set_handle_value(other, float(angle))

        return slider

    # -- Geometry helpers --

    def _build_robot_meshes(self) -> None:
        """Populate `self._geom_handles` with MuJoCo meshes."""
        if trimesh is None:
            print("[Viser] trimesh not installed, skipping mesh visualization", flush=True)
            return

        self._geom_handles.clear()
        self._geom_groups.clear()
        self._geom_base_rgba.clear()
        self._mesh_file_map.clear()
        self._mesh_scale_map.clear()
        self._mesh_quat_map.clear()

        m = self.model

        # Parse materials from XML
        material_colors: Dict[str, Tuple[float, float, float, float]] = {}
        geom_materials: Dict[str, str] = {}

        def parse_materials_from_xml(xml_file: str, visited: set[str]) -> None:
            """Parse material definitions and geom material references."""
            try:
                xml_file = os.path.abspath(xml_file)
                if xml_file in visited:
                    return
                visited.add(xml_file)
                tree = ET.parse(xml_file)
                root = tree.getroot()
                basedir = os.path.dirname(xml_file)

                for material in root.findall(".//material[@rgba]"):
                    mat_name = material.attrib.get("name")
                    rgba_str = material.attrib.get("rgba", "0.5 0.5 0.5 1")
                    rgba = tuple(map(float, rgba_str.split()))
                    if mat_name:
                        material_colors[mat_name] = rgba

                for geom in root.findall(".//geom[@material]"):
                    geom_name = geom.attrib.get("name")
                    mat_ref = geom.attrib.get("material")
                    if geom_name and mat_ref:
                        geom_materials[geom_name] = mat_ref

                for inc in root.findall(".//include"):
                    inc_file = inc.attrib.get("file")
                    if inc_file:
                        inc_path = (
                            inc_file
                            if os.path.isabs(inc_file)
                            else os.path.join(basedir, inc_file)
                        )
                        if os.path.exists(inc_path):
                            parse_materials_from_xml(inc_path, visited)
            except Exception as exc:
                print(
                    f"[Viser] Warning: Failed to parse materials from {xml_file}: {exc}",
                    flush=True,
                )

        if self.xml_path and os.path.exists(self.xml_path):
            parse_materials_from_xml(self.xml_path, set())

        # Parse mesh file paths from XML
        def parse_xml_meshes(xml_file: str, visited: set[str]) -> None:
            try:
                xml_file = os.path.abspath(xml_file)
                if xml_file in visited:
                    return
                visited.add(xml_file)
                tree = ET.parse(xml_file)
                root = tree.getroot()
                basedir = os.path.dirname(xml_file)
                meshdir = root.attrib.get("meshdir", "")
                comp = root.find("compiler")
                if comp is not None and comp.attrib.get("meshdir"):
                    meshdir = comp.attrib.get("meshdir")
                asset_base = os.path.join(basedir, meshdir) if meshdir else basedir

                for mesh in root.findall(".//asset/mesh"):
                    name = mesh.attrib.get("name")
                    file = mesh.attrib.get("file")
                    if not name and file:
                        name = os.path.splitext(os.path.basename(file))[0]
                    if name and file:
                        fpath = (
                            file
                            if os.path.isabs(file)
                            else os.path.join(asset_base, file)
                        )
                        if os.path.exists(fpath):
                            self._mesh_file_map.setdefault(name, fpath)
                        scale_txt = mesh.attrib.get("scale")
                        if scale_txt:
                            try:
                                vals = [float(x) for x in scale_txt.strip().split()]
                                if len(vals) == 1:
                                    s = (vals[0], vals[0], vals[0])
                                elif len(vals) == 3:
                                    s = (vals[0], vals[1], vals[2])
                                else:
                                    s = (1.0, 1.0, 1.0)
                                self._mesh_scale_map[name] = s
                            except Exception:
                                pass
                        quat_txt = mesh.attrib.get("quat")
                        if quat_txt:
                            try:
                                vals = [float(x) for x in quat_txt.strip().split()]
                                if len(vals) == 4:
                                    self._mesh_quat_map[name] = (
                                        vals[0], vals[1], vals[2], vals[3],
                                    )
                            except Exception:
                                pass

                for inc in root.findall(".//include"):
                    inc_file = inc.attrib.get("file")
                    if inc_file:
                        inc_path = (
                            inc_file
                            if os.path.isabs(inc_file)
                            else os.path.join(basedir, inc_file)
                        )
                        if os.path.exists(inc_path):
                            parse_xml_meshes(inc_path, visited)
            except Exception:
                return

        if self.xml_path and os.path.exists(self.xml_path):
            parse_xml_meshes(self.xml_path, set())

        print(
            f"[Viser] Parsed {len(material_colors)} materials, {len(geom_materials)} geom references",
            flush=True,
        )

        # Build geometry handles
        try:
            geom_type = np.array(m.geom_type, dtype=np.int32)
            geom_size = np.array(m.geom_size, dtype=np.float32)
            geom_rgba = (
                np.array(m.geom_rgba, dtype=np.float32)
                if hasattr(m, "geom_rgba")
                else np.tile(
                    np.array([0.7, 0.7, 0.7, 1.0], dtype=np.float32), (m.ngeom, 1)
                )
            )
            geom_group = (
                np.array(m.geom_group, dtype=np.int32)
                if hasattr(m, "geom_group")
                else np.zeros(m.ngeom, dtype=np.int32)
            )
            geom_dataid = (
                np.array(m.geom_dataid, dtype=np.int32)
                if hasattr(m, "geom_dataid")
                else np.full(m.ngeom, -1, dtype=np.int32)
            )
            geom_matid = (
                np.array(m.geom_matid, dtype=np.int32)
                if hasattr(m, "geom_matid")
                else np.full(m.ngeom, -1, dtype=np.int32)
            )
            # Get material RGBA colors from MuJoCo
            mat_rgba = (
                np.array(m.mat_rgba, dtype=np.float32)
                if hasattr(m, "mat_rgba") and m.nmat > 0
                else None
            )
        except Exception:
            geom_type = np.array(
                [m.geom(i).type for i in range(m.ngeom)], dtype=np.int32
            )
            geom_size = np.array(
                [m.geom(i).size for i in range(m.ngeom)], dtype=np.float32
            )
            geom_rgba = np.tile(
                np.array([0.7, 0.7, 0.7, 1.0], dtype=np.float32), (m.ngeom, 1)
            )
            geom_group = np.zeros(m.ngeom, dtype=np.int32)
            geom_dataid = np.full(m.ngeom, -1, dtype=np.int32)
            geom_matid = np.full(m.ngeom, -1, dtype=np.int32)
            mat_rgba = None

        mesh_vert = getattr(m, "mesh_vert", None)
        mesh_face = getattr(m, "mesh_face", None)
        mesh_vertadr = getattr(m, "mesh_vertadr", None)
        mesh_vertnum = getattr(m, "mesh_vertnum", None)
        mesh_faceadr = getattr(m, "mesh_faceadr", None)
        mesh_facenum = getattr(m, "mesh_facenum", None)

        for i in range(m.ngeom):
            try:
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
            except Exception:
                name = ""
            gtype = int(geom_type[i])
            size = np.array(geom_size[i])
            rgba = tuple(map(float, geom_rgba[i]))
            group = int(geom_group[i])
            
            # First, try to get color from MuJoCo material assignment
            mat_id = int(geom_matid[i]) if geom_matid is not None else -1
            if mat_id >= 0 and mat_rgba is not None and mat_id < len(mat_rgba):
                rgba = tuple(map(float, mat_rgba[mat_id]))
            # Fallback: try XML-parsed materials by geom name
            elif name and name in geom_materials:
                mat_name = geom_materials[name]
                if mat_name in material_colors:
                    rgba = material_colors[mat_name]

            self._geom_groups[i] = group
            self._geom_base_rgba[i] = rgba

            try:
                if gtype == int(mujoco.mjtGeom.mjGEOM_PLANE) or "floor" in name:
                    continue
            except Exception:
                pass

            mesh = None
            try:
                if gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                    mesh = trimesh.creation.icosphere(
                        radius=float(size[0]), subdivisions=3
                    )
                elif gtype == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
                    height = float(2.0 * size[1])
                    mesh = trimesh.creation.capsule(
                        radius=float(size[0]), height=height, count=[16, 16]
                    )
                elif gtype == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                    height = float(2.0 * size[1])
                    mesh = trimesh.creation.cylinder(
                        radius=float(size[0]), height=height, sections=24
                    )
                elif gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
                    extents = 2.0 * size[:3]
                    mesh = trimesh.creation.box(extents=extents)
                elif gtype == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
                    sph = trimesh.creation.icosphere(radius=1.0, subdivisions=3)
                    sph.apply_scale(2.0 * size[:3])
                    mesh = sph
                elif gtype == int(mujoco.mjtGeom.mjGEOM_MESH):
                    mid = int(geom_dataid[i]) if geom_dataid is not None else -1
                    built = False
                    if (
                        mid >= 0
                        and mesh_vert is not None
                        and mesh_vertadr is not None
                        and mesh_vertnum is not None
                        and mesh_face is not None
                        and mesh_faceadr is not None
                        and mesh_facenum is not None
                    ):
                        try:
                            vadr = int(mesh_vertadr[mid])
                            vnum = int(mesh_vertnum[mid])
                            fadr = int(mesh_faceadr[mid])
                            fnum = int(mesh_facenum[mid])
                            vert_arr = np.asarray(mesh_vert, dtype=np.float32)
                            face_arr = np.asarray(mesh_face, dtype=np.int32)
                            verts = vert_arr[vadr : vadr + vnum]
                            faces = face_arr[fadr : fadr + fnum]
                            mesh = trimesh.Trimesh(
                                vertices=verts, faces=faces, process=False
                            )
                            built = True
                        except Exception:
                            built = False
                    if not built and mid >= 0:
                        try:
                            mesh_name = mujoco.mj_id2name(
                                m, mujoco.mjtObj.mjOBJ_MESH, mid
                            )
                        except Exception:
                            mesh_name = None
                        fpath = (
                            self._mesh_file_map.get(mesh_name) if mesh_name else None
                        )
                        if fpath and os.path.exists(fpath):
                            try:
                                mesh = trimesh.load(fpath, force="mesh")
                                sc = self._mesh_scale_map.get(mesh_name)
                                if sc is not None:
                                    mesh.apply_scale(sc)
                                q = self._mesh_quat_map.get(mesh_name)
                                if q is not None:
                                    rot = R.from_quat([q[1], q[2], q[3], q[0]])
                                    T = np.eye(4)
                                    T[:3, :3] = rot.as_matrix()
                                    mesh.apply_transform(T)
                                built = True
                            except Exception:
                                built = False
            except Exception:
                mesh = None

            if mesh is None:
                continue

            try:
                color_rgba_255 = [
                    int(rgba[0] * 255),
                    int(rgba[1] * 255),
                    int(rgba[2] * 255),
                    int(rgba[3] * 255),
                ]
                mesh.visual = trimesh.visual.ColorVisuals(
                    mesh, face_colors=color_rgba_255
                )
            except Exception:
                pass

            path = f"/robot/{'collision' if group == 3 else 'visual'}/{i:04d}_{name}"
            handle = None
            try:
                handle = self.server.scene.add_mesh_trimesh(path, mesh)
            except Exception:
                try:
                    handle = self.server.scene.add_mesh_simple(
                        path,
                        vertices=np.asarray(mesh.vertices, dtype=float),
                        faces=np.asarray(mesh.faces, dtype=int),
                        color=(rgba[0], rgba[1], rgba[2]),
                    )
                except Exception:
                    handle = None

            if handle is not None:
                self._geom_handles[i] = handle

        print(
            f"[Viser] Built {len(self._geom_handles)} geom meshes",
            flush=True,
        )

    def _apply_geom_visibility(self) -> None:
        """Update visibility of robot geometry handles."""
        show_collision = (
            bool(self.collision_geom_checked.value)
            if self.collision_geom_checked
            else False
        )
        try:
            show_all = bool(self.show_all_geoms.value) if self.show_all_geoms else False
        except Exception:
            show_all = True

        for i, handle in self._geom_handles.items():
            group = self._geom_groups.get(i, 0)
            base = self._geom_base_rgba.get(i, (0.7, 0.7, 0.7, 1.0))
            if show_all:
                visible = True
            else:
                visible = (group == 3) if show_collision else (group != 3)
            try:
                if hasattr(handle, "visible"):
                    handle.visible = visible
                elif hasattr(handle, "rgba"):
                    handle.rgba = (
                        float(base[0]),
                        float(base[1]),
                        float(base[2]),
                        1.0 if visible else 0.05,
                    )
            except Exception:
                continue

    def _start_scene_updater(self) -> None:
        """Launch the background scene updater thread."""
        if not (self._geom_handles or self._scene_handles or self._com_sphere):
            return
        if self._scene_updater and self._scene_updater.is_alive():
            return

        self._scene_updater = threading.Thread(
            target=self._update_scene_loop,
            name="ViserSceneUpdater",
            daemon=True,
        )
        self._scene_updater.start()

    def _update_scene_loop(self) -> None:
        """Periodically push scene poses and apply visibility toggles."""
        while True:
            try:
                if self._com_sphere is not None:
                    with self.worker_lock:
                        com_pos = self.data.subtree_com[0].copy()
                    try:
                        self._com_sphere.position = tuple(map(float, com_pos))
                    except Exception:
                        pass

                if self._geom_handles:
                    with self.worker_lock:
                        xpos = np.array(self.data.geom_xpos, dtype=np.float32)
                        use_quat = True
                        try:
                            xquat_wxyz = np.array(
                                self.data.geom_xquat, dtype=np.float32
                            )
                        except Exception:
                            use_quat = False
                            xmat = np.array(
                                self.data.geom_xmat, dtype=np.float32
                            ).reshape(-1, 3, 3)
                    for index, handle in list(self._geom_handles.items()):
                        try:
                            position = tuple(map(float, xpos[index]))
                            if use_quat:
                                qw, qx, qy, qz = map(float, xquat_wxyz[index])
                            else:
                                quat_xyzw = R.from_matrix(xmat[index]).as_quat()
                                qx, qy, qz, qw = map(float, quat_xyzw)
                            handle.position = position
                            if hasattr(handle, "wxyz"):
                                handle.wxyz = (qw, qx, qy, qz)
                        except Exception:
                            continue
                    self._apply_geom_visibility()
            except Exception:
                pass
            time.sleep(0.05)

    # -- Trajectory methods --

    def _test_trajectory(self) -> None:
        if len(self.sequence_list) < 2:
            print("[Viser] Action traj: need at least 2 sequence entries.", flush=True)
            return
        start_idx = self.selected_sequence or 0
        action_list: List[np.ndarray] = []
        qpos_list: List[np.ndarray] = []
        times: List[float] = []
        for name, t in self.sequence_list:
            for kf in self.keyframes:
                if kf.name == name and kf.qpos is not None:
                    action_list.append(kf.motor_pos)
                    qpos_list.append(kf.qpos)
                    times.append(t)
                    break
        if len(times) < 2:
            print(f"[Viser] Action traj: collected fewer than 2 timestamps.", flush=True)
            return
        times_arr = np.array(times)
        if np.any(np.diff(times_arr) <= 0):
            print(f"[Viser] Action traj: times not strictly increasing: {times}", flush=True)
            return
        qpos_start = qpos_list[start_idx]
        enabled = bool(self.physics_enabled.value) if self.physics_enabled else True
        action_arr = np.array(action_list)
        times_arr = times_arr - times_arr[0]
        self.traj_times = list(np.arange(0, times_arr[-1], self.dt))
        self.action_traj = []
        for t in self.traj_times:
            if t < times_arr[-1]:
                motor_pos = interpolate_action(t, times_arr, action_arr)
            else:
                motor_pos = action_arr[-1]
            self.action_traj.append(motor_pos)
        traj_start = int(np.searchsorted(self.traj_times, times_arr[start_idx]))
        self.is_qpos_traj = False
        self.is_relative_frame = (
            bool(self.relative_frame_checked.value)
            if self.relative_frame_checked
            else True
        )
        self.worker.request_trajectory_test(
            qpos_start,
            self.action_traj[traj_start:],
            self.dt,
            enabled,
            is_qpos_traj=False,
            is_relative_frame=self.is_relative_frame,
        )

    def _test_qpos_trajectory(self) -> None:
        if len(self.sequence_list) < 2:
            print("[Viser] Qpos traj: need at least 2 sequence entries.", flush=True)
            return
        start_idx = self.selected_sequence or 0
        qpos_list: List[np.ndarray] = []
        times: List[float] = []
        for name, t in self.sequence_list:
            for kf in self.keyframes:
                if kf.name == name and kf.qpos is not None:
                    qpos_list.append(kf.qpos)
                    times.append(t)
                    break
        if len(times) < 2:
            print(f"[Viser] Qpos traj: collected fewer than 2 timestamps.", flush=True)
            return
        
        # Debug: check if qpos values are actually different
        qpos_diff = np.max(np.abs(np.array(qpos_list[0]) - np.array(qpos_list[-1])))
        print(f"[Viser] Qpos traj: max qpos diff between first and last: {qpos_diff:.6f}", flush=True)
        if qpos_diff < 1e-6:
            print("[Viser] WARNING: Keyframes have nearly identical qpos! Move sliders between keyframes.", flush=True)
        times_arr = np.array(times)
        if np.any(np.diff(times_arr) <= 0):
            print(f"[Viser] Qpos traj: times not strictly increasing: {times}", flush=True)
            return
        times_arr = times_arr - times_arr[0]
        self.traj_times = list(np.arange(0, times_arr[-1], self.dt))
        qpos_arr = np.array(qpos_list)
        qpos_traj: List[np.ndarray] = []
        traj_start = int(np.searchsorted(self.traj_times, times_arr[start_idx]))
        for t in self.traj_times:
            if t < times_arr[-1]:
                qpos_t = interpolate_action(t, times_arr, qpos_arr)
            else:
                qpos_t = qpos_arr[-1]
            qpos_traj.append(qpos_t)
        self.is_qpos_traj = True
        self.is_relative_frame = (
            bool(self.relative_frame_checked.value)
            if self.relative_frame_checked
            else True
        )
        self.worker.request_trajectory_test(
            qpos_list[start_idx],
            qpos_traj[traj_start:],
            self.dt,
            physics_enabled=False,
            is_qpos_traj=True,
            is_relative_frame=self.is_relative_frame,
        )

    # -- Data save/load --

    def _save_data(self) -> None:
        try:
            result_dict: Dict[str, object] = {}
            saved_keyframes = [asdict(kf) for kf in self.keyframes]

            result_dict["time"] = np.array(self.traj_times)
            result_dict["qpos"] = np.array(self.qpos_replay)
            result_dict["body_pos"] = np.array(self.body_pos_replay)
            result_dict["body_quat"] = np.array(self.body_quat_replay)
            result_dict["body_lin_vel"] = np.array(self.body_lin_vel_replay)
            result_dict["body_ang_vel"] = np.array(self.body_ang_vel_replay)
            result_dict["site_pos"] = np.array(self.site_pos_replay)
            result_dict["site_quat"] = np.array(self.site_quat_replay)
            result_dict["action"] = (
                None if self.is_qpos_traj else np.array(self.action_traj) if self.action_traj else None
            )
            result_dict["keyframes"] = saved_keyframes
            result_dict["timed_sequence"] = self.sequence_list
            result_dict["is_robot_relative_frame"] = self.is_relative_frame

            motion_name = (
                self.motion_name_input.value if self.motion_name_input else self.config.name
            )
            
            # Ensure directory exists
            os.makedirs(self.result_dir, exist_ok=True)
            
            # Save as {motion_name}.lz4 - overwrites previous save with same name
            result_path = os.path.join(self.result_dir, f"{motion_name}.lz4")
            joblib.dump(result_dict, result_path, compress="lz4")
            print(f"✅ Saved motion to: {result_path}", flush=True)
        except Exception as e:
            print(f"❌ Failed to save motion: {e}", flush=True)

    def _load_data(self) -> bool:
        """Load keyframe data from file. Returns True if data was loaded."""
        self.keyframes.clear()
        self.sequence_list.clear()
        self._refresh_keyframes_table()
        self._refresh_sequence_table()

        # Create default keyframe
        default_motor_pos = np.zeros(self.model.nu, dtype=np.float32)
        default_joint_pos = np.array(
            [self.default_positions.get(name, 0.0) for name in self.joint_names],
            dtype=np.float32
        )

        if not self.data_path:
            print("[Load] No data path provided, starting fresh.", flush=True)
            # Create default keyframe - qpos will be updated after grounding
            self.keyframes.append(
                Keyframe(
                    name="default",
                    motor_pos=default_motor_pos,
                    joint_pos=default_joint_pos,
                    qpos=self.home_qpos.copy(),
                )
            )
            self._refresh_keyframes_table()
            # Don't load keyframe to UI yet - let grounding happen first
            # The grounded position will be applied via _update_default_keyframe_after_ground()
            return False
        
        if not os.path.exists(self.data_path):
            print(f"[Load] ❌ File not found: {self.data_path}", flush=True)
            self.keyframes.append(
                Keyframe(
                    name="default",
                    motor_pos=default_motor_pos,
                    joint_pos=default_joint_pos,
                    qpos=self.home_qpos.copy(),
                )
            )
            self._refresh_keyframes_table()
            self._load_keyframe_to_ui(0)
            return False

        try:
            print(f"[Load] Loading from: {self.data_path}", flush=True)
            data = joblib.load(self.data_path)
        except Exception as e:
            print(f"[Load] ❌ Failed to load: {e}", flush=True)
            data = None

        if data is None:
            self.keyframes.append(
                Keyframe(
                    name="default",
                    motor_pos=default_motor_pos,
                    joint_pos=default_joint_pos,
                    qpos=self.home_qpos.copy(),
                )
            )
            self._refresh_keyframes_table()
            self._load_keyframe_to_ui(0)
            return False

        keyframes_data = data.get("keyframes") if isinstance(data, dict) else None
        if keyframes_data is not None:
            loaded_keyframes: List[Keyframe] = []
            for k in keyframes_data:
                loaded_keyframes.append(
                    Keyframe(
                        name=k["name"],
                        motor_pos=np.array(k["motor_pos"], dtype=np.float32),
                        joint_pos=np.array(k["joint_pos"], dtype=np.float32)
                        if k.get("joint_pos") is not None
                        else None,
                        qpos=np.array(k["qpos"], dtype=np.float32)
                        if k.get("qpos") is not None
                        else None,
                    )
                )
            self.keyframes.extend(loaded_keyframes)
            sequence_entries = data.get("timed_sequence", [])
            self.sequence_list = [
                (n.replace(" ", "_"), float(t)) for (n, t) in sequence_entries
            ]
            self.traj_times = list(map(float, data.get("time", [])))
            self.action_traj = data.get("action", [])
            self.qpos_replay = list(data.get("qpos", []))
            self._refresh_keyframes_table()
            self._refresh_sequence_table()
            print(f"[Load] ✅ Loaded {len(self.keyframes)} keyframes, {len(self.sequence_list)} sequence entries", flush=True)
            if self.keyframes:
                self._load_keyframe_to_ui(0)
            return True
        
        # No keyframes in data
        print("[Load] ⚠️ No keyframes found in file", flush=True)
        self.keyframes.append(
            Keyframe(
                name="default",
                motor_pos=default_motor_pos,
                joint_pos=default_joint_pos,
                qpos=self.home_qpos.copy(),
            )
        )
        self._refresh_keyframes_table()
        self._load_keyframe_to_ui(0)
        return False

    # -- Keyframe operations --

    def _add_keyframe(self) -> None:
        default_motor_pos = np.zeros(self.model.nu, dtype=np.float32)
        default_joint_pos = np.array(
            [self.default_positions.get(name, 0.0) for name in self.joint_names],
            dtype=np.float32
        )

        if self.selected_keyframe is None:
            base_name = "keyframe"
            new_name = self._generate_unique_name(base_name)
            new_kf = Keyframe(
                name=new_name,
                motor_pos=default_motor_pos,
                joint_pos=default_joint_pos,
                qpos=self.home_qpos.copy(),
            )
        else:
            kf = self.keyframes[self.selected_keyframe]
            base_name = self._base_keyframe_name(kf.name)
            new_name = self._generate_unique_name(base_name)
            new_kf = Keyframe(
                name=new_name,
                motor_pos=kf.motor_pos.copy(),
                joint_pos=kf.joint_pos.copy() if kf.joint_pos is not None else None,
                qpos=kf.qpos.copy() if kf.qpos is not None else None,
            )
        self.keyframes.append(new_kf)
        self.selected_keyframe = len(self.keyframes) - 1
        self._refresh_keyframes_table()
        self._load_keyframe_to_ui(self.selected_keyframe)

    def _remove_keyframe(self) -> None:
        if self.selected_keyframe is None:
            return
        name_to_remove = self.keyframes[self.selected_keyframe].name
        self.keyframes.pop(self.selected_keyframe)
        self.sequence_list = [
            (n, t) for (n, t) in self.sequence_list if n != name_to_remove
        ]
        self.selected_keyframe = None
        self._refresh_keyframes_table()
        self._refresh_sequence_table()

    def _duplicate_selected_keyframe(self) -> None:
        if self.selected_keyframe is None:
            return
        kf = self.keyframes[self.selected_keyframe]
        base_name = self._base_keyframe_name(kf.name)
        new_name = self._generate_unique_name(base_name)
        new_kf = Keyframe(
            name=new_name,
            motor_pos=kf.motor_pos.copy(),
            joint_pos=kf.joint_pos.copy() if kf.joint_pos is not None else None,
            qpos=kf.qpos.copy() if kf.qpos is not None else None,
        )
        self.keyframes.append(new_kf)
        self._refresh_keyframes_table()

    def _move_keyframe(self, direction: int) -> None:
        if self.selected_keyframe is None:
            return
        current_idx = self.selected_keyframe
        new_idx = current_idx + direction
        if 0 <= new_idx < len(self.keyframes):
            self.keyframes[current_idx], self.keyframes[new_idx] = (
                self.keyframes[new_idx],
                self.keyframes[current_idx],
            )
            self.selected_keyframe = new_idx
            self._refresh_keyframes_table()

    def _load_keyframe_to_ui(self, idx: int) -> None:
        kf = self.keyframes[idx]
        self.selected_keyframe = idx
        if kf.qpos is not None:
            self.worker.update_qpos(kf.qpos)
        if kf.joint_pos is not None:
            for jname, val in zip(self.joint_names, kf.joint_pos):
                slider = self.slider_widgets.get(jname)
                if slider is not None:
                    self._set_handle_value(slider, float(val))
        if self.keyframe_index_input is not None:
            self._set_handle_value(self.keyframe_index_input, str(idx))
        if self.keyframe_name_input is not None:
            self._set_handle_value(self.keyframe_name_input, kf.name)
        self._refresh_sequence_table()

    def _add_to_sequence(self) -> None:
        if self.selected_keyframe is None:
            return
        kf = self.keyframes[self.selected_keyframe]
        last_t = self.sequence_list[-1][1] if self.sequence_list else 0.0
        self.sequence_list.append((kf.name, last_t + 1.0))
        self._refresh_sequence_table()

    def _remove_from_sequence(self) -> None:
        if self.selected_sequence is None:
            return
        self.sequence_list.pop(self.selected_sequence)
        self.selected_sequence = None
        self._refresh_sequence_table()

    def _move_sequence(self, direction: int) -> None:
        if self.selected_sequence is None:
            return
        current_idx = self.selected_sequence
        new_idx = current_idx + direction
        if 0 <= new_idx < len(self.sequence_list):
            self.sequence_list[current_idx], self.sequence_list[new_idx] = (
                self.sequence_list[new_idx],
                self.sequence_list[current_idx],
            )
            self.selected_sequence = new_idx
            self._refresh_sequence_table()

    def _edit_sequence_time(self, row: int, new_time: float) -> None:
        if not (0 <= row < len(self.sequence_list)):
            return
        name, old_time = self.sequence_list[row]
        delta = float(new_time) - float(old_time)
        for i in range(row, len(self.sequence_list)):
            n, t = self.sequence_list[i]
            self.sequence_list[i] = (n, float(t) + delta)
        self._refresh_sequence_table()

    def _refresh_keyframes_table(self) -> None:
        self._refresh_keyframes_summary()

    def _refresh_keyframes_summary(self) -> None:
        if self.keyframes_summary is None:
            return
        if not self.keyframes:
            self.keyframes_summary.content = (
                '<div style="font-size:0.875em; margin-left:0.75em">No keyframes</div>'
            )
            return
        lines = [f"{i}: {kf.name}" for i, kf in enumerate(self.keyframes)]
        content = "<br/>".join(lines)
        visible_rows = min(len(lines), 3) or 1
        line_height_em = 1.3
        max_height_em = line_height_em * visible_rows
        wrapped = (
            f'<div style="line-height:{line_height_em}em; '
            f"max-height:{max_height_em}em; overflow-y:auto; "
            f'font-size:0.875em; margin-left:0.75em">'
            f"{content}</div>"
        )
        self.keyframes_summary.content = wrapped

    def _refresh_sequence_table(self) -> None:
        self._refresh_sequence_summary()

    def _refresh_sequence_summary(self) -> None:
        if self.sequence_summary is None:
            return
        if not self.sequence_list:
            self.sequence_summary.content = (
                '<div style="font-size:0.875em; margin-left:0.75em">No sequence</div>'
            )
            return
        lines = [
            f"{i}: {n.replace(' ', '_')} &nbsp;&nbsp; t={t}"
            for i, (n, t) in enumerate(self.sequence_list)
        ]
        content = "<br/>".join(lines)
        visible_rows = min(len(lines), 3) or 1
        line_height_em = 1.3
        max_height_em = line_height_em * visible_rows
        wrapped = (
            f'<div style="line-height:{line_height_em}em; '
            f"max-height:{max_height_em}em; overflow-y:auto; "
            f'font-size:0.875em; margin-left:0.75em">'
            f"{content}</div>"
        )
        self.sequence_summary.content = wrapped

    def _update_default_keyframe_after_ground(self) -> None:
        """Update the default keyframe with the current (grounded) qpos."""
        # Get the current qpos from the simulation (grounded position)
        with self.worker_lock:
            grounded_qpos = self.data.qpos.copy()
        
        # Update the default keyframe
        if self.keyframes:
            self.keyframes[0].qpos = grounded_qpos
            # Also update joint positions from the grounded qpos
            joint_pos = []
            for jname in self.joint_names:
                try:
                    jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                    qpos_adr = self.model.jnt_qposadr[jnt_id]
                    joint_pos.append(float(grounded_qpos[qpos_adr]))
                except Exception:
                    joint_pos.append(0.0)
            self.keyframes[0].joint_pos = np.array(joint_pos, dtype=np.float32)
        
        # Now load the keyframe to UI
        self._load_keyframe_to_ui(0)

    def _test_keyframe(self) -> None:
        if self.selected_keyframe is None:
            return
        kf = self.keyframes[self.selected_keyframe]
        if kf.qpos is None:
            return
        self.worker.request_keyframe_test(kf, self.dt)

    def _set_handle_value(self, handle, value: object) -> None:
        hid = id(handle)
        self._updating_handles.add(hid)
        try:
            rounded_value = value
            if hasattr(handle, "precision"):
                try:
                    prec = int(getattr(handle, "precision"))
                except Exception:
                    prec = None
                if prec is not None and isinstance(value, (int, float)):
                    rounded_value = round(float(value), prec)
            handle.value = rounded_value
        finally:
            self._updating_handles.discard(hid)

    def _update_joint_pos(self, joint_name: str, value: float) -> Dict[str, float]:
        updates: Dict[str, float] = {joint_name: float(value)}
        mirror = bool(self.mirror_checked.value) if self.mirror_checked else True
        rev_mirror = (
            bool(self.rev_mirror_checked.value) if self.rev_mirror_checked else False
        )
        if mirror or rev_mirror:
            mirrored_joint_name = self._get_mirror_partner(joint_name)
            if mirrored_joint_name:
                is_left = self._is_left_side(joint_name)
                # Get the mirror sign for this joint pair
                # Sign accounts for joint axis orientation differences:
                # - If axes are same direction: sign = -1 (need opposite angles for geometric mirror)
                # - If axes are opposite direction: sign = +1 (need same angles for geometric mirror)
                mirror_sign = (
                    self.mirror_joint_signs.get(joint_name, -1.0)
                    if is_left
                    else self.mirror_joint_signs.get(mirrored_joint_name, -1.0)
                )
                # Mirror = geometric mirror (symmetric pose)
                # Rev Mirror = opposite of mirror (anti-symmetric pose)
                if mirror:
                    updates[mirrored_joint_name] = value * mirror_sign
                elif rev_mirror:
                    updates[mirrored_joint_name] = -value * mirror_sign
        
        # Apply inverse computation for gear mechanisms: when a driven joint is set,
        # compute the corresponding drive joint value
        # driven = offset + ratio * drive  =>  drive = (driven - offset) / ratio
        gear_inverse_updates: Dict[str, float] = {}
        for driven_joint, driven_value in updates.items():
            if driven_joint in self.joint_couplings_inverse:
                drive_joint, offset, ratio = self.joint_couplings_inverse[driven_joint]
                if abs(ratio) > 1e-6:  # Avoid division by zero
                    drive_value = (driven_value - offset) / ratio
                    gear_inverse_updates[drive_joint] = drive_value
        updates.update(gear_inverse_updates)
        
        # Apply forward computation for gear mechanisms: when a drive joint moves,
        # its coupled driven joint must also move according to the gear ratio
        # driven = offset + ratio * drive
        coupled_updates: Dict[str, float] = {}
        for drive_joint, drive_value in updates.items():
            if drive_joint in self.joint_couplings:
                driven_joint, offset, ratio = self.joint_couplings[drive_joint]
                driven_value = offset + ratio * drive_value
                coupled_updates[driven_joint] = driven_value
        updates.update(coupled_updates)
        
        # Apply inverse computation for differential drives: when a motion joint changes,
        # compute the corresponding motor joint values by solving the tendon constraint system
        differential_updates: Dict[str, float] = {}
        
        # Group motion joints by shared motors (same differential drive system)
        motor_groups: Dict[frozenset, List[Tuple[str, str, float]]] = {}  # {motor_names} -> [(motion_name, tendon_name, motion_coef), ...]
        
        for motion_joint in updates.keys():
            if motion_joint in self.differential_drives:
                tendon_name, motor_list, motion_coef = self.differential_drives[motion_joint]
                motor_names = frozenset(m[0] for m in motor_list)
                if motor_names not in motor_groups:
                    motor_groups[motor_names] = []
                motor_groups[motor_names].append((motion_joint, tendon_name, motion_coef))
        
        # For each motor group, solve the system
        for motor_names, motion_list in motor_groups.items():
            if len(motor_names) == 2 and len(motion_list) == 2:
                # 2x2 system: 2 motors, 2 motions
                # Get all tendons for this system
                all_tendons = []
                for motion_name, tendon_name, _ in motion_list:
                    # Find the tendon and get all coefficients
                    for tendon_id in range(self.model.ntendon):
                        tn = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_TENDON, tendon_id)
                        if tn == tendon_name:
                            adr = self.model.tendon_adr[tendon_id]
                            num = self.model.tendon_num[tendon_id]
                            
                            motors = []
                            motions = []
                            for i in range(num):
                                wrap_idx = adr + i
                                obj_id = self.model.wrap_objid[wrap_idx]
                                coef = float(self.model.wrap_prm[wrap_idx])
                                jname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, obj_id)
                                
                                if jname in motor_names:
                                    motors.append((jname, coef))
                                else:
                                    motions.append((jname, coef))
                            
                            all_tendons.append((motors, motions))
                            break
                
                if len(all_tendons) == 2:
                    # Solve 2x2 system: c1*m1 + c2*m2 + cr*roll = 0, d1*m1 + d2*m2 + dy*yaw = 0
                    (m1_name, c1), (m2_name, c2) = all_tendons[0][0]
                    (m1_name2, d1), (m2_name2, d2) = all_tendons[1][0]
                    
                    (r_name, cr) = all_tendons[0][1][0]
                    (y_name, dy) = all_tendons[1][1][0]
                    
                    # Get current or updated values
                    with self.worker_lock:
                        roll_val = updates.get(r_name, float(self.data.joint(r_name).qpos[0]))
                        yaw_val = updates.get(y_name, float(self.data.joint(y_name).qpos[0]))
                    
                    # Solve: c1*m1 + c2*m2 = -cr*roll
                    #        d1*m1 + d2*m2 = -dy*yaw
                    # Using Cramer's rule:
                    # m1 = (-cr*roll*d2 - (-dy*yaw)*c2) / (c1*d2 - c2*d1)
                    # m2 = (c1*(-dy*yaw) - (-cr*roll)*d1) / (c1*d2 - c2*d1)
                    det = c1 * d2 - c2 * d1
                    if abs(det) > 1e-6:
                        m1_val = (-cr * roll_val * d2 + dy * yaw_val * c2) / det
                        m2_val = (-c1 * dy * yaw_val + cr * roll_val * d1) / det
                        
                        differential_updates[m1_name] = m1_val
                        differential_updates[m2_name] = m2_val
        
        updates.update(differential_updates)
        
        # Apply inverse computation for parallel linkages: when a motion joint changes,
        # compute the corresponding motor joint value
        # motion = ratio * motor  =>  motor = motion / ratio
        parallel_inverse_updates: Dict[str, float] = {}
        for motion_joint, motion_value in updates.items():
            if motion_joint in self.parallel_linkages_inverse:
                motor_joint, ratio = self.parallel_linkages_inverse[motion_joint]
                if abs(ratio) > 1e-6:
                    motor_value = motion_value / ratio
                    parallel_inverse_updates[motor_joint] = motor_value
        updates.update(parallel_inverse_updates)
        
        self.worker.update_joint_angles(updates)
        return updates

    def _on_state(
        self,
        motor_pos: np.ndarray,
        joint_pos: np.ndarray,
        qpos: np.ndarray,
    ) -> None:
        if self.selected_keyframe is None:
            return
        idx = self.selected_keyframe
        self.keyframes[idx].motor_pos = motor_pos.copy()
        self.keyframes[idx].joint_pos = joint_pos.copy()
        self.keyframes[idx].qpos = qpos.copy()
        for jname, val in zip(self.joint_names, joint_pos):
            slider = self.slider_widgets.get(jname)
            if slider is not None:
                self._set_handle_value(slider, float(val))

    def _on_traj(
        self,
        qpos_replay: List[np.ndarray],
        body_pos_replay: List[np.ndarray],
        body_quat_replay: List[np.ndarray],
        body_lin_vel_replay: List[np.ndarray],
        body_ang_vel_replay: List[np.ndarray],
        site_pos_replay: List[np.ndarray],
        site_quat_replay: List[np.ndarray],
    ) -> None:
        self.qpos_replay = qpos_replay
        self.body_pos_replay = body_pos_replay
        self.body_quat_replay = body_quat_replay
        self.body_lin_vel_replay = body_lin_vel_replay
        self.body_ang_vel_replay = body_ang_vel_replay
        self.site_pos_replay = site_pos_replay
        self.site_quat_replay = site_quat_replay

    def _format_joint_label(self, joint_name: str) -> str:
        tokens = joint_name.split("_")
        if not tokens:
            return joint_name
        # Map common prefixes/tokens to abbreviations
        token_map = {"left": "L", "right": "R", "l": "L", "r": "R", "front": "F", "back": "B"}
        formatted_tokens: List[str] = []
        for tok in tokens:
            lower_tok = tok.lower()
            if lower_tok in token_map:
                formatted_tokens.append(token_map[lower_tok])
            elif lower_tok.isdigit():
                formatted_tokens.append(tok)
            else:
                formatted_tokens.append(lower_tok.capitalize())
        return " ".join(formatted_tokens)

    def _discover_end_effector_sites(self) -> List[str]:
        """Auto-discover end-effector sites by finding leaf bodies (bodies with no children).
        
        In a kinematic chain, end effectors are the terminal links that have no child bodies.
        This method finds all leaf bodies and returns the sites attached to them, excluding
        internal constraint sites (group >= 3 or referenced in equality constraints).
        """
        # Find all body parent IDs to identify which bodies have children
        parent_ids = set(self.model.body_parentid)
        
        # Find leaf bodies (bodies that are NOT parents of any other body)
        leaf_body_ids = []
        for body_id in range(self.model.nbody):
            if body_id not in parent_ids:
                leaf_body_ids.append(body_id)
        
        # Build set of sites referenced in equality CONNECT constraints (internal constraint points)
        equality_site_ids = set()
        for eq_id in range(self.model.neq):
            if self.model.eq_type[eq_id] == mujoco.mjtEq.mjEQ_CONNECT:
                # For CONNECT constraints, obj1id and obj2id are site IDs
                equality_site_ids.add(self.model.eq_obj1id[eq_id])
                equality_site_ids.add(self.model.eq_obj2id[eq_id])
        
        # End-effector keywords to identify legitimate end-effector sites even with high group numbers
        ee_keywords = ["foot", "hand", "gripper", "ee", "end_effector", "tip", "toe", "palm"]
        
        # For each leaf body, look for attached sites
        ee_sites = []
        for body_id in leaf_body_ids:
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if body_name is None or body_name == "world":
                continue
            
            # Find sites attached to this body
            for site_id in range(self.model.nsite):
                if self.model.site_bodyid[site_id] == body_id:
                    site_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, site_id)
                    if not site_name:
                        continue
                    
                    # Filter 1: Always exclude sites referenced in equality CONNECT constraints
                    if site_id in equality_site_ids:
                        continue
                    
                    # Filter 2: Exclude sites with group >= 3 UNLESS they have end-effector-like names
                    site_lower = site_name.lower()
                    is_ee_like = any(kw in site_lower for kw in ee_keywords)
                    
                    if self.model.site_group[site_id] >= 3 and not is_ee_like:
                        continue
                    
                    ee_sites.append(site_name)
        
        # If no sites found, fall back to leaf body names
        if not ee_sites:
            body_ee_keywords = ["foot", "hand", "calf", "leg", "lleg", "ankle", "toe", "gripper"]
            for body_id in leaf_body_ids:
                body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                if body_name is None or body_name == "world":
                    continue
                # Check if body name contains any end-effector keywords
                body_lower = body_name.lower()
                if any(kw in body_lower for kw in body_ee_keywords):
                    ee_sites.append(body_name)
            
            # If still no matches, just use all leaf bodies
            if not ee_sites:
                for body_id in leaf_body_ids:
                    body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                    if body_name and body_name != "world":
                        ee_sites.append(body_name)
        
        return ee_sites

    def _build_ee_panel(self) -> None:
        """Build end-effector pose saving/loading panel."""
        with self.server.gui.add_folder("👣 End Effector Poses"):
            # Use configured sites or auto-discover from leaf bodies
            sites = self.config.end_effector_sites
            if sites is None:
                sites = self._discover_end_effector_sites()
                if sites:
                    print(f"[Viser] Auto-discovered EE sites from leaf bodies: {sites}", flush=True)
            
            if not sites:
                self.server.gui.add_markdown("_No end-effector sites found._")
                return

            for site_name in sites[:8]:  # Limit to 8 sites for UI space
                short_name = site_name.replace("_center", "").replace("_site", "").replace("_", " ").title()
                
                save_btn = self.server.gui.add_button(f"Save {short_name}")
                @save_btn.on_click
                def _(_e: GuiEvent, sname=site_name) -> None:
                    with self.worker_lock:
                        try:
                            self.saved_ee_poses[sname] = self._get_site_transform(sname)
                            print(f"[Viser] Saved pose for {sname}", flush=True)
                        except Exception as exc:
                            print(f"[Viser] Failed to save {sname}: {exc}", flush=True)

                apply_btn = self.server.gui.add_button(f"Apply {short_name}")
                @apply_btn.on_click
                def _(_e: GuiEvent, sname=site_name) -> None:
                    if sname in self.saved_ee_poses:
                        self._align_to_pose(sname, self.saved_ee_poses[sname])
                    else:
                        print(f"[Viser] No saved pose for {sname}.", flush=True)

    def _align_to_pose(
        self, site_name: str, target_pose: Optional[np.ndarray]
    ) -> None:
        """Align the root body so that `site_name` matches `target_pose`."""
        if target_pose is None:
            print(f"[Viser] No saved pose for {site_name}.", flush=True)
            return
        with self.worker_lock:
            root_body = self.config.root_body
            root_t_curr = self._get_body_transform(root_body)
            site_t_curr = self._get_site_transform(site_name)
            aligned_root_t = target_pose @ np.linalg.inv(site_t_curr) @ root_t_curr
            self.data.qpos[:3] = aligned_root_t[:3, 3]
            self.data.qpos[3:7] = R.from_matrix(aligned_root_t[:3, :3]).as_quat(
                scalar_first=True
            )
            self._forward()
        print(f"[Viser] {site_name} aligned to saved pose.", flush=True)

    def _build_settings_panel(self) -> None:
        with self.server.gui.add_folder("⚙️ Settings"):
            self.mirror_checked = self.server.gui.add_checkbox("Mirror", True)
            self.rev_mirror_checked = self.server.gui.add_checkbox("Rev. Mirror", False)
            self.physics_enabled = self.server.gui.add_checkbox("Enable Physics", True)
            self.relative_frame_checked = self.server.gui.add_checkbox(
                "Save in Robot Frame",
                True,
            )
            self.collision_geom_checked = self.server.gui.add_checkbox(
                "Show Collision Geoms",
                False,
            )
            self.show_all_geoms = self.server.gui.add_checkbox(
                "Show All Geoms",
                False,
            )

        @self.mirror_checked.on_update
        def _(ev: GuiEvent) -> None:
            if id(ev.target) in self._updating_handles:
                return
            if bool(self.mirror_checked.value) and bool(self.rev_mirror_checked.value):
                self._set_handle_value(self.rev_mirror_checked, False)

        @self.rev_mirror_checked.on_update
        def _(ev: GuiEvent) -> None:
            if id(ev.target) in self._updating_handles:
                return
            if bool(self.rev_mirror_checked.value) and bool(self.mirror_checked.value):
                self._set_handle_value(self.mirror_checked, False)

        @self.collision_geom_checked.on_update
        def _(_event: GuiEvent) -> None:
            self._apply_geom_visibility()

        @self.show_all_geoms.on_update
        def _(_event: GuiEvent) -> None:
            self._apply_geom_visibility()

    def _base_keyframe_name(self, name: str) -> str:
        parts = name.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]
        return name

    def _generate_unique_name(self, base: str) -> str:
        existing_names = {kf.name for kf in self.keyframes}
        if base not in existing_names:
            return base
        suffix = 1
        while f"{base}_{suffix}" in existing_names:
            suffix += 1
        return f"{base}_{suffix}"


def main() -> None:
    """Command-line entry point for the keyframe editor."""
    parser = argparse.ArgumentParser(description="MuJoCo Keyframe Editor (Viser)")
    parser.add_argument(
        "xml_path",
        type=str,
        help="Path to the MuJoCo XML scene file.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="robot",
        help="Name for this project (used in save filenames).",
    )
    parser.add_argument(
        "--root-body",
        type=str,
        default=None,
        help="Name of the root body for ground alignment. If not specified, will be auto-detected.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="",
        help="Path to load existing keyframe data from.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="keyframes",
        help="Directory to save keyframe data.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file. If provided, overrides auto-detection.",
    )
    parser.add_argument(
        "--generate-config",
        type=str,
        default=None,
        help="Generate a YAML configuration file from the model and exit. Path where to save the config.",
    )
    args = parser.parse_args()

    # Handle config generation
    if args.generate_config:
        print(f"Generating configuration from {args.xml_path}...")
        config = EditorConfig.generate_from_model(args.xml_path, name=args.name)
        config.to_yaml(args.generate_config)
        print(f"✓ Configuration saved to {args.generate_config}")
        print("  Edit this file to customize settings, then use --config to load it.")
        return

    # Load config from file or create default
    if args.config:
        print(f"Loading configuration from {args.config}...")
        config = EditorConfig.from_yaml(args.config)
        # Override CLI args if provided
        if args.name != "robot":
            config.name = args.name
        if args.root_body:
            config.root_body = args.root_body
        if args.save_dir != "keyframes":
            config.save_dir = args.save_dir
    else:
        config = EditorConfig(
            name=args.name,
            root_body=args.root_body,
            save_dir=args.save_dir,
        )

    editor = ViserKeyframeEditor(
        args.xml_path,
        config=config,
        data_path=args.data,
    )

    # Get actual port from viser server
    try:
        port = editor.server.get_port()
    except Exception:
        port = 8080
    
    print(f"\n🚀 Keyframe Editor running!")
    print(f"   Open http://localhost:{port} in your browser\n")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()


