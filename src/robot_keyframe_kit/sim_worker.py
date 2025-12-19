"""Background worker thread for MuJoCo simulation state management.

This module contains the SimWorker class that handles physics simulation
in a background thread, allowing the UI to remain responsive.

NOTE: This file needs refactoring to work with raw MuJoCo model/data
instead of the toddlerbot-specific MuJoCoSim and Robot classes.
See PLAN.md for the required transformations.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Dict, List, Optional

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from .keyframe import Keyframe
from .config import EditorConfig


class SimWorker(threading.Thread):
    """Background worker to mutate MuJoCo state and generate replay arrays.

    Handles physics simulation in a background thread, using threading
    instead of Qt. Use the provided lock to synchronize with the UI thread.
    
    NOTE: This is adapted from the original toddlerbot implementation.
    The sim/robot parameters have been replaced with raw MuJoCo model/data.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        config: EditorConfig,
        lock: threading.Lock,
        *,
        joint_names: List[str],
        actuator_names: List[str],
        default_joint_angles: Dict[str, float],
        on_state: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], None]] = None,
        on_traj: Optional[
            Callable[
                [
                    List[np.ndarray],
                    List[np.ndarray],
                    List[np.ndarray],
                    List[np.ndarray],
                    List[np.ndarray],
                    List[np.ndarray],
                    List[np.ndarray],
                ],
                None,
            ]
        ] = None,
    ) -> None:
        super().__init__(daemon=True)
        self.model = model
        self.data = data
        self.config = config
        self.lock = lock
        self.joint_names = joint_names
        self.actuator_names = actuator_names
        self.on_state = on_state
        self.on_traj = on_traj

        self.running = True
        self.is_testing = False
        self.is_qpos_traj = False
        self.is_relative_frame = True

        self.update_joint_angles_requested = False
        self.joint_angles_to_update = default_joint_angles.copy()

        self.update_qpos_requested = False
        self.qpos_to_update = model.qpos0.copy()

        # Stepping state
        self.keyframe_test_counter = -1
        self.keyframe_test_dt = 0.0

        self.traj_test_counter = -1
        self.action_traj: Optional[List[np.ndarray]] = None
        self.traj_test_dt = 0.0
        self.traj_physics_enabled = False

        # Replay buffers
        self.qpos_replay: List[np.ndarray] = []
        self.body_pos_replay: List[np.ndarray] = []
        self.body_quat_replay: List[np.ndarray] = []
        self.body_lin_vel_replay: List[np.ndarray] = []
        self.body_ang_vel_replay: List[np.ndarray] = []
        self.site_pos_replay: List[np.ndarray] = []
        self.site_quat_replay: List[np.ndarray] = []

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

    def _get_joint_angles(self) -> Dict[str, float]:
        """Get current joint angles as a dictionary."""
        joint_angles: Dict[str, float] = {}
        for name in self.joint_names:
            joint_angles[name] = self.data.joint(name).qpos.item()
        return joint_angles

    def _get_joint_angles_array(self) -> np.ndarray:
        """Get current joint angles as an array."""
        return np.array([self.data.joint(name).qpos.item() for name in self.joint_names], dtype=np.float32)

    def _get_actuator_values_array(self) -> np.ndarray:
        """Get current actuator positions as an array."""
        return np.array([self.data.actuator(name).length.item() for name in self.actuator_names], dtype=np.float32)

    def _set_joint_angles(self, joint_angles: Dict[str, float]) -> None:
        """Set joint angles from a dictionary."""
        for name, value in joint_angles.items():
            self.data.joint(name).qpos = value

    def _forward(self) -> None:
        """Run forward kinematics."""
        mujoco.mj_forward(self.model, self.data)

    def _step(self) -> None:
        """Step physics simulation."""
        mujoco.mj_step(self.model, self.data)

    # ----- Requests from UI -----
    def request_state_data(self):
        with self.lock:
            if self.is_testing:
                return
            motor_pos = self._get_actuator_values_array()
            joint_pos = self._get_joint_angles_array()
            qpos = self.data.qpos.copy()
        if self.on_state:
            self.on_state(motor_pos, joint_pos, qpos)

    def update_joint_angles(self, joint_angles_to_update: Dict[str, float]):
        self.update_joint_angles_requested = True
        self.joint_angles_to_update = joint_angles_to_update.copy()

    def update_qpos(self, qpos: np.ndarray):
        self.update_qpos_requested = True
        self.qpos_to_update = qpos.copy()

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
            ee_keywords = ["foot", "hand", "calf", "leg", "lleg", "ankle", "toe", "gripper"]
            for body_id in leaf_body_ids:
                body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                if body_name is None or body_name == "world":
                    continue
                # Check if body name contains any end-effector keywords
                body_lower = body_name.lower()
                if any(kw in body_lower for kw in ee_keywords):
                    ee_sites.append(body_name)
            
            # If still no matches, just use all leaf bodies
            if not ee_sites:
                for body_id in leaf_body_ids:
                    body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                    if body_name and body_name != "world":
                        ee_sites.append(body_name)
        
        return ee_sites

    def request_on_ground(self):
        """Place the robot on the ground by finding the lowest end-effector."""
        with self.lock:
            if self.is_testing:
                return
            
            root_body = self.config.root_body
            torso_t_curr = self._get_body_transform(root_body)
            
            # Find the lowest end-effector site
            site_z_min = float("inf")
            min_site = None
            
            # Use configured end-effector sites or auto-discover from leaf bodies
            sites_to_check = self.config.end_effector_sites
            if sites_to_check is None:
                sites_to_check = self._discover_end_effector_sites()
                if sites_to_check:
                    print(f"[Ground] Auto-discovered EE sites from leaf bodies: {sites_to_check}", flush=True)
            
            for site_name in sites_to_check:
                try:
                    curr_transform = self._get_site_transform(site_name)
                    if curr_transform[2, 3] < site_z_min:
                        site_z_min = curr_transform[2, 3]
                        min_site = site_name
                except Exception:
                    continue  # Site doesn't exist

            if min_site is None:
                print("[Ground] No end-effector sites found", flush=True)
                return

            foot_t = self._get_site_transform(min_site)
            aligned_torso_t = foot_t @ np.linalg.inv(foot_t) @ torso_t_curr
            # Translate torso so the min foot touches z=0 plane (approx)
            dz = foot_t[2, 3]
            aligned_torso_t[2, 3] -= dz
            self.data.qpos[:3] = aligned_torso_t[:3, 3]
            self.data.qpos[3:7] = R.from_matrix(aligned_torso_t[:3, :3]).as_quat(
                scalar_first=True
            )
            self._forward()
            print(
                f"[Ground] Placed {min_site} on ground (moved down {dz:.3f}m)",
                flush=True,
            )

            # Trigger UI update callback so sliders refresh
            motor_pos = self._get_actuator_values_array()
            joint_pos = self._get_joint_angles_array()
            qpos = self.data.qpos.copy()
        if self.on_state:
            self.on_state(motor_pos, joint_pos, qpos)

    def request_keyframe_test(self, keyframe: Keyframe, dt: float):
        with self.lock:
            if self.is_testing:
                return
            self.keyframe_test_counter = -1
            self.traj_test_counter = -1
            self.data.qpos = keyframe.qpos.copy()
            self._forward()
            self.data.ctrl[:] = keyframe.motor_pos.copy()
            self.keyframe_test_dt = dt
            self.keyframe_test_counter = 0
            self.is_testing = True

    def request_trajectory_test(
        self,
        qpos_start: np.ndarray,
        traj: List[np.ndarray],
        dt: float,
        physics_enabled: bool,
        *,
        is_qpos_traj: bool = False,
        is_relative_frame: bool = True,
    ):
        with self.lock:
            if self.is_testing:
                print(
                    "[Viser] Worker: request_trajectory_test ignored (already testing)",
                    flush=True,
                )
                return
            self.keyframe_test_counter = -1
            self.traj_test_counter = -1

            self.data.qpos = qpos_start.copy()
            self.data.qvel[:] = 0
            self.data.ctrl[:] = 0
            self._forward()

            self.action_traj = traj
            self.traj_test_dt = dt
            self.traj_physics_enabled = physics_enabled
            self.traj_test_counter = 0
            self.is_testing = True
            self.is_qpos_traj = is_qpos_traj
            self.is_relative_frame = is_relative_frame

            try:
                print(
                    f"[Viser] Worker: start trajectory test: len={len(traj)}, dt={dt}, physics={physics_enabled}, qpos={is_qpos_traj}, rel={is_relative_frame}",
                    flush=True,
                )
            except Exception:
                pass

            # Clear replay
            self.qpos_replay.clear()
            self.body_pos_replay.clear()
            self.body_quat_replay.clear()
            self.body_lin_vel_replay.clear()
            self.body_ang_vel_replay.clear()
            self.site_pos_replay.clear()
            self.site_quat_replay.clear()

    def stop(self):
        self.running = False

    # ----- Main loop -----
    def run(self) -> None:
        while self.running:
            if self.update_qpos_requested:
                with self.lock:
                    self.is_testing = False
                    self.keyframe_test_counter = -1
                    self.traj_test_counter = -1
                    self.data.qpos = self.qpos_to_update.copy()
                    self._forward()
                    self.update_qpos_requested = False
                time.sleep(0)  # yield
                continue

            if self.update_joint_angles_requested:
                with self.lock:
                    self.is_testing = False
                    self.keyframe_test_counter = -1
                    self.traj_test_counter = -1
                    joint_angles = self._get_joint_angles()
                    joint_angles.update(self.joint_angles_to_update)
                    self._set_joint_angles(joint_angles)
                    self._forward()
                    self.update_joint_angles_requested = False
                time.sleep(0)  # yield
                continue

            if 0 <= self.keyframe_test_counter <= 100:
                with self.lock:
                    if self.keyframe_test_counter == 100:
                        self.keyframe_test_counter = -1
                        self.is_testing = False
                    else:
                        self._step()
                        self.keyframe_test_counter += 1
                time.sleep(self.keyframe_test_dt)
                continue

            # Trajectory test
            if self.traj_test_counter >= 0 and self.action_traj is not None:
                # Check stop
                with self.lock:
                    trajectory_running = self.is_testing
                    current_counter = self.traj_test_counter
                    traj_len = len(self.action_traj)
                if current_counter == 0:
                    try:
                        print(
                            f"[Viser] Worker: stepping trajectory... len={traj_len}, dt={self.traj_test_dt}, physics={self.traj_physics_enabled}",
                            flush=True,
                        )
                    except Exception:
                        pass
                if not trajectory_running:
                    # Emit and clear
                    if self.on_traj:
                        self.on_traj(
                            self.qpos_replay.copy(),
                            self.body_pos_replay.copy(),
                            self.body_quat_replay.copy(),
                            self.body_lin_vel_replay.copy(),
                            self.body_ang_vel_replay.copy(),
                            self.site_pos_replay.copy(),
                            self.site_quat_replay.copy(),
                        )
                    with self.lock:
                        self.traj_test_counter = -1
                        self.keyframe_test_counter = -1
                        self.action_traj = None
                        self.qpos_replay.clear()
                        self.body_pos_replay.clear()
                        self.body_quat_replay.clear()
                        self.body_lin_vel_replay.clear()
                        self.body_ang_vel_replay.clear()
                        self.site_pos_replay.clear()
                        self.site_quat_replay.clear()
                    time.sleep(0)
                    continue

                # If trajectory is exhausted or empty, finalize and emit once
                if current_counter >= traj_len:
                    try:
                        print(
                            f"[Viser] Worker: trajectory complete. frames={len(self.qpos_replay)}",
                            flush=True,
                        )
                    except Exception:
                        pass
                    if self.on_traj:
                        self.on_traj(
                            self.qpos_replay.copy(),
                            self.body_pos_replay.copy(),
                            self.body_quat_replay.copy(),
                            self.body_lin_vel_replay.copy(),
                            self.body_ang_vel_replay.copy(),
                            self.site_pos_replay.copy(),
                            self.site_quat_replay.copy(),
                        )
                    with self.lock:
                        self.traj_test_counter = -1
                        self.keyframe_test_counter = -1
                        self.action_traj = None
                        self.is_testing = False
                        self.qpos_replay.clear()
                        self.body_pos_replay.clear()
                        self.body_quat_replay.clear()
                        self.body_lin_vel_replay.clear()
                        self.body_ang_vel_replay.clear()
                        self.site_pos_replay.clear()
                        self.site_quat_replay.clear()
                    time.sleep(0)
                    continue

                # Step one action
                t1 = time.monotonic()
                with self.lock:
                    if self.is_qpos_traj:
                        qpos_goal = self.action_traj[current_counter]
                        self.data.qpos[:] = qpos_goal
                        self._forward()
                    else:
                        target = self.action_traj[current_counter]
                        if self.traj_physics_enabled:
                            # With physics enabled, use PD control for stiff joints
                            # Get current joint positions and velocities
                            current_pos = self._get_actuator_values_array()
                            current_vel = np.zeros_like(current_pos)
                            for i, act_name in enumerate(self.actuator_names):
                                try:
                                    jnt_id = self.model.actuator(act_name).trnid[0]
                                    qvel_adr = self.model.jnt_dofadr[jnt_id]
                                    current_vel[i] = self.data.qvel[qvel_adr]
                                except Exception:
                                    pass
                            
                            # PD control: torque = kp * (target - current) - kd * velocity
                            kp = self.config.kp
                            kd = self.config.kd
                            pos_error = target - current_pos
                            ctrl_signal = kp * pos_error - kd * current_vel
                            self.data.ctrl[:] = ctrl_signal
                            self._step()
                        else:
                            # Without physics, directly apply joint angles for visible motion
                            # Set actuator positions directly
                            self.data.ctrl[:] = target
                            self._forward()

                    # Record
                    qpos_data = self.data.qpos.copy()
                    root_body = self.config.root_body
                    torso_rot = R.from_quat(
                        self.data.body(root_body).xquat.copy(), scalar_first=True
                    )
                    r_inv = torso_rot.inv()

                    if self.is_relative_frame:
                        body_pos_world = np.array(self.data.xpos, dtype=np.float32)
                        body_quat_world = np.array(
                            self.data.xquat, dtype=np.float32
                        )
                        body_pos = []
                        body_quat = []
                        for i in range(self.model.nbody):
                            p = body_pos_world[i]
                            q = body_quat_world[i]
                            body_pos.append(
                                r_inv.apply(p - self.data.body(root_body).xpos)
                            )
                            # Convert world quat to torso-relative by q_rel = q_inv(torso)*q_body
                            q_rel = (r_inv * R.from_quat(q, scalar_first=True)).as_quat(
                                scalar_first=True
                            )
                            body_quat.append(q_rel)
                        body_pos = np.array(body_pos, dtype=np.float32)
                        body_quat = np.array(body_quat, dtype=np.float32)
                        body_lin_vel_world = np.array(
                            self.data.cvel[:, 3:], dtype=np.float32
                        )
                        body_ang_vel_world = np.array(
                            self.data.cvel[:, :3], dtype=np.float32
                        )
                        body_lin_vel = r_inv.apply(body_lin_vel_world)
                        body_ang_vel = r_inv.apply(body_ang_vel_world)
                        
                        # Record end-effector sites
                        site_pos = []
                        site_quat = []
                        if self.config.end_effector_sites:
                            for sname in self.config.end_effector_sites:
                                try:
                                    ee_pos_world = self.data.site(sname).xpos.copy()
                                    ee_mat = self.data.site(sname).xmat.reshape(3, 3)
                                    ee_quat_world = R.from_matrix(ee_mat).as_quat(
                                        scalar_first=True
                                    )
                                    site_pos.append(ee_pos_world)
                                    site_quat.append(ee_quat_world)
                                except Exception:
                                    continue
                    else:
                        body_pos = np.array(self.data.xpos, dtype=np.float32)
                        body_quat = np.array(self.data.xquat, dtype=np.float32)
                        body_lin_vel = np.array(
                            self.data.cvel[:, 3:], dtype=np.float32
                        )
                        body_ang_vel = np.array(
                            self.data.cvel[:, :3], dtype=np.float32
                        )
                        site_pos = []
                        site_quat = []
                        if self.config.end_effector_sites:
                            for sname in self.config.end_effector_sites:
                                try:
                                    ee_pos_world = self.data.site(sname).xpos.copy()
                                    ee_mat = self.data.site(sname).xmat.reshape(3, 3)
                                    ee_quat_world = R.from_matrix(ee_mat).as_quat(
                                        scalar_first=True
                                    )
                                    site_pos.append(ee_pos_world)
                                    site_quat.append(ee_quat_world)
                                except Exception:
                                    continue

                # Append outside of lock
                self.qpos_replay.append(qpos_data)
                self.body_pos_replay.append(body_pos)
                self.body_quat_replay.append(body_quat)
                self.body_lin_vel_replay.append(body_lin_vel)
                self.body_ang_vel_replay.append(body_ang_vel)
                self.site_pos_replay.append(np.array(site_pos, dtype=np.float32) if site_pos else np.array([], dtype=np.float32))
                self.site_quat_replay.append(np.array(site_quat, dtype=np.float32) if site_quat else np.array([], dtype=np.float32))

                self.traj_test_counter += 1
                try:
                    if (
                        self.traj_test_counter
                        % max(1, int(0.5 / max(self.traj_test_dt, 1e-6)))
                        == 0
                    ):
                        print(
                            f"[Viser] Worker: progressed to step {self.traj_test_counter}/{traj_len}",
                            flush=True,
                        )
                except Exception:
                    pass
                t2 = time.monotonic()
                dt_left = self.traj_test_dt - (t2 - t1)
                if dt_left > 0:
                    time.sleep(dt_left)
                else:
                    time.sleep(0.001)
                continue

            time.sleep(0.005)


