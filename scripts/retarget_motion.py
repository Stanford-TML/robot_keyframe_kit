#!/usr/bin/env python3
"""Optimization-based robot-to-robot retargeting for keyframe-kit motions.

Pipeline summary:
1. Build canonical source tasks (root/chest/4 EE + limb bend proxies).
2. Map to target with non-uniform scaling (legs/arms/torso/root).
3. Solve frame-wise IK for warm start.
4. Run windowed temporal least-squares optimization with:
   - tracking terms over time,
   - contact-phase stick constraints with transition release,
   - velocity/acceleration smoothness,
   - naturalness priors (shoulder-roll suppression on G1).
5. Export editor-compatible .lz4 and report quantitative metrics.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import mujoco
import numpy as np
import scipy.sparse as sp
from scipy.signal import savgol_filter
from scipy.sparse.linalg import lsqr
from scipy.spatial.transform import Rotation as R

try:
    import mink
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "mink is required for this script. Install with: pip install mink==0.0.13"
    ) from exc


FOOT_KEYWORDS = ("foot", "ankle", "toe", "heel")
HAND_KEYWORDS = ("hand", "wrist", "palm", "gripper", "finger", "thumb")
EE_KEYWORDS = FOOT_KEYWORDS + HAND_KEYWORDS + ("tip", "end_effector", "ee")


@dataclass
class CanonicalEE:
    left_foot: str
    right_foot: str
    left_hand: str
    right_hand: str

    def as_dict(self) -> Dict[str, str]:
        return {
            "left_foot": self.left_foot,
            "right_foot": self.right_foot,
            "left_hand": self.left_hand,
            "right_hand": self.right_hand,
        }


@dataclass
class ContactPlan:
    active: np.ndarray  # (T,) bool
    weight: np.ndarray  # (T,) float
    anchor: np.ndarray  # (T,3) float


@dataclass
class TemporalWeights:
    root_pos: float = 5.0
    root_rot: float = 3.0
    chest_pos: float = 2.5
    chest_rot: float = 1.6
    ee_pos_foot: float = 8.0
    ee_pos_hand: float = 5.0
    ee_rot_foot: float = 0.6
    ee_rot_hand: float = 0.2
    hint_knee: float = 1.2
    hint_elbow: float = 1.0
    hint_shoulder: float = 1.15
    contact_foot: float = 26.0
    contact_hand: float = 18.0
    vel: float = 0.8
    acc: float = 2.4
    reg: float = 0.12
    shoulder_roll: float = 1.7
    shoulder_roll_vel: float = 0.7
    arm_pose: float = 1.4
    hand_rot_startup_frames: int = 120
    wrist_neutral: float = 0.9
    wrist_neutral_startup_boost: float = 3.0
    wrist_neutral_roll_yaw: float = 30.0
    wrist_neutral_pitch: float = 10.0
    wrist_strong_vel: float = 5.0
    seq_anchor_joint: float = 6.0
    seq_anchor_root: float = 1.5
    symmetry_hand: float = 2.0
    symmetry_foot: float = 1.2
    symmetry_hint: float = 1.0
    arm_lr_mirror: float = 2.2


def _is_left(name: str) -> bool:
    s = name.lower()
    return "left" in s or s.startswith("l_") or s.endswith("_l") or "_l_" in s


def _is_right(name: str) -> bool:
    s = name.lower()
    return "right" in s or s.startswith("r_") or s.endswith("_r") or "_r_" in s


def _limb_type(name: str) -> Optional[str]:
    s = name.lower()
    if any(k in s for k in FOOT_KEYWORDS):
        return "foot"
    if any(k in s for k in HAND_KEYWORDS):
        return "hand"
    return None


def _safe_quat_wxyz(q: np.ndarray) -> np.ndarray:
    out = np.asarray(q, dtype=np.float64).copy()
    n = float(np.linalg.norm(out))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return out / n


def _auto_detect_root_body(model: mujoco.MjModel) -> str:
    for body_id in range(model.nbody):
        if body_id == 0:
            continue
        if int(model.body_parentid[body_id]) == 0:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if name and name != "world":
                return name
    raise ValueError("Failed to auto-detect root body")


def _discover_ee_frames(model: mujoco.MjModel) -> List[Tuple[str, str]]:
    parent_ids = {int(i) for i in model.body_parentid}
    leaf_body_ids = [i for i in range(model.nbody) if i not in parent_ids]
    candidates: List[Tuple[str, str, int]] = []

    for body_id in leaf_body_ids:
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if not body_name or body_name == "world":
            continue
        for site_id in range(model.nsite):
            if int(model.site_bodyid[site_id]) != body_id:
                continue
            site_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id)
            if not site_name:
                continue
            if any(k in site_name.lower() for k in EE_KEYWORDS):
                candidates.append((site_name, "site", 4))

    for body_id in leaf_body_ids:
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if not body_name or body_name == "world":
            continue
        if any(k in body_name.lower() for k in EE_KEYWORDS):
            candidates.append((body_name, "body", 3))

    if not candidates:
        for body_id in leaf_body_ids:
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if body_name and body_name != "world":
                candidates.append((body_name, "body", 1))

    candidates.sort(key=lambda x: (-x[2], x[0]))
    return [(n, t) for n, t, _ in candidates]


def _canonicalize_ee(model: mujoco.MjModel) -> CanonicalEE:
    slots: Dict[str, Optional[Tuple[str, str]]] = {
        "left_foot": None,
        "right_foot": None,
        "left_hand": None,
        "right_hand": None,
    }

    def _score(name: str, frame_type: str, limb: str) -> int:
        low = name.lower()
        score = 100 if frame_type == "site" else 0
        if "center" in low or "tip" in low:
            score += 30
        if "link" not in low:
            score += 10
        if limb == "hand":
            # Prefer palm/hand contact frames over distal wrist articulation links.
            if "palm" in low or ("hand" in low and "wrist" not in low):
                score += 40
            if "wrist_yaw" in low:
                score -= 30
            elif "wrist_pitch" in low:
                score -= 20
            elif "wrist_roll" in low:
                score += 5
        return score

    for name, frame_type in _discover_ee_frames(model):
        limb = _limb_type(name)
        if limb is None:
            continue
        side = "left" if _is_left(name) else "right" if _is_right(name) else None
        if side is None:
            continue
        key = f"{side}_{limb}"
        cur = slots[key]
        if cur is None or _score(name, frame_type, limb) > _score(cur[0], cur[1], limb):
            slots[key] = (name, frame_type)

    missing = [k for k, v in slots.items() if v is None]
    if missing:
        raise RuntimeError(f"Failed to detect canonical EEs: missing {missing}")

    # If a hand EE lands on distal wrist yaw/pitch links (common in hand-less models),
    # pull it back to wrist_roll when available to avoid unconstrained wrist articulation artifacts.
    for side in ("left", "right"):
        key = f"{side}_hand"
        cur_name, _cur_type = slots[key]  # type: ignore[misc]
        low = cur_name.lower()
        if "wrist_yaw" in low or "wrist_pitch" in low:
            prox = _find_body_name(model, (side, "wrist", "roll"))
            if prox:
                slots[key] = (prox, "body")

    return CanonicalEE(
        left_foot=slots["left_foot"][0],  # type: ignore[index]
        right_foot=slots["right_foot"][0],  # type: ignore[index]
        left_hand=slots["left_hand"][0],  # type: ignore[index]
        right_hand=slots["right_hand"][0],  # type: ignore[index]
    )


def _find_body_name(
    model: mujoco.MjModel, required_tokens: Sequence[str]
) -> Optional[str]:
    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if not name:
            continue
        low = name.lower()
        if all(tok in low for tok in required_tokens):
            return name
    return None


def _find_limb_hints(model: mujoco.MjModel) -> Dict[str, str]:
    hints: Dict[str, str] = {}
    for side in ("left", "right"):
        knee = _find_body_name(model, (side, "knee"))
        if knee:
            hints[f"{side}_knee"] = knee
        elbow = _find_body_name(model, (side, "elbow"))
        if elbow:
            hints[f"{side}_elbow"] = elbow
        shoulder = _find_body_name(model, (side, "shoulder", "yaw"))
        if shoulder is None:
            shoulder = _find_body_name(model, (side, "shoulder", "pitch"))
        if shoulder is None:
            shoulder = _find_body_name(model, (side, "shoulder", "roll"))
        if shoulder is None:
            shoulder = _find_body_name(model, (side, "shoulder"))
        if shoulder:
            hints[f"{side}_shoulder"] = shoulder
    return hints


def _find_chest_frame(model: mujoco.MjModel, root_body: str) -> Optional[str]:
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body)
    if root_id < 0:
        return None
    token_priority = (
        ("chest",),
        ("torso",),
        ("upper", "torso"),
        ("trunk",),
        ("waist",),
        ("spine",),
    )
    for toks in token_priority:
        for body_id in range(model.nbody):
            if body_id in (0, root_id):
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if not name:
                continue
            low = name.lower()
            if all(t in low for t in toks):
                return name
    return None


def _frame_pose(
    model: mujoco.MjModel, data: mujoco.MjData, frame_name: str
) -> Tuple[np.ndarray, np.ndarray, str]:
    frame_type, frame_id = _resolve_frame(model, frame_name)
    if frame_type == "site":
        pos = data.site_xpos[frame_id].copy().astype(np.float64)
        quat = R.from_matrix(data.site_xmat[frame_id].reshape(3, 3)).as_quat(
            scalar_first=True
        )
        return pos, _safe_quat_wxyz(quat), "site"
    if frame_type == "body":
        pos = data.xpos[frame_id].copy().astype(np.float64)
        quat = R.from_matrix(data.xmat[frame_id].reshape(3, 3)).as_quat(
            scalar_first=True
        )
        return pos, _safe_quat_wxyz(quat), "body"
    raise ValueError(f"Unknown site/body frame: {frame_name}")


@lru_cache(maxsize=512)
def _resolve_frame(model: mujoco.MjModel, frame_name: str) -> Tuple[str, int]:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, frame_name)
    if site_id >= 0:
        return "site", int(site_id)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, frame_name)
    if body_id >= 0:
        return "body", int(body_id)
    raise ValueError(f"Unknown site/body frame: {frame_name}")


def _frame_jacobian(
    model: mujoco.MjModel, data: mujoco.MjData, frame_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    frame_type, frame_id = _resolve_frame(model, frame_name)
    if frame_type == "site":
        mujoco.mj_jacSite(model, data, jacp, jacr, frame_id)
        return jacp, jacr
    if frame_type == "body":
        mujoco.mj_jacBody(model, data, jacp, jacr, frame_id)
        return jacp, jacr
    raise ValueError(f"Unknown frame for jacobian: {frame_name}")


def _to_local_pose(
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    world_pos: np.ndarray,
    world_quat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    rr = R.from_quat(root_quat, scalar_first=True)
    rw = R.from_quat(world_quat, scalar_first=True)
    return rr.inv().apply(world_pos - root_pos), _safe_quat_wxyz(
        (rr.inv() * rw).as_quat(scalar_first=True)
    )


def _to_world_pose(
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    local_pos: np.ndarray,
    local_quat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    rr = R.from_quat(root_quat, scalar_first=True)
    rl = R.from_quat(local_quat, scalar_first=True)
    return root_pos + rr.apply(local_pos), _safe_quat_wxyz(
        (rr * rl).as_quat(scalar_first=True)
    )


def _extract_qpos_sequence(motion_data: dict) -> np.ndarray:
    if "qpos" in motion_data:
        q = np.asarray(motion_data["qpos"], dtype=np.float64)
        if q.ndim == 2:
            return q
    keyframes = motion_data.get("keyframes")
    if keyframes:
        q = [np.asarray(k["qpos"], dtype=np.float64) for k in keyframes if "qpos" in k]
        if q:
            return np.stack(q, axis=0)
    raise ValueError("Motion file has neither dense qpos nor keyframe qpos")


def _extract_dt(motion_data: dict, default_dt: float) -> float:
    t = np.asarray(motion_data.get("time", []), dtype=np.float64)
    if t.ndim == 1 and t.size > 1:
        dt = float(np.median(np.diff(t)))
        if np.isfinite(dt) and dt > 1e-6:
            return dt
    return default_dt


def _home_qpos(model: mujoco.MjModel) -> np.ndarray:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0 and model.nkey > 0:
        key_q = np.asarray(model.key_qpos, dtype=np.float64).reshape(
            model.nkey, model.nq
        )
        return key_q[key_id].copy()
    if model.nkey > 0:
        key_q = np.asarray(model.key_qpos, dtype=np.float64).reshape(
            model.nkey, model.nq
        )
        return key_q[0].copy()
    return np.asarray(model.qpos0, dtype=np.float64).copy()


def _compute_limb_scales(
    src_root_pos: np.ndarray,
    src_ee_pos: Dict[str, np.ndarray],
    src_chest_pos: Optional[np.ndarray],
    tgt_root_pos: np.ndarray,
    tgt_ee_pos: Dict[str, np.ndarray],
    tgt_chest_pos: Optional[np.ndarray],
) -> Dict[str, float]:
    def _mean_dist(root: np.ndarray, points: List[np.ndarray]) -> float:
        ds = [float(np.linalg.norm(p - root)) for p in points]
        return float(np.mean(ds)) if ds else 1.0

    src_leg = _mean_dist(
        src_root_pos, [src_ee_pos["left_foot"], src_ee_pos["right_foot"]]
    )
    tgt_leg = _mean_dist(
        tgt_root_pos, [tgt_ee_pos["left_foot"], tgt_ee_pos["right_foot"]]
    )
    src_arm = _mean_dist(
        src_root_pos, [src_ee_pos["left_hand"], src_ee_pos["right_hand"]]
    )
    tgt_arm = _mean_dist(
        tgt_root_pos, [tgt_ee_pos["left_hand"], tgt_ee_pos["right_hand"]]
    )

    leg_scale = float(np.clip(tgt_leg / max(src_leg, 1e-8), 0.35, 3.0))
    arm_scale = float(np.clip(tgt_arm / max(src_arm, 1e-8), 0.35, 3.0))
    if src_chest_pos is not None and tgt_chest_pos is not None:
        src_torso = float(np.linalg.norm(src_chest_pos - src_root_pos))
        tgt_torso = float(np.linalg.norm(tgt_chest_pos - tgt_root_pos))
        torso_scale = float(np.clip(tgt_torso / max(src_torso, 1e-8), 0.35, 3.0))
    else:
        torso_scale = float(np.clip(0.5 * (leg_scale + arm_scale), 0.35, 3.0))
    root_xy_scale = float(np.clip(0.7 * torso_scale + 0.3 * leg_scale, 0.35, 3.0))
    root_z_scale = leg_scale
    return {
        "leg": leg_scale,
        "arm": arm_scale,
        "torso": torso_scale,
        "root_xy": root_xy_scale,
        "root_z": root_z_scale,
    }


def _compute_motor_and_joint_pos(
    model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    prev_q = data.qpos.copy()
    prev_v = data.qvel.copy()
    try:
        data.qpos[:] = np.asarray(qpos, dtype=np.float64)
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        motor_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(model.nu)
        ]
        motor_pos = np.array(
            [
                data.actuator(name).length.item()
                for name in motor_names
                if name is not None
            ],
            dtype=np.float32,
        )
        joint_pos_vals: List[float] = []
        for act_id in range(model.nu):
            j_id = int(model.actuator_trnid[act_id, 0])
            if j_id < 0:
                continue
            qadr = int(model.jnt_qposadr[j_id])
            joint_pos_vals.append(float(qpos[qadr]))
        joint_pos = np.asarray(joint_pos_vals, dtype=np.float32)
    finally:
        data.qpos[:] = prev_q
        data.qvel[:] = prev_v
        mujoco.mj_forward(model, data)
    return motor_pos, joint_pos


def _select_ik_solver() -> str:
    try:
        import qpsolvers  # type: ignore

        available = set(getattr(qpsolvers, "available_solvers", []))
    except Exception:
        available = set()
    for candidate in ("daqp", "quadprog", "osqp", "jaxopt_osqp"):
        if not available or candidate in available:
            return candidate
    return "quadprog"


def _make_mink_target_pose(target_pos: np.ndarray, target_wxyz: np.ndarray) -> object:
    q = _safe_quat_wxyz(target_wxyz)
    rot = R.from_quat(q, scalar_first=True).as_matrix()
    return mink.SE3.from_rotation_and_translation(
        mink.SO3.from_matrix(rot), np.asarray(target_pos, dtype=np.float64)
    )


def _detect_contact_plan(
    ee_pos: np.ndarray,
    dt: float,
    z_on: float,
    z_off: float,
    v_on: float,
    v_off: float,
    min_frames: int,
    edge_release: int,
    max_phase_slip: float,
) -> ContactPlan:
    t = ee_pos.shape[0]
    vel = np.zeros(t, dtype=np.float64)
    if t > 1:
        vel[1:] = np.linalg.norm(np.diff(ee_pos, axis=0), axis=1) / max(dt, 1e-6)
    active = np.zeros(t, dtype=bool)
    state = False
    for i in range(t):
        z = float(ee_pos[i, 2])
        v = float(vel[i])
        if not state:
            if z < z_on and v < v_on:
                state = True
        else:
            if z > z_off or v > v_off:
                state = False
        active[i] = state

    # Remove short phases and short gaps (temporal reasoning).
    i = 0
    while i < t:
        j = i
        while j < t and active[j] == active[i]:
            j += 1
        run = j - i
        if run < min_frames:
            active[i:j] = not active[i]
        i = j

    anchor = ee_pos.copy()
    weight = np.zeros(t, dtype=np.float64)
    i = 0
    while i < t:
        if not active[i]:
            i += 1
            continue
        j = i
        while j < t and active[j]:
            j += 1
        seg_len = j - i
        start = ee_pos[i].copy()
        end = ee_pos[j - 1].copy()
        delta = end - start
        n = float(np.linalg.norm(delta))
        if n > max_phase_slip and n > 1e-8:
            end = start + delta / n * max_phase_slip
        for k in range(i, j):
            if seg_len <= 1:
                alpha = 0.0
            else:
                alpha = float(k - i) / float(seg_len - 1)
            anchor[k] = (1.0 - alpha) * start + alpha * end
            edge = min(k - i, j - 1 - k)
            if edge_release <= 0:
                ramp = 1.0
            else:
                ramp = min(1.0, (edge + 1) / float(edge_release + 1))
            weight[k] = 0.2 + 0.8 * ramp
        i = j
    return ContactPlan(active=active, weight=weight, anchor=anchor)


def _orientation_error_vec(
    desired_wxyz: np.ndarray, current_wxyz: np.ndarray
) -> np.ndarray:
    rd = R.from_quat(_safe_quat_wxyz(desired_wxyz), scalar_first=True)
    rc = R.from_quat(_safe_quat_wxyz(current_wxyz), scalar_first=True)
    return (rd * rc.inv()).as_rotvec().astype(np.float64)


def _infer_robot_name_from_xml(xml_path: str) -> str:
    low = os.path.normpath(xml_path).lower()
    if "unitree_g1" in low or "/g1/" in low:
        return "unitree_g1"
    if "booster_t1" in low or "/t1/" in low:
        return "booster_t1"
    return os.path.splitext(os.path.basename(xml_path))[0]


def _free_joint_and_hinge_vars(
    model: mujoco.MjModel,
) -> Tuple[List[int], List[int], List[Tuple[int, float, float, int]]]:
    qpos_idx: List[int] = []
    dof_idx: List[int] = []
    limits: List[Tuple[int, float, float, int]] = []
    for j in range(model.njnt):
        jt = int(model.jnt_type[j])
        qadr = int(model.jnt_qposadr[j])
        dadr = int(model.jnt_dofadr[j])
        if jt == int(mujoco.mjtJoint.mjJNT_FREE):
            # Optimize root xyz only; keep root quaternion from IK warm start.
            for a in range(3):
                qpos_idx.append(qadr + a)
                dof_idx.append(dadr + a)
                limits.append((len(qpos_idx) - 1, -np.inf, np.inf, qadr + a))
        elif jt in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            qpos_idx.append(qadr)
            dof_idx.append(dadr)
            if int(model.jnt_limited[j]) == 1:
                lo = float(model.jnt_range[j, 0])
                hi = float(model.jnt_range[j, 1])
            else:
                lo, hi = -np.inf, np.inf
            limits.append((len(qpos_idx) - 1, lo, hi, qadr))
    return qpos_idx, dof_idx, limits


def _extract_var_traj(qpos_seq: np.ndarray, qpos_idx: Sequence[int]) -> np.ndarray:
    return np.asarray(qpos_seq[:, qpos_idx], dtype=np.float64).copy()


def _apply_var_traj(
    qpos_base: np.ndarray,
    x: np.ndarray,
    qpos_idx: Sequence[int],
    quat_indices: Sequence[int],
) -> np.ndarray:
    out = np.asarray(qpos_base, dtype=np.float64).copy()
    out[:, qpos_idx] = x
    for t in range(out.shape[0]):
        for qi in quat_indices:
            out[t, qi : qi + 4] = _safe_quat_wxyz(out[t, qi : qi + 4])
    return out


def _quat_qpos_indices(model: mujoco.MjModel) -> List[int]:
    idx: List[int] = []
    for j in range(model.njnt):
        if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
            idx.append(int(model.jnt_qposadr[j]) + 3)
        elif int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_BALL):
            idx.append(int(model.jnt_qposadr[j]))
    return idx


def _solve_temporal_window(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_seq: np.ndarray,
    x_win: np.ndarray,
    qpos_idx: Sequence[int],
    dof_idx: Sequence[int],
    frame_names: Dict[str, str],
    desired: Dict[str, np.ndarray],
    contacts: Dict[str, ContactPlan],
    hint_names: Dict[str, str],
    hint_desired: Dict[str, np.ndarray],
    shoulder_var_idx: List[int],
    arm_var_idx: List[int],
    arm_lr_pairs: List[Tuple[int, int, int]],
    wrist_var_idx: List[int],
    wrist_strong_var_idx: List[int],
    wrist_ref: np.ndarray,
    x_ref: np.ndarray,
    seq_anchor_weight: np.ndarray,
    seq_anchor_ref: np.ndarray,
    symmetry_weight: np.ndarray,
    symmetry_lateral_axis: int,
    global_start: int,
    weights: TemporalWeights,
) -> np.ndarray:
    wlen, nd = x_win.shape
    nvar = wlen * nd
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    rhs: List[float] = []
    row = 0

    def add_dense_row(
        frame_local: int, jac_sub: np.ndarray, err: np.ndarray, w: float
    ) -> None:
        nonlocal row
        if w <= 0:
            return
        sw = float(np.sqrt(w))
        base = frame_local * nd
        for rr in range(jac_sub.shape[0]):
            rhs.append(sw * float(err[rr]))
            jrow = jac_sub[rr]
            nz = np.where(np.abs(jrow) > 1e-12)[0]
            for cc in nz:
                rows.append(row)
                cols.append(base + int(cc))
                vals.append(sw * float(jrow[cc]))
            row += 1

    for k in range(wlen):
        g = global_start + k
        data.qpos[:] = qpos_seq[g]
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)

        root_pos: Optional[np.ndarray] = None
        root_quat: Optional[np.ndarray] = None
        root_jac_sub: Optional[np.ndarray] = None
        if "root" in frame_names:
            root_pos, root_quat, _ = _frame_pose(model, data, frame_names["root"])
            jp_root, _ = _frame_jacobian(model, data, frame_names["root"])
            root_jac_sub = jp_root[:, dof_idx]

        # root and chest
        for key, (wp, wr) in (
            ("root", (weights.root_pos, weights.root_rot)),
            ("chest", (weights.chest_pos, weights.chest_rot)),
        ):
            if key not in frame_names or key not in desired:
                continue
            p_now, q_now, _ = _frame_pose(model, data, frame_names[key])
            jp, jr = _frame_jacobian(model, data, frame_names[key])
            jps = jp[:, dof_idx]
            jrs = jr[:, dof_idx]
            add_dense_row(k, jps, desired[key][g, :3] - p_now, wp)
            add_dense_row(
                k, jrs, _orientation_error_vec(desired[key][g, 3:7], q_now), wr
            )

        # end-effectors
        for ee in ("left_foot", "right_foot", "left_hand", "right_hand"):
            if ee not in frame_names:
                continue
            p_now, q_now, _ = _frame_pose(model, data, frame_names[ee])
            jp, jr = _frame_jacobian(model, data, frame_names[ee])
            jps = jp[:, dof_idx]
            jrs = jr[:, dof_idx]
            wp = weights.ee_pos_foot if "foot" in ee else weights.ee_pos_hand
            wr = weights.ee_rot_foot if "foot" in ee else weights.ee_rot_hand
            if "hand" in ee:
                if weights.hand_rot_startup_frames > 0:
                    startup_scale = min(
                        1.0, float(g + 1) / float(weights.hand_rot_startup_frames)
                    )
                else:
                    startup_scale = 1.0
                wr *= startup_scale
                if "wrist" in frame_names[ee].lower():
                    wr = 0.0
            add_dense_row(k, jps, desired[ee][g, :3] - p_now, wp)
            cp = contacts.get(ee)
            if cp is not None and bool(cp.active[g]) and "hand" in ee:
                wr *= 0.15
            add_dense_row(
                k, jrs, _orientation_error_vec(desired[ee][g, 3:7], q_now), wr
            )
            if cp is not None and bool(cp.active[g]):
                wc = (
                    weights.contact_foot if "foot" in ee else weights.contact_hand
                ) * float(cp.weight[g])
                add_dense_row(k, jps, cp.anchor[g] - p_now, wc)

        # bend hints
        for hk, hname in hint_names.items():
            if hk not in hint_desired:
                continue
            p_now, _, _ = _frame_pose(model, data, hname)
            jp, _ = _frame_jacobian(model, data, hname)
            jps = jp[:, dof_idx]
            if "knee" in hk:
                w = weights.hint_knee
            elif "shoulder" in hk:
                w = weights.hint_shoulder
            else:
                w = weights.hint_elbow
            add_dense_row(k, jps, hint_desired[hk][g] - p_now, w)

        # Bilateral symmetry constraints for frames whose source pose is near-symmetric.
        wsym = float(symmetry_weight[g]) if g < symmetry_weight.shape[0] else 0.0
        if (
            wsym > 1e-5
            and root_pos is not None
            and root_quat is not None
            and root_jac_sub is not None
        ):
            root_rot = R.from_quat(root_quat, scalar_first=True)
            world_to_root = root_rot.inv().as_matrix()

            pair_specs: List[Tuple[str, str, float, bool]] = [
                ("left_hand", "right_hand", weights.symmetry_hand, True),
                ("left_foot", "right_foot", weights.symmetry_foot, True),
                ("left_shoulder", "right_shoulder", 1.2 * weights.symmetry_hint, False),
                ("left_elbow", "right_elbow", weights.symmetry_hint, False),
                ("left_knee", "right_knee", 0.8 * weights.symmetry_hint, False),
            ]

            for left_key, right_key, w_pair, use_frame_dict in pair_specs:
                if w_pair <= 0:
                    continue
                if use_frame_dict:
                    if left_key not in frame_names or right_key not in frame_names:
                        continue
                    lf = frame_names[left_key]
                    rf = frame_names[right_key]
                else:
                    if left_key not in hint_names or right_key not in hint_names:
                        continue
                    lf = hint_names[left_key]
                    rf = hint_names[right_key]

                lp, _, _ = _frame_pose(model, data, lf)
                rp, _, _ = _frame_pose(model, data, rf)
                ljp, _ = _frame_jacobian(model, data, lf)
                rjp, _ = _frame_jacobian(model, data, rf)
                lj = ljp[:, dof_idx] - root_jac_sub
                rj = rjp[:, dof_idx] - root_jac_sub

                ll = world_to_root @ (lp - root_pos)
                rl = world_to_root @ (rp - root_pos)
                lj_local = world_to_root @ lj
                rj_local = world_to_root @ rj

                axis_lat = int(np.clip(symmetry_lateral_axis, 0, 2))
                for ax in range(3):
                    if ax == axis_lat:
                        err_ax = -(ll[ax] + rl[ax])
                        jac_ax = lj_local[ax] + rj_local[ax]
                    else:
                        err_ax = -(ll[ax] - rl[ax])
                        jac_ax = lj_local[ax] - rj_local[ax]
                    add_dense_row(
                        k,
                        jac_ax.reshape(1, -1),
                        np.array([err_ax], dtype=np.float64),
                        wsym * w_pair,
                    )

        # naturalness prior
        for vj in shoulder_var_idx:
            nonlocal_idx = k * nd + vj
            rows.append(row)
            cols.append(nonlocal_idx)
            vals.append(float(np.sqrt(weights.shoulder_roll)))
            rhs.append(
                float(np.sqrt(weights.shoulder_roll) * (x_ref[g, vj] - x_win[k, vj]))
            )
            row += 1
        shoulder_set = set(shoulder_var_idx)
        wrist_set = set(wrist_var_idx)
        for vj in arm_var_idx:
            if vj in shoulder_set or vj in wrist_set:
                continue
            nonlocal_idx = k * nd + vj
            rows.append(row)
            cols.append(nonlocal_idx)
            vals.append(float(np.sqrt(weights.arm_pose)))
            rhs.append(float(np.sqrt(weights.arm_pose) * (x_ref[g, vj] - x_win[k, vj])))
            row += 1

        if wsym > 1e-5 and arm_lr_pairs and weights.arm_lr_mirror > 0:
            swm = float(np.sqrt(weights.arm_lr_mirror * wsym))
            for li, ri, sgn in arm_lr_pairs:
                rows.extend([row, row])
                cols.extend([k * nd + int(li), k * nd + int(ri)])
                vals.extend([swm, -swm * float(sgn)])
                rhs.append(
                    -swm * float(x_win[k, int(li)] - float(sgn) * x_win[k, int(ri)])
                )
                row += 1
        strong_set = set(wrist_strong_var_idx)
        for vj in wrist_var_idx:
            nonlocal_idx = k * nd + vj
            boost = 1.0 + weights.wrist_neutral_startup_boost * max(
                0.0,
                1.0 - float(g) / max(1.0, float(weights.hand_rot_startup_frames)),
            )
            base_w = (
                weights.wrist_neutral_roll_yaw
                if vj in strong_set
                else weights.wrist_neutral_pitch
            ) * weights.wrist_neutral
            w = max(0.0, base_w * boost)
            if w <= 0:
                continue
            rows.append(row)
            cols.append(nonlocal_idx)
            vals.append(float(np.sqrt(w)))
            rhs.append(float(np.sqrt(w) * (wrist_ref[vj] - x_win[k, vj])))
            row += 1

        # Sequence-anchor prior: preserve user-authored keyframe semantics at timed sequence arrivals.
        wa = float(seq_anchor_weight[g]) if g < seq_anchor_weight.shape[0] else 0.0
        if wa > 0:
            for vj in range(nd):
                qadr = int(qpos_idx[vj])
                w = wa * (
                    weights.seq_anchor_root
                    if qadr in (0, 1, 2)
                    else weights.seq_anchor_joint
                )
                if w <= 0:
                    continue
                nonlocal_idx = k * nd + vj
                rows.append(row)
                cols.append(nonlocal_idx)
                vals.append(float(np.sqrt(w)))
                rhs.append(float(np.sqrt(w) * (seq_anchor_ref[g, vj] - x_win[k, vj])))
                row += 1

    sw_v = float(np.sqrt(weights.vel))
    sw_a = float(np.sqrt(weights.acc))
    sw_r = float(np.sqrt(weights.reg))

    # velocity smoothness
    if weights.vel > 0:
        for k in range(1, wlen):
            for j in range(nd):
                rows.extend([row, row])
                cols.extend([k * nd + j, (k - 1) * nd + j])
                vals.extend([sw_v, -sw_v])
                rhs.append(-sw_v * float(x_win[k, j] - x_win[k - 1, j]))
                row += 1

    # acceleration smoothness
    if weights.acc > 0 and wlen >= 3:
        for k in range(1, wlen - 1):
            for j in range(nd):
                rows.extend([row, row, row])
                cols.extend([(k + 1) * nd + j, k * nd + j, (k - 1) * nd + j])
                vals.extend([sw_a, -2.0 * sw_a, sw_a])
                rhs.append(
                    -sw_a * float(x_win[k + 1, j] - 2.0 * x_win[k, j] + x_win[k - 1, j])
                )
                row += 1

    # shoulder-roll velocity prior
    if shoulder_var_idx and weights.shoulder_roll_vel > 0:
        sw_sv = float(np.sqrt(weights.shoulder_roll_vel))
        for k in range(1, wlen):
            for j in shoulder_var_idx:
                rows.extend([row, row])
                cols.extend([k * nd + j, (k - 1) * nd + j])
                vals.extend([sw_sv, -sw_sv])
                rhs.append(-sw_sv * float(x_win[k, j] - x_win[k - 1, j]))
                row += 1

    # wrist roll/yaw velocity prior
    if wrist_strong_var_idx and weights.wrist_strong_vel > 0:
        sw_wv = float(np.sqrt(weights.wrist_strong_vel))
        for k in range(1, wlen):
            for j in wrist_strong_var_idx:
                rows.extend([row, row])
                cols.extend([k * nd + j, (k - 1) * nd + j])
                vals.extend([sw_wv, -sw_wv])
                rhs.append(-sw_wv * float(x_win[k, j] - x_win[k - 1, j]))
                row += 1

    # trust-region regularization
    for i in range(nvar):
        rows.append(row)
        cols.append(i)
        vals.append(sw_r)
        rhs.append(0.0)
        row += 1

    if row == 0:
        return np.zeros_like(x_win)
    a = sp.coo_matrix((vals, (rows, cols)), shape=(row, nvar), dtype=np.float64).tocsr()
    b = np.asarray(rhs, dtype=np.float64)
    sol = lsqr(a, b, atol=1e-5, btol=1e-5, iter_lim=300)
    dq = np.asarray(sol[0], dtype=np.float64).reshape(wlen, nd)
    return dq


def _clip_to_limits(
    x: np.ndarray, limits: Sequence[Tuple[int, float, float, int]]
) -> None:
    for idx, lo, hi, _ in limits:
        if np.isfinite(lo) or np.isfinite(hi):
            x[:, idx] = np.clip(x[:, idx], lo, hi)


def _smooth_qpos(qpos: np.ndarray, window: int = 11, poly: int = 3) -> np.ndarray:
    if qpos.shape[0] < 7:
        return qpos.copy()
    w = min(window, qpos.shape[0] if qpos.shape[0] % 2 == 1 else qpos.shape[0] - 1)
    w = max(5, w)
    if w % 2 == 0:
        w -= 1
    if w <= poly:
        return qpos.copy()
    out = qpos.copy()
    start = 7 if qpos.shape[1] >= 7 else 0
    if start < qpos.shape[1]:
        out[:, start:] = savgol_filter(
            out[:, start:], window_length=w, polyorder=poly, axis=0
        )
    return out


def _shoulder_roll_var_indices(
    model: mujoco.MjModel, qpos_idx: Sequence[int]
) -> List[int]:
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
        for j in range(model.njnt)
    ]
    qadr_to_name: Dict[int, str] = {}
    for j in range(model.njnt):
        qadr_to_name[int(model.jnt_qposadr[j])] = names[j].lower()
    out: List[int] = []
    for i, qadr in enumerate(qpos_idx):
        nm = qadr_to_name.get(int(qadr), "")
        if "shoulder" in nm and "roll" in nm:
            out.append(i)
    return out


def _arm_var_indices(model: mujoco.MjModel, qpos_idx: Sequence[int]) -> List[int]:
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
        for j in range(model.njnt)
    ]
    qadr_to_name: Dict[int, str] = {}
    for j in range(model.njnt):
        qadr_to_name[int(model.jnt_qposadr[j])] = names[j].lower()
    out: List[int] = []
    for i, qadr in enumerate(qpos_idx):
        nm = qadr_to_name.get(int(qadr), "")
        if any(tok in nm for tok in ("shoulder", "elbow", "wrist")):
            out.append(i)
    return out


def _wrist_var_indices(model: mujoco.MjModel, qpos_idx: Sequence[int]) -> List[int]:
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
        for j in range(model.njnt)
    ]
    qadr_to_name: Dict[int, str] = {}
    for j in range(model.njnt):
        qadr_to_name[int(model.jnt_qposadr[j])] = names[j].lower()
    out: List[int] = []
    for i, qadr in enumerate(qpos_idx):
        nm = qadr_to_name.get(int(qadr), "")
        if "wrist" in nm:
            out.append(i)
    return out


def _wrist_roll_yaw_var_indices(
    model: mujoco.MjModel, qpos_idx: Sequence[int]
) -> List[int]:
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
        for j in range(model.njnt)
    ]
    qadr_to_name: Dict[int, str] = {}
    for j in range(model.njnt):
        qadr_to_name[int(model.jnt_qposadr[j])] = names[j].lower()
    out: List[int] = []
    for i, qadr in enumerate(qpos_idx):
        nm = qadr_to_name.get(int(qadr), "")
        if "wrist" in nm and ("roll" in nm or "yaw" in nm):
            out.append(i)
    return out


def _bilateral_arm_joint_pairs(
    model: mujoco.MjModel,
    qpos_idx: Sequence[int],
) -> List[Tuple[int, int]]:
    names = [
        (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}").lower()
        for j in range(model.njnt)
    ]
    qadr_to_name: Dict[int, str] = {}
    for j in range(model.njnt):
        qadr_to_name[int(model.jnt_qposadr[j])] = names[j]

    name_to_var: Dict[str, int] = {}
    for i, qadr in enumerate(qpos_idx):
        nm = qadr_to_name.get(int(qadr), "")
        name_to_var[nm] = i

    def _right_name(left_name: str) -> Optional[str]:
        cands = [
            left_name.replace("left_", "right_", 1),
            left_name.replace("_left", "_right", 1),
            left_name.replace("left", "right", 1),
            left_name.replace("l_", "r_", 1),
            left_name.replace("_l", "_r", 1),
        ]
        for c in cands:
            if c != left_name:
                return c
        return None

    out: List[Tuple[int, int]] = []
    seen = set()
    for nm, li in name_to_var.items():
        if "left" not in nm and not nm.startswith("l_") and "_l" not in nm:
            continue
        if not any(tok in nm for tok in ("shoulder", "elbow", "wrist")):
            continue
        rn = _right_name(nm)
        if rn is None or rn not in name_to_var:
            continue
        ri = name_to_var[rn]
        pair = (min(li, ri), max(li, ri))
        if pair in seen:
            continue
        seen.add(pair)
        out.append((li, ri))
    return out


def _pair_sign_from_ref(
    x_ref: np.ndarray,
    pairs: Sequence[Tuple[int, int]],
    symmetry_weight: np.ndarray,
) -> List[Tuple[int, int, int]]:
    if x_ref.ndim != 2:
        return []
    idx = np.where(symmetry_weight > 0.55)[0]
    if idx.size == 0:
        idx = np.arange(x_ref.shape[0], dtype=np.int64)
    out: List[Tuple[int, int, int]] = []
    for li, ri in pairs:
        lv = x_ref[idx, int(li)]
        rv = x_ref[idx, int(ri)]
        e_same = float(np.mean((lv - rv) ** 2))
        e_opp = float(np.mean((lv + rv) ** 2))
        sgn = 1 if e_same <= e_opp else -1
        out.append((int(li), int(ri), int(sgn)))
    return out


def _sequence_anchor_targets(
    source_data: dict,
    source_time: np.ndarray,
    q_src: np.ndarray,
    q_warm_tgt: np.ndarray,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    n_frames = q_warm_tgt.shape[0]
    seq_w = np.zeros((n_frames,), dtype=np.float64)
    seq_ref = q_warm_tgt.copy()

    name_to_src_idx: Dict[str, int] = {}
    src_keyframes = source_data.get("keyframes", [])
    if isinstance(src_keyframes, list):
        for kf in src_keyframes:
            if not isinstance(kf, dict):
                continue
            nm = str(kf.get("name", ""))
            if not nm or nm in name_to_src_idx or "qpos" not in kf:
                continue
            qk = np.asarray(kf["qpos"], dtype=np.float64)
            if qk.shape != q_src.shape[1:]:
                continue
            src_idx = int(np.argmin(np.linalg.norm(q_src - qk[None, :], axis=1)))
            name_to_src_idx[nm] = int(np.clip(src_idx, 0, n_frames - 1))

    name_to_tgt_template: Dict[str, np.ndarray] = {
        nm: q_warm_tgt[idx].copy() for nm, idx in name_to_src_idx.items()
    }

    src_seq = source_data.get("timed_sequence", [])
    if not isinstance(src_seq, list) or not src_seq:
        return seq_w, seq_ref

    for entry in src_seq:
        if not (isinstance(entry, (list, tuple)) and len(entry) == 2):
            continue
        base_name = str(entry[0])
        try:
            tq = float(entry[1])
        except Exception:
            continue
        if source_time.ndim == 1 and source_time.size > 0:
            idx = int(np.argmin(np.abs(source_time - tq)))
        else:
            idx = int(round(tq / max(dt, 1e-6)))
        idx = int(np.clip(idx, 0, n_frames - 1))
        seq_w[idx] = max(seq_w[idx], 1.0)

        q_template = name_to_tgt_template.get(base_name)
        if q_template is None:
            continue

        q_anchor = seq_ref[idx].copy()
        if q_anchor.shape == q_template.shape and q_anchor.shape[0] >= 7:
            # Preserve trajectory root progression while enforcing pose template on articulated joints.
            q_anchor[7:] = q_template[7:]
        else:
            q_anchor = q_template.copy()
        seq_ref[idx] = q_anchor

    return seq_w, seq_ref


def _source_symmetry_profile(
    src_model: mujoco.MjModel,
    src_data: mujoco.MjData,
    q_src: np.ndarray,
    src_root: str,
    src_ee: CanonicalEE,
    src_hints: Dict[str, str],
) -> Tuple[np.ndarray, int]:
    n = q_src.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.float64), 1

    pair_frames: List[Tuple[str, str]] = [
        (src_ee.left_hand, src_ee.right_hand),
        (src_ee.left_foot, src_ee.right_foot),
    ]
    if "left_elbow" in src_hints and "right_elbow" in src_hints:
        pair_frames.append((src_hints["left_elbow"], src_hints["right_elbow"]))
    if "left_knee" in src_hints and "right_knee" in src_hints:
        pair_frames.append((src_hints["left_knee"], src_hints["right_knee"]))

    # Infer lateral axis from the first valid frame by the largest left-right separation.
    lat_axis = 1
    src_data.qpos[:] = q_src[0]
    src_data.qvel[:] = 0
    mujoco.mj_forward(src_model, src_data)
    root_p0, root_q0, _ = _frame_pose(src_model, src_data, src_root)
    root_rot0 = R.from_quat(root_q0, scalar_first=True).inv()
    max_sep = -1.0
    for lf, rf in pair_frames:
        lp, _, _ = _frame_pose(src_model, src_data, lf)
        rp, _, _ = _frame_pose(src_model, src_data, rf)
        ll = root_rot0.apply(lp - root_p0)
        rl = root_rot0.apply(rp - root_p0)
        sep = np.abs(ll - rl)
        ax = int(np.argmax(sep))
        if float(sep[ax]) > max_sep:
            max_sep = float(sep[ax])
            lat_axis = ax

    wsym = np.zeros((n,), dtype=np.float64)
    prev_q = src_data.qpos.copy()
    prev_v = src_data.qvel.copy()
    try:
        for i in range(n):
            src_data.qpos[:] = q_src[i]
            src_data.qvel[:] = 0
            mujoco.mj_forward(src_model, src_data)
            root_p, root_q, _ = _frame_pose(src_model, src_data, src_root)
            root_rot = R.from_quat(root_q, scalar_first=True).inv()

            errs: List[float] = []
            for lf, rf in pair_frames:
                lp, _, _ = _frame_pose(src_model, src_data, lf)
                rp, _, _ = _frame_pose(src_model, src_data, rf)
                ll = root_rot.apply(lp - root_p)
                rl = root_rot.apply(rp - root_p)
                e = 0.0
                for ax in range(3):
                    if ax == lat_axis:
                        e += float((ll[ax] + rl[ax]) ** 2)
                    else:
                        e += float((ll[ax] - rl[ax]) ** 2)
                errs.append(np.sqrt(e))

            if errs:
                em = float(np.median(np.asarray(errs, dtype=np.float64)))
                wsym[i] = float(np.exp(-((em / 0.10) ** 2)))

    finally:
        src_data.qpos[:] = prev_q
        src_data.qvel[:] = prev_v
        mujoco.mj_forward(src_model, src_data)

    # Keep only clearly symmetric frames and smooth a little.
    wsym = np.where(wsym >= 0.45, wsym, 0.0)
    if n >= 5:
        ker = np.array([0.15, 0.20, 0.30, 0.20, 0.15], dtype=np.float64)
        wsym = np.convolve(wsym, ker, mode="same")
        wsym = np.clip(wsym, 0.0, 1.0)
    return wsym, int(lat_axis)


def _contact_plans_from_source(
    src_tracks: Dict[str, np.ndarray],
    dt: float,
    foot_z: float,
    hand_z: float,
) -> Dict[str, ContactPlan]:
    plans: Dict[str, ContactPlan] = {}
    for ee in ("left_foot", "right_foot", "left_hand", "right_hand"):
        pos = src_tracks[ee][:, :3]
        if "foot" in ee:
            plan = _detect_contact_plan(
                pos,
                dt=dt,
                z_on=foot_z,
                z_off=foot_z + 0.03,
                v_on=0.16,
                v_off=0.24,
                min_frames=5,
                edge_release=4,
                max_phase_slip=0.04,
            )
        else:
            plan = _detect_contact_plan(
                pos,
                dt=dt,
                z_on=hand_z,
                z_off=hand_z + 0.05,
                v_on=0.20,
                v_off=0.30,
                min_frames=5,
                edge_release=5,
                max_phase_slip=0.06,
            )
        plans[ee] = plan
    return plans


def _solve_warm_start_ik(
    src_model: mujoco.MjModel,
    src_data: mujoco.MjData,
    tgt_model: mujoco.MjModel,
    tgt_data: mujoco.MjData,
    q_src: np.ndarray,
    src_root: str,
    tgt_root: str,
    src_chest: Optional[str],
    tgt_chest: Optional[str],
    src_ee: CanonicalEE,
    tgt_ee: CanonicalEE,
    src_hints: Dict[str, str],
    tgt_hints: Dict[str, str],
    scales: Dict[str, float],
    dt: float,
) -> Tuple[
    np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]
]:
    n = q_src.shape[0]
    cfg = mink.Configuration(tgt_model)
    limits: List[object] = [mink.ConfigurationLimit(tgt_model)]
    solver = _select_ik_solver()
    posture = mink.PostureTask(tgt_model, cost=6e-3)

    src_data.qpos[:] = q_src[0]
    src_data.qvel[:] = 0
    mujoco.mj_forward(src_model, src_data)
    src_root0, src_rootq0, _ = _frame_pose(src_model, src_data, src_root)
    r_src0 = R.from_quat(src_rootq0, scalar_first=True)

    q0 = _home_qpos(tgt_model)
    tgt_data.qpos[:] = q0
    tgt_data.qvel[:] = 0
    mujoco.mj_forward(tgt_model, tgt_data)
    tgt_root0, tgt_rootq0, _ = _frame_pose(tgt_model, tgt_data, tgt_root)
    r_tgt0 = R.from_quat(tgt_rootq0, scalar_first=True)

    desired: Dict[str, np.ndarray] = {}
    for key in ("root", "chest", "left_foot", "right_foot", "left_hand", "right_hand"):
        desired[key] = np.zeros((n, 7), dtype=np.float64)
    hint_desired: Dict[str, np.ndarray] = {}
    common_hints = {
        k: (src_hints[k], tgt_hints[k]) for k in sorted(set(src_hints) & set(tgt_hints))
    }
    for hk in common_hints:
        hint_desired[hk] = np.zeros((n, 3), dtype=np.float64)

    q_seq = np.zeros((n, tgt_model.nq), dtype=np.float64)
    q_tgt = q0.copy()

    for i in range(n):
        src_data.qpos[:] = q_src[i]
        src_data.qvel[:] = 0
        mujoco.mj_forward(src_model, src_data)

        src_root_pos, src_root_quat, _ = _frame_pose(src_model, src_data, src_root)
        r_src = R.from_quat(src_root_quat, scalar_first=True)
        dxy = src_root_pos[:2] - src_root0[:2]
        dz = src_root_pos[2] - src_root0[2]
        root_des = np.array(
            [
                tgt_root0[0] + scales["root_xy"] * dxy[0],
                tgt_root0[1] + scales["root_xy"] * dxy[1],
                tgt_root0[2] + scales["root_z"] * dz,
            ],
            dtype=np.float64,
        )
        root_q_des = _safe_quat_wxyz(
            (r_tgt0 * (r_src0.inv() * r_src)).as_quat(scalar_first=True)
        )
        desired["root"][i, :3] = root_des
        desired["root"][i, 3:7] = root_q_des

        if src_chest and tgt_chest:
            cp, cq, _ = _frame_pose(src_model, src_data, src_chest)
            cloc_p, cloc_q = _to_local_pose(src_root_pos, src_root_quat, cp, cq)
            cdes_p, cdes_q = _to_world_pose(
                root_des, root_q_des, cloc_p * scales["torso"], cloc_q
            )
            desired["chest"][i, :3] = cdes_p
            desired["chest"][i, 3:7] = cdes_q
        else:
            desired["chest"][i] = desired["root"][i]

        for ee_key, src_frame in src_ee.as_dict().items():
            sp, sq, _ = _frame_pose(src_model, src_data, src_frame)
            ploc, qloc = _to_local_pose(src_root_pos, src_root_quat, sp, sq)
            limb = scales["leg"] if "foot" in ee_key else scales["arm"]
            pdes, qdes = _to_world_pose(root_des, root_q_des, ploc * limb, qloc)
            desired[ee_key][i, :3] = pdes
            desired[ee_key][i, 3:7] = qdes

        for hk, (src_h, _tgt_h) in common_hints.items():
            hp, hq, _ = _frame_pose(src_model, src_data, src_h)
            ploc, _ = _to_local_pose(src_root_pos, src_root_quat, hp, hq)
            scale = scales["leg"] if "knee" in hk else scales["arm"]
            pdes, _ = _to_world_pose(
                root_des,
                root_q_des,
                ploc * scale,
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            )
            hint_desired[hk][i] = pdes

        cfg.update(q=q_tgt.copy())
        posture.set_target(cfg.q)
        tasks: List[object] = [posture]

        t_root = mink.FrameTask(
            frame_name=tgt_root,
            frame_type="body",
            position_cost=4.0,
            orientation_cost=3.0,
            lm_damping=1e-3,
        )
        t_root.set_target(
            _make_mink_target_pose(desired["root"][i, :3], desired["root"][i, 3:7])
        )
        tasks.append(t_root)

        if tgt_chest:
            t_chest = mink.FrameTask(
                frame_name=tgt_chest,
                frame_type="body",
                position_cost=2.0,
                orientation_cost=1.3,
                lm_damping=1e-3,
            )
            t_chest.set_target(
                _make_mink_target_pose(
                    desired["chest"][i, :3], desired["chest"][i, 3:7]
                )
            )
            tasks.append(t_chest)

        for ee_key, tgt_frame in tgt_ee.as_dict().items():
            frame_type = (
                "site"
                if mujoco.mj_name2id(tgt_model, mujoco.mjtObj.mjOBJ_SITE, tgt_frame)
                >= 0
                else "body"
            )
            hand_rot_cost = 0.14
            if (
                "hand" in ee_key
                and frame_type == "body"
                and "wrist" in tgt_frame.lower()
            ):
                hand_rot_cost = 0.0
            t_ee = mink.FrameTask(
                frame_name=tgt_frame,
                frame_type=frame_type,
                position_cost=6.5 if "foot" in ee_key else 3.8,
                orientation_cost=0.55 if "foot" in ee_key else hand_rot_cost,
                lm_damping=1e-3,
            )
            t_ee.set_target(
                _make_mink_target_pose(desired[ee_key][i, :3], desired[ee_key][i, 3:7])
            )
            tasks.append(t_ee)

        for hk, (_src_h, tgt_h) in common_hints.items():
            t_hint = mink.FrameTask(
                frame_name=tgt_h,
                frame_type="body",
                position_cost=1.2 if "knee" in hk else 1.0,
                orientation_cost=0.0,
                lm_damping=1e-3,
            )
            t_hint.set_target(
                _make_mink_target_pose(
                    hint_desired[hk][i],
                    np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                )
            )
            tasks.append(t_hint)

        best_q = cfg.q.copy()
        best = 1e18
        for _ in range(18):
            tgt_data.qpos[:] = cfg.q
            tgt_data.qvel[:] = 0
            mujoco.mj_forward(tgt_model, tgt_data)
            root_now, _, _ = _frame_pose(tgt_model, tgt_data, tgt_root)
            score = float(np.linalg.norm(root_now - desired["root"][i, :3]))
            for ee_key, tgt_frame in tgt_ee.as_dict().items():
                p_now, _, _ = _frame_pose(tgt_model, tgt_data, tgt_frame)
                score += float(np.linalg.norm(p_now - desired[ee_key][i, :3]))
            if score < best:
                best = score
                best_q = cfg.q.copy()
            try:
                vel = mink.solve_ik(
                    cfg, tasks, 0.09, solver=solver, damping=1e-3, limits=limits
                )
            except Exception:
                break
            nrm = float(np.linalg.norm(vel))
            if nrm > 20.0 and nrm > 1e-12:
                vel *= 20.0 / nrm
            cfg.integrate_inplace(vel, 0.09)
        q_tgt = best_q.copy()
        q_seq[i] = q_tgt
    return q_seq, desired, hint_desired, {k: v[1] for k, v in common_hints.items()}


def _build_mapped_contacts(
    desired: Dict[str, np.ndarray], source_contact: Dict[str, ContactPlan]
) -> Dict[str, ContactPlan]:
    out: Dict[str, ContactPlan] = {}
    for ee in ("left_foot", "right_foot", "left_hand", "right_hand"):
        sc = source_contact[ee]
        plan = ContactPlan(
            active=sc.active.copy(),
            weight=sc.weight.copy(),
            anchor=desired[ee][:, :3].copy(),
        )
        i = 0
        t = plan.active.shape[0]
        while i < t:
            if not plan.active[i]:
                i += 1
                continue
            j = i
            while j < t and plan.active[j]:
                j += 1
            seg = slice(i, j)
            segp = desired[ee][seg, :3]
            if segp.shape[0] == 0:
                i = j
                continue
            start = segp[0]
            end = segp[-1]
            max_slip = 0.04 if "foot" in ee else 0.06
            d = end - start
            n = float(np.linalg.norm(d))
            if n > max_slip and n > 1e-8:
                end = start + d / n * max_slip
            for k in range(i, j):
                if j - i <= 1:
                    alpha = 0.0
                else:
                    alpha = float(k - i) / float(j - i - 1)
                plan.anchor[k] = (1.0 - alpha) * start + alpha * end
            i = j
        out[ee] = plan
    return out


def _temporal_optimize(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q_init: np.ndarray,
    frame_names: Dict[str, str],
    desired: Dict[str, np.ndarray],
    contacts: Dict[str, ContactPlan],
    hint_names: Dict[str, str],
    hint_desired: Dict[str, np.ndarray],
    seq_anchor_weight: np.ndarray,
    seq_anchor_target_q: np.ndarray,
    symmetry_weight: np.ndarray,
    symmetry_lateral_axis: int,
    robot_name: str,
    *,
    outer_iters: int,
    window: int,
    overlap: int,
    step_scale: float,
) -> np.ndarray:
    qpos_idx, dof_idx, limits = _free_joint_and_hinge_vars(model)
    if not qpos_idx:
        return q_init.copy()
    quat_idx = _quat_qpos_indices(model)
    x = _extract_var_traj(q_init, qpos_idx)
    x_ref = x.copy()
    weights = TemporalWeights()
    if robot_name == "booster_t1":
        weights.contact_hand *= 0.85
        weights.ee_rot_hand *= 0.8
    if robot_name == "unitree_g1":
        weights.ee_rot_hand *= 0.45
        weights.wrist_neutral *= 2.2
        weights.wrist_neutral_roll_yaw *= 2.5
        weights.wrist_neutral_pitch *= 2.0
        weights.wrist_strong_vel *= 2.0
        weights.arm_pose *= 8.0
        weights.seq_anchor_joint *= 6.0
        weights.arm_lr_mirror *= 3.0
    shoulder_vars = (
        _shoulder_roll_var_indices(model, qpos_idx)
        if robot_name == "unitree_g1"
        else []
    )
    arm_vars = _arm_var_indices(model, qpos_idx)
    wrist_vars = _wrist_var_indices(model, qpos_idx)
    arm_lr_pairs = _pair_sign_from_ref(
        x_ref,
        _bilateral_arm_joint_pairs(model, qpos_idx),
        symmetry_weight,
    )

    if seq_anchor_target_q.shape == q_init.shape:
        x_anchor_ref = _extract_var_traj(seq_anchor_target_q, qpos_idx)
    else:
        x_anchor_ref = x_ref.copy()

    # Use sequence-anchor templates to build a smooth arm-posture reference trajectory.
    anchor_idx = np.where(seq_anchor_weight > 0.5)[0]
    if anchor_idx.size >= 2:
        grid = np.arange(x.shape[0], dtype=np.float64)
        for vj in arm_vars:
            x_ref[:, vj] = np.interp(
                grid, anchor_idx.astype(np.float64), x_anchor_ref[anchor_idx, vj]
            )
    wrist_ref = x_ref[0].copy()
    wrist_strong_vars = _wrist_roll_yaw_var_indices(model, qpos_idx)
    if wrist_vars:
        q_home = _home_qpos(model)
        for vj in wrist_vars:
            wrist_ref[vj] = float(q_home[int(qpos_idx[vj])])

    n = x.shape[0]
    step = max(1, window - overlap)
    for it in range(outer_iters):
        q_cur = _apply_var_traj(q_init, x, qpos_idx, quat_idx)
        update = np.zeros_like(x)
        wsum = np.zeros((n, 1), dtype=np.float64)
        for s in range(0, n, step):
            e = min(n, s + window)
            if e - s < 8:
                continue
            xw = x[s:e].copy()
            dw = _solve_temporal_window(
                model=model,
                data=data,
                qpos_seq=q_cur,
                x_win=xw,
                qpos_idx=qpos_idx,
                dof_idx=dof_idx,
                frame_names=frame_names,
                desired=desired,
                contacts=contacts,
                hint_names=hint_names,
                hint_desired=hint_desired,
                shoulder_var_idx=shoulder_vars,
                arm_var_idx=arm_vars,
                arm_lr_pairs=arm_lr_pairs,
                wrist_var_idx=wrist_vars,
                wrist_strong_var_idx=wrist_strong_vars,
                wrist_ref=wrist_ref,
                x_ref=x_ref,
                seq_anchor_weight=seq_anchor_weight,
                seq_anchor_ref=x_anchor_ref,
                symmetry_weight=symmetry_weight,
                symmetry_lateral_axis=symmetry_lateral_axis,
                global_start=s,
                weights=weights,
            )
            # triangular taper for overlap blending
            l = e - s
            if l <= 2:
                taper = np.ones((l, 1), dtype=np.float64)
            else:
                idx = np.arange(l, dtype=np.float64)
                taper = (1.0 - np.abs((idx - (l - 1) / 2.0) / ((l + 1) / 2.0))).reshape(
                    l, 1
                )
                taper = np.clip(0.15 + 0.85 * taper, 0.15, 1.0)
            update[s:e] += dw * taper
            wsum[s:e] += taper

        delta = update / np.maximum(wsum, 1e-8)
        # trust-region caps (allow larger arm updates to escape poor warm-start nullspace).
        max_root_step = 0.015
        max_joint_step = 0.18
        max_arm_step = 0.55 if robot_name == "unitree_g1" else 0.35
        arm_set = set(arm_vars)
        for j, qadr in enumerate(qpos_idx):
            if qadr in (0, 1, 2):
                cap = max_root_step
            elif j in arm_set:
                cap = max_arm_step
            else:
                cap = max_joint_step
            delta[:, j] = np.clip(delta[:, j], -cap, cap)
        x += step_scale * delta
        _clip_to_limits(x, limits)

        if np.linalg.norm(delta) / max(1, delta.size) < 1e-5:
            break

    q_out = _apply_var_traj(q_init, x, qpos_idx, quat_idx)
    q_out = _smooth_qpos(q_out, window=9, poly=3)
    return q_out


def _foot_floor_alignment(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_seq: np.ndarray,
    left_foot: str,
    right_foot: str,
    clearance: float,
) -> np.ndarray:
    out = qpos_seq.copy()
    root_z_idx: Optional[int] = None
    for j in range(model.njnt):
        if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
            root_z_idx = int(model.jnt_qposadr[j]) + 2
            break
    if root_z_idx is None:
        return out
    z_vals: List[float] = []
    prev_q = data.qpos.copy()
    prev_v = data.qvel.copy()
    try:
        for i in range(out.shape[0]):
            data.qpos[:] = out[i]
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            zl = float(_frame_pose(model, data, left_foot)[0][2])
            zr = float(_frame_pose(model, data, right_foot)[0][2])
            z_vals.extend([zl, zr])
    finally:
        data.qpos[:] = prev_q
        data.qvel[:] = prev_v
        mujoco.mj_forward(model, data)
    if not z_vals:
        return out
    p5 = float(np.percentile(np.asarray(z_vals, dtype=np.float64), 5.0))
    if p5 < clearance:
        out[:, root_z_idx] += float(clearance - p5)
    return out


def _build_output_dict(
    source_data: dict,
    qpos_tgt_seq: np.ndarray,
    dt: float,
    tgt_model: mujoco.MjModel,
    tgt_data: mujoco.MjData,
) -> dict:
    n = qpos_tgt_seq.shape[0]
    action_seq = np.zeros((n, tgt_model.nu), dtype=np.float32)
    for i in range(n):
        motor_pos, _ = _compute_motor_and_joint_pos(
            tgt_model, tgt_data, qpos_tgt_seq[i]
        )
        if motor_pos.shape[0] == tgt_model.nu:
            action_seq[i] = motor_pos

    out_keyframes: List[dict] = []
    out_seq: List[Tuple[str, float]] = []
    src_time = np.asarray(source_data.get("time", []), dtype=np.float64)
    src_seq = source_data.get("timed_sequence", [])
    q_src_all = _extract_qpos_sequence(source_data)

    def _src_index_from_time(tq: float) -> int:
        if src_time.ndim == 1 and src_time.size > 0:
            return int(np.argmin(np.abs(src_time - tq)))
        return int(np.clip(round(tq / max(dt, 1e-6)), 0, q_src_all.shape[0] - 1))

    if isinstance(src_seq, list) and src_seq:
        for si, entry in enumerate(src_seq):
            if not (isinstance(entry, (list, tuple)) and len(entry) == 2):
                continue
            base_name = str(entry[0])
            try:
                t_arrive = float(entry[1])
            except Exception:
                t_arrive = float(si * dt)

            src_idx = _src_index_from_time(t_arrive)
            dst_idx = int(np.clip(src_idx, 0, n - 1))
            qk = qpos_tgt_seq[dst_idx]
            motor_pos, joint_pos = _compute_motor_and_joint_pos(tgt_model, tgt_data, qk)
            unique_name = f"{base_name}__seq{si:03d}"
            out_keyframes.append(
                {
                    "name": unique_name,
                    "source_name": base_name,
                    "source_time": t_arrive,
                    "source_idx": int(src_idx),
                    "motor_pos": motor_pos.astype(np.float32),
                    "joint_pos": joint_pos.astype(np.float32),
                    "qpos": qk.astype(np.float64),
                }
            )
            out_seq.append((unique_name, t_arrive))
    else:
        for i in range(n):
            qk = qpos_tgt_seq[i]
            motor_pos, joint_pos = _compute_motor_and_joint_pos(tgt_model, tgt_data, qk)
            nm = f"frame_{i:04d}"
            out_keyframes.append(
                {
                    "name": nm,
                    "motor_pos": motor_pos.astype(np.float32),
                    "joint_pos": joint_pos.astype(np.float32),
                    "qpos": qk.astype(np.float64),
                }
            )
            out_seq.append((nm, float(i * dt)))

    out_time = np.arange(n, dtype=np.float64) * dt
    return {
        "time": out_time.astype(np.float64),
        "qpos": qpos_tgt_seq.astype(np.float32),
        "action": action_seq.astype(np.float32),
        "keyframes": out_keyframes,
        "timed_sequence": out_seq,
        "is_robot_relative_frame": bool(
            source_data.get("is_robot_relative_frame", True)
        ),
    }


def _eval_metrics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_seq: np.ndarray,
    desired: Dict[str, np.ndarray],
    contacts: Dict[str, ContactPlan],
    frame_names: Dict[str, str],
    dof_idx: Sequence[int],
    shoulder_roll_dof: Sequence[int],
) -> Dict[str, float]:
    n = qpos_seq.shape[0]
    ee_pos_err = []
    ee_rot_err = []
    hand_asym = []
    foot_asym = []
    for i in range(n):
        data.qpos[:] = qpos_seq[i]
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        for ee in ("left_foot", "right_foot", "left_hand", "right_hand"):
            if ee not in frame_names:
                continue
            p, q, _ = _frame_pose(model, data, frame_names[ee])
            ee_pos_err.append(float(np.linalg.norm(p - desired[ee][i, :3])))
            ee_rot_err.append(
                float(np.linalg.norm(_orientation_error_vec(desired[ee][i, 3:7], q)))
            )

        if "root" in frame_names:
            rp, rq, _ = _frame_pose(model, data, frame_names["root"])
            rot_inv = R.from_quat(rq, scalar_first=True).inv()
            if "left_hand" in frame_names and "right_hand" in frame_names:
                lp, _, _ = _frame_pose(model, data, frame_names["left_hand"])
                rp2, _, _ = _frame_pose(model, data, frame_names["right_hand"])
                ll = rot_inv.apply(lp - rp)
                rr = rot_inv.apply(rp2 - rp)
                hand_asym.append(
                    float(
                        np.sqrt(
                            (ll[1] + rr[1]) ** 2
                            + (ll[0] - rr[0]) ** 2
                            + (ll[2] - rr[2]) ** 2
                        )
                    )
                )
            if "left_foot" in frame_names and "right_foot" in frame_names:
                lf, _, _ = _frame_pose(model, data, frame_names["left_foot"])
                rf, _, _ = _frame_pose(model, data, frame_names["right_foot"])
                fl = rot_inv.apply(lf - rp)
                fr = rot_inv.apply(rf - rp)
                foot_asym.append(
                    float(
                        np.sqrt(
                            (fl[1] + fr[1]) ** 2
                            + (fl[0] - fr[0]) ** 2
                            + (fl[2] - fr[2]) ** 2
                        )
                    )
                )

    stance_drifts = []
    for ee, cp in contacts.items():
        i = 0
        while i < n:
            if not cp.active[i]:
                i += 1
                continue
            j = i
            poses = []
            while j < n and cp.active[j]:
                data.qpos[:] = qpos_seq[j]
                data.qvel[:] = 0
                mujoco.mj_forward(model, data)
                p, _, _ = _frame_pose(model, data, frame_names[ee])
                poses.append(p)
                j += 1
            if len(poses) >= 2:
                pa = np.asarray(poses, dtype=np.float64)
                stance_drifts.append(float(np.max(np.linalg.norm(pa - pa[0], axis=1))))
            i = j

    qv = qpos_seq[:, dof_idx] if dof_idx else qpos_seq
    dq = (
        np.diff(qv, axis=0)
        if qv.shape[0] > 1
        else np.zeros((0, qv.shape[1]), dtype=np.float64)
    )
    ddq = (
        np.diff(dq, axis=0)
        if dq.shape[0] > 1
        else np.zeros((0, qv.shape[1]), dtype=np.float64)
    )
    smooth_dq = float(np.mean(np.linalg.norm(dq, axis=1))) if dq.size else 0.0
    smooth_ddq = float(np.mean(np.linalg.norm(ddq, axis=1))) if ddq.size else 0.0

    sat = 0
    for j in range(model.njnt):
        jt = int(model.jnt_type[j])
        if jt not in (
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        ):
            continue
        if int(model.jnt_limited[j]) == 0:
            continue
        qadr = int(model.jnt_qposadr[j])
        lo, hi = float(model.jnt_range[j, 0]), float(model.jnt_range[j, 1])
        vals = qpos_seq[:, qadr]
        sat += int(
            np.count_nonzero((vals - lo) < 1e-3) + np.count_nonzero((hi - vals) < 1e-3)
        )

    shoulder_metric = np.nan
    if shoulder_roll_dof:
        roll_vals = qv[:, shoulder_roll_dof]
        shoulder_metric = float(np.mean(np.abs(roll_vals)))

    return {
        "ee_pos_mean_m": float(np.mean(ee_pos_err)) if ee_pos_err else 0.0,
        "ee_pos_p95_m": float(np.percentile(ee_pos_err, 95)) if ee_pos_err else 0.0,
        "ee_rot_mean_deg": float(np.degrees(np.mean(ee_rot_err)))
        if ee_rot_err
        else 0.0,
        "ee_rot_p95_deg": float(np.degrees(np.percentile(ee_rot_err, 95)))
        if ee_rot_err
        else 0.0,
        "stance_drift_max_mean_m": float(np.mean(stance_drifts))
        if stance_drifts
        else 0.0,
        "stance_drift_max_p95_m": float(np.percentile(stance_drifts, 95))
        if stance_drifts
        else 0.0,
        "smooth_dq_l2_mean": smooth_dq,
        "smooth_ddq_l2_mean": smooth_ddq,
        "joint_limit_saturation_count": float(sat),
        "shoulder_roll_abs_mean_rad": shoulder_metric,
        "lr_hand_asym_mean_m": float(np.mean(hand_asym)) if hand_asym else 0.0,
        "lr_hand_asym_p95_m": float(np.percentile(hand_asym, 95)) if hand_asym else 0.0,
        "lr_foot_asym_mean_m": float(np.mean(foot_asym)) if foot_asym else 0.0,
        "lr_foot_asym_p95_m": float(np.percentile(foot_asym, 95)) if foot_asym else 0.0,
    }


def _dump_debug_keybody_images(
    *,
    debug_dir: str,
    source_data: dict,
    source_time: np.ndarray,
    q_src: np.ndarray,
    src_model: mujoco.MjModel,
    src_data: mujoco.MjData,
    src_root: str,
    src_chest: Optional[str],
    src_ee: CanonicalEE,
    src_hints: Dict[str, str],
    desired: Dict[str, np.ndarray],
    hint_desired: Dict[str, np.ndarray],
    q_opt: np.ndarray,
    tgt_model: mujoco.MjModel,
    tgt_data: mujoco.MjData,
    frame_names: Dict[str, str],
    hint_names: Dict[str, str],
    symmetry_weight: np.ndarray,
    symmetry_lateral_axis: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[Debug] matplotlib unavailable, skip debug plots: {exc}")
        return

    os.makedirs(debug_dir, exist_ok=True)

    seq = source_data.get("timed_sequence", [])
    selected: List[Tuple[int, str, float]] = []
    seen = set()
    n = q_src.shape[0]
    for si, entry in enumerate(seq):
        if not (isinstance(entry, (list, tuple)) and len(entry) == 2):
            continue
        name = str(entry[0])
        try:
            t_arrive = float(entry[1])
        except Exception:
            continue
        if source_time.ndim == 1 and source_time.size > 0:
            idx = int(np.argmin(np.abs(source_time - t_arrive)))
        else:
            idx = int(
                round(
                    t_arrive
                    / max(
                        1e-6,
                        float(np.median(np.diff(source_time)))
                        if source_time.size > 1
                        else 0.02,
                    )
                )
            )
        idx = int(np.clip(idx, 0, n - 1))
        key = (idx, name)
        if key in seen:
            continue
        keep = (
            si < 12
            or name.startswith("crawl")
            or name.startswith("up")
            or name.startswith("down")
            or name == "default_loco"
        )
        if keep:
            selected.append((idx, name, t_arrive))
            seen.add(key)
        if len(selected) >= 24:
            break
    if not selected:
        selected = [
            (0, "frame0", 0.0),
            (
                min(n - 1, n // 2),
                "mid",
                float(source_time[min(n - 1, n // 2)]) if source_time.size else 0.0,
            ),
            (n - 1, "last", float(source_time[-1]) if source_time.size else 0.0),
        ]

    def _collect_source_local(idx: int) -> Dict[str, np.ndarray]:
        src_data.qpos[:] = q_src[idx]
        src_data.qvel[:] = 0
        mujoco.mj_forward(src_model, src_data)
        rp, rq, _ = _frame_pose(src_model, src_data, src_root)
        out: Dict[str, np.ndarray] = {"root": np.zeros(3, dtype=np.float64)}

        if src_chest:
            p, q, _ = _frame_pose(src_model, src_data, src_chest)
            out["chest"] = _to_local_pose(rp, rq, p, q)[0]
        for k, frm in src_ee.as_dict().items():
            p, q, _ = _frame_pose(src_model, src_data, frm)
            out[k] = _to_local_pose(rp, rq, p, q)[0]
        for hk, frm in src_hints.items():
            p, q, _ = _frame_pose(src_model, src_data, frm)
            out[hk] = _to_local_pose(rp, rq, p, q)[0]
        return out

    def _collect_desired_local(idx: int) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {"root": np.zeros(3, dtype=np.float64)}
        rp = desired["root"][idx, :3]
        rq = desired["root"][idx, 3:7]
        if "chest" in desired:
            out["chest"] = _to_local_pose(
                rp, rq, desired["chest"][idx, :3], desired["chest"][idx, 3:7]
            )[0]
        for k in ("left_foot", "right_foot", "left_hand", "right_hand"):
            out[k] = _to_local_pose(rp, rq, desired[k][idx, :3], desired[k][idx, 3:7])[
                0
            ]
        for hk, arr in hint_desired.items():
            out[hk] = R.from_quat(rq, scalar_first=True).inv().apply(arr[idx] - rp)
        return out

    def _collect_target_local(idx: int) -> Dict[str, np.ndarray]:
        tgt_data.qpos[:] = q_opt[idx]
        tgt_data.qvel[:] = 0
        mujoco.mj_forward(tgt_model, tgt_data)
        rp, rq, _ = _frame_pose(tgt_model, tgt_data, frame_names["root"])
        out: Dict[str, np.ndarray] = {"root": np.zeros(3, dtype=np.float64)}
        if "chest" in frame_names:
            p, q, _ = _frame_pose(tgt_model, tgt_data, frame_names["chest"])
            out["chest"] = _to_local_pose(rp, rq, p, q)[0]
        for k in ("left_foot", "right_foot", "left_hand", "right_hand"):
            p, q, _ = _frame_pose(tgt_model, tgt_data, frame_names[k])
            out[k] = _to_local_pose(rp, rq, p, q)[0]
        for hk, frm in hint_names.items():
            p, q, _ = _frame_pose(tgt_model, tgt_data, frm)
            out[hk] = _to_local_pose(rp, rq, p, q)[0]
        return out

    draw_keys = [
        "root",
        "chest",
        "left_hand",
        "right_hand",
        "left_foot",
        "right_foot",
        "left_elbow",
        "right_elbow",
        "left_knee",
        "right_knee",
    ]
    links = [
        ("root", "chest"),
        ("root", "left_hand"),
        ("root", "right_hand"),
        ("root", "left_foot"),
        ("root", "right_foot"),
        ("left_hand", "left_elbow"),
        ("right_hand", "right_elbow"),
        ("left_foot", "left_knee"),
        ("right_foot", "right_knee"),
    ]

    summary_lines = ["idx,name,time,symmetry_weight"]
    for idx, name, t_arrive in selected:
        src_pts = _collect_source_local(idx)
        des_pts = _collect_desired_local(idx)
        tgt_pts = _collect_target_local(idx)

        fig, axs = plt.subplots(1, 2, figsize=(10, 4), dpi=130)
        sets = [
            (src_pts, "#1f77b4", "source"),
            (des_pts, "#ff7f0e", "mapped"),
            (tgt_pts, "#2ca02c", "target"),
        ]
        proj = [((0, 2), "X-Z"), ((0, 1), "X-Y")]
        for ax, ((a0, a1), ttl) in zip(axs, proj):
            for pts, color, lbl in sets:
                for u, v in links:
                    if u in pts and v in pts:
                        pu = pts[u]
                        pv = pts[v]
                        ax.plot(
                            [pu[a0], pv[a0]],
                            [pu[a1], pv[a1]],
                            color=color,
                            alpha=0.35,
                            linewidth=1.0,
                        )
                xs, ys = [], []
                for k in draw_keys:
                    if k in pts:
                        xs.append(float(pts[k][a0]))
                        ys.append(float(pts[k][a1]))
                if xs:
                    ax.scatter(xs, ys, s=18, c=color, label=lbl)
            ax.axvline(0.0, color="#888", linewidth=0.6)
            ax.axhline(0.0, color="#888", linewidth=0.6)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(ttl)
        axs[0].legend(loc="upper right", fontsize=8)
        sw = float(symmetry_weight[idx]) if idx < symmetry_weight.shape[0] else 0.0
        fig.suptitle(
            f"idx={idx}  name={name}  t={t_arrive:.2f}s  sym_w={sw:.2f}  lat_axis={symmetry_lateral_axis}",
            fontsize=10,
        )
        fig.tight_layout()
        safe_name = name.replace("/", "_")
        out_png = os.path.join(debug_dir, f"keybody_{idx:04d}_{safe_name}.png")
        fig.savefig(out_png)
        plt.close(fig)

        summary_lines.append(f"{idx},{safe_name},{t_arrive:.3f},{sw:.4f}")

    with open(
        os.path.join(debug_dir, "keybody_debug_summary.csv"), "w", encoding="utf-8"
    ) as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"[Debug] Saved keybody matching images to: {debug_dir}")


def retarget(
    source_xml: str,
    target_xml: str,
    source_motion: str,
    output_motion: str,
    *,
    source_root: Optional[str],
    target_root: Optional[str],
    dt_default: float,
    foot_contact_z: float,
    hand_contact_z: float,
    outer_iters: int,
    window: int,
    overlap: int,
    step_scale: float,
    ground_clearance: float,
    metrics_out: Optional[str],
    debug_dir: Optional[str],
) -> Dict[str, float]:
    t0 = time.perf_counter()
    src_model = mujoco.MjModel.from_xml_path(source_xml)
    src_data = mujoco.MjData(src_model)
    tgt_model = mujoco.MjModel.from_xml_path(target_xml)
    tgt_data = mujoco.MjData(tgt_model)

    src_root = source_root if source_root else _auto_detect_root_body(src_model)
    tgt_root = target_root if target_root else _auto_detect_root_body(tgt_model)
    src_ee = _canonicalize_ee(src_model)
    tgt_ee = _canonicalize_ee(tgt_model)
    src_chest = _find_chest_frame(src_model, src_root)
    tgt_chest = _find_chest_frame(tgt_model, tgt_root)
    src_hints = _find_limb_hints(src_model)
    tgt_hints = _find_limb_hints(tgt_model)

    source_data = joblib.load(source_motion)
    q_src = _extract_qpos_sequence(source_data)
    dt = _extract_dt(source_data, dt_default)
    n = q_src.shape[0]
    src_time = np.asarray(source_data.get("time", []), dtype=np.float64)

    src_data.qpos[:] = q_src[0]
    src_data.qvel[:] = 0
    mujoco.mj_forward(src_model, src_data)
    src_root0, _, _ = _frame_pose(src_model, src_data, src_root)
    src_ee0 = {
        k: _frame_pose(src_model, src_data, v)[0] for k, v in src_ee.as_dict().items()
    }
    src_ch0 = _frame_pose(src_model, src_data, src_chest)[0] if src_chest else None

    tgt_q0 = _home_qpos(tgt_model)
    tgt_data.qpos[:] = tgt_q0
    tgt_data.qvel[:] = 0
    mujoco.mj_forward(tgt_model, tgt_data)
    tgt_root0, _, _ = _frame_pose(tgt_model, tgt_data, tgt_root)
    tgt_ee0 = {
        k: _frame_pose(tgt_model, tgt_data, v)[0] for k, v in tgt_ee.as_dict().items()
    }
    tgt_ch0 = _frame_pose(tgt_model, tgt_data, tgt_chest)[0] if tgt_chest else None

    scales = _compute_limb_scales(
        src_root0, src_ee0, src_ch0, tgt_root0, tgt_ee0, tgt_ch0
    )
    print(
        "[Retarget] scales "
        f"leg={scales['leg']:.3f} arm={scales['arm']:.3f} torso={scales['torso']:.3f} "
        f"root_xy={scales['root_xy']:.3f} root_z={scales['root_z']:.3f}"
    )

    # Source contact phases from source canonical EEs.
    src_tracks = {}
    for key, frame in src_ee.as_dict().items():
        arr = np.zeros((n, 7), dtype=np.float64)
        for i in range(n):
            src_data.qpos[:] = q_src[i]
            src_data.qvel[:] = 0
            mujoco.mj_forward(src_model, src_data)
            p, q, _ = _frame_pose(src_model, src_data, frame)
            arr[i, :3] = p
            arr[i, 3:7] = q
        src_tracks[key] = arr
    src_contact = _contact_plans_from_source(
        src_tracks, dt, foot_contact_z, hand_contact_z
    )
    symmetry_weight, symmetry_lateral_axis = _source_symmetry_profile(
        src_model=src_model,
        src_data=src_data,
        q_src=q_src,
        src_root=src_root,
        src_ee=src_ee,
        src_hints=src_hints,
    )

    q_warm, desired, hint_desired, hint_names = _solve_warm_start_ik(
        src_model=src_model,
        src_data=src_data,
        tgt_model=tgt_model,
        tgt_data=tgt_data,
        q_src=q_src,
        src_root=src_root,
        tgt_root=tgt_root,
        src_chest=src_chest,
        tgt_chest=tgt_chest,
        src_ee=src_ee,
        tgt_ee=tgt_ee,
        src_hints=src_hints,
        tgt_hints=tgt_hints,
        scales=scales,
        dt=dt,
    )
    mapped_contacts = _build_mapped_contacts(desired, src_contact)
    seq_anchor_weight, seq_anchor_target_q = _sequence_anchor_targets(
        source_data=source_data,
        source_time=src_time,
        q_src=q_src,
        q_warm_tgt=q_warm,
        dt=dt,
    )

    frame_names = {
        "root": tgt_root,
        "left_foot": tgt_ee.left_foot,
        "right_foot": tgt_ee.right_foot,
    }
    frame_names["left_hand"] = tgt_ee.left_hand
    frame_names["right_hand"] = tgt_ee.right_hand
    if tgt_chest:
        frame_names["chest"] = tgt_chest

    robot_name = _infer_robot_name_from_xml(target_xml)
    q_opt = _temporal_optimize(
        model=tgt_model,
        data=tgt_data,
        q_init=q_warm,
        frame_names=frame_names,
        desired=desired,
        contacts=mapped_contacts,
        hint_names=hint_names,
        hint_desired=hint_desired,
        seq_anchor_weight=seq_anchor_weight,
        seq_anchor_target_q=seq_anchor_target_q,
        symmetry_weight=symmetry_weight,
        symmetry_lateral_axis=symmetry_lateral_axis,
        robot_name=robot_name,
        outer_iters=outer_iters,
        window=window,
        overlap=overlap,
        step_scale=step_scale,
    )
    q_opt = _foot_floor_alignment(
        tgt_model,
        tgt_data,
        q_opt,
        tgt_ee.left_foot,
        tgt_ee.right_foot,
        clearance=ground_clearance,
    )

    if debug_dir:
        _dump_debug_keybody_images(
            debug_dir=debug_dir,
            source_data=source_data,
            source_time=src_time,
            q_src=q_src,
            src_model=src_model,
            src_data=src_data,
            src_root=src_root,
            src_chest=src_chest,
            src_ee=src_ee,
            src_hints=src_hints,
            desired=desired,
            hint_desired=hint_desired,
            q_opt=q_opt,
            tgt_model=tgt_model,
            tgt_data=tgt_data,
            frame_names=frame_names,
            hint_names=hint_names,
            symmetry_weight=symmetry_weight,
            symmetry_lateral_axis=symmetry_lateral_axis,
        )

    out_dict = _build_output_dict(source_data, q_opt, dt, tgt_model, tgt_data)
    os.makedirs(os.path.dirname(output_motion), exist_ok=True)
    joblib.dump(out_dict, output_motion, compress="lz4")
    print(f"[Retarget] Saved: {output_motion}")

    # Metrics against mapped desired tasks.
    qpos_idx, dof_idx, _ = _free_joint_and_hinge_vars(tgt_model)
    shoulder_q_vars = _shoulder_roll_var_indices(tgt_model, qpos_idx)
    metrics = _eval_metrics(
        model=tgt_model,
        data=tgt_data,
        qpos_seq=q_opt,
        desired=desired,
        contacts=mapped_contacts,
        frame_names=frame_names,
        dof_idx=qpos_idx,
        shoulder_roll_dof=shoulder_q_vars,
    )
    metrics["runtime_sec"] = float(time.perf_counter() - t0)
    metrics["n_frames"] = float(n)
    if metrics_out:
        os.makedirs(os.path.dirname(metrics_out), exist_ok=True)
        joblib.dump(metrics, metrics_out)
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Robot-to-robot temporal retargeting")
    p.add_argument("--source-xml", required=True)
    p.add_argument("--target-xml", required=True)
    p.add_argument("--source-motion", required=True)
    p.add_argument("--output-motion", required=True)
    p.add_argument("--source-root", default=None)
    p.add_argument("--target-root", default=None)
    p.add_argument("--dt-default", type=float, default=0.02)
    p.add_argument("--foot-contact-z", type=float, default=0.05)
    p.add_argument("--hand-contact-z", type=float, default=0.10)
    p.add_argument("--outer-iters", type=int, default=4)
    p.add_argument("--window", type=int, default=120)
    p.add_argument("--overlap", type=int, default=60)
    p.add_argument("--step-scale", type=float, default=0.85)
    p.add_argument("--ground-clearance", type=float, default=0.005)
    p.add_argument("--metrics-out", default=None)
    p.add_argument("--debug-dir", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    metrics = retarget(
        source_xml=args.source_xml,
        target_xml=args.target_xml,
        source_motion=args.source_motion,
        output_motion=args.output_motion,
        source_root=args.source_root,
        target_root=args.target_root,
        dt_default=args.dt_default,
        foot_contact_z=args.foot_contact_z,
        hand_contact_z=args.hand_contact_z,
        outer_iters=max(1, int(args.outer_iters)),
        window=max(16, int(args.window)),
        overlap=max(0, int(args.overlap)),
        step_scale=float(np.clip(args.step_scale, 0.1, 1.0)),
        ground_clearance=max(0.0, float(args.ground_clearance)),
        metrics_out=args.metrics_out,
        debug_dir=args.debug_dir,
    )
    print("[Metrics]")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
