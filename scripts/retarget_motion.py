#!/usr/bin/env python3
"""Trajectory-level retargeting using MuJoCo + Mink IK.

This script retargets a source .lz4 motion file to a target MuJoCo model by
solving IK frame-by-frame against canonical end-effectors and root pose.
Output is compatible with robot-keyframe-kit data loading.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import joblib
import mujoco
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

try:
    import mink
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "mink is required for retargeting. Install with: pip install mink==0.0.13"
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
class StanceAnchor:
    active: bool
    pos: np.ndarray
    quat: np.ndarray


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
    """Return candidate EE frames as (name, frame_type)."""
    # Find leaf bodies.
    parent_ids = {int(i) for i in model.body_parentid}
    leaf_body_ids = [i for i in range(model.nbody) if i not in parent_ids]

    candidates: List[Tuple[str, str, int]] = []

    # 1) Prefer sites attached to leaf bodies.
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
            low = site_name.lower()
            if any(k in low for k in EE_KEYWORDS):
                candidates.append((site_name, "site", 3))

    # 2) Fallback to leaf bodies with EE-like names.
    for body_id in leaf_body_ids:
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if not body_name or body_name == "world":
            continue
        low = body_name.lower()
        if any(k in low for k in EE_KEYWORDS):
            candidates.append((body_name, "body", 2))

    # 3) Final fallback: all leaf bodies.
    if not candidates:
        for body_id in leaf_body_ids:
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if body_name and body_name != "world":
                candidates.append((body_name, "body", 1))

    candidates.sort(key=lambda x: (-x[2], x[0]))
    return [(n, t) for n, t, _ in candidates]


def _canonicalize_ee(model: mujoco.MjModel) -> CanonicalEE:
    candidates = _discover_ee_frames(model)
    slots: Dict[str, Optional[Tuple[str, str]]] = {
        "left_foot": None,
        "right_foot": None,
        "left_hand": None,
        "right_hand": None,
    }

    def score(name: str, frame_type: str) -> int:
        s = 100 if frame_type == "site" else 0
        low = name.lower()
        if "center" in low or "tip" in low or "palm" in low:
            s += 20
        if "link" not in low:
            s += 5
        return s

    for name, frame_type in candidates:
        limb = _limb_type(name)
        if limb is None:
            continue
        side = "left" if _is_left(name) else "right" if _is_right(name) else None
        if side is None:
            continue
        key = f"{side}_{limb}"
        cur = slots[key]
        if cur is None or score(name, frame_type) > score(cur[0], cur[1]):
            slots[key] = (name, frame_type)

    missing = [k for k, v in slots.items() if v is None]
    if missing:
        raise RuntimeError(f"Failed to detect canonical EEs: missing {missing}")

    return CanonicalEE(
        left_foot=slots["left_foot"][0],  # type: ignore[index]
        right_foot=slots["right_foot"][0],  # type: ignore[index]
        left_hand=slots["left_hand"][0],  # type: ignore[index]
        right_hand=slots["right_hand"][0],  # type: ignore[index]
    )


def _find_body_name(
    model: mujoco.MjModel, required_tokens: Tuple[str, ...]
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
    """Find optional knee/elbow body frames used as IK bend hints."""
    hints: Dict[str, str] = {}
    for side in ("left", "right"):
        knee = _find_body_name(model, (side, "knee"))
        if knee:
            hints[f"{side}_knee"] = knee
        elbow = _find_body_name(model, (side, "elbow"))
        if elbow:
            hints[f"{side}_elbow"] = elbow
    return hints


def _frame_pose(
    model: mujoco.MjModel, data: mujoco.MjData, frame_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, frame_name)
    if site_id >= 0:
        pos = data.site(frame_name).xpos.copy().astype(np.float64)
        mat = data.site(frame_name).xmat.reshape(3, 3).copy()
        quat = R.from_matrix(mat).as_quat(scalar_first=True).astype(np.float64)
        return pos, quat

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, frame_name)
    if body_id >= 0:
        pos = data.xpos[body_id].copy().astype(np.float64)
        mat = data.xmat[body_id].reshape(3, 3).copy()
        quat = R.from_matrix(mat).as_quat(scalar_first=True).astype(np.float64)
        return pos, quat

    raise ValueError(f"Unknown site/body frame: {frame_name}")


def _to_local_pose(
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    world_pos: np.ndarray,
    world_quat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    rr = R.from_quat(root_quat, scalar_first=True)
    rw = R.from_quat(world_quat, scalar_first=True)
    p_local = rr.inv().apply(world_pos - root_pos)
    q_local = (rr.inv() * rw).as_quat(scalar_first=True)
    return p_local.astype(np.float64), q_local.astype(np.float64)


def _to_world_pose(
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    local_pos: np.ndarray,
    local_quat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    rr = R.from_quat(root_quat, scalar_first=True)
    rl = R.from_quat(local_quat, scalar_first=True)
    p_world = root_pos + rr.apply(local_pos)
    q_world = (rr * rl).as_quat(scalar_first=True)
    return p_world.astype(np.float64), q_world.astype(np.float64)


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
    raise ValueError("Motion file has neither dense qpos trajectory nor keyframe qpos")


def _extract_dt(motion_data: dict, default_dt: float) -> float:
    t = np.asarray(motion_data.get("time", []), dtype=np.float64)
    if t.ndim == 1 and t.size > 1:
        dt = float(np.median(np.diff(t)))
        if np.isfinite(dt) and dt > 1e-6:
            return dt
    return default_dt


def _trajectory_variation(qpos: np.ndarray) -> float:
    if qpos.ndim != 2 or qpos.shape[0] < 2:
        return 0.0
    return float(np.mean(np.std(qpos, axis=0)))


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


def _compute_scale_from_feet(
    src_root_pos: np.ndarray,
    src_ee_pos: Dict[str, np.ndarray],
    tgt_root_pos: np.ndarray,
    tgt_ee_pos: Dict[str, np.ndarray],
) -> float:
    src_len = 0.5 * (
        np.linalg.norm(src_ee_pos["left_foot"] - src_root_pos)
        + np.linalg.norm(src_ee_pos["right_foot"] - src_root_pos)
    )
    tgt_len = 0.5 * (
        np.linalg.norm(tgt_ee_pos["left_foot"] - tgt_root_pos)
        + np.linalg.norm(tgt_ee_pos["right_foot"] - tgt_root_pos)
    )
    return float(tgt_len / max(src_len, 1e-8))


def _make_mink_target_pose(target_pos: np.ndarray, target_wxyz: np.ndarray) -> object:
    q = np.asarray(target_wxyz, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return mink.SE3.from_translation(np.asarray(target_pos, dtype=np.float64))
    q = q / n
    rot = R.from_quat(q, scalar_first=True).as_matrix()
    return mink.SE3.from_rotation_and_translation(
        mink.SO3.from_matrix(rot), np.asarray(target_pos, dtype=np.float64)
    )


def _smooth_qpos(qpos: np.ndarray, window: int = 9, poly: int = 3) -> np.ndarray:
    if qpos.shape[0] < 5:
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
            qpos[:, start:], window_length=w, polyorder=poly, axis=0
        )
    return out


def _clamp_qpos_to_joint_limits(
    model: mujoco.MjModel, qpos_seq: np.ndarray
) -> Tuple[np.ndarray, int, float]:
    out = np.asarray(qpos_seq, dtype=np.float64).copy()
    limited: List[Tuple[int, float, float]] = []
    for j in range(model.njnt):
        jt = int(model.jnt_type[j])
        if jt not in (
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        ):
            continue
        if int(model.jnt_limited[j]) == 0:
            continue
        limited.append(
            (
                int(model.jnt_qposadr[j]),
                float(model.jnt_range[j, 0]),
                float(model.jnt_range[j, 1]),
            )
        )
    if not limited:
        return out, 0, 0.0
    count = 0
    max_change = 0.0
    for i in range(out.shape[0]):
        for adr, lo, hi in limited:
            x = float(out[i, adr])
            xc = float(np.clip(x, lo, hi))
            if xc != x:
                out[i, adr] = xc
                count += 1
                max_change = max(max_change, abs(xc - x))
    return out, count, max_change


def _find_root_z_qpos_index(model: mujoco.MjModel) -> Optional[int]:
    for j in range(model.njnt):
        if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
            return int(model.jnt_qposadr[j]) + 2
    return None


def _sequence_min_foot_z(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_seq: np.ndarray,
    left_foot_frame: str,
    right_foot_frame: str,
) -> float:
    min_z = float("inf")
    prev_q = data.qpos.copy()
    prev_v = data.qvel.copy()
    try:
        for i in range(qpos_seq.shape[0]):
            data.qpos[:] = qpos_seq[i]
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            p_l, _ = _frame_pose(model, data, left_foot_frame)
            p_r, _ = _frame_pose(model, data, right_foot_frame)
            min_z = min(min_z, float(p_l[2]), float(p_r[2]))
    finally:
        data.qpos[:] = prev_q
        data.qvel[:] = prev_v
        mujoco.mj_forward(model, data)
    return min_z


def _foot_z_stats(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_seq: np.ndarray,
    left_foot_frame: str,
    right_foot_frame: str,
) -> Tuple[float, float, float]:
    """Return (first_avg_z, min_z, p5_z) for foot frames over a sequence."""
    prev_q = data.qpos.copy()
    prev_v = data.qvel.copy()
    z_vals: List[float] = []
    try:
        for i in range(qpos_seq.shape[0]):
            data.qpos[:] = qpos_seq[i]
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            p_l, _ = _frame_pose(model, data, left_foot_frame)
            p_r, _ = _frame_pose(model, data, right_foot_frame)
            z_vals.append(float(p_l[2]))
            z_vals.append(float(p_r[2]))
    finally:
        data.qpos[:] = prev_q
        data.qvel[:] = prev_v
        mujoco.mj_forward(model, data)
    if not z_vals:
        return 0.0, 0.0, 0.0
    z_arr = np.asarray(z_vals, dtype=np.float64)
    return (
        float(z_arr[0:2].mean()),
        float(z_arr.min()),
        float(np.percentile(z_arr, 5.0)),
    )


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
    """Pick an installed QP solver for Mink IK."""
    try:
        import qpsolvers  # type: ignore

        available = set(getattr(qpsolvers, "available_solvers", []))
    except Exception:
        available = set()

    for candidate in ("daqp", "quadprog", "osqp", "jaxopt_osqp"):
        if not available or candidate in available:
            return candidate
    return "quadprog"


def retarget_motion(
    source_xml: str,
    target_xml: str,
    source_motion: str,
    output_motion: str,
    *,
    source_root: Optional[str],
    target_root: Optional[str],
    max_frames: Optional[int],
    frame_stride: int,
    smooth: bool,
    dt_default: float,
    foot_contact_z: float,
    foot_contact_v: float,
    hand_contact_z: float,
    hand_contact_v: float,
    ground_clearance: float,
) -> None:
    src_model = mujoco.MjModel.from_xml_path(source_xml)
    src_data = mujoco.MjData(src_model)
    tgt_model = mujoco.MjModel.from_xml_path(target_xml)
    tgt_data = mujoco.MjData(tgt_model)

    src_root = source_root if source_root else _auto_detect_root_body(src_model)
    tgt_root = target_root if target_root else _auto_detect_root_body(tgt_model)

    src_ee = _canonicalize_ee(src_model)
    tgt_ee = _canonicalize_ee(tgt_model)
    print(f"[Retarget] Source root: {src_root}")
    print(f"[Retarget] Target root: {tgt_root}")
    print(f"[Retarget] Source EEs: {src_ee.as_dict()}")
    print(f"[Retarget] Target EEs: {tgt_ee.as_dict()}")

    source_data = joblib.load(source_motion)
    qpos_src_all = _extract_qpos_sequence(source_data)
    dt = _extract_dt(source_data, default_dt=dt_default)
    src_var = _trajectory_variation(qpos_src_all)
    if src_var < 1e-5:
        print(
            "[Retarget] WARNING: source qpos trajectory has very low variation; "
            "motion may be static or only keyframe-level."
        )

    qpos_src = qpos_src_all[:: max(1, frame_stride)]
    if max_frames is not None:
        qpos_src = qpos_src[:max_frames]
    n_frames = qpos_src.shape[0]
    if n_frames < 2:
        raise ValueError("Need at least 2 source frames for retargeting")

    src_data.qpos[:] = qpos_src[0]
    src_data.qvel[:] = 0
    mujoco.mj_forward(src_model, src_data)
    src_root_pos0, src_root_quat0 = _frame_pose(src_model, src_data, src_root)
    src_ee_pos0 = {
        k: _frame_pose(src_model, src_data, v)[0] for k, v in src_ee.as_dict().items()
    }

    q_tgt = _home_qpos(tgt_model)
    tgt_data.qpos[:] = q_tgt
    tgt_data.qvel[:] = 0
    mujoco.mj_forward(tgt_model, tgt_data)
    tgt_root_pos0, tgt_root_quat0 = _frame_pose(tgt_model, tgt_data, tgt_root)
    tgt_ee_pos0 = {
        k: _frame_pose(tgt_model, tgt_data, v)[0] for k, v in tgt_ee.as_dict().items()
    }

    scale = _compute_scale_from_feet(
        src_root_pos0, src_ee_pos0, tgt_root_pos0, tgt_ee_pos0
    )
    print(f"[Retarget] Length scale: {scale:.4f}")
    ik_solver = _select_ik_solver()
    print(f"[Retarget] IK solver: {ik_solver}")
    src_hints = _find_limb_hints(src_model)
    tgt_hints = _find_limb_hints(tgt_model)
    common_hints = {
        k: (src_hints[k], tgt_hints[k]) for k in sorted(set(src_hints) & set(tgt_hints))
    }
    if common_hints:
        print(f"[Retarget] Limb hints: {common_hints}")

    cfg = mink.Configuration(tgt_model)
    posture_task = mink.PostureTask(tgt_model, cost=1e-2)
    limits: List[object] = [mink.ConfigurationLimit(tgt_model)]

    anchors = {
        key: StanceAnchor(
            False,
            np.zeros(3, dtype=np.float64),
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        )
        for key in ("left_foot", "right_foot", "left_hand", "right_hand")
    }
    prev_src_contact = {
        key: _frame_pose(src_model, src_data, src_ee.as_dict()[key])[0].copy()
        for key in ("left_foot", "right_foot", "left_hand", "right_hand")
    }

    r_src0 = R.from_quat(src_root_quat0, scalar_first=True)
    r_tgt0 = R.from_quat(tgt_root_quat0, scalar_first=True)

    qpos_tgt_seq = np.zeros((n_frames, tgt_model.nq), dtype=np.float64)

    for i in range(n_frames):
        src_data.qpos[:] = qpos_src[i]
        src_data.qvel[:] = 0
        mujoco.mj_forward(src_model, src_data)

        src_root_pos, src_root_quat = _frame_pose(src_model, src_data, src_root)
        r_src = R.from_quat(src_root_quat, scalar_first=True)
        root_pos_des = tgt_root_pos0 + scale * (src_root_pos - src_root_pos0)
        root_quat_des = (r_tgt0 * (r_src0.inv() * r_src)).as_quat(scalar_first=True)

        desired_world: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for key, src_frame in src_ee.as_dict().items():
            src_p, src_q = _frame_pose(src_model, src_data, src_frame)
            p_local, q_local = _to_local_pose(src_root_pos, src_root_quat, src_p, src_q)
            p_world, q_world = _to_world_pose(
                root_pos_des, root_quat_des, p_local * scale, q_local
            )
            desired_world[key] = (p_world, q_world)
        desired_hints: Dict[str, np.ndarray] = {}
        for hint_key, (src_frame, _tgt_frame) in common_hints.items():
            src_p, _ = _frame_pose(src_model, src_data, src_frame)
            p_local, _ = _to_local_pose(
                src_root_pos, src_root_quat, src_p, src_root_quat
            )
            p_world, _ = _to_world_pose(
                root_pos_des,
                root_quat_des,
                p_local * scale,
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            )
            desired_hints[hint_key] = p_world

        # Contact anchors for feet and hands (hands matter for crawling motions).
        for contact_key in ("left_foot", "right_foot", "left_hand", "right_hand"):
            src_pos_now = _frame_pose(
                src_model, src_data, src_ee.as_dict()[contact_key]
            )[0]
            src_speed = float(
                np.linalg.norm(
                    (src_pos_now - prev_src_contact[contact_key]) / max(dt, 1e-6)
                )
            )
            prev_src_contact[contact_key] = src_pos_now
            if "foot" in contact_key:
                z_thr = foot_contact_z
                v_thr = foot_contact_v
            else:
                z_thr = hand_contact_z
                v_thr = hand_contact_v
            in_contact = bool(src_pos_now[2] < z_thr and src_speed < v_thr)

            a = anchors[contact_key]
            if in_contact and not a.active:
                a.active = True
                a.pos = desired_world[contact_key][0].copy()
                a.quat = desired_world[contact_key][1].copy()
            elif not in_contact and a.active:
                a.active = False

            if a.active:
                desired_world[contact_key] = (a.pos.copy(), a.quat.copy())

        cfg.update(q=q_tgt.copy())
        posture_task.set_target(cfg.q)
        tasks: List[object] = []

        root_task = mink.FrameTask(
            frame_name=tgt_root,
            frame_type="body",
            position_cost=4.0,
            orientation_cost=3.0,
            lm_damping=1e-3,
        )
        root_task.set_target(_make_mink_target_pose(root_pos_des, root_quat_des))
        tasks.append(root_task)

        for key, tgt_frame in tgt_ee.as_dict().items():
            site_id = mujoco.mj_name2id(tgt_model, mujoco.mjtObj.mjOBJ_SITE, tgt_frame)
            frame_type = "site" if site_id >= 0 else "body"
            pos_cost = 6.0 if "foot" in key else 3.0
            rot_cost = 0.40 if "foot" in key else 0.10
            task = mink.FrameTask(
                frame_name=tgt_frame,
                frame_type=frame_type,
                position_cost=pos_cost,
                orientation_cost=rot_cost,
                lm_damping=1e-3,
            )
            tp, tq = desired_world[key]
            task.set_target(_make_mink_target_pose(tp, tq))
            tasks.append(task)
        for hint_key, (_src_frame, tgt_frame) in common_hints.items():
            task = mink.FrameTask(
                frame_name=tgt_frame,
                frame_type="body",
                position_cost=1.4 if "knee" in hint_key else 1.0,
                orientation_cost=0.0,
                lm_damping=1e-3,
            )
            tp = desired_hints[hint_key]
            task.set_target(
                _make_mink_target_pose(
                    tp, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
                )
            )
            tasks.append(task)

        tasks.append(posture_task)

        best_q = cfg.q.copy()
        best_score = float("inf")
        ik_dt = 0.1
        for _ in range(24):
            tgt_data.qpos[:] = cfg.q
            tgt_data.qvel[:] = 0
            mujoco.mj_forward(tgt_model, tgt_data)
            root_now, _ = _frame_pose(tgt_model, tgt_data, tgt_root)
            score = float(np.linalg.norm(root_now - root_pos_des))
            for key, tgt_frame in tgt_ee.as_dict().items():
                p_now, _ = _frame_pose(tgt_model, tgt_data, tgt_frame)
                score += float(np.linalg.norm(p_now - desired_world[key][0]))
            if score < best_score:
                best_score = score
                best_q = cfg.q.copy()
            try:
                vel = mink.solve_ik(
                    cfg, tasks, ik_dt, solver=ik_solver, damping=1e-3, limits=limits
                )
            except Exception as exc:
                if i == 0:
                    print(f"[Retarget] IK solve failed ({type(exc).__name__}): {exc}")
                break
            n = float(np.linalg.norm(vel))
            if n > 20.0 and n > 1e-12:
                vel *= 20.0 / n
            cfg.integrate_inplace(vel, ik_dt)

        q_tgt = best_q.copy()
        qpos_tgt_seq[i] = q_tgt

        if (i + 1) % 100 == 0 or i == n_frames - 1:
            print(f"[Retarget] Solved frame {i + 1}/{n_frames}")

    if smooth:
        qpos_tgt_seq = _smooth_qpos(qpos_tgt_seq)
        print("[Retarget] Applied Savitzky-Golay smoothing to qpos")

    # Limit clamp postprocess.
    qpos_tgt_seq, clipped, max_clip = _clamp_qpos_to_joint_limits(
        tgt_model, qpos_tgt_seq
    )
    if clipped > 0:
        print(
            f"[Retarget] Clamped {clipped} joint values to limits (max change={max_clip:.4f})"
        )

    # Align sequence vertical placement (if free base exists).
    z_index = _find_root_z_qpos_index(tgt_model)
    if z_index is not None:
        # 1) Align first-frame feet to target home feet height.
        tgt_data.qpos[:] = _home_qpos(tgt_model)
        tgt_data.qvel[:] = 0
        mujoco.mj_forward(tgt_model, tgt_data)
        home_lz = float(_frame_pose(tgt_model, tgt_data, tgt_ee.left_foot)[0][2])
        home_rz = float(_frame_pose(tgt_model, tgt_data, tgt_ee.right_foot)[0][2])
        home_avg = 0.5 * (home_lz + home_rz)

        first_avg, min_foot_z, p5_foot_z = _foot_z_stats(
            tgt_model,
            tgt_data,
            qpos_tgt_seq,
            tgt_ee.left_foot,
            tgt_ee.right_foot,
        )
        dz_align = float(home_avg - first_avg)
        if abs(dz_align) > 1e-6:
            qpos_tgt_seq[:, z_index] += dz_align
            min_foot_z += dz_align
            p5_foot_z += dz_align
            print(
                f"[Retarget] Aligned first-frame foot height by root z shift {dz_align:.4f} m"
            )

        # 2) Robust anti-penetration shift using low percentile, not global min outlier.
        if np.isfinite(p5_foot_z) and p5_foot_z < ground_clearance:
            dz = float(ground_clearance - p5_foot_z)
            qpos_tgt_seq[:, z_index] += dz
            print(
                f"[Retarget] Lifted root z by {dz:.4f} m "
                f"(p5 foot z {p5_foot_z:.4f} -> {ground_clearance:.4f})"
            )

    # Recompute action from final qpos for consistency.
    action_seq = np.zeros((n_frames, tgt_model.nu), dtype=np.float32)
    for i in range(n_frames):
        motor_pos, _ = _compute_motor_and_joint_pos(
            tgt_model, tgt_data, qpos_tgt_seq[i]
        )
        if motor_pos.shape[0] == tgt_model.nu:
            action_seq[i] = motor_pos

    # Build keyframes/sequence.
    # Important: if source timed_sequence repeats names (common in crawl motions),
    # keep one *unique* retargeted keyframe per sequence row so Play Qpos can
    # preserve forward progression instead of reusing identical named poses.
    out_keyframes: List[dict] = []
    out_seq: List[Tuple[str, float]] = []
    src_keyframes = source_data.get("keyframes", [])
    src_time = np.asarray(source_data.get("time", []), dtype=np.float64)
    seq = source_data.get("timed_sequence", [])

    def _src_index_from_time(t_query: float) -> int:
        if src_time.ndim == 1 and src_time.size > 0:
            return int(np.argmin(np.abs(src_time - float(t_query))))
        # Fallback if source has no explicit dense time array.
        return int(
            np.clip(
                round((t_query / max(dt, 1e-6))),
                0,
                qpos_src_all.shape[0] - 1,
            )
        )

    if isinstance(seq, list) and len(seq) > 0:
        for si, it in enumerate(seq):
            if not (isinstance(it, (list, tuple)) and len(it) == 2):
                continue
            base_name = str(it[0])
            try:
                t_arrival = float(it[1])
            except Exception:
                t_arrival = float(si * dt)
            src_idx = _src_index_from_time(t_arrival)
            dst_idx = min(src_idx // max(1, frame_stride), n_frames - 1)
            q_k = qpos_tgt_seq[dst_idx]
            motor_pos, joint_pos = _compute_motor_and_joint_pos(
                tgt_model, tgt_data, q_k
            )
            unique_name = f"{base_name}__seq{si:03d}"
            out_keyframes.append(
                {
                    "name": unique_name,
                    "source_name": base_name,
                    "source_time": t_arrival,
                    "motor_pos": motor_pos.astype(np.float32),
                    "joint_pos": joint_pos.astype(np.float32),
                    "qpos": q_k.astype(np.float64),
                }
            )
            out_seq.append((unique_name, t_arrival))
    elif src_keyframes:
        for i, k in enumerate(src_keyframes):
            name = str(k.get("name", f"kf_{i:04d}"))
            src_idx = min(i, qpos_src_all.shape[0] - 1)
            dst_idx = min(src_idx // max(1, frame_stride), n_frames - 1)
            q_k = qpos_tgt_seq[dst_idx]
            motor_pos, joint_pos = _compute_motor_and_joint_pos(
                tgt_model, tgt_data, q_k
            )
            out_keyframes.append(
                {
                    "name": name,
                    "motor_pos": motor_pos.astype(np.float32),
                    "joint_pos": joint_pos.astype(np.float32),
                    "qpos": q_k.astype(np.float64),
                }
            )
        out_seq = [(k["name"], float(i * dt)) for i, k in enumerate(out_keyframes)]
    else:
        step = max(1, n_frames // 20)
        for i in range(0, n_frames, step):
            q_k = qpos_tgt_seq[i]
            motor_pos, joint_pos = _compute_motor_and_joint_pos(
                tgt_model, tgt_data, q_k
            )
            out_keyframes.append(
                {
                    "name": f"frame_{i:04d}",
                    "motor_pos": motor_pos.astype(np.float32),
                    "joint_pos": joint_pos.astype(np.float32),
                    "qpos": q_k.astype(np.float64),
                }
            )
        out_seq = [(k["name"], float(i * dt)) for i, k in enumerate(out_keyframes)]

    out_time = np.arange(n_frames, dtype=np.float64) * dt

    out_dict = {
        "time": out_time.astype(np.float64),
        "qpos": qpos_tgt_seq.astype(np.float32),
        "action": action_seq.astype(np.float32),
        "keyframes": out_keyframes,
        "timed_sequence": out_seq,
        "is_robot_relative_frame": bool(
            source_data.get("is_robot_relative_frame", True)
        ),
    }

    os.makedirs(os.path.dirname(output_motion), exist_ok=True)
    joblib.dump(out_dict, output_motion, compress="lz4")
    print(f"[Retarget] Saved retargeted motion to: {output_motion}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retarget motion with Mink IK")
    parser.add_argument("--source-xml", required=True)
    parser.add_argument("--target-xml", required=True)
    parser.add_argument("--source-motion", required=True)
    parser.add_argument("--output-motion", required=True)
    parser.add_argument("--source-root", default=None)
    parser.add_argument("--target-root", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--dt-default", type=float, default=1.0 / 30.0)
    parser.add_argument("--no-smooth", action="store_true")
    parser.add_argument("--foot-contact-z", type=float, default=0.04)
    parser.add_argument("--foot-contact-v", type=float, default=0.20)
    parser.add_argument("--hand-contact-z", type=float, default=0.10)
    parser.add_argument("--hand-contact-v", type=float, default=0.25)
    parser.add_argument("--ground-clearance", type=float, default=0.005)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retarget_motion(
        source_xml=args.source_xml,
        target_xml=args.target_xml,
        source_motion=args.source_motion,
        output_motion=args.output_motion,
        source_root=args.source_root,
        target_root=args.target_root,
        max_frames=args.max_frames,
        frame_stride=max(1, args.frame_stride),
        smooth=not args.no_smooth,
        dt_default=args.dt_default,
        foot_contact_z=args.foot_contact_z,
        foot_contact_v=args.foot_contact_v,
        hand_contact_z=args.hand_contact_z,
        hand_contact_v=args.hand_contact_v,
        ground_clearance=max(0.0, float(args.ground_clearance)),
    )


if __name__ == "__main__":
    main()
