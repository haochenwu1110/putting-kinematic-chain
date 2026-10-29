# -*- coding: utf-8 -*-
"""运动学：朝向对齐坐标系 + 各关节角 / 握把轨迹。

坐标约定（已用数据验证）：Bone-Quat 为全局姿态，世界 Y 轴朝上；骨骼局部 +X 指向身体左侧
（左臂沿 +X 指向远端，右臂沿 -X 指向远端），脊柱沿 +Y。

朝向对齐坐标系 H（以准备姿态时的骨盆朝向定义）：
  X_H = 骨盆左向投影到水平面（右手球员 = 目标方向）
  Y_H = 竖直向上
  Z_H = X_H × Y_H = 身体正前方（朝球 / 远离身体）
所有带左右方向的量都乘 hand_sign（右手 +1，左手 -1），并以 lead/trail（前侧/后侧）命名，
左右手球员可以直接合并统计。
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

EX, EY, EZ = np.eye(3)
DISTAL = {"Left": EX, "Right": -EX}  # 前臂/上臂指向远端的局部轴


def avg_quat(q):
    """q 已连续化，取均值再归一化（小角度范围内足够精确）。"""
    m = q.mean(axis=0)
    return m / np.linalg.norm(m)


def heading_matrix(q_hips_ref):
    x = R.from_quat(q_hips_ref).apply(EX)
    x[1] = 0.0
    x /= np.linalg.norm(x)
    y = EY.copy()
    z = np.cross(x, y)
    return np.vstack([x, y, z])  # v_H = M @ v_world


def axis_in_H(q, local_axis, M):
    return (M @ R.from_quat(q).apply(local_axis).T).T


def twist_angle(q_rel, axis):
    """swing-twist 分解：q_rel 绕 axis 的扭转角（rad），已解卷绕。"""
    v = q_rel[:, :3] @ axis
    ang = 2.0 * np.arctan2(v, q_rel[:, 3])
    return np.unwrap(ang)


def _continuous(q):
    q = q.copy()
    if q[0, 3] < 0:
        q[0] = -q[0]
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q


def rel_to_ref(q, q_ref):
    """body-frame 相对旋转 q_ref^-1 * q(t)，已做半球连续化。"""
    return _continuous((R.from_quat(q_ref).inv() * R.from_quat(q)).as_quat())


def angle_between(a, b):
    d = np.clip(np.sum(a * b, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(d))


def compute_series(quats, pos, M, q_ref, hips_ref_pos, hand_sign, fs):
    """返回 dict[name] -> (N,) 时间序列。角度单位 deg，位移 mm，速度 m/s。"""
    hs = hand_sign
    lead, trail = ("Left", "Right") if hs > 0 else ("Right", "Left")
    S = {}

    # ---- 躯干（Spine2 = 胸廓上段）与骨盆 ----
    spine_up = axis_in_H(quats["Spine2"], EY, M)
    sh_line = axis_in_H(quats["Spine2"], EX, M)      # 指向左肩
    hip_line = axis_in_H(quats["Hips"], EX, M)
    S["trunk_fwd_tilt"] = np.degrees(np.arctan2(spine_up[:, 2], spine_up[:, 1]))
    S["shoulder_tilt_leadup"] = hs * np.degrees(np.arcsin(np.clip(sh_line[:, 1], -1, 1)))
    # 水平面内转动：正 = 上杆方向（闭合）；左手球员乘 -1 后同样成立
    S["shoulder_turn_closed"] = hs * np.degrees(np.unwrap(np.arctan2(sh_line[:, 2], sh_line[:, 0])))
    S["pelvis_tilt_leadup"] = hs * np.degrees(np.arcsin(np.clip(hip_line[:, 1], -1, 1)))
    S["pelvis_turn_closed"] = hs * np.degrees(np.unwrap(np.arctan2(hip_line[:, 2], hip_line[:, 0])))
    S["xfactor"] = S["shoulder_turn_closed"] - S["pelvis_turn_closed"]

    hp = (M @ (pos["Hips"] - hips_ref_pos).T).T * 1000.0
    S["pelvis_disp_target"] = hs * hp[:, 0]
    S["pelvis_disp_up"] = hp[:, 1]
    S["pelvis_disp_away"] = hp[:, 2]

    # ---- 手臂 ----
    for side, tag in ((lead, "lead"), (trail, "trail")):
        d = DISTAL[side]
        arm_ax = R.from_quat(quats[f"{side}Arm"]).apply(d)
        fa_ax = R.from_quat(quats[f"{side}ForeArm"]).apply(d)
        S[f"elbow_flex_{tag}"] = angle_between(arm_ax, fa_ax)
        # 前臂绝对滚转（相对准备姿态），正 = 杆面打开方向
        rel = rel_to_ref(quats[f"{side}ForeArm"], q_ref[f"{side}ForeArm"])
        S[f"forearm_roll_{tag}"] = hs * np.degrees(twist_angle(rel, d))
        # 前臂相对上臂的旋前/旋后（相对准备姿态的变化），正 = 杆面打开方向
        q_fa_in_arm = (R.from_quat(quats[f"{side}Arm"]).inv() * R.from_quat(quats[f"{side}ForeArm"])).as_quat()
        q_fa_in_arm_ref = (R.from_quat(q_ref[f"{side}Arm"]).inv() * R.from_quat(q_ref[f"{side}ForeArm"])).as_quat()
        rel2 = rel_to_ref(q_fa_in_arm, q_fa_in_arm_ref)
        S[f"pronsup_{tag}"] = hs * np.degrees(twist_angle(rel2, d))
        # 上臂相对躯干长轴的夹角（手臂下垂角）
        S[f"arm_trunk_angle_{tag}"] = angle_between(arm_ax, -R.from_quat(quats["Spine2"]).apply(EY))
        # 膝关节屈曲
        up_ax = R.from_quat(quats[f"{side}UpLeg"]).apply(EY)
        lo_ax = R.from_quat(quats[f"{side}Leg"]).apply(EY)
        S[f"knee_flex_{tag}"] = angle_between(up_ax, lo_ax)
    S["forearm_roll_mean"] = 0.5 * (S["forearm_roll_lead"] + S["forearm_roll_trail"])

    # ---- 握把点 = 双手腕关节中点（由前臂姿态正向运动学得到），相对准备时骨盆位置 ----
    grip_w = 0.5 * (pos["LeftHand"] + pos["RightHand"])
    g = (M @ (grip_w - hips_ref_pos).T).T
    g[:, 0] *= hs
    S["grip_target_mm"] = g[:, 0] * 1000.0
    S["grip_up_mm"] = g[:, 1] * 1000.0
    S["grip_away_mm"] = g[:, 2] * 1000.0
    v = np.gradient(g, axis=0) * fs
    S["grip_v_target"] = v[:, 0]
    S["grip_v_up"] = v[:, 1]
    S["grip_v_away"] = v[:, 2]
    S["grip_speed"] = np.linalg.norm(v, axis=1)
    S["grip_path_deg"] = np.degrees(np.arctan2(v[:, 2], v[:, 0]))          # 正 = 由内向外（I-O）
    S["grip_attack_deg"] = np.degrees(np.arctan2(v[:, 1], np.hypot(v[:, 0], v[:, 2])))  # 正 = 向上

    # 角速度（前侧前臂整体角速度大小，rad/s）
    q = R.from_quat(quats[f"{lead}ForeArm"])
    dq = (q[1:] * q[:-1].inv()).as_rotvec() * fs
    w = np.vstack([dq[:1], dq])
    S["lead_forearm_angvel"] = np.linalg.norm(w, axis=1)
    return S
