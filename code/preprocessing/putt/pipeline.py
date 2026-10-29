# -*- coding: utf-8 -*-
"""单个 take 的完整处理：清洗 → 两轮事件检测 → 关节角序列 → 特征 / 归一化曲线 → 运动链分解。"""
import numpy as np
from scipy.spatial.transform import Rotation as R

from . import chain, clean, events, kin
from .io_utils import read_take

POS_SEGMENTS = ["Hips", "LeftHand", "RightHand"]

# 做时间归一化曲线的变量（全部为相对准备姿态的变化量 Δ）
CURVE_VARS = [
    "grip_target_mm", "grip_away_mm", "grip_up_mm", "grip_speed",
    "shoulder_tilt_leadup", "shoulder_turn_closed", "trunk_fwd_tilt",
    "pelvis_turn_closed", "pelvis_tilt_leadup", "xfactor",
    "pelvis_disp_target", "pelvis_disp_up", "pelvis_disp_away",
    "elbow_flex_lead", "elbow_flex_trail",
    "forearm_roll_lead", "forearm_roll_trail", "forearm_roll_mean",
    "pronsup_lead", "pronsup_trail",
    "knee_flex_lead", "knee_flex_trail",
]
# 准备姿态绝对值（受标定影响，组间比较时注意）
ADDRESS_VARS = ["trunk_fwd_tilt", "shoulder_tilt_leadup", "pelvis_tilt_leadup",
                "elbow_flex_lead", "elbow_flex_trail", "arm_trunk_angle_lead", "arm_trunk_angle_trail",
                "knee_flex_lead", "knee_flex_trail", "grip_away_mm", "grip_up_mm"]
# 顶点 / 击球时刻取 Δ
EVENT_VARS = ["shoulder_tilt_leadup", "shoulder_turn_closed", "trunk_fwd_tilt", "pelvis_turn_closed",
              "xfactor", "elbow_flex_lead", "elbow_flex_trail",
              "forearm_roll_lead", "forearm_roll_trail", "forearm_roll_mean",
              "pronsup_lead", "pronsup_trail", "grip_away_mm", "grip_up_mm"]
# 起杆→送杆结束 的运动范围（稳定性指标）
RANGE_VARS = ["pelvis_turn_closed", "pelvis_tilt_leadup", "pelvis_disp_target", "pelvis_disp_up",
              "pelvis_disp_away", "knee_flex_lead", "knee_flex_trail", "trunk_fwd_tilt",
              "forearm_roll_lead", "forearm_roll_trail", "pronsup_lead", "pronsup_trail"]


def _grip(pos, M, hips_ref_pos, hs, fs):
    g = 0.5 * (pos["LeftHand"] + pos["RightHand"])
    g = (M @ (g - hips_ref_pos).T).T
    g[:, 0] *= hs
    v = np.gradient(g, axis=0) * fs
    return g[:, 0] * 1000.0, v[:, 0], np.linalg.norm(v, axis=1)


def _at(x, t):
    """在小数帧 t 处线性插值。"""
    return float(np.interp(t, np.arange(len(x)), x))


def _resample(x, t0, t1, n):
    return np.interp(np.linspace(t0, t1, n), np.arange(len(x)), x)


def process_take(path, cfg, hand_sign=1, impact_hint=None):
    out = dict(flags=[])
    df, n_decl = read_take(path)
    if n_decl is not None and n_decl != len(df):
        out["flags"].append(f"frames_{len(df)}_vs_header_{n_decl}")
    real = clean.detect_real_segments(df)
    diff = set(real) ^ set(cfg.REAL_SEGMENTS)
    if diff:
        key = diff & {"Hips", "Spine2", "LeftArm", "RightArm", "LeftForeArm", "RightForeArm"}
        tag = "key_sensor_lost:" if key else "leg_sensor_lost:"
        out["flags"].append(tag + "|".join(sorted(diff)))

    # 运动链需要的环节（含软件推算的脊柱、锁骨环节）
    segs_by_side, chain_err = {}, None
    if getattr(cfg, "CHAIN_DECOMPOSE", True):
        try:
            segs_by_side = {s: chain.chain_segments(df, s) for s in ("Left", "Right")}
        except ValueError as e:
            chain_err = str(e)
            out["flags"].append("chain_unavailable")
    chain_extra = sorted({s for segs in segs_by_side.values() for s in segs})
    quat_segs = list(dict.fromkeys(cfg.REAL_SEGMENTS + chain_extra))
    pos_segs = list(dict.fromkeys(POS_SEGMENTS + chain_extra))

    quats, pos, bad = clean.clean_take(df, cfg.FS, cfg.LOWPASS_HZ, quat_segs, pos_segs, cfg.REAL_SEGMENTS)
    fs = cfg.FS

    # ---------- 第一轮：粗朝向 ----------
    q = R.from_quat(quats["LeftForeArm"])
    w = np.linalg.norm((q[1:] * q[:-1].inv()).as_rotvec(), axis=1)
    p0 = int(np.argmax(w)) if impact_hint is None or np.isnan(impact_hint) else int(impact_hint)
    M0 = kin.heading_matrix(quats["Hips"][p0])
    x, vx, sp = _grip(pos, M0, pos["Hips"][p0], hand_sign, fs)
    ev, fl = events.detect(x, vx, sp, fs, cfg, impact_hint)
    if ev is None:
        out["flags"] += fl
        return out

    # ---------- 第二轮：以准备姿态窗口定义朝向与参考 ----------
    a0, a1 = ev["address_start"], ev["address_end"]
    q_ref = {s: kin.avg_quat(quats[s][a0:a1]) for s in cfg.REAL_SEGMENTS}
    M = kin.heading_matrix(q_ref["Hips"])
    hips_ref = pos["Hips"][a0:a1].mean(axis=0)
    x, vx, sp = _grip(pos, M, hips_ref, hand_sign, fs)
    ev, fl = events.detect(x, vx, sp, fs, cfg, impact_hint)
    if ev is None:
        out["flags"] += fl
        return out
    out["flags"] += fl
    a0, a1 = ev["address_start"], ev["address_end"]

    # 朝向合理性：准备时握把应在骨盆前方（Z_H > 0）
    S = kin.compute_series(quats, pos, M, q_ref, hips_ref, hand_sign, fs)
    if np.mean(S["grip_away_mm"][a0:a1]) <= 0:
        out["flags"].append("grip_not_in_front_check_axes")

    # 丢帧 QC（挥杆窗口内）
    win = slice(max(0, ev["takeaway"] - 10), min(len(bad), ev["finish"] + 10))
    ev["lost_frames_total"] = int(bad.sum())
    ev["lost_frames_in_stroke"] = int(bad[win].sum())
    ev["longest_gap_in_stroke"] = clean.longest_run(bad[win])
    crit = slice(max(0, ev["top"] - 6), min(len(bad), int(ev["impact"]) + 7))
    ev["longest_gap_near_impact"] = clean.longest_run(bad[crit])
    if ev["longest_gap_near_impact"] > cfg.MAX_GAP_FRAMES:
        out["flags"].append(f"gap_{ev['longest_gap_near_impact']}f_near_impact")
    elif ev["longest_gap_in_stroke"] > cfg.MAX_GAP_FRAMES:
        out["flags"].append(f"minor_gap_{ev['longest_gap_in_stroke']}f")

    ti, tt, tk, tf = ev["impact"], ev["top"], ev["takeaway"], ev["finish"]
    addr = {k: float(np.mean(v[a0:a1])) for k, v in S.items()}

    # ---------- 特征 ----------
    F = {}
    F["t_backswing_s"] = (tt - tk) / fs
    F["t_downswing_s"] = (ti - tt) / fs
    F["t_follow_s"] = (tf - ti) / fs
    F["tempo_ratio"] = F["t_backswing_s"] / F["t_downswing_s"] if F["t_downswing_s"] > 0 else np.nan
    F["backswing_len_mm"] = ev["x_address_mm"] - S["grip_target_mm"][tt]
    F["follow_len_mm"] = S["grip_target_mm"][tf] - ev["x_address_mm"]
    F["follow_back_ratio"] = F["follow_len_mm"] / F["backswing_len_mm"] if F["backswing_len_mm"] > 0 else np.nan
    F["grip_speed_peak"] = float(S["grip_speed"][ev["peak_v"]])
    F["grip_speed_impact"] = _at(S["grip_speed"], ti)
    F["grip_path_impact_deg"] = _at(S["grip_path_deg"], ti)
    F["grip_attack_impact_deg"] = _at(S["grip_attack_deg"], ti)
    F["t_peak_minus_impact_s"] = (ev["peak_v"] - ti) / fs
    for k in ADDRESS_VARS:
        F[f"addr_{k}"] = addr[k]
    for k in EVENT_VARS:
        F[f"top_d_{k}"] = S[k][tt] - addr[k]
        F[f"imp_d_{k}"] = _at(S[k], ti) - addr[k]
    for k in ("forearm_roll_lead", "forearm_roll_trail", "forearm_roll_mean", "shoulder_tilt_leadup"):
        rate = np.gradient(S[k]) * fs
        F[f"imp_rate_{k}"] = _at(rate, ti)   # deg/s
    st = slice(tk, tf + 1)
    for k in RANGE_VARS:
        F[f"rng_{k}"] = float(np.ptp(S[k][st]))

    # ---------- 时间归一化曲线（Δ 相对准备姿态）----------
    n = cfg.N_NORM
    curves = {}
    for k in CURVE_VARS:
        base = 0.0 if k == "grip_speed" else addr[k]
        if k == "grip_target_mm":
            base = ev["x_address_mm"]
        y = S[k] - base
        curves[k] = np.concatenate([_resample(y, tk, tt, n), _resample(y, tt, ti, n), _resample(y, ti, tf, n)])

    out.update(events=ev, features=F, curves=curves, series=S, address=addr)

    # ---------- 运动链分解 ----------
    if segs_by_side:
        try:
            CF, CC = chain.chain_outputs(quats, pos, segs_by_side, M, hand_sign, fs, ev, n)
            lo, hi = tk, tf + 1
            CF["chain_link_sd_mm"] = max(chain.link_length_sd(pos, segs_by_side[s], f"{s}Hand", lo, hi)
                                         for s in ("Left", "Right"))
            CF["chain_segments"] = "|".join(segs_by_side["Left"])
            if CF["chain_resid_x_pct"] > cfg.CHAIN_RESID_WARN_PCT:
                out["flags"].append(f"chain_resid_{CF['chain_resid_x_pct']:.0f}pct")
            out.update(chain_features=CF, chain_curves=CC)
        except Exception as e:  # noqa: BLE001
            out["flags"].append(f"chain_error:{type(e).__name__}")
            out["chain_error"] = str(e)
    elif chain_err:
        out["chain_error"] = chain_err

    out["raw"] = dict(quats=quats, pos=pos, M=M, hips_ref=hips_ref, q_ref=q_ref,
                      segs_by_side=segs_by_side, hand_sign=hand_sign, fs=fs)
    return out
