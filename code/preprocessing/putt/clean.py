# -*- coding: utf-8 -*-
"""清洗：丢帧插值（四元数 SLERP、位置 PCHIP）→ 四元数半球连续化 → 零相位低通。"""
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R, Slerp

AX = "xyz"


def quat_cols(seg, kind="Bone"):
    return [f"{seg}-{kind}-Quat-{c}" for c in "xyzw"]


def pos_cols(seg):
    return [f"{seg}-Joint-Posi-{c}" for c in AX]


def detect_real_segments(df):
    lost = [c for c in df.columns if c.endswith("-Sensor-Lost")]
    return [c[: -len("-Sensor-Lost")] for c in lost if df[c].mean() < 0.5]


def lost_mask(df, segments):
    cols = [f"{s}-Sensor-Lost" for s in segments if f"{s}-Sensor-Lost" in df]
    return df[cols].values.astype(bool).any(axis=1)


def longest_run(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def make_continuous(q):
    """q: (N,4) xyzw。保证相邻帧点积 ≥ 0，消除 q / -q 跳变。"""
    q = q.copy()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q


def interp_quat(q, bad):
    if not bad.any():
        return q
    t = np.arange(len(q))
    good = ~bad
    if good.sum() < 2:
        return q
    sl = Slerp(t[good], R.from_quat(q[good]))
    tq = np.clip(t, t[good][0], t[good][-1])
    out = q.copy()
    out[bad] = sl(tq[bad]).as_quat()
    return out


def interp_vec(x, bad):
    if not bad.any():
        return x
    t = np.arange(len(x))
    good = ~bad
    out = x.copy()
    f = PchipInterpolator(t[good], x[good], axis=0, extrapolate=True)
    out[bad] = f(t[bad])
    return out


def lowpass(x, fs, fc, order=4):
    b, a = butter(order, fc / (fs / 2))
    padlen = min(3 * max(len(a), len(b)), len(x) - 1)
    return filtfilt(b, a, x, axis=0, padlen=padlen)


import os
MASK_MODE = os.environ.get("PUTT_MASK_MODE", "per")    # per (v1.1 default) | union (v1.0) | chain | none
CHAIN_SENSORS = ["Hips", "Spine2", "LeftArm", "RightArm", "LeftForeArm", "RightForeArm"]


def _deps(seg, kind):
    """Physical sensors that a segment's orientation (kind='q') or joint position (kind='p') depends on."""
    side = "Left" if seg.startswith("Left") else ("Right" if seg.startswith("Right") else "")
    if seg == "Hips":
        return ["Hips"]
    if seg in ("Spine", "Spine1"):
        return ["Hips", "Spine2"] if kind == "q" else ["Hips"]
    if seg in ("Spine2", "Spine3"):
        return ["Spine2"] if kind == "q" else ["Hips", "Spine2"]
    if seg.endswith("Shoulder"):
        return ["Spine2"] if kind == "q" else ["Hips", "Spine2"]
    if seg.endswith("ForeArm"):
        return [seg] if kind == "q" else ["Hips", "Spine2", f"{side}Arm"]
    if seg.endswith("Arm"):
        return [seg] if kind == "q" else ["Hips", "Spine2"]
    if seg.endswith("Hand"):
        return [f"{side}ForeArm"] if kind == "q" else ["Hips", "Spine2", f"{side}Arm", f"{side}ForeArm"]
    if seg.endswith(("UpLeg", "Leg", "Foot")):
        return [seg] if seg.endswith(("UpLeg", "Leg")) else [f"{side}Leg"]
    return CHAIN_SENSORS


def _mask_for(df, seg, kind, union_bad, chain_bad):
    if MASK_MODE == "union":
        return union_bad
    if MASK_MODE == "chain":
        return chain_bad
    if MASK_MODE == "none":
        return np.zeros(len(df), bool)
    return lost_mask(df, _deps(seg, kind))


def clean_take(df, fs, fc, quat_segments, pos_segments, mask_segments):
    """返回 dict: quats[seg] -> (N,4) 连续且滤波后的全局姿态；pos[seg] -> (N,3) 米。

    PUTT_MASK_MODE: per (default since v1.1) = each segment only over the frames in which the sensors it
    depends on were lost (see _deps); union = v1.0 behaviour (any of the 10 sensors lost -> interpolate every
    segment); chain = any of the 6 upper-body chain sensors; none = keep the manufacturer's estimate.
    The returned mask `bad` (any of the 6 chain sensors lost; union mode: any of the 10 sensors) is used only
    for the QC gap rules.
    """
    union_bad = lost_mask(df, mask_segments)
    chain_bad = lost_mask(df, CHAIN_SENSORS)
    bad = union_bad if MASK_MODE == "union" else chain_bad   # used for QC (gap rules)
    quats, pos = {}, {}
    for s in quat_segments:
        q = df[quat_cols(s)].values.astype(float)
        q = make_continuous(q)
        q = interp_quat(q, _mask_for(df, s, "q", union_bad, chain_bad))
        q = make_continuous(q)
        q = lowpass(q, fs, fc)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        quats[s] = q
    for s in pos_segments:
        p = df[pos_cols(s)].values.astype(float)
        p = interp_vec(p, _mask_for(df, s, "p", union_bad, chain_bad))
        pos[s] = lowpass(p, fs, fc)
    return quats, pos, bad
