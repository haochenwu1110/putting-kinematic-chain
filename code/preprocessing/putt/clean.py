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


def clean_take(df, fs, fc, quat_segments, pos_segments, mask_segments):
    """返回 dict: quats[seg] -> (N,4) 连续且滤波后的全局姿态；pos[seg] -> (N,3) 米。"""
    bad = lost_mask(df, mask_segments)
    quats, pos = {}, {}
    for s in quat_segments:
        q = df[quat_cols(s)].values.astype(float)
        q = make_continuous(q)
        q = interp_quat(q, bad)
        q = make_continuous(q)
        q = lowpass(q, fs, fc)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        quats[s] = q
    for s in pos_segments:
        p = df[pos_cols(s)].values.astype(float)
        p = interp_vec(p, bad)
        pos[s] = lowpass(p, fs, fc)
    return quats, pos, bad
