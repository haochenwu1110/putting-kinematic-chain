# -*- coding: utf-8 -*-
"""事件检测：起杆 → 上杆顶点 → 击球 → 送杆结束。

基于握把点（双手腕中点）沿目标方向的位移 x 和速度 vx：
  下杆峰值   = vx（朝目标）最大值所在帧（主挥杆）
  上杆顶点   = 下杆峰值之前 x 最小（vx 由负转正）的帧
  起杆       = 顶点往前倒查，握把速度首次低于阈值的帧
  击球       = 顶点之后 x 回到准备姿态位置（球在准备时杆头位置）的时刻，亚帧线性插值
               备用估计 = 下杆峰值帧；两者相差过大写入 QC
  送杆结束   = 击球之后 vx 首次降到阈值以下
"""
import numpy as np
from scipy.signal import find_peaks


def detect(x_mm, vx, speed, fs, cfg, impact_hint=None):
    n = len(x_mm)
    flags = []
    # ---- 主挥杆（下杆峰值）----
    if impact_hint is not None and not np.isnan(impact_hint):
        lo, hi = int(max(0, impact_hint - 0.4 * fs)), int(min(n, impact_hint + 0.4 * fs))
        i_pk = lo + int(np.argmax(vx[lo:hi]))
        flags.append("manual_hint")
    else:
        i_pk = int(np.argmax(vx))
    vmax = vx[i_pk]
    pks, _ = find_peaks(vx, height=cfg.MULTI_STROKE_FRAC * vmax, distance=int(1.0 * fs))
    n_strokes = len(pks)
    if n_strokes > 1:
        flags.append(f"multi_stroke({n_strokes})")

    thr = cfg.TAKEAWAY_SPEED_FRAC * speed[i_pk]

    # ---- 上杆顶点：峰值前 vx 最后一次 ≤ 0 ----
    neg = np.where(vx[:i_pk] <= 0)[0]
    if len(neg) == 0:
        # 文件开头就在下杆：退而取峰值前 x 最小处
        i_top = int(np.argmin(x_mm[:i_pk])) if i_pk > 3 else 0
        if i_top < 3:
            return None, flags + ["no_top"]
        flags.append("top_fallback")
    else:
        i_top = int(neg[-1])
    # 顶点取该段内 x 的最小值，更稳
    seg_lo = max(0, i_top - int(0.15 * fs))
    i_top = seg_lo + int(np.argmin(x_mm[seg_lo:i_top + 2]))

    # ---- 起杆：从顶点往前走，沿 x 单调回升到上杆前的局部最高点；
    #      再在该点之后找 x 首次下降超过 max(2 mm, 上杆幅度 3%) 的帧 ----
    i = i_top
    lo = max(0, i_top - int(3.0 * fs))
    while i > lo and x_mm[i - 1] >= x_mm[i] - 0.3:
        i -= 1
    i_max = i
    amp = x_mm[i_max] - x_mm[i_top]
    if amp < 5:
        return None, flags + ["no_backswing"]
    drop = np.where(x_mm[i_max] - x_mm[i_max:i_top + 1] > max(2.0, 0.03 * amp))[0]
    i_take = i_max + int(drop[0]) - 1 if len(drop) else i_max
    i_take = max(i_take, 0)
    if i_take < int(cfg.ADDRESS_WIN_S[0] * fs):
        flags.append("takeaway_at_file_start")

    # ---- 准备姿态窗口 ----
    a0 = i_take - int(cfg.ADDRESS_WIN_S[0] * fs)
    a1 = i_take - int(cfg.ADDRESS_WIN_S[1] * fs)
    if a1 - max(a0, 0) < 3:
        a0, a1 = 0, max(i_take, 3)
        flags.append("short_address_window")
    a0 = max(a0, 0)
    x_addr = float(np.mean(x_mm[a0:a1]))

    # ---- 送杆结束 ----
    after = np.where(vx[i_pk:] < thr)[0]
    if len(after) == 0:
        i_fin = n - 1
        flags.append("finish_at_file_end")
    else:
        i_fin = i_pk + int(after[0])

    # ---- 击球：x 回到准备位置 ----
    t_imp = np.nan
    for i in range(i_top, i_fin):
        if x_mm[i] < x_addr <= x_mm[i + 1]:
            t_imp = i + (x_addr - x_mm[i]) / (x_mm[i + 1] - x_mm[i])
            break
    if np.isnan(t_imp):
        t_imp = float(i_pk)
        flags.append("impact_fallback_peak")
    elif abs(t_imp - i_pk) > cfg.IMPACT_FALLBACK_TOL:
        flags.append(f"impact_vs_peak_{t_imp - i_pk:+.1f}f")

    ev = dict(address_start=a0, address_end=a1, takeaway=i_take, top=i_top,
              impact=float(t_imp), peak_v=i_pk, finish=i_fin, n_strokes=n_strokes,
              x_address_mm=x_addr)
    return ev, flags
