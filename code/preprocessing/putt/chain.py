# -*- coding: utf-8 -*-
"""运动链分解：握把的运动、前臂的滚转，各由哪个关节产生。

原理（刚体链的运动学恒等式，不是统计）：
  设链为 骨盆 → 各脊柱环节 → 肩带 → 上臂 → 前臂，环节 k 的近端关节位置为 p_k
  （即 Noitom 导出的 {环节}-Joint-Posi），全局角速度为 ω_k，末端为腕关节 p_E。则

    腕关节速度   v_E = v_骨盆 + Σ_k (ω_k − ω_父环节) × (p_E − p_k)
    前臂滚转速率 ω_前臂·d = Σ_k (ω_k − ω_父环节)·d        （d = 前臂长轴）

  每一项就是"关节 k 的相对转动"对末端运动的贡献。握把 = 双腕中点，所以握把速度 =
  两条手臂各自分解后取平均。实测速度与各项之和的差记为残差（来自滤波、数值微分、
  软件推算环节与实际的出入），用作质量检查。

功能分组：
  pelvis_trans  骨盆平移（重心移动、下肢）
  pelvis_rot    骨盆转动
  trunk         躯干：各脊柱环节相对骨盆的转动（含肩部转动）
  girdle        肩带：锁骨环节相对躯干（软件推算环节，通常很小）
  shoulder      肩关节：上臂相对肩带（摆臂）
  elbow         肘与前臂：前臂相对上臂（屈肘 + 旋前旋后）
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

from .clean import pos_cols, quat_cols
from .kin import DISTAL, EX, EY, EZ

# 骨盆 / 脊柱 / 锁骨环节的三条转动轴（环节局部系）
AXES_BODY = {"twist": EY, "flex": EX, "bend": EZ}

SPINE = ["Spine", "Spine1", "Spine2", "Spine3"]   # 按层级顺序；文件里没有的自动跳过
GROUP_OF = {"Hips": "pelvis_rot", "Spine": "trunk", "Spine1": "trunk", "Spine2": "trunk", "Spine3": "trunk"}
BODY_GROUPS = ["pelvis_trans", "pelvis_rot", "trunk", "girdle"]
ROLL_GROUPS = ["pelvis_rot", "trunk", "girdle", "shoulder", "elbow"]
# 握把分解的最终分组（左右臂的肩、肘分开，便于看前后侧手臂）
GRIP_GROUPS = BODY_GROUPS + ["shoulder_lead", "shoulder_trail", "elbow_lead", "elbow_trail", "resid"]
# 按转动轴拆分的键：身体组 × (twist, flex, bend)，手臂组 × (axial, swing)
GRIP_AXIS_KEYS = [f"{g}__{ax}" for g in ("pelvis_rot", "trunk", "girdle") for ax in ("twist", "flex", "bend")] + \
                 [f"{g}_{t}__{ax}" for g in ("shoulder", "elbow") for t in ("lead", "trail") for ax in ("axial", "swing")]
ROLL_AXIS_KEYS = [(g, ax) for g in ("pelvis_rot", "trunk", "girdle") for ax in ("twist", "flex", "bend")] + \
                 [(g, ax) for g in ("shoulder", "elbow") for ax in ("axial", "swing")]
AXIS_LABEL = {"twist": "扭转（绕长轴）", "flex": "屈伸（绕左右轴）", "bend": "侧屈（绕前后轴）",
              "axial": "绕长轴旋转", "swing": "摆动"}
GROUP_LABEL = {"pelvis_trans": "骨盆平移", "pelvis_rot": "骨盆转动", "trunk": "躯干", "girdle": "肩带",
               "shoulder_lead": "前侧肩关节", "shoulder_trail": "后侧肩关节",
               "elbow_lead": "前侧肘/前臂", "elbow_trail": "后侧肘/前臂", "resid": "残差",
               "shoulder": "肩关节", "elbow": "肘/前臂"}


def _group(seg):
    if seg in GROUP_OF:
        return GROUP_OF[seg]
    if seg.endswith("Shoulder"):
        return "girdle"
    if seg.endswith("ForeArm"):
        return "elbow"
    if seg.endswith("Arm"):
        return "shoulder"
    raise KeyError(seg)


def _has(df, seg):
    return all(c in df.columns for c in quat_cols(seg) + pos_cols(seg))


def chain_segments(df, side):
    """该侧手臂的运动链（按层级），只保留 CSV 里同时有姿态和位置的环节。"""
    cand = ["Hips"] + SPINE + [f"{side}Shoulder", f"{side}Arm", f"{side}ForeArm"]
    segs = [s for s in cand if _has(df, s)]
    need = ["Hips", f"{side}Arm", f"{side}ForeArm"]
    missing = [s for s in need if s not in segs]
    if missing or not all(c in df.columns for c in pos_cols(f"{side}Hand")):
        raise ValueError(f"运动链缺少必要环节: {missing or [side + 'Hand']}")
    return segs


def angvel(q, fs):
    """全局角速度（世界系，rad/s），中心差分。q: (N,4) 已连续化、滤波。"""
    r = R.from_quat(q)
    n = len(q)
    w = np.empty((n, 3))
    w[1:-1] = (r[2:] * r[:-2].inv()).as_rotvec() * fs / 2.0
    w[0] = (r[1] * r[0].inv()).as_rotvec() * fs
    w[-1] = (r[-1] * r[-2].inv()).as_rotvec() * fs
    return w


def _vel(p, fs):
    return np.gradient(p, axis=0) * fs


def _axis_components(dw, quat_seg, seg):
    """把关节相对角速度 dw (N,3) 投影到转动环节自身的正交轴上，各分量之和 = dw。

    骨盆、脊柱、锁骨环节（局部 +Y 沿长轴，+X 指向身体左侧，Z 前后）：
      twist = 绕长轴扭转，flex = 绕左右轴屈伸（前后弯），bend = 绕前后轴侧屈（左右摇）
    上臂、前臂（局部 ±X 指向远端）：
      axial = 绕长轴旋转（上臂内外旋 / 前臂旋前旋后），swing = 其余摆动
    """
    r = R.from_quat(quat_seg)
    if _group(seg) in ("shoulder", "elbow"):
        side = "Left" if seg.startswith("Left") else "Right"
        a = r.apply(DISTAL[side])
        axial = np.sum(dw * a, axis=1)[:, None] * a
        return {"axial": axial, "swing": dw - axial}
    out = {}
    for name, loc in AXES_BODY.items():
        a = r.apply(loc)
        out[name] = np.sum(dw * a, axis=1)[:, None] * a
    return out


def decompose_arm(quats, pos, segs, end_seg, distal_local, fs):
    """单侧手臂：返回
      v[group]          -> (N,3) 腕速度各项（世界系）
      vE                -> (N,3) 实测腕速度
      roll[group]       -> (N,)  前臂滚转速率各项（rad/s）
      vax[(group, ax)]  -> (N,3) 按转动轴拆分的腕速度各项（同组各轴之和 = v[group]）
      rollax[(group, ax)] -> (N,) 按转动轴拆分的前臂滚转速率各项
    """
    pE = pos[end_seg]
    vE = _vel(pE, fs)
    W = {s: angvel(quats[s], fs) for s in segs}
    d = R.from_quat(quats[segs[-1]]).apply(distal_local)   # 前臂长轴（世界系）
    v = {"pelvis_trans": _vel(pos["Hips"], fs)}
    roll, vax, rollax = {}, {}, {}
    parent_w = np.zeros_like(W[segs[0]])
    for s in segs:
        g = _group(s)
        dw = W[s] - parent_w
        rE = pE - pos[s]
        v[g] = v.get(g, 0.0) + np.cross(dw, rE)
        roll[g] = roll.get(g, 0.0) + np.sum(dw * d, axis=1)
        for ax, w_ax in _axis_components(dw, quats[s], s).items():
            vax[(g, ax)] = vax.get((g, ax), 0.0) + np.cross(w_ax, rE)
            rollax[(g, ax)] = rollax.get((g, ax), 0.0) + np.sum(w_ax * d, axis=1)
        parent_w = W[s]
    return v, vE, roll, vax, rollax


def link_length_sd(pos, segs, end_seg, lo, hi):
    """各相邻关节间距在挥杆窗口内的标准差（mm），检查运动链是否刚性。"""
    pts = segs + [end_seg]
    sd = [np.std(np.linalg.norm(pos[b][lo:hi] - pos[a][lo:hi], axis=1)) * 1000 for a, b in zip(pts[:-1], pts[1:])]
    return float(max(sd)) if sd else np.nan


def _cumint(y, fs):
    """累积梯形积分，c[0] = 0。"""
    c = np.zeros(len(y))
    c[1:] = np.cumsum(0.5 * (y[1:] + y[:-1])) / fs
    return c


def _at(c, t):
    return float(np.interp(t, np.arange(len(c)), c))


def _resample(x, t0, t1, n):
    return np.interp(np.linspace(t0, t1, n), np.arange(len(x)), x)


def grip_and_roll(quats, pos, segs_by_side, M, hand_sign, fs):
    """两臂分解 → 握把（双腕中点）在朝向坐标系 H 中的速度各项 + 前后侧前臂滚转各项。

    返回
      gv[group]   -> (N,3)  握把速度贡献（H 系，x 已乘 hand_sign；m/s）
      gmeas       -> (N,3)  实测握把速度（H 系）
      roll[tag][group] -> (N,) 前臂滚转速率贡献（deg/s，正 = 杆面打开方向）
    """
    hs = hand_sign
    lead, trail = ("Left", "Right") if hs > 0 else ("Right", "Left")
    tag_of = {lead: "lead", trail: "trail"}
    n = len(pos["Hips"])
    gv = {g: np.zeros((n, 3)) for g in GRIP_GROUPS if g != "resid"}
    gax = {k: np.zeros((n, 3)) for k in GRIP_AXIS_KEYS}
    gmeas = np.zeros((n, 3))
    roll, rollax = {}, {}
    for side in ("Left", "Right"):
        v, vE, rl, vax, rlax = decompose_arm(quats, pos, segs_by_side[side], f"{side}Hand", DISTAL[side], fs)
        tag = tag_of[side]
        for g, val in v.items():
            key = f"{g}_{tag}" if g in ("shoulder", "elbow") else g
            gv[key] = gv[key] + 0.5 * val
        for (g, ax), val in vax.items():
            key = f"{g}_{tag}__{ax}" if g in ("shoulder", "elbow") else f"{g}__{ax}"
            gax[key] = gax[key] + 0.5 * val
        gmeas = gmeas + 0.5 * vE
        roll[tag] = {g: hs * np.degrees(rl.get(g, np.zeros(len(vE)))) for g in ROLL_GROUPS}
        rollax[tag] = {f"{g}__{ax}": hs * np.degrees(rlax.get((g, ax), np.zeros(len(vE))))
                       for g, ax in ROLL_AXIS_KEYS}

    def to_H(v):
        out = (M @ np.asarray(v).T).T
        out[:, 0] *= hs
        return out

    gv = {g: to_H(val) for g, val in gv.items()}
    gax = {k: to_H(val) for k, val in gax.items()}
    gmeas = to_H(gmeas)
    gv["resid"] = gmeas - sum(gv[g] for g in GRIP_GROUPS if g != "resid")
    return gv, gmeas, roll, gax, rollax


def chain_outputs(quats, pos, segs_by_side, M, hand_sign, fs, ev, n_norm):
    """单杆的运动链分解结果：逐杆特征 dict + 时间归一化曲线 dict[var] -> (3*n_norm,)。

    特征命名
      bs_{g}_mm     对上杆长度的贡献（准备→顶点，握把背离目标方向的位移，mm；各项之和 = bs_total_mm）
      ds_{g}_mm     对下杆位移的贡献（顶点→击球，握把朝目标方向的位移，mm；之和 = ds_total_mm）
      iv_{g}        击球时握把朝目标方向速度的贡献（m/s；之和 = iv_total）
      ivz_{g}       击球时握把离身方向速度的贡献（m/s，正 = 远离身体，对应轨迹由内向外）
      ivy_{g}       击球时握把竖直速度的贡献（m/s，正 = 向上，对应击球角）
      roll_{side}_{g}_top_deg / _imp_deg
                    准备姿态 → 顶点 / 击球，前臂累积扭转的贡献（deg，正 = 开杆面；之和 = roll_{side}_total_*）
      bsx_ / dsx_ / ivx_ / ivzx_{g}__{ax}
                    同上，按转动轴拆分：身体组 ax ∈ twist / flex / bend，手臂组 ax ∈ axial / swing；
                    同一 g 的各 ax 之和 = 原来的 bs_ / ds_ / iv_ / ivz_{g}
      rollx_{side}_{g}__{ax}_top_deg / _imp_deg
                    前臂累积扭转贡献按转动轴拆分，之和 = roll_{side}_{g}_*
    """
    gv, gmeas, roll, gax, rollax = grip_and_roll(quats, pos, segs_by_side, M, hand_sign, fs)
    tk, tt, ti, tf = ev["takeaway"], ev["top"], ev["impact"], ev["finish"]
    a_mid = 0.5 * (ev["address_start"] + ev["address_end"])
    F = {}

    # 握把沿目标轴
    groups = GRIP_GROUPS
    cum = {g: _cumint(gv[g][:, 0], fs) * 1000.0 for g in groups}       # mm
    cum_meas = _cumint(gmeas[:, 0], fs) * 1000.0
    # 上杆从准备窗口中点算起，与 backswing_len_mm（准备位置 − 顶点位置）的定义一致
    F["bs_total_mm"] = -(_at(cum_meas, tt) - _at(cum_meas, a_mid))
    F["ds_total_mm"] = _at(cum_meas, ti) - _at(cum_meas, tt)
    F["iv_total"] = _at(gmeas[:, 0], ti)
    F["ivz_total"] = _at(gmeas[:, 2], ti)
    F["ivy_total"] = _at(gmeas[:, 1], ti)
    for g in groups:
        F[f"bs_{g}_mm"] = -(_at(cum[g], tt) - _at(cum[g], a_mid))
        F[f"ds_{g}_mm"] = _at(cum[g], ti) - _at(cum[g], tt)
        F[f"iv_{g}"] = _at(gv[g][:, 0], ti)
        F[f"ivz_{g}"] = _at(gv[g][:, 2], ti)
        F[f"ivy_{g}"] = _at(gv[g][:, 1], ti)

    # 前臂累积扭转（从准备姿态窗口中点起算，与 top_d_ / imp_d_ 特征的基线一致）
    croll = {}
    for tag in ("lead", "trail"):
        tot_top = tot_imp = 0.0
        for g in ROLL_GROUPS:
            c = _cumint(roll[tag][g], fs)
            croll[(tag, g)] = c - _at(c, a_mid)
            F[f"roll_{tag}_{g}_top_deg"] = _at(croll[(tag, g)], tt)
            F[f"roll_{tag}_{g}_imp_deg"] = _at(croll[(tag, g)], ti)
            tot_top += F[f"roll_{tag}_{g}_top_deg"]
            tot_imp += F[f"roll_{tag}_{g}_imp_deg"]
        F[f"roll_{tag}_total_top_deg"] = tot_top
        F[f"roll_{tag}_total_imp_deg"] = tot_imp

    # 按转动轴拆分（同一关节各轴之和 = 该关节的原贡献）
    cumx = {}
    for k in GRIP_AXIS_KEYS:
        cumx[k] = _cumint(gax[k][:, 0], fs) * 1000.0
        F[f"bsx_{k}_mm"] = -(_at(cumx[k], tt) - _at(cumx[k], a_mid))
        F[f"dsx_{k}_mm"] = _at(cumx[k], ti) - _at(cumx[k], tt)
        F[f"ivx_{k}"] = _at(gax[k][:, 0], ti)
        F[f"ivzx_{k}"] = _at(gax[k][:, 2], ti)
    for tag in ("lead", "trail"):
        for k, rate in rollax[tag].items():
            c = _cumint(rate, fs)
            c = c - _at(c, a_mid)
            F[f"rollx_{tag}_{k}_top_deg"] = _at(c, tt)
            F[f"rollx_{tag}_{k}_imp_deg"] = _at(c, ti)

    # 质量检查：挥杆窗口内残差的 RMS / 实测握把速度的 RMS
    w = slice(tk, tf + 1)
    rms = lambda a: float(np.sqrt(np.mean(np.sum(a[w] ** 2, axis=1))))
    F["chain_resid_pct"] = 100.0 * rms(gv["resid"]) / max(rms(gmeas), 1e-9)
    F["chain_resid_x_pct"] = 100.0 * float(np.sqrt(np.mean(gv["resid"][w, 0] ** 2)) /
                                           max(np.sqrt(np.mean(gmeas[w, 0] ** 2)), 1e-9))

    # 时间归一化曲线（与 curves_wide 相同的三段式：上杆/下杆/送杆各 n_norm 点）
    def norm3(y):
        return np.concatenate([_resample(y, tk, tt, n_norm), _resample(y, tt, ti, n_norm),
                               _resample(y, ti, tf, n_norm)])
    C = {"cv_measured": norm3(gmeas[:, 0])}
    for g in groups:
        C[f"cv_{g}"] = norm3(gv[g][:, 0])                       # 速度贡献，m/s
        C[f"cd_{g}"] = norm3(cum[g] - _at(cum[g], tk))          # 起杆后的累积位移贡献，mm
    C["cd_measured"] = norm3(cum_meas - _at(cum_meas, tk))
    for tag in ("lead", "trail"):
        for g in ROLL_GROUPS:
            C[f"cr_{tag}_{g}"] = norm3(croll[(tag, g)])         # 累积扭转贡献，deg
    for k in GRIP_AXIS_KEYS:
        if not k.startswith("girdle"):
            C[f"cvx_{k}"] = norm3(gax[k][:, 0])                 # 按轴拆分的速度贡献，m/s
    return F, C
