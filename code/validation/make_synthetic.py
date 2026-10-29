# -*- coding: utf-8 -*-
"""合成推杆数据（Noitom Axis Studio 导出格式）+ 杆面表，用于验证运动链分解。

已知真值：每杆各关节的转角按同一推杆形状 s(t) 缩放：
  骨盆转动 kp*s，骨盆平移 kx*s，躯干转动 kt*s（绕胸廓长轴），双臂摆动 ks*s（绕胸廓前后轴），前臂扭转 kr*s。
运动学由正向运动学生成，位置与姿态完全一致，因此分解残差应接近 0。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

FS = 60.0
import os
TRUNK_AXIS = [float(v) for v in os.environ.get("TRUNK_AXIS", "0,1,0").split(",")]
RNG = np.random.default_rng(7)

OFFS = {  # child: (parent, local offset in parent frame, m)
    "Spine": ("Hips", (0, 0.10, 0)), "Spine1": ("Spine", (0, 0.12, 0)), "Spine2": ("Spine1", (0, 0.12, 0)),
    "LeftShoulder": ("Spine2", (0.03, 0.22, 0)), "LeftArm": ("LeftShoulder", (0.15, 0, 0)),
    "LeftForeArm": ("LeftArm", (0.28, 0, 0)), "LeftHand": ("LeftForeArm", (0.25, 0, 0)),
    "RightShoulder": ("Spine2", (-0.03, 0.22, 0)), "RightArm": ("RightShoulder", (-0.15, 0, 0)),
    "RightForeArm": ("RightArm", (-0.28, 0, 0)), "RightHand": ("RightForeArm", (-0.25, 0, 0)),
    "LeftUpLeg": ("Hips", (0.09, 0, 0)), "LeftLeg": ("LeftUpLeg", (0, -0.42, 0)),
    "RightUpLeg": ("Hips", (-0.09, 0, 0)), "RightLeg": ("RightUpLeg", (0, -0.42, 0)),
}
ORDER = ["Hips", "Spine", "Spine1", "Spine2", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
         "RightShoulder", "RightArm", "RightForeArm", "RightHand", "LeftUpLeg", "LeftLeg", "RightUpLeg", "RightLeg"]
REAL = {"Hips", "RightUpLeg", "RightLeg", "LeftUpLeg", "LeftLeg", "Spine2", "RightArm", "RightForeArm",
        "LeftArm", "LeftForeArm"}


def rot(axis, deg):
    return R.from_rotvec(np.outer(np.radians(deg), axis))


def min_rot(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b); c = np.dot(a, b)
    if np.linalg.norm(v) < 1e-9:
        return R.identity()
    ang = np.arctan2(np.linalg.norm(v), c)
    return R.from_rotvec(v / np.linalg.norm(v) * ang)


def stroke(n, t_ta, Tb, D, Ab, Af):
    t = np.arange(n) / FS
    s = np.zeros(n)
    i1 = (t >= t_ta) & (t < t_ta + Tb)
    s[i1] = -Ab * (1 - np.cos(np.pi * (t[i1] - t_ta) / Tb)) / 2
    i2 = (t >= t_ta + Tb) & (t < t_ta + Tb + D)
    s[i2] = -Ab + (Ab + Af) * (1 - np.cos(np.pi * (t[i2] - t_ta - Tb) / D)) / 2
    s[t >= t_ta + Tb + D] = Af
    return s


def make_take(style, Ab, roll_noise, n=270):
    kp, kx, kt, ks, kr = style
    Tb, D, Af = 0.75 + RNG.normal(0, 0.05), 0.85, 1.6 * Ab
    s = stroke(n, 1.2, Tb, D, Ab, Af)
    rn = roll_noise * np.clip((np.arange(n) / FS - 1.2) / (Tb + D), 0, 1)   # 本杆额外扭转
    tilt = 40.0
    Rg = {}
    Rg["Hips"] = rot([0, 1, 0], kp * s)
    par = "Hips"
    for nm in ["Spine", "Spine1", "Spine2"]:   # 每节相对父环节：固定前倾 + 绕自身轴的转动（相对角速度严格沿该轴）
        Rg[nm] = Rg[par] * rot([1, 0, 0], np.full(n, tilt / 3)) * rot(TRUNK_AXIS, kt * s / 3)
        par = nm
    for side in ("Left", "Right"):
        Rg[f"{side}Shoulder"] = Rg["Spine2"]
    # 静止姿态下的位置，用于确定手臂下垂方向
    pos = {"Hips": np.zeros((n, 3))}
    pos["Hips"][:, 1] = 0.95
    pos["Hips"][:, 0] = kx * s / 1000.0
    def fk(child):
        par, off = OFFS[child]
        pos[child] = pos[par] + Rg[par].apply(np.asarray(off, float))
    for nm in ["Spine", "Spine1", "Spine2", "LeftShoulder", "RightShoulder", "LeftArm", "RightArm"]:
        fk(nm)
    # 握把目标点（准备姿态，第 0 帧）：两肩中点前下方
    mid = 0.5 * (pos["LeftArm"][0] + pos["RightArm"][0])
    G = mid + np.array([0.0, -0.55, 0.12])
    for side, sgn in (("Left", 1), ("Right", -1)):
        p_sh = pos[f"{side}Arm"][0]
        u = (G + np.array([sgn * 0.03, 0, 0])) - p_sh
        distal = np.array([sgn * 1.0, 0, 0])
        R_rest_world = min_rot(distal, u)
        R_rest_rel = Rg["Spine2"][0].inv() * R_rest_world
        Rg[f"{side}Arm"] = Rg["Spine2"] * rot([0, 0, 1], ks * s) * R_rest_rel
        Rg[f"{side}ForeArm"] = Rg[f"{side}Arm"] * rot([0, 0, 1], np.full(n, -8.0 * sgn)) * \
            rot([1, 0, 0], sgn * (kr * s + rn))
        fk(f"{side}ForeArm"); fk(f"{side}Hand")
        Rg[f"{side}Hand"] = Rg[f"{side}ForeArm"]
    for nm in ["LeftUpLeg", "RightUpLeg"]:
        Rg[nm] = Rg["Hips"]; fk(nm)
    for nm in ["LeftLeg", "RightLeg"]:
        Rg[nm] = Rg["Hips"]; fk(nm)
    cols = {}
    for nm in ORDER:
        p = pos[nm] + RNG.normal(0, 0.0003, (n, 3))
        q = Rg[nm].as_quat() + RNG.normal(0, 0.0005, (n, 4))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        for j, c in enumerate("xyz"):
            cols[f"{nm}-Joint-Posi-{c}"] = p[:, j]
        for j, c in enumerate("xyzw"):
            cols[f"{nm}-Bone-Quat-{c}"] = q[:, j]
        cols[f"{nm}-Sensor-Lost"] = np.zeros(n, int) if nm in REAL else np.ones(n, int)
    df = pd.DataFrame(cols)
    df.insert(0, "Frame", np.arange(n))
    return df


def main(out):
    out = Path(out)
    data = out / "data"; data.mkdir(parents=True, exist_ok=True)
    # 受试者风格：(骨盆转 kp°, 骨盆平移 kx mm, 躯干 kt°, 摆臂 ks°, 前臂扭转 kr°)，s=-1 为上杆顶点
    styles = {"Ann": (1.0, 2.0, 7.0, 3.0, 4.0), "Bob": (0.5, 1.0, 3.0, 6.0, 6.0),
              "Cat": (1.5, 4.0, 5.0, 4.5, 3.0), "Dan": (0.8, 0.5, 2.0, 7.0, 8.0),
              "Eve": (1.2, 3.0, 6.0, 3.5, 5.0), "Fay": (0.6, 1.5, 4.0, 5.0, 2.0)}
    rows = [["大二年级", None, None, None, None]]
    truth = []
    for si, (name, st) in enumerate(styles.items()):
        if si == 0:
            rows.append([name, "杆头速度(mph)", "杆面角", "杆头轨迹", "击球角"])
        else:
            rows.append([name, None, None, None, None])
        bias = RNG.normal(0, 2)
        for tr in range(1, 11):
            Ab = 1.0 + RNG.normal(0, 0.15)
            rn = RNG.normal(0, 2.0)
            df = make_take(st, Ab, rn)
            fn = data / f"take{si + 1:03d}{name}{tr:03d}_chr01.csv"
            with open(fn, "w", encoding="utf-8") as fh:
                fh.write(f"{len(df)}\n")
                df.to_csv(fh, index=False)
            speed = 3.3 + 1.2 * (Ab - 1) + RNG.normal(0, 0.08)
            face = bias + 0.6 * rn + RNG.normal(0, 0.5)
            path = RNG.normal(-1, 1.5); att = RNG.normal(0, 1)
            fmt = lambda v, p, m: f"{abs(v):.1f}° {p if v >= 0 else m}"
            rows.append([tr, round(speed, 2), fmt(face, "OP", "CL"), fmt(path, "I-O", "O-I"), fmt(att, "UP", "DN")])
            truth.append(dict(subject=name, trial=tr, Ab=Ab, roll_noise=rn, kp=st[0], kx=st[1], kt=st[2], ks=st[3], kr=st[4]))
    pd.DataFrame(rows).to_excel(out / "face.xlsx", header=False, index=False)
    pd.DataFrame(truth).to_csv(out / "truth.csv", index=False)
    print("written", len(truth), "takes to", data)


if __name__ == "__main__":
    main(sys.argv[1])
