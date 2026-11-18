# -*- coding: utf-8 -*-
"""Pendulum geometry and lower-limb analysis (paper Sections 2.7 and 3.5, Figure 5; added in revision v1.1).

For every retained putt:
  (a) hub test - the hub is the midpoint of the two shoulder-joint centres. Signed displacement of hub and grip
      along the target axis at the top of the backswing (relative to the address-window mean), the hub's 3-D
      excursion from its address position, and the hub-to-grip distance (radius) from takeaway to impact;
  (b) lower limb - rotation of pelvis, thighs and shanks about the body vertical from address to top, pelvis and
      foot translation, and the split of the pelvis-rotation contribution to backswing length (Eq. 4) into
      pelvis-on-thighs rotation (hip joints) and thigh rotation in space (mean thigh angular velocity).
Also writes the time-normalised curves used for Figure 5.

Needs the raw recordings (data/raw). Gap filling follows PUTT_MASK_MODE (default: per-sensor, as in the paper).

    python pendulum_lower_limb.py --proc ../../data/processed --raw ../../data/raw --out ../../results/revision/pendulum
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "code" / "preprocessing"), str(REPO / "code" / "analysis")]
import config as cfg                                   # noqa: E402
import make_results as mr                              # noqa: E402
from putt import clean, kin                            # noqa: E402
from putt.chain import angvel, _cumint, _at            # noqa: E402
from putt.io_utils import read_take                    # noqa: E402

QS = ["Hips", "LeftUpLeg", "RightUpLeg", "LeftLeg", "RightLeg", "Spine2"]
PS = ["Hips", "LeftHand", "RightHand", "LeftArm", "RightArm", "LeftFoot", "RightFoot", "Neck"]
N = 101


def norm3(y, tk, tt, ti, tf):
    r = lambda a, b: np.interp(np.linspace(a, b, N), np.arange(len(y)), y)
    return np.concatenate([r(tk, tt), r(tt, ti), r(ti, tf)])


def one_putt(path, e, fs):
    df, _ = read_take(path)
    quats, pos, _ = clean.clean_take(df, fs, cfg.LOWPASS_HZ, QS, PS, cfg.REAL_SEGMENTS)
    a0, a1 = int(e.address_start), int(e.address_end)
    tk, tt, ti, tf = int(e.takeaway), int(e.top), float(e.impact), int(e.finish)
    amid = 0.5 * (a0 + a1)
    M = kin.heading_matrix(kin.avg_quat(quats["Hips"][a0:a1]))
    H = lambda v: (M @ np.asarray(v).T).T * 1000.0                      # body frame, mm
    rec, curves = {}, {}

    # ---- (a) hub and radius
    grip = H(0.5 * (pos["LeftHand"] + pos["RightHand"]))
    hub = H(0.5 * (pos["LeftArm"] + pos["RightArm"]))
    neck = H(pos["Neck"])
    g0, h0, n0 = grip[a0:a1].mean(0), hub[a0:a1].mean(0), neck[a0:a1].mean(0)
    w = slice(tk, int(np.ceil(ti)) + 1)                                   # takeaway -> impact
    rec["grip_top_mm"] = grip[tt, 0] - g0[0]                              # signed, toward target +
    rec["hub_top_mm"] = hub[tt, 0] - h0[0]
    rec["hub_top_ratio_pct"] = 100.0 * rec["hub_top_mm"] / rec["grip_top_mm"]
    rec["grip_range_mm"] = np.ptp(grip[w, 0])
    rec["hub_range_mm"] = np.ptp(hub[w, 0])
    rec["hub_path3d_mm"] = np.max(np.linalg.norm(hub[w] - h0, axis=1))
    rec["neck_path3d_mm"] = np.max(np.linalg.norm(neck[w] - n0, axis=1))
    rad = np.linalg.norm(grip - hub, axis=1)
    rad_fixed = np.linalg.norm(grip - h0, axis=1)                         # hub fixed at its address position
    rec["radius_mean_mm"] = rad[w].mean()
    rec["radius_range_mm"] = np.ptp(rad[w])
    rec["radius_fixed_hub_range_mm"] = np.ptp(rad_fixed[w])

    # ---- (b) lower limb
    W = {s: angvel(quats[s], fs) for s in QS}
    w_thigh = 0.5 * (W["LeftUpLeg"] + W["RightUpLeg"])
    pE = 0.5 * (pos["LeftHand"] + pos["RightHand"])
    rE = pE - pos["Hips"]
    parts = {"pelvis_rot": np.cross(W["Hips"], rE),                      # = pelvis term of Eq. (4)
             "hip_joint": np.cross(W["Hips"] - w_thigh, rE),             # pelvis rotating on the thighs
             "thigh_in_space": np.cross(w_thigh, rE)}                    # thighs rotating in space
    rec["bs_len_mm"] = g0[0] - grip[tt, 0]
    for k, v in parts.items():
        c = _cumint(H(v)[:, 0], fs)                                      # mm
        rec[f"bs_{k}_mm"] = -(_at(c, tt) - _at(c, amid))
    for s in QS:
        c = np.degrees(_cumint((M @ W[s].T).T[:, 1], fs))
        rec[f"yaw_{s}_deg"] = -(_at(c, tt) - _at(c, amid))               # backswing rotation +
    for s in ("Hips", "LeftFoot", "RightFoot"):
        p = H(pos[s])[:, 0]
        rec[f"dx_{s}_mm"] = abs(p[tt] - p[a0:a1].mean())

    # ---- curves for Figure 5
    curves["grip_x"] = grip[:, 0] - g0[0]
    curves["hub_x"] = hub[:, 0] - h0[0]
    curves["radius_change"] = rad - rad[a0:a1].mean()
    curves["fixed_radius_change"] = rad_fixed - rad_fixed[a0:a1].mean()
    for lab, wv in (("yaw_pelvis", W["Hips"]), ("yaw_thigh", w_thigh),
                    ("yaw_shank", 0.5 * (W["LeftLeg"] + W["RightLeg"]))):
        c = -np.degrees(_cumint((M @ wv.T).T[:, 1], fs))
        curves[lab] = c - c[a0:a1].mean()
    curves = {k: norm3(v, tk, tt, ti, tf) for k, v in curves.items()}
    return rec, curves


def summarise(d):
    for k in ("pelvis_rot", "hip_joint", "thigh_in_space"):
        d[f"{k}_share_pct"] = 100.0 * d[f"bs_{k}_mm"] / d.bs_len_mm
    d["radius_range_pct"] = 100.0 * d.radius_range_mm / d.radius_mean_mm
    d["radius_fixed_range_pct"] = 100.0 * d.radius_fixed_hub_range_mm / d.radius_mean_mm
    pm = d.groupby("subject").median(numeric_only=True)
    S = {c: dict(mean=float(pm[c].mean()), sd=float(pm[c].std()), min=float(pm[c].min()), max=float(pm[c].max()))
         for c in pm.columns}
    S["hub_same_direction_players"] = int((pm.hub_top_ratio_pct > 0).sum())
    S["hub_opposite_direction_players"] = sorted(pm.index[pm.hub_top_ratio_pct <= 0].tolist())
    S["n_putts"] = int(len(d)); S["n_subjects"] = int(len(pm))
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc", type=Path, default=REPO / "data" / "processed")
    ap.add_argument("--raw", type=Path, default=REPO / "data" / "raw")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "revision" / "pendulum")
    ap.add_argument("--no-curves", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    _, ok, _ = mr.load(a.proc)
    ev = pd.read_csv(a.proc / "events.csv").set_index("file") if (a.proc / "events.csv").exists() else None
    if ev is None:
        raise SystemExit("events.csv not found in --proc (run code/preprocessing/run.py first)")
    rows, crow = [], []
    for f in ok.file:
        rec, cur = one_putt(a.raw / f, ev.loc[f], cfg.FS)
        subj = f.split("_")[1]
        rows.append(dict(file=f, subject=subj, **rec))
        if not a.no_curves:
            crow += [[f, subj, k] + list(v) for k, v in cur.items()]
    d = pd.DataFrame(rows)
    d.to_csv(a.out / "pendulum_lower_limb.csv", index=False)
    if crow:
        cols = [f"{p}_{i:03d}" for p in ("back", "down", "follow") for i in range(N)]
        pd.DataFrame(crow, columns=["file", "subject", "var"] + cols).to_csv(a.out / "figure5_curves.csv.gz", index=False)
    S = summarise(d)
    json.dump(S, open(a.out / "pendulum_lower_limb_summary.json", "w"), indent=1)
    keys = ["grip_top_mm", "hub_top_mm", "hub_top_ratio_pct", "hub_path3d_mm", "neck_path3d_mm", "radius_mean_mm",
            "radius_range_mm", "radius_range_pct", "radius_fixed_hub_range_mm", "radius_fixed_range_pct",
            "pelvis_rot_share_pct", "hip_joint_share_pct", "thigh_in_space_share_pct", "yaw_Hips_deg",
            "yaw_LeftUpLeg_deg", "yaw_RightUpLeg_deg", "yaw_LeftLeg_deg", "yaw_RightLeg_deg", "dx_Hips_mm",
            "dx_LeftFoot_mm", "dx_RightFoot_mm"]
    print(f"{S['n_putts']} putts, {S['n_subjects']} players (mean ± SD [min, max] of player medians)")
    for k in keys:
        s = S[k]; print(f"  {k:28s} {s['mean']:8.2f} ± {s['sd']:6.2f}  [{s['min']:.1f}, {s['max']:.1f}]")
    print(f"  hub moves with the grip at the top: {S['hub_same_direction_players']} players; "
          f"opposite: {S['hub_opposite_direction_players']}")


if __name__ == "__main__":
    main()
