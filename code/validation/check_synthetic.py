# -*- coding: utf-8 -*-
"""Synthetic validation of the kinematic-chain decomposition (Section 2.5 of the paper).

1. Generates synthetic putts with known joint rotations in the motion-capture export format
   (make_synthetic.py: 6 players x 10 putts, forward kinematics, so positions and orientations agree).
2. Runs the full preprocessing pipeline on them (code/preprocessing/run.py).
3. Checks that
   - the decomposed terms add up to the measured backswing length,
   - the pelvis-translation term tracks its set value,
   - putt-to-putt changes in forearm roll are attributed to the elbow/forearm.

Usage (from the repository root):
    python code/validation/check_synthetic.py [--out validation_output]
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import make_synthetic  # noqa: E402

BS_TERMS = ["bs_pelvis_trans_mm", "bs_pelvis_rot_mm", "bs_trunk_mm", "bs_girdle_mm",
            "bs_shoulder_lead_mm", "bs_shoulder_trail_mm", "bs_elbow_lead_mm", "bs_elbow_trail_mm", "bs_resid_mm"]
ROLL_TERMS = ["pelvis_rot", "trunk", "girdle", "shoulder", "elbow"]


def within_share(df, parts, total):
    """Share of within-player variance of `total` carried by each part: cov_w(c_j, C) / var_w(C)."""
    w = df[parts + [total]] - df.groupby("subject")[parts + [total]].transform("mean")
    v = (w[total] ** 2).sum()
    return {p: float((w[p] * w[total]).sum() / v) for p in parts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "validation_output")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    make_synthetic.main(a.out)                       # -> out/data/*.csv, out/face.xlsx, out/truth.csv
    proc = a.out / "processed"
    cmd = [sys.executable, str(REPO / "code" / "preprocessing" / "run.py"),
           "--data", str(a.out / "data"), "--face", str(a.out / "face.xlsx"), "--out", str(proc)]
    subprocess.run(cmd, check=True, cwd=REPO / "code" / "preprocessing")

    ch = pd.read_csv(proc / "chain_features.csv")
    fe = pd.read_csv(proc / "features.csv")
    truth = pd.read_csv(a.out / "truth.csv")
    d = ch.merge(fe[["subject", "trial", "backswing_len_mm"]], on=["subject", "trial"]).merge(truth, on=["subject", "trial"])

    err_sum = (d[BS_TERMS].sum(axis=1) - d["bs_total_mm"]).abs().max()
    err_meas = (d["bs_total_mm"] - d["backswing_len_mm"]).abs()
    r_trans = np.corrcoef(d["bs_pelvis_trans_mm"], d["kx"] * d["Ab"])[0, 1]
    roll = within_share(d, [f"roll_lead_{t}_imp_deg" for t in ROLL_TERMS], "roll_lead_total_imp_deg")

    print("\n=== Synthetic validation ===")
    print(f"putts analysed: {len(d)}")
    print(f"max |sum of terms - decomposed backswing length| : {err_sum:.4f} mm")
    print(f"|decomposed - measured backswing length| median / max : {err_meas.median():.3f} / {err_meas.max():.3f} mm")
    print(f"pelvis-translation term vs set value, r : {r_trans:.3f}")
    print("share of within-player variance in lead forearm roll at impact:")
    for k, v in roll.items():
        print(f"    {k:<32s} {100 * v:6.1f}%")


if __name__ == "__main__":
    main()
