# -*- coding: utf-8 -*-
"""Data-quality statistics added in revision v1.1 (Section 3.1, Table S3, Figure S4; R1-3, R2-18, R2-19, R2-35).

  - missing frames per physical sensor in the stroke window (takeaway - 10 frames to finish + 10 frames);
  - longest gap of each chain sensor, and of the six chain sensors combined (any of them lost);
  - the same in the window used by exclusion rule R3 (6 frames before the top to 7 frames after impact);
  - kinematic impact vs forward grip-velocity peak;
  - spectrum of the unfiltered grip velocity along the target axis (frequency below which 99% of the power lies,
    and the fraction of power below 6 Hz) - justification of the 6 Hz low-pass filter;
  - longest chain gaps of the putts with the largest decomposition residuals.

Needs the raw recordings.

    python data_quality.py --proc ../../data/processed --raw ../../data/raw --out ../../results/revision/data_quality
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
from putt.io_utils import read_take                    # noqa: E402

SENS = ["Hips", "Spine2", "LeftArm", "RightArm", "LeftForeArm", "RightForeArm",
        "LeftUpLeg", "RightUpLeg", "LeftLeg", "RightLeg"]
CHAIN = SENS[:6]


def runs(m):
    out, c = [], 0
    for v in m:
        if v: c += 1
        elif c: out.append(c); c = 0
    if c: out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc", type=Path, default=REPO / "data" / "processed")
    ap.add_argument("--raw", type=Path, default=REPO / "data" / "raw")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "revision" / "data_quality")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    _, ok, _ = mr.load(a.proc)
    ev = pd.read_csv(a.proc / "events.csv").set_index("file")
    ch = pd.read_csv(a.proc / "chain_features.csv").set_index("file")
    rows = []
    for f in ok.file:
        e = ev.loc[f]
        df, _ = read_take(a.raw / f)
        L = df[[f"{s}-Sensor-Lost" for s in SENS]].values.astype(bool)
        w = slice(max(0, int(e.takeaway) - 10), int(e.finish) + 11)
        c = slice(max(0, int(e.top) - 6), int(e.impact) + 7)
        rec = dict(file=f, subject=f.split("_")[1], n_win=w.stop - w.start)
        for i, s in enumerate(SENS):
            rec[f"{s}_lost_pct"] = 100 * L[w, i].mean()
            rec[f"{s}_maxgap"] = max(runs(L[w, i]), default=0)
        anyc = L[:, :6].any(1)
        rec["chain_any_lost_pct"] = 100 * anyc[w].mean()
        rec["chain_combined_maxgap"] = max(runs(anyc[w]), default=0)
        rec["chain_combined_maxgap_R3_window"] = max(runs(anyc[c]), default=0)
        rec["chain_single_maxgap"] = max(rec[f"{s}_maxgap"] for s in CHAIN)
        rec["impact_minus_peak_frames"] = float(e.impact - e.peak_v)
        # unfiltered grip velocity spectrum (per-sensor gap filling as in the main analysis, no low-pass)
        q, p, _ = clean.clean_take(df, cfg.FS, 0.99 * cfg.FS / 2, ["Hips"], ["Hips", "LeftHand", "RightHand"],
                                   cfg.REAL_SEGMENTS)
        M = kin.heading_matrix(kin.avg_quat(q["Hips"][int(e.address_start):int(e.address_end)]))
        g = (M @ (0.5 * (p["LeftHand"] + p["RightHand"])).T).T[:, 0]
        v = np.gradient(g) * cfg.FS
        x = v[w] - v[w].mean()
        P = np.abs(np.fft.rfft(x * np.hanning(len(x)), 1024)) ** 2
        fr = np.fft.rfftfreq(1024, 1 / cfg.FS); cp = np.cumsum(P) / P.sum()
        rec["f99_hz"] = float(fr[np.searchsorted(cp, 0.99)])
        rec["power_below_6hz"] = float(cp[np.searchsorted(fr, 6.0)])
        rec["chain_resid_x_pct"] = float(ch.loc[f, "chain_resid_x_pct"])
        rows.append(rec)
    d = pd.DataFrame(rows)
    d.to_csv(a.out / "data_quality.csv", index=False)
    S = dict(n_putts=int(len(d)))
    S["lost_pct_median_by_sensor"] = {s: float(d[f"{s}_lost_pct"].median()) for s in SENS}
    S["maxgap_median_by_sensor"] = {s: float(d[f"{s}_maxgap"].median()) for s in SENS}
    for k in ("chain_single_maxgap", "chain_combined_maxgap"):
        S[k] = dict(median=float(d[k].median()), gt6=int((d[k] > 6).sum()), gt20=int((d[k] > 20).sum()), max=int(d[k].max()))
    k = "chain_combined_maxgap_R3_window"
    S[k] = dict(median=float(d[k].median()), max=int(d[k].max()), gt20=int((d[k] > 20).sum()))
    dd = d.impact_minus_peak_frames
    S["impact_minus_peak"] = dict(median=float(dd.median()), mean=float(dd.mean()), abs_gt5=int((dd.abs() > 5).sum()),
                                  abs_gt5_pct=float(100 * (dd.abs() > 5).mean()))
    S["spectrum"] = dict(f99_median=float(d.f99_hz.median()), f99_max=float(d.f99_hz.max()),
                         below6_median=float(d.power_below_6hz.median()), below6_min=float(d.power_below_6hz.min()))
    S["largest_residuals"] = d.nlargest(4, "chain_resid_x_pct")[
        ["file", "chain_resid_x_pct", "chain_combined_maxgap", "chain_combined_maxgap_R3_window"]].to_dict("records")
    json.dump(S, open(a.out / "data_quality_summary.json", "w"), indent=1, default=float)
    print(json.dumps(S, indent=1, default=float))


if __name__ == "__main__":
    main()
