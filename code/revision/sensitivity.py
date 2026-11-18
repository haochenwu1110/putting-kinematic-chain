# -*- coding: utf-8 -*-
"""Sensitivity analyses added in revision v1.1 (Section 3.6, Supplementary S2, Table S4, Figure S1).

Step 1 - preprocessing variants (need the raw recordings): the full pipeline is rerun with one setting changed
         through environment variables read by code/preprocessing (see the VARIANTS table below).
Step 2 - exclusion variants (from the processed data of the main analysis): selected putts are marked FAIL in a
         copy of qc_report.csv and the analysis is rerun.
Step 3 - pendulum / lower-limb test under the gap-filling and filter variants.
Step 4 - summary table (Table S4) -> sensitivity_table.csv

    python sensitivity.py --raw ../../data/raw --proc ../../data/processed --work ../../results/revision/sensitivity
    python sensitivity.py --skip-preproc ...      # only steps 2 and 4
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PRE, AN, REV = REPO / "code" / "preprocessing", REPO / "code" / "analysis", REPO / "code" / "revision"

VARIANTS = {  # name: (label, environment)
    "gap_union": ("Gap filling: v1.0 method (any of 10 sensors)", {"PUTT_MASK_MODE": "union"}),
    "gap_chain": ("Gap filling: any of 6 chain sensors", {"PUTT_MASK_MODE": "chain"}),
    "gap_none": ("Gap filling: none (manufacturer estimate)", {"PUTT_MASK_MODE": "none"}),
    "impact_m2": ("Impact −2 frames", {"PUTT_IMPACT_SHIFT": "-2"}),
    "impact_m1": ("Impact −1 frame", {"PUTT_IMPACT_SHIFT": "-1"}),
    "impact_p1": ("Impact +1 frame", {"PUTT_IMPACT_SHIFT": "1"}),
    "impact_p2": ("Impact +2 frames", {"PUTT_IMPACT_SHIFT": "2"}),
    "lowpass_4": ("Low-pass 4 Hz", {"PUTT_LOWPASS": "4"}),
    "lowpass_8": ("Low-pass 8 Hz", {"PUTT_LOWPASS": "8"}),
    "axis_ml_p5": ("Axes +5° about medio-lateral axis", {"PUTT_AXIS_PERTURB_DEG": "5", "PUTT_AXIS_PERTURB_DIR": "x"}),
    "axis_ml_m5": ("Axes −5° about medio-lateral axis", {"PUTT_AXIS_PERTURB_DEG": "-5", "PUTT_AXIS_PERTURB_DIR": "x"}),
    "axis_ml_p10": ("Axes +10° about medio-lateral axis", {"PUTT_AXIS_PERTURB_DEG": "10", "PUTT_AXIS_PERTURB_DIR": "x"}),
    "axis_ml_m10": ("Axes −10° about medio-lateral axis", {"PUTT_AXIS_PERTURB_DEG": "-10", "PUTT_AXIS_PERTURB_DIR": "x"}),
    "axis_ap_p5": ("Axes +5° about antero-posterior axis", {"PUTT_AXIS_PERTURB_DEG": "5", "PUTT_AXIS_PERTURB_DIR": "z"}),
    "axis_ap_m5": ("Axes −5° about antero-posterior axis", {"PUTT_AXIS_PERTURB_DEG": "-5", "PUTT_AXIS_PERTURB_DIR": "z"}),
    "axis_ap_p10": ("Axes +10° about antero-posterior axis", {"PUTT_AXIS_PERTURB_DEG": "10", "PUTT_AXIS_PERTURB_DIR": "z"}),
    "axis_ap_m10": ("Axes −10° about antero-posterior axis", {"PUTT_AXIS_PERTURB_DEG": "-10", "PUTT_AXIS_PERTURB_DIR": "z"}),
    "spine_merged": ("Estimated lumbar segments removed", {"PUTT_SPINE_COLLAPSE": "1"}),
    "takeaway_strict": ("Takeaway threshold (1 mm, 1.5%)", {"PUTT_TAKE_MM": "1", "PUTT_TAKE_FRAC": "0.015"}),
    "takeaway_loose": ("Takeaway threshold (4 mm, 6%)", {"PUTT_TAKE_MM": "4", "PUTT_TAKE_FRAC": "0.06"}),
}
PENDULUM_VARIANTS = ["gap_union", "gap_none", "lowpass_4", "lowpass_8"]
EXCLUSIONS = {  # name: (label, function(qc) -> boolean mask of putts to drop)
    "excl_gap_gt20": ("Excl. combined chain gap > 20 frames", lambda q: q.longest_gap > 20),
    "excl_gap_gt6": ("Excl. combined chain gap > 6 frames", lambda q: q.longest_gap > 6),
    "excl_impact_peak_gt5": ("Excl. |impact − velocity peak| > 5 frames", lambda q: (q.impact_frame - q.peak_v_frame).abs() > 5),
    "excl_resid_gt10": ("Excl. residual > 10%", lambda q: q.chain_resid_x_pct > 10),
    "excl_trials_1_2": ("Excl. trials 1–2", lambda q: q.trial <= 2),
    "excl_trials_1_3": ("Excl. trials 1–3", lambda q: q.trial <= 3),
    "excl_P07": ("Excl. participant P07", lambda q: q.subject == "P07"),
}


def run(args, env=None, cwd=AN, log=None):
    e = dict(os.environ); e.update(env or {})
    with open(log, "a") if log else open(os.devnull, "w") as fh:
        subprocess.run([sys.executable, *map(str, args)], cwd=cwd, env=e, check=True, stdout=fh, stderr=fh)


def analyse(proc, res, log):
    run(["chain_results.py", "--dir", proc, "--out", res], log=log)
    run(["table2_timing.py", "--dir", proc, "--out", res / "table2_timing.csv"], log=log)


def summary_row(label, res):
    R = json.load(open(res / "chain_results.json"))
    bs, sh, q2 = R["Q1"]["backswing"], R["axes"]["backswing"]["shares"], R["Q2"]
    amp = q2["amplitude_vs_timing"]["amplitude"]
    fa = {a["x"]: a for a in R["Q3"]["face_assoc"]}["roll_trail_total_imp_deg"]
    t2 = pd.read_csv(res / "table2_timing.csv").set_index("variable")
    return {"analysis": label, "putts": R["n_putts"], "players": R["n_subjects"],
            "resid_max_pct": round(R["qc"]["resid_x_pct_max"], 1),
            "body_pct": round(bs["body"]["mean"]), "arms_pct": round(bs["arms"]["mean"]),
            "pelvis_twist_pct": round(sh["Pelvis twist"]["mean"]), "trunk_twist_pct": round(sh["Trunk twist"]["mean"]),
            "trunk_sidebend_pct": round(sh["Trunk side-bend"]["mean"]), "shoulder_swing_pct": round(sh["Shoulder swing"]["mean"]),
            "amplitude_share_pct": round(100 * amp["share"]), "amp_lo": round(100 * amp["lo"]), "amp_hi": round(100 * amp["hi"]),
            "r_grip_v_head_speed": round(q2["head_speed_links"]["grip_speed_impact"]["r"], 2),
            "r_trail_roll_face": round(fa["r"], 2), "trail_roll_face_robust": bool(fa["q"] < 0.05 and fa["robust"]),
            "n_robust_face": sum(1 for a in R["Q3"]["face_assoc"] if a["q"] < 0.05 and a["robust"]),
            "backswing_time_cv_pct": round(t2.loc["Backswing time (s)", "cv_median_pct"], 1),
            "downswing_time_cv_pct": round(t2.loc["Downswing time (s)", "cv_median_pct"], 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=REPO / "data" / "raw")
    ap.add_argument("--proc", type=Path, default=REPO / "data" / "processed")
    ap.add_argument("--work", type=Path, default=REPO / "results" / "revision" / "sensitivity")
    ap.add_argument("--main-res", type=Path, default=REPO / "results")
    ap.add_argument("--skip-preproc", action="store_true")
    a = ap.parse_args()
    W = a.work.resolve(); W.mkdir(parents=True, exist_ok=True); log = W / "log.txt"
    rows = [summary_row("Main analysis", a.main_res.resolve())]

    if not a.skip_preproc:
        for name, (label, env) in VARIANTS.items():
            proc, res = W / name / "proc", W / name / "res"
            run(["run.py", "--data", a.raw.resolve(), "--out", proc], env=env, cwd=PRE, log=log)
            analyse(proc, res, log)
            rows.append(summary_row(label, res)); print("done", name, flush=True)
        for name in PENDULUM_VARIANTS:
            run(["pendulum_lower_limb.py", "--proc", W / name / "proc", "--raw", a.raw.resolve(),
                 "--out", W / name / "pendulum", "--no-curves"], env=VARIANTS[name][1], cwd=REV, log=log)

    q0 = pd.read_csv(a.proc / "qc_report.csv")
    ch = pd.read_csv(a.proc / "chain_features.csv")[["file", "chain_resid_x_pct"]]
    qm = q0.merge(ch, on="file", how="left")
    for name, (label, rule) in EXCLUSIONS.items():
        proc = W / name / "proc"; proc.mkdir(parents=True, exist_ok=True)
        for f in a.proc.resolve().iterdir():
            t = proc / f.name
            if f.name != "qc_report.csv" and not t.exists():
                t.symlink_to(f)
        q = q0.copy(); q.loc[rule(qm).values, "status"] = "FAIL"; q.to_csv(proc / "qc_report.csv", index=False)
        analyse(proc, W / name / "res", log)
        rows.append(summary_row(label, W / name / "res")); print("done", name, flush=True)

    t = pd.DataFrame(rows); t.to_csv(W / "sensitivity_table.csv", index=False)
    pd.set_option("display.width", 250); print(t.to_string(index=False))


if __name__ == "__main__":
    main()
