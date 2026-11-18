# -*- coding: utf-8 -*-
"""Additional statistics added in revision v1.1 (from the processed data only).

  - ICC(1) with 95% participant-level (cluster) bootstrap CIs              (Tables 2, Section 3.2; R2-38)
  - isochrony: within-player slope of ln(downswing time) on ln(downswing displacement)   (Section 3.3, Table 3 P6)
  - trial-order trends (r_rm with trial number)                             (Section 3.6, Figure S3; R1-7)
  - leave-one-participant-out ranges of the main shares                     (Table S4; R2-33)
  - sex comparison of player medians (reported only in the response letter; R2-59)
  - Figure 2b statistics: slope of shoulder turn on pelvis turn across player medians, within-player r_rm
  - forearm roll in space vs pronation-supination at top and impact         (Section 3.5, Figure S2; R1-4)
  - largest residuals                                                       (Section 3.1; R2-35)

    python extra_stats.py --proc ../../data/processed --out ../../results/revision/extra_stats.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code" / "analysis"))
import make_results as mr  # noqa: E402

N_BOOT = 2000


def icc_arr(groups):
    k = np.array([len(g) for g in groups]); n = len(groups); N = k.sum()
    gm = np.concatenate(groups).mean(); means = np.array([g.mean() for g in groups])
    msb = (k * (means - gm) ** 2).sum() / (n - 1)
    msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / (N - n)
    k0 = (N - (k ** 2).sum() / N) / (n - 1)
    return (msb - msw) / (msb + (k0 - 1) * msw)


def wdev(d, c):
    return d[c] - d.groupby("subject")[c].transform("mean")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc", type=Path, default=REPO / "data" / "processed")
    ap.add_argument("--participants", type=Path, default=REPO / "data" / "participants.csv")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "revision" / "extra_stats.json")
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    _, ok, _ = mr.load(a.proc)
    ch = pd.read_csv(a.proc / "chain_features.csv")
    d = ok.merge(ch[[c for c in ch.columns if c not in ok.columns or c == "file"]], on="file").copy()
    tot = d.bs_total_mm
    d["body_share"] = 100 * (d.bs_pelvis_trans_mm + d.bs_pelvis_rot_mm + d.bs_trunk_mm + d.bs_girdle_mm) / tot
    d["arms_share"] = 100 * (d.bs_shoulder_lead_mm + d.bs_shoulder_trail_mm + d.bs_elbow_lead_mm + d.bs_elbow_trail_mm) / tot
    d["trunk_share"] = 100 * (d.bs_trunk_mm + d.bs_girdle_mm) / tot
    d["pelvis_share"] = 100 * (d.bs_pelvis_trans_mm + d.bs_pelvis_rot_mm) / tot
    d["lnD"], d["lnT"] = np.log(d.ds_total_mm), np.log(d.t_downswing_s)
    d["lnv"] = d.lnD - d.lnT
    subs = sorted(d.subject.unique())
    R = dict(n_putts=int(len(d)), n_subjects=len(subs))

    # ICC(1) with cluster-bootstrap CI (participants resampled with replacement; one RNG stream, variables in order)
    rng = np.random.default_rng(1)
    R["icc"] = {}
    for x in ["trunk_share", "arms_share", "t_downswing_s", "t_backswing_s", "tempo_ratio", "backswing_len_mm",
              "grip_speed_impact", "head_speed"]:
        G = [d.loc[d.subject == s, x].values for s in subs]
        bs = [icc_arr([G[i] for i in rng.integers(0, len(G), len(G))]) for _ in range(N_BOOT)]
        R["icc"][x] = dict(icc=icc_arr(G), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))

    # isochrony elasticity
    def elast(dd):
        x, y = wdev(dd, "lnD"), wdev(dd, "lnT"); return float((x * y).sum() / (x * x).sum())
    G = {s: d[d.subject == s] for s in subs}
    rng = np.random.default_rng(3)
    bs = [elast(pd.concat([G[s].assign(subject=f"b{j}") for j, s in enumerate(rng.choice(subs, len(subs)))]))
          for _ in range(N_BOOT)]
    R["isochrony_elasticity"] = dict(slope=elast(d), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))

    # trial order
    R["trial_order"] = {v: {k: float(mr.rmcorr(d, "trial", v)[k]) for k in ("r", "p", "dof")}
                        for v in ["backswing_len_mm", "t_backswing_s", "body_share", "arms_share", "head_speed",
                                  "t_downswing_s", "grip_speed_impact", "face_deg"]}

    # leave one participant out
    def amp_share(dd):
        x, v = wdev(dd, "lnD"), wdev(dd, "lnv"); return float((x * v).sum() / (v * v).sum())
    lo = []
    for s in subs:
        dd = d[d.subject != s]
        m = dd.groupby("subject")[["body_share", "pelvis_share", "trunk_share", "arms_share"]].median().mean()
        lo.append(dict(amp=100 * amp_share(dd), **m.to_dict()))
    lo = pd.DataFrame(lo)
    R["leave_one_out"] = {c: [float(lo[c].min()), float(lo[c].max())] for c in lo.columns}

    # sex (player medians)
    pp = pd.read_csv(a.participants).set_index("participant_id")
    pm = d.groupby("subject")[["body_share", "arms_share", "trunk_share", "pelvis_share", "head_speed"]].median().join(pp.sex)
    R["sex"] = {v: dict(M=[float(pm[pm.sex == "M"][v].mean()), float(pm[pm.sex == "M"][v].std())],
                        F=[float(pm[pm.sex == "F"][v].mean()), float(pm[pm.sex == "F"][v].std())],
                        p=float(st.mannwhitneyu(pm[pm.sex == "M"][v], pm[pm.sex == "F"][v]).pvalue))
                for v in pm.columns[:-1]}

    # Figure 2b
    m = d.groupby("subject")[["top_d_pelvis_turn_closed", "top_d_shoulder_turn_closed", "top_d_xfactor",
                              "top_d_shoulder_tilt_leadup"]].median()
    lr = st.linregress(m.top_d_pelvis_turn_closed, m.top_d_shoulder_turn_closed)
    R["fig2b"] = dict(slope=lr.slope, lo=lr.slope - 1.96 * lr.stderr, hi=lr.slope + 1.96 * lr.stderr,
                      r_players=lr.rvalue, r_rm=mr.rmcorr(d, "top_d_pelvis_turn_closed", "top_d_shoulder_turn_closed")["r"],
                      xfactor_mean=float(m.top_d_xfactor.mean()), xfactor_sd=float(m.top_d_xfactor.std()))

    # forearm roll vs pronation-supination (player medians)
    R["forearm"] = {}
    for v in ["top_d_forearm_roll_lead", "imp_d_forearm_roll_lead", "top_d_forearm_roll_trail", "imp_d_forearm_roll_trail",
              "top_d_pronsup_lead", "imp_d_pronsup_lead", "top_d_pronsup_trail", "imp_d_pronsup_trail",
              "rng_forearm_roll_lead", "rng_forearm_roll_trail", "rng_pronsup_lead", "rng_pronsup_trail"]:
        s = d.groupby("subject")[v].median()
        R["forearm"][v] = dict(mean=float(s.mean()), sd=float(s.std()), min=float(s.min()), max=float(s.max()))
    for side in ("lead", "trail"):
        cols = [f"roll_{side}_{g}_top_deg" for g in ["pelvis_rot", "trunk", "girdle", "shoulder", "elbow", "total"]]
        R["forearm"][f"roll_sources_top_{side}"] = d.groupby("subject")[cols].median().mean().to_dict()

    # largest residuals
    R["largest_residuals"] = d.nlargest(4, "chain_resid_x_pct")[["file", "chain_resid_x_pct"]].to_dict("records")
    R["grip_vs_head_speed_ms"] = [float(d.grip_speed_impact.mean()), float(d.head_speed.mean() * 0.44704)]

    json.dump(R, open(a.out, "w"), indent=1, default=float)
    f = lambda x: f"{x:.2f}"
    print(f"{R['n_putts']} putts, {R['n_subjects']} players")
    for k, v in R["icc"].items():
        print(f"  ICC {k:18s} {f(v['icc'])} [{f(v['lo'])}, {f(v['hi'])}]")
    e = R["isochrony_elasticity"]; print(f"  isochrony elasticity {f(e['slope'])} [{f(e['lo'])}, {f(e['hi'])}]")
    for k, v in R["trial_order"].items():
        print(f"  trial order {k:18s} r_rm = {v['r']:+.2f} (p = {v['p']:.3f})")
    print("  leave-one-out ranges:", {k: [round(x, 1) for x in v] for k, v in R["leave_one_out"].items()})
    for k, v in R["forearm"].items():
        if "mean" in v:
            print(f"  {k:26s} {v['mean']:6.1f} ± {v['sd']:4.1f}  [{v['min']:.1f}, {v['max']:.1f}]")
    print("  Fig 2b:", {k: round(float(v), 2) for k, v in R["fig2b"].items()})


if __name__ == "__main__":
    main()
