# -*- coding: utf-8 -*-
"""Figures added in revision v1.1: Figure 5 (pendulum geometry and lower limb) and Figures S1-S4.

Run after pendulum_lower_limb.py, data_quality.py and sensitivity.py (see run_revision.py).

    python revision_figures.py --rev ../../results/revision --proc ../../data/processed --main-res ../../results
"""
import argparse, json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code" / "analysis"))
ap = argparse.ArgumentParser()
ap.add_argument("--rev", type=Path, default=REPO / "results" / "revision")
ap.add_argument("--proc", type=Path, default=REPO / "data" / "processed")
ap.add_argument("--main-res", type=Path, default=REPO / "results")
ap.add_argument("--font", default="Liberation Serif")
A = ap.parse_args()
REV, PROC = A.rev.resolve(), A.proc.resolve()
SENS = REV / "sensitivity"
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from pathlib import Path
import chain_figures as cf, make_results as mr
cf.style(A.font)
INK, MUTED, GRID, PALE, COL = cf.INK, cf.MUTED, cf.GRID, cf.PALE, cf.COL
OUT = REV / "figures"; OUT.mkdir(parents=True, exist_ok=True)
cols = [f"{p}_{i:03d}" for p in ("back", "down", "follow") for i in range(101)]
x = np.arange(303)
def phase_axis(ax):
    for v in (101, 202): ax.axvline(v, color=MUTED, lw=0.6, ls=(0, (2, 2)))
    ax.set_xticks([0, 101, 202, 302], ["TA", "Top", "Impact", "Finish"]); ax.set_xlim(0, 302)
def band(ax, c, var, color, label, lw=1.4):
    m = c[c["var"] == var].groupby("subject")[cols].mean()
    mu, sd = m.mean().values, m.std().values
    ax.fill_between(x, mu - sd, mu + sd, color=color, alpha=0.18, lw=0)
    ax.plot(x, mu, color=color, lw=lw, label=label)
    return mu

# ---------------- Figure 5: pendulum geometry and lower body
c = pd.read_csv(REV / "pendulum" / "figure5_curves.csv.gz")
lp = pd.read_csv(REV / "pendulum" / "pendulum_lower_limb.csv")
for k in ["pelvis_rot", "hip_joint", "thigh_in_space"]: lp[k] = 100 * lp[f"bs_{k}_mm"] / lp.bs_len_mm
pm = lp.groupby("subject").median(numeric_only=True)
fig = plt.figure(figsize=(7.2, 5.2))
gs = GridSpec(2, 2, wspace=0.34, hspace=0.62, left=0.09, right=0.98, top=0.94, bottom=0.1)
ax = fig.add_subplot(gs[0, 0])
band(ax, c, "grip_x", INK, "Grip (mid-wrist)"); band(ax, c, "hub_x", COL["Lead shoulder"], "Shoulder midpoint (hub)")
ax.axhline(0, color=MUTED, lw=0.6); phase_axis(ax); cf.tidy(ax)
ax.set_ylabel("Displacement toward target (mm)")
ax.legend(loc="lower right", frameon=False, handlelength=1.6)
ax.text(0.03, 0.97, f"hub at top: {-pm.hub_top_mm.mean():.0f} ± {pm.hub_top_mm.std():.0f} mm toward backswing\n"
        f"= {pm.hub_top_ratio_pct.mean():.0f}% ± {pm.hub_top_ratio_pct.std():.0f}% of grip displacement\n"
        f"same direction as grip in {(pm.hub_top_ratio_pct > 0).sum()} of {len(pm)} players",
        transform=ax.transAxes, va="top", fontsize=6.5, color=INK, zorder=5,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
cf.title(ax, "a", "The hub of the pendulum is not fixed")
ax = fig.add_subplot(gs[0, 1])
band(ax, c, "radius_change", COL["Lead shoulder"], "to moving shoulder midpoint")
band(ax, c, "fixed_radius_change", MUTED, "to shoulder midpoint fixed at address", lw=1.1)
ax.axhline(0, color=MUTED, lw=0.6); phase_axis(ax); cf.tidy(ax)
ax.set_ylabel("Change in hub–grip distance (mm)")
ax.legend(loc="lower left", frameon=False, handlelength=1.6)
ax.text(0.03, 0.97, f"mean hub–grip distance {pm.radius_mean_mm.mean():.0f} mm; "
        f"range TA→impact {pm.radius_range_mm.mean():.1f} mm", transform=ax.transAxes, va="top", fontsize=6.5, color=INK,
        zorder=5, bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
ax.set_ylim(top=17.5)
cf.title(ax, "b", "…but the arm–hand radius stays constant")
ax = fig.add_subplot(gs[1, 0])
band(ax, c, "yaw_pelvis", COL["Pelvis"], "Pelvis"); band(ax, c, "yaw_thigh", "#7f97c7", "Thighs (mean)")
band(ax, c, "yaw_shank", MUTED, "Shanks (mean)")
ax.axhline(0, color=MUTED, lw=0.6); phase_axis(ax); cf.tidy(ax)
ax.set_ylabel("Rotation about vertical (°, backswing +)")
ax.legend(loc="upper right", frameon=False, handlelength=1.6)
cf.title(ax, "c", "Pelvis turns on the hips, legs turn less")
ax = fig.add_subplot(gs[1, 1])
o = pm.sort_values("pelvis_rot")
yy = np.arange(len(o))
ax.barh(yy, o.hip_joint, color=COL["Pelvis"], height=0.72, edgecolor="white", lw=0.6, label="Hip joints (pelvis on thighs)")
ax.barh(yy, o.thigh_in_space, left=o.hip_joint, color="#9fb3d9", height=0.72, edgecolor="white", lw=0.6, label="Thighs rotating in space")
ax.set_yticks(yy, o.index); ax.tick_params(axis="y", length=0, labelsize=6.5)
ax.set_xlabel("Share of backswing length (%)"); cf.tidy(ax, grid="x")
ax.legend(loc="lower right", frameon=False, handlelength=1.2,
          title=f"mean: hip joints {pm.hip_joint.mean():.0f}%, thighs {pm.thigh_in_space.mean():.0f}%",
          title_fontsize=6.5, alignment="left")
ax.set_ylim(-0.7, len(o) - 0.3)
cf.title(ax, "d", "Split of the pelvis-rotation contribution")
fig.savefig(OUT / "Figure5_pendulum_geometry.png", dpi=600); fig.savefig(OUT / "Figure5_pendulum_geometry.pdf"); plt.close(fig)

# ---------------- Figure S1: sensitivity of the three main results
def load(p): return json.load(open(Path(p) / "chain_results.json"))
scen = [("Main analysis (per-sensor gap filling)", A.main_res.resolve()),
        ("Original gap filling (any sensor)", SENS / "gap_union/res"), ("No gap filling", SENS / "gap_none/res"),
        ("Impact −2 frames", SENS / "impact_m2/res"), ("Impact +2 frames", SENS / "impact_p2/res"),
        ("Low-pass 4 Hz", SENS / "lowpass_4/res"), ("Low-pass 8 Hz", SENS / "lowpass_8/res"),
        ("Spine segments merged", SENS / "spine_merged/res"),
        ("Excl. chain gap > 6 frames", SENS / "excl_gap_gt6/res"), ("Excl. impact–peak > 5 frames", SENS / "excl_impact_peak_gt5/res"),
        ("Excl. residual > 10%", SENS / "excl_resid_gt10/res"), ("Excl. trials 1–2", SENS / "excl_trials_1_2/res"),
        ("Excl. trials 1–3", SENS / "excl_trials_1_3/res"), ("Excl. P07", SENS / "excl_P07/res")]
R = [(n, load(p)) for n, p in scen]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.6), sharey=True, gridspec_kw=dict(wspace=0.16, left=0.30, right=0.98, top=0.9, bottom=0.2))
yy = np.arange(len(R))[::-1]
for i, (n, r) in enumerate(R):
    b = r["Q1"]["backswing"]
    axs[0].errorbar(b["body"]["mean"], yy[i], xerr=b["body"]["sd"], fmt="o", color=COL["Pelvis"], ms=4, lw=0.9, capsize=0)
    axs[0].errorbar(b["arms"]["mean"], yy[i], xerr=b["arms"]["sd"], fmt="D", color=COL["Lead shoulder"], mfc="white", ms=3.5, lw=0.9, capsize=0)
    a = r["Q2"]["amplitude_vs_timing"]["amplitude"]
    axs[1].plot([a["lo"] * 100, a["hi"] * 100], [yy[i]] * 2, color=COL["Pelvis"], lw=1.0); axs[1].scatter(a["share"] * 100, yy[i], s=16, color=COL["Pelvis"], zorder=3)
    f = {q["x"]: q for q in r["Q3"]["face_assoc"]}["roll_trail_total_imp_deg"]
    axs[2].plot([f["lo"], f["hi"]], [yy[i]] * 2, color=COL["Trail elbow/forearm"], lw=1.0)
    axs[2].scatter(f["r"], yy[i], s=16, zorder=3, color=COL["Trail elbow/forearm"] if (f["q"] < .05 and f.get("robust")) else "white", edgecolor=COL["Trail elbow/forearm"])
    axs[0].text(-0.02, yy[i], f'n = {r["n_putts"]}', transform=axs[0].get_yaxis_transform(), ha="right", va="center", fontsize=6, color=MUTED)
axs[0].set_yticks(yy, [n for n, _ in R]); axs[0].tick_params(axis="y", length=0, pad=30)
for a_ in axs: cf.tidy(a_, grid="x"); a_.axhspan(yy[0] - 0.5, yy[0] + 0.5, color=GRID, alpha=0.5, lw=0, zorder=0)
axs[0].set_xlabel("Share of backswing length (%)"); axs[0].set_xlim(-5, 110)
axs[0].legend([Line2D([], [], marker="o", ls="", color=COL["Pelvis"]), Line2D([], [], marker="D", ls="", color=COL["Lead shoulder"], mfc="white")],
              ["Body (mean ± SD)", "Arms (mean ± SD)"], loc="lower left", bbox_to_anchor=(0.02, 0.02), bbox_transform=fig.transFigure, ncol=1, frameon=False,
              handletextpad=0.3, columnspacing=1.0)
cf.title(axs[0], "a", "Body and arm shares")
axs[1].set_xlabel("Share of within-player variance\nin ln(mean downswing grip speed) (%)"); axs[1].axvline(100, color=MUTED, lw=0.6, ls=(0, (2, 2)))
cf.title(axs[1], "b", "Amplitude share of grip speed")
axs[2].set_xlabel("$r_{rm}$ with face angle [95% CI]"); axs[2].axvline(0, color=MUTED, lw=0.6)
cf.title(axs[2], "c", "Trail roll vs face angle")
fig.savefig(OUT / "FigureS1_sensitivity.png", dpi=600); plt.close(fig)

# ---------------- Figure S2: forearm roll in space vs pronation-supination
cw = pd.read_csv(PROC / "curves_wide.csv" if (PROC / "curves_wide.csv").exists() else PROC / "curves_wide.csv.gz")
_, ok, _ = mr.load(PROC); cw = cw[cw.file.isin(ok.file)]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True, gridspec_kw=dict(wspace=0.1, left=0.08, right=0.98, top=0.86, bottom=0.18))
for ax, side, lab in zip(axs, ("lead", "trail"), ("(a)  Lead (left) forearm", "(b)  Trail (right) forearm")):
    band(ax, cw, f"forearm_roll_{side}", COL["Trail elbow/forearm"] if side == "trail" else COL["Lead elbow/forearm"], "Forearm roll in space")
    band(ax, cw, f"pronsup_{side}", MUTED, "Pronation–supination (forearm vs upper arm)", lw=1.1)
    ax.axhline(0, color=MUTED, lw=0.6); phase_axis(ax); cf.tidy(ax); ax.set_title(lab, loc="left", pad=6, color=INK)
axs[0].set_ylabel("Change from address (°, face-opening +)"); axs[0].legend(loc="lower left", frameon=False, fontsize=6.5)
fig.savefig(OUT / "FigureS2_forearm_roll.png", dpi=600); plt.close(fig)

# ---------------- Figure S3: trial-order trends
d = ok.merge(pd.read_csv(PROC / "chain_features.csv")[["file", "bs_total_mm", "bs_pelvis_trans_mm", "bs_pelvis_rot_mm", "bs_trunk_mm", "bs_girdle_mm"]], on="file")
d["body_share"] = 100 * (d.bs_pelvis_trans_mm + d.bs_pelvis_rot_mm + d.bs_trunk_mm + d.bs_girdle_mm) / d.bs_total_mm
items = [("backswing_len_mm", "Backswing length (mm)"), ("t_backswing_s", "Backswing time (s)"), ("body_share", "Body share (%)"),
         ("t_downswing_s", "Downswing time (s)"), ("head_speed", "Club-head speed (mph)"), ("face_deg", "Face angle (°)")]
fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.0), gridspec_kw=dict(wspace=0.42, hspace=0.6, left=0.08, right=0.98, top=0.93, bottom=0.1))
for k, (ax, (v, lab)) in enumerate(zip(axs.flat, items)):
    dv = d[v] - d.groupby("subject")[v].transform("mean"); g = dv.groupby(d.trial)
    m, se = g.mean(), g.std() / np.sqrt(g.size())
    ax.fill_between(m.index, m - se, m + se, color=COL["Pelvis"], alpha=0.18, lw=0); ax.plot(m.index, m, "o-", color=COL["Pelvis"], ms=3, lw=1.1)
    r = mr.rmcorr(d, "trial", v); ax.axhline(0, color=MUTED, lw=0.6)
    lo_, hi_ = ax.get_ylim(); ax.set_ylim(lo_, hi_ + 0.25 * (hi_ - lo_))      # room for the r_rm label
    ax.text(0.03, 0.97, f"$r_{{rm}}$ = {cf.num(r['r'])}, p = {r['p']:.3f}".replace("p = 0.000", "p < 0.001"), transform=ax.transAxes, va="top", fontsize=6.5, color=INK, zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    cf.title(ax, "abcdef"[k], lab); ax.set_xticks(range(1, 11)); cf.tidy(ax)
for ax in axs[1]: ax.set_xlabel("Trial number")
axs[0, 0].set_ylabel("Deviation from own mean"); axs[1, 0].set_ylabel("Deviation from own mean")
fig.savefig(OUT / "FigureS3_trial_order.png", dpi=600); plt.close(fig)

# ---------------- Figure S4: data quality (missing frames per sensor, impact vs velocity peak)
lost = pd.read_csv(REV / "data_quality" / "data_quality.csv")
SN = ["Hips", "Spine2", "LeftArm", "RightArm", "LeftForeArm", "RightForeArm", "LeftUpLeg", "RightUpLeg", "LeftLeg", "RightLeg"]
NAMES = ["Pelvis", "Upper thorax", "L upper arm", "R upper arm", "L forearm", "R forearm", "L thigh", "R thigh", "L shank", "R shank"]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw=dict(wspace=0.45, left=0.12, right=0.98, top=0.86, bottom=0.2, width_ratios=[1.3, 1, 1]))
ax = axs[0]; yy = np.arange(len(SN))[::-1]
ax.boxplot([lost[f"{s}_lost_pct"] for s in SN], positions=yy, vert=False, widths=0.55, showfliers=False,
           medianprops=dict(color=COL["Pelvis"], lw=1.2), boxprops=dict(color=MUTED, lw=0.7), whiskerprops=dict(color=MUTED, lw=0.7), capprops=dict(color=MUTED, lw=0.7))
ax.set_yticks(yy, NAMES); ax.tick_params(axis="y", length=0); ax.set_xlabel("Frames lost in stroke window (%)"); cf.tidy(ax, grid="x")
cf.title(ax, "a", "Missing frames per sensor")
ax = axs[1]; mr_ = lost["chain_single_maxgap"]
ax.hist(mr_, bins=np.arange(0, 42, 2), color=COL["Pelvis"], edgecolor="white", lw=0.5); ax.set_xlabel("Longest single-sensor gap (frames)")
ax.set_ylabel("Putts"); cf.tidy(ax); cf.title(ax, "b", "Longest gap, chain sensors")
ev = pd.read_csv(PROC / "events.csv"); ev = ev[ev.file.isin(ok.file)]
ax = axs[2]; ax.hist(ev.impact - ev.peak_v, bins=np.arange(-20, 12, 1.5), color=MUTED, edgecolor="white", lw=0.5)
ax.axvline(0, color=INK, lw=0.7); ax.set_xlabel("Impact − velocity peak (frames)"); cf.tidy(ax); cf.title(ax, "c", "Impact timing")
fig.savefig(OUT / "FigureS4_data_quality.png", dpi=600); plt.close(fig)
print("ok", sorted(p.name for p in OUT.iterdir()))
