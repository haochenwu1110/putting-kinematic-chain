# -*- coding: utf-8 -*-
"""Figures 2–4 of the paper (kinematic-chain decomposition results). 论文图 2–4 的出版版本。

用法：python chain_figures.py --dir <processed 文件夹> --res <chain_results 输出文件夹> --out <出图文件夹> [--font "Times New Roman"]
需要先运行 chain_results.py（读取其 chain_results.json 中的 q 值）。
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import make_results as mr
import chain_results as cr

# 分组配色：身体 = 蓝色系，肩关节 = 橙色系，肘与前臂 = 紫色系（已做色觉校验）
SRC = ["Pelvis", "Trunk", "Lead shoulder", "Trail shoulder", "Lead elbow/forearm", "Trail elbow/forearm"]
COL = dict(zip(SRC, ["#2a5db0", "#3d9be9", "#d65a24", "#e8a30d", "#7550c4", "#ad7ff0"]))
FINE = dict(cr.FIG)                     # 名称 → 细分组列表
INK, MUTED, GRID, PALE = "#1b1b1b", "#5c5b57", "#e4e3df", "#c9c8c3"
DPI = 600


def style(font):
    plt.rcParams.update({
        "font.family": font, "font.size": 7.5, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "mathtext.fontset": "custom", "mathtext.rm": font, "mathtext.it": f"{font}:italic"})


def tidy(ax, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.6)
        ax.set_axisbelow(True)


def title(ax, letter, text):
    ax.set_title(f"({letter})  {text}", loc="left", pad=6, color=INK)


def centered(d, cols):
    return d[cols] - d.groupby("subject")[cols].transform("mean")


def rm_label(r):
    return f"$r_{{rm}}$ = {num(r['r'])} [{num(r['lo'])}, {num(r['hi'])}]"


def num(v, fmt=".2f"):
    """负号用 Unicode 减号。"""
    return format(v, fmt).replace("-", "−")


def share_label(a):
    return f"{num(a['share'] * 100, '.0f')}% [{num(a['lo'] * 100, '.0f')}, {num(a['hi'] * 100, '.0f')}]"


def scatter_rm(ax, d, x, y, xscale=1.0, yscale=1.0, color=INK):
    c = centered(d, [x, y]).dropna()
    r = mr.rmcorr(d, x, y)
    ax.scatter(c[x] * xscale, c[y] * yscale, s=9, color=color, alpha=0.45, lw=0, zorder=2)
    xs = np.linspace(c[x].min(), c[x].max(), 50)
    ax.plot(xs * xscale, r["slope"] * xs * yscale, color=color, lw=1.6, zorder=3)
    ax.axhline(0, color=PALE, lw=0.6, zorder=0)
    ax.axvline(0, color=PALE, lw=0.6, zorder=0)
    tidy(ax, grid=None)
    return r


def source_legend(fig, y=0.0, ncol=6):
    h = [Patch(color=COL[s]) for s in SRC]
    fig.legend(h, SRC, loc="lower center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, y),
               handlelength=1.2, columnspacing=1.4)


# ---------------------------------------------------------------- 图 3
def fig3(d, cc, out):
    sh = cr.shares(d, "bs_", "_mm", "bs_total_mm")
    m = sh.groupby("subject").median(numeric_only=True)
    m = m.sort_values("body")
    mean = m.mean()

    fig = plt.figure(figsize=(7.2, 5.0))
    gs = GridSpec(2, 2, width_ratios=[1.0, 1.05], height_ratios=[1, 1.05], wspace=0.32, hspace=0.55,
                  left=0.11, right=0.98, top=0.94, bottom=0.13)

    # (a) 各人上杆长度的来源
    ax = fig.add_subplot(gs[:, 0])
    rows = [("Mean", mean)] + [(s, r) for s, r in m.iloc[::-1].iterrows()]
    ypos = [0] + [i + 1.6 for i in range(len(m))]
    for yy, (name, r) in zip(ypos, rows):
        lp = ln = 0.0
        for s in SRC:
            v = r[s]
            left = lp if v >= 0 else ln
            ax.barh(yy, v, left=left, color=COL[s], height=0.74, edgecolor="white", lw=0.6)
            if v >= 0:
                lp += v
            else:
                ln += v
        ax.text(max(lp, 100) + 3, yy, f"{r['body']:.0f}%", va="center", fontsize=6.5,
                color=INK if name == "Mean" else MUTED, fontweight="bold" if name == "Mean" else "normal")
    ax.set_yticks(ypos, [n for n, _ in rows])
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.invert_yaxis()
    ax.axvline(0, color=MUTED, lw=0.7)
    ax.axvline(100, color=MUTED, lw=0.6, ls=(0, (2, 2)))
    ax.axhline(0.8, color=GRID, lw=0.8)
    ax.set_xlim(-60, 160)
    ax.set_xticks([-50, 0, 50, 100, 150])
    ax.set_xlabel("Share of backswing length (%)")
    ax.text(1.0, 1.005, "body share", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color=MUTED)
    tidy(ax, grid="x")
    ax.tick_params(axis="y", length=0)
    title(ax, "a", "Who produces the backswing")

    # (b) 骨盆与肩部一起转
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(d.top_d_pelvis_turn_closed, d.top_d_shoulder_turn_closed, s=6, color=PALE, lw=0, zorder=1)
    med = d.groupby("subject")[["top_d_pelvis_turn_closed", "top_d_shoulder_turn_closed"]].median()
    ax.scatter(med.top_d_pelvis_turn_closed, med.top_d_shoulder_turn_closed, s=28, color=COL["Pelvis"],
               edgecolor="white", lw=0.7, zorder=3)
    lim = (-2, 16)
    ax.plot(lim, lim, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=2)
    ax.set_xlim(lim); ax.set_ylim(lim)
    xf = d.groupby("subject").top_d_xfactor.median()
    ax.text(0.04, 0.96, f"X-factor at top: {xf.mean():.1f}° ± {xf.std():.1f}°\n(mean ± SD of player medians)",
            transform=ax.transAxes, va="top", fontsize=6.5, color=INK)
    ax.legend([Line2D([], [], color=MUTED, lw=0.8, ls=(0, (3, 2))),
               Line2D([], [], marker="o", ls="", color=COL["Pelvis"], markersize=5),
               Line2D([], [], marker="o", ls="", color=PALE, markersize=3)],
              ["shoulder = pelvis", "player median", "single putt"], loc="lower right", frameon=False,
              handlelength=1.6, borderaxespad=0.2)
    ax.set_xlabel("Pelvis turn at top (°)")
    ax.set_ylabel("Shoulder turn at top (°)")
    tidy(ax, grid=None)
    title(ax, "b", "Pelvis and thorax turn together")

    # (c) 握把速度的来源随时程
    ax = fig.add_subplot(gs[1, 1])
    cols = [c for c in cc.columns if c.startswith(("back_", "down_", "follow_"))]
    x = np.arange(len(cols))

    def curve(vars_):
        sub = cc[cc["var"].isin(vars_)].groupby(["file", "subject"])[cols].sum()
        return sub.groupby("subject").mean().mean().values
    pos_base = np.zeros(len(x)); neg_base = np.zeros(len(x))
    for s in SRC:
        y = curve([f"cv_{g}" for g in FINE[s]])
        yp, yn = np.clip(y, 0, None), np.clip(y, None, 0)
        ax.fill_between(x, pos_base, pos_base + yp, color=COL[s], lw=0)
        ax.fill_between(x, neg_base, neg_base + yn, color=COL[s], lw=0)
        pos_base += yp; neg_base += yn
    ax.plot(x, curve(["cv_measured"]), color=INK, lw=1.3, label="Grip, measured")
    ax.axhline(0, color=MUTED, lw=0.6)
    for v, lab in ((101, "Top"), (202, "Impact")):
        ax.axvline(v, color=MUTED, lw=0.6, ls=(0, (2, 2)))
    ax.set_xticks([0, 101, 202, 302], ["TA", "Top", "Impact", "Finish"])
    ax.set_xlim(0, 302)
    ax.set_ylabel("Grip velocity toward target (m/s)")
    ax.legend(loc="upper left", frameon=False, handlelength=1.6)
    tidy(ax)
    title(ax, "c", "Contributions through the stroke")

    source_legend(fig, y=0.0, ncol=6)
    fig.savefig(out / "Figure2_who_swings.png", dpi=DPI)
    fig.savefig(out / "Figure2_who_swings.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- 图 4
def forest_shares(ax, attrs, labels, markers, sources, colors, xlab):
    n = len(attrs)
    offs = np.linspace(-0.17, 0.17, n) if n > 1 else [0.0]
    y = np.arange(len(sources))[::-1]
    for (attr, mk), off in zip(zip(attrs, markers), offs):
        for yi, s in zip(y, sources):
            a = attr[s]
            ax.plot([a["lo"] * 100, a["hi"] * 100], [yi + off] * 2, color=colors[s], lw=1.1, solid_capstyle="round")
            ax.scatter(a["share"] * 100, yi + off, marker=mk, s=26, zorder=3,
                       color=colors[s] if mk == "o" else "white", edgecolor=colors[s], lw=1.1)
    ax.set_yticks(y, sources)
    ax.axvline(0, color=MUTED, lw=0.7)
    ax.set_xlabel(xlab)
    tidy(ax, grid="x")
    ax.tick_params(axis="y", length=0)
    ax.legend([Line2D([], [], marker=m, ls="", color=MUTED, markerfacecolor=MUTED if m == "o" else "white",
                      markersize=4.5) for m in markers], labels, loc="upper center", frameon=False, handletextpad=0.3,
              ncol=len(markers), bbox_to_anchor=(0.45, -0.2), columnspacing=1.0)


def fig4(d, R, out):
    fig = plt.figure(figsize=(7.2, 5.2))
    gs = GridSpec(2, 3, width_ratios=[1.25, 1, 1], height_ratios=[1, 0.9], wspace=0.45, hspace=0.72,
                  left=0.13, right=0.98, top=0.94, bottom=0.1)

    # (a) 摆幅的逐杆变化由谁提供
    ax = fig.add_subplot(gs[0, 0])
    forest_shares(ax, [R["Q2"]["attr_backswing"], R["Q2"]["attr_impact_velocity"]],
                  ["Backswing length", "Grip velocity at impact"], ["o", "D"], SRC, COL,
                  "Share of within-player variance (%)")
    title(ax, "a", "Who adds the extra amplitude")

    # (b)(c) 摆幅 vs 下杆时间
    dd = d[(d.ds_total_mm > 0) & (d.t_downswing_s > 0)]
    amp = R["Q2"]["amplitude_vs_timing"]
    axb = fig.add_subplot(gs[0, 1])
    r = scatter_rm(axb, dd, "ds_total_mm", "head_speed", color=COL["Pelvis"])
    axb.set_xlabel("Δ downswing amplitude (mm)")
    axb.set_ylabel("Δ club-head speed (mph)")
    axb.text(0.04, 0.97, f"{rm_label(r)}\nshare of speed variance\n{share_label(amp['amplitude'])}",
             transform=axb.transAxes, va="top", fontsize=6.3, color=INK, zorder=5,
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    title(axb, "b", "Amplitude sets speed")
    axc = fig.add_subplot(gs[0, 2], sharey=axb)
    r = scatter_rm(axc, dd, "t_downswing_s", "head_speed", xscale=1000, color=MUTED)
    axc.set_xlabel("Δ downswing time (ms)")
    axc.tick_params(labelleft=False)
    axc.text(0.04, 0.97, f"{rm_label(r)}\nshare of speed variance\n{share_label(amp['timing'])}",
             transform=axc.transAxes, va="top", fontsize=6.3, color=INK, zorder=5,
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    title(axc, "c", "Downswing time does not")

    # (d) 上杆乱、下杆稳：每人的逐杆变异系数
    ax = fig.add_subplot(gs[1, :])
    g = d.groupby("subject")
    cv = pd.DataFrame({"back": g.t_backswing_s.std() / g.t_backswing_s.mean() * 100,
                       "down": g.t_downswing_s.std() / g.t_downswing_s.mean() * 100}).sort_values("back")
    xs = np.arange(len(cv))
    for xi, (_, r) in zip(xs, cv.iterrows()):
        ax.plot([xi, xi], [r.down, r.back], color=PALE, lw=1.4, zorder=1)
    ax.scatter(xs, cv.back, s=26, color="white", edgecolor=COL["Trail shoulder"], lw=1.3, zorder=3, label="Backswing time")
    ax.scatter(xs, cv.down, s=26, color=COL["Pelvis"], edgecolor="white", lw=0.6, zorder=3, label="Downswing time")
    for col_, c_, lab in (("back", COL["Trail shoulder"], "median"), ("down", COL["Pelvis"], "median")):
        mv = cv[col_].median()
        ax.axhline(mv, color=c_, lw=0.8, ls=(0, (3, 2)), zorder=0)
        ax.text(len(cv) + 1.9, mv + 1.2, f"{lab} {mv:.0f}%" if mv >= 10 else f"{lab} {mv:.1f}%",
                va="bottom", ha="right", fontsize=6.5, color=c_)
    icc_b = mr.icc1(d, "t_backswing_s")[0]; icc_d = mr.icc1(d, "t_downswing_s")[0]
    ax.text(0.01, 0.97, f"ICC: backswing time {icc_b:.2f}, downswing time {icc_d:.2f}",
            transform=ax.transAxes, va="top", fontsize=6.5, color=INK)
    ax.set_xticks(xs, cv.index, rotation=45, ha="right")
    ax.set_xlim(-0.7, len(cv) + 2.2)
    ax.set_ylabel("Putt-to-putt CV (%)")
    ax.legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    tidy(ax)
    ax.tick_params(axis="x", length=0)
    title(ax, "d", "Timing: backswing varies, downswing is a personal constant")

    fig.savefig(out / "Figure3_speed_sources.png", dpi=DPI)
    fig.savefig(out / "Figure3_speed_sources.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- 图 5
def fig5(d, R, out):
    for tag in ("lead", "trail"):
        d[f"roll_{tag}_trunkgirdle_imp_deg"] = d[f"roll_{tag}_trunk_imp_deg"] + d[f"roll_{tag}_girdle_imp_deg"]
    qmap = {r["x"]: r for r in R["Q3"]["face_assoc"]}
    fig = plt.figure(figsize=(7.2, 3.1))
    gs = GridSpec(1, 3, width_ratios=[1.05, 1.15, 0.95], wspace=0.55, left=0.12, right=0.98, top=0.88, bottom=0.25)

    # (a) 前臂扭转的逐杆变化由谁产生
    names = ["Pelvis", "Trunk", "Shoulder", "Elbow/forearm"]
    col = {"Pelvis": COL["Pelvis"], "Trunk": COL["Trunk"], "Shoulder": COL["Lead shoulder"],
           "Elbow/forearm": COL["Lead elbow/forearm"]}
    ax = fig.add_subplot(gs[0])
    forest_shares(ax, [R["Q3"]["attr_roll_imp"]["lead"], R["Q3"]["attr_roll_imp"]["trail"]],
                  ["Lead forearm", "Trail forearm"], ["o", "D"], names, col, "Share of within-player variance (%)")
    title(ax, "a", "Who twists the forearm")

    # (b) 与杆面角的个人内相关
    ax = fig.add_subplot(gs[1])
    items = [("pelvis_rot", "Pelvis"), ("trunkgirdle", "Trunk"), ("shoulder", "Shoulder"),
             ("elbow", "Elbow/forearm"), ("total", "Total twist")]
    rows = []
    for tag in ("lead", "trail"):
        for g, lab in items:
            x = f"roll_{tag}_{g}_imp_deg"
            r = mr.rmcorr(d, x, "face_deg")
            q = qmap.get(x, {}).get("q", np.nan)
            rob = qmap.get(x, {}).get("robust", False)
            rows.append((tag, lab, r, q, rob))
    y = []
    yy = 0.0
    for i, (tag, lab, r, q, rob) in enumerate(rows):
        if i == len(items):
            yy += 0.8
        y.append(yy); yy += 1
    y = np.array(y)
    for yi, (tag, lab, r, q, rob) in zip(y, rows):
        side_col = {"Shoulder": COL["Lead shoulder" if tag == "lead" else "Trail shoulder"],
                    "Elbow/forearm": COL["Lead elbow/forearm" if tag == "lead" else "Trail elbow/forearm"]}
        c = INK if lab == "Total twist" else side_col.get(lab, col.get(lab, INK))
        sig = (q < 0.05) and rob
        ax.plot([r["lo"], r["hi"]], [yi, yi], color=c, lw=1.1)
        ax.scatter(r["r"], yi, s=30, marker="o" if tag == "lead" else "D", zorder=3,
                   color=c if sig else "white", edgecolor=c, lw=1.2)
        if sig:
            ax.text(r["hi"] + 0.03, yi, f"{r['r']:.2f}", va="center", fontsize=6.3, color=INK)
    ax.set_yticks(y, [lab for _, lab, *_ in rows])
    ax.set_ylim(y[-1] + 0.6, y[0] - 1.4)
    ax.axvline(0, color=MUTED, lw=0.7)
    for yy0, tag in ((y[0], "Lead"), (y[len(items)], "Trail")):
        ax.text(-0.02, yy0 - 0.75, f"{tag} forearm", transform=ax.get_yaxis_transform(), ha="right",
                fontsize=6.8, fontweight="bold", color=INK)
    ax.set_xlim(-0.45, 0.6)
    ax.set_xlabel("$r_{rm}$ with face angle (95% CI)")
    ax.tick_params(axis="y", length=0)
    tidy(ax, grid="x")
    title(ax, "b", "Which twist goes with the face")
    ax.text(1.0, -0.24, "filled: FDR q < 0.05 and robust\ncircle: lead, diamond: trail", transform=ax.transAxes, ha="right",
            va="top", fontsize=6.2, color=MUTED)

    # (c) 后侧前臂总扭转与杆面角
    ax = fig.add_subplot(gs[2])
    r = scatter_rm(ax, d, "roll_trail_total_imp_deg", "face_deg", color=COL["Trail elbow/forearm"])
    ax.set_xlabel("Δ trail forearm twist at impact (°)")
    ax.set_ylabel("Δ face angle (°, open +)")
    ax.text(0.04, 0.97, rm_label(r), transform=ax.transAxes, va="top", fontsize=6.3, color=INK, zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    title(ax, "c", "Total trail twist")

    fig.savefig(out / "Figure4_face_sources.png", dpi=DPI)
    fig.savefig(out / "Figure4_face_sources.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--res", type=Path, required=True, help="chain_results.py 的输出文件夹")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--font", default="DejaVu Serif")
    ap.add_argument("--exclude", default="")
    a = ap.parse_args()
    mr.EXCLUDE_SUBJECTS[:] = [x for x in a.exclude.split(",") if x]
    a.out.mkdir(parents=True, exist_ok=True)
    style(a.font)
    _, d, _, cc = cr.load(a.dir)
    R = json.loads((a.res / "chain_results.json").read_text(encoding="utf-8"))
    fig3(d, cc, a.out)
    fig4(d, R, a.out)
    fig5(d, R, a.out)
    print("done:", a.out)


if __name__ == "__main__":
    main()
