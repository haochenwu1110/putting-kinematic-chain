# -*- coding: utf-8 -*-
"""每个 take 一张 QC 图：握把位移/速度 + 事件线，前臂滚转与肩部倾斜。白底。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import warnings

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", message="Glyph .* missing")
warnings.filterwarnings("ignore", message="findfont")

EV_STYLE = [("takeaway", "#1f77b4", "TA"), ("top", "#2ca02c", "TOP"),
            ("impact", "#d62728", "IMP"), ("finish", "#9467bd", "FIN")]


def qc_plot(res, fs, title, path):
    S, ev, addr = res["series"], res["events"], res["address"]
    t = np.arange(len(S["grip_target_mm"])) / fs
    lo = max(0, ev["takeaway"] - int(0.8 * fs)) / fs
    hi = min(len(t) - 1, ev["finish"] + int(0.8 * fs)) / fs
    fig, axs = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True, facecolor="white")
    ax = axs[0]
    ax.plot(t, S["grip_target_mm"] - ev["x_address_mm"], color="black", lw=1.4, label="grip → target (mm)")
    ax.plot(t, S["grip_away_mm"] - addr["grip_away_mm"], color="0.55", lw=1, label="grip → away (mm)")
    ax.plot(t, S["grip_up_mm"] - addr["grip_up_mm"], color="0.55", lw=1, ls="--", label="grip ↑ (mm)")
    ax.axvspan(ev["address_start"] / fs, ev["address_end"] / fs, color="0.9", label="address window")
    ax.set_ylabel("mm")
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax = axs[1]
    ax.plot(t, S["grip_v_target"], color="black", lw=1.4, label="grip v_target (m/s)")
    ax.plot(t, S["grip_speed"], color="0.55", lw=1, label="grip speed (m/s)")
    ax.set_ylabel("m/s")
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax = axs[2]
    ax.plot(t, S["forearm_roll_lead"], color="#d62728", lw=1.3, label="lead forearm roll Δ (+open)")
    ax.plot(t, S["forearm_roll_trail"], color="#ff9896", lw=1.0, label="trail forearm roll Δ")
    ax.plot(t, S["shoulder_tilt_leadup"] - addr["shoulder_tilt_leadup"], color="#1f77b4", lw=1.3,
            label="shoulder tilt Δ (+lead up)")
    ax.plot(t, S["shoulder_turn_closed"] - addr["shoulder_turn_closed"], color="#aec7e8", lw=1.0,
            label="shoulder turn Δ (+closed)")
    ax.set_ylabel("deg")
    ax.set_xlabel("time (s)")
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    for a in axs:
        for k, c, lab in EV_STYLE:
            a.axvline(ev[k] / fs, color=c, lw=0.9, ls=":")
        a.axhline(0, color="0.8", lw=0.6)
        a.set_xlim(lo, hi)
        a.spines[["top", "right"]].set_visible(False)
    for k, c, lab in EV_STYLE:
        axs[0].text(ev[k] / fs, 1.01, lab, color=c, fontsize=8, ha="center", transform=axs[0].get_xaxis_transform())
    flags = "; ".join(res["flags"]) or "OK"
    fig.suptitle(f"{title}    [{flags}]", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor="white")
    plt.close(fig)


def overview(curves, qc, var, n, path):
    """一张图看全体：每人一格，10 杆时间归一化曲线叠加；颜色 = QC 状态，线尾标杆次。"""
    c = curves[curves["var"] == var].merge(qc[["file", "status"]], on="file", how="left")
    cols = [k for k in c.columns if k.startswith(("back_", "down_", "follow_"))]
    subs = sorted(c.subject.unique())
    ncol = 5
    nrow = int(np.ceil(len(subs) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), sharex=True, facecolor="white", squeeze=False)
    x = np.arange(len(cols))
    colr = {"OK": "0.35", "WARN": "#1f77b4", "CHECK": "#d62728"}
    for ax in axs.flat:
        ax.set_visible(False)
    for ax, sub in zip(axs.flat, subs):
        ax.set_visible(True)
        for _, r in c[c.subject == sub].iterrows():
            y = r[cols].values.astype(float)
            ax.plot(x, y, color=colr.get(r.status, "k"), lw=1)
            ax.text(x[-1] + 3, y[-1], str(int(r.trial)), fontsize=7, va="center")
        for v in (n, 2 * n):
            ax.axvline(v, color="0.7", lw=0.8, ls=":")
        ax.axhline(0, color="0.85", lw=0.6)
        ax.set_title(sub, fontsize=10)
        ax.set_xticks([0, n, 2 * n, 3 * n - 1], ["TA", "TOP", "IMP", "FIN"], fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{var}   (灰 OK / 蓝 WARN / 红 CHECK；线尾数字 = 杆次)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=80, facecolor="white")
    plt.close(fig)
