"""Figure 1(e) data panels, pixel-matched to the published Figure 1 (crop at x0=2780, y0=1600)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, str(Path(sys.argv[1]) / "code" / "analysis"))
import chain_figures as cf, make_results as mr, chain_results as cr
proc = Path(sys.argv[1]) / "data" / "processed"; out = Path(sys.argv[2]); xvar_right = sys.argv[3] if len(sys.argv) > 3 else "backswing_len_mm"
cf.style("Liberation Serif")
DPI = 425.0; W, H = 1470, 920
_, d, _, _ = cr.load(proc)
d = d[d.head_speed.notna() & d.backswing_len_mm.notna()]
HL = [("P18", cf.COL["Pelvis"], "Player 1"), ("P19", cf.COL["Lead shoulder"], "Player 2"), ("P01", cf.COL["Lead elbow/forearm"], "Player 3")]
fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
def ax_px(x0, x1, y0, y1):  # pixel box (top-left origin) -> axes
    return fig.add_axes([x0 / W, 1 - y1 / H, (x1 - x0) / W, (y1 - y0) / H])
PALE = cf.PALE
# left: raw
ax = ax_px(166.5, 693, 100, 700)
hl = [p for p, _, _ in HL]
o = d[~d.subject.isin(hl)]
ax.scatter(o.backswing_len_mm, o.head_speed, s=6.5, color=PALE, lw=0, zorder=1)
for p, c, _ in HL:
    s = d[d.subject == p]
    ax.scatter(s.backswing_len_mm, s.head_speed, s=9.5, color=c, lw=0, zorder=3)
    b = np.polyfit(s.backswing_len_mm, s.head_speed, 1); xs = np.array([s.backswing_len_mm.min(), s.backswing_len_mm.max()])
    ax.plot(xs, np.polyval(b, xs), color=c, lw=0.9, zorder=2)
ax.set_yticks([2, 3, 4, 5]); ax.set_xticks([50, 100, 150])
ax.set_xlabel("Backswing length (mm)"); ax.set_ylabel("Club-head speed (mph)")
cf.tidy(ax, grid=None); ax.set_title("Raw: players mixed", loc="left", pad=3.6, color=cf.INK)
# right: within-player
ax2 = ax_px(912.5, 1437, 100, 700)
c = cf.centered(d, [xvar_right, "head_speed"])
oc = c[~d.subject.isin(hl)]
ax2.scatter(oc[xvar_right], oc.head_speed, s=6.5, color=PALE, lw=0, zorder=1)
for p, col, _ in HL:
    s = c[d.subject == p]; ax2.scatter(s[xvar_right], s.head_speed, s=9.5, color=col, lw=0, zorder=3)
r = mr.rmcorr(d, xvar_right, "head_speed")
xs = np.array([c[xvar_right].min(), c[xvar_right].max()])
ax2.plot(xs, r["slope"] * xs, color=cf.INK, lw=1.25, zorder=2)
ax2.axhline(0, color=PALE, lw=0.6, zorder=0); ax2.axvline(0, color=PALE, lw=0.6, zorder=0)
ax2.set_xticks([-50, 0, 50]); ax2.set_yticks(np.arange(-1.5, 1.51, 0.5))
ax2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: cf.num(v, ".1f")))
ax2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: cf.num(v, ".0f")))
ax2.set_xlabel("Δ backswing (mm)"); ax2.set_ylabel("Δ speed (mph)")
cf.tidy(ax2, grid=None); ax2.set_title("Within-player deviations", loc="left", pad=3.6, color=cf.INK)
lab = [l for _, _, l in HL] + [f"Other {d.subject.nunique() - 3} players"]
cols = [col for _, col, _ in HL] + [PALE]
for xm, col, l in zip([255, 515, 770, 1027], cols, lab):
    fig.add_artist(Line2D([xm / W], [1 - 882 / H], marker="o", ls="", color=col, markersize=4.3, transform=fig.transFigure))
    fig.text((xm + 49) / W, 1 - 882 / H, l, ha="left", va="center_baseline", fontsize=7, color=cf.INK)
fig.savefig(out, dpi=DPI, facecolor="white")
print(out, "n =", len(d), "r_rm slope", round(r["slope"], 4), "r", round(r["r"], 3))
