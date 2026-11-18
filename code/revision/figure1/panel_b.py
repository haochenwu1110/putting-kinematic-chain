"""Figure 1(b) right: stroke phases of all retained putts, pixel-matched to the published Figure 1 (crop x0=3100, y0=110)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, str(Path(sys.argv[1]) / "code" / "analysis"))
import chain_figures as cf, make_results as mr
proc = Path(sys.argv[1]) / "data" / "processed"; out = Path(sys.argv[2]); caption = sys.argv[3]
cf.style("Liberation Serif")
DPI = 425.0; W, H = 1150, 1100
_, d, _ = mr.load(proc)
d = d.sort_values("t_backswing_s").reset_index(drop=True)
N = len(d)
fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
ax = fig.add_axes([215.5 / W, 1 - 761.5 / H, (1134 - 215.5) / W, (761.5 - 38.7) / H])
C = [cf.COL["Trunk"], cf.COL["Lead shoulder"], cf.COL["Trail elbow/forearm"]]
y = np.arange(N)
ax.barh(y, d.t_backswing_s, left=-(d.t_backswing_s + d.t_downswing_s), height=1.0, color=C[0], lw=0)
ax.barh(y, d.t_downswing_s, left=-d.t_downswing_s, height=1.0, color=C[1], lw=0)
ax.barh(y, d.t_follow_s, left=0, height=1.0, color=C[2], lw=0)
ax.axvline(0, color=cf.INK, lw=0.75, zorder=3)
ax.set_xlim(-(d.t_backswing_s + d.t_downswing_s).max(), 1.156); ax.set_ylim(-1, N)
ax.set_xticks([-3, -2, -1, 0, 1]); ax.set_yticks([0, N - 1], ["1", str(N)])
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: cf.num(v, ".0f")))
ax.set_xlabel("Time relative to impact (s)", fontsize=7.7, labelpad=3.2); ax.set_ylabel("Putts, sorted by backswing time", fontsize=7.7)
cf.tidy(ax, grid="x"); ax.yaxis.labelpad = 2.9
for x0, col, lab in zip([198, 466, 749], C, ["Backswing", "Downswing", "Follow-through"]):
    fig.add_artist(Rectangle((x0 / W, 1 - 986 / H), 45.5 / W, 32.5 / H, color=col, lw=0, transform=fig.transFigure))
    fig.text((x0 + 59) / W, 1 - 970 / H, lab, ha="left", va="center_baseline", fontsize=6.9, color=cf.INK)
fig.text(591 / W, 1 - 1055 / H, caption, ha="center", va="center", fontsize=8.3, color=cf.MUTED, style="italic")
fig.savefig(out, dpi=DPI, facecolor="white")
print(out, "N =", N, "xmin", round(-(d.t_backswing_s + d.t_downswing_s).max(), 3), "follow max", round(d.t_follow_s.max(), 3))
