# -*- coding: utf-8 -*-
"""Section 5 results: statistics + figures (white background) + numbers JSON for the manuscript tables.

Inputs  (from putt_preproc output): features.csv, qc_report.csv, curves_wide.csv
Outputs: fig*.png, results.json, tables.json
Usage:  python make_results.py --dir <10.8_processed folder> --out <output folder> [--font "Times New Roman"]
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#d9d8d4"
EXCLUDE_SUBJECTS = []                     # R1: whole-session sensor/calibration anomaly（--exclude 指定）
GRADE_EN = {"大二年级": "Year 2", "大三年级": "Year 3"}
N_PERM = 2000
DPI = 600                                  # Bioengineering: figures ≥ 600 dpi
RNG = np.random.default_rng(2026)


# ---------------------------------------------------------------- statistics
def rmcorr(df, x, y, s="subject"):
    d = df[[s, x, y]].dropna()
    d = d.groupby(s).filter(lambda g: len(g) >= 3)
    cx = d[x] - d.groupby(s)[x].transform("mean")
    cy = d[y] - d.groupby(s)[y].transform("mean")
    N, k = len(d), d[s].nunique()
    sxy, ssx, ssy = (cx * cy).sum(), (cx ** 2).sum(), (cy ** 2).sum()
    r = sxy / np.sqrt(ssx * ssy)
    dof = N - k - 1
    t = r * np.sqrt(dof / (1 - r ** 2))
    p = 2 * stats.t.sf(abs(t), dof)
    z, se = np.arctanh(r), 1 / np.sqrt(dof - 1)
    lo, hi = np.tanh([z - 1.96 * se, z + 1.96 * se])
    return dict(r=r, lo=lo, hi=hi, p=p, dof=dof, N=N, k=k, slope=sxy / ssx)


def icc1(df, v, s="subject"):
    d = df[[s, v]].dropna()
    g = d.groupby(s)[v]
    n = g.size(); k = len(n); N = len(d)
    n0 = (N - (n ** 2).sum() / N) / (k - 1)
    msb = (n * (g.mean() - d[v].mean()) ** 2).sum() / (k - 1)
    msw = ((d[v] - g.transform("mean")) ** 2).sum() / (N - k)
    return (msb - msw) / (msb + (n0 - 1) * msw), np.sqrt(msw), g.mean().std()


def bh(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); q = np.empty(m); prev = 1.0
    for i in range(m - 1, -1, -1):
        prev = min(prev, p[o[i]] * m / (i + 1)); q[o[i]] = prev
    return q


def hedges_g(a, b):
    n1, n2 = len(a), len(b)
    s = np.sqrt(((n1 - 1) * a.var() + (n2 - 1) * b.var()) / (n1 + n2 - 2))
    return (b.mean() - a.mean()) / s * (1 - 3 / (4 * (n1 + n2) - 9))


# ---------------------------------------------------------------- data
def load(dirp):
    f = pd.read_csv(dirp / "features.csv")
    q = pd.read_csv(dirp / "qc_report.csv")
    d = q[["file", "subject", "trial", "status", "flags"]].merge(f, on=["file", "subject", "trial"], how="left")

    def reason(r):
        fl = r["flags"] if isinstance(r["flags"], str) else ""
        toks = fl.split("; ")
        if r["subject"] in EXCLUDE_SUBJECTS: return "R1"
        if r["status"] == "FAIL" or any(t.startswith("impact_fallback") for t in toks): return "R2"
        if any(t.startswith("key_sensor_lost") for t in toks): return "R3"
        if any(t.startswith("gap_") for t in toks): return "R4"
        return ""
    d["exclude"] = d.apply(reason, axis=1)
    clean = d[d.exclude == ""].copy()
    clean["grade_en"] = clean["grade"].map(GRADE_EN)
    curves = pd.read_csv(dirp / "curves_wide.csv" if (dirp / "curves_wide.csv").exists() else dirp / "curves_wide.csv.gz")
    return d, clean, curves


def robustness(clean, ranked, x, y):
    """秩相关显著 + 2% 缩尾后仍显著 + 逐一剔除受试者后符号不变 → robust。"""
    rk = rmcorr(ranked, x, y)
    w = clean.copy(); lo, hi = w[x].quantile([0.02, 0.98]); w[x] = w[x].clip(lo, hi)
    lw, hw = w[y].quantile([0.02, 0.98]); w[y] = w[y].clip(lw, hw)
    wr = rmcorr(w, x, y)
    vals = [rmcorr(clean[clean.subject != s], x, y)["r"] for s in clean.subject.unique()]
    same = (min(vals) > 0) or (max(vals) < 0)
    return dict(rank_p=rk["p"], wins_r=wr["r"], wins_p=wr["p"],
                robust=bool(rk["p"] < 0.05 and wr["p"] < 0.05 and same and np.sign(rk["r"]) == np.sign(wr["r"])))


# ---------------------------------------------------------------- SPM-style within-subject regression
def spm_regress(curves, clean, var, y):
    cols = [c for c in curves.columns if c.startswith(("back_", "down_", "follow_"))]
    c = curves[curves["var"] == var].merge(clean[["file", "subject", y]], on=["file", "subject"])
    Y = c[cols].values.astype(float)
    Yc = Y - c.groupby("subject")[cols].transform("mean").values
    yc = (c[y] - c.groupby("subject")[y].transform("mean")).values
    subj = c["subject"].values
    N, k = len(yc), len(np.unique(subj)); dof = N - k - 1
    thr = stats.t.ppf(0.975, dof)

    def tmap(yv):
        r = (Yc * yv[:, None]).sum(0) / np.sqrt((Yc ** 2).sum(0) * (yv ** 2).sum())
        return r, r * np.sqrt(dof / (1 - r ** 2))

    def clusters(t):
        out, i, n = [], 0, len(t)
        while i < n:
            if abs(t[i]) > thr:
                sgn, j = np.sign(t[i]), i
                while j < n and abs(t[j]) > thr and np.sign(t[j]) == sgn: j += 1
                out.append((i, j - 1, np.abs(t[i:j]).sum())); i = j
            else:
                i += 1
        return out

    r, t = tmap(yc)
    sdv = Yc.std(0); bad = sdv < 0.1 * np.median(sdv)      # e.g. grip displacement ≡ 0 at impact by definition
    r[bad] = np.nan; t[bad] = 0.0
    cl = clusters(t)
    groups = [np.where(subj == s)[0] for s in np.unique(subj)]
    mx = np.empty(N_PERM)
    for b in range(N_PERM):
        yp = yc.copy()
        for g in groups: yp[g] = yc[RNG.permutation(g)]
        mx[b] = max([cc[2] for cc in clusters(tmap(yp)[1])], default=0)
    sig = [dict(start=int(a), end=int(e), p=float((mx >= m).mean()), r_mean=float(r[a:e + 1].mean()))
           for a, e, m in cl if (mx >= m).mean() < 0.05]
    # within-subject split for display: trials above vs below own median of y
    med = c.groupby("subject")[y].transform("median")
    hi_m = Y[(c[y] > med).values].mean(0); lo_m = Y[(c[y] <= med).values].mean(0)
    return dict(r=r, thr_r=thr / np.sqrt(dof + thr ** 2), sig=sig, hi=hi_m, lo=lo_m, N=N, k=k)


def phase_label(i, n=101):
    ph = ["backswing", "downswing", "follow-through"][i // n]
    return f"{ph} {i % n}%"


# ---------------------------------------------------------------- figures
def style(font):
    plt.rcParams.update({"font.family": font, "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.8,
                         "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})


def clean_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)


def fig_validation(clean, out):
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.0))
    for ax, (x, lab) in zip(axs, [("grip_speed_impact", "Δ grip speed at impact (m/s)"),
                                   ("backswing_len_mm", "Δ backswing length (mm)")]):
        d = clean[["subject", x, "head_speed"]].dropna()
        cx = d[x] - d.groupby("subject")[x].transform("mean")
        cy = d["head_speed"] - d.groupby("subject")["head_speed"].transform("mean")
        r = rmcorr(clean, x, "head_speed")
        ax.scatter(cx, cy, s=10, color=BLUE, alpha=0.55, lw=0)
        xs = np.linspace(cx.min(), cx.max(), 10); ax.plot(xs, r["slope"] * xs, color=INK, lw=1.2)
        ax.set_xlabel(lab); ax.set_ylabel("Δ club-head speed")
        ax.text(0.03, 0.95, f"$r_{{rm}}$ = {r['r']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]\np < 0.001" if r["p"] < 0.001 else f"$r_{{rm}}$ = {r['r']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]\np = {r['p']:.3f}",
                transform=ax.transAxes, va="top", fontsize=8, color=INK)
        clean_ax(ax)
    fig.tight_layout(); fig.savefig(out / "fig1_validation.png", dpi=DPI); plt.close(fig)


def fig_norms(clean, out):
    m = clean.groupby("subject").median(numeric_only=True).reset_index()
    spec = [("top_d_shoulder_turn_closed", "Shoulder turn at top (°)", (8, 15), None),
            ("follow_back_ratio", "Follow-through / backswing length", (0.9, 1.1), 1.0),
            ("tempo_ratio", "Tempo ratio (backswing / downswing time)", None, 2.0)]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.8))
    for ax, (v, lab, band, ref) in zip(axs, spec):
        y = m[v].values
        x = RNG.uniform(-0.15, 0.15, len(y))
        ax.scatter(x, y, s=24, color=BLUE, lw=0.6, edgecolor="white", zorder=3)
        ax.plot([-0.3, 0.3], [np.median(y)] * 2, color=INK, lw=1.2, zorder=4)
        if band:
            ax.axhspan(*band, color=GRID, alpha=0.6, lw=0, zorder=1)
            ax.text(-0.55, band[1], "reference", fontsize=7, color=MUTED, va="bottom", ha="left")
        if ref:
            ax.axhline(ref, color=MUTED, lw=0.8, ls="--", zorder=2)
        ax.set_xticks([]); ax.set_xlim(-0.6, 0.6)
        ax.set_ylabel(lab, fontsize=8); clean_ax(ax)
    fig.tight_layout(); fig.savefig(out / "fig2_norms.png", dpi=DPI); plt.close(fig)


def fig_spm(res, items, out, name, hi_lab, lo_lab):
    n = len(items)
    fig, axs = plt.subplots(n, 2, figsize=(7.2, 2.1 * n), sharex=True)
    axs = np.atleast_2d(axs)
    x = np.arange(303)
    for row, (key, lab) in enumerate(items):
        R = res[key]
        a = axs[row, 0]
        a.plot(x, R["hi"], color=ORANGE, lw=1.4, label=hi_lab)
        a.plot(x, R["lo"], color=BLUE, lw=1.4, label=lo_lab)
        a.set_ylabel(lab, fontsize=8)
        b = axs[row, 1]
        b.plot(x, R["r"], color=INK, lw=1.2)
        for s in (-1, 1): b.axhline(s * R["thr_r"], color=MUTED, lw=0.7, ls="--")
        b.axhline(0, color=GRID, lw=0.6)
        for sg in R["sig"]:
            for ax_ in (a, b):
                ax_.axvspan(sg["start"], sg["end"], color=GRID, alpha=0.7, lw=0, zorder=0)
            b.text((sg["start"] + sg["end"]) / 2, 0.04, ("cluster p < 0.001" if sg["p"] < 0.001 else f"cluster p = {sg['p']:.3f}"), ha="center",
                   fontsize=7, color=INK, transform=b.get_xaxis_transform(),
                   bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
        b.set_ylabel("$r_{rm}$(t)", fontsize=8)
        for ax_ in (a, b):
            for v in (101, 202): ax_.axvline(v, color=MUTED, lw=0.6, ls=":")
            clean_ax(ax_)
    for ax_ in axs[-1]:
        ax_.set_xticks([0, 101, 202, 302], ["TA", "Top", "Impact", "Finish"])
    for i, ax_ in enumerate(axs.flat):
        ax_.text(0.0, 1.04, f"({chr(97 + i)})", transform=ax_.transAxes, fontsize=9, fontweight="bold", va="bottom")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(out / name, dpi=DPI); plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--font", default="DejaVu Sans")
    ap.add_argument("--exclude", default="", help="整体剔除的受试者，逗号分隔")
    a = ap.parse_args()
    EXCLUDE_SUBJECTS[:] = [x for x in a.exclude.split(",") if x]
    a.out.mkdir(parents=True, exist_ok=True)
    style(a.font)
    full, clean, curves = load(a.dir)
    R = {}

    # T1 sample / exclusions
    R["n_files"] = int(len(full)); R["n_subjects_all"] = int(full.subject.nunique())
    R["excl"] = full.exclude.value_counts().drop("", errors="ignore").to_dict()
    R["n_trials"] = int(len(clean)); R["n_subjects"] = int(clean.subject.nunique())
    R["trials_per_subject"] = clean.groupby("subject").size().agg(["min", "max", "median"]).to_dict()
    R["n_by_grade"] = clean.groupby("grade_en").subject.nunique().to_dict()

    # S1 validation
    R["validation"] = {x: rmcorr(clean, x, "head_speed") for x in
                       ["grip_speed_impact", "grip_speed_peak", "backswing_len_mm", "follow_len_mm", "t_downswing_s"]}
    R["impact_minus_peak_s"] = dict(median=float(clean.t_peak_minus_impact_s.median()),
                                    q1=float(clean.t_peak_minus_impact_s.quantile(.25)),
                                    q3=float(clean.t_peak_minus_impact_s.quantile(.75)))

    # S2 descriptives vs norms (subject medians)
    m = clean.groupby(["grade_en", "subject"]).median(numeric_only=True).reset_index()
    desc = []
    for v, lab, unit, lo, hi in [
        ("t_backswing_s", "Backswing time", "s", None, None),
        ("t_downswing_s", "Downswing time", "s", None, None),
        ("tempo_ratio", "Tempo ratio", "–", None, None),
        ("backswing_len_mm", "Backswing length (grip)", "mm", None, None),
        ("follow_len_mm", "Follow-through length (grip)", "mm", None, None),
        ("follow_back_ratio", "Follow-through / backswing ratio", "–", 0.9, 1.1),
        ("top_d_shoulder_turn_closed", "Shoulder turn at top", "°", 8, 15),
        ("top_d_shoulder_tilt_leadup", "Shoulder tilt at top (lead up +)", "°", None, None),
        ("top_d_forearm_roll_mean", "Forearm roll at top (open +)", "°", None, None),
        ("rng_pelvis_turn_closed", "Pelvis rotation range", "°", None, None),
        ("rng_pelvis_disp_target", "Pelvis displacement range (target axis)", "mm", None, None),
        ("head_speed", "Club-head speed", "device unit", None, None),
        ("face_deg", "Face angle (open +)", "°", None, None),
        ("path_deg", "Club path (in-to-out +)", "°", None, None),
        ("attack_deg", "Attack angle (up +)", "°", None, None)]:
        x = m[v]
        row = dict(var=v, label=lab, unit=unit, mean=x.mean(), sd=x.std(), min=x.min(), max=x.max())
        if lo is not None:
            row.update(norm=f"{lo}–{hi}", in_norm=int(((x >= lo) & (x <= hi)).sum()))
        desc.append(row)
    R["descriptives"] = desc

    # S3 ICC
    R["icc"] = []
    for v, lab in [("face_deg", "Face angle"), ("path_deg", "Club path"), ("attack_deg", "Attack angle"),
                   ("head_speed", "Club-head speed"), ("imp_d_forearm_roll_mean", "Forearm roll at impact"),
                   ("top_d_shoulder_turn_closed", "Shoulder turn at top"), ("backswing_len_mm", "Backswing length"),
                   ("tempo_ratio", "Tempo ratio")]:
        i, sw, sb = icc1(clean, v)
        R["icc"].append(dict(var=v, label=lab, icc=i, sd_within=sw, sd_between=sb))

    # S4 within-subject associations
    X = [c for c in clean.columns if c.startswith(("imp_d_", "top_d_", "imp_rate_", "rng_")) or c in
         ["t_backswing_s", "t_downswing_s", "t_follow_s", "tempo_ratio", "backswing_len_mm", "follow_len_mm",
          "follow_back_ratio", "grip_speed_peak", "grip_speed_impact", "grip_path_impact_deg", "grip_attack_impact_deg"]]
    R["n_candidates"] = len(X)
    ranked = clean.copy()
    for c in X + ["face_deg", "path_deg", "attack_deg", "head_speed"]:
        ranked[c] = clean.groupby("subject")[c].rank()
    assoc = {}
    for y in ["face_deg", "path_deg", "attack_deg", "head_speed"]:
        rows = [dict(x=x, **rmcorr(clean, x, y)) for x in X]
        t = pd.DataFrame(rows); t["q"] = bh(t.p.values)
        t = t.sort_values("p").reset_index(drop=True)
        top = t[t.q < 0.05].copy() if (t.q < 0.05).any() else t.head(3).copy()
        top["rank_r"] = [rmcorr(ranked, x, y)["r"] for x in top.x]
        loso = []
        for x in top.x:
            vals = [rmcorr(clean[clean.subject != s], x, y)["r"] for s in clean.subject.unique()]
            loso.append((min(vals), max(vals)))
        top["loso_min"] = [l[0] for l in loso]; top["loso_max"] = [l[1] for l in loso]
        for i, x in enumerate(top.x):
            rb = robustness(clean, ranked, x, y)
            for k, v in rb.items(): top.loc[top.index[i], k] = v
        assoc[y] = dict(top=top.to_dict("records"), min_q=float(t.q.min()))
    R["assoc"] = assoc

    # primary (pre-specified) hypothesis: forearm roll at impact (mean of both sides) → face angle
    px, py = "imp_d_forearm_roll_mean", "face_deg"
    pr = dict(x=px, **rmcorr(clean, px, py))
    pr["rank_r"] = rmcorr(ranked, px, py)["r"]
    vals = [rmcorr(clean[clean.subject != s], px, py)["r"] for s in clean.subject.unique()]
    pr["loso_min"], pr["loso_max"] = min(vals), max(vals)
    pr.update(robustness(clean, ranked, px, py))
    R["primary"] = pr

    # attack angle: is it simply a by-product of speed?
    def partial(x, y, z):
        d = clean[["subject", x, y, z]].dropna().copy()
        for v in (x, y, z): d[v] = d[v] - d.groupby("subject")[v].transform("mean")
        rx = d[x] - (d[x] * d[z]).sum() / (d[z] ** 2).sum() * d[z]
        ry = d[y] - (d[y] * d[z]).sum() / (d[z] ** 2).sum() * d[z]
        return float(np.corrcoef(rx, ry)[0, 1])
    R["attack_vs_speed"] = rmcorr(clean, "head_speed", "attack_deg")
    R["attack_partial_speed"] = {t["x"]: partial(t["x"], "attack_deg", "head_speed") for t in assoc["attack_deg"]["top"]}
    R["software"] = {"python": __import__("platform").python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": __import__("scipy").__version__,
                     "matplotlib": matplotlib.__version__}

    # S6 grade
    rows = []
    for v in ["head_speed", "face_deg", "path_deg", "attack_deg", "tempo_ratio", "backswing_len_mm",
              "top_d_shoulder_turn_closed", "top_d_shoulder_tilt_leadup", "imp_d_shoulder_tilt_leadup",
              "imp_d_forearm_roll_mean", "addr_trunk_fwd_tilt"]:
        a2 = m[m.grade_en == "Year 2"][v]; a3 = m[m.grade_en == "Year 3"][v]
        t = stats.ttest_ind(a2, a3, equal_var=False)
        rows.append(dict(var=v, y2_mean=a2.mean(), y2_sd=a2.std(), y3_mean=a3.mean(), y3_sd=a3.std(),
                         g=hedges_g(a2, a3), p=t.pvalue))
    gr = pd.DataFrame(rows); gr["q"] = bh(gr.p.values)
    R["grade"] = gr.to_dict("records")

    # S5 curves
    spm_items_face = [(("face_deg", "forearm_roll_trail"), "Trail forearm roll Δ (°)"),
                      (("face_deg", "shoulder_turn_closed"), "Shoulder turn Δ (°)")]
    spm_items_attack = [(("attack_deg", "forearm_roll_trail"), "Trail forearm roll Δ (°)")]
    spm_items_speed = [(("head_speed", "grip_target_mm"), "Grip Δ, target axis (mm)"),
                       (("head_speed", "forearm_roll_mean"), "Forearm roll Δ (°)")]
    spm = {}
    for (y, v), _ in spm_items_face + spm_items_speed + spm_items_attack:
        spm[(y, v)] = spm_regress(curves, clean, v, y)
    R["spm"] = {f"{y}|{v}": [dict(**s, start_label=phase_label(s["start"]), end_label=phase_label(s["end"]))
                             for s in res["sig"]] for (y, v), res in spm.items()}

    fig_validation(clean, a.out)
    fig_norms(clean, a.out)
    fig_spm(spm, spm_items_face, a.out, "fig3_spm_face.png", "face more open than own median", "face more closed than own median")
    fig_spm(spm, spm_items_speed, a.out, "fig4_spm_speed.png", "faster than own median", "slower than own median")

    def conv(o):
        if isinstance(o, (np.floating, np.integer)): return o.item()
        if isinstance(o, np.ndarray): return o.tolist()
        return str(o)
    (a.out / "results.json").write_text(json.dumps(R, indent=1, default=conv, ensure_ascii=False), encoding="utf-8")
    print("done:", a.out)


if __name__ == "__main__":
    main()
