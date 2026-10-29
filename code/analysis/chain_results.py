# -*- coding: utf-8 -*-
"""推杆是谁在摆？——运动链分解的统计与作图。

回答三个问题
  Q1 握把的运动由谁产生：各关节对上杆长度、下杆位移、击球时握把速度的贡献占比（个人中位数 → 群体），
     随推杆时程的变化；"谁在摆"是否是稳定的个人方式（占比的 ICC）。
  Q2 推得快的那几杆，是谁多出了力：
     (a) 幅度还是时间：下杆平均速度 = 下杆位移 / 下杆时间，取对数后做精确的方差分解；
     (b) 哪个关节：个人内方差归因 share_k = cov_w(c_k, 总量) / var_w(总量)（各项之和 = 1）；
     (c) 各关节贡献与 GCQuad 杆头速度的个人内相关（r_rm，FDR）。
  Q3 杆面开的那几杆，是谁变了：前臂扭转在击球时的个人内方差归因；各关节扭转贡献与杆面角的 r_rm；
     时间连续分析（聚类置换检验）。

方差归因和占比都是运动学恒等式的直接结果，不依赖统计模型；与杆头参数的关系才用相关分析。

输入：putt_preproc 的输出文件夹（features.csv, qc_report.csv, curves_wide.csv, chain_features.csv, chain_curves.csv）
用法：python chain_results.py --dir <processed 文件夹> --out <输出文件夹> [--font "Times New Roman"] [--exclude Li]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import make_results as mr

# ---------------------------------------------------------------- 分组与配色（类别色按固定顺序，已做色觉校验）
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
RESID_C = "#bdbcb7"
FINE = ["pelvis_trans", "pelvis_rot", "trunk", "girdle", "shoulder_lead", "shoulder_trail",
        "elbow_lead", "elbow_trail", "resid"]
# 作图用的合并分组：骨盆（平移+转动）、躯干（含软件推算的肩带）、两侧肩关节、两侧肘/前臂
FIG = [("Pelvis", ["pelvis_trans", "pelvis_rot"]), ("Trunk", ["trunk", "girdle"]),
       ("Lead shoulder", ["shoulder_lead"]), ("Trail shoulder", ["shoulder_trail"]),
       ("Lead elbow/forearm", ["elbow_lead"]), ("Trail elbow/forearm", ["elbow_trail"])]
FIG_C = dict(zip([n for n, _ in FIG], SLOT))
ARMS = ["shoulder_lead", "shoulder_trail", "elbow_lead", "elbow_trail"]
BODY = ["pelvis_trans", "pelvis_rot", "trunk", "girdle"]
ROLL = ["pelvis_rot", "trunk", "girdle", "shoulder", "elbow"]
ROLL_FIG = [("Pelvis", ["pelvis_rot"]), ("Trunk", ["trunk", "girdle"]), ("Shoulder", ["shoulder"]),
            ("Elbow/forearm", ["elbow"])]
ROLL_C = {("lead", "Pelvis"): SLOT[0], ("lead", "Trunk"): SLOT[1], ("lead", "Shoulder"): SLOT[2],
          ("lead", "Elbow/forearm"): SLOT[4], ("trail", "Pelvis"): SLOT[0], ("trail", "Trunk"): SLOT[1],
          ("trail", "Shoulder"): SLOT[3], ("trail", "Elbow/forearm"): SLOT[5]}
N_BOOT = 2000
RNG = np.random.default_rng(2027)


# ---------------------------------------------------------------- 数据
def load(dirp):
    full, clean, curves = mr.load(dirp)
    ch = pd.read_csv(dirp / "chain_features.csv")
    drop = ["grade", "head_speed", "face_deg", "path_deg", "attack_deg"]
    ch = ch.drop(columns=[c for c in drop if c in ch.columns])
    d = clean.merge(ch, on=["file", "subject", "trial"], how="inner")
    cc = pd.read_csv(dirp / "chain_curves.csv" if (dirp / "chain_curves.csv").exists() else dirp / "chain_curves.csv.gz")
    cc = cc[cc.file.isin(d.file)]
    return full, d, curves, cc


def within(df, cols, s="subject"):
    """减去个人均值。"""
    return df[cols] - df.groupby(s)[cols].transform("mean")


def sum_cols(df, cols):
    return df[cols].sum(axis=1)


# ---------------------------------------------------------------- Q1：占比
def shares(d, prefix, suffix, total):
    """每杆各分组占总量的比例（%）。"""
    out = pd.DataFrame({"subject": d.subject})
    for g in FINE:
        out[g] = 100 * d[f"{prefix}{g}{suffix}"] / d[total]
    out["body"] = out[BODY].sum(axis=1)
    out["arms"] = out[ARMS].sum(axis=1)
    for name, gs in FIG:
        out[name] = out[gs].sum(axis=1)
    return out


def share_summary(sh):
    m = sh.groupby("subject").median(numeric_only=True)
    res = {c: dict(mean=float(m[c].mean()), sd=float(m[c].std()), min=float(m[c].min()), max=float(m[c].max()))
           for c in m.columns}
    res["n_arms_gt_body"] = int((m["arms"] > m["body"]).sum())
    res["n_subjects"] = int(len(m))
    return res, m


# ---------------------------------------------------------------- 方差归因（个人内）
def var_attr(d, parts, total):
    """parts: dict[name] -> 列名列表。share = cov_w(Σparts, total) / var_w(total)，按受试者整群自助法给 95% CI。"""
    cols = sorted({c for cs in parts.values() for c in cs} | {total})
    w = within(d, cols)
    subj = d.subject.values
    groups = [np.where(subj == s)[0] for s in np.unique(subj)]

    def calc(idx):
        t = w[total].values[idx]
        vt = np.sum(t * t)
        return {k: float(np.sum(w[cs].values[idx].sum(axis=1) * t) / vt) for k, cs in parts.items()}

    base = calc(np.arange(len(d)))
    boots = {k: [] for k in parts}
    for _ in range(N_BOOT):
        pick = RNG.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        for k, v in calc(idx).items():
            boots[k].append(v)
    return {k: dict(share=base[k], lo=float(np.percentile(boots[k], 2.5)), hi=float(np.percentile(boots[k], 97.5)))
            for k in parts}


# ---------------------------------------------------------------- 相关筛查
def assoc_table(d, xs, y):
    rows = [dict(x=x, **mr.rmcorr(d, x, y)) for x in xs if d[x].notna().sum() > 10 and d[x].std() > 0]
    t = pd.DataFrame(rows)
    t["q"] = mr.bh(t.p.values)
    ranked = d.copy()
    for c in xs + [y]:
        ranked[c] = d.groupby("subject")[c].rank()
    t["rank_r"] = [mr.rmcorr(ranked, x, y)["r"] for x in t.x]
    lo, hi = [], []
    for x in t.x:
        vals = [mr.rmcorr(d[d.subject != s], x, y)["r"] for s in d.subject.unique()]
        lo.append(min(vals)); hi.append(max(vals))
    t["loso_min"], t["loso_max"] = lo, hi
    t["robust"] = (t.q < 0.05) & (np.sign(t.rank_r) == np.sign(t.r)) & \
                  ((t.loso_min > 0) | (t.loso_max < 0))
    return t.sort_values("p").reset_index(drop=True)


# ---------------------------------------------------------------- 按转动轴拆分
AXIS_PARTS = [  # 名称, [(按轴列的组键, 轴)]；骨盆平移与残差不按轴拆
    ("Pelvis translation", None),
    ("Pelvis twist", [("pelvis_rot", "twist")]), ("Pelvis flexion", [("pelvis_rot", "flex")]),
    ("Pelvis side-bend", [("pelvis_rot", "bend")]),
    ("Trunk twist", [("trunk", "twist"), ("girdle", "twist")]), ("Trunk flexion", [("trunk", "flex"), ("girdle", "flex")]),
    ("Trunk side-bend", [("trunk", "bend"), ("girdle", "bend")]),
    ("Shoulder swing", [("shoulder_lead", "swing"), ("shoulder_trail", "swing")]),
    ("Shoulder axial rotation", [("shoulder_lead", "axial"), ("shoulder_trail", "axial")]),
    ("Elbow/forearm swing", [("elbow_lead", "swing"), ("elbow_trail", "swing")]),
    ("Elbow/forearm axial rotation", [("elbow_lead", "axial"), ("elbow_trail", "axial")]),
    ("Residual", None)]
AXIS_LABEL_ZH = {"Pelvis translation": "骨盆平移", "Pelvis twist": "骨盆扭转（绕长轴）", "Pelvis flexion": "骨盆屈伸",
                 "Pelvis side-bend": "骨盆侧倾", "Trunk twist": "躯干扭转（绕脊柱轴）", "Trunk flexion": "躯干屈伸",
                 "Trunk side-bend": "躯干侧屈", "Shoulder swing": "肩关节摆动", "Shoulder axial rotation": "肩关节内外旋",
                 "Elbow/forearm swing": "肘屈伸与摆动", "Elbow/forearm axial rotation": "前臂旋前旋后",
                 "Residual": "残差"}


def axis_cols(name, keys, prefix, suffix, base_prefix):
    if keys is None:
        g = "pelvis_trans" if name == "Pelvis translation" else "resid"
        return [f"{base_prefix}{g}{suffix}"]
    return [f"{prefix}{g}__{ax}{suffix}" for g, ax in keys]


def axis_analysis(d):
    out = {}
    specs = {"backswing": ("bsx_", "_mm", "bs_", "bs_total_mm"),
             "downswing": ("dsx_", "_mm", "ds_", "ds_total_mm"),
             "impact_velocity": ("ivx_", "", "iv_", "iv_total")}
    for key, (px, sx, bp, tot) in specs.items():
        parts = {n: axis_cols(n, k, px, sx, bp) for n, k in AXIS_PARTS}
        cols = {n: d[c].sum(axis=1) for n, c in parts.items()}
        # 检查：各部分之和 = 总量
        err = float((sum(cols.values()) - d[tot]).abs().max())
        sh = pd.DataFrame({n: 100 * v / d[tot] for n, v in cols.items()}).assign(subject=d.subject.values)
        m = sh.groupby("subject").median(numeric_only=True)
        out[key] = dict(check_max_abs_err=err,
                        shares={n: dict(mean=float(m[n].mean()), sd=float(m[n].std()),
                                        min=float(m[n].min()), max=float(m[n].max())) for n in m.columns},
                        attr=var_attr(d, parts, tot))
    # 躯干 / 骨盆内部：三个方向各占该部位贡献的多少（个人中位数，按绝对值归一）
    within = {}
    for part, grp in (("Pelvis", ["Pelvis twist", "Pelvis flexion", "Pelvis side-bend"]),
                      ("Trunk", ["Trunk twist", "Trunk flexion", "Trunk side-bend"])):
        cols = {n: d[axis_cols(n, dict(AXIS_PARTS)[n], "bsx_", "_mm", "bs_")].sum(axis=1) for n in grp}
        tot = sum(v.abs() for v in cols.values())
        sh = pd.DataFrame({n: 100 * v.abs() / tot for n, v in cols.items()}).assign(subject=d.subject.values)
        m = sh.groupby("subject").median(numeric_only=True)
        within[part] = {n: dict(mean=float(m[n].mean()), sd=float(m[n].std())) for n in grp}
    out["within_segment_backswing_abs_pct"] = within
    # 各轴分量与杆头速度（主要分析的补充）
    xs = []
    for n, k in AXIS_PARTS:
        if k is None:
            continue
        for px, sx in (("bsx_", "_mm"), ("dsx_", "_mm"), ("ivx_", "")):
            col = f"{px}{n.replace(' ', '_').replace('/', '_')}{sx}"
            d[col] = d[axis_cols(n, k, px, sx, "")].sum(axis=1)
            xs.append(col)
    out["head_speed_assoc"] = assoc_table(d, xs, "head_speed").to_dict("records")
    # 前臂扭转按轴（探索性，放补充材料）
    xs_roll = []
    for tag in ("lead", "trail"):
        for g, axs in (("pelvis_rot", ("twist", "flex", "bend")), ("trunk", ("twist", "flex", "bend")),
                       ("shoulder", ("axial", "swing")), ("elbow", ("axial",))):
            # 肘的 swing 分量垂直于前臂长轴，对前臂扭转恒为 0，不参与检验
            for ax in axs:
                col = f"rollx_{tag}_{g}__{ax}_imp_deg"
                if g == "trunk":
                    d[col + "_tg"] = d[col] + d[f"rollx_{tag}_girdle__{ax}_imp_deg"]
                    col = col + "_tg"
                xs_roll.append(col)
    out["face_assoc_exploratory"] = assoc_table(d, xs_roll, "face_deg").to_dict("records")
    return out


def axis_summary_lines(A):
    L = ["", "## 按转动轴拆分（握把运动的来源）", "",
         f"检查：各部分之和与总量的最大差 {A['backswing']['check_max_abs_err']:.2g} mm。", "",
         "| 来源 | 上杆长度（%） | 击球时握把速度（%） | 上杆长度逐杆变化的份额（95% CI） |", "|---|---|---|---|"]
    for n, _ in AXIS_PARTS:
        s1, s2, at = A["backswing"]["shares"][n], A["impact_velocity"]["shares"][n], A["backswing"]["attr"][n]
        L.append(f"| {AXIS_LABEL_ZH[n]} | {s1['mean']:.0f} ± {s1['sd']:.0f} | {s2['mean']:.0f} ± {s2['sd']:.0f} | "
                 f"{at['share'] * 100:.0f}%（{at['lo'] * 100:.0f}–{at['hi'] * 100:.0f}%） |")
    L += ["", "注：前两列为各人中位数的均值 ± 标准差，各行相加约为 100%。", ""]
    for part, lab in (("Pelvis", "骨盆"), ("Trunk", "躯干")):
        w = A["within_segment_backswing_abs_pct"][part]
        L.append(f"{lab}对上杆长度的贡献中，三个方向各占（按绝对值）：" +
                 "；".join(f"{AXIS_LABEL_ZH[n]} {v['mean']:.0f}%" for n, v in w.items()))
    sig = [r for r in A["head_speed_assoc"] if r["q"] < 0.05]
    L += ["", f"与杆头速度相关（FDR q < 0.05）的分轴特征：{len(sig)} 个；前 6 个："]
    for r in sig[:6]:
        L.append(f"- {r['x']}: r_rm = {r['r']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]，q = {r['q']:.3g}，稳健 = {r['robust']}")
    sig = [r for r in A["face_assoc_exploratory"] if r["q"] < 0.05]
    L += ["", f"前臂扭转分轴与杆面角（探索性，FDR q < 0.05）：{len(sig)} 个"]
    for r in sig[:6]:
        L.append(f"- {r['x']}: r_rm = {r['r']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]，q = {r['q']:.3g}，稳健 = {r['robust']}")
    return L


# ---------------------------------------------------------------- 作图
def label(ax, i):
    ax.text(0.0, 1.03, f"({chr(97 + i)})", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def fig_who_swings(m_bs, cc, d, out):
    """(a) 各人上杆长度的来源占比（个人中位数，按躯干占比排序）；(b) 握把速度各来源随时程的变化（群体均值）。"""
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.6), gridspec_kw=dict(width_ratios=[1.0, 1.25]))
    ax = axs[0]
    m = m_bs.sort_values("Trunk")
    rows = list(m.index)
    mean_row = m.mean()
    ylab = rows + ["Mean"]
    for i, (_, r) in enumerate(list(m.iterrows()) + [("Mean", mean_row)]):
        left_pos, left_neg = 0.0, 0.0
        for name, _ in FIG + [("resid", None)]:
            v = r[name]
            col = FIG_C.get(name, RESID_C)
            if v >= 0:
                ax.barh(i, v, left=left_pos, color=col, height=0.72, edgecolor="white", lw=0.8); left_pos += v
            else:
                ax.barh(i, v, left=left_neg, color=col, height=0.72, edgecolor="white", lw=0.8); left_neg += v
    ax.set_yticks(range(len(ylab)), [str(s) for s in ylab], fontsize=7)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.axvline(0, color=mr.MUTED, lw=0.6); ax.axvline(100, color=mr.GRID, lw=0.6, ls=":")
    ax.set_xlabel("Share of backswing length (%)")
    ax.spines[["top", "right"]].set_visible(False)
    label(ax, 0)

    ax = axs[1]
    cols = [c for c in cc.columns if c.startswith(("back_", "down_", "follow_"))]
    x = np.arange(len(cols))
    def mean_curve(var_list):
        sub = cc[cc["var"].isin(var_list)].groupby(["file", "subject"])[cols].sum()
        return sub.groupby("subject").mean().mean().values
    ax.plot(x, mean_curve(["cv_measured"]), color=mr.INK, lw=2.0, label="Grip (measured)")
    for name, gs in FIG:
        ax.plot(x, mean_curve([f"cv_{g}" for g in gs]), color=FIG_C[name], lw=2.0, label=name)
    ax.axhline(0, color=mr.GRID, lw=0.6)
    for v in (101, 202):
        ax.axvline(v, color=mr.MUTED, lw=0.6, ls=":")
    ax.set_xticks([0, 101, 202, 302], ["TA", "Top", "Impact", "Finish"])
    ax.set_ylabel("Grip velocity toward target (m/s)")
    mr.clean_ax(ax); label(ax, 1)
    h, l = ax.get_legend_handles_labels()
    from matplotlib.patches import Patch
    h.append(Patch(color=RESID_C)); l.append("Residual")
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(out / "figC1_who_swings.png", dpi=mr.DPI); plt.close(fig)


def _attr_panel(ax, attr, names, colors, title):
    y = np.arange(len(names))[::-1]
    for yi, n in zip(y, names):
        a = attr[n]
        ax.barh(yi, a["share"] * 100, color=colors[n], height=0.62)
        ax.plot([a["lo"] * 100, a["hi"] * 100], [yi, yi], color=mr.INK, lw=1.0)
        ax.text(max(a["hi"] * 100, a["share"] * 100) + 2, yi, f"{a['share'] * 100:.0f}%", va="center", fontsize=7,
                color=mr.INK)
    ax.set_yticks(y, names, fontsize=7.5)
    ax.axvline(0, color=mr.MUTED, lw=0.6)
    ax.set_xlabel("Share of within-player variance (%)", fontsize=8)
    ax.set_title(title, fontsize=8.5, loc="left")
    ax.spines[["top", "right"]].set_visible(False)


def fig_speed(attr_bs, attr_iv, amp, out):
    names = [n for n, _ in FIG] + ["Residual"]
    colors = {**FIG_C, "Residual": RESID_C}
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 3.0), gridspec_kw=dict(width_ratios=[1, 1, 0.7]))
    _attr_panel(axs[0], attr_bs, names, colors, "(a) Backswing length")
    _attr_panel(axs[1], attr_iv, names, colors, "(b) Grip velocity at impact")
    axs[1].set_yticklabels([])
    ax = axs[2]
    keys = [("amplitude", "Amplitude"), ("timing", "Timing")]
    for i, (k, lab) in enumerate(keys):
        a = amp[k]
        ax.bar(i, a["share"] * 100, color=[mr.BLUE, mr.MUTED][i], width=0.6)
        ax.plot([i, i], [a["lo"] * 100, a["hi"] * 100], color=mr.INK, lw=1.0)
        ax.text(i, max(a["hi"], a["share"]) * 100 + 3, f"{a['share'] * 100:.0f}%", ha="center", fontsize=7)
    ax.set_xticks([0, 1], [lab for _, lab in keys], fontsize=8)
    ax.set_ylabel("Share of variance in mean\ndownswing grip speed (%)", fontsize=8)
    ax.axhline(0, color=mr.MUTED, lw=0.6)
    ax.set_title("(c) Speed source", fontsize=8.5, loc="left")
    top = max(max(amp[k]["hi"], amp[k]["share"]) for k, _ in keys) * 100
    ax.set_ylim(top=max(110.0, top + 15))
    mr.clean_ax(ax)
    fig.tight_layout(); fig.savefig(out / "figC2_speed_sources.png", dpi=mr.DPI); plt.close(fig)


def fig_face(attr_roll, face_tab, out):
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.9), gridspec_kw=dict(width_ratios=[1, 1, 1.2]))
    for i, tag in enumerate(("lead", "trail")):
        names = [n for n, _ in ROLL_FIG]
        colors = {n: ROLL_C[(tag, n)] for n in names}
        _attr_panel(axs[i], attr_roll[tag], names, colors, f"({'ab'[i]}) {tag.capitalize()} forearm roll at impact")
        if i:
            axs[i].set_yticklabels(names, fontsize=7.5)
    ax = axs[2]
    rows = []
    for tag in ("lead", "trail"):
        for n, gs in ROLL_FIG:
            x = f"roll_{tag}_{'_'.join(gs)}_imp_deg" if len(gs) == 1 else f"roll_{tag}_{n.lower()}_imp_deg"
            r = face_tab.get(x)
            if r is not None:
                rows.append((f"{tag.capitalize()} {n.lower()}", r, ROLL_C[(tag, n)]))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, r, col) in zip(y, rows):
        ax.plot([r["lo"], r["hi"]], [yi, yi], color=mr.INK, lw=1.0)
        filled = r["q"] < 0.05
        ax.scatter(r["r"], yi, s=36, color=col if filled else "white", edgecolor=col, lw=1.4, zorder=3)
    ax.set_yticks(y, [lab for lab, _, _ in rows], fontsize=7)
    ax.axvline(0, color=mr.MUTED, lw=0.6)
    ax.set_xlabel("$r_{rm}$ with face angle (95% CI)\nfilled: FDR q < 0.05", fontsize=8)
    ax.set_title("(c) vs face angle", fontsize=8.5, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "figC3_face_sources.png", dpi=mr.DPI); plt.close(fig)


def fig_qc(d, out):
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.6))
    ax = axs[0]
    ax.hist(d.chain_resid_x_pct, bins=20, color=mr.BLUE, edgecolor="white")
    ax.set_xlabel("Residual RMS / grip velocity RMS (%)"); ax.set_ylabel("Putts")
    mr.clean_ax(ax)
    ax = axs[1]
    for tag, col in (("lead", SLOT[2]), ("trail", SLOT[3])):
        ax.scatter(d[f"imp_d_forearm_roll_{tag}"], d[f"roll_{tag}_total_imp_deg"], s=10, color=col, label=tag, lw=0)
    lim = np.nanpercentile(np.abs(d[["imp_d_forearm_roll_lead", "imp_d_forearm_roll_trail"]].values), 99)
    ax.plot([-lim, lim], [-lim, lim], color=mr.MUTED, lw=0.7, ls="--")
    ax.set_xlabel("Forearm roll at impact, twist angle (°)"); ax.set_ylabel("Sum of joint contributions (°)")
    ax.legend(frameon=False, fontsize=7)
    mr.clean_ax(ax)
    for i, a in enumerate(axs):
        label(a, i)
    fig.tight_layout(); fig.savefig(out / "figS_chain_qc.png", dpi=mr.DPI); plt.close(fig)


# ---------------------------------------------------------------- 中文摘要
def write_summary(R, out):
    s1, iv1, ds1 = R["Q1"]["backswing"], R["Q1"]["impact_velocity"], R["Q1"]["downswing"]
    f = lambda x: f"{x['mean']:.0f}% ± {x['sd']:.0f}%（{x['min']:.0f}%–{x['max']:.0f}%）"
    L = ["# 推杆是谁在摆：运动链分解结果摘要", "",
         f"样本：{R['n_subjects']} 人、{R['n_putts']} 杆。质量检查：目标轴残差中位数 {R['qc']['resid_x_pct_median']:.1f}%，"
         f"最大 {R['qc']['resid_x_pct_max']:.1f}%；前臂扭转各项之和与原扭转角 r = {R['qc']['roll_total_vs_twist_r_lead']:.3f}（前侧）/"
         f" {R['qc']['roll_total_vs_twist_r_trail']:.3f}（后侧）。", "",
         "## Q1 握把的运动由谁产生（各人中位数的均值 ± 标准差，括号内为范围）", "",
         "| 来源 | 上杆长度 | 下杆位移 | 击球时握把速度 |", "|---|---|---|---|"]
    for name, lab in [("Pelvis", "骨盆（平移+转动）"), ("Trunk", "躯干（含肩带）"), ("Lead shoulder", "前侧肩关节"),
                      ("Trail shoulder", "后侧肩关节"), ("Lead elbow/forearm", "前侧肘/前臂"),
                      ("Trail elbow/forearm", "后侧肘/前臂"), ("body", "合计：身体"), ("arms", "合计：手臂"),
                      ("resid", "残差")]:
        L.append(f"| {lab} | {f(s1[name])} | {f(ds1[name])} | {f(iv1[name])} |")
    L += ["", f"上杆长度中，手臂贡献大于身体（骨盆+躯干）的球员：{s1['n_arms_gt_body']}/{s1['n_subjects']} 人。",
          f"占比是否是稳定的个人方式：躯干占比 ICC = {R['Q1']['icc_share']['trunk_share']['icc']:.2f}，"
          f"手臂占比 ICC = {R['Q1']['icc_share']['arms_share']['icc']:.2f}。", "",
          "## Q2 推得快的那几杆，是谁多出了力", ""]
    amp = R["Q2"]["amplitude_vs_timing"]
    L.append(f"下杆平均握把速度的个人内方差：来自下杆幅度 {amp['amplitude']['share'] * 100:.0f}%"
             f"（95% CI {amp['amplitude']['lo'] * 100:.0f}–{amp['amplitude']['hi'] * 100:.0f}%），"
             f"来自下杆时间 {amp['timing']['share'] * 100:.0f}%（{amp['timing']['lo'] * 100:.0f}–{amp['timing']['hi'] * 100:.0f}%）。")
    hv = R["Q2"]["head_speed_links"]
    L.append(f"与杆头速度的个人内相关：上杆长度 r_rm = {hv['backswing_len_mm']['r']:.2f}，下杆时间 {hv['t_downswing_s']['r']:.2f}，"
             f"下杆平均握把速度 {hv['mean_downswing_speed']['r']:.2f}。")
    L += ["", "上杆长度的逐杆变化来自（个人内方差归因，95% CI）：", ""]
    for k, a in R["Q2"]["attr_backswing"].items():
        L.append(f"- {k}：{a['share'] * 100:.0f}%（{a['lo'] * 100:.0f}–{a['hi'] * 100:.0f}%）")
    L += ["", "击球时握把速度的逐杆变化来自：", ""]
    for k, a in R["Q2"]["attr_impact_velocity"].items():
        L.append(f"- {k}：{a['share'] * 100:.0f}%（{a['lo'] * 100:.0f}–{a['hi'] * 100:.0f}%）")
    sig = [r for r in R["Q2"]["head_speed_assoc"] if r["q"] < 0.05]
    L += ["", f"与杆头速度相关（FDR q < 0.05）的关节贡献：{len(sig)} 个；前 5 个："]
    for r in sig[:5]:
        L.append(f"- {r['x']}: r_rm = {r['r']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]，q = {r['q']:.3g}，稳健 = {r['robust']}")
    L += ["", "## Q3 杆面开的那几杆，是谁变了", ""]
    for tag, lab in (("lead", "前侧"), ("trail", "后侧")):
        L.append(f"{lab}前臂击球时扭转的逐杆变化来自：" + "；".join(
            f"{k} {a['share'] * 100:.0f}%（{a['lo'] * 100:.0f}–{a['hi'] * 100:.0f}%）" for k, a in R["Q3"]["attr_roll_imp"][tag].items()))
    sig = [r for r in R["Q3"]["face_assoc"] if r["q"] < 0.05]
    L += ["", f"与杆面角相关（FDR q < 0.05）的运动链特征：{len(sig)} 个"]
    for r in sig[:8]:
        L.append(f"- {r['x']}: r_rm = {r['r']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]，q = {r['q']:.3g}，稳健 = {r['robust']}")
    L += ["", "时间连续分析（聚类置换检验，p < 0.05 的聚类）：", ""]
    for k, cl in R["Q3"]["spm_face"].items():
        txt = "；".join(f"{c['start_label']} 至 {c['end_label']}，p = {c['p']:.3f}，平均 r = {c['r_mean']:.2f}" for c in cl) or "无"
        L.append(f"- {k}：{txt}")
    if "axes" in R:
        L += axis_summary_lines(R["axes"])
    L += ["", "其余输出：chain_results.json（全部数字）、figC1–C3、figS_chain_qc.png。"]
    (out / "chain_summary.md").write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--font", default="DejaVu Sans")
    ap.add_argument("--exclude", default="", help="整体剔除的受试者，逗号分隔")
    ap.add_argument("--n-perm", type=int, default=mr.N_PERM, help="聚类置换次数")
    a = ap.parse_args()
    mr.EXCLUDE_SUBJECTS[:] = [x for x in a.exclude.split(",") if x]
    mr.N_PERM = a.n_perm
    a.out.mkdir(parents=True, exist_ok=True)
    mr.style(a.font)
    full, d, curves, cc = load(a.dir)
    R = dict(n_putts=int(len(d)), n_subjects=int(d.subject.nunique()),
             excluded=full.exclude.value_counts().drop("", errors="ignore").to_dict())

    # ---------- 质量检查
    R["qc"] = dict(resid_x_pct_median=float(d.chain_resid_x_pct.median()), resid_x_pct_max=float(d.chain_resid_x_pct.max()),
                   resid_pct_median=float(d.chain_resid_pct.median()),
                   link_sd_mm_max=float(d.chain_link_sd_mm.max()),
                   backswing_vs_feature_mm_median=float((d.bs_total_mm - d.backswing_len_mm).median()),
                   roll_total_vs_twist_r_lead=float(np.corrcoef(d.roll_lead_total_imp_deg, d.imp_d_forearm_roll_lead)[0, 1]),
                   roll_total_vs_twist_r_trail=float(np.corrcoef(d.roll_trail_total_imp_deg, d.imp_d_forearm_roll_trail)[0, 1]),
                   chain_segments=d.chain_segments.value_counts().to_dict())

    # ---------- Q1
    sh_bs = shares(d, "bs_", "_mm", "bs_total_mm")
    sh_ds = shares(d, "ds_", "_mm", "ds_total_mm")
    sh_iv = shares(d, "iv_", "", "iv_total")
    s_bs, m_bs = share_summary(sh_bs)
    s_ds, _ = share_summary(sh_ds)
    s_iv, _ = share_summary(sh_iv)
    tmp = d[["subject"]].assign(trunk_share=sh_bs["Trunk"], arms_share=sh_bs["arms"])
    icc = {}
    for v in ("trunk_share", "arms_share"):
        i, sw, sb = mr.icc1(tmp, v)
        icc[v] = dict(icc=float(i), sd_within=float(sw), sd_between=float(sb))
    R["Q1"] = dict(backswing=s_bs, downswing=s_ds, impact_velocity=s_iv, icc_share=icc,
                   per_subject_backswing=m_bs.round(2).to_dict("index"))
    m_fig = m_bs[[n for n, _ in FIG] + ["resid"]]

    # ---------- Q2
    parts_bs = {n: [f"bs_{g}_mm" for g in gs] for n, gs in FIG} | {"Residual": ["bs_resid_mm"]}
    parts_iv = {n: [f"iv_{g}" for g in gs] for n, gs in FIG} | {"Residual": ["iv_resid"]}
    attr_bs = var_attr(d, parts_bs, "bs_total_mm")
    attr_iv = var_attr(d, parts_iv, "iv_total")
    dd = d[(d.ds_total_mm > 0) & (d.t_downswing_s > 0)].copy()
    dd["log_amp"] = np.log(dd.ds_total_mm)
    dd["neg_log_time"] = -np.log(dd.t_downswing_s)
    dd["log_mean_speed"] = dd.log_amp + dd.neg_log_time
    dd["mean_downswing_speed"] = dd.ds_total_mm / 1000 / dd.t_downswing_s
    amp = var_attr(dd, {"amplitude": ["log_amp"], "timing": ["neg_log_time"]}, "log_mean_speed")
    links = {x: mr.rmcorr(dd, x, "head_speed") for x in
             ["backswing_len_mm", "ds_total_mm", "t_downswing_s", "t_backswing_s", "mean_downswing_speed",
              "grip_speed_impact"]}
    xs_speed = [c for c in d.columns if c.startswith(("bs_", "ds_", "iv_")) and not c.endswith("_total")
                and c not in ("bs_total_mm", "ds_total_mm")]
    tab_speed = assoc_table(d, xs_speed, "head_speed")
    R["Q2"] = dict(attr_backswing=attr_bs, attr_impact_velocity=attr_iv, amplitude_vs_timing=amp,
                   head_speed_links=links, head_speed_assoc=tab_speed.to_dict("records"))

    # ---------- Q3
    attr_roll = {}
    for tag in ("lead", "trail"):
        parts = {n: [f"roll_{tag}_{g}_imp_deg" for g in gs] for n, gs in ROLL_FIG}
        attr_roll[tag] = var_attr(d, parts, f"roll_{tag}_total_imp_deg")
    for tag in ("lead", "trail"):                          # 躯干 + 肩带合并列（与作图分组一致）
        d[f"roll_{tag}_trunkgirdle_imp_deg"] = d[f"roll_{tag}_trunk_imp_deg"] + d[f"roll_{tag}_girdle_imp_deg"]
    xs_face = [c for c in d.columns if c.startswith("roll_") and c.endswith(("_imp_deg", "_top_deg"))] + \
              [c for c in d.columns if c.startswith(("ivz_", "ivy_", "iv_")) and c != "iv_total"]
    tab_face = assoc_table(d, xs_face, "face_deg")
    tab_path = assoc_table(d, [c for c in d.columns if c.startswith("ivz_")] + [c for c in xs_face if c.startswith("roll_")], "path_deg")
    tab_att = assoc_table(d, [c for c in d.columns if c.startswith("ivy_")] + [c for c in xs_face if c.startswith("roll_")], "attack_deg")
    face_lookup = {r["x"]: r for r in tab_face.to_dict("records")}
    for tag in ("lead", "trail"):                          # 作图用：躯干一栏用"躯干+肩带"
        if f"roll_{tag}_trunkgirdle_imp_deg" in face_lookup:
            face_lookup[f"roll_{tag}_trunk_imp_deg"] = face_lookup[f"roll_{tag}_trunkgirdle_imp_deg"]

    spm_face_vars = [f"cr_{t}_{g}" for t in ("lead", "trail") for g in ("trunk", "shoulder", "elbow")]
    spm_face = {}
    for v in spm_face_vars:
        if (cc["var"] == v).any():
            res = mr.spm_regress(cc, d, v, "face_deg")
            spm_face[v] = [dict(**s, start_label=mr.phase_label(s["start"]), end_label=mr.phase_label(s["end"]))
                           for s in res["sig"]]
    R["Q3"] = dict(attr_roll_imp=attr_roll, face_assoc=tab_face.to_dict("records"),
                   path_assoc=tab_path.to_dict("records"), attack_assoc=tab_att.to_dict("records"),
                   spm_face=spm_face)

    # ---------- 按转动轴拆分（需要新版预处理输出的 bsx_ / ivx_ / rollx_ 列）
    if any(c.startswith("bsx_") for c in d.columns):
        R["axes"] = axis_analysis(d)

    # ---------- 图
    fig_who_swings(m_fig, cc, d, a.out)
    fig_speed(attr_bs, attr_iv, amp, a.out)
    fig_face(attr_roll, face_lookup, a.out)
    fig_qc(d, a.out)

    def conv(o):
        if isinstance(o, (np.floating, np.integer)): return o.item()
        if isinstance(o, np.bool_): return bool(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return str(o)
    (a.out / "chain_results.json").write_text(json.dumps(R, indent=1, default=conv, ensure_ascii=False), encoding="utf-8")
    write_summary(R, a.out)
    print("done:", a.out)


if __name__ == "__main__":
    main()
