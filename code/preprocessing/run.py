# -*- coding: utf-8 -*-
"""推杆 IMU 批量预处理入口。

用法（在 putt_preproc 文件夹下）：
  python run.py --dry-run          # 只扫描文件、建立姓名映射、核对杆面表，不处理
  python run.py                    # 全部处理（含运动链分解）
  python run.py --plots            # 同时输出每杆 QC 图
  python run.py --only P06 --plots  # 只处理一个受试者（调试用）
  python run.py --save-raw         # 另存每杆清洗后的完整姿态/位置序列（npz，供后续分析）
"""
import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg
from putt import io_utils
from putt.pipeline import CURVE_VARS, process_take

PHASES = ("back", "down", "follow")


def log(msg, fh=None):
    print(msg)
    if fh:
        fh.write(msg + "\n")


def curve_row(key, var, c, n):
    row = {**key, "var": var}
    for j, ph in enumerate(PHASES):
        for p in range(n):
            row[f"{ph}_{p:03d}"] = c[j * n + p]
    return row


def save_raw(path, res, ev):
    raw = res["raw"]
    arrays = {f"q_{k}": v for k, v in raw["quats"].items()}
    arrays.update({f"p_{k}": v for k, v in raw["pos"].items()})
    arrays.update({f"qref_{k}": v for k, v in raw["q_ref"].items()})
    np.savez_compressed(path, M=raw["M"], hips_ref=raw["hips_ref"], fs=raw["fs"], hand_sign=raw["hand_sign"],
                        events=np.array([ev["address_start"], ev["address_end"], ev["takeaway"], ev["top"],
                                         ev["impact"], ev["finish"], ev["peak_v"]], dtype=float),
                        event_names=np.array(["address_start", "address_end", "takeaway", "top", "impact",
                                              "finish", "peak_v"]),
                        chain_left=np.array(raw["segs_by_side"].get("Left", [])),
                        chain_right=np.array(raw["segs_by_side"].get("Right", [])), **arrays)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=cfg.DATA_DIR)
    ap.add_argument("--face", type=Path, default=cfg.FACE_XLSX)
    ap.add_argument("--out", type=Path, default=cfg.OUT_DIR)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--plots", action="store_true", help="每杆一张 QC 图（200 张）")
    ap.add_argument("--plots-for", default="", help="只给指定杆出 QC 图，如 P06:8,P08:10")
    ap.add_argument("--save-series", action="store_true", help="另存每杆逐帧角度序列（挥杆窗口）")
    ap.add_argument("--save-raw", action="store_true", help="另存每杆清洗后的完整四元数与关节位置（npz）")
    ap.add_argument("--only", default=None, help="只处理该受试者（杆面表中的名字）")
    args = ap.parse_args()

    if args.out is None:
        args.out = args.data.parent / f"{args.data.name}_processed"
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    fh = open(out / "run_log.txt", "w", encoding="utf-8")

    # ---------- 杆面表 ----------
    if args.face is None or not Path(args.face).exists():
        cands = sorted({p for d in (args.data.parent, args.data) if d.exists()
                        for p in d.glob("*.xlsx") if not p.name.startswith("~$")})
        if len(cands) != 1:
            log(f"[错误] 找不到唯一的杆面表 .xlsx（候选: {[c.name for c in cands]}），请在 config.py 里填 FACE_XLSX 或用 --face 指定", fh)
            sys.exit(1)
        args.face = cands[0]
    log(f"[杆面表] 使用 {args.face}", fh)
    face, bad_rows = io_utils.read_face_table(args.face)
    subjects = list(dict.fromkeys(face.subject))
    log(f"[杆面表] {len(face)} 杆, {len(subjects)} 人: {', '.join(subjects)}", fh)
    if bad_rows:
        log(f"[杆面表] 无法解析的行（Excel 行号从 1 起）: {[r + 1 for r in bad_rows]}", fh)
    cnt = face.groupby("subject").size()
    if (cnt != 10).any():
        log(f"[杆面表] 杆数不是 10 的受试者: {cnt[cnt != 10].to_dict()}", fh)
    face.to_csv(out / "face_table_tidy.csv", index=False, encoding="utf-8-sig")

    # ---------- 文件与映射 ----------
    files = io_utils.scan_files(args.data, cfg.FILE_PATTERN)
    if files.empty:
        log(f"[错误] {args.data} 下没有找到 {cfg.FILE_PATTERN}", fh)
        sys.exit(1)
    unparsed = files[files.tag.isna()]
    if len(unparsed):
        log(f"[文件] 文件名无法解析，已跳过: {unparsed.file.tolist()}", fh)
    files = files[files.tag.notna()].copy()
    files["trial"] = files["trial"].astype(int)
    nmap = io_utils.load_or_create_name_map(out / "name_map.csv", files.tag, subjects, cfg.REDO_SUFFIX)
    kept, dropped, unmapped = io_utils.resolve_files(files, nmap, cfg.REDO_SUFFIX)

    log(f"[文件] 共 {len(files)} 个 take，标签 {files.tag.nunique()} 种", fh)
    log("[映射] 文件标签 → 杆面表受试者（可在 name_map.csv 中手改 subject 列后重跑）:", fh)
    for _, r in nmap.sort_values("tag").iterrows():
        n = (files.tag == r.tag).sum()
        log(f"    {r.tag:<12s} → {r.subject or '??? 未匹配':<12s} ({n} 个文件){'  [重测]' if r.is_redo else ''}", fh)
    if len(unmapped):
        log(f"[映射] 未匹配的标签（这些文件不会处理）: {sorted(unmapped.tag.unique())}", fh)
    if len(dropped):
        log(f"[映射] 已有重测文件，以下原始文件被忽略: {dropped.file.tolist()}", fh)
    no_files = sorted(set(subjects) - set(kept.subject))
    if no_files:
        log(f"[映射] 杆面表里有、但没有对应文件的受试者: {no_files}", fh)

    # 杆次对齐检查
    k2 = kept.merge(face[["subject", "trial"]].assign(has_face=True), on=["subject", "trial"], how="left")
    miss = k2[k2.has_face.isna()]
    if len(miss):
        log(f"[对齐] 以下文件在杆面表中没有对应杆次，已排除（多测的杆）: {miss.file.tolist()}", fh)
        kept = kept[~kept.file.isin(miss.file)]
    noid = files[~files.file.str.contains(r"\d{3}_chr", regex=True)]
    if len(noid):
        log(f"[文件] 文件名没有杆次编号，按第 1 杆处理: {noid.file.tolist()}", fh)
    dup = kept[kept.duplicated(["subject", "trial"], keep=False)]
    if len(dup):
        log(f"[对齐] 同一受试者同一杆次有多个文件，请删掉多余的: {dup.file.tolist()}", fh)
    per = kept.groupby("subject").trial.apply(lambda s: sorted(s.tolist()))
    for s, tr in per.items():
        if tr != list(range(1, 11)):
            log(f"[对齐] {s} 的杆次不是 1–10: 缺 {sorted(set(range(1, 11)) - set(tr))}", fh)
    kept[["file", "tag", "subject", "trial", "is_redo"]].to_csv(out / "file_list.csv", index=False, encoding="utf-8-sig")

    if args.dry_run:
        log("\n[dry-run] 仅检查，未处理任何数据。确认上面的映射和对齐无误后去掉 --dry-run 运行。", fh)
        fh.close()
        return

    # ---------- 处理 ----------
    if args.only:
        kept = kept[kept.subject == args.only]
    hints = {}
    ov = out / "events_override.csv"
    if ov.exists():
        o = pd.read_csv(ov, encoding="utf-8-sig")
        hints = dict(zip(o.file, o.impact_frame))
        log(f"[覆盖] 读取 events_override.csv: {len(hints)} 条", fh)
    plot_set = {(a.split(":")[0], int(a.split(":")[1])) for a in args.plots_for.split(",") if ":" in a}
    if args.plots or plot_set:
        (out / "qc_plots").mkdir(exist_ok=True)
    if args.save_series:
        (out / "series").mkdir(exist_ok=True)
    if args.save_raw:
        (out / "raw").mkdir(exist_ok=True)

    ev_rows, ft_rows, cv_rows, qc_rows = [], [], [], []
    ch_rows, chc_rows, ch_err = [], [], []
    t0 = time.time()
    for i, r in enumerate(kept.sort_values(["subject", "trial"]).itertuples(), 1):
        hs = -1 if r.subject in cfg.LEFT_HANDED else 1
        key = dict(file=r.file, subject=r.subject, trial=r.trial)
        try:
            res = process_take(r.path, cfg, hand_sign=hs, impact_hint=hints.get(r.file))
        except Exception as e:  # noqa: BLE001
            qc_rows.append({**key, "status": "ERROR", "flags": f"{type(e).__name__}: {e}"})
            log(f"  [{i}/{len(kept)}] {r.file}  ERROR {e}", fh)
            fh.write(traceback.format_exc())
            continue
        if "events" not in res:
            qc_rows.append({**key, "status": "FAIL", "flags": "; ".join(res["flags"])})
            log(f"  [{i}/{len(kept)}] {r.file}  FAIL {res['flags']}", fh)
            continue
        ev = res["events"]
        ev_rows.append({**key, **ev})
        ft_rows.append({**key, **res["features"]})
        for v in CURVE_VARS:
            cv_rows.append(curve_row(key, v, res["curves"][v], cfg.N_NORM))
        if "chain_features" in res:
            ch_rows.append({**key, **res["chain_features"]})
            for v, c in res["chain_curves"].items():
                chc_rows.append(curve_row(key, v, c, cfg.N_NORM))
        elif "chain_error" in res:
            ch_err.append(f"{r.file}: {res['chain_error']}")
        severe = ("grip_not_in_front", "impact_fallback", "gap_", "key_sensor_lost", "finish_at_file_end", "top_fallback")
        status = ("CHECK" if any(f.startswith(severe) for f in res["flags"])
                  else "WARN" if res["flags"] else "OK")
        qc_rows.append({**key, "status": status, "flags": "; ".join(res["flags"]),
                        "n_strokes": ev["n_strokes"], "lost_in_stroke": ev["lost_frames_in_stroke"],
                        "longest_gap": ev["longest_gap_in_stroke"],
                        "impact_frame": round(ev["impact"], 2), "peak_v_frame": ev["peak_v"]})
        if args.plots or (r.subject, r.trial) in plot_set:
            from putt.plots import qc_plot
            qc_plot(res, cfg.FS, f"{r.subject} #{r.trial}  ({r.file})",
                    out / "qc_plots" / f"{r.subject}_{r.trial:02d}.png")
        if args.save_series:
            lo, hi = max(0, ev["takeaway"] - 30), ev["finish"] + 30
            pd.DataFrame({k: v[lo:hi] for k, v in res["series"].items()},
                         index=pd.Index(np.arange(lo, min(hi, len(res["series"]["grip_speed"]))), name="frame")
                         ).to_csv(out / "series" / f"{r.subject}_{r.trial:02d}.csv", encoding="utf-8-sig")
        if args.save_raw:
            save_raw(out / "raw" / f"{r.subject}_{r.trial:02d}.npz", res, ev)
        print(f"  [{i}/{len(kept)}] {r.file}  {status} {'; '.join(res['flags'])}")

    # ---------- 输出 ----------
    pd.DataFrame(ev_rows).to_csv(out / "events.csv", index=False, encoding="utf-8-sig")
    feats = pd.DataFrame(ft_rows)
    lead = ["grade", "subject", "trial", "file", "head_speed", "face_deg", "path_deg", "attack_deg"]
    if len(feats):
        feats = feats.merge(face, on=["subject", "trial"], how="left")
        feats = feats[lead + [c for c in feats.columns if c not in lead]]
    feats.to_csv(out / "features.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cv_rows).to_csv(out / "curves_wide.csv", index=False, encoding="utf-8-sig")
    qc = pd.DataFrame(qc_rows)
    qc.to_csv(out / "qc_report.csv", index=False, encoding="utf-8-sig")

    # 运动链分解
    if ch_rows:
        ch = pd.DataFrame(ch_rows).merge(face, on=["subject", "trial"], how="left")
        ch = ch[lead + [c for c in ch.columns if c not in lead]]
        ch.to_csv(out / "chain_features.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(chc_rows).to_csv(out / "chain_curves.csv", index=False, encoding="utf-8-sig")
        segs = ch.chain_segments.value_counts()
        log(f"\n[运动链] {len(ch)} 杆完成分解。左臂链（各杆一致时只有一种）: {segs.to_dict()}", fh)
        log(f"[运动链] 目标轴残差 / 实测握把速度（RMS，挥杆窗口）: 中位数 {ch.chain_resid_x_pct.median():.1f}%，"
            f"最大 {ch.chain_resid_x_pct.max():.1f}%；关节间距标准差最大 {ch.chain_link_sd_mm.max():.2f} mm", fh)
        log(f"[运动链] 上杆长度：分解合计 {ch.bs_total_mm.median():.1f} mm（中位数），"
            f"其中残差 {ch.bs_resid_mm.median():.1f} mm", fh)
    if ch_err:
        log(f"[运动链] 以下文件未能分解（其余结果不受影响）: {ch_err[:10]}{' …' if len(ch_err) > 10 else ''}", fh)

    if cv_rows:
        from putt.plots import overview
        for v in ("grip_target_mm", "forearm_roll_mean", "shoulder_tilt_leadup", "grip_speed"):
            overview(pd.DataFrame(cv_rows), qc, v, cfg.N_NORM, out / f"overview_{v}.png")
        log("[总览] 已输出 overview_*.png（每张 20 人 × 10 杆，先看这 4 张）", fh)
    log(f"\n[完成] {len(kept)} 个文件, 用时 {time.time() - t0:.0f} s", fh)
    if len(qc):
        log(f"[QC] {qc.status.value_counts().to_dict()}  → 查看 qc_report.csv / qc_plots", fh)
    fh.close()


if __name__ == "__main__":
    main()
