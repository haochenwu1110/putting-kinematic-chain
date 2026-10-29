# -*- coding: utf-8 -*-
"""读取 take CSV、解析文件名、读取杆面表、建立 文件名 ↔ 受试者 映射。"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

NAME_RE = re.compile(r"^(?P<prefix>.*?)take\d+_?(?P<tag>.+?)_?(?P<trial>\d{3})?_chr\d+\.csv$", re.IGNORECASE)


def parse_filename(path):
    """take001li重003_chr01.csv -> ('li重', 3)
    大三take001chu003_chr01.csv -> ('chu', 3)      （文件名前缀忽略）
    take001yu_chr01.csv -> ('yu', 1)               （没有杆次编号 = 第 1 杆）
    不匹配返回 (None, None)。"""
    m = NAME_RE.match(Path(path).name)
    if not m:
        return None, None
    tr = m.group("trial")
    return m.group("tag"), (int(tr) if tr else 1)


def read_take(path):
    """第一行是帧数，第二行是表头。去掉行尾逗号造成的空列。"""
    for enc in ("utf-8", "gbk"):
        try:
            with open(path, encoding=enc) as fh:
                first = fh.readline().strip().rstrip(",")
            df = pd.read_csv(path, skiprows=1, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    n_decl = int(float(first)) if first.replace(".", "").isdigit() else None
    return df, n_decl


# ---------------- 杆面表 ----------------
_SIGNS = {"face": ("OP", "CL"), "path": ("I-O", "O-I"), "attack": ("UP", "DN")}


def _signed(s, pos, neg):
    s = str(s).strip()
    m = re.match(r"([\d.]+)\s*°?\s*(\S*)", s)
    if not m:
        return np.nan
    v = float(m.group(1))
    if v == 0:
        return 0.0
    t = m.group(2).upper()
    if t == pos:
        return v
    if t == neg:
        return -v
    return np.nan


def read_face_table(xlsx):
    """返回长表：grade, subject, trial, head_speed, face_deg(+OP), path_deg(+I-O), attack_deg(+UP)。
    Accepts either the original GCQuad Excel sheet or the tidy de-identified CSV (data/gcquad.csv) with columns
    participant_id, trial, head_speed_mph, face_deg, path_deg, attack_deg (signed: open / in-to-out / up positive)."""
    if str(xlsx).lower().endswith(".csv"):
        t = pd.read_csv(xlsx)
        t = t.rename(columns={"participant_id": "subject", "head_speed_mph": "head_speed"})
        if "grade" not in t.columns:
            t.insert(0, "grade", "")
        cols = ["grade", "subject", "trial", "head_speed", "face_deg", "path_deg", "attack_deg"]
        return t[cols].copy(), []
    raw = pd.read_excel(xlsx, header=None)
    rows, subj, grade, bad = [], None, None, []
    for i, r in raw.iterrows():
        a = r[0]
        if pd.isna(a) and r[1:].isna().all():
            continue
        a_str = str(a).strip()
        # 年级标题行
        if pd.isna(r[1]) and ("年级" in a_str or a_str.endswith("级")):
            grade = a_str
            continue
        # 表头行（第一个受试者名和表头在同一行）
        if isinstance(r[1], str) and "杆头" in r[1]:
            subj = a_str
            continue
        # 受试者名单独一行
        if pd.isna(r[1]) and not a_str.isdigit():
            subj = a_str
            continue
        try:
            k = int(float(a))
        except (TypeError, ValueError):
            bad.append(i)
            continue
        rec = dict(grade=grade, subject=subj, trial=k, head_speed=float(r[1]),
                   face_deg=_signed(r[2], *_SIGNS["face"]),
                   path_deg=_signed(r[3], *_SIGNS["path"]),
                   attack_deg=_signed(r[4], *_SIGNS["attack"]))
        if np.isnan([rec["face_deg"], rec["path_deg"], rec["attack_deg"]]).any():
            bad.append(i)
        rows.append(rec)
    df = pd.DataFrame(rows)
    return df, bad


# ---------------- 文件名映射 ----------------
def scan_files(data_dir, pattern):
    recs = []
    for p in sorted(Path(data_dir).glob(pattern)):
        tag, trial = parse_filename(p)
        recs.append(dict(file=p.name, path=str(p), tag=tag, trial=trial))
    return pd.DataFrame(recs)


def auto_name_map(tags, subjects, redo_suffix):
    """tag -> subject。只做大小写不敏感的精确匹配（去掉重测后缀），避免 yang/yangxue/yangzhi 误配。"""
    lut = {s.lower(): s for s in subjects}
    out = []
    for t in sorted(set(tags)):
        base = t.replace(redo_suffix, "")
        out.append(dict(tag=t, subject=lut.get(base.lower(), ""), is_redo=redo_suffix in t))
    return pd.DataFrame(out)


def load_or_create_name_map(map_csv, tags, subjects, redo_suffix):
    map_csv = Path(map_csv)
    auto = auto_name_map(tags, subjects, redo_suffix)
    if map_csv.exists():
        user = pd.read_csv(map_csv, encoding="utf-8-sig", dtype=str).fillna("")
        user["is_redo"] = user["tag"].str.contains(redo_suffix)
        # 新出现的 tag 用自动结果补上
        missing = auto[~auto.tag.isin(user.tag)]
        m = pd.concat([user, missing], ignore_index=True)
    else:
        m = auto
    m.to_csv(map_csv, index=False, encoding="utf-8-sig")
    return m


def resolve_files(files, name_map, redo_suffix):
    """给每个文件挂上 subject；同一受试者同时有原始与重测文件时只保留重测。"""
    f = files.merge(name_map[["tag", "subject", "is_redo"]], on="tag", how="left")
    f["subject"] = f["subject"].fillna("")
    f["is_redo"] = f["is_redo"].fillna(False).astype(bool)
    keep = []
    dropped = []
    for subj, g in f[f.subject != ""].groupby("subject"):
        if g.is_redo.any() and (~g.is_redo).any():
            keep.append(g[g.is_redo])
            dropped.append(g[~g.is_redo])
        else:
            keep.append(g)
    kept = pd.concat(keep, ignore_index=True) if keep else f.iloc[:0]
    dropped = pd.concat(dropped, ignore_index=True) if dropped else f.iloc[:0]
    unmapped = f[f.subject == ""]
    return kept, dropped, unmapped
