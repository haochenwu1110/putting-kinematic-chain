# -*- coding: utf-8 -*-
"""Table 2 of the paper: individual stability of stroke timing and amplitude.

Mean ± SD over the 20 player medians; putt-to-putt CV = within-player SD / mean (median over players);
ICC(1) = one-way intraclass correlation. Uses the 188 strokes retained by the exclusion rules.
Usage: python table2_timing.py --dir ../../data/processed [--out ../../results/table2_timing.csv]
"""
import argparse
from pathlib import Path

import pandas as pd

import make_results as mr

ROWS = [("Backswing time (s)", "t_backswing_s"), ("Downswing time (s)", "t_downswing_s"),
        ("Tempo ratio", "tempo_ratio"), ("Backswing length (mm)", "backswing_len_mm"),
        ("Grip velocity at impact (m/s)", "grip_speed_impact"), ("Club-head speed (mph)", "head_speed")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    _, clean, _ = mr.load(a.dir)
    out = []
    for lab, v in ROWS:
        med = clean.groupby("subject")[v].median()
        g = clean.groupby("subject")[v]
        cv = (g.std(ddof=1) / g.mean()).median() * 100
        out.append(dict(variable=lab, mean=med.mean(), sd=med.std(ddof=1), cv_median_pct=cv, icc1=mr.icc1(clean, v)[0]))
    t = pd.DataFrame(out)
    print(f"n = {len(clean)} strokes, {clean.subject.nunique()} participants")
    print(t.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if a.out:
        t.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
