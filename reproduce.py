# -*- coding: utf-8 -*-
"""Reproduce all statistics, tables and Figures 2-4 of the paper from the de-identified processed data.

    python reproduce.py                 # main analysis (20 participants, 188 putts) -> results_rerun/
    python reproduce.py --sensitivity   # also the sensitivity analysis without participant P07

Takes about 1-3 minutes (2000 bootstrap resamples and cluster permutations).
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
AN = REPO / "code" / "analysis"
PROC = REPO / "data" / "processed"


def run(args):
    print(">>", " ".join(str(a) for a in args))
    subprocess.run([sys.executable, *map(str, args)], check=True, cwd=AN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "results_rerun")
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--font", default="DejaVu Serif")
    a = ap.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    run(["chain_results.py", "--dir", PROC, "--out", out, "--font", a.font])
    run(["table2_timing.py", "--dir", PROC, "--out", out / "table2_timing.csv"])
    run(["chain_figures.py", "--dir", PROC, "--res", out, "--out", out / "figures", "--font", a.font])
    if a.sensitivity:
        s = out / "sensitivity_without_P07"
        run(["chain_results.py", "--dir", PROC, "--out", s, "--font", a.font, "--exclude", "P07"])
    print(f"\nDone. Compare {out / 'chain_results.json'} with results/chain_results.json")


if __name__ == "__main__":
    main()
