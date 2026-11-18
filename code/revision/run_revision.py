# -*- coding: utf-8 -*-
"""Reproduce every analysis added in revision v1.1 -> results/revision/.

Needs the raw recordings in data/raw/ (Zenodo) for the pendulum, data-quality and preprocessing-sensitivity steps.

    python code/revision/run_revision.py                  # everything (~15 min)
    python code/revision/run_revision.py --skip-preproc   # only the steps that use data/processed/ (~3 min)
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def run(script, *args):
    print(">>", script, *args, flush=True)
    subprocess.run([sys.executable, str(HERE / script), *map(str, args)], check=True, cwd=HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=REPO / "data" / "raw")
    ap.add_argument("--proc", type=Path, default=REPO / "data" / "processed")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "revision")
    ap.add_argument("--skip-preproc", action="store_true", help="skip every step that needs the raw recordings")
    ap.add_argument("--font", default="Liberation Serif")
    a = ap.parse_args()
    out = a.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    run("extra_stats.py", "--proc", a.proc, "--out", out / "extra_stats.json")
    if not a.skip_preproc:
        run("pendulum_lower_limb.py", "--proc", a.proc, "--raw", a.raw, "--out", out / "pendulum")
        run("data_quality.py", "--proc", a.proc, "--raw", a.raw, "--out", out / "data_quality")
        run("sensitivity.py", "--proc", a.proc, "--raw", a.raw, "--work", out / "sensitivity")
        run("revision_figures.py", "--rev", out, "--proc", a.proc, "--font", a.font)
    else:
        run("sensitivity.py", "--proc", a.proc, "--raw", a.raw, "--work", out / "sensitivity", "--skip-preproc")
    print(f"\nDone -> {out}")


if __name__ == "__main__":
    main()
