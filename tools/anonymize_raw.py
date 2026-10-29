# -*- coding: utf-8 -*-
"""De-identify the original recordings before publishing them (run ONCE, locally, by the authors).

Inputs (kept private, never committed):
    --raw-dir           folder with the original motion-capture exports (take***<name><trial>_chr01.csv)
    --face              (optional) the original GCQuad table (.xlsx) used for the analysis;
                        data/gcquad.csv is already included, so this is only needed to regenerate it
    --file-key          PRIVATE_file_key.csv         original_file -> participant_id, trial, new_file
    --participant-key   PRIVATE_participant_key.csv  participant_id <-> name used in the GCQuad table

Outputs (safe to publish), written to --out (default: <repo>/data):
    raw/take###_P##_###_chr01.csv   motion-capture files renamed; file contents are copied unchanged
                                    (they contain only frame numbers and sensor/joint values)
    gcquad.csv                      participant_id, trial, head_speed_mph, face_deg, path_deg, attack_deg
                                    signs: face open +, path in-to-out +, attack angle up +
    pairing.csv                     participant_id, trial, raw_file, has_gcquad

The script also cross-checks gcquad.csv against data/processed/features.csv. A mismatch means the
GCQuad table given with --face is not the one the published results were computed from.
"""
import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code" / "preprocessing"))
from putt.io_utils import read_face_table  # noqa: E402

TOL = 1e-6


def write_pairing(fkey, g, out):
    pair = fkey[["participant_id", "trial", "new_file"]].rename(columns={"new_file": "raw_file"})
    pair = pair.merge(g[["participant_id", "trial"]].assign(has_gcquad=True), on=["participant_id", "trial"], how="left")
    pair["has_gcquad"] = pair["has_gcquad"].fillna(False).astype(bool)
    pair.sort_values(["participant_id", "trial"]).to_csv(out / "pairing.csv", index=False)
    print(f"[pairing] {len(pair)} rows, {int((~pair.has_gcquad).sum())} without launch-monitor data")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--face", type=Path, default=None,
                    help="original GCQuad .xlsx; omit to keep the data/gcquad.csv already in the repository")
    ap.add_argument("--file-key", type=Path, required=True)
    ap.add_argument("--participant-key", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=REPO / "data")
    a = ap.parse_args()

    fkey = pd.read_csv(a.file_key, encoding="utf-8-sig")
    pkey = pd.read_csv(a.participant_key, encoding="utf-8-sig")
    raw_out = a.out / "raw"
    raw_out.mkdir(parents=True, exist_ok=True)

    # ---- 1. motion-capture files ----
    index = {p.name: p for p in a.raw_dir.rglob("*.csv")}
    missing, copied = [], 0
    for r in fkey.itertuples():
        src = index.get(r.original_file)
        if src is None:
            missing.append(r.original_file)
            continue
        shutil.copyfile(src, raw_out / r.new_file)
        copied += 1
    print(f"[raw] copied {copied} of {len(fkey)} files to {raw_out}")
    if missing:
        print(f"[raw] NOT FOUND ({len(missing)}): {missing[:10]}{' ...' if len(missing) > 10 else ''}")

    # ---- 2. launch-monitor table ----
    if a.face is None:
        g = pd.read_csv(a.out / "gcquad.csv")
        print(f"[gcquad] --face not given: keeping existing {a.out / 'gcquad.csv'} ({len(g)} strokes)")
        write_pairing(fkey, g, a.out)
        print("\nDone. Do NOT commit the PRIVATE_*.csv keys or the original files.")
        return
    face, bad = read_face_table(a.face)
    if bad:
        print(f"[gcquad] rows that could not be parsed (Excel rows, 1-based): {[b + 1 for b in bad]}")
    name2id = dict(zip(pkey["name_in_gcquad_table"], pkey["participant_id"]))
    unknown = sorted(set(face.subject) - set(name2id))
    if unknown:
        print(f"[gcquad] ERROR: names in the GCQuad table that are not in the participant key: {unknown}")
        print("         -> this is probably not the table used for the analysis; nothing written for gcquad.csv")
        sys.exit(1)
    g = face.assign(participant_id=face.subject.map(name2id))
    g = g.rename(columns={"head_speed": "head_speed_mph"})
    g = g[["participant_id", "trial", "head_speed_mph", "face_deg", "path_deg", "attack_deg"]]
    g = g.sort_values(["participant_id", "trial"])
    g.to_csv(a.out / "gcquad.csv", index=False)
    print(f"[gcquad] {len(g)} strokes, {g.participant_id.nunique()} participants -> {a.out / 'gcquad.csv'}")

    # ---- 3. pairing table ----
    write_pairing(fkey, g, a.out)

    # ---- 4. cross-check against the published processed data ----
    feat = a.out / "processed" / "features.csv"
    if feat.exists():
        f = pd.read_csv(feat)[["subject", "trial", "head_speed", "face_deg", "path_deg", "attack_deg"]]
        m = f.merge(g, left_on=["subject", "trial"], right_on=["participant_id", "trial"], suffixes=("_pub", ""))
        diff = ((m.head_speed - m.head_speed_mph).abs() > TOL) | ((m.face_deg_pub - m.face_deg).abs() > TOL) | \
               ((m.path_deg_pub - m.path_deg).abs() > TOL) | ((m.attack_deg_pub - m.attack_deg).abs() > TOL)
        print(f"[check] compared {len(m)} strokes with data/processed/features.csv: {int(diff.sum())} mismatches")
        if diff.any():
            bad_ids = sorted(m.loc[diff, "participant_id"].unique())
            print(f"[check] WARNING: mismatching participants {bad_ids} -- use the GCQuad table from the analysis folder")

    print("\nDone. Do NOT commit the PRIVATE_*.csv keys or the original files.")


if __name__ == "__main__":
    main()
