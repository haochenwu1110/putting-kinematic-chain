# Golf putting kinematic-chain decomposition — data and code

Data and code for the paper

> Fan, X.; Lyu, W.; Chen, J.; Chen, W. **Testing the Pendulum Model of Golf Putting: A Stroke-by-Stroke
> Kinematic-Chain Analysis Using Wearable Inertial Sensors and a Launch Monitor.** *Bioengineering* (submitted).

Twenty right-handed golf-major university students each performed ten straight 5 m putts while wearing a
full-body inertial motion-capture suit (Perception Neuron, 60 Hz); a launch monitor (GCQuad) recorded every
stroke. Grip motion and forearm roll were decomposed into joint and rotation-axis contributions through a
rigid-body kinematic-chain identity, and all variation was analysed within players.

Running `python reproduce.py` regenerates every statistic, Tables 1–2 and Figures 2–4 of the paper from the
files in `data/processed/` (verified to reproduce `results/chain_results.json` exactly).

## Repository layout

```
├── reproduce.py                 one command: statistics, Table 2 and Figures 2–4 -> results_rerun/
├── requirements.txt
├── data/
│   ├── raw/                     200 motion-capture files (download from Zenodo, see data/raw/README.md)
│   ├── gcquad.csv               launch-monitor data, one row per putt (200 rows)
│   ├── pairing.csv              which raw file belongs to which launch-monitor row
│   ├── participants.csv         participant characteristics (P01–P20)
│   ├── exclusion_log.csv        QC status of all 200 putts and the exclusion rule applied (R1–R3)
│   └── processed/               output of the preprocessing pipeline (input of the analysis)
│       ├── features.csv             one row per putt: timing, amplitude, joint angles, club data
│       ├── chain_features.csv       one row per putt: kinematic-chain contributions
│       ├── qc_report.csv            QC status and flags per putt
│       ├── curves_wide.csv.gz       time-normalised joint-angle curves (3 phases × 101 points)
│       └── chain_curves.csv.gz      time-normalised contribution curves (3 phases × 101 points)
├── code/
│   ├── preprocessing/           raw files -> data/processed/   (run.py, config.py, putt/*.py)
│   ├── analysis/                data/processed/ -> results/    (chain_results.py, chain_figures.py, table2_timing.py)
│   └── validation/              synthetic-data validation of the decomposition (Section 2.5)
├── results/                     results as reported in the paper (chain_results.json, figures, Table 2)
└── tools/anonymize_raw.py       used once by the authors to de-identify the original recordings
```

## Installation

Python ≥ 3.10 (tested with 3.13):

```
pip install -r requirements.txt
```

## How to reproduce

**1. Statistics, tables and figures (from the processed data, ~2 min)**

```
python reproduce.py                 # -> results_rerun/
python reproduce.py --sensitivity   # also the sensitivity analysis without participant P07 (Section 3.5)
```

| Paper | Output |
|---|---|
| Section 3.1 (data retention, residuals) | `chain_results.json` → `n_putts`, `qc`; `data/exclusion_log.csv` |
| Table 1, Section 3.2 (sources of grip motion) | `chain_results.json` → `Q1`, `axes`; `chain_summary.md` |
| Section 3.3 (amplitude vs timing, joint shares) | `chain_results.json` → `Q2`, `axes` |
| Table 2 (timing and amplitude stability) | `table2_timing.csv` |
| Section 3.4 (forearm roll and face angle) | `chain_results.json` → `Q3`, `axes.face_assoc_exploratory` |
| Figures 2–4 | `figures/Figure2_who_swings`, `Figure3_speed_sources`, `Figure4_face_sources` (.png 600 dpi, .pdf) |

`chain_summary.md` is a short human-readable summary (in Chinese) of all key numbers.

**2. Preprocessing (from the raw recordings, ~3 min)**

Download the raw files into `data/raw/`, then

```
cd code/preprocessing
python run.py --dry-run     # checks file names and the pairing with data/gcquad.csv
python run.py               # -> data/processed_rerun/
```

Paths default to the repository folders (`config.py`); use `--data`, `--face` and `--out` to change them.
Point the analysis at the new folder with `python code/analysis/chain_results.py --dir data/processed_rerun --out results_rerun`.

**3. Synthetic validation of the decomposition (Section 2.5, ~1 min)**

```
python code/validation/check_synthetic.py
```

Generates 60 synthetic putts with known joint rotations, runs the full pipeline and reports the
decomposition error (< 0.3 mm), the recovery of pelvis translation (r ≈ 0.99) and the attribution of
forearm roll to the elbow/forearm (> 99 %).

## Data dictionary

### Conventions

- Participants are coded **P01–P20**; trials are numbered 1–10 in recording order.
- Body frame (per stroke, from the pelvis at address): **x** towards the target, **y** up, **z** from the body towards the ball.
- *lead* = left (target-side) arm or leg, *trail* = right; all participants were right-handed.
- Signs: face angle open +; club path in-to-out +; attack angle upward +; forearm roll + in the face-opening direction; turn angles + towards the backswing.
- Units: mm, m/s, s, degrees; club-head speed in mph (as exported by GCQuad).

### `data/raw/take###_P##_###_chr01.csv` (motion capture, one putt per file)

- Line 1: number of frames. Line 2: column header. Then one row per frame at 60 Hz.
- 59 skeleton nodes (`Hips`, `Spine`, `Spine1`, `Spine2`, `LeftArm`, `RightForeArm`, fingers, …), each with
  `-Sensor-Lost` (1 = no physical sensor data in that frame), `-Sensor-Quat-*`, `-Sensor-Gyro-*`, `-Sensor-Acce-*`,
  `-Joint-Velo-*`, `-Joint-Posi-*` (m, global) and `-Bone-Quat-*` (global orientation, x y z w).
- Ten nodes carry physical sensors: `Hips`, `Spine2`, `LeftUpLeg`, `RightUpLeg`, `LeftLeg`, `RightLeg`,
  `LeftArm`, `RightArm`, `LeftForeArm`, `RightForeArm`. All other nodes are estimated by the manufacturer's body
  model (their `-Sensor-Lost` is 1 throughout).
- The grip point is the mid-point of `LeftHand` and `RightHand` joint positions (the wrists).

### `data/gcquad.csv`

| Column | Meaning |
|---|---|
| `participant_id`, `trial` | putt identifier |
| `head_speed_mph` | club-head speed at impact (mph) |
| `face_deg` | face angle at impact (°, open +) |
| `path_deg` | club path (°, in-to-out +) |
| `attack_deg` | attack angle (°, upward +) |

### `data/exclusion_log.csv`

One row per recorded putt (200). `qc_status` / `qc_flags` come from the pipeline; `exclusion_rule` follows the
paper: **R1** stroke events could not be identified reliably (7 putts), **R2** pelvis, thorax or arm sensor
dropped out (3), **R3** gap between top and impact longer than 0.33 s (2). `included` marks the 188 analysed putts.
(Inside the analysis code these rules are labelled R2, R3 and R4; R1 in the code is reserved for excluding a whole participant with `--exclude`.)

### `data/processed/features.csv` (one row per putt)

| Prefix / column | Meaning |
|---|---|
| `head_speed`, `face_deg`, `path_deg`, `attack_deg` | launch-monitor data (as in `gcquad.csv`) |
| `t_backswing_s`, `t_downswing_s`, `t_follow_s`, `tempo_ratio` | phase durations (s) and backswing/downswing ratio |
| `backswing_len_mm`, `follow_len_mm`, `grip_speed_impact`, `grip_speed_peak` | grip amplitude (mm) and speed (m/s) |
| `addr_*` | posture at address (absolute values) |
| `top_d_*`, `imp_d_*` | change from address to the top of the backswing / to impact |
| `imp_rate_*` | angular velocity at impact (°/s) |
| `rng_*` | range of motion from takeaway to finish |

### `data/processed/chain_features.csv` (one row per putt)

`{g}` is one of `pelvis_trans`, `pelvis_rot`, `trunk`, `girdle` (clavicle), `shoulder_lead`, `shoulder_trail`,
`elbow_lead`, `elbow_trail`, `resid` (residual).

| Column pattern | Meaning |
|---|---|
| `bs_{g}_mm` | contribution to backswing length (address → top); terms sum to `bs_total_mm` |
| `ds_{g}_mm` | contribution to downswing displacement (top → impact); sum = `ds_total_mm` |
| `iv_{g}`, `ivz_{g}`, `ivy_{g}` | contribution to grip velocity at impact along x / z / y (m/s) |
| `roll_{lead\|trail}_{g}_{top\|imp}_deg` | contribution to forearm roll at the top / at impact; sum = `roll_*_total_*` |
| `bsx_`, `dsx_`, `ivx_`, `ivzx_`, `rollx_` + `{g}__{axis}` | the same contributions split by rotation axis of the segment: `twist`, `flex`, `bend` (pelvis, trunk, girdle) or `axial`, `swing` (shoulders, elbows); the axis terms sum exactly to the joint term |
| `chain_resid_x_pct`, `chain_link_sd_mm` | quality checks: residual RMS (% of grip speed), joint-distance SD |

### `data/processed/*_curves.csv.gz`

Long-format curves: one row per putt × variable (`var`), with 303 columns `back_000…back_100`,
`down_000…down_100`, `follow_000…follow_100` (time-normalised backswing, downswing and follow-through).
`chain_curves`: `cv_*` grip velocity contributions (m/s), `cd_*` cumulative displacement contributions (mm),
`cr_lead_*` / `cr_trail_*` cumulative forearm-roll contributions (°), `cvx_*` axis-split velocity contributions.

## Notes

- Code comments and console messages of the preprocessing pipeline are partly in Chinese; the logic is
  described in the paper (Sections 2.3–2.6) and in `code/preprocessing/README.md`.
- `code/analysis/make_results.py` provides the statistical helper functions (repeated-measures correlation,
  ICC, Benjamini–Hochberg, cluster permutation) used by `chain_results.py`.
- Bootstrap and permutation results are seeded; with the files provided here they reproduce the published
  numbers exactly.

## Licence and citation

Code: MIT (`LICENSE`). Data: CC BY 4.0 (`LICENSE-DATA.md`). Please cite the paper and this dataset (`CITATION.cff`).
Zenodo DOI: **[to be added]**.

## Contact

Weizheng Chen, School of Mechanical and Aerospace Engineering, Jilin University — wzchen@jlu.edu.cn
