# Changelog

## 1.1.1 (2026-11-16) — figure fixes

- Figure 1 composition scripts added (`code/revision/figure1/`); panels (b) and (e) redrawn from the 190 retained putts.
- Figure 1d: measured backswing length of the example putt (P19, trial 4) labelled 77.0 mm, so that the bars
  (76.72 mm) plus residual (0.24 mm) add up to the label.
- Figure S3: extra head-room above the panels so that the r_rm annotations do not overlap the data.
- Numerical results unchanged from 1.1.0.

## 1.1.0 (2026-11-16) — revision

- Gap filling per segment (`PUTT_MASK_MODE=per`, new default): a segment is interpolated only over the frames in
  which the physical sensors it depends on lost data (`code/preprocessing/putt/clean.py`, `_deps`). v1.0 behaviour:
  `PUTT_MASK_MODE=union`. 190 putts retained (v1.0: 188).
- Environment switches for the sensitivity analyses: `PUTT_LOWPASS`, `PUTT_IMPACT_SHIFT`, `PUTT_TAKE_MM`,
  `PUTT_TAKE_FRAC`, `PUTT_AXIS_PERTURB_DEG` / `PUTT_AXIS_PERTURB_DIR`, `PUTT_SPINE_COLLAPSE` (defaults reproduce the paper).
- Exclusion-rule labels follow the paper (R1–R3); `data/exclusion_log.csv` regenerated.
- Axis-split associations with club-head speed: pronation–supination terms (geometrically zero) removed from the
  test family (27 tests).
- Cluster statistics (cluster mass, threshold) added to `chain_results.json`.
- Figures 2–3: regression line and within-player r_rm in Figure 2b, residual curve in Figure 2c, revised titles and
  annotations in Figure 3.
- New folder `code/revision/` (pendulum geometry and lower limb, data quality, ICC CIs and other statistics,
  sensitivity analyses, Figures 5 and S1–S4) with outputs in `results/revision/`.
- `data/processed/` and `results/` regenerated; `data/processed/events.csv` added.

## 1.0.0 (2026-10-29)

First release with the submitted manuscript.
