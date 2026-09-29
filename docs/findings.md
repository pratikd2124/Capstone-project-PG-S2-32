# RYA baseline: findings and recorded values (28 Sep 2026)

Source of truth: Box folder "Statistical Core Refinement Project" (ID 410147109037). Murat's notebook is reference only. Full step-by-step proof: `walkthrough.ipynb` (Steps 1–9, ~5–7 min).

## Recorded baseline (`python -m scrp_toolkit.cli baseline`, 359/359 checks)
- Files: all 9 inputs' SHA-1 match Box (ONNX, EXP002.json, matrices, 4 instance CSVs, RY25 Pre/Post).
- Scaling: instance_data rebuilt with RYA's net_dataset_init_B.py preprocessing reproduces all 84 min/max values in EXP002.json. Without the rate rescaling, rate min/max fail.
- Dataset: 61,155 rows (notebook 60,810), 40 features; seed-42 split 48,924 / 12,231.
- ry9 example: shape 9.970475608, rate 20.926446973, mean 0.47645.
- Case study: school f7a2542b55bd29aa76cdf00c77805491, 40 four-option items scored (Murat's 46 minus 6 invalid chs items). His 15 four-option rows match exactly.
- EXP002 on seed-42 split: shape MAE 0.37905, rate MAE 1.93312. Not a held-out score.

## Equivalence proofs (walkthrough)
- Step 4 features: identical to RYA's feature_calculation_portal_script.py (0.0) and to R's features in the training CSVs (~1e-14) except 1 corrupt row.
- Step 5 inference: identical to RYA's normalise / ONNX / reverse_normalise path (0.0) after making reverse-normalisation float64.
- Step 6 dataset: RYA's own D_hat_Dataset (net_dataset_init_B.py, run live with only a folder-path adapter and a pandas-3 float() adapter) reproduces EXP002.json min/max to 5e-11; our dataset.py gives identical training rows, features and targets.
- Step 7 model: ONNX weights (PyTorch 1.7 export, opset 9; 21 MatMul / 21 Add / 20 Relu) loaded into our Net give identical outputs (0.0), 1,261,058 parameters. Our K/KL loss is identical to misc_functions.py (0.0), incl. the negative-prediction penalty.

## Differences from Murat's notebook (RYA code wins)
1. Rate rescaling rate <- 2*rate/(1/n_X + 1/n_Y) (net_dataset_init_B.py, used by perform_exp.py for EXP002).
2. chs1-6 are 6-option items; EXP002 takes 4x4 only. Now rejected/skipped.
3. EXP002's split seed was random and unsaved, so there is no true held-out set.
4. The Gamma(shape, rate) is the distribution of the d-hat test statistic between two groups, not an item unreliability score.
5. inference.py reverse-normalises in float64 like RYA's reverse_normalise (was float32, ~5e-7 off). Baseline re-recorded after this fix.

## Unreliability matrices (Step 3)
- Loaded from RYA, not recomputed: the re-test data used to estimate them is not in Box.
- asymm identical to RYA's asymm_calc on all 59 items. ry9 heatmap matches Box Heatmaps/NonTimeReversed/ry9.png.
- JSON vs P.RData differ in 2 of 1,064 cells: sun1 and ry17, row 2 col 1 (RData 0.08, JSON 0.07 / 0.09). Training data is centred on the RData values.
- Training data covers only the 53 four-option items (63,328 raw rows, no chs).

## Data-quality findings
- Corrupt training row orthog_B_3_large_classes_2790 (ry2): n_Y = 1 but Y shares 0, 1, 1, 1. It alone sets 9 of EXP002's scaling limits (e.g. stdev_Y max 10.49 vs real 1.50; n_Y min 1 vs 5). The baseline keeps it; drop it before retraining.
- Training group sizes are 5–9,819 (Pre) / 5–9,940 (Post). The model gives no warning outside this range.
- Zero-variance groups (everyone gives the same answer) and single respondents give NaN features (RYA's code and ours); 68 training rows are dropped for this.

## Open questions for RYA
1. Rate scale (likely resolved): the code implies the model predicts the Gamma of d-hat including the ½(1/n+1/m) factor, so the portal uses the rate as-is. Confirm.
2. Portal JavaScript for the minimal-effects test (epsilon, boundary groups, p-value).
3. EXP002's split seed, if logged.
4. sun1 / ry17 row 2 col 1: which value is correct, JSON (0.07 / 0.09) or RData (0.08)?
5. Can RYA share the re-test data and script used to estimate the matrices?
6. Is the portal aware of the corrupt training row / should groups under 5 students be blocked?

## For the team
- Training instability (Murat): in the notebook's saved run, epoch-15 loss jumped to 36.63 train / 4.55 test after sitting near 0.6. Re-check after the rate-rescaling fix.
- NaN on zero-variance input (SIN-WEI / Shuyun): guard upstream (stdev == 0) or have the detector catch it.
- Edge cases: `edge_case_instance_data.csv` has placeholder targets. Never put it in instance_data/; it is detector data only.
- Any analysis including chs1-6 must drop them until a 6-option model exists.
