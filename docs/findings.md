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
- Only 3 schools appear in both RY25 surveys (Pre 358 school ids, Post 326; accessCode overlap 0). The case-study school f7a2542b… is one of them; the others are 72adf679… (1,720 / 94) and 83dff251… (1,640 / 69). Either most schools took only one survey or ids were hashed differently per file. Check before building before/after comparisons on real data.
- Zero-variance groups (everyone gives the same answer) and single respondents give NaN features (RYA's code and ours); 68 training rows are dropped for this.

## Model efficiency (Murat, 3–12 Oct 2026, branch `murat`)
Re-run any time with `python -m scrp_toolkit.cli profile` (writes `profile_report.json`). Timings are from an Apple M2; other machines differ, the ratios should hold.
- Where time went in one `predict()` (517 µs): feature extraction with pandas 76%, scaling 8%, network 16%. The pandas cost is 16 small Series built and joined per call. RYA's ONNX file takes one row of 40 per call.
- Network: 99.1% of the 1,261,058 parameters are in the 19 hidden 256x256 layers. Weights are 4.8 MB fp32 (1.2 MB as int8); about 2 KB of activations per row. Memory is not a constraint.
- `predict_batch` (inference.py) + `extract_features_batch` (features.py): the same steps on whole columns, network run 4,096 rows per call (same ONNX weights, batch dimension opened in memory only). 200k predictions: 104.9 s -> 1.65 s (64x). `predict` and `extract_features` are unchanged.
- Same answers as `predict`, NaN cases included. On the Apple M2 it is bit-identical: 20,000 random items (0 of 40,006 outputs differ), all 63,328 raw training rows turned back into counts (0 of 126,656), the 3 RY25 schools in both surveys (0 of 240), and the 242 case-study checks in the baseline. The row-wise dot products use `np.matmul` on stacked rows, which goes to a BLAS dot routine like `np.dot`; `(a*b).sum(axis=1)` adds in another order and drifted in the last digit on ~25% of rows.
- Not bit-identical on every CPU: on a GitHub Linux runner (12 Oct, same numpy 2.4.6) BLAS chose a kernel that adds the 4 terms in another order for one path, and 60 of 12,120 test features (skewness/kurtosis) differed by up to 8.9e-16. Model outputs still matched. RYA's own `np.dot` code varies across machines the same way. The tests now allow 4 units in the last place for features and 1e-6 relative for outputs.
- `score_school` reads each survey CSV once (`load_survey` in scoring.py; was twice): 0.80 s -> 0.42 s, identical output. Pass `survey=` to score several schools without reading again. Almost all of the remaining time is the CSV read itself.
- After batching, the network is 88% of `predict_batch` time, so shrinking it is the next lever.
- Post-training int8 does not work for EXP002: `quantize_dynamic` runs the network 2.9x faster but moves gamma_mean by 26% (median) and up to 7.5x; per-channel is worse (66% median). Errors build up over 20 layers. A smaller network (distillation from EXP002), possibly with quantization-aware training, is the route instead.
- fp16 weights (opt-in: `UnreliabilityPredictor(paths, weights="fp16")`, `cli export-model`): EXP002's weights stored as 16-bit with a Cast back to 32-bit. File 4.81 MB -> 2.41 MB, same speed, outputs move ~1e-4 (median), at most 1.1% on 2,000 random items. Decisions: 0.16% flip (see below), half the rounding yardstick. Default stays fp32 (RYA's model exactly); the baseline uses fp32.
- Decision check (scratch script, 12 Oct; not in the toolkit yet): stand-in for the minimal-effects test at epsilon = 0, alpha = 0.05. Observed d_hat with RYA's formula (d_hat_calc in d_hat_simulation_from_LL.R, B_matrix.txt, scale_true) against the Gamma the network predicts when both groups get the pooled answer shares; p = Gamma survival at d_hat. 40,000 synthetic comparisons on the 53 real matrices, group sizes 20–2,000, true change 0–25% (27% significant, 12,608 close calls with 0.01 < p < 0.20). Not RYA's boundary test; only for comparing models with each other.
  - Yardsticks (EXP002 against itself): fixing the 2-decimal rounding of the matrices (rows renormalised) flips 0.32% of decisions (1.0% of close calls); matrix error mass ±1% flips 0.37–0.39% (1.2%).
  - fp16 weights: 0.16% (0.5% of close calls).
  - Distilled networks (trained to copy EXP002, corrupt row dropped, log(shape)/log(rate) loss, mostly null-case inputs): 4x128 0.21 MB 23x faster 0.74%; 4x256 0.80 MB 7x 0.58%; 6x256 1.30 MB 4.5x 0.56%. More width/epochs stopped helping at ~0.56%, above the yardsticks. Against the simulation truth they are also worse than EXP002 (gamma_mean median error 2.6% for 4x256 vs 1.3%), so the gap is real error, not EXP002 noise. Not added to the toolkit; documented as an option if the team accepts more loss or the browser needs more speed. Training only on inputs where the two groups differ was much worse (2.3% flips): the test calls the network on null-case inputs.
- Scale of EXP002's Gamma (open question 1): re-running RYA's simulation in Python for 25 training rows (3,000 trials each), mean simulated scaled d_hat / EXP002's Gamma mean = 1.003 (median; 0.92–1.04). So the Gamma is for d_hat *with* the ½(1/n+1/m) factor and the rate is used as-is.
- Not adopted: running the network in PyTorch is 1.7x faster but not bit-identical (~1e-6 relative) and doesn't help the browser, which runs ONNX. pandas' pyarrow CSV reader would need a new dependency for everyone.
- First `predict_batch` call builds the batched session (~0.1 s, once per predictor); later calls of 40 items take ~1.5 ms.

## Open questions for RYA
1. Rate scale (resolved by simulation, 12 Oct, see Model efficiency): the model predicts the Gamma of d-hat including the ½(1/n+1/m) factor, so the portal uses the rate as-is. Confirm.
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
- Bulk scoring (Shlok, Pratik): use `predictor.predict_batch(x, y, questions)` instead of looping over `predict`; same answers, ~60x faster. For several schools, call `load_survey` once and pass `survey=` to `score_school`.
- Decision needed (all): a shrunk network can't be bit-identical. Agree how close is close enough (e.g. max relative change in gamma_mean) before Murat picks a smaller model. Pratik's confidence ranges could set the bar.
- "Affordable at 200k": Murat has read this as 200,000 predictions (one item for one before/after pair). Confirm, or say what the 200k counts.
- Pre-merge check is now `pytest -q` with 47 passed (was 33).
- Decision tolerance (all): fp16 flips 0.16% of stand-in decisions, below the matrix-rounding yardstick (0.32%). If that is acceptable, fp16 is the browser model. The realistic uncertainty of the matrices (Pratik) decides how much more loss, e.g. a 0.8 MB distilled network at 0.58%, would be acceptable.
