# SCRP toolkit — Statistical Core Refinement Project (PG-S2-32)

A tested Python package, command-line tool and walkthrough notebook for RYA's Gemfinder
statistical core. It runs RYA's original pipeline (from the customer's Box folder) with the
shipped EXP002 model, and gives the team one reproducible baseline to measure every experiment
against. **Nothing is retrained.** It has been checked line by line against RYA's own code
(`walkthrough.ipynb`); where Murat's notebook differed, RYA's code wins.

## Quick start (every teammate)

1. **Get the repo** (see *Team git workflow* below) and open the folder in VS Code.
2. **Create the environment once**, in the **Anaconda Prompt**:
   ```bat
   conda create -n scrp python=3.11 -y
   conda activate scrp
   cd /d "<path to this repo>"
   pip install -r requirements.txt
   python -m ipykernel install --user --name scrp --display-name "Python (scrp)"
   ```
3. **Check everything matches RYA** (about 2 minutes):
   ```bat
   python -m scrp_toolkit.cli baseline
   ```
   Expect `TOTAL: 359/359 checks passed -> BASELINE MATCHES`.
4. **Walk through it**: open `walkthrough.ipynb`, kernel **Python (scrp)**, *Restart → Run All*
   (about 5–7 minutes).

No paths need editing: the customer's data ships inside the repo in `data/`, and every tool
looks there by default.

## Settings (one place)

| Where | Setting | Default |
|---|---|---|
| `walkthrough.ipynb` → **SETTINGS** cell (top) | `DATA_DIR`, `REPORT_CSV`, `EDGE_CASE_DIR`, `EXAMPLE_ITEM/PRE/POST`, `SPLIT_SEED`, `TRAIN_FRAC`, `CHECK_ALL_TRAINING_ROWS`, `RUN_TRAINING_DEMO`, `CORRUPT_ROW` | repo-relative, works as-is |
| environment variable `SCRP_DATA` | data folder for **both** the notebook and the CLI | not set → `<repo>/data` |
| CLI `--project-root` | data folder for one command | `SCRP_DATA`, else `<repo>/data` |

To keep the data somewhere else (e.g. a Box Drive folder): `setx SCRP_DATA "D:\path\to\data"`
once, then reopen the terminal / VS Code.

## Repository layout

```
README.md  requirements.txt  pyproject.toml  walkthrough.ipynb
scrp_toolkit/            the package (config, reliability, features, inference, scoring, dataset,
                         model, train, compare, validate, baseline, edge_cases, cli)
  known_answers/         rya_baseline.json (Box SHA-1s + recorded outputs), murat_notebook_reference.json
tests/                   29 unit + end-to-end tests on synthetic data (CI runs them on every push)
docs/findings.md         findings, data issues and open questions for RYA
data/                    the customer's files, in Box's own folder structure (read-only!)
  Final Network w Code (15-07-2022)/final_network_July15_22/
    final_network_model/ EXP002_nn_optimal_epoch.onnx, EXP002.json
    instance_data/       orthog_B_1..4 *_feature_output_df.csv (training data)
    final_code/          RYA's original code (R + Python), used by the walkthrough for side-by-side checks
  Unreliability Matrices/unreliability-matrices.json
  Data/                  RY25_PreMay10.csv, RY25_PostMay10.csv (real survey responses)
.gitattributes           stops git changing line endings in data/ (the baseline checks SHA-1)
```

## Team git workflow

- `main` is the **baseline**: code that passes `python -m scrp_toolkit.cli baseline` (359/359)
  and `pytest`. Nobody commits to it directly.
- Everyone works on their own branch: `shlok`, `murat`, `sin-wei`, `shuyun`, `pratik`
  (already created). Start of each session: `git switch <you>` then `git merge main`.
- **Never edit `data/` or `scrp_toolkit/known_answers/`.** If you must change the pipeline,
  prove the effect: run `baseline` and attach `baseline_report.csv` to your pull request.
- To get your work into `main`: push your branch → open a pull request → someone else reviews
  → merge. Re-run `baseline` after merging.

**Data confidentiality.** `data/Data/RY25_*.csv` are real student survey responses (school,
postcode, age, gender, culture, free-text comments). Keep this repo **private** and share it
only with the team. Don't push it to a public host, and check the RYA / University data
agreement before putting it on any cloud git service. If that isn't allowed, keep `data/` out
of the remote and share it through Box instead; everything still works via `SCRP_DATA`.

## Setup without Anaconda (venv)

<details><summary>Show</summary>

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass   # only if activation is blocked
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```
</details>

## Walkthrough notebook

`walkthrough.ipynb` goes through the toolkit one file at a time. Each step explains what the file
does, compares it with the matching part of RYA's Box folder, and has cells to run with the
expected output. Keep the notebook in this folder so `import scrp_toolkit` works.

The **SETTINGS** and **SETUP** cells at the top must run first; after that every step can be re-run
on its own. If the notebook file is updated while it's open in VS Code, close the tab without
saving (or Ctrl+Shift+P → "File: Revert File") so the old copy isn't saved over it.

| Step | File | Checks against Box | Status |
|---|---|---|---|
| 1 | environment | packages RYA's code imports | done |
| 2 | `config.py` | the 9 input files (SHA-1) | done |
| 3 | `reliability.py` | `unreliability-matrices.json`, `asymm_calc` in `misc_functions.py`, Heatmaps folder (ry9 heatmap matches Box), `P.RData` + training data (3b) | done |
| 4 | `features.py` | RYA's `feature_calculation_portal_script.py` run side by side (identical, diff 0.0) and R's features in all 63,260 training rows (identical except 1 corrupt row) | done |
| 5 | `inference.py` | RYA's `normalise`/`reverse_normalise` + the shipped ONNX model: identical (diff 0.0) after a float64 fix; baseline re-recorded, 359/359 | done |
| 6 | `dataset.py` | RYA's own `D_hat_Dataset` (`net_dataset_init_B.py`) run live: reproduces EXP002.json (≤5e-11); ours identical to RYA's (same 48,924 training rows, features and targets) | done |
| 7 | `model.py` | ONNX weights loaded into our `Net`: identical outputs (0.0), 1,261,058 parameters; `K`/`KL` identical to `misc_functions.py` (0.0) | done |
| 8 | `scoring.py` | RY25 case-study school: 40 items, identical to the recorded baseline and to Murat's 15 four-option rows | done |
| 9 | `cli.py`, `baseline.py`, tests, `edge_cases.py` | 359/359 baseline checks, 29/29 tests, 8 edge cases (3 flagged for SIN-WEI) | done |

Running the whole notebook takes about 5–7 minutes (Restart → Run All). Step 3b installs `rdata`
automatically if it's missing.

## Running it

```bash
# Section 1 — rank items by reliability-matrix asymmetry
python -m scrp_toolkit.cli explore --top-n 10

# Section 2/2b — score the worked example (item ry9) + a real case-study school
python -m scrp_toolkit.cli score --out ry25_case_study_scores.csv

# Section 3 — train a fresh network from instance_data/ (quick demo: 15 epochs, not 1000)
python -m scrp_toolkit.cli train --quick-demo

# Section 4 — compare a freshly-trained network against the shipped EXP002 model
python -m scrp_toolkit.cli compare --quick-demo
```

Each command only requires the files it actually needs — `explore` and `train` never touch
the ONNX model, for example — and fails with a clear message naming what's missing rather
than a stack trace, if a required file isn't under `--project-root`.

## The RYA baseline (what every experiment is compared against)

The baseline is RYA's own pipeline on RYA's own files, with the shipped EXP002 model. **Nothing
is retrained.** Source of truth: the customer's Box folder *Statistical Core Refinement Project*.
Murat's notebook is a reference only (`known_answers/murat_notebook_reference.json`).

The data ships in the repo's `data/` folder. To re-download it, get from Box `Final Network w Code (15-07-2022)/final_network_July15_22/`
(`final_network_model/` + `instance_data/`), `Unreliability Matrices/unreliability-matrices.json`,
and `Data/RY25_PreMay10.csv` + `Data/RY25_PostMay10.csv` into one folder, e.g. `./data`.
Don't put anything else in `instance_data/`.

```bash
python -m scrp_toolkit.cli baseline 
```

It checks, and prints PASS/FAIL per section (full list in `baseline_report.csv`, exit 1 on any fail):

| Section | What must hold |
|---|---|
| files | SHA-1 of all 9 inputs = the SHA-1 Box reports for the customer's copy. Any edited, re-saved or different file fails. A stray CSV in `instance_data/` fails. |
| explore | 59 matrices (53 four-option, 6 six-option `chs1-6`); top-10 asymmetry `sun12 0.045383 ... ry2 0.021417` |
| model | `Model: 20 layers, width 256, 40 features, P_transform='all_P_asymm'` |
| scaling | `instance_data/` preprocessed our way reproduces the `min_max_df` inside EXP002.json: 42 columns x min/max. This proves our preprocessing is exactly what EXP002 was trained on. |
| ry9 / case_study / dataset | EXP002's outputs: the ry9 example, **every** item for the RY25 case-study school, and EXP002's shape/rate MAE on the seed-42 split. |

**Recorded baseline (28 Sep 2026, customer files, 359/359 checks):**

| | Value |
|---|---|
| Files | all 9 SHA-1s match Box |
| Scaling | 84/84 min/max values match EXP002.json (without the rate rescaling, rate min/max fail) |
| Dataset | 61,155 rows after preprocessing (notebook: 60,810), 40 features; seed-42 split 48,924 / 12,231 |
| ry9 example | shape 9.970475608, rate 20.926446973, mean 0.47645 |
| Case study | school `f7a2542b55bd29aa76cdf00c77805491`, 40 four-option items (top `ry16` mean 0.080818, bottom `ph4` 0.013270) |
| EXP002 error on seed-42 split | shape MAE 0.37905, rate MAE 1.93312 (notebook's 0.486 was on its non-RYA dataset) |

To re-record (only needed if RYA sends new files; refuses unless files + scaling pass):

```bash
python -m scrp_toolkit.cli baseline --capture scrp_toolkit/known_answers/rya_baseline.json
```

Anyone with the same Box files gets `BASELINE MATCHES` (about 2 minutes). Any experiment is judged
by what it changes in `baseline_report.csv`.

Cross-check against Murat's run (107/107 pass: ry9 and his 15 four-option case-study items match
exactly, because inference didn't change):

```bash
python -m scrp_toolkit.cli baseline --baseline scrp_toolkit/known_answers/murat_notebook_reference.json
```

### What the numbers mean

EXP002 predicts `(shape, rate)` of a Gamma distribution: the distribution of RYA's test statistic
d-hat between two groups (Pre vs Post) on one item (Technical Overview s2.4). `gamma_mean =
shape / rate` is that distribution's mean. The rate rescaling makes this the Gamma of d-hat *with*
the ½(1/n + 1/m) factor from the Technical Overview's Definition 1 (the R simulation's
`scale_true` option), so the model's rate should be usable as-is -- to be confirmed with RYA. The notebook called it "item unreliability"; that's
not what it is. The portal uses this Gamma for a minimal-effects test (p-value, 95% threshold);
that step is in RYA's JavaScript, not in Box, so it isn't part of this baseline yet.

## Other checks

| Command | Expect |
|---|---|
| `pytest -q` | `29 passed` (synthetic data; also runs in CI) |
| `validate --known-answers scrp_toolkit/known_answers/ry9_example.json` | `[PASS]` shape `9.9705`, rate `20.9264` |
| `gen-edge-cases --check-response-cases` | 8 cases; `all_same_option`, `single_respondent`, `empty_group` flagged (known NaN finding) |
| `train` / `compare --quick-demo` | Runs. Only for training experiments. Not part of the baseline. |

## Edge-case data (for SIN-WEI's "is this safe?" detector)

```bash
python -m scrp_toolkit.cli gen-edge-cases --check-response-cases
```

Writes three files to `edge_cases/`:

- `response_count_cases.json`: 8 unusual Pre/Post count pairs (single respondent, everyone
  picks the same answer, unpicked option, full reversal, empty group, unknown item, ...).
- `edge_case_instance_data.csv`: synthetic rows in exactly the real `instance_data` schema,
  built around degenerate P-matrices (near-identity, uniform noise, fully asymmetric,
  near-singular).
- `edge_case_labels.csv`: `exp_code → scenario`, the detector's label column.

With `--check-response-cases` it also runs the count cases through the real feature pipeline
and writes `response_count_case_report.csv`.

**Do not put `edge_case_instance_data.csv` into `instance_data/`.** Its shape/rate values are
placeholders, so it would corrupt the regressor's training targets. It's for the detector:
real rows labelled 0, these labelled 1.

**Finding:** two realistic inputs, everyone giving the same answer and a single respondent,
make `features.py` divide 0/0 in skewness/kurtosis and silently return `NaN`. This is inherited
from the original portal script. The team needs to decide whether to guard it upstream
(`stdev == 0`) or have the detector catch it.

## Testing

`pytest -q`: 29 tests on synthetic data, no project zip needed. CI runs them on every push.
They cover `reliability`, `features`, `dataset`, `model`, `edge_cases`, and an end-to-end run
of every CLI command (including `baseline`) on a fake project with the real file layout. They
don't check predictions against the real model; that's the `baseline` command above.
## Differences from Murat's notebook (all toward RYA's original code)

- **Rate rescaling.** EXP002 was trained by `perform_exp.py`, which imports
  `net_dataset_init_B.py`. That version rescales the target `rate <- 2*rate/(1/n_X + 1/n_Y)`
  after `dropna()`. The notebook followed `net_dataset_init.py`, which doesn't. `dataset.py` now
  does it (`normalise_rate=False` gives the notebook's behaviour). This changes training targets,
  the outlier trim, the row count, and any rate comparison. It does not change inference.
  The `scaling` check proves which is right.
- **6-option items.** `chs1-chs6` have 6x6 matrices. EXP002 only takes 4x4. The notebook scored
  them anyway: it used the first 16 cells of the 6x6 matrix and only counted answers 1-4, so its
  scores for them are meaningless (5 of its "10 least reliable" were chs items). They now raise
  a `ValueError` in `features.py` and are skipped by `score`.
- **Split seed.** RYA drew EXP002's train/test seed with `np.random.randint(100)` and never
  saved it. The seed-42 split is not EXP002's held-out set: most of its rows were in EXP002's
  training data. The MAE there is a fixed reference for comparing changes, not a test score.
- `inference.py` reverse-normalises in float64 like RYA's `reverse_normalise` (it used to keep the
  ONNX float32 output, about 5e-7 off RYA). Baseline re-recorded after this fix.
- `inference.py` is the portal-script path unchanged (features -> min-max -> ONNX -> reverse
  min-max). It detects the ONNX input rank. EXP002 was exported with a 1-D dummy input, so it's
  rank-1, as the notebook assumed.
- `config.py` only requires the files each command uses.

## Open questions for RYA

1. **Rate scale at inference (likely resolved).** The code implies the model predicts the Gamma of
   d-hat including the ½(1/n + 1/m) factor, so the portal should use its rate as-is. Confirm with RYA.
2. **Portal test code.** Can we get the JavaScript for the minimal-effects test (epsilon, the
   boundary groups, the p-value)? Then the baseline can cover the full test, not just the model.
3. **EXP002's split seed**, if it's in a log anywhere (`parameters_exp_codes.csv`?). Then the
   baseline can report a true held-out score.
4. **Two cells of the matrices disagree between RYA's copies.** `sun1` and `ry17`, row 2 col 1:
   `P.RData` (and the training data) say 0.08; `unreliability-matrices.json` (used at inference)
   says 0.07 / 0.09, making those rows sum to 0.99 / 1.01. Which is correct? The model was trained
   around the RData values. See walkthrough Step 3b.
5. **Re-test data.** The matrices can't be recomputed from Box (no re-test responses). Can RYA share
   the data and script used to estimate them?

## Worth raising with the team

- **Training instability (for Murat).** In the notebook's saved run, epoch-1 train loss was
  1113.98 and epoch 15 jumped to 36.63 train / 4.55 test after sitting around 0.6. The quick
  demo is diverging, not just under-trained. Re-check after the rate-rescaling fix.
- **NaN on zero-variance input (for SIN-WEI / Shuyun).** See the edge-case finding above.
- **One corrupt training row sets 9 of EXP002's scaling limits (Murat, SIN-WEI).**
  `orthog_B_3_large_classes_2790` has 1 student in group Y but answer shares of 0, 1, 1, 1. It
  alone sets e.g. `stdev_Y` max = 10.49 (real max 1.50), `mean_Y` max = 9.0 (3.92), `n_Y` min = 1
  (5). The baseline must keep it, because EXP002 was trained with these limits. Drop it before any
  retraining. The real training range for group sizes is 5+ (walkthrough Step 4).
- **6-option items (for everyone).** Any analysis that includes chs1-6 must drop them or use a
  different method until a 6-option model exists.
