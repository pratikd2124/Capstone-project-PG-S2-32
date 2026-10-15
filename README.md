# GemFinder Statistical Core Refinement — Clean Final Pipeline V2

## The project in one minute

GemFinder compares survey response distributions.

The existing EXP002 model predicts a Gamma distribution for the scaled `d_hat_norm` sampling behaviour. Before using that model, this pipeline checks whether a comparison looks like data the model was trained to understand. SAFE comparisons use the production EXP002 path. WARN/REJECT comparisons are visibly routed toward fallback inference; WARN rows may still keep an EXP002 estimate for research evaluation, but that does not make them production-safe.

The pipeline also tests how unreliability-matrix (`rho`) assumptions affect EXP002. Separately, a Bayesian evaluation represents uncertainty in the observed response proportions using probability distributions rather than only point estimates. The Bayesian branch is evaluation-only and is not automatically deployed as a replacement for EXP002.

## Folder structure

```text
pipeline_v2/
├── config.py
├── common.py
├── 01_prepare_comparisons.py
├── 02_screen_inputs.py
├── 03_standard_exp002.py
├── 04_fallback_inference.py
├── 05_uncertainty_analysis.py
├── 06_bayesian_evaluation.py
├── 07_historical_benchmark.py
├── 08_compare_and_report.py
├── run_pipeline.py
├── requirements.txt
├── results/
├── figures/
│   ├── final/
│   └── debug/
└── README.md
```

## Source assets expected

The code searches recursively under `--assets-root` for:

- `RY25_PreMay10.csv`
- `RY25_PostMay10.csv`
- `unreliability-matrices.json`
- `EXP002.json`
- `EXP002_nn_optimal_epoch.onnx`
- `B_matrix.txt`
- optional `instance_data/*.csv` training instances

You can either extract the supplied archives yourself or place the zip archives under the assets root. The pipeline can extract only the required members into `pipeline_v2/.asset_cache/`. The original archives and original raw/model files are never modified.

A convenient layout is:

```text
data/
├── Data.zip
├── Unreliability Matrices.zip
└── final_network_July15_22.zip
```

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r pipeline_v2/requirements.txt
```

`onnxruntime` is required because the existing EXP002 ONNX model is used directly. The model is not retrained or changed.

## One-command execution

```bash
python pipeline_v2/run_pipeline.py --assets-root data
```

For a quicker development run you can reduce Bayesian draws without changing the production default in `config.py`:

```bash
python pipeline_v2/run_pipeline.py --assets-root data --draws 2000
```

If genuine test-retest transition counts later become available:

```bash
python pipeline_v2/run_pipeline.py \
  --assets-root data \
  --transition-counts data/real_transition_counts.csv
```

Do **not** supply Pre/Post survey exports as transition counts.

## File-by-file explanation

### `config.py`
Central user-adjustable settings. `OFFICIAL_EPSILON = None` is intentional; the pipeline does not invent an official minimal-effect boundary.

### `common.py`
Shared tested mathematical and engineering logic: asset discovery, one-time rho row normalisation for historical rounding, EXP002 feature construction, training-range checks, ONNX inference, original `d_hat_norm`, Dirichlet posterior sampling, rho perturbation, and utility functions.

The EXP002 output preprocessing preserves the original training transformation. During training the Gamma rate was transformed as:

```text
rate_scaled = 2 * beta / (1/n_X + 1/n_Y)
```

so the pipeline reverses this after ONNX prediction:

```text
beta = rate_scaled * 0.5 * (1/n_X + 1/n_Y)
```

Then:

```text
Gamma mean = alpha / beta
```

where `alpha` is Gamma shape and `beta` is Gamma rate.

### `01_prepare_comparisons.py`
Builds real within-dataset subgroup comparisons. Defaults to grade 4 vs 7, 11 vs 6, and 12 vs 5 in RY25 Post. It does not compare the entire Pre file with the entire Post file.

### `02_screen_inputs.py`
Checks rho validity and EXP002 compatibility. Stored EXP002 min/max feature ranges are used directly. If the supplied training instance CSVs are available, the real training sample-size-ratio range is derived from them rather than guessed.

Statuses mean:

- `SAFE`: validation and stored compatibility checks passed.
- `WARN`: research evaluation may continue, but at least one compatibility feature is outside the stored range or a rho structural warning exists.
- `REJECT`: input is invalid for trustworthy EXP002 execution.

These are **not significance decisions**.

### `03_standard_exp002.py`
Runs the existing ONNX network for SAFE and WARN comparisons. REJECT rows are not presented as trustworthy network results. Reports `alpha`, `beta`, Gamma mean, Gamma q95, runtime, and training-range status.

### `04_fallback_inference.py`
Contains the routing interface but deliberately does not invent a fallback method. Until the team's real fallback implementation is integrated, flagged production cases say:

```text
NOT INTEGRATED — awaiting team fallback implementation
```

### `05_uncertainty_analysis.py`
Runs two transparent sensitivity analyses on one representative historical comparison:

1. rho off-diagonal error-mass perturbations `-10%, -5%, -1%, 0%, +1%, +5%, +10%`;
2. sample-size sensitivity while keeping response proportions approximately fixed.

At 0% rho perturbation the matrix is exactly the normalised baseline matrix.

### `06_bayesian_evaluation.py`
Evaluation-only Bayesian branch.

#### Level A — response-distribution uncertainty

For each group:

```text
p_A | data ~ Dirichlet(prior + counts_A)
p_B | data ~ Dirichlet(prior + counts_B)
```

The Dirichlet-multinomial posterior is conjugate, so this posterior is exact. Monte Carlo draws are then propagated through the nonlinear `d_hat_norm` formula. This is **not variational inference**.

Outputs include posterior mean, median, SD, 95% credible interval, and exploratory threshold exceedance probabilities. If no official epsilon is supplied, diagnostic thresholds come from historical observed `d_hat_norm` quantiles and are explicitly labelled non-official.

#### Level B — rho uncertainty

The script searches only for genuine transition-count formats such as:

```text
item, initial_response, retest_response, count
```

or `item,c11,...,c44`, or a JSON mapping items to 4x4 integer count matrices.

If those counts are missing, it reports:

```text
RHO UNCERTAINTY: NOT IDENTIFIABLE FROM AVAILABLE DATA
```

and creates `results/rho_data_requirement_template.csv` with blank count cells. It never converts rho probabilities to pseudo-counts and never treats Pre/Post data as test-retest data.

### `07_historical_benchmark.py`
Joins all per-comparison outputs into the primary historical evaluation dataset `results/historical_benchmark.csv`.

### `08_compare_and_report.py`
Creates `results/final_pipeline_results.csv`, `results/final_summary.md`, and approximately six final figures.

## Expected core outputs

```text
results/comparisons.csv
results/screening_results.csv
results/exp002_results.csv
results/fallback_results.csv
results/rho_sensitivity.csv
results/sample_size_sensitivity.csv
results/bayesian_response_results.csv
results/diagnostic_thresholds.csv
results/rho_data_requirement_template.csv        # when rho counts are absent
results/bayesian_rho_summary.md
results/historical_benchmark.csv
results/final_pipeline_results.csv
results/final_summary.md
```

Final figures:

```text
figures/final/01_pipeline_screening_status.png
figures/final/02_flag_reasons.png
figures/final/03_rho_sensitivity.png
figures/final/04_bayesian_uncertainty.png
figures/final/05_historical_effect_distribution.png
figures/final/06_posterior_threshold_probability.png
```

## Interpretation of the six final figures

1. **Pipeline screening status** — how many historical comparisons are SAFE, WARN, or REJECT for model compatibility.
2. **Why comparisons were flagged** — which input features most often caused warnings; this is a model-domain diagnostic, not a statistical-significance chart.
3. **Rho sensitivity** — how EXP002 Gamma mean and q95 move when off-diagonal unreliability mass is changed around the baseline rho assumption.
4. **Bayesian uncertainty** — representative observed `d_hat_norm` values together with posterior means and 95% credible intervals while rho is fixed.
5. **Historical effect distribution** — empirical distribution of observed historical `d_hat_norm`; dashed lines are exploratory diagnostic quantiles, not official GemFinder boundaries.
6. **Posterior threshold probability** — representative probabilities `P(d_hat_norm > epsilon | data)` for those exploratory thresholds.

## Bayesian terminology in simple language

- **Prior**: the probability model used before observing the current group counts.
- **Posterior**: the updated probability model after combining the prior with observed counts.
- **Dirichlet distribution**: a distribution over possible four-category response-percentage vectors.
- **Posterior draw**: one plausible response-percentage vector sampled from the posterior.
- **Posterior mean**: the average of posterior draws for the quantity of interest.
- **Posterior probability**: the fraction of posterior mass satisfying a statement such as `d_hat_norm > epsilon`.
- **Credible interval**: an interval containing a stated proportion of posterior uncertainty, here 95%.
- **Minimal-effect threshold epsilon**: the boundary used to define a minimal-effect decision; no official value is assumed here.
- **Rho uncertainty**: uncertainty about the unreliability/transition probabilities themselves.
- **Response-distribution uncertainty**: uncertainty about the underlying response percentages given finite observed counts.

`EXP002 alpha` and `Dirichlet prior_alpha` are completely different quantities. EXP002 `alpha` is the fitted Gamma **shape** parameter. Dirichlet `prior_alpha` is a prior concentration used for response probabilities.

## Production path versus evaluation-only work

```text
Production architecture:
comparison -> screening -> SAFE -> EXP002
                         -> flagged -> fallback required

Evaluation-only branches:
- WARN EXP002 research evaluation
- rho sensitivity
- sample-size sensitivity
- Bayesian response uncertainty
- optional Bayesian rho uncertainty
```

The Bayesian branch does not automatically replace the production method.

## Requirements compliance and current limitations

The pipeline automatically writes a compliance table into `results/final_summary.md` based on what actually ran. Important expected limitations with the supplied assets are:

- fallback implementation: external/team-owned unless separately supplied;
- Bayesian rho uncertainty: data-blocked unless genuine paired test-retest transition counts are supplied;
- official Bayesian minimal-effect decision: waiting for an official epsilon;
- existing-vs-Bayesian decision agreement: not calculated unless the exact existing binary decision rule is supplied;
- variational inference: not claimed for the current conjugate Dirichlet components, because an exact posterior exists and Monte Carlo is only used for nonlinear propagation.

## Statistical rules enforced by the code

The implementation does not invent test-retest counts, does not create pseudo-counts from rho probabilities, does not call diagnostic thresholds official, does not equate EXP002 Gamma means with Bayesian `d_hat_norm` posterior means, does not interpret SAFE/WARN as significance, does not rename Monte Carlo sampling as variational inference, and does not silently trust out-of-training-range EXP002 predictions.
