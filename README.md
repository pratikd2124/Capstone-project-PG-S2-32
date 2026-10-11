# Pratik — Standalone Robustness, Sensitivity and Bayesian Analysis

This folder contains my **independent capstone contribution** for the RYA/GemFinder statistical-core project. It intentionally **does not import or modify `scrp_toolkit`** and can be developed on the `pratik` branch before the team's branches are merged.

The implementation reads the original RYA assets directly and reproduces the feature formulas required by the shipped EXP002 model.

## Scope

The standalone analysis covers four parts:

1. **Unreliability-matrix sensitivity** — vary the off-diagonal error mass of a question's unreliability matrix by controlled amounts (default ±1%, ±5%, ±10%), renormalise each row, and measure how matrix asymmetry changes.
2. **EXP002 propagation** — pass the perturbed matrices through the original `EXP002_nn_optimal_epoch.onnx` model and track changes in Gamma shape, rate, mean (`shape / rate`) and 95th percentile.
3. **Sample-size sensitivity** — preserve approximately the same Pre/Post response proportions while changing `n` from small to very large values, then record changes in EXP002 output and which features move outside its stored training range.
4. **Bayesian alternative and robustness checks** — use Dirichlet posterior draws for the Pre/Post four-category response distributions, map them through the same scaled `d_hat_norm` formula used in the original R simulation, report a 95% credible interval, and classify unusual model inputs as `SAFE`, `WARN`, or `REJECT`.

## Why this is independent

There are **no imports from `scrp_toolkit`**. The shared logic is in `standalone_common.py` and reads these original RYA files directly:

- `unreliability-matrices.json`
- `EXP002.json`
- `EXP002_nn_optimal_epoch.onnx`
- `B_matrix.txt`

The analysis does not retrain or overwrite EXP002 and does not edit customer data.

## Required asset layout

The files can be anywhere below the repository root because the code searches recursively. For example:

```text
repo/
├── final_network_July15_22/
│   ├── final_code/B_matrix.txt
│   └── final_network_model/
│       ├── EXP002.json
│       └── EXP002_nn_optimal_epoch.onnx
├── Unreliability Matrices/
│   └── unreliability-matrices.json
├── standalone_common.py
├── 01_prepare_data.py
├── 02_matrix_sensitivity.py
├── 03_propagate_to_network.py
├── 04_bayesian_minimal_effect.py
├── 05_compare_methods.py
└── 06_input_robustness.py
```

If the assets are elsewhere, pass `--assets-root PATH` or set:

```bash
set RYA_ASSETS_DIR=C:\path\to\assets
```

On macOS/Linux:

```bash
export RYA_ASSETS_DIR=/path/to/assets
```

## Setup

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run order

### 1. Prepare controlled scenarios

```bash
python 01_prepare_data.py
```

This uses the worked `ry9` example (`Pre=[40,52,7,14]`, `Post=[9,17,54,12]`) by default. No raw student-level CSV is needed.

### 2. Matrix sensitivity

```bash
python 02_matrix_sensitivity.py --assets-root .
```

The perturbation changes total off-diagonal error probability while preserving the relative pattern among off-diagonal cells. Each row remains a valid probability vector.

### 3. Propagate sensitivity through EXP002

```bash
python 03_propagate_to_network.py --assets-root .
```

Outputs both sample-size and matrix sensitivity results. The ONNX model and its saved min/max scaling remain unchanged.

### 4. Bayesian alternative

```bash
python 04_bayesian_minimal_effect.py --assets-root .
```

The default prior is symmetric Dirichlet(0.5, 0.5, 0.5, 0.5). Posterior draws are converted to the same **scaled `d_hat_norm`** formula used in RYA's simulation code. The output includes posterior mean, median, SD, 95% credible interval and `P(d_hat_norm > epsilon)` for a user-specified exploratory effect threshold.

Example with a different threshold:

```bash
python 04_bayesian_minimal_effect.py --assets-root . --epsilon 0.10 --draws 50000
```

**Important:** this Bayesian analysis is an experimental uncertainty-aware alternative. It is not claimed to be a complete Bayesian reimplementation of RYA's original minimal-effects boundary test.

### 5. Compare outputs

```bash
python 05_compare_methods.py
```

### 6. Robustness / edge cases

```bash
python 06_input_robustness.py --assets-root .
```

The robustness script tests normal input, zero sample sizes, one respondent, degenerate single-category responses, a missing category, strongly imbalanced Pre/Post sizes, very large samples, and a six-option CHS item where available.

## Outputs

Generated files are written to:

```text
Analysis/results/
Analysis/figures/
```

Main result files include:

```text
matrix_sensitivity.csv
sample_size_sensitivity.csv
matrix_sensitivity_network.csv
bayesian_dhat_analysis.csv
method_comparison.csv
input_robustness.csv
comparison_summary.md
```

## Interpretation of sensitivity analysis

The **matrix sensitivity analysis** asks whether EXP002's fitted retest distribution is stable when assumptions about response unreliability change slightly.

The **sample-size sensitivity analysis** asks whether the predicted Gamma distribution remains stable when the response pattern stays similar but the number of respondents changes. It also explicitly records features outside the min/max values stored with EXP002. A model can still return a number outside this training range, so those cases are labelled `WARN` rather than silently treated as normal.

`shape / rate` is reported as the **mean of the fitted Gamma distribution of scaled `d_hat_norm`**. It should not be described on its own as the complete minimal-effect threshold. The 95th percentile is also reported as a useful upper-tail summary of the fitted distribution.

## Relationship to the team baseline

This branch is intended to remain independent while development is ongoing. Later, the team can compare these standalone results against the verified baseline implementation after branches are merged. That gives a useful validation step without requiring any team member to modify another person's branch now.

## Suggested contribution statement

> Pratik developed an independent robustness-analysis implementation for the existing RYA/GemFinder statistical core. The work evaluates sensitivity to unreliability-matrix assumptions and sample-size extrapolation, propagates those changes directly through the original EXP002 ONNX model, identifies unsafe or out-of-training-range inputs, and implements an exploratory Bayesian uncertainty analysis on the same scaled d_hat metric, without modifying the team's baseline branch.

## Suggested commit message

```text
Add standalone EXP002 sensitivity robustness and Bayesian analysis
```
