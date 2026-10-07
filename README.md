# Objective 3 — Simple Code Version

No `config.py` and no `utils.py`. Each script contains only the settings/functions it needs.

```text
Objective3_MinimalEffects_NoUtils/
├── DATA/                 # put PRE and POST CSV files here
├── REFERENCE/            # supplied matrices/network files
├── OUTPUTS/              # graphs and result CSVs appear here
├── 01_prepare_data.py
├── 02_matrix_sensitivity.py
├── 03_propagate_to_network.py
├── 04_bayesian_minimal_effect.py
└── 05_compare_methods.py
```

## Flow

```text
01  PRE vs POST response graph
02  Unreliability-matrix uncertainty graph
03  Propagate matrix uncertainty through network
04  Bayesian posterior P(effect > threshold)
05  Small result/comparison summary
```

## Run

```bash
pip install -r requirements.txt
python 01_prepare_data.py
python 02_matrix_sensitivity.py
python 03_propagate_to_network.py
python 04_bayesian_minimal_effect.py
python 05_compare_methods.py
```

## Important

Change `QUESTION = "ry9"` inside scripts if needed.

Two values are still provisional:

```python
MATRIX_SAMPLE_SIZE = 100
MIN_EFFECT = 0.10
```

Replace them after confirming the real test-retest sample size and agreed minimal-effect threshold.

Script 05 intentionally does **not** invent the existing Minimal Effects Test formula. Add the verified existing-test decision once the exact rule is confirmed.
