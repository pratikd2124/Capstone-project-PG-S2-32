# SCRP toolkit (PG-S2-32)

Recreates RYA's Gemfinder baseline from the customer's Box files, using the shipped EXP002 model. Nothing is retrained.

## Quick start

1. Clone: `git clone -b main scrp-toolkit.bundle scrp-toolkit`
2. Set up (Anaconda Prompt, once): `setup_env.bat` (Mac/Linux: `bash setup_env.sh`)
3. Check: `conda activate scrp` → `python -m scrp_toolkit.cli baseline` → expect `359/359 ... BASELINE MATCHES`
4. Learn: open `walkthrough.ipynb`, kernel **Python (scrp)**, Restart → Run All (about 5–7 min)

## Files

**Setup**
- `setup_env.bat` / `setup_env.sh`: one-command install and self-check
- `environment.yml`: the `scrp` conda environment (Python 3.11)
- `requirements.txt`: packages (tested ranges); `requirements-lock.txt`: exact versions as a fallback
- `pyproject.toml`: package metadata

**Settings**
- `scrp_toolkit/settings.py`: every path and choice, in one place (shared, don't edit)
- `local_settings.example.py`: copy to `local_settings.py` for your own paths (ignored by git)

**Package: `scrp_toolkit/`** (each file starts with a plain-English summary)
- `config.py`: finds the data files
- `reliability.py`: unreliability matrices, asymmetry
- `features.py`: answer counts → the 40 model inputs
- `inference.py`: runs EXP002 → shape, rate
- `scoring.py`: scores every item for one RY25 school
- `dataset.py`: rebuilds RYA's training data
- `model.py`: network and loss
- `train.py`, `compare.py`: train a fresh network and compare it with EXP002 (experiments only)
- `validate.py`: quick known-answer check
- `baseline.py`: the full 359-check comparison with RYA
- `edge_cases.py`: unusual inputs for the safety detector
- `rds.py`: reads RYA's `P.RData`
- `cli.py`: command line, `python -m scrp_toolkit.cli <command>`
- `known_answers/`: recorded baseline (`rya_baseline.json`), Murat's reference, ry9 example

**Notebook and docs**
- `walkthrough.ipynb`: Steps 1–9, each one compared side by side with Box
- `docs/findings.md`: recorded values, differences from Murat's notebook, data issues, open questions for RYA
- `docs/team_demo_script.md`: 10-minute team demo

**Tests**
- `tests/`: 33 tests on synthetic data (`pytest -q`); `.github/workflows/tests.yml` runs them in CI

**Data: `data/`** (customer's Box files, read-only)
- `Final Network w Code (15-07-2022)/final_network_July15_22/`: model (`final_network_model/`), training data (`instance_data/`), RYA's code (`final_code/`)
- `Unreliability Matrices/unreliability-matrices.json`
- `Data/RY25_PreMay10.csv`, `Data/RY25_PostMay10.csv`: real student survey data

## Commands

- `baseline`: full check against RYA → `baseline_report.csv`
- `explore --top-n 10`: most asymmetric items
- `score`: ry9 example and the case-study school
- `validate --known-answers scrp_toolkit/known_answers/ry9_example.json`
- `gen-edge-cases --check-response-cases`: writes `edge_cases/`
- `train --quick-demo`, `compare --quick-demo`: experiments only, not part of the baseline

## Team rules

- `main` = baseline. Work on your own branch (`shlok`, `murat`, `sin-wei`, `shuyun`, `pratik`) and run `git merge main` at the start of each session.
- Never edit `data/` or `known_answers/`.
- Before merging: `pytest -q` (33 passed) and `baseline` (359/359). Attach `baseline_report.csv` to the pull request.
- The RY25 CSVs are confidential student data. Keep the repo private, and check the RYA/University data agreement before pushing to any cloud host.
