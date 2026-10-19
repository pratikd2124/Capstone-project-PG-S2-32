# Gemfinder prototype pipeline (PG-S2-32)

RYA's Gemfinder statistical core, unchanged, with four additions around it: input screening (O1), a simulation fallback (O2), uncertainty reporting (O3) and a faster, smaller network (O4). This is a prototype that runs on synthetic test cases. It doesn't use any real student data.

This branch contains only the new work. RYA's baseline (the `scrp_toolkit` package, the EXP002 model and the ρ matrices) lives on the **Baseline-Recreation** branch. The first time you run anything here, the files it needs are downloaded from that branch into a local `.baseline/` folder, which git ignores. After that it runs offline.

## Quick start (Anaconda Prompt)

1. Set up once: `setup_env.bat` (Mac/Linux: `bash setup_env.sh`). This creates the `scrp` environment, registers the **Python (scrp)** kernel, downloads the baseline and runs the tests.
2. `conda activate scrp`
3. `jupyter notebook pipeline_showcase.ipynb`, then Kernel → Restart & Run All. It takes about 20 seconds.

## What the pipeline adds

- **Stops invalid inputs:** a broken ρ or a six-option item is rejected (RED) instead of getting a p-value.
- **Handles a small class against a national cohort** without retraining the network.
- **Answers those edge cases reliably:** flagged cases use RYA's own simulation instead of the network.
- **Shows how much a result depends on question reliability:** a p-value range over ρ (AMBER if the decision could flip).
- **Runs on an optimised network:** batched, with optional fp16 weights, giving the same decisions.

## Folders

**Top level**
- `pipeline_showcase.ipynb`: the short presentation demo (all ten cases, then one spotlight per objective)
- `showcase.py`: helper functions for the showcase notebook
- `README.md`, `requirements.txt`, `environment.yml`, `setup_env.bat` / `setup_env.sh`, `pyproject.toml`, `.gitignore`, `.gitattributes`
- `.github/workflows/tests.yml`: runs the tests on GitHub on every push

**`o1_screening/`: Validation screening (SIN-WEI)**
- `o1_screening.py`: ρ checks and the Gamma-compatibility detector (unchanged from her handover)
- `O1_07_compatibility_detector_v1.joblib` + `_metadata.json`: her trained detector
- `detector_params.json`: the same detector exported to JSON, so it doesn't depend on the scikit-learn version
- `O1_demo_cases.json`: her acceptance cases

**`o2_fallback/`: Fallback inference (Shuyun)**
- `backup_methods.py`: estimates the null from simulation draws (empirical tail, Gamma fit), unchanged

**`o3_uncertainty/`: Uncertainty and Bayesian analysis (Pratik)**
- `common.py`, `config.py`, `06_bayesian_evaluation.py`: his pipeline_v2 files, unchanged
- `loader.py`: imports his files without editing them

**`o4_efficiency/`: Efficiency and cost optimisation (Murat)**
- `batched.py`: `EfficientPredictor`, the baseline predictor plus `predict_batch` (same answers, about 60× faster) and optional fp16 weights (model file halved)
- `profiling.py`: where EXP002's time and memory go, projected to 200,000 predictions
- `school_scoring.py`: scores a whole RY25 school in one batched call (needs the RY25 files, which are not in this branch)
- `cli.py`: `python -m o4_efficiency.cli profile` and `export-model` (writes the browser model)

**`integrated_pipeline/`: The integrated pipeline (Shlok)**
- `baseline.py`: downloads RYA's baseline from GitHub (pinned commit); set `SCRP_BASELINE_DIR` to use a local copy instead
- `contracts.py`: the data passed between stages (request, results, flags)
- `simulate.py`: RYA's B matrix, the d̂ statistic and his latent-type simulator
- `screen.py`: Step 1, the ρ check and the PASS / FLAGGED / REJECT decision (O1, plus the small-group rule)
- `boundary.py`: Step 2, the boundary pair and the minimal-effects test
- `nulls.py`: Step 3, the null from the network (O4) or from simulation (O2)
- `uncertainty.py`: Step 4, the p-value range over ρ (O3)
- `run.py`: `Pipeline` and `build_pipeline()`, which join the steps
- `scenarios.py`, `demo.py`: the ten demo cases; `python -m integrated_pipeline.demo`
- `benchmark.py`: 400 random comparisons against RYA's simulation; `python -m integrated_pipeline.benchmark`
- `bayes_eval.py`: Pratik's Bayesian analogue, used for evaluation only
- `pipeline_walkthrough.ipynb`: the detailed, step-by-step walkthrough

**`tests/`: Automated checks (`pytest -q`, 33 tests)**
- `test_pipeline.py`: pipeline maths, routing and flags, plus a real-EXP002 check against RYA's simulator
- `test_o4_batching.py`, `test_o4_profiling.py`, `test_o4_school_scoring.py`: O4 gives the same answers as the baseline
- `synthetic_project.py`: builds a small fake project (fake model and data) for the tests; `conftest.py` loads the baseline first

## Open points for RYA

- **ε:** 0.1 is a placeholder for the smallest difference that counts.
- **Portal test code:** the boundary step follows the draft paper and may change once we see the portal's code.
- **Test-retest counts:** these are needed to size the ρ uncertainty from data instead of using ±5%.
- **Calibration:** when the true gap is exactly ε, the test says "meaningful" 7–8% of the time instead of 5% (12% for very small groups).

## Team rules

- Work on your own branch and open a pull request into this one. Run `pytest -q` before merging.
- Don't commit `.baseline/`, real survey data or anything from Box.
