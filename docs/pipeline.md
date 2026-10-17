# Integrated pipeline (prototype, 17 Oct 2026)

`scrp_toolkit/pipeline/` puts Objectives 1-4 behind one call. RYA's method is not changed: the
response model, rho, d_hat and EXP002 are used as they are.

```
request (counts X, Y + rho)
  -> 1a rho validation            screen.validate_rho          SIN-WEI
  -> boundary pair (d_norm = eps) boundary.find_boundary       provisional (paper)
  -> 1b screening, both pairs     screen.screen                SIN-WEI v2 gate
       PASS    -> 2a EXP002 Gamma nulls.NetworkNull            Murat (fp32 / fp16)
       FLAGGED -> 2b simulated    nulls.simulated_null         Shuyun (MC empirical)
  -> 3 minimal-effects test       boundary.minimal_effects_test
  -> 3+ rho uncertainty           uncertainty.propagate        Pratik
  -> PipelineResult: decision, p-value, flag (GREEN/AMBER/RED), p-range
```

## Run it

```
python -m scrp_toolkit.cli pipeline-demo                 # 10 synthetic cases: direct vs robust vs truth
python -m scrp_toolkit.cli pipeline-demo --weights fp16  # same with the browser model
python -m scrp_toolkit.cli pipeline-benchmark --cases 400
pytest tests/test_pipeline.py
```

In code:

```python
from scrp_toolkit.pipeline.run import build_pipeline
from scrp_toolkit.pipeline.contracts import ComparisonRequest
pl = build_pipeline("data")                     # options: weights, test_mode, fallback_method, rho_delta ...
P = pl.net.predictor.p_matrices["ry9"]
res = pl.run(ComparisonRequest.make("ry9", [2, 1, 2, 3], [1800, 2880, 2520, 1800], P, epsilon=0.1))
res.summary()
```

## Modules

| Module | Stage | Receives -> returns |
| --- | --- | --- |
| `contracts.py` | - | the frozen dataclasses every stage shares |
| `simulate.py` | - | RYA's scaled d_hat and latent-type simulator, vectorised (port of `d_hat_simulation_from_LL.R`) |
| `screen.py` | 1a, 1b | `ComparisonRequest` -> `RhoCheck`; + `BoundaryPair` -> `ScreeningResult` |
| `boundary.py` | 3 | request -> `BoundaryPair`; + `NullDistribution` -> `TestResult` |
| `nulls.py` | 2a, 2b | request + `BoundaryPair` -> `NullDistribution` |
| `uncertainty.py` | 3+ | request + a re-run function -> `Uncertainty` |
| `run.py` | all | `Pipeline.run(request)` -> `PipelineResult`; `mode="direct"` = original path |
| `scenarios.py`, `demo.py` | - | synthetic demonstration cases |
| `benchmark.py` | - | direct vs robust vs simulated truth; fallback candidate comparison |
| `o1/` | - | SIN-WEI's gate (unchanged) + detector exported to `detector_params.json` |
| `o2/` | - | Shuyun's `backup_methods.py` (unchanged) |

## Decisions taken in the integration

- **Scale.** The simulator's mean on the ry9 example is 0.4756 vs EXP002's 0.4765, so EXP002's rate is used as-is and the observed statistic is the scaled d_hat (RYA's `scale_true = TRUE`). This confirms Murat's 12 Oct finding.
- **Fallback simulator is faithful.** RYA's simulator fixes each group's latent-type composition and draws answers through rho. Using the observed counts as that composition is exactly what EXP002 is given as X and Y, so Shuyun's "proxy" is the same model, conditional on the composition.
- **Test mode.** `boundary` (draft paper) by default; `point_null` (Murat's epsilon = 0 stand-in) selectable. Both are provisional until RYA sends the portal code. epsilon = 0.1 is a placeholder.
- **Detector without scikit-learn.** SIN-WEI's joblib was saved with scikit-learn 1.5.1. Its scaler and logistic-regression coefficients are exported to JSON and run in NumPy; a test checks it matches the joblib to 1e-9.
- **Rho uncertainty.** If test-retest counts are given, Dirichlet posterior draws of rho; otherwise Pratik's +/-5% error-mass perturbation. The whole test is re-run for each, giving a p-value range.
- **Flags.** RED = rejected, no p-value. AMBER = fallback used, or the decision flips inside the p-range. GREEN otherwise.

## Results (synthetic, 17 Oct)

Benchmark, 400 synthetic comparisons, truth = RYA's simulator with 10,000 draws:

| Size ratio | Cases | Flagged | Direct agrees with truth | Direct false positives | Robust agrees | Robust false positives | Median ms direct / robust |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1-2x | 100 | 0% | 98% | 0 | 98% | 0 | 3.2 / 6.6 |
| 2-10x | 100 | 0% | 96% | 0 | 96% | 0 | 3.2 / 6.6 |
| 10-100x | 100 | 3% | 97% | 0 | 97% | 0 | 3.2 / 6.5 |
| 100-2000x | 100 | 82% | 89% | 8 | 98% | 0 | 3.2 / 15.3 |

Fallback candidates on 32 flagged cases (q95 error vs a 40,000-draw reference): MC empirical 0.9% median (2.0% p90), Weibull 5.8%, Gamma refit 8.6%, EXP002 network 41%. MC empirical costs about 11 ms per flagged call with 4,000 draws.

Caveat: the truth and the fallback use the same simulator (different seeds), so the benchmark shows the network path's error against RYA's own model; it does not test the model itself.

## Known gaps (fed back to owners)

- O1: at a classroom of about 25 against a national cohort the classifier passes (p_incompatible about 0.05) while EXP002's q95 is about 40% too small (demo case C).
- O3: Pratik's `pipeline_v2/common.py` multiplies the network rate by 0.5(1/n + 1/m), which gives the Gamma of the unscaled d_hat, while his `d_hat_norm` is scaled. Not used here.
- O2: `evaluate_backup_methods.py` imports `legacy_925`, which is not in the repo.
