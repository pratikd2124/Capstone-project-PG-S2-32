##  Gamma Approximation Diagnostics &backup_method

### 1. Proposed Evaluation Criteria

To assess the accuracy of the Gamma approximation and compare potential backup methods, I propose the following metrics:

| Metric | Purpose | Priority |
|---|---|---|
| Relative error in the 95th percentile | Measures the accuracy of the estimated upper-tail critical value | Primary |
| Upper-tail probability error | Measures the difference between the observed exceedance probability and the nominal 5% level | Primary |
| Monte Carlo histogram with fitted Gamma curve | Provides a visual assessment of distributional fit | Supporting |

The evaluation metrics are proposed and are awaiting team agreement. A numerical threshold for declaring Gamma approximation failure has not yet been finalized.

### 2. Completed Simulation Experiments

Four classroom-size configurations were investigated:

- Case 1: n = 100, m = 100
- Case 2: n = 50, m = 200
- Case 3: n = 20, m = 500
- Case 4: n = 10, m = 1000

Gamma fits were assessed using Monte Carlo simulations, 95th-percentile errors, and upper-tail probability errors.

### 3. Case 4: Repeated Independent Validation

A candidate Gamma approximation failure was identified for question `ry2`, with classroom sizes n = 10 and m = 1000 and fixed latent type counts.

Ten independent repetitions were conducted, each using 10,000 training simulations and 10,000 testing simulations.

Results:

- Mean relative 95th-percentile error: **6.07%**
- Mean observed upper-tail probability: **6.074%**
- Nominal upper-tail probability: **5%**
- Repetitions with observed upper-tail probability above 5%: **10 out of 10**

These results provide evidence of a reproducible upper-tail discrepancy under this specific simulation configuration.

### 4.## Exploratory Analysis Using Real Survey Data

### Objective

This analysis investigates where the Gamma approximation may fit poorly using response distributions derived from real survey datasets, as part of the 25 September milestone.

### Data Sources

Three survey datasets were used:

- `RY_24.csv`
- `RY25_PreMay10.csv`
- `RY25_PostMay10.csv`

Four survey questions were examined: `ry10`, `ry11`, `ry13`, and `ry14`.

For each question, three pairwise comparisons were evaluated, resulting in 12 survey-derived configurations.

### Evaluation Method

For each configuration, 10,000 simulated values of the scaled d-hat statistic were generated, with 7,000 used for Gamma fitting and 3,000 held out for evaluation.

The evaluation metrics included:

1. Relative error in the 95th percentile (Q95).
2. Upper-tail probability discrepancy.
3. Kolmogorov–Smirnov (KS) statistic.
4. Visual comparison of the simulated distribution and fitted Gamma density.

### Main Findings

| Question | Survey comparison | Q95 relative error |
|---|---|---:|
| ry14 | RY25_PreMay10 vs RY25_PostMay10 | 5.03% |
| ry11 | RY_24 vs RY25_PreMay10 | 3.70% |
| ry11 | RY25_PreMay10 vs RY25_PostMay10 | 1.77% |
| ry14 | RY_24 vs RY25_PostMay10 | 1.62% |
| ry10 | RY25_PreMay10 vs RY25_PostMay10 | 1.44% |

The largest Q95 discrepancy was observed for question `ry14`, comparing `RY25_PreMay10` with `RY25_PostMay10`.

For this configuration:

- Valid survey responses: 114,770 and 71,063.
- Empirical Q95: approximately 0.000305.
- Fitted Gamma Q95: approximately 0.000320.
- Relative Q95 error: 5.03%.
- Absolute upper-tail probability discrepancy: approximately 1.33 percentage points.
- KS statistic: 0.0299.

This configuration is a priority candidate for further investigation.

### Important Model Limitation

The analysis uses actual survey response counts, but the conditional simulation treats observed response categories as proxies for latent student types.

This assumption has not yet been established as equivalent to the original Gemfinder model. The comparisons also involve survey cohorts rather than verified individual classroom pairs.

Consequently, these findings should be interpreted as **exploratory survey-derived proxy diagnostics**, not confirmed Gamma approximation failures in real classroom instances.

### Proposed Evaluation Criteria

I propose using Q95 relative error and upper-tail probability error as the primary metrics for evaluating the Gamma approximation and future backup methods. Distribution plots and KS statistics would provide supporting diagnostics.

The team still needs to agree on the final metrics and numerical thresholds for declaring a meaningful Gamma approximation failure.

### Next Steps

- Confirm whether observed response categories can validly represent latent types, or identify the project's approved latent-type estimation method.
- Determine whether the milestone requires actual classroom-level instances rather than survey-cohort comparisons.
- Re-evaluate the highest-error configurations using a model consistent with the original Gemfinder assumptions.
- Agree with the team on evaluation metrics and failure thresholds.

### Output Files

- `gamma_failure_analysis.csv`: results for all 12 configurations.
- `worst_10_gamma_cases.csv`: the 10 configurations with the largest Q95 errors.

The raw survey datasets are not included in the public repository.

I have two questions about the real-instance analysis for the 25 September milestone:

In the original Gemfinder model, can observed survey response categories be used as proxies for latent student types, or is there a specific method for estimating latent type proportions from real survey data?

Does "real instance data" specifically mean comparing actual classrooms, or are comparisons between groups derived from real survey datasets acceptable?

I have already analysed the RY_24 and RY25 survey datasets, but I want to make sure the analysis is consistent with the original model before interpreting the Gamma fitting errors as genuine failure


### October 9 Milestone: Backup Method Implementation

Implemented an empirical Monte Carlo backup estimator as an alternative to the Gamma approximation.

The backup method directly estimates the 95th percentile from simulated values of the test statistic, avoiding the assumption that its distribution follows a Gamma distribution.

An independent evaluation framework was also implemented to compare Gamma and Monte Carlo estimates using separate estimation and reference samples.

**Preliminary survey-derived proxy results:**

| Candidate configuration | Gamma Q95 error | Monte Carlo Q95 error |
|---|---:|---:|
| ry14: RY25 Pre vs Post | 3.02% | 0.53% |
| ry11: RY24 vs RY25 Pre | 0.57% | 0.88% |
| ry11: RY25 Pre vs Post | 1.40% | 0.17% |

Monte Carlo achieved lower Q95 error in two of the three evaluated configurations, while Gamma performed better in one.


### Case 4: Gamma vs Monte Carlo Backup Validation

We evaluated the Gamma approximation and Monte Carlo empirical estimation for Case 4 (`n = 10`, `m = 1000`), using fixed latent-type counts of `[2, 3, 3, 2]` and `[200, 300, 300, 200]`.

The R simulation workflow is implemented in `repeat_case4_validation.R`. The exported `d_hat` simulation draws are stored in `validated_draws/case4_original_model_draws.csv`.

To reproduce the Python comparison, run:

```bash
bash run_case4_comparison.sh
```

The evaluation uses 2,000 draws for Gamma fitting, 2,000 draws for Monte Carlo empirical estimation, and 10,000 reference draws.

**Single-evaluation results:**

| Method | Q95 Relative Error |
|---|---:|
| Gamma approximation | 10.59% |
| Monte Carlo empirical | 5.05% |

In this evaluation, Monte Carlo empirical estimation produced a lower Q95 relative error than the Gamma approximation.

Results are saved in `results/case4_original_model/backup_method_comparison.csv`.

This is a single comparison for Case 4, not evidence that Monte Carlo consistently outperforms Gamma across repeated evaluations or other configurations. Equivalence to the original R simulation implementation should also be verified before treating this as fully validated original-model evidence.

**Limitation:** These configurations use observed survey responses as proxies for latent student types. Therefore, the results are exploratory and do not yet establish performance on validated real classroom failure cases.

**Next step:** Evaluate the backup method on confirmed failure configurations using the original Gemfinder simulation model, with repeated independent runs.
