# GemFinder Statistical Core Refinement — Final Pipeline V2 Summary

## 1. What data were analysed?

The pipeline analysed **113 historical subgroup comparisons** from the supplied RY25 survey data using four-response items that have usable 4x4 unreliability matrices. Raw source files and the ONNX model were read only.

## 2. What comparisons were created?

Each row compares two real historical subgroups using grouping column(s) `grade`. Observed subgroup pair(s): 4 vs 7, 11 vs 6, 12 vs 5. The pipeline does not aggregate the entire Pre file against the entire Post file.

## 3. How many passed screening?

SAFE: **109**, WARN: **4**, REJECT: **0**. SAFE/WARN/REJECT is an input-compatibility status, not a significance result.

## 4. Why were comparisons flagged?

- `rho_diagonal`: 3
- `euclid_dist_XY`: 1

## 5. How sensitive was EXP002 to rho?

For representative comparison `post_grade_11_vs_6_ry9`, the baseline Gamma mean was **78.9738**. Across the configured rho perturbations, the Gamma-mean range was **19.9293**. See Figure 3 for direction and magnitude.

## 6. What did Bayesian response uncertainty show?

Exact Dirichlet posteriors with Monte Carlo propagation produced 95% credible intervals for **113** comparisons. The median interval width was **0.0309153**. These intervals quantify response-distribution uncertainty while rho is fixed.

## 7. Could Bayesian rho uncertainty be run?

No. Genuine paired test-retest transition counts were not available, so rho uncertainty is **not identifiable from the supplied data**. The code does not fabricate pseudo-counts or reinterpret Pre/Post data as test-retest transitions.

## 8. What does the historical benchmark show?

The benchmark keeps EXP002 Gamma summaries and Bayesian d_hat_norm posterior summaries separate because they are different statistical objects. Median EXP002 runtime was **0.8133 ms** per comparison. Median Bayesian response-evaluation runtime was **40.02 ms** per comparison. Training-range compatibility and routing are recorded for every historical comparison.

## 9. What requirements are complete?

| Requirement | Status |
|---|---|
| Real rho matrix validation | COMPLETE |
| Compatibility screening | COMPLETE |
| Existing EXP002 inference | COMPLETE |
| Fallback routing | COMPLETE; fallback implementation EXTERNAL |
| Rho sensitivity | COMPLETE |
| Sample-size sensitivity | COMPLETE |
| Historical subgroup benchmarking | COMPLETE |
| Bayesian response uncertainty | COMPLETE |
| Bayesian posterior credible intervals | COMPLETE |
| Diagnostic threshold probabilities | COMPLETE — exploratory only |
| Bayesian rho uncertainty | DATA BLOCKED |
| Alpha/beta uncertainty from rho | DATA BLOCKED |
| Official Bayesian minimal-effect decision | WAITING FOR OFFICIAL EPSILON |
| Existing-vs-Bayesian decision agreement | NOT CALCULATED — existing binary decision rule not supplied |
| Runtime comparison | COMPLETE |
| Variational inference | NOT IMPLEMENTED — exact conjugate posterior is available for current Dirichlet components |

## 10. What remains future work/data-dependent?

The real team-owned fallback inference method still needs integration. Bayesian rho uncertainty requires genuine test-retest transition counts if those are not supplied. An official minimal-effect epsilon is still required before any Bayesian minimal-effect decision can be labelled official. Existing-vs-Bayesian decision agreement also requires the exact existing binary decision rule; it is not inferred from Gamma means. Full variational inference is not claimed because the current Dirichlet-multinomial components have exact conjugate posteriors.
