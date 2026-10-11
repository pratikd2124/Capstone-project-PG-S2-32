# Standalone robustness and Bayesian analysis summary

The analysis is independent of `scrp_toolkit` and reads the original RYA assets directly. EXP002 and the shipped ONNX network are not modified.

## Interpretation

- **Matrix sensitivity** tests whether controlled changes to the fixed unreliability assumptions change EXP002 output.
- **Sample-size sensitivity** tests how EXP002 responds when `n` changes while the Pre/Post response proportions are kept approximately fixed.
- **Bayesian analysis** quantifies posterior uncertainty in the Pre/Post response-probability distributions and the resulting `d_hat_norm`.
- With otherwise similar response proportions, the Bayesian 95% credible interval is expected to become narrower as sample size increases.
- Posterior probabilities depend on the selected epsilon threshold; changing epsilon changes the minimal-effect statement being evaluated.
- **EXP002 Gamma mean and 95th percentile** describe a fitted sampling distribution for the d_hat difference statistic under the EXP002 model.
- **Bayesian posterior mean and 95% credible interval** describe posterior uncertainty in `d_hat_norm` induced by uncertainty in the Pre/Post response probabilities.
- These EXP002 and Bayesian quantities are **not directly equivalent** and must not be interpreted as a mean-vs-mean improvement or disagreement.
- The current Bayesian analysis is exploratory and is **not yet the complete Bayesian minimal-effects boundary test**.
- Most importantly, the current Bayesian implementation **does not propagate uncertainty in the unreliability matrix rho**. Rho is supplied as a fixed matrix because raw test-retest transition counts (or a justified effective sample size for rho) are not available to this analysis. No transition counts are invented.

## Posterior threshold probabilities

The comparison table contains one `P(d_hat_norm > epsilon)` column for each epsilon selected in `04_bayesian_minimal_effect.py`:

- `posterior_prob_dhat_gt_0_05`
- `posterior_prob_dhat_gt_0_25`
- `posterior_prob_dhat_gt_0_50`
- `posterior_prob_dhat_gt_1_00`

## Future rho-uncertainty extension

If genuine row-wise test-retest transition counts become available, a future extension can place an independent Dirichlet posterior on each row of rho:

```text
rho_j | data ~ Dirichlet(prior + transition_counts_j)
```

Sampled rho matrices could then be propagated through `d_hat_norm`. This is not done by the current implementation.

## Sample-size / Bayesian comparison table

```text
item  n_pre  n_post  observed_dhat_norm  bayes_dhat_mean  bayes_dhat_median  bayes_dhat_ci_low  bayes_dhat_ci_high  bayes_dhat_sd  posterior_prob_dhat_gt_0_05  posterior_prob_dhat_gt_0_25  posterior_prob_dhat_gt_0_50  posterior_prob_dhat_gt_1_00  gamma_mean  gamma_q95  outside_training_range_count status outside_training_range error
 ry9     20      20            1.237916         1.192172           1.134544           0.363586            2.329570       0.512283                       0.9998                       0.9922                      0.93365                      0.60620    0.588023   1.338628                             0     OK                             
 ry9     50      50            1.295036         1.272151           1.251207           0.614468            2.043524       0.367351                       1.0000                       0.9998                      0.99245                      0.76280    0.518531   0.920956                             0     OK                             
 ry9    100     100            1.306392         1.293250           1.283218           0.795122            1.841773       0.270220                       1.0000                       1.0000                      0.99960                      0.86235    0.479434   0.746787                             0     OK                             
 ry9    250     250            1.298694         1.293875           1.288422           0.965894            1.645293       0.171528                       1.0000                       1.0000                      1.00000                      0.96180    0.463928   0.625897                             0     OK                             
 ry9    500     500            1.276803         1.275143           1.273464           1.041069            1.516243       0.122030                       1.0000                       1.0000                      1.00000                      0.98960    0.451525   0.563116                             0     OK                             
 ry9   1000    1000            1.272736         1.272144           1.271116           1.105091            1.445433       0.086793                       1.0000                       1.0000                      1.00000                      0.99935    0.433559   0.512000                             0     OK                             
 ry9   5000    5000            1.274691         1.274834           1.274442           1.198497            1.353245       0.039358                       1.0000                       1.0000                      1.00000                      1.00000    0.377573   0.438145                             0     OK                             
 ry9  10000   10000                 NaN              NaN                NaN                NaN                 NaN            NaN                          NaN                          NaN                          NaN                          NaN    0.346459   0.401526                             2   WARN               n_X; n_Y      
 ry9  20000   20000                 NaN              NaN                NaN                NaN                 NaN            NaN                          NaN                          NaN                          NaN                          NaN    0.120345   0.140949                             2   WARN               n_X; n_Y      
```

## Matrix sensitivity summary

```text
item  percent_change  gamma_mean  gamma_q95  delta_gamma_mean  percent_delta_gamma_mean  delta_gamma_q95  outside_training_range_count outside_training_range status  error
 ry9           -10.0    0.552897   0.841941          0.076444                 16.044323         0.093227                             0                    NaN     OK    NaN
 ry9            -5.0    0.508683   0.789168          0.032229                  6.764459         0.040454                             0                    NaN     OK    NaN
 ry9            -1.0    0.477858   0.749364          0.001404                  0.294727         0.000650                             0                    NaN     OK    NaN
 ry9             0.0    0.469681   0.738463         -0.006772                 -1.421432        -0.010250                             0                    NaN     OK    NaN
 ry9             1.0    0.462921   0.730183         -0.013532                 -2.840156        -0.018530                             0                    NaN     OK    NaN
 ry9             5.0    0.438447   0.700688         -0.038007                 -7.976979        -0.048025                             1                    P_1   WARN    NaN
 ry9            10.0    0.405287   0.659991         -0.071167                -14.936740        -0.088722                             1                    P_1   WARN    NaN
```
