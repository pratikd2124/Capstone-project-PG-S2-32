# October 9 backup-method evaluation — draft

Source status: **external_model_draws_unverified_by_script**.

## Methods
- Gamma: fit a continuous Gamma MLE on 2,000 draws.
- Backup: Monte Carlo empirical q95 and empirical survival probability on 2,000 independent draws.
- Reference: q95 and tail rate from 10,000 **independent** draws.
- Both methods evaluated at the **same reference q95 threshold**.

## Important interpretation limits
- The 9.25 candidate-case simulator uses **observed survey responses as proxies for latent types**. It is not a validated reproduction of the original Gemfinder R model. Therefore any `PROXY` results here are exploratory and **cannot establish real-model Gamma failures or successful repairs**.
- To validate, generate statistics from the original latent-type simulator, then use `--draws-dir` with independent draws for each case.
- The reference distribution is itself Monte Carlo estimated, not a known exact truth. For rare-tail probabilities increase simulation counts and quantify Monte Carlo uncertainty.
- This script does **not** measure simulation-generation runtime per method (only estimation time), so it cannot establish overall computational superiority.
- Candidate cases are selected from the 9.25 analysis; evaluation should also include normal controls and multiple seeds before choosing a method.
- Agreement on failure thresholds and backup-method acceptance criteria remains a team decision.

## Outputs
- `backup_method_comparison.csv`: both methods' q95 and tail errors by case.
- `plots/`: q95-error comparisons.

## Next steps
1. Confirm simulation equivalence with original R model / latent-type sampling.
2. Repeat comparisons using validated simulated draws and several random seeds.
3. Check tail accuracy, computation time, and sample-size sensitivity.
4. Report recommendations for October 16 after evidence is sufficient.
