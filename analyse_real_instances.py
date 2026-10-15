"""Survey-derived Gemfinder Gamma diagnostics; see README before interpreting results.

This is a conditional retest PROXY: observed response categories are used as
latent types. Original R simulator instead samples latent types. This assumption
requires project approval before claiming genuine real-instance failure.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gamma
from gamma_fit_evaluation import evaluate_gamma_fit

# EDIT THESE SETTINGS AFTER RUNNING THE INSPECTION CELL IN THE NOTEBOOK.
DATA_DIR = Path("data")
P_MATRIX_FILE = Path("P.RData")
OUTPUT_DIR = Path("results_1")
QUESTION_OPTIONS = {
    "ry10": [1, 2, 3, 4],
    "ry11": [1, 2, 3, 4],
    "ry13": [1, 2, 3, 4],
    "ry14": [1, 2, 3, 4]
}
NUM_TRIALS = 10000
SEED = 2026
COMPARE = [("RY_24", "RY25_PreMay10"),
           ("RY_24", "RY25_PostMay10"),
           ("RY25_PreMay10", "RY25_PostMay10")]
SURVEY_FILES = {
    "RY_24": "RY_24.csv",
    "RY25_PreMay10": "RY25_PreMay10.csv",
    "RY25_PostMay10": "RY25_PostMay10.csv",
}


def load_p_matrices(path):
    import rdata

    P = rdata.read_rds(str(path))
    matrices = {}

    for question, value in P.items():
        matrix = np.asarray(value, dtype=float)

        if matrix.shape != (4, 4):
            continue

        if not np.isfinite(matrix).all():
            raise ValueError(f"{question}: invalid values")

        if (matrix < 0).any():
            raise ValueError(f"{question}: negative probabilities")

        row_sums = matrix.sum(axis=1)

# Allow small discrepancies consistent with rounded probabilities
        if not np.allclose(row_sums, 1.0, atol=0.021, rtol=0):
                raise ValueError(
                    f"{question}: row sums too far from 1: {row_sums}"
                )

# Normalize each row so multinomial probabilities sum to exactly 1
        matrix = matrix / row_sums[:, None]

        matrices[str(question)] = matrix

    return matrices


def response_counts(df, question, options):
    """No demographic identifiers are used; missing and invalid answers excluded."""
    if question not in df:
        return None, 0
    s = pd.to_numeric(df[question], errors="coerce")
    valid = s.isin(options)
    counts = s[valid].value_counts().reindex(options, fill_value=0)
    return counts.to_numpy(dtype=int), int((~valid).sum())


def projection_matrix(k=4):
    """Orthonormal basis of sum-to-zero space, equivalent for quadratic form
    to the B Gram-Schmidt projection used in the R code."""
    basis = np.zeros((k - 1, k))
    for i in range(1, k):
        basis[i-1, :i] = 1 / np.sqrt(i * (i + 1))
        basis[i-1, i] = -i / np.sqrt(i * (i + 1))
    return basis


def simulated_statistics(x_counts, y_counts, p, trials, rng):
    """Conditional retest proxy for one observed survey-derived instance.

    The original R formula uses sigma_hat=sum_j (X_j/n² + Y_j/m²) sigma_j,
    B projection, and 0.5*(1/n+1/m) scaling. Its latent-type model is NOT
    established by raw survey response counts; that substitution is provisional.
    """
    n, m = int(x_counts.sum()), int(y_counts.sum())
    if n < 2 or m < 2:
        raise ValueError("Too few valid responses")
    b = projection_matrix(4)
    sigma = np.array([np.diag(row) - np.outer(row, row) for row in p])
    # Draw retest response counts from the observed-category proxy for latent types
    draws_x = np.zeros((trials, 4), dtype=np.int64)
    draws_y = np.zeros((trials, 4), dtype=np.int64)
    for j in range(4):
        if x_counts[j]:
            draws_x += rng.multinomial(int(x_counts[j]), p[j], size=trials)
        if y_counts[j]:
            draws_y += rng.multinomial(int(y_counts[j]), p[j], size=trials)
    result = np.empty(trials, dtype=float)
    for t in range(trials):
        x, y = draws_x[t].astype(float), draws_y[t].astype(float)
        weights = x / n**2 + y / m**2
        sigma_hat = np.einsum("j,jab->ab", weights, sigma)
        inner = b @ sigma_hat @ b.T
        difference = b @ (x / n - y / m)
        try:
            result[t] = float(difference @ np.linalg.solve(inner, difference))
        except np.linalg.LinAlgError:
            raise ValueError("Singular covariance in a simulation trial")
    result *= 0.5 * (1/n + 1/m)
    return result


def save_case_plot(draws, row, path):
    x = np.linspace(max(0, float(draws.min())), float(draws.max()), 500)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(draws, bins=55, density=True, alpha=0.55, label="Simulated statistic")
    ax.plot(x, gamma.pdf(x, a=row["shape"], scale=1/row["rate"]),
            label="Gamma fit")
    ax.axvline(row["empirical_q95"], linestyle="--", label="Empirical q95")
    ax.axvline(row["gamma_q95"], linestyle=":", label="Gamma q95")
    ax.set(title=f"{row['instance_id']} | q95 error={row['q95_error']:.1%}",
           xlabel="Scaled d_hat", ylabel="Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plots = OUTPUT_DIR / "gamma_fit_plots"
    plots.mkdir(exist_ok=True)
    if not QUESTION_OPTIONS:
        raise ValueError("Set QUESTION_OPTIONS after verifying the scoring instructions")
    matrices = load_p_matrices(P_MATRIX_FILE)
    surveys = {}
    for name, filename in SURVEY_FILES.items():
        path = DATA_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing survey file: {path}")
        surveys[name] = pd.read_csv(path, low_memory=False)
    rng = np.random.default_rng(SEED)
    results, plot_data = [], {}
    for question, options in QUESTION_OPTIONS.items():
        if list(options) != [1, 2, 3, 4]:
            raise ValueError(f"{question}: this R-equivalent pipeline needs 1,2,3,4")
        if question not in matrices:
            print(f"SKIP {question}: no matching P matrix")
            continue
        p = matrices[question]
        for a, b in COMPARE:
            x, missing_x = response_counts(surveys[a], question, options)
            y, missing_y = response_counts(surveys[b], question, options)
            if x is None or y is None:
                print(f"SKIP {question}: missing column in {a} or {b}")
                continue
            ident = f"{question}__{a}__vs__{b}"
            row = {"instance_id": ident, "question": question,
                   "group_x": a, "group_y": b,
                   "n": int(x.sum()), "m": int(y.sum()),
                   "excluded_x": missing_x, "excluded_y": missing_y,
                   "size_ratio": max(x.sum()/max(y.sum(),1), y.sum()/max(x.sum(),1)),
                   "x_counts": json.dumps(x.tolist()), "y_counts": json.dumps(y.tolist()),
                   "instance_source": "observed_survey_counts",
                   "simulation_model": "observed_response_as_latent_type_PROXY"}
            try:
                draws = simulated_statistics(x, y, p, NUM_TRIALS, rng)
                metrics = evaluate_gamma_fit(draws, seed=SEED)
                row.update(metrics)
                row["status"] = "evaluated_proxy"
                plot_data[ident] = draws
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                row["status"] = "needs_review: " + str(exc)
            results.append(row)
            print(ident, row["status"])
    if not results:
        raise RuntimeError("No matching questions/instances. Check names and P matrices")
    table = pd.DataFrame(results).sort_values("q95_error", ascending=False,
                                              na_position="last")
    table.to_csv(OUTPUT_DIR / "gamma_failure_analysis.csv", index=False)
    valid = table[table["status"] == "evaluated_proxy"]
    valid.head(10).to_csv(OUTPUT_DIR / "worst_10_gamma_cases.csv", index=False)
    for _, row in valid.head(3).iterrows():
        save_case_plot(plot_data[row["instance_id"]], row,
                       plots / f"{row['instance_id']}.png")
    if not valid.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(valid["size_ratio"], valid["q95_error"], alpha=0.6)
        ax.set(xlabel="Group size ratio", ylabel="q95 relative error",
               title="Gamma fit error vs group size imbalance")
        fig.tight_layout()
        fig.savefig(plots / "size_ratio_vs_q95_error.png", dpi=160)
        plt.close(fig)
    report = f"""# Gamma evaluation criteria and findings (draft)

## Scope and source

Survey files: RY_24.csv, RY25_PreMay10.csv, RY25_PostMay10.csv.
Instances: aggregated response counts for configured four-option questions.
Matching P matrices: exported from the project's P.RData.

**IMPORTANT MODEL LIMITATION:** The original R script samples latent student types.
This provisional Python pipeline instead uses observed response categories as
latent types when simulating retests. These are not proven equivalent. Results
are *exploratory survey-derived proxy diagnostics*, NOT validated evidence of
"genuinely poor Gamma fit in real instances" until the team approves the
conditional simulation model or supplies a faithful real-instance simulator.

## Evaluation protocol

- {NUM_TRIALS:,} simulated scaled d_hat draws per instance.
- 70% for Gamma MLE (shape/rate), 30% held out for evaluation.
- KS statistic: maximum empirical vs fitted CDF discrepancy on holdout.
- q95 relative error: |fitted q95 - empirical q95| / empirical q95.
- Tail error: absolute fitted vs empirical survival probability at empirical q95.
- Exact zeros, invalid draws, singular covariance: flagged for review, not discarded.
- Failure thresholds: **to be agreed by team**, not asserted automatically.

## Results

Instances attempted: {len(table)}. Successfully evaluated: {len(valid)}.
See `gamma_failure_analysis.csv` and `worst_10_gamma_cases.csv`.
See `gamma_fit_plots/` for plots of the three largest q95 errors.

## Findings (fill in after inspection)

- Which questions/group comparisons have the largest errors?
- Are errors associated with group size imbalance, rare categories, or P structure?
- Which cases remain poor under repeated seeds or more trials?
- Are the simulation assumptions faithful to the original Gemfinder method?

## Notes

KS is descriptive; ordinary KS p-values are not calibrated after fitting parameters.
Group comparisons are not assumed independent if survey respondent overlap exists.
"""
    (OUTPUT_DIR / "evaluation_criteria_and_findings.md").write_text(report, encoding="utf-8")
    print("Saved results to", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
