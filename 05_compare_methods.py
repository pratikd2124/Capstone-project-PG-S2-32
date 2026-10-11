"""Create a defensible comparison report from sensitivity and Bayesian outputs.

The report deliberately avoids plotting EXP002 Gamma means against Bayesian
posterior means as if they were the same estimand.

- EXP002 Gamma mean / 95th percentile summarise a fitted sampling distribution
  for the d_hat difference statistic under the model's test-retest noise setup.
- Bayesian posterior mean / 95% credible interval summarise posterior uncertainty
  in d_hat_norm induced by uncertainty in the Pre/Post response probabilities,
  while rho remains fixed.

The script keeps both sets of quantities in ``method_comparison.csv`` for
reference, but visualises them in separate figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from standalone_common import FIGURES_DIR, RESULTS_DIR, ensure_output_dirs


PROBABILITY_PREFIX = "posterior_prob_dhat_gt_"


def require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run scripts 03 and 04 first.")


def _numeric(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series(np.nan, index=index, dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _coalesce_numeric(
    frame: pd.DataFrame,
    primary: str,
    fallback: str | None = None,
) -> pd.Series:
    result = _numeric(frame[primary] if primary in frame.columns else None, frame.index)
    if fallback and fallback in frame.columns:
        fallback_values = pd.to_numeric(frame[fallback], errors="coerce")
        result = result.where(result.notna(), fallback_values)
    return result


def _coalesce_text(
    frame: pd.DataFrame,
    primary: str,
    fallback: str | None = None,
    default: str = "",
) -> pd.Series:
    if primary in frame.columns:
        result = frame[primary].fillna("").astype(str)
    else:
        result = pd.Series(default, index=frame.index, dtype=object)

    if fallback and fallback in frame.columns:
        fallback_values = frame[fallback].fillna("").astype(str)
        result = result.where(result.str.len() > 0, fallback_values)
    return result


def _combine_status(*values: object) -> str:
    statuses = {
        str(value).upper()
        for value in values
        if value is not None and not pd.isna(value) and str(value).strip()
    }
    if "ERROR" in statuses:
        return "ERROR"
    if "WARN" in statuses:
        return "WARN"
    if "OK" in statuses:
        return "OK"
    return "ERROR"


def _combine_errors(*values: object) -> str:
    parts: list[str] = []
    for value in values:
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if text and text.lower() != "nan" and text not in parts:
            parts.append(text)
    return " | ".join(parts)


def _probability_columns(frame: pd.DataFrame) -> list[str]:
    """Return threshold-probability columns in numeric epsilon order."""

    cols = [c for c in frame.columns if c.startswith(PROBABILITY_PREFIX)]

    def epsilon_value(name: str) -> float:
        suffix = name[len(PROBABILITY_PREFIX) :]
        try:
            return float(suffix.replace("neg_", "-").replace("_", "."))
        except ValueError:
            return float("inf")

    return sorted(cols, key=epsilon_value)


def build_comparison(sample: pd.DataFrame, bayes: pd.DataFrame) -> pd.DataFrame:
    """Merge outputs while preserving OK/WARN/ERROR cases and clear estimands."""

    merged = sample.merge(
        bayes,
        on=["item", "n_pre", "n_post"],
        how="outer",
        suffixes=("_sensitivity", "_bayes"),
    )

    probability_columns = _probability_columns(merged)
    out = pd.DataFrame(index=merged.index)
    out["item"] = merged["item"]
    out["n_pre"] = _coalesce_numeric(merged, "n_pre")
    out["n_post"] = _coalesce_numeric(merged, "n_post")

    for col in [
        "observed_dhat_norm",
        "bayes_dhat_mean",
        "bayes_dhat_median",
        "bayes_dhat_ci_low",
        "bayes_dhat_ci_high",
        "bayes_dhat_sd",
    ]:
        out[col] = _coalesce_numeric(merged, col)

    for col in probability_columns:
        out[col] = _coalesce_numeric(merged, col)

    # Prefer the sample-size sensitivity EXP002 output from script 03. If it is
    # unavailable for a row, fall back to the same EXP002 reference retained by
    # script 04. These are sampling-distribution summaries, not Bayesian means.
    out["gamma_mean"] = _coalesce_numeric(
        merged, "gamma_mean", "exp002_gamma_mean"
    )
    out["gamma_q95"] = _coalesce_numeric(
        merged, "gamma_q95", "exp002_gamma_q95"
    )

    out["outside_training_range_count"] = _coalesce_numeric(
        merged,
        "outside_training_range_count_sensitivity",
        "outside_training_range_count_bayes",
    )
    out["outside_training_range"] = _coalesce_text(
        merged,
        "outside_training_range_sensitivity",
        "outside_training_range_bayes",
    )

    status_sensitivity = (
        merged["status_sensitivity"]
        if "status_sensitivity" in merged.columns
        else pd.Series(np.nan, index=merged.index)
    )
    status_bayes = (
        merged["status_bayes"]
        if "status_bayes" in merged.columns
        else pd.Series(np.nan, index=merged.index)
    )
    out["status"] = [
        _combine_status(a, b) for a, b in zip(status_sensitivity, status_bayes)
    ]

    error_sensitivity = (
        merged["error_sensitivity"]
        if "error_sensitivity" in merged.columns
        else pd.Series("", index=merged.index)
    )
    error_bayes = (
        merged["error_bayes"]
        if "error_bayes" in merged.columns
        else pd.Series("", index=merged.index)
    )
    out["error"] = [
        _combine_errors(a, b) for a, b in zip(error_sensitivity, error_bayes)
    ]

    # Required reporting order: key Bayesian quantities first, then all epsilon
    # probabilities, then EXP002 reference summaries and diagnostic status.
    ordered_columns = [
        "item",
        "n_pre",
        "n_post",
        "observed_dhat_norm",
        "bayes_dhat_mean",
        "bayes_dhat_median",
        "bayes_dhat_ci_low",
        "bayes_dhat_ci_high",
        "bayes_dhat_sd",
        *probability_columns,
        "gamma_mean",
        "gamma_q95",
        "outside_training_range_count",
        "status",
        "outside_training_range",
        "error",
    ]
    return out.reindex(columns=ordered_columns).sort_values(
        ["item", "n_pre", "n_post"], na_position="last"
    )


def _write_summary(
    path: Path,
    comparison: pd.DataFrame,
    matrix_summary: pd.DataFrame,
) -> None:
    probability_columns = _probability_columns(comparison)

    with path.open("w", encoding="utf-8") as f:
        f.write("# Standalone robustness and Bayesian analysis summary\n\n")
        f.write(
            "The analysis is independent of `scrp_toolkit` and reads the original "
            "RYA assets directly. EXP002 and the shipped ONNX network are not modified.\n\n"
        )

        f.write("## Interpretation\n\n")
        f.write(
            "- **Matrix sensitivity** tests whether controlled changes to the fixed "
            "unreliability assumptions change EXP002 output.\n"
        )
        f.write(
            "- **Sample-size sensitivity** tests how EXP002 responds when `n` changes "
            "while the Pre/Post response proportions are kept approximately fixed.\n"
        )
        f.write(
            "- **Bayesian analysis** quantifies posterior uncertainty in the Pre/Post "
            "response-probability distributions and the resulting `d_hat_norm`.\n"
        )
        f.write(
            "- With otherwise similar response proportions, the Bayesian 95% credible "
            "interval is expected to become narrower as sample size increases.\n"
        )
        f.write(
            "- Posterior probabilities depend on the selected epsilon threshold; changing "
            "epsilon changes the minimal-effect statement being evaluated.\n"
        )
        f.write(
            "- **EXP002 Gamma mean and 95th percentile** describe a fitted sampling "
            "distribution for the d_hat difference statistic under the EXP002 model.\n"
        )
        f.write(
            "- **Bayesian posterior mean and 95% credible interval** describe posterior "
            "uncertainty in `d_hat_norm` induced by uncertainty in the Pre/Post response "
            "probabilities.\n"
        )
        f.write(
            "- These EXP002 and Bayesian quantities are **not directly equivalent** and "
            "must not be interpreted as a mean-vs-mean improvement or disagreement.\n"
        )
        f.write(
            "- The current Bayesian analysis is exploratory and is **not yet the complete "
            "Bayesian minimal-effects boundary test**.\n"
        )
        f.write(
            "- Most importantly, the current Bayesian implementation **does not propagate "
            "uncertainty in the unreliability matrix rho**. Rho is supplied as a fixed "
            "matrix because raw test-retest transition counts (or a justified effective "
            "sample size for rho) are not available to this analysis. No transition counts "
            "are invented.\n\n"
        )

        f.write("## Posterior threshold probabilities\n\n")
        if probability_columns:
            f.write(
                "The comparison table contains one `P(d_hat_norm > epsilon)` column for "
                "each epsilon selected in `04_bayesian_minimal_effect.py`:\n\n"
            )
            for col in probability_columns:
                f.write(f"- `{col}`\n")
        else:
            f.write("No posterior threshold-probability columns were found.\n")
        f.write("\n")

        f.write("## Future rho-uncertainty extension\n\n")
        f.write(
            "If genuine row-wise test-retest transition counts become available, a future "
            "extension can place an independent Dirichlet posterior on each row of rho:\n\n"
        )
        f.write("```text\n")
        f.write("rho_j | data ~ Dirichlet(prior + transition_counts_j)\n")
        f.write("```\n\n")
        f.write(
            "Sampled rho matrices could then be propagated through `d_hat_norm`. This is "
            "not done by the current implementation.\n\n"
        )

        f.write("## Sample-size / Bayesian comparison table\n\n```text\n")
        f.write(comparison.to_string(index=False))
        f.write("\n```\n\n")

        f.write("## Matrix sensitivity summary\n\n```text\n")
        f.write(matrix_summary.to_string(index=False))
        f.write("\n```\n")


def main() -> None:
    ensure_output_dirs()

    sample_path = RESULTS_DIR / "sample_size_sensitivity.csv"
    matrix_path = RESULTS_DIR / "matrix_sensitivity_network.csv"
    bayes_path = RESULTS_DIR / "bayesian_dhat_analysis.csv"
    for path in (sample_path, matrix_path, bayes_path):
        require(path)

    sample = pd.read_csv(sample_path)
    matrix = pd.read_csv(matrix_path)
    bayes = pd.read_csv(bayes_path)

    comparison = build_comparison(sample, bayes)
    comparison_path = RESULTS_DIR / "method_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    matrix_cols = [
        "item",
        "percent_change",
        "gamma_mean",
        "gamma_q95",
        "delta_gamma_mean",
        "percent_delta_gamma_mean",
        "delta_gamma_q95",
        "outside_training_range_count",
        "outside_training_range",
        "status",
        "error",
    ]
    matrix_summary = matrix[[c for c in matrix_cols if c in matrix.columns]].copy()
    matrix_summary_path = RESULTS_DIR / "matrix_method_summary.csv"
    matrix_summary.to_csv(matrix_summary_path, index=False)

    # Remove the old misleading mean-vs-mean figure if it remains from a prior run.
    legacy_plot = FIGURES_DIR / "method_comparison.png"
    if legacy_plot.exists():
        legacy_plot.unlink()

    generated: list[tuple[str, str]] = [
        (
            str(comparison_path),
            "CSV: aligned sample-size, Bayesian posterior, threshold-probability, EXP002 reference, and status fields.",
        ),
        (
            str(matrix_summary_path),
            "CSV: concise EXP002 matrix-sensitivity summary, including out-of-training-range diagnostics.",
        ),
    ]

    # Figure 3: EXP002 sampling-distribution summaries only.
    exp_plot = comparison.copy()
    for col in ["n_pre", "gamma_mean", "gamma_q95"]:
        exp_plot[col] = pd.to_numeric(exp_plot[col], errors="coerce")
    exp_good = exp_plot[
        np.isfinite(exp_plot["n_pre"])
        & np.isfinite(exp_plot["gamma_mean"])
        & np.isfinite(exp_plot["gamma_q95"])
    ].sort_values("n_pre")

    if not exp_good.empty:
        x_values = exp_good["n_pre"].to_numpy(dtype=float)
        mean_values = exp_good["gamma_mean"].to_numpy(dtype=float)
        q95_values = exp_good["gamma_q95"].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x_values, mean_values, marker="o", label="EXP002 Gamma mean")
        ax.plot(x_values, q95_values, marker="o", label="EXP002 Gamma 95th percentile")
        ax.set_xscale("log")
        ax.set_xlabel("Pre/Post sample size (log scale)")
        ax.set_ylabel("EXP002 fitted d_hat distribution")
        ax.set_title("EXP002 sample-size sensitivity")
        ax.legend()
        fig.tight_layout()
        exp_fig_path = FIGURES_DIR / "exp002_sample_size.png"
        fig.savefig(exp_fig_path, dpi=160)
        plt.close(fig)
        generated.append(
            (
                str(exp_fig_path),
                "Figure: EXP002 Gamma mean and 95th percentile versus sample size.",
            )
        )

    # Figure 4: Bayesian posterior summaries only. No EXP002 mean is overlaid.
    bayes_plot = comparison.copy()
    for col in [
        "n_pre",
        "observed_dhat_norm",
        "bayes_dhat_mean",
        "bayes_dhat_ci_low",
        "bayes_dhat_ci_high",
    ]:
        bayes_plot[col] = pd.to_numeric(bayes_plot[col], errors="coerce")
    bayes_good = bayes_plot[
        np.isfinite(bayes_plot["n_pre"])
        & np.isfinite(bayes_plot["bayes_dhat_mean"])
        & np.isfinite(bayes_plot["bayes_dhat_ci_low"])
        & np.isfinite(bayes_plot["bayes_dhat_ci_high"])
    ].sort_values("n_pre")

    if not bayes_good.empty:
        x_values = bayes_good["n_pre"].to_numpy(dtype=float)
        mean_values = bayes_good["bayes_dhat_mean"].to_numpy(dtype=float)
        low_values = bayes_good["bayes_dhat_ci_low"].to_numpy(dtype=float)
        high_values = bayes_good["bayes_dhat_ci_high"].to_numpy(dtype=float)
        observed_values = bayes_good["observed_dhat_norm"].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x_values, mean_values, marker="o", label="Bayesian posterior mean")
        ax.fill_between(
            x_values,
            low_values,
            high_values,
            alpha=0.2,
            label="95% Bayesian credible interval",
        )
        if np.isfinite(observed_values).any():
            ax.plot(
                x_values,
                observed_values,
                linestyle=":",
                marker="o",
                label="Observed d_hat_norm",
            )
        ax.set_xscale("log")
        ax.set_xlabel("Pre/Post sample size (log scale)")
        ax.set_ylabel("d_hat_norm")
        ax.set_title("Bayesian response-distribution uncertainty vs sample size")
        ax.legend()
        fig.tight_layout()
        bayes_fig_path = FIGURES_DIR / "bayesian_vs_sample_size.png"
        fig.savefig(bayes_fig_path, dpi=160)
        plt.close(fig)
        generated.append(
            (
                str(bayes_fig_path),
                "Figure: Bayesian posterior mean, observed d_hat_norm, and 95% credible interval versus sample size.",
            )
        )

    summary_path = RESULTS_DIR / "comparison_summary.md"
    _write_summary(summary_path, comparison, matrix_summary)
    generated.append(
        (
            str(summary_path),
            "Markdown: interpretation of matrix sensitivity, sample-size sensitivity, Bayesian uncertainty, and fixed-rho limitation.",
        )
    )

    print(comparison.to_string(index=False))
    print("\nGenerated outputs:")
    for path, description in generated:
        print(f"- {path}\n  {description}")


if __name__ == "__main__":
    main()
