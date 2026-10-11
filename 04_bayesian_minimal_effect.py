"""Exploratory Bayesian response-distribution uncertainty analysis.

This script places independent Dirichlet posteriors on the four-category
Pre and Post *response-probability vectors*. Posterior draws are propagated
through the existing ``d_hat_norm`` calculation while the supplied
unreliability matrix (rho / P) is held fixed.

This is deliberately NOT described as a full Bayesian GemFinder
minimal-effects test. It is also NOT a Bayesian model of uncertainty in rho.
The current project assets provide rho as a fixed matrix; raw test-retest
transition counts (or another justified effective sample size for rho) are not
invented here.

If transition counts become available later, a defensible extension is to put
an independent Dirichlet posterior on each row of rho and propagate sampled
rho matrices through ``d_hat_norm``. A helper for that future extension is
included below, but it is not used by the current analysis.

EXP002 Gamma summaries are retained in the output CSV only as reference:
- EXP002 Gamma mean / 95th percentile describe the fitted EXP002 sampling
  distribution of the d_hat difference statistic under test-retest noise.
- Bayesian posterior mean / credible interval describe posterior uncertainty
  in d_hat_norm induced by uncertainty in the Pre/Post response probabilities.
These are different statistical quantities and are not plotted as equivalent
means.
"""

from __future__ import annotations

import argparse
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from standalone_common import (
    DEFAULT_ITEM,
    DEFAULT_POST,
    DEFAULT_PRE,
    FIGURES_DIR,
    RESULTS_DIR,
    bayesian_d_hat_draws,
    counts_from_proportions,
    d_hat_norm,
    ensure_output_dirs,
    load_context,
    predict_exp002,
    proportions_from_counts,
)


DEFAULT_EPSILONS = [0.05, 0.25, 0.50, 1.00]


def summarise(draws: np.ndarray) -> dict[str, float]:
    """Summarise the posterior draws of d_hat_norm.

    The 2.5% and 97.5% quantiles are the bounds of the 95% Bayesian credible
    interval for d_hat_norm under the current response-probability model.
    """

    draws = np.asarray(draws, dtype=float)
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        raise ValueError("No finite Bayesian d_hat_norm posterior draws were produced")

    q025, q50, q975 = np.quantile(finite, [0.025, 0.5, 0.975])
    return {
        "bayes_dhat_mean": float(np.mean(finite)),
        "bayes_dhat_median": float(q50),
        "bayes_dhat_sd": float(np.std(finite, ddof=1)),
        # Explicitly labelled as the 95% Bayesian credible interval bounds.
        "bayes_dhat_ci_low": float(q025),
        "bayes_dhat_ci_high": float(q975),
    }


def _epsilon_text(epsilon: float) -> str:
    """Stable human-readable epsilon text with at least two decimal places."""

    text = f"{float(epsilon):.10f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".00"
    else:
        decimals = len(text.split(".", 1)[1])
        if decimals < 2:
            text += "0" * (2 - decimals)
    return text


def probability_column_name(epsilon: float) -> str:
    """Column name such as ``posterior_prob_dhat_gt_0_25``."""

    text = _epsilon_text(epsilon).replace("-", "neg_").replace(".", "_")
    return f"posterior_prob_dhat_gt_{text}"


def validate_epsilons(epsilons: Iterable[float]) -> list[float]:
    values = [float(e) for e in epsilons]
    if not values:
        raise ValueError("Provide at least one epsilon threshold")
    if not np.all(np.isfinite(values)):
        raise ValueError("All epsilon thresholds must be finite")
    if any(e < 0 for e in values):
        raise ValueError("Epsilon thresholds must be non-negative")

    # Preserve user order while removing exact duplicates.
    unique: list[float] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return unique


def posterior_probabilities(draws: np.ndarray, epsilons: Iterable[float]) -> dict[str, float]:
    """Return P(d_hat_norm > epsilon | Pre/Post response data) for each epsilon."""

    draws = np.asarray(draws, dtype=float)
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        raise ValueError("No finite posterior draws available for threshold probabilities")

    return {
        probability_column_name(epsilon): float(np.mean(finite > epsilon))
        for epsilon in epsilons
    }


def sample_rho_posterior_from_transition_counts(
    transition_counts: np.ndarray,
    *,
    draws: int = 20_000,
    prior_alpha: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """Future extension: sample rho only when raw transition counts exist.

    Parameters
    ----------
    transition_counts:
        A real 4x4 matrix of observed test-retest transition counts. Row j
        supplies the data for rho_j.

    Returns
    -------
    ndarray, shape (draws, 4, 4)
        Samples satisfying, row by row,
        rho_j | data ~ Dirichlet(prior + transition_counts_j).

    Notes
    -----
    This function is intentionally NOT called by the current analysis because
    the analysis does not have raw transition counts for rho. No effective
    sample size or pseudo-counts are fabricated.
    """

    counts = np.asarray(transition_counts, dtype=float)
    if counts.shape != (4, 4):
        raise ValueError("transition_counts must be a real 4x4 count matrix")
    if np.any(counts < 0) or not np.all(np.isfinite(counts)):
        raise ValueError("transition_counts must be finite and non-negative")
    if draws < 1:
        raise ValueError("draws must be positive")
    if prior_alpha <= 0:
        raise ValueError("prior_alpha must be positive")

    rng = np.random.default_rng(seed)
    samples = np.empty((draws, 4, 4), dtype=float)
    prior = np.full(4, float(prior_alpha))
    for row_index in range(4):
        samples[:, row_index, :] = rng.dirichlet(
            prior + counts[row_index], size=draws
        )
    return samples


def _empty_row(item: str, n_pre: int, n_post: int, epsilons: list[float]) -> dict:
    row = {
        "item": item,
        "n_pre": int(n_pre),
        "n_post": int(n_post),
        "observed_dhat_norm": np.nan,
        "bayes_dhat_mean": np.nan,
        "bayes_dhat_median": np.nan,
        "bayes_dhat_sd": np.nan,
        "bayes_dhat_ci_low": np.nan,
        "bayes_dhat_ci_high": np.nan,
    }
    for epsilon in epsilons:
        row[probability_column_name(epsilon)] = np.nan
    row.update(
        {
            "exp002_gamma_mean": np.nan,
            "exp002_gamma_q95": np.nan,
            "outside_training_range_count": np.nan,
            "outside_training_range": "",
            "status": "ERROR",
            "error": "",
        }
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exploratory Bayesian uncertainty analysis for Pre/Post response distributions."
    )
    parser.add_argument("--assets-root", default=None)
    parser.add_argument("--item", default=DEFAULT_ITEM)
    parser.add_argument("--pre", nargs=4, type=float, default=DEFAULT_PRE.tolist())
    parser.add_argument("--post", nargs=4, type=float, default=DEFAULT_POST.tolist())
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[20, 50, 100, 250, 500, 1000, 5000],
    )
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument(
        "--prior-alpha",
        type=float,
        default=0.5,
        help="Symmetric Dirichlet prior concentration per Pre/Post response category.",
    )
    parser.add_argument(
        "--epsilons",
        nargs="+",
        type=float,
        default=DEFAULT_EPSILONS,
        help=(
            "d_hat_norm thresholds for posterior probability reporting. "
            "Default: 0.05 0.25 0.50 1.00"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed; sample-size scenario i uses seed + i.",
    )
    args = parser.parse_args()

    epsilons = validate_epsilons(args.epsilons)
    ensure_output_dirs()

    # Remove the old misleading combined plot if it remains from an earlier run.
    legacy_plot = FIGURES_DIR / "bayesian_vs_exp002.png"
    if legacy_plot.exists():
        legacy_plot.unlink()

    ctx = load_context(args.assets_root, require_model=True)
    if args.item not in ctx.p_matrices:
        raise KeyError(args.item)

    # rho / P is fixed throughout this current Bayesian analysis.
    # We do not invent test-retest transition counts or an effective sample size.
    p = np.asarray(ctx.p_matrices[args.item], dtype=float)
    if p.shape != (4, 4):
        raise ValueError("Bayesian d_hat analysis currently supports 4-option items only")

    p_pre = proportions_from_counts(args.pre)
    p_post = proportions_from_counts(args.post)

    rows: list[dict] = []
    for idx, n in enumerate(args.sizes):
        x = counts_from_proportions(p_pre, n)
        y = counts_from_proportions(p_post, n)
        row = _empty_row(args.item, int(x.sum()), int(y.sum()), epsilons)
        errors: list[str] = []

        # Bayesian component: uncertainty is in Pre/Post response probabilities.
        try:
            observed = d_hat_norm(x, y, p, ctx.b_matrix)
            draws = bayesian_d_hat_draws(
                x,
                y,
                p,
                ctx.b_matrix,
                draws=args.draws,
                prior_alpha=args.prior_alpha,
                seed=args.seed + idx,
            )
            row["observed_dhat_norm"] = observed
            row.update(summarise(draws))
            row.update(posterior_probabilities(draws, epsilons))
        except Exception as exc:
            errors.append(f"Bayesian analysis: {exc}")

        # EXP002 is retained only as reference to its fitted sampling distribution.
        # Its Gamma summaries are not interpreted as Bayesian posterior means.
        try:
            exp = predict_exp002(ctx, x, y, args.item, p)
            row["exp002_gamma_mean"] = exp.gamma_mean
            row["exp002_gamma_q95"] = exp.gamma_q95
            row["outside_training_range_count"] = len(exp.outside_training_range)
            row["outside_training_range"] = "; ".join(exp.outside_training_range)
            if exp.outside_training_range:
                row["status"] = "WARN"
            elif not errors:
                row["status"] = "OK"
        except Exception as exc:
            errors.append(f"EXP002 reference: {exc}")

        if errors:
            # An EXP002 failure does not erase valid Bayesian summaries, and vice versa.
            row["status"] = "ERROR"
            row["error"] = " | ".join(errors)

        rows.append(row)

    df = pd.DataFrame(rows)

    # Keep a predictable column order, with one probability column per epsilon.
    probability_columns = [probability_column_name(e) for e in epsilons]
    ordered_columns = [
        "item",
        "n_pre",
        "n_post",
        "observed_dhat_norm",
        "bayes_dhat_mean",
        "bayes_dhat_median",
        "bayes_dhat_sd",
        "bayes_dhat_ci_low",
        "bayes_dhat_ci_high",
        *probability_columns,
        "exp002_gamma_mean",
        "exp002_gamma_q95",
        "outside_training_range_count",
        "outside_training_range",
        "status",
        "error",
    ]
    df = df.reindex(columns=ordered_columns)

    csv_path = RESULTS_DIR / "bayesian_dhat_analysis.csv"
    df.to_csv(csv_path, index=False)

    generated: list[tuple[str, str]] = [
        (
            str(csv_path),
            "CSV: posterior summaries for d_hat_norm, threshold probabilities, and EXP002 reference fields.",
        )
    ]

    # Figure 1: Bayesian uncertainty only. No EXP002 Gamma mean is overlaid.
    plot_df = df.copy()
    numeric_cols = [
        "n_pre",
        "observed_dhat_norm",
        "bayes_dhat_mean",
        "bayes_dhat_median",
        "bayes_dhat_ci_low",
        "bayes_dhat_ci_high",
    ]
    for col in numeric_cols:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    good = plot_df[
        np.isfinite(plot_df["n_pre"])
        & np.isfinite(plot_df["bayes_dhat_mean"])
        & np.isfinite(plot_df["bayes_dhat_ci_low"])
        & np.isfinite(plot_df["bayes_dhat_ci_high"])
    ].sort_values("n_pre")

    if not good.empty:
        x_values = good["n_pre"].to_numpy(dtype=float)
        mean_values = good["bayes_dhat_mean"].to_numpy(dtype=float)
        median_values = good["bayes_dhat_median"].to_numpy(dtype=float)
        low_values = good["bayes_dhat_ci_low"].to_numpy(dtype=float)
        high_values = good["bayes_dhat_ci_high"].to_numpy(dtype=float)
        observed_values = good["observed_dhat_norm"].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x_values, mean_values, marker="o", label="Bayesian posterior mean")
        ax.plot(
            x_values,
            median_values,
            linestyle="--",
            marker="o",
            label="Bayesian posterior median",
        )
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
        ax.set_title(f"{args.item}: Bayesian uncertainty in d_hat_norm")
        ax.legend()
        fig.tight_layout()
        fig1_path = FIGURES_DIR / "bayesian_credible_interval.png"
        fig.savefig(fig1_path, dpi=160)
        plt.close(fig)
        generated.append(
            (
                str(fig1_path),
                "Figure: Bayesian posterior mean/median and 95% credible interval for d_hat_norm versus sample size.",
            )
        )

    # Figure 2: threshold-dependent posterior conclusions.
    prob_good = plot_df[np.isfinite(plot_df["n_pre"])].sort_values("n_pre")
    if not prob_good.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        plotted = False
        x_values = prob_good["n_pre"].to_numpy(dtype=float)
        for epsilon in epsilons:
            col = probability_column_name(epsilon)
            values = pd.to_numeric(prob_good[col], errors="coerce").to_numpy(dtype=float)
            finite_mask = np.isfinite(x_values) & np.isfinite(values)
            if not np.any(finite_mask):
                continue
            ax.plot(
                x_values[finite_mask],
                values[finite_mask],
                marker="o",
                label=f"epsilon = {_epsilon_text(epsilon)}",
            )
            plotted = True

        if plotted:
            ax.set_xscale("log")
            ax.set_ylim(0.0, 1.0)
            ax.set_xlabel("Pre/Post sample size (log scale)")
            ax.set_ylabel("Posterior probability P(d_hat_norm > epsilon)")
            ax.set_title(f"{args.item}: posterior probability by minimal-effect threshold")
            ax.legend()
            fig.tight_layout()
            fig2_path = FIGURES_DIR / "bayesian_threshold_probability.png"
            fig.savefig(fig2_path, dpi=160)
            generated.append(
                (
                    str(fig2_path),
                    "Figure: posterior P(d_hat_norm > epsilon) versus sample size for each selected epsilon.",
                )
            )
        plt.close(fig)

    print(df.to_string(index=False))
    print("\nGenerated outputs:")
    for path, description in generated:
        print(f"- {path}\n  {description}")
    print(
        "\nStatistical scope: the Dirichlet posterior is on Pre/Post response probabilities only; "
        "rho is fixed because no test-retest transition counts are supplied."
    )


if __name__ == "__main__":
    main()
