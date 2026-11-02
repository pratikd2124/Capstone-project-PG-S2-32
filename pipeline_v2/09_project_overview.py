"""Create two simple project-status figures for the GemFinder capstone.

Place this file inside ``pipeline_v2/`` and run:

    python 09_project_overview.py

It reads the existing pipeline outputs in ``results/`` and creates:

    figures/final/07_project_scope_overview.png
    figures/final/08_bayesian_analysis_overview.png

The script deliberately does NOT use EXP002 Gamma mean/q95 values in these
summary figures. Those values should only be presented after the EXP002
rate/beta interpretation has been fully validated against the original model.
"""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures" / "final"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

FINAL_RESULTS = RESULTS_DIR / "final_pipeline_results.csv"
SCREENING_RESULTS = RESULTS_DIR / "screening_results.csv"
RHO_SENSITIVITY = RESULTS_DIR / "rho_sensitivity.csv"
RHO_SUMMARY = RESULTS_DIR / "bayesian_rho_summary.md"


def require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing required file: {path}\n"
            "Run the main pipeline first so the result CSVs exist."
        )


def status_from_outputs(final_df: pd.DataFrame, screening_df: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Build a simple scope/progress map from real outputs plus known scope gaps."""

    n_total = len(final_df)
    n_safe = int((screening_df["screening_status"] == "SAFE").sum())
    n_warn = int((screening_df["screening_status"] == "WARN").sum())
    n_reject = int((screening_df["screening_status"] == "REJECT").sum())

    bayes_ok = int((final_df.get("bayes_dhat_mean").notna()).sum())
    rho_sens_done = RHO_SENSITIVITY.is_file() and len(pd.read_csv(RHO_SENSITIVITY)) > 0

    fallback_used = False
    if "fallback_used" in final_df.columns:
        fallback_used = final_df["fallback_used"].fillna(False).astype(bool).any()

    official_epsilon_available = False
    if "official_epsilon" in final_df.columns:
        official_epsilon_available = final_df["official_epsilon"].notna().any()

    rho_uncertainty_available = False
    if RHO_SUMMARY.is_file():
        text = RHO_SUMMARY.read_text(encoding="utf-8", errors="ignore").lower()
        rho_uncertainty_available = (
            "not identifiable" not in text
            and "data blocked" not in text
            and "unavailable" not in text
        )

    rows = [
        (
            "Historical subgroup comparisons",
            "Implemented" if n_total > 0 else "Remaining",
            f"{n_total} real comparisons prepared",
        ),
        (
            "Rho validation + input screening",
            "Implemented" if len(screening_df) == n_total and n_total > 0 else "Partial",
            f"SAFE={n_safe}, WARN={n_warn}, REJECT={n_reject}",
        ),
        (
            "EXP002 routing prototype",
            "Implemented" if "inference_path" in final_df.columns else "Partial",
            "SAFE cases use EXP002; flagged cases are routed",
        ),
        (
            "Rho sensitivity analysis",
            "Implemented" if rho_sens_done else "Remaining",
            "Tests how assumptions about reliability change inference",
        ),
        (
            "Bayesian response uncertainty",
            "Implemented" if bayes_ok == n_total and n_total > 0 else "Partial",
            f"{bayes_ok}/{n_total} comparisons have posterior results",
        ),
        (
            "Gamma-compatibility classifier",
            "Partial",
            "Current screening is rule-based; trained classifier remains",
        ),
        (
            "Fallback inference method",
            "Implemented" if fallback_used else "Partial",
            "Routing exists; final fallback method not yet integrated",
        ),
        (
            "Bayesian rho uncertainty",
            "Implemented" if rho_uncertainty_available else "Data blocked",
            "Needs genuine test-retest transition counts",
        ),
        (
            "Official decision agreement",
            "Implemented" if official_epsilon_available else "Data blocked",
            "Needs official minimal-effect threshold / decision rule",
        ),
    ]
    return rows


def create_scope_overview(final_df: pd.DataFrame, screening_df: pd.DataFrame) -> Path:
    rows = status_from_outputs(final_df, screening_df)

    # Categorical x-axis: the numeric positions are only display positions.
    status_position = {
        "Data blocked": 0,
        "Remaining": 1,
        "Partial": 2,
        "Implemented": 3,
    }

    labels = [r[0] for r in rows]
    statuses = [r[1] for r in rows]
    notes = [r[2] for r in rows]
    x = [status_position[s] for s in statuses]
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.scatter(x, y, s=150)

    for xi, yi, status, note in zip(x, y, statuses, notes):
        ax.text(xi + 0.08, yi, f"{status} — {note}", va="center", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["Data blocked", "Remaining", "Partial", "Implemented"])
    ax.set_xlim(-0.25, 5.6)
    ax.invert_yaxis()
    ax.set_title("GemFinder project progress: implemented evidence and remaining work")
    ax.set_xlabel("Current status")
    ax.grid(axis="x", alpha=0.25)

    fig.text(
        0.01,
        0.01,
        "Note: EXP002 Gamma mean/q95 values are intentionally not summarised here until the rate/beta interpretation is fully validated.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    out = FIGURES_DIR / "07_project_scope_overview.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def representative_bayesian_rows(df: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    """Choose rows spread across the observed d_hat range, not just the largest values."""
    work = df.copy()
    needed = [
        "comparison_id",
        "observed_dhat_norm",
        "bayes_dhat_mean",
        "bayes_dhat_ci_low",
        "bayes_dhat_ci_high",
    ]
    work = work.dropna(subset=needed).sort_values("observed_dhat_norm").reset_index(drop=True)
    if len(work) <= n:
        return work
    idx = np.round(np.linspace(0, len(work) - 1, n)).astype(int)
    return work.iloc[idx].drop_duplicates().reset_index(drop=True)


def short_label(comparison_id: str) -> str:
    text = str(comparison_id)
    text = text.replace("post_grade_", "G")
    text = text.replace("_vs_", " vs ")
    # Example: G4 vs 7_ry9 -> ry9 | G4 vs 7
    if "_" in text:
        prefix, item = text.rsplit("_", 1)
        return f"{item} | {prefix}"
    return text


def create_bayesian_overview(final_df: pd.DataFrame) -> Path:
    rep = representative_bayesian_rows(final_df, n=12)
    if rep.empty:
        raise ValueError("No usable Bayesian posterior rows were found.")

    labels = [short_label(v) for v in rep["comparison_id"]]
    y = np.arange(len(rep))

    mean = rep["bayes_dhat_mean"].to_numpy(float)
    low = rep["bayes_dhat_ci_low"].to_numpy(float)
    high = rep["bayes_dhat_ci_high"].to_numpy(float)
    observed = rep["observed_dhat_norm"].to_numpy(float)

    lower_err = np.maximum(mean - low, 0)
    upper_err = np.maximum(high - mean, 0)

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.errorbar(
        mean,
        y,
        xerr=np.vstack([lower_err, upper_err]),
        fmt="o",
        capsize=3,
        label="Bayesian posterior mean + 95% credible interval",
    )
    ax.scatter(observed, y, marker="x", s=65, label="Observed d_hat_norm")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("d_hat_norm")
    ax.set_title("What the Bayesian analysis adds: uncertainty around observed group differences")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")

    n_total = len(final_df)
    n_bayes = int(final_df["bayes_dhat_mean"].notna().sum())
    median_width = float(
        (final_df["bayes_dhat_ci_high"] - final_df["bayes_dhat_ci_low"]).dropna().median()
    )

    note = (
        f"What has been solved\n"
        f"• Bayesian response uncertainty: {n_bayes}/{n_total} comparisons\n"
        f"• Median 95% interval width: {median_width:.4f}\n"
        f"• Exact Dirichlet posterior + Monte Carlo propagation\n\n"
        f"Current limitations\n"
        f"• rho is treated as fixed\n"
        f"• rho posterior needs genuine test-retest transition counts\n"
        f"• no official epsilon / production decision rule supplied\n"
        f"• diagnostic thresholds are exploratory, not official"
    )

    ax.text(
        1.02,
        0.98,
        note,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.5", alpha=0.08),
    )

    fig.tight_layout(rect=(0, 0, 0.77, 1))
    out = FIGURES_DIR / "08_bayesian_analysis_overview.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    require(FINAL_RESULTS)
    require(SCREENING_RESULTS)

    final_df = pd.read_csv(FINAL_RESULTS)
    screening_df = pd.read_csv(SCREENING_RESULTS)

    scope_path = create_scope_overview(final_df, screening_df)
    bayes_path = create_bayesian_overview(final_df)

    print("Created:")
    print(f"  {scope_path}")
    print(f"  {bayes_path}")
    print()
    print("Interpretation:")
    print("- Figure 07 shows what is implemented, partial, or data-blocked.")
    print("- Figure 08 shows what Bayesian response uncertainty contributes and its limitations.")
    print("- These figures do not claim that synthetic/dummy test-retest data are real evidence.")


if __name__ == "__main__":
    main()
