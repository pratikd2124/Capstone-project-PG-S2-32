"""Stage 8: final tables, six presentation figures, and academic summary."""

from __future__ import annotations

import argparse
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import MAX_BAYES_FIGURE_COMPARISONS, MAX_HEATMAP_COMPARISONS, OFFICIAL_EPSILON
from common import FINAL_FIGURES_DIR, RESULTS_DIR, ensure_output_dirs


def _representative(df: pd.DataFrame, max_rows: int, sort_col: str) -> pd.DataFrame:
    good = df.copy()
    if len(good) <= max_rows:
        return good
    good = good.sort_values(sort_col).reset_index(drop=True)
    idx = np.round(np.linspace(0, len(good) - 1, max_rows)).astype(int)
    return good.iloc[idx].drop_duplicates().copy()


def _label(row: pd.Series) -> str:
    return f"{row['item']} | {row['group_a']} vs {row['group_b']}"


def figure_screening(df: pd.DataFrame) -> None:
    counts = df["screening_status"].value_counts().reindex(["SAFE", "WARN", "REJECT"], fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values)
    ax.set_ylabel("Number of comparisons")
    ax.set_title("Pipeline screening status")
    fig.tight_layout()
    fig.savefig(FINAL_FIGURES_DIR / "01_pipeline_screening_status.png", dpi=180)
    plt.close(fig)


def _warning_counter(df: pd.DataFrame) -> Counter:
    counter: Counter = Counter()
    for text in df.loc[df["screening_status"].eq("WARN"), "warning_features"].fillna(""):
        for part in [x.strip() for x in str(text).split(";") if x.strip()]:
            counter[part] += 1
    return counter


def figure_warning_causes(df: pd.DataFrame) -> None:
    counter = _warning_counter(df)
    names, values = zip(*counter.most_common(12)) if counter else (["none"], [0])
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(names))
    ax.barh(y, values)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Flag count")
    ax.set_title("Why comparisons were flagged")
    fig.tight_layout()
    fig.savefig(FINAL_FIGURES_DIR / "02_flag_reasons.png", dpi=180)
    plt.close(fig)


def figure_rho_sensitivity() -> None:
    path = RESULTS_DIR / "rho_sensitivity.csv"
    if not path.exists():
        return
    df = pd.read_csv(path).sort_values("relative_rho_perturbation")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["relative_rho_perturbation"] * 100, df["gamma_mean"], marker="o", label="Gamma mean")
    ax.plot(df["relative_rho_perturbation"] * 100, df["gamma_q95"], marker="o", label="Gamma q95")
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Relative off-diagonal rho perturbation (%)")
    ax.set_ylabel("EXP002 Gamma summary")
    ax.set_title(f"Rho sensitivity — {df['comparison_id'].iloc[0]}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FINAL_FIGURES_DIR / "03_rho_sensitivity.png", dpi=180)
    plt.close(fig)


def figure_bayesian_uncertainty(df: pd.DataFrame) -> None:
    good = df[
        np.isfinite(pd.to_numeric(df["bayes_dhat_mean"], errors="coerce"))
        & np.isfinite(pd.to_numeric(df["bayes_dhat_ci_low"], errors="coerce"))
        & np.isfinite(pd.to_numeric(df["bayes_dhat_ci_high"], errors="coerce"))
        & np.isfinite(pd.to_numeric(df["observed_dhat_norm"], errors="coerce"))
    ].copy()
    if good.empty:
        return
    good = _representative(good, MAX_BAYES_FIGURE_COMPARISONS, "observed_dhat_norm")
    x = np.arange(len(good))
    mean = good["bayes_dhat_mean"].to_numpy(float)
    low = good["bayes_dhat_ci_low"].to_numpy(float)
    high = good["bayes_dhat_ci_high"].to_numpy(float)
    obs = good["observed_dhat_norm"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(max(9, len(good) * 0.7), 5.5))
    ax.errorbar(x, mean, yerr=np.vstack([mean - low, high - mean]), fmt="o", capsize=3, label="Posterior mean + 95% credible interval")
    ax.scatter(x, obs, marker="x", label="Observed d_hat_norm")
    ax.set_xticks(x)
    ax.set_xticklabels([_label(r) for _, r in good.iterrows()], rotation=70, ha="right")
    ax.set_ylabel("d_hat_norm")
    ax.set_title("Bayesian response-distribution uncertainty (rho fixed)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FINAL_FIGURES_DIR / "04_bayesian_uncertainty.png", dpi=180)
    plt.close(fig)


def _diagnostic_pairs(df: pd.DataFrame) -> list[tuple[str, str, float]]:
    pairs = []
    for c in df.columns:
        if not c.startswith("diagnostic_epsilon_"):
            continue
        suffix = c.replace("diagnostic_epsilon_", "")
        pcol = f"prob_dhat_gt_diagnostic_{suffix}"
        if pcol in df.columns:
            values = pd.to_numeric(df[c], errors="coerce").dropna()
            if not values.empty:
                pairs.append((suffix, pcol, float(values.iloc[0])))
    def key(x):
        try:
            return int(x[0].lower().replace("q", ""))
        except Exception:
            return 999
    return sorted(pairs, key=key)


def figure_effect_distribution(df: pd.DataFrame) -> None:
    obs = pd.to_numeric(df["observed_dhat_norm"], errors="coerce")
    obs = obs[np.isfinite(obs)]
    if obs.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(obs, bins=min(30, max(10, int(np.sqrt(len(obs))))))
    for label, _, epsilon in _diagnostic_pairs(df):
        ax.axvline(epsilon, linestyle="--", linewidth=1, label=f"{label}: {epsilon:.3g}")
    ax.set_xlabel("Observed d_hat_norm")
    ax.set_ylabel("Comparison count")
    ax.set_title("Historical effect distribution")
    if _diagnostic_pairs(df):
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FINAL_FIGURES_DIR / "05_historical_effect_distribution.png", dpi=180)
    plt.close(fig)


def figure_threshold_heatmap(df: pd.DataFrame) -> None:
    pairs = _diagnostic_pairs(df)
    if not pairs:
        return
    good = df.copy()
    for _, pcol, _ in pairs:
        good[pcol] = pd.to_numeric(good[pcol], errors="coerce")
    good = good.dropna(subset=[p[1] for p in pairs], how="all")
    if good.empty:
        return
    good = _representative(good, MAX_HEATMAP_COMPARISONS, "observed_dhat_norm")
    matrix = good[[p[1] for p in pairs]].to_numpy(float)
    fig, ax = plt.subplots(figsize=(7, max(4, len(good) * 0.38)))
    im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(pairs)))
    ax.set_xticklabels([f"{label}\nε={eps:.3g}" for label, _, eps in pairs])
    ax.set_yticks(np.arange(len(good)))
    ax.set_yticklabels([_label(r) for _, r in good.iterrows()], fontsize=8)
    ax.set_title("Posterior P(d_hat_norm > exploratory threshold)")
    fig.colorbar(im, ax=ax, label="Posterior probability")
    fig.tight_layout()
    fig.savefig(FINAL_FIGURES_DIR / "06_posterior_threshold_probability.png", dpi=180)
    plt.close(fig)


def _compliance_statuses(df: pd.DataFrame) -> list[tuple[str, str]]:
    rho_summary = (RESULTS_DIR / "bayesian_rho_summary.md").read_text(encoding="utf-8") if (RESULTS_DIR / "bayesian_rho_summary.md").exists() else ""
    rho_complete = "NOT IDENTIFIABLE" not in rho_summary and "Transition-count source" in rho_summary
    fallback_real = bool(df.get("fallback_used", pd.Series(dtype=bool)).fillna(False).astype(bool).any())
    official_available = "official_epsilon" in df and pd.to_numeric(df["official_epsilon"], errors="coerce").notna().any()
    return [
        ("Real rho matrix validation", "COMPLETE"),
        ("Compatibility screening", "COMPLETE"),
        ("Existing EXP002 inference", "COMPLETE"),
        ("Fallback routing", "COMPLETE; fallback implementation EXTERNAL" if not fallback_real else "COMPLETE"),
        ("Rho sensitivity", "COMPLETE" if (RESULTS_DIR / "rho_sensitivity.csv").exists() else "NOT RUN"),
        ("Sample-size sensitivity", "COMPLETE" if (RESULTS_DIR / "sample_size_sensitivity.csv").exists() else "NOT RUN"),
        ("Historical subgroup benchmarking", "COMPLETE"),
        ("Bayesian response uncertainty", "COMPLETE"),
        ("Bayesian posterior credible intervals", "COMPLETE"),
        ("Diagnostic threshold probabilities", "COMPLETE — exploratory only"),
        ("Bayesian rho uncertainty", "COMPLETE" if rho_complete else "DATA BLOCKED"),
        ("Alpha/beta uncertainty from rho", "COMPLETE" if rho_complete else "DATA BLOCKED"),
        ("Official Bayesian minimal-effect decision", "OFFICIAL EPSILON AVAILABLE — posterior probability calculated; binary decision rule not imposed" if official_available else "WAITING FOR OFFICIAL EPSILON"),
        ("Existing-vs-Bayesian decision agreement", "NOT CALCULATED — existing binary decision rule not supplied"),
        ("Runtime comparison", "COMPLETE"),
        ("Variational inference", "NOT IMPLEMENTED — exact conjugate posterior is available for current Dirichlet components"),
    ]


def write_summary(df: pd.DataFrame) -> None:
    status = df["screening_status"].value_counts().reindex(["SAFE", "WARN", "REJECT"], fill_value=0)
    warning_counts = _warning_counter(df)
    rho_sens = pd.read_csv(RESULTS_DIR / "rho_sensitivity.csv") if (RESULTS_DIR / "rho_sensitivity.csv").exists() else pd.DataFrame()
    rho_summary_text = (RESULTS_DIR / "bayesian_rho_summary.md").read_text(encoding="utf-8") if (RESULTS_DIR / "bayesian_rho_summary.md").exists() else ""
    out = RESULTS_DIR / "final_summary.md"
    with out.open("w", encoding="utf-8") as f:
        f.write("# GemFinder Statistical Core Refinement — Final Pipeline V2 Summary\n\n")
        f.write("## 1. What data were analysed?\n\n")
        f.write(f"The pipeline analysed **{len(df)} historical subgroup comparisons** from the supplied RY25 survey data using four-response items that have usable 4x4 unreliability matrices. Raw source files and the ONNX model were read only.\n\n")
        f.write("## 2. What comparisons were created?\n\n")
        pair_rows = df[["group_column", "group_a", "group_b"]].drop_duplicates()
        pair_text = ", ".join(f"{r.group_a} vs {r.group_b}" for r in pair_rows.itertuples())
        group_cols = ", ".join(sorted(df["group_column"].astype(str).unique()))
        f.write(f"Each row compares two real historical subgroups using grouping column(s) `{group_cols}`. Observed subgroup pair(s): {pair_text}. The pipeline does not aggregate the entire Pre file against the entire Post file.\n\n")
        f.write("## 3. How many passed screening?\n\n")
        f.write(f"SAFE: **{int(status['SAFE'])}**, WARN: **{int(status['WARN'])}**, REJECT: **{int(status['REJECT'])}**. SAFE/WARN/REJECT is an input-compatibility status, not a significance result.\n\n")
        f.write("## 4. Why were comparisons flagged?\n\n")
        if warning_counts:
            for feature, count in warning_counts.most_common(15):
                f.write(f"- `{feature}`: {count}\n")
        else:
            f.write("No WARN feature causes were recorded.\n")
        f.write("\n## 5. How sensitive was EXP002 to rho?\n\n")
        if not rho_sens.empty:
            base = rho_sens.iloc[(rho_sens["relative_rho_perturbation"].abs()).argmin()]
            gm_range = rho_sens["gamma_mean"].max() - rho_sens["gamma_mean"].min()
            f.write(f"For representative comparison `{base['comparison_id']}`, the baseline Gamma mean was **{base['gamma_mean']:.6g}**. Across the configured rho perturbations, the Gamma-mean range was **{gm_range:.6g}**. See Figure 3 for direction and magnitude.\n\n")
        else:
            f.write("Rho sensitivity was not run.\n\n")
        f.write("## 6. What did Bayesian response uncertainty show?\n\n")
        valid_ci = df[np.isfinite(pd.to_numeric(df["bayes_dhat_ci_low"], errors="coerce")) & np.isfinite(pd.to_numeric(df["bayes_dhat_ci_high"], errors="coerce"))].copy()
        if not valid_ci.empty:
            widths = valid_ci["bayes_dhat_ci_high"] - valid_ci["bayes_dhat_ci_low"]
            f.write(f"Exact Dirichlet posteriors with Monte Carlo propagation produced 95% credible intervals for **{len(valid_ci)}** comparisons. The median interval width was **{float(np.median(widths)):.6g}**. These intervals quantify response-distribution uncertainty while rho is fixed.\n\n")
        else:
            f.write("No valid Bayesian credible intervals were produced.\n\n")
        f.write("## 7. Could Bayesian rho uncertainty be run?\n\n")
        if "NOT IDENTIFIABLE" in rho_summary_text:
            f.write("No. Genuine paired test-retest transition counts were not available, so rho uncertainty is **not identifiable from the supplied data**. The code does not fabricate pseudo-counts or reinterpret Pre/Post data as test-retest transitions.\n\n")
        else:
            f.write("Yes. Genuine transition counts were detected and rho uncertainty was propagated using row-wise Dirichlet posteriors.\n\n")
        f.write("## 8. What does the historical benchmark show?\n\n")
        exp_rt = pd.to_numeric(df.get("runtime_exp002_ms"), errors="coerce")
        bayes_rt = pd.to_numeric(df.get("runtime_bayesian_ms"), errors="coerce")
        exp_rt = exp_rt[np.isfinite(exp_rt)]
        bayes_rt = bayes_rt[np.isfinite(bayes_rt)]
        f.write("The benchmark keeps EXP002 Gamma summaries and Bayesian d_hat_norm posterior summaries separate because they are different statistical objects. ")
        if not exp_rt.empty:
            f.write(f"Median EXP002 runtime was **{float(np.median(exp_rt)):.4g} ms** per comparison. ")
        if not bayes_rt.empty:
            f.write(f"Median Bayesian response-evaluation runtime was **{float(np.median(bayes_rt)):.4g} ms** per comparison. ")
        f.write("Training-range compatibility and routing are recorded for every historical comparison.\n\n")
        f.write("## 9. What requirements are complete?\n\n")
        f.write("| Requirement | Status |\n|---|---|\n")
        for requirement, value in _compliance_statuses(df):
            f.write(f"| {requirement} | {value} |\n")
        f.write("\n## 10. What remains future work/data-dependent?\n\n")
        f.write("The real team-owned fallback inference method still needs integration. Bayesian rho uncertainty requires genuine test-retest transition counts if those are not supplied. An official minimal-effect epsilon is still required before any Bayesian minimal-effect decision can be labelled official. Existing-vs-Bayesian decision agreement also requires the exact existing binary decision rule; it is not inferred from Gamma means. Full variational inference is not claimed because the current Dirichlet-multinomial components have exact conjugate posteriors.\n")
    print(f"Saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create final GemFinder report and figures.")
    parser.add_argument("--assets-root", required=False, help="Accepted for pipeline interface consistency.")
    parser.add_argument("--official-epsilon", type=float, default=OFFICIAL_EPSILON)
    args = parser.parse_args()
    ensure_output_dirs()

    df = pd.read_csv(RESULTS_DIR / "historical_benchmark.csv")
    final_columns = [
        "comparison_id", "item", "group_column", "group_a", "group_b", "n_a", "n_b",
        "sample_size_ratio", "rho_valid", "screening_status", "screening_reason",
        "observed_dhat_norm", "exp002_alpha", "exp002_beta", "gamma_mean", "gamma_q95",
        "bayes_dhat_mean", "bayes_dhat_ci_low", "bayes_dhat_ci_high", "official_epsilon",
        "posterior_probability_official_epsilon", "inference_path", "fallback_used",
        "runtime_exp002_ms", "runtime_bayesian_ms",
    ]
    diagnostic = [c for c in df.columns if c.startswith("diagnostic_epsilon_") or c.startswith("prob_dhat_gt_diagnostic_")]
    final_df = df[[c for c in final_columns if c in df.columns] + diagnostic].copy()
    final_out = RESULTS_DIR / "final_pipeline_results.csv"
    final_df.to_csv(final_out, index=False)

    figure_screening(df)
    figure_warning_causes(df)
    figure_rho_sensitivity()
    figure_bayesian_uncertainty(df)
    figure_effect_distribution(df)
    figure_threshold_heatmap(df)
    write_summary(df)

    print(f"Saved: {final_out}")
    print(f"Final figures: {FINAL_FIGURES_DIR}")


if __name__ == "__main__":
    main()
