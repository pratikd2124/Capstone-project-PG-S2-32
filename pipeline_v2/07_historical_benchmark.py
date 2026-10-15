"""Stage 7: assemble the primary real-data benchmark table."""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from common import RESULTS_DIR, ensure_output_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble historical subgroup benchmark results.")
    parser.add_argument("--assets-root", required=False, help="Accepted for pipeline interface consistency.")
    args = parser.parse_args()

    ensure_output_dirs()
    comparisons = pd.read_csv(RESULTS_DIR / "comparisons.csv")
    screening = pd.read_csv(RESULTS_DIR / "screening_results.csv")
    exp002 = pd.read_csv(RESULTS_DIR / "exp002_results.csv")
    fallback = pd.read_csv(RESULTS_DIR / "fallback_results.csv")
    bayes = pd.read_csv(RESULTS_DIR / "bayesian_response_results.csv")

    screen_cols = [
        "comparison_id", "rho_valid", "screening_status", "screening_reason",
        "warning_features", "outside_training_range_count", "outside_training_range_features",
    ]
    exp_cols = [
        "comparison_id", "exp002_alpha", "exp002_beta", "gamma_mean", "gamma_q95",
        "runtime_exp002_ms", "network_training_range_status", "network_outside_training_range",
        "network_error",
    ]
    fallback_cols = [
        "comparison_id", "inference_path", "fallback_required", "fallback_used",
        "fallback_status", "fallback_runtime_ms",
    ]

    df = comparisons.merge(screening[screen_cols], on="comparison_id", how="left")
    df = df.merge(exp002[exp_cols], on="comparison_id", how="left")
    df = df.merge(fallback[fallback_cols], on="comparison_id", how="left")
    df = df.merge(bayes, on=["comparison_id", "item"], how="left")

    preferred = [
        "comparison_id", "item", "group_column", "group_a", "group_b", "n_a", "n_b",
        "sample_size_ratio", "observed_dhat_norm", "rho_valid", "screening_status",
        "screening_reason", "warning_features", "exp002_alpha", "exp002_beta",
        "gamma_mean", "gamma_q95", "bayes_dhat_mean", "bayes_dhat_ci_low",
        "bayes_dhat_ci_high", "official_epsilon", "posterior_probability_official_epsilon",
        "inference_path", "fallback_used", "fallback_status", "runtime_exp002_ms",
        "runtime_bayesian_ms", "fallback_runtime_ms", "network_training_range_status",
        "network_outside_training_range", "network_error", "bayesian_response_status",
        "bayesian_response_error",
    ]
    diagnostic = [c for c in df.columns if c.startswith("diagnostic_epsilon_") or c.startswith("prob_dhat_gt_diagnostic_")]
    ordered = preferred + diagnostic
    remaining = [c for c in df.columns if c not in ordered]
    df = df[[c for c in ordered if c in df.columns] + remaining]

    out = RESULTS_DIR / "historical_benchmark.csv"
    df.to_csv(out, index=False)

    status = df["screening_status"].value_counts().reindex(["SAFE", "WARN", "REJECT"], fill_value=0)
    print(f"Historical benchmark rows: {len(df)}")
    print(f"SAFE/WARN/REJECT: {int(status['SAFE'])}/{int(status['WARN'])}/{int(status['REJECT'])}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
