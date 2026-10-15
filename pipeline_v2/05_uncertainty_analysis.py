"""Stage 5: EXP002 sensitivity to rho assumptions and sample size."""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from config import PREFERRED_SENSITIVITY_ITEM, RHO_PERTURBATIONS, SAMPLE_SIZE_GRID
from common import (
    RESULTS_DIR,
    counts_from_proportions,
    ensure_output_dirs,
    load_context,
    parse_counts_from_row,
    perturb_rho_offdiagonal_mass,
    predict_exp002,
)


def _select_representative(comparisons: pd.DataFrame, screening: pd.DataFrame) -> pd.Series:
    merged = comparisons.merge(screening[["comparison_id", "screening_status"]], on="comparison_id", how="left")
    usable = merged[merged["screening_status"].isin(["SAFE", "WARN"])].copy()
    if usable.empty:
        raise RuntimeError("No SAFE/WARN comparison is available for sensitivity analysis")
    preferred = usable[usable["item"].astype(str) == PREFERRED_SENSITIVITY_ITEM]
    return preferred.iloc[0] if not preferred.empty else usable.iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rho and sample-size sensitivity analyses.")
    parser.add_argument("--assets-root", required=True)
    args = parser.parse_args()

    ensure_output_dirs()
    ctx = load_context(args.assets_root, require_model=True)
    comparisons = pd.read_csv(RESULTS_DIR / "comparisons.csv")
    screening = pd.read_csv(RESULTS_DIR / "screening_results.csv")
    row = _select_representative(comparisons, screening)

    item = str(row["item"])
    x = parse_counts_from_row(row, "a")
    y = parse_counts_from_row(row, "b")
    baseline_rho = ctx.rho_normalised[item]

    rho_rows: list[dict] = []
    for perturbation in RHO_PERTURBATIONS:
        rho = perturb_rho_offdiagonal_mass(baseline_rho, perturbation)
        pred = predict_exp002(ctx, x, y, item, rho)
        rho_rows.append({
            "comparison_id": row["comparison_id"],
            "item": item,
            "relative_rho_perturbation": perturbation,
            "exp002_alpha": pred.alpha,
            "exp002_beta": pred.beta,
            "gamma_mean": pred.gamma_mean,
            "gamma_q95": pred.gamma_q95,
            "training_range_status": "WARN" if pred.outside_training_range else "SAFE",
            "outside_training_range": "; ".join(pred.outside_training_range),
            "runtime_exp002_ms": pred.runtime_ms,
            "zero_perturbation_equals_baseline": bool(
                perturbation != 0 or np.array_equal(rho, baseline_rho)
            ),
        })
    rho_df = pd.DataFrame(rho_rows)
    rho_out = RESULTS_DIR / "rho_sensitivity.csv"
    rho_df.to_csv(rho_out, index=False)

    p_x, p_y = x / x.sum(), y / y.sum()
    original_ratio = float(y.sum() / x.sum())
    size_rows: list[dict] = []
    for n_a in SAMPLE_SIZE_GRID:
        n_b = max(1, int(round(n_a * original_ratio)))
        x_n = counts_from_proportions(p_x, n_a)
        y_n = counts_from_proportions(p_y, n_b)
        try:
            pred = predict_exp002(ctx, x_n, y_n, item, baseline_rho)
            size_rows.append({
                "comparison_id": row["comparison_id"], "item": item,
                "n_a": int(x_n.sum()), "n_b": int(y_n.sum()),
                "sample_size_ratio": max(x_n.sum(), y_n.sum()) / min(x_n.sum(), y_n.sum()),
                "exp002_alpha": pred.alpha, "exp002_beta": pred.beta,
                "gamma_mean": pred.gamma_mean, "gamma_q95": pred.gamma_q95,
                "training_range_status": "WARN" if pred.outside_training_range else "SAFE",
                "outside_training_range": "; ".join(pred.outside_training_range),
                "runtime_exp002_ms": pred.runtime_ms, "error": "",
            })
        except Exception as exc:
            size_rows.append({
                "comparison_id": row["comparison_id"], "item": item,
                "n_a": int(x_n.sum()), "n_b": int(y_n.sum()),
                "sample_size_ratio": max(x_n.sum(), y_n.sum()) / min(x_n.sum(), y_n.sum()),
                "exp002_alpha": np.nan, "exp002_beta": np.nan,
                "gamma_mean": np.nan, "gamma_q95": np.nan,
                "training_range_status": "ERROR", "outside_training_range": "",
                "runtime_exp002_ms": np.nan, "error": str(exc),
            })
    size_df = pd.DataFrame(size_rows)
    size_out = RESULTS_DIR / "sample_size_sensitivity.csv"
    size_df.to_csv(size_out, index=False)

    print(f"Representative comparison: {row['comparison_id']}")
    print(f"Saved: {rho_out}")
    print(f"Saved: {size_out}")


if __name__ == "__main__":
    main()
