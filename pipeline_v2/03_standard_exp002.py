"""Stage 3: run the original EXP002 ONNX network without retraining it."""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from common import (
    RESULTS_DIR,
    ensure_output_dirs,
    load_context,
    parse_counts_from_row,
    predict_exp002,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run standard EXP002 inference for SAFE/WARN comparisons.")
    parser.add_argument("--assets-root", required=True)
    args = parser.parse_args()

    ensure_output_dirs()
    ctx = load_context(args.assets_root, require_model=True)
    comparisons = pd.read_csv(RESULTS_DIR / "comparisons.csv")
    screening = pd.read_csv(RESULTS_DIR / "screening_results.csv")
    df = comparisons.merge(
        screening[["comparison_id", "screening_status", "screening_reason"]],
        on="comparison_id", how="left",
    )

    rows: list[dict] = []
    for _, row in df.iterrows():
        base = {
            "comparison_id": row["comparison_id"],
            "item": row["item"],
            "screening_status": row["screening_status"],
            "exp002_alpha": np.nan,
            "exp002_beta": np.nan,
            "gamma_mean": np.nan,
            "gamma_q95": np.nan,
            "runtime_exp002_ms": np.nan,
            "network_training_range_status": "NOT_RUN",
            "network_outside_training_range": "",
            "network_error": "",
        }
        if row["screening_status"] == "REJECT":
            base["network_training_range_status"] = "REJECT_NOT_RUN"
            rows.append(base)
            continue
        try:
            x = parse_counts_from_row(row, "a")
            y = parse_counts_from_row(row, "b")
            pred = predict_exp002(ctx, x, y, str(row["item"]))
            base.update({
                "exp002_alpha": pred.alpha,
                "exp002_beta": pred.beta,
                "gamma_mean": pred.gamma_mean,
                "gamma_q95": pred.gamma_q95,
                "runtime_exp002_ms": pred.runtime_ms,
                "network_training_range_status": "WARN" if pred.outside_training_range else "SAFE",
                "network_outside_training_range": "; ".join(pred.outside_training_range),
            })
        except Exception as exc:
            base["network_training_range_status"] = "ERROR"
            base["network_error"] = str(exc)
        rows.append(base)

    result = pd.DataFrame(rows)
    out = RESULTS_DIR / "exp002_results.csv"
    result.to_csv(out, index=False)
    print(f"Saved: {out}")
    print(result["network_training_range_status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
