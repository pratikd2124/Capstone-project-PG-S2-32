"""Classify unusual standalone EXP002 inputs as SAFE, WARN, or REJECT."""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from standalone_common import (
    DEFAULT_ITEM, DEFAULT_PRE, DEFAULT_POST, RESULTS_DIR,
    ensure_output_dirs, load_context, predict_exp002,
)


def classify(ctx, name, item, x, y, matrix=None):
    base = {"scenario": name, "item": item, "n_pre": int(np.sum(x)), "n_post": int(np.sum(y))}
    try:
        pred = predict_exp002(ctx, x, y, item, matrix)
        status = "WARN" if pred.outside_training_range else "SAFE"
        reason = (
            "Finite prediction, but feature(s) lie outside EXP002 stored training min/max."
            if pred.outside_training_range else
            "Finite prediction and all features are inside stored training min/max."
        )
        return {**base, "classification": status, "shape": pred.shape, "rate": pred.rate,
                "gamma_mean": pred.gamma_mean, "gamma_q95": pred.gamma_q95,
                "outside_training_range_count": len(pred.outside_training_range),
                "outside_training_range": "; ".join(pred.outside_training_range),
                "reason": reason, "error": ""}
    except Exception as exc:
        return {**base, "classification": "REJECT", "shape": np.nan, "rate": np.nan,
                "gamma_mean": np.nan, "gamma_q95": np.nan,
                "outside_training_range_count": np.nan, "outside_training_range": "",
                "reason": "Input cannot be safely processed by the current 4-option EXP002 pipeline.",
                "error": str(exc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-root", default=None)
    parser.add_argument("--item", default=DEFAULT_ITEM)
    args = parser.parse_args()

    ensure_output_dirs()
    ctx = load_context(args.assets_root, require_model=True)
    cases = [
        ("normal", args.item, DEFAULT_PRE, DEFAULT_POST, None),
        ("zero_samples", args.item, np.zeros(4), np.zeros(4), None),
        ("one_respondent", args.item, np.array([1, 0, 0, 0]), np.array([1, 0, 0, 0]), None),
        ("single_category", args.item, np.array([100, 0, 0, 0]), np.array([100, 0, 0, 0]), None),
        ("missing_category", args.item, np.array([40, 30, 30, 0]), np.array([35, 35, 30, 0]), None),
        ("pre_post_imbalance", args.item, np.array([400, 300, 200, 100]), np.array([4, 3, 2, 1]), None),
        ("very_large_n", args.item, np.array([5000, 5000, 5000, 5000]), np.array([5000, 5000, 5000, 5000]), None),
    ]

    six_item = next((q for q, p in ctx.p_matrices.items() if np.asarray(p).shape == (6, 6)), None)
    if six_item:
        cases.append(("six_option_item", six_item,
                      np.array([10, 10, 10, 10, 10, 10]),
                      np.array([10, 10, 10, 10, 10, 10]),
                      ctx.p_matrices[six_item]))

    df = pd.DataFrame([classify(ctx, *case) for case in cases])
    path = RESULTS_DIR / "input_robustness.csv"
    df.to_csv(path, index=False)
    print(df.to_string(index=False))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
