"""Propagate matrix and sample-size sensitivity directly through EXP002.

No scrp_toolkit imports are used. The script reads EXP002.json and the shipped
ONNX model directly and reproduces RYA's feature calculations locally.
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from standalone_common import (
    DEFAULT_ITEM, DEFAULT_PRE, DEFAULT_POST, FIGURES_DIR, RESULTS_DIR,
    counts_from_proportions, ensure_output_dirs, load_context,
    perturb_error_mass, predict_exp002, proportions_from_counts,
)


def sample_size_experiment(ctx, item, pre, post, sizes):
    p_pre, p_post = proportions_from_counts(pre), proportions_from_counts(post)
    rows = []
    for n in sizes:
        x = counts_from_proportions(p_pre, n)
        y = counts_from_proportions(p_post, n)
        try:
            pred = predict_exp002(ctx, x, y, item)
            rows.append({
                "item": item, "n_pre": int(x.sum()), "n_post": int(y.sum()),
                "shape": pred.shape, "rate": pred.rate,
                "gamma_mean": pred.gamma_mean, "gamma_q95": pred.gamma_q95,
                "outside_training_range_count": len(pred.outside_training_range),
                "outside_training_range": "; ".join(pred.outside_training_range),
                "status": "WARN" if pred.outside_training_range else "OK", "error": "",
            })
        except Exception as exc:
            rows.append({"item": item, "n_pre": int(x.sum()), "n_post": int(y.sum()),
                         "shape": np.nan, "rate": np.nan, "gamma_mean": np.nan,
                         "gamma_q95": np.nan, "outside_training_range_count": np.nan,
                         "outside_training_range": "", "status": "ERROR", "error": str(exc)})
    return pd.DataFrame(rows)


def matrix_experiment(ctx, item, pre, post, changes):
    baseline_p = np.asarray(ctx.p_matrices[item], dtype=float)
    base = predict_exp002(ctx, pre, post, item, baseline_p)
    rows = []
    for change in changes:
        p = perturb_error_mass(baseline_p, change)
        try:
            pred = predict_exp002(ctx, pre, post, item, p)
            rows.append({
                "item": item, "relative_error_mass_change": change,
                "percent_change": change * 100,
                "shape": pred.shape, "rate": pred.rate,
                "gamma_mean": pred.gamma_mean, "gamma_q95": pred.gamma_q95,
                "baseline_gamma_mean": base.gamma_mean,
                "baseline_gamma_q95": base.gamma_q95,
                "delta_gamma_mean": pred.gamma_mean - base.gamma_mean,
                "percent_delta_gamma_mean": (pred.gamma_mean / base.gamma_mean - 1) * 100,
                "delta_gamma_q95": pred.gamma_q95 - base.gamma_q95,
                "outside_training_range_count": len(pred.outside_training_range),
                "outside_training_range": "; ".join(pred.outside_training_range),
                "status": "WARN" if pred.outside_training_range else "OK", "error": "",
            })
        except Exception as exc:
            rows.append({"item": item, "relative_error_mass_change": change,
                         "percent_change": change * 100, "shape": np.nan, "rate": np.nan,
                         "gamma_mean": np.nan, "gamma_q95": np.nan,
                         "baseline_gamma_mean": base.gamma_mean, "baseline_gamma_q95": base.gamma_q95,
                         "delta_gamma_mean": np.nan, "percent_delta_gamma_mean": np.nan,
                         "delta_gamma_q95": np.nan, "outside_training_range_count": np.nan,
                         "outside_training_range": "", "status": "ERROR", "error": str(exc)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-root", default=None)
    parser.add_argument("--item", default=DEFAULT_ITEM)
    parser.add_argument("--pre", nargs=4, type=float, default=DEFAULT_PRE.tolist())
    parser.add_argument("--post", nargs=4, type=float, default=DEFAULT_POST.tolist())
    parser.add_argument("--sizes", nargs="+", type=int,
                        default=[20, 50, 100, 250, 500, 1000, 5000, 10_000, 20_000])
    parser.add_argument("--changes", nargs="+", type=float,
                        default=[-0.10, -0.05, -0.01, 0.0, 0.01, 0.05, 0.10])
    args = parser.parse_args()

    ensure_output_dirs()
    ctx = load_context(args.assets_root, require_model=True)
    pre, post = np.asarray(args.pre), np.asarray(args.post)
    if args.item not in ctx.p_matrices or np.asarray(ctx.p_matrices[args.item]).shape != (4, 4):
        raise ValueError(f"{args.item} is not a supported 4-option EXP002 item")

    sample = sample_size_experiment(ctx, args.item, pre, post, args.sizes)
    matrix = matrix_experiment(ctx, args.item, pre, post, args.changes)
    sample.to_csv(RESULTS_DIR / "sample_size_sensitivity.csv", index=False)
    matrix.to_csv(RESULTS_DIR / "matrix_sensitivity_network.csv", index=False)

    good = sample[np.isfinite(sample["gamma_mean"])].copy()
    if not good.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(good["n_pre"], good["gamma_mean"], marker="o", label="Gamma mean")
        ax.plot(good["n_pre"], good["gamma_q95"], marker="o", label="Gamma 95th percentile")
        ax.set_xscale("log")
        ax.set_xlabel("Pre/Post sample size (log scale)")
        ax.set_ylabel("Predicted d_hat_norm distribution")
        ax.set_title(f"{args.item}: EXP002 sample-size sensitivity")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "sample_size_sensitivity.png", dpi=160)
        plt.close(fig)

    good = matrix[np.isfinite(matrix["gamma_mean"])].copy()
    if not good.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(good["percent_change"], good["gamma_mean"], marker="o", label="Gamma mean")
        ax.plot(good["percent_change"], good["gamma_q95"], marker="o", label="Gamma 95th percentile")
        ax.set_xlabel("Relative change in off-diagonal error mass (%)")
        ax.set_ylabel("Predicted d_hat_norm distribution")
        ax.set_title(f"{args.item}: matrix sensitivity propagated through EXP002")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "matrix_network_sensitivity.png", dpi=160)
        plt.close(fig)

    print("Sample-size sensitivity:\n", sample.to_string(index=False))
    print("\nMatrix sensitivity:\n", matrix.to_string(index=False))


if __name__ == "__main__":
    main()
