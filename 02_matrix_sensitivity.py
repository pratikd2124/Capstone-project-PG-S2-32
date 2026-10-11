"""Sensitivity of an item's unreliability matrix to controlled perturbations."""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from standalone_common import (
    DEFAULT_ITEM, FIGURES_DIR, RESULTS_DIR, asymm_calc, ensure_output_dirs,
    load_context, mean_off_diagonal_mass, perturb_error_mass,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-root", default=None)
    parser.add_argument("--item", default=DEFAULT_ITEM)
    parser.add_argument("--changes", nargs="+", type=float,
                        default=[-0.10, -0.05, -0.01, 0.0, 0.01, 0.05, 0.10])
    args = parser.parse_args()

    ensure_output_dirs()
    ctx = load_context(args.assets_root, require_model=False)
    if args.item not in ctx.p_matrices:
        raise KeyError(f"Unknown survey item: {args.item}")
    baseline = np.asarray(ctx.p_matrices[args.item], dtype=float)
    if baseline.shape != (4, 4):
        raise ValueError(f"This analysis is restricted to 4x4 items; {args.item} has {baseline.shape}")

    base_asymm = asymm_calc(baseline)
    base_offdiag = mean_off_diagonal_mass(baseline)
    rows = []
    for change in args.changes:
        p = perturb_error_mass(baseline, change)
        row = {
            "item": args.item,
            "relative_error_mass_change": change,
            "percent_change": change * 100,
            "baseline_asymm": base_asymm,
            "perturbed_asymm": asymm_calc(p),
            "delta_asymm": asymm_calc(p) - base_asymm,
            "baseline_mean_offdiag_mass": base_offdiag,
            "perturbed_mean_offdiag_mass": mean_off_diagonal_mass(p),
            "frobenius_distance": float(np.linalg.norm(p - baseline)),
            "max_row_sum_error": float(np.max(np.abs(p.sum(axis=1) - 1.0))),
        }
        for i in range(4):
            for j in range(4):
                row[f"p_{i+1}{j+1}"] = p[i, j]
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("relative_error_mass_change")
    path = RESULTS_DIR / "matrix_sensitivity.csv"
    df.to_csv(path, index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["percent_change"], df["perturbed_asymm"], marker="o")
    ax.axhline(base_asymm, linestyle="--", linewidth=1)
    ax.set_xlabel("Relative change in off-diagonal error mass (%)")
    ax.set_ylabel("Asymmetry score")
    ax.set_title(f"{args.item}: unreliability-matrix sensitivity")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "matrix_asymmetry_sensitivity.png", dpi=160)
    plt.close(fig)

    print(df[["percent_change", "perturbed_asymm", "delta_asymm", "frobenius_distance"]].to_string(index=False))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
