"""Stage 2: validate rho matrices and screen EXP002 input compatibility."""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from config import MIN_GROUP_SIZE
from common import (
    RESULTS_DIR,
    asymmetry_mse,
    ensure_output_dirs,
    extract_exp002_features,
    feature_range_violations,
    load_context,
    parse_counts_from_row,
)


def _rho_diagnostics(raw: np.ndarray, normalised: np.ndarray | None) -> dict:
    out = {
        "rho_valid": False,
        "rho_shape": str(tuple(raw.shape)),
        "rho_finite": bool(np.all(np.isfinite(raw))) if raw.size else False,
        "rho_nonnegative": bool(np.all(raw >= 0)) if raw.size else False,
        "rho_row_sums": "",
        "rho_was_normalised": False,
        "rho_diagonal_is_row_max": False,
        "rho_asymmetry": np.nan,
    }
    if raw.ndim == 2:
        row_sums = raw.sum(axis=1)
        out["rho_row_sums"] = ",".join(f"{v:.6g}" for v in row_sums)
    if normalised is None:
        return out
    out["rho_valid"] = True
    out["rho_was_normalised"] = not np.array_equal(raw, normalised)
    out["rho_diagonal_is_row_max"] = bool(
        all(normalised[i, i] >= np.max(np.delete(normalised[i], i)) for i in range(4))
    )
    out["rho_asymmetry"] = asymmetry_mse(normalised)
    return out


def screen_row(ctx, row: pd.Series, min_group_size: int) -> dict:
    item = str(row["item"])
    x = parse_counts_from_row(row, "a")
    y = parse_counts_from_row(row, "b")
    reasons: list[str] = []
    warning_features: list[str] = []
    status = "SAFE"

    raw_rho = ctx.rho_raw.get(item)
    norm_rho = ctx.rho_normalised.get(item)
    if raw_rho is None:
        raw_rho = np.asarray([], dtype=float)
    diag = _rho_diagnostics(raw_rho, norm_rho)

    n_a, n_b = int(x.sum()), int(y.sum())
    if n_a < min_group_size or n_b < min_group_size:
        status = "REJECT"
        reasons.append(f"group size below MIN_GROUP_SIZE={min_group_size}")
    if norm_rho is None:
        status = "REJECT"
        reasons.append(ctx.rho_notes.get(item, "rho missing or invalid"))

    features: dict[str, float] = {}
    violations: list[str] = []
    if status != "REJECT":
        try:
            features = extract_exp002_features(x, y, norm_rho)
            if any(not np.isfinite(features[k]) for k in ctx.metadata["feature_labels"]):
                status = "REJECT"
                bad = [k for k in ctx.metadata["feature_labels"] if not np.isfinite(features[k])]
                reasons.append("non-finite EXP002 feature(s): " + ", ".join(bad))
            else:
                violations = feature_range_violations(features, ctx.metadata, ctx.min_max)
                if violations:
                    status = "WARN"
                    warning_features.extend(v.split("=", 1)[0] for v in violations)
                    reasons.append("EXP002 training-range: " + " | ".join(violations))
        except Exception as exc:
            status = "REJECT"
            reasons.append(f"feature construction failed: {exc}")

    ratio = max(n_a, n_b) / min(n_a, n_b) if min(n_a, n_b) > 0 else np.inf
    if status != "REJECT" and ctx.training_ratio_range is not None:
        lo, hi = ctx.training_ratio_range
        if ratio < lo - 1e-12 or ratio > hi + 1e-12:
            status = "WARN"
            warning_features.append("sample_size_ratio")
            reasons.append(f"sample_size_ratio={ratio:.6g} outside training [{lo:.6g}, {hi:.6g}]")

    if status != "REJECT" and not diag["rho_diagonal_is_row_max"]:
        status = "WARN"
        warning_features.append("rho_diagonal")
        reasons.append("rho diagonal is not the largest probability in at least one row")

    # Sparse-category information is recorded transparently. It is not made a
    # significance rule and is not automatically a warning if training ranges
    # already show zero category proportions were represented.
    sparse_a = int(np.sum(x == 0))
    sparse_b = int(np.sum(y == 0))

    return {
        "comparison_id": row["comparison_id"],
        "item": item,
        "rho_valid": bool(diag["rho_valid"]),
        "rho_shape": diag["rho_shape"],
        "rho_finite": diag["rho_finite"],
        "rho_nonnegative": diag["rho_nonnegative"],
        "rho_row_sums": diag["rho_row_sums"],
        "rho_was_normalised": diag["rho_was_normalised"],
        "rho_preprocessing_note": ctx.rho_notes.get(item, ""),
        "rho_diagonal_is_row_max": diag["rho_diagonal_is_row_max"],
        "rho_asymmetry": diag["rho_asymmetry"],
        "sparse_categories_a": sparse_a,
        "sparse_categories_b": sparse_b,
        "training_ratio_min": ctx.training_ratio_range[0] if ctx.training_ratio_range else np.nan,
        "training_ratio_max": ctx.training_ratio_range[1] if ctx.training_ratio_range else np.nan,
        "outside_training_range_count": len(violations),
        "outside_training_range_features": "; ".join(violations),
        "warning_features": "; ".join(dict.fromkeys(warning_features)),
        "screening_status": status,
        "screening_reason": "; ".join(reasons) if reasons else "all validation and stored training-range checks passed",
        "safe_warn_is_significance_decision": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate rho and screen EXP002 inputs.")
    parser.add_argument("--assets-root", required=True)
    parser.add_argument("--min-group-size", type=int, default=MIN_GROUP_SIZE)
    args = parser.parse_args()

    ensure_output_dirs()
    ctx = load_context(args.assets_root, require_model=False)
    comparisons = pd.read_csv(RESULTS_DIR / "comparisons.csv")
    rows = [screen_row(ctx, row, args.min_group_size) for _, row in comparisons.iterrows()]
    out_df = pd.DataFrame(rows)
    out = RESULTS_DIR / "screening_results.csv"
    out_df.to_csv(out, index=False)

    counts = out_df["screening_status"].value_counts().reindex(["SAFE", "WARN", "REJECT"], fill_value=0)
    print(f"SAFE={counts['SAFE']} WARN={counts['WARN']} REJECT={counts['REJECT']}")
    if ctx.training_ratio_range is None:
        print("Training sample-size ratio range: unavailable (instance_data CSVs not found); ratio was recorded but not guessed.")
    else:
        print(f"Training sample-size ratio range: {ctx.training_ratio_range}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
