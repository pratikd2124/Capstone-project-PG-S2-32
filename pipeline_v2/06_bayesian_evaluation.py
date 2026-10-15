"""Stage 6: exploratory Bayesian evaluation, separate from production inference.

Level A is available now: exact Dirichlet posteriors for the two response
probability vectors with Monte Carlo propagation through the original d_hat_norm
formula while rho is fixed.

Level B is run only if genuine test-retest transition counts are found. Rho
probabilities are never converted to pseudo-counts and Pre/Post exports are never
mislabelled as test-retest transitions.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    DIAGNOSTIC_QUANTILES,
    DIRICHLET_PRIOR_ALPHA,
    OFFICIAL_EPSILON,
    POSTERIOR_DRAWS,
    RANDOM_SEED,
    RHO_NETWORK_DRAWS,
)
from common import (
    ASSET_CACHE_DIR,
    RESULTS_DIR,
    bayesian_d_hat_draws,
    bayesian_d_hat_draws_with_rho,
    d_hat_norm,
    ensure_output_dirs,
    load_context,
    parse_counts_from_row,
    predict_exp002,
    summarise_draws,
)


def _diagnostic_thresholds(observed: np.ndarray) -> list[tuple[str, float]]:
    finite = np.asarray(observed, dtype=float)
    finite = finite[np.isfinite(finite) & (finite >= 0)]
    if finite.size == 0:
        raise RuntimeError("No finite observed d_hat_norm values are available for diagnostic thresholds")
    return [(f"q{int(q * 100):02d}", float(np.quantile(finite, q))) for q in DIAGNOSTIC_QUANTILES]


def _candidate_transition_files(root: Path) -> list[Path]:
    roots = [root]
    if ASSET_CACHE_DIR.exists():
        roots.append(ASSET_CACHE_DIR)
    hits: set[Path] = set()
    patterns = ["*transition*count*.csv", "*transition*count*.json", "*retest*count*.csv", "*retest*count*.json"]
    for r in roots:
        for pattern in patterns:
            hits.update(p.resolve() for p in r.rglob(pattern) if p.is_file())
    # Explicitly exclude supplied probability matrices and raw Pre/Post exports.
    return sorted(
        p for p in hits
        if p.name not in {"unreliability-matrices.json", "RY25_PreMay10.csv", "RY25_PostMay10.csv"}
    )


def _validate_count_matrix(arr: np.ndarray, source: str) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    if a.shape != (4, 4):
        raise ValueError(f"{source}: transition count matrix must be 4x4; got {a.shape}")
    if not np.all(np.isfinite(a)) or np.any(a < 0):
        raise ValueError(f"{source}: transition counts must be finite and non-negative")
    if not np.allclose(a, np.round(a), atol=1e-9):
        raise ValueError(f"{source}: non-integer values look like probabilities rather than genuine counts")
    if a.sum() <= 0:
        raise ValueError(f"{source}: transition count matrix is empty")
    return np.round(a).astype(int)


def _load_transition_counts(path: Path) -> dict[str, np.ndarray]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        out = {}
        for item, matrix in payload.items():
            out[str(item)] = _validate_count_matrix(np.asarray(matrix), f"{path.name}:{item}")
        return out

    df = pd.read_csv(path)
    lower = {str(c).lower(): c for c in df.columns}
    long_required = {"item", "initial_response", "retest_response", "count"}
    if long_required.issubset(lower):
        item_col, i_col, r_col, c_col = [lower[x] for x in ["item", "initial_response", "retest_response", "count"]]
        out: dict[str, np.ndarray] = {}
        for item, group in df.groupby(item_col):
            mat = np.zeros((4, 4), dtype=float)
            for _, row in group.iterrows():
                i, j = int(row[i_col]), int(row[r_col])
                if i not in [1, 2, 3, 4] or j not in [1, 2, 3, 4]:
                    raise ValueError(f"{path.name}:{item}: responses must be 1..4")
                mat[i - 1, j - 1] += float(row[c_col])
            out[str(item)] = _validate_count_matrix(mat, f"{path.name}:{item}")
        return out

    wide_cols = [f"c{i}{j}" for i in range(1, 5) for j in range(1, 5)]
    if "item" in lower and all(c in lower for c in wide_cols):
        out = {}
        for _, row in df.iterrows():
            item = str(row[lower["item"]])
            mat = np.asarray([[row[lower[f"c{i}{j}"]] for j in range(1, 5)] for i in range(1, 5)], dtype=float)
            out[item] = _validate_count_matrix(mat, f"{path.name}:{item}")
        return out
    raise ValueError(
        f"{path.name}: unsupported format. Expected long item/initial_response/retest_response/count "
        "or wide item,c11,...,c44."
    )


def _find_genuine_transition_counts(root: Path, explicit: str | None) -> tuple[Path | None, dict[str, np.ndarray] | None, list[str]]:
    candidates = [Path(explicit).expanduser().resolve()] if explicit else _candidate_transition_files(root)
    rejected: list[str] = []
    for path in candidates:
        if not path.exists():
            rejected.append(f"{path}: file does not exist")
            continue
        try:
            counts = _load_transition_counts(path)
            if counts:
                return path, counts, rejected
        except Exception as exc:
            rejected.append(f"{path}: {exc}")
    return None, None, rejected


def _write_rho_requirement_template(items: list[str]) -> Path:
    rows = []
    for item in sorted(set(items)):
        for initial in range(1, 5):
            for retest in range(1, 5):
                rows.append({
                    "item": item,
                    "initial_response": initial,
                    "retest_response": retest,
                    "count": "",
                })
    path = RESULTS_DIR / "rho_data_requirement_template.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _sample_rho(counts: np.ndarray, draws: int, prior_alpha: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.empty((draws, 4, 4), dtype=float)
    for row in range(4):
        result[:, row, :] = rng.dirichlet(counts[row].astype(float) + prior_alpha, size=draws)
    return result


def run_level_a(ctx, comparisons: pd.DataFrame, draws: int, prior_alpha: float, seed: int, official_epsilon: float | None):
    observed_values: list[float] = []
    observed_by_id: dict[str, float] = {}
    for _, row in comparisons.iterrows():
        item = str(row["item"])
        try:
            value = d_hat_norm(
                parse_counts_from_row(row, "a"),
                parse_counts_from_row(row, "b"),
                ctx.rho_normalised[item],
                ctx.b_matrix,
            )
        except Exception:
            value = np.nan
        observed_values.append(value)
        observed_by_id[str(row["comparison_id"])] = value

    thresholds = _diagnostic_thresholds(np.asarray(observed_values, dtype=float))
    rows: list[dict] = []
    for idx, row in comparisons.iterrows():
        item = str(row["item"])
        base = {
            "comparison_id": row["comparison_id"],
            "item": item,
            "observed_dhat_norm": observed_by_id[str(row["comparison_id"])],
            "bayes_dhat_mean": np.nan,
            "bayes_dhat_median": np.nan,
            "bayes_dhat_sd": np.nan,
            "bayes_dhat_ci_low": np.nan,
            "bayes_dhat_ci_high": np.nan,
            "runtime_bayesian_ms": np.nan,
            "official_epsilon": official_epsilon if official_epsilon is not None else np.nan,
            "posterior_probability_official_epsilon": np.nan,
            "bayesian_response_status": "ERROR",
            "bayesian_response_error": "",
            "rho_treatment": "fixed at normalised historical rho",
            "inference_method": "exact Dirichlet posterior + Monte Carlo propagation",
        }
        for label, epsilon in thresholds:
            base[f"diagnostic_epsilon_{label}"] = epsilon
            base[f"prob_dhat_gt_diagnostic_{label}"] = np.nan
        try:
            x, y = parse_counts_from_row(row, "a"), parse_counts_from_row(row, "b")
            t0 = time.perf_counter()
            posterior = bayesian_d_hat_draws(
                x, y, ctx.rho_normalised[item], ctx.b_matrix,
                draws=draws, prior_alpha=prior_alpha, seed=seed + idx,
            )
            elapsed = (time.perf_counter() - t0) * 1000.0
            base.update(summarise_draws(posterior))
            base["runtime_bayesian_ms"] = elapsed
            for label, epsilon in thresholds:
                base[f"prob_dhat_gt_diagnostic_{label}"] = float(np.mean(posterior > epsilon))
            if official_epsilon is not None:
                base["posterior_probability_official_epsilon"] = float(np.mean(posterior > official_epsilon))
            base["bayesian_response_status"] = "OK"
        except Exception as exc:
            base["bayesian_response_error"] = str(exc)
        rows.append(base)
    return pd.DataFrame(rows), thresholds


def run_level_b(ctx, comparisons: pd.DataFrame, counts_by_item: dict[str, np.ndarray], *, draws: int, prior_alpha: float, seed: int, thresholds: list[tuple[str, float]], official_epsilon: float | None) -> pd.DataFrame:
    rows: list[dict] = []
    rho_cache: dict[str, np.ndarray] = {}
    for item, counts in counts_by_item.items():
        rho_cache[item] = _sample_rho(counts, draws, prior_alpha, seed + abs(hash(item)) % 100_000)

    for idx, row in comparisons.iterrows():
        item = str(row["item"])
        if item not in rho_cache:
            continue
        x, y = parse_counts_from_row(row, "a"), parse_counts_from_row(row, "b")
        out = {
            "comparison_id": row["comparison_id"],
            "item": item,
            "rho_uncertainty_status": "ERROR",
            "rho_uncertainty_error": "",
        }
        try:
            rho_draws = rho_cache[item]
            posterior = bayesian_d_hat_draws_with_rho(
                x, y, rho_draws, ctx.b_matrix,
                prior_alpha=prior_alpha, seed=seed + idx + 900_000,
            )
            out.update(summarise_draws(posterior, prefix="uncertain_rho_dhat"))
            for label, epsilon in thresholds:
                out[f"uncertain_rho_prob_gt_diagnostic_{label}"] = float(np.mean(posterior > epsilon))
            if official_epsilon is not None:
                out["uncertain_rho_prob_gt_official_epsilon"] = float(np.mean(posterior > official_epsilon))

            # Where computationally reasonable, pass a bounded subset of real
            # rho posterior draws through the original network.
            k = min(RHO_NETWORK_DRAWS, len(rho_draws))
            choose = np.linspace(0, len(rho_draws) - 1, k).astype(int)
            alphas, betas = [], []
            for j in choose:
                pred = predict_exp002(ctx, x, y, item, rho_draws[j])
                alphas.append(pred.alpha)
                betas.append(pred.beta)
            for name, values in [("rho_exp002_alpha", np.asarray(alphas)), ("rho_exp002_beta", np.asarray(betas))]:
                out[f"{name}_mean"] = float(np.mean(values))
                out[f"{name}_ci_low"], out[f"{name}_ci_high"] = [float(v) for v in np.quantile(values, [0.025, 0.975])]
            out["rho_network_draws"] = k
            out["rho_uncertainty_status"] = "OK"
        except Exception as exc:
            out["rho_uncertainty_error"] = str(exc)
        rows.append(out)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Bayesian response and optional rho uncertainty evaluation.")
    parser.add_argument("--assets-root", required=True)
    parser.add_argument("--draws", type=int, default=POSTERIOR_DRAWS)
    parser.add_argument("--prior-alpha", type=float, default=DIRICHLET_PRIOR_ALPHA)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--official-epsilon", type=float, default=OFFICIAL_EPSILON)
    parser.add_argument("--transition-counts", default=None, help="Optional explicit genuine test-retest transition-count CSV/JSON.")
    args = parser.parse_args()
    if args.draws < 100:
        raise ValueError("Use at least 100 posterior draws")
    if args.prior_alpha <= 0:
        raise ValueError("--prior-alpha must be positive")

    ensure_output_dirs()
    ctx = load_context(args.assets_root, require_model=False)
    comparisons = pd.read_csv(RESULTS_DIR / "comparisons.csv")

    response_df, thresholds = run_level_a(
        ctx, comparisons, args.draws, args.prior_alpha, args.seed, args.official_epsilon
    )
    response_out = RESULTS_DIR / "bayesian_response_results.csv"
    response_df.to_csv(response_out, index=False)

    transition_path, transition_counts, rejected = _find_genuine_transition_counts(ctx.assets.root, args.transition_counts)
    rho_summary = RESULTS_DIR / "bayesian_rho_summary.md"
    if transition_counts is None:
        template = _write_rho_requirement_template(comparisons["item"].astype(str).unique().tolist())
        with rho_summary.open("w", encoding="utf-8") as f:
            f.write("# Bayesian rho uncertainty\n\n")
            f.write("## RHO UNCERTAINTY: NOT IDENTIFIABLE FROM AVAILABLE DATA\n\n")
            f.write("No usable genuine test-retest transition-count data were found. The supplied rho matrices are probabilities only; they do not identify a Dirichlet concentration or effective sample size.\n\n")
            f.write("RY25 Pre/Post survey exports are not treated as paired test-retest transition counts.\n\n")
            f.write(f"A blank data-requirement template was written to `{template.name}`.\n\n")
            if rejected:
                f.write("### Candidate files rejected\n\n")
                for message in rejected:
                    f.write(f"- {message}\n")
        print("RHO UNCERTAINTY: NOT IDENTIFIABLE FROM AVAILABLE DATA")
    else:
        rho_df = run_level_b(
            ctx, comparisons, transition_counts,
            draws=args.draws, prior_alpha=args.prior_alpha, seed=args.seed,
            thresholds=thresholds, official_epsilon=args.official_epsilon,
        )
        rho_out = RESULTS_DIR / "bayesian_rho_uncertainty.csv"
        rho_df.to_csv(rho_out, index=False)
        with rho_summary.open("w", encoding="utf-8") as f:
            f.write("# Bayesian rho uncertainty\n\n")
            f.write(f"Transition-count source: `{transition_path}`\n\n")
            f.write("Each rho row uses an exact Dirichlet posterior based on real transition counts. Response-distribution and rho uncertainty are jointly propagated through d_hat_norm.\n\n")
            f.write(f"Rows analysed: **{len(rho_df)}**.\n")

    threshold_out = RESULTS_DIR / "diagnostic_thresholds.csv"
    pd.DataFrame([{"label": label, "epsilon": value, "status": "EXPLORATORY — NOT OFFICIAL"} for label, value in thresholds]).to_csv(threshold_out, index=False)

    print(f"Saved: {response_out}")
    print(f"Saved: {threshold_out}")
    print(f"Saved: {rho_summary}")
    print("Inference note: conjugate Dirichlet posteriors are exact; Monte Carlo is used for propagation, not variational inference.")


if __name__ == "__main__":
    main()
