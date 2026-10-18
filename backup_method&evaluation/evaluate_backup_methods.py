"""Run October 9 backup-method comparisons against September 25 candidate cases.

Examples:
  python evaluate_backup_methods.py --demo
  python evaluate_backup_methods.py --case-file ../Gamma_Fit_Evaluation_and_Failure_Cases/results/worst_10_gamma_cases.csv --p-file ../Gamma_Fit_Evaluation_and_Failure_Cases/P.RData
  python evaluate_backup_methods.py --draws-dir validated_draws
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backup_methods import compare_estimates
from legacy_925.analyse_real_instances import simulated_statistics


def load_p_matrices(path: Path):
    """Load P from RDS (sometimes mislabeled .RData), RData, or long-form CSV."""
    if not path.is_file():
        raise FileNotFoundError(f"P matrix file not found: {path}")
    if path.suffix.lower() == ".csv":
        from legacy_925.analyse_real_instances import load_p_matrices as load_csv
        # Original 9.25 loader rejects rows rounded to 2 decimals. Use explicit
        # normalization here while still rejecting substantial invalidity.
        table = pd.read_csv(path)
        required = {"question", "latent_type", "response_option", "probability"}
        if not required.issubset(table.columns):
            raise ValueError(f"P CSV needs columns {sorted(required)}")
        matrices = {}
        for question, group in table.groupby("question"):
            if len(group) != 16:
                continue
            p = np.full((4, 4), np.nan)
            for row in group.itertuples(index=False):
                i, j = int(row.latent_type), int(row.response_option)
                if i not in range(1, 5) or j not in range(1, 5) or np.isfinite(p[i-1, j-1]):
                    raise ValueError(f"Invalid/duplicate P matrix cell for {question}")
                p[i-1, j-1] = float(row.probability)
            matrices[str(question)] = normalize_p(p, str(question))
        return matrices
    try:
        import rdata
    except ImportError as exc:
        raise ImportError("For P.RData run: %pip install rdata (Jupyter) or pip install rdata (Terminal)") from exc
    # The team's P.RData has previously been identified as an RDS file.
    try:
        obj = rdata.read_rds(str(path))
    except Exception:
        obj = rdata.read_rda(str(path))
    if not isinstance(obj, dict):
        raise ValueError("Expected P.RData to contain a named dictionary of P matrices")
    return {str(key): normalize_p(np.asarray(value, dtype=float), str(key))
            for key, value in obj.items() if np.asarray(value).shape == (4, 4)}


def normalize_p(p, question):
    if p.shape != (4, 4) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError(f"{question}: P must be a finite nonnegative 4x4 matrix")
    sums = p.sum(axis=1)
    # Allow original matrices rounded to 2 decimal places, not gross errors.
    if (sums <= 0).any() or not np.allclose(sums, 1, rtol=0, atol=0.021):
        raise ValueError(f"{question}: P rows do not sum close to 1: {sums}")
    return p / sums[:, None]


def parse_counts(value, label):
    try:
        x = np.asarray(json.loads(value) if isinstance(value, str) else value, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid {label} counts: {value}") from exc
    if x.shape != (4,) or (x < 0).any() or not np.isfinite(x).all() or not np.allclose(x, np.round(x)):
        raise ValueError(f"{label} must be four nonnegative integer counts")
    return x.astype(int)


def simulate_case(row, p, *, n_gamma, n_mc, n_reference, seed):
    x = parse_counts(row["x_counts"], "x_counts")
    y = parse_counts(row["y_counts"], "y_counts")
    # Independent RNG streams for both candidate methods and reference.
    rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
    g = simulated_statistics(x, y, p, n_gamma, rngs[0])
    m = simulated_statistics(x, y, p, n_mc, rngs[1])
    r = simulated_statistics(x, y, p, n_reference, rngs[2])
    return g, m, r


def demo_cases(seed, n_gamma, n_mc, n_reference):
    rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(6)]
    cases = []
    for i, name in enumerate(("demo_gamma_like", "demo_mixture_non_gamma")):
        streams = rngs[3*i:3*i+3]
        sizes = (n_gamma, n_mc, n_reference)
        if i == 0:
            arrays = [r.gamma(shape=3.5, scale=1.2, size=n) for r, n in zip(streams, sizes)]
        else:
            arrays = []
            for r, n in zip(streams, sizes):
                mix = r.random(n) < 0.22
                values = r.gamma(shape=2, scale=0.8, size=n)
                values[mix] = r.gamma(shape=15, scale=0.6, size=mix.sum())
                arrays.append(values)
        cases.append((name, *arrays))
    return cases


def validated_draw_cases(folder, n_gamma, n_mc, n_reference, seed):
    """Each CSV has one column named 'statistic' and independent simulated draws."""
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise ValueError(f"No *.csv files in {folder}")
    cases = []
    total = n_gamma + n_mc + n_reference
    for idx, file in enumerate(files):
        table = pd.read_csv(file)
        if "statistic" not in table.columns:
            raise ValueError(f"{file.name} must have a 'statistic' column")
        values = table["statistic"].to_numpy(dtype=float)
        if len(values) < total:
            raise ValueError(f"{file.name} needs at least {total} draws; has {len(values)}")
        shuffled = np.random.default_rng(seed + idx).permutation(values)
        g = shuffled[:n_gamma]
        m = shuffled[n_gamma:n_gamma+n_mc]
        r = shuffled[n_gamma+n_mc:n_gamma+n_mc+n_reference]
        cases.append((file.stem, g, m, r))
    return cases


def make_plot(case_id, rows, out_dir):
    good = [r for r in rows if r["status"] == "ok"]
    if not good:
        return
    labels = [r["method"].replace("monte_carlo_empirical", "MC empirical") for r in good]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar(labels, [r["q95_relative_error"] for r in good])
    ax.set(ylabel="q95 relative error", title=f"Backup method comparison: {case_id}")
    fig.tight_layout()
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", case_id)
    fig.savefig(out_dir / f"{safe_name}_q95_error.png", dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true", help="Run synthetic smoke test, no project data")
    mode.add_argument("--case-file", type=Path, help="9.25 worst_10_gamma_cases.csv")
    mode.add_argument("--draws-dir", type=Path, help="Validated-model draws, one CSV per case")
    parser.add_argument("--p-file", type=Path, help="P.RData or P_matrices_long.csv (for --case-file)")
    parser.add_argument("--n-cases", type=int, default=3)
    parser.add_argument("--gamma-draws", type=int, default=2000)
    parser.add_argument("--mc-draws", type=int, default=2000)
    parser.add_argument("--reference-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    if min(args.gamma_draws, args.mc_draws, args.reference_draws) < 100:
        parser.error("All draw counts must be >= 100")
    if args.n_cases < 1:
        parser.error("--n-cases must be >= 1")
    if args.case_file and not args.p_file:
        parser.error("--case-file requires --p-file")
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    case_runs = []
    if args.demo:
        case_runs = demo_cases(args.seed, args.gamma_draws, args.mc_draws, args.reference_draws)
        source_label = "synthetic_demo_only"
    elif args.draws_dir:
        case_runs = validated_draw_cases(args.draws_dir, args.gamma_draws,
                                         args.mc_draws, args.reference_draws, args.seed)[:args.n_cases]
        source_label = "external_model_draws_unverified_by_script"
    else:
        table = pd.read_csv(args.case_file)
        required = {"instance_id", "question", "x_counts", "y_counts"}
        if not required.issubset(table.columns):
            parser.error(f"Case CSV missing {sorted(required - set(table.columns))}")
        matrices = load_p_matrices(args.p_file)
        source_label = "survey_response_as_latent_type_PROXY_NOT_VALIDATED"
        for idx, row in table.head(args.n_cases).iterrows():
            question = str(row["question"])
            if question not in matrices:
                print(f"SKIP {row['instance_id']}: no P matrix for {question}")
                continue
            try:
                case_runs.append((str(row["instance_id"]), *simulate_case(
                    row, matrices[question], n_gamma=args.gamma_draws,
                    n_mc=args.mc_draws, n_reference=args.reference_draws,
                    seed=args.seed + int(idx))))
            except (ValueError, np.linalg.LinAlgError) as exc:
                print(f"SKIP {row['instance_id']}: {exc}")
    if not case_runs:
        raise RuntimeError("No cases were evaluated. Check input files and model assumptions.")
    output_rows = []
    for case_id, g, m, r in case_runs:
        start = time.perf_counter()
        rows = compare_estimates(g, m, r)
        elapsed = time.perf_counter() - start
        for row in rows:
            row.update({"instance_id": case_id, "source_status": source_label,
                        "evaluation_seconds": round(elapsed, 6),
                        "n_gamma_draws": len(g), "n_mc_draws": len(m)})
        output_rows.extend(rows)
        make_plot(case_id, rows, plots_dir)
        print(f"Evaluated {case_id}: " + ", ".join(
            f"{row['method']} q95 error={row.get('q95_relative_error', float('nan')):.2%}"
            for row in rows))
    result = pd.DataFrame(output_rows)
    result.to_csv(out_dir / "backup_method_comparison.csv", index=False)
    notes = f"""# October 9 backup-method evaluation — draft

Source status: **{source_label}**.

## Methods
- Gamma: fit a continuous Gamma MLE on {args.gamma_draws:,} draws.
- Backup: Monte Carlo empirical q95 and empirical survival probability on {args.mc_draws:,} independent draws.
- Reference: q95 and tail rate from {args.reference_draws:,} **independent** draws.
- Both methods evaluated at the **same reference q95 threshold**.

## Important interpretation limits
- The 9.25 candidate-case simulator uses **observed survey responses as proxies for latent types**. It is not a validated reproduction of the original Gemfinder R model. Therefore any `PROXY` results here are exploratory and **cannot establish real-model Gamma failures or successful repairs**.
- To validate, generate statistics from the original latent-type simulator, then use `--draws-dir` with independent draws for each case.
- The reference distribution is itself Monte Carlo estimated, not a known exact truth. For rare-tail probabilities increase simulation counts and quantify Monte Carlo uncertainty.
- This script does **not** measure simulation-generation runtime per method (only estimation time), so it cannot establish overall computational superiority.
- Candidate cases are selected from the 9.25 analysis; evaluation should also include normal controls and multiple seeds before choosing a method.
- Agreement on failure thresholds and backup-method acceptance criteria remains a team decision.

## Outputs
- `backup_method_comparison.csv`: both methods' q95 and tail errors by case.
- `plots/`: q95-error comparisons.

## Next steps
1. Confirm simulation equivalence with original R model / latent-type sampling.
2. Repeat comparisons using validated simulated draws and several random seeds.
3. Check tail accuracy, computation time, and sample-size sensitivity.
4. Report recommendations for October 16 after evidence is sufficient.
"""
    (out_dir / "backup_method_findings.md").write_text(notes, encoding="utf-8")
    print("Saved", out_dir / "backup_method_comparison.csv")


if __name__ == "__main__":
    main()
