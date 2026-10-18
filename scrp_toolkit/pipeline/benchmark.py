"""Benchmark: the original path against the pipeline, both checked against the truth.

Random comparisons are made in four bands of group-size ratio (1-2x, 2-10x, 10-100x,
100-2000x) across the 53 four-option items. "Truth" is RYA's simulator at the same
boundary pair. The summary says, per band, how often each path agreed with the truth,
how many false positives it gave, and how long it took.

    python -m scrp_toolkit.pipeline.benchmark --cases 400
"""
from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd

from .boundary import find_boundary
from .contracts import FLAGGED, ComparisonRequest
from .demo import truth_p
from .nulls import FALLBACK_METHODS, simulated_null
from .run import Pipeline, PipelineConfig, build_pipeline
from .simulate import integer_composition

BANDS = [(1, 2), (2, 10), (10, 100), (100, 2000)]


def make_cases(p_matrices, n_cases=300, epsilon=0.1, seed=5):
    """Random comparisons: a cohort of 300-9,000 and a smaller, slightly different group."""
    rng = np.random.default_rng(seed)
    items = sorted(k for k, v in p_matrices.items() if np.asarray(v).shape == (4, 4) and not k.startswith("__"))
    cases = []
    for i in range(n_cases):
        low, high = BANDS[i % len(BANDS)]
        ratio = float(np.exp(rng.uniform(np.log(low), np.log(high))))
        m = int(rng.integers(300, 9000))
        n = max(5, int(round(m / ratio)))
        cohort = rng.dirichlet(np.ones(4) * 3)
        group = np.clip(cohort + rng.normal(0, 0.12, 4), 0.005, None)
        group /= group.sum()
        item = items[rng.integers(len(items))]
        cases.append(ComparisonRequest.make(item, integer_composition(group, n), integer_composition(cohort, m),
                                            p_matrices[item], epsilon=epsilon, label=f"bench{i}"))
    return cases


def run_benchmark(project_root="data", n_cases=300, truth_draws=10000, epsilon=0.1):
    warnings.filterwarnings("ignore")
    pipeline = build_pipeline(project_root, uncertainty=False)
    original = Pipeline(pipeline.net.predictor, PipelineConfig(mode="direct"), gate=pipeline.gate)
    rows = []
    for req in make_cases(pipeline.net.predictor.p_matrices, n_cases, epsilon):
        old, new = original.run(req), pipeline.run(req)
        n, m = req.x.sum(), req.y.sum()
        rows.append({"item": req.item, "n_X": int(n), "n_Y": int(m), "ratio": max(n, m) / min(n, m),
                     "route": new.screening.route if new.screening else "-",
                     "p_truth": truth_p(req, draws=truth_draws), "p_direct": old.p_value, "p_robust": new.p_value,
                     "ms_direct": old.provenance["total_ms"], "ms_robust": new.provenance["total_ms"]})
    df = pd.DataFrame(rows)
    edges = [low for low, _ in BANDS] + [BANDS[-1][1] * 10]
    df["band"] = pd.cut(df["ratio"], edges, labels=[f"{a}-{b}x" for a, b in BANDS], right=False)
    return df


def summarise(df, alpha=0.05):
    """Per band: agreement with the truth, false positives, error and time, for both paths."""
    def one_band(g):
        known = g["p_truth"].notna()
        truly_different = g.loc[known, "p_truth"] < alpha
        row = {"cases": len(g), "flagged_%": 100 * (g["route"] == FLAGGED).mean()}
        for path in ("direct", "robust"):
            p = g.loc[known, f"p_{path}"]
            has_p = p.notna()
            says_different = p[has_p] < alpha
            row[f"{path}_no_answer"] = int((~has_p).sum())
            row[f"{path}_agree_%"] = 100 * (says_different == truly_different[has_p]).mean()
            row[f"{path}_false_pos"] = int((says_different & ~truly_different[has_p]).sum())
            row[f"{path}_median_abs_p_err"] = float(np.median(np.abs(p[has_p] - g.loc[known, "p_truth"][has_p])))
            row[f"{path}_median_ms"] = float(g[f"ms_{path}"].median())
        return pd.Series(row)

    rows = [one_band(g).rename(band) for band, g in df.groupby("band", observed=True)]
    rows.append(one_band(df).rename("all"))
    return pd.DataFrame(rows).round(3)


def compare_fallback_candidates(project_root="data", n_cases=40, draws=4000, truth_draws=40000):
    """Objective 2: how far each fallback option's 95th percentile is from a big reference
    simulation, on cases that screening sends to the fallback."""
    warnings.filterwarnings("ignore")
    pipeline = build_pipeline(project_root, uncertainty=False)
    rows = []
    for req in make_cases(pipeline.net.predictor.p_matrices, n_cases * 4, seed=17):
        result = pipeline.run(req)
        if result.screening is None or result.screening.route != FLAGGED:
            continue
        boundary = find_boundary(req)
        reference = simulated_null(req, boundary, "mc_empirical", truth_draws, seed=1234).q95
        network = pipeline.net(req, boundary)
        rows.append({"case": req.label, "method": "network (for reference)",
                     "q95_rel_err": abs(network.q95 - reference) / reference, "ms": network.time_ms})
        for method in FALLBACK_METHODS:
            estimate = simulated_null(req, boundary, method, draws, seed=7)
            rows.append({"case": req.label, "method": method,
                         "q95_rel_err": abs(estimate.q95 - reference) / reference, "ms": estimate.time_ms})
        if len({r["case"] for r in rows}) >= n_cases:
            break
    return (pd.DataFrame(rows).groupby("method")
            .agg(cases=("case", "nunique"), median_q95_err=("q95_rel_err", "median"),
                 p90_q95_err=("q95_rel_err", lambda v: v.quantile(0.9)), median_ms=("ms", "median"))
            .round(4))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default="data")
    parser.add_argument("--cases", type=int, default=300)
    parser.add_argument("--truth-draws", type=int, default=10000)
    parser.add_argument("--out", default="pipeline_benchmark.csv")
    args = parser.parse_args(argv)

    start = time.time()
    df = run_benchmark(args.project_root, args.cases, args.truth_draws)
    df.to_csv(args.out, index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(summarise(df).T.to_string())
    print(f"\n{len(df)} cases in {time.time() - start:.0f} s. Per-case table: {args.out}")


if __name__ == "__main__":
    main()
