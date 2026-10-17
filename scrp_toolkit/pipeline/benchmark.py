"""Benchmark: original direct-to-network path vs the robust pipeline, against simulated truth.

Synthetic comparisons are drawn across group-size ratio bands on the 53 four-option items.
"Truth" is RYA's simulator at the same boundary pair (default 10,000 draws). For each case we
record both configurations' p-values, decisions and time, then summarise by ratio band.

    python -m scrp_toolkit.pipeline.benchmark --cases 300
"""
from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd

from .contracts import ComparisonRequest as R
from .demo import truth_p
from .run import Pipeline, PipelineConfig, build_pipeline
from .simulate import integer_composition

BANDS = [(1, 2), (2, 10), (10, 100), (100, 2000)]


def make_cases(p_matrices, n_cases=300, epsilon=0.1, seed=5):
    rng = np.random.default_rng(seed)
    items = sorted(k for k, v in p_matrices.items() if np.asarray(v).shape == (4, 4) and not k.startswith("__"))
    out = []
    for i in range(n_cases):
        lo, hi = BANDS[i % len(BANDS)]
        ratio = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        m = int(rng.integers(300, 9000))
        n = max(5, int(round(m / ratio)))
        q = rng.dirichlet(np.ones(4) * 3)
        p = np.clip(q + rng.normal(0, 0.12, 4), 0.005, None)   # a shifted classroom
        p /= p.sum()
        item = items[rng.integers(len(items))]
        out.append(R.make(item, integer_composition(p, n), integer_composition(q, m), p_matrices[item],
                          epsilon=epsilon, label=f"bench{i}"))
    return out


def run_benchmark(project_root="data", n_cases=300, truth_draws=10000, epsilon=0.1) -> pd.DataFrame:
    warnings.filterwarnings("ignore")
    robust = build_pipeline(project_root, uncertainty=False)
    direct = Pipeline(robust.net.predictor, PipelineConfig(mode="direct"), gate=robust.gate)
    rows = []
    for req in make_cases(robust.net.predictor.p_matrices, n_cases, epsilon):
        d = direct.run(req)
        r = robust.run(req)
        t = truth_p(req, draws=truth_draws)
        n, m = req.x.sum(), req.y.sum()
        rows.append({"item": req.item, "n_X": int(n), "n_Y": int(m), "ratio": max(n, m) / min(n, m),
                     "route": r.screening.route if r.screening else "-", "p_truth": t,
                     "p_direct": d.p_value, "p_robust": r.p_value,
                     "ms_direct": d.provenance["total_ms"], "ms_robust": r.provenance["total_ms"]})
    df = pd.DataFrame(rows)
    df["band"] = pd.cut(df["ratio"], [b[0] for b in BANDS] + [BANDS[-1][1] * 10],
                        labels=[f"{a}-{b}x" for a, b in BANDS], right=False)
    return df


def summarise(df: pd.DataFrame, alpha=0.05) -> pd.DataFrame:
    def agg(g):
        ok = g["p_truth"].notna()
        tr = g.loc[ok, "p_truth"] < alpha
        out = {"cases": len(g), "flagged_%": 100 * (g["route"] == "FLAGGED").mean()}
        for k in ("direct", "robust"):
            p = g.loc[ok, f"p_{k}"]
            valid = p.notna()
            out[f"{k}_no_answer"] = int((~valid).sum())
            out[f"{k}_agree_%"] = 100 * ((p[valid] < alpha) == tr[valid]).mean()
            out[f"{k}_false_pos"] = int(((p[valid] < alpha) & ~tr[valid]).sum())
            out[f"{k}_median_abs_p_err"] = float(np.median(np.abs(p[valid] - g.loc[ok, "p_truth"][valid])))
            out[f"{k}_median_ms"] = float(g[f"ms_{k}"].median())
        return pd.Series(out)
    parts = [agg(g).rename(b) for b, g in df.groupby("band", observed=True)]
    parts.append(agg(df).rename("all"))
    return pd.DataFrame(parts).round(3)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", default="data")
    ap.add_argument("--cases", type=int, default=300)
    ap.add_argument("--truth-draws", type=int, default=10000)
    ap.add_argument("--out", default="pipeline_benchmark.csv")
    a = ap.parse_args(argv)
    t0 = time.time()
    df = run_benchmark(a.project_root, a.cases, a.truth_draws)
    df.to_csv(a.out, index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(summarise(df).T.to_string())
    print(f"\n{len(df)} cases in {time.time() - t0:.0f} s. Per-case table: {a.out}")


if __name__ == "__main__":
    main()


def compare_fallback_candidates(project_root="data", n_cases=40, draws=4000, truth_draws=40000) -> pd.DataFrame:
    """Objective 2: each fallback candidate's q95 error against a large independent simulation,
    on the cases screening sends to the fallback."""
    from .boundary import find_boundary
    from .nulls import FALLBACK_METHODS, simulated_null
    warnings.filterwarnings("ignore")
    pl = build_pipeline(project_root, uncertainty=False)
    rows = []
    for req in make_cases(pl.net.predictor.p_matrices, n_cases * 4, seed=17):
        res = pl.run(req)
        if res.screening is None or res.screening.route != "FLAGGED":
            continue
        bp = find_boundary(req)
        ref = simulated_null(req, bp, "mc_empirical", truth_draws, seed=1234)
        nn = pl.net(req, bp)
        rows.append({"case": req.label, "method": "network (for reference)",
                     "q95_rel_err": abs(nn.q95 - ref.q95) / ref.q95, "ms": nn.time_ms})
        for m in FALLBACK_METHODS:
            est = simulated_null(req, bp, m, draws, seed=7)
            rows.append({"case": req.label, "method": m, "q95_rel_err": abs(est.q95 - ref.q95) / ref.q95,
                         "ms": est.time_ms})
        if len({r["case"] for r in rows}) >= n_cases:
            break
    df = pd.DataFrame(rows)
    return df.groupby("method").agg(cases=("case", "nunique"), median_q95_err=("q95_rel_err", "median"),
                                    p90_q95_err=("q95_rel_err", lambda v: v.quantile(0.9)),
                                    median_ms=("ms", "median")).round(4)
