"""O4 (Murat): what the integrated pipeline costs, compared with the original direct-to-network path.

Answers three of Objective 4's success criteria: how cost splits across the pipeline's stages,
what the extra stages (screening, fallback, rho range) add at annual volume, and whether the
fp16 model gives the same decisions through the pipeline's own boundary test.

    python -m o4_efficiency.cli pipeline-cost      # report on screen + pipeline_cost.json

Every number is measured on the benchmark's random comparisons (integrated_pipeline/benchmark.py)
on the machine it runs on. "Per year" multiplies the mean time per comparison by the number of
comparisons a year; how many comparisons ~200,000 responses turn into is still to be confirmed,
so the projection is given for a range.
"""
from __future__ import annotations

import cProfile
import pstats
import time
from typing import Dict, List

import numpy as np
import pandas as pd

# (label, function name, file it lives in): stages measured as cumulative time. They nest, so
# the shares don't add up to 100%.
STAGES = [
    ("rho check (O1)", "validate_rho", "screen.py"),
    ("boundary pair", "find_boundary", "boundary.py"),
    ("screening (O1)", "screen", "screen.py"),
    ("network null (O4)", "many", "nulls.py"),
    ("network null, single (O4)", "__call__", "nulls.py"),
    ("simulated null (O2)", "simulated_null", "nulls.py"),
    ("minimal-effects test", "minimal_effects_test", "boundary.py"),
    ("rho range (O3)", "propagate", "uncertainty.py"),
]

COMPARISONS_PER_YEAR = (200_000, 1_000_000, 8_000_000)   # 8M = 200k responses x 40 items


def _time_runs(pipeline, cases) -> tuple[float, list]:
    start = time.perf_counter()
    results = [pipeline.run(r) for r in cases]
    return (time.perf_counter() - start) / len(cases) * 1e3, results


def _stage_table(pipeline, cases) -> pd.DataFrame:
    prof = cProfile.Profile()
    prof.enable()
    for r in cases:
        pipeline.run(r)
    prof.disable()
    stats = pstats.Stats(prof).stats
    total = sum(v[2] for v in stats.values())            # all own-time = total profiled time
    rows = []
    for label, name, file in STAGES:
        hits = [v for (f, _l, n), v in stats.items() if n == name and f.endswith(file)]
        seconds, calls = sum(v[3] for v in hits), sum(v[1] for v in hits)
        if calls:
            rows.append({"stage": label, "share_of_time": seconds / total, "calls_per_comparison": calls / len(cases)})
    return pd.DataFrame(rows)


def run_pipeline_cost(project_root=None, n_cases: int = 200, seed: int = 5) -> dict:
    from integrated_pipeline.benchmark import make_cases
    from integrated_pipeline.run import build_pipeline

    with np.errstate(all="ignore"):
        direct = build_pipeline(project_root, mode="direct")
        robust = {w: build_pipeline(project_root, weights=w) for w in ("fp32", "fp16")}
        cases = make_cases(direct.net.predictor.p_matrices, n_cases=n_cases, seed=seed)
        for p in (direct, *robust.values()):                 # warm-up: sessions, caches
            for r in cases[:3]:
                p.run(r)

        rows: List[Dict] = []
        ms, _ = _time_runs(direct, cases)
        rows.append({"path": "original: direct to network", "ms_per_comparison": ms})
        results = None
        for w, p in robust.items():
            ms, res = _time_runs(p, cases)
            rows.append({"path": f"pipeline ({w} weights)", "ms_per_comparison": ms})
            results = results or res
        paths = pd.DataFrame(rows)
        paths["x_original"] = paths["ms_per_comparison"] / paths["ms_per_comparison"].iloc[0]
        for n in COMPARISONS_PER_YEAR:
            paths[f"cpu_minutes_for_{n}"] = paths["ms_per_comparison"] * n / 60_000

        route = pd.Series([r.provenance.get("route", "?") for r in results])
        total_ms = pd.Series([r.provenance.get("total_ms", np.nan) for r in results])
        routes = (pd.DataFrame({"route": route, "ms": total_ms}).groupby("route")["ms"]
                  .agg(comparisons="size", median_ms="median", mean_ms="mean").reset_index())
        routes["share_of_time"] = routes["comparisons"] * routes["mean_ms"] / total_ms.sum()

        stages = _stage_table(robust["fp32"], cases)
    return {"n_cases": len(cases), "paths": paths, "routes": routes, "stages": stages}


def decision_agreement(project_root=None, n_cases: int = 1000, seed: int = 11) -> pd.DataFrame:
    """fp16 vs fp32 decisions through the whole pipeline (boundary test), per route.

    Yardstick: how often Pratik's rho range (+/-5% error mass, step 3+) already says the fp32
    decision could flip. A model change well below that is smaller than the uncertainty the
    method carries anyway.
    """
    from integrated_pipeline.benchmark import make_cases
    from integrated_pipeline.run import build_pipeline

    with np.errstate(all="ignore"):
        fp32, fp16 = build_pipeline(project_root, weights="fp32"), build_pipeline(project_root, weights="fp16")
        cases = make_cases(fp32.net.predictor.p_matrices, n_cases=n_cases, seed=seed)
        a, b = [fp32.run(r) for r in cases], [fp16.run(r) for r in cases]
    df = pd.DataFrame({
        "route": [r.provenance.get("route", "?") for r in a],
        "has_decision": [r.p_value is not None for r in a],
        "flipped": [x.decision != y.decision for x, y in zip(a, b)],
        "rho_range_flips": [r.uncertainty is not None and not r.uncertainty.decision_stable for r in a],
        "abs_p_change": [abs(x.p_value - y.p_value) if x.p_value is not None and y.p_value is not None else np.nan
                         for x, y in zip(a, b)],
    })
    df = df[df["has_decision"]]

    def row(name, g):
        return {"comparisons": name, "n": len(g), "fp16_decisions_changed": int(g["flipped"].sum()),
                "fp16_share_changed": g["flipped"].mean(), "rho_range_share_could_flip": g["rho_range_flips"].mean(),
                "median_abs_p_change": g["abs_p_change"].median()}

    rows = [row("all with a decision", df)] + [row(f"route {k}", g) for k, g in df.groupby("route")]
    return pd.DataFrame(rows)


def print_report(result: dict) -> None:
    pd.set_option("display.width", 200)
    print(f"Integrated pipeline cost, {result['n_cases']} benchmark comparisons\n")
    print(result["paths"].to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print("\nBy route (fp32):")
    print(result["routes"].to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print("\nBy stage (fp32; stages nest, so shares overlap):")
    print(result["stages"].to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    if result.get("equivalence") is not None:
        print("\nfp16 vs fp32 decisions through the pipeline (yardstick: rho range +/-5%):")
        print(result["equivalence"].to_string(index=False, float_format=lambda v: f"{v:.4g}"))


def to_json_dict(result: dict) -> dict:
    return {k: (v.to_dict(orient="records") if isinstance(v, pd.DataFrame) else v) for k, v in result.items()}
