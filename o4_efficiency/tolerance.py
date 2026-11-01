"""O4 (Murat): evidence for an equivalence tolerance (Objective 4, success criterion 4).

How much do cheaper settings move results, compared with the noise the method already has?

    python -m o4_efficiency.cli tolerance          # report on screen + tolerance.json

Three yardsticks, all measured on the benchmark's random comparisons:

* fallback route: q95 error and decision changes at fewer simulation draws, against a
  40,000-draw reference, and how many decisions change from the seed alone (noise already there);
* network route: fp16 against fp32 weights;
* Pratik's rho range: how often it already says a decision could flip.

Then the whole pipeline is re-run with fewer fallback and rho-variant draws. The settings tried
here are proposals; the defaults in PipelineConfig are not changed.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

REFERENCE_DRAWS, REFERENCE_SEED = 40_000, 999
FALLBACK_DRAWS = (4000, 2000, 1000)               # 4000 is PipelineConfig's default
OTHER_SEED = 1                                    # PipelineConfig uses simulated_null's seed 2026
PIPELINE_SETTINGS = {
    "current (4000 / 1500)": {},
    "fallback 2000 / variants 1500": {"fallback_draws": 2000},
    "fallback 2000 / variants 750": {"fallback_draws": 2000, "uncertainty_draws": 750},
    "fallback 1000 / variants 500": {"fallback_draws": 1000, "uncertainty_draws": 500},
}


def _fallback_draws(pipe, cases) -> pd.DataFrame:
    from integrated_pipeline.boundary import find_boundary, minimal_effects_test
    from integrated_pipeline.nulls import simulated_null

    def run(req, b, draws, seed):
        start = time.perf_counter()
        null = simulated_null(req, b, pipe.cfg.fallback_method, draws, seed=seed)
        return null.q95, minimal_effects_test(req, b, null).decision, 1e3 * (time.perf_counter() - start)

    per_case = []
    for req in cases:
        b = find_boundary(req, pipe.cfg.test_mode)
        q_ref, d_ref, _ = run(req, b, REFERENCE_DRAWS, REFERENCE_SEED)
        per_case.append((q_ref, d_ref, {(n, s): run(req, b, n, s) for n in FALLBACK_DRAWS for s in (2026, OTHER_SEED)}))

    rows = []
    for n in FALLBACK_DRAWS:
        err = np.array([abs(r[(n, 2026)][0] - q) / q for q, _, r in per_case if q > 0])
        rows.append({
            "draws": n,
            "q95_error_median": np.median(err) if err.size else np.nan,
            "q95_error_p95": np.percentile(err, 95) if err.size else np.nan,
            "decisions_differ_from_reference": np.mean([r[(n, 2026)][1] != d for _, d, r in per_case]),
            "decisions_differ_from_4000": np.mean([r[(n, 2026)][1] != r[(4000, 2026)][1] for _, _, r in per_case]),
            "decisions_changed_by_seed": np.mean([r[(n, 2026)][1] != r[(n, OTHER_SEED)][1] for _, _, r in per_case]),
            "ms_per_simulation": np.median([r[(n, 2026)][2] for _, _, r in per_case]),
        })
    return pd.DataFrame(rows)


def _fp16(pipe, fp16, cases) -> pd.DataFrame:
    from integrated_pipeline.boundary import find_boundary, minimal_effects_test

    bs = [find_boundary(r, pipe.cfg.test_mode) for r in cases]
    a, b = pipe.net.many(cases, bs), fp16.net.many(cases, bs)
    change = np.array([abs(x.q95 - y.q95) / x.q95 for x, y in zip(a, b) if x.q95 > 0])
    flips = [minimal_effects_test(r, bd, x).decision != minimal_effects_test(r, bd, y).decision
             for r, bd, x, y in zip(cases, bs, a, b)]
    return pd.DataFrame([{
        "comparisons": len(cases),
        "q95_change_median": np.median(change) if change.size else np.nan,
        "q95_change_p95": np.percentile(change, 95) if change.size else np.nan,
        "q95_change_max": change.max() if change.size else np.nan,
        "decisions_changed": np.mean(flips) if flips else np.nan,
    }])


def _whole_pipeline(project_root, cases, base) -> pd.DataFrame:
    from integrated_pipeline.run import build_pipeline

    rows = []
    for name, settings in PIPELINE_SETTINGS.items():
        pipe = build_pipeline(project_root, **settings)
        for r in cases[:3]:                                     # warm-up
            pipe.run(r)
        start = time.perf_counter()
        res = [pipe.run(r) for r in cases]
        rows.append({
            "settings": name,
            "ms_per_comparison": 1e3 * (time.perf_counter() - start) / len(cases),
            "decisions_changed": np.mean([x.decision != y.decision for x, y in zip(base, res)]),
            "flags_changed": np.mean([x.reliability_flag != y.reliability_flag for x, y in zip(base, res)]),
        })
    return pd.DataFrame(rows)


def run_tolerance_study(project_root=None, n_cases: int = 3000, seed: int = 11) -> dict:
    from integrated_pipeline.benchmark import make_cases
    from integrated_pipeline.run import build_pipeline

    with np.errstate(all="ignore"):
        pipe = build_pipeline(project_root)
        cases = make_cases(pipe.net.predictor.p_matrices, n_cases=n_cases, seed=seed)
        base = [pipe.run(r) for r in cases]
        route = [r.provenance.get("route", "?") for r in base]
        decided = [r.decision is not None for r in base]
        by_route = {k: [c for c, rt, d in zip(cases, route, decided) if rt == k and d] for k in ("PASS", "FLAGGED")}

        rho = pd.DataFrame([{
            "route": k,
            "comparisons": len(v),
            "rho_range_could_flip": np.mean([not r.uncertainty.decision_stable for r, rt, d in zip(base, route, decided)
                                             if rt == k and d and r.uncertainty is not None]) if v else np.nan,
        } for k, v in by_route.items()])
        return {
            "n_cases": len(cases),
            "rho_range": rho,
            "fallback_draws": _fallback_draws(pipe, by_route["FLAGGED"]),
            "fp16": _fp16(pipe, build_pipeline(project_root, weights="fp16"), by_route["PASS"]),
            "whole_pipeline": _whole_pipeline(project_root, cases, base),
        }


def print_report(result: dict) -> None:
    pd.set_option("display.width", 200)
    pct = lambda v: f"{100 * v:.2f}%"
    print(f"Equivalence evidence, {result['n_cases']} benchmark comparisons\n")
    print("Yardstick: decisions Pratik's rho range (+/-5% error mass) says could flip")
    print(result["rho_range"].to_string(index=False, formatters={"rho_range_could_flip": pct}))
    print(f"\nFallback route: simulation draws vs a {REFERENCE_DRAWS:,}-draw reference")
    print(result["fallback_draws"].to_string(index=False, float_format=lambda v: f"{v:.4f}",
                                             formatters={k: pct for k in result["fallback_draws"].columns
                                                         if k not in ("draws", "ms_per_simulation")}))
    print("\nNetwork route: fp16 vs fp32 weights")
    print(result["fp16"].to_string(index=False, formatters={k: pct for k in result["fp16"].columns if k != "comparisons"}))
    print("\nWhole pipeline with fewer draws (changes counted against the current settings)")
    print(result["whole_pipeline"].to_string(index=False, float_format=lambda v: f"{v:.2f}",
                                             formatters={"decisions_changed": pct, "flags_changed": pct}))
