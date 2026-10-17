"""Run the synthetic scenarios through the original (direct) and the robust pipeline.

For each case the "truth" is RYA's own simulator at the same boundary pair with 20,000
draws, so the reader can see which configuration got the p-value right.

    python -m scrp_toolkit.pipeline.demo            # prints a table, writes pipeline_demo.csv
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from ..config import resolve_project_paths
from .boundary import find_boundary
from .nulls import simulated_null
from .run import Pipeline, PipelineConfig, build_pipeline
from .scenarios import build_scenarios


def truth_p(req, test_mode="boundary", draws=20000):
    bp = find_boundary(req, test_mode)
    if not np.isfinite(bp.d_norm_observed):
        return float("nan")
    return simulated_null(req, bp, "mc_empirical", draws, seed=99).sf(bp.d_norm_observed)


def run_demo(project_root="data", epsilon=0.1, weights="fp32", test_mode="boundary") -> pd.DataFrame:
    warnings.filterwarnings("ignore")
    robust = build_pipeline(project_root, weights=weights, test_mode=test_mode)
    direct = Pipeline(robust.net.predictor, PipelineConfig(mode="direct", test_mode=test_mode), gate=robust.gate)
    paths = resolve_project_paths(project_root)
    rows = []
    for sc in build_scenarios(robust.net.predictor.p_matrices, paths.instance_dir, epsilon):
        r = robust.run(sc.request)
        d = direct.run(sc.request)
        valid = r.rho_check.status != "reject"
        t = truth_p(sc.request, test_mode) if valid else float("nan")
        rows.append({
            "scenario": sc.name, "objective": sc.objective, "expected": sc.expect,
            "n_X": int(sc.request.x.sum()), "n_Y": int(sc.request.y.sum()),
            "route": r.summary()["route"], "null": r.summary()["null_method"],
            "d_hat": r.summary()["d_hat_norm"],
            "p_direct": d.p_value, "p_robust": r.p_value, "p_truth": t,
            "p_range": r.uncertainty_range,
            "shape_range": r.summary()["shape_range"], "rate_range": r.summary()["rate_range"],
            "decision_direct": d.decision, "decision_robust": r.decision,
            "decision_truth": None if not np.isfinite(t) else
            ("meaningful difference" if t < sc.request.alpha else "no meaningful difference shown"),
            "flag": r.reliability_flag, "why": "; ".join(r.flag_reasons),
            "ms_robust": round(r.provenance.get("total_ms", np.nan), 1),
        })
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", default="data")
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--weights", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--test-mode", default="boundary", choices=["boundary", "point_null"])
    ap.add_argument("--out", default="pipeline_demo.csv")
    a = ap.parse_args(argv)
    df = run_demo(a.project_root, a.epsilon, a.weights, a.test_mode)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 60)
    show = df[["scenario", "n_X", "n_Y", "route", "null", "p_direct", "p_robust", "p_truth", "p_range", "flag"]].copy()
    show["p_range"] = show["p_range"].map(lambda v: "-" if v is None else f"{v[0]:.3f}-{v[1]:.3f}")
    for c in ("p_direct", "p_robust", "p_truth"):
        show[c] = show[c].map(lambda v: "-" if v is None or not np.isfinite(v) else f"{v:.3f}")
    print(show.to_string(index=False))
    df.to_csv(a.out, index=False)
    print(f"\nFull table: {a.out}")


if __name__ == "__main__":
    main()
