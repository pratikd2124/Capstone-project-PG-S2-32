"""Run the demo cases three ways: the original path, the pipeline, and the truth.

"Truth" is RYA's own simulator at the same boundary pair with 20,000 draws.

    python -m integrated_pipeline.demo      # prints a table and writes pipeline_demo.csv
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from .boundary import MEANINGFUL, NOT_SHOWN, find_boundary
from .nulls import simulated_null
from .run import Pipeline, PipelineConfig, build_pipeline
from .scenarios import build_scenarios


def truth_p(req, test_mode="boundary", draws=20000):
    """The p-value RYA's own simulator gives, with plenty of draws."""
    boundary = find_boundary(req, test_mode)
    if not np.isfinite(boundary.d_norm_observed):
        return float("nan")
    return simulated_null(req, boundary, "mc_empirical", draws, seed=99).sf(boundary.d_norm_observed)


def run_demo(project_root=None, epsilon=0.1, weights="fp32", test_mode="boundary"):
    warnings.filterwarnings("ignore")
    pipeline = build_pipeline(project_root, weights=weights, test_mode=test_mode)
    original = Pipeline(pipeline.net.predictor, PipelineConfig(mode="direct", test_mode=test_mode), gate=pipeline.gate)
    cases = build_scenarios(pipeline.net.predictor.p_matrices, pipeline.net.predictor.paths.instance_dir, epsilon)

    rows = []
    for case in cases:
        req = case.request
        new, old = pipeline.run(req), original.run(req)
        truth = truth_p(req, test_mode) if new.rho_check.status != "reject" else float("nan")
        row = new.summary()
        rows.append({
            "scenario": case.name, "objective": case.objective, "expected": case.expect,
            "n_X": row["n_X"], "n_Y": row["n_Y"], "route": row["route"], "null": row["null_method"],
            "d_hat": row["d_hat_norm"],
            "p_direct": old.p_value, "p_robust": new.p_value, "p_truth": truth,
            "p_range": new.uncertainty_range,
            "shape_range": row["shape_range"], "rate_range": row["rate_range"],
            "decision_direct": old.decision, "decision_robust": new.decision,
            "decision_truth": (MEANINGFUL if truth < req.alpha else NOT_SHOWN) if np.isfinite(truth) else None,
            "flag": new.reliability_flag, "why": "; ".join(new.flag_reasons),
            "ms_robust": row["time_ms"],
        })
    return pd.DataFrame(rows)


def readable(df):
    """The main columns, with numbers rounded for printing."""
    def number(v):
        return "-" if v is None or not np.isfinite(v) else f"{v:.3f}"

    out = df[["scenario", "n_X", "n_Y", "route", "null", "p_direct", "p_robust", "p_truth", "p_range", "flag"]].copy()
    for column in ("p_direct", "p_robust", "p_truth"):
        out[column] = out[column].map(number)
    out["p_range"] = out["p_range"].map(lambda v: "-" if v is None else f"{v[0]:.3f}-{v[1]:.3f}")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=None, help="Baseline data folder (default: downloaded from GitHub)")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--weights", default="fp32", choices=["fp32", "fp16"])
    parser.add_argument("--test-mode", default="boundary", choices=["boundary", "point_null"])
    parser.add_argument("--out", default="pipeline_demo.csv")
    args = parser.parse_args(argv)

    df = run_demo(args.project_root, args.epsilon, args.weights, args.test_mode)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(readable(df).to_string(index=False))
    df.to_csv(args.out, index=False)
    print(f"\nFull table: {args.out}")


if __name__ == "__main__":
    main()
