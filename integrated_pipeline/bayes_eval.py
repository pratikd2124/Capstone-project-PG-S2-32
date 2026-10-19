"""Pratik's Bayesian version of the test, shown beside the pipeline. For evaluation only.

The Bayesian question: given the answers we saw, how likely is it that the real gap is
bigger than epsilon? It puts a Dirichlet posterior on each group's answer shares and
pushes the draws through d_norm (Pratik's functions in o3_uncertainty/, unchanged). Here it counts
as "meaningful" when that probability is at least 0.95.

It answers a slightly different question from the pipeline. The pipeline (RYA's method)
treats each group's make-up as fixed and asks how much d_hat moves because students
answer inconsistently. The Bayesian version treats the answer shares themselves as
unknown. The two can disagree, mostly for small groups.

    python -m integrated_pipeline.bayes_eval      # demo cases, then the validation checks
"""
from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .boundary import MEANINGFUL, NOT_SHOWN, _shrunk
from .contracts import ComparisonRequest
from o3_uncertainty.loader import pratik_modules
from .run import build_pipeline
from .scenarios import build_scenarios
from .simulate import B, d_norm, integer_composition
from .uncertainty import perturb_rho_offdiagonal_mass

PRIOR = 0.5          # Pratik's Dirichlet prior
CONFIDENCE = 0.95


def bayesian_result(req, draws=20000, prior=PRIOR, seed=42):
    """Pratik's posterior for one request, and the decision it leads to."""
    _, common, bayes06 = pratik_modules()
    start = time.perf_counter()
    if req.rho_transition_counts is not None:
        rho_draws = bayes06._sample_rho(np.asarray(req.rho_transition_counts, float), draws, prior, seed)
        posterior = common.bayesian_d_hat_draws_with_rho(req.x, req.y, rho_draws, B, prior_alpha=prior, seed=seed)
        level = "B (rho uncertain)"
    else:
        posterior = common.bayesian_d_hat_draws(req.x, req.y, req.P, B, draws=draws, prior_alpha=prior, seed=seed)
        level = "A (rho fixed)"
    posterior = posterior[np.isfinite(posterior)]
    chance_bigger = float(np.mean(posterior > req.epsilon))
    return {
        "bayes_level": level,
        "bayes_d_median": float(np.median(posterior)),
        "bayes_ci95": (float(np.quantile(posterior, 0.025)), float(np.quantile(posterior, 0.975))),
        "bayes_P_d_gt_eps": chance_bigger,
        "bayes_decision": MEANINGFUL if chance_bigger >= CONFIDENCE else NOT_SHOWN,
        "bayes_ms": 1000 * (time.perf_counter() - start),
    }


def compare_demo(project_root=None, epsilon=0.1):
    """The pipeline's decision and the Bayesian decision on each demo case."""
    warnings.filterwarnings("ignore")
    pipeline = build_pipeline(project_root)
    rows = []
    for case in build_scenarios(pipeline.net.predictor.p_matrices, pipeline.net.predictor.paths.instance_dir, epsilon):
        result = pipeline.run(case.request)
        row = {"scenario": case.name, "n_X": int(case.request.x.sum()), "n_Y": int(case.request.y.sum()),
               "pipeline_flag": result.reliability_flag, "pipeline_p": result.p_value,
               "pipeline_decision": result.decision}
        if result.decision is None:
            row.update(bayes_decision=None, agree=None, note="rejected before testing")
        else:
            row.update(bayesian_result(case.request))
            row["agree"] = row["bayes_decision"] == result.decision
        rows.append(row)
    return pd.DataFrame(rows)


def validate_functions(P):
    """Simple checks that Pratik's functions agree with the pipeline's own maths."""
    _, common, bayes06 = pratik_modules()
    x, y = np.array([40., 52, 7, 14]), np.array([9., 17, 54, 12])
    fixed = common.bayesian_d_hat_draws(x, y, P, B, draws=4000, prior_alpha=PRIOR, seed=1)
    same_rho = common.bayesian_d_hat_draws_with_rho(x, y, np.repeat(P[None], 4000, 0), B, prior_alpha=PRIOR, seed=1)
    widths = []
    for k in (1, 10, 100):
        d = common.bayesian_d_hat_draws(x * k, y * k, P, B, draws=4000, prior_alpha=PRIOR, seed=2)
        widths.append(round(float(np.quantile(d, 0.975) - np.quantile(d, 0.025)), 4))
    rho = bayes06._sample_rho(np.full((4, 4), 25.0), 2000, PRIOR, 3)
    ours = d_norm(x / x.sum(), y / y.sum(), x.sum(), y.sum(), P)
    checks = [
        ("his d_hat_norm == pipeline d_norm (ry9 example)", abs(common.d_hat_norm(x, y, P, B) - ours), "< 1e-12"),
        ("Level B with rho fixed == Level A (same seed)", float(np.max(np.abs(fixed - same_rho))), "== 0"),
        ("credible-interval width, n x1 / x10 / x100", tuple(widths), "shrinks"),
        ("rho posterior rows sum to 1 (max error)", float(np.max(np.abs(rho.sum(axis=2) - 1))), "< 1e-12"),
        ("perturbation at 0% returns rho unchanged", float(np.max(np.abs(perturb_rho_offdiagonal_mass(P, 0) - P))), "== 0"),
    ]
    return pd.DataFrame(checks, columns=["check", "value", "expected"])


def coverage_check(P, p, q, n, m, reps=200, seed=0):
    """How often the 95% credible interval contains the true d_norm (should be about 0.95),
    with answer counts drawn the way Pratik's posterior assumes."""
    _, common, _ = pratik_modules()
    rng = np.random.default_rng(seed)
    true_d = d_norm(np.asarray(p), np.asarray(q), n, m, P)
    hits = 0
    for i in range(reps):
        x, y = rng.multinomial(n, p), rng.multinomial(m, q)
        low, high = np.quantile(common.bayesian_d_hat_draws(x, y, P, B, draws=2000, prior_alpha=PRIOR, seed=i),
                                [0.025, 0.975])
        hits += low <= true_d <= high
    return hits / reps


def boundary_error_rates(pipeline, P, p, q, n, m, epsilon, reps=400, seed=0):
    """Put the true gap exactly at epsilon, simulate data with RYA's model, and count how
    often each method still says "meaningful". A 5% test should do so about 5% of the time."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    s = brentq(lambda s: d_norm(*_shrunk(p, q, s), n, m, P) - epsilon, 0, 1)
    types_x, types_y = (integer_composition(v, size) for v, size in zip(_shrunk(p, q, s), (n, m)))
    rows = P / P.sum(axis=1, keepdims=True)
    rng = np.random.default_rng(seed)
    pipeline_hits = bayes_hits = 0
    for i in range(reps):
        x = sum(rng.multinomial(int(types_x[j]), rows[j]) for j in range(4) if types_x[j])
        y = sum(rng.multinomial(int(types_y[j]), rows[j]) for j in range(4) if types_y[j])
        req = ComparisonRequest.make("ry9", x, y, P, epsilon=epsilon)
        pipeline_hits += pipeline.run(req).decision == MEANINGFUL
        bayes_hits += bayesian_result(req, draws=4000, seed=i)["bayes_decision"] == MEANINGFUL
    return {"n": n, "m": m, "reps": reps, "pipeline_rate": pipeline_hits / reps, "bayes_rate": bayes_hits / reps}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=None, help="Baseline data folder (default: downloaded from GitHub)")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--out", default="bayes_vs_pipeline.csv")
    parser.add_argument("--skip-rates", action="store_true", help="skip the slower boundary error-rate check")
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 55)

    df = compare_demo(args.project_root, args.epsilon)
    columns = ["scenario", "n_X", "n_Y", "pipeline_p", "pipeline_decision", "bayes_P_d_gt_eps", "bayes_decision", "agree"]
    print("Pipeline vs Bayesian analogue (evaluation only)\n")
    print(df[columns].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    df.to_csv(args.out, index=False)

    pipeline = build_pipeline(args.project_root, uncertainty=False)
    P = pipeline.net.predictor.p_matrices["ry9"]
    print("\nValidation of Pratik's functions\n")
    print(validate_functions(P).to_string(index=False))
    print("\n95% credible-interval coverage of the true d_norm (should be about 0.95):")
    for n, m in ((30, 30), (200, 200), (25, 2000)):
        print(f"  n={n}, m={m}: {coverage_check(P, [.2, .3, .3, .2], [.25, .3, .25, .2], n, m):.2f}")
    if not args.skip_rates:
        print("\nHow often each method says 'meaningful difference' when the true gap is exactly epsilon")
        print("(data from RYA's model; a 5% test should be at or below about 0.05):")
        for n, m in ((100, 100), (25, 2000)):
            print("  ", boundary_error_rates(pipeline, P, [.05, .2, .35, .4], [.25, .3, .25, .2], n, m, args.epsilon))
    print(f"\nTable: {args.out}")


if __name__ == "__main__":
    main()
