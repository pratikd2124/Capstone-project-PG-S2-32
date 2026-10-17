"""Objective 3 evaluation: Pratik's Bayesian analogue beside the pipeline. NOT on the deployed path.

Pratik's functions (o3/, unchanged from branch ``pratik`` 9f113d6):
* ``bayesian_d_hat_draws``           Dirichlet posterior on each group's answer shares, pushed
                                      through the scaled d_hat (rho fixed) - his "Level A";
* ``bayesian_d_hat_draws_with_rho``  the same with rho drawn from its own posterior - "Level B";
* ``_sample_rho``                    Dirichlet posterior of rho from test-retest counts.

Bayesian decision used here (one-sided analogue of the pipeline's alpha = 0.05):
"meaningful difference" when P(d_norm > epsilon | data) >= 0.95.

The two approaches model different noise. The pipeline (RYA's method) holds each group's
composition fixed and asks how much d_hat moves because students answer inconsistently
(rho). Pratik's posterior treats the answer shares as unknown and ignores retest noise
except through the covariance in d_hat. So they can disagree, mostly for small groups.

    python -m scrp_toolkit.pipeline.bayes_eval            # demo cases + validation checks
"""
from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd

from ..config import resolve_project_paths
from .o3.loader import pratik_modules
from .run import build_pipeline
from .scenarios import build_scenarios
from .simulate import B, d_norm, integer_composition, simulate_d_hat

PRIOR = 0.5            # Pratik's DIRICHLET_PRIOR_ALPHA
CONFIDENCE = 0.95


def bayesian_result(req, draws: int = 20000, prior: float = PRIOR, seed: int = 42) -> dict:
    """Pratik's posterior of d_norm for one request, and the Bayesian decision."""
    _, common, b06 = pratik_modules()
    t0 = time.perf_counter()
    if req.rho_transition_counts is not None:
        rho_draws = b06._sample_rho(np.asarray(req.rho_transition_counts, float), draws, prior, seed)
        post = common.bayesian_d_hat_draws_with_rho(req.x, req.y, rho_draws, B, prior_alpha=prior, seed=seed)
        level = "B (rho uncertain)"
    else:
        post = common.bayesian_d_hat_draws(req.x, req.y, req.P, B, draws=draws, prior_alpha=prior, seed=seed)
        level = "A (rho fixed)"
    post = post[np.isfinite(post)]
    prob = float(np.mean(post > req.epsilon))
    return {
        "bayes_level": level,
        "bayes_d_median": float(np.median(post)),
        "bayes_ci95": (float(np.quantile(post, 0.025)), float(np.quantile(post, 0.975))),
        "bayes_P_d_gt_eps": prob,
        "bayes_decision": "meaningful difference" if prob >= CONFIDENCE else "no meaningful difference shown",
        "bayes_ms": 1000 * (time.perf_counter() - t0),
    }


def compare_demo(project_root="data", epsilon=0.1) -> pd.DataFrame:
    """Pipeline decision vs Bayesian decision on the synthetic demo cases."""
    warnings.filterwarnings("ignore")
    pl = build_pipeline(project_root)
    paths = resolve_project_paths(project_root)
    rows = []
    for sc in build_scenarios(pl.net.predictor.p_matrices, paths.instance_dir, epsilon):
        r = pl.run(sc.request)
        row = {"scenario": sc.name, "n_X": int(sc.request.x.sum()), "n_Y": int(sc.request.y.sum()),
               "pipeline_flag": r.reliability_flag, "pipeline_p": r.p_value, "pipeline_decision": r.decision}
        if r.decision is None:
            row.update({"bayes_decision": None, "agree": None, "note": "rejected before testing"})
        else:
            b = bayesian_result(sc.request)
            row.update(b)
            row["agree"] = b["bayes_decision"] == r.decision
        rows.append(row)
    return pd.DataFrame(rows)


# ---- validation of Pratik's functions -----------------------------------------------------
def validate_functions(P: np.ndarray) -> pd.DataFrame:
    """Do Pratik's functions agree with the pipeline's own maths?"""
    _, common, b06 = pratik_modules()
    from .uncertainty import perturb_rho_offdiagonal_mass
    x, y = np.array([40., 52, 7, 14]), np.array([9., 17, 54, 12])
    a = common.bayesian_d_hat_draws(x, y, P, B, draws=4000, prior_alpha=PRIOR, seed=1)
    b = common.bayesian_d_hat_draws_with_rho(x, y, np.repeat(P[None], 4000, 0), B, prior_alpha=PRIOR, seed=1)
    widths = []
    for k in (1, 10, 100):
        d = common.bayesian_d_hat_draws(x * k, y * k, P, B, draws=4000, prior_alpha=PRIOR, seed=2)
        widths.append(np.quantile(d, 0.975) - np.quantile(d, 0.025))
    rho = b06._sample_rho(np.full((4, 4), 25.0), 2000, PRIOR, 3)
    checks = [
        ("his d_hat_norm == pipeline d_norm (ry9 example)",
         abs(common.d_hat_norm(x, y, P, B) - d_norm(x / x.sum(), y / y.sum(), x.sum(), y.sum(), P)), "< 1e-12"),
        ("Level B with rho fixed == Level A (same seed)", float(np.max(np.abs(a - b))), "== 0"),
        ("credible-interval width, n x1 / x10 / x100", tuple(round(w, 4) for w in widths), "shrinks"),
        ("rho posterior rows sum to 1 (max error)", float(np.max(np.abs(rho.sum(axis=2) - 1))), "< 1e-12"),
        ("perturbation at 0% returns rho unchanged", float(np.max(np.abs(perturb_rho_offdiagonal_mass(P, 0) - P))), "== 0"),
    ]
    return pd.DataFrame(checks, columns=["check", "value", "expected"])


def coverage_check(P, p, q, n, m, reps=200, seed=0) -> float:
    """Share of 95% credible intervals that contain the true d_norm(p, q) when answer counts
    are drawn Multinomial(n, p) and (m, q) - the model Pratik's posterior assumes."""
    _, common, _ = pratik_modules()
    rng = np.random.default_rng(seed)
    true = d_norm(np.asarray(p), np.asarray(q), n, m, P)
    hit = 0
    for i in range(reps):
        x, y = rng.multinomial(n, p), rng.multinomial(m, q)
        d = common.bayesian_d_hat_draws(x, y, P, B, draws=2000, prior_alpha=PRIOR, seed=i)
        lo, hi = np.quantile(d, [0.025, 0.975])
        hit += lo <= true <= hi
    return hit / reps


def boundary_error_rates(pipeline, P, p, q, n, m, epsilon, reps=400, seed=0) -> dict:
    """At a true pair sitting exactly on the boundary d_norm = epsilon, how often does each
    method call a 'meaningful difference'? Data are drawn by RYA's model (fixed composition,
    answers through rho). A level-0.05 test should do so about 5% of the time or less."""
    from .boundary import _shrunk
    from .contracts import ComparisonRequest as R
    from scipy.optimize import brentq
    p, q = np.asarray(p, float), np.asarray(q, float)
    s = brentq(lambda s: d_norm(*_shrunk(p, q, s), n, m, P) - epsilon, 0, 1)
    pb, qb = _shrunk(p, q, s)
    tx, ty = integer_composition(pb, n), integer_composition(qb, m)
    rng = np.random.default_rng(seed)
    Pn = P / P.sum(axis=1, keepdims=True)
    pipe_hits = bayes_hits = 0
    for i in range(reps):
        x = sum(rng.multinomial(int(tx[j]), Pn[j]) for j in range(4) if tx[j])
        y = sum(rng.multinomial(int(ty[j]), Pn[j]) for j in range(4) if ty[j])
        req = R.make("ry9", x, y, P, epsilon=epsilon)
        pipe_hits += pipeline.run(req).decision == "meaningful difference"
        bayes_hits += bayesian_result(req, draws=4000, seed=i)["bayes_decision"] == "meaningful difference"
    return {"n": n, "m": m, "reps": reps, "pipeline_rate": pipe_hits / reps, "bayes_rate": bayes_hits / reps}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", default="data")
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--out", default="bayes_vs_pipeline.csv")
    ap.add_argument("--skip-rates", action="store_true", help="skip the slower boundary error-rate check")
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 55)

    df = compare_demo(a.project_root, a.epsilon)
    show = df[["scenario", "n_X", "n_Y", "pipeline_p", "pipeline_decision", "bayes_P_d_gt_eps", "bayes_decision", "agree"]]
    print("Pipeline vs Bayesian analogue (evaluation only)\n")
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    df.to_csv(a.out, index=False)

    pl = build_pipeline(a.project_root, uncertainty=False)
    P = pl.net.predictor.p_matrices["ry9"]
    print("\nValidation of Pratik's functions\n")
    print(validate_functions(P).to_string(index=False))
    print("\n95% credible-interval coverage of the true d_norm (should be about 0.95):")
    for n, m in ((30, 30), (200, 200), (25, 2000)):
        print(f"  n={n}, m={m}: {coverage_check(P, [.2, .3, .3, .2], [.25, .3, .25, .2], n, m):.2f}")
    if not a.skip_rates:
        print("\nHow often each method calls 'meaningful difference' when the truth is exactly on the")
        print("boundary (data from RYA's model; a 5%-level test should be at or below about 0.05):")
        for n, m in ((100, 100), (25, 2000)):
            print("  ", boundary_error_rates(pl, P, [.05, .2, .35, .4], [.25, .3, .25, .2], n, m, a.epsilon))
    print(f"\nTable: {a.out}")


if __name__ == "__main__":
    main()
