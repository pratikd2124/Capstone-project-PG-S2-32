"""Where the null distribution is evaluated, and the minimal-effects test itself.

PROVISIONAL until RYA sends the portal JavaScript. Two modes, selectable in config:

``boundary``    (default) the draft paper's construction. H0: d_norm < epsilon. Shrink both
                groups' proportions toward their midpoint, keeping n and m, until
                d_norm = epsilon. The null distribution of d_hat is taken at that pair, and
                p = P(d_hat >= observed d_hat | boundary pair).
``point_null``  Murat's 12 Oct stand-in (epsilon = 0): both groups get the pooled midpoint
                shares; p = P(d_hat >= observed | no difference).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .contracts import BoundaryPair, ComparisonRequest, NullDistribution, TestResult
from .simulate import d_norm

MODES = ("boundary", "point_null")


def _shrunk(p, q, s):
    mid = 0.5 * (p + q)
    return mid + s * (p - mid), mid + s * (q - mid)


def find_boundary(req: ComparisonRequest, mode: str = "boundary") -> BoundaryPair:
    """The (x~, y~) pair the null distribution is evaluated at. n and m never change."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    x, y, P = req.x, req.y, req.P
    n, m = x.sum(), y.sum()
    p, q = x / n, y / m
    d_obs = d_norm(p, q, n, m, P)

    if mode == "point_null":
        mid = 0.5 * (p + q)
        return BoundaryPair(tuple(mid * n), tuple(mid * m), 0.0, d_obs, 0.0, mode,
                            "epsilon treated as 0; both groups at pooled shares")

    eps = req.epsilon

    def f(s):
        a, b = _shrunk(p, q, s)
        return d_norm(a, b, n, m, P) - eps

    # Largest s that keeps both groups' shares non-negative (lets us expand when d_obs < eps).
    mid = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        lim = [(-mid[i]) / (v[i] - mid[i]) for v in (p, q) for i in range(4) if v[i] - mid[i] < 0]
    s_max = float(min(lim)) if lim else 1.0
    s_max = max(1.0, s_max)

    if not np.isfinite(d_obs):
        return BoundaryPair(tuple(x), tuple(y), 1.0, d_obs, float("nan"), mode, "observed d is not finite")
    if d_obs >= eps:
        s = brentq(f, 0.0, 1.0, xtol=1e-10)
        note = "shrunk toward midpoint"
    elif f(s_max) >= 0:
        s = brentq(f, 1.0, s_max, xtol=1e-10)
        note = "observed d below epsilon; expanded to reach the boundary"
    else:
        a, b = _shrunk(p, q, s_max)
        return BoundaryPair(tuple(a * n), tuple(b * m), s_max, d_obs, d_norm(a, b, n, m, P), mode,
                            "epsilon not reachable inside the simplex")
    a, b = _shrunk(p, q, s)
    return BoundaryPair(tuple(a * n), tuple(b * m), float(s), d_obs, d_norm(a, b, n, m, P), mode, note)


def minimal_effects_test(req: ComparisonRequest, bp: BoundaryPair, null: NullDistribution) -> TestResult:
    """p = P(d_hat >= observed d_hat) under the null at the boundary pair."""
    p_value = null.sf(bp.d_norm_observed) if np.isfinite(bp.d_norm_observed) else float("nan")
    eps = 0.0 if bp.mode == "point_null" else req.epsilon
    if not np.isfinite(p_value):
        decision = None
    elif p_value < req.alpha:
        decision = "meaningful difference"
    else:
        decision = "no meaningful difference shown"
    return TestResult(bp.d_norm_observed, eps, float(p_value), decision, bp.mode)
