"""The boundary pair and the minimal-effects test (RYA's method).

The test asks: is the gap between the two groups bigger than epsilon? So the question
becomes how surprising the observed gap would be if the true gap were exactly epsilon.
To get there, both groups' answer shares are slid toward their midpoint (group sizes stay
the same) until their distance is exactly epsilon. That is the boundary pair.

This follows the draft paper and is provisional until RYA sends the portal code.
mode="point_null" is the simpler stand-in Murat used: epsilon = 0, both groups at the
midpoint.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .contracts import BoundaryPair, TestResult
from .simulate import d_norm

MODES = ("boundary", "point_null")
MEANINGFUL = "meaningful difference"
NOT_SHOWN = "no meaningful difference shown"


def _shrunk(p, q, s):
    """Move shares p and q toward their midpoint. s=1 leaves them, s=0 merges them."""
    mid = (p + q) / 2
    return mid + s * (p - mid), mid + s * (q - mid)


def _largest_stretch(p, q):
    """How far apart p and q can be pushed before a share goes negative (at least 1)."""
    mid = (p + q) / 2
    limits = [-mid[i] / (v[i] - mid[i]) for v in (p, q) for i in range(4) if v[i] < mid[i]]
    return max(1.0, min(limits)) if limits else 1.0


def find_boundary(req, mode="boundary"):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    n, m = req.x.sum(), req.y.sum()
    p, q, P = req.x / n, req.y / m, req.P
    d_observed = d_norm(p, q, n, m, P)

    def pair(s, note):
        a, b = _shrunk(p, q, s)
        return BoundaryPair(tuple(a * n), tuple(b * m), float(s), d_observed, d_norm(a, b, n, m, P), mode, note)

    if mode == "point_null":
        mid = (p + q) / 2
        return BoundaryPair(tuple(mid * n), tuple(mid * m), 0.0, d_observed, 0.0, mode,
                            "epsilon treated as 0; both groups at pooled shares")
    if not np.isfinite(d_observed):
        return BoundaryPair(tuple(req.x), tuple(req.y), 1.0, d_observed, float("nan"), mode,
                            "observed d is not finite")

    def gap_minus_epsilon(s):
        a, b = _shrunk(p, q, s)
        return d_norm(a, b, n, m, P) - req.epsilon

    stretch = _largest_stretch(p, q)
    if d_observed >= req.epsilon:
        return pair(brentq(gap_minus_epsilon, 0.0, 1.0, xtol=1e-10), "shrunk toward midpoint")
    if gap_minus_epsilon(stretch) >= 0:
        return pair(brentq(gap_minus_epsilon, 1.0, stretch, xtol=1e-10),
                    "observed d below epsilon; expanded to reach the boundary")
    return pair(stretch, "epsilon not reachable inside the simplex")


def minimal_effects_test(req, boundary, null):
    """p = chance of a d_hat at least as big as the observed one, at the boundary pair."""
    d = boundary.d_norm_observed
    p_value = null.sf(d) if np.isfinite(d) else float("nan")
    if not np.isfinite(p_value):
        decision = None
    else:
        decision = MEANINGFUL if p_value < req.alpha else NOT_SHOWN
    epsilon = 0.0 if boundary.mode == "point_null" else req.epsilon
    return TestResult(d, epsilon, float(p_value), decision, boundary.mode)
