"""Steps 2a and 2b: how d_hat would vary by chance at the boundary pair.

2a, the network (Objective 4, Murat): EXP002 predicts a Gamma(shape, rate). Uses Murat's
    batched inference; weights can be the original fp32 or the half-size fp16 version.
2b, the fallback (Objective 2, Shuyun): run RYA's own simulator at the boundary pair and
    read the answer straight off the simulated values (Shuyun's empirical estimator,
    o2_fallback/backup_methods.py, unchanged). Two other options are kept for comparison: a Weibull
    or a Gamma fitted to the same simulated values.
"""
from __future__ import annotations

import time

import numpy as np
from scipy import stats

from .contracts import NullDistribution
from o2_fallback.backup_methods import empirical_estimate, gamma_estimate
from .simulate import integer_composition, simulate_d_hat

FALLBACK_METHODS = ("mc_empirical", "weibull", "gamma_mle")


def _ms_since(start):
    return 1000 * (time.perf_counter() - start)


class NetworkNull:
    """EXP002 as a null distribution."""

    def __init__(self, predictor):
        self.predictor = predictor

    def gamma_params(self, x, y, P, item):
        # The predictor looks rho up by item name, so register this request's rho under its own name.
        key = f"__req__{item}"
        self.predictor.p_matrices[key] = np.asarray(P, dtype=float)
        shape, rate = self.predictor.predict_batch(np.atleast_2d(x), np.atleast_2d(y), [key])
        return float(shape[0]), float(rate[0])

    def __call__(self, req, boundary, P=None):
        start = time.perf_counter()
        shape, rate = self.gamma_params(boundary.x, boundary.y, req.P if P is None else P, req.item)
        usable = np.isfinite(shape) and np.isfinite(rate) and shape > 0 and rate > 0
        q95 = float(stats.gamma.ppf(0.95, shape, scale=1 / rate)) if usable else float("nan")
        return NullDistribution("gamma", f"nn_{self.predictor.weights}", {"shape": shape, "rate": rate},
                                q95, _ms_since(start))


def simulated_null(req, boundary, method="mc_empirical", draws=4000, seed=2026, P=None):
    """Simulate d_hat at the boundary pair with RYA's model, then summarise the draws."""
    if method not in FALLBACK_METHODS:
        raise ValueError(f"method must be one of {FALLBACK_METHODS}")
    start = time.perf_counter()
    P = req.P if P is None else P

    # The simulator needs whole students, so round the boundary shares to group sizes.
    n, m = int(round(req.x.sum())), int(round(req.y.sum()))
    types_x = integer_composition(boundary.x / boundary.x.sum(), n)
    types_y = integer_composition(boundary.y / boundary.y.sum(), m)
    d = simulate_d_hat(types_x, types_y, P, trials=draws, rng=np.random.default_rng(seed))

    if method == "mc_empirical":
        family, params, q95 = "empirical", {"n_draws": float(d.size)}, empirical_estimate(d)["q95_estimate"]
    elif method == "gamma_mle":
        fit = gamma_estimate(d[d > 0])
        family, params, q95 = "gamma", {"shape": fit["shape"], "rate": fit["rate"]}, fit["q95_estimate"]
    else:
        k, _, lam = stats.weibull_min.fit(d[d > 0], floc=0)
        family, params, q95 = "weibull", {"k": float(k), "lam": float(lam)}, stats.weibull_min.ppf(0.95, k, scale=lam)
    return NullDistribution(family, method, params, float(q95), _ms_since(start), draws=d)
