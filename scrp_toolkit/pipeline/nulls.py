"""Stages 2a and 2b: the null distribution of d_hat at the boundary pair.

2a  EXP002 network (Objective 4, Murat): Gamma(shape, rate) from ``predict_batch``,
    fp32 (RYA's model exactly) or fp16 (half the file, for the browser). Rate used as-is.
2b  Fallback (Objective 2, Shuyun): simulate d_hat with RYA's latent-type model at the
    boundary pair, then estimate the tail. Candidates:
      ``mc_empirical``  Shuyun's Monte Carlo empirical quantile / tail (backup_methods.py)
      ``weibull``       Weibull fitted to the same draws (the original authors found it as
                        good as Gamma by BIC)
      ``gamma_mle``     Gamma fitted to the same draws (what the network is trained to copy)
"""
from __future__ import annotations

import time

import numpy as np
from scipy import stats

from .contracts import BoundaryPair, ComparisonRequest, NullDistribution
from .o2.backup_methods import empirical_estimate, gamma_estimate
from .simulate import integer_composition, simulate_d_hat

FALLBACK_METHODS = ("mc_empirical", "weibull", "gamma_mle")


class NetworkNull:
    """EXP002 behind one call. Holds one predictor per weight option."""

    def __init__(self, predictor):
        self.predictor = predictor

    def gamma_params(self, x: np.ndarray, y: np.ndarray, P: np.ndarray, item: str):
        key = f"__req__{item}"
        self.predictor.p_matrices[key] = np.asarray(P, dtype=float)
        shape, rate = self.predictor.predict_batch(np.atleast_2d(x), np.atleast_2d(y), [key])
        return float(shape[0]), float(rate[0])

    def __call__(self, req: ComparisonRequest, bp: BoundaryPair, P: np.ndarray | None = None) -> NullDistribution:
        t0 = time.perf_counter()
        P = req.P if P is None else P
        shape, rate = self.gamma_params(bp.x, bp.y, P, req.item)
        ms = 1000 * (time.perf_counter() - t0)
        method = f"nn_{self.predictor.weights}"
        if not (np.isfinite(shape) and np.isfinite(rate) and shape > 0 and rate > 0):
            return NullDistribution("gamma", method, {"shape": shape, "rate": rate}, float("nan"), ms)
        q95 = float(stats.gamma.ppf(0.95, shape, scale=1 / rate))
        return NullDistribution("gamma", method, {"shape": shape, "rate": rate}, q95, ms)


def simulated_null(req: ComparisonRequest, bp: BoundaryPair, method: str = "mc_empirical",
                   draws: int = 4000, seed: int = 2026, P: np.ndarray | None = None) -> NullDistribution:
    """Fallback null: RYA's simulator at the boundary pair, then the chosen tail estimator."""
    if method not in FALLBACK_METHODS:
        raise ValueError(f"method must be one of {FALLBACK_METHODS}")
    t0 = time.perf_counter()
    P = req.P if P is None else P
    n, m = int(round(req.x.sum())), int(round(req.y.sum()))
    tx = integer_composition(bp.x / bp.x.sum(), n)
    ty = integer_composition(bp.y / bp.y.sum(), m)
    d = simulate_d_hat(tx, ty, P, trials=draws, rng=np.random.default_rng(seed))
    if method == "mc_empirical":
        est = empirical_estimate(d)
        params = {"n_draws": float(d.size)}
        family, q95 = "empirical", est["q95_estimate"]
    elif method == "gamma_mle":
        est = gamma_estimate(d[d > 0])
        params = {"shape": est["shape"], "rate": est["rate"]}
        family, q95 = "gamma", est["q95_estimate"]
    else:
        k, _, lam = stats.weibull_min.fit(d[d > 0], floc=0)
        params = {"k": float(k), "lam": float(lam)}
        family, q95 = "weibull", float(stats.weibull_min.ppf(0.95, k, scale=lam))
    ms = 1000 * (time.perf_counter() - t0)
    return NullDistribution(family, method, params, float(q95), ms, draws=d)
