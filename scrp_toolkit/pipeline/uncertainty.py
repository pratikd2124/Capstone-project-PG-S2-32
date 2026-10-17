"""Stage 3+: rho uncertainty carried through to the p-value (Objective 3, Pratik).

rho is estimated from test-retest data but the method treats it as fixed. Here the whole
test (boundary pair, observed d_hat, null distribution, p-value) is re-run for a set of
plausible rho matrices, and the spread of p-values is reported.

Where the rho variants come from:
* ``transition_counts`` given: draws from a Dirichlet posterior on each row of rho
  (Pratik's future-extension design: rho_j | data ~ Dirichlet(prior + counts_j));
* otherwise: Pratik's off-diagonal error-mass perturbation, +/- ``delta`` (default 5%).
"""
from __future__ import annotations

from typing import Callable, List

import numpy as np

from .contracts import ComparisonRequest, Uncertainty
from dataclasses import replace


def perturb_rho_offdiagonal_mass(rho: np.ndarray, relative_change: float) -> np.ndarray:
    """Pratik's perturbation (pipeline_v2/common.py): scale each row's off-diagonal mass."""
    base = np.asarray(rho, dtype=float)
    if relative_change == 0:
        return base.copy()
    out = base.copy()
    for i in range(4):
        off = [j for j in range(4) if j != i]
        out[i, off] = base[i, off] * (1.0 + relative_change)
        out[i, i] = 1.0 - out[i, off].sum()
        if out[i, i] < 0:
            raise ValueError("perturbation makes a diagonal probability negative")
    return out


def rho_posterior_draws(counts: np.ndarray, draws: int, prior: float = 0.5, seed: int = 7) -> np.ndarray:
    """Dirichlet posterior draws of rho from test-retest transition counts, shape (draws, 4, 4)."""
    counts = np.asarray(counts, dtype=float)
    rng = np.random.default_rng(seed)
    return np.stack([np.stack([rng.dirichlet(counts[j] + prior) for j in range(4)]) for _ in range(draws)])


def rho_variants(req: ComparisonRequest, delta: float = 0.05, n_posterior: int = 20) -> tuple[List[np.ndarray], str]:
    if req.rho_transition_counts is not None:
        return list(rho_posterior_draws(np.asarray(req.rho_transition_counts), n_posterior)), \
            f"Dirichlet posterior from test-retest counts ({n_posterior} draws)"
    out = []
    for c in (-delta, -delta / 2, delta / 2, delta):
        try:
            out.append(perturb_rho_offdiagonal_mass(req.P, c))
        except ValueError:
            pass
    return out, f"rho error mass +/-{delta:.0%}"


def propagate(req: ComparisonRequest, base_p: float, run_one: Callable[[ComparisonRequest], tuple],
              delta: float = 0.05, n_posterior: int = 20) -> Uncertainty:
    """Re-run the test for each rho variant. ``run_one(req) -> (p_value, shape, rate)``."""
    variants, source = rho_variants(req, delta, n_posterior)
    ps, shapes, rates = [base_p], [], []
    for P in variants:
        p, sh, rt = run_one(replace(req, rho=tuple(map(tuple, P))))
        if np.isfinite(p):
            ps.append(p)
        if sh is not None and np.isfinite(sh):
            shapes.append(sh)
            rates.append(rt)
    lo, hi = float(np.min(ps)), float(np.max(ps))
    stable = (hi < req.alpha) or (lo >= req.alpha)
    rng = lambda v: (float(np.min(v)), float(np.max(v))) if v else (float("nan"), float("nan"))
    return Uncertainty(lo, hi, rng(shapes), rng(rates), bool(stable), source, len(variants))
