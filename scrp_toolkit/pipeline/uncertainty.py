"""Step 3+: how much the answer depends on rho being exactly right (Objective 3, Pratik).

rho is estimated from a test-retest study, but the method treats it as exact. Here the
whole test is re-run with a few other plausible rho matrices, and the spread of p-values
is reported. Where the other matrices come from:

* if the request carries real test-retest counts: random draws from a Dirichlet posterior
  on each row of rho;
* otherwise: rho with its off-diagonal "error" mass scaled up and down by 2.5% and 5%.

Both use Pratik's own functions from o3/, unchanged.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .contracts import Uncertainty
from .o3.loader import pratik_modules


def perturb_rho_offdiagonal_mass(rho, relative_change):
    """Pratik's perturbation (o3/common.py)."""
    return pratik_modules()[1].perturb_rho_offdiagonal_mass(rho, relative_change)


def rho_posterior_draws(counts, draws, prior=0.5, seed=7):
    """Pratik's rho posterior (o3/06_bayesian_evaluation.py), shape (draws, 4, 4)."""
    return pratik_modules()[2]._sample_rho(np.asarray(counts, dtype=float), draws, prior, seed)


def rho_variants(req, delta=0.05, n_posterior=20):
    """The alternative rho matrices to try, and a note of where they came from."""
    if req.rho_transition_counts is not None:
        draws = rho_posterior_draws(req.rho_transition_counts, n_posterior)
        return list(draws), f"Dirichlet posterior from test-retest counts ({n_posterior} draws)"
    variants = []
    for change in (-delta, -delta / 2, delta / 2, delta):
        try:
            variants.append(perturb_rho_offdiagonal_mass(req.P, change))
        except ValueError:          # the change would push a diagonal entry below zero
            pass
    return variants, f"rho error mass +/-{delta:.0%}"


def propagate(req, base_p, run_one, delta=0.05, n_posterior=20):
    """Re-run the test for every rho variant. run_one(request) returns (p, shape, rate)."""
    variants, source = rho_variants(req, delta, n_posterior)
    p_values, shapes, rates = [base_p], [], []
    for P in variants:
        p, shape, rate = run_one(replace(req, rho=tuple(map(tuple, P))))
        if np.isfinite(p):
            p_values.append(p)
        if shape is not None and np.isfinite(shape):
            shapes.append(shape)
            rates.append(rate)

    def spread(values):
        return (float(min(values)), float(max(values))) if values else (float("nan"), float("nan"))

    low, high = spread(p_values)
    stable = high < req.alpha or low >= req.alpha
    return Uncertainty(low, high, spread(shapes), spread(rates), stable, source, len(variants))
