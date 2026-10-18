"""October 9 backup estimators for a nonnegative simulated test statistic.

These functions estimate distribution quantities from *independent* simulation
samples. They do not validate the underlying simulation model.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import gamma


def _clean(draws, *, minimum=100):
    x = np.asarray(draws, dtype=float).reshape(-1)
    if x.size < minimum:
        raise ValueError(f"Need at least {minimum} simulated draws; got {x.size}")
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("Draws must be finite and nonnegative")
    return x


def empirical_estimate(draws, *, quantile=0.95, threshold=None):
    """Monte Carlo empirical quantile and, optionally, P(T > threshold)."""
    x = _clean(draws)
    if not 0 < quantile < 1:
        raise ValueError("quantile must lie between 0 and 1")
    return {
        "q95_estimate": float(np.quantile(x, quantile)),
        "tail_estimate": None if threshold is None else float(np.mean(x > threshold)),
        "n_draws": int(x.size),
    }


def gamma_estimate(draws, *, quantile=0.95, threshold=None):
    """Fit ordinary continuous Gamma by MLE; exact zero mass is not allowed."""
    x = _clean(draws)
    if np.any(x == 0):
        raise ValueError("Gamma MLE not used when draws include exact zeros")
    shape, _, scale = gamma.fit(x, floc=0)
    if not np.isfinite([shape, scale]).all() or shape <= 0 or scale <= 0:
        raise ValueError("Invalid Gamma fit")
    return {
        "q95_estimate": float(gamma.ppf(quantile, shape, loc=0, scale=scale)),
        "tail_estimate": None if threshold is None else float(gamma.sf(threshold, shape, loc=0, scale=scale)),
        "shape": float(shape),
        "rate": float(1.0 / scale),
        "n_draws": int(x.size),
    }


def independent_reference(draws, *, quantile=0.95):
    """Reference q95 and its corresponding upper-tail rate from separate draws."""
    x = _clean(draws)
    q = float(np.quantile(x, quantile))
    return {"q95": q, "tail_at_q95": float(np.mean(x > q)), "n_reference": int(x.size)}


def compare_estimates(gamma_draws, mc_draws, reference_draws, *, quantile=0.95):
    """Compare Gamma and empirical MC on independent estimation/reference draws.

    Both methods estimate P(T > q_reference), where q_reference is computed
    from a separate reference simulation sample. This is a *diagnostic* target.
    """
    reference = independent_reference(reference_draws, quantile=quantile)
    threshold = reference["q95"]
    methods = {
        "gamma": lambda: gamma_estimate(gamma_draws, quantile=quantile, threshold=threshold),
        "monte_carlo_empirical": lambda: empirical_estimate(mc_draws, quantile=quantile, threshold=threshold),
    }
    rows = []
    for method, estimator in methods.items():
        row = {"method": method, "reference_q95": threshold,
               "reference_tail_at_q95": reference["tail_at_q95"],
               "n_reference": reference["n_reference"], "status": "ok"}
        try:
            estimate = estimator()
            row.update(estimate)
            row["q95_relative_error"] = abs(estimate["q95_estimate"] - threshold) / max(abs(threshold), 1e-12)
            row["tail_absolute_error"] = abs(estimate["tail_estimate"] - reference["tail_at_q95"])
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            row["status"] = f"needs_review: {exc}"
            row.update({"q95_estimate": np.nan, "tail_estimate": np.nan,
                        "q95_relative_error": np.nan, "tail_absolute_error": np.nan})
        rows.append(row)
    return rows
