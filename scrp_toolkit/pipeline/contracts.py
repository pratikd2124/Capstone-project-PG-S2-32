"""The small records that the pipeline stages pass to each other.

Each stage takes one of these in and hands one back, so a stage can be tested or swapped
on its own. Only the last step (run.py) decides the colour flag and the p-value range.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats

# Where screening sends a request.
PASS, FLAGGED, REJECT = "PASS", "FLAGGED", "REJECT"
# The flag the user sees.
GREEN, AMBER, RED = "GREEN", "AMBER", "RED"

NAN = float("nan")


def _tuple(values):
    return tuple(float(v) for v in values)


@dataclass(frozen=True)
class ComparisonRequest:
    """One question: do these two groups answer this item differently?

    x_counts, y_counts: how many students in each group chose answers 1-4.
    rho: the item's 4x4 unreliability matrix (row = the student's "real" answer,
         column = the answer they actually gave).
    epsilon: the smallest difference that counts as meaningful.
    rho_transition_counts: optional test-retest counts, used to show how unsure rho is.
    """
    item: str
    x_counts: tuple
    y_counts: tuple
    rho: tuple
    epsilon: float = 0.1
    alpha: float = 0.05
    rho_source: str = "json"
    label: str = ""
    rho_transition_counts: Optional[tuple] = None

    @staticmethod
    def make(item, x_counts, y_counts, rho, rho_transition_counts=None, **options):
        """Build a request from plain lists or arrays."""
        counts = None if rho_transition_counts is None else tuple(map(_tuple, np.asarray(rho_transition_counts)))
        return ComparisonRequest(str(item), _tuple(x_counts), _tuple(y_counts),
                                 tuple(map(_tuple, np.asarray(rho, dtype=float))),
                                 rho_transition_counts=counts, **options)

    @property
    def x(self):
        return np.array(self.x_counts)

    @property
    def y(self):
        return np.array(self.y_counts)

    @property
    def P(self):
        return np.array(self.rho)


@dataclass(frozen=True)
class RhoCheck:
    """Is rho usable? status is 'pass', 'warn' or 'reject'."""
    status: str
    reasons: tuple = ()
    max_row_sum_error: float = NAN
    certainty: float = NAN
    asymmetry: float = NAN


@dataclass(frozen=True)
class BoundaryPair:
    """The two groups moved toward each other until their distance is exactly epsilon.

    The test asks how unusual the observed distance would be if the true distance were
    only epsilon, so this is where the null distribution is worked out. Group sizes do
    not change. shrink = 1 means no move.
    """
    x_counts: tuple
    y_counts: tuple
    shrink: float
    d_norm_observed: float
    d_norm_boundary: float
    mode: str
    note: str = ""

    @property
    def x(self):
        return np.array(self.x_counts)

    @property
    def y(self):
        return np.array(self.y_counts)


@dataclass(frozen=True)
class ScreeningResult:
    """Where the request goes (PASS, FLAGGED or REJECT) and why."""
    route: str
    reasons: tuple
    p_incompatible_observed: float = NAN
    p_incompatible_boundary: float = NAN
    threshold: float = NAN
    outside_training_range: bool = False
    detail: dict = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class NullDistribution:
    """How d_hat would vary by chance at the boundary pair.

    family 'gamma' comes from the network (or a Gamma fitted to simulations), 'empirical'
    is the simulated draws themselves, 'weibull' a Weibull fitted to them.
    """
    family: str
    method: str
    params: dict
    q95: float
    time_ms: float
    draws: Optional[np.ndarray] = field(default=None, compare=False, repr=False)

    def sf(self, t):
        """Chance of a d_hat at least as large as t."""
        if self.family == "gamma":
            return float(stats.gamma.sf(t, self.params["shape"], scale=1 / self.params["rate"]))
        if self.family == "weibull":
            return float(stats.weibull_min.sf(t, self.params["k"], scale=self.params["lam"]))
        if self.family == "empirical":
            return float((np.sum(self.draws >= t) + 1) / (self.draws.size + 1))
        raise ValueError(f"unknown family {self.family}")


@dataclass(frozen=True)
class TestResult:
    """The minimal-effects test."""
    d_hat_norm: float
    epsilon: float
    p_value: float
    decision: Optional[str]
    mode: str


@dataclass(frozen=True)
class Uncertainty:
    """How far the p-value moves when rho is not taken as exact."""
    p_low: float
    p_high: float
    shape_range: tuple = (NAN, NAN)
    rate_range: tuple = (NAN, NAN)
    decision_stable: bool = True
    source: str = ""
    n_variants: int = 0


@dataclass
class PipelineResult:
    """Everything the pipeline worked out for one request."""
    request: ComparisonRequest
    decision: Optional[str]
    p_value: Optional[float]
    reliability_flag: str
    uncertainty_range: Optional[tuple]
    flag_reasons: list
    rho_check: RhoCheck
    screening: Optional[ScreeningResult] = None
    boundary: Optional[BoundaryPair] = None
    null: Optional[NullDistribution] = None
    test: Optional[TestResult] = None
    uncertainty: Optional[Uncertainty] = None
    provenance: dict = field(default_factory=dict)

    def summary(self):
        """One flat row for tables."""
        r, u = self.request, self.uncertainty

        def rounded(pair, digits):
            ok = pair is not None and np.isfinite(pair[0])
            return tuple(round(v, digits) for v in pair) if ok else None

        if self.screening:
            route = self.screening.route
        else:
            route = REJECT if self.rho_check.status == "reject" else "-"
        return {
            "label": r.label or r.item,
            "item": r.item,
            "n_X": int(r.x.sum()),
            "n_Y": int(r.y.sum()),
            "route": route,
            "null_method": self.null.method if self.null else None,
            "d_hat_norm": round(self.test.d_hat_norm, 4) if self.test else None,
            "epsilon": r.epsilon,
            "p_value": None if self.p_value is None else round(self.p_value, 4),
            "p_range": rounded(self.uncertainty_range, 4),
            "shape_range": rounded(u.shape_range, 3) if u else None,
            "rate_range": rounded(u.rate_range, 3) if u else None,
            "decision": self.decision,
            "flag": self.reliability_flag,
            "flag_reasons": "; ".join(self.flag_reasons),
            "time_ms": round(self.provenance.get("total_ms", NAN), 1),
        }
