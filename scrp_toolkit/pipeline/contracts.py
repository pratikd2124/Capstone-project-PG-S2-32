"""The data passed between pipeline stages.

Every stage receives and returns one of these frozen dataclasses, so each stage can be
built, tested and swapped on its own. Only ``run.py`` sets the reliability flag and the
uncertainty range on the final ``PipelineResult``; earlier stages only add evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Route names (shared by screening and run.py).
PASS = "PASS"
FLAGGED = "FLAGGED"
REJECT = "REJECT"

# Reliability flags (set only in run.py).
GREEN = "GREEN"
AMBER = "AMBER"
RED = "RED"


@dataclass(frozen=True)
class ComparisonRequest:
    """One comparison: two groups' answer counts on one four-option item, plus its rho.

    x_counts, y_counts  answer counts 1..4 for group X (e.g. a classroom) and group Y
    rho                 4x4 unreliability matrix (row = latent type, column = answer given)
    epsilon             minimal-effects threshold: H0 is d_norm < epsilon
    """
    item: str
    x_counts: Tuple[float, ...]
    y_counts: Tuple[float, ...]
    rho: Tuple[Tuple[float, ...], ...]
    epsilon: float = 0.1
    alpha: float = 0.05
    rho_source: str = "json"
    label: str = ""
    # Optional genuine test-retest transition counts (4x4) for rho uncertainty (Objective 3).
    rho_transition_counts: Optional[Tuple[Tuple[float, ...], ...]] = None

    @staticmethod
    def make(item, x_counts, y_counts, rho, **kw) -> "ComparisonRequest":
        rc = kw.pop("rho_transition_counts", None)
        return ComparisonRequest(
            item=str(item),
            x_counts=tuple(float(v) for v in x_counts),
            y_counts=tuple(float(v) for v in y_counts),
            rho=tuple(tuple(float(v) for v in row) for row in np.asarray(rho, dtype=float)),
            rho_transition_counts=None if rc is None else tuple(tuple(float(v) for v in r) for r in np.asarray(rc)),
            **kw,
        )

    @property
    def x(self) -> np.ndarray:
        return np.asarray(self.x_counts, dtype=float)

    @property
    def y(self) -> np.ndarray:
        return np.asarray(self.y_counts, dtype=float)

    @property
    def P(self) -> np.ndarray:
        return np.asarray(self.rho, dtype=float)


@dataclass(frozen=True)
class RhoCheck:
    """Stage 1a: is rho usable? status is 'pass', 'warn' or 'reject'."""
    status: str
    reasons: Tuple[str, ...] = ()
    max_row_sum_error: float = float("nan")
    certainty: float = float("nan")
    asymmetry: float = float("nan")


@dataclass(frozen=True)
class BoundaryPair:
    """The pair the minimal-effects test evaluates the null distribution at.

    Group proportions are shrunk toward their midpoint until d_norm = epsilon; group sizes
    n and m are unchanged. ``shrink`` = 1 means the observed pair itself.
    """
    x_counts: Tuple[float, ...]
    y_counts: Tuple[float, ...]
    shrink: float
    d_norm_observed: float
    d_norm_boundary: float
    mode: str
    note: str = ""

    @property
    def x(self) -> np.ndarray:
        return np.asarray(self.x_counts, dtype=float)

    @property
    def y(self) -> np.ndarray:
        return np.asarray(self.y_counts, dtype=float)


@dataclass(frozen=True)
class ScreeningResult:
    """Stage 1b: route plus the evidence behind it."""
    route: str                      # PASS / FLAGGED / REJECT
    reasons: Tuple[str, ...]
    p_incompatible_observed: float = float("nan")
    p_incompatible_boundary: float = float("nan")
    threshold: float = float("nan")
    outside_training_range: bool = False
    detail: Dict[str, Any] = field(default_factory=dict, compare=False, hash=False)


@dataclass(frozen=True)
class NullDistribution:
    """Stage 2a/2b: the sampling distribution of the scaled d_hat at the boundary pair."""
    family: str                     # 'gamma' (network) / 'empirical' / 'weibull' (fallback)
    method: str                     # e.g. 'nn_fp32', 'nn_fp16', 'mc_empirical'
    params: Dict[str, float]
    q95: float
    time_ms: float
    draws: Optional[np.ndarray] = field(default=None, compare=False, hash=False, repr=False)

    def sf(self, t: float) -> float:
        """P(T >= t) under this distribution."""
        from scipy import stats
        if self.family == "gamma":
            return float(stats.gamma.sf(t, self.params["shape"], scale=1.0 / self.params["rate"]))
        if self.family == "weibull":
            return float(stats.weibull_min.sf(t, self.params["k"], scale=self.params["lam"]))
        if self.family == "empirical":
            d = self.draws
            return float((np.sum(d >= t) + 1) / (d.size + 1))
        raise ValueError(f"unknown family {self.family}")


@dataclass(frozen=True)
class TestResult:
    """Stage 3: the minimal-effects test."""
    d_hat_norm: float
    epsilon: float
    p_value: float
    decision: str                   # 'meaningful difference' / 'no meaningful difference shown'
    mode: str


@dataclass(frozen=True)
class Uncertainty:
    """Stage 3+: how much the answer moves when rho is uncertain."""
    p_low: float
    p_high: float
    shape_range: Tuple[float, float] = (float("nan"), float("nan"))
    rate_range: Tuple[float, float] = (float("nan"), float("nan"))
    decision_stable: bool = True
    source: str = ""
    n_variants: int = 0


@dataclass
class PipelineResult:
    """What the user sees."""
    request: ComparisonRequest
    decision: Optional[str]
    p_value: Optional[float]
    reliability_flag: str
    uncertainty_range: Optional[Tuple[float, float]]
    flag_reasons: List[str]
    rho_check: RhoCheck
    screening: Optional[ScreeningResult] = None
    boundary: Optional[BoundaryPair] = None
    null: Optional[NullDistribution] = None
    test: Optional[TestResult] = None
    uncertainty: Optional[Uncertainty] = None
    provenance: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        """One flat row, for tables and the demo."""
        r = self.request
        return {
            "label": r.label or r.item,
            "item": r.item,
            "n_X": int(r.x.sum()),
            "n_Y": int(r.y.sum()),
            "route": self.screening.route if self.screening else ("REJECT" if self.rho_check.status == "reject" else "-"),
            "null_method": self.null.method if self.null else None,
            "d_hat_norm": None if self.test is None else round(self.test.d_hat_norm, 4),
            "epsilon": r.epsilon,
            "p_value": None if self.p_value is None else round(self.p_value, 4),
            "p_range": None if self.uncertainty_range is None else tuple(round(v, 4) for v in self.uncertainty_range),
            "shape_range": None if self.uncertainty is None or not np.isfinite(self.uncertainty.shape_range[0])
            else tuple(round(v, 3) for v in self.uncertainty.shape_range),
            "rate_range": None if self.uncertainty is None or not np.isfinite(self.uncertainty.rate_range[0])
            else tuple(round(v, 3) for v in self.uncertainty.rate_range),
            "decision": self.decision,
            "flag": self.reliability_flag,
            "flag_reasons": "; ".join(self.flag_reasons),
            "time_ms": round(self.provenance.get("total_ms", float("nan")), 1),
        }
