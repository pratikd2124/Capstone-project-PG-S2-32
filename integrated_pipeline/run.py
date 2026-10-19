"""The pipeline: one call takes a comparison to a decision, a p-value and a colour flag.

    check rho -> boundary pair -> screening
        PASS     -> network null (EXP002)
        FLAGGED  -> simulated null (fallback)
    -> minimal-effects test -> p-value range from rho -> result

GREEN means screening passed and the decision holds across the rho range. AMBER means the
fallback was used or the decision flips inside the range. RED means the request was
rejected and there is no p-value.

mode="direct" is the original set-up, kept for comparison: straight to the network, no
checks, no fallback, no range.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .boundary import find_boundary, minimal_effects_test
from .contracts import AMBER, FLAGGED, GREEN, RED, REJECT, PipelineResult, RhoCheck
from .nulls import NetworkNull, simulated_null
from .screen import load_gate, screen, validate_rho
from .uncertainty import propagate


@dataclass
class PipelineConfig:
    mode: str = "robust"                   # or "direct" (the original path)
    test_mode: str = "boundary"            # or "point_null" (epsilon = 0 stand-in)
    weights: str = "fp32"                  # or "fp16" (the half-size browser model)
    fallback_method: str = "mc_empirical"  # or "weibull", "gamma_mle"
    fallback_draws: int = 4000
    uncertainty: bool = True
    rho_delta: float = 0.05
    uncertainty_draws: int = 1500          # simulator draws per rho variant on the fallback route
    screen_threshold: Optional[float] = None   # None keeps SIN-WEI's 0.10
    small_group_n: Optional[int] = 30          # None turns the small-group rule off
    small_group_ratio: float = 20.0


class Pipeline:
    def __init__(self, predictor, config=None, gate=None):
        self.cfg = config or PipelineConfig()
        self.net = NetworkNull(predictor)
        self.gate = gate or load_gate(threshold=self.cfg.screen_threshold)

    def run(self, req):
        start = time.perf_counter()
        if self.cfg.mode == "direct":
            return self._run_direct(req, start)

        def finish(**fields):
            fields.setdefault("provenance", {})["total_ms"] = 1000 * (time.perf_counter() - start)
            return PipelineResult(request=req, **fields)

        # 1a. Is rho usable?
        rho_check = validate_rho(req)
        if rho_check.status == "reject":
            return finish(decision=None, p_value=None, reliability_flag=RED, uncertainty_range=None,
                          flag_reasons=list(rho_check.reasons), rho_check=rho_check,
                          screening=screen(req, rho_check, None, self.gate))

        # Boundary pair, then 1b. can the network be trusted here?
        boundary = find_boundary(req, self.cfg.test_mode)
        screening = screen(req, rho_check, boundary, self.gate, self.cfg.small_group_n, self.cfg.small_group_ratio)
        if screening.route == REJECT:
            return finish(decision=None, p_value=None, reliability_flag=RED, uncertainty_range=None,
                          flag_reasons=list(screening.reasons), rho_check=rho_check,
                          screening=screening, boundary=boundary)

        # 2. Null distribution: network if screening passed, otherwise the fallback.
        reasons = []
        use_fallback = screening.route == FLAGGED
        null = self._null(req, boundary, use_fallback)
        if not use_fallback and not np.isfinite(null.q95):
            reasons.append("network returned no usable Gamma; fallback used")
            use_fallback = True
            null = self._null(req, boundary, True)

        # 3. The test, then 3+. how much it moves with rho.
        test = minimal_effects_test(req, boundary, null)
        spread = None
        if self.cfg.uncertainty and np.isfinite(test.p_value):
            spread = propagate(req, test.p_value, lambda r: self._p_for(r, use_fallback), self.cfg.rho_delta)

        # The flag and the reasons for it.
        unstable = spread is not None and not spread.decision_stable
        if use_fallback:
            reasons += [r for r in screening.reasons if r not in reasons]
            reasons.append(f"fallback null ({null.method})")
        if unstable:
            reasons.append(f"decision flips within rho uncertainty (p {spread.p_low:.3f}-{spread.p_high:.3f})")
        if rho_check.status == "warn":
            reasons += rho_check.reasons

        return finish(
            decision=test.decision, p_value=test.p_value,
            reliability_flag=AMBER if (use_fallback or unstable) else GREEN,
            uncertainty_range=None if spread is None else (spread.p_low, spread.p_high),
            flag_reasons=reasons or ["screening passed; decision stable"],
            rho_check=rho_check, screening=screening, boundary=boundary, null=null, test=test,
            uncertainty=spread,
            provenance={"route": screening.route, "null_method": null.method,
                        "test_mode": self.cfg.test_mode, "weights": self.cfg.weights},
        )

    def run_batch(self, requests):
        return [self.run(r) for r in requests]

    def _null(self, req, boundary, use_fallback, draws=None):
        if use_fallback:
            return simulated_null(req, boundary, self.cfg.fallback_method, draws or self.cfg.fallback_draws)
        return self.net(req, boundary)

    def _p_for(self, req, use_fallback):
        """p-value (and the Gamma, if any) for one rho variant."""
        boundary = find_boundary(req, self.cfg.test_mode)
        null = self._null(req, boundary, use_fallback, self.cfg.uncertainty_draws)
        p = minimal_effects_test(req, boundary, null).p_value
        return p, null.params.get("shape"), null.params.get("rate")

    def _run_direct(self, req, start):
        """The original path: network only, nothing checked."""
        ms = lambda: 1000 * (time.perf_counter() - start)
        unchecked = RhoCheck("unchecked")
        try:
            boundary = find_boundary(req, self.cfg.test_mode)
            null = self.net(req, boundary)
            test = minimal_effects_test(req, boundary, null)
        except Exception as error:   # the original path has no guard, so record what happened
            return PipelineResult(req, None, None, "-", None, [f"error: {error}"], unchecked,
                                  provenance={"route": "direct", "total_ms": ms()})
        return PipelineResult(req, test.decision, test.p_value, "-", None, ["direct to network"], unchecked,
                              None, boundary, null, test, None,
                              provenance={"route": "direct", "null_method": null.method, "total_ms": ms()})


def build_pipeline(project_root=None, **settings):
    """Load EXP002 (with O4's batching) and SIN-WEI's gate, ready to run.

    project_root is the baseline data folder; by default it is downloaded from GitHub
    (integrated_pipeline/baseline.py). Any PipelineConfig field can be passed.
    """
    from scrp_toolkit.config import resolve_project_paths
    from o4_efficiency import EfficientPredictor

    from .baseline import data_root
    config = PipelineConfig(**settings)
    paths = resolve_project_paths(project_root or data_root())
    predictor = EfficientPredictor(paths, weights=config.weights)
    return Pipeline(predictor, config)
