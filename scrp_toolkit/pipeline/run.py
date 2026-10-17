"""The pipeline: one call from a request to a decision with a reliability flag.

    rho validation (1a) -> boundary pair -> screening (1b, both pairs)
        PASS    -> EXP002 network (2a)  -> Gamma
        FLAGGED -> fallback (2b)        -> simulated null
    -> minimal-effects test (3) -> rho uncertainty (3+) -> PipelineResult

``mode="direct"`` reproduces the original configuration for benchmarking: every request goes
straight to the network with no checks, no fallback and no uncertainty.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Iterable, List, Optional

import numpy as np

from .boundary import find_boundary, minimal_effects_test
from .contracts import (AMBER, FLAGGED, GREEN, PASS, RED, REJECT, ComparisonRequest, PipelineResult,
                        RhoCheck, ScreeningResult)
from .nulls import NetworkNull, simulated_null
from .screen import load_gate, screen, validate_rho
from .uncertainty import propagate


@dataclass
class PipelineConfig:
    mode: str = "robust"              # 'robust' or 'direct'
    test_mode: str = "boundary"       # 'boundary' (paper, provisional) or 'point_null' (Murat's stand-in)
    weights: str = "fp32"             # 'fp32' (RYA's model) or 'fp16' (browser model)
    fallback_method: str = "mc_empirical"
    fallback_draws: int = 4000
    uncertainty: bool = True
    rho_delta: float = 0.05
    uncertainty_draws: int = 1500     # simulator draws per rho variant on the fallback route
    screen_threshold: Optional[float] = None   # None = SIN-WEI's 0.10
    small_group_n: Optional[int] = 30          # integration rule; None = SIN-WEI's gate only
    small_group_ratio: float = 20.0


class Pipeline:
    def __init__(self, predictor, config: PipelineConfig | None = None, gate=None):
        self.cfg = config or PipelineConfig()
        self.net = NetworkNull(predictor)
        self.gate = gate or load_gate(threshold=self.cfg.screen_threshold)

    # -- helpers ---------------------------------------------------------------------------
    def _null(self, req, bp, use_fallback, draws=None):
        if use_fallback:
            return simulated_null(req, bp, self.cfg.fallback_method, draws or self.cfg.fallback_draws)
        return self.net(req, bp)

    def _one_p(self, req, use_fallback):
        bp = find_boundary(req, self.cfg.test_mode)
        null = self._null(req, bp, use_fallback, self.cfg.uncertainty_draws)
        t = minimal_effects_test(req, bp, null)
        return t.p_value, null.params.get("shape"), null.params.get("rate")

    # -- main entry ------------------------------------------------------------------------
    def run(self, req: ComparisonRequest) -> PipelineResult:
        t0 = time.perf_counter()
        timings = {}
        if self.cfg.mode == "direct":
            return self._run_direct(req, t0)

        t = time.perf_counter()
        rc = validate_rho(req)
        timings["rho_ms"] = 1000 * (time.perf_counter() - t)
        if rc.status == "reject":
            return PipelineResult(req, None, None, RED, None, list(rc.reasons), rc,
                                  ScreeningResult(REJECT, rc.reasons),
                                  provenance={"total_ms": 1000 * (time.perf_counter() - t0), **timings})

        t = time.perf_counter()
        bp = find_boundary(req, self.cfg.test_mode)
        timings["boundary_ms"] = 1000 * (time.perf_counter() - t)

        t = time.perf_counter()
        sr = screen(req, rc, bp, self.gate, self.cfg.small_group_n, self.cfg.small_group_ratio)
        timings["screen_ms"] = 1000 * (time.perf_counter() - t)
        if sr.route == REJECT:
            return PipelineResult(req, None, None, RED, None, list(sr.reasons), rc, sr, bp,
                                  provenance={"total_ms": 1000 * (time.perf_counter() - t0), **timings})

        reasons: List[str] = []
        use_fallback = sr.route == FLAGGED
        null = self._null(req, bp, use_fallback)
        if not use_fallback and not np.isfinite(null.q95):
            # Screening passed but the network returned no usable Gamma (e.g. NaN features).
            reasons.append("network returned no usable Gamma; fallback used")
            use_fallback = True
            null = self._null(req, bp, True)
        timings["null_ms"] = null.time_ms

        test = minimal_effects_test(req, bp, null)

        unc = None
        if self.cfg.uncertainty and np.isfinite(test.p_value):
            t = time.perf_counter()
            unc = propagate(req, test.p_value, lambda r: self._one_p(r, use_fallback), self.cfg.rho_delta)
            timings["uncertainty_ms"] = 1000 * (time.perf_counter() - t)

        if use_fallback:
            reasons.extend(r for r in sr.reasons if r not in reasons)
            reasons.append(f"fallback null ({null.method})")
        if unc is not None and not unc.decision_stable:
            reasons.append(f"decision flips within rho uncertainty (p {unc.p_low:.3f}-{unc.p_high:.3f})")
        if rc.status == "warn":
            reasons.extend(rc.reasons)
        flag = AMBER if (use_fallback or (unc is not None and not unc.decision_stable)) else GREEN

        return PipelineResult(
            req, test.decision, test.p_value, flag,
            None if unc is None else (unc.p_low, unc.p_high),
            reasons or ["screening passed; decision stable"], rc, sr, bp, null, test, unc,
            provenance={"route": sr.route, "null_method": null.method, "test_mode": self.cfg.test_mode,
                        "weights": self.cfg.weights, "total_ms": 1000 * (time.perf_counter() - t0), **timings},
        )

    def _run_direct(self, req, t0):
        """Original configuration: straight to the network, nothing checked."""
        rc = RhoCheck("unchecked")
        try:
            bp = find_boundary(req, self.cfg.test_mode)
            null = self.net(req, bp)
            test = minimal_effects_test(req, bp, null)
        except Exception as exc:     # the original path has no guard; record what happens
            return PipelineResult(req, None, None, "-", None, [f"error: {exc}"], rc,
                                  provenance={"route": "direct", "total_ms": 1000 * (time.perf_counter() - t0)})
        return PipelineResult(req, test.decision, test.p_value, "-", None, ["direct to network"], rc, None, bp,
                              null, test, None,
                              provenance={"route": "direct", "null_method": null.method,
                                          "total_ms": 1000 * (time.perf_counter() - t0)})

    def run_batch(self, reqs: Iterable[ComparisonRequest]) -> List[PipelineResult]:
        return [self.run(r) for r in reqs]


def build_pipeline(project_root: str = "data", **cfg) -> Pipeline:
    """Convenience: load EXP002 and SIN-WEI's gate from the project folder."""
    from ..config import resolve_project_paths
    from ..inference import UnreliabilityPredictor
    config = PipelineConfig(**cfg)
    predictor = UnreliabilityPredictor(resolve_project_paths(project_root), weights=config.weights)
    return Pipeline(predictor, config)
