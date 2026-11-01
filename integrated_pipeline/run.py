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
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from .boundary import find_boundary, minimal_effects_test
from .contracts import AMBER, FLAGGED, GREEN, RED, REJECT, PipelineResult, RhoCheck
from .nulls import NetworkNull, simulated_null
from .screen import load_gate, screen, validate_rho
from .uncertainty import propagate, rho_variants


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
        early, state = self._check_and_screen(req, start)
        return early if early is not None else self._decide(req, start, *state)

    def run_batch(self, requests):
        """Run several comparisons (for example a school's 40 items) with the same results as run().

        O4: the cheap per-request steps run first; then every network-route request and its rho
        variants go through the network in ONE call, with rho variants shared between requests
        on the same item; then each request is finished. total_ms is the batch time per request.
        """
        requests = list(requests)
        if self.cfg.mode == "direct" or not requests:
            return [self.run(r) for r in requests]
        start = time.perf_counter()
        staged = [(req, *self._check_and_screen(req, start)) for req in requests]
        wanted = [(i, req, state[1]) for i, (req, early, state) in enumerate(staged)
                  if early is None and state[2].route != FLAGGED]
        prefetched = dict(zip([i for i, _, _ in wanted],
                              self._network_nulls_many([(req, b) for _, req, b in wanted])))
        results = [early if early is not None else self._decide(req, start, *state, prefetched=prefetched.get(i))
                   for i, (req, early, state) in enumerate(staged)]
        per_request = 1000 * (time.perf_counter() - start) / len(requests)
        for r in results:
            r.provenance["total_ms"] = per_request
        return results

    def _check_and_screen(self, req, start):
        """Steps 1a and 1b. Returns (a RED result, None) or (None, (rho_check, boundary, screening))."""
        def rejected(**fields):
            return PipelineResult(request=req, decision=None, p_value=None, reliability_flag=RED,
                                  uncertainty_range=None, provenance={"total_ms": 1000 * (time.perf_counter() - start)},
                                  **fields)

        # 1a. Is rho usable?
        rho_check = validate_rho(req)
        if rho_check.status == "reject":
            return rejected(flag_reasons=list(rho_check.reasons), rho_check=rho_check,
                            screening=screen(req, rho_check, None, self.gate)), None

        # Boundary pair, then 1b. can the network be trusted here?
        boundary = find_boundary(req, self.cfg.test_mode)
        screening = screen(req, rho_check, boundary, self.gate, self.cfg.small_group_n, self.cfg.small_group_ratio)
        if screening.route == REJECT:
            return rejected(flag_reasons=list(screening.reasons), rho_check=rho_check,
                            screening=screening, boundary=boundary), None
        return None, (rho_check, boundary, screening)

    def _decide(self, req, start, rho_check, boundary, screening, prefetched=None):
        """Steps 2, 3 and 3+ for a request that passed 1a/1b. prefetched = (null, variant nulls) from the network."""
        # 2. Null distribution: network if screening passed, otherwise the fallback.
        reasons = []
        use_fallback = screening.route == FLAGGED
        variant_nulls = {}
        if not use_fallback:
            # O4: the request and its rho variants (step 3+) go through the network in one call.
            null, variant_nulls = prefetched or self._network_nulls_many([(req, boundary)])[0]
        else:
            null = self._null(req, boundary, use_fallback)
        if not use_fallback and not np.isfinite(null.q95):
            reasons.append("network returned no usable Gamma; fallback used")
            use_fallback = True
            null = self._null(req, boundary, True)

        # 3. The test, then 3+. how much it moves with rho.
        test = minimal_effects_test(req, boundary, null)
        spread = None
        if self.cfg.uncertainty and np.isfinite(test.p_value):
            spread = propagate(req, test.p_value, lambda r: self._p_for(r, use_fallback, variant_nulls),
                               self.cfg.rho_delta)

        # The flag and the reasons for it.
        unstable = spread is not None and not spread.decision_stable
        if use_fallback:
            reasons += [r for r in screening.reasons if r not in reasons]
            reasons.append(f"fallback null ({null.method})")
        if unstable:
            reasons.append(f"decision flips within rho uncertainty (p {spread.p_low:.3f}-{spread.p_high:.3f})")
        if rho_check.status == "warn":
            reasons += rho_check.reasons

        return PipelineResult(
            request=req, decision=test.decision, p_value=test.p_value,
            reliability_flag=AMBER if (use_fallback or unstable) else GREEN,
            uncertainty_range=None if spread is None else (spread.p_low, spread.p_high),
            flag_reasons=reasons or ["screening passed; decision stable"],
            rho_check=rho_check, screening=screening, boundary=boundary, null=null, test=test,
            uncertainty=spread,
            provenance={"route": screening.route, "null_method": null.method,
                        "test_mode": self.cfg.test_mode, "weights": self.cfg.weights,
                        "total_ms": 1000 * (time.perf_counter() - start)},
        )

    def _null(self, req, boundary, use_fallback, draws=None):
        if use_fallback:
            return simulated_null(req, boundary, self.cfg.fallback_method, draws or self.cfg.fallback_draws)
        return self.net(req, boundary)

    def _network_nulls_many(self, pairs):
        """Network nulls for (request, boundary) pairs and, if step 3+ is on, their rho variants, in ONE call.

        Returns one (null, variant_nulls) per pair; variant_nulls maps P-bytes -> (boundary, null)
        for the variants propagate() will ask for. The variants are the same ones propagate()
        builds (rho_variants is deterministic), so p-values and ranges are unchanged. Requests on
        the same item and group pair share their variants' boundaries.
        """
        reqs, boundaries, rows, shared = [], [], [], {}
        for req, boundary in pairs:
            own_row = len(reqs)
            reqs.append(req)
            boundaries.append(boundary)
            variant_keys = []
            if self.cfg.uncertainty:
                for P in rho_variants(req, self.cfg.rho_delta)[0]:
                    variant = replace(req, rho=tuple(map(tuple, P)))
                    key = (variant.P.tobytes(), variant.x.tobytes(), variant.y.tobytes(), variant.epsilon)
                    if key not in shared:
                        shared[key] = len(reqs)
                        reqs.append(variant)
                        boundaries.append(find_boundary(variant, self.cfg.test_mode))
                    variant_keys.append(key)
            rows.append((own_row, variant_keys))
        nulls = self.net.many(reqs, boundaries) if reqs else []
        return [(nulls[own_row], {key[0]: (boundaries[shared[key]], nulls[shared[key]]) for key in variant_keys})
                for own_row, variant_keys in rows]

    def _p_for(self, req, use_fallback, variant_nulls=None):
        """p-value (and the Gamma, if any) for one rho variant."""
        cached = None if use_fallback or not variant_nulls else variant_nulls.get(req.P.tobytes())
        if cached is not None:
            boundary, null = cached
        else:
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
