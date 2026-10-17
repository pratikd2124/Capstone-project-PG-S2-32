"""The integrated pipeline (scrp_toolkit/pipeline): stage contracts, routing and the maths.

Mechanics run on the synthetic project, like the other tests. Checks that need RYA's real
model (the simulator matches EXP002; SIN-WEI's acceptance numbers) are skipped when the real
data folder is not present.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from scrp_toolkit.config import resolve_project_paths
from scrp_toolkit.pipeline.boundary import find_boundary, minimal_effects_test
from scrp_toolkit.pipeline.contracts import AMBER, FLAGGED, GREEN, PASS, RED, REJECT, ComparisonRequest as R
from scrp_toolkit.pipeline.screen import load_gate, validate_rho
from scrp_toolkit.pipeline.simulate import B, d_hat_scaled, d_norm, integer_composition, simulate_d_hat
from scrp_toolkit.pipeline.uncertainty import perturb_rho_offdiagonal_mass, rho_posterior_draws

from . import synthetic_project

warnings.filterwarnings("ignore")
REAL = Path(__file__).resolve().parent.parent / "data"
P_EX = np.array([[0.8, 0.15, 0.04, 0.01], [0.1, 0.7, 0.15, 0.05], [0.03, 0.12, 0.7, 0.15], [0.01, 0.04, 0.15, 0.8]])


# ---- maths ------------------------------------------------------------------------------
def test_B_is_RYA_orthonormal_sum_to_zero_basis():
    assert np.allclose(B @ B.T, np.eye(3), atol=1e-12)
    assert np.allclose(B @ np.ones(4), 0, atol=1e-12)


def test_d_hat_matches_direct_formula():
    X, Y = np.array([10., 20, 5, 5]), np.array([3., 4, 20, 13])
    n, m = X.sum(), Y.sum()
    sig = [np.diag(r) - np.outer(r, r) for r in P_EX]
    S = sum(w * s for w, s in zip(X / n**2 + Y / m**2, sig))
    v = B @ (X / n - Y / m)
    expect = 0.5 * (1 / n + 1 / m) * v @ np.linalg.solve(B @ S @ B.T, v)
    assert d_hat_scaled(X[None], Y[None], P_EX)[0] == pytest.approx(expect, rel=1e-12)
    assert d_norm(X / n, Y / m, n, m, P_EX) == pytest.approx(expect, rel=1e-12)


def test_integer_composition_sums_to_n():
    c = integer_composition(np.array([0.333, 0.333, 0.334, 0.0]), 10)
    assert c.sum() == 10 and (c >= 0).all()


def test_simulator_is_reproducible_and_positive():
    a = simulate_d_hat([5, 5, 5, 5], [2, 8, 8, 2], P_EX, 500, np.random.default_rng(1))
    b = simulate_d_hat([5, 5, 5, 5], [2, 8, 8, 2], P_EX, 500, np.random.default_rng(1))
    assert np.array_equal(a, b) and (a >= 0).all()


def test_boundary_hits_epsilon_and_keeps_group_sizes():
    req = R.make("q", [40, 52, 7, 14], [9, 17, 54, 12], P_EX, epsilon=0.1)
    bp = find_boundary(req)
    assert bp.d_norm_boundary == pytest.approx(0.1, abs=1e-6)
    assert bp.x.sum() == pytest.approx(req.x.sum()) and bp.y.sum() == pytest.approx(req.y.sum())
    assert 0 < bp.shrink < 1
    pn = find_boundary(req, "point_null")
    assert np.allclose(pn.x / pn.x.sum(), pn.y / pn.y.sum())


def test_rho_perturbation_and_posterior_keep_rows_on_simplex():
    for c in (-0.05, 0.05):
        assert np.allclose(perturb_rho_offdiagonal_mass(P_EX, c).sum(axis=1), 1)
    assert np.allclose(perturb_rho_offdiagonal_mass(P_EX, 0), P_EX)
    d = rho_posterior_draws(np.full((4, 4), 10.0), 5)
    assert d.shape == (5, 4, 4) and np.allclose(d.sum(axis=2), 1)


# ---- stage 1a / 1b ----------------------------------------------------------------------
def test_rho_validation():
    ok = R.make("ry9", [1, 2, 3, 4], [4, 3, 2, 1], P_EX)
    assert validate_rho(ok).status == "pass"
    bad = P_EX.copy()
    bad[0] *= 1.06
    assert validate_rho(R.make("ry9", [1, 2, 3, 4], [4, 3, 2, 1], bad)).status == "reject"
    assert validate_rho(R.make("chs1", [1, 2, 3, 4], [4, 3, 2, 1], P_EX)).status == "reject"


def test_numpy_detector_matches_sin_wei_joblib():
    sk = pytest.importorskip("sklearn")
    from scrp_toolkit.pipeline.o1.o1_screening import O1ScreeningGate
    d = Path(__file__).resolve().parent.parent / "scrp_toolkit/pipeline/o1"
    orig = O1ScreeningGate(d / "O1_07_compatibility_detector_v1.joblib", d / "O1_07_compatibility_detector_v1_metadata.json")
    ours = load_gate()
    rng = np.random.default_rng(0)
    for _ in range(50):
        x, y = rng.integers(0, 60, 4) + 1, rng.integers(0, 3000, 4) + 1
        a = orig.screen_case(x, y, P_EX)["classifier"]["incompatibility_probability"]
        b = ours.screen_case(x, y, P_EX)["classifier"]["incompatibility_probability"]
        assert b == pytest.approx(a, rel=1e-9, abs=1e-12)


def test_sin_wei_interface_acceptance_cases():
    g = load_gate()
    P = [[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1], [0.1, 0.1, 0.1, 0.7]]
    r = g.screen_case([10, 0, 0, 0], [3, 2, 4, 1], P)
    assert r["route"] == "fallback_required" and r["reason"] == "original_pair_degenerate"
    assert g.screen_case([1, -1, 3, 2], [1, 2, 3, 4], P)["route"] == "invalid_input"


# ---- whole pipeline on the synthetic project --------------------------------------------
@pytest.fixture(scope="module")
def pipes(tmp_path_factory):
    pytest.importorskip("onnx")
    from scrp_toolkit.inference import UnreliabilityPredictor
    from scrp_toolkit.pipeline.run import Pipeline, PipelineConfig
    pred = UnreliabilityPredictor(resolve_project_paths(synthetic_project.build(tmp_path_factory.mktemp("p"))))
    robust = Pipeline(pred, PipelineConfig(fallback_draws=800, uncertainty_draws=300))
    direct = Pipeline(pred, PipelineConfig(mode="direct"), gate=robust.gate)
    return pred, robust, direct


def test_reject_gives_red_and_no_p_value(pipes):
    _, robust, _ = pipes
    bad = P_EX.copy()
    bad[0] *= 1.06
    res = robust.run(R.make("ry9", [40, 52, 7, 14], [9, 17, 54, 12], bad))
    assert res.reliability_flag == RED and res.p_value is None and res.screening.route == REJECT


def test_degenerate_group_goes_to_fallback_and_amber(pipes):
    pred, robust, _ = pipes
    res = robust.run(R.make("ry9", [0, 0, 12, 0], [30, 41, 25, 14], pred.p_matrices["ry9"]))
    assert res.screening.route == FLAGGED and res.null.method == "mc_empirical"
    assert res.reliability_flag == AMBER and 0 <= res.p_value <= 1
    assert res.uncertainty_range[0] <= res.p_value <= res.uncertainty_range[1]


def test_outside_training_range_is_flagged(pipes):
    pred, robust, _ = pipes
    res = robust.run(R.make("ry9", [3000, 5000, 4000, 3000], [2000, 3000, 2000, 2000], pred.p_matrices["ry9"]))
    assert res.screening.outside_training_range and res.screening.route == FLAGGED


def test_direct_mode_is_the_network_unchanged(pipes):
    pred, _, direct = pipes
    req = R.make("ry9", [40, 52, 7, 14], [9, 17, 54, 12], pred.p_matrices["ry9"])
    res = direct.run(req)
    bp = find_boundary(req)
    shape, rate = pred.predict(bp.x, bp.y, "ry9")
    assert res.null.params["shape"] == pytest.approx(shape, rel=1e-6)
    assert res.null.params["rate"] == pytest.approx(rate, rel=1e-6)
    assert res.screening is None and res.uncertainty is None


def test_summary_row_has_the_user_facing_fields(pipes):
    pred, robust, _ = pipes
    row = robust.run(R.make("ry9", [40, 52, 7, 14], [9, 17, 54, 12], pred.p_matrices["ry9"])).summary()
    for k in ("decision", "p_value", "p_range", "flag", "route"):
        assert k in row


# ---- real model (skipped without RYA's data) --------------------------------------------
@pytest.mark.skipif(not (REAL / "Unreliability Matrices").exists(), reason="RYA data not present")
def test_simulator_matches_EXP002_on_ry9():
    from scrp_toolkit.inference import UnreliabilityPredictor
    pred = UnreliabilityPredictor(resolve_project_paths(REAL))
    x, y = np.array([40, 52, 7, 14]), np.array([9, 17, 54, 12])
    shape, rate = pred.predict(x, y, "ry9")
    assert (shape, rate) == (pytest.approx(9.970475608), pytest.approx(20.926446973))
    sim = simulate_d_hat(x, y, pred.p_matrices["ry9"], 20000, np.random.default_rng(3))
    assert sim.mean() == pytest.approx(shape / rate, rel=0.02)   # rate used as-is
