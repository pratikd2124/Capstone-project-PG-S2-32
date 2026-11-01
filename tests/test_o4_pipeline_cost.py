"""O4 in the integrated pipeline: one network call per comparison, same results, and the cost report."""
import json
import warnings

import numpy as np
import pytest

from integrated_pipeline.benchmark import make_cases
from integrated_pipeline.boundary import find_boundary
from integrated_pipeline.run import Pipeline, build_pipeline
from integrated_pipeline.simulate import sigma_stack
from o4_efficiency.cli import main

from . import synthetic_project

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    pytest.importorskip("onnx")
    return synthetic_project.build(tmp_path_factory.mktemp("project"))


@pytest.fixture(scope="module")
def pipe(project):
    return build_pipeline(str(project))


@pytest.fixture(scope="module")
def cases(pipe):
    return make_cases(pipe.net.predictor.p_matrices, n_cases=40, seed=3)


class _OnePerCall(Pipeline):
    """The pipeline as it was: the request and each rho variant call the network separately."""

    def _network_nulls_many(self, pairs):
        return [(self.net(req, boundary), {}) for req, boundary in pairs]


def test_many_matches_one_call_per_pair(pipe, cases):
    reqs = cases[:12]
    boundaries = [find_boundary(r) for r in reqs]
    together = pipe.net.many(reqs, boundaries)
    one_by_one = [pipe.net(r, b) for r, b in zip(reqs, boundaries)]
    for a, b in zip(together, one_by_one):
        assert a.params == b.params or all(np.isnan(v) for v in (*a.params.values(), *b.params.values()))
        assert a.q95 == b.q95 or (np.isnan(a.q95) and np.isnan(b.q95))


def test_batched_variants_give_the_same_results(pipe, cases):
    old = _OnePerCall(pipe.net.predictor, pipe.cfg, pipe.gate)
    for req in cases:
        a, b = pipe.run(req), old.run(req)
        assert (a.decision, a.reliability_flag, a.flag_reasons) == (b.decision, b.reliability_flag, b.flag_reasons)
        assert a.p_value == b.p_value or (a.p_value is None and b.p_value is None)
        assert a.uncertainty_range == b.uncertainty_range


def _same(a, b):
    return ((a.decision, a.reliability_flag, a.flag_reasons, a.uncertainty_range, a.p_value)
            == (b.decision, b.reliability_flag, b.flag_reasons, b.uncertainty_range, b.p_value))


def test_run_batch_gives_the_same_results_as_run(pipe, cases):
    # Repeat some requests so rho variants are shared between them.
    requests = cases + cases[:10]
    batched = pipe.run_batch(requests)
    assert len(batched) == len(requests)
    assert all(_same(a, pipe.run(r)) for a, r in zip(batched, requests))
    assert pipe.run_batch([]) == []


def test_sigma_stack_is_cached_and_read_only():
    P = np.full((4, 4), 0.25)
    first, second = sigma_stack(P), sigma_stack(P.copy())
    assert first is second and not first.flags.writeable
    with pytest.raises(ValueError):
        first[0, 0, 0] = 1.0


def test_pipeline_cost_command_writes_report(project, tmp_path):
    out = tmp_path / "pipeline_cost.json"
    assert main(["pipeline-cost", "--project-root", str(project), "--n-cases", "20",
                 "--equivalence-cases", "20", "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert [p["path"] for p in report["paths"]][0] == "original: direct to network"
    assert len(report["paths"]) == 4 and all(p["ms_per_comparison"] > 0 for p in report["paths"])
    assert {s["stage"] for s in report["stages"]} >= {"boundary pair", "rho range (O3)"}
    overall = report["equivalence"][0]
    assert overall["comparisons"] == "all with a decision" and overall["n"] > 0
    assert 0 <= overall["fp16_share_changed"] <= 1


def test_closed_form_d_hat_matches_general_path():
    from integrated_pipeline.simulate import _CLOSED_FORM_MIN_ROWS, _quad_form, d_hat_scaled

    rng = np.random.default_rng(0)
    P = np.array([[0.8, 0.15, 0.04, 0.01], [0.1, 0.7, 0.15, 0.05], [0.03, 0.12, 0.7, 0.15], [0.01, 0.04, 0.15, 0.8]])
    X = rng.multinomial(300, [0.25, 0.35, 0.25, 0.15], size=500).astype(float)
    Y = rng.multinomial(40, [0.2, 0.3, 0.3, 0.2], size=500).astype(float)
    fast = d_hat_scaled(X, Y, P)                                   # 500 rows: closed form
    one_by_one = np.array([d_hat_scaled(x, y, P)[0] for x, y in zip(X, Y)])   # 1 row: general path
    assert len(X) >= _CLOSED_FORM_MIN_ROWS
    np.testing.assert_allclose(fast, one_by_one, rtol=1e-12, atol=0)


def test_closed_form_falls_back_on_singular_rows():
    """Where the 3x3 matrix is singular the closed form gives up and the general path decides."""
    from integrated_pipeline.simulate import d_hat_scaled

    X = np.tile([10.0, 5.0, 3.0, 2.0], (100, 1))
    Y = np.tile([8.0, 6.0, 4.0, 2.0], (100, 1))
    for P in (np.eye(4),                                                     # every Sigma_j is zero
              np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0.9, 0.1], [0, 0, 0.1, 0.9]])):   # rank-deficient
        many, one = d_hat_scaled(X, Y, P), d_hat_scaled(X[:1], Y[:1], P)
        np.testing.assert_array_equal(many, np.repeat(one, len(X)))
