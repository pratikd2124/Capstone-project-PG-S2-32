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

    def _network_nulls_with_variants(self, req, boundary):
        return self.net(req, boundary), {}


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
    assert len(report["paths"]) == 3 and all(p["ms_per_comparison"] > 0 for p in report["paths"])
    assert {s["stage"] for s in report["stages"]} >= {"boundary pair", "rho range (O3)"}
    overall = report["equivalence"][0]
    assert overall["comparisons"] == "all with a decision" and overall["n"] > 0
    assert 0 <= overall["fp16_share_changed"] <= 1
