"""The profiler runs on a fake project, and its batched network gives today's answers."""
import json

import numpy as np
import pytest

from scrp_toolkit.cli import main
from scrp_toolkit.config import resolve_project_paths
from scrp_toolkit.inference import UnreliabilityPredictor, batched_onnx_session
from scrp_toolkit.profiling import net_from_onnx, sample_cases

from . import synthetic_project


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    pytest.importorskip("onnx")
    return synthetic_project.build(tmp_path_factory.mktemp("project"))


def test_batched_network_matches_one_row_onnx(project):
    predictor = UnreliabilityPredictor(resolve_project_paths(project))
    X = np.random.default_rng(0).random((32, len(predictor.metadata["feature_labels"]))).astype(np.float32)
    one_row = np.stack([predictor._run(v) for v in X])

    session = batched_onnx_session(predictor.paths.onnx_path)
    batched = session.run(None, {session.get_inputs()[0].name: X})[0]
    np.testing.assert_array_equal(batched, one_row)

    import torch
    with torch.no_grad():
        torch_out = net_from_onnx(predictor.paths.onnx_path, predictor.metadata)(torch.from_numpy(X)).numpy()
    np.testing.assert_allclose(torch_out, one_row, atol=1e-5)


def test_sample_cases_are_valid_four_option_inputs(project):
    predictor = UnreliabilityPredictor(resolve_project_paths(project))
    for x, y, q in sample_cases(predictor.p_matrices, 50):
        assert np.shape(predictor.p_matrices[q]) == (4, 4)
        assert len(x) == len(y) == 4 and x.sum() >= 10 and y.sum() >= 10


def test_profile_command_writes_report(project, tmp_path):
    out = tmp_path / "profile.json"
    assert main(["profile", "--project-root", str(project), "--n-cases", "40", "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert set(report["stages_us"]) >= {"features", "normalise", "network", "reverse_normalise", "predict_total"}
    assert report["max_abs_diff_vs_onnx"]["onnx_batched"] == 0.0
    assert len(report["techniques"]) == 3 and all(t["identical"] for t in report["techniques"])
