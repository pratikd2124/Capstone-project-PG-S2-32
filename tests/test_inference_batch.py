"""predict_batch / extract_features_batch give exactly what predict / extract_features give."""
import numpy as np
import pytest

from scrp_toolkit.config import resolve_project_paths
from scrp_toolkit.features import extract_features, extract_features_batch
from scrp_toolkit.inference import UnreliabilityPredictor
from scrp_toolkit.profiling import sample_cases

from . import synthetic_project


@pytest.fixture(scope="module")
def predictor(tmp_path_factory):
    pytest.importorskip("onnx")
    return UnreliabilityPredictor(resolve_project_paths(synthetic_project.build(tmp_path_factory.mktemp("project"))))


@pytest.fixture(scope="module")
def cases(predictor):
    cases = sample_cases(predictor.p_matrices, 300, seed=3)
    # Awkward inputs: everyone on one answer (NaN skew), a group of one, a very lopsided pair.
    cases += [(np.array([5, 0, 0, 0]), np.array([0, 3, 0, 0]), "ry9"),
              (np.array([1, 0, 0, 0]), np.array([0, 0, 0, 1]), "ry10"),
              (np.array([0, 7, 0, 2]), np.array([9000, 1, 0, 0]), "ry9")]
    return cases


def _unzip(cases):
    return np.array([c[0] for c in cases]), np.array([c[1] for c in cases]), [c[2] for c in cases]


def test_features_are_bit_identical(predictor, cases):
    labels = predictor.metadata["feature_labels"]
    with np.errstate(invalid="ignore", divide="ignore"):
        one_by_one = np.array([extract_features(x, y, q, predictor.p_matrices).reindex(labels).to_numpy()
                               for x, y, q in cases])
        batch = extract_features_batch(*_unzip(cases), predictor.p_matrices)[labels].to_numpy()
    np.testing.assert_array_equal(batch, one_by_one)      # NaNs in the same places count as equal


def test_predictions_are_bit_identical(predictor, cases):
    with np.errstate(invalid="ignore", divide="ignore"):
        one_by_one = np.array([predictor.predict(x, y, q) for x, y, q in cases])
        shape, rate = predictor.predict_batch(*_unzip(cases), chunk_size=64)   # several chunks
    np.testing.assert_array_equal(np.column_stack([shape, rate]), one_by_one)


def test_batch_refuses_six_option_items(predictor):
    with pytest.raises(ValueError, match="4-option"):
        extract_features_batch(np.ones((1, 4)), np.ones((1, 4)), ["chs1"], predictor.p_matrices)


def test_empty_batch(predictor):
    shape, rate = predictor.predict_batch(np.empty((0, 4)), np.empty((0, 4)), [])
    assert shape.shape == rate.shape == (0,)
