"""predict_batch / extract_features_batch give what predict / extract_features give.

On the machine they were developed on (Apple M2) the results are bit-identical. On other CPUs
the BLAS library may add the 4 terms of a dot product in a different order for one path than
the other, so features can differ in the last bit (seen on GitHub's Linux runners: up to
8.9e-16, in skewness/kurtosis only). RYA's own np.dot code differs across machines in the same
way. The tests therefore allow a few units in the last place, which is far below anything
the model can see (its inputs are 32-bit).
"""
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


def test_features_match_to_the_last_bits(predictor, cases):
    labels = predictor.metadata["feature_labels"]
    with np.errstate(invalid="ignore", divide="ignore"):
        one_by_one = np.array([extract_features(x, y, q, predictor.p_matrices).reindex(labels).to_numpy()
                               for x, y, q in cases])
        batch = extract_features_batch(*_unzip(cases), predictor.p_matrices)[labels].to_numpy()
    nan = np.isnan(one_by_one)
    np.testing.assert_array_equal(np.isnan(batch), nan)                    # NaN in exactly the same places
    np.testing.assert_array_max_ulp(batch[~nan], one_by_one[~nan], maxulp=4)


def test_predictions_match(predictor, cases):
    with np.errstate(invalid="ignore", divide="ignore"):
        one_by_one = np.array([predictor.predict(x, y, q) for x, y, q in cases])
        shape, rate = predictor.predict_batch(*_unzip(cases), chunk_size=64)   # several chunks
    # In practice identical; the tolerance only covers a last-bit feature difference flipping
    # one 32-bit model input (relative effect ~1e-7).
    np.testing.assert_allclose(np.column_stack([shape, rate]), one_by_one, rtol=1e-6, atol=0, equal_nan=True)


def test_batch_refuses_six_option_items(predictor):
    with pytest.raises(ValueError, match="4-option"):
        extract_features_batch(np.ones((1, 4)), np.ones((1, 4)), ["chs1"], predictor.p_matrices)


def test_empty_batch(predictor):
    shape, rate = predictor.predict_batch(np.empty((0, 4)), np.empty((0, 4)), [])
    assert shape.shape == rate.shape == (0,)
