import numpy as np
import pytest

from scrp_toolkit.features import extract_features, moment_calculator, transform_p_matrix

P_MATRICES = {
    "ry9": np.array([
        [0.7, 0.1, 0.1, 0.1],
        [0.1, 0.7, 0.1, 0.1],
        [0.1, 0.1, 0.7, 0.1],
        [0.1, 0.1, 0.1, 0.7],
    ])
}


def test_moment_calculator_uniform_distribution_mean_and_std():
    uniform = np.array([0.25, 0.25, 0.25, 0.25])
    moments = moment_calculator(uniform, "X")
    assert moments["mean_X"] == pytest.approx(2.5)
    # variance of a discrete uniform over {1,2,3,4} is 1.25 -> std = sqrt(1.25)
    assert moments["stdev_X"] == pytest.approx(np.sqrt(1.25))


def test_transform_p_matrix_flattens_16_cells_plus_asymm():
    feats = transform_p_matrix("ry9", P_MATRICES)
    assert len(feats) == 17  # P_0..P_15 + asymm
    assert feats["asymm"] == pytest.approx(0.0)


def test_extract_features_shapes_and_normalisation():
    x = np.array([40, 52, 7, 14])
    y = np.array([9, 17, 54, 12])
    feats = extract_features(x, y, "ry9", P_MATRICES)

    # X_i / Y_i are normalised response proportions and sum to ~1
    assert sum(feats[f"X_{i}"] for i in range(1, 5)) == pytest.approx(1.0)
    assert sum(feats[f"Y_{i}"] for i in range(1, 5)) == pytest.approx(1.0)
    assert feats["n_X"] == x.sum()
    assert feats["n_Y"] == y.sum()
    assert feats["euclid_dist_XY"] >= 0

    # Every expected column is present exactly once
    expected_cols = (
        [f"X_{i}" for i in range(1, 5)]
        + [f"Y_{i}" for i in range(1, 5)]
        + [f"XmY_{i}" for i in range(1, 5)]
        + ["mean_X", "stdev_X", "skewness_X", "kurtosis_X"]
        + ["mean_Y", "stdev_Y", "skewness_Y", "kurtosis_Y"]
        + ["euclid_dist_XY", "n_X", "n_Y"]
        + [f"P_{i}" for i in range(16)]
        + ["asymm"]
    )
    assert sorted(feats.index) == sorted(expected_cols)


def test_six_option_items_are_rejected_not_silently_scored():
    import pytest
    P6 = {"chs1": np.full((6, 6), 1 / 6)}
    with pytest.raises(ValueError, match="4-option"):
        extract_features(np.array([1, 2, 3, 4]), np.array([4, 3, 2, 1]), "chs1", P6)
