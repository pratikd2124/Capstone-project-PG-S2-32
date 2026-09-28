import numpy as np

from scrp_toolkit.reliability import asymm_calc, score_asymmetry


def test_asymm_calc_zero_for_symmetric_matrix():
    P = np.array([
        [0.7, 0.1, 0.1, 0.1],
        [0.1, 0.7, 0.1, 0.1],
        [0.1, 0.1, 0.7, 0.1],
        [0.1, 0.1, 0.1, 0.7],
    ])
    assert asymm_calc(P) == 0.0


def test_asymm_calc_positive_for_asymmetric_matrix():
    P = np.array([
        [0.9, 0.05, 0.03, 0.02],
        [0.4, 0.4, 0.1, 0.1],
        [0.2, 0.2, 0.5, 0.1],
        [0.1, 0.1, 0.1, 0.7],
    ])
    assert asymm_calc(P) > 0.0


def test_score_asymmetry_sorts_descending():
    p_matrices = {
        "low": np.eye(4),
        "high": np.array([
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.5, 0.5, 0.0],
            [0.0, 0.0, 0.5, 0.5],
            [0.5, 0.0, 0.0, 0.5],
        ]),
    }
    scores = score_asymmetry(p_matrices)
    assert list(scores.index) == ["high", "low"]
    assert scores.iloc[0] >= scores.iloc[1]
