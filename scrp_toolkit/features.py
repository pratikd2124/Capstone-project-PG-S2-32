"""Turning answer counts into the 40 numbers the model reads.

The model never sees raw survey answers. For one item and two groups of students (x = before,
y = after, each a count of how many picked answers 1..4), it sees:

    X_1..4, Y_1..4        the share of each group picking each answer
    XmY_1..4              the gap between the two groups for each answer
    mean/stdev/skewness/kurtosis of each group's answers
    euclid_dist_XY        the overall distance between the two groups
    n_X, n_Y              how many students are in each group
    P_0..P_15, asymm      the item's unreliability matrix (see reliability.py)

This is the same maths as RYA's feature_calculation_portal_script.py, and it gives identical
numbers (walkthrough Step 4 runs both side by side). The one deliberate difference: we refuse
6-option items, which RYA's script would quietly mis-score.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .reliability import asymm_calc


def moment_calculator(prob_dist: np.ndarray, dist_name: str, num_types: int = 4) -> pd.Series:
    """Mean, spread, skew and "peakedness" of one group's answers.

    If everyone gives the same answer the spread is 0 and skew/kurtosis come out as NaN.
    That's inherited from RYA's code and flagged for the input-safety detector.
    """
    rv_vec = np.arange(1, num_types + 1)          # the answer values 1, 2, 3, 4
    mean = np.dot(rv_vec, prob_dist)
    stdev = np.sqrt(np.dot((rv_vec - mean) ** 2, prob_dist))
    skewness = np.dot((rv_vec - mean) ** 3, prob_dist) / stdev ** 3
    kurtosis = np.dot((rv_vec - mean) ** 4, prob_dist) / stdev ** 4
    return pd.Series({
        f"mean_{dist_name}": mean,
        f"stdev_{dist_name}": stdev,
        f"skewness_{dist_name}": skewness,
        f"kurtosis_{dist_name}": kurtosis,
    })


def transform_p_matrix(question: str, p_matrices: Dict[str, np.ndarray]) -> pd.Series:
    """The item's matrix laid out as P_0..P_15 (row by row), plus its asymm score."""
    P = np.asarray(p_matrices[question])
    d = {f"P_{i}": P.flat[i] for i in range(P.size)}
    d["asymm"] = asymm_calc(P)
    return pd.Series(d)


def extract_features(x: np.ndarray, y: np.ndarray, question: str, p_matrices: Dict[str, np.ndarray]) -> pd.Series:
    """All 40 model inputs for one item, from the before (x) and after (y) answer counts.

    x, y      4 counts each: how many students picked answers 1, 2, 3, 4
    question  the item code, e.g. "ry9" (must be in ``p_matrices``)

    Raises ValueError for anything that isn't a 4-option item. EXP002 was only ever trained on
    4x4 matrices; the matrices file also has six 6-option items (chs1-chs6), and pushing those
    through would silently scramble the matrix and ignore answers 5 and 6.
    """
    P = np.asarray(p_matrices[question])
    if P.shape != (4, 4):
        raise ValueError(f"Item '{question}' has a {P.shape[0]}x{P.shape[1]} unreliability matrix; "
                         "the EXP002 network only supports 4-option items (4x4).")
    if np.size(x) != 4 or np.size(y) != 4:
        raise ValueError(f"Expected 4 response counts for x and y, got {np.size(x)} and {np.size(y)}.")

    # Turn counts into shares of each group.
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n_X, n_Y = x.sum(), y.sum()
    X, Y = x / n_X, y / n_Y

    XmY = np.abs(X - Y)
    euclid_dist_XY = np.round(np.sqrt(np.sum((X - Y) ** 2)), 4)   # RYA rounds this one to 4 dp

    # The order matters: it must match the feature_labels list stored in EXP002.json.
    parts = [
        pd.Series({f"X_{i + 1}": X[i] for i in range(4)}),
        pd.Series({f"Y_{i + 1}": Y[i] for i in range(4)}),
        pd.Series({f"XmY_{i + 1}": XmY[i] for i in range(4)}),
        moment_calculator(X, "X"),
        moment_calculator(Y, "Y"),
        pd.Series({"euclid_dist_XY": euclid_dist_XY}),
        pd.Series({"n_X": n_X, "n_Y": n_Y}),
        transform_p_matrix(question, p_matrices),
    ]
    return pd.concat(parts)


# ---------------------------------------------------------------------------------------------
# Many items at once
# ---------------------------------------------------------------------------------------------

def _rowdot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """np.dot of each row of ``a`` with the same row of ``b``.

    Written as a stack of tiny matrix products on purpose: numpy sends these to the same BLAS
    dot routine as ``np.dot`` on one row, so the result is bit-for-bit what
    ``moment_calculator`` gives. ``(a * b).sum(axis=1)`` adds in a different order and drifts
    in the last digit for about a quarter of rows.
    """
    a, b = np.broadcast_arrays(a, b)
    return np.matmul(a[:, None, :], b[:, :, None])[:, 0, 0]


def extract_features_batch(x: np.ndarray, y: np.ndarray, questions, p_matrices: Dict[str, np.ndarray]) -> pd.DataFrame:
    """``extract_features`` for many items at once: one row per item, the same 40 columns.

    x, y       (N, 4) answer counts, before and after
    questions  N item codes

    Gives exactly the same numbers as calling ``extract_features`` row by row (tested), about
    100x faster, because the maths runs on whole columns instead of building 16 small pandas
    objects per item. Same checks too: 4-option items only.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    questions = list(questions)
    if x.ndim != 2 or x.shape[1] != 4 or y.shape != x.shape or len(questions) != len(x):
        raise ValueError(f"Expected x and y of shape (N, 4) and N questions, got {x.shape}, {y.shape}, {len(questions)}.")

    # The item part (P_0..P_15, asymm) is the same for every row of an item: work it out once.
    item_codes, item_index = np.unique(questions, return_inverse=True)
    item_rows = []
    for q in item_codes:
        P = np.asarray(p_matrices[q])
        if P.shape != (4, 4):
            raise ValueError(f"Item '{q}' has a {P.shape[0]}x{P.shape[1]} unreliability matrix; "
                             "the EXP002 network only supports 4-option items (4x4).")
        item_rows.append(np.append(P.ravel(), asymm_calc(P)))
    item_part = np.array(item_rows).reshape(-1, 17)[item_index]

    n_X, n_Y = x.sum(axis=1), y.sum(axis=1)
    X, Y = x / n_X[:, None], y / n_Y[:, None]
    XmY = np.abs(X - Y)
    euclid_dist_XY = np.round(np.sqrt(np.sum((X - Y) ** 2, axis=1)), 4)

    rv_vec = np.arange(1, 5, dtype=float)

    def moments(prob):
        mean = _rowdot(rv_vec, prob)
        dev = rv_vec - mean[:, None]
        stdev = np.sqrt(_rowdot(dev ** 2, prob))
        return [mean, stdev, _rowdot(dev ** 3, prob) / stdev ** 3, _rowdot(dev ** 4, prob) / stdev ** 4]

    columns = (
        [f"X_{i + 1}" for i in range(4)] + [f"Y_{i + 1}" for i in range(4)] + [f"XmY_{i + 1}" for i in range(4)]
        + [f"{m}_X" for m in ("mean", "stdev", "skewness", "kurtosis")]
        + [f"{m}_Y" for m in ("mean", "stdev", "skewness", "kurtosis")]
        + ["euclid_dist_XY", "n_X", "n_Y"] + [f"P_{i}" for i in range(16)] + ["asymm"]
    )
    values = np.column_stack([X, Y, XmY, *moments(X), *moments(Y), euclid_dist_XY, n_X, n_Y, item_part])
    return pd.DataFrame(values, columns=columns)
