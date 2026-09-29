import numpy as np
import pandas as pd

from scrp_toolkit.dataset import _r_matrix_to_features, build_preprocessed_dataset
from scrp_toolkit.edge_cases import (
    _matrix_to_r_string,
    check_response_count_cases,
    generate_instance_data_rows,
    generate_response_count_cases,
)

P_MATRICES = {
    "ry9": np.array([
        [0.7, 0.1, 0.1, 0.1],
        [0.2, 0.5, 0.2, 0.1],
        [0.05, 0.05, 0.8, 0.1],
        [0.3, 0.3, 0.3, 0.1],
    ])
}


def test_matrix_to_r_string_round_trips_through_dataset_parser():
    P = P_MATRICES["ry9"]
    parsed = _r_matrix_to_features(_matrix_to_r_string(P))
    P_back = np.array([parsed[f"P_{i}"] for i in range(16)]).reshape(4, 4)
    assert np.allclose(P, P_back)


def test_generate_response_count_cases_are_all_length_4():
    cases = generate_response_count_cases()
    assert len(cases) > 0
    for case in cases:
        assert len(case.x) == 4
        assert len(case.y) == 4


def test_known_bad_cases_are_flagged_by_the_safety_check():
    cases = generate_response_count_cases()
    report = check_response_count_cases(cases, P_MATRICES)

    # The genuinely-invalid cases must be caught (either by raising, or the mismatch flag if not).
    unknown_question = report[report["label"] == "unknown_question_code"].iloc[0]
    assert unknown_question["raised"]
    assert unknown_question["matches_expectation"]

    # Zero-variance / single-respondent cases are realistic inputs (not "expect_failure"), but the
    # underlying feature pipeline currently produces NaN for them rather than a real number --
    # this is the known finding this generator exists to surface. If this ever starts passing,
    # the upstream 0/0 guard was added -- update this test (and the note in docs/findings.md) accordingly.
    all_same = report[report["label"] == "all_same_option"].iloc[0]
    assert all_same["produced_nan"]
    assert not all_same["matches_expectation"]


def _real_schema_rows(n, seed):
    """Rows shaped like the real instance_data CSVs (string exp_codes, full raw feature set)."""
    from scrp_toolkit.features import extract_features
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        P = rng.dirichlet(np.ones(4), size=4)
        feats = extract_features(rng.integers(1, 100, 4), rng.integers(1, 100, 4), "ry9", {"ry9": P})
        row = feats.drop([c for c in feats.index if c.startswith("P_")] + ["asymm"]).to_dict()
        row.update({"exp_code": f"orthog_B_1_{i}", "P_matrix": _matrix_to_r_string(P),
                    "P_matrix_question": "ry9", "shape": rng.uniform(1, 10), "rate": rng.uniform(1, 10)})
        rows.append(row)
    return pd.DataFrame(rows)


def test_generate_instance_data_rows_matches_real_schema_exactly():
    rows, labels = generate_instance_data_rows(n_per_case=5, seed=0)
    real = _real_schema_rows(3, seed=0)
    assert set(rows.columns) == set(real.columns)
    assert rows["exp_code"].is_unique
    assert rows["exp_code"].map(type).eq(str).all()
    assert set(labels["exp_code"]) == set(rows["exp_code"])
    assert np.allclose(rows[["X_1", "X_2", "X_3", "X_4"]].sum(axis=1), 1.0)


def test_edge_rows_mixed_with_real_rows_survive_preprocessing(tmp_path):
    """Regression test: an earlier version added label columns and int exp_codes to this CSV.
    Mixed with real data, dropna() then deleted every row and np.unique() raised on int/str."""
    real = _real_schema_rows(200, seed=1)
    rows, _labels = generate_instance_data_rows(n_per_case=10, seed=0)
    real.to_csv(tmp_path / "orthog_B_1.csv", index=False)
    rows.to_csv(tmp_path / "edge.csv", index=False)

    df, feature_cols, _ = build_preprocessed_dataset(str(tmp_path))
    # outlier trimming drops a few percent; the bug dropped 100%
    assert len(df) > 0.9 * (len(real) + len(rows))
    assert len(feature_cols) == 40


def test_generate_instance_data_rows_is_deterministic():
    a, la = generate_instance_data_rows(n_per_case=5, seed=7)
    b, lb = generate_instance_data_rows(n_per_case=5, seed=7)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(la, lb)
