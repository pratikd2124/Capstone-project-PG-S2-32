import numpy as np
import pandas as pd

from scrp_toolkit.dataset import build_preprocessed_dataset, train_test_split_like_original


def _r_vector(values):
    return "c(" + ", ".join(str(v) for v in values) + ")"


def _make_synthetic_instance_csv(path, n_rows=40, seed=0):
    """A minimal stand-in for the real instance_data/*.csv files: enough columns for
    build_preprocessed_dataset to exercise its P_matrix parsing, dedup, normalisation and
    outlier-trimming logic without needing the real (large) project data.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_rows):
        P = rng.dirichlet(np.ones(4), size=4).flatten()  # 16 values, rows sum to 1 (like a real P-matrix)
        rows.append({
            "exp_code": f"orthog_B_1_{i}",  # real data uses string codes like this
            "some_feature": rng.normal(),
            "n_X": int(rng.integers(5, 500)),
            "n_Y": int(rng.integers(5, 500)),
            "P_matrix": _r_vector(P),
            "P_matrix_question": "ry9",
            "shape": rng.uniform(1, 10),
            "rate": rng.uniform(1, 10),
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def test_build_preprocessed_dataset_shapes_and_normalisation(tmp_path):
    csv_path = tmp_path / "orthog_B_0.csv"
    _make_synthetic_instance_csv(csv_path, n_rows=50)

    df, feature_cols, min_max_df = build_preprocessed_dataset(str(tmp_path), two_sided_outlier_percentage=1)

    # exp_code stays raw (not normalised); everything else is min-max normalised into [0, 1]
    assert "exp_code" not in feature_cols
    for col in feature_cols + ["shape", "rate"]:
        assert df[col].between(0, 1).all(), f"{col} not normalised into [0,1]"

    # P-matrix columns were expanded correctly
    for i in range(16):
        assert f"P_{i}" in df.columns
    assert "asymm" in df.columns
    assert "P_matrix" not in df.columns

    # outlier trimming removes some rows from the extremes
    assert len(df) <= 50


def test_train_test_split_is_deterministic_and_disjoint(tmp_path):
    csv_path = tmp_path / "orthog_B_0.csv"
    _make_synthetic_instance_csv(csv_path, n_rows=50)
    df, _feature_cols, _min_max_df = build_preprocessed_dataset(str(tmp_path))

    train_a, test_a = train_test_split_like_original(df, train_frac=0.8, random_seed=42)
    train_b, test_b = train_test_split_like_original(df, train_frac=0.8, random_seed=42)

    assert len(train_a) + len(test_a) == len(df)
    assert set(train_a["exp_code"]).isdisjoint(set(test_a["exp_code"]))
    pd.testing.assert_frame_equal(train_a, train_b)
    pd.testing.assert_frame_equal(test_a, test_b)


def test_rate_is_rescaled_like_rya_net_dataset_init_B(tmp_path):
    """RYA's net_dataset_init_B.py (used to train EXP002): rate <- 2*rate/(1/n_X + 1/n_Y)."""
    csv_path = tmp_path / "orthog_B_0.csv"
    _make_synthetic_instance_csv(csv_path, n_rows=50)
    raw = pd.read_csv(csv_path)
    expected = raw.apply(lambda row: 2 * row["rate"] / (1 / row["n_X"] + 1 / row["n_Y"]), axis=1)

    _df, _cols, mm_rya = build_preprocessed_dataset(str(tmp_path))
    _df, _cols, mm_nb = build_preprocessed_dataset(str(tmp_path), normalise_rate=False)
    rate_rya = mm_rya.set_index("column").loc["rate"]
    rate_nb = mm_nb.set_index("column").loc["rate"]
    assert np.isclose(rate_rya["min"], expected.min()) and np.isclose(rate_rya["max"], expected.max())
    assert np.isclose(rate_nb["max"], raw["rate"].max())
