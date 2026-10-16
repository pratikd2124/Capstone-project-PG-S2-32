from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def rya_certainty(P_matrix: np.ndarray) -> float:
    P = np.asarray(P_matrix, dtype=float)
    if P.shape != (4, 4):
        raise ValueError("Expected a 4x4 P matrix.")
    return float(np.trace(P))


def rya_asymmetry(P_matrix: np.ndarray) -> float:
    P = np.asarray(P_matrix, dtype=float)
    if P.shape != (4, 4):
        raise ValueError("Expected a 4x4 P matrix.")

    sq = []
    for i in range(4):
        for j in range(i + 1, 4):
            sq.append((P[i, j] - P[j, i]) ** 2)

    return float(np.mean(sq))


def detector_asymmetry(P_matrix: np.ndarray) -> float:
    P = np.asarray(P_matrix, dtype=float)
    return float(np.mean((P - P.T) ** 2))


def entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log(p + eps)).sum())


def is_degenerate_counts(counts: Sequence[float]) -> bool:
    arr = np.asarray(counts, dtype=float)
    if arr.shape != (4,):
        raise ValueError("Expected 4 category counts.")
    return int(np.sum(arr > 0)) == 1


def extract_screening_features(
    X: Sequence[float],
    Y: Sequence[float],
    n_x: float,
    n_y: float,
    P_matrix: np.ndarray,
    sparse_threshold: float = 0.02,
) -> Dict[str, float]:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    P = np.asarray(P_matrix, dtype=float)

    if X.shape != (4,) or Y.shape != (4,):
        raise ValueError("O1 current scope expects 4-category X and Y.")
    if P.shape != (4, 4):
        raise ValueError("O1 current scope expects a 4x4 unreliability matrix.")
    if n_x <= 0 or n_y <= 0:
        raise ValueError("Group sizes must be positive.")

    group_size_ratio = max(n_x, n_y) / min(n_x, n_y)
    active_X = int(np.sum(X > 0))
    active_Y = int(np.sum(Y > 0))
    diag = np.diag(P)
    row_sums = P.sum(axis=1)

    feats: Dict[str, float] = {
        "n_X": float(n_x),
        "n_Y": float(n_y),
        "log_n_X": float(np.log1p(n_x)),
        "log_n_Y": float(np.log1p(n_y)),
        "group_size_ratio": float(group_size_ratio),
        "log_group_size_ratio": float(np.log(group_size_ratio)),
        "euclid_dist_XY": float(np.round(np.linalg.norm(X - Y), 4)),
        "min_prop_X": float(X.min()),
        "min_prop_Y": float(Y.min()),
        "max_prop_X": float(X.max()),
        "max_prop_Y": float(Y.max()),
        "active_categories_X": active_X,
        "active_categories_Y": active_Y,
        "degenerate_X": int(active_X == 1),
        "degenerate_Y": int(active_Y == 1),
        "sparse_count_X": int(np.sum(X <= sparse_threshold)),
        "sparse_count_Y": int(np.sum(Y <= sparse_threshold)),
        "entropy_X": entropy(X),
        "entropy_Y": entropy(Y),
        "mean_diag_P": float(diag.mean()),
        "min_diag_P": float(diag.min()),
        "max_diag_P": float(diag.max()),
        "mean_offdiag_mass_P": float(np.mean(1.0 - diag)),
        "asymmetry_P": detector_asymmetry(P),
        "max_row_sum_error_P": float(np.max(np.abs(row_sums - 1.0))),
        "all_rows_diagonal_mode_P": int(np.all(diag >= P.max(axis=1))),
    }

    for i, value in enumerate(X, start=1):
        feats[f"X_{i}"] = float(value)
    for i, value in enumerate(Y, start=1):
        feats[f"Y_{i}"] = float(value)
    for i, value in enumerate(np.abs(X - Y), start=1):
        feats[f"absdiff_{i}"] = float(value)
    for i in range(4):
        for j in range(4):
            feats[f"P_{i+1}{j+1}"] = float(P[i, j])

    return feats


def calculate_P_structure_descriptors(P_matrix: np.ndarray) -> Dict[str, float]:
    P = np.asarray(P_matrix, dtype=float)
    if P.shape != (4, 4):
        raise ValueError("Expected a 4x4 P matrix.")

    diag = np.diag(P)
    row_sums = P.sum(axis=1)

    return {
        "certainty_P": rya_certainty(P),
        "rya_asymmetry_P": rya_asymmetry(P),
        "min_diag_P": float(diag.min()),
        "max_diag_P": float(diag.max()),
        "mean_diag_P": float(diag.mean()),
        "max_row_sum_error_P": float(np.max(np.abs(row_sums - 1.0))),
        "all_rows_diagonal_mode_P": int(np.all(diag >= P.max(axis=1))),
    }


@dataclass
class O1ScreeningGate:
    detector_model_path: str | Path
    detector_metadata_path: str | Path
    screening_reference_csv: Optional[str | Path] = None
    row_sum_tolerance: float = 0.020001
    distance_reference_sample_size: int = 5000
    random_state: int = 2026
    enable_degeneracy_safeguard: bool = True

    def __post_init__(self) -> None:
        self.detector_model_path = Path(self.detector_model_path)
        self.detector_metadata_path = Path(self.detector_metadata_path)

        self.detector = joblib.load(self.detector_model_path)

        with open(self.detector_metadata_path, "r", encoding="utf-8") as f:
            self.detector_metadata = json.load(f)

        self.feature_order = list(self.detector_metadata["feature_order"])
        self.threshold = float(self.detector_metadata["threshold"])

        self._p_scaler = None
        self._p_nn = None
        self._p_reference_df = None
        self._training_nearest_distances = None

        if self.screening_reference_csv is not None:
            self._prepare_training_space_reference(Path(self.screening_reference_csv))

    def _prepare_training_space_reference(self, csv_path: Path) -> None:
        df = pd.read_csv(csv_path)
        self._p_feature_cols = [f"P_{i}{j}" for i in range(1, 5) for j in range(1, 5)]

        missing = [c for c in self._p_feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Training-space reference is missing columns: {missing}")

        P_train = df[self._p_feature_cols].astype(float).copy()

        self._p_scaler = StandardScaler()
        P_scaled = self._p_scaler.fit_transform(P_train)

        self._p_nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
        self._p_nn.fit(P_scaled)

        self._p_reference_df = df

        rng = np.random.default_rng(self.random_state)
        n = min(self.distance_reference_sample_size, len(P_scaled))
        idx = rng.choice(len(P_scaled), size=n, replace=False)

        dists, _ = self._p_nn.kneighbors(P_scaled[idx], n_neighbors=2)
        self._training_nearest_distances = dists[:, 1]

    def _validate_inputs(self, x, y, P, *, pair_name):
        warnings: List[str] = []

        if x.shape != (4,) or y.shape != (4,):
            return {"route": "invalid_input", "reason": f"{pair_name}: x and y must each contain 4 category counts."}, warnings
        if P.shape != (4, 4):
            return {"route": "invalid_input", "reason": "P_matrix must be 4x4."}, warnings
        if np.isnan(x).any() or np.isnan(y).any() or np.isnan(P).any():
            return {"route": "invalid_input", "reason": f"{pair_name}: input contains NaN values."}, warnings
        if np.isinf(x).any() or np.isinf(y).any() or np.isinf(P).any():
            return {"route": "invalid_input", "reason": f"{pair_name}: input contains infinite values."}, warnings
        if np.any(x < 0) or np.any(y < 0):
            return {"route": "invalid_input", "reason": f"{pair_name}: negative category counts are not allowed."}, warnings
        if np.any(P < 0):
            return {"route": "invalid_input", "reason": "P_matrix contains negative probabilities."}, warnings
        if float(x.sum()) <= 0 or float(y.sum()) <= 0:
            return {"route": "invalid_input", "reason": f"{pair_name}: both groups must contain at least one observation."}, warnings

        row_sums = P.sum(axis=1)
        max_row_sum_error = float(np.max(np.abs(row_sums - 1.0)))

        if max_row_sum_error > self.row_sum_tolerance:
            return {
                "route": "invalid_input",
                "reason": f"P_matrix row sums deviate too far from 1. Maximum error = {max_row_sum_error:.6f}",
            }, warnings

        if max_row_sum_error > 1e-6:
            warnings.append(
                "P_matrix row sums contain small rounding deviations "
                f"(maximum error = {max_row_sum_error:.6f}). Original matrix retained without renormalisation."
            )

        return None, warnings

    def _training_space_diagnostic(self, P):
        if self._p_scaler is None or self._p_nn is None:
            return None

        vec = np.array([P[i, j] for i in range(4) for j in range(4)], dtype=float).reshape(1, -1)
        df = pd.DataFrame(vec, columns=self._p_feature_cols)
        scaled = self._p_scaler.transform(df)

        dist, idx = self._p_nn.kneighbors(scaled, n_neighbors=1)
        distance = float(dist[0, 0])
        nearest_idx = int(idx[0, 0])

        percentile = float(np.mean(self._training_nearest_distances <= distance) * 100.0)

        nearest_question = None
        if "P_matrix_question" in self._p_reference_df.columns:
            nearest_question = str(self._p_reference_df.iloc[nearest_idx]["P_matrix_question"])

        return {
            "nearest_distance": distance,
            "empirical_percentile": percentile,
            "nearest_question": nearest_question,
            "note": "Diagnostic only. No hard P-space routing threshold is applied.",
        }

    def _classifier_result(self, nn_x, nn_y, P):
        n_x = float(nn_x.sum())
        n_y = float(nn_y.sum())
        X = nn_x / n_x
        Y = nn_y / n_y

        feats = extract_screening_features(X=X, Y=Y, n_x=n_x, n_y=n_y, P_matrix=P)

        feat_series = pd.Series(feats, dtype=float).reindex(self.feature_order)
        if feat_series.isna().any():
            missing = feat_series[feat_series.isna()].index.tolist()
            raise ValueError(f"Missing detector features: {missing}")

        detector_input = pd.DataFrame([feat_series.values], columns=self.feature_order)
        probability = float(self.detector.predict_proba(detector_input)[0, 1])

        return {
            "incompatibility_probability": probability,
            "threshold": self.threshold,
            "flagged": bool(probability >= self.threshold),
            "pair_used": "nn_pair",
        }

    def screen_case(
        self,
        observed_x: Sequence[float],
        observed_y: Sequence[float],
        P_matrix: Sequence[Sequence[float]],
        question: Optional[str] = None,
        nn_x: Optional[Sequence[float]] = None,
        nn_y: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:

        observed_x_arr = np.asarray(observed_x, dtype=float)
        observed_y_arr = np.asarray(observed_y, dtype=float)
        P = np.asarray(P_matrix, dtype=float)

        invalid, validation_warnings = self._validate_inputs(
            observed_x_arr, observed_y_arr, P, pair_name="observed_pair"
        )
        if invalid is not None:
            return {"question": question, **invalid, "validation_warnings": validation_warnings}

        using_observed_as_nn_pair = (nn_x is None and nn_y is None)

        if (nn_x is None) != (nn_y is None):
            return {
                "question": question,
                "route": "invalid_input",
                "reason": "nn_x and nn_y must either both be supplied or both be omitted.",
                "validation_warnings": validation_warnings,
            }

        if using_observed_as_nn_pair:
            nn_x_arr = observed_x_arr.copy()
            nn_y_arr = observed_y_arr.copy()
        else:
            nn_x_arr = np.asarray(nn_x, dtype=float)
            nn_y_arr = np.asarray(nn_y, dtype=float)

            invalid_nn, nn_warnings = self._validate_inputs(
                nn_x_arr, nn_y_arr, P, pair_name="nn_pair"
            )
            validation_warnings.extend(nn_warnings)

            if invalid_nn is not None:
                return {"question": question, **invalid_nn, "validation_warnings": validation_warnings}

        observed_degenerate_x = is_degenerate_counts(observed_x_arr)
        observed_degenerate_y = is_degenerate_counts(observed_y_arr)
        observed_degenerate_any = observed_degenerate_x or observed_degenerate_y

        p_descriptors = calculate_P_structure_descriptors(P)
        training_space = self._training_space_diagnostic(P)
        classifier = self._classifier_result(nn_x_arr, nn_y_arr, P)

        if self.enable_degeneracy_safeguard and observed_degenerate_any:
            route = "fallback_required"
            reason = "original_pair_degenerate"
        elif classifier["flagged"]:
            route = "fallback_required"
            reason = "compatibility_classifier"
        else:
            route = "pass_to_gamma_nn"
            reason = "screening_passed"

        return {
            "question": question,
            "route": route,
            "reason": reason,
            "observed_group_sizes": {
                "n_X": int(observed_x_arr.sum()),
                "n_Y": int(observed_y_arr.sum()),
                "group_size_ratio": float(max(observed_x_arr.sum(), observed_y_arr.sum()) / min(observed_x_arr.sum(), observed_y_arr.sum())),
            },
            "nn_group_sizes": {
                "n_X": int(nn_x_arr.sum()),
                "n_Y": int(nn_y_arr.sum()),
                "group_size_ratio": float(max(nn_x_arr.sum(), nn_y_arr.sum()) / min(nn_x_arr.sum(), nn_y_arr.sum())),
            },
            "safeguard": {
                "enabled": self.enable_degeneracy_safeguard,
                "observed_degenerate_X": observed_degenerate_x,
                "observed_degenerate_Y": observed_degenerate_y,
                "original_pair_degenerate": observed_degenerate_any,
            },
            "classifier": classifier,
            "matrix_validation": {
                "valid": True,
                "warnings": validation_warnings,
                "certainty_P": p_descriptors["certainty_P"],
                "rya_asymmetry_P": p_descriptors["rya_asymmetry_P"],
                "max_row_sum_error_P": p_descriptors["max_row_sum_error_P"],
            },
            "training_space": training_space,
            "interface": {
                "nn_pair_source": "observed_pair_default" if using_observed_as_nn_pair else "explicit_nn_pair",
                "boundary_transform_implemented_here": False,
            },
        }
