"""Score Pre/Post response counts with the trained ONNX network.

Faithful port of the notebook's Section 2 (``feature_calculation_portal_script.py`` /
``predict_d_hat``), wrapped in a class so the ONNX session and metadata are loaded once and
reused across many predictions (single example, or bulk RY25 scoring).
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import onnxruntime as ort
import pandas as pd

from .config import ProjectPaths
from .features import extract_features
from .reliability import load_p_matrices


class UnreliabilityPredictor:
    """Loads the EXP002 ONNX model + its training metadata and scores response counts.

    Returns (shape, rate) of the Gamma distribution EXP002 predicts for RYA's test statistic
    d-hat between two groups of respondents (Pre = x, Post = y) on one item. Exactly the
    portal-script path: features -> min-max normalise -> ONNX -> reverse min-max.
    """

    def __init__(self, paths: ProjectPaths):
        paths.require("onnx_path", "metadata_path")
        self.paths = paths
        with open(paths.metadata_path) as f:
            self.metadata: dict = json.load(f)

        min_max_df = pd.read_json(io.StringIO(self.metadata["min_max_df"]))
        self._mins = dict(zip(min_max_df["column"], min_max_df["min"]))
        self._maxs = dict(zip(min_max_df["column"], min_max_df["max"]))

        self.p_matrices: Dict[str, np.ndarray] = load_p_matrices(paths.pmatrices_path)

        self.session = ort.InferenceSession(paths.onnx_path)
        self._input_name = self.session.get_inputs()[0].name
        # The original notebook/portal script fed the ONNX session a plain rank-1 vector
        # (no batch dimension). Some exports require rank-2 (batch, features) instead -- detect
        # which one this model expects once, rather than assuming and failing on every call.
        self._expects_batch_dim = len(self.session.get_inputs()[0].shape) == 2

    def describe(self) -> str:
        m = self.metadata
        return (
            f"Model: {m['n_layers']} layers, width {m['width']}, "
            f"{len(m['feature_labels'])} features, P_transform='{m['P_transform']}'"
        )

    def _normalise(self, value: float, col: str) -> float:
        return (value - self._mins[col]) / (self._maxs[col] - self._mins[col])

    def _reverse_normalise(self, value: float, col: str) -> float:
        # float() first: RYA's reverse_normalise works in float64. Without it, NumPy 2 keeps the
        # ONNX float32 output as float32 and the result drifts from RYA's by ~1e-7.
        return (self._maxs[col] - self._mins[col]) * float(value) + self._mins[col]

    def _run(self, vec: np.ndarray) -> np.ndarray:
        """Run the session with ``vec`` shaped to whatever this model's input rank expects."""
        model_input = vec[np.newaxis, :] if self._expects_batch_dim else vec
        raw_out = self.session.run(None, {self._input_name: model_input})[0]
        return raw_out[0] if self._expects_batch_dim else raw_out

    def predict(self, x: np.ndarray, y: np.ndarray, question: str) -> Tuple[float, float]:
        """Return (shape_hat, rate_hat) for one item's Pre (x) / Post (y) response counts."""
        feats = extract_features(x, y, question, self.p_matrices).reindex(self.metadata["feature_labels"])
        vec = np.array(
            [self._normalise(feats[c], c) for c in self.metadata["feature_labels"]], dtype=np.float32
        )
        raw_out = self._run(vec)
        shape_hat = self._reverse_normalise(raw_out[0], "shape")
        rate_hat = self._reverse_normalise(raw_out[1], "rate")
        return float(shape_hat), float(rate_hat)

    def predict_raw_features(self, feats: pd.Series) -> Tuple[float, float]:
        """Same as ``predict`` but starting from an already-computed (unnormalised) feature Series.

        Used by the model-comparison step, where features come from the training pipeline
        rather than raw response counts.
        """
        vec = np.array(
            [self._normalise(feats[c], c) for c in self.metadata["feature_labels"]], dtype=np.float32
        )
        raw_out = self._run(vec)
        shape_hat = self._reverse_normalise(raw_out[0], "shape")
        rate_hat = self._reverse_normalise(raw_out[1], "rate")
        return float(shape_hat), float(rate_hat)
