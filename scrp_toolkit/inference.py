"""Running RYA's trained model (EXP002) on answer counts.

For one item and two groups of students, the steps are:

    1. build the 40 inputs (features.py)
    2. squeeze each into 0-1 using the min/max values stored in EXP002.json
    3. run the trained network in EXP002_nn_optimal_epoch.onnx
    4. turn its two outputs back into real numbers: shape and rate

Shape and rate describe a Gamma distribution: RYA's estimate of how big the difference
statistic d-hat between the two groups would be from students' answer noise alone.
shape / rate is its average. A real before/after change has to clearly beat it.

These are the same steps as normalise() / reverse_normalise() in RYA's
feature_calculation_portal_script.py, and the results are identical (walkthrough Step 5).
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
    """RYA's trained model, ready to score items.

    Loading the model takes a moment, so do it once and call ``predict`` as often as you like.
    """

    def __init__(self, paths: ProjectPaths):
        paths.require("onnx_path", "metadata_path")
        self.paths = paths

        # EXP002.json holds the model's settings and the min/max of every input and output
        # seen during training. We need the same min/max to scale things the same way.
        with open(paths.metadata_path) as f:
            self.metadata: dict = json.load(f)
        min_max_df = pd.read_json(io.StringIO(self.metadata["min_max_df"]))
        self._mins = dict(zip(min_max_df["column"], min_max_df["min"]))
        self._maxs = dict(zip(min_max_df["column"], min_max_df["max"]))

        self.p_matrices: Dict[str, np.ndarray] = load_p_matrices(paths.pmatrices_path)

        self.session = ort.InferenceSession(paths.onnx_path)
        self._input_name = self.session.get_inputs()[0].name
        # RYA exported EXP002 to take a plain list of 40 numbers. Some re-exports expect a
        # batch of lists instead, so check once which kind this file wants.
        self._expects_batch_dim = len(self.session.get_inputs()[0].shape) == 2

    def describe(self) -> str:
        """A one-line summary of the model, e.g. for printing at the top of a report."""
        m = self.metadata
        return (
            f"Model: {m['n_layers']} layers, width {m['width']}, "
            f"{len(m['feature_labels'])} features, P_transform='{m['P_transform']}'"
        )

    def _normalise(self, value: float, col: str) -> float:
        """Scale a raw value into 0-1 with the training min/max (RYA's normalise)."""
        return (value - self._mins[col]) / (self._maxs[col] - self._mins[col])

    def _reverse_normalise(self, value: float, col: str) -> float:
        """Undo the scaling on a model output (RYA's reverse_normalise)."""
        # float() first: RYA works in 64-bit. Without it, the model's 32-bit output stays
        # 32-bit and the answer drifts from RYA's by about 0.0000001.
        return (self._maxs[col] - self._mins[col]) * float(value) + self._mins[col]

    def _run(self, vec: np.ndarray) -> np.ndarray:
        """Run the network on one scaled input vector and return its two raw outputs."""
        model_input = vec[np.newaxis, :] if self._expects_batch_dim else vec
        raw_out = self.session.run(None, {self._input_name: model_input})[0]
        return raw_out[0] if self._expects_batch_dim else raw_out

    def predict(self, x: np.ndarray, y: np.ndarray, question: str) -> Tuple[float, float]:
        """(shape, rate) for one item, from the before (x) and after (y) answer counts."""
        feats = extract_features(x, y, question, self.p_matrices).reindex(self.metadata["feature_labels"])
        vec = np.array(
            [self._normalise(feats[c], c) for c in self.metadata["feature_labels"]], dtype=np.float32
        )
        raw_out = self._run(vec)
        shape_hat = self._reverse_normalise(raw_out[0], "shape")
        rate_hat = self._reverse_normalise(raw_out[1], "rate")
        return float(shape_hat), float(rate_hat)

    def predict_raw_features(self, feats: pd.Series) -> Tuple[float, float]:
        """Like ``predict``, but starting from the 40 features rather than answer counts.

        Used when the features already exist, e.g. rows of the training data.
        """
        vec = np.array(
            [self._normalise(feats[c], c) for c in self.metadata["feature_labels"]], dtype=np.float32
        )
        raw_out = self._run(vec)
        shape_hat = self._reverse_normalise(raw_out[0], "shape")
        rate_hat = self._reverse_normalise(raw_out[1], "rate")
        return float(shape_hat), float(rate_hat)
