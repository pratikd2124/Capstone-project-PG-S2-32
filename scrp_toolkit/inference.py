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
from typing import Dict, Sequence, Tuple

import numpy as np
import onnxruntime as ort
import pandas as pd

from .config import ProjectPaths
from .features import extract_features, extract_features_batch
from .reliability import load_p_matrices


def batched_onnx_session(onnx_path: str) -> ort.InferenceSession:
    """The same ONNX model, but accepting a batch of rows (N x 40) instead of one row.

    RYA's file only takes a single row of 40 numbers, so every item costs a separate call.
    Only the declared input/output shapes change here; the weights and operations are
    untouched, and the outputs are identical (tests/test_inference_batch.py).
    """
    import onnx   # only needed here; reading RYA's file as-is doesn't need it

    model = onnx.load(onnx_path)
    for value in (model.graph.input[0], model.graph.output[0]):
        dims = value.type.tensor_type.shape.dim
        if len(dims) == 1:
            last = dims[0].dim_value
            del dims[:]
            dims.add().dim_param = "batch"
            dims.add().dim_value = last
        else:
            dims[0].dim_param = "batch"
    return ort.InferenceSession(model.SerializeToString())


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
        self._batch_session = None   # built on first predict_batch call

        # The min/max in feature order, for scaling a whole table at once.
        labels = self.metadata["feature_labels"]
        self._min_vec = np.array([self._mins[c] for c in labels], dtype=float)
        self._max_vec = np.array([self._maxs[c] for c in labels], dtype=float)

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

    def predict_batch(self, x: np.ndarray, y: np.ndarray, questions: Sequence[str],
                      chunk_size: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
        """``predict`` for many items at once: arrays of shape and rate, one per row.

        x, y       (N, 4) answer counts, before and after
        questions  N item codes

        Same answers as calling ``predict`` row by row (tests/test_inference_batch.py), but the
        features are worked out on whole columns and the network runs ``chunk_size`` rows per
        call. Use this whenever there's more than a handful of items to score.
        """
        labels = self.metadata["feature_labels"]
        feats = extract_features_batch(x, y, questions, self.p_matrices)[labels].to_numpy()
        scaled = ((feats - self._min_vec) / (self._max_vec - self._min_vec)).astype(np.float32)

        if self._batch_session is None:
            self._batch_session = batched_onnx_session(self.paths.onnx_path)
        in_name = self._batch_session.get_inputs()[0].name
        raw_out = np.concatenate([
            self._batch_session.run(None, {in_name: scaled[i:i + chunk_size]})[0]
            for i in range(0, len(scaled), chunk_size)
        ]) if len(scaled) else np.empty((0, 2), dtype=np.float32)

        # As in _reverse_normalise: back to 64-bit first, then undo the scaling.
        raw_out = raw_out.astype(np.float64)
        shape_hat = (self._maxs["shape"] - self._mins["shape"]) * raw_out[:, 0] + self._mins["shape"]
        rate_hat = (self._maxs["rate"] - self._mins["rate"]) * raw_out[:, 1] + self._mins["rate"]
        return shape_hat, rate_hat
