"""O4 (Murat): faster and smaller EXP002, without changing RYA's baseline code.

Two changes on top of the baseline ``UnreliabilityPredictor``:

* **Batching.** The 40 inputs are built for many items at once on whole columns
  (``extract_features_batch``), and the network runs on a batch of rows instead of one row
  per call. Same answers as the baseline, item by item (tests/test_o4_efficiency.py).
* **fp16 weights (opt-in).** EXP002's weights stored as 16-bit: the model file halves
  (4.8 MB -> 2.4 MB, for the browser). Outputs move by ~1e-4; the default "fp32" is RYA's
  model exactly.

RYA's model file is never changed; every variant is built in memory.
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import onnxruntime as ort
import pandas as pd

from scrp_toolkit.config import ProjectPaths
from scrp_toolkit.inference import UnreliabilityPredictor
from scrp_toolkit.reliability import asymm_calc

WEIGHT_OPTIONS = ("fp32", "fp16")


# ---------------------------------------------------------------------------------------------
# The 40 inputs for many items at once
# ---------------------------------------------------------------------------------------------

def _rowdot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """np.dot of each row of ``a`` with the same row of ``b``.

    Written as a stack of tiny matrix products so numpy uses the same BLAS dot routine as
    ``np.dot`` on one row: bit-for-bit the baseline's ``moment_calculator`` on most CPUs, and at
    most last-bit differences (<= 9e-16) on others.
    """
    a, b = np.broadcast_arrays(a, b)
    return np.matmul(a[:, None, :], b[:, :, None])[:, 0, 0]


def extract_features_batch(x: np.ndarray, y: np.ndarray, questions, p_matrices: Dict[str, np.ndarray]) -> pd.DataFrame:
    """The baseline's ``extract_features`` for many items at once: one row per item, same 40 columns.

    x, y       (N, 4) answer counts, before and after
    questions  N item codes
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


# ---------------------------------------------------------------------------------------------
# EXP002 that takes batches, optionally with fp16 weights
# ---------------------------------------------------------------------------------------------

def _open_batch_dim(model) -> None:
    """Let the model take a batch of rows (N x 40) instead of one row of 40."""
    for value in (model.graph.input[0], model.graph.output[0]):
        dims = value.type.tensor_type.shape.dim
        if len(dims) == 1:
            last = dims[0].dim_value
            del dims[:]
            dims.add().dim_param = "batch"
            dims.add().dim_value = last
        else:
            dims[0].dim_param = "batch"


def _store_weights_fp16(model) -> None:
    """Store every 32-bit weight as 16-bit, with a Cast back to 32-bit inside the model."""
    from onnx import TensorProto, helper, numpy_helper

    g = model.graph
    float_inits = [i for i in g.initializer if i.data_type == TensorProto.FLOAT]
    names = {i.name for i in float_inits}
    halves = [numpy_helper.from_array(numpy_helper.to_array(i).astype(np.float16), i.name + "_fp16")
              for i in float_inits]
    casts = [helper.make_node("Cast", [i.name + "_fp16"], [i.name], to=TensorProto.FLOAT) for i in float_inits]

    keep = [i for i in g.initializer if i.name not in names]
    del g.initializer[:]
    g.initializer.extend(keep + halves)
    # Older exports also list the weights as graph inputs; now the Cast nodes produce them.
    inputs = [v for v in g.input if v.name not in names]
    del g.input[:]
    g.input.extend(inputs)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(casts + nodes)


def onnx_model_bytes(onnx_path: str, batched: bool = False, weights: str = "fp32") -> bytes:
    """RYA's ONNX model, optionally taking batches and/or with fp16 weights. Built in memory."""
    if weights not in WEIGHT_OPTIONS:
        raise ValueError(f"weights must be one of {WEIGHT_OPTIONS}, got {weights!r}")
    import onnx

    model = onnx.load(onnx_path)
    if batched:
        _open_batch_dim(model)
    if weights == "fp16":
        _store_weights_fp16(model)
    return model.SerializeToString()


def batched_onnx_session(onnx_path: str, weights: str = "fp32") -> ort.InferenceSession:
    """EXP002 accepting a batch of rows (N x 40). Only the declared shapes change."""
    return ort.InferenceSession(onnx_model_bytes(onnx_path, batched=True, weights=weights))


class EfficientPredictor(UnreliabilityPredictor):
    """The baseline predictor plus ``predict_batch`` and optional fp16 weights.

    ``predict`` (one item) behaves exactly as the baseline; ``predict_batch`` gives the same
    answers for many items at once, much faster.
    """

    def __init__(self, paths: ProjectPaths, weights: str = "fp32"):
        if weights not in WEIGHT_OPTIONS:
            raise ValueError(f"weights must be one of {WEIGHT_OPTIONS}, got {weights!r}")
        super().__init__(paths)
        self.weights = weights
        if weights == "fp16":
            self.session = ort.InferenceSession(onnx_model_bytes(paths.onnx_path, weights="fp16"))
            self._input_name = self.session.get_inputs()[0].name
            self._expects_batch_dim = len(self.session.get_inputs()[0].shape) == 2
        self._batch_session = None   # built on the first predict_batch call

        labels = self.metadata["feature_labels"]
        self._min_vec = np.array([self._mins[c] for c in labels], dtype=float)
        self._max_vec = np.array([self._maxs[c] for c in labels], dtype=float)

    def describe(self) -> str:
        return super().describe() + (", weights stored as fp16" if self.weights == "fp16" else "")

    def predict_batch(self, x: np.ndarray, y: np.ndarray, questions: Sequence[str],
                      chunk_size: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
        """``predict`` for many items at once: arrays of shape and rate, one per row."""
        labels = self.metadata["feature_labels"]
        feats = extract_features_batch(x, y, questions, self.p_matrices)[labels].to_numpy()
        scaled = ((feats - self._min_vec) / (self._max_vec - self._min_vec)).astype(np.float32)

        if self._batch_session is None:
            self._batch_session = batched_onnx_session(self.paths.onnx_path, weights=self.weights)
        in_name = self._batch_session.get_inputs()[0].name
        raw_out = np.concatenate([
            self._batch_session.run(None, {in_name: scaled[i:i + chunk_size]})[0]
            for i in range(0, len(scaled), chunk_size)
        ]) if len(scaled) else np.empty((0, 2), dtype=np.float32)

        # As in the baseline's _reverse_normalise: back to 64-bit first, then undo the scaling.
        raw_out = raw_out.astype(np.float64)
        shape_hat = (self._maxs["shape"] - self._mins["shape"]) * raw_out[:, 0] + self._mins["shape"]
        rate_hat = (self._maxs["rate"] - self._mins["rate"]) * raw_out[:, 1] + self._mins["rate"]
        return shape_hat, rate_hat
