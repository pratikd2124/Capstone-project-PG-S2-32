"""Standalone helpers for Pratik's robustness and Bayesian analysis.

This module deliberately does NOT import ``scrp_toolkit``.  It reads the
original RYA assets directly:

- ``unreliability-matrices.json``
- ``EXP002.json``
- ``EXP002_nn_optimal_epoch.onnx``
- ``B_matrix.txt``

The feature calculations reproduce the formulas in RYA's
``feature_calculation_portal_script.py``.  The code is intended for an
independent branch and does not modify the baseline, model, or customer data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import io
import json
import os
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "Analysis" / "results"
FIGURES_DIR = ROOT / "Analysis" / "figures"

# Same worked example used by RYA's portal feature script / walkthrough.
DEFAULT_ITEM = "ry9"
DEFAULT_PRE = np.array([40, 52, 7, 14], dtype=float)
DEFAULT_POST = np.array([9, 17, 54, 12], dtype=float)


@dataclass(frozen=True)
class AssetPaths:
    root: Path
    matrices: Path
    metadata: Path
    model: Path
    b_matrix: Path


@dataclass
class StandaloneContext:
    paths: AssetPaths
    p_matrices: dict[str, np.ndarray]
    metadata: dict
    min_max: pd.DataFrame
    b_matrix: np.ndarray
    session: object | None = None

    @property
    def feature_labels(self) -> list[str]:
        return list(self.metadata["feature_labels"])


@dataclass
class Prediction:
    shape: float
    rate: float
    gamma_mean: float
    gamma_q95: float
    features: pd.Series
    normalised_features: pd.Series
    outside_training_range: list[str]


def ensure_output_dirs() -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR, FIGURES_DIR


def _find_unique(root: Path, filename: str) -> Path:
    hits = sorted(p for p in root.rglob(filename) if p.is_file())
    if not hits:
        raise FileNotFoundError(
            f"Could not find {filename!r} under {root}. "
            "Extract/copy the original RYA assets into the repo or pass --assets-root."
        )
    if len(hits) > 1:
        # Prefer paths containing the expected production model folder/name.
        preferred = [p for p in hits if "final_network" in str(p).lower()]
        if len(preferred) == 1:
            return preferred[0]
        print(f"WARNING: found multiple {filename}; using {hits[0]}")
    return hits[0]


def discover_assets(root: str | Path | None = None) -> AssetPaths:
    """Locate original RYA files recursively, without relying on toolkit paths."""

    if root is None:
        root = os.environ.get("RYA_ASSETS_DIR", ROOT)
    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Assets root does not exist: {root}")

    return AssetPaths(
        root=root,
        matrices=_find_unique(root, "unreliability-matrices.json"),
        metadata=_find_unique(root, "EXP002.json"),
        model=_find_unique(root, "EXP002_nn_optimal_epoch.onnx"),
        b_matrix=_find_unique(root, "B_matrix.txt"),
    )


def load_context(root: str | Path | None = None, require_model: bool = False) -> StandaloneContext:
    paths = discover_assets(root)

    with paths.matrices.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    p_matrices = {k: np.asarray(v, dtype=float) for k, v in raw.items()}

    with paths.metadata.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    min_max = pd.read_json(io.StringIO(metadata["min_max_df"])).set_index("column")
    b_matrix = np.loadtxt(paths.b_matrix, dtype=float)
    if b_matrix.shape != (3, 4):
        raise ValueError(f"Expected B_matrix.txt to contain a 3x4 matrix; got {b_matrix.shape}")

    session = None
    if require_model:
        session = make_onnx_session(paths.model)

    return StandaloneContext(paths, p_matrices, metadata, min_max, b_matrix, session)


def make_onnx_session(model_path: Path):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required for EXP002 inference. Run: pip install -r requirements.txt"
        ) from exc
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


# ---------------------------------------------------------------------------
# Original feature calculations (standalone reproduction of RYA formulas)
# ---------------------------------------------------------------------------

def moment_calculator(prob_dist: Iterable[float], name: str) -> pd.Series:
    p = np.asarray(list(prob_dist), dtype=float)
    rv = np.arange(1, len(p) + 1, dtype=float)
    mean = float(np.dot(rv, p))
    stdev = float(np.sqrt(np.dot((rv - mean) ** 2, p)))

    if stdev == 0.0:
        skewness = np.nan
        kurtosis = np.nan
    else:
        skewness = float(np.dot((rv - mean) ** 3, p) / stdev**3)
        kurtosis = float(np.dot((rv - mean) ** 4, p) / stdev**4)

    return pd.Series(
        {
            f"mean_{name}": mean,
            f"stdev_{name}": stdev,
            f"skewness_{name}": skewness,
            f"kurtosis_{name}": kurtosis,
        }
    )


def asymm_calc(matrix: np.ndarray) -> float:
    p = np.asarray(matrix, dtype=float)
    if p.shape != (4, 4):
        raise ValueError(f"asymm_calc is defined here for 4x4 matrices; got {p.shape}")
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    return float(np.mean([(p[i, j] - p[j, i]) ** 2 for i, j in pairs]))


def extract_features(
    x: Iterable[float],
    y: Iterable[float],
    matrix: np.ndarray,
) -> pd.Series:
    """Build the 40 EXP002 inputs directly from RYA's published formulas."""

    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)
    p = np.asarray(matrix, dtype=float)

    if x.shape != (4,) or y.shape != (4,):
        raise ValueError("EXP002 requires four response counts for Pre and Post")
    if p.shape != (4, 4):
        raise ValueError(f"EXP002 requires a 4x4 unreliability matrix; got {p.shape}")
    if np.any(x < 0) or np.any(y < 0):
        raise ValueError("Response counts cannot be negative")

    n_x, n_y = float(x.sum()), float(y.sum())
    if n_x <= 0 or n_y <= 0:
        raise ValueError("Pre and Post counts must each sum to a positive value")

    X, Y = x / n_x, y / n_y
    X_minus_Y = np.abs(X - Y)
    euclid = float(np.round(np.sqrt(np.sum((X - Y) ** 2)), 4))

    parts = [
        pd.Series({f"X_{i+1}": X[i] for i in range(4)}),
        pd.Series({f"Y_{i+1}": Y[i] for i in range(4)}),
        pd.Series({f"XmY_{i+1}": X_minus_Y[i] for i in range(4)}),
        moment_calculator(X, "X"),
        moment_calculator(Y, "Y"),
        pd.Series({"euclid_dist_XY": euclid}),
        pd.Series({"n_X": n_x, "n_Y": n_y}),
        pd.Series({**{f"P_{i}": float(p.flat[i]) for i in range(16)}, "asymm": asymm_calc(p)}),
    ]
    return pd.concat(parts)


def normalise_features(features: pd.Series, ctx: StandaloneContext) -> pd.Series:
    labels = ctx.feature_labels
    f = pd.to_numeric(features.reindex(labels), errors="coerce")
    if f.isna().any():
        bad = f.index[f.isna()].tolist()
        raise ValueError(f"Non-finite/missing feature(s): {', '.join(bad)}")

    mins = ctx.min_max.loc[labels, "min"].astype(float)
    maxs = ctx.min_max.loc[labels, "max"].astype(float)
    width = maxs - mins
    if np.any(np.isclose(width.to_numpy(), 0.0)):
        bad = width.index[np.isclose(width.to_numpy(), 0.0)].tolist()
        raise ValueError(f"Zero min-max width for: {', '.join(bad)}")
    return (f - mins) / width


def _run_onnx(session, vector: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    vector = np.asarray(vector, dtype=np.float32)

    # The shipped model was exported from one feature vector. Most copies expect
    # shape (40,), but the fallback also handles a batch dimension.
    errors: list[str] = []
    for candidate in (vector, vector.reshape(1, -1)):
        try:
            output = session.run(None, {input_name: candidate})[0]
            return np.asarray(output, dtype=float).reshape(-1)
        except Exception as exc:  # pragma: no cover - depends on ONNX runtime/model build
            errors.append(str(exc))
    raise RuntimeError("Could not run EXP002 ONNX model: " + " | ".join(errors))


def predict_exp002(
    ctx: StandaloneContext,
    x: Iterable[float],
    y: Iterable[float],
    item: str,
    matrix: np.ndarray | None = None,
) -> Prediction:
    """Run the original EXP002 ONNX model directly, with no toolkit imports."""

    if item not in ctx.p_matrices:
        raise KeyError(f"Unknown survey item: {item}")
    p = np.asarray(ctx.p_matrices[item] if matrix is None else matrix, dtype=float)

    features = extract_features(x, y, p).reindex(ctx.feature_labels)
    norm = normalise_features(features, ctx)
    outside = norm.index[(norm < 0.0) | (norm > 1.0)].tolist()

    if ctx.session is None:
        ctx.session = make_onnx_session(ctx.paths.model)
    raw = _run_onnx(ctx.session, norm.to_numpy(dtype=np.float32))
    if raw.size < 2:
        raise ValueError(f"Unexpected EXP002 output shape: {raw.shape}")

    shape_min, shape_max = ctx.min_max.loc["shape", ["min", "max"]].astype(float)
    rate_min, rate_max = ctx.min_max.loc["rate", ["min", "max"]].astype(float)
    shape = float(raw[0] * (shape_max - shape_min) + shape_min)
    rate = float(raw[1] * (rate_max - rate_min) + rate_min)
    if shape <= 0 or rate <= 0 or not np.isfinite(shape + rate):
        raise ValueError(f"EXP002 returned invalid Gamma parameters: shape={shape}, rate={rate}")

    from scipy.stats import gamma as gamma_dist

    gamma_mean = float(shape / rate)
    gamma_q95 = float(gamma_dist.ppf(0.95, a=shape, scale=1.0 / rate))
    return Prediction(shape, rate, gamma_mean, gamma_q95, features, norm, outside)


# ---------------------------------------------------------------------------
# Sensitivity helpers
# ---------------------------------------------------------------------------

def proportions_from_counts(counts: Iterable[float]) -> np.ndarray:
    c = np.asarray(list(counts), dtype=float)
    if np.any(c < 0) or not np.all(np.isfinite(c)):
        raise ValueError("Counts must be finite and non-negative")
    total = float(c.sum())
    if total <= 0:
        raise ValueError("Counts must sum to a positive value")
    return c / total


def counts_from_proportions(proportions: Iterable[float], total: int) -> np.ndarray:
    """Largest-remainder conversion so integer counts sum exactly to ``total``."""
    if total < 1:
        raise ValueError("total must be at least 1")
    p = np.asarray(list(proportions), dtype=float)
    if np.any(p < 0) or not np.all(np.isfinite(p)) or p.sum() <= 0:
        raise ValueError("Invalid proportions")
    p = p / p.sum()
    raw = p * int(total)
    counts = np.floor(raw).astype(int)
    remainder = int(total) - int(counts.sum())
    if remainder:
        order = np.argsort(-(raw - counts))
        counts[order[:remainder]] += 1
    return counts


def perturb_error_mass(matrix: np.ndarray, relative_change: float) -> np.ndarray:
    """Change each row's off-diagonal mass while preserving its error pattern.

    ``+0.05`` means 5% more off-diagonal mass relative to that row's baseline;
    ``-0.05`` means 5% less. Rows are kept valid probability vectors.
    """
    p = np.asarray(matrix, dtype=float).copy()
    if p.ndim != 2 or p.shape[0] != p.shape[1]:
        raise ValueError("Matrix must be square")
    if relative_change <= -1:
        raise ValueError("relative_change must be greater than -1")
    if np.any(p < 0) or not np.all(np.isfinite(p)):
        raise ValueError("Matrix entries must be finite and non-negative")

    row_sum = p.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0):
        raise ValueError("Each matrix row must have positive mass")
    p /= row_sum

    out = np.zeros_like(p)
    eps = 1e-12
    k = p.shape[0]
    for i, row in enumerate(p):
        off = row.copy()
        off[i] = 0.0
        off_sum = float(off.sum())
        target = float(np.clip(off_sum * (1.0 + relative_change), 0.0, 1.0 - eps))
        if off_sum > eps:
            off *= target / off_sum
        elif target > eps:
            off[:] = target / (k - 1)
            off[i] = 0.0
        out[i] = off
        out[i, i] = 1.0 - target
    return out


def mean_off_diagonal_mass(matrix: np.ndarray) -> float:
    p = np.asarray(matrix, dtype=float)
    p = p / p.sum(axis=1, keepdims=True)
    return float(np.mean(1.0 - np.diag(p)))


# ---------------------------------------------------------------------------
# d_hat_norm: direct reproduction of the core R formula for Bayesian analysis
# ---------------------------------------------------------------------------

def sigma_matrices(p_matrix: np.ndarray) -> np.ndarray:
    p = np.asarray(p_matrix, dtype=float)
    if p.shape != (4, 4):
        raise ValueError("d_hat_norm implementation requires a 4x4 unreliability matrix")
    sigmas = []
    for rho in p:
        s = -np.outer(rho, rho)
        np.fill_diagonal(s, rho * (1.0 - rho))
        sigmas.append(s)
    return np.stack(sigmas, axis=0)


def d_hat_norm(
    x: Iterable[float],
    y: Iterable[float],
    p_matrix: np.ndarray,
    b_matrix: np.ndarray,
) -> float:
    """Scaled d_hat formula used by the R simulation code (scale_true = TRUE)."""
    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)
    n, m = float(x.sum()), float(y.sum())
    if x.shape != (4,) or y.shape != (4,) or n <= 0 or m <= 0:
        raise ValueError("d_hat_norm requires positive four-category Pre/Post counts")

    sigmas = sigma_matrices(p_matrix)
    weights = x / n**2 + y / m**2
    sigma_hat = np.tensordot(weights, sigmas, axes=(0, 0))
    inner = b_matrix @ sigma_hat @ b_matrix.T
    delta = b_matrix @ (x / n - y / m)
    try:
        value = float(delta.T @ np.linalg.solve(inner, delta))
    except np.linalg.LinAlgError as exc:
        raise ValueError("Projected covariance matrix is singular for this input") from exc
    return float(0.5 * (1.0 / n + 1.0 / m) * value)


def bayesian_d_hat_draws(
    x: Iterable[float],
    y: Iterable[float],
    p_matrix: np.ndarray,
    b_matrix: np.ndarray,
    draws: int = 20_000,
    prior_alpha: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """Exploratory posterior distribution for d_hat_norm.

    A symmetric Dirichlet prior is placed on each classroom response-probability
    vector. Posterior probability draws are mapped through the same scaled d_hat
    formula used by the R simulation. This is an exploratory Bayesian alternative,
    not a replacement for RYA's full minimal-effects boundary test.
    """
    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)
    n, m = float(x.sum()), float(y.sum())
    if x.shape != (4,) or y.shape != (4,) or n <= 0 or m <= 0:
        raise ValueError("Bayesian analysis requires positive four-category counts")
    if draws < 1000:
        raise ValueError("Use at least 1000 posterior draws")
    if prior_alpha <= 0:
        raise ValueError("prior_alpha must be positive")

    rng = np.random.default_rng(seed)
    alpha = np.full(4, float(prior_alpha))
    px = rng.dirichlet(alpha + x, size=draws)
    py = rng.dirichlet(alpha + y, size=draws)

    sigmas = sigma_matrices(p_matrix)  # (4, 4, 4)
    weights = px / n + py / m          # equivalent to pseudo-counts/n^2 + ...
    sigma_hat = np.einsum("dj,jkl->dkl", weights, sigmas)
    inner = np.einsum("ab,dbc,ce->dae", b_matrix, sigma_hat, b_matrix.T)
    delta = (px - py) @ b_matrix.T

    # Batch inverse/solve on 3x3 matrices.
    try:
        solved = np.linalg.solve(inner, delta[..., None])[..., 0]
    except np.linalg.LinAlgError:
        # Extremely degenerate draws can be handled with a Moore-Penrose inverse.
        solved = np.einsum("dij,dj->di", np.linalg.pinv(inner), delta)

    raw = np.einsum("di,di->d", delta, solved)
    scaled = 0.5 * (1.0 / n + 1.0 / m) * raw
    return np.asarray(scaled, dtype=float)
