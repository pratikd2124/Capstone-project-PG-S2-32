"""Shared statistical and I/O helpers for GemFinder pipeline V2.

This module deliberately preserves the original EXP002 feature construction and
scaled d_hat formulation.  It centralises reusable code so the stage scripts stay
small and auditable.
"""

from __future__ import annotations

import json
import math
from io import StringIO
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import gamma as gamma_dist

from config import RHO_ROW_SUM_TOLERANCE

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
FINAL_FIGURES_DIR = PROJECT_DIR / "figures" / "final"
DEBUG_FIGURES_DIR = PROJECT_DIR / "figures" / "debug"
ASSET_CACHE_DIR = PROJECT_DIR / ".asset_cache"

REQUIRED_ASSET_BASENAMES = {
    "RY25_PreMay10.csv",
    "RY25_PostMay10.csv",
    "unreliability-matrices.json",
    "EXP002.json",
    "EXP002_nn_optimal_epoch.onnx",
    "B_matrix.txt",
}


@dataclass(frozen=True)
class AssetPaths:
    root: Path
    pre_csv: Path
    post_csv: Path
    rho_json: Path
    metadata_json: Path
    onnx_model: Path
    b_matrix_txt: Path
    instance_data_dirs: tuple[Path, ...]


@dataclass
class PipelineContext:
    assets: AssetPaths
    rho_raw: dict[str, np.ndarray]
    rho_normalised: dict[str, np.ndarray]
    rho_notes: dict[str, str]
    metadata: dict
    min_max: pd.DataFrame
    b_matrix: np.ndarray
    training_ratio_range: tuple[float, float] | None
    session: object | None = None


@dataclass
class Exp002Prediction:
    alpha: float
    beta: float
    gamma_mean: float
    gamma_q95: float
    runtime_ms: float
    outside_training_range: list[str]
    raw_network_shape_normalised: float
    raw_network_rate_normalised: float
    denormalised_rate_scaled: float


def ensure_output_dirs() -> None:
    for path in (RESULTS_DIR, FINAL_FIGURES_DIR, DEBUG_FIGURES_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _extract_needed_members_from_zips(root: Path) -> Path:
    """Extract only required model/data members from zip files into a cache.

    This creates copies; it never modifies the original archives or raw assets.
    Instance-data CSVs are extracted because they allow a real training sample-
    size-ratio range to be derived rather than guessed.
    """
    ASSET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for archive in sorted(root.rglob("*.zip")):
        try:
            with zipfile.ZipFile(archive) as zf:
                for member in zf.namelist():
                    base = Path(member).name
                    lower = member.lower().replace("\\", "/")
                    needed = base in REQUIRED_ASSET_BASENAMES
                    training_csv = "/instance_data/" in lower and lower.endswith(".csv")
                    if not (needed or training_csv):
                        continue
                    target = ASSET_CACHE_DIR / member
                    if target.exists() and target.stat().st_size > 0:
                        continue
                    zf.extract(member, ASSET_CACHE_DIR)
        except zipfile.BadZipFile:
            continue
    return ASSET_CACHE_DIR


def _find_basename(search_roots: Sequence[Path], basename: str) -> Path:
    hits: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        direct = root / basename
        if direct.is_file():
            hits.append(direct)
        hits.extend(p for p in root.rglob(basename) if p.is_file())
    # de-duplicate while preserving deterministic sort
    hits = sorted(set(p.resolve() for p in hits))
    if not hits:
        raise FileNotFoundError(
            f"Could not find {basename!r}. Point --assets-root at the extracted project assets "
            "or at a directory containing the supplied zip archives."
        )
    # Prefer non-cache/raw file if both exist.
    non_cache = [p for p in hits if ASSET_CACHE_DIR.resolve() not in p.parents]
    return non_cache[0] if non_cache else hits[0]


def resolve_assets(assets_root: str | Path) -> AssetPaths:
    root = Path(assets_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Assets root does not exist: {root}")

    search_roots = [root]
    if any(root.rglob("*.zip")):
        search_roots.append(_extract_needed_members_from_zips(root))

    pre = _find_basename(search_roots, "RY25_PreMay10.csv")
    post = _find_basename(search_roots, "RY25_PostMay10.csv")
    rho = _find_basename(search_roots, "unreliability-matrices.json")
    meta = _find_basename(search_roots, "EXP002.json")
    onnx = _find_basename(search_roots, "EXP002_nn_optimal_epoch.onnx")
    bmat = _find_basename(search_roots, "B_matrix.txt")

    instance_dirs: list[Path] = []
    for search_root in search_roots:
        for p in search_root.rglob("instance_data") if search_root.exists() else []:
            if p.is_dir() and any(p.glob("*.csv")):
                instance_dirs.append(p.resolve())
    instance_dirs = sorted(set(instance_dirs))

    return AssetPaths(
        root=root,
        pre_csv=pre,
        post_csv=post,
        rho_json=rho,
        metadata_json=meta,
        onnx_model=onnx,
        b_matrix_txt=bmat,
        instance_data_dirs=tuple(instance_dirs),
    )


def load_metadata(path: Path) -> tuple[dict, pd.DataFrame]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    min_max = pd.read_json(StringIO(metadata["min_max_df"]))
    required = {"column", "min", "max"}
    if not required.issubset(min_max.columns):
        raise ValueError("EXP002 metadata min_max_df is malformed")
    return metadata, min_max


def normalise_rho_once(rho: np.ndarray, *, tolerance: float = RHO_ROW_SUM_TOLERANCE) -> tuple[np.ndarray, str]:
    arr = np.asarray(rho, dtype=float)
    if arr.shape != (4, 4):
        raise ValueError(f"rho must be 4x4; got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("rho contains non-finite values")
    if np.any(arr < 0):
        raise ValueError("rho contains negative probabilities")
    row_sums = arr.sum(axis=1)
    if np.any(row_sums <= 0):
        raise ValueError("rho contains a row with non-positive total mass")
    max_error = float(np.max(np.abs(row_sums - 1.0)))
    if max_error > tolerance:
        raise ValueError(
            f"rho row sums are too far from 1 for rounding correction; max absolute error={max_error:.6g}"
        )
    out = arr / row_sums[:, None]
    changed = not np.array_equal(out, arr)
    note = (
        f"normalised once from rounded row sums {row_sums.tolist()}"
        if changed
        else "already row-normalised"
    )
    return out, note


def load_rho_matrices(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, str]]:
    raw_json = json.loads(path.read_text(encoding="utf-8"))
    raw: dict[str, np.ndarray] = {}
    norm: dict[str, np.ndarray] = {}
    notes: dict[str, str] = {}
    for item, value in raw_json.items():
        arr = np.asarray(value, dtype=float)
        raw[item] = arr
        try:
            norm_arr, note = normalise_rho_once(arr)
            norm[item] = norm_arr
            notes[item] = note
        except ValueError as exc:
            notes[item] = f"INVALID: {exc}"
    return raw, norm, notes


def derive_training_ratio_range(instance_data_dirs: Sequence[Path]) -> tuple[float, float] | None:
    min_ratio = math.inf
    max_ratio = -math.inf
    seen = False
    for folder in instance_data_dirs:
        for csv_path in sorted(folder.glob("*.csv")):
            try:
                for chunk in pd.read_csv(csv_path, usecols=["n_X", "n_Y"], chunksize=200_000):
                    x = pd.to_numeric(chunk["n_X"], errors="coerce").to_numpy(float)
                    y = pd.to_numeric(chunk["n_Y"], errors="coerce").to_numpy(float)
                    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
                    if not np.any(ok):
                        continue
                    ratios = np.maximum(x[ok], y[ok]) / np.minimum(x[ok], y[ok])
                    min_ratio = min(min_ratio, float(np.min(ratios)))
                    max_ratio = max(max_ratio, float(np.max(ratios)))
                    seen = True
            except (ValueError, OSError):
                continue
    return (min_ratio, max_ratio) if seen else None


def load_context(assets_root: str | Path, *, require_model: bool = False) -> PipelineContext:
    assets = resolve_assets(assets_root)
    metadata, min_max = load_metadata(assets.metadata_json)
    rho_raw, rho_norm, rho_notes = load_rho_matrices(assets.rho_json)
    b_matrix = np.loadtxt(assets.b_matrix_txt, dtype=float)
    if b_matrix.shape != (3, 4):
        raise ValueError(f"B matrix must be 3x4; got {b_matrix.shape}")
    ctx = PipelineContext(
        assets=assets,
        rho_raw=rho_raw,
        rho_normalised=rho_norm,
        rho_notes=rho_notes,
        metadata=metadata,
        min_max=min_max,
        b_matrix=b_matrix,
        training_ratio_range=derive_training_ratio_range(assets.instance_data_dirs),
        session=None,
    )
    if require_model:
        get_onnx_session(ctx)
    return ctx


def asymmetry_mse(rho: np.ndarray) -> float:
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    return float(np.mean([(rho[i, j] - rho[j, i]) ** 2 for i, j in pairs]))


def moment_calculator(prob: Sequence[float], prefix: str) -> dict[str, float]:
    p = np.asarray(prob, dtype=float)
    rv = np.arange(1, 5, dtype=float)
    mean = float(np.dot(rv, p))
    var = float(np.dot((rv - mean) ** 2, p))
    stdev = float(np.sqrt(max(var, 0.0)))
    if stdev <= 0:
        skewness = np.nan
        kurtosis = np.nan
    else:
        skewness = float(np.dot((rv - mean) ** 3, p) / stdev**3)
        kurtosis = float(np.dot((rv - mean) ** 4, p) / stdev**4)
    return {
        f"mean_{prefix}": mean,
        f"stdev_{prefix}": stdev,
        f"skewness_{prefix}": skewness,
        f"kurtosis_{prefix}": kurtosis,
    }


def validate_counts(counts: Sequence[float]) -> np.ndarray:
    arr = np.asarray(counts, dtype=float)
    if arr.shape != (4,):
        raise ValueError(f"counts must contain four response categories; got {arr.shape}")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError("counts must be finite and non-negative")
    if float(arr.sum()) <= 0:
        raise ValueError("counts must sum to a positive sample size")
    return arr


def extract_exp002_features(x: Sequence[float], y: Sequence[float], rho: np.ndarray) -> dict[str, float]:
    x = validate_counts(x)
    y = validate_counts(y)
    n_x, n_y = float(x.sum()), float(y.sum())
    X, Y = x / n_x, y / n_y
    diff = np.abs(X - Y)

    features: dict[str, float] = {}
    features.update({f"X_{i + 1}": float(X[i]) for i in range(4)})
    features.update({f"Y_{i + 1}": float(Y[i]) for i in range(4)})
    features.update({f"XmY_{i + 1}": float(diff[i]) for i in range(4)})
    features.update(moment_calculator(X, "X"))
    features.update(moment_calculator(Y, "Y"))
    features["euclid_dist_XY"] = float(np.round(np.sqrt(np.sum((X - Y) ** 2)), 4))
    features["n_X"] = n_x
    features["n_Y"] = n_y
    for i, value in enumerate(np.asarray(rho, dtype=float).flat):
        features[f"P_{i}"] = float(value)
    features["asymm"] = asymmetry_mse(np.asarray(rho, dtype=float))
    return features


def min_max_lookup(min_max: pd.DataFrame, column: str) -> tuple[float, float]:
    rows = min_max[min_max["column"] == column]
    if rows.empty:
        raise KeyError(f"No min/max metadata for {column!r}")
    return float(rows.iloc[0]["min"]), float(rows.iloc[0]["max"])


def feature_range_violations(features: dict[str, float], metadata: dict, min_max: pd.DataFrame) -> list[str]:
    violations: list[str] = []
    for label in metadata["feature_labels"]:
        value = float(features[label])
        if not np.isfinite(value):
            violations.append(f"{label}=non-finite")
            continue
        lo, hi = min_max_lookup(min_max, label)
        tol = max(1e-12, 1e-9 * max(abs(lo), abs(hi), 1.0))
        if value < lo - tol or value > hi + tol:
            violations.append(f"{label}={value:.6g} outside [{lo:.6g}, {hi:.6g}]")
    return violations


def normalise_features(features: dict[str, float], metadata: dict, min_max: pd.DataFrame) -> np.ndarray:
    values: list[float] = []
    for label in metadata["feature_labels"]:
        value = float(features[label])
        lo, hi = min_max_lookup(min_max, label)
        if hi == lo:
            raise ValueError(f"Feature {label!r} has zero training range")
        values.append((value - lo) / (hi - lo))
    return np.asarray(values, dtype=np.float32)


def reverse_normalise_output(value: float, label: str, min_max: pd.DataFrame) -> float:
    lo, hi = min_max_lookup(min_max, label)
    return float((hi - lo) * value + lo)


def get_onnx_session(ctx: PipelineContext):
    if ctx.session is not None:
        return ctx.session
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required for EXP002 inference. Install dependencies with: "
            "pip install -r pipeline_v2/requirements.txt"
        ) from exc
    ctx.session = ort.InferenceSession(str(ctx.assets.onnx_model), providers=["CPUExecutionProvider"])
    return ctx.session


def predict_exp002(
    ctx: PipelineContext,
    x: Sequence[float],
    y: Sequence[float],
    item: str,
    rho: np.ndarray | None = None,
) -> Exp002Prediction:
    if rho is None:
        if item not in ctx.rho_normalised:
            raise KeyError(f"No valid 4x4 rho matrix for item {item!r}")
        rho = ctx.rho_normalised[item]
    rho = np.asarray(rho, dtype=float)
    x_arr, y_arr = validate_counts(x), validate_counts(y)
    features = extract_exp002_features(x_arr, y_arr, rho)
    violations = feature_range_violations(features, ctx.metadata, ctx.min_max)
    vector = normalise_features(features, ctx.metadata, ctx.min_max)

    session = get_onnx_session(ctx)
    inp = session.get_inputs()[0]
    expected_shape = inp.shape
    # Original torch.onnx export used a 1-D dummy feature tensor. Handle either
    # that exact form or a model later re-exported with an explicit batch axis.
    feed = vector
    if len(expected_shape) == 2:
        feed = vector[None, :]

    t0 = time.perf_counter()
    outputs = session.run(None, {inp.name: feed.astype(np.float32, copy=False)})
    runtime_ms = (time.perf_counter() - t0) * 1000.0
    raw = np.asarray(outputs[0], dtype=float).reshape(-1)
    if raw.size < 2:
        raise ValueError(f"EXP002 returned {raw.size} outputs; expected shape and rate")

    alpha = reverse_normalise_output(float(raw[0]), "shape", ctx.min_max)
    rate_scaled = reverse_normalise_output(float(raw[1]), "rate", ctx.min_max)

    # Training code transformed the original Gamma rate as:
    #   rate_scaled = 2 * beta / (1/n_X + 1/n_Y)
    # Therefore invert that transform for the actual Gamma rate beta.
    n_x, n_y = float(x_arr.sum()), float(y_arr.sum())
    beta = rate_scaled * 0.5 * (1.0 / n_x + 1.0 / n_y)

    if not np.isfinite(alpha) or not np.isfinite(beta) or alpha <= 0 or beta <= 0:
        raise ValueError(f"EXP002 produced invalid Gamma parameters alpha={alpha}, beta={beta}")
    mean = alpha / beta
    q95 = float(gamma_dist.ppf(0.95, a=alpha, scale=1.0 / beta))
    return Exp002Prediction(
        alpha=float(alpha),
        beta=float(beta),
        gamma_mean=float(mean),
        gamma_q95=q95,
        runtime_ms=float(runtime_ms),
        outside_training_range=violations,
        raw_network_shape_normalised=float(raw[0]),
        raw_network_rate_normalised=float(raw[1]),
        denormalised_rate_scaled=float(rate_scaled),
    )


def sigma_maker(row_index: int, rho: np.ndarray) -> np.ndarray:
    r = np.asarray(rho, dtype=float)[row_index]
    sigma = -np.outer(r, r)
    np.fill_diagonal(sigma, r * (1.0 - r))
    return sigma


def d_hat_norm(x: Sequence[float], y: Sequence[float], rho: np.ndarray, b_matrix: np.ndarray) -> float:
    """Original scaled d_hat formula used by the GemFinder R simulation."""
    x = validate_counts(x)
    y = validate_counts(y)
    n, m = float(x.sum()), float(y.sum())
    p_x, p_y = x / n, y / m
    sigmas = np.stack([sigma_maker(j, rho) for j in range(4)])
    coeff = p_x / n + p_y / m
    sigma_hat = np.einsum("j,jab->ab", coeff, sigmas)
    inner = b_matrix @ sigma_hat @ b_matrix.T
    diff_b = b_matrix @ (p_x - p_y)
    solved = np.linalg.solve(inner, diff_b)
    d = float(diff_b @ solved)
    return float(0.5 * (1.0 / n + 1.0 / m) * d)


def bayesian_d_hat_draws(
    x: Sequence[float],
    y: Sequence[float],
    rho: np.ndarray,
    b_matrix: np.ndarray,
    *,
    draws: int,
    prior_alpha: float,
    seed: int,
) -> np.ndarray:
    """Exact Dirichlet posteriors + Monte Carlo propagation through d_hat_norm.

    The Dirichlet-multinomial part is conjugate, so no variational approximation
    is needed here. Monte Carlo is used only to propagate exact posterior draws
    through the nonlinear d_hat calculation.
    """
    x = validate_counts(x)
    y = validate_counts(y)
    if prior_alpha <= 0:
        raise ValueError("prior_alpha must be positive")
    if draws < 1:
        raise ValueError("draws must be positive")
    rng = np.random.default_rng(seed)
    p_x = rng.dirichlet(x + prior_alpha, size=draws)
    p_y = rng.dirichlet(y + prior_alpha, size=draws)
    n, m = float(x.sum()), float(y.sum())

    sigmas = np.stack([sigma_maker(j, rho) for j in range(4)])
    coeff = p_x / n + p_y / m
    sigma_hat = np.einsum("dj,jab->dab", coeff, sigmas)
    inner = np.einsum("ij,djk,lk->dil", b_matrix, sigma_hat, b_matrix)
    diff_b = (p_x - p_y) @ b_matrix.T
    solved = np.linalg.solve(inner, diff_b[..., None])[..., 0]
    quad = np.einsum("di,di->d", diff_b, solved)
    return 0.5 * (1.0 / n + 1.0 / m) * quad


def bayesian_d_hat_draws_with_rho(
    x: Sequence[float],
    y: Sequence[float],
    rho_draws: np.ndarray,
    b_matrix: np.ndarray,
    *,
    prior_alpha: float,
    seed: int,
) -> np.ndarray:
    x = validate_counts(x)
    y = validate_counts(y)
    rhos = np.asarray(rho_draws, dtype=float)
    if rhos.ndim != 3 or rhos.shape[1:] != (4, 4):
        raise ValueError("rho_draws must have shape (draws, 4, 4)")
    draws = rhos.shape[0]
    rng = np.random.default_rng(seed)
    p_x = rng.dirichlet(x + prior_alpha, size=draws)
    p_y = rng.dirichlet(y + prior_alpha, size=draws)
    n, m = float(x.sum()), float(y.sum())

    # sigma[d,j,a,b]
    outer = rhos[:, :, :, None] * rhos[:, :, None, :]
    sigmas = -outer
    diag_idx = np.arange(4)
    sigmas[:, :, diag_idx, diag_idx] = rhos * (1.0 - rhos)
    coeff = p_x / n + p_y / m
    sigma_hat = np.einsum("dj,djab->dab", coeff, sigmas)
    inner = np.einsum("ij,djk,lk->dil", b_matrix, sigma_hat, b_matrix)
    diff_b = (p_x - p_y) @ b_matrix.T
    solved = np.linalg.solve(inner, diff_b[..., None])[..., 0]
    quad = np.einsum("di,di->d", diff_b, solved)
    return 0.5 * (1.0 / n + 1.0 / m) * quad


def counts_from_proportions(prob: Sequence[float], n: int) -> np.ndarray:
    p = np.asarray(prob, dtype=float)
    p = p / p.sum()
    raw = p * int(n)
    counts = np.floor(raw).astype(int)
    remainder = int(n) - int(counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw - counts))
        counts[order[:remainder]] += 1
    return counts.astype(float)


def perturb_rho_offdiagonal_mass(rho: np.ndarray, relative_change: float) -> np.ndarray:
    """Scale each row's off-diagonal error mass and rebalance its diagonal.

    At relative_change == 0 the returned matrix is exactly the supplied baseline
    (after copying), satisfying the pipeline's zero-perturbation requirement.
    """
    base = np.asarray(rho, dtype=float)
    if relative_change == 0:
        return base.copy()
    out = base.copy()
    for i in range(4):
        off_idx = [j for j in range(4) if j != i]
        out[i, off_idx] = base[i, off_idx] * (1.0 + relative_change)
        off_sum = float(out[i, off_idx].sum())
        out[i, i] = 1.0 - off_sum
        if out[i, i] < 0:
            raise ValueError(f"rho perturbation {relative_change:+.1%} creates a negative diagonal probability")
    return out


def parse_counts_from_row(row: pd.Series, side: str) -> np.ndarray:
    return np.asarray([row[f"counts_{side}_{i}"] for i in range(1, 5)], dtype=float)


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def probability_column(label: str) -> str:
    clean = re.sub(r"[^0-9A-Za-z]+", "_", label).strip("_")
    return f"prob_dhat_gt_{clean}"


def summarise_draws(draws: np.ndarray, prefix: str = "bayes_dhat") -> dict[str, float]:
    arr = np.asarray(draws, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("No finite posterior draws")
    q025, q50, q975 = np.quantile(arr, [0.025, 0.50, 0.975])
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_median": float(q50),
        f"{prefix}_sd": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        f"{prefix}_ci_low": float(q025),
        f"{prefix}_ci_high": float(q975),
    }


def parse_group_pair(text: str) -> tuple[str, str]:
    if ":" not in text:
        raise ValueError(f"Group pair {text!r} must use A:B format")
    left, right = text.split(":", 1)
    return left.strip(), right.strip()


def comparable_group_value(series: pd.Series, value: str | int | float) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    try:
        target_num = float(value)
        return numeric.eq(target_num)
    except (TypeError, ValueError):
        return series.astype("string").str.strip().eq(str(value).strip())
