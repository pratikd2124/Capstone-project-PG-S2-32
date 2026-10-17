"""Stages 1a and 1b: rho validation and Gamma-compatibility screening (Objective 1, SIN-WEI).

Wraps SIN-WEI's ``O1ScreeningGate`` (v2, 16 Oct) unchanged, with two integration tweaks:

* the detector runs from ``detector_params.json`` (the same StandardScaler + logistic
  regression, exported from her joblib file) so it no longer depends on the scikit-learn
  version that saved it (1.5.1) and can be ported to the browser;
* both pairs are screened, as agreed: the degeneracy safeguard on the observed pair, and the
  classifier on the observed pair AND on the boundary pair that actually reaches the network.
  FLAGGED if either is flagged. A training-range check (group sizes 5-9,940) is added.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import FLAGGED, PASS, REJECT, BoundaryPair, ComparisonRequest, RhoCheck, ScreeningResult
from .o1.o1_screening import O1ScreeningGate, calculate_P_structure_descriptors

O1_DIR = Path(__file__).parent / "o1"
TRAIN_N_MIN, TRAIN_N_MAX = 5, 9940          # docs/findings.md: training group sizes 5-9,819 / 5-9,940
ROW_SUM_TOL = 0.020001                      # SIN-WEI's tolerance; warn above 1e-6, reject above this
OUT_OF_SCOPE_ITEMS = {f"chs{i}" for i in range(1, 7)}


class _NumpyDetector:
    """StandardScaler + LogisticRegression, from exported parameters."""

    def __init__(self, params: dict):
        self.mean = np.asarray(params["scaler_mean"])
        self.scale = np.asarray(params["scaler_scale"])
        self.coef = np.asarray(params["coef"])
        self.intercept = float(params["intercept"])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        z = (np.asarray(X, dtype=float) - self.mean) / self.scale
        p1 = 1.0 / (1.0 + np.exp(-(z @ self.coef + self.intercept)))
        return np.column_stack([1 - p1, p1])


class Gate(O1ScreeningGate):
    """SIN-WEI's gate, loading the detector from JSON instead of joblib."""

    def __post_init__(self) -> None:
        self.detector_model_path = Path(self.detector_model_path)
        self.detector_metadata_path = Path(self.detector_metadata_path)
        self.detector = _NumpyDetector(json.loads(self.detector_model_path.read_text()))
        self.detector_metadata = json.loads(self.detector_metadata_path.read_text())
        self.feature_order = list(self.detector_metadata["feature_order"])
        self.threshold = float(self.detector_metadata["threshold"])
        self._p_scaler = self._p_nn = self._p_reference_df = self._training_nearest_distances = None
        if self.screening_reference_csv is not None:
            self._prepare_training_space_reference(Path(self.screening_reference_csv))


def load_gate(reference_csv: str | None = None, threshold: float | None = None) -> Gate:
    g = Gate(detector_model_path=O1_DIR / "detector_params.json",
             detector_metadata_path=O1_DIR / "O1_07_compatibility_detector_v1_metadata.json",
             screening_reference_csv=reference_csv)
    if threshold is not None:
        g.threshold = float(threshold)
    return g


def validate_rho(req: ComparisonRequest) -> RhoCheck:
    """Stage 1a. Reject what can't be scored; warn on rounding."""
    reasons = []
    P = req.P
    if req.item.lower() in OUT_OF_SCOPE_ITEMS:
        return RhoCheck("reject", ("six-option chs item: no model exists for it",))
    if P.shape != (4, 4):
        return RhoCheck("reject", (f"rho must be 4x4, got {P.shape}",))
    if not np.isfinite(P).all():
        return RhoCheck("reject", ("rho contains NaN or infinite values",))
    if (P < 0).any():
        return RhoCheck("reject", ("rho contains negative probabilities",))
    err = float(np.max(np.abs(P.sum(axis=1) - 1)))
    desc = calculate_P_structure_descriptors(P)
    if err > ROW_SUM_TOL:
        return RhoCheck("reject", (f"rho row sums off by up to {err:.3f} (limit 0.02)",), err,
                        desc["certainty_P"], desc["rya_asymmetry_P"])
    status = "pass"
    if err > 1e-6:
        status = "warn"
        reasons.append(f"rho row sums rounded (max error {err:.3f}); kept as given")
    if np.any(np.diag(P) < P.max(axis=1)):
        status = "warn"
        reasons.append("a latent type is not most likely to give its own answer")
    if req.rho_source == "json" and req.item in ("sun1", "ry17"):
        reasons.append("JSON and P.RData differ in row 2 col 1 for this item")
    return RhoCheck(status, tuple(reasons), err, desc["certainty_P"], desc["rya_asymmetry_P"])


def screen(req: ComparisonRequest, rho_check: RhoCheck, bp: BoundaryPair | None, gate: Gate) -> ScreeningResult:
    """Stage 1b. Route PASS / FLAGGED / REJECT, screening both pairs."""
    if rho_check.status == "reject":
        return ScreenResultReject(rho_check.reasons)
    obs = gate.screen_case(req.x, req.y, req.P, question=req.item)
    if obs["route"] == "invalid_input":
        return ScreenResultReject((obs["reason"],))
    reasons = []
    p_obs = obs["classifier"]["incompatibility_probability"]
    if obs["reason"] == "original_pair_degenerate":
        reasons.append("observed group answers all in one category")
    if obs["classifier"]["flagged"]:
        reasons.append(f"classifier flags observed pair (p={p_obs:.2f})")

    p_bnd = float("nan")
    if bp is not None and np.isfinite(bp.x).all() and bp.x.sum() > 0:
        bnd = gate.screen_case(req.x, req.y, req.P, question=req.item, nn_x=bp.x, nn_y=bp.y)
        if bnd["route"] != "invalid_input":
            p_bnd = bnd["classifier"]["incompatibility_probability"]
            if bnd["classifier"]["flagged"]:
                reasons.append(f"classifier flags boundary pair (p={p_bnd:.2f})")

    n, m = req.x.sum(), req.y.sum()
    outside = not (TRAIN_N_MIN <= n <= TRAIN_N_MAX and TRAIN_N_MIN <= m <= TRAIN_N_MAX)
    if outside:
        reasons.append(f"group sizes {int(n)}/{int(m)} outside training range {TRAIN_N_MIN}-{TRAIN_N_MAX}")
    route = FLAGGED if reasons else PASS
    return ScreeningResult(route, tuple(reasons) or ("screening passed",), p_obs, p_bnd, gate.threshold,
                           outside, {"o1_observed": obs})


def ScreenResultReject(reasons) -> ScreeningResult:
    return ScreeningResult(REJECT, tuple(reasons))
