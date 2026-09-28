"""The RYA baseline: run RYA's own pipeline on RYA's own files and compare every number
against a saved baseline, flagging each mismatch.

"Identical to RYA" is enforced in three layers, each a section of the report:

files    -- every input file's SHA-1 equals the SHA-1 Box reports for the customer's copy
            (Box folder "Statistical Core Refinement Project"). A different or edited file fails.
scaling  -- instance_data/ run through ``dataset.build_preprocessed_dataset`` (a port of RYA's
            net_dataset_init_B.py) reproduces the min_max_df stored in the customer's
            EXP002.json, all 42 columns x (min, max). This proves our preprocessing is the one
            EXP002 was trained with -- the model itself is never retrained.
outputs  -- the numbers the pipeline produces with the shipped EXP002 ONNX model: P-matrix
            asymmetry ranking, model summary, the ry9 example, a real RY25 school, and EXP002's
            error on the instance_data split. These come from ``--capture`` on the verified files.

Every check is one row of the report (section, check, expected, got, status, note). The
command exits 1 on any FAIL. ``--capture`` refuses to write a baseline unless the files and
scaling sections pass, so a baseline can only ever be recorded from the customer's exact
files and RYA's exact preprocessing.
"""
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import ProjectPaths
from .reliability import load_p_matrices, score_asymmetry

RTOL = 1e-5  # float32 ONNX output: allow tiny cross-platform/runtime differences

KNOWN_ANSWERS = Path(__file__).parent / "known_answers"
RYA_BASELINE = KNOWN_ANSWERS / "rya_baseline.json"
MURAT_REFERENCE = KNOWN_ANSWERS / "murat_notebook_reference.json"

# ProjectPaths attribute -> file name, for the SHA-1 manifest
_PATH_FIELDS = {
    "onnx_path": "EXP002_nn_optimal_epoch.onnx",
    "metadata_path": "EXP002.json",
    "pmatrices_path": "unreliability-matrices.json",
    "pre_csv_path": "RY25_PreMay10.csv",
    "post_csv_path": "RY25_PostMay10.csv",
}


@dataclass
class Check:
    section: str
    check: str
    expected: object
    got: object
    passed: bool
    note: str = ""

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"


def _close(got, expected, atol: float, rtol: float = RTOL) -> bool:
    try:
        return bool(np.isclose(float(got), float(expected), rtol=rtol, atol=atol))
    except (TypeError, ValueError):
        return False


def _num(section, name, got, expected, atol, rtol=RTOL) -> Check:
    return Check(section, name, expected, None if got is None else round(float(got), 6), _close(got, expected, atol, rtol))


def _eq(section, name, got, expected, note="") -> Check:
    return Check(section, name, expected, got, got == expected, note)


def _missing(section, name, expected, why) -> Check:
    return Check(section, name, expected, None, False, why)


def sha1_of(path: str | Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_map(paths: ProjectPaths) -> dict:
    """{file name: local path or None} for every file in the manifest."""
    files = {name: getattr(paths, field) for field, name in _PATH_FIELDS.items()}
    if paths.instance_dir:
        for p in sorted(Path(paths.instance_dir).glob("*.csv")):
            files[p.name] = str(p)
    return files


# ---------------------------------------------------------------------------
# Section runners
# ---------------------------------------------------------------------------


def check_files(paths: ProjectPaths, manifest: dict) -> List[Check]:
    files = _file_map(paths)
    checks = []
    for name, expected_sha1 in manifest.items():
        local = files.get(name)
        if local is None:
            checks.append(_missing("files", name, expected_sha1, "file not found under --project-root"))
        else:
            checks.append(_eq("files", name, sha1_of(local), expected_sha1))
    extra = sorted(n for n in files if n.endswith(".csv") and n not in manifest and files[n]
                   and Path(files[n]).parent == Path(paths.instance_dir or ""))
    for name in extra:  # a stray CSV in instance_data/ would silently change the training data
        checks.append(Check("files", name, "not present", "present", False,
                            "extra CSV in instance_data/ -- remove it (e.g. edge-case rows)"))
    return checks


def run_scaling(paths: ProjectPaths, metadata: dict, normalise_rate: bool = True):
    """Rebuild the dataset; return (df, feature_cols, rebuilt_min_max, stored_min_max)."""
    from .dataset import build_preprocessed_dataset

    df, feature_cols, rebuilt = build_preprocessed_dataset(paths.instance_dir, normalise_rate=normalise_rate)
    stored = pd.read_json(io.StringIO(metadata["min_max_df"]))
    return df, feature_cols, rebuilt, stored


def check_scaling(df, feature_cols, rebuilt: pd.DataFrame, stored: pd.DataFrame, metadata: dict) -> List[Check]:
    checks = [_eq("scaling", "feature_order", list(feature_cols), list(metadata["feature_labels"]))]
    r, s = rebuilt.set_index("column"), stored.set_index("column")
    checks.append(_eq("scaling", "columns", list(r.index), list(s.index)))
    for col in s.index:
        for stat in ("min", "max"):
            got = r[stat].get(col) if col in r.index else None
            # EXP002.json stores values rounded to 10 decimals
            checks.append(_num("scaling", f"{col}.{stat}", got, s.loc[col, stat], atol=1e-9, rtol=1e-9))
    return checks


def run_explore(paths: ProjectPaths, top_n: int = 10) -> dict:
    p = load_p_matrices(paths.pmatrices_path)
    scores = score_asymmetry(p)
    return {"n_items": int(len(p)),
            "n_four_option_items": int(sum(np.shape(m) == (4, 4) for m in p.values())),
            "top_asymm": {q: float(v) for q, v in scores.head(top_n).items()}}


def run_ry9(predictor, case: dict) -> dict:
    shape, rate = predictor.predict(np.array(case["x"]), np.array(case["y"]), case["question"])
    return {"question": case["question"], "x": case["x"], "y": case["y"],
            "shape": shape, "rate": rate, "gamma_mean": shape / rate}


def run_case_study(predictor, paths: ProjectPaths) -> Optional[pd.DataFrame]:
    from .scoring import score_school

    if not paths.have_ry25_data:
        return None
    return score_school(predictor, paths)


def run_split_error(predictor, df, feature_cols, min_max_df, seed: int, train_frac: float) -> dict:
    """EXP002's shape/rate MAE on the ``seed`` test split (no training).

    RYA drew EXP002's split seed with ``np.random.randint(100)`` and never saved it, so this
    split is NOT EXP002's held-out set -- most of these rows were in its training data. It is a
    reproducible reference number for comparing changes, not a generalisation score.
    """
    from .compare import _rescale, mae
    from .dataset import train_test_split_like_original

    train_df, test_df = train_test_split_like_original(df, train_frac=train_frac, random_seed=seed)
    raw = test_df[feature_cols].copy()
    for c in feature_cols:
        raw[c] = _rescale(min_max_df, c, raw[c].values)
    preds = np.array([predictor.predict_raw_features(row) for _, row in raw.iterrows()])
    true_shape = _rescale(min_max_df, "shape", test_df["shape"].values)
    true_rate = _rescale(min_max_df, "rate", test_df["rate"].values)
    return {"n_rows": int(len(df)), "n_train": int(len(train_df)), "n_test": int(len(test_df)),
            "exp002_shape_mae": mae(preds[:, 0], true_shape), "exp002_rate_mae": mae(preds[:, 1], true_rate)}


# ---------------------------------------------------------------------------
# Check / capture
# ---------------------------------------------------------------------------


def _load_metadata(paths: ProjectPaths) -> dict:
    with open(paths.metadata_path) as f:
        return json.load(f)


def check_against_baseline(paths: ProjectPaths, baseline: dict, include_dataset: bool = True) -> List[Check]:
    from .inference import UnreliabilityPredictor

    checks: List[Check] = []
    if "files" in baseline:
        checks += check_files(paths, baseline["files"])

    exp = baseline.get("explore")
    if exp:
        got = run_explore(paths, top_n=len(exp["top_asymm"]))
        checks.append(_eq("explore", "n_items", got["n_items"], exp["n_items"]))
        if "n_four_option_items" in exp:
            checks.append(_eq("explore", "n_four_option_items", got["n_four_option_items"], exp["n_four_option_items"]))
        checks.append(_eq("explore", "top_order", list(got["top_asymm"]), list(exp["top_asymm"])))
        for q, v in exp["top_asymm"].items():
            checks.append(_num("explore", f"asymm[{q}]", got["top_asymm"].get(q), v, exp["atol"]))

    if not (paths.onnx_path and paths.metadata_path):
        checks.append(_missing("model", "EXP002 model files", "present",
                               "EXP002_nn_optimal_epoch.onnx / EXP002.json not found -- later sections skipped"))
        return checks
    metadata = _load_metadata(paths)
    predictor = UnreliabilityPredictor(paths)
    if "model" in baseline:
        checks.append(_eq("model", "describe", predictor.describe(), baseline["model"]["describe"]))

    ry9 = baseline.get("ry9_example")
    if ry9:
        got = run_ry9(predictor, ry9)
        for k in ("shape", "rate", "gamma_mean"):
            if k in ry9:
                checks.append(_num("ry9", k, got[k], ry9[k], ry9["atol"]))

    cs = baseline.get("case_study")
    if cs:
        scores = run_case_study(predictor, paths)
        if scores is None:
            checks.append(_missing("case_study", "school", cs["school"], "RY25 CSVs not found"))
        else:
            checks.append(_eq("case_study", "school", str(scores.attrs.get("school")), cs["school"]))
            if "n_items_scored" in cs:
                checks.append(_eq("case_study", "n_items_scored", int(len(scores)), cs["n_items_scored"]))
            by_q = scores.set_index("question")
            for item in cs["items"]:
                q = item["question"]
                if q not in by_q.index:
                    checks.append(_missing("case_study", q, "scored", "item not scored"))
                    continue
                row = by_q.loc[q]
                for k in ("n_pre", "n_post"):
                    checks.append(_eq("case_study", f"{q}.{k}", int(row[k]), item[k]))
                for k in ("shape_hat", "rate_hat", "gamma_mean", "asymm"):
                    if k in item:
                        checks.append(_num("case_study", f"{q}.{k}", row[k], item[k], cs["atol"]))

    ds = baseline.get("dataset")
    if include_dataset and (ds or baseline.get("check_scaling")):
        if not paths.instance_dir:
            checks.append(_missing("scaling", "instance_data", "present", "instance_data/ not found"))
        else:
            df, feature_cols, rebuilt, stored = run_scaling(paths, metadata)
            if baseline.get("check_scaling"):
                checks += check_scaling(df, feature_cols, rebuilt, stored, metadata)
            if ds:
                got = run_split_error(predictor, df, feature_cols, rebuilt, ds["split_seed"], ds["train_frac"])
                for k in ("n_rows", "n_train", "n_test"):
                    checks.append(_eq("dataset", k, got[k], ds[k]))
                for k in ("exp002_shape_mae", "exp002_rate_mae"):
                    if k in ds:
                        checks.append(_num("dataset", k, got[k], ds[k], ds["mae_atol"]))
    return checks


def report_frame(checks: List[Check]) -> pd.DataFrame:
    return pd.DataFrame([{"section": c.section, "check": c.check, "expected": c.expected,
                          "got": c.got, "status": c.status, "note": c.note} for c in checks])


def print_summary(checks: List[Check]) -> bool:
    df = report_frame(checks)
    fails = df[df["status"] == "FAIL"]
    for section, grp in df.groupby("section", sort=False):
        n_fail = int((grp["status"] == "FAIL").sum())
        print(f"[{'PASS' if n_fail == 0 else 'FAIL'}] {section:<11} {len(grp) - n_fail}/{len(grp)} checks passed")
    if len(fails):
        print("\nMismatches:")
        with pd.option_context("display.max_colwidth", 60, "display.width", 200):
            print(fails[["section", "check", "expected", "got", "note"]].head(60).to_string(index=False))
        if len(fails) > 60:
            print(f"... {len(fails) - 60} more in the CSV report")
    print(f"\nTOTAL: {int((df['status'] == 'PASS').sum())}/{len(df)} checks passed -> "
          f"{'BASELINE MATCHES' if fails.empty else 'BASELINE MISMATCH'}")
    return fails.empty


def capture_baseline(paths: ProjectPaths, template: dict) -> dict:
    """Run everything on verified inputs and return a complete baseline dict.

    Refuses (raises RuntimeError) unless every file matches the template's SHA-1 manifest and
    the rebuilt scaling matches EXP002.json -- so a captured baseline is always RYA's.
    """
    from .inference import UnreliabilityPredictor

    gate = check_files(paths, template["files"]) if "files" in template else []
    paths.require("onnx_path", "metadata_path", "instance_dir")
    metadata = _load_metadata(paths)
    df, feature_cols, rebuilt, stored = run_scaling(paths, metadata)
    gate += check_scaling(df, feature_cols, rebuilt, stored, metadata)
    bad = [c for c in gate if not c.passed]
    if bad:
        print_summary(gate)
        raise RuntimeError(f"Not capturing: {len(bad)} file/scaling check(s) failed -- inputs are not RYA's.")

    predictor = UnreliabilityPredictor(paths)
    out = {k: template[k] for k in ("_source", "files", "check_scaling") if k in template}
    out["_captured"] = "Outputs recorded by `scrp_toolkit.cli baseline --capture` after files + scaling passed."

    exp = run_explore(paths, top_n=len(template["explore"]["top_asymm"]))
    out["explore"] = {"n_items": exp["n_items"], "n_four_option_items": exp["n_four_option_items"],
                      "top_asymm": {q: round(v, 6) for q, v in exp["top_asymm"].items()},
                      "atol": template["explore"]["atol"]}
    out["model"] = {"describe": predictor.describe()}

    t = template.get("ry9_example", {"question": "ry9", "x": [40, 52, 7, 14], "y": [9, 17, 54, 12], "atol": 1e-6})
    ry9 = run_ry9(predictor, t)
    out["ry9_example"] = {**{k: ry9[k] for k in ("question", "x", "y")},
                          **{k: round(ry9[k], 8) for k in ("shape", "rate", "gamma_mean")}, "atol": 1e-6}

    scores = run_case_study(predictor, paths)
    if scores is not None:
        out["case_study"] = {
            "school": str(scores.attrs.get("school")), "n_items_scored": int(len(scores)), "atol": 1e-6,
            "items": [{"question": r.question, "n_pre": int(r.n_pre), "n_post": int(r.n_post),
                       **{k: round(float(getattr(r, k)), 8) for k in ("shape_hat", "rate_hat", "gamma_mean", "asymm")}}
                      for r in scores.itertuples()],  # every scored item, not a sample
        }

    ds_t = template.get("dataset", {"split_seed": 42, "train_frac": 0.8, "mae_atol": 1e-6})
    got = run_split_error(predictor, df, feature_cols, rebuilt, ds_t["split_seed"], ds_t["train_frac"])
    out["dataset"] = {"split_seed": ds_t["split_seed"], "train_frac": ds_t["train_frac"],
                      **{k: got[k] for k in ("n_rows", "n_train", "n_test")},
                      "exp002_shape_mae": round(got["exp002_shape_mae"], 8),
                      "exp002_rate_mae": round(got["exp002_rate_mae"], 8), "mae_atol": 1e-6}
    return out


def load_baseline(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)
