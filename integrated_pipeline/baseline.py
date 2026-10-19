"""RYA's baseline (the scrp_toolkit package and the model files), pulled from GitHub.

This branch holds only the new pipeline work. The baseline lives on the team repo's
Baseline-Recreation branch, so the first time it is needed we download exactly the files we
use from a fixed commit of that branch into a local folder (`.baseline/`, never committed).
After that everything runs offline.

    from integrated_pipeline.baseline import ensure_baseline, data_root
    ensure_baseline()        # makes `import scrp_toolkit` work
    data_root()              # folder with EXP002, the rho matrices and one training file

To use a baseline you already have on disk instead (for example a local clone of the
Baseline-Recreation branch), set the environment variable SCRP_BASELINE_DIR to that folder.
"""
from __future__ import annotations

import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPO = "pratikd2124/Capstone-project-PG-S2-32"
BRANCH = "Baseline-Recreation"
COMMIT = "c61b38356737355bc44c67c480cd093b9caf931f"   # the baseline everything here was tested against

CODE_FILES = [f"scrp_toolkit/{name}.py" for name in (
    "__init__", "baseline", "cli", "compare", "config", "dataset", "edge_cases", "features",
    "inference", "model", "rds", "reliability", "scoring", "settings", "train", "validate")]

_NETWORK = "data/Final Network w Code (15-07-2022)/final_network_July15_22"
DATA_FILES = [
    "data/Unreliability Matrices/unreliability-matrices.json",          # rho for every item
    f"{_NETWORK}/final_network_model/EXP002.json",                      # network settings + scaling
    f"{_NETWORK}/final_network_model/EXP002_nn_optimal_epoch.onnx",     # the trained network
    f"{_NETWORK}/final_code/B_matrix.txt",                              # RYA's B matrix
    f"{_NETWORK}/instance_data/orthog_B_3_large_classes_feature_output_df.csv",  # demo case J
]

CACHE_DIR = Path(__file__).resolve().parent.parent / ".baseline"


def baseline_dir() -> Path:
    """Where the baseline is: SCRP_BASELINE_DIR if set, otherwise the download folder."""
    custom = os.environ.get("SCRP_BASELINE_DIR")
    return Path(custom).expanduser().resolve() if custom else CACHE_DIR


def _download(relative: str, dest: Path) -> None:
    url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{urllib.parse.quote(relative)}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, open(tmp, "wb") as out:
            out.write(response.read())
    except Exception as error:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Could not download {relative} from GitHub ({error}). Check the internet "
                           "connection, or set SCRP_BASELINE_DIR to a local copy of the "
                           f"{BRANCH} branch.") from error
    tmp.replace(dest)


def _fetch(files) -> Path:
    root = baseline_dir()
    missing = [f for f in files if not (root / f).exists()]
    if missing and root != CACHE_DIR:
        raise FileNotFoundError(f"SCRP_BASELINE_DIR={root} is missing: {', '.join(missing)}")
    if missing:
        print(f"Downloading {len(missing)} baseline file(s) from GitHub ({BRANCH} @ {COMMIT[:7]}) ...")
        for f in missing:
            _download(f, root / f)
    return root


def ensure_baseline() -> Path:
    """Make `import scrp_toolkit` work, downloading the baseline code the first time."""
    root = _fetch(CODE_FILES)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def data_root() -> str:
    """The baseline data folder (EXP002, rho matrices, one training file), downloaded if needed."""
    ensure_baseline()
    return str(_fetch(DATA_FILES) / "data")
