"""All the settings for the toolkit, in one place.

Every script, the command-line tool and the walkthrough notebook read their paths and choices
from here, so there is one list to look at instead of values scattered through the code.

How to change a setting on YOUR computer
----------------------------------------
Don't edit this file: it's shared through git, and everyone's paths are different. Instead:

1. Copy ``local_settings.example.py`` (in the repo root) to ``local_settings.py``.
2. Change the values you need in ``local_settings.py``, usually just ``DATA_DIR``.

``local_settings.py`` is ignored by git, so your paths stay on your machine and never clash
with a teammate's.

Order of priority (highest first):
    the SCRP_DATA environment variable  >  local_settings.py  >  the defaults below
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

# The repo root: the folder that holds README.md. Everything below is relative to it, which is
# why the defaults work on anyone's machine without editing.
REPO_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------------------------
# Where the data lives
# ---------------------------------------------------------------------------------------------

# The customer's files from Box (model, training CSVs, matrices, RY25 surveys). The team repo
# ships them in ./data, so the default works as-is. Point this somewhere else if you keep the
# data outside the repo, e.g. Path(r"D:\Box\Statistical Core Refinement Project").
DATA_DIR = REPO_DIR / "data"

# ---------------------------------------------------------------------------------------------
# Where results get written
# ---------------------------------------------------------------------------------------------

REPORT_CSV = REPO_DIR / "baseline_report.csv"   # one row per baseline check (`cli baseline`)
EDGE_CASE_DIR = REPO_DIR / "edge_cases"         # edge-case files for the input-safety detector
PROFILE_JSON = REPO_DIR / "profile_report.json" # where the model's time and memory go (`cli profile`)

# ---------------------------------------------------------------------------------------------
# The worked example: RYA's own, from feature_calculation_portal_script.py
# ---------------------------------------------------------------------------------------------

EXAMPLE_ITEM = "ry9"
EXAMPLE_PRE = [40, 52, 7, 14]      # number of students choosing answers 1..4 before
EXAMPLE_POST = [9, 17, 54, 12]     # ... and after

# ---------------------------------------------------------------------------------------------
# Choices for the reference numbers
# ---------------------------------------------------------------------------------------------

# RYA picked EXP002's train/test split at random and never saved it, so we fix one here to make
# our reference numbers repeatable. Changing it changes the dataset section of the baseline.
SPLIT_SEED = 42
TRAIN_FRAC = 0.8

# ---------------------------------------------------------------------------------------------
# Walkthrough notebook switches
# ---------------------------------------------------------------------------------------------

CHECK_ALL_TRAINING_ROWS = True     # Step 4: False checks a 3,000-row sample instead (faster)
RUN_TRAINING_DEMO = False          # Step 9: True trains a fresh network for 15 epochs (slow)

# ---------------------------------------------------------------------------------------------
# Known data issue (see docs/findings.md)
# ---------------------------------------------------------------------------------------------

# One training row is corrupt: 1 student in the "after" group, but answer shares of 0, 1, 1, 1.
# It stays in the baseline (EXP002 was trained with it); drop it before any retraining.
CORRUPT_ROW = "orthog_B_3_large_classes_2790"


# ---------------------------------------------------------------------------------------------
# Personal overrides: nothing to edit below this line
# ---------------------------------------------------------------------------------------------

def _load_local_overrides() -> None:
    """Apply local_settings.py (if it exists) on top of the defaults above."""
    local_file = REPO_DIR / "local_settings.py"
    if not local_file.is_file():
        return
    spec = importlib.util.spec_from_file_location("local_settings", local_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, value in vars(module).items():
        if name.isupper():                      # only settings, not imports or helpers
            if name.endswith(("_DIR", "_CSV", "_JSON")):
                value = Path(value)             # accept plain strings for paths too
            globals()[name] = value


_load_local_overrides()

# An environment variable beats everything, which is handy for one-off runs:
#   set SCRP_DATA=D:\somewhere\else   (Windows)   /   export SCRP_DATA=...   (Mac/Linux)
if os.environ.get("SCRP_DATA"):
    DATA_DIR = Path(os.environ["SCRP_DATA"])
