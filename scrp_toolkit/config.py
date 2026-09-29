"""Finding the customer's data files on disk.

RYA's original scripts had file paths from their own laptops written into them
(/Users/liam/...), so they only ran there. This module replaces all of that: give it one
folder (by default the repo's ``data/``, see ``settings.py``) and it finds each file by its
name, however the sub-folders inside are arranged. That's why a straight Box download works
as well as the repo layout.
"""
from __future__ import annotations

import glob
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import settings

# Kept here as well because other modules and the tests import it from config.
REPO_DIR = settings.REPO_DIR


def default_project_root() -> str:
    """The data folder to use when a command isn't given one.

    Comes from settings.py, so it follows the usual order: the SCRP_DATA environment
    variable, then your local_settings.py, then the repo's own ./data folder.
    """
    return str(settings.DATA_DIR)


class ProjectFileNotFound(FileNotFoundError):
    """A file or folder the pipeline needs isn't anywhere under the data folder."""


def extract_zip(zip_path: str | os.PathLike, dest: str | os.PathLike) -> Path:
    """Unzip ``zip_path`` into ``dest`` (creating it if needed) and return ``dest``."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    return dest


def find_file(pattern: str, root: str | os.PathLike) -> str:
    """Return the first file called ``pattern`` anywhere under ``root``."""
    matches = glob.glob(os.path.join(str(root), "**", pattern), recursive=True)
    if not matches:
        raise ProjectFileNotFound(f"No file matching '{pattern}' was found under {root}.")
    return matches[0]


def find_dir(dirname: str, root: str | os.PathLike) -> str:
    """Return the first folder called ``dirname`` anywhere under ``root``.

    Names are compared loosely, so Box's "Instance data" matches RYA's "instance_data".
    """
    def norm(name: str) -> str:
        return name.lower().replace(" ", "_").replace("-", "_")

    for dirpath, dirnames, _filenames in os.walk(root):
        if norm(os.path.basename(dirpath)) == norm(dirname):
            return dirpath
    raise ProjectFileNotFound(f"No directory named '{dirname}' was found under {root}.")


@dataclass
class ProjectPaths:
    """Where each of the project's files was found.

    Only the matrices file is required, because every command uses it. The rest are optional
    so that, for example, ``explore`` still works on a folder without the ONNX model. A command
    that does need one of them calls ``require()`` and gets a clear message if it's missing.
    """

    root: str
    pmatrices_path: str
    onnx_path: Optional[str] = None
    metadata_path: Optional[str] = None
    instance_dir: Optional[str] = None
    pre_csv_path: Optional[str] = None
    post_csv_path: Optional[str] = None

    @property
    def have_ry25_data(self) -> bool:
        """True when both RY25 survey files (before and after) were found."""
        return self.pre_csv_path is not None and self.post_csv_path is not None

    def require(self, *fields: str) -> None:
        """Stop with a readable error if any of the named files weren't found."""
        missing = [f for f in fields if getattr(self, f, None) is None]
        if missing:
            raise ProjectFileNotFound(
                f"This command needs {', '.join(missing)}, which could not be found under "
                f"'{self.root}'. Check --project-root points at the full unzipped project folder."
            )


def resolve_project_paths(project_root: str | os.PathLike, zip_path: Optional[str | os.PathLike] = None) -> ProjectPaths:
    """Look for every file the pipeline might need under ``project_root``.

    If the folder doesn't exist yet (or is empty) and a zip is given, the zip is unpacked
    there first.
    """
    project_root = str(project_root)
    if zip_path and (not os.path.isdir(project_root) or not os.listdir(project_root)):
        extract_zip(zip_path, project_root)

    if not os.path.isdir(project_root):
        raise ProjectFileNotFound(
            f"PROJECT_ROOT '{project_root}' does not exist. Pass --project-root, or --zip to extract one."
        )

    # The one file every command needs: without it, stop straight away.
    pmatrices_path = find_file("unreliability-matrices.json", project_root)

    # Everything else is looked up quietly and left as None if it isn't there.
    def _optional_file(pattern: str) -> Optional[str]:
        try:
            return find_file(pattern, project_root)
        except ProjectFileNotFound:
            return None

    def _optional_dir(dirname: str) -> Optional[str]:
        try:
            return find_dir(dirname, project_root)
        except ProjectFileNotFound:
            return None

    return ProjectPaths(
        root=project_root,
        pmatrices_path=pmatrices_path,
        onnx_path=_optional_file("EXP002_nn_optimal_epoch.onnx"),   # the trained network
        metadata_path=_optional_file("EXP002.json"),                 # its settings + scaling values
        instance_dir=_optional_dir("instance_data"),                 # the training data
        pre_csv_path=_optional_file("RY25_PreMay10.csv"),            # real survey, before
        post_csv_path=_optional_file("RY25_PostMay10.csv"),          # real survey, after
    )
