"""Locate the project's data files inside an extracted (or already-unzipped) project folder.

This replaces the Colab-specific "upload a zip / mount Google Drive" cell with a plain,
local-filesystem version: point PROJECT_ROOT at wherever the "Statistical Core Refinement
Project" folder (or its unzipped contents) lives on disk, and everything else is found
automatically regardless of the internal folder structure.
"""
from __future__ import annotations

import glob
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


REPO_DIR = Path(__file__).resolve().parent.parent
"""The repository root (the folder containing README.md)."""


def default_project_root() -> str:
    """Where the customer's data lives unless told otherwise.

    1. the ``SCRP_DATA`` environment variable, if set;
    2. otherwise ``<repo>/data`` -- the folder shipped inside the team git repo.
    """
    return os.environ.get("SCRP_DATA") or str(REPO_DIR / "data")


class ProjectFileNotFound(FileNotFoundError):
    """Raised when a required project file/folder cannot be located under PROJECT_ROOT."""


def extract_zip(zip_path: str | os.PathLike, dest: str | os.PathLike) -> Path:
    """Extract ``zip_path`` into ``dest`` (created if needed) and return ``dest`` as a Path."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    return dest


def find_file(pattern: str, root: str | os.PathLike) -> str:
    """Find a file by name anywhere under ``root``, regardless of folder structure."""
    matches = glob.glob(os.path.join(str(root), "**", pattern), recursive=True)
    if not matches:
        raise ProjectFileNotFound(f"No file matching '{pattern}' was found under {root}.")
    return matches[0]


def find_dir(dirname: str, root: str | os.PathLike) -> str:
    """Find a directory by name anywhere under ``root``, regardless of folder structure."""
    def norm(name: str) -> str:  # "Instance data" (Box download) == "instance_data" (original)
        return name.lower().replace(" ", "_").replace("-", "_")

    for dirpath, dirnames, _filenames in os.walk(root):
        if norm(os.path.basename(dirpath)) == norm(dirname):
            return dirpath
    raise ProjectFileNotFound(f"No directory named '{dirname}' was found under {root}.")


@dataclass
class ProjectPaths:
    """Resolved paths to the project's data files.

    Only ``root`` and ``pmatrices_path`` are guaranteed. Everything else is best-effort: a
    command that needs one of the optional paths and finds it ``None`` should raise its own
    clear error (see ``ProjectPaths.require``) rather than every command failing up-front just
    because, say, the ONNX model isn't needed for that command.
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
        return self.pre_csv_path is not None and self.post_csv_path is not None

    def require(self, *fields: str) -> None:
        """Raise a clear ``ProjectFileNotFound`` if any of the named fields resolved to None."""
        missing = [f for f in fields if getattr(self, f, None) is None]
        if missing:
            raise ProjectFileNotFound(
                f"This command needs {', '.join(missing)}, which could not be found under "
                f"'{self.root}'. Check --project-root points at the full unzipped project folder."
            )


def resolve_project_paths(project_root: str | os.PathLike, zip_path: Optional[str | os.PathLike] = None) -> ProjectPaths:
    """Resolve every file the pipeline might need, starting from ``project_root``.

    If ``project_root`` doesn't exist (or is empty) and ``zip_path`` is given, the zip is
    extracted into ``project_root`` first. Only ``unreliability-matrices.json`` is required
    here (every command needs it); the ONNX model, its metadata, instance_data/, and the RY25
    CSVs are each resolved best-effort so that commands which don't need one of them (e.g.
    ``explore`` never touches the ONNX model) don't fail just because it's missing.
    """
    project_root = str(project_root)
    if zip_path and (not os.path.isdir(project_root) or not os.listdir(project_root)):
        extract_zip(zip_path, project_root)

    if not os.path.isdir(project_root):
        raise ProjectFileNotFound(
            f"PROJECT_ROOT '{project_root}' does not exist. Pass --project-root, or --zip to extract one."
        )

    pmatrices_path = find_file("unreliability-matrices.json", project_root)

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
        onnx_path=_optional_file("EXP002_nn_optimal_epoch.onnx"),
        metadata_path=_optional_file("EXP002.json"),
        instance_dir=_optional_dir("instance_data"),
        pre_csv_path=_optional_file("RY25_PreMay10.csv"),
        post_csv_path=_optional_file("RY25_PostMay10.csv"),
    )
