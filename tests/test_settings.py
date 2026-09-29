import importlib
import os
import subprocess
import sys
from pathlib import Path

import scrp_toolkit.settings as settings


def test_defaults_point_inside_the_repo():
    importlib.reload(settings)
    if not os.environ.get("SCRP_DATA"):
        assert settings.DATA_DIR == settings.REPO_DIR / "data"
    assert settings.EXAMPLE_ITEM == "ry9" and settings.SPLIT_SEED == 42


def test_local_settings_file_overrides_defaults(tmp_path, monkeypatch):
    (tmp_path / "local_settings.py").write_text('DATA_DIR = "/team/data"\nRUN_TRAINING_DEMO = True\nhelper = 1\n')
    monkeypatch.setattr(settings, "REPO_DIR", tmp_path)
    try:
        settings._load_local_overrides()
        assert settings.DATA_DIR == Path("/team/data")          # strings become paths
        assert settings.RUN_TRAINING_DEMO is True
        assert not hasattr(settings, "helper")                  # only UPPER_CASE names are settings
    finally:
        monkeypatch.undo()
        importlib.reload(settings)


def test_environment_variable_wins(tmp_path):
    code = "from scrp_toolkit import settings; print(settings.DATA_DIR)"
    env = {**os.environ, "SCRP_DATA": str(tmp_path)}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                         cwd=Path(settings.__file__).parent.parent)
    assert out.stdout.strip() == str(tmp_path)
