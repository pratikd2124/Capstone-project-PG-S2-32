"""End-to-end on a fake project with the real file layout: every CLI command runs, and the
RYA baseline gate (SHA-1 files + EXP002 scaling) passes on matching inputs and blocks
anything else."""
import json
import shutil

import pytest

from scrp_toolkit.baseline import MURAT_REFERENCE, _file_map, sha1_of
from scrp_toolkit.cli import main
from scrp_toolkit.config import resolve_project_paths

from . import synthetic_project


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    pytest.importorskip("onnx")
    return synthetic_project.build(tmp_path_factory.mktemp("project"))


def _template(project, tmp_path):
    """A baseline template pinned to the fake project's files, like rya_baseline.json is to RYA's."""
    files = {n: sha1_of(p) for n, p in _file_map(resolve_project_paths(project)).items() if p}
    t = {"files": files, "check_scaling": True,
         "explore": {"n_items": 0, "top_asymm": {"ry9": 0}, "atol": 1e-6},
         "ry9_example": {"question": "ry9", "x": [40, 52, 7, 14], "y": [9, 17, 54, 12], "atol": 1e-6}}
    path = tmp_path / "template.json"
    path.write_text(json.dumps(t))
    return path


@pytest.mark.parametrize("argv", [
    ["explore", "--top-n", "3"],
    ["score"],
    ["train", "--quick-demo"],
    ["compare", "--quick-demo"],
    ["gen-edge-cases", "--n-per-case", "2"],
])
def test_every_command_runs(project, tmp_path, argv, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(argv + ["--project-root", str(project)]) in (0, 1)  # gen-edge-cases returns 1 on the known NaN finding


def test_validate_capture_then_check(project, tmp_path):
    ka = tmp_path / "ka.json"
    ka.write_text(json.dumps([{"question": "ry9", "x": [40, 52, 7, 14], "y": [9, 17, 54, 12],
                               "expected_shape": 0, "expected_rate": 0}]))
    assert main(["validate", "--project-root", str(project), "--known-answers", str(ka)]) == 1
    assert main(["validate", "--project-root", str(project), "--known-answers", str(ka), "--capture"]) == 0
    assert main(["validate", "--project-root", str(project), "--known-answers", str(ka)]) == 0


def test_capture_then_check_passes_and_tampering_fails(project, tmp_path, capsys):
    template = _template(project, tmp_path)
    out = tmp_path / "base.json"
    report = tmp_path / "report.csv"
    assert main(["baseline", "--project-root", str(project), "--baseline", str(template), "--capture", str(out)]) == 0
    captured = json.loads(out.read_text())
    assert {"files", "explore", "model", "ry9_example", "case_study", "dataset"} <= set(captured)
    assert all(not it["question"].startswith("chs") for it in captured["case_study"]["items"])

    assert main(["baseline", "--project-root", str(project), "--baseline", str(out), "--report", str(report)]) == 0
    printed = capsys.readouterr().out
    assert "BASELINE MATCHES" in printed and "[PASS] scaling" in printed and "[PASS] files" in printed

    captured["ry9_example"]["shape"] += 0.01
    captured["case_study"]["items"][0]["rate_hat"] *= 1.01
    out.write_text(json.dumps(captured))
    assert main(["baseline", "--project-root", str(project), "--baseline", str(out), "--report", str(report)]) == 1
    printed = capsys.readouterr().out
    assert "BASELINE MISMATCH" in printed and "shape" in printed and "rate_hat" in printed


def test_capture_refuses_changed_file(project, tmp_path, capsys):
    template = _template(project, tmp_path)
    copy = tmp_path / "copy"
    shutil.copytree(project, copy)
    pm = next(copy.rglob("unreliability-matrices.json"))
    pm.write_text(pm.read_text().replace("0.", "0.0", 1))  # any edit changes the SHA-1
    assert main(["baseline", "--project-root", str(copy), "--baseline", str(template), "--capture", str(tmp_path / "x.json")]) == 1
    assert "Not capturing" in capsys.readouterr().out
    assert not (tmp_path / "x.json").exists()


def test_capture_refuses_non_rya_preprocessing(project, tmp_path, capsys):
    """If EXP002.json's scaling doesn't match the rebuilt dataset (e.g. rate not rescaled), no capture."""
    copy = tmp_path / "copy"
    shutil.copytree(project, copy)
    meta_path = next(copy.rglob("EXP002.json"))
    meta = json.loads(meta_path.read_text())
    mm = json.loads(meta["min_max_df"])
    rate_idx = next(k for k, v in mm["column"].items() if v == "rate")
    mm["max"][rate_idx] *= 1.5
    meta["min_max_df"] = json.dumps(mm)
    meta_path.write_text(json.dumps(meta))
    template = _template(copy, tmp_path)  # files pinned to the edited copy -> only scaling can fail
    assert main(["baseline", "--project-root", str(copy), "--baseline", str(template), "--capture", str(tmp_path / "x.json")]) == 1
    assert "rate.max" in capsys.readouterr().out


def test_shipped_baselines_fail_cleanly_on_wrong_data(project, tmp_path):
    """Against fake data, the shipped RYA baseline and Murat reference must FAIL, not crash."""
    assert main(["baseline", "--project-root", str(project), "--report", str(tmp_path / "r.csv")]) == 1
    assert main(["baseline", "--project-root", str(project), "--report", str(tmp_path / "r.csv"),
                 "--baseline", str(MURAT_REFERENCE)]) == 1
