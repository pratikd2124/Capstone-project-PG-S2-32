"""score_school reads the survey CSVs once, and gives the same scores whether or not it's handed them."""
import pandas as pd
import pytest

from scrp_toolkit.config import resolve_project_paths
from o4_efficiency.batched import EfficientPredictor as UnreliabilityPredictor
from o4_efficiency.school_scoring import load_survey, pick_case_study_school, score_school

from . import synthetic_project


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    pytest.importorskip("onnx")
    paths = resolve_project_paths(synthetic_project.build(tmp_path_factory.mktemp("project")))
    return paths, UnreliabilityPredictor(paths)


def _count_full_reads(monkeypatch):
    """Patch pandas.read_csv to count reads of whole files (header-only reads don't count)."""
    calls = []
    real = pd.read_csv

    def counting(*args, **kwargs):
        if kwargs.get("nrows") != 0:
            calls.append(args[0])
        return real(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", counting)
    return calls


def test_each_csv_is_read_once(setup, monkeypatch):
    paths, predictor = setup
    calls = _count_full_reads(monkeypatch)
    score_school(predictor, paths)
    assert sorted(calls) == sorted([paths.pre_csv_path, paths.post_csv_path])


def test_given_survey_means_no_reading_and_same_scores(setup, monkeypatch):
    paths, predictor = setup
    expected = score_school(predictor, paths)
    survey = load_survey(paths, predictor.p_matrices)

    calls = _count_full_reads(monkeypatch)
    got = score_school(predictor, paths, survey=survey)
    assert calls == []
    pd.testing.assert_frame_equal(got, expected, check_exact=True)
    assert got.attrs["school"] == expected.attrs["school"]


def test_pick_school_same_with_or_without_survey(setup):
    paths, predictor = setup
    survey = load_survey(paths, predictor.p_matrices)
    assert pick_case_study_school(paths, predictor.p_matrices, survey) == \
        pick_case_study_school(paths, predictor.p_matrices)
