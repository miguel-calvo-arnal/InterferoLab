"""API-level tests for the pybind11 module ``analysis_backend``.

Covers the public surface: get_reconstruction_methods(), the run_analysis()
result contract, argument validation, and progress-callback behaviour
(monotonicity and robustness against raising callbacks).
"""

from __future__ import annotations

import os

import helpers_backend as hb
import pytest

# The C++ backend (.so) may not be compiled (e.g. on a clean clone or CI):
# skip cleanly instead of blowing up collection with ImportError.
analysis_backend = pytest.importorskip("analysis_backend")


@pytest.fixture
def png_dataset(in_tmp_cwd):
    """Small synthetic PNG dataset written under the temporary cwd."""
    heights, low_mask, high_mask = hb.two_plateau_heights()
    hb.write_png_dataset("dataset", heights)
    return {"folder": "dataset", "heights": heights, "low": low_mask, "high": high_mask}


def test_get_reconstruction_methods_registry():
    """The registry exposes exactly 4 methods with ids 1..4 and non-empty metadata."""
    methods = analysis_backend.get_reconstruction_methods()
    assert isinstance(methods, list)
    assert len(methods) == 4
    assert [m["id"] for m in methods] == [1, 2, 3, 4]
    for m in methods:
        assert set(m.keys()) == {"id", "name", "description"}
        assert isinstance(m["name"], str) and m["name"].strip()
        assert isinstance(m["description"], str) and m["description"].strip()


def test_invalid_method_id_raises(png_dataset):
    """An unknown method id fails fast with a clear RuntimeError, before any output."""
    with pytest.raises(RuntimeError, match="[Uu]nknown reconstruction method id 99"):
        hb.run_backend(png_dataset["folder"], name="bad_method", method=99)
    assert not os.path.exists(os.path.join("output", "bad_method"))


def test_missing_folder_raises(in_tmp_cwd):
    """A non-existent dataset folder raises a clear RuntimeError."""
    with pytest.raises(RuntimeError, match="does not exist"):
        hb.run_backend("no_such_folder_anywhere")


def test_result_dict_contract_on_normal_run(png_dataset):
    """A successful run returns the documented dict and writes the heightmap.

    With name="" the output folder is named after the dataset folder basename.
    """
    result = hb.run_backend(png_dataset["folder"], name="", method=1)

    assert set(result.keys()) == {"output_folder", "heightmap", "cancelled"}
    assert result["cancelled"] is False
    assert os.path.isdir(result["output_folder"])
    assert os.path.basename(result["output_folder"]) == "dataset"
    assert os.path.isfile(result["heightmap"])
    assert os.path.dirname(result["heightmap"]) == result["output_folder"]
    assert result["heightmap"].endswith("dataset_height.npy")


def test_progress_callback_monotonic(png_dataset):
    """Progress is reported with percent in [0, 100], non-decreasing, ending at 100."""
    calls: list[tuple[int, str, int, int]] = []

    def progress(percent, stage, done, total):
        calls.append((percent, stage, done, total))

    result = hb.run_backend(png_dataset["folder"], name="prog", method=2, progress_cb=progress)
    assert result["cancelled"] is False

    assert len(calls) >= 2
    fractions = [c[0] / 100.0 for c in calls]
    assert all(0.0 <= f <= 1.0 for f in fractions)
    assert all(b >= a for a, b in zip(fractions[:-1], fractions[1:], strict=True))
    assert fractions[-1] == 1.0
    for _percent, stage, done, total in calls:
        assert isinstance(stage, str) and stage
        assert 0 <= done <= total
        assert total == hb.NZ


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_raising_progress_callback_does_not_abort(png_dataset):
    """A progress callback that raises must not kill the analysis.

    The backend discards the exception as unraisable (progress is cosmetic)
    and the run still completes successfully.
    """
    n_calls = {"count": 0}

    def bad_progress(percent, stage, done, total):
        n_calls["count"] += 1
        raise ValueError("boom from progress callback")

    result = hb.run_backend(
        png_dataset["folder"], name="raising_cb", method=1, progress_cb=bad_progress
    )
    assert n_calls["count"] >= 1
    assert result["cancelled"] is False
    assert os.path.isfile(result["heightmap"])
