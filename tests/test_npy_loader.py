"""Tests for services/npy_loader.py: dataset listing and heightmap loading."""

from __future__ import annotations

import os

import numpy as np
import pytest

from services import npy_loader
from utils.analysis_paths import height_filename


def _make_dataset(root, name: str, shape=(6, 8)) -> np.ndarray:
    """Create output/<name>/<name>_height.npy under *root*; return the array."""
    ds_dir = root / "output" / name
    ds_dir.mkdir(parents=True)
    arr = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    np.save(ds_dir / height_filename(name), arr)
    return arr


def test_get_base_folder_tracks_cwd(in_tmp_cwd):
    """The base folder is ./output resolved at call time from the cwd."""
    assert npy_loader.get_base_folder() == os.path.join(str(in_tmp_cwd), "output")


def test_list_datasets_sorted(in_tmp_cwd):
    """Dataset folders under ./output are listed sorted by name."""
    _make_dataset(in_tmp_cwd, "zeta")
    _make_dataset(in_tmp_cwd, "alpha")
    _make_dataset(in_tmp_cwd, "mid")
    assert npy_loader.list_datasets() == ["alpha", "mid", "zeta"]


def test_list_datasets_missing_output_returns_empty(in_tmp_cwd):
    """A missing ./output folder yields an empty list (soft failure)."""
    assert npy_loader.list_datasets() == []


def test_list_datasets_ignores_plain_files(in_tmp_cwd):
    """Plain files inside ./output are not reported as datasets."""
    _make_dataset(in_tmp_cwd, "real")
    (in_tmp_cwd / "output" / "stray.npy").write_bytes(b"\x00")
    assert npy_loader.list_datasets() == ["real"]


def test_load_height_returns_eager_array_and_path(in_tmp_cwd):
    """load_height returns an eagerly loaded array and the absolute path.

    Deliberately NOT a memmap: a live mmap would keep the file locked on
    Windows, breaking the backend's delete+rename publish on re-analysis.
    """
    expected = _make_dataset(in_tmp_cwd, "ds1", shape=(5, 7))
    arr, path = npy_loader.load_height("ds1")
    assert not isinstance(arr, np.memmap)
    assert arr.shape == (5, 7)
    np.testing.assert_array_equal(np.asarray(arr), expected)
    assert path == os.path.join(str(in_tmp_cwd), "output", "ds1", "ds1_height.npy")
    assert os.path.isabs(path)


def test_load_height_missing_dataset_raises(in_tmp_cwd):
    """A nonexistent dataset folder raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Dataset folder not found"):
        npy_loader.load_height("no_such_dataset")


def test_load_height_folder_without_height_file_raises(in_tmp_cwd):
    """A dataset folder without <name>_height.npy raises FileNotFoundError."""
    (in_tmp_cwd / "output" / "empty_ds").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="Height file not found"):
        npy_loader.load_height("empty_ds")


def test_load_height_requires_exact_naming_convention(in_tmp_cwd):
    """A height file not following <dataset>_height.npy is not picked up."""
    ds_dir = in_tmp_cwd / "output" / "ds2"
    ds_dir.mkdir(parents=True)
    np.save(ds_dir / "wrong_name.npy", np.zeros((2, 2)))
    with pytest.raises(FileNotFoundError, match="Height file not found"):
        npy_loader.load_height("ds2")
