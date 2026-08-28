"""Tests for utils/analysis_paths.py — analysis output filename conventions."""

from __future__ import annotations

import os

import pytest

from utils.analysis_paths import (
    HEIGHT_SUFFIX,
    PIXEL_FILE_MARKER,
    PIXEL_FILE_REGEX,
    height_filename,
)


def test_height_filename_appends_suffix():
    """height_filename appends the exact '_height.npy' suffix to the dataset name."""
    assert height_filename("2026-03-20_11-24-12_M1") == "2026-03-20_11-24-12_M1_height.npy"
    assert height_filename("foo") == "foo" + HEIGHT_SUFFIX


def test_pixel_regex_matches_real_cxx_generated_name():
    """The pixel regex parses a real filename produced by the C++ backend (reconstruction.cpp)."""
    name = "2026-03-20_11-24-12_M1_pixel_y1003_x1055.npy"
    m = PIXEL_FILE_REGEX.match(name)
    assert m is not None
    assert (int(m.group(1)), int(m.group(2))) == (1003, 1055)
    assert PIXEL_FILE_MARKER in name


def test_pixel_regex_matches_real_output_files(project_root):
    """Every *_pixel_y*_x*.npy file actually present in output/ matches the regex (skipped if no output data exists)."""
    out_dir = os.path.join(project_root, "output")
    if not os.path.isdir(out_dir):
        pytest.skip("no output/ directory with real analysis data")
    names = [
        f
        for d in os.listdir(out_dir)
        if os.path.isdir(os.path.join(out_dir, d))
        for f in os.listdir(os.path.join(out_dir, d))
        if PIXEL_FILE_MARKER in f and f.endswith(".npy")
    ]
    if not names:
        pytest.skip("no pixel files present in output/")
    for name in names[:50]:
        assert PIXEL_FILE_REGEX.match(name), name


def test_pixel_regex_synthetic_coordinates():
    """Synthetic names with small/large coordinates parse into the right (y, x) groups."""
    cases = {
        "ds_pixel_y12_x7.npy": (12, 7),
        "ds_pixel_y0_x0.npy": (0, 0),
        "a_b_c_pixel_y99999_x123456.npy": (99999, 123456),
    }
    for name, (y, x) in cases.items():
        m = PIXEL_FILE_REGEX.match(name)
        assert m is not None, name
        assert (int(m.group(1)), int(m.group(2))) == (y, x)


def test_pixel_regex_rejects_foreign_files():
    """Files that are not pixel plots (heightmaps, swapped axes, wrong extension, missing digits) do not match."""
    non_matches = [
        "2026-03-20_11-24-12_M1_height.npy",
        "ds_pixel_x7_y12.npy",  # swapped axis order
        "ds_pixel_y12_x7.txt",  # wrong extension
        "ds_pixel_y_x7.npy",  # missing y digits
        "ds_pixel_y12_x.npy",  # missing x digits
        "notes.txt",
        "ds.bin12",
    ]
    for name in non_matches:
        assert PIXEL_FILE_REGEX.match(name) is None, name


def test_marker_is_cheap_prefilter_for_regex():
    """Any name matched by the full regex necessarily contains PIXEL_FILE_MARKER (the marker is a sound pre-filter)."""
    name = "x_pixel_y1_x2.npy"
    assert PIXEL_FILE_REGEX.match(name) and PIXEL_FILE_MARKER in name
    assert PIXEL_FILE_MARKER not in "x_height.npy"
