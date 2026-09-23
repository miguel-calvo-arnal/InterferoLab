"""Tests for utils/config_manager.py — persistent app config load/sanitize/save."""

from __future__ import annotations

import json
import logging
import os

import pytest

from utils import config_manager


@pytest.fixture
def cfg_dir(tmp_path, monkeypatch):
    """Anchor the config file to a temporary directory for the duration of the test."""
    monkeypatch.setattr(config_manager, "app_base_dir", lambda: str(tmp_path))
    return tmp_path


def _cfg_path(cfg_dir):
    return cfg_dir / "app_config.json"


def test_config_path_is_anchored_to_app_dir(cfg_dir):
    """config_path joins the application base dir with app_config.json."""
    assert config_manager.config_path() == str(_cfg_path(cfg_dir))


def test_load_missing_file_returns_empty_dict(cfg_dir):
    """A missing config file yields an empty dict without creating anything."""
    assert config_manager.load_config() == {}
    assert not _cfg_path(cfg_dir).exists()


def test_save_load_roundtrip(cfg_dir):
    """A valid config saved with save_config loads back unchanged."""
    cfg = {
        "acq": {
            "piezo_serial": "PZ123",
            "start": 0.0,
            "end": 100.0,
            "step": 0.05,
            "exposure": 33.3,
            "format": "bin12",
            "color_mode": "mono",
            "preview_interval": 500,
        },
        "proc": {"method_id": 2, "cb_r": True, "cb_g": False, "wr": 25.0, "pixel_plots": False},
    }
    config_manager.save_config(cfg)
    assert config_manager.load_config() == cfg


def test_save_is_atomic_valid_json_no_leftover_tmp(cfg_dir):
    """After save_config the file on disk is valid JSON and no *.tmp file remains."""
    config_manager.save_config({"acq": {"start": 1.5}})
    with open(_cfg_path(cfg_dir), encoding="utf-8") as f:
        assert json.load(f) == {"acq": {"start": 1.5}}
    assert [p.name for p in cfg_dir.iterdir()] == ["app_config.json"]


def test_corrupt_json_returns_empty_and_creates_bak(cfg_dir, caplog):
    """Corrupt JSON yields {}, and the broken file is preserved as .bak instead of being overwritten."""
    _cfg_path(cfg_dir).write_text("{not valid json", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="utils.config_manager"):
        assert config_manager.load_config() == {}
    bak = cfg_dir / "app_config.json.bak"
    assert bak.read_text(encoding="utf-8") == "{not valid json"
    assert not _cfg_path(cfg_dir).exists()
    assert "Could not parse config file" in caplog.text


def test_non_dict_root_returns_empty_with_warning(cfg_dir, caplog):
    """A JSON root that is not an object (e.g. a list) is ignored with a warning."""
    _cfg_path(cfg_dir).write_text("[1, 2, 3]", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="utils.config_manager"):
        assert config_manager.load_config() == {}
    assert "not an object" in caplog.text


def test_mistyped_values_discarded_valid_kept(cfg_dir, caplog):
    """Wrongly typed values are dropped with a logged warning while sibling valid values survive."""
    _cfg_path(cfg_dir).write_text(
        json.dumps({"acq": {"exposure": "fast", "start": 5.0, "piezo_serial": 42}}),
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING, logger="utils.config_manager"):
        cfg = config_manager.load_config()
    assert cfg == {"acq": {"start": 5.0}}
    assert "acq.exposure" in caplog.text
    assert "acq.piezo_serial" in caplog.text


def test_unknown_keys_and_sections_preserved(cfg_dir):
    """Unknown keys (legacy method_index) and whole unknown sections pass through unchanged."""
    raw = {
        "proc": {"method_index": 1, "method_id": 3},
        "future_section": {"anything": [1, 2, 3]},
    }
    _cfg_path(cfg_dir).write_text(json.dumps(raw), encoding="utf-8")
    assert config_manager.load_config() == raw


def test_out_of_range_values_discarded(cfg_dir, caplog):
    """Values outside their schema min/max bounds are dropped with a warning."""
    raw = {"acq": {"timeout": 99.0, "preview_interval": 100, "step": 0.5}}
    _cfg_path(cfg_dir).write_text(json.dumps(raw), encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="utils.config_manager"):
        cfg = config_manager.load_config()
    assert cfg == {"acq": {"step": 0.5}}
    assert "acq.timeout" in caplog.text
    assert "acq.preview_interval" in caplog.text


def test_choices_enforced(cfg_dir, caplog):
    """A string outside the allowed choices set is discarded."""
    _cfg_path(cfg_dir).write_text(
        json.dumps({"acq": {"format": "jpeg", "color_mode": "color"}}), encoding="utf-8"
    )
    with caplog.at_level(logging.WARNING, logger="utils.config_manager"):
        cfg = config_manager.load_config()
    assert cfg == {"acq": {"color_mode": "color"}}


def test_color_mode_accepts_mono_superpixel(cfg_dir):
    """The 'Mono (superpixel)' acquisition mode is a valid persisted color_mode."""
    _cfg_path(cfg_dir).write_text(
        json.dumps({"acq": {"color_mode": "mono_superpixel"}}), encoding="utf-8"
    )
    assert config_manager.load_config() == {"acq": {"color_mode": "mono_superpixel"}}


def test_keyboard_step_um_accepted_in_range_rejected_outside(cfg_dir, caplog):
    """Batch 5 (U8): the fine keyboard step is validated like the other
    small µm quantities (e.g. 'step')."""
    raw = {"acq": {"keyboard_step_um": 0.02}}
    _cfg_path(cfg_dir).write_text(json.dumps(raw), encoding="utf-8")
    assert config_manager.load_config() == raw

    raw_bad = {"acq": {"keyboard_step_um": 50.0}}  # above the 5.0 um max
    _cfg_path(cfg_dir).write_text(json.dumps(raw_bad), encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="utils.config_manager"):
        cfg = config_manager.load_config()
    assert cfg == {"acq": {}}
    assert "acq.keyboard_step_um" in caplog.text


def test_color_mode_choices_match_panel_combo(cfg_dir):
    """Schema choices and the AcquisitionPanel combo stay in sync (single source drift guard)."""
    from views.AcquisitionPanel import CHANNEL_MODES

    schema_choices = config_manager._SCHEMA["acq"]["color_mode"]["choices"]
    assert schema_choices == set(CHANNEL_MODES)


def test_bool_is_strict_and_int_rejects_bool(cfg_dir, caplog):
    """The sanitizer neither accepts 1 for a bool key nor True for an int key."""
    raw = {"proc": {"cb_r": 1, "cb_g": True, "method_id": True}}
    _cfg_path(cfg_dir).write_text(json.dumps(raw), encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="utils.config_manager"):
        cfg = config_manager.load_config()
    assert cfg == {"proc": {"cb_g": True}}


def test_int_accepts_integral_float(cfg_dir):
    """An integral float (500.0) for an int key is coerced to int; 2.5 is rejected (leaving its section empty but present)."""
    raw = {"acq": {"preview_interval": 500.0}, "proc": {"method_id": 2.5}}
    _cfg_path(cfg_dir).write_text(json.dumps(raw), encoding="utf-8")
    cfg = config_manager.load_config()
    assert cfg == {"acq": {"preview_interval": 500}, "proc": {}}
    assert isinstance(cfg["acq"]["preview_interval"], int)


def test_non_dict_section_discarded(cfg_dir, caplog):
    """A known section whose value is not an object is discarded entirely."""
    _cfg_path(cfg_dir).write_text(json.dumps({"acq": [1, 2], "proc": {}}), encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="utils.config_manager"):
        cfg = config_manager.load_config()
    assert cfg == {"proc": {}}
    assert "'acq'" in caplog.text


def test_save_to_readonly_dir_does_not_raise(cfg_dir, monkeypatch, caplog):
    """save_config swallows filesystem errors (read-only dir) with a warning instead of raising."""
    ro = cfg_dir / "ro"
    ro.mkdir()
    monkeypatch.setattr(config_manager, "app_base_dir", lambda: str(ro))
    os.chmod(ro, 0o500)
    try:
        with caplog.at_level(logging.WARNING, logger="utils.config_manager"):
            config_manager.save_config({"acq": {"start": 1.0}})
    finally:
        os.chmod(ro, 0o700)
    assert "Could not save config file" in caplog.text
    assert list(ro.iterdir()) == []


def test_save_non_serializable_keeps_previous_file(cfg_dir, caplog):
    """A non-JSON-serializable config logs a warning, leaves the old file intact and no tmp files behind."""
    config_manager.save_config({"acq": {"start": 2.0}})
    with caplog.at_level(logging.WARNING, logger="utils.config_manager"):
        config_manager.save_config({"acq": {"bad": {1, 2, 3}}})
    assert "Could not save config file" in caplog.text
    with open(_cfg_path(cfg_dir), encoding="utf-8") as f:
        assert json.load(f) == {"acq": {"start": 2.0}}
    assert [p.name for p in cfg_dir.iterdir()] == ["app_config.json"]
