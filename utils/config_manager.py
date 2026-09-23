# utils/config_manager.py
"""
Persistent application configuration (app_config.json).

Robustness guarantees:
- The config path is anchored to the application directory, not the cwd.
- Loaded values are sanitized against a per-key type/range schema; invalid
  entries are discarded with a logged warning instead of crashing the UI.
- A corrupt JSON file is renamed to app_config.json.bak (never silently
  overwritten) and an empty config is returned.
- Saving is atomic: write to a temporary file, then os.replace().
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile

from utils.session_log import app_base_dir

log = logging.getLogger(__name__)

_CONFIG_FILE = "app_config.json"

# Schema for known config keys: expected type plus optional range/choices.
# Unknown keys are passed through unchanged (forward/backward compatibility).
_SCHEMA: dict[str, dict[str, dict]] = {
    "acq": {
        "piezo_serial": {"type": str},
        "start": {"type": float, "min": 0.0, "max": 100.0},
        "end": {"type": float, "min": 0.0, "max": 100.0},
        "step": {"type": float, "min": 0.0, "max": 1.0},
        "manual_z": {"type": float, "min": 0.0, "max": 100.0},
        "keyboard_step_um": {"type": float, "min": 0.001, "max": 5.0},
        "exposure": {"type": float, "min": 0.01, "max": 10000.0},
        "timeout": {"type": float, "min": 0.5, "max": 10.0},
        "format": {"type": str, "choices": {"bin12", "tiff", "png"}},
        "color_mode": {"type": str, "choices": {"mono", "color", "mono_superpixel"}},
        "preview_interval": {"type": int, "min": 200, "max": 10000},
    },
    "proc": {
        "method_id": {"type": int, "min": 1},
        "cb_r": {"type": bool},
        "cb_g": {"type": bool},
        "cb_b": {"type": bool},
        "wr": {"type": float, "min": 0.0, "max": 100.0},
        "wg": {"type": float, "min": 0.0, "max": 100.0},
        "wb": {"type": float, "min": 0.0, "max": 100.0},
        "pixel_plots": {"type": bool},
    },
}


def config_path() -> str:
    """Absolute path of the config file, anchored to the application dir."""
    return os.path.join(app_base_dir(), _CONFIG_FILE)


def _coerce(value, spec: dict):
    """
    Validate `value` against a schema `spec`.

    Returns the (possibly coerced) value, or raises ValueError/TypeError
    if the value is invalid.
    """
    expected = spec["type"]

    if expected is bool:
        if not isinstance(value, bool):
            raise TypeError(f"expected bool, got {type(value).__name__}")
        return value

    if expected is int:
        # Accept floats with an integral value, but never bools.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"expected int, got {type(value).__name__}")
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"expected integer value, got {value}")
        value = int(value)
    elif expected is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"expected number, got {type(value).__name__}")
        value = float(value)
    elif expected is str:
        if not isinstance(value, str):
            raise TypeError(f"expected str, got {type(value).__name__}")
    else:  # pragma: no cover - schema misuse
        raise TypeError(f"unsupported schema type {expected!r}")

    if "choices" in spec and value not in spec["choices"]:
        raise ValueError(f"value {value!r} not in {sorted(spec['choices'])}")
    if "min" in spec and value < spec["min"]:
        raise ValueError(f"value {value!r} below minimum {spec['min']}")
    if "max" in spec and value > spec["max"]:
        raise ValueError(f"value {value!r} above maximum {spec['max']}")

    return value


def _sanitize(cfg: dict) -> dict:
    """
    Validate a loaded config dict against _SCHEMA.

    Invalid sections or values are dropped with a logged warning; unknown
    sections/keys are kept as-is. Always returns a dict safe to feed to
    the panels' apply_saved_config().
    """
    if not isinstance(cfg, dict):
        log.warning("Config root is not an object (got %s); ignoring it.", type(cfg).__name__)
        return {}

    clean: dict = {}
    for section, values in cfg.items():
        schema = _SCHEMA.get(section)
        if schema is None:
            clean[section] = values
            continue
        if not isinstance(values, dict):
            log.warning(
                "Config section %r is not an object (got %s); discarding it.",
                section,
                type(values).__name__,
            )
            continue
        clean_section: dict = {}
        for key, value in values.items():
            spec = schema.get(key)
            if spec is None:
                clean_section[key] = value
                continue
            try:
                clean_section[key] = _coerce(value, spec)
            except (TypeError, ValueError) as e:
                log.warning("Discarding invalid config value %s.%s=%r: %s", section, key, value, e)
        clean[section] = clean_section

    return clean


def load_config() -> dict:
    """
    Load and sanitize the persistent config.

    Returns an empty dict if the file is missing. A corrupt JSON file is
    renamed to <file>.bak (so it is never silently overwritten on exit)
    and an empty dict is returned.
    """
    path = config_path()
    if not os.path.isfile(path):
        return {}

    try:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, ValueError) as e:
        log.warning("Could not parse config file %s: %s", path, e)
        backup = path + ".bak"
        try:
            os.replace(path, backup)
            log.warning("Corrupt config file moved to %s", backup)
        except OSError as move_err:
            log.warning("Could not back up corrupt config file: %s", move_err)
        return {}

    return _sanitize(cfg)


def save_config(cfg: dict) -> None:
    """
    Persist config to JSON atomically (temp file + os.replace).

    Errors are logged but never raised (saving happens on app close).
    """
    path = config_path()
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(
            prefix=_CONFIG_FILE + ".",
            suffix=".tmp",
            dir=os.path.dirname(path),
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        os.replace(tmp_path, path)
        tmp_path = None
    except (OSError, TypeError, ValueError) as e:
        log.warning("Could not save config file %s: %s", path, e)
    finally:
        if tmp_path is not None:
            with contextlib.suppress(OSError):
                os.remove(tmp_path)
