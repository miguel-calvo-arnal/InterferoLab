"""
Simulator profile: every hardware figure with its origin.

A profile is a TOML file (see sim/profiles/default.toml).  A figure is an
inline table ``{value = ..., origin = "...", source = "..."}``; anything
else is a plain simulator setting.  User/measured profiles are merged OVER
the default one, so a lab profile only needs the figures it measured.
"""

from __future__ import annotations

import copy
import os
import tomllib
from typing import Any

SIM_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROFILE = os.path.join(SIM_DIR, "profiles", "default.toml")

#: Allowed origins of a hardware figure.
ORIGINS = ("published", "measured", "derived", "estimated")


class ProfileError(ValueError):
    """The profile file is malformed (missing origin/source, bad origin...)."""


def _is_figure(node: Any) -> bool:
    return isinstance(node, dict) and "value" in node


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in over.items():
        if (
            isinstance(val, dict)
            and not _is_figure(val)
            and isinstance(out.get(key), dict)
            and not _is_figure(out.get(key))
        ):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def validate(tree: dict, path: str = "") -> None:
    """Raise ProfileError if any figure lacks a valid origin or a source."""
    for key, node in tree.items():
        here = f"{path}.{key}" if path else key
        if _is_figure(node):
            origin = node.get("origin")
            if origin not in ORIGINS:
                raise ProfileError(f"{here}: origin {origin!r} is not one of {ORIGINS}")
            if not str(node.get("source", "")).strip():
                raise ProfileError(f"{here}: a figure needs a non-empty 'source'")
        elif isinstance(node, dict):
            validate(node, here)


class Profile:
    """Read-only view of a merged profile tree."""

    def __init__(self, tree: dict, files: list[str]) -> None:
        validate(tree)
        self._tree = tree
        self.files = list(files)

    # -- access ----------------------------------------------------------
    def _node(self, dotted: str) -> Any:
        node: Any = self._tree
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                raise KeyError(f"profile has no key {dotted!r}")
            node = node[part]
        return node

    def get(self, dotted: str, default: Any = None) -> Any:
        """Value of a figure or setting (the ``value`` of a figure table)."""
        try:
            node = self._node(dotted)
        except KeyError:
            return default
        return node["value"] if _is_figure(node) else node

    def __getitem__(self, dotted: str) -> Any:
        node = self._node(dotted)
        return node["value"] if _is_figure(node) else node

    def origin(self, dotted: str) -> str | None:
        node = self._node(dotted)
        return node.get("origin") if _is_figure(node) else None

    def figures(self) -> list[tuple[str, Any, str, str]]:
        """Flat list of (key, value, origin, source) for every figure."""
        out: list[tuple[str, Any, str, str]] = []

        def walk(tree: dict, path: str) -> None:
            for key, node in tree.items():
                here = f"{path}.{key}" if path else key
                if _is_figure(node):
                    out.append((here, node["value"], node["origin"], node.get("source", "")))
                elif isinstance(node, dict):
                    walk(node, here)

        walk(self._tree, "")
        return out

    def as_dict(self) -> dict:
        return copy.deepcopy(self._tree)

    def with_overrides(self, overrides: dict) -> Profile:
        """New profile with a nested dict of settings/figures merged on top."""
        return Profile(_deep_merge(self._tree, overrides), self.files + ["<overrides>"])


def load_profile(path: str | None = None, overrides: dict | None = None) -> Profile:
    """Load the default profile, merge ``path`` over it, then ``overrides``."""
    with open(DEFAULT_PROFILE, "rb") as fh:
        tree = tomllib.load(fh)
    files = [DEFAULT_PROFILE]
    if path and os.path.abspath(path) != DEFAULT_PROFILE:
        with open(path, "rb") as fh:
            tree = _deep_merge(tree, tomllib.load(fh))
        files.append(os.path.abspath(path))
    prof = Profile(tree, files)
    if overrides:
        prof = prof.with_overrides(overrides)
    return prof


def toml_figure(value: Any, origin: str, source: str) -> str:
    """Render one figure as a TOML inline table (used by the lab probe)."""
    if origin not in ORIGINS:
        raise ProfileError(f"bad origin {origin!r}")
    if isinstance(value, bool):
        v = "true" if value else "false"
    elif isinstance(value, (int, float)):
        v = repr(float(value)) if isinstance(value, float) else str(value)
    else:
        v = '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'
    s = source.replace("\\", "\\\\").replace('"', '\\"')
    return f'{{ value = {v}, origin = "{origin}", source = "{s}" }}'
