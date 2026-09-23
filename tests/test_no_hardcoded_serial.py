"""Guard: the real piezo serial number must never appear in a versioned file.

Batch 5 removed the hardcoded fallback serial from `views/AcquisitionPanel.py`
(U9) and the implementer's own report claimed `git grep` was clean -- but the
digits had been typed straight into three lines of `CHANGELOG.md` (a tracked
file that reaches the project's public mirror), which is exactly the leak U9
was about (review B2, finding H1). This test walks the files git actually
tracks (not just the working tree, and not relying on `git grep`'s own
pathspec/binary handling) and fails if the real serial shows up in any of
them, so it cannot quietly come back in a future commit message, comment or
changelog entry.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

# The real controller serial (retired from the code in batch 5). Written
# once, here, as the thing being searched for -- never as a literal string
# anywhere else in the repository.
_REAL_SERIAL = "125056199"


def _tracked_files(project_root: str) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in out.stdout.splitlines() if line]


def test_real_piezo_serial_never_appears_in_a_versioned_file(project_root):
    if shutil.which("git") is None:
        pytest.skip("git is not available in this environment")
    try:
        files = _tracked_files(project_root)
    except subprocess.CalledProcessError:
        pytest.skip("not inside a git checkout")

    offenders: list[str] = []
    for rel_path in files:
        path = f"{project_root}/{rel_path}"
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        except (OSError, UnicodeDecodeError):
            continue  # binary or unreadable: not a place a changelog line lives
        if _REAL_SERIAL in text:
            offenders.append(rel_path)

    assert offenders == [], (
        f"The real piezo serial is still versioned in: {offenders}. "
        "It must never be written verbatim in a tracked file (U9) -- "
        "describe it instead of quoting it."
    )
