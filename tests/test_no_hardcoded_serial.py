"""Guard: the real piezo serial number must never appear in a versioned file.

The serial itself is NOT in this file (that would be the leak the guard is
meant to prevent, and it is what made the batch-5 version flag itself): only
its SHA-256 is kept.  Every run of 9 digits found in a file that git tracks
or that is staged/untracked-but-not-ignored (so a file about to be committed
is checked too, not only after the commit) is hashed and compared.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess

import pytest

# sha256 of the real controller serial (9 ASCII digits).  Recompute with
#   python -c "import hashlib;print(hashlib.sha256(b'<serial>').hexdigest())"
_REAL_SERIAL_SHA256 = "0068217c31a8b1194c75db212463670db3c7bf42c1d5f315ca21ebf06f054897"
_NINE_DIGITS = re.compile(r"(?<!\d)\d{9}(?!\d)")


def _candidate_files(project_root: str) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
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
        files = _candidate_files(project_root)
    except subprocess.CalledProcessError:
        pytest.skip("not inside a git checkout")

    offenders: list[str] = []
    for rel_path in files:
        try:
            with open(f"{project_root}/{rel_path}", encoding="utf-8") as fh:
                text = fh.read()
        except (OSError, UnicodeDecodeError):
            continue
        for run in set(_NINE_DIGITS.findall(text)):
            if hashlib.sha256(run.encode()).hexdigest() == _REAL_SERIAL_SHA256:
                offenders.append(rel_path)
                break

    assert offenders == [], (
        f"The real piezo serial is versioned (or about to be) in: {offenders}. "
        "Describe it instead of quoting it."
    )
