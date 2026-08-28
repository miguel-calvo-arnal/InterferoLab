#!/usr/bin/env bash
# =====================================================================
# build_release.sh — Linux: C++ backend -> PyInstaller -> releases/*.tar.gz
#
# Runs the whole release chain in one go:
#   1. Configure + build the C++ backend (Release) with the project venv.
#   2. Copy the resulting .so next to backend/analysis/ (where the spec
#      globs it first; see interferolab.spec).
#   3. Import-check the module BEFORE the slow PyInstaller run, so an ABI
#      or Python-version mismatch fails in seconds instead of minutes.
#   4. Freeze with interferolab.spec into dist/.
#   5. Strip runtime residue and archive dist/InterferoLab into releases/.
#
# Usage:  scripts/build_release.sh [--skip-build] [--native] [--force]
#   --skip-build  reuse the .so already in backend/analysis/ (no cmake)
#   --native      -march=native; FASTER but NOT portable, do not distribute
#   --force       overwrite the release archive if it already exists
#
# The Windows counterpart is scripts/build_release.ps1.
# =====================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
SRC="$ROOT/backend/analysis"
# Linux-only build directory: backend/analysis/build/ belongs to the Windows
# VM (MSVC cache + the vcpkg DLLs interferolab.spec bundles). Sharing a single
# one makes cmake reject the other platform's cache.
BUILD="$SRC/build-linux"
NAME="InterferoLab"
STAMP="$(date +%Y-%m-%d)"
ARCHIVE="$ROOT/releases/$NAME-$STAMP-linux-x64.tar.gz"

SKIP_BUILD=0; NATIVE=0; FORCE=0
for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=1 ;;
        --native)     NATIVE=1 ;;
        --force)      FORCE=1 ;;
        -h|--help)    sed -n '2,22p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "Unknown option: $arg (use --help)" >&2; exit 2 ;;
    esac
done

step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die()  { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------
step "0/5  Preflight checks"
# ---------------------------------------------------------------------
[ -x "$PY" ]                  || die "no venv in .venv/ (see README, Getting started)"
[ -f "$ROOT/interferolab.spec" ] || die "interferolab.spec is missing"
command -v cmake >/dev/null    || die "cmake is not installed"
[ -e "$ARCHIVE" ] && [ $FORCE -eq 0 ] && \
    die "$ARCHIVE already exists (use --force to overwrite it)"
echo "  project : $ROOT"
echo "  python  : $("$PY" -V)"
echo "  target  : $ARCHIVE"

# ---------------------------------------------------------------------
if [ $SKIP_BUILD -eq 0 ]; then
    step "1/5  Building the C++ backend (Release)"
    CMAKE_ARGS=(
        -S "$SRC" -B "$BUILD"
        -DPython3_EXECUTABLE="$PY"
        -DCMAKE_BUILD_TYPE=Release
        -Dpybind11_DIR="$("$PY" -m pybind11 --cmakedir)"
    )
    # Portable by default (-march=x86-64-v2): -march=native crashes with
    # SIGILL on CPUs without AVX2, so it is opt-in only.
    [ $NATIVE -eq 1 ] && CMAKE_ARGS+=(-DINTERFEROLAB_NATIVE_ARCH=ON) \
                      && echo "  WARNING: -march=native binary, do NOT distribute"
    cmake "${CMAKE_ARGS[@]}"
    cmake --build "$BUILD" --config Release -j"$(nproc)"

    step "2/5  Copying the .so to backend/analysis/"
    # Copy (not move): the spec globs backend/analysis/ first, and keeping
    # build/ intact makes an incremental rebuild cheap.
    shopt -s nullglob
    built=("$BUILD"/analysis_backend*.so)
    shopt -u nullglob
    [ ${#built[@]} -gt 0 ] || die "cmake produced no analysis_backend*.so in $BUILD"
    rm -f "$SRC"/analysis_backend*.so
    cp -v "${built[@]}" "$SRC/"
else
    step "1-2/5  Skipped (--skip-build): reusing the existing .so"
fi

# ---------------------------------------------------------------------
step "3/5  Verifying that the module imports"
# ---------------------------------------------------------------------
"$PY" - "$SRC" <<'PYCHECK' || die "the compiled backend does NOT import; aborting before PyInstaller"
import sys
sys.path.insert(0, sys.argv[1])
import analysis_backend as a
ids = [m["id"] for m in a.get_reconstruction_methods()]
print(f"  analysis_backend OK - reconstruction methods: {ids}")
PYCHECK

# ---------------------------------------------------------------------
step "4/5  Freezing with PyInstaller"
# ---------------------------------------------------------------------
rm -rf "$ROOT/dist/$NAME"          # avoid dragging leftovers from a previous build
cd "$ROOT"
"$PY" -m PyInstaller interferolab.spec --noconfirm
[ -d "$ROOT/dist/$NAME" ] || die "PyInstaller did not produce dist/$NAME"

# The app writes app_config.json and logs/ next to the executable on startup;
# if someone ran it from dist/, that residue must not ship in the release.
rm -f  "$ROOT/dist/$NAME/app_config.json"
rm -rf "$ROOT/dist/$NAME/logs"

# ---------------------------------------------------------------------
step "5/5  Creating the archive in releases/"
# ---------------------------------------------------------------------
mkdir -p "$ROOT/releases"
tar -czf "$ARCHIVE" -C "$ROOT/dist" "$NAME"

printf '\n\033[32mDone\033[0m\n'
printf '  dist/%s : %s\n' "$NAME" "$(du -sh "$ROOT/dist/$NAME" | cut -f1)"
printf '  archive    : %s (%s)\n' "$ARCHIVE" "$(du -h "$ARCHIVE" | cut -f1)"
