#!/usr/bin/env bash
# Build the STANDALONE lab timing probe (Linux): dist/lab_timing_probe/lab_timing_probe
# Usage: sim/build_lab_probe.sh [--distpath DIR] [--workpath DIR]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
DIST="$ROOT/dist"; WORK="$ROOT/build/probe"
while [ $# -gt 0 ]; do
    case "$1" in
        --distpath) DIST="$2"; shift 2 ;;
        --workpath) WORK="$2"; shift 2 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done
cd "$ROOT"
"$PY" -m PyInstaller sim/lab_timing_probe.spec --noconfirm --distpath "$DIST" --workpath "$WORK"
echo "Done: $DIST/lab_timing_probe/  (copy the whole folder to the lab PC)"
