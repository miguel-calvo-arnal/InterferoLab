# sim/lab_timing_probe.spec
# -*- mode: python ; coding: utf-8 -*-
#
# STANDALONE lab timing probe: lab_timing_probe[.exe] for the lab PC, which
# only runs packaged executables (no repo, no Python there).
#
#   Windows VM :  scripts\build_lab_probe.bat      -> dist\win\lab_timing_probe\
#   Linux      :  scripts/build_lab_probe.sh       -> dist/lab_timing_probe/
#
# Same hardware dependencies as InterferoLab (see interferolab.spec):
#   * PI E-625 through E816_DLL_x64.dll: bundled from API/PI/ into API/PI/
#     (the probe looks for it at <bundle>/API/PI/E816_DLL_x64.dll, like the
#     app's resource_path("API/PI/E816_DLL_x64.dll"));
#   * Thorlabs camera: NOT bundled, exactly like the app -- pylablib loads
#     thorlabs_tsi_camera_sdk.dll from the ThorCam installation in Program
#     Files, so ThorCam must be installed on the lab PC (it is: the app uses it);
#   * pipython data files, pylablib (+ the numba/pandas/scipy it imports).
# Python's own VCRUNTIME140 ships with the bundle; the analysis backend and
# its vcpkg/MSVC DLLs are not needed (the probe does no analysis).
#
# It must NEVER contain the simulator: the fake pylablib/pipython in
# sim/fakes or any sim.* module.  "sim" is excluded and the guard below
# aborts the build if anything from sim/ other than the probe script itself
# gets in.  Conversely, interferolab.spec rejects anything from sim/ (this
# probe included) in the app bundle.
import os
import sys
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None
probe_dir = os.path.abspath(SPECPATH)            # sim/
project_root = os.path.dirname(probe_dir)
probe_script = os.path.join(probe_dir, "lab_timing_probe.py")
fakes_dir = os.path.join(probe_dir, "fakes")

datas = collect_data_files("pipython")
_pi = os.path.join(project_root, "API", "PI")
if os.path.isdir(_pi):
	datas += [(_pi, os.path.join("API", "PI"))]
else:
	print("WARNING: API/PI not found - the probe will need --dll <path to E816_DLL_x64.dll>")

hiddenimports = [
	"pylablib",
	"pylablib.devices",
	"pylablib.devices.Thorlabs",
	"pipython",
	"pipython.pidevice",
	"pipython.pidevice.gcserror",
	"pipython.pidevice.gcs30",
	"tomllib",
]

excludes = [
	"sim",
	"PyQt5", "PyQt6", "PySide6", "PySide2", "shiboken6",  # no GUI in the probe
	"matplotlib", "PIL", "tkinter", "IPython", "jupyter", "pytest", "setuptools", "cv2",
]

a = Analysis(
	[probe_script],
	# Only the real site-packages: sim/fakes must never be importable here.
	pathex=[],
	binaries=[],
	datas=datas,
	hiddenimports=hiddenimports,
	hookspath=[],
	hooksconfig={},
	runtime_hooks=[],
	excludes=excludes,
	noarchive=False,
	cipher=block_cipher,
)

# >>> probe guard: no simulator inside the standalone probe
_sim_root = os.path.normcase(probe_dir) + os.sep
_allowed = {os.path.normcase(probe_script)}

def _bad(entry):
	name = str(entry[0]).replace("\\", "/")
	src = os.path.normcase(os.path.abspath(str(entry[1]))) if len(entry) > 1 and entry[1] else ""
	if name == "sim" or name.startswith("sim.") or name.startswith("sim/"):
		return True
	if src.startswith(os.path.normcase(fakes_dir) + os.sep):
		return True  # a fake driver under its real name
	return bool(src) and src.startswith(_sim_root) and src not in _allowed

_hits = [e[0] for toc in (a.pure, a.binaries, a.datas) for e in toc if _bad(e)]
if _hits:
	raise SystemExit(
		"ERROR: the standalone probe would bundle simulator code: %s" % ", ".join(sorted(set(map(str, _hits))))
	)
# <<< probe guard

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
	pyz,
	a.scripts,
	[],
	exclude_binaries=True,
	name="lab_timing_probe",
	debug=False,
	strip=not sys.platform.startswith("win"),
	upx=False,
	console=True,  # a console window: progress, prompts and "Press Enter to close"
)

coll = COLLECT(
	exe,
	a.binaries,
	a.zipfiles,
	a.datas,
	strip=not sys.platform.startswith("win"),
	upx=False,
	name="lab_timing_probe",
)
