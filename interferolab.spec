# interferolab.spec
# -*- mode: python ; coding: utf-8 -*-
#
# Slimmed spec (2026-07): the old spec used collect_submodules/collect_data_files
# on PySide6 and pyqtgraph, which dragged the ENTIRE Qt runtime (WebEngine,
# Qml/Quick, Qt3D, Multimedia, designer/linguist tools, qml/, metatypes/,
# translations/, ...) into the bundle: dist/ was ~1.4 GB.
#
# The app only uses PySide6.QtCore/QtGui/QtWidgets/QtSvg (SVG icons) plus
# pyqtgraph's 2D plotting (no OpenGL).  We now rely on PyInstaller's static
# analysis + the official PySide6/pyqtgraph hooks, add explicit excludes for
# heavy modules that would otherwise sneak in via guarded imports, and keep a
# name-pattern based post-Analysis filter as a safety net.  All filters use
# module/file NAME patterns (no absolute paths, no '/' assumptions), so the
# spec still works when building on Windows.
import os
import re
import sys
import glob
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None

# Stripping symbols is only reliable (and available) on POSIX toolchains.
# On Linux it shaves ~18 MB off llvmlite's unstripped libllvmlite.so alone.
strip_binaries = not sys.platform.startswith("win")

# ----------------------------------------------------------
# Base paths
# ----------------------------------------------------------
project_name = "InterferoLab"
# SPECPATH (injected by PyInstaller) is the directory of this .spec file, so
# the build works no matter which directory pyinstaller is invoked from.
project_root = SPECPATH

# ----------------------------------------------------------
# Hidden imports
# ----------------------------------------------------------
# Project submodules
python_packages = [
	"views",
	"viewmodels",
	"services",
	"widgets",
	"utils",
	# "backend.acquisition" # Use non frozen one (allows to edit without running pyinstaller)
]

# Only what the non-frozen backend/acquisition sources need at runtime: they
# are shipped as editable .py data files, so PyInstaller cannot see their
# imports statically and we must list them here.
hiddenimports = [
	"pylablib",
	"pylablib.devices",
	"pylablib.devices.Thorlabs",
	"pipython",
	"pipython.pidevice",
	"pipython.pidevice.gcserror",
	"pipython.pidevice.gcs30",
	"cv2",
	"pipython.pitools",
	# pyqtgraph.Qt.OpenGLHelpers (imported by PlotCurveItem) loads these two via
	# importlib.import_module, invisible to static analysis:
	"PySide6.QtOpenGL",
	"PySide6.QtOpenGLWidgets",
]

for pkg in python_packages:
	hiddenimports += collect_submodules(pkg)

# NOTE: no collect_submodules("PySide6"/"pyqtgraph") here on purpose.
# Static analysis of main.py already pulls PySide6.QtCore/QtGui/QtWidgets/
# QtSvg and the pyqtgraph modules actually used; collecting everything is
# what used to bloat the bundle.

# ----------------------------------------------------------
# Data files
# ----------------------------------------------------------
datas = []

# pipython resources (GCS translator/dll data; small)
datas += collect_data_files("pipython")

# Add resources, API dlls and the non-frozen acquisition backend.
# NOTE: only backend/acquisition is shipped as editable source; shipping the
# whole backend/ tree would drag the CMake build directory into the bundle.
datas += [
	(os.path.join(project_root, "resources"), "resources"),
	(os.path.join(project_root, "backend", "acquisition"), os.path.join("backend", "acquisition")),
	(os.path.join(project_root, "API"), "API"),
]

# analysis_backend compiled extension (.so / .pyd).
# CMake builds into backend/analysis/build/, so we glob both locations and
# place the result flat in backend/analysis/ so main.py's sys.path addition
# finds it without needing to know the build subdirectory.
# MSVC (multi-config generator) writes into build/Release/, single-config
# generators (Makefiles/Ninja) into build/ directly, so both are globbed.
# Linux builds go to build-linux/ instead: when the project folder is shared
# with a Windows VM, a single build/ cannot hold both CMake caches (cmake
# refuses a cache generated for another source path and generator).
# The first match for a given file name wins: a copy already sitting in
# backend/analysis/ takes precedence over the one still in build/.
_so_patterns = [
	os.path.join(project_root, "backend", "analysis", "analysis_backend*.so"),
	os.path.join(project_root, "backend", "analysis", "analysis_backend*.pyd"),
	os.path.join(project_root, "backend", "analysis", "build-linux", "analysis_backend*.so"),
	os.path.join(project_root, "backend", "analysis", "build", "analysis_backend*.so"),
	os.path.join(project_root, "backend", "analysis", "build", "analysis_backend*.pyd"),
	os.path.join(project_root, "backend", "analysis", "build", "Release", "analysis_backend*.pyd"),
]
_seen_ext = set()
for _pat in _so_patterns:
	for _f in glob.glob(_pat):
		_base = os.path.basename(_f)
		if _base in _seen_ext:
			continue
		_seen_ext.add(_base)
		datas += [(_f, os.path.join("backend", "analysis"))]

# A bundle without the analysis extension is never a valid release: it would
# freeze "green" and only fail at first launch. Abort loudly instead.
if not _seen_ext:
	raise SystemExit(
		"ERROR: no compiled analysis_backend (.so/.pyd) found in "
		"backend/analysis/ or its build directories.\n"
		"Build the C++ backend first (see README, Getting started), or run "
		"scripts/build_release.sh / .ps1 which does it for you."
	)

# ----------------------------------------------------------
# Native runtime DLLs required by analysis_backend (Windows)
# ----------------------------------------------------------
# The .pyd links against OpenCV, FFTW and libtiff from vcpkg.  It is shipped
# as a DATA file above, and PyInstaller does not run binary-dependency
# analysis on datas, so nothing pulls those DLLs into the bundle: the frozen
# app then dies at import time with
#     ImportError: DLL load failed while importing analysis_backend
# vcpkg's applocal deploy leaves exactly the right set next to the freshly
# built targets in build/Release/, so that is the source of truth (it matches
# the vcpkg version this .pyd was linked against).  Destination is the bundle
# root (_internal/), which is on the DLL search path and is also the folder
# main.py registers with os.add_dll_directory() when running unfrozen.
# On Linux these globs match nothing, so this is a no-op there.
# Guarded by platform: when the project folder is shared between a Linux host
# and a Windows VM (the usual setup here), build/Release/ still holds the
# Windows DLLs while building on Linux, and they must not ride along.
binaries = []
if sys.platform.startswith("win"):
	_dll_patterns = [
		os.path.join(project_root, "backend", "analysis", "build", "Release", "*.dll"),
		os.path.join(project_root, "backend", "analysis", "build", "*.dll"),
	]
	_seen_dll = set()
	for _pat in _dll_patterns:
		for _f in glob.glob(_pat):
			_base = os.path.basename(_f).lower()
			if _base in _seen_dll:
				continue
			_seen_dll.add(_base)
			binaries += [(_f, ".")]
	if not binaries:
		print("WARNING: no backend DLLs found in backend/analysis/build - "
		      "analysis_backend will fail to import in the frozen app")

	# --- MSVC runtime, app-local deployment ------------------------------
	# analysis_backend.pyd also links MSVCP140.dll (C++ standard library) and
	# VCOMP140.DLL (OpenMP runtime, pulled in by /openmp:experimental).  These
	# are NOT part of Windows: on a machine without the VC++ 2015-2022
	# Redistributable the app dies with the same "DLL load failed while
	# importing analysis_backend".  Microsoft allows shipping them next to the
	# application ("app-local deployment"), so we bundle them and the installer
	# is not needed on the target machine.
	# VCRUNTIME140.dll / VCRUNTIME140_1.dll are already in the bundle (Python
	# ships them), so they are not collected again.
	# Source priority:
	#   1. install/vcredist/x64/  - a copy pinned in the repo folder, if present
	#   2. the redist folder of the local VS / Build Tools install, newest
	#      version first (this is the same toolset that built the .pyd)
	#   3. System32 - what the machine currently has installed
	_msvc_runtime = ("MSVCP140.dll", "vcomp140.dll")

	def _redist_version_key(path):
		# .../VC/Redist/MSVC/14.44.35211/x64/Microsoft.VC143.CRT -> (14, 44, 35211)
		m = re.search(r"[\\/]MSVC[\\/](\d+(?:\.\d+)*)[\\/]", path)
		return tuple(int(n) for n in m.group(1).split(".")) if m else ()

	_redist_roots = [os.path.join(project_root, "install", "vcredist", "x64")]
	_vs_dirs = []
	for _prefix in (r"C:\Program Files", r"C:\Program Files (x86)"):
		for _kind in ("CRT", "OpenMP"):
			_vs_dirs += glob.glob(os.path.join(
				_prefix, "Microsoft Visual Studio", "*", "*", "VC", "Redist",
				"MSVC", "*", "x64", "Microsoft.VC*." + _kind))
	_redist_roots += sorted(_vs_dirs, key=_redist_version_key, reverse=True)
	_redist_roots.append(os.path.join(
		os.environ.get("SystemRoot", r"C:\Windows"), "System32"))

	for _dll in _msvc_runtime:
		for _root in _redist_roots:
			_cand = os.path.join(_root, _dll)
			if os.path.isfile(_cand):
				binaries += [(_cand, ".")]
				print("MSVC runtime: %s <- %s" % (_dll, _root))
				break
		else:
			print("WARNING: %s not found - the frozen app will require the "
			      "VC++ 2015-2022 Redistributable (x64) on the target machine"
			      % _dll)

# ----------------------------------------------------------
# Excludes
# ----------------------------------------------------------
# Every entry below was verified unused:
#   - grep over main.py/views/viewmodels/services/widgets/utils/backend/acquisition
#     shows only PySide6.QtCore/QtGui/QtWidgets/QtSvg imports;
#   - pyqtgraph.Qt imports QtSvg (needed), and QtOpenGLWidgets/QtTest only
#     inside try/except with a FailedImport fallback, so excluding them is safe
#     (the app never enables pyqtgraph's OpenGL rendering: no setUseOpenGL /
#     pyqtgraph.opengl usage in the code);
#   - pylablib probes PyQt5 and pulls numba/pandas/scipy (those three ARE
#     required at import time and stay in the bundle).
excludes = [
	"PyQt5",  # pylablib probes it; we use PySide6
	"PyQt6",
	"backend.acquisition",  # Avoid freeze (shipped as editable data files above)

	# --- Qt modules the app never imports ---
	"PySide6.QtWebEngineCore",  # 195 MB lib + resources; no web views in the app
	"PySide6.QtWebEngineWidgets",
	"PySide6.QtWebEngineQuick",
	"PySide6.QtWebChannel",
	"PySide6.QtWebSockets",
	"PySide6.QtWebView",
	"PySide6.QtQml",            # no QML/QtQuick: pure QtWidgets UI
	"PySide6.QtQuick",
	"PySide6.QtQuick3D",
	"PySide6.QtQuickControls2",
	"PySide6.QtQuickWidgets",
	"PySide6.Qt3DCore",         # no 3D scenegraph
	"PySide6.Qt3DRender",
	"PySide6.Qt3DInput",
	"PySide6.Qt3DLogic",
	"PySide6.Qt3DAnimation",
	"PySide6.Qt3DExtras",
	"PySide6.QtMultimedia",     # no audio/video (drags ffmpeg libs)
	"PySide6.QtMultimediaWidgets",
	"PySide6.QtSpatialAudio",
	"PySide6.QtTextToSpeech",
	"PySide6.QtCharts",         # plotting is pyqtgraph, not QtCharts
	"PySide6.QtGraphs",
	"PySide6.QtGraphsWidgets",
	"PySide6.QtDataVisualization",
	"PySide6.QtPdf",            # no PDF viewer
	"PySide6.QtPdfWidgets",
	"PySide6.QtNetwork",        # no Qt networking (hardware IO is pylablib/pipython/pyserial)
	"PySide6.QtNetworkAuth",
	# NOTE: PySide6.QtOpenGL / QtOpenGLWidgets must NOT be excluded:
	# pyqtgraph.Qt.OpenGLHelpers does importlib.import_module("PySide6.QtOpenGL")
	# unconditionally (imported by PlotCurveItem), so the app crashes without it.
	"PySide6.QtTest",           # pyqtgraph guarded import; tests only
	"PySide6.QtSql",
	"PySide6.QtXml",
	"PySide6.QtDesigner",       # designer/uic tooling
	"PySide6.QtUiTools",
	"PySide6.QtHelp",
	"PySide6.QtPrintSupport",   # app has no print actions; pyqtgraph 0.14 does not use it
	"PySide6.QtLocation",
	"PySide6.QtPositioning",
	"PySide6.QtBluetooth",
	"PySide6.QtNfc",
	"PySide6.QtSerialPort",     # serial IO goes through pyserial, not Qt
	"PySide6.QtSerialBus",
	"PySide6.QtSensors",
	"PySide6.QtScxml",
	"PySide6.QtStateMachine",
	"PySide6.QtRemoteObjects",
	"PySide6.QtHttpServer",
	"PySide6.QtConcurrent",
	"PySide6.QtDBus",           # libQt6DBus still ships via binary deps of the xcb plugin

	# --- pyqtgraph optional GL stack (unused: no pyqtgraph.opengl imports) ---
	"pyqtgraph.opengl",
	"OpenGL",  # PyOpenGL, only needed by pyqtgraph.opengl

	# --- heavy Python packages not used by the app or its runtime deps ---
	"matplotlib",  # installed in the venv but never imported (plots are pyqtgraph)
	"PIL",         # pillow: never imported (image IO is cv2/numpy)
	"tkinter",
	"IPython",
	"jupyter",
	"pytest",
	"setuptools",

	# --- hardware simulator: NEVER ships (see the guard after Analysis) ---
	"sim",
]

# ----------------------------------------------------------
# Analysis
# ----------------------------------------------------------
a = Analysis(
	["main.py"],
	pathex=[project_root],
	binaries=binaries,
	datas=datas,
	hiddenimports=hiddenimports,
	hookspath=[],
	hooksconfig={},
	runtime_hooks=[],
	excludes=excludes,
	win_no_prefer_redirects=False,
	win_private_assemblies=False,
	noarchive=False,
	cipher=block_cipher,
)

# >>> sim-exclusion guard
# The hardware simulator (sim/: fake pylablib/pipython + launcher) must never
# end up in a release: a frozen app with fake drivers would "work" without
# ever touching the instrument. "sim" is in `excludes` above; this guard
# aborts the build if anything from sim/ (by module name or by source path,
# which also catches the fake pylablib/pipython packages) got in anyway.
# tests/test_sim_safeguards.py executes this block against fake TOCs.
_sim_root = os.path.normcase(os.path.join(os.path.abspath(project_root), "sim")) + os.sep

def _is_sim_entry(entry):
	name = str(entry[0]).replace("\\", "/")
	src = str(entry[1]) if len(entry) > 1 and entry[1] else ""
	if name == "sim" or name.startswith("sim.") or name.startswith("sim/"):
		return True
	return bool(src) and os.path.normcase(os.path.abspath(src)).startswith(_sim_root)

_sim_hits = [e[0] for toc in (a.pure, a.binaries, a.datas) for e in toc if _is_sim_entry(e)]
if _sim_hits:
	raise SystemExit(
		"ERROR: the hardware simulator (sim/) would be bundled: %s\n"
		"sim/ must never ship; it is only run through sim/run_simulated.py."
		% ", ".join(sorted(set(map(str, _sim_hits)))[:20])
	)
# <<< sim-exclusion guard

# ----------------------------------------------------------
# Post-Analysis safety-net filter
# ----------------------------------------------------------
# The module excludes above stop the Python side, but Qt shared libraries and
# plugins can still ride in through binary-dependency analysis (e.g. the
# imageformats qpdf plugin pulling libQt6Pdf).  Filter them out of the TOCs by
# NAME pattern.  Patterns are matched case-insensitively against the
# destination path with separators normalised, so they behave identically on
# Linux (.so) and Windows (.dll) builds.

def _norm(name):
	return name.replace("\\", "/").lower()

# Any TOC entry whose normalised destination contains one of these substrings
# is dropped.  Each pattern corresponds to an excluded Qt module above.
_drop_patterns = (
	"qt6webengine", "qtwebengine",          # WebEngine libs + resources/locales
	"qt6pdf", "libqpdf", "imageformats/qpdf",  # Pdf lib + its imageformat plugin
	"qt6qml", "qt6quick", "qt6labs",        # QML/Quick runtime
	"qt63d",                                # Qt3D
	"qt6multimedia", "qt6spatialaudio", "qt6ffmpegmediaplugin",
	"qt6charts", "qt6graphs", "qt6datavisualization",
	"qt6shadertools",
	"qt6designer", "qt6uitools", "qt6help",
	"qt6location", "qt6positioning",
	"qt6bluetooth", "qt6nfc",
	"qt6serialport", "qt6serialbus",
	"qt6sensors", "qt6scxml", "qt6statemachine", "qt6remoteobjects",
	"qt6test", "qt6sql", "qt6xml",
	"qt6network",                           # QtNetwork excluded above
	"qt6printsupport",
	"qt6virtualkeyboard",
	# ffmpeg runtime shipped with Qt Multimedia (NOT opencv's own copy, which
	# lives under opencv_python.libs/ with hashed names and is kept)
	"pyside6/qt/lib/libav", "pyside6/qt/lib/libsw",
	# Qt tooling/dev payload (only present if some hook collects data files)
	"pyside6/qt/qml/", "pyside6/qt/metatypes/", "pyside6/qt/resources/",
	"pyside6/include/", "pyside6/typesystems/", "pyside6/glue/",
	"pyside6/assistant", "pyside6/designer", "pyside6/linguist",
	"pyside6/lupdate", "pyside6/lrelease", "pyside6/qmlformat", "pyside6/qmlls",
	# Qt plugin folders tied to excluded modules
	"plugins/sceneparsers", "plugins/renderers", "plugins/geometryloaders",
	"plugins/sqldrivers", "plugins/multimedia", "plugins/geoservices",
	"plugins/canbus", "plugins/qmltooling", "plugins/scenegraph",
	"plugins/networkinformation", "plugins/tls",
	"plugins/designer", "plugins/assetimporters", "plugins/position",
	"plugins/sensors", "plugins/webview", "plugins/virtualkeyboard",
)

def _keep(entry):
	dest = _norm(entry[0])
	return not any(p in dest for p in _drop_patterns)

a.binaries = [b for b in a.binaries if _keep(b)]
a.datas = [d for d in a.datas if _keep(d)]

# --- GTK platform-theme chain (Linux only) -------------------------------
# The qgtk3 platform theme plugin only provides native GTK theming/dialogs;
# the app ships its own QSS theme and works fine with Qt's default theme.
# Its dependency closure (verified with an ELF NEEDED reverse-dependency scan
# of the bundle: every lib below is referenced ONLY by libqgtk3.so) weighs
# ~62 MB, mostly system ICU 78 pulled via gtk -> tinysparql -> libxml2.
# Exact sonames (with version) are used so we can never clash with Qt's own
# bundled ICU 73 under PySide6/Qt/lib/ or with its top-level symlinks.
# On Windows these files do not exist, so the filter is a no-op there.
_gtk_chain_libs = {
	"libqgtk3.so",
	"libxcomposite.so.1", "libxcursor.so.1", "libxdamage.so.1",
	"libxfixes.so.3", "libxi.so.6", "libxinerama.so.1",
	"libxrandr.so.2", "libxrender.so.1",
	"libatk-1.0.so.0", "libatk-bridge-2.0.so.0", "libatspi.so.0",
	"libblkid.so.1", "libcairo-gobject.so.2", "libcairo.so.2",
	"libcloudproviders.so.0", "libdatrie.so.1", "libepoxy.so.0",
	"libffi.so.8", "libfribidi.so.0",
	"libgdk-3.so.0", "libgdk_pixbuf-2.0.so.0", "libgio-2.0.so.0",
	"libglycin-2.so.0", "libgmodule-2.0.so.0", "libgobject-2.0.so.0",
	"libgraphite2.so.3", "libgtk-3.so.0", "libharfbuzz.so.0",
	"libicudata.so.78", "libicuuc.so.78",
	"libjson-glib-1.0.so.0", "liblcms2.so.2", "libmount.so.1",
	"libpango-1.0.so.0", "libpangocairo-1.0.so.0", "libpangoft2-1.0.so.0",
	"libpixman-1.so.0", "libseccomp.so.2", "libsqlite3.so.0",
	"libthai.so.0", "libtinysparql-3.0.so.0", "libxml2.so.16",
}

def _keep_gtk(entry):
	base = _norm(entry[0]).rsplit("/", 1)[-1]
	return base not in _gtk_chain_libs

a.binaries = [b for b in a.binaries if _keep_gtk(b)]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ----------------------------------------------------------
# Executable
# ----------------------------------------------------------
exe = EXE(
	pyz,
	a.scripts,
	[],
	name=project_name,
	debug=False,
	bootloader_ignore_signals=False,
	strip=strip_binaries,
	upx=False,
	console=False,
	disable_windowed_traceback=False,
	argv_emulation=False,
	target_arch=None,
	codesign_identity=None,
	entitlements_file=None,
	exclude_binaries=True,
	icon=os.path.join(project_root, "resources", "icon.ico"),
)

# ----------------------------------------------------------
# COLLECT
# ----------------------------------------------------------
coll = COLLECT(
	exe,
	a.binaries,
	a.zipfiles,
	a.datas,
	strip=strip_binaries,
	upx=False,
	upx_exclude=[],
	name=project_name,
)
