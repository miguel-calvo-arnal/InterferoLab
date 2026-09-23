"""Safeguards that keep the simulator from ever passing for the real instrument.

* interferolab.spec excludes sim/ and aborts if anything from it is bundled;
* both build_release scripts inspect the finished bundle;
* main.py (the real entry point) never reaches sim/ nor the fake drivers;
* the launcher puts the fakes first and refuses to start otherwise;
* simulated datasets carry sim_metadata.json with "simulated": true.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import types

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = os.path.join(ROOT, "interferolab.spec")
SIM = os.path.join(ROOT, "sim")
FAKES = os.path.join(SIM, "fakes")


def _spec_text() -> str:
    with open(SPEC, encoding="utf-8") as fh:
        return fh.read()


def _guard_block() -> str:
    text = _spec_text()
    m = re.search(r"# >>> sim-exclusion guard\n(.*?)# <<< sim-exclusion guard", text, re.S)
    assert m, "interferolab.spec lost its sim-exclusion guard block"
    return m.group(1)


def _run_guard(pure=(), binaries=(), datas=()):
    ns = {
        "os": os,
        "project_root": ROOT,
        "a": types.SimpleNamespace(pure=list(pure), binaries=list(binaries), datas=list(datas)),
    }
    exec(compile(_guard_block(), SPEC, "exec"), ns)  # noqa: S102 - our own spec


# ----------------------------------------------------------------------
# Packaging
# ----------------------------------------------------------------------
def test_spec_excludes_sim_module():
    tree = ast.parse(_spec_text().expandtabs(4))
    excludes = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "excludes" for t in node.targets
        ):
            excludes = ast.literal_eval(node.value)
    assert excludes is not None and "sim" in excludes


def test_spec_guard_runs_after_analysis():
    text = _spec_text()
    assert (
        text.index("a = Analysis(")
        < text.index("# >>> sim-exclusion guard")
        < text.index("pyz = PYZ(")
    )


def test_spec_guard_accepts_a_clean_bundle():
    _run_guard(
        pure=[
            ("pylablib", os.path.join(ROOT, ".venv", "lib", "pylablib", "__init__.py"), "PYMODULE"),
            ("views.MainWindow", os.path.join(ROOT, "views", "MainWindow.py"), "PYMODULE"),
            ("simplejson", "/usr/lib/simplejson/__init__.py", "PYMODULE"),
        ],
        datas=[
            (
                "backend/acquisition/acquisition_controller.py",
                os.path.join(ROOT, "backend", "acquisition", "acquisition_controller.py"),
                "DATA",
            )
        ],
    )


@pytest.mark.parametrize(
    "entry",
    [
        ("sim.world", os.path.join(SIM, "world.py"), "PYMODULE"),
        ("sim", os.path.join(SIM, "__init__.py"), "PYPACKAGE"),
        # a FAKE driver under its real name is caught by its source path
        ("pylablib", os.path.join(FAKES, "pylablib", "__init__.py"), "PYMODULE"),
        (
            "pipython.pidevice.gcsdevice",
            os.path.join(FAKES, "pipython", "pidevice", "gcsdevice.py"),
            "PYMODULE",
        ),
    ],
)
def test_spec_guard_rejects_simulator_modules(entry):
    with pytest.raises(SystemExit, match="simulator"):
        _run_guard(pure=[entry])


def test_spec_guard_rejects_simulator_data_files():
    with pytest.raises(SystemExit):
        _run_guard(
            datas=[
                ("sim/profiles/default.toml", os.path.join(SIM, "profiles", "default.toml"), "DATA")
            ]
        )


def _sh_checker() -> str:
    with open(os.path.join(ROOT, "scripts", "build_release.sh"), encoding="utf-8") as fh:
        text = fh.read()
    assert "# >>> sim-exclusion check" in text
    m = re.search(r"<<'PYSIM'.*?\n(.*?)\nPYSIM\n", text, re.S)
    assert m, "build_release.sh lost its embedded bundle checker"
    return m.group(1)


def test_both_build_scripts_check_the_bundle():
    _sh_checker()
    with open(os.path.join(ROOT, "scripts", "build_release.ps1"), encoding="utf-8") as fh:
        ps1 = fh.read()
    assert "# >>> sim-exclusion check" in ps1 and "PKG_ITEM_PYZ" in ps1 and "__simulated__" in ps1
    # the check sits between freezing and archiving
    assert (
        ps1.index("PyInstaller interferolab.spec")
        < ps1.index("$simCheck")
        < ps1.index("CreateFromDirectory")
    )


_EXE = os.path.join(ROOT, "dist", "InterferoLab", "InterferoLab")


@pytest.mark.skipif(not os.path.isfile(_EXE), reason="no Linux build in dist/ to inspect")
def test_bundle_checker_on_existing_build(tmp_path):
    script = tmp_path / "check.py"
    script.write_text(_sh_checker())
    clean = subprocess.run(
        [sys.executable, str(script), os.path.dirname(_EXE), _EXE],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert clean.returncode == 0, clean.stdout + clean.stderr
    # A bundle folder that contains sim/ must be rejected (same exe for the PYZ part).
    fake_dist = tmp_path / "dist"
    (fake_dist / "_internal" / "sim").mkdir(parents=True)
    (fake_dist / "_internal" / "sim" / "run_simulated.py").write_text("")
    dirty = subprocess.run(
        [sys.executable, str(script), str(fake_dist), _EXE],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert dirty.returncode == 1 and "sim" in dirty.stdout


# ----------------------------------------------------------------------
# The real entry point never reaches the simulator
# ----------------------------------------------------------------------
def test_main_py_import_chain_never_loads_sim_or_fakes():
    code = f"""
import sys, json, os
for p in ({ROOT!r}, {os.path.join(ROOT, "backend")!r}, {os.path.join(ROOT, "backend", "analysis")!r}):
    sys.path.insert(0, p)
os.chdir({ROOT!r})
import main  # the whole app import chain (views -> services -> acquisition_controller)
import pylablib, pipython
mods = sorted(m for m in sys.modules if m == "sim" or m.startswith("sim."))
print(json.dumps({{"sim_modules": mods,
                  "files": [pylablib.__file__, pipython.__file__],
                  "simulated": [getattr(pylablib, "__simulated__", False), getattr(pipython, "__simulated__", False)],
                  "controller": sys.modules["backend.acquisition.acquisition_controller"].__file__}}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert proc.returncode == 0, proc.stderr
    r = json.loads(proc.stdout.strip().splitlines()[-1])
    assert r["sim_modules"] == []
    assert r["simulated"] == [False, False]
    for f in r["files"]:
        assert not os.path.abspath(f).startswith(SIM + os.sep)


def _app_sources():
    skip = {
        "sim",
        "tests",
        ".venv",
        ".venv-win",
        "build",
        "build-linux",
        "dist",
        "docs",
        "data",
        "output",
        "releases",
        "resultados",
        "auditoria",
        "_agentes",
        "install",
        "logs",
    }
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in skip and not d.startswith(".")]
        for f in filenames:
            if f.endswith(".py"):
                yield os.path.join(dirpath, f)


def test_app_sources_never_import_the_simulator():
    offenders = []
    for path in _app_sources():
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            if any(n == "sim" or n.startswith("sim.") for n in names):
                offenders.append(os.path.relpath(path, ROOT))
        if "run_simulated" in src or "sim/fakes" in src or "sim\\fakes" in src:
            offenders.append(os.path.relpath(path, ROOT))
    assert offenders == []


# ----------------------------------------------------------------------
# Launcher
# ----------------------------------------------------------------------
def test_launcher_loads_fakes_first_and_patches():
    code = f"""
import sys, json
sys.path.insert(0, {SIM!r})
import run_simulated
args, world = run_simulated.bootstrap(["--surface", "step", "--set", "faults.snap_timeout_every=7",
                                       "--set", "camera.timing.arm_s=0.3"])
import pylablib, pipython
from backend.acquisition import acquisition_controller as ac
import views.MainWindow as mw
print(json.dumps({{
    "path0": sys.path[0], "files": [pylablib.__file__, pipython.__file__],
    "surface": world.profile["surface.type"],
    "fault": world.profile["faults.snap_timeout_every"],
    "arm": world.profile["camera.timing.arm_s"], "arm_origin": world.profile.origin("camera.timing.arm_s"),
    "folder_patched": getattr(ac.AcquisitionSession.create_output_folder, "__sim_patch__", False),
    "window_patched": getattr(mw.MainWindow.__init__, "__sim_patch__", False),
    "config_path": __import__("utils.config_manager", fromlist=["x"]).config_path(),
    "log_patched": getattr(__import__("utils.session_log", fromlist=["x"]).setup_log_file, "__sim_patch__", False),
    "sim_dir_on_path": {SIM!r} in sys.path,
}}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert proc.returncode == 0, proc.stderr
    r = json.loads(proc.stdout.strip().splitlines()[-1])
    assert r["path0"] == FAKES
    for f in r["files"]:
        assert os.path.abspath(f).startswith(FAKES + os.sep)
    assert r["surface"] == "step" and r["fault"] == 7
    assert (
        r["arm"] == 0.3 and r["arm_origin"] == "estimated"
    )  # a CLI override never passes as measured
    assert r["folder_patched"] and r["window_patched"] and r["log_patched"]
    # simulator settings never reach the real app_config.json
    assert r["config_path"] == os.path.join(ROOT, "logs", "sim_app_config.json")
    assert not r["sim_dir_on_path"]  # sim/ itself must not shadow top-level modules
    assert "[SIMULATION]" in proc.stderr


def test_launcher_refuses_to_run_on_real_drivers():
    code = f"""
import sys
sys.path.insert(0, {SIM!r})
import run_simulated
run_simulated.assert_fakes_loaded()   # real pylablib/pipython from .venv -> must abort
print("NOT ABORTED")
"""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert proc.returncode != 0
    assert "NOT ABORTED" not in proc.stdout
    assert "ABORT" in proc.stderr


def test_sim_metadata_and_ground_truth(tmp_path):
    from sim import run_simulated
    from sim.world import configure

    fig = lambda v: {"value": v, "origin": "estimated", "source": "test"}  # noqa: E731
    world = configure(None, {"camera": {"sensor": {"height": fig(40), "width": fig(60)}}})
    path = run_simulated.write_sim_metadata(world, str(tmp_path))
    with open(path, encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta["simulated"] is True and meta["pixels_marked"] is False
    assert meta["bayer_phase"] == "blue"
    assert any(f["key"] == "camera.timing.arm_s" and f["origin"] for f in meta["figures"])
    gt = np.load(tmp_path / meta["ground_truth_files"]["height_superpixel_um"])
    assert gt.shape == (20, 30)
    assert gt.mean() == pytest.approx(50.0, abs=2.0)


# ----------------------------------------------------------------------
# Standalone lab probe: self-contained, own spec, never the simulator
# ----------------------------------------------------------------------
PROBE = os.path.join(SIM, "lab_timing_probe.py")
PROBE_SPEC = os.path.join(SIM, "lab_timing_probe.spec")


def test_probe_is_self_contained():
    """The frozen probe must not need (nor carry) anything from sim/."""
    with open(PROBE, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not (node.module == "sim" or node.module.startswith("sim.")), node.module
            assert node.level == 0
        if isinstance(node, ast.Import):
            assert all(not a.name.startswith("sim") for a in node.names)


def _probe_guard(pure=(), datas=()):
    with open(PROBE_SPEC, encoding="utf-8") as fh:
        text = fh.read()
    m = re.search(r"# >>> probe guard.*?\n(.*?)# <<< probe guard", text, re.S)
    assert m
    ns = {
        "os": os,
        "probe_dir": SIM,
        "probe_script": PROBE,
        "fakes_dir": FAKES,
        "a": types.SimpleNamespace(pure=list(pure), binaries=[], datas=list(datas)),
    }
    exec(compile(m.group(1), PROBE_SPEC, "exec"), ns)  # noqa: S102 - our own spec


def test_probe_spec_excludes_and_guards_the_simulator():
    with open(PROBE_SPEC, encoding="utf-8") as fh:
        text = fh.read()
    assert '"sim",' in text and "pathex=[]" in text and "console=True" in text
    assert 'os.path.join("API", "PI")' in text  # PI DLL bundled like the app
    _probe_guard(
        pure=[
            ("lab_timing_probe", PROBE, "PYSOURCE"),
            ("pylablib", os.path.join(ROOT, ".venv", "x", "pylablib", "__init__.py"), "PYMODULE"),
        ]
    )
    for bad in (
        ("pylablib", os.path.join(FAKES, "pylablib", "__init__.py"), "PYMODULE"),
        ("sim.world", os.path.join(SIM, "world.py"), "PYMODULE"),
        ("world", os.path.join(SIM, "world.py"), "PYMODULE"),
    ):
        with pytest.raises(SystemExit):
            _probe_guard(pure=[bad])


def test_app_spec_rejects_the_probe():
    with pytest.raises(SystemExit):
        _run_guard(pure=[("lab_timing_probe", PROBE, "PYMODULE")])
