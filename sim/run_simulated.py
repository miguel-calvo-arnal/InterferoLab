#!/usr/bin/env python
"""
[SIMULATION] Launch InterferoLab against SIMULATED hardware.

    .venv/bin/python sim/run_simulated.py [--profile FILE] [--surface step]
                                          [--mode replay] [--set faults.snap_timeout_every=5]

What it does (the app itself is NOT modified):

1. Puts ``sim/fakes`` first on ``sys.path`` so ``import pylablib`` and
   ``import pipython`` resolve to the fakes, and aborts if they do not.
2. Loads the profile (sim/profiles/default.toml + --profile + overrides).
3. Runtime patches, applied from here and documented in sim/README.md:
   * MainWindow: "[SIMULATION]" prefix in the window title (kept even if the
     title changes) and a fixed red banner in the menu-widget slot at the top.
   * AcquisitionSession.create_output_folder: sweep folders are named
     ``SIM_<timestamp>`` and get ``sim_metadata.json`` (``"simulated": true``)
     plus the ground-truth height map (synthetic mode).
4. Runs ``main.py`` as ``__main__`` from the repository root.

Pixels are never marked.  There is no environment variable or config switch
that turns the simulator on: this script is the only entry point.
"""

from __future__ import annotations

import os
import sys

SIM_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SIM_DIR)
FAKES = os.path.join(SIM_DIR, "fakes")

# Running "python sim/run_simulated.py" puts sim/ itself at sys.path[0]; drop
# it so nothing in sim/ can shadow a top-level module name.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != SIM_DIR]

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import runpy  # noqa: E402
from datetime import datetime  # noqa: E402

log = logging.getLogger("sim.launcher")

BANNER_TEXT = (
    "SIMULATION - camera and piezo are simulated, no real hardware is connected. "
    "Data folders are prefixed SIM_."
)
TITLE_PREFIX = "[SIMULATION] "


# ----------------------------------------------------------------------
# sys.path
# ----------------------------------------------------------------------
def setup_paths() -> None:
    """Fakes first, then the app's own paths (same as main.py adds)."""
    for p in (os.path.join(ROOT, "backend", "analysis"), os.path.join(ROOT, "backend"), ROOT):
        if p not in sys.path:
            sys.path.insert(0, p)
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != FAKES]
    sys.path.insert(0, FAKES)


def assert_fakes_loaded() -> None:
    import pipython
    import pylablib

    for mod in (pylablib, pipython):
        path = os.path.abspath(getattr(mod, "__file__", "") or "")
        if not getattr(mod, "__simulated__", False) or not path.startswith(FAKES + os.sep):
            raise SystemExit(
                f"[SIMULATION] ABORT: '{mod.__name__}' resolved to {path}, not to the fake in {FAKES}. "
                "Refusing to start: this could drive real hardware."
            )


# ----------------------------------------------------------------------
# Profile overrides from the command line
# ----------------------------------------------------------------------
def _parse_scalar(text: str):
    low = text.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    for conv in (int, float):
        try:
            return conv(text)
        except ValueError:
            pass
    return text


def build_overrides(args, base_profile) -> dict:
    over: dict = {}

    def put(dotted: str, value) -> None:
        node = over
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        try:
            is_figure = base_profile.origin(dotted) is not None
        except KeyError:
            is_figure = False
        if is_figure:
            value = {
                "value": value,
                "origin": "estimated",
                "source": "command-line override (--set)",
            }
        node[parts[-1]] = value

    if args.surface:
        put("surface.type", args.surface)
    if args.mode:
        put("signal.mode", args.mode)
    if args.replay_dir:
        put("replay.datasets", [os.path.abspath(args.replay_dir)])
    for item in args.set or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, val = item.split("=", 1)
        put(key.strip(), _parse_scalar(val))
    return over


# ----------------------------------------------------------------------
# Runtime patches (no app file is edited)
# ----------------------------------------------------------------------
def _ground_truth_files(world, folder: str) -> dict:
    src = world.source_if_built() or world.source()
    files = {}
    gt = src.ground_truth()
    if "height_superpixel_um" in gt:
        import numpy as np

        name = "sim_ground_truth_height_superpixel_um.npy"
        np.save(os.path.join(folder, name), gt["height_superpixel_um"])
        files["height_superpixel_um"] = name
    return files


def write_sim_metadata(world, folder: str) -> str:
    """sim_metadata.json (+ ground truth) inside a simulated dataset folder."""
    prof = world.profile
    src = world.source_if_built()
    meta = {
        "simulated": True,
        "generator": "InterferoLab hardware simulator (sim/run_simulated.py)",
        "created": datetime.now().isoformat(timespec="seconds"),
        "profile_files": prof.files,
        "signal_mode": prof.get("signal.mode"),
        "source": src.describe() if src is not None else None,
        "bayer_phase": world.phase,
        "surface": {
            k: prof.get(f"surface.{k}")
            for k in (
                "type",
                "base_height_um",
                "tilt_x_um",
                "tilt_y_um",
                "step_height_um",
                "sphere_radius_um",
                "sphere_cap_um",
                "sample_pixel_um",
            )
        },
        "height_convention": (
            "Stage coordinates in um: a pixel is in focus (envelope peak) when the piezo set-point "
            "equals its height. The superpixel map is the 2x2 mean of the full-resolution map "
            "(sim.signal_model.make_surface(profile, 3000, 4096)); 'mono_superpixel' frames are a "
            "further 2x2 mean. The app's backend reports h = z_max - h_real (tests/README.md)."
        ),
        "pixels_marked": False,
        "figures": [
            {"key": k, "value": v, "origin": o, "source": s} for k, v, o, s in prof.figures()
        ],
    }
    try:
        meta["ground_truth_files"] = _ground_truth_files(world, folder)
    except Exception as e:  # noqa: BLE001 - metadata must still be written
        meta["ground_truth_error"] = str(e)
    path = os.path.join(folder, "sim_metadata.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    return path


def patch_output_folder(world) -> None:
    """SIM_<timestamp> folders + sim_metadata.json for every sweep."""
    from backend.acquisition import acquisition_controller as ac

    def create_output_folder(base_folder: str, log_cb) -> str:
        os.makedirs(base_folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder = os.path.join(base_folder, f"SIM_{timestamp}")
        os.makedirs(folder, exist_ok=True)
        write_sim_metadata(world, folder)
        if log_cb:
            log_cb("info", f"Saving images to: {folder}")
            log_cb("warn", "[SIMULATION] Simulated dataset: SIM_ folder with sim_metadata.json.")
        return folder

    create_output_folder.__sim_patch__ = True
    ac.AcquisitionSession.create_output_folder = staticmethod(create_output_folder)


def patch_config_and_log() -> None:
    """Keep simulator state apart from the real app's.

    * app_config.json -> logs/sim_app_config.json (logs/ is git-ignored), so
      settings changed while simulating never reach the real app.
    * The session log file is renamed SIM_interferolab_<timestamp>.log right
      after the app creates it (before anything is written to it).
    """
    import utils.config_manager as cm
    import utils.session_log as sl

    if not getattr(cm.config_path, "__sim_patch__", False):

        def config_path() -> str:
            return os.path.join(ROOT, "logs", "sim_app_config.json")

        config_path.__sim_patch__ = True
        os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
        cm.config_path = config_path

    original = sl.setup_log_file
    if getattr(original, "__sim_patch__", False):
        return

    def setup_log_file() -> None:
        before = {id(h) for h in logging.getLogger().handlers}
        original()
        for h in logging.getLogger().handlers:
            name = os.path.basename(getattr(h, "baseFilename", "") or "")
            if id(h) in before or not name.startswith("interferolab_"):
                continue
            old = h.baseFilename
            new = os.path.join(os.path.dirname(old), "SIM_" + name)
            h.acquire()
            try:
                h.close()
                os.rename(old, new)
                h.baseFilename, h.mode = new, "a"
            finally:
                h.release()
        log.warning("[SIMULATION] session log: logs/SIM_interferolab_*.log; config: logs/sim_app_config.json")

    setup_log_file.__sim_patch__ = True
    sl.setup_log_file = setup_log_file


def decorate_window(win, quit_after: float | None = None) -> None:
    """Title prefix + fixed red banner on a MainWindow instance."""
    from PySide6.QtCore import QEvent, QObject, Qt, QTimer
    from PySide6.QtWidgets import QApplication, QLabel

    def ensure_title() -> None:
        title = win.windowTitle()
        if not title.startswith(TITLE_PREFIX):
            win.setWindowTitle(TITLE_PREFIX + title)

    class _TitleGuard(QObject):
        def eventFilter(self, obj, event):  # noqa: N802 - Qt API
            if event.type() == QEvent.Type.WindowTitleChange:
                QTimer.singleShot(0, ensure_title)
            return False

    ensure_title()
    win._sim_title_guard = _TitleGuard(win)
    win.installEventFilter(win._sim_title_guard)

    banner = QLabel(BANNER_TEXT, win)
    banner.setObjectName("simBanner")
    banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
    banner.setWordWrap(True)
    banner.setStyleSheet(
        "QLabel#simBanner { background-color: #c00000; color: white; font-weight: bold; "
        "padding: 4px; }"
    )
    # The menu-widget slot sits at the very top, cannot be moved, floated or
    # hidden from a context menu (InterferoLab has no menu bar).
    win.setMenuWidget(banner)
    win._sim_banner = banner
    log.warning("[SIMULATION] Window decorated: title %r, red banner on top", win.windowTitle())

    if quit_after:
        QTimer.singleShot(int(quit_after * 1000), QApplication.quit)


def patch_main_window(quit_after: float | None = None) -> None:
    import views.MainWindow as mw

    cls = mw.MainWindow
    if getattr(cls.__init__, "__sim_patch__", False):
        return
    original = cls.__init__

    def __init__(self, *args, **kwargs):  # noqa: N807
        original(self, *args, **kwargs)
        decorate_window(self, quit_after)

    __init__.__sim_patch__ = True
    cls.__init__ = __init__


# ----------------------------------------------------------------------
# Entry points
# ----------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Run InterferoLab with simulated hardware.")
    ap.add_argument("--profile", help="profile TOML merged over sim/profiles/default.toml")
    ap.add_argument("--surface", choices=("tilted_plane", "step", "sphere", "flat"))
    ap.add_argument("--mode", choices=("synthetic", "replay"))
    ap.add_argument("--replay-dir", help="real stack folder for --mode replay")
    ap.add_argument(
        "--set",
        action="append",
        metavar="KEY=VALUE",
        help="override a profile key, e.g. faults.snap_timeout_every=5 (repeatable)",
    )
    ap.add_argument(
        "--quit-after",
        type=float,
        default=None,
        metavar="SECONDS",
        help="close the app after N seconds (smoke tests)",
    )
    return ap.parse_args(argv)


def bootstrap(argv=None):
    """Everything except running main.py. Returns (args, world)."""
    args = parse_args(argv)
    os.chdir(ROOT)
    setup_paths()
    from sim.hwprofile import load_profile
    from sim.world import configure

    base = load_profile(args.profile)
    world = configure(args.profile, build_overrides(args, base))
    assert_fakes_loaded()
    patch_output_folder(world)
    patch_config_and_log()
    patch_main_window(args.quit_after)
    print(
        "\n" + "=" * 72 + "\n[SIMULATION] InterferoLab with SIMULATED hardware\n"
        f"  profile : {', '.join(world.profile.files)}\n"
        f"  signal  : {world.profile.get('signal.mode')}"
        f" (surface {world.profile.get('surface.type')})\n" + "=" * 72,
        file=sys.stderr,
    )
    return args, world


def main(argv=None) -> None:
    bootstrap(argv)
    runpy.run_path(os.path.join(ROOT, "main.py"), run_name="__main__")


if __name__ == "__main__":
    main()
