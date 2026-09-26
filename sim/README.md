# InterferoLab hardware simulator

Runs the **unmodified** app against a simulated Thorlabs LP126CU camera and a
simulated PI E-625 + P-611.ZS piezo. It exists to test and profile the app
without the instrument. It is never part of a release.

## Launch

```bash
.venv/bin/python sim/run_simulated.py                          # synthetic tilted plane
.venv/bin/python sim/run_simulated.py --surface step            # or sphere, flat
.venv/bin/python sim/run_simulated.py --mode replay             # serve a real stack by z
.venv/bin/python sim/run_simulated.py --profile sim/profiles/measured_2026-10-01_lab.toml
.venv/bin/python sim/run_simulated.py --set faults.snap_timeout_every=5 --set piezo.timing.settle_s=0.05
```

How to tell it is the simulator:
- The window title starts with `[SIMULATION]` and a red banner stays fixed at the top.
- The log shows the device serials `SIM-000000000` and `SIM-LP126CU-00000`, and stderr prints `[SIMULATION]` lines.
- Sweeps are saved to `data/SIM_<timestamp>/` with a `sim_metadata.json` file that contains `"simulated": true`.
- Pixels are never marked.

`sim/` is the only way in. `run_simulated.py` puts `sim/fakes` (fake `pylablib` and `pipython`
packages) first on `sys.path` and aborts if the real drivers load instead.
No environment variable or config switch turns the simulator on.

Runtime patches the launcher applies. No app file is edited:

| Patched object | Change |
|---|---|
| `views.MainWindow.MainWindow.__init__` | Title prefix, kept when the title changes; red banner in the menu-widget slot |
| `AcquisitionSession.create_output_folder` | Creates `SIM_<timestamp>`; writes `sim_metadata.json` and `sim_ground_truth_height_superpixel_um.npy` |
| `utils.config_manager.config_path` | Uses `logs/sim_app_config.json`, so settings changed while simulating never reach the real `app_config.json` |
| `utils.session_log.setup_log_file` | Renames the session log to `logs/SIM_interferolab_<timestamp>.log` |

## Layers and fidelity

| Layer | What is faithful | What is approximate |
|---|---|---|
| **Interface** | The names, signatures, return types and exceptions that InterferoLab uses from pylablib 1.4.5 and pipython 2.13.0.2 (checked against the installed libraries by `tests/test_sim_fakes.py`). `snap()` raises `ThorlabsTLCameraTimeoutError()` with an empty message. `GCSError` messages match. `qONT` returns an `OrderedDict` keyed by the axis string. A missing DLL raises `OSError`. | Only the camera (TLCamera) and GCS subset the app touches. The simulator has no binning. `rgb` output uses a cheap superpixel demosaic. |
| **Timing** | pylablib's own code runs verbatim: each `snap()` stops the acquisition, sets up an 85-frame buffer, arms, sleeps 0.05 s, triggers, waits for the first frame, sleeps 0.2 s and disarms. Frames keep arriving (and are copied) until disarm. pipython sends `ERR?` after every GCS command. `time.sleep`/events only, no busy waits. | Every SDK and USB cost comes from the profile, and most are **estimated**: arm, disarm, open, GCS round trip, settling with the E-625. See the profile and run the lab probe. |
| **Signal** | A known surface → a white-light fringe per channel (period and envelope measured on the clean LP126CU stacks `2026-03-13_13-14-36` and `2026-04-22_12-41-58`; B channel derived; a ~20 nm rms z error per move, **derived** — it is a common-displacement estimator from the fringe analysis, calibrated ×0.8, not a measurement of the piezo, and the P-611.ZS datasheet resolution is well below it) → a raw **BGGR** mosaic (`filter_array_phase = "blue"`) with channel levels from the QE × IR × 3200 K curves → shot and read noise → 12-bit uint16 with saturation. The camera samples the piezo position at mid-exposure. Ground truth is saved with every sweep. | Gaussian envelope × cosine per channel (no dispersion, no NA model). Visibility is estimated. Noise comes from a Gaussian bank read at a random offset (successive frames are shifted copies of one bank). Illumination falloff is a simple quadratic. |
| **Faults** | `[faults]` in the profile, reproducible with a seed: snap timeout (every N or with a probability), camera unplugged after N snaps, exposure not applied or scaled, `GCSError` on MOV, piezo never on target, GCS link drop after N commands, continuous-stream stall after N frames (for one or more arms; `sim/profiles/faults_preview_stall.toml`). | Error codes for unplug events are plausible, not verified on hardware. |
| **Replay** | Real frames by nearest z. Default stack: `2026-03-13_13-14-36`. `11-21-39` and `11-50-20` have 53-64 nm of z drift and are not used. | **APPROXIMATION, logged as such.** The stacks on disk are demosaiced, so each photosite keeps the channel it would have seen. LP126CU TIFF stacks are used at native size. `data/S1F1` and `S1F5` are 8-bit PNG from **another camera**: they are centre-cropped, resized to 2048×1500 and scaled ×4095/255. No noise is added and exposure is ignored. The first read of a TIFF from a hard disk takes about 0.3 s (later reads are prefetched). |

The synthetic generator takes about **8-12 ms per 4096×3000 frame** on the desktop PC (row chunks in up to 8 threads, `signal.generate_threads`; 18 ms serial), less than the camera's 46 ms readout. Since 2026-09-22 every frame delivered while the camera stays armed is generated at the stage position of its own mid-exposure (`camera.regenerate_every_frame = true`), so a continuous preview follows the piezo.
The camera logs a warning if generation ever becomes the bottleneck.

## Profiles

`sim/profiles/default.toml` holds every figure as `{value, origin, source}`.
`origin` is `published` (with a link), `measured`, `derived` (with the calculation) or `estimated` (with the reason).
A profile given with `--profile` is merged over the default, so it only needs the figures it changes.
Command-line `--set` overrides of figures are recorded as `estimated`.

### Loading lab measurements: the lab probe

The lab PC runs **only packaged executables** (no repo, no Python), so the probe ships as its own
standalone program, built from `sim/lab_timing_probe.spec`. It is not part of the InterferoLab
bundle, and it never contains the simulator: a guard in its spec aborts the build if anything
from `sim/fakes` or `sim.*` gets in.

1. **Build it (one step).** On the Windows VM, double-click `scripts\build_lab_probe.bat`. It needs the
   same `.venv\` or `.venv-win\` as `scripts\build_release.bat` and `API\PI\E816_DLL_x64.dll`
   in the repo. The result is `dist\win\lab_timing_probe\`. On Linux, `scripts/build_lab_probe.sh`
   writes to `dist/lab_timing_probe/`.
2. **Run it at the instrument.** Copy the whole `lab_timing_probe` folder to the lab PC, close
   InterferoLab, and double-click `lab_timing_probe.exe`. It takes about 6-10 min. Use
   `lab_timing_probe.exe --interactive` from a console to also take dark frames. The PI DLL is
   inside the folder. The camera SDK comes from the installed ThorCam, as for the app.
3. **Bring back the folder.** The probe writes these files next to the `.exe`:
   - `measured_<date>_<host>.toml`: the profile
   - `measured_<date>_<host>.json`: raw numbers
   - `raw_stack_<date>.npz`: 300 raw frames of 256×256 through focus
   - `lab_timing_probe_<date>.log`: the log
4. **Load the profile:** `.venv/bin/python sim/run_simulated.py --profile <folder>/measured_<date>_<host>.toml`

From a source checkout, the same script also runs directly: `.venv/bin/python sim/lab_timing_probe.py`.

Robustness:
- Every step has a timeout, and a failed step does not stop the others.
- A step that **hangs** marks its device as dead. No later step touches that device, it is left
  unclosed, and the probe tells you to power-cycle it. The other device is still measured.
- The gain (photon transfer) is measured **defocused**, at a z found by a coarse fringe scan, and
  saturation is checked in the window it uses.
`--dry-run --quick` runs the probe against the simulator. It writes `dryrun_*.toml` with origin `estimated` and a "DRY RUN" source, so the result can never pass for a measurement.

## Fidelity gate

`sim/compare_real_vs_sim.py` compares the simulator with the real libraries (interface), the
profile's own predictions (timing) and the real LP126CU stacks (signal: period, envelope, level,
z error per frame). Run it whenever the profile, `sim/fakes` or `sim/signal_model.py` change;
exit code 0 means every check passed. See its docstring for the LZW stack (needs `imagecodecs`
in another interpreter) and the sample cache.

## Performance gate

`sim/perf_preview.py` starts the unmodified app offscreen, drives it through its own
widgets (the buttons and spin boxes a user would click, found by their text) and
measures at two **stable boundaries**, so it works with the acquisition code as it is
and with any redesign of it:

- **camera side**: the fake camera records every frame it delivers while armed in
  `ThorlabsTLCamera.frame_log` (`FrameRecord`: monotonic `seq`, buffer `index`,
  exposure start / mid / ready times, the stage z sampled at mid-exposure, the
  exposure and, when `frame_log_thumbnails` is set, a thumbnail); the fake
  controller records every `MOV` and `ONT` answer in `GCSDevice.command_log`.  Both
  devices register in `World` (`world.camera`, `world.piezo`).
- **screen side**: every paint event of the preview widget (objectName
  `previewLabel`, or `--preview-widget`) is captured and matched **by content**
  (thumbnail correlation, same area-resize grid on both sides) to one delivered
  frame.  A paint counts as a frame on screen only if its pixels changed and it
  matches a frame delivered after the last credited one; repeated or older frames
  do not count, and no more frames can be credited than the camera delivered.

Scenarios: connect, park at 50 µm, piezo still (fastest preview interval the UI
offers), piezo moved by hand in small consecutive steps (seeded jitter), exposure
changes.  Exit code 0 when the on-screen frame rate is at least `--min-fps` (15)
and the median latency from the click on "Move piezo" to the first painted frame
exposed after the stage settled is below `--max-latency-ms` (250); 1 otherwise.
An order that never reaches the controller, or never reaches the screen, counts
as an infinite latency.  `--json` writes every paint and order; `--quick` runs in
about 20 s (used by `tests/test_sim_perf_gate.py`); `--debug` prints the UI state.
The simulator's own generation cost is reported per scenario, with the number of
frames it delayed: when that happens the verdict is flagged SIMULATOR-LIMITED.
`sim/profiles/perf_sdk_half.toml` and `perf_sdk_double.toml` scale the estimated
SDK/GCS durations for sensitivity runs (`--profile`).  Reports (Spanish):
`_agentes/_trabajo/A1_rendimiento.md` (diagnosis of the app of da3c70d) and
`_agentes/_trabajo/B0_puerta.md` (this gate).

## Tests

`tests/test_sim_*.py` covers:
- API surface against the real libraries
- the BGGR phase, the 12-bit range, noise and ground truth
- packaging guards (`interferolab.spec` and `scripts/build_release.*` reject `sim/`)
- an offscreen end-to-end run: connect, preview, move, sweep
- the probe dry run
