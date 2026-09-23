"""
C12 - Stress for the worker/thread "graveyard" (_retired[-3:]).

tests/test_acquisition_service.py documents a SIGBUS when the Python wrappers
of a finished QThread/worker were dropped while deleteLater was still running
in the dying thread.  The service now keeps the last three pairs.  This script
runs hundreds of one-shot preview captures back to back, with snap() replaced
by an instant frame (instrumentation on the fake camera object) so the
thread churn, not the camera timing, dominates.

Exit code 0 and the JSON line mean the graveyard held.

Usage: .venv/bin/python c12_graveyard_stress.py [cycles]
"""

import sys
import time

import numpy as np
from harness import App

cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 600

app = App()
res = {"connected": app.connect(), "cycles": cycles}
cam = app.session._camera
h, w = app.world.sensor_shape
frame = np.full((h, w), 1000, dtype=np.uint16)
cam.snap = lambda timeout=5.0, return_info=False: frame  # instant snap

t0 = time.monotonic()
done = 0
for _ in range(cycles):
    n = len(app.frames)
    if not app.svc.capture_preview():
        continue
    if not app.spin(lambda n=n: len(app.frames) > n and not app.vm.is_previewing(), 5):
        break
    done += 1
res["completed"] = done
res["elapsed_s"] = round(time.monotonic() - t0, 2)
res["ms_per_cycle"] = round(1000 * res["elapsed_s"] / max(done, 1), 1)
res["retired_len"] = len(app.svc._retired)
app.finish(res)
