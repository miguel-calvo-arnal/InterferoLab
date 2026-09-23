# tests/diag — reproduction scripts behind `tests/test_diag_concurrency.py`

Each script boots the real app against the simulated hardware (offscreen),
drives it as a user would and prints one JSON line; the tests run them in a
subprocess and assert the behaviour the app should have. Findings C1-C12 come
from the phase-2 concurrency review (2026-09-21); `b1_single_owner.py` checks
the single camera-owner design (2026-09-22).

Run one by hand from the repository root:

    QT_QPA_PLATFORM=offscreen .venv/bin/python tests/diag/c6_move_vs_preview.py 200 10
