"""
[SIMULATION] Fake pipython.pidevice.gcsdevice.GCSDevice for a PI E-625.SR
(driven through the E816 DLL, as the app does) moving a P-611.ZS stage.

Faithful to pipython 2.13.0.2 where the app can notice it:

* the DLL is not touched until the first connection; a missing DLL file
  raises IOError('<path> not found') like GCSDll._handle;
* every command is a GCS transaction under one RLock, and with
  ``errcheck`` on (pipython's default) EVERY send and every query is
  followed by an ``ERR?`` query; a non-zero answer raises GCSError(code);
* MOV does not wait; qONT returns an OrderedDict keyed by the axis strings
  that were passed;
* controller errors: MOV with servo off -> 5, outside the travel -> 7,
  unknown axis -> 15.

Transaction costs come from the profile (piezo.timing.*), the motion from
sim.stage.PiezoStage.
"""

from __future__ import annotations

import collections
import contextlib
import logging
import os
import threading
import time
from collections import OrderedDict

from sim.world import get_world

from .gcserror import GCSError

log = logging.getLogger("sim.piezo")

_AXES = ("A",)

#: One entry of GCSDevice.command_log: command is "MOV" (value = target um,
#: accepted = the controller applied it) or "ONT" (value = the answer).
CommandRecord = collections.namedtuple("CommandRecord", ["t", "command", "axis", "value", "accepted"])


def _itemslist(items):
    if isinstance(items, dict):
        raise TypeError(f"parameter type mismatch: {items!r}")
    if items in (None, "", {}):
        return []
    items = items if isinstance(items, (list, set, tuple)) else [items]
    return list(items)


def _items_values(items, values):
    if isinstance(items, dict):
        if values is not None:
            raise TypeError(
                'parameter type mismatch: If <items> is a dictionary <values> must be "None"'
            )
        return list(items.keys()), list(items.values())
    items, values = _itemslist(items), _itemslist(values)
    if len(items) != len(values) or not items:
        raise ValueError(
            f"items {items!r} and values {values!r} must have the same, non-zero length"
        )
    return items, values


class GCSDevice:
    """Provide a device connected via the PI GCS DLL or another gateway, can be used as context manager."""

    __simulated__ = True

    def __init__(self, devname="", gcsdll="", gateway=None):
        self._devname = devname
        self._gcsdll = gcsdll
        self._gateway = gateway
        self._world = get_world()
        p = self._world.profile
        self._p = p
        self._t_connect = float(p["piezo.timing.connect_s"])
        self._t_close = float(p["piezo.timing.close_s"])
        self._t_write = float(p["piezo.timing.gcs_write_s"])
        self._t_query = float(p["piezo.timing.gcs_query_s"])
        self._stage = self._world.stage
        self._lock = threading.RLock()
        self._id = -1
        self._err = 0
        self._serial = ""
        self._dll_loaded = False
        self.errcheck = True
        # Stable observation point for tools: every MOV that reached the
        # controller (t = time.monotonic() when it was applied, target um) and
        # every qONT answer, as CommandRecord tuples.  Read-only for the app.
        self.command_log: collections.deque = collections.deque(maxlen=4096)
        self._world.piezos.append(self)

    # ------------------------------------------------------------------
    # context manager / lifetime (pipython: __del__ -> _cleanup -> close)
    # ------------------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup()

    def __del__(self):
        with contextlib.suppress(Exception):
            self._cleanup(quiet=True)

    def _cleanup(self, quiet=False):
        if self._id >= 0:
            self._id = -1
            if not quiet:
                time.sleep(self._t_close)

    def close(self):
        """Close connection to device and daisy chain."""
        self._cleanup()

    # ------------------------------------------------------------------
    # DLL / link
    # ------------------------------------------------------------------
    def _load_dll(self):
        if not self._dll_loaded:
            if self._gcsdll and not os.path.isfile(self._gcsdll):
                raise OSError(f"{self._gcsdll} not found")
            self._dll_loaded = True

    @property
    def connected(self):
        return self._id >= 0

    def IsConnected(self):
        return self._id >= 0

    @property
    def dllpath(self):
        return self._gcsdll

    def ConnectUSB(self, serialnum, pid=None, vid=0x1A72):  # noqa: ARG002 - signature parity
        """Open a USB connection to a device."""
        self._load_dll()
        time.sleep(self._t_connect)
        if self._world.faults.count_gcs_command():
            raise GCSError(-1)
        self._serial = f"{self._p['piezo.serial_prefix']}{serialnum}"
        self._id = 0
        log.warning(
            "[SIMULATION] Fake PI controller connected: %s, serial %s (DLL %s not loaded)",
            self._p["piezo.idn_model"],
            self._serial,
            self._gcsdll or "<none>",
        )

    def CloseConnection(self):
        """Reset axes property and close connection to the device."""
        with self._lock:
            if self._id >= 0:
                time.sleep(self._t_close)
            self._id = -1
        log.warning("[SIMULATION] Fake PI controller %s closed", self._serial)

    # ------------------------------------------------------------------
    # GCS transactions
    # ------------------------------------------------------------------
    def _link(self, read: bool) -> None:
        if self._id < 0:
            raise GCSError(-9)
        if self._world.faults.count_gcs_command():
            self._id = -1
            raise GCSError(-3 if read else -2)

    def _checkerror(self) -> None:
        """pipython's errcheck: ask ERR? after the command."""
        if not self.errcheck:
            return
        time.sleep(self._t_query)
        err, self._err = self._err, 0
        if err:
            raise GCSError(err)

    def _send(self, apply) -> None:
        with self._lock:
            self._link(read=False)
            time.sleep(self._t_write)
            apply()
            self._checkerror()

    def _read(self, answer):
        with self._lock:
            self._link(read=True)
            time.sleep(self._t_query)
            value = answer()
            self._checkerror()
            return value

    def _axis_ok(self, axis) -> bool:
        if str(axis) not in _AXES:
            self._err = 15
            return False
        return True

    def _query_axes(self, axes, getter):
        items = _itemslist(axes) or list(_AXES)

        def answer():
            out = OrderedDict()
            for ax in items:
                if self._axis_ok(ax):
                    out[ax] = getter()
            return out

        return self._read(answer)

    # ------------------------------------------------------------------
    # Commands used by InterferoLab
    # ------------------------------------------------------------------
    def qIDN(self):
        """Get ID of the device as string."""
        return self._read(
            lambda: (
                f"(c)SIMULATION Physik Instrumente (PI), {self._p['piezo.idn_model']}, "
                f"{self._serial}, SIM-FW\n"
            )
        )

    def qERR(self):
        """Get the error state of the controller (clears it)."""
        with self._lock:
            self._link(read=True)
            time.sleep(self._t_query)
            err, self._err = self._err, 0
            return err

    def SVO(self, axes, values=None):
        """Set servo-control "on" or "off" (closed-loop/open-loop mode)."""
        axes, values = _items_values(axes, values)

        def apply():
            for ax, v in zip(axes, values, strict=True):
                if self._axis_ok(ax):
                    on = bool(int(v))
                    if on and not self._stage.servo:
                        self._stage.move(self._stage.position())  # hold where it is
                    self._stage.servo = on

        self._send(apply)

    def qSVO(self, axes=None):
        return self._query_axes(axes, lambda: bool(self._stage.servo))

    def MOV(self, axes, values=None):
        """Move 'axes' to specified absolute positions (does not wait)."""
        axes, values = _items_values(axes, values)
        faults = self._world.faults

        def apply():
            for ax, v in zip(axes, values, strict=True):
                if not self._axis_ok(ax):
                    self._log_mov(ax, v, False)
                    return
                if not self._stage.servo:
                    self._err = 5
                    self._log_mov(ax, v, False)
                    return
                z = float(v)
                if not (self._stage.travel_min <= z <= self._stage.travel_max):
                    self._err = 7
                    self._log_mov(ax, v, False)
                    return
            if faults.mov_fails():
                self._err = int(self._p.get("faults.mov_error_code", -1))
                for ax, v in zip(axes, values, strict=True):
                    self._log_mov(ax, v, False)
                return
            for ax, v in zip(axes, values, strict=True):
                self._stage.move(float(v))
                self._log_mov(ax, v, True)

        self._send(apply)

    def _log_mov(self, axis, value, accepted: bool) -> None:
        self.command_log.append(
            CommandRecord(time.monotonic(), "MOV", str(axis), float(value), bool(accepted))
        )

    def qMOV(self, axes=None):
        return self._query_axes(axes, lambda: self._stage._to)

    def qPOS(self, axes=None):
        return self._query_axes(axes, lambda: self._stage.position())

    def qONT(self, axes=None):
        """Check if 'axes' have reached the target."""

        def answer():
            ont = bool(self._stage.on_target())
            self.command_log.append(CommandRecord(time.monotonic(), "ONT", "A", ont, True))
            return ont

        return self._query_axes(axes, answer)

    def SVA(self, axes, values=None):
        """Set open-loop control value (voltage)."""
        axes, values = _items_values(axes, values)

        def apply():
            for ax, v in zip(axes, values, strict=True):
                if self._axis_ok(ax):
                    self._stage.set_voltage(float(v))

        self._send(apply)

    def qSVA(self, axes=None):
        return self._query_axes(axes, lambda: self._stage.voltage)

    def qVOL(self, channels=None):
        """Get current piezo voltages for 'channels'."""
        items = _itemslist(channels) or list(_AXES)
        return self._read(lambda: OrderedDict((c, self._stage.voltage) for c in items))

    def GetError(self):
        return self.qERR()
