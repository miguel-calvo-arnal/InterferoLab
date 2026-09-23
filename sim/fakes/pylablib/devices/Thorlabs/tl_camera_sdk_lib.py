"""[SIMULATION] Mirror of pylablib.devices.Thorlabs.tl_camera_sdk_lib (errors only)."""

from ...core.devio.base import DeviceError


class ThorlabsTLCameraError(DeviceError):
    """Generic Thorlabs TLCamera error"""


class ThorlabsTLCameraLibError(ThorlabsTLCameraError):
    """Generic Thorlabs TLCamera library error (same message format as pylablib 1.4.5)."""

    def __init__(self, func, code, lib=None):
        self.func = func
        self.code = code
        self.name = "UNKNOWN"
        self.desc = ""
        try:
            if lib is not None:
                self.desc = str(lib.tl_camera_get_last_error())
        except Exception:  # noqa: BLE001 - same tolerance as the real class
            pass
        self.msg = f"function '{func}' raised error {code}({self.name}): {self.desc}"
        ThorlabsTLCameraError.__init__(self, self.msg)


class _SimLastError:
    """Stands in for the SDK handle so the error carries a description."""

    def __init__(self, desc: str) -> None:
        self._desc = desc

    def tl_camera_get_last_error(self) -> str:
        return self._desc


def sim_lib_error(func: str, code: int, desc: str) -> ThorlabsTLCameraLibError:
    return ThorlabsTLCameraLibError(func, code, lib=_SimLastError(desc))
