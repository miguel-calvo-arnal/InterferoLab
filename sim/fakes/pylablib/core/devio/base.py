"""[SIMULATION] Mirror of pylablib.core.devio.base (exception root only)."""


class DeviceError(RuntimeError):
    """Generic device error (same base class as pylablib 1.4.5)."""


class DeviceBackendError(DeviceError):
    """Generic backend error."""
