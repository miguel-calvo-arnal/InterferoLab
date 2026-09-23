"""[SIMULATION] Fake pylablib.devices.Thorlabs: only the TLCamera family is simulated."""

from .TLCamera import (  # noqa: F401
    ThorlabsTLCamera,  # noqa: F401
    ThorlabsTLCameraError,
    ThorlabsTLCameraTimeoutError,
)
from .TLCamera import list_cameras as list_cameras_tlcam  # noqa: F401

__simulated__ = True
