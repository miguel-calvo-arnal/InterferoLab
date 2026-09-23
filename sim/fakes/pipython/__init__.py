"""
[SIMULATION] Fake ``pipython`` for the InterferoLab hardware simulator.

Only loaded when sim/run_simulated.py puts sim/fakes first on sys.path.
Exposes the subset of pipython 2.13.0.2 the app uses: GCSDevice (E-625 via
the E816 DLL path) and GCSError.
"""

from .pidevice.gcsdevice import GCSDevice  # noqa: F401
from .pidevice.gcserror import GCSError  # noqa: F401

__version__ = "2.13.0.2+sim"
__simulated__ = True
__all__ = ["GCSDevice", "GCSError", "__version__"]
