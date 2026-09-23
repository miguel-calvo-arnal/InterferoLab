"""
[SIMULATION] Fake ``pylablib`` for the InterferoLab hardware simulator.

Only loaded when sim/run_simulated.py puts sim/fakes first on sys.path.
It exposes the subset of pylablib 1.4.5 the app uses
(pylablib.devices.Thorlabs.ThorlabsTLCamera); everything else is absent.
"""

__version__ = "1.4.5+sim"
__simulated__ = True
