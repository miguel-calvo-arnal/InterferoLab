"""
[SIMULATION] Mirror of pipython.pidevice.gcserror.

Same class, base and message format ("<text> <code>") as pipython 2.13.0.2;
only the error codes the simulator can raise are in the table (the real
table has ~900 entries; unknown codes render as str(code), like pipython).
"""

from .pierror_base import PIErrorBase

_ERRMSG = {
    -1: "Error during com operation (could not be specified)",
    -2: "Error while sending data",
    -3: "Error while receiving data",
    -7: "Timeout error",
    -9: "There is no interface or DLL handle with the given ID",
    0: "No error",
    1: "Parameter syntax error",
    2: "Unknown command",
    5: "Unallowable move attempted on unreferenced axis, or move attempted with servo off",
    7: "Position out of limits",
    10: "Controller was stopped by command",
    15: "Invalid axis identifier",
    17: "Parameter out of range",
}


def translate_error(value):
    """Only for compatibility reasons. Please use GCSError.translate_error instead"""
    return GCSError.translate_error(value)


class GCSError(PIErrorBase):
    """GCSError exception."""

    def __init__(self, value, message=""):
        PIErrorBase.__init__(self, value, message)

    @staticmethod
    def translate_error(value):
        if not isinstance(value, int):
            return value
        try:
            return f"{_ERRMSG[value]} {value}"
        except KeyError:
            return str(value)
