"""[SIMULATION] Mirror of pipython.pidevice.pierror_base (same behaviour)."""


class PIErrorBase(Exception):
    """GCSError exception."""

    PI_NO_ERROR = 0

    def __init__(self, value, message=""):
        Exception.__init__(self)
        if isinstance(value, PIErrorBase):
            self.val = value.val
            self.msg = value.msg
        else:
            self.val = value
            self.msg = self.translate_error(value)
        if message:
            self.msg += f": {message}"

    def __str__(self):
        return self.msg

    def __repr__(self):
        return self.msg

    def __eq__(self, other):
        return self.val == other

    def __hash__(self):
        return hash(self.val)

    def __ne__(self, other):
        return self.val != other

    @staticmethod
    def translate_error(value):
        return str(value)
