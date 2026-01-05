"""Small utility classes."""

import re

try:  # pragma: no cover
    from collections import OrderedDict as _dict  # noqa
except ImportError:
    _dict = dict


def isidentifier(s):
    # http://stackoverflow.com/questions/2544972/
    if not isinstance(s, str):
        return False
    return re.match(r"^\w+$", s, re.UNICODE) and re.match(r"^[0-9]", s) is None


# Copied with changes from Pyzo/zon
class Parameters(_dict):
    """A dict in which the items can be get/set as attributes."""

    __reserved_names__ = dir(_dict())  # Also from OrderedDict
    __pure_names__ = dir(dict())

    __slots__ = []

    def __repr__(self):
        identifier_items = []
        nonidentifier_items = []
        for key, val in self.items():
            if isidentifier(key):
                identifier_items.append("%s=%r" % (key, val))
            else:
                nonidentifier_items.append("(%r, %r)" % (key, val))
        if nonidentifier_items:
            return "Parameters([%s], %s)" % (", ".join(nonidentifier_items), ", ".join(identifier_items))
        return "Parameters(%s)" % (", ".join(identifier_items))

    def __str__(self):
        # Get alignment value
        c = 0
        for key in self:
            c = max(c, len(key))

        # How many chars left (to print on less than 80 lines)
        charsLeft = 79 - (c + 6)

        s = "<%i parameters>\n" % len(self)
        for key in self.keys():
            valuestr = repr(self[key])
            if len(valuestr) > charsLeft:
                valuestr = valuestr[: charsLeft - 3] + "..."
            s += key.rjust(c + 4) + ": %s\n" % (valuestr)
        return s

    def __getattribute__(self, key):
        try:
            return object.__getattribute__(self, key)
        except AttributeError:
            if key in self:
                return self[key]
            raise

    def __setattr__(self, key, val):
        if key in self.__class__.__reserved_names__:
            # Either let OrderedDict do its work, or disallow
            if key not in self.__class__.__pure_names__:
                return _dict.__setattr__(self, key, val)
            raise AttributeError("Reserved name, this key can only " + "be set via ``d[%r] = X``" % key)
        # if isinstance(val, dict): val = Dict(val) -> no, makes a copy!
        self[key] = val

    def __dir__(self):
        names = [k for k in self.keys() if isidentifier(k)]
        return self.__class__.__reserved_names__ + names


