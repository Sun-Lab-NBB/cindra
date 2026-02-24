"""Statically resolves and stores the Python and library version information used in timing dataclasses."""

import sys

from importlib_metadata import metadata as _metadata

_package_metadata = _metadata("cindra")

# The cindra library version string, resolved from package metadata at import time.
if _package_metadata is None:
    version: str = "unknown"
else:
    version = _package_metadata["version"]

# The Python interpreter version string in major.minor.micro format.
python_version: str = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
