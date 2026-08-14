"""Statically resolves and stores the Python and library version information used in timing dataclasses."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from importlib.metadata import metadata as _metadata

if TYPE_CHECKING:
    from importlib.metadata import PackageMetadata

_package_metadata: PackageMetadata = _metadata("cindra")
"""The distribution metadata cindra was installed with, read once at import time."""

VERSION: str = _package_metadata["version"]
"""The cindra library version string, resolved from package metadata at import time."""

PYTHON_VERSION: str = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
"""The Python interpreter version string in major.minor.micro format."""
