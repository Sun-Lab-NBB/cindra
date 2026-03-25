"""Statically resolves and stores the Python and library version information used in timing dataclasses."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from importlib_metadata import metadata as _metadata

if TYPE_CHECKING:
    from importlib_metadata import PackageMetadata

_package_metadata: PackageMetadata | None = _metadata("cindra")

if _package_metadata is None:  # pragma: no cover — unreachable when package is installed
    version: str = "unknown"
else:
    version = _package_metadata["version"]
"""The cindra library version string, resolved from package metadata at import time."""

python_version: str = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
"""The Python interpreter version string in major.minor.micro format."""
