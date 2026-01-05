"""The splinegrid module implements functionality for spline grids.

Spline grids are used to represent deformations using B-splines.
"""

from ._splinegridclasses import (
    FD,
    SplineGrid,
    GridContainer,
    GridInterface,
    FieldDescription,
)

__all__ = [
    "FD",
    "FieldDescription",
    "GridContainer",
    "GridInterface",
    "SplineGrid",
]
