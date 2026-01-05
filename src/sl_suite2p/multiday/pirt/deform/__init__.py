"""The deform module implements the Deformation class for image registration.

The `Deformation` class represents deformations in world coordinates using
an array for each dimension, describing the deformation for each pixel/voxel.

The `SplineGridHelper` class provides B-spline grid functionality for
diffeomorphic regularization, used internally by registration algorithms.
"""

from .deformation import Deformation, SplineGridHelper

__all__ = ["Deformation", "SplineGridHelper"]
