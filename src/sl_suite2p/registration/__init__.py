"""Provides algorithms for correcting within-recording motion and registering multiple recording sessions to the same
reference field of view.
"""

from .utils import compute_spatial_taper_mask
from .nonrigid import compute_registration_blocks
from .register import register_plane
from .deformation import Deformation
from .diffeomorphic import DiffeomorphicDemonsRegistration

__all__ = [
    "Deformation",
    "DiffeomorphicDemonsRegistration",
    "compute_registration_blocks",
    "compute_spatial_taper_mask",
    "register_plane",
]
