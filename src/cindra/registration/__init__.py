"""Provides algorithms for correcting within-recording motion and registering multiple recording sessions to the same
reference field of view.
"""

from .register import register_plane
from .deformation import Deformation
from .z_alignment import compute_z_position
from .diffeomorphic import DiffeomorphicDemonsRegistration
from .register_sessions import register_sessions, project_templates_to_sessions

__all__ = [
    "Deformation",
    "DiffeomorphicDemonsRegistration",
    "compute_z_position",
    "project_templates_to_sessions",
    "register_plane",
    "register_sessions",
]
