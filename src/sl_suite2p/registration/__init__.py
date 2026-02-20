"""Provides algorithms for correcting within-recording motion and registering multiple recording sessions to the same
reference field of view.
"""

from .register import register_plane
from .deformation import Deformation
from .diffeomorphic import DiffeomorphicDemonsRegistration
from .register_sessions import register_sessions, project_templates_to_sessions

__all__ = [
    "Deformation",
    "DiffeomorphicDemonsRegistration",
    "project_templates_to_sessions",
    "register_plane",
    "register_sessions",
]
