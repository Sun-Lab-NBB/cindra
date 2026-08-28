"""Provides algorithms for correcting within-recording motion and registering recordings to a shared field of view."""

from .register import register_plane
from .register_recordings import register_recordings, project_templates_to_recordings

__all__ = [
    "project_templates_to_recordings",
    "register_plane",
    "register_recordings",
]
