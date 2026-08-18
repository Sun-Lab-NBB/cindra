"""Provides algorithms for segmenting and describing ROIs from motion-corrected recordings."""

from .utils import compute_spatial_taper_mask, compute_registration_blocks
from .detect import detect_plane_rois
from .tracking import track_rois_across_recordings
from .detect_rois import extend_roi
from .roi_statistics import compute_roi_statistics

__all__ = [
    "compute_registration_blocks",
    "compute_roi_statistics",
    "compute_spatial_taper_mask",
    "detect_plane_rois",
    "extend_roi",
    "track_rois_across_recordings",
]
