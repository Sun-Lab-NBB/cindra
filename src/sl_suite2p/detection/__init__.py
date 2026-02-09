"""Provides algorithms for segmenting and describing ROIs from motion-corrected recordings."""

from .detect import detect_plane_rois
from .detect_rois import extend_roi
from .roi_statistics import compute_roi_statistics, compute_median_pixel_position

__all__ = [
    "compute_median_pixel_position",
    "compute_roi_statistics",
    "detect_plane_rois",
    "extend_roi",
]
