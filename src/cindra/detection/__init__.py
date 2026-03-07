"""Provides algorithms for segmenting and describing ROIs from motion-corrected recordings."""

from .utils import compute_spatial_taper_mask, compute_registration_blocks, compute_block_smoothing_kernel
from .detect import detect_plane_rois
from .tracking import track_rois_across_recordings
from .detect_rois import extend_roi
from .roi_statistics import (
    compute_roi_statistics,
    estimate_diameter_from_rois,
    compute_median_pixel_position,
)

__all__ = [
    "compute_block_smoothing_kernel",
    "compute_median_pixel_position",
    "compute_registration_blocks",
    "compute_roi_statistics",
    "compute_spatial_taper_mask",
    "detect_plane_rois",
    "estimate_diameter_from_rois",
    "extend_roi",
    "track_rois_across_recordings",
]
