"""Provides algorithms for extracting the fluorescence from detected ROIs and determining ROI colocalization in
multichannel recordings.
"""

from .masks import create_masks
from .extract import extract_traces
from .deconvolve import apply_oasis_deconvolution, compute_delta_fluorescence

__all__ = [
    "apply_oasis_deconvolution",
    "compute_delta_fluorescence",
    "create_masks",
    "extract_traces",
]
