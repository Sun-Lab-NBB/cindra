"""Provides algorithms for extracting the fluorescence from detected ROIs and determining ROI colocalization in
multichannel recordings.
"""

from .deconvolve import apply_oasis_deconvolution, compute_delta_fluorescence
from .extract import extract_traces
from .masks import create_masks

__all__ = [
    "apply_oasis_deconvolution",
    "compute_delta_fluorescence",
    "create_masks",
    "extract_traces",
]
