"""Provides algorithms for extracting the fluorescence from detected ROIs and determining ROI colocalization in
multichannel recordings."""

from .extract import extract_traces

__all__ = [
    "extract_traces",
]
