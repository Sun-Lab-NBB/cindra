"""This package provides the algorithms and tools for carrying out the multi-day sl-suite2p processing pipeline.
This pipeline is based on the original implementation found here:
https://github.com/sprustonlab/multiday-suite2p-public/tree/main.
"""

from .process import extract_session_traces

__all__ = [
    "extract_session_traces",
]
