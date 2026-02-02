"""Enhanced suite2p implementation that includes a pipeline to track cell activity across sessions (days).

This implementation of the popular suite2p (https://github.com/MouseLand/suite2p) features various documentation, source
code and API augmentations performed in the Sun (NeuroAI) lab to improve its runtime efficiency and user experience.
Additionally, it features the across-day cell tracking pipeline featured in the OSM manuscript
(https://www.nature.com/articles/s41586-024-08548-w), which has been integrated into the suite2p codebase and API
structure.

Original suite2p copyright:
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.

Sun lab code changes copyright:
Copyright © 2025 Cornell University, Authored by Ivan Kondratyev and Kushaan Gupta.

For documentation and additional information, see the sl-suite2p repository: https://github.com/Sun-Lab-NBB/suite2p
"""

# Configures numba threading layer for parallel execution across all modules. This must be set before any numba
# functions are compiled, hence it appears before other imports.
from numba import config

config.THREADING_LAYER = "tbb"

from ataraxis_base_utilities import console

from .multiday import show_images_with_masks
from .pipeline import (
    MultiDayJobNames,
    SingleDayJobNames,
    get_session_root,
    process_multi_day,
    process_single_day,
)
from .detection import ROI
from .multi_day import (
    run_s2p_multiday,
    resolve_multiday_ops,
    discover_multiday_cells,
    extract_multiday_fluorescence,
)
from .single_day import run_s2p, process_plane, combine_planes, resolve_processing_contexts
from .configuration import CombinedData, RuntimeContext, MultiDayConfiguration, SingleDayConfiguration

# Ensures console output is enabled whenever the suite2p library is imported. In sl-suite2p, the 'Console' class is
# used over 'print' for all terminal outputs. With minimal configuration, this class can be extended to log terminal
# outputs instead of or in addition to sending them to the terminal.
if not console.enabled:
    console.enable()

__all__ = [
    "ROI",
    "CombinedData",
    "MultiDayConfiguration",
    "MultiDayJobNames",
    "RuntimeContext",
    "SingleDayConfiguration",
    "SingleDayJobNames",
    "combine_planes",
    "discover_multiday_cells",
    "extract_multiday_fluorescence",
    "get_session_root",
    "process_multi_day",
    "process_plane",
    "process_single_day",
    "resolve_multiday_ops",
    "resolve_processing_contexts",
    "run_s2p",
    "run_s2p_multiday",
    "show_images_with_masks",
]
