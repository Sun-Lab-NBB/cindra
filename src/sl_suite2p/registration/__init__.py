"""This module provides image registration algorithms for the sl-suite2p processing pipelines.

This module unifies two registration approaches: phase correlation-based registration from suite2p (single-day
frame-to-frame motion correction) and diffeomorphic Demons registration from pirt (multi-day cross-session
anatomical alignment).

Copyright:
    Single-day registration code (register.py, rigid.py, nonrigid.py, bidiphase_correction.py, zalign.py, metrics.py,
        utils.py):
        Copyright 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.

    Multi-day registration code (diffeomorphic.py, deformation.py, spline_grid.py, pyramid.py):
        Original pirt library Copyright 2010-2017 Almar Klein, University of Twente.

    All code modifications and integration:
        Copyright 2024-2025 Sun Lab, Authored by Ivan Kondratyev (Inkaros).
"""

from .utils import compute_spatial_taper_mask
from .zalign import compute_zpos
from .metrics import get_pc_metrics
from .register import registration_wrapper, create_enhanced_mean_image, save_registration_outputs_to_ops
from .deformation import Deformation
from .diffeomorphic import DiffeomorphicDemonsRegistration

__all__ = [
    "Deformation",
    "DiffeomorphicDemonsRegistration",
    "compute_spatial_taper_mask",
    "compute_zpos",
    "create_enhanced_mean_image",
    "get_pc_metrics",
    "registration_wrapper",
    "save_registration_outputs_to_ops",
]
