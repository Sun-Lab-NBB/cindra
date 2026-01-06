"""Diffeomorphic Demons registration for multi-day cell tracking.

This module provides the DiffeomorphicDemonsRegistration class for registering
multiple 2D images using a diffeomorphic variant of the Demons algorithm with
B-spline regularization.

Copyright 2010-2017 (C) Almar Klein (original pirt library)
"""

from dataclasses import dataclass

import numpy as np
import scipy.ndimage

from .pyramid import ScaleSpacePyramid
from .deformation import Deformation
from .spline_grid import SplineGrid


@dataclass
class RegistrationParameters:
    """Parameters for DiffeomorphicDemonsRegistration."""

    speed_factor: float = 3.0
    """The relative force of the deformation transform. This is the most important parameter to tune. For most cases,
    a value between 1 and 5 is reasonable."""

    scale_sampling: int = 25
    """The number of iterations for each scale level. Values between 20 and 30 are reasonable in most situations, but
    higher values yield better results. The speed of the algorithm scales linearly with this value."""

    grid_sampling_factor: float = 0.5
    """Determines how grid sampling scales with image scale. Lower values allow more deformation at higher scales.
    Must be between 0 and 1."""

    final_scale: float = 1.0
    """The minimum scale (finest resolution) for the scale-space pyramid. Must be >= 0.5."""

    final_grid_sampling: float = 16.0
    """The B-spline grid spacing at the final (finest) scale level."""

    smooth_scale: bool = True
    """Whether to use smooth scale transitions between pyramid levels."""

    injective: bool = True
    """Whether to enforce injectivity constraint to ensure diffeomorphic (invertible) deformations."""

    frozenedge: bool = True
    """Whether to freeze deformation values at image edges to prevent boundary artifacts."""

    deform_limit: float = 1.0
    """The maximum allowed deformation magnitude per grid cell, relative to grid spacing."""

    noise_factor: float = 1.0
    """Regularization factor for intensity noise in the Demons force calculation."""


class DiffeomorphicDemonsRegistration:
    """Diffeomorphic Demons registration for 2 or more images.

    A variant of the Demons algorithm that produces diffeomorphic (smooth,
    invertible, topology-preserving) deformations using B-spline regularization.
    Uses backward mapping and groupwise registration to align all images to a
    common mean space.

    Parameters
    ----------
    *images : numpy arrays
        Two or more 2D images to register. Images are converted to float32
        if not already floating point.

    Attributes:
    ----------
    params : RegistrationParameters
        Registration parameters. See RegistrationParameters for details.
    """

    def __init__(self, *images):
        if len(images) < 2:
            raise ValueError("Need at least two images for registration.")

        # Convert images to float and store
        self._images = []
        for im in images:
            if not isinstance(im, np.ndarray):
                raise ValueError("Images must be numpy arrays.")
            if im.dtype not in [np.float32, np.float64]:
                im = im.astype(np.float32)
            self._images.append(im)

        # Initialize deformations storage
        self._deforms = {}

        # Initialize parameters with defaults
        self._params = RegistrationParameters()

        # Runtime state
        self._pyramids = None
        self._buffer = {}
        self._current_interp_order = 1
        self._max_scale = None

    @property
    def params(self) -> RegistrationParameters:
        """Get the parameters object."""
        return self._params

    def get_deform(self, i: int) -> Deformation:
        """Get the deformation for image with index i.

        The deformation maps image i to the mean shape (groupwise registration).

        Parameters
        ----------
        i : int
            Image index (0-based).

        Returns:
        -------
        Deformation
            The deformation for the specified image.
        """
        if not isinstance(i, int):
            raise ValueError("Image index must be an integer.")
        if i not in self._deforms:
            raise KeyError(f"Deformation for index {i} is not available.")
        return self._deforms[i]

    def register(self, verbose: int = 1):
        """Perform the registration process.

        Parameters
        ----------
        verbose : int
            Verbosity level. 0 = silent, 1 = progress, 2 = detailed.
        """
        self._current_interp_order = 1

        # Scale parameters
        final_scale = float(self._params.final_scale)
        scale_sampling = int(self._params.scale_sampling)
        smooth_scale = bool(self._params.smooth_scale)
        iter_factor = 0.5 ** (1.0 / scale_sampling)

        # Validate final_scale
        if final_scale < 0.5:
            raise ValueError("final_scale must be at least 0.5.")

        # Create scale-space pyramids for each image
        self._pyramids = [ScaleSpacePyramid(im, final_scale) for im in self._images]

        # Calculate max scale from image dimensions
        ranges = [sh * 1.0 for sh in self._images[0].shape]
        self._max_scale = max(ranges) * 0.25

        # Calculate number of scale levels needed
        scale_levels = 1
        while final_scale * 2 ** (scale_levels - 1) < self._max_scale:
            scale_levels += 1

        if verbose >= 1:
            print(f"{self.__class__.__name__}: ")

        # Main registration loop through scale space (coarse to fine)
        for level in reversed(range(scale_levels)):
            scale = final_scale * 2**level
            if smooth_scale:
                scale *= 2 * iter_factor

            for iteration in range(1, scale_sampling + 1):
                # Skip highest scale level with smooth scaling
                if smooth_scale and level >= scale_levels - 1:
                    continue

                # Use higher interpolation order in final iterations
                if level == 0 and iteration > 0.75 * scale_sampling:
                    self._current_interp_order = 3

                # Perform one registration iteration
                self._register_iteration(level, iteration, scale)

                if verbose == 1:
                    print(f"  iter {level}-{iteration} scale {scale:.2f}")
                elif verbose > 1:
                    print(f"Registration iter {level}-{iteration} at scale {scale:.2f}")

                # Update scale for next iteration
                if smooth_scale:
                    scale = max(final_scale, scale * iter_factor)

    def _register_iteration(self, level: int, iteration: int, scale: float):
        """Perform one iteration of registration at the specified scale."""
        n_images = len(self._images)
        iter_info = (level, iteration, scale)

        # Calculate deformation for each image
        deforms = []
        for i in range(n_images):
            deform = self._compute_groupwise_deform(i, iter_info)
            deforms.append(deform)

        # Apply deformations
        for i in range(n_images):
            self._apply_delta_deform(i, deforms[i])

    def _compute_groupwise_deform(self, i: int, iter_info: tuple) -> Deformation:
        """Compute deformation for image i by averaging deforms to all other images."""
        scale = iter_info[2]

        # Check if grid would be too small
        if self._params.frozenedge:
            grid_sampling = self._get_grid_sampling(scale)
            field_shape = self._images[0].shape
            grid_shape = SplineGrid.compute_grid_shape(field_shape[0], field_shape[1], grid_sampling)
            if all(s < 4 for s in grid_shape):
                return None

        # Accumulate deformations from all image pairs
        total_deform = Deformation()
        count = 0

        for j in range(len(self._images)):
            if i == j:
                continue

            deform = self._compute_demons_deform(i, j, iter_info)
            if deform is not None:
                count += 1
                total_deform = total_deform + deform

        # Average the deformations
        if count > 1:
            total_deform = total_deform.scale(1.0 / count)

        return total_deform

    def _compute_demons_deform(self, i: int, j: int, iter_info: tuple) -> Deformation:
        """Compute demons deformation from image i to image j."""
        scale = iter_info[2]

        # Try to use cached symmetric result
        cached = self._get_cached(f"deform_{i}_{j}", iter_info)
        if cached is not None:
            return cached

        cached = self._get_cached(f"deform_{j}_{i}", iter_info)
        if cached is not None:
            # Negate the cached result for symmetric deformation
            for grid in cached:
                grid._knots = -grid._knots
            return cached

        # Get images and their gradients
        im1, grad1 = self._get_image_and_gradient(i, iter_info)
        im2, grad2 = self._get_image_and_gradient(j, iter_info)

        # Calculate gradient magnitude squared
        norm1 = sum(g**2 for g in grad1)
        norm2 = sum(g**2 for g in grad2)

        # Calculate intensity difference
        diff = im1 - im2
        diff_sq = diff**2

        # Compute denominators with noise regularization
        alpha = float(self._params.noise_factor)
        denom1 = norm1 + alpha**2 * diff_sq
        denom2 = norm2 + alpha**2 * diff_sq

        # Prevent division by zero
        denom1[denom1 == 0] = np.inf
        denom2[denom2 == 0] = np.inf

        # Compute demons force field (negative speed for backward mapping)
        speed = -float(self._params.speed_factor)

        fields = []
        for d in range(diff.ndim):
            force = diff * (grad1[d] / denom1 + grad2[d] / denom2) * speed
            fields.append(force)

        # Regularize using B-spline grid to ensure diffeomorphism
        force_deform = Deformation(*fields)
        deform = self._regularize_diffeomorphic(scale, force_deform)

        # Cache and return
        self._set_cached(f"deform_{i}_{j}", iter_info, deform)
        return deform

    def _get_image_and_gradient(self, image_id: int, iter_info: tuple):
        """Get image at current scale and compute its gradient."""
        scale = iter_info[2]

        # Try cached image
        cached = self._get_cached(f"img_{image_id}", iter_info)
        if cached is not None:
            im = cached
        else:
            # Get deformed image at scale
            im = self._get_deformed_image(image_id, scale)
            self._set_cached(f"img_{image_id}", iter_info, im)

        # Compute gradient using central differences
        gradient = []
        kernel = np.array([0.5, 0, -0.5], dtype="float64")
        for d in range(im.ndim):
            grad = scipy.ndimage.convolve1d(im, kernel, d, mode="nearest")
            gradient.append(grad)

        return im, tuple(gradient)

    def _get_deformed_image(self, i: int, scale: float):
        """Get image i at specified scale, with current deformation applied."""
        # Get image from pyramid at scale
        im = self._pyramids[i].get_scale(scale)

        # Apply current deformation if exists
        deform = self._deforms.get(i, None)
        if deform is not None:
            deform = deform.resize_field(im)
            self._deforms[i] = deform
            im = deform.apply_deformation(im, self._current_interp_order)

        return im

    def _apply_delta_deform(self, i: int, deform: Deformation):
        """Apply incremental deformation to image i's total deformation."""
        if deform is None or deform.is_identity:
            return

        # Get current deformation or create identity
        current = self._deforms.get(i, None)
        if current is None:
            current = Deformation(self._images[0].ndim)

        # Resize to match and compose deformations
        current = current.resize_field(deform)
        self._deforms[i] = current.compose(deform)

    def _regularize_diffeomorphic(self, scale: float, deform: Deformation) -> Deformation:
        """Regularize deformation to ensure diffeomorphism using B-spline grid."""
        grid_sampling = self._get_grid_sampling(scale)

        injective = self._params.injective
        frozenedge = self._params.frozenedge

        # Calculate injectivity constraint factor
        injective_factor = 0.9
        if injective:
            deform_limit = float(self._params.deform_limit)
            injective_factor = min(deform_limit * scale / grid_sampling, 0.9)

        return deform.regularize(grid_sampling, None, injective, injective_factor, frozenedge)

    def _get_grid_sampling(self, scale: float) -> float:
        """Calculate B-spline grid sampling for current scale."""
        final_grid_sampling = float(self._params.final_grid_sampling)
        grid_factor = float(self._params.grid_sampling_factor)
        final_scale = float(self._params.final_scale)

        gsf = grid_factor * final_grid_sampling
        gsb = final_grid_sampling
        return (scale - final_scale) * gsf + gsb

    # Simple caching helpers
    def _get_cached(self, key: str, check):
        """Get cached data if check matches."""
        entry = self._buffer.get(key)
        if entry and entry[0] == check:
            return entry[1]
        return None

    def _set_cached(self, key: str, check, data):
        """Cache data with check value."""
        self._buffer[key] = (check, data)
