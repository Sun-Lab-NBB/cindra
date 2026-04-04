"""Provides the diffeomorphic Demons image registration algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.ndimage
from ataraxis_base_utilities import console

from .pyramid import ScaleSpacePyramid
from .deformation import Deformation
from .spline_grid import SplineGrid

if TYPE_CHECKING:
    from numpy.typing import NDArray

_MINIMUM_GRID_DIMENSION: int = 4
"""The minimum B-spline grid dimension required for frozen edge constraints. Grids smaller than this cannot properly
constrain edge deformations."""


class DiffeomorphicDemonsRegistration:
    """Provides the diffeomorphic Demons registration pipeline for groupwise alignment of 2D images.

    Implements a variant of the Demons algorithm that produces diffeomorphic (smooth, invertible, topology-preserving)
    deformations using B-spline regularization. Uses backward mapping and groupwise registration to align all images
    to a common mean space.

    Args:
        images: Two or more 2D images to register. Images are converted to float32 if not already floating point.
        speed_factor: The relative force of the deformation transform. This is the most important parameter to tune.
            For most cases, a value between 1 and 5 is reasonable.
        scale_sampling: The number of iterations per scale level. Values between 20 and 30 are reasonable, but higher
            values yield better results. Algorithm speed scales linearly with this value.
        grid_sampling_factor: Determines how B-spline grid sampling scales with image scale. Lower values allow more
            deformation at coarser scales. Must be between 0 and 1.
        final_scale: The minimum scale (finest resolution) for the scale-space pyramid. Must be >= 0.5.
        final_grid_sampling: The B-spline grid spacing at the final (finest) scale level.
        smooth_scale: Determines whether to use smooth scale transitions between pyramid levels.
        injective: Determines whether to enforce injectivity constraint to ensure diffeomorphic (invertible)
            deformations.
        freeze_edges: Determines whether to freeze deformation values at image edges to prevent boundary artifacts.
        deformation_limit: The maximum allowed deformation magnitude per grid cell, relative to grid spacing.
        noise_factor: The regularization factor for intensity noise in the Demons force calculation.

    Attributes:
        _images: The processed images.
        _speed_factor: Cached speed_factor parameter.
        _scale_sampling: Cached scale_sampling parameter.
        _grid_sampling_factor: Cached grid_sampling_factor parameter.
        _final_scale: Cached final_scale parameter.
        _final_grid_sampling: Cached final_grid_sampling parameter.
        _smooth_scale: Cached smooth_scale parameter.
        _injective: Cached injective parameter.
        _freeze_edges: Cached freeze_edges parameter.
        _deformation_limit: Cached deformation_limit parameter.
        _noise_factor: Cached noise_factor parameter.
        _deformations: Maps image indices to their computed Deformation objects.
        _pyramids: Scale-space pyramids for each input image, initialized during registration.
        _cache: Internal cache for intermediate computation results.
        _interpolation_order: Current interpolation order used during registration (1 or 3).
    """

    def __init__(
        self,
        images: list[NDArray[np.float32]],
        speed_factor: float = 3.0,
        scale_sampling: int = 30,
        grid_sampling_factor: float = 1.0,
        final_scale: float = 1.0,
        final_grid_sampling: float = 16.0,
        smooth_scale: bool = True,
        injective: bool = True,
        freeze_edges: bool = True,
        deformation_limit: float = 1.0,
        noise_factor: float = 1.0,
    ) -> None:
        # Ensures that the input images use the fp32 precision, consistent with the rest of the cindra codebase.
        self._images: list[NDArray[np.float32]] = []
        for image in images:
            converted_image = image if image.dtype == np.float32 else image.astype(np.float32)
            self._images.append(converted_image)

        # Caches registration parameters to class attributes.
        self._speed_factor: float = speed_factor
        self._scale_sampling: int = scale_sampling
        self._grid_sampling_factor: float = grid_sampling_factor
        self._final_scale: float = final_scale
        self._final_grid_sampling: float = final_grid_sampling
        self._smooth_scale: bool = smooth_scale
        self._injective: bool = injective
        self._freeze_edges: bool = freeze_edges
        self._deformation_limit: float = deformation_limit
        self._noise_factor: float = noise_factor

        self._deformations: dict[int, Deformation] = {}

        # Tracks the runtime state initialized during registration.
        self._pyramids: list[ScaleSpacePyramid] | None = None
        self._cache: dict[str, tuple[tuple[int, int, float], Deformation | NDArray[np.float32]]] = {}
        self._interpolation_order: int = 1

    def get_deformation(self, image_index: int) -> Deformation:
        """Returns the deformation for the specified image.

        The deformation maps the image at the given index to the mean shape (groupwise registration result).

        Args:
            image_index: The index of the image (0-based).

        Returns:
            The deformation that aligns the specified image to the common mean space.
        """
        return self._deformations[image_index]

    def register(self, progress: bool = True) -> None:
        """Performs the multiscale registration process.

        Iteratively computes deformations from coarse to fine scales, updating the groupwise alignment at each step.

        Args:
            progress: Determines whether to display a progress bar to report the registration progress.
        """
        self._interpolation_order = 1  # Initializes the interpolation order to 1

        # The iteration factor controls smooth scale transitions between levels.
        iteration_factor = 0.5 ** (1.0 / self._scale_sampling)

        # Creates scale-space pyramids for each image.
        self._pyramids = [ScaleSpacePyramid(data=image, min_scale=self._final_scale) for image in self._images]

        # Computes maximum scale from image dimensions (quarter of largest dimension).
        max_scale = max(self._images[0].shape) * 0.25

        # Computes the number of scale levels needed to span from final_scale to max_scale.
        scale_level_count = 1
        while self._final_scale * 2 ** (scale_level_count - 1) < max_scale:
            scale_level_count += 1

        # Computes total iterations for the progress bar. When smooth_scale is enabled, the coarsest level is skipped.
        if self._smooth_scale:
            total_iterations = (scale_level_count - 1) * self._scale_sampling
        else:
            total_iterations = scale_level_count * self._scale_sampling  # pragma: no cover — smooth_scale=False path

        # Saves and restores the console's progress state to honor the progress parameter without affecting the global
        # state set by the pipeline entry point.
        previous_state = console.progress_enabled
        if progress:
            console.enable_progress()  # pragma: no cover — only when caller sets progress=True
        else:
            console.disable_progress()

        try:
            # Main registration loop: processes scales from coarse to fine.
            with console.progress(
                total=total_iterations,
                description="Registering recordings to a shared visual space",
                unit="iteration",
            ) as progress_bar:
                for level in reversed(range(scale_level_count)):
                    # Computes the scale at the current level.
                    scale = self._final_scale * 2**level
                    if self._smooth_scale:
                        scale *= 2 * iteration_factor

                    for iteration in range(1, self._scale_sampling + 1):
                        # Skips the coarsest level when using smooth scaling.
                        if self._smooth_scale and level >= scale_level_count - 1:
                            continue

                        # Switches to cubic interpolation for final iterations at finest scale.
                        if level == 0 and iteration > 0.75 * self._scale_sampling:
                            self._interpolation_order = 3

                        self._perform_iteration(level=level, iteration=iteration, scale=scale)
                        progress_bar.update(1)

                        # Smoothly decreases scale within each level.
                        if self._smooth_scale:
                            scale = max(self._final_scale, scale * iteration_factor)
        finally:
            if previous_state:  # pragma: no cover — restores caller's progress state
                console.enable_progress()
            else:
                console.disable_progress()

    def _perform_iteration(self, level: int, iteration: int, scale: float) -> None:
        """Performs one iteration of registration at the specified scale.

        Computes incremental deformations for all images and applies them to the running totals.

        Args:
            level: The current pyramid level index.
            iteration: The current iteration number within this level.
            scale: The current scale value.
        """
        iteration_key = (level, iteration, scale)

        # Computes incremental deformation for each image.
        incremental_deformations = []
        for image_index in range(len(self._images)):
            deformation = self._compute_groupwise_deformation(image_index=image_index, iteration_key=iteration_key)
            incremental_deformations.append(deformation)

        # Applies incremental deformations to the running totals.
        for image_index in range(len(self._images)):
            self._apply_incremental_deformation(
                image_index=image_index, incremental_deformation=incremental_deformations[image_index]
            )

    def _compute_groupwise_deformation(
        self, image_index: int, iteration_key: tuple[int, int, float]
    ) -> Deformation | None:
        """Computes the deformation for one image by averaging pairwise deformations to all other images.

        Args:
            image_index: The index of the image for which to compute the deformation.
            iteration_key: Tuple of (level, iteration, scale) identifying this iteration.

        Returns:
            The averaged deformation, or None if the grid would be too small.
        """
        scale = iteration_key[2]

        # Returns None if the B-spline grid would be too small for frozen edges.
        if self._freeze_edges:
            grid_sampling = self._compute_grid_sampling(scale=scale)
            image_height, image_width = self._images[0].shape
            grid_shape = SplineGrid.compute_grid_shape(
                field_height=image_height, field_width=image_width, grid_sampling=grid_sampling
            )
            # pragma: no cover — grid too coarse at this scale level
            if all(dimension < _MINIMUM_GRID_DIMENSION for dimension in grid_shape):  # pragma: no cover
                return None

        # Accumulates pairwise deformations from this image to all others.
        image_height, image_width = self._images[0].shape
        total_deformation = Deformation.identity(height=image_height, width=image_width)
        pair_count = 0

        for other_index in range(len(self._images)):
            if image_index == other_index:
                continue

            pairwise_deformation = self._compute_pairwise_deformation(
                source_index=image_index, target_index=other_index, iteration_key=iteration_key
            )
            if pairwise_deformation is not None:
                pair_count += 1
                total_deformation += pairwise_deformation

        # Averages the accumulated deformations.
        if pair_count > 1:  # pragma: no cover — only with >2 images in groupwise registration
            total_deformation = total_deformation.scale(factor=1.0 / pair_count)

        return total_deformation

    def _compute_pairwise_deformation(
        self, source_index: int, target_index: int, iteration_key: tuple[int, int, float]
    ) -> Deformation:
        """Computes the Demons deformation from source image to target image.

        Uses symmetric Demons forces computed from both image gradients.

        Args:
            source_index: Index of the source image.
            target_index: Index of the target image.
            iteration_key: Tuple of (level, iteration, scale) identifying this iteration.

        Returns:
            The regularized deformation from source to target.
        """
        scale = iteration_key[2]

        # Checks cache for this pair or its symmetric counterpart.
        cached = self._get_cached(key=f"deform_{source_index}_{target_index}", iteration_key=iteration_key)
        if isinstance(cached, Deformation):  # pragma: no cover — cache hit for previously computed pair
            return cached

        cached = self._get_cached(key=f"deform_{target_index}_{source_index}", iteration_key=iteration_key)
        if isinstance(cached, Deformation):
            # Negates the cached symmetric result by scaling by -1.
            return cached.scale(factor=-1.0)

        # Gets images and their gradients at the current scale.
        source_image, source_gradient = self._get_image_and_gradient(
            image_index=source_index, iteration_key=iteration_key
        )
        target_image, target_gradient = self._get_image_and_gradient(
            image_index=target_index, iteration_key=iteration_key
        )

        # Computes gradient magnitude squared for both images.
        source_gradient_magnitude_squared = source_gradient[0] ** 2 + source_gradient[1] ** 2
        target_gradient_magnitude_squared = target_gradient[0] ** 2 + target_gradient[1] ** 2

        # Computes intensity difference and its square.
        intensity_difference = source_image - target_image
        intensity_difference_squared = intensity_difference**2

        # Computes Demons denominators with noise regularization.
        source_denominator = source_gradient_magnitude_squared + self._noise_factor**2 * intensity_difference_squared
        target_denominator = target_gradient_magnitude_squared + self._noise_factor**2 * intensity_difference_squared

        # Prevents division by zero.
        source_denominator[source_denominator == 0] = np.inf
        target_denominator[target_denominator == 0] = np.inf

        # Computes symmetric Demons force field (negative for backward mapping).
        speed = -self._speed_factor
        field_y = (
            intensity_difference
            * (source_gradient[0] / source_denominator + target_gradient[0] / target_denominator)
            * speed
        )
        field_x = (
            intensity_difference
            * (source_gradient[1] / source_denominator + target_gradient[1] / target_denominator)
            * speed
        )

        # Regularizes using B-spline grid to ensure diffeomorphism.
        force_deformation = Deformation(field_y=field_y.astype(np.float32), field_x=field_x.astype(np.float32))
        regularized_deformation = self._regularize_deformation(
            scale=scale, deformation=force_deformation, image_shape=source_image.shape
        )

        self._set_cached(
            key=f"deform_{source_index}_{target_index}", iteration_key=iteration_key, data=regularized_deformation
        )
        return regularized_deformation

    def _get_image_and_gradient(
        self, image_index: int, iteration_key: tuple[int, int, float]
    ) -> tuple[NDArray[np.float32], tuple[NDArray[np.float32], NDArray[np.float32]]]:
        """Returns the image at the current scale along with its gradient.

        Args:
            image_index: Index of the image to retrieve.
            iteration_key: Tuple of (level, iteration, scale) identifying this iteration.

        Returns:
            A tuple of (image, (gradient_y, gradient_x)).
        """
        scale = iteration_key[2]

        # Tries to retrieve cached image.
        cached = self._get_cached(key=f"img_{image_index}", iteration_key=iteration_key)
        if isinstance(cached, np.ndarray):  # pragma: no cover — cache hit for previously deformed image
            image = cached
        else:
            image = self._get_deformed_image(image_index=image_index, scale=scale)
            self._set_cached(key=f"img_{image_index}", iteration_key=iteration_key, data=image)

        # Computes gradient using central differences.
        gradient_kernel = np.array([0.5, 0, -0.5], dtype=np.float32)
        gradient_y = scipy.ndimage.convolve1d(input=image, weights=gradient_kernel, axis=0, mode="nearest")
        gradient_x = scipy.ndimage.convolve1d(input=image, weights=gradient_kernel, axis=1, mode="nearest")

        return image, (gradient_y, gradient_x)

    def _get_deformed_image(self, image_index: int, scale: float) -> NDArray[np.float32]:
        """Returns the image at the specified scale with current deformation applied.

        Args:
            image_index: Index of the image to retrieve.
            scale: The scale at which to retrieve the image.

        Returns:
            The deformed image at the specified scale.
        """
        # Validates that pyramids have been initialized (should always be true when this method is called).
        if self._pyramids is None:  # pragma: no cover — defensive guard; register() always initializes pyramids
            message = "Unable to retrieve image. The pyramids have not been initialized, call register() first."
            console.error(message=message, error=RuntimeError)

        # noinspection PyUnresolvedReferences
        image = self._pyramids[image_index].get_scale(scale=scale)

        # Applies current accumulated deformation if one exists.
        deformation = self._deformations.get(image_index, None)
        if deformation is not None:
            deformation = deformation.resize_field(new_height=image.shape[0], new_width=image.shape[1])
            self._deformations[image_index] = deformation
            image = deformation.apply_deformation(data=image, interpolation=self._interpolation_order)

        return image

    def _apply_incremental_deformation(self, image_index: int, incremental_deformation: Deformation | None) -> None:
        """Applies an incremental deformation to the running total for an image.

        Args:
            image_index: Index of the image to update.
            incremental_deformation: The incremental deformation to apply, or None to skip.
        """
        if incremental_deformation is None or incremental_deformation.is_identity:  # pragma: no cover — no-op skip
            return

        # Gets or creates the current accumulated deformation.
        current_deformation = self._deformations.get(image_index, None)
        if current_deformation is None:
            image_height, image_width = self._images[0].shape
            current_deformation = Deformation.identity(height=image_height, width=image_width)

        # Resizes to match and composes the deformations.
        current_deformation = current_deformation.resize_field(
            new_height=incremental_deformation.field_shape[0], new_width=incremental_deformation.field_shape[1]
        )
        self._deformations[image_index] = current_deformation.compose(incremental_deformation)

    def _regularize_deformation(
        self, scale: float, deformation: Deformation, image_shape: tuple[int, ...] | None = None
    ) -> Deformation:
        """Regularizes a deformation to ensure diffeomorphism using B-spline constraints.

        Args:
            scale: The current scale level.
            deformation: The raw deformation to regularize.
            image_shape: The shape of the image at the current working resolution. When provided and different from
                the original image resolution, the grid sampling is scaled by the downsample ratio to maintain
                correct regularization strength at reduced resolutions.

        Returns:
            The regularized deformation.
        """
        grid_sampling = self._compute_grid_sampling(scale=scale)

        # Scales grid_sampling to match the working resolution when images are downsampled. The injectivity constraint
        # uses the original-pixel-unit grid_sampling since both scale and grid_sampling must be in the same coordinate
        # system.
        original_grid_sampling = grid_sampling
        if image_shape is not None and self._images is not None and len(self._images) > 0:
            downsample_ratio = image_shape[0] / self._images[0].shape[0]
            if downsample_ratio < 1.0:
                grid_sampling = grid_sampling * downsample_ratio

        # Computes injectivity constraint factor based on scale and grid sampling in original-pixel units.
        injective_factor = 0.9
        if self._injective:
            injective_factor = min(self._deformation_limit * scale / original_grid_sampling, 0.9)

        regularized = deformation.regularize(
            grid_sampling=grid_sampling,
            injective=self._injective,
            injective_factor=injective_factor,
            freeze_edges=self._freeze_edges,
        )

        # Returns original deformation if regularization failed (grid too small).
        if regularized is None:
            return deformation

        return regularized

    def _compute_grid_sampling(self, scale: float) -> float:
        """Computes the B-spline grid sampling for the given scale.

        The grid sampling increases linearly from final_grid_sampling at final_scale.

        Args:
            scale: The current scale value.

        Returns:
            The grid sampling value for this scale.
        """
        scale_difference = scale - self._final_scale
        scale_factor = self._grid_sampling_factor * self._final_grid_sampling
        return scale_difference * scale_factor + self._final_grid_sampling

    def _get_cached(self, key: str, iteration_key: tuple[int, int, float]) -> Deformation | NDArray[np.float32] | None:
        """Retrieves cached data if the iteration key matches.

        Args:
            key: The cache key.
            iteration_key: The iteration identifier to validate against.

        Returns:
            The cached data if valid, otherwise None.
        """
        entry = self._cache.get(key)
        if entry and entry[0] == iteration_key:
            return entry[1]
        return None

    def _set_cached(
        self, key: str, iteration_key: tuple[int, int, float], data: Deformation | NDArray[np.float32]
    ) -> None:
        """Stores data in the cache with an iteration key.

        Args:
            key: The cache key.
            iteration_key: The iteration identifier for validation.
            data: The data to cache.
        """
        self._cache[key] = (iteration_key, data)
