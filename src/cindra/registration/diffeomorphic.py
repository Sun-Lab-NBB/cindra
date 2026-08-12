"""Provides the diffeomorphic Demons image registration algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numba
import numpy as np
import scipy.ndimage
from ataraxis_base_utilities import console

from .pyramid import ScaleSpacePyramid
from .deformation import Deformation
from .spline_grid import SplineGrid, MINIMUM_KNOTS_FOR_FROZEN_EDGES

if TYPE_CHECKING:
    from numpy.typing import NDArray

_MINIMUM_GROUP_SIZE: int = 2
"""The minimum number of images a groupwise registration run requires."""


class DiffeomorphicDemonsRegistration:
    """Provides the diffeomorphic Demons registration pipeline for groupwise alignment of 2D images.

    Implements a variant of the Demons algorithm that produces diffeomorphic (smooth, invertible, topology-preserving)
    deformations using B-spline regularization. Uses backward mapping and groupwise registration to align all images
    to a common mean space.

    Args:
        images: Two or more 2D images to register. Images that do not already use the float32 dtype are converted to
            float32.
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

    Raises:
        ValueError: If fewer than two images are supplied.
    """

    def __init__(
        self,
        images: list[NDArray[np.float32]],
        speed_factor: float = 3.0,
        scale_sampling: int = 30,
        grid_sampling_factor: float = 1.0,
        final_scale: float = 1.0,
        final_grid_sampling: float = 16.0,
        *,
        smooth_scale: bool = True,
        injective: bool = True,
        freeze_edges: bool = True,
        deformation_limit: float = 1.0,
        noise_factor: float = 1.0,
    ) -> None:
        # Rejects a group that resolves no alignment. Each image's deformation is the average of its deformations to
        # the other images, so an image that pairs with nothing has nothing to align to.
        if len(images) < _MINIMUM_GROUP_SIZE:
            message = (
                "Unable to initialize the diffeomorphic demons registration. Groupwise registration aligns the "
                f"images of a group to their common mean space, so it requires at least {_MINIMUM_GROUP_SIZE} "
                f"images, but got {len(images)}."
            )
            console.error(message=message, error=ValueError)

        # Ensures that the input images use the fp32 precision, consistent with the rest of the cindra codebase.
        self._images: list[NDArray[np.float32]] = [
            image if image.dtype == np.float32 else image.astype(np.float32) for image in images
        ]

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

    def __repr__(self) -> str:
        """Returns a string representation of the DiffeomorphicDemonsRegistration instance."""
        return (
            f"DiffeomorphicDemonsRegistration(image_count={len(self._images)}, "
            f"speed_factor={self._speed_factor}, scale_sampling={self._scale_sampling}, "
            f"final_scale={self._final_scale})"
        )

    def get_deformation(self, image_index: int) -> Deformation:
        """Returns the deformation for the specified image.

        The deformation maps the image at the given index to the mean shape (groupwise registration result).

        Args:
            image_index: The index of the image (0-based).

        Returns:
            The deformation that aligns the specified image to the common mean space.

        Raises:
            RuntimeError: If the requested image carries no deformation resolved by a registration run.
        """
        deformation = self._deformations.get(image_index, None)
        if deformation is None:
            message = (
                f"Unable to retrieve the deformation for image {image_index}. The requested index must identify an "
                f"image register() resolved a deformation for, and the resolved indices are "
                f"{sorted(self._deformations)}."
            )
            console.error(message=message, error=RuntimeError)
        return deformation

    def register(self, *, progress: bool = True) -> None:
        """Performs the multiscale registration process.

        Iteratively computes deformations from coarse to fine scales, updating the groupwise alignment at each step.

        Args:
            progress: Determines whether to display a progress bar to report the registration progress.

        Raises:
            RuntimeError: If the finest scale level's knot grid holds too few knots to freeze its edges, which leaves
                every level of the run unable to contribute a deformation.
        """
        # Starts with bilinear (order 1) interpolation for the coarse iterations.
        self._interpolation_order = 1

        # The iteration factor controls smooth scale transitions between levels.
        iteration_factor = 0.5 ** (1.0 / self._scale_sampling)

        # Creates scale-space pyramids for each image.
        self._pyramids = [ScaleSpacePyramid(data=image, minimum_scale=self._final_scale) for image in self._images]

        # Rejects a run whose finest level cannot freeze the edges of its knot grid. The grid sampling grows with the
        # scale while the working resolution shrinks, so the finest level builds the largest grid of the run and a run
        # that skips it skips every level, leaving every image without a deformation.
        if self._freeze_edges:
            field_shape, grid_sampling, grid_shape = self._resolve_level_grid(scale=self._final_scale)
            if min(grid_shape) < MINIMUM_KNOTS_FOR_FROZEN_EDGES:
                message = (
                    f"Unable to register the {self._images[0].shape} images to their common mean space. Freezing the "
                    f"knot grid edges requires at least {MINIMUM_KNOTS_FOR_FROZEN_EDGES} knots along each dimension, "
                    f"but the finest scale level samples its {field_shape} working resolution every {grid_sampling} "
                    f"pixels, which builds a {grid_shape} grid. Register larger images or lower the "
                    "'diffeomorphic_registration.final_grid_sampling' configuration parameter."
                )
                console.error(message=message, error=RuntimeError)

        # Computes maximum scale from image dimensions (quarter of largest dimension).
        maximum_scale = max(self._images[0].shape) * 0.25

        # Computes the number of scale levels needed to span from final_scale to maximum_scale.
        scale_level_count = 1
        while self._final_scale * 2 ** (scale_level_count - 1) < maximum_scale:
            scale_level_count += 1

        # Computes total iterations for the progress bar. When smooth_scale is enabled, the coarsest level is skipped.
        if self._smooth_scale:
            total_iterations = (scale_level_count - 1) * self._scale_sampling
        else:
            total_iterations = scale_level_count * self._scale_sampling

        # Saves and restores the console's progress state to honor the progress parameter without affecting the global
        # state set by the pipeline entry point.
        previous_state = console.progress_enabled
        if progress:
            console.enable_progress()  # pragma: no cover, only when the caller sets progress=True
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
            if previous_state:  # pragma: no cover, restores the caller's progress state
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

        # Computes the incremental deformation of every image in one pass over the image pairs.
        incremental_deformations = self._compute_groupwise_deformations(iteration_key=iteration_key)

        # Applies incremental deformations to the running totals.
        for image_index in range(len(self._images)):
            self._apply_incremental_deformation(
                image_index=image_index, incremental_deformation=incremental_deformations[image_index]
            )

    def _compute_groupwise_deformations(self, iteration_key: tuple[int, int, float]) -> list[Deformation | None]:
        """Computes the deformation of every image by averaging its pairwise deformations to all other images.

        Notes:
            The Demons formulation makes the deformation between two images antisymmetric, so one computation per
            unordered pair supplies both of its images. Visiting each pair once therefore holds two accumulator fields
            per image rather than one field per ordered pair, which keeps the working set linear in the group size
            instead of quadratic.

        Args:
            iteration_key: The (level, iteration, scale) identifier for the current iteration.

        Returns:
            A list holding each image's averaged deformation, or a list of None values when the level's knot grid
            holds too few knots along either dimension to freeze its edges.
        """
        scale = iteration_key[2]
        image_count = len(self._images)

        # Returns None for every image when the knot grid the regularization would build holds too few knots to
        # freeze its edges, so the level contributes nothing instead of an unregularized deformation. The grid is
        # measured at the working resolution the whole group shares at this scale, which is the grid
        # _regularize_deformation builds from every pair's force field.
        if self._freeze_edges:
            grid_shape = self._resolve_level_grid(scale=scale)[2]
            if any(knot_count < MINIMUM_KNOTS_FOR_FROZEN_EDGES for knot_count in grid_shape):
                return [None] * image_count

        # Accumulates into raw field arrays rather than into Deformation instances, so each contribution after the
        # first writes through an existing buffer instead of allocating a fresh pair of fields. An image holds None
        # until its first contribution arrives, which reproduces the copy the identity branch of Deformation.add
        # performs and keeps the sign of a zero-valued field element intact.
        accumulated_y: list[NDArray[np.float32] | None] = [None] * image_count
        accumulated_x: list[NDArray[np.float32] | None] = [None] * image_count

        # Adds each pair's deformation to the first image's total and its negation to the second image's total. Pairs
        # are visited in ascending order, so every image accumulates its contributions in ascending partner order.
        for first_index in range(image_count):
            for second_index in range(first_index + 1, image_count):
                pairwise_deformation = self._compute_pairwise_deformation(
                    source_index=first_index, target_index=second_index, iteration_key=iteration_key
                )
                pair_y = pairwise_deformation.get_field(dimension=0)
                pair_x = pairwise_deformation.get_field(dimension=1)

                first_y, first_x = accumulated_y[first_index], accumulated_x[first_index]
                if first_y is None or first_x is None:
                    accumulated_y[first_index], accumulated_x[first_index] = pair_y.copy(), pair_x.copy()
                else:
                    np.add(first_y, pair_y, out=first_y)
                    np.add(first_x, pair_x, out=first_x)

                second_y, second_x = accumulated_y[second_index], accumulated_x[second_index]
                if second_y is None or second_x is None:
                    accumulated_y[second_index], accumulated_x[second_index] = (
                        np.negative(pair_y),
                        np.negative(pair_x),
                    )
                else:
                    np.subtract(second_y, pair_y, out=second_y)
                    np.subtract(second_x, pair_x, out=second_x)

        # Averages the accumulated deformations. Every image pairs with each of the others exactly once.
        pair_count = image_count - 1
        average_factor = np.float32(1.0 / pair_count) if pair_count > 1 else None

        deformations: list[Deformation | None] = []
        for field_y, field_x in zip(accumulated_y, accumulated_x, strict=True):
            if field_y is None or field_x is None:  # pragma: no cover, defensive guard, every image joins a pair
                message = (
                    "Unable to average the pairwise deformations of the group's images. Every image of the group "
                    "pairs with each of the others exactly once, so each one holds at least one contribution here."
                )
                console.error(message=message, error=RuntimeError)
            if average_factor is not None:  # pragma: no cover, only reached with more than two images
                np.multiply(field_y, average_factor, out=field_y)
                np.multiply(field_x, average_factor, out=field_x)
            deformations.append(Deformation(field_y=field_y, field_x=field_x))

        return deformations

    def _compute_pairwise_deformation(
        self, source_index: int, target_index: int, iteration_key: tuple[int, int, float]
    ) -> Deformation:
        """Computes the Demons deformation from source image to target image.

        Uses symmetric Demons forces computed from both image gradients.

        Args:
            source_index: Index of the source image.
            target_index: Index of the target image.
            iteration_key: The (level, iteration, scale) identifier for the current iteration.

        Returns:
            The regularized deformation from source to target.
        """
        scale = iteration_key[2]

        # Gets images and their gradients at the current scale.
        source_image, source_gradient = self._get_image_and_gradient(
            image_index=source_index, iteration_key=iteration_key
        )
        target_image, target_gradient = self._get_image_and_gradient(
            image_index=target_index, iteration_key=iteration_key
        )

        # Computes the symmetric Demons force field in one fused pass over the two images and their gradients. The
        # kernel writes directly into the two output fields, where the equivalent chain of NumPy operators allocates a
        # full-field temporary per operator and traverses the field once per operator.
        field_y = np.empty(source_image.shape, dtype=np.float32)
        field_x = np.empty(source_image.shape, dtype=np.float32)
        _compute_demons_force(
            source_image=source_image,
            source_gradient_y=source_gradient[0],
            source_gradient_x=source_gradient[1],
            target_image=target_image,
            target_gradient_y=target_gradient[0],
            target_gradient_x=target_gradient[1],
            noise_squared=np.float32(self._noise_factor**2),
            speed=np.float32(-self._speed_factor),
            field_y=field_y,
            field_x=field_x,
        )

        # Regularizes using B-spline grid to ensure diffeomorphism.
        force_deformation = Deformation(field_y=field_y, field_x=field_x)
        return self._regularize_deformation(scale=scale, deformation=force_deformation)

    def _get_image_and_gradient(
        self, image_index: int, iteration_key: tuple[int, int, float]
    ) -> tuple[NDArray[np.float32], tuple[NDArray[np.float32], NDArray[np.float32]]]:
        """Returns the image at the current scale along with its gradient.

        Notes:
            The gradient is cached alongside the deformed image it is derived from. The groupwise loop pairs every
            image with every other image, so it requests each image once per other group member, and recomputing the
            two convolutions on every request would scale with the square of the group size.

        Args:
            image_index: Index of the image to retrieve.
            iteration_key: The (level, iteration, scale) identifier for the current iteration.

        Returns:
            The image resampled to the current scale with the accumulated deformation applied, together with its
            vertical and horizontal central-difference gradients.
        """
        scale = iteration_key[2]

        # Tries to retrieve the cached image together with the gradient computed from it. The gradient is stored as a
        # single stacked array, so that it travels through the same cache as every other cached array.
        cached_image = self._get_cached(key=f"image_{image_index}", iteration_key=iteration_key)
        cached_gradient = self._get_cached(key=f"gradient_{image_index}", iteration_key=iteration_key)
        if isinstance(cached_image, np.ndarray) and isinstance(cached_gradient, np.ndarray):
            return cached_image, (cached_gradient[0], cached_gradient[1])

        image = self._get_deformed_image(image_index=image_index, scale=scale)

        # Computes gradient using central differences.
        gradient_kernel = np.array([0.5, 0, -0.5], dtype=np.float32)
        gradient_y = scipy.ndimage.convolve1d(input=image, weights=gradient_kernel, axis=0, mode="nearest")
        gradient_x = scipy.ndimage.convolve1d(input=image, weights=gradient_kernel, axis=1, mode="nearest")

        self._set_cached(key=f"image_{image_index}", iteration_key=iteration_key, data=image)
        self._set_cached(
            key=f"gradient_{image_index}", iteration_key=iteration_key, data=np.stack((gradient_y, gradient_x))
        )

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
        if self._pyramids is None:  # pragma: no cover, defensive guard because register() always initializes pyramids
            message = "Unable to retrieve image. The pyramids have not been initialized, call register() first."
            console.error(message=message, error=RuntimeError)

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
        # A level whose knot grid cannot freeze its edges supplies None for every image, which leaves the running
        # total untouched.
        if incremental_deformation is None:
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
        self._deformations[image_index] = current_deformation.compose(other=incremental_deformation)

    def _regularize_deformation(self, scale: float, deformation: Deformation) -> Deformation:
        """Regularizes a deformation to ensure diffeomorphism using B-spline constraints.

        Args:
            scale: The current scale level.
            deformation: The raw deformation to regularize.

        Returns:
            The regularized deformation.
        """
        # The injectivity constraint uses the original-pixel-unit grid sampling, since both scale and grid sampling
        # must be in the same coordinate system.
        original_grid_sampling = self._compute_grid_sampling(scale=scale)

        # Scales the sampling by the working resolution the deformation itself carries, which is the resolution the
        # level filter measures, so the grid the filter accepts is the grid this method builds.
        grid_sampling = self._scale_grid_sampling(
            grid_sampling=original_grid_sampling, field_height=deformation.field_shape[0]
        )

        # Computes injectivity constraint factor based on scale and grid sampling in original-pixel units.
        injective_factor = 0.9
        if self._injective:
            injective_factor = min(self._deformation_limit * scale / original_grid_sampling, 0.9)

        return deformation.regularize(
            grid_sampling=grid_sampling,
            injective=self._injective,
            injective_factor=injective_factor,
            freeze_edges=self._freeze_edges,
        )

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

    def _scale_grid_sampling(self, grid_sampling: float, field_height: int) -> float:
        """Converts a grid sampling expressed in original-image pixels to the working resolution of a field.

        Notes:
            Both the level pre-filter and the regularization itself measure the knot grid through this method, so
            the grid the pre-filter rejects is the grid the regularization would have built.

        Args:
            grid_sampling: The grid sampling in original-image pixels.
            field_height: The height of the deformation field at the current working resolution.

        Returns:
            The grid sampling in working-resolution pixels.
        """
        downsample_ratio = field_height / self._images[0].shape[0]
        if downsample_ratio < 1.0:
            return grid_sampling * downsample_ratio
        return grid_sampling

    def _resolve_field_shape(self, scale: float) -> tuple[int, int]:
        """Returns the working resolution the group's images share at the given scale.

        Args:
            scale: The current scale value.

        Returns:
            The shape every image of the group holds at the requested scale, as (height, width).
        """
        if self._pyramids is None:  # pragma: no cover, defensive guard because register() always initializes pyramids
            message = (
                "Unable to resolve the field shape. The pyramids have not been initialized, call register() first."
            )
            console.error(message=message, error=RuntimeError)

        # Every image of the group shares one shape, and the pyramids derive their level shapes from that shape alone,
        # so the first pyramid reports the resolution every image works at.
        return self._pyramids[0].get_scale_shape(scale=scale)

    def _resolve_level_grid(self, scale: float) -> tuple[tuple[int, int], float, tuple[int, int]]:
        """Returns the knot grid the regularization builds at the given scale, together with what it is measured from.

        Args:
            scale: The current scale value.

        Returns:
            A tuple storing the working resolution as (height, width), the grid sampling in working-resolution pixels,
            and the shape of the knot grid that sampling produces.
        """
        field_height, field_width = self._resolve_field_shape(scale=scale)
        grid_sampling = self._scale_grid_sampling(
            grid_sampling=self._compute_grid_sampling(scale=scale), field_height=field_height
        )
        grid_shape = SplineGrid.compute_grid_shape(
            field_height=field_height, field_width=field_width, grid_sampling=grid_sampling
        )
        return (field_height, field_width), grid_sampling, grid_shape

    def _get_cached(self, key: str, iteration_key: tuple[int, int, float]) -> Deformation | NDArray[np.float32] | None:
        """Retrieves cached data if the iteration key matches.

        Args:
            key: The cache key.
            iteration_key: The iteration identifier to validate against.

        Returns:
            The cached data if valid, otherwise None.
        """
        entry = self._cache.get(key)
        if entry is not None and entry[0] == iteration_key:
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


@numba.njit(parallel=True, cache=True)
def _compute_demons_force(  # pragma: no cover
    source_image: NDArray[np.float32],
    source_gradient_y: NDArray[np.float32],
    source_gradient_x: NDArray[np.float32],
    target_image: NDArray[np.float32],
    target_gradient_y: NDArray[np.float32],
    target_gradient_x: NDArray[np.float32],
    noise_squared: np.float32,
    speed: np.float32,
    field_y: NDArray[np.float32],
    field_x: NDArray[np.float32],
) -> None:
    """Computes the symmetric Demons force field between one image pair.

    Notes:
        The arithmetic is ordered to match the equivalent chain of NumPy operators term for term, which keeps the
        result bit-identical to it. Reassociating any term, such as folding the noise factor into the squared
        intensity difference as a single product, changes the low bits of every force value.

        A denominator of zero is replaced by infinity, so the pixel contributes nothing to the force rather than
        producing a non-finite value.

    Args:
        source_image: The source image resampled to the current scale.
        source_gradient_y: The vertical central-difference gradient of the source image.
        source_gradient_x: The horizontal central-difference gradient of the source image.
        target_image: The target image resampled to the current scale.
        target_gradient_y: The vertical central-difference gradient of the target image.
        target_gradient_x: The horizontal central-difference gradient of the target image.
        noise_squared: The square of the intensity-noise regularization factor.
        speed: The negated deformation speed factor, which carries the sign for backward mapping.
        field_y: The pre-allocated output array receiving the vertical force component.
        field_x: The pre-allocated output array receiving the horizontal force component.
    """
    height, width = source_image.shape
    for y in numba.prange(height):
        for x in range(width):
            source_gradient_magnitude_squared = (
                source_gradient_y[y, x] * source_gradient_y[y, x] + source_gradient_x[y, x] * source_gradient_x[y, x]
            )
            target_gradient_magnitude_squared = (
                target_gradient_y[y, x] * target_gradient_y[y, x] + target_gradient_x[y, x] * target_gradient_x[y, x]
            )

            intensity_difference = source_image[y, x] - target_image[y, x]
            regularization = noise_squared * (intensity_difference * intensity_difference)

            source_denominator = source_gradient_magnitude_squared + regularization
            target_denominator = target_gradient_magnitude_squared + regularization
            if source_denominator == np.float32(0.0):
                source_denominator = np.float32(np.inf)
            if target_denominator == np.float32(0.0):
                target_denominator = np.float32(np.inf)

            field_y[y, x] = (
                intensity_difference
                * (source_gradient_y[y, x] / source_denominator + target_gradient_y[y, x] / target_denominator)
                * speed
            )
            field_x[y, x] = (
                intensity_difference
                * (source_gradient_x[y, x] / source_denominator + target_gradient_x[y, x] / target_denominator)
                * speed
            )
