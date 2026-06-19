"""Provides the assets for computing and storing the multi-resolution scale-space image pyramids."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .deformation import zoom, diffuse

_DOWNSAMPLE_ZOOM_THRESHOLD: float = 0.9
"""The maximum zoom factor threshold below which downsampling is applied. A factor of 0.9 means downsampling occurs
when resolution would be reduced by more than 10%."""

_MINIMUM_DOWNSAMPLE_DIMENSION: int = 8
"""The minimum image dimension (in either axis) required before pyramid downsampling is applied."""

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class ScaleSpacePyramid:
    """Manages a scale-space pyramid for multi-resolution image access.

    Given an input 2D image, provides methods to obtain the image at any specified scale. Higher scales correspond
    to smoother images with smaller dimensions. The pyramid is built lazily, adding levels only as needed.

    Args:
        data: The input 2D image array for which to generate the scale space pyramid.
        min_scale: The minimum (finest) scale for the pyramid. The input image is smoothed to this scale before
            creating the base level. If the scale is large enough, the data is also downsampled for efficiency.

    Attributes:
        _levels: List of image arrays at each pyramid level, from finest to coarsest.
        _level_scales: List of scale values corresponding to each pyramid level.
        _level_downsample_factors: List of cumulative downsample factors for each level relative to the original
            image. A factor of 0.5 means the level is at half the original resolution.
    """

    _LEVEL_FACTOR: float = 2.0
    """The factor by which the scale doubles between successive pyramid levels; each level is downsampled by its
    reciprocal (0.5)."""

    def __init__(self, data: NDArray[np.float32], min_scale: float) -> None:
        min_scale = float(min_scale)
        self._levels: list[NDArray[np.float32]] = []
        self._level_scales: list[float] = []
        self._level_downsample_factors: list[float] = []
        self._initialize_base_level(data=data, min_scale=min_scale)

    def _initialize_base_level(self, data: NDArray[np.float32], min_scale: float) -> None:
        """Initializes the base pyramid level by smoothing and optionally downsampling the image data.

        Args:
            data: The input image array.
            min_scale: The target scale for the base level.
        """
        downsample_factor = 1.0

        # Smooths to target scale if min_scale > 0.
        if min_scale > 0:
            data = diffuse(data=data, sigma=min_scale)

            # Downsamples if scale is large enough (reduces resolution by more than 10%).
            zoom_factor = 1.0 / min_scale
            if zoom_factor < _DOWNSAMPLE_ZOOM_THRESHOLD:
                data = zoom(data=data, factor=zoom_factor, order=3)
                downsample_factor = zoom_factor

        self._levels.append(data)
        self._level_scales.append(min_scale)
        self._level_downsample_factors.append(downsample_factor)

    def get_scale(self, scale: float) -> NDArray[np.float32]:
        """Returns the image at the specified scale.

        Retrieves the pyramid level at or below the requested scale, then applies additional smoothing to reach the
        exact target scale. New pyramid levels are created on demand if needed.

        Args:
            scale: The target scale in world coordinates. Must be >= min_scale.

        Returns:
            The image smoothed to the requested scale.
        """
        # Finds the appropriate pyramid level.
        level = 0
        while level < len(self._levels) - 1 and self._level_scales[level + 1] <= scale:
            level += 1

        # Adds new levels if the current highest level is still below the target scale.
        while self._level_scales[level] < scale and level == len(self._levels) - 1:
            self._add_level()
            if self._level_scales[-1] <= scale:
                level = len(self._levels) - 1

        # Gets the base data from the selected level.
        data = self._levels[level]
        current_scale = self._level_scales[level]

        # Applies additional smoothing to reach the exact target scale. Scales the sigma by the level's downsample
        # factor to convert from original-pixel units to downsampled-pixel units.
        if scale > current_scale:
            additional_sigma = (scale**2 - current_scale**2) ** 0.5
            adjusted_sigma = additional_sigma * self._level_downsample_factors[level]
            data = diffuse(data=data, sigma=adjusted_sigma)

        return data

    def _add_level(self) -> None:
        """Adds a new coarser level to the pyramid by smoothing and downsampling the underlying image's data."""
        data = self._levels[-1]
        current_scale = self._level_scales[-1]
        current_factor = self._level_downsample_factors[-1]

        # Computes the target scale for the new level.
        target_scale = max(self._LEVEL_FACTOR, current_scale * 2.0)

        # Computes additional smoothing needed. Scales sigma by the current downsample factor to convert from
        # original-pixel units to the current level's pixel units.
        additional_sigma = (target_scale**2 - current_scale**2) ** 0.5
        adjusted_sigma = additional_sigma * current_factor
        data = diffuse(data=data, sigma=adjusted_sigma)

        # Downsamples if the image is large enough.
        new_factor = current_factor
        if min(data.shape) > _MINIMUM_DOWNSAMPLE_DIMENSION:
            factor = 1.0 / self._LEVEL_FACTOR
            data = zoom(data=data, factor=factor, order=3)
            new_factor = current_factor * factor

        self._levels.append(data)
        self._level_scales.append(target_scale)
        self._level_downsample_factors.append(new_factor)
