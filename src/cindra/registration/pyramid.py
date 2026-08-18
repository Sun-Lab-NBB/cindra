"""Provides the assets for computing and storing the multi-resolution scale-space image pyramids."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .deformation import zoom, diffuse

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

_DOWNSAMPLE_ZOOM_THRESHOLD: float = 0.9
"""The maximum zoom factor threshold below which downsampling is applied. A factor of 0.9 means downsampling occurs
when resolution would be reduced by more than 10%."""

_MINIMUM_DOWNSAMPLE_DIMENSION: int = 8
"""The value that the image's smallest dimension must exceed for a new pyramid level to be downsampled."""


class ScaleSpacePyramid:
    """Manages a scale-space pyramid for multi-resolution image access.

    Given an input 2D image, provides methods to obtain the image at any specified scale. Higher scales correspond
    to smoother images with smaller dimensions. The pyramid is built lazily, adding levels only as needed.

    Args:
        data: The input 2D image array for which to generate the scale space pyramid.
        minimum_scale: The minimum (finest) scale for the pyramid. The input image is smoothed to this scale before
            creating the base level. If the scale is large enough, the data is also downsampled for efficiency.

    Attributes:
        _levels: List of image arrays at each pyramid level, from finest to coarsest.
        _level_scales: List of scale values corresponding to each pyramid level.
        _level_downsample_factors: List of cumulative downsample factors for each level relative to the original
            image. A factor of 0.5 means the level is at half the original resolution.
    """

    _LEVEL_FACTOR: float = 2.0
    """The factor by which the scale doubles between successive pyramid levels. Each level is downsampled by the
    reciprocal of this factor (0.5)."""

    def __init__(self, data: NDArray[np.float32], minimum_scale: float) -> None:
        minimum_scale = float(minimum_scale)
        self._levels: list[NDArray[np.float32]] = []
        self._level_scales: list[float] = []
        self._level_downsample_factors: list[float] = []
        self._initialize_base_level(data=data, minimum_scale=minimum_scale)

    def __repr__(self) -> str:
        """Returns a string representation of the ScaleSpacePyramid instance."""
        return f"ScaleSpacePyramid(level_count={len(self._levels)}, level_scales={self._level_scales})"

    def get_scale(self, scale: float) -> NDArray[np.float32]:
        """Returns the image at the specified scale.

        Retrieves the pyramid level at or below the requested scale, then applies additional smoothing to reach the
        exact target scale. New pyramid levels are created on demand if needed.

        Args:
            scale: The target scale in world coordinates. Must be >= minimum_scale.

        Returns:
            The image smoothed to the requested scale.
        """
        level = self._resolve_level(scale=scale)
        data = self._levels[level]
        current_scale = self._level_scales[level]

        # Applies additional smoothing to reach the exact target scale. Scales the sigma by the level's downsample
        # factor to convert from original-pixel units to downsampled-pixel units.
        if scale > current_scale:
            additional_sigma = (scale**2 - current_scale**2) ** 0.5
            adjusted_sigma = additional_sigma * self._level_downsample_factors[level]
            data = diffuse(data=data, sigma=adjusted_sigma)

        return data

    def get_scale_shape(self, scale: float) -> tuple[int, int]:
        """Returns the shape of the image the pyramid holds at the specified scale.

        Notes:
            The additional smoothing get_scale applies to reach the exact target scale preserves the shape of the
            pyramid level it starts from, so the shape is resolved without paying for that smoothing.

        Args:
            scale: The target scale in world coordinates. Must be >= minimum_scale.

        Returns:
            The shape of the image at the requested scale, as (height, width).
        """
        data = self._levels[self._resolve_level(scale=scale)]
        return data.shape[0], data.shape[1]

    def _initialize_base_level(self, data: NDArray[np.float32], minimum_scale: float) -> None:
        """Initializes the base pyramid level by smoothing and optionally downsampling the image data.

        Args:
            data: The input image array.
            minimum_scale: The target scale for the base level.
        """
        downsample_factor = 1.0

        if minimum_scale > 0:
            data = diffuse(data=data, sigma=minimum_scale)

            # Downsamples if scale is large enough (reduces resolution by more than 10%).
            zoom_factor = 1.0 / minimum_scale
            if zoom_factor < _DOWNSAMPLE_ZOOM_THRESHOLD:
                data = zoom(data=data, factor=zoom_factor, order=3)
                downsample_factor = zoom_factor

        self._levels.append(data)
        self._level_scales.append(minimum_scale)
        self._level_downsample_factors.append(downsample_factor)

    def _resolve_level(self, scale: float) -> int:
        """Returns the index of the coarsest pyramid level whose scale does not exceed the requested scale.

        Levels are built lazily, so this creates every level between the pyramid's current coarsest level and the
        requested scale.

        Args:
            scale: The target scale in world coordinates.

        Returns:
            The index of the level the requested scale resolves to.
        """
        level = 0
        while level < len(self._levels) - 1 and self._level_scales[level + 1] <= scale:
            level += 1

        while self._level_scales[level] < scale and level == len(self._levels) - 1:
            self._add_level()
            if self._level_scales[-1] <= scale:
                level = len(self._levels) - 1

        return level

    def _add_level(self) -> None:
        """Adds a new coarser level to the pyramid by smoothing and downsampling the underlying image's data."""
        data = self._levels[-1]
        current_scale = self._level_scales[-1]
        current_factor = self._level_downsample_factors[-1]

        target_scale = max(self._LEVEL_FACTOR, current_scale * self._LEVEL_FACTOR)

        # Computes additional smoothing needed. Scales sigma by the current downsample factor to convert from
        # original-pixel units to the current level's pixel units.
        additional_sigma = (target_scale**2 - current_scale**2) ** 0.5
        adjusted_sigma = additional_sigma * current_factor
        data = diffuse(data=data, sigma=adjusted_sigma)

        new_factor = current_factor
        if min(data.shape) > _MINIMUM_DOWNSAMPLE_DIMENSION:
            factor = 1.0 / self._LEVEL_FACTOR
            data = zoom(data=data, factor=factor, order=3)
            new_factor = current_factor * factor

        self._levels.append(data)
        self._level_scales.append(target_scale)
        self._level_downsample_factors.append(new_factor)
