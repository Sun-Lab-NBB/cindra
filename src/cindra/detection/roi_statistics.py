"""Provides assets for computing ROI statistics after the initial ROI detection."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar
from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from ataraxis_base_utilities import console

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..dataclasses import ROIMask, ROIStatistics


@dataclass(frozen=True, slots=True)
class _EllipseData:
    """Defines an ellipse fitted to the ROI's pixels via weighted covariance analysis."""

    radii: tuple[float, float]
    """The semi-major and semi-minor axis lengths, ordered from largest to smallest."""

    y_scale: int
    """The y-axis scaling factor that corrects for non-square pixel aspect ratios during fitting."""

    x_scale: int
    """The x-axis scaling factor that corrects for non-square pixel aspect ratios during fitting."""

    @property
    def radius(self) -> float:
        """Returns the effective radius of the ROI ellipse scaled by the mean of y_scale and x_scale."""
        return float(self.radii[0] * np.mean((self.x_scale, self.y_scale)))

    @property
    def aspect_ratio(self) -> float:
        """Returns the normalized aspect ratio bounded between 0 and 2, where 1 indicates a circular shape."""
        major, minor = self.radii
        return 2 * major / (major + minor + 0.01)


def estimate_diameter_from_rois(rois: list[ROIMask], default_diameter: int = 10) -> int:
    """Estimates the ROI diameter from the pixel counts of a list of ROIs.

    Args:
        rois: The list of ROIMask instances to analyze.
        default_diameter: The fallback diameter to return if the ROI list is empty or if the median ROI pixel count
            is zero.

    Returns:
        The estimated ROI diameter in pixels, computed as the diameter of a circle with area equal to the median
        ROI pixel count, truncated toward zero and floored at 1 pixel.
    """
    if not rois:
        return default_diameter

    # Collects pixel counts from all ROIs. Uses the y_pixels array length as the authoritative pixel count.
    pixel_counts = np.array([len(roi.y_pixels) for roi in rois], dtype=np.float32)

    if not pixel_counts.size or np.median(pixel_counts) == 0:
        return default_diameter

    # Computes the diameter of a circle with area equal to the median pixel count: area = π * r², so
    # r = sqrt(area / π) and diameter = 2 * r = 2 * sqrt(median_pixels / π).
    median_pixels = np.median(pixel_counts)
    estimated_diameter = int(2 * np.sqrt(median_pixels / np.pi))

    return max(estimated_diameter, 1)


def compute_roi_statistics(
    rois: list[ROIStatistics],
    frame_height: int,
    frame_width: int,
    aspect: float | None = None,
    diameter: int | None = None,
    maximum_overlap_fraction: float | None = None,
    *,
    crop: bool = True,
    lightweight: bool = False,
) -> None:
    """Computes shape statistics for a list of ROIStatistics instances in-place.

    Notes:
        Computes statistics (compactness, solidity, radius, aspect ratio, etc.) for each input ROI and writes the
        computed values back to the ROIStatistics instances. If maximum_overlap_fraction is specified, ROIs exceeding
        the overlap threshold are removed from the list in-place. When lightweight is True, only the minimal statistics
        required for preclassification (compactness, pixel_count, soma_mask, and normalized_pixel_count) are computed,
        skipping the expensive ellipse fitting, convex hull solidity, and overlap computations.

    Args:
        rois: The list of ROIStatistics instances that define the ROIs to process. Modified in-place.
        frame_height: The height of the recording frames from which ROIs are segmented, in pixels.
        frame_width: The width of the recording frames from which ROIs are segmented, in pixels.
        aspect: The aspect ratio of the recording. If provided, adjusts ROI ellipse fitting. Ignored in lightweight
            mode.
        diameter: The expected ROI diameter in pixels. Used for ROI ellipse fitting normalization and for distance
            normalization in compactness. Applies in both full and lightweight modes.
        maximum_overlap_fraction: The maximum fraction of pixels that can overlap with other ROIs. If specified, ROIs
            exceeding this threshold are removed from the list in-place. Ignored in lightweight mode.
        crop: Determines whether to crop processed ROIs to the soma region before computing statistics.
        lightweight: Determines whether to compute only the minimal statistics needed for preclassification. When True,
            skips ellipse fitting, solidity, and overlap computations. The aspect and maximum_overlap_fraction
            parameters are ignored. The diameter is still used for distance normalization in compactness.

    Raises:
        ValueError: If the input rois list is empty.
    """
    if not rois:
        message = "Unable to compute ROI statistics. The input rois list is empty."
        console.error(message=message, error=ValueError)

    # Initializes centroids for ROIs that lack them. The centroid is required for computing radial statistics.
    for roi in rois:
        if not roi.mask.centroid or roi.mask.centroid == (0, 0):
            roi.mask.centroid = _compute_median_pixel_position(y_pixels=roi.mask.y_pixels, x_pixels=roi.mask.x_pixels)

    # Resolves the ROI diameter for distance normalization. A sensible default is used when no diameter is provided.
    default_diameter = 10
    effective_diameter = default_diameter if diameter is None or diameter == 0 else diameter

    # Wraps each ROIStatistics in an _ROI processing object to compute derived statistics.
    roi_wrappers = [_ROI(data=roi, diameter=effective_diameter, crop=crop) for roi in rois]

    # Resolves aspect correction and overlap image only when full statistics are needed. Lightweight mode skips these
    # because ellipse fitting and overlap filtering are not performed.
    if not lightweight:
        if aspect is not None:
            y_scale, x_scale = int(aspect * effective_diameter), effective_diameter
        else:
            y_scale, x_scale = effective_diameter, effective_diameter

        overlap_counts = _ROI.get_overlap_count_image(rois=roi_wrappers, height=frame_height, width=frame_width)

    # Pre-allocates an array to collect soma pixel counts for normalization during the computation loop.
    roi_count = len(rois)
    soma_pixel_count_values = np.empty(roi_count, dtype=np.float32)

    # Computes shape statistics for each ROI and writes them back to the ROIStatistics instances. In lightweight mode,
    # skips the expensive ellipse fitting, convex hull solidity, and overlap mask computations.
    for roi_index, wrapper in enumerate(roi_wrappers):
        data = wrapper.data
        data.compactness = wrapper.compactness
        data.pixel_count = wrapper.pixel_count
        data.soma_mask = wrapper.soma_mask

        if not lightweight:
            data.solidity = wrapper.solidity
            data.mask.overlap_mask = wrapper.get_overlap_mask(overlap_count_image=overlap_counts)

            ellipse = wrapper.fit_ellipse(y_scale=y_scale, x_scale=x_scale)
            data.mask.radius = ellipse.radius
            data.aspect_ratio = ellipse.aspect_ratio

        # Collects soma pixel counts for normalization to avoid re-iterating over ROIs.
        soma_pixel_count_values[roi_index] = wrapper.soma_pixel_count

    # Normalizes soma pixel count relative to the first 100 ROIs. Detection algorithms typically find high-confidence
    # ROIs first, so early ROIs serve as a reliable baseline for comparing later, lower-confidence detections.
    normalization_count = 100
    soma_pixel_count_normalized = soma_pixel_count_values / np.mean(soma_pixel_count_values[:normalization_count])

    for roi, soma_count_normalized in zip(rois, soma_pixel_count_normalized, strict=True):
        roi.normalized_pixel_count = float(soma_count_normalized)

    # Removes ROIs with excessive overlap. High overlap often indicates over-segmentation or neuropil contamination.
    # Skipped in lightweight mode since overlap computation is not performed.
    if not lightweight and maximum_overlap_fraction is not None and maximum_overlap_fraction < 1.0:
        keep_flags = _ROI.remove_overlapping_rois(
            rois=roi_wrappers, overlap_image=overlap_counts, maximum_overlap_fraction=maximum_overlap_fraction
        )

        rois[:] = [roi for roi, keep in zip(rois, keep_flags, strict=True) if keep]

        # Recomputes overlap masks after removing ROIs, since remaining ROIs may no longer overlap.
        roi_wrappers = [_ROI(data=roi, diameter=effective_diameter, crop=crop) for roi in rois]
        overlap_counts = _ROI.get_overlap_count_image(rois=roi_wrappers, height=frame_height, width=frame_width)
        for wrapper in roi_wrappers:
            wrapper.data.mask.overlap_mask = wrapper.get_overlap_mask(overlap_count_image=overlap_counts)


def _compute_median_pixel_position(y_pixels: NDArray[np.int32], x_pixels: NDArray[np.int32]) -> tuple[int, int]:
    """Computes the ROI centroid as the y and x coordinates of the pixel closest to the coordinate-wise median.

    Args:
        y_pixels: The y-coordinates of the ROI's pixels.
        x_pixels: The x-coordinates of the ROI's pixels.

    Returns:
        The (y, x) coordinates of the pixel closest to the median position.
    """
    y_median = np.median(y_pixels)
    x_median = np.median(x_pixels)
    minimum_index = np.argmin(np.square(x_pixels - x_median) + np.square(y_pixels - y_median))
    return int(y_pixels[minimum_index]), int(x_pixels[minimum_index])


def _compute_distance_kernel(radius: int) -> NDArray[np.float32]:
    """Computes a 2D array of Euclidean distances from the center point.

    Args:
        radius: The radius of the kernel in pixels.

    Returns:
        An array of shape (2*radius+1, 2*radius+1) containing Euclidean distances from the center pixel.
    """
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    y_grid, x_grid = np.meshgrid(offsets, offsets)
    return np.hypot(y_grid, x_grid)


class _ROI:
    """Wraps the ROIStatistics dataclass with methods to compute additional ROI properties.

    Notes:
        The class uses a shared class variable caching sorted baseline distances (keyed by diameter) to avoid
        recomputation across instances. The soma mask is cached after first computation to avoid redundant calculations
        when accessing dependent properties. Distance-based statistics (mean_radius, compactness) are normalized by the
        ROI diameter to make them scale-invariant across different ROI sizes and imaging magnifications.

    Attributes:
        _data: The underlying ROIStatistics instance.
        _diameter: The ROI diameter used for distance normalization.
        _crop: Determines whether to crop to soma region when computing statistics.
        _cached_soma_mask: Cached soma mask array, computed on first access.
        _cached_soma_pixel_count: Cached number of soma pixels, computed on first access.
        _cached_soma_y_pixels: Cached soma-masked y-coordinates, computed on first access.
        _cached_soma_x_pixels: Cached soma-masked x-coordinates, computed on first access.
    """

    _baseline_cache: ClassVar[dict[int, NDArray[np.float32]]] = {}
    """Cache of sorted baseline distances keyed by diameter, avoiding recomputation across instances."""

    def __init__(self, data: ROIStatistics, diameter: int, *, crop: bool = True) -> None:
        """Initializes the _ROI wrapper for computing derived statistics from an ROIStatistics instance.

        Args:
            data: The ROIStatistics instance to wrap.
            diameter: The estimated ROI diameter in pixels, used to normalize distance-based statistics.
            crop: Determines whether to crop to soma region when computing statistics.

        Raises:
            TypeError: If the x_pixels, y_pixels, and pixel_weights arrays do not have the same shape.
        """
        if (
            data.mask.x_pixels.shape != data.mask.y_pixels.shape
            or data.mask.x_pixels.shape != data.mask.pixel_weights.shape
        ):
            message = (
                "Unable to initialize the ROI class. The x_pixels, y_pixels, and pixel_weights arrays in the input "
                "ROIStatistics instance must have the same shape."
            )
            console.error(message=message, error=TypeError)

        self._data: ROIStatistics = data
        self._diameter: int = diameter
        self._crop: bool = crop
        self._cached_soma_mask: NDArray[np.bool_] | None = None
        self._cached_soma_pixel_count: int | None = None
        self._cached_soma_y_pixels: NDArray[np.int32] | None = None
        self._cached_soma_x_pixels: NDArray[np.int32] | None = None

    @property
    def data(self) -> ROIStatistics:
        """Returns the underlying ROIStatistics instance."""
        return self._data

    @property
    def y_pixels(self) -> NDArray[np.int32]:
        """Returns the y-coordinates of the ROI pixels."""
        return self._data.mask.y_pixels

    @property
    def x_pixels(self) -> NDArray[np.int32]:
        """Returns the x-coordinates of the ROI pixels."""
        return self._data.mask.x_pixels

    @property
    def pixel_weights(self) -> NDArray[np.float32]:
        """Returns the pixel weights (lambda values) for the ROI."""
        return self._data.mask.pixel_weights

    @property
    def centroid(self) -> tuple[int, int]:
        """Returns the centroid (y, x) pixel position of the ROI."""
        return self._data.mask.centroid[0], self._data.mask.centroid[1]

    @property
    def soma_mask(self) -> NDArray[np.bool_]:
        """Returns the boolean mask indicating which pixels belong to the soma region of this ROI, computed and
        cached on first access.
        """
        if self._cached_soma_mask is not None:
            return self._cached_soma_mask

        self._cached_soma_mask = self._compute_soma_mask()
        return self._cached_soma_mask

    @property
    def soma_y_pixels(self) -> NDArray[np.int32]:
        """Returns the y-coordinates of the pixels inside the soma region, computed and cached on first access."""
        if self._cached_soma_y_pixels is None:
            self._cached_soma_y_pixels = self.y_pixels[self.soma_mask]
        return self._cached_soma_y_pixels

    @property
    def soma_x_pixels(self) -> NDArray[np.int32]:
        """Returns the x-coordinates of the pixels inside the soma region, computed and cached on first access."""
        if self._cached_soma_x_pixels is None:
            self._cached_soma_x_pixels = self.x_pixels[self.soma_mask]
        return self._cached_soma_x_pixels

    @property
    def mean_radius(self) -> float:
        """Returns the mean diameter-normalized distance from the ROI's soma pixels to their median center."""
        y_pixels = self.soma_y_pixels
        x_pixels = self.soma_x_pixels
        # Normalizes distances by ROI diameter for scale-invariance, matching the original suite2p approach.
        distances = np.hypot(
            (y_pixels - np.median(y_pixels)) / self._diameter,
            (x_pixels - np.median(x_pixels)) / self._diameter,
        )
        return float(np.mean(distances))

    @property
    def baseline_mean_radius(self) -> float:
        """Returns the expected mean radius for a uniformly distributed set of pixels of the same count as the ROI's
        soma region.
        """
        # Uses a diameter-dependent kernel. The kernel is computed from a meshgrid spanning 2*diameter in each
        # direction, with distances normalized by diameter, matching the original suite2p approach.
        diameter = self._diameter
        if diameter not in _ROI._baseline_cache:
            kernel = _compute_distance_kernel(radius=2 * diameter)
            # Normalizes the kernel distances by diameter to match the distance normalization in mean_radius.
            _ROI._baseline_cache[diameter] = np.sort((kernel / diameter).flatten())
        baseline = _ROI._baseline_cache[diameter]
        return float(np.mean(baseline[: self.soma_pixel_count]))

    @property
    def compactness(self) -> float:
        """Returns the ratio of actual to expected mean radius, floored at 1.0, where values near 1 indicate compact
        circular ROIs.
        """
        return max(1.0, self.mean_radius / (1e-10 + self.baseline_mean_radius))

    @property
    def solidity(self) -> float:
        """Returns the ROI solidity as the ratio of soma pixel count to convex hull area, substituting a fixed area
        of 10.0 for ROIs of 10 or fewer pixels and for degenerate hulls.
        """
        minimum_pixels_for_hull = 10
        default_area = 10.0

        pixel_count = self.soma_pixel_count
        if pixel_count <= minimum_pixels_for_hull:
            return pixel_count / default_area

        # ConvexHull requires (N, 2) array of points.
        points = np.column_stack((self.soma_y_pixels, self.soma_x_pixels))
        try:
            area = ConvexHull(points).volume
        except ValueError, QhullError:
            area = default_area

        return pixel_count / area

    @property
    def soma_pixel_count(self) -> int:
        """Returns the number of pixels in the soma region, computed and cached on first access."""
        if self._cached_soma_pixel_count is None:
            self._cached_soma_pixel_count = int(self.soma_mask.sum())
        return self._cached_soma_pixel_count

    @property
    def pixel_count(self) -> int:
        """Returns the total number of pixels in the ROI."""
        return self.x_pixels.size

    def fit_ellipse(self, y_scale: int, x_scale: int) -> _EllipseData:
        """Fits a 2D Gaussian ellipse to the ROI pixels via covariance eigendecomposition.

        Args:
            y_scale: The y-axis scaling factor for correcting non-square pixel aspect ratios.
            x_scale: The x-axis scaling factor for correcting non-square pixel aspect ratios.

        Returns:
            The semi-major and semi-minor radii of the fitted ellipse, together with the axis scaling factors applied
            during the fit.
        """
        y_pixels = self.soma_y_pixels
        x_pixels = self.soma_x_pixels
        pixel_weights = self.pixel_weights[self.soma_mask]

        # Filters zero-weight pixels and normalizes weights to form a probability distribution for weighted statistics.
        valid_mask = pixel_weights > 0
        weights = pixel_weights[valid_mask]
        weights = weights / weights.sum()

        # Scales coordinates to correct for non-square pixel aspect ratios before computing covariance.
        y_scaled = y_pixels[valid_mask].astype(np.float32) / y_scale
        x_scaled = x_pixels[valid_mask].astype(np.float32) / x_scale

        # Computes the weighted centroid and the covariance terms. The covariance encodes ellipse shape and orientation.
        centroid = np.array([np.dot(a=weights, b=y_scaled), np.dot(a=weights, b=x_scaled)], dtype=np.float32)
        sqrt_weights = np.sqrt(weights)
        y_centered = (y_scaled - centroid[0]) * sqrt_weights
        x_centered = (x_scaled - centroid[1]) * sqrt_weights
        covariance_yy = np.dot(a=y_centered, b=y_centered)
        covariance_xx = np.dot(a=x_centered, b=x_centered)
        covariance_yx = np.dot(a=y_centered, b=x_centered)

        # Computes eigenvalues analytically for 2x2 symmetric matrix (faster than np.linalg.eig).
        trace = covariance_yy + covariance_xx
        determinant = covariance_yy * covariance_xx - covariance_yx * covariance_yx
        discriminant = np.sqrt(max(0.0, trace * trace - 4.0 * determinant))
        eigenvalue_1 = (trace + discriminant) / 2.0
        eigenvalue_2 = (trace - discriminant) / 2.0

        # Converts eigenvalues to radii (2.5 sigma boundary captures ~99% of Gaussian distribution).
        sigma_multiplier = 2.5
        radii = sigma_multiplier * np.sqrt(np.maximum(0.0, np.array([eigenvalue_1, eigenvalue_2])))

        # Orders radii as (semi-major, semi-minor) for consistent access.
        sorted_radii = (max(radii[0], radii[1]), min(radii[0], radii[1]))

        return _EllipseData(radii=sorted_radii, y_scale=y_scale, x_scale=x_scale)

    def get_overlap_mask(self, overlap_count_image: NDArray[np.uint16]) -> NDArray[np.bool_]:
        """Computes a boolean mask identifying pixels that overlap with other ROIs.

        Args:
            overlap_count_image: A 2D array where each pixel contains the count of overlapping ROIs.

        Returns:
            A boolean mask indicating pixels that overlap with other ROIs.
        """
        return overlap_count_image[self.y_pixels, self.x_pixels] > 1

    @staticmethod
    def get_overlap_count_image(rois: list[_ROI], height: int, width: int) -> NDArray[np.uint16]:
        """Creates an image showing the count of overlapping ROIs at each pixel.

        Args:
            rois: The list of ROI instances to process.
            height: The height of the field of view from which ROIs are sampled.
            width: The width of the field of view from which ROIs are sampled.

        Returns:
            A 2D array where each pixel contains the count of ROIs covering that pixel.
        """
        overlap = np.zeros((height, width), dtype=np.uint16)
        for roi in rois:
            overlap[roi.y_pixels, roi.x_pixels] += 1
        return overlap

    @staticmethod
    def remove_overlapping_rois(
        rois: list[_ROI],
        overlap_image: NDArray[np.uint16],
        maximum_overlap_fraction: float,
    ) -> list[bool]:
        """Determines which ROIs to keep based on maximum allowed overlap.

        Excessive overlap often indicates over-segmentation or neuropil misidentified as cell body. ROIs are processed
        in reverse order, biasing retention toward earlier-detected (typically higher-quality) ROIs.

        Args:
            rois: The list of ROI instances to filter.
            overlap_image: An image with each pixel set to the number of ROIs overlapping that pixel.
            maximum_overlap_fraction: The maximum fraction of pixels that can overlap with other ROIs.

        Returns:
            A list of booleans indicating which ROIs to keep (True) or remove (False).
        """
        working_overlap = overlap_image.copy()
        keep_flags: list[bool] = []

        for roi in reversed(rois):
            # Caches pixel values to avoid double fancy-indexing when removing ROIs.
            pixels = working_overlap[roi.y_pixels, roi.x_pixels]
            overlap_fraction = np.count_nonzero(pixels > 1) / pixels.size
            keep_roi = bool(overlap_fraction <= maximum_overlap_fraction)
            keep_flags.append(keep_roi)
            if not keep_roi:
                working_overlap[roi.y_pixels, roi.x_pixels] = pixels - 1

        return keep_flags[::-1]

    def _compute_soma_mask(self) -> NDArray[np.bool_]:
        """Computes the soma mask by finding the radius where pixel weight density drops.

        Returns:
            A boolean mask indicating soma pixels.
        """
        minimum_pixels_for_crop = 10

        # Returns all-True mask if cropping is disabled or ROI is too small for meaningful gradient analysis.
        if not self._crop or self.y_pixels.size <= minimum_pixels_for_crop:
            return np.ones(self.y_pixels.size, dtype=np.bool_)

        distances = np.hypot(self.y_pixels - self.centroid[0], self.x_pixels - self.centroid[1])

        # Sorts pixels by distance to enable efficient cumulative weight computation via cumsum.
        sorted_indices = np.argsort(distances)
        sorted_distances = distances[sorted_indices]
        cumulative_weight_sums = np.cumsum(self.pixel_weights[sorted_indices])

        # Samples cumulative weights at integer radii. Uses searchsorted to find the index where each radius would
        # be inserted, then looks up the cumulative weight at that position.
        radii = np.arange(1, int(distances.max()) + 1, dtype=np.float32)
        indices = np.searchsorted(a=sorted_distances, v=radii, side="left")
        cumulative_weights = np.where(
            indices > 0,
            cumulative_weight_sums[np.clip(a=indices - 1, a_min=0, a_max=len(cumulative_weight_sums) - 1)],
            0,
        )

        # Computes radial gradient of cumulative weights. A sharp drop indicates the soma boundary.
        weight_gradient = np.diff(cumulative_weights)
        if weight_gradient.size == 0 or weight_gradient.max() == 0:
            return np.ones(self.y_pixels.size, dtype=np.bool_)

        # Finds the radius where gradient first drops below 1/3 of its peak after rising above threshold.
        gradient_threshold_divisor = 3
        threshold = weight_gradient.max() / gradient_threshold_divisor
        crop_radius = radii[-1]

        above_threshold_indices = np.nonzero(weight_gradient > threshold)[0]
        if above_threshold_indices.size:  # pragma: no branch, non-negative weights guarantee a match
            first_above_index = above_threshold_indices[0]
            below_threshold_after = np.nonzero(weight_gradient[first_above_index:] < threshold)[0]
            if below_threshold_after.size:
                crop_radius = radii[below_threshold_after[0] + first_above_index]

        # Returns mask of pixels within the computed crop radius. The radius is drawn from the same distance array
        # against which the mask is measured, and the gradient that selects it is non-zero only where a pixel sits
        # inside that distance, so at least one pixel always survives the comparison.
        return distances < crop_radius
