"""Provides assets for computing ROI statistics after the initial ROI detection."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes
from scipy.spatial import ConvexHull, QhullError
from ataraxis_base_utilities import console

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..dataclasses import ROIStatistics


def estimate_diameter_from_rois(rois: list[ROIStatistics], default_diameter: int = 10) -> int:
    """Estimates the cell diameter from the pixel counts of a list of ROIs.

    This function computes the median pixel count across all ROIs and derives an equivalent circular diameter. This is
    useful when the original cell diameter is unavailable or when ROI geometry has been transformed (e.g., after
    diffeomorphic registration or template mask generation).

    Args:
        rois: The list of ROIStatistics instances to analyze.
        default_diameter: The fallback diameter to return if the ROI list is empty or all ROIs have zero pixels.

    Returns:
        The estimated cell diameter in pixels, computed as the diameter of a circle with area equal to the median
        ROI pixel count.
    """
    if not rois:
        return default_diameter

    # Collects pixel counts from all ROIs. Uses the y_pixels array length as the authoritative pixel count.
    pixel_counts = np.array([len(roi.y_pixels) for roi in rois], dtype=np.float32)

    if len(pixel_counts) == 0 or np.median(pixel_counts) == 0:
        return default_diameter

    # Computes the diameter of a circle with area equal to the median pixel count: area = π * r², so
    # r = sqrt(area / π) and diameter = 2 * r = 2 * sqrt(median_pixels / π).
    median_pixels = np.median(pixel_counts)
    estimated_diameter = int(2 * np.sqrt(median_pixels / np.pi))

    return max(estimated_diameter, 1)


def compute_median_pixel_position(y_pixels: NDArray[np.int32], x_pixels: NDArray[np.int32]) -> tuple[int, int]:
    """Computes the ROI centroid as the x and y coordinates of the pixel closest to the coordinate-wise median.

    Args:
        y_pixels: The y-coordinates of the ROI's pixels.
        x_pixels: The x-coordinates of the ROI's pixels.

    Returns:
        The (y, x) coordinates of the pixel closest to the median position.
    """
    y_median = np.median(y_pixels)
    x_median = np.median(x_pixels)
    min_index = np.argmin(np.square(x_pixels - x_median) + np.square(y_pixels - y_median))
    return int(y_pixels[min_index]), int(x_pixels[min_index])


_BOUNDARY_PADDING: int = 3
"""The padding added around the mask bounding box for boundary computation."""

_CIRCLE_RADIUS_SCALE: float = 1.25
"""The scale factor applied to the cell radius for circle rendering."""

_CIRCLE_POINT_COUNT: int = 100
"""The number of points used to approximate the circle perimeter."""


def compute_boundary_mask(
    y_pixels: NDArray[np.int32],
    x_pixels: NDArray[np.int32],
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Computes the exterior boundary mask of the ROI specified by the input pixel coordinates.

    Args:
        y_pixels: The row coordinates of the mask pixels.
        x_pixels: The column coordinates of the mask pixels.

    Returns:
        The (y_boundary, x_boundary) arrays containing the boundary mask pixel coordinates.
    """
    # Reshapes the coordinate arrays into column vectors for 2D array indexing.
    y_pixels = np.expand_dims(y_pixels.flatten(), axis=1)
    x_pixels = np.expand_dims(x_pixels.flatten(), axis=1)
    pixel_count = y_pixels.shape[0]

    if not pixel_count:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32)

    # Builds a tight bounding box around the ROI with padding to ensure boundary pixels at the edges of the mask are
    # not clipped during morphological operations.
    y_min = y_pixels.min()
    x_min = x_pixels.min()
    mask = np.zeros(
        (int(y_pixels.max() - y_min) + 2 * _BOUNDARY_PADDING, int(x_pixels.max() - x_min) + 2 * _BOUNDARY_PADDING),
        dtype=np.bool_,
    )

    # Stamps the ROI pixels into the local coordinate system offset by the bounding box origin.
    mask[
        y_pixels - y_min + _BOUNDARY_PADDING,
        x_pixels - x_min + _BOUNDARY_PADDING,
    ] = True

    # Dilates the mask to close single-pixel gaps, then fills interior holes to produce a solid region.
    mask = binary_dilation(mask)
    mask = binary_fill_holes(mask)

    # Uses a 4-connected structuring element (cross pattern) to find the exterior ring. Dilating the background into
    # the foreground and intersecting with the original mask isolates the outermost pixel layer.
    kernel = np.zeros((3, 3), dtype=np.int32)
    kernel[1] = 1
    kernel[:, 1] = 1
    exterior = binary_dilation(mask == 0, structure=kernel) & mask

    # Converts local bounding-box coordinates back to the original frame coordinate system.
    y_boundary, x_boundary = np.nonzero(exterior)
    y_boundary = y_boundary + y_min - _BOUNDARY_PADDING
    x_boundary = x_boundary + x_min - _BOUNDARY_PADDING

    return y_boundary, x_boundary


def compute_circle_mask(
    centroid_y: int,
    centroid_x: int,
    radius: float,
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Computes the pixel coordinates of a circle around the ROI centroid specified by the input coordinates.

    Args:
        centroid_y: The row coordinate of the ROI's center.
        centroid_x: The column coordinate of the ROI's center.
        radius: The radius of the ROI in pixels.

    Returns:
        The (y_circle, x_circle) arrays containing the circle pixel coordinates.
    """
    # Scales the radius up to provide visual clearance around the ROI boundary.
    scaled_radius = radius * _CIRCLE_RADIUS_SCALE
    theta = np.linspace(0.0, 2 * np.pi, num=_CIRCLE_POINT_COUNT)
    y_circle = (scaled_radius * np.sin(theta) + centroid_y).astype(np.int32)
    x_circle = (scaled_radius * np.cos(theta) + centroid_x).astype(np.int32)
    return y_circle, x_circle


def compute_roi_statistics(
    rois: list[ROIStatistics],
    frame_height: int,
    frame_width: int,
    aspect: float | None = None,
    diameter: int | None = None,
    maximum_overlap_fraction: float | None = None,
    crop: bool = True,
    lightweight: bool = False,
) -> None:
    """Computes shape statistics for a list of ROIStatistics instances in-place.

    Notes:
        This function computes statistics (compactness, solidity, radius, aspect ratio, etc.) for each input ROI and
        writes the computed values back to the ROIStatistics instances. If maximum_overlap_fraction is specified, ROIs
        exceeding the overlap threshold are removed from the list in-place. When lightweight is True, only the minimal
        statistics required for preclassification (compactness and normalized_pixel_count) are computed, skipping the
        expensive ellipse fitting, convex hull solidity, and overlap computations.

    Args:
        rois: The list of ROIStatistics instances that define the ROIs to process. Modified in-place.
        frame_height: The height of the recording frames from which ROIs are segmented, in pixels.
        frame_width: The width of the recording frames from which ROIs are segmented, in pixels.
        aspect: The aspect ratio of the recording. If provided, adjusts ROI ellipse fitting. Ignored in lightweight
            mode.
        diameter: The expected cell diameter in pixels. Used for ROI ellipse fitting normalization. Ignored in
            lightweight mode.
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
        if not roi.centroid or roi.centroid == (0, 0):
            roi.centroid = compute_median_pixel_position(y_pixels=roi.y_pixels, x_pixels=roi.x_pixels)

    # Resolves the cell diameter for distance normalization. A sensible default is used when no diameter is provided.
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

    # Pre-allocates arrays to collect statistics for normalization during the computation loop.
    roi_count = len(rois)
    mean_radius_values = np.empty(roi_count, dtype=np.float32)
    pixel_count_values = np.empty(roi_count, dtype=np.float32)
    soma_pixel_count_values = np.empty(roi_count, dtype=np.float32)

    # Computes shape statistics for each ROI and writes them back to the ROIStatistics instances. In lightweight mode,
    # skips the expensive ellipse fitting, convex hull solidity, and overlap mask computations.
    for i, wrapper in enumerate(roi_wrappers):
        data = wrapper.data
        data.mean_radius = wrapper.mean_radius
        data.baseline_mean_radius = wrapper.baseline_mean_radius
        data.compactness = wrapper.compactness
        data.pixel_count = wrapper.pixel_count
        data.soma_pixel_count = wrapper.soma_pixel_count
        data.soma_mask = wrapper.soma_mask

        if not lightweight:
            data.solidity = wrapper.solidity
            # noinspection PyUnboundLocalVariable
            data.overlap_mask = wrapper.get_overlap_mask(overlap_count_image=overlap_counts)

            # noinspection PyUnboundLocalVariable
            ellipse = wrapper.fit_ellipse(y_scale=y_scale, x_scale=x_scale)
            data.radius = ellipse.radius
            data.aspect_ratio = ellipse.aspect_ratio

            boundary_y, boundary_x = compute_boundary_mask(
                y_pixels=data.y_pixels,
                x_pixels=data.x_pixels,
            )
            data.boundary_y_pixels = boundary_y
            data.boundary_x_pixels = boundary_x

            y_circle, x_circle = compute_circle_mask(
                centroid_y=data.centroid[0],
                centroid_x=data.centroid[1],
                radius=data.radius,
            )
            # Clips circle to frame bounds.
            valid = (y_circle >= 0) & (x_circle >= 0) & (y_circle < frame_height) & (x_circle < frame_width)
            data.circle_y_pixels = y_circle[valid]
            data.circle_x_pixels = x_circle[valid]

        # Collects values for normalization to avoid re-iterating over ROIs.
        mean_radius_values[i] = data.mean_radius
        pixel_count_values[i] = data.pixel_count
        soma_pixel_count_values[i] = data.soma_pixel_count

    # Normalizes statistics relative to the first 100 ROIs. Detection algorithms typically find high-confidence ROIs
    # first, so early ROIs serve as a reliable baseline for comparing later, lower-confidence detections.
    normalization_count = 100
    normalization_epsilon = 1e-10

    mean_radius_baseline = np.nanmedian(mean_radius_values[:normalization_count]) + normalization_epsilon
    mean_radius_normalized = mean_radius_values / mean_radius_baseline
    pixel_count_normalized = pixel_count_values / np.mean(pixel_count_values[:normalization_count])
    soma_pixel_count_normalized = soma_pixel_count_values / np.mean(soma_pixel_count_values[:normalization_count])

    for roi, radius_norm, count_norm, soma_count_norm in zip(
        rois, mean_radius_normalized, pixel_count_normalized, soma_pixel_count_normalized, strict=True
    ):
        roi.mean_radius = float(radius_norm)
        roi.normalized_pixel_count_full = float(count_norm)
        roi.normalized_pixel_count = float(soma_count_norm)

    # Removes ROIs with excessive overlap. High overlap often indicates over-segmentation or neuropil contamination.
    # Skipped in lightweight mode since overlap computation is not performed.
    if not lightweight and maximum_overlap_fraction is not None and maximum_overlap_fraction < 1.0:
        keep_flags = _ROI.remove_overlapping_rois(
            rois=roi_wrappers, overlap_image=overlap_counts, maximum_overlap_fraction=maximum_overlap_fraction
        )

        # Uses slice assignment to modify the list in-place.
        rois[:] = [roi for roi, keep in zip(rois, keep_flags, strict=True) if keep]

        # Recomputes overlap masks after removing ROIs, since remaining ROIs may no longer overlap.
        roi_wrappers = [_ROI(data=roi, diameter=effective_diameter, crop=crop) for roi in rois]
        overlap_counts = _ROI.get_overlap_count_image(rois=roi_wrappers, height=frame_height, width=frame_width)
        for wrapper in roi_wrappers:
            wrapper.data.overlap_mask = wrapper.get_overlap_mask(overlap_count_image=overlap_counts)


@dataclass(frozen=True)
class _EllipseData:
    """Defines an ellipse fitted to the ROI's pixels via weighted covariance analysis.

    Notes:
        The radius and aspect ratio derived from this ellipse are used as cell classification features.
    """

    centroid: NDArray[np.float32]
    """The weighted mean of pixel coordinates that serves as the ellipse center point."""

    covariance: NDArray[np.float32]
    """The 2x2 covariance matrix that encodes the ellipse orientation and axis lengths."""

    radii: tuple[float, float]
    """The semi-major and semi-minor axis lengths, ordered from largest to smallest."""

    boundary_points: NDArray[np.float32]
    """The (x, y) coordinates of 100 evenly spaced points along the ellipse perimeter for visualization."""

    y_scale: int
    """The y-axis scaling factor that corrects for non-square pixel aspect ratios during fitting."""

    x_scale: int
    """The x-axis scaling factor that corrects for non-square pixel aspect ratios during fitting."""

    @property
    def area(self) -> float:
        """Returns the area of the ellipse."""
        return float((self.radii[0] * self.radii[1]) ** 0.5 * np.pi)

    @property
    def radius(self) -> float:
        """Returns the effective radius of the ROI ellipse scaled by the mean of y_scale and x_scale."""
        return float(self.radii[0] * np.mean((self.x_scale, self.y_scale)))

    @property
    def aspect_ratio(self) -> float:
        """Returns the normalized aspect ratio bounded between 0 and 2, where 1 indicates a circular shape."""
        major, minor = self.radii
        return 2 * major / (major + minor + 0.01)


def _compute_distance_kernel(radius: int) -> NDArray[np.float32]:
    """Computes a 2D array of Euclidean distances from the center point.

    This function generates a reference distance distribution used to compute the baseline mean R-squared value for ROI
    compactness calculations.

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
        The class uses a shared class variable for the distance kernel to avoid recomputation across instances. The
        soma mask is cached after first computation to avoid redundant calculations when accessing dependent
        properties. Distance-based statistics (mean_radius, compactness) are normalized by the cell diameter to make
        them scale-invariant across different cell sizes and imaging magnifications.

    Args:
        data: The ROIStatistics instance to wrap.
        diameter: The estimated cell diameter in pixels, used to normalize distance-based statistics.
        crop: Determines whether to crop to soma region when computing statistics.

    Attributes:
        _data: The underlying ROIStatistics instance.
        _diameter: The cell diameter used for distance normalization.
        _crop: Determines whether to crop to soma region when computing statistics.
        _cached_soma_mask: Cached soma mask array, computed on first access.

    Raises:
        TypeError: If the x_pixels, y_pixels, and pixel_weights arrays do not have the same shape.
    """

    _baseline_cache: ClassVar[dict[int, NDArray[np.float32]]] = {}
    """Cache of sorted baseline distances keyed by diameter, avoiding recomputation across instances."""

    def __init__(self, data: ROIStatistics, diameter: int, crop: bool = True) -> None:
        if data.x_pixels.shape != data.y_pixels.shape or data.x_pixels.shape != data.pixel_weights.shape:
            message = (
                "Unable to initialize the ROI class. The x_pixels, y_pixels, and pixel_weights arrays in the input "
                "ROIStatistics instance must have the same shape."
            )
            console.error(message=message, error=TypeError)

        self._data: ROIStatistics = data
        self._diameter: int = diameter
        self._crop: bool = crop
        self._cached_soma_mask: NDArray[np.bool_] | None = None

    @property
    def data(self) -> ROIStatistics:
        """Returns the underlying ROIStatistics instance."""
        return self._data

    @property
    def y_pixels(self) -> NDArray[np.int32]:
        """Returns the y-coordinates of the ROI pixels."""
        return self._data.y_pixels

    @property
    def x_pixels(self) -> NDArray[np.int32]:
        """Returns the x-coordinates of the ROI pixels."""
        return self._data.x_pixels

    @property
    def pixel_weights(self) -> NDArray[np.float32]:
        """Returns the pixel weights (lambda values) for the ROI."""
        return self._data.pixel_weights

    @property
    def centroid(self) -> tuple[int, int]:
        """Returns the centroid (y, x) pixel position of the ROI."""
        return self._data.centroid[0], self._data.centroid[1]

    @property
    def soma_mask(self) -> NDArray[np.bool_]:
        """Computes and caches the soma mask for this ROI.

        Returns:
            A boolean mask indicating which pixels belong to the soma region.
        """
        if self._cached_soma_mask is not None:
            return self._cached_soma_mask

        self._cached_soma_mask = self._compute_soma_mask()
        return self._cached_soma_mask

    @property
    def mean_radius(self) -> float:
        """Computes the mean diameter-normalized distance from ROI pixels to their median center."""
        y_pixels = self.y_pixels[self.soma_mask]
        x_pixels = self.x_pixels[self.soma_mask]
        # Normalizes distances by cell diameter for scale-invariance, matching the original suite2p approach.
        distances = np.hypot(
            (y_pixels - np.median(y_pixels)) / self._diameter,
            (x_pixels - np.median(x_pixels)) / self._diameter,
        )
        return float(np.mean(distances))

    @property
    def baseline_mean_radius(self) -> float:
        """Computes the expected mean radius for a uniformly distributed set of pixels of the same count as the ROI."""
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
        """Computes the ratio of actual to expected mean radius, where values near 1 indicate compact circular ROIs."""
        return max(1.0, self.mean_radius / (1e-10 + self.baseline_mean_radius))

    @property
    def solidity(self) -> float:
        """Computes the ROI's solidity as the ratio of pixel count to convex hull area."""
        minimum_pixels_for_hull = 10
        default_area = 10.0

        pixel_count = self.soma_pixel_count
        if pixel_count <= minimum_pixels_for_hull:
            return pixel_count / default_area

        # ConvexHull requires (N, 2) array of points.
        points = np.column_stack((self.y_pixels[self.soma_mask], self.x_pixels[self.soma_mask]))
        try:
            area = ConvexHull(points).volume
        except ValueError, QhullError:
            area = default_area

        return pixel_count / area

    @property
    def soma_pixel_count(self) -> int:
        """Returns the number of pixels in the soma region."""
        return int(self.soma_mask.sum())

    @property
    def pixel_count(self) -> int:
        """Returns the total number of pixels in the ROI."""
        return self.x_pixels.size

    def _compute_soma_mask(self) -> NDArray[np.bool_]:
        """Computes the soma mask by finding the radius where pixel weight density drops.

        Returns:
            A boolean mask indicating soma pixels.
        """
        minimum_pixels_for_crop = 10

        # Returns all-True mask if cropping is disabled or ROI is too small for meaningful gradient analysis.
        if not self._crop or self.y_pixels.size <= minimum_pixels_for_crop:
            return np.ones(self.y_pixels.size, dtype=np.bool_)

        # Computes Euclidean distance from each pixel to the ROI centroid.
        distances = np.hypot(self.y_pixels - self.centroid[0], self.x_pixels - self.centroid[1])

        # Sorts pixels by distance to enable efficient cumulative weight computation via cumsum.
        sorted_indices = np.argsort(distances)
        sorted_distances = distances[sorted_indices]
        cumsum_weights = np.cumsum(self.pixel_weights[sorted_indices])

        # Samples cumulative weights at integer radii. Uses searchsorted to find the index where each radius would
        # be inserted, then looks up the cumulative weight at that position.
        radii = np.arange(1, int(distances.max()) + 1, dtype=np.float32)
        indices = np.searchsorted(sorted_distances, radii, side="left")
        cumulative_weights = np.where(indices > 0, cumsum_weights[np.clip(indices - 1, 0, len(cumsum_weights) - 1)], 0)

        # Computes radial gradient of cumulative weights. A sharp drop indicates the soma boundary.
        weight_gradient = np.diff(cumulative_weights)
        if weight_gradient.size == 0 or weight_gradient.max() == 0:
            return np.ones(self.y_pixels.size, dtype=np.bool_)

        # Finds the radius where gradient first drops below 1/3 of its peak after rising above threshold.
        gradient_threshold_divisor = 3
        threshold = weight_gradient.max() / gradient_threshold_divisor
        crop_radius = radii[-1]

        above_threshold_indices = np.nonzero(weight_gradient > threshold)[0]
        if len(above_threshold_indices) > 0:
            first_above_index = above_threshold_indices[0]
            below_threshold_after = np.nonzero(weight_gradient[first_above_index:] < threshold)[0]
            if len(below_threshold_after) > 0:
                crop_radius = radii[below_threshold_after[0] + first_above_index]

        # Returns mask of pixels within the computed crop radius.
        crop_mask = distances < crop_radius
        if crop_mask.sum() == 0:
            return np.ones(self.y_pixels.size, dtype=np.bool_)

        return crop_mask

    def fit_ellipse(self, y_scale: int, x_scale: int) -> _EllipseData:
        """Fits a 2D Gaussian ellipse to the ROI pixels via covariance eigendecomposition.

        The fitted ellipse's radius and aspect ratio are used as cell classification features.

        Args:
            y_scale: The y-axis scaling factor for correcting non-square pixel aspect ratios.
            x_scale: The x-axis scaling factor for correcting non-square pixel aspect ratios.

        Returns:
            The fitted ellipse parameters including center, covariance, radii, and boundary points, packaged into an
            _EllipseData instance.
        """
        y_pixels = self.y_pixels[self.soma_mask]
        x_pixels = self.x_pixels[self.soma_mask]
        pixel_weights = self.pixel_weights[self.soma_mask]

        # Filters zero-weight pixels and normalizes weights to form a probability distribution for weighted statistics.
        valid_mask = pixel_weights > 0
        weights = pixel_weights[valid_mask]
        weights = weights / weights.sum()

        # Scales coordinates to correct for non-square pixel aspect ratios before computing covariance.
        y_scaled = y_pixels[valid_mask].astype(np.float32) / y_scale
        x_scaled = x_pixels[valid_mask].astype(np.float32) / x_scale

        # Computes weighted centroid and covariance matrix. The covariance encodes ellipse shape and orientation.
        centroid = np.array([np.dot(weights, y_scaled), np.dot(weights, x_scaled)], dtype=np.float32)
        sqrt_weights = np.sqrt(weights)
        y_centered = (y_scaled - centroid[0]) * sqrt_weights
        x_centered = (x_scaled - centroid[1]) * sqrt_weights
        covariance_yy = np.dot(y_centered, y_centered)
        covariance_xx = np.dot(x_centered, x_centered)
        covariance_yx = np.dot(y_centered, x_centered)
        covariance = np.array([[covariance_yy, covariance_yx], [covariance_yx, covariance_xx]], dtype=np.float32)

        # Computes eigenvalues analytically for 2x2 symmetric matrix (faster than np.linalg.eig).
        trace = covariance_yy + covariance_xx
        determinant = covariance_yy * covariance_xx - covariance_yx * covariance_yx
        discriminant = np.sqrt(max(0.0, trace * trace - 4.0 * determinant))
        eigenvalue_1 = (trace + discriminant) / 2.0
        eigenvalue_2 = (trace - discriminant) / 2.0

        # Computes eigenvectors. Falls back to axis-aligned vectors when covariance is diagonal.
        covariance_epsilon = 1e-10
        if abs(covariance_yx) > covariance_epsilon:
            eigenvector_1 = np.array([eigenvalue_1 - covariance_xx, covariance_yx], dtype=np.float32)
            eigenvector_1 = eigenvector_1 / np.hypot(eigenvector_1[0], eigenvector_1[1])
            eigenvector_2 = np.array([-eigenvector_1[1], eigenvector_1[0]], dtype=np.float32)
        else:
            eigenvector_1 = np.array([1.0, 0.0], dtype=np.float32)
            eigenvector_2 = np.array([0.0, 1.0], dtype=np.float32)

        # Converts eigenvalues to radii (2.5 sigma boundary captures ~99% of Gaussian distribution).
        sigma_multiplier = 2.5
        eigenvectors = np.column_stack((eigenvector_1, eigenvector_2))
        radii = sigma_multiplier * np.sqrt(np.maximum(0.0, np.array([eigenvalue_1, eigenvalue_2])))

        # Generates boundary points by transforming a unit circle through the eigenvector basis.
        boundary_point_count = 100
        theta = np.linspace(0, 2 * np.pi, boundary_point_count, dtype=np.float32)
        unit_circle = np.column_stack((np.cos(theta), np.sin(theta)))
        boundary_points = (unit_circle * radii) @ eigenvectors.T + centroid

        # Orders radii as (semi-major, semi-minor) for consistent access.
        sorted_radii = (max(radii[0], radii[1]), min(radii[0], radii[1]))

        return _EllipseData(
            centroid=centroid,
            covariance=covariance,
            radii=sorted_radii,
            boundary_points=boundary_points.astype(np.float32),
            y_scale=y_scale,
            x_scale=x_scale,
        )

    def get_overlap_mask(self, overlap_count_image: NDArray[np.uint16]) -> NDArray[np.bool_]:
        """Computes a mask that communicates which pixels overlap with other ROIs.

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
