"""Provides the core iterative multiscale ROI detection algorithm for calcium imaging data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import mode
from numpy.linalg import norm
from scipy.ndimage import maximum_filter, uniform_filter
from scipy.interpolate import RectBivariateSpline
from ataraxis_base_utilities import LogLevel, console

from .utils import (
    downsample,
    compute_thresholded_variance,
    apply_temporal_high_pass_filter,
    compute_temporal_standard_deviation,
)
from ..dataclasses import ROIStatistics

if TYPE_CHECKING:
    from numpy.typing import NDArray

_MINIMUM_WEIGHT_FRACTION: float = 0.2
"""The fraction of the maximum weight below which pixels are excluded from the ROI."""

_MINIMUM_SPATIAL_SCALE: int = 1
"""The minimum valid spatial scale. Scale 0 is excluded because the base filter size already covers the finest
resolution."""

_NORMALIZATION_EPSILON: float = 1e-6
"""The small epsilon added to denominators during alternating least squares normalization to prevent division by
zero."""


def extend_roi(
    y_pixels: NDArray[np.int32],
    x_pixels: NDArray[np.int32],
    height: int,
    width: int,
    iterations: int = 1,
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Uniformly extends the input ROI by iteratively adding cardinal neighbors to all boundary pixels.

    Notes:
        The expansion follows a Manhattan distance pattern, producing diamond-shaped growth rather than square
        expansion. Each iteration adds one layer of cardinal (up, down, left, right) neighbors to the existing
        pixel set.

    Args:
        y_pixels: The y-coordinates of the ROI pixels.
        x_pixels: The x-coordinates of the ROI pixels.
        height: The height of the recording frame that contains the ROI.
        width: The width of the recording frame that contains the ROI.
        iterations: The number of iterations to use for the ROI growth. Each iteration expands the ROI's
            bounding box by 1 pixel in each direction.

    Returns:
        A tuple of two arrays. The first array stores the extended ROI pixel y-coordinates. The second array stores
        the extended ROI pixel x-coordinates.
    """
    for _ in range(iterations):
        # Expands the ROI by 1 pixel in each cardinal direction (center, right, left, up, down).
        expanded_y = np.concatenate((y_pixels, y_pixels, y_pixels, y_pixels - 1, y_pixels + 1))
        expanded_x = np.concatenate((x_pixels, x_pixels + 1, x_pixels - 1, x_pixels, x_pixels))

        # Filters out any coordinates that fall outside the recording frame boundaries.
        valid_mask = (expanded_y >= 0) & (expanded_y < height) & (expanded_x >= 0) & (expanded_x < width)
        expanded_y = expanded_y[valid_mask]
        expanded_x = expanded_x[valid_mask]

        # Encodes each (y, x) pair as a unique flat index and deduplicates using 1D unique, which is significantly
        # faster than 2D axis-based unique.
        flat_indices = np.unique(expanded_y * width + expanded_x)
        y_pixels = (flat_indices // width).astype(np.int32)
        x_pixels = (flat_indices % width).astype(np.int32)

    return y_pixels, x_pixels


def detect(
    frames: NDArray[np.float32],
    temporal_highpass_window: int,
    spatial_highpass_window: int,
    threshold_scaling: float,
    maximum_iterations: int,
    plane_index: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32], int, list[ROIStatistics]]:
    """Detects ROIs in the input frames using an iterative multiscale sparse detection algorithm.

    Notes:
        The algorithm first preprocesses the frames by applying a temporal high-pass filter and subtracting neuropil
        contamination. It then builds a multiscale representation by repeatedly convolving and downsampling the frames.
        Peaks are iteratively detected in the variance maps across scales, and each detected peak is grown into an
        ROI by correlating neighboring pixel activity. ROIs may be split if a two-component model explains
        significantly more variance. Detected ROIs are subtracted from the residual frames before continuing to the
        next peak.

    Args:
        frames: The binned frames with shape (num_frames, height, width). Modified in-place during detection.
        temporal_highpass_window: The temporal high-pass filter kernel size in frames.
        spatial_highpass_window: The spatial filter size for neuropil subtraction.
        threshold_scaling: The multiplier applied to the base threshold for peak acceptance.
        maximum_iterations: The maximum number of ROIs to detect.
        plane_index: The index of the imaging plane being processed, used for logging.

    Returns:
        A tuple of the maximum intensity projection, the pixel-wise correlation map, the estimated spatial scale in
        pixels, and a list of ROIStatistics instances for each detected ROI.
    """
    scale_count = 5
    base_filter_size = 3
    base_threshold_multiplier = 5.0
    extension_iterations = 3
    split_variance_threshold = 1.25
    reference_frame_count = 1200

    # Preprocessing.
    # Removes slow temporal drift so that transient calcium events dominate the signal.
    apply_temporal_high_pass_filter(frames=frames, kernel_size=int(temporal_highpass_window))

    # Captures the max projection before normalization because downstream consumers expect it in the original
    # intensity scale for visualization.
    maximum_projection = frames.max(axis=0)
    temporal_standard_deviation = compute_temporal_standard_deviation(frames=frames)

    # Divides out per-pixel temporal variability so that dim and bright regions contribute equally to detection, then
    # removes spatially smooth neuropil contamination to isolate somatic signals.
    frames /= temporal_standard_deviation
    _subtract_neuropil(frames=frames, filter_size=spatial_highpass_window)

    _, height, width = frames.shape

    # Multiscale pyramid construction.
    # Builds the finest-resolution coordinate grid in float32 to avoid an int64 intermediate and copy. The grid
    # tracks each scale's pixel positions for mapping detected peaks back to the finest resolution.
    coordinate_grid = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    grid_coordinates = [np.stack(coordinate_grid)]
    downsampled_frames = frames
    convolved_scales: list[NDArray[np.float32]] = []

    # Constructs the scale pyramid by alternating convolution (to aggregate local activity) and downsampling (to
    # capture progressively larger spatial features). Each scale doubles the effective receptive field.
    scale_heights = np.zeros(scale_count, dtype=np.uint16)
    scale_widths = np.zeros(scale_count, dtype=np.uint16)
    for scale_index in range(scale_count):
        convolved_scale = _convolve_square_2d(frames=downsampled_frames, filter_size=base_filter_size)
        downsampled_frames = (2 * downsample(data=downsampled_frames)).astype(np.float32)
        scale_coordinates = downsample(data=grid_coordinates[scale_index], taper_edge=False)
        grid_coordinates.append(scale_coordinates)
        _, scale_heights[scale_index], scale_widths[scale_index] = convolved_scale.shape
        convolved_scales.append(convolved_scale)

    # Upsamples each scale's maximum projection back to the finest grid via bivariate splines so that all scales can
    # be compared in the same coordinate system.
    scale_images = np.zeros(
        (len(grid_coordinates), grid_coordinates[0].shape[1], grid_coordinates[0].shape[2]),
        dtype=np.float32,
    )
    for convolved_scale, scale_coordinates, scale_projection in zip(
        convolved_scales,
        grid_coordinates,
        scale_images,
        strict=False,
    ):
        spline_model = RectBivariateSpline(
            scale_coordinates[1, :, 0],
            scale_coordinates[0, 0, :],
            convolved_scale.max(axis=0),
            kx=min(3, scale_coordinates.shape[1] - 1),
            ky=min(3, scale_coordinates.shape[2] - 1),
        )
        # Evaluates the spline on the finest grid. Returns float64 internally, but truncates to float32 on assignment.
        scale_projection[:] = spline_model(grid_coordinates[0][1, :, 0], grid_coordinates[0][0, 0, :])

    # Computes the cross-scale maximum as the correlation map for visualization and downstream quality assessment.
    correlation_map = scale_images.max(axis=0)

    # Scale selection and threshold computation.
    scale = _find_best_scale(scale_images=scale_images)

    spatial_scale_pixels = base_filter_size * 2**scale
    peak_threshold = threshold_scaling * base_threshold_multiplier * max(1, scale)
    # Scales the threshold by the ratio of actual to reference frame count so that longer recordings, which
    # accumulate more variance, do not produce artificially many detections.
    time_multiplier = max(1, frames.shape[0] / reference_frame_count)
    # Precomputes the effective detection threshold since both factors are constant across iterations.
    scaled_threshold = time_multiplier * peak_threshold
    message = (
        f"Plane {plane_index} detection: estimated target cell diameter ~{int(spatial_scale_pixels)} pixels, "
        f"using {frames.shape[0]} binned frames and a minimum peak activity threshold of {round(scaled_threshold, 2)}."
    )
    console.echo(message=message, level=LogLevel.INFO)

    # Detection loop setup.
    # Initializes variance maps as activity heatmaps from which peaks are drawn. Updates after each ROI subtraction
    # ensure that already-explained activity no longer contributes to future detections.
    variance_maps = [
        compute_thresholded_variance(frames=convolved_scale, intensity_threshold=peak_threshold)
        for convolved_scale in convolved_scales
    ]

    # Flattens the spatial dimensions so that pixel indexing uses 1D flat indices throughout the detection loop.
    convolved_scales = [convolved_scale.reshape(convolved_scale.shape[0], -1) for convolved_scale in convolved_scales]
    frames = frames.reshape(-1, height * width)
    filter_sizes = base_filter_size * 2 ** np.arange(scale_count)

    peak_magnitude = 0.0
    exhausted_activity = False
    roi_statistics: list[ROIStatistics] = []
    for _ in range(maximum_iterations):
        # Peak selection.
        # Selects the globally strongest peak across all spatial scales, then maps it back to the finest grid so that
        # the ROI is grown in full-resolution coordinates.
        scale_maxima = np.array([variance_maps[scale_index].max() for scale_index in range(scale_count)])
        best_scale_index = np.argmax(scale_maxima)
        peak_index = np.argmax(variance_maps[best_scale_index])
        peak_y, peak_x = np.unravel_index(peak_index, (scale_heights[best_scale_index], scale_widths[best_scale_index]))

        peak_y, peak_x = (
            grid_coordinates[best_scale_index][1, peak_y, peak_x],
            grid_coordinates[best_scale_index][0, peak_y, peak_x],
        )
        centroid = (int(peak_y), int(peak_x))

        # Terminates when the strongest remaining peak falls below the detection threshold, indicating that only
        # noise-level activity remains in the residual.
        peak_magnitude = float(scale_maxima[best_scale_index])
        if peak_magnitude < scaled_threshold:
            exhausted_activity = True
            break
        filter_size = filter_sizes[best_scale_index]

        # Initial ROI seed.
        # Seeds the ROI as a square patch at the peak location. The patch size matches the spatial scale's filter size
        # so that the initial footprint is proportional to the expected cell diameter at this scale.
        y_pixels, x_pixels, pixel_weights = _create_initial_square(
            center_y=int(peak_y),
            center_x=int(peak_x),
            square_size=filter_size,
            height=height,
            width=width,
        )

        # Computes the initial time series by projecting the residual frames onto the seed ROI.
        flat_indices = y_pixels * width + x_pixels
        time_projection = frames[:, flat_indices] @ pixel_weights
        active_frame_indices = np.nonzero(time_projection > peak_threshold)[0]

        # ROI growth.
        # Repeatedly extends the ROI boundary and re-estimates weights from the residual. Multiple passes allow the
        # mask to converge to the cell's true spatial extent by incorporating increasingly distant correlated pixels.
        for _extension_pass in range(extension_iterations):
            y_pixels, x_pixels, pixel_weights = _extend_iteratively(
                y_pixels=y_pixels,
                x_pixels=x_pixels,
                frames=frames,
                height=height,
                width=width,
                active_frame_indices=active_frame_indices,
            )
            flat_indices = y_pixels * width + x_pixels
            time_projection = frames[:, flat_indices] @ pixel_weights
            active_frame_indices = np.nonzero(time_projection > peak_threshold)[0]
            if len(active_frame_indices) < 1:
                break
        if len(active_frame_indices) < 1:
            continue

        # Component splitting.
        # Tests whether a two-component model explains significantly more variance than the single-component model. If
        # so, the ROI likely contains two overlapping cells and the dominant component is retained.
        split_ratio, component_pack = _check_split_components(
            pixel_frames=frames[:, flat_indices],
            weights=pixel_weights,
            intensity_threshold=peak_threshold,
        )
        if split_ratio > split_variance_threshold:
            pixel_weights, temporal_projections, active_frame_mask = component_pack
            active_frame_indices = np.nonzero(active_frame_mask)[0]
            time_projection[active_frame_indices] = temporal_projections
            # Discards pixels with negligible weight to tighten the mask around the dominant component's soma.
            valid_weight_mask = pixel_weights > pixel_weights.max() * _MINIMUM_WEIGHT_FRACTION
            x_pixels = x_pixels[valid_weight_mask]
            y_pixels = y_pixels[valid_weight_mask]
            pixel_weights = pixel_weights[valid_weight_mask]
            # Updates the centroid to the pixel closest to the spatial median of the split component.
            median_y = np.median(y_pixels)
            median_x = np.median(x_pixels)
            closest_pixel_index = np.argmin((x_pixels - median_x) ** 2 + (y_pixels - median_y) ** 2)
            centroid = (int(y_pixels[closest_pixel_index]), int(x_pixels[closest_pixel_index]))
            flat_indices = y_pixels * width + x_pixels

        # Residual subtraction.
        # Removes the detected ROI's contribution from the residual frames so that subsequent iterations detect new
        # cells rather than re-detecting the same activity.
        frames[np.ix_(active_frame_indices, flat_indices)] -= (
            time_projection[active_frame_indices][:, np.newaxis] * pixel_weights
        )

        # Propagates the subtraction to all spatial scales so that variance maps remain consistent with the residual.
        # Without this, already-explained variance would persist in coarser scales and attract spurious detections.
        multiscale_y, multiscale_x, multiscale_weights = _compute_multiscale_masks(
            y_pixels=y_pixels,
            x_pixels=x_pixels,
            weights=pixel_weights,
            scale_heights=scale_heights,
            scale_widths=scale_widths,
        )
        for scale_index in range(scale_count):
            scale_flat_indices = multiscale_x[scale_index] + scale_widths[scale_index] * multiscale_y[scale_index]
            convolved_scales[scale_index][np.ix_(active_frame_indices, scale_flat_indices)] -= np.outer(
                time_projection[active_frame_indices], multiscale_weights[scale_index]
            )
            # Fancy indexing produces an independent copy, so in-place zeroing avoids the extra temporary that
            # np.where would allocate.
            residual_activity = convolved_scales[scale_index][:, scale_flat_indices]
            residual_activity[residual_activity <= peak_threshold] = 0
            variance_maps[scale_index][multiscale_y[scale_index], multiscale_x[scale_index]] = norm(
                residual_activity, axis=0
            ).astype(np.float32)

        # Rescales the pixel weights back to the original intensity scale before storing, since the detection operated
        # on temporally-standardized frames.
        roi_statistics.append(
            ROIStatistics(
                y_pixels=y_pixels.astype(np.int32),
                x_pixels=x_pixels.astype(np.int32),
                pixel_weights=pixel_weights * temporal_standard_deviation[y_pixels, x_pixels],
                centroid=centroid,
                footprint=int(best_scale_index),
            )
        )

    if exhausted_activity:
        message = (
            f"Found {len(roi_statistics)} plane {plane_index} ROIs "
            f"(exhausted activity at {round(peak_magnitude, 2)} / {round(scaled_threshold, 2)} threshold)."
        )
    else:
        message = (
            f"Found {len(roi_statistics)} plane {plane_index} ROIs "
            f"(reached iteration limit with peak activity {round(peak_magnitude, 2)} remaining)."
        )
    console.echo(message=message, level=LogLevel.SUCCESS)

    return maximum_projection, correlation_map, spatial_scale_pixels, roi_statistics


def _subtract_neuropil(frames: NDArray[np.float32], filter_size: int) -> None:
    """Subtracts a low-pass filtered version of each frame from itself in-place to remove neuropil contamination.

    Notes:
        Each frame is individually high-pass filtered by subtracting a uniform-filter smoothed version of itself,
        normalized by the filter's spatial response on a constant image.

    Args:
        frames: The frame data with shape (num_frames, height, width). Modified in-place.
        filter_size: The spatial extent of the uniform filter kernel.
    """
    _, height, width = frames.shape

    # Precomputes the reciprocal of the boundary normalization factor. The uniform filter on a constant image with
    # zero-padded boundaries produces values < 1 near edges, so dividing by it corrects for the reduced kernel overlap.
    boundary_response = uniform_filter(np.ones((height, width), dtype=np.float32), size=filter_size, mode="constant")
    inverse_normalization = np.float32(1.0) / boundary_response

    # Applies the uniform filter to all frames simultaneously using a 3D kernel with size 1 along the temporal axis.
    smoothed = uniform_filter(frames, size=(1, filter_size, filter_size), mode="constant")
    smoothed *= inverse_normalization
    frames -= smoothed


def _convolve_square_2d(frames: NDArray[np.float32], filter_size: int) -> NDArray[np.float32]:
    """Convolves each frame with a square uniform kernel.

    Notes:
        The uniform filter computes a local mean, so the result is scaled by filter_size to approximate a box sum
        rather than a box average.

    Args:
        frames: The frame data with shape (num_frames, height, width).
        filter_size: The side length of the square convolution kernel.

    Returns:
        The spatially convolved frames with the same shape as the input.
    """
    # Applies the uniform filter to all frames simultaneously using a 3D kernel with size 1 along the temporal axis.
    convolved_frames = uniform_filter(frames, size=(1, filter_size, filter_size), mode="constant")
    convolved_frames *= filter_size
    return convolved_frames


def _create_initial_square(
    center_y: int,
    center_x: int,
    square_size: int,
    height: int,
    width: int,
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float32]]:
    """Creates a square patch of pixels centered at the specified location with unit-normalized weights.

    Args:
        center_y: The y-coordinate of the square center.
        center_x: The x-coordinate of the square center.
        square_size: The side length of the square patch in pixels.
        height: The full image height in pixels.
        width: The full image width in pixels.

    Returns:
        A tuple of three arrays: the y-coordinates, x-coordinates, and normalized weights for the valid pixels within
        the square patch.
    """
    # Builds a 2D grid of integer offsets from the center. Tiling the 1D offset range row-wise produces x-offsets, and
    # transposing it produces y-offsets, so that adding the center coordinate yields a full square_size x square_size
    # coordinate grid.
    half_size = int((square_size - 1) / 2)
    pixel_offsets = np.tile(np.arange(-half_size, -half_size + square_size, dtype=np.int32), reps=(square_size, 1))
    x_coordinates = center_x + pixel_offsets
    y_coordinates = center_y + pixel_offsets.T

    # Discards any pixels that fall outside the image boundaries and normalizes the surviving weights to unit norm.
    weights = np.ones_like(pixel_offsets, dtype=np.float32)
    valid_mask = np.all(
        (y_coordinates >= 0, y_coordinates < height, x_coordinates >= 0, x_coordinates < width),
        axis=0,
    )
    x_coordinates = x_coordinates[valid_mask]
    y_coordinates = y_coordinates[valid_mask]
    weights = weights[valid_mask]
    weights = (weights / norm(weights)).astype(np.float32)
    return y_coordinates.flatten(), x_coordinates.flatten(), weights.flatten()


def _check_split_components(
    pixel_frames: NDArray[np.float32],
    weights: NDArray[np.float32],
    intensity_threshold: float,
) -> tuple[float, tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_]]]:
    """Checks whether splitting an ROI into two components increases the explained variance.

    Notes:
        Performs alternating least squares to find two non-negative spatial components. If the variance explained by
        the two-component model exceeds that of the single component, the ROI should be split. The pixel_frames array
        is modified in-place as a working buffer to avoid a redundant full-array copy, since the caller always passes
        a fancy-indexed slice that is already an independent copy of the original data.

    Args:
        pixel_frames: The temporal activity of the ROI's pixels, with shape (num_frames, num_roi_pixels). Each row
            is a time-binned frame and each column is one pixel belonging to the ROI. Modified in-place.
        weights: The current pixel weights for the ROI.
        intensity_threshold: The minimum projection intensity for a frame to be considered active.

    Returns:
        A tuple containing the variance ratio (two-component / single-component explained variance) and a tuple of
        the best component's spatial mask, temporal projections on active frames, and active frame boolean mask.
    """
    # Captures the total energy before any in-place modifications, since pixel_frames is used as a working buffer.
    total_energy = np.dot(pixel_frames.ravel(), pixel_frames.ravel())

    # Establishes a single-component baseline to compare against. Computes the actual explained variance as the
    # difference in total energy before and after subtracting the single-component model.
    projection = pixel_frames @ weights
    active_frames = projection > intensity_threshold
    active_projection = projection[active_frames]
    pixel_frames[active_frames, :] -= np.outer(active_projection, weights)
    single_component_variance = total_energy - np.dot(pixel_frames.ravel(), pixel_frames.ravel())

    # Seeds the two-component split from the residual's most energetic frame: pixels with negative vs positive
    # residual are assigned to separate components, capturing the spatial pattern that the single component missed.
    seed_frame_index = np.argmax(np.maximum(pixel_frames, 0).sum(axis=1))
    component_masks = [
        np.where(pixel_frames[seed_frame_index] < 0, weights, np.float32(0)),
        np.where(pixel_frames[seed_frame_index] > 0, weights, np.float32(0)),
    ]

    # Reverses the single-component subtraction in-place rather than allocating a second full copy.
    pixel_frames[active_frames, :] += np.outer(active_projection, weights)

    # Sequentially subtracts each initial component so pixel_frames holds the two-component residual.
    component_frames: list[NDArray[np.bool_]] = []
    projections: list[NDArray[np.float32]] = []
    for component_mask in component_masks:
        component_mask[:] /= norm(component_mask) + _NORMALIZATION_EPSILON
        temporal_projection = pixel_frames @ component_mask
        pixel_frames[active_frames, :] -= np.outer(temporal_projection[active_frames], component_mask)
        component_frames.append(active_frames)
        projections.append(temporal_projection[active_frames])

    # Performs alternating least squares by temporarily restoring each component so it can be re-estimated from the
    # residual left by the other component alone. Clamps negative spatial weights to zero to enforce non-negativity.
    converged = [False, False]
    component_variances = np.zeros(2)
    for _iteration in range(3):
        for component_index in range(2):
            if converged[component_index]:
                continue

            # Restores this component's contribution so re-projection sees the other component's residual only.
            pixel_frames[component_frames[component_index], :] += np.outer(
                projections[component_index], component_masks[component_index]
            )
            temporal_projection = pixel_frames @ component_masks[component_index]
            component_frames[component_index] = temporal_projection > intensity_threshold
            component_variances[component_index] = np.dot(temporal_projection, temporal_projection)
            active_count = np.sum(component_frames[component_index])
            if active_count == 0:
                converged[component_index] = True
                component_variances[component_index] = -1
                continue

            # Updates the spatial mask via projection-weighted mean of active frames, then re-subtracts.
            projections[component_index] = temporal_projection[component_frames[component_index]]
            component_masks[component_index] = (
                projections[component_index] @ pixel_frames[component_frames[component_index], :] / active_count
            ).astype(np.float32)
            component_masks[component_index][component_masks[component_index] < 0] = 0
            component_masks[component_index] /= _NORMALIZATION_EPSILON + norm(component_masks[component_index])
            pixel_frames[component_frames[component_index], :] -= np.outer(
                projections[component_index], component_masks[component_index]
            )

    # Computes the variance ratio, where values above the caller's split threshold indicate that the two-component
    # model explains meaningfully more activity than the single component.
    best_component = np.argmax(component_variances)
    residual_energy = np.dot(pixel_frames.ravel(), pixel_frames.ravel())
    variance_ratio = (total_energy - residual_energy) / single_component_variance
    return variance_ratio, (
        component_masks[best_component],
        projections[best_component],
        component_frames[best_component],
    )


def _extend_mask(
    y_pixels: NDArray[np.int32],
    x_pixels: NDArray[np.int32],
    weights: NDArray[np.float32],
    height: int,
    width: int,
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float32]]:
    """Extends a pixel mask into all 8 surrounding neighbors, distributing weights proportionally.

    Notes:
        Uses a dense bounding-box accumulator instead of coordinate tiling and sorting. Each of the 9 directional
        offsets (center plus 8 neighbors) is scatter-added into a small local grid, avoiding the 9x coordinate
        replication and the O(n log n) np.unique deduplication step.

    Args:
        y_pixels: The y-coordinates of the mask pixels.
        x_pixels: The x-coordinates of the mask pixels.
        weights: The pixel weights for the mask.
        height: The image height in pixels.
        width: The image width in pixels.

    Returns:
        A tuple of three arrays: the extended y-coordinates, x-coordinates, and accumulated weights for the expanded
        mask.
    """
    # Computes a tight bounding box that encloses the original pixels plus 1-pixel expansion in each direction.
    min_y = max(0, int(y_pixels.min()) - 1)
    max_y = min(height - 1, int(y_pixels.max()) + 1)
    min_x = max(0, int(x_pixels.min()) - 1)
    max_x = min(width - 1, int(x_pixels.max()) + 1)
    box_height = max_y - min_y + 1
    box_width = max_x - min_x + 1

    # Shifts coordinates into the local bounding-box frame and divides weights by 3 (each pixel distributes its
    # weight across 3 columns: left, center, right). Uses a copy to avoid mutating the caller's array.
    local_y = y_pixels - min_y
    local_x = x_pixels - min_x
    weights = (weights / 3).astype(np.float32)

    # Accumulates weights from each of the 9 directional offsets into the dense local grid. Boundary checks per
    # offset are cheaper than a single check on the full 9x-expanded array.
    accumulator = np.zeros((box_height, box_width), dtype=np.float32)
    for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)):
        shifted_y = local_y + dy
        shifted_x = local_x + dx
        valid = (shifted_y >= 0) & (shifted_y < box_height) & (shifted_x >= 0) & (shifted_x < box_width)
        np.add.at(accumulator, (shifted_y[valid], shifted_x[valid]), weights[valid])

    # Extracts the non-zero pixels from the accumulator and maps back to full-frame coordinates.
    nonzero_y, nonzero_x = np.nonzero(accumulator)
    extended_y = (nonzero_y + min_y).astype(np.int32)
    extended_x = (nonzero_x + min_x).astype(np.int32)
    # noinspection PyTypeChecker
    return extended_y, extended_x, accumulator[nonzero_y, nonzero_x]


def _estimate_spatial_scale(scale_images: NDArray[np.float32]) -> int:
    """Estimates the dominant spatial scale from multiscale projection images.

    Notes:
        The dominant scale is determined by finding the mode of the best scale index across the top peaks in the
        maximum projection image. This approach identifies which downsampling level captures the most prominent
        features.

    Args:
        scale_images: The multiscale projection images with shape (num_scales, height, width).

    Returns:
        The estimated spatial scale index corresponding to the dominant feature size.
    """
    peak_detection_window = 11
    peak_tolerance = 1e-4
    peak_count = 50

    max_projection = scale_images.max(axis=0)
    scale_map = np.argmax(scale_images, axis=0).ravel()

    # Restricts scale voting to local maxima so that broad bright regions do not dominate the vote count.
    flat_projection = max_projection.ravel()
    neighborhood_max = maximum_filter(max_projection, size=peak_detection_window).ravel()
    is_peak = np.abs(flat_projection - neighborhood_max) < peak_tolerance
    peak_values = flat_projection[is_peak]
    peak_scales = scale_map[is_peak]

    # Focuses on the brightest peaks because they correspond to the most reliable feature detections. Uses partial
    # sort to select the top-k in O(n) instead of a full O(n log n) sort.
    if len(peak_values) > peak_count:
        top_indices = np.argpartition(peak_values, -peak_count)[-peak_count:]
    else:
        top_indices = np.arange(len(peak_values))

    estimated_scale, _ = mode(peak_scales[top_indices], keepdims=False)
    return int(estimated_scale)


def _compute_multiscale_masks(
    y_pixels: NDArray[np.int32],
    x_pixels: NDArray[np.int32],
    weights: NDArray[np.float32],
    scale_heights: NDArray[np.uint16],
    scale_widths: NDArray[np.uint16],
) -> tuple[list[NDArray[np.int32]], list[NDArray[np.int32]], list[NDArray[np.float32]]]:
    """Computes downsampled ROI masks at all spatial scales from a full-resolution ROI mask.

    Notes:
        Starting from the finest scale, each subsequent scale is computed by mapping pixel coordinates to the
        downsampled grid and accumulating weights. The masks at each scale are then extended into neighboring pixels
        using _extend_mask.

    Args:
        y_pixels: The y-coordinates of the ROI pixels at the finest scale.
        x_pixels: The x-coordinates of the ROI pixels at the finest scale.
        weights: The pixel weights at the finest scale.
        scale_heights: The image height at each spatial scale.
        scale_widths: The image width at each spatial scale.

    Returns:
        A tuple of three lists containing the y-coordinates, x-coordinates, and weights for the ROI mask at each
        spatial scale.
    """
    y_coordinates = [y_pixels]
    x_coordinates = [x_pixels]
    scale_weights = [weights]

    # Downsamples coordinates to each coarser scale by halving and deduplicating. Multiple fine-scale pixels that
    # map to the same coarser-grid cell have their weights accumulated via scatter-add.
    for scale_index in range(1, len(scale_heights)):
        coarse_y = y_coordinates[scale_index - 1] // 2
        coarse_x = x_coordinates[scale_index - 1] // 2
        flat_indices, inverse_mapping = np.unique(coarse_x + coarse_y * scale_widths[scale_index], return_inverse=True)
        accumulated_weights = np.zeros(len(flat_indices), dtype=np.float32)
        np.add.at(accumulated_weights, inverse_mapping, scale_weights[scale_index - 1] / 2)
        scale_weights.append(accumulated_weights)
        y_coordinates.append((flat_indices // scale_widths[scale_index]).astype(np.int32))
        x_coordinates.append((flat_indices % scale_widths[scale_index]).astype(np.int32))

    # Extends each scale's mask into neighboring pixels to ensure spatial coverage at all resolution levels.
    for scale_index in range(len(scale_heights)):
        y_coordinates[scale_index], x_coordinates[scale_index], scale_weights[scale_index] = _extend_mask(
            y_pixels=y_coordinates[scale_index],
            x_pixels=x_coordinates[scale_index],
            weights=scale_weights[scale_index],
            height=scale_heights[scale_index],
            width=scale_widths[scale_index],
        )
    return y_coordinates, x_coordinates, scale_weights


def _extend_iteratively(
    y_pixels: NDArray[np.int32],
    x_pixels: NDArray[np.int32],
    frames: NDArray[np.float32],
    height: int,
    width: int,
    active_frame_indices: NDArray[np.intp],
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float32]]:
    """Iteratively grows the ROI mask by adding neighboring pixels whose activity correlates with the ROI.

    Notes:
        At each iteration, the ROI boundary is expanded by one pixel in all cardinal directions. Pixels are retained
        if their mean activity on active frames exceeds a fraction of the peak activity. Growth terminates when
        the pixel count exceeds the maximum pixel limit or when the ROI begins shrinking.

    Args:
        y_pixels: The y-coordinates of the current ROI pixels.
        x_pixels: The x-coordinates of the current ROI pixels.
        frames: The recording data flattened to a shape (num_frames, height * width), where each row is a
            single frame and each column is one spatial pixel. This array is progressively updated as detected ROIs
            are subtracted from it by the caller.
        height: The recording frame height in pixels.
        width: The recording frame width in pixels.
        active_frame_indices: The indices of frames with above-threshold activity.

    Returns:
        A tuple of three arrays: the extended y-coordinates, x-coordinates, and unit-normalized pixel weights for
        the grown ROI.
    """
    max_pixel_count = 10000
    previous_count = 0

    # Initializes weights as a placeholder for static analysis; always overwritten since the loop runs at least once.
    weights = np.empty(y_pixels.size, dtype=np.float32)

    while previous_count < max_pixel_count:
        previous_count = y_pixels.size

        # Extends the processed ROI by 1 pixel in each direction.
        y_pixels, x_pixels = extend_roi(
            y_pixels=y_pixels,
            x_pixels=x_pixels,
            height=height,
            width=width,
            iterations=1,
        )

        # Uses mean activity on active frames as pixel weights, retaining only pixels whose contribution exceeds
        # a fraction of the peak to discard weakly correlated boundary pixels.
        flat_indices = y_pixels * width + x_pixels
        weights = frames[np.ix_(active_frame_indices, flat_indices)].mean(axis=0)
        active_mask = weights > max(0, weights.max() * _MINIMUM_WEIGHT_FRACTION)
        active_count = active_mask.sum()
        if active_count == 0:
            break
        y_pixels, x_pixels, weights = y_pixels[active_mask], x_pixels[active_mask], weights[active_mask]

        # Stops when the surviving pixel count no longer exceeds the pre-extension count, indicating the boundary
        # has reached non-correlated tissue.
        if active_count <= previous_count:
            break
        previous_count = y_pixels.size

    weights = (weights / norm(weights)).astype(np.float32)
    return y_pixels, x_pixels, weights


def _find_best_scale(
    scale_images: NDArray[np.float32],
) -> int:
    """Determines the best spatial scale for ROI detection by estimating it from the multiscale projection data.

    Notes:
        If the automatic estimation fails (returns 0), the scale defaults to 1 with a warning.

    Args:
        scale_images: The multiscale projection images with shape (num_scales, height, width).

    Returns:
        The selected spatial scale index.
    """
    scale = _estimate_spatial_scale(scale_images=scale_images)
    if scale > 0:
        return scale
    console.echo(
        message="Spatial scale estimation failed. Setting spatial scale to 1 in order to continue.",
        level=LogLevel.WARNING,
    )
    return _MINIMUM_SPATIAL_SCALE
