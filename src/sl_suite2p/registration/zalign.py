"""Provides functions for determining the planar z-position of frames in comparison to a reference z-stack."""

from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path  # noqa: TC003 - Path is used at runtime for .exists() and .open()
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from ataraxis_base_utilities import console

from ..io import BinaryFile
from .rigid import apply_edge_taper, compute_edge_taper, compute_rigid_shifts, compute_phase_correlation_kernel
from .utils import apply_spatial_high_pass, apply_spatial_smoothing

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _compute_plane_correlation(
    plane_index: int,
    frames: NDArray[np.float32],
    taper_mask: NDArray[np.float32],
    mean_offset: NDArray[np.float32],
    correlation_kernel: NDArray[np.complex64],
    maximum_shift_fraction: float,
    temporal_smoothing_sigma: float,
) -> tuple[int, NDArray[np.float32]]:
    """Computes phase correlation between frames and a single z-stack plane.

    This helper function is designed for parallel execution across multiple z-planes. It applies edge tapering to the
    input frames and computes the phase correlation against a pre-computed reference kernel.

    Args:
        plane_index: The index of the z-stack plane being processed, returned unchanged for result ordering.
        frames: The batch of frames with shape (num_frames, height, width) to correlate.
        taper_mask: The edge taper mask with shape (height, width) for this z-plane.
        mean_offset: The mean intensity offset with shape (height, width) for this z-plane.
        correlation_kernel: The pre-computed phase correlation kernel for this z-plane.
        maximum_shift_fraction: The maximum allowed shift as a fraction of the minimum frame dimension.
        temporal_smoothing_sigma: The standard deviation in frames for temporal smoothing of correlation values.

    Returns:
        A tuple of (plane_index, correlation_maxima) where correlation_maxima is a 1D array of peak correlation
        values for each frame.
    """
    tapered_frames = apply_edge_taper(
        frames=frames,
        taper_mask=taper_mask,
        mean_offset=mean_offset,
    )
    _, _, correlation_maxima = compute_rigid_shifts(
        frames=tapered_frames,
        reference_kernel=correlation_kernel,
        maximum_shift_fraction=maximum_shift_fraction,
        temporal_smoothing_sigma=temporal_smoothing_sigma,
    )
    return plane_index, correlation_maxima


def compute_z_position(
    z_stack: NDArray[np.float32],
    registered_binary_path: Path,
    frame_height: int,
    frame_width: int,
    frame_count: int,
    batch_size: int = 100,
    one_photon_mode: bool = False,
    pre_smoothing_sigma: float = 0.0,
    spatial_highpass_window: int = 42,
    edge_taper_slope: float = 3.45,
    spatial_smoothing_sigma: float = 1.15,
    maximum_shift_fraction: float = 0.1,
    temporal_smoothing_sigma: float = 0.0,
) -> NDArray[np.float32]:
    """Computes z-position correlation for each frame against a reference z-stack.

    Correlates each frame from the registered binary file against each plane of the z-stack using phase
    correlation. The resulting correlation values indicate how well each frame matches each z-stack plane,
    enabling reconstruction of the focal plane position over time.

    Notes:
        The function reads frames in batches from the binary file and computes phase correlations against
        pre-computed reference kernels for each z-stack plane. For one-photon recordings, additional
        preprocessing (smoothing and high-pass filtering) is applied to both z-stack planes and frames
        before correlation.

    Args:
        z_stack: The reference z-stack with shape (planes, height, width). Each plane represents a
            different focal depth.
        registered_binary_path: The absolute path to the motion-corrected binary file that stores the recording frames.
        frame_height: The height of each frame in pixels.
        frame_width: The width of each frame in pixels.
        frame_count: The total number of frames in the binary file.
        batch_size: The number of frames to load and process at once.
        one_photon_mode: Determines whether to apply one-photon preprocessing (high-pass filtering and
            pre-smoothing) to the data.
        pre_smoothing_sigma: The standard deviation in pixels for Gaussian smoothing applied before
            high-pass filtering in one-photon mode. Only used when one_photon_mode is True.
        spatial_highpass_window: The window size in pixels for spatial high-pass filtering in one-photon
            mode. Only used when one_photon_mode is True.
        edge_taper_slope: Controls the steepness of the edge taper applied before phase correlation.
            Larger values produce a more gradual taper. For two-photon data, typically 3x the spatial
            smoothing sigma.
        spatial_smoothing_sigma: The standard deviation in pixels for Gaussian smoothing of the phase
            correlation surface.
        maximum_shift_fraction: The maximum allowed shift as a fraction of the minimum frame dimension.
        temporal_smoothing_sigma: The standard deviation in frames for temporal smoothing of correlation
            values.

    Returns:
        A 2D array of shape (num_planes, frame_count) containing the peak phase correlation value for each
        frame against each z-stack plane. Higher values indicate better alignment with that z-plane.

    Raises:
        FileNotFoundError: If the registered binary file does not exist at the specified path.
    """
    if not registered_binary_path.exists():
        message = (
            f"Unable to compute z-position alignment. The registered binary file does not exist at the "
            f"specified path: {registered_binary_path}."
        )
        console.error(message=message, error=FileNotFoundError)

    num_planes = z_stack.shape[0]

    # Prepares reference kernels and edge taper masks for each z-stack plane.
    reference_data: list[tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.complex64]]] = []
    for plane in z_stack:
        # Ensures float32 dtype without copying if already float32.
        reference_plane = np.asarray(plane, dtype=np.float32)

        # Applies one-photon preprocessing if enabled.
        if one_photon_mode:
            reference_plane = reference_plane[np.newaxis, :, :]
            if pre_smoothing_sigma > 0:
                reference_plane = apply_spatial_smoothing(
                    data=reference_plane,
                    window=int(pre_smoothing_sigma),
                )
            reference_plane = apply_spatial_high_pass(
                data=reference_plane,
                window=spatial_highpass_window,
            )
            reference_plane = reference_plane.squeeze()

        taper_mask, mean_offset = compute_edge_taper(
            reference_image=reference_plane,
            taper_slope=edge_taper_slope,
        )
        correlation_kernel = compute_phase_correlation_kernel(
            reference_image=reference_plane,
            smoothing_sigma=spatial_smoothing_sigma,
        )
        reference_data.append((taper_mask, mean_offset, correlation_kernel))

    # Allocates output array for correlation values.
    z_correlations = np.zeros((num_planes, frame_count), dtype=np.float32)

    # Processes frames in batches using memory-mapped BinaryFile access.
    total_batches = (frame_count + batch_size - 1) // batch_size

    # Pre-builds partial functions for each plane with plane-specific data bound. Only the frames argument varies
    # per batch and gets bound later. Thread-based parallelism is effective because NumPy FFT releases the GIL.
    plane_processors = [
        partial(
            _compute_plane_correlation,
            plane_index=plane_index,
            taper_mask=taper_mask,
            mean_offset=mean_offset,
            correlation_kernel=correlation_kernel,
            maximum_shift_fraction=maximum_shift_fraction,
            temporal_smoothing_sigma=temporal_smoothing_sigma,
        )
        for plane_index, (taper_mask, mean_offset, correlation_kernel) in enumerate(reference_data)
    ]

    # Reuses a single executor across all batches to avoid repeated thread pool creation overhead.
    with (
        BinaryFile(height=frame_height, width=frame_width, file_path=registered_binary_path) as binary_file,
        ThreadPoolExecutor(max_workers=min(num_planes, 8)) as executor,
    ):
        for batch_index in range(total_batches):
            batch_start = batch_index * batch_size
            batch_end = min(batch_start + batch_size, frame_count)

            # Reads batch of frames from memory-mapped file and converts to float32.
            frames = binary_file[batch_start:batch_end].astype(np.float32)

            # Applies one-photon preprocessing once per batch, before the plane loop.
            if one_photon_mode:
                if pre_smoothing_sigma > 0:
                    frames = apply_spatial_smoothing(data=frames, window=int(pre_smoothing_sigma))
                frames = apply_spatial_high_pass(data=frames, window=spatial_highpass_window)

            # Binds the current batch frames to each plane processor and executes in parallel.
            batch_tasks = [partial(processor, frames=frames) for processor in plane_processors]
            for plane_index, correlation_maxima in executor.map(lambda task: task(), batch_tasks):
                z_correlations[plane_index, batch_start:batch_end] = correlation_maxima

            console.echo(
                message=f"Z-position: batch {batch_index + 1}/{total_batches}, frames {batch_end}/{frame_count}."
            )

    return z_correlations
