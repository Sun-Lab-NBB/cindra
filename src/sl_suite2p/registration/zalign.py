"""Provides functions for determining the planar z-position of frames in comparison to a reference z-stack."""

from __future__ import annotations

from typing import TYPE_CHECKING
from functools import partial
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
import numpy as np
from ataraxis_base_utilities import console

from ..io import BinaryFile
from .rigid import apply_edge_taper, compute_edge_taper, compute_rigid_shifts, compute_phase_correlation_kernel
from .utils import apply_spatial_high_pass, apply_spatial_smoothing

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..configuration import RuntimeContext


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
    context: RuntimeContext,
    z_stack: NDArray[np.float32],
) -> RuntimeContext:
    """Computes z-position correlation for each frame against a reference z-stack.

    Correlates each frame from the registered binary file against each plane of the z-stack using phase correlation.
    The resulting correlation values indicate how well each frame matches each z-stack plane, enabling reconstruction
    of the focal plane position over time.

    Notes:
        This function processes frames from a single processing plane at a time. For multi-plane recordings, call this
        function separately for each plane's RuntimeContext.

        The function reads frames in batches from the binary file and computes phase correlations against pre-computed
        reference kernels for each z-stack plane. For one-photon recordings, additional preprocessing (smoothing and
        high-pass filtering) is applied to both z-stack planes and frames before correlation.

        The computed correlations are stored in context.runtime.registration.z_stack_correlations.

    Args:
        context: The runtime context containing pipeline configuration and runtime data for the current plane.
        z_stack: The reference z-stack with shape (num_z_planes, height, width) constructed before starting the
            acquisition. Each plane represents a different focal depth.

    Returns:
        The updated runtime context with z_stack_correlations populated in the registration data.

    Raises:
        FileNotFoundError: If the registered binary file does not exist at the specified path.
        ValueError: If the registered binary path is not set in the runtime context.
    """
    # Extracts IO parameters from runtime context.
    registered_binary_path = context.runtime.io.registered_binary_path
    if registered_binary_path is None:
        message = (
            "Unable to compute z-position alignment. The registered binary path is not set in the runtime context."
        )
        console.error(message=message, error=ValueError)

    if not registered_binary_path.exists():
        message = (
            f"Unable to compute z-position alignment. The registered binary file does not exist at the "
            f"specified path: {registered_binary_path}."
        )
        console.error(message=message, error=FileNotFoundError)

    frame_height = context.runtime.io.frame_height
    frame_width = context.runtime.io.frame_width
    frame_count = context.runtime.io.frame_count

    # Extracts configuration parameters.
    batch_size = context.config.registration.batch_size
    maximum_shift_fraction = context.config.registration.maximum_shift_fraction
    spatial_smoothing_sigma = context.config.registration.spatial_smoothing_sigma
    temporal_smoothing_sigma = context.config.registration.temporal_smoothing_sigma

    # Extracts one-photon registration parameters.
    one_photon_mode = context.config.one_photon_registration.enabled
    pre_smoothing_sigma = context.config.one_photon_registration.pre_smoothing_sigma
    spatial_highpass_window = context.config.one_photon_registration.spatial_highpass_window
    edge_taper_pixels = context.config.one_photon_registration.edge_taper_pixels

    # Computes edge taper slope based on imaging mode.
    edge_taper_slope = edge_taper_pixels if one_photon_mode else 3.0 * spatial_smoothing_sigma

    # Extracts main configuration parameters.
    display_progress = context.config.main.display_progress_bars
    parallel_workers = context.config.main.parallel_workers

    num_z_planes = z_stack.shape[0]

    # Determines the number of workers for parallel processing.
    if parallel_workers <= 0:
        max_workers = min(num_z_planes, 8)
    else:
        max_workers = min(num_z_planes, parallel_workers, 8)

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
    z_correlations = np.zeros((num_z_planes, frame_count), dtype=np.float32)

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
        ThreadPoolExecutor(max_workers=max_workers) as executor,
        tqdm(
            total=total_batches, desc="Computing frame z-positions", unit="batch", disable=not display_progress
        ) as pbar,
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

            pbar.update(1)

    # Stores the computed correlations in the runtime context.
    context.runtime.registration.z_stack_correlations = z_correlations

    return context
