"""Provides the algorithm for determining the planar z-position of frames in comparison to a reference z-stack."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tqdm import tqdm
import numpy as np
from ataraxis_base_utilities import console

from ..io import BinaryFile
from .rigid import apply_edge_taper, compute_edge_taper, compute_rigid_shifts, compute_phase_correlation_kernel
from .utils import apply_spatial_high_pass, apply_spatial_smoothing

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..dataclasses import RuntimeContext


def compute_z_position(
    context: RuntimeContext,
    z_stack: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Computes z-position correlation for each frame against the input reference z-stack.

    Correlates each frame from the registered binary file against each plane of the z-stack using phase correlation.
    The resulting correlation values indicate how well each frame matches each z-stack plane, enabling reconstruction
    of the focal plane position over time.

    Notes:
        This function processes frames from a single processing plane at a time. For multi-plane recordings, call this
        function separately for each plane's RuntimeContext.

        The function reads frames in batches from the binary file and computes phase correlations against pre-computed
        reference kernels for each z-stack plane. For one-photon recordings, additional preprocessing (smoothing and
        high-pass filtering) is applied to both z-stack planes and frames before correlation.

    Args:
        context: The runtime context containing pipeline configuration and runtime data for the current plane.
        z_stack: The reference z-stack with shape (z_plane_count, height, width) constructed before starting the
            acquisition. Each plane represents a different focal depth.

    Returns:
        The z-position correlation array with shape (z_plane_count, frame_count). Each row contains the correlation
        values between the corresponding z-stack plane and every frame.

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
    batch_size = context.configuration.registration.batch_size
    maximum_shift_fraction = context.configuration.registration.maximum_shift_fraction
    spatial_smoothing_sigma = context.configuration.registration.spatial_smoothing_sigma
    temporal_smoothing_sigma = context.configuration.registration.temporal_smoothing_sigma

    # Extracts one-photon registration parameters.
    one_photon_mode = context.configuration.one_photon_registration.enabled
    pre_smoothing_sigma = context.configuration.one_photon_registration.pre_smoothing_sigma
    spatial_highpass_window = context.configuration.one_photon_registration.spatial_highpass_window
    edge_taper_pixels = context.configuration.one_photon_registration.edge_taper_pixels

    # Extracts runtime settings.
    display_progress = context.configuration.runtime.display_progress_bars
    parallel_workers = context.configuration.runtime.parallel_workers

    num_z_planes = z_stack.shape[0]

    # Uses the pre-resolved worker count for FFT parallelization in phase correlation computation.
    fft_workers = parallel_workers

    # Prepares reference kernels and edge taper masks for each z-stack plane.
    edge_taper_slope = edge_taper_pixels if one_photon_mode else 3.0 * spatial_smoothing_sigma
    reference_data: list[tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.complex64]]] = []
    for plane in z_stack:
        reference_plane = np.asarray(plane, dtype=np.float32)

        # Applies one-photon preprocessing if enabled.
        if one_photon_mode:
            reference_plane = reference_plane[np.newaxis, :, :]
            if pre_smoothing_sigma > 0:
                reference_plane = apply_spatial_smoothing(data=reference_plane, window=int(pre_smoothing_sigma))
            reference_plane = apply_spatial_high_pass(data=reference_plane, window=spatial_highpass_window)
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

    # Processes frames in batches. For each batch, correlates against every z-stack plane sequentially. The inner
    # FFT computation within compute_rigid_shifts already parallelizes across fft_workers, so adding an outer thread
    # pool would cause thread oversubscription without improving throughput.
    total_batches = (frame_count + batch_size - 1) // batch_size
    with (
        BinaryFile(height=frame_height, width=frame_width, file_path=registered_binary_path) as binary_file,
        tqdm(
            total=total_batches, desc="Correlating frames against z-stack", unit="batch", disable=not display_progress
        ) as progress_bar,
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

            # Correlates the batch against each z-stack plane.
            for plane_index, (taper_mask, mean_offset, correlation_kernel) in enumerate(reference_data):
                correlation_maxima = _compute_plane_correlation(
                    frames=frames,
                    taper_mask=taper_mask,
                    mean_offset=mean_offset,
                    correlation_kernel=correlation_kernel,
                    maximum_shift_fraction=maximum_shift_fraction,
                    temporal_smoothing_sigma=temporal_smoothing_sigma,
                    workers=fft_workers,
                )
                z_correlations[plane_index, batch_start:batch_end] = correlation_maxima

            progress_bar.update(1)

    return z_correlations


def _compute_plane_correlation(
    frames: NDArray[np.float32],
    taper_mask: NDArray[np.float32],
    mean_offset: NDArray[np.float32],
    correlation_kernel: NDArray[np.complex64],
    maximum_shift_fraction: float,
    temporal_smoothing_sigma: float,
    workers: int,
) -> NDArray[np.float32]:
    """Computes phase correlation between frames and a single z-stack plane.

    Applies edge tapering to the input frames and computes the phase correlation against a pre-computed reference
    kernel using parallelized FFT operations.

    Args:
        frames: The batch of frames with shape (num_frames, height, width) to correlate.
        taper_mask: The edge taper mask with shape (height, width) for this z-plane.
        mean_offset: The mean intensity offset with shape (height, width) for this z-plane.
        correlation_kernel: The pre-computed phase correlation kernel for this z-plane.
        maximum_shift_fraction: The maximum allowed shift as a fraction of the minimum frame dimension.
        temporal_smoothing_sigma: The standard deviation in frames for temporal smoothing of correlation values.
        workers: The number of parallel workers for FFT computation.

    Returns:
        A 1D array of peak correlation values for each frame.
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
        workers=workers,
    )
    return correlation_maxima
