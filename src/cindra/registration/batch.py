"""Provides the per-batch input and output contracts the CPU and GPU registration backends share."""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

type RegistrationBlocks = tuple[
    list[NDArray[np.int32]], list[NDArray[np.int32]], tuple[int, int], tuple[int, int], NDArray[np.float32]
]
"""The registration block structure returned by compute_registration_blocks. Contains y_blocks, x_blocks,
block_counts, actual_block_size, and smoothing_kernel."""


@dataclass(frozen=True, slots=True)
class ReferenceData:
    """Stores precomputed reference data for phase correlation registration."""

    taper_mask: NDArray[np.float32]
    """The edge taper mask with shape (height, width) for rigid registration."""
    mean_offset: NDArray[np.float32]
    """The mean intensity offset with shape (height, width) for rigid registration."""
    reference_kernel: NDArray[np.complex64]
    """The phase correlation kernel with shape (fft_height, fft_width) for rigid registration."""
    taper_mask_nonrigid: NDArray[np.float32] | None
    """Per-block taper masks with shape (num_blocks, block_height, block_width), or None if nonrigid is disabled."""
    mean_offset_nonrigid: NDArray[np.float32] | None
    """Per-block mean offsets with shape (num_blocks, block_height, block_width), or None if nonrigid is disabled."""
    reference_kernel_nonrigid: NDArray[np.complex64] | None
    """Per-block FFT kernels with shape (num_blocks, block_height, rfft_width), or None if nonrigid is disabled."""
    blocks: RegistrationBlocks | None
    """The registration block structure from compute_registration_blocks, or None if nonrigid is disabled."""


@dataclass(frozen=True, slots=True)
class BatchRegistrationResult:
    """Stores the output from registering a single batch of frames."""

    frames: NDArray[np.int16] | NDArray[np.float32]
    """The registered frames with shape (batch_size, height, width), in the width the backend was handed.

    Notes:
        A backend handed the width the plane binary stores returns that same width, having clipped and narrowed the
        frames itself, and reports the mean image contribution through 'frame_sum'. A backend handed float32 frames
        returns float32 and leaves that narrowing to the caller.
    """
    y_offsets: NDArray[np.int32]
    """The y-direction rigid pixel offsets with shape (batch_size,)."""
    x_offsets: NDArray[np.int32]
    """The x-direction rigid pixel offsets with shape (batch_size,)."""
    correlations: NDArray[np.float32]
    """The phase correlation peak values with shape (batch_size,)."""
    y_offsets_nonrigid: NDArray[np.float32] | None
    """The y-direction nonrigid subpixel offsets with shape (batch_size, num_blocks), or None."""
    x_offsets_nonrigid: NDArray[np.float32] | None
    """The x-direction nonrigid subpixel offsets with shape (batch_size, num_blocks), or None."""
    correlations_nonrigid: NDArray[np.float32] | None
    """The nonrigid correlation values with shape (batch_size, num_blocks), or None."""
    frame_sum: NDArray[np.float32] | None = None
    """The per-pixel sum over the batch with shape (height, width), or None when the caller sums the frames itself.

    Notes:
        The mean image is accumulated from the registered frames before they are narrowed to the storage dtype, so a
        backend that returns already-narrowed frames carries the sum it measured ahead of that narrowing. A backend
        returning float32 frames leaves this None, because the caller reduces those frames directly.
    """
