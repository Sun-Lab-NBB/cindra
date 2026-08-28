"""Provides the CuPy backend that registers single-recording frame batches on a CUDA device."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast
from dataclasses import field, dataclass

import numpy as np
from ataraxis_base_utilities import console

from .batch import BatchRegistrationResult
from .utils import NORMALIZATION_EPSILON, compute_upsampling_kernel

try:
    import cupy
    from cupyx.scipy.ndimage import gaussian_filter1d
except ImportError:  # pragma: no cover
    # The CuPy distribution publishes no macOS wheel, so the dependency marker excludes darwin and this module has to
    # stay importable there. Guarding the import also covers a Linux or Windows host whose installation was trimmed.
    # Every path that reaches a device runs behind the _require_gpu_runtime() call the backend constructor makes, so
    # the names below are read only once the import above has succeeded. Only one of the two branches runs on any
    # single host, so the fallback stays out of coverage measurement.
    cupy = None
    gaussian_filter1d = None

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from numpy.typing import NDArray

    from .batch import ReferenceData

_GPU_REMEDY: str = "Run 'cindra gpu' to report the local CUDA devices and install the runtime libraries they need."
"""The command that resolves an unusable GPU runtime, named by every message this module produces.

Notes:
    The orchestration package states the same command, and this module restates it rather than importing it, because
    the dependency chain runs from orchestration into registration and never back.
"""

_TF32_VARIABLE: str = "CUPY_TF32"
"""The environment variable CuPy reads to decide whether its cuBLAS handle allows TF32 matrix multiplication."""

_MINIMUM_CORRELATION_RADIUS: int = 1
"""The smallest rigid correlation search radius the quadrant rearrangement can express."""

_SIGNAL_TO_NOISE_EPSILON: float = 1e-10
"""The small epsilon value used to prevent division by zero in signal-to-noise ratio calculations."""

_SUBPIXEL_FACTOR: int = 10
"""The upsampling factor for Gaussian RBF subpixel peak localization. A value of 10 provides 0.1 pixel precision."""

_UPSAMPLING_PADDING: int = 3
"""The half-width of the region around integer peaks used for RBF upsampling. A value of 3 uses a 7x7 region."""

_CORRELATION_BATCH_SIZE: int = 64
"""The maximum number of blocks transformed in a single nonrigid phase correlation call. Limits device memory use."""

_PIPELINE_DEPTH: int = 2
"""The number of staging slots the backend cycles through, which is one slot for the batch the device is working on
and one for the batch being staged behind it.
"""

_INT16_MINIMUM: int = int(np.iinfo(np.int16).min)
"""The lowest value the plane binary storage dtype represents, which is the lower clip bound of a narrowed batch."""

_INT16_MAXIMUM: int = int(np.iinfo(np.int16).max)
"""The highest value the plane binary storage dtype represents, which is the upper clip bound of a narrowed batch."""


@dataclass(slots=True)
class _StagingSlot:
    """Stores the reusable buffers one pipeline slot stages a frame batch through.

    Notes:
        A slot serving the download direction asks for its host buffer alone, because the device side of that
        direction is the array the registration itself produced.
    """

    host_buffers: dict[tuple[tuple[int, ...], str], NDArray[np.int16] | NDArray[np.float32]] = field(
        default_factory=dict
    )
    """The page-locked host buffers the slot holds, keyed by the shape and the dtype of the batch each one stages."""
    device_buffers: dict[tuple[tuple[int, ...], str], cupy.ndarray] = field(default_factory=dict)
    """The device buffers the slot uploads into, keyed by the shape and the dtype of the batch each one stages."""


@dataclass(frozen=True, slots=True)
class _DeviceBatchResult:
    """Stores the device arrays one registered batch produces, before they are narrowed and copied to the host."""

    frames: cupy.ndarray
    """The registered frames with shape (batch_size, height, width), on the float32 arithmetic scale."""
    y_offsets: cupy.ndarray
    """The y-direction rigid pixel offsets with shape (batch_size,)."""
    x_offsets: cupy.ndarray
    """The x-direction rigid pixel offsets with shape (batch_size,)."""
    correlations: cupy.ndarray
    """The phase correlation peak values with shape (batch_size,)."""
    y_offsets_nonrigid: cupy.ndarray | None
    """The y-direction nonrigid subpixel offsets with shape (batch_size, block_count), or None."""
    x_offsets_nonrigid: cupy.ndarray | None
    """The x-direction nonrigid subpixel offsets with shape (batch_size, block_count), or None."""
    correlations_nonrigid: cupy.ndarray | None
    """The nonrigid correlation values with shape (batch_size, block_count), or None."""


@dataclass(frozen=True, slots=True)
class _NonrigidDeviceData:
    """Stores the device copies of the block geometry and per-block reference data nonrigid registration reads."""

    taper_mask: cupy.ndarray
    """The per-block edge taper mask with shape (block_count, block_height, block_width)."""
    mean_offset: cupy.ndarray
    """The per-block mean intensity offset with shape (block_count, block_height, block_width)."""
    reference_kernel: cupy.ndarray
    """The per-block phase correlation kernel with shape (block_count, block_height, real_fft_width)."""
    block_rows: cupy.ndarray
    """The source row index of every block pixel, with shape (block_count, block_height)."""
    block_columns: cupy.ndarray
    """The source column index of every block pixel, with shape (block_count, block_width)."""
    smoothing_kernel: cupy.ndarray
    """The block smoothing kernel with shape (block_count, block_count) the SNR loop applies."""
    upsampling_kernel: cupy.ndarray
    """The Gaussian RBF upsampling matrix with shape (region_size squared, upsampled_size squared)."""
    upsampled_size: int
    """The side length of the upsampled correlation surface the subpixel peak is located on."""
    block_counts: tuple[int, int]
    """The number of blocks along the y and x axes."""
    block_row_grid: cupy.ndarray
    """The block-space row coordinate of every frame pixel, with shape (height, width)."""
    block_column_grid: cupy.ndarray
    """The block-space column coordinate of every frame pixel, with shape (height, width)."""
    pixel_row_grid: cupy.ndarray
    """The row index of every frame pixel, with shape (height, width)."""
    pixel_column_grid: cupy.ndarray
    """The column index of every frame pixel, with shape (height, width)."""


class GpuRegistrationBackend:
    """Registers single-recording frame batches on one CUDA device through the CuPy runtime.

    The reference data reaches the device once, while the backend is constructed, and stays there for the lifetime of
    the instance. Every entry point accepts host arrays and returns host arrays, so a batch crosses the bus exactly
    once on the way in and once on the way out. The device algorithms reproduce the arithmetic of the host
    implementation, so the two resolve the same rigid pixel offsets and the same quantized nonrigid offsets from the
    same frames.

    Notes:
        The frames the nonrigid warp writes agree to single-precision rounding rather than exactly. The host bilinear
        kernel subtracts an integer neighbor index from a float32 coordinate, which Numba promotes to double precision,
        so the host accumulates its four-term weighted sum in double precision while the device accumulates in single
        precision. A sample the two backends disagree on sits one storage unit from the value the host writes, and never
        further. The rigid path carries no interpolation, so its frames match the host exactly.

        A batch crosses the bus in the dtype the caller supplies it in. An int16 batch is widened to float32 on the
        device and narrowed back to int16 there. Both directions stage through page-locked host buffers the backend
        allocates on first use and reuses for every batch of the same geometry that follows.

    Args:
        reference_data: The precomputed reference data holding the rigid taper mask, mean offset, and FFT kernel,
            together with the per-block nonrigid equivalents and the block geometry.
        device: The zero-based index of the CUDA device every operation runs on.

    Attributes:
        _device: Cached index of the CUDA device every operation runs on.
        _frame_height: Cached frame height, in pixels, read from the rigid taper mask.
        _frame_width: Cached frame width, in pixels, read from the rigid taper mask.
        _taper_mask: Device copy of the rigid edge taper mask.
        _mean_offset: Device copy of the rigid mean intensity offset.
        _reference_kernel: Device copy of the rigid phase correlation kernel.
        _nonrigid_data: Device copies of the nonrigid reference data and block geometry, None when the reference data
            carries no blocks.
        _normalization_weights: Device high-pass normalization weights, keyed by the smoothing window they correct.
        _input_slots: Staging slots the upload direction cycles through, one per pipeline slot.
        _output_slots: Staging slots the download direction cycles through, one per pipeline slot.
        _upload_events: One CUDA event per pipeline slot, recorded when that slot's upload finishes.
        _staging_slot: Index of the pipeline slot the next batch stages through, advanced by every entry point.
        _compute_stream: The CUDA stream every registration kernel and every download runs on.
        _transfer_stream: The CUDA stream every upload runs on, which is what lets an upload overlap a computation.

    Raises:
        RuntimeError: If the CuPy distribution is absent, or if the device allows TF32 matrix multiplication.
    """

    def __init__(self, reference_data: ReferenceData, device: int) -> None:
        _require_gpu_runtime()

        self._device: int = device
        self._frame_height: int = int(reference_data.taper_mask.shape[0])
        self._frame_width: int = int(reference_data.taper_mask.shape[1])
        self._normalization_weights: dict[int, cupy.ndarray] = {}
        self._input_slots: tuple[_StagingSlot, ...] = tuple(_StagingSlot() for _ in range(_PIPELINE_DEPTH))
        self._output_slots: tuple[_StagingSlot, ...] = tuple(_StagingSlot() for _ in range(_PIPELINE_DEPTH))
        self._staging_slot: int = 0

        with cupy.cuda.Device(self._device):
            _verify_tf32_disabled()
            self._compute_stream: cupy.cuda.Stream = cupy.cuda.Stream(non_blocking=True)
            self._transfer_stream: cupy.cuda.Stream = cupy.cuda.Stream(non_blocking=True)
            self._upload_events: tuple[cupy.cuda.Event, ...] = tuple(cupy.cuda.Event() for _ in range(_PIPELINE_DEPTH))
            self._taper_mask: cupy.ndarray = cupy.asarray(a=reference_data.taper_mask, dtype=cupy.float32)
            self._mean_offset: cupy.ndarray = cupy.asarray(a=reference_data.mean_offset, dtype=cupy.float32)
            self._reference_kernel: cupy.ndarray = cupy.asarray(a=reference_data.reference_kernel, dtype=cupy.complex64)
            self._nonrigid_data: _NonrigidDeviceData | None = _upload_nonrigid_data(
                reference_data=reference_data,
                frame_height=self._frame_height,
                frame_width=self._frame_width,
            )

    def __repr__(self) -> str:
        """Returns a string representation of the GpuRegistrationBackend instance."""
        return (
            f"GpuRegistrationBackend(device={self._device}, frame_height={self._frame_height}, "
            f"frame_width={self._frame_width}, nonrigid={self._nonrigid_data is not None})"
        )

    def register_batches(
        self,
        batches: Iterable[NDArray[np.int16] | NDArray[np.float32]],
        normalization_minimum: float,
        normalization_maximum: float,
        bidirectional_phase_offset: int,
        pre_smoothing_sigma: float,
        spatial_highpass_window: int,
        temporal_smoothing_sigma: float,
        maximum_offset_fraction: float,
        signal_to_noise_threshold: float,
        maximum_block_offset: float,
        *,
        one_photon_enabled: bool,
        nonrigid_enabled: bool,
    ) -> Iterator[BatchRegistrationResult]:
        """Registers a stream of frame batches, overlapping the staging of each batch with the work on the one ahead.

        The scalar parameters describe the whole registration pass rather than one batch, so they are supplied once
        here instead of once per batch. Every batch of the stream carries the same frame geometry and the same dtype.

        Notes:
            The generator holds one batch ahead of the batch the device is working on. It pulls that batch from the
            iterator and starts its upload while the current batch's kernels are still queued. The caller's read of the
            next batch and the bus transfer that follows it therefore overlap the computation.

            The frames of a yielded result alias a page-locked buffer the backend reuses, and the backend cycles
            through two such buffers, so a result stays readable until the second result after it is produced. A
            caller that keeps more than the current batch and the one before it copies the frames it means to keep.

        Args:
            batches: The frame batches to register, each with shape (batch_size, height, width) and a dtype of either
                int16 or float32.
            normalization_minimum: The minimum intensity value for clipping frames before correlation.
            normalization_maximum: The maximum intensity value for clipping frames before correlation.
            bidirectional_phase_offset: The pixel offset to correct bidirectional scanning artifacts.
            pre_smoothing_sigma: The sliding-window (box) smoothing size, in pixels, applied before high-pass
                filtering. Cast to an integer, which must be a positive even number.
            spatial_highpass_window: The window size for the spatial high-pass filter that removes low-frequency
                background.
            temporal_smoothing_sigma: The standard deviation for temporal Gaussian smoothing of correlation maps.
                If 0, no smoothing is applied.
            maximum_offset_fraction: The maximum allowed offset as a fraction of the minimum spatial dimension.
                The search window is limited to min(height, width) * maximum_offset_fraction pixels.
            signal_to_noise_threshold: The SNR threshold below which additional smoothing is applied to correlation
                peaks. Higher values apply more smoothing. Typical values range from 1.0 to 1.5.
            maximum_block_offset: The maximum allowed offset for nonrigid blocks in pixels.
            one_photon_enabled: Determines whether to apply one-photon preprocessing, which includes spatial smoothing
                followed by high-pass filtering.
            nonrigid_enabled: Determines whether to apply nonrigid (piecewise) registration after rigid alignment.

        Yields:
            One result per input batch, holding the registered frames in the dtype of that batch together with the
            per-frame rigid offsets and phase correlation peaks. The nonrigid offsets and correlations are None when
            nonrigid registration is disabled. A result carrying int16 frames also carries the per-pixel sum the batch
            contributes to the mean image, measured before the frames were clipped and narrowed.

        Raises:
            ValueError: If a batch carries a dtype other than int16 or float32, if nonrigid registration is enabled and
                the backend holds no block reference data, or if one-photon preprocessing resolves a smoothing window
                that is not a positive even integer.
        """
        batch_iterator = iter(batches)
        slot = self._staging_slot

        with cupy.cuda.Device(self._device):
            host_batch = next(batch_iterator, None)
            staged_frames = None if host_batch is None else self._upload_batch(frames=host_batch, slot=slot)

        while staged_frames is not None:
            next_slot = (slot + 1) % _PIPELINE_DEPTH
            with cupy.cuda.Device(self._device):
                narrow_frames = staged_frames.dtype == cupy.int16
                self._compute_stream.wait_event(event=self._upload_events[slot])
                with self._compute_stream:
                    device_result = self._register_device_batch(
                        staged_frames=staged_frames,
                        normalization_minimum=normalization_minimum,
                        normalization_maximum=normalization_maximum,
                        bidirectional_phase_offset=bidirectional_phase_offset,
                        pre_smoothing_sigma=pre_smoothing_sigma,
                        spatial_highpass_window=spatial_highpass_window,
                        temporal_smoothing_sigma=temporal_smoothing_sigma,
                        maximum_offset_fraction=maximum_offset_fraction,
                        signal_to_noise_threshold=signal_to_noise_threshold,
                        maximum_block_offset=maximum_block_offset,
                        one_photon_enabled=one_photon_enabled,
                        nonrigid_enabled=nonrigid_enabled,
                    )

                # Stages the next batch while the kernels above are still queued, which is what puts the caller's read
                # and the upload that follows it alongside the computation. The slot it stages into last served a
                # batch whose computation the synchronization below already awaited, so its buffers are free.
                host_batch = next(batch_iterator, None)
                upcoming = None if host_batch is None else self._upload_batch(frames=host_batch, slot=next_slot)

                with self._compute_stream:
                    result = self._download_batch_result(
                        device_result=device_result, slot=slot, narrow_frames=narrow_frames
                    )
                self._compute_stream.synchronize()

            # Hands the cycle on before the result leaves, so that the batch a later call stages first lands in the
            # slot this iteration staged ahead into rather than in the slot the result about to be yielded holds.
            self._staging_slot = next_slot

            yield result
            staged_frames = upcoming
            slot = next_slot

    def apply_precomputed_offsets(
        self,
        frames: NDArray[np.int16] | NDArray[np.float32],
        y_offsets: NDArray[np.int32],
        x_offsets: NDArray[np.int32],
        y_offsets_nonrigid: NDArray[np.float32] | None,
        x_offsets_nonrigid: NDArray[np.float32] | None,
        bidirectional_phase_offset: int,
        *,
        bidirectional_phase_corrected: bool,
        nonrigid_enabled: bool,
    ) -> tuple[NDArray[np.int16] | NDArray[np.float32], NDArray[np.float32] | None]:
        """Applies precomputed registration offsets to a batch of frames.

        Registers the second channel of a plane with the offsets computed from the first channel, which avoids a
        redundant offset computation.

        Notes:
            The returned frames alias a page-locked buffer the backend reuses, and the backend cycles through two
            such buffers, so a batch stays readable until the second batch after it is produced. A caller that keeps
            more than the current batch and the one before it copies the frames it means to keep.

        Args:
            frames: The batch of frames with shape (batch_size, height, width), with a dtype of either int16 or
                float32.
            y_offsets: The y-direction rigid pixel offsets with shape (batch_size,).
            x_offsets: The x-direction rigid pixel offsets with shape (batch_size,).
            y_offsets_nonrigid: The y-direction nonrigid subpixel offsets with shape (batch_size, num_blocks), or None.
            x_offsets_nonrigid: The x-direction nonrigid subpixel offsets with shape (batch_size, num_blocks), or None.
            bidirectional_phase_offset: The pixel offset to correct bidirectional scanning artifacts.
            bidirectional_phase_corrected: Determines whether bidirectional correction was already applied to the
                input frames.
            nonrigid_enabled: Determines whether to apply nonrigid (piecewise) registration after rigid alignment.

        Returns:
            A tuple of the shifted frames with shape (batch_size, height, width), in the dtype of the input batch, and
            the per-pixel sum the batch contributes to the mean image. That sum is measured before the frames are
            clipped and narrowed, and it is None when the input batch was float32.

        Raises:
            ValueError: If the batch carries a dtype other than int16 or float32, or if nonrigid registration is
                enabled and no nonrigid block offsets are supplied.
        """
        slot = self._staging_slot
        self._staging_slot = (slot + 1) % _PIPELINE_DEPTH

        with cupy.cuda.Device(self._device):
            staged_frames = self._upload_batch(frames=frames, slot=slot)
            narrow_frames = staged_frames.dtype == cupy.int16
            self._compute_stream.wait_event(event=self._upload_events[slot])

            with self._compute_stream:
                device_frames = staged_frames.astype(cupy.float32) if narrow_frames else staged_frames

                if bidirectional_phase_offset != 0 and not bidirectional_phase_corrected:
                    self._apply_bidirectional_phase_correction(
                        frames=device_frames, bidirectional_phase_offset=bidirectional_phase_offset
                    )

                device_frames = self._translate_frames(
                    frames=device_frames,
                    y_offsets=cupy.asarray(a=y_offsets, dtype=cupy.int32),
                    x_offsets=cupy.asarray(a=x_offsets, dtype=cupy.int32),
                )

                if nonrigid_enabled:
                    if y_offsets_nonrigid is None or x_offsets_nonrigid is None:
                        message = (
                            f"Unable to apply precomputed registration offsets on the GPU backend for device "
                            f"{self._device}. Nonrigid registration is enabled, but the caller supplied no nonrigid "
                            f"block offsets."
                        )
                        console.error(message=message, error=ValueError)

                    device_frames = self._apply_nonrigid_correction(
                        nonrigid_data=self._resolve_nonrigid_data(),
                        frames=device_frames,
                        y_block_offsets=cupy.asarray(a=y_offsets_nonrigid, dtype=cupy.float32),
                        x_block_offsets=cupy.asarray(a=x_offsets_nonrigid, dtype=cupy.float32),
                    )

                registered_frames, frame_sum = self._download_frames(
                    device_frames=device_frames, slot=slot, narrow_frames=narrow_frames
                )
            self._compute_stream.synchronize()

        return registered_frames, frame_sum

    def _upload_batch(self, frames: NDArray[np.int16] | NDArray[np.float32], slot: int) -> cupy.ndarray:
        """Stages one host batch in the slot's page-locked buffer and starts its upload on the transfer stream.

        Notes:
            The copy into the page-locked buffer is what makes the device copy that follows it asynchronous, so the
            upload proceeds while the caller queues further work. The event the slot carries is recorded behind that
            upload, and the computation reading the batch waits on it.

        Args:
            frames: The batch to stage, with a dtype of either int16 or float32.
            slot: The index of the pipeline slot the batch is staged through.

        Returns:
            The device buffer the batch is being uploaded into, which carries the dtype of the input batch.

        Raises:
            ValueError: If the batch carries a dtype other than int16 or float32.
        """
        dtype = frames.dtype
        if dtype not in (np.int16, np.float32):
            message = (
                f"Unable to stage a frame batch for the GPU registration backend on device {self._device}. The batch "
                f"dtype must be int16 or float32, but got {dtype}."
            )
            console.error(message=message, error=ValueError)

        shape = tuple(int(axis) for axis in frames.shape)
        host_buffer = _resolve_pinned_buffer(slot=self._input_slots[slot], shape=shape, dtype=dtype)
        np.copyto(dst=host_buffer, src=frames)

        device_buffer = _resolve_device_buffer(slot=self._input_slots[slot], shape=shape, dtype=dtype)
        device_buffer.set(arr=host_buffer, stream=self._transfer_stream)
        self._transfer_stream.record(event=self._upload_events[slot])
        return device_buffer

    def _register_device_batch(
        self,
        staged_frames: cupy.ndarray,
        normalization_minimum: float,
        normalization_maximum: float,
        bidirectional_phase_offset: int,
        pre_smoothing_sigma: float,
        spatial_highpass_window: int,
        temporal_smoothing_sigma: float,
        maximum_offset_fraction: float,
        signal_to_noise_threshold: float,
        maximum_block_offset: float,
        *,
        one_photon_enabled: bool,
        nonrigid_enabled: bool,
    ) -> _DeviceBatchResult:
        """Resolves the rigid and optionally nonrigid offsets of one staged batch and applies them to its frames.

        Args:
            staged_frames: The uploaded batch with shape (batch_size, height, width), carrying either the int16
                storage dtype or the float32 arithmetic dtype.
            normalization_minimum: The minimum intensity value for clipping frames before correlation.
            normalization_maximum: The maximum intensity value for clipping frames before correlation.
            bidirectional_phase_offset: The pixel offset to correct bidirectional scanning artifacts.
            pre_smoothing_sigma: The sliding-window (box) smoothing size, in pixels, applied before high-pass
                filtering. Cast to an integer, which must be a positive even number.
            spatial_highpass_window: The window size for the spatial high-pass filter that removes low-frequency
                background.
            temporal_smoothing_sigma: The standard deviation for temporal Gaussian smoothing of correlation maps.
                If 0, no smoothing is applied.
            maximum_offset_fraction: The maximum allowed offset as a fraction of the minimum spatial dimension.
            signal_to_noise_threshold: The SNR threshold below which additional smoothing is applied to correlation
                peaks.
            maximum_block_offset: The maximum allowed offset for nonrigid blocks in pixels.
            one_photon_enabled: Determines whether to apply one-photon preprocessing, which includes spatial smoothing
                followed by high-pass filtering.
            nonrigid_enabled: Determines whether to apply nonrigid (piecewise) registration after rigid alignment.

        Returns:
            The registered frames and the offsets resolved from them, all held on the device.

        Raises:
            ValueError: If nonrigid registration is enabled and the backend holds no block reference data, or if
                one-photon preprocessing resolves a smoothing window that is not a positive even integer.
        """
        # Widens the storage dtype on the device, which is where the arithmetic runs.
        device_frames = staged_frames.astype(cupy.float32) if staged_frames.dtype == cupy.int16 else staged_frames

        if bidirectional_phase_offset != 0:
            self._apply_bidirectional_phase_correction(
                frames=device_frames, bidirectional_phase_offset=bidirectional_phase_offset
            )

        # Holds a working copy for correlation computation only when one-photon preprocessing replaces its
        # contents. On the two-photon path the smoothed frames stay equal to the registered frames through every
        # step below, so the two names share one buffer and the rigid shift is applied to it once.
        frames_smooth = device_frames
        if one_photon_enabled:
            if pre_smoothing_sigma > 0:
                frames_smooth = self._apply_spatial_smoothing(data=frames_smooth, window=int(pre_smoothing_sigma))
            frames_smooth = self._apply_spatial_high_pass(data=frames_smooth, window=spatial_highpass_window)

        frames_for_correlation = self._clip_intensities(
            frames=frames_smooth,
            normalization_minimum=normalization_minimum,
            normalization_maximum=normalization_maximum,
        )

        # Phase 1: rigid registration, which computes whole-frame translation offsets.
        y_offsets, x_offsets, correlations = self._compute_rigid_offsets(
            frames=self._apply_mask(frames=frames_for_correlation, mask=self._taper_mask, offset=self._mean_offset),
            reference_kernel=self._reference_kernel,
            maximum_offset_fraction=maximum_offset_fraction,
            temporal_smoothing_sigma=temporal_smoothing_sigma,
        )

        device_frames = self._translate_frames(frames=device_frames, y_offsets=y_offsets, x_offsets=x_offsets)

        # Phase 2: nonrigid registration, which computes per-block subpixel offsets for local deformations.
        y_offsets_nonrigid: cupy.ndarray | None = None
        x_offsets_nonrigid: cupy.ndarray | None = None
        correlations_nonrigid: cupy.ndarray | None = None
        if nonrigid_enabled:
            nonrigid_data = self._resolve_nonrigid_data()

            # Aligns the smoothed working copy the same way, so the per-block offsets carry local deformation
            # alone rather than the global translation the shift above already removed. The two-photon path
            # shares one buffer between the two names, where that shift already covers it.
            if one_photon_enabled:
                frames_smooth = self._translate_frames(frames=frames_smooth, y_offsets=y_offsets, x_offsets=x_offsets)
            else:
                frames_smooth = device_frames

            frames_for_correlation = self._clip_intensities(
                frames=frames_smooth,
                normalization_minimum=normalization_minimum,
                normalization_maximum=normalization_maximum,
            )

            y_offsets_nonrigid, x_offsets_nonrigid, correlations_nonrigid = self._compute_nonrigid_offsets(
                nonrigid_data=nonrigid_data,
                frames=frames_for_correlation,
                signal_to_noise_threshold=signal_to_noise_threshold,
                maximum_offset=maximum_block_offset,
            )

            device_frames = self._apply_nonrigid_correction(
                nonrigid_data=nonrigid_data,
                frames=device_frames,
                y_block_offsets=y_offsets_nonrigid,
                x_block_offsets=x_offsets_nonrigid,
            )

        return _DeviceBatchResult(
            frames=device_frames,
            y_offsets=y_offsets,
            x_offsets=x_offsets,
            correlations=correlations,
            y_offsets_nonrigid=y_offsets_nonrigid,
            x_offsets_nonrigid=x_offsets_nonrigid,
            correlations_nonrigid=correlations_nonrigid,
        )

    def _download_batch_result(
        self, device_result: _DeviceBatchResult, slot: int, *, narrow_frames: bool
    ) -> BatchRegistrationResult:
        """Copies one registered batch back to the host, narrowing its frames to the storage dtype when asked to.

        Args:
            device_result: The registered frames and offsets the batch produced on the device.
            slot: The index of the pipeline slot whose page-locked buffer receives the frames.
            narrow_frames: Determines whether the frames are clipped and narrowed to int16 before the copy.

        Returns:
            The host copy of the batch, whose frames alias the slot's page-locked buffer.
        """
        registered_frames, frame_sum = self._download_frames(
            device_frames=device_result.frames, slot=slot, narrow_frames=narrow_frames
        )
        return BatchRegistrationResult(
            frames=registered_frames,
            y_offsets=cupy.asnumpy(a=device_result.y_offsets),
            x_offsets=cupy.asnumpy(a=device_result.x_offsets),
            correlations=cupy.asnumpy(a=device_result.correlations),
            y_offsets_nonrigid=(
                None if device_result.y_offsets_nonrigid is None else cupy.asnumpy(a=device_result.y_offsets_nonrigid)
            ),
            x_offsets_nonrigid=(
                None if device_result.x_offsets_nonrigid is None else cupy.asnumpy(a=device_result.x_offsets_nonrigid)
            ),
            correlations_nonrigid=(
                None
                if device_result.correlations_nonrigid is None
                else cupy.asnumpy(a=device_result.correlations_nonrigid)
            ),
            frame_sum=frame_sum,
        )

    def _download_frames(
        self, device_frames: cupy.ndarray, slot: int, *, narrow_frames: bool
    ) -> tuple[NDArray[np.int16] | NDArray[np.float32], NDArray[np.float32] | None]:
        """Copies the registered frames of one batch into the slot's page-locked buffer.

        Notes:
            The per-pixel sum is reduced ahead of the clip, because the mean image the caller accumulates is measured
            on the unclipped frames while the plane binary stores the clipped ones.

        Args:
            device_frames: The registered frames with shape (batch_size, height, width), on the float32 arithmetic
                scale.
            slot: The index of the pipeline slot whose page-locked buffer receives the frames.
            narrow_frames: Determines whether the frames are clipped and narrowed to int16 before the copy.

        Returns:
            A tuple of the host frames, which alias the slot's page-locked buffer, and the per-pixel sum over the
            batch, which is None when the frames stay on the float32 scale.
        """
        device_frame_sum: cupy.ndarray | None = None
        if narrow_frames:
            device_frame_sum = device_frames.sum(axis=0, dtype=cupy.float32)
            device_frames = cupy.clip(a=device_frames, a_min=_INT16_MINIMUM, a_max=_INT16_MAXIMUM).astype(cupy.int16)

        shape = tuple(int(axis) for axis in device_frames.shape)
        host_frames = _resolve_pinned_buffer(slot=self._output_slots[slot], shape=shape, dtype=device_frames.dtype)
        device_frames.get(stream=self._compute_stream, out=host_frames, blocking=False)

        frame_sum = None if device_frame_sum is None else cupy.asnumpy(a=device_frame_sum)
        return host_frames, frame_sum

    def _resolve_nonrigid_data(self) -> _NonrigidDeviceData:
        """Returns the device copies of the nonrigid reference data, aborting when the reference data carries none.

        Returns:
            The nonrigid reference data held on the device.

        Raises:
            ValueError: If the reference data the backend was constructed from carries no block structure.
        """
        nonrigid_data = self._nonrigid_data
        if nonrigid_data is None:
            message = (
                f"Unable to run nonrigid registration on the GPU backend for device {self._device}. The reference "
                f"data the backend holds carries no block structure, so nonrigid registration was requested with a "
                f"reference computed for rigid registration alone."
            )
            console.error(message=message, error=ValueError)
        return nonrigid_data

    def _compute_rigid_offsets(
        self,
        frames: cupy.ndarray,
        reference_kernel: cupy.ndarray,
        maximum_offset_fraction: float,
        temporal_smoothing_sigma: float,
    ) -> tuple[cupy.ndarray, cupy.ndarray, cupy.ndarray]:
        """Computes rigid translation offsets using phase correlation.

        Args:
            frames: The frame data with shape (frame_count, height, width) after edge tapering.
            reference_kernel: The phase correlation kernel with shape (height, real_fft_width).
            maximum_offset_fraction: The maximum allowed offset as a fraction of the minimum spatial dimension. The
                search radius is clamped from below to one pixel, because the quadrant rearrangement reads a
                '-radius:' slice that degenerates to the whole axis at a zero radius. It is clamped from above to
                half the minimum dimension, which is the extent of the wrapped correlation surface.
            temporal_smoothing_sigma: The standard deviation for temporal Gaussian smoothing of correlation maps.
                If 0, no smoothing is applied.

        Returns:
            A tuple of the y offsets, the x offsets, and the correlation maxima, each with shape (frame_count,).
        """
        frame_count, height, width = frames.shape
        minimum_dimension = min(int(height), int(width))
        maximum_radius = minimum_dimension // 2
        requested_radius = int(min(round(maximum_offset_fraction * minimum_dimension), maximum_radius))
        correlation_radius = max(requested_radius, _MINIMUM_CORRELATION_RADIUS)

        correlation_data = self._apply_phase_correlation(frames=frames, kernel=reference_kernel)
        correlation_window = self._rearrange_correlation_quadrants(
            correlation_data=correlation_data, radius=correlation_radius
        )

        if temporal_smoothing_sigma > 0:
            correlation_window = self._apply_temporal_smoothing(
                frames=correlation_window, sigma=temporal_smoothing_sigma
            )

        window_size = 2 * correlation_radius + 1
        flat_window = correlation_window.reshape(int(frame_count), -1)
        flat_indices = flat_window.argmax(axis=1)
        y_offsets = (flat_indices // window_size - correlation_radius).astype(cupy.int32)
        x_offsets = (flat_indices % window_size - correlation_radius).astype(cupy.int32)
        correlation_maxima = flat_window[cupy.arange(int(frame_count)), flat_indices].astype(cupy.float32)

        return y_offsets, x_offsets, correlation_maxima

    def _compute_nonrigid_offsets(
        self,
        nonrigid_data: _NonrigidDeviceData,
        frames: cupy.ndarray,
        signal_to_noise_threshold: float,
        maximum_offset: float,
    ) -> tuple[cupy.ndarray, cupy.ndarray, cupy.ndarray]:
        """Computes nonrigid offsets using block-wise phase correlation.

        Estimates per-block subpixel offsets by correlating every frame block against the matching reference kernel.
        Each block carries a progressive smoothing loop that stops at the first level whose correlation peak clears
        the SNR threshold, so a block whose frames all clear the threshold at one level pays for that level alone.
        The loop keeps its control flow on the host and its arrays on the device, which costs one synchronization per
        level and preserves the early exit.

        Args:
            nonrigid_data: The device copies of the per-block reference data and the block geometry.
            frames: The frame data with shape (frame_count, height, width) to be registered.
            signal_to_noise_threshold: The signal-to-noise ratio threshold below which additional smoothing is applied
                to correlation peaks.
            maximum_offset: The maximum allowed offset in pixels, constrained by the block dimensions.

        Returns:
            A tuple of the y offsets, the x offsets, and the correlation maxima, each with shape (frame_count,
            block_count). The offsets carry the subpixel precision the RBF upsampling factor sets.
        """
        frame_count = int(frames.shape[0])
        block_count = int(nonrigid_data.taper_mask.shape[0])
        block_height = int(nonrigid_data.taper_mask.shape[1])
        block_width = int(nonrigid_data.taper_mask.shape[2])

        maximum_block_radius = min(block_height, block_width) // 2 - _UPSAMPLING_PADDING
        correlation_radius = int(min(round(maximum_offset), maximum_block_radius))

        extracted_blocks = frames[:, nonrigid_data.block_rows[:, :, None], nonrigid_data.block_columns[:, None, :]]
        extracted_blocks = self._apply_mask(
            frames=extracted_blocks, mask=nonrigid_data.taper_mask, offset=nonrigid_data.mean_offset
        )

        batch_size = min(_CORRELATION_BATCH_SIZE, block_count)
        for batch_start in range(0, block_count, batch_size):
            batch_end = min(block_count, batch_start + batch_size)
            extracted_blocks[:, batch_start:batch_end] = self._apply_phase_correlation(
                frames=extracted_blocks[:, batch_start:batch_end],
                kernel=nonrigid_data.reference_kernel[batch_start:batch_end],
            )

        half_window = correlation_radius + _UPSAMPLING_PADDING
        window_size = 2 * half_window + 1
        correlation_window = self._rearrange_correlation_quadrants(
            correlation_data=extracted_blocks, radius=half_window
        )
        correlation_window = correlation_window.transpose(1, 0, 2, 3).reshape(block_count, -1)

        # Applies progressive smoothing based on SNR. The third level squares the kernel before it reaches the
        # correlation surface, which keeps the product small while matching the association the host kernels use.
        smoothing_kernel = nonrigid_data.smoothing_kernel
        smoothing_levels = [
            correlation_window,
            smoothing_kernel @ correlation_window,
            smoothing_kernel @ smoothing_kernel @ correlation_window,
        ]
        smoothing_levels = [
            level.reshape(block_count, frame_count, window_size, window_size) for level in smoothing_levels
        ]
        smoothed_correlation = smoothing_levels[0]

        for block_index in range(block_count):
            signal_to_noise_ratio = cupy.ones(shape=frame_count, dtype=cupy.float32)
            for smoothing_index, smoothed_data in enumerate(smoothing_levels):
                low_signal_to_noise_mask = signal_to_noise_ratio < signal_to_noise_threshold
                if int(low_signal_to_noise_mask.sum()) == 0:
                    break
                block_correlation = smoothed_data[block_index][low_signal_to_noise_mask]
                if smoothing_index > 0:
                    smoothed_correlation[block_index][low_signal_to_noise_mask] = block_correlation
                signal_to_noise_ratio[low_signal_to_noise_mask] = self._compute_correlation_signal_to_noise_ratio(
                    correlation_data=block_correlation, padding=_UPSAMPLING_PADDING
                )

        return self._locate_subpixel_peaks(
            nonrigid_data=nonrigid_data,
            smoothed_correlation=smoothed_correlation,
            correlation_radius=correlation_radius,
        )

    def _locate_subpixel_peaks(
        self,
        nonrigid_data: _NonrigidDeviceData,
        smoothed_correlation: cupy.ndarray,
        correlation_radius: int,
    ) -> tuple[cupy.ndarray, cupy.ndarray, cupy.ndarray]:
        """Locates the subpixel correlation peak of every block and frame through Gaussian RBF upsampling.

        Args:
            nonrigid_data: The device copies of the per-block reference data and the block geometry.
            smoothed_correlation: The SNR-smoothed correlation surfaces with shape (block_count, frame_count,
                window_size, window_size).
            correlation_radius: The integer search radius the central correlation region spans.

        Returns:
            A tuple of the y offsets, the x offsets, and the correlation maxima, each with shape (frame_count,
            block_count).
        """
        block_count = int(smoothed_correlation.shape[0])
        frame_count = int(smoothed_correlation.shape[1])
        upsampled_size = nonrigid_data.upsampled_size
        midpoint = upsampled_size // 2
        region_size = 2 * _UPSAMPLING_PADDING + 1
        central_size = 2 * correlation_radius + 1

        central_regions = smoothed_correlation[
            :, :, _UPSAMPLING_PADDING:-_UPSAMPLING_PADDING, _UPSAMPLING_PADDING:-_UPSAMPLING_PADDING
        ]
        central_flat = central_regions.reshape(block_count * frame_count, -1)
        flat_indices = central_flat.argmax(axis=1)
        y_peaks = (flat_indices // central_size).reshape(block_count, frame_count).astype(cupy.int32)
        x_peaks = (flat_indices % central_size).reshape(block_count, frame_count).astype(cupy.int32)

        # Gathers the region around every peak as one strided read over the block and frame axes.
        region_indices = cupy.arange(region_size, dtype=cupy.int32)
        block_indices = cupy.arange(block_count, dtype=cupy.int32)[:, None, None, None]
        frame_indices = cupy.arange(frame_count, dtype=cupy.int32)[None, :, None, None]
        upsampling_regions = smoothed_correlation[
            block_indices,
            frame_indices,
            y_peaks[:, :, None, None] + region_indices[None, None, :, None],
            x_peaks[:, :, None, None] + region_indices[None, None, None, :],
        ]

        upsampled_flat = upsampling_regions.reshape(block_count * frame_count, -1) @ nonrigid_data.upsampling_kernel
        subpixel_indices = upsampled_flat.argmax(axis=1)
        correlation_maxima = (
            upsampled_flat[cupy.arange(block_count * frame_count), subpixel_indices]
            .reshape(block_count, frame_count)
            .T.astype(cupy.float32)
        )

        y_subpixel = (subpixel_indices // upsampled_size).reshape(block_count, frame_count)
        x_subpixel = (subpixel_indices % upsampled_size).reshape(block_count, frame_count)
        y_offsets = ((y_subpixel - midpoint) / _SUBPIXEL_FACTOR + (y_peaks - correlation_radius)).T
        x_offsets = ((x_subpixel - midpoint) / _SUBPIXEL_FACTOR + (x_peaks - correlation_radius)).T

        return y_offsets.astype(cupy.float32), x_offsets.astype(cupy.float32), correlation_maxima

    def _apply_nonrigid_correction(
        self,
        nonrigid_data: _NonrigidDeviceData,
        frames: cupy.ndarray,
        y_block_offsets: cupy.ndarray,
        x_block_offsets: cupy.ndarray,
    ) -> cupy.ndarray:
        """Applies nonrigid motion correction to the input batch of frames using the block offsets.

        Args:
            nonrigid_data: The device copies of the per-block reference data and the block geometry.
            frames: The frame data with shape (frame_count, height, width).
            y_block_offsets: The y-offsets per block with shape (frame_count, block_count). Positive values shift
                content upward.
            x_block_offsets: The x-offsets per block with shape (frame_count, block_count). Positive values shift
                content leftward.

        Returns:
            The corrected frames with shape (frame_count, height, width).
        """
        frame_count = int(frames.shape[0])
        y_block_count, x_block_count = nonrigid_data.block_counts

        # Interpolates the sparse block offsets onto a per-pixel map, then folds the base coordinate grid into that
        # map so the warp reads one coordinate array per axis instead of a grid and a map.
        y_offset_maps = self._sample_bilinear(
            source=y_block_offsets.reshape(frame_count, y_block_count, x_block_count),
            y_coordinates=nonrigid_data.block_row_grid,
            x_coordinates=nonrigid_data.block_column_grid,
        )
        x_offset_maps = self._sample_bilinear(
            source=x_block_offsets.reshape(frame_count, y_block_count, x_block_count),
            y_coordinates=nonrigid_data.block_row_grid,
            x_coordinates=nonrigid_data.block_column_grid,
        )
        y_offset_maps += nonrigid_data.pixel_row_grid
        x_offset_maps += nonrigid_data.pixel_column_grid

        return self._sample_bilinear(source=frames, y_coordinates=y_offset_maps, x_coordinates=x_offset_maps)

    def _apply_spatial_high_pass(self, data: cupy.ndarray, window: int) -> cupy.ndarray:
        """Applies a spatial high-pass filter using the sliding window method.

        Args:
            data: The frames with shape (frame_count, height, width) to filter.
            window: The window size for the low-pass component to subtract.

        Returns:
            The high-pass filtered frames with the same shape as the input.
        """
        normalization = self._resolve_normalization_weights(window=window)
        low_pass = self._apply_spatial_smoothing(data=data, window=window)
        low_pass /= normalization
        return data - low_pass

    def _resolve_normalization_weights(self, window: int) -> cupy.ndarray:
        """Returns the border normalization weights that correct the zero padding of the smoothing window.

        Args:
            window: The smoothing window size the weights correct.

        Returns:
            The normalization weights with shape (height, width).
        """
        weights = self._normalization_weights.get(window)
        if weights is None:
            ones_array = cupy.ones(shape=(1, self._frame_height, self._frame_width), dtype=cupy.float32)
            weights = self._apply_spatial_smoothing(data=ones_array, window=window)[0]
            self._normalization_weights[window] = weights
        return weights

    @staticmethod
    def _apply_mask(frames: cupy.ndarray, mask: cupy.ndarray, offset: cupy.ndarray) -> cupy.ndarray:
        """Applies a spatial mask to frame data.

        Notes:
            The mask and offset arrays match the shape of a single frame, so this covers both the rigid case, where a
            two-dimensional mask meets three-dimensional frames, and the nonrigid case, where a three-dimensional
            per-block mask meets four-dimensional extracted blocks.

        Args:
            frames: The frame data with shape (frame_count, height, width) or (frame_count, block_count, height,
                width).
            mask: The multiplicative taper mask shaped like one frame.
            offset: The additive offset shaped like one frame.

        Returns:
            The masked frames with the same shape as the input frames.
        """
        return frames * mask + offset

    @staticmethod
    def _apply_phase_correlation(frames: cupy.ndarray, kernel: cupy.ndarray) -> cupy.ndarray:
        """Applies phase correlation between the frames and the reference kernel.

        Notes:
            Every transform is a real FFT, and the inverse names the output size explicitly. A real FFT stores an
            odd-width axis and its even-width neighbor in the same number of frequency bins. An inverse that infers
            its size from the spectrum therefore returns an even width, silently dropping the last column of an
            odd-width frame.

        Args:
            frames: The frames to correlate with shape (frame_count, height, width) or (frame_count, block_count,
                height, width).
            kernel: The conjugated reference spectrum, shaped to broadcast over the leading frame axis.

        Returns:
            The correlation maps with the same shape as the input frames.
        """
        height = int(frames.shape[-2])
        width = int(frames.shape[-1])

        spectra = cupy.fft.rfft2(a=frames, axes=(-2, -1)).astype(cupy.complex64, copy=False)

        # Normalizes by magnitude to extract phase-only information, which makes the correlation robust to intensity
        # variations between frames. Folding the epsilon into the magnitude buffer keeps the normalization down to a
        # single full spectra temporary.
        magnitude = cupy.abs(spectra)
        magnitude += np.float32(NORMALIZATION_EPSILON)
        spectra /= magnitude
        spectra *= kernel

        correlation: cupy.ndarray = cupy.fft.irfft2(a=spectra, s=(height, width), axes=(-2, -1)).astype(
            cupy.float32, copy=False
        )
        return correlation

    @staticmethod
    def _apply_temporal_smoothing(frames: cupy.ndarray, sigma: float) -> cupy.ndarray:
        """Applies Gaussian filtering along the temporal (first) axis.

        Args:
            frames: The frames with shape (frame_count, height, width) to be smoothed.
            sigma: The standard deviation of the Gaussian kernel.

        Returns:
            The temporally smoothed frames with the same shape as the input.
        """
        smoothed: cupy.ndarray = gaussian_filter1d(input=frames, sigma=sigma, axis=0).astype(cupy.float32)
        return smoothed

    @staticmethod
    def _apply_spatial_smoothing(data: cupy.ndarray, window: int) -> cupy.ndarray:
        """Applies spatial smoothing using cumulative sum with a sliding window.

        Args:
            data: The frames with shape (frame_count, height, width) to smooth.
            window: The window size for smoothing. Must be a positive even integer.

        Returns:
            The spatially smoothed frames with the same shape as the input.

        Raises:
            ValueError: If the window size is not a positive even integer.
        """
        # Rejects a zero or negative window here rather than letting it reach the integral-image differencing below,
        # where the ':-window' slice bound degenerates to the empty prefix and the box normalization divides by zero.
        if window <= 0 or window % 2:
            message = (
                f"Unable to apply spatial smoothing on the GPU backend. Filter window must be a positive even "
                f"integer, but got {window}."
            )
            console.error(message=message, error=ValueError)

        # Pads spatial dimensions with zeros to handle window edges. Border pixels are summed over partial
        # (zero-filled) windows but still divided by the full window squared, so their means are under-estimated and
        # corrected later through the normalization weights the high-pass filter divides by.
        half_pad = window // 2
        data_padded = cupy.pad(
            array=data,
            pad_width=((0, 0), (half_pad, half_pad), (half_pad, half_pad)),
            mode="constant",
            constant_values=0,
        )

        # Computes the integral image through cumulative sums along height then width. Both calls name float32 to
        # keep the running totals off the device's float64 path.
        data_summed = data_padded.cumsum(axis=1, dtype=cupy.float32).cumsum(axis=2, dtype=cupy.float32)

        data_summed = data_summed[:, window:, :] - data_summed[:, :-window, :]
        data_summed = data_summed[:, :, window:] - data_summed[:, :, :-window]
        data_summed /= window**2

        return data_summed

    @staticmethod
    def _clip_intensities(
        frames: cupy.ndarray, normalization_minimum: float, normalization_maximum: float
    ) -> cupy.ndarray:
        """Clips the frame intensity range to reduce the influence of outlier pixels on the correlation.

        Args:
            frames: The frames with shape (frame_count, height, width) to clip.
            normalization_minimum: The minimum intensity value, negative infinity when normalization is disabled.
            normalization_maximum: The maximum intensity value, positive infinity when normalization is disabled.

        Returns:
            The clipped frames, which are the input frames themselves when normalization is disabled.
        """
        if normalization_minimum > -np.inf:
            return cupy.clip(a=frames, a_min=normalization_minimum, a_max=normalization_maximum)
        return frames

    @staticmethod
    def _translate_frames(frames: cupy.ndarray, y_offsets: cupy.ndarray, x_offsets: cupy.ndarray) -> cupy.ndarray:
        """Applies a per-frame rigid translation to a batch of frames using circular shifting.

        Args:
            frames: The frames with shape (frame_count, height, width) to translate.
            y_offsets: The vertical offsets with shape (frame_count,). Positive values shift content upward.
            x_offsets: The horizontal offsets with shape (frame_count,). Positive values shift content leftward.

        Returns:
            The translated frames with the same shape as the input.
        """
        frame_count = int(frames.shape[0])
        height = int(frames.shape[1])
        width = int(frames.shape[2])

        rows = (cupy.arange(height, dtype=cupy.int32)[None, :] + y_offsets[:, None]) % height
        columns = (cupy.arange(width, dtype=cupy.int32)[None, :] + x_offsets[:, None]) % width
        frame_indices = cupy.arange(frame_count, dtype=cupy.int32)[:, None, None]

        translated: cupy.ndarray = frames[frame_indices, rows[:, :, None], columns[:, None, :]]
        return translated

    @staticmethod
    def _apply_bidirectional_phase_correction(frames: cupy.ndarray, bidirectional_phase_offset: int) -> None:
        """Applies bidirectional phase correction to the frames in place.

        Notes:
            The source rows are copied before they are written back, because the source and destination slices
            overlap and the device copy that backs a slice assignment carries no overlap check.

        Args:
            frames: The frames with shape (frame_count, height, width) to correct in place.
            bidirectional_phase_offset: The horizontal offset in pixels to apply to odd lines. Positive values shift
                odd lines to the right, negative values shift them to the left, and a zero offset leaves the frames
                untouched.
        """
        if bidirectional_phase_offset == 0:
            return

        if bidirectional_phase_offset > 0:
            # Shifts odd lines right and zeros the left border for consistency with spatial filtering zero-padding.
            shifted_lines = frames[:, 1::2, :-bidirectional_phase_offset].copy()
            frames[:, 1::2, bidirectional_phase_offset:] = shifted_lines
            frames[:, 1::2, :bidirectional_phase_offset] = 0
        else:
            # Shifts odd lines left and zeros the right border for consistency with spatial filtering zero-padding.
            shifted_lines = frames[:, 1::2, -bidirectional_phase_offset:].copy()
            frames[:, 1::2, :bidirectional_phase_offset] = shifted_lines
            frames[:, 1::2, bidirectional_phase_offset:] = 0

    @staticmethod
    def _rearrange_correlation_quadrants(correlation_data: cupy.ndarray, radius: int) -> cupy.ndarray:
        """Rearranges the four quadrants of a wrapped correlation surface into a window centered on zero offset.

        Notes:
            The correlation surface wraps around, so a negative offset appears at the end of its axis. The negative
            half spans the last 'radius' samples and the non-negative half spans the first 'radius + 1' samples,
            which is what makes the two slices asymmetric.

        Args:
            correlation_data: The correlation surfaces, whose last two axes carry the wrapped offsets.
            radius: The search radius, in pixels, the rearranged window spans in each direction.

        Returns:
            The rearranged window, whose last two axes each span 2 * radius + 1 samples.
        """
        upper = cupy.concatenate(
            (correlation_data[..., -radius:, -radius:], correlation_data[..., -radius:, : radius + 1]), axis=-1
        )
        lower = cupy.concatenate(
            (correlation_data[..., : radius + 1, -radius:], correlation_data[..., : radius + 1, : radius + 1]),
            axis=-1,
        )
        window: cupy.ndarray = cupy.concatenate((upper, lower), axis=-2)
        return window

    @staticmethod
    def _sample_bilinear(
        source: cupy.ndarray, y_coordinates: cupy.ndarray, x_coordinates: cupy.ndarray
    ) -> cupy.ndarray:
        """Samples every source image of a batch at the requested coordinates using bilinear interpolation.

        Notes:
            The coordinate is truncated toward zero and its fractional part is taken before the four neighbor indices
            are clamped into range. A coordinate above the last row or column therefore resolves to the edge pixel,
            while a coordinate below zero keeps a negative fraction and is linearly extrapolated from the first two edge
            pixels. Neither the map_coordinates routine nor a normalized grid sampler reproduces that extrapolation.

        Args:
            source: The source images with shape (frame_count, source_height, source_width).
            y_coordinates: The target y-coordinates with shape (frame_count, height, width) or (height, width).
            x_coordinates: The target x-coordinates with shape (frame_count, height, width) or (height, width).

        Returns:
            The sampled images with shape (frame_count, height, width).
        """
        frame_count = int(source.shape[0])
        source_height = int(source.shape[1])
        source_width = int(source.shape[2])

        # Truncates in single precision, so that the fraction the subtraction leaves stays float32 rather than
        # promoting the whole interpolation to the device's float64 path.
        y_truncated = cupy.trunc(y_coordinates)
        x_truncated = cupy.trunc(x_coordinates)
        y_fraction = y_coordinates - y_truncated
        x_fraction = x_coordinates - x_truncated
        y_complement = 1.0 - y_fraction
        x_complement = 1.0 - x_fraction

        y_floor = cupy.clip(a=y_truncated.astype(cupy.int32), a_min=0, a_max=source_height - 1)
        x_floor = cupy.clip(a=x_truncated.astype(cupy.int32), a_min=0, a_max=source_width - 1)
        y_ceiling = cupy.minimum(y_floor + 1, source_height - 1)
        x_ceiling = cupy.minimum(x_floor + 1, source_width - 1)

        frame_indices = cupy.arange(frame_count, dtype=cupy.int32)[:, None, None]
        neighbors = (
            (y_floor, x_floor, y_complement, x_complement),
            (y_floor, x_ceiling, y_complement, x_fraction),
            (y_ceiling, x_floor, y_fraction, x_complement),
            (y_ceiling, x_ceiling, y_fraction, x_fraction),
        )

        output_shape = (frame_count, int(y_coordinates.shape[-2]), int(y_coordinates.shape[-1]))
        output = cupy.zeros(shape=output_shape, dtype=cupy.float32)
        for rows, columns, row_weight, column_weight in neighbors:
            sampled = source[frame_indices, rows, columns]
            sampled *= row_weight
            sampled *= column_weight
            output += sampled

        return output

    @staticmethod
    def _compute_correlation_signal_to_noise_ratio(correlation_data: cupy.ndarray, padding: int) -> cupy.ndarray:
        """Computes the signal-to-noise ratio of the phase correlation peaks.

        Estimates the ratio by comparing the maximum correlation value inside the central region to the maximum
        value outside a padding box anchored at that peak. A low ratio marks an offset estimate additional smoothing
        may improve.

        Args:
            correlation_data: The correlation data with shape (frame_count, window_height, window_width).
            padding: The padding width, in pixels, excluded around the peak when the background is measured.

        Returns:
            The SNR values with shape (frame_count,).
        """
        frame_count = int(correlation_data.shape[0])
        window_height = int(correlation_data.shape[1])
        window_width = int(correlation_data.shape[2])
        central_width = window_width - 2 * padding

        central_flat = correlation_data[:, padding : window_height - padding, padding : window_width - padding].reshape(
            frame_count, -1
        )
        peak_indices = central_flat.argmax(axis=1)
        peak_values = central_flat[cupy.arange(frame_count), peak_indices]
        peak_rows = (peak_indices // central_width)[:, None, None]
        peak_columns = (peak_indices % central_width)[:, None, None]

        rows = cupy.arange(window_height, dtype=cupy.int32)[None, :, None]
        columns = cupy.arange(window_width, dtype=cupy.int32)[None, None, :]
        inside_peak = (
            (rows >= peak_rows)
            & (rows < peak_rows + 2 * padding)
            & (columns >= peak_columns)
            & (columns < peak_columns + 2 * padding)
        )

        background = correlation_data.copy()
        background[inside_peak] = np.float32(-np.inf)

        # Keeps the ratio positive for an outlier frame whose background is vanishingly small or wholly masked.
        signal_to_noise_ratio: cupy.ndarray = (
            peak_values / cupy.maximum(background.max(axis=(1, 2)), np.float32(_SIGNAL_TO_NOISE_EPSILON))
        ).astype(cupy.float32)
        return signal_to_noise_ratio


def _require_gpu_runtime() -> None:
    """Verifies that the CuPy distribution is installed, aborting the caller when it is absent.

    Raises:
        RuntimeError: If the CuPy distribution is not installed.
    """
    if cupy is not None:
        return

    message = (
        f"Unable to initialize the GPU registration backend. The CuPy distribution is not installed, so no CUDA "
        f"device is reachable. {_GPU_REMEDY} Omit the device argument to run the stage on the host CPU instead."
    )
    console.error(message=message, error=RuntimeError)


def _verify_tf32_disabled() -> None:
    """Verifies that the current device performs its matrix multiplications at full single precision.

    Notes:
        The nonrigid subpixel stage multiplies a (block_count * frame_count, 49) correlation matrix by a
        (49, 3721) RBF upsampling matrix and reads the result with an argmax over a 61 by 61 surface. TF32 carries a
        10-bit mantissa, which perturbs neighboring samples of that surface by more than the 0.1 pixel spacing
        between them and therefore moves the reported peak by a whole subpixel quantum. CuPy switches the cuBLAS math
        mode away from its default only when the environment enables TF32, so the mode the handle reports answers the
        question directly.

    Raises:
        RuntimeError: If the cuBLAS handle of the current device allows TF32 matrix multiplication.
    """
    math_mode = cupy.cuda.cublas.getMathMode(cupy.cuda.Device().cublas_handle)
    if math_mode == cupy.cuda.cublas.CUBLAS_DEFAULT_MATH:
        return

    message = (
        f"Unable to initialize the GPU registration backend. The cuBLAS handle of the selected device reports math "
        f"mode {math_mode} rather than the default single-precision mode "
        f"{int(cupy.cuda.cublas.CUBLAS_DEFAULT_MATH)}, so the nonrigid subpixel upsampling would run at reduced "
        f"precision and shift the reported correlation peak by a whole 0.1 pixel quantum. Set the "
        f"'{_TF32_VARIABLE}' environment variable to 0 before starting the process."
    )
    console.error(message=message, error=RuntimeError)


def _upload_nonrigid_data(
    reference_data: ReferenceData, frame_height: int, frame_width: int
) -> _NonrigidDeviceData | None:
    """Uploads the per-block reference data and the block geometry the nonrigid stage reads to the current device.

    Notes:
        The block index arrays, the interpolation grids, and the RBF upsampling matrix depend on the block layout
        alone, so they are derived once on the host and held on the device for the lifetime of the backend.

    Args:
        reference_data: The precomputed reference data the backend was constructed from.
        frame_height: The frame height, in pixels.
        frame_width: The frame width, in pixels.

    Returns:
        The device copies of the nonrigid reference data, or None when the reference data carries no blocks.
    """
    blocks = reference_data.blocks
    taper_mask = reference_data.taper_mask_nonrigid
    mean_offset = reference_data.mean_offset_nonrigid
    reference_kernel = reference_data.reference_kernel_nonrigid
    if blocks is None or taper_mask is None or mean_offset is None or reference_kernel is None:
        return None

    y_blocks, x_blocks, block_counts, _, smoothing_kernel = blocks

    block_rows = np.stack([np.arange(y_range[0], y_range[1], dtype=np.int32) for y_range in y_blocks])
    block_columns = np.stack([np.arange(x_range[0], x_range[1], dtype=np.int32) for x_range in x_blocks])

    # Recovers the block center coordinates from the block boundary arrays, then maps every pixel onto the block
    # index axis so the per-block offsets interpolate onto a per-pixel map.
    y_centers = np.array(y_blocks[:: block_counts[1]], dtype=np.float32).mean(axis=1)
    x_centers = np.array(x_blocks[: block_counts[1]], dtype=np.float32).mean(axis=1)
    y_indices = np.interp(x=np.arange(frame_height), xp=y_centers, fp=np.arange(y_centers.size)).astype(np.float32)
    x_indices = np.interp(x=np.arange(frame_width), xp=x_centers, fp=np.arange(x_centers.size)).astype(np.float32)
    block_column_grid, block_row_grid = np.meshgrid(x_indices, y_indices)

    pixel_column_grid, pixel_row_grid = np.meshgrid(
        np.arange(frame_width, dtype=np.float32), np.arange(frame_height, dtype=np.float32)
    )

    upsampling_kernel, upsampled_size = compute_upsampling_kernel(padding=_UPSAMPLING_PADDING)

    return _NonrigidDeviceData(
        taper_mask=cupy.asarray(a=taper_mask, dtype=cupy.float32),
        mean_offset=cupy.asarray(a=mean_offset, dtype=cupy.float32),
        reference_kernel=cupy.asarray(a=reference_kernel, dtype=cupy.complex64),
        block_rows=cupy.asarray(a=block_rows, dtype=cupy.int32),
        block_columns=cupy.asarray(a=block_columns, dtype=cupy.int32),
        smoothing_kernel=cupy.asarray(a=smoothing_kernel, dtype=cupy.float32),
        upsampling_kernel=cupy.asarray(a=upsampling_kernel, dtype=cupy.float32),
        upsampled_size=int(upsampled_size),
        block_counts=(int(block_counts[0]), int(block_counts[1])),
        block_row_grid=cupy.asarray(a=block_row_grid, dtype=cupy.float32),
        block_column_grid=cupy.asarray(a=block_column_grid, dtype=cupy.float32),
        pixel_row_grid=cupy.asarray(a=pixel_row_grid, dtype=cupy.float32),
        pixel_column_grid=cupy.asarray(a=pixel_column_grid, dtype=cupy.float32),
    )


def _resolve_pinned_buffer(
    slot: _StagingSlot, shape: tuple[int, ...], dtype: np.dtype[np.int16] | np.dtype[np.float32]
) -> NDArray[np.int16] | NDArray[np.float32]:
    """Returns the slot's page-locked host buffer for one batch geometry, allocating it on the first request.

    Notes:
        A page-locked buffer is what makes a device copy asynchronous, so the buffer is held for the lifetime of the
        backend and reused by every batch of the same geometry the slot stages afterwards.

    Args:
        slot: The staging slot the buffer belongs to.
        shape: The shape of the batch the buffer stages.
        dtype: The dtype of the batch the buffer stages.

    Returns:
        The page-locked host buffer, whose contents are those of the previous batch staged through the slot.
    """
    key = (shape, dtype.str)
    buffer = slot.host_buffers.get(key)
    if buffer is not None:
        return buffer

    element_count = math.prod(shape)
    memory = cupy.cuda.alloc_pinned_memory(element_count * dtype.itemsize)

    # The NumPy stubs resolve a union dtype to the generic scalar type, so the concrete dtype the allocation carries
    # is restored here rather than inferred.
    buffer = cast(
        "NDArray[np.int16] | NDArray[np.float32]",
        np.frombuffer(buffer=memory, dtype=dtype, count=element_count).reshape(shape),
    )
    slot.host_buffers[key] = buffer
    return buffer


def _resolve_device_buffer(
    slot: _StagingSlot, shape: tuple[int, ...], dtype: np.dtype[np.int16] | np.dtype[np.float32]
) -> cupy.ndarray:
    """Returns the slot's device buffer for one batch geometry, allocating it on the first request.

    Notes:
        The buffer is held rather than allocated per batch, because the device memory pool sorts its free blocks by
        the stream that released them and the upload stream differs from the stream every computation runs on.

    Args:
        slot: The staging slot the buffer belongs to.
        shape: The shape of the batch the buffer receives.
        dtype: The dtype of the batch the buffer receives.

    Returns:
        The device buffer, whose contents are those of the previous batch staged through the slot.
    """
    key = (shape, dtype.str)
    buffer = slot.device_buffers.get(key)
    if buffer is None:
        buffer = cupy.empty(shape=shape, dtype=dtype)
        slot.device_buffers[key] = buffer
    return buffer
