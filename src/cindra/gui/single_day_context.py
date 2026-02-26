"""Provides data hierarchies for single-day pipeline viewers.

Contains ``ROIViewerData`` for ROI viewer and ROI editor consumption, and ``RegistrationViewerData`` for the
registration quality viewer. Both classes delegate to ``RuntimeContext`` and ``CombinedData`` objects loaded from
cindra pipeline output directories.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from dataclasses import field, dataclass

import numpy as np
from PySide6.QtWidgets import QFileDialog
from ataraxis_base_utilities import console

from ..io import BinaryFile
from ..dataclasses import CombinedData, RuntimeContext

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..dataclasses import (
        IOData,
        ROIStatistics,
        BaselineMethod,
        DetectionData,
        ExtractionData,
        RegistrationData,
        SingleDayConfiguration,
    )


def _memory_map_trace(
    output_path: Path | None,
    file_name: str,
    fallback: NDArray[np.float32] | None,
    default_shape: tuple[int, ...],
    *,
    mmap_mode: str = "r",
) -> NDArray[np.float32]:
    """Memory-maps a trace array from disk.

    Attempts to open the specified .npy file as a memory map with the given mode. Falls back to
    returning a view or copy of the provided array (depending on the mode) if the file does not
    exist, or returns a zero array of the given shape if neither source is available.

    Args:
        output_path: Directory containing the .npy file.
        file_name: Name of the .npy file to memory-map.
        fallback: Array to use if the file does not exist on disk.
        default_shape: Shape for a zero-initialized fallback array.
        mmap_mode: Memory-map mode passed to ``np.load``. Use ``"r"`` for read-only access or
            ``"c"`` for copy-on-write access.

    Returns:
        The memory-mapped, viewed/copied, or zero-initialized array.
    """
    if output_path is not None:
        path = output_path / file_name
        if path.exists():
            mapped: NDArray[np.float32] = np.load(path, mmap_mode=mmap_mode)
            return mapped
    if fallback is not None:
        if mmap_mode == "r":
            result: NDArray[np.float32] = fallback.view()
            result.flags.writeable = False
            return result
        return fallback.copy()
    return np.zeros(default_shape, dtype=np.float32)


def _memory_map_optional_trace(
    output_path: Path | None,
    file_name: str,
    fallback: NDArray[np.float32] | None,
    *,
    mmap_mode: str = "r",
) -> NDArray[np.float32] | None:
    """Memory-maps an optional trace array from disk.

    Attempts to open the specified .npy file as a memory map with the given mode. Falls back to
    returning a view or copy of the provided array (depending on the mode) if the file does not
    exist. Returns None if neither source is available.

    Args:
        output_path: Directory containing the .npy file.
        file_name: Name of the .npy file to memory-map.
        fallback: Array to use if the file does not exist on disk.
        mmap_mode: Memory-map mode passed to ``np.load``. Use ``"r"`` for read-only access or
            ``"c"`` for copy-on-write access.

    Returns:
        The memory-mapped or viewed/copied array, or None if unavailable.
    """
    if output_path is not None:
        path = output_path / file_name
        if path.exists():
            mapped: NDArray[np.float32] = np.load(path, mmap_mode=mmap_mode)
            return mapped
    if fallback is not None:
        if mmap_mode == "r":
            result: NDArray[np.float32] = fallback.view()
            result.flags.writeable = False
            return result
        return fallback.copy()
    return None


def _release_trace_arrays(extraction: ExtractionData) -> None:
    """Releases large trace arrays from the extraction object to free memory.

    Called after memory-mapping the same data from disk, so the in-memory copies held by the extraction object are no
    longer needed.

    Args:
        extraction: The extraction data whose trace arrays will be set to None.
    """
    extraction.cell_fluorescence = None
    extraction.neuropil_fluorescence = None
    extraction.spikes = None
    extraction.subtracted_fluorescence = None
    extraction.cell_fluorescence_channel_2 = None
    extraction.neuropil_fluorescence_channel_2 = None
    extraction.subtracted_fluorescence_channel_2 = None


@dataclass
class ROIViewerData:
    """Wraps imported single-day pipeline data for ROI viewer and editor consumption.

    Holds references to pipeline arrays and delegates access to the underlying context objects. Large trace arrays are
    memory-mapped from disk to minimize memory usage. The ``mutable`` flag controls whether arrays are opened with
    read-only or copy-on-write access.

    Notes:
        Single-day data comes from ``RuntimeContext.load()`` which provides ``CombinedData`` containing detection images
        and extraction results. Construct via the ``from_single_day`` or ``from_dialog`` factory methods.
    """

    contexts: list[RuntimeContext] = field(default_factory=list)
    """Single-day runtime contexts for all processed recording's imaging planes."""

    combined: CombinedData | None = None
    """The combined single-day data for the active view."""

    roi_statistics: list[ROIStatistics] = field(default_factory=list)
    """Spatial and shape statistics for each detected ROI."""

    cell_fluorescence: NDArray[np.float32] = field(default_factory=lambda: np.array([], dtype=np.float32))
    """Cell fluorescence traces with shape (cells, frames)."""

    neuropil_fluorescence: NDArray[np.float32] = field(default_factory=lambda: np.array([], dtype=np.float32))
    """Neuropil fluorescence traces with shape (cells, frames)."""

    spikes: NDArray[np.float32] = field(default_factory=lambda: np.array([], dtype=np.float32))
    """Deconvolved spike traces with shape (cells, frames)."""

    cell_classification_labels: NDArray[np.bool_] = field(default_factory=lambda: np.array([], dtype=np.bool_))
    """Boolean classification array marking each ROI as cell or non-cell."""

    cell_classification_probabilities: NDArray[np.float32] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    """Classifier probability for each ROI being a cell."""

    cell_colocalization_labels: NDArray[np.bool_] = field(default_factory=lambda: np.array([], dtype=np.bool_))
    """Boolean array marking each ROI as colocalized with a channel 2 fluorescence source."""

    cell_colocalization_probabilities: NDArray[np.float32] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    """Classifier probability of each ROI being colocalized with a channel 2 fluorescence source."""

    has_channel_2: bool = False
    """Determines whether channel 2 data is available."""

    not_merged: NDArray[np.bool_] = field(default_factory=lambda: np.array([], dtype=np.bool_))
    """Boolean mask tracking which ROIs have not been merged into other ROIs. Only populated when mutable=True."""

    cell_fluorescence_channel_2: NDArray[np.float32] | None = None
    """Channel 2 cell fluorescence traces. None if single-channel."""

    neuropil_fluorescence_channel_2: NDArray[np.float32] | None = None
    """Channel 2 neuropil fluorescence traces. None if single-channel."""

    _mutable: bool = field(init=False, default=False, repr=False)
    """Determines whether arrays are opened with copy-on-write (True) or read-only (False) access."""

    _subtracted_fluorescence_map: NDArray[np.float32] | None = field(init=False, default=None, repr=False)
    """Cached memory-mapped channel 1 subtracted fluorescence."""

    _subtracted_fluorescence_channel_2_map: NDArray[np.float32] | None = field(init=False, default=None, repr=False)
    """Cached memory-mapped channel 2 subtracted fluorescence."""

    @property
    def frame_height(self) -> int:
        """Returns the combined field-of-view height in pixels."""
        if self.combined is None:
            return 0
        return self.combined.combined_height

    @property
    def frame_width(self) -> int:
        """Returns the combined field-of-view width in pixels."""
        if self.combined is None:
            return 0
        return self.combined.combined_width

    @property
    def sampling_rate(self) -> float:
        """Returns the per-plane sampling rate in Hertz."""
        if self.combined is None:
            return 0.0
        return self.combined.sampling_rate

    @property
    def tau(self) -> float:
        """Returns the calcium indicator timescale in seconds."""
        if self.combined is None:
            return 0.0
        return self.combined.tau

    @property
    def roi_count(self) -> int:
        """Returns the total number of ROIs."""
        return len(self.roi_statistics)

    @property
    def cell_count(self) -> int:
        """Returns the number of ROIs classified as cells."""
        if self.cell_classification_labels.size == 0:
            return 0
        return int(self.cell_classification_labels.sum())

    @property
    def mean_image(self) -> NDArray[np.float32] | None:
        """Returns the mean image from the combined detection data."""
        if self.combined is None:
            return None
        return self.combined.detection.mean_image

    @property
    def enhanced_mean_image(self) -> NDArray[np.float32] | None:
        """Returns the enhanced mean image from the combined detection data."""
        if self.combined is None:
            return None
        return self.combined.detection.enhanced_mean_image

    @property
    def maximum_projection(self) -> NDArray[np.float32] | None:
        """Returns the maximum projection from the combined detection data."""
        if self.combined is None:
            return None
        return self.combined.detection.maximum_projection

    @property
    def correlation_map(self) -> NDArray[np.float32] | None:
        """Returns the correlation map from the combined detection data."""
        if self.combined is None:
            return None
        return self.combined.detection.correlation_map

    @property
    def mean_image_channel_2(self) -> NDArray[np.float32] | None:
        """Returns the channel 2 mean image from the combined detection data."""
        if self.combined is None:
            return None
        return self.combined.detection.mean_image_channel_2

    @property
    def enhanced_mean_image_channel_2(self) -> NDArray[np.float32] | None:
        """Returns the channel 2 enhanced mean image from the combined detection data."""
        if self.combined is None:
            return None
        return self.combined.detection.enhanced_mean_image_channel_2

    @property
    def corrected_structural_mean_image(self) -> NDArray[np.float32] | None:
        """Returns the corrected structural channel mean image."""
        if self.combined is None:
            return None
        return self.combined.extraction.corrected_structural_mean_image

    @property
    def subtracted_fluorescence(self) -> NDArray[np.float32] | None:
        """Returns the memory-mapped baseline-and-neuropil-subtracted fluorescence traces."""
        if self._subtracted_fluorescence_map is not None:
            return self._subtracted_fluorescence_map
        save = self.output_path
        if save is None:
            return None
        path = save / "subtracted_fluorescence.npy"
        if not path.exists():
            return None
        self._subtracted_fluorescence_map = np.load(path, mmap_mode="r")
        return self._subtracted_fluorescence_map

    @property
    def subtracted_fluorescence_channel_2(self) -> NDArray[np.float32] | None:
        """Returns the memory-mapped channel 2 subtracted fluorescence traces."""
        if self._subtracted_fluorescence_channel_2_map is not None:
            return self._subtracted_fluorescence_channel_2_map
        save = self.output_path
        if save is None:
            return None
        path = save / "subtracted_fluorescence_channel_2.npy"
        if not path.exists():
            return None
        self._subtracted_fluorescence_channel_2_map = np.load(path, mmap_mode="r")
        return self._subtracted_fluorescence_channel_2_map

    @property
    def frame_count(self) -> int:
        """Returns the total number of imaging frames."""
        if self.cell_fluorescence.size == 0:
            return 0
        return int(self.cell_fluorescence.shape[1])

    @property
    def cell_diameter(self) -> int:
        """Returns the estimated cell diameter in pixels."""
        if self.combined is None:
            return 0
        return self.combined.detection.cell_diameter

    @property
    def aspect_ratio(self) -> float:
        """Returns the aspect ratio of detected cells."""
        if self.combined is None:
            return 0.0
        return self.combined.detection.aspect_ratio

    @property
    def registered_binary_paths(self) -> tuple[Path, ...]:
        """Returns the channel 1 registered binary file paths, one per plane."""
        if self.combined is None:
            return ()
        return self.combined.registered_binary_paths

    @property
    def registered_binary_paths_channel_2(self) -> tuple[Path, ...] | None:
        """Returns the channel 2 registered binary file paths, one per plane."""
        if self.combined is None:
            return None
        return self.combined.registered_binary_paths_channel_2

    @property
    def plane_y_offsets(self) -> NDArray[np.int32]:
        """Returns the per-plane y-axis offsets for the combined view."""
        if self.combined is None:
            return np.array([], dtype=np.int32)
        return self.combined.plane_y_offsets

    @property
    def plane_x_offsets(self) -> NDArray[np.int32]:
        """Returns the per-plane x-axis offsets for the combined view."""
        if self.combined is None:
            return np.array([], dtype=np.int32)
        return self.combined.plane_x_offsets

    @property
    def plane_heights(self) -> NDArray[np.uint16]:
        """Returns the per-plane frame heights in pixels."""
        if self.combined is None:
            return np.array([], dtype=np.uint16)
        return self.combined.plane_heights

    @property
    def plane_widths(self) -> NDArray[np.uint16]:
        """Returns the per-plane frame widths in pixels."""
        if self.combined is None:
            return np.array([], dtype=np.uint16)
        return self.combined.plane_widths

    @property
    def configuration(self) -> SingleDayConfiguration | None:
        """Returns the pipeline configuration used to produce this data."""
        if self.contexts:
            return self.contexts[0].configuration
        return None

    @property
    def neuropil_coefficient(self) -> float:
        """Returns the neuropil subtraction scaling factor from the configuration."""
        configuration = self.configuration
        if configuration is not None:
            return configuration.spike_deconvolution.neuropil_coefficient
        return 0.7

    @property
    def extraction_batch_size(self) -> int:
        """Returns the signal extraction batch size from the configuration."""
        configuration = self.configuration
        if configuration is not None:
            return configuration.signal_extraction.batch_size
        return 500

    @property
    def baseline_method(self) -> BaselineMethod | str:
        """Returns the baseline computation method from the configuration."""
        configuration = self.configuration
        if configuration is not None:
            return configuration.spike_deconvolution.baseline_method
        return "maximin"

    @property
    def baseline_window(self) -> float:
        """Returns the baseline sliding window size in seconds."""
        configuration = self.configuration
        if configuration is not None:
            return configuration.spike_deconvolution.baseline_window
        return 60.0

    @property
    def baseline_sigma(self) -> float:
        """Returns the baseline Gaussian smoothing sigma in frames."""
        configuration = self.configuration
        if configuration is not None:
            return configuration.spike_deconvolution.baseline_sigma
        return 10.0

    @property
    def baseline_percentile(self) -> float:
        """Returns the baseline percentile for the constant_percentile method."""
        configuration = self.configuration
        if configuration is not None:
            return configuration.spike_deconvolution.baseline_percentile
        return 8.0

    @property
    def allow_overlap(self) -> bool:
        """Returns whether overlapping pixels are included in signal extraction."""
        configuration = self.configuration
        if configuration is not None:
            return configuration.signal_extraction.allow_overlap
        return False

    @property
    def crop_to_soma(self) -> bool:
        """Returns whether dendritic regions are cropped before classification."""
        if self.contexts:
            return self.contexts[0].configuration.roi_detection.crop_to_soma
        return True

    @property
    def cell_probability_percentile(self) -> int:
        """Returns the percentile threshold for cell vs neuropil pixel classification."""
        configuration = self.configuration
        if configuration is not None:
            return configuration.signal_extraction.cell_probability_percentile
        return 50

    @property
    def inner_neuropil_border_radius(self) -> int:
        """Returns the exclusion zone width between cell ROI and neuropil mask in pixels."""
        configuration = self.configuration
        if configuration is not None:
            return configuration.signal_extraction.inner_neuropil_border_radius
        return 2

    @property
    def minimum_neuropil_pixels(self) -> int:
        """Returns the minimum pixel count required for each neuropil mask."""
        configuration = self.configuration
        if configuration is not None:
            return configuration.signal_extraction.minimum_neuropil_pixels
        return 350

    @property
    def output_path(self) -> Path | None:
        """Returns the root output directory path."""
        if self.contexts:
            return self.contexts[0].configuration.file_io.output_path
        return None

    @property
    def data_path(self) -> Path | None:
        """Returns the root input data directory path."""
        if self.contexts:
            return self.contexts[0].configuration.file_io.data_path
        return None

    @property
    def valid_y_range(self) -> tuple[int, int]:
        """Returns the valid Y pixel range from the first plane's registration."""
        if self.contexts:
            return self.contexts[0].runtime.registration.valid_y_range
        return 0, 0

    @property
    def valid_x_range(self) -> tuple[int, int]:
        """Returns the valid X pixel range from the first plane's registration."""
        if self.contexts:
            return self.contexts[0].runtime.registration.valid_x_range
        return 0, 0

    @property
    def basename(self) -> str:
        """Returns the output directory name for display and file operations."""
        path = self.output_path
        if path is not None:
            return path.name
        return ""

    @classmethod
    def from_single_day(cls, root_path: Path, *, mutable: bool = False) -> ROIViewerData:
        """Loads single-day pipeline data into an ROIViewerData wrapper.

        Loads all planes via ``RuntimeContext.load()`` and the combined data from the root cindra directory. Large trace
        arrays are memory-mapped from disk rather than copied into RAM.

        Args:
            root_path: Root cindra output directory containing configuration.yaml.
            mutable: Determines whether arrays are opened with copy-on-write access (True, for the ROI editor) or
                read-only access (False, for the read-only ROI viewer).

        Returns:
            A fully populated ROIViewerData instance.
        """
        contexts = RuntimeContext.load(root_path=root_path, plane_index=-1)
        if not isinstance(contexts, list):
            contexts = [contexts]

        combined = CombinedData.load(root_path=root_path)
        combined.load_results(root_path=root_path)

        instance = cls(contexts=contexts, combined=combined)
        instance._mutable = mutable
        instance._load_arrays(combined=combined)
        return instance

    @classmethod
    def from_dialog(cls, *, mutable: bool = False) -> ROIViewerData | None:
        """Opens a file dialog to select a cindra output directory and loads data from it.

        Args:
            mutable: Determines whether arrays are opened with copy-on-write access (True, for the ROI editor) or
                read-only access (False, for the read-only ROI viewer).

        Returns:
            A fully populated ROIViewerData instance, or None if the dialog was canceled.
        """
        name = QFileDialog.getExistingDirectory(caption="Open cindra output directory")
        if not name:
            return None

        return cls.from_single_day(root_path=Path(name), mutable=mutable)

    def _load_arrays(self, combined: CombinedData | None) -> None:
        """Loads arrays from the given combined data using the current mutability mode.

        Large trace arrays (fluorescence, spikes) are memory-mapped from disk. When mutable is False, arrays receive
        read-only access; when True, copy-on-write access. Small arrays (classification, colocalization) are copied.
        After memory-mapping, the extraction object's in-memory trace copies are released to avoid double-storing the
        data.

        Args:
            combined: The source combined data to load arrays from.
        """
        self.combined = combined

        if combined is None:
            return

        extraction = combined.extraction
        output_path = self.output_path
        mmap_mode = "c" if self._mutable else "r"

        # Copies ROI statistics list (shallow copy).
        if extraction.roi_statistics is not None:
            self.roi_statistics = list(extraction.roi_statistics)
        else:
            self.roi_statistics = []

        roi_count = len(self.roi_statistics)
        default_trace_shape = (roi_count, 0)

        # Memory-maps fluorescence traces.
        self.cell_fluorescence = _memory_map_trace(
            output_path=output_path,
            file_name="cell_fluorescence.npy",
            fallback=extraction.cell_fluorescence,
            default_shape=default_trace_shape,
            mmap_mode=mmap_mode,
        )
        self.neuropil_fluorescence = _memory_map_trace(
            output_path=output_path,
            file_name="neuropil_fluorescence.npy",
            fallback=extraction.neuropil_fluorescence,
            default_shape=default_trace_shape,
            mmap_mode=mmap_mode,
        )
        self.spikes = _memory_map_trace(
            output_path=output_path,
            file_name="spikes.npy",
            fallback=extraction.spikes,
            default_shape=default_trace_shape,
            mmap_mode=mmap_mode,
        )

        # Memory-maps optional channel 2 traces.
        self.cell_fluorescence_channel_2 = _memory_map_optional_trace(
            output_path=output_path,
            file_name="cell_fluorescence_channel_2.npy",
            fallback=extraction.cell_fluorescence_channel_2,
            mmap_mode=mmap_mode,
        )
        self.neuropil_fluorescence_channel_2 = _memory_map_optional_trace(
            output_path=output_path,
            file_name="neuropil_fluorescence_channel_2.npy",
            fallback=extraction.neuropil_fluorescence_channel_2,
            mmap_mode=mmap_mode,
        )

        # Invalidates memory-mapped subtracted fluorescence caches.
        self._subtracted_fluorescence_map = None
        self._subtracted_fluorescence_channel_2_map = None

        # Releases large trace arrays from the extraction object.
        _release_trace_arrays(extraction=extraction)

        # Copies classification arrays (small, one value per ROI).
        if extraction.cell_classification is not None:
            self.cell_classification_probabilities = extraction.cell_classification[:, 0].copy()
            self.cell_classification_labels = extraction.cell_classification[:, 1].astype(np.bool_).copy()
            if not self._mutable:
                self.cell_classification_probabilities.flags.writeable = False
                self.cell_classification_labels.flags.writeable = False
        else:
            self.cell_classification_probabilities = np.ones(roi_count, dtype=np.float32)
            self.cell_classification_labels = np.ones(roi_count, dtype=np.bool_)
            if not self._mutable:
                self.cell_classification_probabilities.flags.writeable = False
                self.cell_classification_labels.flags.writeable = False

        # Copies colocalization arrays (small, one value per ROI).
        if extraction.cell_colocalization is not None:
            self.cell_colocalization_probabilities = extraction.cell_colocalization[:, 0].copy()
            self.cell_colocalization_labels = extraction.cell_colocalization[:, 1].astype(np.bool_).copy()
            self.has_channel_2 = True
            if not self._mutable:
                self.cell_colocalization_probabilities.flags.writeable = False
                self.cell_colocalization_labels.flags.writeable = False
        else:
            self.cell_colocalization_probabilities = np.zeros(roi_count, dtype=np.float32)
            self.cell_colocalization_labels = np.zeros(roi_count, dtype=np.bool_)
            self.has_channel_2 = combined.detection.mean_image_channel_2 is not None
            if not self._mutable:
                self.cell_colocalization_probabilities.flags.writeable = False
                self.cell_colocalization_labels.flags.writeable = False

        # Initializes the not-merged mask (only meaningful for mutable mode).
        if self._mutable:
            self.not_merged = np.ones(roi_count, dtype=np.bool_)


@dataclass
class RegistrationViewerData:
    """Wraps single-day per-plane registration data and serves it to the registration viewer GUI.

    Stores the data from all imaging planes of the processed recording and supports switching between planes for
    independent registration quality review, since each plane is processed and registered independently.

    Notes:
        The backing ``RuntimeContext`` instances are stored privately. All public properties delegate to the current
        plane's ``IOData``, ``RegistrationData``, and ``DetectionData`` without exposing extraction, timing, or
        configuration internals.

        This class manages ``BinaryFile`` handles for the current plane. Call ``close()`` to release file handles when
        the instance is no longer needed.
    """

    _contexts: list[RuntimeContext] = field(repr=False)
    """The RuntimeContext instances for each registered imaging plane of the processed recording."""

    _current_plane_index: int = 0
    """The index of the plane whose data is currently displayed by the GUI application."""

    _binary_file: BinaryFile | None = field(default=None, repr=False)
    """Lazily opened BinaryFile for the current plane's primary registered binary."""

    _binary_file_channel_2: BinaryFile | None = field(default=None, repr=False)
    """Lazily opened BinaryFile for the current plane's channel 2 registered binary, or None if absent."""

    @property
    def current_plane_index(self) -> int:
        """Returns the index of the plane whose data is currently displayed by the GUI application."""
        return self._current_plane_index

    @property
    def plane_count(self) -> int:
        """Returns the total number of imaging planes in the processed recording."""
        return len(self._contexts)

    @property
    def plane_labels(self) -> list[str]:
        """Returns display labels for all planes in the processed recording."""
        return [f"Plane {context.runtime.io.plane_index}" for context in self._contexts]

    @property
    def frame_height(self) -> int:
        """Returns the height of each frame in pixels for the current plane."""
        return self._current_io.frame_height

    @property
    def frame_width(self) -> int:
        """Returns the width of each frame in pixels for the current plane."""
        return self._current_io.frame_width

    @property
    def frame_count(self) -> int:
        """Returns the total number of frames written to the binary file for the current plane."""
        return self._current_io.frame_count

    @property
    def sampling_rate(self) -> float:
        """Returns the per-plane sampling rate in Hertz for the current plane."""
        return self._current_io.sampling_rate

    @property
    def binary_file(self) -> BinaryFile:
        """Returns a read-only BinaryFile for the current plane's primary registered binary.

        The file is opened lazily on first access and cached until the next plane switch.
        """
        if self._binary_file is None:
            path = self._current_io.registered_binary_path
            if path is None or not path.is_file():
                message = "No registered binary found for this plane."
                console.error(message=message, error=FileNotFoundError)
            self._binary_file = BinaryFile(
                height=self.frame_height,
                width=self.frame_width,
                file_path=path,
                read_only=True,
            )
        return self._binary_file

    @property
    def binary_file_channel_2(self) -> BinaryFile | None:
        """Returns a read-only BinaryFile for the current plane's channel 2 registered binary, or None if absent."""
        if self._binary_file_channel_2 is None:
            path = self._current_io.registered_binary_path_channel_2
            if path is None or not path.is_file():
                return None
            self._binary_file_channel_2 = BinaryFile(
                height=self.frame_height,
                width=self.frame_width,
                file_path=path,
                read_only=True,
            )
        return self._binary_file_channel_2

    @property
    def has_channel_2(self) -> bool:
        """Determines whether a channel 2 registered binary exists for the current plane."""
        path = self._current_io.registered_binary_path_channel_2
        return path is not None and path.is_file()

    @property
    def has_nonrigid(self) -> bool:
        """Determines whether nonrigid registration data exists for the current plane."""
        return (
            self._current_registration.nonrigid_y_offsets is not None
            and self._current_registration.nonrigid_x_offsets is not None
        )

    @property
    def recording_label(self) -> str:
        """Returns the trailing components of the recording data path for display labels.

        Starts with the last 3 path components and progressively reduces to 2, then 1, if the label exceeds 45
        characters.
        """
        data_path = self._contexts[self._current_plane_index].configuration.file_io.data_path
        if data_path is not None:
            parts = data_path.parts
            max_characters = 45
            # Tries 3, then 2, then 1 trailing component(s) until the label fits.
            for count in (3, 2, 1):
                if len(parts) >= count:
                    label = str(Path(*parts[-count:]))
                    if len(label) <= max_characters:
                        return label
            return str(data_path.name)
        return ""

    @property
    def output_path(self) -> Path | None:
        """Returns the path to the plane-specific output directory where all results are saved."""
        return self._current_io.output_path

    @property
    def rigid_y_offsets(self) -> NDArray[np.int32]:
        """Returns the vertical (Y) translation offsets from rigid registration, one value per frame.

        Returns a zero array with shape (frame_count,) when the underlying data is None.
        """
        offsets = self._current_registration.rigid_y_offsets
        if offsets is not None:
            return offsets
        return np.zeros((self.frame_count,), dtype=np.int32)

    @property
    def rigid_x_offsets(self) -> NDArray[np.int32]:
        """Returns the horizontal (X) translation offsets from rigid registration, one value per frame.

        Returns a zero array with shape (frame_count,) when the underlying data is None.
        """
        offsets = self._current_registration.rigid_x_offsets
        if offsets is not None:
            return offsets
        return np.zeros((self.frame_count,), dtype=np.int32)

    @property
    def nonrigid_y_offsets(self) -> NDArray[np.float32] | None:
        """Returns the vertical (Y) translation offsets from nonrigid registration, per frame and per block."""
        return self._current_registration.nonrigid_y_offsets

    @property
    def nonrigid_x_offsets(self) -> NDArray[np.float32] | None:
        """Returns the horizontal (X) translation offsets from nonrigid registration, per frame and per block."""
        return self._current_registration.nonrigid_x_offsets

    @property
    def nonrigid_rms(self) -> NDArray[np.float32] | None:
        """Returns the pre-computed Root Mean Square of nonrigid offsets, or None when no nonrigid data exists.

        Computed as ``sqrt(mean(y^2 + x^2, axis=1))``.
        """
        nonrigid_y = self._current_registration.nonrigid_y_offsets
        nonrigid_x = self._current_registration.nonrigid_x_offsets
        if nonrigid_y is None or nonrigid_x is None:
            return None
        return np.sqrt(np.mean(nonrigid_y.astype(np.float32) ** 2 + nonrigid_x.astype(np.float32) ** 2, axis=1)).astype(
            np.float32
        )

    @property
    def principal_component_extreme_images(self) -> NDArray[np.float32] | None:
        """Returns the mean images from frames at extreme ends of each principal component.

        The returned array has shape (2, num_components, height, width). Index 0 contains low-projection means,
        index 1 contains high-projection means.
        """
        return self._current_registration.principal_component_extreme_images

    @property
    def principal_component_shift_metrics(self) -> NDArray[np.float32] | None:
        """Returns the registration shift metrics computed by aligning PC extreme images.

        The returned array has shape (num_components, 3). Column 0 contains mean rigid shift magnitude, column 1
        contains mean nonrigid shift magnitude, and column 2 contains maximum nonrigid shift magnitude.
        """
        return self._current_registration.principal_component_shift_metrics

    @property
    def principal_component_projections(self) -> NDArray[np.float32] | None:
        """Returns the projection of each frame onto the principal components of the registered movie.

        The returned array has shape (num_frames, num_components).
        """
        return self._current_registration.principal_component_projections

    @property
    def principal_component_count(self) -> int:
        """Returns the number of principal components used for registration quality metrics."""
        return self._contexts[
            self._current_plane_index
        ].configuration.registration.registration_metric_principal_components

    @property
    def aspect_ratio(self) -> float:
        """Returns the aspect ratio of detected ROIs, computed as vertical to horizontal diameter ratio."""
        return self._current_detection.aspect_ratio

    @classmethod
    def from_recording(cls, root_path: Path) -> RegistrationViewerData:
        """Loads the registration data for all imaging planes of the target recording for registration review.

        Args:
            root_path: The path to the root single-day pipeline output directory for the processed recording.

        Returns:
            A fully populated RegistrationViewerData instance.
        """
        contexts = RuntimeContext.load(root_path=root_path, plane_index=-1)
        if not isinstance(contexts, list):
            contexts = [contexts]
        return cls(_contexts=contexts)

    def switch_plane(self, plane_index: int) -> None:
        """Switches the data view to a different imaging plane.

        Closes any open BinaryFile handles for the previous plane so they reopen lazily for the new plane.

        Args:
            plane_index: The index of the plane to switch to.

        Raises:
            ValueError: If the index is out of range.
        """
        if plane_index < 0 or plane_index >= len(self._contexts):
            message = (
                f"Unable to switch the registration viewer to plane {plane_index}. Valid range is 0 to "
                f"{len(self._contexts) - 1}."
            )
            console.error(message=message, error=ValueError)
        self._close_binary_files()
        self._current_plane_index = plane_index

    def close(self) -> None:
        """Closes any open BinaryFile handles managed by this instance."""
        self._close_binary_files()

    def _close_binary_files(self) -> None:
        """Closes and resets cached BinaryFile handles."""
        if self._binary_file is not None:
            self._binary_file.close()
            self._binary_file = None
        if self._binary_file_channel_2 is not None:
            self._binary_file_channel_2.close()
            self._binary_file_channel_2 = None

    @property
    def _current_io(self) -> IOData:
        """Returns the IOData instance for the currently selected plane."""
        return self._contexts[self._current_plane_index].runtime.io

    @property
    def _current_registration(self) -> RegistrationData:
        """Returns the RegistrationData instance for the currently selected plane."""
        return self._contexts[self._current_plane_index].runtime.registration

    @property
    def _current_detection(self) -> DetectionData:
        """Returns the DetectionData instance for the currently selected plane."""
        return self._contexts[self._current_plane_index].runtime.detection
