"""Provides the read-only ROIViewerData dataclass that wraps pipeline data for viewer consumption."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from dataclasses import field, dataclass

import numpy as np
from PySide6.QtWidgets import QFileDialog

from ...dataclasses import CombinedData, RuntimeContext

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ...dataclasses import ROIStatistics, ExtractionData


def _memory_map_trace(
    output_path: Path | None,
    file_name: str,
    fallback: NDArray[np.float32] | None,
    default_shape: tuple[int, ...],
) -> NDArray[np.float32]:
    """Memory-maps a trace array from disk with read-only access.

    Attempts to open the specified .npy file as a read-only memory map. Falls back to a read-only view
    of the provided array if the file does not exist, or returns a zero array of the given shape if
    neither source is available.

    Args:
        output_path: Directory containing the .npy file.
        file_name: Name of the .npy file to memory-map.
        fallback: Array to use if the file does not exist on disk.
        default_shape: Shape for a zero-initialized fallback array.

    Returns:
        The memory-mapped, viewed, or zero-initialized array.
    """
    if output_path is not None:
        path = output_path / file_name
        if path.exists():
            mapped: NDArray[np.float32] = np.load(path, mmap_mode="r")
            return mapped
    if fallback is not None:
        result: NDArray[np.float32] = fallback.view()
        result.flags.writeable = False
        return result
    return np.zeros(default_shape, dtype=np.float32)


def _memory_map_optional_trace(
    output_path: Path | None,
    file_name: str,
    fallback: NDArray[np.float32] | None,
) -> NDArray[np.float32] | None:
    """Memory-maps an optional trace array from disk with read-only access.

    Attempts to open the specified .npy file as a read-only memory map. Falls back to a read-only
    view of the provided array if the file does not exist. Returns None if neither source is available.

    Args:
        output_path: Directory containing the .npy file.
        file_name: Name of the .npy file to memory-map.
        fallback: Array to use if the file does not exist on disk.

    Returns:
        The memory-mapped or viewed array, or None if unavailable.
    """
    if output_path is not None:
        path = output_path / file_name
        if path.exists():
            mapped: NDArray[np.float32] = np.load(path, mmap_mode="r")
            return mapped
    if fallback is not None:
        result: NDArray[np.float32] = fallback.view()
        result.flags.writeable = False
        return result
    return None


def _release_trace_arrays(extraction: ExtractionData) -> None:
    """Releases large trace arrays from the extraction object to free memory.

    Called after memory-mapping the same data from disk, so the in-memory copies held by the
    extraction object are no longer needed.

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
    """Wraps imported single-day pipeline data for read-only viewer consumption.

    Holds read-only references to pipeline arrays and delegates access to the underlying context
    objects. Large trace arrays are memory-mapped from disk with read-only access to minimize
    memory usage.

    Notes:
        Single-day data comes from ``RuntimeContext.load()`` which provides ``CombinedData``
        containing detection images and extraction results.
    """

    # Underlying contexts for all planes.
    contexts: list[RuntimeContext] = field(default_factory=list)
    """Single-day runtime contexts for all processed recording's imaging planes."""

    # Combined data from the cindra output directory.
    combined: CombinedData | None = None
    """The combined single-day data for the active view."""

    # Read-only ROI data.
    roi_statistics: list[ROIStatistics] = field(default_factory=list)
    """Spatial and shape statistics for each detected ROI."""

    cell_fluorescence: NDArray[np.float32] = field(default_factory=lambda: np.array([], dtype=np.float32))
    """Cell fluorescence traces with shape (cells, frames). Read-only memory-mapped."""

    neuropil_fluorescence: NDArray[np.float32] = field(default_factory=lambda: np.array([], dtype=np.float32))
    """Neuropil fluorescence traces with shape (cells, frames). Read-only memory-mapped."""

    spikes: NDArray[np.float32] = field(default_factory=lambda: np.array([], dtype=np.float32))
    """Deconvolved spike traces with shape (cells, frames). Read-only memory-mapped."""

    cell_classification_labels: NDArray[np.bool_] = field(default_factory=lambda: np.array([], dtype=np.bool_))
    """Boolean classification array marking each ROI as cell or non-cell. Read-only."""

    cell_classification_probabilities: NDArray[np.float32] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    """Classifier probability for each ROI being a cell. Read-only."""

    cell_colocalization_labels: NDArray[np.bool_] = field(default_factory=lambda: np.array([], dtype=np.bool_))
    """Boolean array marking each ROI as colocalized with a channel 2 fluorescence source. Read-only."""

    cell_colocalization_probabilities: NDArray[np.float32] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    """Classifier probability of each ROI being colocalized with a channel 2 fluorescence source. Read-only."""

    has_channel_2: bool = False
    """Determines whether channel 2 data is available."""

    # Channel 2 traces (optional, memory-mapped when available).
    cell_fluorescence_channel_2: NDArray[np.float32] | None = None
    """Channel 2 cell fluorescence traces. None if single-channel."""

    neuropil_fluorescence_channel_2: NDArray[np.float32] | None = None
    """Channel 2 neuropil fluorescence traces. None if single-channel."""

    # Private caches for memory-mapped subtracted fluorescence arrays.
    _subtracted_fluorescence_map: NDArray[np.float32] | None = field(init=False, default=None, repr=False)
    """Cached memory-mapped channel 1 subtracted fluorescence."""

    _subtracted_fluorescence_channel_2_map: NDArray[np.float32] | None = field(init=False, default=None, repr=False)
    """Cached memory-mapped channel 2 subtracted fluorescence."""

    # Read-only properties delegated to the combined data object.

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
    def from_single_day(cls, root_path: Path) -> ROIViewerData:
        """Loads single-day pipeline data into a read-only ROIViewerData wrapper.

        Loads all planes via ``RuntimeContext.load()`` and the combined data from the root cindra
        directory. Large trace arrays are memory-mapped from disk with read-only access rather than
        copied into RAM.

        Args:
            root_path: Root cindra output directory containing configuration.yaml.

        Returns:
            A fully populated ROIViewerData instance in single-day mode.
        """
        contexts = RuntimeContext.load(root_path=root_path, plane_index=-1)
        if not isinstance(contexts, list):
            contexts = [contexts]

        combined = CombinedData.load(root_path=root_path)
        combined.load_results(root_path=root_path)

        instance = cls(contexts=contexts, combined=combined)
        instance._load_arrays(combined=combined)
        return instance

    @classmethod
    def from_dialog(cls) -> ROIViewerData | None:
        """Opens a file dialog to select a cindra output directory and loads data from it.

        Returns:
            A fully populated ROIViewerData instance, or None if the dialog was canceled.
        """
        name = QFileDialog.getExistingDirectory(caption="Open cindra output directory")
        if not name:
            return None

        return cls.from_single_day(root_path=Path(name))

    def _load_arrays(self, combined: CombinedData | None) -> None:
        """Loads read-only arrays from the given combined data.

        Large trace arrays (fluorescence, spikes) are memory-mapped from disk with read-only access
        to minimize memory usage. Small arrays (classification, colocalization) are copied as
        read-only. After memory-mapping, the extraction object's in-memory trace copies are released
        to avoid double-storing the data.

        Args:
            combined: The source combined data to load arrays from.
        """
        self.combined = combined

        if combined is None:
            return

        extraction = combined.extraction
        output_path = self.output_path

        # Copies ROI statistics list (shallow copy; individual ROIStatistics are read-only for viewer).
        if extraction.roi_statistics is not None:
            self.roi_statistics = list(extraction.roi_statistics)
        else:
            self.roi_statistics = []

        roi_count = len(self.roi_statistics)
        default_trace_shape = (roi_count, 0)

        # Memory-maps fluorescence traces with read-only access.
        self.cell_fluorescence = _memory_map_trace(
            output_path=output_path,
            file_name="cell_fluorescence.npy",
            fallback=extraction.cell_fluorescence,
            default_shape=default_trace_shape,
        )
        self.neuropil_fluorescence = _memory_map_trace(
            output_path=output_path,
            file_name="neuropil_fluorescence.npy",
            fallback=extraction.neuropil_fluorescence,
            default_shape=default_trace_shape,
        )
        self.spikes = _memory_map_trace(
            output_path=output_path,
            file_name="spikes.npy",
            fallback=extraction.spikes,
            default_shape=default_trace_shape,
        )

        # Memory-maps optional channel 2 traces.
        self.cell_fluorescence_channel_2 = _memory_map_optional_trace(
            output_path=output_path,
            file_name="cell_fluorescence_channel_2.npy",
            fallback=extraction.cell_fluorescence_channel_2,
        )
        self.neuropil_fluorescence_channel_2 = _memory_map_optional_trace(
            output_path=output_path,
            file_name="neuropil_fluorescence_channel_2.npy",
            fallback=extraction.neuropil_fluorescence_channel_2,
        )

        # Invalidates memory-mapped subtracted fluorescence caches.
        self._subtracted_fluorescence_map = None
        self._subtracted_fluorescence_channel_2_map = None

        # Releases large trace arrays from the extraction object.
        _release_trace_arrays(extraction=extraction)

        # Copies classification arrays as read-only (small, one value per ROI).
        if extraction.cell_classification is not None:
            self.cell_classification_probabilities = extraction.cell_classification[:, 0].copy()
            self.cell_classification_probabilities.flags.writeable = False
            self.cell_classification_labels = extraction.cell_classification[:, 1].astype(np.bool_).copy()
            self.cell_classification_labels.flags.writeable = False
        else:
            self.cell_classification_probabilities = np.ones(roi_count, dtype=np.float32)
            self.cell_classification_probabilities.flags.writeable = False
            self.cell_classification_labels = np.ones(roi_count, dtype=np.bool_)
            self.cell_classification_labels.flags.writeable = False

        # Copies colocalization arrays as read-only (small, one value per ROI).
        if extraction.cell_colocalization is not None:
            self.cell_colocalization_probabilities = extraction.cell_colocalization[:, 0].copy()
            self.cell_colocalization_probabilities.flags.writeable = False
            self.cell_colocalization_labels = extraction.cell_colocalization[:, 1].astype(np.bool_).copy()
            self.cell_colocalization_labels.flags.writeable = False
            self.has_channel_2 = True
        else:
            self.cell_colocalization_probabilities = np.zeros(roi_count, dtype=np.float32)
            self.cell_colocalization_probabilities.flags.writeable = False
            self.cell_colocalization_labels = np.zeros(roi_count, dtype=np.bool_)
            self.cell_colocalization_labels.flags.writeable = False
            self.has_channel_2 = combined.detection.mean_image_channel_2 is not None
