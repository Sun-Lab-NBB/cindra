"""Provides the ContextData dataclass that wraps pipeline data for GUI consumption."""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import field, dataclass

import numpy as np
from ataraxis_base_utilities import console

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from ..dataclasses import (
        CombinedData,
        BaselineMethod,
        RuntimeContext,
        MultiDayConfiguration,
        SingleDayConfiguration,
    )
    from ..dataclasses.multi_day_data import MultiDayRuntimeData
    from ..dataclasses.single_day_data import ROIStatistics
    from ..dataclasses.runtime_contexts import MultiDayRuntimeContext


@dataclass
class ContextData:
    """Wraps pipeline data for both single-day and multi-day modes.

    Holds mutable copies of GUI-editable arrays and delegates read-only access to the
    underlying context objects. For multi-day mode, additionally provides session
    navigation, multiple mask sets, and transformed reference images.

    Notes:
        Single-day data comes from ``RuntimeContext.load()`` which provides ``CombinedData``
        containing detection images and extraction results. Multi-day data comes from
        ``MultiDayRuntimeContext.load()`` which provides per-session ``MultiDayRuntimeData``
        wrapping the single-day ``CombinedData`` along with registration and tracking data.
    """

    # Mode flag.
    is_multi_day: bool
    """Determines whether the data was loaded from a multi-day pipeline output."""

    # Underlying contexts (one set will be None depending on mode).
    single_day_contexts: list[RuntimeContext] | None = None
    """Single-day runtime contexts for all planes. None in multi-day mode."""

    multi_day_contexts: list[MultiDayRuntimeContext] | None = None
    """Multi-day runtime contexts for all sessions. None in single-day mode."""

    # Combined data is always present (single-day combined, or current session's combined).
    combined: CombinedData | None = None
    """The combined single-day data for the active view."""

    # Multi-day session navigation.
    current_session_index: int = 0
    """Zero-based index of the currently displayed multi-day session."""

    multi_day_runtime: MultiDayRuntimeData | None = None
    """Runtime data for the current multi-day session. None in single-day mode."""

    # Mutable arrays extracted from combined/extraction data. These are copies that the GUI
    # can modify (e.g. reclassifying cells, merging ROIs) without mutating the source objects.
    roi_statistics: list[ROIStatistics] = field(default_factory=list)
    """Spatial and shape statistics for each detected ROI."""

    cell_fluorescence: NDArray[np.float32] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    """Cell fluorescence traces with shape (cells, frames)."""

    neuropil_fluorescence: NDArray[np.float32] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    """Neuropil fluorescence traces with shape (cells, frames)."""

    spikes: NDArray[np.float32] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    """Deconvolved spike traces with shape (cells, frames)."""

    cell_classification_labels: NDArray[np.bool_] = field(
        default_factory=lambda: np.array([], dtype=np.bool_)
    )
    """Boolean classification array marking each ROI as cell or non-cell."""

    cell_classification_probabilities: NDArray[np.float32] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    """Classifier probability for each ROI being a cell."""

    cell_colocalization_labels: NDArray[np.bool_] = field(
        default_factory=lambda: np.array([], dtype=np.bool_)
    )
    """Boolean classification array marking each ROI as a red (channel 2) cell."""

    cell_colocalization_probabilities: NDArray[np.float32] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    """Probability of each ROI being a red (channel 2) cell."""

    has_red_channel: bool = False
    """Determines whether channel 2 (red) data is available."""

    not_merged: NDArray[np.bool_] = field(
        default_factory=lambda: np.array([], dtype=np.bool_)
    )
    """Boolean mask tracking which ROIs have not been merged into other ROIs."""

    # Channel 2 traces (optional).
    cell_fluorescence_channel_2: NDArray[np.float32] | None = None
    """Channel 2 cell fluorescence traces. None if single-channel."""

    neuropil_fluorescence_channel_2: NDArray[np.float32] | None = None
    """Channel 2 neuropil fluorescence traces. None if single-channel."""

    # Behavior data (loaded separately via context_loader).
    behavior: NDArray[np.float32] | None = None
    """Loaded 1D behavioral trace."""

    behavior_time: NDArray[np.float32] | None = None
    """Time axis for the behavioral trace."""

    behavior_resampled: NDArray[np.float32] | None = None
    """Behavioral trace resampled to match imaging frame rate."""

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

    # Background images from combined detection data.

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

    # Multi-day only: transformed images from the registration data.

    @property
    def transformed_mean_image(self) -> NDArray[np.float32] | None:
        """Returns the mean image in deformed (registered) space."""
        if self.multi_day_runtime is None:
            return None
        return self.multi_day_runtime.registration.transformed_mean_image

    @property
    def transformed_enhanced_mean_image(self) -> NDArray[np.float32] | None:
        """Returns the enhanced mean image in deformed (registered) space."""
        if self.multi_day_runtime is None:
            return None
        return self.multi_day_runtime.registration.transformed_enhanced_mean_image

    @property
    def transformed_maximum_projection(self) -> NDArray[np.float32] | None:
        """Returns the maximum projection in deformed (registered) space."""
        if self.multi_day_runtime is None:
            return None
        return self.multi_day_runtime.registration.transformed_maximum_projection

    # Multi-day only: mask sets.

    @property
    def session_count(self) -> int:
        """Returns the number of multi-day sessions."""
        if self.multi_day_contexts is None:
            return 0
        return len(self.multi_day_contexts)

    @property
    def session_ids(self) -> list[str]:
        """Returns the session identifier strings for all multi-day sessions."""
        if self.multi_day_contexts is None:
            return []
        return [context.runtime.io.session_id for context in self.multi_day_contexts]

    @property
    def deformed_masks(self) -> list[ROIStatistics] | None:
        """Returns the registered (deformed) cell masks for the current session."""
        if self.multi_day_runtime is None:
            return None
        return self.multi_day_runtime.registration.deformed_cell_masks

    @property
    def template_masks(self) -> list[ROIStatistics] | None:
        """Returns the shared template masks from cross-session tracking."""
        if self.multi_day_runtime is None:
            return None
        return self.multi_day_runtime.tracking.template_masks

    # Properties from CombinedData (geometry and binary paths).

    @property
    def frame_count(self) -> int:
        """Returns the total number of imaging frames."""
        if self.cell_fluorescence.size == 0:
            return 0
        return self.cell_fluorescence.shape[1]

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

    # Configuration access.

    @property
    def configuration(self) -> SingleDayConfiguration | MultiDayConfiguration | None:
        """Returns the pipeline configuration used to produce this data."""
        if self.single_day_contexts is not None and self.single_day_contexts:
            return self.single_day_contexts[0].configuration
        if self.multi_day_contexts is not None and self.multi_day_contexts:
            return self.multi_day_contexts[0].configuration
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
        if self.single_day_contexts is not None and self.single_day_contexts:
            return self.single_day_contexts[0].configuration.roi_detection.crop_to_soma
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
    def save_path(self) -> Path | None:
        """Returns the root output directory path."""
        if self.single_day_contexts is not None and self.single_day_contexts:
            return self.single_day_contexts[0].configuration.file_io.save_path
        if self.multi_day_contexts is not None and self.multi_day_contexts:
            runtime = self.multi_day_contexts[self.current_session_index].runtime
            return runtime.output_path
        return None

    @property
    def data_path(self) -> Path | None:
        """Returns the root input data directory path."""
        if self.single_day_contexts is not None and self.single_day_contexts:
            return self.single_day_contexts[0].configuration.file_io.data_path
        if self.multi_day_contexts is not None and self.multi_day_contexts:
            runtime = self.multi_day_contexts[self.current_session_index].runtime
            return runtime.io.data_path
        return None

    # Registration data from the first plane.

    @property
    def valid_y_range(self) -> list[int]:
        """Returns the valid Y pixel range from the first plane's registration."""
        if self.single_day_contexts is not None and self.single_day_contexts:
            return self.single_day_contexts[0].runtime.registration.valid_y_range
        return [0, 0]

    @property
    def valid_x_range(self) -> list[int]:
        """Returns the valid X pixel range from the first plane's registration."""
        if self.single_day_contexts is not None and self.single_day_contexts:
            return self.single_day_contexts[0].runtime.registration.valid_x_range
        return [0, 0]

    # Derived convenience properties.

    @property
    def basename(self) -> str:
        """Returns the output directory name for display and file operations."""
        path = self.save_path
        if path is not None:
            return path.name
        return ""

    # Factory methods.

    @classmethod
    def from_single_day(cls, root_path: Path) -> ContextData:
        """Loads single-day pipeline data into a ContextData wrapper.

        Loads all planes via ``RuntimeContext.load()`` and the combined data from the root
        suite2p directory. Mutable arrays are copied from the extraction results.

        Args:
            root_path: Root suite2p output directory containing configuration.yaml.

        Returns:
            A fully populated ContextData instance in single-day mode.
        """
        from ..dataclasses import CombinedData, RuntimeContext  # noqa: PLC0415

        # Loads all planes.
        contexts = RuntimeContext.load(root_path=root_path, plane_index=-1)
        if not isinstance(contexts, list):
            contexts = [contexts]

        # Loads combined data and its extraction results.
        combined = CombinedData.load(root_path=root_path)
        combined.load_results(root_path=root_path)

        return cls._populate_from_combined(
            combined=combined,
            is_multi_day=False,
            single_day_contexts=contexts,
        )

    @classmethod
    def from_multi_day(cls, root_path: Path) -> ContextData:
        """Loads multi-day pipeline data into a ContextData wrapper.

        Loads all sessions via ``MultiDayRuntimeContext.load()`` and initializes the view
        with the first session's data.

        Args:
            root_path: Root multi-day output directory (first session's multiday folder).

        Returns:
            A fully populated ContextData instance in multi-day mode.
        """
        from ..dataclasses.runtime_contexts import MultiDayRuntimeContext  # noqa: PLC0415

        # Loads all sessions.
        contexts = MultiDayRuntimeContext.load(root_path=root_path, session_index=-1)
        if not isinstance(contexts, list):
            contexts = [contexts]

        # Initializes from the first session.
        runtime = contexts[0].runtime

        # Loads extraction results for the first session's combined data.
        if runtime.combined_data is not None and runtime.io.data_path is not None:
            runtime.combined_data.load_results(root_path=runtime.io.data_path)

        # Loads multi-day extraction results.
        if runtime.output_path is not None:
            runtime.extraction.load_results(output_path=runtime.output_path)

        instance = cls._populate_from_combined(
            combined=runtime.combined_data,
            is_multi_day=True,
            multi_day_contexts=contexts,
        )
        instance.multi_day_runtime = runtime
        instance.current_session_index = 0
        return instance

    def switch_session(self, session_index: int) -> None:
        """Switches the active multi-day session and reloads all mutable arrays.

        Args:
            session_index: Zero-based index of the session to switch to.

        Raises:
            ValueError: If not in multi-day mode or the index is out of range.
        """
        if self.multi_day_contexts is None:
            message = "Unable to switch session. The current context is not multi-day."
            console.error(message=message, error=ValueError)

        if session_index < 0 or session_index >= len(self.multi_day_contexts):
            message = (
                f"Unable to switch to session {session_index}. Valid range is "
                f"0 to {len(self.multi_day_contexts) - 1}."
            )
            console.error(message=message, error=ValueError)

        self.current_session_index = session_index
        runtime = self.multi_day_contexts[session_index].runtime
        self.multi_day_runtime = runtime

        # Loads extraction results for this session.
        if runtime.combined_data is not None and runtime.io.data_path is not None:
            runtime.combined_data.load_results(root_path=runtime.io.data_path)
        if runtime.output_path is not None:
            runtime.extraction.load_results(output_path=runtime.output_path)

        # Repopulates mutable arrays from this session's combined data.
        self._reload_mutable_arrays(combined=runtime.combined_data)

    @classmethod
    def _populate_from_combined(
        cls,
        combined: CombinedData | None,
        is_multi_day: bool,
        single_day_contexts: list[RuntimeContext] | None = None,
        multi_day_contexts: list[MultiDayRuntimeContext] | None = None,
    ) -> ContextData:
        """Creates a ContextData instance and populates mutable arrays from combined data.

        Args:
            combined: The combined single-day or multi-day session data.
            is_multi_day: Determines whether the data is from a multi-day pipeline.
            single_day_contexts: Single-day runtime contexts (all planes).
            multi_day_contexts: Multi-day runtime contexts (all sessions).

        Returns:
            A populated ContextData instance.
        """
        instance = cls(
            is_multi_day=is_multi_day,
            single_day_contexts=single_day_contexts,
            multi_day_contexts=multi_day_contexts,
            combined=combined,
        )
        instance._reload_mutable_arrays(combined=combined)
        return instance

    def _reload_mutable_arrays(self, combined: CombinedData | None) -> None:
        """Reloads mutable arrays from the given combined data.

        Args:
            combined: The source combined data to copy arrays from.
        """
        self.combined = combined

        if combined is None:
            return

        extraction = combined.extraction

        # Copies ROI statistics list (shallow copy; individual ROIStatistics are mutable).
        if extraction.roi_statistics is not None:
            self.roi_statistics = list(extraction.roi_statistics)
        else:
            self.roi_statistics = []

        roi_count = len(self.roi_statistics)

        # Copies fluorescence traces.
        if extraction.cell_fluorescence is not None:
            self.cell_fluorescence = extraction.cell_fluorescence.copy()
        else:
            self.cell_fluorescence = np.zeros((roi_count, 0), dtype=np.float32)

        if extraction.neuropil_fluorescence is not None:
            self.neuropil_fluorescence = extraction.neuropil_fluorescence.copy()
        else:
            self.neuropil_fluorescence = np.zeros((roi_count, 0), dtype=np.float32)

        if extraction.spikes is not None:
            self.spikes = extraction.spikes.copy()
        else:
            self.spikes = np.zeros((roi_count, 0), dtype=np.float32)

        # Copies classification arrays.
        if extraction.cell_classification is not None:
            self.cell_classification_probabilities = extraction.cell_classification[:, 0].copy()
            self.cell_classification_labels = extraction.cell_classification[:, 1].astype(np.bool_).copy()
        else:
            self.cell_classification_probabilities = np.ones(roi_count, dtype=np.float32)
            self.cell_classification_labels = np.ones(roi_count, dtype=np.bool_)

        # Copies colocalization arrays.
        if extraction.cell_colocalization is not None:
            self.cell_colocalization_probabilities = extraction.cell_colocalization[:, 0].copy()
            self.cell_colocalization_labels = extraction.cell_colocalization[:, 1].astype(np.bool_).copy()
            self.has_red_channel = True
        else:
            self.cell_colocalization_probabilities = np.zeros(roi_count, dtype=np.float32)
            self.cell_colocalization_labels = np.zeros(roi_count, dtype=np.bool_)
            self.has_red_channel = combined.detection.mean_image_channel_2 is not None

        # Initializes the not-merged mask.
        self.not_merged = np.ones(roi_count, dtype=np.bool_)

        # Copies channel 2 traces if available.
        self.cell_fluorescence_channel_2 = (
            extraction.cell_fluorescence_channel_2.copy()
            if extraction.cell_fluorescence_channel_2 is not None
            else None
        )
        self.neuropil_fluorescence_channel_2 = (
            extraction.neuropil_fluorescence_channel_2.copy()
            if extraction.neuropil_fluorescence_channel_2 is not None
            else None
        )

        # Resets behavior data (must be reloaded after session switch).
        self.behavior = None
        self.behavior_time = None
        self.behavior_resampled = None
