"""Provides the data hierarchy for the per-recording across-day ROI tracking quality visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import field, dataclass

import numpy as np
from ataraxis_base_utilities import LogLevel, console

from .constants import MaskLayer, BackgroundView, CoordinateSpace
from ..dataclasses import MultiDayRuntimeContext

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from ..dataclasses import (
        ROIMask,
        CombinedData,
        DetectionData,
        ROIStatistics,
        MultiDayRuntimeData,
        MultiDayTrackingData,
        MultiDayRegistrationData,
    )
    from .single_day_context import SingleDayViewerData

_EMPTY: NDArray[np.float32] = np.empty(0, dtype=np.float32)
"""Empty array sentinel returned for absent trace or classification data."""


@dataclass
class TrackingViewerData:
    """Wraps multi-day runtime data and serves it to the tracker viewer GUI.

    Provides property-based delegation to the currently viewed recording's multi-day runtime data, supporting recording
    navigation, coordinate space switching, mask layer selection, and channel toggling. All recordings are loaded at
    construction time via the ``from_recording`` factory method.

    Notes:
        The ``combined_data`` field on each recording's runtime contains the combined multi-plane single-day detection
        data (background images, ROI statistics). Registration and tracking data provide transformed images and mask
        sets for cross-recording visualization.
    """

    _contexts: list[MultiDayRuntimeContext] = field(repr=False)
    """The multi-day runtime contexts for all recordings that make up the visualized dataset."""

    _current_recording_index: int = 0
    """The index of the currently displayed recording."""

    @property
    def current_recording_index(self) -> int:
        """Returns the index of the currently displayed recording."""
        return self._current_recording_index

    @property
    def recording_count(self) -> int:
        """Returns the number of recordings in the dataset."""
        return len(self._contexts)

    @property
    def recording_ids(self) -> tuple[str, ...]:
        """Returns the recording identifier strings for all recordings."""
        return tuple(context.runtime.io.session_id for context in self._contexts)

    @property
    def current_recording_id(self) -> str:
        """Returns the recording identifier for the currently displayed recording."""
        return self._current_runtime.io.session_id

    @property
    def frame_height(self) -> int:
        """Returns the combined field-of-view height in pixels."""
        if self._current_combined is None:
            return 0
        return self._current_combined.combined_height

    @property
    def frame_width(self) -> int:
        """Returns the combined field-of-view width in pixels."""
        if self._current_combined is None:
            return 0
        return self._current_combined.combined_width

    @property
    def has_channel_2(self) -> bool:
        """Returns True if channel 2 data is available."""
        if self._current_detection is None:
            return False
        return self._current_detection.mean_image_channel_2 is not None

    @property
    def is_registered(self) -> bool:
        """Returns True if the current recording has been registered to the shared visual space."""
        return self._current_registration.is_registered()

    @property
    def mean_image(self) -> NDArray[np.float32] | None:
        """Returns the native-space mean fluorescence image."""
        if self._current_detection is None:
            return None
        return self._current_detection.mean_image

    @property
    def enhanced_mean_image(self) -> NDArray[np.float32] | None:
        """Returns the native-space contrast-enhanced mean image."""
        if self._current_detection is None:
            return None
        return self._current_detection.enhanced_mean_image

    @property
    def maximum_projection(self) -> NDArray[np.float32] | None:
        """Returns the native-space maximum intensity projection."""
        if self._current_detection is None:
            return None
        return self._current_detection.maximum_projection

    @property
    def correlation_map(self) -> NDArray[np.float32] | None:
        """Returns the native-space pixel correlation map."""
        if self._current_detection is None:
            return None
        return self._current_detection.correlation_map

    @property
    def mean_image_channel_2(self) -> NDArray[np.float32] | None:
        """Returns the channel 2 native-space mean fluorescence image."""
        if self._current_detection is None:
            return None
        return self._current_detection.mean_image_channel_2

    @property
    def enhanced_mean_image_channel_2(self) -> NDArray[np.float32] | None:
        """Returns the channel 2 native-space contrast-enhanced mean image."""
        if self._current_detection is None:
            return None
        return self._current_detection.enhanced_mean_image_channel_2

    @property
    def maximum_projection_channel_2(self) -> NDArray[np.float32] | None:
        """Returns the channel 2 native-space maximum intensity projection."""
        if self._current_detection is None:
            return None
        return self._current_detection.maximum_projection_channel_2

    @property
    def correlation_map_channel_2(self) -> NDArray[np.float32] | None:
        """Returns the channel 2 native-space pixel correlation map."""
        if self._current_detection is None:
            return None
        return self._current_detection.correlation_map_channel_2

    @property
    def transformed_mean_image(self) -> NDArray[np.float32] | None:
        """Returns the mean image in deformed (registered) space."""
        return self._current_registration.transformed_mean_image

    @property
    def transformed_enhanced_mean_image(self) -> NDArray[np.float32] | None:
        """Returns the enhanced mean image in deformed (registered) space."""
        return self._current_registration.transformed_enhanced_mean_image

    @property
    def transformed_maximum_projection(self) -> NDArray[np.float32] | None:
        """Returns the maximum projection in deformed (registered) space."""
        return self._current_registration.transformed_maximum_projection

    @property
    def transformed_mean_image_channel_2(self) -> NDArray[np.float32] | None:
        """Returns the channel 2 mean image in deformed (registered) space."""
        return self._current_registration.transformed_mean_image_channel_2

    @property
    def transformed_enhanced_mean_image_channel_2(self) -> NDArray[np.float32] | None:
        """Returns the channel 2 enhanced mean image in deformed (registered) space."""
        return self._current_registration.transformed_enhanced_mean_image_channel_2

    @property
    def transformed_maximum_projection_channel_2(self) -> NDArray[np.float32] | None:
        """Returns the channel 2 maximum projection in deformed (registered) space."""
        return self._current_registration.transformed_maximum_projection_channel_2

    @property
    def original_masks(self) -> list[ROIStatistics] | None:
        """Returns the selected ROI masks that were used as input to forward deformation.

        Filters the full single-day extraction ROI list to only the cells selected for multi-day tracking, giving a
        1:1 correspondence with the deformed mask layer.
        """
        if self._current_combined is None:
            return None
        all_masks = self._current_combined.extraction.roi_statistics
        if all_masks is None:
            return None
        selected_indices = self._current_runtime.io.selected_cell_indices
        if not selected_indices:
            return all_masks
        return [all_masks[i] for i in selected_indices]

    @property
    def original_masks_channel_2(self) -> list[ROIStatistics] | None:
        """Returns the channel 2 selected ROI masks that were used as input to forward deformation.

        Filters the full single-day extraction ROI list to only the cells selected for multi-day tracking, giving a
        1:1 correspondence with the deformed mask layer.
        """
        if self._current_combined is None:
            return None
        all_masks = self._current_combined.extraction.roi_statistics_channel_2
        if all_masks is None:
            return None
        selected_indices = self._current_runtime.io.selected_cell_indices_channel_2
        if not selected_indices:
            return all_masks
        return [all_masks[i] for i in selected_indices]

    @property
    def deformed_masks(self) -> list[ROIMask] | None:
        """Returns the ROI masks warped to the shared coordinate space."""
        return self._current_registration.deformed_cell_masks

    @property
    def deformed_masks_channel_2(self) -> list[ROIMask] | None:
        """Returns the channel 2 ROI masks warped to the shared coordinate space."""
        return self._current_registration.deformed_cell_masks_channel_2

    @property
    def template_masks(self) -> list[ROIMask] | None:
        """Returns the consensus template masks from cross-recording tracking."""
        return self._current_tracking.template_masks

    @property
    def template_masks_channel_2(self) -> list[ROIMask] | None:
        """Returns the channel 2 consensus template masks from cross-recording tracking."""
        return self._current_tracking.template_masks_channel_2

    @property
    def tracked_masks(self) -> list[ROIStatistics] | None:
        """Returns the template ROI masks backward-deformed to the current recording's native coordinate space."""
        return self._current_runtime.extraction.roi_statistics

    @property
    def tracked_masks_channel_2(self) -> list[ROIStatistics] | None:
        """Returns the channel 2 template ROI masks backward-deformed to native coordinate space."""
        return self._current_runtime.extraction.roi_statistics_channel_2

    @classmethod
    def from_recording(cls, root_path: Path) -> TrackingViewerData:
        """Loads all multi-day recordings from a dataset directory.

        Args:
            root_path: The path to any recording's root processed data directory. The loader searches recursively for
                ``multiday_runtime_data.yaml`` and reconstructs the full dataset hierarchy.

        Returns:
            A fully populated TrackingViewerData instance.
        """
        contexts = MultiDayRuntimeContext.load(root_path=root_path, session_index=-1)
        if not isinstance(contexts, list):
            contexts = [contexts]

        # Explicitly loads arrays since context resolution no longer loads them eagerly.
        for ctx in contexts:
            output_path = ctx.runtime.output_path
            if output_path is not None:
                ctx.runtime.registration.memory_map_arrays(output_path)
                ctx.runtime.tracking.load_arrays(output_path)
                ctx.runtime.extraction.load_arrays(output_path)
                ctx.runtime.extraction.memory_map_results(output_path)
            combined = ctx.runtime.combined_data
            if combined is not None and ctx.runtime.io.data_path is not None:
                combined.detection.memory_map_arrays(ctx.runtime.io.data_path)
                combined.extraction.memory_map_arrays(ctx.runtime.io.data_path)

        return cls(_contexts=contexts)

    def masks_for_layer(
        self,
        layer: MaskLayer,
        *,
        channel_2: bool = False,
    ) -> Sequence[ROIMask | ROIStatistics] | None:
        """Returns the mask set for the specified layer and channel.

        Args:
            layer: The mask layer to retrieve.
            channel_2: Determines whether to return channel 2 masks instead of channel 1.

        Returns:
            The list of ROIStatistics or ROIMask for the requested layer, or None if unavailable.
        """
        if layer == MaskLayer.ORIGINAL:
            return self.original_masks_channel_2 if channel_2 else self.original_masks
        if layer == MaskLayer.DEFORMED:
            return self.deformed_masks_channel_2 if channel_2 else self.deformed_masks
        if layer == MaskLayer.TEMPLATE:
            return self.template_masks_channel_2 if channel_2 else self.template_masks
        if layer == MaskLayer.TRACKED:
            return self.tracked_masks_channel_2 if channel_2 else self.tracked_masks
        return None

    def background_image(
        self,
        image_type: BackgroundView,
        *,
        coordinate_space: CoordinateSpace = CoordinateSpace.NATIVE,
        channel_2: bool = False,
    ) -> NDArray[np.float32] | None:
        """Returns the background image for the specified image type, coordinate space, and channel combination.

        When the coordinate space is ``TRANSFORMED`` and transformed images are available, returns the deformed
        (registered) image. Falls back to native-space images if transformed variants are not available.

        Args:
            image_type: The background image type to retrieve.
            coordinate_space: The coordinate space (native or transformed).
            channel_2: Determines whether to return channel 2 images instead of channel 1.

        Returns:
            The requested image array, or None if unavailable.
        """
        if coordinate_space == CoordinateSpace.TRANSFORMED and not channel_2:
            image = self._transformed_background_image(image_type=image_type)
            if image is not None:
                return image
        if coordinate_space == CoordinateSpace.TRANSFORMED and channel_2:
            image = self._transformed_background_image_channel_2(image_type=image_type)
            if image is not None:
                return image

        if channel_2:
            return self._native_background_image_channel_2(image_type=image_type)
        return self._native_background_image(image_type=image_type)

    def switch_recording(self, recording_index: int) -> None:
        """Switches the active recording.

        Args:
            recording_index: The index of the recording to switch to.

        Raises:
            ValueError: If the index is out of range.
        """
        if recording_index < 0 or recording_index >= len(self._contexts):
            message = (
                f"Unable to switch the tracking viewer to recording {recording_index}. Valid range is 0 to "
                f"{len(self._contexts) - 1}."
            )
            console.error(message=message, error=ValueError)
        self._current_recording_index = recording_index

    def _native_background_image(self, image_type: BackgroundView) -> NDArray[np.float32] | None:
        """Returns the native-space channel 1 background image for the given type."""
        if image_type == BackgroundView.MEAN_IMAGE:
            return self.mean_image
        if image_type == BackgroundView.ENHANCED_MEAN_IMAGE:
            return self.enhanced_mean_image
        if image_type == BackgroundView.MAXIMUM_PROJECTION:
            return self.maximum_projection
        if image_type == BackgroundView.CORRELATION_MAP:
            return self.correlation_map
        return None

    def _native_background_image_channel_2(self, image_type: BackgroundView) -> NDArray[np.float32] | None:
        """Returns the native-space channel 2 background image for the given type."""
        if image_type == BackgroundView.MEAN_IMAGE:
            return self.mean_image_channel_2
        if image_type == BackgroundView.ENHANCED_MEAN_IMAGE:
            return self.enhanced_mean_image_channel_2
        if image_type == BackgroundView.MAXIMUM_PROJECTION:
            return self.maximum_projection_channel_2
        if image_type == BackgroundView.CORRELATION_MAP:
            return self.correlation_map_channel_2
        return None

    def _transformed_background_image(self, image_type: BackgroundView) -> NDArray[np.float32] | None:
        """Returns the transformed-space channel 1 background image for the given type."""
        if image_type == BackgroundView.MEAN_IMAGE:
            return self.transformed_mean_image
        if image_type == BackgroundView.ENHANCED_MEAN_IMAGE:
            return self.transformed_enhanced_mean_image
        if image_type == BackgroundView.MAXIMUM_PROJECTION:
            return self.transformed_maximum_projection
        return None

    def _transformed_background_image_channel_2(self, image_type: BackgroundView) -> NDArray[np.float32] | None:
        """Returns the transformed-space channel 2 background image for the given type."""
        if image_type == BackgroundView.MEAN_IMAGE:
            return self.transformed_mean_image_channel_2
        if image_type == BackgroundView.ENHANCED_MEAN_IMAGE:
            return self.transformed_enhanced_mean_image_channel_2
        if image_type == BackgroundView.MAXIMUM_PROJECTION:
            return self.transformed_maximum_projection_channel_2
        return None

    @property
    def cell_fluorescence(self) -> NDArray[np.float32] | None:
        """Returns the cell fluorescence array for the current recording's extraction."""
        return self._current_runtime.extraction.cell_fluorescence

    @property
    def neuropil_fluorescence(self) -> NDArray[np.float32] | None:
        """Returns the neuropil fluorescence array for the current recording's extraction."""
        return self._current_runtime.extraction.neuropil_fluorescence

    @property
    def spikes(self) -> NDArray[np.float32] | None:
        """Returns the deconvolved spikes array for the current recording's extraction."""
        return self._current_runtime.extraction.spikes

    @property
    def has_traces(self) -> bool:
        """Returns True if cell fluorescence trace data is available for the current recording."""
        f = self.cell_fluorescence
        return f is not None and f.size > 0

    @property
    def trace_frame_count(self) -> int:
        """Returns the number of frames in the trace data for the current recording."""
        f = self.cell_fluorescence
        if f is None or f.size == 0:
            return 0
        return f.shape[1]

    @property
    def sampling_rate(self) -> float:
        """Returns the sampling rate from the current recording's combined metadata."""
        if self._current_combined is not None:
            return self._current_combined.sampling_rate
        return 0.0

    def cell_fluorescence_for_recording(self, recording_index: int) -> NDArray[np.float32] | None:
        """Returns the cell fluorescence array for a specific recording.

        Args:
            recording_index: The recording index.

        Returns:
            The cell fluorescence array, or None if unavailable.
        """
        return self._contexts[recording_index].runtime.extraction.cell_fluorescence

    def neuropil_fluorescence_for_recording(self, recording_index: int) -> NDArray[np.float32] | None:
        """Returns the neuropil fluorescence array for a specific recording.

        Args:
            recording_index: The recording index.

        Returns:
            The neuropil fluorescence array, or None if unavailable.
        """
        return self._contexts[recording_index].runtime.extraction.neuropil_fluorescence

    def spikes_for_recording(self, recording_index: int) -> NDArray[np.float32] | None:
        """Returns the deconvolved spikes array for a specific recording.

        Args:
            recording_index: The recording index.

        Returns:
            The spikes array, or None if unavailable.
        """
        return self._contexts[recording_index].runtime.extraction.spikes

    @property
    def _current_runtime(self) -> MultiDayRuntimeData:
        """Returns the runtime data for the current recording."""
        return self._contexts[self._current_recording_index].runtime

    @property
    def _current_registration(self) -> MultiDayRegistrationData:
        """Returns the registration data for the current recording."""
        return self._current_runtime.registration

    @property
    def _current_tracking(self) -> MultiDayTrackingData:
        """Returns the tracking data for the current recording."""
        return self._current_runtime.tracking

    @property
    def _current_combined(self) -> CombinedData | None:
        """Returns the combined single-day data for the current recording."""
        return self._current_runtime.combined_data

    @property
    def _current_detection(self) -> DetectionData | None:
        """Returns the combined detection data for the current recording."""
        if self._current_combined is None:
            return None
        return self._current_combined.detection


@dataclass
class MultiDayViewerData:
    """Wraps multi-day tracked ROI data for display in the ROI viewer.

    Mirrors the ``SingleDayViewerData`` property interface so both classes can be used interchangeably by the ROI viewer
    and overlay functions. Each instance represents one multi-day dataset that includes the loaded single-day session.
    The consumer creates multiple instances — one per discovered dataset.
    """

    _contexts: list[MultiDayRuntimeContext] = field(repr=False)
    """All recording contexts in the multi-day dataset."""

    _single_day: SingleDayViewerData = field(repr=False)
    """The single-day data for background images and frame dimensions."""

    _session_recording_index: int = 0
    """Index into ``_contexts`` for the loaded single-day session."""

    dataset_name: str = ""
    """Display label for the ROI Source dropdown."""

    _current_recording_index: int = field(init=False, default=0)
    """Index of the currently active recording for trace display."""

    def __post_init__(self) -> None:
        """Memory-maps arrays for all recordings in the dataset."""
        self._current_recording_index = self._session_recording_index

        for ctx in self._contexts:
            output_path = ctx.runtime.output_path
            if output_path is not None:
                ctx.runtime.registration.memory_map_arrays(output_path)
                ctx.runtime.tracking.load_arrays(output_path)
                ctx.runtime.extraction.load_arrays(output_path)
                ctx.runtime.extraction.memory_map_results(output_path)
            combined = ctx.runtime.combined_data
            if combined is not None and ctx.runtime.io.data_path is not None:
                combined.detection.memory_map_arrays(ctx.runtime.io.data_path)
                combined.extraction.memory_map_arrays(ctx.runtime.io.data_path)

    # ------------------------------------------------------------------
    # Property interface matching SingleDayViewerData
    # ------------------------------------------------------------------

    @property
    def roi_statistics(self) -> list[ROIStatistics]:
        """Returns the tracked ROI masks from the current recording's extraction data."""
        statistics = self._current_runtime.extraction.roi_statistics
        return list(statistics) if statistics is not None else []

    @property
    def roi_count(self) -> int:
        """Returns the number of tracked ROIs."""
        return len(self.roi_statistics)

    @property
    def cell_count(self) -> int:
        """Returns the number of tracked ROIs (all tracked ROIs are classified cells)."""
        return self.roi_count

    @property
    def frame_height(self) -> int:
        """Returns the combined field-of-view height from the single-day data."""
        return self._single_day.frame_height

    @property
    def frame_width(self) -> int:
        """Returns the combined field-of-view width from the single-day data."""
        return self._single_day.frame_width

    @property
    def frame_count(self) -> int:
        """Returns the number of frames in the current recording's extraction data."""
        f = self.cell_fluorescence
        return int(f.shape[1]) if f.size > 0 else 0

    @property
    def cell_fluorescence(self) -> NDArray[np.float32]:
        """Returns the cell fluorescence traces from the current recording's extraction."""
        value = self._current_runtime.extraction.cell_fluorescence
        return value if value is not None else _EMPTY

    @property
    def neuropil_fluorescence(self) -> NDArray[np.float32]:
        """Returns the neuropil fluorescence traces from the current recording's extraction."""
        value = self._current_runtime.extraction.neuropil_fluorescence
        return value if value is not None else _EMPTY

    @property
    def spikes(self) -> NDArray[np.float32]:
        """Returns the deconvolved spike traces from the current recording's extraction."""
        value = self._current_runtime.extraction.spikes
        return value if value is not None else _EMPTY

    @property
    def cell_classification(self) -> NDArray[np.float32]:
        """Returns a synthetic classification array where all tracked ROIs are cells with probability 1.0."""
        n = self.roi_count
        if n == 0:
            return _EMPTY
        return np.column_stack([np.ones(n, dtype=np.float32), np.ones(n, dtype=np.float32)])

    @property
    def cell_colocalization(self) -> NDArray[np.float32]:
        """Returns a synthetic colocalization array (not applicable in multi-day mode)."""
        n = self.roi_count
        if n == 0:
            return _EMPTY
        return np.zeros((n, 2), dtype=np.float32)

    @property
    def two_channels(self) -> bool:
        """Returns True if channel 2 data is available in the current recording's combined detection data."""
        detection = self._current_detection
        if detection is None:
            return False
        return detection.mean_image_channel_2 is not None

    @property
    def mean_image(self) -> NDArray[np.float32] | None:
        """Returns the mean image from the single-day data."""
        return self._single_day.mean_image

    @property
    def enhanced_mean_image(self) -> NDArray[np.float32] | None:
        """Returns the enhanced mean image from the single-day data."""
        return self._single_day.enhanced_mean_image

    @property
    def maximum_projection(self) -> NDArray[np.float32] | None:
        """Returns the maximum projection from the single-day data."""
        return self._single_day.maximum_projection

    @property
    def correlation_map(self) -> NDArray[np.float32] | None:
        """Returns the correlation map from the single-day data."""
        return self._single_day.correlation_map

    @property
    def corrected_structural_mean_image(self) -> None:
        """Returns None (corrected structural image is not available in multi-day mode)."""
        return None

    @property
    def mean_image_channel_2(self) -> NDArray[np.float32] | None:
        """Returns the channel 2 mean image from the single-day data."""
        return self._single_day.mean_image_channel_2

    @property
    def enhanced_mean_image_channel_2(self) -> NDArray[np.float32] | None:
        """Returns the channel 2 enhanced mean image from the single-day data."""
        return self._single_day.enhanced_mean_image_channel_2

    @property
    def sampling_rate(self) -> float:
        """Returns the sampling rate from the single-day data."""
        return self._single_day.sampling_rate

    @property
    def tau(self) -> float:
        """Returns the calcium indicator timescale from the single-day data."""
        return self._single_day.tau

    @property
    def valid_y_range(self) -> tuple[int, int]:
        """Returns the full frame Y range (no subregion in multi-day mode)."""
        return (0, self.frame_height)

    @property
    def valid_x_range(self) -> tuple[int, int]:
        """Returns the full frame X range (no subregion in multi-day mode)."""
        return (0, self.frame_width)

    @property
    def output_path(self) -> Path | None:
        """Returns the output directory path from the single-day data."""
        return self._single_day.output_path

    @property
    def recording_label(self) -> str:
        """Returns the session display label from the single-day data."""
        return self._single_day.recording_label

    @property
    def view_labels(self) -> tuple[str, ...]:
        """Returns the available view labels (multi-day only supports the combined view)."""
        return ("Combined",)

    @property
    def view_index(self) -> int:
        """Returns the active view index (always combined in multi-day mode)."""
        return -1

    @property
    def plane_count(self) -> int:
        """Returns the number of planes (always 1 in multi-day combined mode)."""
        return 1

    @property
    def aspect_ratio(self) -> float:
        """Returns the aspect ratio from the single-day data."""
        return self._single_day.aspect_ratio

    @property
    def cell_diameter(self) -> int:
        """Returns the estimated cell diameter from the single-day data."""
        return self._single_day.cell_diameter

    # ------------------------------------------------------------------
    # Multi-day-specific properties and methods
    # ------------------------------------------------------------------

    @property
    def recording_count(self) -> int:
        """Returns the number of recordings in the dataset."""
        return len(self._contexts)

    @property
    def recording_ids(self) -> tuple[str, ...]:
        """Returns the recording identifier strings for all recordings."""
        return tuple(ctx.runtime.io.session_id for ctx in self._contexts)

    @property
    def current_recording_index(self) -> int:
        """Returns the index of the currently active recording."""
        return self._current_recording_index

    def cell_fluorescence_for_recording(self, recording_index: int) -> NDArray[np.float32] | None:
        """Returns the cell fluorescence array for a specific recording.

        Args:
            recording_index: The recording index.

        Returns:
            The cell fluorescence array, or None if unavailable.
        """
        return self._contexts[recording_index].runtime.extraction.cell_fluorescence

    def neuropil_fluorescence_for_recording(self, recording_index: int) -> NDArray[np.float32] | None:
        """Returns the neuropil fluorescence array for a specific recording.

        Args:
            recording_index: The recording index.

        Returns:
            The neuropil fluorescence array, or None if unavailable.
        """
        return self._contexts[recording_index].runtime.extraction.neuropil_fluorescence

    def spikes_for_recording(self, recording_index: int) -> NDArray[np.float32] | None:
        """Returns the deconvolved spikes array for a specific recording.

        Args:
            recording_index: The recording index.

        Returns:
            The spikes array, or None if unavailable.
        """
        return self._contexts[recording_index].runtime.extraction.spikes

    def switch_recording(self, recording_index: int) -> None:
        """Switches the active recording for trace display.

        Args:
            recording_index: The index of the recording to switch to.
        """
        self._current_recording_index = recording_index

    def switch_view(self, view_index: int) -> None:
        """No-op since multi-day only supports the combined view.

        Args:
            view_index: Ignored.
        """

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @property
    def _current_runtime(self) -> MultiDayRuntimeData:
        """Returns the runtime data for the current recording."""
        return self._contexts[self._current_recording_index].runtime

    @property
    def _current_combined(self) -> CombinedData | None:
        """Returns the combined single-day data for the current recording."""
        return self._current_runtime.combined_data

    @property
    def _current_detection(self) -> DetectionData | None:
        """Returns the combined detection data for the current recording."""
        if self._current_combined is None:
            return None
        return self._current_combined.detection

    # ------------------------------------------------------------------
    # Factory method
    # ------------------------------------------------------------------

    @classmethod
    def discover(cls, root_path: Path, single_day: SingleDayViewerData) -> list[MultiDayViewerData]:
        """Discovers multi-day datasets that include the loaded single-day session.

        Searches ``root_path`` for ``multiday_runtime_data.yaml`` files, loads all recording contexts for each
        discovered dataset, identifies which recording corresponds to the loaded session, and returns one
        ``MultiDayViewerData`` per dataset.

        Args:
            root_path: The cindra output root directory to search recursively.
            single_day: The loaded single-day viewer data used to identify the session's recording index.

        Returns:
            A list of ``MultiDayViewerData`` instances, one per discovered dataset. Empty if no datasets are found.
        """
        matches = list(root_path.rglob("multiday_runtime_data.yaml"))
        if not matches:
            return []

        # Groups YAML files by dataset. Each file lives at {cindra_root}/multiday/{dataset}/multiday_runtime_data.yaml.
        # Multiple sessions in the same dataset share the same dataset_name but have different YAML file paths.
        # We only need to load from one YAML per dataset since MultiDayRuntimeContext.load() resolves all sessions.
        datasets: dict[Path, Path] = {}
        for match in matches:
            dataset_dir = match.parent
            if dataset_dir not in datasets:
                datasets[dataset_dir] = match

        # Resolves the cindra root of the loaded single-day session for matching.
        single_day_cindra_root = single_day.output_path.parent if single_day.output_path is not None else None
        if single_day_cindra_root is None:
            single_day_cindra_root = (
                single_day._contexts[0].configuration.file_io.output_path.parent
                if single_day._contexts[0].configuration.file_io.output_path is not None
                else None
            )

        results: list[MultiDayViewerData] = []
        for dataset_dir, yaml_path in datasets.items():
            try:
                contexts = MultiDayRuntimeContext.load(root_path=yaml_path.parent, session_index=-1)
                if not isinstance(contexts, list):
                    contexts = [contexts]
            except Exception:
                console.echo(
                    message=f"Failed to load multi-day dataset at {dataset_dir}, skipping.",
                    level=LogLevel.WARNING,
                )
                continue

            # Finds the session recording index by comparing data_path against the single-day cindra root.
            session_index: int | None = None
            for index, ctx in enumerate(contexts):
                ctx_data_path = ctx.runtime.io.data_path
                if (
                    ctx_data_path is not None
                    and single_day_cindra_root is not None
                    and ctx_data_path == single_day_cindra_root
                ):
                    session_index = index
                    break

            if session_index is None:
                console.echo(
                    message=f"Loaded session not found in multi-day dataset '{dataset_dir.name}', skipping.",
                    level=LogLevel.WARNING,
                )
                continue

            dataset_name = contexts[0].runtime.io.dataset_name or dataset_dir.name
            results.append(
                cls(
                    _contexts=contexts,
                    _single_day=single_day,
                    _session_recording_index=session_index,
                    dataset_name=dataset_name,
                )
            )

        return results
