"""Provides the data hierarchy for the per-recording across-day ROI tracking quality visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import field, dataclass

from ataraxis_base_utilities import console

from .constants import MaskLayer, BackgroundView, CoordinateSpace
from ..dataclasses import MultiDayRuntimeContext

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Sequence

    import numpy as np
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
