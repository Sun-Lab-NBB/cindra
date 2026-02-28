"""Provides data hierarchies for single-day pipeline viewers."""

from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path
from dataclasses import field, dataclass

import numpy as np
from ataraxis_base_utilities import console

from ..io import BinaryFile
from ..dataclasses import CombinedData, RuntimeContext

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..dataclasses import (
        IOData,
        DetectionData,
        ROIStatistics,
        ExtractionData,
        RegistrationData,
    )

_EMPTY: NDArray[np.float32] = np.empty(0, dtype=np.float32)
"""Empty array sentinel returned for absent trace or classification data."""


@dataclass
class SingleDayViewerData:
    """Wraps imported single-day pipeline data for all single-day viewer windows.

    Provides a unified view over combined multi-plane data and individual per-plane data with a switchable view
    index. All trace data is resolved at initialization and routed to callers based on the active view index.
    """

    _contexts: list[RuntimeContext]
    """The RuntimeContext instances for each processed imaging plane."""

    _combined: CombinedData
    """The combined multi-plane data."""

    _view_index: int = -1
    """The active view index. -1 selects the combined view, 0+ selects a per-plane view."""

    _channel_1_binaries: dict[int, BinaryFile] = field(init=False)
    """Per-plane channel 1 binary files keyed by plane index."""

    _channel_2_binaries: dict[int, BinaryFile] = field(init=False)
    """Per-plane channel 2 binary files keyed by plane index. Empty if single-channel."""

    _view_labels: tuple[str, ...] = field(init=False)
    """Cached display labels for all available views including the combined view."""

    def __post_init__(self) -> None:
        """Opens binary file handles and memory-maps all trace arrays for every view."""
        # Opens per-plane binary files for both channels.
        self._channel_1_binaries = {}
        self._channel_2_binaries = {}
        channel_2_paths = self._combined.registered_binary_paths_channel_2
        for index, (height, width, path) in enumerate(
            zip(
                self._combined.plane_heights,
                self._combined.plane_widths,
                self._combined.registered_binary_paths,
                strict=True,
            ),
        ):
            self._channel_1_binaries[index] = BinaryFile(height=int(height), width=int(width), file_path=path)
            if channel_2_paths is not None:
                self._channel_2_binaries[index] = BinaryFile(
                    height=int(height),
                    width=int(width),
                    file_path=channel_2_paths[index],
                )

        # Memory-maps combined extraction data.
        combined_output = self._contexts[0].configuration.file_io.output_path
        if combined_output is None:
            message = (
                "Unable to load combined trace arrays. The output path is not set in the recording's single-day "
                "configuration data."
            )
            console.error(message=message, error=FileNotFoundError)
        self._combined.extraction.memory_map_arrays(combined_output)
        self._combined.extraction.memory_map_results(combined_output)

        # Memory-maps per-plane extraction result arrays (traces, classification, colocalization).
        # memory_map_arrays() was already called by the factory method for ROI statistics and classification.
        for context in self._contexts:
            runtime = context.runtime
            plane_output = runtime.io.output_path
            if plane_output is None:
                message = (
                    f"Unable to load arrays for plane {runtime.io.plane_index}. The output path is not set in the "
                    f"plane's IO data."
                )
                console.error(message=message, error=FileNotFoundError)
            runtime.extraction.memory_map_results(plane_output)

        # Caches display labels for all views.
        self._view_labels = ("Combined", *(f"Plane {context.runtime.io.plane_index}" for context in self._contexts))

    @property
    def view_index(self) -> int:
        """Returns the active view index (-1 for combined, 0+ for per-plane)."""
        return self._view_index

    @property
    def plane_count(self) -> int:
        """Returns the total number of imaging planes in the visualized recording."""
        return len(self._contexts)

    @property
    def view_labels(self) -> tuple[str, ...]:
        """Returns the display labels for all available views including the combined view."""
        return self._view_labels

    @property
    def frame_height(self) -> int:
        """Returns the field-of-view height in pixels for the current view."""
        if self._view_index == -1:
            return self._combined.combined_height
        return self._current_io.frame_height

    @property
    def frame_width(self) -> int:
        """Returns the field-of-view width in pixels for the current view."""
        if self._view_index == -1:
            return self._combined.combined_width
        return self._current_io.frame_width

    @property
    def frame_count(self) -> int:
        """Returns the total number of frames for the visualized recording."""
        fluorescence = self._current_extraction.cell_fluorescence
        return int(fluorescence.shape[1]) if fluorescence is not None else 0

    @property
    def sampling_rate(self) -> float:
        """Returns the per-plane sampling rate in Hertz."""
        return self._combined.sampling_rate

    @property
    def tau(self) -> float:
        """Returns the calcium indicator timescale in seconds."""
        return self._combined.tau

    @property
    def roi_statistics(self) -> list[ROIStatistics]:
        """Returns the spatial and shape statistics for each detected ROI in the current view."""
        statistics = self._current_extraction.roi_statistics
        return list(statistics) if statistics is not None else []

    @property
    def roi_count(self) -> int:
        """Returns the total number of ROIs in the current view."""
        return len(self.roi_statistics)

    @property
    def cell_count(self) -> int:
        """Returns the number of ROIs classified as cells in the current view."""
        classification = self.cell_classification
        return int(classification[:, 1].sum()) if classification.size else 0

    @property
    def cell_fluorescence(self) -> NDArray[np.float32]:
        """Returns the cell fluorescence traces with shape (cells, frames) for the current view."""
        value = self._current_extraction.cell_fluorescence
        return value if value is not None else _EMPTY

    @property
    def cell_fluorescence_channel_2(self) -> NDArray[np.float32]:
        """Returns the channel 2 cell fluorescence traces. Empty with size 0 if single-channel."""
        value = self._current_extraction.cell_fluorescence_channel_2
        return value if value is not None else _EMPTY

    @property
    def neuropil_fluorescence(self) -> NDArray[np.float32]:
        """Returns the neuropil fluorescence traces with shape (cells, frames) for the current view."""
        value = self._current_extraction.neuropil_fluorescence
        return value if value is not None else _EMPTY

    @property
    def neuropil_fluorescence_channel_2(self) -> NDArray[np.float32]:
        """Returns the channel 2 neuropil fluorescence traces. Empty with size 0 if single-channel."""
        value = self._current_extraction.neuropil_fluorescence_channel_2
        return value if value is not None else _EMPTY

    @property
    def subtracted_fluorescence(self) -> NDArray[np.float32]:
        """Returns the memory-mapped baseline-and-neuropil-subtracted fluorescence traces."""
        value = self._current_extraction.subtracted_fluorescence
        return value if value is not None else _EMPTY

    @property
    def subtracted_fluorescence_channel_2(self) -> NDArray[np.float32]:
        """Returns the memory-mapped channel 2 subtracted fluorescence traces. Empty with size 0 if unavailable."""
        value = self._current_extraction.subtracted_fluorescence_channel_2
        return value if value is not None else _EMPTY

    @property
    def spikes(self) -> NDArray[np.float32]:
        """Returns the deconvolved spike traces with shape (cells, frames) for the current view."""
        value = self._current_extraction.spikes
        return value if value is not None else _EMPTY

    @property
    def spikes_channel_2(self) -> NDArray[np.float32]:
        """Returns the channel 2 deconvolved spike traces. Empty with size 0 if single-channel."""
        value = self._current_extraction.spikes_channel_2
        return value if value is not None else _EMPTY

    @property
    def two_channels(self) -> bool:
        """Returns True when the visualized recording has two functional channels."""
        return bool(self._channel_2_binaries)

    @property
    def cell_classification(self) -> NDArray[np.float32]:
        """Returns the cell classification array with shape (roi_count, 2) for the current view.

        Column 0 holds classifier probabilities, column 1 holds labels (0.0/1.0). The array is memory-mapped
        read-write so label modifications propagate directly to disk.
        """
        value = self._current_extraction.cell_classification
        return value if value is not None else _EMPTY

    @property
    def cell_classification_channel_2(self) -> NDArray[np.float32]:
        """Returns the channel 2 cell classification array with shape (roi_count, 2). Empty if single-channel."""
        value = self._current_extraction.cell_classification_channel_2
        return value if value is not None else _EMPTY

    @property
    def cell_colocalization(self) -> NDArray[np.float32]:
        """Returns the cell colocalization array with shape (roi_count, 2) for the current view.

        Column 0 holds classifier probabilities, column 1 holds labels (0.0/1.0). Read-only memory-mapped.
        """
        value = self._current_extraction.cell_colocalization
        return value if value is not None else _EMPTY

    @property
    def mean_image(self) -> NDArray[np.float32] | None:
        """Returns the mean image from the current view's detection data."""
        return self._current_detection.mean_image

    @property
    def enhanced_mean_image(self) -> NDArray[np.float32] | None:
        """Returns the enhanced mean image from the current view's detection data."""
        return self._current_detection.enhanced_mean_image

    @property
    def maximum_projection(self) -> NDArray[np.float32] | None:
        """Returns the maximum projection from the current view's detection data."""
        return self._current_detection.maximum_projection

    @property
    def correlation_map(self) -> NDArray[np.float32] | None:
        """Returns the correlation map from the current view's detection data."""
        return self._current_detection.correlation_map

    @property
    def mean_image_channel_2(self) -> NDArray[np.float32] | None:
        """Returns the channel 2 mean image from the current view's detection data."""
        return self._current_detection.mean_image_channel_2

    @property
    def enhanced_mean_image_channel_2(self) -> NDArray[np.float32] | None:
        """Returns the channel 2 enhanced mean image from the current view's detection data."""
        return self._current_detection.enhanced_mean_image_channel_2

    @property
    def corrected_structural_mean_image(self) -> NDArray[np.float32] | None:
        """Returns the corrected structural channel mean image for the current view."""
        return self._current_extraction.corrected_structural_mean_image

    @property
    def cell_diameter(self) -> int:
        """Returns the estimated cell diameter in pixels."""
        return self._current_detection.cell_diameter

    @property
    def aspect_ratio(self) -> float:
        """Returns the aspect ratio of detected cells."""
        return self._current_detection.aspect_ratio

    @property
    def channel_1_binary(self) -> BinaryFile:
        """Returns the channel 1 binary file for the current plane."""
        return self._channel_1_binaries[self._current_plane_index]

    @property
    def channel_2_binary(self) -> BinaryFile:
        """Returns the channel 2 binary file for the current plane.

        Raises:
            KeyError: If the recording is single-channel.
        """
        return self._channel_2_binaries[self._current_plane_index]

    @property
    def rigid_y_offsets(self) -> NDArray[np.int32]:
        """Returns the vertical (Y) translation offsets from rigid registration, one value per frame or a zero array
        when the underlying data is None.
        """
        value = self._current_registration.rigid_y_offsets
        return value if value is not None else np.zeros(self._current_io.frame_count, dtype=np.int32)

    @property
    def rigid_x_offsets(self) -> NDArray[np.int32]:
        """Returns the horizontal (X) translation offsets from rigid registration, one value per frame or a zero array
        when the underlying data is None.
        """
        value = self._current_registration.rigid_x_offsets
        return value if value is not None else np.zeros(self._current_io.frame_count, dtype=np.int32)

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
        return ""

    @property
    def output_path(self) -> Path | None:
        """Returns the output directory path from the first plane's configuration."""
        return self._contexts[0].configuration.file_io.output_path

    @property
    def valid_y_range(self) -> tuple[int, int]:
        """Returns the valid Y pixel range from the first plane's registration."""
        return self._contexts[0].runtime.registration.valid_y_range

    @property
    def valid_x_range(self) -> tuple[int, int]:
        """Returns the valid X pixel range from the first plane's registration."""
        return self._contexts[0].runtime.registration.valid_x_range

    def switch_view(self, view_index: int) -> None:
        """Switches the active data view to a different plane or the combined view.

        Args:
            view_index: The view to switch to. -1 selects the combined view, 0+ selects a per-plane view.

        Raises:
            ValueError: If the index is out of the valid range.
        """
        if view_index < -1 or view_index >= len(self._contexts):
            message = (
                f"Unable to switch the viewer to index {view_index}. Valid range is -1 to {len(self._contexts) - 1}."
            )
            console.error(message=message, error=ValueError)

        self._view_index = view_index

    @classmethod
    def from_single_day(cls, root_path: Path, view_index: int = -1) -> SingleDayViewerData:
        """Loads single-day pipeline data with combined and per-plane results.

        Loads all planes via ``RuntimeContext.load()`` and the combined data from the root cindra output directory. All
        trace arrays are memory-mapped from disk at initialization.

        Args:
            root_path: Root cindra output directory containing configuration.yaml.
            view_index: The initial view index. -1 selects the combined view, 0+ selects a per-plane view.

        Returns:
            A fully populated SingleDayViewerData instance.
        """
        contexts = RuntimeContext.load(root_path=root_path, plane_index=-1)
        if not isinstance(contexts, list):
            contexts = [contexts]

        # Explicitly memory-maps per-plane arrays since context resolution no longer loads them eagerly.
        for context in contexts:
            plane_output = context.runtime.io.output_path
            if plane_output is not None:
                context.runtime.registration.memory_map_arrays(plane_output)
                context.runtime.detection.memory_map_arrays(plane_output)
                context.runtime.extraction.memory_map_arrays(plane_output)

        combined = CombinedData.load(root_path=root_path)
        combined.detection.load_arrays(root_path)
        return cls(_contexts=contexts, _combined=combined, _view_index=view_index)

    @property
    def _current_plane_index(self) -> int:
        """Returns the plane index for per-plane data access, clamping -1 to 0."""
        return max(0, self._view_index)

    @property
    def _current_extraction(self) -> ExtractionData:
        """Returns the ExtractionData instance for the currently active view."""
        if self._view_index == -1:
            return self._combined.extraction
        return self._contexts[self._view_index].runtime.extraction

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
        """Returns the DetectionData instance for the current view.

        Uses combined detection data when the combined view is active, otherwise falls back to the current plane's
        detection data.
        """
        if self._view_index == -1:
            return self._combined.detection
        return self._contexts[self._current_plane_index].runtime.detection
