"""Provides the unified data hierarchy for all cindra GUI viewer windows."""

from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path
from dataclasses import field, dataclass

import numpy as np
from natsort import natsorted
from ataraxis_base_utilities import LogLevel, console

from ..io import BinaryFile, BinaryFileCombined
from ..dataclasses import CombinedData, RuntimeContext, MultiDayRuntimeContext

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..dataclasses import (
        IOData,
        ROIMask,
        DetectionData,
        ROIStatistics,
        ExtractionData,
        RegistrationData,
        MultiDayRuntimeData,
        MultiDayTrackingData,
        MultiDayRegistrationData,
    )

EMPTY: NDArray[np.float32] = np.empty(0, dtype=np.float32)
"""Empty array sentinel returned for absent trace or classification data."""


@dataclass
class SingleDayData:
    """Wraps the output of the single-day processing pipeline for a single recording and serves it to consumer GUIs.

    Provides a switchable view over combined multi-plane data and individual per-plane data. All trace arrays are
    memory-mapped at initialization and routed to callers based on the active view index.
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

    _combined_binary: BinaryFileCombined = field(init=False)
    """Stitched multi-plane binary for channel 1, used by the binary viewer to display all planes at once."""

    _combined_binary_channel_2: BinaryFileCombined | None = field(init=False)
    """Stitched multi-plane binary for channel 2, or None when the recording is single-channel."""

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

        # Constructs stitched multi-plane binaries for combined frame display.
        self._combined_binary = BinaryFileCombined(
            height=self._combined.combined_height,
            width=self._combined.combined_width,
            plane_heights=self._combined.plane_heights,
            plane_widths=self._combined.plane_widths,
            plane_y_coordinates=self._combined.plane_y_offsets,
            plane_x_coordinates=self._combined.plane_x_offsets,
            file_paths=list(self._combined.registered_binary_paths),
        )
        if channel_2_paths is not None:
            self._combined_binary_channel_2 = BinaryFileCombined(
                height=self._combined.combined_height,
                width=self._combined.combined_width,
                plane_heights=self._combined.plane_heights,
                plane_widths=self._combined.plane_widths,
                plane_y_coordinates=self._combined.plane_y_offsets,
                plane_x_coordinates=self._combined.plane_x_offsets,
                file_paths=list(channel_2_paths),
            )
        else:
            self._combined_binary_channel_2 = None

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

        # Memory-maps per-plane extraction result arrays (traces, classification, colocalization). Skips
        # memory_map_arrays() which was already called by the factory for ROI statistics and classification.
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
        return int(self.cell_fluorescence.shape[1])

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
        if statistics is None:
            console.error(
                message="Unable to retrieve the ROI statistics for the current single-day view. The pipeline "
                "data is incomplete or corrupt.",
                error=RuntimeError,
            )
        return list(statistics)

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
        if value is None:
            console.error(
                message="Unable to retrieve the cell fluorescence traces for the current single-day view. "
                "The pipeline data is incomplete or corrupt.",
                error=RuntimeError,
            )
        return value

    @property
    def cell_fluorescence_channel_2(self) -> NDArray[np.float32]:
        """Returns the channel 2 cell fluorescence traces. Empty with size 0 if single-channel."""
        value = self._current_extraction.cell_fluorescence_channel_2
        return value if value is not None else EMPTY

    @property
    def neuropil_fluorescence(self) -> NDArray[np.float32]:
        """Returns the neuropil fluorescence traces with shape (cells, frames) for the current view."""
        value = self._current_extraction.neuropil_fluorescence
        if value is None:
            console.error(
                message="Unable to retrieve the neuropil fluorescence traces for the current single-day view. "
                "The pipeline data is incomplete or corrupt.",
                error=RuntimeError,
            )
        return value

    @property
    def neuropil_fluorescence_channel_2(self) -> NDArray[np.float32]:
        """Returns the channel 2 neuropil fluorescence traces. Empty with size 0 if single-channel."""
        value = self._current_extraction.neuropil_fluorescence_channel_2
        return value if value is not None else EMPTY

    @property
    def subtracted_fluorescence(self) -> NDArray[np.float32]:
        """Returns the memory-mapped baseline-and-neuropil-subtracted fluorescence traces."""
        value = self._current_extraction.subtracted_fluorescence
        if value is None:
            console.error(
                message="Unable to retrieve the subtracted fluorescence traces for the current single-day view. "
                "The pipeline data is incomplete or corrupt.",
                error=RuntimeError,
            )
        return value

    @property
    def subtracted_fluorescence_channel_2(self) -> NDArray[np.float32]:
        """Returns the memory-mapped channel 2 subtracted fluorescence traces. Empty with size 0 if unavailable."""
        value = self._current_extraction.subtracted_fluorescence_channel_2
        return value if value is not None else EMPTY

    @property
    def spikes(self) -> NDArray[np.float32]:
        """Returns the deconvolved spike traces with shape (cells, frames) for the current view."""
        value = self._current_extraction.spikes
        if value is None:
            console.error(
                message="Unable to retrieve the deconvolved spikes for the current single-day view. "
                "The pipeline data is incomplete or corrupt.",
                error=RuntimeError,
            )
        return value

    @property
    def spikes_channel_2(self) -> NDArray[np.float32]:
        """Returns the channel 2 deconvolved spike traces. Empty with size 0 if single-channel."""
        value = self._current_extraction.spikes_channel_2
        return value if value is not None else EMPTY

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
        if value is None:
            console.error(
                message="Unable to retrieve the cell classification for the current single-day view. "
                "The pipeline data is incomplete or corrupt.",
                error=RuntimeError,
            )
        return value

    @property
    def cell_classification_channel_2(self) -> NDArray[np.float32]:
        """Returns the channel 2 cell classification array with shape (roi_count, 2). Empty if single-channel."""
        value = self._current_extraction.cell_classification_channel_2
        return value if value is not None else EMPTY

    @property
    def cell_colocalization(self) -> NDArray[np.float32]:
        """Returns the cell colocalization array with shape (roi_count, 2) for the current view.

        Column 0 holds classifier probabilities, column 1 holds labels (0.0/1.0). Read-only memory-mapped.
        """
        value = self._current_extraction.cell_colocalization
        if value is None:
            console.error(
                message="Unable to retrieve the cell colocalization for the current single-day view. "
                "The pipeline data is incomplete or corrupt.",
                error=RuntimeError,
            )
        return value

    @property
    def mean_image(self) -> NDArray[np.float32]:
        """Returns the mean image from the current view's detection data."""
        value = self._current_detection.mean_image
        if value is None:
            console.error(
                message="Unable to retrieve the mean image for the current single-day view. The pipeline data "
                "is incomplete or corrupt.",
                error=RuntimeError,
            )
        return value

    @property
    def enhanced_mean_image(self) -> NDArray[np.float32]:
        """Returns the enhanced mean image from the current view's detection data."""
        value = self._current_detection.enhanced_mean_image
        if value is None:
            console.error(
                message="Unable to retrieve the enhanced mean image for the current single-day view. The "
                "pipeline data is incomplete or corrupt.",
                error=RuntimeError,
            )
        return value

    @property
    def maximum_projection(self) -> NDArray[np.float32]:
        """Returns the maximum projection from the current view's detection data."""
        value = self._current_detection.maximum_projection
        if value is None:
            console.error(
                message="Unable to retrieve the maximum projection for the current single-day view. The "
                "pipeline data is incomplete or corrupt.",
                error=RuntimeError,
            )
        return value

    @property
    def correlation_map(self) -> NDArray[np.float32]:
        """Returns the correlation map from the current view's detection data."""
        value = self._current_detection.correlation_map
        if value is None:
            console.error(
                message="Unable to retrieve the correlation map for the current single-day view. The pipeline "
                "data is incomplete or corrupt.",
                error=RuntimeError,
            )
        return value

    @property
    def mean_image_channel_2(self) -> NDArray[np.float32]:
        """Returns the channel 2 mean image from the current view's detection data. Empty if single-channel."""
        value = self._current_detection.mean_image_channel_2
        return value if value is not None else EMPTY

    @property
    def enhanced_mean_image_channel_2(self) -> NDArray[np.float32]:
        """Returns the channel 2 enhanced mean image from the current view's detection data. Empty if single-channel."""
        value = self._current_detection.enhanced_mean_image_channel_2
        return value if value is not None else EMPTY

    @property
    def correlation_map_channel_2(self) -> NDArray[np.float32]:
        """Returns the channel 2 correlation map from the current view's detection data. Empty if single-channel."""
        value = self._current_detection.correlation_map_channel_2
        return value if value is not None else EMPTY

    @property
    def maximum_projection_channel_2(self) -> NDArray[np.float32]:
        """Returns the channel 2 maximum projection from the current view's detection data. Empty if single-channel."""
        value = self._current_detection.maximum_projection_channel_2
        return value if value is not None else EMPTY

    @property
    def corrected_structural_mean_image(self) -> NDArray[np.float32]:
        """Returns the corrected structural channel mean image for the current view. Empty if unavailable."""
        value = self._current_extraction.corrected_structural_mean_image
        return value if value is not None else EMPTY

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
    def output_path(self) -> Path:
        """Returns the output directory path from the first plane's configuration."""
        value = self._contexts[0].configuration.file_io.output_path
        if value is None:
            console.error(
                message="Unable to retrieve the output path for the single-day recording. The pipeline "
                "data is incomplete or corrupt.",
                error=RuntimeError,
            )
        return value

    @property
    def valid_y_range(self) -> tuple[int, int]:
        """Returns the valid Y pixel range from the first plane's registration."""
        return self._contexts[0].runtime.registration.valid_y_range

    @property
    def valid_x_range(self) -> tuple[int, int]:
        """Returns the valid X pixel range from the first plane's registration."""
        return self._contexts[0].runtime.registration.valid_x_range

    @property
    def combined_binary(self) -> BinaryFileCombined:
        """Returns the stitched multi-plane channel 1 binary used for combined frame display."""
        return self._combined_binary

    def read_stitched_frame(self, frame_index: int) -> NDArray[np.int16]:
        """Returns a single stitched frame combining all planes for channel 1.

        Uses slice indexing internally to avoid the integer-indexing shape bug in BinaryFileCombined.
        """
        return self._combined_binary[frame_index : frame_index + 1][0]

    def read_stitched_frame_channel_2(self, frame_index: int) -> NDArray[np.int16]:
        """Returns a single stitched frame combining all planes for channel 2.

        Uses slice indexing internally to avoid the integer-indexing shape bug in BinaryFileCombined.
        """
        if self._combined_binary_channel_2 is None:
            console.error(
                message="Unable to read a stitched channel 2 frame. The recording is single-channel.",
                error=RuntimeError,
            )
        return self._combined_binary_channel_2[frame_index : frame_index + 1][0]

    def plane_rigid_offsets(self, plane_index: int) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
        """Returns the rigid registration (y, x) offset arrays for a specific plane without mutating _view_index.

        Falls back to zero arrays when the underlying registration data is None.
        """
        registration = self._contexts[plane_index].runtime.registration
        frame_count = self._contexts[plane_index].runtime.io.frame_count
        y = registration.rigid_y_offsets
        x = registration.rigid_x_offsets
        zeros = np.zeros(frame_count, dtype=np.int32)
        return (y if y is not None else zeros, x if x is not None else zeros)

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
    def from_data(cls, root_path: Path, view_index: int = -1) -> SingleDayData:
        """Loads single-day pipeline data with combined and per-plane results.

        Loads all planes via ``RuntimeContext.load()`` and the combined data from the root cindra output directory. All
        trace arrays are memory-mapped from disk at initialization.

        Args:
            root_path: Root cindra output directory containing configuration.yaml.
            view_index: The initial view index. -1 selects the combined view, 0+ selects a per-plane view.

        Returns:
            A fully populated SingleDayData instance.
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


@dataclass
class MultiDayData:
    """Wraps the output of the multi-day processing pipeline for a single session and serves it to consumer GUIs.

    Each instance represents one session within a multi-day dataset. All arrays are resolved eagerly at
    initialization: ``.npy`` files are memory-mapped and ``.npz`` files are loaded into RAM.
    """

    _context: MultiDayRuntimeContext = field(repr=False)
    """The multi-day runtime context for this session."""

    def __post_init__(self) -> None:
        """Memory-maps registration arrays and eagerly loads tracking and extraction arrays for this session."""
        output_path = self._runtime.output_path
        if output_path is not None:
            self._runtime.registration.memory_map_arrays(output_path)
            self._runtime.tracking.load_arrays(output_path)
            self._runtime.extraction.load_arrays(output_path)
            self._runtime.extraction.memory_map_results(output_path)
        combined = self._runtime.combined_data
        if combined is not None and self._runtime.io.data_path is not None:
            combined.detection.memory_map_arrays(self._runtime.io.data_path)
            combined.extraction.memory_map_arrays(self._runtime.io.data_path)

    @property
    def session_id(self) -> str:
        """Returns the recording identifier string."""
        return self._runtime.io.session_id

    @property
    def runtime_dataset_name(self) -> str:
        """Returns the dataset name stored in runtime IO metadata."""
        return self._runtime.io.dataset_name

    @property
    def data_path(self) -> Path:
        """Returns the session's single-day cindra root path."""
        value = self._runtime.io.data_path
        if value is None:
            console.error(
                message=f"Unable to retrieve the data path for multi-day session '{self.session_id}'. "
                f"The pipeline data is incomplete or corrupt.",
                error=RuntimeError,
            )
        return value

    @property
    def has_channel_2(self) -> bool:
        """Returns True if channel 2 data is available."""
        return self._detection.mean_image_channel_2 is not None

    @property
    def mean_image(self) -> NDArray[np.float32]:
        """Returns the native-space mean fluorescence image."""
        value = self._detection.mean_image
        if value is None:
            console.error(
                message=f"Unable to retrieve the mean image for multi-day session '{self.session_id}'. "
                f"The pipeline data is incomplete or corrupt.",
                error=RuntimeError,
            )
        return value

    @property
    def enhanced_mean_image(self) -> NDArray[np.float32]:
        """Returns the native-space contrast-enhanced mean image."""
        value = self._detection.enhanced_mean_image
        if value is None:
            console.error(
                message=f"Unable to retrieve the enhanced mean image for multi-day session '{self.session_id}'. "
                f"The pipeline data is incomplete or corrupt.",
                error=RuntimeError,
            )
        return value

    @property
    def maximum_projection(self) -> NDArray[np.float32]:
        """Returns the native-space maximum intensity projection."""
        value = self._detection.maximum_projection
        if value is None:
            console.error(
                message=f"Unable to retrieve the maximum projection for multi-day session '{self.session_id}'. "
                f"The pipeline data is incomplete or corrupt.",
                error=RuntimeError,
            )
        return value

    @property
    def correlation_map(self) -> NDArray[np.float32]:
        """Returns the native-space pixel correlation map."""
        value = self._detection.correlation_map
        if value is None:
            console.error(
                message=f"Unable to retrieve the correlation map for multi-day session '{self.session_id}'. "
                f"The pipeline data is incomplete or corrupt.",
                error=RuntimeError,
            )
        return value

    @property
    def mean_image_channel_2(self) -> NDArray[np.float32]:
        """Returns the channel 2 native-space mean fluorescence image. Empty if single-channel."""
        value = self._detection.mean_image_channel_2
        return value if value is not None else EMPTY

    @property
    def enhanced_mean_image_channel_2(self) -> NDArray[np.float32]:
        """Returns the channel 2 native-space contrast-enhanced mean image. Empty if single-channel."""
        value = self._detection.enhanced_mean_image_channel_2
        return value if value is not None else EMPTY

    @property
    def maximum_projection_channel_2(self) -> NDArray[np.float32]:
        """Returns the channel 2 native-space maximum intensity projection. Empty if single-channel."""
        value = self._detection.maximum_projection_channel_2
        return value if value is not None else EMPTY

    @property
    def correlation_map_channel_2(self) -> NDArray[np.float32]:
        """Returns the channel 2 native-space pixel correlation map. Empty if single-channel."""
        value = self._detection.correlation_map_channel_2
        return value if value is not None else EMPTY

    @property
    def transformed_mean_image(self) -> NDArray[np.float32]:
        """Returns the mean image in deformed (registered) space."""
        value = self._registration.transformed_mean_image
        if value is None:
            console.error(
                message=f"Unable to retrieve the transformed mean image for multi-day session '{self.session_id}'. "
                f"The pipeline data is incomplete or corrupt.",
                error=RuntimeError,
            )
        return value

    @property
    def transformed_enhanced_mean_image(self) -> NDArray[np.float32]:
        """Returns the enhanced mean image in deformed (registered) space."""
        value = self._registration.transformed_enhanced_mean_image
        if value is None:
            console.error(
                message=f"Unable to retrieve the transformed enhanced mean image for multi-day session "
                f"'{self.session_id}'. This indicates incomplete or corrupt pipeline data.",
                error=RuntimeError,
            )
        return value

    @property
    def transformed_maximum_projection(self) -> NDArray[np.float32]:
        """Returns the maximum projection in deformed (registered) space."""
        value = self._registration.transformed_maximum_projection
        if value is None:
            console.error(
                message=f"Unable to retrieve the transformed maximum projection for multi-day session "
                f"'{self.session_id}'. This indicates incomplete or corrupt pipeline data.",
                error=RuntimeError,
            )
        return value

    @property
    def transformed_mean_image_channel_2(self) -> NDArray[np.float32]:
        """Returns the channel 2 mean image in deformed (registered) space. Empty if single-channel."""
        value = self._registration.transformed_mean_image_channel_2
        return value if value is not None else EMPTY

    @property
    def transformed_enhanced_mean_image_channel_2(self) -> NDArray[np.float32]:
        """Returns the channel 2 enhanced mean image in deformed (registered) space. Empty if single-channel."""
        value = self._registration.transformed_enhanced_mean_image_channel_2
        return value if value is not None else EMPTY

    @property
    def transformed_maximum_projection_channel_2(self) -> NDArray[np.float32]:
        """Returns the channel 2 maximum projection in deformed (registered) space. Empty if single-channel."""
        value = self._registration.transformed_maximum_projection_channel_2
        return value if value is not None else EMPTY

    @property
    def original_masks(self) -> list[ROIStatistics]:
        """Returns the selected ROI masks that were used as input to forward deformation."""
        all_masks = self._combined.extraction.roi_statistics
        if all_masks is None:
            console.error(
                message=f"Unable to retrieve the original masks for multi-day session '{self.session_id}'. "
                f"The pipeline data is incomplete or corrupt.",
                error=RuntimeError,
            )
        selected_indices = self._runtime.io.selected_cell_indices
        if not selected_indices:
            return list(all_masks)
        return [all_masks[i] for i in selected_indices]

    @property
    def original_masks_channel_2(self) -> list[ROIStatistics]:
        """Returns the channel 2 selected ROI masks. Empty if single-channel."""
        all_masks = self._combined.extraction.roi_statistics_channel_2
        if all_masks is None:
            return []
        selected_indices = self._runtime.io.selected_cell_indices_channel_2
        if not selected_indices:
            return list(all_masks)
        return [all_masks[i] for i in selected_indices]

    @property
    def deformed_masks(self) -> list[ROIMask]:
        """Returns the ROI masks warped to the shared coordinate space."""
        value = self._registration.deformed_cell_masks
        if value is None:
            console.error(
                message=f"Unable to retrieve the deformed masks for multi-day session '{self.session_id}'. "
                f"The pipeline data is incomplete or corrupt.",
                error=RuntimeError,
            )
        return value

    @property
    def deformed_masks_channel_2(self) -> list[ROIMask]:
        """Returns the channel 2 ROI masks warped to the shared coordinate space. Empty if single-channel."""
        value = self._registration.deformed_cell_masks_channel_2
        return value if value is not None else []

    @property
    def template_masks(self) -> list[ROIMask]:
        """Returns the consensus template masks from cross-recording tracking."""
        value = self._tracking.template_masks
        if value is None:
            console.error(
                message=f"Unable to retrieve the template masks for multi-day session '{self.session_id}'. "
                f"The pipeline data is incomplete or corrupt.",
                error=RuntimeError,
            )
        return value

    @property
    def template_masks_channel_2(self) -> list[ROIMask]:
        """Returns the channel 2 consensus template masks from cross-recording tracking. Empty if single-channel."""
        value = self._tracking.template_masks_channel_2
        return value if value is not None else []

    @property
    def tracked_masks(self) -> list[ROIStatistics]:
        """Returns the template ROI masks backward-deformed to this session's native coordinate space."""
        value = self._runtime.extraction.roi_statistics
        if value is None:
            console.error(
                message=f"Unable to retrieve the tracked masks for multi-day session '{self.session_id}'. "
                f"The pipeline data is incomplete or corrupt.",
                error=RuntimeError,
            )
        return value

    @property
    def tracked_masks_channel_2(self) -> list[ROIStatistics]:
        """Returns the channel 2 template ROI masks backward-deformed to native space. Empty if single-channel."""
        value = self._runtime.extraction.roi_statistics_channel_2
        return value if value is not None else []

    @property
    def cell_fluorescence(self) -> NDArray[np.float32]:
        """Returns the cell fluorescence array for this session's extraction."""
        value = self._runtime.extraction.cell_fluorescence
        if value is None:
            console.error(
                message=f"Unable to retrieve the cell fluorescence traces for multi-day session '{self.session_id}'. "
                f"The pipeline data is incomplete or corrupt.",
                error=RuntimeError,
            )
        return value

    @property
    def neuropil_fluorescence(self) -> NDArray[np.float32]:
        """Returns the neuropil fluorescence array for this session's extraction."""
        value = self._runtime.extraction.neuropil_fluorescence
        if value is None:
            console.error(
                message=f"Unable to retrieve the neuropil fluorescence traces for multi-day session "
                f"'{self.session_id}'. This indicates incomplete or corrupt pipeline data.",
                error=RuntimeError,
            )
        return value

    @property
    def subtracted_fluorescence(self) -> NDArray[np.float32]:
        """Returns the baseline-and-neuropil-subtracted fluorescence traces for this session's extraction."""
        value = self._runtime.extraction.subtracted_fluorescence
        if value is None:
            console.error(
                message=f"Unable to retrieve the subtracted fluorescence traces for multi-day session "
                f"'{self.session_id}'. The pipeline data is incomplete or corrupt.",
                error=RuntimeError,
            )
        return value

    @property
    def spikes(self) -> NDArray[np.float32]:
        """Returns the deconvolved spikes array for this session's extraction."""
        value = self._runtime.extraction.spikes
        if value is None:
            console.error(
                message=f"Unable to retrieve the deconvolved spikes for multi-day session '{self.session_id}'. "
                f"The pipeline data is incomplete or corrupt.",
                error=RuntimeError,
            )
        return value

    @property
    def _runtime(self) -> MultiDayRuntimeData:
        """Returns the runtime data for this session."""
        return self._context.runtime

    @property
    def _registration(self) -> MultiDayRegistrationData:
        """Returns the registration data for this session."""
        return self._runtime.registration

    @property
    def _tracking(self) -> MultiDayTrackingData:
        """Returns the tracking data for this session."""
        return self._runtime.tracking

    @property
    def _combined(self) -> CombinedData:
        """Returns the combined single-day data for this session."""
        value = self._runtime.combined_data
        if value is None:
            console.error(
                message=f"Unable to retrieve the combined single-day data for multi-day session '{self.session_id}'. "
                f"The pipeline data is incomplete or corrupt.",
                error=RuntimeError,
            )
        return value

    @property
    def _detection(self) -> DetectionData:
        """Returns the combined detection data for this session."""
        return self._combined.detection


@dataclass
class ViewerData:
    """Provides the top-level data entry point for all consumer GUIs.

    Binds single-day data with an optionally loaded multi-day dataset. Consumer GUIs access the underlying
    ``single_day`` and per-recording ``MultiDayData`` objects directly via ``current_recording`` or
    ``recording(index)``.
    """

    single_day: SingleDayData
    """The single-day pipeline data, always present."""

    _recordings: list[MultiDayData] = field(default_factory=list)
    """The loaded multi-day dataset's per-session MultiDayData instances. Empty when the visualized recording
    only has single-day data."""

    _available_datasets: tuple[str, ...] = ()
    """Natsorted dataset names discovered under the root path, used to inform consumers on multi-day datasets
    that include the visualized recording session."""

    _active_dataset_name: str = ""
    """The name of the active multi-day dataset. Empty when the visualized recording only has single-day data."""

    _current_recording_index: int = 0
    """The index into ``_recordings`` for the currently active recording."""

    dataset_name: str = ""
    """Display label for the active multi-day dataset."""

    @property
    def is_multi_day(self) -> bool:
        """Returns True when multi-day data is loaded and active."""
        return bool(self._recordings) and bool(self._active_dataset_name)

    @property
    def available_datasets(self) -> tuple[str, ...]:
        """Returns the natsorted names of all discovered multi-day datasets that use the visualized recording."""
        return self._available_datasets

    @property
    def active_dataset_name(self) -> str:
        """Returns the name of the active multi-day dataset, or empty string for single-day mode."""
        return self._active_dataset_name

    @property
    def recording_count(self) -> int:
        """Returns the number of recordings in the loaded dataset."""
        return len(self._recordings)

    @property
    def recording_ids(self) -> tuple[str, ...]:
        """Returns the recording identifier strings for all recordings in the loaded dataset."""
        return tuple(recording.session_id for recording in self._recordings)

    @property
    def current_recording_index(self) -> int:
        """Returns the index of the currently displayed recording."""
        return self._current_recording_index

    @property
    def current_recording_id(self) -> str:
        """Returns the recording identifier for the currently displayed recording."""
        if self._recordings:
            return self._recordings[self._current_recording_index].session_id
        return ""

    @property
    def current_recording(self) -> MultiDayData:
        """Returns the MultiDayData for the currently active recording."""
        return self._recordings[self._current_recording_index]

    def recording(self, index: int) -> MultiDayData:
        """Returns the MultiDayData for the recording at the given index.

        Args:
            index: The recording index.

        Returns:
            The MultiDayData instance for the specified recording.
        """
        return self._recordings[index]

    def load_dataset(self, dataset_name: str) -> None:
        """Unloads current multi-day data and loads the named dataset.

        Constructs ``MultiDayData`` per session in the target dataset. Failed sessions are logged as warnings and
        excluded. Resolves the anchor session index for the loaded single-day data.

        Args:
            dataset_name: The name of the dataset to load (must be in ``available_datasets``).
        """
        if dataset_name not in self._available_datasets:
            console.echo(
                message=f"The requested '{dataset_name}' is not found in available datasets.",
                level=LogLevel.WARNING,
            )
            return

        # Determines the root path to search from. Uses the single-day output path's parent.
        search_root = self.single_day.output_path.parent
        self._load_dataset_from_root(dataset_name=dataset_name, search_root=search_root)

    def unload_dataset(self) -> None:
        """Drops multi-day data and switches to single-day mode."""
        self._recordings = []
        self._active_dataset_name = ""
        self._current_recording_index = 0
        self.dataset_name = ""

    def switch_recording(self, recording_index: int) -> None:
        """Switches the active recording.

        Args:
            recording_index: The index of the recording to switch to.

        Raises:
            ValueError: If the index is out of range.
        """
        if recording_index < 0 or recording_index >= len(self._recordings):
            message = (
                f"Unable to switch the viewer to recording {recording_index}. Valid range is 0 to "
                f"{len(self._recordings) - 1}."
            )
            console.error(message=message, error=ValueError)
        self._current_recording_index = recording_index

    @classmethod
    def from_data(cls, root_path: Path, *, dataset: str | None = None) -> ViewerData:
        """Loads single-day data and optionally a multi-day dataset.

        Loads single-day pipeline data from ``root_path``, then discovers available multi-day dataset names
        (lightweight). If ``dataset`` is provided, loads that dataset; otherwise loads the first natsorted dataset if
        any exist. Starts in multi-day mode if a dataset was loaded.

        Args:
            root_path: Root cindra output directory.
            dataset: Explicit dataset name to load. Loads first available if None.

        Returns:
            A fully populated ViewerData instance.
        """
        single_day = SingleDayData.from_data(root_path=root_path)
        available = cls._discover_dataset_names(root_path=root_path)

        instance = cls(
            single_day=single_day,
            _available_datasets=available,
        )

        # Determines which dataset to load.
        target = dataset if dataset is not None else (available[0] if available else None)
        if target is not None:
            instance._load_dataset_from_root(dataset_name=target, search_root=root_path)

        return instance

    @staticmethod
    def _discover_dataset_names(root_path: Path) -> tuple[str, ...]:
        """Discovers available multi-day dataset names under root_path.

        Searches for multiday_runtime_data.yaml files, extracts the parent directory name (dataset name) from each,
        deduplicates, and returns natsorted.

        Args:
            root_path: The root directory to search recursively.

        Returns:
            A tuple of natsorted unique dataset names.
        """
        matches = list(root_path.rglob("multiday_runtime_data.yaml"))
        if not matches:
            return ()

        return tuple(natsorted({match.parent.name for match in matches}))

    def _load_dataset_from_root(self, dataset_name: str, search_root: Path) -> None:
        """Loads a specific multi-day dataset from the search root.

        Args:
            dataset_name: The dataset name to load.
            search_root: The root path to search for multiday_runtime_data.yaml files.
        """
        # Finds the target dataset directory by matching multiday_runtime_data.yaml parent names.
        matches = list(search_root.rglob("multiday_runtime_data.yaml"))
        target_dir: Path | None = None
        for match in matches:
            if match.parent.name == dataset_name:
                target_dir = match.parent
                break

        if target_dir is None:
            console.echo(
                message=f"Dataset '{dataset_name}' not found under {search_root}.",
                level=LogLevel.WARNING,
            )
            return

        try:
            contexts = MultiDayRuntimeContext.load(root_path=target_dir, session_index=-1)
            if not isinstance(contexts, list):
                contexts = [contexts]
        except Exception:
            console.echo(
                message=f"Failed to load multi-day dataset '{dataset_name}', aborting.",
                level=LogLevel.WARNING,
            )
            return

        # Constructs MultiDayData per session, skipping failures.
        recordings: list[MultiDayData] = []
        for context in contexts:
            try:
                recordings.append(MultiDayData(_context=context))
            except Exception:
                console.echo(
                    message=f"Failed to load multi-day session '{context.runtime.io.session_id}', skipping.",
                    level=LogLevel.WARNING,
                )

        self._recordings = recordings
        self._active_dataset_name = dataset_name
        self._current_recording_index = 0

        # Resolves the dataset display name from runtime data or falls back to the directory name.
        if recordings:
            runtime_name = recordings[0].runtime_dataset_name
            self.dataset_name = runtime_name or dataset_name
        else:
            self.dataset_name = dataset_name

        # Resolves the current recording index to point at the anchor session.
        single_day_root = self.single_day.output_path.parent
        for index, recording in enumerate(recordings):
            if recording.data_path == single_day_root:
                self._current_recording_index = index
                break
