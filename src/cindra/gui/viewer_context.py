"""Provides the unified data hierarchy for all cindra GUI viewer windows."""

from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path
from dataclasses import field, dataclass

import numpy as np
from natsort import natsorted
from ataraxis_base_utilities import LogLevel, console

from ..io import BinaryFile, BinaryFileCombined
from ..dataclasses import CombinedData, RuntimeContext, MultiRecordingRuntimeContext

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..dataclasses import (
        IOData,
        ROIMask,
        DetectionData,
        ROIStatistics,
        ExtractionData,
        RegistrationData,
        MultiRecordingRuntimeData,
        MultiRecordingTrackingData,
        MultiRecordingRegistrationData,
    )

EMPTY: NDArray[np.float32] = np.empty(0, dtype=np.float32)
"""Empty array sentinel returned for absent trace or classification data."""


@dataclass
class SingleRecordingData:
    """Wraps the output of the single-recording processing pipeline for a single recording and serves it to consumer
    GUIs.

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

        # Memory-maps combined extraction data. Derives the cindra output root from the first available plane output
        # path, which is always corrected for relocated datasets by RuntimeContext.load().
        combined_output = self._contexts[0].runtime.io.output_path
        if combined_output is None:
            message = "Unable to load combined trace arrays. The output path is not set in the plane's IO data."
            console.error(message=message, error=FileNotFoundError)
        combined_output = combined_output.parent
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
                message="Unable to retrieve the ROI statistics for the current single-recording view. The pipeline "
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
                message="Unable to retrieve the cell fluorescence traces for the current single-recording view. "
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
                message="Unable to retrieve the neuropil fluorescence traces for the current single-recording view. "
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
                message="Unable to retrieve the subtracted fluorescence traces for the current single-recording view. "
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
                message="Unable to retrieve the deconvolved spikes for the current single-recording view. "
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
        """Returns the memory-mapped read-write cell classification array with shape (roi_count, 2) for the current
        view, where column 0 holds binary labels (0.0/1.0) and column 1 holds classifier probabilities.
        """
        value = self._current_extraction.cell_classification
        if value is None:
            console.error(
                message="Unable to retrieve the cell classification for the current single-recording view. "
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
        """Returns the cell colocalization array with shape (roi_count, 2) for the current view, or an empty array if
        colocalization data is unavailable.
        """
        value = self._current_extraction.cell_colocalization
        return value if value is not None else EMPTY

    @property
    def mean_image(self) -> NDArray[np.float32]:
        """Returns the mean image from the current view's detection data."""
        value = self._current_detection.mean_image
        if value is None:
            console.error(
                message="Unable to retrieve the mean image for the current single-recording view. The pipeline data "
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
                message="Unable to retrieve the enhanced mean image for the current single-recording view. The "
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
                message="Unable to retrieve the maximum projection for the current single-recording view. The pipeline "
                "data is incomplete or corrupt.",
                error=RuntimeError,
            )
        return value

    @property
    def correlation_map(self) -> NDArray[np.float32]:
        """Returns the correlation map from the current view's detection data."""
        value = self._current_detection.correlation_map
        if value is None:
            console.error(
                message="Unable to retrieve the correlation map for the current single-recording view. The pipeline "
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
    def roi_diameter(self) -> int:
        """Returns the estimated ROI diameter in pixels."""
        return self._current_detection.roi_diameter

    @property
    def aspect_ratio(self) -> float:
        """Returns the aspect ratio of detected ROIs."""
        return self._current_detection.aspect_ratio

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
        """Returns the mean images from frames at extreme ends of each principal component as an array with shape
        (2, num_components, height, width) where index 0 contains low-projection means and index 1 contains
        high-projection means.
        """
        return self._current_registration.principal_component_extreme_images

    @property
    def principal_component_shift_metrics(self) -> NDArray[np.float32] | None:
        """Returns the registration offset metrics computed by aligning PC extreme images as an array with shape
        (num_components, 3) where columns contain mean rigid, mean nonrigid, and maximum nonrigid offset magnitudes.
        """
        return self._current_registration.principal_component_shift_metrics

    @property
    def principal_component_projections(self) -> NDArray[np.float32] | None:
        """Returns the projection of each frame onto the principal components of the registered movie as an array
        with shape (num_frames, num_components).
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
        """Returns the trailing components of the recording data path for display labels, starting with the last 3
        path components and progressively reducing to 2, then 1, if the label exceeds 45 characters.
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
        """Returns the cindra output directory path derived from the first plane's runtime IO data."""
        value = self._contexts[0].runtime.io.output_path
        if value is None:
            console.error(
                message="Unable to retrieve the output path for the single-recording recording. The pipeline "
                "data is incomplete or corrupt.",
                error=RuntimeError,
            )
        return value.parent

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
        y_offsets = registration.rigid_y_offsets
        x_offsets = registration.rigid_x_offsets
        zero_offsets = np.zeros(frame_count, dtype=np.int32)
        return (
            y_offsets if y_offsets is not None else zero_offsets,
            x_offsets if x_offsets is not None else zero_offsets,
        )

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
    def from_data(cls, root_path: Path, view_index: int = -1) -> SingleRecordingData:
        """Loads single-recording pipeline data with combined and per-plane results.

        Loads all planes via ``RuntimeContext.load()`` and the combined data from the root cindra output directory. All
        trace arrays are memory-mapped from disk at initialization.

        Args:
            root_path: The path to the recording's root processed data directory. The method searches
                recursively for the cindra output directory via ``RuntimeContext.load()``.
            view_index: The initial view index. -1 selects the combined view, 0+ selects a per-plane view.

        Returns:
            A fully populated SingleRecordingData instance.
        """
        console.echo(message=f"Loading a single recording's data from: {root_path}.")
        console.echo(message="Resolving runtime contexts...")
        contexts = RuntimeContext.load(root_path=root_path, plane_index=-1)
        if not isinstance(contexts, list):
            contexts = [contexts]

        # Explicitly memory-maps per-plane arrays since context resolution no longer loads them eagerly. Also resolves
        # the cindra output root from the first available plane output path, since RuntimeContext.load() already
        # searched root_path recursively for the output directory.
        console.echo(message="Memory-mapping per-plane data...")
        cindra_root: Path | None = None
        for context in contexts:
            plane_output = context.runtime.io.output_path
            if plane_output is not None:
                if cindra_root is None:
                    cindra_root = plane_output.parent
                context.runtime.registration.memory_map_arrays(plane_output)
                context.runtime.detection.memory_map_arrays(plane_output)
                context.runtime.extraction.memory_map_arrays(plane_output)

        if cindra_root is None:
            message = (
                "Unable to load single-recording data. No plane output path was found in any loaded RuntimeContext."
            )
            console.error(message=message, error=FileNotFoundError)

        console.echo(message="Loading combined plane data...")
        combined = CombinedData.load(root_path=cindra_root)
        combined.detection.load_arrays(cindra_root)
        console.echo(message="Recording's data: loaded.", level=LogLevel.SUCCESS)
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
class MultiRecordingData:
    """Wraps the output of the multi-recording processing pipeline for a single recording and serves it to
    consumer GUIs.

    Each instance represents one recording within a multi-recording dataset. All arrays are resolved eagerly at
    initialization: ``.npy`` files are memory-mapped and ``.npz`` files are loaded into RAM.
    """

    _context: MultiRecordingRuntimeContext = field(repr=False)
    """The multi-recording runtime context for this recording."""

    def __post_init__(self) -> None:
        """Memory-maps registration arrays and eagerly loads tracking and extraction arrays for this recording."""
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
    def recording_id(self) -> str:
        """Returns the recording identifier string."""
        return self._runtime.io.recording_id

    @property
    def runtime_dataset_name(self) -> str:
        """Returns the dataset name stored in runtime IO metadata."""
        return self._runtime.io.dataset_name

    @property
    def data_path(self) -> Path:
        """Returns the recording's single-recording cindra root path."""
        value = self._runtime.io.data_path
        if value is None:
            console.error(
                message=f"Unable to retrieve the data path for multi-recording recording '{self.recording_id}'. "
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
                message=f"Unable to retrieve the mean image for multi-recording recording '{self.recording_id}'. "
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
                message=f"Unable to retrieve the enhanced mean image for multi-recording recording "
                f"'{self.recording_id}'. The pipeline data is incomplete or corrupt.",
                error=RuntimeError,
            )
        return value

    @property
    def maximum_projection(self) -> NDArray[np.float32]:
        """Returns the native-space maximum intensity projection."""
        value = self._detection.maximum_projection
        if value is None:
            console.error(
                message=f"Unable to retrieve the maximum projection for multi-recording recording "
                f"'{self.recording_id}'. The pipeline data is incomplete or corrupt.",
                error=RuntimeError,
            )
        return value

    @property
    def correlation_map(self) -> NDArray[np.float32]:
        """Returns the native-space pixel correlation map."""
        value = self._detection.correlation_map
        if value is None:
            console.error(
                message=f"Unable to retrieve the correlation map for multi-recording recording '{self.recording_id}'. "
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
                message=f"Unable to retrieve the transformed mean image for multi-recording recording "
                f"'{self.recording_id}'. The pipeline data is incomplete or corrupt.",
                error=RuntimeError,
            )
        return value

    @property
    def transformed_enhanced_mean_image(self) -> NDArray[np.float32]:
        """Returns the enhanced mean image in deformed (registered) space."""
        value = self._registration.transformed_enhanced_mean_image
        if value is None:
            console.error(
                message=f"Unable to retrieve the transformed enhanced mean image for multi-recording recording "
                f"'{self.recording_id}'. This indicates incomplete or corrupt pipeline data.",
                error=RuntimeError,
            )
        return value

    @property
    def transformed_maximum_projection(self) -> NDArray[np.float32]:
        """Returns the maximum projection in deformed (registered) space."""
        value = self._registration.transformed_maximum_projection
        if value is None:
            console.error(
                message=f"Unable to retrieve the transformed maximum projection for multi-recording recording "
                f"'{self.recording_id}'. The pipeline data is incomplete or corrupt.",
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
                message=f"Unable to retrieve the original masks for multi-recording recording '{self.recording_id}'. "
                f"The pipeline data is incomplete or corrupt.",
                error=RuntimeError,
            )
        selected_indices = self._runtime.io.selected_roi_indices
        if not selected_indices:
            return list(all_masks)
        return [all_masks[i] for i in selected_indices]

    @property
    def original_masks_channel_2(self) -> list[ROIStatistics]:
        """Returns the channel 2 selected ROI masks. Empty if single-channel."""
        all_masks = self._combined.extraction.roi_statistics_channel_2
        if all_masks is None:
            return []
        selected_indices = self._runtime.io.selected_roi_indices_channel_2
        if not selected_indices:
            return list(all_masks)
        return [all_masks[i] for i in selected_indices]

    @property
    def deformed_masks(self) -> list[ROIMask]:
        """Returns the ROI masks warped to the shared coordinate space."""
        value = self._registration.deformed_roi_masks
        if value is None:
            console.error(
                message=f"Unable to retrieve the deformed masks for multi-recording recording '{self.recording_id}'. "
                f"The pipeline data is incomplete or corrupt.",
                error=RuntimeError,
            )
        return value

    @property
    def deformed_masks_channel_2(self) -> list[ROIMask]:
        """Returns the channel 2 ROI masks warped to the shared coordinate space. Empty if single-channel."""
        value = self._registration.deformed_roi_masks_channel_2
        return value if value is not None else []

    @property
    def template_masks(self) -> list[ROIMask]:
        """Returns the consensus template masks from cross-recording tracking."""
        value = self._tracking.template_masks
        if value is None:
            console.error(
                message=f"Unable to retrieve the template masks for multi-recording recording '{self.recording_id}'. "
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
        """Returns the template ROI masks backward-deformed to this recording's native coordinate space."""
        value = self._runtime.extraction.roi_statistics
        if value is None:
            console.error(
                message=f"Unable to retrieve the tracked masks for multi-recording recording '{self.recording_id}'. "
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
        """Returns the cell fluorescence array for this recording's extraction."""
        value = self._runtime.extraction.cell_fluorescence
        if value is None:
            console.error(
                message=f"Unable to retrieve the cell fluorescence traces for multi-recording recording "
                f"'{self.recording_id}'. The pipeline data is incomplete or corrupt.",
                error=RuntimeError,
            )
        return value

    @property
    def neuropil_fluorescence(self) -> NDArray[np.float32]:
        """Returns the neuropil fluorescence array for this recording's extraction."""
        value = self._runtime.extraction.neuropil_fluorescence
        if value is None:
            console.error(
                message=f"Unable to retrieve the neuropil fluorescence traces for multi-recording recording "
                f"'{self.recording_id}'. This indicates incomplete or corrupt pipeline data.",
                error=RuntimeError,
            )
        return value

    @property
    def subtracted_fluorescence(self) -> NDArray[np.float32]:
        """Returns the baseline-and-neuropil-subtracted fluorescence traces for this recording's extraction."""
        value = self._runtime.extraction.subtracted_fluorescence
        if value is None:
            console.error(
                message=f"Unable to retrieve the subtracted fluorescence traces for multi-recording recording "
                f"'{self.recording_id}'. The pipeline data is incomplete or corrupt.",
                error=RuntimeError,
            )
        return value

    @property
    def spikes(self) -> NDArray[np.float32]:
        """Returns the deconvolved spikes array for this recording's extraction."""
        value = self._runtime.extraction.spikes
        if value is None:
            console.error(
                message=f"Unable to retrieve the deconvolved spikes for multi-recording recording "
                f"'{self.recording_id}'. The pipeline data is incomplete or corrupt.",
                error=RuntimeError,
            )
        return value

    @property
    def cell_fluorescence_channel_2(self) -> NDArray[np.float32]:
        """Returns the channel 2 cell fluorescence traces. Empty with size 0 if single-channel."""
        value = self._runtime.extraction.cell_fluorescence_channel_2
        return value if value is not None else EMPTY

    @property
    def neuropil_fluorescence_channel_2(self) -> NDArray[np.float32]:
        """Returns the channel 2 neuropil fluorescence traces. Empty with size 0 if single-channel."""
        value = self._runtime.extraction.neuropil_fluorescence_channel_2
        return value if value is not None else EMPTY

    @property
    def subtracted_fluorescence_channel_2(self) -> NDArray[np.float32]:
        """Returns the channel 2 baseline-and-neuropil-subtracted fluorescence traces. Empty with size 0 if
        single-channel.
        """
        value = self._runtime.extraction.subtracted_fluorescence_channel_2
        return value if value is not None else EMPTY

    @property
    def spikes_channel_2(self) -> NDArray[np.float32]:
        """Returns the channel 2 deconvolved spike traces. Empty with size 0 if single-channel."""
        value = self._runtime.extraction.spikes_channel_2
        return value if value is not None else EMPTY

    @property
    def cell_colocalization(self) -> NDArray[np.float32]:
        """Returns the cell colocalization array with shape (roi_count, 2), or an empty array if colocalization data is
        unavailable.
        """
        value = self._runtime.extraction.cell_colocalization
        return value if value is not None else EMPTY

    @property
    def _runtime(self) -> MultiRecordingRuntimeData:
        """Returns the runtime data for this recording."""
        return self._context.runtime

    @property
    def _registration(self) -> MultiRecordingRegistrationData:
        """Returns the registration data for this recording."""
        return self._runtime.registration

    @property
    def _tracking(self) -> MultiRecordingTrackingData:
        """Returns the tracking data for this recording."""
        return self._runtime.tracking

    @property
    def _combined(self) -> CombinedData:
        """Returns the combined single-recording data for this recording."""
        value = self._runtime.combined_data
        if value is None:
            console.error(
                message=f"Unable to retrieve the combined single-recording data for multi-recording "
                f"recording '{self.recording_id}'. The pipeline data is incomplete or corrupt.",
                error=RuntimeError,
            )
        return value

    @property
    def _detection(self) -> DetectionData:
        """Returns the combined detection data for this recording."""
        return self._combined.detection


@dataclass
class ViewerData:
    """Provides the top-level data entry point for all consumer GUIs.

    Binds single-recording data with an optionally loaded multi-recording dataset. Consumer GUIs access the underlying
    ``single_recording`` and per-recording ``MultiRecordingData`` objects directly via ``current_recording`` or
    ``recording(index)``.
    """

    single_recording: SingleRecordingData
    """The single-recording pipeline data, always present."""

    _recordings: list[MultiRecordingData] = field(default_factory=list)
    """The loaded multi-recording dataset's per-recording MultiRecordingData instances. Empty when the visualized
    recording only has single-recording data."""

    _available_datasets: tuple[str, ...] = ()
    """Natsorted dataset names discovered under the root path, used to inform consumers on multi-recording datasets
    that include the visualized recording."""

    _active_dataset_name: str = ""
    """The name of the active multi-recording dataset. Empty when the visualized recording only has
    single-recording data."""

    _current_recording_index: int = 0
    """The index into ``_recordings`` for the currently active recording."""

    _loaded_dataset_name: str = ""
    """The dataset name whose data is currently held in ``_recordings``. Persists across Original/Dataset toggles so
    that re-activating the same dataset is instant."""

    dataset_name: str = ""
    """Display label for the active multi-recording dataset."""

    @property
    def is_multi_recording(self) -> bool:
        """Returns True when multi-recording data is loaded and active."""
        return bool(self._recordings) and bool(self._active_dataset_name)

    @property
    def available_datasets(self) -> tuple[str, ...]:
        """Returns the natsorted names of all discovered multi-recording datasets that use the visualized recording."""
        return self._available_datasets

    @property
    def active_dataset_name(self) -> str:
        """Returns the name of the active multi-recording dataset, or empty string for single-recording mode."""
        return self._active_dataset_name

    @property
    def recording_count(self) -> int:
        """Returns the number of recordings in the loaded dataset."""
        return len(self._recordings)

    @property
    def recording_ids(self) -> tuple[str, ...]:
        """Returns the recording identifier strings for all recordings in the loaded dataset."""
        return tuple(recording.recording_id for recording in self._recordings)

    @property
    def current_recording_index(self) -> int:
        """Returns the index of the currently displayed recording."""
        return self._current_recording_index

    @property
    def current_recording_id(self) -> str:
        """Returns the recording identifier for the currently displayed recording."""
        if self._recordings:
            return self._recordings[self._current_recording_index].recording_id
        return ""

    @property
    def current_recording(self) -> MultiRecordingData:
        """Returns the MultiRecordingData for the currently active recording."""
        return self._recordings[self._current_recording_index]

    def recording(self, index: int) -> MultiRecordingData:
        """Returns the MultiRecordingData for the recording at the given index.

        Args:
            index: The recording index.

        Returns:
            The MultiRecordingData instance for the specified recording.
        """
        return self._recordings[index]

    def load_dataset(self, dataset_name: str) -> None:
        """Activates the named multi-recording dataset, loading from disk only when switching to a different dataset.

        If the requested dataset is already in memory, re-activates it without reloading. Only one dataset is held in
        memory at a time.

        Args:
            dataset_name: The name of the dataset to load (must be in ``available_datasets``).
        """
        if dataset_name not in self._available_datasets:
            console.echo(
                message=f"The requested '{dataset_name}' is not found in available datasets.",
                level=LogLevel.WARNING,
            )
            return

        # Re-activates the already loaded dataset without reloading from disk.
        if dataset_name == self._loaded_dataset_name and self._recordings:
            self._active_dataset_name = dataset_name
            return

        # Loads a different dataset from disk, replacing the previous one.
        search_root = self.single_recording.output_path.parent
        self._load_dataset_from_root(dataset_name=dataset_name, search_root=search_root)

    def unload_dataset(self) -> None:
        """Deactivates multi-recording mode without dropping the loaded dataset from memory."""
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
        """Loads single-recording data and discovers available multi-recording datasets.

        Loads single-recording pipeline data from ``root_path``, then discovers available multi-recording dataset names
        (lightweight). A dataset is only loaded when explicitly requested via the ``dataset`` parameter; otherwise
        the instance starts in single-recording mode and consumers can load a dataset later via the dropdown.

        Args:
            root_path: Root cindra output directory.
            dataset: Explicit dataset name to load. Stays in single-recording mode if None.

        Returns:
            A fully populated ViewerData instance.
        """
        single_recording = SingleRecordingData.from_data(root_path=root_path)
        available = cls._discover_dataset_names(root_path=root_path)

        instance = cls(
            single_recording=single_recording,
            _available_datasets=available,
        )

        if dataset is not None:
            instance._load_dataset_from_root(dataset_name=dataset, search_root=root_path)

        return instance

    @staticmethod
    def _discover_dataset_names(root_path: Path) -> tuple[str, ...]:
        """Discovers available multi-recording dataset names under root_path.

        Searches for multi_recording_runtime_data.yaml files, extracts the parent directory name (dataset name)
        from each, deduplicates, and returns natsorted.

        Args:
            root_path: The root directory to search recursively.

        Returns:
            A tuple of natsorted unique dataset names.
        """
        matches = list(root_path.rglob("multi_recording_runtime_data.yaml"))
        if not matches:
            return ()

        return tuple(natsorted({match.parent.name for match in matches}))

    def _load_dataset_from_root(self, dataset_name: str, search_root: Path) -> None:
        """Loads a specific multi-recording dataset from the search root.

        Args:
            dataset_name: The dataset name to load.
            search_root: The root path to search for multi_recording_runtime_data.yaml files.
        """
        # Finds the target dataset directory by matching multi_recording_runtime_data.yaml parent names.
        matches = list(search_root.rglob("multi_recording_runtime_data.yaml"))
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
            contexts = MultiRecordingRuntimeContext.load(root_path=target_dir, recording_index=-1)
            if not isinstance(contexts, list):
                contexts = [contexts]
        except Exception:
            console.echo(
                message=f"Failed to load multi-recording dataset '{dataset_name}', aborting.",
                level=LogLevel.WARNING,
            )
            return

        # Constructs MultiRecordingData per recording, skipping failures.
        console.echo(message=f"Loading multi-recording dataset '{dataset_name}' ({len(contexts)} recording(s))...")
        recordings: list[MultiRecordingData] = []
        for index, context in enumerate(contexts):
            recording_id = context.runtime.io.recording_id
            try:
                console.echo(message=f"  Loading recording {index + 1}/{len(contexts)}: {recording_id}...")
                recordings.append(MultiRecordingData(_context=context))
            except Exception:
                console.echo(
                    message=f"Failed to load multi-recording recording '{recording_id}', skipping.",
                    level=LogLevel.WARNING,
                )
        console.echo(
            message=f"Dataset '{dataset_name}' with {len(recordings)} recording(s): loaded.",
            level=LogLevel.SUCCESS,
        )

        self._recordings = recordings
        self._loaded_dataset_name = dataset_name
        self._active_dataset_name = dataset_name
        self._current_recording_index = 0

        # Resolves the dataset display name from runtime data or falls back to the directory name.
        if recordings:
            runtime_name = recordings[0].runtime_dataset_name
            self.dataset_name = runtime_name or dataset_name
        else:
            self.dataset_name = dataset_name

        # Resolves the current recording index to point at the anchor recording. single_recording.output_path already
        # returns the cindra root (plane output_path.parent), which matches recording.data_path directly.
        single_recording_root = self.single_recording.output_path
        for index, recording in enumerate(recordings):
            if recording.data_path == single_recording_root:
                self._current_recording_index = index
                break
