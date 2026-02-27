"""Provides data hierarchies for single-day pipeline viewers."""

from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path
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
        DetectionData,
        ROIStatistics,
        ExtractionData,
        RegistrationData,
    )


@dataclass
class ViewTraces:
    """Holds memory-mapped trace, classification, and colocalization arrays for a single channel of a single view.

    Notes:
        All fluorescence and spike traces are memory-mapped read-only. The cell classification array is memory-mapped
        read-write so reclassification writes propagate directly to disk. Colocalization is shared across channels and
        memory-mapped read-only from the same source file.
    """

    cell_fluorescence: NDArray[np.float32]
    """Cell fluorescence traces with shape (roi_count, frames)."""

    neuropil_fluorescence: NDArray[np.float32]
    """Neuropil fluorescence traces with shape (roi_count, frames)."""

    subtracted_fluorescence: NDArray[np.float32]
    """Memory-mapped baseline-and-neuropil-subtracted fluorescence traces with shape (roi_count, frames)."""

    spikes: NDArray[np.float32]
    """Deconvolved spike traces with shape (roi_count, frames)."""

    cell_classification: NDArray[np.float32]
    """Cell classification array with shape (roi_count, 2)."""

    cell_colocalization: NDArray[np.float32]
    """Cell colocalization array with shape (roi_count, 2). Empty with size 0 if absent."""


_EMPTY_TRACES: ViewTraces = ViewTraces(
    cell_fluorescence=np.empty(0, dtype=np.float32),
    neuropil_fluorescence=np.empty(0, dtype=np.float32),
    subtracted_fluorescence=np.empty(0, dtype=np.float32),
    spikes=np.empty(0, dtype=np.float32),
    cell_classification=np.empty(0, dtype=np.float32),
    cell_colocalization=np.empty(0, dtype=np.float32),
)
"""Sentinel ViewTraces instance with all-empty arrays, returned for single-channel views."""


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

    _channel_1_traces: dict[int, ViewTraces] = field(init=False)
    """Channel 1 trace arrays keyed by view index (-1 for combined, 0+ for per-plane)."""

    _channel_2_traces: dict[int, ViewTraces] = field(init=False)
    """Channel 2 trace arrays keyed by view index (-1 for combined, 0+ for per-plane). Missing keys indicate
    single-channel views."""

    _view_labels: tuple[str, ...] = field(init=False)
    """Cached display labels for all available views including the combined view."""

    def __post_init__(self) -> None:
        """Opens binary file handles and memory-maps all trace arrays for every view."""
        # Opens per-plane binary files for both channels.
        self._channel_1_binaries = {}
        self._channel_2_binaries = {}
        channel_2_paths = self._combined.registered_binary_paths_channel_2
        for index, (height, width, path) in enumerate(
            zip(self._combined.plane_heights, self._combined.plane_widths, self._combined.registered_binary_paths),
        ):
            self._channel_1_binaries[index] = BinaryFile(height=int(height), width=int(width), file_path=path)
            if channel_2_paths is not None:
                self._channel_2_binaries[index] = BinaryFile(
                    height=int(height),
                    width=int(width),
                    file_path=channel_2_paths[index],
                )

        # Memory-maps combined view traces for both channels.
        self._channel_1_traces = {}
        self._channel_2_traces = {}
        combined_output = self._contexts[0].configuration.file_io.output_path
        if combined_output is None:
            message = "Unable to load combined trace arrays. The output path is not set in the configuration."
            console.error(message=message, error=FileNotFoundError)
        self._channel_1_traces[-1] = _create_view_traces(
            extraction=self._combined.extraction,
            output_path=combined_output,
        )
        channel_2 = _create_channel_2_traces(output_path=combined_output)
        if channel_2 is not None:
            self._channel_2_traces[-1] = channel_2

        # Memory-maps per-plane traces for both channels.
        for index, context in enumerate(self._contexts):
            runtime = context.runtime
            plane_output = runtime.io.output_path
            if plane_output is None:
                message = (
                    f"Unable to load arrays for plane {runtime.io.plane_index}. The output path is not set in the "
                    f"plane's IO data."
                )
                console.error(message=message, error=FileNotFoundError)
            self._channel_1_traces[index] = _create_view_traces(
                extraction=runtime.extraction,
                output_path=plane_output,
            )
            channel_2 = _create_channel_2_traces(output_path=plane_output)
            if channel_2 is not None:
                self._channel_2_traces[index] = channel_2

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
        return int(self._current_traces.cell_fluorescence.shape[1])

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
        if self._view_index == -1:
            statistics = self._combined.extraction.roi_statistics
        else:
            statistics = self._contexts[self._view_index].runtime.extraction.roi_statistics
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
        return self._current_traces.cell_fluorescence

    @property
    def cell_fluorescence_channel_2(self) -> NDArray[np.float32]:
        """Returns the channel 2 cell fluorescence traces. Empty with size 0 if single-channel."""
        return self._current_channel_2_traces.cell_fluorescence

    @property
    def neuropil_fluorescence(self) -> NDArray[np.float32]:
        """Returns the neuropil fluorescence traces with shape (cells, frames) for the current view."""
        return self._current_traces.neuropil_fluorescence

    @property
    def neuropil_fluorescence_channel_2(self) -> NDArray[np.float32]:
        """Returns the channel 2 neuropil fluorescence traces. Empty with size 0 if single-channel."""
        return self._current_channel_2_traces.neuropil_fluorescence

    @property
    def subtracted_fluorescence(self) -> NDArray[np.float32]:
        """Returns the memory-mapped baseline-and-neuropil-subtracted fluorescence traces."""
        return self._current_traces.subtracted_fluorescence

    @property
    def subtracted_fluorescence_channel_2(self) -> NDArray[np.float32]:
        """Returns the memory-mapped channel 2 subtracted fluorescence traces. Empty with size 0 if unavailable."""
        return self._current_channel_2_traces.subtracted_fluorescence

    @property
    def spikes(self) -> NDArray[np.float32]:
        """Returns the deconvolved spike traces with shape (cells, frames) for the current view."""
        return self._current_traces.spikes

    @property
    def spikes_channel_2(self) -> NDArray[np.float32]:
        """Returns the channel 2 deconvolved spike traces. Empty with size 0 if single-channel."""
        return self._current_channel_2_traces.spikes

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
        return self._current_traces.cell_classification

    @property
    def cell_classification_channel_2(self) -> NDArray[np.float32]:
        """Returns the channel 2 cell classification array with shape (roi_count, 2). Empty if single-channel."""
        return self._current_channel_2_traces.cell_classification

    @property
    def cell_colocalization(self) -> NDArray[np.float32]:
        """Returns the cell colocalization array with shape (roi_count, 2) for the current view.

        Column 0 holds classifier probabilities, column 1 holds labels (0.0/1.0). Read-only memory-mapped.
        """
        return self._current_traces.cell_colocalization

    @property
    def cell_colocalization_channel_2(self) -> NDArray[np.float32]:
        """Returns the channel 2 cell colocalization array. Empty if single-channel."""
        return self._current_channel_2_traces.cell_colocalization

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
        if self._view_index == -1:
            return self._combined.extraction.corrected_structural_mean_image
        return self._contexts[self._current_plane_index].runtime.extraction.corrected_structural_mean_image

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
        offsets = self._current_registration.rigid_y_offsets
        if offsets is not None:
            return offsets
        return np.zeros((self._current_io.frame_count,), dtype=np.int32)

    @property
    def rigid_x_offsets(self) -> NDArray[np.int32]:
        """Returns the horizontal (X) translation offsets from rigid registration, one value per frame or a zero array
        when the underlying data is None.
        """
        offsets = self._current_registration.rigid_x_offsets
        if offsets is not None:
            return offsets
        return np.zeros((self._current_io.frame_count,), dtype=np.int32)

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
            return str(data_path.name)
        return ""

    @property
    def valid_y_range(self) -> tuple[int, int]:
        """Returns the valid Y pixel range from the first plane's registration."""
        if self._contexts:
            return self._contexts[0].runtime.registration.valid_y_range
        return 0, 0

    @property
    def valid_x_range(self) -> tuple[int, int]:
        """Returns the valid X pixel range from the first plane's registration."""
        if self._contexts:
            return self._contexts[0].runtime.registration.valid_x_range
        return 0, 0

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

    def close(self) -> None:
        """Closes all memory-mapped binary file handles."""
        for binary in self._channel_1_binaries.values():
            binary.close()
        for binary in self._channel_2_binaries.values():
            binary.close()

    @classmethod
    def from_single_day(cls, root_path: Path) -> SingleDayViewerData:
        """Loads single-day pipeline data with combined multi-plane results.

        Loads all planes via ``RuntimeContext.load()`` and the combined data from the root cindra directory. Starts
        in the combined view. All trace arrays are memory-mapped from disk at initialization.

        Args:
            root_path: Root cindra output directory containing configuration.yaml.

        Returns:
            A fully populated SingleDayViewerData instance starting in combined view.
        """
        contexts = RuntimeContext.load(root_path=root_path, plane_index=-1)
        if not isinstance(contexts, list):
            contexts = [contexts]

        combined = CombinedData.load(root_path=root_path)
        return cls(_contexts=contexts, _combined=combined, _view_index=-1)

    @classmethod
    def from_recording(cls, root_path: Path) -> SingleDayViewerData:
        """Loads per-plane and combined data for the registration viewer.

        Loads all planes via ``RuntimeContext.load()`` and combined data from the root cindra directory. Starts in
        the first plane's view. All trace arrays are memory-mapped from disk at initialization.

        Args:
            root_path: The path to the root single-day pipeline output directory.

        Returns:
            A fully populated SingleDayViewerData instance starting in plane 0 view.
        """
        contexts = RuntimeContext.load(root_path=root_path, plane_index=-1)
        if not isinstance(contexts, list):
            contexts = [contexts]
        combined = CombinedData.load(root_path=root_path)
        return cls(_contexts=contexts, _combined=combined, _view_index=0)

    @classmethod
    def from_dialog(cls) -> SingleDayViewerData | None:
        """Opens a file dialog to select a cindra output directory and loads data from it.

        Returns:
            A fully populated SingleDayViewerData instance, or None if the dialog was canceled.
        """
        name = QFileDialog.getExistingDirectory(caption="Open cindra output directory")
        if not name:
            return None

        return cls.from_single_day(root_path=Path(name))

    @property
    def _current_plane_index(self) -> int:
        """Returns the plane index for per-plane data access, clamping -1 to 0."""
        return max(0, self._view_index)

    @property
    def _current_traces(self) -> ViewTraces:
        """Returns the channel 1 trace data for the currently active view."""
        return self._channel_1_traces[self._view_index]

    @property
    def _current_channel_2_traces(self) -> ViewTraces:
        """Returns the channel 2 trace data for the currently active view, or empty sentinel if single-channel."""
        return self._channel_2_traces.get(self._view_index, _EMPTY_TRACES)

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


def _create_view_traces(
    extraction: ExtractionData,
    output_path: Path,
) -> ViewTraces:
    """Memory-maps all channel 1 trace, classification, and colocalization arrays for a single view.

    Args:
        extraction: The extraction data source used to create the classification file when it does not yet exist on
            disk.
        output_path: The output directory containing .npy trace files for memory mapping.

    Returns:
        A fully populated ViewTraces instance with all channel 1 arrays memory-mapped.
    """
    _empty = np.empty(0, dtype=np.float32)

    # Memory-maps fluorescence traces (read-only).
    cell_fluorescence: NDArray[np.float32] = np.load(output_path / "cell_fluorescence.npy", mmap_mode="r")
    neuropil_fluorescence: NDArray[np.float32] = np.load(output_path / "neuropil_fluorescence.npy", mmap_mode="r")
    spikes: NDArray[np.float32] = np.load(output_path / "spikes.npy", mmap_mode="r")

    # Memory-maps subtracted fluorescence traces (read-only).
    subtracted_path = output_path / "subtracted_fluorescence.npy"
    subtracted_fluorescence: NDArray[np.float32] = (
        np.load(subtracted_path, mmap_mode="r") if subtracted_path.exists() else _empty
    )

    # Resolves cell classification (r+ mmap for direct disk writes). Creates the file if it does not exist.
    classification_path = output_path / "cell_classification.npy"
    if not classification_path.exists():
        if extraction.cell_classification is not None:
            classification_data = extraction.cell_classification.astype(np.float32)
        else:
            roi_count = len(extraction.roi_statistics) if extraction.roi_statistics is not None else 0
            classification_data = np.ones((roi_count, 2), dtype=np.float32)
        np.save(file=str(classification_path), arr=classification_data)
    cell_classification: NDArray[np.float32] = np.load(classification_path, mmap_mode="r+")

    # Memory-maps cell colocalization (read-only). Uses an empty array when colocalization data is absent.
    colocalization_path = output_path / "cell_colocalization.npy"
    cell_colocalization: NDArray[np.float32] = (
        np.load(colocalization_path, mmap_mode="r") if colocalization_path.exists() else _empty
    )

    return ViewTraces(
        cell_fluorescence=cell_fluorescence,
        neuropil_fluorescence=neuropil_fluorescence,
        subtracted_fluorescence=subtracted_fluorescence,
        spikes=spikes,
        cell_classification=cell_classification,
        cell_colocalization=cell_colocalization,
    )


def _create_channel_2_traces(output_path: Path) -> ViewTraces | None:
    """Memory-maps all channel 2 trace, classification, and colocalization arrays for a single view.

    Returns None if the channel 2 fluorescence file does not exist, indicating the recording is single-channel.
    Colocalization is memory-mapped read-only from the same source file as channel 1.

    Args:
        output_path: The output directory containing .npy trace files for memory mapping.

    Returns:
        A fully populated ViewTraces instance with all channel 2 arrays memory-mapped, or None if single-channel.
    """
    _empty = np.empty(0, dtype=np.float32)

    # Checks for channel 2 fluorescence file existence as the definitive channel 2 indicator.
    channel_2_fluorescence_path = output_path / "cell_fluorescence_channel_2.npy"
    if not channel_2_fluorescence_path.exists():
        return None

    # Memory-maps channel 2 fluorescence traces (read-only).
    cell_fluorescence: NDArray[np.float32] = np.load(channel_2_fluorescence_path, mmap_mode="r")
    neuropil_fluorescence: NDArray[np.float32] = np.load(
        output_path / "neuropil_fluorescence_channel_2.npy",
        mmap_mode="r",
    )

    # Memory-maps channel 2 subtracted fluorescence (read-only).
    subtracted_path = output_path / "subtracted_fluorescence_channel_2.npy"
    subtracted_fluorescence: NDArray[np.float32] = (
        np.load(subtracted_path, mmap_mode="r") if subtracted_path.exists() else _empty
    )

    # Memory-maps channel 2 spikes (read-only).
    spikes_path = output_path / "spikes_channel_2.npy"
    spikes: NDArray[np.float32] = np.load(spikes_path, mmap_mode="r") if spikes_path.exists() else _empty

    # Memory-maps channel 2 classification (r+ mmap). Uses an empty array if the file does not exist.
    classification_path = output_path / "cell_classification_channel_2.npy"
    cell_classification: NDArray[np.float32] = (
        np.load(classification_path, mmap_mode="r+") if classification_path.exists() else _empty
    )

    # Memory-maps colocalization from the same source file as channel 1 (read-only).
    colocalization_path = output_path / "cell_colocalization.npy"
    cell_colocalization: NDArray[np.float32] = (
        np.load(colocalization_path, mmap_mode="r") if colocalization_path.exists() else _empty
    )

    return ViewTraces(
        cell_fluorescence=cell_fluorescence,
        neuropil_fluorescence=neuropil_fluorescence,
        subtracted_fluorescence=subtracted_fluorescence,
        spikes=spikes,
        cell_classification=cell_classification,
        cell_colocalization=cell_colocalization,
    )
