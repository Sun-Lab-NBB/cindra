"""Provides the data hierarchy for per-plane registration quality visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path
from dataclasses import field, dataclass

import numpy as np
from ataraxis_base_utilities import console

from ...io import BinaryFile
from ...dataclasses import RuntimeContext

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ...dataclasses import IOData, DetectionData, RegistrationData


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

        Starts with the last 3 path components and progressively reduces to 2, then 1, if the label
        exceeds 45 characters.
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
    def output_directory(self) -> Path | None:
        """Returns the path to the plane-specific output directory where all results are saved."""
        return self._current_io.output_directory

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
