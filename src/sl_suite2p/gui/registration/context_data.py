"""Provides the data hierarchy for per-plane registration quality visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import field, dataclass

from ataraxis_base_utilities import console

from ...dataclasses import RuntimeContext

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
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
    """

    _contexts: list[RuntimeContext] = field(repr=False)
    """The RuntimeContext instances for each registered imaging plane of the processed recording."""

    _current_plane_index: int = 0
    """The index of the plane whose data is currently displayed by the GUI application."""

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
    def registered_binary_path(self) -> Path | None:
        """Returns the path to the motion-corrected binary file for the primary imaging channel."""
        return self._current_io.registered_binary_path

    @property
    def registered_binary_path_channel_2(self) -> Path | None:
        """Returns the path to the motion-corrected binary file for the second imaging channel."""
        return self._current_io.registered_binary_path_channel_2

    @property
    def output_directory(self) -> Path | None:
        """Returns the path to the plane-specific output directory where all results are saved."""
        return self._current_io.output_directory

    @property
    def rigid_y_offsets(self) -> NDArray[np.int32] | None:
        """Returns the vertical (Y) translation offsets from rigid registration, one value per frame."""
        return self._current_registration.rigid_y_offsets

    @property
    def rigid_x_offsets(self) -> NDArray[np.int32] | None:
        """Returns the horizontal (X) translation offsets from rigid registration, one value per frame."""
        return self._current_registration.rigid_x_offsets

    @property
    def nonrigid_y_offsets(self) -> NDArray[np.float32] | None:
        """Returns the vertical (Y) translation offsets from nonrigid registration, per frame and per block."""
        return self._current_registration.nonrigid_y_offsets

    @property
    def nonrigid_x_offsets(self) -> NDArray[np.float32] | None:
        """Returns the horizontal (X) translation offsets from nonrigid registration, per frame and per block."""
        return self._current_registration.nonrigid_x_offsets

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
    def aspect_ratio(self) -> float:
        """Returns the aspect ratio of detected ROIs, computed as vertical to horizontal diameter ratio."""
        return self._current_detection.aspect_ratio

    @classmethod
    def from_session(cls, root_path: Path) -> RegistrationViewerData:
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
        self._current_plane_index = plane_index

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
