"""Provides the RegistrationViewerData class for per-plane registration visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import field, dataclass

from ...dataclasses import RuntimeContext

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from numpy.typing import NDArray

    from ...dataclasses import IOData, DetectionData, RegistrationData


@dataclass
class RegistrationViewerData:
    """Wraps single-day per-plane registration data and serves it to the registration viewer GUI.

    Holds all imaging planes from a single-day session and supports switching between planes for independent
    registration quality review. Each plane has its own frame geometry, binary paths, registration offsets, and
    principal component metrics.

    Notes:
        The backing ``RuntimeContext`` instances are stored privately. All public properties delegate to the current
        plane's ``IOData``, ``RegistrationData``, and ``DetectionData`` without exposing extraction, timing, or
        configuration internals.
    """

    _contexts: list[RuntimeContext] = field(repr=False)
    """All loaded per-plane runtime contexts."""

    current_plane_index: int = 0
    """Zero-based index of the currently displayed plane."""

    # Convenience accessors for internal data objects.

    @property
    def _current_io(self) -> IOData:
        """Returns the IOData for the current plane."""
        return self._contexts[self.current_plane_index].runtime.io

    @property
    def _current_registration(self) -> RegistrationData:
        """Returns the RegistrationData for the current plane."""
        return self._contexts[self.current_plane_index].runtime.registration

    @property
    def _current_detection(self) -> DetectionData:
        """Returns the DetectionData for the current plane."""
        return self._contexts[self.current_plane_index].runtime.detection

    # Plane navigation.

    @property
    def plane_count(self) -> int:
        """Returns the total number of imaging planes."""
        return len(self._contexts)

    @property
    def plane_labels(self) -> list[str]:
        """Returns display labels for all planes."""
        return [f"Plane {context.runtime.io.plane_index}" for context in self._contexts]

    # Frame geometry (delegates to current plane's IOData).

    @property
    def frame_height(self) -> int:
        """Returns the frame height in pixels for the current plane."""
        return self._current_io.frame_height

    @property
    def frame_width(self) -> int:
        """Returns the frame width in pixels for the current plane."""
        return self._current_io.frame_width

    @property
    def frame_count(self) -> int:
        """Returns the total number of frames for the current plane."""
        return self._current_io.frame_count

    @property
    def sampling_rate(self) -> float:
        """Returns the per-plane sampling rate in Hertz."""
        return self._current_io.sampling_rate

    # Binary file paths (delegates to current plane's IOData).

    @property
    def registered_binary_path(self) -> Path | None:
        """Returns the registered binary path for the current plane."""
        return self._current_io.registered_binary_path

    @property
    def registered_binary_path_channel_2(self) -> Path | None:
        """Returns the channel 2 registered binary path for the current plane."""
        return self._current_io.registered_binary_path_channel_2

    @property
    def raw_binary_path(self) -> Path | None:
        """Returns the raw binary path if it exists on disk for the current plane."""
        output_directory = self._current_io.output_directory
        if output_directory is None:
            return None
        path = output_directory / "data_raw.bin"
        if path.exists():
            return path
        return None

    @property
    def raw_binary_path_channel_2(self) -> Path | None:
        """Returns the channel 2 raw binary path if it exists on disk for the current plane."""
        output_directory = self._current_io.output_directory
        if output_directory is None:
            return None
        path = output_directory / "data_raw_chan2.bin"
        if path.exists():
            return path
        return None

    # Registration offsets (delegates to current plane's RegistrationData).

    @property
    def rigid_y_offsets(self) -> NDArray[np.int32] | None:
        """Returns the per-frame rigid Y translation offsets for the current plane."""
        return self._current_registration.rigid_y_offsets

    @property
    def rigid_x_offsets(self) -> NDArray[np.int32] | None:
        """Returns the per-frame rigid X translation offsets for the current plane."""
        return self._current_registration.rigid_x_offsets

    @property
    def rigid_correlations(self) -> NDArray[np.float32] | None:
        """Returns the per-frame rigid phase correlation values for the current plane."""
        return self._current_registration.rigid_correlations

    @property
    def nonrigid_y_offsets(self) -> NDArray[np.float32] | None:
        """Returns the per-frame nonrigid Y offsets for the current plane."""
        return self._current_registration.nonrigid_y_offsets

    @property
    def nonrigid_x_offsets(self) -> NDArray[np.float32] | None:
        """Returns the per-frame nonrigid X offsets for the current plane."""
        return self._current_registration.nonrigid_x_offsets

    @property
    def nonrigid_correlations(self) -> NDArray[np.float32] | None:
        """Returns the per-frame nonrigid phase correlation values for the current plane."""
        return self._current_registration.nonrigid_correlations

    # Principal component metrics (delegates to current plane's RegistrationData).

    @property
    def pc_extreme_images(self) -> NDArray[np.float32] | None:
        """Returns the PC extreme mean images with shape (2, num_components, height, width)."""
        return self._current_registration.principal_component_extreme_images

    @property
    def pc_shift_metrics(self) -> NDArray[np.float32] | None:
        """Returns the PC alignment metrics with shape (num_components, 3)."""
        return self._current_registration.principal_component_shift_metrics

    @property
    def pc_projections(self) -> NDArray[np.float32] | None:
        """Returns the per-frame PC projections with shape (num_frames, num_components)."""
        return self._current_registration.principal_component_projections

    # Reference and detection images.

    @property
    def reference_image(self) -> NDArray[np.float32] | None:
        """Returns the registration reference image for the current plane."""
        return self._current_registration.reference_image

    @property
    def mean_image(self) -> NDArray[np.float32] | None:
        """Returns the temporal mean image for the current plane."""
        return self._current_detection.mean_image

    @property
    def aspect_ratio(self) -> float:
        """Returns the cell aspect ratio for the current plane."""
        return self._current_detection.aspect_ratio

    # Registration status.

    @property
    def is_registered(self) -> bool:
        """Checks whether the current plane has been registered."""
        return self._current_registration.is_registered()

    @property
    def bad_frames(self) -> NDArray[np.bool_] | None:
        """Returns the boolean bad-frame mask for the current plane."""
        return self._current_registration.bad_frames

    # Plane switching.

    def switch_plane(self, plane_index: int) -> None:
        """Switches to a different imaging plane.

        Args:
            plane_index: Zero-based index of the plane to switch to.

        Raises:
            ValueError: If the index is out of range.
        """
        if plane_index < 0 or plane_index >= len(self._contexts):
            message = (
                f"Unable to switch to plane {plane_index}. Valid range is "
                f"0 to {len(self._contexts) - 1}."
            )
            raise ValueError(message)
        self.current_plane_index = plane_index

    # Factory method.

    @classmethod
    def from_session(cls, root_path: Path) -> RegistrationViewerData:
        """Loads all planes from a single-day session for registration review.

        Args:
            root_path: Root suite2p output directory containing configuration.yaml.

        Returns:
            A fully populated RegistrationViewerData instance with all planes loaded.
        """
        contexts = RuntimeContext.load(root_path=root_path, plane_index=-1)
        if not isinstance(contexts, list):
            contexts = [contexts]
        return cls(_contexts=contexts)
