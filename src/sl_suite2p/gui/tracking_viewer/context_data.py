"""Provides the data model for the multi-day tracking viewer."""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING
from dataclasses import field, dataclass

from ...dataclasses import MultiDayRuntimeContext

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from numpy.typing import NDArray

    from ...dataclasses.multi_day_data import (
        MultiDayIOData,
        MultiDayRuntimeData,
        MultiDayTrackingData,
        MultiDayRegistrationData,
    )
    from ...dataclasses.single_day_data import CombinedData, DetectionData, ROIStatistics


class MaskLayer(IntEnum):
    """Selects the active ROI mask layer."""

    ORIGINAL = 0
    """Displays the original ROI masks from single-day extraction in native session coordinates."""

    DEFORMED = 1
    """Displays the original ROI masks warped to the shared cross-session coordinate space via multi-day registration
    deformation fields."""

    TEMPLATE = 2
    """Displays the consensus template ROI masks derived from cross-session clustering, defined in the shared
    coordinate space."""


class CoordinateSpace(IntEnum):
    """Selects the coordinate space for reference images."""

    NATIVE = 0
    """Displays reference images in the original recording session coordinate space."""

    TRANSFORMED = 1
    """Displays reference images warped to align with the cross-session template coordinate space."""


class BackgroundImage(IntEnum):
    """Selects the background image type displayed behind mask overlays."""

    MEAN = 0
    """Mean fluorescence image."""

    ENHANCED_MEAN = 1
    """Contrast-enhanced mean image."""

    MAX_PROJECTION = 2
    """Maximum intensity projection."""

    CORRELATION_MAP = 3
    """Pixel correlation map."""


@dataclass
class TrackingViewerData:
    """Wraps multi-day runtime contexts for the tracking viewer.

    Provides property-based delegation to the current session's multi-day runtime data,
    supporting session navigation, coordinate space switching, mask layer selection, and
    channel toggling. All sessions are loaded at construction time via the ``from_session``
    factory method.

    Notes:
        The ``combined_data`` field on each session's runtime contains the combined multi-plane
        single-day detection data (background images, ROI statistics). Registration and tracking
        data provide transformed images and mask sets for cross-session visualization.
    """

    _contexts: list[MultiDayRuntimeContext] = field(repr=False)
    """All multi-day runtime contexts, one per session."""

    current_session_index: int = 0
    """Zero-based index of the currently displayed session."""

    @property
    def session_count(self) -> int:
        """Returns the number of sessions in the dataset."""
        return len(self._contexts)

    @property
    def session_ids(self) -> tuple[str, ...]:
        """Returns the session identifier strings for all sessions."""
        return tuple(context.runtime.io.session_id for context in self._contexts)

    @property
    def current_session_id(self) -> str:
        """Returns the session identifier for the currently displayed session."""
        return self._current_runtime.io.session_id

    @property
    def _current_runtime(self) -> MultiDayRuntimeData:
        """Returns the runtime data for the current session."""
        return self._contexts[self.current_session_index].runtime

    @property
    def _current_io(self) -> MultiDayIOData:
        """Returns the I/O data for the current session."""
        return self._current_runtime.io

    @property
    def _current_registration(self) -> MultiDayRegistrationData:
        """Returns the registration data for the current session."""
        return self._current_runtime.registration

    @property
    def _current_tracking(self) -> MultiDayTrackingData:
        """Returns the tracking data for the current session."""
        return self._current_runtime.tracking

    @property
    def _current_combined(self) -> CombinedData | None:
        """Returns the combined single-day data for the current session."""
        return self._current_runtime.combined_data

    @property
    def _current_detection(self) -> DetectionData | None:
        """Returns the combined detection data for the current session."""
        if self._current_combined is None:
            return None
        return self._current_combined.detection

    # Frame geometry properties.

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
        """Returns True if the current session has been registered."""
        return self._current_registration.is_registered()

    # Native-space background images (from combined single-day detection).

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

    # Channel 2 native-space images.

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

    # Transformed-space background images (from multi-day registration).

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

    # Channel 2 transformed-space images.

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

    # Mask sets.

    @property
    def original_masks(self) -> list[ROIStatistics] | None:
        """Returns the original ROI masks from single-day extraction."""
        if self._current_combined is None:
            return None
        return self._current_combined.extraction.roi_statistics

    @property
    def original_masks_channel_2(self) -> list[ROIStatistics] | None:
        """Returns the channel 2 original ROI masks from single-day extraction."""
        if self._current_combined is None:
            return None
        return self._current_combined.extraction.roi_statistics_channel_2

    @property
    def deformed_masks(self) -> list[ROIStatistics] | None:
        """Returns the ROI masks warped to the shared coordinate space."""
        return self._current_registration.deformed_cell_masks

    @property
    def deformed_masks_channel_2(self) -> list[ROIStatistics] | None:
        """Returns the channel 2 ROI masks warped to the shared coordinate space."""
        return self._current_registration.deformed_cell_masks_channel_2

    @property
    def template_masks(self) -> list[ROIStatistics] | None:
        """Returns the consensus template masks from cross-session tracking."""
        return self._current_tracking.template_masks

    @property
    def template_masks_channel_2(self) -> list[ROIStatistics] | None:
        """Returns the channel 2 consensus template masks from cross-session tracking."""
        return self._current_tracking.template_masks_channel_2

    def masks_for_layer(
        self,
        layer: MaskLayer,
        *,
        channel_2: bool = False,
    ) -> list[ROIStatistics] | None:
        """Returns the mask set for the specified layer and channel.

        Args:
            layer: The mask layer to retrieve.
            channel_2: If True, returns channel 2 masks instead of channel 1.

        Returns:
            The list of ROIStatistics for the requested layer, or None if unavailable.
        """
        if layer == MaskLayer.ORIGINAL:
            return self.original_masks_channel_2 if channel_2 else self.original_masks
        if layer == MaskLayer.DEFORMED:
            return self.deformed_masks_channel_2 if channel_2 else self.deformed_masks
        if layer == MaskLayer.TEMPLATE:
            return self.template_masks_channel_2 if channel_2 else self.template_masks
        return None

    def background_image(
        self,
        image_type: BackgroundImage,
        *,
        coordinate_space: CoordinateSpace = CoordinateSpace.NATIVE,
        channel_2: bool = False,
    ) -> NDArray[np.float32] | None:
        """Returns the background image for the specified type, space, and channel.

        When the coordinate space is ``TRANSFORMED`` and transformed images are available,
        returns the deformed (registered) image. Falls back to native-space images if
        transformed variants are not available.

        Args:
            image_type: The background image type to retrieve.
            coordinate_space: The coordinate space (native or transformed).
            channel_2: If True, returns channel 2 images instead of channel 1.

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

    def switch_session(self, session_index: int) -> None:
        """Switches the active session.

        Args:
            session_index: Zero-based index of the session to switch to.

        Raises:
            ValueError: If the index is out of range.
        """
        if session_index < 0 or session_index >= len(self._contexts):
            message = (
                f"Unable to switch to session {session_index}. Valid range is "
                f"0 to {len(self._contexts) - 1}."
            )
            raise ValueError(message)

        self.current_session_index = session_index

    @classmethod
    def from_session(cls, root_path: Path) -> TrackingViewerData:
        """Loads all multi-day sessions from a dataset directory.

        Args:
            root_path: Path to any session's root processed data directory. The loader
                searches recursively for ``multiday_runtime_data.yaml`` and reconstructs
                the full dataset hierarchy.

        Returns:
            A fully populated TrackingViewerData instance.
        """
        contexts = MultiDayRuntimeContext.load(root_path=root_path, session_index=-1)
        if not isinstance(contexts, list):
            contexts = [contexts]

        return cls(_contexts=contexts)

    def _native_background_image(self, image_type: BackgroundImage) -> NDArray[np.float32] | None:
        """Returns the native-space channel 1 background image for the given type."""
        if image_type == BackgroundImage.MEAN:
            return self.mean_image
        if image_type == BackgroundImage.ENHANCED_MEAN:
            return self.enhanced_mean_image
        if image_type == BackgroundImage.MAX_PROJECTION:
            return self.maximum_projection
        if image_type == BackgroundImage.CORRELATION_MAP:
            return self.correlation_map
        return None

    def _native_background_image_channel_2(self, image_type: BackgroundImage) -> NDArray[np.float32] | None:
        """Returns the native-space channel 2 background image for the given type."""
        if image_type == BackgroundImage.MEAN:
            return self.mean_image_channel_2
        if image_type == BackgroundImage.ENHANCED_MEAN:
            return self.enhanced_mean_image_channel_2
        if image_type == BackgroundImage.MAX_PROJECTION:
            return self.maximum_projection_channel_2
        if image_type == BackgroundImage.CORRELATION_MAP:
            return self.correlation_map_channel_2
        return None

    def _transformed_background_image(self, image_type: BackgroundImage) -> NDArray[np.float32] | None:
        """Returns the transformed-space channel 1 background image for the given type."""
        if image_type == BackgroundImage.MEAN:
            return self.transformed_mean_image
        if image_type == BackgroundImage.ENHANCED_MEAN:
            return self.transformed_enhanced_mean_image
        if image_type == BackgroundImage.MAX_PROJECTION:
            return self.transformed_maximum_projection
        return None

    def _transformed_background_image_channel_2(self, image_type: BackgroundImage) -> NDArray[np.float32] | None:
        """Returns the transformed-space channel 2 background image for the given type."""
        if image_type == BackgroundImage.MEAN:
            return self.transformed_mean_image_channel_2
        if image_type == BackgroundImage.ENHANCED_MEAN:
            return self.transformed_enhanced_mean_image_channel_2
        if image_type == BackgroundImage.MAX_PROJECTION:
            return self.transformed_maximum_projection_channel_2
        return None
