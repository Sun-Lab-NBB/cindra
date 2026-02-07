"""Provides the classes used to organize and temporarily store processed data during the first step
(registration) of the multi-session suite2p processing pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from dataclasses import field, dataclass

from .version import version, python_version
from ..multiday.utils import create_mask_image

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from numpy.typing import NDArray

    from ..registration import Deformation


@dataclass
class MultiDayTimingData:
    """Stores pipeline timing and version information for the multi-day processing pipeline.

    All time durations are stored as integers representing seconds. Each field corresponds to a discrete step in the
    multi-day pipeline: discover_multiday_cells() handles import through backward_transform, while
    extract_session_traces() handles extraction and deconvolution.
    """

    import_time: int = 0
    """The session data import time in seconds."""

    registration_time: int = 0
    """The across-session diffeomorphic demons registration time in seconds."""

    template_mask_time: int = 0
    """The template mask generation (cross-session cell tracking) time in seconds."""

    backward_transform_time: int = 0
    """The backward mask transformation time in seconds."""

    total_registration_time: int = 0
    """The total discover_multiday_cells() step time in seconds."""

    extraction_time: int = 0
    """The fluorescence extraction time in seconds."""

    deconvolution_time: int = 0
    """The spike deconvolution time in seconds."""

    date_processed: str = ""
    """The timestamp when processing completed."""

    python_version: str = python_version
    """The Python interpreter version used for processing."""

    sl_suite2p_version: str = version
    """The sl-suite2p library version used for processing."""


@dataclass()
class Session:
    """Stores all multi-day suite2p pipeline data for a single session."""

    session_id: str
    """Stores the ID of the session."""

    suite2p_folder: Path
    """Stores the path to the session's root suite2p single-day pipeline output folder."""

    # noinspection PyTypeHints
    reference_images: dict[str, NDArray[np.uint32 | np.float32]]
    """Stores the reference images generated during single-day processing. Valid image query keys are: 'mean',
    'enhanced', and 'max'."""

    cell_masks: tuple[dict[str, Any], ...]
    """For each cell ROI discovered during the single-day suite2p processing, stores a dictionary that contains cell
    mask data."""

    image_size: tuple[int, int]
    """Stores the dimensions of the registered session movie in the order of: height, width."""

    # noinspection PyTypeHints
    transformed_images: dict[str, NDArray[np.uint32 | np.float32]] = field(init=False)
    """Same as 'reference_images', but stores the reference images after applying multi-day registration deform offsets
    to translate them to the (deformed) visual space shared by all processed sessions. This represents the session's
    reference images in the registered (deformed) visual space."""

    deform: Deformation = field(init=False)
    """Stores the Deformation instance computed by DiffeomorphicDemonsRegistration to align this session to the shared
    (registered) visual space. Uses backward mapping to transform coordinates from registered to original space."""

    deformed_cell_masks: tuple[dict[str, Any], ...] = field(init=False)
    """Same as 'cell_masks', but stores cell ROI data after multi-day registration deform offsets have been applied
    to the spatial coordinates of each ROI. This represents all session cell masks in the registered (deformed) visual
    space."""

    shared_cell_masks: tuple[dict[str, Any], ...] = field(init=False)
    """Same as 'deformed_cell_masks' but stores cell ROI data for the 'template' (across-session tracked) cells in the
    shared (deformed / registered) visual space."""

    template_cell_masks: tuple[dict[str, Any], ...] = field(init=False)
    """Same as 'cell_masks', but stores cell ROI data for the 'template' (across-session tracked) cells mapped back to
    the original session's visual space."""

    @property
    def unregistered_masks(self) -> NDArray[Any]:
        """Returns an image that shows all single-day cell masks in the original visual space of the session."""
        return create_mask_image(self.cell_masks, self.image_size, mark_overlap=True)

    @property
    def registered_masks(self) -> NDArray[Any]:
        """Returns an image that shows all single-day cell masks in the multi-day deformed (registered) visual space."""
        return create_mask_image(self.deformed_cell_masks, image_size=self.deform.field_shape, mark_overlap=True)

    @property
    def shared_template_masks(self) -> NDArray[Any]:
        """Returns an image that shows all template (across-day-tracked) cell masks in the shared
        (registered / deformed) visual space.
        """
        return create_mask_image(self.shared_cell_masks, self.image_size, mark_overlap=True)

    @property
    def session_template_masks(self) -> NDArray[Any]:
        """Returns an image that shows all template (across-day-tracked) cell masks in the original session visual
        space.
        """
        return create_mask_image(self.template_cell_masks, self.deform.field_shape, mark_overlap=True)


@dataclass()
class MultiDayData:
    """Stores multiple Session class instances and exposes the API for extracting and adding data generated by the
    multiday suite2p pipeline functions.
    """

    sessions: list[Session]
    """Stores Session class instances for each session that needs to be registered across days."""

    template_cell_masks: tuple[dict[str, Any], ...] = field(init=False)
    """For each cell ROI tracked over sessions as part of multi-day processing, stores a dictionary that contains cell
    mask data."""

    timing: MultiDayTimingData = field(default_factory=MultiDayTimingData)
    """The pipeline timing and version information for the multi-day processing pipeline."""
