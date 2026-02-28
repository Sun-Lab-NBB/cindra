"""Provides runtime data classes for the multi-day (across-session) processing pipeline."""

from __future__ import annotations

import copy
from pathlib import Path  # noqa: TC003 - needed at runtime for dacite deserialization
from dataclasses import field, dataclass

import numpy as np
from numpy.typing import NDArray  # noqa: TC002 - needed at runtime for dacite deserialization
from ataraxis_base_utilities import ensure_directory_exists
from ataraxis_data_structures import YamlConfig

from .version import version, python_version
from .single_day_data import ROIMask, CombinedData, ExtractionData, is_memory_mapped


@dataclass
class MultiDayIOData:
    """Stores the Input / Output runtime data for all stages of the multi-day processing pipeline."""

    session_id: str = ""
    """The unique identifier for this session, derived from the distinguishing component of the session directory
    path. This ID is used to name output subdirectories and identify the session in logs."""

    data_path: Path | None = None
    """The path to this session's cindra single-day pipeline output directory. This is the resolved cindra root
    that contains combined_metadata.npz and other single-day outputs. Used to reload CombinedData on demand by
    downstream pipeline stages."""

    dataset_name: str = ""
    """The name of the multi-day dataset, used to create the output subdirectory structure."""

    mroi_region_borders: tuple[int, ...] = ()
    """The x-coordinates of MROI region borders, computed from acquisition parameters during initialization. For MROI
    recordings, these borders mark the boundaries between adjacent imaging regions in the combined field of view.
    ROIs near these borders are filtered out during cell selection to avoid tracking ambiguities. This field is empty
    for non-MROI recordings."""

    dataset_output_paths: tuple[Path, ...] = ()
    """The multiday output paths for every session in the dataset, stored in natural-sorted order. Each entry points
    to a session's multiday output directory (e.g., {cindra_directory}/multiday/{dataset_name}/). Storing this tuple in
    every session enables full dataset hierarchy reconstruction from any single session's serialized YAML file."""

    selected_cell_indices: tuple[int, ...] = ()
    """The indices of channel 1 ROIs selected from CombinedData.extraction.roi_statistics for multi-day tracking.
    These indices reference the original single-day ROI list, avoiding duplication of ROI data."""

    selected_cell_indices_channel_2: tuple[int, ...] = ()
    """The indices of channel 2 ROIs selected from CombinedData.extraction.roi_statistics_channel_2 for multi-day
    tracking. Empty if channel 2 data is not available or no channel 2 ROIs were selected."""


@dataclass
class MultiDayRegistrationData:
    """Stores runtime data from the registration stage."""

    # Deformation fields.
    deform_field_y: NDArray[np.float32] | None = None
    """The Y-dimension displacement field computed by DiffeomorphicDemonsRegistration. Combined with deform_field_x,
    these fields can be used to construct a Deformation instance for warping images."""

    deform_field_x: NDArray[np.float32] | None = None
    """The X-dimension displacement field computed by DiffeomorphicDemonsRegistration. Combined with deform_field_y,
    these fields can be used to construct a Deformation instance for warping images."""

    # Channel 1 transformed images.
    transformed_mean_image: NDArray[np.float32] | None = None
    """The mean image transformed to the shared (deformed) visual space."""

    transformed_enhanced_mean_image: NDArray[np.float32] | None = None
    """The enhanced mean image transformed to the shared (deformed) visual space."""

    transformed_maximum_projection: NDArray[np.float32] | None = None
    """The maximum projection transformed to the shared (deformed) visual space."""

    # Channel 2 transformed images.
    transformed_mean_image_channel_2: NDArray[np.float32] | None = None
    """The channel 2 mean image transformed to the shared (deformed) visual space."""

    transformed_enhanced_mean_image_channel_2: NDArray[np.float32] | None = None
    """The channel 2 enhanced mean image transformed to the shared (deformed) visual space."""

    transformed_maximum_projection_channel_2: NDArray[np.float32] | None = None
    """The channel 2 maximum projection transformed to the shared (deformed) visual space."""

    # Deformed cell masks (intermediate data for cell tracking).
    deformed_cell_masks: list[ROIMask] | None = None
    """The channel 1 cell ROI spatial data after multi-day registration deform offsets have been applied to the
    spatial coordinates of each ROI."""

    deformed_cell_masks_channel_2: list[ROIMask] | None = None
    """The channel 2 cell ROI spatial data after multi-day registration deform offsets have been applied to the
    spatial coordinates of each ROI."""

    def is_registered(self) -> bool:
        """Checks whether registration data exists.

        Returns:
            True if the session has been registered (has deformation fields and deformed cell masks), False otherwise.
        """
        return self.deform_field_y is not None and self.deformed_cell_masks is not None

    def clear(self) -> None:
        """Clears all registration data to prepare for re-registration."""
        self.deform_field_y = None
        self.deform_field_x = None
        self.transformed_mean_image = None
        self.transformed_enhanced_mean_image = None
        self.transformed_maximum_projection = None
        self.transformed_mean_image_channel_2 = None
        self.transformed_enhanced_mean_image_channel_2 = None
        self.transformed_maximum_projection_channel_2 = None
        self.deformed_cell_masks = None
        self.deformed_cell_masks_channel_2 = None

    def prepare_for_saving(self) -> None:
        """Sets array fields to None for YAML serialization."""
        self.deform_field_y = None
        self.deform_field_x = None
        self.transformed_mean_image = None
        self.transformed_enhanced_mean_image = None
        self.transformed_maximum_projection = None
        self.transformed_mean_image_channel_2 = None
        self.transformed_enhanced_mean_image_channel_2 = None
        self.transformed_maximum_projection_channel_2 = None
        self.deformed_cell_masks = None
        self.deformed_cell_masks_channel_2 = None

    def save_arrays(self, output_path: Path) -> None:
        """Saves registration arrays as individual .npy files inside a ``registration_arrays/`` subdirectory.

        Notes:
            Deformed cell masks use the ROIStatistics variable-length serialization pattern and remain as .npz files
            saved directly into ``output_path``.

        Args:
            output_path: The directory where to create the ``registration_arrays/`` subdirectory.
        """
        registration_directory = output_path / "registration_arrays"
        ensure_directory_exists(registration_directory)

        # Saves deformation fields.
        if self.deform_field_y is not None and not is_memory_mapped(self.deform_field_y):
            np.save(registration_directory / "deform_field_y.npy", self.deform_field_y)
        if self.deform_field_x is not None and not is_memory_mapped(self.deform_field_x):
            np.save(registration_directory / "deform_field_x.npy", self.deform_field_x)

        # Saves channel 1 transformed images.
        if self.transformed_mean_image is not None and not is_memory_mapped(self.transformed_mean_image):
            np.save(registration_directory / "transformed_mean_image.npy", self.transformed_mean_image)
        if (
            self.transformed_enhanced_mean_image is not None
            and not is_memory_mapped(self.transformed_enhanced_mean_image)
        ):
            np.save(
                registration_directory / "transformed_enhanced_mean_image.npy", self.transformed_enhanced_mean_image
            )
        if (
            self.transformed_maximum_projection is not None
            and not is_memory_mapped(self.transformed_maximum_projection)
        ):
            np.save(registration_directory / "transformed_maximum_projection.npy", self.transformed_maximum_projection)

        # Saves channel 2 transformed images.
        if (
            self.transformed_mean_image_channel_2 is not None
            and not is_memory_mapped(self.transformed_mean_image_channel_2)
        ):
            np.save(
                registration_directory / "transformed_mean_image_channel_2.npy", self.transformed_mean_image_channel_2
            )
        if (
            self.transformed_enhanced_mean_image_channel_2 is not None
            and not is_memory_mapped(self.transformed_enhanced_mean_image_channel_2)
        ):
            np.save(
                registration_directory / "transformed_enhanced_mean_image_channel_2.npy",
                self.transformed_enhanced_mean_image_channel_2,
            )
        if (
            self.transformed_maximum_projection_channel_2 is not None
            and not is_memory_mapped(self.transformed_maximum_projection_channel_2)
        ):
            np.save(
                registration_directory / "transformed_maximum_projection_channel_2.npy",
                self.transformed_maximum_projection_channel_2,
            )

        # Saves channel 1 deformed cell masks (ROIMask .npz).
        if self.deformed_cell_masks is not None:
            ROIMask.save_list(self.deformed_cell_masks, output_path / "registration_deformed_masks.npz")

        # Saves channel 2 deformed cell masks (ROIMask .npz).
        if self.deformed_cell_masks_channel_2 is not None:
            ROIMask.save_list(
                self.deformed_cell_masks_channel_2, output_path / "registration_deformed_masks_channel_2.npz"
            )

    def load_arrays(self, output_path: Path) -> None:
        """Loads registration arrays from individual .npy files in the ``registration_arrays/`` subdirectory.

        Args:
            output_path: The directory containing the ``registration_arrays/`` subdirectory.
        """
        registration_directory = output_path / "registration_arrays"

        # Loads deformation fields.
        if self.deform_field_y is None:
            path = registration_directory / "deform_field_y.npy"
            if path.exists():
                self.deform_field_y = np.load(path).astype(np.float32)
        if self.deform_field_x is None:
            path = registration_directory / "deform_field_x.npy"
            if path.exists():
                self.deform_field_x = np.load(path).astype(np.float32)

        # Loads channel 1 transformed images.
        if self.transformed_mean_image is None:
            path = registration_directory / "transformed_mean_image.npy"
            if path.exists():
                self.transformed_mean_image = np.load(path).astype(np.float32)
        if self.transformed_enhanced_mean_image is None:
            path = registration_directory / "transformed_enhanced_mean_image.npy"
            if path.exists():
                self.transformed_enhanced_mean_image = np.load(path).astype(np.float32)
        if self.transformed_maximum_projection is None:
            path = registration_directory / "transformed_maximum_projection.npy"
            if path.exists():
                self.transformed_maximum_projection = np.load(path).astype(np.float32)

        # Loads channel 2 transformed images.
        if self.transformed_mean_image_channel_2 is None:
            path = registration_directory / "transformed_mean_image_channel_2.npy"
            if path.exists():
                self.transformed_mean_image_channel_2 = np.load(path).astype(np.float32)
        if self.transformed_enhanced_mean_image_channel_2 is None:
            path = registration_directory / "transformed_enhanced_mean_image_channel_2.npy"
            if path.exists():
                self.transformed_enhanced_mean_image_channel_2 = np.load(path).astype(np.float32)
        if self.transformed_maximum_projection_channel_2 is None:
            path = registration_directory / "transformed_maximum_projection_channel_2.npy"
            if path.exists():
                self.transformed_maximum_projection_channel_2 = np.load(path).astype(np.float32)

        # Loads channel 1 deformed cell masks (ROIMask .npz).
        masks_path = output_path / "registration_deformed_masks.npz"
        if self.deformed_cell_masks is None and masks_path.exists():
            self.deformed_cell_masks = ROIMask.load_list(masks_path)

        # Loads channel 2 deformed cell masks (ROIMask .npz).
        masks_path_channel_2 = output_path / "registration_deformed_masks_channel_2.npz"
        if self.deformed_cell_masks_channel_2 is None and masks_path_channel_2.exists():
            self.deformed_cell_masks_channel_2 = ROIMask.load_list(masks_path_channel_2)

    def memory_map_arrays(self, output_path: Path) -> None:
        """Memory-maps registration arrays from individual .npy files in ``r+`` mode.

        This method mirrors load_arrays() but uses memory mapping for .npy files instead of eager loading.
        ROIMask .npz files are still eagerly loaded because NumPy does not support memory mapping for .npz archives.

        Args:
            output_path: The directory containing the ``registration_arrays/`` subdirectory.
        """
        registration_directory = output_path / "registration_arrays"

        # Memory-maps deformation fields.
        if self.deform_field_y is None:
            path = registration_directory / "deform_field_y.npy"
            if path.exists():
                self.deform_field_y = np.load(path, mmap_mode="r+")
        if self.deform_field_x is None:
            path = registration_directory / "deform_field_x.npy"
            if path.exists():
                self.deform_field_x = np.load(path, mmap_mode="r+")

        # Memory-maps channel 1 transformed images.
        if self.transformed_mean_image is None:
            path = registration_directory / "transformed_mean_image.npy"
            if path.exists():
                self.transformed_mean_image = np.load(path, mmap_mode="r+")
        if self.transformed_enhanced_mean_image is None:
            path = registration_directory / "transformed_enhanced_mean_image.npy"
            if path.exists():
                self.transformed_enhanced_mean_image = np.load(path, mmap_mode="r+")
        if self.transformed_maximum_projection is None:
            path = registration_directory / "transformed_maximum_projection.npy"
            if path.exists():
                self.transformed_maximum_projection = np.load(path, mmap_mode="r+")

        # Memory-maps channel 2 transformed images.
        if self.transformed_mean_image_channel_2 is None:
            path = registration_directory / "transformed_mean_image_channel_2.npy"
            if path.exists():
                self.transformed_mean_image_channel_2 = np.load(path, mmap_mode="r+")
        if self.transformed_enhanced_mean_image_channel_2 is None:
            path = registration_directory / "transformed_enhanced_mean_image_channel_2.npy"
            if path.exists():
                self.transformed_enhanced_mean_image_channel_2 = np.load(path, mmap_mode="r+")
        if self.transformed_maximum_projection_channel_2 is None:
            path = registration_directory / "transformed_maximum_projection_channel_2.npy"
            if path.exists():
                self.transformed_maximum_projection_channel_2 = np.load(path, mmap_mode="r+")

        # Eagerly loads channel 1 deformed cell masks (ROIMask .npz; cannot be memory-mapped).
        masks_path = output_path / "registration_deformed_masks.npz"
        if self.deformed_cell_masks is None and masks_path.exists():
            self.deformed_cell_masks = ROIMask.load_list(masks_path)

        # Eagerly loads channel 2 deformed cell masks (ROIMask .npz; cannot be memory-mapped).
        masks_path_channel_2 = output_path / "registration_deformed_masks_channel_2.npz"
        if self.deformed_cell_masks_channel_2 is None and masks_path_channel_2.exists():
            self.deformed_cell_masks_channel_2 = ROIMask.load_list(masks_path_channel_2)


@dataclass
class MultiDayTimingData:
    """Stores pipeline timing and version data.

    Notes:
        All time durations are stored as integers representing seconds. Discovery phase timing (registration, tracking,
        backward transform) is stored redundantly in each session for simplicity. Extraction phase timing is
        session-specific.
    """

    # Discovery phase timing (stored redundantly per session).
    registration_time: int = 0
    """The across-session diffeomorphic demons registration time in seconds."""

    tracking_time: int = 0
    """The across-session cell tracking time in seconds."""

    backward_transform_time: int = 0
    """The backward across-session cell mask transformation time in seconds."""

    total_discovery_time: int = 0
    """The total discovery phase time in seconds."""

    # Extraction phase timing (session-specific).
    extraction_time: int = 0
    """The fluorescence extraction time for this session in seconds."""

    deconvolution_time: int = 0
    """The spike deconvolution time for this session in seconds."""

    total_extraction_time: int = 0
    """The total extraction phase time for this session in seconds."""

    # Version and timestamp tracking.
    date_processed: str = ""
    """The timestamp when this session's processing completed."""

    python_version: str = python_version
    """The Python interpreter version used for processing this session."""

    cindra_version: str = version
    """The cindra library version used for processing this session."""


@dataclass
class MultiDayTrackingData:
    """Stores template masks from cross-session cell tracking.

    Notes:
        Template masks represent consensus cell ROIs that can be reliably identified across multiple sessions. They are
        generated by clustering deformed cell masks in the shared visual space and extracting pixels that consistently
        appear across sessions.
    """

    template_masks: list[ROIMask] | None = None
    """The template cell masks in shared visual space coordinates. Each ROIMask represents a cell that can be
    tracked across sessions, with pixel coordinates and weights derived from the clustering consensus."""

    template_masks_channel_2: list[ROIMask] | None = None
    """The channel 2 template cell masks in shared visual space coordinates. Only present when tracking channel 2
    cells independently in dual-channel recordings."""

    template_diameter: int = 0
    """The estimated cell diameter in pixels for channel 1 template masks, derived from the median pixel count of
    the generated templates. A value of 0 indicates that no templates have been computed yet."""

    template_diameter_channel_2: int = 0
    """The estimated cell diameter in pixels for channel 2 template masks, derived from the median pixel count of
    the generated templates. A value of 0 indicates that no templates have been computed yet."""

    def prepare_for_saving(self) -> None:
        """Sets all list fields to None for YAML serialization."""
        self.template_masks = None
        self.template_masks_channel_2 = None

    def save_arrays(self, output_path: Path) -> None:
        """Saves template mask arrays to .npz files.

        Args:
            output_path: The directory where to save the tracking data files.
        """
        if self.template_masks is not None:
            ROIMask.save_list(self.template_masks, output_path / "tracking_template_masks.npz")

        if self.template_masks_channel_2 is not None:
            ROIMask.save_list(
                self.template_masks_channel_2, output_path / "tracking_template_masks_channel_2.npz"
            )

    def load_arrays(self, output_path: Path) -> None:
        """Loads template mask arrays from .npz files into this instance.

        Args:
            output_path: The directory containing the tracking data files.
        """
        masks_path = output_path / "tracking_template_masks.npz"
        if self.template_masks is None and masks_path.exists():
            self.template_masks = ROIMask.load_list(masks_path)

        masks_path_channel_2 = output_path / "tracking_template_masks_channel_2.npz"
        if self.template_masks_channel_2 is None and masks_path_channel_2.exists():
            self.template_masks_channel_2 = ROIMask.load_list(masks_path_channel_2)

    def memory_map_arrays(self, output_path: Path) -> None:
        """Loads template mask arrays from .npz files into this instance.

        This method is identical to load_arrays() because template masks are stored as .npz archives, which do not
        support memory mapping. It exists for API consistency with sibling dataclasses.

        Args:
            output_path: The directory containing the tracking data files.
        """
        self.load_arrays(output_path)


@dataclass
class MultiDayRuntimeData(YamlConfig):
    """Aggregates all runtime data for a single session."""

    output_path: Path | None = None
    """The path to the directory where runtime data and array files are stored."""

    io: MultiDayIOData = field(default_factory=MultiDayIOData)
    """The per-session I/O data including session ID, session directory, and dataset name."""

    registration: MultiDayRegistrationData = field(default_factory=MultiDayRegistrationData)
    """The runtime data from the registration stage (deformation fields, transformed images, deformed masks)."""

    tracking: MultiDayTrackingData = field(default_factory=MultiDayTrackingData)
    """The runtime data from the cross-session cell tracking stage (template masks in shared visual space)."""

    extraction: ExtractionData = field(default_factory=ExtractionData)
    """The runtime data from the extraction stage. After backward transformation, tracked cell masks are stored as
    ROIStatistics in roi_statistics. Extraction then populates fluorescence traces and classification fields."""

    timing: MultiDayTimingData = field(default_factory=MultiDayTimingData)
    """The timing information for both discovery and extraction phases."""

    combined_data: CombinedData | None = None
    """The combined single-day processing data for this session, loaded from the session directory. This field is not
    serialized to YAML and is loaded on-demand from the single-day pipeline outputs."""

    def load_arrays(self) -> None:
        """Eagerly loads all multi-day NumPy arrays from disk into memory.

        This is a convenience method that eagerly loads registration, tracking, and extraction arrays. CombinedData
        (single-day data) is NOT loaded by this method and must be loaded separately by the caller. Use the individual
        ``load_arrays()`` / ``memory_map_arrays()`` methods on each child dataclass for fine-grained control over
        which arrays are loaded and how.
        """
        if self.output_path is not None:
            self.registration.load_arrays(self.output_path)
            self.tracking.load_arrays(self.output_path)
            self.extraction.load_arrays(self.output_path)

    def memory_map_arrays(self) -> None:
        """Memory-maps all multi-day NumPy arrays from disk in ``r+`` mode.

        This is a convenience method that memory-maps registration, tracking, and extraction arrays. CombinedData
        (single-day data) is NOT loaded by this method and must be loaded separately by the caller. Use the individual
        ``load_arrays()`` / ``memory_map_arrays()`` methods on each child dataclass for fine-grained control over
        which arrays are loaded and how.
        """
        if self.output_path is not None:
            self.registration.memory_map_arrays(self.output_path)
            self.tracking.memory_map_arrays(self.output_path)
            self.extraction.memory_map_arrays(self.output_path)

    def save(self, output_path: Path) -> None:
        """Saves the runtime data to a YAML file and arrays to .npz/.npy files.

        Notes:
            The combined_data field is NOT saved since it references immutable single-day outputs. It must be loaded
            separately by the caller after deserialization.

        Args:
            output_path: The directory where to save the multiday_runtime_data.yaml file and array files.
        """
        ensure_directory_exists(output_path)
        self.output_path = output_path

        # Saves arrays from each child dataclass.
        self.registration.save_arrays(output_path)
        self.tracking.save_arrays(output_path)
        self.extraction.save_arrays(output_path)

        # Creates a shallow copy for YAML serialization. Child dataclasses are shallow-copied individually so that
        # prepare_for_saving() nulls array fields on the copies without affecting the originals in memory.
        yaml_copy = copy.copy(self)
        yaml_copy.registration = copy.copy(self.registration)
        yaml_copy.tracking = copy.copy(self.tracking)
        yaml_copy.extraction = copy.copy(self.extraction)
        yaml_copy.io = copy.copy(self.io)
        yaml_copy.timing = copy.copy(self.timing)

        # Nulls array fields in child dataclasses for YAML serialization.
        yaml_copy.registration.prepare_for_saving()
        yaml_copy.tracking.prepare_for_saving()
        yaml_copy.extraction.prepare_for_saving()

        # Excludes combined_data from YAML (it references immutable single-day data and is reloaded on load).
        yaml_copy.combined_data = None

        # Saves the YAML file.
        file_path = output_path / "multiday_runtime_data.yaml"
        yaml_copy.to_yaml(file_path=file_path)

    @classmethod
    def load(cls, output_path: Path) -> MultiDayRuntimeData:
        """Deserializes runtime data from a YAML file without loading any NumPy arrays or CombinedData.

        After calling this method, multi-day arrays can be loaded using the ``load_arrays()`` or
        ``memory_map_arrays()`` convenience methods, or individually per-child dataclass. CombinedData must be loaded
        separately by the caller (e.g., ``runtime.combined_data = CombinedData.load(...)``).

        Args:
            output_path: The directory containing the multiday_runtime_data.yaml file.

        Returns:
            A MultiDayRuntimeData instance with all scalar fields deserialized. NumPy array fields and combined_data
            remain None until explicitly loaded.
        """
        file_path = output_path / "multiday_runtime_data.yaml"
        return cls.from_yaml(file_path=file_path)
