"""Provides runtime data classes for the multi-recording (across-recording) processing pipeline."""

from __future__ import annotations

import copy
from pathlib import Path  # noqa: TC003 - needed at runtime for dacite deserialization
from dataclasses import field, dataclass

import numpy as np
from numpy.typing import NDArray  # noqa: TC002 - needed at runtime for dacite deserialization
from ataraxis_base_utilities import ensure_directory_exists
from ataraxis_data_structures import YamlConfig

from .version import version, python_version
from .single_recording_data import ROIMask, CombinedData, ExtractionData, is_memory_mapped


@dataclass
class MultiRecordingIOData:
    """Stores the Input / Output runtime data for all stages of the multi-recording processing pipeline."""

    recording_id: str = ""
    """The unique identifier for this recording, derived from the distinguishing component of the recording directory
    path. This ID is used to name output subdirectories and identify the recording in logs."""

    data_path: Path | None = None
    """The path to this recording's cindra single-recording pipeline output directory. This is the resolved cindra root
    that contains combined_metadata.npz and other single-recording outputs. Used to reload CombinedData on demand by
    downstream pipeline stages."""

    dataset_name: str = ""
    """The name of the multi-recording dataset, used to create the output subdirectory structure."""

    mroi_region_borders: tuple[int, ...] = ()
    """The x-coordinates of MROI region borders, computed from acquisition parameters during initialization. For MROI
    recordings, these borders mark the boundaries between adjacent imaging regions in the combined field of view.
    ROIs near these borders are filtered out during ROI selection to avoid tracking ambiguities. This field is empty
    for non-MROI recordings."""

    dataset_output_paths: tuple[Path, ...] = ()
    """The multi_recording output paths for every recording in the dataset, stored in natural-sorted order. Each
    entry points to a recording's multi_recording output directory
    (e.g., {cindra_directory}/multi_recording/{dataset_name}/). Storing this tuple in every recording enables full
    dataset hierarchy reconstruction from any single recording's serialized YAML file."""

    selected_roi_indices: tuple[int, ...] = ()
    """The indices of channel 1 ROIs selected from CombinedData.extraction.roi_statistics for multi-recording tracking.
    These indices reference the original single-recording ROI list, avoiding duplication of ROI data."""

    selected_roi_indices_channel_2: tuple[int, ...] = ()
    """The indices of channel 2 ROIs selected from CombinedData.extraction.roi_statistics_channel_2 for multi-recording
    tracking. Empty if channel 2 data is not available or no channel 2 ROIs were selected."""


@dataclass
class MultiRecordingRegistrationData:
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

    # Deformed ROI masks (intermediate data for ROI tracking).
    deformed_roi_masks: list[ROIMask] | None = None
    """The channel 1 ROI spatial data after multi-recording registration deform offsets have been applied to the
    spatial coordinates of each ROI."""

    deformed_roi_masks_channel_2: list[ROIMask] | None = None
    """The channel 2 ROI spatial data after multi-recording registration deform offsets have been applied to the
    spatial coordinates of each ROI."""

    def is_registered(self, output_path: Path | None = None) -> bool:
        """Checks whether registration data exists in memory or on disk.

        Args:
            output_path: The directory containing the ``registration_arrays/`` subdirectory. When provided and arrays
                are not loaded in memory, checks for deformation field files on disk. The calling context is responsible
                for resolving the correct path.

        Returns:
            True if deformation fields are loaded in memory or exist on disk at the given output path, False otherwise.
        """
        if self.deform_field_y is not None and self.deformed_roi_masks is not None:
            return True
        if output_path is not None:
            return (output_path / "registration_arrays" / "deform_field_y.npy").exists()
        return False

    def clear(self) -> None:
        """Clears all registration data to prepare for re-registration."""
        self.release_arrays()

    def prepare_for_saving(self) -> None:
        """Sets array fields to None for YAML serialization."""
        self.release_arrays()

    def release_arrays(self) -> None:
        """Releases all array fields to free memory.

        Use ``memory_map_arrays()`` or ``load_arrays()`` to re-acquire the data on demand.
        """
        self.deform_field_y = None
        self.deform_field_x = None
        self.transformed_mean_image = None
        self.transformed_enhanced_mean_image = None
        self.transformed_maximum_projection = None
        self.transformed_mean_image_channel_2 = None
        self.transformed_enhanced_mean_image_channel_2 = None
        self.transformed_maximum_projection_channel_2 = None
        self.deformed_roi_masks = None
        self.deformed_roi_masks_channel_2 = None

    def save_arrays(self, output_path: Path) -> None:
        """Saves registration arrays as individual .npy files inside a ``registration_arrays/`` subdirectory.

        Notes:
            Deformed ROI masks use the ROIStatistics variable-length serialization pattern and remain as .npz files
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
        if self.transformed_enhanced_mean_image is not None and not is_memory_mapped(
            self.transformed_enhanced_mean_image
        ):
            np.save(
                registration_directory / "transformed_enhanced_mean_image.npy", self.transformed_enhanced_mean_image
            )
        if self.transformed_maximum_projection is not None and not is_memory_mapped(
            self.transformed_maximum_projection
        ):
            np.save(registration_directory / "transformed_maximum_projection.npy", self.transformed_maximum_projection)

        # Saves channel 2 transformed images.
        if self.transformed_mean_image_channel_2 is not None and not is_memory_mapped(
            self.transformed_mean_image_channel_2
        ):
            np.save(
                registration_directory / "transformed_mean_image_channel_2.npy", self.transformed_mean_image_channel_2
            )
        if self.transformed_enhanced_mean_image_channel_2 is not None and not is_memory_mapped(
            self.transformed_enhanced_mean_image_channel_2
        ):
            np.save(
                registration_directory / "transformed_enhanced_mean_image_channel_2.npy",
                self.transformed_enhanced_mean_image_channel_2,
            )
        if self.transformed_maximum_projection_channel_2 is not None and not is_memory_mapped(
            self.transformed_maximum_projection_channel_2
        ):
            np.save(
                registration_directory / "transformed_maximum_projection_channel_2.npy",
                self.transformed_maximum_projection_channel_2,
            )

        # Saves channel 1 deformed ROI masks (ROIMask .npz).
        if self.deformed_roi_masks is not None:
            ROIMask.save_list(self.deformed_roi_masks, output_path / "registration_deformed_masks.npz")

        # Saves channel 2 deformed ROI masks (ROIMask .npz).
        if self.deformed_roi_masks_channel_2 is not None:
            ROIMask.save_list(
                self.deformed_roi_masks_channel_2, output_path / "registration_deformed_masks_channel_2.npz"
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

        # Loads channel 1 deformed ROI masks (ROIMask .npz).
        masks_path = output_path / "registration_deformed_masks.npz"
        if self.deformed_roi_masks is None and masks_path.exists():
            self.deformed_roi_masks = ROIMask.load_list(masks_path)

        # Loads channel 2 deformed ROI masks (ROIMask .npz).
        masks_path_channel_2 = output_path / "registration_deformed_masks_channel_2.npz"
        if self.deformed_roi_masks_channel_2 is None and masks_path_channel_2.exists():
            self.deformed_roi_masks_channel_2 = ROIMask.load_list(masks_path_channel_2)

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

        # Eagerly loads channel 1 deformed ROI masks (ROIMask .npz; cannot be memory-mapped).
        masks_path = output_path / "registration_deformed_masks.npz"
        if self.deformed_roi_masks is None and masks_path.exists():
            self.deformed_roi_masks = ROIMask.load_list(masks_path)

        # Eagerly loads channel 2 deformed ROI masks (ROIMask .npz; cannot be memory-mapped).
        masks_path_channel_2 = output_path / "registration_deformed_masks_channel_2.npz"
        if self.deformed_roi_masks_channel_2 is None and masks_path_channel_2.exists():
            self.deformed_roi_masks_channel_2 = ROIMask.load_list(masks_path_channel_2)


@dataclass
class MultiRecordingTimingData:
    """Stores pipeline timing and version data.

    Notes:
        All time durations are stored as integers representing seconds. Discovery phase timing (registration, tracking,
        backward transform) is stored redundantly in each recording for simplicity. Extraction phase timing is
        recording-specific.
    """

    # Discovery phase timing (stored redundantly per recording).
    registration_time: int = 0
    """The across-recording diffeomorphic demons registration time in seconds."""

    tracking_time: int = 0
    """The across-recording ROI tracking time in seconds."""

    backward_transform_time: int = 0
    """The backward across-recording ROI mask transformation time in seconds."""

    total_discovery_time: int = 0
    """The total discovery phase time in seconds."""

    # Extraction phase timing (recording-specific).
    extraction_time: int = 0
    """The fluorescence extraction time for this recording in seconds."""

    deconvolution_time: int = 0
    """The spike deconvolution time for this recording in seconds."""

    total_extraction_time: int = 0
    """The total extraction phase time for this recording in seconds."""

    # Version and timestamp tracking.
    date_processed: str = ""
    """The timestamp when this recording's processing completed."""

    python_version: str = python_version
    """The Python interpreter version used for processing this recording."""

    cindra_version: str = version
    """The cindra library version used for processing this recording."""


@dataclass
class MultiRecordingTrackingData:
    """Stores template masks from cross-recording ROI tracking.

    Notes:
        Template masks represent consensus ROIs that can be reliably identified across multiple recordings. They are
        generated by clustering deformed ROI masks in the shared visual space and extracting pixels that consistently
        appear across recordings.
    """

    template_masks: list[ROIMask] | None = None
    """The template ROI masks in shared visual space coordinates. Each ROIMask represents an ROI that can be
    tracked across recordings, with pixel coordinates and weights derived from the clustering consensus."""

    template_masks_channel_2: list[ROIMask] | None = None
    """The channel 2 template ROI masks in shared visual space coordinates. Only present when tracking channel 2
    ROIs independently in dual-channel recordings."""

    template_diameter: int = 0
    """The estimated ROI diameter in pixels for channel 1 template masks, derived from the median pixel count of
    the generated templates. A value of 0 indicates that no templates have been computed yet."""

    template_diameter_channel_2: int = 0
    """The estimated ROI diameter in pixels for channel 2 template masks, derived from the median pixel count of
    the generated templates. A value of 0 indicates that no templates have been computed yet."""

    def prepare_for_saving(self) -> None:
        """Sets all list fields to None for YAML serialization."""
        self.release_arrays()

    def release_arrays(self) -> None:
        """Releases all list fields to free memory.

        Use ``load_arrays()`` to re-acquire the data on demand.
        """
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
            ROIMask.save_list(self.template_masks_channel_2, output_path / "tracking_template_masks_channel_2.npz")

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
class MultiRecordingRuntimeData(YamlConfig):
    """Aggregates all runtime data for a single recording."""

    output_path: Path | None = None
    """The path to the directory where runtime data and array files are stored."""

    io: MultiRecordingIOData = field(default_factory=MultiRecordingIOData)
    """The per-recording I/O data including recording ID, recording directory, and dataset name."""

    registration: MultiRecordingRegistrationData = field(default_factory=MultiRecordingRegistrationData)
    """The runtime data from the registration stage (deformation fields, transformed images, deformed masks)."""

    tracking: MultiRecordingTrackingData = field(default_factory=MultiRecordingTrackingData)
    """The runtime data from the cross-recording ROI tracking stage (template masks in shared visual space)."""

    extraction: ExtractionData = field(default_factory=ExtractionData)
    """The runtime data from the extraction stage. After backward transformation, tracked ROI masks are stored as
    ROIStatistics in roi_statistics. Extraction then populates fluorescence traces and classification fields."""

    timing: MultiRecordingTimingData = field(default_factory=MultiRecordingTimingData)
    """The timing information for both discovery and extraction phases."""

    combined_data: CombinedData | None = None
    """The combined single-recording processing data for this recording, loaded from the recording directory.
    This field is not serialized to YAML and is loaded on-demand from the single-recording pipeline
    outputs."""

    def release_arrays(self) -> None:
        """Releases all array fields across registration, tracking, extraction, and combined_data to free memory.

        Delegates to the ``release_arrays()`` method on each child dataclass. Also releases combined_data detection
        and extraction arrays if combined_data is loaded.
        """
        self.registration.release_arrays()
        self.tracking.release_arrays()
        self.extraction.release_arrays()
        if self.combined_data is not None:
            self.combined_data.detection.release_arrays()
            self.combined_data.extraction.release_arrays()

    def load_arrays(self) -> None:
        """Eagerly loads all multi-recording NumPy arrays from disk into memory.

        This is a convenience method that eagerly loads registration, tracking, and extraction arrays.
        CombinedData (single-recording data) is NOT loaded by this method and must be loaded separately by
        the caller. Use the individual ``load_arrays()`` / ``memory_map_arrays()`` methods on each child
        dataclass for fine-grained control over which arrays are loaded and how.
        """
        if self.output_path is not None:
            self.registration.load_arrays(self.output_path)
            self.tracking.load_arrays(self.output_path)
            self.extraction.load_arrays(self.output_path)

    def memory_map_arrays(self) -> None:
        """Memory-maps all multi-recording NumPy arrays from disk in ``r+`` mode.

        This is a convenience method that memory-maps registration, tracking, and extraction arrays.
        CombinedData (single-recording data) is NOT loaded by this method and must be loaded separately by
        the caller. Use the individual ``load_arrays()`` / ``memory_map_arrays()`` methods on each child
        dataclass for fine-grained control over which arrays are loaded and how.
        """
        if self.output_path is not None:
            self.registration.memory_map_arrays(self.output_path)
            self.tracking.memory_map_arrays(self.output_path)
            self.extraction.memory_map_arrays(self.output_path)

    def save(self, output_path: Path) -> None:
        """Saves the runtime data to a YAML file and arrays to .npz/.npy files.

        Notes:
            The combined_data field is NOT saved since it references immutable single-recording outputs. It
            must be loaded separately by the caller after deserialization.

        Args:
            output_path: The directory where to save the multi_recording_runtime_data.yaml file and array files.
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

        # Excludes combined_data from YAML (it references immutable single-recording data and is reloaded on load).
        yaml_copy.combined_data = None

        # Saves the YAML file.
        file_path = output_path / "multi_recording_runtime_data.yaml"
        yaml_copy.to_yaml(file_path=file_path)

    @classmethod
    def load(cls, output_path: Path) -> MultiRecordingRuntimeData:
        """Deserializes runtime data from a YAML file without loading any NumPy arrays or CombinedData.

        After calling this method, multi-recording arrays can be loaded using the ``load_arrays()`` or
        ``memory_map_arrays()`` convenience methods, or individually per-child dataclass. CombinedData must be loaded
        separately by the caller (e.g., ``runtime.combined_data = CombinedData.load(...)``).

        Args:
            output_path: The directory containing the multi_recording_runtime_data.yaml file.

        Returns:
            A MultiRecordingRuntimeData instance with all scalar fields deserialized. NumPy array fields
            and combined_data remain None until explicitly loaded.
        """
        file_path = output_path / "multi_recording_runtime_data.yaml"
        return cls.from_yaml(file_path=file_path)
