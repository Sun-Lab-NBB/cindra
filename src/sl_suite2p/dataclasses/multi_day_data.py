"""Provides runtime data classes for the multi-day (across-session) processing pipeline."""

from __future__ import annotations

import copy
from pathlib import Path
from dataclasses import field, dataclass

import numpy as np
from numpy.typing import NDArray
from ataraxis_base_utilities import console, ensure_directory_exists
from ataraxis_data_structures import YamlConfig

from .version import version, python_version
from .single_day_data import CombinedData, ROIStatistics, ExtractionData


@dataclass
class MultiDayIOData:
    """Stores the Input / Output runtime data for all stages of the multi-day processing pipeline."""

    session_id: str = ""
    """The unique identifier for this session, derived from the distinguishing component of the session directory
    path. This ID is used to name output subdirectories and identify the session in logs."""

    data_path: Path | None = None
    """The path to this session's suite2p single-day pipeline output directory. This is the resolved suite2p root
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
    to a session's multiday output directory (e.g., {suite2p_parent}/multiday/{dataset_name}/). Storing this tuple in
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
    deformed_cell_masks: list[ROIStatistics] | None = None
    """The channel 1 cell ROI data after multi-day registration deform offsets have been applied to the spatial
    coordinates of each ROI."""

    deformed_cell_masks_channel_2: list[ROIStatistics] | None = None
    """The channel 2 cell ROI data after multi-day registration deform offsets have been applied to the spatial
    coordinates of each ROI."""

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
        """Saves registration data arrays to .npz files.

        Args:
            output_path: The directory where to save the registration data files.
        """
        # Saves deformation fields.
        if self.deform_field_y is not None and self.deform_field_x is not None:
            np.savez(
                output_path / "registration_deform.npz",
                allow_pickle=False,
                field_y=self.deform_field_y,
                field_x=self.deform_field_x,
            )

        # Saves transformed images.
        images_dict: dict[str, NDArray[np.float32]] = {}
        if self.transformed_mean_image is not None:
            images_dict["mean_image"] = self.transformed_mean_image
        if self.transformed_enhanced_mean_image is not None:
            images_dict["enhanced_mean_image"] = self.transformed_enhanced_mean_image
        if self.transformed_maximum_projection is not None:
            images_dict["maximum_projection"] = self.transformed_maximum_projection
        if self.transformed_mean_image_channel_2 is not None:
            images_dict["mean_image_channel_2"] = self.transformed_mean_image_channel_2
        if self.transformed_enhanced_mean_image_channel_2 is not None:
            images_dict["enhanced_mean_image_channel_2"] = self.transformed_enhanced_mean_image_channel_2
        if self.transformed_maximum_projection_channel_2 is not None:
            images_dict["maximum_projection_channel_2"] = self.transformed_maximum_projection_channel_2
        if images_dict:
            np.savez(output_path / "registration_transformed_images.npz", allow_pickle=False, **images_dict)

        # Saves channel 1 deformed cell masks.
        if self.deformed_cell_masks is not None:
            ROIStatistics.save_list(self.deformed_cell_masks, output_path / "registration_deformed_masks.npz")

        # Saves channel 2 deformed cell masks.
        if self.deformed_cell_masks_channel_2 is not None:
            ROIStatistics.save_list(
                self.deformed_cell_masks_channel_2, output_path / "registration_deformed_masks_channel_2.npz"
            )

    def load_arrays(self, output_path: Path) -> None:
        """Loads registration data arrays from .npz files into this instance.

        Args:
            output_path: The directory containing the registration data files.
        """
        # Loads deformation fields.
        deform_path = output_path / "registration_deform.npz"
        if self.deform_field_y is None and deform_path.exists():
            data = np.load(deform_path, allow_pickle=False)
            self.deform_field_y = data["field_y"].astype(np.float32)
            self.deform_field_x = data["field_x"].astype(np.float32)

        # Loads transformed images.
        transformed_path = output_path / "registration_transformed_images.npz"
        if transformed_path.exists():
            data = np.load(transformed_path, allow_pickle=False)
            if self.transformed_mean_image is None and "mean_image" in data:
                self.transformed_mean_image = data["mean_image"].astype(np.float32)
            if self.transformed_enhanced_mean_image is None and "enhanced_mean_image" in data:
                self.transformed_enhanced_mean_image = data["enhanced_mean_image"].astype(np.float32)
            if self.transformed_maximum_projection is None and "maximum_projection" in data:
                self.transformed_maximum_projection = data["maximum_projection"].astype(np.float32)
            if self.transformed_mean_image_channel_2 is None and "mean_image_channel_2" in data:
                self.transformed_mean_image_channel_2 = data["mean_image_channel_2"].astype(np.float32)
            if self.transformed_enhanced_mean_image_channel_2 is None and "enhanced_mean_image_channel_2" in data:
                self.transformed_enhanced_mean_image_channel_2 = data["enhanced_mean_image_channel_2"].astype(
                    np.float32
                )
            if self.transformed_maximum_projection_channel_2 is None and "maximum_projection_channel_2" in data:
                self.transformed_maximum_projection_channel_2 = data["maximum_projection_channel_2"].astype(np.float32)

        # Loads channel 1 deformed cell masks.
        masks_path = output_path / "registration_deformed_masks.npz"
        if self.deformed_cell_masks is None and masks_path.exists():
            self.deformed_cell_masks = ROIStatistics.load_list(masks_path)

        # Loads channel 2 deformed cell masks.
        masks_path_channel_2 = output_path / "registration_deformed_masks_channel_2.npz"
        if self.deformed_cell_masks_channel_2 is None and masks_path_channel_2.exists():
            self.deformed_cell_masks_channel_2 = ROIStatistics.load_list(masks_path_channel_2)


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

    sl_suite2p_version: str = version
    """The sl-suite2p library version used for processing this session."""


@dataclass
class MultiDayTrackingData:
    """Stores template masks from cross-session cell tracking.

    Notes:
        Template masks represent consensus cell ROIs that can be reliably identified across multiple sessions. They are
        generated by clustering deformed cell masks in the shared visual space and extracting pixels that consistently
        appear across sessions.
    """

    template_masks: list[ROIStatistics] | None = None
    """The template cell masks in shared visual space coordinates. Each ROIStatistics represents a cell that can be
    tracked across sessions, with pixel coordinates and weights derived from the clustering consensus."""

    template_masks_channel_2: list[ROIStatistics] | None = None
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
            ROIStatistics.save_list(self.template_masks, output_path / "tracking_template_masks.npz")

        if self.template_masks_channel_2 is not None:
            ROIStatistics.save_list(
                self.template_masks_channel_2, output_path / "tracking_template_masks_channel_2.npz"
            )

    def load_arrays(self, output_path: Path) -> None:
        """Loads template mask arrays from .npz files into this instance.

        Args:
            output_path: The directory containing the tracking data files.
        """
        masks_path = output_path / "tracking_template_masks.npz"
        if self.template_masks is None and masks_path.exists():
            self.template_masks = ROIStatistics.load_list(masks_path)

        masks_path_channel_2 = output_path / "tracking_template_masks_channel_2.npz"
        if self.template_masks_channel_2 is None and masks_path_channel_2.exists():
            self.template_masks_channel_2 = ROIStatistics.load_list(masks_path_channel_2)


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

    def __post_init__(self) -> None:
        """Loads arrays from existing output files."""
        # Loads arrays from each child dataclass if output_path is set.
        if self.output_path is not None:
            self.registration.load_arrays(self.output_path)
            self.tracking.load_arrays(self.output_path)
            self.extraction.load_arrays(self.output_path)

        # Loads CombinedData from the single-day data path if not already set. Multi-day functionality requires
        # single-day data to be available, so this raises an error if the data cannot be loaded.
        if self.combined_data is None and self.io.data_path is not None:
            combined_metadata_path = self.io.data_path / "combined_metadata.npz"
            if not combined_metadata_path.exists():
                message = (
                    f"Unable to load multi-day runtime data. The single-day combined_metadata.npz file does not exist "
                    f"at the expected path: {combined_metadata_path}. Multi-day processing requires single-day data to "
                    f"be available. Ensure the single-day pipeline completed successfully and the data has not been "
                    f"moved or deleted."
                )
                console.error(message=message, error=FileNotFoundError)
            self.combined_data = CombinedData.load(root_path=self.io.data_path)

    def save(self, output_path: Path) -> None:
        """Saves the runtime data to a YAML file and arrays to .npz/.npy files.

        Notes:
            The combined_data field is NOT saved since it references immutable single-day outputs. It is reloaded
            from io.data_path during deserialization via __post_init__.

        Args:
            output_path: The directory where to save the multiday_runtime_data.yaml file and array files.
        """
        ensure_directory_exists(output_path)
        self.output_path = output_path

        # Saves arrays from each child dataclass.
        self.registration.save_arrays(output_path)
        self.tracking.save_arrays(output_path)
        self.extraction.save_arrays(output_path)

        # Creates a deep copy for YAML serialization. The deep copy is still needed because array fields must be
        # nulled for YAML serialization while keeping the originals intact in memory.
        yaml_copy = copy.deepcopy(self)

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
        """Loads runtime data from a YAML file and associated array files.

        Args:
            output_path: The directory containing the multiday_runtime_data.yaml file.

        Returns:
            A MultiDayRuntimeData instance with all data loaded, including arrays.
        """
        file_path = output_path / "multiday_runtime_data.yaml"
        return cls.from_yaml(file_path=file_path)
