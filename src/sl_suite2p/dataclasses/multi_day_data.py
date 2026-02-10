"""Provides runtime data classes for the multi-day (across-session) processing pipeline."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING
from pathlib import Path
from dataclasses import field, dataclass

import numpy as np
from ataraxis_base_utilities import console, ensure_directory_exists
from ataraxis_data_structures import YamlConfig

from .version import version, python_version
from .single_day_data import CombinedData, ROIStatistics, ExtractionData
from .single_day_configuration import AcquisitionParameters

if TYPE_CHECKING:
    from numpy.typing import NDArray


def find_suite2p_directory(session_directory: Path) -> Path:
    """Discovers the suite2p output directory within a session directory tree.

    Searches recursively for the combined_metadata.npz file created by the single-day pipeline's combination step.
    Unlike the single-day pipeline, multi-day session paths are not pre-sanitized, so the suite2p directory may be
    nested at an arbitrary depth below the session root (e.g., under a processed_data/mesoscope_data/ subdirectory).

    Args:
        session_directory: The path to the session's root directory.

    Returns:
        The path to the suite2p output directory that contains the combined_metadata.npz file.

    Raises:
        FileNotFoundError: If no combined_metadata.npz file is found under the session directory.
        RuntimeError: If multiple combined_metadata.npz files are found under the session directory.
    """
    matches = list(session_directory.rglob("combined_metadata.npz"))

    if len(matches) == 0:
        message = (
            f"Unable to locate suite2p output for session {session_directory}. No combined_metadata.npz file was "
            f"found anywhere in the directory tree. Ensure the single-day pipeline has completed successfully for "
            f"this recording session before running multi-day processing."
        )
        console.error(message=message, error=FileNotFoundError)

    if len(matches) > 1:
        message = (
            f"Unable to locate suite2p output for session {session_directory}. Found {len(matches)} "
            f"combined_metadata.npz files, but expected exactly one unique match."
        )
        console.error(message=message, error=RuntimeError)

    # The combined_metadata.npz file is saved at the suite2p root level by CombinedData.save().
    return matches[0].parent


@dataclass
class MultiDayIOData:
    """Stores the Input / Output runtime data for all stages of the multi-day processing pipeline."""

    session_id: str = ""
    """The unique identifier for this session, derived from the distinguishing component of the session directory
    path. This ID is used to name output subdirectories and identify the session in logs."""

    session_directory: Path | None = None
    """The path to this session's root data directory. This is the raw session path provided by the user, which may
    contain the suite2p output at an arbitrary nesting depth."""

    suite2p_directory: Path | None = None
    """The path to this session's suite2p single-day pipeline output directory, discovered from the session directory
    by searching for the combined_metadata.npz file. This field is populated during initialization and cached for
    subsequent use."""

    dataset_name: str = ""
    """The name of the multi-day dataset, used to create the output subdirectory structure."""

    mroi_region_borders: list[int] = field(default_factory=list)
    """The x-coordinates of MROI region borders, computed from acquisition parameters during initialization. For MROI
    recordings, these borders mark the boundaries between adjacent imaging regions in the combined field of view.
    ROIs near these borders are filtered out during cell selection to avoid tracking ambiguities. This field is empty
    for non-MROI recordings."""

    def __post_init__(self) -> None:
        """Converts string paths to Path objects after YAML loading."""
        if isinstance(self.session_directory, str):
            self.session_directory = Path(self.session_directory) if self.session_directory else None
        if isinstance(self.suite2p_directory, str):
            self.suite2p_directory = Path(self.suite2p_directory) if self.suite2p_directory else None

    def prepare_for_saving(self) -> None:
        """Converts Path fields to strings for YAML serialization."""
        if self.session_directory is not None:
            self.session_directory = str(self.session_directory)  # type: ignore[assignment]
        if self.suite2p_directory is not None:
            self.suite2p_directory = str(self.suite2p_directory)  # type: ignore[assignment]


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
class MultiDayRuntimeData(YamlConfig):
    """Aggregates all runtime data for a single session."""

    output_path: Path | None = None
    """The path to the directory where runtime data and array files are stored."""

    io: MultiDayIOData = field(default_factory=MultiDayIOData)
    """The per-session I/O data including session ID, session directory, and dataset name."""

    registration: MultiDayRegistrationData = field(default_factory=MultiDayRegistrationData)
    """The runtime data from the registration stage (deformation, transformed images, deformed masks)."""

    extraction: ExtractionData = field(default_factory=ExtractionData)
    """The runtime data from the tracking and extraction stages. After backward transformation, tracked cell masks are
    stored as ROIStatistics in roi_statistics. Extraction then populates fluorescence traces and classification 
    fields."""

    timing: MultiDayTimingData = field(default_factory=MultiDayTimingData)
    """The timing information for both discovery and extraction phases."""

    combined_data: CombinedData | None = None
    """The combined single-day processing data for this session, loaded from the session directory. This field is not
    serialized to YAML and is loaded on-demand from the single-day pipeline outputs."""

    def __post_init__(self) -> None:
        """Loads arrays from files and combined data from the session directory."""
        # Converts output_path to Path if it was loaded as a string from YAML.
        if self.output_path is not None and isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)

        # Loads arrays from each child dataclass if output_path is set.
        if self.output_path is not None:
            self.registration.load_arrays(self.output_path)
            self.extraction.load_arrays(self.output_path)

        # Loads combined data from the session directory if available.
        if self.io.session_directory is not None and self.combined_data is None:
            self._load_combined_data()

    def _load_combined_data(self) -> None:
        """Loads CombinedData from the session's single-day pipeline output directory.

        Discovers the suite2p directory by recursively searching for the combined_metadata.npz file if it has not
        already been resolved and cached in the I/O data.

        Raises:
            FileNotFoundError: If no combined_metadata.npz file is found under the session directory.
            RuntimeError: If multiple combined_metadata.npz files are found under the session directory.
        """
        session_directory = self.io.session_directory
        if session_directory is None:
            return

        # Discovers the suite2p directory if it has not already been resolved.
        if self.io.suite2p_directory is None:
            self.io.suite2p_directory = find_suite2p_directory(session_directory=session_directory)

        suite2p_directory = self.io.suite2p_directory
        self.combined_data = CombinedData.load(root_path=suite2p_directory)

        # Loads acquisition parameters and computes MROI region borders if applicable.
        acquisition_path = suite2p_directory / "acquisition_parameters.yaml"
        if acquisition_path.exists():
            acquisition = AcquisitionParameters.from_yaml(file_path=acquisition_path)
            if acquisition.is_mroi:
                # Computes region borders from ROI x-coordinates. The borders are the x-coordinates where one region
                # ends and another begins, which are all x-coordinates except the minimum (leftmost region).
                sorted_x = sorted(acquisition.roi_x_coordinates)
                self.io.mroi_region_borders = sorted_x[1:]  # Excludes the leftmost region's starting position.

    def save(self, output_path: Path) -> None:
        """Saves the runtime data to a YAML file and arrays to .npz/.npy files.

        Args:
            output_path: The directory where to save the multiday_runtime_data.yaml file and array files.
        """
        ensure_directory_exists(output_path)
        self.output_path = output_path

        # Saves arrays from each child dataclass.
        self.registration.save_arrays(output_path)
        self.extraction.save_arrays(output_path)

        # Creates a deep copy for YAML serialization.
        yaml_copy = copy.deepcopy(self)

        # Prepares each child dataclass for YAML serialization.
        yaml_copy.output_path = str(output_path)  # type: ignore[assignment]
        yaml_copy.io.prepare_for_saving()
        yaml_copy.registration.prepare_for_saving()
        yaml_copy.extraction.prepare_for_saving()
        yaml_copy.combined_data = None  # Excludes from serialization; loaded from single-day outputs.

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
