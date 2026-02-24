"""Provides runtime data classes for the single-day (within-session) processing pipeline."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING
from pathlib import Path  # noqa: TC003 - needed at runtime for dacite deserialization
from dataclasses import field, dataclass

import numpy as np
from numpy.typing import NDArray  # noqa: TC002 - needed at runtime for dacite deserialization
from ataraxis_base_utilities import console, ensure_directory_exists
from ataraxis_data_structures import YamlConfig

from .version import version, python_version

if TYPE_CHECKING:
    from numpy.lib.npyio import NpzFile


def _save_optional_array_field(
    field_name: str,
    arrays: list[NDArray[np.float32] | NDArray[np.int32] | NDArray[np.bool_] | tuple[int, ...] | None],
    save_dictionary: dict[str, NDArray[np.float32] | NDArray[np.int32] | NDArray[np.bool_] | NDArray[np.uint32]],
    dtype: type,
) -> None:
    """Saves an optional variable-length array field to the provided save dictionary.

    Notes:
        This function handles the serialization pattern for optional array fields in dataclasses. It stores two arrays
        in the save dictionary: a counts array with the length of each item's array (0 if None), and a concatenated
        data array containing only the non-None values. This enables pickle-free serialization of variable-length
        arrays.

    Args:
        field_name: The base name for the field. The function stores '{field_name}_counts' and '{field_name}' keys.
        arrays: The list of arrays to save. None values and empty arrays are handled by storing 0 in the counts array.
        save_dictionary: The dictionary to populate with the serialized arrays.
        dtype: The numpy dtype to use when converting arrays.
    """
    has_data = [a is not None and len(a) > 0 for a in arrays]
    if not any(has_data):
        return

    counts = np.array(object=[len(a) if a is not None else 0 for a in arrays], dtype=np.uint32)
    valid_arrays: list[NDArray[np.float32] | NDArray[np.int32] | NDArray[np.bool_]] = [
        np.asarray(a=a, dtype=dtype) for a in arrays if a is not None and len(a) > 0
    ]
    if valid_arrays:
        save_dictionary[f"{field_name}_counts"] = counts
        save_dictionary[field_name] = np.concatenate(valid_arrays)  # type: ignore[assignment]


def _load_optional_array_field(
    field_name: str,
    item_count: int,
    data: NpzFile,
    dtype: type,
) -> list[NDArray[np.float32] | NDArray[np.int32] | NDArray[np.bool_] | None]:
    """Loads an optional variable-length array field from a numpy NpzFile.

    Notes:
        This function reverses the serialization pattern used by _save_optional_array_field. It reads the counts array
        and concatenated data array, then splits the data back into per-item arrays based on the stored counts.

    Args:
        field_name: The base name for the field. The function reads '{field_name}_counts' and '{field_name}' keys.
        item_count: The total number of items expected (determines the length of the returned list).
        data: The NpzFile containing the serialized arrays.
        dtype: The numpy dtype to cast the loaded arrays to.

    Returns:
        A list of arrays with length equal to item_count. Items that had no data (count of 0) are returned as None.
    """
    result: list[NDArray[np.float32] | NDArray[np.int32] | NDArray[np.bool_] | None] = [None] * item_count
    counts_key = f"{field_name}_counts"
    if counts_key not in data:
        return result

    counts = data[counts_key]
    array_data = data[field_name]
    index = 0
    for i, count in enumerate(counts):
        if count > 0:
            result[i] = array_data[index : index + count].astype(dtype=dtype)
            index += count
    return result


@dataclass
class IOData:
    """Stores the Input / Output runtime data for all stages of the single-day processing pipeline."""

    frame_height: int = 0
    """The height of each frame in pixels (Y dimension of the imaging field of view)."""

    frame_width: int = 0
    """The width of each frame in pixels (X dimension of the imaging field of view)."""

    frame_count: int = 0
    """The total number of frames written to the binary file during binarization."""

    sampling_rate: float = 0.0
    """The per-plane sampling rate in Hertz, derived from the acquisition frame rate divided by the number of
    imaging planes. This value is computed during binarization from the AcquisitionParameters."""

    registered_binary_path: Path | None = None
    """The absolute path to the motion-corrected binary file for the primary imaging channel."""

    registered_binary_path_channel_2: Path | None = None
    """The absolute path to the motion-corrected binary file for the second imaging channel."""

    output_directory: Path | None = None
    """The absolute path to the plane-specific output directory where all results are saved."""

    mroi_y_offset: int | None = None
    """The vertical offset in pixels for positioning this ROI within the full combined field of view. Only used
    for MROI recordings."""

    mroi_x_offset: int | None = None
    """The horizontal offset in pixels for positioning this ROI within the full combined field of view. Only used
    for MROI recordings."""

    mroi_lines: tuple[int, ...] = ()
    """The tuple of scan line indices used for extracting this ROI from raw multi-ROI data. Only used for MROI
    recordings."""

    plane_index: int | None = None
    """The zero-based index identifying this plane's position in a multi-plane volumetric recording."""


@dataclass
class RegistrationData:
    """Stores runtime data from the registration stage."""

    valid_y_range: tuple[int, int] = (0, 0)
    """The valid Y pixel range (start, end) defining the usable recording region after border cropping."""

    valid_x_range: tuple[int, int] = (0, 0)
    """The valid X pixel range (start, end) defining the usable recording region after border cropping."""

    bad_frames: NDArray[np.bool_] | None = None
    """A boolean array with shape (num_frames,) marking frames with excessive motion or poor correlation. Computed
    during registration crop calculation and used during detection for temporal binning."""

    bidirectional_phase_offset: int = 0
    """The phase offset in pixels used to correct bidirectional scanning artifacts."""

    bidirectional_phase_corrected: bool = False
    """Determines whether bidirectional phase correction was applied during registration."""

    normalization_minimum: int = 0
    """The minimum intensity value used for normalizing frames during registration."""

    normalization_maximum: int = 0
    """The maximum intensity value used for normalizing frames during registration."""

    reference_image: NDArray[np.float32] | None = None
    """The template image used as the alignment target for motion correction."""

    rigid_y_offsets: NDArray[np.int32] | None = None
    """The vertical (Y) translation offsets from rigid registration, one value per frame."""

    rigid_x_offsets: NDArray[np.int32] | None = None
    """The horizontal (X) translation offsets from rigid registration, one value per frame."""

    rigid_correlations: NDArray[np.float32] | None = None
    """The phase correlation values from rigid registration, indicating alignment quality per frame."""

    nonrigid_y_offsets: NDArray[np.float32] | None = None
    """The vertical (Y) translation offsets from non-rigid registration, per frame and per block."""

    nonrigid_x_offsets: NDArray[np.float32] | None = None
    """The horizontal (X) translation offsets from non-rigid registration, per frame and per block."""

    nonrigid_correlations: NDArray[np.float32] | None = None
    """The phase correlation values from non-rigid registration, indicating alignment quality per frame and block."""

    principal_component_extreme_images: NDArray[np.float32] | None = None
    """The mean images from frames at extreme ends of each principal component of the registered recording movie, with
    shape (2, num_components, height, width). Index 0 contains low-projection means, index 1 contains high-projection
    means. Used for visualizing registration quality in the GUI."""

    principal_component_projections: NDArray[np.float32] | None = None
    """The projection of each frame onto the principal components of the registered recording movie, with shape
    (num_frames, num_components). Shows how each frame relates to the computed PCs over time."""

    principal_component_shift_metrics: NDArray[np.float32] | None = None
    """The registration shift metrics computed by aligning PC extreme images of the registered recording movie, with
    shape (num_components, 3). Column 0 contains mean rigid shift magnitude, column 1 contains mean nonrigid shift
    magnitude, and column 2 contains maximum nonrigid shift magnitude. Large values indicate poor registration
    quality."""

    def is_registered(self) -> bool:
        """Checks whether registration data exists.

        Returns:
            True if the plane has been registered (has reference image and offsets), False otherwise.
        """
        return (
            self.reference_image is not None and self.rigid_y_offsets is not None and self.rigid_x_offsets is not None
        )

    def clear(self) -> None:
        """Clears all registration data to prepare for re-registration."""
        self.valid_y_range = (0, 0)
        self.valid_x_range = (0, 0)
        self.bad_frames = None
        self.bidirectional_phase_offset = 0
        self.bidirectional_phase_corrected = False
        self.normalization_minimum = 0
        self.normalization_maximum = 0
        self.reference_image = None
        self.rigid_y_offsets = None
        self.rigid_x_offsets = None
        self.rigid_correlations = None
        self.nonrigid_y_offsets = None
        self.nonrigid_x_offsets = None
        self.nonrigid_correlations = None
        self.principal_component_extreme_images = None
        self.principal_component_projections = None
        self.principal_component_shift_metrics = None

    def prepare_for_saving(self) -> None:
        """Sets all array fields to None for YAML serialization."""
        self.bad_frames = None
        self.reference_image = None
        self.rigid_y_offsets = None
        self.rigid_x_offsets = None
        self.rigid_correlations = None
        self.nonrigid_y_offsets = None
        self.nonrigid_x_offsets = None
        self.nonrigid_correlations = None
        self.principal_component_extreme_images = None
        self.principal_component_projections = None
        self.principal_component_shift_metrics = None

    def save_arrays(self, output_path: Path) -> None:
        """Saves all registration arrays to a single .npz file.

        Args:
            output_path: The directory where to save the registration_data.npz file.
        """
        save_dict: dict[str, NDArray[np.float32] | NDArray[np.int32] | NDArray[np.bool_]] = {}

        if self.bad_frames is not None:
            save_dict["bad_frames"] = self.bad_frames
        if self.reference_image is not None:
            save_dict["reference_image"] = self.reference_image
        if self.rigid_y_offsets is not None:
            save_dict["rigid_y_offsets"] = self.rigid_y_offsets
        if self.rigid_x_offsets is not None:
            save_dict["rigid_x_offsets"] = self.rigid_x_offsets
        if self.rigid_correlations is not None:
            save_dict["rigid_correlations"] = self.rigid_correlations
        if self.nonrigid_y_offsets is not None:
            save_dict["nonrigid_y_offsets"] = self.nonrigid_y_offsets
        if self.nonrigid_x_offsets is not None:
            save_dict["nonrigid_x_offsets"] = self.nonrigid_x_offsets
        if self.nonrigid_correlations is not None:
            save_dict["nonrigid_correlations"] = self.nonrigid_correlations
        if self.principal_component_extreme_images is not None:
            save_dict["principal_component_extreme_images"] = self.principal_component_extreme_images
        if self.principal_component_projections is not None:
            save_dict["principal_component_projections"] = self.principal_component_projections
        if self.principal_component_shift_metrics is not None:
            save_dict["principal_component_shift_metrics"] = self.principal_component_shift_metrics

        if save_dict:
            np.savez(output_path / "registration_data.npz", allow_pickle=False, **save_dict)

    def load_arrays(self, output_path: Path) -> None:
        """Loads registration arrays from a .npz file into this instance.

        Args:
            output_path: The directory containing the registration_data.npz file.
        """
        file_path = output_path / "registration_data.npz"
        if not file_path.exists():
            return

        data = np.load(file_path, allow_pickle=False)

        if "bad_frames" in data:
            self.bad_frames = data["bad_frames"].astype(np.bool_)
        if "reference_image" in data:
            self.reference_image = data["reference_image"].astype(np.float32)
        if "rigid_y_offsets" in data:
            self.rigid_y_offsets = data["rigid_y_offsets"].astype(np.int32)
        if "rigid_x_offsets" in data:
            self.rigid_x_offsets = data["rigid_x_offsets"].astype(np.int32)
        if "rigid_correlations" in data:
            self.rigid_correlations = data["rigid_correlations"].astype(np.float32)
        if "nonrigid_y_offsets" in data:
            self.nonrigid_y_offsets = data["nonrigid_y_offsets"].astype(np.float32)
        if "nonrigid_x_offsets" in data:
            self.nonrigid_x_offsets = data["nonrigid_x_offsets"].astype(np.float32)
        if "nonrigid_correlations" in data:
            self.nonrigid_correlations = data["nonrigid_correlations"].astype(np.float32)
        if "principal_component_extreme_images" in data:
            self.principal_component_extreme_images = data["principal_component_extreme_images"].astype(np.float32)
        if "principal_component_projections" in data:
            self.principal_component_projections = data["principal_component_projections"].astype(np.float32)
        if "principal_component_shift_metrics" in data:
            self.principal_component_shift_metrics = data["principal_component_shift_metrics"].astype(np.float32)


@dataclass
class DetectionData:
    """Stores runtime data from the detection stage."""

    cell_diameter: int = 0
    """The estimated cell diameter in pixels, automatically computed from the spatial scale during detection."""

    aspect_ratio: float = 0.0
    """The aspect ratio of detected cells, computed as the ratio of vertical to horizontal diameter."""

    mean_image: NDArray[np.float32] | None = None
    """The temporal mean of all registered frames, providing a static view of the imaging field."""

    enhanced_mean_image: NDArray[np.float32] | None = None
    """The high-pass filtered mean image that enhances cell boundaries for improved detection."""

    maximum_projection: NDArray[np.float32] | None = None
    """The maximum intensity projection across all frames, highlighting active structures."""

    correlation_map: NDArray[np.float32] | None = None
    """The pixel-wise correlation map used to identify regions with correlated activity for cell detection."""

    cell_diameter_channel_2: int = 0
    """The estimated cell diameter for the second imaging channel in pixels. Computed independently because channel 2
    may label a different cell population with different soma sizes."""

    mean_image_channel_2: NDArray[np.float32] | None = None
    """The temporal mean of all registered frames for the second imaging channel."""

    enhanced_mean_image_channel_2: NDArray[np.float32] | None = None
    """The high-pass filtered mean image for the second imaging channel."""

    maximum_projection_channel_2: NDArray[np.float32] | None = None
    """The maximum intensity projection across all frames for the second imaging channel."""

    correlation_map_channel_2: NDArray[np.float32] | None = None
    """The pixel-wise correlation map for the second imaging channel."""

    def prepare_for_saving(self) -> None:
        """Sets all array fields to None for YAML serialization."""
        self.mean_image = None
        self.enhanced_mean_image = None
        self.maximum_projection = None
        self.correlation_map = None
        self.mean_image_channel_2 = None
        self.enhanced_mean_image_channel_2 = None
        self.maximum_projection_channel_2 = None
        self.correlation_map_channel_2 = None

    def save_arrays(self, output_path: Path) -> None:
        """Saves all detection arrays to a single .npz file.

        Args:
            output_path: The directory where to save the detection_data.npz file.
        """
        save_dict: dict[str, NDArray[np.float32]] = {}

        # Channel 1 arrays.
        if self.mean_image is not None:
            save_dict["mean_image"] = self.mean_image
        if self.enhanced_mean_image is not None:
            save_dict["enhanced_mean_image"] = self.enhanced_mean_image
        if self.maximum_projection is not None:
            save_dict["maximum_projection"] = self.maximum_projection
        if self.correlation_map is not None:
            save_dict["correlation_map"] = self.correlation_map

        # Channel 2 arrays.
        if self.mean_image_channel_2 is not None:
            save_dict["mean_image_channel_2"] = self.mean_image_channel_2
        if self.enhanced_mean_image_channel_2 is not None:
            save_dict["enhanced_mean_image_channel_2"] = self.enhanced_mean_image_channel_2
        if self.maximum_projection_channel_2 is not None:
            save_dict["maximum_projection_channel_2"] = self.maximum_projection_channel_2
        if self.correlation_map_channel_2 is not None:
            save_dict["correlation_map_channel_2"] = self.correlation_map_channel_2

        if save_dict:
            np.savez(output_path / "detection_data.npz", allow_pickle=False, **save_dict)

    def load_arrays(self, output_path: Path) -> None:
        """Loads detection arrays from a .npz file into this instance.

        Args:
            output_path: The directory containing the detection_data.npz file.
        """
        file_path = output_path / "detection_data.npz"
        if not file_path.exists():
            return

        data = np.load(file_path, allow_pickle=False)

        # Channel 1 arrays.
        if "mean_image" in data:
            self.mean_image = data["mean_image"].astype(np.float32)
        if "enhanced_mean_image" in data:
            self.enhanced_mean_image = data["enhanced_mean_image"].astype(np.float32)
        if "maximum_projection" in data:
            self.maximum_projection = data["maximum_projection"].astype(np.float32)
        if "correlation_map" in data:
            self.correlation_map = data["correlation_map"].astype(np.float32)

        # Channel 2 arrays.
        if "mean_image_channel_2" in data:
            self.mean_image_channel_2 = data["mean_image_channel_2"].astype(np.float32)
        if "enhanced_mean_image_channel_2" in data:
            self.enhanced_mean_image_channel_2 = data["enhanced_mean_image_channel_2"].astype(np.float32)
        if "maximum_projection_channel_2" in data:
            self.maximum_projection_channel_2 = data["maximum_projection_channel_2"].astype(np.float32)
        if "correlation_map_channel_2" in data:
            self.correlation_map_channel_2 = data["correlation_map_channel_2"].astype(np.float32)


@dataclass
class ROIStatistics:
    """Stores spatial and statistical properties for a single region of interest (ROI).

    This dataclass represents the complete set of properties computed for each detected cell ROI during the detection,
    extraction, and optional multi-day processing stages. The fields are organized into required core properties
    (always present after detection), shape statistics (computed during ROI detection, with defaults for staged
    construction), optional extraction properties (added during signal extraction), multi-plane/multi-day properties,
    and GUI visualization properties.

    Notes:
        This dataclass replaces the legacy dictionary-based stat.npy format. Shape statistics fields have default
        values to support staged construction where ROIStatistics is first created during detection with only core
        fields, then updated with computed shape statistics.
    """

    # Core pixel data (required, from detection).
    y_pixels: NDArray[np.int32]
    """The y-coordinates (row indices) of all pixels belonging to this ROI."""

    x_pixels: NDArray[np.int32]
    """The x-coordinates (column indices) of all pixels belonging to this ROI."""

    pixel_weights: NDArray[np.float32]
    """The spatial filter weights (lambda values) for each pixel, indicating contribution to the ROI signal."""

    centroid: tuple[int, int]
    """The median (y, x) pixel position of the ROI, representing its approximate center."""

    footprint: int = 0
    """The spatial scale (hop size) used during sparse detection for this ROI."""

    # Shape statistics (computed during ROI detection, with defaults for staged construction).
    mean_radius: float = 0.0
    """The mean Euclidean distance from ROI pixels to their median center."""

    baseline_mean_radius: float = 0.0
    """The expected mean radius for a uniformly distributed set of pixels of the same count as the ROI."""

    compactness: float = 0.0
    """The ratio of actual to expected mean radius, where values near 1 indicate compact circular ROIs."""

    solidity: float = 0.0
    """The ratio of soma pixels to convex hull area, measuring how solid/filled the ROI is."""

    pixel_count: int = 0
    """The total number of pixels in the complete ROI."""

    soma_pixel_count: int = 0
    """The number of pixels in the soma-cropped region of the ROI."""

    soma_mask: NDArray[np.bool_] | None = None
    """The boolean mask indicating which pixels belong to the soma region."""

    overlap_mask: NDArray[np.bool_] | None = None
    """The boolean mask indicating which pixels overlap with other ROIs."""

    radius: float = 0.0
    """The fitted ellipse radius representing the approximate ROI size."""

    aspect_ratio: float = 0.0
    """The ratio of ellipse axes, indicating ROI elongation."""

    normalized_pixel_count: float = 0.0
    """The pixel count normalized by expected cell size (soma region only)."""

    normalized_pixel_count_full: float = 0.0
    """The pixel count normalized by expected cell size (full ROI)."""

    # Optional extraction data (added during signal extraction).
    skewness: float | None = None
    """The skewness of the baseline-subtracted fluorescence time series."""

    standard_deviation: float | None = None
    """The standard deviation of the baseline-subtracted fluorescence time series."""

    neuropil_mask: NDArray[np.int32] | None = None
    """The raveled (flattened) pixel indices used for neuropil signal extraction. Each index refers to a pixel position
    in the row-major flattened representation of the imaging plane (height * width). Use ``np.unravel_index`` with the
    plane dimensions to recover 2D coordinates if needed."""

    # Multi-plane data. The plane_index should be set from IOData.plane_index during ROI creation.
    plane_index: int = 0
    """The index of the imaging plane this ROI belongs to. This field is not set during detection. It is populated
    by the IO layer during multi-plane combination, when ROIs from individual planes are merged into a single list."""

    # Multi-day tracking data. Zero values indicate the ROI has not been processed by multi-day tracking.
    cluster_id: int = 0
    """The multi-day cell cluster ID. Zero indicates unclustered, positive values indicate cluster membership."""

    raveled_pixels: NDArray[np.int32] | None = None
    """The raveled (flattened) pixel indices in the deformed multi-day visual space."""

    session_count: int = 0
    """The number of sessions in which this cell was detected during multi-day tracking."""

    # GUI visualization data (computed on demand by the GUI, persisted for session continuity).
    boundary_y_pixels: NDArray[np.float32] | None = None
    """The y-coordinates of the ROI boundary pixels used for drawing the ROI outline in the GUI."""

    boundary_x_pixels: NDArray[np.float32] | None = None
    """The x-coordinates of the ROI boundary pixels used for drawing the ROI outline in the GUI."""

    circle_y_pixels: NDArray[np.float32] | None = None
    """The y-coordinates of the circle drawn around the ROI centroid in the GUI."""

    circle_x_pixels: NDArray[np.float32] | None = None
    """The x-coordinates of the circle drawn around the ROI centroid in the GUI."""

    merged_roi_indices: tuple[int, ...] | None = None
    """The tuple of original ROI indices that were merged to create this ROI. None indicates this is not a merged 
    ROI."""

    merged_into_roi_index: int | None = None
    """The index of the merged ROI that this ROI was merged into. None indicates this ROI has not been merged."""

    colocalization_probability: float | None = None
    """The probability that this ROI is colocalized across imaging channels. None if not computed."""

    @staticmethod
    def save_list(roi_list: list[ROIStatistics], file_path: Path) -> None:
        """Saves a list of ROIStatistics instances to a compressed .npz file without pickle.

        This method concatenates variable-length arrays and stores pixel counts to enable reconstruction. All scalar
        fields are stored as 1D arrays with one element per ROI.

        Args:
            roi_list: The list of ROIStatistics instances to save.
            file_path: The path to the output .npz file.
        """
        if not roi_list:
            return

        # Concatenates variable-length pixel arrays and stores counts for reconstruction. Uses uint32 for pixel
        # counts since they are always non-negative and can be large for big ROIs.
        pixel_counts = np.array([len(roi.y_pixels) for roi in roi_list], dtype=np.uint32)
        all_y_pixels = np.concatenate([roi.y_pixels for roi in roi_list])
        all_x_pixels = np.concatenate([roi.x_pixels for roi in roi_list])
        all_pixel_weights = np.concatenate([roi.pixel_weights for roi in roi_list])

        # Stores scalar fields as 1D arrays using appropriate types: int32 for pixel coordinates, uint32 for larger
        # counts, uint16 for small non-negative integers, and float32 for real-valued measurements.
        centroids = np.array([roi.centroid for roi in roi_list], dtype=np.int32)
        footprints = np.array([roi.footprint for roi in roi_list], dtype=np.uint16)
        mean_radius = np.array([roi.mean_radius for roi in roi_list], dtype=np.float32)
        baseline_mean_radius = np.array([roi.baseline_mean_radius for roi in roi_list], dtype=np.float32)
        compactness = np.array([roi.compactness for roi in roi_list], dtype=np.float32)
        solidity = np.array([roi.solidity for roi in roi_list], dtype=np.float32)
        pixel_count = np.array([roi.pixel_count for roi in roi_list], dtype=np.uint32)
        soma_pixel_count = np.array([roi.soma_pixel_count for roi in roi_list], dtype=np.uint32)
        radius = np.array([roi.radius for roi in roi_list], dtype=np.float32)
        aspect_ratio = np.array([roi.aspect_ratio for roi in roi_list], dtype=np.float32)
        normalized_pixel_count = np.array([roi.normalized_pixel_count for roi in roi_list], dtype=np.float32)
        normalized_pixel_count_full = np.array([roi.normalized_pixel_count_full for roi in roi_list], dtype=np.float32)

        # Stores optional float fields using NaN for missing values.
        skewness = np.array(
            [roi.skewness if roi.skewness is not None else np.nan for roi in roi_list], dtype=np.float32
        )
        standard_deviation = np.array(
            [roi.standard_deviation if roi.standard_deviation is not None else np.nan for roi in roi_list],
            dtype=np.float32,
        )

        # Stores plane and multi-day tracking fields. Uses unsigned types since these are non-negative counts/indices.
        # Zero indicates "not set" or "unclustered" for multi-day fields.
        plane_index = np.array([roi.plane_index for roi in roi_list], dtype=np.uint8)
        cluster_id = np.array([roi.cluster_id for roi in roi_list], dtype=np.uint32)
        session_count = np.array([roi.session_count for roi in roi_list], dtype=np.uint16)

        # Stores optional scalar float fields using NaN for missing values.
        colocalization_probability = np.array(
            [
                roi.colocalization_probability if roi.colocalization_probability is not None else np.nan
                for roi in roi_list
            ],
            dtype=np.float32,
        )

        # Stores optional integer fields using -1 for missing values (since valid indices are non-negative).
        merged_into_roi_index = np.array(
            [roi.merged_into_roi_index if roi.merged_into_roi_index is not None else -1 for roi in roi_list],
            dtype=np.int32,
        )

        # Builds the save dictionary with core and scalar fields.
        save_dict: dict[str, np.ndarray] = {
            "pixel_counts": pixel_counts,
            "y_pixels": all_y_pixels,
            "x_pixels": all_x_pixels,
            "pixel_weights": all_pixel_weights,
            "centroids": centroids,
            "footprints": footprints,
            "mean_radius": mean_radius,
            "baseline_mean_radius": baseline_mean_radius,
            "compactness": compactness,
            "solidity": solidity,
            "pixel_count": pixel_count,
            "soma_pixel_count": soma_pixel_count,
            "radius": radius,
            "aspect_ratio": aspect_ratio,
            "normalized_pixel_count": normalized_pixel_count,
            "normalized_pixel_count_full": normalized_pixel_count_full,
            "skewness": skewness,
            "standard_deviation": standard_deviation,
            "plane_index": plane_index,
            "cluster_id": cluster_id,
            "session_count": session_count,
            "colocalization_probability": colocalization_probability,
            "merged_into_roi_index": merged_into_roi_index,
        }

        # Saves optional variable-length array fields.
        _save_optional_array_field("soma_mask", [roi.soma_mask for roi in roi_list], save_dict, dtype=np.bool_)
        _save_optional_array_field("overlap_mask", [roi.overlap_mask for roi in roi_list], save_dict, dtype=np.bool_)
        _save_optional_array_field("neuropil_mask", [roi.neuropil_mask for roi in roi_list], save_dict, dtype=np.int32)
        _save_optional_array_field(
            "raveled_pixels", [roi.raveled_pixels for roi in roi_list], save_dict, dtype=np.int32
        )
        _save_optional_array_field(
            "boundary_y_pixels", [roi.boundary_y_pixels for roi in roi_list], save_dict, dtype=np.float32
        )
        _save_optional_array_field(
            "boundary_x_pixels", [roi.boundary_x_pixels for roi in roi_list], save_dict, dtype=np.float32
        )
        _save_optional_array_field(
            "circle_y_pixels", [roi.circle_y_pixels for roi in roi_list], save_dict, dtype=np.float32
        )
        _save_optional_array_field(
            "circle_x_pixels", [roi.circle_x_pixels for roi in roi_list], save_dict, dtype=np.float32
        )
        _save_optional_array_field(
            "merged_roi_indices", [roi.merged_roi_indices for roi in roi_list], save_dict, dtype=np.int32
        )

        np.savez(file_path, allow_pickle=False, **save_dict)

    @staticmethod
    def load_list(file_path: Path) -> list[ROIStatistics]:
        """Loads a list of ROIStatistics instances from a compressed .npz file.

        Args:
            file_path: The path to the .npz file containing the serialized ROI statistics.

        Returns:
            A list of ROIStatistics instances reconstructed from the file.
        """
        data = np.load(file_path, allow_pickle=False)

        pixel_counts = data["pixel_counts"]
        roi_count = len(pixel_counts)

        # Computes split indices for variable-length arrays.
        pixel_splits = np.cumsum(pixel_counts)[:-1]

        # Splits concatenated core pixel arrays back into per-ROI arrays.
        y_pixels_list = np.split(data["y_pixels"], pixel_splits)
        x_pixels_list = np.split(data["x_pixels"], pixel_splits)
        pixel_weights_list = np.split(data["pixel_weights"], pixel_splits)

        # Extracts scalar arrays.
        centroids = data["centroids"]
        footprints = data["footprints"]
        mean_radius = data["mean_radius"]
        baseline_mean_radius = data["baseline_mean_radius"]
        compactness = data["compactness"]
        solidity = data["solidity"]
        pixel_count = data["pixel_count"]
        soma_pixel_count = data["soma_pixel_count"]
        radius = data["radius"]
        aspect_ratio = data["aspect_ratio"]
        normalized_pixel_count = data["normalized_pixel_count"]
        normalized_pixel_count_full = data["normalized_pixel_count_full"]
        skewness = data["skewness"]
        standard_deviation = data["standard_deviation"]
        plane_index = data["plane_index"]
        cluster_id = data["cluster_id"]
        session_count = data["session_count"]
        colocalization_probability = data["colocalization_probability"]
        merged_into_roi_index = data["merged_into_roi_index"]

        # Loads optional variable-length array fields.
        soma_mask_list = _load_optional_array_field("soma_mask", roi_count, data, dtype=np.bool_)
        overlap_mask_list = _load_optional_array_field("overlap_mask", roi_count, data, dtype=np.bool_)
        neuropil_mask_list = _load_optional_array_field("neuropil_mask", roi_count, data, dtype=np.int32)
        raveled_pixels_list = _load_optional_array_field("raveled_pixels", roi_count, data, dtype=np.int32)
        boundary_y_pixels_list = _load_optional_array_field("boundary_y_pixels", roi_count, data, dtype=np.float32)
        boundary_x_pixels_list = _load_optional_array_field("boundary_x_pixels", roi_count, data, dtype=np.float32)
        circle_y_pixels_list = _load_optional_array_field("circle_y_pixels", roi_count, data, dtype=np.float32)
        circle_x_pixels_list = _load_optional_array_field("circle_x_pixels", roi_count, data, dtype=np.float32)
        merged_roi_indices_list = _load_optional_array_field("merged_roi_indices", roi_count, data, dtype=np.int32)

        # Reconstructs ROIStatistics instances.
        roi_list = []
        for i in range(roi_count):
            # Converts merged_roi_indices array back to tuple[int, ...] if present.
            merged_indices = merged_roi_indices_list[i]
            merged_indices_as_tuple = tuple(int(v) for v in merged_indices) if merged_indices is not None else None

            roi = ROIStatistics(
                y_pixels=y_pixels_list[i].astype(np.int32),
                x_pixels=x_pixels_list[i].astype(np.int32),
                pixel_weights=pixel_weights_list[i].astype(np.float32),
                centroid=(int(centroids[i, 0]), int(centroids[i, 1])),
                footprint=int(footprints[i]),
                mean_radius=float(mean_radius[i]),
                baseline_mean_radius=float(baseline_mean_radius[i]),
                compactness=float(compactness[i]),
                solidity=float(solidity[i]),
                pixel_count=int(pixel_count[i]),
                soma_pixel_count=int(soma_pixel_count[i]),
                soma_mask=soma_mask_list[i],  # type: ignore[arg-type]
                overlap_mask=overlap_mask_list[i],  # type: ignore[arg-type]
                radius=float(radius[i]),
                aspect_ratio=float(aspect_ratio[i]),
                normalized_pixel_count=float(normalized_pixel_count[i]),
                normalized_pixel_count_full=float(normalized_pixel_count_full[i]),
                skewness=None if np.isnan(skewness[i]) else float(skewness[i]),
                standard_deviation=None if np.isnan(standard_deviation[i]) else float(standard_deviation[i]),
                neuropil_mask=neuropil_mask_list[i],  # type: ignore[arg-type]
                plane_index=int(plane_index[i]),
                cluster_id=int(cluster_id[i]),
                raveled_pixels=raveled_pixels_list[i],  # type: ignore[arg-type]
                session_count=int(session_count[i]),
                boundary_y_pixels=boundary_y_pixels_list[i],  # type: ignore[arg-type]
                boundary_x_pixels=boundary_x_pixels_list[i],  # type: ignore[arg-type]
                circle_y_pixels=circle_y_pixels_list[i],  # type: ignore[arg-type]
                circle_x_pixels=circle_x_pixels_list[i],  # type: ignore[arg-type]
                merged_roi_indices=merged_indices_as_tuple,
                merged_into_roi_index=None if merged_into_roi_index[i] < 0 else int(merged_into_roi_index[i]),
                colocalization_probability=(
                    None if np.isnan(colocalization_probability[i]) else float(colocalization_probability[i])
                ),
            )
            roi_list.append(roi)

        return roi_list


@dataclass
class ExtractionData:
    """Stores runtime data from the extraction stage."""

    # Channel 1 extraction data.
    roi_statistics: list[ROIStatistics] | None = None
    """The list of ROIStatistics instances containing spatial and shape statistics for each detected ROI."""

    cell_fluorescence: NDArray[np.float32] | None = None
    """The cell fluorescence traces with shape (cells, frames)."""

    neuropil_fluorescence: NDArray[np.float32] | None = None
    """The neuropil fluorescence traces with shape (cells, frames)."""

    subtracted_fluorescence: NDArray[np.float32] | None = None
    """The baseline-and-neuropil-subtracted fluorescence traces with shape (cells, frames)."""

    spikes: NDArray[np.float32] | None = None
    """The deconvolved spike traces with shape (cells, frames)."""

    cell_classification: NDArray[np.float32] | None = None
    """The cell classification results with shape (cells, 2) containing (probability, is_cell_boolean)."""

    # Channel 2 extraction data (when both channels are functional).
    roi_statistics_channel_2: list[ROIStatistics] | None = None
    """The list of ROIStatistics instances containing spatial and shape statistics for each detected ROI for channel
    2 when both channels are functional."""

    cell_fluorescence_channel_2: NDArray[np.float32] | None = None
    """The cell fluorescence traces for channel 2."""

    neuropil_fluorescence_channel_2: NDArray[np.float32] | None = None
    """The neuropil fluorescence traces for channel 2."""

    subtracted_fluorescence_channel_2: NDArray[np.float32] | None = None
    """The baseline-and-neuropil-subtracted fluorescence for channel 2."""

    spikes_channel_2: NDArray[np.float32] | None = None
    """The deconvolved spike traces for channel 2."""

    cell_classification_channel_2: NDArray[np.float32] | None = None
    """The cell classification results for channel 2."""

    # Colocalization data (channel 1 ROIs presence in channel 2).
    cell_colocalization: NDArray[np.float32] | None = None
    """The colocalization results indicating whether channel 1 ROIs are present in channel 2. Shape is (cells, 2)
    containing (probability, is_colocalized_boolean)."""

    corrected_structural_mean_image: NDArray[np.float32] | None = None
    """The bleed-through-corrected mean image for the structural channel, computed during intensity-based
    colocalization. The structural channel is whichever channel is not functional (channel 1 if only channel 2 is
    functional, or channel 2 if only channel 1 is functional). This field is not computed when both channels are
    functional, as spatial colocalization is used instead."""

    def prepare_for_saving(self) -> None:
        """Sets all array and list fields to None for YAML serialization."""
        # Channel 1.
        self.roi_statistics = None
        self.cell_fluorescence = None
        self.neuropil_fluorescence = None
        self.subtracted_fluorescence = None
        self.spikes = None
        self.cell_classification = None

        # Channel 2.
        self.roi_statistics_channel_2 = None
        self.cell_fluorescence_channel_2 = None
        self.neuropil_fluorescence_channel_2 = None
        self.subtracted_fluorescence_channel_2 = None
        self.spikes_channel_2 = None
        self.cell_classification_channel_2 = None

        # Colocalization.
        self.cell_colocalization = None
        self.corrected_structural_mean_image = None

    def save_arrays(self, output_path: Path) -> None:
        """Saves all extraction arrays to .npy files and ROI statistics to .npz files.

        Args:
            output_path: The directory where to save the extraction data files.
        """
        # Channel 1 ROI statistics.
        if self.roi_statistics is not None:
            ROIStatistics.save_list(self.roi_statistics, output_path / "roi_statistics.npz")

        # Channel 1 trace arrays.
        if self.cell_fluorescence is not None:
            np.save(output_path / "cell_fluorescence.npy", self.cell_fluorescence, allow_pickle=False)
        if self.neuropil_fluorescence is not None:
            np.save(output_path / "neuropil_fluorescence.npy", self.neuropil_fluorescence, allow_pickle=False)
        if self.subtracted_fluorescence is not None:
            np.save(output_path / "subtracted_fluorescence.npy", self.subtracted_fluorescence, allow_pickle=False)
        if self.spikes is not None:
            np.save(output_path / "spikes.npy", self.spikes, allow_pickle=False)
        if self.cell_classification is not None:
            np.save(output_path / "cell_classification.npy", self.cell_classification, allow_pickle=False)

        # Channel 2 ROI statistics.
        if self.roi_statistics_channel_2 is not None:
            ROIStatistics.save_list(self.roi_statistics_channel_2, output_path / "roi_statistics_channel_2.npz")

        # Channel 2 trace arrays.
        if self.cell_fluorescence_channel_2 is not None:
            np.save(
                output_path / "cell_fluorescence_channel_2.npy", self.cell_fluorescence_channel_2, allow_pickle=False
            )
        if self.neuropil_fluorescence_channel_2 is not None:
            np.save(
                output_path / "neuropil_fluorescence_channel_2.npy",
                self.neuropil_fluorescence_channel_2,
                allow_pickle=False,
            )
        if self.subtracted_fluorescence_channel_2 is not None:
            np.save(
                output_path / "subtracted_fluorescence_channel_2.npy",
                self.subtracted_fluorescence_channel_2,
                allow_pickle=False,
            )
        if self.spikes_channel_2 is not None:
            np.save(output_path / "spikes_channel_2.npy", self.spikes_channel_2, allow_pickle=False)
        if self.cell_classification_channel_2 is not None:
            np.save(
                output_path / "cell_classification_channel_2.npy",
                self.cell_classification_channel_2,
                allow_pickle=False,
            )

        # Colocalization arrays.
        if self.cell_colocalization is not None:
            np.save(output_path / "cell_colocalization.npy", self.cell_colocalization, allow_pickle=False)
        if self.corrected_structural_mean_image is not None:
            np.save(
                output_path / "corrected_structural_mean_image.npy",
                self.corrected_structural_mean_image,
                allow_pickle=False,
            )

    def load_arrays(self, output_path: Path) -> None:
        """Loads ROI statistics from disk.

        This method loads only ROI statistics, which are the only extraction data needed during pipeline processing
        (specifically for multi-day cell tracking). Fluorescence traces, classification results, and colocalization
        data are not loaded because they are never needed during pipeline execution and consume significant memory.
        Use load_results() to load all result arrays when needed for analysis or visualization.

        Args:
            output_path: The directory containing the extraction data files.
        """
        # Channel 1 ROI statistics.
        roi_stats_path = output_path / "roi_statistics.npz"
        if self.roi_statistics is None and roi_stats_path.exists():
            self.roi_statistics = ROIStatistics.load_list(roi_stats_path)

        # Channel 2 ROI statistics.
        roi_stats_channel_2_path = output_path / "roi_statistics_channel_2.npz"
        if self.roi_statistics_channel_2 is None and roi_stats_channel_2_path.exists():
            self.roi_statistics_channel_2 = ROIStatistics.load_list(roi_stats_channel_2_path)

    def load_results(self, output_path: Path) -> None:
        """Loads all extraction result arrays from disk.

        This method loads fluorescence traces, classification results, and colocalization data. These arrays are
        not loaded by load_arrays() because they are never needed during pipeline processing (they are created
        and saved but never re-loaded) and they consume significant memory. Call this method when result data
        is needed for analysis or visualization.

        Args:
            output_path: The directory containing the result .npy files.
        """
        # Channel 1 traces.
        cell_fluorescence_path = output_path / "cell_fluorescence.npy"
        if self.cell_fluorescence is None and cell_fluorescence_path.exists():
            self.cell_fluorescence = np.load(cell_fluorescence_path, allow_pickle=False).astype(np.float32)

        neuropil_fluorescence_path = output_path / "neuropil_fluorescence.npy"
        if self.neuropil_fluorescence is None and neuropil_fluorescence_path.exists():
            self.neuropil_fluorescence = np.load(neuropil_fluorescence_path, allow_pickle=False).astype(np.float32)

        subtracted_fluorescence_path = output_path / "subtracted_fluorescence.npy"
        if self.subtracted_fluorescence is None and subtracted_fluorescence_path.exists():
            self.subtracted_fluorescence = np.load(subtracted_fluorescence_path, allow_pickle=False).astype(np.float32)

        spikes_path = output_path / "spikes.npy"
        if self.spikes is None and spikes_path.exists():
            self.spikes = np.load(spikes_path, allow_pickle=False).astype(np.float32)

        # Channel 1 classification.
        cell_classification_path = output_path / "cell_classification.npy"
        if self.cell_classification is None and cell_classification_path.exists():
            self.cell_classification = np.load(cell_classification_path, allow_pickle=False).astype(np.float32)

        # Channel 2 traces.
        cell_fluorescence_channel_2_path = output_path / "cell_fluorescence_channel_2.npy"
        if self.cell_fluorescence_channel_2 is None and cell_fluorescence_channel_2_path.exists():
            self.cell_fluorescence_channel_2 = np.load(cell_fluorescence_channel_2_path, allow_pickle=False).astype(
                np.float32
            )

        neuropil_fluorescence_channel_2_path = output_path / "neuropil_fluorescence_channel_2.npy"
        if self.neuropil_fluorescence_channel_2 is None and neuropil_fluorescence_channel_2_path.exists():
            self.neuropil_fluorescence_channel_2 = np.load(
                neuropil_fluorescence_channel_2_path, allow_pickle=False
            ).astype(np.float32)

        subtracted_fluorescence_channel_2_path = output_path / "subtracted_fluorescence_channel_2.npy"
        if self.subtracted_fluorescence_channel_2 is None and subtracted_fluorescence_channel_2_path.exists():
            self.subtracted_fluorescence_channel_2 = np.load(
                subtracted_fluorescence_channel_2_path, allow_pickle=False
            ).astype(np.float32)

        spikes_channel_2_path = output_path / "spikes_channel_2.npy"
        if self.spikes_channel_2 is None and spikes_channel_2_path.exists():
            self.spikes_channel_2 = np.load(spikes_channel_2_path, allow_pickle=False).astype(np.float32)

        # Channel 2 classification.
        cell_classification_channel_2_path = output_path / "cell_classification_channel_2.npy"
        if self.cell_classification_channel_2 is None and cell_classification_channel_2_path.exists():
            self.cell_classification_channel_2 = np.load(cell_classification_channel_2_path, allow_pickle=False).astype(
                np.float32
            )

        # Colocalization arrays.
        cell_colocalization_path = output_path / "cell_colocalization.npy"
        if self.cell_colocalization is None and cell_colocalization_path.exists():
            self.cell_colocalization = np.load(cell_colocalization_path, allow_pickle=False).astype(np.float32)

        corrected_structural_mean_image_path = output_path / "corrected_structural_mean_image.npy"
        if self.corrected_structural_mean_image is None and corrected_structural_mean_image_path.exists():
            self.corrected_structural_mean_image = np.load(
                corrected_structural_mean_image_path, allow_pickle=False
            ).astype(np.float32)


@dataclass
class TimingData:
    """Stores pipeline timing and version data.

    All time durations are stored as integers representing seconds.
    """

    binarization_time: int = 0
    """The TIFF to binary conversion time in seconds."""

    registration_time: int = 0
    """The registration step time in seconds."""

    two_step_registration_time: int = 0
    """The second registration step time in seconds."""

    registration_metrics_time: int = 0
    """The registration metrics computation time in seconds."""

    detection_time: int = 0
    """The ROI detection time in seconds."""

    extraction_time: int = 0
    """The fluorescence extraction time in seconds."""

    classification_time: int = 0
    """The ROI classification time in seconds."""

    deconvolution_time: int = 0
    """The spike deconvolution time in seconds."""

    detection_time_channel_2: int = 0
    """The channel 2 ROI detection time in seconds."""

    extraction_time_channel_2: int = 0
    """The channel 2 fluorescence extraction time in seconds."""

    classification_time_channel_2: int = 0
    """The channel 2 ROI classification time in seconds."""

    deconvolution_time_channel_2: int = 0
    """The channel 2 spike deconvolution time in seconds."""

    total_plane_time: int = 0
    """The total plane processing time in seconds."""

    date_processed: str = ""
    """The timestamp when processing completed in ataraxis-time format (yyyy-mm-dd-hh-mm-ss-us)."""

    python_version: str = python_version
    """Python version used for processing."""

    cindra_version: str = version
    """cindra version used for processing."""


@dataclass
class SingleDayRuntimeData(YamlConfig):
    """Aggregates all runtime data for a single plane."""

    output_path: Path | None = None
    """The path to the directory where runtime data and .npy files are stored."""

    io: IOData = field(default_factory=IOData)
    """The runtime data from the IO/binarization stage."""

    registration: RegistrationData = field(default_factory=RegistrationData)
    """The runtime data from the registration stage."""

    detection: DetectionData = field(default_factory=DetectionData)
    """The runtime data from the detection stage."""

    extraction: ExtractionData = field(default_factory=ExtractionData)
    """The runtime data from the extraction and classification stages."""

    timing: TimingData = field(default_factory=TimingData)
    """The pipeline timing information."""

    def __post_init__(self) -> None:
        """Loads NumPy arrays from .npy files if output_path is set and arrays are None."""
        if self.output_path is None:
            return

        # Loads arrays from each child dataclass.
        self.registration.load_arrays(self.output_path)
        self.detection.load_arrays(self.output_path)
        self.extraction.load_arrays(self.output_path)

    def save(self, output_path: Path) -> None:
        """Saves the runtime data to a YAML file and arrays to .npy files.

        This method saves all NumPy arrays as separate .npy files in the output directory, then creates
        a deep copy of the instance with arrays set to None and Path fields converted to strings before
        writing the YAML file.

        Notes:
            This form of storing the data mitigates the use of pickle serialization in favor of using safer YAML and
            NumPy serialization.

        Args:
            output_path: The directory where to save the runtime_data.yaml file and .npy files.
        """
        ensure_directory_exists(output_path)
        self.output_path = output_path

        # Saves arrays from each child dataclass.
        self.registration.save_arrays(output_path)
        self.detection.save_arrays(output_path)
        self.extraction.save_arrays(output_path)

        # Creates a deep copy for YAML serialization. The deep copy is still needed because array fields must be
        # nulled for YAML serialization while keeping the originals intact in memory.
        yaml_copy = copy.deepcopy(self)

        # Nulls array fields in child dataclasses for YAML serialization.
        yaml_copy.registration.prepare_for_saving()
        yaml_copy.detection.prepare_for_saving()
        yaml_copy.extraction.prepare_for_saving()

        # Saves the YAML file.
        file_path = output_path / "runtime_data.yaml"
        yaml_copy.to_yaml(file_path=file_path)

    @classmethod
    def load(cls, output_path: Path) -> SingleDayRuntimeData:
        """Loads runtime data from a YAML file and associated .npy files.

        Args:
            output_path: The directory containing the runtime_data.yaml file.

        Returns:
            A SingleDayRuntimeData instance with all data loaded, including NumPy arrays.
        """
        file_path = output_path / "runtime_data.yaml"
        return cls.from_yaml(file_path=file_path)

    def load_results(self) -> None:
        """Loads all extraction result arrays for this plane.

        This method loads fluorescence traces, classification results, and colocalization data. These arrays
        are not loaded automatically because they are never needed during pipeline processing and consume
        significant memory. Call this method when result data is needed for analysis or visualization.
        """
        if self.output_path is not None:
            self.extraction.load_results(self.output_path)


@dataclass
class CombinedData:
    """Stores combined multi-plane detection and extraction data.

    This class provides a container for the results of combining processed data from multiple imaging planes
    into a unified dataset. It holds DetectionData (combined images) and ExtractionData (combined ROI statistics,
    fluorescence traces, and classification results) along with metadata about the combined field of view.

    Notes:
        Combined data is saved to the root suite2p directory alongside configuration.yaml and
        acquisition_parameters.yaml. The same filenames are used as per-plane data, but stored at the root
        level rather than in plane subdirectories.
    """

    detection: DetectionData
    """The combined detection data including mean images, correlation maps, and maximum projections for both
    channels."""

    extraction: ExtractionData
    """The combined extraction data including ROI statistics, fluorescence traces, and classification results for
    both channels."""

    plane_count: int = 0
    """The number of planes that were combined."""

    combined_height: int = 0
    """The height of the combined field of view in pixels."""

    combined_width: int = 0
    """The width of the combined field of view in pixels."""

    tau: float = 0.0
    """The timescale of the calcium indicator sensor in seconds, cached from the single-day configuration for use by
    the multi-day extraction pipeline."""

    sampling_rate: float = 0.0
    """The per-plane sampling rate in Hertz, cached from the single-day runtime for use by the multi-day extraction
    pipeline."""

    plane_heights: NDArray[np.uint16] = field(default_factory=lambda: np.array([], dtype=np.uint16))
    """Per-plane frame heights in pixels."""

    plane_widths: NDArray[np.uint16] = field(default_factory=lambda: np.array([], dtype=np.uint16))
    """Per-plane frame widths in pixels."""

    plane_y_offsets: NDArray[np.int32] = field(default_factory=lambda: np.array([], dtype=np.int32))
    """Per-plane y-axis displacement from compute_plane_offsets(), used to arrange planes in the combined view."""

    plane_x_offsets: NDArray[np.int32] = field(default_factory=lambda: np.array([], dtype=np.int32))
    """Per-plane x-axis displacement from compute_plane_offsets(), used to arrange planes in the combined view."""

    registered_binary_paths: tuple[Path, ...] = ()
    """Channel 1 registered binary file paths, one per plane."""

    registered_binary_paths_channel_2: tuple[Path, ...] | None = None
    """Channel 2 registered binary file paths, one per plane. None when the recording is single-channel."""

    def save(self, root_path: Path) -> None:
        """Saves combined data to the root suite2p directory.

        This method saves all combined detection and extraction arrays to the root suite2p directory. Metadata
        (plane count, dimensions) is saved to combined_metadata.npz.

        Args:
            root_path: The root suite2p output directory containing configuration.yaml.
        """
        ensure_directory_exists(root_path)

        # Saves metadata using appropriate unsigned types for counts and dimensions. Binary paths are stored as
        # strings relative to root_path to allow relocating processed data without breaking path references.
        relative_binary_paths = np.array(
            [str(p.relative_to(root_path)) for p in self.registered_binary_paths], dtype=str
        )

        save_kwargs: dict[
            str,
            NDArray[np.uint8]
            | NDArray[np.uint16]
            | NDArray[np.uint32]
            | NDArray[np.int32]
            | NDArray[np.float32]
            | NDArray[np.str_],
        ] = {
            "plane_count": np.array([self.plane_count], dtype=np.uint8),
            "combined_height": np.array([self.combined_height], dtype=np.uint32),
            "combined_width": np.array([self.combined_width], dtype=np.uint32),
            "tau": np.array([self.tau], dtype=np.float32),
            "sampling_rate": np.array([self.sampling_rate], dtype=np.float32),
            "plane_heights": self.plane_heights,
            "plane_widths": self.plane_widths,
            "plane_y_offsets": self.plane_y_offsets,
            "plane_x_offsets": self.plane_x_offsets,
            "registered_binary_paths": relative_binary_paths,
        }

        if self.registered_binary_paths_channel_2 is not None:
            save_kwargs["registered_binary_paths_channel_2"] = np.array(
                [str(p.relative_to(root_path)) for p in self.registered_binary_paths_channel_2], dtype=str
            )

        np.savez(root_path / "combined_metadata.npz", allow_pickle=False, **save_kwargs)

        # Saves combined detection and extraction arrays using existing methods.
        self.detection.save_arrays(root_path)
        self.extraction.save_arrays(root_path)

    @classmethod
    def load(cls, root_path: Path) -> CombinedData:
        """Loads combined data from the root suite2p directory.

        Args:
            root_path: The root suite2p output directory containing combined_metadata.npz.

        Returns:
            A CombinedData instance with all combined arrays loaded.

        Raises:
            FileNotFoundError: If the combined metadata file does not exist.
        """
        metadata_path = root_path / "combined_metadata.npz"
        if not metadata_path.exists():
            message = (
                f"Unable to load combined data. The combined metadata file does not exist at the specified path: "
                f"{metadata_path}."
            )
            console.error(message=message, error=FileNotFoundError)

        # Loads metadata.
        metadata = np.load(metadata_path, allow_pickle=False)
        plane_count = int(metadata["plane_count"][0])
        combined_height = int(metadata["combined_height"][0])
        combined_width = int(metadata["combined_width"][0])

        # Loads tau and sampling_rate.
        tau = float(metadata["tau"][0])
        sampling_rate = float(metadata["sampling_rate"][0])

        # Loads per-plane geometry and binary paths. These keys may be absent in metadata files saved before these
        # fields were added, so defaults are used for backward compatibility.
        plane_heights: NDArray[np.uint16] = np.array([], dtype=np.uint16)
        plane_widths: NDArray[np.uint16] = np.array([], dtype=np.uint16)
        plane_y_offsets: NDArray[np.int32] = np.array([], dtype=np.int32)
        plane_x_offsets: NDArray[np.int32] = np.array([], dtype=np.int32)
        registered_binary_paths: tuple[Path, ...] = ()
        registered_binary_paths_channel_2: tuple[Path, ...] | None = None

        if "plane_heights" in metadata:
            plane_heights = metadata["plane_heights"].astype(np.uint16)
            plane_widths = metadata["plane_widths"].astype(np.uint16)
            plane_y_offsets = metadata["plane_y_offsets"].astype(np.int32)
            plane_x_offsets = metadata["plane_x_offsets"].astype(np.int32)

        if "registered_binary_paths" in metadata:
            registered_binary_paths = tuple(root_path / str(p) for p in metadata["registered_binary_paths"])

        if "registered_binary_paths_channel_2" in metadata:
            registered_binary_paths_channel_2 = tuple(
                root_path / str(p) for p in metadata["registered_binary_paths_channel_2"]
            )

        # Loads detection and extraction arrays using existing methods.
        detection = DetectionData()
        detection.load_arrays(root_path)

        extraction = ExtractionData()
        extraction.load_arrays(root_path)

        return cls(
            detection=detection,
            extraction=extraction,
            plane_count=plane_count,
            combined_height=combined_height,
            combined_width=combined_width,
            tau=tau,
            sampling_rate=sampling_rate,
            plane_heights=plane_heights,
            plane_widths=plane_widths,
            plane_y_offsets=plane_y_offsets,
            plane_x_offsets=plane_x_offsets,
            registered_binary_paths=registered_binary_paths,
            registered_binary_paths_channel_2=registered_binary_paths_channel_2,
        )

    def load_results(self, root_path: Path) -> None:
        """Loads all extraction result arrays for analysis.

        This method loads fluorescence traces, classification results, and colocalization data. These arrays
        are not loaded by the load() classmethod because they are never needed during pipeline processing
        and consume significant memory. Call this method when result data is needed for post-processing
        analysis or visualization.

        Args:
            root_path: The root suite2p output directory containing the result .npy files.
        """
        self.extraction.load_results(root_path)
