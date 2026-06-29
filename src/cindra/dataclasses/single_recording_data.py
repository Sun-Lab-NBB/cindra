"""Provides runtime data classes for the single-recording (within-recording) processing pipeline."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any
from pathlib import Path  # noqa: TC003 - needed at runtime for dacite deserialization
from functools import cached_property
from dataclasses import field, dataclass

import numpy as np
from numpy.typing import NDArray  # noqa: TC002 - needed at runtime for dacite deserialization
from ataraxis_base_utilities import console, ensure_directory_exists
from ataraxis_data_structures import YamlConfig

from .version import version, python_version

if TYPE_CHECKING:
    from numpy.lib.npyio import NpzFile


def is_memory_mapped(array: NDArray[np.generic] | None) -> bool:
    """Checks whether the input array is a memory-mapped numpy array."""
    return isinstance(array, np.memmap)


@dataclass(slots=True)
class IOData:
    """Stores the Input / Output runtime data for all stages of the single-recording processing pipeline."""

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

    output_path: Path | None = None
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


@dataclass(slots=True)
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
    """Indicates whether bidirectional phase correction was applied during registration."""

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
    """The vertical (Y) translation offsets from nonrigid registration, per frame and per block."""

    nonrigid_x_offsets: NDArray[np.float32] | None = None
    """The horizontal (X) translation offsets from nonrigid registration, per frame and per block."""

    nonrigid_correlations: NDArray[np.float32] | None = None
    """The phase correlation values from nonrigid registration, indicating alignment quality per frame and block."""

    principal_component_extreme_images: NDArray[np.float32] | None = None
    """The mean images from frames at extreme ends of each principal component of the registered recording movie, with
    shape (2, num_components, height, width). Index 0 contains low-projection means, index 1 contains high-projection
    means. Used for visualizing registration quality in the GUI."""

    principal_component_projections: NDArray[np.float32] | None = None
    """The projection of each frame onto the principal components of the registered recording movie, with shape
    (num_frames, num_components). Shows how each frame relates to the computed PCs over time."""

    principal_component_shift_metrics: NDArray[np.float32] | None = None
    """The registration offset metrics computed by aligning PC extreme images of the registered recording movie, with
    shape (num_components, 3). Column 0 contains the rigid offset magnitude, column 1 contains mean nonrigid offset
    magnitude, and column 2 contains maximum nonrigid offset magnitude. Large values indicate poor registration
    quality."""

    def is_registered(self, output_path: Path | None = None) -> bool:
        """Checks whether registration data exists in memory or on disk.

        Args:
            output_path: The directory containing the ``registration_data/`` subdirectory. When provided and arrays
                are not loaded in memory, checks for registration files on disk. The calling context is responsible for
                resolving the correct path.

        Returns:
            True if registration arrays are loaded in memory or exist on disk at the given output path, False otherwise.
        """
        arrays_loaded = (
            self.reference_image is not None and self.rigid_y_offsets is not None and self.rigid_x_offsets is not None
        )
        if arrays_loaded:
            return True
        if output_path is not None:
            return (output_path / "registration_data" / "reference_image.npy").exists()
        return False

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

    def release_arrays(self) -> None:
        """Releases all array fields to free memory.

        Scalar fields (valid ranges, normalization bounds, etc.) are preserved. Use ``memory_map_arrays()`` or
        ``load_arrays()`` to re-acquire the data on demand.
        """
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
        """Saves registration arrays as individual .npy files inside a ``registration_data/`` subdirectory.

        Args:
            output_path: The directory where to create the ``registration_data/`` subdirectory.
        """
        registration_directory = output_path / "registration_data"
        ensure_directory_exists(registration_directory)

        if self.bad_frames is not None and not is_memory_mapped(self.bad_frames):
            np.save(registration_directory / "bad_frames.npy", self.bad_frames)
        if self.reference_image is not None and not is_memory_mapped(self.reference_image):
            np.save(registration_directory / "reference_image.npy", self.reference_image)
        if self.rigid_y_offsets is not None and not is_memory_mapped(self.rigid_y_offsets):
            np.save(registration_directory / "rigid_y_offsets.npy", self.rigid_y_offsets)
        if self.rigid_x_offsets is not None and not is_memory_mapped(self.rigid_x_offsets):
            np.save(registration_directory / "rigid_x_offsets.npy", self.rigid_x_offsets)
        if self.rigid_correlations is not None and not is_memory_mapped(self.rigid_correlations):
            np.save(registration_directory / "rigid_correlations.npy", self.rigid_correlations)
        if self.nonrigid_y_offsets is not None and not is_memory_mapped(self.nonrigid_y_offsets):
            np.save(registration_directory / "nonrigid_y_offsets.npy", self.nonrigid_y_offsets)
        if self.nonrigid_x_offsets is not None and not is_memory_mapped(self.nonrigid_x_offsets):
            np.save(registration_directory / "nonrigid_x_offsets.npy", self.nonrigid_x_offsets)
        if self.nonrigid_correlations is not None and not is_memory_mapped(self.nonrigid_correlations):
            np.save(registration_directory / "nonrigid_correlations.npy", self.nonrigid_correlations)
        if self.principal_component_extreme_images is not None and not is_memory_mapped(
            self.principal_component_extreme_images
        ):
            np.save(
                registration_directory / "principal_component_extreme_images.npy",
                self.principal_component_extreme_images,
            )
        if self.principal_component_projections is not None and not is_memory_mapped(
            self.principal_component_projections
        ):
            np.save(
                registration_directory / "principal_component_projections.npy",
                self.principal_component_projections,
            )
        if self.principal_component_shift_metrics is not None and not is_memory_mapped(
            self.principal_component_shift_metrics
        ):
            np.save(
                registration_directory / "principal_component_shift_metrics.npy",
                self.principal_component_shift_metrics,
            )

    def load_arrays(self, output_path: Path) -> None:
        """Loads registration arrays from individual .npy files in the ``registration_data/`` subdirectory.

        Args:
            output_path: The directory containing the ``registration_data/`` subdirectory.
        """
        registration_directory = output_path / "registration_data"
        if not registration_directory.exists():
            return

        path = registration_directory / "bad_frames.npy"
        if path.exists():
            self.bad_frames = np.load(path, allow_pickle=False).astype(np.bool_)
        path = registration_directory / "reference_image.npy"
        if path.exists():
            self.reference_image = np.load(path, allow_pickle=False).astype(np.float32)
        path = registration_directory / "rigid_y_offsets.npy"
        if path.exists():
            self.rigid_y_offsets = np.load(path, allow_pickle=False).astype(np.int32)
        path = registration_directory / "rigid_x_offsets.npy"
        if path.exists():
            self.rigid_x_offsets = np.load(path, allow_pickle=False).astype(np.int32)
        path = registration_directory / "rigid_correlations.npy"
        if path.exists():
            self.rigid_correlations = np.load(path, allow_pickle=False).astype(np.float32)
        path = registration_directory / "nonrigid_y_offsets.npy"
        if path.exists():
            self.nonrigid_y_offsets = np.load(path, allow_pickle=False).astype(np.float32)
        path = registration_directory / "nonrigid_x_offsets.npy"
        if path.exists():
            self.nonrigid_x_offsets = np.load(path, allow_pickle=False).astype(np.float32)
        path = registration_directory / "nonrigid_correlations.npy"
        if path.exists():
            self.nonrigid_correlations = np.load(path, allow_pickle=False).astype(np.float32)
        path = registration_directory / "principal_component_extreme_images.npy"
        if path.exists():
            self.principal_component_extreme_images = np.load(path, allow_pickle=False).astype(np.float32)
        path = registration_directory / "principal_component_projections.npy"
        if path.exists():
            self.principal_component_projections = np.load(path, allow_pickle=False).astype(np.float32)
        path = registration_directory / "principal_component_shift_metrics.npy"
        if path.exists():
            self.principal_component_shift_metrics = np.load(path, allow_pickle=False).astype(np.float32)

    def memory_map_arrays(self, output_path: Path) -> None:
        """Memory-maps registration arrays from individual .npy files in the ``registration_data/`` subdirectory.

        Uses ``r+`` mode to allow both reading and writing through the memory-mapped arrays. This avoids loading the
        full array contents into memory, which is useful when reusing previously-generated data (e.g., single-recording
        outputs consumed by the multi-recording pipeline).

        Args:
            output_path: The directory containing the ``registration_data/`` subdirectory.
        """
        registration_directory = output_path / "registration_data"
        if not registration_directory.exists():
            return

        path = registration_directory / "bad_frames.npy"
        if path.exists():
            self.bad_frames = np.load(path, mmap_mode="r+")
        path = registration_directory / "reference_image.npy"
        if path.exists():
            self.reference_image = np.load(path, mmap_mode="r+")
        path = registration_directory / "rigid_y_offsets.npy"
        if path.exists():
            self.rigid_y_offsets = np.load(path, mmap_mode="r+")
        path = registration_directory / "rigid_x_offsets.npy"
        if path.exists():
            self.rigid_x_offsets = np.load(path, mmap_mode="r+")
        path = registration_directory / "rigid_correlations.npy"
        if path.exists():
            self.rigid_correlations = np.load(path, mmap_mode="r+")
        path = registration_directory / "nonrigid_y_offsets.npy"
        if path.exists():
            self.nonrigid_y_offsets = np.load(path, mmap_mode="r+")
        path = registration_directory / "nonrigid_x_offsets.npy"
        if path.exists():
            self.nonrigid_x_offsets = np.load(path, mmap_mode="r+")
        path = registration_directory / "nonrigid_correlations.npy"
        if path.exists():
            self.nonrigid_correlations = np.load(path, mmap_mode="r+")
        path = registration_directory / "principal_component_extreme_images.npy"
        if path.exists():
            self.principal_component_extreme_images = np.load(path, mmap_mode="r+")
        path = registration_directory / "principal_component_projections.npy"
        if path.exists():
            self.principal_component_projections = np.load(path, mmap_mode="r+")
        path = registration_directory / "principal_component_shift_metrics.npy"
        if path.exists():
            self.principal_component_shift_metrics = np.load(path, mmap_mode="r+")


@dataclass(slots=True)
class DetectionData:
    """Stores runtime data from the detection stage."""

    roi_diameter: int = 0
    """The estimated ROI diameter in pixels, automatically computed from the spatial scale during detection."""

    aspect_ratio: float = 0.0
    """The median normalized aspect ratio across detected ROIs, computed from the fitted ellipse semi-axes as
    2*major/(major+minor), bounded between 0 and 2 where 1 indicates a circular shape."""

    mean_image: NDArray[np.float32] | None = None
    """The temporal mean of all registered frames, providing a static view of the imaging field."""

    enhanced_mean_image: NDArray[np.float32] | None = None
    """The high-pass filtered mean image that enhances ROI boundaries for improved detection."""

    maximum_projection: NDArray[np.float32] | None = None
    """The maximum intensity projection across all frames, highlighting active structures."""

    correlation_map: NDArray[np.float32] | None = None
    """The pixel-wise correlation map used to identify regions with correlated activity for ROI detection."""

    roi_diameter_channel_2: int = 0
    """The estimated ROI diameter for the second imaging channel in pixels. Computed independently because channel 2
    may label a different ROI population with different soma sizes."""

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

    def release_arrays(self) -> None:
        """Releases all array fields to free memory.

        Use ``memory_map_arrays()`` or ``load_arrays()`` to re-acquire the data on demand.
        """
        self.mean_image = None
        self.enhanced_mean_image = None
        self.maximum_projection = None
        self.correlation_map = None
        self.mean_image_channel_2 = None
        self.enhanced_mean_image_channel_2 = None
        self.maximum_projection_channel_2 = None
        self.correlation_map_channel_2 = None

    def save_arrays(self, output_path: Path) -> None:
        """Saves detection arrays as individual .npy files inside a ``detection_data/`` subdirectory.

        Args:
            output_path: The directory where to create the ``detection_data/`` subdirectory.
        """
        detection_directory = output_path / "detection_data"
        ensure_directory_exists(detection_directory)

        # Channel 1 arrays.
        if self.mean_image is not None and not is_memory_mapped(self.mean_image):
            np.save(detection_directory / "mean_image.npy", self.mean_image)
        if self.enhanced_mean_image is not None and not is_memory_mapped(self.enhanced_mean_image):
            np.save(detection_directory / "enhanced_mean_image.npy", self.enhanced_mean_image)
        if self.maximum_projection is not None and not is_memory_mapped(self.maximum_projection):
            np.save(detection_directory / "maximum_projection.npy", self.maximum_projection)
        if self.correlation_map is not None and not is_memory_mapped(self.correlation_map):
            np.save(detection_directory / "correlation_map.npy", self.correlation_map)

        # Channel 2 arrays.
        if self.mean_image_channel_2 is not None and not is_memory_mapped(self.mean_image_channel_2):
            np.save(detection_directory / "mean_image_channel_2.npy", self.mean_image_channel_2)
        if self.enhanced_mean_image_channel_2 is not None and not is_memory_mapped(self.enhanced_mean_image_channel_2):
            np.save(detection_directory / "enhanced_mean_image_channel_2.npy", self.enhanced_mean_image_channel_2)
        if self.maximum_projection_channel_2 is not None and not is_memory_mapped(self.maximum_projection_channel_2):
            np.save(detection_directory / "maximum_projection_channel_2.npy", self.maximum_projection_channel_2)
        if self.correlation_map_channel_2 is not None and not is_memory_mapped(self.correlation_map_channel_2):
            np.save(detection_directory / "correlation_map_channel_2.npy", self.correlation_map_channel_2)

    def load_arrays(self, output_path: Path) -> None:
        """Loads detection arrays from individual .npy files in the ``detection_data/`` subdirectory.

        Args:
            output_path: The directory containing the ``detection_data/`` subdirectory.
        """
        detection_directory = output_path / "detection_data"
        if not detection_directory.exists():
            return

        # Channel 1 arrays.
        path = detection_directory / "mean_image.npy"
        if path.exists():
            self.mean_image = np.load(path, allow_pickle=False).astype(np.float32)
        path = detection_directory / "enhanced_mean_image.npy"
        if path.exists():
            self.enhanced_mean_image = np.load(path, allow_pickle=False).astype(np.float32)
        path = detection_directory / "maximum_projection.npy"
        if path.exists():
            self.maximum_projection = np.load(path, allow_pickle=False).astype(np.float32)
        path = detection_directory / "correlation_map.npy"
        if path.exists():
            self.correlation_map = np.load(path, allow_pickle=False).astype(np.float32)

        # Channel 2 arrays.
        path = detection_directory / "mean_image_channel_2.npy"
        if path.exists():
            self.mean_image_channel_2 = np.load(path, allow_pickle=False).astype(np.float32)
        path = detection_directory / "enhanced_mean_image_channel_2.npy"
        if path.exists():
            self.enhanced_mean_image_channel_2 = np.load(path, allow_pickle=False).astype(np.float32)
        path = detection_directory / "maximum_projection_channel_2.npy"
        if path.exists():
            self.maximum_projection_channel_2 = np.load(path, allow_pickle=False).astype(np.float32)
        path = detection_directory / "correlation_map_channel_2.npy"
        if path.exists():
            self.correlation_map_channel_2 = np.load(path, allow_pickle=False).astype(np.float32)

    def memory_map_arrays(self, output_path: Path) -> None:
        """Memory-maps detection arrays from individual .npy files in the ``detection_data/`` subdirectory.

        Uses ``r+`` mode to allow both reading and writing through the memory-mapped arrays. This avoids loading the
        full array contents into memory, which is useful when reusing previously-generated data (e.g., single-recording
        outputs consumed by the multi-recording pipeline).

        Args:
            output_path: The directory containing the ``detection_data/`` subdirectory.
        """
        detection_directory = output_path / "detection_data"
        if not detection_directory.exists():
            return

        # Channel 1 arrays.
        path = detection_directory / "mean_image.npy"
        if path.exists():
            self.mean_image = np.load(path, mmap_mode="r+")
        path = detection_directory / "enhanced_mean_image.npy"
        if path.exists():
            self.enhanced_mean_image = np.load(path, mmap_mode="r+")
        path = detection_directory / "maximum_projection.npy"
        if path.exists():
            self.maximum_projection = np.load(path, mmap_mode="r+")
        path = detection_directory / "correlation_map.npy"
        if path.exists():
            self.correlation_map = np.load(path, mmap_mode="r+")

        # Channel 2 arrays.
        path = detection_directory / "mean_image_channel_2.npy"
        if path.exists():
            self.mean_image_channel_2 = np.load(path, mmap_mode="r+")
        path = detection_directory / "enhanced_mean_image_channel_2.npy"
        if path.exists():
            self.enhanced_mean_image_channel_2 = np.load(path, mmap_mode="r+")
        path = detection_directory / "maximum_projection_channel_2.npy"
        if path.exists():
            self.maximum_projection_channel_2 = np.load(path, mmap_mode="r+")
        path = detection_directory / "correlation_map_channel_2.npy"
        if path.exists():
            self.correlation_map_channel_2 = np.load(path, mmap_mode="r+")


@dataclass
class ROIMask:
    """Lightweight spatial ROI data for pipeline processing.

    Stores pixel coordinates, weights, and tracking metadata. Used as the working type throughout the multi-recording
    pipeline and as the on-disk format for spatial data in both single-recording and multi-recording outputs.
    """

    y_pixels: NDArray[np.int32]
    """The y-coordinates (row indices) of all pixels belonging to this ROI."""

    x_pixels: NDArray[np.int32]
    """The x-coordinates (column indices) of all pixels belonging to this ROI."""

    pixel_weights: NDArray[np.float32]
    """The spatial filter weights (lambda values) for each pixel, indicating contribution to the ROI signal."""

    centroid: tuple[int, int]
    """The median (y, x) pixel position of the ROI, representing its approximate center."""

    frame_width: int
    """The width of the image frame in pixels, used to compute raveled pixel indices."""

    radius: float = 0.0
    """The fitted ellipse radius representing the approximate ROI size."""

    cluster_id: int = 0
    """The multi-recording ROI cluster ID. Zero indicates unclustered, positive values indicate cluster membership."""

    recording_count: int = 0
    """The number of recordings in which this ROI was detected during multi-recording tracking."""

    overlap_mask: NDArray[np.bool_] | None = None
    """The boolean mask indicating which pixels overlap with other ROIs. Transient; not persisted."""

    @cached_property
    def raveled_pixels(self) -> NDArray[np.int32]:
        """Computes raveled pixel indices (y * frame_width + x) on first access."""
        return (self.y_pixels * self.frame_width + self.x_pixels).astype(np.int32)

    @cached_property
    def circle_pixels(self) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
        """Computes unclipped (y_circle, x_circle) pixel coordinates of a circle with ``1.25 * radius`` and
        100 sample points around the ROI centroid.
        """
        scaled_radius = self.radius * 1.25
        theta = np.linspace(start=0.0, stop=2 * np.pi, num=100)
        y_circle = (scaled_radius * np.sin(theta) + self.centroid[0]).astype(np.int32)
        x_circle = (scaled_radius * np.cos(theta) + self.centroid[1]).astype(np.int32)
        return y_circle, x_circle

    @staticmethod
    def save_list(mask_list: list[ROIMask], file_path: Path) -> None:
        """Saves a list of ROIMask instances to a compressed .npz file without pickle.

        Args:
            mask_list: The list of ROIMask instances to save.
            file_path: The path to the output .npz file.
        """
        if not mask_list:
            return

        pixel_counts = np.array([len(mask.y_pixels) for mask in mask_list], dtype=np.uint32)
        all_y_pixels = np.concatenate([mask.y_pixels for mask in mask_list])
        all_x_pixels = np.concatenate([mask.x_pixels for mask in mask_list])
        all_pixel_weights = np.concatenate([mask.pixel_weights for mask in mask_list])

        centroids = np.array([mask.centroid for mask in mask_list], dtype=np.int32)
        radius = np.array([mask.radius for mask in mask_list], dtype=np.float32)
        cluster_id = np.array([mask.cluster_id for mask in mask_list], dtype=np.uint32)
        recording_count = np.array([mask.recording_count for mask in mask_list], dtype=np.uint16)
        frame_width = np.array([mask_list[0].frame_width], dtype=np.uint32)

        np.savez(
            file_path,
            allow_pickle=False,
            pixel_counts=pixel_counts,
            y_pixels=all_y_pixels,
            x_pixels=all_x_pixels,
            pixel_weights=all_pixel_weights,
            centroids=centroids,
            radius=radius,
            cluster_id=cluster_id,
            recording_count=recording_count,
            frame_width=frame_width,
        )

    @staticmethod
    def load_list(file_path: Path) -> list[ROIMask]:
        """Loads a list of ROIMask instances from a compressed .npz file.

        Args:
            file_path: The path to the .npz file containing the serialized ROI masks.

        Returns:
            A list of ROIMask instances reconstructed from the file.
        """
        data = np.load(file_path, allow_pickle=False)

        pixel_counts = data["pixel_counts"]
        roi_count = len(pixel_counts)
        pixel_splits = np.cumsum(pixel_counts)[:-1]

        y_pixels_list = np.split(data["y_pixels"], pixel_splits)
        x_pixels_list = np.split(data["x_pixels"], pixel_splits)
        pixel_weights_list = np.split(data["pixel_weights"], pixel_splits)

        centroids = data["centroids"]
        radius = data["radius"]
        cluster_id = data["cluster_id"]
        recording_count = data["recording_count"]
        frame_width = int(data["frame_width"][0])

        mask_list: list[ROIMask] = []
        for i in range(roi_count):
            mask = ROIMask(
                y_pixels=y_pixels_list[i].astype(np.int32),
                x_pixels=x_pixels_list[i].astype(np.int32),
                pixel_weights=pixel_weights_list[i].astype(np.float32),
                centroid=(int(centroids[i, 0]), int(centroids[i, 1])),
                frame_width=frame_width,
                radius=float(radius[i]),
                cluster_id=int(cluster_id[i]),
                recording_count=int(recording_count[i]),
            )
            mask_list.append(mask)

        return mask_list


@dataclass(slots=True)
class ROIStatistics:
    """Stores spatial and statistical properties for a single region of interest (ROI).

    This dataclass represents the complete set of properties computed for each detected ROI during the detection,
    extraction, and optional multi-recording processing stages. The fields are organized into required core properties
    (always present after detection), shape statistics (computed during ROI detection, with defaults for staged
    construction), optional extraction properties (added during signal extraction), and
    multi-plane/multi-recording properties.

    Notes:
        This dataclass replaces the legacy dictionary-based stat.npy format. Shape statistics fields have default
        values to support staged construction where ROIStatistics is first created during detection with only core
        fields, then updated with computed shape statistics.
    """

    # Spatial data composed from ROIMask.
    mask: ROIMask
    """The underlying ROIMask containing pixel coordinates, weights, centroid, frame dimensions, and tracking
    metadata."""

    footprint: int = 0
    """The spatial scale (hop size) used during sparse detection for this ROI."""

    # Shape statistics (computed during ROI detection, with defaults for staged construction).
    compactness: float = 0.0
    """The ratio of actual to expected mean radius, where values near 1 indicate compact circular ROIs."""

    solidity: float = 0.0
    """The ratio of soma pixels to convex hull area, measuring how solid/filled the ROI is."""

    pixel_count: int = 0
    """The total number of pixels in the complete ROI."""

    soma_mask: NDArray[np.bool_] | None = None
    """The boolean mask indicating which pixels belong to the soma region."""

    aspect_ratio: float = 0.0
    """The ratio of ellipse axes, indicating ROI elongation."""

    normalized_pixel_count: float = 0.0
    """The pixel count normalized by expected ROI size (soma region only)."""

    # Optional extraction data (added during signal extraction).
    skewness: float | None = None
    """The skewness of the baseline-subtracted fluorescence time series."""

    neuropil_mask: NDArray[np.int32] | None = None
    """The raveled (flattened) pixel indices used for neuropil signal extraction. Each index refers to a pixel position
    in the row-major flattened representation of the imaging plane (height * width). Use ``np.unravel_index`` with the
    plane dimensions to recover 2D coordinates if needed."""

    # Multi-plane data. The plane_index should be set from IOData.plane_index during ROI creation.
    plane_index: int = 0
    """The index of the imaging plane this ROI belongs to. This field is not set during detection. It is populated
    by the IO layer during multi-plane combination, when ROIs from individual planes are merged into a single list."""

    @staticmethod
    def save_list(roi_list: list[ROIStatistics], masks_path: Path, stats_path: Path) -> None:
        """Saves a list of ROIStatistics instances to two companion .npz files without pickle.

        Spatial pixel data (coordinates, weights, centroid) is delegated to ``ROIMask.save_list`` and written to
        ``masks_path``. Shape statistics and extraction statistics are written to
        ``stats_path``.

        Args:
            roi_list: The list of ROIStatistics instances to save.
            masks_path: The path to the output masks .npz file (spatial data).
            stats_path: The path to the output statistics .npz file (shape and extraction data).
        """
        if not roi_list:
            return

        # Delegates spatial core to ROIMask.save_list.
        ROIMask.save_list([roi.mask for roi in roi_list], masks_path)

        # Stores scalar statistics fields.
        footprints = np.array([roi.footprint for roi in roi_list], dtype=np.uint16)
        compactness = np.array([roi.compactness for roi in roi_list], dtype=np.float32)
        solidity = np.array([roi.solidity for roi in roi_list], dtype=np.float32)
        pixel_count = np.array([roi.pixel_count for roi in roi_list], dtype=np.uint32)
        aspect_ratio = np.array([roi.aspect_ratio for roi in roi_list], dtype=np.float32)
        normalized_pixel_count = np.array([roi.normalized_pixel_count for roi in roi_list], dtype=np.float32)

        skewness = np.array(
            [roi.skewness if roi.skewness is not None else np.nan for roi in roi_list], dtype=np.float32
        )

        plane_index = np.array([roi.plane_index for roi in roi_list], dtype=np.int32)

        save_dict: dict[
            str, NDArray[np.float32] | NDArray[np.int32] | NDArray[np.uint16] | NDArray[np.uint32] | NDArray[np.bool_]
        ] = {
            "footprints": footprints,
            "compactness": compactness,
            "solidity": solidity,
            "pixel_count": pixel_count,
            "aspect_ratio": aspect_ratio,
            "normalized_pixel_count": normalized_pixel_count,
            "skewness": skewness,
            "plane_index": plane_index,
        }

        _save_optional_array_field("soma_mask", [roi.soma_mask for roi in roi_list], save_dict, dtype=np.bool_)
        _save_optional_array_field(
            "overlap_mask", [roi.mask.overlap_mask for roi in roi_list], save_dict, dtype=np.bool_
        )
        _save_optional_array_field("neuropil_mask", [roi.neuropil_mask for roi in roi_list], save_dict, dtype=np.int32)
        np.savez(stats_path, allow_pickle=False, **save_dict)

    @staticmethod
    def load_list(masks_path: Path, stats_path: Path) -> list[ROIStatistics]:
        """Loads a list of ROIStatistics instances from companion masks and stats .npz files.

        Args:
            masks_path: The path to the masks .npz file containing spatial pixel data.
            stats_path: The path to the statistics .npz file containing shape and extraction data.

        Returns:
            A list of ROIStatistics instances with pixel data from the masks file and statistics from the stats file.
        """
        masks = ROIMask.load_list(masks_path)
        data = np.load(stats_path, allow_pickle=False)

        roi_count = len(masks)

        footprints = data["footprints"]
        compactness = data["compactness"]
        solidity = data["solidity"]
        pixel_count = data["pixel_count"]
        aspect_ratio = data["aspect_ratio"]
        normalized_pixel_count = data["normalized_pixel_count"]
        skewness = data["skewness"]
        plane_index = data["plane_index"]

        soma_mask_list = _load_optional_array_field("soma_mask", roi_count, data, dtype=np.bool_)
        overlap_mask_list = _load_optional_array_field("overlap_mask", roi_count, data, dtype=np.bool_)
        neuropil_mask_list = _load_optional_array_field("neuropil_mask", roi_count, data, dtype=np.int32)

        roi_list: list[ROIStatistics] = []
        for i in range(roi_count):
            masks[i].overlap_mask = overlap_mask_list[i]  # type: ignore[assignment]
            roi = ROIStatistics(
                mask=masks[i],
                footprint=int(footprints[i]),
                compactness=float(compactness[i]),
                solidity=float(solidity[i]),
                pixel_count=int(pixel_count[i]),
                soma_mask=soma_mask_list[i],  # type: ignore[arg-type]
                aspect_ratio=float(aspect_ratio[i]),
                normalized_pixel_count=float(normalized_pixel_count[i]),
                skewness=None if np.isnan(skewness[i]) else float(skewness[i]),
                neuropil_mask=neuropil_mask_list[i],  # type: ignore[arg-type]
                plane_index=int(plane_index[i]),
            )
            roi_list.append(roi)

        return roi_list


@dataclass(slots=True)
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
    """The cell classification results with shape (cells, 2) containing (is_cell_label, probability)."""

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
    containing (is_colocalized_boolean, probability)."""

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

    def release_arrays(self) -> None:
        """Releases all array and list fields to free memory.

        Use ``memory_map_arrays()`` or ``load_arrays()`` to re-acquire the data on demand.
        """
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
        # Channel 1 ROI statistics (split into masks + stats files).
        if self.roi_statistics is not None:
            ROIStatistics.save_list(
                self.roi_statistics,
                masks_path=output_path / "roi_masks.npz",
                stats_path=output_path / "roi_statistics.npz",
            )

        # Channel 1 trace arrays.
        if self.cell_fluorescence is not None and not is_memory_mapped(self.cell_fluorescence):
            np.save(output_path / "cell_fluorescence.npy", self.cell_fluorescence, allow_pickle=False)
        if self.neuropil_fluorescence is not None and not is_memory_mapped(self.neuropil_fluorescence):
            np.save(output_path / "neuropil_fluorescence.npy", self.neuropil_fluorescence, allow_pickle=False)
        if self.subtracted_fluorescence is not None and not is_memory_mapped(self.subtracted_fluorescence):
            np.save(output_path / "subtracted_fluorescence.npy", self.subtracted_fluorescence, allow_pickle=False)
        if self.spikes is not None and not is_memory_mapped(self.spikes):
            np.save(output_path / "spikes.npy", self.spikes, allow_pickle=False)
        if self.cell_classification is not None and not is_memory_mapped(self.cell_classification):
            np.save(output_path / "cell_classification.npy", self.cell_classification, allow_pickle=False)

        # Channel 2 ROI statistics (split into masks + stats files).
        if self.roi_statistics_channel_2 is not None:
            ROIStatistics.save_list(
                self.roi_statistics_channel_2,
                masks_path=output_path / "roi_masks_channel_2.npz",
                stats_path=output_path / "roi_statistics_channel_2.npz",
            )

        # Channel 2 trace arrays.
        if self.cell_fluorescence_channel_2 is not None and not is_memory_mapped(self.cell_fluorescence_channel_2):
            np.save(
                output_path / "cell_fluorescence_channel_2.npy", self.cell_fluorescence_channel_2, allow_pickle=False
            )
        if self.neuropil_fluorescence_channel_2 is not None and not is_memory_mapped(
            self.neuropil_fluorescence_channel_2
        ):
            np.save(
                output_path / "neuropil_fluorescence_channel_2.npy",
                self.neuropil_fluorescence_channel_2,
                allow_pickle=False,
            )
        if self.subtracted_fluorescence_channel_2 is not None and not is_memory_mapped(
            self.subtracted_fluorescence_channel_2
        ):
            np.save(
                output_path / "subtracted_fluorescence_channel_2.npy",
                self.subtracted_fluorescence_channel_2,
                allow_pickle=False,
            )
        if self.spikes_channel_2 is not None and not is_memory_mapped(self.spikes_channel_2):
            np.save(output_path / "spikes_channel_2.npy", self.spikes_channel_2, allow_pickle=False)
        if self.cell_classification_channel_2 is not None and not is_memory_mapped(self.cell_classification_channel_2):
            np.save(
                output_path / "cell_classification_channel_2.npy",
                self.cell_classification_channel_2,
                allow_pickle=False,
            )

        # Colocalization arrays.
        if self.cell_colocalization is not None and not is_memory_mapped(self.cell_colocalization):
            np.save(output_path / "cell_colocalization.npy", self.cell_colocalization, allow_pickle=False)
        if self.corrected_structural_mean_image is not None and not is_memory_mapped(
            self.corrected_structural_mean_image
        ):
            np.save(
                output_path / "corrected_structural_mean_image.npy",
                self.corrected_structural_mean_image,
                allow_pickle=False,
            )

    def load_arrays(self, output_path: Path) -> None:
        """Loads ROI statistics and classification results from disk.

        This method loads only ROI statistics and cell classification arrays, which are the extraction data
        needed during pipeline processing (specifically for multi-recording ROI selection and tracking).
        Fluorescence traces and colocalization data are not loaded because they are never needed during
        pipeline execution and consume
        significant memory. Use load_results() to load all result arrays when needed for analysis or visualization.

        Args:
            output_path: The directory containing the extraction data files.
        """
        # Channel 1 ROI statistics (loaded from companion masks + stats files).
        roi_masks_path = output_path / "roi_masks.npz"
        roi_stats_path = output_path / "roi_statistics.npz"
        if self.roi_statistics is None and roi_masks_path.exists() and roi_stats_path.exists():
            self.roi_statistics = ROIStatistics.load_list(masks_path=roi_masks_path, stats_path=roi_stats_path)

        # Channel 2 ROI statistics (loaded from companion masks + stats files).
        roi_masks_channel_2_path = output_path / "roi_masks_channel_2.npz"
        roi_stats_channel_2_path = output_path / "roi_statistics_channel_2.npz"
        if (
            self.roi_statistics_channel_2 is None
            and roi_masks_channel_2_path.exists()
            and roi_stats_channel_2_path.exists()
        ):
            self.roi_statistics_channel_2 = ROIStatistics.load_list(
                masks_path=roi_masks_channel_2_path, stats_path=roi_stats_channel_2_path
            )

        # Channel 1 classification.
        cell_classification_path = output_path / "cell_classification.npy"
        if self.cell_classification is None and cell_classification_path.exists():
            self.cell_classification = np.load(cell_classification_path, allow_pickle=False).astype(np.float32)

        # Channel 2 classification.
        cell_classification_channel_2_path = output_path / "cell_classification_channel_2.npy"
        if self.cell_classification_channel_2 is None and cell_classification_channel_2_path.exists():
            self.cell_classification_channel_2 = np.load(cell_classification_channel_2_path, allow_pickle=False).astype(
                np.float32
            )

    def load_results(self, output_path: Path) -> None:
        """Loads all extraction result arrays from disk.

        This method loads fluorescence traces, classification results, and colocalization data. Classification arrays
        may already be loaded by load_arrays() (which loads them for multi-recording pipeline use), in which case the
        guarded loading here is a no-op. Fluorescence traces and colocalization data are not loaded by load_arrays()
        because they consume significant memory and are not needed during pipeline execution. Call this method when
        result data is needed for analysis or visualization.

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

    def memory_map_arrays(self, output_path: Path) -> None:
        """Memory-maps ROI statistics and classification results from disk.

        This method mirrors load_arrays() but uses ``r+`` memory mapping for .npy files instead of eager loading.
        ROI statistics (.npz) are still eagerly loaded because NumPy does not support memory mapping for .npz archives.

        Args:
            output_path: The directory containing the extraction data files.
        """
        # Channel 1 ROI statistics (eagerly loaded from companion masks + stats files; .npz cannot be memory-mapped).
        roi_masks_path = output_path / "roi_masks.npz"
        roi_stats_path = output_path / "roi_statistics.npz"
        if self.roi_statistics is None and roi_masks_path.exists() and roi_stats_path.exists():
            self.roi_statistics = ROIStatistics.load_list(masks_path=roi_masks_path, stats_path=roi_stats_path)

        # Channel 2 ROI statistics (eagerly loaded from companion masks + stats files; .npz cannot be memory-mapped).
        roi_masks_channel_2_path = output_path / "roi_masks_channel_2.npz"
        roi_stats_channel_2_path = output_path / "roi_statistics_channel_2.npz"
        if (
            self.roi_statistics_channel_2 is None
            and roi_masks_channel_2_path.exists()
            and roi_stats_channel_2_path.exists()
        ):
            self.roi_statistics_channel_2 = ROIStatistics.load_list(
                masks_path=roi_masks_channel_2_path, stats_path=roi_stats_channel_2_path
            )

        # Channel 1 classification.
        cell_classification_path = output_path / "cell_classification.npy"
        if self.cell_classification is None and cell_classification_path.exists():
            self.cell_classification = np.load(cell_classification_path, mmap_mode="r+")

        # Channel 2 classification.
        cell_classification_channel_2_path = output_path / "cell_classification_channel_2.npy"
        if self.cell_classification_channel_2 is None and cell_classification_channel_2_path.exists():
            self.cell_classification_channel_2 = np.load(cell_classification_channel_2_path, mmap_mode="r+")

    def memory_map_results(self, output_path: Path) -> None:
        """Memory-maps all extraction result arrays from disk.

        This method mirrors load_results() but uses ``r+`` memory mapping for all .npy files instead of eager loading.
        This avoids loading the full array contents into memory, which is useful when reusing previously-generated data
        (e.g., single-recording outputs consumed by the multi-recording pipeline).

        Args:
            output_path: The directory containing the result .npy files.
        """
        # Channel 1 traces.
        cell_fluorescence_path = output_path / "cell_fluorescence.npy"
        if self.cell_fluorescence is None and cell_fluorescence_path.exists():
            self.cell_fluorescence = np.load(cell_fluorescence_path, mmap_mode="r+")

        neuropil_fluorescence_path = output_path / "neuropil_fluorescence.npy"
        if self.neuropil_fluorescence is None and neuropil_fluorescence_path.exists():
            self.neuropil_fluorescence = np.load(neuropil_fluorescence_path, mmap_mode="r+")

        subtracted_fluorescence_path = output_path / "subtracted_fluorescence.npy"
        if self.subtracted_fluorescence is None and subtracted_fluorescence_path.exists():
            self.subtracted_fluorescence = np.load(subtracted_fluorescence_path, mmap_mode="r+")

        spikes_path = output_path / "spikes.npy"
        if self.spikes is None and spikes_path.exists():
            self.spikes = np.load(spikes_path, mmap_mode="r+")

        # Channel 1 classification.
        cell_classification_path = output_path / "cell_classification.npy"
        if self.cell_classification is None and cell_classification_path.exists():
            self.cell_classification = np.load(cell_classification_path, mmap_mode="r+")

        # Channel 2 traces.
        cell_fluorescence_channel_2_path = output_path / "cell_fluorescence_channel_2.npy"
        if self.cell_fluorescence_channel_2 is None and cell_fluorescence_channel_2_path.exists():
            self.cell_fluorescence_channel_2 = np.load(cell_fluorescence_channel_2_path, mmap_mode="r+")

        neuropil_fluorescence_channel_2_path = output_path / "neuropil_fluorescence_channel_2.npy"
        if self.neuropil_fluorescence_channel_2 is None and neuropil_fluorescence_channel_2_path.exists():
            self.neuropil_fluorescence_channel_2 = np.load(neuropil_fluorescence_channel_2_path, mmap_mode="r+")

        subtracted_fluorescence_channel_2_path = output_path / "subtracted_fluorescence_channel_2.npy"
        if self.subtracted_fluorescence_channel_2 is None and subtracted_fluorescence_channel_2_path.exists():
            self.subtracted_fluorescence_channel_2 = np.load(subtracted_fluorescence_channel_2_path, mmap_mode="r+")

        spikes_channel_2_path = output_path / "spikes_channel_2.npy"
        if self.spikes_channel_2 is None and spikes_channel_2_path.exists():
            self.spikes_channel_2 = np.load(spikes_channel_2_path, mmap_mode="r+")

        # Channel 2 classification.
        cell_classification_channel_2_path = output_path / "cell_classification_channel_2.npy"
        if self.cell_classification_channel_2 is None and cell_classification_channel_2_path.exists():
            self.cell_classification_channel_2 = np.load(cell_classification_channel_2_path, mmap_mode="r+")

        # Colocalization arrays.
        cell_colocalization_path = output_path / "cell_colocalization.npy"
        if self.cell_colocalization is None and cell_colocalization_path.exists():
            self.cell_colocalization = np.load(cell_colocalization_path, mmap_mode="r+")

        corrected_structural_mean_image_path = output_path / "corrected_structural_mean_image.npy"
        if self.corrected_structural_mean_image is None and corrected_structural_mean_image_path.exists():
            self.corrected_structural_mean_image = np.load(corrected_structural_mean_image_path, mmap_mode="r+")


@dataclass(slots=True)
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
class SingleRecordingRuntimeData(YamlConfig):
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

    def release_arrays(self) -> None:
        """Releases all array fields across registration, detection, and extraction to free memory.

        Delegates to the ``release_arrays()`` method on each child dataclass. Scalar fields are preserved.
        """
        self.registration.release_arrays()
        self.detection.release_arrays()
        self.extraction.release_arrays()

    def load_arrays(self) -> None:
        """Eagerly loads all NumPy arrays from .npy files on disk into memory.

        This method reads each array file in full and copies it into a contiguous in-memory buffer. Use this when the
        data will be generated or modified during pipeline processing.
        """
        if self.output_path is None:
            return
        self.registration.load_arrays(self.output_path)
        self.detection.load_arrays(self.output_path)
        self.extraction.load_arrays(self.output_path)

    def memory_map_arrays(self) -> None:
        """Memory-maps all NumPy arrays from .npy files on disk in ``r+`` mode.

        This method opens each .npy file as a read-write memory-mapped array, avoiding full materialization in RAM.
        Use this when reusing previously-generated data that does not need to be copied into memory (e.g.,
        single-recording outputs consumed by the multi-recording pipeline).
        """
        if self.output_path is None:
            return
        self.registration.memory_map_arrays(self.output_path)
        self.detection.memory_map_arrays(self.output_path)
        self.extraction.memory_map_arrays(self.output_path)

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

        # Creates a shallow copy for YAML serialization. Child dataclasses are shallow-copied individually so that
        # prepare_for_saving() nulls array fields on the copies without affecting the originals in memory.
        yaml_copy = copy.copy(self)
        yaml_copy.registration = copy.copy(self.registration)
        yaml_copy.detection = copy.copy(self.detection)
        yaml_copy.extraction = copy.copy(self.extraction)
        yaml_copy.io = copy.copy(self.io)
        yaml_copy.timing = copy.copy(self.timing)

        # Nulls array fields in child dataclasses for YAML serialization.
        yaml_copy.registration.prepare_for_saving()
        yaml_copy.detection.prepare_for_saving()
        yaml_copy.extraction.prepare_for_saving()

        # Saves the YAML file.
        file_path = output_path / "runtime_data.yaml"
        yaml_copy.to_yaml(file_path=file_path)

    @classmethod
    def load(cls, output_path: Path) -> SingleRecordingRuntimeData:
        """Deserializes runtime data from a YAML file without loading any NumPy arrays.

        After calling this method, arrays can be loaded individually per-child dataclass using the ``load_arrays()``
        or ``memory_map_arrays()`` methods on each child (registration, detection, extraction). Alternatively, the
        convenience ``load_arrays()`` / ``memory_map_arrays()`` methods on this class load all children at once.

        Args:
            output_path: The directory containing the runtime_data.yaml file.

        Returns:
            A SingleRecordingRuntimeData instance with all scalar fields deserialized. NumPy array fields
            remain None until explicitly loaded.
        """
        file_path = output_path / "runtime_data.yaml"
        return cls.from_yaml(file_path=file_path)


@dataclass(slots=True)
class CombinedData:
    """Stores combined multi-plane detection and extraction data.

    This class provides a container for the results of combining processed data from multiple imaging planes
    into a unified dataset. It holds DetectionData (combined images) and ExtractionData (combined ROI statistics,
    fluorescence traces, and classification results) along with metadata about the combined field of view.

    Notes:
        Combined data is saved to the root cindra directory alongside configuration.yaml and
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
    """The timescale of the calcium indicator sensor in seconds, cached from the single-recording
    configuration for use by the multi-recording extraction pipeline."""

    sampling_rate: float = 0.0
    """The per-plane sampling rate in Hertz, cached from the single-recording runtime for use by the
    multi-recording extraction pipeline."""

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
        """Saves combined data to the root cindra directory.

        This method saves all combined detection and extraction arrays to the root cindra directory. Metadata
        (plane count, dimensions) is saved to combined_metadata.npz.

        Args:
            root_path: The root cindra output directory containing configuration.yaml.
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
        """Loads combined metadata from the root cindra directory without loading any arrays.

        After calling this method, detection and extraction arrays can be loaded individually using the
        ``load_arrays()`` or ``memory_map_arrays()`` methods on each child (e.g., ``combined.detection.load_arrays(
        root_path)``).

        Args:
            root_path: The root cindra output directory containing combined_metadata.npz.

        Returns:
            A CombinedData instance with metadata loaded and empty detection/extraction containers. NumPy array
            fields remain None until explicitly loaded on the child dataclasses.

        Raises:
            FileNotFoundError: If the combined metadata file does not exist.
        """
        kwargs = cls._load_metadata(root_path)
        return cls(detection=DetectionData(), extraction=ExtractionData(), **kwargs)

    @classmethod
    def _load_metadata(cls, root_path: Path) -> dict[str, Any]:
        """Loads combined metadata from the .npz file and returns constructor keyword arguments.

        This private helper extracts all metadata fields from combined_metadata.npz and returns them as a dictionary
        suitable for passing to the CombinedData constructor. Detection and extraction fields are not included; they
        must be loaded separately by the calling classmethod using the appropriate loading strategy.

        Args:
            root_path: The root cindra output directory containing combined_metadata.npz.

        Returns:
            A dictionary of keyword arguments for CombinedData construction (excludes detection and extraction).
        """
        metadata_path = root_path / "combined_metadata.npz"
        if not metadata_path.exists():
            message = (
                f"Unable to load combined data. The combined metadata file does not exist at the specified path: "
                f"{metadata_path}."
            )
            console.error(message=message, error=FileNotFoundError)

        metadata = np.load(metadata_path, allow_pickle=False)

        kwargs: dict[str, Any] = {
            "plane_count": int(metadata["plane_count"][0]),
            "combined_height": int(metadata["combined_height"][0]),
            "combined_width": int(metadata["combined_width"][0]),
            "tau": float(metadata["tau"][0]),
            "sampling_rate": float(metadata["sampling_rate"][0]),
            "plane_heights": np.array([], dtype=np.uint16),
            "plane_widths": np.array([], dtype=np.uint16),
            "plane_y_offsets": np.array([], dtype=np.int32),
            "plane_x_offsets": np.array([], dtype=np.int32),
            "registered_binary_paths": (),
            "registered_binary_paths_channel_2": None,
        }

        # Per-plane geometry and binary paths may be absent in metadata files saved before these fields were added.
        if "plane_heights" in metadata:
            kwargs["plane_heights"] = metadata["plane_heights"].astype(np.uint16)
            kwargs["plane_widths"] = metadata["plane_widths"].astype(np.uint16)
            kwargs["plane_y_offsets"] = metadata["plane_y_offsets"].astype(np.int32)
            kwargs["plane_x_offsets"] = metadata["plane_x_offsets"].astype(np.int32)

        if "registered_binary_paths" in metadata:
            kwargs["registered_binary_paths"] = tuple(root_path / str(p) for p in metadata["registered_binary_paths"])

        if "registered_binary_paths_channel_2" in metadata:
            kwargs["registered_binary_paths_channel_2"] = tuple(
                root_path / str(p) for p in metadata["registered_binary_paths_channel_2"]
            )

        return kwargs


def _save_optional_array_field(
    field_name: str,
    arrays: list[NDArray[np.float32] | NDArray[np.int32] | NDArray[np.bool_] | tuple[int, ...] | None],
    save_dictionary: dict[
        str, NDArray[np.float32] | NDArray[np.int32] | NDArray[np.uint16] | NDArray[np.uint32] | NDArray[np.bool_]
    ],
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
    has_data = [a is not None and len(a) for a in arrays]
    if not any(has_data):
        return

    counts = np.array(object=[len(a) if a is not None else 0 for a in arrays], dtype=np.uint32)
    valid_arrays: list[NDArray[np.float32] | NDArray[np.int32] | NDArray[np.bool_]] = [
        np.asarray(a=a, dtype=dtype) for a in arrays if a is not None and len(a)
    ]
    if valid_arrays:  # pragma: no branch - line 1579's identical early-return predicate keeps valid_arrays non-empty.
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
