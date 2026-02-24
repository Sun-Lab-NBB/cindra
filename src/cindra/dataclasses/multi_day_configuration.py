"""Provides user-defined configuration classes for the multi-day (across-session) processing pipeline."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path  # noqa: TC003 - needed at runtime for dacite deserialization
from dataclasses import field, dataclass

from natsort import natsorted
from ataraxis_base_utilities import ensure_directory_exists
from ataraxis_data_structures import YamlConfig

from .single_day_configuration import RuntimeSettings, SignalExtraction, SpikeDeconvolution


class ReferenceImageType(StrEnum):
    """Defines the supported reference image types for diffeomorphic registration across sessions."""

    MEAN = "mean"
    """The temporal mean of all registered frames, providing a static view of the imaging field."""

    ENHANCED_MEAN = "enhanced_mean"
    """The high-pass filtered mean image that enhances cell boundaries for improved registration."""

    MAXIMUM_PROJECTION = "maximum_projection"
    """The maximum intensity projection across all frames, highlighting active structures."""


@dataclass
class SessionIO:
    """Stores the parameters that specify input session locations and output directories."""

    session_directories: tuple[Path, ...] = ()
    """Specifies the sessions to register across days as absolute paths to their root directories.
    Sessions are natural-sorted, and the first session after sorting becomes the 'main session' which stores
    the processing tracker file. Each session directory is expected to contain the combined_metadata.npz file created
    by the single-day processing pipeline."""

    dataset_name: str = ""
    """Specifies the name of the multiday dataset. This name is used to create the output directory under each
    session's 'multiday' directory (e.g., session/multiday/{dataset_name}/) and to identify the dataset in the
    tracker file."""

    repeat_selection: bool = False
    """Determines whether to repeat the cell selection step when processing. When True, the pipeline re-runs cell 
    selection filtering using the current ROI selection parameters, even if selected cells already exist. This allows 
    updated single-day results or modified selection criteria to be integrated into multi-day processing. When False 
    (default), existing cell selections are used if present."""

    def __post_init__(self) -> None:
        """Natural-sorts session directories after construction or YAML loading."""
        self.session_directories = tuple(natsorted(self.session_directories))


@dataclass
class ROISelection:
    """Stores parameters for selecting single-day-detected ROIs to be tracked across multiple sessions (days)."""

    probability_threshold: float = 0.85
    """The minimum required cell probability score assigned to the ROI by the single-day cindra classifier. ROIs
    with a lower classifier score are excluded from multi-day processing. This parameter applies to channel 1 ROIs."""

    maximum_size: int = 1000
    """The maximum allowed ROI size, in pixels. ROIs with a larger pixel size are excluded from processing. This
    parameter applies to channel 1 ROIs."""

    mroi_region_margin: int = 30
    """The minimum required distance, in pixels, between the center-point (the median x-coordinate) of the ROI
    and the MROI region border. ROIs that are too close to region borders are excluded from processing to avoid
    ambiguities associated with tracking ROIs that span multiple regions. This parameter is only used for MROI
    recordings where region borders are automatically computed from the acquisition parameters. This parameter applies
    to channel 1 ROIs."""

    probability_threshold_channel_2: float | None = None
    """The minimum required cell probability score for channel 2 ROIs. When set to None (default), channel 2 ROIs use
    the same probability_threshold as channel 1. Set this to a different value when channel 2 cells have different
    classification characteristics."""

    maximum_size_channel_2: int | None = None
    """The maximum allowed ROI size for channel 2, in pixels. When set to None (default), channel 2 ROIs use the same
    maximum_size as channel 1. Set this to a different value when channel 2 cells have different size
    characteristics."""

    mroi_region_margin_channel_2: int | None = None
    """The minimum required distance from MROI region borders for channel 2 ROIs, in pixels. When set to None
    (default), channel 2 ROIs use the same mroi_region_margin as channel 1."""


@dataclass
class DiffeomorphicRegistration:
    """Stores parameters for diffeomorphic demons registration that aligns sessions from multiple days to the same
    visual (sampling) space.
    """

    image_type: ReferenceImageType | str = ReferenceImageType.ENHANCED_MEAN
    """The type of cindra-generated reference image to use for across-day registration. This image is used to
    calculate the deformation fields that register all sessions to a common visual space."""

    grid_sampling_factor: float = 1
    """Determines how the B-spline grid spacing scales with image scale during the multi-scale registration process.
    Must be between 0 and 1. Lower values produce a relatively finer grid at coarser scales, allowing for more
    detailed deformations at those scales."""

    scale_sampling: int = 30
    """The number of registration iterations to perform at each scale level of the multi-scale pyramid. Values between
    20 and 30 are reasonable for most recordings, but higher values yield better alignment at the cost of proportionally
    longer computation time."""

    speed_factor: float = 3
    """The relative force of the deformation transform applied when registering the sessions to the same visual space.
    This is the most important parameter to tune. For most cases, a value between 1 and 5 is reasonable."""

    repeat_registration: bool = False
    """Determines whether to repeat diffeomorphic registration when existing registration data is found. When True,
    the pipeline clears existing deformation fields, transformed images, and deformed cell masks before re-running
    registration. When False (default), existing registration results are reused if present."""


@dataclass
class ROITracking:
    """Stores parameters for tracking ROIs across multiple registered sessions (days) using spatial clustering."""

    threshold: float = 0.75
    """The Jaccard distance threshold for the hierarchical clustering algorithm. Candidate ROI pairs that pass the
    maximum_distance pre-filter are compared by spatial overlap (Jaccard distance, 0 = identical, 1 = no overlap) and
    clustered together as the same ROI if their Jaccard distance is below this value."""

    mask_prevalence: int = 50
    """The minimum percentage of registered sessions that must contain a given ROI for it to be included in the
    tracked cell set. Clusters with members in fewer sessions than this threshold are discarded."""

    pixel_prevalence: int = 50
    """The minimum percentage of registered sessions in which a pixel must appear for it to be included in the ROI's
    cross-session template mask. Pixels below this threshold are excluded, so only spatially stable regions of each
    tracked ROI contribute to the template used for fluorescence extraction across sessions."""

    step_sizes: tuple[int, int] = (200, 200)
    """The block size, in pixels, as (height, width) used to partition the deformed visual space into spatial bins
    for clustering. Smaller blocks reduce memory usage but increase processing overhead."""

    bin_size: int = 50
    """The extension, in pixels, added to each spatial bin boundary in both directions when collecting ROI masks
    for clustering. This overlap between neighboring bins ensures that ROIs near bin borders are clustered
    correctly."""

    maximum_distance: int = 20
    """The maximum centroid distance, in pixels, between two ROI masks for them to be considered a candidate pair.
    Only pairs that pass this spatial pre-filter proceed to the Jaccard overlap comparison controlled by threshold."""

    minimum_size: int = 25
    """The minimum number of non-overlapping pixels a cross-session template mask must contain after removing pixels
    shared with other templates. Templates below this size are discarded as too small to represent a valid cell."""


@dataclass
class MultiDayConfiguration(YamlConfig):
    """Aggregates the user-defined configuration parameters for the multi-day cindra pipeline.

    This class stores all user-configurable parameters that control how the pipeline processes data.
    These parameters are immutable during processing - the pipeline reads them but does not modify them.

    Notes:
        This class is based on the reference implementation here:
        https://github.com/sprustonlab/multiday-suite2p-public.

        For runtime data (computed by the pipeline), see MultiDayRuntimeData.
    """

    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    """Stores runtime behavior settings shared with the single-day pipeline (parallel workers, progress bars)."""
    session_io: SessionIO = field(default_factory=SessionIO)
    """Stores parameters that specify input session locations and output directories."""
    roi_selection: ROISelection = field(default_factory=ROISelection)
    """Stores parameters for selecting single-day-detected ROIs to be tracked across multiple sessions (days)."""
    diffeomorphic_registration: DiffeomorphicRegistration = field(default_factory=DiffeomorphicRegistration)
    """Stores parameters for diffeomorphic demons registration that aligns sessions to the same visual space."""
    roi_tracking: ROITracking = field(default_factory=ROITracking)
    """Stores parameters for tracking ROIs across multiple registered sessions (days) using spatial clustering."""
    signal_extraction: SignalExtraction = field(default_factory=SignalExtraction)
    """Stores parameters for extracting fluorescence signals from ROIs and surrounding neuropil regions of the ROIs
    tracked across days."""
    spike_deconvolution: SpikeDeconvolution = field(default_factory=SpikeDeconvolution)
    """Stores parameters for deconvolving fluorescence signals to infer spike trains."""

    def save(self, file_path: Path) -> None:
        """Saves the configuration to a YAML file.

        Args:
            file_path: The path to the .yaml file where to save the configuration data.
        """
        ensure_directory_exists(file_path)
        self.to_yaml(file_path=file_path)

    @classmethod
    def load(cls, file_path: Path) -> MultiDayConfiguration:
        """Loads configuration from a YAML file.

        Args:
            file_path: The path to the .yaml configuration file.

        Returns:
            A MultiDayConfiguration instance populated with the loaded data.
        """
        return cls.from_yaml(file_path=file_path)
