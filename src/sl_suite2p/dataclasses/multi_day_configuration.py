"""Provides user-defined configuration classes for the multi-day (across-session) processing pipeline."""

from __future__ import annotations

import copy
from pathlib import Path
from functools import cached_property
from dataclasses import field, dataclass

from natsort import natsorted
from ataraxis_base_utilities import console, ensure_directory_exists
from ataraxis_data_structures import YamlConfig

from .single_day_configuration import RuntimeSettings, SignalExtraction, SpikeDeconvolution


def _extract_unique_components(paths: list[Path] | tuple[Path, ...]) -> tuple[str, ...]:
    """Extracts the first component from the end of each input path that uniquely identifies each path globally.

    Notes:
        This function adapts the multi-day pipeline to directory structures where the unique session identifier appears
        at different levels of the path hierarchy. For example, given paths like `/data/day1/session` and
        `/data/day2/session`, the function identifies `day1` and `day2` as the unique components (not `session`, which
        is shared). This allows users to organize sessions using any naming convention, as long as each path contains
        at least one unique component somewhere in its hierarchy.

    Args:
        paths: A list or tuple of Path objects.

    Returns:
        A tuple of unique components, one for each path, stored in the same order as the input paths.

    Raises:
        RuntimeError: If one or more paths do not contain unique components.
    """
    paths_list = list(paths)
    result = []

    for path in paths_list:
        # Gets components from right to left.
        components = list(path.parts)[::-1]
        found_unique = False

        for component in components:
            # Checks if this component appears in any other path.
            is_unique = True

            for other_path in paths_list:
                if path == other_path:
                    continue

                # If the component appears anywhere in the other path, it's not unique.
                if component in other_path.parts:
                    is_unique = False
                    break

            if is_unique:
                result.append(component)
                found_unique = True
                break

        if not found_unique:
            message = f"No unique component found for path: {path}, which is not allowed."
            console.error(message=message, error=RuntimeError)

    return tuple(result)


@dataclass()
class SessionIO:
    """Stores the parameters that specify input session locations and output directories."""

    session_directories: list[Path] = field(default_factory=list)
    """Specifies the sessions to register across days as absolute paths to their root directories.
    Sessions are natural-sorted, and the first session after sorting becomes the 'main session' which stores
    the processing tracker file. Each session directory is expected to contain the combined_metadata.npz file created
    by the single-day processing pipeline."""

    dataset_name: str = ""
    """Specifies the name of the multiday dataset. This name is used to create the output directory under each
    session's 'multiday' directory (e.g., session/multiday/{dataset_name}/) and to identify the dataset in the
    tracker file."""

    def __post_init__(self) -> None:
        """Converts string paths to Path objects and natural-sorts them after YAML loading."""
        self.session_directories = natsorted([
            Path(p) if isinstance(p, str) else p for p in self.session_directories
        ])

    @cached_property
    def session_ids(self) -> tuple[str, ...]:
        """Returns unique session identifiers extracted from session directory paths.

        Uses the first path component from the end that uniquely identifies each session. The result is cached after
        first access.
        """
        return _extract_unique_components(paths=self.session_directories)

    def prepare_for_saving(self) -> None:
        """Converts Path fields to strings for YAML serialization."""
        self.session_directories = [str(p) for p in self.session_directories]  # type: ignore[misc]


@dataclass()
class ROISelection:
    """Stores parameters for selecting single-day-detected ROIs to be tracked across multiple sessions (days)."""

    probability_threshold: float = 0.85
    """The minimum required cell probability score assigned to the ROI by the single-day suite2p classifier. ROIs
    with a lower classifier score are excluded from multi-day processing."""

    maximum_size: int = 1000
    """The maximum allowed ROI size, in pixels. ROIs with a larger pixel size are excluded from processing."""

    mroi_stripe_margin: int = 30
    """The minimum required distance, in pixels, between the center-point (the median x-coordinate) of the ROI
    and the MROI stripe border. ROIs that are too close to stripe borders are excluded from processing to avoid
    ambiguities associated with tracking ROIs that span multiple stripes. This parameter is only used for MROI
    recordings where stripe borders are automatically computed from the acquisition parameters."""


@dataclass()
class DiffeomorphicRegistration:
    """Stores parameters for diffeomorphic demons registration that aligns sessions from multiple days to the same
    visual (sampling) space.
    """

    image_type: str = "enhanced"
    """The type of suite2p-generated reference image to use for across-day registration. Supported options are
    'enhanced', 'mean' and 'max'. This 'template' image is used to calculate the necessary deformation (transformations)
    to register (align) all sessions to the same visual space."""

    grid_sampling_factor: float = 1
    """Determines to what extent the grid sampling scales with the deformed image scale. Has to be between 0 and 1. By
    making this value lower than 1, the grid is relatively fine at the the higher scales, allowing for more
    deformations. This is used when resizing session images as part of the registration process."""

    scale_sampling: int = 30
    """The number of iterations for each level (i.e. between each factor two in scale) to perform when computing the
    deformations. Values between 20 and 30 are reasonable in most situations, but higher values yield better results in
    general. The speed of the algorithm scales linearly with this value."""

    speed_factor: float = 3
    """The relative force of the deformation transform applied when registering the sessions to the same visual space.
    This is the most important parameter to tune. For most cases, a value between 1 and 5 is reasonable."""


@dataclass()
class ROITracking:
    """Stores parameters for tracking ROIs across multiple registered sessions (days) using spatial clustering."""

    criterion: str = "distance"
    """Specifies the criterion for clustering (grouping) ROI masks from different sessions. Currently, the only
    valid option is 'distance'."""

    threshold: float = 0.75
    """Specifies the threshold for the clustering algorithm. ROI masks will be clustered (grouped) together if their
    clustering criterion is below this threshold value."""

    mask_prevalence: int = 50
    """Specifies the minimum percentage of all registered sessions that must include the clustered ROI mask. ROI masks
    present in fewer percent of sessions than this value are excluded from processing. This parameter is used to filter
    out ROIs that are mostly silent or not distinguishable across sessions."""

    pixel_prevalence: int = 50
    """Specifies the minimum percentage of all registered sessions in which an ROI mask pixel must be present for it to
    be used to construct the template mask. Pixels present in fewer percent of sessions than this value are not used to
    define the template masks. Template masks are used to extract the ROI fluorescence from the original (non-deformed)
    visual space of every session. This parameter is used to isolate the part of the ROI that is stable across
    sessions, which is required for the extraction step to work correctly (target only the tracked ROI)."""

    step_sizes: list[int] = field(default_factory=lambda: [200, 200])
    """Specifies the block size for the ROI clustering (across-session tracking) process, in pixels, in the order of
    (height, width). To reduce the memory (RAM) overhead, the algorithm divides the deformed (shared) visual space into
    blocks and then processes one (or more) blocks at a time."""

    bin_size: int = 50
    """Specifies the additional length, in pixels, the algorithm is allowed to extend into the neighboring regions when
    segmenting ROIs into grid bins. Before clustering ROIs across sessions, the algorithms pre-segments them into
    grid bins using 'step_sizes'. Additionally, it uses +- 'bin_size' to extend into neighboring regions to better
    cluster the ROIs around grid borders."""

    maximum_distance: int = 20
    """Specifies the maximum distance, in pixels, that can separate masks across multiple sessions. The clustering
    algorithm will consider ROI masks located at most within this distance from each-other across days as the same
    ROIs during tracking."""

    minimum_size: int = 25
    """The minimum size of the non-overlapping ROI region, in pixels, that has to be covered by the template
    mask, for the ROI to be assigned to that template. This is used to determine which template(s) the ROI belongs to
    (if any), for the purpose of tracking it across sessions."""


@dataclass()
class MultiDayConfiguration(YamlConfig):
    """Aggregates the configuration parameters for the multi-day suite2p pipeline.

    Notes:
        This class is based on the reference implementation here:
        https://github.com/sprustonlab/multiday-suite2p-public.
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

        Converts Path fields to strings before serialization to ensure YAML compatibility.

        Args:
            file_path: The path to the .yaml file where to save the configuration data.
        """
        ensure_directory_exists(file_path)

        # Creates a deep copy to avoid modifying the original instance.
        yaml_copy = copy.deepcopy(self)

        # Prepares each child dataclass for YAML serialization.
        yaml_copy.session_io.prepare_for_saving()

        yaml_copy.to_yaml(file_path=file_path)

    @classmethod
    def load(cls, file_path: Path) -> MultiDayConfiguration:
        """Loads configuration from a YAML file.

        Args:
            file_path: The path to the .yaml configuration file.

        Returns:
            A MultiDayConfiguration instance populated with the loaded data.
        """
        return cls.from_yaml(file_path=file_path)
