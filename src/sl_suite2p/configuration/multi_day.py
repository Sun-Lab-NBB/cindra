"""Provides configuration classes for the multi-day (across-session) sl-suite2p pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import field, dataclass

from ataraxis_base_utilities import ensure_directory_exists
from ataraxis_data_structures import YamlConfig

from .single_day import SignalExtraction, SpikeDeconvolution

if TYPE_CHECKING:
    from pathlib import Path


@dataclass()
class Main:
    """Stores the parameters that broadly affect the runtime and performance of the multi-day pipeline as a whole."""

    parallel_workers: int = 20
    """The number of workers used to parallelize certain processing operations. This worker pool is used by numba when
    it parallelizes certain computations used during registration and ROI processing. There is generally no benefit from
    increasing this parameter above 20 cores per each processed session. On machines with a high number of cores, it is
    recommended to keep this value between 10 and 20 cores and to instead parallelize processing sessions.
    Setting this to -1 or 0 removes worker limits, forcing the pipeline to use all available CPU cores."""

    display_progress_bars: bool = False
    """Determines whether to display progress bars for certain processing steps. Only enable this option when running
    all processing steps sequentially. Having this enabled when running multiple sessions or planes in parallel may 
    interfere with properly communicating progress via the terminal."""


@dataclass()
class IO:
    """Stores the parameters that specify input data location, format, and output directories."""

    session_directories: list[str] = field(default_factory=list)
    """Specifies the sessions to register across days as absolute paths to their root directories.
    Sessions are natural-sorted, and the first session after sorting becomes the 'main session' which stores
    the processing tracker file. Each session directory is expected to contain the combined_metadata.npz file created 
    by the single-day processing pipeline."""

    dataset_name: str = ""
    """Specifies the name of the multiday dataset. This name is used to create the output directory under each 
    session's 'multiday' directory (e.g., session/multiday/{dataset_name}/) and to identify the dataset in the 
    tracker file."""


@dataclass()
class CellSelection:
    """Stores parameters for selecting single-day-registered cells (ROIs) to be tracked across multiple sessions
    (days).
    """

    probability_threshold: float = 0.85
    """The minimum required cell probability score assigned to the ROI by the single-day suite2p classifier. Cells 
    with a lower classifier score are excluded from multi-day processing."""

    maximum_size: int = 1000
    """The maximum allowed ROI size, in pixels. Cells with a larger pixel size are excluded from processing."""

    mroi_stripe_borders: list[int] = field(default_factory=list)
    """Stores the x-coordinates of combined MROI image stripe borders. For MROI recordings, 'stripes' are the individual
    imaging ROIs acquired in the multi-ROI mode. Keep this field set to an empty list to skip stripe border-filtering or
    when working with non-MROI recordings.
    """

    mroi_stripe_margin: int = 30
    """The minimum required distance, in pixels, between the center-point (the median x-coordinate) of the cell (ROI)
    and the MROI stripe border. Cells that are too close to stripe borders are excluded from processing to avoid
    ambiguities associated with tracking cells that span multiple stripes. This parameter is only used if
    'mroi_stripe_borders' field is not set to an empty list."""


@dataclass()
class Registration:
    """Stores parameters for aligning (registering) the sessions from multiple days to the same visual (sampling)
    space.
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
class Clustering:
    """Stores parameters for tracking (clustering) cell (ROI) masks across multiple registered sessions (days)."""

    criterion: str = "distance"
    """Specifies the criterion for clustering (grouping) cell (ROI) masks from different sessions. Currently, the only 
    valid option is 'distance'."""

    threshold: float = 0.75
    """Specifies the threshold for the clustering algorithm. Cell masks will be clustered (grouped) together if their  
    clustering criterion is below this threshold value."""

    mask_prevalence: int = 50
    """Specifies the minimum percentage of all registered sessions that must include the clustered cell mask. Cell masks
    present in fewer percent of sessions than this value are excluded from processing. This parameter is used to filter
    out cells that are mostly silent or not distinguishable across sessions."""

    pixel_prevalence: int = 50
    """Specifies the minimum percentage of all registered sessions in which a cell mask pixel must be present for it to 
    be used to construct the template mask. Pixels present in fewer percent of sessions than this value are not used to 
    define the template masks. Template masks are used to extract the cell fluorescence from the original (non-deformed)
    visual space of every session. This parameter is used to isolate the part of the cell that is stable across 
    sessions, which is required for the extraction step to work correctly (target only the tracked cell)."""

    step_sizes: list[int] = field(default_factory=lambda: [200, 200])
    """Specifies the block size for the cell clustering (across-session tracking) process, in pixels, in the order of 
    (height, width). To reduce the memory (RAM) overhead, the algorithm divides the deformed (shared) visual space into 
    blocks and then processes one (or more) blocks at a time."""

    bin_size: int = 50
    """Specifies the additional length, in pixels, the algorithm is allowed to extend into the neighboring regions when 
    segmenting cells into grid bins. Before clustering cells across sessions, the algorithms pre-segments them into 
    grid bins using 'step_sizes'. Additionally, it uses +- 'bin_size' to extend into neighboring regions to better 
    cluster the cells around grid borders."""

    maximum_distance: int = 20
    """Specifies the maximum distance, in pixels, that can separate masks across multiple sessions. The clustering 
    algorithm will consider cell masks located at most within this distance from each-other across days as the same 
    cells during tacking."""

    minimum_size: int = 25
    """The minimum size of the non-overlapping cell (ROI) region, in pixels, that has to be covered by the template 
    mask, for the cell to be assigned to that template. This is used to determine which template(s) the cell belongs to 
    (if any), for the purpose of tracking it across sessions."""


@dataclass()
class MultiDayConfiguration(YamlConfig):
    """Aggregates the configuration parameters for the multi-day suite2p pipeline.

    Notes:
        This class is based on the reference implementation here:
        https://github.com/sprustonlab/multiday-suite2p-public.
    """

    main: Main = field(default_factory=Main)
    """Stores global parameters that broadly define the suite2p multi-day processing configuration."""
    io: IO = field(default_factory=IO)
    """Stores parameters that control data input and output during various stages of the pipeline."""
    cell_selection: CellSelection = field(default_factory=CellSelection)
    """Stores parameters for selecting single-day-registered cells (ROIs) to be tracked across multiple sessions (days).
    """
    registration: Registration = field(default_factory=Registration)
    """Stores parameters for aligning (registering) the sessions from multiple days to the same visual (sampling) space.
    """
    clustering: Clustering = field(default_factory=Clustering)
    """Stores parameters for tracking (clustering) cell (ROI) masks across multiple registered sessions (days)."""
    signal_extraction: SignalExtraction = field(default_factory=SignalExtraction)
    """Stores parameters for extracting fluorescence signals from ROIs and surrounding neuropil regions of the cells 
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
