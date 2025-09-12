"""This module stores the classes used to configure the multi-day (across-session) sl-suite2p pipeline. This pipeline
extends the original suite2p code to support tracking the same objects (cells) across multiple days.
"""

from typing import Any
from pathlib import Path
from dataclasses import field, asdict, dataclass

import numpy as np
from ataraxis_base_utilities import ensure_directory_exists
from ataraxis_data_structures import YamlConfig

from ..version import version, sl_version, python_version
from .single_day import extract_params_for_section


@dataclass()
class Main:
    """Stores global parameters that broadly define the suite2p multi-day processing configuration."""

    parallel_workers: int = 20
    """The number of workers used to parallelize certain processing operations. This worker pool is used by numba when 
    it parallelizes certain computations used during session ROI processing. Note, there is generally no benefit
    from increasing this parameter above 20 cores per each processed session. On machines with a high number of cores, 
    it is recommended to keep this value between 10 and 20 cores and to manually parallelize processing sessions. See 
    the example notebook for details on manual parallelization. Setting this to -1 or 0 removes worker limits, forcing 
    the pipeline to use all available CPU cores."""

    progress_bars: bool = False
    """Determines whether to display progress bars for certain processing steps. Only enable this option when running 
    a single multi-day processing pipeline at a time. Having this enabled when running multiple multi-day pipelines 
    in-parallel will interfere with properly communicating progress via the terminal."""

    python_version: str = python_version
    """Stores the Python version that was used to instantiate this configuration file. This is a non-user-addressable 
    field that stores important runtime ID information."""

    sl_suite2p_version: str = sl_version
    """Stores the sl-suite2p library version (release) that was used to instantiate this configuration file. This is a 
    non-user-addressable field that stores important runtime ID information."""

    base_suite2p_version: str = version
    """Stores the original suite2p version against which the used sl-suite2p version was built. This is a 
    non-user-addressable field that stores important runtime ID information."""


@dataclass()
class IO:
    """Stores parameters that control data input and output during various stages of the pipeline."""

    session_directories: list[str] = field(default_factory=list)
    """Specifies the list of sessions to register across days, as absolute paths to their root directories. 
    Note, each input directory must contain a 'combined' plane folder created by the single-day suite2p pipeline 
    at some level of the subdirectory tree. The 'combined' folder is created if the 'combined' 
    SingleDayS2PConfiguration attribute is 'True'."""

    multiday_save_path: str = ""
    """Specifies the path to the directory where to generate the output data hierarchy and save the multi-day 
    processing results. Note, all data will be saved under the 'save-folder', which itself will be created under the 
    directory specified by this field."""

    multiday_save_folder: str = "suite2p_multiday"
    """Specifies the name of the folder under which to save the data. This directory will be created under the 
    'save_path' directory as part of runtime. If a directory already exists, its' data will be overwritten as part of 
    runtime. When running multiple multi-day runtimes using partially overlapping datasets, make sure each runtime has 
    a unique 'save_folder' configuration parameter!"""


@dataclass()
class CellSelection:
    """Stores parameters for selecting single-day-registered cells (ROIs) to be tracked across multiple sessions
    (days).
    """

    probability_threshold: float = 0.85
    """The minimum required probability score assigned to the cell (ROI) by the single-day suite2p classifier. Cells 
    with a lower classifier score are excluded from multi-day processing."""

    maximum_size: int = 1000
    """The maximum allowed cell (ROI) size, in pixels. Cells with a larger pixel size are excluded from processing."""

    mesoscope_stripe_borders: list[int] = field(default_factory=list)
    """Stores the x-coordinates of combined mesoscope image stripe (ROI) borders. For mesoscope images, 'stripes' are 
    the individual imaging ROIs acquired in the 'multiple-ROI' mode. Keep this field set to an empty list to skip 
    stripe border-filtering or when working with non-mesoscope images.
    """

    stripe_margin: int = 30
    """The minimum required distance, in pixels, between the center-point (the median x-coordinate) of the cell (ROI) 
    and the mesoscope stripe border. Cells that are too close to stripe borders are excluded from processing to avoid 
    ambiguities associated with tracking cells that span multiple stripes. This parameter is only used if 
    'mesoscope_stripe_borders' field is not set to an empty list."""


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


@dataclass
class SignalExtraction:
    """Stores parameters for extracting fluorescence signals from ROIs and surrounding neuropil regions."""

    neuropil_extract: bool = True
    """Determines whether to extract neuropil signals. Typically, this is set to True to support later 
    delta-fluorescence-over-fluorescence (dff) analysis."""

    allow_overlap: bool = False
    """Determines whether to allow overlap pixels (pixels shared by multiple ROIs) to be used in the signal extraction. 
    Typically this is set to False to avoid contamination."""

    min_neuropil_pixels: int = 350
    """The minimum number of pixels required to compute the neuropil signal for each cell."""

    inner_neuropil_radius: int = 2
    """The number of pixels to keep between the ROI and the surrounding neuropil region to avoid signal bleed-over."""

    lam_percentile: int = 50
    """The percentile of Lambda within area to ignore when excluding the brightest pixels during neuropil extraction.
    Specifically, pixels with relative brightness above this threshold are excluded from neuropil signal to filter 
    out bright speckle outliers.
    """


@dataclass
class SpikeDeconvolution:
    """Stores parameters for deconvolving fluorescence signals to infer spike trains."""

    spikedetect: bool = True
    """Determines whether to perform fluorescence spike deconvolution."""

    neucoeff: float = 0.7
    """The neuropil coefficient applied for signal correction before deconvolution. Specifically, the neuropil signal
    is scaled by this coefficient before it is subtracted from the ROI signal when computing df/f values."""

    baseline: str = "maximin"
    """Specifies the method to compute the baseline of each trace. This baseline is then subtracted from each cell's 
    fluorescence. 'maximin' computes a moving baseline by filtering the data with a Gaussian of width 
    'sig_baseline' * 'fs', and then minimum filtering with a window of 'win_baseline' * 'fs', and then maximum 
    filtering with the same window. 'constant' computes a constant baseline by filtering with a Gaussian of width 
    'sig_baseline' * 'fs' and then taking the minimum value of this filtered trace. 'constant_percentile' computes a 
    constant baseline by taking the 'prctile_baseline' percentile of the trace."""

    win_baseline: float = 60.0
    """The time window, in seconds, over which to compute the baseline filter."""

    sig_baseline: float = 10.0
    """The standard deviation, in seconds, of the Gaussian filter applied to smooth the baseline signal."""

    prctile_baseline: float = 8.0
    """The percentile used to determine the baseline level of each trace (typically a low percentile reflecting 
    minimal activity)."""


@dataclass()
class MultiDayS2PConfiguration(YamlConfig):
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

    def to_npy(self, output_directory: Path) -> None:
        """Saves the managed configuration data as an 'ops.npy' file under the target directory.

        Notes:
            If the target output directory does not exist when this method is called, creates the directory before
            saving the file.

        Args:
            output_directory: The path to the directory where to save the 'ops.npy' file.
        """
        ensure_directory_exists(output_directory)  # Creates the directory, if necessary
        file_path = output_directory.joinpath("ops.npy")  # Computes the output path
        # Dumps the configuration data to the 'ops.npy' file.
        np.save(file_path, self.to_ops(), allow_pickle=True)  # type: ignore

    def to_config(self, file_path: Path) -> None:
        """Saves the managed configuration data as a .yaml file under the target directory.

        Notes:
            If the target output directory does not exist when this method is called, creates the directory before
            saving the file.

        Args:
            file_path: The path to the .yaml file where to save the configuration data.
        """
        ensure_directory_exists(file_path)  # Creates the file's parent directory, if necessary
        self.to_yaml(file_path=file_path)  # Dumps the data to a 'yaml' file.

    def to_ops(self) -> dict[str, Any]:
        """Returns the instance as a dictionary."""
        # Creates an empty dictionary to store all keys and values
        combined_ops = {}

        # Iterates through all dataclass fields
        # noinspection PyTypeChecker
        for section_name, section in asdict(self).items():
            # Adds all keys and values from each section to the combined dictionary
            if isinstance(section, dict):
                combined_ops.update(section)

        return combined_ops

    @classmethod
    def from_ops(cls, ops_dict: dict[str, Any]) -> "MultiDayS2PConfiguration":
        """Creates a MultiDayS2PConfiguration instance from the target 'ops'' dictionary.

        Notes:
            This method replaces any missing parameters with default initialization values and ignores extra parameters
            not recognized by the configuration schema.

        Args:
            ops_dict: The dictionary that contains the multi-day suite2p processing parameters.

        Returns:
            A MultiDayS2PConfiguration instance.
        """
        # Extracts parameters for each configuration section using the imported helper function
        main_params = extract_params_for_section(Main, ops_dict)
        io_params = extract_params_for_section(IO, ops_dict)
        cell_selection_params = extract_params_for_section(CellSelection, ops_dict)
        registration_params = extract_params_for_section(Registration, ops_dict)
        clustering_params = extract_params_for_section(Clustering, ops_dict)

        # Creates the configuration instance using the extracted parameters
        # Missing parameters will automatically use the default values defined in each dataclass
        return cls(
            main=Main(**main_params),
            io=IO(**io_params),
            cell_selection=CellSelection(**cell_selection_params),
            registration=Registration(**registration_params),
            clustering=Clustering(**clustering_params),
        )


def generate_default_multiday_ops(as_dict: bool = True) -> dict[str, Any] | MultiDayS2PConfiguration:
    """Instantiates and returns an 'ops' dictionary or configuration class that contains default multi-day
    pipeline parameters.

    Args:
        as_dict: If True, the function converts the class to dictionary format. Otherwise, returns the class as the
            'MultiDayS2PConfiguration' dataclass instance.
    """
    default_configuration = MultiDayS2PConfiguration()  # Instantiates the default configuration instance.

    if not as_dict:
        return default_configuration
    return default_configuration.to_ops()  # Converts the configuration instance to dictionary format.
