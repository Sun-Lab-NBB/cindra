"""This module stores the classes used to configure the single-day (within-session) sl-suite2p pipeline. This is the
'original' suite2p pipeline used to process brain activity data collected as part of a single continuous recording.
"""

from typing import Any
from pathlib import Path
from dataclasses import field, asdict, fields, dataclass

import numpy as np
from ataraxis_base_utilities import ensure_directory_exists
from ataraxis_data_structures import YamlConfig

from ..version import version, sl_version, python_version


def extract_params_for_section(dataclass_type: Any, ops_dict: dict[str, Any]) -> dict[str, Any]:
    """Extracts parameters from ops_dict that belong to a specific dataclass section.

    This service function is used internally to regenerate the single-day and multi-day configuration files from
    'ops' dictionaries.

    Args:
        dataclass_type: The configuration sub-dataclass type whose parameters should be extracted.
        ops_dict: The dictionary containing the 'ops' parameters.

    Returns:
        A dictionary containing the parameters from the specified dataclass section. This dictionary can then be used
        to overwrite the '__dict__' of the section dataclass to set all class attributes to the extracted values.
    """
    field_names = {f.name for f in fields(dataclass_type)}
    extracted = {}

    for key, value in ops_dict.items():
        if key in field_names:
            # Handles potential type conversion issues
            try:
                field_info = next(f for f in fields(dataclass_type) if f.name == key)
                # For list fields with factory defaults, ensures we have the right type
                if hasattr(field_info.type, "__origin__") and field_info.type.__origin__ is list:
                    if not isinstance(value, list):
                        # Converts single values to lists if the field expects a list
                        extracted[key] = [value] if value is not None else []
                    else:
                        extracted[key] = value
                else:
                    extracted[key] = value
            except (StopIteration, AttributeError):
                # If we can't find field info, just use the value as-is
                extracted[key] = value

    return extracted


@dataclass
class Main:
    """Stores global parameters that broadly define the suite2p single-day processing configuration."""

    nplanes: int = 1
    """The number of imaging planes, stored as a sequence inside each input TIFF file."""

    nchannels: int = 1
    """The number of imaging channels per imaging plane. Typically, this is either 1 or 2. The algorithm expects images
    from different channels of the same plane to be saved sequentially (e.g.: plane 1 ch1, plane 1 ch2, plane 2 ch1, 
    etc.)."""

    functional_chan: int = 1
    """The channel used for extracting functional ROIs (cells). Note, this parameter uses 1-based indexing, where '1' 
    means the first channel and '2' means the second channel."""

    tau: float = 0.4
    """The timescale of the sensor, in seconds, used for computing the deconvolution kernel. The kernel is fixed to 
    have this decay and is not fit to the data. Note, the default value is optimized for GCaMP6f animals recorded with 
    the Mesoscope and likely needs to be increased for most other use cases."""

    fs: float = 10.0014
    """The sampling rate per plane in Hertz. For instance, if you have a 10-plane recording acquired at 30Hz, then the 
    sampling rate per plane is 3Hz, so set this to 3."""

    do_bidiphase: bool = False
    """Determines whether to perform the computation of bidirectional phase offset for misaligned line scanning 
    experiments (applies to two-photon recordings only)."""

    bidiphase: int = 0
    """The user-specified bidirectional phase offset for line scanning experiments. If set to any value besides 0, then 
    this offset is used and applied to all frames in the recording when 'do_bidiphase' is True. If set to 0, then the 
    suite2p will estimate the bidirectional phase offset automatically from 'nimg_init' frames. The computed or 
    user-defined offset is applied to all frames before the main processing pipeline."""

    bidi_corrected: bool = False
    """Tracks whether bidirectional phase correction has been applied to the registered dataset. This argument is 
    generally not meant to be set by the user and is instead written automatically when the algorithm performs 
    the bidirectional phase offset correction."""

    frames_include: int = -1
    """Determines the number of frames of the session's movie to process for each plane. If set to 0, the suite2p will 
    not do any processing. If negative (-1), the suite2p will process all available frames."""

    parallel_workers: int = 20
    """The number of workers used to parallelize certain processing operations. This worker pool is used by numba when 
    it parallelizes certain computations used during registration or ROI processing. Note, there is generally no benefit
    from increasing this parameter above 20 cores per each processed plane. On machines with a high number of cores, it 
    is recommended to keep this value between 10 and 20 cores and to manually parallelize processing sessions and / or 
    planes. See the example notebook for details on manual parallelization. Setting this to -1 or 0 removes worker 
    limits, forcing the pipeline to use all available CPU cores."""

    progress_bars: bool = False
    """Determines whether to display progress bars for certain processing steps. Only enable this option when running 
    all processing steps sequentially. Having this enabled when running multiple sessions or planes in-parallel will 
    interfere with properly communicating progress via the terminal."""

    ignore_flyback: list[int] = field(default_factory=list)
    """The list of 'flyback' plane indices to ignore when processing the data. Flyback planes typically contain no valid
    imaging data, so it is common to exclude them from processing."""

    python_version: str = python_version
    """Stores the Python version that was used to instantiate this configuration file. This is a non-user-addressable 
    field that stores important runtime ID information."""

    sl_suite2p_version: str = sl_version
    """Stores the sl-suite2p library version (release) that was used to instantiate this configuration file. This is a 
    non-user-addressable field that stores important runtime ID information."""

    base_suite2p_version: str = version
    """Stores the original suite2p version against which the used sl-suite2p version was built. This is a 
    non-user-addressable field that stores important runtime ID information."""


@dataclass
class FileIO:
    """Stores general I/O parameters that specify input data location, format, and working and output directories."""

    ignored_file_names: list[str] = field(default_factory=list)
    """Specifies the file names that should be ignored when searching for and loading raw data. Any file whose name 
    exactly matches one of the names stored inside this list will not be processed even if it has the correct 
    extension and is located inside one of the input directories."""

    fast_disk: str = ""
    """The path to the root 'working' directory where to store the temporary binary files created during processing. 
    This field allows optimizing processing on machines with slow storage drives and fast NVME 'work' drives by caching 
    all data on the fast drive during runtime. Do not modify this field unless your use case specifically benefits 
    from caching the data on a different drive than the one that stores the raw data. If this field is not modified, 
    'save_path0' is used to store the temporary files."""

    delete_bin: bool = False
    """Determines whether to delete the binary file(s) created during the frame registration stage (registered .bin 
    file). Note, if the data produced by the 'single-day' pipeline is intended to be later processed as part of the 
    'multi-day' pipeline, this has to be False. The multi-day pipeline reuses the registered binary files to extract 
    the fluorescence of cells tracked across days."""

    mesoscan: bool = False
    """Indicates whether the data submitted to the pipeline are ScanImage Mesoscope multi-page TIFFs."""

    save_path0: str = ""
    """The path to the root output directory where the pipeline results should be saved. Note, the pipeline generates 
    the 'save_folder' under the root directory specified by this argument and output all data to the generated save 
    folder."""

    save_folder: str = "suite2p"
    """The name of the folder under which the pipeline results should be stored. If this is not provided, the pipeline 
    defaults to using 'suite2p' as the save directory, created under the root directory specified by 'save_path0'. Note,
    if the data produced by the 'single-day' pipeline is intended to be later processed as part of the 'multi-day' 
    pipeline, do NOT modify this field. The multi-day pipeline expects the save_folder to be 'suite2p' (default)."""

    data_path: list[str] = field(default_factory=list)
    """The list of paths to the directories where to search for the input TIFF files. This is used during the initial 
    conversion of the raw data (expected to be .tiff / .tif) to the binary file format used by the suite2p pipeline. 
    Note, even if your data is stored in a single folder, add it here as a list with a single item."""

    look_one_level_down: bool = False
    """Determines whether to search for TIFF files in the subfolders of the directories provided as 'data_path' field. 
    If True, the 'subfolders' field has to be set to a valid list of subfolder names to search."""

    subfolders: list[str] = field(default_factory=list)
    """Stores specific subfolder names to search through for TIFF files. All subfolders must be stored under the 
    one or more directories specified by 'data_path'."""

    move_bin: bool = False
    """Determines whether to move the binary file(s) to the save directory after processing, if 'fast_disk' differs 
    from the 'save_path0'. Note, if using non-default 'save_folder' name, enabling this option will move the binary 
    files from the temporary 'suite2p' folder to the 'save_folder'. Otherwise, if the save folder and the temporary 
    folder are both 'suite2p', the binaries are automatically created and stored inside the 'save_folder'."""


@dataclass
class Output:
    """Stores parameters for aggregating and saving the processing results of each plane as a unified directory or
    file.
    """

    save_mat: bool = False
    """Determines whether to save the single-session pipeline output as a MATLAB file (e.g., Fall.mat)."""

    combined: bool = True
    """Determines whether to combine results across planes into a 'combined' folder at the end of processing. If the 
    results of the single-day pipeline are intended to be later processed as part of the multi-day pipeline, this has 
    to be True. This option is safe to use even with single-plane data."""

    aspect: float = 0.666666666
    """The pixel-to-micron ratio (X:Y) used in the GUI to ensure all images are displayed correctly. This field is not 
    used during headless processing."""


@dataclass
class Registration:
    """Stores parameters for rigid registration, which is used to correct motion artifacts between frames by
    counter-shifting the entire frame.
    """

    do_registration: int = 1
    """Determines whether to run the non-rigid motion registration. Note, if this is set to 1 and the processed data 
    appears to be already registered, skips re-registering the data. If this is set to 2, re-registers the data even 
    if it is already registered. Setting this to 0 disables registration."""

    align_by_chan: int = 1
    """The channel to use for frame alignment (registration). This field uses 1-based indexing, so 1 means 1st channel 
    and 2 means 2nd channel. If the recording features both a functional and non-functional channels, it is recommended 
    to use the non-functional channel for this purpose."""

    nimg_init: int = 500
    """The number of frames to use to compute the reference image. During registration, each frame is registered to the
    reference image to remove motion artifacts."""

    batch_size: int = 100
    """The number of frames to register simultaneously in each batch. When processing data on fast (NVME) drives, 
    increasing this parameter has minimal benefits and results in undue RAM use overhead. Therefore, on fast drives, 
    keep this number low. On slow drives, increasing this number may result in faster runtime, at the expense of 
    increased RAM use."""

    maxregshift: float = 0.1
    """The maximum allowed shift during registration, given as a fraction of the frame size, in pixels
    (e.g., 0.1 indicates 10%). This determines how much the algorithm is allowed to shift the entire frame to align it 
    to the reference image."""

    smooth_sigma: float = 1.15
    """The standard deviation (in pixels) of the Gaussian filter used to smooth the phase correlation between the 
    reference image and the current frame."""

    smooth_sigma_time: float = 0.0
    """The standard deviation (in frames) of the Gaussian used to temporally smooth the data before computing 
    phase correlation."""

    keep_movie_raw: bool = False
    """Determines whether to keep the binary file of the raw (non-registered) frames. This is desirable when initially 
    configuring the suite2p parameters, as it allows visually comparing registered frames to non-registered frames in 
    the GUI. For well-calibrated runtimes, it is advised to have this set to False."""

    two_step_registration: bool = False
    """Determines whether to perform a two-step registration. This process consists of the initial registration 
    (first step) followed by refinement (second step) registration. This procedure is helpful when working with low 
    signal-to-noise data and requires 'keep_movie_raw' to be set to True."""

    reg_tif: bool = False
    """Determines whether to write the registered binary data to TIFF files, in addition to keeping it as the .bin 
    (binary) suite2p files."""

    reg_tif_chan2: bool = False
    """Determines whether to generate TIFF files for the registered channel 2 data, if processed data contains two 
    channels."""

    th_badframes: float = 1.0
    """The threshold for excluding poor-quality frames when performing cropping. Primarily, this is used during two-step
    registration to exclude frames with excessive motion from the refinement registration step. Setting this to a 
    smaller value excludes more frames."""

    norm_frames: bool = True
    """Determines whether to normalize frames during shift detection to improve registration accuracy."""

    force_ref_img: bool = False
    """Determines whether to force the use of a pre-stored reference image for registration, instead of recomputing the 
    image during runtime."""

    pad_fft: bool = False
    """Determines whether to pad the image during the FFT portion of the registration to reduce edge effects."""

    compute_registration_metrics: int = 1
    """Determines whether to compute the registration quality metrics. This step is optional, registration metrics are 
    NOT used by the suite2p processing pipeline. However, these metrics are important for assessing the registration 
    quality via the GUI. Note, computing the registration metrics is a fairly expensive operation, sometimes taking as 
    much time as computing the registration offsets. Setting this field to 1 only computes the registration metrics if
    the processed data undergoes registration during runtime (matching the original suite2p behavior). Setting this 
    to 2 computes registration metrics even if the data is not undergoing registration during runtime. Setting this to 
    0 disables registration metrics computation."""

    reg_metric_n_pc: int = 10
    """The number of Principle Components (PCs) used to compute the registration metrics. Note, the time to compute 
    the registration metrics scales with the number of computed PCs, so it is recommended to keep the number as low 
    as feasible for each use case."""


@dataclass
class OnePRegistration:
    """Stores parameters for additional pre-registration processing used to improve the registration of 1-photon
    datasets.
    """

    one_p_reg: bool = False
    """Determines whether to perform high-pass spatial filtering and tapering to improve one-photon image 
    registration. For two-photon datasets, this should be set to False."""

    spatial_hp_reg: int = 42
    """The spatial high-pass filter window size, in pixels."""

    pre_smooth: float = 0.0
    """The standard deviation for Gaussian smoothing applied before spatial high-pass filtering. The smoothing will 
    only be applied if this field is greater than 0.0."""

    spatial_taper: float = 40.0
    """The number of pixels to ignore at the image edges to reduce edge artifacts during registration."""


@dataclass
class NonRigid:
    """Stores parameters for non-rigid registration, which is used to improve motion registration in complex
    datasets by dividing frames into subregions and shifting each subregion independently of other subregions.
    """

    nonrigid: bool = True
    """Determines whether to perform non-rigid registration to correct for local motion and deformation. This is 
    primarily used for correcting non-uniform motion."""

    block_size: list[int] = field(default_factory=lambda: [128, 128])
    """The block size, in pixels, for non-rigid registration, defining the dimensions of subregions used in 
    the correction. It is recommended to keep this size a power of 2 and / or 3 for more efficient FFT computation. 
    During processing, each frame will be split into sub-regions with these dimensions and the registration will then be
    applied to each region."""

    snr_thresh: float = 1.2
    """The signal-to-noise ratio threshold. The phase correlation peak must be this many times higher than the 
    noise level for the algorithm to accept the block shift and apply it to the output dataset."""

    maxregshift_nr: float = 5.0
    """The maximum allowed shift, in pixels, for each block relative to the rigid registration shift."""


@dataclass
class ROIDetection:
    """Stores parameters for cell ROI detection."""

    preclassify: float = 0.5
    """The classifier probability threshold used to pre-filter the cells before signal extraction. This is the minimum 
    classifier confidence value (that the classified ROI is a cell) for the ROI to be processed further. If this is set 
    to 0.0, then all detected ROIs (cells) are kept."""

    roidetect: bool = True
    """Determines whether to perform ROI detection and classification."""

    sparse_mode: bool = True
    """Determines whether to use the sparse mode for cell detection, which is well-suited for data with sparse 
    signals."""

    spatial_scale: int = 0
    """The optimal spatial scale, in pixels, for the processed data. This is used to adjust detection sensitivity. A 
    value of 0 means automatic detection based on the data's spatial scale. Values above 0 are applied in increments of 
    6 pixels (1 -> 6 pixels, 2-> 12 pixels, etc.)."""

    connected: bool = True
    """Determines whether to require the detected ROIs to be fully connected regions."""

    threshold_scaling: float = 2.0
    """The scaling factor for the detection threshold. This determines how distinctly ROIs have to stand out from 
    background noise to be considered valid."""

    spatial_hp_detect: int = 25
    """The window size, in pixels, for spatial high-pass filtering applied before neuropil subtraction during 
    ROI detection."""

    max_overlap: float = 0.75
    """The maximum allowed fraction of overlapping pixels between ROIs. ROIs that overlap above this threshold are be 
    discarded."""

    high_pass: int = 100
    """The window size, in frames, for running mean subtraction over time to remove low-frequency ROI drift."""

    smooth_masks: bool = True
    """Determines whether to smooth the ROI masks in the final pass of cell detection."""

    max_iterations: int = 50
    """The maximum number of iterations allowed for cell extraction. Generally, more iterations lead to more accurate 
    cell detection, but having the value too high may be detrimental."""

    nbinned: int = 5000
    """The maximum number of binned frames to use for ROI detection. Settings this value to a higher number leads to 
    more ROIs being detected, but reduces processing speed and increases RAM overhead."""

    denoise: bool = False
    """Determines whether to denoise the binned movie before cell detection in sparse mode to enhance performance. 
    If enabled, 'sparse_mode' has to be True."""


@dataclass
class CellposeDetection:
    """Stores parameters for the Cellpose algorithm, which can optionally be used to improve cell ROI extraction."""

    anatomical_only: int = 0
    """Specifies the Cellpose mode for cell detection:
        0: Do not use Cellpose. This automatically disables all other fields in this section.
        1: Detect masks on the max projection divided by the mean image.
        2: Detect masks on the mean image.
        3: Detect masks on the enhanced mean image.
        4: Detect masks on the max projection image.
    """

    diameter: int = 0
    """Specifies the diameter, in pixels, to look for when finding cell ROIs. If set to 0, Cellpose estimates the 
    diameter automatically.."""

    cellprob_threshold: float = 0.0
    """The threshold for cell detection, used to filter out low-confidence ROIs."""

    flow_threshold: float = 1.5
    """The flow threshold, used to control the algorithm's sensitivity to cell boundaries."""

    spatial_hp_cp: int = 0
    """The window size, in pixels, for spatial high-pass filtering applied to the image before Cellpose processing."""

    pretrained_model: str = "cyto"
    """Specifies the pretrained model to use for cell detection. Can be a built-in model name (e.g., 'cyto') or a 
    path to a custom model."""


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


@dataclass
class Classification:
    """Stores parameters for classifying detected ROIs as real cells or artifacts."""

    soma_crop: bool = True
    """Determines whether to crop dendritic regions from detected ROIs to focus on the cell body for classification 
    purposes."""

    use_builtin_classifier: bool = False
    """Determines whether to use the built-in classifier for cell detection."""

    classifier_path: str = ""
    """The path to a custom classifier file, if not using the built-in classifier."""


@dataclass
class Channel2:
    """Stores parameters for processing the second channel in multichannel datasets."""

    chan2_thres: float = 0.65
    """The threshold for considering an ROI detected in one channel as detected (present) in the second channel. 
    This threshold specifies the ratio of channel 1 pixels to channel 2 pixels for the ROI to be considered present in 
    both channels."""


@dataclass
class SingleDayS2PConfiguration(YamlConfig):
    """Aggregates the configuration parameters for the single-day suite2p pipeline.

    Notes:
        This class is based on the 'default_ops' dictionary from the original suite2p package. The default parameters
        in for this class are tuned for working with GCaMP6F fluorescence data recorded using 2-Photon Random Access
        Mesoscope (2P-RAM).
    """

    # Define the instances of each nested settings class as fields
    main: Main = field(default_factory=Main)
    """Stores global parameters that broadly define the suite2p single-day processing configuration."""
    file_io: FileIO = field(default_factory=FileIO)
    """Stores general I/O parameters that specify input data location, format, and working and output directories."""
    output: Output = field(default_factory=Output)
    """Stores parameters for aggregating and saving the processing results of each plane as a unified directory or
    file."""
    registration: Registration = field(default_factory=Registration)
    """Stores parameters for rigid registration, which is used to correct motion artifacts between frames by
    counter-shifting the entire frame."""
    one_p_registration: OnePRegistration = field(default_factory=OnePRegistration)
    """Stores parameters for additional pre-registration processing used to improve the registration of 1-photon
    datasets."""
    non_rigid: NonRigid = field(default_factory=NonRigid)
    """Stores parameters for non-rigid registration, which is used to improve motion registration in complex 
    datasets."""
    roi_detection: ROIDetection = field(default_factory=ROIDetection)
    """Stores parameters for cell ROI detection and extraction."""
    cellpose_detection: CellposeDetection = field(default_factory=CellposeDetection)
    """Stores parameters for the Cellpose algorithm, which can optionally be used to improve cell ROI extraction."""
    signal_extraction: SignalExtraction = field(default_factory=SignalExtraction)
    """Stores parameters for extracting fluorescence signals from ROIs and surrounding neuropil regions."""
    spike_deconvolution: SpikeDeconvolution = field(default_factory=SpikeDeconvolution)
    """Stores parameters for deconvolving fluorescence signals to infer spike trains."""
    classification: Classification = field(default_factory=Classification)
    """Stores parameters for classifying detected ROIs as real cells or artifacts."""
    channel2: Channel2 = field(default_factory=Channel2)
    """Stores parameters for processing the second channel in multichannel datasets."""

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
    def from_ops(cls, ops_dict: dict[str, Any]) -> "SingleDayS2PConfiguration":
        """Creates a SingleDayS2PConfiguration instance from the target 'ops'' dictionary.

        Notes:
            This method replaces any missing parameters with default initialization values and ignores extra parameters
            not recognized by the configuration schema.

        Args:
            ops_dict: The dictionary that contains the multi-day suite2p processing parameters.

        Returns:
            A SingleDayS2PConfiguration instance.
        """
        # Extracts parameters for each configuration section
        main_params = extract_params_for_section(Main, ops_dict)
        file_io_params = extract_params_for_section(FileIO, ops_dict)
        output_params = extract_params_for_section(Output, ops_dict)
        registration_params = extract_params_for_section(Registration, ops_dict)
        one_p_registration_params = extract_params_for_section(OnePRegistration, ops_dict)
        non_rigid_params = extract_params_for_section(NonRigid, ops_dict)
        roi_detection_params = extract_params_for_section(ROIDetection, ops_dict)
        cellpose_detection_params = extract_params_for_section(CellposeDetection, ops_dict)
        signal_extraction_params = extract_params_for_section(SignalExtraction, ops_dict)
        spike_deconvolution_params = extract_params_for_section(SpikeDeconvolution, ops_dict)
        classification_params = extract_params_for_section(Classification, ops_dict)
        channel2_params = extract_params_for_section(Channel2, ops_dict)

        # Creates a configuration class instance using the extracted parameters
        # Missing parameters will automatically use the default values defined in each dataclass
        return cls(
            main=Main(**main_params),
            file_io=FileIO(**file_io_params),
            output=Output(**output_params),
            registration=Registration(**registration_params),
            one_p_registration=OnePRegistration(**one_p_registration_params),
            non_rigid=NonRigid(**non_rigid_params),
            roi_detection=ROIDetection(**roi_detection_params),
            cellpose_detection=CellposeDetection(**cellpose_detection_params),
            signal_extraction=SignalExtraction(**signal_extraction_params),
            spike_deconvolution=SpikeDeconvolution(**spike_deconvolution_params),
            classification=Classification(**classification_params),
            channel2=Channel2(**channel2_params),
        )


def generate_default_ops(as_dict: bool = True) -> dict[str, Any] | SingleDayS2PConfiguration:
    """Instantiates and returns an 'ops' dictionary or configuration class that contains default single-day
    pipeline parameters.

    Args:
        as_dict: If True, the function converts the class to dictionary format. Otherwise, returns the class as the
            'SingleDayS2PConfiguration' dataclass instance.
    """
    default_configuration = SingleDayS2PConfiguration()  # Instantiates the default configuration instance.

    if not as_dict:
        return default_configuration
    return default_configuration.to_ops()  # Converts the configuration instance to dictionary format.
