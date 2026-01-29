"""Provides configuration and runtime data classes for the single-day (within-session) sl-suite2p pipeline."""

from __future__ import annotations

import copy
from enum import StrEnum
from typing import TYPE_CHECKING
from pathlib import Path
from dataclasses import field, dataclass

import numpy as np
from ataraxis_base_utilities import ensure_directory_exists
from ataraxis_data_structures import YamlConfig

from ..version import version, python_version

if TYPE_CHECKING:
    from numpy.typing import NDArray


class InputFormat(StrEnum):
    """Defines the supported input data formats for the single-day pipeline."""

    TIFF = "tiff"
    """Standard TIFF files from non-mesoscope two-photon or one-photon imaging systems."""

    MESOSCAN = "mesoscan"
    """ScanImage Mesoscope multi-page TIFF files with embedded ROI metadata."""


class BaselineMethod(StrEnum):
    """Defines the supported methods for computing baseline fluorescence before spike deconvolution."""

    MAXIMIN = "maximin"
    """Applies Gaussian smoothing followed by minimum and maximum filters over a sliding window, tracking the lower
    envelope of slow signal fluctuations."""

    CONSTANT = "constant"
    """Uses the global minimum of the Gaussian-smoothed trace as a single constant baseline for the entire recording."""

    CONSTANT_PERCENTILE = "constant_percentile"
    """Uses a low percentile of the trace as a robust constant baseline, ignoring outliers."""


@dataclass
class Main:
    """Stores the parameters that broadly affect the runtime and performance of the single-day pipeline as a whole."""

    plane_count: int = 1
    """The number of imaging planes stored as a sequence inside each input TIFF file."""

    two_channels: bool = False
    """Determines whether the imaging data contains two channels per plane. When True, the algorithm expects images from
    both channels of the same plane to be saved sequentially (e.g.: plane 1 channel 1, plane 1 channel 2, plane 2
    channel 1, etc.)."""

    first_channel_functional: bool = True
    """Determines whether the first channel is used for ROI detection and signal extraction. This field is only
    applicable when two_channels is True. When both first_channel_functional and second_channel_functional are True,
    the pipeline performs independent ROI detection on both channels."""

    second_channel_functional: bool = False
    """Determines whether the second channel is used for ROI detection and signal extraction. This field is only
    applicable when two_channels is True. When both first_channel_functional and second_channel_functional are True,
    the pipeline performs independent ROI detection on both channels."""

    colocalization_threshold: float = 0.65
    """The threshold for determining whether an ROI detected in one channel is also present in the other channel.
    The algorithm computes the ratio of mean brightness inside the ROI to the combined brightness of the ROI and
    surrounding neuropil. ROIs with a ratio exceeding this threshold are marked as colocalized. When only one channel
    is functional, the analysis checks if its ROIs are present in the other channel. When both channels are functional,
    the analysis checks if first channel ROIs are present in the second channel. The output is a boolean .npy mask
    file with one entry per ROI in the reference channel."""

    tau: float = 0.4
    """The timescale of the sensor in seconds, used for computing the deconvolution kernel. The kernel is fixed to have
    this decay and is not fit to the data. The default value is optimized for GCaMP6f animals recorded with the
    Mesoscope and likely needs to be increased for most other use cases."""

    sampling_rate: float = 10.0014
    """The data sampling rate per each imaging plane in Hertz. For a 10-plane recording acquired at 30 Hz, the 
    sampling rate per plane would be 3 Hz."""

    compute_bidirectional_phase_offset: bool = False
    """Determines whether to compute the bidirectional phase offset for misaligned line scanning in two-photon
    recordings. This correction addresses misalignment between odd and even scan lines caused by bidirectional resonant
    scanning. Most recording software (including ScanImage) handles this correction during acquisition, so this option
    is rarely needed for properly configured systems."""

    bidirectional_phase_offset: int = 0
    """The bidirectional phase offset for line scanning 2-photon recordings. If set to any value besides 0, this
    offset is used and applied to all frames in the recording when compute_bidirectional_phase_offset is True. If set 
    to 0, the pipeline estimates the bidirectional phase offset automatically from the initial reference frames. The 
    computed or user-defined offset is applied to all frames before the main processing pipeline."""

    parallel_workers: int = 20
    """The number of workers used to parallelize certain processing operations. This worker pool is used by numba when
    it parallelizes certain computations used during registration and ROI processing. There is generally no benefit from
    increasing this parameter above 20 cores per each processed plane. On machines with a high number of cores, it is
    recommended to keep this value between 10 and 20 cores and to instead parallelize processing sessions and planes.
    Setting this to -1 or 0 removes worker limits, forcing the pipeline to use all available CPU cores."""

    display_progress_bars: bool = False
    """Determines whether to display progress bars for certain processing steps. Only enable this option when running
    all processing steps sequentially. Having this enabled when running multiple sessions or planes in parallel may 
    interfere with properly communicating progress via the terminal."""

    ignored_flyback_planes: list[int] = field(default_factory=list)
    """The list of flyback plane indices to ignore when processing the data. Flyback planes typically contain no valid
    imaging data, so it is common to exclude them from processing."""


@dataclass
class FileIO:
    """Stores the parameters that specify input data location, format, and output directories."""

    data_path: Path | None = None
    """The path to the root data directory containing the input TIFF files. The pipeline recursively searches this
    directory and all subdirectories for .tiff/.tif files to process."""

    save_path: Path | None = None
    """The path to the root output directory where to save the processing results. The pipeline automatically
    creates a 'suite2p' subdirectory under this path to store all output files."""

    input_format: InputFormat | str = InputFormat.TIFF
    """The format of the input data files. Use InputFormat.TIFF for standard TIFF files or InputFormat.MESOSCAN for
    ScanImage Mesoscope multi-page TIFFs with embedded ROI metadata."""

    ignored_file_names: list[str] = field(default_factory=list)
    """The list of file names to ignore when searching for and loading raw data. Any file whose name exactly matches
    one of the names in this is excluded from processing even if it has the correct extension and is located inside
    the input data directory."""

    def __post_init__(self) -> None:
        """Converts string paths to Path objects and string input_format to InputFormat enum after YAML loading."""
        if isinstance(self.data_path, str):
            self.data_path = Path(self.data_path) if self.data_path else None
        if isinstance(self.save_path, str):
            self.save_path = Path(self.save_path) if self.save_path else None
        if isinstance(self.input_format, str):
            self.input_format = InputFormat(self.input_format)


@dataclass
class Registration:
    """Stores parameters for rigid registration, which is used to correct motion artifacts between frames by
    counter-shifting the entire frame.
    """

    repeat_registration: bool = False
    """Determines whether to re-register data that appears to already be registered. When False, the pipeline skips
    registration if the data is already registered. When True, the pipeline re-registers the data regardless of its
    current registration state."""

    align_by_first_channel: bool = True
    """Determines whether to use the first channel for frame alignment (registration). When False, the second channel
    is used instead. If the recording features both a functional and non-functional channel, it is recommended to use
    the non-functional channel for alignment. This field is only applicable when two_channels is True in the Main
    configuration."""

    reference_frame_count: int = 500
    """The number of frames to use to compute the reference image. During registration, each frame is registered to the
    reference image to remove motion artifacts. The algorithm automatically selects the most stable (correlated) set 
    of frames when computing the reference image."""

    batch_size: int = 100
    """The number of frames to keep in memory at the same time when registering them to the reference image. When 
    processing data on fast (NVME) drives, increasing this parameter has minimal benefits and results in undue RAM use 
    overhead. On slow drives, increasing this number may result in faster runtime, at the expense of increased RAM 
    use."""

    maximum_shift_fraction: float = 0.1
    """The maximum allowed shift during registration, given as a fraction of t88he frame size (e.g., 0.1 indicates 10%).
    This determines how much the algorithm is allowed to shift the entire frame to align it to the reference image."""

    spatial_smoothing_sigma: float = 1.15
    """The standard deviation (in pixels) of the Gaussian filter used to spatially smooth the phase correlation surface
    between the reference image and each processed frame. Smoothing helps reduce noise in the correlation surface,
    improving the accuracy of sub-pixel shift detection. Higher values produce more smoothing but may reduce precision
    for detecting small shifts."""

    temporal_smoothing_sigma: float = 0.0
    """The standard deviation (in frames) of the Gaussian filter used to temporally smooth the phase correlation surface
    across consecutive frames. This reduces frame-to-frame noise in correlation values and can improve registration
    stability for noisy recordings. Setting this to 0.0 disables temporal smoothing."""

    keep_movie_raw: bool = False
    """Determines whether to keep the binary file of the raw (non-registered) frames. This is desirable when initially
    configuring the suite2p parameters, as it allows visually comparing registered frames to non-registered frames in
    the GUI. For well-calibrated runtimes, it is advised to have this set to False."""

    two_step_registration: bool = False
    """Determines whether to perform a two-step registration. This process consists of the initial registration
    (first step) followed by refinement (second step) registration. This procedure is helpful when working with low
    signal-to-noise data and requires 'keep_movie_raw' to be set to True."""

    bad_frame_threshold: float = 1.0
    """The threshold for identifying frames with excessive motion or poor correlation quality. The algorithm computes
    a ratio of motion deviation to phase correlation quality for each frame. Frames exceeding this threshold (scaled
    by 100 internally) are marked as 'bad' and excluded when computing the valid pixel region (yrange, xrange) after
    registration. This prevents a few frames with extreme motion from unnecessarily shrinking the usable field of view.
    Bad frames may also be excluded during movie binning for ROI detection. Lower values are more strict and exclude
    more frames."""

    normalize_frames: bool = True
    """Determines whether to clip pixel intensities to the 1st-99th percentile range during registration. This removes
    extreme outlier pixels from both the reference image and each frame before computing phase correlation, improving
    shift detection accuracy by reducing the influence of anomalously bright or dark pixels."""

    compute_registration_metrics: bool = True
    """Determines whether to compute registration quality metrics after registration completes. These metrics are not
    used by the processing pipeline but are useful for assessing registration quality via the GUI. Computing metrics
    is a fairly expensive operation that can take as long as the registration itself."""

    registration_metric_principal_components: int = 5
    """The number of Principal Components (PCs) used to compute the registration metrics. Note, the time to compute
    the registration metrics scales with the number of computed PCs, so it is recommended to keep the number as low
    as feasible for each use case."""


@dataclass
class OnePhotonRegistration:
    """Stores parameters for additional pre-registration processing used to improve the registration of 1-photon
    datasets.
    """

    enabled: bool = False
    """Determines whether to perform high-pass spatial filtering and tapering to improve one-photon image
    registration. For two-photon datasets, this should be set to False."""

    spatial_highpass_window: int = 42
    """The window size, in pixels, for spatial high-pass filtering. This filter removes low-frequency spatial
    variations such as uneven illumination that are common in one-photon imaging. The filter subtracts a spatially
    smoothed version of the image (using this window size) from the original, preserving only high-frequency
    features useful for registration."""

    pre_smoothing_sigma: float = 0.0
    """The standard deviation, in pixels, for Gaussian smoothing applied before spatial high-pass filtering. This
    reduces high-frequency noise that would otherwise be amplified by the high-pass filter. Setting this to 0.0
    disables pre-smoothing."""

    edge_taper_pixels: float = 40.0
    """The width, in pixels, of the tapering region at image edges. Pixel values are gradually reduced to zero
    within this border region to prevent edge artifacts during FFT-based phase correlation. Larger values provide
    smoother transitions but reduce the usable image area."""


@dataclass
class NonRigidRegistration:
    """Stores parameters for non-rigid registration, which is used to improve motion registration in complex
    datasets by dividing frames into subregions and shifting each subregion independently of other subregions.
    """

    enabled: bool = True
    """Determines whether to perform non-rigid registration to correct for local motion and deformation. This is
    primarily used for correcting non-uniform motion."""

    block_size: list[int] = field(default_factory=lambda: [128, 128])
    """The block size, in pixels, for non-rigid registration, defining the dimensions of subregions used in
    the correction. It is recommended to keep this size a power of 2 and/or 3 for more efficient FFT computation.
    During processing, each frame is split into sub-regions with these dimensions and the registration is applied
    to each region independently."""

    signal_to_noise_threshold: float = 1.2
    """The signal-to-noise ratio threshold. The phase correlation peak must be this many times higher than the
    noise level for the algorithm to accept the block shift and apply it to the output dataset."""

    maximum_block_shift: float = 5.0
    """The maximum allowed shift, in pixels, for each block relative to the rigid registration shift."""


@dataclass
class ROIDetection:
    """Stores parameters for Region of Interest (cell) detection."""

    enabled: bool = True
    """Determines whether to perform ROI detection and classification."""

    preclassification_threshold: float = 0.5
    """The classifier probability threshold used to pre-filter cells before signal extraction. This is the minimum
    classifier confidence value (that the classified ROI is a cell) for the ROI to be processed further. Setting this
    to 0.0 keeps all detected ROIs."""

    spatial_scale: int = 0
    """The optimal spatial scale, in pixels, for the processed data. This is used to adjust detection sensitivity.
    Setting this to 0 forces the algorithm to determine this value automatically. Values above 0 are applied in
    increments of 6 pixels (1 -> 6 pixels, 2 -> 12 pixels, etc.)."""

    diameter: int = 0
    """The expected cell diameter in pixels. Setting this to 0 forces the algorithm to estimate the diameter from the
    spatial scale during detection. The diameter is computed as 3 * 2^spatial_scale."""

    threshold_scaling: float = 2.0
    """The scaling factor for the ROI detection threshold. The final threshold is computed as this value multiplied
    by the spatial scale factor. Higher values require ROIs to stand out more distinctly from background noise,
    resulting in fewer but more confident detections. Lower values detect more ROIs but may include false positives."""

    spatial_highpass_window: int = 25
    """The window size, in pixels, for spatial high-pass filtering used during neuropil subtraction. The algorithm
    subtracts a spatially smoothed version of each frame (using this window size) to remove diffuse neuropil
    fluorescence and isolate cell bodies."""

    maximum_overlap: float = 0.75
    """The maximum allowed fraction of overlapping pixels between two ROIs. When two ROIs share more than this
    fraction of pixels, the ROI with lower signal quality is discarded. Lower values enforce stricter separation
    between detected cells."""

    temporal_highpass_window: int = 100
    """The window size, in frames, for temporal high-pass filtering applied before ROI detection. This removes
    slow fluorescence drifts (such as photobleaching or baseline changes) by subtracting a running mean computed
    over this window. Larger values preserve slower transients but may retain more drift artifacts."""

    maximum_iterations: int = 50
    """The iteration scaling factor for ROI extraction. The algorithm detects ROIs one at a time, subtracting each
    detected ROI's contribution before searching for the next. The actual iteration limit is this value multiplied
    by 250 internally (e.g., 50 allows up to 12,500 iterations). Higher values allow detecting more cells but
    increase processing time."""

    maximum_binned_frames: int = 5000
    """The maximum number of time-binned frames used for ROI detection. Temporal binning averages consecutive frames
    to improve signal-to-noise ratio for detection. Higher values provide better averaging but increase memory usage
    and processing time. The bin size is computed to produce at most this many binned frames."""

    denoise: bool = False
    """Determines whether to apply PCA-based denoising to the binned movie before ROI detection. This can improve
    detection in noisy recordings by removing uncorrelated noise while preserving spatially coherent signals."""


@dataclass
class SignalExtraction:
    """Stores parameters for extracting fluorescence signals from cell ROIs and surrounding neuropil regions."""

    extract_neuropil: bool = True
    """Determines whether to extract neuropil activity. If disabled, neuropil fluorescence is assumed to be zero
    during spike deconvolution."""

    allow_overlap: bool = False
    """Determines whether to include overlapping pixels (shared by multiple ROIs) in signal extraction. When disabled,
    pixels belonging to multiple ROIs are excluded from all of them to prevent signal contamination between
    neighboring cells. Enable this only if ROIs are sparse and overlap is minimal."""

    minimum_neuropil_pixels: int = 350
    """The minimum number of pixels required for each neuropil mask. The algorithm expands outward from the cell
    border until it accumulates at least this many non-cell pixels. Larger values provide more stable neuropil
    estimates but may include pixels from distant regions with different neuropil characteristics."""

    inner_neuropil_border_radius: int = 2
    """The width, in pixels, of the exclusion zone between the cell ROI and its neuropil mask. This gap prevents
    contamination of the neuropil signal by the cell's own fluorescence. Larger values provide better separation
    but reduce the neuropil sampling area near the cell."""

    cell_probability_percentile: int = 50
    """The percentile threshold for classifying pixels as belonging to a cell versus neuropil. Each pixel has a
    probability weight indicating how likely it belongs to a cell. Pixels with weights above this percentile
    (computed locally) are excluded from neuropil masks. Higher values are more permissive, including more
    pixels in neuropil masks but risking cell contamination."""


@dataclass
class SpikeDeconvolution:
    """Stores parameters for deconvolving fluorescence signals to infer spike trains."""

    extract_spikes: bool = True
    """Determines whether to deconvolve spike activity from the extracted fluorescence traces. When disabled, the
    pipeline still computes neuropil-corrected fluorescence (F - coefficient * Fneu) but skips the deconvolution
    step that estimates spike timing."""

    neuropil_coefficient: float = 0.7
    """The scaling factor applied to neuropil fluorescence before subtracting it from cell fluorescence. The corrected
    signal is computed as F_corrected = F_cell - coefficient * F_neuropil. Values typically range from 0.5 to 1.0,
    with 0.7 being a common default. Higher values apply stronger neuropil correction but risk over-subtracting
    signal from cells with weak neuropil contamination."""

    baseline_method: BaselineMethod | str = BaselineMethod.MAXIMIN
    """The method for computing baseline fluorescence to subtract before deconvolution. See BaselineMethod enumeration 
    for available options: MAXIMIN tracks the lower envelope using sliding window filters, CONSTANT uses the global
    minimum, and CONSTANT_PERCENTILE uses a low percentile as a robust constant baseline."""

    baseline_window: float = 60.0
    """The size of the sliding window, in seconds, for the 'maximin' baseline method. The minimum and maximum filters
    operate over this window to track slow baseline drifts while ignoring fast transients. Larger windows produce
    smoother baselines but may fail to track rapid baseline changes."""

    baseline_sigma: float = 10.0
    """The standard deviation, in seconds, of the Gaussian filter applied before baseline computation. Used by both
    'maximin' and 'constant' methods to smooth the trace before finding minima. Larger values produce more aggressive
    smoothing."""

    baseline_percentile: float = 8.0
    """The percentile of trace activity used as baseline for the 'constant_percentile' method. Lower values (e.g., 8)
    select points near the trace minimum, providing a robust estimate that ignores outliers. Only used when
    baseline_method is set to CONSTANT_PERCENTILE."""

    def __post_init__(self) -> None:
        """Converts string baseline_method to BaselineMethod enum after YAML loading."""
        if isinstance(self.baseline_method, str):
            self.baseline_method = BaselineMethod(self.baseline_method)


@dataclass
class Classification:
    """Stores parameters for classifying detected ROIs as real cells or artifacts."""

    crop_to_soma: bool = True
    """Determines whether to crop dendritic regions from detected ROIs before computing classification features.
    When enabled, the algorithm analyzes the radial distribution of fluorescence from each ROI's centroid and
    excludes pixels beyond where fluorescence contribution drops significantly. This focuses classification
    on the cell body, improving accuracy for neurons with extensive dendritic arbors."""

    use_builtin_classifier: bool = True
    """Determines whether to use the classifier bundled with sl-suite2p. When False, the pipeline looks for a
    user-trained classifier at ~/.suite2p/classifiers/classifier_user.npy. If that file does not exist, the
    builtin classifier is used as a fallback. Set this to True to always use the builtin classifier regardless
    of whether a user classifier exists."""

    custom_classifier_path: Path | None = None
    """The absolute path to a custom classifier file. When set, this classifier takes priority over both the
    builtin classifier and the user classifier at the default location. Leave as None to use the standard
    classifier selection logic based on use_builtin_classifier."""

    def __post_init__(self) -> None:
        """Converts string custom_classifier_path to Path after YAML loading."""
        if isinstance(self.custom_classifier_path, str):
            self.custom_classifier_path = Path(self.custom_classifier_path) if self.custom_classifier_path else None


@dataclass
class IOData:
    """Stores runtime data from the IO/binarization stage."""

    frame_height: int = 0
    """Frame height in pixels. Legacy: Ly."""

    frame_width: int = 0
    """Frame width in pixels. Legacy: Lx."""

    frame_count: int = 0
    """Total frames in the binary file. Legacy: nframes."""

    registered_binary_path: Path | None = None
    """Path to registered binary file (channel 1). Legacy: reg_file."""

    raw_binary_path: Path | None = None
    """Path to raw binary file (channel 1). Legacy: raw_file."""

    registered_binary_path_channel_2: Path | None = None
    """Path to registered binary file (channel 2). Legacy: reg_file_chan2."""

    raw_binary_path_channel_2: Path | None = None
    """Path to raw binary file (channel 2). Legacy: raw_file_chan2."""

    output_directory: Path | None = None
    """Plane output directory. Legacy: save_path, ops_path."""

    mesoscope_y_offset: int | None = None
    """Y-offset for mesoscope ROI positioning. Legacy: dy."""

    mesoscope_x_offset: int | None = None
    """X-offset for mesoscope ROI positioning. Legacy: dx."""

    mesoscope_lines: list[int] = field(default_factory=list)
    """Line indices for mesoscope ROI extraction. Legacy: lines."""

    plane_index: int | None = None
    """Plane index in multi-plane recording. Legacy: iplane."""

    def __post_init__(self) -> None:
        """Converts string paths to Path objects after YAML loading."""
        if isinstance(self.registered_binary_path, str):
            self.registered_binary_path = Path(self.registered_binary_path) if self.registered_binary_path else None
        if isinstance(self.raw_binary_path, str):
            self.raw_binary_path = Path(self.raw_binary_path) if self.raw_binary_path else None
        if isinstance(self.registered_binary_path_channel_2, str):
            self.registered_binary_path_channel_2 = (
                Path(self.registered_binary_path_channel_2) if self.registered_binary_path_channel_2 else None
            )
        if isinstance(self.raw_binary_path_channel_2, str):
            self.raw_binary_path_channel_2 = (
                Path(self.raw_binary_path_channel_2) if self.raw_binary_path_channel_2 else None
            )
        if isinstance(self.output_directory, str):
            self.output_directory = Path(self.output_directory) if self.output_directory else None


@dataclass
class RegistrationData:
    """Stores runtime data from the registration stage."""

    valid_y_range: tuple[int, int] = (0, 0)
    """Valid Y pixel range after cropping, as (start, end). Legacy: yrange."""

    valid_x_range: tuple[int, int] = (0, 0)
    """Valid X pixel range after cropping, as (start, end). Legacy: xrange."""

    bidirectional_phase_offset: int = 0
    """Computed or user-specified bidirectional phase offset. Legacy: bidiphase."""

    bidirectional_phase_corrected: bool = False
    """Whether bidirectional phase correction was applied. Legacy: bidi_corrected."""

    normalization_minimum: int = 0
    """Minimum intensity for frame normalization. Legacy: rmin."""

    normalization_maximum: int = 0
    """Maximum intensity for frame normalization. Legacy: rmax."""

    reference_image: NDArray[np.float32] | None = None
    """Reference image for frame alignment. Legacy: refImg."""

    rigid_y_offsets: NDArray[np.float32] | None = None
    """Y rigid registration offsets per frame. Legacy: yoff."""

    rigid_x_offsets: NDArray[np.float32] | None = None
    """X rigid registration offsets per frame. Legacy: xoff."""

    rigid_correlations: NDArray[np.float32] | None = None
    """Rigid registration correlation per frame. Legacy: corrXY."""

    nonrigid_y_offsets: NDArray[np.float32] | None = None
    """Y non-rigid offsets per frame and block. Legacy: yoff1."""

    nonrigid_x_offsets: NDArray[np.float32] | None = None
    """X non-rigid offsets per frame and block. Legacy: xoff1."""

    nonrigid_correlations: NDArray[np.float32] | None = None
    """Non-rigid correlation per frame and block. Legacy: corrXY1."""


@dataclass
class DetectionData:
    """Stores runtime data from the detection/extraction stage."""

    spatial_scale: float = 0.0
    """Computed spatial scale in pixels. Legacy: spatial_scale_pixels."""

    cell_diameter: int = 0
    """Computed or user-specified cell diameter in pixels. Legacy: diameter."""

    aspect_ratio: float = 0.0
    """Computed aspect ratio (diameter[0] / diameter[1]). Legacy: aspect."""

    mean_image: NDArray[np.float32] | None = None
    """Mean image from all registered frames. Legacy: mean_image, meanImg."""

    enhanced_mean_image: NDArray[np.float32] | None = None
    """High-pass filtered mean image. Legacy: enhanced_mean_image, meanImgE."""

    maximum_projection: NDArray[np.float32] | None = None
    """Maximum projection from all frames. Legacy: max_proj."""

    correlation_map: NDArray[np.float32] | None = None
    """Correlation map for cell detection. Legacy: Vcorr."""

    mean_image_channel_2: NDArray[np.float32] | None = None
    """Mean image for channel 2. Legacy: mean_image_channel_2, meanImg_chan2."""

    enhanced_mean_image_channel_2: NDArray[np.float32] | None = None
    """High-pass filtered mean image for channel 2. Legacy: enhanced_mean_image_chan2."""

    maximum_projection_channel_2: NDArray[np.float32] | None = None
    """Maximum projection for channel 2. Legacy: max_proj_chan2."""

    correlation_map_channel_2: NDArray[np.float32] | None = None
    """Correlation map for channel 2 detection. Legacy: Vcorr_chan2."""


@dataclass
class TimingData:
    """Stores pipeline timing information."""

    registration_time: float = 0.0
    """Registration step time in seconds. Legacy: timing['registration']."""

    two_step_registration_time: float = 0.0
    """Second registration step time in seconds. Legacy: timing['two_step_registration']."""

    registration_metrics_time: float = 0.0
    """Registration metrics computation time in seconds. Legacy: timing['registration_metrics']."""

    detection_time: float = 0.0
    """ROI detection time in seconds. Legacy: timing['detection']."""

    extraction_time: float = 0.0
    """Fluorescence extraction time in seconds. Legacy: timing['extraction']."""

    classification_time: float = 0.0
    """ROI classification time in seconds. Legacy: timing['classification']."""

    deconvolution_time: float = 0.0
    """Spike deconvolution time in seconds. Legacy: timing['deconvolution']."""

    detection_time_channel_2: float = 0.0
    """Channel 2 ROI detection time in seconds. Legacy: timing['detection_chan2']."""

    extraction_time_channel_2: float = 0.0
    """Channel 2 fluorescence extraction time in seconds. Legacy: timing['extraction_chan2']."""

    classification_time_channel_2: float = 0.0
    """Channel 2 ROI classification time in seconds. Legacy: timing['classification_chan2']."""

    deconvolution_time_channel_2: float = 0.0
    """Channel 2 spike deconvolution time in seconds. Legacy: timing['deconvolution_chan2']."""

    total_plane_time: float = 0.0
    """Total plane processing time in seconds. Legacy: timing['total_plane_runtime']."""

    date_processed: str = ""
    """ISO timestamp when processing completed. Legacy: date_processed."""

    python_version: str = python_version
    """Python version used for processing."""

    sl_suite2p_version: str = version
    """sl-suite2p version used for processing."""


@dataclass
class SingleDayRuntimeData(YamlConfig):
    """Aggregates all runtime data for a single plane.

    This class combines IO, registration, detection, and timing data into a single structure.

    Notes:
        NumPy arrays (images, registration offsets) are saved as separate .npy files in the output directory and
        loaded into memory during initialization via __post_init__. Path fields are converted to strings when saving
        to YAML and restored to Path objects when loading. When save() is called, all arrays are written to .npy files
        and their fields are set to None in the YAML representation.
    """

    output_path: Path | None = None
    """The path to the directory where runtime data and .npy files are stored."""

    io: IOData = field(default_factory=IOData)
    """The runtime data from the IO/binarization stage."""

    registration: RegistrationData = field(default_factory=RegistrationData)
    """The runtime data from the registration stage."""

    detection: DetectionData = field(default_factory=DetectionData)
    """The runtime data from the detection/extraction stage."""

    timing: TimingData = field(default_factory=TimingData)
    """The pipeline timing information."""

    def __post_init__(self) -> None:
        """Loads NumPy arrays from .npy files if output_path is set and arrays are None."""
        if self.output_path is None:
            return

        # Converts output_path to Path if it was loaded as a string from YAML.
        if isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)

        # Loads registration arrays from .npy files.
        self._load_registration_arrays()

        # Loads detection arrays from .npy files.
        self._load_detection_arrays()

    def _load_registration_arrays(self) -> None:
        """Loads registration arrays from .npy files in the output directory."""
        # Loads reference image.
        path = self.output_path / "reference_image.npy"
        if self.registration.reference_image is None and path.exists():
            self.registration.reference_image = np.load(path, allow_pickle=False).astype(np.float32)

        # Loads rigid registration offsets.
        path = self.output_path / "rigid_y_offsets.npy"
        if self.registration.rigid_y_offsets is None and path.exists():
            self.registration.rigid_y_offsets = np.load(path, allow_pickle=False).astype(np.float32)

        path = self.output_path / "rigid_x_offsets.npy"
        if self.registration.rigid_x_offsets is None and path.exists():
            self.registration.rigid_x_offsets = np.load(path, allow_pickle=False).astype(np.float32)

        path = self.output_path / "rigid_correlations.npy"
        if self.registration.rigid_correlations is None and path.exists():
            self.registration.rigid_correlations = np.load(path, allow_pickle=False).astype(np.float32)

        # Loads non-rigid registration offsets.
        path = self.output_path / "nonrigid_y_offsets.npy"
        if self.registration.nonrigid_y_offsets is None and path.exists():
            self.registration.nonrigid_y_offsets = np.load(path, allow_pickle=False).astype(np.float32)

        path = self.output_path / "nonrigid_x_offsets.npy"
        if self.registration.nonrigid_x_offsets is None and path.exists():
            self.registration.nonrigid_x_offsets = np.load(path, allow_pickle=False).astype(np.float32)

        path = self.output_path / "nonrigid_correlations.npy"
        if self.registration.nonrigid_correlations is None and path.exists():
            self.registration.nonrigid_correlations = np.load(path, allow_pickle=False).astype(np.float32)

    def _load_detection_arrays(self) -> None:
        """Loads detection arrays from .npy files in the output directory."""
        # Channel 1 arrays.
        path = self.output_path / "mean_image.npy"
        if self.detection.mean_image is None and path.exists():
            self.detection.mean_image = np.load(path, allow_pickle=False).astype(np.float32)

        path = self.output_path / "enhanced_mean_image.npy"
        if self.detection.enhanced_mean_image is None and path.exists():
            self.detection.enhanced_mean_image = np.load(path, allow_pickle=False).astype(np.float32)

        path = self.output_path / "maximum_projection.npy"
        if self.detection.maximum_projection is None and path.exists():
            self.detection.maximum_projection = np.load(path, allow_pickle=False).astype(np.float32)

        path = self.output_path / "correlation_map.npy"
        if self.detection.correlation_map is None and path.exists():
            self.detection.correlation_map = np.load(path, allow_pickle=False).astype(np.float32)

        # Channel 2 arrays.
        path = self.output_path / "mean_image_channel_2.npy"
        if self.detection.mean_image_channel_2 is None and path.exists():
            self.detection.mean_image_channel_2 = np.load(path, allow_pickle=False).astype(np.float32)

        path = self.output_path / "enhanced_mean_image_channel_2.npy"
        if self.detection.enhanced_mean_image_channel_2 is None and path.exists():
            self.detection.enhanced_mean_image_channel_2 = np.load(path, allow_pickle=False).astype(np.float32)

        path = self.output_path / "maximum_projection_channel_2.npy"
        if self.detection.maximum_projection_channel_2 is None and path.exists():
            self.detection.maximum_projection_channel_2 = np.load(path, allow_pickle=False).astype(np.float32)

        path = self.output_path / "correlation_map_channel_2.npy"
        if self.detection.correlation_map_channel_2 is None and path.exists():
            self.detection.correlation_map_channel_2 = np.load(path, allow_pickle=False).astype(np.float32)

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

        # Saves registration arrays as .npy files.
        self._save_registration_arrays(output_path)

        # Saves detection arrays as .npy files.
        self._save_detection_arrays(output_path)

        # Creates a deep copy for YAML serialization.
        yaml_copy = copy.deepcopy(self)

        # Converts Path fields to strings for YAML serialization.
        yaml_copy.output_path = str(output_path)  # type: ignore[assignment]
        yaml_copy._convert_paths_to_strings()

        # Sets array fields to None in the YAML representation.
        yaml_copy._nullify_arrays()

        # Saves the YAML file.
        file_path = output_path / "runtime_data.yaml"
        yaml_copy.to_yaml(file_path=file_path)

    def _save_registration_arrays(self, output_path: Path) -> None:
        """Saves registration arrays to .npy files."""
        if self.registration.reference_image is not None:
            np.save(output_path / "reference_image.npy", self.registration.reference_image, allow_pickle=False)

        if self.registration.rigid_y_offsets is not None:
            np.save(output_path / "rigid_y_offsets.npy", self.registration.rigid_y_offsets, allow_pickle=False)

        if self.registration.rigid_x_offsets is not None:
            np.save(output_path / "rigid_x_offsets.npy", self.registration.rigid_x_offsets, allow_pickle=False)

        if self.registration.rigid_correlations is not None:
            np.save(output_path / "rigid_correlations.npy", self.registration.rigid_correlations, allow_pickle=False)

        if self.registration.nonrigid_y_offsets is not None:
            np.save(output_path / "nonrigid_y_offsets.npy", self.registration.nonrigid_y_offsets, allow_pickle=False)

        if self.registration.nonrigid_x_offsets is not None:
            np.save(output_path / "nonrigid_x_offsets.npy", self.registration.nonrigid_x_offsets, allow_pickle=False)

        if self.registration.nonrigid_correlations is not None:
            np.save(
                output_path / "nonrigid_correlations.npy", self.registration.nonrigid_correlations, allow_pickle=False
            )

    def _save_detection_arrays(self, output_path: Path) -> None:
        """Saves detection arrays to .npy files."""
        # Channel 1 arrays.
        if self.detection.mean_image is not None:
            np.save(output_path / "mean_image.npy", self.detection.mean_image, allow_pickle=False)

        if self.detection.enhanced_mean_image is not None:
            np.save(output_path / "enhanced_mean_image.npy", self.detection.enhanced_mean_image, allow_pickle=False)

        if self.detection.maximum_projection is not None:
            np.save(output_path / "maximum_projection.npy", self.detection.maximum_projection, allow_pickle=False)

        if self.detection.correlation_map is not None:
            np.save(output_path / "correlation_map.npy", self.detection.correlation_map, allow_pickle=False)

        # Channel 2 arrays.
        if self.detection.mean_image_channel_2 is not None:
            np.save(output_path / "mean_image_channel_2.npy", self.detection.mean_image_channel_2, allow_pickle=False)

        if self.detection.enhanced_mean_image_channel_2 is not None:
            np.save(
                output_path / "enhanced_mean_image_channel_2.npy",
                self.detection.enhanced_mean_image_channel_2,
                allow_pickle=False,
            )

        if self.detection.maximum_projection_channel_2 is not None:
            np.save(
                output_path / "maximum_projection_channel_2.npy",
                self.detection.maximum_projection_channel_2,
                allow_pickle=False,
            )

        if self.detection.correlation_map_channel_2 is not None:
            np.save(
                output_path / "correlation_map_channel_2.npy",
                self.detection.correlation_map_channel_2,
                allow_pickle=False,
            )

    def _convert_paths_to_strings(self) -> None:
        """Converts all Path fields to strings for YAML serialization."""
        if self.io.registered_binary_path is not None:
            self.io.registered_binary_path = str(self.io.registered_binary_path)  # type: ignore[assignment]
        if self.io.raw_binary_path is not None:
            self.io.raw_binary_path = str(self.io.raw_binary_path)  # type: ignore[assignment]
        if self.io.registered_binary_path_channel_2 is not None:
            self.io.registered_binary_path_channel_2 = str(  # type: ignore[assignment]
                self.io.registered_binary_path_channel_2
            )
        if self.io.raw_binary_path_channel_2 is not None:
            self.io.raw_binary_path_channel_2 = str(self.io.raw_binary_path_channel_2)  # type: ignore[assignment]
        if self.io.output_directory is not None:
            self.io.output_directory = str(self.io.output_directory)  # type: ignore[assignment]

    def _nullify_arrays(self) -> None:
        """Sets all NumPy array fields to None for YAML serialization."""
        # Nullifies registration arrays.
        self.registration.reference_image = None
        self.registration.rigid_y_offsets = None
        self.registration.rigid_x_offsets = None
        self.registration.rigid_correlations = None
        self.registration.nonrigid_y_offsets = None
        self.registration.nonrigid_x_offsets = None
        self.registration.nonrigid_correlations = None

        # Nullifies detection arrays (channel 1).
        self.detection.mean_image = None
        self.detection.enhanced_mean_image = None
        self.detection.maximum_projection = None
        self.detection.correlation_map = None

        # Nullifies detection arrays (channel 2).
        self.detection.mean_image_channel_2 = None
        self.detection.enhanced_mean_image_channel_2 = None
        self.detection.maximum_projection_channel_2 = None
        self.detection.correlation_map_channel_2 = None

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


@dataclass
class SingleDayConfiguration(YamlConfig):
    """Aggregates the user-defined configuration parameters for the single-day suite2p pipeline.

    This class stores all user-configurable parameters that control how the pipeline processes data.
    These parameters are immutable during processing - the pipeline reads them but does not modify them.

    Notes:
        This class is based on the 'default_ops' dictionary from the original suite2p package. The default parameters
        are tuned for working with GCaMP6F fluorescence data recorded using 2-Photon Random Access Mesoscope (2P-RAM).

        For runtime data (computed by the pipeline), see SingleDayRuntimeData.
    """

    # Define the instances of each nested settings class as fields
    main: Main = field(default_factory=Main)
    """Stores global parameters that broadly define the suite2p single-day processing configuration."""
    file_io: FileIO = field(default_factory=FileIO)
    """Stores general I/O parameters that specify input data location, format, and working and output directories."""
    registration: Registration = field(default_factory=Registration)
    """Stores parameters for rigid registration, which is used to correct motion artifacts between frames by
    counter-shifting the entire frame."""
    one_photon_registration: OnePhotonRegistration = field(default_factory=OnePhotonRegistration)
    """Stores parameters for additional pre-registration processing used to improve the registration of 1-photon
    datasets."""
    non_rigid_registration: NonRigidRegistration = field(default_factory=NonRigidRegistration)
    """Stores parameters for non-rigid registration, which is used to improve motion registration in complex
    datasets."""
    roi_detection: ROIDetection = field(default_factory=ROIDetection)
    """Stores parameters for cell ROI detection and extraction."""
    signal_extraction: SignalExtraction = field(default_factory=SignalExtraction)
    """Stores parameters for extracting fluorescence signals from ROIs and surrounding neuropil regions."""
    spike_deconvolution: SpikeDeconvolution = field(default_factory=SpikeDeconvolution)
    """Stores parameters for deconvolving fluorescence signals to infer spike trains."""
    classification: Classification = field(default_factory=Classification)
    """Stores parameters for classifying detected ROIs as real cells or artifacts."""

    def save(self, file_path: Path) -> None:
        """Saves the configuration to a YAML file.

        Converts enum and Path fields to strings before serialization to ensure YAML compatibility.

        Args:
            file_path: The path to the .yaml file where to save the configuration data.
        """
        ensure_directory_exists(file_path)

        # Creates a deep copy to avoid modifying the original instance.
        yaml_copy = copy.deepcopy(self)

        # Converts enums to strings for YAML serialization.
        yaml_copy.file_io.input_format = str(yaml_copy.file_io.input_format)  # type: ignore[assignment]
        yaml_copy.spike_deconvolution.baseline_method = str(  # type: ignore[assignment]
            yaml_copy.spike_deconvolution.baseline_method
        )

        # Converts Path fields to strings for YAML serialization.
        if yaml_copy.file_io.data_path is not None:
            yaml_copy.file_io.data_path = str(yaml_copy.file_io.data_path)  # type: ignore[assignment]
        if yaml_copy.file_io.save_path is not None:
            yaml_copy.file_io.save_path = str(yaml_copy.file_io.save_path)  # type: ignore[assignment]
        if yaml_copy.classification.custom_classifier_path is not None:
            yaml_copy.classification.custom_classifier_path = str(  # type: ignore[assignment]
                yaml_copy.classification.custom_classifier_path
            )

        yaml_copy.to_yaml(file_path=file_path)

    @classmethod
    def load(cls, file_path: Path) -> SingleDayConfiguration:
        """Loads configuration from a YAML file.

        The FileIO.__post_init__ method automatically converts string input_format back to InputFormat enum.

        Args:
            file_path: The path to the .yaml configuration file.

        Returns:
            A SingleDayConfiguration instance populated with the loaded data.
        """
        return cls.from_yaml(file_path=file_path)


@dataclass
class RuntimeContext:
    """Combines configuration and runtime data for pipeline functions.

    Notes:
        This class provides a unified interface for pipeline functions to access both user configuration (immutable)
        and runtime data (computed by pipeline). It replaces the legacy ops dictionary pattern with a type-safe
        structure.
    """

    config: SingleDayConfiguration
    """The user configuration, which remains immutable during processing."""

    runtime: SingleDayRuntimeData
    """The runtime data, which is computed and updated by pipeline stages."""
