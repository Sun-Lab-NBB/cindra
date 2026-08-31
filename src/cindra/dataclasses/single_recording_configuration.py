"""Provides user-defined configuration classes for the single-recording (within-recording) processing pipeline."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path  # noqa: TC003 - needed at runtime for dacite deserialization
from dataclasses import field, dataclass

from ataraxis_base_utilities import console, ensure_directory_exists
from ataraxis_data_structures import YamlConfig


class PipelineType(StrEnum):
    """Defines the supported cindra processing pipeline types."""

    SINGLE_RECORDING = "single-recording"
    """The within-recording pipeline that processes a single recording (binarize, register, process, combine)."""

    MULTI_RECORDING = "multi-recording"
    """The across-recording pipeline that tracks and extracts ROIs across multiple recordings (discover, extract)."""


class BaselineMethod(StrEnum):
    """Defines the supported methods for computing baseline fluorescence before spike deconvolution."""

    MAXIMIN = "maximin"
    """Applies Gaussian smoothing followed by minimum and maximum filters over a sliding window, tracking the lower
    envelope of slow signal fluctuations."""

    CONSTANT = "constant"
    """Uses the global minimum of the Gaussian-smoothed trace as a single constant baseline for the entire recording."""

    CONSTANT_PERCENTILE = "constant_percentile"
    """Uses a low percentile of the trace as a robust constant baseline, ignoring outliers."""


def detect_pipeline_type(file_path: Path) -> PipelineType:  # noqa: RET503 - console.error is typed NoReturn.
    """Detects the pipeline type stored in the specified configuration YAML file.

    Reads the ``pipeline_type`` discriminator field from the configuration file without loading the full configuration
    dataclass.

    Args:
        file_path: The path to the configuration YAML file to inspect.

    Returns:
        The pipeline the configuration file's ``pipeline_type`` field names.

    Raises:
        FileNotFoundError: If the configuration file does not exist or does not have a '.yaml' extension.
        ValueError: If the file does not contain a recognized ``pipeline_type`` value.
    """
    if not file_path.exists() or file_path.suffix != ".yaml":
        message = (
            f"Unable to detect the pipeline type from the specified configuration file. Expected an existing "
            f"'.yaml' file, but received '{file_path}'."
        )
        console.error(message=message, error=FileNotFoundError)

    raw_type = _PipelineHeader.from_yaml(file_path=file_path).pipeline_type

    try:
        if raw_type is not None:
            return PipelineType(raw_type)
    except ValueError:
        pass

    expected = ", ".join(f"'{member.value}'" for member in PipelineType)
    message = (
        f"Unable to detect the pipeline type from the configuration file at '{file_path}'. The 'pipeline_type' "
        f"field is missing or has an unrecognized value '{raw_type}'. Expected one of: {expected}."
    )
    console.error(message=message, error=ValueError)


@dataclass(slots=True)
class RuntimeSettings:
    """Stores runtime behavior settings shared between single-recording and multi-recording processing pipelines."""

    display_progress_bars: bool = False
    """Determines whether to display progress bars for certain processing steps. Only enable this option when running
    all processing steps sequentially. Having this enabled when running multiple recordings or planes in parallel may
    interfere with properly communicating progress via the terminal."""


@dataclass
class AcquisitionParameters(YamlConfig):
    """Stores the data acquisition parameters used by the system that recorded the processed ROI activity data.

    Notes:
        For single-ROI data, only frame_rate, plane_number, and channel_number are required. For MROI data, additional
        fields describe the geometry of each ROI.

        The pipeline expects a cindra_parameters.json file in the data directory containing these parameters.
    """

    frame_rate: float
    """The acquisition frame rate in Hz. For multi-plane recordings, this is the volume rate (rate at which all planes
    are acquired), not the rate per plane."""

    plane_number: int = 1
    """The number of imaging planes acquired per volume. For single-plane recordings, this is 1."""

    channel_number: int = 1
    """The number of channels acquired per plane. Most recordings use either one or two channels. Currently, the
    processing only supports recordings with two or fewer channels."""

    roi_number: int = 1
    """The number of regions of interest (ROIs) acquired per plane. For standard imaging this is 1. For MROI
    line-scanning microscopes (e.g., 2-Photon Random Access Mesoscope), this can be greater than 1."""

    roi_lines: tuple[tuple[int, ...], ...] = ()
    """The line indices for each ROI in MROI acquisitions. Each inner tuple contains the row indices in the raw frame
    that belong to that ROI. The length of the outer tuple must equal roi_number. For single-ROI data, this field is
    empty."""

    roi_x_coordinates: tuple[int, ...] = ()
    """The x-coordinates (in pixels) for positioning each ROI in MROI acquisitions. These define the horizontal position
    of each ROI's top-left corner in the combined field of view. The length must equal roi_number. For single-ROI data,
    this field is empty."""

    roi_y_coordinates: tuple[int, ...] = ()
    """The y-coordinates (in pixels) for positioning each ROI in MROI acquisitions. These define the vertical position
    of each ROI's top-left corner in the combined field of view. The length must equal roi_number. For single-ROI data,
    this field is empty."""

    @property
    def is_mroi(self) -> bool:
        """Returns True if this acquisition uses multi-ROI mode (roi_number > 1)."""
        return self.roi_number > 1

    @property
    def virtual_plane_count(self) -> int:
        """Returns the total number of virtual planes (roi_number * plane_number), where each ROI x plane combination
        becomes a separate virtual plane for processing.
        """
        return self.roi_number * self.plane_number


@dataclass(slots=True)
class Main:
    """Stores the parameters that broadly affect the single-recording pipeline processing behavior.

    Notes:
        For runtime behavior settings shared with the multi-recording pipeline (progress bars), see RuntimeSettings.
        Worker counts are explicit API parameters resolved through ``cindra.orchestration``.
    """

    two_channels: bool = False
    """Determines whether the imaging data contains two channels per plane. When True, the algorithm expects images from
    both channels of the same plane to be saved sequentially (e.g.: plane 1 channel 1, plane 1 channel 2, plane 2
    channel 1, etc.)."""

    first_channel_functional: bool = True
    """Determines whether the first channel is used for ROI detection and signal extraction. This field is only
    applicable when two_channels is True. When both first_channel_functional and second_channel_functional are True, the
    pipeline performs independent ROI detection on both channels."""

    second_channel_functional: bool = False
    """Determines whether the second channel is used for ROI detection and signal extraction. This field is only
    applicable when two_channels is True. When both first_channel_functional and second_channel_functional are True, the
    pipeline performs independent ROI detection on both channels."""

    tau: float = 0.4
    """The timescale of the sensor in seconds, used for computing the deconvolution kernel. The kernel is fixed to have
    this decay and is not fit to the data. The default value is optimized for GCaMP6f animals recorded with the
    Mesoscope and likely needs to be increased for most other use cases."""

    ignored_flyback_planes: tuple[int, ...] = ()
    """The flyback plane indices to ignore when processing the data. Flyback planes typically contain no valid imaging
    data, so it is common to exclude them from processing."""

    custom_classifier_path: Path | None = None
    """The absolute path to a custom classifier file used for ROI classification. When set, this classifier is used
    instead of the built-in classifier for both preclassification during detection and final classification after signal
    extraction. Leave as None to use the built-in classifier bundled with cindra."""


@dataclass(slots=True)
class FileIO:
    """Stores the parameters that specify input data location, format, and output directories."""

    data_path: Path | None = None
    """The path to the root data directory containing the input TIFF files. The pipeline recursively searches this
    directory and all subdirectories for the cindra_parameters.json file, then loads the .tiff/.tif files from the
    single directory that holds it. TIFF discovery inside that directory is not recursive."""

    output_path: Path | None = None
    """The path to the root output directory where processing results are saved. This field is required for pipeline
    execution and must be explicitly configured before running any processing step. The pipeline creates a 'cindra'
    subdirectory under this path to store all output files."""

    ignored_file_names: tuple[str, ...] = ()
    """The file names, given without their extension, to ignore when searching for and loading raw data. Any file whose
    stem (the name with the extension stripped) exactly matches one of the entries in this tuple is excluded from
    processing even if it has the correct extension and is located inside the input data directory."""

    repeat_binarization: bool = False
    """Determines whether to repeat the binarization step when processing. When True, the pipeline re-runs TIFF to
    binary conversion using the data_path from the current configuration, even if binary files already exist. A
    conversion replaces every plane binary, so it first deletes the registration, detection, and extraction outputs of
    every plane directory the output root holds, the recording's combined dataset, and the ROI selections every
    multi-recording dataset holds for this recording. It also resets each plane's recorded runtime sections. When
    False (default), an existing binary is reused when it carries no write marker, is sized to the frame geometry
    recorded for its plane, and is accompanied by the second channel binary a two-channel recording declares.
    Binarization refuses a binary that fails any of those checks and names this parameter as the remedy, because
    enabling it rebuilds every plane binary from the source TIFF files."""


@dataclass(slots=True)
class Registration:
    """Stores parameters for rigid registration, which is used to correct motion artifacts between frames by
    counter-shifting the entire frame.
    """

    repeat_registration: bool = False
    """Determines whether to re-register data that appears to already be registered. When False, the pipeline skips
    registration if the data is already registered. When True, the pipeline re-registers the data regardless of its
    current registration state."""

    align_by_first_channel: bool = True
    """Determines whether to use the first channel for frame alignment (registration). When False, the second channel is
    used instead. If the recording features both a functional and non-functional channel, it is recommended to use the
    non-functional channel for alignment. This field is only applicable when two_channels is True in the Main
    configuration."""

    reference_frame_count: int = 500
    """The number of frames to use to compute the reference image. During registration, each frame is registered to the
    reference image to remove motion artifacts. The algorithm automatically selects the most stable (correlated) set of
    frames when computing the reference image."""

    batch_size: int = 100
    """The number of frames to keep in memory at the same time when registering them to the reference image. When
    processing data on fast (NVME) drives, increasing this parameter has minimal benefits and results in undue RAM use
    overhead. On slow drives, increasing this number may result in faster runtime, at the expense of increased RAM
    use."""

    maximum_offset_fraction: float = 0.1
    """The maximum allowed offset during registration, given as a fraction of the frame size (e.g., 0.1 indicates 10%).
    This determines how much the algorithm is allowed to offset the entire frame to align it to the reference image."""

    spatial_smoothing_sigma: float = 1.15
    """The standard deviation (in pixels) of the Gaussian filter used to spatially smooth the phase correlation surface
    between the reference image and each processed frame. Smoothing helps reduce noise in the correlation surface,
    improving the accuracy of sub-pixel offset detection. Higher values produce more smoothing but may reduce precision
    for detecting small offsets."""

    temporal_smoothing_sigma: float = 0.0
    """The standard deviation (in frames) of the Gaussian filter used to temporally smooth the phase correlation surface
    across consecutive frames. This reduces frame-to-frame noise in correlation values and can improve registration
    stability for noisy recordings. Setting this to 0.0 disables temporal smoothing."""

    two_step_registration: bool = False
    """Determines whether to perform a two-step registration. This process consists of the initial registration (first
    step) followed by refinement (second step) registration. This procedure is helpful when working with low
    signal-to-noise data."""

    gpu_batch_size: int = 0
    """The number of frames to keep in device memory at the same time while registration runs on a CUDA device. This
    overrides batch_size for the alignment pass on that path, where the device memory budget bounds the batch instead of
    the host RAM budget, while the secondary channel pass reads batch_size. Setting this to 0 uses batch_size on the
    device as well."""

    bad_frame_threshold: float = 1.0
    """The threshold for identifying frames with excessive motion or poor correlation quality. The algorithm computes a
    ratio of motion deviation to phase correlation quality for each frame. Frames exceeding this threshold (scaled by
    100 internally) are marked as 'bad' and excluded when computing the valid pixel region (valid_y_range,
    valid_x_range) after registration. This prevents a few frames with extreme motion from unnecessarily shrinking the
    usable field of view. Bad frames may also be excluded during movie binning for ROI detection. Lower values are more
    strict and exclude more frames."""

    normalize_frames: bool = True
    """Determines whether to clip pixel intensities to the 1st-99th percentile range during registration. This removes
    extreme outlier pixels from both the reference image and each frame before computing phase correlation, improving
    offset detection accuracy by reducing the influence of anomalously bright or dark pixels."""

    registration_metric_principal_components: int = 5
    """The number of Principal Components (PCs) used to compute the registration quality metrics. These metrics are not
    used by the processing pipeline but are useful for assessing registration quality via the GUI. Computing metrics is
    a fairly expensive operation that can take as long as the registration itself. The time to compute scales with the
    number of computed PCs, so it is recommended to keep this as low as feasible. Set to 0 to disable registration
    metrics computation entirely."""

    compute_bidirectional_phase_offset: bool = False
    """Determines whether to compute the bidirectional phase offset for misaligned line scanning in two-photon
    recordings. This correction addresses misalignment between odd and even scan lines caused by bidirectional resonant
    scanning. Most recording software (including ScanImage) handles this correction during acquisition, so this option
    is rarely needed for properly configured systems."""

    bidirectional_phase_offset_override: int = 0
    """Manual override for the bidirectional phase offset in line scanning 2-photon recordings. If set to any value
    besides 0, this offset is used instead of computing it automatically. If set to 0 and
    compute_bidirectional_phase_offset is True, the pipeline estimates the offset automatically from the initial
    reference frames."""


@dataclass(slots=True)
class OnePhotonRegistration:
    """Stores parameters for additional pre-registration processing used to improve the registration of 1-photon
    datasets.
    """

    enabled: bool = False
    """Determines whether to perform high-pass spatial filtering and tapering to improve one-photon image registration.
    For two-photon datasets, this should be set to False."""

    spatial_highpass_window: int = 42
    """The window size, in pixels, for spatial high-pass filtering. This filter removes low-frequency spatial variations
    such as uneven illumination that are common in one-photon imaging. The filter subtracts a spatially smoothed version
    of the image (using this window size) from the original, preserving only high-frequency features useful for
    registration."""

    pre_smoothing_sigma: float = 0.0
    """The window size, in pixels, of the uniform (box) filter applied before spatial high-pass filtering. The value is
    cast to an integer window and must be even, because odd windows are rejected by apply_spatial_smoothing(). This
    reduces high-frequency noise that would otherwise be amplified by the high-pass filter. Setting this to 0.0 disables
    pre-smoothing."""

    edge_taper_pixels: float = 40.0
    """The sigmoid falloff scale, in pixels, of the edge taper applied at image borders. The taper begins roughly 2 *
    this value inward from each edge and fades border pixels toward the image mean (not to zero) to prevent edge
    artifacts during FFT-based phase correlation. Larger values provide smoother transitions but reduce the usable image
    area."""


@dataclass(slots=True)
class NonrigidRegistration:
    """Stores parameters for nonrigid registration, which is used to improve motion registration in complex datasets by
    dividing frames into subregions and shifting each subregion independently of other subregions.
    """

    enabled: bool = True
    """Determines whether to perform nonrigid registration to correct for local motion and deformation. This is
    primarily used for correcting non-uniform motion."""

    block_size: tuple[int, int] = (128, 128)
    """The block size, in pixels, for nonrigid registration, defining the dimensions of subregions used in the
    correction. It is recommended to keep this size a power of 2 and/or 3 for more efficient FFT computation. During
    processing, each frame is tiled with blocks of these dimensions that overlap by approximately 50%, and the
    registration is applied to each block independently. A dimension that matches or exceeds the corresponding frame
    dimension collapses to a single block spanning the whole frame."""

    signal_to_noise_threshold: float = 1.2
    """The signal-to-noise ratio threshold below which the block's phase correlation surface receives additional
    smoothing from its neighboring blocks before the block offset is estimated. Block offsets are never rejected by this
    threshold. Higher values simply apply more smoothing. Typical values range from 1.0 to 1.5."""

    maximum_block_offset: float = 5.0
    """The maximum allowed offset, in pixels, for each block relative to the rigid registration offset."""


@dataclass(slots=True)
class ROIDetection:
    """Stores parameters for Region of Interest (ROI) detection."""

    enabled: bool = True
    """Determines whether to perform ROI detection. When disabled, the plane's whole processing stage is skipped, so no
    fluorescence traces are extracted, no ROIs are classified, and no spikes are deconvolved."""

    preclassification_threshold: float = 0.0
    """The classifier probability threshold used to pre-filter ROIs before signal extraction. This is the minimum
    classifier confidence value (that the classified ROI is a cell) for the ROI to be processed further. Setting this to
    0.0 keeps all detected ROIs, which is the default because the filter runs before any fluorescence is extracted and
    therefore judges a region on its shape alone, without the skewness of its trace. Small regions carry a low
    normalized pixel count and are discarded on that evidence, and the discard is permanent, so no later classification
    threshold can recover them. Raising this value trades recall for a smaller extraction workload."""

    threshold_scaling: float = 1.0
    """The scaling factor for the ROI detection threshold. The final threshold multiplies this value by a fixed base
    multiplier of 5.0 and by the selected pyramid scale index, which is clamped to a minimum of 1. The product is then
    multiplied by a recording-length factor, the binned frame count divided by 1200, also clamped to a minimum of 1.
    Higher values require ROIs to stand out more distinctly from background noise, resulting in fewer but more confident
    detections. Lower values detect more ROIs but may include false positives. The default of 1.0 leaves the base
    multiplier unscaled."""

    spatial_highpass_window: int = 25
    """The window size, in pixels, for spatial high-pass filtering used during neuropil subtraction. The algorithm
    subtracts a spatially smoothed version of each frame (using this window size) to remove diffuse neuropil
    fluorescence and isolate cell bodies."""

    maximum_overlap: float = 0.75
    """The maximum allowed fraction of an ROI's pixels that may be shared with any other ROI. ROIs are evaluated in
    reverse detection order, so when the fraction is exceeded the later-detected ROI is discarded, biasing retention
    toward earlier-detected (typically higher-quality) ROIs. Lower values enforce stricter separation between detected
    ROIs."""

    temporal_highpass_window: int = 100
    """The window size, in frames, for temporal high-pass filtering applied before ROI detection. This removes slow
    fluorescence drifts (such as photobleaching or baseline changes) by subtracting a running mean computed over this
    window. Larger values preserve slower transients but may retain more drift artifacts."""

    maximum_iterations: int = 50
    """The iteration scaling factor for ROI extraction. The algorithm detects ROIs one at a time, subtracting each
    detected ROI's contribution before searching for the next. The actual iteration limit is this value multiplied by
    250 internally (e.g., 50 allows up to 12,500 iterations). Higher values allow detecting more ROIs but increase
    processing time."""

    maximum_binned_frames: int = 5000
    """The maximum number of time-binned frames used for ROI detection. Temporal binning averages consecutive frames to
    improve signal-to-noise ratio for detection. The bin size is computed to produce at most this many binned frames, so
    a higher value keeps more binned frames, each averaging fewer source frames, at the cost of increased memory usage
    and processing time."""

    denoise: bool = False
    """Determines whether to apply PCA-based denoising to the binned movie before ROI detection. This can improve
    detection in noisy recordings by removing uncorrelated noise while preserving spatially coherent signals."""

    crop_to_soma: bool = True
    """Determines whether to crop dendritic regions from detected ROIs before computing classification features. When
    enabled, the algorithm analyzes the radial distribution of fluorescence from each ROI's centroid and excludes pixels
    beyond where fluorescence contribution drops significantly. This focuses classification on the cell body, improving
    accuracy for neurons with extensive dendritic arbors."""


@dataclass(slots=True)
class SignalExtraction:
    """Stores parameters for extracting fluorescence signals from ROIs and surrounding neuropil regions."""

    extract_neuropil: bool = True
    """Determines whether to extract neuropil activity. If disabled, neuropil fluorescence is assumed to be zero during
    spike deconvolution."""

    allow_overlap: bool = False
    """Determines whether to include overlapping pixels (shared by multiple ROIs) in signal extraction. When disabled,
    pixels belonging to multiple ROIs are excluded from all of them to prevent signal contamination between neighboring
    ROIs. Enable this only if ROIs are sparse and overlap is minimal."""

    minimum_neuropil_pixels: int = 350
    """The minimum number of pixels required for each neuropil mask. The algorithm expands outward from the cell border
    until it accumulates at least this many non-cell pixels. Larger values provide more stable neuropil estimates but
    may include pixels from distant regions with different neuropil characteristics."""

    inner_neuropil_border_radius: int = 2
    """The width, in pixels, of the exclusion zone between the cell ROI and its neuropil mask. This gap prevents
    contamination of the neuropil signal by the cell's own fluorescence. Larger values provide better separation but
    reduce the neuropil sampling area near the cell."""

    cell_probability_percentile: int = 50
    """The percentile threshold for classifying pixels as belonging to a cell versus neuropil. Each pixel has a
    probability weight indicating how likely it belongs to a cell. Pixels with weights above this percentile (computed
    locally) are excluded from neuropil masks. Higher values are more permissive, including more pixels in neuropil
    masks but risking cell contamination."""

    classification_threshold: float = 0.5
    """The classifier probability threshold used to classify ROIs after signal extraction. This is the minimum
    classifier confidence value (that the classified ROI is a cell) for the ROI to be labeled as a cell. ROIs with
    probabilities below this threshold are labeled as non-cells but are still retained in the output data."""

    batch_size: int = 500
    """The number of frames to process at the same time during fluorescence extraction. This controls memory usage
    during the extraction step. Larger values may improve throughput on fast storage but increase RAM consumption. This
    is independent of the registration batch size. The same value is reused as the number of ROIs per batch during OASIS
    spike deconvolution, where it bounds the four (batch, frames) workspace arrays that stage allocates."""

    colocalization_threshold: float = 0.65
    """The threshold for determining whether ROIs from one channel correspond to ROIs or signals in the other channel.
    When one channel is functional and the other is structural, this threshold applies to intensity-based
    colocalization: ROIs are marked as colocalized if their inside-to-total intensity ratio in the structural channel
    exceeds this value. When both channels are functional, this threshold applies to spatial colocalization: ROIs are
    matched if their pixel overlap fraction exceeds this value."""


@dataclass(slots=True)
class SpikeDeconvolution:
    """Stores parameters for deconvolving fluorescence signals to infer spike trains."""

    extract_spikes: bool = True
    """Determines whether to deconvolve spike activity from the extracted fluorescence traces. When disabled, the
    pipeline still extracts the raw cell and neuropil fluorescence, but both the neuropil-corrected
    (baseline-subtracted) traces and the spike traces are filled with zeros instead of being computed."""

    neuropil_coefficient: float = 0.7
    """The scaling factor applied to neuropil fluorescence before subtracting it from cell fluorescence. The corrected
    signal is computed as F_corrected = F_cell - coefficient * F_neuropil. Values typically range from 0.5 to 1.0, with
    0.7 being a common default. Higher values apply stronger neuropil correction but risk over-subtracting signal from
    cells with weak neuropil contamination."""

    baseline_method: BaselineMethod | str = BaselineMethod.MAXIMIN
    """The method for computing baseline fluorescence to subtract before deconvolution. See BaselineMethod enumeration
    for available options: MAXIMIN tracks the lower envelope using sliding window filters, CONSTANT uses the global
    minimum, and CONSTANT_PERCENTILE uses a low percentile as a robust constant baseline."""

    baseline_window: float = 60.0
    """The size of the sliding window, in seconds, for the 'maximin' baseline method. The minimum and maximum filters
    operate over this window to track slow baseline drifts while ignoring fast transients. Larger windows produce
    smoother baselines but may fail to track rapid baseline changes."""

    baseline_sigma: float = 10.0
    """The standard deviation, in frames, of the Gaussian filter applied before baseline computation. Used by both
    'maximin' and 'constant' methods to smooth the trace before finding minima. Larger values produce more aggressive
    smoothing."""

    baseline_percentile: float = 8.0
    """The percentile of trace activity used as baseline for the 'constant_percentile' method. Lower values (e.g., 8)
    select points near the trace minimum, providing a robust estimate that ignores outliers. Only used when
    baseline_method is set to CONSTANT_PERCENTILE."""


@dataclass
class SingleRecordingConfiguration(YamlConfig):
    """Aggregates the user-defined configuration parameters for the single-recording cindra pipeline.

    The pipeline reads these parameters and treats them as immutable for the duration of processing.

    Notes:
        Derives from the 'default_ops' dictionary of the original suite2p package. The default parameters are tuned
        for working with GCaMP6F fluorescence data recorded using 2-Photon Random Access Mesoscope (2P-RAM).

        For runtime data (computed by the pipeline), see SingleRecordingRuntimeData.
    """

    pipeline_type: PipelineType = field(default=PipelineType.SINGLE_RECORDING, init=False)
    """Identifies this configuration as a single-recording pipeline configuration."""
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    """Stores runtime behavior settings shared with the multi-recording pipeline (progress bar display)."""
    main: Main = field(default_factory=Main)
    """Stores global parameters that broadly define the cindra single-recording processing configuration."""
    file_io: FileIO = field(default_factory=FileIO)
    """Stores general I/O parameters that specify the input data location, the ignored input file names, and the output
    directory."""
    registration: Registration = field(default_factory=Registration)
    """Stores parameters for rigid registration, which is used to correct motion artifacts between frames by
    counter-shifting the entire frame."""
    one_photon_registration: OnePhotonRegistration = field(default_factory=OnePhotonRegistration)
    """Stores parameters for additional pre-registration processing used to improve the registration of 1-photon
    datasets."""
    nonrigid_registration: NonrigidRegistration = field(default_factory=NonrigidRegistration)
    """Stores parameters for nonrigid registration, which is used to improve motion registration in complex datasets."""
    roi_detection: ROIDetection = field(default_factory=ROIDetection)
    """Stores parameters for ROI detection and extraction."""
    signal_extraction: SignalExtraction = field(default_factory=SignalExtraction)
    """Stores parameters for extracting fluorescence signals from ROIs and surrounding neuropil regions."""
    spike_deconvolution: SpikeDeconvolution = field(default_factory=SpikeDeconvolution)
    """Stores parameters for deconvolving fluorescence signals to infer spike trains."""

    def save(self, file_path: Path) -> None:
        """Saves the configuration to a YAML file.

        Args:
            file_path: The path to the .yaml file in which to save the configuration data.
        """
        ensure_directory_exists(path=file_path, is_file=True)
        self.to_yaml(file_path=file_path)

    @classmethod
    def load(cls, file_path: Path) -> SingleRecordingConfiguration:
        """Loads configuration from a YAML file.

        Args:
            file_path: The path to the .yaml configuration file.

        Returns:
            The single-recording configuration the YAML file stores.
        """
        return cls.from_yaml(file_path=file_path)


@dataclass
class _PipelineHeader(YamlConfig):
    """Stores the raw pipeline type discriminator read from a configuration YAML file.

    Extra keys present in the YAML are silently discarded by dacite during deserialization, making this class safe to
    load against any cindra configuration file. The field defaults to None so that a missing key is distinguishable from
    a valid value.
    """

    pipeline_type: str | None = None
    """The raw pipeline type string read from the YAML file, or None if the key is absent."""
