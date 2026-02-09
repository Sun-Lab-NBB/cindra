"""Provides user-defined configuration classes for the single-day (within-session) sl-suite2p pipeline."""

from __future__ import annotations

import copy
from enum import StrEnum
import json
from pathlib import Path
from dataclasses import field, dataclass

from ataraxis_base_utilities import console, ensure_directory_exists
from ataraxis_data_structures import YamlConfig


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
class AcquisitionParameters(YamlConfig):
    """Stores the data acquisition parameters used by the system that recorded the processed cell activity data.

    This dataclass describes the acquisition parameters of the input TIFF files, supporting both single-ROI
    (standard imaging) and multi-ROI (MROI line-scanning) data.

    Notes:
        For single-ROI data, only frame_rate, plane_number, and channel_number are required. For MROI data,
        additional fields describe the geometry of each ROI.

        The pipeline expects a suite2p_parameters.json file in the data directory containing these parameters. Use the
        is_mroi property to determine whether the data uses multi-ROI acquisition.
    """

    frame_rate: float
    """The acquisition frame rate in Hz. For multi-plane recordings, this is the volume rate (rate at which all
    planes are acquired), not the rate per plane."""

    plane_number: int = 1
    """The number of imaging planes acquired per volume. For single-plane recordings, this is 1."""

    channel_number: int = 1
    """The number of channels acquired per plane. Most recordings use either one or two channels. Currently, the
    processing only supports recordings with two or fewer channels."""

    roi_number: int = 1
    """The number of regions of interest (ROIs) acquired per plane. For standard imaging this is 1. For MROI
    line-scanning microscopes (e.g., 2-Photon Random Access Mesoscope), this can be greater than 1."""

    roi_lines: list[list[int]] = field(default_factory=list)
    """The line indices for each ROI in MROI acquisitions. Each inner list contains the row indices in the raw
    frame that belong to that ROI. The length of the outer list must equal roi_number. For single-ROI data, this
    field is empty."""

    roi_x_coordinates: list[int] = field(default_factory=list)
    """The x-coordinates (in pixels) for positioning each ROI in MROI acquisitions. These define the horizontal
    position of each ROI's top-left corner in the combined field of view. The length must equal roi_number. For
    single-ROI data, this field is empty."""

    roi_y_coordinates: list[int] = field(default_factory=list)
    """The y-coordinates (in pixels) for positioning each ROI in MROI acquisitions. These define the vertical
    position of each ROI's top-left corner in the combined field of view. The length must equal roi_number. For
    single-ROI data, this field is empty."""

    @property
    def is_mroi(self) -> bool:
        """Returns True if this acquisition uses multi-ROI mode (roi_number > 1)."""
        return self.roi_number > 1

    @property
    def virtual_plane_count(self) -> int:
        """Returns the total number of virtual planes (roi_number * plane_number).

        For single-ROI data, this equals plane_number. For MROI data, each ROI x plane combination becomes a
        separate virtual plane for processing.
        """
        return self.roi_number * self.plane_number

    @classmethod
    def from_json(cls, path: Path) -> AcquisitionParameters:
        """Loads acquisition parameters from a JSON file.

        Args:
            path: The path to the JSON file containing acquisition parameters.

        Returns:
            An AcquisitionParameters instance populated from the JSON file.

        Raises:
            FileNotFoundError: If the JSON file does not exist.
            ValueError: If required fields are missing. For single-ROI data, frame_rate, plane_number, and
                channel_number are required. For MROI data (roi_number > 1), roi_lines, roi_x_coordinates, and
                roi_y_coordinates are additionally required.
        """
        if not path.exists():
            message = f"Acquisition parameters file not found: {path}"
            console.error(message=message, error=FileNotFoundError)

        with path.open("r") as f:
            data = json.load(f)

        # Extracts frame_rate (required).
        frame_rate = data.get("frame_rate")
        if frame_rate is None:
            message = (
                f"Unable to extract the required field 'frame_rate' from the acquisition parameters file "
                f"located at {path}."
            )
            console.error(message=message, error=ValueError)

        # Extracts plane_number (required).
        plane_number = data.get("plane_number")
        if plane_number is None:
            message = (
                f"Unable to extract the required field 'plane_number' from the acquisition parameters file "
                f"located at {path}."
            )
            console.error(message=message, error=ValueError)

        # Extracts channel_number (required).
        channel_number = data.get("channel_number")
        if channel_number is None:
            message = (
                f"Unable to extract the required field 'channel_number' from the acquisition parameters file "
                f"located at {path}."
            )
            console.error(message=message, error=ValueError)

        # Extracts roi_number (defaults to 1 for single-ROI).
        roi_number = data.get("roi_number", 1)

        # For MROI data (roi_number > 1), validates that all MROI fields are present.
        if roi_number > 1:
            roi_lines = data.get("roi_lines")
            if roi_lines is None:
                message = (
                    f"Unable to extract the required field 'roi_lines' from the acquisition parameters file "
                    f"located at {path}."
                )
                console.error(message=message, error=ValueError)

            roi_x_coordinates = data.get("roi_x_coordinates")
            if roi_x_coordinates is None:
                message = (
                    f"Unable to extract the required field 'roi_x_coordinates' from the acquisition parameters "
                    f"file located at {path}."
                )
                console.error(message=message, error=ValueError)

            roi_y_coordinates = data.get("roi_y_coordinates")
            if roi_y_coordinates is None:
                message = (
                    f"Unable to extract the required field 'roi_y_coordinates' from the acquisition parameters "
                    f"file located at {path}."
                )
                console.error(message=message, error=ValueError)
        else:
            roi_lines = []
            roi_x_coordinates = []
            roi_y_coordinates = []

        return cls(
            frame_rate=frame_rate,
            plane_number=plane_number,
            channel_number=channel_number,
            roi_number=roi_number,
            roi_lines=roi_lines,
            roi_x_coordinates=roi_x_coordinates,
            roi_y_coordinates=roi_y_coordinates,
        )


@dataclass
class Main:
    """Stores the parameters that broadly affect the runtime and performance of the single-day pipeline as a whole."""

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
    """The threshold for determining whether ROIs from one channel correspond to ROIs or signals in the other channel.
    When one channel is functional and the other is structural, this threshold applies to intensity-based
    colocalization: ROIs are marked as colocalized if their inside-to-total intensity ratio in the structural channel
    exceeds this value. When both channels are functional, this threshold applies to spatial colocalization: ROIs are
    matched if their pixel overlap fraction exceeds this value."""

    tau: float = 0.4
    """The timescale of the sensor in seconds, used for computing the deconvolution kernel. The kernel is fixed to have
    this decay and is not fit to the data. The default value is optimized for GCaMP6f animals recorded with the
    Mesoscope and likely needs to be increased for most other use cases."""

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

    custom_classifier_path: Path | None = None
    """The absolute path to a custom classifier file used for ROI classification. When set, this classifier is used
    instead of the built-in classifier for both preclassification during detection and final classification after
    signal extraction. Leave as None to use the built-in classifier bundled with sl-suite2p."""

    def __post_init__(self) -> None:
        """Converts string custom_classifier_path to Path after YAML loading."""
        if isinstance(self.custom_classifier_path, str):
            self.custom_classifier_path = Path(self.custom_classifier_path) if self.custom_classifier_path else None

    def prepare_for_saving(self) -> None:
        """Converts Path fields to strings for YAML serialization."""
        if self.custom_classifier_path is not None:
            self.custom_classifier_path = str(self.custom_classifier_path)  # type: ignore[assignment]


@dataclass
class FileIO:
    """Stores the parameters that specify input data location, format, and output directories."""

    data_path: Path | None = None
    """The path to the root data directory containing the input TIFF files. The pipeline recursively searches this
    directory and all subdirectories for .tiff/.tif files to process."""

    save_path: Path | None = None
    """The path to the root output directory where to save the processing results. The pipeline automatically
    creates a 'suite2p' subdirectory under this path to store all output files."""

    ignored_file_names: list[str] = field(default_factory=list)
    """The list of file names to ignore when searching for and loading raw data. Any file whose name exactly matches
    one of the names in this is excluded from processing even if it has the correct extension and is located inside
    the input data directory."""

    def __post_init__(self) -> None:
        """Converts string paths to Path objects after YAML loading."""
        if isinstance(self.data_path, str):
            self.data_path = Path(self.data_path) if self.data_path else None
        if isinstance(self.save_path, str):
            self.save_path = Path(self.save_path) if self.save_path else None

    def prepare_for_saving(self) -> None:
        """Converts Path fields to strings for YAML serialization."""
        if self.data_path is not None:
            self.data_path = str(self.data_path)  # type: ignore[assignment]
        if self.save_path is not None:
            self.save_path = str(self.save_path)  # type: ignore[assignment]


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
    """The maximum allowed shift during registration, given as a fraction of the frame size (e.g., 0.1 indicates 10%).
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

    two_step_registration: bool = False
    """Determines whether to perform a two-step registration. This process consists of the initial registration
    (first step) followed by refinement (second step) registration. This procedure is helpful when working with low
    signal-to-noise data."""

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

    registration_metric_principal_components: int = 5
    """The number of Principal Components (PCs) used to compute the registration quality metrics. These metrics are
    not used by the processing pipeline but are useful for assessing registration quality via the GUI. Computing
    metrics is a fairly expensive operation that can take as long as the registration itself. The time to compute
    scales with the number of computed PCs, so it is recommended to keep this as low as feasible. Set to 0 to disable
    registration metrics computation entirely."""

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

    crop_to_soma: bool = True
    """Determines whether to crop dendritic regions from detected ROIs before computing classification features.
    When enabled, the algorithm analyzes the radial distribution of fluorescence from each ROI's centroid and
    excludes pixels beyond where fluorescence contribution drops significantly. This focuses classification
    on the cell body, improving accuracy for neurons with extensive dendritic arbors."""


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

    classification_threshold: float = 0.5
    """The classifier probability threshold used to classify ROIs after signal extraction. This is the minimum
    classifier confidence value (that the classified ROI is a cell) for the ROI to be labeled as a cell. ROIs with
    probabilities below this threshold are labeled as non-cells but are still retained in the output data."""

    batch_size: int = 500
    """The number of frames to process at the same time during fluorescence extraction. This controls memory usage
    during the extraction step. Larger values may improve throughput on fast storage but increase RAM consumption.
    This is independent of the registration batch size."""


@dataclass
class SpikeDeconvolution:
    """Stores parameters for deconvolving fluorescence signals to infer spike trains."""

    extract_spikes: bool = True
    """Determines whether to deconvolve spike activity from the extracted fluorescence traces. When disabled, the
    pipeline still computes neuropil-corrected fluorescence (F - coefficient * F_neuropil) but skips the deconvolution
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

    def prepare_for_saving(self) -> None:
        """Converts enum fields to strings for YAML serialization."""
        self.baseline_method = str(self.baseline_method)


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

    def save(self, file_path: Path) -> None:
        """Saves the configuration to a YAML file.

        Converts enum and Path fields to strings before serialization to ensure YAML compatibility.

        Args:
            file_path: The path to the .yaml file where to save the configuration data.
        """
        ensure_directory_exists(file_path)

        # Creates a deep copy to avoid modifying the original instance.
        yaml_copy = copy.deepcopy(self)

        # Prepares each child dataclass for YAML serialization.
        yaml_copy.main.prepare_for_saving()
        yaml_copy.file_io.prepare_for_saving()
        yaml_copy.spike_deconvolution.prepare_for_saving()

        yaml_copy.to_yaml(file_path=file_path)

    @classmethod
    def load(cls, file_path: Path) -> SingleDayConfiguration:
        """Loads configuration from a YAML file.

        Args:
            file_path: The path to the .yaml configuration file.

        Returns:
            A SingleDayConfiguration instance populated with the loaded data.
        """
        return cls.from_yaml(file_path=file_path)
