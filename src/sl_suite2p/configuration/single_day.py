"""Provides configuration and runtime data classes for the single-day (within-session) sl-suite2p pipeline."""

from __future__ import annotations

import copy
from enum import StrEnum
import json
from typing import TYPE_CHECKING
from pathlib import Path
from dataclasses import field, dataclass

import numpy as np
from natsort import natsorted
from ataraxis_base_utilities import console, ensure_directory_exists
from ataraxis_data_structures import YamlConfig

from ..version import version, python_version

if TYPE_CHECKING:
    from numpy.typing import NDArray


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

    spatial_scale: int = 0
    """The optimal spatial scale, in pixels, for the processed data. This is used to adjust detection sensitivity.
    Setting this to 0 forces the algorithm to determine this value automatically. Values above 0 are applied in
    increments of 6 pixels (1 -> 6 pixels, 2 -> 12 pixels, etc.)."""

    cell_diameter: int = 0
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

    def prepare_for_saving(self) -> None:
        """Converts Path fields to strings for YAML serialization."""
        if self.custom_classifier_path is not None:
            self.custom_classifier_path = str(self.custom_classifier_path)  # type: ignore[assignment]


@dataclass
class IOData:
    """Stores runtime data from the IO/binarization stage."""

    frame_height: int = 0
    """The height of each frame in pixels (Y dimension of the imaging field of view)."""

    frame_width: int = 0
    """The width of each frame in pixels (X dimension of the imaging field of view)."""

    frame_count: int = 0
    """The total number of frames written to the binary file during binarization."""

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

    mroi_lines: list[int] = field(default_factory=list)
    """The list of scan line indices used for extracting this ROI from raw multi-ROI data. Only used for MROI
    recordings."""

    plane_index: int | None = None
    """The zero-based index identifying this plane's position in a multi-plane volumetric recording."""

    def __post_init__(self) -> None:
        """Converts string paths to Path objects after YAML loading."""
        if isinstance(self.registered_binary_path, str):
            self.registered_binary_path = Path(self.registered_binary_path) if self.registered_binary_path else None
        if isinstance(self.registered_binary_path_channel_2, str):
            self.registered_binary_path_channel_2 = (
                Path(self.registered_binary_path_channel_2) if self.registered_binary_path_channel_2 else None
            )
        if isinstance(self.output_directory, str):
            self.output_directory = Path(self.output_directory) if self.output_directory else None

    def prepare_for_saving(self) -> None:
        """Converts Path fields to strings for YAML serialization."""
        if self.registered_binary_path is not None:
            self.registered_binary_path = str(self.registered_binary_path)  # type: ignore[assignment]
        if self.registered_binary_path_channel_2 is not None:
            self.registered_binary_path_channel_2 = str(  # type: ignore[assignment]
                self.registered_binary_path_channel_2
            )
        if self.output_directory is not None:
            self.output_directory = str(self.output_directory)  # type: ignore[assignment]


@dataclass
class RegistrationData:
    """Stores runtime data from the registration stage."""

    valid_y_range: list[int] = field(default_factory=lambda: [0, 0])
    """The valid Y pixel range [start, end] defining the usable recording region after border cropping."""

    valid_x_range: list[int] = field(default_factory=lambda: [0, 0])
    """The valid X pixel range [start, end] defining the usable recording region after border cropping."""

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
        self.valid_y_range = [0, 0]
        self.valid_x_range = [0, 0]
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
        save_dict: dict[str, NDArray[np.float32] | NDArray[np.int32]] = {}

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
    """Stores runtime data from the detection/extraction stage."""

    spatial_scale: float = 0.0
    """The estimated spatial scale of the recording in pixels, used for automatic cell diameter detection."""

    cell_diameter: int = 0
    """The cell diameter in pixels, either computed automatically from spatial scale or specified by the user."""

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
    (always present after detection), optional extraction properties (added during signal extraction), and optional
    multi-plane/multi-day properties (added during combined view generation or cross-session tracking).

    Notes:
        This dataclass replaces the legacy dictionary-based stat.npy format.
    """

    # Core pixel data (required, from detection).
    y_pixels: NDArray[np.uint32]
    """The y-coordinates (row indices) of all pixels belonging to this ROI."""

    x_pixels: NDArray[np.uint32]
    """The x-coordinates (column indices) of all pixels belonging to this ROI."""

    pixel_weights: NDArray[np.float32]
    """The spatial filter weights (lambda values) for each pixel, indicating contribution to the ROI signal."""

    centroid: list[float]
    """The median [y, x] position of the ROI, representing its approximate center."""

    footprint: int
    """The spatial scale (hop size) used during sparse detection for this ROI."""

    # Shape statistics (required, from roi_stats computation).
    mean_r_squared: float
    """The normalized mean R-squared value measuring ROI compactness."""

    mean_r_squared_baseline: float
    """The unnormalized mean R-squared baseline value."""

    compactness: float
    """The normalized compactness ratio (mean_r_squared / mean_r_squared_baseline)."""

    solidity: float
    """The ratio of soma pixels to convex hull area, measuring how solid/filled the ROI is."""

    pixel_count: int
    """The total number of pixels in the complete ROI."""

    soma_pixel_count: int
    """The number of pixels in the soma-cropped region of the ROI."""

    soma_mask: NDArray[np.bool_]
    """The boolean mask indicating which pixels belong to the soma region."""

    overlap_mask: NDArray[np.bool_]
    """The boolean mask indicating which pixels overlap with other ROIs."""

    radius: float
    """The fitted ellipse radius representing the approximate ROI size."""

    aspect_ratio: float
    """The ratio of ellipse axes, indicating ROI elongation."""

    normalized_pixel_count: float
    """The pixel count normalized by expected cell size (soma region only)."""

    normalized_pixel_count_full: float
    """The pixel count normalized by expected cell size (full ROI)."""

    # Optional extraction data (added during signal extraction).
    skewness: float | None = None
    """The skewness of the baseline-subtracted fluorescence time series."""

    standard_deviation: float | None = None
    """The standard deviation of the baseline-subtracted fluorescence time series."""

    neuropil_mask: NDArray[np.bool_] | None = None
    """The boolean mask indicating pixels used for neuropil signal extraction."""

    # Multi-plane data. The plane_index should be set from IOData.plane_index during ROI creation.
    plane_index: int = 0
    """The index of the imaging plane this ROI belongs to in multi-plane recordings."""

    # Multi-day tracking data. Zero values indicate the ROI has not been processed by multi-day tracking.
    cluster_id: int = 0
    """The multi-day cell cluster ID. Zero indicates unclustered, positive values indicate cluster membership."""

    raveled_pixels: NDArray[np.int32] | None = None
    """The raveled (flattened) pixel indices in the deformed multi-day visual space."""

    session_count: int = 0
    """The number of sessions in which this cell was detected during multi-day tracking."""

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
        all_soma_mask = np.concatenate([roi.soma_mask for roi in roi_list])
        all_overlap_mask = np.concatenate([roi.overlap_mask for roi in roi_list])

        # Stores scalar fields as 1D arrays using appropriate types: uint16 for small non-negative integers,
        # uint32 for larger counts, and float32 for real-valued measurements.
        centroids = np.array([roi.centroid for roi in roi_list], dtype=np.float32)
        footprints = np.array([roi.footprint for roi in roi_list], dtype=np.uint16)
        mean_r_squared = np.array([roi.mean_r_squared for roi in roi_list], dtype=np.float32)
        mean_r_squared_baseline = np.array([roi.mean_r_squared_baseline for roi in roi_list], dtype=np.float32)
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

        # Builds the save dictionary with required fields.
        save_dict = {
            "pixel_counts": pixel_counts,
            "y_pixels": all_y_pixels,
            "x_pixels": all_x_pixels,
            "pixel_weights": all_pixel_weights,
            "soma_mask": all_soma_mask,
            "overlap_mask": all_overlap_mask,
            "centroids": centroids,
            "footprints": footprints,
            "mean_r_squared": mean_r_squared,
            "mean_r_squared_baseline": mean_r_squared_baseline,
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
        }

        # Handles optional variable-length neuropil_mask arrays. Uses uint32 for counts since they are always
        # non-negative and can be large.
        has_neuropil = [roi.neuropil_mask is not None for roi in roi_list]
        if any(has_neuropil):
            neuropil_counts = np.array(
                [len(roi.neuropil_mask) if roi.neuropil_mask is not None else 0 for roi in roi_list], dtype=np.uint32
            )
            neuropil_masks = [roi.neuropil_mask for roi in roi_list if roi.neuropil_mask is not None]
            if neuropil_masks:
                save_dict["neuropil_counts"] = neuropil_counts
                save_dict["neuropil_mask"] = np.concatenate(neuropil_masks)

        # Handles optional variable-length raveled_pixels arrays. Uses uint32 for counts since they are always
        # non-negative and can be large.
        has_raveled = [roi.raveled_pixels is not None for roi in roi_list]
        if any(has_raveled):
            raveled_counts = np.array(
                [len(roi.raveled_pixels) if roi.raveled_pixels is not None else 0 for roi in roi_list], dtype=np.uint32
            )
            raveled_arrays = [roi.raveled_pixels for roi in roi_list if roi.raveled_pixels is not None]
            if raveled_arrays:
                save_dict["raveled_counts"] = raveled_counts
                save_dict["raveled_pixels"] = np.concatenate(raveled_arrays)

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
        n_rois = len(pixel_counts)

        # Computes split indices for variable-length arrays.
        pixel_splits = np.cumsum(pixel_counts)[:-1]

        # Splits concatenated arrays back into per-ROI arrays.
        y_pixels_list = np.split(data["y_pixels"], pixel_splits)
        x_pixels_list = np.split(data["x_pixels"], pixel_splits)
        pixel_weights_list = np.split(data["pixel_weights"], pixel_splits)
        soma_mask_list = np.split(data["soma_mask"], pixel_splits)
        overlap_mask_list = np.split(data["overlap_mask"], pixel_splits)

        # Extracts scalar arrays.
        centroids = data["centroids"]
        footprints = data["footprints"]
        mean_r_squared = data["mean_r_squared"]
        mean_r_squared_baseline = data["mean_r_squared_baseline"]
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

        # Handles optional neuropil_mask arrays.
        neuropil_mask_list: list[NDArray[np.bool_] | None] = [None] * n_rois
        if "neuropil_counts" in data:
            neuropil_counts = data["neuropil_counts"]
            neuropil_data = data["neuropil_mask"]
            neuropil_idx = 0
            for i, count in enumerate(neuropil_counts):
                if count > 0:
                    neuropil_mask_list[i] = neuropil_data[neuropil_idx : neuropil_idx + count]
                    neuropil_idx += count

        # Handles optional raveled_pixels arrays.
        raveled_pixels_list: list[NDArray[np.int32] | None] = [None] * n_rois
        if "raveled_counts" in data:
            raveled_counts = data["raveled_counts"]
            raveled_data = data["raveled_pixels"]
            raveled_idx = 0
            for i, count in enumerate(raveled_counts):
                if count > 0:
                    raveled_pixels_list[i] = raveled_data[raveled_idx : raveled_idx + count]
                    raveled_idx += count

        # Reconstructs ROIStatistics instances.
        roi_list = []
        for i in range(n_rois):
            roi = ROIStatistics(
                y_pixels=y_pixels_list[i].astype(np.uint32),
                x_pixels=x_pixels_list[i].astype(np.uint32),
                pixel_weights=pixel_weights_list[i].astype(np.float32),
                centroid=[float(centroids[i, 0]), float(centroids[i, 1])],
                footprint=int(footprints[i]),
                mean_r_squared=float(mean_r_squared[i]),
                mean_r_squared_baseline=float(mean_r_squared_baseline[i]),
                compactness=float(compactness[i]),
                solidity=float(solidity[i]),
                pixel_count=int(pixel_count[i]),
                soma_pixel_count=int(soma_pixel_count[i]),
                soma_mask=soma_mask_list[i].astype(np.bool_),
                overlap_mask=overlap_mask_list[i].astype(np.bool_),
                radius=float(radius[i]),
                aspect_ratio=float(aspect_ratio[i]),
                normalized_pixel_count=float(normalized_pixel_count[i]),
                normalized_pixel_count_full=float(normalized_pixel_count_full[i]),
                skewness=None if np.isnan(skewness[i]) else float(skewness[i]),
                standard_deviation=None if np.isnan(standard_deviation[i]) else float(standard_deviation[i]),
                neuropil_mask=neuropil_mask_list[i],
                plane_index=int(plane_index[i]),
                cluster_id=int(cluster_id[i]),
                raveled_pixels=raveled_pixels_list[i],
                session_count=int(session_count[i]),
            )
            roi_list.append(roi)

        return roi_list


@dataclass
class ExtractionData:
    """Stores runtime data from the signal extraction and classification stages.

    This dataclass stores ROI statistics, fluorescence traces, deconvolved spikes, and cell classification results.
    When both channels are functional (independent ROI detection on each channel), channel 2 data is stored in the
    corresponding _channel_2 fields. The cell_colocalization field stores results indicating whether channel 1 ROIs
    are also present in channel 2.
    """

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

        # Colocalization array.
        if self.cell_colocalization is not None:
            np.save(output_path / "cell_colocalization.npy", self.cell_colocalization, allow_pickle=False)

    def load_arrays(self, output_path: Path) -> None:
        """Loads extraction arrays from .npy files and ROI statistics from .npz files into this instance.

        Args:
            output_path: The directory containing the extraction data files.
        """
        # Channel 1 ROI statistics.
        roi_stats_path = output_path / "roi_statistics.npz"
        if self.roi_statistics is None and roi_stats_path.exists():
            self.roi_statistics = ROIStatistics.load_list(roi_stats_path)

        # Channel 1 trace arrays.
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

        cell_classification_path = output_path / "cell_classification.npy"
        if self.cell_classification is None and cell_classification_path.exists():
            self.cell_classification = np.load(cell_classification_path, allow_pickle=False).astype(np.float32)

        # Channel 2 ROI statistics.
        roi_stats_channel_2_path = output_path / "roi_statistics_channel_2.npz"
        if self.roi_statistics_channel_2 is None and roi_stats_channel_2_path.exists():
            self.roi_statistics_channel_2 = ROIStatistics.load_list(roi_stats_channel_2_path)

        # Channel 2 trace arrays.
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

        cell_classification_channel_2_path = output_path / "cell_classification_channel_2.npy"
        if self.cell_classification_channel_2 is None and cell_classification_channel_2_path.exists():
            self.cell_classification_channel_2 = np.load(cell_classification_channel_2_path, allow_pickle=False).astype(
                np.float32
            )

        # Colocalization array.
        cell_colocalization_path = output_path / "cell_colocalization.npy"
        if self.cell_colocalization is None and cell_colocalization_path.exists():
            self.cell_colocalization = np.load(cell_colocalization_path, allow_pickle=False).astype(np.float32)


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

    def save(self, root_path: Path) -> None:
        """Saves combined data to the root suite2p directory.

        This method saves all combined detection and extraction arrays to the root suite2p directory. Metadata
        (plane count, dimensions) is saved to combined_metadata.npz.

        Args:
            root_path: The root suite2p output directory containing configuration.yaml.
        """
        ensure_directory_exists(root_path)

        # Saves metadata using appropriate unsigned types for counts and dimensions.
        np.savez(
            root_path / "combined_metadata.npz",
            allow_pickle=False,
            plane_count=np.array([self.plane_count], dtype=np.uint8),
            combined_height=np.array([self.combined_height], dtype=np.uint32),
            combined_width=np.array([self.combined_width], dtype=np.uint32),
        )

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
        )


@dataclass
class TimingData:
    """Stores pipeline timing information.

    All time durations are stored as integers representing seconds.
    """

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
    """The runtime data from the detection stage."""

    extraction: ExtractionData = field(default_factory=ExtractionData)
    """The runtime data from the extraction and classification stages."""

    timing: TimingData = field(default_factory=TimingData)
    """The pipeline timing information."""

    def __post_init__(self) -> None:
        """Loads NumPy arrays from .npy files if output_path is set and arrays are None."""
        if self.output_path is None:
            return

        # Converts output_path to Path if it was loaded as a string from YAML.
        if isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)

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

        # Creates a deep copy for YAML serialization.
        yaml_copy = copy.deepcopy(self)

        # Prepares each child dataclass for YAML serialization.
        yaml_copy.output_path = str(output_path)  # type: ignore[assignment]
        yaml_copy.io.prepare_for_saving()
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

        # Prepares each child dataclass for YAML serialization.
        yaml_copy.file_io.prepare_for_saving()
        yaml_copy.spike_deconvolution.prepare_for_saving()
        yaml_copy.classification.prepare_for_saving()

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


@dataclass
class RuntimeContext:
    """Combines configuration, acquisition parameters, and runtime data for pipeline functions.

    Notes:
        This class provides a unified interface for pipeline functions to access user configuration (immutable),
        acquisition parameters (from input data), and runtime data (computed by pipeline). It replaces the legacy ops
        dictionary pattern with a type-safe structure.

        Each RuntimeContext instance represents a single plane (or virtual plane for MROI data). The config and
        acquisition fields are shared across all planes, while the runtime field contains plane-specific data.
    """

    config: SingleDayConfiguration
    """The user configuration, which remains immutable during processing."""

    acquisition: AcquisitionParameters
    """The acquisition parameters loaded from the input data's JSON file. This describes the recording setup including
    frame rate, plane count, channel count, and MROI geometry if applicable."""

    runtime: SingleDayRuntimeData
    """The runtime data, which is computed and updated by pipeline stages."""

    def save_shared(self) -> None:
        """Saves shared configuration and acquisition parameters to the root output directory.

        This method derives the root path from self.config.file_io.save_path and creates the suite2p subdirectory
        if it does not exist. It should be called once at pipeline initialization to save the static data shared
        across all planes.

        Raises:
            ValueError: If save_path is not configured in the configuration.
        """
        if self.config.file_io.save_path is None:
            message = (
                "Unable to save shared configuration data. The save_path must be configured in the FileIO section "
                "of the configuration, but it is currently None."
            )
            console.error(message=message, error=ValueError)

        root_path = self.config.file_io.save_path / "suite2p"
        root_path.mkdir(parents=True, exist_ok=True)

        self.config.save(file_path=root_path / "configuration.yaml")
        self.acquisition.to_yaml(file_path=root_path / "acquisition_parameters.yaml")

    def save_runtime(self) -> None:
        """Saves this plane's runtime data to its output directory.

        This method uses self.runtime.io.output_directory as the save location. This directory is set during plane
        initialization and is plane-specific (e.g., plane_0/).

        Raises:
            ValueError: If output_directory is not set in the runtime IOData.
        """
        if self.runtime.io.output_directory is None:
            message = (
                "Unable to save runtime data. The output_directory must be set in the IOData section of the "
                "runtime data, but it is currently None."
            )
            console.error(message=message, error=ValueError)

        self.runtime.save(output_path=self.runtime.io.output_directory)

    @classmethod
    def load(cls, root_path: Path, plane_index: int = -1) -> RuntimeContext | list[RuntimeContext]:
        """Loads one or more instances from disk.

        This method loads the shared configuration and acquisition parameters from the root suite2p directory, then
        loads plane-specific runtime data from plane directories. Use plane_index=-1 to load all available planes,
        or specify a non-negative index to load a single plane.

        Args:
            root_path: The root suite2p output directory containing configuration.yaml and acquisition_parameters.yaml.
            plane_index: The index of the plane to load. Use -1 to load all available planes.

        Returns:
            A single RuntimeContext if plane_index >= 0, or a list of all RuntimeContext instances if plane_index
            is -1.

        Raises:
            FileNotFoundError: If the configuration files or specified plane directory do not exist.
        """
        config_path = root_path / "configuration.yaml"
        acquisition_path = root_path / "acquisition_parameters.yaml"

        if not config_path.exists():
            message = (
                f"Unable to load RuntimeContext. Configuration file does not exist at the specified path: "
                f"{config_path}."
            )
            console.error(message=message, error=FileNotFoundError)

        if not acquisition_path.exists():
            message = (
                f"Unable to load RuntimeContext. Acquisition parameters file does not exist at the specified path: "
                f"{acquisition_path}."
            )
            console.error(message=message, error=FileNotFoundError)

        config = SingleDayConfiguration.load(file_path=config_path)
        acquisition = AcquisitionParameters.from_yaml(file_path=acquisition_path)

        if plane_index == -1:
            # Loads all planes.
            plane_directories = natsorted([d for d in root_path.glob("plane_*") if d.is_dir()])
            contexts: list[RuntimeContext] = []

            for plane_directory in plane_directories:
                runtime = SingleDayRuntimeData.load(output_path=plane_directory)
                contexts.append(cls(config=config, acquisition=acquisition, runtime=runtime))

            return contexts

        # Loads a specific plane.
        plane_path = root_path / f"plane_{plane_index}"
        if not plane_path.exists():
            message = (
                f"Unable to load RuntimeContext. Plane directory does not exist at the specified path: {plane_path}."
            )
            console.error(message=message, error=FileNotFoundError)

        runtime = SingleDayRuntimeData.load(output_path=plane_path)
        return cls(config=config, acquisition=acquisition, runtime=runtime)
