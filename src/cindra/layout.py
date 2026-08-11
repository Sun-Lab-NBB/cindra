"""Provides the on-disk contract of the cindra pipelines: the directory and file names every stage writes under a
caller-supplied output root, and the pure resolvers that build a path from that root.

This module imports nothing from cindra, so every layer from the configuration dataclasses upward reads the contract
from one definition instead of respelling it. An external scheduler that locates a recording's inputs and outputs
therefore names the same strings the pipeline writes, rather than a copy that drifts from them.
"""

from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

OUTPUT_DIRECTORY_NAME: str = "cindra"
"""The name of the directory every pipeline stage writes its output under, created inside the caller's output root."""

MULTI_RECORDING_DIRECTORY_NAME: str = "multi_recording"
"""The name of the directory holding the multi-recording results, created inside a recording's output directory."""

PLANE_SPECIFIER_PREFIX: str = "plane_"
"""The prefix of the specifier that identifies one virtual imaging plane, completed by the plane index.

Notes:
    The specifier names the plane's output directory and identifies the plane a per-plane job processes on its
    tracker, so the directory a plane writes into and the specifier its job carries stay spelled the same way.
"""

REGISTRATION_DATA_DIRECTORY_NAME: str = "registration_data"
"""The name of the per-plane directory holding the registration offsets, the reference image, and the quality
metrics."""

DETECTION_DATA_DIRECTORY_NAME: str = "detection_data"
"""The name of the directory holding the mean, enhanced mean, maximum projection, and correlation images."""

MULTI_RECORDING_ARRAYS_DIRECTORY_NAME: str = "registration_arrays"
"""The name of the directory holding the multi-recording deformation fields and transformed reference images."""

PARAMETERS_FILENAME: str = "cindra_parameters.json"
"""The name of the acquisition parameters file the pipeline reads from a recording's raw imaging directory."""

SINGLE_RECORDING_CONFIGURATION_FILENAME: str = "configuration.yaml"
"""The name of the single-recording configuration file, written into the recording's output directory."""

MULTI_RECORDING_CONFIGURATION_FILENAME: str = "multi_recording_configuration.yaml"
"""The name of the multi-recording configuration file, written into the dataset directory of the main recording,
which is the first of the dataset's recordings after natural sorting."""

ACQUISITION_PARAMETERS_FILENAME: str = "acquisition_parameters.yaml"
"""The name of the resolved acquisition parameters file, written into the recording's output directory."""

SINGLE_RECORDING_RUNTIME_DATA_FILENAME: str = "runtime_data.yaml"
"""The name of the per-plane runtime data file, written into each plane's output directory."""

MULTI_RECORDING_RUNTIME_DATA_FILENAME: str = "multi_recording_runtime_data.yaml"
"""The name of the multi-recording runtime data file, written into the tracked dataset's directory."""

SINGLE_RECORDING_TRACKER_FILENAME: str = "single_recording_tracker.yaml"
"""The name of the tracker file recording the state of every single-recording job of one recording."""

MULTI_RECORDING_TRACKER_FILENAME: str = "multi_recording_tracker.yaml"
"""The name of the tracker file recording the state of every multi-recording job of one tracked dataset."""

COMBINED_METADATA_FILENAME: str = "combined_metadata.npz"
"""The name of the archive holding the combined plane geometry, which doubles as the single-recording completion
marker.

Notes:
    The combination stage writes this archive after its payload arrays and publishes it through an atomic write, so a
    consumer that finds it can rely on every array it describes already being on disk.
"""

TRACKING_TEMPLATE_MASKS_FILENAME: str = "tracking_template_masks.npz"
"""The name of the archive holding the tracked ROI template masks, which doubles as the multi-recording discovery
completion marker."""

DEFORMED_MASKS_FILENAME: str = "registration_deformed_masks.npz"
"""The name of the archive holding the ROI masks deformed into the shared visual space."""

CHANNEL_1_BINARY_FILENAME: str = "channel_1_data.bin"
"""The name of the binary holding the functional channel frames of one imaging plane."""

CHANNEL_2_BINARY_FILENAME: str = "channel_2_data.bin"
"""The name of the binary holding the second channel frames of one imaging plane."""

BINARY_WRITE_MARKER_SUFFIX: str = ".writing"
"""The suffix appended to a plane binary's name while a stage writes frames into that binary.

Notes:
    Binarization sizes a binary to its full frame count before writing its first frame, and registration rewrites that
    binary in place. A binary carrying this marker therefore holds an indeterminate mixture of finished and unfinished
    frames, which the binarization stage treats as invalid and rebuilds from the source images.
"""

CHANNEL_2_ARRAY_SUFFIX: str = "_channel_2"
"""The suffix distinguishing the second channel's copy of a result array from the functional channel's copy."""


class RecordingArrays(StrEnum):
    """Defines the names of the result arrays every extraction stage writes.

    Notes:
        The same names are written into a recording's output directory for the combined multi-plane results, into each
        plane's directory for that plane's results, and into a tracked dataset's directory for the multi-recording
        results, so one name resolves against any of the three roots.
    """

    CELL_FLUORESCENCE = "cell_fluorescence.npy"
    """The raw fluorescence trace of every ROI."""
    NEUROPIL_FLUORESCENCE = "neuropil_fluorescence.npy"
    """The neuropil fluorescence trace surrounding every ROI."""
    SUBTRACTED_FLUORESCENCE = "subtracted_fluorescence.npy"
    """The delta fluorescence of every ROI, after both the scaled neuropil trace and the estimated resting baseline
    have been subtracted."""
    SPIKES = "spikes.npy"
    """The deconvolved spike trace of every ROI."""
    CELL_CLASSIFICATION = "cell_classification.npy"
    """The classification of every ROI, holding its thresholded is-cell label in the first column and the probability
    that label was drawn from in the second."""
    CELL_COLOCALIZATION = "cell_colocalization.npy"
    """The colocalization of every ROI with the recording's second channel, measured against the structural channel
    when the second channel is structural and against the second functional channel's own ROIs otherwise."""
    ROI_MASKS = "roi_masks.npz"
    """The pixel masks and weights of every ROI."""
    ROI_STATISTICS = "roi_statistics.npz"
    """The per-ROI statistics, holding the shape descriptors alongside the soma, overlap, and neuropil masks and the
    trace skewness the extraction stage records."""
    CORRECTED_STRUCTURAL_MEAN_IMAGE = "corrected_structural_mean_image.npy"
    """The structural channel mean image with the functional channel bleed-through removed."""


class DetectionImages(StrEnum):
    """Defines the names of the summary images the detection stage writes into its own subdirectory."""

    MEAN_IMAGE = "mean_image.npy"
    """The mean of the temporally binned movie, which excludes the frames registration marked bad."""
    ENHANCED_MEAN_IMAGE = "enhanced_mean_image.npy"
    """The mean image filtered at the detected ROI scale."""
    MAXIMUM_PROJECTION = "maximum_projection.npy"
    """The maximum of every pixel across the temporally binned and high-pass filtered movie, zero outside the valid
    registration crop."""
    CORRELATION_MAP = "correlation_map.npy"
    """The maximum activity of every pixel across the detection scale pyramid, zero outside the valid registration
    crop."""


class RegistrationArrays(StrEnum):
    """Defines the names of the arrays the registration stage writes into its own subdirectory."""

    BAD_FRAMES = "bad_frames.npy"
    """The mask marking the frames excluded from the valid-region crop and from detection's temporal binning, which
    registration computes from its own offsets after registering every frame.

    Notes:
        The same name identifies an optional input file placed in the raw imaging directory, where it instead holds the
        indices of the frames to mark bad before the crop is computed.
    """
    REFERENCE_IMAGE = "reference_image.npy"
    """The reference image every frame is registered against, whose presence marks a plane as registered."""
    RIGID_Y_OFFSETS = "rigid_y_offsets.npy"
    """The vertical rigid shift applied to every frame."""
    RIGID_X_OFFSETS = "rigid_x_offsets.npy"
    """The horizontal rigid shift applied to every frame."""
    RIGID_CORRELATIONS = "rigid_correlations.npy"
    """The phase correlation peak of every frame's rigid alignment."""
    NONRIGID_Y_OFFSETS = "nonrigid_y_offsets.npy"
    """The vertical shift applied to every block of every frame."""
    NONRIGID_X_OFFSETS = "nonrigid_x_offsets.npy"
    """The horizontal shift applied to every block of every frame."""
    NONRIGID_CORRELATIONS = "nonrigid_correlations.npy"
    """The phase correlation peak of every block's nonrigid alignment."""
    PRINCIPAL_COMPONENT_EXTREME_IMAGES = "principal_component_extreme_images.npy"
    """The low and high projection images of every retained principal component."""
    PRINCIPAL_COMPONENT_PROJECTIONS = "principal_component_projections.npy"
    """The projection of every sampled frame onto every retained principal component."""
    PRINCIPAL_COMPONENT_SHIFT_METRICS = "principal_component_shift_metrics.npy"
    """The residual shift measured between each principal component's extreme images."""


class MultiRecordingArrays(StrEnum):
    """Defines the names of the arrays the multi-recording registration stage writes into its own subdirectory."""

    DEFORM_FIELD_Y = "deform_field_y.npy"
    """The vertical deformation carrying this recording into the shared visual space."""
    DEFORM_FIELD_X = "deform_field_x.npy"
    """The horizontal deformation carrying this recording into the shared visual space."""
    TRANSFORMED_MEAN_IMAGE = "transformed_mean_image.npy"
    """The recording's mean image deformed into the shared visual space."""
    TRANSFORMED_ENHANCED_MEAN_IMAGE = "transformed_enhanced_mean_image.npy"
    """The recording's enhanced mean image deformed into the shared visual space."""
    TRANSFORMED_MAXIMUM_PROJECTION = "transformed_maximum_projection.npy"
    """The recording's maximum projection deformed into the shared visual space."""


type PipelineArray = RecordingArrays | DetectionImages | RegistrationArrays | MultiRecordingArrays
"""Any array name the pipelines write, which the array path resolver accepts from any of the four families."""


def resolve_output_path(output_root: Path) -> Path:
    """Resolves the directory every pipeline stage writes its output under.

    Args:
        output_root: The output root the caller configured for the recording.

    Returns:
        The path to the recording's cindra output directory.
    """
    return output_root.joinpath(OUTPUT_DIRECTORY_NAME)


def resolve_plane_path(output_root: Path, plane_index: int) -> Path:
    """Resolves the output directory of one virtual imaging plane.

    Args:
        output_root: The output root the caller configured for the recording.
        plane_index: The index of the virtual imaging plane.

    Returns:
        The path to the plane's output directory.
    """
    return resolve_output_path(output_root=output_root).joinpath(resolve_plane_specifier(plane_index=plane_index))


def resolve_dataset_path(output_root: Path, dataset_name: str) -> Path:
    """Resolves the output directory of one tracked multi-recording dataset inside a recording's output directory.

    Notes:
        The dataset name is lowered here, matching the fold the multi-recording context resolver applies when it
        builds the directory, so a caller passing the configured name reaches the directory the pipeline wrote.

    Args:
        output_root: The output root the caller configured for the recording.
        dataset_name: The name of the tracked dataset, in any casing.

    Returns:
        The path to the dataset's output directory inside the recording's output directory.
    """
    return resolve_output_path(output_root=output_root).joinpath(MULTI_RECORDING_DIRECTORY_NAME, dataset_name.lower())


def resolve_channel_2_name(name: str) -> str:
    """Resolves the name of the second channel's copy of a result file.

    Args:
        name: The name of the functional channel's copy of the file.

    Returns:
        The name of the second channel's copy, carrying the channel suffix before the extension.
    """
    stem, _, extension = name.rpartition(".")
    return f"{stem}{CHANNEL_2_ARRAY_SUFFIX}.{extension}"


def resolve_array_name(array: PipelineArray, *, second_channel: bool = False) -> str:
    """Resolves the filename one pipeline array is written under.

    Args:
        array: The array to resolve the filename of, named by any of the four pipeline array families.
        second_channel: Determines whether the second channel's copy of the array is named instead of the functional
            channel's copy.

    Returns:
        The filename the array is written under.
    """
    if not second_channel:
        return array.value
    return resolve_channel_2_name(name=array.value)


def resolve_array_path(root_path: Path, array: PipelineArray, *, second_channel: bool = False) -> Path:
    """Resolves the path to one pipeline array under a recording, plane, or dataset directory.

    Args:
        root_path: The directory the array was written into. The result arrays are written into a recording output
            directory, a plane directory, or a dataset directory, while the detection, registration, and
            multi-recording arrays are written into their own subdirectory of one of those.
        array: The array to resolve the path of, named by any of the four pipeline array families.
        second_channel: Determines whether the second channel's copy of the array is resolved instead of the
            functional channel's copy.

    Returns:
        The path to the requested result array.
    """
    return root_path.joinpath(resolve_array_name(array=array, second_channel=second_channel))


def resolve_binary_write_marker_name(binary_name: str) -> str:
    """Resolves the name of the marker written beside a plane binary while a stage writes frames into it.

    Args:
        binary_name: The name of the plane binary being written.

    Returns:
        The name of the marker file guarding the write.
    """
    return f"{binary_name}{BINARY_WRITE_MARKER_SUFFIX}"


def resolve_plane_specifier(plane_index: int) -> str:
    """Resolves the specifier that identifies one virtual imaging plane.

    Notes:
        The specifier names a plane's output directory and identifies the plane a per-plane job processes on its
        tracker, so both spell the plane the same way.

    Args:
        plane_index: The index of the virtual imaging plane.

    Returns:
        The specifier identifying the plane.
    """
    return f"{PLANE_SPECIFIER_PREFIX}{plane_index}"


def parse_plane_specifier(specifier: str) -> int | None:
    """Reads the virtual plane index a per-plane specifier carries.

    Notes:
        The inverse of resolve_plane_specifier. A specifier that names no plane resolves to None rather than raising,
        because a caller routing a mixed job set asks this of every specifier it holds.

    Args:
        specifier: The specifier to read the plane index from.

    Returns:
        The virtual plane index, or None when the specifier does not name a plane.
    """
    if not specifier.startswith(PLANE_SPECIFIER_PREFIX):
        return None
    digits = specifier.removeprefix(PLANE_SPECIFIER_PREFIX)
    # str.isdigit() admits superscripts and other numeric forms int() rejects, so the check is narrowed to ASCII.
    return int(digits) if digits.isascii() and digits.isdigit() else None
