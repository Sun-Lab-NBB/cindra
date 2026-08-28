"""Provides the per-job entry points of the two pipelines, which run exactly one job against a caller-owned tracker."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ataraxis_base_utilities import console

from ..io import (
    RecordingPlanes,
    DatasetRecordings,
    resolve_recording_planes,
    resolve_dataset_recordings,
    resolve_multi_recording_contexts,
    resolve_single_recording_contexts,
)
from .gpu import verify_gpu_runtime
from .jobs import (
    MultiRecordingJobNames,
    SingleRecordingJobNames,
)
from ..layout import OUTPUT_DIRECTORY_NAME, parse_plane_specifier
from ..pipelines import (
    process_plane,
    binarize_recording,
    save_combined_data,
    register_recording_plane,
    discover_multi_recording_cells,
    extract_multi_recording_fluorescence,
)
from .allocation import resolve_stage_workers
from ..dataclasses import RuntimeContext, MultiRecordingConfiguration, SingleRecordingConfiguration

if TYPE_CHECKING:
    from pathlib import Path

    from ataraxis_data_structures import ProcessingTracker

_MINIMUM_DATASET_RECORDINGS: int = 2
"""The minimum number of recordings a multi-recording dataset requires."""


def execute_single_recording_job(
    configuration_path: Path,
    job_name: SingleRecordingJobNames,
    specifier: str,
    job_id: str,
    tracker: ProcessingTracker,
    *,
    workers: int | None = None,
    device: int | None = None,
) -> None:
    """Executes one single-recording job and records its state on a caller-provided tracker.

    Notes:
        This is the tracker-injection entry point. The caller owns the tracker, aligns it with a universe that contains
        job_id, and passes both in, so cindra stages this job into a foreign tracker whose job names and granularity the
        caller controls. The job's start, completion, and failure are recorded onto the provided tracker under job_id.

        Every stage reads the shared bootstrap rather than writing it, so prime_recording must have written it before
        any job runs. Priming is a separate call rather than a flag on this one, because a job that wrote the
        bootstrap while its peers ran would overwrite each peer plane's runtime data with its own stale snapshot.

    Args:
        configuration_path: The path to the single-recording configuration YAML file.
        job_name: The single-recording job to run.
        specifier: The job specifier. For a REGISTER or PROCESS job this encodes the plane index as 'plane_{index}',
            and for a BINARIZE or COMBINE job it is an empty string.
        job_id: The unique hexadecimal identifier under which the job's state is recorded on the provided tracker. It
            must already be present in the tracker's aligned job set.
        tracker: The caller-owned ProcessingTracker onto which this job's start, completion, or failure is recorded.
        workers: The number of parallel workers to allocate to this job. Use None to accept the stage default and -1
            to request every available core. The combination job ignores this parameter.
        device: The zero-based index of the CUDA device on which a registration job runs. Use None to run the
            registration on the host CPU. Every other job ignores this parameter.

    Raises:
        FileNotFoundError: If the configuration file is missing, is not a .yaml file, or is not a valid single-recording
            configuration.
        RuntimeError: If device names a CUDA device on a host that exposes no usable one.
        ValueError: If the configuration does not configure an output path, if job_name is not a recognized
            single-recording job, if workers is zero or a negative value other than -1, or if device names an index
            the host does not expose.
    """
    configuration, _ = load_single_recording_configuration(configuration_path=configuration_path)

    dispatch_single_recording_job(
        configuration=configuration,
        job_name=SingleRecordingJobNames(job_name),
        specifier=specifier,
        job_id=job_id,
        tracker=tracker,
        workers=workers,
        device=device,
    )


def execute_multi_recording_job(
    configuration_path: Path,
    job_name: MultiRecordingJobNames,
    specifier: str,
    job_id: str,
    tracker: ProcessingTracker,
    *,
    workers: int | None = None,
) -> None:
    """Executes one multi-recording job and records its state on a caller-provided tracker.

    Notes:
        This is the tracker-injection entry point. The caller owns the tracker, aligns it with a universe that contains
        job_id, and passes both in, so cindra stages this job into a foreign tracker whose job names and granularity the
        caller controls. The job's start, completion, and failure are recorded onto the provided tracker under job_id.

        Every stage reads the shared bootstrap rather than writing it, so prime_dataset must have written it before
        any job runs. Priming is a separate call rather than a flag on this one, because a job that wrote the
        bootstrap while its peers ran would overwrite each peer recording's runtime data with its own stale snapshot.

    Args:
        configuration_path: The path to the multi-recording configuration YAML file.
        job_name: The multi-recording job to run.
        specifier: The job specifier. For an EXTRACT job this is the recording identifier, and for a DISCOVER job it is
            an empty string.
        job_id: The unique hexadecimal identifier under which the job's state is recorded on the provided tracker. It
            must already be present in the tracker's aligned job set.
        tracker: The caller-owned ProcessingTracker onto which this job's start, completion, or failure is recorded.
        workers: The number of parallel workers to allocate to this job. Use None to accept the stage default and -1
            to request every available core.

    Raises:
        FileNotFoundError: If the configuration file is missing, is not a .yaml file, or is not a valid multi-recording
            configuration.
        ValueError: If the configuration specifies fewer than two recording directories or no dataset name, if
            job_name is not a recognized multi-recording job, or if workers is zero or a negative value other
            than -1.
    """
    configuration = load_multi_recording_configuration(configuration_path=configuration_path)

    dispatch_multi_recording_job(
        configuration=configuration,
        job_name=MultiRecordingJobNames(job_name),
        specifier=specifier,
        job_id=job_id,
        tracker=tracker,
        workers=workers,
    )


def load_single_recording_configuration(configuration_path: Path) -> tuple[SingleRecordingConfiguration, Path]:
    """Loads, validates, and runtime-configures a single-recording configuration from a YAML file.

    Args:
        configuration_path: The path to the single-recording configuration YAML file.

    Returns:
        A tuple of the loaded SingleRecordingConfiguration, with its progress display state applied, and its validated
        output path.

    Raises:
        FileNotFoundError: If the configuration file is missing, is not a .yaml file, or is not a valid single-recording
            configuration.
        ValueError: If the configuration does not configure an output path.
    """
    if not configuration_path.exists() or configuration_path.suffix != ".yaml":
        message = (
            "Unable to run the single-recording cindra processing pipeline. Expected the configuration file to "
            f"end with a '.yaml' extension and exist at the specified path, but encountered: {configuration_path}."
        )
        console.error(message=message, error=FileNotFoundError)

    try:
        configuration: SingleRecordingConfiguration = SingleRecordingConfiguration.from_yaml(
            file_path=configuration_path,
        )
    except Exception:
        message = (
            "Unable to run the single-recording cindra processing pipeline, as the input configuration file is not a "
            "valid single-recording pipeline configuration file. Specifically, failed to load the file's data as a "
            "SingleRecordingConfiguration dataclass instance. Ensure that the 'configuration_path' argument "
            "points to a valid single-recording configuration .yaml file."
        )
        console.error(message=message, error=FileNotFoundError)

    if configuration.runtime.display_progress_bars:
        console.enable_progress()
    else:
        console.disable_progress()

    if configuration.file_io.output_path is None:
        message = (
            "Unable to run the single-recording cindra processing pipeline. The output_path must be configured in the "
            "FileIO section of the configuration, but it is currently None."
        )
        console.error(message=message, error=ValueError)

    return configuration, configuration.file_io.output_path


def load_multi_recording_configuration(configuration_path: Path) -> MultiRecordingConfiguration:
    """Loads, validates, and runtime-configures a multi-recording configuration from a YAML file.

    Args:
        configuration_path: The path to the multi-recording configuration YAML file.

    Returns:
        The loaded MultiRecordingConfiguration with its progress display state applied.

    Raises:
        FileNotFoundError: If the configuration file is missing, is not a .yaml file, or is not a valid multi-recording
            configuration.
        ValueError: If the configuration specifies fewer than two recording directories or no dataset name.
    """
    if not configuration_path.exists() or configuration_path.suffix != ".yaml":
        message = (
            "Unable to run the multi-recording cindra processing pipeline. "
            "Expected the configuration file to end with a '.yaml' extension and "
            f"exist at the specified path, but encountered: {configuration_path}."
        )
        console.error(message=message, error=FileNotFoundError)

    try:
        configuration: MultiRecordingConfiguration = MultiRecordingConfiguration.from_yaml(file_path=configuration_path)
    except Exception:
        message = (
            "Unable to run the multi-recording cindra processing pipeline, as the input configuration file is not a "
            "valid multi-recording pipeline configuration file. Specifically, failed to load the file's data as a "
            "MultiRecordingConfiguration dataclass instance. Ensure that the 'configuration_path' argument points to a "
            "valid multi-recording configuration .yaml file."
        )
        console.error(message=message, error=FileNotFoundError)

    # Validates that the configuration names enough recordings to track ROIs across. Tracking compares each
    # recording's ROIs against those of the other recordings, so a lone recording resolves nothing.
    if len(configuration.recording_io.recording_directories) < _MINIMUM_DATASET_RECORDINGS:
        message = (
            "Unable to run the multi-recording cindra processing pipeline. The "
            "configuration file must specify at least two recording directories "
            "under 'recording_io.recording_directories'. The provided configuration "
            f"specifies {len(configuration.recording_io.recording_directories)}."
        )
        console.error(message=message, error=ValueError)

    if not configuration.recording_io.dataset_name:
        message = (
            "Unable to run the multi-recording cindra processing pipeline. The "
            "configuration file must specify a dataset name under "
            "'recording_io.dataset_name'. The provided configuration has no "
            "dataset name specified."
        )
        console.error(message=message, error=ValueError)

    if configuration.runtime.display_progress_bars:
        console.enable_progress()
    else:
        console.disable_progress()

    return configuration


def dispatch_single_recording_job(
    configuration: SingleRecordingConfiguration,
    job_name: SingleRecordingJobNames,
    specifier: str,
    job_id: str,
    tracker: ProcessingTracker,
    workers: int | None,
    device: int | None = None,
) -> None:
    """Executes a single processing job of the single-recording pipeline.

    Args:
        configuration: The loaded configuration the dispatched stage reads.
        job_name: The job to run.
        specifier: The job specifier string. For REGISTER and PROCESS jobs, this encodes the plane index as
            'plane_{index}'. For BINARIZE and COMBINE jobs, this is an empty string.
        job_id: The unique hexadecimal identifier for this processing job.
        tracker: The tracker that records this job's state transitions.
        workers: The number of parallel workers to allocate to this job. Use None to accept the stage default and -1
            to request every available core. The combination job ignores this parameter.
        device: The zero-based index of the CUDA device on which a registration job runs. Use None to run the
            registration on the host CPU. Every other job ignores it.

    Raises:
        RuntimeError: If a registration job names a CUDA device on a host that exposes no usable one.
        ValueError: If the job_name is not recognized, if the requested worker count is invalid, or if a registration
            job names a device index the host does not expose.
    """
    console.echo(message=f"Running '{job_name}' job (specifier='{specifier}') with ID {job_id}...")

    # The tracker's run_job() context owns the job's state transitions: it marks the job as running, completes it when
    # the block returns, and records the exception's message as the failure reason before re-raising when the block
    # raises. Every worker-consuming stage resolves its budget inside the block, so an invalid request is recorded as
    # a job failure instead of escaping as an untracked error. The resolution happens per branch rather than ahead of
    # the chain, so that an unrecognized job name reaches the job-name guard below.
    with tracker.run_job(job_id=job_id):
        if job_name == SingleRecordingJobNames.BINARIZE:
            binarize_recording(
                configuration=configuration,
                workers=resolve_stage_workers(job_name=job_name, requested_workers=workers),
            )

        elif job_name == SingleRecordingJobNames.REGISTER:
            # Gates the device before the stage writes its registration marker. A request naming an index the host
            # does not expose, or a host carrying no usable runtime, then reports the argument it failed on rather
            # than the CUDA runtime error the backend constructor raises. The check sits inside the tracker's block,
            # so the refusal is recorded as this job's failure rather than escaping untracked. A caller reaching this
            # through run_single_recording_pipeline is verified there as well, which costs one repeat probe against
            # a context that entry point has already opened.
            if device is not None:
                verify_gpu_runtime(device=device)

            register_recording_plane(
                configuration=configuration,
                plane_index=_resolve_job_plane_index(job_name=job_name, specifier=specifier),
                workers=resolve_stage_workers(
                    job_name=job_name, requested_workers=workers, gpu_registration=device is not None
                ),
                device=device,
            )

        elif job_name == SingleRecordingJobNames.PROCESS:
            process_plane(
                configuration=configuration,
                plane_index=_resolve_job_plane_index(job_name=job_name, specifier=specifier),
                workers=resolve_stage_workers(job_name=job_name, requested_workers=workers),
            )

        elif job_name == SingleRecordingJobNames.COMBINE:
            if configuration.file_io.output_path is None:
                message = (
                    "Unable to execute the combination job. The output_path must be configured in the FileIO section "
                    "of the configuration, but it is currently None."
                )
                console.error(message=message, error=ValueError)

            # Loads contexts from disk and combines all processed planes into a dataset. Arrays are not
            # loaded automatically due to their memory footprint, so they must be loaded explicitly before
            # combining. Detection arrays provide background images. Extraction arrays provide ROI statistics
            # and fluorescence traces.
            root_path = configuration.file_io.output_path / OUTPUT_DIRECTORY_NAME
            contexts = RuntimeContext.load(root_path=root_path, plane_index=-1)
            if not isinstance(contexts, list):  # pragma: no cover - load with plane_index=-1 always returns a list
                contexts = [contexts]
            for context in contexts:
                # pragma justification: resolved plane contexts always carry a configured output path.
                if context.runtime.output_path is not None:  # pragma: no branch
                    context.runtime.detection.memory_map_arrays(output_path=context.runtime.output_path)
                    context.runtime.extraction.memory_map_arrays(output_path=context.runtime.output_path)
                    context.runtime.extraction.memory_map_results(output_path=context.runtime.output_path)
            save_combined_data(contexts=contexts)

        else:
            message = (
                f"Unable to execute the requested job '{job_name}' with ID '{job_id}'. The input job name is not "
                f"recognized. Use one of the valid Job names: {list(SingleRecordingJobNames)}."
            )
            console.error(message=message, error=ValueError)


def dispatch_multi_recording_job(
    configuration: MultiRecordingConfiguration,
    job_name: MultiRecordingJobNames,
    specifier: str,
    job_id: str,
    tracker: ProcessingTracker,
    workers: int | None,
) -> None:
    """Executes a single processing job of the multi-recording pipeline.

    Args:
        configuration: The loaded configuration the dispatched stage reads.
        job_name: The job to run.
        specifier: The job specifier string. For EXTRACT jobs, this is the recording ID. For DISCOVER jobs, this is an
            empty string.
        job_id: The unique hexadecimal identifier for this processing job.
        tracker: The tracker that records this job's state transitions.
        workers: The number of parallel workers to allocate to this job. Use None to accept the stage default and -1
            to request every available core.

    Raises:
        ValueError: If the job_name is not recognized or the requested worker count is invalid.
    """
    console.echo(message=f"Running '{job_name}' job (specifier='{specifier}') with ID {job_id}...")

    # The tracker's run_job() context owns the job's state transitions, matching the single-recording executor. Each
    # stage resolves its worker budget inside its own dispatch branch, so an unrecognized job name reports the
    # job-name error below rather than a worker-resolution error. An invalid worker request is still recorded as a job
    # failure instead of escaping untracked.
    with tracker.run_job(job_id=job_id):
        if job_name == MultiRecordingJobNames.DISCOVER:
            discover_multi_recording_cells(
                configuration=configuration,
                workers=resolve_stage_workers(job_name=job_name, requested_workers=workers),
            )

        elif job_name == MultiRecordingJobNames.EXTRACT:
            extract_multi_recording_fluorescence(
                configuration=configuration,
                recording_id=specifier,
                workers=resolve_stage_workers(job_name=job_name, requested_workers=workers),
            )

        else:
            message = (
                f"Unable to execute the requested job '{job_name}' with ID '{job_id}'. The input job name is not "
                f"recognized. Use one of the valid Job names: {list(MultiRecordingJobNames)}."
            )
            console.error(message=message, error=ValueError)


def prime_recording(configuration_path: Path) -> RecordingPlanes:
    """Writes the shared single-recording bootstrap and reports the planes the recording holds.

    Notes:
        Every per-job entry point re-loads this bootstrap with persistence disabled, so this call must precede the
        first job dispatched against a configuration. Priming is single-threaded by contract, because it writes one
        runtime data file per plane and a peer job writing them concurrently would overwrite each other's snapshot.

    Args:
        configuration_path: The path to the single-recording configuration file.

    Returns:
        The recording's plane inventory.

    Raises:
        FileNotFoundError: If the configuration file is missing, is not a .yaml file, or is not a valid
            single-recording configuration.
        ValueError: If the configuration does not configure an output path.
    """
    configuration, output_path = load_single_recording_configuration(configuration_path=configuration_path)
    resolve_single_recording_contexts(configuration=configuration, persist=True)
    return resolve_recording_planes(output_root=output_path, data_path=configuration.file_io.data_path)


def prime_dataset(configuration_path: Path) -> DatasetRecordings:
    """Writes the shared multi-recording bootstrap and reports the recordings the dataset spans.

    Notes:
        Every per-job entry point re-loads this bootstrap with persistence disabled, so this call must precede the
        first job dispatched against a configuration. Priming is single-threaded by contract, for the same reason the
        single-recording bootstrap is.

    Args:
        configuration_path: The path to the multi-recording configuration file.

    Returns:
        The dataset's recording inventory.

    Raises:
        FileNotFoundError: If the configuration file is missing, is not a .yaml file, is not a valid multi-recording
            configuration, or if a recording holds no combined metadata archive.
        ValueError: If the configuration names fewer than two recording directories or no dataset name.
        RuntimeError: If a recording directory holds several combined metadata archives, if the recording paths
            carry no unique identifying component, or if a resolved identifying component contains a colon.
    """
    configuration = load_multi_recording_configuration(configuration_path=configuration_path)
    contexts = resolve_multi_recording_contexts(configuration=configuration, persist=True)
    return resolve_dataset_recordings(
        recording_roots=[
            context.runtime.io.data_path.parent for context in contexts if context.runtime.io.data_path is not None
        ],
        dataset_name=configuration.recording_io.dataset_name,
    )


def _resolve_job_plane_index(job_name: str, specifier: str) -> int:
    """Reads the imaging plane a per-plane job's specifier names.

    Args:
        job_name: The name of the job whose specifier is read, used in the error message.
        specifier: The job's tracker specifier.

    Returns:
        The index of the imaging plane the job processes.

    Raises:
        ValueError: If the specifier does not name an imaging plane.
    """
    plane_index = parse_plane_specifier(specifier=specifier)
    if plane_index is None:
        message = (
            f"Unable to execute the '{job_name}' job. A per-plane job's specifier must name an imaging plane, but "
            f"encountered '{specifier}'."
        )
        console.error(message=message, error=ValueError)
    return plane_index
