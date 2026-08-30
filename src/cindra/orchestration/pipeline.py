"""Provides the centralized pipeline for processing the acquired brain activity data."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ataraxis_base_utilities import LogLevel, console
from ataraxis_data_structures import ProcessingTracker

from ..io import resolve_multi_recording_contexts, resolve_single_recording_contexts
from .gpu import verify_gpu_runtime
from .jobs import (
    PER_PLANE_JOB_NAMES,
    MultiRecordingJobNames,
    SingleRecordingJobNames,
    resolve_multi_recording_jobs,
    resolve_single_recording_jobs,
)
from .openmp import verify_openmp_runtime
from .worker import (
    dispatch_multi_recording_job,
    dispatch_single_recording_job,
    load_multi_recording_configuration,
    load_single_recording_configuration,
)
from ..layout import (
    OUTPUT_DIRECTORY_NAME,
    MULTI_RECORDING_TRACKER_FILENAME,
    SINGLE_RECORDING_TRACKER_FILENAME,
    resolve_plane_specifier,
)

if TYPE_CHECKING:
    from pathlib import Path


def run_single_recording_pipeline(
    configuration_path: Path,
    job_id: str | None = None,
    *,
    binarize: bool = False,
    register: bool = False,
    process: bool = False,
    combine: bool = False,
    target_plane: int = -1,
    binarization_workers: int | None = None,
    registration_workers: int | None = None,
    processing_workers: int | None = None,
    registration_device: int | None = None,
) -> None:
    """Executes the requested single-recording processing pipeline steps for the target data.

    The caller is responsible for writing all path overrides (``file_io.data_path``, ``file_io.output_path``) and the
    ``runtime.display_progress_bars`` flag into the configuration file before invoking this function. The pipeline
    reads these values from the file at ``configuration_path`` and does not accept them as direct parameters. Each
    stage takes its worker count as a direct parameter, which keeps the configuration file immutable and therefore
    safe to share between concurrently dispatched jobs. An invocation that sets none of the four stage flags runs
    every stage in phase order.

    Args:
        configuration_path: The path to the single-recording configuration YAML file.
        job_id: The unique hexadecimal identifier for the processing job to execute. If provided, only the job
            matching this ID is executed. If not provided, all requested jobs are run sequentially.
        binarize: Determines whether to resolve the binary files for plane-specific processing (step 1).
        register: Determines whether to register the target plane(s) to remove motion and compute the registration
            quality metrics (step 2).
        process: Determines whether to process the target plane(s) to discover ROIs and extract their fluorescence
            (step 3).
        combine: Determines whether to combine processed plane data into a uniform dataset (step 4).
        target_plane: The index of the plane to register and process. Setting this to '-1' processes all available
            planes sequentially.
        binarization_workers: The number of parallel workers to allocate to the binarization stage. Use None to accept
            the stage default and -1 to request every available core.
        registration_workers: The number of parallel workers to allocate to each plane-registration job. Use None to
            accept the stage default and -1 to request every available core.
        processing_workers: The number of parallel workers to allocate to each plane-processing job. Use None to accept
            the stage default and -1 to request every available core.
        registration_device: The zero-based index of the CUDA device on which each plane-registration job runs. Use None
            to register every plane on the host CPU.

    Raises:
        FileNotFoundError: If the single-recording configuration data cannot be loaded from the specified file.
        RuntimeError: If the host is macOS and carries no loadable OpenMP runtime for the Numba threading layer, or if
            a registration device is named and the host exposes no usable CUDA device.
        ValueError: If the recording's data validation fails, the specified job_id does not match any available job,
            target_plane names a plane the recording does not hold, or registration_device names a CUDA device index
            the host does not expose.
    """
    # Every stage below reaches a parallelized kernel, so a host whose threading layer has no runtime to load fails
    # here rather than partway through a recording.
    verify_openmp_runtime()

    configuration, output_path = load_single_recording_configuration(configuration_path=configuration_path)

    # Maps each worker-consuming stage to its requested allocation so that every job reads the count intended for its
    # own stage. The lookups below resolve a combination job to None.
    stage_workers: dict[str, int | None] = {
        SingleRecordingJobNames.BINARIZE: binarization_workers,
        SingleRecordingJobNames.REGISTER: registration_workers,
        SingleRecordingJobNames.PROCESS: processing_workers,
    }

    # Resolves RuntimeContext instances for all planes upfront. This determines the plane count without requiring
    # binarization to run first, mirroring how run_multi_recording_pipeline resolves contexts before building jobs.
    # In REMOTE mode (job_id provided, i.e., when dispatched by the MCP job executor), disables bootstrap persistence
    # because the prepare tool already wrote the shared configuration and per-plane runtime_data.yaml files
    # single-threaded. Skipping the per-worker re-save keeps a worker from overwriting every peer plane's file with
    # its own stale snapshot, since this resolver builds a context per plane rather than per dispatched job.
    contexts = resolve_single_recording_contexts(configuration=configuration, persist=job_id is None)
    plane_count = len(contexts)

    # Derives the tracker path from the configuration. The tracker lives under the cindra/ subdirectory, consistent
    # with where batch tools create it and where get_recording_status_tool looks for it.
    tracker_path: Path = output_path / OUTPUT_DIRECTORY_NAME / SINGLE_RECORDING_TRACKER_FILENAME

    # The insertion order of this dictionary sequences the jobs a LOCAL-mode invocation runs, so the registration
    # entry must sit between the binarization and processing entries. Otherwise a run that requests no flags detects
    # ROIs before it removes motion.
    requested_jobs: dict[str, bool] = {
        SingleRecordingJobNames.BINARIZE: binarize,
        SingleRecordingJobNames.REGISTER: register,
        SingleRecordingJobNames.PROCESS: process,
        SingleRecordingJobNames.COMBINE: combine,
    }

    if not any(requested_jobs.values()):
        requested_jobs = dict.fromkeys(requested_jobs, True)

    jobs_to_run = [job_name for job_name, requested in requested_jobs.items() if requested]

    # Builds the universe of every valid job the pipeline could execute for this configuration. REGISTER and PROCESS
    # always expand to every available plane regardless of ``target_plane`` so that foreign-entry detection treats
    # the universe as a configuration fingerprint, not an invocation fingerprint.
    universe: list[tuple[str, str]] = resolve_single_recording_jobs(plane_count=plane_count)

    tracker = ProcessingTracker(file_path=tracker_path)

    if job_id is not None:
        # REMOTE mode.
        resolved_name, resolved_specifier = tracker.resolve_job(job_id=job_id, universe=universe)

        _verify_registration_device(device=registration_device, job_names=[resolved_name])

        tracker.align_jobs(jobs=universe, universe=universe)

        dispatch_single_recording_job(
            configuration=configuration,
            job_name=SingleRecordingJobNames(resolved_name),
            specifier=resolved_specifier,
            job_id=job_id,
            tracker=tracker,
            workers=stage_workers.get(resolved_name),
            device=registration_device,
        )
    else:
        # LOCAL mode.

        # Rejects a plane the recording does not hold before the tracker is aligned. Without this guard the
        # out-of-range job pair reaches align_jobs, which rejects it against the universe with a message naming job
        # identifiers rather than the plane index for which the caller actually asked.
        if target_plane != -1 and target_plane not in range(plane_count):
            message = (
                f"Unable to run the single-recording cindra processing pipeline. The requested 'target_plane' must be "
                f"an index of one of the {plane_count} plane(s) the recording holds, or -1 to process every plane, "
                f"but encountered {target_plane}."
            )
            console.error(message=message, error=ValueError)

        _verify_registration_device(device=registration_device, job_names=jobs_to_run)

        jobs: list[tuple[str, str]] = []
        for base_job_name in jobs_to_run:
            if base_job_name in PER_PLANE_JOB_NAMES:
                if target_plane == -1:
                    jobs.extend(
                        (base_job_name, resolve_plane_specifier(plane_index=plane_index))
                        for plane_index in range(plane_count)
                    )
                else:
                    jobs.append((base_job_name, resolve_plane_specifier(plane_index=target_plane)))
            else:
                jobs.append((base_job_name, ""))

        tracker.align_jobs(jobs=jobs, universe=universe)

        for name, specifier in jobs:
            dispatch_single_recording_job(
                configuration=configuration,
                job_name=SingleRecordingJobNames(name),
                specifier=specifier,
                job_id=ProcessingTracker.generate_job_id(job_name=name, specifier=specifier),
                tracker=tracker,
                workers=stage_workers.get(name),
                device=registration_device,
            )

    console.echo(message="Single-recording processing: Complete.", level=LogLevel.SUCCESS)


def run_multi_recording_pipeline(
    configuration_path: Path,
    job_id: str | None = None,
    *,
    discover: bool = False,
    extract: bool = False,
    target_recording: str | None = None,
    discovery_workers: int | None = None,
    extraction_workers: int | None = None,
) -> None:
    """Executes the requested multi-recording processing pipeline steps for the target data.

    The caller is responsible for writing all runtime overrides (``recording_io.recording_directories``,
    ``runtime.display_progress_bars``) into the configuration file before invoking this function. The pipeline reads
    these values from the file at ``configuration_path`` and does not accept them as direct parameters. Each stage
    takes its worker count as a direct parameter, which keeps the configuration file immutable and therefore safe to
    share between concurrently dispatched jobs. An invocation that sets neither stage flag runs both stages in phase
    order.

    Args:
        configuration_path: The path to the multi-recording configuration YAML file. The configuration must include the
            ``recording_io.recording_directories`` list of recording paths and ``recording_io.dataset_name``.
        job_id: The unique hexadecimal identifier for the processing job to execute. If provided, only the job
            matching this ID is executed. If not provided, all requested jobs are run sequentially.
        discover: Determines whether to discover ROIs whose activity can be tracked across recordings (step 1).
        extract: Determines whether to extract fluorescence from the ROIs tracked across multiple recordings (step 2).
        target_recording: The unique identifier of the recording to process when running the 'extract' job. If None,
            processes all recordings.
        discovery_workers: The number of parallel workers to allocate to the discovery stage. Use None to accept the
            stage default and -1 to request every available core.
        extraction_workers: The number of parallel workers to allocate to each per-recording extraction job. Use None
            to accept the stage default and -1 to request every available core.

    Raises:
        FileNotFoundError: If the multi-recording configuration data cannot be loaded from the specified file, or if a
            recording directory holds no combined_metadata.npz file. It is also raised when a job_id is supplied and a
            recording carries no multi_recording_runtime_data.yaml file, which prepare_multi_recording_batch_tool writes
            before any worker is dispatched.
        RuntimeError: If the host is macOS and carries no loadable OpenMP runtime for the Numba threading layer. It is
            also raised when a recording directory holds multiple combined_metadata.npz files, when the recording paths
            do not contain unique identifying components, or when a resolved identifying component contains a colon.
        ValueError: If recording validation fails, recording_directories names fewer than two recordings,
            target_recording does not name a resolved recording, or the specified job_id does not match any available
            jobs.
    """
    # Every stage below reaches a parallelized kernel, so a host whose threading layer has no runtime to load fails
    # here rather than partway through a dataset.
    verify_openmp_runtime()

    configuration = load_multi_recording_configuration(configuration_path=configuration_path)

    # Maps each stage to its requested allocation so that every job reads the count intended for its own stage.
    stage_workers: dict[str, int | None] = {
        MultiRecordingJobNames.DISCOVER: discovery_workers,
        MultiRecordingJobNames.EXTRACT: extraction_workers,
    }

    console.echo(
        message=f"Processing {len(configuration.recording_io.recording_directories)} recordings for dataset "
        f"'{configuration.recording_io.dataset_name}'...",
    )

    # Resolves MultiRecordingRuntimeContext instances to extract recording IDs and the main recording output
    # path. This also validates that all recording directories contain valid single-recording outputs and
    # handles relocated data. In REMOTE mode (job_id provided, i.e., when dispatched by the MCP job executor),
    # disables bootstrap persistence because the prepare tool already wrote the shared configuration and every
    # recording's multi_recording_runtime_data.yaml single-threaded. Skipping the per-worker re-save keeps a worker
    # from overwriting every peer recording's file with its own stale snapshot.
    contexts = resolve_multi_recording_contexts(configuration=configuration, persist=job_id is None)
    recording_ids: list[str] = [context.runtime.io.recording_id for context in contexts]
    main_recording_path = contexts[0].runtime.output_path
    if main_recording_path is None:
        message = (
            "Unable to run the multi-recording pipeline. The main recording's "
            "output path is not configured in the resolved runtime context."
        )
        console.error(message=message, error=ValueError)

    requested_jobs: dict[str, bool] = {
        MultiRecordingJobNames.DISCOVER: discover,
        MultiRecordingJobNames.EXTRACT: extract,
    }

    if not any(requested_jobs.values()):
        requested_jobs = dict.fromkeys(requested_jobs, True)

    jobs_to_run = [job_name for job_name, requested in requested_jobs.items() if requested]

    # Builds the universe of every valid job the pipeline could execute for this configuration. EXTRACT always
    # expands to every resolved recording ID regardless of ``target_recording`` so that foreign-entry detection
    # treats the universe as a configuration fingerprint, not an invocation fingerprint.
    universe: list[tuple[str, str]] = resolve_multi_recording_jobs(recording_ids=recording_ids)

    tracker = ProcessingTracker(file_path=main_recording_path.joinpath(MULTI_RECORDING_TRACKER_FILENAME))

    if job_id is not None:
        # REMOTE mode.
        resolved_name, resolved_specifier = tracker.resolve_job(job_id=job_id, universe=universe)

        tracker.align_jobs(jobs=universe, universe=universe)

        dispatch_multi_recording_job(
            configuration=configuration,
            job_name=MultiRecordingJobNames(resolved_name),
            specifier=resolved_specifier,
            job_id=job_id,
            tracker=tracker,
            workers=stage_workers[resolved_name],
        )
    else:
        # LOCAL mode.

        # Rejects a recording the dataset does not span before the tracker is aligned. The context resolver applies
        # the same check, but only once the extraction job reaches it, which is after align_jobs would have rejected
        # the unknown job pair with a message naming job identifiers rather than the requested recording.
        if target_recording is not None and target_recording not in recording_ids:
            message = (
                f"Unable to run the multi-recording cindra processing pipeline. The requested 'target_recording' must "
                f"name one of the recordings the dataset spans, but encountered '{target_recording}'. Resolved "
                f"recording identifiers: {recording_ids}."
            )
            console.error(message=message, error=ValueError)

        jobs: list[tuple[str, str]] = []
        for base_job_name in jobs_to_run:
            if base_job_name == MultiRecordingJobNames.EXTRACT:
                if target_recording is None:
                    jobs.extend((MultiRecordingJobNames.EXTRACT, recording_id) for recording_id in recording_ids)
                else:
                    jobs.append((MultiRecordingJobNames.EXTRACT, target_recording))
            else:
                jobs.append((base_job_name, ""))

        tracker.align_jobs(jobs=jobs, universe=universe)

        for name, specifier in jobs:
            dispatch_multi_recording_job(
                configuration=configuration,
                job_name=MultiRecordingJobNames(name),
                specifier=specifier,
                job_id=ProcessingTracker.generate_job_id(job_name=name, specifier=specifier),
                tracker=tracker,
                workers=stage_workers[name],
            )

    console.echo(message="Multi-recording processing: Complete.", level=LogLevel.SUCCESS)


def _verify_registration_device(device: int | None, job_names: list[str]) -> None:
    """Verifies that the host exposes the CUDA device on which a registration job of this invocation runs.

    Notes:
        The verification precedes the first dispatch, so a host that exposes no such device aborts the invocation
        having done no work rather than at the point the registration reaches the device.

    Args:
        device: The zero-based index of the CUDA device on which the registration jobs run, or None while they run on
            the host CPU.
        job_names: The names of the jobs this invocation runs.

    Raises:
        RuntimeError: If a registration device is named and the host exposes no usable CUDA device.
        ValueError: If the named device index is one the host does not expose.
    """
    if device is None or SingleRecordingJobNames.REGISTER not in job_names:
        return

    verify_gpu_runtime(device=device)
