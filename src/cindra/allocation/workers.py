"""Provides the measured CPU worker defaults for the single and multi-recording pipeline stages and the worker count
resolver.

The defaults encode the knee of each stage's measured scaling curve, so a caller that expresses no preference gets the
allocation that maximizes batch throughput rather than the allocation that minimizes the wall time of one job.
"""

from ataraxis_base_utilities import console, resolve_worker_count

from .job_names import MultiRecordingJobNames, SingleRecordingJobNames

BINARIZATION_WORKERS: int = 4
"""The number of workers allocated to the binarization stage by default, measured as the point where the allocated
cores become the TIFF image decode threads and further cores stop shortening the conversion."""

REGISTRATION_WORKERS: int = 8
"""The number of workers allocated to the registration stage by default, measured on a one-plane worker sweep that
took 477 seconds with 8 workers, 454 seconds with 16, and 447 seconds with 30. Widening 8 to 16 buys 2.9 seconds per
added core and widening 16 to 30 buys 0.5 seconds per added core, placing the knee at 8."""

PROCESSING_WORKERS: int = 10
"""The number of workers allocated to the processing stage by default, measured on a one-plane worker sweep. Detection
held at 114.0, 113.9, and 113.8 seconds for 10, 20, and 30 workers, bound by movie binning IO and a serial detection
loop. Extraction scaled from 61.8 to 44.1 to 38.6 seconds over the same sweep, but running more planes concurrently
outweighs that gain, which places the default at 10."""

DISCOVERY_WORKERS: int = 30
"""The number of workers allocated to the multi-recording discovery stage by default, which is the saturating
allocation the stage is admitted at. The stage registers every recording of one animal against the others, so its cost
grows with the square of the recording count."""

EXTRACTION_WORKERS: int = 16
"""The number of workers allocated to the multi-recording extraction stage by default, measured as the point where the
stage stops shortening. Every frame batch the extraction kernel consumes is read serially before the kernel runs, so
the stage plateaus below the width it is given and further cores are spent waiting on batch reads."""

TIFF_DECODE_CEILING: int = 4
"""The maximum number of TIFF decode threads, measured as the point where added decode threads stop shortening the
conversion. The decode pool never exceeds this value regardless of how many cores the surrounding job holds."""

ALL_CORES_REQUEST: int = -1
"""The requested worker count that asks for every available CPU core."""

_STAGE_WORKER_DEFAULTS: dict[SingleRecordingJobNames | MultiRecordingJobNames, int] = {
    SingleRecordingJobNames.BINARIZE: BINARIZATION_WORKERS,
    SingleRecordingJobNames.REGISTER: REGISTRATION_WORKERS,
    SingleRecordingJobNames.PROCESS: PROCESSING_WORKERS,
    MultiRecordingJobNames.DISCOVER: DISCOVERY_WORKERS,
    MultiRecordingJobNames.EXTRACT: EXTRACTION_WORKERS,
}
"""Maps every single and multi-recording pipeline stage that consumes a worker allocation to its measured default
worker count."""


def resolve_stage_workers(
    job_name: SingleRecordingJobNames | MultiRecordingJobNames,
    requested_workers: int | None = None,
) -> int:
    """Resolves the number of workers to allocate to the target pipeline stage.

    Notes:
        A requested count of None resolves to the measured default for the stage, which is the knee of that stage's
        scaling curve. A requested count of -1 resolves to every available CPU core, minus the cores the ataraxis
        worker resolver holds back for system use. A positive requested count is honored exactly. A requested count of
        zero, or any negative count other than -1, is rejected.

        The single-recording binarization, registration, and processing stages resolve through this function, as do the
        multi-recording discovery and extraction stages. Passing the single-recording combination stage's job name
        raises an error, because that stage takes no worker allocation.

    Args:
        job_name: The single or multi-recording pipeline stage to allocate workers for.
        requested_workers: The number of workers the caller asks for. Use None to accept the measured default for the
            stage and -1 to request every available core.

    Returns:
        The number of workers to allocate to the stage, always at least 1.

    Raises:
        ValueError: If job_name does not name a stage that consumes a worker allocation, or if requested_workers is
            zero or is a negative value other than -1.
    """
    default_workers: int | None = _STAGE_WORKER_DEFAULTS.get(job_name)
    if default_workers is None:
        message = (
            f"Unable to resolve the worker count for the '{job_name}' processing stage. The input job name does not "
            f"name a pipeline stage that consumes a worker allocation. Use one of the valid stage names: "
            f"{[stage.value for stage in _STAGE_WORKER_DEFAULTS]}."
        )
        console.error(message=message, error=ValueError)

    if requested_workers is None:
        return default_workers

    if requested_workers == ALL_CORES_REQUEST:
        return resolve_worker_count(requested_workers=ALL_CORES_REQUEST)

    if requested_workers <= 0:
        message = (
            f"Unable to resolve the worker count for the '{job_name}' processing stage. The requested worker count "
            f"must be a positive integer, -1 to request every available core, or None to accept the measured stage "
            f"default, but encountered {requested_workers}."
        )
        console.error(message=message, error=ValueError)

    return requested_workers
