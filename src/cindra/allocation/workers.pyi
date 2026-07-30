from .job_names import (
    MultiRecordingJobNames as MultiRecordingJobNames,
    SingleRecordingJobNames as SingleRecordingJobNames,
)

BINARIZATION_WORKERS: int
REGISTRATION_WORKERS: int
PROCESSING_WORKERS: int
DISCOVERY_WORKERS: int
EXTRACTION_WORKERS: int
TIFF_DECODE_CEILING: int
ALL_CORES_REQUEST: int
_STAGE_WORKER_DEFAULTS: dict[SingleRecordingJobNames | MultiRecordingJobNames, int]

def resolve_stage_workers(
    job_name: SingleRecordingJobNames | MultiRecordingJobNames, requested_workers: int | None = None
) -> int: ...
