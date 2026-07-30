from enum import StrEnum

class SingleRecordingJobNames(StrEnum):
    BINARIZE = "binarization"
    REGISTER = "registration"
    PROCESS = "processing"
    COMBINE = "combination"

class MultiRecordingJobNames(StrEnum):
    DISCOVER = "discovery"
    EXTRACT = "extraction"
