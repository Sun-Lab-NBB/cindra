from pathlib import Path
from dataclasses import dataclass

from .multi_recording_data import MultiRecordingRuntimeData as MultiRecordingRuntimeData
from .single_recording_data import (
    CombinedData as CombinedData,
    SingleRecordingRuntimeData as SingleRecordingRuntimeData,
)
from .multi_recording_configuration import MultiRecordingConfiguration as MultiRecordingConfiguration
from .single_recording_configuration import (
    AcquisitionParameters as AcquisitionParameters,
    SingleRecordingConfiguration as SingleRecordingConfiguration,
)

@dataclass
class RuntimeContext:
    configuration: SingleRecordingConfiguration
    acquisition: AcquisitionParameters
    runtime: SingleRecordingRuntimeData
    def save_shared(self) -> None: ...
    def save_runtime(self) -> None: ...
    @classmethod
    def load(cls, root_path: Path, plane_index: int = -1) -> RuntimeContext | list[RuntimeContext]: ...

@dataclass
class MultiRecordingRuntimeContext:
    configuration: MultiRecordingConfiguration
    runtime: MultiRecordingRuntimeData
    def save_shared(self) -> None: ...
    def save_runtime(self) -> None: ...
    @classmethod
    def load(
        cls, root_path: Path, recording_index: int = -1
    ) -> MultiRecordingRuntimeContext | list[MultiRecordingRuntimeContext]: ...

def _load_single_recording_runtime(plane_directory: Path) -> SingleRecordingRuntimeData: ...
def _compute_relocation_prefixes(old_path: Path, new_path: Path) -> tuple[Path, Path]: ...
def _relocate_cross_recording_path(path: Path, old_prefix: Path, new_prefix: Path) -> Path: ...
def _relocate_runtime_paths(
    runtime: SingleRecordingRuntimeData | MultiRecordingRuntimeData, old_prefix: Path, new_prefix: Path
) -> None: ...
def _load_multi_recording_data(runtime: MultiRecordingRuntimeData) -> None: ...
