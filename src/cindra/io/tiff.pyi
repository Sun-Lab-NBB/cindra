from pathlib import Path
from dataclasses import dataclass

import numpy as np
from tifffile import TiffFile
from numpy.typing import NDArray as NDArray

from .binary import (
    BinaryFile as BinaryFile,
    clear_binarization_marker as clear_binarization_marker,
    clear_registration_marker as clear_registration_marker,
    create_binarization_marker as create_binarization_marker,
)
from .context import find_data_directory as find_data_directory
from ..dataclasses import (
    RuntimeContext as RuntimeContext,
    AcquisitionParameters as AcquisitionParameters,
)

TIFF_EXTENSIONS: tuple[str, ...]
TIFF_DECODE_CEILING: int
_INTERNAL_ELEMENT_BYTES: int
_MULTIDIMENSIONAL_PROCESSING_THRESHOLD: int
_MISMATCH_REPORT_LIMIT: int

@dataclass(frozen=True, slots=True)
class SourceFrameGeometry:
    frame_height: int
    frame_width: int
    element_bytes: int
    frame_count: int

@dataclass(frozen=True, slots=True)
class TiffConversionPlan:
    contexts: tuple[RuntimeContext, ...]
    tiff_files: tuple[Path, ...]
    total_frames: int
    converted_frames: int
    batch_size: int
    decode_workers: int
    frame_heights: tuple[int, ...]
    frame_widths: tuple[int, ...]
    channel_1_paths: tuple[Path, ...]
    channel_2_paths: tuple[Path, ...]
    channel_1_frame_counts: tuple[int, ...]
    channel_2_frame_counts: tuple[int, ...]

def resolve_source_frame_geometry(
    data_directory: Path, ignored_file_names: tuple[str, ...] = ()
) -> SourceFrameGeometry: ...
def resolve_tiff_conversion_plan(contexts: list[RuntimeContext], *, workers: int) -> TiffConversionPlan: ...
def convert_tiffs_to_binary(plan: TiffConversionPlan) -> None: ...
def _discover_tiff_files(data_directory: Path, ignored_file_names: tuple[str, ...] = ()) -> list[Path]: ...
def _collect_tiff_files(data_directory: Path, ignored_file_names: tuple[str, ...] = ()) -> list[Path]: ...
def _read_tiff(tiff: TiffFile, start_index: int, batch_size: int, decode_workers: int) -> NDArray[np.int16] | None: ...
def _write_interleave_selection(
    frames: NDArray[np.int16],
    first_frame_index: int,
    interleave_stride: int,
    roi_lines: tuple[int, ...],
    binary: BinaryFile,
    write_index: int,
    mean_image: NDArray[np.float32] | None,
) -> tuple[int, NDArray[np.float32] | None]: ...
def _scan_source_frames(tiff_files: list[Path]) -> tuple[int, int, int]: ...
def _resolve_plane_dimensions(
    contexts: list[RuntimeContext], acquisition: AcquisitionParameters, base_height: int, base_width: int
) -> tuple[list[int], list[int]]: ...
def _resolve_interleave_frame_count(total_frames: int, interleave_stride: int) -> int: ...
def _validate_interleave_frame_count(
    data_directory: Path, total_frames: int, interleave_stride: int, plane_frame_count: int
) -> None: ...
def _resolve_functional_channel_index(context: RuntimeContext) -> int: ...
def _resolve_binary_paths(contexts: list[RuntimeContext]) -> tuple[tuple[Path, ...], tuple[Path, ...]]: ...
def _create_binary_files(plan: TiffConversionPlan) -> tuple[list[BinaryFile], list[BinaryFile]]: ...
