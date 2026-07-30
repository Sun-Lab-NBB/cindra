from pathlib import Path

import numpy as np
from tifffile import TiffFile
from numpy.typing import NDArray as NDArray

from .binary import (
    BinaryFile as BinaryFile,
    clear_registration_marker as clear_registration_marker,
)
from .context import find_data_directory as find_data_directory
from ..allocation import TIFF_DECODE_CEILING as TIFF_DECODE_CEILING
from ..dataclasses import (
    RuntimeContext as RuntimeContext,
    AcquisitionParameters as AcquisitionParameters,
)

TIFF_EXTENSIONS: tuple[str, ...]
_MULTIDIMENSIONAL_PROCESSING_THRESHOLD: int
_MISMATCH_REPORT_LIMIT: int

def convert_tiffs_to_binary(contexts: list[RuntimeContext], *, workers: int) -> None: ...
def _discover_tiff_files(data_directory: Path, ignored_file_names: tuple[str, ...] = ()) -> list[Path]: ...
def _read_tiff(tiff: TiffFile, start_index: int, batch_size: int, decode_workers: int) -> NDArray[np.int16] | None: ...
def _get_frame_dimensions(
    tiff_files: list[Path], contexts: list[RuntimeContext], acquisition: AcquisitionParameters, decode_workers: int
) -> tuple[list[int], list[int]]: ...
def _validate_uniform_frame_shape(tiff_files: list[Path], base_height: int, base_width: int) -> None: ...
def _resolve_interleave_frame_count(total_frames: int, interleave_stride: int, position: int) -> int: ...
def _create_binary_files(
    contexts: list[RuntimeContext],
    frame_heights: list[int],
    frame_widths: list[int],
    channel_1_frame_counts: list[int],
    channel_2_frame_counts: list[int],
) -> tuple[list[BinaryFile], list[BinaryFile]]: ...
