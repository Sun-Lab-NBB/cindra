from pathlib import Path

import numpy as np
from tifffile import TiffFile
from numpy.typing import NDArray as NDArray

from .binary import BinaryFile as BinaryFile
from .context import find_data_directory as find_data_directory
from ..dataclasses import (
    RuntimeContext as RuntimeContext,
    AcquisitionParameters as AcquisitionParameters,
)

TIFF_EXTENSIONS: tuple[str, ...]
_MULTIDIMENSIONAL_PROCESSING_THRESHOLD: int

def convert_tiffs_to_binary(contexts: list[RuntimeContext]) -> None: ...
def _discover_tiff_files(data_directory: Path, ignored_file_names: tuple[str, ...] = ()) -> list[Path]: ...
def _read_tiff(tiff: TiffFile, start_index: int, batch_size: int) -> NDArray[np.int16] | None: ...
def _get_frame_dimensions(
    tiff_files: list[Path], contexts: list[RuntimeContext], acquisition: AcquisitionParameters
) -> tuple[list[int], list[int]]: ...
def _create_binary_files(
    contexts: list[RuntimeContext], frame_heights: list[int], frame_widths: list[int], frames_per_plane: int
) -> tuple[list[BinaryFile], list[BinaryFile]]: ...
