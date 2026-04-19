from types import TracebackType
from typing import Any, Self
from pathlib import Path

import numpy as np
from numpy.typing import NDArray as NDArray

_INT16_MAX_VALUE: int
_DEFAULT_BIN_BATCH_SIZE: int

class BinaryFile:
    height: int
    width: int
    file_path: Path
    dtype: str
    _read_only: bool
    file: np.memmap[Any, np.dtype[np.int16]]
    def __init__(
        self,
        height: int,
        width: int,
        file_path: str | Path,
        frame_number: int = 0,
        dtype: str = "int16",
        read_only: bool = False,
    ) -> None: ...
    @staticmethod
    def convert_numpy_file_to_binary(source_file_name: Path, destination_file_name: Path) -> None: ...
    @property
    def bytes_per_frame(self) -> int: ...
    @property
    def byte_number(self) -> int: ...
    @property
    def frame_number(self) -> int: ...
    @property
    def shape(self) -> tuple[int, int, int]: ...
    @property
    def size(self) -> np.int64: ...
    def close(self) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        execution_type: type[BaseException] | None,
        execution_value: BaseException | None,
        execution_traceback: TracebackType | None,
    ) -> None: ...
    def __setitem__(self, indices: slice | int | tuple[int, ...] | NDArray[Any], data: NDArray[np.int16]) -> None: ...
    def __getitem__(self, indices: slice | int | tuple[int, ...] | NDArray[Any]) -> NDArray[np.int16]: ...
    @property
    def data(self) -> NDArray[np.int16]: ...
    def subsample_movie(
        self, sample_count: int, x_range: tuple[int, int] | None = None, y_range: tuple[int, int] | None = None
    ) -> NDArray[np.float32]: ...
    def bin_movie(
        self,
        bin_size: int,
        x_range: tuple[int, int] | None = None,
        y_range: tuple[int, int] | None = None,
        bad_frames: NDArray[np.bool_] | None = None,
        reject_threshold: float = 0.5,
    ) -> NDArray[np.float32]: ...
    def write_tiff(
        self,
        file_name: Path,
        frame_range: slice | None = None,
        y_range: slice | None = None,
        x_range: slice | None = None,
    ) -> None: ...

class BinaryFileCombined:
    height: int
    width: int
    plane_heights: NDArray[np.uint16]
    plane_widths: NDArray[np.uint16]
    plane_y_coordinates: NDArray[np.int32]
    plane_x_coordinates: NDArray[np.int32]
    file_paths: tuple[Path, ...]
    files: list[BinaryFile]
    def __init__(
        self,
        height: int,
        width: int,
        plane_heights: NDArray[np.uint16],
        plane_widths: NDArray[np.uint16],
        plane_y_coordinates: NDArray[np.int32],
        plane_x_coordinates: NDArray[np.int32],
        file_paths: list[Path] | tuple[Path, ...],
    ) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        execution_type: type[BaseException] | None,
        execution_value: BaseException | None,
        execution_traceback: TracebackType | None,
    ) -> None: ...
    def close(self) -> None: ...
    @property
    def byte_number(self) -> NDArray[np.int64]: ...
    @property
    def frame_number(self) -> int: ...
    @property
    def shape(self) -> tuple[int, NDArray[np.uint16], NDArray[np.uint16]]: ...
    def __getitem__(self, indices: slice | int | tuple[int, ...] | NDArray[Any]) -> NDArray[np.int16]: ...
