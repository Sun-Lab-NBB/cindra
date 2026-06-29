"""Provides shared synthetic-data fixtures for the cindra integration tests.

The fixtures in this module assemble minimal but fully wired RuntimeContext objects backed by on-disk synthetic binary
movies. They allow the stage-level pipeline entry points (registration, detection, extraction, combination) to run
end-to-end against small, predictable inputs without requiring real acquisition data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import pytest
from ataraxis_base_utilities import ensure_directory_exists

from cindra.io import BinaryFile
from cindra.dataclasses import (
    RuntimeContext,
    AcquisitionParameters,
    SingleRecordingRuntimeData,
    SingleRecordingConfiguration,
)

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

_DEFAULT_FRAME_HEIGHT: int = 48
"""The default synthetic frame height in pixels for single-recording fixtures."""

_DEFAULT_FRAME_WIDTH: int = 48
"""The default synthetic frame width in pixels for single-recording fixtures."""

_DEFAULT_FRAME_COUNT: int = 40
"""The default synthetic frame count for single-recording fixtures."""


def write_binary_movie(file_path: Path, movie: NDArray[np.int16]) -> None:
    """Writes a synthetic movie array to a raw int16 binary file readable by BinaryFile.

    Args:
        file_path: The destination path for the binary file.
        movie: The synthetic movie with shape (frames, height, width). Cast to int16 before writing.
    """
    movie.astype(np.int16).tofile(file_path)


@pytest.fixture
def single_recording_context() -> Callable[..., RuntimeContext]:
    """Returns a factory that builds a minimal single-recording RuntimeContext backed by a synthetic binary movie.

    The returned factory accepts keyword overrides for the frame geometry, the synthetic movie, the channel-2 movie,
    and a configuration mutator callback. It writes the movie to a channel_1_data.bin file inside a plane_0 output
    directory and wires the IOData, configuration, and acquisition parameters so that the registration, detection, and
    extraction stage entry points can run against the context.
    """

    def _make(
        tmp_path: Path,
        *,
        frame_height: int = _DEFAULT_FRAME_HEIGHT,
        frame_width: int = _DEFAULT_FRAME_WIDTH,
        frame_count: int = _DEFAULT_FRAME_COUNT,
        movie: NDArray[np.int16] | None = None,
        movie_channel_2: NDArray[np.int16] | None = None,
        configure: Callable[[SingleRecordingConfiguration], None] | None = None,
    ) -> RuntimeContext:
        output_root = tmp_path / "output"
        plane_directory = output_root / "cindra" / "plane_0"
        ensure_directory_exists(plane_directory)

        if movie is None:
            generator = np.random.default_rng(seed=1234)
            movie = generator.integers(low=100, high=1000, size=(frame_count, frame_height, frame_width)).astype(
                np.int16
            )
        binary_path = plane_directory / "channel_1_data.bin"
        write_binary_movie(file_path=binary_path, movie=movie)

        binary_path_channel_2: Path | None = None
        if movie_channel_2 is not None:
            binary_path_channel_2 = plane_directory / "channel_2_data.bin"
            write_binary_movie(file_path=binary_path_channel_2, movie=movie_channel_2)

        configuration = SingleRecordingConfiguration()
        configuration.file_io.output_path = output_root
        configuration.file_io.data_path = None
        configuration.runtime.parallel_workers = 1
        configuration.runtime.display_progress_bars = False
        configuration.registration.registration_metric_principal_components = 0
        configuration.nonrigid_registration.enabled = False
        configuration.one_photon_registration.enabled = False
        if movie_channel_2 is not None:
            configuration.main.two_channels = True
        if configure is not None:
            configure(configuration)

        acquisition = AcquisitionParameters(frame_rate=30.0)

        runtime = SingleRecordingRuntimeData()
        runtime.output_path = plane_directory
        runtime.io.frame_height = frame_height
        runtime.io.frame_width = frame_width
        runtime.io.frame_count = frame_count
        runtime.io.sampling_rate = 30.0
        runtime.io.plane_index = 0
        runtime.io.output_path = plane_directory
        runtime.io.registered_binary_path = binary_path
        runtime.io.registered_binary_path_channel_2 = binary_path_channel_2

        return RuntimeContext(configuration=configuration, acquisition=acquisition, runtime=runtime)

    return _make


@pytest.fixture
def gaussian_blob_image() -> Callable[..., NDArray[np.float64]]:
    """Returns a builder for a smooth structured image of Gaussian blobs on a flat background.

    Smooth, structured content is required for phase-correlation registration and for detection to localize ROIs;
    white noise produces ambiguous correlation peaks. The returned builder accepts the frame geometry, blob centers,
    blob radius, amplitude, and background level.
    """

    def _make(
        *,
        height: int = _DEFAULT_FRAME_HEIGHT,
        width: int = _DEFAULT_FRAME_WIDTH,
        centers: tuple[tuple[int, int], ...] = ((12, 12), (30, 18), (20, 34), (36, 36)),
        sigma: float = 3.0,
        amplitude: float = 900.0,
        background: float = 100.0,
    ) -> NDArray[np.float64]:
        rows, columns = np.mgrid[0:height, 0:width]
        image = np.full((height, width), background, dtype=np.float64)
        for center_row, center_column in centers:
            squared_distance = (rows - center_row) ** 2 + (columns - center_column) ** 2
            image += amplitude * np.exp(-squared_distance / (2.0 * sigma**2))
        return image

    return _make


@pytest.fixture
def read_binary_movie() -> Callable[[Path, int, int], NDArray[np.int16]]:
    """Returns a helper that reads a binary movie back from disk as an int16 array for output verification."""

    def _read(file_path: Path, frame_height: int, frame_width: int) -> NDArray[np.int16]:
        with BinaryFile(height=frame_height, width=frame_width, file_path=file_path, read_only=True) as binary_file:
            # Copies the data out of the memory map before the context manager closes the underlying file.
            return np.array(binary_file.data, dtype=np.int16)

    return _read
