"""Provides the offscreen Qt application and the synthetic viewer-data fixtures shared by the GUI tests."""

from __future__ import annotations

import os

# Selects the headless Qt platform plugin. Qt resolves the plugin when the GUI library initializes, so this
# assignment has to run before the PySide6 imports below.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from typing import TYPE_CHECKING
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from cindra.dataclasses import ROIMask, ROIStatistics
from cindra.gui.viewer_context import ViewerData

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

_DATASET_RECORDING_ROOTS: tuple[Path, ...] = (
    Path("/data/recording_a/cindra"),
    Path("/data/recording_b/cindra"),
    Path("/data/recording_c/cindra"),
)
"""The cindra roots of the three recordings the synthetic tracked dataset holds."""

_DATASET_RECORDING_IDS: tuple[str, ...] = ("a", "b", "c")
"""The recording identifiers of the three recordings the synthetic tracked dataset holds."""


@pytest.fixture(scope="session")
def qt_application() -> QApplication:
    """Returns the process-wide offscreen QApplication that every widget-building test needs."""
    application = QApplication.instance()
    if application is None:
        application = QApplication([])
    return application


@pytest.fixture
def single_recording_stub() -> Callable[..., object]:
    """Returns a factory that builds a synthetic stand-in for a loaded SingleRecordingData instance."""

    def _make(**overrides: object) -> _StubSingleRecording:
        return _StubSingleRecording(**overrides)

    return _make


@pytest.fixture
def tracked_viewer_data() -> Callable[..., ViewerData]:
    """Returns a factory that builds viewer data over a three-recording tracked dataset anchored on the third one."""

    def _make(anchor_root: Path = _DATASET_RECORDING_ROOTS[2]) -> ViewerData:
        recordings = [
            _StubMultiRecording(data_path=root, recording_id=identifier)
            for root, identifier in zip(_DATASET_RECORDING_ROOTS, _DATASET_RECORDING_IDS, strict=True)
        ]
        return ViewerData(
            single_recording=_StubSingleRecording(output_path=anchor_root),
            _recordings=recordings,
            _available_datasets=("d",),
            _active_dataset_name="d",
            _current_recording_index=2,
            _loaded_dataset_name="d",
            dataset_name="d",
        )

    return _make


@pytest.fixture
def registration_recording_stub() -> Callable[..., object]:
    """Returns a factory that builds a synthetic stand-in for a registered recording's binary and offset data."""

    def _make(**overrides: object) -> _StubRegistrationRecording:
        return _StubRegistrationRecording(**overrides)

    return _make


@pytest.fixture
def principal_component_stub() -> Callable[..., object]:
    """Returns a factory that builds a synthetic stand-in for a recording's principal component metrics."""

    def _make(**overrides: object) -> _StubPrincipalComponentData:
        return _StubPrincipalComponentData(**overrides)

    return _make


class _StubSingleRecording:
    """Stands in for SingleRecordingData, exposing the attributes the ROI and tracking viewers read while they
    load and render a recording."""

    def __init__(
        self,
        cell_labels: list[float] | None = None,
        roi_count: int = 4,
        frame_count: int = 60,
        frame_size: int = 16,
        output_path: Path = Path("/data/recording_a/cindra"),
    ) -> None:
        generator = np.random.default_rng(seed=42)
        self.frame_height = frame_size
        self.frame_width = frame_size
        self.frame_count = frame_count
        self.roi_count = roi_count
        self.sampling_rate = 30.0
        self.tau = 1.0
        self.aspect_ratio = 1.0
        self.recording_label = "stub_recording"
        self.output_path = output_path
        self.two_channels = False

        if cell_labels is None:
            labels = np.zeros(roi_count, dtype=np.float32)
        else:
            labels = np.asarray(cell_labels, dtype=np.float32)
        self.cell_classification = np.column_stack([labels, generator.random(roi_count).astype(np.float32)])
        self.cell_colocalization = np.empty(0, dtype=np.float32)

        self.cell_fluorescence = generator.random((roi_count, frame_count)).astype(np.float32)
        self.neuropil_fluorescence = generator.random((roi_count, frame_count)).astype(np.float32)
        self.subtracted_fluorescence = generator.random((roi_count, frame_count)).astype(np.float32)
        self.spikes = generator.random((roi_count, frame_count)).astype(np.float32)

        self.mean_image = generator.random((frame_size, frame_size)).astype(np.float32)
        self.enhanced_mean_image = generator.random((frame_size, frame_size)).astype(np.float32)
        self.correlation_map = generator.random((frame_size, frame_size)).astype(np.float32)
        self.maximum_projection = generator.random((frame_size, frame_size)).astype(np.float32)
        self.corrected_structural_mean_image = np.empty(0, dtype=np.float32)
        self.mean_image_channel_2 = np.empty(0, dtype=np.float32)
        self.enhanced_mean_image_channel_2 = np.empty(0, dtype=np.float32)
        self.correlation_map_channel_2 = np.empty(0, dtype=np.float32)
        self.maximum_projection_channel_2 = np.empty(0, dtype=np.float32)

        self.roi_statistics = [
            _make_roi_statistics(roi_index=index, frame_width=frame_size) for index in range(roi_count)
        ]


class _StubMultiRecording:
    """Stands in for MultiRecordingData, exposing the identity attributes the viewers read."""

    def __init__(self, data_path: Path, recording_id: str) -> None:
        self.data_path = data_path
        self.recording_id = recording_id
        self.runtime_dataset_name = "stub_dataset"
        self.has_channel_2 = False


class _StubPlaneBinary:
    """Stands in for one plane's BinaryFile, serving the subsampled movie that scales the binary viewer's display."""

    def __init__(self, frame_count: int, frame_size: int) -> None:
        generator = np.random.default_rng(seed=11)
        self._movie = generator.integers(low=0, high=4096, size=(frame_count, frame_size, frame_size)).astype(np.int16)

    def subsample_movie(self, sample_count: int) -> NDArray[np.int16]:
        """Returns the leading frames of the synthetic movie, capped at the requested sample count."""
        return self._movie[:sample_count]


class _StubCombinedBinary:
    """Stands in for BinaryFileCombined, exposing the per-plane binaries the binary viewer samples."""

    def __init__(self, files: list[_StubPlaneBinary]) -> None:
        self.files = files


class _StubRegistrationRecording:
    """Stands in for SingleRecordingData, exposing the attributes the registered binary viewer reads."""

    def __init__(
        self,
        plane_frame_counts: tuple[int, ...] = (21, 20),
        combined_frame_count: int = 20,
        frame_size: int = 8,
    ) -> None:
        self._plane_frame_counts = plane_frame_counts
        self.frame_count = combined_frame_count
        self.plane_count = len(plane_frame_counts)
        self.sampling_rate = 30.0
        self.aspect_ratio = 1.0
        self.recording_label = "stub_recording"
        self.two_channels = False
        self.output_path = Path("/data/recording_a/cindra")
        self.view_index = -1
        self._frame_size = frame_size
        self.combined_binary = _StubCombinedBinary(
            files=[_StubPlaneBinary(frame_count=count, frame_size=frame_size) for count in plane_frame_counts]
        )

    def switch_view(self, view_index: int) -> None:
        """Records the requested view index the way SingleRecordingData does."""
        self.view_index = view_index

    def plane_rigid_offsets(self, plane_index: int) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
        """Returns per-plane offsets spanning that plane's own frame count, as the registration stage stores them."""
        frame_count = self._plane_frame_counts[plane_index]
        offsets = np.full(frame_count, fill_value=plane_index + 1, dtype=np.int32)
        return offsets, offsets * 2

    def read_stitched_frame(self, frame_index: int) -> NDArray[np.int16]:
        """Returns a synthetic stitched frame for the requested frame index."""
        return np.full((self._frame_size * self.plane_count, self._frame_size), fill_value=frame_index, dtype=np.int16)


class _StubPrincipalComponentData:
    """Stands in for SingleRecordingData, exposing the attributes the principal component viewer reads."""

    def __init__(
        self,
        principal_component_count: int = 3,
        plane_count: int = 2,
        frame_size: int = 8,
        frame_count: int = 20,
    ) -> None:
        generator = np.random.default_rng(seed=7)
        self.principal_component_count = principal_component_count
        self.plane_count = plane_count
        self.view_labels = ("Combined", *(f"Plane {index}" for index in range(plane_count)))
        self.view_index = 0
        self.recording_label = "stub_recording"
        extreme_images = generator.random((2, principal_component_count, frame_size, frame_size))
        self.principal_component_extreme_images = extreme_images.astype(np.float32)
        self.principal_component_shift_metrics = generator.random((principal_component_count, 3)).astype(np.float32)
        self.principal_component_projections = generator.random((frame_count, principal_component_count)).astype(
            np.float32
        )

    def switch_view(self, view_index: int) -> None:
        """Records the requested view index the way SingleRecordingData does."""
        self.view_index = view_index


def _make_roi_statistics(roi_index: int, frame_width: int) -> ROIStatistics:
    """Builds a two-pixel ROIStatistics instance placed on the pair of rows unique to the requested ROI index."""
    y_pixels = np.array([roi_index * 2, roi_index * 2 + 1], dtype=np.int32)
    x_pixels = np.array([1, 2], dtype=np.int32)
    mask = ROIMask(
        y_pixels=y_pixels,
        x_pixels=x_pixels,
        pixel_weights=np.ones(2, dtype=np.float32),
        centroid=(int(y_pixels[0]), int(x_pixels[0])),
        frame_width=frame_width,
        radius=2.0,
    )
    return ROIStatistics(
        mask=mask,
        footprint=1,
        compactness=0.5,
        solidity=0.5,
        pixel_count=2,
        aspect_ratio=1.0,
        normalized_pixel_count=1.0,
        skewness=0.5,
    )
