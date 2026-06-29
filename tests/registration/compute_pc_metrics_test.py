"""Contains integration tests for the compute_pc_metrics stage entry point."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from cindra.registration.metrics import compute_pc_metrics

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable

    from numpy.typing import NDArray

    from cindra.dataclasses import RuntimeContext, SingleRecordingConfiguration

_BLOB_CENTERS: tuple[tuple[int, int], ...] = ((32, 32), (76, 44), (50, 90), (96, 96))
"""Blob centers for a 128x128 synthetic frame with distinct, well-separated structure for phase correlation."""

_FRAME_COUNT: int = 36
"""The synthetic frame count for the registration-metric movies."""


def _metric_movie(gaussian_blob_image: Callable[..., NDArray[np.float64]]) -> NDArray[np.int16]:
    """Builds a structured movie with temporal intensity variation so PCA finds a non-degenerate component."""
    base = gaussian_blob_image(height=128, width=128, centers=_BLOB_CENTERS, sigma=4.0, amplitude=2000.0)
    generator = np.random.default_rng(seed=2024)
    movie = np.empty((_FRAME_COUNT, 128, 128), dtype=np.int16)
    scales = np.linspace(start=0.7, stop=1.3, num=_FRAME_COUNT)
    for index in range(_FRAME_COUNT):
        frame = base * scales[index] + generator.normal(loc=0.0, scale=5.0, size=base.shape)
        movie[index] = frame.astype(np.int16)
    return movie


def _registered_context(
    tmp_path: Path,
    single_recording_context: Callable[..., RuntimeContext],
    gaussian_blob_image: Callable[..., NDArray[np.float64]],
    configure: Callable[[SingleRecordingConfiguration], None] | None = None,
) -> RuntimeContext:
    """Builds a context with a structured movie and valid crop ranges ready for metric computation."""
    movie = _metric_movie(gaussian_blob_image)
    context = single_recording_context(
        tmp_path, frame_height=128, frame_width=128, frame_count=_FRAME_COUNT, movie=movie, configure=configure
    )
    context.runtime.registration.valid_y_range = (8, 120)
    context.runtime.registration.valid_x_range = (8, 120)
    return context


def _assert_metric_outputs(context: RuntimeContext) -> None:
    """Asserts that the three principal-component metric arrays are present, correctly shaped, and finite."""
    extreme = context.runtime.registration.principal_component_extreme_images
    projections = context.runtime.registration.principal_component_projections
    shift = context.runtime.registration.principal_component_shift_metrics
    assert extreme is not None
    assert projections is not None
    assert shift is not None
    assert extreme.shape == (2, 3, 112, 112)
    assert projections.shape == (_FRAME_COUNT, 3)
    assert shift.shape == (3, 3)
    assert np.all(np.isfinite(extreme))
    assert np.all(np.isfinite(shift))
    assert np.all(shift >= 0)


class TestComputePcMetrics:
    """Tests compute_pc_metrics."""

    def test_computes_metric_outputs(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
    ) -> None:
        """Verifies that compute_pc_metrics writes the extreme images, projections, and rigid shift metrics."""

        def configure(configuration: SingleRecordingConfiguration) -> None:
            configuration.registration.registration_metric_principal_components = 3
            configuration.nonrigid_registration.enabled = False

        context = _registered_context(tmp_path, single_recording_context, gaussian_blob_image, configure=configure)

        compute_pc_metrics(context=context)

        _assert_metric_outputs(context)

    def test_computes_metric_outputs_with_nonrigid(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
    ) -> None:
        """Verifies that compute_pc_metrics also computes nonrigid shift metrics when nonrigid is enabled."""

        def configure(configuration: SingleRecordingConfiguration) -> None:
            configuration.registration.registration_metric_principal_components = 3
            configuration.nonrigid_registration.enabled = True
            configuration.nonrigid_registration.block_size = (32, 32)

        context = _registered_context(tmp_path, single_recording_context, gaussian_blob_image, configure=configure)

        compute_pc_metrics(context=context)

        _assert_metric_outputs(context)

    def test_applies_bidirectional_correction(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
    ) -> None:
        """Verifies that compute_pc_metrics applies an outstanding bidirectional phase correction to the PC extremes."""

        def configure(configuration: SingleRecordingConfiguration) -> None:
            configuration.registration.registration_metric_principal_components = 3
            configuration.nonrigid_registration.enabled = False

        context = _registered_context(tmp_path, single_recording_context, gaussian_blob_image, configure=configure)
        # Marks an outstanding (un-applied) bidirectional phase offset so the correction branch executes.
        context.runtime.registration.bidirectional_phase_offset = 2
        context.runtime.registration.bidirectional_phase_corrected = False

        compute_pc_metrics(context=context)

        _assert_metric_outputs(context)

    def test_computes_metrics_in_one_photon_mode(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
    ) -> None:
        """Verifies that compute_pc_metrics applies one-photon preprocessing to the PC extremes when enabled."""

        def configure(configuration: SingleRecordingConfiguration) -> None:
            configuration.registration.registration_metric_principal_components = 3
            configuration.nonrigid_registration.enabled = False
            configuration.one_photon_registration.enabled = True
            configuration.one_photon_registration.pre_smoothing_sigma = 2.0

        context = _registered_context(tmp_path, single_recording_context, gaussian_blob_image, configure=configure)

        compute_pc_metrics(context=context)

        _assert_metric_outputs(context)

    def test_raises_when_binary_path_unset(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
    ) -> None:
        """Verifies that compute_pc_metrics raises a ValueError when the registered binary path is missing."""
        context = _registered_context(tmp_path, single_recording_context, gaussian_blob_image)
        context.runtime.io.registered_binary_path = None

        with pytest.raises(ValueError, match="Unable to compute the registration quality metrics"):
            compute_pc_metrics(context=context)

    def test_raises_when_binary_file_missing(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
    ) -> None:
        """Verifies that compute_pc_metrics raises a FileNotFoundError when the registered binary file is absent."""
        context = _registered_context(tmp_path, single_recording_context, gaussian_blob_image)
        context.runtime.io.registered_binary_path = tmp_path / "does_not_exist.bin"

        with pytest.raises(FileNotFoundError):
            compute_pc_metrics(context=context)
