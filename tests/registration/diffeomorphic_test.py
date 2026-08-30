"""Contains tests for the diffeomorphic module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from ataraxis_base_utilities import console, error_format

from cindra.registration.pyramid import ScaleSpacePyramid
from cindra.registration.deformation import Deformation
from cindra.registration.spline_grid import MINIMUM_KNOTS_FOR_FROZEN_EDGES, SplineGrid
from cindra.registration.diffeomorphic import DiffeomorphicDemonsRegistration, _compute_demons_force

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

_ACCURACY_IMAGE_EXTENT: int = 64
"""The frame height and width in pixels used by the registration accuracy tests."""

_ACCURACY_SCALE_SAMPLING: int = 10
"""The number of scale steps per pyramid level used by the registration accuracy tests, measured as the smallest
sampling that reaches stable alignment on the synthetic blob images."""

_ACCURACY_GRID_SAMPLING: float = 8.0
"""The final B-spline grid spacing in pixels used by the registration accuracy tests."""

_ACCURACY_BLOB_CENTERS: tuple[tuple[int, int], ...] = ((20, 20), (20, 44), (44, 20), (44, 44), (32, 32))
"""The blob centers defining the synthetic reference image for the registration accuracy tests. The centers sit away
from the frame edges so that an imposed translation does not move content out of the frame."""

_ACCURACY_BLOB_SIGMA: float = 4.0
"""The Gaussian blob standard deviation in pixels used by the registration accuracy tests."""

_ACCURACY_TRANSLATION: tuple[int, int] = (4, 3)
"""The vertical and horizontal translation in pixels imposed between the two images of the accuracy test pair."""

_ACCURACY_GROUP_TRANSLATIONS: tuple[tuple[int, int], ...] = ((-3, -3), (3, -3), (-3, 3), (3, 3))
"""The per-image translations in pixels imposed on the synthetic recording group in the groupwise accuracy test."""

_ACCURACY_FIELD_MARGIN: int = 8
"""The number of border pixels excluded when measuring displacement and correlation. Excluding them removes the frame
edges, where the deformation is constrained and the warp samples outside the source image."""

_ACCURACY_DISPLACEMENT_TOLERANCE: float = 0.75
"""The maximum accepted error in pixels between the recovered relative displacement and the imposed translation. The
observed error stays at or below 0.25 pixels along both axes of the tested translation."""

_ACCURACY_MINIMUM_CORRELATION: float = 0.95
"""The minimum accepted correlation between warped images after registration. The observed correlation reaches 0.992
or above for every tested configuration."""


class TestComputeDemonsForce:
    """Tests the fused Demons force kernel."""

    @pytest.mark.parametrize(("height", "width"), [(97, 131), (64, 64)])
    @pytest.mark.parametrize(("noise_factor", "speed_factor"), [(1.0, 3.0), (0.5, 1.25), (2.0, 5.0)])
    def test_matches_the_operator_chain(
        self, height: int, width: int, noise_factor: float, speed_factor: float
    ) -> None:
        """Verifies the kernel reproduces the equivalent chain of NumPy operators bit for bit."""
        generator = np.random.default_rng(seed=77)
        fields = [generator.standard_normal((height, width)).astype(np.float32) for _ in range(6)]

        # Flattens one corner of both gradients so the zero gradient-magnitude path is covered. The intensity
        # difference keeps both denominators non-zero there, so the zero-denominator branch stays untaken.
        for index in (1, 2, 4, 5):
            fields[index][:5, :5] = 0.0

        source_image, source_gradient_y, source_gradient_x = fields[0], fields[1], fields[2]
        target_image, target_gradient_y, target_gradient_x = fields[3], fields[4], fields[5]

        source_magnitude = source_gradient_y**2 + source_gradient_x**2
        target_magnitude = target_gradient_y**2 + target_gradient_x**2
        intensity_difference = source_image - target_image
        intensity_difference_squared = intensity_difference**2
        source_denominator = source_magnitude + noise_factor**2 * intensity_difference_squared
        target_denominator = target_magnitude + noise_factor**2 * intensity_difference_squared
        source_denominator[source_denominator == 0] = np.inf
        target_denominator[target_denominator == 0] = np.inf
        speed = -speed_factor
        expected_y = (
            intensity_difference
            * (source_gradient_y / source_denominator + target_gradient_y / target_denominator)
            * speed
        )
        expected_x = (
            intensity_difference
            * (source_gradient_x / source_denominator + target_gradient_x / target_denominator)
            * speed
        )

        field_y = np.empty((height, width), dtype=np.float32)
        field_x = np.empty((height, width), dtype=np.float32)
        _compute_demons_force(
            source_image=source_image,
            source_gradient_y=source_gradient_y,
            source_gradient_x=source_gradient_x,
            target_image=target_image,
            target_gradient_y=target_gradient_y,
            target_gradient_x=target_gradient_x,
            noise_squared=np.float32(noise_factor**2),
            speed=np.float32(speed),
            field_y=field_y,
            field_x=field_x,
        )

        np.testing.assert_array_equal(field_y, expected_y.astype(np.float32))
        np.testing.assert_array_equal(field_x, expected_x.astype(np.float32))


class TestDiffeomorphicDemonsRegistration:
    """Tests the construction, level filtering, caching, and groupwise averaging of the cross-day alignment run."""

    def test_constructor_stores_images(self) -> None:
        """Verifies that the constructor stores images as float32."""
        images = [np.ones((32, 32), dtype=np.uint8) * 100, np.ones((32, 32), dtype=np.uint8) * 200]
        registration = DiffeomorphicDemonsRegistration(images=images)
        assert len(registration._images) == 2
        assert registration._images[0].dtype == np.float32
        assert registration._images[1].dtype == np.float32

    def test_constructor_preserves_float32(self) -> None:
        """Verifies that a float32 image is stored without conversion."""
        image = np.ones((32, 32), dtype=np.float32)
        registration = DiffeomorphicDemonsRegistration(images=[image, image])
        assert registration._images[0] is image

    def test_constructor_rejects_an_undersized_group(self) -> None:
        """Verifies that a group holding fewer than two images is rejected and reports the count supplied."""
        image = np.random.default_rng(seed=7).standard_normal((32, 32)).astype(np.float32)

        expected_message = (
            "Unable to initialize the diffeomorphic demons registration. Groupwise registration aligns the images "
            "of a group to their common mean space, so it requires at least 2 images, but got 1."
        )
        with pytest.raises(ValueError, match=error_format(message=expected_message)):
            DiffeomorphicDemonsRegistration(images=[image])

    def test_compute_grid_sampling(self) -> None:
        """Verifies the grid sampling calculation formula."""
        images = [np.ones((32, 32), dtype=np.float32)] * 2
        registration = DiffeomorphicDemonsRegistration(
            images=images, final_scale=1.0, final_grid_sampling=16.0, grid_sampling_factor=1.0
        )
        # At final_scale=1.0: grid_sampling = (1.0 - 1.0) * 1.0 * 16.0 + 16.0 = 16.0
        assert registration._compute_grid_sampling(scale=1.0) == 16.0
        # At scale=3.0: grid_sampling = (3.0 - 1.0) * 1.0 * 16.0 + 16.0 = 48.0
        assert registration._compute_grid_sampling(scale=3.0) == 48.0

    def test_cache_operations(self) -> None:
        """Verifies cache set and get operations."""
        images = [np.ones((32, 32), dtype=np.float32)] * 2
        registration = DiffeomorphicDemonsRegistration(images=images)
        key = (0, 1, 1.0)
        data = np.zeros((5, 5), dtype=np.float32)
        registration._set_cached(key="test", iteration_key=key, data=data)
        result = registration._get_cached(key="test", iteration_key=key)
        assert result is data

    def test_cache_miss_returns_none(self) -> None:
        """Verifies that a cache miss returns None."""
        images = [np.ones((32, 32), dtype=np.float32)] * 2
        registration = DiffeomorphicDemonsRegistration(images=images)
        result = registration._get_cached(key="missing", iteration_key=(0, 0, 0.0))
        assert result is None

    def test_cache_stale_key_returns_none(self) -> None:
        """Verifies that a stale iteration key causes a cache miss."""
        images = [np.ones((32, 32), dtype=np.float32)] * 2
        registration = DiffeomorphicDemonsRegistration(images=images)
        data = np.zeros((5, 5), dtype=np.float32)
        registration._set_cached(key="test", iteration_key=(0, 0, 1.0), data=data)
        result = registration._get_cached(key="test", iteration_key=(0, 1, 1.0))
        assert result is None

    def test_register_identical_images(self) -> None:
        """Verifies that registering identical images produces near-identity deformations."""
        image = np.random.default_rng(seed=42).standard_normal((32, 32)).astype(np.float32)
        images = [image.copy(), image.copy()]
        registration = DiffeomorphicDemonsRegistration(
            images=images, scale_sampling=5, final_scale=1.0, final_grid_sampling=8.0
        )
        registration.register(progress=False)
        for image_index in range(2):
            deformation = registration.get_deformation(image_index=image_index)
            assert np.max(np.abs(deformation.get_field(dimension=0))) < 2.0
            assert np.max(np.abs(deformation.get_field(dimension=1))) < 2.0

    def test_coarse_level_contributes_nothing(self) -> None:
        """Verifies that a level whose knot grid cannot freeze its edges resolves to None for every image."""
        generator = np.random.default_rng(seed=13)
        images = [generator.standard_normal((64, 64)).astype(np.float32) for _ in range(2)]
        registration = _prepare_level_registration(images=images, final_grid_sampling=8.0)

        # At scale 8.0 the pyramid works at 8x8 pixels and the sampling resolves to 8.0 pixels, which produces a 4x4
        # knot grid. Skipping the level is what keeps an unregularized deformation out of the running total.
        deformations = registration._compute_groupwise_deformations(iteration_key=(3, 1, 8.0))

        assert deformations == [None, None]

    def test_coarse_level_filter_measures_the_working_resolution(self) -> None:
        """Verifies that the level filter rejects a grid the full-resolution image shape would have accepted."""
        generator = np.random.default_rng(seed=17)
        images = [generator.standard_normal((64, 64)).astype(np.float32) for _ in range(2)]
        registration = _prepare_level_registration(images=images, final_grid_sampling=3.75)

        # At scale 8.0 the sampling resolves to 30.0 pixels, which spans the full-resolution image with enough knots
        # to freeze its edges. The level is skipped only because the pyramid works at 8x8 pixels there.
        full_resolution_grid = SplineGrid.compute_grid_shape(field_height=64, field_width=64, grid_sampling=30.0)
        assert min(full_resolution_grid) >= MINIMUM_KNOTS_FOR_FROZEN_EDGES

        deformations = registration._compute_groupwise_deformations(iteration_key=(3, 1, 8.0))

        assert deformations == [None, None]

    def test_register_produces_deformations(self) -> None:
        """Verifies that registration produces finite, full-resolution, non-trivial deformations for distinct images."""
        generator = np.random.default_rng(seed=42)
        first_image = generator.standard_normal((32, 32)).astype(np.float32)
        second_image = generator.standard_normal((32, 32)).astype(np.float32)
        registration = DiffeomorphicDemonsRegistration(
            images=[first_image, second_image], scale_sampling=5, final_scale=1.0, final_grid_sampling=8.0
        )
        registration.register(progress=False)
        assert 0 in registration._deformations
        assert 1 in registration._deformations
        first_deformation = registration.get_deformation(image_index=0)
        second_deformation = registration.get_deformation(image_index=1)
        assert first_deformation.get_field(dimension=0).shape == (32, 32)
        assert first_deformation.get_field(dimension=1).shape == (32, 32)
        assert second_deformation.get_field(dimension=0).shape == (32, 32)
        assert second_deformation.get_field(dimension=1).shape == (32, 32)
        assert np.all(np.isfinite(first_deformation.get_field(dimension=0)))
        assert np.all(np.isfinite(first_deformation.get_field(dimension=1)))
        assert np.all(np.isfinite(second_deformation.get_field(dimension=0)))
        assert np.all(np.isfinite(second_deformation.get_field(dimension=1)))
        assert np.max(np.abs(first_deformation.get_field(dimension=0))) > 1e-3

    def test_register_without_smooth_scale(self) -> None:
        """Verifies that registration runs with smooth scale transitions disabled."""
        generator = np.random.default_rng(seed=7)
        first_image = generator.standard_normal((32, 32)).astype(np.float32)
        second_image = generator.standard_normal((32, 32)).astype(np.float32)
        registration = DiffeomorphicDemonsRegistration(
            images=[first_image, second_image],
            scale_sampling=5,
            final_scale=1.0,
            final_grid_sampling=8.0,
            smooth_scale=False,
        )
        registration.register(progress=False)
        assert 0 in registration._deformations
        assert 1 in registration._deformations

    def test_register_without_freeze_edges(self) -> None:
        """Verifies that registration runs with edge freezing disabled."""
        generator = np.random.default_rng(seed=11)
        first_image = generator.standard_normal((32, 32)).astype(np.float32)
        second_image = generator.standard_normal((32, 32)).astype(np.float32)
        registration = DiffeomorphicDemonsRegistration(
            images=[first_image, second_image],
            scale_sampling=5,
            final_scale=1.0,
            final_grid_sampling=8.0,
            freeze_edges=False,
        )
        registration.register(progress=False)
        assert 0 in registration._deformations
        assert 1 in registration._deformations

    def test_register_reports_a_grid_no_level_can_freeze(self) -> None:
        """Verifies that a run whose finest level cannot freeze its knot grid edges reports the grid it built."""
        generator = np.random.default_rng(seed=23)
        images = [generator.standard_normal((32, 32)).astype(np.float32) for _ in range(2)]
        registration = DiffeomorphicDemonsRegistration(images=images, scale_sampling=5, final_scale=1.0)

        # The default 16.0 pixel sampling spans the 32 pixel images with 5 knots, one short of the 6 that freezing the
        # edges needs. Every coarser level builds a smaller grid, so the run would resolve no deformation at all.
        expected_message = (
            f"Unable to register the (32, 32) images to their common mean space. Freezing the knot grid edges "
            f"requires at least {MINIMUM_KNOTS_FOR_FROZEN_EDGES} knots along each dimension, but the finest scale "
            f"level samples its (32, 32) working resolution every 16.0 pixels, which builds a (5, 5) grid. Register "
            f"larger images or lower the 'diffeomorphic_registration.final_grid_sampling' configuration parameter."
        )
        with pytest.raises(RuntimeError, match=error_format(message=expected_message)):
            registration.register(progress=False)

        assert registration._deformations == {}

    def test_get_deformation_reports_an_unresolved_image(self) -> None:
        """Verifies that requesting a deformation no registration run resolved is reported."""
        image = np.ones((64, 64), dtype=np.float32)
        registration = DiffeomorphicDemonsRegistration(images=[image, image])

        expected_message = (
            "Unable to retrieve the deformation for image 0. The requested index must identify an image register() "
            "resolved a deformation for, and the resolved indices are []."
        )
        with pytest.raises(RuntimeError, match=error_format(message=expected_message)):
            registration.get_deformation(image_index=0)

    def test_regularize_deformation_without_injectivity(self) -> None:
        """Verifies direct regularization with the injectivity constraint disabled returns a Deformation."""
        image = np.ones((32, 32), dtype=np.float32)
        registration = DiffeomorphicDemonsRegistration(images=[image, image], injective=False, final_grid_sampling=4.0)
        deformation = Deformation.identity(height=32, width=32)
        result = registration._regularize_deformation(scale=1.0, deformation=deformation)
        assert isinstance(result, Deformation)

    def test_repr_reports_the_group_and_its_tuning(self) -> None:
        """Verifies the representation carries the group size and the three parameters that drive the run."""
        images = [np.ones((32, 32), dtype=np.float32)] * 3
        registration = DiffeomorphicDemonsRegistration(
            images=images, speed_factor=2.5, scale_sampling=7, final_scale=1.5
        )
        assert repr(registration) == (
            "DiffeomorphicDemonsRegistration(image_count=3, speed_factor=2.5, scale_sampling=7, final_scale=1.5)"
        )

    def test_resolve_field_shape_before_register_is_reported(self) -> None:
        """Verifies that querying the working resolution before register() builds the pyramids is reported."""
        generator = np.random.default_rng(seed=29)
        images = [generator.standard_normal((64, 64)).astype(np.float32) for _ in range(2)]
        registration = DiffeomorphicDemonsRegistration(
            images=images, scale_sampling=5, final_grid_sampling=8.0, final_scale=1.0
        )

        expected_message = (
            "Unable to resolve the field shape. The pyramids have not been initialized. Call register() first."
        )
        with pytest.raises(RuntimeError, match=error_format(message=expected_message)):
            registration._resolve_field_shape(scale=1.0)

    def test_groupwise_deformation_averages_the_signed_pairwise_deformations(self) -> None:
        """Verifies that a three-image group averages each image's signed pairwise deformations over its pairs."""
        generator = np.random.default_rng(seed=101)
        images = [generator.standard_normal((64, 64)).astype(np.float32) for _ in range(3)]
        registration = _prepare_level_registration(images=images, final_grid_sampling=8.0)
        iteration_key = (0, 1, 1.0)

        groupwise = registration._compute_groupwise_deformations(iteration_key=iteration_key)

        # Re-requests the three unordered pairs the groupwise pass visited. The per-iteration cache holds the
        # deformed images and gradients under the same key, so these reproduce the exact fields it accumulated.
        first_second = registration._compute_pairwise_deformation(
            source_index=0, target_index=1, iteration_key=iteration_key
        )
        first_third = registration._compute_pairwise_deformation(
            source_index=0, target_index=2, iteration_key=iteration_key
        )
        second_third = registration._compute_pairwise_deformation(
            source_index=1, target_index=2, iteration_key=iteration_key
        )

        def _average(first: NDArray[np.float32], second: NDArray[np.float32]) -> NDArray[np.float32]:
            """Sums two contributions and divides by the two pairs each image of a three-image group joins."""
            accumulated = first.copy()
            np.add(accumulated, second, out=accumulated)
            np.multiply(accumulated, np.float32(1.0 / 2.0), out=accumulated)
            return accumulated

        # The Demons deformation is antisymmetric, so an image takes a pair's field with a positive sign when it is
        # that pair's first member and with a negative sign when it is the second. Each image joins two of the three
        # pairs, so the divisor is two rather than the group size.
        for dimension in (0, 1):
            np.testing.assert_array_equal(
                groupwise[0].get_field(dimension=dimension),
                _average(
                    first=first_second.get_field(dimension=dimension), second=first_third.get_field(dimension=dimension)
                ),
            )
            np.testing.assert_array_equal(
                groupwise[1].get_field(dimension=dimension),
                _average(
                    first=-first_second.get_field(dimension=dimension),
                    second=second_third.get_field(dimension=dimension),
                ),
            )
            np.testing.assert_array_equal(
                groupwise[2].get_field(dimension=dimension),
                _average(
                    first=-first_third.get_field(dimension=dimension),
                    second=-second_third.get_field(dimension=dimension),
                ),
            )

        # A trivially zero field would satisfy any divisor, so this pins the comparison to real deformation values.
        assert float(np.max(np.abs(groupwise[0].get_field(dimension=0)))) > 0.01

    def test_default_parameters(self) -> None:
        """Verifies that the constructor stores default parameter values."""
        images = [np.ones((32, 32), dtype=np.float32)] * 2
        registration = DiffeomorphicDemonsRegistration(images=images)
        assert registration._speed_factor == 3.0
        assert registration._scale_sampling == 30
        assert registration._grid_sampling_factor == 1.0
        assert registration._final_scale == 1.0
        assert registration._final_grid_sampling == 16.0
        assert registration._smooth_scale
        assert registration._injective
        assert registration._freeze_edges
        assert registration._deformation_limit == 1.0
        assert registration._noise_factor == 1.0


class TestDiffeomorphicRegistrationAccuracy:
    """Tests that DiffeomorphicDemonsRegistration resolves deformations which align its input images.

    Notes:
        The other tests in this module assert the structure of the resolved deformations, such as their shape,
        finiteness, and magnitude bounds. These tests assert alignment instead, so that a change which preserves the
        structure of the output while destroying its accuracy fails the suite.
    """

    def test_recovers_known_translation(self, gaussian_blob_image: Callable[..., NDArray[np.float64]]) -> None:
        """Verifies that registration recovers the relative displacement imposed between two images."""
        translation_y, translation_x = _ACCURACY_TRANSLATION
        reference = _build_blob_image(builder=gaussian_blob_image, translation=(0, 0))
        translated = _build_blob_image(builder=gaussian_blob_image, translation=_ACCURACY_TRANSLATION)
        registration = _register_accuracy_images(images=[reference, translated])

        reference_displacement = _mean_field_displacement(deformation=registration.get_deformation(image_index=0))
        translated_displacement = _mean_field_displacement(deformation=registration.get_deformation(image_index=1))

        # Groupwise registration centers every image on the group mean, so the absolute displacement resolved for one
        # image depends on the whole group. Only the displacement between the two images recovers the translation.
        assert translated_displacement[0] - reference_displacement[0] == pytest.approx(
            translation_y, abs=_ACCURACY_DISPLACEMENT_TOLERANCE
        )
        assert translated_displacement[1] - reference_displacement[1] == pytest.approx(
            translation_x, abs=_ACCURACY_DISPLACEMENT_TOLERANCE
        )

    def test_aligns_translated_pair(self, gaussian_blob_image: Callable[..., NDArray[np.float64]]) -> None:
        """Verifies that registration improves the correlation between a translated image pair."""
        reference = _build_blob_image(builder=gaussian_blob_image, translation=(0, 0))
        translated = _build_blob_image(builder=gaussian_blob_image, translation=_ACCURACY_TRANSLATION)
        registration = _register_accuracy_images(images=[reference, translated])

        interior = slice(_ACCURACY_FIELD_MARGIN, -_ACCURACY_FIELD_MARGIN)
        raw_correlation = _image_correlation(first=reference[interior, interior], second=translated[interior, interior])
        warped_correlation = _image_correlation(
            first=_warped_interior(registration=registration, image=reference, image_index=0),
            second=_warped_interior(registration=registration, image=translated, image_index=1),
        )

        assert warped_correlation > _ACCURACY_MINIMUM_CORRELATION
        assert warped_correlation > raw_correlation

    def test_aligns_recording_group(self, gaussian_blob_image: Callable[..., NDArray[np.float64]]) -> None:
        """Verifies that groupwise registration aligns every image pair within a group of translated images."""
        images = [
            _build_blob_image(builder=gaussian_blob_image, translation=translation)
            for translation in _ACCURACY_GROUP_TRANSLATIONS
        ]
        registration = _register_accuracy_images(images=images)

        interior = slice(_ACCURACY_FIELD_MARGIN, -_ACCURACY_FIELD_MARGIN)
        warped = [
            _warped_interior(registration=registration, image=image, image_index=index)
            for index, image in enumerate(images)
        ]
        pairs = [(first, second) for first in range(len(images)) for second in range(first + 1, len(images))]

        raw_correlation = min(
            _image_correlation(first=images[first][interior, interior], second=images[second][interior, interior])
            for first, second in pairs
        )
        warped_correlation = min(
            _image_correlation(first=warped[first], second=warped[second]) for first, second in pairs
        )

        assert warped_correlation > _ACCURACY_MINIMUM_CORRELATION
        assert warped_correlation > raw_correlation

    def test_produces_identical_fields_across_runs(
        self, gaussian_blob_image: Callable[..., NDArray[np.float64]]
    ) -> None:
        """Verifies that registering the same images twice produces bit-identical deformation fields."""
        reference = _build_blob_image(builder=gaussian_blob_image, translation=(0, 0))
        translated = _build_blob_image(builder=gaussian_blob_image, translation=_ACCURACY_TRANSLATION)

        first_run = _register_accuracy_images(images=[reference, translated])
        second_run = _register_accuracy_images(images=[reference, translated])

        # Bit-identical output makes any numerical change to the registration internals detectable, which allows a
        # refactor to be verified as behavior-preserving.
        for image_index in range(2):
            first_deformation = first_run.get_deformation(image_index=image_index)
            second_deformation = second_run.get_deformation(image_index=image_index)
            np.testing.assert_array_equal(
                first_deformation.get_field(dimension=0), second_deformation.get_field(dimension=0)
            )
            np.testing.assert_array_equal(
                first_deformation.get_field(dimension=1), second_deformation.get_field(dimension=1)
            )


@pytest.mark.xdist_group(name="console_progress_state")
class TestRegisterProgressState:
    """Tests that register() honors its progress argument without leaking the caller's console progress state.

    Notes:
        The batch engine reuses worker processes across jobs, so a progress flag left flipped by one registration
        run would follow every later job in that process.
    """

    @pytest.mark.parametrize("previous_state", [True, False])
    @pytest.mark.parametrize("progress", [True, False])
    def test_applies_and_restores_the_progress_state(
        self, previous_state: bool, progress: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verifies that every caller state and progress argument pair is applied during the run and undone after."""
        restore_state = console.progress_enabled
        try:
            if previous_state:
                console.enable_progress()
            else:
                console.disable_progress()

            registration = TestRegisterProgressState._build_registration()
            observed_states: list[bool] = []
            original_iteration = registration._perform_iteration

            def _record(**keyword_arguments: object) -> None:
                """Records the progress state the console holds while the scale loop is running."""
                observed_states.append(console.progress_enabled)
                original_iteration(**keyword_arguments)  # type: ignore[arg-type]  # The kwargs are typed as object.

            monkeypatch.setattr(registration, "_perform_iteration", _record)

            registration.register(progress=progress)

            # The argument governs the run's own working state, so the loop must see exactly the state it requested
            # rather than whatever the caller happened to hold.
            assert observed_states
            assert set(observed_states) == {progress}

            # Leaving the run restores the caller's own state, whichever way the argument moved it.
            assert console.progress_enabled is previous_state
        finally:
            if restore_state:
                console.enable_progress()
            else:
                console.disable_progress()

    def test_restores_the_callers_progress_state_on_the_exception_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verifies that a run failing inside the scale loop still restores the caller's progress state."""
        restore_state = console.progress_enabled
        try:
            console.enable_progress()
            registration = TestRegisterProgressState._build_registration()

            def _fail(**_keyword_arguments: object) -> None:
                """Stands in for the per-iteration step and fails on the first call the scale loop makes."""
                raise ZeroDivisionError("iteration failed")

            monkeypatch.setattr(registration, "_perform_iteration", _fail)

            with pytest.raises(ZeroDivisionError):
                registration.register(progress=False)

            # register() disabled progress on the way in, so only the finally block can put it back.
            assert console.progress_enabled
        finally:
            if restore_state:
                console.enable_progress()
            else:
                console.disable_progress()

    @staticmethod
    def _build_registration() -> DiffeomorphicDemonsRegistration:
        """Builds the smallest registration instance whose finest level can still freeze its knot grid edges."""
        generator = np.random.default_rng(seed=71)
        images = [generator.standard_normal((32, 32)).astype(np.float32) for _ in range(2)]
        return DiffeomorphicDemonsRegistration(
            images=images, scale_sampling=2, final_scale=1.0, final_grid_sampling=8.0
        )


def _prepare_level_registration(
    images: list[NDArray[np.float32]], final_grid_sampling: float
) -> DiffeomorphicDemonsRegistration:
    """Builds a registration instance carrying the scale-space pyramids register() would have created.

    Notes:
        The level filter reads the working resolution of a scale from the pyramids, so a test that queries a single
        level directly has to stand them up itself.
    """
    registration = DiffeomorphicDemonsRegistration(
        images=images, final_scale=1.0, final_grid_sampling=final_grid_sampling
    )
    registration._pyramids = [ScaleSpacePyramid(data=image, minimum_scale=1.0) for image in registration._images]
    return registration


def _build_blob_image(builder: Callable[..., NDArray[np.float64]], translation: tuple[int, int]) -> NDArray[np.float32]:
    """Builds the synthetic accuracy-test image with its blob centers offset by the requested translation."""
    translation_y, translation_x = translation
    centers = tuple((row + translation_y, column + translation_x) for row, column in _ACCURACY_BLOB_CENTERS)
    image = builder(
        height=_ACCURACY_IMAGE_EXTENT,
        width=_ACCURACY_IMAGE_EXTENT,
        centers=centers,
        sigma=_ACCURACY_BLOB_SIGMA,
    )
    return image.astype(np.float32)


def _register_accuracy_images(images: list[NDArray[np.float32]]) -> DiffeomorphicDemonsRegistration:
    """Runs groupwise diffeomorphic registration over the given images using the accuracy-test parameters.

    Notes:
        The images are copied before registration, so that repeated runs over the same inputs stay independent of
        each other.
    """
    registration = DiffeomorphicDemonsRegistration(
        images=[image.copy() for image in images],
        scale_sampling=_ACCURACY_SCALE_SAMPLING,
        final_scale=1.0,
        final_grid_sampling=_ACCURACY_GRID_SAMPLING,
    )
    registration.register(progress=False)
    return registration


def _mean_field_displacement(deformation: Deformation) -> tuple[float, float]:
    """Returns the mean vertical and horizontal displacement over the interior of a deformation field."""
    interior = slice(_ACCURACY_FIELD_MARGIN, -_ACCURACY_FIELD_MARGIN)
    return (
        float(deformation.get_field(dimension=0)[interior, interior].mean()),
        float(deformation.get_field(dimension=1)[interior, interior].mean()),
    )


def _warped_interior(
    registration: DiffeomorphicDemonsRegistration, image: NDArray[np.float32], image_index: int
) -> NDArray[np.float32]:
    """Returns the interior of an image after applying the deformation resolved for it."""
    deformation = registration.get_deformation(image_index=image_index)
    warped = deformation.apply_deformation(data=image)
    interior = slice(_ACCURACY_FIELD_MARGIN, -_ACCURACY_FIELD_MARGIN)
    return warped[interior, interior]


def _image_correlation(first: NDArray[np.float32], second: NDArray[np.float32]) -> float:
    """Returns the Pearson correlation coefficient between two images of matching shape."""
    first_centered = first.ravel() - first.mean()
    second_centered = second.ravel() - second.mean()
    return float(
        np.dot(first_centered, second_centered) / (np.linalg.norm(first_centered) * np.linalg.norm(second_centered))
    )
