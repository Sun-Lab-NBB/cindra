"""Contains tests for the diffeomorphic module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from cindra.registration.pyramid import ScaleSpacePyramid
from cindra.registration.deformation import Deformation
from cindra.registration.spline_grid import SplineGrid, MINIMUM_KNOTS_FOR_FROZEN_EDGES
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

        # Flattens one corner of both gradients so the zero-denominator branch is exercised.
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
    """Tests DiffeomorphicDemonsRegistration."""

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
        image = np.random.default_rng(42).standard_normal((32, 32)).astype(np.float32)
        images = [image.copy(), image.copy()]
        registration = DiffeomorphicDemonsRegistration(
            images=images, scale_sampling=5, final_scale=1.0, final_grid_sampling=8.0
        )
        registration.register(progress=False)
        for image_index in range(2):
            deformation = registration.get_deformation(image_index=image_index)
            # Deformations should be near-zero for identical images.
            assert np.max(np.abs(deformation[0])) < 2.0
            assert np.max(np.abs(deformation[1])) < 2.0

    def test_single_image_group_accumulates_an_identity_deformation(self) -> None:
        """Verifies that an image pairing with nothing accumulates no contribution and resolves to an identity."""
        image = np.random.default_rng(seed=7).standard_normal((32, 32)).astype(np.float32)
        registration = _prepare_level_registration(images=[image], final_grid_sampling=8.0)

        deformations = registration._compute_groupwise_deformations(iteration_key=(0, 1, 1.0))

        assert len(deformations) == 1
        assert deformations[0] is not None
        assert deformations[0].is_identity
        assert deformations[0].field_shape == (32, 32)

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
        # Final deformation fields span the full image resolution and contain only finite values.
        assert first_deformation[0].shape == (32, 32)
        assert first_deformation[1].shape == (32, 32)
        assert second_deformation[0].shape == (32, 32)
        assert second_deformation[1].shape == (32, 32)
        assert np.all(np.isfinite(first_deformation[0]))
        assert np.all(np.isfinite(first_deformation[1]))
        assert np.all(np.isfinite(second_deformation[0]))
        assert np.all(np.isfinite(second_deformation[1]))
        # Distinct input images must produce a non-trivial (non-zero) deformation.
        assert np.max(np.abs(first_deformation[0])) > 1e-3

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
        with pytest.raises(RuntimeError, match="Unable to register the"):
            registration.register(progress=False)

        assert registration._deformations == {}

    def test_get_deformation_reports_an_unresolved_image(self) -> None:
        """Verifies that requesting a deformation no registration run resolved is reported."""
        image = np.ones((64, 64), dtype=np.float32)
        registration = DiffeomorphicDemonsRegistration(images=[image, image])

        with pytest.raises(RuntimeError, match="Unable to retrieve the deformation"):
            registration.get_deformation(image_index=0)

    def test_regularize_deformation_without_injectivity(self) -> None:
        """Verifies direct regularization with the injectivity constraint disabled returns a Deformation."""
        image = np.ones((32, 32), dtype=np.float32)
        registration = DiffeomorphicDemonsRegistration(images=[image, image], injective=False, final_grid_sampling=4.0)
        deformation = Deformation.identity(height=32, width=32)
        result = registration._regularize_deformation(scale=1.0, deformation=deformation)
        assert isinstance(result, Deformation)

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
            np.testing.assert_array_equal(first_deformation[0], second_deformation[0])
            np.testing.assert_array_equal(first_deformation[1], second_deformation[1])


def _prepare_level_registration(
    images: list[NDArray[np.float32]], final_grid_sampling: float
) -> DiffeomorphicDemonsRegistration:
    """Builds a registration instance carrying the scale-space pyramids register() would have created.

    Notes:
        The level filter reads the working resolution of a scale from the pyramids, so a test that queries a single
        level directly has to stand them up itself.

    Args:
        images: The images to register against their common mean.
        final_grid_sampling: The B-spline grid spacing at the finest scale level.

    Returns:
        The registration instance, with its pyramids initialized and no deformation resolved yet.
    """
    registration = DiffeomorphicDemonsRegistration(
        images=images, final_scale=1.0, final_grid_sampling=final_grid_sampling
    )
    registration._pyramids = [ScaleSpacePyramid(data=image, minimum_scale=1.0) for image in registration._images]
    return registration


def _build_blob_image(builder: Callable[..., NDArray[np.float64]], translation: tuple[int, int]) -> NDArray[np.float32]:
    """Builds the synthetic accuracy-test image with its blob centers offset by the requested translation.

    Args:
        builder: The gaussian_blob_image fixture factory used to render the image.
        translation: The vertical and horizontal offset in pixels applied to every blob center.

    Returns:
        The rendered image as a float32 array.
    """
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

    Args:
        images: The images to register against their common mean.

    Returns:
        The registration instance, with its deformations already resolved.
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
    """Returns the mean vertical and horizontal displacement over the interior of a deformation field.

    Args:
        deformation: The deformation whose displacement fields are measured.

    Returns:
        A tuple storing the mean vertical displacement followed by the mean horizontal displacement, in pixels.
    """
    interior = slice(_ACCURACY_FIELD_MARGIN, -_ACCURACY_FIELD_MARGIN)
    return (
        float(deformation[0][interior, interior].mean()),
        float(deformation[1][interior, interior].mean()),
    )


def _warped_interior(
    registration: DiffeomorphicDemonsRegistration, image: NDArray[np.float32], image_index: int
) -> NDArray[np.float32]:
    """Returns the interior of an image after applying the deformation resolved for it.

    Args:
        registration: The registration instance holding the resolved deformations.
        image: The image to warp into the group's common space.
        image_index: The index identifying which resolved deformation applies to the image.

    Returns:
        The warped image with its border margin removed.
    """
    deformation = registration.get_deformation(image_index=image_index)
    warped = deformation.apply_deformation(data=image)
    interior = slice(_ACCURACY_FIELD_MARGIN, -_ACCURACY_FIELD_MARGIN)
    return warped[interior, interior]


def _image_correlation(first: NDArray[np.float32], second: NDArray[np.float32]) -> float:
    """Returns the Pearson correlation coefficient between two images of matching shape.

    Args:
        first: The first image to correlate.
        second: The second image to correlate.

    Returns:
        The correlation coefficient, ranging from -1.0 for anticorrelated images to 1.0 for identical images.
    """
    first_centered = first.ravel() - first.mean()
    second_centered = second.ravel() - second.mean()
    return float(
        np.dot(first_centered, second_centered) / (np.linalg.norm(first_centered) * np.linalg.norm(second_centered))
    )
