"""Contains tests for the deformation module."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.ndimage
from ataraxis_base_utilities import error_format

from cindra.registration.deformation import (
    Deformation,
    zoom,
    _warp,
    _resize,
    diffuse,
    _project,
    _make_samples_absolute,
    _create_diffusion_kernel,
)


class TestCreateDiffusionKernel:
    """Tests _create_diffusion_kernel."""

    def test_small_sigma_returns_delta(self) -> None:
        """Verifies that sigma below threshold returns a single-element delta kernel."""
        kernel = _create_diffusion_kernel(sigma=0.05)
        np.testing.assert_array_equal(kernel, [1.0])

    def test_kernel_sums_to_one(self) -> None:
        """Verifies that the kernel sums to approximately 1.0 for all sigma values."""
        for sigma in [0.05, 0.5, 2.0, 5.0]:
            kernel = _create_diffusion_kernel(sigma=sigma)
            np.testing.assert_allclose(kernel.sum(), 1.0, atol=1e-6)

    def test_kernel_is_symmetric(self) -> None:
        """Verifies that the kernel is symmetric about its center."""
        kernel = _create_diffusion_kernel(sigma=3.0)
        np.testing.assert_allclose(kernel, kernel[::-1], atol=1e-6)

    def test_kernel_dtype(self) -> None:
        """Verifies the kernel dtype is float32."""
        kernel = _create_diffusion_kernel(sigma=2.0)
        assert kernel.dtype == np.float32

    def test_large_sigma_produces_wide_kernel(self) -> None:
        """Verifies that sigma above threshold produces a multi-element kernel."""
        kernel = _create_diffusion_kernel(sigma=2.0)
        assert kernel.size > 1

    def test_kernel_peak_is_at_center(self) -> None:
        """Verifies that the kernel's maximum value is at the center element."""
        kernel = _create_diffusion_kernel(sigma=3.0)
        assert np.argmax(kernel) == kernel.size // 2

    def test_kernel_values_decrease_from_center(self) -> None:
        """Verifies that kernel values monotonically decrease from center to tails."""
        kernel = _create_diffusion_kernel(sigma=2.0)
        center = kernel.size // 2
        right_half = kernel[center:]
        assert np.all(np.diff(right_half) <= 0)


class TestDiffuse:
    """Tests the diffuse function."""

    def test_small_sigma_identity(self) -> None:
        """Verifies that a very small sigma produces no smoothing."""
        generator = np.random.default_rng(seed=42)
        data = generator.standard_normal((16, 16)).astype(np.float32)
        result = diffuse(data=data, sigma=0.01)
        np.testing.assert_array_equal(result, data)

    def test_output_shape_and_dtype(self) -> None:
        """Verifies that diffuse returns data with correct shape and dtype."""
        generator = np.random.default_rng(seed=42)
        data = generator.standard_normal((32, 32)).astype(np.float32)
        result = diffuse(data=data, sigma=3.0)
        assert result.shape == data.shape
        assert result.dtype == np.float32

    def test_per_dimension_sigma(self) -> None:
        """Verifies that per-dimension sigma list is accepted."""
        generator = np.random.default_rng(seed=42)
        data = generator.standard_normal((32, 32)).astype(np.float32)
        result = diffuse(data=data, sigma=[2.0, 0.01])
        assert result.dtype == np.float32
        assert result.shape == (32, 32)


class TestZoom:
    """Tests the zoom function."""

    def test_upscale_shape(self) -> None:
        """Verifies the output shape after upscaling."""
        data = np.ones((10, 10), dtype=np.float32)
        result = zoom(data=data, factor=2.0)
        assert result.shape == (20, 20)

    def test_downscale_shape(self) -> None:
        """Verifies the output shape after downscaling."""
        data = np.ones((20, 20), dtype=np.float32)
        result = zoom(data=data, factor=0.5)
        assert result.shape == (10, 10)

    def test_per_dimension_factors(self) -> None:
        """Verifies that per-dimension factor tuple is accepted."""
        data = np.ones((10, 10), dtype=np.float32)
        result = zoom(data=data, factor=(2.0, 0.5))
        assert result.shape == (20, 5)

    def test_uniform_image_preserved(self) -> None:
        """Verifies that a uniform image remains uniform after zooming."""
        data = np.ones((10, 10), dtype=np.float32) * 42.0
        result = zoom(data=data, factor=2.0)
        np.testing.assert_allclose(result, 42.0, atol=1e-4)

    def test_dtype_preserved(self) -> None:
        """Verifies the output dtype matches the input."""
        data = np.ones((10, 10), dtype=np.float32)
        result = zoom(data=data, factor=1.5)
        assert result.dtype == np.float32

    @pytest.mark.parametrize("order", [0, 1, 3])
    def test_interpolation_orders(self, order: int) -> None:
        """Verifies that all interpolation orders produce valid output."""
        data = np.arange(100, dtype=np.float32).reshape(10, 10)
        result = zoom(data=data, factor=2.0, order=order)
        assert result.shape == (20, 20)
        assert np.isfinite(result).all()


class TestMakeSamplesAbsolute:
    """Tests _make_samples_absolute."""

    def test_zero_deltas_give_identity_grid(self) -> None:
        """Verifies that zero displacement fields produce identity coordinate grids."""
        delta_x = np.zeros((4, 6), dtype=np.float32)
        delta_y = np.zeros((4, 6), dtype=np.float32)
        absolute_x, absolute_y = _make_samples_absolute(delta_x=delta_x, delta_y=delta_y)

        expected_x = np.arange(6, dtype=np.float32).reshape(1, 6)
        expected_y = np.arange(4, dtype=np.float32).reshape(4, 1)
        np.testing.assert_allclose(absolute_x, np.broadcast_to(expected_x, (4, 6)))
        np.testing.assert_allclose(absolute_y, np.broadcast_to(expected_y, (4, 6)))

    def test_with_known_deltas(self) -> None:
        """Verifies correct absolute coordinates with known displacement values."""
        delta_x = np.ones((3, 3), dtype=np.float32) * 0.5
        delta_y = np.ones((3, 3), dtype=np.float32) * -0.5
        absolute_x, absolute_y = _make_samples_absolute(delta_x=delta_x, delta_y=delta_y)
        # At pixel (1, 2): absolute_x = 2 + 0.5 = 2.5, absolute_y = 1 + (-0.5) = 0.5
        np.testing.assert_allclose(absolute_x[1, 2], 2.5)
        np.testing.assert_allclose(absolute_y[1, 2], 0.5)


class TestWarp:
    """Tests the _warp backward-sampling kernel against independent interpolation references."""

    def test_bilinear_matches_map_coordinates(self) -> None:
        """Verifies the bilinear branch reproduces scipy.ndimage.map_coordinates at the same sample locations."""
        generator = np.random.default_rng(seed=5)
        height, width = 24, 29
        data = generator.standard_normal((height, width)).astype(np.float32)

        # Keeps every sample at least one pixel away from the borders, so the kernel takes its interior branch and
        # the reference never has to extrapolate.
        samples_y = generator.uniform(1.0, height - 3.0, size=200).astype(np.float32)
        samples_x = generator.uniform(1.0, width - 3.0, size=200).astype(np.float32)

        result = np.empty(200, dtype=np.float32)
        _warp(data=data, result=result, samples_x=samples_x, samples_y=samples_y, order=1)

        reference = scipy.ndimage.map_coordinates(
            input=data.astype(np.float64),
            coordinates=np.stack([samples_y.astype(np.float64), samples_x.astype(np.float64)]),
            order=1,
            mode="nearest",
        )

        # The two agree to float32 rounding. Swapping the sample axes or dropping a fractional weight moves values by
        # order 1, which is seven orders of magnitude above this bound.
        np.testing.assert_allclose(result, reference, atol=1e-6)

    def test_nearest_matches_the_rounded_lookup(self) -> None:
        """Verifies the nearest-neighbor branch returns the pixel at the rounded sample coordinates exactly."""
        generator = np.random.default_rng(seed=6)
        height, width = 17, 23
        data = generator.standard_normal((height, width)).astype(np.float32)
        samples_y = generator.uniform(0.0, height - 1.0, size=150).astype(np.float32)
        samples_x = generator.uniform(0.0, width - 1.0, size=150).astype(np.float32)

        result = np.empty(150, dtype=np.float32)
        _warp(data=data, result=result, samples_x=samples_x, samples_y=samples_y, order=0)

        # Pixels are centered at integer coordinates, so the nearest pixel index is floor(sample + 0.5).
        reference = data[np.floor(samples_y + 0.5).astype(np.int64), np.floor(samples_x + 0.5).astype(np.int64)]
        np.testing.assert_array_equal(result, reference)


class TestProject:
    """Tests the _project forward splatting kernel, which Deformation.inverse is built on."""

    def test_identity_projection_is_exact(self) -> None:
        """Verifies that splatting to the identity grid reproduces the source array element for element."""
        height, width = 6, 7
        data = np.arange(height * width, dtype=np.float32).reshape(height, width) + 1.0
        samples_x, samples_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))

        result = np.zeros((height, width), dtype=np.float32)
        _project(data=data, result=result, samples_x=samples_x, samples_y=samples_y)

        # Every source pixel lands on its own integer position, and the tent kernel of unit radius gives its
        # neighbors a weight of exactly zero, so the normalized accumulation returns the source unchanged.
        np.testing.assert_array_equal(result, data)

    def test_splat_carries_source_values_forward(self) -> None:
        """Verifies that a source pixel is written to its target location rather than read from it."""
        extent = 16
        data = np.arange(extent * extent, dtype=np.float32).reshape(extent, extent) + 1.0
        samples_x, samples_y = np.meshgrid(np.arange(extent, dtype=np.float32), np.arange(extent, dtype=np.float32))

        result = np.zeros((extent, extent), dtype=np.float32)
        _project(data=data, result=result, samples_x=samples_x + 2.0, samples_y=samples_y + 1.0)

        # Forward splatting writes source pixel (y, x) to (y + 1, x + 2), so the destination holds the source value
        # one row up and two columns left of it. A backward read would instead place data[y + 1, x + 2] there.
        expected = np.zeros((extent, extent), dtype=np.float32)
        expected[1:, 2:] = data[:-1, :-2]

        # The first four columns receive the widened splats of the left-edge source column, whose out-of-field
        # neighbors push its bounds to the field border. Everything to their right is written exactly.
        np.testing.assert_array_equal(result[:, 4:], expected[:, 4:])


class TestResize:
    """Tests the _resize function."""

    def test_shape_change(self) -> None:
        """Verifies the output has the requested dimensions."""
        data = np.arange(100, dtype=np.float32).reshape(10, 10)
        result = _resize(data=data, new_height=20, new_width=30)
        assert result.shape == (20, 30)

    def test_same_size_preserves_values(self) -> None:
        """Verifies that resizing to the same dimensions preserves values."""
        generator = np.random.default_rng(seed=42)
        data = generator.standard_normal((10, 10)).astype(np.float32)
        result = _resize(data=data, new_height=10, new_width=10)
        np.testing.assert_allclose(result, data, atol=1e-4)

    def test_uniform_image_preserved(self) -> None:
        """Verifies that a uniform image remains uniform after resizing."""
        data = np.ones((10, 10), dtype=np.float32) * 7.0
        result = _resize(data=data, new_height=25, new_width=25)
        np.testing.assert_allclose(result, 7.0, atol=1e-4)


class TestDeformationIdentity:
    """Tests Deformation identity creation and properties."""

    def test_identity_creation(self) -> None:
        """Verifies identity deformation is created with correct shape."""
        deformation = Deformation.identity(height=10, width=20)
        assert deformation.is_identity
        assert deformation.field_shape == (10, 20)

    def test_identity_repr(self) -> None:
        """Verifies the string representation for identity deformations."""
        deformation = Deformation.identity(height=10, width=20)
        assert "identity" in repr(deformation)

    def test_identity_copy(self) -> None:
        """Verifies that copying an identity deformation returns a new identity."""
        original = Deformation.identity(height=10, width=20)
        copied = original.copy()
        assert copied.is_identity
        assert copied.field_shape == (10, 20)

    def test_identity_apply_returns_data_unchanged(self) -> None:
        """Verifies that applying an identity deformation returns data unchanged."""
        data = np.arange(100, dtype=np.float32).reshape(10, 10)
        deformation = Deformation.identity(height=10, width=10)
        result = deformation.apply_deformation(data=data)
        np.testing.assert_array_equal(result, data)

    def test_identity_inverse(self) -> None:
        """Verifies that the inverse of identity is identity."""
        deformation = Deformation.identity(height=10, width=10)
        inverse = deformation.inverse()
        assert inverse.is_identity


class TestDeformationConstructor:
    """Tests Deformation constructed from displacement fields."""

    def test_non_identity_properties(self) -> None:
        """Verifies properties of a non-identity deformation."""
        field_y = np.zeros((10, 20), dtype=np.float32)
        field_x = np.zeros((10, 20), dtype=np.float32)
        deformation = Deformation(field_y=field_y, field_x=field_x)
        assert not deformation.is_identity
        assert deformation.field_shape == (10, 20)

    def test_repr_includes_shape(self) -> None:
        """Verifies the string representation includes the field shape."""
        field_y = np.zeros((10, 20), dtype=np.float32)
        field_x = np.zeros((10, 20), dtype=np.float32)
        deformation = Deformation(field_y=field_y, field_x=field_x)
        assert "field_shape=(10, 20)" in repr(deformation)

    def test_get_field(self) -> None:
        """Verifies get_field returns the correct field array."""
        field_y = np.ones((5, 5), dtype=np.float32) * 3.0
        field_x = np.ones((5, 5), dtype=np.float32) * 4.0
        deformation = Deformation(field_y=field_y, field_x=field_x)
        np.testing.assert_array_equal(deformation.get_field(dimension=0), field_y)
        np.testing.assert_array_equal(deformation.get_field(dimension=1), field_x)


class TestDeformationScale:
    """Tests Deformation.scale."""

    def test_scale_by_factor(self) -> None:
        """Verifies that scaling multiplies all displacement values."""
        field_y = np.ones((5, 5), dtype=np.float32) * 2.0
        field_x = np.ones((5, 5), dtype=np.float32) * 3.0
        deformation = Deformation(field_y=field_y, field_x=field_x)
        scaled = deformation.scale(factor=0.5)
        np.testing.assert_allclose(scaled.get_field(dimension=0), 1.0)
        np.testing.assert_allclose(scaled.get_field(dimension=1), 1.5)

    def test_scale_by_one_copies(self) -> None:
        """Verifies that scale(1.0) creates a copy with identical values."""
        field_y = np.ones((5, 5), dtype=np.float32) * 2.0
        field_x = np.ones((5, 5), dtype=np.float32) * 3.0
        deformation = Deformation(field_y=field_y, field_x=field_x)
        copied = deformation.scale(factor=1.0)
        np.testing.assert_array_equal(copied.get_field(dimension=0), field_y)
        # Confirms the result is a copy rather than a view.
        copied.get_field(dimension=0)[0, 0] = 999.0
        assert deformation.get_field(dimension=0)[0, 0] != 999.0


class TestDeformationCompose:
    """Tests Deformation.compose."""

    def test_compose_with_identity_left(self) -> None:
        """Verifies that identity.compose(deformation) returns a copy of the deformation."""
        identity = Deformation.identity(height=10, width=10)
        field_y = np.ones((10, 10), dtype=np.float32) * 0.5
        field_x = np.ones((10, 10), dtype=np.float32) * -0.5
        deformation = Deformation(field_y=field_y, field_x=field_x)
        result = identity.compose(other=deformation)
        np.testing.assert_allclose(result.get_field(dimension=0), 0.5, atol=1e-5)
        np.testing.assert_allclose(result.get_field(dimension=1), -0.5, atol=1e-5)

    def test_compose_with_identity_right(self) -> None:
        """Verifies that deformation.compose(identity) returns a copy of the deformation."""
        identity = Deformation.identity(height=10, width=10)
        field_y = np.ones((10, 10), dtype=np.float32) * 0.5
        field_x = np.ones((10, 10), dtype=np.float32) * -0.5
        deformation = Deformation(field_y=field_y, field_x=field_x)
        result = deformation.compose(other=identity)
        np.testing.assert_allclose(result.get_field(dimension=0), 0.5, atol=1e-5)

    def test_compose_two_uniform_deformations(self) -> None:
        """Verifies composition of two small uniform displacements."""
        field_y = np.ones((20, 20), dtype=np.float32) * 0.3
        field_x = np.ones((20, 20), dtype=np.float32) * 0.2
        first_deformation = Deformation(field_y=field_y, field_x=field_x)
        second_deformation = Deformation(field_y=field_y.copy(), field_x=field_x.copy())
        result = first_deformation.compose(other=second_deformation)
        # For small uniform displacements, composition ≈ addition at interior pixels.
        np.testing.assert_allclose(result.get_field(dimension=0)[5:-5, 5:-5], 0.6, atol=0.05)
        np.testing.assert_allclose(result.get_field(dimension=1)[5:-5, 5:-5], 0.4, atol=0.05)


class TestDeformationResizeField:
    """Tests Deformation.resize_field."""

    def test_resize_identity(self) -> None:
        """Verifies that resizing an identity deformation returns a new identity."""
        deformation = Deformation.identity(height=10, width=10)
        resized = deformation.resize_field(new_height=20, new_width=20)
        assert resized.is_identity
        assert resized.field_shape == (20, 20)

    def test_resize_same_size_returns_self(self) -> None:
        """Verifies that resizing to the same dimensions returns self."""
        field_y = np.zeros((10, 10), dtype=np.float32)
        field_x = np.zeros((10, 10), dtype=np.float32)
        deformation = Deformation(field_y=field_y, field_x=field_x)
        resized = deformation.resize_field(new_height=10, new_width=10)
        assert resized is deformation

    def test_resize_changes_shape(self) -> None:
        """Verifies that resizing produces a deformation with new dimensions."""
        field_y = np.ones((10, 10), dtype=np.float32) * 0.5
        field_x = np.ones((10, 10), dtype=np.float32) * 0.5
        deformation = Deformation(field_y=field_y, field_x=field_x)
        resized = deformation.resize_field(new_height=20, new_width=20)
        assert resized.field_shape == (20, 20)


class TestDeformationApply:
    """Tests Deformation.apply_deformation."""

    def test_zero_displacement_preserves_image(self) -> None:
        """Verifies that zero displacement fields preserve the image."""
        generator = np.random.default_rng(seed=42)
        data = generator.standard_normal((20, 20)).astype(np.float32)
        field_y = np.zeros((20, 20), dtype=np.float32)
        field_x = np.zeros((20, 20), dtype=np.float32)
        deformation = Deformation(field_y=field_y, field_x=field_x)
        result = deformation.apply_deformation(data=data)
        np.testing.assert_allclose(result, data, atol=1e-5)

    def test_applies_to_different_sized_data(self) -> None:
        """Verifies that a resized field samples the data at the displacement it carries."""
        # A varying image makes the sampled location visible in the result, which a constant image hides.
        data = np.arange(400, dtype=np.float32).reshape(20, 20) + 1.0
        field_y = np.full((10, 10), fill_value=2.0, dtype=np.float32)
        field_x = np.full((10, 10), fill_value=-1.0, dtype=np.float32)
        deformation = Deformation(field_y=field_y, field_x=field_x)

        result = deformation.apply_deformation(data=data)

        assert result.shape == (20, 20)
        # Backward mapping samples the source at (y + 2, x - 1), so the result holds that exact slice of the data.
        # Values are separated by 1 within a row and by 20 between rows, so this bound admits no shifted slice.
        np.testing.assert_allclose(result[0:18, 1:20], data[2:20, 0:19], atol=1e-3)


class TestDeformationInverse:
    """Tests Deformation.inverse."""

    def test_inverse_of_small_displacement(self) -> None:
        """Verifies that the inverse of a uniform displacement is its exact negation at interior pixels."""
        field_y = np.ones((20, 20), dtype=np.float32) * 0.3
        field_x = np.ones((20, 20), dtype=np.float32) * -0.2
        deformation = Deformation(field_y=field_y, field_x=field_x)
        inverse = deformation.inverse()
        # A uniform field translates every pixel identically, so its inverse is the exact negation. The observed
        # error is 3e-08, which this bound holds to within a factor of thirty.
        np.testing.assert_allclose(inverse.get_field(dimension=0)[5:-5, 5:-5], -0.3, atol=1e-6)
        np.testing.assert_allclose(inverse.get_field(dimension=1)[5:-5, 5:-5], 0.2, atol=1e-6)

    def test_inverse_of_spatially_varying_field(self) -> None:
        """Verifies that the inverse of a linearly varying displacement matches its analytic inverse."""
        extent = 64
        slope = 0.05
        center = 32.0
        _, columns = np.meshgrid(
            np.arange(extent, dtype=np.float32), np.arange(extent, dtype=np.float32), indexing="ij"
        )
        field_x = (slope * (columns - center)).astype(np.float32)
        field_y = np.zeros((extent, extent), dtype=np.float32)

        inverse = Deformation(field_y=field_y, field_x=field_x).inverse()

        # The backward map sends p to q = p + slope * (p - center), so p - center = (q - center) / (1 + slope) and
        # the inverse displacement at q is -slope * (q - center) / (1 + slope).
        analytic_x = -slope * (columns - center) / (1.0 + slope)
        interior = (slice(8, -8), slice(8, -8))
        np.testing.assert_allclose(inverse.get_field(dimension=1)[interior], analytic_x[interior], atol=1e-5)
        np.testing.assert_allclose(inverse.get_field(dimension=0)[interior], 0.0, atol=1e-5)

        # Confirms the bound above discriminates: simply negating the field, which a uniform displacement would make
        # correct, misses the analytic inverse by more than three orders of magnitude beyond that bound.
        assert float(np.max(np.abs((-field_x)[interior] - analytic_x[interior]))) > 0.05


class TestDeformationGetDeformationLocations:
    """Tests Deformation.get_deformation_locations."""

    def test_returns_absolute_coordinates(self) -> None:
        """Verifies that the returned coordinates are absolute pixel positions."""
        field_y = np.ones((5, 5), dtype=np.float32) * 0.5
        field_x = np.ones((5, 5), dtype=np.float32) * -0.5
        deformation = Deformation(field_y=field_y, field_x=field_x)
        absolute_x, absolute_y = deformation.get_deformation_locations()
        # At pixel (2, 3): absolute_x = 3 + (-0.5) = 2.5, absolute_y = 2 + 0.5 = 2.5
        np.testing.assert_allclose(absolute_x[2, 3], 2.5)
        np.testing.assert_allclose(absolute_y[2, 3], 2.5)


class TestDeformationRegularize:
    """Tests Deformation.regularize."""

    def test_regularize_identity(self) -> None:
        """Verifies that regularizing an identity deformation returns identity."""
        deformation = Deformation.identity(height=20, width=20)
        result = deformation.regularize(grid_sampling=5.0)
        assert result.is_identity

    def test_regularize_smooth_field(self) -> None:
        """Verifies that regularization produces a valid deformation."""
        field_y = np.ones((30, 30), dtype=np.float32) * 0.5
        field_x = np.ones((30, 30), dtype=np.float32) * 0.5
        deformation = Deformation(field_y=field_y, field_x=field_x)
        result = deformation.regularize(grid_sampling=5.0)
        assert result.field_shape == (30, 30)

    def test_regularize_grid_too_coarse_to_freeze_edges(self) -> None:
        """Verifies that a sampling producing fewer than six knots per dimension raises a RuntimeError."""
        field_y = np.ones((16, 16), dtype=np.float32) * 0.5
        field_x = np.ones((16, 16), dtype=np.float32) * 0.5
        deformation = Deformation(field_y=field_y, field_x=field_x)
        expected_message = (
            "Unable to regularize the deformation. Freezing the edges of the knot grid requires at least 6 knots "
            "along each dimension, but sampling the (16, 16) field every 32.0 pixels produces a (4, 4) grid."
        )
        with pytest.raises(RuntimeError, match=error_format(expected_message)):
            deformation.regularize(grid_sampling=32.0)


class TestDeformationCrop:
    """Tests Deformation.crop."""

    def test_crop_identity(self) -> None:
        """Verifies that cropping an identity deformation returns a smaller identity."""
        deformation = Deformation.identity(height=20, width=20)
        cropped, origin = deformation.crop(origin=(5, 5), crop_size=(10, 10))
        assert cropped.is_identity
        assert cropped.field_shape == (10, 10)
        assert origin == (5, 5)

    def test_crop_clamps_origin(self) -> None:
        """Verifies that an origin past the field bound is clamped and the crop reads the clamped slice."""
        field_y = np.arange(1600, dtype=np.float32).reshape(40, 40)
        field_x = field_y * -1.0
        deformation = Deformation(field_y=field_y, field_x=field_x)

        cropped, origin = deformation.crop(origin=(35, 38), crop_size=(10, 6))

        # The crop must fit inside the field, so the origin clamps to (40 - 10, 40 - 6).
        assert origin == (30, 34)
        np.testing.assert_array_equal(cropped.get_field(dimension=0), field_y[30:40, 34:40])
        np.testing.assert_array_equal(cropped.get_field(dimension=1), field_x[30:40, 34:40])

    def test_crop_extracts_the_asymmetric_slice(self) -> None:
        """Verifies that an in-bounds asymmetric crop returns exactly the field slice its origin and size name."""
        field_y = np.arange(1600, dtype=np.float32).reshape(40, 40)
        field_x = field_y * 2.0
        deformation = Deformation(field_y=field_y, field_x=field_x)

        cropped, origin = deformation.crop(origin=(7, 11), crop_size=(8, 6))

        assert origin == (7, 11)
        assert cropped.field_shape == (8, 6)
        # The crop origin is (row, column), so the Y field's first element is field_y[7, 11] = 7 * 40 + 11 = 291.
        # Transposing the origin would read field_y[11, 7] = 447 instead.
        assert float(cropped.get_field(dimension=0)[0, 0]) == 291.0
        np.testing.assert_array_equal(cropped.get_field(dimension=0), field_y[7:15, 11:17])
        np.testing.assert_array_equal(cropped.get_field(dimension=1), field_x[7:15, 11:17])
