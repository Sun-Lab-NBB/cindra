"""Contains tests for the deformation module."""

import numpy as np
import pytest

from cindra.registration.deformation import (
    Deformation,
    zoom,
    _resize,
    diffuse,
    _make_samples_absolute,
    _create_diffusion_kernel,
)


class TestCreateDiffusionKernel:
    """Tests for _create_diffusion_kernel."""

    def test_small_sigma_returns_delta(self):
        """Verifies that sigma below threshold returns a single-element delta kernel."""
        kernel = _create_diffusion_kernel(sigma=0.05)
        np.testing.assert_array_equal(kernel, [1.0])

    def test_kernel_sums_to_one(self):
        """Verifies that the kernel sums to approximately 1.0 for all sigma values."""
        for sigma in [0.05, 0.5, 2.0, 5.0]:
            kernel = _create_diffusion_kernel(sigma=sigma)
            np.testing.assert_allclose(kernel.sum(), 1.0, atol=1e-6)

    def test_kernel_is_symmetric(self):
        """Verifies that the kernel is symmetric about its center."""
        kernel = _create_diffusion_kernel(sigma=3.0)
        np.testing.assert_allclose(kernel, kernel[::-1], atol=1e-6)

    def test_kernel_dtype(self):
        """Verifies the kernel dtype is float32."""
        kernel = _create_diffusion_kernel(sigma=2.0)
        assert kernel.dtype == np.float32


class TestDiffuse:
    """Tests for the diffuse function."""

    def test_small_sigma_identity(self):
        """Verifies that a very small sigma produces no smoothing."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((16, 16)).astype(np.float32)
        result = diffuse(data=data, sigma=0.01)
        np.testing.assert_array_equal(result, data)

    def test_output_shape_and_dtype(self):
        """Verifies that diffuse returns data with correct shape and dtype."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((32, 32)).astype(np.float32)
        result = diffuse(data=data, sigma=3.0)
        assert result.shape == data.shape
        assert result.dtype == np.float32

    def test_per_dimension_sigma(self):
        """Verifies that per-dimension sigma list is accepted."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((32, 32)).astype(np.float32)
        result = diffuse(data=data, sigma=[2.0, 0.01])
        assert result.dtype == np.float32
        assert result.shape == (32, 32)


class TestZoom:
    """Tests for the zoom function."""

    def test_upscale_shape(self):
        """Verifies the output shape after upscaling."""
        data = np.ones((10, 10), dtype=np.float32)
        result = zoom(data=data, factor=2.0)
        assert result.shape == (20, 20)

    def test_downscale_shape(self):
        """Verifies the output shape after downscaling."""
        data = np.ones((20, 20), dtype=np.float32)
        result = zoom(data=data, factor=0.5)
        assert result.shape == (10, 10)

    def test_per_dimension_factors(self):
        """Verifies that per-dimension factor tuple is accepted."""
        data = np.ones((10, 10), dtype=np.float32)
        result = zoom(data=data, factor=(2.0, 0.5))
        assert result.shape == (20, 5)

    def test_uniform_image_preserved(self):
        """Verifies that a uniform image remains uniform after zooming."""
        data = np.ones((10, 10), dtype=np.float32) * 42.0
        result = zoom(data=data, factor=2.0)
        np.testing.assert_allclose(result, 42.0, atol=1e-4)

    def test_dtype_preserved(self):
        """Verifies the output dtype matches the input."""
        data = np.ones((10, 10), dtype=np.float32)
        result = zoom(data=data, factor=1.5)
        assert result.dtype == np.float32

    @pytest.mark.parametrize("order", [0, 1, 3])
    def test_interpolation_orders(self, order):
        """Verifies that all interpolation orders produce valid output."""
        data = np.arange(100, dtype=np.float32).reshape(10, 10)
        result = zoom(data=data, factor=2.0, order=order)
        assert result.shape == (20, 20)
        assert np.isfinite(result).all()


class TestMakeSamplesAbsolute:
    """Tests for _make_samples_absolute."""

    def test_zero_deltas_give_identity_grid(self):
        """Verifies that zero displacement fields produce identity coordinate grids."""
        delta_x = np.zeros((4, 6), dtype=np.float32)
        delta_y = np.zeros((4, 6), dtype=np.float32)
        abs_x, abs_y = _make_samples_absolute(delta_x=delta_x, delta_y=delta_y)

        expected_x = np.arange(6, dtype=np.float32).reshape(1, 6)
        expected_y = np.arange(4, dtype=np.float32).reshape(4, 1)
        np.testing.assert_allclose(abs_x, np.broadcast_to(expected_x, (4, 6)))
        np.testing.assert_allclose(abs_y, np.broadcast_to(expected_y, (4, 6)))

    def test_with_known_deltas(self):
        """Verifies correct absolute coordinates with known displacement values."""
        delta_x = np.ones((3, 3), dtype=np.float32) * 0.5
        delta_y = np.ones((3, 3), dtype=np.float32) * -0.5
        abs_x, abs_y = _make_samples_absolute(delta_x=delta_x, delta_y=delta_y)
        # At pixel (1, 2): abs_x = 2 + 0.5 = 2.5, abs_y = 1 + (-0.5) = 0.5
        np.testing.assert_allclose(abs_x[1, 2], 2.5)
        np.testing.assert_allclose(abs_y[1, 2], 0.5)


class TestResize:
    """Tests for the _resize function."""

    def test_shape_change(self):
        """Verifies the output has the requested dimensions."""
        data = np.arange(100, dtype=np.float32).reshape(10, 10)
        result = _resize(data=data, new_height=20, new_width=30)
        assert result.shape == (20, 30)

    def test_same_size_preserves_values(self):
        """Verifies that resizing to the same dimensions preserves values."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((10, 10)).astype(np.float32)
        result = _resize(data=data, new_height=10, new_width=10)
        np.testing.assert_allclose(result, data, atol=1e-4)

    def test_uniform_image_preserved(self):
        """Verifies that a uniform image remains uniform after resizing."""
        data = np.ones((10, 10), dtype=np.float32) * 7.0
        result = _resize(data=data, new_height=25, new_width=25)
        np.testing.assert_allclose(result, 7.0, atol=1e-4)


class TestDeformationIdentity:
    """Tests for Deformation identity creation and properties."""

    def test_identity_creation(self):
        """Verifies identity deformation is created with correct shape."""
        deformation = Deformation.identity(height=10, width=20)
        assert deformation.is_identity
        assert deformation.field_shape == (10, 20)
        assert deformation.ndim == 2
        assert len(deformation) == 0

    def test_identity_repr(self):
        """Verifies the string representation for identity deformations."""
        deformation = Deformation.identity(height=10, width=20)
        assert "identity" in repr(deformation)

    def test_identity_copy(self):
        """Verifies that copying an identity deformation returns a new identity."""
        original = Deformation.identity(height=10, width=20)
        copied = original.copy()
        assert copied.is_identity
        assert copied.field_shape == (10, 20)

    def test_identity_apply_returns_data_unchanged(self):
        """Verifies that applying an identity deformation returns data unchanged."""
        data = np.arange(100, dtype=np.float32).reshape(10, 10)
        deformation = Deformation.identity(height=10, width=10)
        result = deformation.apply_deformation(data=data)
        np.testing.assert_array_equal(result, data)

    def test_identity_inverse(self):
        """Verifies that the inverse of identity is identity."""
        deformation = Deformation.identity(height=10, width=10)
        inverse = deformation.inverse()
        assert inverse.is_identity


class TestDeformationConstructor:
    """Tests for Deformation constructed from displacement fields."""

    def test_non_identity_properties(self):
        """Verifies properties of a non-identity deformation."""
        field_y = np.zeros((10, 20), dtype=np.float32)
        field_x = np.zeros((10, 20), dtype=np.float32)
        deformation = Deformation(field_y=field_y, field_x=field_x)
        assert not deformation.is_identity
        assert deformation.field_shape == (10, 20)
        assert deformation.ndim == 2
        assert len(deformation) == 2

    def test_repr_includes_shape(self):
        """Verifies the string representation includes the field shape."""
        field_y = np.zeros((10, 20), dtype=np.float32)
        field_x = np.zeros((10, 20), dtype=np.float32)
        deformation = Deformation(field_y=field_y, field_x=field_x)
        assert "10x20" in repr(deformation)

    def test_getitem_and_iter(self):
        """Verifies field access via indexing and iteration."""
        field_y = np.ones((5, 5), dtype=np.float32) * 1.0
        field_x = np.ones((5, 5), dtype=np.float32) * 2.0
        deformation = Deformation(field_y=field_y, field_x=field_x)
        np.testing.assert_array_equal(deformation[0], field_y)
        np.testing.assert_array_equal(deformation[1], field_x)
        fields = list(deformation)
        assert len(fields) == 2

    def test_get_field(self):
        """Verifies get_field returns the correct field array."""
        field_y = np.ones((5, 5), dtype=np.float32) * 3.0
        field_x = np.ones((5, 5), dtype=np.float32) * 4.0
        deformation = Deformation(field_y=field_y, field_x=field_x)
        np.testing.assert_array_equal(deformation.get_field(dimension=0), field_y)
        np.testing.assert_array_equal(deformation.get_field(dimension=1), field_x)


class TestDeformationScale:
    """Tests for Deformation.scale."""

    def test_scale_by_factor(self):
        """Verifies that scaling multiplies all displacement values."""
        field_y = np.ones((5, 5), dtype=np.float32) * 2.0
        field_x = np.ones((5, 5), dtype=np.float32) * 3.0
        deformation = Deformation(field_y=field_y, field_x=field_x)
        scaled = deformation.scale(factor=0.5)
        np.testing.assert_allclose(scaled[0], 1.0)
        np.testing.assert_allclose(scaled[1], 1.5)

    def test_scale_by_one_copies(self):
        """Verifies that scale(1.0) creates a copy with identical values."""
        field_y = np.ones((5, 5), dtype=np.float32) * 2.0
        field_x = np.ones((5, 5), dtype=np.float32) * 3.0
        deformation = Deformation(field_y=field_y, field_x=field_x)
        copied = deformation.scale(factor=1.0)
        np.testing.assert_array_equal(copied[0], field_y)
        # Verify it's a copy, not a view.
        copied[0][0, 0] = 999.0
        assert deformation[0][0, 0] != 999.0


class TestDeformationAdd:
    """Tests for Deformation.add and __add__."""

    def test_add_identity_left(self):
        """Verifies that identity + deformation returns a copy of the deformation."""
        identity = Deformation.identity(height=5, width=5)
        field_y = np.ones((5, 5), dtype=np.float32) * 2.0
        field_x = np.ones((5, 5), dtype=np.float32) * 3.0
        deformation = Deformation(field_y=field_y, field_x=field_x)
        result = identity.add(deformation)
        np.testing.assert_array_equal(result[0], field_y)
        np.testing.assert_array_equal(result[1], field_x)

    def test_add_identity_right(self):
        """Verifies that deformation + identity returns a copy of the deformation."""
        identity = Deformation.identity(height=5, width=5)
        field_y = np.ones((5, 5), dtype=np.float32) * 2.0
        field_x = np.ones((5, 5), dtype=np.float32) * 3.0
        deformation = Deformation(field_y=field_y, field_x=field_x)
        result = deformation + identity
        np.testing.assert_array_equal(result[0], field_y)

    def test_add_two_deformations(self):
        """Verifies element-wise addition of two deformations."""
        field_y_1 = np.ones((5, 5), dtype=np.float32) * 1.0
        field_x_1 = np.ones((5, 5), dtype=np.float32) * 2.0
        field_y_2 = np.ones((5, 5), dtype=np.float32) * 3.0
        field_x_2 = np.ones((5, 5), dtype=np.float32) * 4.0
        deformation_1 = Deformation(field_y=field_y_1, field_x=field_x_1)
        deformation_2 = Deformation(field_y=field_y_2, field_x=field_x_2)
        result = deformation_1 + deformation_2
        np.testing.assert_allclose(result[0], 4.0)
        np.testing.assert_allclose(result[1], 6.0)


class TestDeformationCompose:
    """Tests for Deformation.compose."""

    def test_compose_with_identity_left(self):
        """Verifies that identity.compose(d) returns a copy of d."""
        identity = Deformation.identity(height=10, width=10)
        field_y = np.ones((10, 10), dtype=np.float32) * 0.5
        field_x = np.ones((10, 10), dtype=np.float32) * -0.5
        deformation = Deformation(field_y=field_y, field_x=field_x)
        result = identity.compose(deformation)
        np.testing.assert_allclose(result[0], 0.5, atol=1e-5)
        np.testing.assert_allclose(result[1], -0.5, atol=1e-5)

    def test_compose_with_identity_right(self):
        """Verifies that d.compose(identity) returns a copy of d."""
        identity = Deformation.identity(height=10, width=10)
        field_y = np.ones((10, 10), dtype=np.float32) * 0.5
        field_x = np.ones((10, 10), dtype=np.float32) * -0.5
        deformation = Deformation(field_y=field_y, field_x=field_x)
        result = deformation.compose(identity)
        np.testing.assert_allclose(result[0], 0.5, atol=1e-5)

    def test_compose_two_uniform_deformations(self):
        """Verifies composition of two small uniform displacements."""
        field_y = np.ones((20, 20), dtype=np.float32) * 0.3
        field_x = np.ones((20, 20), dtype=np.float32) * 0.2
        deformation_1 = Deformation(field_y=field_y, field_x=field_x)
        deformation_2 = Deformation(field_y=field_y.copy(), field_x=field_x.copy())
        result = deformation_1.compose(deformation_2)
        # For small uniform displacements, composition ≈ addition at interior pixels.
        np.testing.assert_allclose(result[0][5:-5, 5:-5], 0.6, atol=0.05)
        np.testing.assert_allclose(result[1][5:-5, 5:-5], 0.4, atol=0.05)


class TestDeformationResizeField:
    """Tests for Deformation.resize_field."""

    def test_resize_identity(self):
        """Verifies that resizing an identity deformation returns a new identity."""
        deformation = Deformation.identity(height=10, width=10)
        resized = deformation.resize_field(new_height=20, new_width=20)
        assert resized.is_identity
        assert resized.field_shape == (20, 20)

    def test_resize_same_size_returns_self(self):
        """Verifies that resizing to the same dimensions returns self."""
        field_y = np.zeros((10, 10), dtype=np.float32)
        field_x = np.zeros((10, 10), dtype=np.float32)
        deformation = Deformation(field_y=field_y, field_x=field_x)
        resized = deformation.resize_field(new_height=10, new_width=10)
        assert resized is deformation

    def test_resize_changes_shape(self):
        """Verifies that resizing produces a deformation with new dimensions."""
        field_y = np.ones((10, 10), dtype=np.float32) * 0.5
        field_x = np.ones((10, 10), dtype=np.float32) * 0.5
        deformation = Deformation(field_y=field_y, field_x=field_x)
        resized = deformation.resize_field(new_height=20, new_width=20)
        assert resized.field_shape == (20, 20)


class TestDeformationApply:
    """Tests for Deformation.apply_deformation."""

    def test_zero_displacement_preserves_image(self):
        """Verifies that zero displacement fields preserve the image."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((20, 20)).astype(np.float32)
        field_y = np.zeros((20, 20), dtype=np.float32)
        field_x = np.zeros((20, 20), dtype=np.float32)
        deformation = Deformation(field_y=field_y, field_x=field_x)
        result = deformation.apply_deformation(data=data)
        np.testing.assert_allclose(result, data, atol=1e-5)

    def test_applies_to_different_sized_data(self):
        """Verifies that deformation resizes its field to match data shape."""
        data = np.ones((20, 20), dtype=np.float32) * 5.0
        field_y = np.zeros((10, 10), dtype=np.float32)
        field_x = np.zeros((10, 10), dtype=np.float32)
        deformation = Deformation(field_y=field_y, field_x=field_x)
        result = deformation.apply_deformation(data=data)
        assert result.shape == (20, 20)
        np.testing.assert_allclose(result, 5.0, atol=1e-4)


class TestDeformationInverse:
    """Tests for Deformation.inverse."""

    def test_inverse_of_small_displacement(self):
        """Verifies that inverse approximately negates the displacement for small fields."""
        field_y = np.ones((20, 20), dtype=np.float32) * 0.3
        field_x = np.ones((20, 20), dtype=np.float32) * -0.2
        deformation = Deformation(field_y=field_y, field_x=field_x)
        inverse = deformation.inverse()
        # For small displacements, inverse ≈ negation at interior pixels.
        np.testing.assert_allclose(inverse[0][5:-5, 5:-5], -0.3, atol=0.1)
        np.testing.assert_allclose(inverse[1][5:-5, 5:-5], 0.2, atol=0.1)


class TestDeformationGetDeformationLocations:
    """Tests for Deformation.get_deformation_locations."""

    def test_returns_absolute_coordinates(self):
        """Verifies that the returned coordinates are absolute pixel positions."""
        field_y = np.ones((5, 5), dtype=np.float32) * 0.5
        field_x = np.ones((5, 5), dtype=np.float32) * -0.5
        deformation = Deformation(field_y=field_y, field_x=field_x)
        abs_x, abs_y = deformation.get_deformation_locations()
        # At pixel (2, 3): abs_x = 3 + (-0.5) = 2.5, abs_y = 2 + 0.5 = 2.5
        np.testing.assert_allclose(abs_x[2, 3], 2.5)
        np.testing.assert_allclose(abs_y[2, 3], 2.5)


class TestDeformationRegularize:
    """Tests for Deformation.regularize."""

    def test_regularize_identity(self):
        """Verifies that regularizing an identity deformation returns identity."""
        deformation = Deformation.identity(height=20, width=20)
        result = deformation.regularize(grid_sampling=5.0)
        assert result.is_identity

    def test_regularize_smooth_field(self):
        """Verifies that regularization produces a valid deformation."""
        field_y = np.ones((30, 30), dtype=np.float32) * 0.5
        field_x = np.ones((30, 30), dtype=np.float32) * 0.5
        deformation = Deformation(field_y=field_y, field_x=field_x)
        result = deformation.regularize(grid_sampling=5.0)
        assert result is not None
        assert result.field_shape == (30, 30)


class TestDeformationCrop:
    """Tests for Deformation.crop."""

    def test_crop_identity(self):
        """Verifies that cropping an identity deformation returns a smaller identity."""
        deformation = Deformation.identity(height=20, width=20)
        cropped, origin = deformation.crop(origin=(5, 5), crop_size=(10, 10))
        assert cropped.is_identity
        assert cropped.field_shape == (10, 10)
        assert origin == (5, 5)

    def test_crop_clamps_origin(self):
        """Verifies that cropping clamps the origin to valid bounds."""
        field_y = np.zeros((20, 20), dtype=np.float32)
        field_x = np.zeros((20, 20), dtype=np.float32)
        deformation = Deformation(field_y=field_y, field_x=field_x)
        _cropped, origin = deformation.crop(origin=(15, 15), crop_size=(10, 10))
        assert origin == (10, 10)

    def test_crop_shape(self):
        """Verifies the cropped deformation has the requested shape."""
        field_y = np.zeros((20, 20), dtype=np.float32)
        field_x = np.zeros((20, 20), dtype=np.float32)
        deformation = Deformation(field_y=field_y, field_x=field_x)
        cropped, _ = deformation.crop(origin=(2, 3), crop_size=(8, 8))
        assert cropped.field_shape == (8, 8)
