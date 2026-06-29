"""Contains tests for the spline_grid module."""

from __future__ import annotations

import numpy as np

from cindra.registration.spline_grid import SplineGrid


class TestSplineGridInit:
    """Tests SplineGrid initialization and properties."""

    def test_properties(self) -> None:
        """Verifies core properties after initialization."""
        grid = SplineGrid(field_height=100, field_width=200, sampling=10.0)
        assert grid.ndim == 2
        assert grid.field_shape == (100, 200)
        assert grid.grid_sampling == 10.0

    def test_grid_shape_formula(self) -> None:
        """Verifies the grid shape follows int((dim - 1) / sampling) + 4."""
        grid = SplineGrid(field_height=100, field_width=50, sampling=10.0)
        expected_height = int((100 - 1) / 10.0) + 4
        expected_width = int((50 - 1) / 10.0) + 4
        assert grid.grid_shape == (expected_height, expected_width)

    def test_compute_grid_shape_static(self) -> None:
        """Verifies the static method produces the same result as the constructor."""
        grid = SplineGrid(field_height=100, field_width=200, sampling=10.0)
        static_shape = SplineGrid.compute_grid_shape(field_height=100, field_width=200, grid_sampling=10.0)
        assert static_shape == grid.grid_shape

    def test_initial_knots_are_zero(self) -> None:
        """Verifies that the initial knot arrays are all zeros."""
        grid = SplineGrid(field_height=50, field_width=50, sampling=5.0)
        fields = grid.deformation_fields
        np.testing.assert_array_equal(fields[0], 0.0)
        np.testing.assert_array_equal(fields[1], 0.0)


class TestSplineGridDeformationFields:
    """Tests SplineGrid.deformation_fields property."""

    def test_output_shapes(self) -> None:
        """Verifies the deformation fields have the correct shape."""
        grid = SplineGrid(field_height=50, field_width=60, sampling=5.0)
        field_y, field_x = grid.deformation_fields
        assert field_y.shape == (50, 60)
        assert field_x.shape == (50, 60)

    def test_output_dtypes(self) -> None:
        """Verifies the deformation fields are float32."""
        grid = SplineGrid(field_height=50, field_width=50, sampling=5.0)
        field_y, field_x = grid.deformation_fields
        assert field_y.dtype == np.float32
        assert field_x.dtype == np.float32


class TestSplineGridSetFromFields:
    """Tests SplineGrid.set_from_fields and roundtrip behavior."""

    def test_set_from_fields_returns_true(self) -> None:
        """Verifies that set_from_fields succeeds for a valid grid."""
        grid = SplineGrid(field_height=50, field_width=50, sampling=5.0)
        field_y = np.ones((50, 50), dtype=np.float32) * 0.5
        field_x = np.ones((50, 50), dtype=np.float32) * 0.5
        success = grid.set_from_fields(field_y=field_y, field_x=field_x)
        assert success

    def test_roundtrip_approximation(self) -> None:
        """Verifies that setting a uniform field and reading back produces approximate values."""
        grid = SplineGrid(field_height=50, field_width=50, sampling=5.0)
        target_value = 0.3
        field_y = np.ones((50, 50), dtype=np.float32) * target_value
        field_x = np.ones((50, 50), dtype=np.float32) * target_value
        grid.set_from_fields(field_y=field_y, field_x=field_x)
        recovered_y, recovered_x = grid.deformation_fields
        # B-spline fit is approximate; check interior pixels avoid edge effects.
        np.testing.assert_allclose(recovered_y[10:-10, 10:-10], target_value, atol=0.15)
        np.testing.assert_allclose(recovered_x[10:-10, 10:-10], target_value, atol=0.15)

    def test_set_from_fields_without_injective(self) -> None:
        """Verifies set_from_fields works without injectivity constraint."""
        grid = SplineGrid(field_height=50, field_width=50, sampling=5.0)
        field_y = np.ones((50, 50), dtype=np.float32) * 0.5
        field_x = np.ones((50, 50), dtype=np.float32) * 0.5
        success = grid.set_from_fields(field_y=field_y, field_x=field_x, injective=False)
        assert success
        # Without the injectivity constraint, the interior should faithfully reproduce the uniform input field.
        recovered_y, recovered_x = grid.deformation_fields
        np.testing.assert_allclose(recovered_y[10:-10, 10:-10], 0.5, atol=0.15)
        np.testing.assert_allclose(recovered_x[10:-10, 10:-10], 0.5, atol=0.15)

    def test_set_from_fields_without_frozen_edges(self) -> None:
        """Verifies set_from_fields works without frozen edges."""
        grid = SplineGrid(field_height=50, field_width=50, sampling=5.0)
        field_y = np.ones((50, 50), dtype=np.float32) * 0.5
        field_x = np.ones((50, 50), dtype=np.float32) * 0.5
        success = grid.set_from_fields(field_y=field_y, field_x=field_x, freeze_edges=False)
        assert success
        # With edges left unfrozen, the interior should track the uniform input field closely.
        recovered_y, recovered_x = grid.deformation_fields
        np.testing.assert_allclose(recovered_y[10:-10, 10:-10], 0.5, atol=0.15)
        np.testing.assert_allclose(recovered_x[10:-10, 10:-10], 0.5, atol=0.15)


class TestSplineGridFreezeEdges:
    """Tests SplineGrid._freeze_edges behavior."""

    def test_frozen_edges_produce_zero_at_boundary(self) -> None:
        """Verifies that frozen edges produce approximately zero deformation at boundaries."""
        grid = SplineGrid(field_height=50, field_width=50, sampling=5.0)
        field_y = np.ones((50, 50), dtype=np.float32) * 0.5
        field_x = np.ones((50, 50), dtype=np.float32) * 0.5
        grid.set_from_fields(field_y=field_y, field_x=field_x, freeze_edges=True)
        recovered_y, recovered_x = grid.deformation_fields
        # Freezing edges clamps the boundary knots, so boundary pixels collapse toward zero.
        np.testing.assert_allclose(recovered_y[0, :], 0.0, atol=0.05)
        np.testing.assert_allclose(recovered_y[-1, :], 0.0, atol=0.05)
        np.testing.assert_allclose(recovered_x[:, 0], 0.0, atol=0.05)
        np.testing.assert_allclose(recovered_x[:, -1], 0.0, atol=0.05)

    def test_freeze_edges_fails_for_small_grid(self) -> None:
        """Verifies that freeze_edges returns False when the grid is too small."""
        # With very large sampling, the grid will have fewer than 6 knots.
        grid = SplineGrid(field_height=10, field_width=10, sampling=50.0)
        field_y = np.ones((10, 10), dtype=np.float32) * 0.5
        field_x = np.ones((10, 10), dtype=np.float32) * 0.5
        success = grid.set_from_fields(field_y=field_y, field_x=field_x, freeze_edges=True)
        assert not success


class TestSplineGridUnfold:
    """Tests SplineGrid._unfold injectivity constraint."""

    def test_unfold_limits_large_knots(self) -> None:
        """Verifies that unfold constrains large displacement values."""
        grid = SplineGrid(field_height=50, field_width=50, sampling=5.0)
        # Sets very large displacement to test the constraint.
        field_y = np.ones((50, 50), dtype=np.float32) * 100.0
        field_x = np.ones((50, 50), dtype=np.float32) * 100.0
        grid.set_from_fields(field_y=field_y, field_x=field_x, injective=True, freeze_edges=False)
        recovered_y, _ = grid.deformation_fields
        # After injectivity constraint, values should be bounded below the theoretical limit.
        limit = (1.0 / 2.046392675) * 5.0 * 0.9
        assert np.max(np.abs(recovered_y)) < limit * 2.0
