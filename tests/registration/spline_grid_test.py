"""Contains tests for the spline_grid module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.interpolate import BSpline

from cindra.registration.spline_grid import (
    SplineGrid,
    _sample_grid,
    _fit_knots_to_field,
    _compute_basis_coefficients,
    compute_cardinal_coefficients,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

_CATMULL_ROM_BASIS: NDArray[np.float64] = 0.5 * np.array(
    [[0.0, 2.0, 0.0, 0.0], [-1.0, 0.0, 1.0, 0.0], [2.0, -5.0, 4.0, -1.0], [-1.0, 3.0, -3.0, 1.0]], dtype=np.float64
)
"""The standard Catmull-Rom basis matrix, whose rows weight the powers 1, t, t squared, and t cubed."""

_UNIFORM_BSPLINE_KNOTS: NDArray[np.float64] = np.arange(-3, 5, dtype=np.float64)
"""The uniform knot vector spanning the four cubic B-spline basis functions that are non-zero on the unit interval."""


class TestComputeCardinalCoefficients:
    """Tests compute_cardinal_coefficients."""

    @pytest.mark.parametrize("interpolation_factor", [0.0, 0.125, 0.25, 0.5, 0.75, 1.0])
    def test_matches_the_catmull_rom_basis_matrix(self, interpolation_factor: float) -> None:
        """Verifies the coefficients equal the standard Catmull-Rom basis matrix applied to the power vector."""
        coefficients = np.empty(4, dtype=np.float32)
        compute_cardinal_coefficients(interpolation_factor=interpolation_factor, coefficients=coefficients)

        powers = np.array(
            [1.0, interpolation_factor, interpolation_factor**2, interpolation_factor**3], dtype=np.float64
        )
        np.testing.assert_allclose(coefficients, powers @ _CATMULL_ROM_BASIS, atol=1e-7)

    # Deliberately samples positions the basis-matrix test above does not, so that this invariant covers points that
    # test leaves unchecked rather than restating a consequence of it.
    @pytest.mark.parametrize("interpolation_factor", [0.05, 0.33, 0.4, 0.61, 0.87, 0.99])
    def test_coefficients_form_a_partition_of_unity(self, interpolation_factor: float) -> None:
        """Verifies the four coefficients sum to one, which is what keeps a constant image constant under warping."""
        coefficients = np.empty(4, dtype=np.float32)
        compute_cardinal_coefficients(interpolation_factor=interpolation_factor, coefficients=coefficients)
        assert float(np.sum(coefficients.astype(np.float64))) == pytest.approx(1.0, abs=1e-7)

    def test_interpolates_through_its_control_points(self) -> None:
        """Verifies that a factor of zero and of one select the two central control points exactly."""
        coefficients = np.empty(4, dtype=np.float32)

        compute_cardinal_coefficients(interpolation_factor=0.0, coefficients=coefficients)
        np.testing.assert_array_equal(np.abs(coefficients), np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))

        compute_cardinal_coefficients(interpolation_factor=1.0, coefficients=coefficients)
        np.testing.assert_array_equal(np.abs(coefficients), np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32))


class TestComputeBasisCoefficients:
    """Tests _compute_basis_coefficients."""

    @pytest.mark.parametrize("interpolation_factor", [0.0, 0.2, 0.5, 0.75, 0.9])
    def test_matches_the_scipy_bspline_basis(self, interpolation_factor: float) -> None:
        """Verifies the coefficients equal scipy's uniform cubic B-spline basis functions at the same position."""
        coefficients = np.empty(4, dtype=np.float32)
        _compute_basis_coefficients(interpolation_factor=interpolation_factor, coefficients=coefficients)
        np.testing.assert_allclose(
            coefficients, _reference_bspline_basis(interpolation_factor=interpolation_factor), atol=1e-7
        )

    # Deliberately samples positions the scipy-oracle test above does not, so that this invariant covers points that
    # test leaves unchecked rather than restating a consequence of it.
    @pytest.mark.parametrize("interpolation_factor", [0.05, 0.33, 0.61, 0.87, 0.99])
    def test_coefficients_form_a_partition_of_unity(self, interpolation_factor: float) -> None:
        """Verifies the four B-spline coefficients sum to one at every position in the unit interval."""
        coefficients = np.empty(4, dtype=np.float32)
        _compute_basis_coefficients(interpolation_factor=interpolation_factor, coefficients=coefficients)
        assert float(np.sum(coefficients.astype(np.float64))) == pytest.approx(1.0, abs=1e-7)

    def test_the_basis_is_symmetric_about_the_interval_center(self) -> None:
        """Verifies that reflecting the position reverses the coefficient order, as a uniform basis requires."""
        forward = np.empty(4, dtype=np.float32)
        reflected = np.empty(4, dtype=np.float32)
        _compute_basis_coefficients(interpolation_factor=0.3, coefficients=forward)
        _compute_basis_coefficients(interpolation_factor=0.7, coefficients=reflected)
        np.testing.assert_allclose(forward, reflected[::-1], atol=1e-7)


class TestSampleGrid:
    """Tests the _sample_grid kernel."""

    def test_matches_a_direct_four_by_four_basis_sum(self) -> None:
        """Verifies each sampled pixel equals the scipy-basis weighted sum over its own 4x4 knot neighborhood."""
        generator = np.random.default_rng(seed=21)
        knots = generator.standard_normal((9, 10)).astype(np.float32)
        grid_sampling = 4.0
        height, width = 12, 14

        result = np.empty((height, width), dtype=np.float32)
        _sample_grid(result=result, grid_sampling=grid_sampling, knots=knots)

        reference = np.zeros((height, width), dtype=np.float64)
        for row in range(height):
            grid_position_y = row / grid_sampling + 1
            knot_index_y = int(grid_position_y)
            row_basis = _reference_bspline_basis(interpolation_factor=grid_position_y - knot_index_y)
            for column in range(width):
                grid_position_x = column / grid_sampling + 1
                knot_index_x = int(grid_position_x)
                column_basis = _reference_bspline_basis(interpolation_factor=grid_position_x - knot_index_x)
                neighborhood = knots[knot_index_y - 1 : knot_index_y + 3, knot_index_x - 1 : knot_index_x + 3]
                reference[row, column] = row_basis @ neighborhood.astype(np.float64) @ column_basis

        # The knot values are of order one, so this bound rules out any misplaced knot index or basis term.
        np.testing.assert_allclose(result, reference, atol=1e-5)


class TestFitKnotsToField:
    """Tests the _fit_knots_to_field least-squares kernel."""

    def test_matches_a_direct_lee_accumulation(self) -> None:
        """Verifies the fitted knots equal an independent accumulation of the Lee least-squares contributions."""
        grid_sampling = 4.0
        rows, columns = np.meshgrid(np.arange(13.0), np.arange(11.0), indexing="ij")

        # A smooth field the spline can follow, so the fitted knots carry values of order one rather than the near
        # cancellation an unstructured field produces.
        field = (np.sin(rows / 3.0) * np.cos(columns / 4.0)).astype(np.float32)
        grid_shape = SplineGrid.compute_grid_shape(field_height=13, field_width=11, grid_sampling=grid_sampling)

        knots = np.zeros(grid_shape, dtype=np.float32)
        _fit_knots_to_field(grid_sampling=grid_sampling, knots=knots, field=field)

        numerator = np.zeros(grid_shape, dtype=np.float64)
        denominator = np.zeros(grid_shape, dtype=np.float64)
        for row in range(field.shape[0]):
            grid_position_y = row / grid_sampling + 1
            knot_index_y = int(grid_position_y)
            row_basis = _reference_bspline_basis(interpolation_factor=grid_position_y - knot_index_y)
            for column in range(field.shape[1]):
                grid_position_x = column / grid_sampling + 1
                knot_index_x = int(grid_position_x)
                column_basis = _reference_bspline_basis(interpolation_factor=grid_position_x - knot_index_x)

                # Each pixel spreads its value over the 4x4 knot neighborhood weighted by the squared basis product,
                # after being pre-normalized by the sum of those squares (Lee, Wolberg and Shin 1997).
                weights = np.outer(row_basis, column_basis)
                normalized_value = float(field[row, column]) / float(np.sum(weights * weights))
                window = (
                    slice(knot_index_y - 1, knot_index_y + 3),
                    slice(knot_index_x - 1, knot_index_x + 3),
                )
                numerator[window] += weights * weights * (normalized_value * weights)
                denominator[window] += weights * weights

        reference = np.zeros(grid_shape, dtype=np.float64)
        populated = denominator > 0.0
        reference[populated] = numerator[populated] / denominator[populated]

        np.testing.assert_allclose(knots, reference, atol=1e-4)

        # Confirms the comparison is not dominated by untouched zeros. Every knot row but the trailing padding one
        # lies within some pixel's 4x4 neighborhood, and the fitted values are of the same order as the field.
        assert int(np.count_nonzero(populated)) == (grid_shape[0] - 1) * grid_shape[1]
        assert float(np.max(np.abs(knots))) > 0.5


class TestSplineGridInit:
    """Tests SplineGrid initialization and properties."""

    def test_properties(self) -> None:
        """Verifies core properties after initialization."""
        grid = SplineGrid(field_height=100, field_width=200, sampling=10.0)
        assert grid.dimension_count == 2
        assert grid.field_shape == (100, 200)
        assert grid.grid_sampling == 10.0

    def test_grid_shape_formula(self) -> None:
        """Verifies the grid shape follows int((dimension - 1) / sampling) + 4."""
        grid = SplineGrid(field_height=100, field_width=50, sampling=10.0)
        expected_height = int((100 - 1) / 10.0) + 4
        expected_width = int((50 - 1) / 10.0) + 4
        assert grid.grid_shape == (expected_height, expected_width)

    def test_compute_grid_shape_static(self) -> None:
        """Verifies the static method produces the same result as the constructor."""
        grid = SplineGrid(field_height=100, field_width=200, sampling=10.0)
        static_shape = SplineGrid.compute_grid_shape(field_height=100, field_width=200, grid_sampling=10.0)
        assert static_shape == grid.grid_shape

    def test_repr_reports_the_derived_knot_grid(self) -> None:
        """Verifies the representation carries the knot grid shape the constructor derived from field and sampling."""
        grid = SplineGrid(field_height=100, field_width=50, sampling=10.0)

        # The grid spans int((dimension - 1) / sampling) whole intervals plus four padding knots, which is
        # int(99 / 10) + 4 = 13 rows and int(49 / 10) + 4 = 8 columns.
        assert repr(grid) == "SplineGrid(field_shape=(100, 50), grid_shape=(13, 8), grid_sampling=10.0)"

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
        # B-spline fit is approximate, so the check targets interior pixels that avoid edge effects.
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
        # A displacement this large engages the injectivity constraint.
        field_y = np.ones((50, 50), dtype=np.float32) * 100.0
        field_x = np.ones((50, 50), dtype=np.float32) * 100.0
        grid.set_from_fields(field_y=field_y, field_x=field_x, injective=True, freeze_edges=False)
        recovered_y, _ = grid.deformation_fields
        # Expects the injectivity constraint to bound the recovered values below the theoretical limit.
        limit = (1.0 / 2.046392675) * 5.0 * 0.9
        assert np.max(np.abs(recovered_y)) < limit * 2.0


def _reference_bspline_basis(interpolation_factor: float) -> NDArray[np.float64]:
    """Evaluates the four non-zero uniform cubic B-spline basis functions through scipy's de Boor implementation."""
    values = []
    for basis_index in range(4):
        coefficients = np.zeros(len(_UNIFORM_BSPLINE_KNOTS) - 4, dtype=np.float64)
        coefficients[basis_index] = 1.0
        values.append(float(BSpline(_UNIFORM_BSPLINE_KNOTS, coefficients, 3)(interpolation_factor)))
    return np.array(values, dtype=np.float64)
