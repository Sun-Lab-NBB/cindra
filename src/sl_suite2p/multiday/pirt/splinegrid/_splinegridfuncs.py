"""Low level functions for spline grids, implemented with Numba to make it super fast.

Provides functionality for (cubic B-spline) grids.

Copyright 2010-2012 (C) Almar Klein, University of Twente.
Copyright 2012-2017 (C) Almar Klein

A Note on pointsets:
In this module, all pointsets are 2D numpy arrays shaped (N, ndim),
which express world coordinates. The coordinates they represent
should be expressed using doubles wx,wy,wz, in contrast to pixel
coordinates, which are expressed using integers x, y, z.
"""

import numba
import numpy as np

from ..interp.spline_coefficients import compute_basis_coefficients

## Functions to obtain the field that the grid represents


def get_field(grid):
    """get_field(grid)
    Sample the 2D grid at all the pixels of the underlying field.
    """
    if grid.ndim != 2:
        raise ValueError("Only 2D grids are supported.")

    # Init resulting array, make it float32, since for deformation-grids
    # the result is a sampler to be used in interp().
    result = np.zeros(grid.field_shape, dtype=np.float32)
    _get_field2(result, grid.grid_sampling_in_pixels, grid.knots)

    return result


def get_field_sparse(grid, pp):
    """get_field_sparse(grid, pp)

    Sparsely sample the 2D grid at a specified set of points (which are in
    world coordinates).

    Also see get_field_at().

    """
    assert isinstance(pp, np.ndarray) and pp.ndim == 2

    if grid.ndim != 2:
        raise ValueError("Only 2D grids are supported.")
    if pp.shape[1] != 2:
        raise ValueError("Pointset must have 2 columns for 2D grid.")

    # Create samples
    samples = [pp[:, 0], pp[:, 1]]

    # Init result
    result = np.zeros_like(samples[0], dtype=np.float32)

    # Determine sampling
    grid_sampling_in_pixels = tuple([grid.grid_sampling for _ in grid.grid_sampling_in_pixels])

    _get_field_at2(result.ravel(), grid_sampling_in_pixels, grid.knots, samples[0].ravel(), samples[1].ravel())

    return result


def get_field_at(grid, samples):
    """get_field_at(grid, samples)

    Sample the 2D grid at specified sample locations (in pixels, x-y order),
    similar to pirt.interp.interp().

    Also see get_field_sparse().

    """
    if grid.ndim != 2:
        raise ValueError("Only 2D grids are supported.")
    if not isinstance(samples, (tuple, list)):
        raise ValueError("Samples must be list or tuple.")
    if len(samples) != 2:
        raise ValueError("Samples must contain exactly 2 elements for 2D grid.")
    if samples[0].shape != samples[1].shape:
        raise ValueError("Sample arrays must have the same shape.")

    # Init result
    result = np.zeros_like(samples[0], dtype=np.float32)

    _get_field_at2(result.ravel(), grid.grid_sampling_in_pixels, grid.knots, samples[0].ravel(), samples[1].ravel())

    return result


## Workhorse functions to get the field


@numba.jit(nopython=True, nogil=True)
def _get_field2(result, grid_sampling_in_pixels, knots):
    if result.ndim != 2:
        raise ValueError("This function can only sample 2D grids.")

    ccy = np.empty((4,), np.float32)
    ccx = np.empty((4,), np.float32)

    grid_ySpacing = grid_sampling_in_pixels[0]
    grid_xSpacing = grid_sampling_in_pixels[1]

    # For each pixel ...
    for y in range(result.shape[0]):
        for x in range(result.shape[1]):
            # Calculate what is the leftmost (reference) knot on the grid,
            # and the ratio between closest and second closest knot.
            # Note the +1 to correct for padding.
            tmp = y / grid_ySpacing + 1
            gy = int(tmp)
            ty = tmp - gy
            tmp = x / grid_xSpacing + 1
            gx = int(tmp)
            tx = tmp - gx

            # Get coefficients
            compute_basis_coefficients(ty, ccy)
            compute_basis_coefficients(tx, ccx)

            # Init value
            val = 0.0

            # For each knot ...
            jj = gy - 1  # y-location of first knot
            for j in range(4):
                ii = gx - 1  # x-location of first knot
                for i in range(4):
                    # Calculate interpolated value.
                    val += ccy[j] * ccx[i] * knots[jj, ii]
                    ii += 1
                jj += 1

            # Store value in result array
            result[y, x] = val


@numba.jit(nopython=True, nogil=True)
def _get_field_at2(result_, grid_sampling_in_pixels, knots, samplesx_, samplesy_):
    assert samplesx_.ndim == 1
    assert samplesy_.ndim == 1

    ccy = np.empty((4,), np.float32)
    ccx = np.empty((4,), np.float32)

    grid_ySpacing = grid_sampling_in_pixels[0]
    grid_xSpacing = grid_sampling_in_pixels[1]

    gridShapey = knots.shape[0]
    gridShapex = knots.shape[1]

    # For each point in the set
    for p in range(samplesx_.size):
        # Calculate wx and wy
        wx = samplesx_[p]
        wy = samplesy_[p]

        # Calculate what is the leftmost (reference) knot on the grid,
        # and the ratio between closest and second closest knot.
        # Note the +1 to correct for padding.
        tmp = wy / grid_ySpacing + 1
        gy = int(tmp)
        ty = tmp - gy
        tmp = wx / grid_xSpacing + 1
        gx = int(tmp)
        tx = tmp - gx

        # Check if within bounds of interpolatable domain
        if (gy < 1 or gy >= gridShapey - 2) or (gx < 1 or gx >= gridShapex - 2):
            result_[p] = 0.0
            continue

        # Get coefficients
        compute_basis_coefficients(ty, ccy)
        compute_basis_coefficients(tx, ccx)

        # Init value
        val = 0.0

        # For each knot ...
        jj = gy - 1  # y-location of first knot
        for j in range(4):
            ii = gx - 1  # x-location of first knot
            for i in range(4):
                # Calculate interpolated value.
                val += ccy[j] * ccx[i] * knots[jj, ii]
                ii += 1
            jj += 1

        # Store
        result_[p] = val


## Functions to set the grid using a field


@numba.jit(nopython=True, nogil=True)
def _set_field_using_num_and_dnum(knots_, num_, dnum_):
    for i in range(knots_.size):
        n = dnum_[i]
        if n > 0.0:
            knots_[i] = num_[i] / n


def set_field(grid, field, weights):
    """set_field(grid, field, weights)
    Set the 2D grid using the specified field and weights.
    """
    if grid.ndim != 2:
        raise ValueError("Only 2D grids are supported.")
    if grid.field_shape != field.shape:
        raise ValueError("Dimension of grid-field and field do not match.")
    if field.dtype != weights.dtype:
        raise ValueError("Field and weights must be of the same type.")

    num, dnum = _set_field2(grid.grid_sampling_in_pixels, grid.knots, field, weights)
    _set_field_using_num_and_dnum(grid.knots.ravel(), num.ravel(), dnum.ravel())


def set_field_sparse(grid, pp, values):
    """set_field_sparse(grid, pp, values)

    Set the 2D grid by providing the field values at a set of points (which
    are in world coordinates).

    """
    assert isinstance(pp, np.ndarray) and pp.ndim == 2

    if grid.ndim != 2:
        raise ValueError("Only 2D grids are supported.")
    if pp.shape[1] != 2:
        raise ValueError("Pointset must have 2 columns for 2D grid.")

    num, dnum = _set_field_sparse2(grid.grid_sampling, grid.knots, pp, values)
    _set_field_using_num_and_dnum(grid.knots.ravel(), num.ravel(), dnum.ravel())


## Workhorse functions to set the field


@numba.jit(nopython=True, nogil=True)
def _set_field2(grid_sampling_in_pixels, knots, field, weights):
    ccy = np.empty((4,), np.float32)
    ccx = np.empty((4,), np.float32)

    grid_ySpacing = grid_sampling_in_pixels[0]
    grid_xSpacing = grid_sampling_in_pixels[1]

    # Create temporary arrays the same size as the grid
    num = np.zeros_like(knots)
    dnum = np.zeros_like(knots)

    # For each pixel ...
    for y in range(field.shape[0]):
        for x in range(field.shape[1]):
            # Get val and alpha
            val = field[y, x]
            weight = weights[y, x]

            # Evaluate this one?
            if weight <= 0.0:
                continue

            # Calculate what is the leftmost (reference) knot on the grid,
            # and the ratio between closest and second closest knot.
            # Note the +1 to correct for padding.
            tmp = y / grid_ySpacing + 1
            gy = int(tmp)
            ty = tmp - gy
            tmp = x / grid_xSpacing + 1
            gx = int(tmp)
            tx = tmp - gx

            # Get coefficients
            compute_basis_coefficients(ty, ccy)
            compute_basis_coefficients(tx, ccx)

            # Pre-normalize value
            omsum = 0.0
            for j in range(4):
                for i in range(4):
                    omsum += ccy[j] * ccx[i] * ccy[j] * ccx[i]
            val_n = val / omsum

            # For each knot that this point influences
            # Following Lee et al. we update a numerator and a denumerator for
            # each knot.
            for j in range(4):
                jj = j + gy - 1
                for i in range(4):
                    ii = i + gx - 1
                    omega = ccy[j] * ccx[i]
                    omega2 = weight * omega * omega
                    num[jj, ii] += omega2 * (val_n * omega)
                    dnum[jj, ii] += omega2

    # Done
    return num, dnum


@numba.jit(nopython=True, nogil=True)
def _set_field_sparse2(grid_sampling, knots, pp, values):
    ccy = np.empty((4,), np.float32)
    ccx = np.empty((4,), np.float32)

    wGridSampling = grid_sampling

    # Create num, dnum
    num = np.zeros_like(knots)
    dnum = np.zeros_like(knots)

    # For each point ...
    for p in range(pp.shape[0]):
        # Get wx and wy
        wx = pp[p, 0]
        wy = pp[p, 1]

        # Calculate which is the closest point on the lattice to the top-left
        # corner and find ratio's of influence between lattice point.
        tmp = wy / wGridSampling + 1
        gy = int(tmp)
        ty = tmp - gy
        tmp = wx / wGridSampling + 1
        gx = int(tmp)
        tx = tmp - gx

        # Get coefficients
        compute_basis_coefficients(ty, ccy)
        compute_basis_coefficients(tx, ccx)

        # Precalculate omsum (denominator of eq 4 in Lee 1996)
        omsum = 0.0
        for j in range(4):
            for i in range(4):
                omsum += ccy[j] * ccx[i] * ccy[j] * ccx[i]

        # Get val
        val = values[p]

        # For each knot that this point influences
        # Following Lee et al. we update a numerator and a denumerator for
        # each knot.
        jj = gy - 1  # y-location of first knot
        for j in range(4):
            ii = gx - 1  # x-location of first knot
            for i in range(4):
                omega = ccy[j] * ccx[i]
                omega2 = omega * omega
                num[jj, ii] += omega2 * (val * omega / omsum)
                dnum[jj, ii] += omega2
                ii += 1
            jj += 1

    # Done
    return num, dnum


