"""Unified deformation module for image registration.

This module provides a single `Deformation` class that represents all types of
deformations (field-based, identity) with support for both forward and backward
mapping.
"""

import numpy as np

from .. import interp
from ..splinegrid import FD, SplineGrid, GridContainer


class Deformation:
    """Unified deformation class for 2D image registration.

    A deformation maps one 2D image to another. It can represent:
    - An identity (null) deformation when no fields are provided
    - A field-based deformation when field arrays are provided

    Parameters
    ----------
    *fields : arrays or int or FD
        The deformation fields (one per dimension, in z-y-x order).
        Can also be:
        - No arguments: creates an identity deformation
        - Single int: creates null deformation with specified ndim
        - FD instance: creates null deformation matching the field description
    forward_mapping : bool, optional
        Whether this is a forward mapping (default False = backward mapping).
        Forward mapping: pixels are mapped to new locations.
        Backward mapping: result pixels sample from original locations.
    """

    def __init__(self, *fields, forward_mapping: bool = False):
        self._forward_mapping = forward_mapping

        if len(fields) == 1 and isinstance(fields[0], (list, tuple)):
            fields = fields[0]

        if not fields:
            # Identity deformation (no fields)
            self._field_shape = (1, 1)
            self._field_sampling = (1.0, 1.0)
            self._fields = []

        elif len(fields) == 1 and isinstance(fields[0], int):
            # Null deformation with specified ndim
            ndim = fields[0]
            self._field_shape = tuple([1 for _ in range(ndim)])
            self._field_sampling = tuple([1.0 for _ in range(ndim)])
            self._fields = []

        elif len(fields) == 1 and isinstance(fields[0], FD):
            # Null deformation with known dimensions
            self._field_shape = fields[0].shape
            self._field_sampling = fields[0].sampling
            self._fields = []

        else:
            # Actual field deformation
            if not self._check_fields_same_shape(fields):
                raise ValueError("Fields must all have the same shape.")
            if len(fields) != fields[0].ndim:
                raise ValueError("There must be a field for each dimension.")

            self._field_shape = fields[0].shape
            # Sampling is always 1.0 for all dimensions in multiday usage
            self._field_sampling = tuple(1.0 for _ in range(fields[0].ndim))
            self._fields = list(fields)

    def __repr__(self):
        mapping = ["backward", "forward"][self.forward_mapping]
        if self.is_identity:
            return f"<Deformation ({mapping}) {self.ndim}D identity>"
        shapestr = "x".join([str(s) for s in self.field_shape])
        samplingstr = "x".join([f"{s:1.2f}" for s in self.field_sampling])
        return f"<Deformation ({mapping}) shape {shapestr} sampling {samplingstr}>"

    @staticmethod
    def _check_fields_same_shape(fields):
        """Check whether the given fields all have the same shape."""
        shape = fields[0].shape
        for field in fields:
            if field.shape != shape:
                return False
        return True

    # --- Properties ---

    @property
    def forward_mapping(self) -> bool:
        """Whether this deformation uses forward mapping."""
        return self._forward_mapping

    @property
    def is_identity(self) -> bool:
        """Whether this represents no deformation (identity)."""
        return len(self._fields) == 0

    @property
    def ndim(self) -> int:
        """The number of dimensions of the deformation."""
        return len(self._field_shape)

    @property
    def field_shape(self) -> tuple:
        """The shape of the deformation field."""
        return tuple(self._field_shape)

    @property
    def field_sampling(self) -> tuple:
        """The sampling (pixel spacing) for each dimension."""
        return tuple(self._field_sampling)

    # --- Sequence access ---

    def __len__(self):
        return len(self._fields)

    def __getitem__(self, item):
        if isinstance(item, int):
            if 0 <= item < len(self._fields):
                return self._fields[item]
            raise IndexError("Field index out of range.")
        raise IndexError("Deformation only supports integer indices.")

    def __iter__(self):
        return iter(self._fields)

    # --- Operators ---

    def __add__(self, other):
        return self.add(other)

    def __mul__(self, other):
        if isinstance(other, Deformation):
            return other.compose(self)
        return self.scale(other)

    # --- Core methods ---

    def copy(self):
        """Create a deep copy of this deformation."""
        if self.is_identity:
            return Deformation(FD(self), forward_mapping=self._forward_mapping)
        return self.scale(1.0)

    def scale(self, factor: float):
        """Scale the deformation by the given factor.

        Note that the result is diffeomorphic only if the original is
        diffeomorphic and the factor is between -1 and 1.
        """
        fields = []
        for d in range(self.ndim):
            if factor == 1.0:
                fields.append(self._fields[d].copy())
            else:
                fields.append(self._fields[d] * factor)
        return Deformation(*fields, forward_mapping=self._forward_mapping)

    def add(self, other):
        """Combine two deformations by addition.

        The mapping direction is taken from the left (self) deformation.
        """
        if not isinstance(other, Deformation):
            raise ValueError("Can only combine Deformations.")

        if self.is_identity:
            return other.copy()
        if other.is_identity:
            return self.copy()

        if self.field_shape != other.field_shape:
            raise ValueError("Can only combine deforms with same field shape.")
        if self.forward_mapping != other.forward_mapping:
            raise ValueError("Can only combine deforms with the same mapping.")

        fields = []
        for d in range(self.ndim):
            fields.append(self.get_field(d) + other.get_field(d))
        return Deformation(*fields, forward_mapping=self._forward_mapping)

    def compose(self, other):
        """Combine two deformations by composition.

        The left (self) is the "static" deformation, and the right (other)
        is the "delta" deformation. Returns a new Deformation instance.

        The mapping direction is taken from the left (self) deformation.
        """
        if not isinstance(other, Deformation):
            raise ValueError("Can only combine Deformations.")

        if self.is_identity:
            return other.copy()
        if other.is_identity:
            return self.copy()

        if self.field_shape != other.field_shape:
            raise ValueError("Can only combine deforms with same field shape.")
        if self.forward_mapping != other.forward_mapping:
            raise ValueError("Can only combine deforms with the same mapping.")

        if self.forward_mapping:
            fields = self._compose_forward(other)
        else:
            fields = self._compose_backward(other)
        return Deformation(*fields, forward_mapping=self._forward_mapping)

    def _compose_forward(self, other):
        """Compose for forward mapping: sample in other at locations of self."""
        # Get sample positions in pixel coordinates
        sample_locations = self.get_deformation_locations()

        fields = []
        for d in range(self.ndim):
            field1 = self._fields[d]
            field2 = other._fields[d]
            # Composition with a field introduces interpolation artifacts
            field = interp.warp(field2, sample_locations, "linear")
            fields.append(field1 + field)
        return fields

    def _compose_backward(self, other):
        """Compose for backward mapping: sample in self at locations of other."""
        return other._compose_forward(self)

    def resize_field(self, new_shape):
        """Create a new Deformation with the field resized to match new_shape.

        Parameters
        ----------
        new_shape : array, FD, or Deformation
            The target shape/sampling to resize to.

        Returns:
        -------
        Deformation
            A new deformation with resized fields, or self if already correct size.
        """
        if self.is_identity:
            return Deformation(FD(new_shape), forward_mapping=self._forward_mapping)

        fd1 = FD(self)
        fd2 = FD(new_shape)

        if fd1.shape == fd2.shape and self._sampling_equal(fd1, fd2):
            return self

        return self._resize_field(fd2)

    def _sampling_equal(self, fd1, fd2):
        """Check if two FDs have equal sampling (within tolerance)."""
        sam_errors = [abs(s1 - s2) for s1, s2 in zip(fd1.sampling, fd2.sampling)]
        return max(sam_errors) <= 0.0001 * min(fd1.sampling)

    def _resize_field(self, fd):
        """Resize field to match given FD."""
        fields = []
        for field in self._fields:
            resized = interp.resize(field, fd.shape, 3, "C", prefilter=False, extra=False)
            fields.append(resized)
        return Deformation(*fields, forward_mapping=self._forward_mapping)

    # --- Getting field values ---

    def get_field(self, d: int):
        """Get the field for dimension d."""
        return self._fields[d]

    def get_deformation_locations(self):
        """Get absolute sample locations in pixel coordinates (x-y-z order).

        These locations can be fed directly to interp functions.
        """
        # Reverse fields from z-y-x to x-y-z order
        deltas = [s for s in reversed(self._fields)]
        return interp.make_samples_absolute(deltas)

    def get_field_in_points(self, pp, d: int, interpolation: int = 1):
        """Get field values at specified points.

        Parameters
        ----------
        pp : array
            Point set in x-y-z order, shape (N, ndim).
        d : int
            Dimension to get field for.
        interpolation : int
            Interpolation order.
        """
        assert isinstance(pp, np.ndarray) and pp.ndim == 2
        data = self._fields[d]
        samples = []
        sampling_xyz = [s for s in reversed(self.field_sampling)]
        for i in range(self.ndim):
            s = pp[:, i] / sampling_xyz[i]
            samples.append(s)
        return interp.warp(data, samples, order=interpolation)

    # --- Applying deformation ---

    def apply_deformation(self, data, interpolation: int = 3):
        """Apply the deformation to the given data.

        Parameters
        ----------
        data : array
            The data to deform.
        interpolation : int
            Interpolation order (0, 1, or 3).

        Returns:
        -------
        array
            The deformed data.
        """
        if self.is_identity:
            return data

        # Need upsampling?
        deform = self.resize_field(data)

        # Reverse from z-y-x to x-y-z
        samples = [s for s in reversed(deform._fields)]

        # Deform!
        if self.forward_mapping:
            result = interp.deform_forward(data, samples)
        else:
            result = interp.deform_backward(data, samples, interpolation)

        return result

    # --- Conversion methods ---

    def inverse(self):
        """Get the inverse deformation.

        Only valid if the current deformation is diffeomorphic.
        """
        if self.is_identity:
            return self

        # Get samples
        samples = [s for s in reversed(self._fields)]

        # Get inverse fields
        fields = []
        for field in self._fields:
            fields.append(interp.deform_forward(-field, samples))

        return Deformation(*fields, forward_mapping=self._forward_mapping)

    def as_forward(self):
        """Return as forward mapping deformation.

        If already forward, returns self. Otherwise computes inverse.
        """
        if self.forward_mapping:
            return self
        fields = list(self.inverse())
        return Deformation(*fields, forward_mapping=True)

    def as_backward(self):
        """Return as backward mapping deformation.

        If already backward, returns self. Otherwise computes inverse.
        """
        if not self.forward_mapping:
            return self
        fields = list(self.inverse())
        return Deformation(*fields, forward_mapping=False)

    def as_other(self, other):
        """Return deformation matching the mapping direction of other."""
        if other.forward_mapping:
            return self.as_forward()
        return self.as_backward()

    def as_forward_inverse(self):
        """Return inverse as forward mapping.

        For backward mapping with same data, this is a quick operation.
        For forward mapping, computes the full inverse.
        """
        if self.forward_mapping:
            return self.inverse()
        # Quick: same data wrapped in forward
        return Deformation(*self._fields, forward_mapping=True)

    def as_backward_inverse(self):
        """Return inverse as backward mapping.

        For forward mapping with same data, this is a quick operation.
        For backward mapping, computes the full inverse.
        """
        if not self.forward_mapping:
            return self.inverse()
        # Quick: same data wrapped in backward
        return Deformation(*self._fields, forward_mapping=False)


class SplineGridHelper(GridContainer):
    """Internal B-spline grid for diffeomorphic regularization.

    This class is only used internally by registration algorithms to convert
    deformation fields to B-spline grids, apply diffeomorphic constraints,
    and convert back to Deformation instances.

    Not intended for external use.
    """

    def __init__(self, *args, forward_mapping: bool = False, **kwargs):
        GridContainer.__init__(self, *args, **kwargs)
        self._forward_mapping = forward_mapping

        # Create sub grids
        for d in range(self.ndim):
            grid = SplineGrid(*args, **kwargs)
            grid._thisDim = d
            self._grids.append(grid)

    @classmethod
    def from_deformation(
        cls,
        deform: Deformation,
        sampling: float,
        weights=None,
        injective: bool = True,
        frozenedge: bool = True,
    ):
        """Create a grid from a Deformation with diffeomorphic constraints.

        Parameters
        ----------
        deform : Deformation
            The deformation to convert.
        sampling : float
            The grid sampling (knot spacing).
        weights : array, optional
            Weights for field elements.
        injective : bool
            Whether to prevent grid folding.
        frozenedge : bool
            Whether to freeze edges to zero deformation.

        Returns:
        -------
        _SplineGridHelper
            The grid with constraints applied.
        """
        fd = deform._fields[0] if deform._fields else FD(deform)
        grid = cls(fd, sampling, forward_mapping=deform.forward_mapping)
        grid._set_using_field(deform, weights, injective, frozenedge)
        return grid

    def to_deformation(self) -> Deformation:
        """Convert this grid to a Deformation."""
        fields = [g.get_field() for g in self]
        return Deformation(*fields, forward_mapping=self._forward_mapping)

    def _set_using_field(self, deform, weights=None, injective=True, frozenedge=True):
        """Set the grid from deformation fields with constraints."""
        if deform.is_identity:
            return

        if len(deform) != self.ndim:
            raise ValueError("Deformation must have a field for each dimension.")

        # Apply using SplineGrid's method
        for d in range(self.ndim):
            self[d]._set_using_field(deform[d], weights)

        # Diffeomorphic constraints
        if injective:
            self._unfold(injective)
        if frozenedge:
            self._freeze_edges()

    def _unfold(self, factor):
        """Prevent folds in the grid by limiting knot values.

        Based on Choi, Yongchoel, and Seungyong Lee. 2000. "Injectivity conditions of
        2d and 3d uniform cubic b-spline functions".
        """
        mode = 2
        if factor is False:
            return
        if factor is True:
            factor = 0.9
        elif factor < 0:
            mode = 1
            factor = -factor

        # K factor for 2D B-spline injectivity
        K = 2.046392675

        limit = (1.0 / K) * self.grid_sampling * factor

        for d in range(self.ndim):
            knots = self[d].knots.ravel()

            if mode == 1:
                # Hard limit
                (I,) = np.where(np.abs(knots) > limit)
                knots[I] = limit * np.sign(knots[I])
            elif mode == 2:
                # Smooth limit
                f = np.exp(-np.abs(knots) / limit)
                knots[:] = limit * (f - 1) * -np.sign(knots)

    def _freeze_edges(self):
        """Freeze outer knots to zero so deformation is zero at image edges."""

        def get_t_factor(grid, d):
            field_edge = (grid.field_shape[d] - 1) * grid.field_sampling[d]
            grid_edge = (grid.grid_shape[d] - 4) * grid.grid_sampling
            return 1.0 - (field_edge - grid_edge) / grid.grid_sampling

        for d in range(len(self)):
            grid = self[d]

            # Check if grid is large enough
            if grid._knots.shape[d] < 6:
                grid._knots[:] = 0
                continue

            if d == 0:
                grid._knots[0] = 0
                grid._knots[1] = -0.25 * grid._knots[2]

                t = get_t_factor(grid, d)
                c1, c2, c3, c4 = interp.compute_spline_coefficients(t, interp.SplineTypes.BASIS)

                grid._knots[-3] = (1 - t) * grid._knots[-3]
                grid._knots[-1] = 0
                k3, k4 = grid._knots[-3], grid._knots[-4]
                grid._knots[-2] = -(k3 * c3 + k4 * c4) / c2

            elif d == 1:
                grid._knots[:, 0] = 0
                grid._knots[:, 1] = -0.25 * grid._knots[:, 2]

                t = get_t_factor(grid, d)
                c1, c2, c3, c4 = interp.compute_spline_coefficients(t, interp.SplineTypes.BASIS)

                grid._knots[:, -3] = (1 - t) * grid._knots[:, -3]
                grid._knots[:, -1] = 0
                k3, k4 = grid._knots[:, -3], grid._knots[:, -4]
                grid._knots[:, -2] = -(k3 * c3 + k4 * c4) / c2
