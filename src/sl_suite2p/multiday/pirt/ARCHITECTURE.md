# PIRT Package Architecture Reference

This document provides a comprehensive reference for all functions in the pirt (Python Image Registration Toolkit) package and how they interact with each other and the multiday code.

> **Note**: This package has been trimmed to include only the functions used by the sl-suite2p multiday pipeline. Unused functions from the original pirt library have been removed to simplify maintenance.

## Package Structure

```
pirt/
├── __init__.py          → Exports: Deformation, DiffeomorphicDemonsRegistration, RegistrationParameters
├── pyramid.py           → Scale-space pyramid for multi-resolution processing
├── spline_grid.py       → B-spline grid for diffeomorphic regularization + spline coefficient functions
├── deformation.py       → Deformation class + image transformation functions + diffusion filtering
└── registration.py      → RegistrationParameters dataclass + DiffeomorphicDemonsRegistration class
```

---

## Module-by-Module Function Reference

### `pyramid.py`

| Class/Method | Purpose | Used By |
|---|---|---|
| `ScaleSpacePyramid.__init__(data, min_scale, scale_offset, use_buffer, level_factor)` | Initializes pyramid with optional smoothing | `DiffeomorphicDemonsRegistration.register()` |
| `ScaleSpacePyramid._initialize_level0(data, min_scale, scale_offset)` | Smooths input to min_scale, downsamples if appropriate | Constructor |
| `ScaleSpacePyramid.get_scale(scale)` | Gets image at specific scale (with smoothing) | `DiffeomorphicDemonsRegistration._get_deformed_image()` |
| `ScaleSpacePyramid._add_Level()` | Adds next pyramid level (smooth + downsample) | `get_scale()` |

### `spline_grid.py`

| Class/Function | Purpose | Used By |
|---|---|---|
| `compute_cardinal_coefficients(t, out, tension)` | Numba: Cardinal spline coefficients for image interpolation | `_warp2()` |
| `compute_basis_coefficients(t, out)` | Numba: cubic B-spline coefficients for grid operations | `_get_field2()`, `_set_field2()`, `SplineGrid._freeze_edges()` |
| `SplineGrid.__init__(field_shape, sampling)` | Creates B-spline grid for given field shape and knot spacing | `Deformation.regularize()` |
| `SplineGrid.field_shape` | Shape of the underlying field (height, width) | Various |
| `SplineGrid.grid_shape` | Shape of the knot grid | Various |
| `SplineGrid.grid_sampling` | Spacing between knots in pixels | Various |
| `SplineGrid.get_fields()` | Samples grid to produce dense deformation field arrays | `Deformation.regularize()` |
| `SplineGrid.set_from_fields(fields, weights, injective, frozenedge)` | Sets knots from field arrays with diffeomorphic constraints | `Deformation.regularize()` |
| `SplineGrid.compute_grid_shape(field_shape, grid_sampling)` | Static: computes grid shape without creating instance | `DiffeomorphicDemonsRegistration._compute_groupwise_deform()` |
| `SplineGrid._unfold(factor)` | Prevents grid folding (injectivity constraint) | `set_from_fields()` |
| `SplineGrid._freeze_edges()` | Freezes edges to zero deformation | `set_from_fields()` |
| `_get_field2(result, grid_sampling, knots)` | Numba: 2D grid sampling kernel | `SplineGrid.get_fields()` |
| `_set_field_using_num_and_dnum(knots, num, dnum)` | Numba: divides numerator by denominator | `SplineGrid.set_from_fields()` |
| `_set_field2(grid_sampling, knots, field, weights)` | Numba: computes num/dnum for grid setting | `SplineGrid.set_from_fields()` |

### `deformation.py`

#### Diffusion Filtering

| Function | Purpose | Used By |
|---|---|---|
| `diffusionkernel(sigma, N, returnt)` | Creates discrete diffusion kernel (Bessel functions) | `diffuse()` |
| **`diffuse(L, sigma, mode)`** | True discrete diffusion filtering | **`ScaleSpacePyramid`**, `resize()` |

#### Transformation Functions

| Function | Purpose | Used By |
|---|---|---|
| **`make_samples_absolute(samples)`** | Converts relative deltas to absolute pixel coordinates | **`Deformation.get_deformation_locations()`, `deform_backward()`, `deform_forward()`** |
| `_floor(i)` | Numba floor handling negatives | `_warp2()`, `_project2()` |
| **`warp(data, samples, order, tension)`** | Backward warping using Cardinal spline interpolation | **`Deformation._compose_forward()`, `deform_backward()`, `resize()`** |
| `_warp2(data, result, samples_x, samples_y, order, tension)` | Numba: 2D backward warping kernel | `warp()` |
| `project(data, samples)` | Forward splatting (move pixels to targets) | `deform_forward()` |
| `_project2(data, result, deformx, deformy)` | Numba: 2D forward projection kernel | `project()` |
| **`deform_backward(data, deltas, order, tension)`** | Backward deformation (primary method) | **`Deformation.apply_deformation()`** |
| `deform_forward(data, deltas)` | Forward deformation (splatting) | `Deformation.inverse()` |
| **`resize(data, new_shape, order, tension, prefilter, extra)`** | Resize array to new shape | **`Deformation.resize_field()`** |
| **`zoom(data, factor, order, tension, prefilter, extra)`** | Scale array by factor | **`ScaleSpacePyramid._initialize_level0()`, `ScaleSpacePyramid._add_Level()`** |

#### Deformation Class

| Class/Method | Purpose | Used By |
|---|---|---|
| **`Deformation.__init__(*fields)`** | Creates deformation (identity, from ndim, from shape tuple, or from field arrays) | **Throughout multiday** |
| `Deformation.is_identity` | Whether this is null/identity deformation | Many methods |
| `Deformation.ndim` | Number of dimensions | Many methods |
| `Deformation.field_shape` | Field dimensions | Many methods |
| `Deformation.copy()` | Deep copy | `add()`, `compose()` |
| `Deformation.scale(factor)` | Scales deformation magnitude | `DiffeomorphicDemonsRegistration._compute_groupwise_deform()` |
| `Deformation.add(other)` | Combines by addition | `DiffeomorphicDemonsRegistration._compute_groupwise_deform()` |
| `Deformation.compose(other)` | Combines by composition | `DiffeomorphicDemonsRegistration._apply_delta_deform()` |
| `Deformation._compose_forward(other)` | Forward composition implementation | `_compose_backward()` |
| `Deformation._compose_backward(other)` | Backward composition (delegates to forward) | `compose()` |
| **`Deformation.resize_field(new_shape)`** | Resizes field to new shape | **`DiffeomorphicDemonsRegistration._apply_delta_deform()`, `_get_deformed_image()`** |
| `Deformation.get_field(d)` | Gets field for dimension d | `add()`, `_compose_forward()` |
| `Deformation.get_deformation_locations()` | Gets absolute sample locations (x-y-z order) | `_compose_forward()` |
| **`Deformation.apply_deformation(data, interpolation)`** | Applies deformation to image data | **`transform.py`, `utils.py:deform_masks()`** |
| `Deformation.inverse()` | Computes inverse deformation | Internal |
| **`Deformation.regularize(grid_sampling, weights, injective, frozenedge)`** | Regularizes deformation using B-spline grid constraints | **`DiffeomorphicDemonsRegistration._regularize_diffeomorphic()`** |

### `registration.py`

| Class/Method | Purpose | Used By |
|---|---|---|
| `RegistrationParameters` | Dataclass with registration parameters | `DiffeomorphicDemonsRegistration` |
| **`DiffeomorphicDemonsRegistration.__init__(*images)`** | Creates registration object with images | **`transform.py:register_sessions()`** |
| **`DiffeomorphicDemonsRegistration.params`** | Gets RegistrationParameters object for configuration | **`transform.py:register_sessions()`** |
| **`DiffeomorphicDemonsRegistration.register(verbose)`** | Runs the registration process | **`transform.py:register_sessions()`** |
| **`DiffeomorphicDemonsRegistration.get_deform(i)`** | Gets deformation for image i | **`transform.py`** |
| `DiffeomorphicDemonsRegistration._register_iteration(level, iter, scale)` | One iteration at specified scale | `register()` |
| `DiffeomorphicDemonsRegistration._compute_groupwise_deform(i, iter_info)` | Averages deforms from all other images | `_register_iteration()` |
| `DiffeomorphicDemonsRegistration._compute_demons_deform(i, j, iter_info)` | Computes demons deformation between image pair | `_compute_groupwise_deform()` |
| `DiffeomorphicDemonsRegistration._get_image_and_gradient(image_id, iter_info)` | Gets image and its gradient | `_compute_demons_deform()` |
| `DiffeomorphicDemonsRegistration._get_deformed_image(i, scale)` | Gets deformed image at scale | `_get_image_and_gradient()` |
| `DiffeomorphicDemonsRegistration._apply_delta_deform(i, deform)` | Combines delta with current deform | `_register_iteration()` |
| `DiffeomorphicDemonsRegistration._regularize_diffeomorphic(scale, deform)` | Regularizes deformation via `deform.regularize()` | `_compute_demons_deform()` |
| `DiffeomorphicDemonsRegistration._get_grid_sampling(scale)` | Computes B-spline grid sampling at scale | `_regularize_diffeomorphic()`, `_compute_groupwise_deform()` |

---

## Multiday Code <-> PIRT Interactions

### `transform.py`

```
register_sessions()
├── Creates DiffeomorphicDemonsRegistration(*images)
├── Sets params: grid_sampling_factor, scale_sampling, speed_factor
├── Calls registration.register(verbose=0)
│   └── Triggers full scale-space registration pipeline
└── For each session: _register_session()
    ├── session.deform = registration.get_deform(index)
    ├── session.deform.apply_deformation(image) for each reference image
    └── deform_masks(cell_masks, session.deform)

backward_transform_masks()
└── For each session: _backward_transform_session()
    └── deform_masks(template_masks, session.deform.inverse())
```

### `utils.py`

```
deform_masks(cell_masks, deform)
├── create_cropped_deform_field(deform, origin, size)
│   └── Creates Deformation from sliced field arrays
└── crop_def.apply_deformation(lam_img, interpolation=0)
    └── Uses nearest-neighbor interpolation for mask warping

create_cropped_deform_field(deform, origin, crop_size)
└── Deformation(deform[0][slice], deform[1][slice])
```

---

## Complete Call Graph for Registration

```
DiffeomorphicDemonsRegistration(*images)
└── Stores images, creates _deforms dict, initializes params

registration.register(verbose=0)
├── Creates ScaleSpacePyramid for each image
│   └── _initialize_level0() -> diffuse() + zoom()
└── For each scale level (coarse->fine):
    └── _register_iteration(level, iteration, scale)
        └── For each image i:
            ├── _compute_groupwise_deform(i, iter_info)
            │   └── For each other image j:
            │       └── _compute_demons_deform(i, j, iter_info)
            │           ├── _get_image_and_gradient(i, iter_info)
            │           │   └── _get_deformed_image(i, scale)
            │           │       └── pyramid.get_scale(scale) -> diffuse()
            │           │       └── deform.apply_deformation() -> deform_backward() -> warp()
            │           ├── Computes demons force field (gradient-based)
            │           └── _regularize_diffeomorphic(scale, deformForce)
            │               └── deform.regularize(grid_sampling, ...)
            │                   ├── SplineGrid(field_shape, sampling)
            │                   ├── grid.set_from_fields() -> _unfold() + _freeze_edges()
            │                   └── Deformation(*grid.get_fields())
            └── _apply_delta_deform(i, deform)
                └── current.compose(deform)

registration.get_deform(i)
└── Returns _deforms[i] (Deformation instance)
```

---

## Key Data Flow Summary

1. **Registration Input**: Multiple session reference images
2. **Scale-Space**: `ScaleSpacePyramid` creates multi-resolution versions using `diffuse()` and `zoom()`
3. **Demons Algorithm**: Computes force field from image gradients and differences
4. **Diffeomorphic Constraint**: `Deformation.regularize()` converts raw deformation to B-spline grid with injectivity/edge constraints
5. **Deformation Output**: `Deformation` objects storing backward-mapping offset fields
6. **Application**: `Deformation.apply_deformation()` warps images/masks using `deform_backward()` -> `warp()`
7. **Inverse Transform**: `Deformation.inverse()` for mapping templates back to session space

---

## Key Concepts

### Backward Mapping

All deformations use **backward mapping**: for each output pixel, sample from the corresponding source location. This avoids holes in output and is the standard approach for image registration.

The `deform_forward()` function is retained internally because it's needed for computing inverse deformations via `Deformation.inverse()`.

### Diffeomorphic Deformations

A diffeomorphic deformation is:
1. **Smooth** - No discontinuities
2. **Invertible** - One-to-one mapping
3. **Topology-preserving** - No folding or tearing

Achieved via:
- **B-spline regularization**: `SplineGrid` converts raw deformation to smooth B-spline representation
- **Injectivity constraint**: `_unfold()` limits knot values to prevent grid folding
- **Frozen edges**: `_freeze_edges()` sets boundary deformation to zero

### Spline Types

The package uses two types of splines for different purposes:

- **Cardinal splines** (`compute_cardinal_coefficients`): Used for image interpolation in `warp()`. Pass through control points exactly. The `tension` parameter controls curve tightness (0 = Catmull-Rom, smooth curves).

- **B-splines** (`compute_basis_coefficients`): Used for deformation field regularization in `SplineGrid`. Approximating splines that provide C2 continuity and minimize bending energy.

### Scale-Space Registration

Registration proceeds from coarse to fine scale:
1. Start at high scale (blurry images, large deformations)
2. Progressively decrease scale
3. Refine deformation at each level
4. End at `final_scale` (typically 1.0)

This prevents local minima and handles large deformations.

---

## Default Parameters

### RegistrationParameters

| Parameter | Default | Description |
|---|---|---|
| `speed_factor` | `3.0` | Demons force multiplier (most important tuning parameter) |
| `scale_sampling` | `25` | Iterations per scale level |
| `grid_sampling_factor` | `0.5` | Grid scaling with image scale |
| `final_scale` | `1.0` | Minimum scale (finest resolution) |
| `final_grid_sampling` | `16.0` | B-spline grid spacing at final scale |
| `smooth_scale` | `True` | Use smooth scale transitions |
| `injective` | `True` | Enforce injectivity constraint |
| `frozenedge` | `True` | Freeze edges to zero deformation |
| `deform_limit` | `1.0` | Deformation magnitude limit |
| `noise_factor` | `1.0` | Regularization for noise |

---

## Structural Changes from Original PIRT

The following modules were merged/reorganized to simplify the package structure:

### Merged into `spline_grid.py`
- `interp/spline_coefficients.py` - Spline coefficient computation functions (only Cardinal and B-spline retained)
- `splinegrid/_splinegridclasses.py` - SplineGrid class (merged with SplineGridHelper)
- `splinegrid/_splinegridfuncs.py` - Numba grid sampling functions

### Merged into `deformation.py`
- `deform/deformation.py` - Deformation class
- `interp/transformations.py` - All transformation functions (warp, resize, zoom, deform_backward, deform_forward)
- `gaussfun.py` - Diffusion kernel and filtering functions

### Merged into `registration.py`
- `reg/reg_base.py` - AbstractRegistration, BaseRegistration, GDGRegistration classes
- `reg/reg_demons.py` - BaseDemonsRegistration, DiffeomorphicDemonsRegistration classes

### Removed Classes/Concepts
- `FieldDescription` - Removed; shape tuples used directly
- `SplineGridHelper` - Merged into `SplineGrid`
- `SplineTypes` enum - Removed; Cardinal/B-spline functions called directly
- `forward_mapping` parameter - Removed; always backward mapping
- `field_sampling` - Removed; always 1.0 (unit pixel spacing)
- `grid_sampling_in_pixels` - Removed; redundant with `grid_sampling`

### Simplified Class Hierarchies

**Spline Grid:**
- Single `SplineGrid` class (no inheritance, no helper class)
- Conversion to/from Deformation happens in `Deformation.regularize()`

**Registration:**
- Flattened 5-class hierarchy into single `DiffeomorphicDemonsRegistration` class
- Replaced `Parameters` dict-subclass with `RegistrationParameters` dataclass
- Hardcoded backward mapping and groupwise registration

---

## Removed Functions

The following were removed from the original pirt library as unused by the multiday pipeline:

### gaussfun.py (merged into deformation.py)
- `_gaussiankernel()`, `gaussiankernel()`, `gfilter()` - Gaussian filtering (diffusion used instead)
- `gaussiankernel2()`, `gfilter2()` - 2D variants

### spline_grid.py
- `SplineTypes` enum, `compute_quadratic_coefficients()` - Quadratic splines unused
- `compute_spline_coefficients()`, `set_spline_coefficients()` - Generic wrappers removed
- `FieldDescription` class - Shape tuples used directly
- `SplineGridHelper` class - Merged into SplineGrid
- `SplineGrid.from_deformation()`, `SplineGrid.to_deformation()` - Moved to `Deformation.regularize()`
- Grid copy/refine/add/resize operations - Unused

### deformation.py
- `Deformation.forward_mapping` property - Always backward mapping
- `Deformation.as_backward_inverse()` - Use `inverse()` directly
- `Deformation.as_forward()`, `as_backward()`, `as_other()` - Mapping conversion removed
- `Deformation.get_field_in_points()` - Sparse sampling unused

### registration.py
- `AbstractRegistration.get_final_deform()` - Composed deforms unused
- Pairwise registration modes - Only groupwise used
- Add mode for deformation combination - Only compose used
