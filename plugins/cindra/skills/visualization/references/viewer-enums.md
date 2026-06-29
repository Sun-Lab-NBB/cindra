# Viewer enum value reference

State fields report the lowercase enum value (e.g. `maximum_projection`), while the on-screen dropdowns
show a title-case label (e.g. "Maximum Projection"). When telling the user which control to operate,
translate the state value to its dropdown label. This applies to `background_view`, `roi_color_mode`,
`mask_layer`, and `coordinate_space`. This reference is loaded on demand by `/visualization`.

---

## Background views

Reported in `background_view` state field. Values correspond to the background image behind ROI
overlays.

| Value                  | Description                                               |
|------------------------|-----------------------------------------------------------|
| `rois_only`            | Blank background with ROI overlays only                   |
| `mean_image`           | Temporal mean image (channel 1 or 2 based on toggle)      |
| `enhanced_mean_image`  | High-pass filtered mean image                             |
| `correlation_map`      | Pixel-wise activity correlation map                       |
| `maximum_projection`   | Maximum intensity projection                              |
| `corrected_structural` | Bleed-through-corrected structural channel (dual-channel) |

---

## ROI color modes

Reported in `roi_color_mode` state field. Values correspond to the statistic used to color ROI
overlays.

| Value                        | Description                                                                         |
|------------------------------|-------------------------------------------------------------------------------------|
| `random`                     | Random color per ROI from active colormap                                           |
| `skewness`                   | Fluorescence skewness                                                               |
| `compactness`                | Circularity of spatial footprint                                                    |
| `footprint`                  | Spatial detection scale (hop size) used during sparse detection                     |
| `aspect_ratio`               | Bounding ellipse aspect ratio                                                       |
| `solidity`                   | Soma-to-convex-hull area ratio                                                      |
| `colocalization_probability` | Channel 2 colocalization probability                                                |
| `recording_count`            | Number of recordings the ROI was tracked across                                     |
| `cell_probability`           | Classifier cell-probability gradient                                                |
| `correlations`               | Pairwise activity correlation with selected ROI                                     |
| `cell_classification`        | Binary cell/non-cell label (non-cell = colormap low endpoint, cell = high endpoint) |

---

## Mask layers

Reported in `mask_layer` state field (tracking viewer only).

| Value      | Description                                                         |
|------------|---------------------------------------------------------------------|
| `original` | Original ROI masks from single-recording extraction (native coords) |
| `deformed` | Original masks warped to shared cross-recording coordinate space    |
| `template` | Consensus template masks from cross-recording clustering            |
| `tracked`  | Template masks backward-deformed to each recording's native coords  |

---

## Coordinate spaces

Reported in `coordinate_space` state field (tracking viewer only).

| Value         | Description                                                    |
|---------------|----------------------------------------------------------------|
| `native`      | Original recording coordinate space                            |
| `transformed` | Warped to align with cross-recording template coordinate space |
