# Viewer enum value reference

State fields report the lowercase enum value (e.g. `maximum_projection`), while the on-screen dropdowns show a
display label that is not always the title-cased value (e.g. `maximum_projection` shows as "Maximum Projection",
`rois_only` shows as "ROIs"). When telling the user which control to operate, read the exact string from the
Dropdown label column of the matching table below. This applies to `background_view`, `roi_color_mode`,
`mask_layer`, and `coordinate_space`. This reference is loaded on demand by `/visualization`.

---

## Background views

Reported in `background_view` state field. Values correspond to the background image behind ROI overlays. The ROI
viewer offers all six values and hides `corrected_structural` when the recording carries no colocalization data.
The tracking viewer offers `rois_only`, `mean_image`, `enhanced_mean_image`, `maximum_projection`, and
`correlation_map`.

| Value                  | Dropdown label        | Description                                               |
|------------------------|-----------------------|-----------------------------------------------------------|
| `rois_only`            | ROIs                  | Blank background with ROI overlays only                   |
| `mean_image`           | Mean Image            | Temporal mean image (channel 1 or 2 based on toggle)      |
| `enhanced_mean_image`  | Mean Image (Enhanced) | High-pass filtered mean image                             |
| `correlation_map`      | Correlation Map       | Pixel-wise activity correlation map                       |
| `maximum_projection`   | Maximum Projection    | Maximum intensity projection                              |
| `corrected_structural` | Corrected Structural  | Bleed-through-corrected structural channel (dual-channel) |

---

## ROI color modes

Reported in `roi_color_mode` state field. Values correspond to the statistic used to color ROI overlays. The ROI
viewer shows `colocalization_probability` for dual-channel recordings and `recording_count` for multi-recording
tracked ROIs.

| Value                        | Dropdown label       | Description                                                 |
|------------------------------|----------------------|-------------------------------------------------------------|
| `random`                     | Random               | Random color per ROI from active colormap                   |
| `skewness`                   | Skewness             | Fluorescence skewness                                       |
| `compactness`                | Compactness          | Circularity of spatial footprint                            |
| `footprint`                  | Footprint            | Spatial detection scale (hop size) used in sparse detection |
| `aspect_ratio`               | Aspect Ratio         | Bounding ellipse aspect ratio                               |
| `solidity`                   | Solidity             | Soma-to-convex-hull area ratio                              |
| `colocalization_probability` | Colocalization       | Channel 2 colocalization probability                        |
| `recording_count`            | Recording Count      | Number of recordings the ROI was tracked across             |
| `cell_probability`           | Cell Probability     | Classifier cell-probability gradient                        |
| `correlations`               | Activity Correlation | Pairwise activity correlation with selected ROI             |
| `cell_classification`        | Classification       | Binary cell/non-cell label mapped to colormap endpoints     |

---

## Mask layers

Reported in `mask_layer` state field (tracking viewer only).

| Value      | Dropdown label | Description                                                         |
|------------|----------------|---------------------------------------------------------------------|
| `original` | Original       | Original ROI masks from single-recording extraction (native coords) |
| `deformed` | Deformed       | Original masks warped to shared cross-recording coordinate space    |
| `template` | Template       | Consensus template masks from cross-recording clustering            |
| `tracked`  | Tracked        | Template masks backward-deformed to each recording's native coords  |

---

## Coordinate spaces

Reported in `coordinate_space` state field (tracking viewer only).

| Value         | Dropdown label | Description                                                    |
|---------------|----------------|----------------------------------------------------------------|
| `native`      | Native         | Original recording coordinate space                            |
| `transformed` | Transformed    | Warped to align with cross-recording template coordinate space |
