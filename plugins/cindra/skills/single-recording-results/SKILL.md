---
name: single-recording-results
description: >-
  Complete reference for single-recording pipeline output data formats. Documents every file, directory, array shape,
  dtype, and NPZ key produced by the pipeline, plus verification checklists for output completeness. Use when evaluating
  single-recording processing results or when the user asks about single-recording output data.
user-invocable: true
---

# Single-recording results data reference

Complete output data format documentation for the single-recording (within-recording) cindra processing pipeline.

---

## Scope

**Covers:**
- Complete output data reference: every file, directory, array shape, dtype, and NPZ key produced by the pipeline
- Directory structure for combined and per-plane results
- Processing phase and file creation timeline
- Data type conventions and memory-mapping guidance
- Multi-recording compatibility requirements
- Output completeness verification

**Does not cover:**
- Configuration parameters (see `/single-recording-configuration`)
- Input data format, TIFF requirements, and acquisition parameters (see `/acquisition-data-preparation`)
- Processing workflow, batch operations, or status monitoring (see `/single-recording-processing`)
- Multi-recording output data formats (see `/multi-recording-results`)

---

## Agent requirements

You MUST use the cindra MCP query and verification tools to inspect output data rather than reading output files
directly when a tool exists for the task. If MCP tools are not available, invoke `/cindra-mcp-environment-setup` to
diagnose and resolve connectivity issues.

---

## Available tools

Use these cindra MCP tools to query and verify single-recording output data programmatically. Prefer these over manual
file reads whenever possible.

### Verification tool

| Tool                                  | Purpose                                                         |
|---------------------------------------|-----------------------------------------------------------------|
| `verify_single_recording_output_tool` | Verifies completeness of all expected output files and NPZ keys |

### Query tools

| Tool                                   | Purpose                                                                               |
|----------------------------------------|---------------------------------------------------------------------------------------|
| `query_single_recording_metadata_tool` | Queries recording dimensions, frame count, sampling rate, ROI/cell counts, and timing |
| `query_detection_summary_tool`         | Queries detection image intensity statistics and estimated ROI diameter               |
| `query_registration_quality_tool`      | Queries per-plane registration offset summaries, correlations, bad frames, PC metrics |
| `query_roi_statistics_tool`            | Queries per-ROI spatial statistics and classification with sorting and filtering      |
| `query_traces_tool`                    | Queries fluorescence trace arrays for specific ROIs with optional downsampling        |

### Recommended query order

1. `query_single_recording_metadata_tool`: understand recording properties and processing status
2. `query_registration_quality_tool`: assess motion correction quality per plane
3. `query_detection_summary_tool`: review detection image quality and ROI diameter
4. `query_roi_statistics_tool`: inspect ROI quality metrics and classification
5. `query_traces_tool`: examine fluorescence activity for specific ROIs

### Query tool argument semantics

The `recording_path` argument for the verify and query tools must be the recording output directory, the parent of the
`cindra/` folder. This equals the `recording_output_paths` entries passed to and returned by the prepare tool when the
output root differs from the raw-data root, not the raw-data path itself. The tools resolve the `cindra/` subdirectory
automatically.

`query_single_recording_metadata_tool` reports the top-level `frame_count` from the combined traces and each
`plane_timing` entry's `frame_count` from that plane's `runtime_data.yaml`. A plane count above the top-level count is
expected rather than corruption: it marks a plane whose trailing frames were trimmed out of the combined traces.

The ROI indices accepted by `query_traces_tool` and `query_roi_statistics_tool` are 0-based positional row indices into
the per-recording arrays, not a tracking identity. Both tools silently drop individual out-of-range indices, so always
compare the returned `roi_index` values against what you requested. When every requested index is out of range,
`query_roi_statistics_tool` returns an empty `rois` list with `success=true`, while `query_traces_tool` fails with "No
valid ROI indices provided", so a confidently "successful" empty result can only come from the statistics tool.

---

## Output data reference

All results are saved under `{output_path}/cindra/`. The pipeline produces combined (multi-plane merged) data at the
root level and per-plane data in numbered subdirectories. Channel 2 output depends on whether the second channel is
functional, meaning both `main.first_channel_functional` and `main.second_channel_functional` are True. Every
two-channel recording produces, in each `plane_N/` directory, `channel_2_data.bin`,
`detection_data/mean_image_channel_2.npy`, `cell_fluorescence_channel_2.npy`, and `neuropil_fluorescence_channel_2.npy`,
plus the combined `detection_data/mean_image_channel_2.npy` at the root. A structural (non-functional) channel 2 still
gets a full fluorescence extraction pass, but it borrows the channel 1 ROI masks instead of detecting its own ROIs, so
its traces carry one row per channel 1 ROI and come with no independent detection, classification, or deconvolution.
Every other channel 2 file requires both channels to be functional: the other three detection images at both levels, the
ROI `.npz` files, `subtracted_fluorescence_channel_2.npy`, `spikes_channel_2.npy`, `cell_classification_channel_2.npy`,
and the root-level combined `cell_fluorescence_channel_2.npy` and `neuropil_fluorescence_channel_2.npy`. Those last two
are the asymmetry to watch, because combination omits them for a structural channel 2 even though every plane directory
holds its own copy.

### Directory structure

```text
cindra/
├── configuration.yaml                          # Saved pipeline configuration
├── acquisition_parameters.yaml                 # Saved acquisition metadata
├── combined_metadata.npz                       # Combined multi-plane metadata
├── detection_data/                             # Combined detection images
│   ├── mean_image.npy
│   ├── enhanced_mean_image.npy
│   ├── maximum_projection.npy
│   └── correlation_map.npy
├── roi_masks.npz                               # Combined ROI spatial data
├── roi_statistics.npz                          # Combined ROI shape statistics
├── cell_fluorescence.npy                       # Combined fluorescence traces
├── neuropil_fluorescence.npy
├── subtracted_fluorescence.npy
├── spikes.npy
├── cell_classification.npy
├── plane_0/                                    # Per-plane processing results
│   ├── runtime_data.yaml                       # Plane runtime metadata
│   ├── channel_1_data.bin                      # Registered binary data
│   ├── channel_1_data.bin.registering          # Present only while registration rewrites the binary
│   ├── registration_data/                      # Registration arrays
│   │   ├── reference_image.npy
│   │   ├── bad_frames.npy
│   │   ├── rigid_y_offsets.npy
│   │   ├── rigid_x_offsets.npy
│   │   ├── rigid_correlations.npy
│   │   ├── nonrigid_y_offsets.npy
│   │   ├── nonrigid_x_offsets.npy
│   │   ├── nonrigid_correlations.npy
│   │   ├── principal_component_extreme_images.npy
│   │   ├── principal_component_projections.npy
│   │   └── principal_component_shift_metrics.npy
│   ├── detection_data/                         # Plane detection images
│   │   ├── mean_image.npy
│   │   ├── enhanced_mean_image.npy
│   │   ├── maximum_projection.npy
│   │   └── correlation_map.npy
│   ├── roi_masks.npz                           # Plane ROI spatial data
│   ├── roi_statistics.npz                      # Plane ROI shape statistics
│   ├── cell_fluorescence.npy
│   ├── neuropil_fluorescence.npy
│   ├── subtracted_fluorescence.npy
│   ├── spikes.npy
│   └── cell_classification.npy
├── plane_1/
└── ...
```

### Processing phase and file creation timeline

**Phase 1 (binarization):** `configuration.yaml`, `acquisition_parameters.yaml`, and the initial per-plane
`runtime_data.yaml` are already on disk before this phase runs, because `prepare_single_recording_batch_tool` (or the
`cindra run` entry point) writes them while resolving the plane contexts, so their presence does not indicate that
binarization ran. Phase 1 itself creates the per-plane `channel_1_data.bin` (and `channel_2_data.bin` if two-channel)
and per-plane `detection_data/mean_image.npy` (plus `mean_image_channel_2.npy` if two-channel), then records
`binarization_time` into each plane's `runtime_data.yaml`. Each plane binary is sized by that plane's own interleave
frame count, so a recording whose acquisition stopped partway through a volume gives its leading planes one frame more
than its trailing planes, and channel 2 may hold one frame more or fewer than channel 1 of the same plane. Binarization
also rebuilds an existing binary whose size disagrees with its recorded plane geometry, or that an interrupted
registration left marked, without requiring `repeat_binarization`.

**Phase 2 (registration, per-plane):** Creates `registration_data/`, rewrites the plane binary in place, refreshes
`detection_data/mean_image.npy`, and updates `runtime_data.yaml` with the registration section,
`total_registration_time`, and `registration_workers`. For the duration of the in-place rewrite, a
`{binary}.registering` marker sits beside the binary. A marker left on disk means the registration was interrupted, so
the binary holds corrected frames up to an unknown point and raw frames after it. Registration refuses to run against a
marked binary, and re-running binarization rebuilds the binary and clears the marker.

**Phase 3 (processing, per-plane):** Creates the remaining `detection_data/` images (`enhanced_mean_image.npy`,
`maximum_projection.npy`, `correlation_map.npy`), the ROI `.npz` files, the fluorescence `.npy` traces, and updates
`runtime_data.yaml` with `total_processing_time`, `processing_workers`, and `date_processed`. Detection also overwrites
`detection_data/mean_image.npy` with the mean of the temporally binned frames, which drop the bad frames and are cropped
to the registration valid range before being embedded into a full-frame array that is zero outside that range, so the
surviving file is not the whole-movie temporal mean registration wrote.

**Phase 4 (combination):** Creates combined `detection_data/` and the combined ROI and trace files at the root level by
merging all per-plane results, then writes `combined_metadata.npz` last, publishing it through an atomic write that
moving it into place. The metadata file therefore doubles as an atomic completion marker: it never exists while the
payload it describes is missing or partially written.

For every file, array shape, dtype, NPZ key, and data type convention the pipeline produces, see
[references/output-formats.md](references/output-formats.md).

---

## Multi-recording compatibility requirements

For recordings intended for multi-recording processing, single-recording processing must complete all four phases
(binarization, registration, processing, combination). No special configuration is required. For the authoritative list
of the single-recording outputs the multi-recording pipeline consumes, see `/multi-recording-configuration`
(Prerequisites from single-recording processing).

---

## Related skills

| Skill                             | Relationship                                                               |
|-----------------------------------|----------------------------------------------------------------------------|
| `/cindra-pipeline`                | Overview: end-to-end phases, handoffs, and the single-vs-multi entry point |
| `/cindra-mcp-environment-setup`   | Prerequisite: cindra MCP server for query and verification tools           |
| `/acquisition-data-preparation`   | Upstream: input data, TIFF requirements, and acquisition parameters        |
| `/single-recording-configuration` | Configuration parameter reference for the single-recording pipeline        |
| `/single-recording-processing`    | Processing workflow that produces this output                              |
| `/multi-recording-results`        | Companion output data reference for the multi-recording pipeline           |
| `/multi-recording-configuration`  | Multi-recording configuration requires these outputs as prerequisites      |
| `/visualization`                  | Launch viewers and query tools to visualize and inspect this output data   |

---

## Verification checklist

Use `verify_single_recording_output_tool` to automate this verification. The tool checks all expected files and NPZ keys
and returns a completeness verdict with any missing items listed. Fall back to the manual checklist below only if the
MCP tool is unavailable. Replace N with the expected plane count from the acquisition parameters.

```text
Single-Recording Output Completeness:
Root-level files:
- [ ] `configuration.yaml` exists
- [ ] `acquisition_parameters.yaml` exists
- [ ] `combined_metadata.npz` exists and contains `plane_count`, `frame_count`, `plane_frame_counts`,
      `combined_height`, `combined_width` keys (note that `verify_single_recording_output_tool` checks only the
      pre-existing keys, so the two frame-count keys are verified here rather than by that tool)

Combined detection images (cindra/detection_data/):
- [ ] `mean_image.npy` exists
- [ ] `enhanced_mean_image.npy` exists
- [ ] `maximum_projection.npy` exists
- [ ] `correlation_map.npy` exists
- [ ] `mean_image_channel_2.npy` exists if `main.two_channels` is True, whether or not channel 2 is functional
- [ ] `enhanced_mean_image_channel_2.npy`, `maximum_projection_channel_2.npy`, and `correlation_map_channel_2.npy`
      exist if both channels are functional

Combined extraction data (cindra/):
- [ ] `roi_masks.npz` exists and contains `pixel_counts`, `y_pixels`, `x_pixels`, `pixel_weights` keys
- [ ] `roi_statistics.npz` exists and contains `footprints`, `compactness`, `plane_index` keys
- [ ] `cell_fluorescence.npy` exists with shape (num_rois, frames)
- [ ] `neuropil_fluorescence.npy` exists with shape matching cell_fluorescence
- [ ] `subtracted_fluorescence.npy` exists with shape matching cell_fluorescence
- [ ] `spikes.npy` exists with shape matching cell_fluorescence (zero-filled when
      spike_deconvolution.extract_spikes is False)
- [ ] `cell_classification.npy` exists with shape (num_rois, 2)
- [ ] Every channel 2 trace and classification file exists if both channels are functional. This includes
      `cell_fluorescence_channel_2.npy` and `neuropil_fluorescence_channel_2.npy`, which combination omits at this
      level for a structural channel 2 even though each plane directory holds them

Per-plane directories (cindra/plane_0/ through cindra/plane_{N-1}/):
- [ ] Each expected plane directory exists
- [ ] Each plane contains `runtime_data.yaml` with non-zero `io.frame_count` and `io.sampling_rate`
- [ ] Each plane contains `channel_1_data.bin` (registered binary)
- [ ] Each plane contains `channel_2_data.bin` if `main.two_channels` is True

Per-plane registration data (plane_N/registration_data/):
- [ ] `reference_image.npy` exists
- [ ] `bad_frames.npy` exists
- [ ] `rigid_y_offsets.npy` and `rigid_x_offsets.npy` exist
- [ ] `rigid_correlations.npy` exists
- [ ] Nonrigid arrays exist if nonrigid_registration.enabled is True
- [ ] PC metric arrays exist if registration.registration_metric_principal_components > 0 and the plane holds at
      least 1500 frames

Per-plane detection and extraction data (plane_N/):
- [ ] `detection_data/` contains the four channel-1 images: `mean_image.npy`, `enhanced_mean_image.npy`,
      `maximum_projection.npy`, and `correlation_map.npy`
- [ ] `roi_masks.npz` and `roi_statistics.npz` exist
- [ ] Fluorescence trace .npy files exist with consistent shapes across all traces
- [ ] `cell_classification.npy` exists with shape (num_rois, 2)
- [ ] `detection_data/mean_image_channel_2.npy`, `cell_fluorescence_channel_2.npy`, and
      `neuropil_fluorescence_channel_2.npy` exist if `main.two_channels` is True, whether or not channel 2 is
      functional. The two trace files carry one row per channel 1 ROI when channel 2 is structural
- [ ] The remaining channel 2 detection images, ROI `.npz` files, trace files, and `cell_classification_channel_2.npy`
      exist if both channels are functional

Multi-recording readiness (if multi-recording processing is planned):
- [ ] `combined_metadata.npz` contains `registered_binary_paths` key
- [ ] All registered binary files referenced in `registered_binary_paths` exist on disk
- [ ] `combined_metadata.npz` `plane_frame_counts` entries are all equal, or differ only within the tolerance the
      combined view applies (multi-recording extraction opens the plane binaries as one combined view whose frame count
      is that of the shortest plane, so unequal counts mean the trailing frames of the longer planes are not extracted)
```
