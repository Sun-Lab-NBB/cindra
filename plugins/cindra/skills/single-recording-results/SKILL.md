---
name: single-recording-results
description: >-
  Complete reference for single-recording pipeline output data formats. Documents every file, directory, array shape,
  dtype, and NPZ key produced by the pipeline, plus verification checklists for output completeness. Use when evaluating
  single-recording processing results or when the user asks about single-recording output data.
user-invocable: true
---

# Single-recording results data reference

Documents the output data the single-recording pipeline writes and the MCP tools that verify and query it.

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

### Verification tool

| Tool                                  | Purpose                                                         |
|---------------------------------------|-----------------------------------------------------------------|
| `verify_single_recording_output_tool` | Verifies completeness of all expected output files and NPZ keys |

The verification response reports `success`, `complete`, the echoed `output_root`, `cindra_path` naming the resolved
`cindra/` directory the tool inspected, `plane_count` counted from the plane directories on disk, `two_channels`,
`total_checks`, `passed`, `failed`, `missing`, and `warnings`. `failed` counts the required checks that did not pass, so
it always equals the length of `missing`, and `complete` is False whenever `missing` is non-empty. The `warnings` list
is always present and holds non-fatal issues such as a registered-binary path that does not resolve on disk, so a
response carrying warnings can still report `complete` as True. The `two_channels` flag derives from the channel-2
registered binary paths in `combined_metadata.npz`, so it means channel 2 is present AND functional rather than merely
that `main.two_channels` is True. A recording with a structural channel 2 reports it False, and the tool then runs none
of the optional channel-2 checks even though each plane still holds `detection_data/mean_image_channel_2.npy`. An
`optional_absent` list appears only when it holds entries, carrying the same label form for the optional outputs the
recording does not hold. It is informational and never gates `complete`. The three principal-component registration
arrays land there whenever the recording holds fewer than 1500 frames, which is the threshold below which the
registration metrics are skipped. A recording naming flyback planes also carries `flyback_planes`, whose registration,
projection, and extraction items count as optional.

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

The verify and query tools all name their path argument `output_root`. It must be the pipeline output root, the parent
of the `cindra/` folder, which equals the `output_roots` entries passed to the prepare tool and the per-recording
`output_root` entries it returns. It is never the `raw_data_paths` entry, which is the directory holding the TIFF
files. The tools resolve the `cindra/` subdirectory automatically.

`verify_single_recording_output_tool` and `query_single_recording_metadata_tool` both report `cindra_path`, the resolved
`cindra/` directory they inspected under the `output_root` they were given. It reaches the caller as a response key
rather than as an `output_root`, so keep passing the `output_root` back into the next tool call.

`query_single_recording_metadata_tool` reports the top-level `frame_count` from the combined traces and each
`plane_timing` entry's `frame_count` from that plane's `runtime_data.yaml`. The counts agree, because binarization
gives every plane of the recording the same frame count.

The ROI indices accepted by `query_traces_tool` and `query_roi_statistics_tool` are 0-based positional row indices into
the per-recording arrays, not a tracking identity. Both tools silently drop individual out-of-range indices, so always
compare the returned `roi_index` values against what you requested. When every requested index is out of range,
`query_roi_statistics_tool` returns an empty `rois` list with `success=true`, while `query_traces_tool` fails with "No
valid ROI indices provided", so a confidently "successful" empty result can only come from the statistics tool.

`query_traces_tool` accepts at most 50 ROI indices per call and rejects a longer request with "Unable to query traces.
Requested N ROIs, maximum is 50." before it resolves any path. `query_roi_statistics_tool` returns at most 500 ROIs.
Batch a larger pull across several calls.

Every `plane_index` argument names a VIRTUAL plane, so it indexes the `plane_N/` directories on disk rather than the
`plane_number` of the acquisition parameters. On an MROI recording the two differ, as the Output data reference
explains.

**`plane_index` does not default the same way across the query tools.** `query_registration_quality_tool` defaults to
`0`, which is the first imaging plane, while `query_detection_summary_tool`, `query_roi_statistics_tool`, and
`query_traces_tool` default to `-1`, which is the combined view. Comparing a default registration-quality result
against a default detection or trace result therefore compares one plane against the whole recording. You MUST pass
`plane_index` explicitly whenever you relate the two, and an unknown plane fails with "Plane directory plane_N not
found. Available: ...".

| Argument            | Tools                                                                            | Accepted values                                                                                             |
|---------------------|----------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| `plane_index`       | `query_detection_summary_tool`, `query_roi_statistics_tool`, `query_traces_tool` | `-1` combined view, `0`+ a specific plane                                                                   |
| `plane_index`       | `query_registration_quality_tool`                                                | `0`+ a specific plane. `-1` always fails, because no combined `registration_data/` exists                   |
| `trace_type`        | `query_traces_tool`, cross-recording traces                                      | `fluorescence`, `neuropil`, `corrected`, `spikes`                                                           |
| `downsample_factor` | `query_traces_tool`, cross-recording traces                                      | `1` none, `N` every Nth sample. Below 1 clamps to 1                                                         |
| `start_frame`       | `query_traces_tool`, cross-recording traces                                      | inclusive, applied before downsampling, default `0`                                                         |
| `end_frame`         | `query_traces_tool`, cross-recording traces                                      | exclusive, applied before downsampling, default all                                                         |
| `sort_by`           | `query_roi_statistics_tool`                                                      | `skewness`, `compactness`, `footprint`, `aspect_ratio`, `pixel_count`, `solidity`, `normalized_pixel_count` |
| `top_n`             | `query_roi_statistics_tool`                                                      | with `sort_by`, the top N. Without it, the first N                                                          |

`trace_type` names the trace, not the file: `corrected` returns the neuropil-subtracted trace and `spikes` returns the
deconvolved trace. An unrecognized value fails with "Invalid trace_type '...'. Valid options: ...".

---

## Output data reference

All results are saved under `{output_root}/cindra/`. The pipeline produces combined (multi-plane merged) data at the
root level and per-plane data in numbered subdirectories.

**The number of `plane_N/` directories is the VIRTUAL plane count, not `plane_number`.** For standard imaging, where
`roi_number` is 1, the two are equal. For MROI line-scanning acquisitions, where `roi_number` is above 1, the pipeline
creates one virtual plane per ROI and physical-plane combination, so the directory count is `roi_number *
plane_number`. A recording with `plane_number` 1 and `roi_number` 3 therefore produces `plane_0/`, `plane_1/`, and
`plane_2/`. Virtual planes are numbered ROI-major, so `roi_index = virtual_plane_index // plane_number` and the first
`plane_number` directories all belong to ROI 0. Each carries that ROI's line block and its x and y offsets into the
combined field of view. The per-plane sampling rate stays `frame_rate / plane_number`, because splitting a frame into
ROIs adds no acquisition time.

Channel 2 output depends on whether the second channel is functional, meaning both `main.first_channel_functional` and
`main.second_channel_functional` are True. Every two-channel recording produces, in each `plane_N/` directory,
`channel_2_data.bin`, `detection_data/mean_image_channel_2.npy`, `cell_fluorescence_channel_2.npy`, and
`neuropil_fluorescence_channel_2.npy`, plus the combined `detection_data/mean_image_channel_2.npy` at the root. A
structural (non-functional) channel 2 still gets a full fluorescence extraction pass, but it borrows the channel 1 ROI
masks instead of detecting its own ROIs. Its traces therefore carry one row per channel 1 ROI and come with no
independent detection, classification, or deconvolution. Every other channel 2 file requires both channels to be
functional: the other three detection images at both levels, the ROI `.npz` files,
`subtracted_fluorescence_channel_2.npy`, `spikes_channel_2.npy`, `cell_classification_channel_2.npy`, and the root-level
combined `cell_fluorescence_channel_2.npy` and `neuropil_fluorescence_channel_2.npy`. Those last two are the asymmetry
to watch, because combination omits them for a structural channel 2 even though every plane directory holds its own
copy.

### Directory structure

```text
cindra/
├── configuration.yaml                          # Saved pipeline configuration
├── acquisition_parameters.yaml                 # Saved acquisition metadata
├── single_recording_tracker.yaml               # Processing tracker
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
├── plane_0/                                    # Per-VIRTUAL-plane processing results
│   ├── runtime_data.yaml                       # Plane runtime metadata
│   ├── channel_1_data.bin                      # Registered binary data
│   ├── channel_1_data.bin.binarizing           # Present only while binarization fills the binary
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
`cindra run` entry point) writes them while resolving the plane contexts. Their presence therefore says nothing about
whether binarization ran. Phase 1 itself creates the per-plane `channel_1_data.bin` (and `channel_2_data.bin` if
two-channel) and per-plane `detection_data/mean_image.npy` (plus `mean_image_channel_2.npy` if two-channel), then
records `binarization_time` into each plane's `runtime_data.yaml`. Binarization consumes whole plane and channel
interleave cycles, so every plane binary of the recording holds the same frame count, and channel 2 holds exactly as
many frames as channel 1 of the same plane. Binarization refuses an existing binary whose size disagrees with its
recorded plane geometry, one that an interrupted write left marked, and a two-channel plane holding no channel 2 binary,
naming `repeat_binarization` as the remedy in each message.

**Phase 2 (registration, per-plane):** Creates `registration_data/`, rewrites the plane binary in place, refreshes
`detection_data/mean_image.npy`, and updates `runtime_data.yaml` with the registration section,
`total_registration_time`, and `registration_workers`. For the duration of the in-place rewrite, a
`{binary}.registering` marker sits beside the binary, the parallel of the `{binary}.binarizing` marker binarization
writes while it fills that binary. Either marker left on disk means that phase's write was interrupted, so the binary
holds finished frames up to an unknown point and unfinished frames after it. The suffix names the phase that died and
nothing else, because registration and binarization both refuse a binary carrying either one, and enabling
`file_io.repeat_binarization` rebuilds the binary and clears the marker in both cases.

**Phase 3 (processing, per-plane):** Creates the remaining `detection_data/` images (`enhanced_mean_image.npy`,
`maximum_projection.npy`, `correlation_map.npy`), the ROI `.npz` files, the fluorescence `.npy` traces, and updates
`runtime_data.yaml` with `total_processing_time`, `processing_workers`, and `date_processed`. Detection also overwrites
`detection_data/mean_image.npy` with the mean of the temporally binned frames. Those frames drop the bad frames and are
cropped to the registration valid range before being embedded into a full-frame array that is zero outside that range,
so the surviving file is not the whole-movie temporal mean registration wrote.

**Phase 4 (combination):** Creates combined `detection_data/` and the combined ROI and trace files at the root level by
merging all per-plane results, then writes `combined_metadata.npz` last, publishing it through an atomic write that
renames it into place. The metadata file therefore doubles as an atomic completion marker: it never exists while the
payload it describes is missing or partially written.

For every file, array shape, dtype, NPZ key, and data type convention the pipeline produces, see
[output-formats.md](references/output-formats.md).

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
MCP tool is unavailable. Replace N with the recording's virtual plane count, which is `roi_number * plane_number` from
the acquisition parameters and equals `plane_number` only when `roi_number` is 1.

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
- [ ] Each of the `roi_number * plane_number` virtual plane directories exists
- [ ] A plane named by `main.ignored_flyback_planes` is binarized and never registered or processed, so only its
      `runtime_data.yaml`, `channel_1_data.bin`, and `detection_data/mean_image.npy` are required.
      `verify_single_recording_output_tool` reports those indices under `flyback_planes` and treats every registration,
      projection, and extraction item below as optional for them
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
- [ ] `combined_metadata.npz` `plane_frame_counts` entries are all equal, which binarization guarantees by keeping
      whole plane and channel interleave cycles
```
