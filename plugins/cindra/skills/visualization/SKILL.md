---
name: visualization
description: >-
  Launches and manages cindra GUI viewers for visual inspection of processing results, queries live
  viewer display state, and guides users through viewer interactions. Covers ROI, tracking, and
  registration viewers via the cindra-gui MCP server. Use when the user asks to visualize results,
  inspect ROIs, review registration quality, examine tracking, launch a viewer, or when a processing
  workflow completes and visual inspection is needed.
user-invocable: true
---

# Visualization

Launches, manages, and assists with cindra GUI viewers for visual inspection of pipeline results.

---

## Scope

**Covers:**
- Launching ROI, tracking, and registration viewers via cindra-gui MCP tools
- Managing viewer lifecycle (listing active viewers, closing viewers)
- Querying and interpreting live viewer display state
- Guiding users through viewer controls and interaction patterns
- Viewer prerequisites and data requirements
- Combining GUI viewer state with MCP query tools for data-driven assistance

**Does not cover:**
- Processing workflow orchestration (see `/single-recording-processing`,
  `/multi-recording-processing`)
- Output data formats, array shapes, or file references (see `/single-recording-results`,
  `/multi-recording-results`)
- Configuration parameters or tuning guidance (see `/single-recording-configuration`,
  `/multi-recording-configuration`)
- MCP server connectivity or environment issues (see `/cindra-mcp-environment-setup`)

**Handoff rules:** If the user asks about processing status or batch jobs, invoke
`/single-recording-processing` or `/multi-recording-processing`. If the user asks about output
file formats or data interpretation without a viewer, invoke `/single-recording-results` or
`/multi-recording-results`. If cindra-gui MCP tools are unavailable, invoke
`/cindra-mcp-environment-setup`.

---

## Agent requirements

You MUST use the cindra-gui MCP tools for all viewer operations. Do not launch viewer windows via
CLI commands, scripts, or Python imports. If cindra-gui MCP tools are not available, invoke
`/cindra-mcp-environment-setup` to diagnose and resolve connectivity issues.

When assisting the user with data interpretation in a viewer, combine `query_viewer_state_tool`
from the cindra-gui MCP server with query tools from the cindra MCP server (headless). The GUI
tools manage the viewer window; the headless query tools provide the underlying data.

---

## Available tools

### Viewer lifecycle tools (cindra-gui MCP server)

| Tool                      | Purpose                                                                     |
|---------------------------|-----------------------------------------------------------------------------|
| `launch_viewer_tool`      | Spawns a GUI viewer subprocess for the user to interact with                |
| `list_viewers_tool`       | Lists active viewers with type, path, alive status, and live active dataset |
| `close_viewer_tool`       | Terminates a viewer subprocess and cleans up state files                    |
| `query_viewer_state_tool` | Returns the live display state of an active viewer                          |

### Data query tools (cindra MCP server)

Use these headless query tools alongside viewer state to provide data-driven assistance. These
tools are documented in detail by `/single-recording-results` and `/multi-recording-results`.

| Tool                                              | Use with viewer |
|---------------------------------------------------|-----------------|
| `query_single_recording_metadata_tool`            | Any viewer      |
| `query_registration_quality_tool`                 | Registration    |
| `query_detection_summary_tool`                    | ROI             |
| `query_roi_statistics_tool`                       | ROI             |
| `query_traces_tool`                               | ROI             |
| `query_multi_recording_overview_tool`             | ROI, Tracking   |
| `query_multi_recording_registration_quality_tool` | Tracking        |
| `query_multi_recording_tracking_summary_tool`     | Tracking        |
| `query_cross_recording_traces_tool`               | ROI (multi-rec) |

---

## Viewer types

### ROI viewer

Inspects ROI masks and fluorescence traces from single-recording or multi-recording data.

**Launch:**

```python
launch_viewer_tool(viewer_type="roi", recording_path="<path>")
launch_viewer_tool(viewer_type="roi", recording_path="<path>", dataset="<name>")
```

**Prerequisites:**
- Single-recording processing must be complete (`combined_metadata.npz` exists)
- For multi-recording mode, multi-recording processing must be complete for the specified dataset

**Parameters:**
- `recording_path`: Absolute path to the recording directory containing `cindra/` output
- `dataset`: Optional multi-recording dataset name; enables tracked ROI mode when provided

**Capabilities:**
- View ROI spatial masks overlaid on detection images (mean, enhanced mean, correlation map,
  maximum projection, corrected structural)
- Color ROIs by statistics (random, skewness, compactness, footprint, aspect ratio, solidity,
  colocalization probability, recording count, cell probability, correlations, classification)
- Inspect fluorescence traces (raw, neuropil, corrected, spikes) for selected ROIs
- Toggle channel 2 overlay for dual-channel recordings
- Filter ROIs by cell classification
- Adjust opacity and colormap
- View multi-recording tracked ROIs across datasets

### Tracking viewer

Inspects multi-recording ROI tracking quality across recordings within a dataset.

**Launch:**

```python
launch_viewer_tool(viewer_type="tracking", recording_path="<path>", dataset="<name>")
```

**Prerequisites:**
- Multi-recording processing must be complete for the specified dataset

**Parameters:**
- `recording_path`: Absolute path to any recording in the multi-recording dataset
- `dataset`: Multi-recording dataset name (defaults to first available if omitted)

**Capabilities:**
- Cycle through recordings to compare ROI positions across sessions
- Switch between native and transformed coordinate spaces
- View original, deformed, template, and tracked mask layers
- Toggle channel 2 overlay for dual-channel recordings
- Auto-cycle through recordings at 500 ms intervals
- Select individual ROIs to inspect tracking consistency

### Registration viewer

Inspects motion correction (registration) quality for a single recording. Launches two windows:
a binary player for frame-by-frame playback and a PC viewer for principal component metrics.

**Launch:**

```python
launch_viewer_tool(viewer_type="registration", recording_path="<path>")
```

**Prerequisites:**
- Single-recording processing must be complete (at least through the registration phase)

**Parameters:**
- `recording_path`: Absolute path to the recording directory containing `cindra/` output
- `dataset` parameter is not used by the registration viewer

**Capabilities:**
- **Binary player**: Play back registered frames at 5x speed, step through frames with arrow
  keys, toggle channel 2 overlay
- **PC viewer**: Animate principal component extreme images per plane, cycle through PCs to
  identify residual motion artifacts

---

## Viewer state reference

Use `query_viewer_state_tool` to read the live display state of any active viewer. The returned
dictionary structure depends on the viewer type.

### ROI viewer state

| Field                      | Type      | Description                                         |
|----------------------------|-----------|-----------------------------------------------------|
| `viewer_type`              | str       | Always `"roi"`                                      |
| `loaded`                   | bool      | Whether recording data has finished loading         |
| `channel_2_active`         | bool      | Whether channel 2 overlay is toggled on             |
| `background_view`          | str       | Active background image (see Background views)      |
| `roi_color_mode`           | str       | Active ROI coloring statistic (see ROI color modes) |
| `colormap`                 | str       | Active colormap name                                |
| `selected_roi_indices`     | list[int] | Indices of currently selected ROIs                  |
| `opacity`                  | int       | ROI overlay opacity (slider value)                  |
| `classify_mode`            | bool      | Whether cell classification filter is active        |
| `trace_visibility`         | dict      | Visibility flags for each trace type (see below)    |
| `temporal_bin_size`        | int       | Temporal binning window for correlation computation |
| `colocalization_threshold` | float     | Probability threshold for channel 2 classification  |
| `roi_count`                | int       | Total number of ROIs in the recording               |
| `frame_count`              | int       | Total number of frames in the recording             |
| `two_channels`             | bool      | Whether the recording has two functional channels   |
| `all_recordings_visible`   | bool      | Whether all multi-recording ROIs are shown          |
| `roi_source`               | str       | Current ROI source dropdown text                    |
| `active_dataset`           | str\|null | Active multi-recording dataset name, or null        |
| `available_datasets`       | list[str] | List of available multi-recording dataset names     |

**`trace_visibility` sub-fields:**

| Field          | Type | Description                                 |
|----------------|------|---------------------------------------------|
| `fluorescence` | bool | Raw cell fluorescence trace visible         |
| `neuropil`     | bool | Neuropil fluorescence trace visible         |
| `corrected`    | bool | Neuropil-subtracted corrected trace visible |
| `spikes`       | bool | Deconvolved spike estimate trace visible    |

### Tracking viewer state

| Field                     | Type            | Description                                             |
|---------------------------|-----------------|---------------------------------------------------------|
| `viewer_type`             | str             | Always `"tracking"`                                     |
| `loaded`                  | bool            | Whether multi-recording data has finished loading       |
| `active_dataset`          | str             | Active multi-recording dataset name                     |
| `available_datasets`      | list[str]       | List of available dataset names                         |
| `current_recording_index` | int             | Index of the currently displayed recording              |
| `current_recording_id`    | str             | Identifier of the currently displayed recording         |
| `recording_count`         | int             | Total number of recordings in the dataset               |
| `background_view`         | str             | Active background image (see Background views)          |
| `coordinate_space`        | str             | Active coordinate space (`"native"` or `"transformed"`) |
| `mask_layer`              | str             | Active mask layer (see Mask layers)                     |
| `channel_2_active`        | bool            | Whether channel 2 overlay is toggled on                 |
| `opacity`                 | int             | ROI overlay opacity (slider value)                      |
| `selected_roi_indices`    | list[int]\|null | Indices of selected ROIs, or null if none               |
| `mask_count`              | int             | Number of masks in the active layer                     |
| `auto_cycling`            | bool            | Whether auto-recording cycling is active                |

### Registration viewer state

Returns a nested dictionary with two sub-viewers:

| Field           | Type | Description                     |
|-----------------|------|---------------------------------|
| `viewer_type`   | str  | Always `"registration"`         |
| `binary_player` | dict | Binary player state (see below) |
| `pc_viewer`     | dict | PC viewer state (see below)     |

**`binary_player` sub-fields:**

| Field              | Type | Description                                  |
|--------------------|------|----------------------------------------------|
| `current_frame`    | int  | Currently displayed frame index              |
| `frame_count`      | int  | Total number of frames                       |
| `channel_2_active` | bool | Whether channel 2 is displayed               |
| `two_channels`     | bool | Whether the recording has two channels       |
| `playing`          | bool | Whether playback is active                   |
| `frame_step`       | int  | Frame step size for navigation (default 100) |

**`pc_viewer` sub-fields:**

| Field           | Type | Description                                    |
|-----------------|------|------------------------------------------------|
| `current_plane` | int  | Currently displayed plane index                |
| `plane_count`   | int  | Total number of imaging planes                 |
| `current_pc`    | int  | Currently displayed principal component number |
| `pc_count`      | int  | Total number of principal components           |
| `playing`       | bool | Whether PC extreme animation is active         |
| `loaded`        | bool | Whether PC data has finished loading           |

---

## Enum value reference

### Background views

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

### ROI color modes

Reported in `roi_color_mode` state field. Values correspond to the statistic used to color ROI
overlays.

| Value                        | Description                                               |
|------------------------------|-----------------------------------------------------------|
| `random`                     | Random color per ROI from active colormap                 |
| `skewness`                   | Fluorescence skewness                                     |
| `compactness`                | Circularity of spatial footprint                          |
| `footprint`                  | Total spatial footprint area                              |
| `aspect_ratio`               | Bounding ellipse aspect ratio                             |
| `solidity`                   | Soma-to-convex-hull area ratio                            |
| `colocalization_probability` | Channel 2 colocalization probability                      |
| `recording_count`            | Number of recordings the ROI was tracked across           |
| `cell_probability`           | Classifier cell-probability gradient                      |
| `correlations`               | Pairwise activity correlation with selected ROI           |
| `cell_classification`        | Binary cell/non-cell label (green=cell, magenta=non-cell) |

### Mask layers

Reported in `mask_layer` state field (tracking viewer only).

| Value      | Description                                                         |
|------------|---------------------------------------------------------------------|
| `original` | Original ROI masks from single-recording extraction (native coords) |
| `deformed` | Original masks warped to shared cross-recording coordinate space    |
| `template` | Consensus template masks from cross-recording clustering            |
| `tracked`  | Template masks backward-deformed to each recording's native coords  |

### Coordinate spaces

Reported in `coordinate_space` state field (tracking viewer only).

| Value         | Description                                                    |
|---------------|----------------------------------------------------------------|
| `native`      | Original recording coordinate space                            |
| `transformed` | Warped to align with cross-recording template coordinate space |

---

## Workflows

### Launch and inspect workflow

1. **Check prerequisites** — Verify processing is complete for the recording. Use
   `get_recording_status_tool` from the cindra MCP server.

2. **Launch viewer** — Call `launch_viewer_tool` with the appropriate `viewer_type`,
   `recording_path`, and optional `dataset`. Store the returned `viewer_id`.

3. **Wait for loading** — Query state with `query_viewer_state_tool` until `loaded` is `true`.
   The viewer subprocess needs time to read data from disk. If `loaded` remains `false` after
   10-15 seconds, check for errors by verifying the viewer is still alive via `list_viewers_tool`.

4. **Assist the user** — Respond to user questions by combining viewer state with headless query
   tools. For example, if the user asks about a specific ROI, query its statistics via
   `query_roi_statistics_tool` while referencing the viewer state to understand
   what the user is currently seeing.

5. **Clean up** — When the user is done, close the viewer with `close_viewer_tool`. If the user
   closes the viewer window directly, the next `list_viewers_tool` or `query_viewer_state_tool`
   call will detect the dead process and clean up automatically.

### State-driven assistance workflow

When the user asks questions about what they see in a viewer:

1. **Query viewer state** — Call `query_viewer_state_tool` to understand the current display
   configuration (which background, which color mode, which ROIs are selected).

2. **Query underlying data** — Use the appropriate headless query tool to retrieve the actual
   data values. For example:
   - User sees colored ROIs → query `roi_color_mode` from state, then use
     `query_roi_statistics_tool` to get the statistic values
   - User asks about a trace → check `trace_visibility` and `selected_roi_indices` from state,
     then use `query_traces_tool` for the actual trace data
   - User asks about registration quality → check `binary_player.current_frame` from state,
     then use `query_registration_quality_tool` for offset statistics

3. **Explain in context** — Combine the viewer state with the queried data to give the user a
   contextual answer about what they are seeing.

### Runtime state awareness

Viewers are interactive — the user can switch datasets, change display settings, and navigate
recordings at any time via the GUI controls. The launch-time parameters passed to
`launch_viewer_tool` may not reflect the current viewer state.

**Always re-query before answering.** Before responding to any user question about a viewer,
call `query_viewer_state_tool` to read the live display state. Do not rely on cached state from
previous queries or launch-time parameters.

**Dataset tracking.** Both `list_viewers_tool` and `query_viewer_state_tool` report the
`active_dataset` field, which reflects the dataset currently displayed by the viewer. This may
differ from the `dataset` parameter provided at launch if the user switched datasets via the
viewer's dropdown controls. Use `active_dataset` (not `dataset`) when determining what the
viewer is currently showing.

### Multi-viewer workflow

You can launch multiple viewers simultaneously for the same or different recordings. Each viewer
gets a unique `viewer_id`. Use `list_viewers_tool` to track all active instances.

Common multi-viewer patterns:
- Registration viewer + ROI viewer for the same recording (verify registration then inspect ROIs)
- ROI viewers for different recordings in a multi-recording dataset (compare across sessions)
- Tracking viewer + ROI viewer for the same dataset (verify tracking then inspect traced activity)

---

## User assistance guide

### ROI viewer assistance

**"What am I looking at?"** — Query viewer state. Report the background view, ROI color mode,
number of ROIs, whether classification filter is active, and which traces are visible.

**"Are these good ROIs?"** — Query `roi_color_mode` and `classify_mode` from state. If not
already in classification mode, suggest switching to `cell_classification` or `cell_probability`
color mode. Use `query_roi_statistics_tool` to retrieve compactness, solidity,
and skewness statistics for the visible ROIs. Explain what each statistic means:
- **Compactness** near 1.0 indicates circular footprints (typical neurons)
- **Solidity** near 1.0 indicates filled footprints without holes
- **Skewness** > 0 indicates right-skewed fluorescence (active cells tend to have positive skew)

**"Show me the most active cells"** — Suggest coloring by `skewness` (high skewness correlates
with activity) or by `cell_probability` to see classifier confidence. Use
`query_roi_statistics_tool` sorted by skewness descending to identify the top
ROIs.

**"What do the traces look like?"** — Check `trace_visibility` and `selected_roi_indices` from
state. If no ROIs are selected, inform the user they need to click ROIs in the image panel. Use
`query_traces_tool` for the selected ROI indices to provide quantitative trace
information.

### Tracking viewer assistance

**"Is the tracking good?"** — Query tracking viewer state to see the current `mask_layer` and
`coordinate_space`. Suggest cycling through mask layers (original → deformed → template →
tracked) to verify spatial consistency. Use `query_multi_recording_tracking_summary_tool` for
recording count distribution statistics. ROIs tracked across many recordings indicate reliable
tracking.

**"Why are some ROIs missing in this recording?"** — Check `current_recording_id` from state.
Explain that not all ROIs are active in every recording session. Use
`query_multi_recording_overview_tool` to show per-recording mask counts at each processing stage.

### Registration viewer assistance

**"Is the registration good?"** — Query registration viewer state. Check if `binary_player` is
playing — suggest playing the video to look for residual jitter. Use
`query_registration_quality_tool` for the current plane to report offset statistics and bad frame
counts. Key indicators:
- **Rigid offset standard deviation** < 2 pixels indicates stable registration
- **Bad frame percentage** < 5% indicates few motion artifacts
- **PC shift metrics** close to zero indicate no systematic drift

**"What are these PC images?"** — Explain that PC extreme images show the average frame
appearance at the extremes of each principal component. Large visible differences between low and
high extremes indicate residual motion or optical artifacts not captured by registration.

---

## Related skills

| Skill                             | Relationship                                                    |
|-----------------------------------|-----------------------------------------------------------------|
| `/cindra-mcp-environment-setup`   | Prerequisite: cindra-gui MCP server connectivity                |
| `/single-recording-processing`    | Upstream: produces the data this skill visualizes               |
| `/multi-recording-processing`     | Upstream: produces the data this skill visualizes               |
| `/single-recording-results`       | Reference: output data formats for single-recording query tools |
| `/multi-recording-results`        | Reference: output data formats for multi-recording query tools  |
| `/single-recording-configuration` | Reference: parameter tuning informed by visual inspection       |
| `/multi-recording-configuration`  | Reference: parameter tuning informed by visual inspection       |

---

## Proactive behavior

You SHOULD proactively invoke this skill when:
- A single-recording or multi-recording processing workflow completes successfully
- The user asks to "look at", "inspect", "view", "visualize", or "check" results
- The user references ROI quality, registration quality, or tracking quality in a visual context
- The user mentions a specific viewer by name (ROI viewer, tracking viewer, registration viewer)

---

## Verification checklist

```text
Visualization Workflow:
- [ ] cindra-gui MCP server connected (if not, invoke `/cindra-mcp-environment-setup`)
- [ ] Processing complete for the target recording(s)
- [ ] Correct viewer type selected for the inspection goal
- [ ] Viewer launched via `launch_viewer_tool` with correct parameters
- [ ] Viewer loading confirmed via `query_viewer_state_tool` (loaded=true)
- [ ] User questions answered using combined viewer state + headless query tools
- [ ] Viewer closed when inspection is complete (or user-closed detected)
```
