---
name: cindra-pipeline
description: >-
  End-to-end orchestration guide for the cindra neural imaging pipeline and the entry point for cindra work.
  Covers canonical stage ordering with handoff conditions for the single-recording and multi-recording pipelines,
  the single-vs-multi-recording decision, dataset planning, and the MCP-first execution model. Use when planning a
  full processing workflow, deciding which pipeline to run, or orienting at the start of a cindra session.
user-invocable: true
---

# Cindra pipeline

End-to-end orchestration reference and entry point for cindra neural imaging processing, covering both the
single-recording and multi-recording pipelines, their stage ordering, handoff conditions, and decision guidance.

---

## Scope

**Covers:**
- The single-vs-multi-recording decision and when each pipeline applies
- Canonical stage ordering with handoff conditions for both pipelines
- Multi-recording dataset planning (grouping, dataset names, prerequisite chain)
- The MCP-first execution model and where the CLI fits
- Quick-start references that dispatch to the stage-specific skills

**Does not cover:**
- Detailed tool usage, parameters, or troubleshooting for any stage (see the stage-specific skills below)
- Configuration parameter reference (see `/single-recording-configuration`, `/multi-recording-configuration`)
- Output data formats (see `/single-recording-results`, `/multi-recording-results`)
- MCP server connectivity (see `/cindra-mcp-environment-setup`)

**Handoff rules:** This skill dispatches to a stage-specific skill at each stage. You MUST invoke the relevant skill for
detailed tool usage, parameter reference, and troubleshooting. This skill owns the cross-stage map and the
single-vs-multi decision. The work performed within a single stage belongs to that stage's skill.

---

## Single-vs-multi-recording decision

Cindra provides two pipelines. Determine which the user needs before planning any work.

```text
Does the goal require tracking the SAME ROIs across multiple recordings (e.g. cross-day longitudinal analysis)?
  NO  → Single-recording pipeline (within-recording ROI detection and signal extraction). Run once per recording.
  YES → Multi-recording pipeline (cross-recording ROI tracking). Requires >= 2 recordings, each of which must
        ALREADY be fully single-recording processed first.
```

The multi-recording pipeline is not a replacement for the single-recording pipeline. It is a downstream pipeline that
consumes single-recording outputs. Every recording in a multi-recording dataset must complete all four single-recording
phases before multi-recording processing can run.

---

## Single-recording pipeline

```text
Environment    Acquisition    Configuration   Processing    Results       Visual
Setup       →  Data Prep   →               →             →            →  Inspection
    |              |              |              |              |              |
/cindra-mcp-   /acquisition-  /single-       /single-      /single-      /visualization
 environment-   data-          recording-     recording-    recording-
 setup          preparation    configuration  processing    results
```

### Stage 1: Environment setup

- **Skill:** `/cindra-mcp-environment-setup`
- **Actions:** Verify the cindra MCP server is connected and the `cindra` command is available
- **Handoff condition:** cindra MCP tools are accessible
- **Skip condition:** MCP already verified in this session

### Stage 2: Acquisition data preparation

- **Skill:** `/acquisition-data-preparation`
- **Actions:** Create and validate `cindra_parameters.json`, confirm TIFF layout, run
  `validate_recording_readiness_tool`
- **Handoff condition:** `validate_recording_readiness_tool` reports the recording ready
- **Skip condition:** Recording already binarized or beyond (confirm via `get_recording_status_tool`)

### Stage 3: Configuration

- **Skill:** `/single-recording-configuration`
- **Actions:** Generate a template configuration with `generate_config_file_tool`, set `main.tau` and
  `main.two_channels`, validate with `validate_config_file_tool`
- **Handoff condition:** A validated template configuration file exists (one template can serve many recordings)

### Stage 4: Processing

- **Skill:** `/single-recording-processing`
- **Actions:** Prepare and execute the four-phase pipeline (binarize, register, process, combine) via the MCP execution
  tools
- **Handoff condition:** All recordings report `completed`. `verify_single_recording_output_tool` returns
  `complete: true`

### Stage 5: Results

- **Skill:** `/single-recording-results`
- **Actions:** Verify output completeness and query metadata, registration quality, ROI statistics, and traces
- **Handoff condition:** Outputs verified and metrics reviewed

### Stage 6: Visual inspection

- **Skill:** `/visualization`
- **Actions:** Launch the ROI and registration viewers to inspect detection and motion-correction quality

---

## Multi-recording pipeline

Prerequisite: every recording in the dataset has completed all four single-recording phases, which are binarization,
registration, processing, and combination (Stages 1-5 above).

```text
Single-Recording   Configuration    Processing       Results         Visual
Complete (all)   →                →               →               →  Inspection
    |                  |                |                |                |
(see single-       /multi-          /multi-          /multi-          /visualization
 recording          recording-       recording-       recording-
 pipeline)          configuration    processing       results
```

### Stage 1: Configuration

- **Skill:** `/multi-recording-configuration`
- **Actions:** Generate a multi-recording template configuration, set `recording_io.dataset_name` to a non-empty name
  (`resolve_dataset_name_tool` builds a qualified one), set ROI selection and registration/tracking parameters, validate
  it
- **Handoff condition:** A validated multi-recording template configuration file exists. A freshly generated
  multi-recording template leaves `recording_io.dataset_name` empty, which `validate_config_file_tool` reports as an
  error, so set it before validating. `prepare_multi_recording_batch_tool` later writes the lowercased dataset name into
  the per-dataset configuration copy it saves beside the tracker, leaving the template untouched

### Stage 2: Processing

- **Skill:** `/multi-recording-processing`
- **Actions:** Confirm all recordings are single-recording complete, group recordings into datasets, resolve dataset
  names with `resolve_dataset_name_tool`, then prepare and execute the two-phase pipeline (discover, extract)
- **Handoff condition:** All datasets report `completed`. `verify_multi_recording_output_tool` returns `complete: true`

### Stage 3: Results

- **Skill:** `/multi-recording-results`
- **Actions:** Verify output completeness and query dataset overview, cross-recording registration quality, tracking
  summary, and cross-recording traces
- **Handoff condition:** Outputs verified and tracking reviewed

### Stage 4: Visual inspection

- **Skill:** `/visualization`
- **Actions:** Launch the tracking and ROI viewers to confirm backward-deformed templates overlap the same structures
  across recordings (the only reliable cross-day registration-quality check)

---

## Multi-recording dataset planning

A dataset is a named group of recordings tracked together. Plan datasets before preparing a multi-recording batch.

- **Prerequisite chain:** If any recording is not single-recording complete, route to the earliest missing step:
  `/acquisition-data-preparation` → `/single-recording-configuration` → `/single-recording-processing`.
- **Grouping:** Group recordings by common parent directory, explicit user grouping, or semantic analysis of recording
  paths. Each group becomes one dataset.
- **Dataset names:** Call `resolve_dataset_name_tool` once per group to construct a unique qualified name from a shared
  base name and a per-batch specifier. See `/multi-recording-processing` for the full workflow.

Multi-recording input preparation is "single-recording processing complete", so the single-recording pipeline handles
raw-data preparation for both pipelines and there is no separate multi-recording preparation skill.

---

## Execution interface

Cindra is MCP-first for agentic work. Every stage skill mandates the cindra MCP tools for its operations and routes to
`/cindra-mcp-environment-setup` when they are unavailable.

| Operation                                | Use                                                             |
|------------------------------------------|-----------------------------------------------------------------|
| Discovery, configuration, processing     | cindra MCP tools (`cindra-mcp` server) via the stage skills     |
| Results querying and output verification | cindra MCP tools (`cindra-mcp` server) via the results skills   |
| Viewer lifecycle and live display state  | cindra-gui MCP tools (`cindra-gui` server) via `/visualization` |

The `cindra` and `cindra-gui` CLIs (`cindra run`, `cindra-gui roi`, etc.) exist for manual, non-agentic execution. You
MUST NOT drive the pipeline through the CLI or direct Python imports during agentic work. Use the MCP tools so resource
management, prerequisite validation, and phase sequencing are handled consistently.

---

## Quick-start scenarios

### Single recording, first run

1. `/cindra-mcp-environment-setup`, to verify MCP connectivity (if first session)
2. `/acquisition-data-preparation`, to create and validate `cindra_parameters.json`
3. `/single-recording-configuration`, to generate and validate a template configuration
4. `/single-recording-processing`, to run binarize, register, process, combine
5. `/single-recording-results`, to verify and review outputs
6. `/visualization`, to inspect ROIs and registration

### Batch of recordings sharing parameters

1. `/single-recording-configuration`, to create one reusable template configuration
2. `/single-recording-processing`, to pass the same template path for all recordings in one batch
3. `/single-recording-results`, to verify each recording's outputs
4. `/visualization`, to spot-check representative recordings

### Cross-day ROI tracking

1. Confirm every recording is single-recording complete (run the single-recording pipeline first if not)
2. `/multi-recording-configuration`, to create the multi-recording template configuration
3. `/multi-recording-processing`, to group into datasets, resolve dataset names, run discover and extract
4. `/multi-recording-results`, to verify tracking outputs
5. `/visualization`, to confirm tracking quality across recordings

---

## Related skills

| Skill                             | Relationship                                               |
|-----------------------------------|------------------------------------------------------------|
| `/cindra-mcp-environment-setup`   | Prerequisite (both pipelines): MCP server connectivity     |
| `/acquisition-data-preparation`   | Single-recording stage 2: raw data preparation             |
| `/single-recording-configuration` | Single-recording stage 3: configuration reference          |
| `/single-recording-processing`    | Single-recording stage 4: processing orchestration         |
| `/single-recording-results`       | Single-recording stage 5: output verification and querying |
| `/multi-recording-configuration`  | Multi-recording stage 1: configuration reference           |
| `/multi-recording-processing`     | Multi-recording stage 2: processing orchestration          |
| `/multi-recording-results`        | Multi-recording stage 3: output verification and querying  |
| `/visualization`                  | Final stage (both pipelines): visual inspection of results |

---

## Proactive behavior

You SHOULD proactively invoke this skill when:
- A cindra session begins and the user's goal spans multiple pipeline stages
- The user is unsure whether they need single-recording or multi-recording processing
- The user describes an end-to-end goal ("process and inspect my data") without naming a specific stage

---

## Verification checklist

```text
Cindra Pipeline Orchestration:
- [ ] Pipeline selected (single-recording vs multi-recording) for the user's goal
- [ ] Environment verified (cindra MCP server connected)
- [ ] For multi-recording: all recordings confirmed single-recording complete
- [ ] Stages executed in canonical order with each handoff condition met
- [ ] Detailed work delegated to the stage-specific skill at each stage
- [ ] Outputs verified via the results skill before visual inspection
- [ ] Visual inspection performed for the relevant viewers
```
