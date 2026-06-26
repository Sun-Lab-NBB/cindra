---
name: cindra-pipeline
description: >-
  End-to-end orchestration guide for the cindra neural imaging pipeline and the entry point for cindra work.
  Covers canonical phase ordering with handoff conditions for the single-recording and multi-recording pipelines,
  the single-vs-multi-recording decision, dataset planning, and the MCP-first execution model. Use when planning a
  full processing workflow, deciding which pipeline to run, or orienting at the start of a cindra session.
user-invocable: true
---

# Cindra pipeline

End-to-end orchestration reference and entry point for cindra neural imaging processing, covering both the
single-recording and multi-recording pipelines, their phase ordering, handoff conditions, and decision guidance.

---

## Scope

**Covers:**
- The single-vs-multi-recording decision and when each pipeline applies
- Canonical phase ordering with handoff conditions for both pipelines
- Multi-recording dataset planning (grouping, dataset names, prerequisite chain)
- The MCP-first execution model and where the CLI fits
- Quick-start references that dispatch to the phase-specific skills

**Does not cover:**
- Detailed tool usage, parameters, or troubleshooting for any phase (see the phase-specific skills below)
- Configuration parameter reference (see `/single-recording-configuration`, `/multi-recording-configuration`)
- Output data formats (see `/single-recording-results`, `/multi-recording-results`)
- MCP server connectivity (see `/cindra-mcp-environment-setup`)

**Handoff rules:** This skill dispatches to phase-specific skills at each stage. Always invoke the relevant skill
for detailed tool usage, parameter reference, and troubleshooting. This skill owns the cross-phase map and the
single-vs-multi decision only — not the work performed within any single phase.

---

## Single-vs-multi-recording decision

Cindra provides two pipelines. Determine which the user needs before planning any work.

```text
Does the goal require tracking the SAME ROIs across multiple recordings (e.g. cross-day longitudinal analysis)?
  NO  → Single-recording pipeline (within-recording ROI detection and signal extraction). Run once per recording.
  YES → Multi-recording pipeline (cross-recording ROI tracking). Requires >= 2 recordings, each of which must
        ALREADY be fully single-recording processed first.
```

The multi-recording pipeline is not a replacement for the single-recording pipeline — it is a second stage that
consumes single-recording outputs. Every recording in a multi-recording dataset must complete all three
single-recording phases before multi-recording processing can run.

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

### Phase 1: Environment setup

- **Skill:** `/cindra-mcp-environment-setup`
- **Actions:** Verify the cindra MCP server is connected and the `cindra` command is available
- **Handoff condition:** cindra MCP tools are accessible
- **Skip condition:** MCP already verified in this session

### Phase 2: Acquisition data preparation

- **Skill:** `/acquisition-data-preparation`
- **Actions:** Create and validate `cindra_parameters.json`, confirm TIFF layout, run
  `validate_recording_readiness_tool`
- **Handoff condition:** `validate_recording_readiness_tool` reports the recording ready
- **Skip condition:** Recording already binarized or beyond (confirm via `get_recording_status_tool`)

### Phase 3: Configuration

- **Skill:** `/single-recording-configuration`
- **Actions:** Generate a template configuration with `generate_config_file_tool`, set `main.tau` and
  `main.two_channels`, validate with `validate_config_file_tool`
- **Handoff condition:** A validated template configuration file exists (one template can serve many recordings)

### Phase 4: Processing

- **Skill:** `/single-recording-processing`
- **Actions:** Prepare and execute the three-phase pipeline (binarize, process, combine) via the MCP execution tools
- **Handoff condition:** All recordings report `completed`; `verify_single_recording_output_tool` returns
  `complete: true`

### Phase 5: Results

- **Skill:** `/single-recording-results`
- **Actions:** Verify output completeness and query metadata, registration quality, ROI statistics, and traces
- **Handoff condition:** Outputs verified; metrics reviewed

### Phase 6: Visual inspection

- **Skill:** `/visualization`
- **Actions:** Launch the ROI and registration viewers to inspect detection and motion-correction quality

---

## Multi-recording pipeline

Prerequisite: every recording in the dataset has completed all three single-recording phases (Phases 1-5 above).

```text
Single-Recording   Configuration    Processing       Results         Visual
Complete (all)   →                →               →               →  Inspection
    |                  |                |                |                |
(see single-       /multi-          /multi-          /multi-          /visualization
 recording          recording-       recording-       recording-
 pipeline)          configuration    processing       results
```

### Phase 1: Configuration

- **Skill:** `/multi-recording-configuration`
- **Actions:** Generate a multi-recording template configuration, set ROI selection and registration/tracking
  parameters, validate it
- **Handoff condition:** A validated multi-recording template configuration file exists

### Phase 2: Processing

- **Skill:** `/multi-recording-processing`
- **Actions:** Confirm all recordings are single-recording complete, group recordings into datasets, resolve
  dataset names with `resolve_dataset_name_tool`, then prepare and execute the two-phase pipeline (discover,
  extract)
- **Handoff condition:** All datasets report `completed`; `verify_multi_recording_output_tool` returns
  `complete: true`

### Phase 3: Results

- **Skill:** `/multi-recording-results`
- **Actions:** Verify output completeness and query dataset overview, cross-recording registration quality,
  tracking summary, and cross-recording traces
- **Handoff condition:** Outputs verified; tracking reviewed

### Phase 4: Visual inspection

- **Skill:** `/visualization`
- **Actions:** Launch the tracking and ROI viewers to confirm backward-deformed templates overlap the same
  structures across recordings (the only reliable cross-day registration-quality check)

---

## Multi-recording dataset planning

A dataset is a named group of recordings tracked together. Plan datasets before preparing a multi-recording batch.

- **Prerequisite chain:** If any recording is not single-recording complete, route to the earliest missing step:
  `/acquisition-data-preparation` → `/single-recording-configuration` → `/single-recording-processing`.
- **Grouping:** Group recordings by common parent directory, explicit user grouping, or semantic analysis of
  recording paths. Each group becomes one dataset.
- **Dataset names:** Call `resolve_dataset_name_tool` once per group to construct a unique qualified name from a
  shared base name and a per-batch specifier. See `/multi-recording-processing` for the full workflow.

There is no separate multi-recording data-preparation skill: multi-recording input preparation is simply
"single-recording processing complete," so raw-data preparation is handled entirely by the single-recording
pipeline. This asymmetry is intentional.

---

## Execution interface

Cindra is MCP-first for agentic work. Every phase skill mandates the cindra MCP tools for its operations and routes
to `/cindra-mcp-environment-setup` when they are unavailable.

| Operation                                | Use                                                             |
|------------------------------------------|-----------------------------------------------------------------|
| Discovery, configuration, processing     | cindra MCP tools (`cindra-mcp` server) via the phase skills     |
| Results querying and output verification | cindra MCP tools (`cindra-mcp` server) via the results skills   |
| Viewer lifecycle and live display state  | cindra-gui MCP tools (`cindra-gui` server) via `/visualization` |

The `cindra` and `cindra-gui` CLIs (`cindra run`, `cindra-gui roi`, etc.) exist for manual, non-agentic execution.
Do not drive the pipeline through the CLI or direct Python imports during agentic work — use the MCP tools so
resource management, prerequisite validation, and phase sequencing are handled consistently.

---

## Quick-start scenarios

### Single recording, first run

1. `/cindra-mcp-environment-setup` — verify MCP connectivity (if first session)
2. `/acquisition-data-preparation` — create and validate `cindra_parameters.json`
3. `/single-recording-configuration` — generate and validate a template configuration
4. `/single-recording-processing` — run binarize, process, combine
5. `/single-recording-results` — verify and review outputs
6. `/visualization` — inspect ROIs and registration

### Batch of recordings sharing parameters

1. `/single-recording-configuration` — create one reusable template configuration
2. `/single-recording-processing` — pass the same template path for all recordings in one batch
3. `/single-recording-results` — verify each recording's outputs
4. `/visualization` — spot-check representative recordings

### Cross-day ROI tracking

1. Confirm every recording is single-recording complete (run the single-recording pipeline first if not)
2. `/multi-recording-configuration` — create the multi-recording template configuration
3. `/multi-recording-processing` — group into datasets, resolve dataset names, run discover and extract
4. `/multi-recording-results` — verify tracking outputs
5. `/visualization` — confirm tracking quality across recordings

---

## Related skills

| Skill                             | Relationship                                               |
|-----------------------------------|------------------------------------------------------------|
| `/cindra-mcp-environment-setup`   | Phase 1 (both pipelines): MCP server connectivity          |
| `/acquisition-data-preparation`   | Single-recording phase 2: raw data preparation             |
| `/single-recording-configuration` | Single-recording phase 3: configuration reference          |
| `/single-recording-processing`    | Single-recording phase 4: processing orchestration         |
| `/single-recording-results`       | Single-recording phase 5: output verification and querying |
| `/multi-recording-configuration`  | Multi-recording phase 1: configuration reference           |
| `/multi-recording-processing`     | Multi-recording phase 2: processing orchestration          |
| `/multi-recording-results`        | Multi-recording phase 3: output verification and querying  |
| `/visualization`                  | Final phase (both pipelines): visual inspection of results |

---

## Proactive behavior

You SHOULD proactively invoke this skill when:
- A cindra session begins and the user's goal spans multiple pipeline phases
- The user is unsure whether they need single-recording or multi-recording processing
- The user describes an end-to-end goal ("process and inspect my data") without naming a specific phase

---

## Verification checklist

```text
Cindra Pipeline Orchestration:
- [ ] Pipeline selected (single-recording vs multi-recording) for the user's goal
- [ ] Environment verified (cindra MCP server connected)
- [ ] For multi-recording: all recordings confirmed single-recording complete
- [ ] Phases executed in canonical order with each handoff condition met
- [ ] Detailed work delegated to the phase-specific skill at every stage
- [ ] Outputs verified via the results skill before visual inspection
- [ ] Visual inspection performed for the relevant viewers
```
