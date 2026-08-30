---
name: single-recording-configuration
description: >-
  Complete reference for single-recording pipeline configuration parameters and MCP configuration tools. Documents all
  9 configuration sections, parameter meanings, default values, and available MCP tools for generating configurations
  and discovering recordings. Use when configuring single-recording processing or when the user asks about
  single-recording configuration parameters.
user-invocable: true
---

# Single-recording configuration reference

Complete parameter reference for the single-recording (within-recording) cindra processing pipeline.

---

## Scope

**Covers:**
- All 9 configuration sections and their parameters for the `SingleRecordingConfiguration` dataclass
- Default values, types, and descriptions for every parameter
- Per-section tuning guidance for common scenarios (more cells, noisy data, new sensors)
- Pipeline-set parameters
- MCP tools for configuration generation and recording discovery
- Configuration compliance verification

**Does not cover:**
- Input data format, TIFF requirements, and acquisition parameters (see `/acquisition-data-preparation`)
- Output data formats and file references (see `/single-recording-results`)
- Processing workflow, batch operations, or status monitoring (see `/single-recording-processing`)
- Multi-recording configuration (see `/multi-recording-configuration`)

---

## Agent requirements

You MUST use the cindra MCP tools for all configuration operations. Do not hand-edit configuration files or import
cindra Python functions directly when an MCP tool exists for the task. If MCP tools are not available, invoke
`/cindra-mcp-environment-setup` to diagnose and resolve connectivity issues.

---

## Available tools

These tools are registered on the `cindra-mcp` server. Tool parameters and return values are self-documented via MCP
introspection.

| Tool                                | Purpose                                                                       |
|-------------------------------------|-------------------------------------------------------------------------------|
| `generate_config_file_tool`         | Generates a default configuration YAML for the specified pipeline type        |
| `discover_recordings_tool`          | Discovers single and multi-recording candidates under a root directory        |
| `read_config_file_tool`             | Reads any YAML file as a raw dictionary (supports legacy and non-cindra)      |
| `validate_config_file_tool`         | Validates a cindra configuration against schema, reports errors, non-defaults |
| `set_config_values_tool`            | Writes new values into an existing cindra configuration file                  |
| `validate_recording_readiness_tool` | Validates the raw TIFFs and acquisition parameters of a `raw_data_path`       |

`set_config_values_tool` takes the `file_path` of an existing configuration and a `values` map keyed by the same
`section.parameter` dotted paths `validate_config_file_tool` reports under `non_default_parameters`. Every entry is
resolved before any is applied, so one rejected entry leaves the file byte-identical and reports every rejection at
once under `errors`. Values arrive in the form the YAML document carries them, so a path is a string, an enumeration is
its raw value, and a tuple is a list. An integer is accepted for a float-typed parameter, and no other substitution is,
`bool` included. The response carries `changed`, pairing each dotted path with its `previous` and `current` values, and
the `valid` status to which the rewritten file validates. Gate on `valid` rather than on `success`. The pipeline reads
its configuration from disk when it dispatches a job, so never write against a configuration whose jobs are running.

---

## Configuration overview

The single-recording pipeline uses `SingleRecordingConfiguration`, a dataclass with 9 nested sections. Default values
are optimized for GCaMP6f data from 2-Photon Random Access Mesoscope (2P-RAM).

All parameters are specified in the `SingleRecordingConfiguration` YAML file. The pipeline loads the fully resolved
configuration directly from the file without any runtime overrides.

CPU worker allocation lives outside the configuration file. Each processing stage receives its worker count as an
invocation argument, supplied by the `cindra run` options `-bw/--binarize-workers`, `-rw/--register-workers` and
`-pw/--process-workers`, or by `execute_processing_jobs_tool` and `execute_full_pipeline_tool` at dispatch time. Both
interfaces share one convention. Omitting a `cindra run` worker option, or leaving the MCP `workers_per_job` as None,
applies the measured default of 3 workers for binarization, 8 for registration on the host CPU, 2 for a device-backed
registration, and 8 for processing. On the MCP execute tools that default is a floor rather than a fixed width. A
session that leaves `workers_per_job` as None widens each registration and processing job at dispatch as its queue
drains, up to the 32 and 16 core ceilings the classes declare. Setting either to -1 requests every available core. Any
positive value is used exactly, and on the MCP tools it overrides every non-fixed resource class alike. The CUDA device
on which registration runs is an invocation argument too. It is named by the `--register-device` option of `cindra run`,
or by the `gpu_devices` argument of the execute tools, and naming neither registers on the host CPU.
`registration.gpu_batch_size` is the only field this file owns for that path.

---

## Pipeline-set parameters

These parameters are set automatically by the pipeline and should not be manually configured:

| Parameter                       | Set by     | Value                                         |
|---------------------------------|------------|-----------------------------------------------|
| `file_io.data_path`             | batch tool | The `raw_data_paths` entry, holding the TIFFs |
| `file_io.output_path`           | user/batch | The `output_roots` entry, parent of `cindra/` |
| `runtime.display_progress_bars` | CLI/MCP    | Whether to show progress bars                 |

---

## Configuration sections

The nine sections, every parameter each one holds with its type, default, and meaning, and the tuning guidance that
accompanies them are in [parameter-reference.md](references/parameter-reference.md).

---

## User-configurable vs auto-set parameters

### Parameters users should configure

| Parameter                     | When to change                                       |
|-------------------------------|------------------------------------------------------|
| `main.tau`                    | Different calcium indicator (GCaMP6s, GCaMP7f, etc.) |
| `main.two_channels`           | Recording has two channels                           |
| `main.ignored_flyback_planes` | Flyback planes present in the recording              |
| `file_io.ignored_file_names`  | Specific TIFFs to exclude (file stems, no extension) |

### Parameters typically left at default

The registration, ROI detection, signal extraction, and spike deconvolution parameters all suit 2P GCaMP6f data.

---

## Configuration file format

```yaml
pipeline_type: single-recording

runtime:
  display_progress_bars: false

main:
  two_channels: false
  tau: 0.4
  ignored_flyback_planes: []

file_io:
  ignored_file_names: []

# Other sections use defaults...
```

The `pipeline_type` discriminator is mandatory. `generate_config_file_tool` writes it automatically, but a manually
authored file that omits it is rejected by both `validate_config_file_tool` and `cindra run`.

---

## Configuration lifecycle

1. **Template configurations**: de-novo configurations generated via `generate_config_file_tool` or manually created.
   Templates can live anywhere (e.g., `/Data/CA1_GCaMP6f_SD.yaml`) and are reusable across recordings. The batch MCP
   tools never modify a template, but `cindra run -i <file>` DOES write back into the file it is given, saving
   `runtime.display_progress_bars` and any `--data-path` or `--output-path` override into it before dispatching. Never
   pass a shared template to `cindra run`, because the first run stamps one recording's paths into the file every other
   recording shares. Pass a per-recording copy, or the resolved copy the prepare tool already wrote.

2. **Resolved copies**: when `prepare_single_recording_batch_tool` runs, it loads the template, applies
   recording-specific overrides (`file_io.data_path` from `raw_data_paths`, `file_io.output_path` from the required
   `output_roots` parameter, `runtime.display_progress_bars=False`), and saves the resolved copy as
   `cindra/configuration.yaml` inside each recording's output root. Preparation never rewrites a copy it already wrote,
   so a re-prepare with different paths reports them under `path_conflicts` and keeps the stored ones. Amend a resolved
   copy with `set_config_values_tool` instead, and only while none of its jobs are running.
   `execute_processing_jobs_tool` resolves worker allocation at dispatch time and passes it to each job as a dispatch
   argument, so one configuration file serves every job dispatched against it. These resolved copies are what the
   pipeline executes.

**Do NOT** create per-recording configuration files manually. Pass a single template path to the batch tool and let it
handle per-recording fine-tuning automatically.

---

## Configuration workflow

1. **Discover recordings** using `discover_recordings_tool` to find directories with raw data. Every entry of
   `single_recording_candidates` is an object carrying `recording_root` and `raw_data_path`, so read
   `candidate["raw_data_path"]` for every downstream tool and `candidate["recording_root"]` only when naming the session
   to the user. The recording root is a session-level directory that usually does not hold the TIFF files itself. Every
   tool taking a raw data path accepts it too, because each resolves the imaging directory by locating
   `cindra_parameters.json` beneath the path it is given.
2. **Verify data readiness**: use `validate_recording_readiness_tool` with `raw_data_path` set to that
   `candidate["raw_data_path"]`, which is the directory that directly holds the TIFF files and the
   `cindra_parameters.json` file. Passing `candidate["recording_root"]` works equally well, because the tool resolves
   the imaging directory the way step 1 describes. If any recording fails validation, invoke
   `/acquisition-data-preparation` to resolve before continuing.
3. **Generate a template configuration** using `generate_config_file_tool` with `pipeline_type="single-recording"`. Save
   it at a user-chosen location (e.g., `/Data/CA1_GCaMP6f_SD.yaml`). Alternatively, use `read_config_file_tool` to
   inspect an existing or legacy configuration for conversion.
4. **Review and modify** the template using `set_config_values_tool`, setting at minimum `main.tau` and
   `main.two_channels`. Pass every change in one `values` map, because a rejected entry leaves the file unchanged.
5. **Validate** the configuration using `validate_config_file_tool` to check for errors, warnings, and non-default
   parameters. The generated template leaves `file_io.output_path` as None, which the planning tools
   `size_pipeline_jobs_tool` and `get_pipeline_job_universe_tool` reject, so set `file_io.data_path` and
   `file_io.output_path` on a per-recording copy before planning against it.
6. **Configuration complete**: the validated template file is ready for use. This skill does not start processing. To
   run it, proceed to `/single-recording-processing`. If invoked from another skill, return control to the caller.

---

## Related skills

| Skill                            | Relationship                                                               |
|----------------------------------|----------------------------------------------------------------------------|
| `/cindra-pipeline`               | Overview: end-to-end phases, handoffs, and the single-vs-multi entry point |
| `/cindra-mcp-environment-setup`  | Prerequisite: MCP server must be connected for configuration tools         |
| `/cli-reference`                 | Reference: `cindra configure` and the `cindra run` worker options          |
| `/acquisition-data-preparation`  | Prerequisite: raw data must be prepared before configuring the pipeline    |
| `/single-recording-processing`   | Next step: processing workflow that consumes this configuration            |
| `/single-recording-results`      | Output data format reference for evaluating processing results             |
| `/multi-recording-configuration` | Companion configuration reference for the multi-recording pipeline         |
| `/multi-recording-processing`    | Downstream: multi-recording requires single-recording processing first     |
| `/visualization`                 | Downstream: launch viewers to inspect results after processing             |

---

## Verification checklist

You MUST verify configuration files against this checklist before starting single-recording processing. Use
`validate_config_file_tool` for automated validation of YAML structure, parameter constraints, and pipeline-set
parameter detection.

```text
Single-Recording Configuration Compliance, tool-settled (run `validate_config_file_tool` and
`validate_recording_readiness_tool`):
- [ ] `validate_config_file_tool` reports no errors (run this first)
- [ ] Review any warnings from `validate_config_file_tool` (pipeline-set parameters, channel consistency)
- [ ] Acquisition data prepared, `validate_recording_readiness_tool` passed against each `raw_data_path` (else run
      `/acquisition-data-preparation`)

Single-Recording Configuration Compliance, reader-judged:
- [ ] cindra MCP server is connected (if not, invoke `/cindra-mcp-environment-setup`)
- [ ] `main.tau` matches the calcium indicator used (0.4 for GCaMP6f, ~1.5 for GCaMP6s)
- [ ] `main.two_channels` set correctly for the recording type
- [ ] `main.ignored_flyback_planes` lists correct flyback plane indices if applicable
- [ ] `file_io.ignored_file_names` excludes every TIFF in the data directory that is not part of the recording (a
      differently shaped file, such as an anatomical z-stack, fails binarization)
- [ ] No shared template was passed to `cindra run -i`, which writes back into the file it is given
```
