---
name: cli-reference
description: >-
  Documents the human-facing cindra and cindra-gui command-line interfaces. Covers every command and option with its
  short form, long form, type, default, and effect, the MCP tool each command maps to, its failure modes, and how the
  CLI path diverges from the MCP path. Use when a user asks what a cindra or cindra-gui command or option does, or when
  the MCP server is unavailable and the user must be told what to run by hand.
user-invocable: true
---

# CLI reference

> **The `cindra` and `cindra-gui` CLIs are HUMAN-FACING tools. You MUST never invoke them.** Print the command, ask the
> user to run it, and ask them to paste the output back.

**The one exemption** is `--help`. `cindra --help`, `cindra COMMAND --help`, `cindra-gui --help`, and
`cindra-gui COMMAND --help` may be run, and no other invocation is exempt. `/cindra-mcp-environment-setup` owns that
exemption along with the `cindra omp` recovery command it drives.

---

## Scope

**Covers:**
- The complete `cindra` and `cindra-gui` command surface, each command's purpose, and its MCP equivalent
- Every declared option: short form, long form, type, default, required or flag status, and effect
- Per-command failure modes and how the error routing guard renders them
- How `cindra run` diverges from an MCP batch, and how a hand-launched viewer diverges from `launch_viewer_tool`
- What to tell a user to run when the MCP server cannot be restored in this session

**Does not cover:**
- The MCP batch workflow, prerequisite validation, and resource sizing (see `/single-recording-processing`,
  `/multi-recording-processing`)
- Configuration parameter meanings and tuning guidance (see `/single-recording-configuration`,
  `/multi-recording-configuration`)
- Output data formats the commands produce (see `/single-recording-results`, `/multi-recording-results`)
- Viewer controls, display state, and the classifier panel (see `/visualization`)
- Diagnosing why the MCP server is down, and the `cindra omp` linking workflow (see
  `/cindra-mcp-environment-setup`)
- Raw data preparation and acquisition parameter files (see `/acquisition-data-preparation`)

**Handoff rules:** If the user wants an operation performed rather than explained, use the MCP tools and invoke the
owning skill. If the MCP tools are unavailable, invoke `/cindra-mcp-environment-setup` first, and fall back to the
handoff table below only after the server cannot be restored.

---

## Agent requirements

You MUST answer CLI questions from this skill or from `cindra COMMAND --help`, never from memory. When a user's report
disagrees with this reference, ask them to run `cindra COMMAND --help` and read the installed build's answer.

---

## Command surface

Two entry points ship from `pyproject.toml`. `cindra` maps to `cindra.interface.cli:cindra_cli` and `cindra-gui` maps
to `cindra.interface.gui_cli:cindra_gui`. The GUI entry point is separate so that a headless pipeline run never loads
Qt.

| Command                   | Kind    | Purpose                                                | MCP equivalent                                               |
|---------------------------|---------|--------------------------------------------------------|--------------------------------------------------------------|
| `cindra`                  | group   | Entry point. Dispatches to the subcommands             | None, dispatch only                                          |
| `cindra mcp`              | command | Starts the data processing MCP server                  | None, it hosts the server                                    |
| `cindra omp`              | command | Reports and links the macOS OpenMP runtime Numba loads | None, and the command errors off macOS                       |
| `cindra gpu`              | command | Reports the CUDA devices the host exposes              | `check_gpu_runtime_tool`                                     |
| `cindra configure`        | command | Generates a default pipeline configuration file        | `generate_config_file_tool`                                  |
| `cindra run`              | command | Runs a pipeline from a configuration file              | `execute_full_pipeline_tool`, `execute_processing_jobs_tool` |
| `cindra-gui`              | group   | Entry point. Dispatches to the viewer subcommands      | None, dispatch only                                          |
| `cindra-gui roi`          | command | Launches the ROI viewer                                | `launch_viewer_tool(viewer_type="roi")`                      |
| `cindra-gui registration` | command | Launches the registration quality viewer               | `launch_viewer_tool(viewer_type="registration")`             |
| `cindra-gui tracking`     | command | Launches the multi-recording tracking viewer           | `launch_viewer_tool(viewer_type="tracking")`                 |
| `cindra-gui mcp`          | command | Starts the GUI MCP server                              | None, it hosts the server                                    |

`cindra omp` is the only command carrying no MCP substitute for its main effect, and it is also the only command
confined to one platform. It runs on macOS alone and errors on every other host, so it reports nothing about the TBB
runtime a Linux or Windows host needs. `check_threading_runtime_tool` is the portable check. It runs everywhere,
reports the layer the host's platform selects under `required_layer`, and names a remedy, so it substitutes for the
report half of `cindra omp` on macOS alone and covers every other host on its own. Creating the macOS link still needs
the CLI, because the write target usually requires sudo.

---

## Option reference

Both groups set a help width of 120 characters. Neither group declares options of its own.

### `cindra mcp`

| Short | Long          | Type                                        | Default | Status   | Effect                         |
|-------|---------------|---------------------------------------------|---------|----------|--------------------------------|
| `-t`  | `--transport` | choice of `stdio`, `sse`, `streamable-http` | `stdio` | optional | Selects the transport to serve |

The `stdio` transport disables the console, because that transport carries the JSON-RPC stream over stdout and a logged
line renders the message it interleaves with unparsable.

### `cindra omp`

| Short | Long       | Type          | Default | Status   | Effect                                                     |
|-------|------------|---------------|---------|----------|------------------------------------------------------------|
| `-s`  | `--source` | existing file | `None`  | optional | Names the runtime to link, omit to search the host for one |
| `-t`  | `--target` | file path     | `None`  | optional | Names the link to write, omit to derive the loader default |
| `-f`  | `--force`  | flag          | `False` | flag     | Links even on a host whose runtime already loads           |
| `-y`  | `--yes`    | flag          | `False` | flag     | Creates the link, omit to report and change nothing        |

Without `-y` the command is a report. `/cindra-mcp-environment-setup` owns the workflow that drives it.

### `cindra gpu`

Declares no options. The command reports every CUDA device the host exposes, with the memory and compute capability of
each, and exits with a non-zero status when it reaches none. `check_gpu_runtime_tool` returns the same report as JSON,
so call the tool and hand over the command only while the server is down.

### `cindra configure`

| Short | Long            | Type                                                        | Default | Status   | Effect                                       |
|-------|-----------------|-------------------------------------------------------------|---------|----------|----------------------------------------------|
| `-p`  | `--pipeline`    | choice of `single-recording`, `sd`, `multi-recording`, `md` | none    | required | Selects the pipeline to configure            |
| `-od` | `--output-path` | existing directory                                          | none    | required | Names the directory to write into            |
| `-n`  | `--name`        | string                                                      | `None`  | optional | Names the file, defaulting per pipeline type |

Omitting `-n` writes `cindra_sd_conf.yaml` for a single-recording pipeline and `cindra_md_conf.yaml` for a
multi-recording one. The name keeps every component it carries and gains the `.yaml` suffix, so `-n mouse5_2024.03.01`
writes `mouse5_2024.03.01.yaml` rather than truncating at the last dot. A `.yml` name is normalized to `.yaml`. The MCP
equivalent takes the whole file path, name included, as its `output_path`, and normalizes any suffix other than `.yaml`
through `Path.with_suffix('.yaml')`, which replaces the component following the last dot. The two paths therefore agree
on every name carrying no dot and on every name ending in `.yaml` or `.yml`. They diverge on any other name carrying a
dot, where the MCP tool writes `mouse5_2024.03.yaml` for the name the CLI writes in full. Ask a user reproducing an
agent's file by hand for a dotless name, or hand them the `file_path` the MCP tool returned.

### `cindra run`

The pipeline type is detected from the configuration file, so one command serves both pipelines and each pipeline
ignores the other's flags. With no phase flag set, every phase of the detected pipeline runs in order. With `--job-id`
set, only the matching job runs and every phase flag is ignored.

| Short | Long                 | Type               | Default | Status     | Effect                                         |
|-------|----------------------|--------------------|---------|------------|------------------------------------------------|
| `-i`  | `--input-path`       | file path          | none    | required   | Names the configuration file to run            |
| `-bw` | `--binarize-workers` | integer            | `None`  | optional   | Workers for the binarization stage             |
| `-rw` | `--register-workers` | integer            | `None`  | optional   | Workers for the registration stage             |
| `-rd` | `--register-device`  | integer            | `None`  | optional   | CUDA device the registration stage runs on     |
| `-pw` | `--process-workers`  | integer            | `None`  | optional   | Workers for the processing stage               |
| `-dw` | `--discover-workers` | integer            | `None`  | optional   | Workers for the discovery stage                |
| `-ew` | `--extract-workers`  | integer            | `None`  | optional   | Workers for the extraction stage               |
| `-np` | `--no-progress`      | flag               | `False` | flag       | Suppresses the progress bars                   |
| `-id` | `--job-id`           | string             | `None`  | optional   | Runs only the named job                        |
| `-b`  | `--binarize`         | flag               | `False` | flag       | Runs the binarization phase                    |
| `-r`  | `--register`         | flag               | `False` | flag       | Runs the registration phase                    |
| `-p`  | `--process`          | flag               | `False` | flag       | Runs the processing phase                      |
| `-c`  | `--combine`          | flag               | `False` | flag       | Runs the combination phase                     |
| `-tp` | `--target-plane`     | integer            | `-1`    | optional   | Limits the run to one plane, `-1` meaning all  |
| `-dp` | `--data-path`        | existing directory | `None`  | optional   | Overrides the configured raw data directory    |
| `-s`  | `--output-path`      | directory          | `None`  | optional   | Overrides the configured output directory      |
| `-d`  | `--discover`         | flag               | `False` | flag       | Runs the discovery phase                       |
| `-e`  | `--extract`          | flag               | `False` | flag       | Runs the extraction phase                      |
| `-tr` | `--target-recording` | string             | `None`  | optional   | Limits the run to one recording of the dataset |
| `-rp` | `--recording-path`   | existing directory | `()`    | repeatable | Overrides the configured recording directories |

Two spellings invite a mistake. The output directory is `-s` here and `-od` on `cindra configure`, and `-p` is
`--process` here while it is `--pipeline` on `cindra configure`. Quote the long form whenever you hand a user one of
these.

The five worker options follow the shared sentinel convention. Omitting the option accepts the measured default for
that stage, `-1` requests every available core, and a positive integer is used exactly. Any other non-positive value is
rejected. `/single-recording-configuration` owns that convention.

`--register-device` carries a contract of its own. Omitting it registers every plane of the run on the host CPU, and a
zero-based CUDA device index registers every plane on that device. Every negative value is a usage error here,
including the `-1` the worker options read as a request for the whole host. The option is single-recording only, so
passing it against a multi-recording configuration is a usage error as well.

**A single-recording run needs an output path before it starts.** The command applies `-dp` and `-s` to the loaded
configuration, then aborts with a usage error when `file_io.output_path` is still None, naming both the field and the
`--output-path` flag. A freshly generated configuration carries None there, and `generate_config_file_tool` leaves it
None as well, so a template the agent produced runs only once the user passes `-s <output-root>` or sets the field in
the file. That output root is the parent of the `cindra` directory the run writes, which is the same value the MCP
tools name `output_root`. Include `-s` in every single-recording `cindra run` command you hand over, and confirm the
field is set before omitting it.

### `cindra-gui roi`, `cindra-gui registration`, and `cindra-gui tracking`

| Short | Long               | Type               | Default | Status   | Effect                                          |
|-------|--------------------|--------------------|---------|----------|-------------------------------------------------|
| `-r`  | `--recording-path` | existing directory | none    | required | Names the pipeline output root to open          |
| `-d`  | `--dataset`        | string             | `None`  | optional | Selects the multi-recording dataset             |
| `-sf` | `--state-file`     | file path          | `None`  | optional | Names the file the viewer writes its state into |

`cindra-gui registration` declares no `--dataset`, so passing it there is a usage error. On `roi`, supplying
`--dataset` switches the viewer into tracked ROI mode. `--recording-path` takes the pipeline output root, which is the
parent of the recording's `cindra` directory, and `launch_viewer_tool` names that same value `output_root` while
passing it through this flag. `launch_viewer_tool` builds exactly these invocations and always supplies `--state-file`,
which is how the MCP path reads live viewer state.

### `cindra-gui mcp`

Carries the same `-t` / `--transport` option, with the same choices and the same `stdio` default, as `cindra mcp`.

---

## Error routing and failure modes

Every command of both CLIs carries a decorator that reports a failure through the console instead of an interpreter
traceback. The message reaches stderr at the ERROR level, and the command still exits zero, so the exit code reports
whether Click accepted the invocation rather than whether the work succeeded.

| Observed                                                   | Exit | Meaning                                                              |
|------------------------------------------------------------|------|----------------------------------------------------------------------|
| An ERROR line naming the failure, and no traceback         | 0    | A library error. The message names the cause                         |
| `Usage: ...` followed by `Error: ...`                      | 2    | A rejected option value, whether Click or the command body caught it |
| `Aborted!`                                                 | 1    | The user interrupted the command                                     |
| A report ending in an unresolved status, from `cindra omp` | 1    | No OpenMP runtime resolved on this host                              |
| A device report, from `cindra gpu`                         | 1    | No usable CUDA device on this host                                   |

**Never read a zero exit code as success.** A library failure exits zero, so ask the user for the terminal output
rather than for the exit status, and read the ERROR line to decide what failed. A Click exception passes through the
decorator, so every malformed invocation exits 2 whether Click's own validation or the command body caught it.

**A traceback means an outdated build.** Every command of both groups carries the decorator, so a user reporting a
Python stack dump is running a build from before it landed. Ask them to upgrade rather than interpreting the stack.

The failures a user hits most, and where each belongs:

| Message names                                           | Cause                                                   | Route to                          |
|---------------------------------------------------------|---------------------------------------------------------|-----------------------------------|
| No configuration file, or a bad pipeline type key       | `-i` points at a missing, malformed, or non-YAML file   | `/single-recording-configuration` |
| A missing configuration field                           | A hand-edited file dropped a required key               | `/single-recording-configuration` |
| No `cindra_parameters.json`, or an acquisition mismatch | The recording was never prepared                        | `/acquisition-data-preparation`   |
| TIFF frames of differing shape                          | The data directory mixes frame geometries               | `/acquisition-data-preparation`   |
| A plane that must be registered before detection        | A phase flag ran a phase out of order                   | `/single-recording-processing`    |
| No `combined_metadata.npz` under a recording            | A multi-recording input never finished single-recording | `/single-recording-processing`    |
| No `configuration.yaml` under a viewer's output root    | The recording was never processed                       | `/single-recording-processing`    |
| An OpenMP runtime that cannot be located, on macOS      | The threading runtime is not on the loader search path  | `/cindra-mcp-environment-setup`   |
| No usable CUDA device, after --register-device          | The host reaches no device through the CuPy runtime     | `/cindra-mcp-environment-setup`   |
| A lock acquisition that timed out                       | Another cindra process holds the tracker                | See the contention warning below  |

---

## How the CLI diverges from the MCP path

`cindra run` is not a hand-operated version of the batch engine. It calls the sequential pipeline entry points
directly, so the two paths differ in ways that change results rather than only timing. State these before recommending
any `cindra run` command.

| Topic            | `cindra run`                                                | The MCP batch                                            |
|------------------|-------------------------------------------------------------|----------------------------------------------------------|
| Dispatch         | One job at a time, in phase order, inside the CLI's process | Every job in its own process, several at once            |
| A failing job    | Unwinds the run, later jobs never start and stay SCHEDULED  | Confined to its dependency subtree, other work continues |
| A killed job     | Leaves the job at RUNNING in the tracker permanently        | Converged to a terminal state naming the likely cause    |
| Prerequisites    | Validated for the target plane or recording alone           | Validated against the tracker before any job starts      |
| Completed work   | Re-run unconditionally                                      | Skipped, reporting that the phases are already complete  |
| Resource budgets | Per-stage counts, with no session core or memory budget     | Session core and memory budgets bound admission          |
| Backend threads  | Sized to the whole host                                     | Each worker pinned to one backend thread                 |
| CUDA device      | One device for every plane of the run                       | One device per running job, from the session list        |

Three of these are hazards rather than trade-offs, so warn the user before handing over a command.

**`cindra run` rewrites the file `-i` names.** It writes the resolved progress-bar setting and any `-dp`, `-s`, or
`-rp` override back into that file before dispatching. A template run once by hand therefore carries one recording's
paths, and the next batch the agent prepares inherits them. Tell the user to point `-i` at a per-recording copy, never
at a template the agent uses.

**`cindra run` rewrites the shared bootstrap other workers read.** Invoked without `--job-id`, it persists the
recording's configuration and every plane's runtime data from its own start-of-run snapshot. Run against an output
directory a live MCP session is working on, it overwrites results those workers already recorded.

**Nothing excludes the two paths from each other.** Both lock the same tracker file, and that lock serializes
individual writes rather than whole jobs. A hand run against a directory the agent is executing can time out on the
lock, or restart a job the engine is already running so that two processes write the same plane. Never start an MCP
batch over a directory a user has a run in, and ask the user to wait for a session to drain before running anything.

Two smaller mismatches shape the advice you give. `-bw` genuinely changes binarization, while the MCP binarization
class is fixed and discards `workers_per_job`, so an agent has never changed it. The CLI's per-stage granularity has no
MCP counterpart either, because `workers_per_job` is one value applied to every non-fixed class.

A hand-launched viewer is untracked. `launch_viewer_tool` records the process so `list_viewers_tool`,
`query_viewer_state_tool`, and `close_viewer_tool` reach it, and a viewer the user starts is reachable by none of them.

---

## Fallback: what to tell a user when MCP is unavailable

| Blocked MCP tool                                             | Tell the user to run                             |
|--------------------------------------------------------------|--------------------------------------------------|
| `generate_config_file_tool`                                  | `cindra configure -p single-recording -od <dir>` |
| `prepare_single_recording_batch_tool` plus the execute tools | `cindra run -i <config> -s <output-root>`        |
| `prepare_multi_recording_batch_tool` plus the execute tools  | `cindra run -i <config>`                         |
| `check_threading_runtime_tool`, on macOS alone               | `cindra omp`                                     |
| `check_gpu_runtime_tool`                                     | `cindra gpu`                                     |
| `launch_viewer_tool(viewer_type="roi")`                      | `cindra-gui roi -r <output-root>`                |
| `launch_viewer_tool(viewer_type="registration")`             | `cindra-gui registration -r <output-root>`       |
| `launch_viewer_tool(viewer_type="tracking")`                 | `cindra-gui tracking -r <output-root> -d <name>` |

Every rule the divergence section states applies to these rows. One caveat is new here, that neither CLI reports its
results in a machine-readable form, so ask the user to paste the terminal output.

Two rows carry a condition the table cannot hold. The single-recording `cindra run` row keeps `-s` because the run
aborts without a configured output path, and the multi-recording row omits it because that pipeline reads
`recording_io.recording_directories` and `recording_io.dataset_name` from the file instead. The `cindra omp` row holds
on macOS alone, so on Linux and Windows tell the user to check the TBB runtime with `python -c "import tbb"` in the
environment cindra runs in, and to install `tbb4py` when that import fails.

Everything else blocks until the server is back. That covers acquisition parameter generation and validation, recording
discovery, dataset name resolution, and configuration reading, validation, and modification. It also covers two of the
four planning tools, `get_pipeline_job_universe_tool` and `size_pipeline_jobs_tool`, because the table above already
substitutes for `check_threading_runtime_tool` and `check_gpu_runtime_tool`. It covers every batch status, timing,
cancel, reset, and cleanup tool, all output verification and query tools, and live viewer state. Say so plainly rather
than improvising a substitute.

---

## Related skills

| Skill                             | Relationship                                                                  |
|-----------------------------------|-------------------------------------------------------------------------------|
| `/cindra-pipeline`                | Context: where each command sits in the end-to-end pipeline                   |
| `/cindra-mcp-environment-setup`   | Owns the `--help` exemption, the `cindra omp` workflow, and MCP recovery      |
| `/single-recording-processing`    | Owns the MCP batch workflow `cindra run` substitutes for                      |
| `/multi-recording-processing`     | Owns the multi-recording batch workflow and its re-run semantics              |
| `/single-recording-configuration` | Owns the file `cindra configure` generates and the worker sentinel convention |
| `/multi-recording-configuration`  | Owns the multi-recording configuration file and its parameters                |
| `/acquisition-data-preparation`   | Owns the raw data and acquisition parameters a run reads                      |
| `/visualization`                  | Owns viewer controls and the tracked state a hand-launched viewer forfeits    |
| `/single-recording-results`       | Downstream: the output a completed run writes                                 |
| `/multi-recording-results`        | Downstream: the output a completed multi-recording run writes                 |

---

## Verification checklist

```text
Answering a CLI question, tool-settled (run `cindra COMMAND --help`, the sole sanctioned invocation):
- [ ] Answered from this skill or from `COMMAND --help`, never from memory
- [ ] Quoted the long option form, since `-s`, `-p`, and `-od` differ in meaning between commands

Answering a CLI question, reader-judged:
- [ ] Named the MCP equivalent alongside any command the agent could have run itself
- [ ] Invoked no `cindra` or `cindra-gui` command other than `--help`
- [ ] Treated a reported traceback as an outdated build rather than interpreting the stack

Handing a user a CLI command, reader-judged:
- [ ] Confirmed MCP is genuinely unavailable via `/cindra-mcp-environment-setup` first
- [ ] Printed the command for the user instead of running it
- [ ] Passed `-s <output-root>` on a single-recording `cindra run`, or confirmed `file_io.output_path` is already set
- [ ] Named `check_threading_runtime_tool` rather than `cindra omp` for a threading check off macOS
- [ ] Stated that `--register-device` takes a non-negative CUDA device index alone, since `-1` is a usage error there
- [ ] Warned that `cindra run` rewrites the configuration file `-i` names
- [ ] Warned about sequential dispatch and abort-on-first-failure before recommending `cindra run`
- [ ] Confirmed no MCP session holds the tracker for that output directory
- [ ] Asked the user to paste the terminal output, since neither CLI reports machine-readable results
- [ ] Read the pasted output for an ERROR line rather than treating a zero exit code as success
```
