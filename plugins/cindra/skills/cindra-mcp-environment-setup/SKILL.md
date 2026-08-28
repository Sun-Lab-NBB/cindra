---
name: cindra-mcp-environment-setup
description: >-
  Diagnoses and resolves cindra and cindra-gui MCP server connectivity issues. Covers environment verification, command
  availability, Python version checks, dependency validation, and conda/pip/uv environment configuration. Use when the
  cindra or cindra-gui MCP tools are unavailable, when either MCP server fails to start, when the user reports cindra
  connection issues, or when starting a session that requires the cindra MCP tools.
user-invocable: true
---

# MCP environment setup

Diagnoses and resolves cindra and cindra-gui MCP server connectivity and environment configuration issues.

---

## Scope

**Covers:**
- Verifying the cindra and cindra-gui MCP servers are reachable and functional
- Diagnosing why the `cindra` or `cindra-gui` commands are unavailable
- Checking Python version compatibility
- Validating cindra package installation and dependencies
- Verifying the numeric threading runtime each platform needs (OpenMP on macOS, TBB elsewhere)
- Verifying the CUDA runtime a device-backed registration needs, and resolving an unusable one
- Environment-specific guidance for conda, pip, and uv workflows

**Does not cover:**
- MCP tool usage for data processing (see `/single-recording-processing`, `/multi-recording-processing`)
- MCP tool usage for configuration (see `/single-recording-configuration`, `/multi-recording-configuration`)
- MCP tool usage for results querying and output verification (see `/single-recording-results`,
  `/multi-recording-results`)
- MCP tool usage for visualization (see `/visualization`)
- cindra package development and contribution workflows

---

## Agent requirements

You MUST use shell and CLI diagnostics (for example `which cindra`, `python --version`, `pip check cindra`) while this
skill is active, because it runs precisely when the cindra MCP tools are unavailable. This skill is the deliberate
exception to the MCP-first rule the other cindra skills follow. Once connectivity is restored, the other cindra skills
resume using the MCP tools.

---

## Architecture

cindra provides two separate MCP servers, each accessed through its own CLI entry point defined in `pyproject.toml`:

```toml
[project.scripts]
cindra = "cindra.interface.cli:cindra_cli"
cindra-gui = "cindra.interface.gui_cli:cindra_gui"
```

| Server       | CLI command      | Purpose                                                                     |
|--------------|------------------|-----------------------------------------------------------------------------|
| `cindra-mcp` | `cindra mcp`     | Headless processing: discovery, configuration, results querying, batch jobs |
| `cindra-gui` | `cindra-gui mcp` | GUI viewer lifecycle management and live display-state queries              |

Both servers accept a `--transport` option (defaults to `stdio`). The cindra Claude Code plugin registers both servers
in its `plugin.json`:

```json
{
  "mcpServers": {
    "cindra-mcp": {
      "command": "cindra",
      "args": ["mcp"]
    },
    "cindra-gui": {
      "command": "cindra-gui",
      "args": ["mcp"]
    }
  }
}
```

When the plugin is installed, Claude Code automatically discovers and starts both servers. The `cindra` and `cindra-gui`
commands must be on PATH when Claude Code starts. This means the Python environment where cindra is installed must be
active before launching Claude Code.

### Dual-distribution model

| Component                                        | Distributed via           | What it provides                                                        |
|--------------------------------------------------|---------------------------|-------------------------------------------------------------------------|
| Skills (`/single-recording-processing`, etc.)    | cindra Claude Code plugin | Skill files that guide agents through workflows                         |
| MCP server registrations                         | cindra Claude Code plugin | `plugin.json` mcpServers entries that register servers with Claude Code |
| MCP server code (`cindra mcp`, `cindra-gui mcp`) | cindra pip package        | The actual CLI commands and server implementations                      |

Installing the plugin alone registers the MCP servers and makes skills available, but the servers will fail to start
because the `cindra` and `cindra-gui` CLI commands are not present. The pip package must also be installed in the active
Python environment for the MCP servers to function.

This is the most common cause of MCP failures after initial setup. Either the plugin is installed while the pip package
is missing, or the pip package sits in a different Python environment than the one active when Claude Code launches.

### Package version requirement

cindra is distributed as a pre-release build, so every install, upgrade, and reinstall command MUST carry pip's `--pre`
flag. The MCP tool surface these skills document (the four single-recording phases, the `workers_per_job` and
`max_parallel_jobs` arguments of the execute tools, and the measured worker defaults in `cindra.orchestration`,
including the multi-recording discovery and extraction defaults) ships in cindra 2.0.0+. Without `--pre`, pip resolves
an older build whose MCP tools do not match the documented surface, which presents as tools that reject a documented
phase name or argument while the server itself reports as connected.

---

## Diagnostic workflow

You MUST follow these steps in order when MCP tools are unavailable or a server fails to start. Apply these steps to
whichever server is affected (`cindra` for the headless server, `cindra-gui` for the GUI server). If both are affected,
diagnose them in sequence.

### Step 1: Check MCP server status

Use the `/mcp` slash command or inspect available tools to determine whether the affected MCP server is connected. If
connected, the issue is not environmental, so investigate tool-specific errors instead.

### Step 2: Verify command availability

```bash
which cindra
which cindra-gui
```

If the affected command is not found, proceed to step 3. If found, skip to step 4.

### Step 3: Identify the environment type and resolve

Run these commands to determine the user's environment setup:

```bash
echo "CONDA_PREFIX: ${CONDA_PREFIX:-not set}"
echo "VIRTUAL_ENV: ${VIRTUAL_ENV:-not set}"
python --version
pip list 2>/dev/null | grep cindra
```

Based on the output, guide the user through the appropriate resolution:

**Conda environment (CONDA_PREFIX is set but cindra is missing):**

The user has an active conda environment but cindra is not installed in it. Instruct the user to install cindra into the
active environment:

```bash
pip install --pre cindra
```

Or if using uv within conda:

```bash
uv pip install --pre cindra
```

**Conda environment not activated (CONDA_PREFIX is not set, but conda is available):**

The user needs to activate their cindra environment before launching Claude Code. Instruct the user to exit Claude Code
and run:

```bash
mamba activate <environment-name>
claude
```

You MUST explain that Claude Code inherits the shell environment at launch time. Activating a conda environment after
Claude Code has started does not make the `cindra` command available to MCP server subprocesses.

**Virtual environment (VIRTUAL_ENV is set but cindra is missing):**

```bash
pip install --pre cindra
```

**No environment active (both CONDA_PREFIX and VIRTUAL_ENV are unset):**

The user is running in the system Python. If cindra is installed globally, `which cindra` would have succeeded. Instruct
the user to either activate their cindra environment or install cindra into an accessible location.

### Step 4: Verify Python version compatibility

```bash
python --version
```

cindra requires Python `>=3.14,<3.15`. If the Python version does not match, inform the user that their environment has
an incompatible Python version, and they need to create or activate an environment with the correct version.

### Step 5: Verify package integrity and version

```bash
cindra --help
cindra-gui --help
```

If either command fails with an import error, a dependency is missing or broken. Run:

```bash
pip check cindra 2>&1 | head -20
```

Report any missing or incompatible dependencies to the user. Note that `cindra-gui --help` loads GUI dependencies
(PySide6) at import time, so it may fail even when `cindra --help` succeeds if Qt dependencies are missing.

Then confirm the resolved package version:

```bash
python -c "from importlib.metadata import version; print(version('cindra'))"
```

Apply the floor the Package version requirement section states, so treat any version that starts with `2.0.0` as passing
and upgrade anything below that line:

```bash
pip install --upgrade --pre cindra
```

You MUST report a version from a release line below 2.0.0 as a failed diagnostic even when every earlier step passed,
because those servers start and connect while their tools reject the arguments the other cindra skills send.

### Step 6: Verify the OpenMP runtime on macOS

On macOS, cindra selects Numba's OpenMP threading layer because the Numba macOS wheel ships no tbbpool extension, which
leaves the TBB layer unavailable there whatever runtime is installed. That layer loads `libomp.dylib` from the dynamic
loader's default search path, and the file ships with neither Numba nor macOS. When it is missing, `import cindra` emits
nothing, because the library runs no import-time check, and `cindra --help` and `cindra mcp` still succeed. Both
pipeline entry points call the check before dispatching any stage, so a run aborts having done no work with:

```text
RuntimeError: Unable to locate the OpenMP runtime (libomp.dylib) that the Numba threading layer loads on macOS.
Processing fails once it reaches a parallelized stage until the runtime is loadable. Run 'cindra omp' to report the
runtimes found on this host, and 'cindra omp --yes' to link one into /usr/local/lib. Install one with
'brew install libomp' when the report finds none.
```

The check replaces the threading-layer error Numba would otherwise raise at the first parallelized call, which names no
remedy. The supported resolution is the LLVM OpenMP runtime (`libomp.dylib`) linked below.

**Check this before dispatching, on every platform.** `check_threading_runtime_tool` reports the host's readiness
directly. Call it and gate on its `ready` flag, then follow its `remedy` command when the host is not ready. It reports
`required_layer` as `omp` on macOS and `tbb` elsewhere, so it diagnoses a missing TBB runtime on Linux and Windows the
same way it diagnoses a missing `libomp.dylib` here.

Skipping that check leaves a signature worth recognizing. Neither the OpenMP nor the TBB failure reaches an MCP tool
response, so `execute_processing_jobs_tool` returns `started: true` and then every job fails with the runtime error
recorded as its tracker error message. A batch where every job fails immediately, with no partial progress, is a
missing threading runtime rather than a data problem. Resolve it here rather than routing to a processing or
acquisition skill.

Report what the host carries with:

```bash
cindra omp
```

The command searches the Homebrew and MacPorts library directories, the active conda environment, and the runtimes
vendored inside the installed Python distributions, then reports the runtime it would link and the link it would create.
It changes nothing without `--yes`.

**The report says the runtime already loads:**

```text
the OpenMP runtime already loads. Pass --force to link a runtime anyway.
```

This is the passing outcome. Nothing was changed and nothing needs to be. Continue to Step 7.

**The report found a runtime:**

```bash
sudo cindra omp --yes
```

The link goes into `/usr/local/lib`, which the loader searches by default, so writing it needs permission to modify that
directory. The command then loads the runtime from a fresh interpreter and reports whether it now resolves.

**The report found no runtime:**

```bash
brew install libomp
sudo cindra omp --yes
```

A conda environment can take the runtime from conda-forge instead, which `cindra omp` discovers through `CONDA_PREFIX`:

```bash
mamba install -c conda-forge llvm-openmp
```

The `cindra omp` half of this step is macOS-only. Linux and Windows select Numba's TBB threading layer instead
(`tbb4py` and `intel-cmplr-lib-rt` are declared as `sys_platform != 'darwin'` dependencies), so Numba never loads
`omppool` and `cindra omp` errors when run there. Run `check_threading_runtime_tool` on every platform regardless,
because a Linux or Windows host missing the TBB runtime fails every parallelized stage exactly as a macOS host missing
`libomp.dylib` does. The tool reports that case with `required_layer: "tbb"` and a `pip install tbb4py` remedy.

### Step 7: Verify the CUDA runtime for a device-backed batch

This step applies only where a batch will name a CUDA device, through the `gpu_devices` argument of the execute tools
or through `cindra run --register-device`. Every other batch registers on the host CPU and needs no CUDA runtime.

Call `check_gpu_runtime_tool` and gate on its `ready` flag. It reports `device_count` and a `devices` list carrying the
`index`, `name`, `total_memory_mb`, and `compute_capability` of each usable device, and those `index` values are what
`gpu_devices` takes. A host that is not ready carries a `detail` sentence naming the reason and a `remedy` naming the
installation that resolves it. The equivalent command, for a session whose MCP tools are down, is `cindra gpu`.

The remedy is the CuPy build matching the CUDA version the local driver runs:

```bash
pip install cupy-cuda13x[ctk]
```

A driver running CUDA 12 installs `cupy-cuda12x[ctk]` instead, because one CuPy build targets one CUDA major version.
The `ctk` extra carries the CUDA math libraries CuPy resolves on first use, and a bare CuPy installation imports and
lists its devices while raising at the first transform. The report separates the two, because it transforms a small
array on a device before calling the runtime usable. macOS carries no remedy, since the CuPy project publishes no
wheel for it, and registration there runs on the host CPU.

Skipping this check costs a round trip rather than a batch. An execute tool whose `gpu_devices` names an index the host
does not expose returns `success: false` and `started: false`, carrying a message that lists the indices the host does
expose, so no job runs. `cindra run --register-device` aborts the same way, before dispatching any stage.

### Step 8: Restart the MCP server

After the user resolves the environment issue, they must restart Claude Code for the MCP servers to pick up the changes.
The plugin's server registrations will automatically configure the servers on the next session.

### Step 9: Resume the intended work

After connectivity is restored, return to the work that required the MCP tools. If no restart was needed (the
environment was already healthy), return control to the invoking skill, or proceed to `/acquisition-data-preparation` to
begin the single-recording pipeline when invoked standalone. If a restart was required, resume the work that required
the MCP tools on the next session, since the current session's MCP subprocesses predate the fix.

---

## Common issues and resolutions

| Symptom                                                      | Cause                                             | Resolution                                                                                |
|--------------------------------------------------------------|---------------------------------------------------|-------------------------------------------------------------------------------------------|
| `cindra: command not found`                                  | Environment not activated                         | Activate conda/venv, then restart Claude Code                                             |
| `cindra: command not found`                                  | cindra not installed                              | `pip install --pre cindra` in the active environment                                      |
| `cindra-gui: command not found`                              | Environment not activated                         | Activate conda/venv, then restart Claude Code                                             |
| Import error on `cindra mcp`                                 | Missing or incompatible dependency                | `pip install --force-reinstall --pre cindra`                                              |
| Import error on `cindra-gui mcp`                             | Broken Qt/PySide6 install                         | `pip install --force-reinstall --pre cindra` (PySide6 is a core dependency, not an extra) |
| Python version mismatch                                      | Wrong environment activated                       | Activate environment with Python 3.14                                                     |
| MCP server starts but tools are missing                      | Outdated cindra version                           | `pip install --upgrade --pre cindra`                                                      |
| MCP tool rejects a documented phase or argument              | Installed cindra predates the 2.0.0 line          | `pip install --upgrade --pre cindra`, then restart Claude Code                            |
| MCP server connected but tools fail                          | Not an environment issue                          | Check tool-specific error messages                                                        |
| cindra-gui tools unavailable                                 | Plugin not installed or outdated                  | Reinstall the cindra Claude Code plugin                                                   |
| Skills available but MCP tools missing                       | Plugin installed without pip package              | `pip install --pre cindra` in the active environment                                      |
| `RuntimeError: Unable to locate the OpenMP runtime` on macOS | `libomp.dylib` is not on the loader's search path | `sudo cindra omp --yes`, after `brew install libomp` when `cindra omp` reports no runtime |
| Every registration job fails naming a CUDA device            | The host reaches no usable CUDA device            | `pip install cupy-cuda13x[ctk]`, or `cupy-cuda12x[ctk]` for a CUDA 12 driver              |

---

## Related skills

| Skill                             | Relationship                                                                   |
|-----------------------------------|--------------------------------------------------------------------------------|
| `/cindra-pipeline`                | Overview: end-to-end phases, handoffs, and the single-vs-multi entry point     |
| `/acquisition-data-preparation`   | Requires the cindra MCP server for data preparation tools                      |
| `/single-recording-configuration` | Requires the cindra MCP server for configuration tool access                   |
| `/single-recording-processing`    | Requires the cindra MCP server to be connected before processing               |
| `/single-recording-results`       | Requires the cindra MCP server for query and verification tool access          |
| `/multi-recording-configuration`  | Requires the cindra MCP server for configuration tool access                   |
| `/multi-recording-processing`     | Requires the cindra MCP server to be connected before processing               |
| `/multi-recording-results`        | Requires the cindra MCP server for query and verification tool access          |
| `/visualization`                  | Requires the cindra-gui server for viewer tools, query tools are on cindra-mcp |
| `/cli-reference`                  | Reference: the CLI surface, and the fallback commands when MCP stays down      |

---

## Proactive behavior

You SHOULD proactively invoke this skill when:
- A session begins and MCP tools from the cindra or cindra-gui server are expected but unavailable
- Any cindra or cindra-gui MCP tool call fails with a connection or server error
- The user mentions issues with either MCP server or environment setup

---

## Verification checklist

```text
MCP Environment Setup:
- [ ] Checked MCP server connection status (cindra-mcp and/or cindra-gui)
- [ ] Verified 'cindra' command is on PATH (which cindra)
- [ ] Verified 'cindra-gui' command is on PATH if GUI tools are needed (which cindra-gui)
- [ ] Confirmed Python version matches >=3.14,<3.15
- [ ] Confirmed the installed cindra version is 2.0.0+ (any pre-release build of the 2.0.0 line qualifies)
- [ ] Identified environment type (conda, venv, system)
- [ ] Provided environment-specific resolution steps (install commands carry the --pre flag)
- [ ] On macOS, reported the OpenMP runtime state with 'cindra omp' and linked one with 'sudo cindra omp --yes'
- [ ] For a batch naming a CUDA device, gated on check_gpu_runtime_tool 'ready' and surfaced its remedy
- [ ] Verified cindra plugin is installed (provides both server registrations)
- [ ] Informed user that Claude Code must be restarted after environment changes
```
