---
name: cindra-mcp-environment-setup
description: >-
  Diagnoses and resolves cindra and cindra-gui MCP server connectivity issues. Covers environment
  verification, command availability, Python version checks, dependency validation, and conda/pip/uv
  environment configuration. Use when MCP tools are unavailable, when either MCP server fails to
  start, when the user reports connection issues, or when starting a session that requires MCP tools.
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
- Environment-specific guidance for conda, pip, and uv workflows

**Does not cover:**
- MCP tool usage for data processing (see `/single-recording-processing`, `/multi-recording-processing`)
- MCP tool usage for configuration (see `/single-recording-configuration`, `/multi-recording-configuration`)
- MCP tool usage for visualization (see `/visualization`)
- cindra package development or contribution workflows

---

## Architecture

cindra provides two separate MCP servers, each accessed through its own CLI entry point defined in
`pyproject.toml`:

```toml
[project.scripts]
cindra = "cindra.interface.cli:cindra_cli"
cindra-gui = "cindra.interface.gui_cli:cindra_gui"
```

| Server       | CLI command      | Purpose                                                     |
|--------------|------------------|-------------------------------------------------------------|
| `cindra-mcp` | `cindra mcp`     | Headless processing: discovery, configuration, batch jobs   |
| `cindra-gui` | `cindra-gui mcp` | GUI viewers and data querying (ROI, registration, tracking) |

Both servers accept a `--transport` option (defaults to `stdio`). The cindra Claude Code plugin
registers both servers in its `plugin.json`:

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

When the plugin is installed, Claude Code automatically discovers and starts both servers. The
`cindra` and `cindra-gui` commands must be on PATH when Claude Code starts. This means the Python
environment where cindra is installed must be active before launching Claude Code.

### Dual-distribution model

cindra's Claude Code integration is split across two distribution channels:

| Component                                        | Distributed via           | What it provides                                                        |
|--------------------------------------------------|---------------------------|-------------------------------------------------------------------------|
| Skills (`/single-recording-processing`, etc.)    | cindra Claude Code plugin | Skill files that guide agents through workflows                         |
| MCP server registrations                         | cindra Claude Code plugin | `plugin.json` mcpServers entries that register servers with Claude Code |
| MCP server code (`cindra mcp`, `cindra-gui mcp`) | cindra pip package        | The actual CLI commands and server implementations                      |

Installing the plugin alone registers the MCP servers and makes skills available, but the servers
will fail to start because the `cindra` and `cindra-gui` CLI commands are not present. The pip
package must also be installed in the active Python environment for the MCP servers to function.

This is the most common cause of MCP failures after initial setup: the plugin is installed but the
pip package is not, or the pip package is installed in a different Python environment than the one
active when Claude Code launches.

---

## Diagnostic workflow

You MUST follow these steps in order when MCP tools are unavailable or a server fails to start.
Apply these steps to whichever server is affected (`cindra` for the headless server, `cindra-gui`
for the GUI server). If both are affected, diagnose them in sequence.

### Step 1: Check MCP server status

Use the `/mcp` slash command or inspect available tools to determine whether the affected MCP
server is connected. If connected, the issue is not environmental — investigate tool-specific
errors instead.

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

The user has an active conda environment but cindra is not installed in it. Instruct the user to
install cindra into the active environment:

```bash
pip install cindra
```

Or if using uv within conda:

```bash
uv pip install cindra
```

**Conda environment not activated (CONDA_PREFIX is not set, but conda is available):**

The user needs to activate their cindra environment before launching Claude Code. Instruct the
user to exit Claude Code and run:

```bash
mamba activate <environment-name>
claude
```

You MUST explain that Claude Code inherits the shell environment at launch time. Activating a
conda environment after Claude Code has started does not make the `cindra` command available to
MCP server subprocesses.

**Virtual environment (VIRTUAL_ENV is set but cindra is missing):**

```bash
pip install cindra
```

**No environment active (both CONDA_PREFIX and VIRTUAL_ENV are unset):**

The user is running in the system Python. If cindra is installed globally, `which cindra` would
have succeeded. Instruct the user to either activate their cindra environment or install cindra
into an accessible location.

### Step 4: Verify Python version compatibility

```bash
python --version
```

cindra requires Python `>=3.14,<3.15`. If the Python version does not match, inform the user that
their environment has an incompatible Python version, and they need to create or activate an
environment with the correct version.

### Step 5: Verify package integrity

```bash
cindra --help
cindra-gui --help
```

If either command fails with an import error, a dependency is missing or broken. Run:

```bash
pip check cindra 2>&1 | head -20
```

Report any missing or incompatible dependencies to the user. Note that `cindra-gui --help` loads
GUI dependencies (PySide6) at import time, so it may fail even when `cindra --help` succeeds if
Qt dependencies are missing.

### Step 6: Verify OpenMP runtime on macOS

On macOS, cindra selects Numba's OpenMP threading layer because `tbb4py` has no Apple Silicon
wheel. Numba's `omppool.cpython-314-darwin.so` dynamically loads `libomp.dylib` at import time.
This library is not part of macOS or Apple's clang toolchain and must be provided by the active
Python environment. When `libomp.dylib` is not resolvable, `import cindra` (and therefore
`cindra --help` and `cindra mcp`) fails with:

```text
ValueError: No threading layer could be loaded.
HINT:
Intel TBB is required, try:
$ conda/pip install tbb
```

The error message mentions TBB, but on macOS the correct runtime is OpenMP. Run:

```bash
python -c "from numba.np.ufunc import omppool"
```

If this command fails with `Library not loaded: @rpath/libomp.dylib`, guide the user through the
resolution that matches their environment:

**Conda environment (recommended):**

```bash
conda install -c conda-forge llvm-openmp
```

or

```bash
mamba install -c conda-forge llvm-openmp
```

This installs `libomp.dylib` into `$CONDA_PREFIX/lib/`, where Numba's compiled omppool extension
locates it via its build-time rpath.

**pip-only virtual environment:**

Install libomp via Homebrew, then make it visible to the venv's Python. Either symlink:

```bash
brew install libomp
ln -s "$(brew --prefix libomp)/lib/libomp.dylib" "${VIRTUAL_ENV}/lib/libomp.dylib"
```

or export `DYLD_LIBRARY_PATH` before launching Claude Code:

```bash
export DYLD_LIBRARY_PATH="$(brew --prefix libomp)/lib:${DYLD_LIBRARY_PATH}"
```

You MUST skip this step on Linux and Windows — the OpenMP runtime is present by default (via
`libgomp` on Linux, Intel OpenMP bundled with the Intel compiler runtime on Windows) and this
diagnostic does not apply.

### Step 7: Restart the MCP server

After the user resolves the environment issue, they must restart Claude Code for the MCP servers
to pick up the changes. The plugin's server registrations will automatically configure the
servers on the next session.

On macOS, if the resolution was a `DYLD_LIBRARY_PATH` export, the export MUST be in effect in the
shell that launches Claude Code — subsequently activating it from within Claude Code does not
propagate to already-spawned MCP server subprocesses.

---

## Common issues and resolutions

| Symptom                                                   | Cause                                                       | Resolution                                                                                                           |
|-----------------------------------------------------------|-------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| `cindra: command not found`                               | Environment not activated                                   | Activate conda/venv, then restart Claude Code                                                                        |
| `cindra: command not found`                               | cindra not installed                                        | `pip install cindra` in the active environment                                                                       |
| `cindra-gui: command not found`                           | Environment not activated                                   | Activate conda/venv, then restart Claude Code                                                                        |
| Import error on `cindra mcp`                              | Missing or incompatible dependency                          | `pip install --force-reinstall cindra`                                                                               |
| Import error on `cindra-gui mcp`                          | Broken Qt/PySide6 install                                   | `pip install --force-reinstall cindra` (PySide6 is a core dependency, not an extra)                                  |
| Python version mismatch                                   | Wrong environment activated                                 | Activate environment with Python 3.14                                                                                |
| MCP server starts but tools are missing                   | Outdated cindra version                                     | `pip install --upgrade cindra`                                                                                       |
| MCP server connected but tools fail                       | Not an environment issue                                    | Check tool-specific error messages                                                                                   |
| cindra-gui tools unavailable                              | Plugin not installed or outdated                            | Reinstall the cindra Claude Code plugin                                                                              |
| Skills available but MCP tools missing                    | Plugin installed without pip package                        | `pip install cindra` in the active environment                                                                       |
| `ValueError: No threading layer could be loaded` on macOS | `libomp.dylib` not on rpath (Apple ships no OpenMP runtime) | `conda install -c conda-forge llvm-openmp` (conda) or `brew install libomp` + symlink/`DYLD_LIBRARY_PATH` (pip-only) |

---

## Related skills

| Skill                             | Relationship                                                        |
|-----------------------------------|---------------------------------------------------------------------|
| `/acquisition-data-preparation`   | Requires the cindra MCP server for data preparation tools           |
| `/single-recording-configuration` | Requires the cindra MCP server for configuration tool access        |
| `/single-recording-processing`    | Requires the cindra MCP server to be connected before processing    |
| `/multi-recording-configuration`  | Requires the cindra MCP server for configuration tool access        |
| `/multi-recording-processing`     | Requires the cindra MCP server to be connected before processing    |
| `/visualization`                  | Requires the cindra-gui MCP server for viewer and query tool access |

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
- [ ] Identified environment type (conda, venv, system)
- [ ] Provided environment-specific resolution steps
- [ ] On macOS, verified 'libomp.dylib' is resolvable (python -c "from numba.np.ufunc import omppool")
- [ ] Verified cindra plugin is installed (provides both server registrations)
- [ ] Informed user that Claude Code must be restarted after environment changes
```
