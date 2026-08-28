# Claude Code Instructions

## Session start behavior

At the beginning of each coding session, before making any code changes, you MUST build a comprehensive understanding of
the codebase by invoking the `/explore-codebase` skill.

## Style guide compliance

You MUST invoke the appropriate style skill before performing ANY of the following tasks:

| Task                                          | Skill to invoke    |
|-----------------------------------------------|--------------------|
| Writing or modifying Python code              | `/python-style`    |
| Writing or modifying README files             | `/readme-style`    |
| Writing or modifying skill files or this file | `/skill-design`    |
| Writing or modifying pyproject.toml           | `/pyproject-style` |
| Writing or modifying tox.ini                  | `/tox-config`      |
| Writing or modifying Sphinx docs files        | `/api-docs`        |
| Creating or verifying project structure       | `/project-layout`  |
| Committing local changes                      | `/commit`          |

Each skill contains a verification checklist that you MUST complete before submitting any work.

## Cross-referenced library verification

cindra depends on several `ataraxis-*` libraries. These libraries may be stored locally in the same parent directory as
this project, reachable as `../` from the repository root.

**Before writing code that interacts with a cross-referenced library, you MUST:**

1. **Check for local version**: Look for the library in the parent directory (e.g., `../ataraxis-time/`,
   `../ataraxis-base-utilities/`, `../ataraxis-data-structures/`).

2. **Compare versions**: If a local copy exists, compare its version against the latest release or main branch on
   GitHub:
   - Read the local `pyproject.toml` to get the current version
   - Use `gh api repos/Sun-Lab-NBB/{repo-name}/releases/latest` to check the latest release
   - Alternatively, check the main branch version on GitHub

3. **Handle version mismatches**: If the local version differs from the latest release or main branch, notify the user
   with the following options:
   - **Use online version**: Fetch documentation and API details from the GitHub repository
   - **Update local copy**: The user will pull the latest changes locally before proceeding

4. **Proceed with correct source**: Use whichever version the user selects as the authoritative reference for API usage,
   patterns, and documentation.

**Why this matters**: Skills and documentation may reference outdated APIs. Always verify against the actual library
state to prevent integration errors.

## Available skills

For cindra pipeline work, `/cindra-pipeline` is the end-to-end orchestration entry point that routes to the
phase-specific skills.

**Cindra plugin skills** (`plugins/cindra/skills/`):

| Skill                             | Description                                                     |
|-----------------------------------|-----------------------------------------------------------------|
| `/cindra-pipeline`                | Orchestrates the end-to-end pipeline and opens a cindra session |
| `/single-recording-processing`    | Orchestrates single-recording batch processing via MCP          |
| `/multi-recording-processing`     | Orchestrates multi-recording batch processing via MCP           |
| `/single-recording-configuration` | Documents single-recording configuration parameters and tools   |
| `/multi-recording-configuration`  | Documents multi-recording configuration parameters and tools    |
| `/single-recording-results`       | Documents single-recording output data formats and verification |
| `/multi-recording-results`        | Documents multi-recording output data formats and verification  |
| `/acquisition-data-preparation`   | Prepares raw imaging data and acquisition parameter files       |
| `/visualization`                  | Launches and manages cindra GUI viewers for visual inspection   |
| `/cli-reference`                  | Documents the human-facing cindra and cindra-gui CLI commands   |
| `/cindra-mcp-environment-setup`   | Diagnoses and resolves MCP server connectivity issues           |

**Ataraxis automation plugin skills** (external, shared across projects):

| Skill                   | Description                                                       |
|-------------------------|-------------------------------------------------------------------|
| `/explore-codebase`     | Performs in-depth codebase exploration at session start           |
| `/explore-dependencies` | Explores installed ataraxis dependency APIs for a live snapshot   |
| `/python-style`         | Applies Python coding conventions (REQUIRED for Python work)      |
| `/readme-style`         | Applies README conventions (REQUIRED for README work)             |
| `/pyproject-style`      | Applies pyproject.toml conventions (REQUIRED for pyproject.toml)  |
| `/tox-config`           | Applies tox.ini conventions (REQUIRED for tox.ini work)           |
| `/api-docs`             | Applies Sphinx documentation conventions (REQUIRED for docs work) |
| `/project-layout`       | Applies project directory layout conventions                      |
| `/skill-design`         | Applies skill and CLAUDE.md conventions (REQUIRED for this file)  |
| `/audit-correctness`    | Audits source for active and latent bugs                          |
| `/audit-facts`          | Fact-checks documentation against authoritative source            |
| `/audit-performance`    | Audits source for algorithmic, allocation, and dtype costs        |
| `/audit-project`        | Orchestrates all four audits and merges their findings            |
| `/audit-style`          | Audits files against the applicable style checklists              |
| `/commit`               | Stages changes and creates a style-compliant commit               |
| `/pr`                   | Drafts a style-compliant pull request summary                     |
| `/release`              | Drafts style-compliant release notes from merged PRs              |

## MCP server

cindra provides two MCP servers that expose neural imaging pipeline tools for agentic work. When working with this
project or its dependencies, prefer using available MCP tools over direct code execution when appropriate.

**Servers:**

| Server       | CLI command      | Purpose                                            |
|--------------|------------------|----------------------------------------------------|
| `cindra-mcp` | `cindra mcp`     | Data processing, configuration, discovery, results |
| `cindra-gui` | `cindra-gui mcp` | GUI viewer lifecycle management and state queries  |

**Guidelines for MCP usage:**

1. **Discover available tools**: At the start of a session, check which MCP servers are connected and what tools they
   provide. Use these tools when they offer functionality relevant to the current task.

2. **Prefer MCP for runtime operations**: For operations like batch processing orchestration, configuration generation,
   recording discovery, and result querying, use MCP tools rather than writing and executing Python code directly. MCP
   tools provide consistent, tested interfaces with proper resource management.

3. **Use MCP for cross-library operations**: When dependency libraries (e.g., `ataraxis-data-structures`,
   `ataraxis-time`) provide MCP servers, explore and use their tools for interacting with those libraries.

4. **Fall back to code when necessary**: Use direct code execution when no MCP tool exists for the required
   functionality, the task requires custom logic, or you are writing or modifying library source code.

## Distribution model

This project ships through two channels that both have to be installed before any MCP tool resolves. The library, both
CLIs, and both MCP server implementations live under `src/cindra/` and reach the user through PyPI. The Claude Code
assets live under `plugins/cindra/` in this same repository and reach Claude Code through the marketplace declared in
`.claude-plugin/marketplace.json`. That plugin's `.claude-plugin/plugin.json` points at its `skills/` directory and
carries the `mcpServers` registrations for `cindra mcp` and `cindra-gui mcp`.

When modifying a skill, edit the SKILL.md under `plugins/cindra/skills/` and bump `version` in
`plugins/cindra/.claude-plugin/plugin.json` once per branch. When modifying an MCP tool, edit the matching tool module
under `src/cindra/interface/`.

## Project context

This is **cindra**, a reimplementation of the [suite2p](https://github.com/MouseLand/suite2p) neural imaging processing
library with expanded documentation, optimized algorithms, modern Python 3.14 support, and a novel multi-recording ROI
tracking pipeline based on the [OSM manuscript](https://www.nature.com/articles/s41586-024-08548-w). The library
provides CLI and MCP server interfaces for agentic processing, and interactive GUIs for visualization of pipeline
outputs.

### Key areas

| Directory                    | Purpose                                                         |
|------------------------------|-----------------------------------------------------------------|
| `src/cindra/`                | Main library source code                                        |
| `src/cindra/orchestration/`  | Job model, worker allocation, batch execution engine, pipelines |
| `src/cindra/classification/` | Cell type classification (distinguishing cells from artifacts)  |
| `src/cindra/dataclasses/`    | Configuration and runtime data structures (YamlConfig-based)    |
| `src/cindra/detection/`      | ROI detection, tracking, and statistics computation             |
| `src/cindra/extraction/`     | Fluorescence trace extraction, neuropil subtraction, OASIS      |
| `src/cindra/gui/`            | Interactive PySide6/PyQtGraph viewers for pipeline outputs      |
| `src/cindra/interface/`      | CLI, MCP servers, and tool modules for user-facing entry points |
| `src/cindra/io/`             | TIFF loading, binary file management, multi-plane combination   |
| `src/cindra/pipelines/`      | Stage entry points the pipelines dispatch                       |
| `src/cindra/registration/`   | Motion correction, diffeomorphic registration, deformation      |
| `tests/`                     | Test suite (mirrors source module structure)                    |
| `docs/`                      | Sphinx API documentation source                                 |

The pipeline architecture and the recurring implementation patterns live in the imported reference file named at
the end of this document.

### Code standards

- MyPy type checking with full type annotations (`disallow_untyped_defs`, `warn_unused_ignores`)
- Google-style docstrings
- 120 character line limit
- Ruff for formatting and linting
- Python 3.14 only
- See `/python-style` for complete conventions

### Development commands

```bash
tox -e lint        # ruff format, ruff check, and mypy
tox -e stubs       # Generate the py.typed marker and the .pyi stub files
tox -e py314-test  # Run the pytest suite on Python 3.14
tox -e coverage    # Combine the test run's coverage data into an HTML report and apply the 100% gate
tox -e docs        # Build the Sphinx API documentation
tox -e deploy      # Upload the documentation built by 'docs' to the project's Netlify site
tox                # Full envlist: uninstall, export, lint, stubs, py314-test, coverage, docs, build, install
```

The `deploy` and `upload` tasks stay out of the envlist and are invoked manually for a release. The environment tasks
(`create`, `remove`, `provision`, `import`) target the `cindra_dev` mamba environment and are also manual.

### Testing

Tests use pytest with pytest-xdist for parallel execution (`-n logical --dist loadgroup`). Test files mirror the source
structure under `tests/` with a `_test.py` suffix, across `classification/`, `dataclasses/`, `detection/`,
`extraction/`, `gui/`, `interface/`, `io/`, `orchestration/`, `pipelines/`, and `registration/`.

The component map, CLI command reference, dependency table, and per-area workflow guidance live in an imported reference
file:

@.claude/cindra-reference.md
