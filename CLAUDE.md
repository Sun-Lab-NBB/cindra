# Claude Code Instructions

## Session Start Behavior

At the beginning of each coding session, before making any code changes, you should build a comprehensive
understanding of the codebase by invoking the `/explore-codebase` skill.

This ensures you:
- Understand the project architecture before modifying code
- Follow existing patterns and conventions
- Don't introduce inconsistencies or break integrations

## Style Guide Requirements

You MUST invoke `/sun-lab-style` and read the appropriate guide before performing ANY of the following tasks:

| Task                              | Guide to Read      |
|-----------------------------------|--------------------|
| Writing or modifying Python code  | PYTHON_STYLE.md    |
| Writing or modifying README files | README_STYLE.md    |
| Writing git commit messages       | COMMIT_STYLE.md    |
| Writing or modifying skill files  | SKILL_STYLE.md     |

This is non-negotiable. The skill contains verification checklists that you MUST complete before submitting any work.
Failure to read the appropriate guide results in style violations.

## Cross-Referenced Library Verification

Sun Lab projects often depend on other `ataraxis-*` or `sl-*` libraries. These libraries may be stored locally in the
same parent directory as this project (`/home/cyberaxolotl/Desktop/GitHubRepos/`).

**Before writing code that interacts with a cross-referenced library, you MUST:**

1. **Check for local version**: Look for the library in the parent directory (e.g., `../sl-shared-assets/`,
   `../sl-experiment/`).

2. **Compare versions**: If a local copy exists, compare its version against the latest release or main branch on
   GitHub:
   - Read the local `pyproject.toml` to get the current version
   - Use `gh api repos/Sun-Lab-NBB/{repo-name}/releases/latest` to check the latest release
   - Alternatively, check the main branch version on GitHub

3. **Handle version mismatches**: If the local version differs from the latest release or main branch, notify the user
   with the following options:
   - **Use online version**: Fetch documentation and API details from the GitHub repository
   - **Update local copy**: The user will pull the latest changes locally before proceeding

4. **Proceed with correct source**: Use whichever version the user selects as the authoritative reference for API
   usage, patterns, and documentation.

**Why this matters**: Skills and documentation may reference outdated APIs. Always verify against the actual library
state to prevent integration errors.

## Available Skills

| Skill                     | Description                                                         |
|---------------------------|---------------------------------------------------------------------|
| `/explore-codebase`       | Perform in-depth codebase exploration at session start              |
| `/sun-lab-style`          | Apply Sun Lab coding conventions (REQUIRED for all code changes)    |
| `/single-day-processing`  | Guide agents through single-day suite2p processing using MCP tools  |
| `/single-day-config`      | Complete reference for single-day pipeline configuration parameters |
| `/multi-day-processing`   | Guide agents through multi-day cell tracking using MCP tools        |
| `/multi-day-config`       | Complete reference for multi-day pipeline configuration parameters  |

## Project Context

This is **sl-suite2p**, a Python library for neural imaging analysis in the Sun Lab at Cornell University. The library
is a reimplementation of the original suite2p library with enhanced documentation, modern Python support, and a new
multi-day cell tracking pipeline.

### Key Areas

| Directory                          | Purpose                                         |
|------------------------------------|-------------------------------------------------|
| `src/sl_suite2p/`                  | Main library source code                        |
| `src/sl_suite2p/registration/`     | Image registration and motion correction        |
| `src/sl_suite2p/detection/`        | Cell detection algorithms                       |
| `src/sl_suite2p/extraction/`       | Signal extraction from detected cells           |
| `src/sl_suite2p/classification/`   | Cell classification and filtering               |
| `src/sl_suite2p/multiday/`         | Multi-day cell tracking pipeline                |
| `src/sl_suite2p/configuration/`    | Pipeline configuration dataclasses              |
| `src/sl_suite2p/mcp/`              | MCP server for AI agent integration             |
| `src/sl_suite2p/gui/`              | GUI components (from original suite2p)          |
| `src/sl_suite2p/io/`               | Input/output utilities                          |
| `notebooks/`                       | Example notebooks for single-day and multi-day  |
| `tests/`                           | Test suite                                      |

### Architecture

- **Single-day pipeline**: `pipeline.py` and `single_day.py` orchestrate the main processing workflow (registration,
  detection, extraction, classification)
- **Multi-day pipeline**: `multi_day.py` and `multiday/` handle cross-session cell tracking and alignment
- **Configuration**: Dataclasses in `configuration/` define pipeline parameters
- **CLI**: `cli.py` provides command-line entry points for all pipeline operations
- **MCP Server**: `mcp/` exposes pipeline functionality for AI agent integration

### Code Standards

- MyPy strict mode with full type annotations
- Google-style docstrings
- 120 character line limit
- See `/sun-lab-style` for complete conventions

### Workflow Guidance

**Modifying the single-day pipeline:**

1. Review `src/sl_suite2p/pipeline.py` for orchestration logic
2. Check the relevant module (`registration/`, `detection/`, `extraction/`, `classification/`)
3. Follow existing patterns for configuration and processing
4. Update CLI commands in `cli.py` if adding new functionality

**Modifying the multi-day pipeline:**

1. Review `src/sl_suite2p/multi_day.py` for pipeline orchestration
2. Check `src/sl_suite2p/multiday/` for cross-session tracking logic
3. Understand the relationship between single-day outputs and multi-day inputs

**Adding MCP tools:**

1. Review existing tools in `src/sl_suite2p/mcp/`
2. Follow MCP tool docstring conventions from PYTHON_STYLE.md
3. Keep tool responses concise and information-dense
