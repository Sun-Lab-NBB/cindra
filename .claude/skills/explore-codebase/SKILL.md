---
name: exploring-codebase
description: >-
  Performs in-depth codebase exploration at the start of a coding session. Builds comprehensive
  understanding of project structure, architecture, key components, and patterns. Use when starting
  a new session, when asked to understand or explore the codebase, when asked "what does this project
  do", when exploring unfamiliar code, or when the user asks about project structure or architecture.
---

# Codebase Exploration

Performs thorough codebase exploration to build deep understanding before coding work begins.

---

## Exploration Approach

Use the Task tool with `subagent_type: Explore` to investigate the codebase. Focus on understanding:

1. **Project purpose and structure** - README, documentation, directory layout
2. **Architecture** - Main components, how they interact, communication patterns
3. **Core code** - Key classes, data models, utilities
4. **Configuration** - How the project is configured and customized
5. **Dependencies** - External libraries and integrations
6. **Patterns and conventions** - Coding style, naming conventions, design patterns

Adapt exploration depth based on project size and complexity. For small projects, a quick overview
suffices. For large projects, explore systematically.

---

## Guiding Questions

Answer these questions during exploration:

### Architecture
- What is the main entry point or controller?
- How do components communicate (IPC, APIs, events)?
- What external systems does this integrate with?

### Patterns
- What naming conventions are used?
- What design patterns appear (factories, dataclasses, protocols)?
- How is configuration managed?

### Structure
- Where is the core business logic?
- Where are tests located?
- What build/tooling configuration exists?

---

## Output Format

Provide a structured summary including:

- Project purpose (1-2 sentences)
- Key components table
- Important files list with paths
- Notable patterns or conventions
- Any areas of complexity or concern

### Example Output

```markdown
## Project Purpose

Provides a reimplemented suite2p library for neural imaging analysis with multi-day cell tracking capabilities for the
Sun Lab at Cornell University.

## Key Components

| Component           | Location                         | Purpose                                           |
|---------------------|----------------------------------|---------------------------------------------------|
| Pipeline            | src/sl_suite2p/pipeline.py       | Main single-day processing pipeline orchestration |
| Multi-day Tracking  | src/sl_suite2p/multiday/         | Cross-session cell tracking and alignment         |
| Registration        | src/sl_suite2p/registration/     | Image registration and motion correction          |
| Detection           | src/sl_suite2p/detection/        | Cell detection algorithms                         |
| Extraction          | src/sl_suite2p/extraction/       | Signal extraction from detected cells             |
| Classification      | src/sl_suite2p/classification/   | Cell classification and filtering                 |
| Configuration       | src/sl_suite2p/configuration/    | Pipeline configuration management                 |
| MCP Server          | src/sl_suite2p/mcp/              | AI agent integration via MCP                      |
| CLI                 | src/sl_suite2p/cli.py            | Command-line interface entry points               |

## Important Files

- `src/sl_suite2p/pipeline.py` - Main pipeline orchestration
- `src/sl_suite2p/single_day.py` - Single-day processing logic
- `src/sl_suite2p/multi_day.py` - Multi-day tracking pipeline
- `src/sl_suite2p/configuration/` - Configuration dataclasses
- `pyproject.toml` - Project configuration and dependencies

## Notable Patterns

- Single-day and multi-day pipeline separation
- Configuration dataclasses for pipeline parameters
- Numba-accelerated computation
- MCP server for AI agent integration
- MyPy strict mode with full type annotations

## Areas of Concern

- Large refactoring effort from original suite2p codebase
- Integration with original suite2p algorithms
- Multi-day tracking complexity
```

---

## Usage

Invoke at session start to ensure full context before making changes. Prevents blind modifications
and ensures understanding of existing patterns.
