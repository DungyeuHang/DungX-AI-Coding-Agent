# Local-first Autonomous AI Software Engineer

`agent` is a small, local-first coding-agent MVP. It analyzes a project on disk,
builds structured context, asks a pluggable provider for a plan and file
operations, runs detected validation commands, repairs failures up to a bounded
limit, and reviews the resulting diff.

The target project is always an explicit local directory (or the current
directory). The agent does not require GitHub or any other remote service, never
commits or pushes automatically, and confines file operations to the target.

## Installation

Python 3.11+ is required. From this repository:

```powershell
python -m pip install -e .
```

The project has no runtime dependencies. Tests use the Python standard library.

## Usage

Analyze a local project:

```powershell
agent analyze --project "D:\Projects\my-app"
```

Run a task in the current directory or another project:

```powershell
agent run "Add input validation"
agent run --project "D:\Projects\my-app" "Fix the failing login test"
```

The default `mock` provider is deliberately offline and makes no source
changes. It is useful for exercising the workflow and safety boundaries. To
enable model-backed planning and edits, configure an OpenAI-compatible provider:

```powershell
$env:AGENT_PROVIDER = "openai"
$env:OPENAI_API_KEY = "..."
$env:OPENAI_MODEL = "gpt-4.1-mini"
agent run --project "D:\Projects\my-app" "Implement feature X"
```

You can also use `python -m local_agent` instead of the installed `agent`
command. Use `--max-iterations N` to bound autonomous repair attempts and
`--validation "python -m unittest"` to add an explicit validation command.

## Architecture

```text
CLI -> Orchestrator
       |-- RepositoryAnalyzer -> ProjectContext
       |-- Planner -> Plan
       |-- CodingAgent -> sandboxed FileSystem
       |-- CommandRunner -> ValidationResult
       |-- FailureAnalyzer -> FailureAnalysis
       |-- Reviewer -> ReviewResult
       `-- AIProvider (mock or OpenAI-compatible)
```

The `local_agent` package keeps each responsibility small:

- `config.py` centralizes environment and CLI configuration.
- `analyzer.py` discovers project metadata without sending the whole repository
  to a provider.
- `filesystem.py` enforces the target-project boundary and protects Git and
  common secret files.
- `git.py` provides read-only local status, diff, branch, and log inspection.
- `commands.py` executes discovered commands without a shell and captures
  command, exit code, output, and duration.
- `providers.py`, `planner.py`, `coding_agent.py`, `failure.py`, and
  `reviewer.py` implement the provider and development workflow.
- `orchestrator.py` coordinates a bounded plan/implement/validate/repair/review
  loop.

## Safety and current limitations

File paths are resolved beneath the selected project directory. `.git` is never
deleted, destructive commands are rejected, secrets are not read by analysis,
and unrelated pre-existing Git changes are reported and left untouched. The
agent never commits, pushes, resets, or silently discards changes.

The mock provider does not invent code changes. The OpenAI provider requires a
working API key and a model that returns the documented JSON shapes; it has not
been tested here without credentials. Command discovery is intentionally
conservative, and language-specific semantic indexing, interactive approvals,
parallel agents, CI integration, dashboard/IDE integrations, long-term memory,
and remote Git workflows are not included in this MVP.

## Roadmap

1. Add semantic code search and incremental project indexing.
2. Add more providers and provider capability negotiation.
3. Add approval policies and richer command permissions.
4. Add parallel specialist agents, CI integration, and project memory.
5. Add optional Git commits, pull requests, dashboard, and IDE integrations.
