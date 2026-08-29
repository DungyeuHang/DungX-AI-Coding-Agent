# Local-first Autonomous AI Software Engineer

`agent` is a local-first coding-agent MVP. It analyzes a project on disk,
selects task-relevant context, asks a pluggable provider for a plan and
structured patch changes, validates and applies those changes safely, runs
detected validation commands, repairs failures up to a bounded limit, and
reviews the resulting diff.

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

Gemini is supported without an additional SDK. The provider sends structured
JSON requests to the Gemini REST API and prefers unified patches for edits:

```powershell
$env:AGENT_PROVIDER = "gemini"
$env:GEMINI_API_KEY = "..."
$env:GEMINI_MODEL = "gemini-2.5-flash"
agent run --project "D:\Projects\my-app" --provider gemini "Implement feature X"
```

The model name is configurable with `--model` or `GEMINI_MODEL`. API keys are
read only from environment variables and are never printed.

Phase 3 context and retry controls are configurable with `AGENT_*` settings,
including `AGENT_MAX_CONTEXT_FILES`, `AGENT_MAX_CONTEXT_FILE_BYTES`,
`AGENT_MAX_CONTEXT_TOKENS`, and `AGENT_DEPENDENCY_DEPTH`,
`AGENT_PLANNING_CONTEXT_BYTES`, `AGENT_IMPLEMENTATION_CONTEXT_BYTES`,
`AGENT_REPAIR_CONTEXT_BYTES`, `AGENT_REVIEW_CONTEXT_BYTES`,
`AGENT_PROVIDER_MAX_RETRIES`, and `AGENT_MAX_RETRY_WAIT_SECONDS`. Set
`AGENT_METRICS=true` to include approximate request sizes, token counts,
duration, model, and success/failure metrics in the run report. Metrics never
include keys or source contents.

Phase 3.5 adds deterministic repository intelligence. The scanner builds a
typed local repository map with file metadata, stacks, frameworks, entry
points, tests, configuration, ignore/protected-path decisions, and lightweight
Python/JavaScript/TypeScript relationships. It reads locally and does not call
an AI provider:

```powershell
agent inspect --project "D:\Projects\my-app"
agent context --project "D:\Projects\my-app" "Add class management to the teacher dashboard"
```

`agent context` reports task relevance scores, explanations, dependency
expansion, exclusions, and context-budget usage. Dependency expansion defaults
to one hop and is configurable with `AGENT_DEPENDENCY_DEPTH`.

The official Antigravity managed-agent API is also available as a separate
provider. It uses the documented Gemini Interactions API and does not reuse
Antigravity CLI or browser credentials:

```powershell
$env:GEMINI_API_KEY = "..."
$env:GEMINI_MODEL = "gemini-3.7-flash"
agent run --project "D:\Projects\my-app" --provider antigravity "Implement feature X"
```

You can also use `python -m local_agent` instead of the installed `agent`
command. Use `--max-iterations N` to bound autonomous repair attempts and
`--validation "python -m unittest"` to add an explicit validation command.

## Local desktop UI

For machines where the terminal environment does not share the expected API-key
environment, launch the lightweight Tkinter UI:

```powershell
python -m local_agent.ui
# or, after installation:
agent-ui
```

The UI lets you choose a provider and model, enter a key in a masked field,
select a local project, test Gemini connectivity, run dry-runs, and require
approval before changes are applied. Keys are held only in the running process;
the UI passes them directly through runtime configuration to the provider and
does not write them to process environment variables, configuration files,
project files, logs, or reports. CLI usage continues to fall back to
`GEMINI_API_KEY` or `OPENAI_API_KEY` when no runtime key is supplied.

Use dry-run to inspect proposed changes without writing files, or approval mode
to pause immediately before applying validated changes:

```powershell
agent run --project "D:\Projects\my-app" --provider gemini --dry-run "Implement feature X"
agent run --project "D:\Projects\my-app" --provider gemini --approval always "Implement feature X"
```

## Architecture

```text
CLI -> Orchestrator
       |-- RepositoryAnalyzer -> ProjectContext
       |-- Planner -> Plan
       |-- CodingAgent -> sandboxed FileSystem
       |-- CommandRunner -> ValidationResult
       |-- FailureAnalyzer -> FailureAnalysis
       |-- Reviewer -> ReviewResult
       `-- AIProvider (mock, OpenAI, Gemini, or Antigravity API)
```

The `local_agent` package keeps each responsibility small:

- `config.py` centralizes environment and CLI configuration.
- `analyzer.py` discovers project metadata without sending the whole repository
  to a provider.
- `repository.py` builds the deterministic repository map and local file
  relationships used by inspection and context selection.
- `context.py` ranks candidate files by task keywords, path/content matches,
  tests, configuration, recent Git changes, and dependency relationships before
  provider calls, subject to file/byte/token budgets. React/TypeScript tasks
  additionally use router, layout, navigation, page, reverse-import, test, and
  package-dependency relationships; generic words receive lower weight.
- `filesystem.py` enforces the target-project boundary and protects Git and
  common secret files.
- `git.py` provides read-only local status, diff, branch, and log inspection.
- `commands.py` executes discovered commands without a shell and captures
  command, exit code, output, and duration. A bare `python`/`pytest` name that is
  not on `PATH` is executed via the running interpreter (`sys.executable`), so a
  missing console script cannot turn a test run into a silent no-op; the logical
  command is preserved for display and evidence fingerprints.
- `patching.py` validates and applies strict unified diffs in memory before any
  filesystem write.
- `providers.py`, `planner.py`, `coding_agent.py`, `failure.py`, and
  `reviewer.py` implement the provider and development workflow.
- `orchestrator.py` coordinates a bounded plan/implement/validate/repair/review
  loop.

## Semantic change-impact validation (Phase 4.17)

When `semantic_impact_analysis_enabled` is on, candidate validation selects which
tests to run from the import/reference graph instead of from filenames alone.

The pipeline is `changed files -> changed symbols -> import/reference graph ->
affected modules -> ranked validation targets -> confidence -> validation scope`.
It runs against the Phase 4.16 candidate tree, so it reasons about the proposed
edit before anything is written to the real repository.

- `indexing/ast_python_indexer.py` extracts symbols, imports, references and
  per-symbol body hashes using the standard library `ast` module. It emits the
  existing `SymbolDefinition` model rather than a parallel symbol model. A class
  is hashed over only the lines it owns directly, so editing one method reports
  that method, not its whole class.
- `semantic_impact.py` diffs symbols between the candidate and the workspace's
  frozen BASE snapshot, propagates impact backwards through reverse dependencies
  under explicit depth/node bounds, and ranks each candidate test.
- `evidence.py` records every executed command as structured evidence and decides
  whether a post-apply rerun can be replaced by a candidate result.

### Target ranking and confidence

Each test is assigned its single strongest evidence tier and carries a sentence
explaining that choice: `direct_symbol_match`, `direct_import_match`,
`call_graph_match`, `reverse_dependency_match`, `module_match`, `filename_match`,
then `broad_fallback`. The first four come from the graph; the rest are naming
heuristics and are always ranked below graph evidence. A symbol name defined in
more than three files is treated as ambiguous and stops counting as
reference-only evidence, so a bare `run` or `to_dict` does not associate
unrelated tests.

Confidence is `high` only with direct import/symbol evidence and good graph
coverage, `medium` for weaker graph evidence, and `low` for lexical-only
evidence or when any degradation was recorded. Scope follows confidence:
`targeted`, `expanded`, or `broad`.

### Uncertainty only ever widens validation

A parse failure, a non-Python file, a missing BASE snapshot, a star or dynamic
import, a removed public symbol, high fan-out, or a traversal bound being hit all
lower confidence, and lower confidence maps to a broader scope. `escalate_scope`
is the only way a scope is reassigned and it is monotone, so there is no code
path from "analysis failed" to "run fewer tests". A semantic analysis failure
falls back to the pre-existing lexical selection, never to no validation.

Because a skipped command counts as succeeded, `CandidateValidationReport.passed`
alone cannot distinguish "the selected tests passed" from "the test runner is not
installed". Use `has_test_evidence` for that; the model-facing feedback says so
explicitly when nothing ran.

### Candidate evidence reuse

With `reuse_candidate_validation_evidence` on, a post-apply targeted command is
dropped only when every assumption is re-verified against the authoritative tree:
identical command vector, a recorded pass (never a failure or a skip), identical
relevant file and symbol sets, an impact confidence meeting
`validation_confidence_threshold`, and a byte-identical content fingerprint over
those files. Any mismatch yields a machine-readable reason and the command reruns.
The mandatory full validation run still happens regardless, so this removes
duplicated work, not validation.

### Configuration

| Setting | Env | Default |
| --- | --- | --- |
| `semantic_impact_analysis_enabled` | `AGENT_SEMANTIC_IMPACT_ANALYSIS` | `false` |
| `max_impact_depth` | `AGENT_MAX_IMPACT_DEPTH` | `3` |
| `max_affected_symbols` | `AGENT_MAX_AFFECTED_SYMBOLS` | `200` |
| `max_affected_tests` | `AGENT_MAX_AFFECTED_TESTS` | `8` |
| `validation_confidence_threshold` | `AGENT_VALIDATION_CONFIDENCE_THRESHOLD` | `high` |
| `reuse_candidate_validation_evidence` | `AGENT_REUSE_CANDIDATE_EVIDENCE` | `false` |

Both features are off by default, so existing single-shot, interactive and
prospective-validation modes behave exactly as before.

### Limitations

Graph analysis is Python-only, because `ast` is the only parser this repository
can depend on being present. Non-Python changes are recorded as
`unsupported_files`, which lowers confidence and broadens scope rather than being
read as "no impact". Symbol nesting is tracked one level deep (`Class.method`),
matching the existing indexer's single-name `parent` field, so a definition inside
a function is not indexed. Call-graph evidence is name-based, not a resolved call
graph: it proves a test mentions a changed symbol name, not that it invokes that
definition, which is why it ranks below a resolved import edge. Dynamic imports,
`__getattr__` indirection and generated code cannot be resolved statically and are
recorded as degradations.

## Safety and current limitations

File paths are resolved beneath the selected project directory. `.git` is never
deleted, destructive commands are rejected, secrets are not read by analysis,
and unrelated pre-existing Git changes are reported and left untouched. The
agent never commits, pushes, resets, or silently discards changes.

The mock provider does not invent code changes. OpenAI and Gemini require a
working API key and models that return the documented JSON shapes; live provider
calls are not part of the automated tests. AI output is treated as untrusted:
paths, operations, patches, protected files, and plan scope are checked before
writing. Command discovery is intentionally conservative, and language-specific
semantic indexing, parallel agents, CI integration, dashboard/IDE integrations,
long-term memory, and remote Git workflows are not included in this MVP.

## Safe Gemini manual test

Create a temporary project with a small Python module and unittest, then run:

```powershell
$env:GEMINI_API_KEY = "..."
$env:GEMINI_MODEL = "gemini-2.5-flash"
agent run --project "D:\Projects\agent-test" --provider gemini --dry-run "Add a multiply function and tests"
agent run --project "D:\Projects\agent-test" --provider gemini --approval always "Add a multiply function and tests"
```

Review the dry-run diff first. The second command asks for confirmation before
writing. Use `--max-iterations 1` for a bounded smoke test. Never place the API
key in the command line or commit it to the target project.

## Roadmap

1. Add semantic code search and incremental project indexing. (Completed in Phase 3.15)
    - The agent now maintains an incremental semantic code index with semantic code indexing using tree-sitter for Python, JavaScript, and TypeScript.
    - The context selector uses this index to find relevant files based on symbol definitions, including qualified symbol matches (e.g., `UserService.save`) and dependency-aware retrieval (including files that import or are imported by a semantically matched file).
2. Add more providers and provider capability negotiation.
3. Add approval policies and richer command permissions.
4. Add parallel specialist agents, CI integration, and project memory.
5. Add optional Git commits, pull requests, dashboard, and IDE integrations.
