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
| `evidence_max_age_seconds` | `AGENT_EVIDENCE_MAX_AGE_SECONDS` | `0` (no age limit) |

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
definition, which is why it ranks below a resolved import edge (see Phase 4.18
below for the cases this now resolves more precisely). `__getattr__` indirection
and generated code still cannot be resolved statically and are recorded as
degradations.

## Semantic dependency resolution & evidence calibration (Phase 4.18)

Phase 4.17's evidence tiers were name-based: an import alias, a base class, a
decorator or a type annotation all collapsed into one undifferentiated
"reference". `dependency_resolution.py` adds a provenance-typed layer on top
(never a replacement) that explains *which construct* produced a piece of
evidence, and `validation_decision.py` centralizes the scope/reuse decision
that used to be duplicated between the candidate-time and post-apply code
paths.

### What is newly resolved

- **Import aliases** (`from x import y as z`) are resolved back to the
  original definition via the existing `SemanticGraph.imported_symbol_origins`
  table (already built by Phase 4.17, but previously unused for this), so an
  aliased reference now upgrades from `direct_import_match` to
  `direct_symbol_match` instead of under-reporting its own strength. Re-export
  chains (`a` re-imports from `b`, `c` imports from `a`) are followed up to a
  bounded number of hops with a visited-set guard against cycles.
- **Module aliases** (`import x as m`, then `m.symbol()`) needed no new
  resolution: the attribute name is captured directly regardless of the module
  alias, which Phase 4.17 already did correctly. A regression test locks this in.
- **Inheritance, decorators and annotations** get their own evidence type
  (`inheritance_match`, `decorator_match`, `annotation_match`) by tracking base
  classes, decorator expressions and parameter/return/variable annotations
  separately during the AST walk, rather than folding them into the generic
  reference set.
- **`__all__` re-exports** are labelled `reexport_match` when a dependent is
  reached through a file that explicitly re-exports the changed symbol,
  distinguishing a deliberate public re-export from an incidental transitive
  import.
- **Dynamic imports with a resolvable literal** (`importlib.import_module("pkg.x")`,
  `__import__("pkg.x")`) are resolved into a real import edge instead of being
  treated as unresolvable. A computed argument (a variable, an f-string, a
  package-relative literal) is unchanged from Phase 4.17: still a genuine
  degradation, because it genuinely cannot be known statically. This is the
  proportional policy Phase 4.17's report flagged as a follow-up: resolvability,
  not file location (`tests/` vs elsewhere), decides whether a dynamic
  construct counts as uncertainty. A resolvable literal in a test file no
  longer degrades confidence; a computed import in a test file still does,
  exactly as it would in production code.

Every evidence type has a fixed, reviewable confidence value in
`CONFIDENCE_BY_EVIDENCE_TYPE` (`direct_symbol_match` highest,
`dynamic_import_unresolved` zero) - a lookup table, not a learned or adjustable
score. This layer is purely additive: it can attach more explanation and, in
the alias case, upgrade a tier the ladder already supported, but it never
introduces a new way to reach a narrower scope than
`recommend_validation_scope` already permits.

### Evidence identity hardening

`EvidenceLedger.find_reusable` gained four additional opt-in checks, each
skipped entirely when the caller passes `None` (byte-identical Phase 4.17
behaviour) and strict when a caller passes a real value:

- `max_age_seconds` - rejects evidence older than a caller-given limit
  (`REASON_STALE`); an unparseable/missing timestamp fails closed as stale.
- `policy_fingerprint` - a digest of the settings that feed the decision
  (`max_impact_depth`, `max_affected_tests`, `validation_confidence_threshold`,
  `knowledge_graph_enabled`); a mismatch is `REASON_POLICY_MISMATCH`.
- `executable_fingerprint` - a digest of what the *logical* command actually
  resolves to right now (via `commands.resolve_executable`) plus the
  interpreter version; a mismatch is `REASON_ENVIRONMENT_MISMATCH`.
- `analyzer_version` - `SEMANTIC_ANALYZER_SCHEMA_VERSION`; a mismatch is
  `REASON_ANALYZER_VERSION_MISMATCH`.

Evidence recorded before these fields existed has an empty stored value for
each, which can never equal a real one a caller supplies - upgrading a caller
to check them therefore correctly invalidates every pre-upgrade evidence entry
rather than silently grandfathering it in. The orchestrator's post-apply reuse
path passes all four; nothing else in the codebase currently opts in to them
(reuse itself remains off by default).

### `ValidationDecisionEngine`

`local_agent/validation_decision.py` is the one place the "which commands run,
and which of those are satisfied by reused evidence" decision is made.
`Orchestrator._semantic_targeted_commands` and `._apply_evidence_reuse`
(previously two independent, hand-written implementations of overlapping
logic) now both delegate to it. `ValidationDecisionEngine.decide()` returns a
`ValidationDecision`: scope, a numeric confidence score alongside the existing
level, the selected commands, one `ReuseAttempt` per considered command
(reusable or not, with its reason), the scope reasons, and the uncertainty
sources - fully serialisable for checkpointing. `apply_reuse()` is exposed
separately so a caller with its own already-selected command list (for example
one that also merged in lexical-heuristic commands the impact report never
produced) can still route reuse through the same engine without re-deriving
targets.

### Limitations

Attribute-based evidence (`service.calculate_total()`) still proves nothing
about what `service` actually is - a same-named method on an unrelated class
produces the same weak `attribute_resolution` evidence, deliberately ranked
below every edge backed by a real import. `exec`/`eval` and a package-relative
dynamic import (`importlib.import_module(".sub", package=...)`) are not
resolved even when their arguments are literals, since the former can do
anything and the latter would need the `package=` argument resolved too.
Confidence values in `CONFIDENCE_BY_EVIDENCE_TYPE` are a fixed table set by
inspection, not calibrated against real-world false-positive/negative rates -
see "Empirical validation intelligence" below.

## Empirical validation intelligence (Phase 4.19)

Phase 4.17/4.18 gave every validation run a *decision* (scope, confidence,
selected commands, reuse verdicts). Nothing recorded whether that decision was
subsequently borne out. Phase 4.19 adds that observability layer -
`local_agent/validation_telemetry.py` - without changing what any validation
run actually does: it is instrumentation, not a new decision-making path.

**Terminology, used deliberately over more ambitious-sounding alternatives:**
this is **empirical calibration** and **telemetry**, not "self-learning" or
"AI-powered validation". The confidence table in `dependency_resolution.py` is
still the fixed, reviewed table Phase 4.18 shipped; nothing in this phase
changes it or feeds back into it automatically.

### Validation decision records

Every semantic validation decision (when `validation_telemetry_enabled` is on)
produces one bounded `ValidationDecisionRecord`: a repository-id digest (not a
raw path), a content fingerprint over the relevant files, the changed
files/symbols (each list capped at 50 entries), the evidence-type labels
behind the decision, the scope/confidence chosen, how many commands were
reused vs. rerun and why (a small tally over the existing
`local_agent.evidence.REASON_*` vocabulary, never raw text), and the policy/
analyzer version in effect. No source code, stdout, or stderr is ever stored -
the same privacy bar Phase 4.17/4.18 already established for
`ValidationEvidence`. Records persist in their own bounded, cross-task store
(`validation_telemetry.json`, capped by `validation_metrics_retention`),
completely separate from `Task`/`Checkpoint` - an old checkpoint or task is
unaffected by this phase in every way, because nothing was added to either.

### Outcome linking: validation outcome vs. decision quality

`classify_outcome()` is the one place Phase 4.19 makes precise the difference
between "did the run pass" and "was picking that scope defensible":

| Scope chosen | Targeted result | Mandatory broad/full-suite result | Decision quality |
|---|---|---|---|
| targeted | passed | passed | `consistent_no_contradiction` (not proof of sufficiency - no narrower alternative to compare against) |
| targeted | passed | **failed** | `targeted_missed_defect` - the escape signal this phase exists to surface |
| targeted | **failed** | (not reached) | `targeted_caught_defect` - a positive signal about the decision even though the run failed |
| expanded/broad | - | passed | `broad_not_proven_necessary` - passing proves nothing about whether the broader scope was needed |
| any | - | failed (other cases) | `validation_failed_no_scope_judgement` |

A `CalibrationObservation` is derived from every finalized record
(`later_broader_validation_found_defect=True` exactly for the escape-signal
row above) and appended to the same bounded store.

### Reliability estimation

`compute_reliability()` turns the observation history into a per-evidence-type
`EvidenceTypeReliability`: trials, successes, failures, a plain point estimate,
and a **Wilson lower-bound** confidence-interval estimate
(`wilson_lower_bound`). The lower bound, not the point estimate, is what
calibration ever consults - two observations that both passed yields a lower
bound well under 1.0, not 100% reliable, by construction.

### Calibration signal and the safety floor

`compute_calibration_signal()` is bidirectional and asymmetric on purpose:

- **Downward** (widens validation): triggered by even a single recorded
  `targeted_missed_defect` for a relevant evidence type, regardless of sample
  size. Widening is always safe, so it needs no minimum-sample gate.
- **Upward** (would narrow validation, shadow-only - see below): requires
  *every* evidence type in the decision to have at least
  `validation_calibration_min_samples` resolved trials **and** zero recorded
  escapes; the adjustment is capped by `validation_calibration_max_adjustment`
  and the result is always clamped to `[0, 1]`.
- **Hard gate**: `impact_is_degraded()` - a recorded degradation, an unresolved
  dependent symbol, or a genuine `dependency_resolution` evidence type whose
  fixed confidence is `0.0` (currently only `dynamic_import_unresolved`) -
  unconditionally blocks any upward adjustment, no matter how good the
  historical statistics look. A dynamic import cannot become "safe" merely
  because past runs happened to pass.

### Shadow mode

`ShadowCalibrationEngine.evaluate()` computes what a calibrated confidence
*would* have recommended - `would_narrow`, `would_broaden`, `confidence_delta`,
`shadow_scope`, `safety_override` (true when the degraded-evidence gate
suppressed an upward move the raw statistics would otherwise have allowed) -
and stores it on the record for comparison. **This is the only mode
implemented.** There is no code path, gated or otherwise, that applies a
shadow decision to real validation; `validation_calibration_shadow_mode`
exists for a future live mode and is currently always treated as `True`.
Enabling `validation_calibration_enabled` changes how much is *recorded*, never
what runs.

### `ValidationIntelligenceHealth`

`compute_health()` is a read-only diagnostic summary over the store: scope
distribution, broad-validation rate, reuse hit rate, reuse-rejection-reason
tally, per-evidence-type reliability, a false-confidence-incident count
(occurrences of `targeted_missed_defect`), and a `calibration_status` of
`no_observations` / `insufficient_data` / `shadow_only` - there is currently no
status beyond `shadow_only`, since no live mode exists. Nothing consults this
automatically; it exists to be read.

It additionally reports an `analysis_degradation_rate` (how often the *analyzer*
was the weak link, from the `degraded_analysis` flag now stored on each
record), an `evidence_corruption_rate` measured against everything the store was
ever offered rather than only what it kept, and the shadow aggregates
(`shadow_comparisons`, `shadow_would_narrow`, `shadow_would_broaden`,
`shadow_safety_overrides`, and `calibration_drift` - the mean *absolute*
confidence delta calibration wanted to apply). All are zero by default, since
`validation_calibration_enabled` is off.

### Decision-quality metrics, and what the data cannot support

`compute_decision_quality_metrics()` implements the metrics list, with naming
chosen to avoid overclaiming:

- **No recall figure is reported, ever.** Recall needs the count of defects a
  change actually introduced, including those *no* scope ever detected. Nothing
  observes that, so `recall_available` is permanently `False` and there is no
  recall field to misread.
- What is observable is the **observed escape rate**: of the targeted-scope
  decisions where a broader run also executed and *could* have contradicted the
  narrow one, how often it did. This is a *lower* bound on the true escape rate,
  so `observed_escape_rate_upper_bound` (the pessimistic Wilson end) is what any
  safety argument should quote. With zero trials it is `1.0`, not `0.0` - no
  data must never look safe.
- `targeted_agreement_rate` is called *agreement*, not *precision*: it measures
  that the broader run did not contradict the narrow decision, which is weaker
  than the narrow scope having been sufficient.
- Reuse hit/rejection rates and the per-reason tally come from the same bounded
  reason vocabulary as Phase 4.18; `stale_evidence_rejections` is the only
  freshness signal available, since raw evidence timestamps are not retained
  here. Measured over the ten Part 14 reuse scenarios, eight of the eleven
  defined reasons actually occur (scenarios 7 and 8 - environment changed and
  tool unavailable - both correctly surface as an environment mismatch).
- `confidence_buckets` groups resolved observations by *predicted* confidence,
  each with a Wilson lower bound, which is the raw material a future live
  calibration would need.

### Validation cost model

`compute_cost_model()` reports measured cost only. A record whose duration is
`0.0` is treated as **unmeasured and excluded**, never as a zero-cost run -
averaging in a placeholder zero would fabricate a cheaper-looking cost profile.
Every average is therefore accompanied by its `*_samples` count, the broad/
targeted ratio is computed only over runs where *both* ends were measured on the
same run, and `measured` is `False` when there is nothing to report. Nothing in
the package consumes a cost model to make, weaken, or narrow a decision: it is
input to the diagnostic report and to human judgement, not to policy.

### Empirical false-negative findings

`tests/test_empirical_validation_calibration.py` builds ten fixture
repositories, one per dependency-relationship type, and analyzes each with the
real analyzer. The measured result is worth stating plainly: **the graph misses
some real dependencies.** Attribute-only access and an unresolved dynamic import
do not surface the dependent in `affected_files` at all. What makes that
survivable is that those same fixtures never come out at TARGETED scope - the
confidence policy escalates them to EXPANDED, so the miss is compensated rather
than silent. That property is asserted directly, and a genuinely isolated change
is validated BROADly, which is a *cost* false positive and the correct direction
to be wrong in.

Synthetic defect injection confirms the escape mechanism is real rather than
theoretical: a defect introduced into a dependent reached only by module-
attribute access passes a targeted `pytest` run and fails the broad one, both
executed as actual subprocesses in a temporary repository, and classifies as
`targeted_missed_defect`.

### Concurrency and storage bounds

`ValidationTelemetryManager` mirrors `KnowledgeGraphManager`'s pattern (one
manager per `Orchestrator`, same storage/root), with one addition: a
process-wide lock keyed by resolved project path serializes each manager's
read-modify-write cycle against the underlying JSON file, so parallel
worktree orchestrators (Phase 4.14, threads within one process) cannot lose
each other's decision records. This is stronger than the pre-existing
knowledge-graph persistence, which has no such lock. Cross-*process*
concurrency remains out of scope, consistent with the rest of
`local_agent/storage.py`. Both the decision list and the observation list are
independently bounded by `validation_metrics_retention` (oldest evicted
first); a malformed stored entry is skipped and counted, never allowed to
raise or silently corrupt an aggregate.

### Configuration

| Setting | Default | Effect |
|---|---|---|
| `validation_telemetry_enabled` | `False` | Record a `ValidationDecisionRecord` per semantic decision. Pure observability. |
| `validation_calibration_enabled` | `False` | Additionally compute the shadow comparison. Requires telemetry; never changes real validation. |
| `validation_calibration_shadow_mode` | `True` | Reserved; only shadow mode exists in this build. |
| `validation_calibration_min_samples` | `20` | Minimum resolved trials before calibration may raise a confidence estimate. |
| `validation_calibration_max_adjustment` | `0.15` | Maximum absolute confidence-score move calibration may propose, either direction. |
| `validation_metrics_retention` | `500` | Cap on stored decision records and on stored observations (independent bounds). |

### Limitations

- No real fleet data exists yet - every number in this phase's own test suite
  is synthetic/small-scale by necessity. `calibration_status` will read
  `insufficient_data` or `no_observations` on any repository that has not
  accumulated real history.
- Decision records are per-iteration. A cross-iteration repair/abandon
  lifecycle (did a later iteration in the *same* task fix what an earlier
  targeted decision missed) is not linked - each iteration's decision and
  outcome stand alone.
- `ShadowComparison`'s `would_narrow`/`would_broaden` isolate the
  confidence-derived component of scope only; they do not re-run the other
  escalation rules in `recommend_validation_scope` (fan-out, removed public
  symbols, traversal bounds), so a shadow "would narrow" result does not claim
  the fully-escalated real scope would have been narrower too.
- No live (non-shadow) calibration mode exists. Promoting shadow to live is
  future work, gated on accumulating enough real observations to make
  `calibration_status` read anything other than `insufficient_data`.

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
