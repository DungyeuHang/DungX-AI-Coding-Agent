from __future__ import annotations

import difflib
import logging
import time
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Any

from .evidence import (
    STATUS_FAILED,
    STATUS_PASSED,
    STATUS_SKIPPED,
    EvidenceLedger,
    compute_state_fingerprint,
)
from .filesystem import ProjectFilesystem, ProtectedPathError, SandboxViolation, SECRET_NAMES
from .models import (
    FailureAnalysis,
    FileOperation,
    ImplementationResult,
    ImplementationTerminationReason,
    Plan,
    PlanAmendment,
    PreparedChange,
    ProjectContext,
    ProviderCapability,
    ReviewResult,
    RunReport,
    ScopeExpansionProposal,
    Subtask,
    SubtaskContract,
    ToolCall,
    ToolExecutionPolicy,
    ToolResult,
)
from .patching import PatchApplicationError, UnifiedPatchApplier
from .sandbox import (
    CandidateValidationReport,
    CandidateWorkspace,
    CandidateWorkspaceError,
    ProspectiveValidator,
)
from .tool_engine import ToolEngine, history_to_dict
from .tools import ToolRegistry

LOGGER = logging.getLogger(__name__)

PROTECTED_DIRS = {".git", ".hg", ".svn", ".agent_data", ".agent_worktrees"}


class UnsafeModificationError(PermissionError):
    pass


class PatchValidationError(UnsafeModificationError):
    """A strict patch failure with enough detail for a focused repair request."""

    def __init__(self, path: str, original: str, patch: str, reason: str):
        self.path = path
        self.original = original
        self.patch = patch
        self.reason = reason
        super().__init__(f"invalid patch for {path}: {reason}")


class ScopeAmendmentGuard:
    """Deterministic validator for scope expansion proposals."""

    def __init__(
        self,
        filesystem: ProjectFilesystem,
        max_total_amendments: int = 5,
        max_scope_growth_factor: float = 2.0,
    ):
        self.filesystem = filesystem
        self.max_total_amendments = max_total_amendments
        self.max_scope_growth_factor = max_scope_growth_factor

    def evaluate(
        self,
        proposal: ScopeExpansionProposal,
        active_plan: Plan,
        initial_scope_count: int = 1,
        max_total_amendments: int | None = None,
        max_scope_growth_factor: float | None = None,
    ) -> tuple[bool, str]:
        """
        Deterministic validation of a scope expansion proposal against 10 safety invariants.
        Returns (is_valid, reason).
        """
        eff_max_amendments = max_total_amendments if max_total_amendments is not None else self.max_total_amendments
        eff_max_growth = max_scope_growth_factor if max_scope_growth_factor is not None else self.max_scope_growth_factor

        # 1. Empty path rejection
        if not proposal.path or not proposal.path.strip():
            return False, "Scope expansion rejected: proposal path cannot be empty"

        raw_path = proposal.path.strip()

        # Check path traversal markers before normalization
        if ".." in Path(raw_path).parts or raw_path.startswith(("/", "\\")) or (len(raw_path) > 1 and raw_path[1] == ":"):
            return False, f"Scope expansion rejected: path traversal or absolute path not allowed: {raw_path}"

        # 2. Guard path traversal and security sandbox
        try:
            resolved = self.filesystem.resolve(raw_path)
            rel_path = resolved.relative_to(self.filesystem.root).as_posix()
        except (SandboxViolation, ValueError, ProtectedPathError) as exc:
            return False, f"Scope expansion rejected: sandbox violation: {exc}"

        # 3. Project-root confinement & Protected directory rejection
        rel_parts = resolved.relative_to(self.filesystem.root).parts
        for part in rel_parts:
            if part.lower() in PROTECTED_DIRS:
                return False, f"Scope expansion rejected: protected directory in path: {part}"

        # 4. Secret / protected file rejection
        if resolved.name.lower() in {name.lower() for name in SECRET_NAMES}:
            return False, f"Scope expansion rejected: secret file cannot be added to plan: {resolved.name}"

        # 5. Duplicate scope rejection
        allowed = active_plan.allowed_paths if hasattr(active_plan, "allowed_paths") else set(active_plan.files_likely_to_change + active_plan.files_likely_to_create)
        if rel_path in allowed:
            return False, f"Scope expansion rejected: path '{rel_path}' is already in plan"

        # 6. Existing / non-existing consistency
        exists = self.filesystem.file_exists(rel_path)
        if not proposal.is_create and not exists:
            return False, f"Scope expansion rejected: cannot modify non-existent file '{rel_path}' (set is_create=True)"
        if proposal.is_create and exists:
            return False, f"Scope expansion rejected: cannot create already existing file '{rel_path}'"

        # 7. Total amendments budget limit
        current_amendments = len(getattr(active_plan, "amendments", []))
        if current_amendments >= eff_max_amendments:
            return False, f"Scope expansion rejected: maximum amendment budget reached ({eff_max_amendments})"

        # 8. Scope growth factor limit
        current_scope_size = len(allowed)
        effective_initial = max(1, initial_scope_count)
        max_allowed_scope = int(effective_initial * eff_max_growth) + 2
        if (current_scope_size + 1) > max_allowed_scope:
            return False, f"Scope expansion rejected: scope growth limit reached (growth factor {eff_max_growth}x exceeded: current={current_scope_size}, max={max_allowed_scope})"

        return True, f"Approved scope expansion for '{rel_path}'"


class CodingAgent:
    VALID_ACTIONS = {"write", "create", "modify", "delete"}

    def __init__(self, filesystem: ProjectFilesystem, protected_paths: set[str] | None = None):
        self.filesystem = filesystem
        self.protected_paths = {self._normalize(path) for path in (protected_paths or set())}
        self._originals: dict[str, str | None] = {}

    def find_unlisted_operations(self, operations: list[FileOperation], plan: Plan | None = None) -> list[FileOperation]:
        """Find operations in the batch that target paths outside the plan's allowed scope."""
        if isinstance(plan, dict):
            allowed = set(plan.get("files_likely_to_change", []) + plan.get("files_likely_to_create", []))
        elif plan:
            allowed = plan.allowed_paths if hasattr(plan, "allowed_paths") else set(getattr(plan, "files_likely_to_change", []) + getattr(plan, "files_likely_to_create", []))
        else:
            allowed = set()
        normalized_allowed = {self._normalize(item) for item in allowed}
        unlisted = []
        for op in operations:
            rel = self._normalize(op.path)
            if normalized_allowed and rel not in normalized_allowed:
                unlisted.append(op)
        return unlisted

    def prepare(self, operations: list[FileOperation], plan: Plan | None = None) -> list[PreparedChange]:
        """Validate all operations and calculate results without writing files."""
        if isinstance(plan, dict):
            allowed = set(plan.get("files_likely_to_change", []) + plan.get("files_likely_to_create", []))
        elif plan:
            allowed = plan.allowed_paths if hasattr(plan, "allowed_paths") else set(getattr(plan, "files_likely_to_change", []) + getattr(plan, "files_likely_to_create", []))
        else:
            allowed = set()
        normalized_allowed = {self._normalize(item) for item in allowed}
        prepared: list[PreparedChange] = []
        seen: set[str] = set()
        for operation in operations:
            action = operation.action.lower().strip()
            relative = self._normalize(operation.path)
            self._validate_path(relative, action, normalized_allowed)
            if relative in seen:
                raise UnsafeModificationError(f"multiple operations target the same file: {relative}")
            seen.add(relative)
            exists = self.filesystem.file_exists(relative)
            original = self.filesystem.read_file(relative) if exists else None
            if action == "create" and exists:
                raise UnsafeModificationError(f"file already exists: {relative}")
            if action == "modify" and not exists:
                raise UnsafeModificationError(f"cannot modify missing file: {relative}")
            if action == "delete" and not exists:
                raise UnsafeModificationError(f"cannot delete missing file: {relative}")
            try:
                resulting = self._resulting_content(action, operation, original, relative)
            except PatchApplicationError as exc:
                raise PatchValidationError(relative, original or "", operation.patch or "", str(exc)) from exc
            diff = "".join(difflib.unified_diff(
                (original or "").splitlines(keepends=True),
                (resulting or "").splitlines(keepends=True),
                fromfile=f"a/{relative}", tofile=f"b/{relative}",
            ))
            prepared.append(PreparedChange(action, relative, original, resulting, diff, operation.reason))
        return prepared

    def apply_prepared(self, changes: list[PreparedChange]) -> list[str]:
        """Write a previously validated batch, recording originals for review."""
        for change in changes:
            if change.path not in self._originals:
                self._originals[change.path] = change.original
        applied: list[PreparedChange] = []
        try:
            for change in changes:
                if change.action == "delete":
                    self.filesystem.delete_file(change.path)
                elif change.action == "create":
                    self.filesystem.create_file(change.path, change.resulting or "")
                else:
                    self.filesystem.write_file(change.path, change.resulting or "")
                applied.append(change)
        except (ProtectedPathError, SandboxViolation, OSError) as exc:
            for change in reversed(applied):
                try:
                    if change.original is None:
                        self.filesystem.delete_file(change.path)
                    else:
                        self.filesystem.write_file(change.path, change.original)
                except (ProtectedPathError, SandboxViolation, OSError):
                    pass
            raise UnsafeModificationError(f"could not apply {change.path}: {exc}; applied changes were rolled back") from exc
        return [change.path for change in changes]

    def apply(self, operations: list[FileOperation], plan: Plan | None = None) -> list[str]:
        return self.apply_prepared(self.prepare(operations, plan))

    def diff(self) -> str:
        pieces: list[str] = []
        for relative, original in self._originals.items():
            path = self.filesystem.resolve(relative)
            current = self.filesystem.read_file(relative) if path.exists() else ""
            pieces.extend(difflib.unified_diff(
                (original or "").splitlines(keepends=True), current.splitlines(keepends=True),
                fromfile=f"a/{relative}", tofile=f"b/{relative}",
            ))
        return "".join(pieces)

    def _validate_path(self, relative: str, action: str, allowed: set[str]) -> None:
        if action not in self.VALID_ACTIONS:
            raise UnsafeModificationError(f"unsupported file operation: {action}")
        if not relative or relative == "." or relative.startswith("../") or relative.startswith("/") or (len(relative) > 1 and relative[1] == ":") or ".." in Path(relative).parts:
            raise SandboxViolation(f"invalid relative path: {relative}")
        is_protected = any(relative == protected or relative.startswith(protected.rstrip("/") + "/") for protected in self.protected_paths)
        if is_protected and relative not in allowed:
            raise UnsafeModificationError(f"refusing to overwrite unrelated existing change: {relative}")
        if allowed and relative not in allowed:
            raise UnsafeModificationError(f"file operation is outside the approved plan: {relative}")

    @staticmethod
    def _resulting_content(action: str, operation: FileOperation, original: str | None, relative: str) -> str | None:
        if action == "delete":
            return None
        if action in {"write", "create", "modify"} and operation.content is not None:
            return operation.content
        if operation.patch is not None:
            return UnifiedPatchApplier().apply(original or "", operation.patch, expected_path=relative)
        raise PatchApplicationError(f"{action} requires complete content or a unified patch: {relative}")

    @staticmethod
    def _normalize(path: str) -> str:
        value = path.replace("\\", "/")
        if value.startswith("/") or (len(value) > 1 and value[1] == ":"):
            return value
        return Path(value).as_posix()


# ---------------------------------------------------------------------------
# Phase 4.15: Interactive, tool-assisted implementation agent
# ---------------------------------------------------------------------------

#: Intentionally restricted tool surface for the implementation specialist.
#: The agent may inspect the repository (read/search/structure) and run bounded
#: read-only probes, but it never mutates the filesystem through a tool. All
#: mutations are emitted as FileOperation objects and flow through the existing
#: CodingAgent.prepare -> approval -> apply_prepared pipeline, so plan-scope
#: checks, protected-path checks, patch validation and approval boundaries all
#: remain in force.
IMPLEMENTATION_TOOL_SURFACE: frozenset[str] = frozenset({
    "read_file_range",      # INSPECT
    "search_symbols",       # SEARCH (semantic / AST index)
    "grep_code",            # SEARCH (textual)
    "find_files",           # INSPECT repository structure
    "run_command_sandbox",  # RUN PROBE / targeted validation, git diff, git status
})

DEFAULT_MAX_IMPLEMENTATION_TOOL_STEPS = 15
DEFAULT_MAX_REFINE_ROUNDS = 2
DEFAULT_MAX_CANDIDATE_ITERATIONS = 2

_PRECHECK_TOOL_NAME = "implementation_precheck"
_CANDIDATE_TOOL_NAME = "candidate_validation"


@dataclass
class _CandidateStats:
    """Mutable accumulator for Phase 4.16 candidate-loop telemetry."""

    iterations: int = 0
    validation_attempts: int = 0
    validation_successes: int = 0
    validation_failures: int = 0
    recovery_attempts: int = 0
    commands_run: int = 0
    validation_runtime: float = 0.0
    elapsed: float = 0.0
    cleanup_failures: int = 0
    changed_files: list[str] = dataclass_field(default_factory=list)
    # --- Phase 4.17 semantic impact accumulators ---
    #: Most recent ChangeImpactReport produced against a candidate tree.
    impact_report: Any = None
    #: Wall-clock seconds spent inside semantic impact analysis.
    impact_seconds: float = 0.0


@dataclass
class _CandidateOutcome:
    """Result of one PROPOSE -> APPLY -> VALIDATE round against a candidate tree."""

    passed: bool = False
    report: CandidateValidationReport | None = None
    invalid_reason: str | None = None
    changed_files: list[str] = dataclass_field(default_factory=list)

    def render_feedback(self) -> str:
        if self.invalid_reason is not None:
            return (
                "The proposed edits could NOT be applied to the isolated candidate tree, "
                "so nothing was validated and the real project was not modified.\n"
                f"- {self.invalid_reason}\n"
                "Re-emit only operations that stay inside the approved plan scope, avoid "
                "protected paths, and whose patches apply to the current file contents."
            )
        if self.report is not None:
            return self.report.render_feedback()
        return "Candidate validation produced no result."


class InteractiveCodingAgent:
    """Interactive, tool-using implementation specialist.

    Drives a bounded UNDERSTAND -> INSPECT -> SEARCH -> REASON -> EDIT ->
    PROBE -> INSPECT RESULT -> REFINE -> RECHECK -> FINALIZE loop by composing
    the existing :class:`~local_agent.tool_engine.ToolEngine` (never forking it)
    with the existing :class:`~local_agent.tools.ToolRegistry`.

    Exploration turns are executed by ToolEngine, which already provides the
    anti-repetition circuit breaker, per-tool limits, byte budgets, step budgets
    and deterministic context compaction. This class adds:

    * a restricted implementation tool surface (via ``ToolExecutionPolicy``),
    * implementation-specific prompt scaffolding (contracts, knowledge, failure),
    * an EDIT -> RECHECK -> REFINE outer loop driven by a deterministic
      pre-mutation check of the emitted operations, and
    * a structured :class:`~local_agent.models.ImplementationResult` carrying
      telemetry and a classified failure category.

    All mutations are confined to ``filesystem.root``. When running inside a Git
    worktree the caller supplies a worktree-rooted ``ProjectFilesystem`` and
    ``ToolRegistry``, so no state is shared between parallel workers and the
    process working directory is never relied upon.
    """

    def __init__(
        self,
        filesystem: ProjectFilesystem,
        registry: ToolRegistry,
        policy: ToolExecutionPolicy | None = None,
        max_tool_steps: int = DEFAULT_MAX_IMPLEMENTATION_TOOL_STEPS,
        max_refine_rounds: int = DEFAULT_MAX_REFINE_ROUNDS,
        allowed_tools: set[str] | frozenset[str] | None = None,
        sandbox: CandidateWorkspace | None = None,
        validator: ProspectiveValidator | None = None,
        max_candidate_iterations: int = DEFAULT_MAX_CANDIDATE_ITERATIONS,
        cleanup_sandbox: bool = True,
        evidence_ledger: "EvidenceLedger | None" = None,
    ):
        self.filesystem = filesystem
        self.registry = registry
        self.policy = policy
        self.max_tool_steps = max(1, int(max_tool_steps))
        self.max_refine_rounds = max(0, int(max_refine_rounds))
        self.allowed_tools = frozenset(allowed_tools) if allowed_tools is not None else IMPLEMENTATION_TOOL_SURFACE
        # Phase 4.16: prospective validation. When ``sandbox`` is None the agent
        # behaves exactly as in Phase 4.15 (mode B); when it is supplied the
        # PROPOSE -> APPLY -> VALIDATE -> REFINE candidate loop runs (mode C).
        self.sandbox = sandbox
        self.validator = validator or ProspectiveValidator()
        self.max_candidate_iterations = max(1, int(max_candidate_iterations))
        self.cleanup_sandbox = bool(cleanup_sandbox)
        # Phase 4.17: bounded, iteration-aware evidence history. Instance-scoped
        # (never module-level), so parallel worktree workers cannot see or
        # corrupt each other's evidence.
        self.evidence_ledger = evidence_ledger if evidence_ledger is not None else EvidenceLedger()

    @property
    def prospective_validation_enabled(self) -> bool:
        return self.sandbox is not None

    # -- policy ------------------------------------------------------------

    def build_policy(self) -> ToolExecutionPolicy:
        """Derive the effective execution policy for the implementation loop.

        Reuses the caller-supplied (config-derived) policy but clamps the step
        budget to ``max_tool_steps`` and narrows the tool surface by adding every
        registry tool outside :attr:`allowed_tools` to ``disallowed_tools``.
        """
        base = self.policy
        registry_tools = {definition.name for definition in self.registry.definitions()}
        out_of_surface = {name for name in registry_tools if name not in self.allowed_tools}

        if base is None:
            return ToolExecutionPolicy(
                max_tool_steps=self.max_tool_steps,
                disallowed_tools=set(out_of_surface),
            )

        return ToolExecutionPolicy(
            max_tool_steps=min(int(base.max_tool_steps), self.max_tool_steps),
            max_tool_output_bytes=base.max_tool_output_bytes,
            total_tool_budget_bytes=base.total_tool_budget_bytes,
            max_consecutive_repeats=base.max_consecutive_repeats,
            per_tool_limits=dict(base.per_tool_limits),
            disallowed_tools=set(base.disallowed_tools) | out_of_surface,
            compaction_window=base.compaction_window,
            max_context_bytes=base.max_context_bytes,
        )

    # -- prompt ------------------------------------------------------------

    def build_prompt(
        self,
        task_objective: str,
        subtask: Subtask | None = None,
        upstream_contracts: list[SubtaskContract] | None = None,
        context: ProjectContext | None = None,
        failure: FailureAnalysis | None = None,
    ) -> str:
        """Compose the implementation-specialist prompt for the tool loop."""
        header = task_objective
        if subtask is not None and getattr(subtask, "goal", None) and subtask.goal != task_objective:
            header = f"Subtask Goal: {subtask.goal}\n\nTask Objective: {task_objective}"

        lines: list[str] = [
            header,
            "",
            "INTERACTIVE IMPLEMENTATION PROTOCOL",
            "===================================",
            "You are the implementation specialist working in an interactive, tool-assisted loop.",
            "Discover before you edit. Do NOT rewrite the whole task from scratch after a failure.",
            "",
            "1. UNDERSTAND  - restate the concrete change required by the objective and plan.",
            "2. INSPECT     - use read_file_range and find_files to read the actual current code.",
            "3. SEARCH      - use search_symbols (AST index) and grep_code to locate call sites,",
            "                 definitions and existing conventions before introducing new ones.",
            "4. REASON      - decide the minimal, focused edit consistent with the approved plan.",
            "5. EDIT        - return the final list of file operations to conclude the loop.",
            "6. PROBE       - before concluding you may run bounded read-only checks with",
            "                 run_command_sandbox (e.g. python -m py_compile, a targeted test,",
            "                 git diff, git status).",
            "7. REFINE      - if a probe or pre-mutation check reports a problem, inspect the",
            "                 specific failure and correct only the affected operation.",
            "",
            "CONSTRAINTS",
            "-----------",
            "- Stay strictly inside the approved plan scope; do not touch unrelated files.",
            "- Prefer surgical edits over wholesale file rewrites.",
            "- Never attempt to read or modify protected directories or secret-like files.",
            "- Repeating an identical tool call trips a circuit breaker; vary your investigation.",
            f"- You have a hard budget of {self.max_tool_steps} tool steps for this implementation.",
            f"- Available tools: {', '.join(sorted(self.allowed_tools))}.",
        ]

        if self.sandbox is not None:
            lines.extend([
                "",
                "PROSPECTIVE VALIDATION",
                "======================",
                "Every tool you run operates on an isolated CANDIDATE COPY of the repository,",
                "never on the real project. When you return file operations they are applied to",
                "that candidate copy and real validation commands (syntax check, targeted tests,",
                "static analysis) are actually executed against it. If validation fails you will",
                "receive the real command output and must correct the specific defect.",
                f"- You have a budget of {self.max_candidate_iterations} candidate iteration(s).",
                "- The real project is only modified after a candidate passes and the change is",
                "  approved through the normal apply pipeline.",
            ])

        if upstream_contracts:
            lines.append("")
            lines.append("UPSTREAM INTERFACE CONSTRAINTS")
            lines.append("==============================")
            lines.append(
                "Verified outputs of completed dependencies. Treat as authoritative interfaces."
            )
            for contract in upstream_contracts:
                if hasattr(contract, "format_for_prompt"):
                    lines.append(contract.format_for_prompt(max_chars=1200))

        knowledge = None
        metadata = getattr(context, "metadata", None) if context is not None else None
        if isinstance(metadata, dict):
            knowledge = metadata.get("persistent_knowledge")
        if knowledge:
            lines.append("")
            lines.append("PERSISTENT REPOSITORY KNOWLEDGE")
            lines.append("===============================")
            lines.append(str(knowledge))

        if failure is not None:
            lines.append("")
            lines.append("PRIOR FAILURE TO REPAIR")
            lines.append("=======================")
            lines.append(f"Probable root cause: {getattr(failure, 'probable_root_cause', '')}")
            recommended = getattr(failure, "recommended_fix", None)
            if recommended:
                lines.append(f"Recommended fix: {recommended}")
            lines.append(
                "Inspect the affected code with tools before editing; do not regenerate unrelated files."
            )

        return "\n".join(lines)

    # -- execution ---------------------------------------------------------

    def execute(
        self,
        provider: Any,
        task_objective: str,
        plan: Plan,
        context: ProjectContext,
        subtask: Subtask | None = None,
        upstream_contracts: list[SubtaskContract] | None = None,
        failure: FailureAnalysis | None = None,
        review: ReviewResult | None = None,
        initial_history: list[tuple[ToolCall, ToolResult]] | None = None,
        report: RunReport | None = None,
    ) -> ImplementationResult:
        """Run the interactive implementation loop and return a structured result.

        Provider transport errors (``ProviderError`` and subclasses such as
        ``RateLimitError``/``QuotaExceededError``) are intentionally allowed to
        propagate so the orchestrator's existing specialist routing, fallback and
        pause machinery handles them; they are never silently swallowed here.
        """
        start_time = time.perf_counter()
        provider_name = getattr(
            provider, "provider_id", getattr(provider, "name", provider.__class__.__name__)
        )
        model_name = getattr(provider, "model", None)

        prompt = self.build_prompt(
            task_objective=task_objective,
            subtask=subtask,
            upstream_contracts=upstream_contracts,
            context=context,
            failure=failure,
        )

        if not self._supports_tool_use(provider):
            return self._single_shot(
                provider=provider,
                prompt=prompt,
                plan=plan,
                context=context,
                failure=failure,
                review=review,
                provider_name=provider_name,
                model_name=model_name,
                start_time=start_time,
            )

        effective_policy = self.build_policy()
        prospective = self.sandbox is not None
        stats = _CandidateStats()
        registry = self.registry

        if prospective:
            setup_started = time.perf_counter()
            try:
                self.sandbox.setup()
            except (CandidateWorkspaceError, OSError, ValueError) as exc:
                LOGGER.warning("Could not materialise candidate workspace: %s", exc)
                return self._candidate_setup_failure(
                    exc, provider_name, model_name, start_time
                )
            stats.elapsed += time.perf_counter() - setup_started
            # Every tool the agent uses is now rooted at the candidate tree:
            # reads, searches and every run_command_sandbox subprocess get
            # cwd=<candidate root>. ToolEngine and ToolRegistry stay untouched.
            registry = self.sandbox.registry

        history: list[tuple[ToolCall, ToolResult]] = list(initial_history or [])
        engine_result = None
        refine_rounds = 0
        precheck_errors: list[str] = []
        outcome: _CandidateOutcome | None = None

        try:
            while True:
                engine = ToolEngine(provider=provider, registry=registry, policy=effective_policy)
                engine_result = engine.run(
                    task=prompt,
                    plan=plan,
                    context=context,
                    initial_history=history,
                    failure=failure,
                    review=review,
                )
                history = list(engine_result.tool_history)

                if not engine_result.completed or engine_result.file_operations is None:
                    break

                # RECHECK: deterministic pre-mutation validation of the emitted edits.
                precheck_errors = self._precheck_operations(engine_result.file_operations)
                if precheck_errors:
                    if refine_rounds >= self.max_refine_rounds:
                        break
                    # REFINE: feed the specific defect back into the loop instead
                    # of discarding the work and regenerating the whole task.
                    refine_rounds += 1
                    history.append(self._precheck_feedback(precheck_errors, refine_rounds))
                    continue

                if not prospective:
                    break

                # PROSPECTIVE VALIDATION: rebuild BASE + operations in the
                # isolated candidate tree and run real commands against it.
                outcome = self._evaluate_candidate(
                    engine_result.file_operations, plan, context, stats
                )
                if outcome.passed:
                    break
                if stats.iterations >= self.max_candidate_iterations:
                    break
                stats.recovery_attempts += 1
                history.append(self._candidate_feedback(outcome, stats.iterations))
        finally:
            if prospective and self.cleanup_sandbox:
                self.sandbox.cleanup()
                stats.cleanup_failures = self.sandbox.cleanup_failures

        return self._build_result(
            engine_result=engine_result,
            plan=plan,
            provider_name=provider_name,
            model_name=model_name,
            start_time=start_time,
            refine_rounds=refine_rounds,
            precheck_errors=precheck_errors,
            prospective=prospective,
            stats=stats,
            outcome=outcome,
        )

    # -- Phase 4.16 candidate loop ----------------------------------------

    def _evaluate_candidate(
        self,
        operations: list[FileOperation],
        plan: Plan,
        context: ProjectContext,
        stats: _CandidateStats,
    ) -> _CandidateOutcome:
        """Rebuild the candidate tree and run real validation against it.

        The candidate is always ``BASE + operations``: :meth:`CandidateWorkspace.rebuild`
        reverts the previous iteration's paths before applying the new set, so
        no patch-on-patch drift is possible. Operations are applied through the
        very same ``CodingAgent.prepare``/``apply_prepared`` pipeline used for
        the authoritative tree, so plan scope, protected paths and traversal
        protection reject a candidate exactly as they would reject a real apply.
        """
        assert self.sandbox is not None  # guarded by caller
        started = time.perf_counter()
        stats.iterations += 1

        try:
            changed = self.sandbox.rebuild(operations, plan)
        except (
            UnsafeModificationError,
            SandboxViolation,
            ProtectedPathError,
            PatchApplicationError,
            CandidateWorkspaceError,
            OSError,
        ) as exc:
            # Nothing was applied, so no candidate file actually changed.
            stats.changed_files = []
            stats.elapsed += time.perf_counter() - started
            return _CandidateOutcome(passed=False, invalid_reason=str(exc))

        stats.changed_files = list(changed)
        report = self.validator.validate(
            self.sandbox, changed, getattr(context, "repository_map", None)
        )
        stats.validation_attempts += 1
        if report.passed:
            stats.validation_successes += 1
        else:
            stats.validation_failures += 1
        stats.commands_run += report.commands_run
        stats.validation_runtime += sum(r.duration_seconds for r in report.executed_results)

        # Phase 4.17: capture the impact analysis the validator just performed
        # and turn each executed command into a reusable evidence entry.
        impact = getattr(self.validator, "last_impact_report", None)
        if impact is not None:
            stats.impact_report = impact
            stats.impact_seconds += float(getattr(impact, "analysis_seconds", 0.0) or 0.0)
        self._record_candidate_evidence(report, impact, changed, stats.iterations)

        stats.elapsed += time.perf_counter() - started
        return _CandidateOutcome(
            passed=report.passed, report=report, changed_files=list(changed)
        )

    def _record_candidate_evidence(
        self,
        report: CandidateValidationReport,
        impact: Any | None,
        changed_files: list[str],
        iteration: int,
    ) -> None:
        """Record one evidence entry per executed candidate command.

        The fingerprint covers the *union* of the changed files, every module
        the impact analysis found downstream of them, and the target test file
        itself. That is what a post-apply reuse decision will re-hash: if any of
        those bytes differ in the authoritative tree, reuse is refused and the
        command is rerun. Hashing content (not paths, mtimes or the root) is
        what makes candidate evidence comparable across two different
        directories at all.
        """
        if self.sandbox is None or not self.sandbox.is_active:
            return
        confidence = str(getattr(impact, "confidence", "low") or "low")
        symbols = (
            sorted({symbol.qualified_name for symbol in impact.changed_symbols})
            if impact is not None
            else []
        )
        relevant_base = set(changed_files)
        if impact is not None:
            relevant_base |= set(getattr(impact, "affected_files", []) or [])

        explanations: dict[tuple[str, ...], Any] = {}
        for target in (getattr(impact, "validation_targets", None) or []):
            explanations[tuple(target.command)] = target

        for result in report.results:
            command = tuple(result.command)
            target = explanations.get(command)
            relevant = set(relevant_base)
            tail = command[-1] if len(command) > 1 else ""
            if tail.endswith(".py"):
                relevant.add(tail.replace("\\", "/"))
            if result.skipped:
                status = STATUS_SKIPPED
            elif result.exit_code == 0:
                status = STATUS_PASSED
            else:
                status = STATUS_FAILED
            self.evidence_ledger.record(
                command=command,
                status=status,
                exit_code=result.exit_code,
                duration_seconds=result.duration_seconds,
                category=result.tier,
                selected_because=(
                    target.selected_because if target is not None
                    else f"{result.tier} tier command for the candidate change"
                ),
                tier=(target.tier if target is not None else result.tier),
                impacted_files=sorted(relevant),
                impacted_symbols=symbols,
                confidence=confidence,
                stdout=result.stdout,
                stderr=result.stderr,
                candidate_iteration=iteration,
                environment_root="candidate",
                skipped_reason=result.skip_reason,
                fingerprint=compute_state_fingerprint(self.sandbox.root, sorted(relevant)),
            )

    @staticmethod
    def _candidate_feedback(
        outcome: _CandidateOutcome, iteration: int
    ) -> tuple[ToolCall, ToolResult]:
        """Synthetic history turn carrying real candidate-validation evidence."""
        call = ToolCall(
            call_id=f"candidate_{iteration}",
            tool_name=_CANDIDATE_TOOL_NAME,
            arguments={"iteration": iteration},
        )
        result = ToolResult(
            call_id=call.call_id,
            tool_name=_CANDIDATE_TOOL_NAME,
            output=outcome.render_feedback(),
            is_error=True,
        )
        return call, result

    def _candidate_setup_failure(
        self,
        exc: Exception,
        provider_name: str,
        model_name: str | None,
        start_time: float,
    ) -> ImplementationResult:
        reason = ImplementationTerminationReason.CANDIDATE_SETUP_FAILED
        return ImplementationResult(
            success=False,
            file_operations=None,
            summary="Interactive implementation aborted: candidate workspace unavailable",
            elapsed_time_seconds=time.perf_counter() - start_time,
            provider=provider_name,
            model=model_name,
            termination_reason=reason,
            error_message=f"Could not materialise candidate workspace: {exc}",
            failure_category=ImplementationTerminationReason.categorize(reason),
            prospective_validation_used=True,
        )

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _supports_tool_use(provider: Any) -> bool:
        """Whether the provider can drive a multi-turn tool loop."""
        capabilities = getattr(provider, "capabilities", None)
        return (
            isinstance(capabilities, (set, frozenset))
            and ProviderCapability.TOOL_USE in capabilities
            and hasattr(provider, "generate_code_with_tools")
        )

    def _single_shot(
        self,
        provider: Any,
        prompt: str,
        plan: Plan,
        context: ProjectContext,
        failure: FailureAnalysis | None,
        review: ReviewResult | None,
        provider_name: str,
        model_name: str | None,
        start_time: float,
    ) -> ImplementationResult:
        """Legacy single-shot generation, used when the provider cannot use tools."""
        operations = provider.generate_code(prompt, plan, context, failure=failure, review=review)
        elapsed = time.perf_counter() - start_time
        operations_list = list(operations) if operations is not None else None
        modified = [op.path for op in (operations_list or [])]
        return ImplementationResult(
            success=operations_list is not None,
            file_operations=operations_list,
            summary=(
                f"Single-shot implementation produced {len(operations_list or [])} operation(s) "
                "(provider does not support tool use)"
            ),
            files_modified=modified,
            tool_steps_used=0,
            elapsed_time_seconds=elapsed,
            provider=provider_name,
            model=model_name,
            termination_reason=ImplementationTerminationReason.SINGLE_SHOT_FALLBACK,
            used_fallback=True,
            failure_category=ImplementationTerminationReason.categorize(
                ImplementationTerminationReason.SINGLE_SHOT_FALLBACK
            ),
        )

    def _precheck_operations(self, operations: list[FileOperation]) -> list[str]:
        """Deterministically validate emitted operations without touching the disk.

        Detects the failure modes the model can actually fix in-loop: an
        unusable/ambiguous action, a patch that does not apply to the current
        file content, and Python source that does not parse. Plan-scope
        violations are deliberately *not* treated as in-loop defects because the
        orchestrator handles them through the scope-expansion amendment flow.
        """
        errors: list[str] = []
        for operation in operations:
            action = (operation.action or "").lower().strip()
            relative = CodingAgent._normalize(operation.path)

            if action not in CodingAgent.VALID_ACTIONS:
                errors.append(f"{relative}: unsupported action '{operation.action}'")
                continue
            if action == "delete":
                continue
            if operation.content is None and operation.patch is None:
                errors.append(
                    f"{relative}: operation provides neither complete content nor a unified patch"
                )
                continue

            try:
                exists = self.filesystem.file_exists(relative)
                original = self.filesystem.read_file(relative) if exists else None
            except (ProtectedPathError, SandboxViolation, ValueError, OSError) as exc:
                errors.append(f"{relative}: cannot read current content: {exc}")
                continue

            if operation.content is not None:
                resulting = operation.content
            else:
                try:
                    resulting = UnifiedPatchApplier().apply(
                        original or "", operation.patch or "", expected_path=relative
                    )
                except PatchApplicationError as exc:
                    errors.append(f"{relative}: patch does not apply cleanly: {exc}")
                    continue

            if relative.endswith(".py") and resulting is not None:
                try:
                    compile(resulting, relative, "exec")
                except SyntaxError as exc:
                    errors.append(f"{relative}: resulting Python source is invalid: {exc}")

        return errors

    @staticmethod
    def _precheck_feedback(errors: list[str], round_index: int) -> tuple[ToolCall, ToolResult]:
        """Build a synthetic history turn describing a failed pre-mutation check."""
        call = ToolCall(
            call_id=f"precheck_{round_index}",
            tool_name=_PRECHECK_TOOL_NAME,
            arguments={"round": round_index},
        )
        detail = "\n".join(f"- {message}" for message in errors)
        result = ToolResult(
            call_id=call.call_id,
            tool_name=_PRECHECK_TOOL_NAME,
            output=(
                "Pre-mutation check rejected the proposed edits. Nothing was written.\n"
                f"{detail}\n"
                "Inspect the affected file(s) with read_file_range and re-emit ONLY the "
                "corrected operation(s). Do not regenerate unrelated files."
            ),
            is_error=True,
        )
        return call, result

    def _collect_inspected_files(
        self, history: list[tuple[ToolCall, ToolResult]]
    ) -> list[str]:
        """Derive the set of repository files the agent actually looked at."""
        inspected: set[str] = set()

        def _record(candidate: str) -> None:
            candidate = candidate.strip().strip("`'\"")
            if not candidate or candidate.startswith("..."):
                return
            try:
                normalized = CodingAgent._normalize(candidate)
                if self.filesystem.file_exists(normalized):
                    inspected.add(normalized)
            except (ProtectedPathError, SandboxViolation, ValueError, OSError):
                return

        for call, result in history:
            if result.is_error:
                continue
            if call.tool_name == "read_file_range":
                path = call.arguments.get("path")
                if isinstance(path, str):
                    _record(path)
            elif call.tool_name == "find_files":
                for line in result.output.splitlines():
                    _record(line)
            elif call.tool_name == "grep_code":
                for line in result.output.splitlines():
                    # "relative/path.py:12: matched text"
                    _record(line.split(":", 1)[0])
            elif call.tool_name == "search_symbols":
                for line in result.output.splitlines():
                    # "- name (kind) in relative/path.py:12"
                    if " in " not in line:
                        continue
                    _record(line.rsplit(" in ", 1)[1].rsplit(":", 1)[0])

        return sorted(inspected)

    @staticmethod
    def _tool_telemetry(history: list[tuple[ToolCall, ToolResult]]) -> dict[str, int]:
        """Aggregate implementation-level tool telemetry from the canonical history."""
        successes = failures = 0
        validation_attempts = validation_failures = 0
        recovery_attempts = 0
        circuit_breaker_events = 0
        previous_failed = False

        for call, result in history:
            if result.is_error:
                failures += 1
                if "Circuit breaker" in result.output:
                    circuit_breaker_events += 1
                previous_failed = True
            else:
                successes += 1
                if previous_failed:
                    recovery_attempts += 1
                previous_failed = False

            if call.tool_name == "run_command_sandbox":
                validation_attempts += 1
                if result.is_error:
                    validation_failures += 1

        return {
            "tool_call_successes": successes,
            "tool_call_failures": failures,
            "validation_attempts": validation_attempts,
            "validation_failures": validation_failures,
            "recovery_attempts": recovery_attempts,
            "circuit_breaker_events": circuit_breaker_events,
        }

    def _build_result(
        self,
        engine_result: Any,
        plan: Plan,
        provider_name: str,
        model_name: str | None,
        start_time: float,
        refine_rounds: int,
        precheck_errors: list[str],
        prospective: bool = False,
        stats: _CandidateStats | None = None,
        outcome: _CandidateOutcome | None = None,
    ) -> ImplementationResult:
        elapsed = time.perf_counter() - start_time
        history = list(getattr(engine_result, "tool_history", []) or [])
        operations = getattr(engine_result, "file_operations", None)
        telemetry = self._tool_telemetry(history)
        stats = stats or _CandidateStats()

        termination_reason = getattr(engine_result, "termination_reason", None)
        error_message = getattr(engine_result, "error_message", None)
        success = bool(getattr(engine_result, "completed", False)) and operations is not None

        if success and precheck_errors:
            # Refine budget exhausted while the edits are still not applicable.
            success = False
            operations = None
            termination_reason = ImplementationTerminationReason.NO_OPERATIONS
            error_message = (
                "Proposed edits failed the pre-mutation check after "
                f"{refine_rounds} refinement round(s): " + "; ".join(precheck_errors)
            )

        candidate_success = False
        if success and prospective:
            if outcome is not None and outcome.passed:
                candidate_success = True
                termination_reason = ImplementationTerminationReason.CANDIDATE_VALIDATION_PASSED
            else:
                # A candidate that never proved itself is never handed to the
                # approval/apply pipeline.
                success = False
                operations = None
                if outcome is None:
                    termination_reason = ImplementationTerminationReason.CANDIDATE_VALIDATION_FAILED
                    error_message = "Candidate validation never produced a result."
                elif outcome.invalid_reason is not None:
                    termination_reason = ImplementationTerminationReason.CANDIDATE_INVALID_OPERATIONS
                    error_message = (
                        "Proposed edits could not be applied to the candidate tree: "
                        f"{outcome.invalid_reason}"
                    )
                elif stats.iterations >= self.max_candidate_iterations:
                    termination_reason = ImplementationTerminationReason.CANDIDATE_BUDGET_EXHAUSTED
                    error_message = (
                        f"Candidate validation still failing after {stats.iterations} "
                        f"candidate iteration(s) (budget {self.max_candidate_iterations}); "
                        f"first failing tier: {getattr(outcome.report, 'failed_tier', None)}"
                    )
                else:
                    termination_reason = ImplementationTerminationReason.CANDIDATE_VALIDATION_FAILED
                    error_message = (
                        "Real validation against the candidate tree failed; first failing "
                        f"tier: {getattr(outcome.report, 'failed_tier', None)}"
                    )

        scope_violations: list[str] = []
        if operations:
            scope_violations = [
                op.path for op in CodingAgent(self.filesystem).find_unlisted_operations(operations, plan)
            ]

        if success:
            summary = (
                f"Interactive implementation completed: {len(operations or [])} operation(s), "
                f"{getattr(engine_result, 'steps_used', 0)} tool step(s), "
                f"{refine_rounds} refinement round(s)"
            )
            if prospective:
                summary += (
                    f", {stats.iterations} candidate iteration(s), "
                    f"{stats.commands_run} real validation command(s)"
                )
        else:
            summary = (
                f"Interactive implementation terminated ({termination_reason}) after "
                f"{getattr(engine_result, 'steps_used', 0)} tool step(s)"
            )

        descriptor: dict[str, Any] = {}
        if prospective:
            descriptor = {
                "base_root": str(self.filesystem.root),
                "iteration": stats.iterations,
                "max_candidate_iterations": self.max_candidate_iterations,
                "operations": [
                    {"action": op.action, "path": op.path}
                    for op in (getattr(engine_result, "file_operations", None) or [])
                ],
                "operations_digest": _operations_digest(
                    getattr(engine_result, "file_operations", None) or []
                ),
                "validation_passed": candidate_success,
                "failed_tier": getattr(
                    getattr(outcome, "report", None), "failed_tier", None
                ),
            }

        impact = stats.impact_report
        impact_targets = list(getattr(impact, "validation_targets", None) or [])
        return ImplementationResult(
            success=success,
            file_operations=operations,
            summary=summary,
            files_inspected=self._collect_inspected_files(history),
            files_modified=[op.path for op in (operations or [])],
            tool_steps_used=int(getattr(engine_result, "steps_used", 0)),
            elapsed_time_seconds=elapsed,
            provider=provider_name,
            model=model_name,
            termination_reason=termination_reason,
            used_fallback=False,
            scope_violations=scope_violations,
            tool_history=history_to_dict(history),
            metrics=getattr(engine_result, "metrics", None),
            error_message=error_message,
            failure_category=(
                "none" if success else ImplementationTerminationReason.categorize(termination_reason)
            ),
            prospective_validation_used=prospective,
            candidate_iterations=stats.iterations,
            candidate_validation_attempts=stats.validation_attempts,
            candidate_validation_successes=stats.validation_successes,
            candidate_validation_failures=stats.validation_failures,
            candidate_recovery_attempts=stats.recovery_attempts,
            candidate_files_changed=list(stats.changed_files),
            candidate_elapsed_seconds=stats.elapsed,
            candidate_cleanup_failures=stats.cleanup_failures,
            validation_commands_run=stats.commands_run,
            validation_runtime_seconds=stats.validation_runtime,
            final_candidate_success=candidate_success,
            candidate_descriptor=descriptor,
            candidate_validation_report=(
                outcome.report.to_dict()
                if outcome is not None and outcome.report is not None
                else None
            ),
            semantic_impact_used=impact is not None,
            impact_confidence=str(getattr(impact, "confidence", "") or ""),
            impact_recommended_scope=str(getattr(impact, "recommended_scope", "") or ""),
            impact_changed_symbols=len(getattr(impact, "changed_symbols", None) or []),
            impact_affected_symbols=len(getattr(impact, "affected_symbols", None) or []),
            impact_tests_considered=int(getattr(impact, "tests_considered", 0) or 0),
            impact_tests_selected=len(impact_targets),
            impact_semantic_targets=sum(1 for t in impact_targets if t.is_semantic),
            impact_analysis_seconds=stats.impact_seconds,
            impact_report=impact.to_dict() if impact is not None else None,
            validation_evidence_reused=self.evidence_ledger.reuse_grants,
            validation_evidence_invalidated=self.evidence_ledger.reuse_denials,
            validation_time_saved_seconds=self.evidence_ledger.time_saved_seconds,
            validation_evidence=[
                entry.to_dict() for entry in self.evidence_ledger.entries
            ],
            **telemetry,
        )


def _operations_digest(operations: list[FileOperation]) -> str:
    """Stable digest of a candidate operation set, for checkpoint descriptors."""
    import hashlib

    hasher = hashlib.sha256()
    for operation in sorted(operations, key=lambda op: (op.path, op.action)):
        hasher.update(f"{operation.action}\x00{operation.path}\x00".encode("utf-8"))
        hasher.update((operation.content or "").encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update((operation.patch or "").encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()[:16]
