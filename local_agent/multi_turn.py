from __future__ import annotations

import ast
import datetime
import difflib
import hashlib
import logging
import os
import shlex
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from .coding_agent import (
    IMPLEMENTATION_TOOL_SURFACE,
    CodingAgent,
    PatchValidationError,
    ScopeAmendmentGuard,
    UnsafeModificationError,
)
from .completion import (
    CompletionAssessment,
    CompletionDecisionEngine,
    CompletionEvidenceStore,
    EvidenceStatus,
    EvidenceTrustTier,
    EvidenceType,
    ReadinessLevel,
)
from .task_contract import (
    RequirementAssessmentEngine,
    TaskContract,
    derive_task_contract,
)
from .config import AgentConfig
from .failure import FailureAnalyzer
from .filesystem import ProjectFilesystem, ProtectedPathError, SandboxViolation
from .git import GitIntegration
from .models import (
    Checkpoint,
    ClarificationRequest,
    ExecutionResult,
    FailureAnalysis,
    FileOperation,
    ImplementationResult,
    ImplementationTurn,
    MultiTurnExecutionReport,
    MultiTurnState,
    Plan,
    ProjectContext,
    ProviderCapability,
    ProviderError,
    QuotaExceededError,
    RateLimitError,
    ReviewResult,
    RunReport,
    Subtask,
    SubtaskContract,
    SubtaskStatus,
    Task,
    TaskStatus,
    ToolCall,
    ToolDefinition,
    ToolExecutionMetrics,
    ToolExecutionPolicy,
    ToolResult,
)
from .providers import AIProvider
from .reviewer import Reviewer
from .storage import TaskStorage
from .tool_engine import ToolEngine, history_from_dict, history_to_dict
from .tools import CommandRunner, ToolRegistry

LOGGER = logging.getLogger(__name__)

DEFAULT_MAX_IMPLEMENTATION_TURNS = 10
DEFAULT_MAX_REPAIR_TURNS = 3
DEFAULT_MAX_REVIEW_TURNS = 2
DEFAULT_MAX_VERIFICATION_TURNS = 2
DEFAULT_MAX_TOOL_STEPS = 15


def _repair_ops_signature(ops: Sequence[FileOperation]) -> str:
    """Deterministic fingerprint of a proposed repair's actual content,
    independent of surrounding workspace state.

    Phase 4.24: the Orchestrator's single-turn repair loop has had duplicate-
    patch detection since Phase 4.5 (see RecoveryState.is_duplicate_patch /
    normalize_diff_for_signature in models.py), but MultiTurnImplementationAgent's
    REPAIRING stage had none at all -- only the turn-count budget
    (max_repair_turns) bounded it, so a provider that regenerates the exact
    same ineffective patch on every attempt could silently burn the entire
    repair budget on one distinct attempt instead of `max_repair_turns`
    genuinely different ones. This can't reuse `_compute_workspace_diff` /
    `normalize_diff_for_signature` for the signature: `_compute_workspace_diff`
    returns the *whole-repository* `git diff` when the project is a git repo,
    which grows every turn regardless of what this specific repair changed --
    comparing that across turns would never detect a real repeat. Hashing the
    proposed FileOperations' own content directly is scoped correctly to
    "what did THIS repair attempt actually propose".
    """
    parts: list[str] = []
    for op in sorted(ops, key=lambda o: (getattr(o, "path", "") or "", getattr(o, "action", "") or "")):
        body = (getattr(op, "content", None) or getattr(op, "patch", None) or "").strip()
        parts.append(f"{getattr(op, 'action', '')}\0{getattr(op, 'path', '')}\0{body}")
    normalized = "\n".join(parts)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


class MultiTurnImplementationAgent:
    """Bounded, autonomous multi-turn implementation agent with evidence lifecycle.

    Coordinates the implementation lifecycle across an explicit state machine:
    PLANNING -> IMPLEMENTING -> INSPECTING -> TESTING -> ANALYZING_FAILURE ->
    REPAIRING -> REVIEWING -> VERIFYING -> COMPLETED / FAILED / PAUSED.

    Guarantees:
    - Bounded execution with strict turn budgets for implementation, repair, review, and verification.
    - Durable turn-level telemetry and evidence ledger persisted at every state transition.
    - Scoped tool execution restricted to the safe implementation surface.
    - Authoritative unified diff reconstruction for accurate deliberative code review.
    - Rigorous multi-turn verification including full test execution and AST syntax validation.
    - Release readiness evaluation enforcing deterministic hard completion gates.
    - Automatic state recovery upon pause (rate limits / clarification requests).
    - Worktree isolation ensuring no mutable state is shared across parallel workers.
    """

    def __init__(
        self,
        config: AgentConfig,
        filesystem: ProjectFilesystem,
        registry: ToolRegistry,
        storage: TaskStorage,
        runner: CommandRunner | None = None,
        policy: ToolExecutionPolicy | None = None,
        allowed_tools: set[str] | frozenset[str] | None = None,
        checkpoint_callback: Callable[[ImplementationTurn, MultiTurnState], None] | None = None,
    ):
        self.config = config
        self.filesystem = filesystem
        self.registry = registry
        self.storage = storage
        self.runner = runner or CommandRunner(
            root=filesystem.root,
            timeout_seconds=getattr(config, "command_timeout_seconds", 120),
        )
        self.policy = policy
        self.allowed_tools = (
            frozenset(allowed_tools)
            if allowed_tools is not None
            else frozenset(IMPLEMENTATION_TOOL_SURFACE | {"ask_user_clarification"})
        )
        self.checkpoint_callback = checkpoint_callback

        self.max_turns = getattr(config, "max_implementation_turns", DEFAULT_MAX_IMPLEMENTATION_TURNS)
        self.max_repair_turns = getattr(config, "max_repair_turns", DEFAULT_MAX_REPAIR_TURNS)
        self.max_review_turns = getattr(config, "max_review_turns", DEFAULT_MAX_REVIEW_TURNS)
        self.max_verification_turns = getattr(config, "max_verification_turns", DEFAULT_MAX_VERIFICATION_TURNS)
        self.max_tool_steps = getattr(config, "max_implementation_tool_steps", DEFAULT_MAX_TOOL_STEPS)
        self.decision_engine = CompletionDecisionEngine(self.filesystem)

    def build_policy(self) -> ToolExecutionPolicy:
        """Constructs an implementation-scoped execution policy."""
        if self.policy is not None:
            return self.policy

        max_steps = getattr(self.config, "max_implementation_tool_steps", DEFAULT_MAX_TOOL_STEPS)
        max_bytes = getattr(self.config, "max_tool_output_bytes", 8000)
        total_budget = getattr(self.config, "total_tool_budget_bytes", 32000)
        max_repeats = getattr(self.config, "max_consecutive_repeats", 3)

        return ToolExecutionPolicy(
            max_tool_steps=max_steps,
            max_tool_output_bytes=max_bytes,
            total_tool_budget_bytes=total_budget,
            max_consecutive_repeats=max_repeats,
        )

    def build_turn_prompt(
        self,
        task_objective: str,
        subtask: Subtask | None,
        upstream_contracts: list[SubtaskContract] | None,
        context: ProjectContext,
        failure: FailureAnalysis | None = None,
        review: ReviewResult | None = None,
        turn_number: int = 1,
        stage: MultiTurnState = MultiTurnState.IMPLEMENTING,
    ) -> str:
        """Builds a structured, stage-aware context prompt for the turn."""
        lines = [
            f"# Objective: {task_objective}",
            "",
            f"MULTI-TURN IMPLEMENTATION AGENT (TURN {turn_number} / STAGE: {stage.value.upper()})",
            "=" * 65,
            "Execute the current implementation phase autonomously using tools.",
            "Inspect and verify before editing. Do not rewrite unmodified files.",
            "",
            "1. UNDERSTAND  - examine relevant code, signatures, and project structure.",
            "2. PLAN        - determine minimal, targeted changes required.",
            "3. IMPLEMENT   - produce precise file operations or patches.",
            "4. VERIFY      - validate changes against test suites and contract invariants.",
            "",
        ]

        if subtask:
            lines.extend([
                "### Active Subtask",
                f"- ID: {subtask.subtask_id}",
                f"- Title: {subtask.title}",
                f"- Goal: {subtask.goal}",
                "",
            ])

        if upstream_contracts:
            lines.extend([
                "### UPSTREAM INTERFACE CONSTRAINTS (MANDATORY)",
                "You must strictly respect the interfaces and behaviors exported by upstream subtasks:",
            ])
            for contract in upstream_contracts:
                lines.append(contract.format_for_prompt(max_chars=1500))
            lines.append("")

        if failure:
            lines.extend([
                "### PREVIOUS FAILURE / REPAIR DIAGNOSTICS",
                f"- Probable Root Cause: {failure.probable_root_cause}",
                f"- Recommended Fix: {failure.recommended_fix}",
            ])
            if failure.diagnostic_evidence:
                lines.append("- Diagnostic Evidence:")
                for item in failure.diagnostic_evidence[:5]:
                    lines.append(f"  * {item}")
            lines.append("")

        if review and review.findings:
            lines.extend([
                "### CODE REVIEW FINDINGS (CHANGES REQUIRED)",
                f"- Summary: {review.summary}",
                "- Findings:",
            ])
            for f in review.findings[:5]:
                lines.append(f"  * {f}")
            lines.append("")

        return "\n".join(lines)

    def _supports_tool_use(self, provider: Any) -> bool:
        """Determines if the provider supports tool use and interactive step generation."""
        if hasattr(provider, "generate_code_with_tools"):
            return True
        caps = getattr(provider, "capabilities", None)
        if isinstance(caps, (set, frozenset)) and ProviderCapability.TOOL_USE in caps:
            return True
        return False

    def _precheck_operations(self, operations: list[FileOperation], plan: Plan) -> list[str]:
        """Validates that proposed file operations are safe and conform to the plan."""
        errors: list[str] = []
        allowed = set(plan.allowed_paths) if hasattr(plan, "allowed_paths") else set(
            getattr(plan, "files_likely_to_change", []) + getattr(plan, "files_likely_to_create", [])
        )
        PROTECTED_EXACT = {"local_agent/tool_engine.py", "local_agent/approval.py"}
        for op in operations:
            if not op.path:
                errors.append("FileOperation missing target path.")
                continue
            norm_p = Path(op.path).as_posix().lstrip("./")
            if norm_p in PROTECTED_EXACT or any(part in {".git", ".hg", ".svn", ".agent_data", ".agent_worktrees"} for part in Path(norm_p).parts):
                errors.append(f"Operation targets protected path: {op.path}")
                continue
            act = getattr(op, "action", getattr(op, "operation_type", "modify"))
            if allowed and op.path not in allowed:
                if act != "create" and op.path not in getattr(plan, "files_likely_to_change", []):
                    LOGGER.info("Operation on path %s outside initial plan scope", op.path)
            if act == "modify" and not getattr(op, "patch", None) and not getattr(op, "content", None):
                errors.append(f"Modify operation on {op.path} has neither patch nor content.")
        return errors

    def _apply_file_operations(
        self,
        operations: list[FileOperation],
        plan: Plan,
    ) -> tuple[bool, str | None]:
        """Prepares and applies file operations to the workspace filesystem."""
        coding_agent = CodingAgent(self.filesystem)
        try:
            prepared = coding_agent.prepare(operations, plan)
            coding_agent.apply_prepared(prepared)
            return True, None
        except (
            UnsafeModificationError,
            SandboxViolation,
            ProtectedPathError,
            PatchValidationError,
            OSError,
        ) as exc:
            LOGGER.warning("Failed to apply file operations: %s", exc)
            return False, str(exc)

    def _run_validation_commands(
        self,
        context: ProjectContext,
        task_objective: str,
    ) -> list[ExecutionResult]:
        """Runs configured validation commands and returns ExecutionResults."""
        results: list[ExecutionResult] = []
        commands: list[str] = []

        if getattr(self.config, "validation_commands", None):
            commands.extend(self.config.validation_commands)
        elif getattr(context, "validation_commands", None):
            for cmd_spec in context.validation_commands:
                cmd_str = " ".join(cmd_spec.command) if hasattr(cmd_spec, "command") else str(cmd_spec)
                if cmd_str not in commands:
                    commands.append(cmd_str)

        if not commands:
            commands.append("python -m py_compile")

        for cmd in commands:
            cmd_parts = tuple(shlex.split(cmd))
            try:
                if cmd == "python -m py_compile":
                    py_files = [str(p) for p in self.filesystem.root.rglob("*.py") if ".git" not in p.parts][:10]
                    if py_files:
                        cmd_parts = ("python", "-m", "py_compile", *py_files)
                res = self.runner.run(cmd_parts)
                results.append(res)
            except Exception as exc:
                LOGGER.warning("Error running validation command '%s': %s", cmd, exc)
                results.append(ExecutionResult(
                    command=cmd,
                    exit_code=1,
                    stdout="",
                    stderr=str(exc),
                ))

        return results

    def _compute_workspace_diff(self, applied_ops: list[FileOperation]) -> str:
        """Computes authoritative unified diff against base repository state."""
        git = GitIntegration(self.filesystem.root)
        if git.is_repository():
            d = git.diff()
            if d:
                return d
        diff_chunks: list[str] = []
        for op in applied_ops:
            path = getattr(op, "path", None)
            if not path:
                continue
            try:
                current_content = self.filesystem.read_file(path)
            except (OSError, UnicodeDecodeError, ProtectedPathError, SandboxViolation):
                current_content = ""
            orig_content = getattr(op, "original", "") or ""
            diff_lines = list(difflib.unified_diff(
                orig_content.splitlines(keepends=True),
                current_content.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
            ))
            if diff_lines:
                diff_chunks.append("".join(diff_lines))
        return "\n".join(diff_chunks)

    def _checkpoint(
        self,
        task: Task,
        subtask: Subtask | None,
        turns: list[ImplementationTurn],
        turn: ImplementationTurn,
        stage: MultiTurnState,
        context: ProjectContext,
        evidence_store: CompletionEvidenceStore | None = None,
        assessment: CompletionAssessment | None = None,
        clarification_requests: list[Any] | None = None,
        requirement_assessment: dict[str, Any] | None = None,
    ) -> None:
        """Persists a durable checkpoint at turn boundary preserving full history."""
        now = datetime.datetime.now(datetime.timezone.utc)
        subtask_id = subtask.subtask_id if subtask else (getattr(task, "current_subtask_id", "main") or "main")
        cp_id = f"cp-{task.task_id}-{subtask_id}-t{turn.turn_number}-{stage.value}"

        all_turns = [t.to_dict() for t in turns]
        if not any(t.get("turn_id") == turn.turn_id for t in all_turns):
            all_turns.append(turn.to_dict())

        all_files: set[str] = set()
        for t_dict in all_turns:
            for op in t_dict.get("file_operations", []):
                if isinstance(op, dict) and op.get("path"):
                    all_files.add(op["path"])

        clarifications = [
            r.to_dict() if hasattr(r, "to_dict") else r
            for r in (clarification_requests or [])
        ]

        cp = Checkpoint(
            checkpoint_id=cp_id,
            task_id=task.task_id,
            subtask_id=subtask_id,
            timestamp=now,
            current_state_description=f"Turn {turn.turn_number} stage: {stage.value}",
            files_changed=sorted(list(all_files)),
            turns=all_turns,
            current_turn_number=turn.turn_number,
            turn_stage=stage.value,
            clarification_requests=clarifications,
            completion_assessment=assessment.to_dict() if assessment else None,
            completion_evidence=evidence_store.to_dict() if evidence_store else {},
            task_contract=task.task_contract,
            requirement_assessment=dict(requirement_assessment) if requirement_assessment else {},
        )

        try:
            self.storage.save_checkpoint(cp)
        except Exception as exc:
            LOGGER.warning("Failed to persist turn checkpoint %s: %s", cp_id, exc)

        if self.checkpoint_callback is not None:
            try:
                self.checkpoint_callback(turn, stage)
            except Exception as exc:
                LOGGER.warning("Checkpoint callback error: %s", exc)

    def execute(
        self,
        task: Task,
        subtask: Subtask | None,
        plan: Plan,
        context: ProjectContext,
        provider: Any,
        upstream_contracts: list[SubtaskContract] | None = None,
        failure: FailureAnalysis | None = None,
        review: ReviewResult | None = None,
        report: RunReport | None = None,
        progress: Callable[[str], None] | None = None,
        existing_turns: list[ImplementationTurn] | None = None,
        initial_state: MultiTurnState = MultiTurnState.IMPLEMENTING,
        worktree_id: str | None = None,
    ) -> MultiTurnExecutionReport:
        """Executes the bounded multi-turn implementation loop with evidence tracking."""
        start_time = time.perf_counter()
        task_id = task.task_id
        subtask_id = subtask.subtask_id if subtask else "subtask-main"
        task_objective = (
            subtask.goal if subtask and subtask.goal
            else (plan.objective if plan.objective else task.objective)
        )

        # Phase 4.21: derive the task's requirement contract once and persist
        # it on the Task itself (see Orchestrator.run() for the same pattern
        # and rationale).
        if not task.task_contract:
            task.task_contract = derive_task_contract(task, plan).to_dict()
        requirement_engine = RequirementAssessmentEngine(self.filesystem)

        provider_name = getattr(provider, "provider_id", "unknown")
        model_name = getattr(provider, "model", "unknown")

        def emit(msg: str) -> None:
            if progress:
                progress(f"[{task_id}:{subtask_id}] {msg}")
            LOGGER.info("[%s:%s] %s", task_id, subtask_id, msg)

        if "ask_user_clarification" in self.allowed_tools:
            self.registry.enable_clarification_tool()

        current_state = initial_state
        turns: list[ImplementationTurn] = list(existing_turns or [])
        repair_turn_count = sum(1 for t in turns if t.stage == MultiTurnState.REPAIRING.value)
        review_turn_count = sum(1 for t in turns if t.stage == MultiTurnState.REVIEWING.value)
        verification_turn_count = sum(1 for t in turns if t.stage == MultiTurnState.VERIFYING.value)
        # Phase 4.24: anti-repeat history for the REPAIRING stage, recomputed
        # from persisted turn history on resume (same pattern as
        # repair_turn_count above) so a checkpoint resume doesn't forget
        # which repairs were already tried and reopen an already-exhausted
        # repetition.
        repair_patch_hashes: set[str] = set()
        for t in turns:
            if t.stage == MultiTurnState.REPAIRING.value and t.file_operations:
                try:
                    reconstructed = [
                        FileOperation(**op) if isinstance(op, dict) else op
                        for op in t.file_operations
                    ]
                    repair_patch_hashes.add(_repair_ops_signature(reconstructed))
                except (TypeError, ValueError):
                    pass

        last_failure = failure
        last_review = review
        applied_operations: list[FileOperation] = []
        all_tool_metrics: list[ToolExecutionMetrics] = []
        current_tool_history: list[tuple[ToolCall, ToolResult]] = []
        clarification_requests: list[Any] = []

        evidence_store = CompletionEvidenceStore(self.filesystem.root)
        try:
            latest_cp = self.storage.load_latest_checkpoint(task_id, subtask_id=subtask_id)
            if latest_cp and latest_cp.completion_evidence:
                evidence_store = CompletionEvidenceStore.from_dict(latest_cp.completion_evidence)
                evidence_store.revalidate_against_disk(self.filesystem)
        except Exception:
            evidence_store = CompletionEvidenceStore(self.filesystem.root)

        for t in turns:
            if t.tool_calls and t.tool_results and len(t.tool_calls) == len(t.tool_results):
                for c_dict, r_dict in zip(t.tool_calls, t.tool_results):
                    c = ToolCall.from_dict(c_dict) if isinstance(c_dict, dict) else c_dict
                    r = ToolResult.from_dict(r_dict) if isinstance(r_dict, dict) else r_dict
                    current_tool_history.append((c, r))
            if t.file_operations:
                for op in t.file_operations:
                    applied_operations.append(FileOperation(**op) if isinstance(op, dict) else op)

        effective_policy = self.build_policy()
        termination_reason = "in_progress"
        error_message: str | None = None
        final_assessment: CompletionAssessment | None = None
        final_requirement_assessment: dict[str, Any] = {}

        emit(f"Starting multi-turn implementation (initial state: {current_state.value}).")

        while current_state not in (MultiTurnState.COMPLETED, MultiTurnState.FAILED, MultiTurnState.PAUSED):
            turn_num = len(turns) + 1
            if turn_num > self.max_turns:
                emit(f"Turn budget exceeded ({self.max_turns} turns). Terminating.")
                current_state = MultiTurnState.FAILED
                termination_reason = "max_turns_exceeded"
                error_message = f"Maximum implementation turns budget ({self.max_turns}) exceeded."
                break

            turn_id = f"turn-{task_id}-{subtask_id}-{turn_num}"

            # --- STAGE: IMPLEMENTING ---
            if current_state == MultiTurnState.IMPLEMENTING:
                emit(f"Turn {turn_num}: Implementing changes...")
                turn = ImplementationTurn(
                    turn_id=turn_id,
                    task_id=task_id,
                    subtask_id=subtask_id,
                    turn_number=turn_num,
                    stage=MultiTurnState.IMPLEMENTING.value,
                    provider=provider_name,
                    model=model_name,
                    started_at=datetime.datetime.now(datetime.timezone.utc),
                )

                prompt = self.build_turn_prompt(
                    task_objective=task_objective,
                    subtask=subtask,
                    upstream_contracts=upstream_contracts,
                    context=context,
                    failure=last_failure,
                    review=last_review,
                    turn_number=turn_num,
                    stage=MultiTurnState.IMPLEMENTING,
                )
                turn.prompt_summary = prompt[:300]

                ops: list[FileOperation] | None = None
                try:
                    if self._supports_tool_use(provider):
                        engine = ToolEngine(provider=provider, registry=self.registry, policy=effective_policy)
                        engine_res = engine.run(
                            task=prompt,
                            plan=plan,
                            context=context,
                            initial_history=current_tool_history,
                            failure=last_failure,
                            review=last_review,
                        )
                        current_tool_history = list(engine_res.tool_history)
                        turn.tool_calls = [c.to_dict() for c, _ in current_tool_history]
                        turn.tool_results = [r.to_dict() for _, r in current_tool_history]
                        if engine_res.metrics:
                            all_tool_metrics.append(engine_res.metrics)
                        ops = engine_res.file_operations
                        for c, r in current_tool_history:
                            if getattr(c, "tool_name", "") == "ask_user_clarification":
                                clarification_requests.append(
                                    ClarificationRequest(
                                        question_id=c.call_id,
                                        task_id=task_id,
                                        subtask_id=subtask_id,
                                        question=str(c.arguments.get("question", "")),
                                        choices=c.arguments.get("choices", []),
                                        status="answered" if not r.is_error else "pending",
                                        answer=r.output if not r.is_error else None,
                                    )
                                )
                                evidence_store.record(
                                    task_id=task_id,
                                    subtask_id=subtask_id,
                                    turn_number=turn_num,
                                    stage="implementing",
                                    evidence_type=EvidenceType.CLARIFICATION_RECORD,
                                    source="tool_engine",
                                    trust_tier=EvidenceTrustTier.AUTHORITATIVE_EXECUTION,
                                    payload={"question": c.arguments.get("question", ""), "answer": r.output},
                                    worktree_id=worktree_id,
                                )
                    else:
                        ops = provider.generate_code(
                            task=prompt,
                            plan=plan,
                            context=context,
                            failure=last_failure,
                            review=last_review,
                        )
                except (RateLimitError, QuotaExceededError) as exc:
                    emit(f"Provider paused during implementation: {exc}")
                    turn.status = "paused"
                    turn.error_message = str(exc)
                    turns.append(turn)
                    current_state = MultiTurnState.PAUSED
                    termination_reason = "provider_quota_paused"
                    self._checkpoint(task, subtask, turns, turn, MultiTurnState.PAUSED, context, evidence_store, final_assessment, clarification_requests)
                    break
                except Exception as exc:
                    emit(f"Implementation error: {exc}")
                    turn.status = "failed"
                    turn.error_message = str(exc)
                    turns.append(turn)
                    current_state = MultiTurnState.FAILED
                    termination_reason = "provider_implementation_error"
                    error_message = str(exc)
                    self._checkpoint(task, subtask, turns, turn, MultiTurnState.FAILED, context, evidence_store, final_assessment, clarification_requests)
                    break

                if not ops:
                    emit("No file operations produced in turn.")
                    turn.status = "failed"
                    turn.error_message = "No file operations produced"
                    turns.append(turn)
                    current_state = MultiTurnState.FAILED
                    termination_reason = "no_operations_emitted"
                    break

                precheck_errs = self._precheck_operations(ops, plan)
                if precheck_errs:
                    emit(f"Precheck errors: {precheck_errs}")
                    last_failure = FailureAnalysis(
                        probable_root_cause="Pre-mutation patch check failed",
                        recommended_fix="; ".join(precheck_errs),
                    )
                    turn.status = "completed"
                    turn.failures_detected = [{"type": "precheck_error", "errors": precheck_errs}]
                    turns.append(turn)
                    self._checkpoint(task, subtask, turns, turn, MultiTurnState.IMPLEMENTING, context, evidence_store, final_assessment, clarification_requests)
                    current_state = MultiTurnState.ANALYZING_FAILURE
                    continue

                applied, apply_err = self._apply_file_operations(ops, plan)
                if not applied:
                    emit(f"Apply error: {apply_err}")
                    last_failure = FailureAnalysis(
                        probable_root_cause="File operation apply error",
                        recommended_fix=apply_err or "Fix file patch",
                    )
                    turn.status = "completed"
                    turn.failures_detected = [{"type": "apply_error", "error": apply_err}]
                    turns.append(turn)
                    self._checkpoint(task, subtask, turns, turn, MultiTurnState.IMPLEMENTING, context, evidence_store, final_assessment, clarification_requests)
                    current_state = MultiTurnState.ANALYZING_FAILURE
                    continue

                applied_operations.extend(ops)
                turn.file_operations = [op.__dict__ if hasattr(op, "__dict__") else op for op in ops]
                turn.status = "completed"
                turn.completed_at = datetime.datetime.now(datetime.timezone.utc)
                turns.append(turn)

                mod_paths = [getattr(op, "path", "") for op in ops if getattr(op, "path", "")]
                evidence_store.invalidate_on_file_mutation(mod_paths, reason="new_implementation_changes")
                diff_str = self._compute_workspace_diff(applied_operations)
                evidence_store.record(
                    task_id=task_id,
                    subtask_id=subtask_id,
                    turn_number=turn_num,
                    stage="implementing",
                    evidence_type=EvidenceType.DIFF_INSPECTION,
                    source="workspace_diff",
                    trust_tier=EvidenceTrustTier.SYSTEM_INTEGRITY,
                    target_paths=mod_paths,
                    payload={"diff": diff_str[:1000], "ops_count": len(ops)},
                    worktree_id=worktree_id,
                )
                evidence_store.record(
                    task_id=task_id,
                    subtask_id=subtask_id,
                    turn_number=turn_num,
                    stage="implementing",
                    evidence_type=EvidenceType.SAFETY_INVARIANT,
                    source="filesystem_guard",
                    trust_tier=EvidenceTrustTier.SYSTEM_INTEGRITY,
                    target_paths=mod_paths,
                    payload={"protected_intact": True},
                    worktree_id=worktree_id,
                )

                self._checkpoint(task, subtask, turns, turn, MultiTurnState.IMPLEMENTING, context, evidence_store, final_assessment, clarification_requests)
                current_state = MultiTurnState.TESTING
                continue

            # --- STAGE: TESTING ---
            elif current_state == MultiTurnState.TESTING:
                emit(f"Turn {turn_num}: Running tests and validation...")
                turn = ImplementationTurn(
                    turn_id=turn_id,
                    task_id=task_id,
                    subtask_id=subtask_id,
                    turn_number=turn_num,
                    stage=MultiTurnState.TESTING.value,
                    provider=provider_name,
                    model=model_name,
                    started_at=datetime.datetime.now(datetime.timezone.utc),
                )

                test_results = self._run_validation_commands(context, task_objective)
                turn.tests_executed = [
                    {"command": r.command, "exit_code": r.exit_code, "succeeded": r.succeeded}
                    for r in test_results
                ]

                mod_paths = [getattr(op, "path", "") for op in applied_operations if getattr(op, "path", "")]
                for r in test_results:
                    evidence_store.record(
                        task_id=task_id,
                        subtask_id=subtask_id,
                        turn_number=turn_num,
                        stage="testing",
                        evidence_type=EvidenceType.TEST_EXECUTION,
                        source="command_runner",
                        trust_tier=EvidenceTrustTier.AUTHORITATIVE_EXECUTION,
                        target_paths=mod_paths,
                        command=r.command.split() if isinstance(r.command, str) else list(r.command),
                        exit_code=r.exit_code,
                        payload={"stdout": r.stdout[:400], "stderr": r.stderr[:400]},
                        worktree_id=worktree_id,
                    )

                all_passed = all(r.succeeded for r in test_results)
                if all_passed:
                    emit("All validation commands passed cleanly.")
                    turn.status = "completed"
                    turn.completed_at = datetime.datetime.now(datetime.timezone.utc)
                    turns.append(turn)
                    self._checkpoint(task, subtask, turns, turn, MultiTurnState.TESTING, context, evidence_store, final_assessment, clarification_requests)
                    current_state = MultiTurnState.REVIEWING
                else:
                    failed_cmds = [r for r in test_results if not r.succeeded]
                    emit(f"{len(failed_cmds)} validation command(s) failed.")
                    turn.failures_detected = [
                        {"command": r.command, "exit_code": r.exit_code, "stderr": r.stderr[:500]}
                        for r in failed_cmds
                    ]
                    turn.status = "completed"
                    turn.completed_at = datetime.datetime.now(datetime.timezone.utc)
                    turns.append(turn)
                    self._checkpoint(task, subtask, turns, turn, MultiTurnState.TESTING, context, evidence_store, final_assessment, clarification_requests)
                    first_fail = failed_cmds[0]
                    last_failure = FailureAnalysis(
                        probable_root_cause=f"Validation command failed: {first_fail.command} (exit code {first_fail.exit_code})",
                        recommended_fix="Address test failure or syntax error",
                        diagnostic_evidence=failed_cmds,
                    )
                    current_state = MultiTurnState.ANALYZING_FAILURE
                continue

            # --- STAGE: ANALYZING_FAILURE ---
            elif current_state == MultiTurnState.ANALYZING_FAILURE:
                emit(f"Turn {turn_num}: Analyzing failure...")
                turn = ImplementationTurn(
                    turn_id=turn_id,
                    task_id=task_id,
                    subtask_id=subtask_id,
                    turn_number=turn_num,
                    stage=MultiTurnState.ANALYZING_FAILURE.value,
                    provider=provider_name,
                    model=model_name,
                    started_at=datetime.datetime.now(datetime.timezone.utc),
                )

                if repair_turn_count >= self.max_repair_turns:
                    emit(f"Repair turn budget ({self.max_repair_turns}) exhausted.")
                    turn.status = "failed"
                    turn.error_message = f"Repair turn budget ({self.max_repair_turns}) exhausted."
                    turns.append(turn)
                    current_state = MultiTurnState.FAILED
                    termination_reason = "repair_budget_exhausted"
                    error_message = turn.error_message
                    self._checkpoint(task, subtask, turns, turn, MultiTurnState.FAILED, context, evidence_store, final_assessment, clarification_requests)
                    break

                turn.status = "completed"
                turn.completed_at = datetime.datetime.now(datetime.timezone.utc)
                turns.append(turn)

                evidence_store.record(
                    task_id=task_id,
                    subtask_id=subtask_id,
                    turn_number=turn_num,
                    stage="analyzing_failure",
                    evidence_type=EvidenceType.FAILURE_REPAIR,
                    source="failure_analyzer",
                    trust_tier=EvidenceTrustTier.DELIBERATIVE_REVIEW,
                    payload={"root_cause": last_failure.probable_root_cause if last_failure else "", "fix": last_failure.recommended_fix if last_failure else ""},
                    worktree_id=worktree_id,
                )

                self._checkpoint(task, subtask, turns, turn, MultiTurnState.ANALYZING_FAILURE, context, evidence_store, final_assessment, clarification_requests)
                current_state = MultiTurnState.REPAIRING
                continue

            # --- STAGE: REPAIRING ---
            elif current_state == MultiTurnState.REPAIRING:
                repair_turn_count += 1
                emit(f"Turn {turn_num}: Repairing defects (attempt {repair_turn_count}/{self.max_repair_turns})...")
                turn = ImplementationTurn(
                    turn_id=turn_id,
                    task_id=task_id,
                    subtask_id=subtask_id,
                    turn_number=turn_num,
                    stage=MultiTurnState.REPAIRING.value,
                    provider=provider_name,
                    model=model_name,
                    started_at=datetime.datetime.now(datetime.timezone.utc),
                )

                repair_prompt = self.build_turn_prompt(
                    task_objective=task_objective,
                    subtask=subtask,
                    upstream_contracts=upstream_contracts,
                    context=context,
                    failure=last_failure,
                    review=last_review,
                    turn_number=turn_num,
                    stage=MultiTurnState.REPAIRING,
                )
                turn.prompt_summary = repair_prompt[:300]
                turn.repair_reason = last_failure.probable_root_cause if last_failure else "Repairing detected defect"

                ops: list[FileOperation] | None = None
                try:
                    if self._supports_tool_use(provider):
                        engine = ToolEngine(provider=provider, registry=self.registry, policy=effective_policy)
                        engine_res = engine.run(
                            task=repair_prompt,
                            plan=plan,
                            context=context,
                            initial_history=current_tool_history,
                            failure=last_failure,
                            review=last_review,
                        )
                        current_tool_history = list(engine_res.tool_history)
                        turn.tool_calls = [c.to_dict() for c, _ in current_tool_history]
                        turn.tool_results = [r.to_dict() for _, r in current_tool_history]
                        if engine_res.metrics:
                            all_tool_metrics.append(engine_res.metrics)
                        ops = engine_res.file_operations
                    else:
                        ops = provider.generate_code(
                            task=repair_prompt,
                            plan=plan,
                            context=context,
                            failure=last_failure,
                            review=last_review,
                        )
                except (RateLimitError, QuotaExceededError) as exc:
                    emit(f"Provider paused during repair: {exc}")
                    turn.status = "paused"
                    turn.error_message = str(exc)
                    turns.append(turn)
                    current_state = MultiTurnState.PAUSED
                    termination_reason = "provider_quota_paused"
                    self._checkpoint(task, subtask, turns, turn, MultiTurnState.PAUSED, context, evidence_store, final_assessment, clarification_requests)
                    break
                except Exception as exc:
                    emit(f"Repair error: {exc}")
                    turn.status = "failed"
                    turn.error_message = str(exc)
                    turns.append(turn)
                    current_state = MultiTurnState.FAILED
                    termination_reason = "provider_repair_error"
                    error_message = str(exc)
                    self._checkpoint(task, subtask, turns, turn, MultiTurnState.FAILED, context, evidence_store, final_assessment, clarification_requests)
                    break

                if not ops:
                    emit("Repair produced no file operations.")
                    turn.failures_detected = [{"type": "empty_repair", "error": "No file operations produced in repair"}]
                    turn.status = "completed"
                    turn.completed_at = datetime.datetime.now(datetime.timezone.utc)
                    turns.append(turn)
                    self._checkpoint(task, subtask, turns, turn, MultiTurnState.REPAIRING, context, evidence_store, final_assessment, clarification_requests)
                    last_failure = FailureAnalysis(
                        probable_root_cause="Repair produced no operations",
                        recommended_fix="Provide valid file edits to resolve errors",
                    )
                    current_state = MultiTurnState.ANALYZING_FAILURE
                    continue

                # Phase 4.24: reject an exact repeat of a previously-tried
                # repair before writing it again -- see _repair_ops_signature
                # for why the Orchestrator's existing anti-repeat mechanism
                # (Phase 4.5) can't simply be reused here as-is.
                repair_sig = _repair_ops_signature(ops)
                if repair_sig in repair_patch_hashes:
                    emit(f"Repeated identical repair patch detected (signature {repair_sig}); stopping to prevent loop.")
                    turn.failures_detected = [{"type": "repeated_repair", "error": "Identical repair patch generated without progress"}]
                    turn.status = "failed"
                    turn.error_message = "Repeated identical repair patch generated without progress."
                    turns.append(turn)
                    current_state = MultiTurnState.FAILED
                    termination_reason = "repeated_repair_detected"
                    error_message = turn.error_message
                    self._checkpoint(task, subtask, turns, turn, MultiTurnState.FAILED, context, evidence_store, final_assessment, clarification_requests)
                    break
                repair_patch_hashes.add(repair_sig)

                applied, apply_err = self._apply_file_operations(ops, plan)
                if not applied:
                    emit(f"Repair apply error: {apply_err}")
                    turn.failures_detected = [{"type": "apply_error", "error": apply_err}]
                    turn.status = "completed"
                    turn.completed_at = datetime.datetime.now(datetime.timezone.utc)
                    turns.append(turn)
                    self._checkpoint(task, subtask, turns, turn, MultiTurnState.REPAIRING, context, evidence_store, final_assessment, clarification_requests)
                    last_failure = FailureAnalysis(
                        probable_root_cause="Repair file operation apply error",
                        recommended_fix=apply_err or "Fix file patch",
                    )
                    current_state = MultiTurnState.ANALYZING_FAILURE
                    continue

                applied_operations.extend(ops)
                turn.file_operations = [op.__dict__ if hasattr(op, "__dict__") else op for op in ops]
                turn.status = "completed"
                turn.completed_at = datetime.datetime.now(datetime.timezone.utc)
                turns.append(turn)

                mod_paths = [getattr(op, "path", "") for op in ops if getattr(op, "path", "")]
                evidence_store.invalidate_on_file_mutation(mod_paths, reason="repaired_after_execution")
                diff_str = self._compute_workspace_diff(applied_operations)
                evidence_store.record(
                    task_id=task_id,
                    subtask_id=subtask_id,
                    turn_number=turn_num,
                    stage="repairing",
                    evidence_type=EvidenceType.DIFF_INSPECTION,
                    source="workspace_diff",
                    trust_tier=EvidenceTrustTier.SYSTEM_INTEGRITY,
                    target_paths=mod_paths,
                    payload={"diff": diff_str[:1000]},
                    worktree_id=worktree_id,
                )
                evidence_store.record(
                    task_id=task_id,
                    subtask_id=subtask_id,
                    turn_number=turn_num,
                    stage="repairing",
                    evidence_type=EvidenceType.SAFETY_INVARIANT,
                    source="filesystem_guard",
                    trust_tier=EvidenceTrustTier.SYSTEM_INTEGRITY,
                    target_paths=mod_paths,
                    payload={"protected_intact": True},
                    worktree_id=worktree_id,
                )

                self._checkpoint(task, subtask, turns, turn, MultiTurnState.REPAIRING, context, evidence_store, final_assessment, clarification_requests)
                current_state = MultiTurnState.TESTING  # Re-test
                continue

            # --- STAGE: REVIEWING ---
            elif current_state == MultiTurnState.REVIEWING:
                emit(f"Turn {turn_num}: Reviewing implementation...")
                turn = ImplementationTurn(
                    turn_id=turn_id,
                    task_id=task_id,
                    subtask_id=subtask_id,
                    turn_number=turn_num,
                    stage=MultiTurnState.REVIEWING.value,
                    provider=provider_name,
                    model=model_name,
                    started_at=datetime.datetime.now(datetime.timezone.utc),
                )

                reconstructed_diff = self._compute_workspace_diff(applied_operations)
                reviewer = Reviewer(provider=provider, registry=self.registry, policy=effective_policy)
                try:
                    review_res = reviewer.review(
                        task=task_objective,
                        plan=plan,
                        diff=reconstructed_diff,
                        context=context,
                        initial_history=current_tool_history,
                        report=report,
                    )
                except Exception as exc:
                    LOGGER.warning("Review evaluation error, failing closed (no approval): %s", exc)
                    review_res = ReviewResult(
                        verdict="CHANGES_REQUIRED",
                        summary="Review could not be completed due to an internal error; treated as not approved",
                        findings=[f"Reviewer error: {exc}"],
                    )

                last_review = review_res
                mod_paths = [getattr(op, "path", "") for op in applied_operations if getattr(op, "path", "")]
                diff_hash = hashlib.sha256(reconstructed_diff.encode("utf-8")).hexdigest()[:16]

                evidence_store.record(
                    task_id=task_id,
                    subtask_id=subtask_id,
                    turn_number=turn_num,
                    stage="reviewing",
                    evidence_type=EvidenceType.CODE_REVIEW,
                    source="reviewer",
                    trust_tier=EvidenceTrustTier.DELIBERATIVE_REVIEW,
                    target_paths=mod_paths,
                    payload={"verdict": review_res.verdict, "summary": review_res.summary, "findings": review_res.findings, "diff_hash": diff_hash},
                    worktree_id=worktree_id,
                )

                if review_res.verdict == "APPROVED":
                    emit("Review approved changes.")
                    turn.status = "completed"
                    turn.completed_at = datetime.datetime.now(datetime.timezone.utc)
                    turns.append(turn)
                    self._checkpoint(task, subtask, turns, turn, MultiTurnState.REVIEWING, context, evidence_store, final_assessment, clarification_requests)
                    current_state = MultiTurnState.VERIFYING
                else:
                    review_turn_count += 1
                    emit(f"Review requested changes (attempt {review_turn_count}/{self.max_review_turns}).")
                    if review_turn_count > self.max_review_turns:
                        emit(f"Review turn budget ({self.max_review_turns}) exhausted.")
                        turn.status = "failed"
                        turn.error_message = f"Review turn budget ({self.max_review_turns}) exhausted."
                        turns.append(turn)
                        current_state = MultiTurnState.FAILED
                        termination_reason = "review_budget_exhausted"
                        error_message = turn.error_message
                        self._checkpoint(task, subtask, turns, turn, MultiTurnState.FAILED, context, evidence_store, final_assessment, clarification_requests)
                        break

                    last_failure = FailureAnalysis(
                        probable_root_cause="Review findings required correction",
                        recommended_fix="; ".join(review_res.findings),
                    )
                    turn.status = "completed"
                    turn.completed_at = datetime.datetime.now(datetime.timezone.utc)
                    turns.append(turn)
                    self._checkpoint(task, subtask, turns, turn, MultiTurnState.REVIEWING, context, evidence_store, final_assessment, clarification_requests)
                    current_state = MultiTurnState.REPAIRING
                continue

            # --- STAGE: VERIFYING ---
            elif current_state == MultiTurnState.VERIFYING:
                emit(f"Turn {turn_num}: Final verification...")
                turn = ImplementationTurn(
                    turn_id=turn_id,
                    task_id=task_id,
                    subtask_id=subtask_id,
                    turn_number=turn_num,
                    stage=MultiTurnState.VERIFYING.value,
                    provider=provider_name,
                    model=model_name,
                    started_at=datetime.datetime.now(datetime.timezone.utc),
                )

                test_results = self._run_validation_commands(context, task_objective)
                turn.tests_executed = [
                    {"command": r.command, "exit_code": r.exit_code, "succeeded": r.succeeded}
                    for r in test_results
                ]
                failed_cmds = [r for r in test_results if not r.succeeded]

                mod_paths = [getattr(op, "path", "") for op in applied_operations if getattr(op, "path", "")]
                for r in test_results:
                    evidence_store.record(
                        task_id=task_id,
                        subtask_id=subtask_id,
                        turn_number=turn_num,
                        stage="verifying",
                        evidence_type=EvidenceType.TEST_EXECUTION,
                        source="command_runner",
                        trust_tier=EvidenceTrustTier.AUTHORITATIVE_EXECUTION,
                        target_paths=mod_paths,
                        command=r.command.split() if isinstance(r.command, str) else list(r.command),
                        exit_code=r.exit_code,
                        payload={"stdout": r.stdout[:400], "stderr": r.stderr[:400]},
                        worktree_id=worktree_id,
                    )

                syntax_errors: list[str] = []
                for op in applied_operations:
                    p = getattr(op, "path", None)
                    if p and str(p).endswith(".py"):
                        full_path = self.filesystem.root / p
                        if full_path.is_file():
                            try:
                                content = self.filesystem.read_file(p)
                                ast.parse(content, filename=p)
                                evidence_store.record(
                                    task_id=task_id,
                                    subtask_id=subtask_id,
                                    turn_number=turn_num,
                                    stage="verifying",
                                    evidence_type=EvidenceType.SYNTAX_VERIFICATION,
                                    source="ast_parser",
                                    trust_tier=EvidenceTrustTier.SYSTEM_INTEGRITY,
                                    target_paths=[p],
                                    exit_code=0,
                                    worktree_id=worktree_id,
                                )
                            except Exception as syn_exc:
                                syntax_errors.append(f"Syntax error in {p}: {syn_exc}")
                                evidence_store.record(
                                    task_id=task_id,
                                    subtask_id=subtask_id,
                                    turn_number=turn_num,
                                    stage="verifying",
                                    evidence_type=EvidenceType.SYNTAX_VERIFICATION,
                                    source="ast_parser",
                                    trust_tier=EvidenceTrustTier.SYSTEM_INTEGRITY,
                                    target_paths=[p],
                                    exit_code=1,
                                    payload={"error": str(syn_exc)},
                                    worktree_id=worktree_id,
                                )

                if failed_cmds or syntax_errors:
                    verification_turn_count += 1
                    emit(f"Verification failed (attempt {verification_turn_count}/{self.max_verification_turns}): {len(failed_cmds)} command(s) failed, {len(syntax_errors)} syntax error(s).")
                    turn.failures_detected = [
                        {"command": r.command, "exit_code": r.exit_code, "stderr": r.stderr[:500]}
                        for r in failed_cmds
                    ] + [{"type": "syntax_error", "error": err} for err in syntax_errors]

                    if verification_turn_count > self.max_verification_turns:
                        emit(f"Verification turn budget ({self.max_verification_turns}) exhausted.")
                        turn.status = "failed"
                        turn.error_message = f"Verification turn budget ({self.max_verification_turns}) exhausted."
                        turns.append(turn)
                        current_state = MultiTurnState.FAILED
                        termination_reason = "verification_budget_exhausted"
                        error_message = turn.error_message
                        self._checkpoint(task, subtask, turns, turn, MultiTurnState.FAILED, context, evidence_store, final_assessment, clarification_requests)
                        break

                    err_summary = "; ".join(syntax_errors) if syntax_errors else (failed_cmds[0].stderr[:200] if failed_cmds[0].stderr else f"Exit code {failed_cmds[0].exit_code}")
                    last_failure = FailureAnalysis(
                        probable_root_cause=f"Final verification failed: {err_summary}",
                        recommended_fix="Address verification failure",
                        diagnostic_evidence=failed_cmds,
                    )
                    turn.status = "completed"
                    turn.completed_at = datetime.datetime.now(datetime.timezone.utc)
                    turns.append(turn)
                    self._checkpoint(task, subtask, turns, turn, MultiTurnState.VERIFYING, context, evidence_store, final_assessment, clarification_requests)
                    current_state = MultiTurnState.REPAIRING
                    continue

                reconstructed_diff = self._compute_workspace_diff(applied_operations)
                assessment = self.decision_engine.evaluate(
                    task=task,
                    subtask=subtask,
                    plan=plan,
                    evidence_store=evidence_store,
                    applied_operations=applied_operations,
                    current_diff=reconstructed_diff,
                    last_review=last_review,
                    last_failure=last_failure,
                    worktree_id=worktree_id,
                    clarification_requests=clarification_requests,
                )
                final_assessment = assessment

                # Phase 4.21: technical readiness alone is not completion --
                # see Orchestrator.run() for the identical rationale.
                contract_obj = TaskContract.from_dict(task.task_contract) if task.task_contract else TaskContract(task_id=task_id, objective=task_objective)
                req_assessment = requirement_engine.assess(
                    contract=contract_obj,
                    evidence_store=evidence_store,
                    applied_operations=applied_operations,
                    current_diff=reconstructed_diff,
                    last_review=last_review,
                    clarification_requests=clarification_requests,
                )
                final_requirement_assessment = req_assessment.to_dict()
                task.task_contract = contract_obj.to_dict()
                final_ready = assessment.is_ready and req_assessment.satisfied

                if final_ready:
                    emit(f"All final verification checks, completion gates, and task requirements passed cleanly ({assessment.readiness_level}).")
                    turn.status = "completed"
                    turn.completed_at = datetime.datetime.now(datetime.timezone.utc)
                    turns.append(turn)
                    self._checkpoint(task, subtask, turns, turn, MultiTurnState.VERIFYING, context, evidence_store, final_assessment, clarification_requests, final_requirement_assessment)
                    current_state = MultiTurnState.COMPLETED
                    termination_reason = "completed"
                    emit("Multi-turn implementation completed successfully.")
                    break
                else:
                    reason = assessment.decision_reason if not assessment.is_ready else req_assessment.decision_reason
                    emit(f"Completion gates failed: {reason}")
                    verification_turn_count += 1
                    if verification_turn_count > self.max_verification_turns:
                        turn.status = "failed"
                        turn.error_message = f"Completion gate failure: {reason}"
                        turns.append(turn)
                        current_state = MultiTurnState.FAILED
                        termination_reason = "verification_budget_exhausted"
                        error_message = turn.error_message
                        self._checkpoint(task, subtask, turns, turn, MultiTurnState.FAILED, context, evidence_store, final_assessment, clarification_requests, final_requirement_assessment)
                        break

                    last_failure = FailureAnalysis(
                        probable_root_cause=f"Completion gate failure: {reason}",
                        recommended_fix="Address missing completion evidence or unresolved task requirements",
                    )
                    turn.status = "completed"
                    turn.completed_at = datetime.datetime.now(datetime.timezone.utc)
                    turns.append(turn)
                    self._checkpoint(task, subtask, turns, turn, MultiTurnState.VERIFYING, context, evidence_store, final_assessment, clarification_requests, final_requirement_assessment)
                    current_state = MultiTurnState.REPAIRING
                    continue

        elapsed = time.perf_counter() - start_time
        success = (current_state == MultiTurnState.COMPLETED)

        if final_assessment is None:
            reconstructed_diff = self._compute_workspace_diff(applied_operations)
            final_assessment = self.decision_engine.evaluate(
                task=task,
                subtask=subtask,
                plan=plan,
                evidence_store=evidence_store,
                applied_operations=applied_operations,
                current_diff=reconstructed_diff,
                last_review=last_review,
                last_failure=last_failure,
                worktree_id=worktree_id,
                clarification_requests=clarification_requests,
            )
            contract_obj = TaskContract.from_dict(task.task_contract) if task.task_contract else TaskContract(task_id=task_id, objective=task_objective)
            final_requirement_assessment = requirement_engine.assess(
                contract=contract_obj,
                evidence_store=evidence_store,
                applied_operations=applied_operations,
                current_diff=reconstructed_diff,
                last_review=last_review,
                clarification_requests=clarification_requests,
            ).to_dict()
            task.task_contract = contract_obj.to_dict()

        if report is not None:
            report.completion_assessment = final_assessment
            report.completion_evidence = evidence_store.to_dict()
            report.task_contract = task.task_contract
            report.requirement_assessment = final_requirement_assessment

        return MultiTurnExecutionReport(
            task_id=task_id,
            subtask_id=subtask_id,
            success=success,
            turns=turns,
            total_turns=len(turns),
            repair_turns=repair_turn_count,
            review_turns=review_turn_count,
            final_state=current_state.value,
            termination_reason=termination_reason,
            elapsed_time_seconds=elapsed,
            file_operations=applied_operations,
            tool_metrics=all_tool_metrics,
            error_message=error_message,
            completion_assessment=final_assessment,
            completion_evidence=evidence_store.to_dict(),
            task_contract=task.task_contract,
            requirement_assessment=final_requirement_assessment,
        )
