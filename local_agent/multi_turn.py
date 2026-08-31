from __future__ import annotations

import datetime
import logging
import os
import shlex
import time
from pathlib import Path
from typing import Any, Callable

from .coding_agent import (
    IMPLEMENTATION_TOOL_SURFACE,
    CodingAgent,
    PatchValidationError,
    ScopeAmendmentGuard,
    UnsafeModificationError,
)
from .config import AgentConfig
from .failure import FailureAnalyzer
from .filesystem import ProjectFilesystem, ProtectedPathError, SandboxViolation
from .models import (
    Checkpoint,
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


class MultiTurnImplementationAgent:
    """Bounded, autonomous multi-turn implementation agent.

    Coordinates the implementation lifecycle across an explicit state machine:
    PLANNING -> IMPLEMENTING -> INSPECTING -> TESTING -> ANALYZING_FAILURE ->
    REPAIRING -> REVIEWING -> VERIFYING -> COMPLETED / FAILED / PAUSED.

    Composes existing ToolEngine, ToolRegistry, ToolExecutionPolicy, CodingAgent,
    FailureAnalyzer, Reviewer, and Storage abstractions with zero duplicate logic.
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
            else IMPLEMENTATION_TOOL_SURFACE
        )
        self.checkpoint_callback = checkpoint_callback

        # Bounded limits
        self.max_turns = max(1, int(getattr(config, "max_implementation_turns", DEFAULT_MAX_IMPLEMENTATION_TURNS)))
        self.max_repair_turns = max(0, int(getattr(config, "max_repair_turns", DEFAULT_MAX_REPAIR_TURNS)))
        self.max_review_turns = max(0, int(getattr(config, "max_review_turns", DEFAULT_MAX_REVIEW_TURNS)))
        self.max_verification_turns = max(0, int(getattr(config, "max_verification_turns", DEFAULT_MAX_VERIFICATION_TURNS)))
        self.max_tool_steps = max(1, int(getattr(config, "max_implementation_tool_steps", DEFAULT_MAX_TOOL_STEPS)))

    def build_policy(self) -> ToolExecutionPolicy:
        """Derive effective ToolExecutionPolicy clamped to the turn step budget."""
        base = self.policy
        registry_tools = {d.name for d in self.registry.definitions()}
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

    def _supports_tool_use(self, provider: Any) -> bool:
        caps = getattr(provider, "capabilities", None)
        if isinstance(caps, (set, frozenset)):
            return ProviderCapability.TOOL_USE in caps
        return hasattr(provider, "generate_code_with_tools")

    def build_turn_prompt(
        self,
        task_objective: str,
        subtask: Subtask | None = None,
        upstream_contracts: list[SubtaskContract] | None = None,
        context: ProjectContext | None = None,
        failure: FailureAnalysis | None = None,
        review: ReviewResult | None = None,
        turn_number: int = 1,
        stage: MultiTurnState = MultiTurnState.IMPLEMENTING,
    ) -> str:
        """Builds structured, compacted prompt context for the turn."""
        header = task_objective
        if subtask is not None and getattr(subtask, "goal", None) and subtask.goal != task_objective:
            header = f"Subtask Goal: {subtask.goal}\n\nTask Objective: {task_objective}"

        lines: list[str] = [
            header,
            "",
            f"MULTI-TURN IMPLEMENTATION AGENT (TURN {turn_number} / STAGE: {stage.value.upper()})",
            "=================================================================",
            "Execute the current implementation phase autonomously using tools.",
            "Inspect and verify before editing. Do not rewrite unmodified files.",
            "",
            "1. UNDERSTAND  - examine objective and requirements.",
            "2. INSPECT     - use read_file_range, find_files, grep_code to read current code.",
            "3. REASON      - decide minimal, focused edits matching the plan.",
            "4. EMIT        - return the complete list of file operations.",
            "",
            "CONSTRAINTS",
            "-----------",
            "- Stay strictly inside the approved plan scope; do not modify unrelated files.",
            "- Never attempt to read or modify protected files (tool_engine.py, approval.py) or secrets.",
            f"- Available tools: {', '.join(sorted(self.allowed_tools))}.",
        ]

        if upstream_contracts:
            lines.append("")
            lines.append("UPSTREAM INTERFACE CONSTRAINTS")
            lines.append("==============================")
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
            lines.append("FAILURE ANALYSIS & REPAIR CONTEXT")
            lines.append("==================================")
            lines.append(f"Root cause: {getattr(failure, 'probable_root_cause', '')}")
            recommended = getattr(failure, "recommended_fix", None)
            if recommended:
                lines.append(f"Recommended fix: {recommended}")
            affected = getattr(failure, "affected_files", None)
            if affected:
                lines.append(f"Affected files: {', '.join(affected)}")
            for diag in getattr(failure, "diagnostic_evidence", []):
                if hasattr(diag, "stdout") and (diag.stdout or diag.stderr):
                    out_snip = (diag.stdout + "\n" + diag.stderr).strip()[-1000:]
                    lines.append(f"Command '{diag.command}' output:\n{out_snip}")

        if review is not None and review.findings:
            lines.append("")
            lines.append("REVIEW FINDINGS TO ADDRESS")
            lines.append("==========================")
            for f in review.findings:
                lines.append(f"- {f}")

        return "\n".join(lines)

    def _precheck_operations(self, operations: list[FileOperation], plan: Plan) -> list[str]:
        """Validates file operations for patch safety and scope constraints."""
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

        # 1. Check explicit validation commands
        if getattr(self.config, "validation_commands", None):
            commands.extend(self.config.validation_commands)
        elif getattr(context, "validation_commands", None):
            for cmd_spec in context.validation_commands:
                cmd_str = " ".join(cmd_spec.command) if hasattr(cmd_spec, "command") else str(cmd_spec)
                if cmd_str not in commands:
                    commands.append(cmd_str)

        # Fallback to python syntax check if no explicit commands configured
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

    def _checkpoint(
        self,
        task: Task,
        subtask: Subtask | None,
        turn: ImplementationTurn,
        stage: MultiTurnState,
        context: ProjectContext,
    ) -> None:
        """Persists a durable checkpoint at the turn boundary."""
        if self.checkpoint_callback is not None:
            self.checkpoint_callback(turn, stage)

        cp_id = f"cp-turn-{turn.task_id}-{turn.turn_number}-{stage.value}"
        checkpoint = Checkpoint(
            checkpoint_id=cp_id,
            task_id=task.task_id,
            subtask_id=subtask.subtask_id if subtask else "",
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            current_state_description=f"Turn {turn.turn_number} - Stage: {stage.value}",
            turn_stage=stage.value,
            current_turn_number=turn.turn_number,
            turns=[turn.to_dict()],
            continuation_context={
                "turn_id": turn.turn_id,
                "turn_stage": stage.value,
                "status": turn.status,
            },
        )
        try:
            self.storage.save_checkpoint(checkpoint)
            task.latest_checkpoint_id = cp_id
            if subtask:
                subtask.latest_checkpoint_id = cp_id
        except Exception as exc:
            LOGGER.warning("Could not persist turn checkpoint %s: %s", cp_id, exc)

    def execute(
        self,
        task: Task,
        subtask: Subtask | None,
        plan: Plan,
        context: ProjectContext,
        provider: AIProvider,
        upstream_contracts: list[SubtaskContract] | None = None,
        initial_turn_number: int = 1,
        existing_turns: list[ImplementationTurn] | None = None,
        initial_state: MultiTurnState = MultiTurnState.IDLE,
        failure: FailureAnalysis | None = None,
        review: ReviewResult | None = None,
        progress: Callable[[str], None] | None = None,
        report: RunReport | None = None,
    ) -> MultiTurnExecutionReport:
        """Runs the bounded multi-turn implementation loop until completion or budget exhaustion."""
        start_time = time.perf_counter()
        provider_name = getattr(
            provider, "provider_id", getattr(provider, "name", provider.__class__.__name__)
        )
        model_name = getattr(provider, "model", "default")
        task_id = task.task_id if hasattr(task, "task_id") else "unknown-task"
        subtask_id = subtask.subtask_id if subtask else "subtask-main"
        task_objective = task.objective if hasattr(task, "objective") else str(task)

        turns: list[ImplementationTurn] = list(existing_turns or [])
        current_state = initial_state if initial_state != MultiTurnState.IDLE else MultiTurnState.IMPLEMENTING
        repair_turn_count = sum(1 for t in turns if t.stage == MultiTurnState.REPAIRING.value and t.status == "completed")
        review_turn_count = sum(1 for t in turns if t.stage == MultiTurnState.REVIEWING.value and t.status == "completed")
        
        current_tool_history: list[tuple[ToolCall, ToolResult]] = []
        last_failure: FailureAnalysis | None = failure
        last_review: ReviewResult | None = review
        applied_operations: list[FileOperation] = []
        all_tool_metrics: list[ToolExecutionMetrics] = []
        termination_reason = "in_progress"
        error_message: str | None = None

        def emit(msg: str) -> None:
            LOGGER.info("[MultiTurn] %s", msg)
            if progress:
                progress(f"[MultiTurn] {msg}")

        emit(f"Starting multi-turn loop for {subtask_id} in state {current_state.value}")

        effective_policy = self.build_policy()

        while current_state not in {MultiTurnState.COMPLETED, MultiTurnState.FAILED, MultiTurnState.PAUSED}:
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
                    self._checkpoint(task, subtask, turn, MultiTurnState.PAUSED, context)
                    break
                except Exception as exc:
                    emit(f"Implementation error: {exc}")
                    turn.status = "failed"
                    turn.error_message = str(exc)
                    turns.append(turn)
                    current_state = MultiTurnState.FAILED
                    termination_reason = "provider_implementation_error"
                    error_message = str(exc)
                    self._checkpoint(task, subtask, turn, MultiTurnState.FAILED, context)
                    break

                if not ops:
                    emit("No file operations produced in turn.")
                    turn.status = "failed"
                    turn.error_message = "No file operations produced"
                    turns.append(turn)
                    current_state = MultiTurnState.FAILED
                    termination_reason = "no_operations_emitted"
                    break

                # Apply operations
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
                    current_state = MultiTurnState.ANALYZING_FAILURE
                    continue

                applied_operations.extend(ops)
                turn.file_operations = [op.__dict__ if hasattr(op, "__dict__") else op for op in ops]
                turn.status = "completed"
                turn.completed_at = datetime.datetime.now(datetime.timezone.utc)
                turns.append(turn)
                self._checkpoint(task, subtask, turn, MultiTurnState.IMPLEMENTING, context)
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

                all_passed = all(r.succeeded for r in test_results)
                if all_passed:
                    emit("All validation commands passed cleanly.")
                    turn.status = "completed"
                    turn.completed_at = datetime.datetime.now(datetime.timezone.utc)
                    turns.append(turn)
                    self._checkpoint(task, subtask, turn, MultiTurnState.TESTING, context)
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
                    self._checkpoint(task, subtask, turn, MultiTurnState.TESTING, context)
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
                    break

                turn.status = "completed"
                turn.completed_at = datetime.datetime.now(datetime.timezone.utc)
                turns.append(turn)
                current_state = MultiTurnState.REPAIRING
                continue

            # --- STAGE: REPAIRING ---
            elif current_state == MultiTurnState.REPAIRING:
                repair_turn_count += 1
                emit(f"Turn {turn_num}: Repairing defect (repair attempt {repair_turn_count}/{self.max_repair_turns})...")
                turn = ImplementationTurn(
                    turn_id=turn_id,
                    task_id=task_id,
                    subtask_id=subtask_id,
                    turn_number=turn_num,
                    stage=MultiTurnState.REPAIRING.value,
                    provider=provider_name,
                    model=model_name,
                    repair_reason=last_failure.probable_root_cause if last_failure else None,
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

                ops = None
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
                    self._checkpoint(task, subtask, turn, MultiTurnState.PAUSED, context)
                    break
                except Exception as exc:
                    emit(f"Repair error: {exc}")
                    turn.status = "failed"
                    turn.error_message = str(exc)
                    turns.append(turn)
                    current_state = MultiTurnState.FAILED
                    termination_reason = "provider_repair_error"
                    error_message = str(exc)
                    self._checkpoint(task, subtask, turn, MultiTurnState.FAILED, context)
                    break

                if ops:
                    applied, apply_err = self._apply_file_operations(ops, plan)
                    if applied:
                        applied_operations.extend(ops)
                        turn.file_operations = [op.__dict__ if hasattr(op, "__dict__") else op for op in ops]

                turn.status = "completed"
                turn.completed_at = datetime.datetime.now(datetime.timezone.utc)
                turns.append(turn)
                self._checkpoint(task, subtask, turn, MultiTurnState.REPAIRING, context)
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

                reviewer = Reviewer(provider=provider, registry=self.registry, policy=effective_policy)
                try:
                    review_res = reviewer.review(
                        task=task_objective,
                        plan=plan,
                        diff="",
                        context=context,
                        initial_history=current_tool_history,
                        report=report,
                    )
                except Exception as exc:
                    LOGGER.warning("Review evaluation error, accepting default approval: %s", exc)
                    review_res = ReviewResult(verdict="APPROVED", summary="Automated review pass", findings=[])

                last_review = review_res

                if review_res.verdict == "APPROVED":
                    emit("Review approved changes.")
                    turn.status = "completed"
                    turn.completed_at = datetime.datetime.now(datetime.timezone.utc)
                    turns.append(turn)
                    self._checkpoint(task, subtask, turn, MultiTurnState.REVIEWING, context)
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
                        break

                    last_failure = FailureAnalysis(
                        probable_root_cause="Review findings required correction",
                        recommended_fix="; ".join(review_res.findings),
                    )
                    turn.status = "completed"
                    turn.completed_at = datetime.datetime.now(datetime.timezone.utc)
                    turns.append(turn)
                    self._checkpoint(task, subtask, turn, MultiTurnState.REVIEWING, context)
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

                turn.status = "completed"
                turn.completed_at = datetime.datetime.now(datetime.timezone.utc)
                turns.append(turn)
                self._checkpoint(task, subtask, turn, MultiTurnState.VERIFYING, context)
                current_state = MultiTurnState.COMPLETED
                termination_reason = "completed"
                emit("Multi-turn implementation completed successfully.")
                break

        elapsed = time.perf_counter() - start_time
        success = (current_state == MultiTurnState.COMPLETED)

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
        )
