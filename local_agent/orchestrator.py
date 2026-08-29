from __future__ import annotations

import datetime
import logging
import shlex
import uuid
from pathlib import Path
from typing import Any

import threading
from .approval import ApprovalPolicyEngine, RISK_LEVEL_MAP
from .coding_agent import (
    DEFAULT_MAX_IMPLEMENTATION_TOOL_STEPS,
    CodingAgent,
    InteractiveCodingAgent,
    PatchValidationError,
    ScopeAmendmentGuard,
    UnsafeModificationError,
)
from .commands import CommandRunner
from .config import AgentConfig
from .context import ContextSelector
from .failure import FailureAnalyzer
from .filesystem import ProjectFilesystem
from .git import GitIntegration
from .impact import ChangeImpactAnalyzer
from .knowledge import KnowledgeGraphManager
from .models import (
    ApprovalPolicy,
    ChangeImpact,
    Checkpoint,
    CommandSpec,
    DAGAmendmentGuard,
    DAGProposal,
    ExecutionResult,
    FailureAnalysis,
    ImplementationResult,
    Plan,
    PlanAmendment,
    PlanProposal,
    ProjectMemory,
    Memory,
    ProjectContext,
    ProviderCapability,
    RecoveryState,
    RepairSignature,
    ReviewConsensusRecord,
    ReviewResult,
    RunReport,
    ScopeExpansionProposal,
    SpecialistRole,
    Subtask,
    SubtaskContract,
    SubtaskStatus,
    Task,
    TaskPlan,
    TaskPlanAmendment,
    TaskStatus,
    TestExecutionRecord,
    ToolCall,
    ToolExecutionMetrics,
    ToolResult,
    ValidationPlan,
    VerificationGap,
    normalize_diff_for_signature,
)
from .contract_extractor import ContractExtractor
from .models import MemoryCategory
from .planner import GraphValidator, Planner
from .providers import (
    AIProvider,
    ProviderError,
    QuotaExceededError,
    RateLimitError,
    SpecialistModelRouter,
    build_provider,
)
from .repository import RepositoryIntelligence
from .reviewer import DeliberativeReviewConsensus, Reviewer
from .evidence import EvidenceLedger, compute_executable_fingerprint, compute_policy_fingerprint
from .sandbox import CandidateWorkspace, CandidateWorkspaceError, ProspectiveValidator
from .semantic_impact import (
    SEMANTIC_ANALYZER_SCHEMA_VERSION,
    ChangeImpactReport,
    SemanticChangeImpactAnalyzer,
    apply_knowledge_support,
)
from .test_synthesizer import BehavioralVerifier, TestSynthesizer, VerificationGapAnalyzer
from .validation_decision import ValidationDecisionEngine
from .tool_engine import IterationHistoryCompactor, ToolEngine, history_from_dict, history_to_dict
from .tools import ToolRegistry
from .validation import ValidationIntelligence

LOGGER = logging.getLogger(__name__)


def _context_semantic_index(context: Any) -> Any | None:
    """Return the project's :class:`SemanticIndex`, wherever it is stored.

    :class:`~local_agent.models.ProjectContext` has no ``semantic_index``
    attribute - :meth:`RepositoryIntelligence.scan` puts the index in
    ``context.metadata["semantic_index"]``. Earlier call sites used
    ``getattr(context, "semantic_index", None)``, which therefore always
    evaluated to ``None`` and silently disabled the ``search_symbols`` tool.
    The attribute is still checked first so a caller that supplies a
    context-like object carrying one directly keeps working.
    """
    direct = getattr(context, "semantic_index", None)
    if direct is not None:
        return direct
    metadata = getattr(context, "metadata", None)
    if isinstance(metadata, dict):
        return metadata.get("semantic_index")
    return None


class Orchestrator:
    def __init__(self, config: AgentConfig, storage: "TaskStorage", scheduler: "Scheduler", repo_lock: threading.Lock, memory_lock: threading.Lock):
        self.config = config
        self.storage = storage
        self.scheduler = scheduler
        self.repo_lock = repo_lock
        self.memory_lock = memory_lock
        cred_store = getattr(self.scheduler, "credential_store", None) if self.scheduler else None
        sched_state = getattr(self.scheduler, "state", None) if self.scheduler else None
        self.router = SpecialistModelRouter(
            config,
            credential_store=cred_store,
            scheduler_state=sched_state,
            provider_factory=lambda cfg, api_key=None: build_provider(cfg, api_key=api_key),
        )
        self.analyzer = RepositoryIntelligence(config.project)
        self.filesystem = ProjectFilesystem(config.project)
        self.git = GitIntegration(config.project)
        self.impact_analyzer = ChangeImpactAnalyzer(config.project)
        self.runner = CommandRunner(config.project, config.command_timeout_seconds)
        self.validation_intelligence = ValidationIntelligence(config.project)
        policies = [ApprovalPolicy.from_dict(p) for p in config.approval_policies]
        self.approval_engine = ApprovalPolicyEngine(policies)
        self.knowledge_manager = KnowledgeGraphManager(storage, config.project) if getattr(config, "knowledge_graph_enabled", True) else None

    def analyze(self):
        return self.analyzer.analyze()

    def run(self, task: Task | str, subtask_id: str | None = None, progress=None, approval_callback=None) -> RunReport:
        def emit(message: str) -> None:
            LOGGER.info(message)
            if progress:
                progress(message)

        if isinstance(task, str):
            task = self._create_new_task(task)

        emit("[1/7] Analyzing project...")
        context = self.analyzer.scan()
        if self.knowledge_manager and getattr(self.config, "knowledge_graph_enabled", True):
            self.knowledge_manager.sync_with_scan(context)
        project_memory = self.storage.load_project_memory() if self.config.memory_enabled else ProjectMemory()
        if self.config.validation_commands:
            context.validation_commands = [CommandSpec(f"explicit-{index}", tuple(shlex.split(command)), "explicit CLI configuration") for index, command in enumerate(self.config.validation_commands, 1)]
        current_subtask = next((s for s in (task.plan.subtasks if task.plan else []) if s.subtask_id == subtask_id), None)
        task.current_subtask_id = subtask_id

        # Phase 4.10: Resolve upstream contracts for the current subtask
        upstream_contracts: list[SubtaskContract] = []
        if task.plan and current_subtask and hasattr(task.plan, "get_upstream_contracts"):
            upstream_contracts = task.plan.get_upstream_contracts(current_subtask.subtask_id)

        if upstream_contracts:
            context.metadata["upstream_contracts"] = [c.to_dict() for c in upstream_contracts]

        emit("[2/7] Building context...")
        ContextSelector(
            self.config.project,
            max_files=self.config.max_context_files,
            max_chars=self.config.planning_context_bytes,
            max_file_chars=self.config.max_context_file_bytes,
            max_tokens=self.config.max_context_tokens,
            dependency_depth=self.config.dependency_depth,
            project_memory=project_memory,
            knowledge_manager=self.knowledge_manager if getattr(self.config, "knowledge_graph_enabled", True) else None,
            knowledge_graph=self.knowledge_manager.get_graph() if (self.knowledge_manager and getattr(self.config, "knowledge_graph_enabled", True)) else None,
        ).select(
            task.objective,
            context,
            subtask_goal=current_subtask.goal if current_subtask else None,
            upstream_contracts=upstream_contracts,
        )

        emit("[2.5/7] Analyzing change impact...")
        impact = self.impact_analyzer.analyze(task.objective, context)
        context.metadata["change_impact"] = impact.to_dict()
        context.metadata["current_subtask_id"] = subtask_id

        report = RunReport(project=context, task_id=task.task_id, subtask_id=subtask_id)
        report.impact = impact

        discovered_commands = self.validation_intelligence.discover_commands(context.repository_map) if context.repository_map else []
        for cmd in (getattr(context, "validation_commands", None) or []):
            if cmd not in discovered_commands:
                discovered_commands.append(cmd)
        validation_plan = self.validation_intelligence.select_commands(
            task.objective, impact, discovered_commands
        )
        report.validation_plan = validation_plan
        context.metadata["validation_plan"] = validation_plan.to_dict()

        try:
            emit("[3/7] Creating implementation plan...")
            planning_registry = ToolRegistry(
                self.config.project,
                filesystem=self.filesystem,
                command_runner=self.runner,
                semantic_index=_context_semantic_index(context),
            )
            planning_policy = getattr(self.config, "tool_policy", None)
            if current_subtask:
                plan = self._execute_with_specialist(
                    task,
                    ProviderCapability.PLANNING,
                    lambda provider: Planner(
                        provider,
                        registry=planning_registry,
                        policy=planning_policy,
                    ).create_subtask_plan(current_subtask, context, report=report, upstream_contracts=upstream_contracts),
                    "planning"
                )
            else:
                plan = self._execute_with_specialist(
                    task,
                    ProviderCapability.PLANNING,
                    lambda provider: Planner(
                        provider,
                        registry=planning_registry,
                        policy=planning_policy,
                    ).create_plan_for_task(task.objective, context, report=report),
                    "planning"
                )
            if plan is None:
                raise ProviderError("No available provider for planning")
        except (RateLimitError, QuotaExceededError) as exc:
            self._handle_temporary_provider_error(task, current_subtask, exc, "planning provider request failed", context)
            emit(f"[3/7] Provider temporarily unavailable: {task.outcome}. Task paused.")
            return self._build_run_report(task)
        except ProviderError as exc:
            self._record_provider_failure(task, current_subtask, exc, "planning provider request failed")
            task.status = TaskStatus.FAILED
            if current_subtask:
                current_subtask.status = SubtaskStatus.FAILED
                current_subtask.completed_at = datetime.datetime.now(datetime.timezone.utc)
            report.outcome = task.outcome
            emit(f"[3/7] Provider stopped the run: {report.outcome}")
            return report
        except Exception as e:
            LOGGER.exception("Unhandled exception during planning: %s", e)
        report.plan = plan
        preexisting = self._git_changed_paths()
        coding_agent = CodingAgent(self.filesystem, preexisting)
        failure: FailureAnalysis | None = None
        review = None

        # If resuming from a paused state, reset status to allow completion
        if task.status == TaskStatus.PAUSED:
            task.status = TaskStatus.PENDING
            if current_subtask is not None:
                current_subtask.status = SubtaskStatus.PENDING
                current_subtask.completed_at = None

        # Phase 4.5: Autonomous Recovery State
        recovery_state = RecoveryState()
        current_diagnostic_evidence: list[ExecutionResult] = []
        executed_diagnostic_names_this_iteration: list[str] = []
        current_tool_history: list[tuple[ToolCall, ToolResult]] = []
        latest_checkpoint_id = (current_subtask.latest_checkpoint_id if current_subtask else None) or task.latest_checkpoint_id
        if latest_checkpoint_id:
            try:
                checkpoint = self.storage.load_checkpoint(latest_checkpoint_id)
                if checkpoint is not None:
                    raw_plan = checkpoint.continuation_context.get("plan")
                    if raw_plan and isinstance(raw_plan, dict):
                        plan = Plan.from_dict(raw_plan)
                        report.plan = plan
                        report.amendments = list(plan.amendments)
                    raw_recovery = checkpoint.continuation_context.get("recovery_state")
                    if raw_recovery and isinstance(raw_recovery, dict):
                        recovery_state = RecoveryState.from_dict(raw_recovery)
                    current_diagnostic_evidence = [ExecutionResult.from_dict(d) for d in checkpoint.continuation_context.get("diagnostic_evidence", [])]
                    executed_diagnostic_names_this_iteration = list( # Ensure it's a list for mutability
                        checkpoint.continuation_context.get("executed_diagnostic_names_this_iteration", [])
                    )
                    raw_tool_history = checkpoint.continuation_context.get("tool_history", [])
                    if raw_tool_history:
                        current_tool_history = history_from_dict(raw_tool_history)
                        report.tool_history = list(current_tool_history)
                    raw_tool_metrics = checkpoint.continuation_context.get("tool_metrics", [])
                    if raw_tool_metrics:
                        report.tool_metrics = [ToolExecutionMetrics.from_dict(m) if isinstance(m, dict) else m for m in raw_tool_metrics]
                    raw_impl_result = checkpoint.continuation_context.get("implementation_result")
                    if isinstance(raw_impl_result, dict):
                        report.implementation_result = ImplementationResult.from_dict(raw_impl_result)
                    raw_task_plan = checkpoint.continuation_context.get("task_plan")
                    if raw_task_plan and isinstance(raw_task_plan, dict) and task.plan:
                        restored_tp = TaskPlan.from_dict(raw_task_plan)
                        task.plan.version = restored_tp.version
                        task.plan.amendments = list(restored_tp.amendments)
                        report.dag_amendments = list(restored_tp.amendments)
                        for restored_sub in restored_tp.subtasks:
                            matching = next((s for s in task.plan.subtasks if s.subtask_id == restored_sub.subtask_id), None)
                            if matching and restored_sub.contract and not matching.contract:
                                matching.contract = restored_sub.contract
                    # Load the last failure from checkpoint to continue repair iterations
                    last_failures = checkpoint.continuation_context.get("validation_state", {}).get("last_failures", [])
                    if last_failures:
                        failure = FailureAnalysis.from_dict(last_failures[0])
                    else:
                        failure = None
            except FileNotFoundError:
                pass  # Checkpoint may be missing if the task is new or corrupted

        report.recovery_state = recovery_state

        initial_scope_count = len(plan.allowed_paths) if hasattr(plan, "allowed_paths") else len(getattr(plan, "files_likely_to_change", []) + getattr(plan, "files_likely_to_create", []))
        scope_guard = ScopeAmendmentGuard(
            self.filesystem,
            max_total_amendments=getattr(self.config, "max_plan_amendments", 5),
            max_scope_growth_factor=getattr(self.config, "max_scope_growth_factor", 2.0),
        )

        start_iteration = recovery_state.completed_iterations + 1
        try:
            for iteration in range(start_iteration, self.config.max_iterations + 1):
                report.iterations = iteration

                # Phase 3.23: Seed the first iteration with CI failure context if available.
                if iteration == 1 and task.initial_failure_context:
                    emit("[5/7] Seeding repair cycle with initial CI failure context.")
                    ci_context = task.initial_failure_context
                    initial_execution = ExecutionResult(
                        command=ci_context.failed_command,
                        exit_code=ci_context.exit_code,
                        stdout=ci_context.stdout,
                        stderr=ci_context.stderr
                    )
                    failure = FailureAnalysis(
                        probable_root_cause="Initial failure provided by CI environment.",
                        recommended_fix="Analyze the provided logs and fix the issue.",
                        diagnostic_evidence=[initial_execution]
                    )
                    task.initial_failure_context = None  # Consume the context
                    self.storage.save_task(task)

                stage_name = "implementation" if iteration == 1 and not failure else "repair"
                emit("[4/7] Implementing changes..." if iteration == 1 and not failure else f"[5/7] Applying repair for iteration {iteration}...")
                try:
                    operations, current_tool_history = self._execute_code_generation(
                        task,
                        plan,
                        context,
                        failure=failure,
                        review=review,
                        stage_name=stage_name,
                        tool_history=current_tool_history,
                        report=report,
                        recovery_state=recovery_state,
                        subtask=current_subtask,
                    )
                except ProviderError as exc:
                    self._record_provider_failure(task, current_subtask, exc, f"{stage_name} provider request failed")
                    report.outcome = task.outcome
                    report.failures.append(
                        FailureAnalysis(
                            probable_root_cause=f"{stage_name} provider request failed",
                            recommended_fix=str(exc),
                            category=getattr(exc, "category", "PROVIDER_ERROR"),
                            details={"retry_after_seconds": getattr(exc, "retry_after_seconds", None)},
                        )
                    )
                    emit(f"[4/7] Provider stopped the run: {report.outcome}")
                    if isinstance(exc, (RateLimitError, QuotaExceededError)):
                        task.status = TaskStatus.PAUSED
                        if current_subtask:
                            current_subtask.status = SubtaskStatus.PAUSED
                            current_subtask.completed_at = datetime.datetime.now(datetime.timezone.utc)
                        # Create a checkpoint to preserve failure state, diagnostic state, and tool history for resumption
                        extra_context = {
                            'diagnostic_evidence': [e.to_dict() for e in current_diagnostic_evidence],
                            'executed_diagnostic_names_this_iteration': executed_diagnostic_names_this_iteration,
                            'tool_history': history_to_dict(current_tool_history),
                            'tool_metrics': [m.to_dict() if hasattr(m, "to_dict") else m for m in report.tool_metrics],
                        }
                        self._create_checkpoint(task, current_subtask, "After code generation (Paused due to provider error)", context, report, extra_context=extra_context)
                    else:
                        task.status = TaskStatus.FAILED
                        if current_subtask:
                            current_subtask.status = SubtaskStatus.FAILED
                            current_subtask.completed_at = datetime.datetime.now(datetime.timezone.utc)
                    break

                # Phase 4.8: Dynamic Scope Amendment for discovered required files
                unlisted_ops = coding_agent.find_unlisted_operations(operations, plan)
                if unlisted_ops:
                    for unlisted_op in unlisted_ops:
                        is_create = unlisted_op.action.lower().strip() == "create" or not self.filesystem.file_exists(unlisted_op.path)
                        proposal = ScopeExpansionProposal(
                            path=unlisted_op.path,
                            reason=unlisted_op.reason or "Implementation discovered required file",
                            relationship="implementation_dependency",
                            evidence=unlisted_op.patch or (unlisted_op.content[:200] if unlisted_op.content else ""),
                            originating_stage=stage_name,
                            is_create=is_create,
                        )
                        is_valid, guard_reason = scope_guard.evaluate(proposal, plan, initial_scope_count)
                        if is_valid:
                            amendment = plan.apply_amendment(proposal, approved_by="deterministic_policy")
                            report.amendments.append(amendment)
                            emit(f"[{ '4' if stage_name == 'implementation' else '5' }/7] Plan amended to v{plan.version}: added '{proposal.path}' to approved scope ({guard_reason})")
                        else:
                            emit(f"[{ '4' if stage_name == 'implementation' else '5' }/7] Scope expansion for '{proposal.path}' rejected: {guard_reason}")

                try:
                    prepared = coding_agent.prepare(operations, plan)
                except PatchValidationError as exc:
                    failure = FailureAnalysis(
                        "AI-generated patch failed strict validation",
                        [exc.path],
                        "Regenerate a patch that matches the original target file exactly.",
                        category="PATCH_VALIDATION",
                        details={
                            "path": exc.path,
                            "original_file": exc.original,
                            "generated_patch": exc.patch,
                            "validation_error": exc.reason,
                        },
                    )
                    recovery_state.record_failure(failure)
                    report.failures.append(failure)
                    emit(f"[4/7] Rejected generated changes: {exc}")
                    recovery_state.completed_iterations = iteration
                    if iteration < self.config.max_iterations:
                        continue
                    break
                except UnsafeModificationError as exc:
                    failure = FailureAnalysis("AI-generated change was rejected by the safety/patch validator", [], str(exc), category="UNSAFE_MODIFICATION")
                    recovery_state.record_failure(failure)
                    report.failures.append(failure)
                    emit(f"[4/7] Rejected generated changes: {exc}")
                    recovery_state.completed_iterations = iteration
                    if iteration < self.config.max_iterations:
                        continue
                    break
                proposed_diff = "".join(change.diff for change in prepared)

                # Phase 4.5: Anti-Repeat Detection for identical ineffective patches
                patch_hash = normalize_diff_for_signature(proposed_diff)
                if iteration > 1 and stage_name == "repair" and recovery_state.is_duplicate_patch(patch_hash):
                    emit(f"[5/7] Repeated identical repair patch detected (hash {patch_hash}); rejecting to prevent loop")
                    failure = FailureAnalysis(
                        "Repeated identical repair patch generated without progress.",
                        [c.path for c in prepared],
                        "Do not regenerate the same patch. Explore an alternative repair strategy.",
                        category="REPEATED_REPAIR",
                        details={"patch_hash": patch_hash},
                    )
                    recovery_state.record_failure(failure)
                    report.failures.append(failure)
                    recovery_state.completed_iterations = iteration
                    recovery_state.abort_reason = "REPEATED_REPAIR_DETECTED"
                    task.outcome = "REPEATED_REPAIR_DETECTED"
                    report.outcome = "REPEATED_REPAIR_DETECTED"
                    break

                affected_paths = [c.path for c in prepared]
                recovery_state.record_attempt(iteration, failure, proposed_diff, affected_paths)

                if self.config.dry_run:
                    report.dry_run = True
                    report.proposed_diff = proposed_diff
                    emit(f"[4/7] Dry run: {len(prepared)} proposed file changes; no files written")
                    break

                requires_manual_approval = False
                if prepared:
                    if self.config.approval == "always" and not task.autonomous:
                        requires_manual_approval = True
                    elif self.config.approval == "policy":
                        emit("[4.5/7] Evaluating approval policies...")
                        requires_manual_approval = self.approval_engine.is_manual_approval_required(prepared, report.impact)

                if requires_manual_approval:
                    report.approval_required = True
                    approved = bool(approval_callback(prepared)) if approval_callback else False
                    if not approved:
                        emit("[4/7] Changes not approved; stopping before write")
                        report.proposed_diff = proposed_diff
                        break
                with self.repo_lock: # Acquire lock for file system mutation
                    report.changed_files.extend(coding_agent.apply_prepared(prepared))
                report.proposed_diff = coding_agent.diff()
                self._create_checkpoint(task, current_subtask, "After code generation, before validation", context, report) # Checkpoint

                emit("[5/7] Running validation...")
                executions = []
                failed = None

                # Phase 4.4: Targeted Validation Intelligence
                targeted_commands = self.validation_intelligence.discover_targeted_commands(report.changed_files, context.repository_map)
                # Phase 4.17: replace/augment the filename heuristics with
                # graph-derived, explained targets, then let passing candidate
                # evidence stand in for a rerun where that is provably safe.
                # The mandatory full-suite run below is untouched by both.
                targeted_commands = self._semantic_targeted_commands(
                    report, context, targeted_commands, emit
                )
                targeted_commands = self._apply_evidence_reuse(
                    report, targeted_commands, emit
                )
                if targeted_commands:
                    emit(f"[5/7] Running targeted validation ({len(targeted_commands)} command(s))...")
                    targeted_plan = ValidationPlan(
                        commands=targeted_commands,
                        primary_commands=targeted_commands,
                        secondary_commands=[],
                        skipped_commands=[],
                        reasons=["Targeted verification for modified files"],
                        risk_level="low",
                    )
                    targeted_executions = self._validate(targeted_plan)
                    executions.extend(targeted_executions)
                    report.executions.extend(targeted_executions)
                    sub_id = current_subtask.subtask_id if current_subtask else None
                    task.execution_history.extend([{"type": "execution", "subtask_id": sub_id, "execution": exec.to_dict()} for exec in targeted_executions])
                    self.storage.save_task(task)
                    failed = next((result for result in targeted_executions if not result.succeeded), None)

                # Final full-suite validation (mandatory authority if targeted passed or none found)
                if failed is None:
                    full_executions = self._validate(validation_plan)
                    executions.extend(full_executions)
                    report.executions.extend(full_executions)
                    sub_id = current_subtask.subtask_id if current_subtask else None
                    task.execution_history.extend([{"type": "execution", "subtask_id": sub_id, "execution": exec.to_dict()} for exec in full_executions])
                    self.storage.save_task(task)
                    failed = next((result for result in full_executions if not result.succeeded), None)

                # Phase 4.11: Autonomous Verification Gap Analysis & Behavioral Test Synthesis
                if failed is None and report.changed_files:
                    try:
                        extractor = ContractExtractor(filesystem=self.filesystem, project_root=self.config.project)
                        prelim_sub = current_subtask or Subtask(subtask_id=task.task_id)
                        prelim_contract = extractor.extract_contract(
                            prelim_sub,
                            report,
                            preexisting_files=set(preexisting) if preexisting else None,
                        )
                        gap_analyzer = VerificationGapAnalyzer(self.config.project, filesystem=self.filesystem)
                        gap = gap_analyzer.analyze(
                            report.changed_files,
                            prelim_contract.exported_symbols,
                            context,
                            targeted_commands=targeted_commands,
                        )
                        report.verification_gap = gap
                        if gap and (gap.missing_test_symbols or gap.untested_files):
                            emit(f"[5.5/7] Verification gap detected ({len(gap.missing_test_symbols)} untested symbol(s)); synthesizing behavioral verification fixture...")
                            verification_prov = self.router.get_provider(SpecialistRole.VERIFICATION)
                            synthesizer = TestSynthesizer(
                                self.config.project,
                                provider=verification_prov,
                                filesystem=self.filesystem,
                            )
                            test_code = synthesizer.synthesize_test(prelim_sub, gap, context)
                            if test_code:
                                verifier = BehavioralVerifier(
                                    self.config.project,
                                    runner=self.runner,
                                    filesystem=self.filesystem,
                                )
                                record = verifier.verify(
                                    test_code,
                                    prelim_sub.subtask_id,
                                    exercised_symbols=gap.missing_test_symbols,
                                )
                                report.behavioral_evidence.append(record)
                                if record.status == "passed":
                                    emit(f"[5.6/7] Behavioral verification fixture PASSED for {len(record.exercised_symbols)} symbol(s).")
                                else:
                                    emit(f"[5.6/7] Behavioral verification fixture FAILED (exit {record.exit_code}): {record.stderr_summary[:100]}")
                                    failed = ExecutionResult(
                                        command=record.command,
                                        exit_code=record.exit_code,
                                        stdout=record.stdout_summary,
                                        stderr=record.stderr_summary,
                                        duration_seconds=record.duration_seconds,
                                    )
                                    executions.append(failed)
                                    report.executions.append(failed)
                    except Exception as e:
                        LOGGER.warning("Behavioral verification synthesis encountered an error: %s", e)

                if failed is not None:
                    emit("[5/7] Validation failed")

                    # Phase 3.13: Dynamic Secondary Validation Execution
                    if self.config.max_secondary_validations_per_iteration > 0 and validation_plan.secondary_commands:
                        emit("[5.1/7] Considering secondary validation...")

                        # Diagnostic sub-loop
                        for _ in range(self.config.max_secondary_validations_per_iteration - len(executed_diagnostic_names_this_iteration)):
                            available_diagnostics = [cmd for cmd in validation_plan.secondary_commands if cmd.name not in executed_diagnostic_names_this_iteration]
                            if not available_diagnostics:
                                break

                            target_goal = current_subtask.goal if current_subtask else task.objective
                            selected_cmd_spec = self._execute_with_specialist(
                                task,
                                ProviderCapability.REPAIR,
                                lambda provider: provider.select_diagnostic_command(target_goal, plan, context, failed, available_diagnostics),
                                "diagnostic command selection",
                            )
                            if selected_cmd_spec is None or selected_cmd_spec not in available_diagnostics:
                                emit("[5.2/7] Provider did not select a valid diagnostic command. Proceeding to repair.")
                                break
                            emit(f"[5.3/7] Executing selected diagnostic command: {selected_cmd_spec.display()}")
                            diagnostic_result = self.runner.run(selected_cmd_spec)
                            truncated_result = self._truncate_execution_result(diagnostic_result)

                            current_diagnostic_evidence.append(truncated_result)
                            executed_diagnostic_names_this_iteration.append(selected_cmd_spec.name)

                            # Create a checkpoint after each diagnostic run to persist state
                            extra_context = {
                                'diagnostic_evidence': [e.to_dict() for e in current_diagnostic_evidence],
                                'executed_diagnostic_names_this_iteration': executed_diagnostic_names_this_iteration
                            }
                            self._create_checkpoint(task, current_subtask, f"After running diagnostic: {selected_cmd_spec.name}", context, report, extra_context=extra_context)

                    try:
                        failure_registry = ToolRegistry(
                            self.config.project,
                            filesystem=self.filesystem,
                            command_runner=self.runner,
                            semantic_index=_context_semantic_index(context),
                        )
                        failure_policy = getattr(self.config, "tool_policy", None)
                        failure = self._execute_with_specialist(
                            task,
                            ProviderCapability.REPAIR,
                            lambda provider: FailureAnalyzer(
                                provider,
                                registry=failure_registry,
                                policy=failure_policy,
                            ).analyze(
                                failed,
                                coding_agent.diff() or self.git.diff(),
                                context,
                                plan,
                                report=report,
                            ),
                            "failure analysis",
                        )
                    except ProviderError as exc:
                        self._record_provider_failure(task, current_subtask, exc, "failure-analysis provider request failed")
                        emit(f"[5/7] Provider stopped the run: {report.outcome}")
                        break

                    # Attach diagnostic evidence to the failure analysis
                    failure.diagnostic_evidence = current_diagnostic_evidence

                    # Phase 4.8: Dynamic Scope Amendment for diagnosed affected files
                    unlisted_diagnosed = [f for f in failure.affected_files if f and f not in (plan.allowed_paths if hasattr(plan, "allowed_paths") else set())]
                    if unlisted_diagnosed:
                        for unlisted_file in unlisted_diagnosed:
                            proposal = ScopeExpansionProposal(
                                path=unlisted_file,
                                reason=failure.recommended_fix or failure.probable_root_cause or "Diagnosed dependency",
                                relationship="diagnosed_dependency",
                                evidence=failure.probable_root_cause,
                                originating_stage="repair",
                                is_create=not self.filesystem.file_exists(unlisted_file),
                            )
                            is_valid, guard_reason = scope_guard.evaluate(proposal, plan, initial_scope_count)
                            if is_valid:
                                amendment = plan.apply_amendment(proposal, approved_by="deterministic_policy")
                                report.amendments.append(amendment)
                                emit(f"[5.4/7] Plan amended to v{plan.version}: added '{unlisted_file}' to approved scope ({guard_reason})")
                            else:
                                emit(f"[5.4/7] Diagnosed scope expansion for '{unlisted_file}' rejected: {guard_reason}")

                    # Phase 3.14: Propose a plan modification if the failure seems architectural
                    if failure.category in {"MISSING_DEPENDENCY", "ARCHITECTURAL_FLAW"}: # Example categories
                        emit("[5.5/7] Failure suggests a planning issue. Proposing plan modification...")
                        plan_proposal: PlanProposal | None = self._execute_with_specialist(
                            task,
                            ProviderCapability.PLANNING,
                            lambda provider: provider.propose_plan_modification(task.objective, task.plan, failure),
                            "plan modification proposal"
                        )
                        if plan_proposal:
                            report.plan_proposal = plan_proposal
                            # The orchestrator's job is done; it returns the proposal in the report.
                            # The Scheduler will handle the task state transition.
                            break # Exit the repair loop

                    failure.category = failure.category or "VALIDATION_FAILURE"
                    recovery_state.record_failure(failure)
                    report.failures.append(failure)
                    recovery_state.completed_iterations = iteration

                    # Phase 4.5: Check for stagnation across consecutive failures
                    if recovery_state.consecutive_same_failure_count >= 3:
                        emit("[5/7] Stagnation detected: same failure encountered across 3 consecutive iterations.")
                        recovery_state.abort_reason = "STAGNATION_DETECTED"
                        task.outcome = "STAGNATION_DETECTED"
                        report.outcome = "STAGNATION_DETECTED"
                        break

                    emit("[5/7] Analyzing failure...")
                    if iteration < self.config.max_iterations:
                        continue
                    break
                self._create_checkpoint(task, current_subtask, "After successful validation, before review", context, report) # Checkpoint

                emit("[6/7] Reviewing changes...")
                try:
                    review_registry = ToolRegistry(
                        self.config.project,
                        filesystem=self.filesystem,
                        command_runner=self.runner,
                        semantic_index=_context_semantic_index(context),
                    )
                    review_policy = getattr(self.config, "tool_policy", None)
                    report.review = self._execute_with_specialist(
                        task,
                        SpecialistRole.REVIEW,
                        lambda provider: self._review_with_consensus(
                            provider=provider,
                            task=task,
                            subtask=current_subtask,
                            plan=plan,
                            diff=coding_agent.diff() or self.git.diff(),
                            context=context,
                            review_registry=review_registry,
                            review_policy=review_policy,
                            initial_history=None,
                            report=report,
                        ),
                        "review"
                    )
                except (RateLimitError, QuotaExceededError) as exc: # Handle temporary provider errors
                    self._handle_temporary_provider_error(task, current_subtask, exc, "review provider request failed", context)
                    emit(f"[6/7] Provider temporarily unavailable: {task.outcome}. Task paused.")
                    return self._build_run_report(task)
                except ProviderError as exc:
                    self._record_provider_failure(task, current_subtask, exc, "review provider request failed")
                    emit(f"[6/7] Provider stopped the run: {task.outcome}")
                    break
                review = report.review
                recovery_state.completed_iterations = iteration
                if review.verdict == "APPROVED":
                    report.completed = True
                    break
                else:
                    recovery_state.record_review(review)
                    if len(recovery_state.review_history) >= 3 and all(r.verdict != "APPROVED" for r in recovery_state.review_history[-3:]):
                        emit("[6/7] Repeated review rejections detected across 3 iterations; aborting.")
                        recovery_state.abort_reason = "REPEATED_REVIEW_REJECTION"
                        task.outcome = "REPEATED_REVIEW_REJECTION"
                        report.outcome = "REPEATED_REVIEW_REJECTION"
                        break
                if iteration >= self.config.max_iterations:
                    break
                emit("[6/7] Review requested changes; continuing...")
                emit("[7/7] Complete")

            # Status setting logic moved outside the loop to ensure it always runs
            if not report.plan_proposal and not getattr(report, "dag_proposal", None):
                if report.completed:
                    if current_subtask:
                        current_subtask.status = SubtaskStatus.COMPLETED
                        current_subtask.completed_at = datetime.datetime.now(datetime.timezone.utc)
                        # Phase 4.10 / Phase 4.11: Extract authoritative SubtaskContract with behavioral evidence
                        try:
                            extractor = ContractExtractor(filesystem=self.filesystem, project_root=self.config.project)
                            current_subtask.contract = extractor.extract_contract(
                                current_subtask,
                                report,
                                preexisting_files=set(preexisting) if preexisting else None,
                                behavioral_records=report.behavioral_evidence,
                            )
                            if current_subtask.contract and self.knowledge_manager and getattr(self.config, "knowledge_graph_enabled", True):
                                with self.memory_lock:
                                    try:
                                        self.knowledge_manager.promote_subtask_contract(current_subtask.contract, task_id=task.task_id)
                                        self.knowledge_manager.save()
                                    except Exception as err:
                                        LOGGER.warning("Could not promote subtask contract to knowledge graph: %s", err)
                        except Exception as e:
                            LOGGER.warning("Could not extract subtask contract for %s: %s", current_subtask.subtask_id, e)
                    # Phase 3.24: Persist changed files on successful completion
                    task.changed_files = sorted(list(set(task.changed_files + report.changed_files)))
                    active_subs = getattr(task.plan, "active_subtasks", task.plan.subtasks if task.plan else [])
                    if task.plan and all(s.status == SubtaskStatus.COMPLETED for s in active_subs):
                        task.status = TaskStatus.COMPLETED
                    elif not task.plan:
                        task.status = TaskStatus.COMPLETED
                    else:
                        task.status = TaskStatus.PENDING
                else:
                    if task.status != TaskStatus.PAUSED: # Only set FAILED if not already paused
                        task.status = TaskStatus.FAILED
                        if current_subtask:
                            current_subtask.status = SubtaskStatus.FAILED
                            current_subtask.completed_at = datetime.datetime.now(datetime.timezone.utc)
                        if recovery_state.abort_reason:
                            task.outcome = recovery_state.abort_reason
                            report.outcome = recovery_state.abort_reason
                        elif recovery_state.completed_iterations >= self.config.max_iterations:
                            task.outcome = "MAX_ITERATIONS_EXCEEDED"
                            report.outcome = "MAX_ITERATIONS_EXCEEDED"

        except Exception as e:
            LOGGER.exception("Unhandled exception during task execution: %s", e)
            task.status = TaskStatus.FAILED
            if current_subtask:
                current_subtask.status = SubtaskStatus.FAILED
                current_subtask.completed_at = datetime.datetime.now(datetime.timezone.utc)
            report.outcome = "UNHANDLED_EXCEPTION"
            report.failures.append(FailureAnalysis(probable_root_cause="Unhandled exception", recommended_fix=str(e)))
        finally:
            # Consolidate metrics from all specialists used in this run
            if report.completed:
                # Phase 3.24: Autonomous commit on completion
                branch_name = f"agent-task/{task.task_id[:8]}"
                committed = False
                if task.autonomous and self.config.git_commit_on_completion:
                    emit("[7/7] Autonomous commit enabled, attempting to commit changes...")
                    committed = self._perform_git_commit(task, branch_name)

                # Phase 3.25: Autonomous push and PR creation
                if committed and task.autonomous and self.config.git_push_on_completion:
                    emit("[7/7] Autonomous push enabled, attempting to push changes...")
                    pushed = self._perform_git_push(task, branch_name)
                    if pushed and self.config.git_pr_on_completion:
                        emit("[7/7] Autonomous PR creation enabled, attempting to create PR...")
                        self._perform_pr_creation(task, branch_name)

                self._create_memories_from_run(task, report)
            all_metrics = []
            if self.scheduler is not None and getattr(self.scheduler, "registry", None):
                for reg_provider in self.scheduler.registry.providers.values():
                    try:
                        provider_instance = self.scheduler._build_provider_instance(reg_provider.provider_id)
                        if provider_instance:
                            all_metrics.extend(provider_instance.provider_metrics)
                    except Exception:
                        pass # Ignore if a provider can't be built

            task.updated_at = datetime.datetime.now(datetime.timezone.utc)
            self.storage.save_task(task)
            report.provider_metrics = all_metrics
            task.provider_execution_history.extend(report.provider_metrics) # Add to task history
            self.storage.save_task(task) # Save task with updated metrics

            report.task_id = task.task_id
            report.subtask_id = current_subtask.subtask_id if current_subtask else None
            if not report.outcome and task.outcome:
                report.outcome = task.outcome

        return report

    def _map_role(self, capability: ProviderCapability | SpecialistRole | str) -> SpecialistRole:
        if isinstance(capability, SpecialistRole):
            return capability
        role_map = {
            ProviderCapability.PLANNING: SpecialistRole.PLANNING,
            ProviderCapability.IMPLEMENTATION: SpecialistRole.IMPLEMENTATION,
            ProviderCapability.REPAIR: SpecialistRole.REPAIR,
            ProviderCapability.REVIEW: SpecialistRole.REVIEW,
            "planning": SpecialistRole.PLANNING,
            "implementation": SpecialistRole.IMPLEMENTATION,
            "repair": SpecialistRole.REPAIR,
            "review": SpecialistRole.REVIEW,
            "verification": SpecialistRole.VERIFICATION,
        }
        return role_map.get(capability, SpecialistRole.IMPLEMENTATION)

    def _select_specialists(self, task: Task, capability: ProviderCapability | SpecialistRole | str):
        role = self._map_role(capability)
        cap = capability if isinstance(capability, ProviderCapability) else {
            SpecialistRole.PLANNING: ProviderCapability.PLANNING,
            SpecialistRole.IMPLEMENTATION: ProviderCapability.IMPLEMENTATION,
            SpecialistRole.REPAIR: ProviderCapability.REPAIR,
            SpecialistRole.REVIEW: ProviderCapability.REVIEW,
        }.get(role, ProviderCapability.IMPLEMENTATION)

        if self.scheduler is not None:
            provider_configs = self.scheduler._select_providers(task, {cap})
            for provider_config in provider_configs:
                provider = self.scheduler._build_provider_instance(provider_config.provider)
                if provider:
                    yield provider
            return

        for provider in self.router.get_provider_chain(role):
            yield provider

    def _execute_with_specialist(self, task: Task, capability: ProviderCapability | SpecialistRole | str, action, stage_name: str):
        role = self._map_role(capability)
        if self.scheduler is not None:
            for provider in self._select_specialists(task, capability):
                return action(provider)
            raise ProviderError(f"No available and capable provider found for stage: {stage_name}")

        return self.router.execute_with_fallback(role, action, stage_name)

    def _get_secondary_review_provider(self, primary_provider: AIProvider, task: Task | None = None) -> AIProvider | None:
        if self.scheduler is not None:
            if task:
                for p in self._select_specialists(task, ProviderCapability.REVIEW):
                    if p != primary_provider:
                        return p
            return None

        # Check explicit fallbacks for review role
        _, _, fallbacks = self.router.get_role_config(SpecialistRole.REVIEW)
        for fb_id in fallbacks:
            fb_p = self.router._build_provider_for_spec(fb_id)
            if fb_p and fb_p != primary_provider:
                return fb_p

        # Check if planning or implementation has an explicitly different provider configured
        for role in (SpecialistRole.PLANNING, SpecialistRole.IMPLEMENTATION):
            role_override = getattr(self.config, f"{role.value}_provider", None)
            if role_override:
                p = self.router.get_provider(role)
                if p and p != primary_provider:
                    return p

        return None

    def _review_with_consensus(
        self,
        provider: AIProvider,
        task: Task,
        subtask: Subtask | None,
        plan: Plan,
        diff: str,
        context: ProjectContext,
        review_registry: ToolRegistry | None,
        review_policy: Any | None,
        initial_history: list[tuple[ToolCall, ToolResult]] | None,
        report: RunReport,
    ) -> ReviewResult:
        primary_reviewer = Reviewer(provider, registry=review_registry, policy=review_policy)

        # Determine secondary review provider if explicitly available
        secondary_provider = self._get_secondary_review_provider(provider, task=task)
        secondary_reviewer = Reviewer(secondary_provider, registry=review_registry, policy=review_policy) if secondary_provider else None

        consensus_engine = DeliberativeReviewConsensus(
            primary_reviewer=primary_reviewer,
            secondary_reviewer=secondary_reviewer,
            dual_review_enabled=self.config.dual_review_enabled,
            high_risk_dual_review=self.config.high_risk_dual_review,
        )

        return consensus_engine.review(
            task=subtask.goal if subtask else task.objective,
            plan=plan,
            diff=diff,
            context=context,
            changed_files=report.changed_files,
            initial_history=initial_history,
            report=report,
        )

    # -- Phase 4.17/4.18: semantic targeting and evidence reuse -------------

    def _decision_policy_fingerprint(self) -> str:
        """Digest of the configuration that could change a scope/reuse decision.

        Deliberately narrow: only settings that feed
        :class:`~local_agent.validation_decision.ValidationDecisionEngine`
        belong here. Changing any of them between when evidence was recorded
        and a later reuse attempt means that attempt was evaluated under
        different rules and must not silently reuse a conclusion reached under
        the old ones.
        """
        return compute_policy_fingerprint({
            "max_impact_depth": getattr(self.config, "max_impact_depth", 3),
            "max_affected_symbols": getattr(self.config, "max_affected_symbols", 200),
            "max_affected_tests": getattr(self.config, "max_affected_tests", 8),
            "validation_confidence_threshold": getattr(
                self.config, "validation_confidence_threshold", "high"
            ),
            "knowledge_graph_enabled": getattr(self.config, "knowledge_graph_enabled", False),
        })

    def _decision_engine(self, *, reuse_enabled: bool) -> ValidationDecisionEngine:
        max_age = getattr(self.config, "evidence_max_age_seconds", 0) or 0
        return ValidationDecisionEngine(
            min_confidence=getattr(self.config, "validation_confidence_threshold", "high"),
            reuse_enabled=reuse_enabled,
            max_age_seconds=float(max_age) if max_age > 0 else None,
            policy_fingerprint=self._decision_policy_fingerprint() if reuse_enabled else None,
            analyzer_version=SEMANTIC_ANALYZER_SCHEMA_VERSION if reuse_enabled else None,
        )

    def _semantic_targeted_commands(
        self,
        report: RunReport,
        context: ProjectContext,
        lexical_commands: list[CommandSpec],
        emit: Any,
    ) -> list[CommandSpec]:
        """Select post-apply targeted commands from the impact graph.

        Preference order:

        1. The :class:`ChangeImpactReport` the candidate loop already produced.
           It is the most accurate available, because the candidate workspace
           supplied exact BASE contents for symbol-level diffing.
        2. A fresh analysis against the authoritative tree. Without BASE
           contents it cannot classify added-vs-modified symbols, so it records
           a degradation, reports LOW confidence and recommends BROAD - it can
           therefore only *add* targets, never justify narrowing.
        3. The pre-existing lexical commands, always retained unless the
           analysis reached at least the configured confidence threshold.

        The actual narrow-vs-union decision is
        :class:`~local_agent.validation_decision.ValidationDecisionEngine`'s -
        this method's job is only to obtain the impact report and hand it over,
        so that decision is made in exactly one place for both this post-apply
        path and the Phase 4.16 candidate-time path.

        Returns the commands to run. On any failure it returns
        ``lexical_commands`` unchanged, so a defect here degrades to Phase 4.4
        behaviour rather than dropping validation.
        """
        if not getattr(self.config, "semantic_impact_analysis_enabled", False):
            return lexical_commands
        if not report.changed_files:
            return lexical_commands

        impact: ChangeImpactReport | None = None
        implementation = getattr(report, "implementation_result", None)
        raw = getattr(implementation, "impact_report", None) if implementation else None
        if isinstance(raw, dict) and raw.get("changed_files"):
            impact = ChangeImpactReport.from_dict(raw)
        if impact is None:
            try:
                analyzer = SemanticChangeImpactAnalyzer(
                    self.config.project,
                    semantic_index=_context_semantic_index(context),
                    max_impact_depth=getattr(self.config, "max_impact_depth", 3),
                    max_affected_symbols=getattr(self.config, "max_affected_symbols", 200),
                    max_affected_tests=getattr(self.config, "max_affected_tests", 8),
                )
                impact = analyzer.analyze(
                    report.changed_files,
                    validation_intelligence=self.validation_intelligence,
                )
            except (OSError, ValueError, RecursionError) as exc:
                LOGGER.warning("Post-apply semantic impact analysis failed: %s", exc)
                return lexical_commands

        # Knowledge is supporting evidence only: it can add notes and widen the
        # scope, never raise confidence or narrow validation.
        if getattr(self.config, "knowledge_graph_enabled", False):
            impact = apply_knowledge_support(
                impact, getattr(self, "knowledge_manager", None), root=self.config.project
            )

        report.semantic_impact = impact.to_dict()

        decision = self._decision_engine(reuse_enabled=False).decide(
            impact, current_root=self.config.project, lexical_commands=lexical_commands,
        )
        if decision.selected_commands:
            emit(
                f"[5/7] Semantic impact: {decision.confidence_level.upper()} confidence, "
                f"scope {decision.scope}; {len(decision.selected_commands)} targeted "
                "command(s) selected"
            )
        return decision.selected_commands

    def _apply_evidence_reuse(
        self,
        report: RunReport,
        targeted_commands: list[CommandSpec],
        emit: Any,
    ) -> list[CommandSpec]:
        """Drop targeted commands whose candidate evidence provably still holds.

        A command is dropped only when
        :meth:`~local_agent.validation_decision.ValidationDecisionEngine.apply_reuse`
        (via :meth:`~local_agent.evidence.EvidenceLedger.find_reusable`)
        confirms *every* assumption: identical command vector, a recorded pass,
        identical relevant file and symbol sets, an impact confidence meeting
        the configured threshold, a policy fingerprint matching the
        configuration currently in effect, an executable fingerprint matching
        what the command would actually run under right now, a matching
        analyzer schema version, and a byte-identical content fingerprint over
        the relevant files recomputed against the authoritative tree right now.

        This can never skip validation overall: the mandatory full-suite
        ``validation_plan`` run happens immediately afterwards regardless. What
        it removes is only the duplicated *targeted* execution of a command that
        already ran, against provably-identical inputs, minutes earlier.
        """
        if not getattr(self.config, "reuse_candidate_validation_evidence", False):
            return targeted_commands
        implementation = getattr(report, "implementation_result", None)
        raw_evidence = list(getattr(implementation, "validation_evidence", None) or [])
        if not raw_evidence or not targeted_commands:
            return targeted_commands

        ledger = EvidenceLedger.from_dict({"entries": raw_evidence})
        impact_dict = getattr(report, "semantic_impact", None) or {}
        impact = ChangeImpactReport.from_dict(impact_dict) if impact_dict else ChangeImpactReport()
        if not impact.changed_files:
            impact.changed_files = list(report.changed_files)

        remaining, attempts, saved = self._decision_engine(reuse_enabled=True).apply_reuse(
            targeted_commands,
            impact,
            self.config.project,
            ledger,
            executable_fingerprint_of=compute_executable_fingerprint,
        )
        reused = sum(1 for attempt in attempts if attempt.reusable)
        for attempt in attempts:
            if attempt.reusable and attempt.evidence is not None:
                report.validation_evidence.append(attempt.evidence)
                LOGGER.info(
                    "Reusing candidate validation evidence for '%s' (%s)",
                    " ".join(attempt.command), attempt.reason,
                )
            elif not attempt.reusable:
                LOGGER.debug(
                    "Candidate evidence for '%s' not reusable: %s",
                    " ".join(attempt.command), attempt.reason,
                )

        if reused:
            emit(
                f"[5/7] Reused {reused} passing candidate validation result(s) "
                f"({saved:.2f}s of command runtime avoided); the full validation "
                "suite still runs"
            )
        if implementation is not None:
            implementation.validation_evidence_reused = ledger.reuse_grants
            implementation.validation_evidence_invalidated = ledger.reuse_denials
            implementation.validation_time_saved_seconds = ledger.time_saved_seconds
        return remaining

    def _execute_code_generation(
        self,
        task: Task,
        plan: Plan,
        context: ProjectContext,
        failure: FailureAnalysis | None,
        review: ReviewResult | None,
        stage_name: str,
        tool_history: list[tuple[ToolCall, ToolResult]] | None = None,
        report: RunReport | None = None,
        recovery_state: RecoveryState | None = None,
        subtask: Subtask | None = None,
    ) -> tuple[list[FileOperation], list[tuple[ToolCall, ToolResult]]]:
        if failure and recovery_state:
            recovery_summary = IterationHistoryCompactor.build_cross_iteration_context(recovery_state, plan, report=report)
            if recovery_summary:
                if failure.details is None:
                    failure.details = {}
                failure.details["recovery_summary"] = recovery_summary

        capability = ProviderCapability.REPAIR if failure else ProviderCapability.IMPLEMENTATION

        def _action(provider: AIProvider) -> tuple[list[FileOperation], list[tuple[ToolCall, ToolResult]]]:
            # Phase 4.15: interactive, tool-assisted implementation loop.
            # Runs against self.config.project / self.filesystem, which the
            # ParallelExecutionCoordinator rebinds to the isolated worktree path
            # for worktree-backed workers, so all mutation stays in-worktree.
            if getattr(self.config, "interactive_implementation", False):
                registry = ToolRegistry(
                    self.config.project,
                    filesystem=self.filesystem,
                    command_runner=self.runner,
                    semantic_index=_context_semantic_index(context),
                )
                # Phase 4.16: when prospective validation is enabled the agent
                # gets a disposable candidate tree. It is created here (one per
                # implementation attempt, explicitly owned - no global sandbox)
                # and torn down by the agent itself; the authoritative tree is
                # never written by the candidate loop.
                sandbox = None
                if getattr(self.config, "prospective_validation_enabled", False):
                    try:
                        # Same protected-path set the authoritative CodingAgent
                        # uses, so a candidate is rejected exactly as a real
                        # apply would be.
                        candidate_protected = set(self._git_changed_paths() or [])
                        sandbox = CandidateWorkspace(
                            self.config.project,
                            protected_paths=candidate_protected or None,
                            command_timeout_seconds=getattr(
                                self.config, "candidate_validation_timeout_seconds", 120
                            ),
                            semantic_index=_context_semantic_index(context),
                        )
                    except CandidateWorkspaceError as exc:
                        LOGGER.warning(
                            "Prospective validation unavailable, falling back to "
                            "pre-mutation checks only: %s", exc,
                        )
                        sandbox = None

                # Phase 4.17: the validator carries the semantic-impact settings.
                # When semantic_impact_analysis_enabled is False this is exactly
                # the Phase 4.16 default validator, so modes A/B/C are unchanged.
                validator = ProspectiveValidator(
                    semantic_impact_enabled=getattr(
                        self.config, "semantic_impact_analysis_enabled", False
                    ),
                    max_impact_depth=getattr(self.config, "max_impact_depth", 3),
                    max_affected_symbols=getattr(self.config, "max_affected_symbols", 200),
                    max_affected_tests=getattr(self.config, "max_affected_tests", 8),
                )
                agent = InteractiveCodingAgent(
                    filesystem=self.filesystem,
                    registry=registry,
                    policy=getattr(self.config, "tool_policy", None),
                    max_tool_steps=getattr(
                        self.config,
                        "max_implementation_tool_steps",
                        DEFAULT_MAX_IMPLEMENTATION_TOOL_STEPS,
                    ),
                    sandbox=sandbox,
                    validator=validator,
                    max_candidate_iterations=getattr(
                        self.config, "max_candidate_iterations", 2
                    ),
                )

                raw_contracts = context.metadata.get("upstream_contracts")
                upstream_contracts = [
                    SubtaskContract.from_dict(c) if isinstance(c, dict) else c
                    for c in raw_contracts
                ] if raw_contracts else None

                active_subtask = subtask or next(
                    (
                        s for s in (task.plan.subtasks if task.plan else [])
                        if s.subtask_id == getattr(task, "current_subtask_id", None)
                    ),
                    None,
                )

                impl_result = agent.execute(
                    provider=provider,
                    task_objective=task.objective if hasattr(task, "objective") else str(task),
                    plan=plan,
                    context=context,
                    subtask=active_subtask,
                    upstream_contracts=upstream_contracts,
                    failure=failure,
                    review=review,
                    initial_history=tool_history,
                    report=report,
                )

                restored_history = (
                    history_from_dict(impl_result.tool_history)
                    if impl_result.tool_history
                    else list(tool_history or [])
                )

                if report is not None:
                    report.implementation_result = impl_result
                    if impl_result.metrics is not None:
                        report.tool_metrics.append(impl_result.metrics)
                    if impl_result.tool_history:
                        report.tool_history = restored_history

                if impl_result.scope_violations:
                    LOGGER.info(
                        "Interactive implementation proposed %d out-of-plan path(s): %s",
                        len(impl_result.scope_violations),
                        ", ".join(impl_result.scope_violations),
                    )

                if not impl_result.success or impl_result.file_operations is None:
                    # Surfaced as a ProviderError so existing specialist fallback,
                    # repair and pause machinery handles it. The structured result
                    # stays on the report for checkpointing and diagnostics.
                    raise ProviderError(
                        f"Interactive implementation failed "
                        f"[{impl_result.failure_category}/{impl_result.termination_reason}]: "
                        f"{impl_result.error_message or 'No file operations produced'}"
                    )

                return impl_result.file_operations, restored_history

            # Legacy single-shot path (when interactive_implementation is disabled)
            try:
                provider_caps = getattr(provider, "capabilities", None)
            except Exception:
                provider_caps = None
            task_obj = task.objective if hasattr(task, "objective") else str(task)
            raw_contracts = context.metadata.get("upstream_contracts")
            if raw_contracts and isinstance(raw_contracts, list):
                constraints_lines = [
                    "\n\nUPSTREAM INTERFACE CONSTRAINTS\n============================\n"
                    "These constraints describe verified outputs from completed direct dependencies.\n"
                    "Treat them as authoritative interface constraints. Do NOT modify files outside the approved plan.\n"
                ]
                for raw_c in raw_contracts:
                    contract = SubtaskContract.from_dict(raw_c) if isinstance(raw_c, dict) else raw_c
                    if hasattr(contract, "format_for_prompt"):
                        constraints_lines.append(contract.format_for_prompt(max_chars=1200))
                task_obj = f"{task_obj}\n" + "\n\n".join(constraints_lines)

            if isinstance(provider_caps, (set, frozenset)) and ProviderCapability.TOOL_USE in provider_caps:
                registry = ToolRegistry(
                    self.config.project,
                    filesystem=self.filesystem,
                    command_runner=self.runner,
                    semantic_index=_context_semantic_index(context),
                )
                policy = getattr(self.config, "tool_policy", None)
                engine = ToolEngine(provider=provider, registry=registry, policy=policy)
                result = engine.run(
                    task=task_obj,
                    plan=plan,
                    context=context,
                    initial_history=tool_history,
                    failure=failure,
                    review=review,
                )
                if report is not None:
                    report.tool_metrics.append(result.metrics)
                    report.tool_history = list(result.tool_history)

                if not result.completed or result.file_operations is None:
                    raise ProviderError(
                        f"ToolEngine failed to generate code ({result.termination_reason}): {result.error_message or 'No file operations produced'}"
                    )
                return result.file_operations, result.tool_history

            ops = provider.generate_code(
                task_obj,
                plan,
                context,
                failure=failure,
                review=review,
            )
            return ops, (tool_history or [])

        return self._execute_with_specialist(task, capability, _action, stage_name)

    def _perform_git_commit(self, task: Task, branch_name: str) -> bool:
        """Handles the logic for creating a local git commit for a completed task."""
        if not self.git.is_repository():
            LOGGER.warning("Not a Git repository, skipping commit.")
            return False

        if self.git.is_dirty(expected_changes=task.changed_files):
            LOGGER.warning("Working tree contains unexpected changes, skipping commit to avoid including unrelated work.")
            return False

        if not self.git.create_branch(branch_name):
            LOGGER.error(f"Failed to create branch '{branch_name}'.")
            return False

        if not self.git.add(task.changed_files):
            LOGGER.error("Failed to stage changed files for commit.")
            return False

        commit_message = f"feat: Complete task '{task.objective}'\n\nTask-ID: {task.task_id}"
        if not self.git.commit(commit_message):
            LOGGER.error("Failed to create commit.")
            return False

        LOGGER.info(f"Successfully committed changes to branch '{branch_name}'.")
        return True

    def _perform_git_push(self, task: Task, branch_name: str) -> bool:
        """Handles pushing a task branch to the remote."""
        if not self.git.push(self.config.git_default_remote, branch_name, set_upstream=True):
            LOGGER.error(f"Failed to push branch '{branch_name}' to remote '{self.config.git_default_remote}'.")
            return False
        LOGGER.info(f"Successfully pushed branch '{branch_name}'.")
        return True

    def _perform_pr_creation(self, task: Task, branch_name: str):
        """Handles creating a pull request for a task."""
        from .remotes import build_remote_provider, RemoteError

        try:
            remote_provider = build_remote_provider(self.config)
            if not remote_provider:
                LOGGER.warning("No remote provider configured, skipping PR creation.")
                return

            existing_pr = remote_provider.find_pull_request_for_branch(branch_name)
            if existing_pr:
                LOGGER.info(f"Pull request for branch '{branch_name}' already exists: {existing_pr.url}")
                task.pull_request = existing_pr
                self.storage.save_task(task)
                return

            new_pr = remote_provider.create_pull_request(task, branch_name)
            LOGGER.info(f"Successfully created pull request: {new_pr.url}")
            task.pull_request = new_pr
            self.storage.save_task(task)

        except RemoteError as e:
            LOGGER.error(f"Failed to create pull request: {e}")
        except Exception as e:
            LOGGER.exception(f"An unexpected error occurred during PR creation: {e}")

    def _create_memories_from_run(self, task: Task, report: RunReport):
        if task.autonomous and self.config.git_commit_on_completion:
            self._perform_git_commit(task)

        if self.knowledge_manager and getattr(self.config, "knowledge_graph_enabled", True) and report.completed:
            with self.memory_lock:
                try:
                    self.knowledge_manager.promote_run_report(task, report)
                    self.knowledge_manager.save()
                except Exception as e:
                    LOGGER.warning("Could not promote knowledge from run report: %s", e)

        if not self.config.memory_enabled or not report.changed_files:
            return

        with self.memory_lock:
            memory = self.storage.load_project_memory()

            # Create a memory about the changed files' roles
            for file_path in report.changed_files:
                # Avoid creating duplicate memories for the same file from the same task
                if any(m.category == MemoryCategory.FILE_ROLE and m.related_path == file_path and m.source_task_id == task.task_id for m in memory.memories):
                    continue

                new_memory = Memory(
                    memory_id=str(uuid.uuid4()),
                    category=MemoryCategory.FILE_ROLE,
                    content=f"File '{file_path}' was modified to accomplish the task: '{task.objective}'. It likely plays a key role in this type of feature.",
                    timestamp=datetime.datetime.now(datetime.timezone.utc),
                    source_task_id=task.task_id,
                    related_path=file_path,
                    confidence=0.8
                )
                memory.memories.append(new_memory)

            self.storage.save_project_memory(memory)

    def _create_new_task(self, objective: str) -> Task:
        now = datetime.datetime.now(datetime.timezone.utc)
        return Task(
            task_id=str(uuid.uuid4()),
            objective=objective,
            status=TaskStatus.PENDING,
            created_at=now,
            updated_at=now,
        )

    def _create_checkpoint(self, task: Task, subtask: Subtask | None, description: str, context: ProjectContext, report: RunReport, extra_context: dict[str, Any] | None = None) -> Checkpoint:
        checkpoint_id = str(uuid.uuid4())
        now = datetime.datetime.now(datetime.timezone.utc)

        subtasks_in_plan = task.plan.subtasks if task.plan else []
        # Generate provider-agnostic continuation context
        continuation_context = {
            "task_objective": task.objective,
            "current_subtask_goal": subtask.goal if subtask else None,
            "completed_subtasks_summary": [s.title for s in subtasks_in_plan if s.status == SubtaskStatus.COMPLETED],
            "current_progress_description": description,
            "modified_files_summary": sorted(list(set(report.changed_files))),
            "repository_diff": self.git.diff(),
            "validation_state": {
                "last_executions": [exec.to_dict() for exec in report.executions[-3:]], # Last 3 executions
                "last_failures": [fail.to_dict() for fail in report.failures[-1:]], # Last failure
                "validation_plan": report.validation_plan.to_dict() if report.validation_plan else None,
            },
            "last_provider_event": report.outcome,
            "next_recommended_action": "Continue the current subtask from the existing repository state.",
        }
        # Merge extra context for Phase 3.13 (diagnostic state etc.)
        if extra_context:
            continuation_context.update(extra_context)

        # Merge tool exploration state if present in report
        if report:
            if report.plan and "plan" not in continuation_context:
                continuation_context["plan"] = report.plan.to_dict()
                continuation_context["plan_version"] = report.plan.version
                continuation_context["amendments"] = [a.to_dict() for a in report.plan.amendments]
            if report.tool_history and "tool_history" not in continuation_context:
                continuation_context["tool_history"] = history_to_dict(report.tool_history)
            if report.tool_metrics and "tool_metrics" not in continuation_context:
                continuation_context["tool_metrics"] = [m.to_dict() if hasattr(m, "to_dict") else m for m in report.tool_metrics]
            if report.recovery_state and "recovery_state" not in continuation_context:
                continuation_context["recovery_state"] = report.recovery_state.to_dict()
            if task.plan and isinstance(task.plan, TaskPlan) and "task_plan" not in continuation_context:
                continuation_context["task_plan"] = task.plan.to_dict()
                continuation_context["task_plan_version"] = getattr(task.plan, "version", 1)
                continuation_context["dag_amendments"] = [a.to_dict() for a in getattr(task.plan, "amendments", []) if hasattr(a, "to_dict")]
            if getattr(report, "review_consensus", None):
                continuation_context["review_consensus"] = [r.to_dict() if hasattr(r, "to_dict") else r for r in report.review_consensus]
            if getattr(report, "implementation_result", None):
                continuation_context["implementation_result"] = report.implementation_result.to_dict()
            if hasattr(self, "router"):
                continuation_context["specialist_routing_state"] = {
                    "planning": list(self.router.get_role_config(SpecialistRole.PLANNING)),
                    "implementation": list(self.router.get_role_config(SpecialistRole.IMPLEMENTATION)),
                    "repair": list(self.router.get_role_config(SpecialistRole.REPAIR)),
                    "review": list(self.router.get_role_config(SpecialistRole.REVIEW)),
                    "verification": list(self.router.get_role_config(SpecialistRole.VERIFICATION)),
                }

        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            task_id=task.task_id,
            subtask_id=subtask.subtask_id if subtask else "",
            timestamp=now,
            current_state_description=description,
            files_changed=sorted(list(set(report.changed_files))),
            repository_diff=self.git.diff(),
            validation_state=continuation_context["validation_state"], # Use part of continuation context
            last_provider_result={"outcome": report.outcome} if report.outcome else None,
            next_recommended_action=continuation_context["next_recommended_action"],
            continuation_context=continuation_context,
        )
        self.storage.save_checkpoint(checkpoint)
        task.latest_checkpoint_id = checkpoint_id
        if subtask:
            subtask.latest_checkpoint_id = checkpoint_id # Save on subtask
        self.storage.save_task(task)
        return checkpoint

    def _handle_temporary_provider_error(self, task: Task, subtask: Subtask, error: ProviderError, description: str, context: ProjectContext) -> None:
        self._record_provider_failure(task, subtask, error, description)
        task.status = TaskStatus.PAUSED
        subtask.status = SubtaskStatus.PAUSED
        subtask.completed_at = datetime.datetime.now(datetime.timezone.utc) # Mark as paused at this time
        # Create a checkpoint for resumption
        report = self._build_run_report(task) # Need a temporary report to build checkpoint context
        self._create_checkpoint(task, subtask, f"Paused due to {error.category}", context, report)
        self.storage.save_task(task)

    def _record_provider_failure(self, task: Task, subtask: Subtask | None, error: ProviderError, description: str) -> None:
        failure = FailureAnalysis(
            probable_root_cause=description,
            recommended_fix=str(error),
            category=getattr(error, "category", "PROVIDER_ERROR"),
            details={"retry_after_seconds": getattr(error, "retry_after_seconds", None)},
        )
        subtask_id = subtask.subtask_id if subtask else None
        task.execution_history.append({"type": "failure", "subtask_id": subtask_id, "failure": failure.to_dict()})
        task.outcome = getattr(error, "category", "PROVIDER_ERROR") # Update task outcome
        if subtask:
            subtask.provider_attempts.append({"outcome": getattr(error, "category", "PROVIDER_ERROR"), "error": str(error)})
        task.updated_at = datetime.datetime.now(datetime.timezone.utc)
        self.storage.save_task(task)

    def _validate(self, validation_plan: ValidationPlan):
        if not validation_plan.primary_commands:
            LOGGER.warning("No primary validation commands in plan")
            return []
        results = []
        for command_spec in validation_plan.primary_commands:
            if command_spec.destructive or command_spec.risk == "high":
                LOGGER.warning("Skipping high-risk validation command: %s", command_spec.display())
                continue

            LOGGER.info("Running validation: %s", command_spec.display())
            results.append(self.runner.run(command_spec))
            if not results[-1].succeeded:
                break
        return results

    def _git_changed_paths(self) -> set[str]:
        status = self.git.status()
        paths: set[str] = set()
        for line in status.splitlines():
            if line.startswith("##") or len(line) < 4:
                continue
            value = line[3:].strip()
            if " -> " in value:
                value = value.rsplit(" -> ", 1)[-1]
            paths.add(Path(value.strip('"')).as_posix())
        return paths

    def _truncate_execution_result(self, result: ExecutionResult) -> ExecutionResult:
        """Truncates stdout/stderr of an ExecutionResult to a configured limit."""
        limit = self.config.max_diagnostic_output_bytes

        def truncate(text: str) -> str:
            if len(text.encode('utf-8')) > limit:
                # A simple byte-based truncation might cut mid-character. This is a safer approach.
                return text[:limit] + "\n...[truncated]..."
            return text

        return ExecutionResult(command=result.command, exit_code=result.exit_code, stdout=truncate(result.stdout), stderr=truncate(result.stderr), duration_seconds=result.duration_seconds, timed_out=result.timed_out)

    def _build_run_report(self, task: Task) -> RunReport:
        # This function constructs a RunReport from the current Task state for CLI display.
        # It's a snapshot, not the persistent state.
        current_subtask = next((s for s in (task.plan.subtasks if task.plan else []) if s.subtask_id == task.current_subtask_id), None)

        # Placeholder for project context, plan, etc.
        # In a real scenario, these would be loaded from the checkpoint or derived.
        # For now, we'll use dummy values or assume they are available in the Orchestrator's state.
        dummy_context = ProjectContext(root=str(self.config.project))
        dummy_plan = Plan(objective=task.objective)
        dummy_validation_plan = ValidationPlan(commands=[], primary_commands=[], secondary_commands=[], skipped_commands=[], reasons=[], risk_level="low")
        dummy_impact = ChangeImpact(summary="No impact analysis available in report snapshot", targets=[])
        tool_metrics: list[ToolExecutionMetrics] = []
        tool_history: list[tuple[ToolCall, ToolResult]] = []
        recovery_state: RecoveryState | None = None
        review_consensus: list[ReviewConsensusRecord] = []
        specialist_routing_state: dict[str, Any] = {}
        implementation_result: ImplementationResult | None = None

        # Attempt to load from latest checkpoint if available
        if task.latest_checkpoint_id:
            try:
                checkpoint = self.storage.load_checkpoint(task.latest_checkpoint_id)
                # Reconstruct some report fields from checkpoint
                # This is a simplification; a full restoration would be more complex
                if checkpoint and checkpoint.validation_state and checkpoint.validation_state.get("validation_plan"):
                    dummy_validation_plan = ValidationPlan.from_dict(checkpoint.validation_state["validation_plan"])
                if checkpoint and checkpoint.continuation_context:
                    raw_metrics = checkpoint.continuation_context.get("tool_metrics", [])
                    if raw_metrics:
                        tool_metrics = [ToolExecutionMetrics.from_dict(m) if isinstance(m, dict) else m for m in raw_metrics]
                    raw_history = checkpoint.continuation_context.get("tool_history", [])
                    if raw_history:
                        tool_history = history_from_dict(raw_history)
                    raw_rec = checkpoint.continuation_context.get("recovery_state")
                    if raw_rec and isinstance(raw_rec, dict):
                        recovery_state = RecoveryState.from_dict(raw_rec)
                    raw_consensus = checkpoint.continuation_context.get("review_consensus", [])
                    if raw_consensus:
                        review_consensus = [ReviewConsensusRecord.from_dict(r) if isinstance(r, dict) else r for r in raw_consensus]
                    raw_impl = checkpoint.continuation_context.get("implementation_result")
                    if isinstance(raw_impl, dict):
                        implementation_result = ImplementationResult.from_dict(raw_impl)
                    specialist_routing_state = checkpoint.continuation_context.get("specialist_routing_state", {})
            except FileNotFoundError:
                pass # Checkpoint might be missing if task is new or corrupted

        # Extract executions and failures from task.execution_history
        executions = []
        failures = []
        for entry in task.execution_history:
            if entry.get("type") == "execution":
                executions.append(ExecutionResult.from_dict(entry["execution"]))
            elif entry.get("type") == "failure":
                failures.append(FailureAnalysis.from_dict(entry["failure"]))

        return RunReport(
            project=dummy_context, # Placeholder
            plan=dummy_plan, # Placeholder
            executions=executions,
            failures=failures,
            review=None, # Placeholder
            changed_files=[], # Placeholder
            iterations=current_subtask.attempts if current_subtask else 0,
            completed=task.status == TaskStatus.COMPLETED,
            validation_plan=dummy_validation_plan, # Placeholder
            impact=dummy_impact, # Placeholder
            dry_run=self.config.dry_run,
            approval_required=self.config.approval == "always",
            proposed_diff="", # Placeholder
            outcome=task.outcome,
            provider_metrics=task.provider_execution_history,
            task_id=task.task_id,
            subtask_id=current_subtask.subtask_id if current_subtask else None,
            tool_metrics=tool_metrics,
            tool_history=tool_history,
            recovery_state=recovery_state,
            review_consensus=review_consensus,
            specialist_routing_state=specialist_routing_state,
            implementation_result=implementation_result,
        )

