from __future__ import annotations

import datetime
import logging
import shlex
import uuid
from pathlib import Path

import threading
from .approval import ApprovalPolicyEngine, RISK_LEVEL_MAP
from .coding_agent import CodingAgent, PatchValidationError, UnsafeModificationError
from .commands import CommandRunner
from .config import AgentConfig
from .context import ContextSelector
from .failure import FailureAnalyzer
from .filesystem import ProjectFilesystem
from .git import GitIntegration
from .impact import ChangeImpactAnalyzer
from .models import (
    ApprovalPolicy,
    ChangeImpact,
    Checkpoint,
    CommandSpec,
    ExecutionResult,
    FailureAnalysis,
    Plan,
    PlanProposal,
    ProjectMemory,
    Memory,
    ProjectContext,
    ProviderCapability,
    ReviewResult,
    RunReport,
    Subtask,
    SubtaskStatus,
    Task,
    TaskStatus,
    ToolCall,
    ToolExecutionMetrics,
    ToolResult,
    ValidationPlan,
)
from .models import MemoryCategory
from .planner import GraphValidator, Planner
from .providers import AIProvider, ProviderError, QuotaExceededError, RateLimitError, build_provider
from .repository import RepositoryIntelligence
from .reviewer import Reviewer
from .tool_engine import ToolEngine, history_from_dict, history_to_dict
from .tools import ToolRegistry
from .validation import ValidationIntelligence

LOGGER = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, config: AgentConfig, storage: "TaskStorage", scheduler: "Scheduler", repo_lock: threading.Lock, memory_lock: threading.Lock):
        self.config = config
        self.storage = storage
        self.scheduler = scheduler
        self.repo_lock = repo_lock
        self.memory_lock = memory_lock
        self.analyzer = RepositoryIntelligence(config.project)
        self.filesystem = ProjectFilesystem(config.project)
        self.git = GitIntegration(config.project)
        self.impact_analyzer = ChangeImpactAnalyzer(config.project)
        self.runner = CommandRunner(config.project, config.command_timeout_seconds)
        self.validation_intelligence = ValidationIntelligence(config.project)
        policies = [ApprovalPolicy.from_dict(p) for p in config.approval_policies]
        self.approval_engine = ApprovalPolicyEngine(policies)

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
        project_memory = self.storage.load_project_memory() if self.config.memory_enabled else ProjectMemory()
        if self.config.validation_commands:
            context.validation_commands = [CommandSpec(f"explicit-{index}", tuple(shlex.split(command)), "explicit CLI configuration") for index, command in enumerate(self.config.validation_commands, 1)]
        emit("[2/7] Building context...")
        ContextSelector(
            self.config.project,
            max_files=self.config.max_context_files,
            max_chars=self.config.planning_context_bytes,
            max_file_chars=self.config.max_context_file_bytes,
            max_tokens=self.config.max_context_tokens,
            dependency_depth=self.config.dependency_depth,
            project_memory=project_memory,
        ).select(task.objective, context)

        emit("[2.5/7] Analyzing change impact...")
        impact = self.impact_analyzer.analyze(task.objective, context)
        context.metadata["change_impact"] = impact.to_dict()
        context.metadata["current_subtask_id"] = subtask_id

        report = RunReport(project=context, task_id=task.task_id, subtask_id=subtask_id)
        report.impact = impact
        current_subtask = next((s for s in (task.plan.subtasks if task.plan else []) if s.subtask_id == subtask_id), None)
        task.current_subtask_id = subtask_id

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
            if current_subtask:
                plan = self._execute_with_specialist(
                    task,
                    ProviderCapability.PLANNING,
                    lambda provider: Planner(provider).create_subtask_plan(current_subtask, context),
                    "planning"
                )
            else:
                raw_plan = self._execute_with_specialist(
                    task,
                    ProviderCapability.PLANNING,
                    lambda provider: provider.generate_plan(task.objective, context),
                    "planning"
                )
                if isinstance(raw_plan, Plan):
                    plan = raw_plan
                elif isinstance(raw_plan, dict):
                    plan = Plan.from_dict(raw_plan)
                else:
                    plan = Plan(summary=task.objective, steps=[str(task.objective)])
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

        # Phase 3.14: Resume diagnostic / secondary-validation state and Phase 4.0 tool history from the latest checkpoint, if present.
        current_diagnostic_evidence: list[ExecutionResult] = []
        executed_diagnostic_names_this_iteration: list[str] = []
        current_tool_history: list[tuple[ToolCall, ToolResult]] = []
        latest_checkpoint_id = (current_subtask.latest_checkpoint_id if current_subtask else None) or task.latest_checkpoint_id
        if latest_checkpoint_id:
            try:
                checkpoint = self.storage.load_checkpoint(latest_checkpoint_id)
                if checkpoint is not None:
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
                    # Load the last failure from checkpoint to continue repair iterations
                    last_failures = checkpoint.continuation_context.get("validation_state", {}).get("last_failures", [])
                    if last_failures:
                        failure = FailureAnalysis.from_dict(last_failures[0])
                    else:
                        failure = None
            except FileNotFoundError:
                pass  # Checkpoint may be missing if the task is new or corrupted


        try:
            for iteration in range(1, self.config.max_iterations + 1):
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
                    report.failures.append(failure)
                    emit(f"[4/7] Rejected generated changes: {exc}")
                    if iteration < self.config.max_iterations:
                        continue
                    break
                except UnsafeModificationError as exc:
                    failure = FailureAnalysis("AI-generated change was rejected by the safety/patch validator", [], str(exc))
                    report.failures.append(failure)
                    emit(f"[4/7] Rejected generated changes: {exc}")
                    if iteration < self.config.max_iterations:
                        continue
                    break
                proposed_diff = "".join(change.diff for change in prepared)
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
                executions = self._validate(validation_plan)
                report.executions.extend(executions)
                sub_id = current_subtask.subtask_id if current_subtask else None
                task.execution_history.extend([{"type": "execution", "subtask_id": sub_id, "execution": exec.to_dict()} for exec in executions]) # Record executions
                self.storage.save_task(task)
                failed = next((result for result in executions if not result.succeeded), None)
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
                            semantic_index=getattr(context, "semantic_index", None),
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
                    report.failures.append(failure)
                    emit("[5/7] Analyzing failure...")
                    if iteration < self.config.max_iterations:
                        continue
                    break
                self._create_checkpoint(task, current_subtask, "After successful validation, before review", context, report) # Checkpoint

                emit("[6/7] Reviewing changes...")
                try:
                    report.review = self._execute_with_specialist(
                        task,
                        ProviderCapability.REVIEW,
                        lambda provider: Reviewer(provider).review(current_subtask.goal if current_subtask else task.objective, plan, coding_agent.diff() or self.git.diff(), context),
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
                if review.verdict == "APPROVED":
                    report.completed = True
                    break
                if iteration >= self.config.max_iterations:
                    break
                emit("[6/7] Review requested changes; continuing...")
                emit("[7/7] Complete")

            # Status setting logic moved outside the loop to ensure it always runs
            if not report.plan_proposal:
                if report.completed:
                    if current_subtask:
                        current_subtask.status = SubtaskStatus.COMPLETED
                        current_subtask.completed_at = datetime.datetime.now(datetime.timezone.utc)
                    # Phase 3.24: Persist changed files on successful completion
                    task.changed_files = sorted(list(set(task.changed_files + report.changed_files)))
                    if task.plan and all(s.status == SubtaskStatus.COMPLETED for s in task.plan.subtasks):
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

    def _select_specialists(self, task: Task, capability: ProviderCapability):
        if self.scheduler is not None:
            provider_configs = self.scheduler._select_providers(task, {capability})
            for provider_config in provider_configs:
                provider = self.scheduler._build_provider_instance(provider_config.provider)
                if provider:
                    yield provider
            return
        yield build_provider(self.config)

    def _execute_with_specialist(self, task: Task, capability: ProviderCapability, action, stage_name: str):
        for provider in self._select_specialists(task, capability):
            return action(provider)
        raise ProviderError(f"No available and capable provider found for stage: {stage_name}")

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
    ) -> tuple[list[FileOperation], list[tuple[ToolCall, ToolResult]]]:
        capability = ProviderCapability.REPAIR if failure else ProviderCapability.IMPLEMENTATION

        def _action(provider: AIProvider) -> tuple[list[FileOperation], list[tuple[ToolCall, ToolResult]]]:
            try:
                provider_caps = getattr(provider, "capabilities", None)
            except Exception:
                provider_caps = None
            task_obj = task.objective if hasattr(task, "objective") else str(task)

            if isinstance(provider_caps, (set, frozenset)) and ProviderCapability.TOOL_USE in provider_caps:
                registry = ToolRegistry(
                    self.config.project,
                    filesystem=self.filesystem,
                    command_runner=self.runner,
                    semantic_index=getattr(context, "semantic_index", None),
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
            if report.tool_history and "tool_history" not in continuation_context:
                continuation_context["tool_history"] = history_to_dict(report.tool_history)
            if report.tool_metrics and "tool_metrics" not in continuation_context:
                continuation_context["tool_metrics"] = [m.to_dict() if hasattr(m, "to_dict") else m for m in report.tool_metrics]

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
        )

