from __future__ import annotations

import datetime
import logging
import shlex
import uuid
from pathlib import Path

from .coding_agent import CodingAgent, PatchValidationError, UnsafeModificationError
from .commands import CommandRunner
from .config import AgentConfig
from .context import ContextSelector
from .failure import FailureAnalyzer
from .filesystem import ProjectFilesystem
from .git import GitIntegration
from .impact import ChangeImpactAnalyzer
from .models import ChangeImpact, Checkpoint, CommandSpec, ExecutionResult, FailureAnalysis, Plan, PlanProposal, ProjectContext, RunReport, Subtask, SubtaskStatus, TaskStatus, ValidationPlan
from .planner import Planner
from .providers import AIProvider, ProviderError, QuotaExceededError, RateLimitError
from .repository import RepositoryIntelligence
from .reviewer import Reviewer
from .validation import ValidationIntelligence

LOGGER = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, config: AgentConfig, provider: AIProvider, storage: TaskStorage): # Orchestrator now takes storage
        self.config = config
        self.provider = provider
        self.analyzer = RepositoryIntelligence(config.project)
        self.filesystem = ProjectFilesystem(config.project)
        self.git = GitIntegration(config.project)
        self.planner = Planner(provider)
        self.failure_analyzer = FailureAnalyzer(provider)
        self.reviewer = Reviewer(provider)
        self.impact_analyzer = ChangeImpactAnalyzer(config.project)
        self.runner = CommandRunner(config.project, config.command_timeout_seconds)
        self.validation_intelligence = ValidationIntelligence(config.project)
        self.storage = storage

    def analyze(self):
        return self.analyzer.analyze()

    def run(self, task: Task, subtask_id: str, progress=None, approval_callback=None) -> RunReport: # Modified signature for Phase 3.11
        def emit(message: str) -> None:
            LOGGER.info(message)
            if progress:
                progress(message)

        emit("[1/7] Analyzing project...")
        context = self.analyzer.scan()
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
        ).select(task.objective, context)

        emit("[2.5/7] Analyzing change impact...")
        impact = self.impact_analyzer.analyze(task.objective, context)
        context.metadata["change_impact"] = impact.to_dict()

        report = RunReport(project=context)
        report.impact = impact
        current_subtask = next((s for s in (task.plan.subtasks if task.plan else []) if s.subtask_id == subtask_id), None)
        task.current_subtask_id = subtask_id

        emit("[3/7] Creating implementation plan...")
        try:
            plan = self.planner.create_subtask_plan(current_subtask, context)
        except ProviderError as exc:
            self._record_provider_failure(task, current_subtask, exc, "planning provider request failed")
            emit(f"[3/7] Provider stopped the run: {report.outcome}")
            return report
        report.plan = plan
        preexisting = self._git_changed_paths()
        coding_agent = CodingAgent(self.filesystem, preexisting)
        failure: FailureAnalysis | None = None
        review = None

        # If resuming from a paused state, reset status to allow completion
        if task.status == TaskStatus.PAUSED and current_subtask is not None and current_subtask.latest_checkpoint_id:
            task.status = TaskStatus.PENDING
            current_subtask.status = SubtaskStatus.PENDING
            current_subtask.completed_at = None

        # Phase 3.14: Resume diagnostic / secondary-validation state from the latest checkpoint, if present.
        current_diagnostic_evidence: list[ExecutionResult] = []
        executed_diagnostic_names_this_iteration: list[str] = []
        if current_subtask and current_subtask.latest_checkpoint_id:
            try:
                checkpoint = self.storage.load_checkpoint(current_subtask.latest_checkpoint_id)
                current_diagnostic_evidence = [ExecutionResult.from_dict(d) for d in checkpoint.continuation_context.get("diagnostic_evidence", [])]
                executed_diagnostic_names_this_iteration = list( # Ensure it's a list for mutability
                    checkpoint.continuation_context.get("executed_diagnostic_names_this_iteration", [])
                )
                # Load the last failure from checkpoint to continue repair iterations
                last_failures = checkpoint.continuation_context.get("validation_state", {}).get("last_failures", [])
                if last_failures:
                    failure = FailureAnalysis.from_dict(last_failures[0])
                else:
                    failure = None
            except FileNotFoundError:
                pass  # Checkpoint may be missing if the task is new or corrupted

        validation_plan = self.validation_intelligence.select_commands(
            task.objective, impact, list(getattr(context, "validation_commands", None) or [])
        )

        try:
            for iteration in range(1, self.config.max_iterations + 1):
                report.iterations = iteration
                emit("[4/7] Implementing changes..." if iteration == 1 else f"[5/7] Applying repair for iteration {iteration}...")
                try:
                    operations = self.provider.generate_code(task, plan, context, failure=failure, review=review)
                except ProviderError as exc:
                    self._record_provider_failure(task, current_subtask, exc, "implementation provider request failed")
                    emit(f"[4/7] Provider stopped the run: {report.outcome}")
                    if isinstance(exc, (RateLimitError, QuotaExceededError)):
                        task.status = TaskStatus.PAUSED
                        current_subtask.status = SubtaskStatus.PAUSED
                        current_subtask.completed_at = datetime.datetime.now(datetime.timezone.utc)
                        # Create a checkpoint to preserve failure state and diagnostic state for resumption
                        extra_context = {
                            'diagnostic_evidence': [e.to_dict() for e in current_diagnostic_evidence],
                            'executed_diagnostic_names_this_iteration': executed_diagnostic_names_this_iteration
                        }
                        self._create_checkpoint(task, current_subtask, f"Paused due to provider error during repair", context, report, extra_context=extra_context)
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
                if prepared and self.config.approval == "always":
                    report.approval_required = True
                    approved = bool(approval_callback(prepared)) if approval_callback else False
                    if not approved:
                        emit("[4/7] Changes not approved; stopping before write")
                        report.proposed_diff = proposed_diff
                        break
                report.changed_files.extend(coding_agent.apply_prepared(prepared))
                report.proposed_diff = coding_agent.diff()
                self._create_checkpoint(task, current_subtask, "After code generation, before validation", context, report) # Checkpoint

                emit("[5/7] Running validation...")
                executions = self._validate(validation_plan)
                report.executions.extend(executions)
                task.execution_history.extend([{"type": "execution", "subtask_id": current_subtask.subtask_id, "execution": exec.to_dict()} for exec in executions]) # Record executions
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

                            selected_cmd_spec = self.provider.select_diagnostic_command(current_subtask.goal, plan, context, failed, available_diagnostics)

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
                        failure = self.failure_analyzer.analyze(failed, coding_agent.diff() or self.git.diff(), context, plan)
                    except ProviderError as exc:
                        self._record_provider_failure(task, current_subtask, exc, "failure-analysis provider request failed")
                        emit(f"[5/7] Provider stopped the run: {report.outcome}")
                        break

                    # Attach diagnostic evidence to the failure analysis
                    failure.diagnostic_evidence = current_diagnostic_evidence

                    # Phase 3.14: Propose a plan modification if the failure seems architectural
                    if failure.category in {"MISSING_DEPENDENCY", "ARCHITECTURAL_FLAW"}: # Example categories
                        emit("[5.5/7] Failure suggests a planning issue. Proposing plan modification...")
                        plan_proposal: PlanProposal | None = self.provider.propose_plan_modification(task.objective, task.plan, failure)
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
                    report.review = self.reviewer.review(current_subtask.goal, plan, coding_agent.diff() or self.git.diff(), context)
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
            if report.completed:
                task.status = TaskStatus.COMPLETED
                current_subtask.status = SubtaskStatus.COMPLETED
                current_subtask.completed_at = datetime.datetime.now(datetime.timezone.utc)
            else:
                if task.status != TaskStatus.PAUSED: # Only set FAILED if not already paused
                    task.status = TaskStatus.FAILED
                    current_subtask.status = SubtaskStatus.FAILED
                    current_subtask.completed_at = datetime.datetime.now(datetime.timezone.utc)

        except Exception as e:
            LOGGER.exception("Unhandled exception during task execution: %s", e)
            task.status = TaskStatus.FAILED
            current_subtask.status = SubtaskStatus.FAILED
            current_subtask.completed_at = datetime.datetime.now(datetime.timezone.utc)
            report.outcome = "UNHANDLED_EXCEPTION"
            report.failures.append(FailureAnalysis(probable_root_cause="Unhandled exception", recommended_fix=str(e)))
        finally:
            task.updated_at = datetime.datetime.now(datetime.timezone.utc)
            self.storage.save_task(task)
            report.provider_metrics = list(self.provider.provider_metrics) # Capture metrics for the run
            task.provider_execution_history.extend(report.provider_metrics) # Add to task history
            self.storage.save_task(task) # Save task with updated metrics

        return self._build_run_report(task)

    def _create_new_task(self, objective: str) -> Task:
        now = datetime.datetime.now(datetime.timezone.utc)
        return Task(
            task_id=str(uuid.uuid4()),
            objective=objective,
            status=TaskStatus.PENDING,
            created_at=now,
            updated_at=now,
        )

    def _create_checkpoint(self, task: Task, subtask: Subtask, description: str, context: ProjectContext, report: RunReport, extra_context: dict[str, Any] | None = None) -> Checkpoint:
        checkpoint_id = str(uuid.uuid4())
        now = datetime.datetime.now(datetime.timezone.utc)

        subtasks_in_plan = task.plan.subtasks if task.plan else []
        # Generate provider-agnostic continuation context
        continuation_context = {
            "task_objective": task.objective,
            "current_subtask_goal": subtask.goal,
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

        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            task_id=task.task_id,
            subtask_id=subtask.subtask_id,
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

    def _record_provider_failure(self, task: Task, subtask: Subtask, error: ProviderError, description: str) -> None:
        failure = FailureAnalysis(
            probable_root_cause=description,
            recommended_fix=str(error),
            category=error.category,
            details={"retry_after_seconds": error.retry_after_seconds},
        )
        task.execution_history.append({"type": "failure", "subtask_id": subtask.subtask_id, "failure": failure.to_dict()})
        task.outcome = error.category # Update task outcome
        subtask.provider_attempts.append({"outcome": error.category, "error": str(error)})
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

        # Attempt to load from latest checkpoint if available
        if task.latest_checkpoint_id:
            try:
                checkpoint = self.storage.load_checkpoint(task.latest_checkpoint_id)
                # Reconstruct some report fields from checkpoint
                # This is a simplification; a full restoration would be more complex
                if checkpoint.validation_state and checkpoint.validation_state.get("validation_plan"):
                    dummy_validation_plan = ValidationPlan.from_dict(checkpoint.validation_state["validation_plan"])
                # You might also reconstruct plan, impact from checkpoint.continuation_context
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
        )

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
