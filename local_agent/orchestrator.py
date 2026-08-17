from __future__ import annotations

import logging
import shlex
from pathlib import Path

from .coding_agent import CodingAgent, PatchValidationError, UnsafeModificationError
from .commands import CommandRunner
from .config import AgentConfig
from .context import ContextSelector
from .failure import FailureAnalyzer
from .filesystem import ProjectFilesystem
from .git import GitIntegration
from .impact import ChangeImpactAnalyzer
from .models import CommandSpec, FailureAnalysis, RunReport
from .planner import Planner
from .providers import AIProvider, ProviderError
from .reviewer import Reviewer

LOGGER = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, config: AgentConfig, provider: AIProvider):
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

    def analyze(self):
        return self.analyzer.analyze()

    def run(self, task: str, progress=None, approval_callback=None) -> RunReport:
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
        ).select(task, context)

        emit("[2.5/7] Analyzing change impact...")
        impact = self.impact_analyzer.analyze(task, context)
        context.metadata["change_impact"] = impact.to_dict()

        report = RunReport(project=context)
        report.impact = impact
        emit("[3/7] Creating implementation plan...")
        try:
            plan = self.planner.create_plan(task, context)
        except ProviderError as exc:
            self._record_provider_failure(report, exc, "planning provider request failed")
            emit(f"[3/7] Provider stopped the run: {report.outcome}")
            return report
        report.plan = plan
        preexisting = self._git_changed_paths()
        coding_agent = CodingAgent(self.filesystem, preexisting)
        failure: FailureAnalysis | None = None
        review = None

        for iteration in range(1, self.config.max_iterations + 1):
            report.iterations = iteration
            emit("[4/7] Implementing changes..." if iteration == 1 else f"[5/7] Applying repair for iteration {iteration}...")
            try:
                operations = self.provider.generate_code(task, plan, context, failure=failure, review=review)
            except ProviderError as exc:
                self._record_provider_failure(report, exc, "implementation provider request failed")
                emit(f"[4/7] Provider stopped the run: {report.outcome}")
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
            emit("[5/7] Running validation...")
            executions = self._validate(context.validation_commands)
            report.executions.extend(executions)
            failed = next((result for result in executions if not result.succeeded), None)
            if failed is not None:
                emit("[5/7] Validation failed")
                try:
                    failure = self.failure_analyzer.analyze(failed, coding_agent.diff() or self.git.diff(), context, plan)
                except ProviderError as exc:
                    self._record_provider_failure(report, exc, "failure-analysis provider request failed")
                    emit(f"[5/7] Provider stopped the run: {report.outcome}")
                    break
                failure.category = failure.category or "VALIDATION_FAILURE"
                report.failures.append(failure)
                emit("[5/7] Analyzing failure...")
                if iteration < self.config.max_iterations:
                    continue
                break
            emit("[6/7] Reviewing changes...")
            try:
                report.review = self.reviewer.review(task, plan, coding_agent.diff() or self.git.diff(), context)
            except ProviderError as exc:
                self._record_provider_failure(report, exc, "review provider request failed")
                emit(f"[6/7] Provider stopped the run: {report.outcome}")
                break
            review = report.review
            if review.verdict == "APPROVED":
                report.completed = True
                break
            if iteration >= self.config.max_iterations:
                break
            emit("[6/7] Review requested changes; continuing...")
        emit("[7/7] Complete")
        report.provider_metrics = list(self.provider.provider_metrics)
        if report.completed:
            report.outcome = "COMPLETED"
        return report

    def _record_provider_failure(self, report: RunReport, error: ProviderError, description: str) -> None:
        report.failures.append(FailureAnalysis(
            probable_root_cause=description,
            recommended_fix=str(error),
            category=error.category,
            details={"retry_after_seconds": error.retry_after_seconds},
        ))
        report.outcome = error.category
        report.provider_metrics = list(self.provider.provider_metrics)

    def _validate(self, commands: list[CommandSpec]):
        if not commands:
            LOGGER.warning("No validation commands detected")
            return []
        results = []
        for command in commands:
            LOGGER.info("Running validation: %s", command.display())
            results.append(self.runner.run(command))
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
