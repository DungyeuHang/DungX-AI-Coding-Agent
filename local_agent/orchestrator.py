from __future__ import annotations

import logging
import shlex
from pathlib import Path

from .analyzer import RepositoryAnalyzer
from .coding_agent import CodingAgent
from .commands import CommandRunner
from .config import AgentConfig
from .failure import FailureAnalyzer
from .filesystem import ProjectFilesystem
from .git import GitIntegration
from .models import CommandSpec, FailureAnalysis, RunReport
from .planner import Planner
from .providers import AIProvider
from .reviewer import Reviewer

LOGGER = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, config: AgentConfig, provider: AIProvider):
        self.config = config
        self.provider = provider
        self.analyzer = RepositoryAnalyzer(config.project)
        self.filesystem = ProjectFilesystem(config.project)
        self.git = GitIntegration(config.project)
        self.planner = Planner(provider)
        self.failure_analyzer = FailureAnalyzer(provider)
        self.reviewer = Reviewer(provider)
        self.runner = CommandRunner(config.project, config.command_timeout_seconds)

    def analyze(self):
        return self.analyzer.analyze()

    def run(self, task: str, progress=None) -> RunReport:
        def emit(message: str) -> None:
            LOGGER.info(message)
            if progress:
                progress(message)

        emit("[1/7] Analyzing project...")
        context = self.analyzer.analyze()
        if self.config.validation_commands:
            context.validation_commands = [CommandSpec(f"explicit-{index}", tuple(shlex.split(command)), "explicit CLI configuration") for index, command in enumerate(self.config.validation_commands, 1)]
        emit("[2/7] Building context...")
        emit("[3/7] Creating implementation plan...")
        plan = self.planner.create_plan(task, context)
        report = RunReport(project=context, plan=plan)
        preexisting = self._git_changed_paths()
        coding_agent = CodingAgent(self.filesystem, preexisting)
        failure: FailureAnalysis | None = None
        review = None

        for iteration in range(1, self.config.max_iterations + 1):
            report.iterations = iteration
            emit("[4/7] Implementing changes..." if iteration == 1 else f"[5/7] Applying repair for iteration {iteration}...")
            operations = self.provider.generate_code(task, plan, context, failure=failure, review=review)
            report.changed_files.extend(coding_agent.apply(operations, plan))
            emit("[5/7] Running validation...")
            executions = self._validate(context.validation_commands)
            report.executions.extend(executions)
            failed = next((result for result in executions if not result.succeeded), None)
            if failed is not None:
                emit("[5/7] Validation failed")
                failure = self.failure_analyzer.analyze(failed, self.git.diff(), context, plan)
                report.failures.append(failure)
                emit("[5/7] Analyzing failure...")
                if iteration < self.config.max_iterations:
                    continue
                break
            emit("[6/7] Reviewing changes...")
            report.review = self.reviewer.review(task, plan, self.git.diff(), context)
            review = report.review
            if review.verdict == "APPROVED":
                report.completed = True
                break
            if iteration >= self.config.max_iterations:
                break
            emit("[6/7] Review requested changes; continuing...")
        emit("[7/7] Complete")
        return report

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
