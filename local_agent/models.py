from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CommandSpec:
    name: str
    command: tuple[str, ...]
    reason: str = ""

    def display(self) -> str:
        return " ".join(self.command)


@dataclass
class ExecutionResult:
    command: str
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0
    timed_out: bool = False

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


@dataclass
class ProjectContext:
    root: str
    directories: list[str] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)
    config_files: list[str] = field(default_factory=list)
    dependency_files: list[str] = field(default_factory=list)
    documentation_files: list[str] = field(default_factory=list)
    test_files: list[str] = field(default_factory=list)
    build_files: list[str] = field(default_factory=list)
    lint_files: list[str] = field(default_factory=list)
    typecheck_files: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    validation_commands: list[CommandSpec] = field(default_factory=list)
    git_status: str = ""

    def compact(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "directories": self.directories[:200],
            "source_files": self.source_files[:400],
            "config_files": self.config_files,
            "dependency_files": self.dependency_files,
            "documentation_files": self.documentation_files,
            "test_files": self.test_files[:200],
            "build_files": self.build_files,
            "lint_files": self.lint_files,
            "typecheck_files": self.typecheck_files,
            "metadata": self.metadata,
            "validation_commands": [c.display() for c in self.validation_commands],
            "git_status": self.git_status,
        }


@dataclass
class FileOperation:
    action: str
    path: str
    content: str | None = None
    reason: str = ""


@dataclass
class Plan:
    objective: str
    files_to_inspect: list[str] = field(default_factory=list)
    files_likely_to_change: list[str] = field(default_factory=list)
    files_likely_to_create: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    validation_strategy: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)


@dataclass
class FailureAnalysis:
    probable_root_cause: str
    affected_files: list[str] = field(default_factory=list)
    recommended_fix: str = ""


@dataclass
class ReviewResult:
    verdict: str
    summary: str
    findings: list[str] = field(default_factory=list)


@dataclass
class RunReport:
    project: ProjectContext
    plan: Plan | None = None
    executions: list[ExecutionResult] = field(default_factory=list)
    failures: list[FailureAnalysis] = field(default_factory=list)
    review: ReviewResult | None = None
    changed_files: list[str] = field(default_factory=list)
    iterations: int = 0
    completed: bool = False
