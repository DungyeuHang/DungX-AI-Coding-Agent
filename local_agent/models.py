from __future__ import annotations

import datetime
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Literal, Self


@dataclass(frozen=True)
class CommandSpec:
    name: str
    command: tuple[str, ...]
    reason: str = ""
    category: str = "other" # Added for ValidationIntelligence
    risk: Literal["low", "medium", "high"] = "low" # Added for ValidationIntelligence
    destructive: bool = False # Added for ValidationIntelligence

    def display(self) -> str:
        return " ".join(self.command)


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    PLAN_REVIEW = "plan_review" # Added for Phase 3.12
    PLAN_PROPOSED = "plan_proposed" # Added for Phase 3.14
    REJECTED = "rejected" # Added for Phase 3.12
    CANCELLED = "cancelled"

class ProviderCapability(str, Enum):
    PLANNING = "planning"
    IMPLEMENTATION = "implementation"
    REPAIR = "repair"
    REVIEW = "review"

class ProviderAvailability(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    COOLDOWN = "cooldown"
    FAILED = "failed" # Permanent failure, e.g., auth
    NOT_CONFIGURED = "not_configured"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"

class SubtaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused" # e.g., due to quota
    BLOCKED = "blocked"

@dataclass
class ExecutionResult:
    command: str
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0
    timed_out: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(**data)

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


@dataclass
class RepositoryFile:
    path: str
    extension: str
    size_bytes: int | None
    line_count: int | None
    language: str
    is_test: bool
    is_configuration: bool
    is_documentation: bool
    is_generated: bool
    is_entry_point: bool


@dataclass(frozen=True)
class FileRelationship:
    source: str
    target: str
    kind: str


@dataclass
class RepositoryMap:
    root: str
    project_metadata: dict[str, Any]
    languages: list[str]
    frameworks: list[str]
    files: list[RepositoryFile]
    directories: list[str]
    tests: list[str]
    configuration_files: list[str]
    entry_points: list[str]
    relationships: list[FileRelationship]
    ignored_paths: list[dict[str, str]]
    protected_paths: list[dict[str, str]]

    def compact(self) -> dict[str, Any]:
        return {
            "root": self.root, "project_metadata": self.project_metadata, "languages": self.languages,
            "frameworks": self.frameworks, "file_count": len(self.files), "directory_count": len(self.directories),
            "test_file_count": len(self.tests), "entry_point_count": len(self.entry_points),
            "relationship_count": len(self.relationships),
        }


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
    repository_map: RepositoryMap | None = None

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
            "repository_map": self.repository_map.compact() if self.repository_map else None,
        }


@dataclass
class FileOperation:
    action: str
    path: str
    content: str | None = None
    reason: str = ""
    patch: str | None = None


@dataclass
class PreparedChange:
    action: str
    path: str
    original: str | None
    resulting: str | None
    diff: str
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
    category: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    diagnostic_evidence: list[ExecutionResult] = field(default_factory=list) # Added for Phase 3.13

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["diagnostic_evidence"] = [e.to_dict() for e in self.diagnostic_evidence]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        # Make a copy to avoid mutating the caller's dict
        data = dict(data)
        data["diagnostic_evidence"] = [ExecutionResult.from_dict(e) for e in data.get("diagnostic_evidence", [])]
        return cls(**data)

@dataclass
class SymbolLocation:
    start_line: int
    end_line: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(**data)

@dataclass
class SymbolDefinition:
    name: str
    kind: Literal["class", "function", "method"]
    location: SymbolLocation
    parent: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["location"] = self.location.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        data = dict(data)
        data["location"] = SymbolLocation.from_dict(data["location"])
        return cls(**data)

@dataclass
class FileIndex:
    path: str
    language: str
    content_hash: str
    symbols: list[SymbolDefinition] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["symbols"] = [s.to_dict() for s in self.symbols]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        data = dict(data)
        data["symbols"] = [SymbolDefinition.from_dict(s) for s in data.get("symbols", [])]
        return cls(**data)

@dataclass
class SemanticIndex:
    files: Dict[str, FileIndex] = field(default_factory=dict) # Keyed by file path

    def to_dict(self) -> dict[str, Any]:
        return {"files": {path: fi.to_dict() for path, fi in self.files.items()}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(files={path: FileIndex.from_dict(fi) for path, fi in data.get("files", {}).items()})

    def find_symbol(self, name: str) -> list[SymbolDefinition]:
        matching_symbols: list[SymbolDefinition] = []
        for file_index in self.files.values():
            for symbol in file_index.symbols:
                if symbol.name == name:
                    matching_symbols.append(symbol)
        return matching_symbols

    def find_symbols(self, names: set[str]) -> list[tuple[str, SymbolDefinition]]:
        """Finds all symbol definitions matching any of the given names.
        Returns a list of (file_path, SymbolDefinition) tuples, sorted by symbol name, path, and line.
        """
        matching_symbols: list[tuple[str, SymbolDefinition]] = []
        for path in sorted(self.files.keys()):
            file_index = self.files[path]
            for symbol in file_index.symbols:
                if symbol.name in names:
                    matching_symbols.append((path, symbol))
        # Sort for deterministic output
        matching_symbols.sort(key=lambda item: (item[1].name, item[0], item[1].location.start_line))
        return matching_symbols

    def find_file_for_symbol(self, name: str) -> list[str]:
        """Finds all file paths containing a symbol definition with an exact name match."""
        matching_files: set[str] = set()
        for path, file_index in self.files.items():
            for symbol in file_index.symbols:
                if symbol.name == name:
                    matching_files.add(path)
                    break # Move to the next file
        return sorted(list(matching_files))

    def search_symbols(self, query: str) -> list[tuple[str, SymbolDefinition]]:
        """Finds all symbol definitions with a case-insensitive substring match.
        Returns a list of (file_path, SymbolDefinition) tuples, sorted by symbol name, path, and line.
        """
        matching_symbols: list[tuple[str, SymbolDefinition]] = []
        query_lower = query.lower()
        for path in sorted(self.files.keys()):
            file_index = self.files[path]
            for symbol in file_index.symbols:
                if query_lower in symbol.name.lower():
                    matching_symbols.append((path, symbol))
        # Sort for deterministic output
        matching_symbols.sort(key=lambda item: (item[1].name, item[0], item[1].location.start_line))
        return matching_symbols

@dataclass
class ProviderMetric:
    request_type: str
    input_size: int
    output_size: int
    model: str
    duration_seconds: float
    succeeded: bool
    error_category: str = ""
    approximate_input_tokens: int = 0
    approximate_output_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(**data)

# New for Phase 3.25
@dataclass
class PullRequestInfo:
    provider: str
    pr_id: str
    url: str
    status: str
    created_at: datetime.datetime

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        data["created_at"] = datetime.datetime.fromisoformat(data["created_at"])
        return cls(**data)

@dataclass
class ChangeTarget:
    path: str
    role: Literal["create", "modify", "test", "architecture", "unrelated"]
    confidence: float
    reason: str
    relationship: str | None = None
    risk: Literal["low", "medium", "high"] = "low"


@dataclass
class ChangeImpact:
    summary: str
    targets: list[ChangeTarget]

    def to_dict(self) -> dict[str, Any]:
        return {"summary": self.summary, "targets": [asdict(t) for t in self.targets]}

@dataclass
class ValidationCommand:
    name: str
    command: tuple[str, ...]
    category: Literal["unit_test", "integration_test", "e2e_test", "type_check", "lint", "build", "format", "destructive", "other"]
    confidence: float
    reason: str
    working_directory: str = "."
    destructive: bool = False
    timeout: int | None = None
    risk: Literal["low", "medium", "high"] = "low"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class ValidationPlan:
    commands: list[ValidationCommand]
    primary_commands: list[CommandSpec]
    secondary_commands: list[CommandSpec]
    skipped_commands: list[CommandSpec]
    reasons: list[str]
    risk_level: Literal["low", "medium", "high"]

    def to_dict(self) -> dict[str, Any]:
        return {"commands": [cmd.to_dict() for cmd in self.commands], "primary_commands": [cmd.to_dict() for cmd in self.primary_commands], "secondary_commands": [cmd.to_dict() for cmd in self.secondary_commands], "skipped_commands": [cmd.to_dict() for cmd in self.skipped_commands], "reasons": self.reasons, "risk_level": self.risk_level}

@dataclass
class Checkpoint:
    checkpoint_id: str
    task_id: str
    subtask_id: str
    timestamp: datetime.datetime
    current_state_description: str
    files_changed: list[str] = field(default_factory=list)
    repository_diff: str = ""
    validation_state: dict[str, Any] = field(default_factory=dict)
    last_provider_result: dict[str, Any] | None = None
    next_recommended_action: str = ""
    continuation_context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        data["timestamp"] = datetime.datetime.fromisoformat(data["timestamp"])
        return cls(**data)

@dataclass
class Subtask:
    subtask_id: str
    description: str = ""
    status: SubtaskStatus = SubtaskStatus.PENDING
    created_at: datetime.datetime | None = None
    updated_at: datetime.datetime | None = None
    dependencies: list[str] = field(default_factory=list)
    attempts: int = 0
    started_at: datetime.datetime | None = None
    title: str = "" # Added for Phase 3.11
    goal: str = "" # Added for Phase 3.11
    acceptance_criteria: list[str] = field(default_factory=list) # Added for Phase 3.11
    completed_at: datetime.datetime | None = None
    provider_attempts: list[dict[str, Any]] = field(default_factory=list)
    latest_checkpoint_id: str | None = None
    completion_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        if self.created_at:
            data["created_at"] = self.created_at.isoformat()
        if self.updated_at:
            data["updated_at"] = self.updated_at.isoformat()
        if self.started_at:
            data["started_at"] = self.started_at.isoformat()
        if self.completed_at:
            data["completed_at"] = self.completed_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        data["status"] = SubtaskStatus(data["status"])
        if data.get("created_at"):
            data["created_at"] = datetime.datetime.fromisoformat(data["created_at"])
        if data.get("updated_at"):
            data["updated_at"] = datetime.datetime.fromisoformat(data["updated_at"])
        if data.get("started_at"):
            data["started_at"] = datetime.datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            data["completed_at"] = datetime.datetime.fromisoformat(data["completed_at"])
        return cls(**data)

@dataclass
class TaskPlan:
    objective: str
    subtasks: list[Subtask]
    risks: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"objective": self.objective, "subtasks": [s.to_dict() for s in self.subtasks], "risks": self.risks, "assumptions": self.assumptions}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(objective=data["objective"], subtasks=[Subtask.from_dict(s) for s in data["subtasks"]], risks=data.get("risks", []), assumptions=data.get("assumptions", []))

# Phase 3.14: Typed Plan Modification Models
@dataclass
class SubtaskModification:
    subtask_id: str
    title: str | None = None
    goal: str | None = None
    acceptance_criteria: list[str] | None = None
    dependencies: list[str] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(**data)

@dataclass
class AddSubtask:
    subtask: Subtask

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self: # Existing fix, ensuring it's not regressed
        # Required for PlanProposal deserialization
        return cls(subtask=Subtask.from_dict(data["subtask"]))

@dataclass
class PlanProposal:
    reason: str
    modifications: list[SubtaskModification] = field(default_factory=list)
    additions: list[AddSubtask] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            reason=data.get("reason", ""),
            modifications=[SubtaskModification.from_dict(m) for m in data.get("modifications", [])],
            additions=[AddSubtask.from_dict(a) for a in data.get("additions", [])],
        )


@dataclass
class ApprovalPolicy:
    name: str
    action: Literal["auto_approve", "require_approval"]
    # Conditions (all must be met for the policy to match)
    if_risk_is_at_most: Literal["low", "medium", "high"] | None = None
    if_path_matches: list[str] | None = None  # glob patterns
    if_path_does_not_match: list[str] | None = None  # glob patterns
    if_max_lines_changed: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(**data)


# New for Phase 3.22
class MemoryCategory(str, Enum):
    ARCHITECTURE = "architecture"
    FILE_ROLE = "file_role"
    RECURRING_ERROR = "recurring_error"
    SUCCESSFUL_FIX = "successful_fix"
    PROJECT_CONVENTION = "project_convention"

@dataclass
class Memory:
    memory_id: str
    category: MemoryCategory
    content: str
    timestamp: datetime.datetime
    source_task_id: str
    related_path: str | None = None
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["category"] = self.category.value
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        data["category"] = MemoryCategory(data["category"])
        data["timestamp"] = datetime.datetime.fromisoformat(data["timestamp"])
        return cls(**data)

@dataclass
class ProjectMemory:
    memories: list[Memory] = field(default_factory=list)
    def to_dict(self) -> dict[str, Any]: return {"memories": [m.to_dict() for m in self.memories]}
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self: return cls(memories=[Memory.from_dict(m) for m in data.get("memories", [])])

# New for Phase 3.23
@dataclass
class CIFailureContext:
    failed_command: str
    exit_code: int
    stdout: str
    stderr: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(**data)

@dataclass
class Task:
    task_id: str
    objective: str
    status: TaskStatus
    created_at: datetime.datetime
    updated_at: datetime.datetime
    plan: TaskPlan | None = None # Changed from subtasks list to a full plan
    plan_proposal: PlanProposal | None = None # Added for Phase 3.14
    current_subtask_id: str | None = None
    changed_files: list[str] = field(default_factory=list) # New for Phase 3.24
    pull_request: PullRequestInfo | None = None # New for Phase 3.25
    initial_failure_context: CIFailureContext | None = None # New for Phase 3.23
    autonomous: bool = False # New for autonomous mode
    execution_history: list[dict[str, Any]] = field(default_factory=list)
    latest_checkpoint_id: str | None = None
    retry_info: dict[str, Any] = field(default_factory=dict)
    provider_execution_history: list[ProviderMetric] = field(default_factory=list)
    outcome: str = ""
    next_retry_at: datetime.datetime | None = None # Added for Phase 3.10
    assigned_to: str | None = None # Added for Phase 3.10

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        if self.plan:
            data["plan"] = self.plan.to_dict()
        if self.plan_proposal:
            data["plan_proposal"] = self.plan_proposal.to_dict()
        if self.initial_failure_context:
            data["initial_failure_context"] = self.initial_failure_context.to_dict()
        if self.pull_request:
            data["pull_request"] = self.pull_request.to_dict()
        data["provider_execution_history"] = [metric.to_dict() for metric in self.provider_execution_history] # Hardened
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        data["status"] = TaskStatus(data["status"])
        data["created_at"] = datetime.datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.datetime.fromisoformat(data["updated_at"])
        if data.get("plan"):
            data["plan"] = TaskPlan.from_dict(data["plan"])
        if data.get("plan_proposal"):
            data["plan_proposal"] = PlanProposal.from_dict(data["plan_proposal"])
        if data.get("initial_failure_context"):
            data["initial_failure_context"] = CIFailureContext.from_dict(data["initial_failure_context"])
        if data.get("pull_request"):
            data["pull_request"] = PullRequestInfo.from_dict(data["pull_request"])
        data["provider_execution_history"] = [ProviderMetric.from_dict(metric_data) for metric_data in data.get("provider_execution_history", [])]
        if data.get("next_retry_at"):
            data["next_retry_at"] = datetime.datetime.fromisoformat(data["next_retry_at"])
        return cls(**data)

    @property
    def subtasks(self) -> list[Subtask]:
        return self.plan.subtasks if self.plan else []

@dataclass
class ProviderRuntimeState:
    provider_id: str
    availability: ProviderAvailability
    cooldown_until: datetime.datetime | None = None
    last_error: str | None = None
    last_success: datetime.datetime | None = None
    consecutive_failures: int = 0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["availability"] = self.availability.value
        if self.cooldown_until:
            data["cooldown_until"] = self.cooldown_until.isoformat()
        if self.last_success:
            data["last_success"] = self.last_success.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        data["availability"] = ProviderAvailability(data["availability"])
        if data.get("cooldown_until"):
            data["cooldown_until"] = datetime.datetime.fromisoformat(data["cooldown_until"])
        if data.get("last_success"):
            data["last_success"] = datetime.datetime.fromisoformat(data["last_success"])
        return cls(**data)

@dataclass
class SchedulerState:
    provider_states: dict[str, ProviderRuntimeState] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"provider_states": {pid: state.to_dict() for pid, state in self.provider_states.items()}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(provider_states={pid: ProviderRuntimeState.from_dict(state_data) for pid, state_data in data.get("provider_states", {}).items()})

@dataclass
class RegisteredProvider:
    provider_id: str
    config: "AgentConfig"
    capabilities: set[ProviderCapability]
    priority: int

@dataclass
class ProviderConfig:
    provider_id: str
    priority: int
    enabled: bool
    config_overrides: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(**data)

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
    validation_plan: ValidationPlan | None = None # Added for ValidationIntelligence
    impact: ChangeImpact | None = None
    task_id: str | None = None # Added for Phase 3.9
    subtask_id: str | None = None # Added for Phase 3.9
    dry_run: bool = False
    approval_required: bool = False
    proposed_diff: str = ""
    outcome: str = ""
    provider_metrics: list[ProviderMetric] = field(default_factory=list)
