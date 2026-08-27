from __future__ import annotations

from dataclasses import asdict, dataclass, field
import datetime
from enum import Enum
import hashlib
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        d = dict(data)
        if isinstance(d.get("command"), list):
            d["command"] = tuple(d["command"])
        return cls(**d)


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
    TOOL_USE = "tool_use"

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


def normalize_diff_for_signature(diff: str) -> str:
    """Normalize unified diff content to create a stable hash invariant to trivial whitespace and timestamps."""
    if not diff:
        return ""
    lines = []
    for line in diff.splitlines():
        stripped = line.strip()
        if stripped.startswith(("---", "+++")):
            parts = stripped.split()
            if len(parts) >= 2:
                lines.append(parts[0] + " " + parts[1])
            else:
                lines.append(parts[0])
        elif stripped.startswith(("+", "-")) and not stripped.startswith(("+++", "---")):
            lines.append(stripped[0] + " " + stripped[1:].strip())
        elif stripped.startswith("@@"):
            lines.append("@@")
    normalized = "\n".join(lines)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


@dataclass
class RepairSignature:
    """Fingerprint of a repair attempt for anti-repeat detection."""
    iteration: int
    failure_category: str
    root_cause_hash: str  # truncated hash of normalized failure cause / target
    patch_hash: str       # truncated hash of normalized proposed_diff
    affected_files: list[str] = field(default_factory=list)
    failed_target: str = ""
    strategy_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            iteration=data.get("iteration", 0),
            failure_category=data.get("failure_category", ""),
            root_cause_hash=data.get("root_cause_hash", ""),
            patch_hash=data.get("patch_hash", ""),
            affected_files=list(data.get("affected_files", [])),
            failed_target=data.get("failed_target", ""),
            strategy_note=data.get("strategy_note", ""),
        )


@dataclass
class RecoveryState:
    """Tracks recovery progress across iterations within a single task run."""
    completed_iterations: int = 0
    repair_signatures: list[RepairSignature] = field(default_factory=list)
    failure_history: list[FailureAnalysis] = field(default_factory=list)
    review_history: list[ReviewResult] = field(default_factory=list)
    consecutive_same_failure_count: int = 0
    abort_reason: str = ""

    def has_duplicate_signature(self, sig: RepairSignature) -> bool:
        """Check if this exact failure+patch combination was already tried."""
        return any(
            s.root_cause_hash == sig.root_cause_hash and s.patch_hash == sig.patch_hash
            for s in self.repair_signatures
        )

    def is_duplicate_patch(self, patch_hash: str) -> bool:
        """Check if this exact patch hash was already attempted in previous iterations."""
        if not patch_hash:
            return False
        return any(s.patch_hash == patch_hash for s in self.repair_signatures)

    def record_attempt(
        self,
        iteration: int,
        failure: FailureAnalysis | None,
        diff: str,
        affected_files: list[str],
        strategy_note: str = "",
    ) -> RepairSignature:
        category = failure.category if failure else "INITIAL_IMPLEMENTATION"
        root_cause_str = (failure.probable_root_cause if failure else "").strip().lower()
        root_cause_hash = hashlib.sha256(root_cause_str.encode("utf-8")).hexdigest()[:16] if root_cause_str else ""
        patch_hash = normalize_diff_for_signature(diff)
        failed_target = ""
        if failure:
            if failure.diagnostic_evidence:
                failed_target = failure.diagnostic_evidence[0].command
            elif failure.details and "path" in failure.details:
                failed_target = str(failure.details["path"])

        sig = RepairSignature(
            iteration=iteration,
            failure_category=category,
            root_cause_hash=root_cause_hash,
            patch_hash=patch_hash,
            affected_files=list(affected_files),
            failed_target=failed_target,
            strategy_note=strategy_note,
        )
        self.repair_signatures.append(sig)
        return sig

    def record_failure(self, failure: FailureAnalysis) -> None:
        if self.failure_history:
            prev = self.failure_history[-1]
            if prev.probable_root_cause.strip().lower() == failure.probable_root_cause.strip().lower() or (
                prev.category and prev.category == failure.category and prev.category in {"PATCH_VALIDATION", "UNSAFE_MODIFICATION"}
            ):
                self.consecutive_same_failure_count += 1
            else:
                self.consecutive_same_failure_count = 1
        else:
            self.consecutive_same_failure_count = 1
        self.failure_history.append(failure)

    def record_review(self, review: ReviewResult) -> None:
        self.review_history.append(review)

    def build_recovery_summary(self, max_chars: int = 1500) -> str:
        """Produce a concise summary of previous attempts to guide model repair without token explosion."""
        if not self.failure_history and not self.review_history and not self.repair_signatures:
            return ""

        lines = ["--- Recovery Context & History ---"]
        if self.repair_signatures:
            lines.append("Previous Attempts:")
            for sig in self.repair_signatures[-3:]:
                files_str = ", ".join(sig.affected_files) if sig.affected_files else "none"
                lines.append(f"  - Iteration {sig.iteration}: Modified [{files_str}] for {sig.failure_category}")

        if self.failure_history:
            last_f = self.failure_history[-1]
            lines.append(f"Latest Failure: {last_f.probable_root_cause[:200]}")
            if last_f.recommended_fix:
                lines.append(f"Recommended Fix Direction: {last_f.recommended_fix[:200]}")

        if self.review_history:
            last_r = self.review_history[-1]
            if last_r.verdict != "APPROVED":
                findings_str = "; ".join(last_r.findings[:3])
                lines.append(f"Latest Review Feedback: {last_r.summary[:150]} (Findings: {findings_str[:150]})")

        lines.append("Guidance: Do NOT repeat identical failed modifications. Explore alternative implementations.")
        text = "\n".join(lines)
        if len(text) > max_chars:
            suffix = "\n...[truncated]"

    def to_dict(self) -> dict[str, Any]:
        return {
            "completed_iterations": self.completed_iterations,
            "repair_signatures": [s.to_dict() for s in self.repair_signatures],
            "failure_history": [f.to_dict() for f in self.failure_history],
            "review_history": [asdict(r) for r in self.review_history],
            "consecutive_same_failure_count": self.consecutive_same_failure_count,
            "abort_reason": self.abort_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        if not data or not isinstance(data, dict):
            return cls()
        return cls(
            completed_iterations=data.get("completed_iterations", 0),
            repair_signatures=[RepairSignature.from_dict(s) for s in data.get("repair_signatures", []) if isinstance(s, dict)],
            failure_history=[FailureAnalysis.from_dict(f) for f in data.get("failure_history", []) if isinstance(f, dict)],
            review_history=[ReviewResult(**r) for r in data.get("review_history", []) if isinstance(r, dict)],
            consecutive_same_failure_count=data.get("consecutive_same_failure_count", 0),
            abort_reason=data.get("abort_reason", ""),
        )


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
    # Actual token counts extracted from the API response (None when not reported).
    actual_input_tokens: int | None = None
    actual_output_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        # Accept pre-existing serialised metrics that pre-date the actual_* fields.
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        safe_data = {k: v for k, v in data.items() if k in known_fields}
        return cls(**safe_data)

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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        data = dict(data)
        if isinstance(data.get("command"), list):
            data["command"] = tuple(data["command"])
        return cls(**data)

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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            commands=[ValidationCommand.from_dict(c) if isinstance(c, dict) else c for c in data.get("commands", [])],
            primary_commands=[CommandSpec.from_dict(c) if isinstance(c, dict) else c for c in data.get("primary_commands", [])],
            secondary_commands=[CommandSpec.from_dict(c) if isinstance(c, dict) else c for c in data.get("secondary_commands", [])],
            skipped_commands=[CommandSpec.from_dict(c) if isinstance(c, dict) else c for c in data.get("skipped_commands", [])],
            reasons=list(data.get("reasons", [])),
            risk_level=data.get("risk_level", "low"),
        )

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
    plan_proposal: PlanProposal | None = None
    tool_metrics: list[ToolExecutionMetrics] = field(default_factory=list)
    tool_history: list[tuple[ToolCall, ToolResult]] = field(default_factory=list)
    recovery_state: RecoveryState | None = None


class ProviderError(RuntimeError):
    """Raised when an AI provider cannot complete a structured operation."""

    category = "UNKNOWN_PROVIDER_ERROR"

    def __init__(self, message: str, *, retry_after_seconds: float | None = None):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class AuthenticationError(ProviderError):
    category = "AUTHENTICATION_ERROR"


class RateLimitError(ProviderError):
    category = "RATE_LIMIT"


class QuotaExceededError(RateLimitError):
    category = "QUOTA_EXCEEDED"


class InvalidRequestError(ProviderError):
    category = "INVALID_REQUEST"


class ModelUnavailableError(ProviderError):
    category = "MODEL_UNAVAILABLE"


class NetworkError(ProviderError):
    category = "NETWORK_ERROR"


class UnknownProviderError(ProviderError):
    category = "UNKNOWN_PROVIDER_ERROR"


# Phase 4.0: Dynamic Agentic Tool Engine Models
@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("ToolDefinition.name must be a non-empty string")
        if not self.description or not isinstance(self.description, str):
            raise ValueError("ToolDefinition.description must be a non-empty string")
        if not isinstance(self.parameters, dict):
            raise ValueError("ToolDefinition.parameters must be a dict")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(**data)


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    tool_name: str
    arguments: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.call_id or not isinstance(self.call_id, str):
            raise ValueError("ToolCall.call_id must be a non-empty string")
        if not self.tool_name or not isinstance(self.tool_name, str):
            raise ValueError("ToolCall.tool_name must be a non-empty string")
        if not isinstance(self.arguments, dict):
            raise ValueError("ToolCall.arguments must be a dict")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(**data)


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    tool_name: str
    output: str
    is_error: bool = False
    truncated: bool = False

    def __post_init__(self) -> None:
        if not self.call_id or not isinstance(self.call_id, str):
            raise ValueError("ToolResult.call_id must be a non-empty string")
        if not self.tool_name or not isinstance(self.tool_name, str):
            raise ValueError("ToolResult.tool_name must be a non-empty string")
        if not isinstance(self.output, str):
            raise ValueError("ToolResult.output must be a string")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(**data)


@dataclass
class ToolExecutionMetrics:
    """Detailed telemetry and observability metrics for a ToolEngine execution session."""
    total_calls: int = 0
    unique_calls: int = 0
    repeated_calls: int = 0
    calls_by_tool: dict[str, int] = field(default_factory=dict)
    total_output_bytes: int = 0
    output_bytes_by_tool: dict[str, int] = field(default_factory=dict)
    truncated_results: int = 0
    tool_errors: int = 0
    circuit_breaker_events: int = 0
    steps_used: int = 0
    history_entries: int = 0
    termination_reason: str | None = None
    completed: bool = False
    elapsed_ms: float = 0.0
    compacted_entries: int = 0
    model_context_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "unique_calls": self.unique_calls,
            "repeated_calls": self.repeated_calls,
            "calls_by_tool": dict(self.calls_by_tool),
            "total_output_bytes": self.total_output_bytes,
            "output_bytes_by_tool": dict(self.output_bytes_by_tool),
            "truncated_results": self.truncated_results,
            "tool_errors": self.tool_errors,
            "circuit_breaker_events": self.circuit_breaker_events,
            "steps_used": self.steps_used,
            "history_entries": self.history_entries,
            "termination_reason": self.termination_reason,
            "completed": self.completed,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "compacted_entries": self.compacted_entries,
            "model_context_bytes": self.model_context_bytes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            total_calls=data.get("total_calls", 0),
            unique_calls=data.get("unique_calls", 0),
            repeated_calls=data.get("repeated_calls", 0),
            calls_by_tool=dict(data.get("calls_by_tool", {})),
            total_output_bytes=data.get("total_output_bytes", 0),
            output_bytes_by_tool=dict(data.get("output_bytes_by_tool", {})),
            truncated_results=data.get("truncated_results", 0),
            tool_errors=data.get("tool_errors", 0),
            circuit_breaker_events=data.get("circuit_breaker_events", 0),
            steps_used=data.get("steps_used", 0),
            history_entries=data.get("history_entries", 0),
            termination_reason=data.get("termination_reason"),
            completed=data.get("completed", False),
            elapsed_ms=float(data.get("elapsed_ms", 0.0)),
            compacted_entries=data.get("compacted_entries", 0),
            model_context_bytes=data.get("model_context_bytes", 0),
        )


class PolicyAction(str, Enum):
    ALLOW = "allow"
    REJECT = "reject"
    TERMINATE = "terminate"
    CIRCUIT_BREAKER = "circuit_breaker"


@dataclass(frozen=True)
class PolicyDecision:
    action: PolicyAction
    reason: str | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.action, PolicyAction):
            if isinstance(self.action, str):
                object.__setattr__(self, "action", PolicyAction(self.action))
            else:
                raise ValueError(f"Invalid PolicyAction: {self.action}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "reason": self.reason,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            action=PolicyAction(data["action"]),
            reason=data.get("reason"),
            message=data.get("message"),
        )


@dataclass(frozen=True)
class ToolExecutionPolicy:
    """Configurable execution policy governing tool usage, limits, and circuit breakers."""
    max_tool_steps: int = 8
    max_tool_output_bytes: int = 4000
    total_tool_budget_bytes: int = 32000
    max_consecutive_repeats: int = 3
    per_tool_limits: dict[str, int] = field(default_factory=dict)
    disallowed_tools: set[str] = field(default_factory=set)
    compaction_window: int = 2
    max_context_bytes: int = 8000

    def __post_init__(self) -> None:
        if not isinstance(self.max_tool_steps, int) or self.max_tool_steps <= 0:
            raise ValueError(f"max_tool_steps must be an integer > 0, got {self.max_tool_steps}")
        if not isinstance(self.max_tool_output_bytes, int) or self.max_tool_output_bytes <= 0:
            raise ValueError(f"max_tool_output_bytes must be an integer > 0, got {self.max_tool_output_bytes}")
        if not isinstance(self.total_tool_budget_bytes, int) or self.total_tool_budget_bytes <= 0:
            raise ValueError(f"total_tool_budget_bytes must be an integer > 0, got {self.total_tool_budget_bytes}")
        if not isinstance(self.max_consecutive_repeats, int) or self.max_consecutive_repeats <= 0:
            raise ValueError(f"max_consecutive_repeats must be an integer > 0, got {self.max_consecutive_repeats}")
        if not isinstance(self.compaction_window, int) or self.compaction_window <= 0:
            raise ValueError(f"compaction_window must be an integer > 0, got {self.compaction_window}")
        if not isinstance(self.max_context_bytes, int) or self.max_context_bytes <= 0:
            raise ValueError(f"max_context_bytes must be an integer > 0, got {self.max_context_bytes}")

        if not isinstance(self.per_tool_limits, dict):
            raise ValueError("per_tool_limits must be a dictionary")
        for tool_name, limit in self.per_tool_limits.items():
            if not isinstance(tool_name, str) or not tool_name.strip():
                raise ValueError(f"Tool name in per_tool_limits must be a non-empty string, got {tool_name!r}")
            if not isinstance(limit, int) or limit < 0:
                raise ValueError(f"Limit for tool '{tool_name}' in per_tool_limits must be a non-negative integer, got {limit}")

        if not isinstance(self.disallowed_tools, (set, frozenset, list, tuple)):
            raise ValueError("disallowed_tools must be a set or list of strings")
        for tool_name in self.disallowed_tools:
            if not isinstance(tool_name, str) or not tool_name.strip():
                raise ValueError(f"Tool name in disallowed_tools must be a non-empty string, got {tool_name!r}")
        if isinstance(self.disallowed_tools, (list, tuple)):
            object.__setattr__(self, "disallowed_tools", set(self.disallowed_tools))

    def evaluate_call(
        self,
        tool_call: ToolCall,
        steps_used: int,
        total_output_bytes: int,
        calls_by_tool: dict[str, int],
        consecutive_repeat_count: int,
    ) -> PolicyDecision:
        """Evaluate whether a candidate tool call should be allowed, rejected, circuit-broken, or terminated."""
        # 1. Step budget check
        if steps_used >= self.max_tool_steps:
            return PolicyDecision(
                action=PolicyAction.TERMINATE,
                reason="max_steps_exceeded",
                message=f"Reached maximum allowed tool steps ({self.max_tool_steps}).",
            )

        # 2. Total byte budget check
        if total_output_bytes >= self.total_tool_budget_bytes:
            return PolicyDecision(
                action=PolicyAction.TERMINATE,
                reason="budget_exhausted",
                message=f"Exceeded total tool output budget ({self.total_tool_budget_bytes} bytes).",
            )

        # 3. Disallowed tool check
        if tool_call.tool_name in self.disallowed_tools:
            return PolicyDecision(
                action=PolicyAction.REJECT,
                reason="disallowed_tool",
                message=f"Tool '{tool_call.tool_name}' is disallowed by execution policy.",
            )

        # 4. Per-tool invocation limit check
        tool_call_count = calls_by_tool.get(tool_call.tool_name, 0)
        tool_limit = self.per_tool_limits.get(tool_call.tool_name)
        if tool_limit is not None and tool_call_count > tool_limit:
            return PolicyDecision(
                action=PolicyAction.REJECT,
                reason="tool_limit_exceeded",
                message=f"Tool '{tool_call.tool_name}' exceeded its per-tool invocation limit of {tool_limit}.",
            )

        # 5. Consecutive repeat circuit breaker
        if consecutive_repeat_count >= self.max_consecutive_repeats:
            return PolicyDecision(
                action=PolicyAction.CIRCUIT_BREAKER,
                reason="consecutive_repeats_exceeded",
                message=(
                    f"Circuit breaker triggered: repeated identical tool call '{tool_call.tool_name}' "
                    f"detected {self.max_consecutive_repeats} times consecutively. Please proceed to generate code changes or try a different action."
                ),
            )

        return PolicyDecision(action=PolicyAction.ALLOW)

    def evaluate_result(
        self,
        tool_call: ToolCall,
        tool_result: ToolResult,
        steps_used: int,
        total_output_bytes: int,
    ) -> PolicyDecision:
        """Evaluate session policy after a tool result has been added."""
        if total_output_bytes >= self.total_tool_budget_bytes:
            return PolicyDecision(
                action=PolicyAction.TERMINATE,
                reason="budget_exhausted",
                message=f"Exceeded total tool output budget ({self.total_tool_budget_bytes} bytes).",
            )
        if steps_used >= self.max_tool_steps:
            return PolicyDecision(
                action=PolicyAction.TERMINATE,
                reason="max_steps_exceeded",
                message=f"Reached maximum allowed tool steps ({self.max_tool_steps}).",
            )
        return PolicyDecision(action=PolicyAction.ALLOW)

    def evaluate_continuation(
        self,
        steps_used: int,
        total_output_bytes: int,
    ) -> PolicyDecision:
        """Evaluate whether another exploration turn is allowed under policy constraints."""
        if steps_used >= self.max_tool_steps:
            return PolicyDecision(
                action=PolicyAction.TERMINATE,
                reason="max_steps_exceeded",
                message=f"Reached maximum allowed tool steps ({self.max_tool_steps}).",
            )
        if total_output_bytes >= self.total_tool_budget_bytes:
            return PolicyDecision(
                action=PolicyAction.TERMINATE,
                reason="budget_exhausted",
                message=f"Exceeded total tool output budget ({self.total_tool_budget_bytes} bytes).",
            )
        return PolicyDecision(action=PolicyAction.ALLOW)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_tool_steps": self.max_tool_steps,
            "max_tool_output_bytes": self.max_tool_output_bytes,
            "total_tool_budget_bytes": self.total_tool_budget_bytes,
            "max_consecutive_repeats": self.max_consecutive_repeats,
            "per_tool_limits": dict(self.per_tool_limits),
            "disallowed_tools": sorted(list(self.disallowed_tools)),
            "compaction_window": self.compaction_window,
            "max_context_bytes": self.max_context_bytes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            max_tool_steps=data.get("max_tool_steps", 8),
            max_tool_output_bytes=data.get("max_tool_output_bytes", 4000),
            total_tool_budget_bytes=data.get("total_tool_budget_bytes", 32000),
            max_consecutive_repeats=data.get("max_consecutive_repeats", 3),
            per_tool_limits=dict(data.get("per_tool_limits", {})),
            disallowed_tools=set(data.get("disallowed_tools", [])),
            compaction_window=data.get("compaction_window", 2),
            max_context_bytes=data.get("max_context_bytes", 8000),
        )


