from __future__ import annotations

from dataclasses import asdict, dataclass, field
import datetime
from enum import Enum
import hashlib
from typing import Any, Dict, Literal, Self
import uuid


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

class SpecialistRole(str, Enum):
    PLANNING = "planning"
    IMPLEMENTATION = "implementation"
    REPAIR = "repair"
    REVIEW = "review"
    VERIFICATION = "verification"

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
    SUPERSEDED = "superseded" # Phase 4.9: invalidated upstream subtask preserved in history
    PRUNED = "pruned" # Phase 4.9: obsolete subtask pruned from active execution

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


@dataclass(frozen=True)
class ScopeExpansionProposal:
    """Structured proposal generated when implementation or planning discovers a missing file."""
    path: str
    reason: str
    relationship: str = "dependency"
    evidence: str = ""
    originating_stage: str = "implementation"
    is_create: bool = False
    confidence: float = 1.0
    subtask_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        if not isinstance(data, dict):
            return cls(path="", reason="")
        return cls(
            path=str(data.get("path", "")),
            reason=str(data.get("reason", "")),
            relationship=str(data.get("relationship", "dependency")),
            evidence=str(data.get("evidence", "")),
            originating_stage=str(data.get("originating_stage", "implementation")),
            is_create=bool(data.get("is_create", False)),
            confidence=float(data.get("confidence", 1.0)),
            subtask_id=data.get("subtask_id"),
        )


@dataclass(frozen=True)
class PlanAmendment:
    """Record of an accepted scope amendment to a Plan."""
    amendment_id: str
    version: int
    timestamp: datetime.datetime
    proposal: ScopeExpansionProposal
    approved_by: Literal["deterministic_policy", "user_approval"] = "deterministic_policy"
    previous_allowed_paths: list[str] = field(default_factory=list)
    new_allowed_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "amendment_id": self.amendment_id,
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
            "proposal": self.proposal.to_dict(),
            "approved_by": self.approved_by,
            "previous_allowed_paths": sorted(list(self.previous_allowed_paths)),
            "new_allowed_paths": sorted(list(self.new_allowed_paths)),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        raw_ts = data.get("timestamp")
        if isinstance(raw_ts, str):
            ts = datetime.datetime.fromisoformat(raw_ts)
        elif isinstance(raw_ts, datetime.datetime):
            ts = raw_ts
        else:
            ts = datetime.datetime.now(datetime.timezone.utc)

        raw_prop = data.get("proposal", {})
        prop = ScopeExpansionProposal.from_dict(raw_prop) if isinstance(raw_prop, dict) else raw_prop

        return cls(
            amendment_id=str(data.get("amendment_id", "")),
            version=int(data.get("version", 1)),
            timestamp=ts,
            proposal=prop,
            approved_by=data.get("approved_by", "deterministic_policy"),
            previous_allowed_paths=list(data.get("previous_allowed_paths", [])),
            new_allowed_paths=list(data.get("new_allowed_paths", [])),
        )


@dataclass
class Plan:
    objective: str
    files_to_inspect: list[str] = field(default_factory=list)
    files_likely_to_change: list[str] = field(default_factory=list)
    files_likely_to_create: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    validation_strategy: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    version: int = 1
    amendments: list[PlanAmendment] = field(default_factory=list)

    @property
    def allowed_paths(self) -> set[str]:
        return set(self.files_likely_to_change + self.files_likely_to_create)

    def apply_amendment(
        self,
        proposal: ScopeExpansionProposal,
        approved_by: Literal["deterministic_policy", "user_approval"] = "deterministic_policy",
    ) -> PlanAmendment:
        if proposal.path in self.allowed_paths:
            raise ValueError(f"Cannot amend plan: path '{proposal.path}' is already in allowed scope")

        prev = sorted(list(self.allowed_paths))
        if proposal.is_create:
            if proposal.path not in self.files_likely_to_create:
                self.files_likely_to_create.append(proposal.path)
        else:
            if proposal.path not in self.files_likely_to_change:
                self.files_likely_to_change.append(proposal.path)

        self.version += 1
        amendment = PlanAmendment(
            amendment_id=str(uuid.uuid4()),
            version=self.version,
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            proposal=proposal,
            approved_by=approved_by,
            previous_allowed_paths=prev,
            new_allowed_paths=sorted(list(self.allowed_paths)),
        )
        self.amendments.append(amendment)
        return amendment

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "files_to_inspect": list(self.files_to_inspect),
            "files_likely_to_change": list(self.files_likely_to_change),
            "files_likely_to_create": list(self.files_likely_to_create),
            "steps": list(self.steps),
            "validation_strategy": list(self.validation_strategy),
            "risks": list(self.risks),
            "version": self.version,
            "amendments": [a.to_dict() for a in self.amendments],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        if not isinstance(data, dict):
            return cls(objective="")
        raw_amendments = data.get("amendments", [])
        amendments = [PlanAmendment.from_dict(a) if isinstance(a, dict) else a for a in raw_amendments] if isinstance(raw_amendments, list) else []
        return cls(
            objective=str(data.get("objective", "")),
            files_to_inspect=list(data.get("files_to_inspect", [])),
            files_likely_to_change=list(data.get("files_likely_to_change", []) or data.get("files_to_modify", [])),
            files_likely_to_create=list(data.get("files_likely_to_create", []) or data.get("files_to_create", [])),
            steps=list(data.get("steps", [])),
            validation_strategy=list(data.get("validation_strategy", []) or data.get("validation_commands", [])),
            risks=list(data.get("risks", [])),
            version=int(data.get("version", 1)),
            amendments=amendments,
        )


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
            if max_chars >= len(suffix):
                text = text[: max_chars - len(suffix)] + suffix
            else:
                text = text[:max_chars]
        return text

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

class DAGExecutionStage(str, Enum):
    INIT = "init"
    PLANNED = "planned"
    READY = "ready"
    RUNNING = "running"
    WORKER_COMMITTED = "worker_committed"
    INTEGRATING = "integrating"
    INTEGRATED = "integrated"
    TIER2_VERIFIED = "tier2_verified"
    KNOWLEDGE_PROMOTED = "knowledge_promoted"
    CLEANED = "cleaned"
    COMPLETED = "completed"
    FAILED = "failed"
    RECOVERY_BLOCKED = "recovery_blocked"


# Phase 4.16: Multi-Turn Interactive Implementation State Machine and Models
class MultiTurnState(str, Enum):
    IDLE = "idle"
    PLANNING = "planning"
    IMPLEMENTING = "implementing"
    INSPECTING = "inspecting"
    TESTING = "testing"
    ANALYZING_FAILURE = "analyzing_failure"
    REPAIRING = "repairing"
    REVIEWING = "reviewing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

    @classmethod
    def valid_transitions(cls) -> dict[MultiTurnState, set[MultiTurnState]]:
        return {
            cls.IDLE: {cls.PLANNING, cls.IMPLEMENTING, cls.FAILED, cls.PAUSED},
            cls.PLANNING: {cls.IMPLEMENTING, cls.FAILED, cls.PAUSED},
            cls.IMPLEMENTING: {cls.INSPECTING, cls.TESTING, cls.REVIEWING, cls.FAILED, cls.PAUSED},
            cls.INSPECTING: {cls.IMPLEMENTING, cls.TESTING, cls.FAILED, cls.PAUSED},
            cls.TESTING: {cls.ANALYZING_FAILURE, cls.REVIEWING, cls.VERIFYING, cls.COMPLETED, cls.FAILED, cls.PAUSED},
            cls.ANALYZING_FAILURE: {cls.REPAIRING, cls.FAILED, cls.PAUSED},
            cls.REPAIRING: {cls.INSPECTING, cls.TESTING, cls.FAILED, cls.PAUSED},
            cls.REVIEWING: {cls.VERIFYING, cls.REPAIRING, cls.COMPLETED, cls.FAILED, cls.PAUSED},
            cls.VERIFYING: {cls.COMPLETED, cls.REPAIRING, cls.FAILED, cls.PAUSED},
            cls.PAUSED: {cls.PLANNING, cls.IMPLEMENTING, cls.REPAIRING, cls.TESTING, cls.REVIEWING, cls.VERIFYING, cls.FAILED},
            cls.COMPLETED: set(),
            cls.FAILED: set(),
        }

    @classmethod
    def can_transition(cls, from_state: MultiTurnState | str, to_state: MultiTurnState | str) -> bool:
        try:
            src = cls(from_state) if isinstance(from_state, str) else from_state
            dst = cls(to_state) if isinstance(to_state, str) else to_state
            return dst in cls.valid_transitions().get(src, set())
        except ValueError:
            return False


@dataclass
class ImplementationTurn:
    turn_id: str
    task_id: str
    subtask_id: str
    turn_number: int
    stage: str = "idle"
    provider: str = ""
    model: str = ""
    prompt_summary: str = ""
    prompt_metadata: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    tests_executed: list[dict[str, Any]] = field(default_factory=list)
    failures_detected: list[dict[str, Any]] = field(default_factory=list)
    repair_reason: str | None = None
    started_at: datetime.datetime = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    completed_at: datetime.datetime | None = None
    status: str = "running"
    file_operations: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "task_id": self.task_id,
            "subtask_id": self.subtask_id,
            "turn_number": self.turn_number,
            "stage": self.stage,
            "provider": self.provider,
            "model": self.model,
            "prompt_summary": self.prompt_summary,
            "prompt_metadata": dict(self.prompt_metadata),
            "tool_calls": list(self.tool_calls),
            "tool_results": list(self.tool_results),
            "tests_executed": list(self.tests_executed),
            "failures_detected": list(self.failures_detected),
            "repair_reason": self.repair_reason,
            "started_at": self.started_at.isoformat() if isinstance(self.started_at, datetime.datetime) else str(self.started_at),
            "completed_at": self.completed_at.isoformat() if isinstance(self.completed_at, datetime.datetime) else (str(self.completed_at) if self.completed_at else None),
            "status": self.status,
            "file_operations": list(self.file_operations),
            "metadata": dict(self.metadata),
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImplementationTurn:
        if not isinstance(data, dict):
            return cls(turn_id="", task_id="", subtask_id="", turn_number=0)
        raw_start = data.get("started_at")
        started_at = datetime.datetime.fromisoformat(raw_start) if isinstance(raw_start, str) else (raw_start or datetime.datetime.now(datetime.timezone.utc))
        raw_comp = data.get("completed_at")
        completed_at = datetime.datetime.fromisoformat(raw_comp) if isinstance(raw_comp, str) else raw_comp
        return cls(
            turn_id=str(data.get("turn_id", "")),
            task_id=str(data.get("task_id", "")),
            subtask_id=str(data.get("subtask_id", "")),
            turn_number=int(data.get("turn_number", 0)),
            stage=str(data.get("stage", "idle")),
            provider=str(data.get("provider", "")),
            model=str(data.get("model", "")),
            prompt_summary=str(data.get("prompt_summary", "")),
            prompt_metadata=dict(data.get("prompt_metadata") or {}),
            tool_calls=list(data.get("tool_calls") or []),
            tool_results=list(data.get("tool_results") or []),
            tests_executed=list(data.get("tests_executed") or []),
            failures_detected=list(data.get("failures_detected") or []),
            repair_reason=data.get("repair_reason"),
            started_at=started_at,
            completed_at=completed_at,
            status=str(data.get("status", "running")),
            file_operations=list(data.get("file_operations") or []),
            metadata=dict(data.get("metadata") or {}),
            error_message=data.get("error_message"),
        )


@dataclass
class MultiTurnExecutionReport:
    task_id: str = ""
    subtask_id: str = ""
    success: bool = False
    turns: list[ImplementationTurn] = field(default_factory=list)
    total_turns: int = 0
    repair_turns: int = 0
    review_turns: int = 0
    final_state: str = "idle"
    termination_reason: str = "none"
    elapsed_time_seconds: float = 0.0
    file_operations: list[FileOperation] = field(default_factory=list)
    tool_metrics: list[ToolExecutionMetrics] = field(default_factory=list)
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "subtask_id": self.subtask_id,
            "success": self.success,
            "turns": [t.to_dict() if hasattr(t, "to_dict") else t for t in self.turns],
            "total_turns": self.total_turns,
            "repair_turns": self.repair_turns,
            "review_turns": self.review_turns,
            "final_state": self.final_state,
            "termination_reason": self.termination_reason,
            "elapsed_time_seconds": round(self.elapsed_time_seconds, 4),
            "file_operations": [op.to_dict() if hasattr(op, "to_dict") else (op.__dict__ if hasattr(op, "__dict__") else op) for op in self.file_operations],
            "tool_metrics": [m.to_dict() if hasattr(m, "to_dict") else m for m in self.tool_metrics],
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MultiTurnExecutionReport:
        if not isinstance(data, dict):
            return cls()
        raw_turns = data.get("turns") or []
        turns = [ImplementationTurn.from_dict(t) if isinstance(t, dict) else t for t in raw_turns]
        raw_ops = data.get("file_operations") or []
        file_ops = [FileOperation(**op) if isinstance(op, dict) else op for op in raw_ops]
        raw_metrics = data.get("tool_metrics") or []
        tool_metrics = [ToolExecutionMetrics.from_dict(m) if isinstance(m, dict) else m for m in raw_metrics]
        return cls(
            task_id=str(data.get("task_id", "")),
            subtask_id=str(data.get("subtask_id", "")),
            success=bool(data.get("success", False)),
            turns=turns,
            total_turns=int(data.get("total_turns", len(turns))),
            repair_turns=int(data.get("repair_turns", 0)),
            review_turns=int(data.get("review_turns", 0)),
            final_state=str(data.get("final_state", "idle")),
            termination_reason=str(data.get("termination_reason", "none")),
            elapsed_time_seconds=float(data.get("elapsed_time_seconds", 0.0)),
            file_operations=file_ops,
            tool_metrics=tool_metrics,
            error_message=data.get("error_message"),
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
    active_worktrees: list[dict[str, Any]] = field(default_factory=list)
    integration_branch: str | None = None
    schema_version: str = "4.15.0"
    dag_stage: str = "init"
    subtask_states: dict[str, str] = field(default_factory=dict)
    integrated_subtasks: list[str] = field(default_factory=list)
    verified_subtasks: list[str] = field(default_factory=list)
    promoted_subtasks: list[str] = field(default_factory=list)
    cleaned_worktrees: list[str] = field(default_factory=list)
    base_commit: str = ""
    integration_commit: str | None = None
    turns: list[dict[str, Any]] = field(default_factory=list)
    current_turn_number: int = 0
    turn_stage: str = "idle"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat() if isinstance(self.timestamp, datetime.datetime) else str(self.timestamp)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        d = dict(data)
        raw_ts = d.get("timestamp")
        if isinstance(raw_ts, str):
            try:
                d["timestamp"] = datetime.datetime.fromisoformat(raw_ts)
            except Exception:
                d["timestamp"] = datetime.datetime.now(datetime.timezone.utc)
        elif not isinstance(raw_ts, datetime.datetime):
            d["timestamp"] = datetime.datetime.now(datetime.timezone.utc)
        d.setdefault("active_worktrees", [])
        d.setdefault("integration_branch", None)
        d.setdefault("schema_version", "4.15.0")
        d.setdefault("dag_stage", "init")
        d.setdefault("subtask_states", {})
        d.setdefault("integrated_subtasks", [])
        d.setdefault("verified_subtasks", [])
        d.setdefault("promoted_subtasks", [])
        d.setdefault("cleaned_worktrees", [])
        d.setdefault("base_commit", "")
        d.setdefault("integration_commit", None)
        d.setdefault("turns", [])
        d.setdefault("current_turn_number", 0)
        d.setdefault("turn_stage", "idle")
        return cls(**d)

# Phase 4.10 / Phase 4.11: Cross-Subtask Semantic Contract & Behavioral Verification Models
@dataclass
class ExportedSymbol:
    symbol_id: str
    name: str
    kind: str  # "class", "function", "variable", "type", "endpoint"
    file_path: str
    signature: str = ""
    description: str = ""
    verified: bool = False
    verification_source: str = ""

    def __post_init__(self):
        if len(self.signature) > 500:
            self.signature = self.signature[:497] + "..."
        if len(self.description) > 500:
            self.description = self.description[:497] + "..."
        if len(self.verification_source) > 200:
            self.verification_source = self.verification_source[:197] + "..."

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol_id": self.symbol_id,
            "name": self.name,
            "kind": self.kind,
            "file_path": self.file_path,
            "signature": self.signature,
            "description": self.description,
            "verified": self.verified,
            "verification_source": self.verification_source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            symbol_id=str(data.get("symbol_id", "")),
            name=str(data.get("name", "")),
            kind=str(data.get("kind", "function")),
            file_path=str(data.get("file_path", "")),
            signature=str(data.get("signature", "")),
            description=str(data.get("description", "")),
            verified=bool(data.get("verified", False)),
            verification_source=str(data.get("verification_source", "")),
        )


@dataclass
class TestExecutionRecord:
    __test__ = False
    test_id: str
    command: str
    status: Literal["passed", "failed", "timeout", "inconclusive", "skipped"]
    exit_code: int
    duration_seconds: float = 0.0
    stdout_summary: str = ""
    stderr_summary: str = ""
    synthesized: bool = True
    exercised_symbols: list[str] = field(default_factory=list)
    timestamp: datetime.datetime = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    failure_classification: str = ""

    def __post_init__(self):
        if len(self.stdout_summary) > 500:
            self.stdout_summary = self.stdout_summary[:497] + "..."
        if len(self.stderr_summary) > 500:
            self.stderr_summary = self.stderr_summary[:497] + "..."
        if len(self.failure_classification) > 200:
            self.failure_classification = self.failure_classification[:197] + "..."
        if len(self.exercised_symbols) > 20:
            self.exercised_symbols = self.exercised_symbols[:20]

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "command": self.command,
            "status": self.status,
            "exit_code": self.exit_code,
            "duration_seconds": self.duration_seconds,
            "stdout_summary": self.stdout_summary,
            "stderr_summary": self.stderr_summary,
            "synthesized": self.synthesized,
            "exercised_symbols": list(self.exercised_symbols),
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime.datetime) else str(self.timestamp),
            "failure_classification": self.failure_classification,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        raw_ts = data.get("timestamp")
        if isinstance(raw_ts, str):
            try:
                ts = datetime.datetime.fromisoformat(raw_ts)
            except ValueError:
                ts = datetime.datetime.now(datetime.timezone.utc)
        elif isinstance(raw_ts, datetime.datetime):
            ts = raw_ts
        else:
            ts = datetime.datetime.now(datetime.timezone.utc)
        return cls(
            test_id=str(data.get("test_id", "")),
            command=str(data.get("command", "")),
            status=data.get("status", "inconclusive"),
            exit_code=int(data.get("exit_code", 0)),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            stdout_summary=str(data.get("stdout_summary", "")),
            stderr_summary=str(data.get("stderr_summary", "")),
            synthesized=bool(data.get("synthesized", True)),
            exercised_symbols=list(data.get("exercised_symbols", [])),
            timestamp=ts,
            failure_classification=str(data.get("failure_classification", "")),
        )


@dataclass
class VerificationGap:
    missing_test_symbols: list[ExportedSymbol] = field(default_factory=list)
    untested_files: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    severity: Literal["low", "medium", "high"] = "low"

    def __post_init__(self):
        if len(self.missing_test_symbols) > 20:
            self.missing_test_symbols = self.missing_test_symbols[:20]
        if len(self.untested_files) > 20:
            self.untested_files = self.untested_files[:20]
        if len(self.reasons) > 20:
            self.reasons = self.reasons[:20]

    def to_dict(self) -> dict[str, Any]:
        return {
            "missing_test_symbols": [s.to_dict() for s in self.missing_test_symbols],
            "untested_files": list(self.untested_files),
            "reasons": list(self.reasons),
            "severity": self.severity,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        raw_syms = data.get("missing_test_symbols", [])
        syms = [
            ExportedSymbol.from_dict(s) if isinstance(s, dict) else s
            for s in raw_syms
        ] if isinstance(raw_syms, list) else []
        return cls(
            missing_test_symbols=syms,
            untested_files=list(data.get("untested_files", [])),
            reasons=list(data.get("reasons", [])),
            severity=data.get("severity", "low"),
        )


@dataclass
class SubtaskContract:
    subtask_id: str
    title: str
    exported_symbols: list[ExportedSymbol] = field(default_factory=list)
    modified_files: list[str] = field(default_factory=list)
    created_files: list[str] = field(default_factory=list)
    validation_commands: list[str] = field(default_factory=list)
    architectural_notes: list[str] = field(default_factory=list)
    behavioral_evidence: list[TestExecutionRecord] = field(default_factory=list)
    created_at: datetime.datetime = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))

    def __post_init__(self):
        if len(self.exported_symbols) > 10:
            self.exported_symbols = self.exported_symbols[:10]
        if len(self.modified_files) > 20:
            self.modified_files = self.modified_files[:20]
        if len(self.created_files) > 20:
            self.created_files = self.created_files[:20]
        if len(self.validation_commands) > 10:
            self.validation_commands = self.validation_commands[:10]
        if len(self.architectural_notes) > 10:
            self.architectural_notes = self.architectural_notes[:10]
        if len(self.behavioral_evidence) > 10:
            self.behavioral_evidence = self.behavioral_evidence[:10]

    def format_for_prompt(self, max_chars: int = 2000) -> str:
        lines = [f"### Subtask Contract: '{self.title}' ({self.subtask_id})"]
        if self.created_files:
            lines.append(f"- Created Files: {', '.join(self.created_files)}")
        if self.modified_files:
            lines.append(f"- Modified Files: {', '.join(self.modified_files)}")
        if self.exported_symbols:
            lines.append("- Exported Interfaces & Symbols:")
            for sym in self.exported_symbols:
                sig = sym.signature or sym.name
                desc = f" ({sym.description})" if sym.description else ""
                status_tag = " [VERIFIED]" if sym.verified else " [UNVERIFIED]"
                lines.append(f"  * [{sym.kind}] `{sig}` in `{sym.file_path}`{desc}{status_tag}")
        if self.behavioral_evidence:
            lines.append("- Behavioral Verification Evidence:")
            for rec in self.behavioral_evidence:
                lines.append(f"  * [{rec.status.upper()}] `{rec.command}` (exit {rec.exit_code})")
        if self.validation_commands:
            lines.append(f"- Verified Validation Commands: {'; '.join(self.validation_commands)}")
        if self.architectural_notes:
            lines.append("- Architectural Invariants:")
            for note in self.architectural_notes:
                lines.append(f"  * {note}")
        formatted = "\n".join(lines)
        if len(formatted) > max_chars:
            formatted = formatted[:max_chars - 3] + "..."
        return formatted

    def to_dict(self) -> dict[str, Any]:
        return {
            "subtask_id": self.subtask_id,
            "title": self.title,
            "exported_symbols": [s.to_dict() for s in self.exported_symbols],
            "modified_files": list(self.modified_files),
            "created_files": list(self.created_files),
            "validation_commands": list(self.validation_commands),
            "architectural_notes": list(self.architectural_notes),
            "behavioral_evidence": [r.to_dict() for r in self.behavioral_evidence],
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        raw_ts = data.get("created_at")
        if isinstance(raw_ts, str):
            try:
                ts = datetime.datetime.fromisoformat(raw_ts)
            except ValueError:
                ts = datetime.datetime.now(datetime.timezone.utc)
        elif isinstance(raw_ts, datetime.datetime):
            ts = raw_ts
        else:
            ts = datetime.datetime.now(datetime.timezone.utc)
        raw_symbols = data.get("exported_symbols", [])
        symbols = [
            ExportedSymbol.from_dict(s) if isinstance(s, dict) else s
            for s in raw_symbols
        ] if isinstance(raw_symbols, list) else []
        raw_evidence = data.get("behavioral_evidence", [])
        evidence = [
            TestExecutionRecord.from_dict(r) if isinstance(r, dict) else r
            for r in raw_evidence
        ] if isinstance(raw_evidence, list) else []
        return cls(
            subtask_id=str(data.get("subtask_id", "")),
            title=str(data.get("title", "")),
            exported_symbols=symbols,
            modified_files=list(data.get("modified_files", [])),
            created_files=list(data.get("created_files", [])),
            validation_commands=list(data.get("validation_commands", [])),
            architectural_notes=list(data.get("architectural_notes", [])),
            behavioral_evidence=evidence,
            created_at=ts,
        )


# Phase 4.14: Worktree Session & Parallel DAG Execution Models
@dataclass
class WorktreeSession:
    session_id: str
    subtask_id: str
    worktree_path: str
    branch_name: str
    base_commit: str
    created_at: datetime.datetime = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    status: Literal["active", "merged", "failed", "cleaned", "abandoned"] = "active"

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "subtask_id": self.subtask_id,
            "worktree_path": self.worktree_path,
            "branch_name": self.branch_name,
            "base_commit": self.base_commit,
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime.datetime) else str(self.created_at),
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        if not isinstance(data, dict):
            return cls(session_id="", subtask_id="", worktree_path="", branch_name="", base_commit="")
        raw_ts = data.get("created_at")
        if isinstance(raw_ts, str):
            try:
                created_at = datetime.datetime.fromisoformat(raw_ts)
            except Exception:
                created_at = datetime.datetime.now(datetime.timezone.utc)
        elif isinstance(raw_ts, datetime.datetime):
            created_at = raw_ts
        else:
            created_at = datetime.datetime.now(datetime.timezone.utc)
        return cls(
            session_id=str(data.get("session_id", "")),
            subtask_id=str(data.get("subtask_id", "")),
            worktree_path=str(data.get("worktree_path", "")),
            branch_name=str(data.get("branch_name", "")),
            base_commit=str(data.get("base_commit", "")),
            created_at=created_at,
            status=data.get("status", "active"),
        )


class ImplementationTerminationReason:
    """Canonical termination reasons produced by the interactive implementation agent.

    Values mirror ToolEngine/ToolExecutionPolicy reasons so downstream repair machinery
    can key off a single vocabulary regardless of which path produced the result.
    """
    COMPLETED = "completed"
    SINGLE_SHOT_FALLBACK = "single_shot_fallback"
    MAX_STEPS_EXCEEDED = "max_steps_exceeded"
    BUDGET_EXHAUSTED = "budget_exhausted"
    CONSECUTIVE_REPEATS_EXCEEDED = "consecutive_repeats_exceeded"
    INVALID_PROVIDER_RESPONSE = "invalid_provider_response"
    NO_OPERATIONS = "no_operations"
    SCOPE_VIOLATION = "scope_violation"
    PROVIDER_ERROR = "provider_error"
    ERROR = "error"

    # --- Phase 4.16: prospective (candidate tree) validation outcomes ---
    #: The proposed edits were applied to an isolated candidate tree and real
    #: validation commands executed against that tree succeeded.
    CANDIDATE_VALIDATION_PASSED = "candidate_validation_passed"
    #: Real validation against the candidate tree failed and no refinement
    #: budget remained.
    CANDIDATE_VALIDATION_FAILED = "candidate_validation_failed"
    #: The candidate refinement budget was consumed without a passing candidate.
    CANDIDATE_BUDGET_EXHAUSTED = "candidate_budget_exhausted"
    #: The proposed operations could not be applied to the candidate tree at all
    #: (scope violation, protected path, traversal, unusable patch).
    CANDIDATE_INVALID_OPERATIONS = "candidate_invalid_operations"
    #: The candidate tree itself could not be materialised.
    CANDIDATE_SETUP_FAILED = "candidate_setup_failed"

    #: Reasons that indicate the loop stopped without usable file operations.
    FAILURE_REASONS = frozenset({
        MAX_STEPS_EXCEEDED,
        BUDGET_EXHAUSTED,
        CONSECUTIVE_REPEATS_EXCEEDED,
        INVALID_PROVIDER_RESPONSE,
        NO_OPERATIONS,
        PROVIDER_ERROR,
        ERROR,
        CANDIDATE_VALIDATION_FAILED,
        CANDIDATE_BUDGET_EXHAUSTED,
        CANDIDATE_INVALID_OPERATIONS,
        CANDIDATE_SETUP_FAILED,
    })

    #: Coarse failure categories consumable by orchestrator/repair machinery.
    CATEGORIES = {
        MAX_STEPS_EXCEEDED: "budget_exhaustion",
        BUDGET_EXHAUSTED: "budget_exhaustion",
        CONSECUTIVE_REPEATS_EXCEEDED: "loop_detected",
        INVALID_PROVIDER_RESPONSE: "provider_failure",
        PROVIDER_ERROR: "provider_failure",
        NO_OPERATIONS: "incomplete_implementation",
        SCOPE_VIOLATION: "scope_violation",
        ERROR: "internal_error",
        COMPLETED: "none",
        SINGLE_SHOT_FALLBACK: "none",
        CANDIDATE_VALIDATION_PASSED: "none",
        CANDIDATE_VALIDATION_FAILED: "candidate_validation_failure",
        CANDIDATE_BUDGET_EXHAUSTED: "budget_exhaustion",
        CANDIDATE_INVALID_OPERATIONS: "invalid_operations",
        CANDIDATE_SETUP_FAILED: "internal_error",
    }

    @classmethod
    def categorize(cls, reason: str | None) -> str:
        return cls.CATEGORIES.get(reason or "", "unknown")


@dataclass
class ImplementationResult:
    """Structured result returned by the interactive implementation agent."""
    success: bool = False
    file_operations: list[FileOperation] | None = None
    summary: str = ""
    files_inspected: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    tool_steps_used: int = 0
    elapsed_time_seconds: float = 0.0
    provider: str | None = None
    model: str | None = None
    termination_reason: str | None = None
    used_fallback: bool = False
    scope_violations: list[str] = field(default_factory=list)
    tool_history: list[dict[str, Any]] = field(default_factory=list)
    metrics: ToolExecutionMetrics | None = None
    error_message: str | None = None
    # --- Phase 4.15 implementation-level telemetry ---
    #: Number of tool calls that returned is_error=True (bad args, denied path, failed probe).
    tool_call_failures: int = 0
    #: Number of successful tool calls.
    tool_call_successes: int = 0
    #: Number of run_command_sandbox probe invocations (targeted validation attempts).
    validation_attempts: int = 0
    #: Number of probe invocations that failed (non-zero exit).
    validation_failures: int = 0
    #: Number of times the loop recovered after an errored tool result.
    recovery_attempts: int = 0
    #: Circuit breaker trips observed during the loop.
    circuit_breaker_events: int = 0
    #: Coarse failure category derived from termination_reason.
    failure_category: str = "none"
    # --- Phase 4.16 prospective (candidate tree) validation telemetry ---
    #: Whether the candidate sandbox loop ran at all for this implementation.
    prospective_validation_used: bool = False
    #: Number of candidate trees built (BASE + operations) during the loop.
    candidate_iterations: int = 0
    #: Number of times real validation was executed against a candidate tree.
    candidate_validation_attempts: int = 0
    #: Candidate validations where every executed command exited zero.
    candidate_validation_successes: int = 0
    #: Candidate validations where at least one executed command failed.
    candidate_validation_failures: int = 0
    #: Refinement rounds triggered specifically by a failed candidate validation.
    candidate_recovery_attempts: int = 0
    #: Candidate-relative paths mutated by the final candidate build.
    candidate_files_changed: list[str] = field(default_factory=list)
    #: Wall-clock seconds spent materialising, rebuilding and validating candidates.
    candidate_elapsed_seconds: float = 0.0
    #: Candidate trees that could not be fully removed (Windows file locks etc.).
    candidate_cleanup_failures: int = 0
    #: Total real validation commands actually executed against candidate trees.
    validation_commands_run: int = 0
    #: Wall-clock seconds spent inside real candidate validation commands.
    validation_runtime_seconds: float = 0.0
    #: Whether the final candidate passed real validation.
    final_candidate_success: bool = False
    #: Deterministic, resumable description of the last candidate (no filesystem
    #: snapshot): base root, operation digest, iteration index, validation state.
    candidate_descriptor: dict[str, Any] = field(default_factory=dict)
    #: Structured report of the last candidate validation run.
    candidate_validation_report: dict[str, Any] | None = None
    # --- Phase 4.17 semantic change-impact + evidence telemetry ---
    #: Whether semantic (graph-based) impact analysis ran for this implementation.
    semantic_impact_used: bool = False
    #: Impact confidence of the final candidate analysis ("high"/"medium"/"low").
    impact_confidence: str = ""
    #: Adaptive validation scope the analysis recommended.
    impact_recommended_scope: str = ""
    #: Distinct symbols the change added, removed or modified.
    impact_changed_symbols: int = 0
    #: Modules reachable from the change through reverse dependencies.
    impact_affected_symbols: int = 0
    #: Test files the analyzer examined for an association.
    impact_tests_considered: int = 0
    #: Validation targets actually selected, after ranking and bounding.
    impact_tests_selected: int = 0
    #: Targets selected from graph evidence rather than filename heuristics.
    impact_semantic_targets: int = 0
    #: Wall-clock seconds spent building the graph and analysing impact.
    impact_analysis_seconds: float = 0.0
    #: Compact serialised ChangeImpactReport for the final candidate.
    impact_report: dict[str, Any] | None = None
    #: Post-apply validations satisfied by reusing candidate evidence.
    validation_evidence_reused: int = 0
    #: Reuse requests refused because an assumption no longer held.
    validation_evidence_invalidated: int = 0
    #: Command runtime avoided by evidence reuse.
    validation_time_saved_seconds: float = 0.0
    #: Bounded evidence history across candidate rebuild/revalidate cycles.
    validation_evidence: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_recoverable_failure(self) -> bool:
        """Whether the orchestrator's repair machinery could plausibly retry this."""
        return (
            not self.success
            and self.termination_reason in ImplementationTerminationReason.FAILURE_REASONS
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "file_operations": [op.__dict__ if hasattr(op, "__dict__") else op for op in (self.file_operations or [])] if self.file_operations is not None else None,
            "summary": self.summary,
            "files_inspected": list(self.files_inspected),
            "files_modified": list(self.files_modified),
            "tool_steps_used": self.tool_steps_used,
            "elapsed_time_seconds": self.elapsed_time_seconds,
            "provider": self.provider,
            "model": self.model,
            "termination_reason": self.termination_reason,
            "used_fallback": self.used_fallback,
            "scope_violations": list(self.scope_violations),
            "tool_history": self.tool_history,
            "metrics": self.metrics.to_dict() if hasattr(self.metrics, "to_dict") else self.metrics,
            "error_message": self.error_message,
            "tool_call_failures": self.tool_call_failures,
            "tool_call_successes": self.tool_call_successes,
            "validation_attempts": self.validation_attempts,
            "validation_failures": self.validation_failures,
            "recovery_attempts": self.recovery_attempts,
            "circuit_breaker_events": self.circuit_breaker_events,
            "failure_category": self.failure_category,
            "prospective_validation_used": self.prospective_validation_used,
            "candidate_iterations": self.candidate_iterations,
            "candidate_validation_attempts": self.candidate_validation_attempts,
            "candidate_validation_successes": self.candidate_validation_successes,
            "candidate_validation_failures": self.candidate_validation_failures,
            "candidate_recovery_attempts": self.candidate_recovery_attempts,
            "candidate_files_changed": list(self.candidate_files_changed),
            "candidate_elapsed_seconds": self.candidate_elapsed_seconds,
            "candidate_cleanup_failures": self.candidate_cleanup_failures,
            "validation_commands_run": self.validation_commands_run,
            "validation_runtime_seconds": self.validation_runtime_seconds,
            "final_candidate_success": self.final_candidate_success,
            "candidate_descriptor": dict(self.candidate_descriptor),
            "candidate_validation_report": self.candidate_validation_report,
            "semantic_impact_used": self.semantic_impact_used,
            "impact_confidence": self.impact_confidence,
            "impact_recommended_scope": self.impact_recommended_scope,
            "impact_changed_symbols": self.impact_changed_symbols,
            "impact_affected_symbols": self.impact_affected_symbols,
            "impact_tests_considered": self.impact_tests_considered,
            "impact_tests_selected": self.impact_tests_selected,
            "impact_semantic_targets": self.impact_semantic_targets,
            "impact_analysis_seconds": round(self.impact_analysis_seconds, 4),
            "impact_report": self.impact_report,
            "validation_evidence_reused": self.validation_evidence_reused,
            "validation_evidence_invalidated": self.validation_evidence_invalidated,
            "validation_time_saved_seconds": round(self.validation_time_saved_seconds, 4),
            "validation_evidence": list(self.validation_evidence),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        if not isinstance(data, dict):
            return cls()
        file_ops_data = data.get("file_operations")
        file_ops = [FileOperation(**op) if isinstance(op, dict) else op for op in file_ops_data] if file_ops_data is not None else None
        metrics_data = data.get("metrics")
        metrics = ToolExecutionMetrics.from_dict(metrics_data) if isinstance(metrics_data, dict) else (metrics_data if isinstance(metrics_data, ToolExecutionMetrics) else None)
        return cls(
            success=bool(data.get("success", False)),
            file_operations=file_ops,
            summary=str(data.get("summary", "")),
            files_inspected=list(data.get("files_inspected", [])),
            files_modified=list(data.get("files_modified", [])),
            tool_steps_used=int(data.get("tool_steps_used", 0)),
            elapsed_time_seconds=float(data.get("elapsed_time_seconds", 0.0)),
            provider=data.get("provider"),
            model=data.get("model"),
            termination_reason=data.get("termination_reason"),
            used_fallback=bool(data.get("used_fallback", False)),
            scope_violations=list(data.get("scope_violations", [])),
            tool_history=list(data.get("tool_history", [])),
            metrics=metrics,
            error_message=data.get("error_message"),
            tool_call_failures=int(data.get("tool_call_failures", 0)),
            tool_call_successes=int(data.get("tool_call_successes", 0)),
            validation_attempts=int(data.get("validation_attempts", 0)),
            validation_failures=int(data.get("validation_failures", 0)),
            recovery_attempts=int(data.get("recovery_attempts", 0)),
            circuit_breaker_events=int(data.get("circuit_breaker_events", 0)),
            failure_category=str(data.get("failure_category", "none")),
            prospective_validation_used=bool(data.get("prospective_validation_used", False)),
            candidate_iterations=int(data.get("candidate_iterations", 0)),
            candidate_validation_attempts=int(data.get("candidate_validation_attempts", 0)),
            candidate_validation_successes=int(data.get("candidate_validation_successes", 0)),
            candidate_validation_failures=int(data.get("candidate_validation_failures", 0)),
            candidate_recovery_attempts=int(data.get("candidate_recovery_attempts", 0)),
            candidate_files_changed=list(data.get("candidate_files_changed", [])),
            candidate_elapsed_seconds=float(data.get("candidate_elapsed_seconds", 0.0)),
            candidate_cleanup_failures=int(data.get("candidate_cleanup_failures", 0)),
            validation_commands_run=int(data.get("validation_commands_run", 0)),
            validation_runtime_seconds=float(data.get("validation_runtime_seconds", 0.0)),
            final_candidate_success=bool(data.get("final_candidate_success", False)),
            candidate_descriptor=dict(data.get("candidate_descriptor") or {}),
            candidate_validation_report=data.get("candidate_validation_report"),
            semantic_impact_used=bool(data.get("semantic_impact_used", False)),
            impact_confidence=str(data.get("impact_confidence", "")),
            impact_recommended_scope=str(data.get("impact_recommended_scope", "")),
            impact_changed_symbols=int(data.get("impact_changed_symbols", 0) or 0),
            impact_affected_symbols=int(data.get("impact_affected_symbols", 0) or 0),
            impact_tests_considered=int(data.get("impact_tests_considered", 0) or 0),
            impact_tests_selected=int(data.get("impact_tests_selected", 0) or 0),
            impact_semantic_targets=int(data.get("impact_semantic_targets", 0) or 0),
            impact_analysis_seconds=float(data.get("impact_analysis_seconds", 0.0) or 0.0),
            impact_report=data.get("impact_report"),
            validation_evidence_reused=int(data.get("validation_evidence_reused", 0) or 0),
            validation_evidence_invalidated=int(data.get("validation_evidence_invalidated", 0) or 0),
            validation_time_saved_seconds=float(data.get("validation_time_saved_seconds", 0.0) or 0.0),
            validation_evidence=list(data.get("validation_evidence") or []),
        )


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
    contract: SubtaskContract | None = None # Added for Phase 4.10
    worktree_session: WorktreeSession | None = None # Added for Phase 4.14
    integration_commit: str | None = None # Added for Phase 4.14

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
        if self.contract:
            data["contract"] = self.contract.to_dict()
        if self.worktree_session:
            data["worktree_session"] = self.worktree_session.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        data = dict(data)
        if "status" in data:
            raw_status = data["status"]
            if isinstance(raw_status, SubtaskStatus):
                data["status"] = raw_status
            elif isinstance(raw_status, str):
                try:
                    data["status"] = SubtaskStatus(raw_status.lower())
                except ValueError:
                    data["status"] = SubtaskStatus(raw_status)
        else:
            data["status"] = SubtaskStatus.PENDING
        if data.get("created_at"):
            data["created_at"] = datetime.datetime.fromisoformat(data["created_at"]) if isinstance(data["created_at"], str) else data["created_at"]
        if data.get("updated_at"):
            data["updated_at"] = datetime.datetime.fromisoformat(data["updated_at"]) if isinstance(data["updated_at"], str) else data["updated_at"]
        if data.get("started_at"):
            data["started_at"] = datetime.datetime.fromisoformat(data["started_at"]) if isinstance(data["started_at"], str) else data["started_at"]
        if data.get("completed_at"):
            data["completed_at"] = datetime.datetime.fromisoformat(data["completed_at"]) if isinstance(data["completed_at"], str) else data["completed_at"]
        if data.get("contract"):
            if isinstance(data["contract"], dict):
                data["contract"] = SubtaskContract.from_dict(data["contract"])
        else:
            data["contract"] = None
        if data.get("worktree_session"):
            if isinstance(data["worktree_session"], dict):
                data["worktree_session"] = WorktreeSession.from_dict(data["worktree_session"])
        else:
            data["worktree_session"] = None
        data.setdefault("integration_commit", None)
        return cls(**data)


# Phase 4.9: Typed DAG Restructuring & Evolution Models
@dataclass
class SubtaskAddition:
    subtask: Subtask

    def to_dict(self) -> dict[str, Any]:
        return {"subtask": self.subtask.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(subtask=Subtask.from_dict(data["subtask"]) if isinstance(data.get("subtask"), dict) else data["subtask"])


@dataclass
class SubtaskRemoval:
    subtask_id: str
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"subtask_id": self.subtask_id, "reason": self.reason}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(subtask_id=str(data.get("subtask_id", "")), reason=str(data.get("reason", "")))


@dataclass
class DependencyUpdate:
    subtask_id: str
    dependencies: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"subtask_id": self.subtask_id, "dependencies": list(self.dependencies)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(subtask_id=str(data.get("subtask_id", "")), dependencies=list(data.get("dependencies", [])))


@dataclass
class SubtaskInvalidation:
    subtask_id: str
    reason: str = ""
    replacement_subtask: Subtask | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "subtask_id": self.subtask_id,
            "reason": self.reason,
            "replacement_subtask": self.replacement_subtask.to_dict() if self.replacement_subtask else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        rep = data.get("replacement_subtask")
        return cls(
            subtask_id=str(data.get("subtask_id", "")),
            reason=str(data.get("reason", "")),
            replacement_subtask=Subtask.from_dict(rep) if isinstance(rep, dict) else None,
        )


# Phase 3.14: Typed Plan Modification Models (Retained for backward compatibility)
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
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(subtask=Subtask.from_dict(data["subtask"]) if isinstance(data.get("subtask"), dict) else data["subtask"])


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
class DAGProposal:
    reason: str = ""
    additions: list[SubtaskAddition] = field(default_factory=list)
    removals: list[SubtaskRemoval] = field(default_factory=list)
    dependency_updates: list[DependencyUpdate] = field(default_factory=list)
    invalidations: list[SubtaskInvalidation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "additions": [a.to_dict() for a in self.additions],
            "removals": [r.to_dict() for r in self.removals],
            "dependency_updates": [d.to_dict() for d in self.dependency_updates],
            "invalidations": [i.to_dict() for i in self.invalidations],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        if not isinstance(data, dict):
            return cls()
        return cls(
            reason=str(data.get("reason", "")),
            additions=[SubtaskAddition.from_dict(a) if isinstance(a, dict) else a for a in data.get("additions", [])],
            removals=[SubtaskRemoval.from_dict(r) if isinstance(r, dict) else r for r in data.get("removals", [])],
            dependency_updates=[DependencyUpdate.from_dict(d) if isinstance(d, dict) else d for d in data.get("dependency_updates", [])],
            invalidations=[SubtaskInvalidation.from_dict(i) if isinstance(i, dict) else i for i in data.get("invalidations", [])],
        )

    @classmethod
    def from_plan_proposal(cls, prop: PlanProposal) -> Self:
        additions = [SubtaskAddition(subtask=a.subtask) for a in prop.additions]
        dependency_updates = []
        for mod in prop.modifications:
            if mod.dependencies is not None:
                dependency_updates.append(DependencyUpdate(subtask_id=mod.subtask_id, dependencies=mod.dependencies))
        return cls(
            reason=prop.reason,
            additions=additions,
            dependency_updates=dependency_updates,
        )


@dataclass(frozen=True)
class TaskPlanAmendment:
    amendment_id: str
    version: int
    timestamp: datetime.datetime
    proposal: DAGProposal
    approved_by: Literal["deterministic_policy", "user_approval"] = "deterministic_policy"
    previous_active_subtask_ids: list[str] = field(default_factory=list)
    new_active_subtask_ids: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "amendment_id": self.amendment_id,
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
            "proposal": self.proposal.to_dict(),
            "approved_by": self.approved_by,
            "previous_active_subtask_ids": list(self.previous_active_subtask_ids),
            "new_active_subtask_ids": list(self.new_active_subtask_ids),
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        raw_ts = data.get("timestamp")
        if isinstance(raw_ts, str):
            ts = datetime.datetime.fromisoformat(raw_ts)
        elif isinstance(raw_ts, datetime.datetime):
            ts = raw_ts
        else:
            ts = datetime.datetime.now(datetime.timezone.utc)
        raw_prop = data.get("proposal", {})
        prop = DAGProposal.from_dict(raw_prop) if isinstance(raw_prop, dict) else raw_prop
        return cls(
            amendment_id=str(data.get("amendment_id", "")),
            version=int(data.get("version", 1)),
            timestamp=ts,
            proposal=prop,
            approved_by=data.get("approved_by", "deterministic_policy"),
            previous_active_subtask_ids=list(data.get("previous_active_subtask_ids", [])),
            new_active_subtask_ids=list(data.get("new_active_subtask_ids", [])),
            reason=str(data.get("reason", "")),
        )


@dataclass
class TaskPlan:
    objective: str
    subtasks: list[Subtask]
    risks: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    version: int = 1
    amendments: list[TaskPlanAmendment] = field(default_factory=list)

    @property
    def active_subtasks(self) -> list[Subtask]:
        return [s for s in self.subtasks if s.status not in {SubtaskStatus.SUPERSEDED, SubtaskStatus.PRUNED}]

    @property
    def active_subtask_ids(self) -> list[str]:
        return [s.subtask_id for s in self.active_subtasks]

    def get_upstream_contracts(self, subtask_id: str) -> list[SubtaskContract]:
        """
        Resolves SubtaskContract records from direct, active COMPLETED dependencies.
        Never returns contracts from SUPERSEDED, PRUNED, FAILED, RUNNING, or PENDING nodes.
        Preserves deterministic dependency order.
        """
        active_map = {s.subtask_id: s for s in self.active_subtasks}
        target = active_map.get(subtask_id)
        if not target or not target.dependencies:
            return []

        contracts: list[SubtaskContract] = []
        for dep_id in target.dependencies:
            dep_sub = active_map.get(dep_id)
            if dep_sub and dep_sub.status == SubtaskStatus.COMPLETED and dep_sub.contract is not None:
                contracts.append(dep_sub.contract)
        return contracts

    def apply_amendment(
        self,
        proposal: DAGProposal | PlanProposal,
        approved_by: Literal["deterministic_policy", "user_approval"] = "deterministic_policy",
    ) -> TaskPlanAmendment:
        raw_plan_proposal = proposal if isinstance(proposal, PlanProposal) else None
        if isinstance(proposal, PlanProposal):
            proposal = DAGProposal.from_plan_proposal(proposal)

        prev_active_ids = self.active_subtask_ids
        subtask_map = {s.subtask_id: s for s in self.subtasks}

        # 0. If original was PlanProposal, apply field modifications
        if raw_plan_proposal and raw_plan_proposal.modifications:
            for mod in raw_plan_proposal.modifications:
                if mod.subtask_id in subtask_map:
                    target = subtask_map[mod.subtask_id]
                    if mod.title is not None:
                        target.title = mod.title
                    if mod.goal is not None:
                        target.goal = mod.goal
                    if mod.acceptance_criteria is not None:
                        target.acceptance_criteria = list(mod.acceptance_criteria)
                    if mod.dependencies is not None:
                        target.dependencies = list(mod.dependencies)
                    target.updated_at = datetime.datetime.now(datetime.timezone.utc)

        # 1. Process invalidations (mark original as SUPERSEDED, insert replacement if given)
        for inv in proposal.invalidations:
            if inv.subtask_id in subtask_map:
                target = subtask_map[inv.subtask_id]
                target.status = SubtaskStatus.SUPERSEDED
                target.updated_at = datetime.datetime.now(datetime.timezone.utc)
                if inv.replacement_subtask:
                    if inv.replacement_subtask.subtask_id not in subtask_map:
                        self.subtasks.append(inv.replacement_subtask)
                        subtask_map[inv.replacement_subtask.subtask_id] = inv.replacement_subtask

        # 2. Process removals / pruning
        for rem in proposal.removals:
            if rem.subtask_id in subtask_map:
                target = subtask_map[rem.subtask_id]
                target.status = SubtaskStatus.PRUNED
                target.updated_at = datetime.datetime.now(datetime.timezone.utc)

        # 3. Process dependency updates
        for dep_up in proposal.dependency_updates:
            if dep_up.subtask_id in subtask_map:
                subtask_map[dep_up.subtask_id].dependencies = list(dep_up.dependencies)
                subtask_map[dep_up.subtask_id].updated_at = datetime.datetime.now(datetime.timezone.utc)

        # 4. Process additions
        for add in proposal.additions:
            if add.subtask.subtask_id not in subtask_map:
                self.subtasks.append(add.subtask)
                subtask_map[add.subtask.subtask_id] = add.subtask

        self.version += 1
        new_active_ids = self.active_subtask_ids
        amendment = TaskPlanAmendment(
            amendment_id=str(uuid.uuid4()),
            version=self.version,
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            proposal=proposal,
            approved_by=approved_by,
            previous_active_subtask_ids=prev_active_ids,
            new_active_subtask_ids=new_active_ids,
            reason=proposal.reason,
        )
        self.amendments.append(amendment)
        return amendment

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "subtasks": [s.to_dict() for s in self.subtasks],
            "risks": self.risks,
            "assumptions": self.assumptions,
            "version": self.version,
            "amendments": [a.to_dict() for a in self.amendments],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        raw_amendments = data.get("amendments", [])
        amendments = [
            TaskPlanAmendment.from_dict(a) if isinstance(a, dict) else a
            for a in raw_amendments
        ] if isinstance(raw_amendments, list) else []
        return cls(
            objective=data["objective"],
            subtasks=[Subtask.from_dict(s) for s in data.get("subtasks", [])],
            risks=data.get("risks", []),
            assumptions=data.get("assumptions", []),
            version=int(data.get("version", 1)),
            amendments=amendments,
        )


@dataclass
class DAGAmendmentGuard:
    """Deterministic guard enforcing safety, acyclicity, quotas, and limits on DAG amendments."""
    max_dag_amendments: int = 3
    max_subtask_additions: int = 5
    max_total_subtasks: int = 15
    max_subtask_invalidations_per_node: int = 2

    def evaluate(
        self,
        proposal: DAGProposal | PlanProposal,
        plan: TaskPlan,
        initial_subtask_count: int = 0,
    ) -> tuple[bool, str]:
        # 1. Amendment budget check
        if len(plan.amendments) >= self.max_dag_amendments:
            return False, f"Maximum DAG amendments limit ({self.max_dag_amendments}) reached"

        # 2. Check empty proposal & convert PlanProposal
        if isinstance(proposal, PlanProposal):
            if not proposal.additions and not proposal.modifications:
                return False, "Proposal contains no additions or modifications"
            subtask_ids = {s.subtask_id for s in plan.subtasks}
            for mod in proposal.modifications:
                if mod.subtask_id not in subtask_ids:
                    return False, f"Modification target subtask '{mod.subtask_id}' does not exist in plan"
            dag_prop = DAGProposal.from_plan_proposal(proposal)
            if not dag_prop.additions and not dag_prop.removals and not dag_prop.dependency_updates and not dag_prop.invalidations:
                return True, "Approved under deterministic DAG policy"
            proposal = dag_prop
        else:
            if not proposal.additions and not proposal.removals and not proposal.dependency_updates and not proposal.invalidations:
                return False, "DAG proposal contains no additions, removals, updates, or invalidations"

        # 3. Additions budget check
        total_prev_additions = sum(len(a.proposal.additions) for a in plan.amendments if hasattr(a.proposal, "additions"))
        if total_prev_additions + len(proposal.additions) > self.max_subtask_additions:
            return False, f"Maximum subtask additions limit ({self.max_subtask_additions}) exceeded"

        # 4. Total active subtasks limit check & clone state
        sim_subtasks: dict[str, Subtask] = {
            s.subtask_id: Subtask(
                subtask_id=s.subtask_id,
                title=s.title,
                goal=s.goal,
                status=s.status,
                dependencies=list(s.dependencies),
                acceptance_criteria=list(s.acceptance_criteria),
            )
            for s in plan.subtasks
        }

        # 5. Invalidation limits check & simulation
        invalidation_counts: dict[str, int] = {}
        for a in plan.amendments:
            if hasattr(a.proposal, "invalidations"):
                for inv in a.proposal.invalidations:
                    invalidation_counts[inv.subtask_id] = invalidation_counts.get(inv.subtask_id, 0) + 1

        for inv in proposal.invalidations:
            if inv.subtask_id not in sim_subtasks:
                return False, f"Invalidation target subtask '{inv.subtask_id}' does not exist in plan"
            curr_count = invalidation_counts.get(inv.subtask_id, 0) + 1
            if curr_count > self.max_subtask_invalidations_per_node:
                return False, f"Subtask '{inv.subtask_id}' exceeded max invalidations limit ({self.max_subtask_invalidations_per_node})"

            target = sim_subtasks[inv.subtask_id]
            target.status = SubtaskStatus.SUPERSEDED
            if inv.replacement_subtask:
                rep = inv.replacement_subtask
                if not rep.subtask_id or not rep.title.strip() or not rep.goal.strip():
                    return False, "Replacement subtask must have a valid non-empty ID, title, and goal"
                if rep.subtask_id in sim_subtasks and sim_subtasks[rep.subtask_id].status not in {SubtaskStatus.SUPERSEDED, SubtaskStatus.PRUNED}:
                    return False, f"Replacement subtask ID '{rep.subtask_id}' conflicts with existing active subtask"
                sim_subtasks[rep.subtask_id] = Subtask(
                    subtask_id=rep.subtask_id,
                    title=rep.title,
                    goal=rep.goal,
                    status=rep.status if rep.status != SubtaskStatus.SUPERSEDED else SubtaskStatus.PENDING,
                    dependencies=list(rep.dependencies),
                    acceptance_criteria=list(rep.acceptance_criteria),
                )

        # 6. Removal / Pruning check & simulation
        for rem in proposal.removals:
            if rem.subtask_id not in sim_subtasks:
                return False, f"Removal target subtask '{rem.subtask_id}' does not exist in plan"
            target = sim_subtasks[rem.subtask_id]
            if target.status == SubtaskStatus.RUNNING:
                return False, f"Cannot prune currently running subtask '{rem.subtask_id}'"
            target.status = SubtaskStatus.PRUNED

        # 7. Additions simulation
        for add in proposal.additions:
            sub = add.subtask
            if not sub.subtask_id or not sub.title.strip() or not sub.goal.strip():
                return False, "Added subtask must have a valid non-empty ID, title, and goal"
            if sub.subtask_id in sim_subtasks and sim_subtasks[sub.subtask_id].status not in {SubtaskStatus.SUPERSEDED, SubtaskStatus.PRUNED}:
                return False, f"Duplicate active subtask ID: '{sub.subtask_id}'"
            sim_subtasks[sub.subtask_id] = Subtask(
                subtask_id=sub.subtask_id,
                title=sub.title,
                goal=sub.goal,
                status=sub.status if sub.status != SubtaskStatus.SUPERSEDED else SubtaskStatus.PENDING,
                dependencies=list(sub.dependencies),
                acceptance_criteria=list(sub.acceptance_criteria),
            )

        # 8. Dependency updates simulation
        for dep_up in proposal.dependency_updates:
            if dep_up.subtask_id not in sim_subtasks:
                return False, f"Dependency update target '{dep_up.subtask_id}' does not exist in plan"
            if sim_subtasks[dep_up.subtask_id].status in {SubtaskStatus.SUPERSEDED, SubtaskStatus.PRUNED}:
                return False, f"Cannot update dependencies for superseded/pruned subtask '{dep_up.subtask_id}'"
            sim_subtasks[dep_up.subtask_id].dependencies = list(dep_up.dependencies)

        # 9. Verify candidate active subtasks
        active_sim = [s for s in sim_subtasks.values() if s.status not in {SubtaskStatus.SUPERSEDED, SubtaskStatus.PRUNED}]
        if not active_sim:
            return False, "DAG proposal would result in an empty active task plan"
        if len(active_sim) > self.max_total_subtasks:
            return False, f"Total active subtasks ({len(active_sim)}) would exceed max limit ({self.max_total_subtasks})"

        # 10. Dependency validity, self-dependencies, and acyclicity check
        active_sim_map = {s.subtask_id: s for s in active_sim}
        for sub in active_sim:
            for dep_id in sub.dependencies:
                if dep_id == sub.subtask_id:
                    return False, f"Subtask '{sub.subtask_id}' has a self-dependency"
                if dep_id not in active_sim_map:
                    return False, f"Subtask '{sub.subtask_id}' references non-existent or pruned dependency '{dep_id}'"

        # DFS Cycle detection on active graph
        visited: set[str] = set()
        path: set[str] = set()

        def _is_cyclic(node_id: str) -> bool:
            visited.add(node_id)
            path.add(node_id)
            for dep in active_sim_map[node_id].dependencies:
                if dep not in visited:
                    if _is_cyclic(dep):
                        return True
                elif dep in path:
                    return True
            path.remove(node_id)
            return False

        for node_id in active_sim_map:
            if node_id not in visited:
                if _is_cyclic(node_id):
                    return False, "Dependency cycle detected in proposed DAG amendment"

        return True, "Approved under deterministic DAG policy"

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_dag_amendments": self.max_dag_amendments,
            "max_subtask_additions": self.max_subtask_additions,
            "max_total_subtasks": self.max_total_subtasks,
            "max_subtask_invalidations_per_node": self.max_subtask_invalidations_per_node,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            max_dag_amendments=data.get("max_dag_amendments", 3),
            max_subtask_additions=data.get("max_subtask_additions", 5),
            max_total_subtasks=data.get("max_total_subtasks", 15),
            max_subtask_invalidations_per_node=data.get("max_subtask_invalidations_per_node", 2),
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
    # Phase 4.14: branch that parallel subtask branches are integrated onto.
    # Persisted so a resumed run rejoins the same integration branch instead of
    # merging onto whatever branch happens to be checked out.
    integration_branch: str | None = None

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
        data.setdefault("integration_branch", None)
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
class ReviewConsensusRecord:
    role: str = "review"
    primary_provider: str = ""
    primary_model: str = ""
    secondary_provider: str = ""
    secondary_model: str = ""
    primary_verdict: str = ""
    secondary_verdict: str = ""
    final_consensus_verdict: str = ""
    is_high_risk: bool = False
    high_risk_reason: str = ""
    timestamp: datetime.datetime = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "primary_provider": self.primary_provider,
            "primary_model": self.primary_model,
            "secondary_provider": self.secondary_provider,
            "secondary_model": self.secondary_model,
            "primary_verdict": self.primary_verdict,
            "secondary_verdict": self.secondary_verdict,
            "final_consensus_verdict": self.final_consensus_verdict,
            "is_high_risk": self.is_high_risk,
            "high_risk_reason": self.high_risk_reason,
            "timestamp": self.timestamp.isoformat() if hasattr(self.timestamp, "isoformat") else str(self.timestamp),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReviewConsensusRecord":
        ts_raw = data.get("timestamp")
        ts = datetime.datetime.now(datetime.timezone.utc)
        if isinstance(ts_raw, str):
            try:
                ts = datetime.datetime.fromisoformat(ts_raw)
            except Exception:
                pass
        elif isinstance(ts_raw, datetime.datetime):
            ts = ts_raw
        return cls(
            role=str(data.get("role", "review")),
            primary_provider=str(data.get("primary_provider", "")),
            primary_model=str(data.get("primary_model", "")),
            secondary_provider=str(data.get("secondary_provider", "")),
            secondary_model=str(data.get("secondary_model", "")),
            primary_verdict=str(data.get("primary_verdict", "")),
            secondary_verdict=str(data.get("secondary_verdict", "")),
            final_consensus_verdict=str(data.get("final_consensus_verdict", "")),
            is_high_risk=bool(data.get("is_high_risk", False)),
            high_risk_reason=str(data.get("high_risk_reason", "")),
            timestamp=ts,
        )


@dataclass
class ReviewResult:
    verdict: str
    summary: str
    findings: list[str] = field(default_factory=list)
    consensus_records: list[ReviewConsensusRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "summary": self.summary,
            "findings": list(self.findings),
            "consensus_records": [r.to_dict() if hasattr(r, "to_dict") else r for r in self.consensus_records],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReviewResult":
        raw_consensus = data.get("consensus_records", [])
        records = []
        if isinstance(raw_consensus, list):
            for r in raw_consensus:
                if isinstance(r, dict):
                    records.append(ReviewConsensusRecord.from_dict(r))
                elif isinstance(r, ReviewConsensusRecord):
                    records.append(r)
        return cls(
            verdict=str(data.get("verdict", "CHANGES_REQUIRED")),
            summary=str(data.get("summary", "")),
            findings=list(data.get("findings", [])),
            consensus_records=records,
        )


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
    amendments: list[PlanAmendment] = field(default_factory=list)
    dag_proposal: DAGProposal | None = None
    dag_amendments: list[TaskPlanAmendment] = field(default_factory=list)
    behavioral_evidence: list[TestExecutionRecord] = field(default_factory=list)
    verification_gap: VerificationGap | None = None
    specialist_routing_state: dict[str, Any] = field(default_factory=dict)
    review_consensus: list[ReviewConsensusRecord] = field(default_factory=list)
    implementation_result: ImplementationResult | None = None
    # Phase 4.17: serialised ChangeImpactReport for the applied change, and the
    # bounded validation-evidence ledger backing post-apply reuse decisions.
    semantic_impact: dict[str, Any] | None = None
    validation_evidence: list[dict[str, Any]] = field(default_factory=list)
    # Phase 4.19: id of this run's ValidationDecisionRecord in the (separately
    # persisted) telemetry store, empty when telemetry is disabled or no
    # semantic decision was made. RunReport itself is a transient, in-process
    # object with no to_dict/from_dict, so this adds no checkpoint/task
    # serialization surface.
    validation_decision_id: str = ""
    # Phase 4.19: the reuse verdicts _apply_evidence_reuse just computed, kept
    # here (rather than changing that method's return type) so the several
    # existing Phase 4.18 tests that call it directly and compare its return
    # value to a plain command list are unaffected.
    validation_reuse_attempts: list[Any] = field(default_factory=list, repr=False)
    # Phase 4.16: multi-turn interactive implementation execution report
    multi_turn_report: MultiTurnExecutionReport | None = None


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


# Phase 4.13: Persistent Codebase Knowledge Graph & Cross-Task Architectural Memory

@dataclass
class BehavioralAssertion:
    """A concrete runtime behavioral property proven by automated test execution."""
    assertion_id: str
    description: str
    test_command: str
    status: Literal["passed", "failed"]
    commit_sha: str | None = None
    verified_at: datetime.datetime = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "assertion_id": self.assertion_id,
            "description": self.description,
            "test_command": self.test_command,
            "status": self.status,
            "commit_sha": self.commit_sha,
            "verified_at": self.verified_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        d = dict(data)
        if isinstance(d.get("verified_at"), str):
            d["verified_at"] = datetime.datetime.fromisoformat(d["verified_at"])
        elif not d.get("verified_at"):
            d["verified_at"] = datetime.datetime.now(datetime.timezone.utc)
        return cls(**d)


@dataclass
class KnowledgeSymbolNode:
    """Persistent representation of an exported symbol with verified behavioral contracts."""
    symbol_id: str  # Canonical: "path/to/file.py::SymbolName"
    name: str
    kind: Literal["class", "function", "method", "type", "variable"] = "function"
    file_path: str = ""
    signature: str = ""
    docstring: str = ""
    content_hash: str = ""
    verified_behaviors: list[BehavioralAssertion] = field(default_factory=list)
    confidence: float = 1.0
    provenance: Literal["behavioral_test", "subtask_contract", "ast_scan", "ai_inferred"] = "ast_scan"
    last_verified_at: datetime.datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol_id": self.symbol_id,
            "name": self.name,
            "kind": self.kind,
            "file_path": self.file_path,
            "signature": self.signature,
            "docstring": self.docstring,
            "content_hash": self.content_hash,
            "verified_behaviors": [b.to_dict() for b in self.verified_behaviors],
            "confidence": self.confidence,
            "provenance": self.provenance,
            "last_verified_at": self.last_verified_at.isoformat() if self.last_verified_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        d = dict(data)
        d["verified_behaviors"] = [
            BehavioralAssertion.from_dict(b) if isinstance(b, dict) else b
            for b in d.get("verified_behaviors", [])
        ]
        if isinstance(d.get("last_verified_at"), str):
            d["last_verified_at"] = datetime.datetime.fromisoformat(d["last_verified_at"])
        return cls(**d)


@dataclass
class KnowledgeFileNode:
    """Persistent repository file node tracking invariants, symbols, and dependencies."""
    path: str
    content_hash: str = ""
    language: str = ""
    module_role: str = "module"
    exported_symbol_ids: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    dependents: list[str] = field(default_factory=list)
    validation_commands: list[str] = field(default_factory=list)
    risk_level: Literal["standard", "high_risk"] = "standard"
    last_modified_task_id: str | None = None
    last_modified_at: datetime.datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "content_hash": self.content_hash,
            "language": self.language,
            "module_role": self.module_role,
            "exported_symbol_ids": list(self.exported_symbol_ids),
            "dependencies": list(self.dependencies),
            "dependents": list(self.dependents),
            "validation_commands": list(self.validation_commands),
            "risk_level": self.risk_level,
            "last_modified_task_id": self.last_modified_task_id,
            "last_modified_at": self.last_modified_at.isoformat() if self.last_modified_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        d = dict(data)
        if isinstance(d.get("last_modified_at"), str):
            d["last_modified_at"] = datetime.datetime.fromisoformat(d["last_modified_at"])
        return cls(**d)


@dataclass
class ArchitecturalInvariant:
    """Repo-wide or module-wide invariant rule learned across tasks."""
    invariant_id: str
    scope: Literal["repository", "module", "file"] = "repository"
    target_path: str = "*"
    rule_text: str = ""
    enforcement_type: Literal["contract", "security", "dependency", "concurrency"] = "contract"
    source_task_id: str = ""
    confidence: float = 1.0
    created_at: datetime.datetime = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "invariant_id": self.invariant_id,
            "scope": self.scope,
            "target_path": self.target_path,
            "rule_text": self.rule_text,
            "enforcement_type": self.enforcement_type,
            "source_task_id": self.source_task_id,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        d = dict(data)
        if isinstance(d.get("created_at"), str):
            d["created_at"] = datetime.datetime.fromisoformat(d["created_at"])
        elif not d.get("created_at"):
            d["created_at"] = datetime.datetime.now(datetime.timezone.utc)
        return cls(**d)


@dataclass
class FailurePatternRecord:
    """Recurring failure pattern and its verified successful repair recipe."""
    pattern_id: str
    error_signature: str
    failing_command: str = ""
    root_cause_summary: str = ""
    successful_repair_summary: str = ""
    affected_files: list[str] = field(default_factory=list)
    occurrence_count: int = 1
    confidence: float = 0.8
    last_seen_at: datetime.datetime = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "error_signature": self.error_signature,
            "failing_command": self.failing_command,
            "root_cause_summary": self.root_cause_summary,
            "successful_repair_summary": self.successful_repair_summary,
            "affected_files": list(self.affected_files),
            "occurrence_count": self.occurrence_count,
            "confidence": self.confidence,
            "last_seen_at": self.last_seen_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        d = dict(data)
        if isinstance(d.get("last_seen_at"), str):
            d["last_seen_at"] = datetime.datetime.fromisoformat(d["last_seen_at"])
        elif not d.get("last_seen_at"):
            d["last_seen_at"] = datetime.datetime.now(datetime.timezone.utc)
        return cls(**d)


@dataclass
class RepositoryKnowledgeGraph:
    """Top-level persistent repository knowledge graph."""
    version: int = 1
    repo_id: str = ""
    updated_at: datetime.datetime = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    files: dict[str, KnowledgeFileNode] = field(default_factory=dict)
    symbols: dict[str, KnowledgeSymbolNode] = field(default_factory=dict)
    invariants: list[ArchitecturalInvariant] = field(default_factory=list)
    failure_patterns: list[FailurePatternRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "repo_id": self.repo_id,
            "updated_at": self.updated_at.isoformat(),
            "files": {path: node.to_dict() for path, node in self.files.items()},
            "symbols": {sym_id: node.to_dict() for sym_id, node in self.symbols.items()},
            "invariants": [inv.to_dict() for inv in self.invariants],
            "failure_patterns": [pat.to_dict() for pat in self.failure_patterns],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        d = dict(data)
        files = {
            path: KnowledgeFileNode.from_dict(fn) if isinstance(fn, dict) else fn
            for path, fn in d.get("files", {}).items()
        }
        symbols = {
            sym_id: KnowledgeSymbolNode.from_dict(sn) if isinstance(sn, dict) else sn
            for sym_id, sn in d.get("symbols", {}).items()
        }
        invariants = [
            ArchitecturalInvariant.from_dict(inv) if isinstance(inv, dict) else inv
            for inv in d.get("invariants", [])
        ]
        failure_patterns = [
            FailurePatternRecord.from_dict(pat) if isinstance(pat, dict) else pat
            for pat in d.get("failure_patterns", [])
        ]
        updated_at = d.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.datetime.fromisoformat(updated_at)
        elif not updated_at:
            updated_at = datetime.datetime.now(datetime.timezone.utc)

        return cls(
            version=d.get("version", 1),
            repo_id=d.get("repo_id", ""),
            updated_at=updated_at,
            files=files,
            symbols=symbols,
            invariants=invariants,
            failure_patterns=failure_patterns,
            metadata=d.get("metadata", {}),
        )


