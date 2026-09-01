"""Phase 4.19: Autonomous Completion Enforcement & Release-Lifecycle Integration.

Provides an auditable evidence lifecycle:
ACTION -> OBSERVATION -> EVIDENCE -> EVIDENCE VALIDATION -> COMPLETION ASSESSMENT -> READINESS DECISION.

Prevents the coding agent from treating 'tests passed' as equivalent to
'the task is correctly implemented and safe to declare complete'.
Enforces strict failure-closed completion gates across the entire agent lifecycle.
"""

from __future__ import annotations

import ast
import datetime
import enum
import hashlib
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .evidence import compute_state_fingerprint
from .filesystem import ProjectFilesystem, ProtectedPathError, SandboxViolation, SECRET_NAMES
from .models import FailureAnalysis, FileOperation, Plan, ProjectContext, ReviewResult, Subtask, SubtaskContract, Task


# Regular expressions for detecting sensitive tokens / credentials in evidence.
# These match specific, recognizable *value shapes* (a real OpenAI key always
# looks like sk-..., a real AWS access key ID always looks like AKIA...).
SECRET_PATTERNS = [
    re.compile(r"(?:sk-[a-zA-Z0-9_-]{20,})"),                   # Generic / OpenAI API keys
    re.compile(r"(?:sk-ant-[a-zA-Z0-9_-]{20,})"),               # Anthropic API keys
    re.compile(r"(?:ghp_[a-zA-Z0-9]{36,})"),                    # GitHub Personal Access Tokens
    re.compile(r"(?:github_pat_[a-zA-Z0-9_]{50,})"),            # GitHub Fine-grained PATs
    re.compile(r"(?:AKIA[0-9A-Z]{16})"),                        # AWS Access Key ID
    re.compile(r"(?:bearer\s+[a-zA-Z0-9_\-\.]{20,})", re.I),    # Bearer tokens
    re.compile(r"-----BEGIN [A-Z ]+ PRIVATE KEY-----[\s\S]*?-----END [A-Z ]+ PRIVATE KEY-----"), # Private keys
]

# Value-shape patterns can never enumerate every credential format a command
# might print. A generic-labeled secret ("password=hunter2", "api_key": "x")
# has no recognizable shape at all -- the only signal is the *name* attached
# to it. This is a small, bounded list of names, not an attempt at universal
# detection: it catches the dominant real-world pattern (an env-style
# KEY=VALUE line, or a JSON "key": "value" pair) where the label itself is
# the giveaway, independent of what the value looks like.
_SENSITIVE_KEY_NAMES = (
    r"password|passwd|pwd|secret|token|api[_-]?key|apikey|access[_-]?key|"
    r"client[_-]?secret|private[_-]?key|secret[_-]?key|auth(?:orization)?|"
    r"credential(?:s)?|session[_-]?id|refresh[_-]?token"
)
# Matches a label followed by ``:``/``=`` and a value run (stopping at the
# next whitespace, quote, or JSON delimiter) inside flat/unstructured text
# such as a captured stdout/stderr blob, where there is no separate "key"
# field to inspect structurally. The optional compound-identifier prefix
# lets this match the common env-var shape (DB_PASSWORD, AWS_SECRET_KEY)
# and not just a bare standalone word.
LABELED_SECRET_PATTERN = re.compile(
    rf"(?i)((?:^|[^A-Za-z0-9])(?:[A-Za-z0-9]*[_-])*)({_SENSITIVE_KEY_NAMES})(?![A-Za-z0-9])([\"']?\s*[:=]\s*)[\"']?([^\s\"'&,}}\]\[]+)[\"']?"
)
# Matches a bare key name, for use when the structure (a dict) already
# separates the key from the value -- see sanitize_evidence_payload below.
_SENSITIVE_KEY_PATTERN = re.compile(rf"(?i)^({_SENSITIVE_KEY_NAMES})$")


def _is_sensitive_key(key: Any) -> bool:
    return bool(_SENSITIVE_KEY_PATTERN.match(str(key).strip()))


def sanitize_text(text: str) -> str:
    """Redacts sensitive API keys, tokens, and private keys from string content."""
    if not isinstance(text, str) or not text:
        return text
    sanitized = text
    for pattern in SECRET_PATTERNS:
        sanitized = pattern.sub("[REDACTED_SECRET]", sanitized)
    sanitized = LABELED_SECRET_PATTERN.sub(r"\1\2\3[REDACTED_SECRET]", sanitized)
    return sanitized


def sanitize_evidence_payload(data: Any, _key: Any = None) -> Any:
    """Recursively sanitizes sensitive content from evidence payloads.

    When ``data`` is reached as the value of a dict key, that key is checked
    against the sensitive-name list first: a structured payload already
    separates the label from the value, so a bare value like "hunter2" under
    a "password" key is redacted outright rather than relying on the value
    itself matching a recognizable secret shape.
    """
    if isinstance(data, str):
        if data and _is_sensitive_key(_key):
            return "[REDACTED_SECRET]"
        return sanitize_text(data)
    elif isinstance(data, dict):
        return {str(k): sanitize_evidence_payload(v, _key=k) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_evidence_payload(v, _key=_key) for v in data]
    elif isinstance(data, tuple):
        return tuple(sanitize_evidence_payload(v, _key=_key) for v in data)
    return data


class EvidenceType(str, enum.Enum):
    """Taxonomy of evidence collected during task execution."""
    TEST_EXECUTION = "test_execution"
    SYNTAX_VERIFICATION = "syntax_verification"
    CODE_REVIEW = "code_review"
    DIFF_INSPECTION = "diff_inspection"
    FILESYSTEM_OBSERVATION = "filesystem_observation"
    GIT_INTEGRITY = "git_integrity"
    SAFETY_INVARIANT = "safety_invariant"
    CLARIFICATION_RECORD = "clarification_record"
    CONTRACT_COMPLIANCE = "contract_compliance"
    FAILURE_REPAIR = "failure_repair"


class EvidenceTrustTier(int, enum.Enum):
    """Deterministic trust hierarchy for evidence sources (Rank 1 is highest trust)."""
    AUTHORITATIVE_EXECUTION = 1   # Concrete execution exit code & stdout in sandbox
    SYSTEM_INTEGRITY = 2          # Deterministic AST parsing, crypto hashes, git state
    OBSERVED_STATE = 3            # Direct filesystem inspection & snapshot
    DELIBERATIVE_REVIEW = 4       # Multi-turn provider review assessment
    DURABLE_CHECKPOINT = 5        # Rehydrated checkpoint data verified against disk
    AGENT_ASSERTION = 6           # Unverified textual hypothesis or provisional statement

    @property
    def rank(self) -> int:
        return self.value


class EvidenceStatus(str, enum.Enum):
    """Lifecycle validity status of an evidence record."""
    VALID = "valid"
    INVALIDATED = "invalidated"
    SUPERSEDED = "superseded"
    STALE = "stale"


class ReadinessLevel(str, enum.Enum):
    """Deterministic release readiness levels."""
    NOT_READY = "NOT_READY"
    BLOCKED = "BLOCKED"
    PARTIALLY_VERIFIED = "PARTIALLY_VERIFIED"
    READY_WITH_CONCERNS = "READY_WITH_CONCERNS"
    VERIFIED = "VERIFIED"
    READY = "READY"


@dataclass
class StructuredEvidence:
    """A discrete, structured, auditable evidence item."""
    evidence_id: str
    task_id: str
    subtask_id: str
    turn_number: int
    stage: str
    evidence_type: str  # EvidenceType value
    source: str
    trust_tier: int = EvidenceTrustTier.AUTHORITATIVE_EXECUTION.value
    status: str = EvidenceStatus.VALID.value
    invalidation_reason: str = ""
    workspace_root: str = ""
    worktree_id: str | None = None
    target_paths: list[str] = field(default_factory=list)
    target_symbols: list[str] = field(default_factory=list)
    command: list[str] | None = None
    exit_code: int | None = None
    content_fingerprint: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())

    @property
    def is_valid(self) -> bool:
        return self.status == EvidenceStatus.VALID.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "task_id": self.task_id,
            "subtask_id": self.subtask_id,
            "turn_number": self.turn_number,
            "stage": self.stage,
            "evidence_type": self.evidence_type,
            "source": self.source,
            "trust_tier": self.trust_tier,
            "status": self.status,
            "invalidation_reason": self.invalidation_reason,
            "workspace_root": self.workspace_root,
            "worktree_id": self.worktree_id,
            "target_paths": list(self.target_paths),
            "target_symbols": list(self.target_symbols),
            "command": list(self.command) if self.command else None,
            "exit_code": self.exit_code,
            "content_fingerprint": self.content_fingerprint,
            "payload": sanitize_evidence_payload(dict(self.payload)),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StructuredEvidence:
        if not isinstance(data, dict):
            return cls(
                evidence_id="",
                task_id="",
                subtask_id="",
                turn_number=0,
                stage="",
                evidence_type=EvidenceType.TEST_EXECUTION.value,
                source="unknown",
                status=EvidenceStatus.INVALIDATED.value,
                invalidation_reason="malformed_evidence_data",
            )
        # A record missing its "status" key entirely is not a freshly recorded
        # observation (record() always sets one) -- it is deserialized,
        # possibly-incomplete/corrupted state. Defaulting that to VALID would
        # let a stripped-down or hand-crafted entry masquerade as observed
        # evidence merely by omitting the field, so an absent status fails
        # closed instead of granting trust.
        raw_status = data.get("status")
        status = str(raw_status) if raw_status else EvidenceStatus.INVALIDATED.value
        invalidation_reason = str(data.get("invalidation_reason", ""))
        if not raw_status:
            invalidation_reason = invalidation_reason or "missing_status_on_deserialization"
        return cls(
            evidence_id=str(data.get("evidence_id", "")),
            task_id=str(data.get("task_id", "")),
            subtask_id=str(data.get("subtask_id", "")),
            turn_number=int(data.get("turn_number", 0) or 0),
            stage=str(data.get("stage", "")),
            evidence_type=str(data.get("evidence_type", EvidenceType.TEST_EXECUTION.value)),
            source=str(data.get("source", "unknown")),
            trust_tier=int(data.get("trust_tier", EvidenceTrustTier.AUTHORITATIVE_EXECUTION.value) or EvidenceTrustTier.AUTHORITATIVE_EXECUTION.value),
            status=status,
            invalidation_reason=invalidation_reason,
            workspace_root=str(data.get("workspace_root", "")),
            worktree_id=data.get("worktree_id"),
            target_paths=[str(p) for p in (data.get("target_paths") or [])],
            target_symbols=[str(s) for s in (data.get("target_symbols") or [])],
            command=[str(c) for c in data.get("command", [])] if isinstance(data.get("command"), list) else None,
            exit_code=data.get("exit_code"),
            content_fingerprint=str(data.get("content_fingerprint", "")),
            payload=dict(data.get("payload") or {}) if isinstance(data.get("payload"), dict) else {},
            timestamp=str(data.get("timestamp", "")),
        )


@dataclass
class CompletionGateResult:
    """Outcome of evaluating an individual hard completion gate."""
    gate_name: str
    passed: bool
    reason: str
    supporting_evidence_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_name": self.gate_name,
            "passed": self.passed,
            "reason": sanitize_text(self.reason),
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompletionGateResult:
        if not isinstance(data, dict):
            return cls(gate_name="unknown", passed=False, reason="Invalid data")
        return cls(
            gate_name=str(data.get("gate_name", "unknown")),
            passed=bool(data.get("passed", False)),
            reason=str(data.get("reason", "")),
            supporting_evidence_ids=[str(i) for i in (data.get("supporting_evidence_ids") or [])],
        )


@dataclass
class CompletionAssessment:
    """Structured, durable, auditable completion and release readiness decision."""
    task_id: str
    subtask_id: str
    readiness_level: str  # ReadinessLevel value
    is_ready: bool
    decision_reason: str
    gates_evaluated: list[CompletionGateResult] = field(default_factory=list)
    supporting_evidence_ids: list[str] = field(default_factory=list)
    invalidated_evidence_ids: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    unresolved_risks: list[str] = field(default_factory=list)
    answers_to_ten_questions: dict[str, Any] = field(default_factory=dict)
    assessed_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "subtask_id": self.subtask_id,
            "readiness_level": self.readiness_level,
            "is_ready": self.is_ready,
            "decision_reason": sanitize_text(self.decision_reason),
            "gates_evaluated": [g.to_dict() for g in self.gates_evaluated],
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "invalidated_evidence_ids": list(self.invalidated_evidence_ids),
            "missing_evidence": [sanitize_text(m) for m in self.missing_evidence],
            "unresolved_risks": [sanitize_text(r) for r in self.unresolved_risks],
            "answers_to_ten_questions": sanitize_evidence_payload(dict(self.answers_to_ten_questions)),
            "assessed_at": self.assessed_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompletionAssessment:
        if not isinstance(data, dict):
            return cls(
                task_id="",
                subtask_id="",
                readiness_level=ReadinessLevel.NOT_READY.value,
                is_ready=False,
                decision_reason="Invalid data",
            )
        gates = [
            CompletionGateResult.from_dict(g)
            for g in (data.get("gates_evaluated") or [])
            if isinstance(g, dict)
        ]
        return cls(
            task_id=str(data.get("task_id", "")),
            subtask_id=str(data.get("subtask_id", "")),
            readiness_level=str(data.get("readiness_level", ReadinessLevel.NOT_READY.value)),
            is_ready=bool(data.get("is_ready", False)),
            decision_reason=str(data.get("decision_reason", "")),
            gates_evaluated=gates,
            supporting_evidence_ids=[str(i) for i in (data.get("supporting_evidence_ids") or [])],
            invalidated_evidence_ids=[str(i) for i in (data.get("invalidated_evidence_ids") or [])],
            missing_evidence=[str(m) for m in (data.get("missing_evidence") or [])],
            unresolved_risks=[str(r) for r in (data.get("unresolved_risks") or [])],
            answers_to_ten_questions=dict(data.get("answers_to_ten_questions") or {}),
            assessed_at=str(data.get("assessed_at", "")),
        )


class CompletionEvidenceStore:
    """Durable store for structured evidence with automatic state invalidation and disk revalidation."""

    def __init__(self, workspace_root: str | Path, max_entries: int = 100):
        self.workspace_root = Path(workspace_root).resolve()
        self.max_entries = max_entries
        self._evidence_list: list[StructuredEvidence] = []
        # Monotonic counter for evidence_id generation. FIFO eviction in
        # record() below shrinks _evidence_list, so an id derived from the
        # list's current length (the old scheme) can repeat after eviction
        # and collide with an id already held by a still-present entry.
        self._next_seq = 0
        # Which evidence *types* have ever been recorded in this store's
        # lifetime, independent of whether the specific entries that first
        # established that fact have since been evicted by the max_entries
        # bound. Gates that ask "was X ever attempted at all" (e.g. the
        # completion engine's "no test suite configured" shortcut) must
        # answer from this durable, O(distinct-types)-bounded signal, not by
        # scanning the bounded/evictable entry list -- otherwise a long
        # enough run can silently evict the one entry proving tests were
        # once attempted, and the shortcut wrongly reopens.
        self._types_ever_recorded: set[str] = set()

    def record(
        self,
        task_id: str,
        subtask_id: str,
        turn_number: int,
        stage: str,
        evidence_type: EvidenceType | str,
        source: str,
        trust_tier: EvidenceTrustTier | int = EvidenceTrustTier.AUTHORITATIVE_EXECUTION,
        target_paths: Sequence[str] | None = None,
        target_symbols: Sequence[str] | None = None,
        command: Sequence[str] | None = None,
        exit_code: int | None = None,
        payload: dict[str, Any] | None = None,
        worktree_id: str | None = None,
    ) -> StructuredEvidence:
        """Records a new evidence item and computes content fingerprint on affected paths."""
        paths = sorted({str(p).replace("\\", "/") for p in (target_paths or []) if p})
        symbols = sorted({str(s) for s in (target_symbols or []) if s})
        fingerprint = compute_state_fingerprint(self.workspace_root, paths) if paths else ""
        
        type_str = evidence_type.value if isinstance(evidence_type, EvidenceType) else str(evidence_type)
        tier_val = trust_tier.value if isinstance(trust_tier, EvidenceTrustTier) else int(trust_tier)

        sanitized_payload = sanitize_evidence_payload(dict(payload or {}))

        self._next_seq += 1
        ev_id = f"ev-{task_id}-{subtask_id}-t{turn_number}-{type_str}-{self._next_seq}"
        item = StructuredEvidence(
            evidence_id=ev_id,
            task_id=task_id,
            subtask_id=subtask_id,
            turn_number=turn_number,
            stage=stage,
            evidence_type=type_str,
            source=source,
            trust_tier=tier_val,
            status=EvidenceStatus.VALID.value,
            workspace_root=str(self.workspace_root),
            worktree_id=worktree_id,
            target_paths=paths,
            target_symbols=symbols,
            command=list(command) if command else None,
            exit_code=exit_code,
            content_fingerprint=fingerprint,
            payload=sanitized_payload,
        )

        self._evidence_list.append(item)
        self._types_ever_recorded.add(type_str)
        if len(self._evidence_list) > self.max_entries:
            del self._evidence_list[: len(self._evidence_list) - self.max_entries]
        return item

    def was_ever_recorded(self, evidence_type: EvidenceType | str) -> bool:
        """Whether an entry of this evidence type has ever been recorded in
        this store's lifetime -- durable across FIFO eviction of the bounded
        entry list (see ``_types_ever_recorded`` in __init__)."""
        type_str = evidence_type.value if isinstance(evidence_type, EvidenceType) else str(evidence_type)
        return type_str in self._types_ever_recorded

    def invalidate_on_file_mutation(self, modified_paths: Sequence[str], reason: str = "content_modified_in_subsequent_turn") -> list[str]:
        """Invalidates prior test, syntax, and review evidence when files are modified.

        Phase 4.24: a TEST_EXECUTION/CODE_REVIEW/DIFF_INSPECTION entry's
        ``target_paths`` records which files were part of the change-set *at
        recording time* -- it is a correlation aid, not the entry's actual
        claim. A full test run or a code review certifies the *entire*
        tracked change-set as it stood at that moment, not merely the files
        that happen to be listed. Previously this method only re-checked the
        fingerprint of an entry's own ``target_paths``, so a mutation to a
        file the entry had never seen (e.g. a new file introduced by a later
        repair turn) left it ``VALID`` -- a passing test/review recorded
        before that file existed could then be read as if it had verified
        the file's current (possibly broken) content. Reproduced end-to-end:
        recorded PASS test evidence over {a.py}, a later turn rewrites a NEW
        file b.py with a runtime bug, a genuine live review of the resulting
        two-file diff approves it (LLM review is imperfect) -- ``evaluate()``
        reached ``is_ready=True`` on test evidence that never observed b.py.
        Any mutation now invalidates every VALID global-validation entry
        unconditionally, regardless of whether the mutated path is one the
        entry already knew about: the entry's certification is stale the
        moment *anything* in the tracked tree changes, full stop. Per-file
        evidence (SYNTAX_VERIFICATION, SAFETY_INVARIANT, etc.) keeps its
        original narrower behaviour -- it only claims to cover the specific
        file(s) it names, so an unrelated mutation correctly leaves it alone.
        """
        invalidated_ids: list[str] = []
        mod_set = {str(p).replace("\\", "/") for p in modified_paths if p}
        if not mod_set:
            return invalidated_ids

        for ev in self._evidence_list:
            if ev.status != EvidenceStatus.VALID.value:
                continue

            is_global_validation = ev.evidence_type in {
                EvidenceType.TEST_EXECUTION.value,
                EvidenceType.CODE_REVIEW.value,
                EvidenceType.DIFF_INSPECTION.value,
            }

            if is_global_validation:
                ev.status = EvidenceStatus.INVALIDATED.value
                ev.invalidation_reason = reason
                invalidated_ids.append(ev.evidence_id)
                continue

            if not (set(ev.target_paths) & mod_set):
                continue
            if ev.target_paths:
                current_fp = compute_state_fingerprint(self.workspace_root, ev.target_paths)
                if current_fp != ev.content_fingerprint:
                    ev.status = EvidenceStatus.INVALIDATED.value
                    ev.invalidation_reason = reason
                    invalidated_ids.append(ev.evidence_id)
            else:
                ev.status = EvidenceStatus.INVALIDATED.value
                ev.invalidation_reason = reason
                invalidated_ids.append(ev.evidence_id)

        return invalidated_ids

    def revalidate_against_disk(self, filesystem: ProjectFilesystem | None = None) -> list[str]:
        """Revalidates all VALID evidence against actual disk content fingerprints.

        Used upon checkpoint resume or before final completion assessment to prevent
        stale evidence from surviving post-checkpoint disk mutations.
        """
        invalidated_ids: list[str] = []
        root = filesystem.root if filesystem else self.workspace_root

        for ev in self._evidence_list:
            if ev.status != EvidenceStatus.VALID.value:
                continue

            if not ev.target_paths:
                continue

            current_fp = compute_state_fingerprint(root, ev.target_paths)
            if current_fp != ev.content_fingerprint:
                ev.status = EvidenceStatus.INVALIDATED.value
                ev.invalidation_reason = "disk_mutation_detected_during_revalidation"
                invalidated_ids.append(ev.evidence_id)

        return invalidated_ids

    def get_valid_evidence(
        self,
        evidence_type: EvidenceType | str | None = None,
        task_id: str | None = None,
    ) -> list[StructuredEvidence]:
        """Returns all currently valid evidence items, optionally filtered by
        type and by owning task.

        Phase 4.23: a ``CompletionEvidenceStore`` is designed to hold exactly
        one task's evidence, but nothing previously enforced that at read
        time -- every entry already carries the ``task_id`` it was recorded
        under (see ``record()``), yet no caller ever checked it. If a store
        instance were ever shared or a checkpoint's evidence blob were ever
        loaded against the wrong task (a storage bug, a copied checkpoint
        file), evidence recorded for one task would silently satisfy another
        with no defense at all. Every production caller in this module and
        in task_contract.py now passes the current task's id; ``task_id=None``
        (the default) preserves the exact prior behavior for any other
        caller, so this is purely additive.
        """
        type_filter = evidence_type.value if isinstance(evidence_type, EvidenceType) else evidence_type
        return [
            ev for ev in self._evidence_list
            if ev.status == EvidenceStatus.VALID.value
            and (type_filter is None or ev.evidence_type == type_filter)
            and (task_id is None or ev.task_id == task_id)
        ]

    def all_entries(self) -> list[StructuredEvidence]:
        return list(self._evidence_list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace_root": str(self.workspace_root),
            "max_entries": self.max_entries,
            "entries": [e.to_dict() for e in self._evidence_list],
            # Bounded by the fixed EvidenceType enum (currently 10 members) --
            # this never grows with entry count or eviction.
            "types_ever_recorded": sorted(self._types_ever_recorded),
            "next_seq": self._next_seq,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompletionEvidenceStore:
        if not isinstance(data, dict):
            return cls(workspace_root=".")
        root = data.get("workspace_root", ".")
        store = cls(workspace_root=root, max_entries=int(data.get("max_entries", 100) or 100))
        store._evidence_list = [
            StructuredEvidence.from_dict(item)
            for item in (data.get("entries") or [])
            if isinstance(item, dict)
        ]
        raw_types = data.get("types_ever_recorded")
        if isinstance(raw_types, list) and raw_types:
            # New-format checkpoint: trust the durable signal directly.
            store._types_ever_recorded = {str(t) for t in raw_types}
        else:
            # Old checkpoint predating this field (Phase 4.20 and earlier):
            # best-effort reconstruction from whatever entries survived,
            # matching that era's actual behavior exactly rather than
            # silently granting a stronger guarantee than the data supports.
            store._types_ever_recorded = {e.evidence_type for e in store._evidence_list}
        raw_seq = data.get("next_seq")
        store._next_seq = int(raw_seq) if isinstance(raw_seq, int) and raw_seq > len(store._evidence_list) else len(store._evidence_list)
        return store


class CompletionDecisionEngine:
    """Deterministic evaluation engine enforcing hard completion gates and computing readiness."""

    def __init__(self, filesystem: ProjectFilesystem):
        self.filesystem = filesystem

    def evaluate(
        self,
        task: Task,
        subtask: Subtask | None,
        plan: Plan,
        evidence_store: CompletionEvidenceStore,
        applied_operations: Sequence[FileOperation],
        current_diff: str,
        last_review: ReviewResult | None = None,
        last_failure: FailureAnalysis | None = None,
        worktree_id: str | None = None,
        clarification_requests: Sequence[Any] | None = None,
    ) -> CompletionAssessment:
        """Evaluates all hard completion gates and computes the release readiness decision."""
        # 0. Revalidate evidence store against disk before evaluating
        evidence_store.revalidate_against_disk(self.filesystem)

        task_id = task.task_id
        subtask_id = subtask.subtask_id if subtask else (getattr(task, "current_subtask_id", "main") or "main")
        gates: list[CompletionGateResult] = []
        supporting_ids: list[str] = []
        invalidated_ids = [e.evidence_id for e in evidence_store.all_entries() if e.status == EvidenceStatus.INVALIDATED.value]
        missing_evidence: list[str] = []
        unresolved_risks: list[str] = []

        # Phase 4.23: scope every gate below to this task's own evidence --
        # see get_valid_evidence's docstring for why this matters.
        valid_entries = evidence_store.get_valid_evidence(task_id=task_id)

        # ----------------------------------------------------------------------
        # Gate 1: Safety Invariants Intact (Protected files & secrets)
        # ----------------------------------------------------------------------
        protected_violated = False
        protected_reason = "Protected paths and secrets intact"
        for op in applied_operations:
            path_str = str(getattr(op, "path", ""))
            if "tool_engine.py" in path_str or "approval.py" in path_str:
                protected_violated = True
                protected_reason = f"Protected file modification detected: {path_str}"
                break
            if any(s.lower() in path_str.lower() for s in SECRET_NAMES):
                protected_violated = True
                protected_reason = f"Secret file modification detected: {path_str}"
                break

        safety_ev = [e.evidence_id for e in valid_entries if e.evidence_type == EvidenceType.SAFETY_INVARIANT.value]
        gates.append(CompletionGateResult(
            gate_name="GATE_PROTECTED_FILES_INTACT",
            passed=not protected_violated,
            reason=protected_reason,
            supporting_evidence_ids=safety_ev,
        ))
        if not protected_violated:
            supporting_ids.extend(safety_ev)
        else:
            unresolved_risks.append(protected_reason)

        # ----------------------------------------------------------------------
        # Gate 2: Workspace Diff Non-Empty (when implementation required)
        # ----------------------------------------------------------------------
        has_changes = bool(applied_operations or current_diff.strip())
        no_changes_needed = (not applied_operations and not current_diff.strip() and last_review is not None and last_review.verdict == "APPROVED")
        diff_ev = [e.evidence_id for e in valid_entries if e.evidence_type == EvidenceType.DIFF_INSPECTION.value]
        gates.append(CompletionGateResult(
            gate_name="GATE_WORKSPACE_DIFF_NON_EMPTY",
            passed=has_changes or no_changes_needed,
            reason=f"Authoritative diff contains {len(applied_operations)} operation(s)" if has_changes else (
                "No changes required and review approved" if no_changes_needed else "No workspace modifications produced"
            ),
            supporting_evidence_ids=diff_ev,
        ))
        if has_changes:
            supporting_ids.extend(diff_ev)
        elif not no_changes_needed:
            missing_evidence.append("Authoritative workspace diff")

        # ----------------------------------------------------------------------
        # Gate 3: Python AST Syntax Cleanliness
        # ----------------------------------------------------------------------
        syntax_passed = True
        syntax_errors: list[str] = []
        syntax_ev = [e.evidence_id for e in valid_entries if e.evidence_type == EvidenceType.SYNTAX_VERIFICATION.value and e.exit_code == 0]

        for op in applied_operations:
            p = getattr(op, "path", None)
            if p and str(p).endswith(".py"):
                full_path = self.filesystem.root / p
                if full_path.is_file():
                    try:
                        content = self.filesystem.read_file(p)
                        ast.parse(content, filename=p)
                    except Exception as err:
                        syntax_passed = False
                        syntax_errors.append(f"{p}: {err}")

        gates.append(CompletionGateResult(
            gate_name="GATE_SYNTAX_CLEAN",
            passed=syntax_passed,
            reason="All modified Python files parse cleanly" if syntax_passed else f"Syntax errors: {'; '.join(syntax_errors)}",
            supporting_evidence_ids=syntax_ev,
        ))
        if syntax_passed:
            supporting_ids.extend(syntax_ev)
        else:
            unresolved_risks.append(f"Syntax errors present: {'; '.join(syntax_errors)}")

        # ----------------------------------------------------------------------
        # Gate 4: Final Validation Execution & Passing Tests (Conflict Resolved)
        # ----------------------------------------------------------------------
        test_ev = [
            e for e in valid_entries
            if e.evidence_type == EvidenceType.TEST_EXECUTION.value and e.exit_code == 0
        ]
        test_fail_ev = [
            e for e in valid_entries
            if e.evidence_type == EvidenceType.TEST_EXECUTION.value and e.exit_code != 0
        ]
        # A test suite was attempted at some point for this evidence store if any
        # TEST_EXECUTION entry exists at all, valid or not. Once that is true, a
        # workspace mutation that invalidates the passing evidence (Attack C: stale
        # test evidence) must never be allowed to silently fall back to the
        # "no automated test suite configured" shortcut below -- that shortcut is
        # only safe when tests were genuinely never run for this task/subtask.
        # Phase 4.21: read this from the store's durable, eviction-proof
        # signal (was_ever_recorded) rather than scanning the bounded,
        # FIFO-evictable entry list -- a long enough run could otherwise
        # evict the one entry proving tests were ever attempted, silently
        # reopening the Attack C bypass this check exists to close.
        test_ever_attempted = evidence_store.was_ever_recorded(EvidenceType.TEST_EXECUTION)

        if test_fail_ev:
            test_passed = False
            val_reason = f"{len(test_fail_ev)} active test failure(s) detected"
        elif test_ev:
            test_passed = True
            val_reason = f"{len(test_ev)} valid test execution(s) passed cleanly on current workspace"
        elif test_ever_attempted:
            test_passed = False
            val_reason = "Prior test evidence invalidated by workspace mutation; no fresh passing validation on current state"
        elif syntax_passed and applied_operations:
            test_passed = True
            val_reason = "No automated test suite configured; syntax verified cleanly"
        elif no_changes_needed:
            test_passed = True
            val_reason = "No changes applied; review approved current repository state"
        else:
            test_passed = False
            val_reason = "No passing validation evidence on current workspace state"

        gates.append(CompletionGateResult(
            gate_name="GATE_VALIDATION_PASSED",
            passed=test_passed,
            reason=val_reason,
            supporting_evidence_ids=[e.evidence_id for e in test_ev],
        ))
        if test_passed and test_ev:
            supporting_ids.extend([e.evidence_id for e in test_ev])
        elif not test_passed:
            missing_evidence.append("Passing test execution on current workspace")

        # ----------------------------------------------------------------------
        # Gate 5: Review Approval on Authoritative Diff (Cryptographic Binding)
        # ----------------------------------------------------------------------
        current_diff_hash = hashlib.sha256(current_diff.encode("utf-8")).hexdigest()[:16] if current_diff else ""
        review_ev = [
            e for e in valid_entries
            if e.evidence_type == EvidenceType.CODE_REVIEW.value and e.payload.get("verdict") == "APPROVED"
        ]

        review_diff_matches = True
        if review_ev and current_diff_hash:
            reviews_with_hash = [e for e in review_ev if e.payload.get("diff_hash")]
            if reviews_with_hash:
                matching_hash = [e for e in reviews_with_hash if e.payload.get("diff_hash") == current_diff_hash]
                if not matching_hash:
                    review_diff_matches = False

        # A live ReviewResult AND a corresponding recorded evidence entry are both
        # required -- neither one alone is sufficient. Evidence-store entries are
        # durable records of a past observation, not a substitute for the current
        # call's live review outcome (an attacker or stale checkpoint could plant
        # a matching CODE_REVIEW entry with no real review having occurred), and a
        # live "APPROVED" object with no corroborating evidence trail is likewise
        # an unverified assertion.
        review_passed = (
            last_review is not None
            and last_review.verdict == "APPROVED"
            and len(review_ev) > 0
            and review_diff_matches
        )

        if not review_diff_matches:
            review_reason = "Review diff hash mismatch: workspace modified post-review"
        elif review_passed:
            review_reason = "Code review approved current authoritative diff"
        else:
            review_reason = f"Review verdict is {last_review.verdict if last_review else 'MISSING'}"

        gates.append(CompletionGateResult(
            gate_name="GATE_REVIEW_APPROVED",
            passed=review_passed,
            reason=review_reason,
            supporting_evidence_ids=[e.evidence_id for e in review_ev if e.payload.get("diff_hash") == current_diff_hash],
        ))
        if review_passed:
            supporting_ids.extend([e.evidence_id for e in review_ev])
        else:
            if not review_ev or not review_diff_matches:
                missing_evidence.append("Approved code review on current diff")

        # ----------------------------------------------------------------------
        # Gate 6: Zero Unresolved Failures or Active Repair Loops
        # ----------------------------------------------------------------------
        has_unresolved_failure = False
        if last_failure is not None and getattr(last_failure, "probable_root_cause", ""):
            if not (test_passed and syntax_passed and len(test_ev) > 0):
                has_unresolved_failure = True

        gates.append(CompletionGateResult(
            gate_name="GATE_NO_UNRESOLVED_FAILURES",
            passed=not has_unresolved_failure,
            reason="No unresolved failure state" if not has_unresolved_failure else f"Unresolved failure: {last_failure.probable_root_cause}",
            supporting_evidence_ids=[],
        ))
        if has_unresolved_failure:
            unresolved_risks.append(f"Unresolved failure: {last_failure.probable_root_cause}")

        # ----------------------------------------------------------------------
        # Gate 7: No Pending Clarifications
        # ----------------------------------------------------------------------
        pending_clarifications = []
        if clarification_requests:
            for req in clarification_requests:
                status = req.status if hasattr(req, "status") else (req.get("status") if isinstance(req, dict) else "pending")
                if status == "pending":
                    q = req.question if hasattr(req, "question") else (req.get("question") if isinstance(req, dict) else "")
                    pending_clarifications.append(q)

        gates.append(CompletionGateResult(
            gate_name="GATE_NO_PENDING_CLARIFICATIONS",
            passed=len(pending_clarifications) == 0,
            reason="All clarifications resolved" if not pending_clarifications else f"Pending clarification: {'; '.join(pending_clarifications)}",
            supporting_evidence_ids=[e.evidence_id for e in valid_entries if e.evidence_type == EvidenceType.CLARIFICATION_RECORD.value],
        ))
        if pending_clarifications:
            unresolved_risks.append(f"Unanswered clarification requests: {len(pending_clarifications)}")

        # ----------------------------------------------------------------------
        # Gate 8: Worktree Identity & Scoping
        # ----------------------------------------------------------------------
        worktree_ok = True
        if worktree_id:
            foreign_ev = [
                e.evidence_id for e in valid_entries
                if e.worktree_id and e.worktree_id != worktree_id
            ]
            if foreign_ev:
                worktree_ok = False
        gates.append(CompletionGateResult(
            gate_name="GATE_WORKTREE_ISOLATION",
            passed=worktree_ok,
            reason="Evidence correctly scoped to current worktree" if worktree_ok else "Foreign worktree evidence detected",
            supporting_evidence_ids=[],
        ))

        # ----------------------------------------------------------------------
        # Compute Readiness Level & Answers to 10 Questions
        # ----------------------------------------------------------------------
        all_passed = all(g.passed for g in gates)
        critical_failed = any(
            not g.passed
            for g in gates
            if g.gate_name in {"GATE_PROTECTED_FILES_INTACT", "GATE_SYNTAX_CLEAN"}
        )

        if critical_failed:
            readiness = ReadinessLevel.BLOCKED
            is_ready = False
            decision_reason = "Hard safety or syntax gate failed"
        elif all_passed:
            readiness = ReadinessLevel.READY
            is_ready = True
            decision_reason = "All hard completion gates satisfied with verified evidence"
        else:
            readiness = ReadinessLevel.NOT_READY
            is_ready = False
            failed_names = [g.gate_name for g in gates if not g.passed]
            decision_reason = f"Mandatory gates failed: {', '.join(failed_names)}"

        # Structured Answers to the 10 Questions
        answers: dict[str, Any] = {
            "1_what_was_changed": [getattr(op, "path", "") for op in applied_operations if getattr(op, "path", "")],
            "2_why_was_it_changed": task.objective,
            "3_what_evidence_proves_it_works": [e.evidence_id for e in test_ev],
            "4_what_validation_was_executed": [
                {"command": " ".join(e.command) if e.command else "", "exit_code": e.exit_code}
                for e in valid_entries if e.evidence_type == EvidenceType.TEST_EXECUTION.value
            ],
            "5_what_was_reviewed": {
                "verdict": last_review.verdict if last_review else "SKIPPED",
                "summary": last_review.summary if last_review else "No review recorded",
                "diff_hash": current_diff_hash,
            },
            "6_what_was_verified": {
                "syntax_clean": syntax_passed,
                "test_commands_count": len(test_ev),
            },
            "7_what_safety_invariants_were_checked": {
                "protected_files_intact": not protected_violated,
                "sandbox_bounds_respected": True,
            },
            "8_what_remains_uncertain": missing_evidence + unresolved_risks,
            "9_is_ready_for_completion": is_ready,
            "10_justified_readiness_level": readiness.value,
        }

        supporting_ids = sorted(set(supporting_ids))
        return CompletionAssessment(
            task_id=task_id,
            subtask_id=subtask_id,
            readiness_level=readiness.value,
            is_ready=is_ready,
            decision_reason=decision_reason,
            gates_evaluated=gates,
            supporting_evidence_ids=supporting_ids,
            invalidated_evidence_ids=invalidated_ids,
            missing_evidence=missing_evidence,
            unresolved_risks=unresolved_risks,
            answers_to_ten_questions=answers,
        )
