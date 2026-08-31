"""Phase 4.21: Task Outcome Integrity & Requirement Traceability.

Phase 4.18 established that completion must be supported by evidence. Phase
4.19 made the agent obey that evidence. Phase 4.20 hardened that trust
boundary against adversarial state. All three answer the question "did the
agent's technical process succeed?" -- none of them answer "did the agent
actually do what the user asked?"

A task can pass every technical completion gate (diff non-empty, syntax
clean, tests green, review approved) while only partially implementing a
multi-part request, silently dropping a stated constraint, or leaving a
required deliverable (like documentation) untouched. This module adds a
second, independent dimension to the completion decision: whether the
concrete, checkable requirements derived from the user's task are backed by
evidence, deterministically and fail-closed -- never by a provider's claim of
"done", a reviewer's blanket approval, or a global "tests passed" signal that
says nothing about *which* requirement it covers.

This is deliberately not a requirements-management system. There is no
priority scoring, no dependency graph, no natural-language understanding.
Requirements are derived once, deterministically, from the task's own
objective text (not the planner's output -- a planner, especially a
minimal/offline one, commonly emits generic process steps rather than
user-facing deliverables), and are checked by simple, explainable rules
that reuse the Phase 4.18 evidence store, fingerprinting, and trust tiers
rather than inventing a second one.
"""

from __future__ import annotations

import datetime
import enum
import math
import re
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

from .completion import (
    CompletionEvidenceStore,
    EvidenceTrustTier,
    EvidenceType,
    sanitize_text,
)
from .filesystem import ProjectFilesystem
from .models import FileOperation, Plan, ReviewResult, Task

# ---------------------------------------------------------------------------
# Bounds. A malicious or careless task objective must not be able to grow
# the contract, or any one requirement, without limit.
# ---------------------------------------------------------------------------
MAX_REQUIREMENTS = 50
MAX_NON_GOALS = 20
MAX_TARGET_PATHS_PER_REQUIREMENT = 20
MAX_EVIDENCE_IDS_PER_REQUIREMENT = 20
MAX_STATEMENT_CHARS = 500
MAX_REASON_CHARS = 500
MAX_OBJECTIVE_SCAN_CHARS = 5000


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class RequirementType(str, enum.Enum):
    """Deterministic taxonomy. Not every category is useful for every task;
    the extractor only emits the ones it has concrete evidence to justify."""
    FUNCTIONAL = "functional"
    BEHAVIORAL = "behavioral"
    NON_FUNCTIONAL = "non_functional"
    CONSTRAINT = "constraint"
    SAFETY = "safety"
    COMPATIBILITY = "compatibility"
    DOCUMENTATION = "documentation"
    TESTING = "testing"


class RequirementImportance(str, enum.Enum):
    """MUST-importance requirements gate final completion. SHOULD and
    OPTIONAL are reported but never block -- they exist so a requirement's
    weight is explicit rather than implied, and so a provider/reviewer can
    never silently turn a MUST into an OPTIONAL by omission (see
    RequirementAssessmentEngine: importance is only ever set at contract
    creation/amendment, never by the assessment pass itself)."""
    MUST = "must"
    SHOULD = "should"
    OPTIONAL = "optional"


class RequirementState(str, enum.Enum):
    UNVERIFIED = "unverified"
    IN_PROGRESS = "in_progress"
    SATISFIED = "satisfied"
    FAILED = "failed"
    BLOCKED = "blocked"
    NOT_APPLICABLE = "not_applicable"


class VerificationStrategy(str, enum.Enum):
    """How a requirement's state is decided. Each strategy is a small,
    deterministic rule over authoritative state (the changeset, the evidence
    store, a resolved clarification) -- never a provider's or reviewer's
    say-so."""
    # A concrete change plus (heuristic) content correlation with the
    # requirement's own wording -- the closest this module gets to "was this
    # actually implemented", and explicitly a proxy, not a proof.
    DIFF_PRESENCE = "diff_presence"
    # A negative requirement: satisfied exactly when its target paths are
    # absent from the changeset, failed exactly when present.
    CONSTRAINT_ABSENCE = "constraint_absence"
    # Requires a currently-valid passing TEST_EXECUTION evidence entry.
    TEST_EVIDENCE = "test_evidence"
    # No deterministic rule applies (e.g. "preserve backward compatibility"
    # names no concrete path or command) -- stays UNVERIFIED until either a
    # clarification pins it down to something checkable, or a human/operator
    # marks it NOT_APPLICABLE. Never silently defaults to satisfied.
    MANUAL_CLARIFICATION = "manual_clarification"


# ---------------------------------------------------------------------------
# Requirement extraction
# ---------------------------------------------------------------------------

# Deliberately small and generic (not per-project, not per-task) -- filters
# out connective words so token-overlap correlation compares content words,
# not sentence glue.
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "to", "of", "in", "on", "for",
    "with", "without", "is", "are", "be", "as", "at", "by", "it", "this",
    "that", "these", "those", "also", "please", "should", "must", "will",
    "can", "not", "do", "does", "did", "from", "into", "so", "than", "then",
    # Generic task-instruction vocabulary: carries no requirement-specific
    # content, so it must not count toward (or dilute) the token-overlap
    # threshold below -- otherwise a near-universal word like "add" or
    # "functionality" could make an unrelated change look like it satisfies
    # any requirement that happens to also use that word.
    "add", "added", "adding", "implement", "implemented", "implementing",
    "create", "created", "creating", "update", "updated", "updating",
    "change", "changed", "changing", "modify", "modified", "modifying",
    "make", "making", "ensure", "ensuring", "include", "included",
    "including", "support", "supporting", "supported", "provide",
    "provided", "providing", "allow", "allows", "allowing", "using", "use",
    "new", "feature", "features", "functionality", "capability",
    "capabilities", "need", "needs", "needed", "want", "wants", "wanted",
    "task", "requirement", "requirements",
    # Units and measurement words: these describe a numeric value in prose
    # but essentially never appear as their own word in the code that sets
    # that value (a timeout constant is written as a number, not the word
    # "seconds"), so requiring them as content terms produces false
    # negatives on entirely correct implementations.
    "second", "seconds", "minute", "minutes", "hour", "hours", "day", "days",
    "millisecond", "milliseconds", "ms", "byte", "bytes", "kb", "mb", "gb",
    "percent", "percentage", "times", "value", "values", "number",
})

_CONSTRAINT_TRIGGERS = re.compile(
    r"(?i)\b(?:do\s+not|don't|never|must\s+not|should\s+not|avoid)\s+"
    r"(?:modify(?:ing)?|chang(?:e|ing)|touch(?:ing)?|edit(?:ing)?|alter(?:ing)?|"
    r"delet(?:e|ing)|remov(?:e|ing))\s+"
    # The path token must end in a word/slash/dash character, not a dot --
    # otherwise a sentence-ending period after the filename (".py.") gets
    # swallowed into the captured path.
    r"([`\"']?[\w./\\-]*[\w/\\-][`\"']?)"
)
_PRESERVE_TRIGGER = re.compile(
    r"(?i)\b(?:preserve|keep|maintain)\s+([\w./\\ -]+?)(?=[.,;]|$)"
)
# Deliberately narrower than a bare "documentation|docs?" / "tests?" match:
# both words are common English vocabulary used generically in countless
# objectives that are not asking for a documentation or test deliverable
# ("Test repo lock", "document the design in your head first") -- only an
# explicit action phrase ("update the docs", "add tests", "test coverage")
# is treated as the user actually asking for that deliverable.
_DOC_HINT = re.compile(r"(?i)\b(?:update|write|add|include|provide)\s+(?:the\s+|a\s+)?(?:documentation|docs?|readme|changelog)\b")
_TEST_HINT = re.compile(r"(?i)\b(?:add|write|include|provide)\s+(?:unit\s+)?tests?\b|\btest\s+coverage\b")


# Tier 1: unambiguous separators -- a numbered/bulleted list item, a
# semicolon, or a sentence boundary genuinely does mark a new, independent
# clause with no real ambiguity.
_UNAMBIGUOUS_SEPARATOR = re.compile(r"(?i)\s*(?:\n\s*(?:[-*]|\d+[.)])\s*|;\s*|\.\s+(?=[A-Z]))\s*")
# Tier 2: an " and " conjunction is genuinely ambiguous in English -- "add
# CSV export and JSON export" joins two distinct deliverables, but "pause
# and resume telemetry" joins two verbs describing one cohesive capability.
# Splitting is only accepted when *both* resulting sides are long enough to
# plausibly stand alone as their own requirement (>= 2 words); otherwise the
# split is abandoned and the whole clause is kept intact, because a false
# split produces a fabricated requirement that can never legitimately be
# satisfied, while a missed split only under-decomposes a compound ask.
_AND_SEPARATOR = re.compile(r"(?i)\s*,?\s+and\s+\s*")


def _split_objective_clauses(objective: str) -> list[str]:
    text = (objective or "").strip()
    if not text:
        return []

    sentence_parts = [p.strip().rstrip(".").strip() for p in _UNAMBIGUOUS_SEPARATOR.split(text)]
    sentence_parts = [p for p in sentence_parts if p]

    clauses: list[str] = []
    for part in sentence_parts:
        and_parts = [p.strip().rstrip(".").strip() for p in _AND_SEPARATOR.split(part)]
        and_parts = [p for p in and_parts if p]
        if len(and_parts) > 1 and all(len(p.split()) >= 2 for p in and_parts):
            clauses.extend(and_parts)
        else:
            clauses.append(part)
    return clauses or ([text] if text else [])


def _content_tokens(text: str) -> set[str]:
    words = re.findall(r"[A-Za-z0-9_]{3,}", text.lower())
    return {w for w in words if w not in _STOPWORDS}


# A vague, generic ask ("implement a new feature", "fix the bug") gives no
# specific, checkable claim to hold the diff to, so it is verified leniently
# (any real change counts). A *specific* one names a concrete value to
# change to/from, a quoted identifier, or an explicit numeric target -- that
# is exactly the shape a "changed the wrong thing" defect (Attack 2) hides
# in, so it is always correlated strictly even when it is the task's only
# requirement.
_STRONG_SIGNAL = re.compile(r"(?i)\b\d+(?:\.\d+)?\b|[`\"'].+[`\"']|\bfrom\b.+\bto\b")


def _has_strong_signal(text: str) -> bool:
    return bool(_STRONG_SIGNAL.search(text))


def _clean_path_token(raw: str) -> str:
    return raw.strip().strip("`\"'").replace("\\", "/")


@dataclass
class Requirement:
    """A single, checkable statement derived from the task. Stable identity
    (``requirement_id``) is what lets evidence bind to a specific
    requirement rather than to the task as an undifferentiated whole."""
    requirement_id: str
    statement: str
    requirement_type: str = RequirementType.FUNCTIONAL.value
    importance: str = RequirementImportance.MUST.value
    verification_strategy: str = VerificationStrategy.DIFF_PRESENCE.value
    source: str = "user_task"  # user_task | plan | clarification
    target_paths: list[str] = field(default_factory=list)
    state: str = RequirementState.UNVERIFIED.value
    evidence_ids: list[str] = field(default_factory=list)
    unsatisfied_reason: str = ""
    clarification_id: str | None = None
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    def __post_init__(self) -> None:
        if len(self.statement) > MAX_STATEMENT_CHARS:
            self.statement = self.statement[: MAX_STATEMENT_CHARS - 3] + "..."
        if len(self.target_paths) > MAX_TARGET_PATHS_PER_REQUIREMENT:
            self.target_paths = self.target_paths[:MAX_TARGET_PATHS_PER_REQUIREMENT]
        if len(self.evidence_ids) > MAX_EVIDENCE_IDS_PER_REQUIREMENT:
            self.evidence_ids = self.evidence_ids[-MAX_EVIDENCE_IDS_PER_REQUIREMENT:]
        if len(self.unsatisfied_reason) > MAX_REASON_CHARS:
            self.unsatisfied_reason = self.unsatisfied_reason[: MAX_REASON_CHARS - 3] + "..."

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "statement": sanitize_text(self.statement),
            "requirement_type": self.requirement_type,
            "importance": self.importance,
            "verification_strategy": self.verification_strategy,
            "source": self.source,
            "target_paths": list(self.target_paths),
            "state": self.state,
            "evidence_ids": list(self.evidence_ids),
            "unsatisfied_reason": sanitize_text(self.unsatisfied_reason),
            "clarification_id": self.clarification_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Requirement:
        if not isinstance(data, dict):
            return cls(requirement_id="", statement="", state=RequirementState.UNVERIFIED.value)
        valid_types = {t.value for t in RequirementType}
        valid_importance = {i.value for i in RequirementImportance}
        valid_states = {s.value for s in RequirementState}
        valid_strategies = {s.value for s in VerificationStrategy}

        raw_type = str(data.get("requirement_type", RequirementType.FUNCTIONAL.value))
        raw_importance = str(data.get("importance", RequirementImportance.MUST.value))
        # A malformed/unknown state must fail closed, not silently become
        # SATISFIED -- the safe unknown-state default is UNVERIFIED.
        raw_state = str(data.get("state", RequirementState.UNVERIFIED.value))
        raw_strategy = str(data.get("verification_strategy", VerificationStrategy.MANUAL_CLARIFICATION.value))

        return cls(
            requirement_id=str(data.get("requirement_id", "")),
            statement=str(data.get("statement", "")),
            requirement_type=raw_type if raw_type in valid_types else RequirementType.FUNCTIONAL.value,
            importance=raw_importance if raw_importance in valid_importance else RequirementImportance.MUST.value,
            verification_strategy=raw_strategy if raw_strategy in valid_strategies else VerificationStrategy.MANUAL_CLARIFICATION.value,
            source=str(data.get("source", "user_task")),
            target_paths=[str(p) for p in (data.get("target_paths") or [])],
            state=raw_state if raw_state in valid_states else RequirementState.UNVERIFIED.value,
            evidence_ids=[str(e) for e in (data.get("evidence_ids") or [])],
            unsatisfied_reason=str(data.get("unsatisfied_reason", "")),
            clarification_id=data.get("clarification_id"),
            created_at=str(data.get("created_at", "")) or _now_iso(),
            updated_at=str(data.get("updated_at", "")) or _now_iso(),
        )


@dataclass
class TaskContract:
    """The durable, checkable decomposition of a task's objective. Persisted
    on the ``Task`` itself (not just the transient run report) so it survives
    across every ``run()``/multi-turn invocation for the task's lifetime,
    the same way ``Task.plan`` does."""
    task_id: str
    objective: str
    requirements: list[Requirement] = field(default_factory=list)
    non_goals: list[str] = field(default_factory=list)
    version: int = 1
    source: str = "derived_from_plan"
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    def __post_init__(self) -> None:
        if len(self.requirements) > MAX_REQUIREMENTS:
            self.requirements = self.requirements[:MAX_REQUIREMENTS]
        if len(self.non_goals) > MAX_NON_GOALS:
            self.non_goals = self.non_goals[:MAX_NON_GOALS]
        # Duplicate ids are a malformed-contract hazard (evidence could bind
        # to the wrong requirement) -- de-duplicate deterministically by
        # keeping the first occurrence and renumbering the rest.
        seen: set[str] = set()
        deduped: list[Requirement] = []
        for i, req in enumerate(self.requirements):
            rid = req.requirement_id or f"REQ-{i + 1:03d}"
            if rid in seen:
                rid = f"{rid}-DUP{i + 1}"
            seen.add(rid)
            deduped.append(replace(req, requirement_id=rid) if req.requirement_id != rid else req)
        self.requirements = deduped

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "objective": sanitize_text(self.objective),
            "requirements": [r.to_dict() for r in self.requirements],
            "non_goals": [sanitize_text(g) for g in self.non_goals],
            "version": self.version,
            "source": self.source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskContract:
        if not isinstance(data, dict):
            return cls(task_id="", objective="")
        raw_reqs = data.get("requirements") or []
        requirements = [
            Requirement.from_dict(r) for r in raw_reqs if isinstance(r, dict)
        ] if isinstance(raw_reqs, list) else []
        return cls(
            task_id=str(data.get("task_id", "")),
            objective=str(data.get("objective", "")),
            requirements=requirements,
            non_goals=[str(g) for g in (data.get("non_goals") or [])],
            version=int(data.get("version", 1) or 1),
            source=str(data.get("source", "derived_from_plan")),
            created_at=str(data.get("created_at", "")) or _now_iso(),
            updated_at=str(data.get("updated_at", "")) or _now_iso(),
        )


def derive_task_contract(task: Task, plan: Plan | None) -> TaskContract:
    """Deterministically decomposes a task's objective (and, once available,
    its plan) into a checkable requirement list.

    No provider call, no NLU model: a fixed set of pattern-based rules over
    text the orchestrator already has. This is intentionally conservative --
    it under-extracts (an implicit requirement it cannot recognize simply
    never becomes a tracked Requirement) rather than over-extracts, because a
    tracked MUST-requirement that can never be satisfied would itself become
    a false block on legitimate completions.
    """
    # Bound the text every extraction pattern below scans: an attacker- or
    # bug-supplied task.objective has no length limit of its own, and this
    # module must not become an unbounded-work sink over one.
    objective = (task.objective or "")[:MAX_OBJECTIVE_SCAN_CHARS]
    requirements: list[Requirement] = []
    seq = 0

    def next_id() -> str:
        nonlocal seq
        seq += 1
        return f"REQ-{seq:03d}"

    plan_files = sorted(set((plan.files_likely_to_change if plan else []) + (plan.files_likely_to_create if plan else [])))

    # Functional requirements come from the user's own objective, not the
    # plan's steps: a planner (especially a minimal/offline one) commonly
    # emits generic process boilerplate ("use the impact analysis to guide
    # the change") rather than user-facing deliverables, and turning every
    # such step into a MUST-requirement would block completion on content
    # that was never actually asked for. The objective is the one place a
    # compound, multi-part request ("Add CSV export and JSON export") is
    # reliably expressed in the user's own words.
    #
    # A single-clause objective becomes exactly one requirement, verified
    # leniently (see RequirementAssessmentEngine): with nothing else to
    # disambiguate against, "some change was made" is the correct bar --
    # this is what an ordinary, non-compound task looks like, and it must
    # not regress to demanding a content-correlation match against
    # boilerplate. Only once an objective genuinely names multiple distinct
    # parts does the stricter per-clause correlation check apply, because
    # that is exactly the shape Attack 1 (silently skipping part of a
    # multi-part request) exploits.
    clauses = _split_objective_clauses(objective)
    for clause in clauses[:MAX_REQUIREMENTS]:
        # A clause that is itself a documentation/testing ask gets its own,
        # more reliable dedicated requirement below (path-pattern or
        # evidence-backed, rather than literal word-in-diff correlation --
        # a real doc update's diff essentially never contains the literal
        # word "documentation"). Adding it here too would create a second,
        # harder-to-satisfy requirement duplicating the same ask.
        if _DOC_HINT.search(clause) or _TEST_HINT.search(clause):
            continue
        requirements.append(Requirement(
            requirement_id=next_id(),
            statement=clause,
            requirement_type=RequirementType.FUNCTIONAL.value,
            importance=RequirementImportance.MUST.value,
            verification_strategy=VerificationStrategy.DIFF_PRESENCE.value,
            source="user_task",
            target_paths=plan_files,
        ))

    # Explicit negative requirements ("do not modify X").
    scan_text = objective
    for match in _CONSTRAINT_TRIGGERS.finditer(scan_text):
        path_token = _clean_path_token(match.group(1))
        if not path_token:
            continue
        requirements.append(Requirement(
            requirement_id=next_id(),
            statement=match.group(0).strip(),
            requirement_type=RequirementType.CONSTRAINT.value,
            importance=RequirementImportance.MUST.value,
            verification_strategy=VerificationStrategy.CONSTRAINT_ABSENCE.value,
            source="user_task",
            target_paths=[path_token],
        ))

    # "Preserve/keep/maintain X" -- tracked only when X itself names a
    # compatibility/regression concern (API, interface, behavior, contract),
    # since that is the shape that cannot be auto-verified and must not be
    # silently assumed satisfied. "Keep addition correct" or "keep it
    # simple" are not concrete, checkable asks distinct from the task's own
    # functional requirement already covering that same code -- treating
    # every such turn of phrase as its own permanently-unverifiable MUST
    # requirement would block completion on ordinary, ambient phrasing that
    # was never a distinct second requirement to begin with.
    for match in _PRESERVE_TRIGGER.finditer(scan_text):
        clause = match.group(1).strip()
        if not clause or len(clause) > 120:
            continue
        if not re.search(r"(?i)\bapi\b|compat|interface|contract|behavior|behaviour|backward", clause):
            continue
        req_type = RequirementType.COMPATIBILITY.value
        requirements.append(Requirement(
            requirement_id=next_id(),
            statement=match.group(0).strip(),
            requirement_type=req_type,
            importance=RequirementImportance.MUST.value,
            verification_strategy=VerificationStrategy.MANUAL_CLARIFICATION.value,
            source="user_task",
        ))

    # Explicit documentation / testing asks.
    if _DOC_HINT.search(objective):
        requirements.append(Requirement(
            requirement_id=next_id(),
            statement="Update documentation as requested.",
            requirement_type=RequirementType.DOCUMENTATION.value,
            importance=RequirementImportance.MUST.value,
            verification_strategy=VerificationStrategy.DIFF_PRESENCE.value,
            source="user_task",
            target_paths=[p for p in plan_files if re.search(r"(?i)readme|\.md$|/docs?/", p)] or ["README.md", "docs/"],
        ))
    if _TEST_HINT.search(objective):
        requirements.append(Requirement(
            requirement_id=next_id(),
            statement="Provide passing test coverage as requested.",
            requirement_type=RequirementType.TESTING.value,
            importance=RequirementImportance.MUST.value,
            verification_strategy=VerificationStrategy.TEST_EVIDENCE.value,
            source="user_task",
        ))

    return TaskContract(
        task_id=task.task_id,
        objective=objective,
        requirements=requirements[:MAX_REQUIREMENTS],
        source="derived_from_objective",
    )


# ---------------------------------------------------------------------------
# Requirement satisfaction assessment
# ---------------------------------------------------------------------------

@dataclass
class RequirementSatisfactionAssessment:
    task_id: str
    requirements: list[Requirement] = field(default_factory=list)
    satisfied: bool = False
    unsatisfied_requirement_ids: list[str] = field(default_factory=list)
    failed_requirement_ids: list[str] = field(default_factory=list)
    decision_reason: str = ""
    assessed_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "requirements": [r.to_dict() for r in self.requirements],
            "satisfied": self.satisfied,
            "unsatisfied_requirement_ids": list(self.unsatisfied_requirement_ids),
            "failed_requirement_ids": list(self.failed_requirement_ids),
            "decision_reason": sanitize_text(self.decision_reason),
            "assessed_at": self.assessed_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RequirementSatisfactionAssessment:
        if not isinstance(data, dict):
            return cls(task_id="", satisfied=False, decision_reason="Invalid data")
        raw_reqs = data.get("requirements") or []
        requirements = [Requirement.from_dict(r) for r in raw_reqs if isinstance(r, dict)] if isinstance(raw_reqs, list) else []
        return cls(
            task_id=str(data.get("task_id", "")),
            requirements=requirements,
            satisfied=bool(data.get("satisfied", False)),
            unsatisfied_requirement_ids=[str(i) for i in (data.get("unsatisfied_requirement_ids") or [])],
            failed_requirement_ids=[str(i) for i in (data.get("failed_requirement_ids") or [])],
            decision_reason=str(data.get("decision_reason", "")),
            assessed_at=str(data.get("assessed_at", "")),
        )


class RequirementAssessmentEngine:
    """Deterministic evaluator: turns a TaskContract plus authoritative
    current state (changeset, evidence store, live review, clarifications)
    into a RequirementSatisfactionAssessment. Never consults provider or
    reviewer *text* for a verdict -- only the structural facts already
    trusted by the Phase 4.18-4.20 completion engine."""

    def __init__(self, filesystem: ProjectFilesystem):
        self.filesystem = filesystem

    def assess(
        self,
        contract: TaskContract,
        evidence_store: CompletionEvidenceStore,
        applied_operations: Sequence[FileOperation],
        current_diff: str,
        last_review: ReviewResult | None = None,
        clarification_requests: Sequence[Any] | None = None,
    ) -> RequirementSatisfactionAssessment:
        changed_paths = {
            str(getattr(op, "path", "")).replace("\\", "/")
            for op in applied_operations if getattr(op, "path", "")
        }
        diff_text = current_diff or ""

        clarified_by_id: dict[str, Any] = {}
        for req_c in clarification_requests or []:
            qid = req_c.question_id if hasattr(req_c, "question_id") else (req_c.get("question_id") if isinstance(req_c, dict) else None)
            if qid:
                clarified_by_id[str(qid)] = req_c

        # A FUNCTIONAL/DIFF_PRESENCE requirement is held to strict content
        # correlation when either (a) it has functional siblings to be
        # distinguished from -- a compound request (Attack 1) -- or (b) its
        # own statement names something specific enough to be individually
        # checkable (a number, a quoted value, an explicit "from X to Y") --
        # a single but specific request (Attack 2). A single *vague*
        # requirement (the ordinary, non-compound task -- "implement a new
        # feature") has nothing to disambiguate or correlate against, and is
        # satisfied by any real change, matching the technical engine's own
        # "workspace diff non-empty" gate rather than second-guessing it.
        # Non-functional DIFF_PRESENCE requirements (documentation) are
        # always checked on their own, more reliable terms (see
        # _assess_one) and do not affect this count.
        functional_count = sum(
            1 for r in contract.requirements
            if r.requirement_type == RequirementType.FUNCTIONAL.value
            and r.verification_strategy == VerificationStrategy.DIFF_PRESENCE.value
        )

        results: list[Requirement] = []
        for req in contract.requirements:
            lenient = functional_count <= 1 and not _has_strong_signal(req.statement)
            state, reason, evidence_ids = self._assess_one(
                req, changed_paths, diff_text, evidence_store, last_review, clarified_by_id,
                lenient,
            )
            results.append(replace(
                req,
                state=state.value,
                unsatisfied_reason=reason,
                evidence_ids=(req.evidence_ids + evidence_ids)[-MAX_EVIDENCE_IDS_PER_REQUIREMENT:],
                updated_at=_now_iso(),
            ))
            # Durable, auditable trace of the assessment itself, reusing the
            # same evidence store / trust-tier vocabulary as the technical
            # completion engine rather than a parallel logging mechanism.
            evidence_store.record(
                task_id=contract.task_id,
                subtask_id=req.requirement_id,
                turn_number=0,
                stage="requirement_assessment",
                evidence_type=EvidenceType.CONTRACT_COMPLIANCE,
                source="requirement_assessment_engine",
                trust_tier=EvidenceTrustTier.SYSTEM_INTEGRITY,
                target_paths=req.target_paths,
                payload={"requirement_id": req.requirement_id, "state": state.value, "reason": reason},
            )

        unsatisfied_must = [
            r for r in results
            if r.importance == RequirementImportance.MUST.value
            and r.state not in (RequirementState.SATISFIED.value, RequirementState.NOT_APPLICABLE.value)
        ]
        failed = [r for r in results if r.state == RequirementState.FAILED.value]
        satisfied = len(unsatisfied_must) == 0

        if satisfied:
            reason = "All MUST-importance requirements are satisfied or not applicable"
        else:
            names = ", ".join(f"{r.requirement_id} ({r.state})" for r in unsatisfied_must[:5])
            reason = f"{len(unsatisfied_must)} MUST-importance requirement(s) unresolved: {names}"

        return RequirementSatisfactionAssessment(
            task_id=contract.task_id,
            requirements=results,
            satisfied=satisfied,
            unsatisfied_requirement_ids=[r.requirement_id for r in unsatisfied_must],
            failed_requirement_ids=[r.requirement_id for r in failed],
            decision_reason=reason,
        )

    def _assess_one(
        self,
        req: Requirement,
        changed_paths: set[str],
        diff_text: str,
        evidence_store: CompletionEvidenceStore,
        last_review: ReviewResult | None,
        clarified_by_id: dict[str, Any],
        lenient_diff_presence: bool,
    ) -> tuple[RequirementState, str, list[str]]:
        # A BLOCKED requirement pinned to a clarification is only ever
        # unblocked by that specific clarification being answered -- a
        # provider cannot route around it via any other evidence.
        if req.clarification_id:
            clar = clarified_by_id.get(req.clarification_id)
            status = getattr(clar, "status", None) if clar is not None else None
            if status is None and isinstance(clar, dict):
                status = clar.get("status")
            if status != "answered":
                return RequirementState.BLOCKED, "Awaiting clarification response", []

        strategy = req.verification_strategy

        if strategy == VerificationStrategy.CONSTRAINT_ABSENCE.value:
            norm_targets = {p.replace("\\", "/") for p in req.target_paths}
            violated = sorted(p for p in norm_targets if any(p in c or c in p for c in changed_paths))
            if violated:
                return RequirementState.FAILED, f"Constraint violated: modified {', '.join(violated)}", []
            return RequirementState.SATISFIED, "", []

        if strategy == VerificationStrategy.TEST_EVIDENCE.value:
            valid_tests = evidence_store.get_valid_evidence(EvidenceType.TEST_EXECUTION)
            passing = [e for e in valid_tests if e.exit_code == 0]
            failing = [e for e in valid_tests if e.exit_code != 0]
            if failing:
                return RequirementState.FAILED, f"{len(failing)} active test failure(s)", [e.evidence_id for e in failing]
            if passing:
                return RequirementState.SATISFIED, "", [e.evidence_id for e in passing]
            return RequirementState.UNVERIFIED, "No passing test evidence on current workspace state", []

        if strategy == VerificationStrategy.DIFF_PRESENCE.value:
            if not changed_paths and not diff_text.strip():
                # Mirrors the technical engine's own "no_changes_needed" gate
                # (completion.py Gate 2): a live APPROVED review is itself the
                # authoritative signal that the current, unmodified workspace
                # already satisfies the request -- not a claim this engine
                # invents on its own, but the same evidence the technical
                # layer already requires and trusts for exactly this case.
                if last_review is not None and last_review.verdict == "APPROVED":
                    return RequirementState.SATISFIED, "", []
                return RequirementState.UNVERIFIED, "No workspace changes to evaluate against this requirement", []

            if req.requirement_type == RequirementType.DOCUMENTATION.value:
                # Content-correlating against a fixed, generic statement
                # ("update documentation as requested") is unreliable --
                # what actually proves a documentation requirement is a
                # changed path that looks like documentation.
                doc_touched = any(
                    re.search(r"(?i)readme|changelog|\.md$|/docs?/", p) for p in changed_paths
                )
                if doc_touched:
                    return RequirementState.SATISFIED, "", []
                return RequirementState.UNVERIFIED, "No documentation-like path was modified", []

            if lenient_diff_presence:
                # No sibling functional requirement to disambiguate from --
                # a real change is itself sufficient evidence.
                return RequirementState.SATISFIED, "", []
            tokens = _content_tokens(req.statement)
            if not tokens:
                # Nothing distinctive to correlate against (e.g. an empty or
                # entirely-stopword statement) -- a real change existing at
                # all is the only signal available.
                return (RequirementState.SATISFIED, "", []) if changed_paths else (
                    RequirementState.UNVERIFIED, "No workspace changes to evaluate against this requirement", []
                )
            haystack = " ".join(sorted(changed_paths)).lower() + "\n" + diff_text.lower()
            found = sum(1 for t in tokens if t in haystack)
            required = len(tokens) if len(tokens) <= 4 else math.ceil(len(tokens) * 0.75)
            if found >= required:
                return RequirementState.SATISFIED, "", []
            return (
                RequirementState.UNVERIFIED,
                f"Changeset does not evidently address this requirement ({found}/{len(tokens)} content terms found)",
                [],
            )

        # MANUAL_CLARIFICATION (or any strategy this engine does not know how
        # to auto-verify): fails closed to UNVERIFIED rather than assuming
        # satisfaction. A requirement can only leave this state via an
        # explicit clarification binding (handled above) or contract
        # amendment -- never implicitly.
        return RequirementState.UNVERIFIED, "No deterministic verification strategy could confirm this requirement", []
