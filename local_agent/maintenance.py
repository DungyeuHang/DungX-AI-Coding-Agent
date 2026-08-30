"""Phase 4.21: the persistent maintenance intelligence model.

This module holds *data*, not authority. It defines the normalised,
bounded, serialisable representation of a maintenance opportunity - what was
observed, where, how often, how confident we are, and what state the
opportunity is in - plus the hierarchical budget ledger that bounds any work
derived from it.

Three properties are load-bearing and are asserted by the test-suite rather
than merely documented here:

* **Deterministic identity.** A candidate's id is a hash of its kind and its
  normalised subject, so the same underlying problem observed on two different
  days is the *same* candidate whose occurrence count grows, not two
  candidates that both look new.
* **Bounded everything.** Every persistent collection has an explicit cap and
  an explicit eviction rule. Maintenance intelligence accumulates for the
  lifetime of a repository; a single unbounded list here would eventually be
  the largest file the agent writes.
* **Untrusted content.** Candidate text and paths are treated as hostile
  input, because some of it is derived from subprocess output, filenames and
  historical records that a previous run wrote. Everything is sanitised on the
  way in *and* on the way out of serialisation.

Nothing in this module imports :mod:`local_agent.validation_decision`,
:mod:`local_agent.approval` or :mod:`local_agent.tool_engine`, and that
absence is proved structurally in the tests: maintenance intelligence is
advisory, and the only way to keep it advisory is to make it structurally
incapable of reaching the authorities it must not override.
"""

from __future__ import annotations

import datetime
import hashlib
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .semantic_impact import escapes_root, normalize_relative

MAINTENANCE_SCHEMA_VERSION = 1

# -- bounds -------------------------------------------------------------------
#
# Deliberately conservative. These are the *storage* bounds; the per-run work
# bounds live in MaintenanceBudget below and are far tighter.

DEFAULT_MAX_CANDIDATES = 300
DEFAULT_MAX_RUNS = 50
MAX_TEXT_CHARS = 400
MAX_LIST_ENTRIES = 20
MAX_EVIDENCE_REFS = 20
MAX_HISTORY_ENTRIES = 30
MAX_METRIC_ENTRIES = 20

#: Repository-relative paths the maintenance layer must never propose changing.
#: This is a floor, not the whole protection story: the real enforcement lives
#: in :mod:`local_agent.coding_agent` / :mod:`local_agent.filesystem`, which the
#: maintenance layer cannot reach past. Duplicating the two most sensitive
#: entries here means a dangerous candidate is refused at *triage* time, before
#: anything is planned, rather than only at write time.
PROTECTED_RELATIVE_PATHS: frozenset[str] = frozenset(
    {
        "local_agent/tool_engine.py",
        "local_agent/approval.py",
    }
)

#: Directory names that are never legitimate maintenance targets.
PROTECTED_DIRECTORY_NAMES: frozenset[str] = frozenset(
    {".git", ".venv", "venv", "node_modules", "__pycache__"}
)

#: Case-folded views of the two protection sets.
#:
#: Every comparison against them is case-insensitive, because the agent's two
#: primary platforms (Windows, macOS) have case-insensitive filesystems: there,
#: ``Local_Agent/Tool_Engine.py`` and ``.GIT/config`` name exactly the same
#: objects as their lowercase spellings, and a case-sensitive membership test
#: would let a candidate walk straight past the protection floor. Folding is
#: strictly more conservative than not folding, so it is applied on every
#: platform rather than being made platform-dependent - a candidate loaded from
#: a store written on Windows must be refused when it is read on Linux too.
_PROTECTED_RELATIVE_PATHS_FOLDED: frozenset[str] = frozenset(
    path.casefold() for path in PROTECTED_RELATIVE_PATHS
)
_PROTECTED_DIRECTORY_NAMES_FOLDED: frozenset[str] = frozenset(
    name.casefold() for name in PROTECTED_DIRECTORY_NAMES
)


def is_protected_directory_segment(segment: Any) -> bool:
    """True when ``segment`` names a protected directory, ignoring case."""
    return str(segment).casefold() in _PROTECTED_DIRECTORY_NAMES_FOLDED


def is_protected_relative_path(path: Any, *, extra: Iterable[str] = ()) -> bool:
    """True when ``path`` is on the protected floor, ignoring case.

    ``extra`` lets a caller add its own protected paths without having to
    re-implement the folding rule; they are folded here on the way in.
    """
    folded = str(path).casefold()
    if folded in _PROTECTED_RELATIVE_PATHS_FOLDED:
        return True
    return any(folded == str(item).casefold() for item in extra)


# -- signal taxonomy ----------------------------------------------------------


class MaintenanceSignal:
    """The kinds of maintenance opportunity this build can actually detect.

    Every member here is backed by a real extractor in
    :mod:`local_agent.maintenance_analysis` that reads a real subsystem's
    output. Speculative signal kinds are deliberately absent: a taxonomy entry
    with no extractor behind it is a claim the system cannot honour.
    """

    RECURRING_DEFECT = "recurring_defect"
    REPEATED_REPAIR = "repeated_repair"
    ABANDONED_WORK = "abandoned_work"
    CANDIDATE_INSTABILITY = "candidate_instability"
    BROAD_VALIDATION_PRESSURE = "broad_validation_pressure"
    EVIDENCE_REUSE_FAILURE = "evidence_reuse_failure"
    FALSE_CONFIDENCE = "false_confidence"
    ANALYSIS_DEGRADATION = "analysis_degradation"
    ARCHITECTURAL_RISK = "architectural_risk"
    ANALYZER_BLIND_SPOT = "analyzer_blind_spot"
    PARSE_FAILURE = "parse_failure"
    TEST_GAP = "test_gap"
    KNOWN_FAILURE_PATTERN = "known_failure_pattern"


ALL_SIGNAL_KINDS: tuple[str, ...] = (
    MaintenanceSignal.RECURRING_DEFECT,
    MaintenanceSignal.REPEATED_REPAIR,
    MaintenanceSignal.ABANDONED_WORK,
    MaintenanceSignal.CANDIDATE_INSTABILITY,
    MaintenanceSignal.BROAD_VALIDATION_PRESSURE,
    MaintenanceSignal.EVIDENCE_REUSE_FAILURE,
    MaintenanceSignal.FALSE_CONFIDENCE,
    MaintenanceSignal.ANALYSIS_DEGRADATION,
    MaintenanceSignal.ARCHITECTURAL_RISK,
    MaintenanceSignal.ANALYZER_BLIND_SPOT,
    MaintenanceSignal.PARSE_FAILURE,
    MaintenanceSignal.TEST_GAP,
    MaintenanceSignal.KNOWN_FAILURE_PATTERN,
)

#: Which subsystem a signal is derived from. Provenance is not decoration: an
#: operator asking "why do you think this?" is asking this question, and a
#: candidate whose provenance is unknown is one whose evidence cannot be
#: re-checked at reassessment time.
PROVENANCE_LIFECYCLE = "validation_lifecycle"
PROVENANCE_TELEMETRY = "validation_telemetry"
PROVENANCE_SEMANTIC_GRAPH = "semantic_graph"
PROVENANCE_KNOWLEDGE_GRAPH = "knowledge_graph"
PROVENANCE_REPOSITORY = "repository"

ALL_PROVENANCES: tuple[str, ...] = (
    PROVENANCE_LIFECYCLE,
    PROVENANCE_TELEMETRY,
    PROVENANCE_SEMANTIC_GRAPH,
    PROVENANCE_KNOWLEDGE_GRAPH,
    PROVENANCE_REPOSITORY,
)


# -- severity -----------------------------------------------------------------

SEVERITY_INFO = "info"
SEVERITY_LOW = "low"
SEVERITY_MEDIUM = "medium"
SEVERITY_HIGH = "high"
SEVERITY_CRITICAL = "critical"

SEVERITY_ORDER: tuple[str, ...] = (
    SEVERITY_INFO,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
    SEVERITY_HIGH,
    SEVERITY_CRITICAL,
)


def severity_rank(severity: Any) -> int:
    """Rank of ``severity``; an unrecognised value ranks *lowest*.

    Unknown severities come from corrupted or hostile records. Ranking them at
    the bottom means a forged ``severity: "catastrophic"`` cannot jump the
    queue - the worst it can do is sort last.
    """
    try:
        return SEVERITY_ORDER.index(str(severity))
    except ValueError:
        return 0


def highest_severity(*severities: Any) -> str:
    """The most severe of the arguments, defaulting to INFO when empty."""
    best = SEVERITY_INFO
    for severity in severities:
        if severity_rank(severity) > severity_rank(best):
            best = SEVERITY_ORDER[severity_rank(severity)]
    return best


# -- candidate lifecycle ------------------------------------------------------


class CandidateState:
    """Where a candidate is in the maintenance lifecycle.

    Distinct from :class:`ReassessmentOutcome`: the *state* says how far the
    machinery got, the *outcome* says what the evidence showed afterwards. A
    candidate can reach ``REASSESSED`` with outcome ``PERSISTING`` - the
    machinery finished, the problem did not go away - and conflating those two
    axes is exactly the mistake Part 9 of the specification forbids.
    """

    DETECTED = "detected"
    TRIAGED = "triaged"
    SELECTED = "selected"
    PLANNED = "planned"
    EXECUTING = "executing"
    VALIDATED = "validated"
    REASSESSED = "reassessed"
    DEFERRED = "deferred"
    REJECTED = "rejected"
    BLOCKED = "blocked"


ALL_CANDIDATE_STATES: tuple[str, ...] = (
    CandidateState.DETECTED,
    CandidateState.TRIAGED,
    CandidateState.SELECTED,
    CandidateState.PLANNED,
    CandidateState.EXECUTING,
    CandidateState.VALIDATED,
    CandidateState.REASSESSED,
    CandidateState.DEFERRED,
    CandidateState.REJECTED,
    CandidateState.BLOCKED,
)

#: Terminal-for-this-run states. A candidate in one of these is not carried
#: further by the current run; it may still be re-detected by a later scan,
#: which is the point - maintenance problems recur.
TERMINAL_CANDIDATE_STATES: frozenset[str] = frozenset(
    {
        CandidateState.REASSESSED,
        CandidateState.DEFERRED,
        CandidateState.REJECTED,
        CandidateState.BLOCKED,
    }
)

#: Legal state transitions. Modelled on the Phase 4.20 lifecycle machine for
#: the same reason: an illegal transition is a bug, and a state machine that
#: silently accepts one hides it.
ALLOWED_CANDIDATE_TRANSITIONS: dict[str, frozenset[str]] = {
    CandidateState.DETECTED: frozenset(
        {
            CandidateState.TRIAGED,
            CandidateState.DEFERRED,
            CandidateState.REJECTED,
            CandidateState.BLOCKED,
        }
    ),
    CandidateState.TRIAGED: frozenset(
        {
            CandidateState.SELECTED,
            CandidateState.DEFERRED,
            CandidateState.REJECTED,
            CandidateState.BLOCKED,
        }
    ),
    CandidateState.SELECTED: frozenset(
        {
            CandidateState.PLANNED,
            CandidateState.DEFERRED,
            CandidateState.REJECTED,
            CandidateState.BLOCKED,
        }
    ),
    CandidateState.PLANNED: frozenset(
        {
            CandidateState.EXECUTING,
            CandidateState.DEFERRED,
            CandidateState.BLOCKED,
        }
    ),
    CandidateState.EXECUTING: frozenset(
        {
            CandidateState.VALIDATED,
            CandidateState.BLOCKED,
            CandidateState.REASSESSED,
        }
    ),
    CandidateState.VALIDATED: frozenset(
        {CandidateState.REASSESSED, CandidateState.BLOCKED}
    ),
    CandidateState.REASSESSED: frozenset(),
    CandidateState.DEFERRED: frozenset(),
    CandidateState.REJECTED: frozenset(),
    CandidateState.BLOCKED: frozenset(),
}


class InvalidCandidateTransition(ValueError):
    """Raised when a caller asks for a transition the machine forbids."""

    def __init__(self, current: str, requested: str):
        super().__init__(
            f"cannot move a maintenance candidate from '{current}' to '{requested}'"
        )
        self.current = current
        self.requested = requested


def can_transition_candidate(current: Any, requested: Any) -> bool:
    """True when ``current -> requested`` is a legal candidate transition."""
    return str(requested) in ALLOWED_CANDIDATE_TRANSITIONS.get(str(current), frozenset())


# -- reassessment outcomes ----------------------------------------------------


class ReassessmentOutcome:
    """What the *evidence* said after maintenance work finished.

    ``PENDING`` is the honest default. A candidate that has never been
    reassessed has no outcome, and defaulting to anything else - especially
    anything that reads as success - would let "we did some work" masquerade
    as "the problem is gone".
    """

    PENDING = "pending"
    RESOLVED = "resolved"
    PARTIALLY_RESOLVED = "partially_resolved"
    PERSISTING = "persisting"
    REGRESSED = "regressed"
    INCONCLUSIVE = "inconclusive"
    BLOCKED = "blocked"


ALL_REASSESSMENT_OUTCOMES: tuple[str, ...] = (
    ReassessmentOutcome.PENDING,
    ReassessmentOutcome.RESOLVED,
    ReassessmentOutcome.PARTIALLY_RESOLVED,
    ReassessmentOutcome.PERSISTING,
    ReassessmentOutcome.REGRESSED,
    ReassessmentOutcome.INCONCLUSIVE,
    ReassessmentOutcome.BLOCKED,
)

#: Outcomes that mean the original signal genuinely improved. Used by the
#: learning layer; note that ``PARTIALLY_RESOLVED`` is *not* here, because a
#: partially-resolved candidate is evidence the fix was incomplete, and
#: counting it as a success would inflate every actionability rate.
SUCCESSFUL_OUTCOMES: frozenset[str] = frozenset({ReassessmentOutcome.RESOLVED})


# -- sanitisation -------------------------------------------------------------

_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE_RUN = re.compile(r"\s+")


def sanitize_text(value: Any, *, limit: int = MAX_TEXT_CHARS) -> str:
    """Render ``value`` as a bounded, single-line, control-character-free string.

    Applied to every free-text field on the way in. Maintenance text is derived
    from filenames, exception messages and prior persisted records, none of
    which are trustworthy: a candidate title containing ANSI escapes or a
    newline-injected fake log line would otherwise be printed verbatim by the
    CLI.
    """
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    text = _CONTROL_CHARACTERS.sub("", text)
    text = _WHITESPACE_RUN.sub(" ", text).strip()
    limit = max(1, int(limit))
    if len(text) > limit:
        text = text[: limit - 3].rstrip() + "..."
    return text


def sanitize_relative_path(value: Any) -> str:
    """Normalise ``value`` to a safe repository-relative path, or ``""``.

    Returning the empty string rather than raising is deliberate: a single
    hostile path inside an otherwise useful candidate should drop that path,
    not destroy the candidate. Callers filter empties out.

    Rejects, in order: non-strings, absolute paths, Windows drive-qualified
    paths, UNC paths, anything that normalises to an escape from the root, and
    anything whose first segment is a protected directory. Embedded NUL bytes
    are stripped before any other check, since some lower-level filesystem
    APIs historically truncate a string at the first NUL.
    """
    if not isinstance(value, str):
        return ""
    raw = value.replace("\x00", "").strip()
    if not raw:
        return ""
    if raw.startswith("\\\\"):  # UNC
        return ""
    if len(raw) >= 2 and raw[1] == ":":  # drive-qualified
        return ""
    normalised = normalize_relative(raw)
    if not normalised:
        return ""
    if normalised.startswith("/"):
        return ""
    if escapes_root(normalised):
        return ""
    segments = normalised.split("/")
    if any(is_protected_directory_segment(segment) for segment in segments):
        return ""
    if any(segment in {"", ".", ".."} for segment in segments):
        return ""
    return normalised


def sanitize_path_list(values: Any, *, limit: int = MAX_LIST_ENTRIES) -> list[str]:
    """Sanitise and de-duplicate a path collection, sorted and bounded."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        return []
    cleaned = {sanitize_relative_path(item) for item in values}
    cleaned.discard("")
    return sorted(cleaned)[: max(0, int(limit))]


def sanitize_string_list(values: Any, *, limit: int = MAX_LIST_ENTRIES) -> list[str]:
    """Sanitise a free-text collection preserving order, de-duplicated."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        return []
    seen: set[str] = set()
    result: list[str] = []
    for item in values:
        text = sanitize_text(item)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if len(result) >= max(0, int(limit)):
            break
    return result


def sanitize_metrics(values: Any, *, limit: int = MAX_METRIC_ENTRIES) -> dict[str, float]:
    """Coerce a mapping to bounded ``str -> finite float``.

    Non-finite values (``inf``, ``nan``) are dropped rather than clamped. They
    only arrive from corrupted persistence, and a NaN that survives into the
    priority engine silently poisons every comparison it touches.
    """
    if not isinstance(values, Mapping):
        return {}
    result: dict[str, float] = {}
    for key, raw in sorted(values.items(), key=lambda item: str(item[0])):
        try:
            number = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if number != number or number in (float("inf"), float("-inf")):
            continue
        name = sanitize_text(key, limit=64)
        if not name:
            continue
        result[name] = number
        if len(result) >= max(0, int(limit)):
            break
    return result


def clamp_unit(value: Any, *, default: float = 0.0) -> float:
    """Coerce to a float in ``[0.0, 1.0]``, falling back to ``default``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN
        return default
    return max(0.0, min(1.0, number))


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def candidate_identity(kind: Any, subject: Any) -> str:
    """A stable 16-hex-character id for ``(kind, subject)``.

    Determinism is what makes recurrence measurable: the same weak module
    detected in January and in June must hash to the same candidate so that
    ``occurrence_count`` means something. The subject is normalised (sorted,
    lowercased, whitespace-collapsed) so that incidental ordering differences
    in the extractor do not fork the identity.
    """
    kind_text = sanitize_text(kind, limit=64).lower()
    if isinstance(subject, (list, tuple, set, frozenset)):
        parts = sorted(sanitize_text(item, limit=200).lower() for item in subject)
        subject_text = "|".join(part for part in parts if part)
    else:
        subject_text = sanitize_text(subject, limit=400).lower()
    payload = f"{kind_text}\x1f{subject_text}".encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()[:16]


# -- the candidate ------------------------------------------------------------


@dataclass
class MaintenanceCandidate:
    """One normalised maintenance opportunity.

    Constructed by extractors, ranked by the priority engine, gated by the
    execution policy, and re-checked by the reassessment loop. It carries no
    behaviour that could act on the repository - deliberately: a data class
    cannot be tricked into executing anything.
    """

    kind: str = MaintenanceSignal.ARCHITECTURAL_RISK
    subject: str = ""
    candidate_id: str = ""
    title: str = ""
    detail: str = ""
    provenance: str = PROVENANCE_REPOSITORY
    severity: str = SEVERITY_LOW
    confidence: float = 0.0
    #: How many independent observations back ``confidence``. Reported next to
    #: it everywhere, because a confidence of 1.0 from two samples and one from
    #: two hundred are not the same claim.
    sample_size: int = 0
    uncertainty: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    affected_files: list[str] = field(default_factory=list)
    affected_symbols: list[str] = field(default_factory=list)
    recommended_action: str = ""
    estimated_effort: float = 1.0
    occurrence_count: int = 1
    first_seen_at: str = field(default_factory=_now)
    last_seen_at: str = field(default_factory=_now)
    state: str = CandidateState.DETECTED
    outcome: str = ReassessmentOutcome.PENDING
    causal_links: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    history: list[dict[str, str]] = field(default_factory=list)
    #: Rejected/blocked candidates carry why, so the CLI can answer "why is
    #: this not being worked on?" without re-deriving the policy verdict.
    blocked_reasons: list[str] = field(default_factory=list)
    #: How many times a *run* has attempted this candidate. Repeated failure is
    #: itself a policy signal (Part 17, case 7).
    attempt_count: int = 0
    failure_count: int = 0

    def __post_init__(self) -> None:
        self.kind = (
            self.kind if self.kind in ALL_SIGNAL_KINDS else MaintenanceSignal.ARCHITECTURAL_RISK
        )
        self.provenance = (
            self.provenance if self.provenance in ALL_PROVENANCES else PROVENANCE_REPOSITORY
        )
        self.subject = sanitize_text(self.subject)
        self.title = sanitize_text(self.title) or self.kind.replace("_", " ")
        self.detail = sanitize_text(self.detail)
        self.recommended_action = sanitize_text(self.recommended_action)
        self.severity = (
            self.severity if self.severity in SEVERITY_ORDER else SEVERITY_LOW
        )
        self.state = self.state if self.state in ALL_CANDIDATE_STATES else CandidateState.DETECTED
        self.outcome = (
            self.outcome if self.outcome in ALL_REASSESSMENT_OUTCOMES else ReassessmentOutcome.PENDING
        )
        self.confidence = clamp_unit(self.confidence)
        self.sample_size = max(0, _safe_int(self.sample_size))
        self.occurrence_count = max(1, _safe_int(self.occurrence_count, default=1))
        self.attempt_count = max(0, _safe_int(self.attempt_count))
        self.failure_count = max(0, _safe_int(self.failure_count))
        self.estimated_effort = max(0.0, min(100.0, _safe_float(self.estimated_effort, 1.0)))
        self.uncertainty = sanitize_string_list(self.uncertainty)
        self.evidence_refs = sanitize_string_list(self.evidence_refs, limit=MAX_EVIDENCE_REFS)
        self.affected_files = sanitize_path_list(self.affected_files)
        self.affected_symbols = sanitize_string_list(self.affected_symbols)
        self.causal_links = sanitize_string_list(self.causal_links)
        self.metrics = sanitize_metrics(self.metrics)
        self.blocked_reasons = sanitize_string_list(self.blocked_reasons)
        self.history = _sanitize_history(self.history)
        self.first_seen_at = sanitize_text(self.first_seen_at, limit=64) or _now()
        self.last_seen_at = sanitize_text(self.last_seen_at, limit=64) or self.first_seen_at
        if not self.candidate_id:
            self.candidate_id = candidate_identity(self.kind, self.subject or self.title)
        else:
            self.candidate_id = sanitize_text(self.candidate_id, limit=64)

    # -- derived -----------------------------------------------------------

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_CANDIDATE_STATES

    @property
    def touches_protected_path(self) -> bool:
        """True when any affected file is on the protected floor.

        Note this reads ``affected_files`` *after* sanitisation, so a path that
        was rejected outright (traversal, absolute, ``.git``) never reaches
        here - it simply is not in the list any more. Both conditions are
        checked separately by :mod:`local_agent.maintenance_policy`.
        """
        return any(is_protected_relative_path(path) for path in self.affected_files)

    @property
    def failure_ratio(self) -> float:
        if self.attempt_count <= 0:
            return 0.0
        return min(1.0, self.failure_count / float(self.attempt_count))

    def describe(self) -> str:
        return (
            f"{self.kind}[{self.severity}] {self.title} "
            f"(x{self.occurrence_count}, confidence {self.confidence:.2f} "
            f"from {self.sample_size} sample(s))"
        )

    # -- mutation ----------------------------------------------------------

    def transition(self, requested: str, *, reason: str = "") -> str:
        """Move to ``requested``, refusing illegal moves.

        Refusal raises rather than returning a flag, because every caller in
        this codebase treats an illegal maintenance transition as a
        programming error. :meth:`try_transition` exists for the two places
        that genuinely want to probe.
        """
        requested = str(requested)
        if not can_transition_candidate(self.state, requested):
            raise InvalidCandidateTransition(self.state, requested)
        self.state = requested
        self._append_history(requested, reason)
        return self.state

    def try_transition(self, requested: str, *, reason: str = "") -> bool:
        try:
            self.transition(requested, reason=reason)
        except InvalidCandidateTransition:
            return False
        return True

    def record_outcome(self, outcome: str, *, reason: str = "") -> str:
        """Set the reassessment outcome; unknown outcomes become INCONCLUSIVE.

        Refusing to store an unrecognised outcome matters more than it looks:
        the learning layer counts outcomes, and an unrecognised string would be
        counted as "not a failure", quietly biasing every rate upward.
        """
        outcome = str(outcome)
        if outcome not in ALL_REASSESSMENT_OUTCOMES:
            outcome = ReassessmentOutcome.INCONCLUSIVE
        self.outcome = outcome
        self._append_history(f"outcome:{outcome}", reason)
        return self.outcome

    def merge_observation(self, other: "MaintenanceCandidate") -> "MaintenanceCandidate":
        """Fold a fresh observation of the same candidate into this one.

        Called when a scan re-detects an existing candidate. The merge is
        deliberately *pessimistic*: severity takes the maximum, uncertainty
        accumulates, and confidence takes the new value only when it is backed
        by at least as many samples. Otherwise a single lucky observation could
        overwrite a well-supported low confidence with a high one.
        """
        if other.candidate_id != self.candidate_id:
            raise ValueError(
                f"refusing to merge candidate {other.candidate_id} into {self.candidate_id}"
            )
        self.occurrence_count = min(10_000, self.occurrence_count + 1)
        self.last_seen_at = other.last_seen_at or _now()
        self.severity = highest_severity(self.severity, other.severity)
        if other.sample_size >= self.sample_size:
            self.confidence = other.confidence
            self.sample_size = other.sample_size
        self.title = other.title or self.title
        self.detail = other.detail or self.detail
        self.recommended_action = other.recommended_action or self.recommended_action
        self.estimated_effort = other.estimated_effort
        self.affected_files = sanitize_path_list(
            set(self.affected_files) | set(other.affected_files)
        )
        self.affected_symbols = sanitize_string_list(
            list(self.affected_symbols) + list(other.affected_symbols)
        )
        self.evidence_refs = sanitize_string_list(
            list(other.evidence_refs) + list(self.evidence_refs), limit=MAX_EVIDENCE_REFS
        )
        self.uncertainty = sanitize_string_list(
            list(self.uncertainty) + list(other.uncertainty)
        )
        self.metrics = sanitize_metrics({**self.metrics, **other.metrics})
        return self

    def _append_history(self, event: str, reason: str) -> None:
        self.history.append(
            {
                "at": _now(),
                "event": sanitize_text(event, limit=64),
                "reason": sanitize_text(reason, limit=200),
            }
        )
        if len(self.history) > MAX_HISTORY_ENTRIES:
            del self.history[: len(self.history) - MAX_HISTORY_ENTRIES]

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "kind": self.kind,
            "subject": self.subject,
            "title": self.title,
            "detail": self.detail,
            "provenance": self.provenance,
            "severity": self.severity,
            "confidence": self.confidence,
            "sample_size": self.sample_size,
            "uncertainty": list(self.uncertainty),
            "evidence_refs": list(self.evidence_refs),
            "affected_files": list(self.affected_files),
            "affected_symbols": list(self.affected_symbols),
            "recommended_action": self.recommended_action,
            "estimated_effort": self.estimated_effort,
            "occurrence_count": self.occurrence_count,
            "first_seen_at": self.first_seen_at,
            "last_seen_at": self.last_seen_at,
            "state": self.state,
            "outcome": self.outcome,
            "causal_links": list(self.causal_links),
            "metrics": dict(self.metrics),
            "history": [dict(entry) for entry in self.history],
            "blocked_reasons": list(self.blocked_reasons),
            "attempt_count": self.attempt_count,
            "failure_count": self.failure_count,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "MaintenanceCandidate":
        """Rebuild from persisted data, tolerating anything.

        Every field goes back through ``__post_init__``'s sanitisation, so a
        record hand-edited to claim ``severity: "critical"`` with
        ``confidence: 9999`` loads as a clamped, well-formed object rather than
        as a privileged one.
        """
        if not isinstance(data, Mapping):
            return cls()
        known = {
            key: data.get(key)
            for key in (
                "candidate_id",
                "kind",
                "subject",
                "title",
                "detail",
                "provenance",
                "severity",
                "confidence",
                "sample_size",
                "uncertainty",
                "evidence_refs",
                "affected_files",
                "affected_symbols",
                "recommended_action",
                "estimated_effort",
                "occurrence_count",
                "first_seen_at",
                "last_seen_at",
                "state",
                "outcome",
                "causal_links",
                "metrics",
                "history",
                "blocked_reasons",
                "attempt_count",
                "failure_count",
            )
            if key in data
        }
        return cls(**known)  # type: ignore[arg-type]


def _sanitize_history(entries: Any) -> list[dict[str, str]]:
    if isinstance(entries, (str, bytes)) or not isinstance(entries, Iterable):
        return []
    result: list[dict[str, str]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        result.append(
            {
                "at": sanitize_text(entry.get("at"), limit=64),
                "event": sanitize_text(entry.get("event"), limit=64),
                "reason": sanitize_text(entry.get("reason"), limit=200),
            }
        )
        if len(result) >= MAX_HISTORY_ENTRIES:
            break
    return result


def _safe_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _positive_or(value: Any, default: int) -> int:
    """``value`` as a positive int, or ``default`` when it is not one."""
    parsed = _safe_int(value, default=0)
    return parsed if parsed > 0 else default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number or number in (float("inf"), float("-inf")):
        return default
    return number


# -- budgets ------------------------------------------------------------------


class BudgetExceeded(RuntimeError):
    """Raised only where exceeding a budget is a programming error.

    The normal path never sees this: :meth:`BudgetLedger.try_consume` returns
    ``False`` and the caller degrades gracefully. This exists for
    :meth:`BudgetLedger.consume`, used where a silent over-run would be worse
    than a crash.
    """

    def __init__(self, name: str, limit: float, requested: float):
        super().__init__(
            f"maintenance budget '{name}' exhausted: limit {limit}, requested {requested}"
        )
        self.name = name
        self.limit = limit
        self.requested = requested


@dataclass(frozen=True)
class MaintenanceBudget:
    """Hierarchical limits on a maintenance run.

    Split across five levels - run, candidate, task, implementation,
    validation - because a single global cap is not enough: one candidate that
    legitimately needs three validation commands and one that wants three
    hundred must be distinguishable *before* either runs.

    Defaults are deliberately small. A maintenance run is background work; if
    it needs a large budget to be useful, that is a signal the candidate should
    have been a human-authored task instead.
    """

    # run level
    max_candidates_considered: int = 200
    max_candidates_selected: int = 5
    max_candidates_executed: int = 3
    max_elapsed_seconds: float = 1800.0
    max_estimated_cost_units: float = 100.0
    max_dag_width: int = 2

    # candidate level
    max_subtasks_per_candidate: int = 4
    max_changed_files_per_candidate: int = 10
    max_changed_lines_per_candidate: int = 400
    max_repair_iterations_per_candidate: int = 2

    # implementation / validation level
    max_tool_steps_per_subtask: int = 40
    max_candidate_iterations: int = 3
    max_validation_commands: int = 12

    def validate(self) -> None:
        """Reject a budget that cannot bound anything.

        Zero is allowed for the *count* limits (a run configured to execute
        nothing is a legitimate, and very safe, configuration); negative is
        not, and neither is a non-positive time or cost allowance, which would
        make every consumption attempt fail confusingly rather than
        immediately.
        """
        for name in (
            "max_candidates_considered",
            "max_candidates_selected",
            "max_candidates_executed",
            "max_dag_width",
            "max_subtasks_per_candidate",
            "max_changed_files_per_candidate",
            "max_changed_lines_per_candidate",
            "max_repair_iterations_per_candidate",
            "max_tool_steps_per_subtask",
            "max_candidate_iterations",
            "max_validation_commands",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
        for name in ("max_elapsed_seconds", "max_estimated_cost_units"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive number, got {value!r}")
        if self.max_candidates_executed > self.max_candidates_selected:
            raise ValueError(
                "max_candidates_executed cannot exceed max_candidates_selected"
            )
        if self.max_candidates_selected > self.max_candidates_considered:
            raise ValueError(
                "max_candidates_selected cannot exceed max_candidates_considered"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_candidates_considered": self.max_candidates_considered,
            "max_candidates_selected": self.max_candidates_selected,
            "max_candidates_executed": self.max_candidates_executed,
            "max_elapsed_seconds": self.max_elapsed_seconds,
            "max_estimated_cost_units": self.max_estimated_cost_units,
            "max_dag_width": self.max_dag_width,
            "max_subtasks_per_candidate": self.max_subtasks_per_candidate,
            "max_changed_files_per_candidate": self.max_changed_files_per_candidate,
            "max_changed_lines_per_candidate": self.max_changed_lines_per_candidate,
            "max_repair_iterations_per_candidate": self.max_repair_iterations_per_candidate,
            "max_tool_steps_per_subtask": self.max_tool_steps_per_subtask,
            "max_candidate_iterations": self.max_candidate_iterations,
            "max_validation_commands": self.max_validation_commands,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "MaintenanceBudget":
        if not isinstance(data, Mapping):
            return cls()
        defaults = cls()
        kwargs: dict[str, Any] = {}
        for name in defaults.to_dict():
            if name not in data:
                continue
            default_value = getattr(defaults, name)
            if isinstance(default_value, int):
                kwargs[name] = _safe_int(data[name], default=default_value)
            else:
                kwargs[name] = _safe_float(data[name], default_value)
        return cls(**kwargs)

    @classmethod
    def from_config(cls, config: Any) -> "MaintenanceBudget":
        """Build from an :class:`~local_agent.config.AgentConfig`-like object.

        Reads through ``getattr`` with the dataclass default as the fallback,
        so a config object from before this phase produces exactly the default
        budget instead of an ``AttributeError``.
        """
        defaults = cls()
        kwargs: dict[str, Any] = {}
        for name in defaults.to_dict():
            attribute = f"maintenance_{name}" if not name.startswith("maintenance_") else name
            if hasattr(config, attribute):
                default_value = getattr(defaults, name)
                raw = getattr(config, attribute)
                kwargs[name] = (
                    _safe_int(raw, default=default_value)
                    if isinstance(default_value, int)
                    else _safe_float(raw, default_value)
                )
        return cls(**kwargs)


class BudgetLedger:
    """Monotonic consumption tracking against a :class:`MaintenanceBudget`.

    Two properties are enforced rather than assumed:

    * **Monotonic.** ``consumed`` only ever increases. There is no release, no
      refund and no reset - a maintenance run cannot recover budget by
      abandoning work, which is precisely the loophole that would let a
      failing candidate retry forever.
    * **Never exceeded.** A consumption that would cross a limit is refused
      *in full*; partial consumption would leave the ledger in a state where
      the limit had been passed.

    Child ledgers (one per candidate) share the run-level ledger for time and
    cost, so per-candidate accounting cannot be used to escape the run cap.
    """

    def __init__(self, budget: MaintenanceBudget, *, parent: "BudgetLedger | None" = None):
        self.budget = budget
        self.parent = parent
        self._consumed: dict[str, float] = {}
        self._refusals: dict[str, int] = {}

    # -- accounting --------------------------------------------------------

    def limit_for(self, name: str) -> float:
        value = getattr(self.budget, name, None)
        if value is None:
            raise KeyError(f"unknown maintenance budget '{name}'")
        return float(value)

    def consumed(self, name: str) -> float:
        return self._consumed.get(name, 0.0)

    def remaining(self, name: str) -> float:
        return max(0.0, self.limit_for(name) - self.consumed(name))

    def would_exceed(self, name: str, amount: float = 1.0) -> bool:
        amount = max(0.0, _safe_float(amount, 0.0))
        if self.consumed(name) + amount > self.limit_for(name) + 1e-9:
            return True
        if self.parent is not None and name in _SHARED_RUN_LEVEL_BUDGETS:
            return self.parent.would_exceed(name, amount)
        return False

    def try_consume(self, name: str, amount: float = 1.0) -> bool:
        """Consume ``amount`` if it fits; otherwise consume nothing.

        Returns ``False`` rather than raising, because "we ran out of budget"
        is the expected, designed end of a maintenance run, not an error.
        """
        amount = max(0.0, _safe_float(amount, 0.0))
        if self.would_exceed(name, amount):
            self._refusals[name] = self._refusals.get(name, 0) + 1
            return False
        self._consumed[name] = self.consumed(name) + amount
        if self.parent is not None and name in _SHARED_RUN_LEVEL_BUDGETS:
            self.parent.try_consume(name, amount)
        return True

    def consume(self, name: str, amount: float = 1.0) -> None:
        if not self.try_consume(name, amount):
            raise BudgetExceeded(name, self.limit_for(name), amount)

    def observe(self, name: str, amount: float) -> None:
        """Record consumption that already happened, saturating at the limit.

        Used for measured quantities (elapsed time) where refusing after the
        fact is meaningless. Saturation keeps the invariant ``consumed <=
        limit`` true, so no report can ever show 110% of a budget spent while
        also claiming the budget was enforced. Whether the limit was reached is
        still visible through :meth:`exhausted`.
        """
        amount = max(0.0, _safe_float(amount, 0.0))
        self._consumed[name] = min(self.limit_for(name), self.consumed(name) + amount)
        if self.parent is not None and name in _SHARED_RUN_LEVEL_BUDGETS:
            self.parent.observe(name, amount)

    def exhausted(self, name: str) -> bool:
        if self.remaining(name) <= 1e-9:
            return True
        if self.parent is not None and name in _SHARED_RUN_LEVEL_BUDGETS:
            return self.parent.exhausted(name)
        return False

    def refusals(self) -> dict[str, int]:
        return dict(self._refusals)

    def child(self, budget: MaintenanceBudget | None = None) -> "BudgetLedger":
        """A sub-ledger that can never be more permissive than this one.

        Shared run-level budgets are already policed through ``parent`` on
        every consumption. The per-candidate counts are not - they are meant to
        be spent independently - so without clamping, handing ``child()`` a
        larger budget would raise the effective limit above the parent's and
        break ``child_budget <= parent_remaining_budget``. Every limit is
        therefore clamped to the parent's *remaining* allowance at the moment
        the child is created.
        """
        if budget is None:
            return BudgetLedger(self.budget, parent=self)
        clamped: dict[str, Any] = {}
        for name, value in budget.to_dict().items():
            remaining = self.remaining(name)
            clamped[name] = (
                min(int(value), int(remaining))
                if isinstance(value, int) and not isinstance(value, bool)
                else min(float(value), remaining)
            )
        # A fully-spent parent leaves a child with a zero time/cost allowance,
        # which ``MaintenanceBudget.validate`` rejects; the ledger still bounds
        # it correctly, so validation is deliberately not re-run here.
        return BudgetLedger(MaintenanceBudget(**clamped), parent=self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "limits": self.budget.to_dict(),
            "consumed": {name: self._consumed[name] for name in sorted(self._consumed)},
            "refusals": {name: self._refusals[name] for name in sorted(self._refusals)},
        }


#: Budgets a candidate-level ledger shares with its run-level parent. Counts
#: that are genuinely per-candidate (subtasks, changed files) are *not* here:
#: those are meant to be spent independently by each candidate.
_SHARED_RUN_LEVEL_BUDGETS: frozenset[str] = frozenset(
    {"max_elapsed_seconds", "max_estimated_cost_units"}
)


# -- run records --------------------------------------------------------------


RUN_MODE_SCAN = "scan"
RUN_MODE_DRY_RUN = "dry_run"
RUN_MODE_EXECUTE = "execute"

ALL_RUN_MODES: tuple[str, ...] = (RUN_MODE_SCAN, RUN_MODE_DRY_RUN, RUN_MODE_EXECUTE)

RUN_STATUS_RUNNING = "running"
RUN_STATUS_COMPLETED = "completed"
RUN_STATUS_PARTIAL = "partial"
RUN_STATUS_FAILED = "failed"
RUN_STATUS_CANCELLED = "cancelled"

ALL_RUN_STATUSES: tuple[str, ...] = (
    RUN_STATUS_RUNNING,
    RUN_STATUS_COMPLETED,
    RUN_STATUS_PARTIAL,
    RUN_STATUS_FAILED,
    RUN_STATUS_CANCELLED,
)


@dataclass
class CandidateRunOutcome:
    """What one run did to one candidate. Bounded and serialisable."""

    candidate_id: str = ""
    kind: str = ""
    title: str = ""
    priority: float = 0.0
    granted_tier: str = ""
    state: str = CandidateState.DETECTED
    outcome: str = ReassessmentOutcome.PENDING
    executed: bool = False
    validation_passed: bool | None = None
    reasons: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def __post_init__(self) -> None:
        self.candidate_id = sanitize_text(self.candidate_id, limit=64)
        self.kind = sanitize_text(self.kind, limit=64)
        self.title = sanitize_text(self.title)
        self.granted_tier = sanitize_text(self.granted_tier, limit=64)
        self.state = self.state if self.state in ALL_CANDIDATE_STATES else CandidateState.DETECTED
        self.outcome = (
            self.outcome if self.outcome in ALL_REASSESSMENT_OUTCOMES else ReassessmentOutcome.PENDING
        )
        self.priority = _safe_float(self.priority, 0.0)
        self.elapsed_seconds = max(0.0, _safe_float(self.elapsed_seconds, 0.0))
        self.reasons = sanitize_string_list(self.reasons)
        self.errors = sanitize_string_list(self.errors)
        self.changed_files = sanitize_path_list(self.changed_files)
        self.executed = bool(self.executed)
        if self.validation_passed is not None:
            self.validation_passed = bool(self.validation_passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "kind": self.kind,
            "title": self.title,
            "priority": self.priority,
            "granted_tier": self.granted_tier,
            "state": self.state,
            "outcome": self.outcome,
            "executed": self.executed,
            "validation_passed": self.validation_passed,
            "reasons": list(self.reasons),
            "errors": list(self.errors),
            "changed_files": list(self.changed_files),
            "elapsed_seconds": self.elapsed_seconds,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "CandidateRunOutcome":
        if not isinstance(data, Mapping):
            return cls()
        fields = {
            key: data.get(key)
            for key in (
                "candidate_id",
                "kind",
                "title",
                "priority",
                "granted_tier",
                "state",
                "outcome",
                "executed",
                "validation_passed",
                "reasons",
                "errors",
                "changed_files",
                "elapsed_seconds",
            )
            if key in data
        }
        return cls(**fields)  # type: ignore[arg-type]


@dataclass
class MaintenanceRunRecord:
    """A bounded audit record of one maintenance run."""

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    started_at: str = field(default_factory=_now)
    finished_at: str = ""
    mode: str = RUN_MODE_SCAN
    configured_tier: str = ""
    status: str = RUN_STATUS_RUNNING
    candidates_discovered: int = 0
    candidates_rejected: int = 0
    candidates_selected: int = 0
    execution_attempts: int = 0
    executions_succeeded: int = 0
    executions_failed: int = 0
    reassessments: int = 0
    elapsed_seconds: float = 0.0
    outcomes: list[CandidateRunOutcome] = field(default_factory=list)
    budget: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.run_id = sanitize_text(self.run_id, limit=64) or uuid.uuid4().hex[:16]
        self.mode = self.mode if self.mode in ALL_RUN_MODES else RUN_MODE_SCAN
        self.status = self.status if self.status in ALL_RUN_STATUSES else RUN_STATUS_RUNNING
        self.configured_tier = sanitize_text(self.configured_tier, limit=64)
        self.started_at = sanitize_text(self.started_at, limit=64) or _now()
        self.finished_at = sanitize_text(self.finished_at, limit=64)
        for name in (
            "candidates_discovered",
            "candidates_rejected",
            "candidates_selected",
            "execution_attempts",
            "executions_succeeded",
            "executions_failed",
            "reassessments",
        ):
            setattr(self, name, max(0, _safe_int(getattr(self, name))))
        self.elapsed_seconds = max(0.0, _safe_float(self.elapsed_seconds, 0.0))
        self.errors = sanitize_string_list(self.errors)
        self.notes = sanitize_string_list(self.notes)
        self.outcomes = [
            entry if isinstance(entry, CandidateRunOutcome) else CandidateRunOutcome.from_dict(entry)
            for entry in list(self.outcomes)[:MAX_LIST_ENTRIES]
        ]
        if not isinstance(self.budget, Mapping):
            self.budget = {}
        else:
            self.budget = dict(self.budget)

    @property
    def outcome_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self.outcomes:
            counts[entry.outcome] = counts.get(entry.outcome, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "mode": self.mode,
            "configured_tier": self.configured_tier,
            "status": self.status,
            "candidates_discovered": self.candidates_discovered,
            "candidates_rejected": self.candidates_rejected,
            "candidates_selected": self.candidates_selected,
            "execution_attempts": self.execution_attempts,
            "executions_succeeded": self.executions_succeeded,
            "executions_failed": self.executions_failed,
            "reassessments": self.reassessments,
            "elapsed_seconds": self.elapsed_seconds,
            "outcomes": [entry.to_dict() for entry in self.outcomes],
            "budget": dict(self.budget),
            "errors": list(self.errors),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "MaintenanceRunRecord":
        if not isinstance(data, Mapping):
            return cls()
        fields = {
            key: data.get(key)
            for key in (
                "run_id",
                "started_at",
                "finished_at",
                "mode",
                "configured_tier",
                "status",
                "candidates_discovered",
                "candidates_rejected",
                "candidates_selected",
                "execution_attempts",
                "executions_succeeded",
                "executions_failed",
                "reassessments",
                "elapsed_seconds",
                "outcomes",
                "budget",
                "errors",
                "notes",
            )
            if key in data
        }
        return cls(**fields)  # type: ignore[arg-type]


# -- the store ----------------------------------------------------------------


class MaintenanceStore:
    """Bounded persistent state: known candidates plus recent run history.

    Eviction policy for candidates is *not* plain FIFO. Dropping the oldest
    would discard exactly the long-lived, repeatedly-observed problems that
    matter most. Instead the least-interesting candidate is evicted: lowest
    severity first, then fewest occurrences, then oldest last-seen. Run records
    are FIFO, because a run record genuinely is only interesting while recent.
    """

    def __init__(
        self,
        *,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        max_runs: int = DEFAULT_MAX_RUNS,
    ):
        self.max_candidates = max(1, _safe_int(max_candidates, default=DEFAULT_MAX_CANDIDATES))
        self.max_runs = max(1, _safe_int(max_runs, default=DEFAULT_MAX_RUNS))
        self._candidates: dict[str, MaintenanceCandidate] = {}
        self._runs: list[MaintenanceRunRecord] = []
        #: Number of persisted records that failed to load. Non-zero makes the
        #: whole history untrustworthy for learning purposes; see
        #: :meth:`history_trustworthy`.
        self.corrupted_records_skipped = 0
        self.schema_version = MAINTENANCE_SCHEMA_VERSION

    # -- access ------------------------------------------------------------

    @property
    def candidates(self) -> list[MaintenanceCandidate]:
        """Every known candidate in deterministic order."""
        return sorted(self._candidates.values(), key=lambda c: c.candidate_id)

    @property
    def runs(self) -> list[MaintenanceRunRecord]:
        return list(self._runs)

    def __len__(self) -> int:
        return len(self._candidates)

    def find(self, candidate_id: str) -> MaintenanceCandidate | None:
        return self._candidates.get(sanitize_text(candidate_id, limit=64))

    def find_run(self, run_id: str) -> MaintenanceRunRecord | None:
        target = sanitize_text(run_id, limit=64)
        for record in self._runs:
            if record.run_id == target:
                return record
        return None

    def latest_run(self) -> MaintenanceRunRecord | None:
        return self._runs[-1] if self._runs else None

    def history_trustworthy(self) -> bool:
        return self.corrupted_records_skipped == 0

    # -- mutation ----------------------------------------------------------

    def upsert(self, candidate: MaintenanceCandidate) -> MaintenanceCandidate:
        """Insert ``candidate``, or fold it into the existing one with that id."""
        existing = self._candidates.get(candidate.candidate_id)
        if existing is not None:
            return existing.merge_observation(candidate)
        self._candidates[candidate.candidate_id] = candidate
        self._evict_candidates()
        return candidate

    def remove(self, candidate_id: str) -> bool:
        return self._candidates.pop(sanitize_text(candidate_id, limit=64), None) is not None

    def record_run(self, record: MaintenanceRunRecord) -> MaintenanceRunRecord:
        existing = self.find_run(record.run_id)
        if existing is not None:
            self._runs[self._runs.index(existing)] = record
        else:
            self._runs.append(record)
        if len(self._runs) > self.max_runs:
            del self._runs[: len(self._runs) - self.max_runs]
        return record

    def _evict_candidates(self) -> None:
        while len(self._candidates) > self.max_candidates:
            victim = min(
                self._candidates.values(),
                key=lambda c: (
                    severity_rank(c.severity),
                    c.occurrence_count,
                    c.last_seen_at,
                    c.candidate_id,
                ),
            )
            del self._candidates[victim.candidate_id]

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "max_candidates": self.max_candidates,
            "max_runs": self.max_runs,
            "corrupted_records_skipped": self.corrupted_records_skipped,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "runs": [record.to_dict() for record in self._runs],
        }

    @classmethod
    def from_dict(cls, data: Any) -> "MaintenanceStore":
        """Rebuild, skipping (and counting) any record that will not load.

        A record that raises is dropped and tallied rather than aborting the
        load: one corrupt candidate must not cost the operator their entire
        maintenance history. The tally is what stops the survivors from being
        mistaken for a complete, trustworthy record.
        """
        if not isinstance(data, Mapping):
            store = cls()
            store.corrupted_records_skipped = 1
            return store
        # A persisted retention cap is honoured only when it is a usable
        # positive integer. A corrupted or hostile file carrying
        # ``max_candidates: -5`` (or ``"abc"``, or ``0``) would otherwise clamp
        # to one and silently evict the operator's entire history during the
        # load - data loss driven by the very record that is not to be trusted.
        store = cls(
            max_candidates=_positive_or(data.get("max_candidates"), DEFAULT_MAX_CANDIDATES),
            max_runs=_positive_or(data.get("max_runs"), DEFAULT_MAX_RUNS),
        )
        store.corrupted_records_skipped = max(
            0, _safe_int(data.get("corrupted_records_skipped"))
        )
        store.schema_version = _safe_int(
            data.get("schema_version"), default=MAINTENANCE_SCHEMA_VERSION
        )
        raw_candidates = data.get("candidates")
        if isinstance(raw_candidates, list):
            for entry in raw_candidates:
                if not isinstance(entry, Mapping):
                    store.corrupted_records_skipped += 1
                    continue
                try:
                    candidate = MaintenanceCandidate.from_dict(entry)
                except Exception:
                    store.corrupted_records_skipped += 1
                    continue
                if not candidate.candidate_id:
                    store.corrupted_records_skipped += 1
                    continue
                store._candidates[candidate.candidate_id] = candidate
        elif raw_candidates is not None:
            store.corrupted_records_skipped += 1
        raw_runs = data.get("runs")
        if isinstance(raw_runs, list):
            for entry in raw_runs:
                if not isinstance(entry, Mapping):
                    store.corrupted_records_skipped += 1
                    continue
                try:
                    store._runs.append(MaintenanceRunRecord.from_dict(entry))
                except Exception:
                    store.corrupted_records_skipped += 1
        elif raw_runs is not None:
            store.corrupted_records_skipped += 1
        store._evict_candidates()
        if len(store._runs) > store.max_runs:
            del store._runs[: len(store._runs) - store.max_runs]
        return store


def summarize_candidates(candidates: Sequence[MaintenanceCandidate]) -> dict[str, Any]:
    """Aggregate counts over a candidate collection, for reporting."""
    by_kind: dict[str, int] = {}
    by_severity: dict[str, int] = {}
    by_state: dict[str, int] = {}
    by_outcome: dict[str, int] = {}
    for candidate in candidates:
        by_kind[candidate.kind] = by_kind.get(candidate.kind, 0) + 1
        by_severity[candidate.severity] = by_severity.get(candidate.severity, 0) + 1
        by_state[candidate.state] = by_state.get(candidate.state, 0) + 1
        by_outcome[candidate.outcome] = by_outcome.get(candidate.outcome, 0) + 1
    return {
        "total": len(candidates),
        "by_kind": dict(sorted(by_kind.items())),
        "by_severity": dict(sorted(by_severity.items())),
        "by_state": dict(sorted(by_state.items())),
        "by_outcome": dict(sorted(by_outcome.items())),
    }
