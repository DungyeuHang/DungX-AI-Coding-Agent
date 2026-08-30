"""Phase 4.20: the autonomous validation *lifecycle*.

Phase 4.17 gave DungX a semantic impact analysis, 4.18 a single authoritative
validation decision, and 4.19 a bounded telemetry record of that decision plus
a shadow-mode calibration comparison. What none of them could answer is any
question that spans more than one iteration:

    "This subtask needed three repairs. What did each one break, which
    validation stage caught it, did the second repair actually fix the first
    defect or just move it, and did the same defect come back?"

Every Phase 4.19 record stands alone - it links one decision to the outcome of
that same decision's run and stops there. There is no causal edge from a repair
attempt back to the iteration whose defect provoked it, so "repair success
rate" and "repeated defect rate" were literally uncomputable.

This module adds that missing spine, in six pieces:

``LifecycleState`` + :data:`ALLOWED_TRANSITIONS`
    An explicit state machine over one implementation attempt's life, from
    CREATED through APPLIED/POST_VALIDATED to a terminal COMPLETED /
    ABANDONED / FAILED. Invalid transitions raise
    :class:`InvalidLifecycleTransition` rather than silently succeeding; a
    terminal state has no outgoing edges at all.

``ValidationIterationRecord``
    One candidate/repair iteration, carrying ``parent_iteration_id`` - a real
    causal edge, not an incrementing counter - so the repair lineage is a
    reconstructable tree rather than a flat list that merely happens to be in
    chronological order.

``DefectSignature``
    A conservative, deterministic fingerprint of one validation failure with
    every volatile component (absolute paths, temp directories, PIDs, memory
    addresses, timestamps, uuids, bare line numbers) normalised out. Matching
    is exact-fingerprint equality only: there is deliberately no fuzzy
    similarity, because a *false merge* - deciding two genuinely different
    defects are "the same recurring defect" - would understate how badly the
    repair loop is doing, which is the dangerous direction to be wrong in.

``RepairEffectivenessMetrics``
    Repair success rate, repeated-defect rate, abandonment, first-pass
    success, stage distribution and recurrence-by-signature, all computed with
    Phase 4.19's conventions: Wilson bounds rather than bare point estimates,
    an explicit ``insufficient_data`` flag for small samples, and no metric
    that the recorded data cannot actually support.

``AdaptiveValidationRecommender``
    An **advisory** layer. It produces a recommendation *and*, separately, the
    safety floor it may never go below, and its ``effective_scope`` is always
    the broader of the two. It is not a decision authority: the authoritative
    choice remains
    :class:`~local_agent.validation_decision.ValidationDecisionEngine`, and
    nothing in this module is wired into it. See
    :meth:`AdaptiveValidationRecommender.recommend`.

``ValidationLifecycleStore`` / ``ValidationLifecycleManager``
    A bounded, tolerant, separately-persisted history mirroring
    :class:`~local_agent.validation_telemetry.ValidationTelemetryStore`
    exactly - same eviction shape, same "corrupt entries are counted and
    dropped, never trusted and never fatal" policy, same per-repository
    threading lock. Cross-*process* concurrency is out of scope here for the
    same reason it is throughout :mod:`local_agent.storage`: no file in this
    codebase has cross-process locking, and claiming it here would be a lie.

**The safety rule this whole module is subordinate to.** Learning may improve
validation *efficiency*; it may never silently reduce validation *safety*.
Concretely, and enforced by code rather than by comment:

* :func:`safest_scope` is the only way two scopes are ever combined, and it
  returns the broader one. Nothing in this module ever narrows a scope.
* Insufficient history yields ``recommended_scope = safety_floor`` and
  ``data_sufficient = False`` - never a narrower guess.
* A corrupt store raises the corruption count, sets ``history_trustworthy =
  False``, and is treated as *no* history, which by the previous rule means
  the floor stands.
* A perfect historical record cannot move the recommendation below the floor,
  because the floor is applied last and unconditionally.
"""

from __future__ import annotations

import datetime
import hashlib
import logging
import re
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .semantic_impact import SCOPE_BROAD, SCOPE_EXPANDED, SCOPE_ORDER, SCOPE_TARGETED
from .validation_telemetry import (
    repository_identity,
    wilson_bounds,
    wilson_lower_bound,
)

LOGGER = logging.getLogger(__name__)

#: Shape version of the lifecycle record schema. A record written by an older
#: build has no such key and loads as ``"0"``; a record written by a *newer*
#: build keeps its own value and its unknown fields are preserved verbatim (see
#: :attr:`ValidationLifecycleRecord.extra`) so a round-trip through this build
#: does not destroy data it did not understand.
LIFECYCLE_SCHEMA_VERSION = "4.20.0"

#: Default bound on retained lifecycles. Same order of magnitude and the same
#: rationale as :data:`~local_agent.validation_telemetry.DEFAULT_MAX_DECISIONS`:
#: history must stay cheap and bounded no matter how long the agent has run.
DEFAULT_MAX_LIFECYCLES = 200
#: Per-lifecycle bound on retained iterations. A runaway repair loop must not be
#: able to make one lifecycle dominate the store.
DEFAULT_MAX_ITERATIONS_PER_LIFECYCLE = 50


# -- state machine -------------------------------------------------------------


class LifecycleState:
    """The vocabulary of one implementation attempt's life.

    Deliberately *not* a competitor to
    :class:`~local_agent.models.TaskStatus` (a task's scheduling state) or
    :class:`~local_agent.models.ImplementationTerminationReason` (why one
    provider loop stopped). Those describe different subjects; neither has a
    state for "applied but not yet post-validated", which is precisely the
    window this machine exists to make explicit. Terminal outcomes are mapped
    back onto the existing failure-category vocabulary by
    :func:`failure_category_for`, so no second categorisation vocabulary is
    introduced.
    """

    CREATED = "created"
    CANDIDATE_GENERATED = "candidate_generated"
    VALIDATED = "validated"
    APPROVED = "approved"
    APPLIED = "applied"
    POST_VALIDATED = "post_validated"
    REPAIR_REQUIRED = "repair_required"
    REPAIRED = "repaired"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    FAILED = "failed"

    #: States from which nothing may follow. Enforced structurally: they have
    #: no entry in :data:`ALLOWED_TRANSITIONS` with a non-empty target set.
    TERMINAL = frozenset({COMPLETED, ABANDONED, FAILED})

    ALL = (
        CREATED,
        CANDIDATE_GENERATED,
        VALIDATED,
        APPROVED,
        APPLIED,
        POST_VALIDATED,
        REPAIR_REQUIRED,
        REPAIRED,
        COMPLETED,
        ABANDONED,
        FAILED,
    )


#: The complete transition relation. Anything not listed here is invalid, which
#: is what makes ``COMPLETED -> CREATED``, ``ABANDONED -> APPLIED`` and
#: ``FAILED -> POST_VALIDATED`` rejections rather than silent no-ops.
ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    LifecycleState.CREATED: frozenset({
        LifecycleState.CANDIDATE_GENERATED,
        LifecycleState.ABANDONED,
        LifecycleState.FAILED,
    }),
    LifecycleState.CANDIDATE_GENERATED: frozenset({
        LifecycleState.VALIDATED,
        LifecycleState.REPAIR_REQUIRED,
        LifecycleState.ABANDONED,
        LifecycleState.FAILED,
    }),
    LifecycleState.VALIDATED: frozenset({
        LifecycleState.APPROVED,
        LifecycleState.REPAIR_REQUIRED,
        LifecycleState.ABANDONED,
        LifecycleState.FAILED,
    }),
    LifecycleState.APPROVED: frozenset({
        LifecycleState.APPLIED,
        LifecycleState.ABANDONED,
        LifecycleState.FAILED,
    }),
    LifecycleState.APPLIED: frozenset({
        LifecycleState.POST_VALIDATED,
        LifecycleState.REPAIR_REQUIRED,
        LifecycleState.ABANDONED,
        LifecycleState.FAILED,
    }),
    LifecycleState.POST_VALIDATED: frozenset({
        LifecycleState.COMPLETED,
        LifecycleState.REPAIR_REQUIRED,
        LifecycleState.ABANDONED,
        LifecycleState.FAILED,
    }),
    LifecycleState.REPAIR_REQUIRED: frozenset({
        LifecycleState.REPAIRED,
        LifecycleState.ABANDONED,
        LifecycleState.FAILED,
    }),
    LifecycleState.REPAIRED: frozenset({
        LifecycleState.CANDIDATE_GENERATED,
        LifecycleState.COMPLETED,
        LifecycleState.ABANDONED,
        LifecycleState.FAILED,
    }),
    LifecycleState.COMPLETED: frozenset(),
    LifecycleState.ABANDONED: frozenset(),
    LifecycleState.FAILED: frozenset(),
}


class InvalidLifecycleTransition(ValueError):
    """Raised when a caller attempts a transition the machine forbids."""

    def __init__(self, current: str, requested: str):
        self.current = current
        self.requested = requested
        allowed = sorted(ALLOWED_TRANSITIONS.get(current, frozenset()))
        super().__init__(
            f"cannot transition from '{current}' to '{requested}'; "
            f"allowed from '{current}': {allowed or 'none (terminal state)'}"
        )


def can_transition(current: str, requested: str) -> bool:
    """Whether ``current -> requested`` is a legal edge. Never raises.

    An unknown ``current`` state (e.g. one written by a newer build) permits
    *nothing*: an unrecognised state is treated as terminal rather than as
    permissive, so a forward-compatibility gap fails closed.
    """
    return requested in ALLOWED_TRANSITIONS.get(current, frozenset())


#: Terminal state -> the coarse failure category vocabulary already used by
#: :class:`~local_agent.models.ImplementationTerminationReason`.
_CATEGORY_BY_TERMINAL = {
    LifecycleState.COMPLETED: "none",
    LifecycleState.ABANDONED: "abandoned",
    LifecycleState.FAILED: "validation_failure",
}


def failure_category_for(state: str, fallback: str = "unknown") -> str:
    """Coarse failure category for a terminal ``state``."""
    return _CATEGORY_BY_TERMINAL.get(state, fallback)


def safest_scope(*scopes: str) -> str:
    """The broadest of ``scopes`` - the only way this module combines scopes.

    An unrecognised scope name is treated as :data:`SCOPE_BROAD`, not ignored:
    a value this build does not understand must never be able to make the
    result narrower than it would otherwise have been.
    """
    best = 0
    for scope in scopes:
        if scope not in SCOPE_ORDER:
            return SCOPE_BROAD
        best = max(best, SCOPE_ORDER.index(scope))
    return SCOPE_ORDER[best]


# -- defect signatures ---------------------------------------------------------

#: Validation stages a defect can be caught at, weakest isolation first. Used
#: only for reporting *where* defects are being caught, never for a decision.
STAGE_CANDIDATE = "candidate"
STAGE_TARGETED = "targeted"
STAGE_BROAD = "broad"
STAGE_POST_APPLY = "post_apply"
STAGE_UNKNOWN = "unknown"
ALL_STAGES: tuple[str, ...] = (
    STAGE_CANDIDATE, STAGE_TARGETED, STAGE_BROAD, STAGE_POST_APPLY, STAGE_UNKNOWN
)

RESULT_PASSED = "passed"
RESULT_FAILED = "failed"
RESULT_NOT_RUN = "not_run"

#: Ordered normalisation rules. Order matters: the path rules run before the
#: bare-number rules so that ``C:\\Temp\\x\\test.py:41`` collapses to
#: ``<path>:<n>`` rather than leaving a half-normalised fragment behind.
_NORMALISERS: tuple[tuple[re.Pattern[str], str], ...] = (
    # ISO-8601-ish timestamps, with or without a timezone.
    (re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"), "<ts>"),
    # Windows absolute paths (drive-letter rooted), and UNC paths.
    (re.compile(r"(?:[A-Za-z]:[\\/]|\\\\)[^\s'\"<>|]*"), "<path>"),
    # POSIX absolute paths of at least two segments.
    (re.compile(r"(?<![\w.])/(?:[^\s'\"<>|/]+/)+[^\s'\"<>|/]*"), "<path>"),
    # Memory addresses.
    (re.compile(r"0x[0-9a-fA-F]{4,}"), "<addr>"),
    # UUIDs (canonical, then bare 32-hex).
    (re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"), "<id>"),
    (re.compile(r"\b[0-9a-fA-F]{32,}\b"), "<id>"),
    # PIDs, however they are spelled.
    (re.compile(r"\b(?:pid|PID|process)[ =:]+\d+"), "pid <pid>"),
    # "line 41" / "line: 41" and trailing ":41:" position markers.
    (re.compile(r"\bline[ :]+\d+"), "line <n>"),
    (re.compile(r":\d+(?=:|\b)"), ":<n>"),
    # Durations and byte counts, which vary run to run.
    (re.compile(r"\b\d+(?:\.\d+)?\s*(?:s|ms|sec|secs|seconds)\b"), "<duration>"),
    # Pytest's per-run summary counts.
    (re.compile(r"\b\d+ (passed|failed|error|errors|skipped|warning|warnings|xfailed)\b"), r"<n> \1"),
)

#: How much normalised diagnostic text a signature keeps. Long enough to
#: distinguish real defects, short enough that one signature cannot bloat the
#: bounded store.
_MAX_DIAGNOSTIC_CHARS = 400


def normalize_diagnostic(text: Any, *, limit: int = _MAX_DIAGNOSTIC_CHARS) -> str:
    """Strip every run-to-run-volatile component out of diagnostic text.

    Deterministic and total: any input (including ``None`` or a non-string)
    yields a string, and the same input always yields the same output, which is
    what makes a signature computed on one machine comparable with one computed
    on another.

    This is the *only* place volatility is removed. It removes absolute paths,
    temporary directories (a special case of absolute paths), timestamps,
    memory addresses, uuids and other long hex ids, PIDs, bare line numbers and
    position markers, durations, and pytest's per-run counts. It deliberately
    does **not** lowercase, stem, or otherwise fuzz the remaining text: two
    messages that differ in any surviving character are different defects.
    """
    if text is None:
        return ""
    raw = text if isinstance(text, str) else str(text)
    for pattern, replacement in _NORMALISERS:
        raw = pattern.sub(replacement, raw)
    # Collapse all whitespace runs so a wrapped vs. unwrapped traceback of the
    # same failure does not produce two signatures.
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw[:limit]


def normalize_command(command: Any) -> tuple[str, ...]:
    """Canonical command vector with volatile arguments normalised.

    A command that names a file inside a candidate workspace (an OS temp
    directory that changes every iteration) must not produce a different
    signature every time, so each argument is run through the same
    normalisation as diagnostic text.
    """
    if isinstance(command, str):
        parts: Iterable[Any] = command.split()
    elif isinstance(command, (list, tuple)):
        parts = command
    else:
        return ()
    return tuple(normalize_diagnostic(part, limit=120) for part in parts if str(part) != "")


@dataclass(frozen=True)
class DefectSignature:
    """A conservative, stable identity for one validation failure.

    Frozen and hashable so a set/dict of these deduplicates for free.
    :attr:`fingerprint` is a digest of *every* field below, which makes
    matching exact by construction - see :func:`signatures_match` for why no
    looser rule is offered.
    """

    failure_category: str = ""
    command: tuple[str, ...] = ()
    exit_code: int = 0
    diagnostic: str = ""
    affected_file: str = ""
    affected_symbol: str = ""
    validation_tier: str = STAGE_UNKNOWN
    exception_class: str = ""
    stderr_fragment: str = ""
    stdout_fragment: str = ""

    @property
    def fingerprint(self) -> str:
        """Digest over the canonical form of every field.

        Field values are joined with ``\\x1f`` (unit separator), a character
        that cannot appear in any normalised component, so two different field
        splits can never produce the same joined string - a concatenation
        ambiguity would be exactly the kind of false merge this class exists to
        avoid.
        """
        parts = (
            self.failure_category,
            "\x1e".join(self.command),
            str(self.exit_code),
            self.diagnostic,
            self.affected_file,
            self.affected_symbol,
            self.validation_tier,
            self.exception_class,
            self.stderr_fragment,
            self.stdout_fragment,
        )
        joined = "\x1f".join(parts)
        return hashlib.sha256(joined.encode("utf-8", "replace")).hexdigest()[:32]

    @property
    def is_empty(self) -> bool:
        """True when nothing distinguishing was recorded.

        An empty signature must never be treated as "the same defect" as
        another empty one - :func:`signatures_match` checks this explicitly.
        """
        return not any((
            self.failure_category,
            self.command,
            self.diagnostic,
            self.affected_file,
            self.affected_symbol,
            self.exception_class,
            self.stderr_fragment,
            self.stdout_fragment,
        ))

    def describe(self) -> str:
        bits = [self.failure_category or "uncategorised"]
        if self.command:
            bits.append(" ".join(self.command))
        if self.exception_class:
            bits.append(self.exception_class)
        if self.diagnostic:
            bits.append(self.diagnostic[:120])
        return " | ".join(bits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_category": self.failure_category,
            "command": list(self.command),
            "exit_code": self.exit_code,
            "diagnostic": self.diagnostic,
            "affected_file": self.affected_file,
            "affected_symbol": self.affected_symbol,
            "validation_tier": self.validation_tier,
            "exception_class": self.exception_class,
            "stderr_fragment": self.stderr_fragment,
            "stdout_fragment": self.stdout_fragment,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "DefectSignature":
        if not isinstance(data, dict):
            return cls()
        try:
            exit_code = int(data.get("exit_code", 0) or 0)
        except (TypeError, ValueError):
            exit_code = 0
        return cls(
            failure_category=str(data.get("failure_category", "")),
            command=tuple(str(c) for c in (data.get("command") or [])),
            exit_code=exit_code,
            diagnostic=str(data.get("diagnostic", "")),
            affected_file=str(data.get("affected_file", "")),
            affected_symbol=str(data.get("affected_symbol", "")),
            validation_tier=str(data.get("validation_tier", STAGE_UNKNOWN)),
            exception_class=str(data.get("exception_class", "")),
            stderr_fragment=str(data.get("stderr_fragment", "")),
            stdout_fragment=str(data.get("stdout_fragment", "")),
        )


#: How much of stderr/stdout a signature retains. The *tail* is kept, because
#: that is where a failing command's actual error lives.
_MAX_STREAM_CHARS = 300


def compute_defect_signature(
    *,
    failure_category: str = "",
    command: Any = (),
    exit_code: Any = 0,
    diagnostic: Any = "",
    affected_file: Any = "",
    affected_symbol: Any = "",
    validation_tier: str = STAGE_UNKNOWN,
    exception_class: Any = "",
    stderr: Any = "",
    stdout: Any = "",
) -> DefectSignature:
    """Build one :class:`DefectSignature`, normalising every volatile input.

    Pure: same inputs always produce the same signature, on any machine, in any
    working directory, at any time. That determinism is what makes recurrence
    detection meaningful at all.
    """
    try:
        code = int(exit_code or 0)
    except (TypeError, ValueError):
        code = 0
    return DefectSignature(
        failure_category=str(failure_category or ""),
        command=normalize_command(command),
        exit_code=code,
        diagnostic=normalize_diagnostic(diagnostic),
        # A repository-relative path is not volatile, but an absolute one is;
        # running it through the same normaliser collapses the latter.
        affected_file=normalize_diagnostic(affected_file, limit=200),
        affected_symbol=str(affected_symbol or "")[:200],
        validation_tier=validation_tier if validation_tier in ALL_STAGES else STAGE_UNKNOWN,
        exception_class=str(exception_class or "")[:120],
        stderr_fragment=_stream_tail(stderr),
        stdout_fragment=_stream_tail(stdout),
    )


def _stream_tail(stream: Any) -> str:
    """The normalised *tail* of a captured stream.

    The tail specifically, at both ends of the operation: a failing command's
    actual error is the last thing it prints, so slicing the head off before
    normalising is not enough - normalisation can also shrink the text (paths
    collapse to ``<path>``), which would otherwise let the truncation step
    re-admit leading filler and push the real error back off the end.
    """
    raw = str(stream or "")
    if not raw:
        return ""
    # Normalise generously (no limit), then keep the tail.
    normalised = normalize_diagnostic(raw[-_MAX_STREAM_CHARS * 20:], limit=10 ** 9)
    return normalised[-_MAX_STREAM_CHARS:]


def signatures_match(left: DefectSignature, right: DefectSignature) -> bool:
    """Whether two signatures denote the same defect. Exact match only.

    There is deliberately no similarity threshold, no partial credit and no
    "close enough" rule. Two failure modes are possible and they are not
    symmetric:

    * A *false merge* (two different defects judged the same) makes the repair
      loop look like it is failing to fix a recurring problem when in fact it
      is fixing one problem and hitting another - it corrupts every recurrence
      statistic and could justify abandoning a lifecycle that was progressing.
    * A *missed match* (the same defect judged different) merely understates
      recurrence, which shows up as a lower recurrence rate and never causes
      anything to be validated less.

    The second is strictly safer, so matching is exact-fingerprint equality.
    Two *empty* signatures never match: "nothing was recorded" is not evidence
    that two failures were the same failure.
    """
    if left.is_empty or right.is_empty:
        return False
    return left.fingerprint == right.fingerprint


# -- iteration + lifecycle records ----------------------------------------------

ITERATION_IMPLEMENTATION = "implementation"
ITERATION_REPAIR = "repair"


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _known_keys(data: dict[str, Any], known: Iterable[str]) -> dict[str, Any]:
    """Everything in ``data`` this build does not recognise.

    Preserved verbatim on the record so a payload written by a newer build
    survives a load/save cycle here instead of being silently truncated.
    """
    known_set = set(known)
    return {k: v for k, v in data.items() if k not in known_set}


_ITERATION_KEYS = (
    "iteration_id", "iteration_number", "parent_iteration_id", "kind", "candidate_id",
    "decision_id", "evidence_fingerprint", "scope", "confidence_level", "commands",
    "validation_result", "validation_stage", "apply_result", "defect_signature",
    "failure_category", "provider", "model", "duration_seconds", "started_at",
    "ended_at", "notes",
)


@dataclass
class ValidationIterationRecord:
    """One candidate or repair iteration inside a lifecycle.

    ``parent_iteration_id`` is the load-bearing field: it is a real causal
    edge ("this repair exists *because* that iteration failed"), which is what
    lets :meth:`ValidationLifecycleRecord.repair_chain` reconstruct a lineage
    tree. An incrementing ``iteration_number`` alone could not distinguish a
    repair of iteration 2 from an unrelated retry that merely happens to run
    third.
    """

    iteration_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    iteration_number: int = 1
    #: The iteration whose defect provoked this one; empty for the first.
    parent_iteration_id: str = ""
    kind: str = ITERATION_IMPLEMENTATION
    candidate_id: str = ""
    #: Links back to the Phase 4.19 ``ValidationDecisionRecord.decision_id``.
    decision_id: str = ""
    #: The Phase 4.18 evidence/tree fingerprint this iteration validated against.
    evidence_fingerprint: str = ""
    scope: str = ""
    confidence_level: str = ""
    commands: list[list[str]] = field(default_factory=list)
    validation_result: str = RESULT_NOT_RUN
    validation_stage: str = STAGE_UNKNOWN
    apply_result: str = ""
    defect_signature: DefectSignature | None = None
    failure_category: str = ""
    provider: str = ""
    model: str = ""
    duration_seconds: float = 0.0
    started_at: str = field(default_factory=_now)
    ended_at: str = ""
    notes: list[str] = field(default_factory=list)
    #: Unrecognised fields from a newer schema, preserved for round-tripping.
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return self.validation_result == RESULT_FAILED

    @property
    def passed(self) -> bool:
        return self.validation_result == RESULT_PASSED

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "iteration_id": self.iteration_id,
            "iteration_number": self.iteration_number,
            "parent_iteration_id": self.parent_iteration_id,
            "kind": self.kind,
            "candidate_id": self.candidate_id,
            "decision_id": self.decision_id,
            "evidence_fingerprint": self.evidence_fingerprint,
            "scope": self.scope,
            "confidence_level": self.confidence_level,
            "commands": [list(c) for c in self.commands],
            "validation_result": self.validation_result,
            "validation_stage": self.validation_stage,
            "apply_result": self.apply_result,
            "defect_signature": (
                self.defect_signature.to_dict() if self.defect_signature is not None else None
            ),
            "failure_category": self.failure_category,
            "provider": self.provider,
            "model": self.model,
            "duration_seconds": round(self.duration_seconds, 4),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "notes": list(self.notes),
        }
        payload.update(self.extra)
        return payload

    @classmethod
    def from_dict(cls, data: Any) -> "ValidationIterationRecord":
        if not isinstance(data, dict):
            return cls()
        try:
            number = int(data.get("iteration_number", 1) or 1)
        except (TypeError, ValueError):
            number = 1
        try:
            duration = float(data.get("duration_seconds", 0.0) or 0.0)
        except (TypeError, ValueError):
            duration = 0.0
        raw_commands = data.get("commands") or []
        commands: list[list[str]] = []
        if isinstance(raw_commands, list):
            for entry in raw_commands:
                if isinstance(entry, (list, tuple)):
                    commands.append([str(part) for part in entry])
                elif isinstance(entry, str):
                    commands.append(entry.split())
        signature_raw = data.get("defect_signature")
        return cls(
            iteration_id=str(data.get("iteration_id") or uuid.uuid4().hex),
            iteration_number=number,
            parent_iteration_id=str(data.get("parent_iteration_id", "") or ""),
            kind=str(data.get("kind", ITERATION_IMPLEMENTATION) or ITERATION_IMPLEMENTATION),
            candidate_id=str(data.get("candidate_id", "") or ""),
            decision_id=str(data.get("decision_id", "") or ""),
            evidence_fingerprint=str(data.get("evidence_fingerprint", "") or ""),
            scope=str(data.get("scope", "") or ""),
            confidence_level=str(data.get("confidence_level", "") or ""),
            commands=commands,
            validation_result=str(data.get("validation_result", RESULT_NOT_RUN) or RESULT_NOT_RUN),
            validation_stage=str(data.get("validation_stage", STAGE_UNKNOWN) or STAGE_UNKNOWN),
            apply_result=str(data.get("apply_result", "") or ""),
            defect_signature=(
                DefectSignature.from_dict(signature_raw)
                if isinstance(signature_raw, dict)
                else None
            ),
            failure_category=str(data.get("failure_category", "") or ""),
            provider=str(data.get("provider", "") or ""),
            model=str(data.get("model", "") or ""),
            duration_seconds=duration,
            started_at=str(data.get("started_at", "") or ""),
            ended_at=str(data.get("ended_at", "") or ""),
            notes=[str(n) for n in (data.get("notes") or [])],
            extra=_known_keys(data, _ITERATION_KEYS),
        )


_LIFECYCLE_KEYS = (
    "lifecycle_id", "schema_version", "repository_id", "task_id", "subtask_id",
    "state", "state_history", "iterations", "terminal_outcome", "failure_category",
    "created_at", "updated_at", "provider", "model", "notes",
)


@dataclass
class ValidationLifecycleRecord:
    """The durable trace of one implementation attempt, end to end."""

    lifecycle_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    schema_version: str = LIFECYCLE_SCHEMA_VERSION
    repository_id: str = ""
    task_id: str = ""
    subtask_id: str = ""
    state: str = LifecycleState.CREATED
    #: ``(state, timestamp, reason)`` for every accepted transition, including
    #: the implicit initial one, so the machine's path is auditable after the
    #: fact rather than only its endpoint.
    state_history: list[dict[str, str]] = field(default_factory=list)
    iterations: list[ValidationIterationRecord] = field(default_factory=list)
    terminal_outcome: str = ""
    failure_category: str = ""
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    provider: str = ""
    model: str = ""
    notes: list[str] = field(default_factory=list)
    max_iterations: int = DEFAULT_MAX_ITERATIONS_PER_LIFECYCLE
    #: Unrecognised fields from a newer schema, preserved for round-tripping.
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.state_history:
            self.state_history = [
                {"state": self.state, "at": self.created_at, "reason": "lifecycle created"}
            ]

    # -- state machine -----------------------------------------------------

    @property
    def is_terminal(self) -> bool:
        return self.state in LifecycleState.TERMINAL

    def transition(self, requested: str, *, reason: str = "") -> str:
        """Move to ``requested``, or raise :class:`InvalidLifecycleTransition`.

        Raising (rather than returning a bool) is deliberate: a caller that
        gets a lifecycle into a state the machine forbids has a real bug, and
        silently ignoring it would leave the persisted history claiming a path
        that never happened.
        """
        if not can_transition(self.state, requested):
            raise InvalidLifecycleTransition(self.state, requested)
        self.state = requested
        self.updated_at = _now()
        self.state_history.append(
            {"state": requested, "at": self.updated_at, "reason": reason}
        )
        if requested in LifecycleState.TERMINAL:
            self.terminal_outcome = requested
            if not self.failure_category:
                self.failure_category = failure_category_for(requested)
        return self.state

    def try_transition(self, requested: str, *, reason: str = "") -> bool:
        """Non-raising :meth:`transition`; returns whether it was accepted."""
        try:
            self.transition(requested, reason=reason)
            return True
        except InvalidLifecycleTransition:
            return False

    # -- iterations --------------------------------------------------------

    def add_iteration(self, iteration: ValidationIterationRecord) -> ValidationIterationRecord:
        """Append an iteration, enforcing the per-lifecycle bound.

        The *oldest* iterations are evicted when the bound is hit, matching
        every other bounded store in this codebase; the newest are what any
        diagnosis needs. Eviction can orphan a ``parent_iteration_id``, which
        :meth:`repair_chain` handles explicitly rather than assuming the parent
        is always present.
        """
        self.iterations.append(iteration)
        if len(self.iterations) > max(1, self.max_iterations):
            del self.iterations[: len(self.iterations) - max(1, self.max_iterations)]
        self.updated_at = _now()
        return iteration

    def find_iteration(self, iteration_id: str) -> ValidationIterationRecord | None:
        for record in self.iterations:
            if record.iteration_id == iteration_id:
                return record
        return None

    @property
    def latest_iteration(self) -> ValidationIterationRecord | None:
        return self.iterations[-1] if self.iterations else None

    def children_of(self, iteration_id: str) -> list[ValidationIterationRecord]:
        return [i for i in self.iterations if i.parent_iteration_id == iteration_id]

    def repair_chain(self, iteration_id: str) -> list[ValidationIterationRecord]:
        """The causal ancestry of ``iteration_id``, root first.

        Walks ``parent_iteration_id`` upward with a visited-set, so a corrupted
        store containing a parent cycle terminates instead of hanging. A parent
        that is not present (evicted, or never written) simply ends the walk -
        the chain is then partial, which is the honest answer.
        """
        # One index built up front, rather than a linear scan per hop: the walk
        # is then O(n) in the number of iterations instead of O(n^2). The index
        # is local to the call, so there is no cached structure to keep in sync
        # with eviction.
        by_id = {record.iteration_id: record for record in self.iterations}
        chain: list[ValidationIterationRecord] = []
        seen: set[str] = set()
        current = by_id.get(iteration_id)
        while current is not None and current.iteration_id not in seen:
            seen.add(current.iteration_id)
            chain.append(current)
            if not current.parent_iteration_id:
                break
            current = by_id.get(current.parent_iteration_id)
        chain.reverse()
        return chain

    @property
    def repair_count(self) -> int:
        return sum(1 for i in self.iterations if i.kind == ITERATION_REPAIR)

    def defect_signatures(self) -> list[DefectSignature]:
        return [
            i.defect_signature for i in self.iterations
            if i.defect_signature is not None and not i.defect_signature.is_empty
        ]

    def recurring_defects(self) -> dict[str, int]:
        """Fingerprint -> occurrence count, for fingerprints seen more than once."""
        counts: dict[str, int] = {}
        for signature in self.defect_signatures():
            counts[signature.fingerprint] = counts.get(signature.fingerprint, 0) + 1
        return {k: v for k, v in counts.items() if v > 1}

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "lifecycle_id": self.lifecycle_id,
            "schema_version": self.schema_version,
            "repository_id": self.repository_id,
            "task_id": self.task_id,
            "subtask_id": self.subtask_id,
            "state": self.state,
            "state_history": [dict(entry) for entry in self.state_history],
            "iterations": [i.to_dict() for i in self.iterations],
            "terminal_outcome": self.terminal_outcome,
            "failure_category": self.failure_category,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "provider": self.provider,
            "model": self.model,
            "notes": list(self.notes),
        }
        payload.update(self.extra)
        return payload

    @classmethod
    def from_dict(cls, data: Any) -> "ValidationLifecycleRecord":
        """Tolerant load. Missing fields get safe defaults; unknown fields are
        preserved; a state this build does not recognise is *kept* (so the data
        is not falsified) but, per :func:`can_transition`, permits no further
        transitions."""
        if not isinstance(data, dict):
            return cls()
        iterations: list[ValidationIterationRecord] = []
        raw_iterations = data.get("iterations")
        if isinstance(raw_iterations, list):
            for entry in raw_iterations:
                if isinstance(entry, dict):
                    iterations.append(ValidationIterationRecord.from_dict(entry))
        history: list[dict[str, str]] = []
        raw_history = data.get("state_history")
        if isinstance(raw_history, list):
            for entry in raw_history:
                if isinstance(entry, dict):
                    history.append({str(k): str(v) for k, v in entry.items()})
        created = str(data.get("created_at", "") or _now())
        record = cls(
            lifecycle_id=str(data.get("lifecycle_id") or uuid.uuid4().hex),
            # A record with no schema_version predates this field entirely.
            schema_version=str(data.get("schema_version", "0") or "0"),
            repository_id=str(data.get("repository_id", "") or ""),
            task_id=str(data.get("task_id", "") or ""),
            subtask_id=str(data.get("subtask_id", "") or ""),
            state=str(data.get("state", LifecycleState.CREATED) or LifecycleState.CREATED),
            state_history=history,
            iterations=iterations,
            terminal_outcome=str(data.get("terminal_outcome", "") or ""),
            failure_category=str(data.get("failure_category", "") or ""),
            created_at=created,
            updated_at=str(data.get("updated_at", "") or created),
            provider=str(data.get("provider", "") or ""),
            model=str(data.get("model", "") or ""),
            notes=[str(n) for n in (data.get("notes") or [])],
            extra=_known_keys(data, _LIFECYCLE_KEYS),
        )
        return record


# -- structured events ----------------------------------------------------------

EVENT_LIFECYCLE_STARTED = "lifecycle_started"
EVENT_VALIDATION_STARTED = "validation_started"
EVENT_VALIDATION_COMPLETED = "validation_completed"
EVENT_VALIDATION_FAILED = "validation_failed"
EVENT_DECISION_MADE = "decision_made"
EVENT_EVIDENCE_REUSED = "evidence_reused"
EVENT_CANDIDATE_REJECTED = "candidate_rejected"
EVENT_REPAIR_STARTED = "repair_started"
EVENT_REPAIR_COMPLETED = "repair_completed"
EVENT_LIFECYCLE_COMPLETED = "lifecycle_completed"
EVENT_LIFECYCLE_ABANDONED = "lifecycle_abandoned"

ALL_EVENTS: tuple[str, ...] = (
    EVENT_LIFECYCLE_STARTED,
    EVENT_VALIDATION_STARTED,
    EVENT_VALIDATION_COMPLETED,
    EVENT_VALIDATION_FAILED,
    EVENT_DECISION_MADE,
    EVENT_EVIDENCE_REUSED,
    EVENT_CANDIDATE_REJECTED,
    EVENT_REPAIR_STARTED,
    EVENT_REPAIR_COMPLETED,
    EVENT_LIFECYCLE_COMPLETED,
    EVENT_LIFECYCLE_ABANDONED,
)


@dataclass(frozen=True)
class ValidationEvent:
    """One structured observability event. Data only; carries no behaviour."""

    name: str
    lifecycle_id: str = ""
    task_id: str = ""
    subtask_id: str = ""
    iteration_id: str = ""
    iteration_number: int = 0
    scope: str = ""
    result: str = ""
    defect_fingerprint: str = ""
    detail: str = ""
    at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.name,
            "lifecycle_id": self.lifecycle_id,
            "task_id": self.task_id,
            "subtask_id": self.subtask_id,
            "iteration_id": self.iteration_id,
            "iteration_number": self.iteration_number,
            "scope": self.scope,
            "result": self.result,
            "defect_fingerprint": self.defect_fingerprint,
            "detail": self.detail,
            "at": self.at,
        }


class ValidationEventEmitter:
    """A deliberately minimal fan-out hook, not an event-bus framework.

    This repository has no existing event infrastructure (there is no
    publisher/subscriber, no message broker, no dispatcher anywhere in
    :mod:`local_agent`), and inventing one to carry eleven event names would be
    over-engineering. What exists instead: a list of plain callables plus a
    logger, matching how the rest of the codebase already does optional
    observability (``progress=print`` callbacks in the orchestrator, the
    module-level ``LOGGER`` everywhere else).

    A subscriber that raises is *isolated and dropped from consideration for
    that event only* - observability must never be able to break a validation
    run - and the failure is logged at warning level so a broken subscriber is
    not silent.
    """

    def __init__(self, subscribers: Iterable[Callable[[ValidationEvent], None]] | None = None):
        self._subscribers: list[Callable[[ValidationEvent], None]] = list(subscribers or [])
        #: Bounded ring of recently emitted events, for tests and diagnostics.
        self.emitted: list[ValidationEvent] = []
        self.max_retained = 200
        self.subscriber_errors = 0

    def subscribe(self, callback: Callable[[ValidationEvent], None]) -> None:
        self._subscribers.append(callback)

    def emit(self, event: ValidationEvent) -> ValidationEvent:
        self.emitted.append(event)
        if len(self.emitted) > self.max_retained:
            del self.emitted[: len(self.emitted) - self.max_retained]
        LOGGER.debug("validation event %s", event.to_dict())
        for subscriber in list(self._subscribers):
            try:
                subscriber(event)
            except Exception as exc:  # noqa: BLE001 - a subscriber must never break a run
                self.subscriber_errors += 1
                LOGGER.warning("validation event subscriber failed for %s: %s", event.name, exc)
        return event

    def emit_named(self, name: str, **kwargs: Any) -> ValidationEvent:
        return self.emit(ValidationEvent(name=name, **kwargs))


# -- bounded store ---------------------------------------------------------------


class ValidationLifecycleStore:
    """Bounded, tolerant history of lifecycles.

    Intentionally the same shape as
    :class:`~local_agent.validation_telemetry.ValidationTelemetryStore`: a
    bounded list, a ``corrupted_records_skipped`` counter, and a ``from_dict``
    that drops what it cannot parse rather than raising. A second, differently
    behaved persistence idiom would be a maintenance hazard for no benefit.
    """

    def __init__(
        self,
        *,
        max_lifecycles: int = DEFAULT_MAX_LIFECYCLES,
        max_iterations_per_lifecycle: int = DEFAULT_MAX_ITERATIONS_PER_LIFECYCLE,
    ):
        self.max_lifecycles = max(1, int(max_lifecycles))
        self.max_iterations_per_lifecycle = max(1, int(max_iterations_per_lifecycle))
        self._lifecycles: list[ValidationLifecycleRecord] = []
        #: Entries that could not be deserialised and were dropped. Never
        #: silently ignored: this feeds ``history_trustworthy`` below, which is
        #: what makes corrupt history behave conservatively instead of merely
        #: behaving.
        self.corrupted_records_skipped = 0

    @property
    def lifecycles(self) -> list[ValidationLifecycleRecord]:
        return list(self._lifecycles)

    def __len__(self) -> int:
        return len(self._lifecycles)

    def record(self, lifecycle: ValidationLifecycleRecord) -> ValidationLifecycleRecord:
        """Insert or replace ``lifecycle`` by id, then enforce the bound.

        Replacing in place (rather than appending a second copy) is what makes
        a duplicate lifecycle event idempotent: an orchestrator that re-records
        the same lifecycle after a crash-resume updates it instead of creating
        a divergent twin that every statistic would then double-count.
        """
        lifecycle.max_iterations = self.max_iterations_per_lifecycle
        if len(lifecycle.iterations) > self.max_iterations_per_lifecycle:
            del lifecycle.iterations[
                : len(lifecycle.iterations) - self.max_iterations_per_lifecycle
            ]
        for index, existing in enumerate(self._lifecycles):
            if existing.lifecycle_id == lifecycle.lifecycle_id:
                self._lifecycles[index] = lifecycle
                return lifecycle
        self._lifecycles.append(lifecycle)
        if len(self._lifecycles) > self.max_lifecycles:
            del self._lifecycles[: len(self._lifecycles) - self.max_lifecycles]
        return lifecycle

    def find(self, lifecycle_id: str) -> ValidationLifecycleRecord | None:
        for record in reversed(self._lifecycles):
            if record.lifecycle_id == lifecycle_id:
                return record
        return None

    def for_task(self, task_id: str) -> list[ValidationLifecycleRecord]:
        return [r for r in self._lifecycles if r.task_id == task_id]

    @property
    def history_trustworthy(self) -> bool:
        """False once anything in this store failed to deserialise.

        Consumers treat an untrustworthy store as *no history at all*, which by
        the conservatism rule means the safety floor stands unchanged.
        """
        return self.corrupted_records_skipped == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": LIFECYCLE_SCHEMA_VERSION,
            "max_lifecycles": self.max_lifecycles,
            "max_iterations_per_lifecycle": self.max_iterations_per_lifecycle,
            "corrupted_records_skipped": self.corrupted_records_skipped,
            "lifecycles": [r.to_dict() for r in self._lifecycles],
        }

    @classmethod
    def from_dict(cls, data: Any) -> "ValidationLifecycleStore":
        if not isinstance(data, dict):
            # Not "empty store, carry on": a payload that is not even a mapping
            # is corruption, and must be visible as such.
            store = cls()
            store.corrupted_records_skipped = 1
            return store
        def _int(key: str, default: int) -> int:
            try:
                return int(data.get(key, default) or default)
            except (TypeError, ValueError):
                return default

        store = cls(
            max_lifecycles=_int("max_lifecycles", DEFAULT_MAX_LIFECYCLES),
            max_iterations_per_lifecycle=_int(
                "max_iterations_per_lifecycle", DEFAULT_MAX_ITERATIONS_PER_LIFECYCLE
            ),
        )
        corrupted = 0
        records: list[ValidationLifecycleRecord] = []
        raw = data.get("lifecycles")
        if raw is not None and not isinstance(raw, list):
            corrupted += 1
            raw = []
        for entry in (raw or []):
            if not isinstance(entry, dict):
                corrupted += 1
                continue
            try:
                records.append(ValidationLifecycleRecord.from_dict(entry))
            except Exception:  # noqa: BLE001 - one bad record must not lose the rest
                corrupted += 1
        store._lifecycles = records[-store.max_lifecycles:]
        for record in store._lifecycles:
            record.max_iterations = store.max_iterations_per_lifecycle
        try:
            declared = int(data.get("corrupted_records_skipped", 0) or 0)
        except (TypeError, ValueError):
            declared = 0
            corrupted += 1
        store.corrupted_records_skipped = max(0, declared) + corrupted
        return store


# -- repair-effectiveness metrics --------------------------------------------------


@dataclass
class DefectRecurrence:
    fingerprint: str = ""
    occurrences: int = 0
    lifecycles: int = 0
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "occurrences": self.occurrences,
            "lifecycles": self.lifecycles,
            "description": self.description,
        }


@dataclass
class RepairEffectivenessMetrics:
    """What the lifecycle history actually supports saying about repairs.

    Every rate here is accompanied by its sample size and, where it is used to
    argue anything, by a Wilson bound rather than the bare point estimate -
    the same convention Phase 4.19 established for evidence reliability, for
    the same reason: three lucky samples must not be able to look like a
    hundred. :attr:`insufficient_data` is set whenever the sample is below the
    configured minimum, and consumers are expected to say so rather than quote
    the number as if it were established.

    Deliberately absent, for the same reason Phase 4.19 reports no "recall":
    there is no *repair correctness* metric. Nothing observes whether a repair
    was semantically right - only whether validation subsequently passed - so
    calling the latter "correctness" would be false confidence.
    """

    lifecycles: int = 0
    completed: int = 0
    abandoned: int = 0
    failed: int = 0
    in_progress: int = 0
    #: Lifecycles that reached a terminal state - the only denominator that
    #: supports an outcome-rate claim at all.
    resolved: int = 0
    first_pass_successes: int = 0
    first_pass_success_rate: float = 0.0
    first_pass_success_lower_bound: float = 0.0
    lifecycles_needing_repair: int = 0
    repaired_successfully: int = 0
    repair_success_rate: float = 0.0
    repair_success_lower_bound: float = 0.0
    total_repair_iterations: int = 0
    median_repair_iterations: float = 0.0
    max_repair_iterations: int = 0
    mean_candidate_iterations: float = 0.0
    abandonment_rate: float = 0.0
    abandonment_rate_upper_bound: float = 1.0
    repeated_defect_lifecycles: int = 0
    repeated_defect_rate: float = 0.0
    repeated_defect_rate_upper_bound: float = 1.0
    #: Where defects were caught, by validation stage.
    stage_distribution: dict[str, int] = field(default_factory=dict)
    #: Candidate-stage defects vs. defects that only surfaced after apply - the
    #: quantity that says whether prospective validation is doing its job.
    candidate_stage_defects: int = 0
    post_apply_defects: int = 0
    #: Measured validation duration. Iterations with an unmeasured (0.0)
    #: duration are *excluded*, never averaged in as free, matching
    #: :class:`~local_agent.validation_telemetry.ValidationCostModel`.
    measured_duration_samples: int = 0
    mean_validation_seconds: float = 0.0
    median_validation_seconds: float = 0.0
    top_recurring_defects: list[DefectRecurrence] = field(default_factory=list)
    insufficient_data: bool = True
    min_samples: int = 0
    history_trustworthy: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "lifecycles": self.lifecycles,
            "completed": self.completed,
            "abandoned": self.abandoned,
            "failed": self.failed,
            "in_progress": self.in_progress,
            "resolved": self.resolved,
            "first_pass_successes": self.first_pass_successes,
            "first_pass_success_rate": round(self.first_pass_success_rate, 4),
            "first_pass_success_lower_bound": round(self.first_pass_success_lower_bound, 4),
            "lifecycles_needing_repair": self.lifecycles_needing_repair,
            "repaired_successfully": self.repaired_successfully,
            "repair_success_rate": round(self.repair_success_rate, 4),
            "repair_success_lower_bound": round(self.repair_success_lower_bound, 4),
            "total_repair_iterations": self.total_repair_iterations,
            "median_repair_iterations": round(self.median_repair_iterations, 4),
            "max_repair_iterations": self.max_repair_iterations,
            "mean_candidate_iterations": round(self.mean_candidate_iterations, 4),
            "abandonment_rate": round(self.abandonment_rate, 4),
            "abandonment_rate_upper_bound": round(self.abandonment_rate_upper_bound, 4),
            "repeated_defect_lifecycles": self.repeated_defect_lifecycles,
            "repeated_defect_rate": round(self.repeated_defect_rate, 4),
            "repeated_defect_rate_upper_bound": round(self.repeated_defect_rate_upper_bound, 4),
            "stage_distribution": dict(self.stage_distribution),
            "candidate_stage_defects": self.candidate_stage_defects,
            "post_apply_defects": self.post_apply_defects,
            "measured_duration_samples": self.measured_duration_samples,
            "mean_validation_seconds": round(self.mean_validation_seconds, 4),
            "median_validation_seconds": round(self.median_validation_seconds, 4),
            "top_recurring_defects": [d.to_dict() for d in self.top_recurring_defects],
            "insufficient_data": self.insufficient_data,
            "min_samples": self.min_samples,
            "history_trustworthy": self.history_trustworthy,
        }


def _mean(values: list[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def compute_repair_effectiveness(
    store: ValidationLifecycleStore, *, min_samples: int = 10, top_defects: int = 5
) -> RepairEffectivenessMetrics:
    """Aggregate the lifecycle history. Read-only, pure, order-independent."""
    lifecycles = store.lifecycles
    metrics = RepairEffectivenessMetrics(
        lifecycles=len(lifecycles),
        min_samples=max(1, int(min_samples)),
        history_trustworthy=store.history_trustworthy,
    )
    if not lifecycles:
        return metrics

    completed = [r for r in lifecycles if r.state == LifecycleState.COMPLETED]
    abandoned = [r for r in lifecycles if r.state == LifecycleState.ABANDONED]
    failed = [r for r in lifecycles if r.state == LifecycleState.FAILED]
    resolved = completed + abandoned + failed

    metrics.completed = len(completed)
    metrics.abandoned = len(abandoned)
    metrics.failed = len(failed)
    metrics.resolved = len(resolved)
    metrics.in_progress = len(lifecycles) - len(resolved)

    # First-pass success: reached COMPLETED with zero repair iterations.
    first_pass = [r for r in completed if r.repair_count == 0]
    metrics.first_pass_successes = len(first_pass)
    if resolved:
        metrics.first_pass_success_rate = len(first_pass) / len(resolved)
        metrics.first_pass_success_lower_bound = wilson_lower_bound(
            len(first_pass), len(resolved)
        )
        metrics.abandonment_rate = len(abandoned) / len(resolved)
        _, metrics.abandonment_rate_upper_bound = wilson_bounds(len(abandoned), len(resolved))

    # Repair success: of resolved lifecycles that needed at least one repair,
    # how many still reached COMPLETED.
    needed_repair = [r for r in resolved if r.repair_count > 0]
    repaired_ok = [r for r in needed_repair if r.state == LifecycleState.COMPLETED]
    metrics.lifecycles_needing_repair = len(needed_repair)
    metrics.repaired_successfully = len(repaired_ok)
    if needed_repair:
        metrics.repair_success_rate = len(repaired_ok) / len(needed_repair)
        metrics.repair_success_lower_bound = wilson_lower_bound(
            len(repaired_ok), len(needed_repair)
        )
        repair_counts = [float(r.repair_count) for r in needed_repair]
        metrics.median_repair_iterations = _median(repair_counts)
        metrics.max_repair_iterations = int(max(repair_counts))
    metrics.total_repair_iterations = sum(r.repair_count for r in lifecycles)
    metrics.mean_candidate_iterations = _mean([float(len(r.iterations)) for r in lifecycles])

    # Recurrence, counted per-lifecycle (a defect that recurs inside one
    # lifecycle) and globally by fingerprint.
    recurrence_occurrences: dict[str, int] = {}
    recurrence_lifecycles: dict[str, int] = {}
    descriptions: dict[str, str] = {}
    repeated_lifecycles = 0
    for record in lifecycles:
        signatures = record.defect_signatures()
        if record.recurring_defects():
            repeated_lifecycles += 1
        seen_here: set[str] = set()
        for signature in signatures:
            key = signature.fingerprint
            recurrence_occurrences[key] = recurrence_occurrences.get(key, 0) + 1
            descriptions.setdefault(key, signature.describe())
            if key not in seen_here:
                seen_here.add(key)
                recurrence_lifecycles[key] = recurrence_lifecycles.get(key, 0) + 1
    metrics.repeated_defect_lifecycles = repeated_lifecycles
    if lifecycles:
        metrics.repeated_defect_rate = repeated_lifecycles / len(lifecycles)
        _, metrics.repeated_defect_rate_upper_bound = wilson_bounds(
            repeated_lifecycles, len(lifecycles)
        )

    ranked = sorted(
        recurrence_occurrences.items(), key=lambda kv: (-kv[1], kv[0])
    )[: max(0, int(top_defects))]
    metrics.top_recurring_defects = [
        DefectRecurrence(
            fingerprint=key,
            occurrences=count,
            lifecycles=recurrence_lifecycles.get(key, 0),
            description=descriptions.get(key, ""),
        )
        for key, count in ranked
    ]

    stages: dict[str, int] = {}
    durations: list[float] = []
    for record in lifecycles:
        for iteration in record.iterations:
            if iteration.duration_seconds > 0:
                durations.append(iteration.duration_seconds)
            if iteration.failed:
                stage = (
                    iteration.validation_stage
                    if iteration.validation_stage in ALL_STAGES
                    else STAGE_UNKNOWN
                )
                stages[stage] = stages.get(stage, 0) + 1
    metrics.stage_distribution = stages
    metrics.candidate_stage_defects = stages.get(STAGE_CANDIDATE, 0)
    metrics.post_apply_defects = stages.get(STAGE_POST_APPLY, 0) + stages.get(STAGE_BROAD, 0)
    metrics.measured_duration_samples = len(durations)
    metrics.mean_validation_seconds = _mean(durations)
    metrics.median_validation_seconds = _median(durations)

    metrics.insufficient_data = metrics.resolved < metrics.min_samples
    return metrics


# -- adaptive recommendation (ADVISORY ONLY) ---------------------------------------


@dataclass
class ValidationScopeRecommendation:
    """An advisory scope suggestion *plus* the floor it may not go below.

    The three scope fields are deliberately kept separate rather than collapsed
    into one answer, so a reader (and a test) can see the safety argument
    rather than having to trust it:

    ``safety_floor``
        The broadest scope any hard constraint demands. Comes from the already
        authoritative analysis (Phase 4.17's
        :func:`~local_agent.semantic_impact.recommend_validation_scope`, whose
        output the caller passes in) plus this layer's own hard gates
        (degraded analysis, recurring defects, untrustworthy history).
    ``recommended_scope``
        What history alone would suggest. Advisory. May be narrower than the
        floor - that is exactly the case worth being able to see.
    ``effective_scope``
        ``safest_scope(recommended_scope, safety_floor)`` - by construction the
        broader of the two, computed in one place, never assigned directly.

    Nothing consumes :attr:`recommended_scope` on the authoritative path.
    :attr:`advisory` is a constant ``True`` and exists so that a consumer which
    ignores everything else still cannot mistake this object for a decision.
    """

    recommended_scope: str = SCOPE_BROAD
    safety_floor: str = SCOPE_BROAD
    reasons: list[str] = field(default_factory=list)
    safety_reasons: list[str] = field(default_factory=list)
    data_sufficient: bool = False
    history_trustworthy: bool = True
    samples: int = 0
    advisory: bool = True

    @property
    def effective_scope(self) -> str:
        """The only scope a consumer may act on. Never narrower than the floor."""
        return safest_scope(self.recommended_scope, self.safety_floor)

    @property
    def conflicts_with_floor(self) -> bool:
        """True when history wanted something narrower than safety permits."""
        return self.effective_scope != self.recommended_scope

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommended_scope": self.recommended_scope,
            "safety_floor": self.safety_floor,
            "effective_scope": self.effective_scope,
            "conflicts_with_floor": self.conflicts_with_floor,
            "reasons": list(self.reasons),
            "safety_reasons": list(self.safety_reasons),
            "data_sufficient": self.data_sufficient,
            "history_trustworthy": self.history_trustworthy,
            "samples": self.samples,
            "advisory": self.advisory,
        }


class AdaptiveValidationRecommender:
    """Turns lifecycle history into an *advisory* scope recommendation.

    This class is not a decision authority and has no path to becoming one: it
    has no reference to
    :class:`~local_agent.validation_decision.ValidationDecisionEngine`, returns
    a value object rather than mutating anything, and every consumer must go
    through :attr:`ValidationScopeRecommendation.effective_scope`, which
    re-applies the floor. The authoritative decision remains exactly where
    Phase 4.18 put it.
    """

    def __init__(self, *, min_samples: int = 10):
        self.min_samples = max(1, int(min_samples))

    def recommend(
        self,
        *,
        safety_floor: str,
        store: ValidationLifecycleStore | None = None,
        degraded_analysis: bool = False,
        recent_defect_fingerprints: Iterable[str] = (),
    ) -> ValidationScopeRecommendation:
        """Compute one recommendation.

        ``safety_floor`` is the scope the authoritative analysis already
        demands; it is the *input*, not something this method may reconsider.
        Every branch below can only ever leave the floor where it is or push it
        broader - there is no code path that lowers it.
        """
        metrics = (
            compute_repair_effectiveness(store, min_samples=self.min_samples)
            if store is not None
            else RepairEffectivenessMetrics(min_samples=self.min_samples)
        )
        floor = safest_scope(safety_floor)
        safety_reasons: list[str] = [
            f"authoritative analysis requires at least '{safety_floor}' scope"
        ]

        if degraded_analysis:
            floor = safest_scope(floor, SCOPE_BROAD)
            safety_reasons.append(
                "impact analysis reported degraded or unresolved evidence; "
                "history cannot argue a narrower scope for an analysis that is "
                "itself incomplete"
            )

        if not metrics.history_trustworthy:
            floor = safest_scope(floor, SCOPE_BROAD)
            safety_reasons.append(
                "lifecycle history contains records that failed to deserialise; "
                "corrupt history is treated as no history and cannot narrow anything"
            )

        recurring = set(str(f) for f in recent_defect_fingerprints if f)
        known_recurring = {
            d.fingerprint for d in metrics.top_recurring_defects if d.occurrences > 1
        }
        overlap = recurring & known_recurring
        if overlap:
            floor = safest_scope(floor, SCOPE_BROAD)
            safety_reasons.append(
                f"{len(overlap)} defect signature(s) in this lifecycle have recurred "
                "before; a repeatedly-escaping defect is the last thing a narrower "
                "scope should be trusted with"
            )

        reasons: list[str] = []
        if store is None or metrics.resolved == 0:
            reasons.append("no lifecycle history; recommendation defaults to the safety floor")
            recommended = floor
        elif metrics.insufficient_data:
            reasons.append(
                f"only {metrics.resolved} resolved lifecycle(s), below the configured "
                f"minimum of {metrics.min_samples}; insufficient data may not narrow "
                "validation, so the recommendation stays at the safety floor"
            )
            recommended = floor
        elif not metrics.history_trustworthy:
            reasons.append("history is not trustworthy; recommendation stays at the safety floor")
            recommended = floor
        elif metrics.repeated_defect_rate_upper_bound > 0.25:
            recommended = safest_scope(floor, SCOPE_EXPANDED)
            reasons.append(
                "the pessimistic bound on the repeated-defect rate is "
                f"{metrics.repeated_defect_rate_upper_bound:.2f}; recurring defects "
                "argue for widening, never narrowing"
            )
        elif metrics.abandonment_rate_upper_bound > 0.5:
            recommended = safest_scope(floor, SCOPE_EXPANDED)
            reasons.append(
                "the pessimistic bound on the abandonment rate is "
                f"{metrics.abandonment_rate_upper_bound:.2f}; a loop that frequently "
                "gives up is not one to trust with a narrower scope"
            )
        elif metrics.first_pass_success_lower_bound >= 0.9 and metrics.candidate_stage_defects >= (
            metrics.post_apply_defects
        ):
            # The only branch that can suggest anything narrower than BROAD -
            # and even it cannot go below the floor, because ``recommended`` is
            # combined with ``floor`` in ``effective_scope``.
            recommended = SCOPE_TARGETED
            reasons.append(
                "conservative lower bound on first-pass success is "
                f"{metrics.first_pass_success_lower_bound:.2f} with defects caught "
                "predominantly at candidate stage; history alone would support a "
                "targeted scope (advisory only - the safety floor still applies)"
            )
        else:
            recommended = floor
            reasons.append(
                "history is trustworthy and sufficient but does not clear the bar "
                "for suggesting anything narrower; recommendation stays at the floor"
            )

        return ValidationScopeRecommendation(
            recommended_scope=recommended,
            safety_floor=floor,
            reasons=reasons,
            safety_reasons=safety_reasons,
            data_sufficient=not metrics.insufficient_data and metrics.resolved > 0,
            history_trustworthy=metrics.history_trustworthy,
            samples=metrics.resolved,
        )


# -- persistence manager -----------------------------------------------------------

_LIFECYCLE_LOCKS: dict[str, threading.Lock] = {}
_LIFECYCLE_LOCKS_GUARD = threading.Lock()


def _lock_for(key: str) -> threading.Lock:
    """Process-wide lock keyed by repository path.

    Identical in shape and in stated limits to
    :func:`local_agent.validation_telemetry._lock_for`: it serialises
    read-modify-write cycles between threads in *one* process (the
    parallel-worktree case). Cross-process concurrency is explicitly **not**
    provided, matching every other store in :mod:`local_agent.storage`.
    """
    with _LIFECYCLE_LOCKS_GUARD:
        lock = _LIFECYCLE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _LIFECYCLE_LOCKS[key] = lock
        return lock


class ValidationLifecycleManager:
    """Owns read-modify-write access to one repository's lifecycle store.

    Every mutating method reloads under the per-repository lock, mutates, and
    saves before releasing - the same trade Phase 4.19 made, for the same
    reason: an extra load+save per lifecycle event is negligible next to a
    validation command run, and it means a concurrent writer's update is never
    lost to a stale in-memory cache.
    """

    def __init__(
        self,
        storage: Any,
        project_root: str | Path,
        *,
        max_lifecycles: int = DEFAULT_MAX_LIFECYCLES,
        max_iterations_per_lifecycle: int = DEFAULT_MAX_ITERATIONS_PER_LIFECYCLE,
        emitter: ValidationEventEmitter | None = None,
    ):
        self.storage = storage
        self.project_root = Path(project_root)
        self.max_lifecycles = max_lifecycles
        self.max_iterations_per_lifecycle = max_iterations_per_lifecycle
        self.emitter = emitter or ValidationEventEmitter()
        try:
            key = str(self.project_root.resolve())
        except OSError:  # pragma: no cover - defensive, e.g. a vanished cwd
            key = str(self.project_root)
        self._lock = _lock_for(key)

    def _load(self) -> ValidationLifecycleStore:
        if not hasattr(self.storage, "load_validation_lifecycle"):
            return ValidationLifecycleStore(
                max_lifecycles=self.max_lifecycles,
                max_iterations_per_lifecycle=self.max_iterations_per_lifecycle,
            )
        store = self.storage.load_validation_lifecycle()
        store.max_lifecycles = max(1, int(self.max_lifecycles))
        store.max_iterations_per_lifecycle = max(1, int(self.max_iterations_per_lifecycle))
        return store

    def _save(self, store: ValidationLifecycleStore) -> None:
        if hasattr(self.storage, "save_validation_lifecycle"):
            self.storage.save_validation_lifecycle(store)

    # -- lifecycle operations ---------------------------------------------

    def start(
        self,
        *,
        task_id: str = "",
        subtask_id: str = "",
        provider: str = "",
        model: str = "",
    ) -> ValidationLifecycleRecord:
        record = ValidationLifecycleRecord(
            repository_id=repository_identity(self.project_root),
            task_id=task_id,
            subtask_id=subtask_id,
            provider=provider,
            model=model,
            max_iterations=self.max_iterations_per_lifecycle,
        )
        with self._lock:
            store = self._load()
            store.record(record)
            self._save(store)
        self.emitter.emit_named(
            EVENT_LIFECYCLE_STARTED,
            lifecycle_id=record.lifecycle_id,
            task_id=task_id,
            subtask_id=subtask_id,
        )
        return record

    def get(self, lifecycle_id: str) -> ValidationLifecycleRecord | None:
        with self._lock:
            store = self._load()
        return store.find(lifecycle_id)

    def lifecycles(self) -> list[ValidationLifecycleRecord]:
        with self._lock:
            store = self._load()
        return store.lifecycles

    def transition(
        self, lifecycle_id: str, requested: str, *, reason: str = ""
    ) -> ValidationLifecycleRecord | None:
        """Apply a transition durably. Raises on an invalid transition.

        Returns ``None`` when ``lifecycle_id`` is unknown - a missing lifecycle
        must never be silently treated as a successful transition.
        """
        with self._lock:
            store = self._load()
            record = store.find(lifecycle_id)
            if record is None:
                return None
            record.transition(requested, reason=reason)
            store.record(record)
            self._save(store)
        if requested == LifecycleState.COMPLETED:
            self.emitter.emit_named(
                EVENT_LIFECYCLE_COMPLETED, lifecycle_id=lifecycle_id, result=requested
            )
        elif requested in (LifecycleState.ABANDONED, LifecycleState.FAILED):
            self.emitter.emit_named(
                EVENT_LIFECYCLE_ABANDONED, lifecycle_id=lifecycle_id, result=requested,
                detail=reason,
            )
        return record

    def record_iteration(
        self, lifecycle_id: str, iteration: ValidationIterationRecord
    ) -> ValidationIterationRecord | None:
        """Append one iteration durably; ``None`` when the lifecycle is unknown."""
        with self._lock:
            store = self._load()
            record = store.find(lifecycle_id)
            if record is None:
                return None
            record.add_iteration(iteration)
            store.record(record)
            self._save(store)
        name = (
            EVENT_VALIDATION_FAILED
            if iteration.failed
            else EVENT_VALIDATION_COMPLETED
            if iteration.passed
            else EVENT_VALIDATION_STARTED
        )
        self.emitter.emit_named(
            name,
            lifecycle_id=lifecycle_id,
            iteration_id=iteration.iteration_id,
            iteration_number=iteration.iteration_number,
            scope=iteration.scope,
            result=iteration.validation_result,
            defect_fingerprint=(
                iteration.defect_signature.fingerprint
                if iteration.defect_signature is not None
                else ""
            ),
        )
        if iteration.kind == ITERATION_REPAIR:
            self.emitter.emit_named(
                EVENT_REPAIR_COMPLETED if iteration.passed else EVENT_REPAIR_STARTED,
                lifecycle_id=lifecycle_id,
                iteration_id=iteration.iteration_id,
                iteration_number=iteration.iteration_number,
                result=iteration.validation_result,
            )
        return iteration

    def effectiveness(self, *, min_samples: int = 10) -> RepairEffectivenessMetrics:
        with self._lock:
            store = self._load()
        return compute_repair_effectiveness(store, min_samples=min_samples)

    def recommend(
        self,
        *,
        safety_floor: str,
        degraded_analysis: bool = False,
        recent_defect_fingerprints: Iterable[str] = (),
        min_samples: int = 10,
    ) -> ValidationScopeRecommendation:
        with self._lock:
            store = self._load()
        return AdaptiveValidationRecommender(min_samples=min_samples).recommend(
            safety_floor=safety_floor,
            store=store,
            degraded_analysis=degraded_analysis,
            recent_defect_fingerprints=recent_defect_fingerprints,
        )
