"""Phase 4.19: empirical validation intelligence telemetry and calibration.

Phase 4.17/4.18 gave DungX a validation *decision*: given an impact analysis,
which scope (targeted/expanded/broad) and which commands. What was still
missing was any record of that decision surviving past the moment it was made
- there was no way to ask, after the fact, "was that decision actually
correct", "which evidence types are trustworthy", or "would a different
confidence threshold have chosen differently".

This module adds that observability layer, in four pieces:

``ValidationDecisionRecord``
    One structured, bounded snapshot of a single validation decision: what
    evidence supported it, what was selected, what reuse happened, and -
    filled in later, once known - what actually happened when it ran and a
    later, unconditionally-run broader validation was checked against it.

Outcome linking (:func:`classify_outcome`)
    Turns "targeted passed, then mandatory full-suite also passed/failed"
    into an explicit, principled distinction between *validation outcome*
    (did the run pass) and *decision quality* (did picking that scope turn
    out to be defensible) - the two are not the same thing, and conflating
    them is exactly the kind of false confidence this phase exists to avoid.

``ValidationTelemetryStore`` + reliability estimation
    A bounded, append-mostly history of decisions and calibration
    observations, plus a conservative (Wilson lower-bound) reliability
    estimate per evidence type - never a point estimate alone, so a handful
    of lucky observations cannot look as trustworthy as a hundred.

``ShadowCalibrationEngine``
    Computes, for one decision, what a calibrated confidence *would* have
    recommended, without ever feeding that back into the real decision. It
    is read-only with respect to the authoritative validation path: nothing
    in this module can narrow, skip, or otherwise weaken what
    :class:`~local_agent.validation_decision.ValidationDecisionEngine`
    already decided. See ``README.md`` for the full safety-floor writeup.

Persistence is intentionally separate from :class:`~local_agent.models.Task`
and :class:`~local_agent.models.Checkpoint`: telemetry is long-lived,
cross-task history, not per-task resumable state, so it lives in its own
bounded store (mirroring how :class:`~local_agent.knowledge.KnowledgeGraphManager`
persists the repository knowledge graph separately). Old checkpoints and old
tasks are completely unaffected by anything in this module.
"""

from __future__ import annotations

import datetime
import hashlib
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .dependency_resolution import ALL_EVIDENCE_TYPES, CONFIDENCE_BY_EVIDENCE_TYPE, confidence_for
from .evidence import REASON_OK, REASON_STALE

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a runtime import cycle
    from .semantic_impact import ChangeImpactReport
    from .validation_decision import ValidationDecision

#: Shape/behaviour version of the decision-record schema and the classification
#: rules in this module. Distinct from
#: :data:`local_agent.semantic_impact.SEMANTIC_ANALYZER_SCHEMA_VERSION` (what
#: produced the impact report) and from the evidence-reuse policy fingerprint
#: (what configuration was in effect) - this one versions "how this module
#: itself interprets a decision", so a future change to the classification
#: rules below does not get silently applied to historical records.
DECISION_POLICY_VERSION = "4.19.0"
#: Version of the calibration math in :func:`compute_calibration_signal`. Bumped
#: whenever the formula (not just its inputs) changes, so a stored calibration
#: observation can be excluded from analysis under a different algorithm
#: version rather than silently mixed in.
CALIBRATION_ALGORITHM_VERSION = "1"

# -- outcome / decision-quality vocabulary -----------------------------------

OUTCOME_PENDING = "pending"
OUTCOME_VALIDATION_PASSED = "validation_passed"
OUTCOME_VALIDATION_FAILED = "validation_failed"

#: Decision quality is a judgement about the *scope choice*, not about whether
#: the run happened to pass. See ``classify_outcome`` for exactly when each
#: applies; this list intentionally keeps validation outcome and decision
#: quality as two separate axes rather than collapsing them into one status.
QUALITY_UNCONFIRMED = "unconfirmed"
QUALITY_CONSISTENT = "consistent_no_contradiction"
QUALITY_BROAD_NOT_PROVEN_NECESSARY = "broad_not_proven_necessary"
QUALITY_TARGETED_CAUGHT_DEFECT = "targeted_caught_defect"
QUALITY_TARGETED_MISSED_DEFECT = "targeted_missed_defect"
QUALITY_VALIDATION_FAILED = "validation_failed_no_scope_judgement"

#: A pseudo evidence-type recorded when the impact analysis itself reported a
#: degradation (unresolved dependency, unparseable file, dynamic import that
#: could not be resolved, ...). Never a real dependency-evidence type, so it
#: cannot collide with anything in ``dependency_resolution``; it exists purely
#: as an explicit hard gate input for the safety floor in
#: :func:`compute_calibration_signal`.
DEGRADED_EVIDENCE_MARKER = "analysis_degraded"

#: Confidence bucketed to a 0..1 score the same way
#: :func:`local_agent.validation_decision._confidence_score` does, kept as an
#: independent constant here (not imported) so this module's classification
#: logic is not accidentally coupled to a private helper in another module.
_CONFIDENCE_ORDER: tuple[str, ...] = ("low", "medium", "high")

#: Default bounds. Conservative on purpose: telemetry must stay cheap and
#: bounded regardless of how long a repository has been running the agent.
DEFAULT_MAX_DECISIONS = 500
DEFAULT_MAX_OBSERVATIONS = 500
#: How many changed-file / changed-symbol / uncertainty-source entries a single
#: record keeps. This is telemetry, not an audit log: enough to explain a
#: decision, never enough to make one record dominate the bounded store.
_MAX_LIST_FIELD_ENTRIES = 50


def confidence_score(level: str) -> float:
    """Numeric position of ``level`` in the low/medium/high order, in ``[0, 1]``.

    Pure presentation/analysis convenience - nothing upstream is decided from
    this value. Unknown input fails to ``0.0`` (the safe/low end), never to a
    mid-point guess.
    """
    if level not in _CONFIDENCE_ORDER:
        return 0.0
    return _CONFIDENCE_ORDER.index(level) / (len(_CONFIDENCE_ORDER) - 1)


def _level_for_score(score: float) -> str:
    """Inverse of :func:`confidence_score`: nearest bucket, ties round down.

    Used only to translate a calibrated *score* back into a scope-mapping
    bucket for the shadow comparison; never used on the authoritative path.
    """
    score = max(0.0, min(1.0, score))
    thresholds = [i / (len(_CONFIDENCE_ORDER) - 1) for i in range(len(_CONFIDENCE_ORDER))]
    best_index = 0
    best_distance = abs(score - thresholds[0])
    for index, threshold in enumerate(thresholds[1:], start=1):
        distance = abs(score - threshold)
        if distance < best_distance:
            best_index, best_distance = index, distance
    return _CONFIDENCE_ORDER[best_index]


def _scope_for_level(level: str) -> str:
    """The confidence-only scope mapping, mirroring
    :func:`local_agent.semantic_impact.recommend_validation_scope`'s first
    branch exactly (HIGH -> targeted, MEDIUM -> expanded, LOW -> broad).
    Deliberately does not consider fan-out/degradation/bounds - the shadow
    comparison is explicit that it isolates the confidence-derived component
    only; see :class:`ShadowComparison`.
    """
    return {"high": "targeted", "medium": "expanded", "low": "broad"}.get(level, "broad")


def repository_identity(root: str | Path) -> str:
    """Stable, privacy-conscious identifier for a repository.

    A digest of the resolved path, not the path itself: telemetry that lives
    on disk indefinitely should not need to carry a user's real directory
    layout in plain text, and a digest is all any consumer of this module
    actually needs (grouping records by repository, not displaying a path).
    """
    resolved = str(Path(root).expanduser().resolve()).replace("\\", "/").lower()
    return hashlib.sha256(resolved.encode("utf-8", "replace")).hexdigest()[:16]


def _bounded(values: Any, limit: int = _MAX_LIST_FIELD_ENTRIES) -> list[str]:
    items = sorted({str(v) for v in (values or []) if v})
    return items[:limit]


# -- evidence-type extraction --------------------------------------------------


def evidence_types_for_impact(impact: "ChangeImpactReport") -> frozenset[str]:
    """The set of evidence-vocabulary labels backing one impact report.

    Prefers each target's fine-grained :mod:`dependency_resolution` evidence
    types where present; falls back to the coarser tier name (e.g.
    ``call_graph_match``) for targets that only ever carry tier-level
    evidence. This is deliberately a *label set*, not a score: reliability is
    always computed from recorded outcomes, never from this function.
    """
    labels: set[str] = set()
    for target in impact.validation_targets:
        if target.dependency_evidence:
            labels.update(evidence.evidence_type for evidence in target.dependency_evidence)
        elif target.tier:
            labels.add(target.tier)
    return frozenset(labels)


def impact_is_degraded(impact: "ChangeImpactReport") -> bool:
    """True when the impact analysis itself reports reduced trustworthiness.

    Any of: a recorded degradation reason, an unresolved dependent symbol, or
    a genuine :mod:`dependency_resolution` evidence type in this decision
    whose fixed confidence is exactly 0.0 (currently only an unresolved
    dynamic import) counts. This is the single predicate the calibration
    safety floor consults to refuse ever treating a structurally-uncertain
    decision as calibratable - see :func:`compute_calibration_signal`.

    Deliberately only checks labels that are members of
    :data:`~local_agent.dependency_resolution.ALL_EVIDENCE_TYPES`: a coarse
    tier-name fallback (e.g. ``call_graph_match``, used by
    :func:`evidence_types_for_impact` for targets with no fine-grained
    dependency evidence) is not a member of that vocabulary, and
    :func:`~local_agent.dependency_resolution.confidence_for` fails closed to
    ``0.0`` for *any* unrecognised label - checking it here would make almost
    every decision look "degraded" and defeat the safety floor by making it
    unconditional rather than evidence-specific.
    """
    if impact.evidence.degradations or impact.unresolved_symbols:
        return True
    return any(
        confidence_for(label) <= 0.0
        for label in evidence_types_for_impact(impact)
        if label in ALL_EVIDENCE_TYPES
    )


# -- the decision record -------------------------------------------------------


@dataclass
class ShadowComparison:
    """What a calibrated confidence would have recommended, for comparison
    only. Never applied to the real decision; see module docstring."""

    computed: bool = False
    would_narrow: bool = False
    would_broaden: bool = False
    confidence_delta: float = 0.0
    shadow_scope: str = ""
    shadow_confidence_level: str = ""
    reasons: list[str] = field(default_factory=list)
    #: True when a hard safety-floor gate (degraded evidence, insufficient
    #: samples) suppressed an upward adjustment that the raw statistics alone
    #: would otherwise have supported. A concrete, testable witness that the
    #: floor in Part 7 is a real code path, not just a comment.
    safety_override: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "computed": self.computed,
            "would_narrow": self.would_narrow,
            "would_broaden": self.would_broaden,
            "confidence_delta": round(self.confidence_delta, 4),
            "shadow_scope": self.shadow_scope,
            "shadow_confidence_level": self.shadow_confidence_level,
            "reasons": list(self.reasons),
            "safety_override": self.safety_override,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "ShadowComparison":
        if not isinstance(data, dict):
            return cls()
        return cls(
            computed=bool(data.get("computed", False)),
            would_narrow=bool(data.get("would_narrow", False)),
            would_broaden=bool(data.get("would_broaden", False)),
            confidence_delta=float(data.get("confidence_delta", 0.0) or 0.0),
            shadow_scope=str(data.get("shadow_scope", "")),
            shadow_confidence_level=str(data.get("shadow_confidence_level", "")),
            reasons=[str(r) for r in (data.get("reasons") or [])],
            safety_override=bool(data.get("safety_override", False)),
        )


@dataclass
class ValidationDecisionRecord:
    """One bounded, serialisable snapshot of a validation decision.

    Carries no raw source code, stdout/stderr, or file system paths outside
    the repository-relative changed/affected file list already surfaced
    elsewhere in this codebase (:class:`~local_agent.evidence.ValidationEvidence`
    keeps the same kind of information) - nothing here raises the privacy bar
    already accepted by Phase 4.17/4.18.
    """

    decision_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    repository_id: str = ""
    tree_fingerprint: str = ""
    changed_files: list[str] = field(default_factory=list)
    changed_symbols: list[str] = field(default_factory=list)
    evidence_types: list[str] = field(default_factory=list)
    scope: str = "broad"
    confidence_level: str = "low"
    confidence_score: float = 0.0
    uncertainty_sources: list[str] = field(default_factory=list)
    selected_command_count: int = 0
    reused_command_count: int = 0
    denied_command_count: int = 0
    #: Tally of :mod:`local_agent.evidence` ``REASON_*`` denial reasons for
    #: this decision's reuse attempts. Bounded by construction: the key
    #: vocabulary is the small fixed set of reason constants, never free text.
    reuse_reasons: dict[str, int] = field(default_factory=dict)
    policy_fingerprint: str = ""
    analyzer_version: str = ""
    decision_policy_version: str = DECISION_POLICY_VERSION
    outcome: str = OUTCOME_PENDING
    decision_quality: str = QUALITY_UNCONFIRMED
    targeted_duration_seconds: float = 0.0
    broad_duration_seconds: float = 0.0
    time_saved_seconds: float = 0.0
    #: True when :func:`impact_is_degraded` held for the impact report behind
    #: this decision. Stored (rather than recomputed) because the impact report
    #: itself is deliberately *not* retained, and Part 15's "analyzer failure
    #: rate" needs it. Defaults to ``False`` so a record written by the
    #: original Phase 4.19 build - which had no such field - loads unchanged
    #: and is simply not counted as degraded.
    degraded_analysis: bool = False
    shadow: ShadowComparison = field(default_factory=ShadowComparison)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "timestamp": self.timestamp,
            "repository_id": self.repository_id,
            "tree_fingerprint": self.tree_fingerprint,
            "changed_files": list(self.changed_files),
            "changed_symbols": list(self.changed_symbols),
            "evidence_types": list(self.evidence_types),
            "scope": self.scope,
            "confidence_level": self.confidence_level,
            "confidence_score": round(self.confidence_score, 4),
            "uncertainty_sources": list(self.uncertainty_sources),
            "selected_command_count": self.selected_command_count,
            "reused_command_count": self.reused_command_count,
            "denied_command_count": self.denied_command_count,
            "reuse_reasons": dict(self.reuse_reasons),
            "policy_fingerprint": self.policy_fingerprint,
            "analyzer_version": self.analyzer_version,
            "decision_policy_version": self.decision_policy_version,
            "outcome": self.outcome,
            "decision_quality": self.decision_quality,
            "targeted_duration_seconds": round(self.targeted_duration_seconds, 4),
            "broad_duration_seconds": round(self.broad_duration_seconds, 4),
            "time_saved_seconds": round(self.time_saved_seconds, 4),
            "degraded_analysis": self.degraded_analysis,
            "shadow": self.shadow.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "ValidationDecisionRecord":
        if not isinstance(data, dict):
            return cls()
        reuse_reasons = data.get("reuse_reasons")
        return cls(
            decision_id=str(data.get("decision_id") or uuid.uuid4().hex),
            timestamp=str(data.get("timestamp", "")),
            repository_id=str(data.get("repository_id", "")),
            tree_fingerprint=str(data.get("tree_fingerprint", "")),
            changed_files=[str(f) for f in (data.get("changed_files") or [])],
            changed_symbols=[str(s) for s in (data.get("changed_symbols") or [])],
            evidence_types=[str(e) for e in (data.get("evidence_types") or [])],
            scope=str(data.get("scope", "broad")),
            confidence_level=str(data.get("confidence_level", "low")),
            confidence_score=float(data.get("confidence_score", 0.0) or 0.0),
            uncertainty_sources=[str(u) for u in (data.get("uncertainty_sources") or [])],
            selected_command_count=int(data.get("selected_command_count", 0) or 0),
            reused_command_count=int(data.get("reused_command_count", 0) or 0),
            denied_command_count=int(data.get("denied_command_count", 0) or 0),
            reuse_reasons={
                str(k): int(v) for k, v in reuse_reasons.items()
            } if isinstance(reuse_reasons, dict) else {},
            policy_fingerprint=str(data.get("policy_fingerprint", "")),
            analyzer_version=str(data.get("analyzer_version", "")),
            decision_policy_version=str(
                data.get("decision_policy_version", DECISION_POLICY_VERSION)
            ),
            outcome=str(data.get("outcome", OUTCOME_PENDING)),
            decision_quality=str(data.get("decision_quality", QUALITY_UNCONFIRMED)),
            targeted_duration_seconds=float(data.get("targeted_duration_seconds", 0.0) or 0.0),
            broad_duration_seconds=float(data.get("broad_duration_seconds", 0.0) or 0.0),
            time_saved_seconds=float(data.get("time_saved_seconds", 0.0) or 0.0),
            degraded_analysis=bool(data.get("degraded_analysis", False)),
            shadow=ShadowComparison.from_dict(data.get("shadow")),
        )


def build_decision_record(
    impact: "ChangeImpactReport",
    decision: "ValidationDecision",
    *,
    root: str | Path,
    reuse_reasons: dict[str, int] | None = None,
) -> ValidationDecisionRecord:
    """Construct the record for one already-made
    :class:`~local_agent.validation_decision.ValidationDecision`.

    Pure with respect to its inputs (given the same impact/decision it always
    produces the same record modulo ``decision_id``/``timestamp``) so it can
    be unit-tested without an orchestrator, a ledger, or the filesystem.
    """
    from .evidence import compute_state_fingerprint

    relevant_files = sorted(set(impact.changed_files) | set(impact.affected_files))
    return ValidationDecisionRecord(
        repository_id=repository_identity(root),
        tree_fingerprint=compute_state_fingerprint(root, relevant_files),
        changed_files=_bounded(impact.changed_files),
        changed_symbols=_bounded(symbol.qualified_name for symbol in impact.changed_symbols),
        evidence_types=_bounded(evidence_types_for_impact(impact)),
        scope=decision.scope,
        confidence_level=decision.confidence_level,
        confidence_score=decision.confidence_score,
        uncertainty_sources=_bounded(decision.uncertainty_sources),
        selected_command_count=len(decision.selected_commands),
        reused_command_count=decision.reused_count,
        denied_command_count=decision.denied_count,
        reuse_reasons=dict(reuse_reasons or {}),
        time_saved_seconds=decision.time_saved_seconds,
        degraded_analysis=impact_is_degraded(impact),
    )


# -- outcome classification ----------------------------------------------------


def classify_outcome(
    *,
    scope: str,
    targeted_ran: bool,
    targeted_failed: bool,
    broad_ran: bool,
    broad_failed: bool,
) -> tuple[str, str]:
    """Return ``(outcome, decision_quality)`` for one finished validation run.

    This is the one place Part 3's distinction between *validation outcome*
    and *decision quality* is made concrete:

    * A targeted-scope decision whose targeted commands passed, followed by a
      mandatory broader run that *failed*, means the targeted scope let a
      defect through - :data:`QUALITY_TARGETED_MISSED_DEFECT`. This is the
      escape signal the whole phase exists to surface.
    * A targeted-scope decision whose targeted commands themselves failed
      caught exactly what they were selected to catch -
      :data:`QUALITY_TARGETED_CAUGHT_DEFECT` - a positive signal about the
      decision even though the overall run's outcome is a failure.
    * A broad (or expanded) decision that passed proves nothing about whether
      broad was *necessary* - there was no narrower alternative run to
      contradict it - so it is recorded as
      :data:`QUALITY_BROAD_NOT_PROVEN_NECESSARY`, not treated as validated.
    * Anything where a step failed outside of the two cases above is
      :data:`QUALITY_VALIDATION_FAILED`: a real failure, but not one that says
      anything about whether the scope choice itself was right or wrong.

    ``targeted_ran=False`` (no targeted commands were selected at all) always
    yields :data:`QUALITY_UNCONFIRMED` for a nominally-targeted scope, since
    there is nothing to compare against.
    """
    if targeted_ran and targeted_failed:
        return OUTCOME_VALIDATION_FAILED, QUALITY_TARGETED_CAUGHT_DEFECT

    if not broad_ran:
        # The mandatory broad/full-suite run did not execute (e.g. an earlier
        # failure short-circuited it via a path this function was not told
        # about). Nothing further can be concluded.
        return (
            OUTCOME_VALIDATION_FAILED if targeted_failed else OUTCOME_PENDING,
            QUALITY_UNCONFIRMED,
        )

    if broad_failed:
        if scope == "targeted" and targeted_ran:
            return OUTCOME_VALIDATION_FAILED, QUALITY_TARGETED_MISSED_DEFECT
        return OUTCOME_VALIDATION_FAILED, QUALITY_VALIDATION_FAILED

    # Broad run passed.
    if scope == "targeted":
        if targeted_ran:
            return OUTCOME_VALIDATION_PASSED, QUALITY_CONSISTENT
        return OUTCOME_VALIDATION_PASSED, QUALITY_UNCONFIRMED
    return OUTCOME_VALIDATION_PASSED, QUALITY_BROAD_NOT_PROVEN_NECESSARY


# -- calibration observations --------------------------------------------------


@dataclass
class CalibrationObservation:
    """One (predicted confidence, observed outcome) pair, kept for analysis.

    Deliberately does not store source code, diffs, or command output - only
    the evidence-type labels, the scope/outcome vocabulary, and small numeric
    costs, so a bounded history of these can be retained indefinitely without
    becoming a second copy of the repository's history.
    """

    evidence_types: tuple[str, ...] = ()
    predicted_confidence: str = "low"
    selected_scope: str = "broad"
    actual_validation_scope: str = "broad"
    outcome: str = OUTCOME_PENDING
    decision_quality: str = QUALITY_UNCONFIRMED
    later_broader_validation_found_defect: bool = False
    validation_cost_seconds: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    policy_fingerprint: str = ""
    analyzer_version: str = ""
    calibration_version: str = CALIBRATION_ALGORITHM_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_types": list(self.evidence_types),
            "predicted_confidence": self.predicted_confidence,
            "selected_scope": self.selected_scope,
            "actual_validation_scope": self.actual_validation_scope,
            "outcome": self.outcome,
            "decision_quality": self.decision_quality,
            "later_broader_validation_found_defect": self.later_broader_validation_found_defect,
            "validation_cost_seconds": round(self.validation_cost_seconds, 4),
            "timestamp": self.timestamp,
            "policy_fingerprint": self.policy_fingerprint,
            "analyzer_version": self.analyzer_version,
            "calibration_version": self.calibration_version,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "CalibrationObservation":
        if not isinstance(data, dict):
            return cls()
        return cls(
            evidence_types=tuple(str(e) for e in (data.get("evidence_types") or [])),
            predicted_confidence=str(data.get("predicted_confidence", "low")),
            selected_scope=str(data.get("selected_scope", "broad")),
            actual_validation_scope=str(data.get("actual_validation_scope", "broad")),
            outcome=str(data.get("outcome", OUTCOME_PENDING)),
            decision_quality=str(data.get("decision_quality", QUALITY_UNCONFIRMED)),
            later_broader_validation_found_defect=bool(
                data.get("later_broader_validation_found_defect", False)
            ),
            validation_cost_seconds=float(data.get("validation_cost_seconds", 0.0) or 0.0),
            timestamp=str(data.get("timestamp", "")),
            policy_fingerprint=str(data.get("policy_fingerprint", "")),
            analyzer_version=str(data.get("analyzer_version", "")),
            calibration_version=str(data.get("calibration_version", CALIBRATION_ALGORITHM_VERSION)),
        )

    @classmethod
    def from_record(cls, record: ValidationDecisionRecord) -> "CalibrationObservation":
        return cls(
            evidence_types=tuple(sorted(record.evidence_types)),
            predicted_confidence=record.confidence_level,
            selected_scope=record.scope,
            actual_validation_scope=record.scope,
            outcome=record.outcome,
            decision_quality=record.decision_quality,
            later_broader_validation_found_defect=(
                record.decision_quality == QUALITY_TARGETED_MISSED_DEFECT
            ),
            validation_cost_seconds=round(
                record.targeted_duration_seconds + record.broad_duration_seconds, 4
            ),
            policy_fingerprint=record.policy_fingerprint,
            analyzer_version=record.analyzer_version,
        )


# -- reliability estimation ----------------------------------------------------


def wilson_bounds(successes: int, trials: int, *, z: float = 1.96) -> tuple[float, float]:
    """Both ends of the Wilson score interval for ``successes/trials``.

    Two callers need opposite ends of the same interval and must not disagree
    about the arithmetic: :func:`wilson_lower_bound` wants the conservative end
    of a *reliability* estimate (low is pessimistic), while
    :func:`compute_decision_quality_metrics` wants the conservative end of an
    *escape-rate* estimate (high is pessimistic). Sharing one implementation
    keeps the two consistent by construction.

    Returns ``(0.0, 1.0)`` for zero trials: with no data the true rate could be
    anything, and neither end may pretend otherwise.
    """
    if trials <= 0:
        return 0.0, 1.0
    successes = max(0, min(successes, trials))
    p_hat = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    centre = p_hat + z2 / (2 * trials)
    margin = z * ((p_hat * (1 - p_hat) / trials + z2 / (4 * trials * trials)) ** 0.5)
    lower = (centre - margin) / denominator
    upper = (centre + margin) / denominator
    return max(0.0, min(1.0, lower)), max(0.0, min(1.0, upper))


def wilson_lower_bound(successes: int, trials: int, *, z: float = 1.96) -> float:
    """Conservative (lower-bound) Wilson score confidence-interval estimate.

    Chosen over a plain success-rate point estimate specifically because it
    accounts for sample size: with 2 trials and 2 successes this returns a
    modest value (well under 1.0), not ``1.0``, exactly the property Part 6
    demands ("do NOT conclude 100% reliable"). ``z=1.96`` is the standard
    95% two-sided z-score; the function is otherwise a direct implementation
    of the closed-form Wilson interval and introduces no external dependency.

    Returns ``0.0`` for zero trials - "no data" must never look reliable.
    """
    if trials <= 0:
        return 0.0
    return wilson_bounds(successes, trials, z=z)[0]


@dataclass
class EvidenceTypeReliability:
    evidence_type: str
    trials: int = 0
    successes: int = 0
    failures: int = 0
    point_estimate: float = 0.0
    lower_bound: float = 0.0
    sufficient_data: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_type": self.evidence_type,
            "trials": self.trials,
            "successes": self.successes,
            "failures": self.failures,
            "point_estimate": round(self.point_estimate, 4),
            "lower_bound": round(self.lower_bound, 4),
            "sufficient_data": self.sufficient_data,
        }


#: Outcomes/qualities that resolve a calibration trial one way or the other.
#: Anything else (pending, unconfirmed) is excluded from trial counts entirely
#: - Property: calibration must be deterministic and must never let an
#: unresolved observation contribute either a success or a failure.
_RESOLVED_SUCCESS_QUALITIES = frozenset({QUALITY_CONSISTENT, QUALITY_TARGETED_CAUGHT_DEFECT})
_RESOLVED_FAILURE_QUALITIES = frozenset({QUALITY_TARGETED_MISSED_DEFECT})


def compute_reliability(
    observations: list[CalibrationObservation], *, min_samples: int
) -> dict[str, EvidenceTypeReliability]:
    """Per-evidence-type reliability from a bounded observation history.

    Order-independent: every observation contributes the same trial/success/
    failure counts regardless of the order they are iterated in, so two callers
    given the same (unordered) observation set always agree - Property 9.
    """
    trials: dict[str, int] = {}
    successes: dict[str, int] = {}
    failures: dict[str, int] = {}
    for obs in observations:
        if obs.decision_quality in _RESOLVED_SUCCESS_QUALITIES:
            delta = 1
        elif obs.decision_quality in _RESOLVED_FAILURE_QUALITIES:
            delta = -1
        else:
            continue  # unresolved: contributes to no evidence type's trial count
        for evidence_type in obs.evidence_types:
            trials[evidence_type] = trials.get(evidence_type, 0) + 1
            if delta > 0:
                successes[evidence_type] = successes.get(evidence_type, 0) + 1
            else:
                failures[evidence_type] = failures.get(evidence_type, 0) + 1

    result: dict[str, EvidenceTypeReliability] = {}
    for evidence_type, trial_count in trials.items():
        success_count = successes.get(evidence_type, 0)
        failure_count = failures.get(evidence_type, 0)
        result[evidence_type] = EvidenceTypeReliability(
            evidence_type=evidence_type,
            trials=trial_count,
            successes=success_count,
            failures=failure_count,
            point_estimate=success_count / trial_count if trial_count else 0.0,
            lower_bound=wilson_lower_bound(success_count, trial_count),
            sufficient_data=trial_count >= max(1, min_samples),
        )
    return result


# -- calibration signal + shadow mode ------------------------------------------


@dataclass
class CalibrationSignal:
    direction: str = "none"  # "up" | "down" | "none"
    calibrated_confidence_score: float = 0.0
    reason: str = ""
    suppressed_by_safety_floor: bool = False


def compute_calibration_signal(
    evidence_types: frozenset[str],
    reliability: dict[str, EvidenceTypeReliability],
    base_confidence_score: float,
    *,
    min_samples: int,
    max_adjustment: float,
    degraded: bool,
) -> CalibrationSignal:
    """Conservative, bidirectional calibration signal for one decision.

    Safety floor (Part 7), enforced structurally rather than by convention:

    * ``degraded=True`` (the impact analysis itself reported reduced
      trustworthiness - see :func:`impact_is_degraded`) unconditionally
      blocks any *upward* adjustment: a dynamic import or unresolved
      dependency cannot become "safe" merely because historical runs passed.
      A downward adjustment is still allowed, since widening is always safe.
    * An upward adjustment additionally requires every evidence type present
      to have ``sufficient_data`` (at least ``min_samples`` resolved trials)
      **and** zero recorded failures - insufficient data can raise confidence
      by exactly zero, regardless of how good the few samples it does have
      look (Part 6's "2 observations, both passed" example).
    * A downward adjustment requires no minimum sample size: even a single
      observed :data:`QUALITY_TARGETED_MISSED_DEFECT` for a relevant evidence
      type is reason enough to distrust it, because that direction can only
      ever widen validation, never narrow it.
    * The magnitude of either adjustment is capped by ``max_adjustment`` and
      the result is always clamped to ``[0, 1]``.
    """
    if not evidence_types:
        return CalibrationSignal(reason="no evidence types to calibrate against")

    applicable = [reliability[t] for t in evidence_types if t in reliability]
    if not applicable:
        return CalibrationSignal(reason="no reliability data for this decision's evidence types")

    any_failures = any(r.failures > 0 for r in applicable)
    if any_failures:
        worst = max(applicable, key=lambda r: (r.failures / r.trials) if r.trials else 0.0)
        failure_rate = worst.failures / worst.trials if worst.trials else 0.0
        drop = min(max_adjustment, failure_rate * max_adjustment)
        calibrated = max(0.0, base_confidence_score - drop)
        return CalibrationSignal(
            direction="down",
            calibrated_confidence_score=calibrated,
            reason=(
                f"evidence type '{worst.evidence_type}' has {worst.failures}/{worst.trials} "
                "recorded targeted-scope escape(s); lowering confidence can only widen scope"
            ),
        )

    all_sufficient = all(r.sufficient_data for r in applicable)
    if not all_sufficient:
        return CalibrationSignal(
            reason="at least one evidence type has fewer than the configured minimum samples; "
            "no upward adjustment without sufficient data"
        )

    weakest = min(applicable, key=lambda r: r.lower_bound)
    if weakest.lower_bound <= base_confidence_score:
        return CalibrationSignal(
            reason="conservative reliability estimate does not exceed the fixed baseline"
        )

    if degraded:
        return CalibrationSignal(
            reason="impact analysis reported degraded/unresolved evidence; "
            "upward calibration is blocked regardless of historical reliability",
            suppressed_by_safety_floor=True,
        )

    calibrated = min(1.0, base_confidence_score + max_adjustment, weakest.lower_bound)
    return CalibrationSignal(
        direction="up",
        calibrated_confidence_score=calibrated,
        reason=(
            f"all {len(applicable)} evidence type(s) have >= {min_samples} resolved trials with "
            f"zero recorded escapes (weakest conservative lower bound {weakest.lower_bound:.3f})"
        ),
    )


class ShadowCalibrationEngine:
    """Computes a hypothetical calibrated decision for comparison only.

    Nothing this class returns is ever applied to
    :class:`~local_agent.validation_decision.ValidationDecision` - it has no
    method that mutates one, and the orchestrator wiring only ever reads its
    output into a telemetry record. This is an architectural invariant, not a
    configuration default: there is currently no code path, gated or
    otherwise, that lets a calibrated confidence change what actually runs.
    """

    def __init__(self, *, min_samples: int, max_adjustment: float):
        self.min_samples = max(1, int(min_samples))
        self.max_adjustment = max(0.0, min(1.0, float(max_adjustment)))

    def evaluate(
        self,
        impact: "ChangeImpactReport",
        decision: "ValidationDecision",
        observations: list[CalibrationObservation],
    ) -> ShadowComparison:
        evidence_types = evidence_types_for_impact(impact)
        reliability = compute_reliability(observations, min_samples=self.min_samples)
        base_score = confidence_score(decision.confidence_level)
        degraded = impact_is_degraded(impact)

        signal = compute_calibration_signal(
            evidence_types,
            reliability,
            base_score,
            min_samples=self.min_samples,
            max_adjustment=self.max_adjustment,
            degraded=degraded,
        )

        if signal.direction == "none":
            return ShadowComparison(
                computed=True,
                shadow_scope=decision.scope,
                shadow_confidence_level=decision.confidence_level,
                reasons=[signal.reason] if signal.reason else [],
                safety_override=signal.suppressed_by_safety_floor,
            )

        shadow_level = _level_for_score(signal.calibrated_confidence_score)
        shadow_scope = _scope_for_level(shadow_level)
        base_scope = _scope_for_level(decision.confidence_level)
        from .semantic_impact import SCOPE_ORDER

        would_narrow = SCOPE_ORDER.index(shadow_scope) < SCOPE_ORDER.index(base_scope)
        would_broaden = SCOPE_ORDER.index(shadow_scope) > SCOPE_ORDER.index(base_scope)
        return ShadowComparison(
            computed=True,
            would_narrow=would_narrow,
            would_broaden=would_broaden,
            confidence_delta=round(signal.calibrated_confidence_score - base_score, 4),
            shadow_scope=shadow_scope,
            shadow_confidence_level=shadow_level,
            reasons=[signal.reason],
            safety_override=signal.suppressed_by_safety_floor,
        )


# -- bounded store --------------------------------------------------------------


class ValidationTelemetryStore:
    """Bounded, serialisable history of decisions and calibration observations.

    Mirrors :class:`local_agent.evidence.EvidenceLedger`'s shape (bounded
    entry list, tolerant ``to_dict``/``from_dict``) for the same reasons: this
    needs to be cheap to persist, safe to load from a payload written by an
    older or newer build, and safe to reason about in isolation from any
    particular orchestrator run.
    """

    def __init__(
        self,
        *,
        max_decisions: int = DEFAULT_MAX_DECISIONS,
        max_observations: int = DEFAULT_MAX_OBSERVATIONS,
    ):
        self.max_decisions = max(1, int(max_decisions))
        self.max_observations = max(1, int(max_observations))
        self._decisions: list[ValidationDecisionRecord] = []
        self._observations: list[CalibrationObservation] = []
        #: Records that failed to deserialise cleanly and were dropped, rather
        #: than allowed to raise or to silently contribute corrupt data to any
        #: analysis. Exposed for :class:`ValidationIntelligenceHealth`.
        self.corrupted_records_skipped = 0

    @property
    def decisions(self) -> list[ValidationDecisionRecord]:
        return list(self._decisions)

    @property
    def observations(self) -> list[CalibrationObservation]:
        return list(self._observations)

    def __len__(self) -> int:
        return len(self._decisions)

    def record_decision(self, record: ValidationDecisionRecord) -> ValidationDecisionRecord:
        self._decisions.append(record)
        if len(self._decisions) > self.max_decisions:
            del self._decisions[: len(self._decisions) - self.max_decisions]
        return record

    def find_decision(self, decision_id: str) -> ValidationDecisionRecord | None:
        for record in reversed(self._decisions):
            if record.decision_id == decision_id:
                return record
        return None

    def finalize_decision(
        self,
        decision_id: str,
        *,
        outcome: str,
        decision_quality: str,
        broad_duration_seconds: float = 0.0,
        targeted_duration_seconds: float | None = None,
    ) -> CalibrationObservation | None:
        """Fill in the outcome for a previously-recorded decision and derive
        its :class:`CalibrationObservation`. Idempotent: calling this again
        for the same ``decision_id`` overwrites the outcome fields and appends
        a new observation rather than raising - a caller that retries after a
        crash cannot corrupt the store, only add one extra (still bounded,
        still individually correct) observation.

        Returns ``None`` (and records nothing) when ``decision_id`` is not
        found - a missing record must never be silently treated as success.
        """
        record = self.find_decision(decision_id)
        if record is None:
            return None
        record.outcome = outcome
        record.decision_quality = decision_quality
        record.broad_duration_seconds = float(broad_duration_seconds or 0.0)
        if targeted_duration_seconds is not None:
            record.targeted_duration_seconds = float(targeted_duration_seconds)
        observation = CalibrationObservation.from_record(record)
        self._observations.append(observation)
        if len(self._observations) > self.max_observations:
            del self._observations[: len(self._observations) - self.max_observations]
        return observation

    # -- analytics -----------------------------------------------------------

    def reuse_reason_totals(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for record in self._decisions:
            for reason, count in record.reuse_reasons.items():
                totals[reason] = totals.get(reason, 0) + count
        return totals

    def reliability(self, *, min_samples: int) -> dict[str, EvidenceTypeReliability]:
        return compute_reliability(self._observations, min_samples=min_samples)

    # -- serialisation ---------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_decisions": self.max_decisions,
            "max_observations": self.max_observations,
            "corrupted_records_skipped": self.corrupted_records_skipped,
            "decisions": [record.to_dict() for record in self._decisions],
            "observations": [obs.to_dict() for obs in self._observations],
        }

    @classmethod
    def from_dict(cls, data: Any) -> "ValidationTelemetryStore":
        if not isinstance(data, dict):
            return cls()
        store = cls(
            max_decisions=int(data.get("max_decisions", DEFAULT_MAX_DECISIONS) or DEFAULT_MAX_DECISIONS),
            max_observations=int(
                data.get("max_observations", DEFAULT_MAX_OBSERVATIONS) or DEFAULT_MAX_OBSERVATIONS
            ),
        )
        corrupted = 0
        decisions: list[ValidationDecisionRecord] = []
        for item in (data.get("decisions") or []):
            if not isinstance(item, dict):
                corrupted += 1
                continue
            decisions.append(ValidationDecisionRecord.from_dict(item))
        observations: list[CalibrationObservation] = []
        for item in (data.get("observations") or []):
            if not isinstance(item, dict):
                corrupted += 1
                continue
            observations.append(CalibrationObservation.from_dict(item))
        store._decisions = decisions[-store.max_decisions:]
        store._observations = observations[-store.max_observations:]
        store.corrupted_records_skipped = int(
            data.get("corrupted_records_skipped", 0) or 0
        ) + corrupted
        return store


# -- cost model (Part 11) --------------------------------------------------------


def _mean(values: list[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


@dataclass
class ValidationCostModel:
    """Measured validation cost, never a modelled or extrapolated one.

    Every number here is an aggregate over durations that were *actually
    recorded* by a finalized :class:`ValidationDecisionRecord`. A record whose
    duration is ``0.0`` is treated as **unmeasured and excluded**, not as a
    zero-cost run: averaging in a placeholder zero would silently invent a
    cheaper-looking cost profile, which is precisely the kind of fabricated
    metric Part 11 forbids ("just create reliable measurements"). The
    ``*_samples`` counts therefore always accompany the corresponding
    averages, so a caller can see how much data a number rests on.

    This is deliberately *only* measurement. Nothing in this module consumes a
    cost model to make, weaken, or narrow a validation decision - Part 11's
    "do NOT let this phase automatically trade safety for cost".
    """

    decisions: int = 0
    targeted_samples: int = 0
    broad_samples: int = 0
    mean_targeted_seconds: float = 0.0
    median_targeted_seconds: float = 0.0
    mean_broad_seconds: float = 0.0
    median_broad_seconds: float = 0.0
    #: Mean ``broad / targeted`` cost ratio, computed only over decisions where
    #: *both* durations were measured on the same run (a cross-record ratio of
    #: two independent means would not describe any real run).
    paired_samples: int = 0
    mean_broad_to_targeted_ratio: float = 0.0
    #: Candidate-stage command runtime avoided by evidence reuse, summed from
    #: what :class:`~local_agent.evidence.EvidenceLedger` actually measured when
    #: the command originally ran.
    total_reuse_time_saved_seconds: float = 0.0
    mean_reuse_time_saved_per_decision_seconds: float = 0.0
    mean_commands_selected: float = 0.0
    mean_commands_reused: float = 0.0
    #: False when there is nothing measured at all; a consumer must not present
    #: any of the averages above as a cost estimate in that case.
    measured: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "decisions": self.decisions,
            "targeted_samples": self.targeted_samples,
            "broad_samples": self.broad_samples,
            "mean_targeted_seconds": round(self.mean_targeted_seconds, 4),
            "median_targeted_seconds": round(self.median_targeted_seconds, 4),
            "mean_broad_seconds": round(self.mean_broad_seconds, 4),
            "median_broad_seconds": round(self.median_broad_seconds, 4),
            "paired_samples": self.paired_samples,
            "mean_broad_to_targeted_ratio": round(self.mean_broad_to_targeted_ratio, 4),
            "total_reuse_time_saved_seconds": round(self.total_reuse_time_saved_seconds, 4),
            "mean_reuse_time_saved_per_decision_seconds": round(
                self.mean_reuse_time_saved_per_decision_seconds, 4
            ),
            "mean_commands_selected": round(self.mean_commands_selected, 4),
            "mean_commands_reused": round(self.mean_commands_reused, 4),
            "measured": self.measured,
        }


def compute_cost_model(store: "ValidationTelemetryStore") -> ValidationCostModel:
    """Aggregate measured validation cost over a bounded decision history."""
    decisions = store.decisions
    if not decisions:
        return ValidationCostModel()

    targeted = [r.targeted_duration_seconds for r in decisions if r.targeted_duration_seconds > 0]
    broad = [r.broad_duration_seconds for r in decisions if r.broad_duration_seconds > 0]
    ratios = [
        r.broad_duration_seconds / r.targeted_duration_seconds
        for r in decisions
        if r.targeted_duration_seconds > 0 and r.broad_duration_seconds > 0
    ]
    saved = [r.time_saved_seconds for r in decisions]

    return ValidationCostModel(
        decisions=len(decisions),
        targeted_samples=len(targeted),
        broad_samples=len(broad),
        mean_targeted_seconds=_mean(targeted),
        median_targeted_seconds=_median(targeted),
        mean_broad_seconds=_mean(broad),
        median_broad_seconds=_median(broad),
        paired_samples=len(ratios),
        mean_broad_to_targeted_ratio=_mean(ratios),
        total_reuse_time_saved_seconds=sum(saved),
        mean_reuse_time_saved_per_decision_seconds=_mean(saved),
        mean_commands_selected=_mean([float(r.selected_command_count) for r in decisions]),
        mean_commands_reused=_mean([float(r.reused_command_count) for r in decisions]),
        measured=bool(targeted or broad),
    )


# -- decision-quality metrics (Part 4) -------------------------------------------


@dataclass
class ConfidenceBucket:
    """Observed behaviour of one predicted-confidence level."""

    predicted_confidence: str = ""
    resolved_trials: int = 0
    escapes: int = 0
    agreement_rate: float = 0.0
    agreement_lower_bound: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_confidence": self.predicted_confidence,
            "resolved_trials": self.resolved_trials,
            "escapes": self.escapes,
            "agreement_rate": round(self.agreement_rate, 4),
            "agreement_lower_bound": round(self.agreement_lower_bound, 4),
        }


@dataclass
class DecisionQualityMetrics:
    """Part 4's metrics, with deliberately conservative naming.

    **Terminology, and what this data genuinely cannot support.** Part 4 asks
    for "targeted recall". Recall requires knowing the total number of defects
    a change actually introduced, including those *no* validation scope ran by
    this agent ever detected. Nothing observes that, so no recall figure is
    reported and :attr:`recall_available` is permanently ``False``. What *is*
    observable is: of the targeted-scope decisions where a broader run also
    executed and could therefore contradict the narrow one, how often it did.
    That is an **observed escape rate**, and it is a *lower* bound on the true
    escape rate (a defect neither the targeted nor the broad run caught is
    invisible to it), which is why
    :attr:`observed_escape_rate_upper_bound` - the pessimistic end of the
    Wilson interval, not the point estimate - is what any safety argument
    should quote.

    Likewise :attr:`targeted_agreement_rate` is called *agreement*, not
    *precision*: it measures how often the narrow decision was not contradicted
    by the broader run, which is a weaker claim than the narrow scope having
    been sufficient.
    """

    resolved_observations: int = 0
    #: Targeted-scope decisions that reached a state where a broader run either
    #: confirmed or contradicted them - the only denominator that supports an
    #: escape-rate claim at all.
    targeted_resolved_trials: int = 0
    targeted_escapes: int = 0
    targeted_caught_defects: int = 0
    observed_escape_rate: float = 0.0
    observed_escape_rate_upper_bound: float = 1.0
    targeted_agreement_rate: float = 0.0
    broad_validation_rate: float = 0.0
    broad_not_proven_necessary: int = 0
    reuse_attempts: int = 0
    reuse_grants: int = 0
    reuse_denials: int = 0
    reuse_hit_rate: float = 0.0
    reuse_rejection_rate: float = 0.0
    reuse_rejection_reasons: dict[str, int] = field(default_factory=dict)
    #: Denials attributed specifically to evidence age - the only freshness
    #: signal available, since raw evidence timestamps are not retained here.
    stale_evidence_rejections: int = 0
    confidence_buckets: dict[str, ConfidenceBucket] = field(default_factory=dict)
    #: Permanently ``False``; see the class docstring.
    recall_available: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "resolved_observations": self.resolved_observations,
            "targeted_resolved_trials": self.targeted_resolved_trials,
            "targeted_escapes": self.targeted_escapes,
            "targeted_caught_defects": self.targeted_caught_defects,
            "observed_escape_rate": round(self.observed_escape_rate, 4),
            "observed_escape_rate_upper_bound": round(self.observed_escape_rate_upper_bound, 4),
            "targeted_agreement_rate": round(self.targeted_agreement_rate, 4),
            "broad_validation_rate": round(self.broad_validation_rate, 4),
            "broad_not_proven_necessary": self.broad_not_proven_necessary,
            "reuse_attempts": self.reuse_attempts,
            "reuse_grants": self.reuse_grants,
            "reuse_denials": self.reuse_denials,
            "reuse_hit_rate": round(self.reuse_hit_rate, 4),
            "reuse_rejection_rate": round(self.reuse_rejection_rate, 4),
            "reuse_rejection_reasons": dict(self.reuse_rejection_reasons),
            "stale_evidence_rejections": self.stale_evidence_rejections,
            "confidence_buckets": {k: v.to_dict() for k, v in self.confidence_buckets.items()},
            "recall_available": self.recall_available,
        }


def compute_decision_quality_metrics(
    store: "ValidationTelemetryStore",
) -> DecisionQualityMetrics:
    """Derive Part 4's metrics from a bounded store. Read-only and pure."""
    decisions = store.decisions
    observations = store.observations

    resolved = [
        o
        for o in observations
        if o.decision_quality in _RESOLVED_SUCCESS_QUALITIES
        or o.decision_quality in _RESOLVED_FAILURE_QUALITIES
    ]
    targeted_resolved = [o for o in resolved if o.selected_scope == "targeted"]
    escapes = sum(
        1 for o in targeted_resolved if o.decision_quality == QUALITY_TARGETED_MISSED_DEFECT
    )
    caught = sum(
        1 for o in targeted_resolved if o.decision_quality == QUALITY_TARGETED_CAUGHT_DEFECT
    )
    trials = len(targeted_resolved)
    escape_rate = (escapes / trials) if trials else 0.0
    # The pessimistic end of the interval on the escape rate. With zero trials
    # this is 1.0: no data must never look like a low escape rate.
    _, escape_upper = wilson_bounds(escapes, trials)

    buckets: dict[str, ConfidenceBucket] = {}
    for level in _CONFIDENCE_ORDER:
        in_bucket = [o for o in resolved if o.predicted_confidence == level]
        if not in_bucket:
            continue
        bucket_escapes = sum(
            1 for o in in_bucket if o.decision_quality == QUALITY_TARGETED_MISSED_DEFECT
        )
        agreements = len(in_bucket) - bucket_escapes
        buckets[level] = ConfidenceBucket(
            predicted_confidence=level,
            resolved_trials=len(in_bucket),
            escapes=bucket_escapes,
            agreement_rate=agreements / len(in_bucket),
            agreement_lower_bound=wilson_lower_bound(agreements, len(in_bucket)),
        )

    reason_totals = store.reuse_reason_totals()
    grants = reason_totals.get(REASON_OK, 0)
    attempts = sum(reason_totals.values())
    denials = attempts - grants
    rejection_reasons = {k: v for k, v in reason_totals.items() if k != REASON_OK}

    total_decisions = len(decisions)
    broad_count = sum(1 for r in decisions if r.scope == "broad")

    return DecisionQualityMetrics(
        resolved_observations=len(resolved),
        targeted_resolved_trials=trials,
        targeted_escapes=escapes,
        targeted_caught_defects=caught,
        observed_escape_rate=escape_rate,
        observed_escape_rate_upper_bound=escape_upper,
        targeted_agreement_rate=((trials - escapes) / trials) if trials else 0.0,
        broad_validation_rate=(broad_count / total_decisions) if total_decisions else 0.0,
        broad_not_proven_necessary=sum(
            1 for o in observations if o.decision_quality == QUALITY_BROAD_NOT_PROVEN_NECESSARY
        ),
        reuse_attempts=attempts,
        reuse_grants=grants,
        reuse_denials=denials,
        reuse_hit_rate=(grants / attempts) if attempts else 0.0,
        reuse_rejection_rate=(denials / attempts) if attempts else 0.0,
        reuse_rejection_reasons=rejection_reasons,
        stale_evidence_rejections=reason_totals.get(REASON_STALE, 0),
        confidence_buckets=buckets,
    )


# -- health report --------------------------------------------------------------


@dataclass
class ValidationIntelligenceHealth:
    total_decisions: int = 0
    total_observations: int = 0
    resolved_observations: int = 0
    scope_counts: dict[str, int] = field(default_factory=dict)
    broad_validation_rate: float = 0.0
    reuse_hit_rate: float = 0.0
    reuse_rejection_reasons: dict[str, int] = field(default_factory=dict)
    evidence_type_reliability: dict[str, EvidenceTypeReliability] = field(default_factory=dict)
    false_confidence_incidents: int = 0
    unresolved_dependency_rate: float = 0.0
    dynamic_import_rate: float = 0.0
    corrupted_records_skipped: int = 0
    #: Fraction of decisions whose impact analysis was itself degraded
    #: (:func:`impact_is_degraded`). Part 15's "analyzer failure rate": a high
    #: value means the *analyzer*, not the calibration, is the weak link.
    analysis_degradation_rate: float = 0.0
    #: Corrupted entries as a fraction of everything that was offered to the
    #: store (kept entries plus skipped ones), so the number stays meaningful
    #: as history accumulates instead of only ever growing.
    evidence_corruption_rate: float = 0.0
    #: Shadow-mode aggregates. ``calibration_drift`` is the mean *absolute*
    #: confidence delta over shadow comparisons that were actually computed -
    #: how far calibration wants to move from the fixed table, in either
    #: direction. Zero when nothing has been computed, which is the default
    #: state since ``validation_calibration_enabled`` defaults to ``False``.
    shadow_comparisons: int = 0
    shadow_would_narrow: int = 0
    shadow_would_broaden: int = 0
    shadow_safety_overrides: int = 0
    calibration_drift: float = 0.0
    cost: ValidationCostModel = field(default_factory=ValidationCostModel)
    quality: DecisionQualityMetrics = field(default_factory=DecisionQualityMetrics)
    calibration_status: str = "insufficient_data"

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_decisions": self.total_decisions,
            "total_observations": self.total_observations,
            "resolved_observations": self.resolved_observations,
            "scope_counts": dict(self.scope_counts),
            "broad_validation_rate": round(self.broad_validation_rate, 4),
            "reuse_hit_rate": round(self.reuse_hit_rate, 4),
            "reuse_rejection_reasons": dict(self.reuse_rejection_reasons),
            "evidence_type_reliability": {
                k: v.to_dict() for k, v in self.evidence_type_reliability.items()
            },
            "false_confidence_incidents": self.false_confidence_incidents,
            "unresolved_dependency_rate": round(self.unresolved_dependency_rate, 4),
            "dynamic_import_rate": round(self.dynamic_import_rate, 4),
            "corrupted_records_skipped": self.corrupted_records_skipped,
            "analysis_degradation_rate": round(self.analysis_degradation_rate, 4),
            "evidence_corruption_rate": round(self.evidence_corruption_rate, 4),
            "shadow_comparisons": self.shadow_comparisons,
            "shadow_would_narrow": self.shadow_would_narrow,
            "shadow_would_broaden": self.shadow_would_broaden,
            "shadow_safety_overrides": self.shadow_safety_overrides,
            "calibration_drift": round(self.calibration_drift, 4),
            "cost": self.cost.to_dict(),
            "quality": self.quality.to_dict(),
            "calibration_status": self.calibration_status,
        }


def compute_health(
    store: ValidationTelemetryStore, *, min_samples: int
) -> ValidationIntelligenceHealth:
    """Diagnostic-only summary. Never consulted to change policy automatically
    - see module docstring and Part 15 ("do not let it automatically change
    policy unless explicitly enabled and safe"; no such wiring exists here).
    """
    decisions = store.decisions
    observations = store.observations
    total = len(decisions)

    scope_counts: dict[str, int] = {}
    for record in decisions:
        scope_counts[record.scope] = scope_counts.get(record.scope, 0) + 1
    broad_rate = (scope_counts.get("broad", 0) / total) if total else 0.0

    total_selected = sum(r.selected_command_count + r.reused_command_count for r in decisions)
    total_reused = sum(r.reused_command_count for r in decisions)
    reuse_hit_rate = (total_reused / total_selected) if total_selected else 0.0

    unresolved_hits = sum(
        1 for r in decisions if any("unresolved" in e for e in r.evidence_types)
    )
    dynamic_hits = sum(
        1 for r in decisions if any("dynamic_import" in e for e in r.evidence_types)
    )
    degraded_hits = sum(1 for r in decisions if r.degraded_analysis)
    shadow_records = [r for r in decisions if r.shadow.computed]
    # Everything the store was ever offered: what it kept plus what it refused
    # to deserialise. Using this as the denominator keeps the corruption *rate*
    # honest even after eviction has trimmed the kept entries.
    offered = total + len(observations) + store.corrupted_records_skipped

    resolved = [o for o in observations if o.outcome != OUTCOME_PENDING]
    false_confidence = sum(
        1 for o in observations if o.decision_quality == QUALITY_TARGETED_MISSED_DEFECT
    )
    reliability = compute_reliability(observations, min_samples=min_samples)
    sufficient = [r for r in reliability.values() if r.sufficient_data]
    if not observations:
        status = "no_observations"
    elif not sufficient:
        status = "insufficient_data"
    else:
        status = "shadow_only"

    return ValidationIntelligenceHealth(
        total_decisions=total,
        total_observations=len(observations),
        resolved_observations=len(resolved),
        scope_counts=scope_counts,
        broad_validation_rate=broad_rate,
        reuse_hit_rate=reuse_hit_rate,
        reuse_rejection_reasons=store.reuse_reason_totals(),
        evidence_type_reliability=reliability,
        false_confidence_incidents=false_confidence,
        unresolved_dependency_rate=(unresolved_hits / total) if total else 0.0,
        dynamic_import_rate=(dynamic_hits / total) if total else 0.0,
        corrupted_records_skipped=store.corrupted_records_skipped,
        analysis_degradation_rate=(degraded_hits / total) if total else 0.0,
        evidence_corruption_rate=(
            store.corrupted_records_skipped / offered if offered else 0.0
        ),
        shadow_comparisons=len(shadow_records),
        shadow_would_narrow=sum(1 for r in shadow_records if r.shadow.would_narrow),
        shadow_would_broaden=sum(1 for r in shadow_records if r.shadow.would_broaden),
        shadow_safety_overrides=sum(1 for r in shadow_records if r.shadow.safety_override),
        calibration_drift=(
            sum(abs(r.shadow.confidence_delta) for r in shadow_records) / len(shadow_records)
            if shadow_records
            else 0.0
        ),
        cost=compute_cost_model(store),
        quality=compute_decision_quality_metrics(store),
        calibration_status=status,
    )


# -- persistence manager ---------------------------------------------------------

_STORE_LOCKS: dict[str, threading.Lock] = {}
_STORE_LOCKS_GUARD = threading.Lock()


def _lock_for(key: str) -> threading.Lock:
    """Process-wide lock keyed by repository identity.

    Not global *mutable state* in the sense this phase's instructions warn
    against - the dict holds only synchronisation primitives, never business
    data - but it does mean two :class:`ValidationTelemetryManager` instances
    pointed at the same project root, in the same process (exactly the
    parallel-worktree-orchestrator situation), serialise their read-modify-
    write cycles on the underlying store instead of racing. Cross-*process*
    concurrency is out of scope, same as the rest of :mod:`local_agent.storage`
    (no file already has cross-process locking).
    """
    with _STORE_LOCKS_GUARD:
        lock = _STORE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _STORE_LOCKS[key] = lock
        return lock


class ValidationTelemetryManager:
    """Owns read-modify-write access to one repository's telemetry store.

    Every public method reloads from storage under a per-repository lock
    before mutating and saves before releasing it, rather than trusting an
    in-memory cache across calls. That costs an extra load+save per decision
    (measured: negligible next to a validation command run) in exchange for
    never losing a concurrent writer's update - see :func:`_lock_for`.
    """

    def __init__(self, storage: Any, project_root: str | Path, *, max_decisions: int = DEFAULT_MAX_DECISIONS, max_observations: int = DEFAULT_MAX_OBSERVATIONS):
        self.storage = storage
        self.project_root = Path(project_root)
        self.max_decisions = max_decisions
        self.max_observations = max_observations
        self._lock = _lock_for(str(self.project_root.resolve()))

    def _load(self) -> ValidationTelemetryStore:
        if not hasattr(self.storage, "load_validation_telemetry"):
            return ValidationTelemetryStore(
                max_decisions=self.max_decisions, max_observations=self.max_observations
            )
        store = self.storage.load_validation_telemetry()
        store.max_decisions = self.max_decisions
        store.max_observations = self.max_observations
        return store

    def _save(self, store: ValidationTelemetryStore) -> None:
        if hasattr(self.storage, "save_validation_telemetry"):
            self.storage.save_validation_telemetry(store)

    def record_decision(self, record: ValidationDecisionRecord) -> ValidationDecisionRecord:
        with self._lock:
            store = self._load()
            store.record_decision(record)
            self._save(store)
        return record

    def finalize_decision(
        self,
        decision_id: str,
        *,
        outcome: str,
        decision_quality: str,
        broad_duration_seconds: float = 0.0,
        targeted_duration_seconds: float | None = None,
    ) -> CalibrationObservation | None:
        if not decision_id:
            return None
        with self._lock:
            store = self._load()
            observation = store.finalize_decision(
                decision_id,
                outcome=outcome,
                decision_quality=decision_quality,
                broad_duration_seconds=broad_duration_seconds,
                targeted_duration_seconds=targeted_duration_seconds,
            )
            if observation is not None:
                self._save(store)
        return observation

    def health(self, *, min_samples: int) -> ValidationIntelligenceHealth:
        with self._lock:
            store = self._load()
        return compute_health(store, min_samples=min_samples)

    def observations(self) -> list[CalibrationObservation]:
        with self._lock:
            store = self._load()
        return store.observations
