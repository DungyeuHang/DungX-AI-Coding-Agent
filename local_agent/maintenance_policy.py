"""Phase 4.21: maintenance prioritisation and the autonomy policy.

Two decisions live here, and keeping them apart matters:

* **How important is this?** :class:`MaintenancePriorityEngine` produces a
  deterministic, explainable ordering. It answers "what should we look at
  first".
* **How much are we allowed to do about it?** :class:`MaintenanceExecutionPolicy`
  produces an autonomy tier. It answers "what may the agent do without a human".

Conflating them would be the classic autonomy bug: the most important problem
becoming, by that very fact, the one the machine is most willing to act on
alone. Here the relationship is the opposite. A CRITICAL false-confidence
incident sorts to the very top of the queue *and* is capped at RECOMMEND,
because the correct response to "our safety analysis was wrong" is a human
reading it, not the same analysis trying again.

The policy is a *ceiling*, never a floor. It can only ever lower the tier the
operator configured; there is no code path by which a candidate is granted more
autonomy than the configuration allows. It also cannot reach the systems it
must not override: this module does not import
:mod:`local_agent.validation_decision`, :mod:`local_agent.approval` or
:mod:`local_agent.tool_engine`, and the test-suite proves that structurally
from the AST rather than trusting this sentence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .maintenance import (
    PROTECTED_RELATIVE_PATHS,
    MaintenanceBudget,
    MaintenanceCandidate,
    MaintenanceSignal,
    SEVERITY_CRITICAL,
    SEVERITY_ORDER,
    is_protected_directory_segment,
    is_protected_relative_path,
    sanitize_relative_path,
    sanitize_text,
    severity_rank,
)

# -- autonomy tiers -----------------------------------------------------------


class AutonomyTier:
    """What the maintenance layer may do with a candidate."""

    OBSERVE_ONLY = "observe_only"
    RECOMMEND = "recommend"
    PLAN_ONLY = "plan_only"
    EXECUTE_WITH_EXISTING_APPROVAL = "execute_with_existing_approval"
    EXECUTE_AUTONOMOUSLY = "execute_autonomously"


#: Ordered weakest to strongest. Every comparison in this module is an index
#: comparison against this tuple, so "stronger tier" has exactly one meaning.
TIER_ORDER: tuple[str, ...] = (
    AutonomyTier.OBSERVE_ONLY,
    AutonomyTier.RECOMMEND,
    AutonomyTier.PLAN_ONLY,
    AutonomyTier.EXECUTE_WITH_EXISTING_APPROVAL,
    AutonomyTier.EXECUTE_AUTONOMOUSLY,
)

#: Tiers at which real repository modification may be attempted. Both still go
#: through the existing CandidateWorkspace -> validation -> approval -> apply
#: pipeline; the difference between them is only whether the operator is asked
#: interactively, which is decided by the *existing* approval machinery, not
#: here.
EXECUTING_TIERS: frozenset[str] = frozenset(
    {AutonomyTier.EXECUTE_WITH_EXISTING_APPROVAL, AutonomyTier.EXECUTE_AUTONOMOUSLY}
)


def tier_rank(tier: Any) -> int:
    """Rank of ``tier``; an unrecognised value ranks *lowest* (safest)."""
    try:
        return TIER_ORDER.index(str(tier))
    except ValueError:
        return 0


def weakest_tier(*tiers: Any) -> str:
    """The least-permissive of the arguments.

    This is the only combinator the policy uses. Everything that wants to
    influence the outcome can lower it; nothing can raise it. That asymmetry is
    the whole safety argument for the module.
    """
    best = AutonomyTier.EXECUTE_AUTONOMOUSLY
    for tier in tiers:
        if tier_rank(tier) < tier_rank(best):
            best = TIER_ORDER[tier_rank(tier)]
    return best


#: Signal kinds for which a code change is a plausible fix at all.
#:
#: Everything outside this set is capped at RECOMMEND regardless of severity or
#: confidence. The excluded kinds are excluded for a reason, not an oversight:
#: broad-validation pressure and analysis degradation are weaknesses in the
#: agent's *own* analysis, false confidence demands human review by definition,
#: architectural risk is a design judgement, and abandoned-work rate is a
#: process observation with no localisable fix.
AUTONOMOUSLY_ACTIONABLE_KINDS: frozenset[str] = frozenset(
    {
        MaintenanceSignal.PARSE_FAILURE,
        MaintenanceSignal.TEST_GAP,
        MaintenanceSignal.RECURRING_DEFECT,
        MaintenanceSignal.KNOWN_FAILURE_PATTERN,
        MaintenanceSignal.REPEATED_REPAIR,
    }
)


# -- prioritisation -----------------------------------------------------------

#: Component weights. They sum to 1.0 so a raw score is directly readable as a
#: fraction, and they are module constants rather than inline literals so that
#: an operator can audit the entire ranking policy by reading twelve lines.
WEIGHT_SEVERITY = 0.30
WEIGHT_CONFIDENCE = 0.20
WEIGHT_RECURRENCE = 0.15
WEIGHT_BLAST_RADIUS = 0.10
WEIGHT_FRESHNESS = 0.05
WEIGHT_EFFORT = 0.10
WEIGHT_HISTORY = 0.10

#: Penalty applied for explicit uncertainty. Subtracted after the weighted sum,
#: so it can pull a score down but never push one up.
WEIGHT_UNCERTAINTY_PENALTY = 0.15

#: Saturation points: the value at which a component is considered maximal.
#: Without these, one candidate touching 400 files would compress every other
#: candidate's blast-radius component to nearly zero.
RECURRENCE_SATURATION = 5
BLAST_RADIUS_SATURATION = 8
EFFORT_SATURATION = 8.0
UNCERTAINTY_SATURATION = 3

_ALL_WEIGHTS = (
    WEIGHT_SEVERITY,
    WEIGHT_CONFIDENCE,
    WEIGHT_RECURRENCE,
    WEIGHT_BLAST_RADIUS,
    WEIGHT_FRESHNESS,
    WEIGHT_EFFORT,
    WEIGHT_HISTORY,
)


@dataclass(frozen=True)
class PriorityExplanation:
    """Why a candidate got the rank it got, in operator-readable form."""

    candidate_id: str
    severity: str
    severity_rank: int
    score: float
    components: dict[str, float]
    weights: dict[str, float]
    uncertainty_penalty: float
    reasons: list[str]

    @property
    def sort_key(self) -> tuple[int, float, str]:
        """Deterministic ordering key, strongest first when reverse-sorted.

        Severity comes first and is *not* traded off against score: this is the
        structural expression of "safety dominates optimization". A cheap,
        certain, low-severity candidate cannot overtake a dangerous one no
        matter how favourable every other component is. Ties break on the
        candidate id, which is a content hash, so ordering is stable across
        processes and machines.
        """
        return (self.severity_rank, self.score, self.candidate_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "severity": self.severity,
            "severity_rank": self.severity_rank,
            "score": self.score,
            "components": dict(sorted(self.components.items())),
            "weights": dict(sorted(self.weights.items())),
            "uncertainty_penalty": self.uncertainty_penalty,
            "reasons": list(self.reasons),
        }


class MaintenancePriorityEngine:
    """Deterministic, bounded, explainable candidate ranking.

    Every component is normalised into ``[0, 1]`` before weighting, so no
    single input can dominate by being measured on a larger scale, and the
    final score is clamped into ``[0, 1]`` so it is comparable across runs.

    The engine is stateless and reads nothing but the candidate, which is what
    makes ordering reproducible: the same candidate set ranks identically in
    any process, on any machine, in any order of presentation.
    """

    def __init__(self, *, now: str | None = None):
        #: Optional "current time" for freshness scoring. Injected rather than
        #: read from the clock so that ranking is reproducible in tests and in
        #: a resumed run.
        self.now = now

    def explain(self, candidate: MaintenanceCandidate) -> PriorityExplanation:
        components = {
            "severity": severity_rank(candidate.severity) / float(len(SEVERITY_ORDER) - 1),
            "confidence": candidate.confidence,
            "recurrence": min(1.0, candidate.occurrence_count / float(RECURRENCE_SATURATION)),
            "blast_radius": min(
                1.0, len(candidate.affected_files) / float(BLAST_RADIUS_SATURATION)
            ),
            "freshness": _freshness(candidate, self.now),
            # Cheaper work scores higher, but only through a 0.10 weight: this
            # is an efficiency tie-breaker, never a reason to skip something
            # more serious.
            "effort": 1.0 - min(1.0, candidate.estimated_effort / EFFORT_SATURATION),
            # Repeatedly-failing candidates sink. Otherwise a candidate that
            # cannot be fixed would monopolise every future run.
            "history": 1.0 - candidate.failure_ratio,
        }
        weights = {
            "severity": WEIGHT_SEVERITY,
            "confidence": WEIGHT_CONFIDENCE,
            "recurrence": WEIGHT_RECURRENCE,
            "blast_radius": WEIGHT_BLAST_RADIUS,
            "freshness": WEIGHT_FRESHNESS,
            "effort": WEIGHT_EFFORT,
            "history": WEIGHT_HISTORY,
        }
        raw = sum(components[name] * weights[name] for name in components)
        penalty = WEIGHT_UNCERTAINTY_PENALTY * min(
            1.0, len(candidate.uncertainty) / float(UNCERTAINTY_SATURATION)
        )
        score = max(0.0, min(1.0, raw - penalty))

        reasons: list[str] = [
            f"severity {candidate.severity} (band {severity_rank(candidate.severity)} "
            f"of {len(SEVERITY_ORDER) - 1}) dominates ordering",
            f"confidence {candidate.confidence:.2f} from {candidate.sample_size} sample(s)",
            f"observed {candidate.occurrence_count} time(s)",
            f"{len(candidate.affected_files)} affected file(s)",
            f"estimated effort {candidate.estimated_effort:.1f}",
        ]
        if candidate.uncertainty:
            reasons.append(
                f"uncertainty penalty {penalty:.3f} for "
                f"{len(candidate.uncertainty)} caveat(s)"
            )
        if candidate.failure_count:
            reasons.append(
                f"downweighted: {candidate.failure_count} of {candidate.attempt_count} "
                "previous attempt(s) failed"
            )
        return PriorityExplanation(
            candidate_id=candidate.candidate_id,
            severity=candidate.severity,
            severity_rank=severity_rank(candidate.severity),
            score=score,
            components=components,
            weights=weights,
            uncertainty_penalty=penalty,
            reasons=reasons,
        )

    def rank(
        self, candidates: Sequence[MaintenanceCandidate]
    ) -> list[tuple[MaintenanceCandidate, PriorityExplanation]]:
        """Rank ``candidates`` most-important first, deterministically."""
        scored = [(candidate, self.explain(candidate)) for candidate in candidates]
        scored.sort(key=lambda pair: pair[1].sort_key, reverse=True)
        return scored


def _freshness(candidate: MaintenanceCandidate, now: str | None) -> float:
    """1.0 when last seen in the newest slice of the candidate's own history.

    Deliberately coarse: a lexicographic comparison of ISO-8601 timestamps,
    which is correct for UTC strings and needs no parsing, no timezone
    handling and no clock. A candidate whose ``last_seen_at`` is at or after
    the reference time is fresh; otherwise it decays to a floor of 0.25 rather
    than to zero, because an old signal that keeps being re-detected is not
    uninteresting - it is chronic.
    """
    if not now:
        return 1.0
    if candidate.last_seen_at >= now:
        return 1.0
    if candidate.first_seen_at >= now:
        return 0.75
    return 0.25


# -- execution policy ---------------------------------------------------------


@dataclass(frozen=True)
class PolicyThresholds:
    """Gates on autonomy. All named, all auditable, none inline."""

    #: Below this confidence a candidate can be recommended but not acted on.
    min_confidence_to_execute: float = 0.60
    #: Below this, or below ``min_samples_for_autonomy``, unattended execution
    #: is refused even when the operator asked for it.
    min_confidence_for_autonomy: float = 0.80
    min_samples_for_autonomy: int = 5
    #: A candidate that has failed this many times stops being retried. Part 17
    #: case 7: repeated failure must not become an infinite loop.
    max_failures_before_block: int = 2
    #: Occurrences required before a signal is treated as established enough to
    #: act on unattended.
    min_occurrences_for_autonomy: int = 2

    def validate(self) -> None:
        for name in ("min_confidence_to_execute", "min_confidence_for_autonomy"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not (0.0 <= float(value) <= 1.0):
                raise ValueError(f"{name} must be within [0.0, 1.0], got {value!r}")
        if self.min_confidence_for_autonomy < self.min_confidence_to_execute:
            raise ValueError(
                "min_confidence_for_autonomy cannot be below min_confidence_to_execute"
            )
        for name in (
            "min_samples_for_autonomy",
            "max_failures_before_block",
            "min_occurrences_for_autonomy",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {value!r}")


@dataclass
class PolicyVerdict:
    """The tier a candidate is granted, and every reason it was capped."""

    candidate_id: str = ""
    configured_tier: str = AutonomyTier.OBSERVE_ONLY
    granted_tier: str = AutonomyTier.OBSERVE_ONLY
    blocked: bool = False
    #: Reasons the tier was lowered, in evaluation order.
    cap_reasons: list[str] = field(default_factory=list)
    #: Hard refusals. A non-empty list always implies ``blocked``.
    blocking_reasons: list[str] = field(default_factory=list)
    #: Paths that were refused outright (traversal, protected, outside root).
    rejected_paths: list[str] = field(default_factory=list)

    @property
    def may_execute(self) -> bool:
        return not self.blocked and self.granted_tier in EXECUTING_TIERS

    @property
    def may_plan(self) -> bool:
        return not self.blocked and tier_rank(self.granted_tier) >= tier_rank(
            AutonomyTier.PLAN_ONLY
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "configured_tier": self.configured_tier,
            "granted_tier": self.granted_tier,
            "blocked": self.blocked,
            "may_execute": self.may_execute,
            "may_plan": self.may_plan,
            "cap_reasons": list(self.cap_reasons),
            "blocking_reasons": list(self.blocking_reasons),
            "rejected_paths": list(self.rejected_paths),
        }


class MaintenanceExecutionPolicy:
    """Decides how much autonomy a maintenance candidate may be granted.

    Structural properties the tests assert:

    * The granted tier is never stronger than the configured tier.
    * A candidate touching a protected file is never granted an executing tier.
    * A candidate with an unsafe path is blocked outright, and the path is
      named in the verdict rather than silently dropped.
    * The decision is a pure function of ``(candidate, configured tier, budget,
      thresholds, root)`` - no clock, no filesystem writes, no global state.

    What it deliberately does *not* do is decide anything about validation.
    Which commands run against a maintenance change, and at what scope, remains
    entirely :class:`~local_agent.validation_decision.ValidationDecisionEngine`'s
    business - this policy has no opinion, no override and no import path to it.
    """

    def __init__(
        self,
        *,
        thresholds: PolicyThresholds | None = None,
        protected_paths: frozenset[str] | None = None,
        repository_root: str | Path | None = None,
    ):
        self.thresholds = thresholds or PolicyThresholds()
        self.thresholds.validate()
        # The floor is unioned, never replaced: a caller supplying its own
        # protected set can only ever add to the built-in one.
        self.protected_paths = PROTECTED_RELATIVE_PATHS | frozenset(protected_paths or ())
        self.repository_root = Path(repository_root).resolve() if repository_root else None

    def is_protected(self, path: Any) -> bool:
        """True when ``path`` is on this policy's protected floor.

        Case-insensitive, for the reason documented on
        :func:`~local_agent.maintenance.is_protected_relative_path`: on
        Windows and macOS ``Local_Agent/Tool_Engine.py`` is the very same file
        as ``local_agent/tool_engine.py``, and a case-sensitive membership test
        would grant an executing tier to a candidate targeting the tool engine.
        """
        return is_protected_relative_path(path, extra=self.protected_paths)

    def decide(
        self,
        candidate: MaintenanceCandidate,
        *,
        configured_tier: str,
        budget: MaintenanceBudget | None = None,
        raw_paths: Sequence[str] | None = None,
    ) -> PolicyVerdict:
        """Grant a tier no stronger than ``configured_tier``.

        ``raw_paths`` lets a caller submit the *pre-sanitisation* paths a
        candidate claimed, so that a traversal attempt is reported as a
        blocking reason rather than merely vanishing during normalisation.
        """
        budget = budget or MaintenanceBudget()
        configured = (
            str(configured_tier)
            if str(configured_tier) in TIER_ORDER
            else AutonomyTier.OBSERVE_ONLY
        )
        verdict = PolicyVerdict(
            candidate_id=candidate.candidate_id,
            configured_tier=configured,
            granted_tier=configured,
        )
        if str(configured_tier) not in TIER_ORDER:
            verdict.cap_reasons.append(
                f"unrecognised autonomy tier {sanitize_text(configured_tier, limit=64)!r}; "
                "defaulted to observe_only"
            )

        caps: list[str] = [configured]

        # -- hard refusals -------------------------------------------------
        rejected = self._rejected_paths(candidate, raw_paths)
        if rejected:
            verdict.rejected_paths = rejected
            verdict.blocking_reasons.append(
                "candidate references path(s) outside the repository or inside a "
                "protected directory: " + ", ".join(rejected[:5])
            )
        protected_hits = sorted(
            path for path in candidate.affected_files if self.is_protected(path)
        )
        if protected_hits:
            verdict.blocking_reasons.append(
                "candidate targets protected file(s): " + ", ".join(protected_hits)
            )
        if candidate.failure_count >= self.thresholds.max_failures_before_block > 0:
            verdict.blocking_reasons.append(
                f"candidate already failed {candidate.failure_count} time(s); "
                "further automated attempts are refused"
            )
        if verdict.blocking_reasons:
            verdict.blocked = True
            verdict.granted_tier = weakest_tier(configured, AutonomyTier.OBSERVE_ONLY)
            return verdict

        # -- graduated caps ------------------------------------------------
        if candidate.kind not in AUTONOMOUSLY_ACTIONABLE_KINDS:
            caps.append(AutonomyTier.RECOMMEND)
            verdict.cap_reasons.append(
                f"signal kind '{candidate.kind}' has no safe automated remedy; "
                "recommendation only"
            )
        if not candidate.affected_files:
            caps.append(AutonomyTier.RECOMMEND)
            verdict.cap_reasons.append(
                "candidate names no affected file, so there is nothing to change"
            )
        if candidate.confidence < self.thresholds.min_confidence_to_execute:
            caps.append(AutonomyTier.RECOMMEND)
            verdict.cap_reasons.append(
                f"confidence {candidate.confidence:.2f} is below the execution "
                f"threshold {self.thresholds.min_confidence_to_execute:.2f}"
            )
        if len(candidate.affected_files) > budget.max_changed_files_per_candidate:
            caps.append(AutonomyTier.PLAN_ONLY)
            verdict.cap_reasons.append(
                f"{len(candidate.affected_files)} affected file(s) exceeds the "
                f"per-candidate budget of {budget.max_changed_files_per_candidate}"
            )
        if budget.max_candidates_executed <= 0:
            caps.append(AutonomyTier.PLAN_ONLY)
            verdict.cap_reasons.append(
                "budget permits no executions in this run"
            )
        if candidate.severity == SEVERITY_CRITICAL and (
            candidate.confidence < self.thresholds.min_confidence_for_autonomy
        ):
            # A critical problem we are not sure about is the worst possible
            # thing to act on unattended: the blast radius of being wrong is
            # maximal exactly where the evidence is weakest.
            caps.append(AutonomyTier.PLAN_ONLY)
            verdict.cap_reasons.append(
                "critical severity with sub-threshold confidence; planning only"
            )
        if candidate.uncertainty:
            caps.append(AutonomyTier.EXECUTE_WITH_EXISTING_APPROVAL)
            verdict.cap_reasons.append(
                f"{len(candidate.uncertainty)} explicit uncertainty caveat(s); "
                "human approval required"
            )
        if (
            candidate.confidence < self.thresholds.min_confidence_for_autonomy
            or candidate.sample_size < self.thresholds.min_samples_for_autonomy
            or candidate.occurrence_count < self.thresholds.min_occurrences_for_autonomy
        ):
            caps.append(AutonomyTier.EXECUTE_WITH_EXISTING_APPROVAL)
            verdict.cap_reasons.append(
                f"evidence below the unattended-execution bar (confidence "
                f"{candidate.confidence:.2f} / {candidate.sample_size} sample(s) / "
                f"{candidate.occurrence_count} occurrence(s)); human approval required"
            )

        verdict.granted_tier = weakest_tier(*caps)
        return verdict

    # -- helpers -----------------------------------------------------------

    def _rejected_paths(
        self, candidate: MaintenanceCandidate, raw_paths: Sequence[str] | None
    ) -> list[str]:
        """Paths the candidate claimed that must never be acted on.

        Checks the raw claims *and* re-checks the sanitised ones. Re-checking
        looks redundant - the constructor already normalised them - and is kept
        deliberately: a candidate can be constructed by loading a persisted
        record written by a different version of this code, and the policy is
        the last gate before planning.
        """
        rejected: list[str] = []
        for raw in list(raw_paths or []):
            text = sanitize_text(raw, limit=200)
            if text and not sanitize_relative_path(raw):
                rejected.append(text)
        for path in candidate.affected_files:
            if sanitize_relative_path(path) != path:
                rejected.append(sanitize_text(path, limit=200))
                continue
            if any(is_protected_directory_segment(segment) for segment in path.split("/")):
                rejected.append(path)
                continue
            if self.repository_root is not None and not self._inside_root(path):
                rejected.append(path)
        # Deterministic and de-duplicated: this list is printed to operators
        # and compared in tests.
        return sorted(set(rejected))

    def _inside_root(self, relative: str) -> bool:
        """True when ``relative`` resolves inside the repository root.

        Resolution is done without touching the filesystem for the common case,
        but ``Path.resolve`` is still used because a symlinked directory inside
        the repository could otherwise be used to reach outside it. A path that
        cannot be resolved at all is treated as outside - refusing an
        unresolvable path is always safe, whereas admitting one is not.
        """
        assert self.repository_root is not None
        try:
            resolved = (self.repository_root / relative).resolve()
        except (OSError, ValueError, RuntimeError):
            return False
        try:
            resolved.relative_to(self.repository_root)
        except ValueError:
            return False
        return True


def describe_tier(tier: str) -> str:
    """One-line operator-facing description of an autonomy tier."""
    return {
        AutonomyTier.OBSERVE_ONLY: "record the signal only; produce no recommendation",
        AutonomyTier.RECOMMEND: "surface a recommendation for a human to act on",
        AutonomyTier.PLAN_ONLY: "generate a plan but never execute it",
        AutonomyTier.EXECUTE_WITH_EXISTING_APPROVAL: (
            "execute through the normal pipeline, subject to the existing approval gate"
        ),
        AutonomyTier.EXECUTE_AUTONOMOUSLY: (
            "execute through the normal pipeline without an interactive approval prompt; "
            "every other safety control still applies"
        ),
    }.get(str(tier), "unknown tier")


def weights_sum() -> float:
    """Sum of the priority weights. Asserted to be 1.0 by the tests."""
    return sum(_ALL_WEIGHTS)
