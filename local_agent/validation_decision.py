"""Phase 4.18: the single authoritative post-change validation decision.

Before this module, "which commands actually run after a change" was decided
by two separate, hand-written pieces of logic living inside
:class:`~local_agent.orchestrator.Orchestrator`:
``_semantic_targeted_commands`` (impact analysis -> target commands) and
``_apply_evidence_reuse`` (candidate evidence -> which of those commands can
be skipped). They agreed by convention, not by construction - nothing stopped
the two from drifting apart, and nothing besides the orchestrator could ever
exercise either without standing up a whole ``Orchestrator``.

:class:`ValidationDecisionEngine` is that logic, moved into one pure,
independently-testable place. It consumes a
:class:`~local_agent.semantic_impact.ChangeImpactReport` (already computed;
this module does not build one), an optional
:class:`~local_agent.evidence.EvidenceLedger`, and a small bundle of policy
values, and produces one :class:`ValidationDecision`: the commands to run,
which of them were satisfied by reused evidence and why, and a plain-language
account of every reason involved.

It does not re-decide confidence or scope - those are
:class:`~local_agent.semantic_impact.ChangeImpactReport`'s job and are already
audited (:func:`~local_agent.semantic_impact.recommend_validation_scope` is the
single source of truth for "uncertainty only ever widens validation"). This
module only decides, given that already-computed scope, which concrete
commands satisfy it and which of those can be skipped because equivalent
evidence already exists - a strictly narrower and lower-risk kind of decision:
skipping a command based on reuse never runs *fewer* commands than the
mandatory full-suite validation still requires; it only avoids re-executing
one that provably already ran against byte-identical, policy-identical,
environment-identical inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .evidence import (
    EvidenceLedger,
    EvidenceReuseDecision,
    REASON_OK,
)
from .models import CommandSpec
from .semantic_impact import (
    CONFIDENCE_ORDER,
    SCOPE_TARGETED,
    TIER_BROAD,
    ChangeImpactReport,
    confidence_at_least,
)

#: Coarse safety label for a decision, independent of the specific scope name.
#: Exists so a caller that only wants "is this being careful" does not need to
#: know the scope vocabulary.
SAFETY_NARROW = "safe_narrow"
SAFETY_EXPANDED = "safe_expanded"
SAFETY_BROAD = "conservative_broad"

_SAFETY_BY_SCOPE = {
    "targeted": SAFETY_NARROW,
    "expanded": SAFETY_EXPANDED,
    "broad": SAFETY_BROAD,
}


def _confidence_score(confidence_level: str) -> float:
    """Numeric position of ``confidence_level`` in ``CONFIDENCE_ORDER``, as a
    fraction in ``[0, 1]``. Purely a presentation convenience over the
    already-authoritative ordinal level; nothing is decided from this value."""
    if confidence_level not in CONFIDENCE_ORDER:
        return 0.0
    return CONFIDENCE_ORDER.index(confidence_level) / (len(CONFIDENCE_ORDER) - 1)


@dataclass
class ReuseAttempt:
    """One command's reuse verdict, kept even when reuse was refused."""

    command: tuple[str, ...]
    reusable: bool
    reason: str
    time_saved_seconds: float = 0.0
    #: The evidence entry that justified reuse, when ``reusable`` is True - the
    #: caller typically wants to persist this onto its own audit trail.
    evidence: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": list(self.command),
            "reusable": self.reusable,
            "reason": self.reason,
            "time_saved_seconds": round(self.time_saved_seconds, 4),
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "ReuseAttempt":
        if not isinstance(data, dict):
            return cls(command=(), reusable=False, reason="")
        evidence = data.get("evidence")
        return cls(
            command=tuple(str(t) for t in (data.get("command") or [])),
            reusable=bool(data.get("reusable", False)),
            reason=str(data.get("reason", "")),
            time_saved_seconds=float(data.get("time_saved_seconds", 0.0) or 0.0),
            evidence=evidence if isinstance(evidence, dict) else None,
        )


@dataclass
class ValidationDecision:
    """The one thing the rest of the system needs: what to run, and why."""

    scope: str = SCOPE_TARGETED
    confidence_level: str = "low"
    confidence_score: float = 0.0
    selected_tests: list[str] = field(default_factory=list)
    selected_commands: list[CommandSpec] = field(default_factory=list)
    reuse_attempts: list[ReuseAttempt] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    uncertainty_sources: list[str] = field(default_factory=list)
    safety_level: str = SAFETY_BROAD
    time_saved_seconds: float = 0.0

    @property
    def reused_count(self) -> int:
        return sum(1 for attempt in self.reuse_attempts if attempt.reusable)

    @property
    def denied_count(self) -> int:
        return sum(1 for attempt in self.reuse_attempts if not attempt.reusable)

    @property
    def reused_evidence(self) -> list[dict[str, Any]]:
        """Evidence dicts for every granted reuse, for a caller's own audit trail."""
        return [a.evidence for a in self.reuse_attempts if a.reusable and a.evidence is not None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "confidence_level": self.confidence_level,
            "confidence_score": round(self.confidence_score, 4),
            "selected_tests": list(self.selected_tests),
            "selected_commands": [c.to_dict() for c in self.selected_commands],
            "reuse_attempts": [a.to_dict() for a in self.reuse_attempts],
            "reasons": list(self.reasons),
            "uncertainty_sources": list(self.uncertainty_sources),
            "safety_level": self.safety_level,
            "time_saved_seconds": round(self.time_saved_seconds, 4),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "ValidationDecision":
        if not isinstance(data, dict):
            return cls()
        return cls(
            scope=str(data.get("scope", SCOPE_TARGETED)),
            confidence_level=str(data.get("confidence_level", "low")),
            confidence_score=float(data.get("confidence_score", 0.0) or 0.0),
            selected_tests=[str(s) for s in (data.get("selected_tests") or [])],
            selected_commands=[
                CommandSpec.from_dict(c) for c in (data.get("selected_commands") or [])
                if isinstance(c, dict)
            ],
            reuse_attempts=[
                ReuseAttempt.from_dict(a) for a in (data.get("reuse_attempts") or [])
            ],
            reasons=[str(r) for r in (data.get("reasons") or [])],
            uncertainty_sources=[str(u) for u in (data.get("uncertainty_sources") or [])],
            safety_level=str(data.get("safety_level", SAFETY_BROAD)),
            time_saved_seconds=float(data.get("time_saved_seconds", 0.0) or 0.0),
        )


class ValidationDecisionEngine:
    """Turns one impact report (+ optional evidence) into one decision.

    Holds no mutable state across calls beyond its constructor arguments, so a
    fresh instance per decision (the normal usage) can never leak state between
    unrelated changes, worktrees, or checkpoint/resume cycles.
    """

    def __init__(
        self,
        *,
        min_confidence: str = "high",
        reuse_enabled: bool = False,
        max_age_seconds: float | None = None,
        policy_fingerprint: str | None = None,
        analyzer_version: str | None = None,
    ):
        self.min_confidence = min_confidence
        self.reuse_enabled = reuse_enabled
        self.max_age_seconds = max_age_seconds
        self.policy_fingerprint = policy_fingerprint
        self.analyzer_version = analyzer_version

    def decide(
        self,
        impact: ChangeImpactReport,
        *,
        current_root: str | Path,
        lexical_commands: list[CommandSpec],
        ledger: EvidenceLedger | None = None,
        executable_fingerprint_of: Any | None = None,
    ) -> ValidationDecision:
        """Build the decision for one already-computed impact report.

        ``lexical_commands`` are the pre-existing filename-heuristic commands
        (Phase 4.4); they are the floor this decision can never go below when
        confidence is not high enough to narrow. ``executable_fingerprint_of``,
        when given, is a ``command -> str`` callable used to compute each
        command's environment-identity check for reuse; omitting it (the
        default) skips that specific check, matching a caller that does not
        care about environment drift.
        """
        semantic_specs, seen = self._semantic_commands(impact, current_root)
        narrow_allowed = (
            confidence_at_least(impact.confidence, self.min_confidence)
            and impact.recommended_scope == SCOPE_TARGETED
            and bool(semantic_specs)
        )

        reasons = list(impact.scope_reasons)
        if narrow_allowed:
            selected = semantic_specs
            reasons.append(
                f"{impact.confidence.upper()} confidence at {impact.recommended_scope} "
                f"scope permits {len(selected)} targeted command(s) selected from the "
                "dependency graph"
            )
        else:
            selected = list(semantic_specs)
            for spec in lexical_commands:
                if tuple(spec.command) not in seen:
                    seen.add(tuple(spec.command))
                    selected.append(spec)
            if selected:
                reasons.append(
                    f"{impact.confidence.upper()} confidence at {impact.recommended_scope} "
                    f"scope is not narrow enough to drop the {len(lexical_commands)} "
                    "filename-heuristic command(s); running the union"
                )

        decision = ValidationDecision(
            scope=impact.recommended_scope,
            confidence_level=impact.confidence,
            confidence_score=_confidence_score(impact.confidence),
            selected_tests=sorted({spec.command[-1] for spec in selected if len(spec.command) > 1}),
            reasons=reasons,
            uncertainty_sources=list(impact.evidence.degradations),
            safety_level=_SAFETY_BY_SCOPE.get(impact.recommended_scope, SAFETY_BROAD),
        )

        remaining, reuse_attempts, saved = self.apply_reuse(
            selected, impact, current_root, ledger, executable_fingerprint_of
        )
        decision.selected_commands = remaining
        decision.reuse_attempts = reuse_attempts
        decision.time_saved_seconds = saved
        return decision

    # -- internals -----------------------------------------------------

    def _semantic_commands(
        self, impact: ChangeImpactReport, current_root: str | Path
    ) -> tuple[list[CommandSpec], set[tuple[str, ...]]]:
        specs: list[CommandSpec] = []
        seen: set[tuple[str, ...]] = set()
        root = Path(current_root)
        for target in impact.validation_targets:
            if target.tier == TIER_BROAD:
                continue
            command = tuple(target.command)
            if command in seen:
                continue
            if len(command) > 1 and not (root / command[-1]).is_file():
                continue
            seen.add(command)
            specs.append(
                CommandSpec(
                    name=f"impact_{Path(target.path).stem or 'target'}",
                    command=command,
                    reason=target.selected_because,
                    category="unit_test",
                    risk="low",
                    destructive=False,
                )
            )
        return specs, seen

    def apply_reuse(
        self,
        selected: list[CommandSpec],
        impact: ChangeImpactReport,
        current_root: str | Path,
        ledger: EvidenceLedger | None,
        executable_fingerprint_of: Any | None = None,
    ) -> tuple[list[CommandSpec], list[ReuseAttempt], float]:
        """Filter ``selected`` against ``ledger``, standalone from :meth:`decide`.

        Public so a caller that already has its own selected command list (for
        example one that also merged in lexical-heuristic commands the impact
        report never produced) can still route reuse decisions through this one
        engine instead of re-deriving targets from ``impact`` a second time.
        """
        if ledger is None or not selected:
            return list(selected), [], 0.0

        # Qualified names ("A.m"), not the plain ``changed_symbol_names``: this
        # must byte-for-byte match what :meth:`CandidateWorkspaceLoop
        # ._record_candidate_evidence` recorded evidence against, or every
        # reuse attempt would spuriously fail on a symbol-set mismatch.
        symbols = sorted({symbol.qualified_name for symbol in impact.changed_symbols})
        base_relevant = set(impact.changed_files) | set(impact.affected_files)

        remaining: list[CommandSpec] = []
        attempts: list[ReuseAttempt] = []
        saved = 0.0
        for spec in selected:
            command = tuple(spec.command)
            relevant = set(base_relevant)
            tail = command[-1] if len(command) > 1 else ""
            if tail.endswith(".py"):
                relevant.add(tail.replace("\\", "/"))
            executable_fp = (
                executable_fingerprint_of(command) if executable_fingerprint_of else None
            )
            outcome: EvidenceReuseDecision = ledger.find_reusable(
                command=command,
                current_root=current_root,
                relevant_files=sorted(relevant),
                relevant_symbols=symbols,
                min_confidence=self.min_confidence,
                enabled=self.reuse_enabled,
                max_age_seconds=self.max_age_seconds,
                policy_fingerprint=self.policy_fingerprint,
                executable_fingerprint=executable_fp,
                analyzer_version=self.analyzer_version,
            )
            attempts.append(
                ReuseAttempt(
                    command=command,
                    reusable=outcome.reusable,
                    reason=outcome.reason,
                    time_saved_seconds=outcome.time_saved_seconds,
                    evidence=(
                        outcome.evidence.to_dict()
                        if outcome.reusable and outcome.evidence is not None
                        else None
                    ),
                )
            )
            if outcome.reusable:
                saved += outcome.time_saved_seconds
                continue
            remaining.append(spec)
        return remaining, attempts, saved
