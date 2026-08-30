"""Phase 4.21: maintenance signal extraction.

The analyzer *reads* the intelligence the agent already produces and turns it
into normalised :class:`~local_agent.maintenance.MaintenanceCandidate` records.
It computes almost nothing of its own. Every extractor here consumes an
existing subsystem's output:

===========================  ==================================================
source                       signals derived
===========================  ==================================================
``ValidationLifecycleStore``  recurring defects, repeated repairs, abandoned
                              work, candidate-stage instability
``ValidationTelemetryStore``  broad-validation pressure, evidence-reuse
                              failure, false confidence, analysis degradation
``SemanticGraph``             architectural risk, analyzer blind spots, parse
                              failures, test gaps
``RepositoryKnowledgeGraph``  known failure patterns
``GitIntegration``            file churn (an *enrichment* only - churn alone
                              never produces a candidate)
===========================  ==================================================

Two design commitments run through all of it.

**Evidence, not heuristics firing.** Every candidate carries the ids of the
records it was derived from, and a confidence that is a conservative Wilson
lower bound over the number of observations wherever the signal is a *rate*.
A structural observation ("this file does not parse") is reported at full
confidence with a sample size of one, because it is a fact rather than an
inference - and the two are distinguished explicitly rather than blurred into
one number.

**Extractor isolation.** Each extractor runs inside its own guard. A
malformed historical record, a missing attribute from an older schema, or an
exception deep inside a graph traversal costs the run *that extractor's*
signals and nothing else. The failures are collected and reported, never
swallowed silently - a scan that quietly produced half its signals would be
worse than one that produced none.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .maintenance import (
    MAX_EVIDENCE_REFS,
    MaintenanceCandidate,
    MaintenanceSignal,
    PROVENANCE_KNOWLEDGE_GRAPH,
    PROVENANCE_LIFECYCLE,
    PROVENANCE_SEMANTIC_GRAPH,
    PROVENANCE_TELEMETRY,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
    candidate_identity,
    sanitize_path_list,
    sanitize_relative_path,
    sanitize_text,
)
from .semantic_impact import SCOPE_BROAD, looks_like_test_path
from .validation_telemetry import wilson_lower_bound

#: Confidence assigned to a *directly observed structural fact* (a file that
#: fails to parse, an import that resolves to nothing). These are not
#: statistical inferences, so a Wilson bound would be the wrong instrument;
#: the sample size is reported as 1 so nothing downstream can mistake this for
#: a well-supported rate.
STRUCTURAL_OBSERVATION_CONFIDENCE = 1.0


@dataclass(frozen=True)
class MaintenanceThresholds:
    """Every threshold the analyzer uses, in one auditable place.

    Named constants rather than inline numbers, because Part 4 of the
    specification forbids hidden magic values - and because an operator who
    disagrees with "two occurrences is recurrence" needs somewhere to say so.

    The values are chosen conservatively: each one is the point at which the
    underlying subsystem's own conventions start to consider a pattern real.
    ``min_defect_recurrence = 2`` matches Phase 4.20's repeated-defect rate,
    which counts a defect as repeated the second time its exact signature is
    seen.
    """

    # lifecycle-derived
    min_defect_recurrence: int = 2
    min_repair_iterations: int = 2
    min_lifecycles_for_rates: int = 5
    abandonment_rate_threshold: float = 0.3
    candidate_failure_threshold: int = 3

    # telemetry-derived
    min_decisions_for_rates: int = 10
    broad_scope_rate_threshold: float = 0.6
    reuse_rejection_rate_threshold: float = 0.7
    degradation_rate_threshold: float = 0.3
    false_confidence_minimum: int = 1

    # graph-derived
    min_fan_in_for_risk: int = 8
    min_churn_for_risk: int = 4
    min_unresolved_imports: int = 3
    test_gap_min_fan_in: int = 3

    # knowledge-derived
    min_failure_pattern_occurrences: int = 3

    # global
    max_candidates_per_kind: int = 10
    churn_commit_window: int = 200

    def validate(self) -> None:
        for name, value in (
            ("min_defect_recurrence", self.min_defect_recurrence),
            ("min_repair_iterations", self.min_repair_iterations),
            ("min_lifecycles_for_rates", self.min_lifecycles_for_rates),
            ("candidate_failure_threshold", self.candidate_failure_threshold),
            ("min_decisions_for_rates", self.min_decisions_for_rates),
            ("false_confidence_minimum", self.false_confidence_minimum),
            ("min_fan_in_for_risk", self.min_fan_in_for_risk),
            ("min_churn_for_risk", self.min_churn_for_risk),
            ("min_unresolved_imports", self.min_unresolved_imports),
            ("test_gap_min_fan_in", self.test_gap_min_fan_in),
            ("min_failure_pattern_occurrences", self.min_failure_pattern_occurrences),
            ("max_candidates_per_kind", self.max_candidates_per_kind),
            ("churn_commit_window", self.churn_commit_window),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        for name, value in (
            ("abandonment_rate_threshold", self.abandonment_rate_threshold),
            ("broad_scope_rate_threshold", self.broad_scope_rate_threshold),
            ("reuse_rejection_rate_threshold", self.reuse_rejection_rate_threshold),
            ("degradation_rate_threshold", self.degradation_rate_threshold),
        ):
            if not isinstance(value, (int, float)) or not (0.0 < float(value) <= 1.0):
                raise ValueError(f"{name} must be in (0.0, 1.0], got {value!r}")


@dataclass
class AnalysisResult:
    """Everything one scan produced, including what went wrong."""

    candidates: list[MaintenanceCandidate] = field(default_factory=list)
    #: ``extractor name -> error text`` for every extractor that raised.
    extractor_errors: dict[str, str] = field(default_factory=dict)
    #: Extractors that ran but had no data to work from.
    skipped: dict[str, str] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    sources_available: dict[str, bool] = field(default_factory=dict)

    @property
    def degraded(self) -> bool:
        """True when the scan could not see everything it wanted to.

        A degraded scan is not a failed one, but it must never be treated as a
        clean bill of health: "we found no problems" and "we could not look"
        are different statements, and only the second one is compatible with
        the repository being in trouble.
        """
        return bool(self.extractor_errors) or not all(self.sources_available.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "extractor_errors": dict(sorted(self.extractor_errors.items())),
            "skipped": dict(sorted(self.skipped.items())),
            "elapsed_seconds": self.elapsed_seconds,
            "sources_available": dict(sorted(self.sources_available.items())),
            "degraded": self.degraded,
        }


class MaintenanceAnalyzer:
    """Turns existing agent intelligence into maintenance candidates.

    Holds no repository handles of its own beyond a root path used for
    relative-path normalisation, and mutates nothing. Every input is passed in,
    which is what makes the whole thing testable against synthetic stores and
    what stops it from acquiring a second, divergent copy of any subsystem's
    view of the repository.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        thresholds: MaintenanceThresholds | None = None,
    ):
        self.root = Path(root)
        self.thresholds = thresholds or MaintenanceThresholds()
        self.thresholds.validate()

    # -- entry point -------------------------------------------------------

    def analyze(
        self,
        *,
        lifecycle_store: Any = None,
        telemetry_store: Any = None,
        semantic_graph: Any = None,
        knowledge_graph: Any = None,
        churn: Mapping[str, int] | None = None,
        max_candidates: int | None = None,
    ) -> AnalysisResult:
        """Run every extractor and return the merged, bounded candidate set."""
        started = time.perf_counter()
        result = AnalysisResult(
            sources_available={
                "lifecycle": lifecycle_store is not None,
                "telemetry": telemetry_store is not None,
                "semantic_graph": semantic_graph is not None,
                "knowledge_graph": knowledge_graph is not None,
                "churn": bool(churn),
            }
        )
        churn_map = _normalize_churn(churn)

        extractors: list[tuple[str, Callable[[], list[MaintenanceCandidate]]]] = [
            ("recurring_defects", lambda: self._recurring_defects(lifecycle_store)),
            ("repeated_repairs", lambda: self._repeated_repairs(lifecycle_store)),
            ("abandoned_work", lambda: self._abandoned_work(lifecycle_store)),
            ("candidate_instability", lambda: self._candidate_instability(lifecycle_store)),
            ("validation_pressure", lambda: self._validation_pressure(telemetry_store)),
            ("evidence_reuse", lambda: self._evidence_reuse(telemetry_store)),
            ("false_confidence", lambda: self._false_confidence(telemetry_store)),
            ("analysis_degradation", lambda: self._analysis_degradation(telemetry_store)),
            (
                "architectural_risk",
                lambda: self._architectural_risk(semantic_graph, churn_map),
            ),
            ("analyzer_blind_spots", lambda: self._analyzer_blind_spots(semantic_graph)),
            ("parse_failures", lambda: self._parse_failures(semantic_graph)),
            ("test_gaps", lambda: self._test_gaps(semantic_graph)),
            ("failure_patterns", lambda: self._failure_patterns(knowledge_graph)),
        ]

        collected: dict[str, MaintenanceCandidate] = {}
        per_kind: dict[str, int] = {}
        for name, extractor in extractors:
            try:
                produced = extractor()
            except Exception as exc:  # pragma: no cover - defensive by design
                # Isolate: one bad extractor costs its own signals only. The
                # error text is sanitised because it can embed file paths and
                # subprocess output.
                result.extractor_errors[name] = sanitize_text(f"{type(exc).__name__}: {exc}")
                continue
            if not produced:
                result.skipped.setdefault(name, "no signal")
                continue
            for candidate in produced:
                if per_kind.get(candidate.kind, 0) >= self.thresholds.max_candidates_per_kind:
                    continue
                existing = collected.get(candidate.candidate_id)
                if existing is not None:
                    existing.merge_observation(candidate)
                    continue
                collected[candidate.candidate_id] = candidate
                per_kind[candidate.kind] = per_kind.get(candidate.kind, 0) + 1

        candidates = sorted(collected.values(), key=lambda c: c.candidate_id)
        if max_candidates is not None:
            candidates = candidates[: max(0, int(max_candidates))]
        result.candidates = candidates
        result.elapsed_seconds = time.perf_counter() - started
        return result

    # -- lifecycle-derived signals ----------------------------------------

    def _recurring_defects(self, store: Any) -> list[MaintenanceCandidate]:
        """The same exact defect signature seen across multiple lifecycles.

        Uses Phase 4.20's conservative exact-match fingerprint, so two defects
        only count as "the same" when every normalised field agrees. That is
        biased against merging - which is the bias we want here, since a false
        merge would invent recurrence that never happened.
        """
        lifecycles = _lifecycles_of(store)
        if not lifecycles:
            return []
        occurrences: dict[str, int] = {}
        lifecycle_counts: dict[str, set[str]] = {}
        descriptions: dict[str, str] = {}
        files: dict[str, set[str]] = {}
        for record in lifecycles:
            seen_here: set[str] = set()
            for iteration in getattr(record, "iterations", []) or []:
                signature = getattr(iteration, "defect_signature", None)
                if signature is None or getattr(signature, "is_empty", False):
                    continue
                fingerprint = str(getattr(signature, "fingerprint", ""))
                if not fingerprint:
                    continue
                occurrences[fingerprint] = occurrences.get(fingerprint, 0) + 1
                seen_here.add(fingerprint)
                descriptions.setdefault(fingerprint, _describe_signature(signature))
                affected = sanitize_relative_path(getattr(signature, "affected_file", ""))
                if affected:
                    files.setdefault(fingerprint, set()).add(affected)
            for fingerprint in seen_here:
                lifecycle_counts.setdefault(fingerprint, set()).add(
                    str(getattr(record, "lifecycle_id", ""))
                )

        results: list[MaintenanceCandidate] = []
        total = len(lifecycles)
        for fingerprint, count in sorted(occurrences.items()):
            spread = len(lifecycle_counts.get(fingerprint, ()))
            if spread < self.thresholds.min_defect_recurrence:
                continue
            severity = SEVERITY_HIGH if spread >= 4 else SEVERITY_MEDIUM
            results.append(
                MaintenanceCandidate(
                    kind=MaintenanceSignal.RECURRING_DEFECT,
                    subject=fingerprint,
                    title=f"Defect {fingerprint} recurs across {spread} lifecycles",
                    detail=descriptions.get(fingerprint, ""),
                    provenance=PROVENANCE_LIFECYCLE,
                    severity=severity,
                    confidence=wilson_lower_bound(spread, max(total, spread)),
                    sample_size=total,
                    uncertainty=(
                        []
                        if total >= self.thresholds.min_lifecycles_for_rates
                        else [f"only {total} lifecycle(s) recorded"]
                    ),
                    evidence_refs=sorted(lifecycle_counts.get(fingerprint, ()))[:MAX_EVIDENCE_REFS],
                    affected_files=sorted(files.get(fingerprint, ())),
                    recommended_action=(
                        "Investigate the shared root cause behind this repeated "
                        "validation failure and add a regression test for it."
                    ),
                    estimated_effort=2.0,
                    metrics={"occurrences": float(count), "lifecycles": float(spread)},
                )
            )
        return results

    def _repeated_repairs(self, store: Any) -> list[MaintenanceCandidate]:
        """Areas where implementation routinely needs several repair rounds."""
        lifecycles = _lifecycles_of(store)
        if not lifecycles:
            return []
        by_file: dict[str, list[str]] = {}
        repair_totals: dict[str, int] = {}
        for record in lifecycles:
            repairs = int(getattr(record, "repair_count", 0) or 0)
            if repairs < self.thresholds.min_repair_iterations:
                continue
            for path in _lifecycle_files(record):
                by_file.setdefault(path, []).append(str(getattr(record, "lifecycle_id", "")))
                repair_totals[path] = repair_totals.get(path, 0) + repairs

        results: list[MaintenanceCandidate] = []
        total = len(lifecycles)
        for path, ids in sorted(by_file.items()):
            spread = len(ids)
            results.append(
                MaintenanceCandidate(
                    kind=MaintenanceSignal.REPEATED_REPAIR,
                    subject=path,
                    title=f"{path} needed repair rounds in {spread} lifecycle(s)",
                    detail=(
                        f"{repair_totals.get(path, 0)} repair iteration(s) recorded "
                        f"against this file across {spread} lifecycle(s)."
                    ),
                    provenance=PROVENANCE_LIFECYCLE,
                    severity=SEVERITY_MEDIUM if spread >= 2 else SEVERITY_LOW,
                    confidence=wilson_lower_bound(spread, max(total, spread)),
                    sample_size=total,
                    uncertainty=(
                        []
                        if total >= self.thresholds.min_lifecycles_for_rates
                        else [f"only {total} lifecycle(s) recorded"]
                    ),
                    evidence_refs=sorted(set(ids))[:MAX_EVIDENCE_REFS],
                    affected_files=[path],
                    recommended_action=(
                        "Review this file's tests and interfaces; implementation "
                        "attempts here repeatedly fail validation on the first pass."
                    ),
                    estimated_effort=3.0,
                    metrics={
                        "repair_iterations": float(repair_totals.get(path, 0)),
                        "lifecycles": float(spread),
                    },
                )
            )
        return results

    def _abandoned_work(self, store: Any) -> list[MaintenanceCandidate]:
        """A high share of lifecycles that never reached completion.

        Reported as a single repository-level candidate rather than one per
        file: abandonment is a property of the *process*, and attributing it to
        whichever files happened to be open at the time would be a fabricated
        localisation.
        """
        lifecycles = _lifecycles_of(store)
        resolved = [
            record
            for record in lifecycles
            if getattr(record, "is_terminal", False)
        ]
        if len(resolved) < self.thresholds.min_lifecycles_for_rates:
            return []
        abandoned = [
            record
            for record in resolved
            if str(getattr(record, "state", "")) in {"abandoned", "failed"}
        ]
        rate = len(abandoned) / float(len(resolved))
        if rate < self.thresholds.abandonment_rate_threshold:
            return []
        return [
            MaintenanceCandidate(
                kind=MaintenanceSignal.ABANDONED_WORK,
                subject="repository",
                title=f"{rate:.0%} of completed lifecycles were abandoned or failed",
                detail=(
                    f"{len(abandoned)} of {len(resolved)} terminal lifecycle(s) ended "
                    "without a successful outcome."
                ),
                provenance=PROVENANCE_LIFECYCLE,
                severity=SEVERITY_HIGH if rate >= 0.5 else SEVERITY_MEDIUM,
                confidence=wilson_lower_bound(len(abandoned), len(resolved)),
                sample_size=len(resolved),
                evidence_refs=[
                    str(getattr(record, "lifecycle_id", "")) for record in abandoned
                ][:MAX_EVIDENCE_REFS],
                recommended_action=(
                    "Review recent abandoned lifecycles for a shared blocker before "
                    "queueing further autonomous work."
                ),
                estimated_effort=4.0,
                metrics={"abandonment_rate": rate, "terminal_lifecycles": float(len(resolved))},
            )
        ]

    def _candidate_instability(self, store: Any) -> list[MaintenanceCandidate]:
        """Files whose candidate-stage validation keeps failing.

        NOTE on data availability: candidate-stage iterations are only recorded
        when the orchestrator is wired to emit them. Where no candidate-stage
        iteration exists, this extractor legitimately produces nothing rather
        than falling back to post-apply data, which would measure a different
        thing entirely.
        """
        lifecycles = _lifecycles_of(store)
        if not lifecycles:
            return []
        failures: dict[str, int] = {}
        refs: dict[str, set[str]] = {}
        for record in lifecycles:
            for iteration in getattr(record, "iterations", []) or []:
                if str(getattr(iteration, "validation_stage", "")) != "candidate":
                    continue
                if not getattr(iteration, "failed", False):
                    continue
                signature = getattr(iteration, "defect_signature", None)
                path = sanitize_relative_path(getattr(signature, "affected_file", "") if signature else "")
                if not path:
                    continue
                failures[path] = failures.get(path, 0) + 1
                refs.setdefault(path, set()).add(str(getattr(record, "lifecycle_id", "")))

        results: list[MaintenanceCandidate] = []
        for path, count in sorted(failures.items()):
            if count < self.thresholds.candidate_failure_threshold:
                continue
            results.append(
                MaintenanceCandidate(
                    kind=MaintenanceSignal.CANDIDATE_INSTABILITY,
                    subject=path,
                    title=f"{path} failed candidate validation {count} time(s)",
                    detail=(
                        "Repeated failures at the sandboxed candidate stage suggest "
                        "this area is hard to change correctly."
                    ),
                    provenance=PROVENANCE_LIFECYCLE,
                    severity=SEVERITY_MEDIUM,
                    confidence=wilson_lower_bound(count, count + 1),
                    sample_size=count,
                    evidence_refs=sorted(refs.get(path, ()))[:MAX_EVIDENCE_REFS],
                    affected_files=[path],
                    recommended_action=(
                        "Strengthen local test coverage and clarify this module's "
                        "contract before further automated edits."
                    ),
                    estimated_effort=3.0,
                    metrics={"candidate_failures": float(count)},
                )
            )
        return results

    # -- telemetry-derived signals ----------------------------------------

    def _validation_pressure(self, store: Any) -> list[MaintenanceCandidate]:
        """Validation is broad far more often than not.

        A high broad rate is not itself a defect: broad is always *safe*. It
        is a signal that the impact analysis is not able to justify anything
        narrower, which is a weakness in the dependency model rather than in
        the code under test - hence the recommended action targets the
        analyzer, not the files.
        """
        decisions = _decisions_of(store)
        if len(decisions) < self.thresholds.min_decisions_for_rates:
            return []
        broad = [d for d in decisions if str(getattr(d, "scope", "")) == SCOPE_BROAD]
        rate = len(broad) / float(len(decisions))
        if rate < self.thresholds.broad_scope_rate_threshold:
            return []
        return [
            MaintenanceCandidate(
                kind=MaintenanceSignal.BROAD_VALIDATION_PRESSURE,
                subject="validation_scope",
                title=f"{rate:.0%} of validation decisions escalated to broad scope",
                detail=(
                    f"{len(broad)} of {len(decisions)} recorded decisions could not be "
                    "justified at a narrower scope."
                ),
                provenance=PROVENANCE_TELEMETRY,
                severity=SEVERITY_LOW,
                confidence=wilson_lower_bound(len(broad), len(decisions)),
                sample_size=len(decisions),
                uncertainty=[
                    "broad scope is always safe; this measures analysis strength, "
                    "not correctness"
                ],
                evidence_refs=[str(getattr(d, "decision_id", "")) for d in broad][
                    :MAX_EVIDENCE_REFS
                ],
                recommended_action=(
                    "Improve dependency/impact resolution coverage so more changes "
                    "can be validated at targeted scope. Do not narrow scope manually."
                ),
                estimated_effort=5.0,
                metrics={"broad_rate": rate, "decisions": float(len(decisions))},
            )
        ]

    def _evidence_reuse(self, store: Any) -> list[MaintenanceCandidate]:
        """Evidence reuse is being denied for one dominant reason."""
        decisions = _decisions_of(store)
        if len(decisions) < self.thresholds.min_decisions_for_rates:
            return []
        totals: dict[str, int] = {}
        for decision in decisions:
            reasons = getattr(decision, "reuse_reasons", None)
            if not isinstance(reasons, Mapping):
                continue
            for reason, count in reasons.items():
                try:
                    totals[str(reason)] = totals.get(str(reason), 0) + int(count)
                except (TypeError, ValueError):
                    continue
        overall = sum(totals.values())
        if overall <= 0:
            return []
        results: list[MaintenanceCandidate] = []
        for reason, count in sorted(totals.items()):
            share = count / float(overall)
            if share < self.thresholds.reuse_rejection_rate_threshold:
                continue
            if reason in {"assumptions_still_hold", "no_matching_evidence", "reuse_disabled"}:
                # These are not degradations: the first is a success, the
                # second is the expected state of a fresh repository, and the
                # third is a configuration choice.
                continue
            results.append(
                MaintenanceCandidate(
                    kind=MaintenanceSignal.EVIDENCE_REUSE_FAILURE,
                    subject=f"reuse:{reason}",
                    title=f"Evidence reuse denied mostly for '{reason}' ({share:.0%})",
                    detail=(
                        f"{count} of {overall} reuse denials cite this reason, so cached "
                        "validation evidence is almost never usable."
                    ),
                    provenance=PROVENANCE_TELEMETRY,
                    severity=SEVERITY_LOW,
                    confidence=wilson_lower_bound(count, overall),
                    sample_size=overall,
                    uncertainty=["denial is the safe outcome; this measures cost, not risk"],
                    recommended_action=(
                        "Investigate why evidence assumptions keep breaking; reuse is "
                        "currently paying no dividend."
                    ),
                    estimated_effort=4.0,
                    metrics={"denial_share": share, "denials": float(count)},
                )
            )
        return results

    def _false_confidence(self, store: Any) -> list[MaintenanceCandidate]:
        """A narrow decision was followed by a failure it should have caught.

        This is the single most serious signal the analyzer can raise: it is
        direct evidence that a *safety* decision was wrong, not merely
        expensive. It is therefore CRITICAL from the very first occurrence, and
        - see :mod:`local_agent.maintenance_policy` - is never autonomously
        actionable, because the correct response is human attention.
        """
        decisions = _decisions_of(store)
        if not decisions:
            return []
        incidents = [
            decision
            for decision in decisions
            if str(getattr(decision, "decision_quality", "")) == "false_confidence"
        ]
        if len(incidents) < self.thresholds.false_confidence_minimum:
            return []
        files: set[str] = set()
        for decision in incidents:
            files.update(sanitize_path_list(getattr(decision, "changed_files", []) or []))
        return [
            MaintenanceCandidate(
                kind=MaintenanceSignal.FALSE_CONFIDENCE,
                subject="false_confidence",
                title=f"{len(incidents)} narrow validation decision(s) missed a real failure",
                detail=(
                    "A targeted or expanded validation scope passed and the change "
                    "subsequently failed. The impact model was wrong, not merely tight."
                ),
                provenance=PROVENANCE_TELEMETRY,
                severity=SEVERITY_CRITICAL,
                confidence=STRUCTURAL_OBSERVATION_CONFIDENCE,
                sample_size=len(incidents),
                uncertainty=[],
                evidence_refs=[str(getattr(d, "decision_id", "")) for d in incidents][
                    :MAX_EVIDENCE_REFS
                ],
                affected_files=sorted(files),
                recommended_action=(
                    "Human review required. Determine which dependency edge the impact "
                    "analysis missed before relying on targeted scope in this area again."
                ),
                estimated_effort=8.0,
                metrics={"incidents": float(len(incidents))},
            )
        ]

    def _analysis_degradation(self, store: Any) -> list[MaintenanceCandidate]:
        """Impact analysis is frequently degrading (hitting its own bounds)."""
        decisions = _decisions_of(store)
        if len(decisions) < self.thresholds.min_decisions_for_rates:
            return []
        degraded = [d for d in decisions if bool(getattr(d, "degraded_analysis", False))]
        rate = len(degraded) / float(len(decisions))
        if rate < self.thresholds.degradation_rate_threshold:
            return []
        return [
            MaintenanceCandidate(
                kind=MaintenanceSignal.ANALYSIS_DEGRADATION,
                subject="impact_analysis",
                title=f"Impact analysis degraded on {rate:.0%} of decisions",
                detail=(
                    f"{len(degraded)} of {len(decisions)} decisions were made against a "
                    "truncated or low-confidence impact report."
                ),
                provenance=PROVENANCE_TELEMETRY,
                severity=SEVERITY_MEDIUM,
                confidence=wilson_lower_bound(len(degraded), len(decisions)),
                sample_size=len(decisions),
                evidence_refs=[str(getattr(d, "decision_id", "")) for d in degraded][
                    :MAX_EVIDENCE_REFS
                ],
                recommended_action=(
                    "Raise the impact analyzer's traversal bounds or reduce module "
                    "coupling so analysis stops truncating."
                ),
                estimated_effort=5.0,
                metrics={"degradation_rate": rate, "decisions": float(len(decisions))},
            )
        ]

    # -- graph-derived signals --------------------------------------------

    def _architectural_risk(
        self, graph: Any, churn: Mapping[str, int]
    ) -> list[MaintenanceCandidate]:
        """High fan-in *and* frequently changed.

        Requiring both is the point. A widely-imported module that nobody
        touches is stable infrastructure, and a churning leaf module is
        someone's active work; only the intersection is architecturally risky.
        Firing on fan-in alone would flag every utility module in the
        repository, which is the definition of inventing a problem because a
        heuristic fired.
        """
        reverse = getattr(graph, "reverse_deps", None)
        if not isinstance(reverse, Mapping) or not reverse:
            return []
        results: list[MaintenanceCandidate] = []
        for path in sorted(reverse):
            relative = sanitize_relative_path(path)
            if not relative or looks_like_test_path(relative):
                continue
            fan_in = len(reverse.get(path) or ())
            commits = int(churn.get(relative, 0))
            if fan_in < self.thresholds.min_fan_in_for_risk:
                continue
            if commits < self.thresholds.min_churn_for_risk:
                continue
            severity = SEVERITY_HIGH if fan_in >= self.thresholds.min_fan_in_for_risk * 2 else SEVERITY_MEDIUM
            results.append(
                MaintenanceCandidate(
                    kind=MaintenanceSignal.ARCHITECTURAL_RISK,
                    subject=relative,
                    title=f"{relative}: fan-in {fan_in}, {commits} recent commit(s)",
                    detail=(
                        "Frequently modified module with many reverse dependencies; "
                        "changes here have a wide blast radius."
                    ),
                    provenance=PROVENANCE_SEMANTIC_GRAPH,
                    severity=severity,
                    confidence=STRUCTURAL_OBSERVATION_CONFIDENCE,
                    sample_size=1,
                    uncertainty=[
                        "structural observation; risk is inferred from coupling and "
                        "churn, not from an observed defect"
                    ],
                    affected_files=[relative],
                    recommended_action=(
                        "Consider narrowing this module's public surface or adding "
                        "characterisation tests for its dependents."
                    ),
                    estimated_effort=6.0,
                    metrics={"fan_in": float(fan_in), "recent_commits": float(commits)},
                )
            )
        return results

    def _analyzer_blind_spots(self, graph: Any) -> list[MaintenanceCandidate]:
        """Files with many imports the dependency resolver could not resolve."""
        unresolved = getattr(graph, "unresolved_imports", None)
        if not isinstance(unresolved, Mapping) or not unresolved:
            return []
        results: list[MaintenanceCandidate] = []
        for path in sorted(unresolved):
            relative = sanitize_relative_path(path)
            if not relative:
                continue
            names = sorted(str(name) for name in (unresolved.get(path) or ()))
            # Third-party imports are *expected* to be unresolved: the graph
            # only contains this repository. Only count dotted names that look
            # like they should have resolved locally.
            local_looking = [name for name in names if _looks_local(name, graph)]
            if len(local_looking) < self.thresholds.min_unresolved_imports:
                continue
            results.append(
                MaintenanceCandidate(
                    kind=MaintenanceSignal.ANALYZER_BLIND_SPOT,
                    subject=relative,
                    title=f"{relative} has {len(local_looking)} unresolved local import(s)",
                    detail="Unresolved: " + ", ".join(local_looking[:10]),
                    provenance=PROVENANCE_SEMANTIC_GRAPH,
                    severity=SEVERITY_LOW,
                    confidence=STRUCTURAL_OBSERVATION_CONFIDENCE,
                    sample_size=1,
                    affected_files=[relative],
                    recommended_action=(
                        "These edges are invisible to impact analysis; either simplify "
                        "the imports or extend the resolver to cover them."
                    ),
                    estimated_effort=3.0,
                    metrics={"unresolved_imports": float(len(local_looking))},
                )
            )
        return results

    def _parse_failures(self, graph: Any) -> list[MaintenanceCandidate]:
        """Files the indexer could not parse at all."""
        failures = getattr(graph, "parse_failures", None)
        if not isinstance(failures, Mapping) or not failures:
            return []
        results: list[MaintenanceCandidate] = []
        for path in sorted(failures):
            relative = sanitize_relative_path(path)
            if not relative:
                continue
            results.append(
                MaintenanceCandidate(
                    kind=MaintenanceSignal.PARSE_FAILURE,
                    subject=relative,
                    title=f"{relative} could not be parsed",
                    detail=sanitize_text(failures.get(path)),
                    provenance=PROVENANCE_SEMANTIC_GRAPH,
                    severity=SEVERITY_HIGH,
                    confidence=STRUCTURAL_OBSERVATION_CONFIDENCE,
                    sample_size=1,
                    affected_files=[relative],
                    recommended_action=(
                        "This file is invisible to every semantic analysis in the agent. "
                        "Fix the syntax error or exclude the file deliberately."
                    ),
                    estimated_effort=1.0,
                    metrics={"parse_failures": 1.0},
                )
            )
        return results

    def _test_gaps(self, graph: Any) -> list[MaintenanceCandidate]:
        """Non-trivial modules that no test file imports.

        "No test imports it" is a weaker claim than "it is untested" - a test
        could exercise it indirectly - so this is reported at MEDIUM at most
        and always carries that caveat in ``uncertainty``.
        """
        reverse = getattr(graph, "reverse_deps", None)
        files = getattr(graph, "files", None)
        if not isinstance(reverse, Mapping) or not isinstance(files, Mapping):
            return []
        results: list[MaintenanceCandidate] = []
        for path in sorted(files):
            relative = sanitize_relative_path(path)
            if not relative or looks_like_test_path(relative):
                continue
            dependents = reverse.get(path) or ()
            if len(dependents) < self.thresholds.test_gap_min_fan_in:
                continue
            if any(looks_like_test_path(str(dep)) for dep in dependents):
                continue
            results.append(
                MaintenanceCandidate(
                    kind=MaintenanceSignal.TEST_GAP,
                    subject=relative,
                    title=f"{relative} is imported by {len(dependents)} module(s) but no test",
                    detail=(
                        "No test file in this repository imports this module directly."
                    ),
                    provenance=PROVENANCE_SEMANTIC_GRAPH,
                    severity=SEVERITY_MEDIUM,
                    confidence=STRUCTURAL_OBSERVATION_CONFIDENCE,
                    sample_size=1,
                    uncertainty=[
                        "direct-import evidence only; indirect coverage is not measured"
                    ],
                    affected_files=[relative],
                    recommended_action=(
                        "Add direct tests for this module's public interface."
                    ),
                    estimated_effort=4.0,
                    metrics={"dependents": float(len(dependents))},
                )
            )
        return results

    # -- knowledge-derived signals ----------------------------------------

    def _failure_patterns(self, graph: Any) -> list[MaintenanceCandidate]:
        """Recurring failure patterns the knowledge graph already recorded."""
        patterns = getattr(graph, "failure_patterns", None)
        if not isinstance(patterns, (list, tuple)) or not patterns:
            return []
        results: list[MaintenanceCandidate] = []
        for pattern in patterns:
            occurrences = int(getattr(pattern, "occurrence_count", 0) or 0)
            if occurrences < self.thresholds.min_failure_pattern_occurrences:
                continue
            signature = sanitize_text(getattr(pattern, "error_signature", ""), limit=200)
            if not signature:
                continue
            results.append(
                MaintenanceCandidate(
                    kind=MaintenanceSignal.KNOWN_FAILURE_PATTERN,
                    subject=signature,
                    title=f"Known failure pattern seen {occurrences} time(s)",
                    detail=sanitize_text(getattr(pattern, "root_cause_summary", "")) or signature,
                    provenance=PROVENANCE_KNOWLEDGE_GRAPH,
                    severity=SEVERITY_MEDIUM if occurrences >= 5 else SEVERITY_LOW,
                    # The knowledge graph stores its own confidence; it is
                    # clamped rather than trusted, and the sample size travels
                    # with it so a 0.99 backed by three observations reads as
                    # what it is.
                    confidence=min(
                        float(getattr(pattern, "confidence", 0.5) or 0.5),
                        wilson_lower_bound(occurrences, occurrences + 1),
                    ),
                    sample_size=occurrences,
                    evidence_refs=[sanitize_text(getattr(pattern, "pattern_id", ""), limit=64)],
                    affected_files=sanitize_path_list(
                        getattr(pattern, "affected_files", []) or []
                    ),
                    recommended_action=(
                        sanitize_text(getattr(pattern, "successful_repair_summary", ""))
                        or "Apply the recorded repair recipe or remove the underlying cause."
                    ),
                    estimated_effort=3.0,
                    metrics={"occurrences": float(occurrences)},
                )
            )
        return results


# -- helpers ------------------------------------------------------------------


def _lifecycles_of(store: Any) -> list[Any]:
    if store is None:
        return []
    records = getattr(store, "lifecycles", None)
    if callable(records):
        records = records()
    return list(records) if isinstance(records, (list, tuple)) else []


def _decisions_of(store: Any) -> list[Any]:
    if store is None:
        return []
    records = getattr(store, "decisions", None)
    if callable(records):
        records = records()
    return list(records) if isinstance(records, (list, tuple)) else []


def _lifecycle_files(record: Any) -> list[str]:
    """Every repository-relative file a lifecycle's defects point at."""
    paths: set[str] = set()
    for iteration in getattr(record, "iterations", []) or []:
        signature = getattr(iteration, "defect_signature", None)
        if signature is None:
            continue
        path = sanitize_relative_path(getattr(signature, "affected_file", ""))
        if path:
            paths.add(path)
    return sorted(paths)


def _describe_signature(signature: Any) -> str:
    describe = getattr(signature, "describe", None)
    if callable(describe):
        try:
            return sanitize_text(describe())
        except Exception:
            return ""
    return ""


def _looks_local(name: str, graph: Any) -> bool:
    """Heuristic: does this unresolved dotted name look like it should be local?

    A relative import (leading dot) always should. An absolute one is judged by
    whether its first segment matches a top-level package that actually exists
    in the graph's module map - so ``requests`` is correctly ignored while
    ``local_agent.something_removed`` is correctly flagged.
    """
    if not isinstance(name, str) or not name:
        return False
    if name.startswith("."):
        return True
    head = name.split(".", 1)[0]
    modules = getattr(graph, "module_to_file", None)
    if not isinstance(modules, Mapping):
        return False
    return any(
        module == head or str(module).startswith(head + ".") for module in modules
    )


def _normalize_churn(churn: Mapping[str, int] | None) -> dict[str, int]:
    if not isinstance(churn, Mapping):
        return {}
    result: dict[str, int] = {}
    for path, count in churn.items():
        relative = sanitize_relative_path(path)
        if not relative:
            continue
        try:
            result[relative] = max(result.get(relative, 0), int(count))
        except (TypeError, ValueError):
            continue
    return result


def collect_churn(git: Any, *, window: int = 200) -> dict[str, int]:
    """Read recent per-file commit counts, tolerating any git failure.

    Churn is enrichment: without it the architectural-risk extractor simply
    finds nothing, which is the correct degradation for a repository whose
    history is unavailable (a fresh clone with ``--depth 1``, a non-git
    directory, or git missing from ``PATH``).
    """
    reader = getattr(git, "file_change_counts", None)
    if not callable(reader):
        return {}
    try:
        counts = reader(limit=window)
    except Exception:
        return {}
    return _normalize_churn(counts if isinstance(counts, Mapping) else {})


def candidates_for_files(
    candidates: Sequence[MaintenanceCandidate], paths: Iterable[str]
) -> list[MaintenanceCandidate]:
    """Every candidate touching any of ``paths``. Used by overlap detection."""
    wanted = {sanitize_relative_path(path) for path in paths}
    wanted.discard("")
    if not wanted:
        return []
    return [
        candidate
        for candidate in candidates
        if wanted.intersection(candidate.affected_files)
    ]


def signal_fingerprint(candidate: MaintenanceCandidate) -> str:
    """A stable digest of a candidate's *observable magnitude*.

    Reassessment compares this before and after maintenance work. It folds in
    the metrics and severity but deliberately *not* the state, outcome or
    occurrence count - those change as a result of the maintenance process
    itself, and including them would make every candidate look changed simply
    because it was worked on.
    """
    parts = [candidate.kind, candidate.subject, candidate.severity]
    for name in sorted(candidate.metrics):
        parts.append(f"{name}={candidate.metrics[name]:.6f}")
    parts.extend(candidate.affected_files)
    return candidate_identity("signal", parts)
