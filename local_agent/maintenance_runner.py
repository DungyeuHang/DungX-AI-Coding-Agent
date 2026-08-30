"""Phase 4.21: the maintenance run lifecycle.

``DISCOVER -> SCORE -> SELECT -> PLAN -> EXECUTE -> VALIDATE -> REVIEW ->
INTEGRATE -> LEARN -> REASSESS``

The runner owns the *sequencing* of maintenance work and nothing else. Every
stage that could touch the repository is delegated:

* planning, implementation, sandboxing, validation and approval are performed
  by the existing pipeline, reached through a single injected ``executor``
  seam that takes a :class:`MaintenanceWorkOrder` and returns a
  :class:`MaintenanceExecutionOutcome`;
* when no executor is injected - the default, and the only configuration this
  build ships with enabled - the runner cannot execute anything. It plans,
  reports, and (optionally) enqueues an ordinary ``Task`` for the existing
  scheduler to pick up exactly as if a human had created it.

That seam is the whole integration story. There is no second implementation
agent, no second validation runner, no second approval gate and no second
worktree manager in this module, and the tests assert that structurally.

**Reassessment is the part worth reading carefully.** A maintenance candidate
is only ever marked ``RESOLVED`` when a *fresh scan* of the repository no
longer produces the signal. "The task succeeded" is not evidence of anything
except that the task succeeded; a change that validates cleanly and leaves the
original problem exactly where it was is a very common outcome, and this
module is built so that outcome is reported as ``PERSISTING`` rather than
quietly booked as a win.
"""

from __future__ import annotations

import datetime
import threading
import time
import uuid
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .maintenance import (
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_MAX_RUNS,
    RUN_MODE_DRY_RUN,
    RUN_MODE_EXECUTE,
    RUN_MODE_SCAN,
    RUN_STATUS_CANCELLED,
    RUN_STATUS_COMPLETED,
    RUN_STATUS_FAILED,
    RUN_STATUS_PARTIAL,
    BudgetLedger,
    CandidateRunOutcome,
    CandidateState,
    InvalidCandidateTransition,
    MaintenanceBudget,
    MaintenanceCandidate,
    MaintenanceRunRecord,
    MaintenanceStore,
    ReassessmentOutcome,
    sanitize_path_list,
    sanitize_string_list,
    sanitize_text,
    severity_rank,
)
from .maintenance_analysis import (
    AnalysisResult,
    MaintenanceAnalyzer,
    MaintenanceThresholds,
    signal_fingerprint,
)
from .maintenance_policy import (
    EXECUTING_TIERS,
    AutonomyTier,
    MaintenanceExecutionPolicy,
    MaintenancePriorityEngine,
    PolicyThresholds,
    PolicyVerdict,
    PriorityExplanation,
    TIER_ORDER,
)

STAGE_DISCOVER = "discover"
STAGE_SCORE = "score"
STAGE_SELECT = "select"
STAGE_PLAN = "plan"
STAGE_EXECUTE = "execute"
STAGE_VALIDATE = "validate"
STAGE_REVIEW = "review"
STAGE_INTEGRATE = "integrate"
STAGE_LEARN = "learn"
STAGE_REASSESS = "reassess"

ALL_STAGES: tuple[str, ...] = (
    STAGE_DISCOVER,
    STAGE_SCORE,
    STAGE_SELECT,
    STAGE_PLAN,
    STAGE_EXECUTE,
    STAGE_VALIDATE,
    STAGE_REVIEW,
    STAGE_INTEGRATE,
    STAGE_LEARN,
    STAGE_REASSESS,
)


class MaintenanceCancelled(RuntimeError):
    """Raised internally when a cancellation callback asks the run to stop."""


# -- work orders --------------------------------------------------------------


@dataclass
class MaintenanceWorkOrder:
    """The bounded unit of work derived from one candidate.

    Deliberately *not* a ``Task``. Building a ``Task`` here would mean this
    module deciding what the plan looks like, which is the planner's job. A
    work order is a request - objective, scope, acceptance criteria, budget -
    that the existing pipeline turns into a plan, a DAG and subtasks by its own
    rules.
    """

    candidate_id: str = ""
    objective: str = ""
    rationale: str = ""
    scope_files: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    granted_tier: str = AutonomyTier.PLAN_ONLY
    max_subtasks: int = 1
    max_changed_files: int = 1
    max_tool_steps: int = 1
    max_validation_commands: int = 1
    max_repair_iterations: int = 0

    def __post_init__(self) -> None:
        self.candidate_id = sanitize_text(self.candidate_id, limit=64)
        self.objective = sanitize_text(self.objective, limit=400)
        self.rationale = sanitize_text(self.rationale, limit=400)
        self.scope_files = sanitize_path_list(self.scope_files)
        self.acceptance_criteria = sanitize_string_list(self.acceptance_criteria)
        for name in (
            "max_subtasks",
            "max_changed_files",
            "max_tool_steps",
            "max_validation_commands",
            "max_repair_iterations",
        ):
            try:
                setattr(self, name, max(0, int(getattr(self, name))))
            except (TypeError, ValueError):
                setattr(self, name, 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "objective": self.objective,
            "rationale": self.rationale,
            "scope_files": list(self.scope_files),
            "acceptance_criteria": list(self.acceptance_criteria),
            "granted_tier": self.granted_tier,
            "max_subtasks": self.max_subtasks,
            "max_changed_files": self.max_changed_files,
            "max_tool_steps": self.max_tool_steps,
            "max_validation_commands": self.max_validation_commands,
            "max_repair_iterations": self.max_repair_iterations,
        }


@dataclass
class MaintenanceExecutionOutcome:
    """What the existing pipeline reported back about one work order."""

    succeeded: bool = False
    validation_passed: bool | None = None
    changed_files: list[str] = field(default_factory=list)
    task_id: str = ""
    error: str = ""
    notes: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def __post_init__(self) -> None:
        self.changed_files = sanitize_path_list(self.changed_files)
        self.task_id = sanitize_text(self.task_id, limit=64)
        self.error = sanitize_text(self.error)
        self.notes = sanitize_string_list(self.notes)
        try:
            self.elapsed_seconds = max(0.0, float(self.elapsed_seconds))
        except (TypeError, ValueError):
            self.elapsed_seconds = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "succeeded": self.succeeded,
            "validation_passed": self.validation_passed,
            "changed_files": list(self.changed_files),
            "task_id": self.task_id,
            "error": self.error,
            "notes": list(self.notes),
            "elapsed_seconds": self.elapsed_seconds,
        }


ExecutorFn = Callable[[MaintenanceWorkOrder], MaintenanceExecutionOutcome]
ScanFn = Callable[[], AnalysisResult]


def build_work_order(
    candidate: MaintenanceCandidate,
    *,
    granted_tier: str,
    budget: MaintenanceBudget,
) -> MaintenanceWorkOrder:
    """Derive a bounded work order from a candidate.

    The scope is the candidate's affected files and nothing else. That is the
    contract the review stage later checks against: any file the execution
    touched that is not in ``scope_files`` is an out-of-scope change, and the
    candidate is failed for it even if validation passed.
    """
    return MaintenanceWorkOrder(
        candidate_id=candidate.candidate_id,
        objective=f"Maintenance: {candidate.title}",
        rationale=candidate.recommended_action or candidate.detail,
        scope_files=list(candidate.affected_files),
        acceptance_criteria=[
            f"The '{candidate.kind}' maintenance signal for "
            f"{candidate.subject or 'this repository'} is no longer detected.",
            "No file outside the declared scope is modified.",
            "All existing validation continues to pass.",
        ],
        granted_tier=granted_tier,
        max_subtasks=budget.max_subtasks_per_candidate,
        max_changed_files=budget.max_changed_files_per_candidate,
        max_tool_steps=budget.max_tool_steps_per_subtask,
        max_validation_commands=budget.max_validation_commands,
        max_repair_iterations=budget.max_repair_iterations_per_candidate,
    )


# -- concurrency planning -----------------------------------------------------


def plan_execution_batches(
    candidates: Sequence[MaintenanceCandidate], *, max_width: int
) -> list[list[MaintenanceCandidate]]:
    """Group candidates into batches that may safely run in parallel.

    Two candidates land in different batches when their affected-file sets
    intersect. That is the only safe rule available here: the existing
    worktree/DAG machinery isolates *filesystem* state per worker, but two
    concurrent changes to the same file still merge into the same branch
    afterwards, and letting them race would make the merge order decide the
    result.

    Candidates are consumed in the order given (already priority order), and
    each is placed in the first batch that does not conflict, so the output is
    deterministic for a given input. ``max_width`` caps every batch, which is
    how the Phase 4.14 worker limit is honoured without this module knowing
    anything about workers.
    """
    max_width = max(1, int(max_width))
    batches: list[list[MaintenanceCandidate]] = []
    batch_files: list[set[str]] = []
    for candidate in candidates:
        files = set(candidate.affected_files)
        placed = False
        for index, existing in enumerate(batch_files):
            if len(batches[index]) >= max_width:
                continue
            if files and existing.intersection(files):
                continue
            batches[index].append(candidate)
            existing.update(files)
            placed = True
            break
        if not placed:
            batches.append([candidate])
            batch_files.append(set(files))
    return batches


def overlapping_candidates(
    candidates: Sequence[MaintenanceCandidate],
) -> list[tuple[str, str]]:
    """Every unordered pair of candidates sharing at least one file."""
    pairs: list[tuple[str, str]] = []
    ordered = sorted(candidates, key=lambda c: c.candidate_id)
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            if set(left.affected_files).intersection(right.affected_files):
                pairs.append((left.candidate_id, right.candidate_id))
    return pairs


# -- reassessment -------------------------------------------------------------


@dataclass
class ReassessmentVerdict:
    """The evidence-backed answer to "did the maintenance actually work?"."""

    candidate_id: str = ""
    outcome: str = ReassessmentOutcome.INCONCLUSIVE
    before_fingerprint: str = ""
    after_fingerprint: str = ""
    before_severity: str = ""
    after_severity: str = ""
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "outcome": self.outcome,
            "before_fingerprint": self.before_fingerprint,
            "after_fingerprint": self.after_fingerprint,
            "before_severity": self.before_severity,
            "after_severity": self.after_severity,
            "reasons": list(self.reasons),
        }


def reassess(
    before: MaintenanceCandidate,
    after: MaintenanceCandidate | None,
    *,
    executed: bool,
    validation_passed: bool | None,
    rescan_degraded: bool = False,
) -> ReassessmentVerdict:
    """Compare a candidate before and after maintenance work.

    The rules, in the order they are applied, with the reasoning that makes
    each one non-negotiable:

    1. **Nothing was executed** -> ``INCONCLUSIVE``. Even if the signal has
       vanished. A signal can disappear because someone else changed the file,
       because a bound was hit differently, or because the analyzer degraded;
       attributing that to maintenance we did not perform would be inventing a
       success.
    2. **The rescan was degraded** -> ``INCONCLUSIVE``. A scan that could not
       see everything cannot testify that something is gone.
    3. **Validation failed** -> ``PERSISTING`` at best. A change that does not
       validate has not fixed anything, whatever the signal now says.
    4. **Signal absent** -> ``RESOLVED``. This is the only path to RESOLVED,
       and it requires 1-3 to have passed first.
    5. **Severity increased, or the fingerprint changed for the worse** ->
       ``REGRESSED``.
    6. **Severity decreased** -> ``PARTIALLY_RESOLVED``.
    7. **Otherwise** -> ``PERSISTING``.
    """
    verdict = ReassessmentVerdict(
        candidate_id=before.candidate_id,
        before_fingerprint=signal_fingerprint(before),
        before_severity=before.severity,
        after_severity=after.severity if after is not None else "",
        after_fingerprint=signal_fingerprint(after) if after is not None else "",
    )
    if not executed:
        verdict.outcome = ReassessmentOutcome.INCONCLUSIVE
        verdict.reasons.append(
            "no maintenance work was executed for this candidate, so any change in "
            "the signal cannot be attributed to it"
        )
        return verdict
    if rescan_degraded:
        verdict.outcome = ReassessmentOutcome.INCONCLUSIVE
        verdict.reasons.append(
            "the reassessment scan was degraded; absence of the signal is not "
            "evidence of its resolution"
        )
        return verdict
    if validation_passed is False:
        verdict.outcome = ReassessmentOutcome.PERSISTING
        verdict.reasons.append(
            "validation failed, so no fix can be credited regardless of the signal"
        )
        return verdict
    if after is None:
        verdict.outcome = ReassessmentOutcome.RESOLVED
        verdict.reasons.append(
            "a fresh scan no longer produces this signal, and the work that "
            "preceded the rescan validated cleanly"
        )
        return verdict

    before_rank = severity_rank(before.severity)
    after_rank = severity_rank(after.severity)
    if after_rank > before_rank:
        verdict.outcome = ReassessmentOutcome.REGRESSED
        verdict.reasons.append(
            f"severity rose from {before.severity} to {after.severity}"
        )
        return verdict
    if after_rank < before_rank:
        verdict.outcome = ReassessmentOutcome.PARTIALLY_RESOLVED
        verdict.reasons.append(
            f"severity fell from {before.severity} to {after.severity} but the "
            "signal is still present"
        )
        return verdict
    if _metrics_improved(before, after):
        verdict.outcome = ReassessmentOutcome.PARTIALLY_RESOLVED
        verdict.reasons.append(
            "the signal's measured magnitude decreased but it is still detected"
        )
        return verdict
    if _metrics_worsened(before, after):
        verdict.outcome = ReassessmentOutcome.REGRESSED
        verdict.reasons.append("the signal's measured magnitude increased")
        return verdict
    verdict.outcome = ReassessmentOutcome.PERSISTING
    verdict.reasons.append(
        "the signal is unchanged; the work completed but did not address it"
    )
    return verdict


def _metrics_improved(before: MaintenanceCandidate, after: MaintenanceCandidate) -> bool:
    shared = set(before.metrics).intersection(after.metrics)
    if not shared:
        return False
    return all(after.metrics[name] <= before.metrics[name] for name in shared) and any(
        after.metrics[name] < before.metrics[name] for name in shared
    )


def _metrics_worsened(before: MaintenanceCandidate, after: MaintenanceCandidate) -> bool:
    shared = set(before.metrics).intersection(after.metrics)
    if not shared:
        return False
    return any(after.metrics[name] > before.metrics[name] for name in shared)


# -- learning -----------------------------------------------------------------


@dataclass
class SignalActionability:
    """How often acting on a signal kind actually resolved it.

    ADVISORY ONLY. Nothing reads this back to change a threshold, a scope, a
    safety floor or a policy decision. It exists so an operator can see which
    signals are worth their attention, and so a future phase has a real
    measurement to start from instead of an assumption.
    """

    kind: str = ""
    attempts: int = 0
    resolved: int = 0
    persisting: int = 0
    regressed: int = 0
    inconclusive: int = 0

    @property
    def resolution_rate(self) -> float:
        return self.resolved / float(self.attempts) if self.attempts else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "attempts": self.attempts,
            "resolved": self.resolved,
            "persisting": self.persisting,
            "regressed": self.regressed,
            "inconclusive": self.inconclusive,
            "resolution_rate": self.resolution_rate,
        }


def compute_actionability(
    store: MaintenanceStore, *, min_samples: int = 5
) -> dict[str, Any]:
    """Aggregate historical outcomes per signal kind.

    Reports ``data_sufficient`` alongside the rates so that a 100% resolution
    rate over two attempts is legible as what it is. Part 17 case 13 exists
    precisely because such a number is otherwise indistinguishable from a
    well-earned one.
    """
    by_kind: dict[str, SignalActionability] = {}
    total = 0
    for record in store.runs:
        for entry in record.outcomes:
            if not entry.executed:
                continue
            stats = by_kind.setdefault(entry.kind, SignalActionability(kind=entry.kind))
            stats.attempts += 1
            total += 1
            if entry.outcome == ReassessmentOutcome.RESOLVED:
                stats.resolved += 1
            elif entry.outcome == ReassessmentOutcome.PERSISTING:
                stats.persisting += 1
            elif entry.outcome == ReassessmentOutcome.REGRESSED:
                stats.regressed += 1
            else:
                stats.inconclusive += 1
    return {
        "advisory_only": True,
        "min_samples": int(min_samples),
        "total_attempts": total,
        "data_sufficient": total >= int(min_samples),
        "history_trustworthy": store.history_trustworthy(),
        "by_kind": {
            kind: by_kind[kind].to_dict() for kind in sorted(by_kind)
        },
    }


# -- persistence manager ------------------------------------------------------

_LOCKS: "weakref.WeakValueDictionary[str, threading.Lock]" = weakref.WeakValueDictionary()
_LOCKS_GUARD = threading.Lock()
_LOCK_ANCHORS: dict[str, threading.Lock] = {}


def _lock_for(key: str) -> threading.Lock:
    """One process-wide lock per storage key.

    Same pattern and the same documented boundary as Phase 4.19/4.20: this
    serialises *threads*, not processes. Two agent processes sharing one data
    directory can still interleave a read-modify-write; the store's atomic
    replace means the file is never torn, but one process's run record can be
    lost. Two concurrent maintenance runs against one data directory is not a
    supported configuration, and the CLI says so.
    """
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _LOCKS[key] = lock
            _LOCK_ANCHORS[key] = lock
        return lock


class MaintenanceManager:
    """Load/save wrapper around :class:`MaintenanceStore`.

    Mirrors :class:`~local_agent.validation_lifecycle.ValidationLifecycleManager`
    rather than inventing a new persistence idiom, and tolerates a storage
    backend that predates this phase: a backend without the maintenance methods
    simply retains nothing, and every operation still succeeds.
    """

    def __init__(
        self,
        storage: Any,
        project_root: str | Path,
        *,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        max_runs: int = DEFAULT_MAX_RUNS,
    ):
        self.storage = storage
        self.project_root = Path(project_root)
        self.max_candidates = max_candidates
        self.max_runs = max_runs
        self._key = str(Path(project_root).resolve())

    def load(self) -> MaintenanceStore:
        loader = getattr(self.storage, "load_maintenance", None)
        if not callable(loader):
            return MaintenanceStore(
                max_candidates=self.max_candidates, max_runs=self.max_runs
            )
        try:
            store = loader()
        except Exception:
            store = MaintenanceStore(
                max_candidates=self.max_candidates, max_runs=self.max_runs
            )
            store.corrupted_records_skipped = 1
            return store
        if not isinstance(store, MaintenanceStore):
            store = MaintenanceStore(
                max_candidates=self.max_candidates, max_runs=self.max_runs
            )
            store.corrupted_records_skipped = 1
        store.max_candidates = self.max_candidates
        store.max_runs = self.max_runs
        return store

    def save(self, store: MaintenanceStore) -> bool:
        """Persist, returning whether it worked.

        A persistence failure never propagates. Maintenance history is
        valuable but it is not the work: losing a run record must not fail a
        run that otherwise completed correctly.
        """
        saver = getattr(self.storage, "save_maintenance", None)
        if not callable(saver):
            return False
        try:
            saver(store)
        except Exception:
            return False
        return True

    def mutate(self, mutation: Callable[[MaintenanceStore], Any]) -> Any:
        """Run ``mutation`` against the store under the storage lock."""
        with _lock_for(self._key):
            store = self.load()
            result = mutation(store)
            self.save(store)
            return result


# -- the runner ---------------------------------------------------------------


@dataclass
class MaintenanceRunResult:
    """Everything one run produced, for the CLI and for the tests."""

    record: MaintenanceRunRecord
    analysis: AnalysisResult
    ranked: list[tuple[MaintenanceCandidate, PriorityExplanation]] = field(
        default_factory=list
    )
    verdicts: dict[str, PolicyVerdict] = field(default_factory=dict)
    work_orders: dict[str, MaintenanceWorkOrder] = field(default_factory=dict)
    reassessments: dict[str, ReassessmentVerdict] = field(default_factory=dict)
    batches: list[list[str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run": self.record.to_dict(),
            "analysis": self.analysis.to_dict(),
            "ranking": [
                {"candidate": candidate.to_dict(), "priority": explanation.to_dict()}
                for candidate, explanation in self.ranked
            ],
            "policy": {key: value.to_dict() for key, value in sorted(self.verdicts.items())},
            "work_orders": {
                key: value.to_dict() for key, value in sorted(self.work_orders.items())
            },
            "reassessments": {
                key: value.to_dict() for key, value in sorted(self.reassessments.items())
            },
            "batches": [list(batch) for batch in self.batches],
        }


class MaintenanceRunner:
    """Drives one maintenance run through the full lifecycle.

    Every dependency is injected. In particular ``executor`` is the *only* way
    anything in this class can cause a repository modification, and it defaults
    to ``None``, which is what makes "maintenance is off by default" a
    structural fact rather than a configuration convention.
    """

    def __init__(
        self,
        *,
        analyzer: MaintenanceAnalyzer,
        manager: MaintenanceManager,
        scan: ScanFn,
        budget: MaintenanceBudget | None = None,
        policy: MaintenanceExecutionPolicy | None = None,
        priority_engine: MaintenancePriorityEngine | None = None,
        executor: ExecutorFn | None = None,
        configured_tier: str = AutonomyTier.OBSERVE_ONLY,
        cancelled: Callable[[], bool] | None = None,
        progress: Callable[[str], None] | None = None,
    ):
        self.analyzer = analyzer
        self.manager = manager
        self.scan = scan
        self.budget = budget or MaintenanceBudget()
        self.budget.validate()
        self.policy = policy or MaintenanceExecutionPolicy()
        self.priority_engine = priority_engine or MaintenancePriorityEngine()
        self.executor = executor
        # An unrecognised tier degrades to the safest one rather than raising:
        # a typo in a configuration file must not be a way to get *more*
        # autonomy, and it must not stop an operator scanning either.
        self.configured_tier = (
            str(configured_tier)
            if str(configured_tier) in TIER_ORDER
            else AutonomyTier.OBSERVE_ONLY
        )
        self.cancelled = cancelled
        self.progress = progress
        #: Candidates that survived SELECT, handed from that stage to PLAN.
        self._selected: list[MaintenanceCandidate] = []

    # -- public API --------------------------------------------------------

    def run(self, *, mode: str = RUN_MODE_SCAN) -> MaintenanceRunResult:
        """Execute the lifecycle in ``mode``.

        ``scan`` discovers and reports. ``dry_run`` additionally scores,
        selects and plans, but the execute stage is structurally skipped.
        ``execute`` is the only mode that may call the executor, and even then
        only for candidates the policy granted an executing tier.
        """
        mode = mode if mode in (RUN_MODE_SCAN, RUN_MODE_DRY_RUN, RUN_MODE_EXECUTE) else RUN_MODE_SCAN
        started = time.perf_counter()
        ledger = BudgetLedger(self.budget)
        record = MaintenanceRunRecord(
            run_id=uuid.uuid4().hex[:16],
            mode=mode,
            configured_tier=self.configured_tier,
        )

        analysis = self._discover(record, ledger)
        result = MaintenanceRunResult(record=record, analysis=analysis)

        try:
            self._score_and_select(result, ledger, mode)
            if mode != RUN_MODE_SCAN:
                self._plan_and_execute(result, ledger, mode)
        except MaintenanceCancelled:
            record.status = RUN_STATUS_CANCELLED
            record.notes.append("run cancelled by request")
        except Exception as exc:  # pragma: no cover - defensive
            record.status = RUN_STATUS_FAILED
            record.errors.append(sanitize_text(f"{type(exc).__name__}: {exc}"))

        elapsed = time.perf_counter() - started
        ledger.observe("max_elapsed_seconds", elapsed)
        record.elapsed_seconds = elapsed
        record.finished_at = _timestamp()
        record.budget = ledger.to_dict()
        if record.status not in (RUN_STATUS_CANCELLED, RUN_STATUS_FAILED):
            record.status = (
                RUN_STATUS_PARTIAL
                if record.executions_failed or analysis.extractor_errors
                else RUN_STATUS_COMPLETED
            )
        self._persist(result)
        return result

    # -- stages ------------------------------------------------------------

    def _discover(self, record: MaintenanceRunRecord, ledger: BudgetLedger) -> AnalysisResult:
        self._emit(f"[{STAGE_DISCOVER}] scanning repository intelligence")
        try:
            analysis = self.scan()
        except Exception as exc:
            analysis = AnalysisResult()
            analysis.extractor_errors["scan"] = sanitize_text(
                f"{type(exc).__name__}: {exc}"
            )
        considered: list[MaintenanceCandidate] = []
        for candidate in analysis.candidates:
            if not ledger.try_consume("max_candidates_considered"):
                record.notes.append(
                    "candidate consideration budget exhausted; remaining signals were "
                    "discovered but not triaged"
                )
                break
            considered.append(candidate)
        analysis.candidates = considered
        record.candidates_discovered = len(considered)
        for name, error in sorted(analysis.extractor_errors.items()):
            record.errors.append(f"{name}: {error}")
        return analysis

    def _score_and_select(
        self, result: MaintenanceRunResult, ledger: BudgetLedger, mode: str
    ) -> None:
        self._check_cancelled()
        self._emit(f"[{STAGE_SCORE}] ranking {len(result.analysis.candidates)} candidate(s)")
        result.ranked = self.priority_engine.rank(result.analysis.candidates)

        self._emit(f"[{STAGE_SELECT}] applying execution policy and budgets")
        selected: list[MaintenanceCandidate] = []
        for candidate, explanation in result.ranked:
            verdict = self.policy.decide(
                candidate,
                configured_tier=self.configured_tier,
                budget=self.budget,
            )
            result.verdicts[candidate.candidate_id] = verdict
            candidate.blocked_reasons = list(verdict.blocking_reasons or verdict.cap_reasons)

            if verdict.blocked:
                _safe_transition(candidate, CandidateState.BLOCKED, "policy blocked")
                result.record.candidates_rejected += 1
                self._record_outcome(
                    result, candidate, explanation, verdict, ReassessmentOutcome.BLOCKED
                )
                continue

            _safe_transition(candidate, CandidateState.TRIAGED, "policy evaluated")
            if not verdict.may_plan:
                result.record.candidates_rejected += 1
                self._record_outcome(result, candidate, explanation, verdict)
                continue
            if not ledger.try_consume("max_candidates_selected"):
                _safe_transition(
                    candidate, CandidateState.DEFERRED, "selection budget exhausted"
                )
                self._record_outcome(result, candidate, explanation, verdict)
                continue
            _safe_transition(candidate, CandidateState.SELECTED, "selected for this run")
            selected.append(candidate)

        result.record.candidates_selected = len(selected)
        batches = plan_execution_batches(selected, max_width=self.budget.max_dag_width)
        result.batches = [[c.candidate_id for c in batch] for batch in batches]
        self._selected = selected

    def _plan_and_execute(
        self, result: MaintenanceRunResult, ledger: BudgetLedger, mode: str
    ) -> None:
        for candidate in self._selected:
            self._check_cancelled()
            verdict = result.verdicts[candidate.candidate_id]
            explanation = next(
                exp for cand, exp in result.ranked if cand.candidate_id == candidate.candidate_id
            )
            order = build_work_order(
                candidate, granted_tier=verdict.granted_tier, budget=self.budget
            )
            result.work_orders[candidate.candidate_id] = order
            _safe_transition(candidate, CandidateState.PLANNED, "work order built")
            self._emit(f"[{STAGE_PLAN}] {candidate.candidate_id}: {order.objective}")

            if mode != RUN_MODE_EXECUTE:
                self._record_outcome(result, candidate, explanation, verdict)
                continue
            if verdict.granted_tier not in EXECUTING_TIERS:
                self._record_outcome(result, candidate, explanation, verdict)
                continue
            if self.executor is None:
                result.record.notes.append(
                    "no executor is wired in; execution-tier candidates were planned only"
                )
                self._record_outcome(result, candidate, explanation, verdict)
                continue
            if not ledger.try_consume("max_candidates_executed"):
                _safe_transition(
                    candidate, CandidateState.DEFERRED, "execution budget exhausted"
                )
                self._record_outcome(result, candidate, explanation, verdict)
                continue
            if ledger.exhausted("max_elapsed_seconds"):
                _safe_transition(candidate, CandidateState.DEFERRED, "time budget exhausted")
                self._record_outcome(result, candidate, explanation, verdict)
                continue

            self._execute_one(result, candidate, explanation, verdict, order, ledger)

    def _execute_one(
        self,
        result: MaintenanceRunResult,
        candidate: MaintenanceCandidate,
        explanation: PriorityExplanation,
        verdict: PolicyVerdict,
        order: MaintenanceWorkOrder,
        ledger: BudgetLedger,
    ) -> None:
        """EXECUTE -> VALIDATE -> REVIEW -> INTEGRATE -> LEARN -> REASSESS.

        A failure anywhere in here is isolated to this candidate: it is
        recorded, the candidate's failure counter is incremented, and the run
        continues with the next candidate. One bad candidate poisoning the
        whole run is exactly the failure mode Part 13 forbids.
        """
        _safe_transition(candidate, CandidateState.EXECUTING, "handed to the existing pipeline")
        result.record.execution_attempts += 1
        candidate.attempt_count += 1
        started = time.perf_counter()
        errors: list[str] = []
        self._emit(f"[{STAGE_EXECUTE}] {candidate.candidate_id}")

        try:
            outcome = self.executor(order)  # type: ignore[misc]
            if not isinstance(outcome, MaintenanceExecutionOutcome):
                raise TypeError(
                    f"executor returned {type(outcome).__name__}, expected "
                    "MaintenanceExecutionOutcome"
                )
        except Exception as exc:
            outcome = MaintenanceExecutionOutcome(
                succeeded=False, error=f"{type(exc).__name__}: {exc}"
            )
            errors.append(outcome.error)

        elapsed = time.perf_counter() - started
        ledger.observe("max_elapsed_seconds", outcome.elapsed_seconds or elapsed)

        # VALIDATE: the runner reads the pipeline's verdict; it never decides
        # what validation should have run. That authority stays where it is.
        validation_passed = outcome.validation_passed
        if validation_passed is None and outcome.succeeded:
            validation_passed = None  # unknown stays unknown, never assumed true
        if validation_passed is not False:
            _safe_transition(candidate, CandidateState.VALIDATED, "pipeline validation read")

        # REVIEW: scope containment. Checked here because it is the *maintenance*
        # contract - the candidate declared what it would touch - and it is
        # additional to, never instead of, the pipeline's own scope enforcement.
        out_of_scope = sorted(set(outcome.changed_files) - set(order.scope_files))
        if out_of_scope:
            errors.append(
                "changed file(s) outside the declared maintenance scope: "
                + ", ".join(out_of_scope[:5])
            )
        protected_touched = sorted(
            path
            for path in outcome.changed_files
            if path in self.policy.protected_paths
        )
        if protected_touched:
            errors.append(
                "changed protected file(s): " + ", ".join(protected_touched)
            )

        succeeded = bool(outcome.succeeded) and not errors and validation_passed is not False

        # LEARN
        if succeeded:
            result.record.executions_succeeded += 1
        else:
            result.record.executions_failed += 1
            candidate.failure_count += 1

        # REASSESS
        reassessment = self._reassess(candidate, executed=True, validation_passed=validation_passed)
        result.reassessments[candidate.candidate_id] = reassessment
        result.record.reassessments += 1
        if succeeded is False and reassessment.outcome == ReassessmentOutcome.RESOLVED:
            # Defensive: a failed or out-of-scope execution can never be
            # credited with a resolution, even if the rescan happens to agree.
            reassessment.outcome = ReassessmentOutcome.INCONCLUSIVE
            reassessment.reasons.append(
                "execution did not complete cleanly, so resolution cannot be credited"
            )
        candidate.record_outcome(reassessment.outcome, reason="; ".join(reassessment.reasons[:2]))
        _safe_transition(candidate, CandidateState.REASSESSED, reassessment.outcome)

        self._record_outcome(
            result,
            candidate,
            explanation,
            verdict,
            reassessment.outcome,
            executed=True,
            validation_passed=validation_passed,
            changed_files=outcome.changed_files,
            errors=errors + ([outcome.error] if outcome.error else []),
            elapsed=elapsed,
        )

    def _reassess(
        self,
        candidate: MaintenanceCandidate,
        *,
        executed: bool,
        validation_passed: bool | None,
    ) -> ReassessmentVerdict:
        self._emit(f"[{STAGE_REASSESS}] {candidate.candidate_id}")
        try:
            fresh = self.scan()
        except Exception as exc:
            verdict = ReassessmentVerdict(
                candidate_id=candidate.candidate_id,
                outcome=ReassessmentOutcome.INCONCLUSIVE,
                before_fingerprint=signal_fingerprint(candidate),
                before_severity=candidate.severity,
            )
            verdict.reasons.append(
                sanitize_text(f"reassessment scan failed: {type(exc).__name__}: {exc}")
            )
            return verdict
        after = next(
            (
                item
                for item in fresh.candidates
                if item.candidate_id == candidate.candidate_id
            ),
            None,
        )
        return reassess(
            candidate,
            after,
            executed=executed,
            validation_passed=validation_passed,
            rescan_degraded=fresh.degraded,
        )

    # -- bookkeeping -------------------------------------------------------

    def _record_outcome(
        self,
        result: MaintenanceRunResult,
        candidate: MaintenanceCandidate,
        explanation: PriorityExplanation,
        verdict: PolicyVerdict,
        outcome: str | None = None,
        *,
        executed: bool = False,
        validation_passed: bool | None = None,
        changed_files: Iterable[str] | None = None,
        errors: Iterable[str] | None = None,
        elapsed: float = 0.0,
    ) -> None:
        result.record.outcomes.append(
            CandidateRunOutcome(
                candidate_id=candidate.candidate_id,
                kind=candidate.kind,
                title=candidate.title,
                priority=explanation.score,
                granted_tier=verdict.granted_tier,
                state=candidate.state,
                outcome=outcome or candidate.outcome,
                executed=executed,
                validation_passed=validation_passed,
                reasons=list(verdict.cap_reasons or verdict.blocking_reasons),
                errors=list(errors or []),
                changed_files=list(changed_files or []),
                elapsed_seconds=elapsed,
            )
        )

    def _persist(self, result: MaintenanceRunResult) -> None:
        """Fold this run's candidates and record into the persistent store.

        Wrapped because persistence is *not* the work: a read-only data
        directory, a full disk or a storage backend that predates this phase
        must cost the operator their history, not their run.
        """

        def mutation(store: MaintenanceStore) -> None:
            for candidate in result.analysis.candidates:
                store.upsert(candidate)
            store.record_run(result.record)

        try:
            self.manager.mutate(mutation)
        except Exception as exc:
            result.record.notes.append(
                sanitize_text(f"maintenance history not persisted: {type(exc).__name__}")
            )

    def _emit(self, message: str) -> None:
        if self.progress is not None:
            try:
                self.progress(message)
            except Exception:
                pass

    def _check_cancelled(self) -> None:
        if self.cancelled is not None and self.cancelled():
            raise MaintenanceCancelled()


def _safe_transition(candidate: MaintenanceCandidate, state: str, reason: str) -> bool:
    """Transition, tolerating an illegal request.

    Used where the runner's control flow already guarantees legality; a
    violation is recorded in the candidate history rather than aborting a run
    that is otherwise proceeding correctly.
    """
    try:
        candidate.transition(state, reason=reason)
    except InvalidCandidateTransition:
        return False
    return True


def _timestamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# -- convenience wiring -------------------------------------------------------


def build_scan_function(
    analyzer: MaintenanceAnalyzer,
    *,
    lifecycle_provider: Callable[[], Any] | None = None,
    telemetry_provider: Callable[[], Any] | None = None,
    graph_provider: Callable[[], Any] | None = None,
    knowledge_provider: Callable[[], Any] | None = None,
    churn_provider: Callable[[], Mapping[str, int]] | None = None,
) -> ScanFn:
    """Compose a zero-argument scan from lazily-resolved sources.

    Each provider is called fresh on every scan, which is what makes
    reassessment meaningful: a cached semantic graph would report the
    pre-change state and every candidate would look like it persisted.
    A provider that raises costs its own source only - the scan continues with
    that source recorded as unavailable, and the resulting analysis is marked
    degraded, which in turn blocks any RESOLVED verdict.
    """

    def resolve(provider: Callable[[], Any] | None) -> Any:
        if provider is None:
            return None
        try:
            return provider()
        except Exception:
            return None

    def scan() -> AnalysisResult:
        return analyzer.analyze(
            lifecycle_store=resolve(lifecycle_provider),
            telemetry_store=resolve(telemetry_provider),
            semantic_graph=resolve(graph_provider),
            knowledge_graph=resolve(knowledge_provider),
            churn=resolve(churn_provider) or {},
        )

    return scan


def default_thresholds() -> MaintenanceThresholds:
    return MaintenanceThresholds()


def default_policy_thresholds() -> PolicyThresholds:
    return PolicyThresholds()
