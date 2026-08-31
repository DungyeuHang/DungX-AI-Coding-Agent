"""Phase 4.22/4.23: the one narrow maintenance execution seam.

Phase 4.21 built a complete maintenance *triage* system - DISCOVER, PRIORITIZE,
POLICY, PLAN - and then stopped, because
:class:`~local_agent.maintenance_runner.MaintenanceRunner` was wired with
``executor=None``. This module is that missing seam, and *only* that seam.

What it does
------------
It takes one :class:`~local_agent.maintenance_runner.MaintenanceWorkOrder` that
the policy has already granted an executing tier, re-checks every claim the
order makes, and drives the change through the machinery that already exists:

``MaintenanceWorkOrder`` -> freshness/idempotency/budget gates ->
:class:`~local_agent.coding_agent.InteractiveCodingAgent` ->
:class:`~local_agent.sandbox.CandidateWorkspace` (prospective validation) ->
:class:`~local_agent.coding_agent.CodingAgent` ``prepare`` -> the existing
approval boundary -> ``apply_prepared`` ->
:class:`~local_agent.validation_decision.ValidationDecisionEngine` (authoritative
post-apply validation scope) -> real subprocesses -> lifecycle / evidence /
telemetry.

What it deliberately does not do
--------------------------------
* It does not implement anything. Every byte written to a source file is written
  by :class:`~local_agent.coding_agent.CodingAgent`, from a
  :class:`~local_agent.models.FileOperation` the interactive agent emitted.
  :class:`MaintenanceExecutor`'s own class body contains no ``write_text``, no
  ``open``, no ``shutil`` copy and no ``chdir``, and the module imports no
  ``subprocess``; the only subprocesses it causes are run by the existing
  :class:`~local_agent.commands.CommandRunner`. The one place in this module
  that does touch the filesystem is :class:`ExecutionJournal`, which writes
  bookkeeping files into its own directory and nowhere else - it is a separate
  class precisely so the AST invariant tests can assert the above about the
  executor without an exception carved out for it.
* It does not decide validation scope. That is
  :class:`~local_agent.validation_decision.ValidationDecisionEngine`'s job and
  the executor consumes its answer without amending it downward.
* It does not invent an approval mechanism. The only tier the supported signal
  can ever reach is ``execute_with_existing_approval``, so the pre-existing
  approval boundary is *in force*, not bypassed. There is no autonomous-apply
  path in this build; see :data:`SUPPORTED_TIERS`.
* It does not widen anything. Every bound it computes is a ``min`` against what
  it was given.

The supported signal
--------------------
Exactly one: :data:`~local_agent.maintenance.MaintenanceSignal.PARSE_FAILURE`.

Phase 4.23 changed *how* that is decided rather than *what* it decides. The set
is no longer a literal in this file; it is
:data:`~local_agent.maintenance_oracle.AUTONOMOUS_SIGNAL_KINDS`, derived from
the signal inventory and the bound execution oracles. A signal is executable
only when its inventory entry says so **and** its oracle is deterministic, so a
future signal cannot be made autonomous by editing a list here - it has to
bring a machine-checkable success predicate with it. Everything else is refused
with ``unsupported_signal``, structurally, before any workspace is created.

What Phase 4.23 corrected
-------------------------
Phase 4.22's success predicate for a parse failure was "the file now parses",
checked inline. That is satisfiable by deleting the file's contents - ``pass``
parses - and an experiment against that build confirmed a sixteen-line module
could be replaced wholesale with ``pass``, applied to the authoritative tree,
and reported as ``completed`` with ``signal_resolved=True``. The predicate now
lives on :class:`~local_agent.maintenance_oracle.ParseOracle` and additionally
requires that the repair preserved the module's named surface and substance.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .coding_agent import CodingAgent, InteractiveCodingAgent, UnsafeModificationError
from .commands import CommandRunner, UnsafeCommandError
from .evidence import (
    STATUS_FAILED,
    STATUS_PASSED,
    STATUS_SKIPPED,
    EvidenceLedger,
    compute_policy_fingerprint,
    compute_state_fingerprint,
)
from .filesystem import ProjectFilesystem, ProtectedPathError, SandboxViolation
from .maintenance import (
    BudgetLedger,
    MaintenanceBudget,
    MaintenanceCandidate,
    candidate_identity,
    is_protected_relative_path,
    sanitize_relative_path,
    sanitize_text,
)
from .maintenance_oracle import (
    AUTONOMOUS_SIGNAL_KINDS,
    ORACLE_FRAMEWORK_VERSION,
    ExecutionOracle,
    OracleObservation,
    oracle_for,
)
from .maintenance_policy import (
    AutonomyTier,
    MaintenanceExecutionPolicy,
    tier_rank,
)
from .maintenance_runner import MaintenanceExecutionOutcome, MaintenanceWorkOrder
from .models import CommandSpec, FileOperation, Plan, ProjectContext
from .patching import PatchApplicationError
from .sandbox import CandidateWorkspace, ProspectiveValidator
from .validation_decision import ValidationDecisionEngine
from .validation_lifecycle import (
    RESULT_FAILED,
    RESULT_NOT_RUN,
    RESULT_PASSED,
    STAGE_POST_APPLY,
    LifecycleState,
    ValidationIterationRecord,
)

LOGGER = logging.getLogger(__name__)

#: Version stamp for this executor's behaviour. Recorded on every result and
#: folded into the policy fingerprint, so a work order planned under one
#: executor generation is not silently executed by another.
MAINTENANCE_EXECUTOR_VERSION = "4.23.0"


# =============================================================================
# Signal selection
# =============================================================================

#: The complete set of maintenance signals this build can execute.
#:
#: **SUPPORTED SIGNAL:** ``parse_failure``, and it is the only one.
#:
#: This is no longer a literal. It is
#: :data:`~local_agent.maintenance_oracle.AUTONOMOUS_SIGNAL_KINDS`, which is
#: computed from two independent facts about every signal: its
#: :class:`~local_agent.maintenance_oracle.SignalInventoryEntry` must declare
#: ``autonomous_execution``, *and* the oracle bound to it must be
#: deterministic. Both are required, so making a signal autonomous means
#: writing an oracle that can mechanically prove remediation - not editing a
#: set. Every rejected signal's reasoning is recorded, per signal, on its
#: inventory entry, and the test-suite asserts that the twelve rejected kinds
#: each carry at least one reason.
SUPPORTED_SIGNAL_KINDS: frozenset[str] = AUTONOMOUS_SIGNAL_KINDS

#: Autonomy tiers this executor will act at.
#:
#: **AUTONOMY TIER:** ``execute_with_existing_approval`` only.
#:
#: ``execute_autonomously`` is deliberately absent. The supported signal cannot
#: reach it anyway - a structural observation has ``sample_size == 1``, which the
#: policy's unattended-execution bar (five samples) permanently caps - but
#: listing only the lower tier means that even if a future signal *could* reach
#: the higher one, it would not inherit an autonomous-apply path from here.
#: There is no such path in this build: every apply passes through the existing
#: approval boundary.
SUPPORTED_TIERS: frozenset[str] = frozenset(
    {AutonomyTier.EXECUTE_WITH_EXISTING_APPROVAL}
)

#: **MAX SCOPE / MAX FILES.** One file, and it must be the Python file the
#: signal names. Not configurable upward: a parse failure is a single-file fact.
MAX_SCOPE_FILES = 1

#: **MAX ITERATIONS.** Hard ceilings applied on top of whatever the work order
#: asks for. ``min(order, these)`` is always taken.
MAX_TOOL_STEPS_CEILING = 20
MAX_CANDIDATE_ITERATIONS_CEILING = 3
MAX_VALIDATION_COMMANDS_CEILING = 8
#: **MAX DIFF SIZE.** Unified-diff lines across all prepared changes.
MAX_DIFF_LINES_CEILING = 400

#: File extensions the executor will touch. A parse failure is a Python fact.
SUPPORTED_SUFFIXES: frozenset[str] = frozenset({".py"})


# =============================================================================
# Budget enforcement, declared rather than implied
# =============================================================================
#
# Phase 4.21's audit found several maintenance budgets to be decorative. Phase
# 4.22 wired some and documented others in prose. Prose rots, so Phase 4.23
# states it as data and the test-suite asserts that these two sets together
# cover every field of :class:`~local_agent.maintenance.MaintenanceBudget` -
# a new budget field cannot be added without a deliberate decision about
# whether this executor honours it.

#: Budget fields this executor genuinely enforces: exceeding one changes the
#: outcome, and the test-suite proves each with a real refusal.
ENFORCED_BUDGET_FIELDS: frozenset[str] = frozenset(
    {
        # min()-ed into the effective limits, zero-checked, and re-checked
        # against what the interactive agent actually consumed.
        "max_tool_steps_per_subtask",
        "max_candidate_iterations",
        "max_validation_commands",
        # Bounds the prepared diff; a breach refuses before the apply.
        "max_changed_lines_per_candidate",
        # Caps the tier in the policy, and is now compared against the files
        # the authoritative apply actually reported.
        "max_changed_files_per_candidate",
        # Re-checked before execution and again immediately before the apply.
        "max_elapsed_seconds",
        # Zero means "no execution permitted", enforced by the policy.
        "max_candidates_executed",
    }
)

#: Budget fields this executor does **not** enforce, each with the reason.
#:
#: They are not silently ignored and they are not pretended to be controls.
#: Several are enforced elsewhere - the run-level ones belong to
#: :class:`~local_agent.maintenance_runner.MaintenanceRunner`, which is the
#: component that selects and batches candidates - and the rest describe
#: machinery this build does not have.
UNENFORCED_BUDGET_FIELDS: Mapping[str, str] = {
    "max_candidates_considered": (
        "run-level; enforced by MaintenanceRunner during selection, not by a "
        "single execution"
    ),
    "max_candidates_selected": (
        "run-level; enforced by MaintenanceRunner during selection"
    ),
    "max_dag_width": (
        "run-level; enforced by MaintenanceRunner.plan_execution_batches"
    ),
    "max_subtasks_per_candidate": (
        "PARTIALLY ENFORCED: this executor performs exactly one implementation "
        "and never decomposes a candidate, so the only value that can bind is "
        "zero - and a work order permitting no subtask is refused. Above one "
        "the bound constrains nothing, because there is nothing to count"
    ),
    "max_repair_iterations_per_candidate": (
        "ADVISORY here: this build has no repair loop. A candidate that fails "
        "prospective validation is refused, not retried, so no repair "
        "iteration is ever consumed. The bound will become real only when a "
        "repair loop exists"
    ),
    "max_estimated_cost_units": (
        "NOT ENFORCED ANYWHERE: no component estimates the cost of a "
        "maintenance action, so there is no quantity to compare against. It is "
        "carried through configuration and persisted, and it constrains "
        "nothing"
    ),
}


# =============================================================================
# Result vocabulary
# =============================================================================


class MaintenanceExecutionStatus:
    """Every distinguishable way one maintenance execution can end.

    Deliberately *not* collapsed into success/failure. An operator needs to tell
    "the policy refused" from "the provider was rate-limited" from "the change
    applied and then failed post-apply validation" - those imply completely
    different next actions, and a single ``failed`` would hide all of it.
    """

    COMPLETED = "completed"
    NO_CHANGE = "no_change"

    MALFORMED_WORK_ORDER = "malformed_work_order"
    UNSUPPORTED_SIGNAL = "unsupported_signal"
    REFUSED_BY_POLICY = "refused_by_policy"
    STALE_CANDIDATE = "stale_candidate"
    DUPLICATE_EXECUTION = "duplicate_execution"
    BUDGET_EXHAUSTED = "budget_exhausted"
    SCOPE_VIOLATION = "scope_violation"
    APPROVAL_REQUIRED = "approval_required"

    PROVIDER_FAILURE = "provider_failure"
    IMPLEMENTATION_FAILURE = "implementation_failure"
    PROSPECTIVE_VALIDATION_FAILED = "prospective_validation_failed"
    VALIDATION_DECISION_REJECTED = "validation_decision_rejected"
    APPLY_FAILED = "apply_failed"
    POST_VALIDATION_FAILED = "post_validation_failed"
    #: The change applied, post-apply validation genuinely passed, and the
    #: maintenance signal is *still there*. Kept distinct from
    #: ``post_validation_failed`` because the two say opposite things about the
    #: repository's health: this one means the repository is fine and the repair
    #: was ineffective. The change is rolled back either way.
    SIGNAL_NOT_RESOLVED = "signal_not_resolved"
    #: The apply touched more files than the per-candidate budget allowed. A
    #: separate status from ``budget_exhausted`` precisely because the tree
    #: *was* written before the breach was noticed, so it must not appear in
    #: :data:`NO_MUTATION_STATUSES`. Defence in depth: with
    #: :data:`MAX_SCOPE_FILES` at one and the scope check upstream this should
    #: be unreachable, and reaching it means a wiring defect rather than a bad
    #: candidate - which is why it is not retryable either.
    POST_APPLY_BUDGET_BREACH = "post_apply_budget_breach"


ALL_EXECUTION_STATUSES: tuple[str, ...] = (
    MaintenanceExecutionStatus.COMPLETED,
    MaintenanceExecutionStatus.NO_CHANGE,
    MaintenanceExecutionStatus.MALFORMED_WORK_ORDER,
    MaintenanceExecutionStatus.UNSUPPORTED_SIGNAL,
    MaintenanceExecutionStatus.REFUSED_BY_POLICY,
    MaintenanceExecutionStatus.STALE_CANDIDATE,
    MaintenanceExecutionStatus.DUPLICATE_EXECUTION,
    MaintenanceExecutionStatus.BUDGET_EXHAUSTED,
    MaintenanceExecutionStatus.SCOPE_VIOLATION,
    MaintenanceExecutionStatus.APPROVAL_REQUIRED,
    MaintenanceExecutionStatus.PROVIDER_FAILURE,
    MaintenanceExecutionStatus.IMPLEMENTATION_FAILURE,
    MaintenanceExecutionStatus.PROSPECTIVE_VALIDATION_FAILED,
    MaintenanceExecutionStatus.VALIDATION_DECISION_REJECTED,
    MaintenanceExecutionStatus.APPLY_FAILED,
    MaintenanceExecutionStatus.POST_VALIDATION_FAILED,
    MaintenanceExecutionStatus.SIGNAL_NOT_RESOLVED,
    MaintenanceExecutionStatus.POST_APPLY_BUDGET_BREACH,
)

#: Statuses after which the same candidate may legitimately be attempted again
#: later. Everything else is a *decision* about the candidate rather than an
#: accident, and retrying it would just reproduce the same refusal.
#:
#: Note that retryability is not the same as unbounded retry: the candidate's
#: ``failure_count`` still rises on every failed attempt, and
#: :class:`~local_agent.maintenance_policy.PolicyThresholds` blocks it outright
#: once that reaches ``max_failures_before_block``. That is what bounds the loop.
RETRYABLE_STATUSES: frozenset[str] = frozenset(
    {
        MaintenanceExecutionStatus.PROVIDER_FAILURE,
        MaintenanceExecutionStatus.IMPLEMENTATION_FAILURE,
        MaintenanceExecutionStatus.PROSPECTIVE_VALIDATION_FAILED,
        MaintenanceExecutionStatus.BUDGET_EXHAUSTED,
        MaintenanceExecutionStatus.APPROVAL_REQUIRED,
    }
)

#: Statuses that prove the authoritative tree was never written by this
#: execution. Asserted by the test-suite against real bytes on disk, not merely
#: documented here.
NO_MUTATION_STATUSES: frozenset[str] = frozenset(
    {
        MaintenanceExecutionStatus.NO_CHANGE,
        MaintenanceExecutionStatus.MALFORMED_WORK_ORDER,
        MaintenanceExecutionStatus.UNSUPPORTED_SIGNAL,
        MaintenanceExecutionStatus.REFUSED_BY_POLICY,
        MaintenanceExecutionStatus.STALE_CANDIDATE,
        MaintenanceExecutionStatus.DUPLICATE_EXECUTION,
        MaintenanceExecutionStatus.BUDGET_EXHAUSTED,
        MaintenanceExecutionStatus.SCOPE_VIOLATION,
        MaintenanceExecutionStatus.APPROVAL_REQUIRED,
        MaintenanceExecutionStatus.PROVIDER_FAILURE,
        MaintenanceExecutionStatus.IMPLEMENTATION_FAILURE,
        MaintenanceExecutionStatus.PROSPECTIVE_VALIDATION_FAILED,
        MaintenanceExecutionStatus.VALIDATION_DECISION_REJECTED,
        MaintenanceExecutionStatus.APPLY_FAILED,
    }
)


@dataclass
class MaintenanceExecutionResult:
    """The structured account of one attempted maintenance execution.

    ``validation_passed`` is ``None`` unless a real post-apply verdict exists.
    That tri-state is load-bearing: the runner's reassessment refuses to credit
    ``RESOLVED`` without a verdict, and returning ``False`` for "we never got
    there" would misreport a refusal as a validation failure.
    """

    candidate_id: str = ""
    status: str = MaintenanceExecutionStatus.MALFORMED_WORK_ORDER
    signal_kind: str = ""
    granted_tier: str = ""
    executor_version: str = MAINTENANCE_EXECUTOR_VERSION
    execution_key: str = ""

    applied: bool = False
    rolled_back: bool = False
    validation_passed: bool | None = None
    prospective_validation_passed: bool | None = None
    signal_resolved: bool | None = None

    changed_files: list[str] = field(default_factory=list)
    inspected_files: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # -- Phase 4.23 oracle observability ---------------------------------
    #: Which oracle answered for this signal, and how strong it can ever be.
    oracle_name: str = ""
    oracle_class: str = ""
    #: The BEFORE/AFTER observations, as plain dicts. ``None`` when the stage
    #: was never reached - never an empty dict standing in for "nothing wrong".
    oracle_precondition: dict[str, Any] | None = None
    oracle_postcondition: dict[str, Any] | None = None
    #: Post-apply commands that genuinely executed, and those that could not.
    post_apply_executed_commands: list[list[str]] = field(default_factory=list)
    post_apply_skipped_commands: list[list[str]] = field(default_factory=list)
    #: True only when at least one post-apply command actually ran. Recorded
    #: separately from ``validation_passed`` so telemetry can never present
    #: "nothing ran" as validation coverage.
    post_apply_executed_any: bool = False

    provider: str = ""
    model: str = ""
    tool_steps_used: int = 0
    candidate_iterations: int = 0
    prospective_commands_run: int = 0
    post_apply_commands_run: int = 0
    post_apply_commands: list[list[str]] = field(default_factory=list)
    diff_lines: int = 0
    validation_scope: str = ""
    validation_confidence: str = ""
    lifecycle_id: str = ""
    lifecycle_state: str = ""
    decision_id: str = ""
    evidence_recorded: int = 0
    elapsed_seconds: float = 0.0

    @property
    def retryable(self) -> bool:
        return self.status in RETRYABLE_STATUSES

    @property
    def succeeded(self) -> bool:
        """True only for a change that applied *and* post-validated.

        A successful write is not successful maintenance, and ``no_change`` -
        the agent legitimately concluding there was nothing to do - is not a
        success either, because nothing was proved about the signal.
        """
        return (
            self.status == MaintenanceExecutionStatus.COMPLETED
            and self.applied
            and self.validation_passed is True
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "status": self.status,
            "signal_kind": self.signal_kind,
            "granted_tier": self.granted_tier,
            "executor_version": self.executor_version,
            "execution_key": self.execution_key,
            "applied": self.applied,
            "rolled_back": self.rolled_back,
            "validation_passed": self.validation_passed,
            "prospective_validation_passed": self.prospective_validation_passed,
            "signal_resolved": self.signal_resolved,
            "succeeded": self.succeeded,
            "retryable": self.retryable,
            "changed_files": list(self.changed_files),
            "inspected_files": list(self.inspected_files),
            "reasons": list(self.reasons),
            "errors": list(self.errors),
            "oracle_name": self.oracle_name,
            "oracle_class": self.oracle_class,
            "oracle_precondition": self.oracle_precondition,
            "oracle_postcondition": self.oracle_postcondition,
            "post_apply_executed_commands": [
                list(c) for c in self.post_apply_executed_commands
            ],
            "post_apply_skipped_commands": [
                list(c) for c in self.post_apply_skipped_commands
            ],
            "post_apply_executed_any": self.post_apply_executed_any,
            "provider": self.provider,
            "model": self.model,
            "tool_steps_used": self.tool_steps_used,
            "candidate_iterations": self.candidate_iterations,
            "prospective_commands_run": self.prospective_commands_run,
            "post_apply_commands_run": self.post_apply_commands_run,
            "post_apply_commands": [list(c) for c in self.post_apply_commands],
            "diff_lines": self.diff_lines,
            "validation_scope": self.validation_scope,
            "validation_confidence": self.validation_confidence,
            "lifecycle_id": self.lifecycle_id,
            "lifecycle_state": self.lifecycle_state,
            "decision_id": self.decision_id,
            "evidence_recorded": self.evidence_recorded,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
        }

    def to_outcome(self) -> MaintenanceExecutionOutcome:
        """Adapt to the Phase 4.21 executor-seam return type.

        ``validation_passed`` is passed through unchanged, ``None`` included:
        the runner's reassessment treats an unknown verdict as
        ``INCONCLUSIVE`` and never as a pass, which is exactly the invariant
        this executor must not weaken.
        """
        notes = [f"maintenance execution status: {self.status}"]
        notes.extend(self.reasons[:8])
        if self.applied and not self.rolled_back:
            notes.append("change applied to the authoritative tree")
        if self.rolled_back:
            notes.append("apply was rolled back after post-apply validation failed")
        return MaintenanceExecutionOutcome(
            succeeded=self.succeeded,
            validation_passed=self.validation_passed,
            changed_files=list(self.changed_files) if not self.rolled_back else [],
            task_id=self.lifecycle_id,
            error="; ".join(self.errors[:3]),
            notes=notes,
            elapsed_seconds=self.elapsed_seconds,
        )


# =============================================================================
# Idempotency journal
# =============================================================================


class ExecutionJournal:
    """Durable, cross-process claim ledger keyed by execution identity.

    The key is *not* the candidate id. Candidate ids are content hashes of
    ``(kind, subject)``, so the same file failing to parse a week later has the
    identical id - and must be executable again. The key therefore folds in the
    observed repository state and the policy that permitted the attempt, so:

    * same candidate + same repository state -> same key -> refused as duplicate;
    * same candidate + changed repository state -> new key -> a fresh attempt is
      allowed (and the freshness gate independently re-checks that the signal
      still reproduces).

    The claim is an ``O_CREAT | O_EXCL`` file creation, which is atomic on both
    POSIX and Windows, so two processes racing on the same key cannot both win.
    A claim that is never completed - a crash between apply and persistence -
    stays on disk as ``in_progress`` and permanently refuses that key, which is
    the fail-closed answer: a half-finished autonomous change must be looked at
    by a human, not silently retried.

    This class exists so :class:`MaintenanceExecutor` itself contains no
    filesystem writes at all; the AST invariant tests assert exactly that about
    the executor class body.
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self._lock = threading.Lock()

    def _path(self, key: str) -> Path:
        safe = "".join(ch for ch in str(key) if ch.isalnum() or ch in "-_")[:64]
        return self.directory / f"{safe or 'unkeyed'}.json"

    def claim(self, key: str, *, detail: Mapping[str, Any] | None = None) -> bool:
        """Atomically take ownership of ``key``. False when already claimed."""
        path = self._path(key)
        payload = json.dumps(
            {
                "key": str(key),
                "status": "in_progress",
                "claimed_at": _timestamp(),
                "executor_version": MAINTENANCE_EXECUTOR_VERSION,
                "detail": dict(detail or {}),
            },
            sort_keys=True,
        )
        with self._lock:
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                descriptor = os.open(
                    str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
                )
            except FileExistsError:
                return False
            except OSError as exc:
                # A journal we cannot write is a journal that cannot prevent a
                # duplicate. Fail closed: refuse the execution.
                LOGGER.warning("Could not claim maintenance execution %s: %s", key, exc)
                return False
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(payload)
            except OSError as exc:  # pragma: no cover - defensive
                LOGGER.warning("Could not write maintenance journal entry: %s", exc)
            return True

    def complete(self, key: str, status: str, detail: Mapping[str, Any] | None = None) -> None:
        """Record the terminal status of a claimed key."""
        path = self._path(key)
        payload = json.dumps(
            {
                "key": str(key),
                "status": str(status),
                "completed_at": _timestamp(),
                "executor_version": MAINTENANCE_EXECUTOR_VERSION,
                "detail": dict(detail or {}),
            },
            sort_keys=True,
        )
        with self._lock:
            try:
                path.write_text(payload, encoding="utf-8")
            except OSError as exc:  # pragma: no cover - defensive
                LOGGER.warning("Could not finalise maintenance journal entry: %s", exc)

    def release(self, key: str) -> None:
        """Drop a claim so a retryable failure may be attempted again.

        Only ever called for statuses in :data:`RETRYABLE_STATUSES`. The retry
        is still bounded by the candidate's ``failure_count`` and the policy's
        ``max_failures_before_block``; the journal bounds *duplication*, not
        retries.
        """
        with self._lock:
            try:
                self._path(key).unlink()
            except (OSError, FileNotFoundError):
                pass

    def status_of(self, key: str) -> str:
        try:
            data = json.loads(self._path(key).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ""
        return str(data.get("status", "")) if isinstance(data, dict) else ""


# =============================================================================
# Approval boundary
# =============================================================================


@dataclass
class ApprovalOutcome:
    """Whether the existing approval boundary lets this change be written."""

    approved: bool = False
    manual_required: bool = False
    reason: str = ""


class MaintenanceApprovalGate:
    """The existing approval boundary, consulted - never replaced.

    The executor holds no opinion of its own about approval. It asks this gate,
    and this gate asks exactly what the orchestrator asks: the configured
    ``approval`` mode, and, in ``policy`` mode, the unmodified
    :class:`~local_agent.approval.ApprovalPolicyEngine`.

    Two *additional* conditions apply on top, never instead:

    * ``apply_enabled`` must be true. This is the explicit operator opt-in
      (``maintenance run --apply``). Discovery being enabled is never sufficient.
    * ``approver`` must actually approve when manual approval is required. There
      is no default-approve path; a missing approver means refusal.
    """

    def __init__(
        self,
        *,
        approval_mode: str = "never",
        policy_engine: Any | None = None,
        approver: Callable[[list[Any]], bool] | None = None,
        apply_enabled: bool = False,
    ):
        self.approval_mode = str(approval_mode or "never").lower()
        self.policy_engine = policy_engine
        self.approver = approver
        self.apply_enabled = bool(apply_enabled)

    def evaluate(self, prepared: list[Any], impact: Any | None = None) -> ApprovalOutcome:
        if not self.apply_enabled:
            return ApprovalOutcome(
                approved=False,
                manual_required=True,
                reason=(
                    "autonomous maintenance apply is not enabled for this run; "
                    "re-run with --apply to permit the authoritative write"
                ),
            )
        if not prepared:
            return ApprovalOutcome(approved=False, reason="nothing to approve")

        manual_required = True
        if self.approval_mode == "never":
            manual_required = False
        elif self.approval_mode == "policy":
            if self.policy_engine is None:
                manual_required = True
            else:
                try:
                    manual_required = bool(
                        self.policy_engine.is_manual_approval_required(prepared, impact)
                    )
                except Exception as exc:  # noqa: BLE001 - fail closed
                    LOGGER.warning("Approval policy evaluation failed: %s", exc)
                    manual_required = True
        else:  # "always" and anything unrecognised
            manual_required = True

        if not manual_required:
            return ApprovalOutcome(
                approved=True,
                manual_required=False,
                reason=f"approval mode '{self.approval_mode}' requires no manual approval",
            )
        if self.approver is None:
            return ApprovalOutcome(
                approved=False,
                manual_required=True,
                reason=(
                    "the existing approval boundary requires manual approval and no "
                    "approver is available in this run"
                ),
            )
        try:
            approved = bool(self.approver(prepared))
        except Exception as exc:  # noqa: BLE001 - fail closed
            LOGGER.warning("Maintenance approver raised: %s", exc)
            approved = False
        return ApprovalOutcome(
            approved=approved,
            manual_required=True,
            reason=("approved by the operator" if approved else "operator declined the change"),
        )


# =============================================================================
# The executor
# =============================================================================


class MaintenanceExecutor:
    """Executes exactly one permitted maintenance work order.

    Responsibilities, and the ones that stay elsewhere:

    ===========================  ================================================
    this class                   somewhere else
    ===========================  ================================================
    re-check the order's claims  discovery (``MaintenanceAnalyzer``)
    enforce the execution ceiling ranking (``MaintenancePriorityEngine``)
    gate on freshness/identity   permission (``MaintenanceExecutionPolicy``)
    drive one implementation     implementing (``InteractiveCodingAgent``)
    keep prospective validation  validating a candidate (``CandidateWorkspace``)
    ask the approval boundary    deciding approval (``ApprovalPolicyEngine``)
    read the validation verdict  choosing scope (``ValidationDecisionEngine``)
    record what happened         storing it (lifecycle/telemetry managers)
    ===========================  ================================================

    Every collaborator is injected. Nothing here reads ``os.getcwd``, mutates the
    process working directory, or holds module-level mutable state.
    """

    def __init__(
        self,
        *,
        root: str | Path,
        provider_factory: Callable[[], Any],
        policy: MaintenanceExecutionPolicy,
        budget: MaintenanceBudget,
        configured_tier: str,
        journal: ExecutionJournal,
        approval_gate: MaintenanceApprovalGate,
        context_provider: Callable[[], ProjectContext],
        lifecycle_manager: Any | None = None,
        telemetry_manager: Any | None = None,
        ledger: BudgetLedger | None = None,
        workspace_parent: str | Path | None = None,
        command_timeout_seconds: int = 300,
        tool_policy: Any | None = None,
        semantic_index: Any | None = None,
        repo_lock: Any | None = None,
        progress: Callable[[str], None] | None = None,
        deadline: Callable[[], bool] | None = None,
        min_impact_confidence: str = "high",
    ):
        self.root = Path(root).expanduser().resolve()
        self.provider_factory = provider_factory
        self.policy = policy
        self.budget = budget
        self.configured_tier = str(configured_tier)
        self.journal = journal
        self.approval_gate = approval_gate
        self.context_provider = context_provider
        self.lifecycle_manager = lifecycle_manager
        self.telemetry_manager = telemetry_manager
        self.ledger = ledger
        self.workspace_parent = workspace_parent
        self.command_timeout_seconds = int(command_timeout_seconds)
        self.tool_policy = tool_policy
        self.semantic_index = semantic_index
        self.repo_lock = repo_lock or threading.Lock()
        self.progress = progress
        self.deadline = deadline
        self.min_impact_confidence = str(min_impact_confidence)
        #: Every result this executor produced, newest last. Instance state, so
        #: two executors never see each other's history.
        self.results: list[MaintenanceExecutionResult] = []

    # -- the Phase 4.21 seam ----------------------------------------------

    def __call__(self, order: MaintenanceWorkOrder) -> MaintenanceExecutionOutcome:
        """Adapt :meth:`execute` to ``ExecutorFn``."""
        return self.execute(order).to_outcome()

    # -- main flow ---------------------------------------------------------

    def execute(self, order: MaintenanceWorkOrder) -> MaintenanceExecutionResult:
        started = time.perf_counter()
        result = MaintenanceExecutionResult(
            candidate_id=getattr(order, "candidate_id", ""),
            granted_tier=str(getattr(order, "granted_tier", "")),
        )
        claimed_key = ""
        try:
            claimed_key = self._run(order, result)
        except Exception as exc:  # noqa: BLE001 - one bad order must not kill a run
            LOGGER.exception("Maintenance execution raised")
            result.status = MaintenanceExecutionStatus.IMPLEMENTATION_FAILURE
            result.errors.append(sanitize_text(f"{type(exc).__name__}: {exc}"))
        finally:
            result.elapsed_seconds = time.perf_counter() - started
            if claimed_key:
                if result.status in RETRYABLE_STATUSES and not result.applied:
                    self.journal.release(claimed_key)
                else:
                    self.journal.complete(
                        claimed_key,
                        result.status,
                        {"applied": result.applied, "validated": result.validation_passed},
                    )
            self.results.append(result)
        return result

    def _run(self, order: MaintenanceWorkOrder, result: MaintenanceExecutionResult) -> str:
        """Returns the claimed journal key (empty when nothing was claimed)."""
        # -- 1. work order shape ------------------------------------------
        candidate = self._validate_order(order, result)
        if candidate is None:
            return ""
        result.signal_kind = candidate.kind
        relative = order.scope_files[0]

        # -- 2. supported signal, and the oracle that will judge it --------
        if candidate.kind not in SUPPORTED_SIGNAL_KINDS:
            return self._refuse(
                result,
                MaintenanceExecutionStatus.UNSUPPORTED_SIGNAL,
                f"signal kind '{candidate.kind}' is not executable by this build; "
                f"supported: {', '.join(sorted(SUPPORTED_SIGNAL_KINDS))}",
            )
        oracle = oracle_for(candidate.kind)
        result.oracle_name = oracle.name
        result.oracle_class = oracle.oracle_class
        # Second, independent gate. ``SUPPORTED_SIGNAL_KINDS`` is already
        # derived from oracle promotability, so this can only fire if that set
        # was monkey-patched or the registry was tampered with at run time -
        # which is exactly the case worth refusing, because a non-deterministic
        # oracle cannot prove a repair worked.
        if not oracle.promotable:
            return self._refuse(
                result,
                MaintenanceExecutionStatus.UNSUPPORTED_SIGNAL,
                f"the oracle bound to '{candidate.kind}' is '{oracle.oracle_class}', "
                "not deterministic; autonomous execution requires a mechanically "
                "checkable success predicate",
            )
        if Path(relative).suffix.lower() not in oracle.supported_suffixes:
            return self._refuse(
                result,
                MaintenanceExecutionStatus.MALFORMED_WORK_ORDER,
                f"'{relative}' is not a file type oracle '{oracle.name}' can judge "
                f"({', '.join(sorted(oracle.supported_suffixes)) or 'none'})",
            )
        if len(order.scope_files) > max(0, int(oracle.max_scope_files)):
            return self._refuse(
                result,
                MaintenanceExecutionStatus.SCOPE_VIOLATION,
                f"oracle '{oracle.name}' can judge at most "
                f"{oracle.max_scope_files} file(s); the order declares "
                f"{len(order.scope_files)}",
            )

        # -- 3. protected path, early (authoritative check is downstream) --
        if self.policy.is_protected(relative) or is_protected_relative_path(relative):
            return self._refuse(
                result,
                MaintenanceExecutionStatus.REFUSED_BY_POLICY,
                f"'{relative}' is on the protected floor and may never be modified "
                "by maintenance",
            )

        # -- 4. policy, re-decided from scratch ---------------------------
        verdict = self.policy.decide(
            candidate,
            configured_tier=self.configured_tier,
            budget=self.budget,
        )
        if not verdict.may_execute:
            return self._refuse(
                result,
                MaintenanceExecutionStatus.REFUSED_BY_POLICY,
                "policy re-check refused execution: "
                + "; ".join(verdict.blocking_reasons or verdict.cap_reasons)[:300],
            )
        if verdict.granted_tier not in SUPPORTED_TIERS:
            return self._refuse(
                result,
                MaintenanceExecutionStatus.REFUSED_BY_POLICY,
                f"granted tier '{verdict.granted_tier}' is outside the set this "
                f"executor acts at ({', '.join(sorted(SUPPORTED_TIERS))})",
            )
        if tier_rank(order.granted_tier) > tier_rank(verdict.granted_tier):
            return self._refuse(
                result,
                MaintenanceExecutionStatus.REFUSED_BY_POLICY,
                f"the work order claims tier '{order.granted_tier}' but the policy "
                f"now grants only '{verdict.granted_tier}'",
            )
        result.granted_tier = verdict.granted_tier

        # -- 5. budgets ----------------------------------------------------
        limits = self._effective_limits(order, result)
        if limits is None:
            return ""

        # -- 6. freshness / TOCTOU, and the oracle's BEFORE observation ----
        before = self._check_freshness(order, oracle, relative, result)
        if before is None:
            return ""

        # -- 7. idempotency ------------------------------------------------
        key = self._execution_key(order, candidate, relative)
        result.execution_key = key
        if not self.journal.claim(key, detail={"candidate_id": candidate.candidate_id}):
            existing = self.journal.status_of(key)
            return self._refuse(
                result,
                MaintenanceExecutionStatus.DUPLICATE_EXECUTION,
                "this candidate has already been executed against this exact "
                f"repository state (journal status: {existing or 'unknown'})",
            )

        # -- 8. implementation, prospectively validated --------------------
        plan = self._build_plan(order, oracle, relative)
        implementation, evidence_ledger = self._implement(
            order, oracle, plan, relative, limits, result
        )
        if implementation is None:
            return key

        operations = list(implementation.file_operations or [])
        if not operations:
            result.status = MaintenanceExecutionStatus.NO_CHANGE
            result.reasons.append(
                "the implementation agent produced no file operations; nothing was applied"
            )
            return key

        # -- 9. scope containment ------------------------------------------
        if not self._check_scope(operations, plan, result):
            return key

        # -- 10. prepare (still no write) ----------------------------------
        prepared = self._prepare(operations, plan, result)
        if prepared is None:
            return key

        result.diff_lines = sum(len(change.diff.splitlines()) for change in prepared)
        if result.diff_lines > limits["max_diff_lines"]:
            # NOTE the ``return key``, not ``return self._refuse(...)``: every
            # refusal *after* the journal claim must still hand the key back, or
            # the claim is neither completed nor released and the candidate is
            # permanently locked out even though this outcome is retryable.
            self._refuse(
                result,
                MaintenanceExecutionStatus.BUDGET_EXHAUSTED,
                f"prepared diff is {result.diff_lines} line(s), over the "
                f"{limits['max_diff_lines']}-line maintenance ceiling",
            )
            return key

        # -- 11. approval boundary -----------------------------------------
        approval = self.approval_gate.evaluate(prepared, None)
        if not approval.approved:
            self._refuse(
                result,
                MaintenanceExecutionStatus.APPROVAL_REQUIRED,
                approval.reason,
            )
            return key
        result.reasons.append(f"approval boundary: {approval.reason}")

        # -- 11b. last deadline/elapsed check before the authoritative write.
        # Everything above can take minutes (a provider round-trip, a candidate
        # rebuild, real subprocesses). Checking the deadline only at the start
        # would let a run that has already overrun still perform the one action
        # that mutates the repository.
        if self._out_of_time():
            self._refuse(
                result,
                MaintenanceExecutionStatus.BUDGET_EXHAUSTED,
                "the run's time budget was exhausted before the authoritative "
                "apply; nothing was written",
            )
            return key

        # -- 12. apply + post-apply validation -----------------------------
        self._apply_and_validate(
            order, oracle, before, prepared, relative, limits, evidence_ledger, result
        )
        return key

    # -- stage helpers -----------------------------------------------------

    def _validate_order(
        self, order: MaintenanceWorkOrder, result: MaintenanceExecutionResult
    ) -> MaintenanceCandidate | None:
        """Reject anything malformed before a workspace is ever created."""
        if not isinstance(order, MaintenanceWorkOrder):
            self._refuse(
                result,
                MaintenanceExecutionStatus.MALFORMED_WORK_ORDER,
                f"expected a MaintenanceWorkOrder, got {type(order).__name__}",
            )
            return None
        if not order.candidate_id:
            self._refuse(
                result,
                MaintenanceExecutionStatus.MALFORMED_WORK_ORDER,
                "work order carries no candidate id",
            )
            return None
        if len(order.scope_files) != MAX_SCOPE_FILES:
            self._refuse(
                result,
                MaintenanceExecutionStatus.MALFORMED_WORK_ORDER,
                f"work order declares {len(order.scope_files)} scope file(s); this "
                f"executor supports exactly {MAX_SCOPE_FILES}",
            )
            return None
        relative = sanitize_relative_path(order.scope_files[0])
        if not relative or relative != order.scope_files[0]:
            self._refuse(
                result,
                MaintenanceExecutionStatus.MALFORMED_WORK_ORDER,
                f"scope path {order.scope_files[0]!r} is not a safe repository-relative path",
            )
            return None
        if Path(relative).suffix.lower() not in SUPPORTED_SUFFIXES:
            self._refuse(
                result,
                MaintenanceExecutionStatus.MALFORMED_WORK_ORDER,
                f"'{relative}' is not a supported file type "
                f"({', '.join(sorted(SUPPORTED_SUFFIXES))})",
            )
            return None
        snapshot = getattr(order, "candidate_snapshot", None)
        if not isinstance(snapshot, Mapping) or not snapshot:
            self._refuse(
                result,
                MaintenanceExecutionStatus.MALFORMED_WORK_ORDER,
                "work order carries no candidate snapshot, so its claims cannot be "
                "re-checked against the policy",
            )
            return None
        candidate = MaintenanceCandidate.from_dict(snapshot)
        if candidate.candidate_id != order.candidate_id:
            self._refuse(
                result,
                MaintenanceExecutionStatus.MALFORMED_WORK_ORDER,
                "the work order's candidate snapshot does not match its candidate id",
            )
            return None
        # Phase 4.23: re-derive the identity instead of trusting the persisted
        # one. ``MaintenanceCandidate.__post_init__`` keeps a stored
        # ``candidate_id`` verbatim and only computes one when it is missing,
        # so the comparison above proves the snapshot and the order agree with
        # each other - not that either agrees with the snapshot's *contents*.
        # Recomputing binds the id to ``(kind, subject)``, so a persisted
        # record whose kind or subject was edited is refused here rather than
        # being carried into an execution under a borrowed identity.
        derived = candidate_identity(
            candidate.kind, candidate.subject or candidate.title
        )
        if derived != candidate.candidate_id:
            self._refuse(
                result,
                MaintenanceExecutionStatus.MALFORMED_WORK_ORDER,
                "the candidate snapshot's id is not the identity its own kind and "
                "subject produce; the record has been edited or was written by an "
                "incompatible schema",
            )
            return None
        if sorted(candidate.affected_files) != sorted(order.scope_files):
            self._refuse(
                result,
                MaintenanceExecutionStatus.MALFORMED_WORK_ORDER,
                "the work order's scope does not match the candidate's affected files",
            )
            return None
        return candidate

    def _effective_limits(
        self, order: MaintenanceWorkOrder, result: MaintenanceExecutionResult
    ) -> dict[str, int] | None:
        """``min`` of every bound in sight. Never widens anything.

        ``effective = min(maintenance ceiling, policy budget, work order)`` for
        each limit, exactly as Part 12 of the specification requires. A zero
        anywhere means "not permitted", and is reported as budget exhaustion
        rather than silently treated as unlimited.
        """
        limits = {
            "max_tool_steps": min(
                MAX_TOOL_STEPS_CEILING,
                self.budget.max_tool_steps_per_subtask,
                order.max_tool_steps,
            ),
            # The work order carries no candidate-iteration field of its own
            # (Phase 4.21 did not model one), so the bound is the executor's
            # ceiling against the run budget. Adding a third, invented term here
            # would be a bound nobody configured.
            "max_candidate_iterations": min(
                MAX_CANDIDATE_ITERATIONS_CEILING,
                self.budget.max_candidate_iterations,
            ),
            "max_validation_commands": min(
                MAX_VALIDATION_COMMANDS_CEILING,
                self.budget.max_validation_commands,
                order.max_validation_commands,
            ),
            "max_changed_files": min(
                MAX_SCOPE_FILES,
                self.budget.max_changed_files_per_candidate,
                order.max_changed_files,
            ),
            "max_diff_lines": min(
                MAX_DIFF_LINES_CEILING,
                self.budget.max_changed_lines_per_candidate,
            ),
        }
        # This executor performs exactly one implementation, so a work order
        # that permits fewer than one subtask permits nothing. Cheap, exact,
        # and it means the field is a real gate at the only value where it can
        # be one rather than a number nobody reads.
        if int(getattr(order, "max_subtasks", 1) or 0) < 1:
            self._refuse(
                result,
                MaintenanceExecutionStatus.BUDGET_EXHAUSTED,
                "the work order permits no subtasks, and this executor needs "
                "exactly one implementation",
            )
            return None
        for name, value in sorted(limits.items()):
            if value <= 0:
                self._refuse(
                    result,
                    MaintenanceExecutionStatus.BUDGET_EXHAUSTED,
                    f"effective budget '{name}' is {value}; no execution is permitted",
                )
                return None
        # Routed through ``_out_of_time`` rather than calling the ledger and
        # the deadline directly. Phase 4.22 called them inline, so a ledger or
        # deadline that *raised* escaped to the top-level handler and was
        # reported as an implementation failure - a corrupted budget control
        # was misfiled as a bug in the repair, and the operator lost the one
        # signal that says "your budget bookkeeping is broken".
        if self._out_of_time():
            self._refuse(
                result,
                MaintenanceExecutionStatus.BUDGET_EXHAUSTED,
                "the run's time budget is exhausted, or its budget ledger could "
                "not be consulted; no execution is permitted",
            )
            return None
        result.reasons.append(
            "effective bounds: "
            + ", ".join(f"{name}={limits[name]}" for name in sorted(limits))
        )
        return limits

    def _check_freshness(
        self,
        order: MaintenanceWorkOrder,
        oracle: ExecutionOracle,
        relative: str,
        result: MaintenanceExecutionResult,
    ) -> OracleObservation | None:
        """Refuse to execute a plan made against a different repository state.

        Three independent checks, cheapest first:

        1. the target file still exists;
        2. its content fingerprint still matches the one recorded at plan time
           (when the planner supplied one) - this is the general TOCTOU guard
           and reuses :func:`~local_agent.evidence.compute_state_fingerprint`
           rather than inventing a second hashing scheme;
        3. the oracle's *failure predicate* still holds. This is the BEFORE
           observation, and it is returned rather than discarded: the AFTER
           observation is evaluated against it, so the post-condition can be
           "the defect is gone **and** the module survived" rather than the
           weaker "the defect is gone".

        Returns the BEFORE observation, or ``None`` when execution is refused.
        An *inconclusive* observation refuses: an oracle that cannot establish
        the defect is present now will not be able to establish it is gone
        later, and executing against a repository the oracle cannot read is
        exactly the case that must fail closed.
        """
        target = self.root / relative
        if not target.is_file():
            self._refuse(
                result,
                MaintenanceExecutionStatus.STALE_CANDIDATE,
                f"'{relative}' no longer exists in the repository",
            )
            return None

        expected = str(getattr(order, "scope_fingerprint", "") or "")
        if expected:
            observed = compute_state_fingerprint(self.root, [relative])
            if observed != expected:
                self._refuse(
                    result,
                    MaintenanceExecutionStatus.STALE_CANDIDATE,
                    f"'{relative}' changed between planning and execution "
                    f"(fingerprint {expected[:12]} -> {observed[:12]})",
                )
                return None

        before = oracle.observe_failure(self.root, relative)
        result.oracle_precondition = before.to_dict()
        if not before.failing:
            self._refuse(
                result,
                MaintenanceExecutionStatus.STALE_CANDIDATE,
                f"oracle '{oracle.name}' did not re-observe the failure "
                f"({before.outcome}): {before.detail}",
            )
            return None
        result.reasons.append(
            f"signal re-observed at execution time by oracle '{oracle.name}': "
            f"{before.detail}"
        )
        return before

    def _execution_key(
        self,
        order: MaintenanceWorkOrder,
        candidate: MaintenanceCandidate,
        relative: str,
    ) -> str:
        """Strongest identity available for "this exact work, on this exact tree"."""
        policy_fingerprint = compute_policy_fingerprint(
            {
                "executor_version": MAINTENANCE_EXECUTOR_VERSION,
                "configured_tier": self.configured_tier,
                "granted_tier": order.granted_tier,
                "budget": self.budget.to_dict(),
                "supported_signals": sorted(SUPPORTED_SIGNAL_KINDS),
                # A changed oracle contract is a changed decision about what
                # counts as success, so a key minted under one generation must
                # not be honoured by another.
                "oracle_framework": ORACLE_FRAMEWORK_VERSION,
                "oracle": oracle_for(candidate.kind).name,
            }
        )
        tree = compute_state_fingerprint(self.root, [relative])
        payload = "\x1f".join(
            [
                candidate.candidate_id,
                candidate.kind,
                candidate.subject,
                tree,
                policy_fingerprint,
                MAINTENANCE_EXECUTOR_VERSION,
            ]
        )
        return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()[:32]

    def _build_plan(
        self, order: MaintenanceWorkOrder, oracle: ExecutionOracle, relative: str
    ) -> Plan:
        """A plan whose allowed scope is exactly one file.

        The prose comes from the oracle, so the instructions given to the
        implementing agent and the predicate that will judge its work are
        written in one place and cannot drift apart. ``files_likely_to_create``
        is empty on purpose: the executor repairs an existing file and may
        never create one, and ``CodingAgent.prepare`` enforces that from the
        plan without needing a second rule here.
        """
        fragment = oracle.plan_fragment(relative)
        return Plan(
            objective=order.objective or str(fragment.get("objective") or relative),
            files_to_inspect=[relative],
            files_likely_to_change=[relative],
            files_likely_to_create=[],
            steps=[str(step) for step in fragment.get("steps") or ()],
            validation_strategy=[
                str(item) for item in fragment.get("validation_strategy") or ()
            ],
            risks=[str(item) for item in fragment.get("risks") or ()],
        )

    def _implement(
        self,
        order: MaintenanceWorkOrder,
        oracle: ExecutionOracle,
        plan: Plan,
        relative: str,
        limits: Mapping[str, int],
        result: MaintenanceExecutionResult,
    ) -> tuple[Any | None, EvidenceLedger | None]:
        """Run the existing interactive agent inside a candidate workspace.

        Returns ``(implementation, evidence_ledger)``; ``implementation`` is
        ``None`` on any refusal, and the ledger is handed back rather than
        stashed on ``self`` so two executions can never share evidence.
        """
        self._emit(f"[execute] implementing {relative}")
        filesystem = ProjectFilesystem(self.root)
        workspace = CandidateWorkspace(
            self.root,
            workspace_parent=self.workspace_parent,
            protected_paths=set(self.policy.protected_paths),
            command_timeout_seconds=self.command_timeout_seconds,
            semantic_index=self.semantic_index,
        )
        # ``ProspectiveValidator`` multiplies ``max_targeted_commands`` by up to
        # three when the impact scope is broad (weaker evidence buys *more*
        # validation), and adds one syntax command on top. Dividing by that
        # worst-case multiplier here is what makes the executor's
        # ``max_validation_commands`` an honest bound on what actually runs,
        # rather than a number that the validator can legitimately overshoot -
        # which would then trip the post-hoc budget check and throw away a
        # perfectly good repair.
        command_budget = int(limits["max_validation_commands"])
        validator = ProspectiveValidator(
            max_targeted_commands=max(1, (command_budget - 1) // 3),
            max_static_commands=0,
            enable_static_analysis=False,
            semantic_impact_enabled=True,
        )
        ledger = EvidenceLedger()
        agent = InteractiveCodingAgent(
            filesystem,
            ToolRegistryFactory(self.root, filesystem, self.semantic_index).build(),
            policy=self.tool_policy,
            max_tool_steps=int(limits["max_tool_steps"]),
            sandbox=workspace,
            validator=validator,
            max_candidate_iterations=int(limits["max_candidate_iterations"]),
            evidence_ledger=ledger,
        )
        try:
            provider = self.provider_factory()
        except Exception as exc:  # noqa: BLE001 - provider construction is a provider failure
            self._refuse(
                result,
                MaintenanceExecutionStatus.PROVIDER_FAILURE,
                f"no usable provider: {type(exc).__name__}: {exc}",
            )
            return None, ledger

        # The agent-facing objective is the oracle's, so the instruction and
        # the acceptance predicate are authored together. In particular it is
        # the oracle that tells the agent a deletion will be rejected.
        objective = str(
            oracle.plan_fragment(relative).get("agent_objective")
            or plan.objective
        )
        try:
            context = self.context_provider()
        except Exception as exc:  # noqa: BLE001
            self._refuse(
                result,
                MaintenanceExecutionStatus.IMPLEMENTATION_FAILURE,
                f"could not build repository context: {type(exc).__name__}: {exc}",
            )
            return None, ledger

        try:
            implementation = agent.execute(
                provider=provider,
                task_objective=objective,
                plan=plan,
                context=context,
            )
        except Exception as exc:  # noqa: BLE001
            # ProviderError and its rate-limit/quota subclasses land here. The
            # existing router/fallback machinery owns retrying providers; this
            # executor's only job is to make sure a provider failure leaves the
            # authoritative tree exactly as it found it.
            self._refuse(
                result,
                MaintenanceExecutionStatus.PROVIDER_FAILURE,
                f"provider failed during implementation: {type(exc).__name__}: {exc}",
            )
            return None, ledger

        result.provider = str(getattr(implementation, "provider", "") or "")
        result.model = str(getattr(implementation, "model", "") or "")
        result.tool_steps_used = int(getattr(implementation, "tool_steps_used", 0) or 0)
        result.candidate_iterations = int(
            getattr(implementation, "candidate_iterations", 0) or 0
        )
        result.prospective_commands_run = int(
            getattr(implementation, "validation_commands_run", 0) or 0
        )
        result.inspected_files = list(getattr(implementation, "files_inspected", []) or [])
        result.prospective_validation_passed = bool(
            getattr(implementation, "final_candidate_success", False)
        )

        # Post-hoc budget verification. The interactive agent already enforces
        # its own budgets, so a breach here would be a defect in this wiring -
        # which is exactly why it is checked rather than assumed.
        for measured, limit_name in (
            (result.tool_steps_used, "max_tool_steps"),
            (result.candidate_iterations, "max_candidate_iterations"),
            (result.prospective_commands_run, "max_validation_commands"),
        ):
            if measured > int(limits[limit_name]):
                self._refuse(
                    result,
                    MaintenanceExecutionStatus.BUDGET_EXHAUSTED,
                    f"{limit_name} was exceeded during implementation "
                    f"({measured} > {limits[limit_name]}); refusing to apply",
                )
                return None, ledger

        if not getattr(implementation, "success", False):
            reason = str(getattr(implementation, "termination_reason", "") or "")
            message = str(getattr(implementation, "error_message", "") or reason)
            if reason.startswith("candidate_"):
                status = MaintenanceExecutionStatus.PROSPECTIVE_VALIDATION_FAILED
            elif reason == "provider_error":
                status = MaintenanceExecutionStatus.PROVIDER_FAILURE
            else:
                status = MaintenanceExecutionStatus.IMPLEMENTATION_FAILURE
            self._refuse(result, status, f"implementation did not complete: {message}")
            return None, ledger

        if not result.prospective_validation_passed:
            # Defensive: ``InteractiveCodingAgent`` never returns success with a
            # failed candidate, but the invariant is too important to assume.
            self._refuse(
                result,
                MaintenanceExecutionStatus.PROSPECTIVE_VALIDATION_FAILED,
                "the implementation reported success without a passing candidate "
                "validation; refusing to apply",
            )
            return None, ledger
        return implementation, ledger

    def _check_scope(
        self,
        operations: Sequence[FileOperation],
        plan: Plan,
        result: MaintenanceExecutionResult,
    ) -> bool:
        allowed = plan.allowed_paths
        for operation in operations:
            path = CodingAgent._normalize(str(getattr(operation, "path", "")))
            if path not in allowed:
                self._refuse(
                    result,
                    MaintenanceExecutionStatus.SCOPE_VIOLATION,
                    f"the implementation proposed '{path}', which is outside the "
                    "maintenance scope",
                )
                return False
            if self.policy.is_protected(path) or is_protected_relative_path(path):
                self._refuse(
                    result,
                    MaintenanceExecutionStatus.SCOPE_VIOLATION,
                    f"the implementation proposed the protected file '{path}'",
                )
                return False
            action = str(getattr(operation, "action", "")).lower().strip()
            if action not in {"modify", "write"}:
                self._refuse(
                    result,
                    MaintenanceExecutionStatus.SCOPE_VIOLATION,
                    f"action '{action}' on '{path}' is not permitted for a parse "
                    "repair; only in-place modification is",
                )
                return False
        return True

    def _prepare(
        self,
        operations: Sequence[FileOperation],
        plan: Plan,
        result: MaintenanceExecutionResult,
    ) -> list[Any] | None:
        """Validate the edits against the authoritative tree without writing.

        This is the authoritative protected-path/sandbox enforcement point: the
        earlier check in :meth:`_run` is an efficiency shortcut, this one is the
        one that actually decides, and it is the unmodified ``CodingAgent``.
        """
        agent = CodingAgent(
            ProjectFilesystem(self.root), protected_paths=set(self.policy.protected_paths)
        )
        try:
            return agent.prepare(list(operations), plan)
        except (
            UnsafeModificationError,
            SandboxViolation,
            ProtectedPathError,
            PatchApplicationError,
            OSError,
            ValueError,
        ) as exc:
            self._refuse(
                result,
                MaintenanceExecutionStatus.APPLY_FAILED,
                f"the change could not be prepared against the authoritative tree: {exc}",
            )
            return None

    def _apply_and_validate(
        self,
        order: MaintenanceWorkOrder,
        oracle: ExecutionOracle,
        before: OracleObservation,
        prepared: list[Any],
        relative: str,
        limits: Mapping[str, int],
        evidence_ledger: EvidenceLedger | None,
        result: MaintenanceExecutionResult,
    ) -> None:
        """APPLY -> POST-VALIDATE -> (rollback | complete), with records."""
        lifecycle_id = self._lifecycle_start(order, result)
        base_contents = {
            change.path: change.original for change in prepared
        }
        agent = CodingAgent(
            ProjectFilesystem(self.root), protected_paths=set(self.policy.protected_paths)
        )

        self._lifecycle_advance(
            lifecycle_id,
            result,
            LifecycleState.CANDIDATE_GENERATED,
            LifecycleState.VALIDATED,
            LifecycleState.APPROVED,
            reason="maintenance candidate validated prospectively and approved",
        )

        self._emit(f"[apply] {relative}")
        try:
            with self.repo_lock:
                changed = agent.apply_prepared(prepared)
        except (UnsafeModificationError, SandboxViolation, ProtectedPathError, OSError) as exc:
            self._lifecycle_advance(
                lifecycle_id, result, LifecycleState.FAILED, reason="apply failed"
            )
            self._refuse(
                result,
                MaintenanceExecutionStatus.APPLY_FAILED,
                f"the authoritative apply failed and was rolled back: {exc}",
            )
            return

        result.applied = True
        result.changed_files = list(changed)
        self._lifecycle_advance(
            lifecycle_id, result, LifecycleState.APPLIED, reason="maintenance change applied"
        )

        # Everything from here to the rollback decision runs inside one guard.
        #
        # Real defect, inherited from Phase 4.22 and found by injecting an
        # exception into the post-apply path: the authoritative tree has
        # already been written at this point, and an exception raised by the
        # oracle, the impact analyser, the validation engine or the command
        # runner escaped to ``execute``'s top-level handler - which records a
        # status but performs no rollback. The run then ended with
        # ``implementation_failure``, ``applied=True``, ``rolled_back=False``
        # and the change still on disk. An unreviewed half-applied change is
        # exactly what this executor exists to prevent, so any failure in here
        # is now treated as "not passed", which rolls back.
        overshot = False
        verdict: bool | None = None
        executions: list[Any] = []
        decision = impact = None
        try:
            # ``max_changed_files`` was computed, zero-checked and then never
            # compared to anything in Phase 4.22 - a decorative control. It is
            # now enforced against what the authoritative apply reported.
            overshot = len(changed) > int(limits["max_changed_files"])

            verdict, executions, decision, impact = self._post_apply_validation(
                oracle, changed, base_contents, limits, result
            )

            # The signal's own acceptance criterion, judged by its oracle
            # against the BEFORE observation - never inferred from a command's
            # exit code and never from the agent's own claim of success.
            after = oracle.observe_success(self.root, relative, before)
            result.oracle_postcondition = after.to_dict()
            result.signal_resolved = after.resolved
            if not after.resolved:
                result.reasons.append(
                    f"oracle '{oracle.name}' could not confirm the repair "
                    f"({after.outcome}): {after.detail}"
                )
            if overshot:
                result.errors.append(
                    sanitize_text(
                        f"the apply touched {len(changed)} file(s), over the "
                        f"{limits['max_changed_files']}-file budget"
                    )
                )

            # ``validation_passed`` is the *validation* verdict and nothing
            # else - never a blend of "validation passed" and "the signal went
            # away". The runner's reassessment reads it, and a blended value
            # would tell it the repository is broken when in fact only the
            # repair was ineffective.
            result.validation_passed = verdict
            passed = (
                verdict is True and result.signal_resolved is True and not overshot
            )
        except Exception as exc:  # noqa: BLE001 - must not leave a change behind
            LOGGER.exception("Post-apply evaluation raised after an authoritative apply")
            result.errors.append(
                sanitize_text(
                    "post-apply evaluation failed after the change was applied "
                    f"({type(exc).__name__}: {exc}); rolling back"
                )
            )
            # No verdict was established, and ``None`` is the honest value for
            # that. Leaving a stale ``True`` here would let the runner credit a
            # resolution that was never validated.
            result.validation_passed = None
            result.signal_resolved = False
            passed = False

        self._record_evidence(evidence_ledger, executions, changed, result)
        self._record_telemetry(decision, impact, verdict, executions, result)
        self._lifecycle_iteration(lifecycle_id, decision, verdict, executions, result)

        if passed:
            self._lifecycle_advance(
                lifecycle_id,
                result,
                LifecycleState.POST_VALIDATED,
                LifecycleState.COMPLETED,
                reason="post-apply validation passed and the signal is gone",
            )
            result.status = MaintenanceExecutionStatus.COMPLETED
            result.reasons.append(
                "post-apply validation produced a real PASS verdict and oracle "
                f"'{oracle.name}' confirmed the signal no longer reproduces"
            )
            return

        # Failure: put the tree back. A half-repaired file left behind by an
        # autonomous run is strictly worse than the original defect, because the
        # operator did not choose it.
        self._rollback(agent, prepared, base_contents, result)
        self._lifecycle_advance(
            lifecycle_id,
            result,
            LifecycleState.REPAIR_REQUIRED,
            LifecycleState.FAILED,
            reason="post-apply validation failed; change rolled back",
        )
        if overshot:
            result.status = MaintenanceExecutionStatus.POST_APPLY_BUDGET_BREACH
            result.reasons.append(
                "the apply exceeded the per-candidate changed-file budget; the "
                "change was rolled back rather than credited"
            )
        elif verdict is True:
            # Validation was genuinely fine; the change simply did not fix the
            # thing it was authorised to fix. Reporting that as a validation
            # failure would be a lie about the repository's health.
            result.status = MaintenanceExecutionStatus.SIGNAL_NOT_RESOLVED
            result.reasons.append(
                "post-apply validation passed but the oracle could not confirm the "
                "repair; the change was rolled back rather than credited"
            )
        else:
            result.status = MaintenanceExecutionStatus.POST_VALIDATION_FAILED
        if verdict is None:
            result.reasons.append(
                "no post-apply validation command actually executed, so no verdict "
                "exists; the change was rolled back rather than credited"
            )

    def _post_apply_validation(
        self,
        oracle: ExecutionOracle,
        changed: Sequence[str],
        base_contents: Mapping[str, str | None],
        limits: Mapping[str, int],
        result: MaintenanceExecutionResult,
    ) -> tuple[bool | None, list[Any], Any | None, Any | None]:
        """Run the authoritative validation.

        Returns ``(verdict, executions, decision, impact)``.

        The *scope* is chosen by :class:`ValidationDecisionEngine`, never here.
        This method contributes exactly one thing the engine cannot know about:
        the oracle's own acceptance commands, which for a parse failure is a
        mandatory ``compileall`` over the changed files. They are additive and
        run first - they can only cause more validation to run, never less, and
        never displace what the engine chose.

        ``verdict`` is ``None`` when nothing actually executed. That is not a
        pass, and callers must not treat it as one.
        """
        from .semantic_impact import SemanticChangeImpactAnalyzer
        from .validation import ValidationIntelligence

        changed = [str(path) for path in changed]
        impact: Any | None = None
        try:
            analyzer = SemanticChangeImpactAnalyzer(self.root, semantic_index=None)
            impact = analyzer.analyze(list(changed), base_contents=dict(base_contents))
        except Exception as exc:  # noqa: BLE001 - analysis failure widens, never narrows
            LOGGER.warning("Post-apply impact analysis failed: %s", exc)
            result.reasons.append(
                f"impact analysis failed ({type(exc).__name__}); validation was not narrowed"
            )
            impact = None

        lexical: list[CommandSpec] = []
        try:
            lexical = list(
                ValidationIntelligence(self.root).discover_targeted_commands(changed, None)
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Lexical validation discovery failed: %s", exc)

        decision = None
        selected: list[CommandSpec] = []
        if impact is not None:
            engine = ValidationDecisionEngine(
                min_confidence=self.min_impact_confidence,
                reuse_enabled=False,
                policy_fingerprint=MAINTENANCE_EXECUTOR_VERSION,
            )
            decision = engine.decide(
                impact,
                current_root=self.root,
                lexical_commands=lexical,
                ledger=None,
            )
            selected = list(decision.selected_commands)
            result.validation_scope = decision.scope
            result.validation_confidence = decision.confidence_level
        else:
            selected = list(lexical)
            result.validation_scope = "broad"
            result.validation_confidence = "low"

        commands: list[CommandSpec] = list(oracle.acceptance_commands(self.root, changed))
        seen = {tuple(spec.command) for spec in commands}
        for spec in selected:
            if tuple(spec.command) in seen:
                continue
            seen.add(tuple(spec.command))
            commands.append(spec)
        commands = commands[: int(limits["max_validation_commands"])]

        runner = CommandRunner(self.root, self.command_timeout_seconds)
        executions: list[Any] = []
        executed: list[list[str]] = []
        skipped: list[list[str]] = []
        verdict: bool | None = None
        for spec in commands:
            try:
                execution = runner.run(spec)
            except (UnsafeCommandError, PermissionError) as exc:
                result.reasons.append(f"validation command refused by policy: {exc}")
                skipped.append(list(spec.command))
                continue
            executions.append((spec, execution))
            if _executable_missing(execution):
                # A missing runner is not evidence of anything; it cannot pass
                # and it must not fail the change either. Recorded as skipped
                # so telemetry never presents it as validation coverage.
                result.reasons.append(
                    f"skipped '{spec.display()}': executable not available"
                )
                skipped.append(list(spec.command))
                continue
            executed.append(list(spec.command))
            if not execution.succeeded:
                verdict = False
                result.errors.append(
                    sanitize_text(
                        f"post-apply validation failed: {spec.display()} "
                        f"(exit {execution.exit_code})"
                    )
                )
                break
        if verdict is None and executed:
            verdict = True
        # Phase 4.22 counted "ran" as ``exit_code != 127``, a different
        # predicate from the skip test above, so a genuine exit-127 failure was
        # reported as not having run even though it set the verdict. The two
        # now share one predicate.
        result.post_apply_executed_commands = executed
        result.post_apply_skipped_commands = skipped
        result.post_apply_executed_any = bool(executed)
        result.post_apply_commands_run = len(executed)
        result.post_apply_commands = [list(spec.command) for spec, _ in executions]
        return verdict, executions, decision, impact

    def _rollback(
        self,
        agent: CodingAgent,
        prepared: Sequence[Any],
        base_contents: Mapping[str, str | None],
        result: MaintenanceExecutionResult,
    ) -> None:
        """Restore the pre-apply bytes through the same write pipeline."""
        operations = [
            FileOperation(
                action="modify",
                path=change.path,
                content=base_contents.get(change.path) or "",
                reason="rollback of a maintenance change that failed post-apply validation",
            )
            for change in prepared
            if base_contents.get(change.path) is not None
        ]
        if not operations:
            return
        plan = Plan(
            objective="rollback",
            files_likely_to_change=[op.path for op in operations],
        )
        try:
            with self.repo_lock:
                agent.apply_prepared(agent.prepare(operations, plan))
            result.rolled_back = True
            result.reasons.append(
                "the authoritative tree was restored to its pre-apply contents"
            )
        except Exception as exc:  # noqa: BLE001
            result.errors.append(
                sanitize_text(
                    "ROLLBACK FAILED - the repository still contains the maintenance "
                    f"change and needs manual attention: {type(exc).__name__}: {exc}"
                )
            )

    # -- recording ---------------------------------------------------------

    def _lifecycle_start(
        self,
        order: MaintenanceWorkOrder,
        result: MaintenanceExecutionResult,
    ) -> str:
        if self.lifecycle_manager is None:
            return ""
        try:
            record = self.lifecycle_manager.start(
                task_id=f"maintenance:{order.candidate_id}",
                subtask_id=result.execution_key,
                provider=result.provider,
                model=result.model,
            )
        except Exception as exc:  # noqa: BLE001 - history is not the work
            LOGGER.warning("Could not start a maintenance lifecycle: %s", exc)
            return ""
        result.lifecycle_id = record.lifecycle_id
        result.lifecycle_state = record.state
        return record.lifecycle_id

    def _lifecycle_advance(
        self,
        lifecycle_id: str,
        result: MaintenanceExecutionResult,
        *states: str,
        reason: str = "",
    ) -> None:
        if not lifecycle_id or self.lifecycle_manager is None:
            return
        for state in states:
            try:
                record = self.lifecycle_manager.transition(lifecycle_id, state, reason=reason)
            except Exception as exc:  # noqa: BLE001 - an illegal edge is recorded, not fatal
                LOGGER.warning("Maintenance lifecycle transition refused: %s", exc)
                result.reasons.append(f"lifecycle transition to '{state}' refused: {exc}")
                return
            if record is not None:
                result.lifecycle_state = record.state

    def _lifecycle_iteration(
        self,
        lifecycle_id: str,
        decision: Any | None,
        verdict: bool | None,
        executions: Sequence[Any],
        result: MaintenanceExecutionResult,
    ) -> None:
        if not lifecycle_id or self.lifecycle_manager is None:
            return
        iteration = ValidationIterationRecord(
            iteration_number=1,
            candidate_id=result.candidate_id,
            decision_id=result.decision_id,
            scope=result.validation_scope,
            confidence_level=result.validation_confidence,
            commands=[list(spec.command) for spec, _ in executions],
            validation_result=(
                RESULT_PASSED
                if verdict is True
                else RESULT_FAILED
                if verdict is False
                else RESULT_NOT_RUN
            ),
            validation_stage=STAGE_POST_APPLY,
            apply_result="applied" if result.applied else "not_applied",
            provider=result.provider,
            model=result.model,
            duration_seconds=sum(
                float(getattr(e, "duration_seconds", 0.0) or 0.0) for _, e in executions
            ),
            notes=[f"maintenance signal: {result.signal_kind}"],
        )
        try:
            self.lifecycle_manager.record_iteration(lifecycle_id, iteration)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Could not record a maintenance lifecycle iteration: %s", exc)

    def _record_evidence(
        self,
        ledger: EvidenceLedger | None,
        executions: Sequence[Any],
        changed: Sequence[str],
        result: MaintenanceExecutionResult,
    ) -> None:
        if ledger is None:
            return
        relevant = sorted({str(path) for path in changed})
        for spec, execution in executions:
            if _executable_missing(execution):
                status = STATUS_SKIPPED
            elif execution.succeeded:
                status = STATUS_PASSED
            else:
                status = STATUS_FAILED
            try:
                ledger.record(
                    command=tuple(spec.command),
                    status=status,
                    exit_code=int(execution.exit_code),
                    duration_seconds=float(getattr(execution, "duration_seconds", 0.0) or 0.0),
                    category=getattr(spec, "category", "other"),
                    selected_because=getattr(spec, "reason", "post-apply maintenance validation"),
                    tier="post_apply",
                    impacted_files=relevant,
                    impacted_symbols=[],
                    confidence=result.validation_confidence or "low",
                    stdout=getattr(execution, "stdout", "") or "",
                    stderr=getattr(execution, "stderr", "") or "",
                    candidate_iteration=result.candidate_iterations,
                    environment_root="authoritative",
                    fingerprint=compute_state_fingerprint(self.root, relevant),
                )
                result.evidence_recorded += 1
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Could not record maintenance evidence: %s", exc)

    def _record_telemetry(
        self,
        decision: Any | None,
        impact: Any | None,
        verdict: bool | None,
        executions: Sequence[Any],
        result: MaintenanceExecutionResult,
    ) -> None:
        if self.telemetry_manager is None or decision is None or impact is None:
            return
        from .validation_telemetry import build_decision_record, classify_outcome

        try:
            record = build_decision_record(impact, decision, root=self.root)
            self.telemetry_manager.record_decision(record)
            result.decision_id = record.decision_id
            # ``ran`` must mean "a command genuinely executed", not "a command
            # was selected". Phase 4.22 passed ``bool(executions)``, which is
            # true even when every command was skipped for a missing
            # executable - telemetry would then have claimed validation
            # coverage the run did not have.
            really_ran = bool(result.post_apply_executed_any)
            outcome, quality = classify_outcome(
                scope=decision.scope,
                targeted_ran=really_ran,
                targeted_failed=verdict is False,
                broad_ran=really_ran,
                broad_failed=verdict is False,
            )
            self.telemetry_manager.finalize_decision(
                record.decision_id,
                outcome=outcome,
                decision_quality=quality,
                broad_duration_seconds=sum(
                    float(getattr(e, "duration_seconds", 0.0) or 0.0) for _, e in executions
                ),
            )
        except Exception as exc:  # noqa: BLE001 - telemetry is never the work
            LOGGER.warning("Could not record maintenance validation telemetry: %s", exc)

    # -- small helpers -----------------------------------------------------

    def _out_of_time(self) -> bool:
        """Whether the run's time budget is spent, asked fresh.

        Phase 4.21 found the elapsed-time budget double-counted and Phase 4.22
        fixed the arithmetic, but the executor still only consulted it once,
        before any work started. Since everything between that check and the
        apply can take minutes, the check is now repeated immediately before
        the one action that writes to the repository. A ledger or deadline that
        raises is treated as exhausted: refusing is always safe.
        """
        if self.ledger is not None:
            try:
                if self.ledger.exhausted("max_elapsed_seconds"):
                    return True
            except Exception:  # noqa: BLE001 - a broken ledger stops the work
                return True
        if self.deadline is not None:
            try:
                return bool(self.deadline())
            except Exception:  # noqa: BLE001
                return True
        return False

    def _refuse(
        self, result: MaintenanceExecutionResult, status: str, reason: str
    ) -> str:
        result.status = status
        result.reasons.append(sanitize_text(reason, limit=400))
        self._emit(f"[refused:{status}] {reason}")
        return ""

    def _emit(self, message: str) -> None:
        if self.progress is None:
            return
        try:
            self.progress(message)
        except Exception:  # noqa: BLE001 - progress is cosmetic
            pass


class ToolRegistryFactory:
    """Builds the authoritative-root tool registry the interactive agent needs.

    Trivial, and separate purely so :class:`MaintenanceExecutor` never imports
    :mod:`local_agent.tools` directly - the registry the agent actually uses at
    run time is the *candidate* one supplied by ``CandidateWorkspace``, and this
    one exists only to describe the available tool surface.
    """

    def __init__(self, root: Path, filesystem: ProjectFilesystem, semantic_index: Any | None):
        self.root = root
        self.filesystem = filesystem
        self.semantic_index = semantic_index

    def build(self) -> Any:
        from .tools import ToolRegistry

        return ToolRegistry(
            self.root, filesystem=self.filesystem, semantic_index=self.semantic_index
        )


#: Exit code :class:`~local_agent.commands.CommandRunner` synthesises when it
#: cannot find an executable at all.
_EXECUTABLE_MISSING_EXIT_CODE = 127


def _executable_missing(execution: Any) -> bool:
    """Whether a result means "the tool was not there", not "the tool failed".

    One predicate, used by the verdict, the run counters and the evidence
    ledger alike. Phase 4.22 had two subtly different tests for the same
    question, which made a genuine exit-127 failure count as not-run.

    The stderr clause matters: 127 is also a legitimate exit code for a child
    process, and a real command that fails with 127 must fail the change rather
    than be silently skipped. Only the runner's own synthesised message means
    the executable was never found.
    """
    if int(getattr(execution, "exit_code", 0) or 0) != _EXECUTABLE_MISSING_EXIT_CODE:
        return False
    return "executable not found" in (getattr(execution, "stderr", "") or "")


def _timestamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
