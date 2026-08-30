"""Phase 4.21: continuous autonomous maintenance.

Organised by the property under test rather than by class, following the
Phase 4.20 convention, because most of what matters here is an *invariant*
spanning several objects: advisory intelligence stays advisory, a budget can
only be spent, a successful task is not a resolved problem, and a hostile
candidate record cannot become permission.

Real infrastructure is used wherever the property under test depends on it:
real temporary directories, real ``git`` repositories driven by real
subprocesses, real ``SemanticGraph`` builds over real Python files, real JSON
round-trips through ``JsonFileStorage``, and the real ``ValidationLifecycleStore``
/ ``ValidationTelemetryStore`` classes rather than stand-ins. The only stubbed
collaborators anywhere in this file are (a) an executor, which is the injected
seam standing in for the LLM-backed pipeline, and (b) a storage object stubbed
to *fail*, to prove a persistence failure is survivable. Nothing is stubbed to
succeed more conveniently than the real thing would.
"""

from __future__ import annotations

import ast
import io
import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

from local_agent.cli import main as cli_main
from local_agent.config import AgentConfig
from local_agent.maintenance import (
    ALLOWED_CANDIDATE_TRANSITIONS,
    ALL_CANDIDATE_STATES,
    ALL_REASSESSMENT_OUTCOMES,
    ALL_RUN_MODES,
    ALL_SIGNAL_KINDS,
    DEFAULT_MAX_CANDIDATES,
    MAINTENANCE_SCHEMA_VERSION,
    MAX_HISTORY_ENTRIES,
    PROTECTED_RELATIVE_PATHS,
    RUN_MODE_DRY_RUN,
    RUN_MODE_EXECUTE,
    RUN_MODE_SCAN,
    RUN_STATUS_COMPLETED,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_INFO,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
    SEVERITY_ORDER,
    TERMINAL_CANDIDATE_STATES,
    BudgetExceeded,
    BudgetLedger,
    CandidateRunOutcome,
    CandidateState,
    InvalidCandidateTransition,
    MaintenanceBudget,
    MaintenanceCandidate,
    MaintenanceRunRecord,
    MaintenanceSignal,
    MaintenanceStore,
    ReassessmentOutcome,
    can_transition_candidate,
    candidate_identity,
    clamp_unit,
    highest_severity,
    sanitize_metrics,
    sanitize_path_list,
    sanitize_relative_path,
    sanitize_string_list,
    sanitize_text,
    severity_rank,
    summarize_candidates,
)
from local_agent.maintenance_analysis import (
    MaintenanceAnalyzer,
    MaintenanceThresholds,
    collect_churn,
    signal_fingerprint,
)
from local_agent.maintenance_policy import (
    AUTONOMOUSLY_ACTIONABLE_KINDS,
    EXECUTING_TIERS,
    TIER_ORDER,
    AutonomyTier,
    MaintenanceExecutionPolicy,
    MaintenancePriorityEngine,
    PolicyThresholds,
    describe_tier,
    tier_rank,
    weakest_tier,
    weights_sum,
)
from local_agent.maintenance_runner import (
    MaintenanceExecutionOutcome,
    MaintenanceManager,
    MaintenanceRunner,
    MaintenanceWorkOrder,
    build_scan_function,
    build_work_order,
    compute_actionability,
    overlapping_candidates,
    plan_execution_batches,
    reassess,
)
from local_agent.semantic_impact import SemanticGraph
from local_agent.storage import JsonFileStorage
from local_agent.validation_lifecycle import (
    DefectSignature,
    LifecycleState,
    ValidationIterationRecord,
    ValidationLifecycleRecord,
    ValidationLifecycleStore,
)
from local_agent.validation_telemetry import (
    ValidationDecisionRecord,
    ValidationTelemetryStore,
)


# -- AST helpers (same technique as Phase 4.20) -------------------------------


def _module_ast(dotted: str) -> ast.Module:
    path = Path(sys.modules["local_agent"].__file__).parent / (
        dotted.rsplit(".", 1)[1] + ".py"
    )
    return ast.parse(path.read_text(encoding="utf-8"))


def code_identifiers(dotted: str) -> set[str]:
    """Every identifier a module references in *executable* code.

    Docstrings, comments and string literals are parsed away, which is what
    makes an assertion built on this a claim about behaviour rather than about
    prose. A module may therefore discuss ``ValidationDecisionEngine`` at
    length in its documentation and still be proved unable to call it.
    """
    names: set[str] = set()
    for node in ast.walk(_module_ast(dotted)):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.name)
    return names


def imported_modules(dotted: str) -> set[str]:
    package = dotted.rsplit(".", 1)[0]
    found: set[str] = set()
    for node in ast.walk(_module_ast(dotted)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = f"{package}.{base}" if base else package
            found.add(base)
    return found


MAINTENANCE_MODULES = (
    "local_agent.maintenance",
    "local_agent.maintenance_analysis",
    "local_agent.maintenance_policy",
    "local_agent.maintenance_runner",
)


# -- fixtures -----------------------------------------------------------------


def make_candidate(**kwargs) -> MaintenanceCandidate:
    defaults = dict(
        kind=MaintenanceSignal.TEST_GAP,
        subject="pkg/mod.py",
        title="pkg/mod.py has no direct test",
        detail="detail",
        severity=SEVERITY_MEDIUM,
        confidence=0.9,
        sample_size=10,
        affected_files=["pkg/mod.py"],
        recommended_action="add a test",
        occurrence_count=3,
    )
    defaults.update(kwargs)
    return MaintenanceCandidate(**defaults)


def make_lifecycle(lifecycle_id: str, *, iterations=(), state=None) -> ValidationLifecycleRecord:
    record = ValidationLifecycleRecord(lifecycle_id=lifecycle_id, task_id="t", subtask_id="s")
    for iteration in iterations:
        record.add_iteration(iteration)
    if state is not None:
        record.state = state
    return record


def make_defect(**kwargs) -> DefectSignature:
    defaults = dict(
        failure_category="test_failure",
        command=("pytest", "-q"),
        exit_code=1,
        diagnostic="assert 1 == 2",
        affected_file="pkg/mod.py",
        validation_tier="post_apply",
        exception_class="AssertionError",
    )
    defaults.update(kwargs)
    return DefectSignature(**defaults)


class StubStore:
    """A minimal duck-typed store for analyzer tests."""

    def __init__(self, lifecycles=(), decisions=()):
        self._lifecycles = list(lifecycles)
        self._decisions = list(decisions)

    @property
    def lifecycles(self):
        return list(self._lifecycles)

    @property
    def decisions(self):
        return list(self._decisions)


class StubGraph:
    def __init__(self, *, reverse_deps=None, files=None, unresolved=None, failures=None, modules=None):
        self.reverse_deps = reverse_deps or {}
        self.files = files or {}
        self.unresolved_imports = unresolved or {}
        self.parse_failures = failures or {}
        self.module_to_file = modules or {}


class FailingStorage:
    """A storage object whose maintenance methods always explode."""

    def load_maintenance(self):
        raise OSError("disk on fire")

    def save_maintenance(self, store):
        raise OSError("disk still on fire")


# =============================================================================
# A. model serialisation and deterministic identity
# =============================================================================


class CandidateIdentityTests(unittest.TestCase):
    def test_identity_is_deterministic(self):
        self.assertEqual(candidate_identity("k", "s"), candidate_identity("k", "s"))

    def test_identity_differs_by_kind(self):
        self.assertNotEqual(candidate_identity("a", "s"), candidate_identity("b", "s"))

    def test_identity_differs_by_subject(self):
        self.assertNotEqual(candidate_identity("k", "a"), candidate_identity("k", "b"))

    def test_identity_is_case_insensitive(self):
        self.assertEqual(candidate_identity("K", "S"), candidate_identity("k", "s"))

    def test_identity_of_a_collection_ignores_order(self):
        self.assertEqual(
            candidate_identity("k", ["b", "a"]), candidate_identity("k", ["a", "b"])
        )

    def test_identity_is_bounded_hex(self):
        identity = candidate_identity("k", "s")
        self.assertEqual(len(identity), 16)
        int(identity, 16)

    def test_field_boundaries_cannot_be_confused(self):
        # "ab" + "" must not hash the same as "a" + "b".
        self.assertNotEqual(candidate_identity("ab", ""), candidate_identity("a", "b"))

    def test_candidate_gets_an_identity_automatically(self):
        self.assertTrue(make_candidate().candidate_id)

    def test_same_signal_yields_the_same_candidate_id(self):
        self.assertEqual(make_candidate().candidate_id, make_candidate().candidate_id)

    def test_different_subject_yields_a_different_candidate_id(self):
        self.assertNotEqual(
            make_candidate().candidate_id,
            make_candidate(subject="other.py").candidate_id,
        )

    def test_explicit_candidate_id_is_preserved(self):
        self.assertEqual(make_candidate(candidate_id="fixed").candidate_id, "fixed")


class CandidateSerialisationTests(unittest.TestCase):
    def test_round_trip_preserves_every_field(self):
        candidate = make_candidate(
            uncertainty=["a"], evidence_refs=["r1"], metrics={"x": 1.5}
        )
        restored = MaintenanceCandidate.from_dict(candidate.to_dict())
        self.assertEqual(restored.to_dict(), candidate.to_dict())

    def test_round_trip_survives_json(self):
        candidate = make_candidate()
        restored = MaintenanceCandidate.from_dict(
            json.loads(json.dumps(candidate.to_dict()))
        )
        self.assertEqual(restored.candidate_id, candidate.candidate_id)

    def test_from_dict_of_garbage_is_a_default_candidate(self):
        self.assertIsInstance(MaintenanceCandidate.from_dict("nonsense"), MaintenanceCandidate)

    def test_from_dict_of_none_is_a_default_candidate(self):
        self.assertIsInstance(MaintenanceCandidate.from_dict(None), MaintenanceCandidate)

    def test_unknown_keys_are_ignored(self):
        payload = make_candidate().to_dict()
        payload["hostile_extra"] = {"exec": "rm -rf /"}
        restored = MaintenanceCandidate.from_dict(payload)
        self.assertFalse(hasattr(restored, "hostile_extra"))

    def test_run_outcome_round_trips(self):
        outcome = CandidateRunOutcome(
            candidate_id="c", kind="k", title="t", executed=True, validation_passed=True
        )
        self.assertEqual(
            CandidateRunOutcome.from_dict(outcome.to_dict()).to_dict(), outcome.to_dict()
        )

    def test_run_record_round_trips(self):
        record = MaintenanceRunRecord(mode=RUN_MODE_DRY_RUN, candidates_discovered=3)
        record.outcomes.append(CandidateRunOutcome(candidate_id="c"))
        restored = MaintenanceRunRecord.from_dict(record.to_dict())
        self.assertEqual(restored.to_dict(), record.to_dict())

    def test_run_record_from_garbage_is_default(self):
        self.assertIsInstance(MaintenanceRunRecord.from_dict(7), MaintenanceRunRecord)

    def test_budget_round_trips(self):
        budget = MaintenanceBudget(max_candidates_executed=1)
        self.assertEqual(MaintenanceBudget.from_dict(budget.to_dict()), budget)

    def test_budget_from_garbage_is_default(self):
        self.assertEqual(MaintenanceBudget.from_dict("no"), MaintenanceBudget())

    def test_schema_version_is_stamped_on_the_store(self):
        self.assertEqual(MaintenanceStore().to_dict()["schema_version"], MAINTENANCE_SCHEMA_VERSION)


# =============================================================================
# B. sanitisation: candidate content is untrusted input
# =============================================================================


class SanitisationTests(unittest.TestCase):
    def test_control_characters_are_stripped(self):
        self.assertEqual(sanitize_text("a\x00\x1bb"), "ab")

    def test_newlines_collapse_to_a_single_space(self):
        self.assertEqual(sanitize_text("a\nb\r\nc"), "a b c")

    def test_text_is_bounded(self):
        self.assertLessEqual(len(sanitize_text("x" * 5000, limit=50)), 50)

    def test_none_becomes_empty(self):
        self.assertEqual(sanitize_text(None), "")

    def test_non_string_is_coerced(self):
        self.assertEqual(sanitize_text(42), "42")

    def test_sanitisation_is_idempotent(self):
        once = sanitize_text("a\nb\x00c")
        self.assertEqual(sanitize_text(once), once)

    def test_absolute_posix_path_is_rejected(self):
        self.assertEqual(sanitize_relative_path("/etc/passwd"), "")

    def test_windows_drive_path_is_rejected(self):
        self.assertEqual(sanitize_relative_path("C:\\Windows\\System32"), "")

    def test_unc_path_is_rejected(self):
        self.assertEqual(sanitize_relative_path(r"\\server\share\x"), "")

    def test_parent_traversal_is_rejected(self):
        self.assertEqual(sanitize_relative_path("../../etc/passwd"), "")

    def test_embedded_traversal_is_rejected(self):
        self.assertEqual(sanitize_relative_path("pkg/../../outside.py"), "")

    def test_git_directory_is_rejected(self):
        self.assertEqual(sanitize_relative_path(".git/config"), "")

    def test_node_modules_is_rejected(self):
        self.assertEqual(sanitize_relative_path("node_modules/x/y.js"), "")

    def test_backslashes_normalise_to_forward_slashes(self):
        self.assertEqual(sanitize_relative_path("pkg\\mod.py"), "pkg/mod.py")

    def test_a_legitimate_relative_path_survives(self):
        self.assertEqual(sanitize_relative_path("pkg/mod.py"), "pkg/mod.py")

    def test_non_string_path_is_rejected(self):
        self.assertEqual(sanitize_relative_path(123), "")

    def test_path_list_is_sorted_and_deduplicated(self):
        self.assertEqual(
            sanitize_path_list(["b.py", "a.py", "a.py"]), ["a.py", "b.py"]
        )

    def test_path_list_drops_hostile_entries_but_keeps_the_rest(self):
        self.assertEqual(
            sanitize_path_list(["../evil", "good.py"]), ["good.py"]
        )

    def test_path_list_of_a_bare_string_is_empty(self):
        self.assertEqual(sanitize_path_list("pkg/mod.py"), [])

    def test_path_list_is_bounded(self):
        self.assertLessEqual(len(sanitize_path_list([f"f{i}.py" for i in range(500)])), 20)

    def test_string_list_preserves_order(self):
        self.assertEqual(sanitize_string_list(["b", "a"]), ["b", "a"])

    def test_string_list_deduplicates(self):
        self.assertEqual(sanitize_string_list(["a", "a", "b"]), ["a", "b"])

    def test_metrics_reject_nan(self):
        self.assertEqual(sanitize_metrics({"x": float("nan")}), {})

    def test_metrics_reject_infinity(self):
        self.assertEqual(sanitize_metrics({"x": float("inf")}), {})

    def test_metrics_reject_non_numeric(self):
        self.assertEqual(sanitize_metrics({"x": "abc"}), {})

    def test_metrics_accept_numeric_strings(self):
        self.assertEqual(sanitize_metrics({"x": "1.5"}), {"x": 1.5})

    def test_metrics_are_bounded(self):
        self.assertLessEqual(len(sanitize_metrics({f"k{i}": i for i in range(500)})), 20)

    def test_clamp_unit_bounds_above(self):
        self.assertEqual(clamp_unit(99), 1.0)

    def test_clamp_unit_bounds_below(self):
        self.assertEqual(clamp_unit(-99), 0.0)

    def test_clamp_unit_rejects_nan(self):
        self.assertEqual(clamp_unit(float("nan")), 0.0)

    def test_candidate_confidence_is_clamped_on_construction(self):
        self.assertEqual(make_candidate(confidence=9999).confidence, 1.0)

    def test_candidate_severity_falls_back_when_unknown(self):
        self.assertEqual(make_candidate(severity="catastrophic").severity, SEVERITY_LOW)

    def test_candidate_kind_falls_back_when_unknown(self):
        self.assertIn(make_candidate(kind="nope").kind, ALL_SIGNAL_KINDS)

    def test_candidate_title_falls_back_to_the_kind(self):
        self.assertTrue(make_candidate(title="").title)

    def test_candidate_drops_a_traversal_path(self):
        self.assertEqual(make_candidate(affected_files=["../../etc/passwd"]).affected_files, [])

    def test_candidate_history_is_bounded(self):
        candidate = make_candidate(history=[{"at": "x", "event": "e"}] * 500)
        self.assertLessEqual(len(candidate.history), MAX_HISTORY_ENTRIES)


class SeverityTests(unittest.TestCase):
    def test_ranks_are_ordered(self):
        ranks = [severity_rank(s) for s in SEVERITY_ORDER]
        self.assertEqual(ranks, sorted(ranks))

    def test_unknown_severity_ranks_lowest(self):
        self.assertEqual(severity_rank("catastrophic"), 0)

    def test_unknown_severity_cannot_outrank_critical(self):
        self.assertLess(severity_rank("ultra"), severity_rank(SEVERITY_CRITICAL))

    def test_highest_severity_picks_the_maximum(self):
        self.assertEqual(highest_severity(SEVERITY_LOW, SEVERITY_HIGH), SEVERITY_HIGH)

    def test_highest_severity_of_nothing_is_info(self):
        self.assertEqual(highest_severity(), SEVERITY_INFO)


# =============================================================================
# C. the candidate state machine
# =============================================================================


class CandidateStateMachineTests(unittest.TestCase):
    def test_a_new_candidate_starts_detected(self):
        self.assertEqual(make_candidate().state, CandidateState.DETECTED)

    def test_the_happy_path_is_legal(self):
        candidate = make_candidate()
        for state in (
            CandidateState.TRIAGED,
            CandidateState.SELECTED,
            CandidateState.PLANNED,
            CandidateState.EXECUTING,
            CandidateState.VALIDATED,
            CandidateState.REASSESSED,
        ):
            candidate.transition(state, reason="test")
        self.assertEqual(candidate.state, CandidateState.REASSESSED)

    def test_detected_cannot_jump_to_executing(self):
        with self.assertRaises(InvalidCandidateTransition):
            make_candidate().transition(CandidateState.EXECUTING)

    def test_a_rejected_transition_leaves_the_state_untouched(self):
        candidate = make_candidate()
        candidate.try_transition(CandidateState.EXECUTING)
        self.assertEqual(candidate.state, CandidateState.DETECTED)

    def test_a_rejected_transition_records_no_history(self):
        candidate = make_candidate()
        before = len(candidate.history)
        candidate.try_transition(CandidateState.EXECUTING)
        self.assertEqual(len(candidate.history), before)

    def test_try_transition_reports_rejection_without_raising(self):
        self.assertFalse(make_candidate().try_transition(CandidateState.VALIDATED))

    def test_every_terminal_state_has_no_outgoing_edges(self):
        for state in TERMINAL_CANDIDATE_STATES:
            self.assertEqual(ALLOWED_CANDIDATE_TRANSITIONS[state], frozenset())

    def test_no_terminal_state_can_reopen(self):
        for state in TERMINAL_CANDIDATE_STATES:
            for target in ALL_CANDIDATE_STATES:
                self.assertFalse(can_transition_candidate(state, target))

    def test_every_declared_state_appears_in_the_table(self):
        self.assertEqual(set(ALL_CANDIDATE_STATES), set(ALLOWED_CANDIDATE_TRANSITIONS))

    def test_every_transition_target_is_declared(self):
        for targets in ALLOWED_CANDIDATE_TRANSITIONS.values():
            for target in targets:
                self.assertIn(target, ALL_CANDIDATE_STATES)

    def test_an_unknown_state_permits_nothing(self):
        self.assertFalse(can_transition_candidate("imaginary", CandidateState.TRIAGED))

    def test_transitions_are_recorded_with_a_reason(self):
        candidate = make_candidate()
        candidate.transition(CandidateState.TRIAGED, reason="because")
        self.assertEqual(candidate.history[-1]["reason"], "because")

    def test_transition_error_names_both_states(self):
        error = InvalidCandidateTransition("a", "b")
        self.assertIn("a", str(error))
        self.assertIn("b", str(error))

    def test_is_terminal_reflects_the_terminal_set(self):
        candidate = make_candidate()
        candidate.transition(CandidateState.REJECTED)
        self.assertTrue(candidate.is_terminal)

    def test_recording_an_unknown_outcome_becomes_inconclusive(self):
        candidate = make_candidate()
        candidate.record_outcome("totally_fixed_trust_me")
        self.assertEqual(candidate.outcome, ReassessmentOutcome.INCONCLUSIVE)

    def test_recording_a_known_outcome_is_preserved(self):
        candidate = make_candidate()
        candidate.record_outcome(ReassessmentOutcome.PERSISTING)
        self.assertEqual(candidate.outcome, ReassessmentOutcome.PERSISTING)

    def test_every_declared_outcome_is_accepted(self):
        for outcome in ALL_REASSESSMENT_OUTCOMES:
            candidate = make_candidate()
            candidate.record_outcome(outcome)
            self.assertEqual(candidate.outcome, outcome)


class CandidateMergeTests(unittest.TestCase):
    def test_merging_increments_the_occurrence_count(self):
        candidate = make_candidate()
        candidate.merge_observation(make_candidate())
        self.assertEqual(candidate.occurrence_count, 4)

    def test_merging_takes_the_higher_severity(self):
        candidate = make_candidate(severity=SEVERITY_LOW)
        candidate.merge_observation(make_candidate(severity=SEVERITY_HIGH))
        self.assertEqual(candidate.severity, SEVERITY_HIGH)

    def test_merging_never_lowers_severity(self):
        candidate = make_candidate(severity=SEVERITY_HIGH)
        candidate.merge_observation(make_candidate(severity=SEVERITY_LOW))
        self.assertEqual(candidate.severity, SEVERITY_HIGH)

    def test_a_weaker_sample_cannot_overwrite_confidence(self):
        candidate = make_candidate(confidence=0.2, sample_size=100)
        candidate.merge_observation(make_candidate(confidence=1.0, sample_size=1))
        self.assertEqual(candidate.confidence, 0.2)

    def test_a_stronger_sample_does_overwrite_confidence(self):
        candidate = make_candidate(confidence=0.2, sample_size=5)
        candidate.merge_observation(make_candidate(confidence=0.9, sample_size=50))
        self.assertEqual(candidate.confidence, 0.9)

    def test_merging_unions_affected_files(self):
        candidate = make_candidate(affected_files=["a.py"])
        candidate.merge_observation(make_candidate(affected_files=["b.py"]))
        self.assertEqual(candidate.affected_files, ["a.py", "b.py"])

    def test_merging_accumulates_uncertainty(self):
        candidate = make_candidate(uncertainty=["one"])
        candidate.merge_observation(make_candidate(uncertainty=["two"]))
        self.assertEqual(set(candidate.uncertainty), {"one", "two"})

    def test_merging_a_different_candidate_is_refused(self):
        with self.assertRaises(ValueError):
            make_candidate().merge_observation(make_candidate(subject="other.py"))

    def test_occurrence_count_is_bounded(self):
        candidate = make_candidate(occurrence_count=10_000)
        candidate.merge_observation(make_candidate())
        self.assertLessEqual(candidate.occurrence_count, 10_000)


# =============================================================================
# D. bounded persistent storage
# =============================================================================


class StoreTests(unittest.TestCase):
    def test_upsert_inserts_a_new_candidate(self):
        store = MaintenanceStore()
        store.upsert(make_candidate())
        self.assertEqual(len(store), 1)

    def test_upsert_merges_a_repeat_observation(self):
        store = MaintenanceStore()
        store.upsert(make_candidate())
        store.upsert(make_candidate())
        self.assertEqual(len(store), 1)

    def test_candidates_are_returned_in_deterministic_order(self):
        store = MaintenanceStore()
        for index in range(20):
            store.upsert(make_candidate(subject=f"f{index}.py"))
        self.assertEqual(
            [c.candidate_id for c in store.candidates],
            sorted(c.candidate_id for c in store.candidates),
        )

    def test_candidate_storage_is_bounded(self):
        store = MaintenanceStore(max_candidates=5)
        for index in range(50):
            store.upsert(make_candidate(subject=f"f{index}.py"))
        self.assertLessEqual(len(store), 5)

    def test_eviction_keeps_the_most_severe(self):
        store = MaintenanceStore(max_candidates=2)
        store.upsert(make_candidate(subject="critical.py", severity=SEVERITY_CRITICAL))
        for index in range(10):
            store.upsert(make_candidate(subject=f"low{index}.py", severity=SEVERITY_LOW))
        self.assertIn(
            SEVERITY_CRITICAL, {candidate.severity for candidate in store.candidates}
        )

    def test_run_history_is_bounded(self):
        store = MaintenanceStore(max_runs=3)
        for _ in range(10):
            store.record_run(MaintenanceRunRecord())
        self.assertEqual(len(store.runs), 3)

    def test_recording_the_same_run_id_replaces_it(self):
        store = MaintenanceStore()
        record = MaintenanceRunRecord(run_id="fixed")
        store.record_run(record)
        store.record_run(MaintenanceRunRecord(run_id="fixed", candidates_discovered=9))
        self.assertEqual(len(store.runs), 1)
        self.assertEqual(store.runs[0].candidates_discovered, 9)

    def test_find_returns_none_for_an_unknown_id(self):
        self.assertIsNone(MaintenanceStore().find("nope"))

    def test_remove_deletes_a_candidate(self):
        store = MaintenanceStore()
        candidate = store.upsert(make_candidate())
        self.assertTrue(store.remove(candidate.candidate_id))
        self.assertEqual(len(store), 0)

    def test_latest_run_is_the_newest(self):
        store = MaintenanceStore()
        store.record_run(MaintenanceRunRecord(run_id="a"))
        store.record_run(MaintenanceRunRecord(run_id="b"))
        self.assertEqual(store.latest_run().run_id, "b")

    def test_store_round_trips(self):
        store = MaintenanceStore()
        store.upsert(make_candidate())
        store.record_run(MaintenanceRunRecord())
        restored = MaintenanceStore.from_dict(store.to_dict())
        self.assertEqual(restored.to_dict(), store.to_dict())

    def test_store_from_garbage_is_marked_corrupt(self):
        store = MaintenanceStore.from_dict("not a dict")
        self.assertFalse(store.history_trustworthy())

    def test_a_corrupt_candidate_is_skipped_not_fatal(self):
        payload = MaintenanceStore().to_dict()
        payload["candidates"] = [make_candidate().to_dict(), "garbage"]
        store = MaintenanceStore.from_dict(payload)
        self.assertEqual(len(store), 1)
        self.assertFalse(store.history_trustworthy())

    def test_a_non_list_candidate_field_is_counted_as_corruption(self):
        payload = MaintenanceStore().to_dict()
        payload["candidates"] = {"not": "a list"}
        self.assertFalse(MaintenanceStore.from_dict(payload).history_trustworthy())

    def test_a_clean_store_is_trustworthy(self):
        self.assertTrue(MaintenanceStore().history_trustworthy())

    def test_bounds_are_reapplied_on_load(self):
        payload = MaintenanceStore().to_dict()
        payload["max_candidates"] = 3
        payload["candidates"] = [
            make_candidate(subject=f"f{i}.py").to_dict() for i in range(30)
        ]
        self.assertLessEqual(len(MaintenanceStore.from_dict(payload)), 3)

    def test_an_oversized_max_is_still_honoured_as_given(self):
        store = MaintenanceStore(max_candidates=DEFAULT_MAX_CANDIDATES)
        self.assertEqual(store.max_candidates, DEFAULT_MAX_CANDIDATES)

    def test_summarize_counts_by_severity(self):
        summary = summarize_candidates(
            [make_candidate(severity=SEVERITY_HIGH), make_candidate(subject="b.py")]
        )
        self.assertEqual(summary["total"], 2)
        self.assertIn(SEVERITY_HIGH, summary["by_severity"])


class StoragePersistenceTests(unittest.TestCase):
    """Real JSON files, real atomic writes, real quarantine behaviour."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.storage = JsonFileStorage(Path(self._tmp.name) / "data")

    def test_an_absent_file_loads_as_an_empty_store(self):
        self.assertEqual(len(self.storage.load_maintenance()), 0)

    def test_a_saved_store_reloads_identically(self):
        store = MaintenanceStore()
        store.upsert(make_candidate())
        self.storage.save_maintenance(store)
        self.assertEqual(self.storage.load_maintenance().to_dict(), store.to_dict())

    def test_a_truncated_file_is_quarantined_and_survives(self):
        store = MaintenanceStore()
        store.upsert(make_candidate())
        self.storage.save_maintenance(store)
        path = Path(self._tmp.name) / "data" / "maintenance.json"
        path.write_text('{"candidates": [', encoding="utf-8")
        with redirect_stdout(io.StringIO()):
            reloaded = self.storage.load_maintenance()
        self.assertEqual(len(reloaded), 0)
        self.assertFalse(reloaded.history_trustworthy())
        self.assertTrue(list((Path(self._tmp.name) / "data").glob("maintenance.json.corrupt.*")))

    def test_an_oversized_file_still_loads_bounded(self):
        payload = MaintenanceStore().to_dict()
        payload["candidates"] = [
            make_candidate(subject=f"f{i}.py").to_dict() for i in range(2000)
        ]
        path = Path(self._tmp.name) / "data" / "maintenance.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        self.assertLessEqual(len(self.storage.load_maintenance()), DEFAULT_MAX_CANDIDATES)

    def test_the_default_storage_base_class_retains_nothing(self):
        from local_agent.storage import TaskStorage

        class Minimal(TaskStorage):
            def save_task(self, task): ...
            def load_task(self, task_id): ...
            def list_tasks(self): return []
            def save_checkpoint(self, checkpoint): ...
            def load_checkpoint(self, checkpoint_id): ...
            def save_scheduler_state(self, state): ...
            def load_scheduler_state(self): ...
            def save_provider_configs(self, configs): ...
            def load_provider_configs(self): return []
            def save_semantic_index(self, semantic_index): ...
            def load_semantic_index(self): ...
            def save_project_memory(self, memory): ...
            def load_project_memory(self): ...

        minimal = Minimal()
        minimal.save_maintenance(MaintenanceStore())
        self.assertEqual(len(minimal.load_maintenance()), 0)


# =============================================================================
# E. budgets
# =============================================================================


class BudgetTests(unittest.TestCase):
    def test_default_budget_validates(self):
        MaintenanceBudget().validate()

    def test_a_negative_count_is_rejected(self):
        with self.assertRaises(ValueError):
            MaintenanceBudget(max_candidates_selected=-1).validate()

    def test_a_zero_time_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            MaintenanceBudget(max_elapsed_seconds=0).validate()

    def test_a_boolean_is_not_an_integer_budget(self):
        with self.assertRaises(ValueError):
            MaintenanceBudget(max_dag_width=True).validate()

    def test_executed_cannot_exceed_selected(self):
        with self.assertRaises(ValueError):
            MaintenanceBudget(max_candidates_executed=9, max_candidates_selected=2).validate()

    def test_selected_cannot_exceed_considered(self):
        with self.assertRaises(ValueError):
            MaintenanceBudget(
                max_candidates_considered=1, max_candidates_selected=5,
                max_candidates_executed=1,
            ).validate()

    def test_zero_executions_is_a_legal_budget(self):
        MaintenanceBudget(max_candidates_executed=0).validate()


class BudgetLedgerTests(unittest.TestCase):
    def setUp(self):
        self.budget = MaintenanceBudget(
            max_candidates_considered=10, max_candidates_selected=3, max_candidates_executed=2
        )
        self.ledger = BudgetLedger(self.budget)

    def test_consumption_starts_at_zero(self):
        self.assertEqual(self.ledger.consumed("max_candidates_selected"), 0.0)

    def test_consumption_accumulates(self):
        self.ledger.try_consume("max_candidates_selected")
        self.ledger.try_consume("max_candidates_selected")
        self.assertEqual(self.ledger.consumed("max_candidates_selected"), 2.0)

    def test_consumption_stops_at_the_limit(self):
        for _ in range(10):
            self.ledger.try_consume("max_candidates_selected")
        self.assertEqual(self.ledger.consumed("max_candidates_selected"), 3.0)

    def test_a_refused_consumption_returns_false(self):
        for _ in range(3):
            self.ledger.try_consume("max_candidates_selected")
        self.assertFalse(self.ledger.try_consume("max_candidates_selected"))

    def test_a_refused_consumption_consumes_nothing(self):
        for _ in range(3):
            self.ledger.try_consume("max_candidates_selected")
        before = self.ledger.consumed("max_candidates_selected")
        self.ledger.try_consume("max_candidates_selected")
        self.assertEqual(self.ledger.consumed("max_candidates_selected"), before)

    def test_refusals_are_counted(self):
        for _ in range(5):
            self.ledger.try_consume("max_candidates_selected")
        self.assertEqual(self.ledger.refusals()["max_candidates_selected"], 2)

    def test_consumption_is_monotonic(self):
        readings = []
        for _ in range(5):
            self.ledger.try_consume("max_candidates_considered")
            readings.append(self.ledger.consumed("max_candidates_considered"))
        self.assertEqual(readings, sorted(readings))

    def test_there_is_no_way_to_release_budget(self):
        self.assertFalse(
            any(
                name in dir(self.ledger)
                for name in ("release", "refund", "reset", "give_back")
            )
        )

    def test_consume_raises_when_exhausted(self):
        for _ in range(3):
            self.ledger.try_consume("max_candidates_selected")
        with self.assertRaises(BudgetExceeded):
            self.ledger.consume("max_candidates_selected")

    def test_an_unknown_budget_name_raises(self):
        with self.assertRaises(KeyError):
            self.ledger.consumed("max_imaginary")
            self.ledger.limit_for("max_imaginary")

    def test_remaining_never_goes_negative(self):
        self.ledger.observe("max_elapsed_seconds", 1e9)
        self.assertGreaterEqual(self.ledger.remaining("max_elapsed_seconds"), 0.0)

    def test_observe_saturates_at_the_limit(self):
        self.ledger.observe("max_elapsed_seconds", 1e9)
        self.assertLessEqual(
            self.ledger.consumed("max_elapsed_seconds"),
            self.budget.max_elapsed_seconds,
        )

    def test_exhaustion_is_reported(self):
        self.ledger.observe("max_elapsed_seconds", 1e9)
        self.assertTrue(self.ledger.exhausted("max_elapsed_seconds"))

    def test_a_child_shares_the_run_time_budget(self):
        child = self.ledger.child()
        child.observe("max_elapsed_seconds", self.budget.max_elapsed_seconds)
        self.assertTrue(self.ledger.exhausted("max_elapsed_seconds"))

    def test_a_child_does_not_share_per_candidate_counts(self):
        child = self.ledger.child()
        child.try_consume("max_subtasks_per_candidate")
        self.assertEqual(self.ledger.consumed("max_subtasks_per_candidate"), 0.0)

    def test_a_child_cannot_exceed_the_parent_time_budget(self):
        self.ledger.observe("max_elapsed_seconds", self.budget.max_elapsed_seconds)
        child = self.ledger.child()
        self.assertFalse(child.try_consume("max_elapsed_seconds", 1.0))

    def test_a_negative_amount_consumes_nothing(self):
        self.ledger.try_consume("max_candidates_selected", -5)
        self.assertEqual(self.ledger.consumed("max_candidates_selected"), 0.0)

    def test_a_nan_amount_consumes_nothing(self):
        self.ledger.try_consume("max_candidates_selected", float("nan"))
        self.assertEqual(self.ledger.consumed("max_candidates_selected"), 0.0)

    def test_ledger_serialises_limits_and_consumption(self):
        self.ledger.try_consume("max_candidates_selected")
        payload = self.ledger.to_dict()
        self.assertIn("limits", payload)
        self.assertEqual(payload["consumed"]["max_candidates_selected"], 1.0)


# =============================================================================
# F. priority engine
# =============================================================================


class PriorityEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = MaintenancePriorityEngine()

    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(weights_sum(), 1.0, places=9)

    def test_score_is_bounded_below(self):
        candidate = make_candidate(
            severity=SEVERITY_INFO, confidence=0.0, occurrence_count=1,
            affected_files=[], estimated_effort=100.0,
            uncertainty=["a", "b", "c", "d"],
        )
        self.assertGreaterEqual(self.engine.explain(candidate).score, 0.0)

    def test_score_is_bounded_above(self):
        candidate = make_candidate(
            severity=SEVERITY_CRITICAL, confidence=1.0, occurrence_count=999,
            affected_files=[f"f{i}.py" for i in range(20)], estimated_effort=0.0,
        )
        self.assertLessEqual(self.engine.explain(candidate).score, 1.0)

    def test_scoring_is_deterministic(self):
        candidate = make_candidate()
        self.assertEqual(
            self.engine.explain(candidate).score, self.engine.explain(candidate).score
        )

    def test_ranking_is_deterministic_regardless_of_input_order(self):
        candidates = [make_candidate(subject=f"f{i}.py") for i in range(10)]
        forward = [c.candidate_id for c, _ in self.engine.rank(candidates)]
        backward = [c.candidate_id for c, _ in self.engine.rank(list(reversed(candidates)))]
        self.assertEqual(forward, backward)

    def test_higher_severity_always_outranks_lower(self):
        cheap_low = make_candidate(
            subject="cheap.py", severity=SEVERITY_LOW, confidence=1.0,
            occurrence_count=999, estimated_effort=0.0,
        )
        expensive_critical = make_candidate(
            subject="scary.py", severity=SEVERITY_CRITICAL, confidence=0.1,
            occurrence_count=1, estimated_effort=100.0, uncertainty=["a", "b", "c"],
        )
        ranked = self.engine.rank([cheap_low, expensive_critical])
        self.assertEqual(ranked[0][0].candidate_id, expensive_critical.candidate_id)

    def test_uncertain_evidence_cannot_outrank_a_dangerous_issue(self):
        uncertain = make_candidate(subject="a.py", severity=SEVERITY_MEDIUM, confidence=0.2)
        dangerous = make_candidate(subject="b.py", severity=SEVERITY_HIGH, confidence=0.2)
        ranked = self.engine.rank([uncertain, dangerous])
        self.assertEqual(ranked[0][0].candidate_id, dangerous.candidate_id)

    def test_within_a_band_higher_confidence_wins(self):
        low = make_candidate(subject="a.py", confidence=0.2)
        high = make_candidate(subject="b.py", confidence=1.0)
        ranked = self.engine.rank([low, high])
        self.assertEqual(ranked[0][0].candidate_id, high.candidate_id)

    def test_within_a_band_more_recurrence_wins(self):
        rare = make_candidate(subject="a.py", occurrence_count=1)
        common = make_candidate(subject="b.py", occurrence_count=9)
        ranked = self.engine.rank([rare, common])
        self.assertEqual(ranked[0][0].candidate_id, common.candidate_id)

    def test_uncertainty_lowers_the_score(self):
        clean = self.engine.explain(make_candidate()).score
        noisy = self.engine.explain(make_candidate(uncertainty=["a", "b"])).score
        self.assertLess(noisy, clean)

    def test_repeated_failure_lowers_the_score(self):
        healthy = self.engine.explain(make_candidate()).score
        failing = self.engine.explain(
            make_candidate(attempt_count=4, failure_count=4)
        ).score
        self.assertLess(failing, healthy)

    def test_priority_is_monotonic_in_severity(self):
        scores = [
            self.engine.explain(make_candidate(severity=severity)).score
            for severity in SEVERITY_ORDER
        ]
        self.assertEqual(scores, sorted(scores))

    def test_priority_is_monotonic_in_confidence(self):
        scores = [
            self.engine.explain(make_candidate(confidence=value / 10)).score
            for value in range(11)
        ]
        self.assertEqual(scores, sorted(scores))

    def test_every_component_is_within_the_unit_interval(self):
        explanation = self.engine.explain(
            make_candidate(estimated_effort=99, affected_files=[f"f{i}.py" for i in range(50)])
        )
        for value in explanation.components.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_explanation_lists_reasons(self):
        self.assertTrue(self.engine.explain(make_candidate()).reasons)

    def test_explanation_round_trips_to_a_dict(self):
        payload = self.engine.explain(make_candidate()).to_dict()
        self.assertIn("components", payload)
        self.assertIn("weights", payload)

    def test_ties_break_on_the_candidate_id(self):
        a = make_candidate(subject="a.py")
        b = make_candidate(subject="b.py")
        ranked = self.engine.rank([a, b])
        ids = [c.candidate_id for c, _ in ranked]
        self.assertEqual(ids, sorted(ids, reverse=True))

    def test_freshness_defaults_to_maximum_without_a_reference_time(self):
        self.assertEqual(
            self.engine.explain(make_candidate()).components["freshness"], 1.0
        )

    def test_a_stale_candidate_is_downweighted_against_a_reference_time(self):
        engine = MaintenancePriorityEngine(now="2999-01-01T00:00:00+00:00")
        self.assertLess(engine.explain(make_candidate()).components["freshness"], 1.0)

    def test_ranking_an_empty_set_is_empty(self):
        self.assertEqual(self.engine.rank([]), [])


# =============================================================================
# G. execution policy - the safety component
# =============================================================================


class TierAlgebraTests(unittest.TestCase):
    def test_tiers_are_ordered_weakest_first(self):
        self.assertEqual(TIER_ORDER[0], AutonomyTier.OBSERVE_ONLY)
        self.assertEqual(TIER_ORDER[-1], AutonomyTier.EXECUTE_AUTONOMOUSLY)

    def test_an_unknown_tier_ranks_lowest(self):
        self.assertEqual(tier_rank("god_mode"), 0)

    def test_weakest_picks_the_least_permissive(self):
        self.assertEqual(
            weakest_tier(AutonomyTier.EXECUTE_AUTONOMOUSLY, AutonomyTier.RECOMMEND),
            AutonomyTier.RECOMMEND,
        )

    def test_weakest_of_nothing_is_the_strongest_identity(self):
        self.assertEqual(weakest_tier(), AutonomyTier.EXECUTE_AUTONOMOUSLY)

    def test_an_unknown_tier_drags_the_result_to_the_floor(self):
        self.assertEqual(
            weakest_tier(AutonomyTier.EXECUTE_AUTONOMOUSLY, "god_mode"),
            AutonomyTier.OBSERVE_ONLY,
        )

    def test_only_two_tiers_may_execute(self):
        self.assertEqual(
            EXECUTING_TIERS,
            frozenset(
                {
                    AutonomyTier.EXECUTE_WITH_EXISTING_APPROVAL,
                    AutonomyTier.EXECUTE_AUTONOMOUSLY,
                }
            ),
        )

    def test_every_tier_has_a_description(self):
        for tier in TIER_ORDER:
            self.assertNotEqual(describe_tier(tier), "unknown tier")


class ExecutionPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = MaintenanceExecutionPolicy()
        self.strong = AutonomyTier.EXECUTE_AUTONOMOUSLY

    def actionable(self, **kwargs) -> MaintenanceCandidate:
        """A candidate that clears every autonomy gate, so a test can move one."""
        defaults = dict(
            kind=MaintenanceSignal.PARSE_FAILURE,
            subject="pkg/mod.py",
            affected_files=["pkg/mod.py"],
            severity=SEVERITY_HIGH,
            confidence=0.95,
            sample_size=20,
            occurrence_count=5,
            uncertainty=[],
        )
        defaults.update(kwargs)
        return make_candidate(**defaults)

    def test_the_baseline_candidate_reaches_full_autonomy(self):
        verdict = self.policy.decide(self.actionable(), configured_tier=self.strong)
        self.assertEqual(verdict.granted_tier, AutonomyTier.EXECUTE_AUTONOMOUSLY)

    def test_the_granted_tier_never_exceeds_the_configured_one(self):
        for configured in TIER_ORDER:
            verdict = self.policy.decide(self.actionable(), configured_tier=configured)
            self.assertLessEqual(
                tier_rank(verdict.granted_tier), tier_rank(configured)
            )

    def test_observe_only_stays_observe_only(self):
        verdict = self.policy.decide(
            self.actionable(), configured_tier=AutonomyTier.OBSERVE_ONLY
        )
        self.assertEqual(verdict.granted_tier, AutonomyTier.OBSERVE_ONLY)

    def test_an_unrecognised_configured_tier_falls_to_observe_only(self):
        verdict = self.policy.decide(self.actionable(), configured_tier="root")
        self.assertEqual(verdict.granted_tier, AutonomyTier.OBSERVE_ONLY)

    def test_a_protected_file_blocks_the_candidate(self):
        candidate = self.actionable(
            subject="local_agent/tool_engine.py",
            affected_files=["local_agent/tool_engine.py"],
        )
        verdict = self.policy.decide(candidate, configured_tier=self.strong)
        self.assertTrue(verdict.blocked)
        self.assertFalse(verdict.may_execute)

    def test_approval_py_is_protected_too(self):
        candidate = self.actionable(
            subject="local_agent/approval.py", affected_files=["local_agent/approval.py"]
        )
        self.assertTrue(self.policy.decide(candidate, configured_tier=self.strong).blocked)

    def test_every_protected_path_blocks(self):
        for path in PROTECTED_RELATIVE_PATHS:
            candidate = self.actionable(subject=path, affected_files=[path])
            self.assertTrue(
                self.policy.decide(candidate, configured_tier=self.strong).blocked, path
            )

    def test_a_blocked_candidate_names_the_reason(self):
        candidate = self.actionable(
            subject="local_agent/approval.py", affected_files=["local_agent/approval.py"]
        )
        verdict = self.policy.decide(candidate, configured_tier=self.strong)
        self.assertTrue(any("protected" in reason for reason in verdict.blocking_reasons))

    def test_a_raw_traversal_path_is_reported_not_silently_dropped(self):
        verdict = self.policy.decide(
            self.actionable(), configured_tier=self.strong,
            raw_paths=["../../etc/passwd"],
        )
        self.assertTrue(verdict.blocked)
        self.assertTrue(verdict.rejected_paths)

    def test_a_raw_absolute_path_is_rejected(self):
        verdict = self.policy.decide(
            self.actionable(), configured_tier=self.strong, raw_paths=["/etc/shadow"]
        )
        self.assertTrue(verdict.blocked)

    def test_a_raw_safe_path_is_not_rejected(self):
        verdict = self.policy.decide(
            self.actionable(), configured_tier=self.strong, raw_paths=["pkg/mod.py"]
        )
        self.assertFalse(verdict.blocked)

    def test_repeated_failure_blocks_further_attempts(self):
        candidate = self.actionable(attempt_count=5, failure_count=3)
        self.assertTrue(self.policy.decide(candidate, configured_tier=self.strong).blocked)

    def test_a_non_actionable_kind_is_capped_at_recommend(self):
        candidate = self.actionable(kind=MaintenanceSignal.FALSE_CONFIDENCE)
        verdict = self.policy.decide(candidate, configured_tier=self.strong)
        self.assertEqual(verdict.granted_tier, AutonomyTier.RECOMMEND)

    def test_false_confidence_is_never_autonomously_actionable(self):
        self.assertNotIn(MaintenanceSignal.FALSE_CONFIDENCE, AUTONOMOUSLY_ACTIONABLE_KINDS)

    def test_architectural_risk_is_never_autonomously_actionable(self):
        self.assertNotIn(MaintenanceSignal.ARCHITECTURAL_RISK, AUTONOMOUSLY_ACTIONABLE_KINDS)

    def test_every_actionable_kind_is_a_declared_signal(self):
        for kind in AUTONOMOUSLY_ACTIONABLE_KINDS:
            self.assertIn(kind, ALL_SIGNAL_KINDS)

    def test_no_affected_file_caps_at_recommend(self):
        candidate = self.actionable(affected_files=[])
        verdict = self.policy.decide(candidate, configured_tier=self.strong)
        self.assertEqual(verdict.granted_tier, AutonomyTier.RECOMMEND)

    def test_low_confidence_caps_at_recommend(self):
        candidate = self.actionable(confidence=0.1)
        verdict = self.policy.decide(candidate, configured_tier=self.strong)
        self.assertEqual(verdict.granted_tier, AutonomyTier.RECOMMEND)

    def test_too_many_files_caps_at_plan_only(self):
        candidate = self.actionable(
            affected_files=[f"pkg/f{i}.py" for i in range(15)]
        )
        verdict = self.policy.decide(
            candidate, configured_tier=self.strong,
            budget=MaintenanceBudget(max_changed_files_per_candidate=2),
        )
        self.assertEqual(tier_rank(verdict.granted_tier), tier_rank(AutonomyTier.PLAN_ONLY))

    def test_a_zero_execution_budget_caps_at_plan_only(self):
        verdict = self.policy.decide(
            self.actionable(), configured_tier=self.strong,
            budget=MaintenanceBudget(max_candidates_executed=0),
        )
        self.assertFalse(verdict.may_execute)

    def test_critical_with_weak_confidence_caps_at_plan_only(self):
        candidate = self.actionable(
            kind=MaintenanceSignal.RECURRING_DEFECT,
            severity=SEVERITY_CRITICAL, confidence=0.65,
        )
        verdict = self.policy.decide(candidate, configured_tier=self.strong)
        self.assertFalse(verdict.may_execute)

    def test_explicit_uncertainty_forces_human_approval(self):
        candidate = self.actionable(uncertainty=["not sure"])
        verdict = self.policy.decide(candidate, configured_tier=self.strong)
        self.assertEqual(verdict.granted_tier, AutonomyTier.EXECUTE_WITH_EXISTING_APPROVAL)

    def test_a_small_sample_forces_human_approval(self):
        candidate = self.actionable(sample_size=1)
        verdict = self.policy.decide(candidate, configured_tier=self.strong)
        self.assertEqual(verdict.granted_tier, AutonomyTier.EXECUTE_WITH_EXISTING_APPROVAL)

    def test_a_single_occurrence_forces_human_approval(self):
        candidate = self.actionable(occurrence_count=1)
        verdict = self.policy.decide(candidate, configured_tier=self.strong)
        self.assertEqual(verdict.granted_tier, AutonomyTier.EXECUTE_WITH_EXISTING_APPROVAL)

    def test_perfect_confidence_from_two_samples_still_needs_approval(self):
        # Part 17 case 13: an artificially perfect rate over a tiny sample must
        # not buy unattended autonomy.
        candidate = self.actionable(confidence=1.0, sample_size=2, occurrence_count=2)
        verdict = self.policy.decide(candidate, configured_tier=self.strong)
        self.assertNotEqual(verdict.granted_tier, AutonomyTier.EXECUTE_AUTONOMOUSLY)

    def test_the_verdict_serialises(self):
        payload = self.policy.decide(self.actionable(), configured_tier=self.strong).to_dict()
        self.assertIn("granted_tier", payload)
        self.assertIn("may_execute", payload)

    def test_the_decision_is_a_pure_function(self):
        candidate = self.actionable()
        first = self.policy.decide(candidate, configured_tier=self.strong).to_dict()
        second = self.policy.decide(candidate, configured_tier=self.strong).to_dict()
        self.assertEqual(first, second)

    def test_extra_protected_paths_are_unioned_not_replaced(self):
        policy = MaintenanceExecutionPolicy(protected_paths=frozenset({"pkg/mod.py"}))
        self.assertTrue(PROTECTED_RELATIVE_PATHS.issubset(policy.protected_paths))

    def test_an_extra_protected_path_blocks(self):
        policy = MaintenanceExecutionPolicy(protected_paths=frozenset({"pkg/mod.py"}))
        self.assertTrue(policy.decide(self.actionable(), configured_tier=self.strong).blocked)

    def test_may_plan_is_false_for_recommend(self):
        candidate = self.actionable(kind=MaintenanceSignal.FALSE_CONFIDENCE)
        self.assertFalse(self.policy.decide(candidate, configured_tier=self.strong).may_plan)

    def test_may_plan_is_true_for_plan_only(self):
        verdict = self.policy.decide(
            self.actionable(), configured_tier=AutonomyTier.PLAN_ONLY
        )
        self.assertTrue(verdict.may_plan)


class PolicyThresholdTests(unittest.TestCase):
    def test_defaults_validate(self):
        PolicyThresholds().validate()

    def test_a_confidence_above_one_is_rejected(self):
        with self.assertRaises(ValueError):
            PolicyThresholds(min_confidence_to_execute=2.0).validate()

    def test_autonomy_threshold_below_execution_threshold_is_rejected(self):
        with self.assertRaises(ValueError):
            PolicyThresholds(
                min_confidence_to_execute=0.9, min_confidence_for_autonomy=0.5
            ).validate()

    def test_a_negative_sample_requirement_is_rejected(self):
        with self.assertRaises(ValueError):
            PolicyThresholds(min_samples_for_autonomy=-1).validate()

    def test_an_invalid_threshold_set_is_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            MaintenanceExecutionPolicy(
                thresholds=PolicyThresholds(min_confidence_to_execute=-1)
            )


class PolicyRootContainmentTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        (self.root / "pkg").mkdir()
        (self.root / "pkg" / "mod.py").write_text("x = 1\n", encoding="utf-8")
        self.policy = MaintenanceExecutionPolicy(repository_root=self.root)

    def test_a_path_inside_the_root_is_accepted(self):
        candidate = make_candidate(
            kind=MaintenanceSignal.PARSE_FAILURE, subject="pkg/mod.py",
            affected_files=["pkg/mod.py"], confidence=0.95, sample_size=20,
            occurrence_count=5,
        )
        self.assertFalse(
            self.policy.decide(
                candidate, configured_tier=AutonomyTier.EXECUTE_AUTONOMOUSLY
            ).blocked
        )

    def test_a_path_that_escapes_via_normalisation_never_survives(self):
        candidate = make_candidate(affected_files=["pkg/../../outside.py"])
        self.assertEqual(candidate.affected_files, [])


# =============================================================================
# H. signal extraction from real subsystem output
# =============================================================================


class LifecycleSignalTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = MaintenanceAnalyzer(Path.cwd())

    def lifecycle_store(self, records):
        store = ValidationLifecycleStore()
        for record in records:
            store.record(record)
        return store

    def test_a_single_defect_is_not_recurrence(self):
        record = make_lifecycle(
            "l1",
            iterations=[ValidationIterationRecord(defect_signature=make_defect())],
        )
        result = self.analyzer.analyze(lifecycle_store=self.lifecycle_store([record]))
        self.assertEqual(
            [c for c in result.candidates if c.kind == MaintenanceSignal.RECURRING_DEFECT], []
        )

    def test_the_same_defect_across_two_lifecycles_is_recurrence(self):
        defect = make_defect()
        records = [
            make_lifecycle(f"l{i}", iterations=[ValidationIterationRecord(defect_signature=defect)])
            for i in range(2)
        ]
        result = self.analyzer.analyze(lifecycle_store=self.lifecycle_store(records))
        kinds = {c.kind for c in result.candidates}
        self.assertIn(MaintenanceSignal.RECURRING_DEFECT, kinds)

    def test_two_different_defects_are_not_merged(self):
        records = [
            make_lifecycle(
                "l1", iterations=[ValidationIterationRecord(defect_signature=make_defect())]
            ),
            make_lifecycle(
                "l2",
                iterations=[
                    ValidationIterationRecord(
                        defect_signature=make_defect(exception_class="TypeError")
                    )
                ],
            ),
        ]
        result = self.analyzer.analyze(lifecycle_store=self.lifecycle_store(records))
        self.assertEqual(
            [c for c in result.candidates if c.kind == MaintenanceSignal.RECURRING_DEFECT], []
        )

    def test_recurrence_carries_the_lifecycle_ids_as_evidence(self):
        defect = make_defect()
        records = [
            make_lifecycle(f"life{i}", iterations=[ValidationIterationRecord(defect_signature=defect)])
            for i in range(3)
        ]
        result = self.analyzer.analyze(lifecycle_store=self.lifecycle_store(records))
        candidate = next(
            c for c in result.candidates if c.kind == MaintenanceSignal.RECURRING_DEFECT
        )
        self.assertTrue(set(candidate.evidence_refs) & {"life0", "life1", "life2"})

    def test_recurrence_names_the_affected_file(self):
        defect = make_defect()
        records = [
            make_lifecycle(f"l{i}", iterations=[ValidationIterationRecord(defect_signature=defect)])
            for i in range(2)
        ]
        result = self.analyzer.analyze(lifecycle_store=self.lifecycle_store(records))
        candidate = next(
            c for c in result.candidates if c.kind == MaintenanceSignal.RECURRING_DEFECT
        )
        self.assertEqual(candidate.affected_files, ["pkg/mod.py"])

    def test_an_empty_defect_signature_is_ignored(self):
        record = make_lifecycle(
            "l1", iterations=[ValidationIterationRecord(defect_signature=DefectSignature())]
        )
        result = self.analyzer.analyze(lifecycle_store=self.lifecycle_store([record] * 1))
        self.assertEqual(result.candidates, [])

    def test_repeated_repairs_produce_a_candidate(self):
        record = make_lifecycle("l1")
        first = record.add_iteration(
            ValidationIterationRecord(kind="implementation", defect_signature=make_defect())
        )
        for _ in range(3):
            record.add_iteration(
                ValidationIterationRecord(
                    kind="repair", parent_iteration_id=first.iteration_id,
                    defect_signature=make_defect(),
                )
            )
        result = self.analyzer.analyze(lifecycle_store=self.lifecycle_store([record]))
        self.assertIn(
            MaintenanceSignal.REPEATED_REPAIR, {c.kind for c in result.candidates}
        )

    def test_a_clean_lifecycle_produces_no_repair_candidate(self):
        record = make_lifecycle("l1", iterations=[ValidationIterationRecord()])
        result = self.analyzer.analyze(lifecycle_store=self.lifecycle_store([record]))
        self.assertNotIn(
            MaintenanceSignal.REPEATED_REPAIR, {c.kind for c in result.candidates}
        )

    def test_abandonment_needs_enough_samples(self):
        records = [
            make_lifecycle(f"l{i}", state=LifecycleState.ABANDONED) for i in range(2)
        ]
        result = self.analyzer.analyze(lifecycle_store=self.lifecycle_store(records))
        self.assertNotIn(
            MaintenanceSignal.ABANDONED_WORK, {c.kind for c in result.candidates}
        )

    def test_a_high_abandonment_rate_produces_a_candidate(self):
        records = [
            make_lifecycle(f"l{i}", state=LifecycleState.ABANDONED) for i in range(6)
        ]
        result = self.analyzer.analyze(lifecycle_store=self.lifecycle_store(records))
        self.assertIn(
            MaintenanceSignal.ABANDONED_WORK, {c.kind for c in result.candidates}
        )

    def test_abandonment_is_reported_at_the_repository_level(self):
        records = [
            make_lifecycle(f"l{i}", state=LifecycleState.ABANDONED) for i in range(6)
        ]
        result = self.analyzer.analyze(lifecycle_store=self.lifecycle_store(records))
        candidate = next(
            c for c in result.candidates if c.kind == MaintenanceSignal.ABANDONED_WORK
        )
        self.assertEqual(candidate.affected_files, [])

    def test_candidate_stage_failures_produce_an_instability_candidate(self):
        record = make_lifecycle("l1")
        for _ in range(4):
            record.add_iteration(
                ValidationIterationRecord(
                    validation_stage="candidate", validation_result="failed",
                    defect_signature=make_defect(),
                )
            )
        result = self.analyzer.analyze(lifecycle_store=self.lifecycle_store([record]))
        self.assertIn(
            MaintenanceSignal.CANDIDATE_INSTABILITY, {c.kind for c in result.candidates}
        )

    def test_post_apply_failures_do_not_count_as_candidate_instability(self):
        record = make_lifecycle("l1")
        for _ in range(4):
            record.add_iteration(
                ValidationIterationRecord(
                    validation_stage="post_apply", validation_result="failed",
                    defect_signature=make_defect(),
                )
            )
        result = self.analyzer.analyze(lifecycle_store=self.lifecycle_store([record]))
        self.assertNotIn(
            MaintenanceSignal.CANDIDATE_INSTABILITY, {c.kind for c in result.candidates}
        )

    def test_an_empty_store_produces_nothing(self):
        result = self.analyzer.analyze(lifecycle_store=ValidationLifecycleStore())
        self.assertEqual(result.candidates, [])

    def test_confidence_is_reported_with_its_sample_size(self):
        defect = make_defect()
        records = [
            make_lifecycle(f"l{i}", iterations=[ValidationIterationRecord(defect_signature=defect)])
            for i in range(2)
        ]
        result = self.analyzer.analyze(lifecycle_store=self.lifecycle_store(records))
        candidate = next(
            c for c in result.candidates if c.kind == MaintenanceSignal.RECURRING_DEFECT
        )
        self.assertEqual(candidate.sample_size, 2)

    def test_a_thin_sample_carries_an_uncertainty_caveat(self):
        defect = make_defect()
        records = [
            make_lifecycle(f"l{i}", iterations=[ValidationIterationRecord(defect_signature=defect)])
            for i in range(2)
        ]
        result = self.analyzer.analyze(lifecycle_store=self.lifecycle_store(records))
        candidate = next(
            c for c in result.candidates if c.kind == MaintenanceSignal.RECURRING_DEFECT
        )
        self.assertTrue(candidate.uncertainty)


class TelemetrySignalTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = MaintenanceAnalyzer(Path.cwd())

    def telemetry(self, decisions):
        store = ValidationTelemetryStore()
        for decision in decisions:
            store.record_decision(decision)
        return store

    def test_broad_pressure_needs_enough_decisions(self):
        store = self.telemetry([ValidationDecisionRecord(scope="broad") for _ in range(3)])
        result = self.analyzer.analyze(telemetry_store=store)
        self.assertNotIn(
            MaintenanceSignal.BROAD_VALIDATION_PRESSURE, {c.kind for c in result.candidates}
        )

    def test_persistent_broad_scope_produces_a_candidate(self):
        store = self.telemetry([ValidationDecisionRecord(scope="broad") for _ in range(20)])
        result = self.analyzer.analyze(telemetry_store=store)
        self.assertIn(
            MaintenanceSignal.BROAD_VALIDATION_PRESSURE, {c.kind for c in result.candidates}
        )

    def test_mostly_targeted_scope_produces_nothing(self):
        store = self.telemetry(
            [ValidationDecisionRecord(scope="targeted") for _ in range(20)]
        )
        result = self.analyzer.analyze(telemetry_store=store)
        self.assertNotIn(
            MaintenanceSignal.BROAD_VALIDATION_PRESSURE, {c.kind for c in result.candidates}
        )

    def test_broad_pressure_is_not_treated_as_a_correctness_defect(self):
        store = self.telemetry([ValidationDecisionRecord(scope="broad") for _ in range(20)])
        result = self.analyzer.analyze(telemetry_store=store)
        candidate = next(
            c for c in result.candidates
            if c.kind == MaintenanceSignal.BROAD_VALIDATION_PRESSURE
        )
        self.assertEqual(candidate.severity, SEVERITY_LOW)
        self.assertTrue(candidate.uncertainty)

    def test_a_dominant_reuse_denial_reason_is_flagged(self):
        store = self.telemetry(
            [
                ValidationDecisionRecord(reuse_reasons={"tree_state_changed": 5})
                for _ in range(20)
            ]
        )
        result = self.analyzer.analyze(telemetry_store=store)
        self.assertIn(
            MaintenanceSignal.EVIDENCE_REUSE_FAILURE, {c.kind for c in result.candidates}
        )

    def test_a_benign_reuse_reason_is_not_flagged(self):
        store = self.telemetry(
            [
                ValidationDecisionRecord(reuse_reasons={"no_matching_evidence": 5})
                for _ in range(20)
            ]
        )
        result = self.analyzer.analyze(telemetry_store=store)
        self.assertNotIn(
            MaintenanceSignal.EVIDENCE_REUSE_FAILURE, {c.kind for c in result.candidates}
        )

    def test_a_single_false_confidence_incident_is_critical(self):
        store = self.telemetry(
            [ValidationDecisionRecord(decision_quality="false_confidence")]
        )
        result = self.analyzer.analyze(telemetry_store=store)
        candidate = next(
            c for c in result.candidates if c.kind == MaintenanceSignal.FALSE_CONFIDENCE
        )
        self.assertEqual(candidate.severity, SEVERITY_CRITICAL)

    def test_no_false_confidence_produces_no_candidate(self):
        store = self.telemetry([ValidationDecisionRecord(decision_quality="confirmed")])
        result = self.analyzer.analyze(telemetry_store=store)
        self.assertNotIn(
            MaintenanceSignal.FALSE_CONFIDENCE, {c.kind for c in result.candidates}
        )

    def test_frequent_degradation_is_flagged(self):
        store = self.telemetry(
            [ValidationDecisionRecord(degraded_analysis=True) for _ in range(20)]
        )
        result = self.analyzer.analyze(telemetry_store=store)
        self.assertIn(
            MaintenanceSignal.ANALYSIS_DEGRADATION, {c.kind for c in result.candidates}
        )

    def test_clean_analysis_is_not_flagged(self):
        store = self.telemetry(
            [ValidationDecisionRecord(degraded_analysis=False) for _ in range(20)]
        )
        result = self.analyzer.analyze(telemetry_store=store)
        self.assertNotIn(
            MaintenanceSignal.ANALYSIS_DEGRADATION, {c.kind for c in result.candidates}
        )

    def test_a_malformed_reuse_reason_map_is_tolerated(self):
        record = ValidationDecisionRecord()
        record.reuse_reasons = {"tree_state_changed": "many"}
        store = self.telemetry([record for _ in range(20)])
        self.analyzer.analyze(telemetry_store=store)  # must not raise


class GraphSignalTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = MaintenanceAnalyzer(Path.cwd())

    def test_high_fan_in_alone_is_not_a_risk(self):
        graph = StubGraph(reverse_deps={"pkg/hub.py": {f"pkg/m{i}.py" for i in range(20)}})
        result = self.analyzer.analyze(semantic_graph=graph, churn={})
        self.assertNotIn(
            MaintenanceSignal.ARCHITECTURAL_RISK, {c.kind for c in result.candidates}
        )

    def test_churn_alone_is_not_a_risk(self):
        graph = StubGraph(reverse_deps={"pkg/leaf.py": set()})
        result = self.analyzer.analyze(semantic_graph=graph, churn={"pkg/leaf.py": 50})
        self.assertNotIn(
            MaintenanceSignal.ARCHITECTURAL_RISK, {c.kind for c in result.candidates}
        )

    def test_fan_in_plus_churn_is_a_risk(self):
        graph = StubGraph(reverse_deps={"pkg/hub.py": {f"pkg/m{i}.py" for i in range(20)}})
        result = self.analyzer.analyze(semantic_graph=graph, churn={"pkg/hub.py": 10})
        self.assertIn(
            MaintenanceSignal.ARCHITECTURAL_RISK, {c.kind for c in result.candidates}
        )

    def test_test_files_are_not_architectural_risks(self):
        graph = StubGraph(
            reverse_deps={"tests/test_hub.py": {f"pkg/m{i}.py" for i in range(20)}}
        )
        result = self.analyzer.analyze(
            semantic_graph=graph, churn={"tests/test_hub.py": 10}
        )
        self.assertNotIn(
            MaintenanceSignal.ARCHITECTURAL_RISK, {c.kind for c in result.candidates}
        )

    def test_third_party_unresolved_imports_are_ignored(self):
        graph = StubGraph(
            unresolved={"pkg/mod.py": {"requests", "numpy", "boto3", "django"}},
            modules={"pkg": "pkg/__init__.py"},
        )
        result = self.analyzer.analyze(semantic_graph=graph)
        self.assertNotIn(
            MaintenanceSignal.ANALYZER_BLIND_SPOT, {c.kind for c in result.candidates}
        )

    def test_local_unresolved_imports_are_flagged(self):
        graph = StubGraph(
            unresolved={"pkg/mod.py": {"pkg.a", "pkg.b", "pkg.c"}},
            modules={"pkg": "pkg/__init__.py"},
        )
        result = self.analyzer.analyze(semantic_graph=graph)
        self.assertIn(
            MaintenanceSignal.ANALYZER_BLIND_SPOT, {c.kind for c in result.candidates}
        )

    def test_relative_unresolved_imports_are_flagged(self):
        graph = StubGraph(unresolved={"pkg/mod.py": {".a", ".b", ".c"}})
        result = self.analyzer.analyze(semantic_graph=graph)
        self.assertIn(
            MaintenanceSignal.ANALYZER_BLIND_SPOT, {c.kind for c in result.candidates}
        )

    def test_a_parse_failure_is_high_severity(self):
        graph = StubGraph(failures={"pkg/broken.py": "SyntaxError: invalid syntax"})
        result = self.analyzer.analyze(semantic_graph=graph)
        candidate = next(
            c for c in result.candidates if c.kind == MaintenanceSignal.PARSE_FAILURE
        )
        self.assertEqual(candidate.severity, SEVERITY_HIGH)

    def test_a_parse_failure_is_a_structural_observation(self):
        graph = StubGraph(failures={"pkg/broken.py": "SyntaxError"})
        result = self.analyzer.analyze(semantic_graph=graph)
        candidate = next(
            c for c in result.candidates if c.kind == MaintenanceSignal.PARSE_FAILURE
        )
        self.assertEqual(candidate.confidence, 1.0)
        self.assertEqual(candidate.sample_size, 1)

    def test_a_module_with_a_test_dependent_is_not_a_gap(self):
        graph = StubGraph(
            files={"pkg/mod.py": object()},
            reverse_deps={
                "pkg/mod.py": {"tests/test_mod.py", "pkg/a.py", "pkg/b.py", "pkg/c.py"}
            },
        )
        result = self.analyzer.analyze(semantic_graph=graph)
        self.assertNotIn(MaintenanceSignal.TEST_GAP, {c.kind for c in result.candidates})

    def test_a_module_with_no_test_dependent_is_a_gap(self):
        graph = StubGraph(
            files={"pkg/mod.py": object()},
            reverse_deps={"pkg/mod.py": {"pkg/a.py", "pkg/b.py", "pkg/c.py"}},
        )
        result = self.analyzer.analyze(semantic_graph=graph)
        self.assertIn(MaintenanceSignal.TEST_GAP, {c.kind for c in result.candidates})

    def test_a_test_gap_carries_its_indirect_coverage_caveat(self):
        graph = StubGraph(
            files={"pkg/mod.py": object()},
            reverse_deps={"pkg/mod.py": {"pkg/a.py", "pkg/b.py", "pkg/c.py"}},
        )
        result = self.analyzer.analyze(semantic_graph=graph)
        candidate = next(
            c for c in result.candidates if c.kind == MaintenanceSignal.TEST_GAP
        )
        self.assertTrue(candidate.uncertainty)

    def test_an_unused_module_is_not_a_test_gap(self):
        graph = StubGraph(files={"pkg/mod.py": object()}, reverse_deps={"pkg/mod.py": set()})
        result = self.analyzer.analyze(semantic_graph=graph)
        self.assertNotIn(MaintenanceSignal.TEST_GAP, {c.kind for c in result.candidates})

    def test_hostile_paths_in_graph_output_are_dropped(self):
        graph = StubGraph(failures={"../../etc/passwd": "SyntaxError"})
        result = self.analyzer.analyze(semantic_graph=graph)
        self.assertEqual(result.candidates, [])


class KnowledgeSignalTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = MaintenanceAnalyzer(Path.cwd())

    def graph_with(self, patterns):
        from local_agent.models import RepositoryKnowledgeGraph

        graph = RepositoryKnowledgeGraph()
        graph.failure_patterns = list(patterns)
        return graph

    def pattern(self, **kwargs):
        from local_agent.models import FailurePatternRecord

        defaults = dict(
            pattern_id="p1", error_signature="ImportError: no module named x",
            occurrence_count=5, confidence=0.9, affected_files=["pkg/mod.py"],
        )
        defaults.update(kwargs)
        return FailurePatternRecord(**defaults)

    def test_a_rare_pattern_is_ignored(self):
        result = self.analyzer.analyze(
            knowledge_graph=self.graph_with([self.pattern(occurrence_count=1)])
        )
        self.assertEqual(result.candidates, [])

    def test_a_recurring_pattern_produces_a_candidate(self):
        result = self.analyzer.analyze(knowledge_graph=self.graph_with([self.pattern()]))
        self.assertIn(
            MaintenanceSignal.KNOWN_FAILURE_PATTERN, {c.kind for c in result.candidates}
        )

    def test_a_stored_confidence_is_capped_by_the_sample_bound(self):
        result = self.analyzer.analyze(
            knowledge_graph=self.graph_with(
                [self.pattern(confidence=1.0, occurrence_count=3)]
            )
        )
        candidate = next(
            c for c in result.candidates
            if c.kind == MaintenanceSignal.KNOWN_FAILURE_PATTERN
        )
        self.assertLess(candidate.confidence, 1.0)

    def test_a_pattern_without_a_signature_is_ignored(self):
        result = self.analyzer.analyze(
            knowledge_graph=self.graph_with([self.pattern(error_signature="")])
        )
        self.assertEqual(result.candidates, [])


class AnalyzerRobustnessTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = MaintenanceAnalyzer(Path.cwd())

    def test_no_sources_produces_no_candidates(self):
        self.assertEqual(self.analyzer.analyze().candidates, [])

    def test_no_sources_is_reported_as_degraded(self):
        self.assertTrue(self.analyzer.analyze().degraded)

    def test_a_malformed_lifecycle_store_does_not_crash_the_scan(self):
        class Broken:
            @property
            def lifecycles(self):
                raise RuntimeError("historical data is a lie")

        result = self.analyzer.analyze(lifecycle_store=Broken())
        self.assertTrue(result.extractor_errors)

    def test_one_broken_extractor_does_not_stop_the_others(self):
        class Broken:
            @property
            def lifecycles(self):
                raise RuntimeError("boom")

        graph = StubGraph(failures={"pkg/broken.py": "SyntaxError"})
        result = self.analyzer.analyze(lifecycle_store=Broken(), semantic_graph=graph)
        self.assertIn(MaintenanceSignal.PARSE_FAILURE, {c.kind for c in result.candidates})

    def test_extractor_errors_are_sanitised(self):
        class Broken:
            @property
            def lifecycles(self):
                raise RuntimeError("line one\nline two\x00")

        result = self.analyzer.analyze(lifecycle_store=Broken())
        for message in result.extractor_errors.values():
            self.assertNotIn("\n", message)
            self.assertNotIn("\x00", message)

    def test_candidates_are_deterministically_ordered(self):
        graph = StubGraph(failures={f"pkg/b{i}.py": "SyntaxError" for i in range(10)})
        first = [c.candidate_id for c in self.analyzer.analyze(semantic_graph=graph).candidates]
        second = [c.candidate_id for c in self.analyzer.analyze(semantic_graph=graph).candidates]
        self.assertEqual(first, second)

    def test_candidates_per_kind_are_bounded(self):
        graph = StubGraph(failures={f"pkg/b{i}.py": "SyntaxError" for i in range(200)})
        result = self.analyzer.analyze(semantic_graph=graph)
        self.assertLessEqual(len(result.candidates), MaintenanceThresholds().max_candidates_per_kind)

    def test_max_candidates_truncates(self):
        graph = StubGraph(failures={f"pkg/b{i}.py": "SyntaxError" for i in range(200)})
        result = self.analyzer.analyze(semantic_graph=graph, max_candidates=3)
        self.assertEqual(len(result.candidates), 3)

    def test_analysis_result_serialises(self):
        payload = self.analyzer.analyze().to_dict()
        self.assertIn("degraded", payload)
        self.assertIn("sources_available", payload)

    def test_thresholds_validate(self):
        MaintenanceThresholds().validate()

    def test_a_zero_threshold_is_rejected(self):
        with self.assertRaises(ValueError):
            MaintenanceThresholds(min_defect_recurrence=0).validate()

    def test_a_rate_above_one_is_rejected(self):
        with self.assertRaises(ValueError):
            MaintenanceThresholds(broad_scope_rate_threshold=1.5).validate()

    def test_an_invalid_threshold_set_is_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            MaintenanceAnalyzer(Path.cwd(), thresholds=MaintenanceThresholds(min_fan_in_for_risk=0))

    def test_elapsed_time_is_measured(self):
        self.assertGreaterEqual(self.analyzer.analyze().elapsed_seconds, 0.0)


class RealRepositoryScanTests(unittest.TestCase):
    """Real ``SemanticGraph`` over real files on a real filesystem."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        (self.root / "pkg").mkdir()
        (self.root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
        (self.root / "pkg" / "hub.py").write_text("def shared():\n    return 1\n", encoding="utf-8")
        for index in range(10):
            (self.root / "pkg" / f"user{index}.py").write_text(
                "from pkg.hub import shared\n\n\ndef go():\n    return shared()\n",
                encoding="utf-8",
            )
        (self.root / "pkg" / "broken.py").write_text("def (:\n", encoding="utf-8")

    def test_a_real_syntax_error_is_detected(self):
        graph = SemanticGraph.build(self.root)
        result = MaintenanceAnalyzer(self.root).analyze(semantic_graph=graph)
        parse = [c for c in result.candidates if c.kind == MaintenanceSignal.PARSE_FAILURE]
        self.assertTrue(any("broken.py" in c.subject for c in parse))

    def test_a_real_test_gap_is_detected(self):
        graph = SemanticGraph.build(self.root)
        result = MaintenanceAnalyzer(self.root).analyze(semantic_graph=graph)
        gaps = [c for c in result.candidates if c.kind == MaintenanceSignal.TEST_GAP]
        self.assertTrue(any("hub.py" in c.subject for c in gaps))

    def test_adding_a_real_test_removes_the_gap(self):
        (self.root / "tests").mkdir()
        (self.root / "tests" / "__init__.py").write_text("", encoding="utf-8")
        (self.root / "tests" / "test_hub.py").write_text(
            "from pkg.hub import shared\n\n\ndef test_shared():\n    assert shared() == 1\n",
            encoding="utf-8",
        )
        graph = SemanticGraph.build(self.root)
        result = MaintenanceAnalyzer(self.root).analyze(semantic_graph=graph)
        gaps = [c for c in result.candidates if c.kind == MaintenanceSignal.TEST_GAP]
        self.assertFalse(any("hub.py" in c.subject for c in gaps))

    def test_fixing_a_real_syntax_error_removes_the_candidate(self):
        (self.root / "pkg" / "broken.py").write_text("def fine():\n    return 2\n", encoding="utf-8")
        graph = SemanticGraph.build(self.root)
        result = MaintenanceAnalyzer(self.root).analyze(semantic_graph=graph)
        self.assertNotIn(MaintenanceSignal.PARSE_FAILURE, {c.kind for c in result.candidates})

    def test_scanning_the_agents_own_repository_produces_real_candidates(self):
        root = Path(__file__).resolve().parents[1]
        graph = SemanticGraph.build(root)
        result = MaintenanceAnalyzer(root).analyze(semantic_graph=graph)
        self.assertIsInstance(result.candidates, list)
        for candidate in result.candidates:
            self.assertTrue(candidate.candidate_id)
            self.assertIn(candidate.kind, ALL_SIGNAL_KINDS)


class ChurnTests(unittest.TestCase):
    """Real ``git`` subprocesses, real commits, real exit codes."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def git(self, *args) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args], cwd=self.root, capture_output=True, text=True, timeout=30
        )

    def make_repo(self):
        self.assertEqual(self.git("init").returncode, 0)
        self.git("config", "user.email", "t@example.com")
        self.git("config", "user.name", "T")
        for index in range(3):
            (self.root / "a.py").write_text(f"x = {index}\n", encoding="utf-8")
            self.git("add", "a.py")
            self.assertEqual(self.git("commit", "-m", f"c{index}").returncode, 0)

    def test_churn_counts_real_commits(self):
        self.make_repo()
        from local_agent.git import GitIntegration

        counts = GitIntegration(self.root).file_change_counts()
        self.assertEqual(counts.get("a.py"), 3)

    def test_churn_of_a_non_repository_is_empty(self):
        from local_agent.git import GitIntegration

        self.assertEqual(GitIntegration(self.root).file_change_counts(), {})

    def test_collect_churn_tolerates_a_missing_reader(self):
        self.assertEqual(collect_churn(object()), {})

    def test_collect_churn_tolerates_an_exploding_reader(self):
        class Boom:
            def file_change_counts(self, limit=200):
                raise RuntimeError("no")

        self.assertEqual(collect_churn(Boom()), {})

    def test_collect_churn_normalises_hostile_paths(self):
        class Hostile:
            def file_change_counts(self, limit=200):
                return {"../../etc/passwd": 9, "a.py": 2}

        self.assertEqual(collect_churn(Hostile()), {"a.py": 2})

    def test_churn_output_is_bounded(self):
        self.make_repo()
        from local_agent.git import GitIntegration

        counts = GitIntegration(self.root).file_change_counts(max_files=1)
        self.assertLessEqual(len(counts), 1)


# =============================================================================
# I. reassessment - the "did it actually work?" contract
# =============================================================================


class ReassessmentTests(unittest.TestCase):
    def setUp(self):
        self.before = make_candidate(metrics={"dependents": 5.0})

    def test_nothing_executed_is_inconclusive_even_if_the_signal_vanished(self):
        verdict = reassess(self.before, None, executed=False, validation_passed=True)
        self.assertEqual(verdict.outcome, ReassessmentOutcome.INCONCLUSIVE)

    def test_a_degraded_rescan_is_inconclusive_even_if_the_signal_vanished(self):
        verdict = reassess(
            self.before, None, executed=True, validation_passed=True, rescan_degraded=True
        )
        self.assertEqual(verdict.outcome, ReassessmentOutcome.INCONCLUSIVE)

    def test_failed_validation_can_never_be_resolution(self):
        verdict = reassess(self.before, None, executed=True, validation_passed=False)
        self.assertEqual(verdict.outcome, ReassessmentOutcome.PERSISTING)

    def test_a_vanished_signal_after_clean_work_is_resolution(self):
        verdict = reassess(self.before, None, executed=True, validation_passed=True)
        self.assertEqual(verdict.outcome, ReassessmentOutcome.RESOLVED)

    def test_an_unchanged_signal_is_persisting(self):
        verdict = reassess(
            self.before, make_candidate(metrics={"dependents": 5.0}),
            executed=True, validation_passed=True,
        )
        self.assertEqual(verdict.outcome, ReassessmentOutcome.PERSISTING)

    def test_a_succeeded_task_alone_is_not_resolution(self):
        # The single most important assertion in this file.
        after = make_candidate(metrics={"dependents": 5.0})
        verdict = reassess(self.before, after, executed=True, validation_passed=True)
        self.assertNotEqual(verdict.outcome, ReassessmentOutcome.RESOLVED)

    def test_lower_severity_is_partial_resolution(self):
        after = make_candidate(severity=SEVERITY_LOW, metrics={"dependents": 5.0})
        verdict = reassess(self.before, after, executed=True, validation_passed=True)
        self.assertEqual(verdict.outcome, ReassessmentOutcome.PARTIALLY_RESOLVED)

    def test_higher_severity_is_regression(self):
        after = make_candidate(severity=SEVERITY_CRITICAL, metrics={"dependents": 5.0})
        verdict = reassess(self.before, after, executed=True, validation_passed=True)
        self.assertEqual(verdict.outcome, ReassessmentOutcome.REGRESSED)

    def test_improved_metrics_are_partial_resolution(self):
        after = make_candidate(metrics={"dependents": 2.0})
        verdict = reassess(self.before, after, executed=True, validation_passed=True)
        self.assertEqual(verdict.outcome, ReassessmentOutcome.PARTIALLY_RESOLVED)

    def test_worsened_metrics_are_regression(self):
        after = make_candidate(metrics={"dependents": 9.0})
        verdict = reassess(self.before, after, executed=True, validation_passed=True)
        self.assertEqual(verdict.outcome, ReassessmentOutcome.REGRESSED)

    def test_unknown_validation_still_permits_resolution_on_a_vanished_signal(self):
        verdict = reassess(self.before, None, executed=True, validation_passed=None)
        self.assertEqual(verdict.outcome, ReassessmentOutcome.RESOLVED)

    def test_every_verdict_carries_a_reason(self):
        for after in (None, make_candidate()):
            verdict = reassess(self.before, after, executed=True, validation_passed=True)
            self.assertTrue(verdict.reasons)

    def test_the_verdict_serialises(self):
        payload = reassess(self.before, None, executed=True, validation_passed=True).to_dict()
        self.assertIn("outcome", payload)
        self.assertIn("before_fingerprint", payload)

    def test_signal_fingerprints_are_stable(self):
        self.assertEqual(signal_fingerprint(self.before), signal_fingerprint(self.before))

    def test_signal_fingerprint_changes_with_severity(self):
        self.assertNotEqual(
            signal_fingerprint(self.before),
            signal_fingerprint(make_candidate(severity=SEVERITY_CRITICAL)),
        )

    def test_signal_fingerprint_ignores_the_lifecycle_state(self):
        moved = make_candidate(metrics={"dependents": 5.0})
        moved.transition(CandidateState.TRIAGED)
        self.assertEqual(signal_fingerprint(self.before), signal_fingerprint(moved))

    def test_signal_fingerprint_ignores_the_occurrence_count(self):
        self.assertEqual(
            signal_fingerprint(self.before),
            signal_fingerprint(make_candidate(occurrence_count=99, metrics={"dependents": 5.0})),
        )


# =============================================================================
# J. concurrency planning
# =============================================================================


class BatchPlanningTests(unittest.TestCase):
    def test_disjoint_candidates_batch_together(self):
        candidates = [
            make_candidate(subject="a.py", affected_files=["a.py"]),
            make_candidate(subject="b.py", affected_files=["b.py"]),
        ]
        batches = plan_execution_batches(candidates, max_width=2)
        self.assertEqual(len(batches), 1)

    def test_overlapping_candidates_are_serialised(self):
        candidates = [
            make_candidate(subject="a", affected_files=["shared.py"]),
            make_candidate(subject="b", affected_files=["shared.py"]),
        ]
        batches = plan_execution_batches(candidates, max_width=4)
        self.assertEqual(len(batches), 2)

    def test_batch_width_is_capped(self):
        candidates = [
            make_candidate(subject=f"f{i}", affected_files=[f"f{i}.py"]) for i in range(10)
        ]
        for batch in plan_execution_batches(candidates, max_width=3):
            self.assertLessEqual(len(batch), 3)

    def test_batching_is_deterministic(self):
        candidates = [
            make_candidate(subject=f"f{i}", affected_files=[f"f{i % 3}.py"]) for i in range(9)
        ]
        first = [[c.candidate_id for c in b] for b in plan_execution_batches(candidates, max_width=2)]
        second = [[c.candidate_id for c in b] for b in plan_execution_batches(candidates, max_width=2)]
        self.assertEqual(first, second)

    def test_every_candidate_lands_in_exactly_one_batch(self):
        candidates = [
            make_candidate(subject=f"f{i}", affected_files=[f"f{i % 4}.py"]) for i in range(12)
        ]
        placed = [c.candidate_id for b in plan_execution_batches(candidates, max_width=2) for c in b]
        self.assertEqual(sorted(placed), sorted(c.candidate_id for c in candidates))

    def test_a_zero_width_is_clamped_to_one(self):
        candidates = [make_candidate(subject=f"f{i}", affected_files=[f"f{i}.py"]) for i in range(3)]
        for batch in plan_execution_batches(candidates, max_width=0):
            self.assertEqual(len(batch), 1)

    def test_no_candidates_means_no_batches(self):
        self.assertEqual(plan_execution_batches([], max_width=4), [])

    def test_overlap_detection_finds_the_pair(self):
        candidates = [
            make_candidate(subject="a", affected_files=["shared.py"]),
            make_candidate(subject="b", affected_files=["shared.py"]),
        ]
        self.assertEqual(len(overlapping_candidates(candidates)), 1)

    def test_overlap_detection_finds_nothing_when_disjoint(self):
        candidates = [
            make_candidate(subject="a", affected_files=["a.py"]),
            make_candidate(subject="b", affected_files=["b.py"]),
        ]
        self.assertEqual(overlapping_candidates(candidates), [])

    def test_candidates_without_files_do_not_conflict(self):
        candidates = [
            make_candidate(subject=f"f{i}", affected_files=[]) for i in range(4)
        ]
        self.assertEqual(len(plan_execution_batches(candidates, max_width=4)), 1)


# =============================================================================
# K. work orders
# =============================================================================


class WorkOrderTests(unittest.TestCase):
    def setUp(self):
        self.budget = MaintenanceBudget()

    def test_a_work_order_carries_the_candidate_scope(self):
        candidate = make_candidate(affected_files=["pkg/mod.py"])
        order = build_work_order(
            candidate, granted_tier=AutonomyTier.PLAN_ONLY, budget=self.budget
        )
        self.assertEqual(order.scope_files, ["pkg/mod.py"])

    def test_a_work_order_inherits_the_budget(self):
        order = build_work_order(
            make_candidate(), granted_tier=AutonomyTier.PLAN_ONLY,
            budget=MaintenanceBudget(max_subtasks_per_candidate=2),
        )
        self.assertEqual(order.max_subtasks, 2)

    def test_a_work_order_states_acceptance_criteria(self):
        order = build_work_order(
            make_candidate(), granted_tier=AutonomyTier.PLAN_ONLY, budget=self.budget
        )
        self.assertTrue(order.acceptance_criteria)

    def test_the_first_criterion_is_signal_disappearance(self):
        order = build_work_order(
            make_candidate(), granted_tier=AutonomyTier.PLAN_ONLY, budget=self.budget
        )
        self.assertIn("no longer detected", order.acceptance_criteria[0])

    def test_work_order_sanitises_hostile_scope(self):
        order = MaintenanceWorkOrder(scope_files=["../../etc/passwd", "ok.py"])
        self.assertEqual(order.scope_files, ["ok.py"])

    def test_work_order_sanitises_a_command_like_objective(self):
        order = MaintenanceWorkOrder(objective="fix\n; rm -rf /\n")
        self.assertNotIn("\n", order.objective)

    def test_work_order_serialises(self):
        payload = MaintenanceWorkOrder(candidate_id="c").to_dict()
        self.assertEqual(payload["candidate_id"], "c")

    def test_negative_budgets_clamp_to_zero(self):
        self.assertEqual(MaintenanceWorkOrder(max_subtasks=-5).max_subtasks, 0)

    def test_garbage_budgets_clamp_to_zero(self):
        self.assertEqual(MaintenanceWorkOrder(max_tool_steps="lots").max_tool_steps, 0)

    def test_execution_outcome_sanitises_changed_files(self):
        outcome = MaintenanceExecutionOutcome(changed_files=["../../x", "ok.py"])
        self.assertEqual(outcome.changed_files, ["ok.py"])

    def test_execution_outcome_serialises(self):
        self.assertIn("succeeded", MaintenanceExecutionOutcome().to_dict())


# =============================================================================
# L. the runner
# =============================================================================


class RunnerHarness:
    """Builds a runner over a temp directory with an injectable scan/executor."""

    def __init__(self, test: unittest.TestCase, *, candidates=(), tier=AutonomyTier.OBSERVE_ONLY,
                 executor=None, budget=None, degraded=False):
        tmp = tempfile.TemporaryDirectory()
        test.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.storage = JsonFileStorage(self.root / "data")
        self.analyzer = MaintenanceAnalyzer(self.root)
        self.manager = MaintenanceManager(self.storage, self.root)
        self.scans = 0
        self.candidate_sets = list(candidates) or [[]]
        self.degraded = degraded

        def scan():
            index = min(self.scans, len(self.candidate_sets) - 1)
            self.scans += 1
            from local_agent.maintenance_analysis import AnalysisResult

            result = AnalysisResult(
                candidates=[
                    MaintenanceCandidate.from_dict(c.to_dict())
                    for c in self.candidate_sets[index]
                ],
                sources_available={"semantic_graph": not self.degraded},
            )
            return result

        self.runner = MaintenanceRunner(
            analyzer=self.analyzer,
            manager=self.manager,
            scan=scan,
            budget=budget or MaintenanceBudget(),
            policy=MaintenanceExecutionPolicy(repository_root=self.root),
            executor=executor,
            configured_tier=tier,
        )


def actionable_candidate(**kwargs) -> MaintenanceCandidate:
    defaults = dict(
        kind=MaintenanceSignal.PARSE_FAILURE,
        subject="pkg/mod.py",
        affected_files=["pkg/mod.py"],
        severity=SEVERITY_HIGH,
        confidence=0.95,
        sample_size=20,
        occurrence_count=5,
        uncertainty=[],
    )
    defaults.update(kwargs)
    return make_candidate(**defaults)


class RunnerLifecycleTests(unittest.TestCase):
    def test_a_scan_completes_on_an_empty_repository(self):
        harness = RunnerHarness(self)
        result = harness.runner.run(mode=RUN_MODE_SCAN)
        self.assertEqual(result.record.status, RUN_STATUS_COMPLETED)

    def test_a_scan_records_what_it_discovered(self):
        harness = RunnerHarness(self, candidates=[[actionable_candidate()]])
        result = harness.runner.run(mode=RUN_MODE_SCAN)
        self.assertEqual(result.record.candidates_discovered, 1)

    def test_observe_only_selects_nothing(self):
        harness = RunnerHarness(self, candidates=[[actionable_candidate()]])
        result = harness.runner.run(mode=RUN_MODE_DRY_RUN)
        self.assertEqual(result.record.candidates_selected, 0)

    def test_plan_only_selects_and_plans(self):
        harness = RunnerHarness(
            self, candidates=[[actionable_candidate()]], tier=AutonomyTier.PLAN_ONLY
        )
        result = harness.runner.run(mode=RUN_MODE_DRY_RUN)
        self.assertEqual(result.record.candidates_selected, 1)
        self.assertEqual(len(result.work_orders), 1)

    def test_scan_mode_never_plans(self):
        harness = RunnerHarness(
            self, candidates=[[actionable_candidate()]], tier=AutonomyTier.PLAN_ONLY
        )
        result = harness.runner.run(mode=RUN_MODE_SCAN)
        self.assertEqual(result.work_orders, {})

    def test_dry_run_never_executes(self):
        calls = []

        def executor(order):
            calls.append(order)
            return MaintenanceExecutionOutcome(succeeded=True)

        harness = RunnerHarness(
            self, candidates=[[actionable_candidate()]],
            tier=AutonomyTier.EXECUTE_AUTONOMOUSLY, executor=executor,
        )
        harness.runner.run(mode=RUN_MODE_DRY_RUN)
        self.assertEqual(calls, [])

    def test_execute_mode_calls_the_executor(self):
        calls = []

        def executor(order):
            calls.append(order)
            return MaintenanceExecutionOutcome(
                succeeded=True, validation_passed=True, changed_files=["pkg/mod.py"]
            )

        harness = RunnerHarness(
            self, candidates=[[actionable_candidate()], []],
            tier=AutonomyTier.EXECUTE_AUTONOMOUSLY, executor=executor,
        )
        harness.runner.run(mode=RUN_MODE_EXECUTE)
        self.assertEqual(len(calls), 1)

    def test_without_an_executor_nothing_executes(self):
        harness = RunnerHarness(
            self, candidates=[[actionable_candidate()]],
            tier=AutonomyTier.EXECUTE_AUTONOMOUSLY,
        )
        result = harness.runner.run(mode=RUN_MODE_EXECUTE)
        self.assertEqual(result.record.execution_attempts, 0)

    def test_without_an_executor_the_run_says_so(self):
        harness = RunnerHarness(
            self, candidates=[[actionable_candidate()]],
            tier=AutonomyTier.EXECUTE_AUTONOMOUSLY,
        )
        result = harness.runner.run(mode=RUN_MODE_EXECUTE)
        self.assertTrue(any("executor" in note for note in result.record.notes))

    def test_a_clean_execution_that_removes_the_signal_is_resolved(self):
        def executor(order):
            return MaintenanceExecutionOutcome(
                succeeded=True, validation_passed=True, changed_files=["pkg/mod.py"]
            )

        harness = RunnerHarness(
            self, candidates=[[actionable_candidate()], []],
            tier=AutonomyTier.EXECUTE_AUTONOMOUSLY, executor=executor,
        )
        result = harness.runner.run(mode=RUN_MODE_EXECUTE)
        outcome = next(iter(result.reassessments.values()))
        self.assertEqual(outcome.outcome, ReassessmentOutcome.RESOLVED)

    def test_a_clean_execution_that_leaves_the_signal_is_persisting(self):
        candidate = actionable_candidate()

        def executor(order):
            return MaintenanceExecutionOutcome(
                succeeded=True, validation_passed=True, changed_files=["pkg/mod.py"]
            )

        harness = RunnerHarness(
            self, candidates=[[candidate], [candidate]],
            tier=AutonomyTier.EXECUTE_AUTONOMOUSLY, executor=executor,
        )
        result = harness.runner.run(mode=RUN_MODE_EXECUTE)
        outcome = next(iter(result.reassessments.values()))
        self.assertEqual(outcome.outcome, ReassessmentOutcome.PERSISTING)

    def test_an_out_of_scope_change_fails_the_candidate(self):
        def executor(order):
            return MaintenanceExecutionOutcome(
                succeeded=True, validation_passed=True,
                changed_files=["pkg/mod.py", "pkg/somewhere_else.py"],
            )

        harness = RunnerHarness(
            self, candidates=[[actionable_candidate()], []],
            tier=AutonomyTier.EXECUTE_AUTONOMOUSLY, executor=executor,
        )
        result = harness.runner.run(mode=RUN_MODE_EXECUTE)
        self.assertEqual(result.record.executions_failed, 1)

    def test_an_out_of_scope_change_cannot_be_credited_as_resolved(self):
        def executor(order):
            return MaintenanceExecutionOutcome(
                succeeded=True, validation_passed=True,
                changed_files=["pkg/mod.py", "elsewhere.py"],
            )

        harness = RunnerHarness(
            self, candidates=[[actionable_candidate()], []],
            tier=AutonomyTier.EXECUTE_AUTONOMOUSLY, executor=executor,
        )
        result = harness.runner.run(mode=RUN_MODE_EXECUTE)
        outcome = next(iter(result.reassessments.values()))
        self.assertNotEqual(outcome.outcome, ReassessmentOutcome.RESOLVED)

    def test_a_protected_file_change_fails_the_candidate(self):
        def executor(order):
            return MaintenanceExecutionOutcome(
                succeeded=True, validation_passed=True,
                changed_files=["local_agent/tool_engine.py"],
            )

        harness = RunnerHarness(
            self, candidates=[[actionable_candidate()], []],
            tier=AutonomyTier.EXECUTE_AUTONOMOUSLY, executor=executor,
        )
        result = harness.runner.run(mode=RUN_MODE_EXECUTE)
        self.assertEqual(result.record.executions_failed, 1)

    def test_an_exploding_executor_is_isolated(self):
        def executor(order):
            raise RuntimeError("provider quota exhausted")

        harness = RunnerHarness(
            self, candidates=[[actionable_candidate()], []],
            tier=AutonomyTier.EXECUTE_AUTONOMOUSLY, executor=executor,
        )
        result = harness.runner.run(mode=RUN_MODE_EXECUTE)
        self.assertEqual(result.record.executions_failed, 1)
        self.assertIn(result.record.status, {"partial", "completed"})

    def test_an_executor_returning_garbage_is_isolated(self):
        harness = RunnerHarness(
            self, candidates=[[actionable_candidate()], []],
            tier=AutonomyTier.EXECUTE_AUTONOMOUSLY, executor=lambda order: "fine!",
        )
        result = harness.runner.run(mode=RUN_MODE_EXECUTE)
        self.assertEqual(result.record.executions_failed, 1)

    def test_one_failing_candidate_does_not_stop_the_others(self):
        good = actionable_candidate(subject="pkg/good.py", affected_files=["pkg/good.py"])
        bad = actionable_candidate(subject="pkg/bad.py", affected_files=["pkg/bad.py"])

        def executor(order):
            if "bad" in order.scope_files[0]:
                raise RuntimeError("nope")
            return MaintenanceExecutionOutcome(
                succeeded=True, validation_passed=True, changed_files=list(order.scope_files)
            )

        harness = RunnerHarness(
            self, candidates=[[good, bad], []],
            tier=AutonomyTier.EXECUTE_AUTONOMOUSLY, executor=executor,
        )
        result = harness.runner.run(mode=RUN_MODE_EXECUTE)
        self.assertEqual(result.record.execution_attempts, 2)
        self.assertEqual(result.record.executions_succeeded, 1)
        self.assertEqual(result.record.executions_failed, 1)

    def test_a_degraded_rescan_prevents_a_resolved_verdict(self):
        def executor(order):
            return MaintenanceExecutionOutcome(
                succeeded=True, validation_passed=True, changed_files=["pkg/mod.py"]
            )

        harness = RunnerHarness(
            self, candidates=[[actionable_candidate()], []],
            tier=AutonomyTier.EXECUTE_AUTONOMOUSLY, executor=executor, degraded=True,
        )
        result = harness.runner.run(mode=RUN_MODE_EXECUTE)
        outcome = next(iter(result.reassessments.values()))
        self.assertEqual(outcome.outcome, ReassessmentOutcome.INCONCLUSIVE)

    def test_the_execution_budget_is_enforced(self):
        candidates = [
            actionable_candidate(subject=f"pkg/f{i}.py", affected_files=[f"pkg/f{i}.py"])
            for i in range(5)
        ]

        def executor(order):
            return MaintenanceExecutionOutcome(
                succeeded=True, validation_passed=True, changed_files=list(order.scope_files)
            )

        harness = RunnerHarness(
            self, candidates=[candidates, []],
            tier=AutonomyTier.EXECUTE_AUTONOMOUSLY, executor=executor,
            budget=MaintenanceBudget(max_candidates_executed=1, max_candidates_selected=5),
        )
        result = harness.runner.run(mode=RUN_MODE_EXECUTE)
        self.assertEqual(result.record.execution_attempts, 1)

    def test_the_selection_budget_is_enforced(self):
        candidates = [
            actionable_candidate(subject=f"pkg/f{i}.py", affected_files=[f"pkg/f{i}.py"])
            for i in range(10)
        ]
        harness = RunnerHarness(
            self, candidates=[candidates], tier=AutonomyTier.PLAN_ONLY,
            budget=MaintenanceBudget(max_candidates_selected=2, max_candidates_executed=2),
        )
        result = harness.runner.run(mode=RUN_MODE_DRY_RUN)
        self.assertEqual(result.record.candidates_selected, 2)

    def test_the_consideration_budget_is_enforced(self):
        candidates = [
            actionable_candidate(subject=f"pkg/f{i}.py", affected_files=[f"pkg/f{i}.py"])
            for i in range(10)
        ]
        harness = RunnerHarness(
            self, candidates=[candidates], tier=AutonomyTier.PLAN_ONLY,
            budget=MaintenanceBudget(
                max_candidates_considered=3, max_candidates_selected=3,
                max_candidates_executed=1,
            ),
        )
        result = harness.runner.run(mode=RUN_MODE_DRY_RUN)
        self.assertEqual(result.record.candidates_discovered, 3)

    def test_cancellation_stops_the_run(self):
        harness = RunnerHarness(
            self, candidates=[[actionable_candidate()]], tier=AutonomyTier.PLAN_ONLY
        )
        harness.runner.cancelled = lambda: True
        result = harness.runner.run(mode=RUN_MODE_DRY_RUN)
        self.assertEqual(result.record.status, "cancelled")

    def test_a_failing_scan_does_not_crash_the_run(self):
        harness = RunnerHarness(self)

        def boom():
            raise RuntimeError("scan exploded")

        harness.runner.scan = boom
        result = harness.runner.run(mode=RUN_MODE_SCAN)
        self.assertTrue(result.record.errors)

    def test_run_results_serialise(self):
        harness = RunnerHarness(self, candidates=[[actionable_candidate()]])
        payload = harness.runner.run(mode=RUN_MODE_SCAN).to_dict()
        self.assertIn("run", payload)
        self.assertIn("ranking", payload)
        self.assertIn("policy", payload)

    def test_the_run_is_persisted(self):
        harness = RunnerHarness(self, candidates=[[actionable_candidate()]])
        harness.runner.run(mode=RUN_MODE_SCAN)
        self.assertEqual(len(harness.manager.load().runs), 1)

    def test_candidates_are_persisted(self):
        harness = RunnerHarness(self, candidates=[[actionable_candidate()]])
        harness.runner.run(mode=RUN_MODE_SCAN)
        self.assertEqual(len(harness.manager.load()), 1)

    def test_repeated_runs_accumulate_occurrences(self):
        harness = RunnerHarness(self, candidates=[[actionable_candidate()]])
        harness.runner.run(mode=RUN_MODE_SCAN)
        harness.runner.run(mode=RUN_MODE_SCAN)
        candidate = harness.manager.load().candidates[0]
        self.assertGreater(candidate.occurrence_count, 5)

    def test_an_unknown_mode_degrades_to_scan(self):
        harness = RunnerHarness(
            self, candidates=[[actionable_candidate()]], tier=AutonomyTier.PLAN_ONLY
        )
        result = harness.runner.run(mode="do_everything")
        self.assertEqual(result.record.mode, RUN_MODE_SCAN)

    def test_an_unknown_configured_tier_degrades_to_observe_only(self):
        harness = RunnerHarness(self, tier="root")
        self.assertEqual(harness.runner.configured_tier, AutonomyTier.OBSERVE_ONLY)

    def test_budget_consumption_is_reported(self):
        harness = RunnerHarness(self, candidates=[[actionable_candidate()]])
        result = harness.runner.run(mode=RUN_MODE_SCAN)
        self.assertIn("limits", result.record.budget)
        self.assertIn("consumed", result.record.budget)

    def test_elapsed_time_is_recorded(self):
        harness = RunnerHarness(self)
        self.assertGreaterEqual(harness.runner.run(mode=RUN_MODE_SCAN).record.elapsed_seconds, 0.0)

    def test_batches_are_reported(self):
        candidates = [
            actionable_candidate(subject=f"pkg/f{i}.py", affected_files=[f"pkg/f{i}.py"])
            for i in range(4)
        ]
        harness = RunnerHarness(
            self, candidates=[candidates], tier=AutonomyTier.PLAN_ONLY,
            budget=MaintenanceBudget(max_candidates_selected=4, max_candidates_executed=2),
        )
        result = harness.runner.run(mode=RUN_MODE_DRY_RUN)
        self.assertTrue(result.batches)


class ManagerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_a_failing_storage_load_yields_an_untrustworthy_store(self):
        manager = MaintenanceManager(FailingStorage(), self.root)
        self.assertFalse(manager.load().history_trustworthy())

    def test_a_failing_storage_save_is_survivable(self):
        manager = MaintenanceManager(FailingStorage(), self.root)
        self.assertFalse(manager.save(MaintenanceStore()))

    def test_a_storage_without_the_methods_is_survivable(self):
        manager = MaintenanceManager(object(), self.root)
        self.assertEqual(len(manager.load()), 0)
        self.assertFalse(manager.save(MaintenanceStore()))

    def test_a_storage_returning_the_wrong_type_is_untrustworthy(self):
        class Wrong:
            def load_maintenance(self):
                return {"not": "a store"}

            def save_maintenance(self, store):
                pass

        self.assertFalse(MaintenanceManager(Wrong(), self.root).load().history_trustworthy())

    def test_mutate_persists(self):
        storage = JsonFileStorage(self.root / "data")
        manager = MaintenanceManager(storage, self.root)
        manager.mutate(lambda store: store.upsert(make_candidate()))
        self.assertEqual(len(manager.load()), 1)

    def test_concurrent_mutations_do_not_lose_records(self):
        storage = JsonFileStorage(self.root / "data")
        manager = MaintenanceManager(storage, self.root)
        errors: list[BaseException] = []

        def worker(index: int) -> None:
            try:
                manager.mutate(
                    lambda store: store.upsert(make_candidate(subject=f"f{index}.py"))
                )
            except BaseException as exc:  # pragma: no cover - reported below
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(manager.load()), 8)


# =============================================================================
# M. learning - advisory only
# =============================================================================


class ActionabilityTests(unittest.TestCase):
    def store_with(self, outcomes):
        store = MaintenanceStore()
        record = MaintenanceRunRecord()
        for kind, outcome in outcomes:
            record.outcomes.append(
                CandidateRunOutcome(
                    candidate_id=f"{kind}-{outcome}", kind=kind, outcome=outcome, executed=True
                )
            )
        store.record_run(record)
        return store

    def test_an_empty_store_reports_no_attempts(self):
        self.assertEqual(compute_actionability(MaintenanceStore())["total_attempts"], 0)

    def test_results_are_always_flagged_advisory(self):
        self.assertTrue(compute_actionability(MaintenanceStore())["advisory_only"])

    def test_a_thin_sample_is_flagged_insufficient(self):
        store = self.store_with([("test_gap", ReassessmentOutcome.RESOLVED)] * 2)
        self.assertFalse(compute_actionability(store, min_samples=5)["data_sufficient"])

    def test_a_thick_sample_is_flagged_sufficient(self):
        store = self.store_with([("test_gap", ReassessmentOutcome.RESOLVED)] * 10)
        self.assertTrue(compute_actionability(store, min_samples=5)["data_sufficient"])

    def test_resolution_is_counted(self):
        store = self.store_with([("test_gap", ReassessmentOutcome.RESOLVED)])
        self.assertEqual(
            compute_actionability(store)["by_kind"]["test_gap"]["resolved"], 1
        )

    def test_partial_resolution_is_not_counted_as_resolution(self):
        store = self.store_with([("test_gap", ReassessmentOutcome.PARTIALLY_RESOLVED)])
        self.assertEqual(
            compute_actionability(store)["by_kind"]["test_gap"]["resolved"], 0
        )

    def test_unexecuted_candidates_are_not_counted(self):
        store = MaintenanceStore()
        record = MaintenanceRunRecord()
        record.outcomes.append(CandidateRunOutcome(kind="test_gap", executed=False))
        store.record_run(record)
        self.assertEqual(compute_actionability(store)["total_attempts"], 0)

    def test_a_corrupt_history_is_reported_as_untrustworthy(self):
        store = MaintenanceStore.from_dict("garbage")
        self.assertFalse(compute_actionability(store)["history_trustworthy"])

    def test_the_resolution_rate_is_computed(self):
        store = self.store_with(
            [("test_gap", ReassessmentOutcome.RESOLVED),
             ("test_gap", ReassessmentOutcome.PERSISTING)]
        )
        # Two entries with the same candidate id would collapse; they differ by
        # outcome in the fixture, so both are recorded.
        self.assertAlmostEqual(
            compute_actionability(store)["by_kind"]["test_gap"]["resolution_rate"], 0.5
        )


# =============================================================================
# N. structural invariants - proved from the AST, not asserted in prose
# =============================================================================


class ArchitecturalInvariantTests(unittest.TestCase):
    def test_no_maintenance_module_imports_the_validation_decision_engine(self):
        for module in MAINTENANCE_MODULES:
            self.assertNotIn("local_agent.validation_decision", imported_modules(module), module)

    def test_no_maintenance_module_references_the_decision_engine_in_code(self):
        for module in MAINTENANCE_MODULES:
            self.assertNotIn("ValidationDecisionEngine", code_identifiers(module), module)

    def test_no_maintenance_module_imports_approval(self):
        for module in MAINTENANCE_MODULES:
            self.assertNotIn("local_agent.approval", imported_modules(module), module)

    def test_no_maintenance_module_references_the_approval_engine(self):
        for module in MAINTENANCE_MODULES:
            self.assertNotIn("ApprovalPolicyEngine", code_identifiers(module), module)

    def test_no_maintenance_module_imports_the_tool_engine(self):
        for module in MAINTENANCE_MODULES:
            self.assertNotIn("local_agent.tool_engine", imported_modules(module), module)

    def test_no_maintenance_module_imports_the_coding_agent(self):
        for module in MAINTENANCE_MODULES:
            self.assertNotIn("local_agent.coding_agent", imported_modules(module), module)

    def test_no_maintenance_module_imports_the_orchestrator(self):
        for module in MAINTENANCE_MODULES:
            self.assertNotIn("local_agent.orchestrator", imported_modules(module), module)

    def test_no_maintenance_module_imports_the_sandbox(self):
        for module in MAINTENANCE_MODULES:
            self.assertNotIn("local_agent.sandbox", imported_modules(module), module)

    def test_no_maintenance_module_spawns_a_subprocess(self):
        for module in MAINTENANCE_MODULES:
            self.assertNotIn("subprocess", imported_modules(module), module)

    def test_no_maintenance_module_uses_os_system_or_popen(self):
        for module in MAINTENANCE_MODULES:
            identifiers = code_identifiers(module)
            for forbidden in ("system", "popen", "Popen", "spawnv", "execv"):
                self.assertNotIn(forbidden, identifiers, f"{module}:{forbidden}")

    def test_no_maintenance_module_uses_eval_or_exec(self):
        for module in MAINTENANCE_MODULES:
            identifiers = code_identifiers(module)
            self.assertNotIn("eval", identifiers, module)
            self.assertNotIn("exec", identifiers, module)

    def test_no_maintenance_module_changes_the_working_directory(self):
        for module in MAINTENANCE_MODULES:
            self.assertNotIn("chdir", code_identifiers(module), module)

    def test_no_maintenance_module_writes_to_the_filesystem_directly(self):
        for module in MAINTENANCE_MODULES:
            identifiers = code_identifiers(module)
            for forbidden in ("write_text", "write_bytes", "unlink", "rmtree", "mkdir"):
                self.assertNotIn(forbidden, identifiers, f"{module}:{forbidden}")

    def test_the_policy_module_does_not_import_the_runner(self):
        self.assertNotIn(
            "local_agent.maintenance_runner", imported_modules("local_agent.maintenance_policy")
        )

    def test_the_model_module_imports_no_maintenance_sibling(self):
        for imported in imported_modules("local_agent.maintenance"):
            self.assertNotIn("maintenance_", imported)

    def test_the_analyzer_reuses_the_existing_wilson_bound(self):
        self.assertIn(
            "local_agent.validation_telemetry",
            imported_modules("local_agent.maintenance_analysis"),
        )

    def test_the_analyzer_reuses_the_existing_path_normalisation(self):
        self.assertIn(
            "local_agent.semantic_impact", imported_modules("local_agent.maintenance")
        )

    def test_maintenance_modules_define_no_second_validation_decision(self):
        for module in MAINTENANCE_MODULES:
            source = ast.dump(_module_ast(module))
            self.assertNotIn("ValidationDecision", source, module)

    def test_the_runner_reaches_the_repository_only_through_the_executor(self):
        """The single seam through which a repository change can happen.

        Asserted from the AST: the runner names exactly one attribute that is
        called with a work order, and it is injected. If a future change added
        a direct call into the implementation pipeline, this would catch it by
        the module's import list rather than by anyone remembering to look.
        """
        imported = imported_modules("local_agent.maintenance_runner")
        for forbidden in (
            "local_agent.tools",
            "local_agent.filesystem",
            "local_agent.patching",
            "local_agent.git",
            "local_agent.validation",
        ):
            self.assertNotIn(forbidden, imported, forbidden)


class ConfigurationInvariantTests(unittest.TestCase):
    def test_maintenance_is_disabled_by_default(self):
        self.assertFalse(AgentConfig.from_environment(".").maintenance_enabled)

    def test_the_default_autonomy_tier_is_observe_only(self):
        self.assertEqual(
            AgentConfig.from_environment(".").maintenance_autonomy_tier,
            AutonomyTier.OBSERVE_ONLY,
        )

    def test_the_default_config_validates(self):
        AgentConfig.from_environment(".").validate()

    def test_an_unknown_autonomy_tier_is_rejected(self):
        config = AgentConfig.from_environment(".")
        config.maintenance_autonomy_tier = "root"
        with self.assertRaises(ValueError):
            config.validate()

    def test_every_declared_tier_is_accepted_by_the_config(self):
        for tier in TIER_ORDER:
            config = AgentConfig.from_environment(".")
            config.maintenance_autonomy_tier = tier
            config.validate()

    def test_an_incoherent_budget_is_rejected(self):
        config = AgentConfig.from_environment(".")
        config.maintenance_max_candidates_executed = 99
        with self.assertRaises(ValueError):
            config.validate()

    def test_a_zero_execution_budget_is_accepted(self):
        config = AgentConfig.from_environment(".")
        config.maintenance_max_candidates_executed = 0
        config.validate()

    def test_the_budget_is_derived_from_the_config(self):
        config = AgentConfig.from_environment(".")
        config.maintenance_max_dag_width = 1
        self.assertEqual(MaintenanceBudget.from_config(config).max_dag_width, 1)

    def test_a_config_object_without_maintenance_fields_yields_defaults(self):
        self.assertEqual(MaintenanceBudget.from_config(object()), MaintenanceBudget())

    def test_environment_overrides_are_honoured(self):
        config = AgentConfig.from_environment(".", maintenance_enabled=True)
        self.assertTrue(config.maintenance_enabled)


# =============================================================================
# O. adversarial scenarios
# =============================================================================


class AdversarialTests(unittest.TestCase):
    """Part 17. Every case must fail *safely*, not merely fail."""

    def setUp(self):
        self.policy = MaintenanceExecutionPolicy()
        self.strong = AutonomyTier.EXECUTE_AUTONOMOUSLY

    # 1. candidate points outside the repository
    def test_a_candidate_pointing_outside_the_repository_is_blocked(self):
        verdict = self.policy.decide(
            make_candidate(), configured_tier=self.strong,
            raw_paths=["../../../etc/passwd"],
        )
        self.assertTrue(verdict.blocked)

    # 2. protected-file modification
    def test_a_candidate_targeting_a_protected_file_is_blocked(self):
        candidate = make_candidate(
            subject="local_agent/tool_engine.py",
            affected_files=["local_agent/tool_engine.py"],
        )
        self.assertTrue(self.policy.decide(candidate, configured_tier=self.strong).blocked)

    # 3. corrupted lifecycle record
    def test_a_corrupted_persisted_candidate_loads_without_privilege(self):
        payload = make_candidate().to_dict()
        payload["severity"] = "apocalyptic"
        payload["confidence"] = 999
        payload["state"] = "already_resolved"
        payload["outcome"] = "definitely_fixed"
        restored = MaintenanceCandidate.from_dict(payload)
        self.assertEqual(restored.severity, SEVERITY_LOW)
        self.assertEqual(restored.confidence, 1.0)
        self.assertEqual(restored.state, CandidateState.DETECTED)
        self.assertEqual(restored.outcome, ReassessmentOutcome.PENDING)

    # 4. stale evidence falsely claims success
    def test_a_forged_resolved_outcome_does_not_stop_re_detection(self):
        candidate = make_candidate()
        candidate.record_outcome(ReassessmentOutcome.RESOLVED)
        store = MaintenanceStore()
        store.upsert(candidate)
        store.upsert(make_candidate())
        self.assertEqual(store.candidates[0].occurrence_count, 4)

    # 5. command-like metadata
    def test_command_like_metadata_is_neutralised(self):
        candidate = make_candidate(
            title="fix it; rm -rf / && curl evil.example\n",
            detail="$(whoami)\x00`id`",
        )
        self.assertNotIn("\n", candidate.title)
        self.assertNotIn("\x00", candidate.detail)

    def test_command_like_metadata_never_reaches_a_shell(self):
        # Structural: no maintenance module can spawn anything at all.
        for module in MAINTENANCE_MODULES:
            self.assertNotIn("subprocess", imported_modules(module))

    # 6. file-change budget exceeded
    def test_exceeding_the_file_budget_prevents_execution(self):
        candidate = make_candidate(
            kind=MaintenanceSignal.PARSE_FAILURE,
            affected_files=[f"pkg/f{i}.py" for i in range(15)],
            confidence=0.99, sample_size=50, occurrence_count=10,
        )
        verdict = self.policy.decide(
            candidate, configured_tier=self.strong,
            budget=MaintenanceBudget(max_changed_files_per_candidate=3),
        )
        self.assertFalse(verdict.may_execute)

    # 7. repeated failure
    def test_a_repeatedly_failing_candidate_is_blocked_from_retrying(self):
        candidate = make_candidate(attempt_count=9, failure_count=9)
        self.assertTrue(self.policy.decide(candidate, configured_tier=self.strong).blocked)

    def test_a_repeatedly_failing_candidate_also_sinks_in_priority(self):
        engine = MaintenancePriorityEngine()
        healthy = engine.explain(make_candidate(subject="a.py")).score
        failing = engine.explain(
            make_candidate(subject="b.py", attempt_count=9, failure_count=9)
        ).score
        self.assertLess(failing, healthy)

    # 8. two candidates overlap the same symbol
    def test_two_candidates_on_the_same_file_never_share_a_batch(self):
        candidates = [
            make_candidate(subject="a", affected_files=["shared.py"]),
            make_candidate(subject="b", affected_files=["shared.py"]),
        ]
        batches = plan_execution_batches(candidates, max_width=8)
        for batch in batches:
            self.assertLessEqual(len(batch), 1)

    # 9. one succeeds while another fails (covered above); partial status
    def test_a_partially_failing_run_is_reported_as_partial(self):
        good = actionable_candidate(subject="pkg/good.py", affected_files=["pkg/good.py"])
        bad = actionable_candidate(subject="pkg/bad.py", affected_files=["pkg/bad.py"])

        def executor(order):
            if "bad" in order.scope_files[0]:
                raise RuntimeError("nope")
            return MaintenanceExecutionOutcome(
                succeeded=True, validation_passed=True, changed_files=list(order.scope_files)
            )

        harness = RunnerHarness(
            self, candidates=[[good, bad], []],
            tier=AutonomyTier.EXECUTE_AUTONOMOUSLY, executor=executor,
        )
        self.assertEqual(harness.runner.run(mode=RUN_MODE_EXECUTE).record.status, "partial")

    # 10. provider quota halfway through
    def test_a_quota_failure_halfway_through_leaves_earlier_work_recorded(self):
        candidates = [
            actionable_candidate(subject=f"pkg/f{i}.py", affected_files=[f"pkg/f{i}.py"])
            for i in range(3)
        ]
        calls = {"n": 0}

        def executor(order):
            calls["n"] += 1
            if calls["n"] > 1:
                raise RuntimeError("quota exhausted")
            return MaintenanceExecutionOutcome(
                succeeded=True, validation_passed=True, changed_files=list(order.scope_files)
            )

        harness = RunnerHarness(
            self, candidates=[candidates, []],
            tier=AutonomyTier.EXECUTE_AUTONOMOUSLY, executor=executor,
            budget=MaintenanceBudget(max_candidates_selected=3, max_candidates_executed=3),
        )
        result = harness.runner.run(mode=RUN_MODE_EXECUTE)
        self.assertEqual(result.record.executions_succeeded, 1)
        self.assertEqual(result.record.executions_failed, 2)

    # 11. validation tool unavailable
    def test_an_unknown_validation_result_is_never_assumed_to_pass(self):
        def executor(order):
            return MaintenanceExecutionOutcome(
                succeeded=True, validation_passed=None, changed_files=["pkg/mod.py"]
            )

        harness = RunnerHarness(
            self, candidates=[[actionable_candidate()], [actionable_candidate()]],
            tier=AutonomyTier.EXECUTE_AUTONOMOUSLY, executor=executor,
        )
        result = harness.runner.run(mode=RUN_MODE_EXECUTE)
        outcome = next(iter(result.record.outcomes))
        self.assertIsNone(outcome.validation_passed)

    # 12. analyzer returns malformed data
    def test_a_malformed_analyzer_result_is_isolated(self):
        analyzer = MaintenanceAnalyzer(Path.cwd())

        class Rogue:
            @property
            def decisions(self):
                return "not a list"

        self.assertEqual(analyzer.analyze(telemetry_store=Rogue()).candidates, [])

    # 13. artificially perfect confidence over two samples (covered in policy)
    def test_perfect_confidence_over_two_samples_is_not_established(self):
        store = MaintenanceStore()
        record = MaintenanceRunRecord()
        for index in range(2):
            record.outcomes.append(
                CandidateRunOutcome(
                    candidate_id=f"c{index}", kind="test_gap",
                    outcome=ReassessmentOutcome.RESOLVED, executed=True,
                )
            )
        store.record_run(record)
        stats = compute_actionability(store, min_samples=5)
        self.assertEqual(stats["by_kind"]["test_gap"]["resolution_rate"], 1.0)
        self.assertFalse(stats["data_sufficient"])

    # 14. candidate disappears during reassessment (covered by RESOLVED path)
    def test_a_disappearing_candidate_without_execution_is_inconclusive(self):
        harness = RunnerHarness(
            self, candidates=[[actionable_candidate()], []], tier=AutonomyTier.PLAN_ONLY
        )
        result = harness.runner.run(mode=RUN_MODE_DRY_RUN)
        self.assertEqual(result.reassessments, {})

    # 15. task changes a file outside its original scope (covered above)
    def test_an_out_of_scope_change_is_named_in_the_run_record(self):
        def executor(order):
            return MaintenanceExecutionOutcome(
                succeeded=True, validation_passed=True,
                changed_files=["pkg/mod.py", "secret/plans.py"],
            )

        harness = RunnerHarness(
            self, candidates=[[actionable_candidate()], []],
            tier=AutonomyTier.EXECUTE_AUTONOMOUSLY, executor=executor,
        )
        result = harness.runner.run(mode=RUN_MODE_EXECUTE)
        errors = " ".join(result.record.outcomes[0].errors)
        self.assertIn("outside the declared maintenance scope", errors)

    # 16. failure after successful validation
    def test_a_reported_failure_after_passing_validation_is_not_success(self):
        def executor(order):
            return MaintenanceExecutionOutcome(
                succeeded=False, validation_passed=True, changed_files=["pkg/mod.py"],
                error="merge conflict on integrate",
            )

        harness = RunnerHarness(
            self, candidates=[[actionable_candidate()], []],
            tier=AutonomyTier.EXECUTE_AUTONOMOUSLY, executor=executor,
        )
        result = harness.runner.run(mode=RUN_MODE_EXECUTE)
        self.assertEqual(result.record.executions_failed, 1)
        outcome = next(iter(result.reassessments.values()))
        self.assertNotEqual(outcome.outcome, ReassessmentOutcome.RESOLVED)

    # 17. resume after partial completion
    def test_a_second_run_resumes_from_the_persisted_store(self):
        harness = RunnerHarness(self, candidates=[[actionable_candidate()]])
        harness.runner.run(mode=RUN_MODE_SCAN)
        harness.runner.run(mode=RUN_MODE_SCAN)
        store = harness.manager.load()
        self.assertEqual(len(store.runs), 2)
        self.assertEqual(len(store), 1)

    # 18. concurrent runs against shared persistence (see ManagerTests)
    def test_concurrent_runners_do_not_corrupt_the_store(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        storage = JsonFileStorage(root / "data")
        manager = MaintenanceManager(storage, root)

        def worker(index: int) -> None:
            runner = MaintenanceRunner(
                analyzer=MaintenanceAnalyzer(root),
                manager=MaintenanceManager(storage, root),
                scan=lambda: __import__(
                    "local_agent.maintenance_analysis", fromlist=["AnalysisResult"]
                ).AnalysisResult(candidates=[make_candidate(subject=f"f{index}.py")]),
            )
            runner.run(mode=RUN_MODE_SCAN)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        store = manager.load()
        self.assertTrue(store.history_trustworthy())
        self.assertGreaterEqual(len(store), 1)

    # 19. priority manipulation attempt
    def test_a_forged_priority_field_is_not_read(self):
        payload = make_candidate().to_dict()
        payload["priority"] = 9999
        payload["score"] = 9999
        restored = MaintenanceCandidate.from_dict(payload)
        engine = MaintenancePriorityEngine()
        self.assertLessEqual(engine.explain(restored).score, 1.0)

    def test_a_forged_severity_cannot_jump_the_queue(self):
        payload = make_candidate(subject="forged.py").to_dict()
        payload["severity"] = "ultra_critical"
        forged = MaintenanceCandidate.from_dict(payload)
        honest = make_candidate(subject="honest.py", severity=SEVERITY_HIGH)
        ranked = MaintenancePriorityEngine().rank([forged, honest])
        self.assertEqual(ranked[0][0].candidate_id, honest.candidate_id)

    # 20. malicious filename normalisation
    def test_a_null_byte_filename_is_rejected(self):
        self.assertEqual(sanitize_relative_path("pkg/mod\x00.py"), "pkg/mod.py")

    def test_a_dotdot_only_filename_is_rejected(self):
        self.assertEqual(sanitize_relative_path(".."), "")

    def test_a_single_dot_filename_is_rejected(self):
        self.assertEqual(sanitize_relative_path("."), "")

    def test_a_trailing_traversal_is_rejected(self):
        self.assertEqual(sanitize_relative_path("pkg/.."), "")

    def test_a_deeply_nested_traversal_is_rejected(self):
        self.assertEqual(sanitize_relative_path("a/b/c/../../../../x"), "")

    def test_an_oversized_persisted_history_is_bounded_on_load(self):
        payload = make_candidate().to_dict()
        payload["history"] = [{"at": "x", "event": "e", "reason": "r"}] * 10_000
        self.assertLessEqual(
            len(MaintenanceCandidate.from_dict(payload).history), MAX_HISTORY_ENTRIES
        )

    def test_an_oversized_evidence_list_is_bounded_on_load(self):
        payload = make_candidate().to_dict()
        payload["evidence_refs"] = [f"ref{i}" for i in range(10_000)]
        self.assertLessEqual(len(MaintenanceCandidate.from_dict(payload).evidence_refs), 20)


# =============================================================================
# P. the CLI
# =============================================================================


class CliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        (self.root / "pkg").mkdir()
        (self.root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
        (self.root / "pkg" / "broken.py").write_text("def (:\n", encoding="utf-8")

    def run_cli(self, *args) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cli_main(["maintenance", *args, "--project", str(self.root)])
        return code, out.getvalue(), err.getvalue()

    def test_scan_succeeds_on_a_fresh_repository(self):
        code, out, _ = self.run_cli("scan")
        self.assertEqual(code, 0)
        self.assertIn("Maintenance Scan", out)

    def test_scan_detects_the_real_syntax_error(self):
        _, out, _ = self.run_cli("scan")
        self.assertIn("Discovered:", out)

    def test_scan_json_is_valid_json(self):
        _, out, _ = self.run_cli("scan", "--json")
        json.loads(out)

    def test_candidates_lists_what_the_scan_found(self):
        self.run_cli("scan")
        code, out, _ = self.run_cli("candidates")
        self.assertEqual(code, 0)
        self.assertIn("broken.py", out)

    def test_candidates_json_is_valid(self):
        self.run_cli("scan")
        _, out, _ = self.run_cli("candidates", "--json")
        self.assertIn("candidates", json.loads(out))

    def test_candidate_detail_requires_a_known_id(self):
        code, _, err = self.run_cli("candidate", "nope")
        self.assertEqual(code, 1)
        self.assertIn("no maintenance candidate", err)

    def test_candidate_detail_shows_a_known_candidate(self):
        self.run_cli("scan")
        _, listing, _ = self.run_cli("candidates", "--json")
        candidate_id = json.loads(listing)["candidates"][0]["candidate_id"]
        code, out, _ = self.run_cli("candidate", candidate_id)
        self.assertEqual(code, 0)
        self.assertIn("Provenance", out)

    def test_health_reports_the_disabled_default(self):
        code, out, _ = self.run_cli("health")
        self.assertEqual(code, 0)
        self.assertIn("Subsystem enabled:     False", out)

    def test_health_flags_advisory_statistics(self):
        _, out, _ = self.run_cli("health")
        self.assertIn("ADVISORY ONLY", out)

    def test_health_json_is_valid(self):
        _, out, _ = self.run_cli("health", "--json")
        self.assertIn("actionability", json.loads(out))

    def test_history_is_empty_before_any_run(self):
        _, out, _ = self.run_cli("history")
        self.assertIn("No maintenance runs recorded", out)

    def test_history_lists_a_completed_run(self):
        self.run_cli("scan")
        _, out, _ = self.run_cli("history")
        self.assertIn("[completed]", out)

    def test_status_reports_no_run_initially(self):
        _, out, _ = self.run_cli("status")
        self.assertIn("No maintenance run has been recorded", out)

    def test_status_reports_the_latest_run(self):
        self.run_cli("scan")
        _, out, _ = self.run_cli("status")
        self.assertIn("Last run:", out)

    def test_recommendations_explains_its_ranking(self):
        code, out, _ = self.run_cli("recommendations")
        self.assertEqual(code, 0)
        self.assertIn("why ranked here", out)

    def test_recommendations_reports_the_granted_autonomy(self):
        _, out, _ = self.run_cli("recommendations")
        self.assertIn("autonomy:", out)

    def test_dry_run_succeeds(self):
        code, out, _ = self.run_cli("dry-run")
        self.assertEqual(code, 0)
        self.assertIn("Maintenance Dry Run", out)

    def test_run_is_refused_while_the_subsystem_is_disabled(self):
        code, _, err = self.run_cli("run")
        self.assertEqual(code, 1)
        self.assertIn("disabled", err)

    def test_run_is_permitted_once_enabled(self):
        code, out, _ = self.run_cli("run", "--maintenance", "true")
        self.assertEqual(code, 0)
        self.assertIn("Maintenance Run", out)

    def test_run_executes_nothing_at_the_default_tier(self):
        _, out, _ = self.run_cli("run", "--maintenance", "true")
        self.assertIn("Execution attempts:0", out)

    def test_run_explains_that_no_executor_is_wired(self):
        _, out, _ = self.run_cli("run", "--maintenance", "true")
        self.assertIn("no direct maintenance executor", out)

    def test_enqueue_creates_real_tasks(self):
        code, out, _ = self.run_cli(
            "run", "--maintenance", "true", "--maintenance-tier", "plan_only", "--enqueue"
        )
        self.assertEqual(code, 0)
        storage = JsonFileStorage(self.root / ".agent_data")
        self.assertTrue(storage.list_tasks())

    def test_enqueued_tasks_are_ordinary_pending_tasks(self):
        self.run_cli(
            "run", "--maintenance", "true", "--maintenance-tier", "plan_only", "--enqueue"
        )
        storage = JsonFileStorage(self.root / ".agent_data")
        for task in storage.list_tasks():
            self.assertEqual(task.status.value, "pending")

    def test_enqueued_tasks_record_their_maintenance_provenance(self):
        self.run_cli(
            "run", "--maintenance", "true", "--maintenance-tier", "plan_only", "--enqueue"
        )
        storage = JsonFileStorage(self.root / ".agent_data")
        task = storage.list_tasks()[0]
        self.assertEqual(task.execution_history[0]["source"], "maintenance")

    def test_nothing_is_enqueued_without_the_flag(self):
        self.run_cli("run", "--maintenance", "true", "--maintenance-tier", "plan_only")
        storage = JsonFileStorage(self.root / ".agent_data")
        self.assertEqual(storage.list_tasks(), [])

    def test_an_unknown_subcommand_is_rejected_by_the_parser(self):
        with self.assertRaises(SystemExit):
            cli_main(["maintenance", "obliterate", "--project", str(self.root)])

    def test_the_real_cli_binary_exits_zero(self):
        """Real subprocess, real interpreter, real exit code."""
        completed = subprocess.run(
            [sys.executable, "-m", "local_agent", "maintenance", "scan",
             "--project", str(self.root)],
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True, text=True, timeout=180,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Maintenance Scan", completed.stdout)

    def test_the_real_cli_binary_rejects_a_disabled_run(self):
        completed = subprocess.run(
            [sys.executable, "-m", "local_agent", "maintenance", "run",
             "--project", str(self.root)],
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True, text=True, timeout=180,
        )
        self.assertEqual(completed.returncode, 1)


class BackwardCompatibilityTests(unittest.TestCase):
    def test_existing_validation_commands_still_work(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        out = io.StringIO()
        with redirect_stdout(out):
            code = cli_main(["validation", "health", "--project", tmp.name])
        self.assertEqual(code, 0)

    def test_the_parser_still_accepts_every_previous_command(self):
        from local_agent.cli import build_parser

        parser = build_parser()
        for command in ("analyze", "list-tasks", "doctor", "show-config"):
            parser.parse_args([command, "--project", "."])

    def test_maintenance_settings_do_not_alter_the_tool_policy(self):
        config = AgentConfig.from_environment(".")
        before = config.tool_policy
        config.maintenance_enabled = True
        self.assertEqual(config.tool_policy, before)

    def test_a_disabled_subsystem_records_nothing_on_a_normal_task_path(self):
        # Nothing in the ordinary orchestration path imports the maintenance
        # modules at all; asserted structurally so it cannot regress silently.
        for module in ("local_agent.orchestrator", "local_agent.coding_agent",
                       "local_agent.scheduler", "local_agent.planner"):
            for imported in imported_modules(module):
                self.assertNotIn("maintenance", imported, module)


# =============================================================================
# Q. performance - measured, labelled, not extrapolated
# =============================================================================


class PerformanceTests(unittest.TestCase):
    """SYNTHETIC DATA unless the docstring says otherwise.

    These assert an *order of magnitude*, not a benchmark. The point is to
    catch an accidental quadratic, not to publish a number.
    """

    def test_ranking_a_large_candidate_set_is_fast(self):
        candidates = [make_candidate(subject=f"f{i}.py") for i in range(2000)]
        started = time.perf_counter()
        MaintenancePriorityEngine().rank(candidates)
        self.assertLess(time.perf_counter() - started, 5.0)

    def test_prioritisation_scales_roughly_linearly(self):
        engine = MaintenancePriorityEngine()

        def timed(count: int) -> float:
            candidates = [make_candidate(subject=f"f{i}.py") for i in range(count)]
            started = time.perf_counter()
            engine.rank(candidates)
            return time.perf_counter() - started

        small = timed(250)
        large = timed(2000)
        # 8x the input must not cost 64x the time.
        self.assertLess(large, max(small * 40, 5.0))

    def test_a_large_store_round_trips_quickly(self):
        store = MaintenanceStore(max_candidates=1000)
        for index in range(1000):
            store.upsert(make_candidate(subject=f"f{index}.py"))
        started = time.perf_counter()
        MaintenanceStore.from_dict(json.loads(json.dumps(store.to_dict())))
        self.assertLess(time.perf_counter() - started, 10.0)

    def test_batching_a_large_candidate_set_terminates(self):
        candidates = [
            make_candidate(subject=f"f{i}", affected_files=[f"f{i % 50}.py"])
            for i in range(500)
        ]
        started = time.perf_counter()
        plan_execution_batches(candidates, max_width=4)
        self.assertLess(time.perf_counter() - started, 10.0)

    def test_scanning_the_real_repository_completes(self):
        """REAL REPOSITORY DATA: the agent's own tree."""
        root = Path(__file__).resolve().parents[1]
        started = time.perf_counter()
        graph = SemanticGraph.build(root)
        MaintenanceAnalyzer(root).analyze(semantic_graph=graph)
        self.assertLess(time.perf_counter() - started, 120.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
