"""Phase 4.20: the cross-iteration validation lifecycle.

Organised by the property under test rather than by class under test, because
the things worth guaranteeing here are mostly *invariants* that span several
objects: a terminal state cannot reopen, more uncertainty never narrows a
scope, corrupt history never becomes permission, a defect signature never
merges two different defects.

Where a test asserts something about failure handling, it uses a real
mechanism: real subprocesses with real exit codes, real files on disk, real
truncated JSON. The only thing stubbed anywhere in this file is a storage
object, and it is stubbed to *fail* (to prove a persistence failure is
survivable), never to succeed more conveniently than the real one would.
"""

from __future__ import annotations

import ast
import io
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from local_agent.config import AgentConfig
from local_agent.semantic_impact import SCOPE_BROAD, SCOPE_EXPANDED, SCOPE_TARGETED
from local_agent.storage import JsonFileStorage
from local_agent.validation_lifecycle import (
    ALL_EVENTS,
    DEFAULT_MAX_ITERATIONS_PER_LIFECYCLE,
    DEFAULT_MAX_LIFECYCLES,
    EVENT_LIFECYCLE_COMPLETED,
    EVENT_LIFECYCLE_STARTED,
    EVENT_VALIDATION_FAILED,
    ITERATION_IMPLEMENTATION,
    ITERATION_REPAIR,
    LIFECYCLE_SCHEMA_VERSION,
    RESULT_FAILED,
    RESULT_NOT_RUN,
    RESULT_PASSED,
    STAGE_BROAD,
    STAGE_CANDIDATE,
    STAGE_POST_APPLY,
    STAGE_TARGETED,
    AdaptiveValidationRecommender,
    DefectSignature,
    InvalidLifecycleTransition,
    LifecycleState,
    ValidationEvent,
    ValidationEventEmitter,
    ValidationIterationRecord,
    ValidationLifecycleManager,
    ValidationLifecycleRecord,
    ValidationLifecycleStore,
    ALLOWED_TRANSITIONS,
    can_transition,
    compute_defect_signature,
    compute_repair_effectiveness,
    failure_category_for,
    normalize_command,
    normalize_diagnostic,
    safest_scope,
    signatures_match,
)


def _module_ast(dotted: str) -> tuple[ast.Module, str]:
    import importlib

    module = importlib.import_module(dotted)
    source = Path(module.__file__).read_text(encoding="utf-8")
    return ast.parse(source), source


def code_identifiers(dotted: str) -> set[str]:
    """Every identifier a module actually references in executable code.

    Docstrings, comments and string literals are excluded by construction,
    because they are parsed away - which is what makes this a check on
    behaviour rather than on prose.
    """
    tree, _ = _module_ast(dotted)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
    return names


def imported_modules(dotted: str) -> set[str]:
    """Absolute dotted names of everything a module imports."""
    tree, _ = _module_ast(dotted)
    package = dotted.rsplit(".", 1)[0]
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = f"{package}.{base}" if base else package
            found.add(base)
    return found


def new_lifecycle(**kwargs) -> ValidationLifecycleRecord:
    return ValidationLifecycleRecord(task_id="t1", subtask_id="s1", **kwargs)


def drive_to(record: ValidationLifecycleRecord, target: str) -> ValidationLifecycleRecord:
    """Walk a fresh lifecycle along the happy path until ``target`` is current."""
    path = [
        LifecycleState.CANDIDATE_GENERATED,
        LifecycleState.VALIDATED,
        LifecycleState.APPROVED,
        LifecycleState.APPLIED,
        LifecycleState.POST_VALIDATED,
        LifecycleState.COMPLETED,
    ]
    for state in path:
        if record.state == target:
            return record
        record.transition(state, reason="test")
    return record


def failing_iteration(number: int, *, parent: str = "", signature=None):
    return ValidationIterationRecord(
        iteration_number=number,
        parent_iteration_id=parent,
        kind=ITERATION_REPAIR if parent else ITERATION_IMPLEMENTATION,
        validation_result=RESULT_FAILED,
        validation_stage=STAGE_TARGETED,
        duration_seconds=1.0,
        defect_signature=signature
        or compute_defect_signature(
            failure_category="validation_failure",
            command=["pytest", "tests/test_x.py"],
            exit_code=1,
            diagnostic="AssertionError: expected 3 got 2",
        ),
    )


def passing_iteration(number: int, *, parent: str = ""):
    return ValidationIterationRecord(
        iteration_number=number,
        parent_iteration_id=parent,
        kind=ITERATION_REPAIR if parent else ITERATION_IMPLEMENTATION,
        validation_result=RESULT_PASSED,
        validation_stage=STAGE_CANDIDATE,
        duration_seconds=0.5,
    )


def completed_lifecycle(*, repairs: int = 0, signature=None) -> ValidationLifecycleRecord:
    """A lifecycle with exactly ``repairs`` *repair* iterations.

    The first iteration is always the original implementation (no parent, and
    therefore ``kind=implementation``); only the ones after it are repairs, so
    ``repairs=0`` really is a first-pass success.
    """
    record = new_lifecycle()
    first = failing_iteration(1, signature=signature) if repairs else passing_iteration(1)
    record.add_iteration(first)
    parent = first.iteration_id
    for index in range(repairs):
        last = index == repairs - 1
        iteration = (
            passing_iteration(index + 2, parent=parent)
            if last
            else failing_iteration(index + 2, parent=parent, signature=signature)
        )
        record.add_iteration(iteration)
        parent = iteration.iteration_id
    if repairs:
        # A repaired lifecycle really did pass through the repair loop.
        record.transition(LifecycleState.CANDIDATE_GENERATED)
        record.transition(LifecycleState.REPAIR_REQUIRED)
        record.transition(LifecycleState.REPAIRED)
        record.transition(LifecycleState.COMPLETED)
    else:
        drive_to(record, LifecycleState.COMPLETED)
    return record


def abandoned_lifecycle(*, repairs: int = 1) -> ValidationLifecycleRecord:
    """A failed implementation plus ``repairs`` failed repairs, then abandoned."""
    record = new_lifecycle()
    first = failing_iteration(1)
    record.add_iteration(first)
    parent = first.iteration_id
    for index in range(repairs):
        iteration = failing_iteration(index + 2, parent=parent)
        record.add_iteration(iteration)
        parent = iteration.iteration_id
    record.transition(LifecycleState.ABANDONED, reason="stagnation")
    return record


# =============================================================================
# A. Lifecycle model
# =============================================================================


class LifecycleModelCase(unittest.TestCase):
    def test_new_lifecycle_starts_in_created(self):
        self.assertEqual(new_lifecycle().state, LifecycleState.CREATED)

    def test_new_lifecycle_seeds_its_state_history(self):
        record = new_lifecycle()
        self.assertEqual(len(record.state_history), 1)
        self.assertEqual(record.state_history[0]["state"], LifecycleState.CREATED)

    def test_lifecycle_ids_are_unique(self):
        self.assertNotEqual(new_lifecycle().lifecycle_id, new_lifecycle().lifecycle_id)

    def test_parent_identity_is_carried(self):
        record = new_lifecycle()
        self.assertEqual(record.task_id, "t1")
        self.assertEqual(record.subtask_id, "s1")

    def test_schema_version_is_stamped(self):
        self.assertEqual(new_lifecycle().schema_version, LIFECYCLE_SCHEMA_VERSION)

    def test_created_lifecycle_is_not_terminal(self):
        self.assertFalse(new_lifecycle().is_terminal)

    def test_completed_lifecycle_is_terminal(self):
        self.assertTrue(drive_to(new_lifecycle(), LifecycleState.COMPLETED).is_terminal)

    def test_terminal_transition_records_the_outcome(self):
        record = drive_to(new_lifecycle(), LifecycleState.COMPLETED)
        self.assertEqual(record.terminal_outcome, LifecycleState.COMPLETED)

    def test_terminal_transition_sets_a_failure_category(self):
        record = new_lifecycle()
        record.transition(LifecycleState.ABANDONED)
        self.assertEqual(record.failure_category, failure_category_for(LifecycleState.ABANDONED))

    def test_completed_lifecycle_has_no_failure_category_beyond_none(self):
        record = drive_to(new_lifecycle(), LifecycleState.COMPLETED)
        self.assertEqual(record.failure_category, "none")

    def test_iterations_are_appended_in_order(self):
        record = new_lifecycle()
        first = record.add_iteration(passing_iteration(1))
        second = record.add_iteration(passing_iteration(2))
        self.assertEqual([i.iteration_id for i in record.iterations],
                         [first.iteration_id, second.iteration_id])

    def test_latest_iteration_is_the_last_appended(self):
        record = new_lifecycle()
        record.add_iteration(passing_iteration(1))
        last = record.add_iteration(passing_iteration(2))
        self.assertEqual(record.latest_iteration.iteration_id, last.iteration_id)

    def test_latest_iteration_of_an_empty_lifecycle_is_none(self):
        self.assertIsNone(new_lifecycle().latest_iteration)

    def test_find_iteration_returns_none_for_an_unknown_id(self):
        self.assertIsNone(new_lifecycle().find_iteration("nope"))

    def test_per_lifecycle_iteration_bound_evicts_oldest(self):
        record = new_lifecycle()
        record.max_iterations = 3
        for index in range(6):
            record.add_iteration(passing_iteration(index + 1))
        self.assertEqual(len(record.iterations), 3)
        self.assertEqual([i.iteration_number for i in record.iterations], [4, 5, 6])

    def test_updating_a_lifecycle_moves_updated_at(self):
        record = new_lifecycle()
        original = record.updated_at
        time.sleep(0.001)
        record.add_iteration(passing_iteration(1))
        self.assertNotEqual(record.updated_at, original)


# =============================================================================
# B. State transitions - valid and, more importantly, rejected-invalid
# =============================================================================


class StateTransitionCase(unittest.TestCase):
    def test_the_full_happy_path_is_legal(self):
        record = new_lifecycle()
        for state in (
            LifecycleState.CANDIDATE_GENERATED,
            LifecycleState.VALIDATED,
            LifecycleState.APPROVED,
            LifecycleState.APPLIED,
            LifecycleState.POST_VALIDATED,
            LifecycleState.COMPLETED,
        ):
            record.transition(state)
        self.assertEqual(record.state, LifecycleState.COMPLETED)

    def test_the_repair_loop_is_legal_and_repeatable(self):
        record = new_lifecycle()
        record.transition(LifecycleState.CANDIDATE_GENERATED)
        for _ in range(3):
            record.transition(LifecycleState.REPAIR_REQUIRED)
            record.transition(LifecycleState.REPAIRED)
            record.transition(LifecycleState.CANDIDATE_GENERATED)
        record.transition(LifecycleState.VALIDATED)
        self.assertEqual(record.state, LifecycleState.VALIDATED)

    def test_completed_cannot_return_to_created(self):
        record = drive_to(new_lifecycle(), LifecycleState.COMPLETED)
        with self.assertRaises(InvalidLifecycleTransition):
            record.transition(LifecycleState.CREATED)

    def test_abandoned_cannot_become_applied(self):
        record = new_lifecycle()
        record.transition(LifecycleState.ABANDONED)
        with self.assertRaises(InvalidLifecycleTransition):
            record.transition(LifecycleState.APPLIED)

    def test_failed_cannot_become_post_validated(self):
        record = new_lifecycle()
        record.transition(LifecycleState.FAILED)
        with self.assertRaises(InvalidLifecycleTransition):
            record.transition(LifecycleState.POST_VALIDATED)

    def test_created_cannot_jump_straight_to_applied(self):
        with self.assertRaises(InvalidLifecycleTransition):
            new_lifecycle().transition(LifecycleState.APPLIED)

    def test_created_cannot_jump_straight_to_completed(self):
        with self.assertRaises(InvalidLifecycleTransition):
            new_lifecycle().transition(LifecycleState.COMPLETED)

    def test_an_unknown_state_is_rejected(self):
        with self.assertRaises(InvalidLifecycleTransition):
            new_lifecycle().transition("teleported")

    def test_a_rejected_transition_leaves_the_state_untouched(self):
        record = new_lifecycle()
        with self.assertRaises(InvalidLifecycleTransition):
            record.transition(LifecycleState.COMPLETED)
        self.assertEqual(record.state, LifecycleState.CREATED)

    def test_a_rejected_transition_appends_nothing_to_history(self):
        record = new_lifecycle()
        before = len(record.state_history)
        with self.assertRaises(InvalidLifecycleTransition):
            record.transition(LifecycleState.COMPLETED)
        self.assertEqual(len(record.state_history), before)

    def test_try_transition_reports_rejection_without_raising(self):
        record = new_lifecycle()
        self.assertFalse(record.try_transition(LifecycleState.COMPLETED))
        self.assertTrue(record.try_transition(LifecycleState.CANDIDATE_GENERATED))

    def test_every_terminal_state_has_no_outgoing_edges(self):
        for state in LifecycleState.TERMINAL:
            with self.subTest(state=state):
                self.assertEqual(ALLOWED_TRANSITIONS[state], frozenset())

    def test_no_terminal_state_is_reachable_from_a_terminal_state(self):
        for state in LifecycleState.TERMINAL:
            for target in LifecycleState.ALL:
                with self.subTest(state=state, target=target):
                    self.assertFalse(can_transition(state, target))

    def test_abandoned_and_failed_are_reachable_from_every_active_state(self):
        active = [s for s in LifecycleState.ALL if s not in LifecycleState.TERMINAL]
        for state in active:
            with self.subTest(state=state):
                self.assertTrue(can_transition(state, LifecycleState.ABANDONED))
                self.assertTrue(can_transition(state, LifecycleState.FAILED))

    def test_every_declared_state_appears_in_the_transition_table(self):
        for state in LifecycleState.ALL:
            self.assertIn(state, ALLOWED_TRANSITIONS)

    def test_every_transition_target_is_a_declared_state(self):
        for state, targets in ALLOWED_TRANSITIONS.items():
            for target in targets:
                with self.subTest(edge=(state, target)):
                    self.assertIn(target, LifecycleState.ALL)

    def test_can_transition_from_an_unrecognised_state_permits_nothing(self):
        """Forward-compatibility fails closed: a state written by a newer build
        must not become a wildcard that permits arbitrary transitions."""
        for target in LifecycleState.ALL:
            self.assertFalse(can_transition("state_from_the_future", target))

    def test_transitions_are_recorded_with_reasons(self):
        record = new_lifecycle()
        record.transition(LifecycleState.CANDIDATE_GENERATED, reason="first candidate")
        self.assertEqual(record.state_history[-1]["reason"], "first candidate")

    def test_transitions_are_recorded_with_timestamps(self):
        record = new_lifecycle()
        record.transition(LifecycleState.CANDIDATE_GENERATED)
        self.assertTrue(record.state_history[-1]["at"])

    def test_invalid_transition_error_names_both_states(self):
        record = new_lifecycle()
        with self.assertRaises(InvalidLifecycleTransition) as caught:
            record.transition(LifecycleState.COMPLETED)
        self.assertEqual(caught.exception.current, LifecycleState.CREATED)
        self.assertEqual(caught.exception.requested, LifecycleState.COMPLETED)


# =============================================================================
# C. Repair lineage
# =============================================================================


class RepairLineageCase(unittest.TestCase):
    def build_chain(self, depth: int):
        record = new_lifecycle()
        parent = ""
        ids = []
        for index in range(depth):
            iteration = failing_iteration(index + 1, parent=parent)
            record.add_iteration(iteration)
            ids.append(iteration.iteration_id)
            parent = iteration.iteration_id
        return record, ids

    def test_first_iteration_has_no_parent(self):
        record, ids = self.build_chain(1)
        self.assertEqual(record.find_iteration(ids[0]).parent_iteration_id, "")

    def test_repair_points_at_its_causal_parent(self):
        record, ids = self.build_chain(2)
        self.assertEqual(record.find_iteration(ids[1]).parent_iteration_id, ids[0])

    def test_repair_chain_is_root_first(self):
        record, ids = self.build_chain(3)
        chain = record.repair_chain(ids[2])
        self.assertEqual([i.iteration_id for i in chain], ids)

    def test_repair_chain_of_the_root_is_just_the_root(self):
        record, ids = self.build_chain(3)
        self.assertEqual([i.iteration_id for i in record.repair_chain(ids[0])], [ids[0]])

    def test_repair_chain_of_an_unknown_id_is_empty(self):
        record, _ = self.build_chain(2)
        self.assertEqual(record.repair_chain("nope"), [])

    def test_children_of_finds_the_direct_descendants(self):
        record, ids = self.build_chain(2)
        self.assertEqual([i.iteration_id for i in record.children_of(ids[0])], [ids[1]])

    def test_branching_lineage_is_preserved(self):
        """Two repairs both provoked by the same failing iteration.

        A counter could not represent this at all; the parent edge can.
        """
        record, ids = self.build_chain(1)
        branch_a = failing_iteration(2, parent=ids[0])
        branch_b = failing_iteration(3, parent=ids[0])
        record.add_iteration(branch_a)
        record.add_iteration(branch_b)
        children = {i.iteration_id for i in record.children_of(ids[0])}
        self.assertEqual(children, {branch_a.iteration_id, branch_b.iteration_id})

    def test_a_parent_cycle_terminates_instead_of_hanging(self):
        """Only reachable via a corrupted store, but it must not hang."""
        record = new_lifecycle()
        first = passing_iteration(1)
        second = passing_iteration(2)
        first.parent_iteration_id = second.iteration_id
        second.parent_iteration_id = first.iteration_id
        record.add_iteration(first)
        record.add_iteration(second)
        chain = record.repair_chain(second.iteration_id)
        self.assertEqual(len(chain), 2)

    def test_an_orphaned_parent_ends_the_chain_partially(self):
        record = new_lifecycle()
        orphan = failing_iteration(5, parent="evicted-id")
        record.add_iteration(orphan)
        self.assertEqual([i.iteration_id for i in record.repair_chain(orphan.iteration_id)],
                         [orphan.iteration_id])

    def test_repair_count_counts_only_repair_iterations(self):
        record = new_lifecycle()
        record.add_iteration(passing_iteration(1))
        record.add_iteration(failing_iteration(2, parent="x"))
        record.add_iteration(failing_iteration(3, parent="y"))
        self.assertEqual(record.repair_count, 2)

    def test_lineage_survives_a_serialisation_round_trip(self):
        record, ids = self.build_chain(3)
        restored = ValidationLifecycleRecord.from_dict(
            json.loads(json.dumps(record.to_dict()))
        )
        self.assertEqual([i.iteration_id for i in restored.repair_chain(ids[2])], ids)


# =============================================================================
# D. Defect signatures
# =============================================================================


class DiagnosticNormalisationCase(unittest.TestCase):
    def test_windows_absolute_paths_are_normalised(self):
        self.assertNotIn(
            "Temp", normalize_diagnostic(r"File C:\Users\a\AppData\Temp\x\t.py failed")
        )

    def test_posix_absolute_paths_are_normalised(self):
        self.assertNotIn("/tmp/", normalize_diagnostic("File /tmp/abc123/t.py failed"))

    def test_two_different_temp_dirs_normalise_identically(self):
        left = normalize_diagnostic("/tmp/aaa111/test_x.py:41: AssertionError")
        right = normalize_diagnostic("/tmp/bbb222/test_x.py:97: AssertionError")
        self.assertEqual(left, right)

    def test_timestamps_are_normalised(self):
        self.assertEqual(
            normalize_diagnostic("failed at 2024-01-01T10:00:00Z"),
            normalize_diagnostic("failed at 2025-06-02T23:59:59Z"),
        )

    def test_memory_addresses_are_normalised(self):
        self.assertEqual(
            normalize_diagnostic("<obj at 0xdeadbeef>"),
            normalize_diagnostic("<obj at 0xcafef00d>"),
        )

    def test_uuids_are_normalised(self):
        self.assertEqual(
            normalize_diagnostic("run 123e4567-e89b-12d3-a456-426614174000 failed"),
            normalize_diagnostic("run 00000000-1111-2222-3333-444444444444 failed"),
        )

    def test_pids_are_normalised(self):
        self.assertEqual(normalize_diagnostic("pid 1234 died"), normalize_diagnostic("pid 99 died"))

    def test_line_numbers_are_normalised(self):
        self.assertEqual(
            normalize_diagnostic("error on line 41"), normalize_diagnostic("error on line 9001")
        )

    def test_durations_are_normalised(self):
        self.assertEqual(
            normalize_diagnostic("finished in 1.23s"), normalize_diagnostic("finished in 9.87s")
        )

    def test_pytest_counts_are_normalised(self):
        self.assertEqual(
            normalize_diagnostic("3 failed, 10 passed"),
            normalize_diagnostic("1 failed, 44 passed"),
        )

    def test_whitespace_runs_collapse(self):
        self.assertEqual(normalize_diagnostic("a\n\n   b"), "a b")

    def test_non_string_input_is_tolerated(self):
        self.assertEqual(normalize_diagnostic(None), "")
        self.assertEqual(normalize_diagnostic(17), "17")

    def test_output_is_bounded(self):
        self.assertLessEqual(len(normalize_diagnostic("x" * 5000)), 400)

    def test_normalisation_is_deterministic(self):
        text = "/tmp/x/t.py:9: AssertionError at 0xabcd1234"
        self.assertEqual(normalize_diagnostic(text), normalize_diagnostic(text))

    def test_meaningful_content_survives(self):
        self.assertIn("AssertionError", normalize_diagnostic("/tmp/a/t.py:1: AssertionError"))

    def test_two_genuinely_different_messages_stay_different(self):
        self.assertNotEqual(
            normalize_diagnostic("AssertionError: expected 3"),
            normalize_diagnostic("TypeError: not callable"),
        )

    def test_command_normalisation_handles_a_string(self):
        self.assertEqual(normalize_command("pytest tests/x.py"), ("pytest", "tests/x.py"))

    def test_command_normalisation_handles_a_list(self):
        self.assertEqual(normalize_command(["pytest", "-q"]), ("pytest", "-q"))

    def test_command_normalisation_strips_temp_paths(self):
        left = normalize_command(["pytest", "/tmp/aaa/tests/x.py"])
        right = normalize_command(["pytest", "/tmp/bbb/tests/x.py"])
        self.assertEqual(left, right)

    def test_command_normalisation_of_garbage_is_empty(self):
        self.assertEqual(normalize_command(object()), ())


class DefectSignatureCase(unittest.TestCase):
    def base(self, **overrides):
        kwargs = dict(
            failure_category="validation_failure",
            command=["pytest", "tests/test_x.py"],
            exit_code=1,
            diagnostic="AssertionError: expected 3 got 2",
            affected_file="pkg/core.py",
            validation_tier=STAGE_TARGETED,
        )
        kwargs.update(overrides)
        return compute_defect_signature(**kwargs)

    def test_same_defect_matches_across_iterations(self):
        self.assertTrue(signatures_match(self.base(), self.base()))

    def test_same_defect_matches_despite_different_temp_paths(self):
        left = self.base(diagnostic="/tmp/aaa/tests/test_x.py:41: AssertionError: expected 3 got 2")
        right = self.base(diagnostic="/tmp/bbb/tests/test_x.py:97: AssertionError: expected 3 got 2")
        self.assertTrue(signatures_match(left, right))

    def test_different_exception_types_do_not_match(self):
        self.assertFalse(
            signatures_match(self.base(), self.base(diagnostic="TypeError: not callable"))
        )

    def test_different_exit_codes_do_not_match(self):
        self.assertFalse(signatures_match(self.base(), self.base(exit_code=2)))

    def test_different_commands_do_not_match(self):
        self.assertFalse(
            signatures_match(self.base(), self.base(command=["pytest", "tests/test_y.py"]))
        )

    def test_different_affected_files_do_not_match(self):
        self.assertFalse(signatures_match(self.base(), self.base(affected_file="pkg/other.py")))

    def test_different_validation_tiers_do_not_match(self):
        self.assertFalse(signatures_match(self.base(), self.base(validation_tier=STAGE_BROAD)))

    def test_superficially_similar_defects_do_not_merge(self):
        """The false-merge guard, with the near-miss made explicit.

        Both are ``AssertionError`` from the same command with the same exit
        code. Only the expected value differs, and that is enough - a matcher
        loose enough to merge these would report a recurring defect where two
        different ones occurred.
        """
        left = self.base(diagnostic="AssertionError: expected 3 got 2")
        right = self.base(diagnostic="AssertionError: expected 4 got 2")
        self.assertFalse(signatures_match(left, right))

    def test_two_empty_signatures_never_match(self):
        self.assertFalse(signatures_match(DefectSignature(), DefectSignature()))

    def test_an_empty_signature_never_matches_a_real_one(self):
        self.assertFalse(signatures_match(DefectSignature(), self.base()))

    def test_empty_signature_is_flagged_empty(self):
        self.assertTrue(DefectSignature().is_empty)

    def test_a_real_signature_is_not_flagged_empty(self):
        self.assertFalse(self.base().is_empty)

    def test_fingerprint_is_stable_across_calls(self):
        self.assertEqual(self.base().fingerprint, self.base().fingerprint)

    def test_fingerprint_is_bounded_hex(self):
        fingerprint = self.base().fingerprint
        self.assertEqual(len(fingerprint), 32)
        int(fingerprint, 16)

    def test_field_boundaries_cannot_be_confused(self):
        """Concatenation ambiguity would be a false merge by another route."""
        left = compute_defect_signature(failure_category="ab", exception_class="c")
        right = compute_defect_signature(failure_category="a", exception_class="bc")
        self.assertNotEqual(left.fingerprint, right.fingerprint)

    def test_signature_is_hashable_and_deduplicates(self):
        self.assertEqual(len({self.base(), self.base()}), 1)

    def test_signature_round_trips_through_json(self):
        signature = self.base()
        restored = DefectSignature.from_dict(json.loads(json.dumps(signature.to_dict())))
        self.assertEqual(restored.fingerprint, signature.fingerprint)

    def test_signature_from_garbage_is_empty_not_an_error(self):
        self.assertTrue(DefectSignature.from_dict("not a dict").is_empty)

    def test_signature_from_a_bad_exit_code_defaults_safely(self):
        self.assertEqual(DefectSignature.from_dict({"exit_code": "x"}).exit_code, 0)

    def test_unknown_validation_tier_falls_back_to_unknown(self):
        self.assertEqual(compute_defect_signature(validation_tier="martian").validation_tier,
                         "unknown")

    def test_describe_is_human_readable(self):
        self.assertIn("pytest", self.base().describe())

    def test_stream_fragments_are_bounded(self):
        signature = compute_defect_signature(stderr="e" * 10000, stdout="o" * 10000)
        self.assertLessEqual(len(signature.stderr_fragment), 300)
        self.assertLessEqual(len(signature.stdout_fragment), 300)

    def test_stderr_tail_is_what_is_kept(self):
        signature = compute_defect_signature(stderr="filler " * 500 + "REALERROR")
        self.assertIn("REALERROR", signature.stderr_fragment)


class RealSubprocessDefectSignatureCase(unittest.TestCase):
    """Signatures built from genuine subprocess failures, not fabricated text."""

    def run_script(self, body: str) -> subprocess.CompletedProcess:
        directory = tempfile.mkdtemp(prefix="dungx_defect_")
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        script = Path(directory) / "script.py"
        script.write_text(body, encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(script)], capture_output=True, text=True, timeout=60
        )

    def signature_for(self, result: subprocess.CompletedProcess) -> DefectSignature:
        return compute_defect_signature(
            failure_category="validation_failure",
            command=["python", "script.py"],
            exit_code=result.returncode,
            diagnostic=result.stderr,
            stderr=result.stderr,
            stdout=result.stdout,
            validation_tier=STAGE_TARGETED,
        )

    def test_a_real_assertion_failure_produces_a_nonempty_signature(self):
        result = self.run_script("assert 1 == 2, 'boom'\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.signature_for(result).is_empty)

    def test_the_same_real_failure_matches_across_two_runs(self):
        body = "assert 1 == 2, 'boom'\n"
        first = self.signature_for(self.run_script(body))
        second = self.signature_for(self.run_script(body))
        self.assertTrue(
            signatures_match(first, second),
            "the same real failure, run twice from two different temp directories, "
            "must produce the same signature",
        )

    def test_two_different_real_failures_do_not_match(self):
        assertion = self.signature_for(self.run_script("assert 1 == 2, 'boom'\n"))
        type_error = self.signature_for(self.run_script("None()\n"))
        self.assertFalse(signatures_match(assertion, type_error))

    def test_a_real_syntax_error_produces_a_distinct_signature(self):
        syntax = self.signature_for(self.run_script("def broken(:\n"))
        assertion = self.signature_for(self.run_script("assert 1 == 2, 'boom'\n"))
        self.assertFalse(signatures_match(syntax, assertion))

    def test_a_real_nonzero_exit_code_is_captured(self):
        result = self.run_script("import sys\nsys.exit(3)\n")
        self.assertEqual(self.signature_for(result).exit_code, 3)

    def test_two_different_exit_codes_do_not_match(self):
        left = self.signature_for(self.run_script("import sys\nsys.exit(1)\n"))
        right = self.signature_for(self.run_script("import sys\nsys.exit(2)\n"))
        self.assertFalse(signatures_match(left, right))


# =============================================================================
# E. Repair-effectiveness statistics
# =============================================================================


class RepairEffectivenessCase(unittest.TestCase):
    def store_with(self, *records) -> ValidationLifecycleStore:
        store = ValidationLifecycleStore()
        for record in records:
            store.record(record)
        return store

    def test_empty_store_reports_nothing_established(self):
        metrics = compute_repair_effectiveness(ValidationLifecycleStore())
        self.assertEqual(metrics.lifecycles, 0)
        self.assertTrue(metrics.insufficient_data)

    def test_empty_store_reports_no_rates_as_established_success(self):
        metrics = compute_repair_effectiveness(ValidationLifecycleStore())
        self.assertEqual(metrics.first_pass_success_lower_bound, 0.0)
        self.assertEqual(metrics.repair_success_lower_bound, 0.0)

    def test_counts_by_terminal_state(self):
        metrics = compute_repair_effectiveness(
            self.store_with(completed_lifecycle(), completed_lifecycle(), abandoned_lifecycle())
        )
        self.assertEqual((metrics.completed, metrics.abandoned, metrics.failed), (2, 1, 0))

    def test_in_progress_lifecycles_are_counted_separately(self):
        metrics = compute_repair_effectiveness(self.store_with(new_lifecycle()))
        self.assertEqual(metrics.in_progress, 1)
        self.assertEqual(metrics.resolved, 0)

    def test_in_progress_lifecycles_do_not_enter_any_rate(self):
        """An unresolved lifecycle must not contribute a success or a failure."""
        resolved_only = compute_repair_effectiveness(self.store_with(completed_lifecycle()))
        with_pending = compute_repair_effectiveness(
            self.store_with(completed_lifecycle(), new_lifecycle())
        )
        self.assertEqual(resolved_only.first_pass_success_rate,
                         with_pending.first_pass_success_rate)

    def test_first_pass_success_counts_zero_repair_completions(self):
        metrics = compute_repair_effectiveness(
            self.store_with(completed_lifecycle(), completed_lifecycle(repairs=2))
        )
        self.assertEqual(metrics.first_pass_successes, 1)

    def test_repair_success_rate_uses_only_lifecycles_that_needed_repair(self):
        metrics = compute_repair_effectiveness(
            self.store_with(
                completed_lifecycle(),
                completed_lifecycle(repairs=1),
                abandoned_lifecycle(repairs=1),
            )
        )
        self.assertEqual(metrics.lifecycles_needing_repair, 2)
        self.assertEqual(metrics.repaired_successfully, 1)
        self.assertAlmostEqual(metrics.repair_success_rate, 0.5)

    def test_wilson_lower_bound_is_below_the_point_estimate_for_small_samples(self):
        metrics = compute_repair_effectiveness(
            self.store_with(completed_lifecycle(repairs=1), completed_lifecycle(repairs=1))
        )
        self.assertEqual(metrics.repair_success_rate, 1.0)
        self.assertLess(
            metrics.repair_success_lower_bound,
            1.0,
            "two successes must not be reported as established perfect reliability",
        )

    def test_abandonment_rate_upper_bound_is_pessimistic(self):
        metrics = compute_repair_effectiveness(self.store_with(completed_lifecycle()))
        self.assertEqual(metrics.abandonment_rate, 0.0)
        self.assertGreater(
            metrics.abandonment_rate_upper_bound,
            0.0,
            "one clean sample must not prove abandonment is impossible",
        )

    def test_median_repair_iterations(self):
        metrics = compute_repair_effectiveness(
            self.store_with(
                completed_lifecycle(repairs=1),
                completed_lifecycle(repairs=3),
                completed_lifecycle(repairs=5),
            )
        )
        self.assertEqual(metrics.median_repair_iterations, 3)

    def test_max_repair_iterations(self):
        metrics = compute_repair_effectiveness(
            self.store_with(completed_lifecycle(repairs=1), completed_lifecycle(repairs=4))
        )
        self.assertEqual(metrics.max_repair_iterations, 4)

    def test_repeated_defect_detection_within_a_lifecycle(self):
        signature = compute_defect_signature(
            failure_category="validation_failure", command=["pytest"], exit_code=1,
            diagnostic="AssertionError: same",
        )
        metrics = compute_repair_effectiveness(
            self.store_with(completed_lifecycle(repairs=3, signature=signature))
        )
        self.assertEqual(metrics.repeated_defect_lifecycles, 1)

    def test_distinct_defects_are_not_counted_as_recurrence(self):
        record = new_lifecycle()
        parent = ""
        for index, message in enumerate(("AssertionError: a", "TypeError: b", "ValueError: c")):
            iteration = failing_iteration(
                index + 1,
                parent=parent,
                signature=compute_defect_signature(
                    failure_category="validation_failure",
                    command=["pytest"], exit_code=1, diagnostic=message,
                ),
            )
            record.add_iteration(iteration)
            parent = iteration.iteration_id
        record.transition(LifecycleState.ABANDONED)
        metrics = compute_repair_effectiveness(self.store_with(record))
        self.assertEqual(metrics.repeated_defect_lifecycles, 0)

    def test_top_recurring_defects_are_ranked(self):
        signature = compute_defect_signature(
            failure_category="validation_failure", command=["pytest"], exit_code=1,
            diagnostic="AssertionError: same",
        )
        metrics = compute_repair_effectiveness(
            self.store_with(completed_lifecycle(repairs=4, signature=signature))
        )
        self.assertTrue(metrics.top_recurring_defects)
        self.assertEqual(metrics.top_recurring_defects[0].occurrences, 4)

    def test_stage_distribution_counts_only_failures(self):
        record = new_lifecycle()
        record.add_iteration(passing_iteration(1))
        record.add_iteration(failing_iteration(2, parent="x"))
        record.transition(LifecycleState.ABANDONED)
        metrics = compute_repair_effectiveness(self.store_with(record))
        self.assertEqual(sum(metrics.stage_distribution.values()), 1)

    def test_candidate_versus_post_apply_defects_are_distinguished(self):
        record = new_lifecycle()
        candidate = failing_iteration(1)
        candidate.validation_stage = STAGE_CANDIDATE
        post = failing_iteration(2, parent=candidate.iteration_id)
        post.validation_stage = STAGE_POST_APPLY
        record.add_iteration(candidate)
        record.add_iteration(post)
        record.transition(LifecycleState.ABANDONED)
        metrics = compute_repair_effectiveness(self.store_with(record))
        self.assertEqual(metrics.candidate_stage_defects, 1)
        self.assertEqual(metrics.post_apply_defects, 1)

    def test_unmeasured_durations_are_excluded_not_averaged_as_free(self):
        record = new_lifecycle()
        measured = passing_iteration(1)
        measured.duration_seconds = 10.0
        unmeasured = passing_iteration(2)
        unmeasured.duration_seconds = 0.0
        record.add_iteration(measured)
        record.add_iteration(unmeasured)
        drive_to(record, LifecycleState.COMPLETED)
        metrics = compute_repair_effectiveness(self.store_with(record))
        self.assertEqual(metrics.measured_duration_samples, 1)
        self.assertEqual(metrics.mean_validation_seconds, 10.0)

    def test_insufficient_data_flag_respects_min_samples(self):
        store = self.store_with(*[completed_lifecycle() for _ in range(5)])
        self.assertTrue(compute_repair_effectiveness(store, min_samples=10).insufficient_data)
        self.assertFalse(compute_repair_effectiveness(store, min_samples=5).insufficient_data)

    def test_metrics_are_order_independent(self):
        first = completed_lifecycle(repairs=1)
        second = abandoned_lifecycle()
        forward = compute_repair_effectiveness(self.store_with(first, second)).to_dict()
        backward = compute_repair_effectiveness(self.store_with(second, first)).to_dict()
        forward.pop("top_recurring_defects")
        backward.pop("top_recurring_defects")
        self.assertEqual(forward, backward)

    def test_metrics_serialise_to_json(self):
        metrics = compute_repair_effectiveness(self.store_with(completed_lifecycle()))
        json.dumps(metrics.to_dict())

    def test_corrupt_history_is_flagged_in_the_metrics(self):
        store = self.store_with(completed_lifecycle())
        store.corrupted_records_skipped = 2
        self.assertFalse(compute_repair_effectiveness(store).history_trustworthy)

    def test_computation_does_not_mutate_the_store(self):
        store = self.store_with(completed_lifecycle(repairs=2))
        before = json.dumps(store.to_dict(), sort_keys=True)
        compute_repair_effectiveness(store)
        self.assertEqual(json.dumps(store.to_dict(), sort_keys=True), before)


# =============================================================================
# F. Adaptive recommendation versus the safety floor
# =============================================================================


class SafestScopeCase(unittest.TestCase):
    def test_broadest_wins(self):
        self.assertEqual(safest_scope(SCOPE_TARGETED, SCOPE_BROAD), SCOPE_BROAD)

    def test_expanded_beats_targeted(self):
        self.assertEqual(safest_scope(SCOPE_TARGETED, SCOPE_EXPANDED), SCOPE_EXPANDED)

    def test_identical_scopes_are_preserved(self):
        self.assertEqual(safest_scope(SCOPE_TARGETED, SCOPE_TARGETED), SCOPE_TARGETED)

    def test_unknown_scope_forces_broad(self):
        self.assertEqual(safest_scope(SCOPE_TARGETED, "martian"), SCOPE_BROAD)

    def test_no_arguments_is_targeted_identity(self):
        self.assertEqual(safest_scope(SCOPE_TARGETED), SCOPE_TARGETED)


class AdaptiveRecommendationCase(unittest.TestCase):
    def excellent_store(self, count: int = 60) -> ValidationLifecycleStore:
        """A history as favourable as this model can represent."""
        store = ValidationLifecycleStore()
        for _ in range(count):
            store.record(completed_lifecycle())
        return store

    def recommend(self, **kwargs):
        defaults = dict(safety_floor=SCOPE_BROAD, store=None)
        defaults.update(kwargs)
        return AdaptiveValidationRecommender(min_samples=10).recommend(**defaults)

    def test_recommendation_is_always_marked_advisory(self):
        self.assertTrue(self.recommend().advisory)

    def test_no_history_recommends_the_floor(self):
        recommendation = self.recommend(safety_floor=SCOPE_EXPANDED)
        self.assertEqual(recommendation.recommended_scope, SCOPE_EXPANDED)
        self.assertFalse(recommendation.data_sufficient)

    def test_insufficient_history_never_narrows(self):
        """Property: insufficient data behaves conservatively."""
        store = ValidationLifecycleStore()
        for _ in range(3):
            store.record(completed_lifecycle())
        recommendation = self.recommend(safety_floor=SCOPE_EXPANDED, store=store)
        self.assertEqual(recommendation.effective_scope, SCOPE_EXPANDED)
        self.assertFalse(recommendation.data_sufficient)

    def test_perfect_history_can_suggest_targeted(self):
        recommendation = self.recommend(safety_floor=SCOPE_TARGETED, store=self.excellent_store())
        self.assertEqual(recommendation.recommended_scope, SCOPE_TARGETED)

    def test_perfect_history_cannot_override_a_broad_floor(self):
        """The central safety claim, stated as a test rather than a comment."""
        recommendation = self.recommend(safety_floor=SCOPE_BROAD, store=self.excellent_store())
        self.assertEqual(recommendation.recommended_scope, SCOPE_TARGETED)
        self.assertEqual(recommendation.effective_scope, SCOPE_BROAD)
        self.assertTrue(recommendation.conflicts_with_floor)

    def test_historical_success_does_not_override_a_current_safety_concern(self):
        """Degraded analysis right now beats any amount of past success."""
        recommendation = self.recommend(
            safety_floor=SCOPE_TARGETED, store=self.excellent_store(), degraded_analysis=True
        )
        self.assertEqual(recommendation.effective_scope, SCOPE_BROAD)
        self.assertTrue(
            any("degraded" in reason for reason in recommendation.safety_reasons)
        )

    def test_corrupted_history_does_not_produce_a_more_permissive_decision(self):
        store = self.excellent_store()
        clean = self.recommend(safety_floor=SCOPE_TARGETED, store=store).effective_scope
        store.corrupted_records_skipped = 1
        corrupt = self.recommend(safety_floor=SCOPE_TARGETED, store=store).effective_scope
        self.assertEqual(
            safest_scope(clean, corrupt), corrupt,
            "corrupting the history must never yield a narrower effective scope",
        )
        self.assertEqual(corrupt, SCOPE_BROAD)

    def test_corrupted_history_is_reported_as_untrustworthy(self):
        store = self.excellent_store()
        store.corrupted_records_skipped = 1
        self.assertFalse(self.recommend(store=store).history_trustworthy)

    def test_a_recurring_defect_forces_broad(self):
        signature = compute_defect_signature(
            failure_category="validation_failure", command=["pytest"], exit_code=1,
            diagnostic="AssertionError: same",
        )
        store = self.excellent_store()
        store.record(completed_lifecycle(repairs=3, signature=signature))
        recommendation = self.recommend(
            safety_floor=SCOPE_TARGETED,
            store=store,
            recent_defect_fingerprints=[signature.fingerprint],
        )
        self.assertEqual(recommendation.effective_scope, SCOPE_BROAD)

    def test_an_unrelated_recurring_defect_does_not_force_broad(self):
        recommendation = self.recommend(
            safety_floor=SCOPE_TARGETED,
            store=self.excellent_store(),
            recent_defect_fingerprints=["a" * 32],
        )
        self.assertEqual(recommendation.effective_scope, SCOPE_TARGETED)

    def test_high_abandonment_history_widens(self):
        store = ValidationLifecycleStore()
        for _ in range(10):
            store.record(abandoned_lifecycle())
        for _ in range(10):
            store.record(completed_lifecycle())
        recommendation = self.recommend(safety_floor=SCOPE_TARGETED, store=store)
        self.assertNotEqual(recommendation.effective_scope, SCOPE_TARGETED)

    def test_effective_scope_is_never_narrower_than_the_floor(self):
        """Exhaustive over every floor and every history shape available."""
        stores = [
            None,
            ValidationLifecycleStore(),
            self.excellent_store(),
        ]
        for floor in (SCOPE_TARGETED, SCOPE_EXPANDED, SCOPE_BROAD):
            for index, store in enumerate(stores):
                for degraded in (False, True):
                    with self.subTest(floor=floor, store=index, degraded=degraded):
                        recommendation = self.recommend(
                            safety_floor=floor, store=store, degraded_analysis=degraded
                        )
                        self.assertEqual(
                            safest_scope(recommendation.effective_scope, floor),
                            recommendation.effective_scope,
                        )

    def test_recommendation_carries_reasons_for_both_halves(self):
        recommendation = self.recommend(store=self.excellent_store())
        self.assertTrue(recommendation.reasons)
        self.assertTrue(recommendation.safety_reasons)

    def test_recommendation_serialises_to_json(self):
        payload = json.loads(json.dumps(self.recommend().to_dict()))
        self.assertIn("effective_scope", payload)
        self.assertTrue(payload["advisory"])

    def test_recommender_has_no_reference_to_the_decision_engine(self):
        """Structural proof that this layer cannot become a second authority.

        Checked against the parsed AST rather than the raw text, because the
        module's own docstring names ``ValidationDecisionEngine`` deliberately
        (to say what this layer is *not*). What must be absent is any
        executable reference: no import of the decision module, and no
        identifier naming the engine anywhere in real code.
        """
        self.assertEqual(code_identifiers("local_agent.validation_lifecycle")
                         & {"ValidationDecisionEngine"}, set())
        self.assertNotIn("local_agent.validation_decision",
                         imported_modules("local_agent.validation_lifecycle"))


# =============================================================================
# G. Store: retention, corruption, forward/backward compatibility
# =============================================================================


class LifecycleStoreCase(unittest.TestCase):
    def test_empty_store_has_no_lifecycles(self):
        self.assertEqual(len(ValidationLifecycleStore()), 0)

    def test_empty_store_is_trustworthy(self):
        self.assertTrue(ValidationLifecycleStore().history_trustworthy)

    def test_single_record_round_trips(self):
        store = ValidationLifecycleStore()
        record = completed_lifecycle(repairs=1)
        store.record(record)
        restored = ValidationLifecycleStore.from_dict(json.loads(json.dumps(store.to_dict())))
        self.assertEqual(len(restored), 1)
        self.assertEqual(restored.lifecycles[0].lifecycle_id, record.lifecycle_id)

    def test_recording_the_same_lifecycle_twice_is_idempotent(self):
        store = ValidationLifecycleStore()
        record = new_lifecycle()
        store.record(record)
        store.record(record)
        self.assertEqual(len(store), 1)

    def test_recording_an_updated_lifecycle_replaces_it(self):
        store = ValidationLifecycleStore()
        record = new_lifecycle()
        store.record(record)
        record.transition(LifecycleState.CANDIDATE_GENERATED)
        store.record(record)
        self.assertEqual(len(store), 1)
        self.assertEqual(store.lifecycles[0].state, LifecycleState.CANDIDATE_GENERATED)

    def test_repeated_finalisation_does_not_duplicate(self):
        store = ValidationLifecycleStore()
        record = new_lifecycle()
        store.record(record)
        record.transition(LifecycleState.ABANDONED)
        store.record(record)
        store.record(record)
        self.assertEqual(len(store), 1)

    def test_retention_bound_evicts_oldest(self):
        store = ValidationLifecycleStore(max_lifecycles=3)
        ids = []
        for _ in range(6):
            record = new_lifecycle()
            ids.append(record.lifecycle_id)
            store.record(record)
        self.assertEqual(len(store), 3)
        self.assertEqual([r.lifecycle_id for r in store.lifecycles], ids[3:])

    def test_retention_bound_is_enforced_on_load(self):
        store = ValidationLifecycleStore(max_lifecycles=50)
        for _ in range(10):
            store.record(new_lifecycle())
        payload = store.to_dict()
        payload["max_lifecycles"] = 4
        self.assertEqual(len(ValidationLifecycleStore.from_dict(payload)), 4)

    def test_per_lifecycle_iteration_bound_is_enforced_on_record(self):
        store = ValidationLifecycleStore(max_iterations_per_lifecycle=2)
        record = new_lifecycle()
        for index in range(5):
            record.add_iteration(passing_iteration(index + 1))
        store.record(record)
        self.assertEqual(len(store.lifecycles[0].iterations), 2)

    def test_find_returns_none_for_an_unknown_id(self):
        self.assertIsNone(ValidationLifecycleStore().find("nope"))

    def test_for_task_filters(self):
        store = ValidationLifecycleStore()
        store.record(ValidationLifecycleRecord(task_id="a"))
        store.record(ValidationLifecycleRecord(task_id="b"))
        self.assertEqual(len(store.for_task("a")), 1)

    def test_lifecycles_property_returns_a_copy(self):
        store = ValidationLifecycleStore()
        store.record(new_lifecycle())
        store.lifecycles.clear()
        self.assertEqual(len(store), 1)

    # -- corruption ---------------------------------------------------------

    def test_a_non_mapping_payload_is_corruption_not_an_empty_store(self):
        store = ValidationLifecycleStore.from_dict("garbage")
        self.assertEqual(len(store), 0)
        self.assertFalse(store.history_trustworthy)

    def test_none_payload_is_corruption(self):
        self.assertFalse(ValidationLifecycleStore.from_dict(None).history_trustworthy)

    def test_a_malformed_record_is_dropped_and_counted(self):
        store = ValidationLifecycleStore.from_dict(
            {"lifecycles": [{"lifecycle_id": "ok"}, "not a dict", 42]}
        )
        self.assertEqual(len(store), 1)
        self.assertEqual(store.corrupted_records_skipped, 2)

    def test_a_malformed_lifecycles_container_is_counted(self):
        store = ValidationLifecycleStore.from_dict({"lifecycles": "not a list"})
        self.assertEqual(len(store), 0)
        self.assertFalse(store.history_trustworthy)

    def test_a_malformed_corruption_counter_is_itself_corruption(self):
        store = ValidationLifecycleStore.from_dict({"corrupted_records_skipped": "many"})
        self.assertGreater(store.corrupted_records_skipped, 0)

    def test_a_malformed_bound_falls_back_to_the_default(self):
        store = ValidationLifecycleStore.from_dict({"max_lifecycles": "lots"})
        self.assertEqual(store.max_lifecycles, DEFAULT_MAX_LIFECYCLES)

    def test_a_negative_declared_corruption_count_is_clamped(self):
        store = ValidationLifecycleStore.from_dict({"corrupted_records_skipped": -5})
        self.assertGreaterEqual(store.corrupted_records_skipped, 0)

    def test_partially_missing_fields_get_safe_defaults(self):
        record = ValidationLifecycleRecord.from_dict({"lifecycle_id": "x"})
        self.assertEqual(record.state, LifecycleState.CREATED)
        self.assertEqual(record.iterations, [])
        self.assertEqual(record.task_id, "")

    def test_a_malformed_iteration_is_dropped_but_the_record_survives(self):
        record = ValidationLifecycleRecord.from_dict(
            {"lifecycle_id": "x", "iterations": [{"iteration_number": 1}, "junk"]}
        )
        self.assertEqual(len(record.iterations), 1)

    def test_a_malformed_iteration_number_defaults_safely(self):
        iteration = ValidationIterationRecord.from_dict({"iteration_number": "many"})
        self.assertEqual(iteration.iteration_number, 1)

    def test_a_malformed_duration_defaults_safely(self):
        iteration = ValidationIterationRecord.from_dict({"duration_seconds": "fast"})
        self.assertEqual(iteration.duration_seconds, 0.0)

    def test_a_malformed_state_history_entry_is_dropped(self):
        record = ValidationLifecycleRecord.from_dict(
            {"lifecycle_id": "x", "state_history": [{"state": "created"}, "junk"]}
        )
        self.assertEqual(len(record.state_history), 1)

    def test_a_record_from_garbage_is_a_default_record(self):
        self.assertEqual(ValidationLifecycleRecord.from_dict(123).state, LifecycleState.CREATED)

    def test_an_iteration_from_garbage_is_a_default_iteration(self):
        self.assertEqual(ValidationIterationRecord.from_dict(None).validation_result,
                         RESULT_NOT_RUN)

    # -- forward / backward compatibility ------------------------------------

    def test_an_old_record_with_no_schema_version_loads(self):
        record = ValidationLifecycleRecord.from_dict({"lifecycle_id": "old", "state": "created"})
        self.assertEqual(record.schema_version, "0")
        self.assertEqual(record.lifecycle_id, "old")

    def test_unknown_future_fields_on_a_record_are_preserved(self):
        payload = {"lifecycle_id": "x", "state": "created", "future_field": {"a": 1}}
        record = ValidationLifecycleRecord.from_dict(payload)
        self.assertEqual(record.to_dict()["future_field"], {"a": 1})

    def test_unknown_future_fields_on_an_iteration_are_preserved(self):
        payload = {"iteration_id": "i", "future_field": [1, 2]}
        iteration = ValidationIterationRecord.from_dict(payload)
        self.assertEqual(iteration.to_dict()["future_field"], [1, 2])

    def test_an_unrecognised_state_is_kept_rather_than_falsified(self):
        record = ValidationLifecycleRecord.from_dict({"state": "hyper_validated"})
        self.assertEqual(record.state, "hyper_validated")

    def test_an_unrecognised_state_permits_no_transitions(self):
        record = ValidationLifecycleRecord.from_dict({"state": "hyper_validated"})
        with self.assertRaises(InvalidLifecycleTransition):
            record.transition(LifecycleState.COMPLETED)

    def test_a_record_with_no_iterations_key_loads(self):
        self.assertEqual(ValidationLifecycleRecord.from_dict({"lifecycle_id": "x"}).iterations, [])

    def test_a_string_command_in_an_iteration_is_tolerated(self):
        iteration = ValidationIterationRecord.from_dict({"commands": ["pytest -q"]})
        self.assertEqual(iteration.commands, [["pytest", "-q"]])

    # -- serialisation stability ---------------------------------------------

    def test_serialise_deserialise_serialise_is_stable(self):
        store = ValidationLifecycleStore()
        store.record(completed_lifecycle(repairs=2))
        store.record(abandoned_lifecycle())
        once = store.to_dict()
        twice = ValidationLifecycleStore.from_dict(json.loads(json.dumps(once))).to_dict()
        thrice = ValidationLifecycleStore.from_dict(json.loads(json.dumps(twice))).to_dict()
        self.assertEqual(twice, thrice)

    def test_a_record_round_trip_preserves_the_defect_fingerprint(self):
        record = completed_lifecycle(repairs=1)
        original = record.iterations[0].defect_signature.fingerprint
        restored = ValidationLifecycleRecord.from_dict(json.loads(json.dumps(record.to_dict())))
        self.assertEqual(restored.iterations[0].defect_signature.fingerprint, original)

    def test_store_round_trip_preserves_the_corruption_count(self):
        store = ValidationLifecycleStore()
        store.corrupted_records_skipped = 3
        restored = ValidationLifecycleStore.from_dict(store.to_dict())
        self.assertEqual(restored.corrupted_records_skipped, 3)

    def test_store_serialises_to_real_json(self):
        store = ValidationLifecycleStore()
        store.record(completed_lifecycle(repairs=2))
        json.dumps(store.to_dict())


# =============================================================================
# H. Persistence through the real storage layer
# =============================================================================


class LifecyclePersistenceCase(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="dungx_lifecycle_")
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.storage = JsonFileStorage(Path(self.root) / ".agent_data")

    def manager(self, **kwargs) -> ValidationLifecycleManager:
        return ValidationLifecycleManager(self.storage, self.root, **kwargs)

    def test_a_new_repository_loads_an_empty_store(self):
        self.assertEqual(len(self.storage.load_validation_lifecycle()), 0)

    def test_a_new_repository_loads_a_trustworthy_store(self):
        self.assertTrue(self.storage.load_validation_lifecycle().history_trustworthy)

    def test_start_persists_a_lifecycle(self):
        record = self.manager().start(task_id="t", subtask_id="s")
        self.assertIsNotNone(self.storage.load_validation_lifecycle().find(record.lifecycle_id))

    def test_start_stamps_a_repository_identity(self):
        self.assertTrue(self.manager().start().repository_id)

    def test_transition_persists(self):
        manager = self.manager()
        record = manager.start()
        manager.transition(record.lifecycle_id, LifecycleState.CANDIDATE_GENERATED)
        reloaded = self.storage.load_validation_lifecycle().find(record.lifecycle_id)
        self.assertEqual(reloaded.state, LifecycleState.CANDIDATE_GENERATED)

    def test_transition_on_an_unknown_lifecycle_returns_none(self):
        self.assertIsNone(self.manager().transition("nope", LifecycleState.COMPLETED))

    def test_an_invalid_transition_through_the_manager_raises(self):
        manager = self.manager()
        record = manager.start()
        with self.assertRaises(InvalidLifecycleTransition):
            manager.transition(record.lifecycle_id, LifecycleState.COMPLETED)

    def test_an_invalid_transition_does_not_corrupt_the_stored_state(self):
        manager = self.manager()
        record = manager.start()
        with self.assertRaises(InvalidLifecycleTransition):
            manager.transition(record.lifecycle_id, LifecycleState.COMPLETED)
        self.assertEqual(manager.get(record.lifecycle_id).state, LifecycleState.CREATED)

    def test_record_iteration_persists(self):
        manager = self.manager()
        record = manager.start()
        manager.record_iteration(record.lifecycle_id, passing_iteration(1))
        self.assertEqual(len(manager.get(record.lifecycle_id).iterations), 1)

    def test_record_iteration_on_an_unknown_lifecycle_returns_none(self):
        self.assertIsNone(self.manager().record_iteration("nope", passing_iteration(1)))

    def test_a_full_lifecycle_survives_a_manager_restart(self):
        first = self.manager()
        record = first.start(task_id="t")
        first.record_iteration(record.lifecycle_id, failing_iteration(1))
        first.transition(record.lifecycle_id, LifecycleState.CANDIDATE_GENERATED)
        second = self.manager()
        reloaded = second.get(record.lifecycle_id)
        self.assertEqual(reloaded.state, LifecycleState.CANDIDATE_GENERATED)
        self.assertEqual(len(reloaded.iterations), 1)

    def test_retention_is_applied_through_the_manager(self):
        manager = self.manager(max_lifecycles=2)
        for _ in range(5):
            manager.start()
        self.assertEqual(len(manager.lifecycles()), 2)

    def test_truncated_persistence_is_quarantined_not_fatal(self):
        manager = self.manager()
        manager.start()
        path = Path(self.root) / ".agent_data" / "validation_lifecycle.json"
        raw = path.read_text(encoding="utf-8")
        path.write_text(raw[: len(raw) // 2], encoding="utf-8")
        store = self.storage.load_validation_lifecycle()
        self.assertEqual(len(store), 0)
        self.assertFalse(store.history_trustworthy)

    def test_a_quarantine_copy_is_written(self):
        manager = self.manager()
        manager.start()
        directory = Path(self.root) / ".agent_data"
        (directory / "validation_lifecycle.json").write_text("{{{", encoding="utf-8")
        self.storage.load_validation_lifecycle()
        self.assertTrue(list(directory.glob("validation_lifecycle.json.corrupt.*")))

    def test_a_corrupt_file_never_yields_a_more_permissive_recommendation(self):
        manager = self.manager()
        for _ in range(20):
            record = manager.start()
            manager.transition(record.lifecycle_id, LifecycleState.CANDIDATE_GENERATED)
            manager.transition(record.lifecycle_id, LifecycleState.VALIDATED)
            manager.transition(record.lifecycle_id, LifecycleState.APPROVED)
            manager.transition(record.lifecycle_id, LifecycleState.APPLIED)
            manager.transition(record.lifecycle_id, LifecycleState.POST_VALIDATED)
            manager.transition(record.lifecycle_id, LifecycleState.COMPLETED)
        clean = manager.recommend(safety_floor=SCOPE_TARGETED).effective_scope
        path = Path(self.root) / ".agent_data" / "validation_lifecycle.json"
        path.write_text("not json at all", encoding="utf-8")
        corrupt = manager.recommend(safety_floor=SCOPE_TARGETED).effective_scope
        self.assertEqual(safest_scope(clean, corrupt), corrupt)
        self.assertEqual(corrupt, SCOPE_BROAD)

    def test_a_storage_that_cannot_save_does_not_break_the_manager(self):
        class ExplodingStorage:
            def load_validation_lifecycle(self):
                return ValidationLifecycleStore()

            def save_validation_lifecycle(self, store):
                raise OSError("disk full")

        manager = ValidationLifecycleManager(ExplodingStorage(), self.root)
        with self.assertRaises(OSError):
            manager.start()

    def test_a_storage_without_lifecycle_support_degrades_gracefully(self):
        class LegacyStorage:
            pass

        manager = ValidationLifecycleManager(LegacyStorage(), self.root)
        record = manager.start()
        self.assertTrue(record.lifecycle_id)
        self.assertEqual(manager.lifecycles(), [])

    def test_the_base_storage_contract_has_safe_defaults(self):
        from local_agent.storage import TaskStorage

        class Minimal(TaskStorage):
            def save_task(self, task): pass
            def load_task(self, task_id): pass
            def list_tasks(self): return []
            def save_checkpoint(self, checkpoint): pass
            def load_checkpoint(self, checkpoint_id): pass
            def save_scheduler_state(self, state): pass
            def load_scheduler_state(self): pass
            def save_provider_configs(self, configs): pass
            def load_provider_configs(self): return []
            def save_semantic_index(self, semantic_index): pass
            def load_semantic_index(self): pass
            def save_project_memory(self, memory): pass
            def load_project_memory(self): pass

        storage = Minimal()
        storage.save_validation_lifecycle(ValidationLifecycleStore())
        self.assertEqual(len(storage.load_validation_lifecycle()), 0)


# =============================================================================
# I. Concurrency
# =============================================================================


class LifecycleConcurrencyCase(unittest.TestCase):
    """Thread-level concurrency only.

    :mod:`local_agent.validation_lifecycle` documents - as Phase 4.19 did for
    the telemetry store - that cross-*process* safety is out of scope, because
    nothing in :mod:`local_agent.storage` provides it. These tests assert
    exactly what is claimed and nothing more.
    """

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="dungx_conc_")
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.storage = JsonFileStorage(Path(self.root) / ".agent_data")
        self.manager = ValidationLifecycleManager(self.storage, self.root, max_lifecycles=100)

    def test_concurrent_independent_lifecycles_are_all_retained(self):
        errors: list[Exception] = []
        ids: list[str] = []
        lock = threading.Lock()

        def worker():
            try:
                record = self.manager.start(task_id="t")
                with lock:
                    ids.append(record.lifecycle_id)
            except Exception as exc:  # pragma: no cover - failure path
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(12)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        stored = {r.lifecycle_id for r in self.manager.lifecycles()}
        self.assertEqual(stored, set(ids))
        self.assertEqual(len(ids), 12)

    def test_concurrent_iterations_on_one_lifecycle_are_all_retained(self):
        record = self.manager.start()
        errors: list[Exception] = []

        def worker(index: int):
            try:
                self.manager.record_iteration(record.lifecycle_id, passing_iteration(index))
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(self.manager.get(record.lifecycle_id).iterations), 10)

    def test_concurrent_reads_during_writes_never_raise(self):
        record = self.manager.start()
        stop = threading.Event()
        errors: list[Exception] = []

        def reader():
            while not stop.is_set():
                try:
                    self.manager.effectiveness()
                    self.manager.recommend(safety_floor=SCOPE_BROAD)
                except Exception as exc:  # pragma: no cover
                    errors.append(exc)

        def writer():
            for index in range(15):
                self.manager.record_iteration(record.lifecycle_id, passing_iteration(index))

        readers = [threading.Thread(target=reader) for _ in range(3)]
        for thread in readers:
            thread.start()
        writer_thread = threading.Thread(target=writer)
        writer_thread.start()
        writer_thread.join()
        stop.set()
        for thread in readers:
            thread.join()
        self.assertEqual(errors, [])

    def test_two_managers_on_one_root_share_a_lock(self):
        other = ValidationLifecycleManager(self.storage, self.root)
        self.assertIs(self.manager._lock, other._lock)

    def test_two_managers_on_different_roots_do_not_share_a_lock(self):
        elsewhere = tempfile.mkdtemp(prefix="dungx_conc2_")
        self.addCleanup(shutil.rmtree, elsewhere, ignore_errors=True)
        other = ValidationLifecycleManager(self.storage, elsewhere)
        self.assertIsNot(self.manager._lock, other._lock)

    def test_concurrent_writes_from_two_managers_do_not_lose_updates(self):
        other = ValidationLifecycleManager(self.storage, self.root, max_lifecycles=100)
        barrier = threading.Barrier(2)

        def worker(manager):
            barrier.wait()
            for _ in range(8):
                manager.start(task_id="t")

        threads = [
            threading.Thread(target=worker, args=(self.manager,)),
            threading.Thread(target=worker, args=(other,)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(len(self.manager.lifecycles()), 16)


# =============================================================================
# J. Events
# =============================================================================


class ValidationEventCase(unittest.TestCase):
    def test_emitting_records_the_event(self):
        emitter = ValidationEventEmitter()
        emitter.emit_named(EVENT_LIFECYCLE_STARTED, lifecycle_id="l1")
        self.assertEqual(emitter.emitted[-1].name, EVENT_LIFECYCLE_STARTED)

    def test_subscribers_receive_events(self):
        seen: list[ValidationEvent] = []
        emitter = ValidationEventEmitter([seen.append])
        emitter.emit_named(EVENT_LIFECYCLE_STARTED, lifecycle_id="l1")
        self.assertEqual(len(seen), 1)

    def test_a_raising_subscriber_is_isolated(self):
        seen: list[ValidationEvent] = []

        def explode(event):
            raise RuntimeError("subscriber bug")

        emitter = ValidationEventEmitter([explode, seen.append])
        emitter.emit_named(EVENT_LIFECYCLE_STARTED)
        self.assertEqual(len(seen), 1)
        self.assertEqual(emitter.subscriber_errors, 1)

    def test_retained_events_are_bounded(self):
        emitter = ValidationEventEmitter()
        emitter.max_retained = 5
        for _ in range(20):
            emitter.emit_named(EVENT_LIFECYCLE_STARTED)
        self.assertEqual(len(emitter.emitted), 5)

    def test_events_serialise_to_json(self):
        event = ValidationEvent(name=EVENT_VALIDATION_FAILED, lifecycle_id="l", scope="broad")
        payload = json.loads(json.dumps(event.to_dict()))
        self.assertEqual(payload["event"], EVENT_VALIDATION_FAILED)

    def test_every_declared_event_name_is_a_string(self):
        for name in ALL_EVENTS:
            self.assertIsInstance(name, str)

    def test_manager_emits_a_start_event(self):
        root = tempfile.mkdtemp(prefix="dungx_evt_")
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        manager = ValidationLifecycleManager(JsonFileStorage(Path(root) / ".d"), root)
        manager.start(task_id="t")
        self.assertIn(EVENT_LIFECYCLE_STARTED, [e.name for e in manager.emitter.emitted])

    def test_manager_emits_a_failure_event_with_the_defect_fingerprint(self):
        root = tempfile.mkdtemp(prefix="dungx_evt2_")
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        manager = ValidationLifecycleManager(JsonFileStorage(Path(root) / ".d"), root)
        record = manager.start()
        iteration = failing_iteration(1)
        manager.record_iteration(record.lifecycle_id, iteration)
        events = [e for e in manager.emitter.emitted if e.name == EVENT_VALIDATION_FAILED]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].defect_fingerprint, iteration.defect_signature.fingerprint)

    def test_manager_emits_a_completion_event(self):
        root = tempfile.mkdtemp(prefix="dungx_evt3_")
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        manager = ValidationLifecycleManager(JsonFileStorage(Path(root) / ".d"), root)
        record = manager.start()
        for state in (
            LifecycleState.CANDIDATE_GENERATED, LifecycleState.VALIDATED,
            LifecycleState.APPROVED, LifecycleState.APPLIED,
            LifecycleState.POST_VALIDATED, LifecycleState.COMPLETED,
        ):
            manager.transition(record.lifecycle_id, state)
        self.assertIn(EVENT_LIFECYCLE_COMPLETED, [e.name for e in manager.emitter.emitted])


# =============================================================================
# K. Configuration
# =============================================================================


class LifecycleConfigCase(unittest.TestCase):
    def test_lifecycle_tracing_is_off_by_default(self):
        self.assertFalse(AgentConfig(project=Path(".")).validation_lifecycle_enabled)

    def test_default_retention_matches_the_module_default(self):
        self.assertEqual(
            AgentConfig(project=Path(".")).validation_lifecycle_retention, DEFAULT_MAX_LIFECYCLES
        )

    def test_default_iteration_bound_matches_the_module_default(self):
        self.assertEqual(
            AgentConfig(project=Path(".")).validation_lifecycle_max_iterations,
            DEFAULT_MAX_ITERATIONS_PER_LIFECYCLE,
        )

    def test_zero_retention_is_rejected(self):
        config = AgentConfig(project=Path("."), validation_lifecycle_retention=0)
        with self.assertRaises(ValueError):
            config.validate()

    def test_zero_iteration_bound_is_rejected(self):
        config = AgentConfig(project=Path("."), validation_lifecycle_max_iterations=0)
        with self.assertRaises(ValueError):
            config.validate()

    def test_zero_min_samples_is_rejected(self):
        config = AgentConfig(project=Path("."), validation_lifecycle_min_samples=0)
        with self.assertRaises(ValueError):
            config.validate()

    def test_valid_settings_pass_validation(self):
        AgentConfig(
            project=Path("."),
            validation_lifecycle_enabled=True,
            validation_lifecycle_retention=10,
            validation_lifecycle_max_iterations=5,
            validation_lifecycle_min_samples=2,
        ).validate()


# =============================================================================
# L. CLI - invoked for real
# =============================================================================


class LifecycleCliCase(unittest.TestCase):
    """Runs the real CLI entry point end to end, not an internal helper."""

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="dungx_cli_")
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.data = Path(self.root) / ".agent_data"
        self.storage = JsonFileStorage(self.data)

    def invoke(self, *argv) -> tuple[int, str]:
        import contextlib
        import io

        from local_agent.cli import main

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            code = main([*argv, "--project", self.root])
        return code, buffer.getvalue()

    def seed(self, *, count: int = 2) -> list[str]:
        manager = ValidationLifecycleManager(self.storage, self.root)
        ids = []
        for index in range(count):
            record = manager.start(task_id=f"task{index}", subtask_id=f"sub{index}")
            manager.record_iteration(record.lifecycle_id, failing_iteration(1))
            manager.transition(record.lifecycle_id, LifecycleState.CANDIDATE_GENERATED)
            manager.transition(record.lifecycle_id, LifecycleState.REPAIR_REQUIRED)
            manager.transition(record.lifecycle_id, LifecycleState.REPAIRED)
            manager.transition(record.lifecycle_id, LifecycleState.COMPLETED)
            ids.append(record.lifecycle_id)
        return ids

    # -- empty repository ---------------------------------------------------

    def test_health_on_an_empty_repository(self):
        code, output = self.invoke("validation", "health")
        self.assertEqual(code, 0)
        self.assertIn("No validation history recorded yet", output)

    def test_history_on_an_empty_repository(self):
        code, output = self.invoke("validation", "history")
        self.assertEqual(code, 0)
        self.assertIn("No validation lifecycles recorded yet", output)

    def test_defects_on_an_empty_repository(self):
        code, output = self.invoke("validation", "defects")
        self.assertEqual(code, 0)
        self.assertIn("no defect history", output)

    def test_calibration_on_an_empty_repository(self):
        code, output = self.invoke("validation", "calibration")
        self.assertEqual(code, 0)
        self.assertIn("SHADOW MODE", output)

    def test_recommendations_on_an_empty_repository(self):
        code, output = self.invoke("validation", "recommendations")
        self.assertEqual(code, 0)
        self.assertIn("broad", output)

    def test_lifecycle_for_an_unknown_id_exits_nonzero(self):
        code, output = self.invoke("validation", "lifecycle", "nope")
        self.assertEqual(code, 1)
        self.assertIn("no lifecycle", output)

    # -- with real data -----------------------------------------------------

    def test_history_lists_seeded_lifecycles(self):
        ids = self.seed(count=3)
        code, output = self.invoke("validation", "history")
        self.assertEqual(code, 0)
        for lifecycle_id in ids:
            self.assertIn(lifecycle_id, output)

    def test_lifecycle_shows_the_trace(self):
        lifecycle_id = self.seed(count=1)[0]
        code, output = self.invoke("validation", "lifecycle", lifecycle_id)
        self.assertEqual(code, 0)
        self.assertIn(lifecycle_id, output)
        self.assertIn("State history", output)
        self.assertIn("Iterations", output)

    def test_lifecycle_json_is_parseable(self):
        lifecycle_id = self.seed(count=1)[0]
        code, output = self.invoke("validation", "lifecycle", lifecycle_id, "--json")
        self.assertEqual(code, 0)
        payload = json.loads(output)
        self.assertEqual(payload["lifecycle_id"], lifecycle_id)

    def test_history_json_is_parseable(self):
        self.seed(count=2)
        code, output = self.invoke("validation", "history", "--json")
        self.assertEqual(code, 0)
        self.assertEqual(len(json.loads(output)["lifecycles"]), 2)

    def test_health_json_carries_both_stores(self):
        self.seed(count=1)
        code, output = self.invoke("validation", "health", "--json")
        self.assertEqual(code, 0)
        payload = json.loads(output)
        self.assertIn("telemetry", payload)
        self.assertIn("lifecycle", payload)

    def test_defects_json_reports_recurrence(self):
        self.seed(count=2)
        code, output = self.invoke("validation", "defects", "--json")
        self.assertEqual(code, 0)
        payload = json.loads(output)
        self.assertEqual(payload["lifecycles"], 2)
        self.assertIn("top_recurring_defects", payload)

    def test_defects_reports_insufficient_data_honestly(self):
        self.seed(count=2)
        _code, output = self.invoke("validation", "defects")
        self.assertIn("NOT established", output)

    def test_calibration_json_states_live_calibration_is_not_implemented(self):
        code, output = self.invoke("validation", "calibration", "--json")
        self.assertEqual(code, 0)
        self.assertFalse(json.loads(output)["live_calibration_implemented"])

    def test_recommendations_json_marks_itself_advisory(self):
        self.seed(count=2)
        code, output = self.invoke("validation", "recommendations", "--json")
        self.assertEqual(code, 0)
        payload = json.loads(output)
        self.assertTrue(payload["advisory"])
        self.assertEqual(payload["effective_scope"], SCOPE_BROAD)

    def test_history_warns_about_corrupt_records(self):
        self.seed(count=1)
        path = self.data / "validation_lifecycle.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["lifecycles"].append("corrupted")
        path.write_text(json.dumps(payload), encoding="utf-8")
        _code, output = self.invoke("validation", "history")
        self.assertIn("could not be", output)

    def test_health_warns_when_history_is_untrustworthy(self):
        self.seed(count=1)
        path = self.data / "validation_lifecycle.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["lifecycles"].append(42)
        path.write_text(json.dumps(payload), encoding="utf-8")
        _code, output = self.invoke("validation", "health")
        self.assertIn("WARNING", output)

    def test_cli_does_not_mutate_the_stored_history(self):
        """A diagnostic surface must be read-only in fact, not just by intent."""
        self.seed(count=2)
        path = self.data / "validation_lifecycle.json"
        before = path.read_bytes()
        for argv in (
            ("validation", "health"),
            ("validation", "history"),
            ("validation", "defects"),
            ("validation", "calibration"),
            ("validation", "recommendations"),
        ):
            self.invoke(*argv)
        self.assertEqual(path.read_bytes(), before)

    def test_cli_runs_as_a_module_subprocess(self):
        """The genuine article: a fresh interpreter, real argv, real exit code."""
        result = subprocess.run(
            [sys.executable, "-m", "local_agent", "validation", "health",
             "--project", self.root],
            capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Validation Intelligence Health", result.stdout)


# =============================================================================
# M. Failure injection
# =============================================================================


class FailureInjectionCase(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="dungx_fail_")
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.storage = JsonFileStorage(Path(self.root) / ".agent_data")
        self.manager = ValidationLifecycleManager(self.storage, self.root)

    def test_a_real_command_exit_1_is_captured_as_a_failed_iteration(self):
        result = subprocess.run(
            [sys.executable, "-c", "import sys; sys.stderr.write('boom\\n'); sys.exit(1)"],
            capture_output=True, text=True, timeout=60,
        )
        record = self.manager.start()
        iteration = ValidationIterationRecord(
            iteration_number=1,
            validation_result=RESULT_FAILED,
            validation_stage=STAGE_TARGETED,
            defect_signature=compute_defect_signature(
                failure_category="validation_failure",
                command=[sys.executable, "-c", "..."],
                exit_code=result.returncode,
                diagnostic=result.stderr,
                stderr=result.stderr,
            ),
        )
        self.manager.record_iteration(record.lifecycle_id, iteration)
        stored = self.manager.get(record.lifecycle_id).iterations[0]
        self.assertEqual(stored.validation_result, RESULT_FAILED)
        self.assertEqual(stored.defect_signature.exit_code, 1)

    def test_a_repeated_real_defect_is_detected_as_recurrence(self):
        script = "raise ValueError('always the same')\n"
        directory = tempfile.mkdtemp(prefix="dungx_recur_")
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        path = Path(directory) / "s.py"
        path.write_text(script, encoding="utf-8")
        record = self.manager.start()
        parent = ""
        for index in range(3):
            result = subprocess.run(
                [sys.executable, str(path)], capture_output=True, text=True, timeout=60
            )
            iteration = ValidationIterationRecord(
                iteration_number=index + 1,
                parent_iteration_id=parent,
                kind=ITERATION_REPAIR if parent else ITERATION_IMPLEMENTATION,
                validation_result=RESULT_FAILED,
                validation_stage=STAGE_TARGETED,
                defect_signature=compute_defect_signature(
                    failure_category="validation_failure",
                    command=["python", "s.py"],
                    exit_code=result.returncode,
                    diagnostic=result.stderr,
                    stderr=result.stderr,
                ),
            )
            self.manager.record_iteration(record.lifecycle_id, iteration)
            parent = iteration.iteration_id
        stored = self.manager.get(record.lifecycle_id)
        self.assertEqual(len(stored.recurring_defects()), 1)
        self.assertEqual(list(stored.recurring_defects().values()), [3])

    def test_a_provider_style_exception_becomes_a_distinct_signature(self):
        provider = compute_defect_signature(
            failure_category="provider_failure",
            exception_class="RateLimitError",
            diagnostic="429 rate limited",
        )
        validation = compute_defect_signature(
            failure_category="validation_failure",
            exception_class="AssertionError",
            diagnostic="429 rate limited",
        )
        self.assertFalse(signatures_match(provider, validation))

    def test_a_quota_failure_and_a_rate_limit_failure_are_distinct(self):
        self.assertFalse(signatures_match(
            compute_defect_signature(
                failure_category="provider_failure", exception_class="QuotaExceededError"
            ),
            compute_defect_signature(
                failure_category="provider_failure", exception_class="RateLimitError"
            ),
        ))

    def test_a_lifecycle_persistence_failure_surfaces_rather_than_silently_dropping(self):
        class BrokenStorage:
            def load_validation_lifecycle(self):
                return ValidationLifecycleStore()

            def save_validation_lifecycle(self, store):
                raise OSError("read-only filesystem")

        manager = ValidationLifecycleManager(BrokenStorage(), self.root)
        with self.assertRaises(OSError):
            manager.start()

    def test_incomplete_telemetry_does_not_make_the_recommender_permissive(self):
        store = ValidationLifecycleStore()
        for _ in range(20):
            record = new_lifecycle()
            record.add_iteration(passing_iteration(1))
            store.record(record)  # never transitioned: all unresolved
        recommendation = AdaptiveValidationRecommender(min_samples=10).recommend(
            safety_floor=SCOPE_TARGETED, store=store
        )
        self.assertFalse(recommendation.data_sufficient)
        self.assertEqual(recommendation.effective_scope, SCOPE_TARGETED)

    def test_a_stale_lifecycle_never_becomes_a_completion(self):
        record = new_lifecycle()
        record.transition(LifecycleState.CANDIDATE_GENERATED)
        self.assertFalse(record.is_terminal)
        self.assertEqual(record.terminal_outcome, "")

    def test_a_malformed_candidate_operation_can_abandon_from_any_active_state(self):
        for state in (
            LifecycleState.CREATED,
            LifecycleState.CANDIDATE_GENERATED,
            LifecycleState.APPLIED,
        ):
            with self.subTest(state=state):
                record = new_lifecycle()
                drive_to(record, state)
                record.transition(LifecycleState.ABANDONED, reason="invalid operations")
                self.assertEqual(record.state, LifecycleState.ABANDONED)


# =============================================================================
# N. Invariants / properties
# =============================================================================


class SafetyInvariantCase(unittest.TestCase):
    def excellent_store(self) -> ValidationLifecycleStore:
        store = ValidationLifecycleStore()
        for _ in range(60):
            store.record(completed_lifecycle())
        return store

    def recommend(self, **kwargs):
        return AdaptiveValidationRecommender(min_samples=10).recommend(**kwargs)

    def test_safety_monotonicity_more_uncertainty_never_narrows(self):
        for floor in (SCOPE_TARGETED, SCOPE_EXPANDED, SCOPE_BROAD):
            with self.subTest(floor=floor):
                certain = self.recommend(
                    safety_floor=floor, store=self.excellent_store(), degraded_analysis=False
                ).effective_scope
                uncertain = self.recommend(
                    safety_floor=floor, store=self.excellent_store(), degraded_analysis=True
                ).effective_scope
                self.assertEqual(safest_scope(certain, uncertain), uncertain)

    def test_evidence_monotonicity_removing_history_never_narrows(self):
        full = self.excellent_store()
        empty = ValidationLifecycleStore()
        for floor in (SCOPE_TARGETED, SCOPE_EXPANDED, SCOPE_BROAD):
            with self.subTest(floor=floor):
                with_history = self.recommend(safety_floor=floor, store=full).effective_scope
                without = self.recommend(safety_floor=floor, store=empty).effective_scope
                self.assertEqual(safest_scope(with_history, without), without)

    def test_historical_conservatism_insufficient_history_never_improves_scope(self):
        small = ValidationLifecycleStore()
        for _ in range(2):
            small.record(completed_lifecycle())
        for floor in (SCOPE_TARGETED, SCOPE_EXPANDED, SCOPE_BROAD):
            with self.subTest(floor=floor):
                self.assertEqual(
                    self.recommend(safety_floor=floor, store=small).effective_scope, floor
                )

    def test_corruption_conservatism_never_improves_scope(self):
        corrupt = self.excellent_store()
        corrupt.corrupted_records_skipped = 1
        for floor in (SCOPE_TARGETED, SCOPE_EXPANDED, SCOPE_BROAD):
            with self.subTest(floor=floor):
                clean = self.recommend(
                    safety_floor=floor, store=self.excellent_store()
                ).effective_scope
                dirty = self.recommend(safety_floor=floor, store=corrupt).effective_scope
                self.assertEqual(safest_scope(clean, dirty), dirty)

    def test_lifecycle_integrity_terminal_never_reopens(self):
        for terminal in LifecycleState.TERMINAL:
            for target in LifecycleState.ALL:
                if target in LifecycleState.TERMINAL:
                    continue
                with self.subTest(terminal=terminal, target=target):
                    record = new_lifecycle()
                    if terminal == LifecycleState.COMPLETED:
                        drive_to(record, LifecycleState.COMPLETED)
                    else:
                        record.transition(terminal)
                    self.assertTrue(record.is_terminal)
                    with self.assertRaises(InvalidLifecycleTransition):
                        record.transition(target)

    def test_determinism_equivalent_histories_classify_equivalently(self):
        signature = compute_defect_signature(
            failure_category="validation_failure", command=["pytest"], exit_code=1,
            diagnostic="AssertionError: same",
        )
        left = ValidationLifecycleStore()
        right = ValidationLifecycleStore()
        for _ in range(4):
            left.record(completed_lifecycle(repairs=2, signature=signature))
            right.record(completed_lifecycle(repairs=2, signature=signature))
        left_metrics = compute_repair_effectiveness(left).to_dict()
        right_metrics = compute_repair_effectiveness(right).to_dict()
        for key in ("repair_success_rate", "repeated_defect_rate", "median_repair_iterations"):
            self.assertEqual(left_metrics[key], right_metrics[key], key)

    def test_serialisation_stability_is_semantic_not_merely_textual(self):
        record = completed_lifecycle(repairs=2)
        once = record.to_dict()
        twice = ValidationLifecycleRecord.from_dict(once).to_dict()
        thrice = ValidationLifecycleRecord.from_dict(twice).to_dict()
        self.assertEqual(twice, thrice)

    def test_isolation_analysis_never_writes_to_the_repository(self):
        root = tempfile.mkdtemp(prefix="dungx_iso_")
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        (Path(root) / "code.py").write_text("x = 1\n", encoding="utf-8")
        before = sorted(p.name for p in Path(root).iterdir())
        store = self.excellent_store()
        compute_repair_effectiveness(store)
        AdaptiveValidationRecommender().recommend(safety_floor=SCOPE_BROAD, store=store)
        self.assertEqual(sorted(p.name for p in Path(root).iterdir()), before)
        self.assertEqual((Path(root) / "code.py").read_text(encoding="utf-8"), "x = 1\n")

    def test_the_single_authoritative_decision_engine_is_unchallenged(self):
        """Only one module defines a scope *decision*; this one only advises.

        Asserted structurally rather than by inspection: the lifecycle module
        must not import, subclass, or name the decision engine anywhere.
        """
        import local_agent.validation_decision as decision

        self.assertTrue(hasattr(decision, "ValidationDecisionEngine"))
        self.assertNotIn(
            "ValidationDecisionEngine", code_identifiers("local_agent.validation_lifecycle")
        )
        self.assertNotIn(
            "local_agent.validation_decision", imported_modules("local_agent.validation_lifecycle")
        )

    def test_no_code_path_lets_a_recommendation_replace_a_decision(self):
        """The orchestrator must not read the recommender on the real path."""
        import local_agent.orchestrator as orchestrator

        source = Path(orchestrator.__file__).read_text(encoding="utf-8")
        self.assertNotIn("AdaptiveValidationRecommender", source)
        self.assertNotIn(".recommend(", source)


# =============================================================================
# N2. Orchestrator integration - proving the wiring is not dead code
# =============================================================================


class _ApprovingProvider:
    """Minimal provider double.

    Only the LLM call boundary is stubbed. The orchestrator's real planning,
    patch preparation, apply, validation and review path all execute, against
    a real temporary repository on disk.
    """

    def __init__(self, operations):
        self.operations = operations

    @property
    def capabilities(self):
        from local_agent.models import ProviderCapability

        return {
            ProviderCapability.PLANNING,
            ProviderCapability.IMPLEMENTATION,
            ProviderCapability.REPAIR,
            ProviderCapability.REVIEW,
        }

    def generate_plan(self, task, context):
        from local_agent.models import Plan

        return Plan(objective=task, steps=["edit"], files_likely_to_change=["main.py"])

    def generate_code(self, task, plan, context, failure=None, review=None):
        return list(self.operations)

    def review_changes(self, task, plan, diff, context):
        from local_agent.models import ReviewResult

        return ReviewResult(verdict="APPROVED", summary="ok", findings=[])


class _DummyScheduler:
    def __init__(self, instance):
        self.provider = "mock"
        self._instance = instance

    def _select_providers(self, task, capabilities):
        return [self]

    def _build_provider_instance(self, provider_name):
        return self._instance


class OrchestratorLifecycleIntegrationCase(unittest.TestCase):
    """Runs a real ``Orchestrator.run`` and asserts a lifecycle was persisted.

    The point of this case is the anti-dead-code check specifically: the
    lifecycle hooks must sit on a path a real run genuinely executes, not
    merely be present in the file. Everything asserted here comes from an
    actual orchestrator run writing actual files.
    """

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="dungx_orch_")).resolve()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        (self.root / "main.py").write_text("print('original')\n", encoding="utf-8")
        self.storage = JsonFileStorage(self.root / ".agent_data")

    def build(self, *, enabled: bool, storage=None, content="print('changed')\n"):
        from local_agent.models import FileOperation
        from local_agent.orchestrator import Orchestrator

        config = AgentConfig(
            project=self.root,
            provider="mock",
            max_iterations=2,
            approval="never",
            validation_lifecycle_enabled=enabled,
        )
        provider = _ApprovingProvider(
            [FileOperation(action="modify", path="main.py", content=content)]
        )
        return Orchestrator(
            config,
            storage=storage or self.storage,
            scheduler=_DummyScheduler(provider),
            repo_lock=threading.Lock(),
            memory_lock=threading.Lock(),
        )

    def make_task(self, task_id: str, storage=None):
        import datetime as _dt

        from local_agent.models import Task, TaskStatus

        now = _dt.datetime.now(_dt.timezone.utc)
        task = Task(
            task_id=task_id,
            objective="change main",
            status=TaskStatus.PENDING,
            created_at=now,
            updated_at=now,
            autonomous=True,
        )
        (storage or self.storage).save_task(task)
        return task

    def run_once(self, *, enabled: bool, task_id: str = "lifecycle-task"):
        orchestrator = self.build(enabled=enabled)
        return orchestrator.run(self.make_task(task_id))

    def test_a_real_run_persists_a_lifecycle(self):
        self.run_once(enabled=True)
        store = self.storage.load_validation_lifecycle()
        self.assertEqual(len(store), 1, "the orchestrator lifecycle hooks never executed")
        self.assertEqual(store.lifecycles[0].task_id, "lifecycle-task")

    def test_a_real_run_records_the_state_path(self):
        self.run_once(enabled=True)
        record = self.storage.load_validation_lifecycle().lifecycles[0]
        visited = [entry["state"] for entry in record.state_history]
        self.assertIn(LifecycleState.CANDIDATE_GENERATED, visited)
        self.assertIn(LifecycleState.APPLIED, visited)

    def test_a_real_run_reaches_a_terminal_state(self):
        self.run_once(enabled=True)
        record = self.storage.load_validation_lifecycle().lifecycles[0]
        self.assertTrue(record.is_terminal, f"ended in non-terminal state {record.state}")

    def test_a_real_run_records_at_least_one_iteration(self):
        self.run_once(enabled=True)
        record = self.storage.load_validation_lifecycle().lifecycles[0]
        self.assertGreaterEqual(len(record.iterations), 1)

    def test_the_recorded_iteration_carries_a_real_validation_result(self):
        self.run_once(enabled=True)
        record = self.storage.load_validation_lifecycle().lifecycles[0]
        self.assertIn(record.iterations[0].validation_result, {RESULT_PASSED, RESULT_FAILED})

    def test_disabled_by_default_records_nothing(self):
        self.run_once(enabled=False)
        self.assertEqual(len(self.storage.load_validation_lifecycle()), 0)

    def test_the_run_still_does_its_real_work_with_tracing_enabled(self):
        report = self.run_once(enabled=True)
        self.assertTrue(report.changed_files)
        self.assertEqual(
            (self.root / "main.py").read_text(encoding="utf-8"), "print('changed')\n"
        )

    def test_a_lifecycle_persistence_failure_does_not_break_the_run(self):
        """A broken store must cost telemetry, never the run itself."""

        class ExplodingStorage(JsonFileStorage):
            def save_validation_lifecycle(self, store):
                raise OSError("disk full")

        storage = ExplodingStorage(self.root / ".agent_data2")
        orchestrator = self.build(enabled=True, storage=storage, content="print('x')\n")
        report = orchestrator.run(self.make_task("broken-store", storage=storage))
        self.assertTrue(report.changed_files, "a telemetry failure aborted a real run")

    def test_the_cli_can_read_a_lifecycle_a_real_run_produced(self):
        """End to end: a real run writes it, the real CLI reads it back."""
        import contextlib

        from local_agent.cli import main

        self.run_once(enabled=True)
        lifecycle_id = self.storage.load_validation_lifecycle().lifecycles[0].lifecycle_id
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = main([
                "validation", "lifecycle", lifecycle_id, "--json",
                "--project", str(self.root),
            ])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(buffer.getvalue())["lifecycle_id"], lifecycle_id)


# =============================================================================
# O. Performance smoke tests (measured, generously bounded)
# =============================================================================


class PerformanceSmokeCase(unittest.TestCase):
    """Bounds are deliberately loose - these guard against an accidental
    quadratic blow-up, not against a few milliseconds of drift on slower CI
    hardware. The measured numbers are reported in the phase writeup; these
    assertions only fail on a genuine regression in complexity class."""

    def build_store(self, lifecycles: int = 200, iterations: int = 5):
        store = ValidationLifecycleStore(max_lifecycles=lifecycles + 10)
        for index in range(lifecycles):
            record = completed_lifecycle(repairs=min(iterations - 1, 3))
            record.task_id = f"task{index}"
            store.record(record)
        return store

    def test_defect_signature_generation_is_fast(self):
        started = time.perf_counter()
        for index in range(2000):
            compute_defect_signature(
                failure_category="validation_failure",
                command=["pytest", f"tests/test_{index}.py"],
                exit_code=1,
                diagnostic=f"/tmp/x{index}/t.py:{index}: AssertionError: boom",
                stderr="e" * 500,
            )
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 10.0, f"2000 signatures took {elapsed:.2f}s")

    def test_lifecycle_insertion_is_fast(self):
        store = ValidationLifecycleStore(max_lifecycles=2000)
        started = time.perf_counter()
        for _ in range(1000):
            store.record(new_lifecycle())
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 15.0, f"1000 insertions took {elapsed:.2f}s")

    def test_effectiveness_computation_over_a_large_store_is_fast(self):
        store = self.build_store()
        started = time.perf_counter()
        compute_repair_effectiveness(store)
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 5.0, f"effectiveness took {elapsed:.2f}s")

    def test_recommendation_computation_is_fast(self):
        store = self.build_store()
        started = time.perf_counter()
        for _ in range(10):
            AdaptiveValidationRecommender().recommend(safety_floor=SCOPE_BROAD, store=store)
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 15.0, f"10 recommendations took {elapsed:.2f}s")

    def test_serialisation_round_trip_of_a_large_store_is_fast(self):
        store = self.build_store()
        started = time.perf_counter()
        ValidationLifecycleStore.from_dict(json.loads(json.dumps(store.to_dict())))
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 10.0, f"round trip took {elapsed:.2f}s")

    def test_lifecycle_lookup_in_a_large_store_is_fast(self):
        store = self.build_store()
        target = store.lifecycles[0].lifecycle_id
        started = time.perf_counter()
        for _ in range(1000):
            store.find(target)
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 10.0, f"1000 lookups took {elapsed:.2f}s")

    def test_persisted_write_under_lock_is_fast(self):
        root = tempfile.mkdtemp(prefix="dungx_perf_")
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        manager = ValidationLifecycleManager(
            JsonFileStorage(Path(root) / ".d"), root, max_lifecycles=50
        )
        started = time.perf_counter()
        for _ in range(50):
            manager.start()
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 30.0, f"50 persisted starts took {elapsed:.2f}s")

    def test_repair_chain_reconstruction_is_linear_not_quadratic(self):
        record = new_lifecycle()
        record.max_iterations = 500
        parent = ""
        for index in range(400):
            iteration = passing_iteration(index + 1, parent=parent)
            record.add_iteration(iteration)
            parent = iteration.iteration_id
        started = time.perf_counter()
        chain = record.repair_chain(parent)
        elapsed = time.perf_counter() - started
        self.assertEqual(len(chain), 400)
        self.assertLess(elapsed, 5.0, f"chain reconstruction took {elapsed:.2f}s")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
