"""Phase 4.24: Completion Proof Attack Surface Discovery & Adversarial Hardening.

Phase 4.23 hardened *which requirement/obligation* a piece of evidence may
bind to (task/path/component scoping, cross-obligation ambiguity exclusion).
It did not examine whether a single piece of evidence, once bound, actually
still describes the *current* workspace state at the moment completion is
decided. This phase attacks that dimension directly.

One real, reproducible vulnerability was confirmed against the Phase 4.23
baseline (commit 653ad03) before any fix was written:

    STALE GLOBAL-VALIDATION EVIDENCE SURVIVES AN UNRELATED FILE MUTATION.

``CompletionEvidenceStore.invalidate_on_file_mutation`` is the mechanism both
``Orchestrator.run`` and ``MultiTurnImplementationAgent.execute`` call, every
iteration, immediately after applying file changes, to keep the evidence
store honest. For TEST_EXECUTION / CODE_REVIEW / DIFF_INSPECTION evidence --
entries that certify the *entire* tracked change-set as it stood when they
were recorded, not merely the specific files listed in their own
``target_paths`` -- the pre-fix logic only re-checked the fingerprint of an
entry's *own* ``target_paths``. A mutation to a file the entry had never
seen (e.g. a brand-new file introduced by a later repair turn) left the
entry ``VALID`` untouched.

Reproduced end-to-end via ``CompletionDecisionEngine.evaluate()``:
  1. Turn 1: only ``a.py`` exists. A full test run passes; TEST_EXECUTION
     evidence recorded with ``target_paths=["a.py"]``.
  2. Turn 2: a NEW file ``b.py`` is written containing a real runtime bug.
     No fresh test run ever observes it.
  3. A genuine, *live* reviewer approves the resulting two-file diff (an
     LLM review is imperfect and does not catch the bug) -- this is not a
     forged or stale review, it is a real APPROVED verdict on the CURRENT
     diff, so Gate 5 (GATE_REVIEW_APPROVED) legitimately passes.
  4. Gate 4 (GATE_VALIDATION_PASSED) read the turn-1 evidence as still
     "1 valid test execution(s) passed cleanly on current workspace" even
     though that evidence never ran against b.py's actual content.
  5. Result: ``is_ready=True`` / ``READY`` -- a false positive completion
     signal for a workspace containing an untested, broken file.

Fix: ``invalidate_on_file_mutation`` now invalidates every VALID
TEST_EXECUTION/CODE_REVIEW/DIFF_INSPECTION entry unconditionally whenever
ANY file mutation occurs, regardless of whether the mutated path intersects
the entry's own recorded ``target_paths`` -- these evidence *types*
inherently certify a whole-tree/whole-diff claim, so any tree mutation
invalidates that claim. Per-file evidence (SYNTAX_VERIFICATION,
SAFETY_INVARIANT, ...) keeps its original, narrower, path-intersecting
behaviour: it only claims to cover the specific file(s) it names, so an
unrelated mutation correctly leaves it alone (verified below in the
false-positive campaign, so the fix does not overreach).
"""

from __future__ import annotations

import datetime
import hashlib
import inspect
import time

import pytest

from local_agent.completion import (
    CompletionDecisionEngine,
    CompletionEvidenceStore,
    EvidenceStatus,
    EvidenceTrustTier,
    EvidenceType,
)
from local_agent.filesystem import ProjectFilesystem
from local_agent.models import FileOperation, Plan, ReviewResult, Task, TaskStatus
from local_agent.multi_turn import MultiTurnImplementationAgent
from local_agent.orchestrator import Orchestrator
from local_agent.task_contract import (
    AcceptanceObligation,
    Requirement,
    RequirementAssessmentEngine,
    RequirementState,
    TaskContract,
)


def make_task(task_id: str = "t1", objective: str = "Implement feature") -> Task:
    now = datetime.datetime.now(datetime.timezone.utc)
    return Task(task_id=task_id, objective=objective, status=TaskStatus.PENDING, created_at=now, updated_at=now)


def diff_hash_of(diff: str) -> str:
    return hashlib.sha256(diff.encode("utf-8")).hexdigest()[:16]


# =============================================================================
# Section 1: the reproduced vulnerability (before/after)
# =============================================================================

class TestReproducedVulnerability:
    def test_store_level_stale_global_evidence_survives_unrelated_mutation_pre_fix_shape(self, tmp_path):
        """Store-level regression lock: TEST_EXECUTION evidence recorded over
        a.py only must NOT remain valid after b.py (never covered) mutates."""
        (tmp_path / "a.py").write_text("print('a v1')\n", encoding="utf-8")
        (tmp_path / "b.py").write_text("print('b v1')\n", encoding="utf-8")
        store = CompletionEvidenceStore(tmp_path)
        ev = store.record(
            task_id="t1", subtask_id="main", turn_number=1, stage="verifying",
            evidence_type=EvidenceType.TEST_EXECUTION, source="pytest",
            target_paths=["a.py"], command=["pytest"], exit_code=0,
        )
        (tmp_path / "b.py").write_text("print('b v2 broken')\n", encoding="utf-8")
        invalidated = store.invalidate_on_file_mutation(["b.py"], reason="repaired_after_execution")
        assert ev.evidence_id in invalidated
        assert ev.status == EvidenceStatus.INVALIDATED.value
        assert store.get_valid_evidence(EvidenceType.TEST_EXECUTION, task_id="t1") == []

    def test_end_to_end_false_readiness_closed(self, tmp_path):
        """Full CompletionDecisionEngine.evaluate() reproduction: a genuine
        live review of the CURRENT (two-file) diff plus stale single-file
        test evidence must NOT reach is_ready=True."""
        (tmp_path / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
        (tmp_path / "b.py").write_text("def b():\n    return 2\n", encoding="utf-8")
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        engine = CompletionDecisionEngine(fs)
        task = make_task("t1", "Add a() and b()")
        plan = Plan(objective="Add a() and b()")

        store.record(
            task_id="t1", subtask_id="main", turn_number=1, stage="verifying",
            evidence_type=EvidenceType.TEST_EXECUTION, source="pytest",
            target_paths=["a.py"], command=["pytest"], exit_code=0,
        )

        (tmp_path / "b.py").write_text("def b():\n    raise RuntimeError('actually broken')\n", encoding="utf-8")
        store.invalidate_on_file_mutation(["b.py"], reason="repaired_after_execution")

        applied_ops = [FileOperation(action="modify", path="a.py"), FileOperation(action="modify", path="b.py")]
        current_diff = "--- a.py\n--- b.py\n+def b():\n+    raise RuntimeError('actually broken')\n"
        review = ReviewResult(verdict="APPROVED", summary="looks fine", findings=[])
        store.record(
            task_id="t1", subtask_id="main", turn_number=2, stage="reviewing",
            evidence_type=EvidenceType.CODE_REVIEW, source="reviewer",
            trust_tier=EvidenceTrustTier.DELIBERATIVE_REVIEW,
            target_paths=["a.py", "b.py"],
            payload={"verdict": "APPROVED", "summary": "ok", "diff_hash": diff_hash_of(current_diff)},
        )

        assessment = engine.evaluate(
            task=task, subtask=None, plan=plan, evidence_store=store,
            applied_operations=applied_ops, current_diff=current_diff, last_review=review,
        )
        assert assessment.is_ready is False
        gate = next(g for g in assessment.gates_evaluated if g.gate_name == "GATE_VALIDATION_PASSED")
        assert gate.passed is False

    def test_legitimate_case_fresh_full_coverage_still_reaches_ready(self, tmp_path):
        """Sanity check: the fix must not make ordinary legitimate completion
        impossible -- fresh evidence covering the CURRENT full diff still
        reaches is_ready=True."""
        (tmp_path / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
        (tmp_path / "b.py").write_text("def b():\n    return 2\n", encoding="utf-8")
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        engine = CompletionDecisionEngine(fs)
        task = make_task("t1", "Add a() and b()")
        plan = Plan(objective="Add a() and b()")

        applied_ops = [FileOperation(action="modify", path="a.py"), FileOperation(action="modify", path="b.py")]
        current_diff = "--- a.py\n--- b.py\n+def b():\n+    return 2\n"
        review = ReviewResult(verdict="APPROVED", summary="looks fine", findings=[])
        # Fresh test evidence recorded AFTER both files reached their final
        # state, covering both -- exactly what a real post-repair retest run
        # looks like in production.
        store.record(
            task_id="t1", subtask_id="main", turn_number=2, stage="verifying",
            evidence_type=EvidenceType.TEST_EXECUTION, source="pytest",
            target_paths=["a.py", "b.py"], command=["pytest"], exit_code=0,
        )
        store.record(
            task_id="t1", subtask_id="main", turn_number=2, stage="reviewing",
            evidence_type=EvidenceType.CODE_REVIEW, source="reviewer",
            trust_tier=EvidenceTrustTier.DELIBERATIVE_REVIEW,
            target_paths=["a.py", "b.py"],
            payload={"verdict": "APPROVED", "summary": "ok", "diff_hash": diff_hash_of(current_diff)},
        )
        assessment = engine.evaluate(
            task=task, subtask=None, plan=plan, evidence_store=store,
            applied_operations=applied_ops, current_diff=current_diff, last_review=review,
        )
        assert assessment.is_ready is True
        assert assessment.readiness_level == "READY"


# =============================================================================
# Section 2: invalidate_on_file_mutation -- unit-level provenance model
# =============================================================================

class TestGlobalValidationInvalidation:
    @pytest.mark.parametrize("etype", [
        EvidenceType.TEST_EXECUTION, EvidenceType.CODE_REVIEW, EvidenceType.DIFF_INSPECTION,
    ])
    def test_global_type_invalidated_by_unrelated_file_mutation(self, tmp_path, etype):
        (tmp_path / "a.py").write_text("v1", encoding="utf-8")
        (tmp_path / "b.py").write_text("v1", encoding="utf-8")
        store = CompletionEvidenceStore(tmp_path)
        ev = store.record(
            task_id="t1", subtask_id="s1", turn_number=1, stage="x",
            evidence_type=etype, source="x", target_paths=["a.py"],
        )
        store.invalidate_on_file_mutation(["b.py"])
        assert ev.status == EvidenceStatus.INVALIDATED.value

    def test_global_type_with_no_target_paths_still_invalidated(self, tmp_path):
        store = CompletionEvidenceStore(tmp_path)
        ev = store.record(
            task_id="t1", subtask_id="s1", turn_number=1, stage="x",
            evidence_type=EvidenceType.TEST_EXECUTION, source="x",
        )
        store.invalidate_on_file_mutation(["anything.py"])
        assert ev.status == EvidenceStatus.INVALIDATED.value

    @pytest.mark.parametrize("etype", [
        EvidenceType.SYNTAX_VERIFICATION, EvidenceType.SAFETY_INVARIANT,
        EvidenceType.FILESYSTEM_OBSERVATION, EvidenceType.GIT_INTEGRITY,
    ])
    def test_per_file_type_not_invalidated_by_unrelated_mutation(self, tmp_path, etype):
        """False-positive guard: the fix must not overreach into evidence
        types that only ever claim to cover the specific file(s) they name."""
        (tmp_path / "a.py").write_text("v1", encoding="utf-8")
        (tmp_path / "b.py").write_text("v1", encoding="utf-8")
        store = CompletionEvidenceStore(tmp_path)
        ev = store.record(
            task_id="t1", subtask_id="s1", turn_number=1, stage="x",
            evidence_type=etype, source="x", target_paths=["a.py"],
        )
        store.invalidate_on_file_mutation(["b.py"])
        assert ev.status == EvidenceStatus.VALID.value

    @pytest.mark.parametrize("etype", [
        EvidenceType.SYNTAX_VERIFICATION, EvidenceType.SAFETY_INVARIANT,
    ])
    def test_per_file_type_still_invalidated_when_its_own_path_mutates(self, tmp_path, etype):
        """Regression guard: narrowing the "unrelated" case must not weaken
        the original, correct, path-intersecting invalidation."""
        (tmp_path / "a.py").write_text("v1", encoding="utf-8")
        store = CompletionEvidenceStore(tmp_path)
        ev = store.record(
            task_id="t1", subtask_id="s1", turn_number=1, stage="x",
            evidence_type=etype, source="x", target_paths=["a.py"],
        )
        (tmp_path / "a.py").write_text("v2 -- mutated", encoding="utf-8")
        invalidated = store.invalidate_on_file_mutation(["a.py"])
        assert ev.evidence_id in invalidated
        assert ev.status == EvidenceStatus.INVALIDATED.value

    def test_already_invalidated_entry_is_left_alone(self, tmp_path):
        store = CompletionEvidenceStore(tmp_path)
        ev = store.record(
            task_id="t1", subtask_id="s1", turn_number=1, stage="x",
            evidence_type=EvidenceType.TEST_EXECUTION, source="x", target_paths=["a.py"],
        )
        store.invalidate_on_file_mutation(["a.py"])
        assert ev.status == EvidenceStatus.INVALIDATED.value
        reason_after_first = ev.invalidation_reason
        store.invalidate_on_file_mutation(["a.py"], reason="second_call")
        # Idempotent: an already-INVALIDATED entry is skipped (status filter
        # at the top of the loop), so its reason is not overwritten.
        assert ev.invalidation_reason == reason_after_first

    def test_empty_modified_paths_invalidates_nothing(self, tmp_path):
        store = CompletionEvidenceStore(tmp_path)
        ev = store.record(
            task_id="t1", subtask_id="s1", turn_number=1, stage="x",
            evidence_type=EvidenceType.TEST_EXECUTION, source="x", target_paths=["a.py"],
        )
        invalidated = store.invalidate_on_file_mutation([])
        assert invalidated == []
        assert ev.status == EvidenceStatus.VALID.value

    def test_multiple_global_entries_all_invalidated_by_one_mutation(self, tmp_path):
        store = CompletionEvidenceStore(tmp_path)
        ev1 = store.record(task_id="t1", subtask_id="s1", turn_number=1, stage="x",
                            evidence_type=EvidenceType.TEST_EXECUTION, source="x", target_paths=["a.py"])
        ev2 = store.record(task_id="t1", subtask_id="s1", turn_number=1, stage="x",
                            evidence_type=EvidenceType.CODE_REVIEW, source="x", target_paths=["a.py"])
        ev3 = store.record(task_id="t1", subtask_id="s1", turn_number=1, stage="x",
                            evidence_type=EvidenceType.DIFF_INSPECTION, source="x", target_paths=["a.py"])
        invalidated = store.invalidate_on_file_mutation(["unrelated.py"])
        assert set(invalidated) == {ev1.evidence_id, ev2.evidence_id, ev3.evidence_id}


# =============================================================================
# Section 3: obligation-level behavioral staleness (task_contract.py)
# =============================================================================

class TestObligationBehavioralStaleness:
    def test_stale_behavioral_evidence_no_longer_grants_executable_behavioral_after_unrelated_mutation(self, tmp_path):
        """The fix's benefit automatically extends to Phase 4.22/4.23
        obligation-level binding, since both read the same
        CompletionEvidenceStore.get_valid_evidence()."""
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        (tmp_path / "impl.py").write_text("def export_csv(): return 'csv'\n", encoding="utf-8")
        (tmp_path / "other.py").write_text("x = 1\n", encoding="utf-8")
        store.record(
            task_id="t1", subtask_id="main", turn_number=1, stage="verifying",
            evidence_type=EvidenceType.TEST_EXECUTION, source="behavioral_verification_synthesizer",
            trust_tier=EvidenceTrustTier.AUTHORITATIVE_EXECUTION,
            target_paths=["impl.py"], target_symbols=["export_csv"],
            command=["pytest"], exit_code=0, payload={"synthesized": True, "trivial": False},
        )
        ob = AcceptanceObligation(obligation_id="OB1", description="CSV export", target_tokens=["csv"], target_paths=["impl.py"])
        req = Requirement(requirement_id="REQ-001", statement="Add CSV export", acceptance_obligations=[ob], target_paths=["impl.py"])
        contract = TaskContract(task_id="t1", objective="Add CSV export", requirements=[req])
        engine = RequirementAssessmentEngine(fs)

        # Before any mutation: fresh, legitimate EXECUTABLE_BEHAVIORAL match.
        result = engine.assess(contract, store, [FileOperation(action="modify", path="impl.py")], "+def export_csv(): ...", last_review=None)
        assert result.requirements[0].acceptance_obligations[0].method == "executable_behavioral"
        assert result.requirements[0].state == RequirementState.SATISFIED.value

        # A later, unrelated file mutation occurs (e.g. a repair turn on a
        # different file) without a fresh test run.
        (tmp_path / "other.py").write_text("x = 2  -- changed\n", encoding="utf-8")
        store.invalidate_on_file_mutation(["other.py"])

        result2 = engine.assess(contract, store, [FileOperation(action="modify", path="impl.py")], "+def export_csv(): ...", last_review=None)
        # The stale behavioral evidence is gone; the obligation must not
        # still claim EXECUTABLE_BEHAVIORAL from it.
        assert result2.requirements[0].acceptance_obligations[0].method != "executable_behavioral"


# =============================================================================
# Section 4: attack surfaces A-L (per Phase 4.24 investigation targets)
# =============================================================================

class TestAttackSurfaces:
    # --- B: negative/failure evidence dominance, order-invariant ---
    @pytest.mark.parametrize("order", [("fail", "pass"), ("pass", "fail")])
    def test_b_fail_and_pass_coexist_fail_always_dominates_regardless_of_order(self, tmp_path, order):
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        engine = CompletionDecisionEngine(fs)
        for kind in order:
            store.record(task_id="t1", subtask_id="s1", turn_number=1, stage="testing",
                         evidence_type=EvidenceType.TEST_EXECUTION, source="x",
                         command=["pytest"], exit_code=(0 if kind == "pass" else 1))
        task = make_task("t1")
        plan = Plan(objective="x")
        assessment = engine.evaluate(task, None, plan, store, [FileOperation(action="modify", path="a.py")], "diff")
        gate = next(g for g in assessment.gates_evaluated if g.gate_name == "GATE_VALIDATION_PASSED")
        assert gate.passed is False

    # --- C: partial compound completion, one fresh + one stale ---
    def test_c_one_obligation_fresh_one_stale_does_not_satisfy_whole_requirement(self, tmp_path):
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        ob_a = AcceptanceObligation(obligation_id="OB-A", description="csv export", target_tokens=["csv"], target_paths=["a.py"])
        ob_b = AcceptanceObligation(obligation_id="OB-B", description="json export", target_tokens=["json"], target_paths=["b.py"])
        req = Requirement(requirement_id="REQ-001", statement="Add CSV and JSON export", acceptance_obligations=[ob_a, ob_b])
        contract = TaskContract(task_id="t1", objective="Add CSV and JSON export", requirements=[req])
        store.record(task_id="t1", subtask_id="main", turn_number=1, stage="verifying",
                     evidence_type=EvidenceType.TEST_EXECUTION, source="behavioral_verification_synthesizer",
                     trust_tier=EvidenceTrustTier.AUTHORITATIVE_EXECUTION,
                     target_paths=["a.py"], target_symbols=["export_csv"],
                     command=["pytest"], exit_code=0, payload={"synthesized": True, "trivial": False})
        # b.py's evidence never recorded at all -- OB-B has no proof.
        engine = RequirementAssessmentEngine(fs)
        result = engine.assess(contract, store, [FileOperation(action="modify", path="a.py"), FileOperation(action="modify", path="b.py")],
                               "+def export_csv(): ...", last_review=None)
        assert result.satisfied is False
        assert result.requirements[0].state != RequirementState.SATISFIED.value

    # --- E: multi-turn/orchestrator parity on invalidation ---
    def test_e_both_execution_paths_call_invalidate_on_file_mutation_after_applying_changes(self):
        orch_src = inspect.getsource(Orchestrator.run)
        mt_src = inspect.getsource(MultiTurnImplementationAgent.execute)
        assert "invalidate_on_file_mutation" in orch_src
        assert "invalidate_on_file_mutation" in mt_src

    # --- H: empty/missing values must not become dangerous defaults ---
    def test_h_test_execution_with_empty_target_paths_and_no_mutation_still_counts(self, tmp_path):
        """Sanity: an entry with empty target_paths (legacy-shaped, never
        anchored to any file) is untouched by a mutation call with an empty
        modified-paths list -- only a real mutation call invalidates it,
        matching the "global, no anchor" semantics documented on the method."""
        store = CompletionEvidenceStore(tmp_path)
        ev = store.record(task_id="t1", subtask_id="s1", turn_number=1, stage="x",
                          evidence_type=EvidenceType.TEST_EXECUTION, source="x", exit_code=0)
        assert ev.status == EvidenceStatus.VALID.value

    def test_h_forged_evidence_type_string_does_not_match_any_gate(self, tmp_path):
        store = CompletionEvidenceStore(tmp_path)
        store.record(task_id="t1", subtask_id="s1", turn_number=1, stage="x",
                     evidence_type="not_a_real_type", source="x", exit_code=0)
        assert store.get_valid_evidence(EvidenceType.TEST_EXECUTION, task_id="t1") == []

    # --- I: type/case confusion in evidence_type filter ---
    def test_i_evidence_type_filter_is_case_and_type_sensitive(self, tmp_path):
        store = CompletionEvidenceStore(tmp_path)
        store.record(task_id="t1", subtask_id="s1", turn_number=1, stage="x",
                     evidence_type=EvidenceType.TEST_EXECUTION, source="x", exit_code=0)
        assert store.get_valid_evidence("TEST_EXECUTION", task_id="t1") == []  # wrong case
        assert len(store.get_valid_evidence("test_execution", task_id="t1")) == 1  # correct value

    # --- J: provider/reviewer approval alone cannot substitute for TEST_EVIDENCE strategy ---
    def test_j_approved_review_without_any_test_evidence_leaves_test_evidence_requirement_unverified(self, tmp_path):
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        from local_agent.task_contract import VerificationStrategy
        req = Requirement(requirement_id="REQ-001", statement="Provide passing test coverage as requested.",
                          verification_strategy=VerificationStrategy.TEST_EVIDENCE.value)
        contract = TaskContract(task_id="t1", objective="Add tests", requirements=[req])
        review = ReviewResult(verdict="APPROVED", summary="looks great, trust me")
        engine = RequirementAssessmentEngine(fs)
        result = engine.assess(contract, store, [FileOperation(action="modify", path="a.py")], "+x=1", last_review=review)
        assert result.requirements[0].state == RequirementState.UNVERIFIED.value
        assert result.satisfied is False

    # --- K: ordering invariance of duplicate identical evidence ---
    def test_k_duplicate_identical_passing_evidence_is_order_invariant(self, tmp_path):
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        engine = CompletionDecisionEngine(fs)
        for _ in range(3):
            store.record(task_id="t1", subtask_id="s1", turn_number=1, stage="testing",
                         evidence_type=EvidenceType.TEST_EXECUTION, source="x",
                         target_paths=["a.py"], command=["pytest"], exit_code=0)
        task = make_task("t1")
        plan = Plan(objective="x", files_likely_to_change=["a.py"])
        ops = [FileOperation(action="modify", path="a.py")]
        assessment = engine.evaluate(task, None, plan, store, ops, "diff")
        gate = next(g for g in assessment.gates_evaluated if g.gate_name == "GATE_VALIDATION_PASSED")
        assert gate.passed is True

    # --- L: false-positive campaign ---
    def test_l_syntax_evidence_for_untouched_sibling_file_survives_repair(self, tmp_path):
        """A syntax-verification pass for an untouched file must remain
        valid while a sibling file is repaired -- the fix must not make
        every unrelated turn re-verify syntax on files nobody touched."""
        (tmp_path / "stable.py").write_text("def stable(): pass\n", encoding="utf-8")
        (tmp_path / "unstable.py").write_text("def unstable(): pass\n", encoding="utf-8")
        store = CompletionEvidenceStore(tmp_path)
        ev = store.record(task_id="t1", subtask_id="s1", turn_number=1, stage="verifying",
                          evidence_type=EvidenceType.SYNTAX_VERIFICATION, source="ast_parser",
                          target_paths=["stable.py"], exit_code=0)
        (tmp_path / "unstable.py").write_text("def unstable(): pass  # repaired\n", encoding="utf-8")
        store.invalidate_on_file_mutation(["unstable.py"], reason="repaired_after_execution")
        assert ev.status == EvidenceStatus.VALID.value

    def test_l_camelcase_symbol_false_positive_still_excluded_after_fix(self, tmp_path):
        """Regression guard: the Phase 4.24 fix must not interact badly with
        the Phase 4.23 component-matching fix for camelCase symbols."""
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        store.record(task_id="t1", subtask_id="main", turn_number=1, stage="verifying",
                     evidence_type=EvidenceType.TEST_EXECUTION, source="behavioral_verification_synthesizer",
                     trust_tier=EvidenceTrustTier.AUTHORITATIVE_EXECUTION,
                     target_paths=["impl.py"], target_symbols=["exportCsvImportant"],
                     command=["pytest"], exit_code=0, payload={"synthesized": True, "trivial": False})
        ob = AcceptanceObligation(obligation_id="OB1", description="import feature", target_tokens=["import"], target_paths=["impl.py"])
        req = Requirement(requirement_id="REQ-001", statement="Add import feature", acceptance_obligations=[ob])
        contract = TaskContract(task_id="t1", objective="Add import feature", requirements=[req])
        engine = RequirementAssessmentEngine(fs)
        result = engine.assess(contract, store, [FileOperation(action="modify", path="impl.py")], "+import stuff", last_review=None)
        # "import" is not a real component of "exportCsvImportant" (its
        # components are export/Csv/Important) -- must not grant behavioral
        # proof via fuzzy substring containment.
        assert result.requirements[0].acceptance_obligations[0].method != "executable_behavioral"


# =============================================================================
# Section 5: invariants
# =============================================================================

class TestInvariants:
    def test_p1_global_evidence_cannot_outlive_any_tree_mutation(self, tmp_path):
        store = CompletionEvidenceStore(tmp_path)
        ev = store.record(task_id="t1", subtask_id="s1", turn_number=1, stage="x",
                          evidence_type=EvidenceType.TEST_EXECUTION, source="x", target_paths=["a.py"])
        store.invalidate_on_file_mutation(["z_never_seen_before.py"])
        assert ev.status == EvidenceStatus.INVALIDATED.value

    def test_p2_per_file_evidence_scope_is_preserved_not_widened(self, tmp_path):
        (tmp_path / "a.py").write_text("v1", encoding="utf-8")
        store = CompletionEvidenceStore(tmp_path)
        ev = store.record(task_id="t1", subtask_id="s1", turn_number=1, stage="x",
                          evidence_type=EvidenceType.SAFETY_INVARIANT, source="x", target_paths=["a.py"])
        store.invalidate_on_file_mutation(["b.py"])
        assert ev.status == EvidenceStatus.VALID.value

    def test_p3_readiness_engine_and_requirement_engine_both_see_the_same_invalidation(self, tmp_path):
        """No second evidence ledger: both completion.py and task_contract.py
        read through the same CompletionEvidenceStore.get_valid_evidence, so
        an invalidation is visible to both consumers atomically."""
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        store.record(task_id="t1", subtask_id="s1", turn_number=1, stage="testing",
                     evidence_type=EvidenceType.TEST_EXECUTION, source="x",
                     target_paths=["a.py"], command=["pytest"], exit_code=0)
        store.invalidate_on_file_mutation(["b.py"])
        completion_engine = CompletionDecisionEngine(fs)
        task = make_task("t1")
        plan = Plan(objective="x")
        assessment = completion_engine.evaluate(task, None, plan, store, [FileOperation(action="modify", path="a.py")], "diff")
        gate = next(g for g in assessment.gates_evaluated if g.gate_name == "GATE_VALIDATION_PASSED")
        assert gate.passed is False

        req = Requirement(requirement_id="REQ-001", statement="x",
                          verification_strategy="test_evidence")
        contract = TaskContract(task_id="t1", objective="x", requirements=[req])
        req_engine = RequirementAssessmentEngine(fs)
        result = req_engine.assess(contract, store, [FileOperation(action="modify", path="a.py")], "diff", last_review=None)
        assert result.requirements[0].state == RequirementState.UNVERIFIED.value


# =============================================================================
# Section 6: performance (measured)
# =============================================================================

class TestPerformance:
    def test_invalidate_on_file_mutation_scales_linearly_not_quadratically(self, tmp_path):
        store = CompletionEvidenceStore(tmp_path, max_entries=2000)
        for i in range(500):
            store.record(task_id="t1", subtask_id="s1", turn_number=i, stage="x",
                         evidence_type=EvidenceType.TEST_EXECUTION, source="x", target_paths=[f"f{i}.py"])
        start = time.perf_counter()
        for i in range(50):
            store.invalidate_on_file_mutation([f"m{i}.py"])
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0, f"50 invalidation passes over 500 entries took {elapsed:.3f}s"


# =============================================================================
# Section 7: backward compatibility
# =============================================================================

class TestBackwardCompatibility:
    def test_old_style_checkpoint_evidence_dict_still_invalidates_correctly(self, tmp_path):
        """A hand-built dict matching the pre-4.24 on-disk shape (no new
        fields were added by this fix, so this is a pure behavioural check)
        loads and is subject to the corrected invalidation logic."""
        old_dict = {
            "workspace_root": str(tmp_path),
            "max_entries": 100,
            "entries": [{
                "evidence_id": "ev-1", "task_id": "t1", "subtask_id": "s1",
                "turn_number": 1, "stage": "testing", "evidence_type": "test_execution",
                "source": "command_runner", "trust_tier": 1, "status": "valid",
                "target_paths": ["a.py"], "target_symbols": [], "exit_code": 0,
                "content_fingerprint": "", "payload": {}, "timestamp": "2020-01-01T00:00:00+00:00",
            }],
            "types_ever_recorded": ["test_execution"],
            "next_seq": 1,
        }
        store = CompletionEvidenceStore.from_dict(old_dict)
        invalidated = store.invalidate_on_file_mutation(["b.py"])
        assert invalidated == ["ev-1"]

    def test_get_valid_evidence_without_task_id_unchanged(self, tmp_path):
        store = CompletionEvidenceStore(tmp_path)
        store.record(task_id="t1", subtask_id="s1", turn_number=1, stage="x",
                     evidence_type=EvidenceType.TEST_EXECUTION, source="x", exit_code=0)
        assert len(store.get_valid_evidence(EvidenceType.TEST_EXECUTION)) == 1
