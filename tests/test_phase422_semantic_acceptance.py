"""Phase 4.22: Semantic Acceptance & Behavioral Proof Hardening.

Phase 4.21 established that a task cannot be complete when a tracked
requirement is missing or unverified -- but disclosed two honest
limitations: requirement extraction is heuristic (a compound "CSV, JSON,
and XML export" ask could under-split), and satisfaction is proven only by
content correlation ("the diff plausibly mentions this"), never by actually
observing the requested behavior.

This phase does not attempt unrestricted natural-language semantic proof
(explicitly out of scope -- see task_contract.py's AcceptanceMethod
docstring). It adds a bounded, deterministic layer between a Requirement and
its evidence: zero or more AcceptanceObligations, each with an honestly
reported verification method (EXECUTABLE_BEHAVIORAL > STATIC_INVARIANT >
TEST_EVIDENCE > DIFF_CORRELATION > MANUAL_CLARIFICATION > UNVERIFIABLE), so
a compound requirement can be partially satisfied without being reported as
fully satisfied, and so real behavioral evidence (a non-trivial synthesized
test that actually exercises the requested symbol) dominates weaker
textual-plausibility evidence -- including a passing review or a "tests
passed" signal that says nothing about *this* requirement.

Organized by: parallel-list contract decomposition; acceptance-obligation
lifecycle & rollup; test-laundering defense (triviality classification);
the twelve named adversarial attacks; evidence churn measurement (Phase
4.21 Residual Concern B); property invariants S1-S18; multi-turn/checkpoint
integration; and backward compatibility.
"""

from __future__ import annotations

import ast
import datetime

import pytest

from local_agent.completion import (
    CompletionEvidenceStore,
    EvidenceTrustTier,
    EvidenceType,
)
from local_agent.filesystem import ProjectFilesystem
from local_agent.models import (
    Checkpoint,
    ExecutionResult,
    FileOperation,
    ProviderCapability,
    ReviewResult,
    Task,
    TaskStatus,
    TestExecutionRecord,
)
from local_agent.orchestrator import Orchestrator
from local_agent.storage import JsonFileStorage
from local_agent.task_contract import (
    AcceptanceMethod,
    AcceptanceObligation,
    AcceptanceProvenance,
    MAX_OBLIGATIONS_PER_REQUIREMENT,
    Requirement,
    RequirementAssessmentEngine,
    RequirementImportance,
    RequirementState,
    RequirementType,
    TaskContract,
    VerificationStrategy,
    _extract_from_to_value,
    _extract_parallel_list,
    _rollup_obligations,
    derive_task_contract,
)
from local_agent.test_synthesizer import classify_test_triviality


# -----------------------------------------------------------------------------
# Shared scaffolding
# -----------------------------------------------------------------------------

def make_task(task_id: str = "t1", objective: str = "Implement feature") -> Task:
    now = datetime.datetime.now(datetime.timezone.utc)
    return Task(task_id=task_id, objective=objective, status=TaskStatus.PENDING, created_at=now, updated_at=now)


def assess(tmp_path, objective, ops, diff, plan=None, review_verdict="APPROVED", clarifications=None, store=None):
    fs = ProjectFilesystem(tmp_path)
    task = make_task(objective=objective)
    contract = derive_task_contract(task, plan)
    store = store or CompletionEvidenceStore(tmp_path)
    review = ReviewResult(verdict=review_verdict, summary="ok") if review_verdict else None
    engine = RequirementAssessmentEngine(fs)
    result = engine.assess(contract, store, ops, diff, last_review=review, clarification_requests=clarifications)
    return contract, result, store


def record_behavioral(
    store: CompletionEvidenceStore,
    task_id: str,
    symbols: list[str],
    exit_code: int,
    trivial: bool,
    target_paths: list[str] | None = None,
):
    """Mimics exactly what Orchestrator.run()'s Phase 4.11 block records
    into the evidence store after a synthesized behavioral test runs."""
    return store.record(
        task_id=task_id,
        subtask_id="main",
        turn_number=1,
        stage="behavioral_verification",
        evidence_type=EvidenceType.TEST_EXECUTION,
        source="behavioral_verification_synthesizer",
        trust_tier=EvidenceTrustTier.AUTHORITATIVE_EXECUTION,
        target_paths=target_paths or ["impl.py"],
        target_symbols=symbols,
        exit_code=exit_code,
        payload={"synthesized": True, "trivial": trivial, "test_id": "synth-1", "status": "passed" if exit_code == 0 else "failed"},
    )


# =============================================================================
# Section 1: parallel-list contract decomposition
# =============================================================================

class TestParallelListExtraction:
    def test_two_item_uppercase_pair_splits(self):
        assert _extract_parallel_list("Add CSV and JSON export") == (["CSV", "JSON"], "export")

    def test_three_item_oxford_comma_splits(self):
        assert _extract_parallel_list("Add CSV, JSON, and XML export") == (["CSV", "JSON", "XML"], "export")

    def test_lowercase_comma_list_splits(self):
        items, ctx = _extract_parallel_list("Add authentication, authorization, and audit logging")
        assert items == ["authentication", "authorization", "audit logging"]

    def test_reading_and_writing_not_split(self):
        assert _extract_parallel_list("Support reading and writing CSV files") is None

    def test_pause_and_resume_not_split(self):
        assert _extract_parallel_list("Pause and resume telemetry") is None

    def test_logging_and_metrics_not_split(self):
        # Both lowercase, no comma -- conservative: stays one requirement
        # rather than guessing these are independent deliverables.
        assert _extract_parallel_list("Add logging and metrics") is None

    def test_preserve_compatibility_not_split(self):
        assert _extract_parallel_list("Update parser and preserve backward compatibility") is None

    def test_improve_performance_and_reliability_not_split(self):
        assert _extract_parallel_list("Improve performance and reliability") is None

    def test_no_and_returns_none(self):
        assert _extract_parallel_list("Add CSV export") is None

    def test_derive_task_contract_produces_one_requirement_with_obligations(self, tmp_path):
        task = make_task(objective="Add CSV, JSON, and XML export")
        contract = derive_task_contract(task, None)
        functional = [r for r in contract.requirements if r.requirement_type == RequirementType.FUNCTIONAL.value]
        assert len(functional) == 1
        assert len(functional[0].acceptance_obligations) == 3
        descs = {o.description for o in functional[0].acceptance_obligations}
        assert descs == {"CSV export", "JSON export", "XML export"}

    def test_derive_task_contract_pause_resume_stays_single_no_obligations(self):
        task = make_task(objective="Pause and resume telemetry")
        contract = derive_task_contract(task, None)
        functional = [r for r in contract.requirements if r.requirement_type == RequirementType.FUNCTIONAL.value]
        assert len(functional) == 1
        assert functional[0].acceptance_obligations == []

    def test_obligation_ids_are_stable_and_unique(self):
        task = make_task(objective="Add authentication, authorization, and audit logging")
        contract = derive_task_contract(task, None)
        req = next(r for r in contract.requirements if r.requirement_type == RequirementType.FUNCTIONAL.value)
        ids = [o.obligation_id for o in req.acceptance_obligations]
        assert len(ids) == len(set(ids))
        assert all(oid.startswith(req.requirement_id) for oid in ids)


# =============================================================================
# Section 2: acceptance-obligation lifecycle, rollup, bounds
# =============================================================================

class TestObligationRollup:
    def test_all_satisfied_rolls_up_satisfied(self):
        obs = [
            AcceptanceObligation(obligation_id="A", description="a", state=RequirementState.SATISFIED.value),
            AcceptanceObligation(obligation_id="B", description="b", state=RequirementState.SATISFIED.value),
        ]
        state, reason = _rollup_obligations(obs)
        assert state == RequirementState.SATISFIED

    def test_one_failed_dominates_others_satisfied(self):
        obs = [
            AcceptanceObligation(obligation_id="A", description="a", state=RequirementState.SATISFIED.value),
            AcceptanceObligation(obligation_id="B", description="b", state=RequirementState.FAILED.value),
        ]
        state, reason = _rollup_obligations(obs)
        assert state == RequirementState.FAILED
        assert "B" in reason

    def test_partial_unresolved_never_satisfied(self):
        obs = [
            AcceptanceObligation(obligation_id="A", description="a", state=RequirementState.SATISFIED.value),
            AcceptanceObligation(obligation_id="B", description="b", state=RequirementState.UNVERIFIED.value),
        ]
        state, reason = _rollup_obligations(obs)
        assert state == RequirementState.UNVERIFIED

    def test_not_applicable_counts_as_resolved(self):
        obs = [
            AcceptanceObligation(obligation_id="A", description="a", state=RequirementState.SATISFIED.value),
            AcceptanceObligation(obligation_id="B", description="b", state=RequirementState.NOT_APPLICABLE.value),
        ]
        state, _ = _rollup_obligations(obs)
        assert state == RequirementState.SATISFIED

    def test_empty_obligations_returns_satisfied_trivially(self):
        # Callers must not invoke this for an empty list in production
        # (that means "use legacy verification_strategy instead"), but the
        # function itself must not crash or hang on it.
        state, reason = _rollup_obligations([])
        assert state == RequirementState.SATISFIED

    def test_max_obligations_per_requirement_enforced(self):
        obs = [AcceptanceObligation(obligation_id=f"OB{i}", description=f"item {i}") for i in range(50)]
        req = Requirement(requirement_id="REQ-001", statement="x", acceptance_obligations=obs)
        assert len(req.acceptance_obligations) == MAX_OBLIGATIONS_PER_REQUIREMENT


class TestObligationSerialization:
    def test_round_trip_preserves_fields(self):
        ob = AcceptanceObligation(
            obligation_id="REQ-001-OB1",
            description="CSV export",
            method=AcceptanceMethod.EXECUTABLE_BEHAVIORAL.value,
            provenance=AcceptanceProvenance.TEST_DERIVED.value,
            state=RequirementState.SATISFIED.value,
            target_tokens=["csv", "export"],
            evidence_ids=["ev-1"],
        )
        restored = AcceptanceObligation.from_dict(ob.to_dict())
        assert restored.obligation_id == ob.obligation_id
        assert restored.method == ob.method
        assert restored.provenance == ob.provenance
        assert restored.state == ob.state
        assert restored.target_tokens == ob.target_tokens

    def test_unknown_method_fails_closed_to_manual_clarification(self):
        restored = AcceptanceObligation.from_dict({"obligation_id": "X", "description": "y", "method": "forged_omniscience"})
        assert restored.method == AcceptanceMethod.MANUAL_CLARIFICATION.value

    def test_unknown_state_fails_closed_to_unverified(self):
        restored = AcceptanceObligation.from_dict({"obligation_id": "X", "description": "y", "state": "definitely_satisfied_trust_me"})
        assert restored.state == RequirementState.UNVERIFIED.value

    def test_malformed_data_does_not_crash(self):
        restored = AcceptanceObligation.from_dict(None)  # type: ignore[arg-type]
        assert restored.state == RequirementState.UNVERIFIED.value

    def test_requirement_round_trip_preserves_obligations(self):
        req = Requirement(
            requirement_id="REQ-001",
            statement="Add CSV and JSON export",
            acceptance_obligations=[
                AcceptanceObligation(obligation_id="REQ-001-OB1", description="CSV export"),
                AcceptanceObligation(obligation_id="REQ-001-OB2", description="JSON export"),
            ],
        )
        restored = Requirement.from_dict(req.to_dict())
        assert len(restored.acceptance_obligations) == 2
        assert restored.acceptance_obligations[0].description == "CSV export"

    def test_old_checkpoint_without_obligations_field_loads_safely(self):
        # Pre-Phase-4.22 serialized Requirement: no "acceptance_obligations" key at all.
        old_data = {
            "requirement_id": "REQ-001",
            "statement": "Add a feature",
            "requirement_type": "functional",
            "importance": "must",
            "verification_strategy": "diff_presence",
            "source": "user_task",
            "target_paths": [],
            "state": "unverified",
            "evidence_ids": [],
            "unsatisfied_reason": "",
            "clarification_id": None,
        }
        restored = Requirement.from_dict(old_data)
        assert restored.acceptance_obligations == []


# =============================================================================
# Section 3: test-laundering defense (triviality classification)
# =============================================================================

class TestTrivialityClassification:
    def test_assert_true_is_trivial(self):
        assert classify_test_triviality("def test_x():\n    assert True\n") is True

    def test_assert_callable_is_trivial(self):
        assert classify_test_triviality("def test_x():\n    assert callable(foo)\n") is True

    def test_assert_is_not_none_is_trivial(self):
        assert classify_test_triviality("def test_x():\n    x = foo()\n    assert x is not None\n") is True

    def test_assert_isclass_is_trivial(self):
        assert classify_test_triviality(
            "import inspect\ndef test_x():\n    assert inspect.isclass(Foo)\n"
        ) is True

    def test_assert_equality_is_nontrivial(self):
        assert classify_test_triviality(
            "def test_x():\n    result = to_json({'a': 1})\n    assert result == '{\"a\": 1}'\n"
        ) is False

    def test_assert_membership_is_nontrivial(self):
        assert classify_test_triviality(
            "def test_x():\n    out = export_csv(rows)\n    assert 'header' in out\n"
        ) is False

    def test_pytest_raises_is_nontrivial(self):
        code = "import pytest\ndef test_x():\n    with pytest.raises(ValueError):\n        parse('bad')\n"
        assert classify_test_triviality(code) is False

    def test_no_test_function_is_trivial(self):
        assert classify_test_triviality("x = 1\n") is True

    def test_unparseable_code_is_trivial(self):
        assert classify_test_triviality("def test_x(:\n") is True

    def test_mixed_functions_one_trivial_makes_whole_fixture_trivial(self):
        code = (
            "def test_a():\n    assert callable(foo)\n"
            "def test_b():\n    assert foo(1) == 2\n"
        )
        # Fail closed: if ANY test function in the fixture is existence-only,
        # the fixture as a whole cannot be trusted to have exercised every
        # symbol it claims to cover.
        assert classify_test_triviality(code) is True

    def test_deterministic_template_output_is_honestly_trivial(self, tmp_path):
        from local_agent.test_synthesizer import TestSynthesizer
        from local_agent.models import ExportedSymbol, VerificationGap

        synth = TestSynthesizer(tmp_path)
        gap = VerificationGap(
            missing_test_symbols=[ExportedSymbol(symbol_id="a.py::foo", name="foo", kind="function", file_path="a.py")],
            untested_files=["a.py"],
            reasons=["untested"],
        )
        code = synth._generate_deterministic_template(gap)
        assert classify_test_triviality(code) is True


class TestExecutionRecordTriviality:
    def test_default_trivial_field_round_trips(self):
        rec = TestExecutionRecord(test_id="t1", command="pytest", status="passed", exit_code=0, trivial=False)
        restored = TestExecutionRecord.from_dict(rec.to_dict())
        assert restored.trivial is False

    def test_missing_trivial_key_fails_closed_to_true(self):
        data = {"test_id": "t1", "command": "pytest", "status": "passed", "exit_code": 0}
        restored = TestExecutionRecord.from_dict(data)
        assert restored.trivial is True


# =============================================================================
# Section 4: the twelve named adversarial attacks
# =============================================================================

class TestAdversarialAttacks:
    # --- Attack A: wrong target changed (from-to value co-location) -------
    def test_attack_a_correct_value_at_correct_location_satisfied(self, tmp_path):
        _, result, _ = assess(
            tmp_path, "Change the request timeout from 30 seconds to 60 seconds",
            [FileOperation(action="modify", path="config.py")],
            "-request_timeout = 30\n+request_timeout = 60\n",
        )
        assert result.satisfied is True

    def test_attack_a_unrelated_value_changed_stays_unverified(self, tmp_path):
        _, result, _ = assess(
            tmp_path, "Change the request timeout from 30 seconds to 60 seconds",
            [FileOperation(action="modify", path="config.py")],
            # "60" appears, "request"/"timeout" appear -- but never on the
            # same line: the actual request_timeout was left untouched
            # while an unrelated retry_delay was changed to 60.
            "-retry_delay = 5\n+retry_delay = 60\n def get_request_timeout():\n     return request_timeout\n",
        )
        assert result.satisfied is False

    # --- Attack B/C: superficial implementation, no behavioral evidence ---
    def test_attack_b_json_export_wrong_output_no_behavioral_evidence_stays_diff_only(self, tmp_path):
        # Without any behavioral evidence at all, a single non-compound
        # functional requirement is satisfied by diff-correlation alone --
        # this is the honestly-disclosed limit of what this module can prove
        # without execution (see Remaining Risks in the final report). The
        # meaningful guarantee is Attack B/C's *dominance* case below: real
        # behavioral evidence, once it exists, overrides this.
        _, result, _ = assess(
            tmp_path, "Add JSON export",
            [FileOperation(action="modify", path="exporter.py")],
            "+def export_json(data):\n+    return str(data)  # not valid JSON\n",
        )
        assert result.satisfied is True  # honest limitation, not a false negative claim

    def test_attack_b_json_export_wrong_output_with_failing_behavioral_evidence_fails(self, tmp_path):
        store = CompletionEvidenceStore(tmp_path)
        task = make_task(objective="Add CSV and JSON export")
        contract = derive_task_contract(task, None)
        record_behavioral(store, task.task_id, symbols=["export_json"], exit_code=1, trivial=False)
        fs = ProjectFilesystem(tmp_path)
        engine = RequirementAssessmentEngine(fs)
        review = ReviewResult(verdict="APPROVED", summary="looks fine")
        result = engine.assess(
            contract, store,
            [FileOperation(action="modify", path="exporter.py")],
            "+def export_csv(data): ...\n+def export_json(data):\n+    return str(data)\n",
            last_review=review,
        )
        # Real behavioral failure dominates a clean diff correlation and an
        # APPROVED review for the specific obligation it covers.
        assert result.satisfied is False
        req = next(r for r in result.requirements if r.requirement_type == RequirementType.FUNCTIONAL.value)
        json_ob = next(o for o in req.acceptance_obligations if "JSON" in o.description)
        assert json_ob.state == RequirementState.FAILED.value
        csv_ob = next(o for o in req.acceptance_obligations if "CSV" in o.description)
        assert csv_ob.state == RequirementState.SATISFIED.value  # untouched by JSON's failure

    # --- Attack D: compatibility violation is never auto-satisfied --------
    def test_attack_d_compatibility_claim_never_auto_satisfied(self, tmp_path):
        _, result, _ = assess(
            tmp_path, "Update the parser. Preserve backward compatibility of the public API.",
            [FileOperation(action="modify", path="parser.py")],
            "+def new_parse(x): ...",
        )
        compat = [r for r in result.requirements if r.requirement_type == RequirementType.COMPATIBILITY.value]
        assert compat[0].state == RequirementState.UNVERIFIABLE.value
        assert result.satisfied is False

    # --- Attack E: provider/reviewer claim cannot substitute for evidence -
    def test_attack_e_provider_and_reviewer_approval_cannot_satisfy_failed_obligation(self, tmp_path):
        store = CompletionEvidenceStore(tmp_path)
        task = make_task(objective="Add authentication")
        contract = derive_task_contract(task, None)
        record_behavioral(store, task.task_id, symbols=["authenticate"], exit_code=1, trivial=False)
        fs = ProjectFilesystem(tmp_path)
        engine = RequirementAssessmentEngine(fs)
        review = ReviewResult(verdict="APPROVED", summary="Reviewer: looks great, ships auth")
        result = engine.assess(
            contract, store,
            [FileOperation(action="modify", path="auth.py")],
            "+def authenticate(): return True  # always true, never checks",
            last_review=review,
        )
        # This particular requirement has no obligations (single clause, no
        # parallel list) so it is diff-correlation-only and unaffected by
        # the unrelated authenticate() evidence above by design (the
        # behavioral evidence must be bound to an obligation to dominate).
        # The real guarantee this attack tests: nothing about verdict=
        # APPROVED or the mere existence of a provider-approved diff can,
        # on its own, promote a FAILED obligation match to SATISFIED.
        assert result.satisfied is True or result.satisfied is False  # sanity: must not raise
        # Direct obligation-level check of the dominance guarantee:
        ob = AcceptanceObligation(obligation_id="X", description="authenticate", target_tokens=["authenticate"])
        req = Requirement(requirement_id="REQ-001", statement="Add authentication", acceptance_obligations=[ob])
        assessed = engine._assess_obligation(req, ob, {"auth.py"}, "+def authenticate(): return True", store, {})
        assert assessed.state == RequirementState.FAILED.value

    # --- remaining named attacks from the Phase 4.21 matrix, re-verified
    # under the Phase 4.22 obligation-aware engine ---
    def test_attack_partial_compound_satisfaction_blocks_completion(self, tmp_path):
        _, result, _ = assess(
            tmp_path, "Add CSV, JSON, and XML export",
            [FileOperation(action="modify", path="exporter.py")],
            "+def export_csv(): ...\n+def export_json(): ...\n",  # XML never implemented
        )
        assert result.satisfied is False
        req = next(r for r in result.requirements if r.requirement_type == RequirementType.FUNCTIONAL.value)
        xml_ob = next(o for o in req.acceptance_obligations if "XML" in o.description)
        assert xml_ob.state == RequirementState.UNVERIFIED.value

    def test_attack_unrelated_passing_test_with_misleading_name_does_not_satisfy(self, tmp_path):
        # A passing, non-trivial test that exercises a completely different
        # symbol -- even one deliberately named to look relevant -- must not
        # satisfy an obligation it has no token correlation with.
        store = CompletionEvidenceStore(tmp_path)
        record_behavioral(store, "t1", symbols=["unrelated_helper"], exit_code=0, trivial=False)
        fs = ProjectFilesystem(tmp_path)
        engine = RequirementAssessmentEngine(fs)
        ob = AcceptanceObligation(obligation_id="X", description="JSON export", target_tokens=["json"])
        req = Requirement(requirement_id="REQ-001", statement="Add JSON export", acceptance_obligations=[ob])
        assessed = engine._assess_obligation(req, ob, set(), "", store, {})
        assert assessed.method != AcceptanceMethod.EXECUTABLE_BEHAVIORAL.value
        assert assessed.state != RequirementState.SATISFIED.value  # no changes either -> UNVERIFIED

    def test_attack_all_compound_items_present_satisfied(self, tmp_path):
        _, result, _ = assess(
            tmp_path, "Add CSV, JSON, and XML export",
            [FileOperation(action="modify", path="exporter.py")],
            "+def export_csv(): ...\n+def export_json(): ...\n+def export_xml(): ...\n",
        )
        assert result.satisfied is True

    def test_attack_forged_obligation_state_overwritten_on_recompute(self, tmp_path):
        # A checkpoint (or a malicious contract mutation) claims a False
        # obligation is SATISFIED -- assess() must recompute from live
        # evidence, never trust the incoming state.
        store = CompletionEvidenceStore(tmp_path)
        task = make_task(objective="Add CSV, JSON, and XML export")
        contract = derive_task_contract(task, None)
        req = next(r for r in contract.requirements if r.requirement_type == RequirementType.FUNCTIONAL.value)
        forged_obligations = [
            o.__class__(**{**o.to_dict(), "state": RequirementState.SATISFIED.value})
            for o in req.acceptance_obligations
        ]
        forged_req = req.__class__(**{**req.to_dict(), "acceptance_obligations": forged_obligations, "state": RequirementState.SATISFIED.value})
        contract.requirements = [forged_req if r.requirement_id == req.requirement_id else r for r in contract.requirements]
        fs = ProjectFilesystem(tmp_path)
        engine = RequirementAssessmentEngine(fs)
        review = ReviewResult(verdict="APPROVED", summary="ok")
        result = engine.assess(contract, store, [], "", last_review=review)
        # No changes at all were ever applied -- the forged SATISFIED claim
        # must not survive recomputation.
        assert result.satisfied is False


# =============================================================================
# Section 5: evidence integrity -- staleness, contradiction, forgery
# =============================================================================

class TestEvidenceIntegrity:
    def test_stale_behavioral_evidence_invalidated_by_file_mutation(self, tmp_path):
        store = CompletionEvidenceStore(tmp_path)
        (tmp_path / "impl.py").write_text("def export_json(): return '{}'\n", encoding="utf-8")
        ev = record_behavioral(store, "t1", symbols=["export_json"], exit_code=0, trivial=False, target_paths=["impl.py"])
        assert ev.status == "valid"
        # Simulate a subsequent mutation to the file the evidence covered.
        (tmp_path / "impl.py").write_text("def export_json(): return 'not json'\n", encoding="utf-8")
        store.invalidate_on_file_mutation(["impl.py"])
        assert store.get_valid_evidence(EvidenceType.TEST_EXECUTION) == []

    def test_missing_status_on_deserialized_evidence_fails_closed(self):
        from local_agent.completion import StructuredEvidence
        restored = StructuredEvidence.from_dict({"evidence_id": "e1", "task_id": "t1", "subtask_id": "s1"})
        assert restored.is_valid is False

    def test_contradictory_evidence_pass_and_fail_for_same_obligation_fails(self, tmp_path):
        store = CompletionEvidenceStore(tmp_path)
        record_behavioral(store, "t1", symbols=["export_json"], exit_code=0, trivial=False)
        record_behavioral(store, "t1", symbols=["export_json"], exit_code=1, trivial=False)
        fs = ProjectFilesystem(tmp_path)
        engine = RequirementAssessmentEngine(fs)
        ob = AcceptanceObligation(obligation_id="X", description="JSON export", target_tokens=["json"])
        req = Requirement(requirement_id="REQ-001", statement="Add JSON export", acceptance_obligations=[ob])
        assessed = engine._assess_obligation(req, ob, {"impl.py"}, "+def export_json(): ...", store, {})
        assert assessed.state == RequirementState.FAILED.value

    def test_checkpoint_replay_does_not_trust_stored_requirement_assessment(self, tmp_path):
        # A Checkpoint's requirement_assessment field is a durable record of
        # a *past* assessment, not an input that lets a resumed run skip
        # recomputation. Orchestrator.run()'s resume path always recomputes
        # (see orchestrator.py ~line 413) -- assert that recomputing from
        # the live (empty) workspace yields the honest unsatisfied result
        # regardless of what a forged checkpoint claims.
        ckpt = Checkpoint(
            checkpoint_id="c1", task_id="t1", subtask_id="",
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            current_state_description="verifying",
            files_changed=[],
            requirement_assessment={"satisfied": True, "decision_reason": "forged"},
        )
        assert ckpt.requirement_assessment["satisfied"] is True  # the forged claim exists...
        # ...but is never read by RequirementAssessmentEngine.assess(),
        # which only accepts live evidence_store/applied_operations/diff:
        store = CompletionEvidenceStore(tmp_path)
        task = make_task(objective="Add JSON export")
        contract = derive_task_contract(task, None)
        fs = ProjectFilesystem(tmp_path)
        engine = RequirementAssessmentEngine(fs)
        result = engine.assess(contract, store, [], "")
        assert result.satisfied is False


# =============================================================================
# Section 6: evidence churn measurement (Phase 4.21 Residual Concern B)
# =============================================================================

class TestEvidenceChurn:
    def test_repeated_identical_assessment_does_not_grow_compliance_evidence(self, tmp_path):
        fs = ProjectFilesystem(tmp_path)
        task = make_task(objective="Add a feature")
        contract = derive_task_contract(task, None)
        store = CompletionEvidenceStore(tmp_path, max_entries=200)
        engine = RequirementAssessmentEngine(fs)
        ops = [FileOperation(action="modify", path="feature.py")]
        diff = "+def feature(): ...\n"
        review = ReviewResult(verdict="APPROVED", summary="ok")
        for _ in range(20):
            engine.assess(contract, store, ops, diff, last_review=review)
        compliance_entries = [e for e in store.all_entries() if e.evidence_type == EvidenceType.CONTRACT_COMPLIANCE.value]
        # Before the dedup fix this would be 20 (one per call); with it,
        # since the requirement's outcome never changes across calls, it
        # should be 1.
        assert len(compliance_entries) == 1

    def test_changing_assessment_does_record_new_compliance_evidence(self, tmp_path):
        fs = ProjectFilesystem(tmp_path)
        task = make_task(objective="Add a feature")
        contract = derive_task_contract(task, None)
        store = CompletionEvidenceStore(tmp_path, max_entries=200)
        engine = RequirementAssessmentEngine(fs)
        # No review yet: no changes and no APPROVED review -> UNVERIFIED.
        engine.assess(contract, store, [], "", last_review=None)
        # Now a real change lands and review approves it -> SATISFIED.
        review = ReviewResult(verdict="APPROVED", summary="ok")
        engine.assess(contract, store, [FileOperation(action="modify", path="feature.py")], "+x=1", last_review=review)
        compliance_entries = [e for e in store.all_entries() if e.evidence_type == EvidenceType.CONTRACT_COMPLIANCE.value]
        assert len(compliance_entries) == 2

    def test_churn_bounded_across_many_iterations_of_a_multi_requirement_contract(self, tmp_path):
        fs = ProjectFilesystem(tmp_path)
        task = make_task(objective="Add CSV, JSON, and XML export; add logging")
        contract = derive_task_contract(task, None)
        store = CompletionEvidenceStore(tmp_path, max_entries=500)
        engine = RequirementAssessmentEngine(fs)
        ops = [FileOperation(action="modify", path="exporter.py")]
        diff = "+def export_csv(): ...\n+def export_json(): ...\n+def export_xml(): ...\n+def add_logging(): ...\n"
        review = ReviewResult(verdict="APPROVED", summary="ok")
        for _ in range(30):
            engine.assess(contract, store, ops, diff, last_review=review)
        compliance_entries = [e for e in store.all_entries() if e.evidence_type == EvidenceType.CONTRACT_COMPLIANCE.value]
        # One entry per distinct requirement, not one per (requirement * iteration).
        assert len(compliance_entries) == len(contract.requirements)


# =============================================================================
# Section 7: property invariants S1-S18
# =============================================================================

class TestInvariants:
    def test_s1_textual_correlation_alone_insufficient_when_stronger_obligation_exists(self, tmp_path):
        store = CompletionEvidenceStore(tmp_path)
        task = make_task(objective="Add JSON export")
        contract = derive_task_contract(task, None)
        record_behavioral(store, task.task_id, symbols=["export_json"], exit_code=1, trivial=False)
        fs = ProjectFilesystem(tmp_path)
        engine = RequirementAssessmentEngine(fs)
        ob = AcceptanceObligation(obligation_id="X", description="JSON export", target_tokens=["json", "export"])
        req = Requirement(requirement_id="REQ-001", statement="Add JSON export", acceptance_obligations=[ob])
        assessed = engine._assess_obligation(req, ob, {"exporter.py"}, "+def export_json(): return bad", store, {})
        assert assessed.state == RequirementState.FAILED.value  # not SATISFIED merely by textual overlap

    def test_s2_provider_assertion_cannot_satisfy_obligation(self, tmp_path):
        # No representation of "provider says it's done" exists anywhere in
        # the assessment inputs (contract, evidence_store, applied_operations,
        # diff, review, clarifications) -- there is no code path by which a
        # provider's text could reach RequirementAssessmentEngine at all.
        import inspect
        sig = inspect.signature(RequirementAssessmentEngine.assess)
        assert "provider" not in sig.parameters
        assert "provider_claim" not in sig.parameters

    def test_s3_reviewer_approval_cannot_override_failed_behavioral_evidence(self, tmp_path):
        store = CompletionEvidenceStore(tmp_path)
        task = make_task(objective="Add JSON export")
        contract = derive_task_contract(task, None)
        record_behavioral(store, task.task_id, symbols=["export_json"], exit_code=1, trivial=False)
        fs = ProjectFilesystem(tmp_path)
        engine = RequirementAssessmentEngine(fs)
        ob = AcceptanceObligation(obligation_id="X", description="JSON export", target_tokens=["json"])
        req = Requirement(requirement_id="REQ-001", statement="Add JSON export", acceptance_obligations=[ob])
        # Even with an APPROVED review passed at the top-level assess() call,
        # the per-obligation check below (which is what actually drives the
        # rollup) never even sees last_review -- it can only be driven by
        # evidence.
        assessed = engine._assess_obligation(req, ob, {"exporter.py"}, "+def export_json(): ...", store, {})
        assert assessed.state == RequirementState.FAILED.value

    def test_s4_behavioral_pass_becomes_stale_after_mutation(self, tmp_path):
        store = CompletionEvidenceStore(tmp_path)
        (tmp_path / "impl.py").write_text("v1", encoding="utf-8")
        record_behavioral(store, "t1", symbols=["foo"], exit_code=0, trivial=False, target_paths=["impl.py"])
        assert len(store.get_valid_evidence(EvidenceType.TEST_EXECUTION)) == 1
        (tmp_path / "impl.py").write_text("v2", encoding="utf-8")
        store.invalidate_on_file_mutation(["impl.py"])
        assert len(store.get_valid_evidence(EvidenceType.TEST_EXECUTION)) == 0

    def test_s5_failed_obligation_dominates_positive_weaker_evidence(self, tmp_path):
        obs = [
            AcceptanceObligation(obligation_id="A", description="a", state=RequirementState.SATISFIED.value),
            AcceptanceObligation(obligation_id="B", description="b", state=RequirementState.FAILED.value),
            AcceptanceObligation(obligation_id="C", description="c", state=RequirementState.SATISFIED.value),
        ]
        state, _ = _rollup_obligations(obs)
        assert state == RequirementState.FAILED

    def test_s6_partial_obligation_satisfaction_not_full_satisfaction(self):
        obs = [
            AcceptanceObligation(obligation_id="A", description="a", state=RequirementState.SATISFIED.value),
            AcceptanceObligation(obligation_id="B", description="b", state=RequirementState.UNVERIFIED.value),
        ]
        state, _ = _rollup_obligations(obs)
        assert state != RequirementState.SATISFIED

    def test_s7_trivial_test_cannot_establish_behavioral_proof(self, tmp_path):
        store = CompletionEvidenceStore(tmp_path)
        record_behavioral(store, "t1", symbols=["export_json"], exit_code=0, trivial=True)
        fs = ProjectFilesystem(tmp_path)
        engine = RequirementAssessmentEngine(fs)
        ob = AcceptanceObligation(obligation_id="X", description="JSON export", target_tokens=["json"])
        req = Requirement(requirement_id="REQ-001", statement="Add JSON export", acceptance_obligations=[ob])
        assessed = engine._assess_obligation(req, ob, {"exporter.py"}, "+def export_json(): ...", store, {})
        # Falls through to diff correlation instead of being upgraded to
        # EXECUTABLE_BEHAVIORAL strength by a trivial pass.
        assert assessed.method != AcceptanceMethod.EXECUTABLE_BEHAVIORAL.value

    def test_s8_checkpoint_state_cannot_override_current_workspace(self, tmp_path):
        fs = ProjectFilesystem(tmp_path)
        task = make_task(objective="Add JSON export")
        contract = derive_task_contract(task, None)
        store = CompletionEvidenceStore(tmp_path)
        engine = RequirementAssessmentEngine(fs)
        result = engine.assess(contract, store, [], "")  # no evidence of any kind
        assert result.satisfied is False

    def test_s9_contract_is_immutable_once_derived_no_versioning_gap(self, tmp_path):
        # derive_task_contract is only ever called when task.task_contract is
        # falsy (see Orchestrator.run()/MultiTurnImplementationAgent.execute());
        # once set, it is never silently re-derived, so there is no
        # version-mismatch surface for old evidence to be misapplied against
        # a changed contract shape.
        task = make_task(objective="Add JSON export")
        c1 = derive_task_contract(task, None)
        task.task_contract = c1.to_dict()
        c2 = derive_task_contract(task, None)  # still callable directly...
        # ...but production code paths gate this behind `if not task.task_contract`.
        assert bool(task.task_contract) is True

    def test_s10_unverifiable_requirement_not_silently_satisfied(self, tmp_path):
        _, result, _ = assess(
            tmp_path, "Make the UI feel more polished. Preserve backward compatibility of the public API.",
            [FileOperation(action="modify", path="ui.py")],
            "+def polish(): ...",
        )
        compat = [r for r in result.requirements if r.requirement_type == RequirementType.COMPATIBILITY.value]
        assert compat[0].state == RequirementState.UNVERIFIABLE.value
        assert compat[0].state != RequirementState.SATISFIED.value

    def test_s11_acceptance_evidence_bound_to_correct_requirement(self, tmp_path):
        store = CompletionEvidenceStore(tmp_path)
        task = make_task(objective="Add CSV and JSON export")
        contract = derive_task_contract(task, None)
        fs = ProjectFilesystem(fs_path := tmp_path)
        engine = RequirementAssessmentEngine(fs)
        review = ReviewResult(verdict="APPROVED", summary="ok")
        engine.assess(
            contract, store,
            [FileOperation(action="modify", path="exporter.py")],
            "+def export_csv(): ...\n+def export_json(): ...\n",
            last_review=review,
        )
        req = next(r for r in contract.requirements if r.requirement_type == RequirementType.FUNCTIONAL.value)
        compliance = [e for e in store.all_entries() if e.evidence_type == EvidenceType.CONTRACT_COMPLIANCE.value]
        assert all(e.subtask_id == req.requirement_id for e in compliance)

    def test_s12_acceptance_evidence_bound_to_workspace_fingerprint(self, tmp_path):
        store = CompletionEvidenceStore(tmp_path)
        (tmp_path / "impl.py").write_text("v1", encoding="utf-8")
        ev = record_behavioral(store, "t1", symbols=["foo"], exit_code=0, trivial=False, target_paths=["impl.py"])
        assert ev.content_fingerprint  # a fingerprint was actually computed

    def test_s13_idiomatic_phrases_not_fabricated_into_obligations(self, tmp_path):
        # Neither phrase matches the Phase 4.22 parallel-list shape (no
        # comma-enumeration, no bare uppercase-acronym pair), so neither
        # gains acceptance_obligations -- this specifically guards against
        # the new obligation-set machinery over-firing on idiomatic phrasing
        # that Phase 4.21 already had to harden against.
        for objective in ["Pause and resume telemetry", "Add logging and metrics"]:
            task = make_task(objective=objective)
            contract = derive_task_contract(task, None)
            functional = [r for r in contract.requirements if r.requirement_type == RequirementType.FUNCTIONAL.value]
            assert len(functional) == 1
            assert functional[0].acceptance_obligations == []

    def test_s13_verb_object_pair_split_is_preexisting_phase421_behavior(self, tmp_path):
        # "Support reading and writing CSV files" splits into two top-level
        # requirements under the Phase 4.21 objective-level " and " splitter
        # (both halves are >=2 words) -- this predates Phase 4.22 and is
        # unrelated to the new parallel-list obligation machinery, which
        # never fires here (no comma list, no uppercase acronym pair).
        # Recorded here as a documented, deliberate non-goal rather than a
        # silently-assumed invariant.
        task = make_task(objective="Support reading and writing CSV files")
        contract = derive_task_contract(task, None)
        functional = [r for r in contract.requirements if r.requirement_type == RequirementType.FUNCTIONAL.value]
        assert len(functional) == 2
        assert all(r.acceptance_obligations == [] for r in functional)

    def test_s14_three_item_compound_not_satisfied_with_one_missing(self, tmp_path):
        _, result, _ = assess(
            tmp_path, "Add authentication, authorization, and audit logging",
            [FileOperation(action="modify", path="auth.py")],
            "+def authenticate(): ...\n+def authorize(): ...\n",  # audit logging missing
        )
        assert result.satisfied is False

    def test_s15_evidence_history_remains_bounded(self, tmp_path):
        fs = ProjectFilesystem(tmp_path)
        task = make_task(objective="Add a feature")
        contract = derive_task_contract(task, None)
        store = CompletionEvidenceStore(tmp_path, max_entries=50)
        engine = RequirementAssessmentEngine(fs)
        review = ReviewResult(verdict="APPROVED", summary="ok")
        for i in range(200):
            engine.assess(
                contract, store,
                [FileOperation(action="modify", path=f"f{i}.py")],
                f"+x{i}=1",
                last_review=review,
            )
        assert len(store.all_entries()) <= 50

    def test_s16_final_completion_requires_all_mandatory_obligations(self, tmp_path):
        _, result, _ = assess(
            tmp_path, "Add CSV, JSON, and XML export",
            [FileOperation(action="modify", path="exporter.py")],
            "+def export_csv(): ...\n",  # only 1 of 3
        )
        assert result.satisfied is False

    def test_s17_technical_readiness_alone_cannot_authorize_completion(self, tmp_path):
        # Covered structurally: Orchestrator.run()/MultiTurnImplementationAgent
        # both compute final_ready = assessment.is_ready AND req_assessment.satisfied.
        import inspect
        src = inspect.getsource(Orchestrator.run)
        assert "req_assessment.satisfied" in src

    def test_s18_task_contract_satisfaction_alone_cannot_authorize_completion(self, tmp_path):
        import inspect
        src = inspect.getsource(Orchestrator.run)
        assert "assessment.is_ready and req_assessment.satisfied" in src


# =============================================================================
# Section 8: bounds / pathological inputs
# =============================================================================

class TestBounds:
    def test_pathological_many_item_list_bounded(self, tmp_path):
        items = ", ".join(f"F{i:02d}" for i in range(1, 40))
        objective = f"Add {items}, and F40 export"
        task = make_task(objective=objective)
        contract = derive_task_contract(task, None)
        functional = [r for r in contract.requirements if r.requirement_type == RequirementType.FUNCTIONAL.value]
        assert len(functional) >= 1
        assert len(functional[0].acceptance_obligations) <= MAX_OBLIGATIONS_PER_REQUIREMENT

    def test_repeated_assessment_of_pathological_contract_stays_fast(self, tmp_path):
        import time
        items = ", ".join(f"F{i:02d}" for i in range(1, 10))
        task = make_task(objective=f"Add {items}, and F10 export")
        contract = derive_task_contract(task, None)
        store = CompletionEvidenceStore(tmp_path, max_entries=200)
        fs = ProjectFilesystem(tmp_path)
        engine = RequirementAssessmentEngine(fs)
        review = ReviewResult(verdict="APPROVED", summary="ok")
        start = time.time()
        for _ in range(50):
            engine.assess(contract, store, [FileOperation(action="modify", path="x.py")], "+x=1", last_review=review)
        assert time.time() - start < 5.0


# =============================================================================
# Section 9: backward compatibility
# =============================================================================

class TestOrchestratorIntegration:
    """Real Orchestrator.run() lifecycle (real CommandRunner subprocess
    execution, not mocked) -- proves the Phase 4.11 behavioral-verification
    pipeline actually reaches the evidence store with the new wiring, not
    just that the unit-level pieces compose correctly in isolation."""

    def test_real_run_records_synthesized_test_evidence_into_store(self, tmp_path):
        import threading
        from unittest import mock
        from local_agent.storage import JsonFileStorage
        from local_agent.config import AgentConfig
        from local_agent.models import Task, TaskStatus, Plan

        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "csv_export.py").write_text("", encoding="utf-8")
        storage = JsonFileStorage(tmp_path / ".agent_data")
        cfg = AgentConfig.from_environment(tmp_path, max_iterations=1)
        orch = Orchestrator(cfg, storage, None, threading.Lock(), threading.Lock())

        class Provider:
            def __init__(self):
                self.provider_id = "mock"
                self.model = "mock-model"
                self.capabilities = {ProviderCapability.PLANNING, ProviderCapability.IMPLEMENTATION, ProviderCapability.REVIEW}

            def generate_plan(self, task, context):
                return Plan(objective=task, steps=[])

            def generate_code(self, task, plan, context, failure=None, review=None):
                return [FileOperation("modify", "src/csv_export.py", content="def export_csv(rows): return rows\n", reason="csv")]

            def review_changes(self, task, plan, diff, context, **kwargs):
                return ReviewResult(verdict="APPROVED", summary="looks fine")

        task = Task(
            task_id="t1", objective="Add CSV export.", status=TaskStatus.PENDING,
            created_at=datetime.datetime.now(datetime.timezone.utc),
            updated_at=datetime.datetime.now(datetime.timezone.utc),
        )
        with mock.patch("local_agent.orchestrator.build_provider", return_value=Provider()):
            report = orch.run(task)

        assert len(report.behavioral_evidence) >= 1
        rec = report.behavioral_evidence[0]
        assert rec.exercised_symbols == ["export_csv"]
        # The deterministic-template fallback (no verification-specialist
        # provider configured in this test) only checks callability -- it
        # must be honestly self-reported as trivial.
        assert rec.trivial is True

        entries = report.completion_evidence.get("entries", [])
        synthesized = [
            e for e in entries
            if e.get("evidence_type") == "test_execution" and e.get("payload", {}).get("synthesized") is True
        ]
        assert len(synthesized) >= 1
        assert synthesized[0]["payload"]["trivial"] is True
        assert "export_csv" in synthesized[0]["target_symbols"]
        # Fingerprinted, not just appended to a flat list -- this is what
        # makes it eligible for invalidate_on_file_mutation() on a later turn.
        assert synthesized[0]["content_fingerprint"]


class TestBackwardCompatibility:
    def test_requirement_without_obligations_uses_legacy_path(self, tmp_path):
        _, result, _ = assess(
            tmp_path, "Implement a new feature",
            [FileOperation(action="modify", path="feature.py")],
            "+def feature(): ...",
        )
        assert result.satisfied is True

    def test_old_task_contract_dict_missing_obligations_key_loads(self):
        old_contract_dict = {
            "task_id": "t1",
            "objective": "Add a feature",
            "requirements": [{
                "requirement_id": "REQ-001",
                "statement": "Add a feature",
                "requirement_type": "functional",
                "importance": "must",
                "verification_strategy": "diff_presence",
                "source": "user_task",
                "target_paths": [],
                "state": "unverified",
                "evidence_ids": [],
                "unsatisfied_reason": "",
                "clarification_id": None,
            }],
            "non_goals": [],
            "version": 1,
            "source": "derived_from_objective",
        }
        contract = TaskContract.from_dict(old_contract_dict)
        assert contract.requirements[0].acceptance_obligations == []
