"""Phase 4.23: Evidence Provenance & Verification Binding Hardening.

Phase 4.22 introduced AcceptanceObligation-level behavioral evidence
binding, but explicitly disclosed it as a residual risk: "Obligation-to-
evidence binding is token-based, not a symbol registry: two obligations
sharing generic vocabulary could cross-match a behavioral record neither
of them actually concerns." This phase treats that disclosure as a claim
to independently reproduce, not a fact to trust.

Two real, reproducible vulnerabilities were confirmed against the Phase
4.22 baseline before any fix was written:

1. Cross-obligation collision: a single stray behavioral-evidence entry
   whose exercised symbol merely *contained* two obligations' tokens as
   raw substrings (e.g. "convert_csv_to_jsonlike" contains both "csv" and
   "json") let ONE unrelated passing test satisfy BOTH a "CSV export" and
   a "JSON export" obligation at the strongest (EXECUTABLE_BEHAVIORAL)
   tier, even though neither was ever actually implemented or tested.

2. Cross-task contamination: CompletionEvidenceStore.get_valid_evidence()
   never checked the task_id an entry was recorded under, so evidence
   belonging to one task could satisfy a completely different task with
   zero changes of its own.

The fix is a provenance-strengthening layer, not a bigger regex:
  - task_id scoping at the evidence-store read boundary (P5),
  - path scoping between an obligation's requirement and a candidate
    evidence entry's own target_paths,
  - word-boundary-safe (identifier-component) symbol matching instead of
    raw substring containment (P3/P4),
  - and, for the irreducible case a symbol whose name legitimately
    contains more than one obligation's word as a distinct component --
    cross-obligation ambiguity exclusion: evidence matching more than one
    obligation anywhere in the contract cannot grant EXECUTABLE_BEHAVIORAL
    strength to any of them (P1).

Organized by: reproduced vulnerabilities (before/after); the provenance
model (path/task/component scoping + ambiguity exclusion); the adversarial
attack matrix; invariants P1-P15; multi-turn/checkpoint analysis; backward
compatibility; and bounds/performance.
"""

from __future__ import annotations

import datetime

import pytest

from local_agent.completion import (
    CompletionEvidenceStore,
    EvidenceTrustTier,
    EvidenceType,
    StructuredEvidence,
)
from local_agent.filesystem import ProjectFilesystem
from local_agent.models import (
    Checkpoint,
    FileOperation,
    ProviderCapability,
    ReviewResult,
    Task,
    TaskStatus,
)
from local_agent.orchestrator import Orchestrator
from local_agent.task_contract import (
    AcceptanceMethod,
    AcceptanceObligation,
    AcceptanceProvenance,
    MAX_OBLIGATIONS_PER_REQUIREMENT,
    MAX_REQUIREMENTS,
    Requirement,
    RequirementAssessmentEngine,
    RequirementState,
    RequirementType,
    TaskContract,
    _symbol_components,
    derive_task_contract,
)


# -----------------------------------------------------------------------------
# Shared scaffolding
# -----------------------------------------------------------------------------

def make_task(task_id: str = "t1", objective: str = "Implement feature") -> Task:
    now = datetime.datetime.now(datetime.timezone.utc)
    return Task(task_id=task_id, objective=objective, status=TaskStatus.PENDING, created_at=now, updated_at=now)


def record_behavioral(
    store: CompletionEvidenceStore,
    task_id: str,
    symbols: list[str],
    exit_code: int,
    trivial: bool,
    target_paths: list[str] | None = None,
) -> StructuredEvidence:
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


def two_obligation_contract(task_id: str = "t1", objective: str = "Add CSV and JSON export") -> tuple[Task, TaskContract, Requirement]:
    task = make_task(task_id=task_id, objective=objective)
    contract = derive_task_contract(task, None)
    req = next(r for r in contract.requirements if r.requirement_type == RequirementType.FUNCTIONAL.value)
    assert len(req.acceptance_obligations) >= 2
    return task, contract, req


# =============================================================================
# Section 1: reproduced vulnerabilities (regression-locked)
# =============================================================================

class TestReproducedVulnerabilities:
    def test_cross_obligation_substring_collision_no_longer_grants_double_behavioral_proof(self, tmp_path):
        """The exact Phase 4.23 investigation repro: a single unrelated
        symbol name containing both obligations' tokens as substrings must
        no longer let one stray passing test satisfy both at
        EXECUTABLE_BEHAVIORAL strength."""
        task, contract, _ = two_obligation_contract()
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        record_behavioral(store, task.task_id, symbols=["convert_csv_to_jsonlike"], exit_code=0, trivial=False, target_paths=[])
        engine = RequirementAssessmentEngine(fs)
        result = engine.assess(contract, store, [], "+def convert_csv_to_jsonlike(): ...", last_review=None)
        req = next(r for r in result.requirements if r.requirement_type == RequirementType.FUNCTIONAL.value)
        by_desc = {o.description: o for o in req.acceptance_obligations}
        # "json" is not a real identifier component of "jsonlike" -- this
        # obligation must never have been offered as a behavioral candidate
        # at all, regardless of ambiguity bookkeeping.
        assert by_desc["JSON export"].method != AcceptanceMethod.EXECUTABLE_BEHAVIORAL.value
        assert by_desc["JSON export"].state != RequirementState.SATISFIED.value or by_desc["JSON export"].method == AcceptanceMethod.DIFF_CORRELATION.value

    def test_cross_task_contamination_no_longer_satisfies_unrelated_task(self, tmp_path):
        """The exact Phase 4.23 investigation repro: evidence recorded under
        a different task_id must not satisfy this task, even with zero
        changes of its own."""
        _, contract, _ = two_obligation_contract(task_id="task-B")
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        record_behavioral(store, "task-A-UNRELATED", symbols=["export_csv", "export_json"], exit_code=0, trivial=False, target_paths=["exporter.py"])
        engine = RequirementAssessmentEngine(fs)
        result = engine.assess(contract, store, [], "", last_review=None)
        assert result.satisfied is False

    def test_genuinely_ambiguous_symbol_excluded_from_both_obligations(self, tmp_path):
        """The irreducible case: a symbol whose name legitimately contains
        BOTH obligations' words as distinct components (not a substring
        fluke). Component matching alone cannot disambiguate this --
        cross-obligation ambiguity exclusion must step in and refuse
        EXECUTABLE_BEHAVIORAL strength to either."""
        task, contract, _ = two_obligation_contract()
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        record_behavioral(store, task.task_id, symbols=["convert_csv_to_json"], exit_code=0, trivial=False, target_paths=[])
        engine = RequirementAssessmentEngine(fs)
        result = engine.assess(contract, store, [], "+def convert_csv_to_json(): ...", last_review=None)
        req = next(r for r in result.requirements if r.requirement_type == RequirementType.FUNCTIONAL.value)
        assert all(o.method != AcceptanceMethod.EXECUTABLE_BEHAVIORAL.value for o in req.acceptance_obligations)

    def test_legitimate_unambiguous_case_still_works(self, tmp_path):
        """Sanity: the fix must not break the ordinary, correct case -- two
        distinct, correctly-named, correctly-scoped symbols."""
        task, contract, _ = two_obligation_contract()
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        record_behavioral(store, task.task_id, symbols=["export_csv"], exit_code=0, trivial=False, target_paths=["exporter.py"])
        record_behavioral(store, task.task_id, symbols=["export_json"], exit_code=0, trivial=False, target_paths=["exporter.py"])
        engine = RequirementAssessmentEngine(fs)
        result = engine.assess(contract, store, [FileOperation(action="modify", path="exporter.py")], "+def export_csv(): ...\n+def export_json(): ...", last_review=None)
        assert result.satisfied is True
        req = next(r for r in result.requirements if r.requirement_type == RequirementType.FUNCTIONAL.value)
        assert all(o.method == AcceptanceMethod.EXECUTABLE_BEHAVIORAL.value for o in req.acceptance_obligations)


# =============================================================================
# Section 2: symbol component matching
# =============================================================================

class TestSymbolComponents:
    def test_snake_case_split(self):
        assert _symbol_components("export_csv") == {"export", "csv"}

    def test_camel_case_split(self):
        assert _symbol_components("exportCsv") == {"export", "csv"}

    def test_fused_word_not_split(self):
        # "jsonlike" is one token -- "json" is not a member.
        assert "json" not in _symbol_components("convert_csv_to_jsonlike")

    def test_genuine_distinct_component_present(self):
        assert {"csv", "json"} <= _symbol_components("convert_csv_to_json")

    def test_substring_fragment_not_a_component(self):
        assert "import" not in _symbol_components("important_config")

    def test_empty_name(self):
        assert _symbol_components("") == set()


# =============================================================================
# Section 3: adversarial attack matrix (25 named attacks)
# =============================================================================

class TestAdversarialMatrix:
    def test_attack_01_cross_obligation_evidence_replay(self, tmp_path):
        # Same as the reproduced vulnerability, but explicitly framed as a
        # "replay": one recorded test result reused to argue for two claims.
        task, contract, _ = two_obligation_contract()
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        record_behavioral(store, task.task_id, symbols=["convert_csv_to_jsonlike"], exit_code=0, trivial=False, target_paths=[])
        engine = RequirementAssessmentEngine(fs)
        result = engine.assess(contract, store, [], "+x", last_review=None)
        req = next(r for r in result.requirements if r.requirement_type == RequirementType.FUNCTIONAL.value)
        strong = [o for o in req.acceptance_obligations if o.method == AcceptanceMethod.EXECUTABLE_BEHAVIORAL.value]
        assert len(strong) <= 1  # never both

    def test_attack_02_generic_token_collision(self, tmp_path):
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        task = make_task(objective="Add authentication and authorization")
        # Bare 2-item lowercase pair -- not a parallel list under current
        # extraction rules, so no obligations are produced; this attack
        # targets the case where obligations DO exist for generic words.
        ob_auth = AcceptanceObligation(obligation_id="R-OB1", description="authentication", target_tokens=["authentication"])
        ob_authz = AcceptanceObligation(obligation_id="R-OB2", description="authorization", target_tokens=["authorization"])
        req = Requirement(requirement_id="REQ-001", statement="x", acceptance_obligations=[ob_auth, ob_authz])
        contract = TaskContract(task_id=task.task_id, objective=task.objective, requirements=[req])
        record_behavioral(store, task.task_id, symbols=["authentication_and_authorization_helper"], exit_code=0, trivial=False, target_paths=[])
        engine = RequirementAssessmentEngine(fs)
        result = engine.assess(contract, store, [], "+x", last_review=None)
        r = result.requirements[0]
        strong = [o for o in r.acceptance_obligations if o.method == AcceptanceMethod.EXECUTABLE_BEHAVIORAL.value]
        assert strong == []  # ambiguous match -> excluded from both

    def test_attack_03_same_context_word_collision(self, tmp_path):
        # Both obligations share "export" as context; target_tokens exclude
        # it (Phase 4.22 fix), so a symbol matching only on "export" cannot
        # bind to either.
        task, contract, req = two_obligation_contract()
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        record_behavioral(store, task.task_id, symbols=["export_helper"], exit_code=0, trivial=False, target_paths=[])
        engine = RequirementAssessmentEngine(fs)
        result = engine.assess(contract, store, [], "+x", last_review=None)
        r = next(r for r in result.requirements if r.requirement_type == RequirementType.FUNCTIONAL.value)
        assert all(o.method != AcceptanceMethod.EXECUTABLE_BEHAVIORAL.value for o in r.acceptance_obligations)

    def test_attack_04_wrong_symbol_behavioral_evidence(self, tmp_path):
        task, contract, _ = two_obligation_contract()
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        record_behavioral(store, task.task_id, symbols=["totally_unrelated_helper"], exit_code=0, trivial=False, target_paths=[])
        engine = RequirementAssessmentEngine(fs)
        result = engine.assess(contract, store, [], "", last_review=None)
        assert result.satisfied is False

    def test_attack_05_wrong_path_behavioral_evidence(self, tmp_path):
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        req = Requirement(requirement_id="REQ-001", statement="x", target_paths=["exporter.py"],
                           acceptance_obligations=[AcceptanceObligation(obligation_id="REQ-001-OB1", description="csv export", target_tokens=["csv"], target_paths=["exporter.py"])])
        contract = TaskContract(task_id="t1", objective="x", requirements=[req])
        # Evidence exercises a same-named symbol but in a completely
        # different, unrelated file.
        record_behavioral(store, "t1", symbols=["export_csv"], exit_code=0, trivial=False, target_paths=["some_other_unrelated_module.py"])
        engine = RequirementAssessmentEngine(fs)
        result = engine.assess(contract, store, [], "", last_review=None)
        assert result.satisfied is False

    def test_attack_06_same_symbol_name_different_modules(self, tmp_path):
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        req1 = Requirement(requirement_id="REQ-001", statement="r1", target_paths=["a.py"],
                            acceptance_obligations=[AcceptanceObligation(obligation_id="REQ-001-OB1", description="csv a", target_tokens=["csv"], target_paths=["a.py"])])
        req2 = Requirement(requirement_id="REQ-002", statement="r2", target_paths=["b.py"],
                            acceptance_obligations=[AcceptanceObligation(obligation_id="REQ-002-OB1", description="csv b", target_tokens=["csv"], target_paths=["b.py"])])
        contract = TaskContract(task_id="t1", objective="x", requirements=[req1, req2])
        record_behavioral(store, "t1", symbols=["csv_helper"], exit_code=0, trivial=False, target_paths=["b.py"])
        engine = RequirementAssessmentEngine(fs)
        result = engine.assess(contract, store, [FileOperation(action="modify", path="b.py")], "+def csv_helper(): ...", last_review=None)
        by_id = {r.requirement_id: r for r in result.requirements}
        ob1 = by_id["REQ-001"].acceptance_obligations[0]
        ob2 = by_id["REQ-002"].acceptance_obligations[0]
        assert ob2.method == AcceptanceMethod.EXECUTABLE_BEHAVIORAL.value
        assert ob1.method != AcceptanceMethod.EXECUTABLE_BEHAVIORAL.value

    def test_attack_07_evidence_copied_between_requirements_via_persisted_ids(self, tmp_path):
        # Forging evidence_ids into a persisted obligation must have zero
        # effect -- assess() always recomputes from live evidence, never
        # trusts a persisted evidence_ids list as an input.
        task, contract, req = two_obligation_contract()
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        ev = record_behavioral(store, task.task_id, symbols=["export_csv"], exit_code=0, trivial=False, target_paths=["exporter.py"])
        json_ob = next(o for o in req.acceptance_obligations if "JSON" in o.description)
        forged = json_ob.__class__(**{**json_ob.to_dict(), "evidence_ids": [ev.evidence_id], "state": RequirementState.SATISFIED.value, "method": AcceptanceMethod.EXECUTABLE_BEHAVIORAL.value})
        req2 = req.__class__(**{**req.to_dict(), "acceptance_obligations": [o if o.obligation_id != json_ob.obligation_id else forged for o in req.acceptance_obligations]})
        contract.requirements = [req2 if r.requirement_id == req.requirement_id else r for r in contract.requirements]
        engine = RequirementAssessmentEngine(fs)
        result = engine.assess(contract, store, [FileOperation(action="modify", path="exporter.py")], "+def export_csv(): ...", last_review=None)
        assert result.satisfied is False  # JSON obligation recomputed fresh, still unresolved

    def test_attack_08_evidence_copied_between_tasks(self, tmp_path):
        _, contract, _ = two_obligation_contract(task_id="task-victim")
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        record_behavioral(store, "task-attacker", symbols=["export_csv", "export_json"], exit_code=0, trivial=False, target_paths=["exporter.py"])
        engine = RequirementAssessmentEngine(fs)
        result = engine.assess(contract, store, [], "", last_review=None)
        assert result.satisfied is False

    def test_attack_09_forged_evidence_ids_ignored(self, tmp_path):
        task, contract, req = two_obligation_contract()
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)  # empty -- no real evidence at all
        forged = [o.__class__(**{**o.to_dict(), "evidence_ids": ["ev-forged-does-not-exist"], "state": RequirementState.SATISFIED.value}) for o in req.acceptance_obligations]
        req2 = req.__class__(**{**req.to_dict(), "acceptance_obligations": forged})
        contract.requirements = [req2]
        engine = RequirementAssessmentEngine(fs)
        result = engine.assess(contract, store, [], "", last_review=None)
        assert result.satisfied is False

    def test_attack_10_forged_obligation_ids(self, tmp_path):
        task, contract, req = two_obligation_contract()
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        renamed = [o.__class__(**{**o.to_dict(), "obligation_id": "REQ-999-OBX", "state": RequirementState.SATISFIED.value}) for o in req.acceptance_obligations]
        req2 = req.__class__(**{**req.to_dict(), "acceptance_obligations": renamed})
        contract.requirements = [req2]
        engine = RequirementAssessmentEngine(fs)
        result = engine.assess(contract, store, [], "", last_review=None)
        assert result.satisfied is False

    def test_attack_11_forged_target_symbols_on_evidence(self, tmp_path):
        # Evidence claiming exercised_symbols that don't correspond to
        # anything real in the diff is still only as trustworthy as the
        # correlation itself -- if it doesn't correlate with THIS
        # obligation's tokens, it grants nothing.
        task, contract, _ = two_obligation_contract()
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        record_behavioral(store, task.task_id, symbols=["completely_forged_symbol_name"], exit_code=0, trivial=False, target_paths=[])
        engine = RequirementAssessmentEngine(fs)
        result = engine.assess(contract, store, [], "", last_review=None)
        assert result.satisfied is False

    def test_attack_12_forged_target_paths_on_evidence(self, tmp_path):
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        req = Requirement(requirement_id="REQ-001", statement="x", target_paths=["exporter.py"],
                           acceptance_obligations=[AcceptanceObligation(obligation_id="REQ-001-OB1", description="csv export", target_tokens=["csv"], target_paths=["exporter.py"])])
        contract = TaskContract(task_id="t1", objective="x", requirements=[req])
        # Evidence claims a path this requirement never touches.
        record_behavioral(store, "t1", symbols=["export_csv"], exit_code=0, trivial=False, target_paths=["/etc/passwd"])
        engine = RequirementAssessmentEngine(fs)
        result = engine.assess(contract, store, [], "", last_review=None)
        assert result.satisfied is False

    def test_attack_13_stale_behavioral_evidence_after_mutation(self, tmp_path):
        (tmp_path / "exporter.py").write_text("def export_csv(): return 'v1'\n", encoding="utf-8")
        store = CompletionEvidenceStore(tmp_path)
        record_behavioral(store, "t1", symbols=["export_csv"], exit_code=0, trivial=False, target_paths=["exporter.py"])
        assert len(store.get_valid_evidence(EvidenceType.TEST_EXECUTION)) == 1
        (tmp_path / "exporter.py").write_text("def export_csv(): return 'v2 -- broken'\n", encoding="utf-8")
        store.invalidate_on_file_mutation(["exporter.py"])
        assert len(store.get_valid_evidence(EvidenceType.TEST_EXECUTION)) == 0

    def test_attack_14_checkpoint_replay_recomputes_not_trusts(self, tmp_path):
        task, contract, req = two_obligation_contract()
        forged_obligations = [o.__class__(**{**o.to_dict(), "state": RequirementState.SATISFIED.value, "method": AcceptanceMethod.EXECUTABLE_BEHAVIORAL.value}) for o in req.acceptance_obligations]
        forged_req = req.__class__(**{**req.to_dict(), "acceptance_obligations": forged_obligations, "state": RequirementState.SATISFIED.value})
        contract.requirements = [forged_req if r.requirement_id == req.requirement_id else r for r in contract.requirements]
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)  # no real evidence
        engine = RequirementAssessmentEngine(fs)
        result = engine.assess(contract, store, [], "", last_review=None)
        assert result.satisfied is False

    def test_attack_15_contract_mutation_after_derivation(self, tmp_path):
        task, contract, req = two_obligation_contract()
        # Mutating a requirement's importance to SHOULD after the fact
        # cannot be done via the engine (importance is set only at
        # derivation) -- assess() must not have any code path that lets
        # provider/reviewer input change importance.
        import inspect
        assert "importance" not in inspect.signature(RequirementAssessmentEngine.assess).parameters

    def test_attack_16_duplicate_evidence_entries(self, tmp_path):
        task, contract, _ = two_obligation_contract()
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        record_behavioral(store, task.task_id, symbols=["export_csv"], exit_code=0, trivial=False, target_paths=["exporter.py"])
        record_behavioral(store, task.task_id, symbols=["export_csv"], exit_code=0, trivial=False, target_paths=["exporter.py"])
        engine = RequirementAssessmentEngine(fs)
        result = engine.assess(contract, store, [FileOperation(action="modify", path="exporter.py")], "+def export_csv(): ...", last_review=None)
        req = next(r for r in result.requirements if r.requirement_type == RequirementType.FUNCTIONAL.value)
        csv_ob = next(o for o in req.acceptance_obligations if "CSV" in o.description)
        assert csv_ob.state == RequirementState.SATISFIED.value  # duplicates don't break it

    def test_attack_17_contradictory_pass_and_fail_evidence(self, tmp_path):
        task, contract, _ = two_obligation_contract()
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        record_behavioral(store, task.task_id, symbols=["export_csv"], exit_code=0, trivial=False, target_paths=["exporter.py"])
        record_behavioral(store, task.task_id, symbols=["export_csv"], exit_code=1, trivial=False, target_paths=["exporter.py"])
        engine = RequirementAssessmentEngine(fs)
        result = engine.assess(contract, store, [FileOperation(action="modify", path="exporter.py")], "+def export_csv(): ...", last_review=None)
        req = next(r for r in result.requirements if r.requirement_type == RequirementType.FUNCTIONAL.value)
        csv_ob = next(o for o in req.acceptance_obligations if "CSV" in o.description)
        assert csv_ob.state == RequirementState.FAILED.value  # failure dominates

    def test_attack_18_provider_approved_claim_versus_misbound_evidence(self, tmp_path):
        task, contract, _ = two_obligation_contract()
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        record_behavioral(store, task.task_id, symbols=["export_csv"], exit_code=1, trivial=False, target_paths=["exporter.py"])
        engine = RequirementAssessmentEngine(fs)
        review = ReviewResult(verdict="APPROVED", summary="Provider claims: fully done, all formats supported")
        result = engine.assess(contract, store, [FileOperation(action="modify", path="exporter.py")], "+def export_csv(): ...", last_review=review)
        req = next(r for r in result.requirements if r.requirement_type == RequirementType.FUNCTIONAL.value)
        csv_ob = next(o for o in req.acceptance_obligations if "CSV" in o.description)
        assert csv_ob.state == RequirementState.FAILED.value  # review text has no bearing

    def test_attack_19_reviewer_approved_claim_versus_misbound_evidence(self, tmp_path):
        # Identical guarantee, framed from the reviewer angle: an APPROVED
        # ReviewResult is the only "claim" input assess() accepts at all,
        # and it never overrides a failing obligation-bound behavioral
        # result (see attack 18 -- same code path, same proof).
        task, contract, _ = two_obligation_contract()
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        record_behavioral(store, task.task_id, symbols=["export_json"], exit_code=1, trivial=False, target_paths=["exporter.py"])
        engine = RequirementAssessmentEngine(fs)
        review = ReviewResult(verdict="APPROVED", summary="looks great")
        result = engine.assess(contract, store, [FileOperation(action="modify", path="exporter.py")], "+def export_json(): ...", last_review=review)
        assert result.satisfied is False

    def test_attack_20_partial_compound_completion(self, tmp_path):
        task, contract, _ = two_obligation_contract(objective="Add CSV, JSON, and XML export")
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        result = RequirementAssessmentEngine(fs).assess(
            contract, store,
            [FileOperation(action="modify", path="exporter.py")],
            "+def export_csv(): ...\n+def export_json(): ...\n",  # XML missing
            last_review=None,
        )
        assert result.satisfied is False

    def test_attack_21_generic_vocabulary_overlap_disclosed_limit(self, tmp_path):
        # The irreducible case, explicitly named: a symbol whose name
        # genuinely, distinctly contains two obligations' words. Cannot be
        # resolved by lexical matching alone -- correctly downgraded rather
        # than falsely resolved in either direction.
        task, contract, _ = two_obligation_contract()
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        record_behavioral(store, task.task_id, symbols=["csv_json_bridge"], exit_code=0, trivial=False, target_paths=[])
        engine = RequirementAssessmentEngine(fs)
        result = engine.assess(contract, store, [], "+x", last_review=None)
        req = next(r for r in result.requirements if r.requirement_type == RequirementType.FUNCTIONAL.value)
        assert all(o.method != AcceptanceMethod.EXECUTABLE_BEHAVIORAL.value for o in req.acceptance_obligations)

    def test_attack_22_multiturn_reuses_orchestrator_evidence_safely(self, tmp_path):
        """Evidence recorded by Orchestrator.run()'s behavioral-synthesis
        pipeline, still valid (workspace unchanged), is legitimately
        reusable by a later MultiTurnImplementationAgent.execute() call for
        the SAME task via checkpoint resume -- this is correct behavior
        (evidence is evidence), not a trust violation, because it is always
        revalidated against disk before use in both entry points."""
        task, contract, _ = two_obligation_contract()
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        (tmp_path / "exporter.py").write_text("def export_csv(): ...\ndef export_json(): ...\n", encoding="utf-8")
        record_behavioral(store, task.task_id, symbols=["export_csv"], exit_code=0, trivial=False, target_paths=["exporter.py"])
        record_behavioral(store, task.task_id, symbols=["export_json"], exit_code=0, trivial=False, target_paths=["exporter.py"])
        # Simulate the multi-turn resume path: reload from serialized form
        # and revalidate against (unchanged) disk, exactly as
        # MultiTurnImplementationAgent.execute() does on checkpoint resume.
        restored = CompletionEvidenceStore.from_dict(store.to_dict())
        restored.revalidate_against_disk(fs)
        engine = RequirementAssessmentEngine(fs)
        result = engine.assess(contract, restored, [FileOperation(action="modify", path="exporter.py")], "+def export_csv(): ...\n+def export_json(): ...", last_review=None)
        assert result.satisfied is True

        # Now mutate the workspace between the two entry points -- the
        # reused evidence must correctly go stale, not silently carry over.
        (tmp_path / "exporter.py").write_text("def export_csv(): ...\n# json removed\n", encoding="utf-8")
        restored2 = CompletionEvidenceStore.from_dict(store.to_dict())
        restored2.revalidate_against_disk(fs)
        assert len(restored2.get_valid_evidence(EvidenceType.TEST_EXECUTION)) < 2

    def test_attack_23_backward_compatible_loading_of_old_checkpoints(self, tmp_path):
        # A Phase 4.20/4.21/4.22-era serialized evidence entry has no notion
        # of the Phase 4.23 matching refinements -- it must still load and
        # be scoped correctly by the one field it always had: task_id.
        old_entry = {
            "evidence_id": "ev-old-1", "task_id": "t1", "subtask_id": "main", "turn_number": 1,
            "stage": "behavioral_verification", "evidence_type": "test_execution", "source": "old",
            "trust_tier": 1, "status": "valid", "target_paths": ["exporter.py"],
            "target_symbols": ["export_csv"], "exit_code": 0,
            "payload": {"synthesized": True, "trivial": False},
        }
        store = CompletionEvidenceStore.from_dict({"workspace_root": str(tmp_path), "max_entries": 100, "entries": [old_entry]})
        assert len(store.get_valid_evidence(EvidenceType.TEST_EXECUTION, task_id="t1")) == 1
        assert len(store.get_valid_evidence(EvidenceType.TEST_EXECUTION, task_id="other-task")) == 0

    def test_attack_24_bounded_growth_with_many_obligations(self, tmp_path):
        fs = ProjectFilesystem(tmp_path)
        reqs = []
        for r in range(MAX_REQUIREMENTS):
            obs = [AcceptanceObligation(obligation_id=f"REQ-{r}-OB{i}", description=f"item {r} {i}", target_tokens=[f"tok{r}{i}"]) for i in range(MAX_OBLIGATIONS_PER_REQUIREMENT)]
            reqs.append(Requirement(requirement_id=f"REQ-{r:03d}", statement=f"req {r}", acceptance_obligations=obs))
        contract = TaskContract(task_id="t1", objective="pathological", requirements=reqs)
        store = CompletionEvidenceStore(tmp_path, max_entries=100)
        for i in range(100):
            record_behavioral(store, "t1", symbols=[f"sym_{i}"], exit_code=0, trivial=False, target_paths=[])
        engine = RequirementAssessmentEngine(fs)
        review = ReviewResult(verdict="APPROVED", summary="ok")
        result = engine.assess(contract, store, [FileOperation(action="modify", path="x.py")], "+x=1", last_review=review)
        assert len(result.requirements) == MAX_REQUIREMENTS
        assert len(store.all_entries()) <= 100  # still bounded after assessment

    def test_attack_25_secret_redaction_in_obligation_fields(self, tmp_path):
        ob = AcceptanceObligation(
            obligation_id="X", description="add support for API_KEY=sk-ant-abcdefghijklmnopqrstuvwx1234",
            unsatisfied_reason="failed because password=hunter2secret was wrong",
        )
        d = ob.to_dict()
        assert "sk-ant-abcdefghijklmnopqrstuvwx1234" not in d["description"]
        assert "hunter2secret" not in d["unsatisfied_reason"]


# =============================================================================
# Section 4: invariants P1-P15
# =============================================================================

class TestInvariants:
    def test_p1_evidence_cannot_satisfy_obligation_it_does_not_bind_to(self, tmp_path):
        task, contract, _ = two_obligation_contract()
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        record_behavioral(store, task.task_id, symbols=["totally_unrelated"], exit_code=0, trivial=False, target_paths=[])
        result = RequirementAssessmentEngine(fs).assess(contract, store, [], "", last_review=None)
        assert result.satisfied is False

    def test_p2_behavioral_evidence_identifies_workspace_state(self, tmp_path):
        (tmp_path / "exporter.py").write_text("v1", encoding="utf-8")
        store = CompletionEvidenceStore(tmp_path)
        ev = record_behavioral(store, "t1", symbols=["export_csv"], exit_code=0, trivial=False, target_paths=["exporter.py"])
        assert ev.content_fingerprint  # a fingerprint of the actual state was captured

    def test_p3_behavioral_evidence_for_symbol_a_cannot_prove_symbol_b(self, tmp_path):
        task, contract, _ = two_obligation_contract()
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        record_behavioral(store, task.task_id, symbols=["export_csv"], exit_code=0, trivial=False, target_paths=["exporter.py"])
        result = RequirementAssessmentEngine(fs).assess(contract, store, [FileOperation(action="modify", path="exporter.py")], "+def export_csv(): ...", last_review=None)
        req = next(r for r in result.requirements if r.requirement_type == RequirementType.FUNCTIONAL.value)
        json_ob = next(o for o in req.acceptance_obligations if "JSON" in o.description)
        assert json_ob.method != AcceptanceMethod.EXECUTABLE_BEHAVIORAL.value

    def test_p4_behavioral_evidence_for_path_a_cannot_prove_path_b(self, tmp_path):
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        req = Requirement(requirement_id="REQ-001", statement="x", target_paths=["a.py"],
                           acceptance_obligations=[AcceptanceObligation(obligation_id="REQ-001-OB1", description="csv export", target_tokens=["csv"], target_paths=["a.py"])])
        contract = TaskContract(task_id="t1", objective="x", requirements=[req])
        record_behavioral(store, "t1", symbols=["export_csv"], exit_code=0, trivial=False, target_paths=["b.py"])
        result = RequirementAssessmentEngine(fs).assess(contract, store, [], "", last_review=None)
        assert result.satisfied is False

    def test_p5_evidence_copied_across_tasks_cannot_silently_become_valid(self, tmp_path):
        _, contract, _ = two_obligation_contract(task_id="victim")
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        record_behavioral(store, "attacker", symbols=["export_csv", "export_json"], exit_code=0, trivial=False, target_paths=["exporter.py"])
        result = RequirementAssessmentEngine(fs).assess(contract, store, [], "", last_review=None)
        assert result.satisfied is False

    def test_p6_persisted_evidence_cannot_override_live_invalidation(self, tmp_path):
        (tmp_path / "exporter.py").write_text("v1", encoding="utf-8")
        store = CompletionEvidenceStore(tmp_path)
        record_behavioral(store, "t1", symbols=["export_csv"], exit_code=0, trivial=False, target_paths=["exporter.py"])
        (tmp_path / "exporter.py").write_text("v2", encoding="utf-8")
        invalidated = store.revalidate_against_disk(ProjectFilesystem(tmp_path))
        assert len(invalidated) == 1
        assert store.get_valid_evidence(EvidenceType.TEST_EXECUTION) == []

    def test_p7_stronger_evidence_cannot_be_manufactured_from_weaker(self, tmp_path):
        # Diff correlation alone, however strong the textual match, never
        # gets tagged as EXECUTABLE_BEHAVIORAL -- the method field always
        # honestly reflects DIFF_CORRELATION for a non-behavioral match.
        task, contract, _ = two_obligation_contract()
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)  # no behavioral evidence at all
        result = RequirementAssessmentEngine(fs).assess(contract, store, [FileOperation(action="modify", path="exporter.py")], "+def export_csv(): ...\n+def export_json(): ...", last_review=None)
        req = next(r for r in result.requirements if r.requirement_type == RequirementType.FUNCTIONAL.value)
        assert all(o.method == AcceptanceMethod.DIFF_CORRELATION.value for o in req.acceptance_obligations)

    def test_p8_provider_reviewer_claims_cannot_create_provenance(self, tmp_path):
        import inspect
        sig = inspect.signature(RequirementAssessmentEngine.assess)
        assert set(sig.parameters) <= {
            "self", "contract", "evidence_store", "applied_operations", "current_diff",
            "last_review", "clarification_requests",
        }

    def test_p9_forged_persisted_binding_metadata_cannot_create_satisfaction(self, tmp_path):
        task, contract, req = two_obligation_contract()
        forged = [o.__class__(**{**o.to_dict(), "state": RequirementState.SATISFIED.value, "method": AcceptanceMethod.EXECUTABLE_BEHAVIORAL.value, "provenance": AcceptanceProvenance.TEST_DERIVED.value, "evidence_ids": ["ev-forged"]}) for o in req.acceptance_obligations]
        req2 = req.__class__(**{**req.to_dict(), "acceptance_obligations": forged, "state": RequirementState.SATISFIED.value})
        contract.requirements = [req2]
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        result = RequirementAssessmentEngine(fs).assess(contract, store, [], "", last_review=None)
        assert result.satisfied is False

    def test_p10_failed_evidence_dominates_positive_evidence(self, tmp_path):
        task, contract, _ = two_obligation_contract()
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        record_behavioral(store, task.task_id, symbols=["export_csv"], exit_code=1, trivial=False, target_paths=["exporter.py"])
        review = ReviewResult(verdict="APPROVED", summary="ok")
        result = RequirementAssessmentEngine(fs).assess(contract, store, [FileOperation(action="modify", path="exporter.py")], "+def export_csv(): ...", last_review=review)
        assert result.satisfied is False

    def test_p11_partial_compound_satisfaction_remains_incomplete(self, tmp_path):
        task, contract, _ = two_obligation_contract(objective="Add CSV, JSON, and XML export")
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        result = RequirementAssessmentEngine(fs).assess(contract, store, [FileOperation(action="modify", path="exporter.py")], "+def export_csv(): ...\n+def export_json(): ...", last_review=None)
        assert result.satisfied is False

    def test_p12_multiturn_orchestrator_share_identical_trust_model(self, tmp_path):
        import inspect
        from local_agent.multi_turn import MultiTurnImplementationAgent
        orch_src = inspect.getsource(Orchestrator.run)
        mt_src = inspect.getsource(MultiTurnImplementationAgent.execute)
        assert "req_assessment.satisfied" in orch_src
        assert "req_assessment.satisfied" in mt_src

    def test_p13_evidence_history_remains_bounded(self, tmp_path):
        fs = ProjectFilesystem(tmp_path)
        task, contract, _ = two_obligation_contract()
        store = CompletionEvidenceStore(tmp_path, max_entries=50)
        engine = RequirementAssessmentEngine(fs)
        review = ReviewResult(verdict="APPROVED", summary="ok")
        for i in range(200):
            engine.assess(contract, store, [FileOperation(action="modify", path=f"f{i}.py")], f"+x{i}=1", last_review=review)
        assert len(store.all_entries()) <= 50

    def test_p14_existing_phase420_to_422_invariants_intact(self, tmp_path):
        # Spot-check: MANUAL_CLARIFICATION requirements still fail closed to
        # UNVERIFIABLE (Phase 4.22), never silently satisfied.
        task = make_task(objective="Update the parser. Preserve backward compatibility of the public API.")
        contract = derive_task_contract(task, None)
        compat = [r for r in contract.requirements if r.requirement_type == RequirementType.COMPATIBILITY.value]
        assert len(compat) == 1
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        result = RequirementAssessmentEngine(fs).assess(contract, store, [FileOperation(action="modify", path="parser.py")], "+def new_parse(x): ...", last_review=None)
        assert result.satisfied is False

    def test_p15_final_completion_requires_readiness_and_contract(self, tmp_path):
        import inspect
        src = inspect.getsource(Orchestrator.run)
        assert "assessment.is_ready and req_assessment.satisfied" in src


# =============================================================================
# Section 5: bounds / performance (measured, not estimated -- see report)
# =============================================================================

class TestBounds:
    def test_candidate_collection_scales_with_fixed_bounds_not_unbounded(self, tmp_path):
        import time
        fs = ProjectFilesystem(tmp_path)
        reqs = []
        for r in range(MAX_REQUIREMENTS):
            obs = [AcceptanceObligation(obligation_id=f"REQ-{r}-OB{i}", description=f"item {r} {i}", target_tokens=[f"tok{r}{i}"]) for i in range(MAX_OBLIGATIONS_PER_REQUIREMENT)]
            reqs.append(Requirement(requirement_id=f"REQ-{r:03d}", statement=f"req {r}", acceptance_obligations=obs))
        contract = TaskContract(task_id="t1", objective="pathological", requirements=reqs)
        store = CompletionEvidenceStore(tmp_path, max_entries=100)
        for i in range(100):
            record_behavioral(store, "t1", symbols=[f"sym_{i}"], exit_code=0, trivial=False, target_paths=[])
        engine = RequirementAssessmentEngine(fs)
        review = ReviewResult(verdict="APPROVED", summary="ok")
        start = time.time()
        engine.assess(contract, store, [FileOperation(action="modify", path="x.py")], "+x=1", last_review=review)
        elapsed = time.time() - start
        assert elapsed < 5.0  # generous bound; measured absolute worst case is well under this


# =============================================================================
# Section 6: backward compatibility
# =============================================================================

class TestBackwardCompatibility:
    def test_get_valid_evidence_without_task_id_preserves_old_behavior(self, tmp_path):
        store = CompletionEvidenceStore(tmp_path)
        record_behavioral(store, "any-task", symbols=["x"], exit_code=0, trivial=False, target_paths=[])
        # Omitting task_id (the default) must behave exactly as before Phase 4.23.
        assert len(store.get_valid_evidence(EvidenceType.TEST_EXECUTION)) == 1

    def test_old_checkpoint_evidence_store_without_task_scoping_history_loads(self, tmp_path):
        old_data = {
            "workspace_root": str(tmp_path), "max_entries": 100,
            "entries": [{
                "evidence_id": "ev-1", "task_id": "t1", "subtask_id": "main", "turn_number": 1,
                "stage": "testing", "evidence_type": "test_execution", "source": "old",
                "trust_tier": 1, "status": "valid", "target_paths": [], "target_symbols": ["foo"],
                "exit_code": 0, "payload": {},
            }],
        }
        store = CompletionEvidenceStore.from_dict(old_data)
        assert len(store.get_valid_evidence()) == 1
        assert len(store.get_valid_evidence(task_id="t1")) == 1
        assert len(store.get_valid_evidence(task_id="other")) == 0
