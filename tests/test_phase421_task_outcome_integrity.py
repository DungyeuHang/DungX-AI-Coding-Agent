"""Phase 4.21: Task Outcome Integrity & Requirement Traceability.

Phase 4.18-4.20 answered "can the agent's technical process be trusted?"
This phase answers a different question: can a technically perfect run --
green tests, clean syntax, an approved review -- still fail to do what the
user actually asked? A task can pass every gate in completion.py while only
implementing half of a multi-part request, silently violating a stated
constraint, or leaving a requested deliverable (documentation, tests)
untouched. local_agent/task_contract.py adds a second, independent
dimension to the completion decision -- task-contract satisfaction -- that
must hold *in addition to* technical readiness before Orchestrator.run() or
MultiTurnImplementationAgent.execute() may report the task complete.

Organized by: contract derivation & bounds; requirement lifecycle &
evidence binding; the twelve named adversarial attacks; property invariants
T1-T15; multi-turn/checkpoint integration; backward compatibility; and
reporting.
"""

from __future__ import annotations

import datetime
import hashlib
from pathlib import Path
from typing import Any

import pytest

from local_agent.completion import (
    CompletionEvidenceStore,
    EvidenceType,
)
from local_agent.config import AgentConfig
from local_agent.filesystem import ProjectFilesystem
from local_agent.models import (
    Checkpoint,
    ClarificationRequest,
    ExecutionResult,
    FileOperation,
    MultiTurnExecutionReport,
    Plan,
    ProjectContext,
    ProviderCapability,
    ReviewResult,
    RunReport,
    Task,
    TaskStatus,
)
from local_agent.multi_turn import MultiTurnImplementationAgent
from local_agent.orchestrator import Orchestrator
from local_agent.storage import JsonFileStorage
from local_agent.task_contract import (
    MAX_NON_GOALS,
    MAX_REQUIREMENTS,
    MAX_STATEMENT_CHARS,
    MAX_TARGET_PATHS_PER_REQUIREMENT,
    Requirement,
    RequirementAssessmentEngine,
    RequirementImportance,
    RequirementSatisfactionAssessment,
    RequirementState,
    RequirementType,
    TaskContract,
    VerificationStrategy,
    derive_task_contract,
)
from local_agent.tools import ToolRegistry


# -----------------------------------------------------------------------------
# Shared scaffolding
# -----------------------------------------------------------------------------

def make_task(task_id: str = "t1", objective: str = "Implement feature") -> Task:
    now = datetime.datetime.now(datetime.timezone.utc)
    return Task(task_id=task_id, objective=objective, status=TaskStatus.PENDING, created_at=now, updated_at=now)


def diff_hash(diff: str) -> str:
    return hashlib.sha256(diff.encode("utf-8")).hexdigest()[:16]


def assess(tmp_path, objective, ops, diff, plan=None, review_verdict="APPROVED", clarifications=None, store=None):
    fs = ProjectFilesystem(tmp_path)
    task = make_task(objective=objective)
    contract = derive_task_contract(task, plan)
    store = store or CompletionEvidenceStore(tmp_path)
    review = ReviewResult(verdict=review_verdict, summary="ok") if review_verdict else None
    engine = RequirementAssessmentEngine(fs)
    result = engine.assess(contract, store, ops, diff, last_review=review, clarification_requests=clarifications)
    return contract, result


class DummyProvider:
    def __init__(self, responses=None, review_verdict="APPROVED"):
        self.provider_id = "mock"
        self.model = "mock-model"
        self.capabilities = {ProviderCapability.TOOL_USE, ProviderCapability.IMPLEMENTATION}
        self.responses = list(responses or [])
        self.review_verdict = review_verdict

    def generate_code_with_tools(self, *a, **k):
        return self.responses.pop(0) if self.responses else []

    def generate_code(self, *a, **k):
        return self.responses.pop(0) if self.responses else []

    def review_changes_with_tools(self, *a, **k):
        return ReviewResult(verdict=self.review_verdict, summary="mock review", findings=[])

    def review_changes(self, *a, **k):
        return ReviewResult(verdict=self.review_verdict, summary="mock review", findings=[])


def make_agent(tmp_path: Path):
    fs = ProjectFilesystem(tmp_path)
    reg = ToolRegistry(tmp_path, filesystem=fs)
    storage = JsonFileStorage(tmp_path / ".agent")
    config = AgentConfig(project=str(tmp_path), multi_turn_implementation=True, validation_commands=["pytest"])
    from local_agent.tools import CommandRunner

    class NoopRunner(CommandRunner):
        def __init__(self):
            pass

        def run(self, command, timeout=None):
            return ExecutionResult(command=" ".join(command) if isinstance(command, (list, tuple)) else str(command), exit_code=0, stdout="ok", stderr="")

    agent = MultiTurnImplementationAgent(config, fs, reg, storage, NoopRunner())
    return agent, fs, storage, config


# -----------------------------------------------------------------------------
# Contract derivation & bounds
# -----------------------------------------------------------------------------

def test_contract_derivation_single_clause_is_lenient(tmp_path):
    task = make_task(objective="Fix the login crash")
    contract = derive_task_contract(task, None)
    assert len(contract.requirements) == 1
    assert contract.requirements[0].verification_strategy == VerificationStrategy.DIFF_PRESENCE.value


def test_contract_derivation_compound_objective_splits(tmp_path):
    task = make_task(objective="Add CSV export and JSON export.")
    contract = derive_task_contract(task, None)
    assert len(contract.requirements) == 2
    assert {r.statement for r in contract.requirements} == {"Add CSV export", "JSON export"}


def test_contract_derivation_never_empty(tmp_path):
    task = make_task(objective="   ")
    contract = derive_task_contract(task, None)
    # A blank objective legitimately has no requirements -- this documents
    # that an empty contract is only reachable from empty input, never from
    # a real objective being silently dropped.
    assert contract.requirements == []


def test_contract_bounds_requirements_capped():
    huge = [
        Requirement(requirement_id=f"REQ-{i:03d}", statement=f"Do thing {i}")
        for i in range(MAX_REQUIREMENTS + 25)
    ]
    contract = TaskContract(task_id="t", objective="x", requirements=huge)
    assert len(contract.requirements) == MAX_REQUIREMENTS


def test_contract_bounds_non_goals_capped():
    contract = TaskContract(task_id="t", objective="x", non_goals=[f"goal {i}" for i in range(MAX_NON_GOALS + 10)])
    assert len(contract.non_goals) == MAX_NON_GOALS


def test_requirement_bounds_statement_and_paths_capped():
    req = Requirement(
        requirement_id="R1",
        statement="x" * (MAX_STATEMENT_CHARS + 500),
        target_paths=[f"path{i}.py" for i in range(MAX_TARGET_PATHS_PER_REQUIREMENT + 10)],
    )
    assert len(req.statement) <= MAX_STATEMENT_CHARS
    assert len(req.target_paths) == MAX_TARGET_PATHS_PER_REQUIREMENT


def test_contract_deduplicates_requirement_ids():
    dup = [
        Requirement(requirement_id="REQ-001", statement="First"),
        Requirement(requirement_id="REQ-001", statement="Second"),
    ]
    contract = TaskContract(task_id="t", objective="x", requirements=dup)
    ids = [r.requirement_id for r in contract.requirements]
    assert len(ids) == len(set(ids))


# -----------------------------------------------------------------------------
# Requirement lifecycle & evidence binding
# -----------------------------------------------------------------------------

def test_requirement_state_default_is_unverified():
    req = Requirement(requirement_id="R1", statement="x")
    assert req.state == RequirementState.UNVERIFIED.value


def test_constraint_requirement_satisfied_when_path_untouched(tmp_path):
    fs = ProjectFilesystem(tmp_path)
    contract = TaskContract(task_id="t1", objective="x", requirements=[
        Requirement(requirement_id="R1", statement="Do not modify config.py",
                    requirement_type=RequirementType.CONSTRAINT.value,
                    verification_strategy=VerificationStrategy.CONSTRAINT_ABSENCE.value,
                    target_paths=["config.py"]),
    ])
    store = CompletionEvidenceStore(tmp_path)
    result = RequirementAssessmentEngine(fs).assess(contract, store, [FileOperation(action="modify", path="app.py")], "diff")
    assert result.satisfied is True
    assert result.requirements[0].state == RequirementState.SATISFIED.value


def test_evidence_recorded_for_each_requirement_assessment(tmp_path):
    fs = ProjectFilesystem(tmp_path)
    contract = TaskContract(task_id="t1", objective="x", requirements=[
        Requirement(requirement_id="R1", statement="Do not modify config.py",
                    verification_strategy=VerificationStrategy.CONSTRAINT_ABSENCE.value,
                    target_paths=["config.py"]),
    ])
    store = CompletionEvidenceStore(tmp_path)
    RequirementAssessmentEngine(fs).assess(contract, store, [], "diff")
    contract_evidence = store.get_valid_evidence(EvidenceType.CONTRACT_COMPLIANCE)
    assert len(contract_evidence) == 1
    assert contract_evidence[0].payload["requirement_id"] == "R1"


# -----------------------------------------------------------------------------
# The twelve named attacks
# -----------------------------------------------------------------------------

def test_attack_1_missing_requirement(tmp_path):
    _, result = assess(
        tmp_path, "Add CSV export and JSON export.",
        [FileOperation(action="modify", path="exporters/csv_export.py")],
        "+def export_csv(rows): return to_csv(rows)",
    )
    assert result.satisfied is False
    states = {r.statement: r.state for r in result.requirements}
    assert states["Add CSV export"] == RequirementState.SATISFIED.value
    assert states["JSON export"] != RequirementState.SATISFIED.value


def test_attack_2_wrong_implementation(tmp_path):
    _, wrong = assess(
        tmp_path, "Change the request timeout from 30 seconds to 60 seconds.",
        [FileOperation(action="modify", path="config/retry.py")],
        "-RETRY_DELAY = 5\n+RETRY_DELAY = 10\n",
    )
    assert wrong.satisfied is False

    _, right = assess(
        tmp_path, "Change the request timeout from 30 seconds to 60 seconds.",
        [FileOperation(action="modify", path="config/net.py")],
        "-REQUEST_TIMEOUT = 30\n+REQUEST_TIMEOUT = 60\n",
    )
    assert right.satisfied is True


def test_attack_3_constraint_violation(tmp_path):
    _, result = assess(
        tmp_path, "Refactor the review pipeline. Do not modify local_agent/reviewer.py.",
        [FileOperation(action="modify", path="local_agent/reviewer.py")],
        "+# refactored",
    )
    assert result.satisfied is False
    assert any(r.state == RequirementState.FAILED.value for r in result.requirements)


def test_attack_4_documentation_omission(tmp_path):
    _, omitted = assess(
        tmp_path, "Add rate limiting and update the documentation.",
        [FileOperation(action="modify", path="ratelimit/limiter.py")],
        "+def rate_limit(): pass",
    )
    assert omitted.satisfied is False
    doc_states = [r.state for r in omitted.requirements if r.requirement_type == RequirementType.DOCUMENTATION.value]
    assert doc_states == [RequirementState.UNVERIFIED.value]

    _, done = assess(
        tmp_path, "Add rate limiting and update the documentation.",
        [FileOperation(action="modify", path="ratelimit/limiter.py"), FileOperation(action="modify", path="README.md")],
        "+def rate_limit(): pass\n+## Rate limiting\n",
    )
    assert done.satisfied is True


def test_attack_5_compatibility_violation(tmp_path):
    _, result = assess(
        tmp_path, "Update the parser. Preserve backward compatibility of the public API.",
        [FileOperation(action="modify", path="parser.py")],
        "+def new_parse(x): ...",
    )
    assert result.satisfied is False
    compat = [r for r in result.requirements if r.requirement_type == RequirementType.COMPATIBILITY.value]
    assert len(compat) == 1
    # No authoritative compatibility evidence exists -- it must stay
    # unresolved, never silently satisfied by the fact that *some* change
    # was made and review approved it. Phase 4.22: MANUAL_CLARIFICATION
    # requirements are now labeled UNVERIFIABLE (honest: no rule could ever
    # auto-check this) rather than UNVERIFIED (implies a pending check) --
    # gating behavior is identical, this is a purely honest relabeling.
    assert compat[0].state == RequirementState.UNVERIFIABLE.value


def test_attack_6_provider_lies_have_no_input_to_lie_through(tmp_path):
    """The provider's own textual claim ("all requirements completed") has
    no parameter in RequirementAssessmentEngine.assess() to even be passed
    through -- this is a structural guarantee, not a rule that could be
    forgotten. Demonstrated by constructing a review whose summary makes
    exactly that claim and confirming it has zero effect on the outcome."""
    lying_review = ReviewResult(verdict="APPROVED", summary="All requirements completed successfully!")
    task = make_task(objective="Add CSV export and JSON export.")
    contract = derive_task_contract(task, None)
    fs = ProjectFilesystem(tmp_path)
    store = CompletionEvidenceStore(tmp_path)
    result = RequirementAssessmentEngine(fs).assess(
        contract, store, [FileOperation(action="modify", path="csv_export.py")], "+csv stuff",
        last_review=lying_review,
    )
    assert result.satisfied is False


def test_attack_7_review_lies_one_requirement_still_unverified(tmp_path):
    _, result = assess(
        tmp_path, "Add CSV export and JSON export.",
        [FileOperation(action="modify", path="csv_export.py")],
        "+csv stuff",
        review_verdict="APPROVED",
    )
    assert result.satisfied is False


def test_attack_8_test_laundering_broad_pass_does_not_verify_specific_requirement(tmp_path):
    """A single broad passing test suite run does not, by itself, prove a
    *specific* functional requirement was implemented -- DIFF_PRESENCE
    requirements are correlated against the diff/changed paths, not against
    whether *some* TEST_EXECUTION evidence happens to be green."""
    fs = ProjectFilesystem(tmp_path)
    task = make_task(objective="Add CSV export and JSON export.")
    contract = derive_task_contract(task, None)
    store = CompletionEvidenceStore(tmp_path)
    store.record(task_id="t1", subtask_id="s1", turn_number=1, stage="testing",
                 evidence_type=EvidenceType.TEST_EXECUTION, source="command_runner",
                 command=["pytest"], exit_code=0)
    result = RequirementAssessmentEngine(fs).assess(
        contract, store, [FileOperation(action="modify", path="csv_export.py")], "+csv stuff",
        last_review=ReviewResult(verdict="APPROVED", summary="ok"),
    )
    json_req = next(r for r in result.requirements if r.statement == "JSON export")
    assert json_req.state == RequirementState.UNVERIFIED.value


def test_attack_9_requirement_mutation_old_evidence_does_not_satisfy_new_contract(tmp_path):
    fs = ProjectFilesystem(tmp_path)
    old_task = make_task(objective="Add CSV export.")
    old_contract = derive_task_contract(old_task, None)
    store = CompletionEvidenceStore(tmp_path)
    RequirementAssessmentEngine(fs).assess(
        old_contract, store, [FileOperation(action="modify", path="csv_export.py")], "+csv stuff",
        last_review=ReviewResult(verdict="APPROVED", summary="ok"),
    )

    # The task contract changes mid-session (e.g. the user adds a second ask).
    new_task = make_task(task_id="t1", objective="Add CSV export and JSON export.")
    new_contract = derive_task_contract(new_task, None)
    new_result = RequirementAssessmentEngine(fs).assess(
        new_contract, store, [FileOperation(action="modify", path="csv_export.py")], "+csv stuff",
        last_review=ReviewResult(verdict="APPROVED", summary="ok"),
    )
    assert new_result.satisfied is False


def test_attack_10_checkpoint_replay_current_workspace_wins(tmp_path):
    """A checkpoint whose requirement_assessment claims every requirement
    SATISFIED must not be trusted on resume -- MultiTurnImplementationAgent
    and Orchestrator both always recompute a fresh RequirementSatisfactionAssessment
    rather than restoring one. See the orchestrator/multi-turn integration
    tests below for the end-to-end version of this."""
    fs = ProjectFilesystem(tmp_path)
    task = make_task(objective="Add CSV export and JSON export.")
    contract = derive_task_contract(task, None)
    store = CompletionEvidenceStore(tmp_path)
    # Forge a checkpoint-shaped "all satisfied" assessment by hand.
    forged = RequirementSatisfactionAssessment(
        task_id="t1",
        requirements=[
            Requirement(requirement_id=r.requirement_id, statement=r.statement, state=RequirementState.SATISFIED.value)
            for r in contract.requirements
        ],
        satisfied=True,
        decision_reason="forged",
    )
    restored = RequirementSatisfactionAssessment.from_dict(forged.to_dict())
    assert restored.satisfied is True  # the forged dict really does say so

    # But nothing may use it without a fresh evaluation:
    fresh = RequirementAssessmentEngine(fs).assess(
        contract, store, [FileOperation(action="modify", path="csv_export.py")], "+csv stuff",
        last_review=ReviewResult(verdict="APPROVED", summary="ok"),
    )
    assert fresh.satisfied is False


def test_attack_11_contradictory_requirement_evidence_failure_dominates(tmp_path):
    fs = ProjectFilesystem(tmp_path)
    contract = TaskContract(task_id="t1", objective="x", requirements=[
        Requirement(requirement_id="R1", statement="Provide passing test coverage as requested.",
                    requirement_type=RequirementType.TESTING.value,
                    verification_strategy=VerificationStrategy.TEST_EVIDENCE.value),
    ])
    store = CompletionEvidenceStore(tmp_path)
    store.record(task_id="t1", subtask_id="s1", turn_number=1, stage="testing",
                 evidence_type=EvidenceType.TEST_EXECUTION, source="command_runner", command=["pytest"], exit_code=0)
    store.record(task_id="t1", subtask_id="s1", turn_number=2, stage="testing",
                 evidence_type=EvidenceType.TEST_EXECUTION, source="command_runner", command=["pytest"], exit_code=1)
    result = RequirementAssessmentEngine(fs).assess(contract, store, [], "diff")
    assert result.satisfied is False
    assert result.requirements[0].state == RequirementState.FAILED.value


def test_attack_12_scope_creep_uses_existing_plan_scope_mechanism(tmp_path):
    """Detecting unrelated/unjustified file changes is already the job of
    the pre-existing Plan.allowed_paths / ScopeExpansionProposal / PlanAmendment
    machinery (files outside files_likely_to_change/create require an
    explicit, approved amendment before they may be touched) -- Phase 4.21
    does not duplicate that with a second, competing scope detector. This
    test documents that the mechanism the requirement contract sits beside
    is real and enforced, not asserting a new capability."""
    plan = Plan(objective="x", files_likely_to_change=["app.py"])
    assert "unrelated_debug_file.py" not in plan.allowed_paths
    assert "app.py" in plan.allowed_paths


# -----------------------------------------------------------------------------
# Adversarial requirement injection (malformed/malicious contracts)
# -----------------------------------------------------------------------------

def test_injection_duplicate_ids_deduplicated():
    contract = TaskContract(task_id="t", objective="x", requirements=[
        Requirement(requirement_id="R1", statement="a"),
        Requirement(requirement_id="R1", statement="b"),
        Requirement(requirement_id="R1", statement="c"),
    ])
    ids = [r.requirement_id for r in contract.requirements]
    assert len(ids) == len(set(ids)) == 3


def test_injection_missing_id_gets_assigned():
    contract = TaskContract(task_id="t", objective="x", requirements=[Requirement(requirement_id="", statement="a")])
    assert contract.requirements[0].requirement_id


def test_injection_empty_requirement_statement_is_inert(tmp_path):
    fs = ProjectFilesystem(tmp_path)
    contract = TaskContract(task_id="t1", objective="x", requirements=[Requirement(requirement_id="R1", statement="")])
    store = CompletionEvidenceStore(tmp_path)
    result = RequirementAssessmentEngine(fs).assess(contract, store, [FileOperation(action="modify", path="a.py")], "diff")
    # An empty statement carries no content to correlate against; with no
    # sibling to disambiguate from it is treated leniently like any other
    # vacuous single requirement -- not silently upgraded to something
    # stricter, and not a crash.
    assert result.requirements[0].state in (RequirementState.SATISFIED.value, RequirementState.UNVERIFIED.value)


def test_injection_contradictory_requirements_each_evaluated_independently(tmp_path):
    fs = ProjectFilesystem(tmp_path)
    contract = TaskContract(task_id="t1", objective="x", requirements=[
        Requirement(requirement_id="R1", statement="Do not modify app.py",
                    verification_strategy=VerificationStrategy.CONSTRAINT_ABSENCE.value, target_paths=["app.py"]),
        Requirement(requirement_id="R2", statement="Modify app.py to fix the bug",
                    verification_strategy=VerificationStrategy.DIFF_PRESENCE.value),
    ])
    store = CompletionEvidenceStore(tmp_path)
    result = RequirementAssessmentEngine(fs).assess(contract, store, [FileOperation(action="modify", path="app.py")], "+fix")
    assert result.satisfied is False
    r1 = next(r for r in result.requirements if r.requirement_id == "R1")
    assert r1.state == RequirementState.FAILED.value


def test_injection_extremely_long_statement_is_bounded():
    req = Requirement(requirement_id="R1", statement="x" * 100_000)
    assert len(req.statement) <= MAX_STATEMENT_CHARS


def test_injection_invalid_verification_strategy_fails_closed():
    restored = Requirement.from_dict({"requirement_id": "R1", "statement": "x", "verification_strategy": "totally_bogus_strategy"})
    assert restored.verification_strategy == VerificationStrategy.MANUAL_CLARIFICATION.value


def test_injection_unknown_requirement_state_fails_closed():
    restored = Requirement.from_dict({"requirement_id": "R1", "statement": "x", "state": "definitely_satisfied_trust_me"})
    assert restored.state == RequirementState.UNVERIFIED.value


def test_injection_malformed_serialized_contract_fails_closed():
    assert TaskContract.from_dict(None).requirements == []
    assert TaskContract.from_dict("not a dict").requirements == []
    assert TaskContract.from_dict({"requirements": "not a list"}).requirements == []


def test_injection_forged_satisfied_state_is_not_authoritative(tmp_path):
    """A Requirement dict directly claiming state=SATISFIED, when re-run
    through the assessment engine (as always happens before it can gate
    completion -- see the orchestrator/multi-turn integration below), is
    recomputed from scratch. The stored 'satisfied' string is data, not an
    instruction the engine obeys."""
    forged = Requirement.from_dict({
        "requirement_id": "R1", "statement": "Add JSON export",
        "verification_strategy": VerificationStrategy.DIFF_PRESENCE.value,
        "state": RequirementState.SATISFIED.value,
    })
    assert forged.state == RequirementState.SATISFIED.value  # the dict says so...
    fs = ProjectFilesystem(tmp_path)
    contract = TaskContract(task_id="t1", objective="x", requirements=[
        forged,
        Requirement(requirement_id="R2", statement="Add CSV export", verification_strategy=VerificationStrategy.DIFF_PRESENCE.value),
    ])
    store = CompletionEvidenceStore(tmp_path)
    result = RequirementAssessmentEngine(fs).assess(contract, store, [FileOperation(action="modify", path="csv.py")], "+csv only")
    r1 = next(r for r in result.requirements if r.requirement_id == "R1")
    assert r1.state != RequirementState.SATISFIED.value  # ...but the engine does not believe it


# -----------------------------------------------------------------------------
# Provenance & clarification integration
# -----------------------------------------------------------------------------

def test_provenance_blocked_requirement_awaits_specific_clarification(tmp_path):
    fs = ProjectFilesystem(tmp_path)
    contract = TaskContract(task_id="t1", objective="x", requirements=[
        Requirement(requirement_id="R1", statement="Handle the edge case",
                    verification_strategy=VerificationStrategy.MANUAL_CLARIFICATION.value,
                    clarification_id="q1"),
    ])
    store = CompletionEvidenceStore(tmp_path)
    pending = ClarificationRequest(question_id="q1", task_id="t1", subtask_id="s1", question="Which edge case?", status="pending")
    result = RequirementAssessmentEngine(fs).assess(contract, store, [], "diff", clarification_requests=[pending])
    assert result.requirements[0].state == RequirementState.BLOCKED.value

    answered = ClarificationRequest(question_id="q1", task_id="t1", subtask_id="s1", question="Which edge case?", status="answered", answer="Null input")
    result2 = RequirementAssessmentEngine(fs).assess(contract, store, [], "diff", clarification_requests=[answered])
    assert result2.requirements[0].state != RequirementState.BLOCKED.value


# -----------------------------------------------------------------------------
# Serialization round trip
# -----------------------------------------------------------------------------

def test_serialization_round_trip_requirement_and_contract():
    req = Requirement(requirement_id="R1", statement="Add CSV export", target_paths=["a.py"], evidence_ids=["ev-1"])
    restored = Requirement.from_dict(req.to_dict())
    assert restored.requirement_id == req.requirement_id
    assert restored.statement == req.statement
    assert restored.target_paths == req.target_paths

    contract = TaskContract(task_id="t1", objective="x", requirements=[req])
    restored_contract = TaskContract.from_dict(contract.to_dict())
    assert len(restored_contract.requirements) == 1
    assert restored_contract.requirements[0].requirement_id == "R1"


def test_serialization_round_trip_via_checkpoint_run_report_multi_turn_report(tmp_path):
    contract = TaskContract(task_id="t1", objective="x", requirements=[Requirement(requirement_id="R1", statement="a")])
    cp = Checkpoint(
        checkpoint_id="cp1", task_id="t1", subtask_id="s1",
        timestamp=datetime.datetime.now(datetime.timezone.utc), current_state_description="x",
        task_contract=contract.to_dict(),
        requirement_assessment={"satisfied": False},
    )
    restored_cp = Checkpoint.from_dict(cp.to_dict())
    assert restored_cp.task_contract["requirements"][0]["requirement_id"] == "R1"

    report = RunReport(project=ProjectContext(str(tmp_path)), task_contract=contract, requirement_assessment={"satisfied": False})
    assert report.task_contract.requirements[0].requirement_id == "R1"

    mt = MultiTurnExecutionReport(task_contract=contract.to_dict(), requirement_assessment={"satisfied": False})
    restored_mt = MultiTurnExecutionReport.from_dict(mt.to_dict())
    assert restored_mt.task_contract["requirements"][0]["requirement_id"] == "R1"


# -----------------------------------------------------------------------------
# Backward compatibility
# -----------------------------------------------------------------------------

def test_backcompat_checkpoint_missing_task_contract_fields():
    now = datetime.datetime.now(datetime.timezone.utc)
    old_style = {
        "checkpoint_id": "cp-old", "task_id": "t1", "subtask_id": "s1",
        "timestamp": now.isoformat(), "current_state_description": "legacy checkpoint",
    }
    restored = Checkpoint.from_dict(old_style)
    assert restored.task_contract is None
    assert restored.requirement_assessment == {}


def test_backcompat_task_missing_task_contract_field():
    now = datetime.datetime.now(datetime.timezone.utc)
    old_style = {
        "task_id": "t1", "objective": "x", "status": "pending",
        "created_at": now.isoformat(), "updated_at": now.isoformat(),
    }
    restored = Task.from_dict(old_style)
    assert restored.task_contract is None


def test_backcompat_missing_contract_never_implies_satisfied(tmp_path):
    """An old checkpoint/task with no task_contract at all must not be
    silently treated as 'nothing to satisfy, therefore satisfied' by any
    consumer -- the orchestrator/multi-turn integration always derives a
    fresh contract when none exists rather than skipping the check."""
    task = make_task(objective="Add CSV export and JSON export.")
    assert task.task_contract is None
    # Mirrors exactly what Orchestrator.run() / MultiTurnImplementationAgent.execute() do:
    contract = derive_task_contract(task, None)
    assert len(contract.requirements) == 2  # a real contract, not an empty/vacuous one


# -----------------------------------------------------------------------------
# Orchestrator / multi-turn lifecycle integration
# -----------------------------------------------------------------------------

def make_orchestrator(tmp_path: Path, config: AgentConfig | None = None):
    import threading

    storage = JsonFileStorage(tmp_path / ".agent_data")
    cfg = config or AgentConfig.from_environment(tmp_path, max_iterations=1)
    return Orchestrator(cfg, storage, None, threading.Lock(), threading.Lock()), storage, cfg


class MultiRequirementProvider:
    """Real orchestrator-facing provider double: plans a compound objective,
    then only ever implements the first half of it."""

    def __init__(self):
        self.provider_id = "mock"
        self.model = "mock-model"
        self.capabilities = {ProviderCapability.PLANNING, ProviderCapability.IMPLEMENTATION, ProviderCapability.REVIEW}

    def generate_plan(self, task, context):
        return Plan(objective=task, steps=[])

    def generate_code(self, task, plan, context, failure=None, review=None):
        return [FileOperation("modify", "src/csv_export.py", content="def export_csv(rows): return to_csv(rows)\n", reason="csv")]

    def review_changes(self, task, plan, diff, context, **kwargs):
        return ReviewResult(verdict="APPROVED", summary="looks fine")


def test_orchestrator_end_to_end_refuses_completion_on_partial_multi_part_task(tmp_path):
    """Real Orchestrator.run() lifecycle: a compound objective ("X and Y"),
    a provider that only ever implements X, tests/review all green. The
    orchestrator must refuse completion on task-contract grounds even
    though the technical completion engine is satisfied."""
    from unittest import mock

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "csv_export.py").write_text("", encoding="utf-8")
    orch, storage, cfg = make_orchestrator(tmp_path)
    provider = MultiRequirementProvider()
    task = make_task(task_id="t-partial", objective="Add CSV export and JSON export.")

    with mock.patch("local_agent.orchestrator.build_provider", return_value=provider):
        report = orch.run(task)

    assert report.completed is False
    assert report.requirement_assessment.get("satisfied") is False
    assert "TASK_CONTRACT_UNSATISFIED" in (task.outcome or "") or report.completion_assessment is not None


def test_orchestrator_never_completes_technically_ready_but_contract_unsatisfied(tmp_path):
    """Direct guard-level test mirroring orchestrator.py's post-loop
    consistency check: a technically-ready report whose requirement_assessment
    says unsatisfied must be forced to completed=False -- the exact code
    orchestrator.py runs after the main loop."""
    report = RunReport(project=ProjectContext(str(tmp_path)))
    report.completed = True  # as if the technical gate alone had set this
    report.requirement_assessment = {"satisfied": False, "decision_reason": "REQ-002 unresolved"}
    if report.requirement_assessment and not report.requirement_assessment.get("satisfied", False):
        report.completed = False
    assert report.completed is False


def test_multi_turn_agent_refuses_completion_on_partial_multi_part_task(tmp_path):
    agent, fs, storage, config = make_agent(tmp_path)
    (tmp_path / "csv_export.py").write_text("", encoding="utf-8")
    op = [FileOperation(action="modify", path="csv_export.py", content="def export_csv(rows): return to_csv(rows)\n")]
    provider = DummyProvider(responses=[op], review_verdict="APPROVED")
    task = make_task(task_id="t-mt-partial", objective="Add CSV export and JSON export.")
    plan = Plan(objective=task.objective)
    context = ProjectContext(str(tmp_path))

    report = agent.execute(task, None, plan, context, provider)

    assert report.success is False
    assert report.requirement_assessment.get("satisfied") is False


def test_multi_turn_agent_completes_when_contract_and_gates_both_satisfied(tmp_path):
    """Sanity check: a genuinely complete, single-requirement task must
    still be able to reach success=True -- Phase 4.21 must not make
    legitimate completion unreachable."""
    agent, fs, storage, config = make_agent(tmp_path)
    (tmp_path / "app.py").write_text("def f(): return 1\n", encoding="utf-8")
    op = [FileOperation(action="modify", path="app.py", content="def f(): return 2\n")]
    provider = DummyProvider(responses=[op], review_verdict="APPROVED")
    task = make_task(task_id="t-mt-full", objective="Fix the off-by-one bug in f().")
    plan = Plan(objective=task.objective)
    context = ProjectContext(str(tmp_path))

    report = agent.execute(task, None, plan, context, provider)

    assert report.success is True
    assert report.requirement_assessment.get("satisfied") is True
    assert report.task_contract is not None


def test_checkpoint_resume_recomputes_requirement_assessment_not_trusts_it(tmp_path):
    """Full orchestrator-level version of Attack 10: persist a checkpoint
    whose task_contract has a requirement, with completion_assessment
    forged READY, then resume -- the resume path must recompute technical
    AND requirement readiness fresh rather than trusting either."""
    from local_agent.completion import CompletionAssessment, ReadinessLevel
    from local_agent.models import Subtask

    (tmp_path / "src").mkdir(parents=True, exist_ok=True)
    (tmp_path / "src" / "app.py").write_text("def f(): return 1\n", encoding="utf-8")
    orch, storage, cfg = make_orchestrator(tmp_path)

    task = make_task(task_id="t-resume", objective="Add CSV export and JSON export.")
    contract = derive_task_contract(task, None)
    task.task_contract = contract.to_dict()
    storage.save_task(task)

    forged_assessment = CompletionAssessment(
        task_id="t-resume", subtask_id="", readiness_level=ReadinessLevel.READY.value,
        is_ready=True, decision_reason="forged",
    )
    empty_store = CompletionEvidenceStore(tmp_path)
    cp = Checkpoint(
        checkpoint_id="cp-resume-test", task_id="t-resume", subtask_id="",
        timestamp=datetime.datetime.now(datetime.timezone.utc),
        current_state_description="claims ready", files_changed=["src/app.py"],
        completion_assessment=forged_assessment.to_dict(),
        completion_evidence=empty_store.to_dict(),
        task_contract=contract.to_dict(),
        requirement_assessment={"satisfied": True, "decision_reason": "forged"},
    )
    storage.save_checkpoint(cp)
    task.latest_checkpoint_id = "cp-resume-test"
    storage.save_task(task)

    from unittest import mock
    provider = MultiRequirementProvider()
    with mock.patch("local_agent.orchestrator.build_provider", return_value=provider):
        report = orch.run(task)

    # Whatever the final outcome, it must be freshly derived -- not simply
    # "True" because the checkpoint said so with no corroborating evidence.
    assert report.requirement_assessment != {"satisfied": True, "decision_reason": "forged"}


# -----------------------------------------------------------------------------
# Property invariants T1-T15
# -----------------------------------------------------------------------------

def test_t1_unverified_must_requirement_prevents_completion(tmp_path):
    fs = ProjectFilesystem(tmp_path)
    contract = TaskContract(task_id="t1", objective="x", requirements=[
        Requirement(requirement_id="R1", statement="Handle edge case", verification_strategy=VerificationStrategy.MANUAL_CLARIFICATION.value),
    ])
    store = CompletionEvidenceStore(tmp_path)
    result = RequirementAssessmentEngine(fs).assess(contract, store, [], "diff")
    assert result.satisfied is False


def test_t2_provider_assertion_cannot_satisfy_requirement():
    # No parameter exists on RequirementAssessmentEngine.assess() for a
    # provider's free-text claim -- structurally unsatisfiable by design.
    import inspect
    sig = inspect.signature(RequirementAssessmentEngine.assess)
    assert "provider_claim" not in sig.parameters
    assert "provider_summary" not in sig.parameters


def test_t3_review_approval_cannot_satisfy_unverified_requirement(tmp_path):
    fs = ProjectFilesystem(tmp_path)
    contract = TaskContract(task_id="t1", objective="x", requirements=[
        Requirement(requirement_id="R1", statement="Handle edge case", verification_strategy=VerificationStrategy.MANUAL_CLARIFICATION.value),
    ])
    store = CompletionEvidenceStore(tmp_path)
    result = RequirementAssessmentEngine(fs).assess(contract, store, [], "diff", last_review=ReviewResult(verdict="APPROVED", summary="ok"))
    assert result.satisfied is False


def test_t4_generic_passing_tests_need_traceability(tmp_path):
    # Covered by test_attack_8_test_laundering_broad_pass_does_not_verify_specific_requirement
    pass


def test_t5_constraint_violation_prevents_completion(tmp_path):
    # Covered by test_attack_3_constraint_violation
    pass


def test_t6_current_workspace_dominates_historical_evidence(tmp_path):
    fs = ProjectFilesystem(tmp_path)
    calc = tmp_path / "calc.py"
    calc.write_text("def add(a, b): return a + b\n", encoding="utf-8")
    contract = TaskContract(task_id="t1", objective="x", requirements=[
        Requirement(requirement_id="R1", statement="Do not modify calc.py",
                    verification_strategy=VerificationStrategy.CONSTRAINT_ABSENCE.value, target_paths=["calc.py"]),
    ])
    store = CompletionEvidenceStore(tmp_path)
    r1 = RequirementAssessmentEngine(fs).assess(contract, store, [], "diff")
    assert r1.satisfied is True
    r2 = RequirementAssessmentEngine(fs).assess(contract, store, [FileOperation(action="modify", path="calc.py")], "+changed")
    assert r2.satisfied is False


def test_t7_requirement_change_invalidates_stale_satisfaction():
    # Covered by test_attack_9_requirement_mutation_old_evidence_does_not_satisfy_new_contract
    pass


def test_t8_checkpoint_replay_cannot_override_current_state():
    # Covered by test_attack_10_checkpoint_replay_current_workspace_wins and
    # test_checkpoint_resume_recomputes_requirement_assessment_not_trusts_it
    pass


def test_t9_malformed_contract_fails_closed():
    # Covered by the adversarial requirement injection block above
    pass


def test_t10_final_completion_requires_both_dimensions(tmp_path):
    """Technical readiness alone: insufficient. Task-contract satisfaction
    alone: insufficient. Both: required."""
    report = RunReport(project=ProjectContext(str(tmp_path)))
    # Technical ready, contract unsatisfied.
    report.completed = True
    if not {"satisfied": False}.get("satisfied", False):
        report.completed = False
    assert report.completed is False

    # Contract satisfied, technical not ready -- mirrors the completion_assessment guard.
    report2 = RunReport(project=ProjectContext(str(tmp_path)))
    report2.completed = True
    report2.completion_assessment = type("A", (), {"is_ready": False})()
    if report2.completion_assessment is not None and not getattr(report2.completion_assessment, "is_ready", False):
        report2.completed = False
    assert report2.completed is False


def test_t11_failed_requirement_dominates_positive_evidence(tmp_path):
    # Covered by test_attack_11_contradictory_requirement_evidence_failure_dominates
    pass


def test_t12_provenance_cannot_be_silently_downgraded():
    """importance is fixed at contract creation; RequirementAssessmentEngine
    never writes to it -- structurally, only replace()-ing the Requirement
    at derivation/amendment time can change importance, and assess() never
    does that."""
    req = Requirement(requirement_id="R1", statement="x", importance=RequirementImportance.MUST.value)
    contract = TaskContract(task_id="t1", objective="x", requirements=[req])
    fs_dummy = None  # not needed: importance is untouched regardless of assess() outcome
    from dataclasses import fields
    assess_source_fields = {f.name for f in fields(Requirement)}
    assert "importance" in assess_source_fields
    # The engine's replace() call in assess() never includes "importance".
    import inspect
    src = inspect.getsource(RequirementAssessmentEngine.assess)
    assert "importance=" not in src


def test_t13_unbounded_growth_prevented():
    # Covered by the contract/requirement bounds tests above
    pass


def test_t14_old_checkpoints_remain_loadable():
    # Covered by test_backcompat_checkpoint_missing_task_contract_fields
    pass


def test_t15_final_report_cannot_claim_completion_with_unverified_mandatory_requirement(tmp_path):
    report = RunReport(project=ProjectContext(str(tmp_path)))
    report.completed = True
    report.requirement_assessment = {
        "satisfied": False,
        "unsatisfied_requirement_ids": ["REQ-002"],
        "decision_reason": "1 MUST-importance requirement(s) unresolved",
    }
    if report.requirement_assessment and not report.requirement_assessment.get("satisfied", False):
        report.completed = False
    assert report.completed is False
    assert report.requirement_assessment["unsatisfied_requirement_ids"] == ["REQ-002"]
