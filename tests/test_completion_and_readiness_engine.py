import ast
import datetime
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from local_agent.completion import (
    CompletionAssessment,
    CompletionDecisionEngine,
    CompletionEvidenceStore,
    CompletionGateResult,
    EvidenceStatus,
    EvidenceTrustTier,
    EvidenceType,
    ReadinessLevel,
    StructuredEvidence,
)
from local_agent.config import AgentConfig
from local_agent.filesystem import ProjectFilesystem
from local_agent.models import (
    Checkpoint,
    ClarificationRequest,
    ExecutionResult,
    FailureAnalysis,
    FileOperation,
    ImplementationTurn,
    MultiTurnExecutionReport,
    MultiTurnState,
    Plan,
    ProjectContext,
    ProviderCapability,
    ReviewResult,
    RunReport,
    Subtask,
    Task,
    TaskStatus,
    ToolCall,
    ToolExecutionMetrics,
    ToolResult,
)
from local_agent.multi_turn import MultiTurnImplementationAgent
from local_agent.storage import JsonFileStorage
from local_agent.tools import CommandRunner, ToolRegistry


class DummyProvider:
    def __init__(
        self,
        responses: list[Any] | None = None,
        code_ops: list[FileOperation] | None = None,
        review_verdict: str = "APPROVED",
        review_findings: list[str] | None = None,
    ):
        self.provider_id = "mock"
        self.model = "mock-model"
        self.capabilities = {ProviderCapability.TOOL_USE, ProviderCapability.IMPLEMENTATION}
        self.responses = list(responses or [])
        self.code_ops = code_ops or []
        self.review_verdict = review_verdict
        self.review_findings = review_findings or []
        self.captured_review_diffs: list[str] = []

    def generate_code_with_tools(self, *args, **kwargs) -> Any:
        if self.responses:
            return self.responses.pop(0)
        return self.code_ops

    def generate_code(self, *args, **kwargs) -> list[FileOperation]:
        return self.code_ops

    def review_changes_with_tools(self, task, plan, diff, context, tools, tool_history, **kwargs) -> Any:
        self.captured_review_diffs.append(diff)
        return ReviewResult(verdict=self.review_verdict, summary="Mock review", findings=self.review_findings)

    def review_changes(self, task, plan, diff, context, **kwargs) -> ReviewResult:
        self.captured_review_diffs.append(diff)
        return ReviewResult(verdict=self.review_verdict, summary="Mock review", findings=self.review_findings)


class DummyCommandRunner:
    def __init__(self, sequence: list[ExecutionResult] | None = None):
        self.sequence = list(sequence or [])
        self.executed_commands: list[tuple[str, ...]] = []

    def run(self, command: tuple[str, ...], timeout: float | None = None) -> ExecutionResult:
        self.executed_commands.append(command)
        if self.sequence:
            return self.sequence.pop(0)
        return ExecutionResult(command=" ".join(command), exit_code=0, stdout="OK", stderr="")


def make_test_task(task_id: str = "task-1", objective: str = "Implement feature") -> Task:
    now = datetime.datetime.now(datetime.timezone.utc)
    return Task(
        task_id=task_id,
        objective=objective,
        status=TaskStatus.PENDING,
        created_at=now,
        updated_at=now,
    )


# -----------------------------------------------------------------------------
# Test 1: Evidence Hierarchy & Trust Ranking
# -----------------------------------------------------------------------------

def test_evidence_trust_hierarchy_ranking():
    assert EvidenceTrustTier.AUTHORITATIVE_EXECUTION.rank < EvidenceTrustTier.SYSTEM_INTEGRITY.rank
    assert EvidenceTrustTier.SYSTEM_INTEGRITY.rank < EvidenceTrustTier.OBSERVED_STATE.rank
    assert EvidenceTrustTier.OBSERVED_STATE.rank < EvidenceTrustTier.DELIBERATIVE_REVIEW.rank
    assert EvidenceTrustTier.DELIBERATIVE_REVIEW.rank < EvidenceTrustTier.DURABLE_CHECKPOINT.rank
    assert EvidenceTrustTier.DURABLE_CHECKPOINT.rank < EvidenceTrustTier.AGENT_ASSERTION.rank

    assert EvidenceStatus.VALID.value == "valid"
    assert EvidenceStatus.INVALIDATED.value == "invalidated"
    assert ReadinessLevel.READY.value == "READY"
    assert ReadinessLevel.BLOCKED.value == "BLOCKED"


# -----------------------------------------------------------------------------
# Test 2: Structured Evidence & Assessment Serialization Roundtrip
# -----------------------------------------------------------------------------

def test_structured_evidence_and_assessment_serialization():
    ev = StructuredEvidence(
        evidence_id="ev-123",
        task_id="task-1",
        subtask_id="sub-1",
        turn_number=1,
        stage="testing",
        evidence_type=EvidenceType.TEST_EXECUTION.value,
        source="command_runner",
        trust_tier=EvidenceTrustTier.AUTHORITATIVE_EXECUTION.value,
        status=EvidenceStatus.VALID.value,
        workspace_root="/test",
        target_paths=["src/app.py"],
        command=["pytest", "tests/"],
        exit_code=0,
        content_fingerprint="sha256:abc123",
        payload={"stdout": "passed"},
    )

    ev_dict = ev.to_dict()
    ev_restored = StructuredEvidence.from_dict(ev_dict)
    assert ev_restored.evidence_id == "ev-123"
    assert ev_restored.is_valid is True
    assert ev_restored.target_paths == ["src/app.py"]
    assert ev_restored.exit_code == 0

    gate = CompletionGateResult(
        gate_name="GATE_VALIDATION_PASSED",
        passed=True,
        reason="Tests passed",
        supporting_evidence_ids=["ev-123"],
    )
    assessment = CompletionAssessment(
        task_id="task-1",
        subtask_id="sub-1",
        readiness_level=ReadinessLevel.READY.value,
        is_ready=True,
        decision_reason="All gates passed",
        gates_evaluated=[gate],
        supporting_evidence_ids=["ev-123"],
        answers_to_ten_questions={"1_what_was_changed": ["src/app.py"]},
    )
    ass_dict = assessment.to_dict()
    ass_restored = CompletionAssessment.from_dict(ass_dict)
    assert ass_restored.is_ready is True
    assert ass_restored.readiness_level == ReadinessLevel.READY.value
    assert len(ass_restored.gates_evaluated) == 1
    assert ass_restored.gates_evaluated[0].gate_name == "GATE_VALIDATION_PASSED"


# -----------------------------------------------------------------------------
# Test 3: Evidence Store Invalidation on File Mutation
# -----------------------------------------------------------------------------

def test_evidence_store_invalidation_on_mutation(tmp_path: Path):
    app_file = tmp_path / "app.py"
    app_file.write_text("def hello(): return 1\n", encoding="utf-8")

    store = CompletionEvidenceStore(tmp_path, max_entries=50)
    ev = store.record(
        task_id="task-1",
        subtask_id="sub-1",
        turn_number=1,
        stage="testing",
        evidence_type=EvidenceType.TEST_EXECUTION,
        source="command_runner",
        target_paths=["app.py"],
        command=["pytest"],
        exit_code=0,
    )

    assert ev.is_valid is True
    assert len(store.get_valid_evidence()) == 1

    # Mutate app.py content
    app_file.write_text("def hello(): return 2 # modified\n", encoding="utf-8")
    invalidated = store.invalidate_on_file_mutation(["app.py"], reason="repaired")

    assert ev.evidence_id in invalidated
    assert ev.status == EvidenceStatus.INVALIDATED.value
    assert ev.invalidation_reason == "repaired"
    assert len(store.get_valid_evidence()) == 0


# -----------------------------------------------------------------------------
# Test 4: Hard Completion Gates & Deterministic Decision Engine
# -----------------------------------------------------------------------------

def test_hard_completion_gates_evaluation(tmp_path: Path):
    calc_file = tmp_path / "calc.py"
    calc_file.write_text("def add(a, b): return a + b\n", encoding="utf-8")
    fs = ProjectFilesystem(tmp_path)
    engine = CompletionDecisionEngine(fs)
    store = CompletionEvidenceStore(tmp_path)

    task = make_test_task("task-math", "Implement addition")
    plan = Plan(objective="Implement addition", files_likely_to_change=["calc.py"])
    ops = [FileOperation(action="modify", path="calc.py", content="def add(a, b): return a + b\n")]
    diff = "--- a/calc.py\n+++ b/calc.py\n@@ -1 +1 @@\n+def add(a, b): return a + b\n"

    # Case 1: No evidence -> Not ready
    assessment = engine.evaluate(task, None, plan, store, ops, diff)
    assert assessment.is_ready is False
    assert assessment.readiness_level == ReadinessLevel.NOT_READY.value

    # Case 2: Add passing test and review evidence -> Ready
    store.record(
        task_id="task-math",
        subtask_id="main",
        turn_number=1,
        stage="testing",
        evidence_type=EvidenceType.TEST_EXECUTION,
        source="command_runner",
        target_paths=["calc.py"],
        exit_code=0,
    )
    store.record(
        task_id="task-math",
        subtask_id="main",
        turn_number=1,
        stage="reviewing",
        evidence_type=EvidenceType.CODE_REVIEW,
        source="reviewer",
        target_paths=["calc.py"],
        payload={"verdict": "APPROVED", "summary": "Looks good"},
    )

    assessment2 = engine.evaluate(
        task, None, plan, store, ops, diff,
        last_review=ReviewResult(verdict="APPROVED", summary="Looks good", findings=[])
    )
    assert assessment2.is_ready is True
    assert assessment2.readiness_level == ReadinessLevel.READY.value
    assert len(assessment2.supporting_evidence_ids) >= 2
    assert "1_what_was_changed" in assessment2.answers_to_ten_questions
    assert "calc.py" in assessment2.answers_to_ten_questions["1_what_was_changed"]
    assert assessment2.answers_to_ten_questions["10_justified_readiness_level"] == "READY"


# -----------------------------------------------------------------------------
# Test 5: Protected File Violation Fails Hard Gate (Blocked)
# -----------------------------------------------------------------------------

def test_protected_file_violation_blocks_readiness(tmp_path: Path):
    fs = ProjectFilesystem(tmp_path)
    engine = CompletionDecisionEngine(fs)
    store = CompletionEvidenceStore(tmp_path)
    task = make_test_task("task-prot", "Modify approval engine")
    plan = Plan(objective="Modify approval engine")

    ops = [FileOperation(action="modify", path="local_agent/approval.py", content="pass")]
    diff = "+pass"

    assessment = engine.evaluate(task, None, plan, store, ops, diff)
    assert assessment.is_ready is False
    assert assessment.readiness_level == ReadinessLevel.BLOCKED.value
    prot_gate = next(g for g in assessment.gates_evaluated if g.gate_name == "GATE_PROTECTED_FILES_INTACT")
    assert prot_gate.passed is False


# -----------------------------------------------------------------------------
# Test 6: Python Syntax Error Blocks Readiness
# -----------------------------------------------------------------------------

def test_syntax_error_blocks_readiness(tmp_path: Path):
    broken_file = tmp_path / "broken.py"
    broken_file.write_text("def broken(:\n", encoding="utf-8")
    fs = ProjectFilesystem(tmp_path)
    engine = CompletionDecisionEngine(fs)
    store = CompletionEvidenceStore(tmp_path)
    task = make_test_task("task-syn", "Fix code")
    plan = Plan(objective="Fix code")

    ops = [FileOperation(action="modify", path="broken.py", content="def broken(:\n")]
    diff = "+def broken(:\n"

    assessment = engine.evaluate(task, None, plan, store, ops, diff)
    assert assessment.is_ready is False
    assert assessment.readiness_level == ReadinessLevel.BLOCKED.value
    syn_gate = next(g for g in assessment.gates_evaluated if g.gate_name == "GATE_SYNTAX_CLEAN")
    assert syn_gate.passed is False


# -----------------------------------------------------------------------------
# Test 7: Multi-Turn End-to-End Evidence-Backed Completion
# -----------------------------------------------------------------------------

def test_multi_turn_evidence_backed_completion(tmp_path: Path):
    src_file = tmp_path / "calc.py"
    src_file.write_text("def add(): return 0\n", encoding="utf-8")

    fs = ProjectFilesystem(tmp_path)
    reg = ToolRegistry(tmp_path, filesystem=fs)
    storage = JsonFileStorage(tmp_path / ".agent")
    runner = DummyCommandRunner([
        ExecutionResult(command="pytest", exit_code=0, stdout="1 passed", stderr=""),
        ExecutionResult(command="pytest", exit_code=0, stdout="1 passed", stderr=""),
    ])

    config = AgentConfig(
        project=str(tmp_path),
        multi_turn_implementation=True,
        validation_commands=["pytest"],
    )

    ops = [FileOperation(action="modify", path="calc.py", content="def add(a, b): return a + b\n")]
    provider = DummyProvider(code_ops=ops, review_verdict="APPROVED")

    agent = MultiTurnImplementationAgent(config, fs, reg, storage, runner)
    task = make_test_task("task-e2e-evidence", "Implement add")
    plan = Plan(objective="Implement add", files_likely_to_change=["calc.py"])
    context = ProjectContext(str(tmp_path))

    report = agent.execute(task, None, plan, context, provider)

    assert report.success is True
    assert report.final_state == MultiTurnState.COMPLETED.value
    assert report.completion_assessment is not None
    assert report.completion_assessment.is_ready is True
    assert report.completion_assessment.readiness_level == ReadinessLevel.READY.value
    assert len(report.completion_assessment.supporting_evidence_ids) >= 2
    assert "calc.py" in report.completion_assessment.answers_to_ten_questions["1_what_was_changed"]


# -----------------------------------------------------------------------------
# Test 8: Stale Evidence Invalidated Across Repair Turn
# -----------------------------------------------------------------------------

def test_stale_evidence_invalidated_across_repair(tmp_path: Path):
    src_file = tmp_path / "calc.py"
    src_file.write_text("def add(): return 0\n", encoding="utf-8")

    fs = ProjectFilesystem(tmp_path)
    reg = ToolRegistry(tmp_path, filesystem=fs)
    storage = JsonFileStorage(tmp_path / ".agent")

    runner = DummyCommandRunner([
        ExecutionResult(command="pytest", exit_code=1, stdout="", stderr="Failed"),
        ExecutionResult(command="pytest", exit_code=0, stdout="Passed", stderr=""),
        ExecutionResult(command="pytest", exit_code=0, stdout="Passed", stderr=""),
    ])

    config = AgentConfig(
        project=str(tmp_path),
        multi_turn_implementation=True,
        validation_commands=["pytest"],
    )

    op1 = [FileOperation(action="modify", path="calc.py", content="def add(a): return a\n")]
    op2 = [FileOperation(action="modify", path="calc.py", content="def add(a, b): return a + b\n")]
    provider = DummyProvider(responses=[op1, op2], review_verdict="APPROVED")

    agent = MultiTurnImplementationAgent(config, fs, reg, storage, runner)
    task = make_test_task("task-repair-ev", "Fix add")
    plan = Plan(objective="Fix add", files_likely_to_change=["calc.py"])
    context = ProjectContext(str(tmp_path))

    report = agent.execute(task, None, plan, context, provider)

    assert report.success is True
    assert report.repair_turns == 1
    assert report.completion_assessment is not None
    assert report.completion_assessment.is_ready is True
    assert len(report.completion_assessment.invalidated_evidence_ids) >= 1


# -----------------------------------------------------------------------------
# Test 9: Checkpoint Evidence and Assessment Rehydration Roundtrip
# -----------------------------------------------------------------------------

def test_checkpoint_evidence_and_assessment_rehydration(tmp_path: Path):
    storage = JsonFileStorage(tmp_path / ".agent")
    now = datetime.datetime.now(datetime.timezone.utc)

    assessment = CompletionAssessment(
        task_id="task-cp",
        subtask_id="sub-1",
        readiness_level=ReadinessLevel.READY.value,
        is_ready=True,
        decision_reason="Verified",
        supporting_evidence_ids=["ev-1"],
    )

    cp = Checkpoint(
        checkpoint_id="cp-test-evidence",
        task_id="task-cp",
        subtask_id="sub-1",
        timestamp=now,
        current_state_description="Verified checkpoint",
        files_changed=["app.py"],
        completion_assessment=assessment.to_dict(),
        completion_evidence={"entries": [{"evidence_id": "ev-1", "task_id": "task-cp"}]},
    )

    storage.save_checkpoint(cp)
    loaded = storage.load_checkpoint("cp-test-evidence")

    assert loaded is not None
    assert loaded.completion_assessment is not None
    assert loaded.completion_assessment["readiness_level"] == ReadinessLevel.READY.value
    assert loaded.completion_evidence["entries"][0]["evidence_id"] == "ev-1"


# -----------------------------------------------------------------------------
# Test 10: Strict 0 Diff on Protected Files
# -----------------------------------------------------------------------------

def test_protected_files_strict_0_diff():
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        ["git", "diff", "--", "local_agent/tool_engine.py", "local_agent/approval.py"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "", f"Protected files must have 0 diff:\n{result.stdout}"
