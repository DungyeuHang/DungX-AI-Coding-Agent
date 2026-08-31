"""Phase 4.19: Comprehensive Completion Enforcement & Release-Lifecycle Integration Tests.

Adversarially audits all 10 architectural completion invariants, failure-closed
guarantees, disk revalidation, secret sanitization, and orchestrator lifecycle enforcement.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from local_agent.completion import (
    CompletionAssessment,
    CompletionDecisionEngine,
    CompletionEvidenceStore,
    EvidenceStatus,
    EvidenceTrustTier,
    EvidenceType,
    ReadinessLevel,
    StructuredEvidence,
    sanitize_evidence_payload,
    sanitize_text,
)
from local_agent.filesystem import ProjectFilesystem
from local_agent.models import (
    Checkpoint,
    ClarificationRequest,
    ExecutionResult,
    FailureAnalysis,
    FileOperation,
    MultiTurnExecutionReport,
    MultiTurnState,
    Plan,
    ProjectContext,
    ReviewResult,
    RunReport,
    Subtask,
    SubtaskStatus,
    Task,
    TaskStatus,
)
from local_agent.orchestrator import Orchestrator
from local_agent.storage import JsonFileStorage


def make_test_task(task_id: str = "task-001", objective: str = "Test objective") -> Task:
    now = datetime.datetime.now(datetime.timezone.utc)
    return Task(
        task_id=task_id,
        objective=objective,
        status=TaskStatus.PENDING,
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
def temp_project(tmp_path: Path) -> tuple[ProjectFilesystem, Path]:
    src = tmp_path / "src"
    src.mkdir(parents=True, exist_ok=True)
    (src / "calculator.py").write_text("def add(a: int, b: int) -> int:\n    return a + b\n", encoding="utf-8")
    (src / "main.py").write_text("import sys\nprint('hello')\n", encoding="utf-8")
    fs = ProjectFilesystem(tmp_path)
    return fs, tmp_path


def test_invariant1_no_validation_evidence_refuses_completion(temp_project):
    fs, root = temp_project
    engine = CompletionDecisionEngine(fs)
    store = CompletionEvidenceStore(root)

    task = make_test_task("task-001", "Implement subtraction")
    plan = Plan(objective="Implement subtraction", files_likely_to_change=["src/calculator.py"])
    ops = [FileOperation(action="modify", path="src/calculator.py")]
    diff = "--- a/src/calculator.py\n+++ b/src/calculator.py\n@@ -1 +1,2 @@\n+def sub(a, b): return a - b"

    assessment = engine.evaluate(
        task=task,
        subtask=None,
        plan=plan,
        evidence_store=store,
        applied_operations=ops,
        current_diff=diff,
    )

    assert assessment.is_ready is False
    assert assessment.readiness_level in (ReadinessLevel.NOT_READY.value, ReadinessLevel.PARTIALLY_VERIFIED.value)
    assert "Authoritative workspace diff" not in assessment.missing_evidence


def test_invariant2_mutation_after_validation_invalidates_evidence(temp_project):
    fs, root = temp_project
    calc_path = root / "src" / "calculator.py"
    calc_path.write_text("def add(a, b): return a + b\n", encoding="utf-8")

    engine = CompletionDecisionEngine(fs)
    store = CompletionEvidenceStore(root)

    # 1. Record valid test execution evidence on initial calculator.py
    ev = store.record(
        task_id="t1",
        subtask_id="s1",
        turn_number=1,
        stage="testing",
        evidence_type=EvidenceType.TEST_EXECUTION,
        source="command_runner",
        target_paths=["src/calculator.py"],
        command=["pytest", "tests/test_calc.py"],
        exit_code=0,
    )
    assert ev.status == EvidenceStatus.VALID.value

    # 2. Mutate file on disk
    calc_path.write_text("def add(a, b): return a + b + 1\n", encoding="utf-8")

    # 3. Disk revalidation must detect mutation and invalidate evidence
    invalidated = store.revalidate_against_disk(fs)
    assert ev.evidence_id in invalidated
    assert ev.status == EvidenceStatus.INVALIDATED.value

    # 4. Completion assessment must refuse readiness due to invalidated test evidence
    task = make_test_task("t1", "Fix add function")
    plan = Plan(objective="Fix add function", files_likely_to_change=["src/calculator.py"])
    ops = [FileOperation(action="modify", path="src/calculator.py")]
    diff = "--- a/src/calculator.py\n+++ b/src/calculator.py\n@@ -1 +1 @@\n+def add(a, b): return a + b + 1"

    assessment = engine.evaluate(
        task=task,
        subtask=None,
        plan=plan,
        evidence_store=store,
        applied_operations=ops,
        current_diff=diff,
    )
    assert assessment.is_ready is False
    assert assessment.readiness_level == ReadinessLevel.NOT_READY.value
    assert ev.evidence_id in assessment.invalidated_evidence_ids


def test_invariant3_protected_file_mutation_fails_closed(temp_project):
    fs, root = temp_project
    engine = CompletionDecisionEngine(fs)
    store = CompletionEvidenceStore(root)

    store.record(
        task_id="t1",
        subtask_id="s1",
        turn_number=1,
        stage="testing",
        evidence_type=EvidenceType.TEST_EXECUTION,
        source="command_runner",
        command=["pytest"],
        exit_code=0,
    )

    task = make_test_task("t1", "Modify tool engine")
    plan = Plan(objective="Modify tool engine", files_likely_to_change=["local_agent/tool_engine.py"])
    ops = [FileOperation(action="modify", path="local_agent/tool_engine.py")]
    diff = "--- a/local_agent/tool_engine.py\n+++ b/local_agent/tool_engine.py\n@@ -1 +1 @@\n+# changed"

    assessment = engine.evaluate(
        task=task,
        subtask=None,
        plan=plan,
        evidence_store=store,
        applied_operations=ops,
        current_diff=diff,
    )

    assert assessment.is_ready is False
    assert assessment.readiness_level == ReadinessLevel.BLOCKED.value
    gate_prot = next(g for g in assessment.gates_evaluated if g.gate_name == "GATE_PROTECTED_FILES_INTACT")
    assert gate_prot.passed is False
    assert "Protected file modification detected" in gate_prot.reason


def test_invariant4_unresolved_failure_refuses_completion(temp_project):
    fs, root = temp_project
    engine = CompletionDecisionEngine(fs)
    store = CompletionEvidenceStore(root)

    task = make_test_task("t1", "Repair defect")
    plan = Plan(objective="Repair defect", files_likely_to_change=["src/calculator.py"])
    ops = [FileOperation(action="modify", path="src/calculator.py")]
    diff = "diff content"

    failure = FailureAnalysis(
        probable_root_cause="IndexError in array lookup",
        recommended_fix="Add boundary check",
    )

    assessment = engine.evaluate(
        task=task,
        subtask=None,
        plan=plan,
        evidence_store=store,
        applied_operations=ops,
        current_diff=diff,
        last_failure=failure,
    )

    assert assessment.is_ready is False
    gate_fail = next(g for g in assessment.gates_evaluated if g.gate_name == "GATE_NO_UNRESOLVED_FAILURES")
    assert gate_fail.passed is False
    assert "IndexError in array lookup" in gate_fail.reason


def test_invariant5_pending_clarification_refuses_completion(temp_project):
    fs, root = temp_project
    engine = CompletionDecisionEngine(fs)
    store = CompletionEvidenceStore(root)

    store.record(
        task_id="t1",
        subtask_id="s1",
        turn_number=1,
        stage="testing",
        evidence_type=EvidenceType.TEST_EXECUTION,
        source="command_runner",
        command=["pytest"],
        exit_code=0,
    )

    task = make_test_task("t1", "Ambiguous requirement")
    plan = Plan(objective="Ambiguous requirement", files_likely_to_change=["src/calculator.py"])
    ops = [FileOperation(action="modify", path="src/calculator.py")]
    diff = "diff content"

    pending_req = ClarificationRequest(
        question_id="q1",
        task_id="t1",
        subtask_id="s1",
        question="Should float division truncate?",
        status="pending",
    )

    assessment = engine.evaluate(
        task=task,
        subtask=None,
        plan=plan,
        evidence_store=store,
        applied_operations=ops,
        current_diff=diff,
        clarification_requests=[pending_req],
    )

    assert assessment.is_ready is False
    gate_clar = next(g for g in assessment.gates_evaluated if g.gate_name == "GATE_NO_PENDING_CLARIFICATIONS")
    assert gate_clar.passed is False
    assert "Should float division truncate?" in gate_clar.reason


def test_invariant6_review_of_old_diff_rejected_on_diff_mutation(temp_project):
    fs, root = temp_project
    engine = CompletionDecisionEngine(fs)
    store = CompletionEvidenceStore(root)

    diff_a = "--- a/src/calculator.py\n+++ b/src/calculator.py\n@@ -1 +1 @@\n+def add(a, b): return a + b"
    diff_a_hash = hashlib.sha256(diff_a.encode("utf-8")).hexdigest()[:16]

    # Valid test execution
    store.record(
        task_id="t1",
        subtask_id="s1",
        turn_number=1,
        stage="testing",
        evidence_type=EvidenceType.TEST_EXECUTION,
        source="command_runner",
        command=["pytest"],
        exit_code=0,
    )

    # Approved review on diff A
    store.record(
        task_id="t1",
        subtask_id="s1",
        turn_number=1,
        stage="reviewing",
        evidence_type=EvidenceType.CODE_REVIEW,
        source="reviewer",
        trust_tier=EvidenceTrustTier.DELIBERATIVE_REVIEW,
        payload={"verdict": "APPROVED", "summary": "Looks good", "diff_hash": diff_a_hash},
    )

    # Now workspace diff changes to diff B
    diff_b = "--- a/src/calculator.py\n+++ b/src/calculator.py\n@@ -1 +1 @@\n+def add(a, b): return a + b + 99"
    task = make_test_task("t1", "Update add")
    plan = Plan(objective="Update add", files_likely_to_change=["src/calculator.py"])
    ops = [FileOperation(action="modify", path="src/calculator.py")]

    review = ReviewResult(verdict="APPROVED", summary="Looks good")

    assessment = engine.evaluate(
        task=task,
        subtask=None,
        plan=plan,
        evidence_store=store,
        applied_operations=ops,
        current_diff=diff_b,
        last_review=review,
    )

    assert assessment.is_ready is False
    gate_rev = next(g for g in assessment.gates_evaluated if g.gate_name == "GATE_REVIEW_APPROVED")
    assert gate_rev.passed is False
    assert "Review diff hash mismatch" in gate_rev.reason


def test_invariant7_failed_validation_dominates_and_blocks_completion(temp_project):
    fs, root = temp_project
    engine = CompletionDecisionEngine(fs)
    store = CompletionEvidenceStore(root)

    # 1. An older test passed
    store.record(
        task_id="t1",
        subtask_id="s1",
        turn_number=1,
        stage="testing",
        evidence_type=EvidenceType.TEST_EXECUTION,
        source="command_runner",
        command=["pytest", "tests/unit"],
        exit_code=0,
    )

    # 2. A newer or concurrent test failed
    store.record(
        task_id="t1",
        subtask_id="s1",
        turn_number=2,
        stage="testing",
        evidence_type=EvidenceType.TEST_EXECUTION,
        source="command_runner",
        command=["pytest", "tests/integration"],
        exit_code=1,
    )

    task = make_test_task("t1", "Test execution")
    plan = Plan(objective="Test execution", files_likely_to_change=["src/calculator.py"])
    ops = [FileOperation(action="modify", path="src/calculator.py")]
    diff = "diff content"

    assessment = engine.evaluate(
        task=task,
        subtask=None,
        plan=plan,
        evidence_store=store,
        applied_operations=ops,
        current_diff=diff,
    )

    assert assessment.is_ready is False
    gate_val = next(g for g in assessment.gates_evaluated if g.gate_name == "GATE_VALIDATION_PASSED")
    assert gate_val.passed is False
    assert "active test failure" in gate_val.reason


def test_invariant8_checkpoint_resume_disk_mutation_downgrades_readiness(temp_project):
    fs, root = temp_project
    calc_path = root / "src" / "calculator.py"
    calc_path.write_text("def add(a, b): return a + b\n", encoding="utf-8")

    store = CompletionEvidenceStore(root)
    ev = store.record(
        task_id="t1",
        subtask_id="s1",
        turn_number=1,
        stage="testing",
        evidence_type=EvidenceType.TEST_EXECUTION,
        source="command_runner",
        target_paths=["src/calculator.py"],
        command=["pytest"],
        exit_code=0,
    )

    # Checkpoint saved with READY assessment
    assessment = CompletionAssessment(
        task_id="t1",
        subtask_id="s1",
        readiness_level=ReadinessLevel.READY.value,
        is_ready=True,
        decision_reason="All passed",
        supporting_evidence_ids=[ev.evidence_id],
    )

    cp = Checkpoint(
        checkpoint_id="cp-101",
        task_id="t1",
        subtask_id="s1",
        timestamp=datetime.datetime.now(datetime.timezone.utc),
        current_state_description="Ready state",
        files_changed=["src/calculator.py"],
        completion_assessment=assessment.to_dict(),
        completion_evidence=store.to_dict(),
    )

    storage = JsonFileStorage(root / ".agent_storage")
    storage.save_checkpoint(cp)

    # Now disk changes after checkpoint
    calc_path.write_text("def add(a, b): return a * b # modified after cp\n", encoding="utf-8")

    # Resuming and revalidating from storage
    loaded_cp = storage.load_checkpoint("cp-101")
    restored_store = CompletionEvidenceStore.from_dict(loaded_cp.completion_evidence)
    invalidated_ids = restored_store.revalidate_against_disk(fs)

    assert ev.evidence_id in invalidated_ids
    assert len(restored_store.get_valid_evidence(EvidenceType.TEST_EXECUTION)) == 0


def test_invariant9_secret_redaction_in_evidence_payloads(temp_project):
    fs, root = temp_project
    store = CompletionEvidenceStore(root)

    secret_output = (
        "Execution output:\n"
        "OPENAI_KEY=sk-1234567890abcdef1234567890abcdef\n"
        "ANTHROPIC_KEY=sk-ant-1234567890abcdef1234567890abcdef\n"
        "GITHUB_TOKEN=ghp_123456789012345678901234567890123456\n"
        "Authorization: Bearer secret-token-value-1234567890\n"
    )

    ev = store.record(
        task_id="t1",
        subtask_id="s1",
        turn_number=1,
        stage="testing",
        evidence_type=EvidenceType.TEST_EXECUTION,
        source="command_runner",
        command=["python", "test.py"],
        exit_code=0,
        payload={"stdout": secret_output, "nested": {"key": "sk-1234567890abcdef1234567890abcdef"}},
    )

    serialized = ev.to_dict()
    payload_str = json.dumps(serialized)

    assert "sk-1234567890" not in payload_str
    assert "sk-ant-1234567890" not in payload_str
    assert "ghp_1234567890" not in payload_str
    assert "[REDACTED_SECRET]" in payload_str


def test_invariant10_orchestrator_refuses_completion_on_syntax_error_despite_approved_review(temp_project):
    fs, root = temp_project
    broken_file = root / "src" / "broken.py"
    broken_file.write_text("def broken_syntax(:\n    pass\n", encoding="utf-8")

    engine = CompletionDecisionEngine(fs)
    store = CompletionEvidenceStore(root)

    # Valid test passed
    store.record(
        task_id="t1",
        subtask_id="s1",
        turn_number=1,
        stage="testing",
        evidence_type=EvidenceType.TEST_EXECUTION,
        source="command_runner",
        command=["pytest"],
        exit_code=0,
    )

    # Review approved
    diff = "--- /dev/null\n+++ b/src/broken.py\n@@ -0,0 +1,2 @@\n+def broken_syntax(:\n+    pass\n"
    diff_hash = hashlib.sha256(diff.encode("utf-8")).hexdigest()[:16]
    store.record(
        task_id="t1",
        subtask_id="s1",
        turn_number=1,
        stage="reviewing",
        evidence_type=EvidenceType.CODE_REVIEW,
        source="reviewer",
        payload={"verdict": "APPROVED", "summary": "Approved", "diff_hash": diff_hash},
    )

    task = make_test_task("t1", "Add broken function")
    plan = Plan(objective="Add broken function", files_likely_to_create=["src/broken.py"])
    ops = [FileOperation(action="create", path="src/broken.py")]
    review = ReviewResult(verdict="APPROVED", summary="Approved")

    assessment = engine.evaluate(
        task=task,
        subtask=None,
        plan=plan,
        evidence_store=store,
        applied_operations=ops,
        current_diff=diff,
        last_review=review,
    )

    assert assessment.is_ready is False
    assert assessment.readiness_level == ReadinessLevel.BLOCKED.value
    gate_syn = next(g for g in assessment.gates_evaluated if g.gate_name == "GATE_SYNTAX_CLEAN")
    assert gate_syn.passed is False
    assert "Syntax errors" in gate_syn.reason


def test_protected_files_strict_0_diff():
    res = subprocess.run(
        ["git", "diff", "--", "local_agent/tool_engine.py", "local_agent/approval.py"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert res.stdout.strip() == "", f"Protected files modified: {res.stdout}"
