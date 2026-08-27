"""Behavioral tests for Phase 4.5 Autonomous Recovery and Anti-Repeat Intelligence."""

import os
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from local_agent.config import AgentConfig
from local_agent.context import ContextSelector
from local_agent.filesystem import ProjectFilesystem
from local_agent.models import (
    Checkpoint,
    CommandSpec,
    ExecutionResult,
    FailureAnalysis,
    FileOperation,
    Plan,
    PreparedChange,
    ProjectContext,
    ProviderCapability,
    RecoveryState,
    RepairSignature,
    ReviewResult,
    RunReport,
    Task,
    TaskStatus,
    ToolCall,
    ToolExecutionMetrics,
    ToolResult,
    ValidationPlan,
    normalize_diff_for_signature,
)
from local_agent.orchestrator import Orchestrator
from local_agent.providers import AIProvider, MockProvider
from local_agent.storage import JsonFileStorage


class DummyRecoverableProvider(AIProvider):
    def __init__(self, code_responses=None, failure_responses=None, review_responses=None):
        self.code_responses = list(code_responses or [])
        self.failure_responses = list(failure_responses or [])
        self.review_responses = list(review_responses or [])
        self.code_call_count = 0
        self.failure_call_count = 0
        self.review_call_count = 0
        self.received_failures = []
        self.received_reviews = []

    @property
    def capabilities(self):
        return {
            ProviderCapability.PLANNING,
            ProviderCapability.IMPLEMENTATION,
            ProviderCapability.REPAIR,
            ProviderCapability.REVIEW,
        }

    def generate_plan(self, task, context):
        return Plan(
            objective=str(task),
            files_likely_to_change=["src/core.py", "src/helper.py"],
            steps=["Step 1"],
            validation_strategy=["pytest tests/test_core.py"],
        )

    def generate_code(self, task, plan, context, failure=None, review=None):
        self.code_call_count += 1
        if failure:
            self.received_failures.append(failure)
        if review:
            self.received_reviews.append(review)
        if self.code_responses:
            return self.code_responses.pop(0)
        return [FileOperation("modify", "src/core.py", content="def run():\n    return 42\n")]

    def analyze_failure(self, execution, diff, context, plan):
        self.failure_call_count += 1
        if self.failure_responses:
            return self.failure_responses.pop(0)
        return FailureAnalysis(
            probable_root_cause=f"Command failed: {execution.command}",
            affected_files=["src/core.py"],
            recommended_fix="Fix core.py",
            category="VALIDATION_FAILURE",
        )

    def review_changes(self, task, plan, diff, context):
        self.review_call_count += 1
        if self.review_responses:
            return self.review_responses.pop(0)
        return ReviewResult("APPROVED", "Looks great")


class AutonomousRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_path = Path(self.temp_dir.name)
        (self.project_path / "src").mkdir(parents=True, exist_ok=True)
        (self.project_path / "tests").mkdir(parents=True, exist_ok=True)
        (self.project_path / "src" / "core.py").write_text("def run():\n    return 42\n", encoding="utf-8")
        (self.project_path / "tests" / "test_core.py").write_text("def test_run():\n    assert True\n", encoding="utf-8")

        self.config = AgentConfig(project=self.project_path, max_iterations=3)
        self.storage = JsonFileStorage(self.project_path)
        self.filesystem = ProjectFilesystem(self.project_path)
        self.repo_lock = threading.Lock()
        self.memory_lock = threading.Lock()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _build_orchestrator(self, config=None):
        cfg = config or self.config
        orch = Orchestrator(cfg, self.storage, scheduler=None, repo_lock=self.repo_lock, memory_lock=self.memory_lock)
        return orch

    def test_recovery_state_defaults(self):
        rec = RecoveryState()
        self.assertEqual(rec.completed_iterations, 0)
        self.assertEqual(len(rec.repair_signatures), 0)
        self.assertEqual(len(rec.failure_history), 0)
        self.assertEqual(len(rec.review_history), 0)
        self.assertEqual(rec.consecutive_same_failure_count, 0)
        self.assertEqual(rec.abort_reason, "")

    def test_repair_signature_normalization(self):
        diff1 = "--- a/src/core.py\t2026-08-27 10:00:00\n+++ b/src/core.py\t2026-08-27 10:01:00\n@@ -1,2 +1,2 @@\n-old line\n+new line\n"
        diff2 = "--- a/src/core.py\n+++ b/src/core.py\n@@ -10,2 +10,2 @@\n-old line   \n+new line\n"
        hash1 = normalize_diff_for_signature(diff1)
        hash2 = normalize_diff_for_signature(diff2)
        self.assertEqual(hash1, hash2)
        self.assertTrue(len(hash1) > 0)

    def test_distinct_files_produce_different_diff_signatures(self):
        diff1 = "--- a/src/core.py\n+++ b/src/core.py\n@@ -1,1 +1,1 @@\n-a\n+b\n"
        diff2 = "--- a/src/helper.py\n+++ b/src/helper.py\n@@ -1,1 +1,1 @@\n-a\n+b\n"
        hash1 = normalize_diff_for_signature(diff1)
        hash2 = normalize_diff_for_signature(diff2)
        self.assertNotEqual(hash1, hash2)

    def test_equivalent_repair_detection(self):
        rec = RecoveryState()
        diff = "--- a/src/core.py\n+++ b/src/core.py\n@@ -1,1 +1,1 @@\n-a\n+b\n"
        sig = rec.record_attempt(1, None, diff, ["src/core.py"])
        self.assertTrue(rec.is_duplicate_patch(sig.patch_hash))
        self.assertTrue(rec.has_duplicate_signature(sig))

    def test_distinct_repair_detection(self):
        rec = RecoveryState()
        diff1 = "--- a/src/core.py\n+++ b/src/core.py\n@@ -1,1 +1,1 @@\n-a\n+b\n"
        diff2 = "--- a/src/core.py\n+++ b/src/core.py\n@@ -1,1 +1,1 @@\n-a\n+c\n"
        rec.record_attempt(1, None, diff1, ["src/core.py"])
        hash2 = normalize_diff_for_signature(diff2)
        self.assertFalse(rec.is_duplicate_patch(hash2))

    def test_first_repair_attempt_is_allowed(self):
        rec = RecoveryState()
        diff = "--- a/src/core.py\n+++ b/src/core.py\n@@ -1,1 +1,1 @@\n-old\n+fixed\n"
        hash_val = normalize_diff_for_signature(diff)
        self.assertFalse(rec.is_duplicate_patch(hash_val))

    def test_repeated_repair_is_blocked(self):
        rec = RecoveryState()
        diff = "--- a/src/core.py\n+++ b/src/core.py\n@@ -1,1 +1,1 @@\n-old\n+fixed\n"
        rec.record_attempt(1, None, diff, ["src/core.py"])
        hash_val = normalize_diff_for_signature(diff)
        self.assertTrue(rec.is_duplicate_patch(hash_val))

    def test_failure_history_accumulation(self):
        rec = RecoveryState()
        f1 = FailureAnalysis("ZeroDivisionError in line 10", ["src/core.py"], "Check denominator")
        f2 = FailureAnalysis("KeyError token", ["src/core.py"], "Check auth dict")
        rec.record_failure(f1)
        rec.record_failure(f2)
        self.assertEqual(len(rec.failure_history), 2)
        self.assertEqual(rec.failure_history[0].probable_root_cause, "ZeroDivisionError in line 10")
        self.assertEqual(rec.failure_history[1].probable_root_cause, "KeyError token")

    def test_failure_history_bounded_summary(self):
        rec = RecoveryState()
        for i in range(20):
            f = FailureAnalysis(f"Extremely large error cause description {i} " * 50, ["src/core.py"], "Fix advice " * 50)
            rec.record_failure(f)
            rec.record_attempt(i + 1, f, f"--- a/file{i}\n+++ b/file{i}\n+line", [f"file{i}"])
        summary = rec.build_recovery_summary(max_chars=1500)
        self.assertTrue(len(summary) <= 1500)
        self.assertIn("Previous Attempts:", summary)
        self.assertIn("Guidance:", summary)

    def test_review_rejection_history_accumulation(self):
        rec = RecoveryState()
        r1 = ReviewResult("CHANGES_REQUESTED", "Need better error handling", ["Handle ValueError"])
        r2 = ReviewResult("CHANGES_REQUESTED", "Missing docstrings", ["Add docstrings"])
        rec.record_review(r1)
        rec.record_review(r2)
        self.assertEqual(len(rec.review_history), 2)
        self.assertEqual(rec.review_history[0].summary, "Need better error handling")

    def test_repeated_review_rejection_summary(self):
        rec = RecoveryState()
        r = ReviewResult("CHANGES_REQUESTED", "Security issue", ["SQL injection risk"])
        rec.record_review(r)
        summary = rec.build_recovery_summary()
        self.assertIn("Latest Review Feedback: Security issue", summary)
        self.assertIn("SQL injection risk", summary)

    def test_iteration_accounting(self):
        rec = RecoveryState()
        rec.completed_iterations = 2
        d = rec.to_dict()
        restored = RecoveryState.from_dict(d)
        self.assertEqual(restored.completed_iterations, 2)

    def test_checkpoint_serialization_roundtrip(self):
        rec = RecoveryState(completed_iterations=1, abort_reason="TEST_ABORT")
        rec.record_failure(FailureAnalysis("Test cause", ["src/core.py"], "Fix it"))
        rec.record_failure(FailureAnalysis("Test cause", ["src/core.py"], "Fix it"))
        rec.record_review(ReviewResult("CHANGES_REQUESTED", "Not approved", ["Fix logic"]))
        rec.record_attempt(1, None, "--- diff", ["src/core.py"])

        data = rec.to_dict()
        restored = RecoveryState.from_dict(data)
        self.assertEqual(restored.completed_iterations, 1)
        self.assertEqual(restored.abort_reason, "TEST_ABORT")
        self.assertEqual(restored.consecutive_same_failure_count, 2)
        self.assertEqual(len(restored.failure_history), 2)
        self.assertEqual(len(restored.review_history), 1)
        self.assertEqual(len(restored.repair_signatures), 1)

    def test_backward_compatible_checkpoint_loading(self):
        empty_data = {}
        restored = RecoveryState.from_dict(empty_data)
        self.assertEqual(restored.completed_iterations, 0)
        self.assertEqual(len(restored.repair_signatures), 0)

        none_data = None
        restored_none = RecoveryState.from_dict(none_data)
        self.assertEqual(restored_none.completed_iterations, 0)

    def test_consecutive_same_failure_tracking(self):
        rec = RecoveryState()
        f1 = FailureAnalysis("Exact same error", ["src/core.py"], "Fix")
        f2 = FailureAnalysis("Exact same error", ["src/core.py"], "Fix")
        f3 = FailureAnalysis("Different error", ["src/core.py"], "Fix")

        rec.record_failure(f1)
        self.assertEqual(rec.consecutive_same_failure_count, 1)
        rec.record_failure(f2)
        self.assertEqual(rec.consecutive_same_failure_count, 2)
        rec.record_failure(f3)
        self.assertEqual(rec.consecutive_same_failure_count, 1)

    def test_orchestrator_populates_recovery_state(self):
        provider = DummyRecoverableProvider()
        orchestrator = self._build_orchestrator()
        with patch("local_agent.orchestrator.build_provider", return_value=provider):
            report = orchestrator.run("Simple task")
        self.assertIsNotNone(report.recovery_state)
        self.assertTrue(isinstance(report.recovery_state, RecoveryState))
        self.assertEqual(report.recovery_state.completed_iterations, 1)

    def test_anti_repeat_blocks_duplicate_patch(self):
        patch_a = "--- a/src/core.py\n+++ b/src/core.py\n@@ -1,2 +1,2 @@\n-def run():\n-    return 42\n+def run():\n+    return 100\n"
        patch_b = "--- a/src/core.py\n+++ b/src/core.py\n@@ -1,2 +1,2 @@\n-def run():\n-    return 100\n+def run():\n+    return 42\n"
        ops1 = [FileOperation("modify", "src/core.py", patch=patch_a)]
        ops2 = [FileOperation("modify", "src/core.py", patch=patch_b)]
        ops3 = [FileOperation("modify", "src/core.py", patch=patch_a)]

        provider = DummyRecoverableProvider(code_responses=[ops1, ops2, ops3])
        orchestrator = self._build_orchestrator()

        with patch("local_agent.orchestrator.build_provider", return_value=provider):
            with patch.object(orchestrator, "_validate") as mock_val:
                mock_val.return_value = [ExecutionResult(("pytest",), 1, "FAILED", "AssertionError: 100 != 42")]
                report = orchestrator.run("Task that fails validation and oscillates")

        self.assertEqual(report.outcome, "REPEATED_REPAIR_DETECTED")
        self.assertIn("REPEATED_REPAIR", [f.category for f in report.failures])

    def test_permitted_distinct_repair_allows_continuation(self):
        ops1 = [FileOperation("modify", "src/core.py", content="def run():\n    return 1\n")]
        ops2 = [FileOperation("modify", "src/core.py", content="def run():\n    return 2\n")]

        provider = DummyRecoverableProvider(code_responses=[ops1, ops2])
        orchestrator = self._build_orchestrator()

        call_count = [0]
        def fake_validate(plan):
            call_count[0] += 1
            if call_count[0] == 1:
                return [ExecutionResult(("pytest",), 1, "FAILED", "1 != 42")]
            return [ExecutionResult(("pytest",), 0, "PASSED", "")]

        with patch("local_agent.orchestrator.build_provider", return_value=provider):
            with patch.object(orchestrator, "_validate", side_effect=fake_validate):
                report = orchestrator.run("Task that succeeds on second try")

        self.assertTrue(report.completed)
        self.assertEqual(report.recovery_state.completed_iterations, 2)
        self.assertEqual(len(report.recovery_state.repair_signatures), 2)

    def test_stagnation_detected_on_consecutive_same_failures(self):
        config = AgentConfig(project=self.project_path, max_iterations=5)
        ops1 = [FileOperation("modify", "src/core.py", content="def run():\n    return 1\n")]
        ops2 = [FileOperation("modify", "src/core.py", content="def run():\n    return 2\n")]
        ops3 = [FileOperation("modify", "src/core.py", content="def run():\n    return 3\n")]

        provider = DummyRecoverableProvider(
            code_responses=[ops1, ops2, ops3],
            failure_responses=[
                FailureAnalysis("Same persistent bug", ["src/core.py"], "Fix"),
                FailureAnalysis("Same persistent bug", ["src/core.py"], "Fix"),
                FailureAnalysis("Same persistent bug", ["src/core.py"], "Fix"),
            ]
        )
        orchestrator = self._build_orchestrator(config)

        with patch("local_agent.orchestrator.build_provider", return_value=provider):
            with patch.object(orchestrator, "_validate") as mock_val:
                mock_val.return_value = [ExecutionResult(("pytest",), 1, "FAILED", "error")]
                report = orchestrator.run("Stagnant task")

        self.assertEqual(report.outcome, "STAGNATION_DETECTED")
        self.assertEqual(report.recovery_state.abort_reason, "STAGNATION_DETECTED")

    def test_repeated_review_rejection_abort(self):
        config = AgentConfig(project=self.project_path, max_iterations=5)
        ops1 = [FileOperation("modify", "src/core.py", content="def run():\n    return 1\n")]
        ops2 = [FileOperation("modify", "src/core.py", content="def run():\n    return 2\n")]
        ops3 = [FileOperation("modify", "src/core.py", content="def run():\n    return 3\n")]

        provider = DummyRecoverableProvider(
            code_responses=[ops1, ops2, ops3],
            review_responses=[
                ReviewResult("CHANGES_REQUESTED", "Reject 1", ["Fix 1"]),
                ReviewResult("CHANGES_REQUESTED", "Reject 2", ["Fix 2"]),
                ReviewResult("CHANGES_REQUESTED", "Reject 3", ["Fix 3"]),
            ]
        )
        orchestrator = self._build_orchestrator(config)

        with patch("local_agent.orchestrator.build_provider", return_value=provider):
            with patch.object(orchestrator, "_validate") as mock_val:
                mock_val.return_value = [ExecutionResult(("pytest",), 0, "PASSED", "")]
                report = orchestrator.run("Task rejected by reviewer 3 times")

        self.assertEqual(report.outcome, "REPEATED_REVIEW_REJECTION")
        self.assertEqual(report.recovery_state.abort_reason, "REPEATED_REVIEW_REJECTION")

    # Critical Audit Scenario 1: Resume completed=2 with max=3 -> runs iteration 3 only
    def test_resume_scenario1_completed_2_of_3(self):
        now = datetime.now(timezone.utc)
        task = Task(task_id="test-scenario-1", objective="Scenario 1", status=TaskStatus.PENDING, created_at=now, updated_at=now)
        self.storage.save_task(task)

        rec = RecoveryState(completed_iterations=2)
        rec.record_attempt(1, None, "--- diff1", ["src/core.py"])
        rec.record_attempt(2, None, "--- diff2", ["src/core.py"])
        chk = Checkpoint(
            checkpoint_id="chk-sc1",
            task_id=task.task_id,
            subtask_id="",
            timestamp=now,
            current_state_description="Paused after iteration 2",
            files_changed=["src/core.py"],
            repository_diff="",
            validation_state={"last_executions": [], "last_failures": []},
            continuation_context={"recovery_state": rec.to_dict()},
        )
        self.storage.save_checkpoint(chk)
        task.latest_checkpoint_id = "chk-sc1"
        self.storage.save_task(task)

        provider = DummyRecoverableProvider()
        orchestrator = self._build_orchestrator()
        with patch("local_agent.orchestrator.build_provider", return_value=provider):
            report = orchestrator.run(task)

        self.assertEqual(report.iterations, 3)
        self.assertEqual(provider.code_call_count, 1)

    # Critical Audit Scenario 2: Resume completed=3 with max=3 -> zero iterations run
    def test_resume_scenario2_completed_equals_max(self):
        now = datetime.now(timezone.utc)
        task = Task(task_id="test-scenario-2", objective="Scenario 2", status=TaskStatus.PENDING, created_at=now, updated_at=now)
        self.storage.save_task(task)

        rec = RecoveryState(completed_iterations=3)
        chk = Checkpoint(
            checkpoint_id="chk-sc2",
            task_id=task.task_id,
            subtask_id="",
            timestamp=now,
            current_state_description="Paused after iteration 3",
            files_changed=["src/core.py"],
            repository_diff="",
            validation_state={"last_executions": [], "last_failures": []},
            continuation_context={"recovery_state": rec.to_dict()},
        )
        self.storage.save_checkpoint(chk)
        task.latest_checkpoint_id = "chk-sc2"
        self.storage.save_task(task)

        provider = DummyRecoverableProvider()
        orchestrator = self._build_orchestrator()
        with patch("local_agent.orchestrator.build_provider", return_value=provider):
            report = orchestrator.run(task)

        self.assertEqual(provider.code_call_count, 0)
        self.assertEqual(report.outcome, "MAX_ITERATIONS_EXCEEDED")

    # Critical Audit Scenario 3: Prior repair history contains patch A -> resume -> agent generates patch A -> REPEATED_REPAIR_DETECTED
    def test_resume_scenario3_repeated_repair_across_resume(self):
        now = datetime.now(timezone.utc)
        task = Task(task_id="test-scenario-3", objective="Scenario 3", status=TaskStatus.PENDING, created_at=now, updated_at=now)
        self.storage.save_task(task)

        patch_a = "--- a/src/core.py\n+++ b/src/core.py\n@@ -1,2 +1,2 @@\n def run():\n-    return 42\n+    return 100\n"
        rec = RecoveryState(completed_iterations=1)
        rec.record_attempt(1, None, patch_a, ["src/core.py"])
        rec.record_failure(FailureAnalysis("Bug in 100", ["src/core.py"], "Fix"))

        chk = Checkpoint(
            checkpoint_id="chk-sc3",
            task_id=task.task_id,
            subtask_id="",
            timestamp=now,
            current_state_description="Paused after iteration 1",
            files_changed=["src/core.py"],
            repository_diff="",
            validation_state={"last_executions": [], "last_failures": []},
            continuation_context={"recovery_state": rec.to_dict()},
        )
        self.storage.save_checkpoint(chk)
        task.latest_checkpoint_id = "chk-sc3"
        self.storage.save_task(task)

        # Agent proposes patch_a again in resumed iteration 2
        ops = [FileOperation("modify", "src/core.py", patch=patch_a)]
        provider = DummyRecoverableProvider(code_responses=[ops])
        orchestrator = self._build_orchestrator()

        with patch("local_agent.orchestrator.build_provider", return_value=provider):
            with patch.object(orchestrator, "_validate") as mock_val:
                mock_val.return_value = [ExecutionResult(("pytest",), 1, "FAILED", "fail")]
                report = orchestrator.run(task)

        self.assertEqual(report.outcome, "REPEATED_REPAIR_DETECTED")
        self.assertEqual(report.recovery_state.abort_reason, "REPEATED_REPAIR_DETECTED")

    # Critical Audit Scenario 4: Prior failure history has 2 consecutive failures -> resume -> 1 more same failure -> STAGNATION_DETECTED
    def test_resume_scenario4_stagnation_across_resume(self):
        now = datetime.now(timezone.utc)
        task = Task(task_id="test-scenario-4", objective="Scenario 4", status=TaskStatus.PENDING, created_at=now, updated_at=now)
        self.storage.save_task(task)

        rec = RecoveryState(completed_iterations=2, consecutive_same_failure_count=2)
        rec.record_failure(FailureAnalysis("Persistent Root Cause", ["src/core.py"], "Fix"))
        rec.record_failure(FailureAnalysis("Persistent Root Cause", ["src/core.py"], "Fix"))

        chk = Checkpoint(
            checkpoint_id="chk-sc4",
            task_id=task.task_id,
            subtask_id="",
            timestamp=now,
            current_state_description="Paused after iteration 2",
            files_changed=["src/core.py"],
            repository_diff="",
            validation_state={"last_executions": [], "last_failures": []},
            continuation_context={"recovery_state": rec.to_dict()},
        )
        self.storage.save_checkpoint(chk)
        task.latest_checkpoint_id = "chk-sc4"
        self.storage.save_task(task)

        ops = [FileOperation("modify", "src/core.py", content="def run():\n    return 3\n")]
        provider = DummyRecoverableProvider(
            code_responses=[ops],
            failure_responses=[FailureAnalysis("Persistent Root Cause", ["src/core.py"], "Fix")]
        )
        orchestrator = self._build_orchestrator()

        with patch("local_agent.orchestrator.build_provider", return_value=provider):
            with patch.object(orchestrator, "_validate") as mock_val:
                mock_val.return_value = [ExecutionResult(("pytest",), 1, "FAILED", "fail")]
                report = orchestrator.run(task)

        self.assertEqual(report.outcome, "STAGNATION_DETECTED")
        self.assertEqual(report.recovery_state.abort_reason, "STAGNATION_DETECTED")

    # Critical Audit Scenario 5: Prior review history has 2 rejections -> resume -> 1 more rejection -> REPEATED_REVIEW_REJECTION
    def test_resume_scenario5_review_rejection_across_resume(self):
        now = datetime.now(timezone.utc)
        task = Task(task_id="test-scenario-5", objective="Scenario 5", status=TaskStatus.PENDING, created_at=now, updated_at=now)
        self.storage.save_task(task)

        rec = RecoveryState(completed_iterations=2)
        rec.record_review(ReviewResult("CHANGES_REQUESTED", "Reject 1", ["Fix 1"]))
        rec.record_review(ReviewResult("CHANGES_REQUESTED", "Reject 2", ["Fix 2"]))

        chk = Checkpoint(
            checkpoint_id="chk-sc5",
            task_id=task.task_id,
            subtask_id="",
            timestamp=now,
            current_state_description="Paused after iteration 2",
            files_changed=["src/core.py"],
            repository_diff="",
            validation_state={"last_executions": [], "last_failures": []},
            continuation_context={"recovery_state": rec.to_dict()},
        )
        self.storage.save_checkpoint(chk)
        task.latest_checkpoint_id = "chk-sc5"
        self.storage.save_task(task)

        ops = [FileOperation("modify", "src/core.py", content="def run():\n    return 3\n")]
        provider = DummyRecoverableProvider(
            code_responses=[ops],
            review_responses=[ReviewResult("CHANGES_REQUESTED", "Reject 3", ["Fix 3"])]
        )
        orchestrator = self._build_orchestrator()

        with patch("local_agent.orchestrator.build_provider", return_value=provider):
            with patch.object(orchestrator, "_validate") as mock_val:
                mock_val.return_value = [ExecutionResult(("pytest",), 0, "PASSED", "")]
                report = orchestrator.run(task)

        self.assertEqual(report.outcome, "REPEATED_REVIEW_REJECTION")
        self.assertEqual(report.recovery_state.abort_reason, "REPEATED_REVIEW_REJECTION")

    def test_checkpoint_continuation_context_contains_recovery_state(self):
        provider = DummyRecoverableProvider()
        orchestrator = self._build_orchestrator()
        with patch("local_agent.orchestrator.build_provider", return_value=provider):
            report = orchestrator.run("Task with checkpoint")
        latest_id = orchestrator.storage.load_task(report.task_id).latest_checkpoint_id
        checkpoint = orchestrator.storage.load_checkpoint(latest_id)
        self.assertIn("recovery_state", checkpoint.continuation_context)
        rec_data = checkpoint.continuation_context["recovery_state"]
        self.assertIn("completed_iterations", rec_data)
        self.assertIn("repair_signatures", rec_data)
        self.assertIn("failure_history", rec_data)

    def test_recovery_summary_injected_into_failure_details(self):
        ops1 = [FileOperation("modify", "src/core.py", content="def run():\n    return 1\n")]
        ops2 = [FileOperation("modify", "src/core.py", content="def run():\n    return 2\n")]

        provider = DummyRecoverableProvider(code_responses=[ops1, ops2])
        orchestrator = self._build_orchestrator()

        call_count = [0]
        def fake_validate(plan):
            call_count[0] += 1
            if call_count[0] == 1:
                return [ExecutionResult(("pytest",), 1, "FAILED", "fail")]
            return [ExecutionResult(("pytest",), 0, "PASSED", "")]

        with patch("local_agent.orchestrator.build_provider", return_value=provider):
            with patch.object(orchestrator, "_validate", side_effect=fake_validate):
                orchestrator.run("Task testing summary injection")

        self.assertTrue(len(provider.received_failures) > 0)
        self.assertIn("recovery_summary", provider.received_failures[0].details)
        self.assertIn("Previous Attempts:", provider.received_failures[0].details["recovery_summary"])

    def test_build_run_report_restores_recovery_state(self):
        now = datetime.now(timezone.utc)
        task = Task(task_id="test-report-snapshot", objective="Snapshot test", status=TaskStatus.PENDING, created_at=now, updated_at=now)
        self.storage.save_task(task)

        rec = RecoveryState(completed_iterations=2, abort_reason="TEST_SNAPSHOT")
        chk = Checkpoint(
            checkpoint_id="chk-snap",
            task_id=task.task_id,
            subtask_id="",
            timestamp=now,
            current_state_description="Snapshot test",
            files_changed=["src/core.py"],
            repository_diff="",
            validation_state={"last_executions": [], "last_failures": []},
            continuation_context={"recovery_state": rec.to_dict()},
        )
        self.storage.save_checkpoint(chk)
        task.latest_checkpoint_id = "chk-snap"
        self.storage.save_task(task)

        orchestrator = self._build_orchestrator()
        report = orchestrator._build_run_report(task)
        self.assertIsNotNone(report.recovery_state)
        self.assertEqual(report.recovery_state.completed_iterations, 2)
        self.assertEqual(report.recovery_state.abort_reason, "TEST_SNAPSHOT")


if __name__ == "__main__":
    unittest.main()
