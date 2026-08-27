"""Comprehensive tests for Phase 4.8 Dynamic Re-Planning & Adaptive Scope Execution."""

import datetime
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from local_agent.approval import ApprovalPolicyEngine
from local_agent.coding_agent import (
    CodingAgent,
    PatchValidationError,
    ScopeAmendmentGuard,
    UnsafeModificationError,
)
from local_agent.commands import CommandRunner
from local_agent.config import AgentConfig
from local_agent.context import ContextSelector
from local_agent.filesystem import ProjectFilesystem, SandboxViolation, SECRET_NAMES
from local_agent.models import (
    Checkpoint,
    CommandSpec,
    ExecutionResult,
    FailureAnalysis,
    FileOperation,
    Plan,
    PlanAmendment,
    ProjectContext,
    ProviderCapability,
    RecoveryState,
    RepairSignature,
    ReviewResult,
    RunReport,
    ScopeExpansionProposal,
    Subtask,
    SubtaskStatus,
    Task,
    TaskPlan,
    TaskStatus,
    ToolCall,
    ToolDefinition,
    ToolExecutionMetrics,
    ToolExecutionPolicy,
    ToolResult,
    ValidationPlan,
    normalize_diff_for_signature,
)
from local_agent.orchestrator import Orchestrator
from local_agent.planner import Planner
from local_agent.providers import AIProvider, MockProvider
from local_agent.storage import JsonFileStorage
from local_agent.tool_engine import IterationHistoryCompactor, ToolEngine
from local_agent.tools import ToolRegistry


class TestScopeAmendmentGuard(unittest.TestCase):
    """Test deterministic ScopeAmendmentGuard security and budget rules."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.fs = ProjectFilesystem(self.root)
        self.guard = ScopeAmendmentGuard(self.fs, max_total_amendments=5, max_scope_growth_factor=2.0)

        # Create sample files
        self.fs.create_file("src/app.py", "def main(): pass\n")
        self.fs.create_file("src/existing.py", "# existing helper\n")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_scope_expansion_valid_existing_file(self):
        plan = Plan(objective="test", files_likely_to_change=["src/app.py"])
        proposal = ScopeExpansionProposal(
            path="src/existing.py",
            reason="Discovered missing helper module",
            relationship="imports",
            is_create=False,
        )
        valid, reason = self.guard.evaluate(proposal, plan, initial_scope_count=1)
        self.assertTrue(valid)
        self.assertIn("Approved", reason)

    def test_scope_expansion_valid_create_file(self):
        plan = Plan(objective="test", files_likely_to_change=["src/app.py"])
        proposal = ScopeExpansionProposal(
            path="src/new_module.py",
            reason="Need new module for types",
            relationship="imports",
            is_create=True,
        )
        valid, reason = self.guard.evaluate(proposal, plan, initial_scope_count=1)
        self.assertTrue(valid)
        self.assertIn("Approved", reason)

    def test_scope_expansion_rejects_empty_path(self):
        plan = Plan(objective="test", files_likely_to_change=["src/app.py"])
        proposal = ScopeExpansionProposal(path="", reason="empty")
        valid, reason = self.guard.evaluate(proposal, plan, initial_scope_count=1)
        self.assertFalse(valid)
        self.assertIn("empty", reason)

    def test_scope_expansion_rejects_path_traversal(self):
        plan = Plan(objective="test", files_likely_to_change=["src/app.py"])
        proposal = ScopeExpansionProposal(
            path="../outside.py",
            reason="escape sandbox",
            is_create=False,
        )
        valid, reason = self.guard.evaluate(proposal, plan, initial_scope_count=1)
        self.assertFalse(valid)
        self.assertIn("path traversal", reason.lower())

    def test_scope_expansion_rejects_absolute_path(self):
        plan = Plan(objective="test", files_likely_to_change=["src/app.py"])
        abs_path = "/etc/passwd" if os.name != "nt" else "C:/Windows/System32/drivers/etc/hosts"
        proposal = ScopeExpansionProposal(
            path=abs_path,
            reason="absolute escape",
            is_create=False,
        )
        valid, reason = self.guard.evaluate(proposal, plan, initial_scope_count=1)
        self.assertFalse(valid)
        self.assertTrue("absolute path" in reason.lower() or "sandbox violation" in reason.lower())

    def test_scope_expansion_rejects_protected_directory(self):
        plan = Plan(objective="test", files_likely_to_change=["src/app.py"])
        # Attempt to target a .git or .agent_data path
        proposal = ScopeExpansionProposal(
            path=".git/config",
            reason="git access",
            is_create=False,
        )
        valid, reason = self.guard.evaluate(proposal, plan, initial_scope_count=1)
        self.assertFalse(valid)
        self.assertIn("protected directory", reason.lower())

    def test_scope_expansion_rejects_secret_file(self):
        plan = Plan(objective="test", files_likely_to_change=["src/app.py"])
        proposal = ScopeExpansionProposal(
            path=".env",
            reason="read env",
            is_create=True,
        )
        valid, reason = self.guard.evaluate(proposal, plan, initial_scope_count=1)
        self.assertFalse(valid)
        self.assertIn("secret file", reason.lower())

    def test_scope_expansion_rejects_duplicate_path(self):
        plan = Plan(objective="test", files_likely_to_change=["src/app.py"])
        proposal = ScopeExpansionProposal(
            path="src/app.py",
            reason="already in plan",
            is_create=False,
        )
        valid, reason = self.guard.evaluate(proposal, plan, initial_scope_count=1)
        self.assertFalse(valid)
        self.assertIn("already in plan", reason.lower())

    def test_scope_expansion_rejects_modify_on_nonexistent(self):
        plan = Plan(objective="test", files_likely_to_change=["src/app.py"])
        proposal = ScopeExpansionProposal(
            path="src/nonexistent.py",
            reason="modify file that does not exist",
            is_create=False,
        )
        valid, reason = self.guard.evaluate(proposal, plan, initial_scope_count=1)
        self.assertFalse(valid)
        self.assertIn("cannot modify non-existent", reason.lower())

    def test_scope_expansion_rejects_create_on_existing(self):
        plan = Plan(objective="test", files_likely_to_change=["src/app.py"])
        proposal = ScopeExpansionProposal(
            path="src/existing.py",
            reason="create file that already exists",
            is_create=True,
        )
        valid, reason = self.guard.evaluate(proposal, plan, initial_scope_count=1)
        self.assertFalse(valid)
        self.assertIn("cannot create already existing", reason.lower())

    def test_scope_expansion_budget_limit_exhaustion(self):
        guard = ScopeAmendmentGuard(self.fs, max_total_amendments=2)
        plan = Plan(objective="test", files_likely_to_change=["src/app.py"])
        # Add 2 dummy amendments
        plan.apply_amendment(ScopeExpansionProposal(path="src/a.py", reason="a", is_create=True))
        plan.apply_amendment(ScopeExpansionProposal(path="src/b.py", reason="b", is_create=True))

        proposal = ScopeExpansionProposal(path="src/c.py", reason="c", is_create=True)
        valid, reason = guard.evaluate(proposal, plan, initial_scope_count=1)
        self.assertFalse(valid)
        self.assertIn("maximum amendment budget reached", reason.lower())

    def test_scope_expansion_growth_limit_exhaustion(self):
        # Base count = 1, max_factor = 2.0 -> max_allowed = max(2, 4) = 4
        guard = ScopeAmendmentGuard(self.fs, max_total_amendments=10, max_scope_growth_factor=2.0)
        plan = Plan(objective="test", files_likely_to_change=["src/app.py"])
        plan.apply_amendment(ScopeExpansionProposal(path="src/a.py", reason="a", is_create=True))
        plan.apply_amendment(ScopeExpansionProposal(path="src/b.py", reason="b", is_create=True))
        plan.apply_amendment(ScopeExpansionProposal(path="src/c.py", reason="c", is_create=True))
        # Total files now = 4 (app, a, b, c). Adding a 5th should exceed max_allowed = 4

        proposal = ScopeExpansionProposal(path="src/d.py", reason="d", is_create=True)
        valid, reason = guard.evaluate(proposal, plan, initial_scope_count=1)
        self.assertFalse(valid)
        self.assertIn("scope growth limit reached", reason.lower())


class TestPlanVersioning(unittest.TestCase):
    """Test Plan versioning and PlanAmendment mechanics."""

    def test_initial_plan_is_v1(self):
        plan = Plan(objective="initial objective", files_likely_to_change=["src/main.py"])
        self.assertEqual(plan.version, 1)
        self.assertEqual(len(plan.amendments), 0)
        self.assertEqual(plan.allowed_paths, {"src/main.py"})

    def test_accepted_amendment_creates_v2(self):
        plan = Plan(objective="feature", files_likely_to_change=["src/main.py"])
        proposal = ScopeExpansionProposal(path="src/types.py", reason="Need types", is_create=True)
        amendment = plan.apply_amendment(proposal, approved_by="deterministic_policy")

        self.assertEqual(plan.version, 2)
        self.assertEqual(len(plan.amendments), 1)
        self.assertEqual(amendment.version, 2)
        self.assertEqual(amendment.proposal.path, "src/types.py")
        self.assertEqual(amendment.previous_allowed_paths, ["src/main.py"])
        self.assertEqual(amendment.new_allowed_paths, ["src/main.py", "src/types.py"])
        self.assertIn("src/types.py", plan.files_likely_to_create)
        self.assertEqual(plan.allowed_paths, {"src/main.py", "src/types.py"})

    def test_second_amendment_creates_v3(self):
        plan = Plan(objective="feature", files_likely_to_change=["src/main.py"])
        plan.apply_amendment(ScopeExpansionProposal(path="src/types.py", reason="types", is_create=True))
        amendment2 = plan.apply_amendment(ScopeExpansionProposal(path="src/utils.py", reason="utils", is_create=False))

        self.assertEqual(plan.version, 3)
        self.assertEqual(len(plan.amendments), 2)
        self.assertEqual(amendment2.version, 3)
        self.assertIn("src/utils.py", plan.files_likely_to_change)
        self.assertEqual(plan.allowed_paths, {"src/main.py", "src/types.py", "src/utils.py"})

    def test_amendment_serialization_roundtrip(self):
        plan = Plan(
            objective="test objective",
            files_likely_to_change=["src/app.py"],
            steps=["Step 1", "Step 2"],
            validation_strategy=["pytest"],
            risks=["Risk 1"],
        )
        plan.apply_amendment(ScopeExpansionProposal(path="src/types.py", reason="types", relationship="imports", is_create=True))

        plan_dict = plan.to_dict()
        self.assertEqual(plan_dict["version"], 2)
        self.assertEqual(len(plan_dict["amendments"]), 1)

        restored_plan = Plan.from_dict(plan_dict)
        self.assertEqual(restored_plan.version, 2)
        self.assertEqual(len(restored_plan.amendments), 1)
        self.assertEqual(restored_plan.amendments[0].proposal.path, "src/types.py")
        self.assertEqual(restored_plan.allowed_paths, {"src/app.py", "src/types.py"})

    def test_backward_compatibility_old_plan_json(self):
        old_json = {
            "objective": "old task",
            "files_likely_to_change": ["old/file.py"],
            "steps": ["step A"],
            "validation_strategy": ["pytest"],
        }
        plan = Plan.from_dict(old_json)
        self.assertEqual(plan.version, 1)
        self.assertEqual(plan.amendments, [])
        self.assertEqual(plan.allowed_paths, {"old/file.py"})


class TestIterationHistoryCompactor(unittest.TestCase):
    """Test cross-iteration context compaction and byte budgeting."""

    def test_cross_iteration_summary_respects_byte_budget(self):
        plan = Plan(objective="test", files_likely_to_change=["src/app.py"], version=2)
        plan.amendments = [
            PlanAmendment(
                amendment_id="a1",
                version=2,
                timestamp=datetime.datetime.now(datetime.timezone.utc),
                proposal=ScopeExpansionProposal(path="src/types.py", reason="Discovered types requirement", is_create=True),
                previous_allowed_paths=["src/app.py"],
                new_allowed_paths=["src/app.py", "src/types.py"],
            )
        ]
        recovery_state = RecoveryState()
        # Add multiple failure history entries
        for i in range(10):
            recovery_state.record_attempt(
                iteration=i + 1,
                failure=FailureAnalysis(
                    probable_root_cause=f"Failure number {i}: Very detailed long explanation " * 20,
                    recommended_fix=f"Fix recommendation {i} " * 20,
                    diagnostic_evidence=[ExecutionResult(command="pytest tests/", exit_code=1, stdout="Failed tests")],
                ),
                diff=f"+ line {i}\n",
                affected_files=["src/app.py"],
            )

        summary = IterationHistoryCompactor.build_cross_iteration_context(
            recovery_state, plan, max_bytes=1000
        )
        self.assertLessEqual(len(summary.encode("utf-8")), 1000)
        self.assertIn("Plan Version: v2", summary)
        self.assertIn("src/types.py", summary)

    def test_summary_includes_latest_failure_and_amendments(self):
        plan = Plan(objective="test", files_likely_to_change=["src/app.py"], version=2)
        plan.amendments = [
            PlanAmendment(
                amendment_id="a1",
                version=2,
                timestamp=datetime.datetime.now(datetime.timezone.utc),
                proposal=ScopeExpansionProposal(path="src/extra.py", reason="Extra helper", is_create=False),
                previous_allowed_paths=["src/app.py"],
                new_allowed_paths=["src/app.py", "src/extra.py"],
            )
        ]
        recovery_state = RecoveryState()
        failure = FailureAnalysis(
            probable_root_cause="ImportError: cannot import name 'Helper' from 'src.extra'",
            recommended_fix="Define Helper in src/extra.py",
            category="MISSING_DEPENDENCY",
            diagnostic_evidence=[ExecutionResult(command="pytest tests/", exit_code=1)],
        )
        recovery_state.record_failure(failure)
        recovery_state.record_attempt(1, failure, "+ diff", ["src/app.py"])

        summary = IterationHistoryCompactor.build_cross_iteration_context(recovery_state, plan, max_bytes=4000)
        self.assertIn("Plan Version: v2", summary)
        self.assertIn("src/extra.py", summary)
        self.assertIn("ImportError", summary)
        self.assertIn("Define Helper", summary)


class TestAdaptiveOrchestratorExecution(unittest.TestCase):
    """Test Orchestrator closed-loop dynamic replanning and scope adaptation."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.fs = ProjectFilesystem(self.root)
        self.storage = JsonFileStorage(self.root / ".agent_data")
        self.config = AgentConfig(project=self.root, max_iterations=3, max_plan_amendments=5)
        self.repo_lock = threading.Lock()
        self.memory_lock = threading.Lock()

        # Seed files
        self.fs.create_file("src/main.py", "def add(a, b):\n    return 0\n")
        self.fs.create_file("tests/test_main.py", "def test_add():\n    from src.main import add\n    assert add(1, 2) == 3\n")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_implementation_discovers_missing_dependency_and_amends_plan(self):
        """AI generates modifications for an unlisted file; orchestrator validates and amends plan."""
        # Initial plan only lists src/main.py
        initial_plan = Plan(
            objective="Add helper and implement add()",
            files_likely_to_change=["src/main.py"],
            steps=["Implement add using helper"],
            validation_strategy=["pytest"],
        )

        class DiscoveringProvider(MockProvider):
            def generate_plan(self, task, context):
                return initial_plan

            def generate_code(self, task, plan, context, failure=None, review=None):
                # Propose modifying src/main.py AND creating unlisted src/helper.py
                return [
                    FileOperation("create", "src/helper.py", content="def compute_sum(a, b):\n    return a + b\n"),
                    FileOperation("modify", "src/main.py", content="from src.helper import compute_sum\ndef add(a, b):\n    return compute_sum(a, b)\n"),
                ]

            def analyze_failure(self, execution, diff, context, plan):
                return FailureAnalysis("Unknown", [], "")

            def review_changes(self, task, plan, diff, context):
                return ReviewResult("APPROVED", "Looks good", [])

        orchestrator = Orchestrator(
            self.config,
            storage=self.storage,
            scheduler=None,
            repo_lock=self.repo_lock,
            memory_lock=self.memory_lock,
        )

        with patch("local_agent.orchestrator.build_provider", return_value=DiscoveringProvider()), \
             patch.object(CommandRunner, "run", return_value=ExecutionResult(command="pytest", exit_code=0)):
            report = orchestrator.run("Implement add with helper")

        self.assertTrue(report.completed)
        self.assertIsNotNone(report.plan)
        self.assertEqual(report.plan.version, 2)
        self.assertEqual(len(report.amendments), 1)
        self.assertEqual(report.amendments[0].proposal.path, "src/helper.py")
        self.assertIn("src/helper.py", report.plan.allowed_paths)
        self.assertTrue(self.fs.file_exists("src/helper.py"))

    def test_implementation_rejected_amendment_routes_to_safe_failure(self):
        """AI attempts to expand scope to a protected secret file; amendment is rejected and change blocked."""
        initial_plan = Plan(
            objective="Read secret and write main",
            files_likely_to_change=["src/main.py"],
            steps=["Modify main and create .env"],
        )

        class MaliciousProvider(MockProvider):
            def generate_plan(self, task, context):
                return initial_plan

            def generate_code(self, task, plan, context, failure=None, review=None):
                return [
                    FileOperation("create", ".env", content="SECRET=12345\n"),
                    FileOperation("modify", "src/main.py", content="def add(a, b):\n    return a + b\n"),
                ]

        orchestrator = Orchestrator(
            self.config,
            storage=self.storage,
            scheduler=None,
            repo_lock=self.repo_lock,
            memory_lock=self.memory_lock,
        )

        with patch("local_agent.orchestrator.build_provider", return_value=MaliciousProvider()):
            report = orchestrator.run("Write secret")

        self.assertFalse(report.completed)
        self.assertFalse(self.fs.file_exists(".env"))
        self.assertTrue(any(f.category == "UNSAFE_MODIFICATION" for f in report.failures))

    def test_validation_failure_triggers_diagnosed_scope_amendment(self):
        """Validation failure diagnoses missing file; orchestrator amends plan and repair iteration fixes it."""
        initial_plan = Plan(
            objective="Fix calculation",
            files_likely_to_change=["src/main.py"],
        )

        call_count = {"code": 0, "validate": 0}

        class AdaptiveRepairProvider(MockProvider):
            def generate_plan(self, task, context):
                return initial_plan

            def generate_code(self, task, plan, context, failure=None, review=None):
                call_count["code"] += 1
                if call_count["code"] == 1:
                    # First turn: modifies main only, causing validation failure
                    return [FileOperation("modify", "src/main.py", content="import math\ndef add(a, b):\n    return a + b\n")]
                else:
                    # Second turn (repair): creates diagnosed file and modifies main
                    return [
                        FileOperation("create", "src/math_helper.py", content="def safe_add(a, b):\n    return a + b\n"),
                        FileOperation("modify", "src/main.py", content="from src.math_helper import safe_add\ndef add(a, b):\n    return safe_add(a, b)\n"),
                    ]

            def analyze_failure(self, execution, diff, context, plan):
                # Diagnoses missing src/math_helper.py
                return FailureAnalysis(
                    probable_root_cause="Missing math helper module",
                    affected_files=["src/math_helper.py"],
                    recommended_fix="Create src/math_helper.py with safe_add()",
                    category="MISSING_DEPENDENCY",
                )

            def review_changes(self, task, plan, diff, context):
                return ReviewResult("APPROVED", "Looks good", [])

        orchestrator = Orchestrator(
            self.config,
            storage=self.storage,
            scheduler=None,
            repo_lock=self.repo_lock,
            memory_lock=self.memory_lock,
        )

        def mock_validate(spec):
            call_count["validate"] += 1
            if call_count["validate"] == 1:
                return ExecutionResult(command="pytest", exit_code=1, stdout="ModuleNotFoundError: src.math_helper")
            return ExecutionResult(command="pytest", exit_code=0, stdout="All tests passed")

        with patch("local_agent.orchestrator.build_provider", return_value=AdaptiveRepairProvider()), \
             patch.object(CommandRunner, "run", side_effect=mock_validate):
            report = orchestrator.run("Fix calculation")

        self.assertTrue(report.completed)
        self.assertEqual(report.iterations, 2)
        self.assertGreaterEqual(report.plan.version, 2)
        self.assertTrue(self.fs.file_exists("src/math_helper.py"))

    def test_stagnation_prevents_infinite_replanning(self):
        """Repeated identical failures abort when stagnation limit is reached."""
        initial_plan = Plan(objective="stagnant task", files_likely_to_change=["src/main.py"])

        class StagnantProvider(MockProvider):
            def generate_plan(self, task, context):
                return initial_plan

            def generate_code(self, task, plan, context, failure=None, review=None):
                return [FileOperation("modify", "src/main.py", content=f"# attempt {datetime.datetime.now()}\n")]

            def analyze_failure(self, execution, diff, context, plan):
                return FailureAnalysis(
                    probable_root_cause="Identical unchanging bug",
                    recommended_fix="Try something else",
                    category="STAGNANT_BUG",
                )

        orchestrator = Orchestrator(
            AgentConfig(project=self.root, max_iterations=5),
            storage=self.storage,
            scheduler=None,
            repo_lock=self.repo_lock,
            memory_lock=self.memory_lock,
        )

        with patch("local_agent.orchestrator.build_provider", return_value=StagnantProvider()), \
             patch.object(CommandRunner, "run", return_value=ExecutionResult(command="pytest", exit_code=1)):
            report = orchestrator.run("Stagnant run")

        self.assertFalse(report.completed)
        self.assertEqual(report.outcome, "STAGNATION_DETECTED")
        self.assertLessEqual(report.iterations, 4)


class TestCheckpointAndResumeWithAmendments(unittest.TestCase):
    """Test checkpointing and resumption with amended plan states."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.fs = ProjectFilesystem(self.root)
        self.storage = JsonFileStorage(self.root / ".agent_data")
        self.config = AgentConfig(project=self.root)
        self.repo_lock = threading.Lock()
        self.memory_lock = threading.Lock()

        self.fs.create_file("src/app.py", "def app(): pass\n")
        self.fs.create_file("src/helper.py", "def helper(): pass\n")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_checkpoint_serializes_and_restores_amendments(self):
        plan = Plan(objective="test task", files_likely_to_change=["src/app.py"])
        plan.apply_amendment(
            ScopeExpansionProposal(path="src/helper.py", reason="helper needed", is_create=False)
        )
        self.assertEqual(plan.version, 2)

        orchestrator = Orchestrator(
            self.config,
            storage=self.storage,
            scheduler=None,
            repo_lock=self.repo_lock,
            memory_lock=self.memory_lock,
        )

        now = datetime.datetime.now(datetime.timezone.utc)
        task = Task(task_id="task-resume-test", objective="test task", status=TaskStatus.PENDING, created_at=now, updated_at=now)
        self.storage.save_task(task)

        context = ProjectContext(root=str(self.root))
        report = RunReport(project=context, plan=plan)
        checkpoint = orchestrator._create_checkpoint(task, None, "Checkpoint after amendment", context, report)

        # Load checkpoint directly
        loaded = self.storage.load_checkpoint(checkpoint.checkpoint_id)
        self.assertIsNotNone(loaded)
        raw_plan = loaded.continuation_context.get("plan")
        self.assertIsNotNone(raw_plan)
        restored_plan = Plan.from_dict(raw_plan)

        self.assertEqual(restored_plan.version, 2)
        self.assertEqual(len(restored_plan.amendments), 1)
        self.assertEqual(restored_plan.allowed_paths, {"src/app.py", "src/helper.py"})

    def test_resume_preserves_amended_plan_version_and_scope(self):
        """When resuming a task from checkpoint, orchestrator restores amended plan v2."""
        plan = Plan(objective="test task", files_likely_to_change=["src/app.py"])
        plan.apply_amendment(
            ScopeExpansionProposal(path="src/helper.py", reason="helper needed", is_create=False)
        )

        now = datetime.datetime.now(datetime.timezone.utc)
        task = Task(task_id="task-resume-flow", objective="test task", status=TaskStatus.PAUSED, created_at=now, updated_at=now)
        self.storage.save_task(task)

        context = ProjectContext(root=str(self.root))
        report = RunReport(project=context, plan=plan)
        orchestrator = Orchestrator(
            self.config,
            storage=self.storage,
            scheduler=None,
            repo_lock=self.repo_lock,
            memory_lock=self.memory_lock,
        )
        orchestrator._create_checkpoint(task, None, "Paused checkpoint", context, report)

        # Provider on resume produces a change to src/helper.py
        class ResumingProvider(MockProvider):
            def generate_plan(self, task, context):
                # If plan was not restored, generate_plan would return v1 plan without src/helper.py
                return Plan(objective=task, files_likely_to_change=["src/app.py"])

            def generate_code(self, task, plan, context, failure=None, review=None):
                return [FileOperation("modify", "src/helper.py", content="def helper(): return 42\n")]

            def review_changes(self, task, plan, diff, context):
                return ReviewResult("APPROVED", "Looks good", [])

        with patch("local_agent.orchestrator.build_provider", return_value=ResumingProvider()), \
             patch.object(CommandRunner, "run", return_value=ExecutionResult(command="pytest", exit_code=0)):
            resumed_report = orchestrator.run(task)

        self.assertTrue(resumed_report.completed)
        self.assertEqual(resumed_report.plan.version, 2)
        self.assertIn("src/helper.py", resumed_report.plan.allowed_paths)


class TestPlannerReplanWithContext(unittest.TestCase):
    """Test Planner.replan_with_context functionality."""

    def test_replan_preserves_version_and_amendments(self):
        provider = MagicMock()
        provider.capabilities = {ProviderCapability.PLANNING, ProviderCapability.TOOL_USE}
        provider.generate_plan_with_tools.return_value = Plan(
            objective="revised objective",
            files_likely_to_change=["src/app.py", "src/extra.py"],
            steps=["Step 1 revised", "Step 2 revised"],
        )

        planner = Planner(provider)
        current_plan = Plan(
            objective="original objective",
            files_likely_to_change=["src/app.py"],
            version=2,
            amendments=[
                PlanAmendment(
                    amendment_id="a1",
                    version=2,
                    timestamp=datetime.datetime.now(datetime.timezone.utc),
                    proposal=ScopeExpansionProposal(path="src/extra.py", reason="extra"),
                )
            ],
        )
        context = ProjectContext(root=".")
        failure = FailureAnalysis(probable_root_cause="Missing extra.py", affected_files=["src/extra.py"])

        revised = planner.replan_with_context("task", current_plan, context, failure=failure)
        self.assertEqual(revised.version, 2)
        self.assertEqual(len(revised.amendments), 1)
        self.assertEqual(revised.amendments[0].proposal.path, "src/extra.py")


class TestSecurityRegressions(unittest.TestCase):
    """Test security invariants and negative cases."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.fs = ProjectFilesystem(self.root)
        self.fs.create_file("src/app.py", "def app(): pass\n")
        self.fs.create_file("src/secret_source.py", "TOKEN = 'abc'\n")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_coding_agent_still_rejects_unauthorized_modification(self):
        agent = CodingAgent(self.fs)
        plan = Plan(objective="test", files_likely_to_change=["src/app.py"])
        # Attempt to modify src/secret_source.py without an amendment
        ops = [FileOperation("modify", "src/secret_source.py", content="TOKEN = 'hacked'\n")]
        with self.assertRaises(UnsafeModificationError):
            agent.prepare(ops, plan)

    def test_accepted_amendment_required_before_newly_discovered_file_can_mutate(self):
        agent = CodingAgent(self.fs)
        plan = Plan(objective="test", files_likely_to_change=["src/app.py"])
        ops = [FileOperation("create", "src/new.py", content="# new\n")]
        # Before amendment -> rejected
        with self.assertRaises(UnsafeModificationError):
            agent.prepare(ops, plan)

        # After amendment -> accepted
        plan.apply_amendment(ScopeExpansionProposal(path="src/new.py", reason="new", is_create=True))
        prepared = agent.prepare(ops, plan)
        self.assertEqual(len(prepared), 1)
        self.assertEqual(prepared[0].path, "src/new.py")

    def test_protected_files_cannot_become_authorized_through_amendment(self):
        guard = ScopeAmendmentGuard(self.fs)
        plan = Plan(objective="test", files_likely_to_change=["src/app.py"])
        for secret_name in SECRET_NAMES:
            proposal = ScopeExpansionProposal(path=secret_name, reason="attempt secret access", is_create=False)
            valid, reason = guard.evaluate(proposal, plan)
            self.assertFalse(valid)
            self.assertIn("secret file cannot be added", reason.lower())

    def test_path_traversal_cannot_become_authorized_through_amendment(self):
        guard = ScopeAmendmentGuard(self.fs)
        plan = Plan(objective="test", files_likely_to_change=["src/app.py"])
        traversal_paths = [
            "../../etc/passwd",
            "../sibling.py",
            "src/../../outside.py",
            "/absolute/root.py",
        ]
        for p in traversal_paths:
            proposal = ScopeExpansionProposal(path=p, reason="traversal attempt", is_create=False)
            valid, reason = guard.evaluate(proposal, plan)
            self.assertFalse(valid)

    def test_old_checkpoint_compatibility_without_amendment_fields(self):
        storage = JsonFileStorage(self.root / ".agent_data")
        old_checkpoint = Checkpoint(
            checkpoint_id="chk-old-123",
            task_id="task-old",
            subtask_id="",
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            current_state_description="Old checkpoint",
            continuation_context={
                "task_objective": "Old task",
                "plan": {
                    "objective": "Old task",
                    "files_likely_to_change": ["src/app.py"],
                    "steps": ["Step 1"],
                },
                "recovery_state": {
                    "completed_iterations": 1,
                },
            },
        )
        storage.save_checkpoint(old_checkpoint)

        loaded = storage.load_checkpoint("chk-old-123")
        self.assertIsNotNone(loaded)
        raw_plan = loaded.continuation_context.get("plan")
        plan = Plan.from_dict(raw_plan)
        self.assertEqual(plan.version, 1)
        self.assertEqual(plan.amendments, [])
        self.assertEqual(plan.allowed_paths, {"src/app.py"})

    def test_canonical_history_remains_intact_while_context_compacts(self):
        recovery_state = RecoveryState()
        plan = Plan(objective="test", files_likely_to_change=["src/app.py"])
        report = RunReport(project=ProjectContext(root=str(self.root)), plan=plan)

        # Record 5 tool turns
        for i in range(5):
            t_call = ToolCall(call_id=f"c_{i}", tool_name="read_file_range", arguments={"path": "src/app.py", "start_line": 1, "end_line": 10})
            t_res = ToolResult(call_id=f"c_{i}", tool_name="read_file_range", output=f"def app_{i}(): pass\n" * 20)
            report.tool_history.append((t_call, t_res))

        summary = IterationHistoryCompactor.build_cross_iteration_context(recovery_state, plan, report=report, max_bytes=500)
        # Verify compact summary is bounded
        self.assertLessEqual(len(summary.encode("utf-8")), 500)
        # Verify canonical report.tool_history still has all 5 turns
        self.assertEqual(len(report.tool_history), 5)


class TestDeepAdversarialAudits(unittest.TestCase):
    """Deep adversarial audit tests verifying Phase 4.8 security contracts."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.fs = ProjectFilesystem(self.root)
        self.guard = ScopeAmendmentGuard(self.fs, max_total_amendments=5, max_scope_growth_factor=2.0)
        self.fs.create_file("src/app.py", "def app(): pass\n")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_adversarial_paths_matrix(self):
        plan = Plan(objective="test", files_likely_to_change=["src/app.py"])
        adversarial_cases = [
            "../outside.py",
            "../../outside.py",
            "./../outside.py",
            "/etc/passwd",
            "/root/.ssh/id_rsa",
            "C:\\Windows\\System32\\cmd.exe",
            "D:\\data.txt",
            ".",
            "..",
            "",
            "   ",
            "src/..\\outside.py",
            ".git/config",
            ".git/objects/00/123",
            ".agent_data/tasks.json",
            ".hg/store",
            ".svn/entries",
            ".env",
            ".env.local",
            ".env.production",
            "credentials.json",
            "secrets.json",
            "id_rsa",
            "id_ed25519",
            "token.json",
            ".npmrc",
            ".pypirc",
        ]
        for bad_path in adversarial_cases:
            proposal = ScopeExpansionProposal(path=bad_path, reason="adversarial test", is_create=True)
            valid, reason = self.guard.evaluate(proposal, plan, initial_scope_count=1)
            self.assertFalse(valid, f"Expected path '{bad_path}' to be rejected, but was accepted with: {reason}")

    def test_legitimate_double_dot_filename(self):
        """A filename containing '..' as a substring but not a traversal component is valid."""
        plan = Plan(objective="test", files_likely_to_change=["src/app.py"])
        proposal = ScopeExpansionProposal(path="src/foo..bar.py", reason="dotted filename", is_create=True)
        valid, reason = self.guard.evaluate(proposal, plan, initial_scope_count=1)
        self.assertTrue(valid, f"Expected legitimate dotted filename to be accepted, but got: {reason}")

    def test_plan_apply_amendment_duplicate_rejection_at_model_level(self):
        plan = Plan(objective="test", files_likely_to_change=["src/app.py"])
        proposal1 = ScopeExpansionProposal(path="src/new.py", reason="new file", is_create=True)
        amendment1 = plan.apply_amendment(proposal1)
        self.assertEqual(plan.version, 2)
        self.assertEqual(len(plan.amendments), 1)

        # Attempting to apply amendment on already allowed path must raise ValueError
        proposal2 = ScopeExpansionProposal(path="src/new.py", reason="duplicate new file", is_create=True)
        with self.assertRaises(ValueError):
            plan.apply_amendment(proposal2)
        self.assertEqual(plan.version, 2)
        self.assertEqual(len(plan.amendments), 1)

    def test_deterministic_serialization_roundtrip(self):
        plan = Plan(objective="Deterministic test", files_likely_to_change=["src/app.py"])
        plan.apply_amendment(ScopeExpansionProposal(path="src/b.py", reason="b", is_create=True))
        plan.apply_amendment(ScopeExpansionProposal(path="src/a.py", reason="a", is_create=True))
        self.assertEqual(plan.version, 3)

        serialized_1 = json.dumps(plan.to_dict(), sort_keys=True)
        restored_1 = Plan.from_dict(json.loads(serialized_1))
        serialized_2 = json.dumps(restored_1.to_dict(), sort_keys=True)

        self.assertEqual(serialized_1, serialized_2)
        self.assertEqual(restored_1.version, 3)
        self.assertEqual(len(restored_1.amendments), 2)
        self.assertEqual(restored_1.allowed_paths, {"src/app.py", "src/a.py", "src/b.py"})

    def test_budget_accounting_independent_enforcement(self):
        """Plan amendments do not reset recovery telemetry or counters."""
        recovery_state = RecoveryState()
        recovery_state.completed_iterations = 2
        failure1 = FailureAnalysis(probable_root_cause="err1", category="STAGNANT_BUG")
        failure2 = FailureAnalysis(probable_root_cause="err1", category="STAGNANT_BUG")
        recovery_state.record_failure(failure1)
        recovery_state.record_failure(failure2)
        self.assertEqual(recovery_state.consecutive_same_failure_count, 2)

        patch_hash = "abc123hash"
        recovery_state.record_attempt(1, failure1, "diff1", ["src/app.py"])
        self.assertTrue(recovery_state.is_duplicate_patch(normalize_diff_for_signature("diff1")))

        # Amending plan
        plan = Plan(objective="test", files_likely_to_change=["src/app.py"])
        plan.apply_amendment(ScopeExpansionProposal(path="src/helper.py", reason="h", is_create=True))

        # Invariants: recovery state counters remain intact
        self.assertEqual(recovery_state.completed_iterations, 2)
        self.assertEqual(recovery_state.consecutive_same_failure_count, 2)
        self.assertTrue(recovery_state.is_duplicate_patch(normalize_diff_for_signature("diff1")))

    def test_approval_policy_still_controls_amended_paths(self):
        """Amended paths undergo full ApprovalPolicyEngine evaluation and cannot bypass manual approval."""
        from local_agent.models import ApprovalPolicy, ChangeImpact, ChangeTarget, PreparedChange

        policy = ApprovalPolicy(
            name="Protect Core Files",
            action="require_approval",
            if_path_matches=["src/core/**"],
        )
        engine = ApprovalPolicyEngine([policy])

        plan = Plan(objective="test", files_likely_to_change=["src/app.py"])
        plan.apply_amendment(ScopeExpansionProposal(path="src/core/security.py", reason="core file", is_create=True))

        prepared_change = PreparedChange(
            action="create",
            path="src/core/security.py",
            original=None,
            resulting="# security\n",
            diff="+ # security\n",
        )
        impact = ChangeImpact(
            summary="core modification",
            targets=[ChangeTarget("src/core/security.py", "create", 1.0, "core change", "core", "high")],
        )

        self.assertTrue(engine.is_manual_approval_required([prepared_change], impact))

    def test_context_compactor_extreme_unicode_and_length(self):
        """Cross-iteration compactor handles 100KB text, emoji, and non-ASCII within exact byte limit."""
        recovery_state = RecoveryState()
        huge_text = "🔥 Multi-byte Unicode test: 中文 / 日本語 / Español / 🚀 " + ("A" * 100000)
        failure = FailureAnalysis(
            probable_root_cause=huge_text,
            recommended_fix=huge_text,
            diagnostic_evidence=[ExecutionResult(command="pytest " + ("x" * 2000), exit_code=1, stdout=huge_text)],
        )
        recovery_state.record_failure(failure)

        plan = Plan(objective="Huge task", files_likely_to_change=["src/app.py"])
        summary = IterationHistoryCompactor.build_cross_iteration_context(recovery_state, plan, max_bytes=2000)

        # Hard UTF-8 byte boundary verification
        encoded = summary.encode("utf-8")
        self.assertLessEqual(len(encoded), 2000)
        # Verify valid UTF-8 string
        self.assertEqual(encoded.decode("utf-8"), summary)
        self.assertIn("Active Plan State", summary)

    def test_config_validation_bounds(self):
        config_valid = AgentConfig(project=self.root, max_plan_amendments=5, max_scope_growth_factor=2.0)
        config_valid.validate()

        config_neg_amendments = AgentConfig(project=self.root, max_plan_amendments=-1)
        with self.assertRaises(ValueError):
            config_neg_amendments.validate()

        config_low_growth = AgentConfig(project=self.root, max_scope_growth_factor=0.5)
        with self.assertRaises(ValueError):
            config_low_growth.validate()


if __name__ == "__main__":
    unittest.main()
