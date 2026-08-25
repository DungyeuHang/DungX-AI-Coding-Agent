from __future__ import annotations

import datetime
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from local_agent.config import AgentConfig
from local_agent.models import (
    FailureAnalysis,
    FileOperation,
    Plan,
    ProjectContext,
    ProviderCapability,
    ReviewResult,
    RunReport,
    Subtask,
    SubtaskStatus,
    Task,
    TaskPlan,
    TaskStatus,
    ToolCall,
    ToolDefinition,
    ToolExecutionMetrics,
    ToolResult,
)
from local_agent.orchestrator import Orchestrator
from local_agent.providers import AIProvider, RateLimitError
from local_agent.storage import JsonFileStorage


class ToolEnabledMockProvider(AIProvider):
    def __init__(self, responses: list[Any]):
        self.responses = list(responses)

    @property
    def capabilities(self) -> set[ProviderCapability]:
        return {
            ProviderCapability.IMPLEMENTATION,
            ProviderCapability.TOOL_USE,
            ProviderCapability.PLANNING,
            ProviderCapability.REPAIR,
            ProviderCapability.REVIEW,
        }

    def generate_plan(self, objective: str, context: ProjectContext) -> Plan:
        return Plan(objective=objective, steps=["step 1"])

    def generate_code_with_tools(
        self,
        task: str,
        plan: Plan,
        context: ProjectContext,
        tools: list[ToolDefinition],
        tool_history: list[tuple[ToolCall, ToolResult]] | None = None,
        failure: Any = None,
        review: Any = None,
    ) -> Any:
        if not self.responses:
            return [FileOperation("modify", "src/main.py", "print('hello')", "test", None)]
        return self.responses.pop(0)

    def generate_code(self, *args, **kwargs) -> list[FileOperation]:
        return [FileOperation("modify", "src/main.py", "print('hello')", "test", None)]

    def review_changes(self, task: str, plan: Plan, diff: str, context: ProjectContext) -> ReviewResult:
        return ReviewResult(verdict="APPROVED", summary="Looks good", findings=[])

    def analyze_failure(self, execution: Any, diff: str, context: ProjectContext, plan: Plan) -> FailureAnalysis:
        return FailureAnalysis(probable_root_cause="Validation failed", recommended_fix="Fix code")


class DummyScheduler:
    def __init__(self, prov_inst):
        self.provider = "mock"
        self._prov_inst = prov_inst

    def _select_providers(self, task, capabilities):
        return [self]

    def _build_provider_instance(self, provider_name):
        return self._prov_inst


class TestPhase413Reporting(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        (self.root / "src").mkdir(parents=True, exist_ok=True)
        (self.root / "src" / "main.py").write_text("print('old')", encoding="utf-8")
        self.config = AgentConfig(project=self.root)
        self.storage = JsonFileStorage(self.root / ".agent")
        self.repo_lock = threading.Lock()
        self.memory_lock = threading.Lock()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_run_report_model_fields(self):
        context = ProjectContext(root=str(self.root))
        report = RunReport(project=context)
        self.assertEqual(report.tool_metrics, [])
        self.assertEqual(report.tool_history, [])

        metric = ToolExecutionMetrics(total_calls=2, steps_used=2, completed=True)
        report.tool_metrics.append(metric)
        self.assertEqual(len(report.tool_metrics), 1)
        self.assertEqual(report.tool_metrics[0].total_calls, 2)

    def test_orchestrator_captures_tool_metrics_in_report(self):
        """Orchestrator._execute_code_generation records ToolEngine metrics in report."""
        tool_call = ToolCall("c1", "find_files", {"pattern": "*.py"})
        final_ops = [FileOperation("modify", "src/main.py", "print('new')", "update", None)]
        provider = ToolEnabledMockProvider([tool_call, final_ops])

        scheduler = DummyScheduler(provider)

        orchestrator = Orchestrator(
            self.config, self.storage, scheduler, self.repo_lock, self.memory_lock
        )

        now = datetime.datetime.now(datetime.timezone.utc)
        task = Task(
            task_id="task-413-1",
            objective="Test tool reporting",
            status=TaskStatus.PENDING,
            created_at=now,
            updated_at=now,
        )
        plan = Plan(objective="Test plan", steps=["step 1"])
        context = ProjectContext(root=str(self.root))
        report = RunReport(project=context)

        ops, history = orchestrator._execute_code_generation(
            task=task,
            plan=plan,
            context=context,
            failure=None,
            review=None,
            stage_name="implementation",
            report=report,
        )

        self.assertEqual(len(ops), 1)
        self.assertEqual(len(history), 1)
        self.assertEqual(len(report.tool_metrics), 1)
        self.assertEqual(report.tool_metrics[0].total_calls, 1)
        self.assertEqual(report.tool_metrics[0].steps_used, 1)
        self.assertTrue(report.tool_metrics[0].completed)
        self.assertEqual(len(report.tool_history), 1)

    def test_checkpoint_and_resume_preserves_tool_metrics(self):
        """Checkpoints preserve tool_metrics and tool_history, and resume restores them."""
        scheduler = DummyScheduler(ToolEnabledMockProvider([]))
        orchestrator = Orchestrator(
            self.config, self.storage, scheduler, self.repo_lock, self.memory_lock
        )

        now = datetime.datetime.now(datetime.timezone.utc)
        task = Task(
            task_id="task-413-2",
            objective="Resume tool reporting test",
            status=TaskStatus.PENDING,
            created_at=now,
            updated_at=now,
            plan=TaskPlan(
                objective="Main plan",
                subtasks=[
                    Subtask(subtask_id="sub-1", title="Subtask 1", goal="Goal 1", status=SubtaskStatus.RUNNING)
                ],
            ),
        )
        context = ProjectContext(root=str(self.root))
        report = RunReport(project=context, task_id=task.task_id, subtask_id="sub-1")

        call = ToolCall("c1", "find_files", {"pattern": "*.py"})
        res = ToolResult("c1", "find_files", "src/main.py", is_error=False)
        metric = ToolExecutionMetrics(total_calls=1, steps_used=1, total_output_bytes=12, completed=True)

        report.tool_history = [(call, res)]
        report.tool_metrics = [metric]

        subtask = task.plan.subtasks[0]
        checkpoint = orchestrator._create_checkpoint(
            task=task,
            subtask=subtask,
            description="Paused test checkpoint",
            context=context,
            report=report,
        )

        self.assertIn("tool_metrics", checkpoint.continuation_context)
        self.assertIn("tool_history", checkpoint.continuation_context)
        self.assertEqual(len(checkpoint.continuation_context["tool_metrics"]), 1)
        self.assertEqual(checkpoint.continuation_context["tool_metrics"][0]["total_calls"], 1)

        # Build snapshot report
        built_report = orchestrator._build_run_report(task)
        self.assertEqual(len(built_report.tool_metrics), 1)
        self.assertEqual(built_report.tool_metrics[0].total_calls, 1)
        self.assertEqual(len(built_report.tool_history), 1)
        self.assertEqual(built_report.tool_history[0][0].tool_name, "find_files")

    def test_orchestrator_run_end_to_end_captures_tool_metrics(self):
        """Full orchestrator.run(task) captures tool metrics in returned RunReport."""
        tool_call = ToolCall("c1", "find_files", {"pattern": "*.py"})
        final_ops = [FileOperation("modify", "src/main.py", "print('tool modified')\n", "update", None)]
        provider = ToolEnabledMockProvider([tool_call, final_ops])

        scheduler = DummyScheduler(provider)
        orchestrator = Orchestrator(
            self.config, self.storage, scheduler, self.repo_lock, self.memory_lock
        )

        now = datetime.datetime.now(datetime.timezone.utc)
        task = Task(
            task_id="task-413-e2e",
            objective="End-to-end tool metrics test",
            status=TaskStatus.PENDING,
            created_at=now,
            updated_at=now,
        )

        report = orchestrator.run(task)

        self.assertTrue(report.completed)
        self.assertEqual(len(report.tool_metrics), 1)
        self.assertEqual(report.tool_metrics[0].total_calls, 1)
        self.assertTrue(report.tool_metrics[0].completed)
        self.assertEqual(len(report.tool_history), 1)

    def test_oneshot_provider_does_not_populate_tool_metrics(self):
        """Providers without ProviderCapability.TOOL_USE leave tool_metrics empty."""
        class OneShotMockProvider(AIProvider):
            @property
            def capabilities(self) -> set[ProviderCapability]:
                return {ProviderCapability.PLANNING, ProviderCapability.IMPLEMENTATION, ProviderCapability.REVIEW}

            def generate_plan(self, objective: str, context: ProjectContext) -> Plan:
                return Plan(objective=objective, steps=["step 1"])

            def generate_code(self, *args, **kwargs) -> list[FileOperation]:
                return [FileOperation("modify", "src/main.py", "print('oneshot')\n", "update", None)]

            def review_changes(self, task: str, plan: Plan, diff: str, context: ProjectContext) -> ReviewResult:
                return ReviewResult(verdict="APPROVED", summary="Looks good", findings=[])

        provider = OneShotMockProvider()
        scheduler = DummyScheduler(provider)
        orchestrator = Orchestrator(
            self.config, self.storage, scheduler, self.repo_lock, self.memory_lock
        )

        now = datetime.datetime.now(datetime.timezone.utc)
        task = Task(
            task_id="task-413-oneshot",
            objective="One-shot test",
            status=TaskStatus.PENDING,
            created_at=now,
            updated_at=now,
        )

        report = orchestrator.run(task)

        self.assertTrue(report.completed)
        self.assertEqual(report.tool_metrics, [])
        self.assertEqual(report.tool_history, [])

    def test_pause_and_resume_preserves_tool_history_and_accumulates_on_resume(self):
        """Verify tool_history is preserved across pause/resume and final session metrics are recorded."""
        # Session A: 1 tool call then RateLimitError
        call_a = ToolCall("call_a", "find_files", {"pattern": "*.py"})
        provider_a = ToolEnabledMockProvider([])

        def generate_with_tools_a(task, plan, context, tools, tool_history=None, failure=None, review=None):
            if not tool_history:
                return call_a
            raise RateLimitError("Rate limit hit during exploration")

        provider_a.generate_code_with_tools = generate_with_tools_a

        scheduler_a = DummyScheduler(provider_a)
        orchestrator_a = Orchestrator(
            self.config, self.storage, scheduler_a, self.repo_lock, self.memory_lock
        )

        now = datetime.datetime.now(datetime.timezone.utc)
        task = Task(
            task_id="task-413-pause-resume",
            objective="Pause and resume telemetry test",
            status=TaskStatus.PENDING,
            created_at=now,
            updated_at=now,
        )

        report_a = orchestrator_a.run(task)
        self.assertEqual(task.status, TaskStatus.PAUSED)
        self.assertIsNotNone(task.latest_checkpoint_id)

        # Verify checkpoint has tool_history saved
        checkpoint = self.storage.load_checkpoint(task.latest_checkpoint_id)
        saved_history = checkpoint.continuation_context.get("tool_history", [])
        self.assertEqual(len(saved_history), 1)
        self.assertEqual(saved_history[0]["call"]["call_id"], "call_a")

        # Session B: Resume from checkpoint with provider that does 1 more tool call then finishes
        call_b = ToolCall("call_b", "read_file_range", {"path": "src/main.py", "start_line": 1, "end_line": 1})
        final_ops = [FileOperation("modify", "src/main.py", "print('done')\n", "done", None)]

        calls_executed_in_b: list[str] = []

        def generate_with_tools_b(task, plan, context, tools, tool_history=None, failure=None, review=None):
            # Verify history passed to provider in session B has call_a
            self.assertIsNotNone(tool_history)
            self.assertGreaterEqual(len(tool_history), 1)
            self.assertEqual(tool_history[0][0].call_id, "call_a")
            if len(tool_history) == 1:
                calls_executed_in_b.append("call_b")
                return call_b
            return final_ops

        provider_b = ToolEnabledMockProvider([])
        provider_b.generate_code_with_tools = generate_with_tools_b

        scheduler_b = DummyScheduler(provider_b)
        orchestrator_b = Orchestrator(
            self.config, self.storage, scheduler_b, self.repo_lock, self.memory_lock
        )

        report_b = orchestrator_b.run(task)
        self.assertTrue(report_b.completed)

        # Verify session B recorded metrics with cumulative history
        self.assertEqual(len(report_b.tool_metrics), 1)
        self.assertEqual(report_b.tool_metrics[0].total_calls, 2)
        self.assertTrue(report_b.tool_metrics[0].completed)

        # Verify tool_history represents complete cumulative history [call_a, call_b]
        self.assertEqual(len(report_b.tool_history), 2)
        self.assertEqual(report_b.tool_history[0][0].call_id, "call_a")
        self.assertEqual(report_b.tool_history[1][0].call_id, "call_b")

        # Verify call_a was not re-executed in session B
        self.assertEqual(calls_executed_in_b, ["call_b"])

    def test_multi_iteration_accumulates_distinct_metric_sessions_no_duplication(self):
        """Verify multi-iteration repair loops accumulate exactly [Session 1, Session 2] with no duplication."""
        # Iteration 1: 1 tool call -> FileOp
        # Iteration 2: 1 tool call -> FileOp
        call_1 = ToolCall("call_1", "find_files", {"pattern": "*.py"})
        call_2 = ToolCall("call_2", "read_file_range", {"path": "src/main.py", "start_line": 1, "end_line": 1})

        it1_ops = [FileOperation("modify", "src/main.py", "print('it1')\n", "it1", None)]
        it2_ops = [FileOperation("modify", "src/main.py", "print('it2')\n", "it2", None)]

        step_counter = {"val": 0}

        def generate_with_tools(task, plan, context, tools, tool_history=None, failure=None, review=None):
            step_counter["val"] += 1
            if failure is None:
                # Iteration 1
                if not tool_history:
                    return call_1
                return it1_ops
            else:
                # Iteration 2 (Repair)
                if not tool_history or len(tool_history) == 1:
                    return call_2
                return it2_ops

        provider = ToolEnabledMockProvider([])
        provider.generate_code_with_tools = generate_with_tools

        scheduler = DummyScheduler(provider)
        config = AgentConfig(
            project=self.root,
            max_iterations=2,
            validation_commands=["python -c \"import sys; sys.exit(0 if 'it2' in open('src/main.py').read() else 1)\""],
        )
        orchestrator = Orchestrator(
            config, self.storage, scheduler, self.repo_lock, self.memory_lock
        )

        now = datetime.datetime.now(datetime.timezone.utc)
        task = Task(
            task_id="task-413-multi-iter",
            objective="Multi-iteration tool test",
            status=TaskStatus.PENDING,
            created_at=now,
            updated_at=now,
        )

        report = orchestrator.run(task)
        self.assertTrue(report.completed)

        # Exactly 2 metric sessions: [Iteration 1, Iteration 2] (NO duplication like [1, 1, 2])
        self.assertEqual(len(report.tool_metrics), 2, "Must contain exactly [Iteration 1, Iteration 2] metrics")
        self.assertEqual(report.tool_metrics[0].total_calls, 1)
        self.assertTrue(report.tool_metrics[0].completed)
        self.assertEqual(report.tool_metrics[1].total_calls, 2)
        self.assertTrue(report.tool_metrics[1].completed)

    def test_checkpoint_roundtrip_preserves_all_tool_fields(self):
        """Verify tool_metrics and tool_history round-trip preserving all metadata fields."""
        call = ToolCall(
            call_id="call-xyz",
            tool_name="read_file_range",
            arguments={"path": "src/main.py", "start_line": 1, "end_line": 5},
        )
        res = ToolResult(
            call_id="call-xyz",
            tool_name="read_file_range",
            output="print('hello')",
            is_error=False,
            truncated=True,
        )
        metrics = ToolExecutionMetrics(
            total_calls=5,
            unique_calls=3,
            repeated_calls=2,
            calls_by_tool={"read_file_range": 3, "grep_code": 2},
            total_output_bytes=1024,
            output_bytes_by_tool={"read_file_range": 600, "grep_code": 424},
            truncated_results=1,
            tool_errors=0,
            circuit_breaker_events=0,
            steps_used=5,
            history_entries=5,
            termination_reason="completed",
            completed=True,
            elapsed_ms=123.45,
        )

        now = datetime.datetime.now(datetime.timezone.utc)
        task = Task(
            task_id="task-roundtrip",
            objective="Roundtrip test",
            status=TaskStatus.PENDING,
            created_at=now,
            updated_at=now,
            plan=TaskPlan(
                objective="Main plan",
                subtasks=[
                    Subtask(subtask_id="sub-rt", title="Subtask RT", goal="Goal RT", status=SubtaskStatus.RUNNING)
                ],
            ),
        )
        context = ProjectContext(root=str(self.root))
        report = RunReport(
            project=context,
            task_id=task.task_id,
            subtask_id="sub-rt",
            tool_history=[(call, res)],
            tool_metrics=[metrics],
        )

        scheduler = DummyScheduler(ToolEnabledMockProvider([]))
        orchestrator = Orchestrator(
            self.config, self.storage, scheduler, self.repo_lock, self.memory_lock
        )

        subtask = task.plan.subtasks[0]
        checkpoint = orchestrator._create_checkpoint(task, subtask, "Checkpoint RT", context, report)

        # Verify disk serialization
        loaded_checkpoint = self.storage.load_checkpoint(checkpoint.checkpoint_id)
        self.assertIn("tool_metrics", loaded_checkpoint.continuation_context)
        self.assertIn("tool_history", loaded_checkpoint.continuation_context)

        # Snapshot reconstruction
        snapshot_report = orchestrator._build_run_report(task)
        self.assertEqual(len(snapshot_report.tool_metrics), 1)
        m = snapshot_report.tool_metrics[0]
        self.assertEqual(m.total_calls, 5)
        self.assertEqual(m.unique_calls, 3)
        self.assertEqual(m.repeated_calls, 2)
        self.assertEqual(m.calls_by_tool, {"read_file_range": 3, "grep_code": 2})
        self.assertEqual(m.total_output_bytes, 1024)
        self.assertEqual(m.output_bytes_by_tool, {"read_file_range": 600, "grep_code": 424})
        self.assertEqual(m.truncated_results, 1)
        self.assertEqual(m.completed, True)
        self.assertEqual(m.termination_reason, "completed")
        self.assertAlmostEqual(m.elapsed_ms, 123.45)

        self.assertEqual(len(snapshot_report.tool_history), 1)
        c, r = snapshot_report.tool_history[0]
        self.assertEqual(c.call_id, "call-xyz")
        self.assertEqual(c.tool_name, "read_file_range")
        self.assertEqual(c.arguments, {"path": "src/main.py", "start_line": 1, "end_line": 5})
        self.assertEqual(r.call_id, "call-xyz")
        self.assertEqual(r.tool_name, "read_file_range")
        self.assertEqual(r.output, "print('hello')")
        self.assertEqual(r.is_error, False)
        self.assertEqual(r.truncated, True)

    def test_legacy_checkpoint_backward_compatibility(self):
        """Verify legacy checkpoints lacking tool_metrics or tool_history deserialize safely."""
        now = datetime.datetime.now(datetime.timezone.utc)
        task = Task(
            task_id="task-legacy",
            objective="Legacy test",
            status=TaskStatus.PENDING,
            created_at=now,
            updated_at=now,
        )
        context = ProjectContext(root=str(self.root))
        report = RunReport(project=context, task_id=task.task_id)

        scheduler = DummyScheduler(ToolEnabledMockProvider([]))
        orchestrator = Orchestrator(
            self.config, self.storage, scheduler, self.repo_lock, self.memory_lock
        )

        checkpoint = orchestrator._create_checkpoint(
            task, None, "Legacy checkpoint without tools", context, report
        )

        # Explicitly remove tool keys from stored checkpoint JSON to simulate legacy checkpoint
        ckpt_data = self.storage.load_checkpoint(checkpoint.checkpoint_id)
        ckpt_data.continuation_context.pop("tool_metrics", None)
        ckpt_data.continuation_context.pop("tool_history", None)
        self.storage.save_checkpoint(ckpt_data)

        # Reconstruct report snapshot
        snapshot_report = orchestrator._build_run_report(task)
        self.assertEqual(snapshot_report.tool_metrics, [])
        self.assertEqual(snapshot_report.tool_history, [])


if __name__ == "__main__":
    unittest.main()
