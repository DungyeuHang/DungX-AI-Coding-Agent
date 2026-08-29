from __future__ import annotations

import argparse
import datetime
import os
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any

from local_agent.coding_agent import (
    IMPLEMENTATION_TOOL_SURFACE,
    CodingAgent,
    InteractiveCodingAgent,
    PatchValidationError,
    ScopeAmendmentGuard,
    UnsafeModificationError,
)
from local_agent.config import AgentConfig, add_common_arguments, config_from_args
from local_agent.coordinator import ParallelExecutionCoordinator
from local_agent.filesystem import ProjectFilesystem, ProtectedPathError, SandboxViolation
from local_agent.git import GitIntegration
from local_agent.models import (
    Checkpoint,
    ExecutionResult,
    ExportedSymbol,
    FailureAnalysis,
    FileOperation,
    ImplementationResult,
    ImplementationTerminationReason,
    Plan,
    ProjectContext,
    ProviderCapability,
    ProviderError,
    QuotaExceededError,
    RateLimitError,
    ReviewResult,
    RunReport,
    Subtask,
    SubtaskContract,
    SubtaskStatus,
    Task,
    TaskPlan,
    TaskStatus,
    ToolCall,
    ToolDefinition,
    ToolExecutionMetrics,
    ToolExecutionPolicy,
    ToolResult,
)
from local_agent.orchestrator import Orchestrator
from local_agent.providers import AIProvider, MockProvider
from local_agent.storage import JsonFileStorage, TaskStorage
from local_agent.tool_engine import ToolEngine
from local_agent.tools import ToolRegistry
from local_agent.worktree import WorktreeManager


def _make_task(task_id: str, objective: str, plan: TaskPlan | None = None) -> Task:
    now = datetime.datetime.now(datetime.timezone.utc)
    return Task(
        task_id=task_id,
        objective=objective,
        status=TaskStatus.PENDING,
        created_at=now,
        updated_at=now,
        plan=plan,
    )


class MockInteractiveProvider(AIProvider):
    """Configurable mock AI provider for testing InteractiveCodingAgent multi-turn tool loops."""

    def __init__(
        self,
        name: str = "mock-interactive",
        model: str = "mock-coder-v1",
        tool_responses: list[ToolCall | list[FileOperation]] | None = None,
        single_shot_ops: list[FileOperation] | None = None,
        capabilities: set[ProviderCapability] | None = None,
    ):
        super().__init__()
        self.provider_id = name
        self.model = model
        self.tool_responses = list(tool_responses or [])
        self.single_shot_ops = single_shot_ops if single_shot_ops is not None else []
        self._capabilities = capabilities or {
            ProviderCapability.PLANNING,
            ProviderCapability.IMPLEMENTATION,
            ProviderCapability.REPAIR,
            ProviderCapability.REVIEW,
            ProviderCapability.TOOL_USE,
        }
        self.calls_received: list[dict[str, Any]] = []

    @property
    def capabilities(self) -> set[ProviderCapability]:
        return self._capabilities

    def generate_plan(self, task: str | Task, context: ProjectContext) -> Plan:
        task_text = task.objective if hasattr(task, "objective") else str(task)
        inspect = list(context.source_files) or ["src/main.py", "src/calc.py"]
        return Plan(
            objective=task_text,
            files_to_inspect=inspect,
            files_likely_to_change=inspect,
            files_likely_to_create=[],
            steps=["Implement changes"],
            validation_strategy=["python -m unittest"],
            risks=[],
        )

    def generate_code(
        self,
        task: str | Task,
        plan: Plan,
        context: ProjectContext,
        failure: FailureAnalysis | None = None,
        review: ReviewResult | None = None,
    ) -> list[FileOperation]:
        self.calls_received.append({
            "type": "single_shot",
            "task": str(task),
            "plan": plan,
        })
        return self.single_shot_ops

    def generate_code_with_tools(
        self,
        task: str,
        plan: Plan,
        context: ProjectContext,
        tools: list[ToolDefinition],
        tool_history: list[tuple[ToolCall, ToolResult]] | None = None,
        failure: FailureAnalysis | None = None,
        review: ReviewResult | None = None,
    ) -> ToolCall | list[FileOperation]:
        self.calls_received.append({
            "type": "tool_step",
            "task": task,
            "history_len": len(tool_history or []),
        })
        if self.tool_responses:
            return self.tool_responses.pop(0)
        return self.single_shot_ops

    def review_changes(self, task: str, plan: Plan, diff: str, context: ProjectContext) -> ReviewResult:
        return ReviewResult("APPROVED", "Changes verified and approved.", [])

    def analyze_failure(self, execution: ExecutionResult, diff: str, context: ProjectContext, plan: Plan) -> FailureAnalysis:
        return FailureAnalysis(
            probable_root_cause="Test failure",
            recommended_fix="Update logic",
            affected_files=list(plan.files_likely_to_change),
        )


class TestImplementationResultModel(unittest.TestCase):
    """Tests for ImplementationResult serialization, deserialization, and field integrity."""

    def test_implementation_result_defaults(self):
        res = ImplementationResult()
        self.assertFalse(res.success)
        self.assertIsNone(res.file_operations)
        self.assertEqual(res.summary, "")
        self.assertEqual(res.files_inspected, [])
        self.assertEqual(res.files_modified, [])
        self.assertEqual(res.tool_steps_used, 0)
        self.assertFalse(res.used_fallback)

    def test_implementation_result_to_dict_and_from_dict_roundtrip(self):
        ops = [
            FileOperation("create", "src/auth.py", content="def login(): pass", reason="New auth module"),
            FileOperation("modify", "src/main.py", patch="--- a/src/main.py\n+++ b/src/main.py\n", reason="Add auth import"),
        ]
        metrics = ToolExecutionMetrics(
            total_calls=4,
            calls_by_tool={"read_file_range": 2, "grep_code": 2},
            steps_used=4,
            completed=True,
            elapsed_ms=120.5,
        )
        res = ImplementationResult(
            success=True,
            file_operations=ops,
            summary="Interactive coding finished in 4 steps",
            files_inspected=["src/main.py", "src/utils.py"],
            files_modified=["src/auth.py", "src/main.py"],
            tool_steps_used=4,
            elapsed_time_seconds=1.25,
            provider="mock-interactive",
            model="mock-coder-v1",
            termination_reason="completed",
            used_fallback=False,
            scope_violations=[],
            tool_history=[{"call": {"call_id": "c1", "tool_name": "read_file_range", "arguments": {"path": "src/main.py"}}, "result": {"call_id": "c1", "tool_name": "read_file_range", "output": "print('hello')", "is_error": False}}],
            metrics=metrics,
            error_message=None,
        )

        d = res.to_dict()
        self.assertTrue(d["success"])
        self.assertEqual(len(d["file_operations"]), 2)
        self.assertEqual(d["provider"], "mock-interactive")
        self.assertEqual(d["model"], "mock-coder-v1")
        self.assertEqual(d["tool_steps_used"], 4)

        restored = ImplementationResult.from_dict(d)
        self.assertTrue(restored.success)
        self.assertEqual(len(restored.file_operations), 2)
        self.assertEqual(restored.file_operations[0].path, "src/auth.py")
        self.assertEqual(restored.files_inspected, ["src/main.py", "src/utils.py"])
        self.assertEqual(restored.files_modified, ["src/auth.py", "src/main.py"])
        self.assertEqual(restored.tool_steps_used, 4)
        self.assertEqual(restored.metrics.total_calls, 4)
        self.assertEqual(restored.metrics.calls_by_tool["read_file_range"], 2)

    def test_run_report_with_implementation_result(self):
        ctx = ProjectContext(root=".")
        res = ImplementationResult(
            success=True,
            file_operations=[FileOperation("modify", "src/main.py", "content", "reason")],
            summary="Success",
            files_modified=["src/main.py"],
            tool_steps_used=2,
        )
        report = RunReport(project=ctx, implementation_result=res)
        self.assertIsNotNone(report.implementation_result)
        self.assertTrue(report.implementation_result.success)
        self.assertEqual(report.implementation_result.tool_steps_used, 2)

    def test_implementation_result_with_none_or_empty_metrics(self):
        res = ImplementationResult(success=True, metrics=None)
        d = res.to_dict()
        self.assertIsNone(d["metrics"])
        restored = ImplementationResult.from_dict(d)
        self.assertIsNone(restored.metrics)

    def test_implementation_result_with_scope_violations_roundtrip(self):
        res = ImplementationResult(
            success=False,
            scope_violations=["out_of_scope.py"],
            termination_reason="scope_violation",
        )
        d = res.to_dict()
        self.assertEqual(d["scope_violations"], ["out_of_scope.py"])
        restored = ImplementationResult.from_dict(d)
        self.assertEqual(restored.scope_violations, ["out_of_scope.py"])


class TestInteractiveCodingAgentUnit(unittest.TestCase):
    """Unit tests for InteractiveCodingAgent execution loop, tool interaction, and telemetry."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir)
        self.filesystem = ProjectFilesystem(self.project_path)
        self.filesystem.create_file("src/main.py", "def main():\n    return 42\n")
        self.filesystem.create_file("src/helper.py", "def helper():\n    return 'help'\n")
        self.registry = ToolRegistry(self.project_path, filesystem=self.filesystem)
        self.plan = Plan(
            objective="Enhance main",
            files_likely_to_change=["src/main.py"],
            files_likely_to_create=["src/utils.py"],
            steps=["Inspect main", "Modify main"],
        )
        self.context = ProjectContext(root=str(self.project_path), source_files=["src/main.py", "src/helper.py"])

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_interactive_execution_successful_multi_turn(self):
        tool_call1 = ToolCall(call_id="call_1", tool_name="read_file_range", arguments={"path": "src/main.py", "start_line": 1, "end_line": 10})
        tool_call2 = ToolCall(call_id="call_2", tool_name="find_files", arguments={"pattern": "*.py"})
        final_ops = [
            FileOperation("modify", "src/main.py", content="def main():\n    return 100\n", reason="Updated return value"),
        ]
        provider = MockInteractiveProvider(tool_responses=[tool_call1, tool_call2, final_ops])
        agent = InteractiveCodingAgent(self.filesystem, self.registry, max_tool_steps=10)

        result = agent.execute(
            provider=provider,
            task_objective="Update main to return 100",
            plan=self.plan,
            context=self.context,
        )

        self.assertTrue(result.success)
        self.assertEqual(len(result.file_operations), 1)
        self.assertEqual(result.file_operations[0].path, "src/main.py")
        self.assertEqual(result.tool_steps_used, 2)
        self.assertIn("src/main.py", result.files_inspected)
        self.assertEqual(result.files_modified, ["src/main.py"])
        self.assertEqual(result.termination_reason, "completed")
        self.assertFalse(result.used_fallback)
        self.assertGreater(result.elapsed_time_seconds, 0.0)

    def test_interactive_execution_runs_command_sandbox_probe(self):
        probe_call = ToolCall(call_id="call_probe", tool_name="run_command_sandbox", arguments={"command": ["python", "-c", "print('probe ok')"]})
        final_ops = [
            FileOperation("create", "src/utils.py", content="def util(): pass", reason="Created util"),
        ]
        provider = MockInteractiveProvider(tool_responses=[probe_call, final_ops])
        agent = InteractiveCodingAgent(self.filesystem, self.registry, max_tool_steps=10)

        result = agent.execute(
            provider=provider,
            task_objective="Create utils after probe",
            plan=self.plan,
            context=self.context,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.tool_steps_used, 1)
        self.assertEqual(result.files_modified, ["src/utils.py"])
        self.assertEqual(len(result.tool_history), 1)
        self.assertIn("probe ok", result.tool_history[0]["result"]["output"])

    def test_interactive_execution_fallback_when_provider_lacks_tool_use(self):
        provider = MockInteractiveProvider(
            single_shot_ops=[FileOperation("modify", "src/main.py", "content", "reason")],
            capabilities={ProviderCapability.IMPLEMENTATION}, # No TOOL_USE
        )
        agent = InteractiveCodingAgent(self.filesystem, self.registry, max_tool_steps=10)

        result = agent.execute(
            provider=provider,
            task_objective="Single shot fallback task",
            plan=self.plan,
            context=self.context,
        )

        self.assertTrue(result.success)
        self.assertTrue(result.used_fallback)
        self.assertEqual(result.termination_reason, "single_shot_fallback")
        self.assertEqual(result.tool_steps_used, 0)
        self.assertEqual(len(result.file_operations), 1)

    def test_interactive_execution_terminates_when_max_tool_steps_reached(self):
        repeated_calls = [
            ToolCall(call_id=f"call_{i}", tool_name="read_file_range", arguments={"path": "src/main.py", "start_line": i + 1, "end_line": i + 5})
            for i in range(10)
        ]
        provider = MockInteractiveProvider(tool_responses=repeated_calls)
        agent = InteractiveCodingAgent(self.filesystem, self.registry, max_tool_steps=3)

        result = agent.execute(
            provider=provider,
            task_objective="Infinite exploration task",
            plan=self.plan,
            context=self.context,
        )

        self.assertFalse(result.success)
        self.assertIsNone(result.file_operations)
        self.assertEqual(result.tool_steps_used, 3)
        self.assertEqual(result.termination_reason, "max_steps_exceeded")

    def test_interactive_execution_detects_scope_violations_in_result(self):
        final_ops = [
            FileOperation("modify", "src/main.py", "def main(): pass", "Allowed"),
            FileOperation("create", "unplanned/secret.py", "secret", "Unplanned file outside plan"),
        ]
        provider = MockInteractiveProvider(tool_responses=[final_ops])
        agent = InteractiveCodingAgent(self.filesystem, self.registry, max_tool_steps=5)

        result = agent.execute(
            provider=provider,
            task_objective="Modify with out of scope file",
            plan=self.plan,
            context=self.context,
        )

        self.assertTrue(result.success)
        self.assertIn("unplanned/secret.py", result.scope_violations)

    def test_interactive_execution_respects_upstream_contracts(self):
        contract = SubtaskContract(
            subtask_id="sub-1",
            title="Helper Subtask",
            modified_files=["src/helper.py"],
            exported_symbols=[ExportedSymbol(symbol_id="src/helper.py::helper", name="helper", kind="function", file_path="src/helper.py", verified=True)],
        )
        final_ops = [FileOperation("modify", "src/main.py", "content", "reason")]
        provider = MockInteractiveProvider(tool_responses=[final_ops])
        agent = InteractiveCodingAgent(self.filesystem, self.registry, max_tool_steps=5)

        result = agent.execute(
            provider=provider,
            task_objective="Use upstream helper",
            plan=self.plan,
            context=self.context,
            upstream_contracts=[contract],
        )

        self.assertTrue(result.success)
        self.assertTrue(len(provider.calls_received) > 0)
        prompt_received = provider.calls_received[0]["task"]
        self.assertIn("UPSTREAM INTERFACE CONSTRAINTS", prompt_received)
        self.assertIn("helper", prompt_received)

    def test_interactive_execution_handles_persistent_knowledge_context(self):
        self.context.metadata["persistent_knowledge"] = "- Invariant: helper must not throw\n- File: src/helper.py"
        final_ops = [FileOperation("modify", "src/main.py", "content", "reason")]
        provider = MockInteractiveProvider(tool_responses=[final_ops])
        agent = InteractiveCodingAgent(self.filesystem, self.registry, max_tool_steps=5)

        result = agent.execute(
            provider=provider,
            task_objective="Task with persistent memory",
            plan=self.plan,
            context=self.context,
        )

        self.assertTrue(result.success)
        prompt_received = provider.calls_received[0]["task"]
        self.assertIn("PERSISTENT REPOSITORY KNOWLEDGE", prompt_received)
        self.assertIn("Invariant: helper must not throw", prompt_received)

    def test_interactive_execution_handles_circuit_breaker_repetition(self):
        # 4 identical tool calls to trigger circuit breaker
        call = ToolCall(call_id="c_rep", tool_name="read_file_range", arguments={"path": "src/main.py", "start_line": 1, "end_line": 10})
        provider = MockInteractiveProvider(tool_responses=[call, call, call, call, [FileOperation("modify", "src/main.py", "fixed", "reason")]])
        policy = ToolExecutionPolicy(max_tool_steps=10, max_consecutive_repeats=3)
        agent = InteractiveCodingAgent(self.filesystem, self.registry, policy=policy, max_tool_steps=10)

        result = agent.execute(
            provider=provider,
            task_objective="Repetition test",
            plan=self.plan,
            context=self.context,
        )

        self.assertTrue(result.success)
        self.assertGreaterEqual(result.tool_steps_used, 3)

    def test_interactive_execution_handles_provider_error_gracefully(self):
        class CrashingProvider(MockInteractiveProvider):
            def generate_code_with_tools(self, *args, **kwargs):
                raise ProviderError("Inference engine connection lost")

        provider = CrashingProvider()
        agent = InteractiveCodingAgent(self.filesystem, self.registry, max_tool_steps=5)

        with self.assertRaises(ProviderError):
            agent.execute(
                provider=provider,
                task_objective="Crash test",
                plan=self.plan,
                context=self.context,
            )

    def test_interactive_execution_aggregates_inspected_files_across_tools(self):
        call1 = ToolCall(call_id="c1", tool_name="read_file_range", arguments={"path": "src/main.py"})
        call2 = ToolCall(call_id="c2", tool_name="read_file_range", arguments={"path": "src/helper.py"})
        final_ops = [FileOperation("modify", "src/main.py", "done", "reason")]
        provider = MockInteractiveProvider(tool_responses=[call1, call2, final_ops])
        agent = InteractiveCodingAgent(self.filesystem, self.registry, max_tool_steps=5)

        result = agent.execute(
            provider=provider,
            task_objective="Inspect multiple files",
            plan=self.plan,
            context=self.context,
        )

        self.assertTrue(result.success)
        self.assertIn("src/main.py", result.files_inspected)
        self.assertIn("src/helper.py", result.files_inspected)


class TestInteractiveCodingSafetyAndSandbox(unittest.TestCase):
    """Tests verifying safety boundaries, protected directories, and sandbox isolation."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir)
        self.filesystem = ProjectFilesystem(self.project_path)
        self.filesystem.create_file("app.py", "print('app')")
        self.registry = ToolRegistry(self.project_path, filesystem=self.filesystem)
        self.plan = Plan(objective="Safety test", files_likely_to_change=["app.py"])
        self.context = ProjectContext(root=str(self.project_path), source_files=["app.py"])

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_tool_registry_rejects_reading_protected_dirs(self):
        call_git = ToolCall(call_id="c1", tool_name="read_file_range", arguments={"path": ".git/config"})
        res = self.registry.execute(call_git)
        self.assertTrue(res.is_error)
        self.assertIn("protected", res.output.lower())

        call_worktree = ToolCall(call_id="c2", tool_name="read_file_range", arguments={"path": ".agent_worktrees/task-1/sub-1/app.py"})
        res_wt = self.registry.execute(call_worktree)
        self.assertTrue(res_wt.is_error)
        self.assertIn("protected", res_wt.output.lower())

    def test_tool_registry_rejects_reading_secret_files(self):
        call_secret = ToolCall(call_id="c_sec", tool_name="read_file_range", arguments={"path": ".env"})
        res = self.registry.execute(call_secret)
        self.assertTrue(res.is_error)
        self.assertIn("secret", res.output.lower())

    def test_tool_registry_rejects_path_traversal_escape(self):
        call_escape = ToolCall(call_id="c_esc", tool_name="read_file_range", arguments={"path": "../../outside.txt"})
        res = self.registry.execute(call_escape)
        self.assertTrue(res.is_error)
        self.assertTrue("outside" in res.output.lower() or "denied" in res.output.lower())

    def test_interactive_agent_does_not_mutate_outside_worktree(self):
        call_find = ToolCall(call_id="c_f", tool_name="find_files", arguments={"pattern": "*.py"})
        final_ops = [FileOperation("create", "new_file.py", content="test", reason="create")]
        provider = MockInteractiveProvider(tool_responses=[call_find, final_ops])
        agent = InteractiveCodingAgent(self.filesystem, self.registry, max_tool_steps=5)

        res = agent.execute(provider, "test task", self.plan, self.context)
        self.assertTrue(res.success)
        # Verify InteractiveCodingAgent only returns operations without direct side-effect file write
        self.assertFalse((self.project_path / "new_file.py").exists())

    def test_tool_registry_rejects_unsafe_command_injection(self):
        call_unsafe = ToolCall(call_id="c_inj", tool_name="run_command_sandbox", arguments={"command": ["rm", "-rf", "/"]})
        res = self.registry.execute(call_unsafe)
        self.assertTrue(res.is_error)


class TestConfigurationAndCLI(unittest.TestCase):
    """Tests for interactive implementation configuration parsing, validation, and CLI flags."""

    def test_default_config_interactive_implementation_is_false(self):
        cfg = AgentConfig.from_environment(".")
        self.assertFalse(cfg.interactive_implementation)
        self.assertEqual(cfg.max_implementation_tool_steps, 15)

    def test_config_from_environment_overrides(self):
        old_env = os.environ.copy()
        try:
            os.environ["AGENT_INTERACTIVE_IMPLEMENTATION"] = "true"
            os.environ["AGENT_MAX_IMPLEMENTATION_TOOL_STEPS"] = "20"
            cfg = AgentConfig.from_environment(".")
            self.assertTrue(cfg.interactive_implementation)
            self.assertEqual(cfg.max_implementation_tool_steps, 20)
        finally:
            os.environ.clear()
            os.environ.update(old_env)

    def test_config_validation_max_implementation_tool_steps_positive(self):
        cfg = AgentConfig(project=Path(".").resolve(), max_implementation_tool_steps=0)
        with self.assertRaises(ValueError) as ctx:
            cfg.validate()
        self.assertIn("max_implementation_tool_steps must be positive", str(ctx.exception))

    def test_cli_argument_parsing_for_interactive_implementation(self):
        parser = argparse.ArgumentParser()
        add_common_arguments(parser)
        args = parser.parse_args(["--interactive-implementation", "true", "--max-implementation-tool-steps", "12"])
        cfg = config_from_args(args)
        self.assertTrue(cfg.interactive_implementation)
        self.assertEqual(cfg.max_implementation_tool_steps, 12)

    def test_config_validation_interactive_implementation_type(self):
        cfg = AgentConfig(project=Path(".").resolve(), interactive_implementation=True)
        cfg.validate() # Should not raise
        self.assertTrue(cfg.interactive_implementation)


class TestOrchestratorInteractiveIntegration(unittest.TestCase):
    """Integration tests for Orchestrator Stage [4/7] with InteractiveCodingAgent enabled vs disabled."""

    def setUp(self):
        import subprocess
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir)
        self.storage_dir = self.project_path / ".agent_data"
        self.storage = JsonFileStorage(self.storage_dir)
        self.repo_lock = threading.Lock()
        self.memory_lock = threading.Lock()

        # Initialize Git repo
        subprocess.run(["git", "init"], cwd=self.project_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "TestUser"], cwd=self.project_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=self.project_path, check=True, capture_output=True)
        self.git = GitIntegration(self.project_path)
        (self.project_path / "src").mkdir(parents=True, exist_ok=True)
        (self.project_path / "src" / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
        (self.project_path / "tests").mkdir(parents=True, exist_ok=True)
        (self.project_path / "tests" / "test_calc.py").write_text(
            "import unittest\nfrom src.calc import add\n\nclass TestCalc(unittest.TestCase):\n    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n",
            encoding="utf-8",
        )
        self.git.add(["."])
        self.git.commit("Initial commit with bug")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_orchestrator_runs_interactive_implementation_when_enabled(self):
        cfg = AgentConfig(
            project=self.project_path,
            provider="mock",
            interactive_implementation=True,
            max_implementation_tool_steps=10,
            validation_commands=["python -m unittest tests/test_calc.py"],
            max_iterations=2,
        )
        orchestrator = Orchestrator(cfg, self.storage, None, self.repo_lock, self.memory_lock)

        tool_call = ToolCall(call_id="call_inspect", tool_name="read_file_range", arguments={"path": "src/calc.py", "start_line": 1, "end_line": 10})
        fixed_ops = [
            FileOperation("modify", "src/calc.py", content="def add(a, b):\n    return a + b\n", reason="Fix addition bug"),
        ]
        mock_provider = MockInteractiveProvider(tool_responses=[tool_call, fixed_ops])
        orchestrator.router.get_provider_chain = lambda role: [mock_provider]

        task = _make_task("task-interactive-1", "Fix add function in calc.py")
        report = orchestrator.run(task)

        self.assertTrue(report.completed)
        self.assertEqual(task.status, TaskStatus.COMPLETED)
        self.assertIsNotNone(report.implementation_result)
        self.assertTrue(report.implementation_result.success)
        self.assertEqual(report.implementation_result.tool_steps_used, 1)
        self.assertIn("src/calc.py", report.implementation_result.files_inspected)
        self.assertIn("src/calc.py", report.changed_files)

        updated_content = (self.project_path / "src" / "calc.py").read_text(encoding="utf-8")
        self.assertIn("return a + b", updated_content)

    def test_orchestrator_runs_single_shot_when_interactive_disabled(self):
        cfg = AgentConfig(
            project=self.project_path,
            provider="mock",
            interactive_implementation=False,
            validation_commands=["python -m unittest tests/test_calc.py"],
            max_iterations=1,
        )
        orchestrator = Orchestrator(cfg, self.storage, None, self.repo_lock, self.memory_lock)

        fixed_ops = [
            FileOperation("modify", "src/calc.py", content="def add(a, b):\n    return a + b\n", reason="Fix addition bug"),
        ]
        mock_provider = MockInteractiveProvider(
            single_shot_ops=fixed_ops,
            capabilities={ProviderCapability.PLANNING, ProviderCapability.IMPLEMENTATION, ProviderCapability.REPAIR, ProviderCapability.REVIEW},
        )
        orchestrator.router.get_provider_chain = lambda role: [mock_provider]

        task = _make_task("task-single-shot-1", "Fix add function")
        report = orchestrator.run(task)

        self.assertTrue(report.completed)
        self.assertIsNone(report.implementation_result)
        self.assertTrue(any(call.get("type") == "single_shot" for call in mock_provider.calls_received))

    def test_checkpoint_persists_implementation_result(self):
        cfg = AgentConfig(
            project=self.project_path,
            provider="mock",
            interactive_implementation=True,
            max_implementation_tool_steps=10,
        )
        orchestrator = Orchestrator(cfg, self.storage, None, self.repo_lock, self.memory_lock)
        task = _make_task("task-chk-1", "Test checkpoint persistence")
        report = RunReport(
            project=ProjectContext(root=str(self.project_path)),
            implementation_result=ImplementationResult(
                success=True,
                file_operations=[FileOperation("modify", "src/calc.py", "content", "reason")],
                summary="Interactive implementation completed",
                files_inspected=["src/calc.py"],
                tool_steps_used=2,
                provider="mock-interactive",
            ),
        )
        chk = orchestrator._create_checkpoint(task, None, "After implementation", report.project, report)
        self.assertIn("implementation_result", chk.continuation_context)
        self.assertEqual(chk.continuation_context["implementation_result"]["provider"], "mock-interactive")
        self.assertEqual(chk.continuation_context["implementation_result"]["tool_steps_used"], 2)

    def test_orchestrator_handles_interactive_provider_error_pause(self):
        cfg = AgentConfig(
            project=self.project_path,
            provider="mock",
            interactive_implementation=True,
            max_implementation_tool_steps=10,
        )
        orchestrator = Orchestrator(cfg, self.storage, None, self.repo_lock, self.memory_lock)

        class RateLimitedProvider(MockInteractiveProvider):
            def generate_code_with_tools(self, *args, **kwargs):
                raise RateLimitError("Rate limit exceeded; retry after 30s", retry_after_seconds=30)

        orchestrator.router.get_provider_chain = lambda role: [RateLimitedProvider()]

        task = _make_task("task-rl-1", "Test rate limit handling")
        report = orchestrator.run(task)

        self.assertEqual(task.status, TaskStatus.PAUSED)
        self.assertEqual(task.outcome, "RATE_LIMIT")

    def test_orchestrator_interactive_downstream_contract_extraction(self):
        cfg = AgentConfig(
            project=self.project_path,
            provider="mock",
            interactive_implementation=True,
            max_implementation_tool_steps=10,
            validation_commands=["python -m unittest tests/test_calc.py"],
        )
        orchestrator = Orchestrator(cfg, self.storage, None, self.repo_lock, self.memory_lock)

        fixed_ops = [
            FileOperation("modify", "src/calc.py", content="def add(a, b):\n    return a + b\n\ndef multiply(a, b):\n    return a * b\n", reason="Add multiply"),
        ]
        mock_provider = MockInteractiveProvider(tool_responses=[fixed_ops])
        orchestrator.router.get_provider_chain = lambda role: [mock_provider]

        subtask = Subtask(subtask_id="sub-add-1", title="Add multiply function", goal="Implement multiply in calc.py")
        task = _make_task(
            "task-contract-1",
            "Implement multiply in calc.py",
            plan=TaskPlan(objective="DAG", subtasks=[subtask]),
        )
        report = orchestrator.run(task, subtask_id="sub-add-1")

        self.assertTrue(report.completed)
        self.assertEqual(subtask.status, SubtaskStatus.COMPLETED)
        self.assertIsNotNone(subtask.contract)
        self.assertTrue(any(s.name == "multiply" for s in subtask.contract.exported_symbols))


class TestWorktreeInteractiveCompatibility(unittest.TestCase):
    """Tests verifying that InteractiveCodingAgent operates cleanly inside isolated Git worktrees."""

    def setUp(self):
        import subprocess
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir)
        subprocess.run(["git", "init"], cwd=self.project_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "TestUser"], cwd=self.project_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=self.project_path, check=True, capture_output=True)
        self.git = GitIntegration(self.project_path)
        (self.project_path / "module.py").write_text("def get_value():\n    return 0\n", encoding="utf-8")
        self.git.add(["."])
        self.git.commit("Initial commit")
        self.worktree_manager = WorktreeManager(self.project_path, self.git)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_interactive_coding_inside_worktree_does_not_mutate_parent_before_merge(self):
        session = self.worktree_manager.create_worktree("task-wt-1", "sub-1")
        wt_path = Path(session.worktree_path)

        wt_fs = ProjectFilesystem(wt_path)
        wt_registry = ToolRegistry(wt_path, filesystem=wt_fs)

        tool_call = ToolCall(call_id="c_wt", tool_name="read_file_range", arguments={"path": "module.py", "start_line": 1, "end_line": 5})
        wt_ops = [
            FileOperation("modify", "module.py", content="def get_value():\n    return 999\n", reason="Worktree edit"),
        ]
        provider = MockInteractiveProvider(tool_responses=[tool_call, wt_ops])
        agent = InteractiveCodingAgent(wt_fs, wt_registry, max_tool_steps=5)

        plan = Plan(objective="Worktree task", files_likely_to_change=["module.py"])
        ctx = ProjectContext(root=str(wt_path), source_files=["module.py"])

        res = agent.execute(provider, "Worktree task", plan, ctx)
        self.assertTrue(res.success)

        # Apply prepared changes inside worktree
        coding_agent = CodingAgent(wt_fs)
        prepared = coding_agent.prepare(res.file_operations, plan)
        coding_agent.apply_prepared(prepared)

        # Verify worktree file is updated
        wt_content = (wt_path / "module.py").read_text(encoding="utf-8")
        self.assertIn("return 999", wt_content)

        # Verify parent repository file is UNTOUCHED
        parent_content = (self.project_path / "module.py").read_text(encoding="utf-8")
        self.assertIn("return 0", parent_content)

        # Clean up worktree
        self.worktree_manager.remove_worktree(session, force=True)

    def test_concurrent_interactive_workers_in_disjoint_worktrees(self):
        session1 = self.worktree_manager.create_worktree("task-par-1", "sub-1")
        session2 = self.worktree_manager.create_worktree("task-par-1", "sub-2")

        wt1 = Path(session1.worktree_path)
        wt2 = Path(session2.worktree_path)

        fs1 = ProjectFilesystem(wt1)
        fs2 = ProjectFilesystem(wt2)

        reg1 = ToolRegistry(wt1, filesystem=fs1)
        reg2 = ToolRegistry(wt2, filesystem=fs2)

        agent1 = InteractiveCodingAgent(fs1, reg1, max_tool_steps=5)
        agent2 = InteractiveCodingAgent(fs2, reg2, max_tool_steps=5)

        p1 = MockInteractiveProvider(tool_responses=[[FileOperation("modify", "module.py", "def get_value():\n    return 111\n", "worker 1")]])
        p2 = MockInteractiveProvider(tool_responses=[[FileOperation("modify", "module.py", "def get_value():\n    return 222\n", "worker 2")]])

        res1 = agent1.execute(p1, "Worker 1 task", Plan(objective="1", files_likely_to_change=["module.py"]), ProjectContext(root=str(wt1)))
        res2 = agent2.execute(p2, "Worker 2 task", Plan(objective="2", files_likely_to_change=["module.py"]), ProjectContext(root=str(wt2)))

        self.assertTrue(res1.success)
        self.assertTrue(res2.success)

        CodingAgent(fs1).apply(res1.file_operations)
        CodingAgent(fs2).apply(res2.file_operations)

        self.assertIn("111", (wt1 / "module.py").read_text(encoding="utf-8"))
        self.assertIn("222", (wt2 / "module.py").read_text(encoding="utf-8"))

        self.worktree_manager.remove_worktree(session1, force=True)
        self.worktree_manager.remove_worktree(session2, force=True)

    def test_interactive_implementation_with_parallel_coordinator(self):
        cfg = AgentConfig(
            project=self.project_path,
            provider="mock",
            parallel_worktree_execution=True,
            interactive_implementation=True,
            max_parallel_subtasks=2,
        )
        storage = JsonFileStorage(self.project_path / ".agent_data")
        coordinator = ParallelExecutionCoordinator(cfg, storage)

        sub1 = Subtask(subtask_id="sub-1", title="Feature A", goal="Implement feature A in file_a.py")
        sub2 = Subtask(subtask_id="sub-2", title="Feature B", goal="Implement feature B in file_b.py")
        task = _make_task("task-coord-1", "Parallel interactive task", plan=TaskPlan(objective="DAG", subtasks=[sub1, sub2]))

        runnable = coordinator.identify_runnable_subtasks(task)
        self.assertEqual(len(runnable), 2)
        batches = coordinator.partition_parallel_batches(runnable, coordinator.predict_file_conflicts(runnable))
        self.assertTrue(len(batches) >= 1)

class TestInteractiveToolSurfacePolicy(unittest.TestCase):
    """Tests for the intentionally restricted implementation tool surface."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir)
        self.filesystem = ProjectFilesystem(self.project_path)
        self.filesystem.create_file("src/main.py", "def main():\n    return 42\n")
        self.registry = ToolRegistry(self.project_path, filesystem=self.filesystem)
        self.plan = Plan(objective="Surface test", files_likely_to_change=["src/main.py"])
        self.context = ProjectContext(root=str(self.project_path), source_files=["src/main.py"])

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_default_surface_allows_only_inspection_and_probe_tools(self):
        agent = InteractiveCodingAgent(self.filesystem, self.registry)
        policy = agent.build_policy()
        # Every registry tool is inside the default implementation surface.
        self.assertEqual(policy.disallowed_tools, set())
        self.assertEqual(
            IMPLEMENTATION_TOOL_SURFACE,
            frozenset({"read_file_range", "search_symbols", "grep_code", "find_files", "run_command_sandbox"}),
        )

    def test_narrowed_surface_disallows_other_registry_tools(self):
        agent = InteractiveCodingAgent(
            self.filesystem, self.registry, allowed_tools={"read_file_range"}
        )
        policy = agent.build_policy()
        self.assertIn("grep_code", policy.disallowed_tools)
        self.assertIn("run_command_sandbox", policy.disallowed_tools)
        self.assertNotIn("read_file_range", policy.disallowed_tools)

    def test_policy_clamps_step_budget_to_implementation_limit(self):
        base = ToolExecutionPolicy(max_tool_steps=50, max_tool_output_bytes=1234)
        agent = InteractiveCodingAgent(self.filesystem, self.registry, policy=base, max_tool_steps=5)
        policy = agent.build_policy()
        self.assertEqual(policy.max_tool_steps, 5)
        # Other policy dimensions are inherited, not reinvented.
        self.assertEqual(policy.max_tool_output_bytes, 1234)

    def test_policy_never_raises_a_lower_base_budget(self):
        base = ToolExecutionPolicy(max_tool_steps=3)
        agent = InteractiveCodingAgent(self.filesystem, self.registry, policy=base, max_tool_steps=30)
        self.assertEqual(agent.build_policy().max_tool_steps, 3)

    def test_policy_preserves_configured_disallowed_tools(self):
        base = ToolExecutionPolicy(max_tool_steps=10, disallowed_tools={"run_command_sandbox"})
        agent = InteractiveCodingAgent(self.filesystem, self.registry, policy=base)
        self.assertIn("run_command_sandbox", agent.build_policy().disallowed_tools)

    def test_out_of_surface_tool_call_is_rejected_at_runtime(self):
        blocked = ToolCall(call_id="c_blocked", tool_name="grep_code", arguments={"pattern": "main"})
        final_ops = [FileOperation("modify", "src/main.py", content="def main():\n    return 1\n", reason="ok")]
        provider = MockInteractiveProvider(tool_responses=[blocked, final_ops])
        agent = InteractiveCodingAgent(
            self.filesystem, self.registry, max_tool_steps=5, allowed_tools={"read_file_range"}
        )

        result = agent.execute(provider, "Blocked tool task", self.plan, self.context)

        self.assertTrue(result.success)
        self.assertEqual(result.tool_call_failures, 1)
        self.assertEqual(len(result.tool_history), 1)
        self.assertTrue(result.tool_history[0]["result"]["is_error"])
        self.assertIn("disallowed", result.tool_history[0]["result"]["output"].lower())
        # The blocked tool never ran, so nothing was recorded as inspected.
        self.assertEqual(result.files_inspected, [])


class TestInteractiveRefinementLoop(unittest.TestCase):
    """Tests for the EDIT -> RECHECK -> REFINE pre-mutation correction loop."""

    BAD_PATCH = (
        "--- a/src/main.py\n"
        "+++ b/src/main.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def main():\n"
        "-    return 999\n"
        "+    return 100\n"
    )

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir)
        self.filesystem = ProjectFilesystem(self.project_path)
        self.filesystem.create_file("src/main.py", "def main():\n    return 42\n")
        self.registry = ToolRegistry(self.project_path, filesystem=self.filesystem)
        self.plan = Plan(objective="Refine", files_likely_to_change=["src/main.py"])
        self.context = ProjectContext(root=str(self.project_path), source_files=["src/main.py"])

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_unappliable_patch_triggers_refinement_then_succeeds(self):
        bad_ops = [FileOperation("modify", "src/main.py", patch=self.BAD_PATCH, reason="bad patch")]
        good_ops = [FileOperation("modify", "src/main.py", content="def main():\n    return 100\n", reason="fixed")]
        provider = MockInteractiveProvider(tool_responses=[bad_ops, good_ops])
        agent = InteractiveCodingAgent(self.filesystem, self.registry, max_tool_steps=8, max_refine_rounds=2)

        result = agent.execute(provider, "Refine after bad patch", self.plan, self.context)

        self.assertTrue(result.success)
        self.assertEqual(result.file_operations, good_ops)
        # The rejected edit was fed back as a synthetic history turn, not silently dropped.
        precheck_turns = [h for h in result.tool_history if h["call"]["tool_name"] == "implementation_precheck"]
        self.assertEqual(len(precheck_turns), 1)
        self.assertTrue(precheck_turns[0]["result"]["is_error"])
        self.assertIn("patch does not apply cleanly", precheck_turns[0]["result"]["output"])
        # The provider actually saw the feedback on its second turn.
        self.assertEqual(provider.calls_received[0]["history_len"], 0)
        self.assertEqual(provider.calls_received[1]["history_len"], 1)
        # Nothing was written to disk by the agent itself.
        self.assertEqual(
            (self.project_path / "src" / "main.py").read_text(encoding="utf-8"),
            "def main():\n    return 42\n",
        )

    def test_invalid_python_syntax_triggers_refinement(self):
        broken = [FileOperation("modify", "src/main.py", content="def main(\n", reason="broken")]
        good = [FileOperation("modify", "src/main.py", content="def main():\n    return 7\n", reason="fixed")]
        provider = MockInteractiveProvider(tool_responses=[broken, good])
        agent = InteractiveCodingAgent(self.filesystem, self.registry, max_tool_steps=8, max_refine_rounds=2)

        result = agent.execute(provider, "Refine after syntax error", self.plan, self.context)

        self.assertTrue(result.success)
        self.assertEqual(result.file_operations, good)
        precheck = [h for h in result.tool_history if h["call"]["tool_name"] == "implementation_precheck"]
        self.assertEqual(len(precheck), 1)
        self.assertIn("resulting Python source is invalid", precheck[0]["result"]["output"])

    def test_repeated_precheck_failure_exhausts_refine_budget(self):
        bad_ops = [FileOperation("modify", "src/main.py", patch=self.BAD_PATCH, reason="bad patch")]
        # single_shot_ops is returned once tool_responses is exhausted, so every
        # round proposes the same unappliable edit.
        provider = MockInteractiveProvider(tool_responses=[], single_shot_ops=bad_ops)
        agent = InteractiveCodingAgent(self.filesystem, self.registry, max_tool_steps=8, max_refine_rounds=1)

        result = agent.execute(provider, "Never converges", self.plan, self.context)

        self.assertFalse(result.success)
        self.assertIsNone(result.file_operations)
        self.assertEqual(result.termination_reason, "no_operations")
        self.assertEqual(result.failure_category, "incomplete_implementation")
        self.assertTrue(result.is_recoverable_failure)
        self.assertIn("pre-mutation check", result.error_message)
        self.assertIn("patch does not apply cleanly", result.error_message)

    def test_refinement_can_be_disabled(self):
        bad_ops = [FileOperation("modify", "src/main.py", patch=self.BAD_PATCH, reason="bad patch")]
        provider = MockInteractiveProvider(tool_responses=[bad_ops])
        agent = InteractiveCodingAgent(self.filesystem, self.registry, max_tool_steps=8, max_refine_rounds=0)

        result = agent.execute(provider, "No refinement", self.plan, self.context)

        self.assertFalse(result.success)
        self.assertEqual(len(provider.calls_received), 1)
        self.assertEqual([h for h in result.tool_history if h["call"]["tool_name"] == "implementation_precheck"], [])

    def test_operation_without_content_or_patch_is_rejected(self):
        empty = [FileOperation("modify", "src/main.py", reason="no payload")]
        provider = MockInteractiveProvider(tool_responses=[], single_shot_ops=empty)
        agent = InteractiveCodingAgent(self.filesystem, self.registry, max_tool_steps=4, max_refine_rounds=0)

        result = agent.execute(provider, "Empty op", self.plan, self.context)

        self.assertFalse(result.success)
        self.assertIn("neither complete content nor a unified patch", result.error_message)

    def test_deletion_and_valid_creation_pass_the_precheck(self):
        ops = [
            FileOperation("create", "src/new_mod.py", content="VALUE = 1\n", reason="new"),
            FileOperation("delete", "src/gone.py", reason="remove"),
        ]
        provider = MockInteractiveProvider(tool_responses=[ops])
        agent = InteractiveCodingAgent(self.filesystem, self.registry, max_tool_steps=4)

        result = agent.execute(provider, "Create and delete", self.plan, self.context)

        self.assertTrue(result.success)
        self.assertEqual(result.files_modified, ["src/new_mod.py", "src/gone.py"])


class TestInteractiveImplementationTelemetry(unittest.TestCase):
    """Tests for implementation-level telemetry and failure classification."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir)
        self.filesystem = ProjectFilesystem(self.project_path)
        self.filesystem.create_file("src/main.py", "def main():\n    return 42\n")
        self.registry = ToolRegistry(self.project_path, filesystem=self.filesystem)
        self.plan = Plan(objective="Telemetry", files_likely_to_change=["src/main.py"])
        self.context = ProjectContext(root=str(self.project_path), source_files=["src/main.py"])

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_failed_probe_then_recovery_is_counted(self):
        failing_probe = ToolCall(
            call_id="p1",
            tool_name="run_command_sandbox",
            arguments={"command": ["python", "-c", "import sys; sys.exit(3)"]},
        )
        passing_probe = ToolCall(
            call_id="p2",
            tool_name="run_command_sandbox",
            arguments={"command": ["python", "-c", "print('ok')"]},
        )
        ops = [FileOperation("modify", "src/main.py", content="def main():\n    return 5\n", reason="fix")]
        provider = MockInteractiveProvider(tool_responses=[failing_probe, passing_probe, ops])
        agent = InteractiveCodingAgent(self.filesystem, self.registry, max_tool_steps=8)

        result = agent.execute(provider, "Probe telemetry", self.plan, self.context)

        self.assertTrue(result.success)
        self.assertEqual(result.validation_attempts, 2)
        self.assertEqual(result.validation_failures, 1)
        self.assertEqual(result.tool_call_failures, 1)
        self.assertEqual(result.tool_call_successes, 1)
        self.assertEqual(result.recovery_attempts, 1)
        self.assertEqual(result.failure_category, "none")
        self.assertFalse(result.is_recoverable_failure)

    def test_metrics_object_is_the_shared_tool_execution_metrics(self):
        call = ToolCall(call_id="c1", tool_name="read_file_range", arguments={"path": "src/main.py"})
        ops = [FileOperation("modify", "src/main.py", content="def main():\n    return 5\n", reason="fix")]
        provider = MockInteractiveProvider(tool_responses=[call, ops])
        agent = InteractiveCodingAgent(self.filesystem, self.registry, max_tool_steps=8)

        result = agent.execute(provider, "Metrics reuse", self.plan, self.context)

        self.assertIsInstance(result.metrics, ToolExecutionMetrics)
        self.assertEqual(result.metrics.total_calls, 1)
        self.assertEqual(result.metrics.calls_by_tool["read_file_range"], 1)
        self.assertTrue(result.metrics.completed)
        self.assertGreater(result.metrics.elapsed_ms, 0.0)

    def test_budget_exhaustion_is_classified(self):
        calls = [
            ToolCall(call_id=f"c{i}", tool_name="read_file_range", arguments={"path": "src/main.py", "start_line": i + 1, "end_line": i + 3})
            for i in range(10)
        ]
        provider = MockInteractiveProvider(tool_responses=calls)
        agent = InteractiveCodingAgent(self.filesystem, self.registry, max_tool_steps=2)

        result = agent.execute(provider, "Budget", self.plan, self.context)

        self.assertFalse(result.success)
        self.assertEqual(result.termination_reason, "max_steps_exceeded")
        self.assertEqual(result.failure_category, "budget_exhaustion")
        self.assertTrue(result.is_recoverable_failure)

    def test_invalid_provider_response_is_classified(self):
        class GarbageProvider(MockInteractiveProvider):
            def generate_code_with_tools(self, *args, **kwargs):
                return "not a tool call or operation list"

        agent = InteractiveCodingAgent(self.filesystem, self.registry, max_tool_steps=4)
        result = agent.execute(GarbageProvider(), "Garbage", self.plan, self.context)

        self.assertFalse(result.success)
        self.assertEqual(result.termination_reason, "invalid_provider_response")
        self.assertEqual(result.failure_category, "provider_failure")

    def test_termination_reason_categorisation_table(self):
        self.assertEqual(ImplementationTerminationReason.categorize("completed"), "none")
        self.assertEqual(ImplementationTerminationReason.categorize("budget_exhausted"), "budget_exhaustion")
        self.assertEqual(ImplementationTerminationReason.categorize("consecutive_repeats_exceeded"), "loop_detected")
        self.assertEqual(ImplementationTerminationReason.categorize("scope_violation"), "scope_violation")
        self.assertEqual(ImplementationTerminationReason.categorize(None), "unknown")
        self.assertEqual(ImplementationTerminationReason.categorize("something_new"), "unknown")

    def test_telemetry_survives_serialization_roundtrip(self):
        result = ImplementationResult(
            success=False,
            termination_reason="budget_exhausted",
            failure_category="budget_exhaustion",
            tool_call_failures=2,
            tool_call_successes=3,
            validation_attempts=2,
            validation_failures=1,
            recovery_attempts=1,
            circuit_breaker_events=1,
        )
        restored = ImplementationResult.from_dict(result.to_dict())
        self.assertEqual(restored.tool_call_failures, 2)
        self.assertEqual(restored.tool_call_successes, 3)
        self.assertEqual(restored.validation_attempts, 2)
        self.assertEqual(restored.validation_failures, 1)
        self.assertEqual(restored.recovery_attempts, 1)
        self.assertEqual(restored.circuit_breaker_events, 1)
        self.assertEqual(restored.failure_category, "budget_exhaustion")
        self.assertTrue(restored.is_recoverable_failure)


class TestInteractiveProviderBehaviour(unittest.TestCase):
    """Provider compatibility: tool-capable, tool-incapable, erroring, and fallback."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir)
        self.filesystem = ProjectFilesystem(self.project_path)
        self.filesystem.create_file("src/main.py", "def main():\n    return 42\n")
        self.registry = ToolRegistry(self.project_path, filesystem=self.filesystem)
        self.plan = Plan(objective="Providers", files_likely_to_change=["src/main.py"])
        self.context = ProjectContext(root=str(self.project_path), source_files=["src/main.py"])

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_builtin_mock_provider_falls_back_to_single_shot(self):
        agent = InteractiveCodingAgent(self.filesystem, self.registry, max_tool_steps=5)
        result = agent.execute(MockProvider(), "Offline mock task", self.plan, self.context)

        self.assertTrue(result.success)
        self.assertTrue(result.used_fallback)
        self.assertEqual(result.termination_reason, "single_shot_fallback")
        self.assertEqual(result.file_operations, [])
        self.assertEqual(result.tool_steps_used, 0)

    def test_quota_exceeded_error_propagates_for_router_fallback(self):
        class QuotaProvider(MockInteractiveProvider):
            def generate_code_with_tools(self, *args, **kwargs):
                raise QuotaExceededError("quota exhausted")

        agent = InteractiveCodingAgent(self.filesystem, self.registry, max_tool_steps=5)
        with self.assertRaises(QuotaExceededError):
            agent.execute(QuotaProvider(), "Quota", self.plan, self.context)

    def test_single_shot_fallback_propagates_provider_error(self):
        class BrokenSingleShot(MockInteractiveProvider):
            def generate_code(self, *args, **kwargs):
                raise ProviderError("single-shot transport failure")

        provider = BrokenSingleShot(capabilities={ProviderCapability.IMPLEMENTATION})
        agent = InteractiveCodingAgent(self.filesystem, self.registry, max_tool_steps=5)
        with self.assertRaises(ProviderError):
            agent.execute(provider, "Broken fallback", self.plan, self.context)

    def test_prompt_includes_protocol_and_failure_repair_section(self):
        agent = InteractiveCodingAgent(self.filesystem, self.registry, max_tool_steps=5)
        failure = FailureAnalysis(
            probable_root_cause="add() subtracts instead of adding",
            recommended_fix="Use + instead of -",
            affected_files=["src/main.py"],
        )
        prompt = agent.build_prompt("Fix add", failure=failure, context=self.context)

        self.assertIn("INTERACTIVE IMPLEMENTATION PROTOCOL", prompt)
        self.assertIn("UNDERSTAND", prompt)
        self.assertIn("REFINE", prompt)
        self.assertIn("PRIOR FAILURE TO REPAIR", prompt)
        self.assertIn("add() subtracts instead of adding", prompt)
        self.assertIn("read_file_range", prompt)

    def test_prompt_prefixes_subtask_goal(self):
        agent = InteractiveCodingAgent(self.filesystem, self.registry)
        subtask = Subtask(subtask_id="s1", title="T", goal="Implement multiply()")
        prompt = agent.build_prompt("Overall objective", subtask=subtask)
        self.assertIn("Subtask Goal: Implement multiply()", prompt)
        self.assertIn("Task Objective: Overall objective", prompt)


class TestOrchestratorInteractiveFallbackAndScope(unittest.TestCase):
    """Orchestrator-level behaviour: specialist fallback and out-of-scope handling."""

    def setUp(self):
        import subprocess
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir)
        self.storage = JsonFileStorage(self.project_path / ".agent_data")
        self.repo_lock = threading.Lock()
        self.memory_lock = threading.Lock()
        subprocess.run(["git", "init"], cwd=self.project_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "TestUser"], cwd=self.project_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=self.project_path, check=True, capture_output=True)
        self.git = GitIntegration(self.project_path)
        (self.project_path / "src").mkdir(parents=True, exist_ok=True)
        (self.project_path / "src" / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
        (self.project_path / "tests").mkdir(parents=True, exist_ok=True)
        (self.project_path / "tests" / "test_calc.py").write_text(
            "import unittest\nfrom src.calc import add\n\n"
            "class TestCalc(unittest.TestCase):\n    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n",
            encoding="utf-8",
        )
        self.git.add(["."])
        self.git.commit("Initial commit with bug")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _config(self, **overrides):
        params = dict(
            project=self.project_path,
            provider="mock",
            interactive_implementation=True,
            max_implementation_tool_steps=10,
            validation_commands=["python -m unittest tests/test_calc.py"],
            max_iterations=2,
        )
        params.update(overrides)
        return AgentConfig(**params)

    def test_interactive_failure_falls_back_to_next_specialist_provider(self):
        orchestrator = Orchestrator(self._config(), self.storage, None, self.repo_lock, self.memory_lock)

        class DeadProvider(MockInteractiveProvider):
            def generate_code_with_tools(self, *args, **kwargs):
                raise ProviderError("primary implementation provider is down")

        healthy = MockInteractiveProvider(
            tool_responses=[[FileOperation("modify", "src/calc.py", content="def add(a, b):\n    return a + b\n", reason="fix")]],
        )
        orchestrator.router.get_provider_chain = lambda role: [DeadProvider(name="dead"), healthy]

        report = orchestrator.run(_make_task("task-fb-1", "Fix add function in calc.py"))

        self.assertTrue(report.completed)
        self.assertIsNotNone(report.implementation_result)
        self.assertTrue(report.implementation_result.success)
        self.assertEqual(report.implementation_result.provider, "mock-interactive")
        self.assertIn("return a + b", (self.project_path / "src" / "calc.py").read_text(encoding="utf-8"))

    def test_interactive_result_records_telemetry_on_the_report(self):
        orchestrator = Orchestrator(self._config(), self.storage, None, self.repo_lock, self.memory_lock)
        inspect_call = ToolCall(
            call_id="c1", tool_name="read_file_range", arguments={"path": "src/calc.py", "start_line": 1, "end_line": 5}
        )
        provider = MockInteractiveProvider(
            tool_responses=[inspect_call, [FileOperation("modify", "src/calc.py", content="def add(a, b):\n    return a + b\n", reason="fix")]],
        )
        orchestrator.router.get_provider_chain = lambda role: [provider]

        report = orchestrator.run(_make_task("task-tel-1", "Fix add function in calc.py"))

        self.assertTrue(report.completed)
        impl = report.implementation_result
        self.assertEqual(impl.tool_call_successes, 1)
        self.assertEqual(impl.tool_call_failures, 0)
        self.assertEqual(impl.failure_category, "none")
        self.assertGreater(impl.elapsed_time_seconds, 0.0)
        # The shared tool metrics list is reused, not duplicated by a parallel system.
        self.assertTrue(any(m is impl.metrics for m in report.tool_metrics))
        self.assertEqual(len(report.tool_history), 1)

    def test_protected_path_operation_is_refused_by_the_apply_pipeline(self):
        orchestrator = Orchestrator(self._config(max_iterations=1), self.storage, None, self.repo_lock, self.memory_lock)
        provider = MockInteractiveProvider(
            tool_responses=[[FileOperation("modify", "../escape.py", content="pwned = True\n", reason="escape")]],
        )
        orchestrator.router.get_provider_chain = lambda role: [provider]

        report = orchestrator.run(_make_task("task-esc-1", "Attempt sandbox escape"))

        self.assertFalse(report.completed)
        # Nothing was written outside the project root.
        self.assertFalse((self.project_path.parent / "escape.py").exists())

    def test_config_toggle_switches_between_interactive_and_single_shot(self):
        for enabled in (True, False):
            with self.subTest(interactive=enabled):
                (self.project_path / "src" / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
                orchestrator = Orchestrator(
                    self._config(interactive_implementation=enabled, max_iterations=1),
                    self.storage, None, self.repo_lock, self.memory_lock,
                )
                ops = [FileOperation("modify", "src/calc.py", content="def add(a, b):\n    return a + b\n", reason="fix")]
                provider = MockInteractiveProvider(tool_responses=[ops] if enabled else [], single_shot_ops=ops)
                if not enabled:
                    provider._capabilities = {
                        ProviderCapability.PLANNING, ProviderCapability.IMPLEMENTATION,
                        ProviderCapability.REPAIR, ProviderCapability.REVIEW,
                    }
                orchestrator.router.get_provider_chain = lambda role: [provider]

                report = orchestrator.run(_make_task(f"task-toggle-{enabled}", "Fix add function in calc.py"))

                self.assertTrue(report.completed)
                if enabled:
                    self.assertIsNotNone(report.implementation_result)
                else:
                    self.assertIsNone(report.implementation_result)
                    self.assertTrue(any(c.get("type") == "single_shot" for c in provider.calls_received))


class TestInteractiveWorktreeIsolation(unittest.TestCase):
    """Worktree-rooted agents must not touch the parent checkout or each other."""

    def setUp(self):
        import subprocess
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir)
        subprocess.run(["git", "init"], cwd=self.project_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "TestUser"], cwd=self.project_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=self.project_path, check=True, capture_output=True)
        self.git = GitIntegration(self.project_path)
        (self.project_path / "module.py").write_text("def get_value():\n    return 0\n", encoding="utf-8")
        self.git.add(["."])
        self.git.commit("Initial commit")
        self.worktree_manager = WorktreeManager(self.project_path, self.git)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_worktree_agent_inspection_is_rooted_in_the_worktree(self):
        session = self.worktree_manager.create_worktree("task-iso-1", "sub-1")
        wt_path = Path(session.worktree_path)
        try:
            wt_fs = ProjectFilesystem(wt_path)
            wt_registry = ToolRegistry(wt_path, filesystem=wt_fs)
            agent = InteractiveCodingAgent(wt_fs, wt_registry, max_tool_steps=5)

            find_call = ToolCall(call_id="c1", tool_name="find_files", arguments={"pattern": "*.py"})
            ops = [FileOperation("modify", "module.py", content="def get_value():\n    return 7\n", reason="edit")]
            provider = MockInteractiveProvider(tool_responses=[find_call, ops])

            result = agent.execute(
                provider,
                "Worktree scoped task",
                Plan(objective="wt", files_likely_to_change=["module.py"]),
                ProjectContext(root=str(wt_path)),
            )

            self.assertTrue(result.success)
            self.assertEqual(result.files_inspected, ["module.py"])
            # Agent itself performs no writes anywhere.
            self.assertIn("return 0", (wt_path / "module.py").read_text(encoding="utf-8"))
            self.assertIn("return 0", (self.project_path / "module.py").read_text(encoding="utf-8"))
        finally:
            self.worktree_manager.remove_worktree(session, force=True)

    def test_parallel_worktree_agents_hold_no_shared_state(self):
        s1 = self.worktree_manager.create_worktree("task-iso-2", "sub-1")
        s2 = self.worktree_manager.create_worktree("task-iso-2", "sub-2")
        try:
            fs1, fs2 = ProjectFilesystem(Path(s1.worktree_path)), ProjectFilesystem(Path(s2.worktree_path))
            a1 = InteractiveCodingAgent(fs1, ToolRegistry(Path(s1.worktree_path), filesystem=fs1), max_tool_steps=4)
            a2 = InteractiveCodingAgent(fs2, ToolRegistry(Path(s2.worktree_path), filesystem=fs2), max_tool_steps=4)

            self.assertIsNot(a1.filesystem, a2.filesystem)
            self.assertIsNot(a1.registry, a2.registry)
            self.assertNotEqual(a1.registry.root, a2.registry.root)
            self.assertNotEqual(a1.filesystem.root, a2.filesystem.root)

            results = {}

            def _run(key, agent, value):
                provider = MockInteractiveProvider(
                    tool_responses=[[FileOperation("modify", "module.py", content=f"def get_value():\n    return {value}\n", reason=key)]],
                )
                results[key] = agent.execute(
                    provider, f"{key} task",
                    Plan(objective=key, files_likely_to_change=["module.py"]),
                    ProjectContext(root=str(agent.filesystem.root)),
                )

            threads = [
                threading.Thread(target=_run, args=("w1", a1, 111)),
                threading.Thread(target=_run, args=("w2", a2, 222)),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

            self.assertTrue(results["w1"].success)
            self.assertTrue(results["w2"].success)
            self.assertEqual(results["w1"].file_operations[0].content.strip().endswith("return 111"), True)
            self.assertEqual(results["w2"].file_operations[0].content.strip().endswith("return 222"), True)

            CodingAgent(fs1).apply(results["w1"].file_operations)
            CodingAgent(fs2).apply(results["w2"].file_operations)

            self.assertIn("111", (Path(s1.worktree_path) / "module.py").read_text(encoding="utf-8"))
            self.assertIn("222", (Path(s2.worktree_path) / "module.py").read_text(encoding="utf-8"))
            self.assertIn("return 0", (self.project_path / "module.py").read_text(encoding="utf-8"))
        finally:
            self.worktree_manager.remove_worktree(s1, force=True)
            self.worktree_manager.remove_worktree(s2, force=True)

class TestInteractiveCheckpointRoundTrip(unittest.TestCase):
    """Checkpoint write AND restore for the interactive implementation result."""

    def setUp(self):
        import subprocess
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir)
        self.storage = JsonFileStorage(self.project_path / ".agent_data")
        self.repo_lock = threading.Lock()
        self.memory_lock = threading.Lock()
        subprocess.run(["git", "init"], cwd=self.project_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "TestUser"], cwd=self.project_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=self.project_path, check=True, capture_output=True)
        self.git = GitIntegration(self.project_path)
        (self.project_path / "src").mkdir(parents=True, exist_ok=True)
        (self.project_path / "src" / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
        self.git.add(["."])
        self.git.commit("Initial commit")
        self.cfg = AgentConfig(
            project=self.project_path,
            provider="mock",
            interactive_implementation=True,
            max_implementation_tool_steps=10,
        )
        self.orchestrator = Orchestrator(self.cfg, self.storage, None, self.repo_lock, self.memory_lock)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _sample_result(self) -> ImplementationResult:
        return ImplementationResult(
            success=True,
            file_operations=[FileOperation("modify", "src/calc.py", content="def add(a, b):\n    return a + b\n", reason="fix")],
            summary="Interactive implementation completed",
            files_inspected=["src/calc.py"],
            files_modified=["src/calc.py"],
            tool_steps_used=3,
            provider="mock-interactive",
            model="mock-coder-v1",
            termination_reason="completed",
            tool_call_successes=3,
            validation_attempts=1,
            validation_failures=1,
            recovery_attempts=1,
            failure_category="none",
        )

    def test_checkpoint_restores_implementation_result_into_report(self):
        task = _make_task("task-chk-rt", "Fix add")
        report = RunReport(project=ProjectContext(root=str(self.project_path)), implementation_result=self._sample_result())
        checkpoint = self.orchestrator._create_checkpoint(task, None, "After implementation", report.project, report)
        self.storage.save_checkpoint(checkpoint)

        loaded = self.storage.load_checkpoint(checkpoint.checkpoint_id)
        restored = ImplementationResult.from_dict(loaded.continuation_context["implementation_result"])

        self.assertTrue(restored.success)
        self.assertEqual(restored.tool_steps_used, 3)
        self.assertEqual(restored.provider, "mock-interactive")
        self.assertEqual(restored.model, "mock-coder-v1")
        self.assertEqual(restored.files_inspected, ["src/calc.py"])
        self.assertEqual(restored.validation_attempts, 1)
        self.assertEqual(restored.validation_failures, 1)
        self.assertEqual(restored.recovery_attempts, 1)
        self.assertEqual(restored.file_operations[0].content, "def add(a, b):\n    return a + b\n")

    def test_build_run_report_rehydrates_implementation_result_from_checkpoint(self):
        task = _make_task("task-chk-rt2", "Fix add")
        report = RunReport(project=ProjectContext(root=str(self.project_path)), implementation_result=self._sample_result())
        checkpoint = self.orchestrator._create_checkpoint(task, None, "After implementation", report.project, report)
        self.storage.save_checkpoint(checkpoint)
        task.latest_checkpoint_id = checkpoint.checkpoint_id

        rebuilt = self.orchestrator._build_run_report(task)

        self.assertIsNotNone(rebuilt.implementation_result)
        self.assertTrue(rebuilt.implementation_result.success)
        self.assertEqual(rebuilt.implementation_result.tool_steps_used, 3)
        self.assertEqual(rebuilt.implementation_result.provider, "mock-interactive")

    def test_build_run_report_without_checkpoint_has_no_implementation_result(self):
        rebuilt = self.orchestrator._build_run_report(_make_task("task-chk-rt3", "No checkpoint"))
        self.assertIsNone(rebuilt.implementation_result)


if __name__ == "__main__":
    unittest.main()