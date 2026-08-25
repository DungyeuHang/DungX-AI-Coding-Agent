from __future__ import annotations

import datetime
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

from local_agent.config import AgentConfig
from local_agent.models import (
    FileOperation,
    Plan,
    ProjectContext,
    ProviderCapability,
    ReviewResult,
    RunReport,
    Task,
    TaskStatus,
    ToolCall,
    ToolDefinition,
    ToolExecutionMetrics,
    ToolExecutionPolicy,
    ToolResult,
)
from local_agent.orchestrator import Orchestrator
from local_agent.providers import AIProvider
from local_agent.storage import JsonFileStorage
from local_agent.tool_engine import (
    ToolContextCompactor,
    ToolEngine,
    history_from_dict,
    history_to_dict,
)
from local_agent.tools import ToolRegistry


class DummyProvider(AIProvider):
    def __init__(self, responses: list[Any]):
        self.responses = list(responses)
        self.received_histories: list[list[tuple[ToolCall, ToolResult]]] = []

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
        self.received_histories.append(list(tool_history or []))
        if not self.responses:
            return [FileOperation("modify", "src/main.py", "print('hello')", "test", None)]
        return self.responses.pop(0)

    def generate_code(self, *args, **kwargs) -> list[FileOperation]:
        return [FileOperation("modify", "src/main.py", "print('hello')", "test", None)]

    def review_changes(self, task: str, plan: Plan, diff: str, context: ProjectContext) -> ReviewResult:
        return ReviewResult(verdict="APPROVED", summary="Looks good", findings=[])


class DummyScheduler:
    def __init__(self, prov_inst: Any):
        self.provider = "mock"
        self._prov_inst = prov_inst

    def _select_providers(self, task: Any, capabilities: Any) -> list[Any]:
        return [self]

    def _build_provider_instance(self, provider_name: str) -> Any:
        return self._prov_inst


class TestToolContextCompactor(unittest.TestCase):
    """Focused tests for ToolContextCompactor and Phase 4.1.4 context efficiency."""

    def setUp(self):
        self.compactor = ToolContextCompactor(window=2, max_context_bytes=8000)

    def test_empty_history(self):
        """Empty history returns empty model history and zero metrics."""
        model_hist, compacted, bytes_count = self.compactor.compact([])
        self.assertEqual(model_hist, [])
        self.assertEqual(compacted, 0)
        self.assertEqual(bytes_count, 0)

    def test_single_entry_history_uncompacted(self):
        """Single-entry history is within the recent window and is uncompacted."""
        call = ToolCall("c1", "find_files", {"pattern": "*.py"})
        res = ToolResult("c1", "find_files", "file1.py\nfile2.py", False, False)
        history = [(call, res)]

        model_hist, compacted, bytes_count = self.compactor.compact(history)
        self.assertEqual(len(model_hist), 1)
        self.assertEqual(compacted, 0)
        self.assertEqual(model_hist[0][1].output, "file1.py\nfile2.py")
        self.assertEqual(bytes_count, len("file1.py\nfile2.py".encode("utf-8")))

    def test_recent_window_remains_full_fidelity(self):
        """The last K turns (window=2) always remain 100% full fidelity."""
        long_output_1 = "\n".join([f"{i}: var_{i} = {i}" for i in range(1, 30)])
        long_output_2 = "\n".join([f"{i}: class Class_{i}: pass" for i in range(1, 30)])
        long_output_3 = "\n".join([f"{i}: var_{i} = {i}" for i in range(1, 30)])

        call1 = ToolCall("c1", "read_file_range", {"path": "a.py"})
        res1 = ToolResult("c1", "read_file_range", long_output_1, False, False)
        call2 = ToolCall("c2", "read_file_range", {"path": "b.py"})
        res2 = ToolResult("c2", "read_file_range", long_output_2, False, False)
        call3 = ToolCall("c3", "read_file_range", {"path": "c.py"})
        res3 = ToolResult("c3", "read_file_range", long_output_3, False, False)

        history = [(call1, res1), (call2, res2), (call3, res3)]

        model_hist, compacted, bytes_count = self.compactor.compact(history)
        self.assertEqual(len(model_hist), 3)
        self.assertEqual(compacted, 1)

        # Item 1 was compacted
        self.assertNotEqual(model_hist[0][1].output, long_output_1)
        self.assertIn("omitted for compaction", model_hist[0][1].output)

        # Items 2 and 3 (within window=2) are 100% identical to originals
        self.assertEqual(model_hist[1][1].output, long_output_2)
        self.assertEqual(model_hist[2][1].output, long_output_3)

    def test_canonical_history_never_mutated(self):
        """Compactor must never mutate the canonical history passed to it."""
        long_output = "\n".join([f"{i}: line content {i}" for i in range(1, 50)])
        call1 = ToolCall("c1", "read_file_range", {"path": "a.py"})
        res1 = ToolResult("c1", "read_file_range", long_output, False, False)
        call2 = ToolCall("c2", "find_files", {"pattern": "*.py"})
        res2 = ToolResult("c2", "find_files", "a.py\nb.py", False, False)
        call3 = ToolCall("c3", "find_files", {"pattern": "*.ts"})
        res3 = ToolResult("c3", "find_files", "c.ts\nd.ts", False, False)

        history = [(call1, res1), (call2, res2), (call3, res3)]
        original_output_1 = res1.output

        model_hist, compacted, _ = self.compactor.compact(history)
        self.assertEqual(compacted, 1)

        # Verify canonical history objects and output strings are strictly unmutated
        self.assertEqual(history[0][1].output, original_output_1)
        self.assertEqual(len(history), 3)

    def test_errors_and_circuit_breakers_always_preserved(self):
        """Errors and circuit-breaker results are NEVER compacted even when older than window."""
        err_output = "Error: File not found: nonexistent.py\nTraceback:\n  ..."
        breaker_output = "Circuit breaker triggered: repeated identical tool call 'grep_code' detected."

        call1 = ToolCall("c1", "read_file_range", {"path": "nonexistent.py"})
        res1 = ToolResult("c1", "read_file_range", err_output, is_error=True, truncated=False)

        call2 = ToolCall("c2", "grep_code", {"pattern": "bad"})
        res2 = ToolResult("c2", "grep_code", breaker_output, is_error=True, truncated=False)

        call3 = ToolCall("c3", "find_files", {"pattern": "*.py"})
        res3 = ToolResult("c3", "find_files", "main.py", is_error=False, truncated=False)

        call4 = ToolCall("c4", "find_files", {"pattern": "*.ts"})
        res4 = ToolResult("c4", "find_files", "app.ts", is_error=False, truncated=False)

        history = [(call1, res1), (call2, res2), (call3, res3), (call4, res4)]

        model_hist, compacted, _ = self.compactor.compact(history)
        self.assertEqual(compacted, 0)
        self.assertEqual(model_hist[0][1].output, err_output)
        self.assertEqual(model_hist[1][1].output, breaker_output)

    def test_compact_read_file_range_preserves_definitions_and_boundaries(self):
        """read_file_range preserves head/tail lines and definition signatures."""
        lines = [
            "1: from __future__ import annotations",
            "2: import os",
            "3: import sys",
            "4: x = 1",
            "5: y = 2",
            "6: z = 3",
            "7: def calculate_total(a, b):",
            "8:     return a + b",
            "9: class EngineRunner:",
            "10:     pass",
            "11: w = 4",
            "12: u = 5",
            "13: v = 6",
            "14: def final_step():",
            "15:     return True",
            "16: result = final_step()",
            "17: # end of file",
        ]
        output = "\n".join(lines)
        call1 = ToolCall("c1", "read_file_range", {"path": "src/calc.py", "start_line": 1, "end_line": 17})
        res1 = ToolResult("c1", "read_file_range", output, False, False)

        call2 = ToolCall("c2", "find_files", {"pattern": "*.py"})
        res2 = ToolResult("c2", "find_files", "calc.py", False, False)
        call3 = ToolCall("c3", "find_files", {"pattern": "*.ts"})
        res3 = ToolResult("c3", "find_files", "calc.ts", False, False)

        history = [(call1, res1), (call2, res2), (call3, res3)]
        model_hist, compacted, _ = self.compactor.compact(history)

        compacted_text = model_hist[0][1].output
        self.assertIn("1: from __future__ import annotations", compacted_text)
        self.assertIn("def calculate_total", compacted_text)
        self.assertIn("class EngineRunner:", compacted_text)
        self.assertIn("17: # end of file", compacted_text)
        self.assertIn("body lines omitted for compaction", compacted_text)

    def test_compact_search_and_grep_preserves_initial_matches(self):
        """find_files, grep_code, and search_symbols retain top 10 matches and count."""
        grep_lines = [f"src/file_{i}.py:{i}: match_{i}" for i in range(1, 25)]
        output = "\n".join(grep_lines)

        call1 = ToolCall("c1", "grep_code", {"pattern": "match"})
        res1 = ToolResult("c1", "grep_code", output, False, False)
        call2 = ToolCall("c2", "find_files", {"pattern": "*.py"})
        res2 = ToolResult("c2", "find_files", "a.py", False, False)
        call3 = ToolCall("c3", "find_files", {"pattern": "*.ts"})
        res3 = ToolResult("c3", "find_files", "b.ts", False, False)

        history = [(call1, res1), (call2, res2), (call3, res3)]
        model_hist, compacted, _ = self.compactor.compact(history)

        compacted_text = model_hist[0][1].output
        self.assertIn("src/file_1.py:1: match_1", compacted_text)
        self.assertIn("src/file_10.py:10: match_10", compacted_text)
        self.assertNotIn("src/file_15.py:15: match_15", compacted_text)
        self.assertIn("14 additional matches omitted for compaction", compacted_text)

    def test_preserves_call_id_and_tool_name(self):
        """Compacted ToolResult preserves call_id and tool_name identically."""
        long_output = "\n".join([f"line_{i}" for i in range(100)])
        call1 = ToolCall("unique-call-id-99", "read_file_range", {"path": "a.py"})
        res1 = ToolResult("unique-call-id-99", "read_file_range", long_output, False, False)
        call2 = ToolCall("c2", "find_files", {})
        res2 = ToolResult("c2", "find_files", "a.py", False, False)
        call3 = ToolCall("c3", "find_files", {})
        res3 = ToolResult("c3", "find_files", "b.py", False, False)

        history = [(call1, res1), (call2, res2), (call3, res3)]
        model_hist, _, _ = self.compactor.compact(history)

        self.assertEqual(model_hist[0][0].call_id, "unique-call-id-99")
        self.assertEqual(model_hist[0][0].tool_name, "read_file_range")
        self.assertEqual(model_hist[0][1].call_id, "unique-call-id-99")
        self.assertEqual(model_hist[0][1].tool_name, "read_file_range")


class TestToolEngineCompactionIntegration(unittest.TestCase):
    """Integration tests for ToolEngine context compaction in exploration loops."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "src").mkdir()
        (self.root / "src" / "main.py").write_text(
            "\n".join([f"line_{i} = {i}" for i in range(1, 60)]) + "\n",
            encoding="utf-8",
        )
        self.registry = ToolRegistry(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_tool_engine_runs_compaction_and_records_telemetry(self):
        """ToolEngine passes compacted history to provider and records telemetry metrics."""
        call1 = ToolCall("c1", "read_file_range", {"path": "src/main.py", "start_line": 1, "end_line": 50})
        call2 = ToolCall("c2", "find_files", {"pattern": "*.py"})
        call3 = ToolCall("c3", "find_files", {"pattern": "*.txt"})
        final_ops = [FileOperation("modify", "src/main.py", "print('done')", "done", None)]

        provider = DummyProvider([call1, call2, call3, final_ops])
        policy = ToolExecutionPolicy(max_tool_steps=8, compaction_window=2)
        engine = ToolEngine(provider, self.registry, policy=policy)

        task = "Explore repository"
        plan = Plan(objective="Explore", steps=["step 1"])
        context = ProjectContext(root=str(self.root))

        result = engine.run(task, plan, context)

        self.assertTrue(result.completed)
        self.assertEqual(len(result.tool_history), 3, "Canonical history must have all 3 calls")
        self.assertEqual(result.metrics.total_calls, 3)

        # On the 4th call (final_ops), history had 3 items; with window=2, item 1 (read_file_range) was compacted
        self.assertGreaterEqual(result.metrics.compacted_entries, 1)
        self.assertLess(result.metrics.model_context_bytes, result.metrics.total_output_bytes)

        # Verify provider received the compacted history on its 4th invocation
        last_received_history = provider.received_histories[-1]
        self.assertEqual(len(last_received_history), 3)
        self.assertIn("body lines omitted for compaction", last_received_history[0][1].output)
        # But canonical history in result has the full output!
        self.assertNotIn("body lines omitted for compaction", result.tool_history[0][1].output)

    def test_checkpoint_roundtrip_preserves_canonical_uncompacted_history(self):
        """Checkpoint serialization and deserialization retains canonical full output."""
        call1 = ToolCall("c1", "read_file_range", {"path": "src/main.py", "start_line": 1, "end_line": 50})
        call2 = ToolCall("c2", "find_files", {"pattern": "*.py"})
        final_ops = [FileOperation("modify", "src/main.py", "print('done')", "done", None)]

        provider = DummyProvider([call1, call2, final_ops])
        policy = ToolExecutionPolicy(max_tool_steps=8, compaction_window=1)
        engine = ToolEngine(provider, self.registry, policy=policy)

        plan = Plan(objective="Explore", steps=["step 1"])
        context = ProjectContext(root=str(self.root))

        result = engine.run("task", plan, context)

        # Serialize canonical history to dict (as checkpoint does)
        serialized = history_to_dict(result.tool_history)
        deserialized = history_from_dict(serialized)

        self.assertEqual(len(deserialized), 2)
        # Output is complete and uncompacted
        self.assertIn("line_1 = 1", deserialized[0][1].output)
        self.assertIn("line_50 = 50", deserialized[0][1].output)

    def test_orchestrator_end_to_end_telemetry_reporting(self):
        """Orchestrator records compacted_entries and model_context_bytes in RunReport."""
        call1 = ToolCall("c1", "read_file_range", {"path": "src/main.py", "start_line": 1, "end_line": 50})
        call2 = ToolCall("c2", "find_files", {"pattern": "*.py"})
        call3 = ToolCall("c3", "find_files", {"pattern": "*.py"})
        final_ops = [FileOperation("modify", "src/main.py", "print('done')\n", "done", None)]

        provider = DummyProvider([call1, call2, call3, final_ops])
        scheduler = DummyScheduler(provider)
        config = AgentConfig(
            project=self.root,
            max_tool_steps=8,
            tool_history_compaction_window=2,
        )
        storage = JsonFileStorage(self.root / ".agent_data")
        orchestrator = Orchestrator(
            config, storage, scheduler, threading.Lock(), threading.Lock()
        )

        now = datetime.datetime.now(datetime.timezone.utc)
        task = Task(
            task_id="task-compaction-e2e",
            objective="E2E compaction test",
            status=TaskStatus.PENDING,
            created_at=now,
            updated_at=now,
        )

        report = orchestrator.run(task)
        self.assertTrue(report.completed)
        self.assertEqual(len(report.tool_metrics), 1)
        self.assertGreaterEqual(report.tool_metrics[0].compacted_entries, 1)
        self.assertLess(report.tool_metrics[0].model_context_bytes, report.tool_metrics[0].total_output_bytes)
        self.assertEqual(len(report.tool_history), 3)

    def test_resume_reconstructs_fresh_model_facing_context(self):
        """Resumed session reconstructs fresh model-facing context from canonical history."""
        call1 = ToolCall("c1", "read_file_range", {"path": "src/main.py", "start_line": 1, "end_line": 50})
        call2 = ToolCall("c2", "read_file_range", {"path": "src/main.py", "start_line": 1, "end_line": 50})
        res1 = ToolResult("c1", "read_file_range", "\n".join([f"{i}: x = {i}" for i in range(1, 50)]), False, False)
        res2 = ToolResult("c2", "read_file_range", "\n".join([f"{i}: y = {i}" for i in range(1, 50)]), False, False)

        initial_history = [(call1, res1), (call2, res2)]
        call3 = ToolCall("c3", "find_files", {"pattern": "*.py"})
        final_ops = [FileOperation("modify", "src/main.py", "print('resumed')\n", "resumed", None)]

        provider = DummyProvider([call3, final_ops])
        policy = ToolExecutionPolicy(max_tool_steps=8, compaction_window=2)
        engine = ToolEngine(provider, self.registry, policy=policy)

        plan = Plan(objective="Explore", steps=["step 1"])
        context = ProjectContext(root=str(self.root))

        result = engine.run("task", plan, context, initial_history=initial_history)
        self.assertTrue(result.completed)
        self.assertEqual(len(result.tool_history), 3)

        # On the step with final_ops (3 history items), call1 was compacted while call2 and call3 were full fidelity
        last_hist = provider.received_histories[-1]
        self.assertIn("body lines omitted for compaction", last_hist[0][1].output)
        self.assertNotIn("body lines omitted for compaction", last_hist[1][1].output)
        self.assertNotIn("body lines omitted for compaction", last_hist[2][1].output)

    def test_legacy_metrics_and_checkpoints_backward_compatibility(self):
        """Legacy metrics and checkpoint dicts without compaction fields deserialize cleanly."""
        legacy_metrics_dict = {
            "total_calls": 3,
            "unique_calls": 3,
            "repeated_calls": 0,
            "calls_by_tool": {"read_file_range": 3},
            "total_output_bytes": 1200,
            "output_bytes_by_tool": {"read_file_range": 1200},
            "truncated_results": 0,
            "tool_errors": 0,
            "circuit_breaker_events": 0,
            "steps_used": 3,
            "history_entries": 3,
            "termination_reason": "completed",
            "completed": True,
            "elapsed_ms": 150.0,
        }
        metrics = ToolExecutionMetrics.from_dict(legacy_metrics_dict)
        self.assertEqual(metrics.compacted_entries, 0)
        self.assertEqual(metrics.model_context_bytes, 0)

        legacy_policy_dict = {
            "max_tool_steps": 8,
            "max_tool_output_bytes": 4000,
            "total_tool_budget_bytes": 32000,
            "max_consecutive_repeats": 3,
        }
        policy = ToolExecutionPolicy.from_dict(legacy_policy_dict)
        self.assertEqual(policy.compaction_window, 2)
        self.assertEqual(policy.max_context_bytes, 8000)

    def test_run_command_sandbox_compaction_preserves_head_and_tail(self):
        """run_command_sandbox older output retains first 5 and last 5 lines."""
        lines = [f"Output line {i}" for i in range(1, 30)]
        output = "\n".join(lines)
        call1 = ToolCall("c1", "run_command_sandbox", {"command": ["pytest"]})
        res1 = ToolResult("c1", "run_command_sandbox", output, is_error=False, truncated=False)
        call2 = ToolCall("c2", "find_files", {})
        res2 = ToolResult("c2", "find_files", "a.py", False, False)
        call3 = ToolCall("c3", "find_files", {})
        res3 = ToolResult("c3", "find_files", "b.py", False, False)

        compactor = ToolContextCompactor(window=2)
        model_hist, compacted, _ = compactor.compact([(call1, res1), (call2, res2), (call3, res3)])

        compacted_text = model_hist[0][1].output
        self.assertIn("Output line 1", compacted_text)
        self.assertIn("Output line 5", compacted_text)
        self.assertIn("Output line 29", compacted_text)
        self.assertIn("stdout lines omitted", compacted_text)


if __name__ == "__main__":
    unittest.main()
