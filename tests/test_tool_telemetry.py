from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from local_agent.models import (
    FileOperation,
    Plan,
    ProjectContext,
    ToolCall,
    ToolDefinition,
    ToolExecutionMetrics,
    ToolResult,
)
from local_agent.tool_engine import (
    ToolEngine,
    ToolEngineResult,
    canonicalize_arguments,
    history_from_dict,
    history_to_dict,
)
from local_agent.tools import ToolRegistry


class ScriptedMockProvider:
    def __init__(self, responses: list[Any]):
        self.responses = list(responses)
        self.call_count = 0
        self.history_received: list[list[tuple[ToolCall, ToolResult]]] = []

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
        self.call_count += 1
        self.history_received.append(list(tool_history or []))
        if not self.responses:
            return [FileOperation("modify", "src/main.py", "content", "reason", None)]
        return self.responses.pop(0)


class ToolTelemetryTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        (self.root / "src").mkdir(parents=True, exist_ok=True)
        (self.root / "src" / "app.py").write_text(
            "def hello():\n    print('Hello World')\n    return 42\n",
            encoding="utf-8",
        )
        (self.root / "src" / "utils.py").write_text(
            "def add(a, b):\n    return a + b\n",
            encoding="utf-8",
        )
        self.registry = ToolRegistry(self.root)
        self.plan = Plan(objective="Telemetry Test", steps=["step1"])
        self.context = ProjectContext(root=str(self.root))

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_empty_session_metrics(self):
        """1. Empty session metrics: provider returns final operations immediately."""
        final_ops = [FileOperation("create", "src/new.py", "x = 1\n", "add file", None)]
        provider = ScriptedMockProvider([final_ops])
        engine = ToolEngine(provider, self.registry)

        result = engine.run("Empty session task", self.plan, self.context)

        self.assertTrue(result.completed)
        self.assertEqual(result.termination_reason, "completed")
        self.assertEqual(result.steps_used, 0)
        self.assertEqual(result.total_tool_output_bytes, 0)

        metrics = result.metrics
        self.assertIsInstance(metrics, ToolExecutionMetrics)
        self.assertEqual(metrics.total_calls, 0)
        self.assertEqual(metrics.unique_calls, 0)
        self.assertEqual(metrics.repeated_calls, 0)
        self.assertEqual(metrics.calls_by_tool, {})
        self.assertEqual(metrics.total_output_bytes, 0)
        self.assertEqual(metrics.output_bytes_by_tool, {})
        self.assertEqual(metrics.truncated_results, 0)
        self.assertEqual(metrics.tool_errors, 0)
        self.assertEqual(metrics.circuit_breaker_events, 0)
        self.assertEqual(metrics.steps_used, 0)
        self.assertEqual(metrics.history_entries, 0)
        self.assertEqual(metrics.termination_reason, "completed")
        self.assertTrue(metrics.completed)
        self.assertGreaterEqual(metrics.elapsed_ms, 0.0)

    def test_one_successful_tool_call(self):
        """2. Single successful tool call metrics."""
        call1 = ToolCall("call-1", "read_file_range", {"path": "src/app.py", "start_line": 1, "end_line": 2})
        final_ops = [FileOperation("modify", "src/app.py", "# new\n", "update", None)]
        provider = ScriptedMockProvider([call1, final_ops])
        engine = ToolEngine(provider, self.registry)

        result = engine.run("Single tool task", self.plan, self.context)

        self.assertTrue(result.completed)
        self.assertEqual(result.termination_reason, "completed")
        self.assertEqual(result.steps_used, 1)

        metrics = result.metrics
        self.assertEqual(metrics.total_calls, 1)
        self.assertEqual(metrics.unique_calls, 1)
        self.assertEqual(metrics.repeated_calls, 0)
        self.assertEqual(metrics.calls_by_tool, {"read_file_range": 1})
        self.assertGreater(metrics.total_output_bytes, 0)
        self.assertEqual(metrics.output_bytes_by_tool["read_file_range"], metrics.total_output_bytes)
        self.assertEqual(metrics.truncated_results, 0)
        self.assertEqual(metrics.tool_errors, 0)
        self.assertEqual(metrics.circuit_breaker_events, 0)
        self.assertEqual(metrics.steps_used, 1)
        self.assertEqual(metrics.history_entries, 1)
        self.assertTrue(metrics.completed)

    def test_multiple_different_tools(self):
        """3. Multiple distinct tools metrics."""
        call1 = ToolCall("call-1", "find_files", {"pattern": "*.py"})
        call2 = ToolCall("call-2", "grep_code", {"pattern": "def hello"})
        call3 = ToolCall("call-3", "read_file_range", {"path": "src/app.py", "start_line": 1, "end_line": 3})
        final_ops = [FileOperation("modify", "src/app.py", "# done\n", "refactor", None)]
        provider = ScriptedMockProvider([call1, call2, call3, final_ops])
        engine = ToolEngine(provider, self.registry)

        result = engine.run("Multi tool task", self.plan, self.context)

        self.assertTrue(result.completed)
        self.assertEqual(result.steps_used, 3)

        metrics = result.metrics
        self.assertEqual(metrics.total_calls, 3)
        self.assertEqual(metrics.unique_calls, 3)
        self.assertEqual(metrics.repeated_calls, 0)
        self.assertEqual(
            metrics.calls_by_tool,
            {"find_files": 1, "grep_code": 1, "read_file_range": 1},
        )
        self.assertEqual(len(metrics.output_bytes_by_tool), 3)
        self.assertEqual(
            metrics.total_output_bytes,
            sum(metrics.output_bytes_by_tool.values()),
        )
        self.assertEqual(metrics.tool_errors, 0)
        self.assertEqual(metrics.history_entries, 3)

    def test_repeated_tool_calls_accounting(self):
        """4. Repeated calls tracking."""
        call1 = ToolCall("call-1", "read_file_range", {"path": "src/app.py", "start_line": 1, "end_line": 2})
        call2 = ToolCall("call-2", "find_files", {"pattern": "*.py"})
        call3 = ToolCall("call-3", "read_file_range", {"path": "src/app.py", "start_line": 1, "end_line": 2})
        final_ops = [FileOperation("modify", "src/app.py", "# mod\n", "edit", None)]
        provider = ScriptedMockProvider([call1, call2, call3, final_ops])
        engine = ToolEngine(provider, self.registry)

        result = engine.run("Repeat call task", self.plan, self.context)

        metrics = result.metrics
        self.assertEqual(metrics.total_calls, 3)
        self.assertEqual(metrics.unique_calls, 2)
        self.assertEqual(metrics.repeated_calls, 1)
        self.assertEqual(metrics.calls_by_tool, {"read_file_range": 2, "find_files": 1})
        self.assertEqual(metrics.circuit_breaker_events, 0)

    def test_circuit_breaker_event_accounting(self):
        """5. Consecutive identical calls trigger circuit breaker and update metrics."""
        call = ToolCall("call-1", "read_file_range", {"path": "src/app.py", "start_line": 1, "end_line": 2})
        final_ops = [FileOperation("modify", "src/app.py", "# mod\n", "edit", None)]
        provider = ScriptedMockProvider([call, call, call, final_ops])
        engine = ToolEngine(provider, self.registry)

        result = engine.run("Circuit breaker task", self.plan, self.context)

        metrics = result.metrics
        self.assertEqual(metrics.circuit_breaker_events, 1)
        self.assertGreaterEqual(metrics.tool_errors, 1)
        self.assertEqual(metrics.calls_by_tool["read_file_range"], 3)
        self.assertTrue(result.completed)

    def test_tool_error_accounting(self):
        """6. Tool execution errors correctly tracked."""
        bad_call1 = ToolCall("err-1", "read_file_range", {"path": "non_existent.py"})
        bad_call2 = ToolCall("err-2", "unknown_tool", {})
        final_ops = [FileOperation("modify", "src/app.py", "# fix\n", "fix", None)]
        provider = ScriptedMockProvider([bad_call1, bad_call2, final_ops])
        engine = ToolEngine(provider, self.registry)

        result = engine.run("Error accounting task", self.plan, self.context)

        metrics = result.metrics
        self.assertEqual(metrics.total_calls, 2)
        self.assertEqual(metrics.tool_errors, 2)
        self.assertEqual(metrics.circuit_breaker_events, 0)
        self.assertTrue(result.completed)

    def test_truncation_accounting(self):
        """7. Output truncation tracked in metrics."""
        large_file = self.root / "src" / "large.txt"
        large_file.write_text("A" * 10000, encoding="utf-8")

        call = ToolCall("trunc-1", "read_file_range", {"path": "src/large.txt", "start_line": 1, "end_line": 50})
        final_ops = [FileOperation("modify", "src/app.py", "# mod\n", "edit", None)]
        provider = ScriptedMockProvider([call, final_ops])
        engine = ToolEngine(provider, self.registry, max_tool_output_bytes=500)

        result = engine.run("Truncation task", self.plan, self.context)

        metrics = result.metrics
        self.assertEqual(metrics.truncated_results, 1)
        self.assertTrue(result.tool_history[0][1].truncated)

    def test_output_byte_accounting_exact_utf8(self):
        """8. Exact UTF-8 byte length summation across tool calls."""
        unicode_file = self.root / "src" / "unicode.py"
        unicode_content = "# Chào thế giới 🚀\ndef test(): pass\n"
        unicode_file.write_text(unicode_content, encoding="utf-8")

        call1 = ToolCall("u-1", "read_file_range", {"path": "src/unicode.py", "start_line": 1, "end_line": 2})
        call2 = ToolCall("u-2", "find_files", {"pattern": "*.py"})
        final_ops = [FileOperation("modify", "src/app.py", "# end\n", "done", None)]
        provider = ScriptedMockProvider([call1, call2, final_ops])
        engine = ToolEngine(provider, self.registry)

        result = engine.run("UTF-8 byte task", self.plan, self.context)

        expected_bytes = sum(len(res.output.encode("utf-8")) for _, res in result.tool_history)
        self.assertEqual(result.total_tool_output_bytes, expected_bytes)
        self.assertEqual(result.metrics.total_output_bytes, expected_bytes)

    def test_max_steps_exceeded_termination_metrics(self):
        """9 & 10. Correct steps_used and termination_reason on step limit."""
        call = ToolCall("step-1", "find_files", {"pattern": "*"})
        provider = ScriptedMockProvider([call] * 10)
        engine = ToolEngine(provider, self.registry, max_tool_steps=3)

        result = engine.run("Step limit task", self.plan, self.context)

        self.assertFalse(result.completed)
        self.assertEqual(result.termination_reason, "max_steps_exceeded")
        self.assertEqual(result.steps_used, 3)
        self.assertEqual(result.metrics.steps_used, 3)
        self.assertEqual(result.metrics.termination_reason, "max_steps_exceeded")
        self.assertFalse(result.metrics.completed)

    def test_budget_exhausted_termination_metrics(self):
        """10. Correct termination_reason on total budget exhaustion."""
        call1 = ToolCall("b-1", "read_file_range", {"path": "src/app.py", "start_line": 1, "end_line": 3})
        call2 = ToolCall("b-2", "read_file_range", {"path": "src/utils.py", "start_line": 1, "end_line": 3})
        provider = ScriptedMockProvider([call1, call2])
        engine = ToolEngine(provider, self.registry, total_tool_budget_bytes=50)

        result = engine.run("Budget limit task", self.plan, self.context)

        self.assertFalse(result.completed)
        self.assertEqual(result.termination_reason, "budget_exhausted")
        self.assertEqual(result.metrics.termination_reason, "budget_exhausted")
        self.assertFalse(result.metrics.completed)

    def test_invalid_provider_response_metrics(self):
        """10. Correct termination_reason on invalid provider response."""
        provider = ScriptedMockProvider(["invalid string response"])
        engine = ToolEngine(provider, self.registry)

        result = engine.run("Invalid response task", self.plan, self.context)

        self.assertFalse(result.completed)
        self.assertEqual(result.termination_reason, "invalid_provider_response")
        self.assertEqual(result.metrics.termination_reason, "invalid_provider_response")
        self.assertFalse(result.metrics.completed)

    def test_metrics_survive_result_dict_serialization(self):
        """13 & 14. Metrics roundtrip serialization via to_dict / from_dict."""
        call1 = ToolCall("call-1", "find_files", {"pattern": "*.py"})
        final_ops = [FileOperation("modify", "src/app.py", "# patch\n", "patch", None)]
        provider = ScriptedMockProvider([call1, final_ops])
        engine = ToolEngine(provider, self.registry)

        result = engine.run("Serialization task", self.plan, self.context)
        result_dict = result.to_dict()

        self.assertIn("metrics", result_dict)
        self.assertEqual(result_dict["metrics"]["total_calls"], 1)
        self.assertEqual(result_dict["metrics"]["termination_reason"], "completed")

        restored_result = ToolEngineResult.from_dict(result_dict)
        self.assertIsInstance(restored_result.metrics, ToolExecutionMetrics)
        self.assertEqual(restored_result.metrics.total_calls, 1)
        self.assertEqual(restored_result.metrics.unique_calls, 1)
        self.assertEqual(restored_result.metrics.calls_by_tool, {"find_files": 1})
        self.assertEqual(restored_result.metrics.termination_reason, "completed")
        self.assertTrue(restored_result.metrics.completed)

    def test_metrics_restoration_from_initial_history(self):
        """14. Initial history metrics restoration across pause/resume."""
        call1 = ToolCall("hist-1", "read_file_range", {"path": "src/app.py", "start_line": 1, "end_line": 2})
        result1 = self.registry.execute(call1)
        initial_history = [(call1, result1)]

        call2 = ToolCall("hist-2", "find_files", {"pattern": "*.py"})
        final_ops = [FileOperation("modify", "src/app.py", "# resume\n", "resume", None)]
        provider = ScriptedMockProvider([call2, final_ops])
        engine = ToolEngine(provider, self.registry)

        result = engine.run("Resumed session task", self.plan, self.context, initial_history=initial_history)

        metrics = result.metrics
        self.assertEqual(metrics.total_calls, 2)
        self.assertEqual(metrics.unique_calls, 2)
        self.assertEqual(metrics.repeated_calls, 0)
        self.assertEqual(metrics.calls_by_tool, {"read_file_range": 1, "find_files": 1})
        self.assertEqual(metrics.steps_used, 2)
        self.assertEqual(metrics.history_entries, 2)
        self.assertTrue(result.completed)

    def test_baseline_representative_exploration_session(self):
        """15. Deterministic baseline representative tool exploration session."""
        call1 = ToolCall("step1", "find_files", {"pattern": "*.py"})
        call2 = ToolCall("step2", "read_file_range", {"path": "src/app.py", "start_line": 1, "end_line": 10})
        call3 = ToolCall("step3", "grep_code", {"pattern": "def add"})
        final_ops = [
            FileOperation(
                action="modify",
                path="src/app.py",
                content="def hello():\n    print('Hello World')\n    return 100\n",
                reason="Update return value",
                patch="--- src/app.py\n+++ src/app.py\n@@ -3,1 +3,1 @@\n-    return 42\n+    return 100\n",
            )
        ]

        provider = ScriptedMockProvider([call1, call2, call3, final_ops])
        engine = ToolEngine(provider, self.registry)

        result = engine.run("Baseline exploration task", self.plan, self.context)

        self.assertTrue(result.completed)
        self.assertEqual(result.termination_reason, "completed")
        self.assertEqual(len(result.file_operations), 1)

        m = result.metrics
        self.assertEqual(m.total_calls, 3)
        self.assertEqual(m.unique_calls, 3)
        self.assertEqual(m.repeated_calls, 0)
        self.assertEqual(m.calls_by_tool, {"find_files": 1, "read_file_range": 1, "grep_code": 1})
        self.assertGreater(m.total_output_bytes, 0)
        self.assertEqual(m.truncated_results, 0)
        self.assertEqual(m.tool_errors, 0)
        self.assertEqual(m.circuit_breaker_events, 0)
        self.assertEqual(m.steps_used, 3)
        self.assertEqual(m.history_entries, 3)
        self.assertTrue(m.completed)
        self.assertGreater(m.elapsed_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
