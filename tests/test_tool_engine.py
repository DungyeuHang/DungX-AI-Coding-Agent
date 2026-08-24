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
    """Deterministic mock provider that returns a pre-programmed sequence of responses."""

    def __init__(self, responses: list[Any]):
        self.responses = list(responses)
        self.calls_received: list[dict[str, Any]] = []

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
        self.calls_received.append({
            "task": task,
            "plan": plan,
            "context": context,
            "tools": tools,
            "history_len": len(tool_history or []),
            "last_result": tool_history[-1][1] if tool_history else None,
        })
        if not self.responses:
            raise RuntimeError("ScriptedMockProvider ran out of configured responses.")
        return self.responses.pop(0)


class ToolEngineTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name).resolve()

        # Create test files
        (self.root / "src").mkdir(parents=True, exist_ok=True)
        (self.root / "src" / "hello.py").write_text("print('hello world')\n", encoding="utf-8")

        self.registry = ToolRegistry(self.root)
        self.context = ProjectContext(root=str(self.root))
        self.plan = Plan(objective="Test plan", files_likely_to_change=["src/hello.py"])

    def tearDown(self):
        self.temp_dir.cleanup()

    # -------------------------------------------------------------------------
    # 1. Basic loop & 2. Multiple tool calls
    # -------------------------------------------------------------------------

    def test_single_tool_call_then_final_patch(self):
        call1 = ToolCall("call-1", "read_file_range", {"path": "src/hello.py", "start_line": 1, "end_line": 2})
        final_ops = [FileOperation(action="modify", path="src/hello.py", content="print('hello universe')\n")]

        provider = ScriptedMockProvider([call1, final_ops])
        engine = ToolEngine(provider, self.registry)

        result = engine.run("Update hello.py", self.plan, self.context)

        self.assertTrue(result.completed)
        self.assertEqual(result.termination_reason, "completed")
        self.assertEqual(result.file_operations, final_ops)
        self.assertEqual(len(result.tool_history), 1)
        self.assertEqual(result.tool_history[0][0].call_id, "call-1")
        self.assertFalse(result.tool_history[0][1].is_error)
        self.assertIn("1: print('hello world')", result.tool_history[0][1].output)
        self.assertEqual(result.steps_used, 1)

    def test_multiple_tool_calls_in_exact_order(self):
        call_a = ToolCall("a", "find_files", {"pattern": "*.py"})
        call_b = ToolCall("b", "read_file_range", {"path": "src/hello.py", "start_line": 1, "end_line": 1})
        call_c = ToolCall("c", "grep_code", {"pattern": "hello"})
        final_ops = [FileOperation(action="write", path="src/new.py", content="# new\n")]

        provider = ScriptedMockProvider([call_a, call_b, call_c, final_ops])
        engine = ToolEngine(provider, self.registry)

        result = engine.run("Multi-tool search", self.plan, self.context)

        self.assertTrue(result.completed)
        self.assertEqual(result.steps_used, 3)
        self.assertEqual(len(result.tool_history), 3)
        self.assertEqual([c.call_id for c, _ in result.tool_history], ["a", "b", "c"])
        self.assertEqual(result.file_operations, final_ops)

    # -------------------------------------------------------------------------
    # 3. Immediate final response
    # -------------------------------------------------------------------------

    def test_immediate_final_response_without_tools(self):
        final_ops = [FileOperation(action="create", path="src/one_shot.py", content="# oneshot\n")]
        provider = ScriptedMockProvider([final_ops])
        engine = ToolEngine(provider, self.registry)

        result = engine.run("One shot task", self.plan, self.context)

        self.assertTrue(result.completed)
        self.assertEqual(result.steps_used, 0)
        self.assertEqual(len(result.tool_history), 0)
        self.assertEqual(result.file_operations, final_ops)

    # -------------------------------------------------------------------------
    # 4. Step bounds & 5. Invalid parameters
    # -------------------------------------------------------------------------

    def test_max_tool_steps_bound(self):
        # Create an endless stream of tool calls
        endless_calls = [
            ToolCall(f"call-{i}", "find_files", {"pattern": f"*{i}.py"}) for i in range(20)
        ]
        provider = ScriptedMockProvider(endless_calls)
        engine = ToolEngine(provider, self.registry, max_tool_steps=4)

        result = engine.run("Endless exploration", self.plan, self.context)

        self.assertFalse(result.completed)
        self.assertEqual(result.termination_reason, "max_steps_exceeded")
        self.assertEqual(result.steps_used, 4)
        self.assertEqual(len(result.tool_history), 4)
        self.assertIsNone(result.file_operations)

    def test_invalid_max_steps_rejected(self):
        provider = ScriptedMockProvider([])
        with self.assertRaises(ValueError):
            ToolEngine(provider, self.registry, max_tool_steps=0)
        with self.assertRaises(ValueError):
            ToolEngine(provider, self.registry, max_tool_steps=-5)
        with self.assertRaises(ValueError):
            ToolEngine(provider, self.registry, total_tool_budget_bytes=0)

    # -------------------------------------------------------------------------
    # 6. Output budget enforcement
    # -------------------------------------------------------------------------

    def test_output_budget_exhaustion(self):
        (self.root / "src" / "big.txt").write_text("x" * 500, encoding="utf-8")
        call1 = ToolCall("c1", "read_file_range", {"path": "src/big.txt", "start_line": 1, "end_line": 10})
        call2 = ToolCall("c2", "read_file_range", {"path": "src/big.txt", "start_line": 1, "end_line": 10})
        provider = ScriptedMockProvider([call1, call2])

        # Budget of 200 bytes will be exhausted after 1 call
        engine = ToolEngine(provider, self.registry, total_tool_budget_bytes=200)

        result = engine.run("Budget test", self.plan, self.context)

        self.assertFalse(result.completed)
        self.assertEqual(result.termination_reason, "budget_exhausted")
        self.assertEqual(result.steps_used, 1)

    # -------------------------------------------------------------------------
    # 7. Repeated call circuit breaker & 8. Non-consecutive repeats
    # -------------------------------------------------------------------------

    def test_repeated_call_circuit_breaker_triggers_on_3_consecutive(self):
        # 3 identical calls
        same_call = ToolCall("c-rep", "read_file_range", {"path": "src/hello.py", "start_line": 1, "end_line": 5})
        final_ops = [FileOperation(action="modify", path="src/hello.py", content="# ok\n")]
        provider = ScriptedMockProvider([same_call, same_call, same_call, final_ops])

        engine = ToolEngine(provider, self.registry)
        result = engine.run("Repeat test", self.plan, self.context)

        self.assertTrue(result.completed)
        # First 2 executions succeed, 3rd execution triggers circuit breaker ToolResult
        self.assertEqual(len(result.tool_history), 3)
        self.assertTrue(result.tool_history[2][1].is_error)
        self.assertIn("Circuit breaker triggered", result.tool_history[2][1].output)

    def test_non_consecutive_repeats_do_not_trigger_circuit_breaker(self):
        call_1 = ToolCall("c1", "read_file_range", {"path": "src/hello.py", "start_line": 1, "end_line": 5})
        call_different = ToolCall("c2", "find_files", {"pattern": "*.py"})
        call_3 = ToolCall("c3", "read_file_range", {"path": "src/hello.py", "start_line": 1, "end_line": 5})
        final_ops = [FileOperation(action="modify", path="src/hello.py", content="# done\n")]

        provider = ScriptedMockProvider([call_1, call_different, call_3, final_ops])
        engine = ToolEngine(provider, self.registry)
        result = engine.run("Non consecutive", self.plan, self.context)

        self.assertTrue(result.completed)
        self.assertEqual(len(result.tool_history), 3)
        for _, tr in result.tool_history:
            self.assertFalse(tr.is_error)

    # -------------------------------------------------------------------------
    # 9. Argument canonicalization
    # -------------------------------------------------------------------------

    def test_argument_canonicalization_ignores_key_order(self):
        dict1 = {"path": "a.py", "start_line": 1, "end_line": 10}
        dict2 = {"end_line": 10, "path": "a.py", "start_line": 1}
        self.assertEqual(canonicalize_arguments(dict1), canonicalize_arguments(dict2))

    # -------------------------------------------------------------------------
    # 10. Tool error handling & 11. Invalid provider response
    # -------------------------------------------------------------------------

    def test_tool_error_handled_and_fed_to_provider(self):
        bad_call = ToolCall("err-1", "read_file_range", {"path": "non_existent.py"})
        final_ops = [FileOperation(action="create", path="src/fixed.py", content="# fix\n")]
        provider = ScriptedMockProvider([bad_call, final_ops])

        engine = ToolEngine(provider, self.registry)
        result = engine.run("Error recovery", self.plan, self.context)

        self.assertTrue(result.completed)
        self.assertEqual(len(result.tool_history), 1)
        self.assertTrue(result.tool_history[0][1].is_error)
        self.assertIn("File not found", result.tool_history[0][1].output)
        self.assertEqual(result.file_operations, final_ops)

    def test_invalid_provider_response_terminates_gracefully(self):
        provider = ScriptedMockProvider(["invalid_string_response"])
        engine = ToolEngine(provider, self.registry)

        result = engine.run("Bad response", self.plan, self.context)

        self.assertFalse(result.completed)
        self.assertEqual(result.termination_reason, "invalid_provider_response")
        self.assertIn("invalid response type", result.error_message.lower())

    # -------------------------------------------------------------------------
    # 12. History & 13. State serialization / Resume
    # -------------------------------------------------------------------------

    def test_history_serialization_round_trip(self):
        call = ToolCall("c1", "find_files", {"pattern": "*.py"})
        res = ToolResult("c1", "find_files", "src/hello.py", is_error=False, truncated=False)
        history = [(call, res)]

        data = history_to_dict(history)
        restored = history_from_dict(data)

        self.assertEqual(len(restored), 1)
        self.assertEqual(restored[0][0], call)
        self.assertEqual(restored[0][1], res)

    def test_resume_from_initial_history_does_not_reexecute(self):
        call_prior = ToolCall("c0", "find_files", {"pattern": "*.py"})
        res_prior = ToolResult("c0", "find_files", "src/hello.py")
        prior_history = [(call_prior, res_prior)]

        call_next = ToolCall("c1", "read_file_range", {"path": "src/hello.py"})
        final_ops = [FileOperation(action="modify", path="src/hello.py", content="# res\n")]
        provider = ScriptedMockProvider([call_next, final_ops])

        engine = ToolEngine(provider, self.registry)
        result = engine.run("Resume run", self.plan, self.context, initial_history=prior_history)

        self.assertTrue(result.completed)
        self.assertEqual(result.steps_used, 2)
        self.assertEqual(len(result.tool_history), 2)
        self.assertEqual(result.tool_history[0][0].call_id, "c0")
        self.assertEqual(result.tool_history[1][0].call_id, "c1")

    def test_tool_engine_result_serialization_round_trip(self):
        call = ToolCall("c1", "grep_code", {"pattern": "hello"})
        res = ToolResult("c1", "grep_code", "src/hello.py:1: print('hello world')")
        ops = [FileOperation(action="modify", path="src/hello.py", content="# update\n")]

        res_obj = ToolEngineResult(
            file_operations=ops,
            tool_history=[(call, res)],
            steps_used=1,
            total_tool_output_bytes=40,
            completed=True,
            termination_reason="completed",
        )

        d = res_obj.to_dict()
        restored = ToolEngineResult.from_dict(d)

        self.assertEqual(restored.completed, res_obj.completed)
        self.assertEqual(restored.steps_used, res_obj.steps_used)
        self.assertEqual(restored.termination_reason, res_obj.termination_reason)
        self.assertEqual(len(restored.file_operations), 1)
        self.assertEqual(restored.file_operations[0].path, "src/hello.py")
        self.assertEqual(len(restored.tool_history), 1)
        self.assertEqual(restored.tool_history[0][0].call_id, "c1")

    # -------------------------------------------------------------------------
    # 14. File mutation invariant
    # -------------------------------------------------------------------------

    def test_tool_engine_never_mutates_filesystem(self):
        before_files = list(self.root.rglob("*"))
        before_contents = {f: f.read_text(encoding="utf-8") for f in before_files if f.is_file()}

        call1 = ToolCall("c1", "find_files", {"pattern": "*"})
        call2 = ToolCall("c2", "read_file_range", {"path": "src/hello.py", "start_line": 1, "end_line": 10})
        final_ops = [FileOperation(action="write", path="src/hello.py", content="SHOULD_NOT_BE_WRITTEN_YET\n")]

        provider = ScriptedMockProvider([call1, call2, final_ops])
        engine = ToolEngine(provider, self.registry)
        result = engine.run("Mutation test", self.plan, self.context)

        self.assertTrue(result.completed)

        after_files = list(self.root.rglob("*"))
        after_contents = {f: f.read_text(encoding="utf-8") for f in after_files if f.is_file()}

        # Verify exact filesystem match
        self.assertEqual(before_files, after_files)
        self.assertEqual(before_contents, after_contents)


if __name__ == "__main__":
    unittest.main()
