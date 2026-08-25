from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from local_agent.config import AgentConfig
from local_agent.models import (
    FileOperation,
    Plan,
    PolicyAction,
    PolicyDecision,
    ProjectContext,
    ToolCall,
    ToolDefinition,
    ToolExecutionPolicy,
    ToolResult,
)
from local_agent.tool_engine import ToolEngine
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


class ToolPolicyUnitTests(unittest.TestCase):
    def test_policy_decision_model(self):
        decision = PolicyDecision(action=PolicyAction.ALLOW)
        self.assertEqual(decision.action, PolicyAction.ALLOW)
        self.assertIsNone(decision.reason)

        d_dict = decision.to_dict()
        self.assertEqual(d_dict["action"], "allow")
        restored = PolicyDecision.from_dict(d_dict)
        self.assertEqual(restored.action, PolicyAction.ALLOW)

        reject_dec = PolicyDecision(action=PolicyAction.REJECT, reason="test_reason", message="Test message")
        self.assertEqual(reject_dec.action, PolicyAction.REJECT)
        self.assertEqual(reject_dec.reason, "test_reason")
        self.assertEqual(reject_dec.message, "Test message")

    def test_policy_defaults(self):
        policy = ToolExecutionPolicy()
        self.assertEqual(policy.max_tool_steps, 8)
        self.assertEqual(policy.max_tool_output_bytes, 4000)
        self.assertEqual(policy.total_tool_budget_bytes, 32000)
        self.assertEqual(policy.max_consecutive_repeats, 3)
        self.assertEqual(policy.per_tool_limits, {})
        self.assertEqual(policy.disallowed_tools, set())

    def test_policy_validation_invalid_numeric_limits(self):
        with self.assertRaises(ValueError):
            ToolExecutionPolicy(max_tool_steps=0)
        with self.assertRaises(ValueError):
            ToolExecutionPolicy(max_tool_steps=-1)
        with self.assertRaises(ValueError):
            ToolExecutionPolicy(max_tool_output_bytes=0)
        with self.assertRaises(ValueError):
            ToolExecutionPolicy(total_tool_budget_bytes=-100)
        with self.assertRaises(ValueError):
            ToolExecutionPolicy(max_consecutive_repeats=0)

    def test_policy_validation_invalid_tool_limits_and_disallowed(self):
        with self.assertRaises(ValueError):
            ToolExecutionPolicy(per_tool_limits={"tool": -1})
        with self.assertRaises(ValueError):
            ToolExecutionPolicy(per_tool_limits={"": 5})
        with self.assertRaises(ValueError):
            ToolExecutionPolicy(disallowed_tools={""})

    def test_policy_evaluate_call_rules(self):
        policy = ToolExecutionPolicy(
            max_tool_steps=5,
            total_tool_budget_bytes=1000,
            max_consecutive_repeats=2,
            per_tool_limits={"find_files": 2},
            disallowed_tools={"run_command_sandbox"},
        )

        call1 = ToolCall("c1", "grep_code", {"pattern": "abc"})
        # 1. Normal allowed call
        dec = policy.evaluate_call(call1, steps_used=0, total_output_bytes=0, calls_by_tool={}, consecutive_repeat_count=1)
        self.assertEqual(dec.action, PolicyAction.ALLOW)

        # 2. Disallowed tool call
        disallowed_call = ToolCall("c2", "run_command_sandbox", {"command": ["ls"]})
        dec = policy.evaluate_call(disallowed_call, steps_used=0, total_output_bytes=0, calls_by_tool={}, consecutive_repeat_count=1)
        self.assertEqual(dec.action, PolicyAction.REJECT)
        self.assertEqual(dec.reason, "disallowed_tool")

        # 3. Per-tool limit exceeded (3rd call when limit is 2)
        find_call = ToolCall("c3", "find_files", {"pattern": "*.py"})
        dec = policy.evaluate_call(find_call, steps_used=2, total_output_bytes=100, calls_by_tool={"find_files": 3}, consecutive_repeat_count=1)
        self.assertEqual(dec.action, PolicyAction.REJECT)
        self.assertEqual(dec.reason, "tool_limit_exceeded")

        # 4. Consecutive repeats circuit breaker
        dec = policy.evaluate_call(call1, steps_used=1, total_output_bytes=50, calls_by_tool={"grep_code": 1}, consecutive_repeat_count=2)
        self.assertEqual(dec.action, PolicyAction.CIRCUIT_BREAKER)
        self.assertEqual(dec.reason, "consecutive_repeats_exceeded")

        # 5. Steps exceeded
        dec = policy.evaluate_call(call1, steps_used=5, total_output_bytes=100, calls_by_tool={}, consecutive_repeat_count=1)
        self.assertEqual(dec.action, PolicyAction.TERMINATE)
        self.assertEqual(dec.reason, "max_steps_exceeded")

        # 6. Total budget exceeded
        dec = policy.evaluate_call(call1, steps_used=2, total_output_bytes=1000, calls_by_tool={}, consecutive_repeat_count=1)
        self.assertEqual(dec.action, PolicyAction.TERMINATE)
        self.assertEqual(dec.reason, "budget_exhausted")

    def test_policy_serialization_roundtrip(self):
        policy = ToolExecutionPolicy(
            max_tool_steps=12,
            max_tool_output_bytes=5000,
            total_tool_budget_bytes=64000,
            max_consecutive_repeats=4,
            per_tool_limits={"read_file_range": 20, "grep_code": 5},
            disallowed_tools={"run_command_sandbox"},
        )
        p_dict = policy.to_dict()
        self.assertIsInstance(p_dict["disallowed_tools"], list)
        self.assertEqual(p_dict["max_tool_steps"], 12)

        restored = ToolExecutionPolicy.from_dict(p_dict)
        self.assertEqual(restored.max_tool_steps, 12)
        self.assertEqual(restored.max_tool_output_bytes, 5000)
        self.assertEqual(restored.total_tool_budget_bytes, 64000)
        self.assertEqual(restored.max_consecutive_repeats, 4)
        self.assertEqual(restored.per_tool_limits, {"read_file_range": 20, "grep_code": 5})
        self.assertEqual(restored.disallowed_tools, {"run_command_sandbox"})


class AgentConfigToolPolicyTests(unittest.TestCase):
    def test_agent_config_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AgentConfig(project=Path(tmpdir))
            self.assertEqual(config.max_tool_steps, 8)
            self.assertEqual(config.max_tool_output_bytes, 4000)
            self.assertEqual(config.total_tool_budget_bytes, 32000)
            self.assertEqual(config.max_consecutive_repeats, 3)
            self.assertEqual(config.per_tool_limits, {})
            self.assertEqual(config.disallowed_tools, [])

            policy = config.tool_policy
            self.assertIsInstance(policy, ToolExecutionPolicy)
            self.assertEqual(policy.max_tool_steps, 8)
            self.assertEqual(policy.max_tool_output_bytes, 4000)
            self.assertEqual(policy.total_tool_budget_bytes, 32000)
            self.assertEqual(policy.max_consecutive_repeats, 3)

    def test_agent_config_from_environment_parsing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_vars = {
                "AGENT_MAX_TOOL_STEPS": "15",
                "AGENT_MAX_TOOL_OUTPUT_BYTES": "6000",
                "AGENT_TOTAL_TOOL_BUDGET_BYTES": "50000",
                "AGENT_MAX_CONSECUTIVE_REPEATS": "4",
                "AGENT_PER_TOOL_LIMITS": json.dumps({"grep_code": 10, "run_command_sandbox": 2}),
                "AGENT_DISALLOWED_TOOLS": "run_command_sandbox, search_symbols",
            }
            with mock.patch.dict(os.environ, env_vars):
                config = AgentConfig.from_environment(tmpdir)
                self.assertEqual(config.max_tool_steps, 15)
                self.assertEqual(config.max_tool_output_bytes, 6000)
                self.assertEqual(config.total_tool_budget_bytes, 50000)
                self.assertEqual(config.max_consecutive_repeats, 4)
                self.assertEqual(config.per_tool_limits, {"grep_code": 10, "run_command_sandbox": 2})
                self.assertEqual(set(config.disallowed_tools), {"run_command_sandbox", "search_symbols"})

                policy = config.tool_policy
                self.assertEqual(policy.max_tool_steps, 15)
                self.assertEqual(policy.per_tool_limits, {"grep_code": 10, "run_command_sandbox": 2})
                self.assertEqual(policy.disallowed_tools, {"run_command_sandbox", "search_symbols"})


class ToolEnginePolicyIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        (self.root / "src").mkdir(parents=True, exist_ok=True)
        (self.root / "src" / "app.py").write_text("def hello(): pass\n", encoding="utf-8")
        self.registry = ToolRegistry(self.root)
        self.plan = Plan(objective="Policy Integration Test", steps=["step1"])
        self.context = ProjectContext(root=str(self.root))

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_disallowed_tool_rejected_in_engine_loop(self):
        """Disallowed tool is rejected with error ToolResult, loop continues to next turn."""
        policy = ToolExecutionPolicy(disallowed_tools={"find_files"})
        disallowed_call = ToolCall("c1", "find_files", {"pattern": "*.py"})
        allowed_call = ToolCall("c2", "read_file_range", {"path": "src/app.py", "start_line": 1, "end_line": 1})
        final_ops = [FileOperation("modify", "src/app.py", "# new\n", "update", None)]

        provider = ScriptedMockProvider([disallowed_call, allowed_call, final_ops])
        engine = ToolEngine(provider, self.registry, policy=policy)

        result = engine.run("Disallowed test", self.plan, self.context)

        self.assertTrue(result.completed)
        self.assertEqual(len(result.tool_history), 2)
        # First call was rejected
        first_call, first_result = result.tool_history[0]
        self.assertEqual(first_call.tool_name, "find_files")
        self.assertTrue(first_result.is_error)
        self.assertIn("disallowed", first_result.output.lower())

        # Second call was executed normally
        second_call, second_result = result.tool_history[1]
        self.assertEqual(second_call.tool_name, "read_file_range")
        self.assertFalse(second_result.is_error)

        self.assertEqual(result.metrics.tool_errors, 1)
        self.assertEqual(result.metrics.total_calls, 2)

    def test_per_tool_limit_rejected_in_engine_loop(self):
        """Tool call exceeding per_tool_limits is rejected with error ToolResult."""
        policy = ToolExecutionPolicy(per_tool_limits={"read_file_range": 1})
        call1 = ToolCall("c1", "read_file_range", {"path": "src/app.py", "start_line": 1, "end_line": 1})
        call2 = ToolCall("c2", "read_file_range", {"path": "src/app.py", "start_line": 1, "end_line": 1}) # exceeds limit
        final_ops = [FileOperation("modify", "src/app.py", "# patch\n", "patch", None)]

        provider = ScriptedMockProvider([call1, call2, final_ops])
        engine = ToolEngine(provider, self.registry, policy=policy)

        result = engine.run("Per tool limit test", self.plan, self.context)

        self.assertTrue(result.completed)
        self.assertEqual(len(result.tool_history), 2)
        self.assertFalse(result.tool_history[0][1].is_error)
        self.assertTrue(result.tool_history[1][1].is_error)
        self.assertIn("limit", result.tool_history[1][1].output.lower())

    def test_backward_compatibility_without_policy(self):
        """ToolEngine instantiated without policy behaves identically to default policy."""
        call1 = ToolCall("c1", "read_file_range", {"path": "src/app.py", "start_line": 1, "end_line": 1})
        final_ops = [FileOperation("modify", "src/app.py", "# patch\n", "patch", None)]

        provider = ScriptedMockProvider([call1, final_ops])
        engine = ToolEngine(provider, self.registry) # no policy passed

        result = engine.run("Default compatibility test", self.plan, self.context)

        self.assertTrue(result.completed)
        self.assertEqual(result.steps_used, 1)
        self.assertEqual(result.metrics.total_calls, 1)
        self.assertEqual(result.termination_reason, "completed")


if __name__ == "__main__":
    unittest.main()
