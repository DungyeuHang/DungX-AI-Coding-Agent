from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import urllib.error

from local_agent.config import AgentConfig
from local_agent.models import (
    CommandSpec,
    ExecutionResult,
    FailureAnalysis,
    FileOperation,
    Plan,
    PreparedChange,
    ProjectContext,
    ProviderCapability,
    ProviderError,
    ReviewResult,
    RunReport,
    Task,
    TaskStatus,
    ToolCall,
    ToolDefinition,
    ToolExecutionMetrics,
    ToolExecutionPolicy,
    ToolResult,
    ValidationPlan,
)
from local_agent.orchestrator import Orchestrator
from local_agent.providers import (
    AIProvider,
    AnthropicProvider,
    AntigravityProvider,
    DeepSeekProvider,
    GeminiProvider,
    MockProvider,
    OpenAIProvider,
)
from local_agent.storage import JsonFileStorage
from local_agent.tools import ToolRegistry
from local_agent.validation import (
    ValidationIntelligence,
    VerificationIntelligence,
    VerificationResult,
)


class ScriptedVerificationProvider(AIProvider):
    """Provider double that yields scripted ToolCalls then a final verification dict."""

    def __init__(self, responses: list[ToolCall | dict[str, object] | VerificationResult]):
        self.responses = list(responses)
        self.tool_calls_received: list[list[tuple[ToolCall, ToolResult]]] = []
        self.verify_called = False

    @property
    def capabilities(self) -> set[ProviderCapability]:
        return {
            ProviderCapability.PLANNING,
            ProviderCapability.IMPLEMENTATION,
            ProviderCapability.REPAIR,
            ProviderCapability.REVIEW,
            ProviderCapability.TOOL_USE,
        }

    def verify_changes_with_tools(
        self,
        task: str,
        plan: Plan,
        context: ProjectContext,
        diff: str,
        changed_files: list[str],
        tools: list[ToolDefinition],
        tool_history: list[tuple[ToolCall, ToolResult]] | None = None,
    ) -> ToolCall | dict[str, object]:
        self.verify_called = True
        self.tool_calls_received.append(list(tool_history or []))
        if not self.responses:
            return {"verified": True, "notes": "No scripted responses left"}
        return self.responses.pop(0)


class ValidationToolsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_path = Path(self.temp_dir.name)
        (self.project_path / "src").mkdir(parents=True, exist_ok=True)
        (self.project_path / "src" / "calculator.py").write_text(
            "def add(a, b):\n    return a + b\n\ndef multiply(a, b):\n    return a * b\n",
            encoding="utf-8",
        )
        (self.project_path / "tests").mkdir(parents=True, exist_ok=True)
        (self.project_path / "tests" / "test_calculator.py").write_text(
            "from src.calculator import add, multiply\n\ndef test_add():\n    assert add(1, 2) == 3\n",
            encoding="utf-8",
        )

        self.context = ProjectContext(root=str(self.project_path))
        self.plan = Plan(objective="Update calculator functions", files_likely_to_change=["src/calculator.py"])
        self.registry = ToolRegistry(self.project_path)
        self.validation_intel = ValidationIntelligence(self.project_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    # -------------------------------------------------------------------------
    # 1. Targeted test discovery from changed files
    # -------------------------------------------------------------------------

    def test_discover_targeted_commands_for_python_source(self):
        """ValidationIntelligence detects corresponding test file for modified source."""
        targeted = self.validation_intel.discover_targeted_commands(["src/calculator.py"])
        self.assertEqual(len(targeted), 1)
        self.assertEqual(targeted[0].name, "targeted_pytest_test_calculator")
        self.assertEqual(targeted[0].command, ("pytest", "tests/test_calculator.py"))

    def test_discover_targeted_commands_for_test_file_itself(self):
        """ValidationIntelligence identifies a modified test file directly."""
        targeted = self.validation_intel.discover_targeted_commands(["tests/test_calculator.py"])
        self.assertEqual(len(targeted), 1)
        self.assertEqual(targeted[0].command, ("pytest", "tests/test_calculator.py"))

    def test_discover_targeted_commands_fallback_when_no_tests_exist(self):
        """ValidationIntelligence returns empty list when no matching test exists."""
        (self.project_path / "src" / "orphan.py").write_text("x = 1\n", encoding="utf-8")
        targeted = self.validation_intel.discover_targeted_commands(["src/orphan.py"])
        self.assertEqual(targeted, [])

    # -------------------------------------------------------------------------
    # 2. Targeted test discovery for JavaScript/TypeScript
    # -------------------------------------------------------------------------

    def test_discover_targeted_commands_for_js_ts(self):
        """ValidationIntelligence detects npm test command for modified TS test file."""
        (self.project_path / "src" / "button.tsx").write_text("export const Button = () => null;\n", encoding="utf-8")
        (self.project_path / "tests" / "button.test.ts").write_text("test('renders', () => {});\n", encoding="utf-8")

        targeted = self.validation_intel.discover_targeted_commands(["src/button.tsx"])
        self.assertEqual(len(targeted), 1)
        self.assertEqual(targeted[0].name, "targeted_test_button")
        self.assertEqual(targeted[0].command, ("npm", "test", "--", "tests/button.test.ts"))

    # -------------------------------------------------------------------------
    # 3. Tool-assisted verification with VerificationIntelligence
    # -------------------------------------------------------------------------

    def test_tool_assisted_verification_multi_turn(self):
        """VerificationIntelligence coordinates tool exploration prior to validation."""
        provider = ScriptedVerificationProvider([
            ToolCall("call_1", "grep_code", {"query": "def test_add"}),
            ToolCall("call_2", "read_file_range", {"path": "tests/test_calculator.py", "start_line": 1, "end_line": 4}),
            {"verified": True, "notes": "Verified assertion passes"},
        ])
        verifier = VerificationIntelligence(provider, registry=self.registry)
        report = RunReport(project=self.context, task_id="task-v1")

        result = verifier.verify("Update calculator", self.plan, self.context, "", ["src/calculator.py"], report=report)

        self.assertTrue(result.verified)
        self.assertEqual(result.notes, "Verified assertion passes")
        self.assertEqual(len(report.tool_history), 2)
        self.assertEqual(len(report.tool_metrics), 1)
        self.assertEqual(report.tool_metrics[0].total_calls, 2)

    # -------------------------------------------------------------------------
    # 4. Capability gating & fallback
    # -------------------------------------------------------------------------

    def test_verification_fallback_when_tool_use_missing(self):
        """VerificationIntelligence falls back to standard verification when TOOL_USE is absent."""
        mock_p = MockProvider()
        verifier = VerificationIntelligence(mock_p, registry=self.registry)
        report = RunReport(project=self.context, task_id="task-v2")

        result = verifier.verify("Update calculator", self.plan, self.context, "", ["src/calculator.py"], report=report)

        self.assertTrue(result.verified)
        self.assertEqual(len(report.tool_metrics), 0)

    # -------------------------------------------------------------------------
    # 5. Sandbox command execution during verification
    # -------------------------------------------------------------------------

    def test_run_command_sandbox_in_verification(self):
        """Verification agent can execute read-only sandbox command."""
        provider = ScriptedVerificationProvider([
            ToolCall("call_cmd", "run_command_sandbox", {"command": ["python", "-c", "print('verify-ok')"]}),
            {"verified": True, "notes": "Command executed successfully"},
        ])
        verifier = VerificationIntelligence(provider, registry=self.registry)
        report = RunReport(project=self.context, task_id="task-v3")

        result = verifier.verify("Verify", self.plan, self.context, "", ["src/calculator.py"], report=report)

        self.assertTrue(result.verified)
        self.assertEqual(len(report.tool_history), 1)
        self.assertIn("verify-ok", report.tool_history[0][1].output)

    # -------------------------------------------------------------------------
    # 6. Policy step limits in verification
    # -------------------------------------------------------------------------

    def test_verification_policy_step_limits(self):
        """Verification terminates cleanly when policy step limit is reached."""
        policy = ToolExecutionPolicy(max_tool_steps=2)
        provider = ScriptedVerificationProvider([
            ToolCall("call_1", "read_file_range", {"path": "src/calculator.py"}),
            ToolCall("call_2", "read_file_range", {"path": "tests/test_calculator.py"}),
            ToolCall("call_3", "read_file_range", {"path": "src/calculator.py"}),
        ])
        verifier = VerificationIntelligence(provider, registry=self.registry, policy=policy)
        report = RunReport(project=self.context, task_id="task-v4")

        result = verifier.verify("Verify", self.plan, self.context, "", ["src/calculator.py"], report=report)

        self.assertTrue(result.verified)
        self.assertEqual(len(report.tool_metrics), 1)
        self.assertEqual(report.tool_metrics[0].termination_reason, "max_steps_exceeded")

    # -------------------------------------------------------------------------
    # 7. Real Providers: OpenAI verify_changes_with_tools
    # -------------------------------------------------------------------------

    @patch("urllib.request.urlopen")
    def test_openai_verify_changes_with_tools(self, mock_urlopen):
        """OpenAIProvider handles tool calling and final response in verify_changes_with_tools."""
        config = AgentConfig(
            project=self.project_path,
            provider="openai",
            model="gpt-4o",
            api_key="test-key",
            api_base_url="https://api.openai.com/v1",
        )
        provider = OpenAIProvider(config)

        resp1 = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_v_oai",
                                "type": "function",
                                "function": {
                                    "name": "grep_code",
                                    "arguments": json.dumps({"query": "assert"}),
                                },
                            }
                        ],
                    }
                }
            ]
        }
        resp2 = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({
                            "verified": True,
                            "notes": "OpenAI verified assertions",
                        }),
                    }
                }
            ]
        }

        mock_responses = [
            json.dumps(resp1).encode("utf-8"),
            json.dumps(resp2).encode("utf-8"),
        ]

        def fake_urlopen(req, *args, **kwargs):
            resp = MagicMock()
            resp.read.return_value = mock_responses.pop(0)
            resp.__enter__.return_value = resp
            return resp

        mock_urlopen.side_effect = fake_urlopen

        verifier = VerificationIntelligence(provider, registry=self.registry)
        report = RunReport(project=self.context, task_id="task-oai-v")

        result = verifier.verify("Verify", self.plan, self.context, "", ["src/calculator.py"], report=report)

        self.assertTrue(result.verified)
        self.assertEqual(result.notes, "OpenAI verified assertions")
        self.assertEqual(len(report.tool_history), 1)

    # -------------------------------------------------------------------------
    # 8. Real Providers: Anthropic verify_changes_with_tools
    # -------------------------------------------------------------------------

    @patch("urllib.request.urlopen")
    def test_anthropic_verify_changes_with_tools(self, mock_urlopen):
        """AnthropicProvider handles tool calling and final response in verify_changes_with_tools."""
        config = AgentConfig(
            project=self.project_path,
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            api_key="test-key-anthropic",
        )
        provider = AnthropicProvider(config)

        resp1 = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_anthropic_v",
                    "name": "read_file_range",
                    "input": {"path": "src/calculator.py"},
                }
            ],
            "stop_reason": "tool_use",
        }
        resp2 = {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({
                        "verified": True,
                        "notes": "Anthropic verified calculator",
                    }),
                }
            ],
            "stop_reason": "end_turn",
        }

        mock_responses = [
            json.dumps(resp1).encode("utf-8"),
            json.dumps(resp2).encode("utf-8"),
        ]

        def fake_urlopen(req, *args, **kwargs):
            resp = MagicMock()
            resp.read.return_value = mock_responses.pop(0)
            resp.__enter__.return_value = resp
            return resp

        mock_urlopen.side_effect = fake_urlopen

        verifier = VerificationIntelligence(provider, registry=self.registry)
        report = RunReport(project=self.context, task_id="task-anthropic-v")

        result = verifier.verify("Verify", self.plan, self.context, "", ["src/calculator.py"], report=report)

        self.assertTrue(result.verified)
        self.assertEqual(result.notes, "Anthropic verified calculator")
        self.assertEqual(len(report.tool_history), 1)

    # -------------------------------------------------------------------------
    # 9. Real Providers: Gemini verify_changes_with_tools
    # -------------------------------------------------------------------------

    @patch("urllib.request.urlopen")
    def test_gemini_verify_changes_with_tools(self, mock_urlopen):
        """GeminiProvider handles functionCall and final response in verify_changes_with_tools."""
        config = AgentConfig(
            project=self.project_path,
            provider="gemini",
            model="gemini-2.5-flash",
            api_key="test-key-gemini",
            gemini_base_url="https://generativelanguage.googleapis.com/v1beta",
        )
        provider = GeminiProvider(config)

        model_discovery = {
            "models": [{"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]}]
        }
        resp1 = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "find_files",
                                    "args": {"pattern": "*.py"},
                                }
                            }
                        ]
                    }
                }
            ]
        }
        resp2 = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps({
                                    "verified": True,
                                    "notes": "Gemini verified files",
                                })
                            }
                        ]
                    }
                }
            ]
        }

        mock_responses = [
            json.dumps(model_discovery).encode("utf-8"),
            json.dumps(resp1).encode("utf-8"),
            json.dumps(resp2).encode("utf-8"),
        ]

        def fake_urlopen(req, *args, **kwargs):
            resp = MagicMock()
            resp.read.return_value = mock_responses.pop(0)
            resp.__enter__.return_value = resp
            return resp

        mock_urlopen.side_effect = fake_urlopen

        verifier = VerificationIntelligence(provider, registry=self.registry)
        report = RunReport(project=self.context, task_id="task-gemini-v")

        result = verifier.verify("Verify", self.plan, self.context, "", ["src/calculator.py"], report=report)

        self.assertTrue(result.verified)
        self.assertEqual(result.notes, "Gemini verified files")
        self.assertEqual(len(report.tool_history), 1)

    # -------------------------------------------------------------------------
    # 10. DeepSeek and Antigravity inherit verify_changes_with_tools
    # -------------------------------------------------------------------------

    def test_deepseek_and_antigravity_verify_support(self):
        """DeepSeekProvider and AntigravityProvider expose verify_changes_with_tools."""
        ds_config = AgentConfig(project=self.project_path, provider="deepseek", api_key="k")
        ag_config = AgentConfig(project=self.project_path, provider="antigravity", api_key="k")

        ds_p = DeepSeekProvider(ds_config)
        ag_p = AntigravityProvider(ag_config)

        self.assertTrue(hasattr(ds_p, "verify_changes_with_tools"))
        self.assertTrue(hasattr(ag_p, "verify_changes_with_tools"))

    # -------------------------------------------------------------------------
    # 11. Orchestrator: Targeted validation followed by full validation
    # -------------------------------------------------------------------------

    @patch("local_agent.orchestrator.CodingAgent")
    def test_orchestrator_targeted_then_full_validation(self, mock_coding_agent_cls):
        """Orchestrator runs targeted validation first, then full validation when targeted passes."""
        config = AgentConfig(
            project=self.project_path,
            provider="mock",
            max_iterations=1,
        )
        storage = JsonFileStorage(self.project_path / ".agent")
        repo_lock = MagicMock()
        repo_lock.__enter__.return_value = repo_lock
        repo_lock.__exit__.return_value = None
        memory_lock = MagicMock()
        memory_lock.__enter__.return_value = memory_lock
        memory_lock.__exit__.return_value = None

        mock_ca_instance = MagicMock()
        mock_ca_instance.prepare.return_value = [
            PreparedChange("modify", "src/calculator.py", "orig", "updated", "--- a\n+++ b\n")
        ]
        mock_ca_instance.diff.return_value = "--- a\n+++ b\n"
        mock_ca_instance.apply_prepared.return_value = ["src/calculator.py"]
        mock_coding_agent_cls.return_value = mock_ca_instance

        orchestrator = Orchestrator(config, storage, scheduler=None, repo_lock=repo_lock, memory_lock=memory_lock)

        executed_plans: list[list[str]] = []
        def fake_validate(v_plan):
            cmds = [cmd.name for cmd in v_plan.primary_commands]
            executed_plans.append(cmds)
            return [ExecutionResult("test", 0, "PASS", "")]

        orchestrator._validate = fake_validate

        report = orchestrator.run("Update calculator")

        self.assertIsNotNone(report)
        self.assertGreaterEqual(len(executed_plans), 2)
        self.assertTrue(any("targeted_pytest_test_calculator" in plan for plan in executed_plans))

    # -------------------------------------------------------------------------
    # 12. Orchestrator: Targeted validation failure halts and routes to repair
    # -------------------------------------------------------------------------

    @patch("local_agent.orchestrator.CodingAgent")
    def test_orchestrator_targeted_validation_failure_routes_to_repair(self, mock_coding_agent_cls):
        """Orchestrator enters failure analysis and repair immediately when targeted validation fails."""
        config = AgentConfig(
            project=self.project_path,
            provider="mock",
            max_iterations=2,
        )
        storage = JsonFileStorage(self.project_path / ".agent")
        repo_lock = MagicMock()
        repo_lock.__enter__.return_value = repo_lock
        repo_lock.__exit__.return_value = None
        memory_lock = MagicMock()
        memory_lock.__enter__.return_value = memory_lock
        memory_lock.__exit__.return_value = None

        mock_ca_instance = MagicMock()
        mock_ca_instance.prepare.return_value = [
            PreparedChange("modify", "src/calculator.py", "orig", "updated", "--- a\n+++ b\n")
        ]
        mock_ca_instance.diff.return_value = "--- a\n+++ b\n"
        mock_ca_instance.apply_prepared.return_value = ["src/calculator.py"]
        mock_coding_agent_cls.return_value = mock_ca_instance

        orchestrator = Orchestrator(config, storage, scheduler=None, repo_lock=repo_lock, memory_lock=memory_lock)

        call_count = 0
        def fake_validate(v_plan):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Targeted validation fails on first iteration
                return [ExecutionResult("pytest", 1, "FAIL", "FAILED")]
            # Passes subsequently
            return [ExecutionResult("pytest", 0, "PASS", "")]

        orchestrator._validate = fake_validate

        report = orchestrator.run("Fix calculator")

        self.assertIsNotNone(report)
        self.assertGreater(len(report.failures), 0)

    # -------------------------------------------------------------------------
    # 13. Circuit breaker & repeated call protection in verification
    # -------------------------------------------------------------------------

    def test_verification_circuit_breaker_repeated_calls(self):
        """Verification triggers circuit breaker when repeated tool calls occur."""
        policy = ToolExecutionPolicy(max_tool_steps=10, max_consecutive_repeats=2)
        provider = ScriptedVerificationProvider([
            ToolCall("call_1", "read_file_range", {"path": "src/calculator.py"}),
            ToolCall("call_2", "read_file_range", {"path": "src/calculator.py"}),
            ToolCall("call_3", "read_file_range", {"path": "src/calculator.py"}),
        ])
        verifier = VerificationIntelligence(provider, registry=self.registry, policy=policy)
        report = RunReport(project=self.context, task_id="task-cb")

        result = verifier.verify("Verify", self.plan, self.context, "", ["src/calculator.py"], report=report)

        self.assertTrue(result.verified)
        self.assertEqual(len(report.tool_metrics), 1)
        self.assertGreaterEqual(report.tool_metrics[0].circuit_breaker_events, 1)
        self.assertTrue(any("Circuit breaker" in hist[1].output for hist in report.tool_history))

    # -------------------------------------------------------------------------
    # 14. Zero-call verification does not create empty metric session
    # -------------------------------------------------------------------------

    def test_verification_zero_calls_does_not_create_empty_metrics(self):
        """Verification with 0 tool calls does not append an empty metrics session to report."""
        provider = ScriptedVerificationProvider([
            {"verified": True, "notes": "Immediate verification with zero tool calls"},
        ])
        verifier = VerificationIntelligence(provider, registry=self.registry)
        report = RunReport(project=self.context, task_id="task-zero")

        result = verifier.verify("Verify", self.plan, self.context, "", ["src/calculator.py"], report=report)

        self.assertTrue(result.verified)
        self.assertEqual(len(report.tool_metrics), 0)

    # -------------------------------------------------------------------------
    # 15. Security preservation in verification
    # -------------------------------------------------------------------------

    def test_verification_respects_secret_and_path_security(self):
        """Verification tools reject path traversal and secret access."""
        (self.project_path / ".env").write_text("SECRET_KEY=12345", encoding="utf-8")
        provider = ScriptedVerificationProvider([
            ToolCall("call_secret", "read_file_range", {"path": ".env"}),
            ToolCall("call_traversal", "read_file_range", {"path": "../outside.py"}),
            {"verified": False, "notes": "Rejected unsafe operations"},
        ])
        verifier = VerificationIntelligence(provider, registry=self.registry)
        report = RunReport(project=self.context, task_id="task-sec")

        result = verifier.verify("Verify", self.plan, self.context, "", ["src/calculator.py"], report=report)

        self.assertFalse(result.verified)
        self.assertEqual(len(report.tool_history), 2)
        self.assertTrue(report.tool_history[0][1].is_error)
        self.assertTrue(report.tool_history[1][1].is_error)

    # -------------------------------------------------------------------------
    # 16. Canonical history preservation with compaction
    # -------------------------------------------------------------------------

    def test_verification_preserves_canonical_history_with_compaction(self):
        """Canonical tool history in report retains full uncompacted outputs."""
        policy = ToolExecutionPolicy(max_tool_output_bytes=20, compaction_window=1, max_context_bytes=50)
        provider = ScriptedVerificationProvider([
            ToolCall("call_large", "read_file_range", {"path": "src/calculator.py"}),
            {"verified": True, "notes": "Verified"},
        ])
        verifier = VerificationIntelligence(provider, registry=self.registry, policy=policy)
        report = RunReport(project=self.context, task_id="task-compact")

        result = verifier.verify("Verify", self.plan, self.context, "", ["src/calculator.py"], report=report)

        self.assertTrue(result.verified)
        self.assertEqual(len(report.tool_history), 1)
        self.assertIn("def add", report.tool_history[0][1].output)


if __name__ == "__main__":
    unittest.main()
