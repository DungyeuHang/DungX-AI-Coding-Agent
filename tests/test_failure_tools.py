from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import urllib.error

from local_agent.config import AgentConfig
from local_agent.failure import FailureAnalyzer
from local_agent.models import (
    ExecutionResult,
    FailureAnalysis,
    FileOperation,
    Plan,
    PolicyAction,
    PolicyDecision,
    ProjectContext,
    ProviderCapability,
    ProviderError,
    ReviewResult,
    RunReport,
    ToolCall,
    ToolDefinition,
    ToolExecutionMetrics,
    ToolExecutionPolicy,
    ToolResult,
)
from local_agent.orchestrator import Orchestrator
from local_agent.providers import (
    AIProvider,
    AnthropicProvider,
    GeminiProvider,
    MockProvider,
    OpenAIProvider,
)
from local_agent.storage import JsonFileStorage
from local_agent.tool_engine import ToolContextCompactor, ToolEngine
from local_agent.tools import ToolRegistry


class ScriptedDiagnosticProvider(AIProvider):
    """Provider double that yields scripted ToolCalls then a final FailureAnalysis."""

    def __init__(self, responses: list[ToolCall | FailureAnalysis]):
        self.responses = list(responses)
        self.tool_calls_received: list[list[tuple[ToolCall, ToolResult]]] = []
        self.analyze_failure_called = False

    @property
    def capabilities(self) -> set[ProviderCapability]:
        return {
            ProviderCapability.PLANNING,
            ProviderCapability.IMPLEMENTATION,
            ProviderCapability.REPAIR,
            ProviderCapability.REVIEW,
            ProviderCapability.TOOL_USE,
        }

    def analyze_failure(self, execution: ExecutionResult, diff: str, context: ProjectContext, plan: Plan) -> FailureAnalysis:
        self.analyze_failure_called = True
        return FailureAnalysis(
            probable_root_cause="One-shot fallback diagnosis",
            affected_files=["src/app.py"],
            recommended_fix="Fix app.py",
        )

    def analyze_failure_with_tools(
        self,
        execution: ExecutionResult,
        diff: str,
        context: ProjectContext,
        plan: Plan,
        tools: list[ToolDefinition],
        tool_history: list[tuple[ToolCall, ToolResult]] | None = None,
    ) -> ToolCall | FailureAnalysis:
        self.tool_calls_received.append(list(tool_history or []))
        if not self.responses:
            return self.analyze_failure(execution, diff, context, plan)
        return self.responses.pop(0)


class NonToolProvider(AIProvider):
    """Provider double without TOOL_USE capability."""

    def __init__(self):
        self.analyze_failure_called = False

    @property
    def capabilities(self) -> set[ProviderCapability]:
        return {
            ProviderCapability.PLANNING,
            ProviderCapability.IMPLEMENTATION,
            ProviderCapability.REPAIR,
            ProviderCapability.REVIEW,
        }

    def analyze_failure(self, execution: ExecutionResult, diff: str, context: ProjectContext, plan: Plan) -> FailureAnalysis:
        self.analyze_failure_called = True
        return FailureAnalysis(
            probable_root_cause="Non-tool diagnosis",
            affected_files=["src/fallback.py"],
            recommended_fix="Check fallback.py",
        )


class FailureToolsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_path = Path(self.temp_dir.name)
        (self.project_path / "src").mkdir(parents=True, exist_ok=True)
        (self.project_path / "src" / "app.py").write_text("def run():\n    raise ValueError('broken')\n", encoding="utf-8")
        (self.project_path / "tests").mkdir(parents=True, exist_ok=True)
        (self.project_path / "tests" / "test_app.py").write_text("def test_run():\n    assert run() == 1\n", encoding="utf-8")

        self.context = ProjectContext(root=str(self.project_path))
        self.plan = Plan(objective="Fix app.py bug", files_likely_to_change=["src/app.py"])
        self.failed_exec = ExecutionResult(
            command="pytest tests/test_app.py",
            exit_code=1,
            stdout="FAILED tests/test_app.py::test_run - ValueError: broken",
            stderr="Traceback (most recent call last):\n  File 'src/app.py', line 2, in run\n    raise ValueError('broken')",
        )
        self.registry = ToolRegistry(self.project_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    # -------------------------------------------------------------------------
    # 1. TOOL_USE capability enables tool-assisted diagnosis
    # -------------------------------------------------------------------------

    def test_tool_assisted_diagnosis_with_tool_provider(self):
        """FailureAnalyzer executes diagnostic tool loop when provider has TOOL_USE."""
        provider = ScriptedDiagnosticProvider([
            ToolCall("call_1", "read_file_range", {"path": "src/app.py", "start_line": 1, "end_line": 2}),
            FailureAnalysis(
                probable_root_cause="ValueError raised in run()",
                affected_files=["src/app.py"],
                recommended_fix="Return 1 instead of raising error",
            ),
        ])
        analyzer = FailureAnalyzer(provider, registry=self.registry)
        report = RunReport(project=self.context, task_id="task-1")

        result = analyzer.analyze(self.failed_exec, "", self.context, self.plan, report=report)

        self.assertIsInstance(result, FailureAnalysis)
        self.assertEqual(result.probable_root_cause, "ValueError raised in run()")
        self.assertEqual(result.affected_files, ["src/app.py"])
        self.assertEqual(result.recommended_fix, "Return 1 instead of raising error")
        self.assertEqual(len(report.tool_metrics), 1)
        self.assertEqual(report.tool_metrics[0].total_calls, 1)
        self.assertEqual(len(report.tool_history), 1)
        self.assertEqual(report.tool_history[0][0].tool_name, "read_file_range")
        self.assertIn("raise ValueError", report.tool_history[0][1].output)

    # -------------------------------------------------------------------------
    # 2. Provider without TOOL_USE falls back to 1-shot analyze_failure
    # -------------------------------------------------------------------------

    def test_non_tool_provider_fallback(self):
        """Provider without TOOL_USE falls back to analyze_failure()."""
        provider = NonToolProvider()
        analyzer = FailureAnalyzer(provider, registry=self.registry)
        report = RunReport(project=self.context, task_id="task-2")

        result = analyzer.analyze(self.failed_exec, "", self.context, self.plan, report=report)

        self.assertTrue(provider.analyze_failure_called)
        self.assertEqual(result.probable_root_cause, "Non-tool diagnosis")
        self.assertEqual(len(report.tool_metrics), 0)

    # -------------------------------------------------------------------------
    # 3. Reading failing file & 4. Searching repository evidence
    # -------------------------------------------------------------------------

    def test_multi_tool_diagnostic_exploration(self):
        """FailureAnalyzer performs multi-turn exploration with find_files, grep_code, and read_file_range."""
        provider = ScriptedDiagnosticProvider([
            ToolCall("call_1", "find_files", {"pattern": "*.py"}),
            ToolCall("call_2", "grep_code", {"query": "ValueError"}),
            ToolCall("call_3", "read_file_range", {"path": "src/app.py", "start_line": 1, "end_line": 3}),
            FailureAnalysis(
                probable_root_cause="Found broken ValueError in src/app.py",
                affected_files=["src/app.py"],
                recommended_fix="Remove raise ValueError",
            ),
        ])
        analyzer = FailureAnalyzer(provider, registry=self.registry)
        report = RunReport(project=self.context, task_id="task-3")

        result = analyzer.analyze(self.failed_exec, "", self.context, self.plan, report=report)

        self.assertEqual(result.probable_root_cause, "Found broken ValueError in src/app.py")
        self.assertEqual(len(report.tool_history), 3)
        self.assertEqual(report.tool_metrics[0].total_calls, 3)
        self.assertEqual(report.tool_metrics[0].calls_by_tool["find_files"], 1)
        self.assertEqual(report.tool_metrics[0].calls_by_tool["grep_code"], 1)
        self.assertEqual(report.tool_metrics[0].calls_by_tool["read_file_range"], 1)

    # -------------------------------------------------------------------------
    # 5. Diagnostic command execution & 6. Feedback into turns
    # -------------------------------------------------------------------------

    def test_run_command_sandbox_in_diagnostic_loop(self):
        """FailureAnalyzer can execute run_command_sandbox during failure diagnosis."""
        provider = ScriptedDiagnosticProvider([
            ToolCall("call_cmd", "run_command_sandbox", {"command": ["python", "-c", "print('diagnostic-output')"]}),
            FailureAnalysis(
                probable_root_cause="Verified via python sandbox",
                affected_files=["src/app.py"],
                recommended_fix="Apply fix",
            ),
        ])
        analyzer = FailureAnalyzer(provider, registry=self.registry)
        report = RunReport(project=self.context, task_id="task-4")

        result = analyzer.analyze(self.failed_exec, "", self.context, self.plan, report=report)

        self.assertEqual(result.probable_root_cause, "Verified via python sandbox")
        self.assertEqual(len(report.tool_history), 1)
        self.assertIn("diagnostic-output", report.tool_history[0][1].output)
        # Verify subsequent turn received tool output
        self.assertEqual(len(provider.tool_calls_received), 2)
        self.assertEqual(provider.tool_calls_received[1][0][0].tool_name, "run_command_sandbox")
        self.assertIn("diagnostic-output", provider.tool_calls_received[1][0][1].output)

    # -------------------------------------------------------------------------
    # 7. Policy enforcement & Step limit fallback
    # -------------------------------------------------------------------------

    def test_step_limit_fallback_to_one_shot(self):
        """When diagnostic loop exceeds policy step limit, it falls back to 1-shot analyze_failure."""
        policy = ToolExecutionPolicy(max_tool_steps=2)
        # Provider produces 3 tool calls without concluding
        provider = ScriptedDiagnosticProvider([
            ToolCall("call_1", "read_file_range", {"path": "src/app.py"}),
            ToolCall("call_2", "read_file_range", {"path": "tests/test_app.py"}),
            ToolCall("call_3", "read_file_range", {"path": "src/app.py"}),
        ])
        analyzer = FailureAnalyzer(provider, registry=self.registry, policy=policy)
        report = RunReport(project=self.context, task_id="task-5")

        result = analyzer.analyze(self.failed_exec, "", self.context, self.plan, report=report)

        # Fallback to one-shot is triggered
        self.assertTrue(provider.analyze_failure_called)
        self.assertEqual(result.probable_root_cause, "One-shot fallback diagnosis")
        self.assertEqual(len(report.tool_metrics), 1)
        self.assertEqual(report.tool_metrics[0].termination_reason, "max_steps_exceeded")

    # -------------------------------------------------------------------------
    # 8. Circuit breaker in failure diagnosis
    # -------------------------------------------------------------------------

    def test_circuit_breaker_in_diagnostic_loop(self):
        """Repeated identical tool calls in diagnostic loop trigger circuit breaker then continue."""
        provider = ScriptedDiagnosticProvider([
            ToolCall("call_1", "read_file_range", {"path": "src/app.py"}),
            ToolCall("call_2", "read_file_range", {"path": "src/app.py"}),
            ToolCall("call_3", "read_file_range", {"path": "src/app.py"}),
            FailureAnalysis(
                probable_root_cause="Diagnosis after breaker",
                affected_files=["src/app.py"],
                recommended_fix="Fix",
            ),
        ])
        analyzer = FailureAnalyzer(provider, registry=self.registry)
        report = RunReport(project=self.context, task_id="task-6")

        result = analyzer.analyze(self.failed_exec, "", self.context, self.plan, report=report)

        self.assertEqual(result.probable_root_cause, "Diagnosis after breaker")
        self.assertGreater(report.tool_metrics[0].circuit_breaker_events, 0)

    # -------------------------------------------------------------------------
    # 9. Real Providers: OpenAIProvider analyze_failure_with_tools
    # -------------------------------------------------------------------------

    @patch("urllib.request.urlopen")
    def test_openai_analyze_failure_with_tools(self, mock_urlopen):
        """OpenAIProvider handles tool calling and final response in analyze_failure_with_tools."""
        config = AgentConfig(
            project=self.project_path,
            provider="openai",
            model="gpt-4o",
            api_key="test-key",
            api_base_url="https://api.openai.com/v1",
        )
        provider = OpenAIProvider(config)

        # 1. Tool call response
        resp1 = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_oai_diag",
                                "type": "function",
                                "function": {
                                    "name": "read_file_range",
                                    "arguments": json.dumps({"path": "src/app.py"}),
                                },
                            }
                        ],
                    }
                }
            ]
        }
        # 2. Final FailureAnalysis JSON response
        resp2 = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({
                            "probable_root_cause": "OpenAI diagnosed bug in app.py",
                            "affected_files": ["src/app.py"],
                            "recommended_fix": "Fix error handling",
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

        analyzer = FailureAnalyzer(provider, registry=self.registry)
        report = RunReport(project=self.context, task_id="task-oai")

        result = analyzer.analyze(self.failed_exec, "", self.context, self.plan, report=report)

        self.assertEqual(result.probable_root_cause, "OpenAI diagnosed bug in app.py")
        self.assertEqual(result.affected_files, ["src/app.py"])
        self.assertEqual(result.recommended_fix, "Fix error handling")
        self.assertEqual(len(report.tool_history), 1)

    # -------------------------------------------------------------------------
    # 10. Real Providers: AnthropicProvider analyze_failure_with_tools
    # -------------------------------------------------------------------------

    @patch("urllib.request.urlopen")
    def test_anthropic_analyze_failure_with_tools(self, mock_urlopen):
        """AnthropicProvider handles tool calling and final response in analyze_failure_with_tools."""
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
                    "id": "toolu_anthropic_diag",
                    "name": "grep_code",
                    "input": {"query": "broken"},
                }
            ],
            "stop_reason": "tool_use",
        }
        resp2 = {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({
                        "probable_root_cause": "Anthropic diagnosed broken string",
                        "affected_files": ["src/app.py"],
                        "recommended_fix": "Replace broken with valid output",
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

        analyzer = FailureAnalyzer(provider, registry=self.registry)
        report = RunReport(project=self.context, task_id="task-anthropic")

        result = analyzer.analyze(self.failed_exec, "", self.context, self.plan, report=report)

        self.assertEqual(result.probable_root_cause, "Anthropic diagnosed broken string")
        self.assertEqual(len(report.tool_history), 1)

    # -------------------------------------------------------------------------
    # 11. Real Providers: GeminiProvider analyze_failure_with_tools
    # -------------------------------------------------------------------------

    @patch("urllib.request.urlopen")
    def test_gemini_analyze_failure_with_tools(self, mock_urlopen):
        """GeminiProvider handles functionCall and final response in analyze_failure_with_tools."""
        config = AgentConfig(
            project=self.project_path,
            provider="gemini",
            model="gemini-2.5-flash",
            api_key="test-key-gemini",
            gemini_base_url="https://generativelanguage.googleapis.com/v1beta",
        )
        provider = GeminiProvider(config)

        # Mock model discovery response
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
                                    "name": "read_file_range",
                                    "args": {"path": "src/app.py"},
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
                                    "probable_root_cause": "Gemini diagnosed error",
                                    "affected_files": ["src/app.py"],
                                    "recommended_fix": "Fix error",
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

        analyzer = FailureAnalyzer(provider, registry=self.registry)
        report = RunReport(project=self.context, task_id="task-gemini")

        result = analyzer.analyze(self.failed_exec, "", self.context, self.plan, report=report)

        self.assertEqual(result.probable_root_cause, "Gemini diagnosed error")
        self.assertEqual(len(report.tool_history), 1)

    # -------------------------------------------------------------------------
    # 12. Orchestrator integration: Failure diagnosis -> Repair
    # -------------------------------------------------------------------------

    def test_orchestrator_failure_diagnosis_integration(self):
        """Orchestrator invokes FailureAnalyzer with ToolRegistry and collects diagnostic telemetry."""
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

        orchestrator = Orchestrator(config, storage, scheduler=None, repo_lock=repo_lock, memory_lock=memory_lock)

        # Mock validator to fail first iteration, pass second
        call_count = 0
        def fake_validate(plan):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [ExecutionResult("pytest", 1, "FAIL", "FAILED")]
            return [ExecutionResult("pytest", 0, "PASS", "")]

        orchestrator._validate = fake_validate

        # Mock coding agent to avoid empty patch errors
        orchestrator.coding_agent = MagicMock()
        orchestrator.coding_agent.prepare.return_value = []
        orchestrator.coding_agent.diff.return_value = "--- a\n+++ b\n"
        orchestrator.coding_agent.apply_prepared.return_value = []

        report = orchestrator.run("Fix failing test")

        self.assertIsNotNone(report)
        self.assertGreater(len(report.failures), 0)

    # -------------------------------------------------------------------------
    # 13. DeepSeek and Antigravity inheritance
    # -------------------------------------------------------------------------

    def test_deepseek_and_antigravity_have_analyze_failure_with_tools(self):
        """DeepSeekProvider and AntigravityProvider expose analyze_failure_with_tools."""
        from local_agent.providers import DeepSeekProvider, AntigravityProvider

        ds_config = AgentConfig(project=self.project_path, provider="deepseek", api_key="k")
        ag_config = AgentConfig(project=self.project_path, provider="antigravity", api_key="k")

        ds_p = DeepSeekProvider(ds_config)
        ag_p = AntigravityProvider(ag_config)

        self.assertTrue(hasattr(ds_p, "analyze_failure_with_tools"))
        self.assertTrue(hasattr(ag_p, "analyze_failure_with_tools"))
        self.assertIn(ProviderCapability.TOOL_USE, ds_p.capabilities)
        self.assertIn(ProviderCapability.TOOL_USE, ag_p.capabilities)

    # -------------------------------------------------------------------------
    # 14. Malformed tool arguments & Error handling
    # -------------------------------------------------------------------------

    @patch("urllib.request.urlopen")
    def test_anthropic_analyze_failure_malformed_input(self, mock_urlopen):
        """Malformed tool arguments in analyze_failure_with_tools raise ProviderError."""
        config = AgentConfig(project=self.project_path, provider="anthropic", api_key="k")
        provider = AnthropicProvider(config)

        mock_payload = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_bad",
                    "name": "read_file_range",
                    "input": "{invalid_json",
                }
            ]
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(mock_payload).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        analyzer = FailureAnalyzer(provider, registry=self.registry)
        with self.assertRaises(ProviderError):
            analyzer.analyze(self.failed_exec, "", self.context, self.plan)

    # -------------------------------------------------------------------------
    # 15. HTTP Errors in Failure Analysis
    # -------------------------------------------------------------------------

    @patch("urllib.request.urlopen")
    def test_http_error_in_failure_analysis(self, mock_urlopen):
        """HTTP errors during failure diagnosis propagate properly."""
        from local_agent.models import RateLimitError

        config = AgentConfig(project=self.project_path, provider="anthropic", api_key="k")
        provider = AnthropicProvider(config)

        err_429 = urllib.error.HTTPError(
            "https://api.anthropic.com/v1/messages", 429, "Too Many Requests", {}, io.BytesIO(b'{"error":{"message":"rate limit"}}')
        )
        mock_urlopen.side_effect = err_429

        analyzer = FailureAnalyzer(provider, registry=self.registry)
        with self.assertRaises(RateLimitError):
            analyzer.analyze(self.failed_exec, "", self.context, self.plan)


if __name__ == "__main__":
    unittest.main()
