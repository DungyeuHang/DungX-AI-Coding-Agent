"""Comprehensive behavioral tests for Phase 4.6 Tool-Assisted Code Review Intelligence."""

import json
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
    ProjectContext,
    ProviderCapability,
    ProviderError,
    RecoveryState,
    RepairSignature,
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
    BaseHTTPProvider,
    DeepSeekProvider,
    GeminiProvider,
    MockProvider,
    OpenAIProvider,
)
from local_agent.reviewer import Reviewer
from local_agent.storage import JsonFileStorage
from local_agent.tools import ToolRegistry


class ScriptedReviewProvider(AIProvider):
    """Provider double yielding scripted ToolCalls then a final ReviewResult."""

    def __init__(self, review_steps: list[ToolCall | ReviewResult | dict]):
        self.review_steps = list(review_steps)
        self.tool_calls_received: list[list[tuple[ToolCall, ToolResult]]] = []
        self.single_shot_review_called = False

    @property
    def capabilities(self) -> set[ProviderCapability]:
        return {
            ProviderCapability.PLANNING,
            ProviderCapability.IMPLEMENTATION,
            ProviderCapability.REPAIR,
            ProviderCapability.REVIEW,
            ProviderCapability.TOOL_USE,
        }

    def generate_plan(self, task: str, context: ProjectContext) -> Plan:
        return Plan(
            objective=task,
            files_likely_to_change=["src/main.py"],
            steps=["Step 1"],
            validation_strategy=["pytest"],
        )

    def generate_code(self, task: str, plan: Plan, context: ProjectContext, failure=None, review=None) -> list[FileOperation]:
        return [FileOperation("modify", "src/main.py", content="def add(a, b):\n    return a + b\n")]

    def analyze_failure(self, execution: ExecutionResult, diff: str, context: ProjectContext, plan: Plan) -> FailureAnalysis:
        return FailureAnalysis("Execution error", ["src/main.py"], "Fix code")

    def review_changes(self, task: str, plan: Plan, diff: str, context: ProjectContext) -> ReviewResult:
        self.single_shot_review_called = True
        return ReviewResult("APPROVED", "Single-shot review fallback")

    def review_changes_with_tools(
        self,
        task: str,
        plan: Plan,
        diff: str,
        context: ProjectContext,
        tools: list[ToolDefinition],
        tool_history: list[tuple[ToolCall, ToolResult]] | None = None,
    ) -> ToolCall | ReviewResult:
        if tool_history is not None:
            self.tool_calls_received.append(list(tool_history))
        if self.review_steps:
            item = self.review_steps.pop(0)
            if isinstance(item, dict):
                return ReviewResult(**item)
            return item
        return ReviewResult("APPROVED", "Default approved")


class NonToolReviewProvider(AIProvider):
    """Provider without TOOL_USE capability."""

    def __init__(self, verdict="APPROVED", summary="Looks good", findings=None):
        self.verdict = verdict
        self.summary = summary
        self.findings = findings or []
        self.review_called = False

    @property
    def capabilities(self) -> set[ProviderCapability]:
        return {
            ProviderCapability.PLANNING,
            ProviderCapability.IMPLEMENTATION,
            ProviderCapability.REPAIR,
            ProviderCapability.REVIEW,
        }

    def generate_plan(self, task: str, context: ProjectContext) -> Plan:
        return Plan(objective=task, files_likely_to_change=["src/main.py"], steps=["Step 1"])

    def generate_code(self, task: str, plan: Plan, context: ProjectContext, failure=None, review=None) -> list[FileOperation]:
        return [FileOperation("modify", "src/main.py", content="def add(a, b):\n    return a + b\n")]

    def analyze_failure(self, execution: ExecutionResult, diff: str, context: ProjectContext, plan: Plan) -> FailureAnalysis:
        return FailureAnalysis("Fail", ["src/main.py"], "Fix")

    def review_changes(self, task: str, plan: Plan, diff: str, context: ProjectContext) -> ReviewResult:
        self.review_called = True
        return ReviewResult(self.verdict, self.summary, self.findings)


class ReviewToolsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_path = Path(self.temp_dir.name)
        (self.project_path / "src").mkdir(parents=True, exist_ok=True)
        (self.project_path / "tests").mkdir(parents=True, exist_ok=True)
        (self.project_path / "src" / "main.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
        (self.project_path / "src" / "caller.py").write_text("from src.main import add\nprint(add(1, 2))\n", encoding="utf-8")
        (self.project_path / "tests" / "test_main.py").write_text("def test_add():\n    assert True\n", encoding="utf-8")

        self.config = AgentConfig(project=self.project_path, max_iterations=3)
        self.storage = JsonFileStorage(self.project_path)
        self.filesystem = ProjectFilesystem(self.project_path)
        self.registry = ToolRegistry(self.project_path, filesystem=self.filesystem)
        self.policy = ToolExecutionPolicy(max_tool_steps=5, max_tool_output_bytes=4000)
        self.context = ProjectContext(root=str(self.project_path))
        self.plan = Plan("Update add function", files_likely_to_change=["src/main.py"], steps=["Step 1"])
        self.diff = "--- a/src/main.py\n+++ b/src/main.py\n@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a + b\n+    return int(a) + int(b)\n"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_non_tool_provider_falls_back_to_review_changes(self):
        provider = NonToolReviewProvider(verdict="APPROVED", summary="Clean implementation")
        reviewer = Reviewer(provider, registry=self.registry, policy=self.policy)
        result = reviewer.review("Update add function", self.plan, self.diff, self.context)
        self.assertTrue(provider.review_called)
        self.assertEqual(result.verdict, "APPROVED")
        self.assertEqual(result.summary, "Clean implementation")

    def test_tool_assisted_review_multi_turn_approval(self):
        step1 = ToolCall("call_1", "read_file_range", {"path": "src/caller.py", "start_line": 1, "end_line": 10})
        step2 = ToolCall("call_2", "grep_code", {"query": "add("})
        final_review = ReviewResult("APPROVED", "Verified callers and usages; types are safe.", [])

        provider = ScriptedReviewProvider([step1, step2, final_review])
        reviewer = Reviewer(provider, registry=self.registry, policy=self.policy)
        report = RunReport(project=self.context)

        result = reviewer.review("Update add function", self.plan, self.diff, self.context, report=report)

        self.assertEqual(result.verdict, "APPROVED")
        self.assertIn("Verified callers", result.summary)
        self.assertEqual(len(report.tool_metrics), 1)
        self.assertEqual(report.tool_metrics[0].total_calls, 2)
        self.assertEqual(len(report.tool_history), 2)
        self.assertEqual(report.tool_history[0][0].tool_name, "read_file_range")
        self.assertEqual(report.tool_history[1][0].tool_name, "grep_code")

    def test_tool_assisted_review_rejection_with_evidence(self):
        step1 = ToolCall("call_1", "read_file_range", {"path": "src/caller.py", "start_line": 1, "end_line": 10})
        final_review = ReviewResult("CHANGES_REQUESTED", "Caller passes non-convertible types", ["caller.py may pass None"])

        provider = ScriptedReviewProvider([step1, final_review])
        reviewer = Reviewer(provider, registry=self.registry, policy=self.policy)
        report = RunReport(project=self.context)

        result = reviewer.review("Update add function", self.plan, self.diff, self.context, report=report)

        self.assertEqual(result.verdict, "CHANGES_REQUESTED")
        self.assertEqual(len(result.findings), 1)
        self.assertIn("caller.py may pass None", result.findings[0])
        self.assertEqual(len(report.tool_metrics), 1)

    def test_zero_tool_calls_does_not_create_empty_metrics(self):
        final_review = ReviewResult("APPROVED", "Direct approval without tools")
        provider = ScriptedReviewProvider([final_review])
        reviewer = Reviewer(provider, registry=self.registry, policy=self.policy)
        report = RunReport(project=self.context)

        result = reviewer.review("Update add function", self.plan, self.diff, self.context, report=report)

        self.assertEqual(result.verdict, "APPROVED")
        self.assertEqual(len(report.tool_metrics), 0)

    def test_policy_step_limit_falls_back_gracefully(self):
        policy = ToolExecutionPolicy(max_tool_steps=2)
        step1 = ToolCall("call_1", "read_file_range", {"path": "src/main.py", "start_line": 1, "end_line": 5})
        step2 = ToolCall("call_2", "read_file_range", {"path": "src/caller.py", "start_line": 1, "end_line": 5})
        step3 = ToolCall("call_3", "read_file_range", {"path": "tests/test_main.py", "start_line": 1, "end_line": 5})

        provider = ScriptedReviewProvider([step1, step2, step3])
        reviewer = Reviewer(provider, registry=self.registry, policy=policy)
        report = RunReport(project=self.context)

        result = reviewer.review("Update add function", self.plan, self.diff, self.context, report=report)

        self.assertEqual(result.verdict, "APPROVED")
        self.assertTrue(provider.single_shot_review_called)
        self.assertEqual(len(report.tool_metrics), 1)
        self.assertEqual(report.tool_metrics[0].total_calls, 2)

    def test_security_reviewer_cannot_write_or_escape(self):
        write_call = ToolCall("c_write", "write_file", {"path": "src/hack.py", "content": "bad"})
        final_review = ReviewResult("APPROVED", "Done")

        provider = ScriptedReviewProvider([write_call, final_review])
        reviewer = Reviewer(provider, registry=self.registry, policy=self.policy)

        result = reviewer.review("Check security", self.plan, self.diff, self.context)
        self.assertEqual(result.verdict, "APPROVED")
        self.assertFalse((self.project_path / "src" / "hack.py").exists())

    def test_orchestrator_review_integration_with_recovery_state(self):
        config = AgentConfig(project=self.project_path, max_iterations=1)
        step1 = ToolCall("call_1", "read_file_range", {"path": "src/caller.py", "start_line": 1, "end_line": 5})
        final_review = ReviewResult("CHANGES_REQUESTED", "Review rejected change", ["Refactor error handling"])

        provider = ScriptedReviewProvider([step1, final_review])
        repo_lock = threading.Lock()
        memory_lock = threading.Lock()
        orchestrator = Orchestrator(config, self.storage, scheduler=None, repo_lock=repo_lock, memory_lock=memory_lock)

        with patch("local_agent.orchestrator.build_provider", return_value=provider):
            with patch.object(orchestrator, "_validate") as mock_val:
                mock_val.return_value = [ExecutionResult(("pytest",), 0, "PASSED", "")]
                report = orchestrator.run("Task testing review integration")

        self.assertIsNotNone(report.review)
        self.assertEqual(report.review.verdict, "CHANGES_REQUESTED")
        self.assertIsNotNone(report.recovery_state)
        self.assertEqual(len(report.recovery_state.review_history), 1)
        self.assertEqual(report.recovery_state.review_history[0].findings, ["Refactor error handling"])

    def test_openai_review_changes_with_tools_parses_tool_call(self):
        cfg = AgentConfig(project=self.project_path, api_key="sk-test", model="gpt-4o")
        provider = OpenAIProvider(cfg)

        mock_payload = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "function": {
                                    "name": "read_file_range",
                                    "arguments": "{\"path\": \"src/main.py\", \"start_line\": 1, \"end_line\": 5}",
                                },
                            }
                        ]
                    }
                }
            ]
        }
        with patch.object(provider, "_request_json_api", return_value=mock_payload):
            tools = [ToolDefinition("read_file_range", "Read file", {"type": "object"})]
            res = provider.review_changes_with_tools("Review", self.plan, self.diff, self.context, tools)
            self.assertIsInstance(res, ToolCall)
            self.assertEqual(res.tool_name, "read_file_range")
            self.assertEqual(res.arguments["path"], "src/main.py")

    def test_openai_review_changes_with_tools_parses_final_review(self):
        cfg = AgentConfig(project=self.project_path, api_key="sk-test", model="gpt-4o")
        provider = OpenAIProvider(cfg)

        mock_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({"verdict": "APPROVED", "summary": "All good", "findings": []})
                    }
                }
            ]
        }
        with patch.object(provider, "_request_json_api", return_value=mock_payload):
            res = provider.review_changes_with_tools("Review", self.plan, self.diff, self.context, [])
            self.assertIsInstance(res, ReviewResult)
            self.assertEqual(res.verdict, "APPROVED")
            self.assertEqual(res.summary, "All good")

    def test_gemini_review_changes_with_tools_parses_function_call(self):
        cfg = AgentConfig(project=self.project_path, api_key="gemini-key", model="gemini-2.5-flash")
        provider = GeminiProvider(cfg)

        mock_payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "grep_code",
                                    "args": {"query": "add"},
                                }
                            }
                        ]
                    }
                }
            ]
        }
        with patch.object(provider, "_ensure_model"):
            with patch.object(provider, "_request_json", return_value=mock_payload):
                tools = [ToolDefinition("grep_code", "Grep", {"type": "object"})]
                res = provider.review_changes_with_tools("Review", self.plan, self.diff, self.context, tools)
                self.assertIsInstance(res, ToolCall)
                self.assertEqual(res.tool_name, "grep_code")
                self.assertEqual(res.arguments["query"], "add")

    def test_gemini_review_changes_with_tools_parses_final_review(self):
        cfg = AgentConfig(project=self.project_path, api_key="gemini-key", model="gemini-2.5-flash")
        provider = GeminiProvider(cfg)

        mock_payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps({"verdict": "CHANGES_REQUESTED", "summary": "Missing docs", "findings": ["Add docstring"]})
                            }
                        ]
                    }
                }
            ]
        }
        with patch.object(provider, "_ensure_model"):
            with patch.object(provider, "_request_json", return_value=mock_payload):
                res = provider.review_changes_with_tools("Review", self.plan, self.diff, self.context, [])
                self.assertIsInstance(res, ReviewResult)
                self.assertEqual(res.verdict, "CHANGES_REQUESTED")
                self.assertIn("Add docstring", res.findings)

    def test_anthropic_review_changes_with_tools_parses_tool_use(self):
        cfg = AgentConfig(project=self.project_path, api_key="anthropic-key", model="claude-3-5-sonnet-20241022")
        provider = AnthropicProvider(cfg)

        mock_payload = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "read_file_range",
                    "input": {"path": "src/caller.py", "start_line": 1, "end_line": 5},
                }
            ]
        }
        with patch.object(provider, "_request_json_api", return_value=mock_payload):
            tools = [ToolDefinition("read_file_range", "Read", {"type": "object"})]
            res = provider.review_changes_with_tools("Review", self.plan, self.diff, self.context, tools)
            self.assertIsInstance(res, ToolCall)
            self.assertEqual(res.tool_name, "read_file_range")
            self.assertEqual(res.arguments["path"], "src/caller.py")

    def test_anthropic_review_changes_with_tools_parses_final_review(self):
        cfg = AgentConfig(project=self.project_path, api_key="anthropic-key", model="claude-3-5-sonnet-20241022")
        provider = AnthropicProvider(cfg)

        mock_payload = {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({"verdict": "APPROVED", "summary": "Looks great", "findings": []}),
                }
            ]
        }
        with patch.object(provider, "_request_json_api", return_value=mock_payload):
            res = provider.review_changes_with_tools("Review", self.plan, self.diff, self.context, [])
            self.assertIsInstance(res, ReviewResult)
            self.assertEqual(res.verdict, "APPROVED")
            self.assertEqual(res.summary, "Looks great")

    def test_deepseek_and_antigravity_support_review_with_tools(self):
        cfg_ds = AgentConfig(project=self.project_path, api_key="ds-key", model="deepseek-chat")
        ds_provider = DeepSeekProvider(cfg_ds)
        self.assertTrue(hasattr(ds_provider, "review_changes_with_tools"))

        cfg_ag = AgentConfig(project=self.project_path, api_key="ag-key", model="gemini-3.7-flash")
        ag_provider = AntigravityProvider(cfg_ag)
        self.assertTrue(hasattr(ag_provider, "review_changes_with_tools"))

    def test_malformed_tool_call_raises_clean_provider_error(self):
        cfg = AgentConfig(project=self.project_path, api_key="sk-test", model="gpt-4o")
        provider = OpenAIProvider(cfg)

        mock_payload = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_bad",
                                "function": {
                                    "name": "read_file_range",
                                    "arguments": "{bad_json",
                                },
                            }
                        ]
                    }
                }
            ]
        }
        with patch.object(provider, "_request_json_api", return_value=mock_payload):
            tools = [ToolDefinition("read_file_range", "Read", {"type": "object"})]
            with self.assertRaises(ProviderError):
                provider.review_changes_with_tools("Review", self.plan, self.diff, self.context, tools)

    def test_repeated_review_rejection_triggers_abort(self):
        config = AgentConfig(project=self.project_path, max_iterations=5)
        step = ToolCall("call_1", "read_file_range", {"path": "src/main.py", "start_line": 1, "end_line": 5})
        rej1 = ReviewResult("CHANGES_REQUESTED", "Reject 1", ["Fix 1"])
        rej2 = ReviewResult("CHANGES_REQUESTED", "Reject 2", ["Fix 2"])
        rej3 = ReviewResult("CHANGES_REQUESTED", "Reject 3", ["Fix 3"])

        provider = ScriptedReviewProvider([step, rej1, step, rej2, step, rej3])
        repo_lock = threading.Lock()
        memory_lock = threading.Lock()
        orchestrator = Orchestrator(config, self.storage, scheduler=None, repo_lock=repo_lock, memory_lock=memory_lock)

        with patch("local_agent.orchestrator.build_provider", return_value=provider):
            with patch.object(orchestrator, "_validate") as mock_val:
                mock_val.return_value = [ExecutionResult(("pytest",), 0, "PASSED", "")]
                report = orchestrator.run("Task with 3 rejections")

        self.assertEqual(report.outcome, "REPEATED_REVIEW_REJECTION")
        self.assertEqual(report.recovery_state.abort_reason, "REPEATED_REVIEW_REJECTION")

    def test_checkpoint_continuation_context_preserves_review_recovery_history(self):
        now = datetime.now(timezone.utc)
        task = Task(task_id="test-review-chk", objective="Review chk", status=TaskStatus.PENDING, created_at=now, updated_at=now)
        self.storage.save_task(task)

        rec = RecoveryState(completed_iterations=1)
        rec.record_review(ReviewResult("CHANGES_REQUESTED", "Fix caller types", ["Check None"]))
        chk = Checkpoint(
            checkpoint_id="chk-rev",
            task_id=task.task_id,
            subtask_id="",
            timestamp=now,
            current_state_description="Paused after review",
            files_changed=["src/main.py"],
            repository_diff="",
            validation_state={"last_executions": [], "last_failures": []},
            continuation_context={"recovery_state": rec.to_dict()},
        )
        self.storage.save_checkpoint(chk)
        task.latest_checkpoint_id = "chk-rev"
        self.storage.save_task(task)

        repo_lock = threading.Lock()
        memory_lock = threading.Lock()
        orchestrator = Orchestrator(self.config, self.storage, scheduler=None, repo_lock=repo_lock, memory_lock=memory_lock)
        report = orchestrator._build_run_report(task)
        self.assertIsNotNone(report.recovery_state)
        self.assertEqual(len(report.recovery_state.review_history), 1)
        self.assertEqual(report.recovery_state.review_history[0].summary, "Fix caller types")

    def test_backward_compatible_checkpoint_loading_without_review_metrics(self):
        now = datetime.now(timezone.utc)
        task = Task(task_id="test-old-chk", objective="Old chk", status=TaskStatus.PENDING, created_at=now, updated_at=now)
        self.storage.save_task(task)

        chk = Checkpoint(
            checkpoint_id="chk-old",
            task_id=task.task_id,
            subtask_id="",
            timestamp=now,
            current_state_description="Old checkpoint without recovery_state",
            files_changed=["src/main.py"],
            repository_diff="",
            validation_state={"last_executions": [], "last_failures": []},
            continuation_context={},
        )
        self.storage.save_checkpoint(chk)
        task.latest_checkpoint_id = "chk-old"
        self.storage.save_task(task)

        repo_lock = threading.Lock()
        memory_lock = threading.Lock()
        orchestrator = Orchestrator(self.config, self.storage, scheduler=None, repo_lock=repo_lock, memory_lock=memory_lock)
        report = orchestrator._build_run_report(task)
        self.assertIsNone(report.recovery_state)

    def test_unknown_tool_call_is_handled_safely(self):
        bad_tool = ToolCall("call_unknown", "non_existent_tool", {"arg": "val"})
        final_review = ReviewResult("APPROVED", "Fallback approved after unknown tool")

        provider = ScriptedReviewProvider([bad_tool, final_review])
        reviewer = Reviewer(provider, registry=self.registry, policy=self.policy)
        report = RunReport(project=self.context)

        result = reviewer.review("Update add function", self.plan, self.diff, self.context, report=report)
        self.assertEqual(result.verdict, "APPROVED")
        self.assertTrue(len(report.tool_history) > 0)
        self.assertTrue(report.tool_history[0][1].is_error)

    def test_review_preserves_canonical_history_with_compaction(self):
        policy = ToolExecutionPolicy(max_tool_steps=5, compaction_window=1, max_context_bytes=200)
        step1 = ToolCall("call_1", "read_file_range", {"path": "src/main.py", "start_line": 1, "end_line": 5})
        step2 = ToolCall("call_2", "read_file_range", {"path": "src/caller.py", "start_line": 1, "end_line": 5})
        final_review = ReviewResult("APPROVED", "Compacted review passed")

        provider = ScriptedReviewProvider([step1, step2, final_review])
        reviewer = Reviewer(provider, registry=self.registry, policy=policy)
        report = RunReport(project=self.context)

        result = reviewer.review("Compacted review", self.plan, self.diff, self.context, report=report)
        self.assertEqual(result.verdict, "APPROVED")
        self.assertEqual(len(report.tool_history), 2)
        # Canonical history in report retains full uncompacted outputs
        self.assertIn("def add", report.tool_history[0][1].output)


if __name__ == "__main__":
    unittest.main()
