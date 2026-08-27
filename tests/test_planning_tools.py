"""Comprehensive behavioral tests for Phase 4.7 Tool-Assisted Architectural Planning & Task Decomposition Intelligence."""

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
)
from local_agent.orchestrator import Orchestrator
from local_agent.planner import GraphValidator, Planner
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
from local_agent.storage import JsonFileStorage
from local_agent.tools import ToolRegistry


class ScriptedPlanningProvider(AIProvider):
    """Provider double yielding scripted ToolCalls then a final Plan or dict."""

    def __init__(self, planning_steps: list[ToolCall | Plan | TaskPlan | dict]):
        self.planning_steps = list(planning_steps)
        self.tool_calls_received: list[list[tuple[ToolCall, ToolResult]]] = []
        self.single_shot_plan_called = False

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
        self.single_shot_plan_called = True
        return Plan(
            objective=task,
            files_likely_to_change=["src/main.py"],
            steps=["Single-shot fallback step"],
            validation_strategy=["pytest"],
        )

    def generate_plan_with_tools(
        self,
        task: str,
        context: ProjectContext,
        tools: list[ToolDefinition],
        tool_history: list[tuple[ToolCall, ToolResult]] | None = None,
    ) -> ToolCall | Plan:
        if tool_history is not None:
            self.tool_calls_received.append(list(tool_history))
        if self.planning_steps:
            return self.planning_steps.pop(0)
        return self.generate_plan(task, context)

    def generate_code(self, task: str, plan: Plan, context: ProjectContext, failure=None, review=None) -> list[FileOperation]:
        return [FileOperation("modify", "src/main.py", content="def add(a, b):\n    return a + b\n")]

    def analyze_failure(self, execution: ExecutionResult, diff: str, context: ProjectContext, plan: Plan) -> FailureAnalysis:
        return FailureAnalysis("Execution error", ["src/main.py"], "Fix code")

    def review_changes(self, task: str, plan: Plan, diff: str, context: ProjectContext) -> ReviewResult:
        return ReviewResult("APPROVED", "LGTM")


class PlanningToolsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_dir = Path(self.temp_dir.name)

        # Create basic directory structure
        self.src_dir = self.project_dir / "src"
        self.src_dir.mkdir(parents=True, exist_ok=True)
        self.main_file = self.src_dir / "main.py"
        self.main_file.write_text(
            "def calculate(a: int, b: int) -> int:\n    return a + b\n\ndef helper() -> str:\n    return 'ready'\n",
            encoding="utf-8",
        )
        self.utils_file = self.src_dir / "utils.py"
        self.utils_file.write_text("class ConfigManager:\n    debug = True\n", encoding="utf-8")

        self.filesystem = ProjectFilesystem(self.project_dir)
        self.registry = ToolRegistry(self.project_dir, filesystem=self.filesystem)
        self.context = ProjectContext(root=str(self.project_dir))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_base_provider_fallback_to_generate_plan(self):
        """Base AIProvider.generate_plan_with_tools falls back to generate_plan."""
        class MinimalProvider(AIProvider):
            def generate_plan(self, task: str, context: ProjectContext) -> Plan:
                return Plan(objective=task, steps=["Step A"])

        provider = MinimalProvider()
        result = provider.generate_plan_with_tools("Test task", self.context, tools=[])
        self.assertIsInstance(result, Plan)
        self.assertEqual(result.objective, "Test task")

    def test_planner_fallback_when_tool_use_unavailable(self):
        """Planner falls back to single-shot generate_plan if provider lacks TOOL_USE."""
        class NonToolProvider(AIProvider):
            @property
            def capabilities(self) -> set[ProviderCapability]:
                return {ProviderCapability.PLANNING}

            def generate_plan(self, task: str, context: ProjectContext) -> Plan:
                return Plan(objective=task, steps=["Single shot plan"])

        provider = NonToolProvider()
        planner = Planner(provider, registry=self.registry)
        plan = planner.create_plan_for_task("Build feature", self.context)
        self.assertEqual(plan.steps, ["Single shot plan"])

    def test_planner_fallback_when_registry_is_none(self):
        """Planner falls back to single-shot generate_plan when registry is None."""
        provider = ScriptedPlanningProvider([
            ToolCall("call_1", "find_files", {"pattern": "*.py"}),
        ])
        planner = Planner(provider, registry=None)
        plan = planner.create_plan_for_task("Task X", self.context)
        self.assertTrue(provider.single_shot_plan_called)
        self.assertEqual(plan.steps, ["Single-shot fallback step"])

    def test_tool_assisted_planner_single_turn_plan(self):
        """Planner executes a tool step and then captures the final Plan."""
        provider = ScriptedPlanningProvider([
            ToolCall("call_1", "read_file_range", {"path": "src/main.py", "start_line": 1, "end_line": 2}),
            Plan(
                objective="Refactor calculate",
                files_to_inspect=["src/main.py"],
                files_likely_to_change=["src/main.py"],
                steps=["Inspect calculate", "Update logic"],
                validation_strategy=["pytest"],
                risks=["Breaking callers"],
            ),
        ])
        report = RunReport(self.context, "task-1")
        planner = Planner(provider, registry=self.registry)
        plan = planner.create_plan_for_task("Refactor calculate", self.context, report=report)

        self.assertEqual(plan.objective, "Refactor calculate")
        self.assertEqual(plan.steps, ["Inspect calculate", "Update logic"])
        self.assertEqual(len(report.tool_metrics), 1)
        self.assertEqual(report.tool_metrics[0].total_calls, 1)
        self.assertEqual(report.tool_metrics[0].calls_by_tool.get("read_file_range"), 1)

    def test_tool_assisted_planner_multi_turn_exploration(self):
        """Planner explores multiple tools across turns before concluding."""
        provider = ScriptedPlanningProvider([
            ToolCall("call_1", "find_files", {"pattern": "*.py"}),
            ToolCall("call_2", "grep_code", {"query": "ConfigManager"}),
            ToolCall("call_3", "read_file_range", {"path": "src/utils.py", "start_line": 1, "end_line": 2}),
            {
                "objective": "Enable debug configuration",
                "files_to_inspect": ["src/utils.py"],
                "files_likely_to_change": ["src/utils.py"],
                "steps": ["Update ConfigManager", "Verify debug mode"],
                "risks": [],
            },
        ])
        report = RunReport(self.context, "task-2")
        planner = Planner(provider, registry=self.registry)
        plan = planner.create_plan_for_task("Enable debug configuration", self.context, report=report)

        self.assertEqual(plan.objective, "Enable debug configuration")
        self.assertIn("src/utils.py", plan.files_likely_to_change)
        self.assertEqual(len(report.tool_metrics), 1)
        self.assertEqual(report.tool_metrics[0].total_calls, 3)
        self.assertEqual(report.tool_metrics[0].calls_by_tool.get("find_files"), 1)
        self.assertEqual(report.tool_metrics[0].calls_by_tool.get("grep_code"), 1)
        self.assertEqual(report.tool_metrics[0].calls_by_tool.get("read_file_range"), 1)

    def test_read_file_range_usage_in_planning(self):
        """Planner reads file content via read_file_range tool safely."""
        provider = ScriptedPlanningProvider([
            ToolCall("call_read", "read_file_range", {"path": "src/main.py", "start_line": 1, "end_line": 2}),
            Plan(objective="Task", steps=["Step 1"]),
        ])
        planner = Planner(provider, registry=self.registry)
        planner.create_plan_for_task("Task", self.context)
        self.assertTrue(len(provider.tool_calls_received) >= 1)
        call, result = provider.tool_calls_received[-1][0]
        self.assertEqual(call.tool_name, "read_file_range")
        self.assertIn("def calculate", result.output)

    def test_grep_code_usage_in_planning(self):
        """Planner finds symbol references via grep_code tool safely."""
        provider = ScriptedPlanningProvider([
            ToolCall("call_grep", "grep_code", {"pattern": "helper"}),
            Plan(objective="Task", steps=["Step 1"]),
        ])
        planner = Planner(provider, registry=self.registry)
        planner.create_plan_for_task("Task", self.context)
        self.assertTrue(len(provider.tool_calls_received) >= 1)
        call, result = provider.tool_calls_received[-1][0]
        self.assertEqual(call.tool_name, "grep_code")
        self.assertIn("helper", result.output)

    def test_find_files_usage_in_planning(self):
        """Planner locates project files via find_files tool."""
        provider = ScriptedPlanningProvider([
            ToolCall("call_find", "find_files", {"pattern": "*.py"}),
            Plan(objective="Task", steps=["Step 1"]),
        ])
        planner = Planner(provider, registry=self.registry)
        planner.create_plan_for_task("Task", self.context)
        self.assertTrue(len(provider.tool_calls_received) >= 1)
        call, result = provider.tool_calls_received[-1][0]
        self.assertEqual(call.tool_name, "find_files")
        self.assertIn("src/main.py", result.output)

    def test_search_symbols_usage_in_planning(self):
        """Planner uses search_symbols when semantic index is provided."""
        mock_index = MagicMock()
        mock_symbol = MagicMock()
        mock_symbol.name = "calculate"
        mock_symbol.kind = "function"
        mock_symbol.location.start_line = 1
        mock_index.search_symbols.return_value = [("src/main.py", mock_symbol)]
        registry = ToolRegistry(self.project_dir, filesystem=self.filesystem, semantic_index=mock_index)

        provider = ScriptedPlanningProvider([
            ToolCall("call_sym", "search_symbols", {"symbol_name": "calculate"}),
            Plan(objective="Task", steps=["Step 1"]),
        ])
        planner = Planner(provider, registry=registry)
        planner.create_plan_for_task("Task", self.context)
        self.assertTrue(len(provider.tool_calls_received) >= 1)
        call, result = provider.tool_calls_received[-1][0]
        self.assertEqual(call.tool_name, "search_symbols")
        self.assertIn("calculate", result.output)

    def test_policy_step_limit_falls_back_gracefully(self):
        """When max_tool_steps is reached, Planner falls back to single-shot generate_plan."""
        infinite_calls = [
            ToolCall(f"call_{i}", "find_files", {"pattern": "*.py"})
            for i in range(10)
        ]
        provider = ScriptedPlanningProvider(infinite_calls)
        policy = ToolExecutionPolicy(max_tool_steps=3)
        planner = Planner(provider, registry=self.registry, policy=policy)
        plan = planner.create_plan_for_task("Exhaust steps", self.context)

        self.assertTrue(provider.single_shot_plan_called)
        self.assertEqual(plan.steps, ["Single-shot fallback step"])

    def test_policy_output_byte_budget_enforcement(self):
        """Tool outputs exceeding max_tool_output_bytes are clamped."""
        large_file = self.src_dir / "large.py"
        large_file.write_text("x = 1\n" * 2000, encoding="utf-8")

        provider = ScriptedPlanningProvider([
            ToolCall("call_large", "read_file_range", {"path": "src/large.py", "start_line": 1, "end_line": 1500}),
            Plan(objective="Task", steps=["Step 1"]),
        ])
        policy = ToolExecutionPolicy(max_tool_output_bytes=500)
        planner = Planner(provider, registry=self.registry, policy=policy)
        planner.create_plan_for_task("Task", self.context)

        self.assertTrue(len(provider.tool_calls_received) >= 1)
        _, result = provider.tool_calls_received[-1][0]
        self.assertTrue(len(result.output.encode("utf-8")) <= 650)

    def test_planning_preserves_canonical_history_with_compaction(self):
        """Canonical tool history retains complete outputs when compaction applies."""
        provider = ScriptedPlanningProvider([
            ToolCall("call_1", "read_file_range", {"path": "src/main.py", "start_line": 1, "end_line": 3}),
            ToolCall("call_2", "read_file_range", {"path": "src/utils.py", "start_line": 1, "end_line": 2}),
            Plan(objective="Task", steps=["Done"]),
        ])
        report = RunReport(self.context, "task-compact")
        policy = ToolExecutionPolicy(max_context_bytes=200, compaction_window=1)
        planner = Planner(provider, registry=self.registry, policy=policy)
        planner.create_plan_for_task("Task", self.context, report=report)

        self.assertEqual(len(report.tool_history), 2)
        self.assertIn("def calculate", report.tool_history[0][1].output)

    def test_zero_tool_calls_does_not_create_empty_metrics(self):
        """Planner with 0 tool calls does not append an empty metrics session to report."""
        provider = ScriptedPlanningProvider([
            Plan(objective="Immediate plan", steps=["Step 1"]),
        ])
        report = RunReport(self.context, "task-zero")
        planner = Planner(provider, registry=self.registry)
        planner.create_plan_for_task("Task", self.context, report=report)
        self.assertEqual(len(report.tool_metrics), 0)

    def test_create_task_plan_with_subtask_graph(self):
        """Planner creates a valid TaskPlan with subtask graph."""
        provider = ScriptedPlanningProvider([
            ToolCall("call_1", "find_files", {"pattern": "*.py"}),
            Plan(
                objective="Implement auth",
                steps=["Create models", "Add endpoints", "Write tests"],
                risks=["Token expiry bug"],
            ),
        ])
        report = RunReport(self.context, "task-graph")
        planner = Planner(provider, registry=self.registry)
        task_plan = planner.create_task_plan("Implement auth", self.context, report=report)

        self.assertIsInstance(task_plan, TaskPlan)
        self.assertEqual(len(task_plan.subtasks), 3)
        self.assertEqual(task_plan.subtasks[0].title, "Create models")
        self.assertEqual(task_plan.subtasks[1].dependencies, [task_plan.subtasks[0].subtask_id])
        self.assertEqual(task_plan.risks, ["Token expiry bug"])

    def test_create_subtask_plan_for_single_subtask(self):
        """Planner creates a focused Plan for a specific subtask."""
        subtask = Subtask(
            subtask_id="sub-1",
            title="Update calculator",
            goal="Add multiply support",
        )
        provider = ScriptedPlanningProvider([
            ToolCall("call_1", "read_file_range", {"path": "src/main.py", "start_line": 1, "end_line": 3}),
            Plan(objective="Add multiply support", steps=["Implement multiply", "Validate with tests"]),
        ])
        planner = Planner(provider, registry=self.registry)
        plan = planner.create_subtask_plan(subtask, self.context)

        self.assertIsInstance(plan, Plan)
        self.assertEqual(plan.objective, "Add multiply support")
        self.assertEqual(plan.steps, ["Implement multiply", "Validate with tests"])

    def test_openai_generate_plan_with_tools_parses_tool_call(self):
        """OpenAIProvider formats tools and parses tool_calls message."""
        cfg = AgentConfig(project=self.project_dir, api_key="fake-key", model="gpt-4.1-mini")
        provider = OpenAIProvider(cfg)
        mock_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_openai_1",
                                "type": "function",
                                "function": {
                                    "name": "find_files",
                                    "arguments": json.dumps({"pattern": "*.py"}),
                                },
                            }
                        ],
                    }
                }
            ]
        }
        with patch.object(provider, "_request_json_api", return_value=mock_response):
            resp = provider.generate_plan_with_tools("Plan task", self.context, tools=self.registry.definitions())
            self.assertIsInstance(resp, ToolCall)
            self.assertEqual(resp.tool_name, "find_files")
            self.assertEqual(resp.arguments, {"pattern": "*.py"})

    def test_openai_generate_plan_with_tools_parses_final_plan(self):
        """OpenAIProvider parses final JSON response into a Plan."""
        cfg = AgentConfig(project=self.project_dir, api_key="fake-key", model="gpt-4.1-mini")
        provider = OpenAIProvider(cfg)
        mock_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({
                            "objective": "Add division",
                            "files_to_inspect": ["src/main.py"],
                            "files_to_modify": ["src/main.py"],
                            "steps": ["Implement div", "Handle div by zero"],
                            "validation_strategy": ["pytest tests/"],
                            "risks": ["ZeroDivisionError"],
                        }),
                    }
                }
            ]
        }
        with patch.object(provider, "_request_json_api", return_value=mock_response):
            resp = provider.generate_plan_with_tools("Plan task", self.context, tools=[])
            self.assertIsInstance(resp, Plan)
            self.assertEqual(resp.objective, "Add division")
            self.assertEqual(resp.files_likely_to_change, ["src/main.py"])
            self.assertEqual(resp.steps, ["Implement div", "Handle div by zero"])

    def test_gemini_generate_plan_with_tools_parses_function_call(self):
        """GeminiProvider parses functionCall part into ToolCall."""
        cfg = AgentConfig(project=self.project_dir, api_key="fake-key", model="gemini-2.5-flash")
        provider = GeminiProvider(cfg)
        mock_response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "grep_code",
                                    "args": {"pattern": "calculate"},
                                }
                            }
                        ]
                    }
                }
            ]
        }
        with patch.object(provider, "_ensure_model"):
            with patch.object(provider, "_request_json", return_value=mock_response):
                resp = provider.generate_plan_with_tools("Plan task", self.context, tools=self.registry.definitions())
                self.assertIsInstance(resp, ToolCall)
                self.assertEqual(resp.tool_name, "grep_code")
                self.assertEqual(resp.arguments, {"pattern": "calculate"})

    def test_gemini_generate_plan_with_tools_parses_final_plan(self):
        """GeminiProvider parses text candidate JSON into Plan."""
        cfg = AgentConfig(project=self.project_dir, api_key="fake-key", model="gemini-2.5-flash")
        provider = GeminiProvider(cfg)
        mock_response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps({
                                    "objective": "Build API",
                                    "files_to_modify": ["src/main.py"],
                                    "steps": ["Step A", "Step B"],
                                    "validation_commands": ["npm test"],
                                    "risks": ["Latency"],
                                })
                            }
                        ]
                    }
                }
            ]
        }
        with patch.object(provider, "_ensure_model"):
            with patch.object(provider, "_request_json", return_value=mock_response):
                resp = provider.generate_plan_with_tools("Plan task", self.context, tools=[])
                self.assertIsInstance(resp, Plan)
                self.assertEqual(resp.objective, "Build API")
                self.assertEqual(resp.steps, ["Step A", "Step B"])

    def test_anthropic_generate_plan_with_tools_parses_tool_use(self):
        """AnthropicProvider parses tool_use block into ToolCall."""
        cfg = AgentConfig(project=self.project_dir, api_key="fake-key", model="claude-3-7-sonnet")
        provider = AnthropicProvider(cfg)
        mock_response = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "read_file_range",
                    "input": {"path": "src/main.py", "start_line": 1, "end_line": 5},
                }
            ]
        }
        with patch.object(provider, "_request_json_api", return_value=mock_response):
            resp = provider.generate_plan_with_tools("Plan task", self.context, tools=self.registry.definitions())
            self.assertIsInstance(resp, ToolCall)
            self.assertEqual(resp.call_id, "toolu_01")
            self.assertEqual(resp.tool_name, "read_file_range")
            self.assertEqual(resp.arguments, {"path": "src/main.py", "start_line": 1, "end_line": 5})

    def test_anthropic_generate_plan_with_tools_parses_final_plan(self):
        """AnthropicProvider parses text content block into Plan."""
        cfg = AgentConfig(project=self.project_dir, api_key="fake-key", model="claude-3-7-sonnet")
        provider = AnthropicProvider(cfg)
        mock_response = {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({
                        "objective": "Add cache layer",
                        "files_to_modify": ["src/utils.py"],
                        "steps": ["Define cache class", "Integrate cache"],
                        "validation_commands": ["pytest"],
                        "risks": ["Stale cache"],
                    }),
                }
            ]
        }
        with patch.object(provider, "_request_json_api", return_value=mock_response):
            resp = provider.generate_plan_with_tools("Plan task", self.context, tools=[])
            self.assertIsInstance(resp, Plan)
            self.assertEqual(resp.objective, "Add cache layer")
            self.assertEqual(resp.steps, ["Define cache class", "Integrate cache"])

    def test_deepseek_and_antigravity_support_plan_with_tools(self):
        """DeepSeekProvider and AntigravityProvider expose generate_plan_with_tools via inheritance."""
        cfg_ds = AgentConfig(project=self.project_dir, api_key="fake-key", model="deepseek-chat")
        deepseek = DeepSeekProvider(cfg_ds)
        cfg_ag = AgentConfig(project=self.project_dir, api_key="fake-key", model="gemini-3.7-flash")
        antigravity = AntigravityProvider(cfg_ag)

        self.assertTrue(hasattr(deepseek, "generate_plan_with_tools"))
        self.assertTrue(hasattr(antigravity, "generate_plan_with_tools"))
        self.assertIn(ProviderCapability.TOOL_USE, deepseek.capabilities)
        self.assertIn(ProviderCapability.TOOL_USE, antigravity.capabilities)

    def test_security_planner_cannot_write_or_escape(self):
        """Planning tool registry does not provide mutation operations and blocks path escape."""
        tool_names = [d.name for d in self.registry.definitions()]
        self.assertNotIn("write_file", tool_names)
        self.assertNotIn("delete_file", tool_names)
        self.assertNotIn("patch_file", tool_names)

        res = self.registry.execute(ToolCall("call_sec", "read_file_range", {"path": "../../outside.txt", "start_line": 1, "end_line": 10}))
        self.assertTrue(res.is_error)
        self.assertIn("Access denied", res.output)

    def test_secret_protection_in_planning(self):
        """Planning registry blocks access to secret files."""
        secret_file = self.project_dir / ".env"
        secret_file.write_text("SECRET_KEY=supersecret123", encoding="utf-8")

        res = self.registry.execute(ToolCall("call_sec", "read_file_range", {"path": ".env", "start_line": 1, "end_line": 5}))
        self.assertTrue(res.is_error)
        self.assertIn("Access denied", res.output)

    def test_sandbox_command_safety_in_planning(self):
        """Sandbox command runner blocks shell operators."""
        res = self.registry.execute(ToolCall("call_sand", "run_command_sandbox", {"command": ["cat", "src/main.py", "|", "rm", "-rf", "/"]}))
        self.assertTrue(res.is_error)
        self.assertIn("security violation", res.output.lower())

    def test_orchestrator_stage_3_planning_integration(self):
        """Orchestrator Stage [3/7] executes tool-assisted planning and captures telemetry."""
        config = AgentConfig(project=self.project_dir, provider="mock", max_iterations=1)
        storage = JsonFileStorage(self.project_dir)
        repo_lock = threading.Lock()
        memory_lock = threading.Lock()
        orchestrator = Orchestrator(config, storage=storage, scheduler=None, repo_lock=repo_lock, memory_lock=memory_lock)

        provider = ScriptedPlanningProvider([
            ToolCall("call_1", "read_file_range", {"path": "src/main.py", "start_line": 1, "end_line": 2}),
            Plan(
                objective="Add logging",
                files_likely_to_change=["src/main.py"],
                steps=["Add log statement"],
                validation_strategy=["pytest"],
            ),
        ])

        with patch("local_agent.orchestrator.build_provider", return_value=provider):
            with patch.object(orchestrator, "_validate", return_value=[]):
                report = orchestrator.run("Add logging")

        self.assertTrue(report.completed)
        self.assertTrue(len(report.tool_metrics) >= 1)
        planning_metrics = report.tool_metrics[0]
        self.assertEqual(planning_metrics.total_calls, 1)
        self.assertEqual(planning_metrics.calls_by_tool.get("read_file_range"), 1)

    def test_malformed_tool_call_raises_clean_provider_error(self):
        """Provider raising malformed JSON in tool call parameters triggers ProviderError."""
        cfg = AgentConfig(project=self.project_dir, api_key="fake-key", model="gpt-4.1-mini")
        provider = OpenAIProvider(cfg)
        mock_response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_bad",
                                "type": "function",
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
        with patch.object(provider, "_request_json_api", return_value=mock_response):
            with self.assertRaises(ProviderError):
                provider.generate_plan_with_tools("Task", self.context, tools=self.registry.definitions())

    def test_unknown_tool_call_handled_safely(self):
        """Unknown tool call requested by model returns structured error result."""
        res = self.registry.execute(ToolCall("call_bad", "non_existent_tool", {"arg": "val"}))
        self.assertTrue(res.is_error)
        self.assertIn("Unknown tool", res.output)

    def test_checkpoint_compatibility_with_planning_metrics(self):
        """Checkpoint preserves and reloads planning tool metrics cleanly."""
        storage = JsonFileStorage(self.project_dir)
        task = Task(
            task_id="task-chk-plan",
            objective="Test planning checkpoint",
            status=TaskStatus.RUNNING,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        report = RunReport(self.context, task.task_id)
        report.tool_metrics.append(ToolExecutionMetrics(
            total_calls=2,
            steps_used=2,
            total_output_bytes=350,
            calls_by_tool={"find_files": 1, "read_file_range": 1},
        ))

        chk = Checkpoint(
            checkpoint_id="chk-plan-1",
            task_id=task.task_id,
            subtask_id="",
            timestamp=datetime.now(timezone.utc),
            current_state_description="Planning complete",
            files_changed=["src/main.py"],
            repository_diff="",
            validation_state={"last_executions": [], "last_failures": []},
            continuation_context={
                "tool_metrics": [m.to_dict() for m in report.tool_metrics],
            },
        )
        storage.save_checkpoint(chk)
        loaded = storage.load_checkpoint(chk.checkpoint_id)

        self.assertEqual(loaded.checkpoint_id, "chk-plan-1")
        self.assertIn("tool_metrics", loaded.continuation_context)
        self.assertEqual(len(loaded.continuation_context["tool_metrics"]), 1)
        self.assertEqual(loaded.continuation_context["tool_metrics"][0]["total_calls"], 2)


if __name__ == "__main__":
    unittest.main()

