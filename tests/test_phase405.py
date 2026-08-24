import datetime
import tempfile
import threading
from pathlib import Path
from typing import Any
import unittest

from local_agent.config import AgentConfig
from local_agent.models import (
    CIFailureContext,
    Checkpoint,
    ExecutionResult,
    FailureAnalysis,
    FileOperation,
    Plan,
    ProjectContext,
    ProviderCapability,
    ProviderError,
    RateLimitError,
    ReviewResult,
    Subtask,
    SubtaskStatus,
    Task,
    TaskStatus,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from local_agent.orchestrator import Orchestrator
from local_agent.providers import AIProvider, MockProvider
from local_agent.storage import JsonFileStorage


class ScriptedToolProvider(AIProvider):
    """Provider double with TOOL_USE capability returning scripted responses."""

    def __init__(self, responses: list[Any]):
        self.responses = list(responses)
        self.tool_calls_made: list[ToolCall] = []
        self.generate_code_called = False

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
        return Plan(objective=task, steps=["Inspect", "Modify"], files_likely_to_change=["main.py"])

    def generate_code(self, task: str, plan: Plan, context: ProjectContext, failure=None, review=None) -> list[FileOperation]:
        self.generate_code_called = True
        return [FileOperation(action="modify", path="main.py", content="print('from 1-shot')\n")]

    def generate_code_with_tools(
        self,
        task: str,
        plan: Plan,
        context: ProjectContext,
        tools: list[ToolDefinition],
        tool_history: list[tuple[ToolCall, ToolResult]] | None = None,
        failure=None,
        review=None,
    ) -> ToolCall | list[FileOperation]:
        if not self.responses:
            raise RuntimeError("ScriptedToolProvider ran out of responses")
        resp = self.responses.pop(0)
        if isinstance(resp, ToolCall):
            self.tool_calls_made.append(resp)
        return resp

    def review_changes(self, task: str, plan: Plan, diff: str, context: ProjectContext) -> ReviewResult:
        return ReviewResult(verdict="APPROVED", summary="Looks good", findings=[])


class ScriptedOneShotProvider(AIProvider):
    """Provider double WITHOUT TOOL_USE capability."""

    def __init__(self, file_operations: list[FileOperation]):
        self.file_operations = file_operations
        self.generate_code_called = False
        self.generate_code_with_tools_called = False

    @property
    def capabilities(self) -> set[ProviderCapability]:
        return {
            ProviderCapability.PLANNING,
            ProviderCapability.IMPLEMENTATION,
            ProviderCapability.REPAIR,
            ProviderCapability.REVIEW,
        }

    def generate_plan(self, task: str, context: ProjectContext) -> Plan:
        return Plan(objective=task, steps=["Inspect", "Modify"], files_likely_to_change=["main.py"])

    def generate_code(self, task: str, plan: Plan, context: ProjectContext, failure=None, review=None) -> list[FileOperation]:
        self.generate_code_called = True
        return self.file_operations

    def review_changes(self, task: str, plan: Plan, diff: str, context: ProjectContext) -> ReviewResult:
        return ReviewResult(verdict="APPROVED", summary="Looks good", findings=[])


class Phase405OrchestratorIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_dir = Path(self.temp_dir.name).resolve()

        # Initialize test repository
        (self.project_dir / "main.py").write_text("print('original')\n", encoding="utf-8")

        self.config = AgentConfig(
            project=self.project_dir,
            provider="mock",
            max_iterations=2,
            approval="never",
        )
        self.storage = JsonFileStorage(self.project_dir / ".agent_data")
        self.repo_lock = threading.Lock()
        self.memory_lock = threading.Lock()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _make_orchestrator(self, provider_instance: AIProvider) -> Orchestrator:
        class DummyScheduler:
            def __init__(self, prov_inst):
                self.provider = "mock"
                self._prov_inst = prov_inst

            def _select_providers(self, task, capabilities):
                return [self]

            def _build_provider_instance(self, provider_name):
                return self._prov_inst

        return Orchestrator(
            self.config,
            storage=self.storage,
            scheduler=DummyScheduler(provider_instance),
            repo_lock=self.repo_lock,
            memory_lock=self.memory_lock,
        )

    def _make_task(self, task_id: str, objective: str, **kwargs) -> Task:
        now = datetime.datetime.now(datetime.timezone.utc)
        return Task(
            task_id=task_id,
            objective=objective,
            created_at=now,
            updated_at=now,
            **kwargs,
        )

    # -------------------------------------------------------------------------
    # 1. IMPLEMENTATION + TOOL_USE
    # -------------------------------------------------------------------------

    def test_implementation_with_tool_use(self):
        tool_call = ToolCall("c1", "read_file_range", {"path": "main.py", "start_line": 1, "end_line": 1})
        final_ops = [FileOperation(action="modify", path="main.py", content="print('tool modified')\n")]

        provider = ScriptedToolProvider([tool_call, final_ops])
        orchestrator = self._make_orchestrator(provider)

        task = self._make_task("t-1", "Update main.py", status=TaskStatus.PENDING)
        report = orchestrator.run(task)

        self.assertEqual(len(provider.tool_calls_made), 1)
        self.assertEqual(provider.tool_calls_made[0].call_id, "c1")
        self.assertFalse(provider.generate_code_called)
        self.assertEqual((self.project_dir / "main.py").read_text(encoding="utf-8"), "print('tool modified')\n")
        self.assertTrue(report.completed)

    # -------------------------------------------------------------------------
    # 2. IMPLEMENTATION without TOOL_USE
    # -------------------------------------------------------------------------

    def test_implementation_without_tool_use_uses_1shot(self):
        final_ops = [FileOperation(action="modify", path="main.py", content="print('1-shot modified')\n")]
        provider = ScriptedOneShotProvider(final_ops)
        orchestrator = self._make_orchestrator(provider)

        task = self._make_task("t-2", "Update main.py without tools", status=TaskStatus.PENDING)
        report = orchestrator.run(task)

        self.assertTrue(provider.generate_code_called)
        self.assertEqual((self.project_dir / "main.py").read_text(encoding="utf-8"), "print('1-shot modified')\n")
        self.assertTrue(report.completed)

    # -------------------------------------------------------------------------
    # 3. REPAIR + TOOL_USE
    # -------------------------------------------------------------------------

    def test_repair_with_tool_use(self):
        tool_call = ToolCall("c_repair", "read_file_range", {"path": "main.py", "start_line": 1, "end_line": 2})
        final_ops = [FileOperation(action="modify", path="main.py", content="print('repair success')\n")]

        provider = ScriptedToolProvider([tool_call, final_ops])
        orchestrator = self._make_orchestrator(provider)

        task = self._make_task(
            "t-3",
            "Repair main.py",
            status=TaskStatus.PENDING,
            initial_failure_context=CIFailureContext(failed_command="pytest", exit_code=1, stdout="", stderr="AssertionError"),
        )
        report = orchestrator.run(task)

        self.assertEqual(len(provider.tool_calls_made), 1)
        self.assertEqual(provider.tool_calls_made[0].call_id, "c_repair")
        self.assertEqual((self.project_dir / "main.py").read_text(encoding="utf-8"), "print('repair success')\n")
        self.assertTrue(report.completed)

    # -------------------------------------------------------------------------
    # 4. PLANNING & 5. REVIEW Isolation
    # -------------------------------------------------------------------------

    def test_planning_and_review_do_not_invoke_tool_engine(self):
        final_ops = [FileOperation(action="modify", path="main.py", content="print('clean')\n")]
        # Only 1 response provided: the final operations
        provider = ScriptedToolProvider([final_ops])
        orchestrator = self._make_orchestrator(provider)

        task = self._make_task("t-4", "Planning isolation test", status=TaskStatus.PENDING)
        report = orchestrator.run(task)

        self.assertTrue(report.completed)
        self.assertEqual(len(provider.tool_calls_made), 0)

    # -------------------------------------------------------------------------
    # 6. Tool History Persistence & 7. Resume
    # -------------------------------------------------------------------------

    def test_tool_history_persisted_and_restored_on_resume(self):
        # Create an existing checkpoint with prior tool history
        now = datetime.datetime.now(datetime.timezone.utc)
        prior_call = ToolCall("prior-1", "read_file_range", {"path": "main.py"})
        prior_result = ToolResult("prior-1", "read_file_range", "1: print('original')")
        prior_history = [(prior_call, prior_result)]

        subtask = Subtask(subtask_id="sub-1", goal="Modify main.py", status=SubtaskStatus.PAUSED, latest_checkpoint_id="chk-1")
        task = self._make_task(
            "t-5",
            "Resume task",
            status=TaskStatus.PAUSED,
            latest_checkpoint_id="chk-1",
        )
        task.plan = None

        from local_agent.tool_engine import history_to_dict
        checkpoint = Checkpoint(
            checkpoint_id="chk-1",
            task_id="t-5",
            subtask_id="sub-1",
            timestamp=now,
            current_state_description="Paused with exploration",
            files_changed=[],
            repository_diff="",
            validation_state={},
            last_provider_result=None,
            next_recommended_action="Resume",
            continuation_context={
                "tool_history": history_to_dict(prior_history),
            },
        )
        self.storage.save_checkpoint(checkpoint)
        self.storage.save_task(task)

        # Provider emits next tool call then final operations
        next_call = ToolCall("next-2", "find_files", {"pattern": "*.py"})
        final_ops = [FileOperation(action="modify", path="main.py", content="print('resumed')\n")]
        provider = ScriptedToolProvider([next_call, final_ops])
        orchestrator = self._make_orchestrator(provider)

        report = orchestrator.run(task, subtask_id="sub-1")

        self.assertTrue(report.completed)
        # prior-1 was not re-executed; only next-2 was called
        self.assertEqual(len(provider.tool_calls_made), 1)
        self.assertEqual(provider.tool_calls_made[0].call_id, "next-2")

    # -------------------------------------------------------------------------
    # 8. Quota/Rate-Limit Pause Preserves Tool History
    # -------------------------------------------------------------------------

    def test_rate_limit_pause_preserves_tool_history(self):
        class RateLimitedToolProvider(AIProvider):
            def __init__(self):
                self.calls = 0

            @property
            def capabilities(self) -> set[ProviderCapability]:
                return {
                    ProviderCapability.PLANNING,
                    ProviderCapability.IMPLEMENTATION,
                    ProviderCapability.TOOL_USE,
                }

            def generate_plan(self, task: str, context: ProjectContext) -> Plan:
                return Plan(objective=task, steps=["Inspect"], files_likely_to_change=["main.py"])

            def generate_code_with_tools(self, *args, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return ToolCall("call_rl_1", "read_file_range", {"path": "main.py"})
                raise RateLimitError("Rate limit hit during exploration", retry_after_seconds=10)

        provider = RateLimitedToolProvider()
        orchestrator = self._make_orchestrator(provider)

        task = self._make_task("t-6", "Rate limit task", status=TaskStatus.PENDING)
        report = orchestrator.run(task)

        self.assertEqual(task.status, TaskStatus.PAUSED)
        self.assertIsNotNone(task.latest_checkpoint_id)

        checkpoint = self.storage.load_checkpoint(task.latest_checkpoint_id)
        self.assertIsNotNone(checkpoint)
        saved_history = checkpoint.continuation_context.get("tool_history", [])
        self.assertEqual(len(saved_history), 1)
        self.assertEqual(saved_history[0]["call"]["call_id"], "call_rl_1")

    # -------------------------------------------------------------------------
    # 9. ToolEngine Unsuccessful Termination
    # -------------------------------------------------------------------------

    def test_tool_engine_unsuccessful_termination_fails_cleanly(self):
        # Provider that continuously requests tools exceeding max steps
        class EndlessToolProvider(AIProvider):
            @property
            def capabilities(self) -> set[ProviderCapability]:
                return {
                    ProviderCapability.PLANNING,
                    ProviderCapability.IMPLEMENTATION,
                    ProviderCapability.TOOL_USE,
                }

            def generate_plan(self, task: str, context: ProjectContext) -> Plan:
                return Plan(objective=task, steps=["Inspect"], files_likely_to_change=["main.py"])

            def generate_code_with_tools(self, *args, **kwargs):
                return ToolCall("endless", "find_files", {"pattern": "*.py"})

        provider = EndlessToolProvider()
        orchestrator = self._make_orchestrator(provider)

        task = self._make_task("t-7", "Endless exploration", status=TaskStatus.PENDING)
        report = orchestrator.run(task)

        self.assertFalse(report.completed)
        self.assertEqual(task.status, TaskStatus.FAILED)

    # -------------------------------------------------------------------------
    # 10. File Mutation Invariant & 11. MockProvider Compatibility
    # -------------------------------------------------------------------------

    def test_mock_provider_full_compatibility(self):
        orchestrator = self._make_orchestrator(MockProvider())
        task = self._make_task("t-8", "Mock provider test", status=TaskStatus.PENDING)
        report = orchestrator.run(task)
        self.assertIsNotNone(report)

    def test_file_mutation_invariant_routes_through_patch_pipeline(self):
        # Verify that ToolEngine does not directly mutate files, but routes changes through CodingAgent.prepare()
        tool_call = ToolCall("c1", "find_files", {"pattern": "*.py"})
        final_ops = [FileOperation(action="modify", path="main.py", content="print('applied by patch applier')\n")]

        provider = ScriptedToolProvider([tool_call, final_ops])
        orchestrator = self._make_orchestrator(provider)

        task = self._make_task("t-9", "Verify patch applier pipeline", status=TaskStatus.PENDING)
        report = orchestrator.run(task)

        self.assertTrue(report.completed)
        self.assertEqual((self.project_dir / "main.py").read_text(encoding="utf-8"), "print('applied by patch applier')\n")
        self.assertIn("main.py", report.changed_files)

    def test_capability_gating_integrity(self):
        # A provider that defines generate_code_with_tools but does NOT advertise TOOL_USE in capabilities
        class DuckTypeProvider(AIProvider):
            def __init__(self):
                self.one_shot_called = False
                self.tools_called = False

            @property
            def capabilities(self) -> set[ProviderCapability]:
                # Notice: TOOL_USE is NOT present
                return {
                    ProviderCapability.PLANNING,
                    ProviderCapability.IMPLEMENTATION,
                    ProviderCapability.REVIEW,
                }

            def generate_plan(self, task: str, context: ProjectContext) -> Plan:
                return Plan(objective=task, steps=["Plan"], files_likely_to_change=["main.py"])

            def generate_code(self, task: str, plan: Plan, context: ProjectContext, failure=None, review=None) -> list[FileOperation]:
                self.one_shot_called = True
                return [FileOperation(action="modify", path="main.py", content="print('from 1-shot only')\n")]

            def generate_code_with_tools(self, *args, **kwargs):
                self.tools_called = True
                return [FileOperation(action="modify", path="main.py", content="print('should not be called')\n")]

            def review_changes(self, task: str, plan: Plan, diff: str, context: ProjectContext) -> ReviewResult:
                return ReviewResult(verdict="APPROVED", summary="OK", findings=[])

        provider = DuckTypeProvider()
        orchestrator = self._make_orchestrator(provider)

        task = self._make_task("t-10", "Gating test", status=TaskStatus.PENDING)
        report = orchestrator.run(task)

        self.assertTrue(report.completed)
        self.assertTrue(provider.one_shot_called)
        self.assertFalse(provider.tools_called)
        self.assertEqual((self.project_dir / "main.py").read_text(encoding="utf-8"), "print('from 1-shot only')\n")


if __name__ == "__main__":
    unittest.main()
