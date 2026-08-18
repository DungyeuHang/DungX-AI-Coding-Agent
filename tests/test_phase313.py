from __future__ import annotations

import datetime
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_agent.config import AgentConfig
from local_agent.models import (
    CommandSpec, ExecutionResult, FailureAnalysis, FileOperation, Plan,
    ProjectContext, ReviewResult, Subtask, SubtaskStatus, Task, TaskPlan,
    TaskStatus, ValidationPlan,
)
from local_agent.orchestrator import Orchestrator
from local_agent.providers import AIProvider, MockProvider
from local_agent.storage import JsonFileStorage


class MockProviderWithDiagnostics(MockProvider):
    def __init__(self, command_to_select: CommandSpec | None = None, fail_repair: bool = False):
        super().__init__()
        self.command_to_select = command_to_select
        self.select_diagnostic_command_calls = []
        self.generate_code_calls = []
        self.fail_repair = fail_repair

    def select_diagnostic_command(self, task: str, plan: Plan, context: ProjectContext, primary_failure: ExecutionResult, available_commands: list[CommandSpec]) -> CommandSpec | None:
        self.select_diagnostic_command_calls.append(available_commands)
        if self.command_to_select and self.command_to_select in available_commands:
            return self.command_to_select
        return None

    def generate_code(self, task: str, plan: Plan, context: ProjectContext, failure: FailureAnalysis | None = None, review: ReviewResult | None = None) -> list[FileOperation]:
        self.generate_code_calls.append({"failure": failure, "review": review})
        if self.fail_repair and failure:
            # Still return an empty list, but we can check that the failure was received
            return []
        if failure: # This is a repair attempt
            return [FileOperation("modify", "src/app.py", content="# repaired")]
        return [FileOperation("modify", "src/app.py", content="# initial change")]


class Phase313Tests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.storage = JsonFileStorage(self.root / ".agent_data")
        (self.root / "src").mkdir()
        (self.root / "src" / "app.py").write_text("# original content")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root)

    def _get_config(self, **overrides) -> AgentConfig:
        return AgentConfig.from_environment(self.root, **overrides)

    def _create_task_with_plan(self) -> Task:
        now = datetime.datetime.now(datetime.timezone.utc)
        subtask = Subtask(subtask_id="sub1", title="Test Subtask", goal="Goal", status=SubtaskStatus.PENDING, created_at=now, updated_at=now)
        plan = TaskPlan(objective="Test Task", subtasks=[subtask])
        task = Task(task_id="task1", objective="Test Task", status=TaskStatus.PENDING, created_at=now, updated_at=now, plan=plan)
        self.storage.save_task(task)
        return task

    @mock.patch("local_agent.orchestrator.CommandRunner")
    def test_secondary_command_is_selected_and_run(self, MockCommandRunner):
        # Arrange
        primary_fail = ExecutionResult("primary", 1, stderr="primary fail")
        secondary_success = ExecutionResult("secondary", 0, stdout="secondary ok")
        MockCommandRunner.return_value.run.side_effect = [primary_fail, secondary_success]

        secondary_cmd = CommandSpec("secondary_cmd", ("npm", "run", "test:secondary"))
        validation_plan = ValidationPlan(commands=[], primary_commands=[CommandSpec("primary_cmd", ("npm", "test"))], secondary_commands=[secondary_cmd], skipped_commands=[], reasons=[], risk_level="low")
        
        provider = MockProviderWithDiagnostics(command_to_select=secondary_cmd, fail_repair=True)
        config = self._get_config(max_iterations=2)
        orchestrator = Orchestrator(config, provider, self.storage)

        with mock.patch.object(orchestrator, "validation_intelligence") as mock_vi:
            mock_vi.select_commands.return_value = validation_plan

            # Act
            task = self._create_task_with_plan()
            orchestrator.run(task, "sub1")

        # Assert
        self.assertEqual(len(provider.select_diagnostic_command_calls), 1)
        self.assertEqual(MockCommandRunner.return_value.run.call_count, 2) # Primary + Secondary
        self.assertEqual(MockCommandRunner.return_value.run.call_args_list[1].args[0], secondary_cmd)
        
        # Check that evidence was passed to the repair call
        self.assertEqual(len(provider.generate_code_calls), 2)
        repair_call = provider.generate_code_calls[1]
        self.assertIsNotNone(repair_call["failure"])
        self.assertEqual(len(repair_call["failure"].diagnostic_evidence), 1)
        self.assertEqual(repair_call["failure"].diagnostic_evidence[0].command, "secondary")

    def test_secondary_validation_is_bounded_by_config(self):
        # Arrange
        MockCommandRunner.return_value.run.side_effect = [
            ExecutionResult("primary", 1), # Primary fail
            ExecutionResult("secondary1", 0), # Diagnostic 1
            ExecutionResult("secondary2", 0), # Should not be called
        ]

        cmd1 = CommandSpec("cmd1", ("npm", "run", "diag1"))
        cmd2 = CommandSpec("cmd2", ("npm", "run", "diag2"))
        validation_plan = ValidationPlan(commands=[], primary_commands=[CommandSpec("primary", ("npm", "test"))], secondary_commands=[cmd1, cmd2], skipped_commands=[], reasons=[], risk_level="low")
        
        provider = MockProviderWithDiagnostics(command_to_select=cmd1)
        config = self._get_config(max_iterations=2, max_secondary_validations_per_iteration=1) # Set budget to 1
        orchestrator = Orchestrator(config, provider, self.storage)

        with mock.patch.object(orchestrator, "validation_intelligence") as mock_vi, \
             mock.patch("local_agent.orchestrator.CommandRunner") as MockCommandRunner:
            MockCommandRunner.return_value.run.side_effect = [ExecutionResult("primary", 1), ExecutionResult("secondary1", 0)]
            mock_vi.select_commands.return_value = validation_plan

            # Act
            task = self._create_task_with_plan()
            orchestrator.run(task, "sub1")

        # Assert
        self.assertEqual(MockCommandRunner.return_value.run.call_count, 2) # Primary + 1 Diagnostic
        self.assertEqual(len(provider.select_diagnostic_command_calls), 1) # Only asked once

    def test_provider_hallucinated_command_is_rejected(self):
        # Arrange
        primary_fail = ExecutionResult("primary", 1)
        MockCommandRunner.return_value.run.return_value = primary_fail

        secondary_cmd = CommandSpec("secondary_cmd", ("npm", "run", "test:secondary"))
        hallucinated_cmd = CommandSpec("hallucinated", ("rm", "-rf", "/"))
        validation_plan = ValidationPlan(commands=[], primary_commands=[CommandSpec("primary", ("npm", "test"))], secondary_commands=[secondary_cmd], skipped_commands=[], reasons=[], risk_level="low")
        
        provider = MockProviderWithDiagnostics(command_to_select=hallucinated_cmd)
        config = self._get_config(max_iterations=2)
        orchestrator = Orchestrator(config, provider, self.storage)

        with mock.patch.object(orchestrator, "validation_intelligence") as mock_vi, \
             mock.patch("local_agent.orchestrator.CommandRunner") as MockCommandRunner:
            MockCommandRunner.return_value.run.return_value = primary_fail
            mock_vi.select_commands.return_value = validation_plan

            # Act
            task = self._create_task_with_plan()
            orchestrator.run(task, "sub1")

        # Assert
        self.assertEqual(MockCommandRunner.return_value.run.call_count, 1) # Only primary validation runs
        self.assertEqual(len(provider.select_diagnostic_command_calls), 1)

    def test_output_is_truncated(self):
        # Arrange
        long_output = "a" * 5000
        primary_fail = ExecutionResult("primary", 1)
        secondary_result = ExecutionResult("secondary", 0, stdout=long_output, stderr=long_output)
        
        secondary_cmd = CommandSpec("secondary_cmd", ("npm", "run", "test:secondary"))
        validation_plan = ValidationPlan(commands=[], primary_commands=[CommandSpec("primary", ("npm", "test"))], secondary_commands=[secondary_cmd], skipped_commands=[], reasons=[], risk_level="low")
        
        provider = MockProviderWithDiagnostics(command_to_select=secondary_cmd, fail_repair=True)
        config = self._get_config(max_iterations=2, max_diagnostic_output_bytes=100)
        orchestrator = Orchestrator(config, provider, self.storage)

        with mock.patch.object(orchestrator, "validation_intelligence") as mock_vi, \
             mock.patch("local_agent.orchestrator.CommandRunner") as MockCommandRunner:
            MockCommandRunner.return_value.run.side_effect = [primary_fail, secondary_result]
            mock_vi.select_commands.return_value = validation_plan

            # Act
            task = self._create_task_with_plan()
            orchestrator.run(task, "sub1")

        # Assert
        repair_call = provider.generate_code_calls[1]
        evidence = repair_call["failure"].diagnostic_evidence[0]
        self.assertLess(len(evidence.stdout), 200)
        self.assertLess(len(evidence.stderr), 200)
        self.assertIn("...[truncated]...", evidence.stdout)
        self.assertIn("...[truncated]...", evidence.stderr)

    def test_pause_resume_preserves_diagnostic_state(self):
        # Arrange
        primary_fail = ExecutionResult("primary", 1)
        secondary_success = ExecutionResult("secondary", 0, stdout="secondary ok")
        secondary_cmd = CommandSpec("secondary_cmd", ("npm", "run", "test:secondary"))
        validation_plan = ValidationPlan(commands=[], primary_commands=[CommandSpec("primary", ("npm", "test"))], secondary_commands=[secondary_cmd], skipped_commands=[], reasons=[], risk_level="low")
        
        # Run 1: Fail primary, run diagnostic, then provider fails with QuotaExceededError
        provider1 = MockProviderWithDiagnostics(command_to_select=secondary_cmd)
        config = self._get_config(max_iterations=2)
        orchestrator1 = Orchestrator(config, provider1, self.storage)
        with mock.patch.object(orchestrator1, "validation_intelligence") as mock_vi, \
             mock.patch("local_agent.orchestrator.CommandRunner") as MockCommandRunner:
            mock_vi.select_commands.return_value = validation_plan
            MockCommandRunner.return_value.run.side_effect = [primary_fail, secondary_success]
            # The second call to generate_code (the repair) will fail
            provider1.generate_code = mock.Mock(side_effect=[[FileOperation("modify", "src/app.py")], QuotaExceededError("quota fail")])

            task = self._create_task_with_plan()
            orchestrator1.run(task, "sub1")

        # Assert after Run 1
        task_after_pause = self.storage.load_task(task.task_id)
        self.assertEqual(task_after_pause.status, TaskStatus.PAUSED)
        subtask = task_after_pause.plan.subtasks[0]
        self.assertEqual(subtask.status, SubtaskStatus.PAUSED)
        checkpoint = self.storage.load_checkpoint(subtask.latest_checkpoint_id)
        self.assertIn("executed_diagnostic_names_this_iteration", checkpoint.continuation_context)
        self.assertEqual(checkpoint.continuation_context["executed_diagnostic_names_this_iteration"], ["secondary_cmd"])

        # Run 2: Resume. The diagnostic should not run again.
        provider2 = MockProviderWithDiagnostics(command_to_select=secondary_cmd)
        orchestrator2 = Orchestrator(config, provider2, self.storage)
        with mock.patch.object(orchestrator2, "validation_intelligence") as mock_vi, \
             mock.patch("local_agent.orchestrator.CommandRunner") as MockCommandRunner:
            mock_vi.select_commands.return_value = validation_plan
            # On resume, only the primary validation of the *new* patch runs.
            MockCommandRunner.return_value.run.return_value = ExecutionResult("primary", 0)

            orchestrator2.run(task_after_pause, "sub1")

        # Assert after Run 2
        # select_diagnostic_command should not have been called because the loop was resumed *after* the diagnostic phase
        self.assertEqual(len(provider2.select_diagnostic_command_calls), 0)
        final_task = self.storage.load_task(task.task_id)
        self.assertEqual(final_task.status, TaskStatus.COMPLETED)