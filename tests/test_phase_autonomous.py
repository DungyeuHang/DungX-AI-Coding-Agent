from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_agent.cli import main as cli_main
from local_agent.config import AgentConfig
from local_agent.credentials import MockCredentialStore
from local_agent.models import (
    FailureAnalysis,
    FileOperation,
    Plan,
    ProjectContext,
    ProviderCapability,
    ProviderConfig,
    ReviewResult,
    Subtask,
    TaskPlan,
    TaskStatus,
)
from local_agent.providers import AIProvider
from local_agent.scheduler import Scheduler
from local_agent.storage import JsonFileStorage


class AutonomousTestProvider(AIProvider):
    """A deterministic provider for end-to-end autonomous tests."""

    def __init__(self, initial_code_is_buggy: bool = False, always_fail: bool = False):
        self.initial_code_is_buggy = initial_code_is_buggy
        self.always_fail = always_fail
        self.repair_attempts = 0

    @property
    def capabilities(self) -> set[ProviderCapability]:
        return {
            ProviderCapability.PLANNING,
            ProviderCapability.IMPLEMENTATION,
            ProviderCapability.REPAIR,
            ProviderCapability.REVIEW,
        }

    def generate_plan(self, task: str, context: ProjectContext) -> Plan:
        s1 = Subtask(subtask_id="1", title="Modify app", goal="Modify app.py to print 'hello world'")
        return TaskPlan(objective=task, subtasks=[s1])

    def generate_code(self, task: str, plan: Plan, context: ProjectContext, failure: FailureAnalysis | None = None, review: ReviewResult | None = None) -> list[FileOperation]:
        if failure:  # This is a repair attempt
            self.repair_attempts += 1
            if self.always_fail:
                return [FileOperation("modify", "app.py", content='def main():\n    print("buggy again")')]
            else: # The repair is correct
                return [
                    FileOperation("modify", "app.py", content='def main():\n    print("hello world")'),
                    FileOperation("modify", "test_app.py", content='import unittest\nfrom unittest.mock import patch\nfrom io import StringIO\nfrom app import main\n\nclass TestApp(unittest.TestCase):\n    def test_main(self):\n        with patch("sys.stdout", new=StringIO()) as fake_out:\n            main()\n            self.assertEqual(fake_out.getvalue().strip(), "hello world")\n'),
                ]

        # Initial implementation
        if self.initial_code_is_buggy:
            return [FileOperation("modify", "app.py", content='def main():\n    print("buggy")')]
        else:
            return [
                FileOperation("modify", "app.py", content='def main():\n    print("hello world")'),
                FileOperation("create", "test_app.py", content='import unittest\nfrom unittest.mock import patch\nfrom io import StringIO\nfrom app import main\n\nclass TestApp(unittest.TestCase):\n    def test_main(self):\n        with patch("sys.stdout", new=StringIO()) as fake_out:\n            main()\n            self.assertEqual(fake_out.getvalue().strip(), "hello world")\n'),
            ]

    def analyze_failure(self, execution, diff, context, plan):
        return FailureAnalysis("Test failed as expected", ["app.py"], "Fix the print statement.")

    def review_changes(self, task, plan, diff, context):
        return ReviewResult("APPROVED", "LGTM", [])


class AutonomousModeE2ETests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        (self.root / "app.py").write_text('def main():\n    print("hello")\n')
        self.storage = JsonFileStorage(self.root / ".agent_data")
        self.credential_store = MockCredentialStore()
        
        # Mock build_provider to return our deterministic provider
        self.mock_provider_patcher = mock.patch("local_agent.scheduler.build_provider")
        self.mock_build_provider = self.mock_provider_patcher.start()
        
        self.mock_planner_provider_patcher = mock.patch("local_agent.planner.build_provider")
        self.mock_planner_build_provider = self.mock_planner_provider_patcher.start()

    def tearDown(self):
        self.mock_provider_patcher.stop()
        self.mock_planner_provider_patcher.stop()
        import shutil
        shutil.rmtree(self.root)

    def _run_scheduler_until_terminal(self, scheduler: Scheduler, task_id: str, max_runs=10) -> "Task":
        for _ in range(max_runs):
            scheduler.run_once()
            task = self.storage.load_task(task_id)
            if task.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.PAUSED, TaskStatus.REJECTED}:
                return task
        self.fail(f"Task did not reach a terminal state after {max_runs} scheduler runs.")

    def test_e2e_successful_autonomous_task(self):
        # Arrange
        self.mock_build_provider.return_value = AutonomousTestProvider()
        self.mock_planner_build_provider.return_value = AutonomousTestProvider()
        
        provider_configs = [ProviderConfig(provider_id="mock", priority=10, enabled=True)]
        self.storage.save_provider_configs(provider_configs)
        self.credential_store.save("dungx-ai-coding-agent", "mock", "mock-key")

        # Act: Create autonomous task via CLI
        with mock.patch("sys.stdout", new=__import__("io").StringIO()):
            cli_main(["create-task", "--project", str(self.root), "--autonomous", "Update message"])
        
        task = self.storage.list_tasks()[0]
        self.assertTrue(task.autonomous)

        # Act: Run scheduler until task is terminal
        config = AgentConfig.from_environment(self.root, approval_mode="plan_review", validation_commands=["python -m unittest discover"])
        scheduler = Scheduler(config, self.storage, self.credential_store)
        final_task = self._run_scheduler_until_terminal(scheduler, task.task_id)

        # Assert
        self.assertEqual(final_task.status, TaskStatus.COMPLETED)
        final_content = (self.root / "app.py").read_text()
        self.assertIn("hello world", final_content)
        self.assertTrue((self.root / "test_app.py").exists())

    def test_e2e_autonomous_repair_cycle(self):
        # Arrange
        provider = AutonomousTestProvider(initial_code_is_buggy=True)
        self.mock_build_provider.return_value = provider
        self.mock_planner_build_provider.return_value = provider
        
        provider_configs = [ProviderConfig(provider_id="mock", priority=10, enabled=True)]
        self.storage.save_provider_configs(provider_configs)
        self.credential_store.save("dungx-ai-coding-agent", "mock", "mock-key")

        # Act
        with mock.patch("sys.stdout", new=__import__("io").StringIO()):
            cli_main(["create-task", "--project", str(self.root), "--autonomous", "Update message"])
        task = self.storage.list_tasks()[0]

        config = AgentConfig.from_environment(self.root, approval_mode="plan_review", validation_commands=["python -m unittest discover"])
        scheduler = Scheduler(config, self.storage, self.credential_store)
        final_task = self._run_scheduler_until_terminal(scheduler, task.task_id)

        # Assert
        self.assertEqual(final_task.status, TaskStatus.COMPLETED)
        final_content = (self.root / "app.py").read_text()
        self.assertIn("hello world", final_content)
        self.assertEqual(provider.repair_attempts, 1)

    def test_e2e_autonomous_failure_termination(self):
        # Arrange
        provider = AutonomousTestProvider(always_fail=True)
        self.mock_build_provider.return_value = provider
        self.mock_planner_build_provider.return_value = provider
        
        provider_configs = [ProviderConfig(provider_id="mock", priority=10, enabled=True)]
        self.storage.save_provider_configs(provider_configs)
        self.credential_store.save("dungx-ai-coding-agent", "mock", "mock-key")

        # Act
        with mock.patch("sys.stdout", new=__import__("io").StringIO()):
            cli_main(["create-task", "--project", str(self.root), "--autonomous", "Update message"])
        task = self.storage.list_tasks()[0]

        config = AgentConfig.from_environment(self.root, max_iterations=2, approval_mode="plan_review", validation_commands=["python -m unittest discover"])
        scheduler = Scheduler(config, self.storage, self.credential_store)
        final_task = self._run_scheduler_until_terminal(scheduler, task.task_id)

        # Assert
        self.assertEqual(final_task.status, TaskStatus.FAILED)
        self.assertGreaterEqual(provider.repair_attempts, 1)

    def test_non_autonomous_task_still_requires_approval(self):
        # Arrange
        self.mock_build_provider.return_value = AutonomousTestProvider()
        self.mock_planner_build_provider.return_value = AutonomousTestProvider()
        
        provider_configs = [ProviderConfig(provider_id="mock", priority=10, enabled=True)]
        self.storage.save_provider_configs(provider_configs)
        self.credential_store.save("dungx-ai-coding-agent", "mock", "mock-key")

        # Act: Create a non-autonomous task
        with mock.patch("sys.stdout", new=__import__("io").StringIO()):
            cli_main(["create-task", "--project", str(self.root), "Update message"])
        task = self.storage.list_tasks()[0]
        self.assertFalse(task.autonomous)

        config = AgentConfig.from_environment(self.root, approval_mode="plan_review")
        scheduler = Scheduler(config, self.storage, self.credential_store)
        scheduler.run_once() # Should generate a plan and stop

        # Assert
        final_task = self.storage.load_task(task.task_id)
        self.assertEqual(final_task.status, TaskStatus.PLAN_REVIEW)