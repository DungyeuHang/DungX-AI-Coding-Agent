from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_agent.cli import main as cli_main
from local_agent.config import AgentConfig
from local_agent.credentials import MockCredentialStore
from local_agent.models import (
    CIFailureContext,
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


class CIRepairTestProvider(AIProvider):
    def __init__(self, always_fail: bool = False):
        self.always_fail = always_fail

    @property
    def capabilities(self) -> set[ProviderCapability]:
        return {ProviderCapability.PLANNING, ProviderCapability.IMPLEMENTATION, ProviderCapability.REPAIR, ProviderCapability.REVIEW}

    def generate_plan(self, task: str, context: ProjectContext) -> Plan:
        return TaskPlan(objective=task, subtasks=[Subtask(subtask_id="1", title="Fix CI failure", goal=task)])

    def generate_code(self, task: str, plan: Plan, context: ProjectContext, failure: FailureAnalysis | None = None, review: ReviewResult | None = None) -> list[FileOperation]:
        if failure: # This is a repair attempt
            if self.always_fail:
                return [FileOperation("modify", "app.py", content="raise ValueError('Still broken')")]
            else: # The repair is correct
                return [FileOperation("modify", "app.py", content="def main():\n    return 'fixed'")]
        return [] # Should not be called without a failure context in this test

    def analyze_failure(self, execution, diff, context, plan):
        return FailureAnalysis("Test failed", ["app.py"], "Fix the code.")

    def review_changes(self, task, plan, diff, context):
        return ReviewResult("APPROVED", "LGTM", [])


class Phase323_CI_IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.storage = JsonFileStorage(self.root / ".agent_data")
        self.credential_store = MockCredentialStore()
        self.failure_file = self.root / "failure.json"

        self.mock_build_provider_patcher = mock.patch("local_agent.scheduler.build_provider")
        self.mock_build_provider = self.mock_build_provider_patcher.start()
        self.mock_planner_build_provider_patcher = mock.patch("local_agent.planner.build_provider")
        self.mock_planner_build_provider = self.mock_planner_build_provider_patcher.start()

    def tearDown(self):
        self.mock_build_provider_patcher.stop()
        self.mock_planner_build_provider_patcher.stop()
        import shutil
        shutil.rmtree(self.root)

    def test_ci_repair_command_creates_autonomous_task_with_context(self):
        # Arrange
        failure_data = {"failed_command": "pytest", "exit_code": 1, "stdout": "...", "stderr": "AssertionError"}
        self.failure_file.write_text(json.dumps(failure_data))

        # Act
        with mock.patch("sys.stdout", new=__import__("io").StringIO()):
            cli_main(["ci-repair", "--project", str(self.root), "--failure-file", str(self.failure_file)])

        # Assert
        tasks = self.storage.list_tasks()
        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertTrue(task.autonomous)
        self.assertIsInstance(task.initial_failure_context, CIFailureContext)
        self.assertEqual(task.initial_failure_context.failed_command, "pytest")
        self.assertEqual(task.initial_failure_context.exit_code, 1)

    def test_ci_repair_handles_invalid_json(self):
        self.failure_file.write_text("{not_json")
        with mock.patch("sys.stderr", new=__import__("io").StringIO()) as fake_err:
            result = cli_main(["ci-repair", "--project", str(self.root), "--failure-file", str(self.failure_file)])
        self.assertEqual(result, 1)
        self.assertIn("Invalid or missing failure file", fake_err.getvalue())

    @mock.patch("local_agent.orchestrator.build_provider")
    def test_e2e_ci_repair_success(self, mock_orch_build_provider):
        # Arrange
        (self.root / "app.py").write_text("def main():\n    return 'broken'")
        failure_data = {"failed_command": "pytest", "exit_code": 1, "stdout": "...", "stderr": "AssertionError"}
        self.failure_file.write_text(json.dumps(failure_data))

        self.mock_build_provider.return_value = CIRepairTestProvider()
        self.mock_planner_build_provider.return_value = CIRepairTestProvider()
        mock_orch_build_provider.return_value = CIRepairTestProvider()

        provider_configs = [ProviderConfig(provider_id="mock", priority=10, enabled=True)]
        self.storage.save_provider_configs(provider_configs)
        self.credential_store.save("dungx-ai-coding-agent", "mock", "mock-key")

        # Act
        with mock.patch("sys.stdout", new=__import__("io").StringIO()):
            cli_main(["ci-repair", "--project", str(self.root), "--failure-file", str(self.failure_file)])
        task = self.storage.list_tasks()[0]

        config = AgentConfig.from_environment(self.root, validation_commands=["python -c \"import app; assert app.main() == 'fixed'\""])
        scheduler = Scheduler(config, self.storage, self.credential_store)
        for _ in range(5): # Run scheduler a few times to complete the task
            scheduler.run_once()

        # Assert
        final_task = self.storage.load_task(task.task_id)
        self.assertEqual(final_task.status, TaskStatus.COMPLETED)
        self.assertEqual((self.root / "app.py").read_text(), "def main():\n    return 'fixed'")