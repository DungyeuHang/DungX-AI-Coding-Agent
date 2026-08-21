from __future__ import annotations

import datetime
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, Mock, patch

from local_agent.cli import main as cli_main
from local_agent.config import AgentConfig
from local_agent.credentials import MockCredentialStore
from local_agent.models import (
    Plan,
    ProjectContext,
    ProviderAvailability,
    ProviderCapability,
    ProviderConfig,
    RunReport,
    Subtask,
    SubtaskStatus,
    Task,
    TaskPlan,
    TaskStatus,
)
from local_agent.orchestrator import Orchestrator
from local_agent.planner import Planner
from local_agent.providers import AuthenticationError
from local_agent.repository import RepositoryIntelligence
from local_agent.scheduler import Scheduler
from local_agent.storage import JsonFileStorage, TaskStorage


class SchedulerCallSitesTests(unittest.TestCase):
    def test_scheduler_constructs_orchestrator_with_correct_types(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            storage = JsonFileStorage(root / ".agent")
            cred_store = MockCredentialStore()
            cred_store.save("dungx-ai-coding-agent", "mock", "mock-key")

            now = datetime.datetime.now(datetime.timezone.utc)
            subtask = Subtask(subtask_id="st-1", title="test subtask", description="desc", dependencies=[], status=SubtaskStatus.PENDING)
            task = Task(
                task_id="task-1",
                objective="test task",
                status=TaskStatus.PENDING,
                created_at=now,
                updated_at=now,
                plan=TaskPlan(objective="test task", subtasks=[subtask]),
            )
            storage.save_task(task)

            provider_configs = [ProviderConfig(provider_id="mock", priority=10, enabled=True)]
            storage.save_provider_configs(provider_configs)

            config = AgentConfig(project=root, provider="mock")
            scheduler = Scheduler(config, storage, cred_store)

            with patch.object(scheduler, "_select_providers", return_value=[config]), \
                 patch.object(Orchestrator, "__init__", return_value=None) as mock_orch_init, \
                 patch.object(Orchestrator, "run") as mock_orch_run:
                def fake_run(task, subtask_id, progress=None):
                    t = storage.load_task(task.task_id)
                    t.plan.subtasks[0].status = SubtaskStatus.COMPLETED
                    storage.save_task(t)
                    return RunReport(project=ProjectContext(root=root))

                mock_orch_run.side_effect = fake_run

                scheduler.run_once()

                mock_orch_init.assert_called_once()
                args, _ = mock_orch_init.call_args
                self.assertEqual(len(args), 5)
                self.assertIsInstance(args[0], AgentConfig)
                self.assertIsInstance(args[1], TaskStorage)
                self.assertIsInstance(args[2], Scheduler)
                self.assertIsInstance(args[3], type(threading.Lock()))
                self.assertIsInstance(args[4], type(threading.Lock()))

    def test_plan_task_uses_repository_intelligence_not_orchestrator(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            storage = JsonFileStorage(root / ".agent")
            cred_store = MockCredentialStore()
            cred_store.save("dungx-ai-coding-agent", "mock", "mock-key")

            now = datetime.datetime.now(datetime.timezone.utc)
            task = Task(
                task_id="task-1",
                objective="plan task",
                status=TaskStatus.PENDING,
                created_at=now,
                updated_at=now,
            )
            storage.save_task(task)

            config = AgentConfig(project=root, provider="mock")
            scheduler = Scheduler(config, storage, cred_store)

            with patch.object(RepositoryIntelligence, "scan", return_value=ProjectContext(root=root)) as mock_scan, \
                 patch.object(Orchestrator, "__init__") as mock_orch_init, \
                 patch.object(Planner, "create_task_plan", return_value=TaskPlan(objective="plan task", subtasks=[])) as mock_create_plan:
                scheduler._plan_task_with_provider(task, config)
                mock_scan.assert_called_once()
                mock_orch_init.assert_not_called()
                self.assertIsNotNone(task.plan)

    def test_cli_analyze_command_exits_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patch.object(RepositoryIntelligence, "scan", return_value=ProjectContext(root=root)):
                exit_code = cli_main(["analyze", "--project", str(root)])
                self.assertEqual(exit_code, 0)

    def test_cli_context_command_exits_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patch.object(RepositoryIntelligence, "scan", return_value=ProjectContext(root=root)):
                exit_code = cli_main(["context", "--project", str(root), "add multiply"])
                self.assertEqual(exit_code, 0)

    def test_missing_api_key_raises_authentication_error_not_type_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            storage = JsonFileStorage(root / ".agent")
            cred_store = MockCredentialStore()
            # Do NOT save any key in cred_store

            now = datetime.datetime.now(datetime.timezone.utc)
            subtask = Subtask(subtask_id="st-1", title="test subtask", description="desc", dependencies=[], status=SubtaskStatus.PENDING)
            task = Task(
                task_id="task-1",
                objective="test task",
                status=TaskStatus.PENDING,
                created_at=now,
                updated_at=now,
                plan=TaskPlan(objective="test task", subtasks=[subtask]),
            )
            storage.save_task(task)

            provider_configs = [ProviderConfig(provider_id="mock", priority=10, enabled=True)]
            storage.save_provider_configs(provider_configs)

            config = AgentConfig(project=root, provider="mock")
            scheduler = Scheduler(config, storage, cred_store)

            # run_once should catch AuthenticationError (subclass of ProviderError) and not crash with TypeError
            scheduler.run_once()

            # Verify the provider state is updated with NOT_CONFIGURED (which is set for AUTHENTICATION_ERROR)
            scheduler_state = storage.load_scheduler_state()
            provider_state = scheduler_state.provider_states.get("mock")
            self.assertIsNotNone(provider_state)
            self.assertEqual(provider_state.availability, ProviderAvailability.NOT_CONFIGURED)


if __name__ == "__main__":
    unittest.main()

