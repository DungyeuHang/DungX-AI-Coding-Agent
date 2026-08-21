from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_agent.config import AgentConfig
from local_agent.credentials import MockCredentialStore
from local_agent.models import (
    Checkpoint, ProviderAvailability, ProviderConfig,
    SchedulerState, Subtask, SubtaskStatus, Task, TaskPlan, TaskStatus,
)
from local_agent.scheduler import Scheduler
from local_agent.storage import JsonFileStorage


class Phase310_1_Tests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.storage = JsonFileStorage(self.root / ".agent_data")
        self.credential_store = MockCredentialStore()
        self.base_config = AgentConfig.from_environment(self.root)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root)

    def _create_task(self, status=TaskStatus.PENDING) -> Task:
        now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
        task = Task(task_id="test-task-1", objective="Test", status=status, created_at=now, updated_at=now)
        self.storage.save_task(task)
        return task

    def test_scheduler_loads_persistent_config(self):
        provider_configs = [
            ProviderConfig(provider_id="gemini", priority=10, enabled=True, config_overrides={"model": "gemini-test"}),
        ]
        self.storage.save_provider_configs(provider_configs)

        scheduler = Scheduler(self.base_config, self.storage, self.credential_store)
        self.assertIn("gemini", scheduler.registry.providers)
        self.assertEqual(scheduler.registry.providers["gemini"].priority, 10)

    def test_provider_selection_with_credentials(self):
        provider_configs = [
            ProviderConfig(provider_id="gemini", priority=10, enabled=True),
            ProviderConfig(provider_id="openai", priority=20, enabled=True),
        ]
        self.storage.save_provider_configs(provider_configs)
        self.credential_store.save("dungx-ai-coding-agent", "gemini", "gemini-key")

        scheduler = Scheduler(self.base_config, self.storage, self.credential_store)
        self.assertEqual(scheduler.state.provider_states["gemini"].availability, ProviderAvailability.AVAILABLE)
        self.assertEqual(scheduler.state.provider_states["openai"].availability, ProviderAvailability.NOT_CONFIGURED)

        selected_configs = scheduler._select_providers(self._create_task())
        self.assertTrue(selected_configs)
        self.assertEqual(selected_configs[0].provider, "gemini")

    def test_fallback_with_missing_credentials(self):
        provider_configs = [
            ProviderConfig(provider_id="gemini", priority=10, enabled=True),
            ProviderConfig(provider_id="openai", priority=20, enabled=True),
        ]
        self.storage.save_provider_configs(provider_configs)
        # Only save credential for the lower-priority provider
        self.credential_store.save("dungx-ai-coding-agent", "openai", "openai-key")

        scheduler = Scheduler(self.base_config, self.storage, self.credential_store)
        self.assertEqual(scheduler.state.provider_states["gemini"].availability, ProviderAvailability.NOT_CONFIGURED)
        self.assertEqual(scheduler.state.provider_states["openai"].availability, ProviderAvailability.AVAILABLE)

        selected_configs = scheduler._select_providers(self._create_task())
        self.assertTrue(selected_configs)
        self.assertEqual(selected_configs[0].provider, "openai")

    def test_all_providers_not_configured(self):
        provider_configs = [
            ProviderConfig(provider_id="gemini", priority=10, enabled=True),
        ]
        self.storage.save_provider_configs(provider_configs)
        # No credentials saved

        scheduler = Scheduler(self.base_config, self.storage, self.credential_store)
        selected_configs = scheduler._select_providers(self._create_task())
        self.assertFalse(selected_configs)

    @mock.patch("local_agent.scheduler.Orchestrator")
    def test_secrets_not_in_state_files(self, mock_orchestrator):
        secret = "TEST_SECRET_VALUE_GEMINI"
        provider_configs = [
            ProviderConfig(provider_id="gemini", priority=10, enabled=True, config_overrides={"model": "gemini-test"}),
        ]
        self.storage.save_provider_configs(provider_configs)
        self.credential_store.save("dungx-ai-coding-agent", "gemini", secret)
        task = self._create_task()
        now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
        subtask = Subtask(subtask_id="sub-1", title="test", goal="test", status=SubtaskStatus.PENDING, created_at=now)
        task.plan = TaskPlan(objective="Test", subtasks=[subtask])
        self.storage.save_task(task)

        # Mock orchestrator to create a checkpoint
        def orchestrator_run_effect(*args, **kwargs):
            checkpoint = Checkpoint(
                checkpoint_id="chk-1", task_id=task.task_id, subtask_id="sub-1",
                timestamp=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
                current_state_description="test state", files_changed=[], repository_diff="",
                validation_state={}, last_provider_result={"outcome": "TEST"},
                next_recommended_action="resume", continuation_context={"objective": "Test"}
            )
            self.storage.save_checkpoint(checkpoint)
            task.latest_checkpoint_id = "chk-1"
            self.storage.save_task(task)
            return mock.MagicMock(outcome="COMPLETED", plan_proposal=None)
        mock_orchestrator.return_value.run.side_effect = orchestrator_run_effect

        scheduler = Scheduler(self.base_config, self.storage, self.credential_store)
        scheduler.run_once()

        # Check persisted files for the secret
        providers_json = (self.root / ".agent_data" / "providers.json").read_text()
        self.assertNotIn(secret, providers_json)

        scheduler_state_json = (self.root / ".agent_data" / "scheduler_state.json").read_text()
        self.assertNotIn(secret, scheduler_state_json)

        task_json = (self.root / ".agent_data" / "tasks" / f"{task.task_id}.json").read_text()
        self.assertNotIn(secret, task_json)

        checkpoint_json = (self.root / ".agent_data" / "checkpoints" / "chk-1.json").read_text()
        self.assertNotIn(secret, checkpoint_json)

    def test_restart_behavior(self):
        secret = "my-secret-key"
        provider_configs = [ProviderConfig(provider_id="gemini", priority=10, enabled=True)]
        self.storage.save_provider_configs(provider_configs)
        self.credential_store.save("dungx-ai-coding-agent", "gemini", secret)

        # Simulate restart
        scheduler = Scheduler(self.base_config, self.storage, self.credential_store)
        selected_configs = scheduler._select_providers(self._create_task())
        self.assertTrue(selected_configs)
        self.assertEqual(selected_configs[0].provider, "gemini")

        # The scheduler should be able to get the key to build the final config
        final_key = self.credential_store.get("dungx-ai-coding-agent", "gemini")
        self.assertEqual(final_key, secret)