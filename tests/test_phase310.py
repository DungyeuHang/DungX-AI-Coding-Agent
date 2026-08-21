from __future__ import annotations

import datetime
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_agent.config import AgentConfig
from local_agent.credentials import MockCredentialStore
from local_agent.models import (
    Plan, ProviderAvailability, ProviderCapability,
    ProviderConfig, ProviderError,
    QuotaExceededError, SubtaskStatus, Task, TaskStatus,
)
from local_agent.providers import AIProvider, BaseHTTPProvider, build_provider
from local_agent.scheduler import Scheduler
from local_agent.storage import JsonFileStorage
from .test_phase39 import MockTaskStorage


class MockProviderA(AIProvider):
    def __init__(self, config):
        self.config = config
        self.should_fail = False

    @property
    def capabilities(self) -> set[ProviderCapability]:
        return {ProviderCapability.PLANNING, ProviderCapability.IMPLEMENTATION}

    def generate_plan(self, task, context):
        return Plan(objective="mock plan", steps=["step1"])

    def generate_code(self, task, plan, context, failure=None, review=None):
        if self.should_fail:
            raise QuotaExceededError("Quota exceeded on Provider A")
        return []

    def analyze_failure(self, execution, diff, context, plan):
        return mock.MagicMock()

    def review_changes(self, task, plan, diff, context):
        return {"verdict": "APPROVED", "summary": "mock review", "findings": []}

class MockProviderB(AIProvider):
    def __init__(self, config):
        self.config = config
        self.continuation_context_received = None

    @property
    def capabilities(self) -> set[ProviderCapability]:
        return {ProviderCapability.PLANNING, ProviderCapability.IMPLEMENTATION}

    def generate_plan(self, task, context):
        self.continuation_context_received = context.metadata.get("continuation_context")
        return Plan(objective="mock plan", steps=["step1"])

    def generate_code(self, task, plan, context, failure=None, review=None):
        return []

    def analyze_failure(self, execution, diff, context, plan):
        return mock.MagicMock()

    def review_changes(self, task, plan, diff, context):
        return mock.MagicMock(verdict="APPROVED")


class Phase310Tests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.storage = MockTaskStorage() # Use mock storage for simplicity
        self.base_config = AgentConfig.from_environment(self.root)
        self.config_a = AgentConfig.from_environment(self.root, provider="provider_a")
        self.config_b = AgentConfig.from_environment(self.root, provider="provider_b")
        self.provider_map = {"provider_a": MockProviderA, "provider_b": MockProviderB}
        self.mock_credential_store = MockCredentialStore()

    def tearDown(self):
        shutil.rmtree(self.root)

    def _create_task(self, status=TaskStatus.PENDING, next_retry_at=None) -> Task:
        now = datetime.datetime.now(datetime.timezone.utc)
        task = Task(
            task_id="test-task-1",
            objective="Test objective",
            status=status,
            created_at=now,
            updated_at=now,
            next_retry_at=next_retry_at,
        )
        self.storage.save_task(task)
        return task

    @mock.patch("local_agent.scheduler.build_provider")
    @mock.patch("local_agent.scheduler.Orchestrator")
    def test_provider_selection_and_priority(self, mock_orchestrator, mock_build_provider):
        self._create_task()
        provider_configs = [
            ProviderConfig(provider_id="provider_a", priority=20, enabled=True),
            ProviderConfig(provider_id="provider_b", priority=10, enabled=True),
        ]
        self.storage.save_provider_configs(provider_configs)
        self.mock_credential_store.save("dungx-ai-coding-agent", "provider_a", "key-a")
        self.mock_credential_store.save("dungx-ai-coding-agent", "provider_b", "key-b")

        def build_side_effect(config, *args, **kwargs):
            return self.provider_map[config.provider](config)
        mock_build_provider.side_effect = build_side_effect

        # Provider B has higher priority (lower number)
        scheduler = Scheduler(self.base_config, self.storage, self.mock_credential_store)

        scheduler.run_once()

        # Assert that provider B was selected
        self.assertEqual(mock_orchestrator.call_args[0][0].provider, "provider_b")

    @mock.patch("local_agent.scheduler.build_provider")
    @mock.patch("local_agent.scheduler.Orchestrator")
    def test_unavailable_provider_is_skipped(self, mock_orchestrator, mock_build_provider):
        self._create_task()
        provider_configs = [
            ProviderConfig(provider_id="provider_a", priority=20, enabled=True),
            ProviderConfig(provider_id="provider_b", priority=10, enabled=True),
        ]
        self.storage.save_provider_configs(provider_configs)
        self.mock_credential_store.save("dungx-ai-coding-agent", "provider_a", "key-a")
        self.mock_credential_store.save("dungx-ai-coding-agent", "provider_b", "key-b")

        def build_side_effect(config, *args, **kwargs):
            return self.provider_map[config.provider](config)
        mock_build_provider.side_effect = build_side_effect

        scheduler = Scheduler(self.base_config, self.storage, self.mock_credential_store)
        scheduler.state.provider_states["provider_b"].availability = ProviderAvailability.UNAVAILABLE
        self.storage.save_scheduler_state(scheduler.state)

        scheduler.run_once()

        # Assert that provider A was selected because B is unavailable
        self.assertEqual(mock_orchestrator.call_args[0][0].provider, "provider_a")

    @mock.patch("local_agent.scheduler.build_provider")
    @mock.patch("local_agent.scheduler.Orchestrator")
    def test_fallback_on_quota_exhaustion(self, mock_orchestrator, mock_build_provider):
        task = self._create_task()
        provider_configs = [
            ProviderConfig(provider_id="provider_a", priority=10, enabled=True),
            ProviderConfig(provider_id="provider_b", priority=20, enabled=True),
        ]
        self.storage.save_provider_configs(provider_configs)
        self.mock_credential_store.save("dungx-ai-coding-agent", "provider_a", "key-a")
        self.mock_credential_store.save("dungx-ai-coding-agent", "provider_b", "key-b")

        # First run: Provider A fails with QuotaExceededError
        provider_a_instance = MockProviderA(self.config_a)
        provider_a_instance.should_fail = True
        
        def build_side_effect_run1(config, *args, **kwargs):
            if config.provider == "provider_a":
                return provider_a_instance
            return self.provider_map[config.provider](config)
        mock_build_provider.side_effect = build_side_effect_run1

        scheduler = Scheduler(self.base_config, self.storage, self.mock_credential_store)

        # Mock orchestrator to simulate a paused task
        def orchestrator_run_effect(*args, **kwargs):
            task.status = TaskStatus.PAUSED
            task.outcome = "QUOTA_EXCEEDED"
            task.execution_history.append({"type": "failure", "failure": {"category": "QUOTA_EXCEEDED", "details": {}}})
            self.storage.save_task(task)
            return mock.MagicMock(outcome="QUOTA_EXCEEDED")
        mock_orchestrator.return_value.run.side_effect = orchestrator_run_effect

        scheduler.run_once()

        # Verify state after first run
        provider_a_state = scheduler.state.provider_states["provider_a"]
        self.assertEqual(provider_a_state.availability, ProviderAvailability.COOLDOWN)
        self.assertIsNotNone(provider_a_state.cooldown_until)
        paused_task = self.storage.load_task(task.task_id)
        self.assertEqual(paused_task.status, TaskStatus.PAUSED)

        # Second run: Provider A is on cooldown, so Provider B should be chosen
        provider_b_instance = MockProviderB(self.config_b)
        def build_side_effect_run2(config, *args, **kwargs):
            if config.provider == "provider_b":
                return provider_b_instance
            return self.provider_map[config.provider](config)
        mock_build_provider.side_effect = build_side_effect_run2

        mock_orchestrator.reset_mock()
        def orchestrator_run_effect_2(*args, **kwargs):
            task.status = TaskStatus.COMPLETED
            if task.plan and task.plan.subtasks:
                task.plan.subtasks[0].status = SubtaskStatus.COMPLETED
            self.storage.save_task(task)
            return mock.MagicMock(outcome="COMPLETED", plan_proposal=None)
        mock_orchestrator.return_value.run.side_effect = orchestrator_run_effect_2

        # Simulate time passing to get past cooldown for the task retry, but not provider
        paused_task.next_retry_at = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=1)
        self.storage.save_task(paused_task)

        scheduler.run_once()

        # Verify state after second run
        mock_orchestrator.return_value.run.assert_called_once()
        self.assertEqual(mock_orchestrator.call_args[0][0].provider, "provider_b")
        completed_task = self.storage.load_task(task.task_id)
        self.assertEqual(completed_task.status, TaskStatus.COMPLETED)

    def test_all_providers_unavailable_pauses_task(self):
        task = self._create_task()
        provider_configs = [ProviderConfig(provider_id="provider_a", priority=10, enabled=True)]
        self.storage.save_provider_configs(provider_configs)
        self.mock_credential_store.save("dungx-ai-coding-agent", "provider_a", "key-a")
        with mock.patch("local_agent.scheduler.build_provider") as mock_build_provider:
            mock_build_provider.return_value.capabilities = {ProviderCapability.PLANNING}
            scheduler = Scheduler(self.base_config, self.storage, self.mock_credential_store)
        scheduler.state.provider_states["provider_a"].availability = ProviderAvailability.COOLDOWN
        scheduler.state.provider_states["provider_a"].cooldown_until = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=5)

        scheduler.run_once()

        paused_task = self.storage.load_task(task.task_id)
        self.assertEqual(paused_task.status, TaskStatus.PAUSED)
        self.assertIsNotNone(paused_task.next_retry_at)
        self.assertGreater(paused_task.next_retry_at, datetime.datetime.now(datetime.timezone.utc))

    @mock.patch("local_agent.scheduler.build_provider")
    def test_scheduler_persistence(self, mock_build_provider):
        real_storage = JsonFileStorage(self.root / ".agent_data")
        provider_configs = [ProviderConfig(provider_id="provider_a", priority=10, enabled=True)]
        real_storage.save_provider_configs(provider_configs)
        mock_credential_store = MockCredentialStore()
        mock_credential_store.save("dungx-ai-coding-agent", "provider_a", "key-a")

        def build_side_effect(config, api_key=None):
            return self.provider_map[config.provider](config)
        mock_build_provider.side_effect = build_side_effect

        scheduler1 = Scheduler(self.base_config, real_storage, mock_credential_store)
        scheduler1.state.provider_states["provider_a"].availability = ProviderAvailability.COOLDOWN
        cooldown_time = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=10)
        scheduler1.state.provider_states["provider_a"].cooldown_until = cooldown_time
        real_storage.save_scheduler_state(scheduler1.state)

        # Simulate restart
        scheduler2 = Scheduler(self.base_config, real_storage, mock_credential_store)
        provider_a_state = scheduler2.state.provider_states["provider_a"]
        self.assertEqual(provider_a_state.availability, ProviderAvailability.COOLDOWN)
        self.assertAlmostEqual(provider_a_state.cooldown_until, cooldown_time, delta=datetime.timedelta(seconds=1))

    def test_respects_retry_after(self):
        provider_configs = [ProviderConfig(provider_id="provider_a", priority=10, enabled=True)]
        self.storage.save_provider_configs(provider_configs)
        self.mock_credential_store.save("dungx-ai-coding-agent", "provider_a", "key-a")
        scheduler = Scheduler(self.base_config, self.storage, self.mock_credential_store)
        now = datetime.datetime.now(datetime.timezone.utc)
        
        with mock.patch('datetime.datetime') as mock_dt:
            mock_dt.now.return_value = now
            scheduler._update_provider_state_after_failure("provider_a", "RATE_LIMIT", 120)

        state = scheduler.state.provider_states["provider_a"]
        self.assertEqual(state.availability, ProviderAvailability.COOLDOWN)
        self.assertEqual(state.cooldown_until, now + datetime.timedelta(seconds=120))

    def test_no_duplicate_execution(self):
        task = self._create_task()
        task.assigned_to = "some_other_worker"
        self.storage.save_task(task)

        scheduler = Scheduler(self.base_config, self.storage, self.mock_credential_store)
        
        # The scheduler should find no runnable tasks because the only one is assigned
        runnable = scheduler._find_runnable_task(self.storage.list_tasks())
        self.assertIsNone(runnable)

    def test_completed_task_is_not_executed(self):
        self._create_task(status=TaskStatus.COMPLETED)
        scheduler = Scheduler(self.base_config, self.storage, self.mock_credential_store)
        runnable = scheduler._find_runnable_task(self.storage.list_tasks())
        self.assertIsNone(runnable)