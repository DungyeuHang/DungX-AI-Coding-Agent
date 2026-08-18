from __future__ import annotations

import datetime
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_agent.config import AgentConfig
from local_agent.models import (
    ProviderAvailability, ProviderCapability, ProviderConfig, ProviderError, ProviderRuntimeState,
    QuotaExceededError, RateLimitError, SchedulerState, Subtask, SubtaskStatus,
    Task, TaskStatus,
)
from local_agent.providers import AIProvider, build_provider
from local_agent.scheduler import Scheduler
from local_agent.storage import JsonFileStorage
from .test_phase39 import MockTaskStorage


class MockProviderA(AIProvider):
    def __init__(self, config):
        self.config = config
        self.capabilities = {ProviderCapability.PLANNING, ProviderCapability.IMPLEMENTATION}
        self.should_fail = False

    def generate_plan(self, task, context):
        if self.should_fail:
            raise QuotaExceededError("Quota exceeded on Provider A")
        return mock.MagicMock()

    def generate_code(self, task, plan, context, failure=None, review=None):
        if self.should_fail:
            raise QuotaExceededError("Quota exceeded on Provider A")
        return []

    def analyze_failure(self, execution, diff, context, plan):
        return mock.MagicMock()

    def review_changes(self, task, plan, diff, context):
        return mock.MagicMock(verdict="APPROVED")

class MockProviderB(AIProvider):
    def __init__(self, config):
        self.config = config
        self.capabilities = {ProviderCapability.PLANNING, ProviderCapability.IMPLEMENTATION}
        self.continuation_context_received = None

    def generate_plan(self, task, context):
        self.continuation_context_received = context.metadata.get("continuation_context")
        return mock.MagicMock()

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
        self.mock_credential_store = mock.MagicMock() # This will be used in 3.10.1

    def tearDown(self):
        import shutil
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
        # In 3.10, provider configs were passed directly. In 3.10.1 this moves to storage.
        # We adapt the test to the 3.10 style.
        provider_configs_tuples = [
            (self.config_a, 20),
            (self.config_b, 10),
        ]
        # Provider B has higher priority (lower number)
        scheduler = Scheduler(provider_configs_tuples, self.storage)
        
        def build_side_effect(config):
            return self.provider_map[config.provider](config)
        mock_build_provider.side_effect = build_side_effect

        scheduler.run_once()

        # Assert that provider B was selected
        self.assertEqual(mock_orchestrator.call_args[0][0].provider, "provider_b")

    @mock.patch("local_agent.scheduler.build_provider")
    @mock.patch("local_agent.scheduler.Orchestrator")
    def test_unavailable_provider_is_skipped(self, mock_orchestrator, mock_build_provider):
        self._create_task()
        provider_configs_tuples = [
            (self.config_a, 20),
            (self.config_b, 10),
        ]
        scheduler = Scheduler(provider_configs_tuples, self.storage)
        scheduler.state.provider_states["provider_b"].availability = ProviderAvailability.UNAVAILABLE

        def build_side_effect(config):
            return self.provider_map[config.provider](config)
        mock_build_provider.side_effect = build_side_effect

        scheduler.run_once()

        # Assert that provider A was selected because B is unavailable
        self.assertEqual(mock_orchestrator.call_args[0][0].provider, "provider_a")

    @mock.patch("local_agent.scheduler.build_provider")
    @mock.patch("local_agent.scheduler.Orchestrator")
    def test_fallback_on_quota_exhaustion(self, mock_orchestrator, mock_build_provider):
        task = self._create_task()
        provider_configs_tuples = [
            (self.config_a, 10),
            (self.config_b, 20),
        ]
        scheduler = Scheduler(provider_configs_tuples, self.storage)

        # First run: Provider A fails with QuotaExceededError
        provider_a_instance = MockProviderA(self.config_a)
        provider_a_instance.should_fail = True
        
        def build_side_effect_run1(config):
            if config.provider == "provider_a":
                return provider_a_instance
            return self.provider_map[config.provider](config)
        mock_build_provider.side_effect = build_side_effect_run1

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
        def build_side_effect_run2(config):
            if config.provider == "provider_b":
                return provider_b_instance
            return self.provider_map[config.provider](config)
        mock_build_provider.side_effect = build_side_effect_run2

        # Reset orchestrator mock for the second run
        mock_orchestrator.reset_mock()
        def orchestrator_run_effect_2(*args, **kwargs):
            task.status = TaskStatus.COMPLETED
            self.storage.save_task(task)
            return mock.MagicMock(outcome="COMPLETED")
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
        provider_configs_tuples = [(self.config_a, 10)]
        scheduler = Scheduler(provider_configs_tuples, self.storage)
        scheduler.state.provider_states["provider_a"].availability = ProviderAvailability.COOLDOWN
        scheduler.state.provider_states["provider_a"].cooldown_until = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=5)

        scheduler.run_once()

        paused_task = self.storage.load_task(task.task_id)
        self.assertEqual(paused_task.status, TaskStatus.PAUSED)
        self.assertIsNotNone(paused_task.next_retry_at)
        self.assertGreater(paused_task.next_retry_at, datetime.datetime.now(datetime.timezone.utc))

    def test_scheduler_persistence(self):
        real_storage = JsonFileStorage(self.root / ".agent_data")
        provider_configs_tuples = [(self.config_a, 10)]
        scheduler1 = Scheduler(provider_configs_tuples, real_storage)
        scheduler1.state.provider_states["provider_a"].availability = ProviderAvailability.COOLDOWN
        cooldown_time = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=10)
        scheduler1.state.provider_states["provider_a"].cooldown_until = cooldown_time
        real_storage.save_scheduler_state(scheduler1.state)

        # Simulate restart
        scheduler2 = Scheduler(provider_configs_tuples, real_storage)
        provider_a_state = scheduler2.state.provider_states["provider_a"]
        self.assertEqual(provider_a_state.availability, ProviderAvailability.COOLDOWN)
        self.assertAlmostEqual(provider_a_state.cooldown_until, cooldown_time, delta=datetime.timedelta(seconds=1))

    def test_respects_retry_after(self):
        scheduler = Scheduler([], self.storage) # No providers needed for this direct call
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

        scheduler = Scheduler([], self.storage)
        
        # The scheduler should find no runnable tasks because the only one is assigned
        runnable = scheduler._find_runnable_task(self.storage.list_tasks())
        self.assertIsNone(runnable)

    def test_completed_task_is_not_executed(self):
        self._create_task(status=TaskStatus.COMPLETED)
        scheduler = Scheduler([], self.storage)
        runnable = scheduler._find_runnable_task(self.storage.list_tasks())
        self.assertIsNone(runnable)