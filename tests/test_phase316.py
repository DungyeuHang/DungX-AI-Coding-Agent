from __future__ import annotations

import datetime
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_agent.config import AgentConfig
from local_agent.models import ( # Added ProviderConfig, SchedulerState, ProviderAvailability
    Plan, ProjectContext, ProviderAvailability, ProviderConfig, ProviderCapability,
    RegisteredProvider, SchedulerState, Subtask, Task, TaskPlan, TaskStatus, SemanticIndex,
)
from local_agent.orchestrator import Orchestrator # Keep Orchestrator for context analysis in _plan_task_with_provider
from local_agent.providers import AIProvider, ProviderError
from local_agent.storage import TaskStorage


# Re-using mocks from other tests for consistency
class MockTaskStorage(TaskStorage):
    def __init__(self):
        self.tasks: dict[str, Task] = {}
        self.checkpoints: dict[str, "Checkpoint"] = {}
        self._semantic_indexes: dict[Path, SemanticIndex] = {}

    def save_task(self, task: Task) -> None:
        self.tasks[task.task_id] = task

    def load_task(self, task_id: str) -> Task:
        if task_id not in self.tasks:
            raise FileNotFoundError(f"Task with ID {task_id} not found.")
        return self.tasks[task_id]

    def list_tasks(self) -> list[Task]:
        return list(self.tasks.values())

    def find_next_task(self, capabilities: set[ProviderCapability]) -> Task | None:
        # Simplified for test: find first pending task
        for task in sorted(self.tasks.values(), key=lambda t: t.created_at):
            if task.status == TaskStatus.PENDING and not task.plan:
                return task
        return None

    def save_checkpoint(self, checkpoint: "Checkpoint") -> None:
        self.checkpoints[checkpoint.checkpoint_id] = checkpoint

    def load_checkpoint(self, checkpoint_id: str) -> "Checkpoint":
        if checkpoint_id not in self.checkpoints:
            raise FileNotFoundError(f"Checkpoint with ID {checkpoint_id} not found.")
        return self.checkpoints[checkpoint_id]

    # New methods for Scheduler interaction
    def save_provider_configs(self, configs: list[ProviderConfig]) -> None:
        self.provider_configs = {c.provider_id: c for c in configs}

    def load_provider_configs(self) -> list[ProviderConfig]:
        return list(self.provider_configs.values())

    def save_scheduler_state(self, state: SchedulerState) -> None:
        self.scheduler_state = state

    def load_scheduler_state(self) -> SchedulerState:
        return self.scheduler_state

    def load_semantic_index(self, project_root: Path) -> SemanticIndex | None:
        return self._semantic_indexes.get(project_root)

    def save_semantic_index(self, project_root: Path, semantic_index: SemanticIndex) -> None:
        self._semantic_indexes[project_root] = semantic_index

    provider_configs: dict[str, ProviderConfig] = {}
    scheduler_state: SchedulerState = SchedulerState()


class FailingPlannerProvider(AIProvider):
    def __init__(self):
        self.generate_plan_calls = 0
        self.capabilities = {ProviderCapability.PLANNING}

    def generate_plan(self, task: str, context: ProjectContext) -> Plan:
        self.generate_plan_calls += 1
        raise ProviderError("Planning failed as designed for test")


class SuccessfulPlannerProvider(AIProvider):
    def __init__(self):
        self.generate_plan_calls = 0
        self.capabilities = {ProviderCapability.PLANNING}

    def generate_plan(self, task: str, context: ProjectContext) -> Plan:
        self.generate_plan_calls += 1
        return Plan(objective=task, steps=["Generated Step 1"])


# Mock CredentialStore for the real Scheduler
class MockCredentialStore:
    def __init__(self):
        self.keys: dict[str, str] = {}

    def get(self, namespace: str, key_id: str) -> str | None:
        return self.keys.get(key_id)

    def has(self, namespace: str, key_id: str) -> bool:
        return self.get(namespace, key_id) is not None

    def set(self, namespace: str, key_id: str, value: str) -> None:
        self.keys[key_id] = value

    def delete(self, namespace: str, key_id: str) -> None:
        if key_id in self.keys:
            del self.keys[key_id]


# Import the REAL Scheduler
from local_agent.scheduler import Scheduler

# Mock the Orchestrator's analyze method to avoid actual project scanning in the test
mock_orchestrator_analyze = mock.MagicMock(spec=Orchestrator.analyze)
mock_orchestrator_analyze.return_value = ProjectContext(root="/mock/project")


class Phase316_SchedulerPlanningFallbackTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.config = AgentConfig.from_environment(self.root)
        self.storage = MockTaskStorage()
        self.credential_store = MockCredentialStore()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root)

    def test_planning_fallback_to_secondary_provider(self):
        """
        Tests that if the primary planning provider fails, the scheduler
        falls back to the next available provider to generate a plan.
        """
        failing_provider = FailingPlannerProvider()
        successful_provider = SuccessfulPlannerProvider()
        
        # Create AgentConfig for each provider, as Scheduler expects them
        failing_config = AgentConfig.from_environment(self.root, provider="failing_mock", model="failing-model")
        successful_config = AgentConfig.from_environment(self.root, provider="successful_mock", model="successful-model")

        # Register providers with the storage for the Scheduler to load
        provider_configs_to_save = [
            ProviderConfig(provider_id="failing_mock", enabled=True, priority=1, config_overrides=failing_config.__dict__),
            ProviderConfig(provider_id="successful_mock", enabled=True, priority=2, config_overrides=successful_config.__dict__),
        ]
        self.storage.save_provider_configs(provider_configs_to_save)

        # Mock build_provider to return our specific instances
        # This is crucial because Scheduler.build_provider will try to instantiate real providers.
        # We need to intercept this and return our mock instances.
        with mock.patch('local_agent.scheduler.build_provider') as mock_build_provider, \
             mock.patch('local_agent.orchestrator.Orchestrator.analyze', new=mock_orchestrator_analyze):
            def side_effect_build_provider(agent_config, api_key=None):
                if agent_config.provider == "failing_mock":
                    return failing_provider
                elif agent_config.provider == "successful_mock":
                    return successful_provider
                raise ValueError(f"Unknown mock provider: {agent_config.provider}")
            mock_build_provider.side_effect = side_effect_build_provider

            # Provide API keys for the mock providers, as Scheduler checks for them
            self.credential_store.set("dungx-ai-coding-agent", "failing_mock", "dummy-key-failing")
            self.credential_store.set("dungx-ai-coding-agent", "successful_mock", "dummy-key-successful")

            task = Task(
                task_id="task-1", objective="Test planning fallback", status=TaskStatus.PENDING,
                created_at=datetime.datetime.now(datetime.timezone.utc), updated_at=datetime.datetime.now(datetime.timezone.utc),
            )
            self.storage.save_task(task)

            scheduler = Scheduler(self.config, self.storage, self.credential_store)
            scheduler.run_once()

        self.assertEqual(failing_provider.generate_plan_calls, 1)
        self.assertEqual(successful_provider.generate_plan_calls, 1)

        updated_task = self.storage.load_task("task-1")
        self.assertIsNotNone(updated_task.plan)
        self.assertEqual(len(updated_task.plan.subtasks), 1)
        self.assertEqual(updated_task.plan.subtasks[0].title, "Generated Step 1")
        self.assertEqual(updated_task.status, TaskStatus.PENDING) # Should be PENDING for execution after planning

        # Verify provider states were updated
        scheduler_state = self.storage.load_scheduler_state()
        self.assertIn("failing_mock", scheduler_state.provider_states)
        self.assertEqual(scheduler_state.provider_states["failing_mock"].availability, ProviderAvailability.COOLDOWN)
        self.assertIn("successful_mock", scheduler_state.provider_states)
        self.assertEqual(scheduler_state.provider_states["successful_mock"].availability, ProviderAvailability.AVAILABLE)