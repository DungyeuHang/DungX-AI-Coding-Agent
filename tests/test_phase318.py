from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_agent.config import AgentConfig
from local_agent.credentials import MockCredentialStore
from local_agent.models import (
    ProviderCapability,
    ProviderConfig,
    Task,
    TaskStatus,
)
from local_agent.providers import AIProvider, MockProvider
from local_agent.scheduler import Scheduler
from local_agent.storage import JsonFileStorage


class PlanningOnlyProvider(AIProvider):
    @property
    def capabilities(self) -> set[ProviderCapability]:
        return {ProviderCapability.PLANNING}

    def generate_plan(self, task, context):
        return mock.MagicMock()


class ImplementationOnlyProvider(AIProvider):
    @property
    def capabilities(self) -> set[ProviderCapability]:
        return {
            ProviderCapability.IMPLEMENTATION,
            ProviderCapability.REPAIR,
            ProviderCapability.REVIEW,
        }

    def generate_code(self, task, plan, context, failure=None, review=None):
        return []


class Phase318_DynamicCapabilityTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.storage = JsonFileStorage(self.root / ".agent_data")
        self.credential_store = MockCredentialStore()
        self.base_config = AgentConfig.from_environment(self.root)

        self.provider_map = {
            "planning_only": PlanningOnlyProvider,
            "implementation_only": ImplementationOnlyProvider,
            "full_mock": MockProvider,
        }

    def tearDown(self):
        import shutil

        shutil.rmtree(self.root)

    def _create_task(self, status=TaskStatus.PENDING, plan=None) -> Task:
        now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
        task = Task(
            task_id="test-task-1",
            objective="Test",
            status=status,
            created_at=now,
            updated_at=now,
            plan=plan,
        )
        self.storage.save_task(task)
        return task

    @mock.patch("local_agent.scheduler.build_provider")
    def test_scheduler_selects_provider_based_on_dynamic_capabilities(
        self, mock_build_provider
    ):
        # Arrange: Configure two providers with distinct, non-overlapping capabilities
        provider_configs = [
            ProviderConfig(provider_id="planning_only", priority=10, enabled=True),
            ProviderConfig(provider_id="implementation_only", priority=20, enabled=True),
        ]
        self.storage.save_provider_configs(provider_configs)
        self.credential_store.save("dungx-ai-coding-agent", "planning_only", "key1")
        self.credential_store.save(
            "dungx-ai-coding-agent", "implementation_only", "key2"
        )

        def build_side_effect(config, api_key=None):
            return self.provider_map[config.provider]()

        mock_build_provider.side_effect = build_side_effect

        scheduler = Scheduler(self.base_config, self.storage, self.credential_store)

        # Act & Assert for PLANNING
        task_no_plan = self._create_task()
        planning_providers = scheduler._select_providers(
            task_no_plan, required_capabilities={ProviderCapability.PLANNING}
        )

        self.assertEqual(len(planning_providers), 1)
        self.assertEqual(planning_providers[0].provider, "planning_only")

        # Act & Assert for IMPLEMENTATION
        mock_plan = __import__("local_agent.models").TaskPlan(
            objective="Test", subtasks=[]
        )
        task_with_plan = self._create_task(plan=mock_plan)
        implementation_providers = scheduler._select_providers(
            task_with_plan, required_capabilities={ProviderCapability.IMPLEMENTATION}
        )

        self.assertEqual(len(implementation_providers), 1)
        self.assertEqual(implementation_providers[0].provider, "implementation_only")