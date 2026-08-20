from __future__ import annotations

import tempfile
import json
import unittest
from pathlib import Path
from unittest import mock

from local_agent.config import AgentConfig
from local_agent.credentials import MockCredentialStore
from local_agent.models import (
    Plan,
    ProviderCapability,
    ProviderConfig,
    Subtask,
    Task,
    TaskPlan,
    TaskStatus,
)
from local_agent.orchestrator import Orchestrator
from local_agent.providers import AIProvider, ProviderError
from local_agent.scheduler import Scheduler
from local_agent.storage import JsonFileStorage


class PlanningOnlyProvider(AIProvider):
    @property
    def capabilities(self) -> set[ProviderCapability]:
        return {ProviderCapability.PLANNING}

    def generate_plan(self, task, context):
        return json.loads('{"objective": "Planned by PlanningOnlyProvider", "steps": ["step1"]}')


class CoderOnlyProvider(AIProvider):
    @property
    def capabilities(self) -> set[ProviderCapability]:
        return {ProviderCapability.IMPLEMENTATION}

    def generate_code(self, task, plan, context, failure=None, review=None):
        return []


class Phase320_SpecialistTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.storage = JsonFileStorage(self.root / ".agent_data")
        self.credential_store = MockCredentialStore()
        self.base_config = AgentConfig.from_environment(self.root)

        self.provider_map = {
            "planner": PlanningOnlyProvider,
            "coder": CoderOnlyProvider,
        }

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root)

    def _create_task(self, status=TaskStatus.PENDING, plan: TaskPlan | None = None) -> Task:
        now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
        task = Task(task_id="test-task-1", objective="Test", status=status, created_at=now, updated_at=now, plan=plan)
        self.storage.save_task(task)
        return task

    @mock.patch("local_agent.scheduler.build_provider")
    def test_orchestrator_selects_different_specialists_for_stages(self, mock_build_provider):
        # Arrange
        provider_configs = [
            ProviderConfig(provider_id="planner", priority=10, enabled=True),
            ProviderConfig(provider_id="coder", priority=10, enabled=True),
        ]
        self.storage.save_provider_configs(provider_configs)
        self.credential_store.save("dungx-ai-coding-agent", "planner", "key1")
        self.credential_store.save("dungx-ai-coding-agent", "coder", "key2")

        def build_side_effect(config, api_key=None):
            provider_class = self.provider_map[config.provider]
            provider_instance = provider_class()
            provider_instance.config = config
            return provider_instance

        mock_build_provider.side_effect = build_side_effect

        scheduler = Scheduler(self.base_config, self.storage, self.credential_store)
        orchestrator = Orchestrator(self.base_config, self.storage, scheduler)

        subtask = Subtask(subtask_id="sub1", title="Test", goal="Test Goal")
        task = self._create_task(plan=TaskPlan(objective="Test", subtasks=[subtask]))

        # Mock parts of the orchestrator run that aren't under test
        with mock.patch.object(orchestrator, "_validate", return_value=[]), \
             mock.patch.object(orchestrator, "reviewer"), \
             mock.patch.object(orchestrator, "impact_analyzer"), \
             mock.patch.object(orchestrator, "analyzer"):

            # Act
            # We call the internal methods that use specialists directly
            plan_result = orchestrator._execute_with_specialist(task, ProviderCapability.PLANNING, lambda p: p.generate_plan(None, None), "planning")
            code_result = orchestrator._execute_with_specialist(task, ProviderCapability.IMPLEMENTATION, lambda p: p.generate_code(None, None, None), "implementation")

        # Assert
        self.assertIsNotNone(plan_result)
        self.assertEqual(plan_result.objective, "Planned by PlanningOnlyProvider")
        self.assertIsNotNone(code_result)
        self.assertEqual(code_result, [])

    def test_orchestrator_fails_if_no_specialist_is_found(self):
        # Arrange: No providers are configured
        scheduler = Scheduler(self.base_config, self.storage, self.credential_store)
        orchestrator = Orchestrator(self.base_config, self.storage, scheduler)
        task = self._create_task()

        # Act & Assert
        with self.assertRaisesRegex(ProviderError, "No available and capable provider found for stage: planning"):
            orchestrator._execute_with_specialist(task, ProviderCapability.PLANNING, lambda p: p, "planning")