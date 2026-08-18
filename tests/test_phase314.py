from __future__ import annotations

import datetime
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_agent.config import AgentConfig
from local_agent.credentials import MockCredentialStore
from local_agent.models import (
    AddSubtask, FailureAnalysis, PlanProposal, ProviderConfig, Subtask,
    SubtaskStatus, Task, TaskPlan, TaskStatus,
)
from local_agent.orchestrator import Orchestrator
from local_agent.providers import MockProvider
from local_agent.scheduler import Scheduler
from local_agent.storage import JsonFileStorage


class MockPlannerProvider(MockProvider):
    def propose_plan_modification(self, task: str, plan: TaskPlan, failure: FailureAnalysis) -> PlanProposal | None:
        new_subtask = Subtask(subtask_id="new_sub", title="New Prerequisite", goal="Do this first", status=SubtaskStatus.PENDING)
        return PlanProposal(
            reason="The original plan was missing a prerequisite.",
            additions=[AddSubtask(subtask=new_subtask)]
        )


class Phase314Tests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.storage = JsonFileStorage(self.root / ".agent_data")
        self.credential_store = MockCredentialStore()
        self.base_config = AgentConfig.from_environment(self.root)

        provider_configs = [ProviderConfig(provider_id="mock", priority=10, enabled=True)]
        self.storage.save_provider_configs(provider_configs)
        self.credential_store.save("dungx-ai-coding-agent", "mock", "mock-key")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root)

    def _create_task(self, task_id="test-task-1", status=TaskStatus.PENDING, plan: TaskPlan | None = None) -> Task:
        now = datetime.datetime.now(datetime.timezone.utc)
        task = Task(task_id=task_id, objective="Test", status=status, created_at=now, updated_at=now, plan=plan)
        self.storage.save_task(task)
        return task

    def _get_scheduler(self) -> Scheduler:
        return Scheduler(self.base_config, self.storage, self.credential_store)

    @mock.patch("local_agent.scheduler.Orchestrator")
    def test_scheduler_accepts_valid_proposal_and_sets_plan_proposed(self, MockOrchestrator):
        # Arrange
        s1 = Subtask(subtask_id="1", title="A", goal="A")
        plan = TaskPlan(objective="Test", subtasks=[s1])
        task = self._create_task(plan=plan, status=TaskStatus.RUNNING)

        new_subtask = Subtask(subtask_id="new_sub", title="New Prerequisite", goal="Do this first")
        proposal = PlanProposal(reason="Missing step", additions=[AddSubtask(subtask=new_subtask)])
        
        # Mock Orchestrator to return a report with a proposal
        mock_report = mock.MagicMock()
        mock_report.plan_proposal = proposal
        MockOrchestrator.return_value.run.return_value = mock_report

        scheduler = self._get_scheduler()

        # Act
        scheduler.run_once()

        # Assert
        loaded_task = self.storage.load_task(task.task_id)
        self.assertEqual(loaded_task.status, TaskStatus.PLAN_PROPOSED)
        self.assertIsNotNone(loaded_task.plan_proposal)
        self.assertEqual(loaded_task.plan_proposal.reason, "Missing step")
        self.assertEqual(len(loaded_task.plan_proposal.additions), 1)

    def test_plan_proposed_task_is_not_executed(self):
        task = self._create_task(status=TaskStatus.PLAN_PROPOSED)
        scheduler = self._get_scheduler()
        
        with mock.patch("local_agent.scheduler.Orchestrator") as mock_orchestrator:
            scheduler.run_once()
            mock_orchestrator.return_value.run.assert_not_called()

    def test_approve_proposal_applies_changes_and_sets_pending(self):
        # Arrange
        s1 = Subtask(subtask_id="1", title="A", goal="A")
        plan = TaskPlan(objective="Test", subtasks=[s1])
        new_subtask = Subtask(subtask_id="new_sub", title="New Prerequisite", goal="Do this first")
        proposal = PlanProposal(reason="Missing step", additions=[AddSubtask(subtask=new_subtask)])
        task = self._create_task(plan=plan, status=TaskStatus.PLAN_PROPOSED)
        task.plan_proposal = proposal
        self.storage.save_task(task)

        # Act
        from local_agent.cli import main
        with mock.patch("sys.stdout", new=io.StringIO()):
            main(["approve-proposal", "--project", str(self.root), task.task_id])

        # Assert
        loaded_task = self.storage.load_task(task.task_id)
        self.assertEqual(loaded_task.status, TaskStatus.PENDING)
        self.assertIsNone(loaded_task.plan_proposal)
        self.assertEqual(len(loaded_task.plan.subtasks), 2)
        self.assertIn("new_sub", [s.subtask_id for s in loaded_task.plan.subtasks])

    def test_reject_proposal_clears_proposal_and_sets_paused(self):
        # Arrange
        s1 = Subtask(subtask_id="1", title="A", goal="A")
        plan = TaskPlan(objective="Test", subtasks=[s1])
        new_subtask = Subtask(subtask_id="new_sub", title="New Prerequisite", goal="Do this first")
        proposal = PlanProposal(reason="Missing step", additions=[AddSubtask(subtask=new_subtask)])
        task = self._create_task(plan=plan, status=TaskStatus.PLAN_PROPOSED)
        task.plan_proposal = proposal
        self.storage.save_task(task)

        # Act
        from local_agent.cli import main
        with mock.patch("sys.stdout", new=io.StringIO()):
            main(["reject-proposal", "--project", str(self.root), task.task_id])

        # Assert
        loaded_task = self.storage.load_task(task.task_id)
        self.assertEqual(loaded_task.status, TaskStatus.PAUSED)
        self.assertIsNone(loaded_task.plan_proposal)
        self.assertEqual(len(loaded_task.plan.subtasks), 1) # Plan should be unchanged

    def test_proposal_survives_restart(self):
        # Arrange
        s1 = Subtask(subtask_id="1", title="A", goal="A")
        plan = TaskPlan(objective="Test", subtasks=[s1])
        new_subtask = Subtask(subtask_id="new_sub", title="New Prerequisite", goal="Do this first")
        proposal = PlanProposal(reason="Missing step", additions=[AddSubtask(subtask=new_subtask)])
        task = self._create_task(plan=plan, status=TaskStatus.PLAN_PROPOSED)
        task.plan_proposal = proposal
        self.storage.save_task(task)

        # Act: Simulate restart by creating a new storage instance and loading
        reloaded_storage = JsonFileStorage(self.root / ".agent_data")
        reloaded_task = reloaded_storage.load_task(task.task_id)

        # Assert
        self.assertEqual(reloaded_task.status, TaskStatus.PLAN_PROPOSED)
        self.assertIsNotNone(reloaded_task.plan_proposal)
        self.assertEqual(reloaded_task.plan_proposal.reason, "Missing step")

    def test_backward_compatibility_with_pre_314_tasks(self):
        # Arrange: Create a task JSON without the 'plan_proposal' field
        task_id = "pre-314-task"
        task_path = self.storage.tasks_dir / f"{task_id}.json"
        pre_314_task_data = {
            "task_id": task_id, "objective": "Old Task", "status": "pending",
            "created_at": datetime.datetime.now().isoformat(), "updated_at": datetime.datetime.now().isoformat()
        }
        with open(task_path, 'w') as f:
            json.dump(pre_314_task_data, f)

        # Act & Assert: Loading should not fail
        try:
            loaded_task = self.storage.load_task(task_id)
            self.assertEqual(loaded_task.task_id, task_id)
            self.assertIsNone(loaded_task.plan_proposal)
        except Exception as e:
            self.fail(f"Loading a pre-3.14 task failed with: {e}")