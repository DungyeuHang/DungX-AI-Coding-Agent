from __future__ import annotations

import datetime
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_agent.config import AgentConfig
from local_agent.credentials import MockCredentialStore
from local_agent.models import (
    ProviderAvailability, ProviderCapability, ProviderConfig, QuotaExceededError, Subtask,
    SubtaskStatus, Task, TaskPlan, TaskStatus,
)
from local_agent.planner import GraphValidator
from local_agent.scheduler import Scheduler
from local_agent.storage import JsonFileStorage


class Phase312Tests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.storage = JsonFileStorage(self.root / ".agent_data")
        self.credential_store = MockCredentialStore()
        self.base_config = AgentConfig.from_environment(self.root)

        # Configure a mock provider for planning and execution
        self.provider_configs = [ProviderConfig(provider_id="mock", priority=10, enabled=True)]
        self.storage.save_provider_configs(self.provider_configs)
        self.credential_store.save("dungx-ai-coding-agent", "mock", "mock-key")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root)

    def _create_task(self, task_id="test-task-1", objective="Test", status=TaskStatus.PENDING, plan: TaskPlan | None = None) -> Task:
        now = datetime.datetime.now(datetime.timezone.utc)
        task = Task(task_id=task_id, objective=objective, status=status, created_at=now, updated_at=now, plan=plan)
        self.storage.save_task(task)
        return task

    def _get_scheduler(self, approval_mode="never") -> Scheduler:
        config = AgentConfig.from_environment(self.root, approval_mode=approval_mode)
        return Scheduler(config, self.storage, self.credential_store)

    def _mock_planner_create_task_plan(self, mock_create_task_plan, subtasks: list[Subtask]):
        mock_plan = TaskPlan(objective="Test Plan", subtasks=subtasks)
        mock_create_task_plan.return_value = mock_plan

    def _mock_orchestrator_run_success(self, mock_orchestrator):
        def successful_run(task: Task, subtask_id: str, **kwargs):
            t = self.storage.load_task(task.task_id)
            subtask = next(s for s in t.plan.subtasks if s.subtask_id == subtask_id)
            subtask.status = SubtaskStatus.COMPLETED
            self.storage.save_task(t)
            return mock.MagicMock(outcome="COMPLETED")
        mock_orchestrator.return_value.run.side_effect = successful_run

    # region AgentConfig Tests
    def test_approval_mode_parsing(self):
        config = AgentConfig.from_environment(self.root, approval_mode="plan_review")
        self.assertEqual(config.approval_mode, "plan_review")
        config = AgentConfig.from_environment(self.root, approval_mode="never")
        self.assertEqual(config.approval_mode, "never")
        with self.assertRaisesRegex(ValueError, "approval_mode must be 'never', 'plan_review', or 'always'"):
            AgentConfig.from_environment(self.root, approval_mode="invalid")
    # endregion

    # region Plan Generation and Approval Flow
    @mock.patch("local_agent.planner.Planner.create_task_plan")
    def test_new_task_enters_plan_review_when_approval_mode_is_plan_review(self, mock_create_task_plan):
        task = self._create_task()
        s1 = Subtask(subtask_id="1", title="A", goal="A")
        self._mock_planner_create_task_plan(mock_create_task_plan, [s1])

        scheduler = self._get_scheduler(approval_mode="plan_review")
        scheduler.run_once()

        loaded_task = self.storage.load_task(task.task_id)
        self.assertEqual(loaded_task.status, TaskStatus.PLAN_REVIEW)
        self.assertIsNotNone(loaded_task.plan)
        mock_create_task_plan.assert_called_once()

    @mock.patch("local_agent.planner.Planner.create_task_plan")
    def test_new_task_remains_pending_when_approval_mode_is_never(self, mock_create_task_plan):
        task = self._create_task()
        s1 = Subtask(subtask_id="1", title="A", goal="A")
        self._mock_planner_create_task_plan(mock_create_task_plan, [s1])

        scheduler = self._get_scheduler(approval_mode="never")
        scheduler.run_once()

        loaded_task = self.storage.load_task(task.task_id)
        self.assertEqual(loaded_task.status, TaskStatus.PENDING)
        self.assertIsNotNone(loaded_task.plan)
        mock_create_task_plan.assert_called_once()

    @mock.patch("local_agent.scheduler.Orchestrator")
    def test_plan_review_task_is_never_executed_by_scheduler(self, mock_orchestrator):
        s1 = Subtask(subtask_id="1", title="A", goal="A", status=SubtaskStatus.PENDING)
        plan = TaskPlan(objective="Test", subtasks=[s1])
        task = self._create_task(status=TaskStatus.PLAN_REVIEW, plan=plan)

        scheduler = self._get_scheduler(approval_mode="plan_review")
        scheduler.run_once()

        mock_orchestrator.return_value.run.assert_not_called()
        self.assertEqual(self.storage.load_task(task.task_id).status, TaskStatus.PLAN_REVIEW)

    def test_approve_plan_changes_plan_review_to_pending(self):
        s1 = Subtask(subtask_id="1", title="A", goal="A")
        plan = TaskPlan(objective="Test", subtasks=[s1])
        task = self._create_task(status=TaskStatus.PLAN_REVIEW, plan=plan)

        # Simulate CLI command
        from local_agent.cli import main
        with mock.patch("sys.stdout", new=io.StringIO()):
            main(["approve-plan", "--project", str(self.root), task.task_id])

        loaded_task = self.storage.load_task(task.task_id)
        self.assertEqual(loaded_task.status, TaskStatus.PENDING)

    def test_approve_plan_rejects_invalid_plans(self):
        # Create a plan with a cycle
        s1 = Subtask(subtask_id="1", title="A", goal="A", dependencies=["2"])
        s2 = Subtask(subtask_id="2", title="B", goal="B", dependencies=["1"])
        plan = TaskPlan(objective="Test Cycle", subtasks=[s1, s2])
        task = self._create_task(status=TaskStatus.PLAN_REVIEW, plan=plan)

        # Simulate CLI command
        from local_agent.cli import main
        with mock.patch("sys.stdout", new=io.StringIO()) as mock_stdout:
            with self.assertRaises(SystemExit) as cm:
                main(["approve-plan", "--project", str(self.root), task.task_id])
            self.assertEqual(cm.exception.code, 1)
            self.assertIn("Plan for task", mock_stdout.getvalue())
            self.assertIn("is invalid and cannot be approved", mock_stdout.getvalue())

        loaded_task = self.storage.load_task(task.task_id)
        self.assertEqual(loaded_task.status, TaskStatus.PLAN_REVIEW) # Should remain in review

    def test_approve_plan_rejects_incorrect_task_states(self):
        task = self._create_task(status=TaskStatus.PENDING) # Not in PLAN_REVIEW

        # Simulate CLI command
        from local_agent.cli import main
        with mock.patch("sys.stdout", new=io.StringIO()) as mock_stdout:
            with self.assertRaises(SystemExit) as cm:
                main(["approve-plan", "--project", str(self.root), task.task_id])
            self.assertEqual(cm.exception.code, 1)
            self.assertIn("is not in PLAN_REVIEW status", mock_stdout.getvalue())

        loaded_task = self.storage.load_task(task.task_id)
        self.assertEqual(loaded_task.status, TaskStatus.PENDING) # Should not mutate

    def test_reject_plan_changes_plan_review_to_rejected(self):
        s1 = Subtask(subtask_id="1", title="A", goal="A")
        plan = TaskPlan(objective="Test", subtasks=[s1])
        task = self._create_task(status=TaskStatus.PLAN_REVIEW, plan=plan)

        # Simulate CLI command
        from local_agent.cli import main
        with mock.patch("sys.stdout", new=io.StringIO()):
            main(["reject-plan", "--project", str(self.root), task.task_id])

        loaded_task = self.storage.load_task(task.task_id)
        self.assertEqual(loaded_task.status, TaskStatus.REJECTED)
        self.assertIsNotNone(loaded_task.plan) # Plan should still be there

    def test_rejected_task_is_never_executed(self):
        s1 = Subtask(subtask_id="1", title="A", goal="A", status=SubtaskStatus.PENDING)
        plan = TaskPlan(objective="Test", subtasks=[s1])
        task = self._create_task(status=TaskStatus.REJECTED, plan=plan)

        scheduler = self._get_scheduler(approval_mode="never")
        scheduler.run_once()

        loaded_task = self.storage.load_task(task.task_id)
        self.assertEqual(loaded_task.status, TaskStatus.REJECTED) # Should remain rejected

    def test_edit_plan_title_and_goal(self):
        s1 = Subtask(subtask_id="1", title="Old Title", goal="Old Goal", status=SubtaskStatus.PENDING)
        plan = TaskPlan(objective="Test", subtasks=[s1])
        task = self._create_task(status=TaskStatus.PLAN_REVIEW, plan=plan)

        # Simulate CLI command
        from local_agent.cli import main
        with mock.patch("sys.stdout", new=io.StringIO()):
            main(["edit-plan", "--project", str(self.root), task.task_id, "--subtask", "1", "--title", "New Title", "--goal", "New Goal"])

        loaded_task = self.storage.load_task(task.task_id)
        self.assertEqual(loaded_task.plan.subtasks[0].title, "New Title")
        self.assertEqual(loaded_task.plan.subtasks[0].goal, "New Goal")
        self.assertEqual(loaded_task.status, TaskStatus.PLAN_REVIEW) # Should remain in review

    def test_edit_plan_acceptance_criteria_and_dependencies(self):
        s1 = Subtask(subtask_id="1", title="A", goal="A", acceptance_criteria=["old_ac"], dependencies=[])
        s2 = Subtask(subtask_id="2", title="B", goal="B", dependencies=[])
        plan = TaskPlan(objective="Test", subtasks=[s1, s2])
        task = self._create_task(status=TaskStatus.PLAN_REVIEW, plan=plan)

        # Simulate CLI command
        from local_agent.cli import main
        with mock.patch("sys.stdout", new=io.StringIO()):
            main(["edit-plan", "--project", str(self.root), task.task_id, "--subtask", "1", "--acceptance-criteria", "new_ac1", "new_ac2", "--dependencies", "2"])

        loaded_task = self.storage.load_task(task.task_id)
        self.assertEqual(loaded_task.plan.subtasks[0].acceptance_criteria, ["new_ac1", "new_ac2"])
        self.assertEqual(loaded_task.plan.subtasks[0].dependencies, ["2"])
        self.assertEqual(loaded_task.status, TaskStatus.PLAN_REVIEW)

    def test_edit_plan_rejects_invalid_dependency_edit(self):
        s1 = Subtask(subtask_id="1", title="A", goal="A")
        plan = TaskPlan(objective="Test", subtasks=[s1])
        task = self._create_task(status=TaskStatus.PLAN_REVIEW, plan=plan)

        # Simulate CLI command to create a self-dependency
        from local_agent.cli import main
        with mock.patch("sys.stdout", new=io.StringIO()) as mock_stdout:
            with self.assertRaises(SystemExit) as cm:
                main(["edit-plan", "--project", str(self.root), task.task_id, "--subtask", "1", "--dependencies", "1"])
            self.assertEqual(cm.exception.code, 1)
            self.assertIn("Edited plan is invalid", mock_stdout.getvalue())
            self.assertIn("Subtask '1' has a self-dependency.", mock_stdout.getvalue())

        loaded_task = self.storage.load_task(task.task_id)
        self.assertEqual(loaded_task.plan.subtasks[0].dependencies, []) # Should not persist invalid edit
        self.assertEqual(loaded_task.status, TaskStatus.PLAN_REVIEW)

    def test_editing_pending_task_forces_plan_review(self):
        s1 = Subtask(subtask_id="1", title="A", goal="A")
        plan = TaskPlan(objective="Test", subtasks=[s1])
        task = self._create_task(status=TaskStatus.PENDING, plan=plan)

        # Simulate CLI command
        from local_agent.cli import main
        with mock.patch("sys.stdout", new=io.StringIO()):
            main(["edit-plan", "--project", str(self.root), task.task_id, "--subtask", "1", "--title", "New Title"])

        loaded_task = self.storage.load_task(task.task_id)
        self.assertEqual(loaded_task.status, TaskStatus.PLAN_REVIEW)
        self.assertEqual(loaded_task.plan.subtasks[0].title, "New Title")

    def test_plan_review_state_persists_across_restart(self):
        task = self._create_task(task_id="restart-test", objective="Test Restart")
        s1 = Subtask(subtask_id="1", title="A", goal="A")
        plan = TaskPlan(objective="Test Plan", subtasks=[s1])
        task.plan = plan
        task.status = TaskStatus.PLAN_REVIEW
        self.storage.save_task(task)

        # Simulate restart
        reloaded_task = self.storage.load_task("restart-test")
        self.assertEqual(reloaded_task.status, TaskStatus.PLAN_REVIEW)

    def test_rejected_state_persists_across_restart(self):
        task = self._create_task(task_id="rejected-restart-test", objective="Test Rejected Restart")
        s1 = Subtask(subtask_id="1", title="A", goal="A")
        plan = TaskPlan(objective="Test Plan", subtasks=[s1])
        task.plan = plan
        task.status = TaskStatus.REJECTED
        self.storage.save_task(task)

        # Simulate restart
        reloaded_task = self.storage.load_task("rejected-restart-test")
        self.assertEqual(reloaded_task.status, TaskStatus.REJECTED)

    @mock.patch("local_agent.scheduler.Orchestrator")
    @mock.patch("local_agent.planner.Planner.create_task_plan")
    def test_full_lifecycle_with_approval(self, mock_create_task_plan, mock_orchestrator):
        # 1. Setup: Create a task and configure scheduler for plan_review
        task = self._create_task(task_id="full-lifecycle-test")
        s1 = Subtask(subtask_id="1", title="Step 1", goal="First step", status=SubtaskStatus.PENDING)
        mock_plan = TaskPlan(objective="Test Plan", subtasks=[s1])
        mock_create_task_plan.return_value = mock_plan

        scheduler = self._get_scheduler(approval_mode="plan_review")

        # 2. Run 1: Scheduler generates plan and sets task to PLAN_REVIEW
        scheduler.run_once()
        loaded_task = self.storage.load_task(task.task_id)
        self.assertEqual(loaded_task.status, TaskStatus.PLAN_REVIEW)
        mock_orchestrator.return_value.run.assert_not_called()

        # 3. Run 2: Scheduler should not execute while in PLAN_REVIEW
        scheduler.run_once()
        mock_orchestrator.return_value.run.assert_not_called()

        # 4. Approve the plan via CLI
        from local_agent.cli import main
        with mock.patch("sys.stdout", new=io.StringIO()):
            main(["approve-plan", "--project", str(self.root), task.task_id])
        loaded_task = self.storage.load_task(task.task_id)
        self.assertEqual(loaded_task.status, TaskStatus.PENDING)

        # 5. Run 3: Scheduler should now execute Subtask 1
        def successful_run(task: Task, subtask_id: str, **kwargs):
            t = self.storage.load_task(task.task_id)
            subtask = next(s for s in t.plan.subtasks if s.subtask_id == subtask_id)
            subtask.status = SubtaskStatus.COMPLETED
            self.storage.save_task(t)
            return mock.MagicMock(outcome="COMPLETED", plan_proposal=None)
        mock_orchestrator.return_value.run.side_effect = successful_run

        scheduler.run_once()
        mock_orchestrator.return_value.run.assert_called_once_with(task=mock.ANY, subtask_id="1", progress=mock.ANY)
        loaded_task = self.storage.load_task(task.task_id)
        self.assertEqual(loaded_task.status, TaskStatus.COMPLETED)
        self.assertEqual(loaded_task.plan.subtasks[0].status, SubtaskStatus.COMPLETED)

    def test_show_task_displays_plan_and_subtask_info(self):
        s1 = Subtask(subtask_id="1", title="Subtask One", goal="Goal One", dependencies=[], acceptance_criteria=["AC1"])
        s2 = Subtask(subtask_id="2", title="Subtask Two", goal="Goal Two", dependencies=["1"], status=SubtaskStatus.PAUSED)
        plan = TaskPlan(objective="Overall Objective", subtasks=[s1, s2])
        task = self._create_task(task_id="show-task-test", objective="Overall Objective", status=TaskStatus.PLAN_REVIEW, plan=plan)

        from local_agent.cli import main
        with mock.patch("sys.stdout", new=io.StringIO()) as mock_stdout:
            main(["show-task", "--project", str(self.root), task.task_id])

        output = mock_stdout.getvalue()
        self.assertIn("Task ID: show-task-test", output)
        self.assertIn("Status: plan_review", output)
        self.assertIn("Task Plan:", output)
        self.assertIn("Objective: Overall Objective", output)
        self.assertIn("Subtasks:", output)
        self.assertIn("Subtask One (ID: 1, Status: pending)", output)
        self.assertIn("Goal: Goal One", output)
        self.assertIn("Acceptance Criteria: AC1", output)
        self.assertIn("Subtask Two (ID: 2, Status: paused) (depends on: 1)", output)
        self.assertIn("Goal: Goal Two", output)
    # endregion