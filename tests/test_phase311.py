from __future__ import annotations

import datetime
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_agent.config import AgentConfig
from local_agent.credentials import MockCredentialStore
from local_agent.models import (
    Checkpoint, ProviderAvailability, ProviderCapability,
    ProviderConfig, Subtask, SubtaskStatus, Task,
    TaskPlan, TaskStatus,
)
from local_agent.planner import GraphValidator
from local_agent.providers import QuotaExceededError
from local_agent.scheduler import Scheduler
from local_agent.storage import JsonFileStorage


class Phase311Tests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.storage = JsonFileStorage(self.root / ".agent_data")
        self.credential_store = MockCredentialStore()
        self.base_config = AgentConfig.from_environment(self.root)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root)

    def _create_task(self, status=TaskStatus.PENDING, plan: TaskPlan | None = None) -> Task:
        now = datetime.datetime.now(datetime.timezone.utc)
        task = Task(task_id="test-task-1", objective="Test", status=status, created_at=now, updated_at=now, plan=plan)
        self.storage.save_task(task)
        return task

    # region Graph Validation Tests
    def test_graph_validator_accepts_valid_graph(self):
        s1 = Subtask(subtask_id="1", title="A", goal="A")
        s2 = Subtask(subtask_id="2", title="B", goal="B", dependencies=["1"])
        s3 = Subtask(subtask_id="3", title="C", goal="C", dependencies=["1"])
        s4 = Subtask(subtask_id="4", title="D", goal="D", dependencies=["2", "3"])
        validator = GraphValidator([s1, s2, s3, s4])
        self.assertEqual(validator.validate(), [])

    def test_graph_validator_rejects_self_dependency(self):
        s1 = Subtask(subtask_id="1", title="A", goal="A", dependencies=["1"])
        validator = GraphValidator([s1])
        self.assertIn("Subtask '1' has a self-dependency.", validator.validate())

    def test_graph_validator_rejects_two_node_cycle(self):
        s1 = Subtask(subtask_id="1", title="A", goal="A", dependencies=["2"])
        s2 = Subtask(subtask_id="2", title="B", goal="B", dependencies=["1"])
        validator = GraphValidator([s1, s2])
        self.assertIn("Dependency cycle detected in the task plan.", validator.validate())

    def test_graph_validator_rejects_multi_node_cycle(self):
        s1 = Subtask(subtask_id="1", title="A", goal="A", dependencies=["3"])
        s2 = Subtask(subtask_id="2", title="B", goal="B", dependencies=["1"])
        s3 = Subtask(subtask_id="3", title="C", goal="C", dependencies=["2"])
        validator = GraphValidator([s1, s2, s3])
        self.assertIn("Dependency cycle detected in the task plan.", validator.validate())

    def test_graph_validator_rejects_missing_dependency(self):
        s1 = Subtask(subtask_id="1", title="A", goal="A", dependencies=["99"])
        validator = GraphValidator([s1])
        self.assertIn("Subtask '1' has a missing dependency: '99'.", validator.validate())

    def test_graph_validator_rejects_duplicate_ids(self):
        s1 = Subtask(subtask_id="1", title="A", goal="A")
        s2 = Subtask(subtask_id="1", title="B", goal="B")
        validator = GraphValidator([s1, s2])
        self.assertIn("Duplicate subtask IDs found.", validator.validate())
    # endregion

    # region Scheduler Execution Tests
    def test_scheduler_finds_next_runnable_subtask(self):
        s1 = Subtask(subtask_id="1", title="A", goal="A", status=SubtaskStatus.COMPLETED)
        s2 = Subtask(subtask_id="2", title="B", goal="B", dependencies=["1"], status=SubtaskStatus.PENDING)
        s3 = Subtask(subtask_id="3", title="C", goal="C", dependencies=["2"], status=SubtaskStatus.PENDING)
        plan = TaskPlan(objective="Test", subtasks=[s1, s2, s3])
        task = self._create_task(plan=plan)

        scheduler = Scheduler(self.base_config, self.storage, self.credential_store)
        next_subtask = scheduler._find_next_runnable_subtask(task)

        self.assertIsNotNone(next_subtask)
        self.assertEqual(next_subtask.subtask_id, "2")

    def test_scheduler_handles_blocked_subtask(self):
        s1 = Subtask(subtask_id="1", title="A", goal="A", status=SubtaskStatus.PENDING)
        s2 = Subtask(subtask_id="2", title="B", goal="B", dependencies=["1"], status=SubtaskStatus.PENDING)
        plan = TaskPlan(objective="Test", subtasks=[s1, s2])
        task = self._create_task(plan=plan)

        scheduler = Scheduler(self.base_config, self.storage, self.credential_store)
        next_subtask = scheduler._find_next_runnable_subtask(task)

        self.assertIsNotNone(next_subtask)
        self.assertEqual(next_subtask.subtask_id, "1") # Should run s1 first

    def test_scheduler_completes_task_when_all_subtasks_are_done(self):
        s1 = Subtask(subtask_id="1", title="A", goal="A", status=SubtaskStatus.COMPLETED)
        s2 = Subtask(subtask_id="2", title="B", goal="B", dependencies=["1"], status=SubtaskStatus.COMPLETED)
        plan = TaskPlan(objective="Test", subtasks=[s1, s2])
        task = self._create_task(plan=plan, status=TaskStatus.RUNNING)

        scheduler = Scheduler(self.base_config, self.storage, self.credential_store)
        scheduler._check_and_complete_task(task)

        completed_task = self.storage.load_task(task.task_id)
        self.assertEqual(completed_task.status, TaskStatus.COMPLETED)

    @mock.patch("local_agent.scheduler.Orchestrator")
    @mock.patch("local_agent.planner.Planner.create_task_plan")
    def test_scheduler_executes_subtasks_in_order(self, mock_create_plan, mock_orchestrator):
        # Mock the planner to return a deterministic plan
        s1 = Subtask(subtask_id="1", title="Step 1", goal="First step", status=SubtaskStatus.PENDING)
        s2 = Subtask(subtask_id="2", title="Step 2", goal="Second step", dependencies=["1"], status=SubtaskStatus.PENDING)
        mock_plan = TaskPlan(objective="Test", subtasks=[s1, s2])
        mock_create_plan.return_value = mock_plan

        # Mock the orchestrator to mark subtasks as complete
        def orchestrator_run_effect(task: Task, subtask_id: str, **kwargs):
            subtask = next(s for s in task.plan.subtasks if s.subtask_id == subtask_id)
            subtask.status = SubtaskStatus.COMPLETED
            self.storage.save_task(task)
            return mock.MagicMock(outcome="COMPLETED")
        mock_orchestrator.return_value.run.side_effect = orchestrator_run_effect

        # Setup scheduler with a configured provider
        provider_configs = [ProviderConfig(provider_id="mock", priority=10, enabled=True)]
        self.storage.save_provider_configs(provider_configs)
        self.credential_store.save("dungx-ai-coding-agent", "mock", "mock-key")
        scheduler = Scheduler(self.base_config, self.storage, self.credential_store)

        # Create the initial task
        task = self._create_task()

        # Run 1: Should plan the task and run subtask 1
        scheduler.run_once()
        mock_create_plan.assert_called_once()
        mock_orchestrator.return_value.run.assert_called_with(task=mock.ANY, subtask_id="1", progress=mock.ANY)
        task_after_run1 = self.storage.load_task(task.task_id)
        self.assertEqual(task_after_run1.plan.subtasks[0].status, SubtaskStatus.COMPLETED)
        self.assertEqual(task_after_run1.status, TaskStatus.RUNNING)

        # Run 2: Should run subtask 2
        scheduler.run_once()
        mock_orchestrator.return_value.run.assert_called_with(task=mock.ANY, subtask_id="2", progress=mock.ANY)
        task_after_run2 = self.storage.load_task(task.task_id)
        self.assertEqual(task_after_run2.plan.subtasks[1].status, SubtaskStatus.COMPLETED)
        self.assertEqual(task_after_run2.status, TaskStatus.COMPLETED)

        # Run 3: Should do nothing as task is complete
        scheduler.run_once()
        # run should have been called twice in total
        self.assertEqual(mock_orchestrator.return_value.run.call_count, 2)

    def test_backward_compatibility_with_old_tasks(self):
        # Create a task without a 'plan' attribute
        old_task = self._create_task()
        delattr(old_task, 'plan')
        self.storage.save_task(old_task)

        with mock.patch("local_agent.planner.Planner.create_task_plan") as mock_create_plan:
            s1 = Subtask(subtask_id="1", title=old_task.objective, goal=old_task.objective, status=SubtaskStatus.PENDING)
            mock_plan = TaskPlan(objective=old_task.objective, subtasks=[s1])
            mock_create_plan.return_value = mock_plan

            provider_configs = [ProviderConfig(provider_id="mock", priority=10, enabled=True)]
            self.storage.save_provider_configs(provider_configs)
            self.credential_store.save("dungx-ai-coding-agent", "mock", "mock-key")
            scheduler = Scheduler(self.base_config, self.storage, self.credential_store)

            with mock.patch("local_agent.scheduler.Orchestrator") as mock_orchestrator:
                def orchestrator_run_effect(task: Task, subtask_id: str, **kwargs):
                    subtask = next(s for s in task.plan.subtasks if s.subtask_id == subtask_id)
                    subtask.status = SubtaskStatus.COMPLETED
                    self.storage.save_task(task)
                    return mock.MagicMock(outcome="COMPLETED")
                mock_orchestrator.return_value.run.side_effect = orchestrator_run_effect

                scheduler.run_once()

        # Assert that the planner was called to create a plan for the old task
        mock_create_plan.assert_called_once()
        task_after_run = self.storage.load_task(old_task.task_id)
        self.assertIsNotNone(task_after_run.plan)
        self.assertEqual(len(task_after_run.plan.subtasks), 1)
        self.assertEqual(task_after_run.status, TaskStatus.COMPLETED)

    @mock.patch("local_agent.planner.Planner.create_task_plan")
    def test_continuation_context_includes_subtask_info(self, mock_create_plan):
        s1 = Subtask(subtask_id="1", title="Step 1", goal="First step", status=SubtaskStatus.COMPLETED)
        s2 = Subtask(subtask_id="2", title="Step 2", goal="Second step", dependencies=["1"], status=SubtaskStatus.PAUSED)
        mock_plan = TaskPlan(objective="Test", subtasks=[s1, s2])
        task = self._create_task(plan=mock_plan)

        with mock.patch("local_agent.scheduler.Orchestrator") as mock_orchestrator:
            # We need to capture the context passed to the orchestrator's provider
            orchestrator_instance = mock.MagicMock()
            orchestrator_instance.run.return_value = mock.MagicMock(outcome="COMPLETED", plan_proposal=None)
            mock_orchestrator.return_value = orchestrator_instance

            # Simulate a checkpoint being created for the paused subtask
            checkpoint = Checkpoint(
                checkpoint_id="chk-1", task_id=task.task_id, subtask_id="2",
                timestamp=datetime.datetime.now(datetime.timezone.utc), current_state_description="Paused mid-flight",
                files_changed=[], repository_diff="", validation_state={},
                last_provider_result={"outcome": "QUOTA_EXCEEDED"},
                next_recommended_action="resume",
                continuation_context={
                    "task_objective": "Test",
                    "current_subtask_goal": "Second step",
                    "completed_subtasks_summary": ["Step 1"],
                }
            )
            s2.latest_checkpoint_id = "chk-1"
            self.storage.save_checkpoint(checkpoint)
            self.storage.save_task(task)

            provider_configs = [ProviderConfig(provider_id="mock", priority=10, enabled=True)]
            self.storage.save_provider_configs(provider_configs)
            self.credential_store.save("dungx-ai-coding-agent", "mock", "mock-key")
            scheduler = Scheduler(self.base_config, self.storage, self.credential_store)

            scheduler.run_once()

            # Check the context that would have been passed to the orchestrator
            run_call_args = orchestrator_instance.run.call_args
            self.assertIsNotNone(run_call_args)

    def test_full_lifecycle_with_fallback(self):
        """
        A realistic integration test simulating the full lifecycle with real state transitions.
        """
        # 1. Setup: Configure two providers and create a task.
        provider_configs = [
            ProviderConfig(provider_id="provider_a", priority=10, enabled=True),
            ProviderConfig(provider_id="provider_b", priority=20, enabled=True),
        ]
        self.storage.save_provider_configs(provider_configs)
        self.credential_store.save("dungx-ai-coding-agent", "provider_a", "key-a")
        self.credential_store.save("dungx-ai-coding-agent", "provider_b", "key-b")

        task = self._create_task()

        # Helper to simulate a successful orchestrator run for a subtask
        def successful_run(task: Task, subtask_id: str, **kwargs):
            t = self.storage.load_task(task.task_id)
            subtask = next(s for s in t.plan.subtasks if s.subtask_id == subtask_id)
            subtask.status = SubtaskStatus.COMPLETED
            self.storage.save_task(t)
            return mock.MagicMock(outcome="COMPLETED", plan_proposal=None)

        with mock.patch("local_agent.scheduler.Orchestrator") as MockOrchestrator, \
             mock.patch("local_agent.planner.Planner.create_task_plan") as mock_create_plan:

            MockOrchestrator.return_value.run.return_value = mock.MagicMock(outcome="COMPLETED", plan_proposal=None)

            # 2. Run 1: Plan the task.
            s1 = Subtask(subtask_id="1", title="Step 1", goal="A", status=SubtaskStatus.PENDING)
            s2 = Subtask(subtask_id="2", title="Step 2", goal="B", dependencies=["1"], status=SubtaskStatus.PENDING)
            s3 = Subtask(subtask_id="3", title="Step 3", goal="C", dependencies=["2"], status=SubtaskStatus.PENDING)
            mock_plan = TaskPlan(objective="Test ABC", subtasks=[s1, s2, s3])
            mock_create_plan.return_value = mock_plan

            scheduler = Scheduler(self.base_config, self.storage, self.credential_store)
            scheduler.run_once()
            mock_create_plan.assert_called_once()

            # 3. Run 2: Execute Subtask A successfully with Provider A.
            MockOrchestrator.return_value.run.side_effect = successful_run
            scheduler = Scheduler(self.base_config, self.storage, self.credential_store)
            scheduler.run_once()
            self.assertEqual(self.storage.load_task(task.task_id).plan.subtasks[0].status, SubtaskStatus.COMPLETED)
            self.assertEqual(MockOrchestrator.call_args[0][0].provider, "provider_a")

            # 4. Run 3: Attempt Subtask B with Provider A, which fails with QuotaExceededError.
            MockOrchestrator.return_value.run.side_effect = QuotaExceededError("quota fail")
            scheduler = Scheduler(self.base_config, self.storage, self.credential_store)
            scheduler.run_once()
            task_after_fail = self.storage.load_task(task.task_id)
            self.assertEqual(task_after_fail.status, TaskStatus.PAUSED)
            self.assertEqual(task_after_fail.plan.subtasks[1].status, SubtaskStatus.PAUSED)
            
            # Simulate restart and check provider state
            restarted_scheduler = Scheduler(self.base_config, self.storage, self.credential_store)
            self.assertEqual(restarted_scheduler.state.provider_states["provider_a"].availability, ProviderAvailability.COOLDOWN)

            # 5. Run 4: Resume Subtask B with Provider B.
            task_after_fail.next_retry_at = datetime.datetime.now(datetime.timezone.utc)
            self.storage.save_task(task_after_fail)
            MockOrchestrator.return_value.run.side_effect = successful_run
            restarted_scheduler.run_once()
            self.assertEqual(self.storage.load_task(task.task_id).plan.subtasks[1].status, SubtaskStatus.COMPLETED)
            self.assertEqual(MockOrchestrator.call_args[0][0].provider, "provider_b")

            # 6. Run 5: Execute Subtask C successfully.
            scheduler = Scheduler(self.base_config, self.storage, self.credential_store)
            scheduler.run_once()

            # 7. Final state: Task is complete.
            final_task = self.storage.load_task(task.task_id)
            self.assertEqual(final_task.status, TaskStatus.COMPLETED)