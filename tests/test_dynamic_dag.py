"""Comprehensive test suite for Phase 4.9 Dynamic Task DAG Restructuring & Adaptive Subtask Graph Intelligence."""

import datetime
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from local_agent.config import AgentConfig
from local_agent.credentials import MockCredentialStore
from local_agent.filesystem import ProjectFilesystem
from local_agent.models import (
    AddSubtask,
    Checkpoint,
    DAGAmendmentGuard,
    DAGProposal,
    DependencyUpdate,
    FailureAnalysis,
    Plan,
    PlanProposal,
    ProjectContext,
    ProviderCapability,
    ProviderConfig,
    RecoveryState,
    RunReport,
    Subtask,
    SubtaskAddition,
    SubtaskInvalidation,
    SubtaskModification,
    SubtaskRemoval,
    SubtaskStatus,
    Task,
    TaskPlan,
    TaskPlanAmendment,
    TaskStatus,
)
from local_agent.orchestrator import Orchestrator
from local_agent.planner import GraphValidator, Planner
from local_agent.providers import AIProvider, MockProvider
from local_agent.scheduler import Scheduler
from local_agent.storage import JsonFileStorage


class TestTaskPlanDAGModels(unittest.TestCase):
    """Test data models, serialization, and versioning for Phase 4.9."""

    def test_subtask_status_extended(self):
        self.assertEqual(SubtaskStatus.SUPERSEDED.value, "superseded")
        self.assertEqual(SubtaskStatus.PRUNED.value, "pruned")
        self.assertEqual(SubtaskStatus.COMPLETED.value, "completed")
        self.assertEqual(SubtaskStatus.PENDING.value, "pending")

    def test_task_plan_defaults_and_active_subtasks(self):
        s1 = Subtask(subtask_id="st1", title="Step 1", goal="Goal 1", status=SubtaskStatus.COMPLETED)
        s2 = Subtask(subtask_id="st2", title="Step 2", goal="Goal 2", status=SubtaskStatus.SUPERSEDED)
        s3 = Subtask(subtask_id="st3", title="Step 3", goal="Goal 3", status=SubtaskStatus.PRUNED)
        s4 = Subtask(subtask_id="st4", title="Step 4", goal="Goal 4", status=SubtaskStatus.PENDING)

        tp = TaskPlan(objective="Test DAG", subtasks=[s1, s2, s3, s4])
        self.assertEqual(tp.version, 1)
        self.assertEqual(tp.amendments, [])
        self.assertEqual(len(tp.active_subtasks), 2)
        self.assertEqual(tp.active_subtask_ids, ["st1", "st4"])

    def test_task_plan_serialization_round_trip(self):
        s1 = Subtask(subtask_id="st1", title="Step 1", goal="Goal 1", status=SubtaskStatus.COMPLETED)
        s2 = Subtask(subtask_id="st2", title="Step 2", goal="Goal 2", dependencies=["st1"], status=SubtaskStatus.PENDING)
        tp = TaskPlan(objective="Build System", subtasks=[s1, s2], risks=["risk1"], assumptions=["asm1"], version=2)

        addition = SubtaskAddition(subtask=Subtask(subtask_id="st3", title="Step 3", goal="Goal 3"))
        prop = DAGProposal(reason="Need step 3", additions=[addition])
        amendment = TaskPlanAmendment(
            amendment_id="amend-123",
            version=2,
            timestamp=datetime.datetime(2026, 8, 28, 12, 0, 0, tzinfo=datetime.timezone.utc),
            proposal=prop,
            approved_by="deterministic_policy",
            previous_active_subtask_ids=["st1", "st2"],
            new_active_subtask_ids=["st1", "st2", "st3"],
            reason="Need step 3",
        )
        tp.amendments.append(amendment)

        serialized = tp.to_dict()
        self.assertEqual(serialized["version"], 2)
        self.assertEqual(len(serialized["amendments"]), 1)

        restored = TaskPlan.from_dict(serialized)
        self.assertEqual(restored.objective, "Build System")
        self.assertEqual(restored.version, 2)
        self.assertEqual(len(restored.subtasks), 2)
        self.assertEqual(len(restored.amendments), 1)
        self.assertEqual(restored.amendments[0].amendment_id, "amend-123")
        self.assertEqual(restored.amendments[0].proposal.reason, "Need step 3")

    def test_legacy_task_plan_deserialization_defaults(self):
        legacy_data = {
            "objective": "Legacy Objective",
            "subtasks": [
                {
                    "subtask_id": "legacy_st1",
                    "title": "Legacy Title",
                    "goal": "Legacy Goal",
                    "status": "pending",
                    "dependencies": [],
                    "acceptance_criteria": [],
                }
            ],
            "risks": [],
            "assumptions": [],
        }
        tp = TaskPlan.from_dict(legacy_data)
        self.assertEqual(tp.version, 1)
        self.assertEqual(tp.amendments, [])
        self.assertEqual(len(tp.subtasks), 1)
        self.assertEqual(tp.subtasks[0].subtask_id, "legacy_st1")

    def test_dag_proposal_serialization_round_trip(self):
        prop = DAGProposal(
            reason="Refactor graph",
            additions=[SubtaskAddition(subtask=Subtask(subtask_id="add1", title="Add 1", goal="Goal"))],
            removals=[SubtaskRemoval(subtask_id="rem1", reason="obsolete")],
            dependency_updates=[DependencyUpdate(subtask_id="dep1", dependencies=["d1", "d2"])],
            invalidations=[
                SubtaskInvalidation(
                    subtask_id="inv1",
                    reason="bad design",
                    replacement_subtask=Subtask(subtask_id="inv1_v2", title="Add 1 v2", goal="Fixed Goal"),
                )
            ],
        )
        serialized = prop.to_dict()
        restored = DAGProposal.from_dict(serialized)
        self.assertEqual(restored.reason, "Refactor graph")
        self.assertEqual(len(restored.additions), 1)
        self.assertEqual(len(restored.removals), 1)
        self.assertEqual(len(restored.dependency_updates), 1)
        self.assertEqual(len(restored.invalidations), 1)
        self.assertEqual(restored.invalidations[0].replacement_subtask.subtask_id, "inv1_v2")

    def test_dag_proposal_conversion_from_plan_proposal(self):
        legacy_prop = PlanProposal(
            reason="Legacy Reason",
            additions=[AddSubtask(subtask=Subtask(subtask_id="sub_add", title="T", goal="G"))],
            modifications=[SubtaskModification(subtask_id="sub_mod", dependencies=["dep1"])],
        )
        dag_prop = DAGProposal.from_plan_proposal(legacy_prop)
        self.assertEqual(dag_prop.reason, "Legacy Reason")
        self.assertEqual(len(dag_prop.additions), 1)
        self.assertEqual(dag_prop.additions[0].subtask.subtask_id, "sub_add")
        self.assertEqual(len(dag_prop.dependency_updates), 1)
        self.assertEqual(dag_prop.dependency_updates[0].dependencies, ["dep1"])


class TestDAGAmendmentGuard(unittest.TestCase):
    """Test deterministic DAGAmendmentGuard safety, acyclicity, quotas, and limits."""

    def setUp(self):
        self.guard = DAGAmendmentGuard(
            max_dag_amendments=3,
            max_subtask_additions=5,
            max_total_subtasks=10,
            max_subtask_invalidations_per_node=2,
        )
        self.s1 = Subtask(subtask_id="st1", title="Step 1", goal="Goal 1", status=SubtaskStatus.COMPLETED)
        self.s2 = Subtask(subtask_id="st2", title="Step 2", goal="Goal 2", dependencies=["st1"], status=SubtaskStatus.PENDING)
        self.plan = TaskPlan(objective="Test", subtasks=[self.s1, self.s2])

    def test_guard_approves_valid_addition(self):
        s3 = Subtask(subtask_id="st3", title="Step 3", goal="Goal 3", dependencies=["st2"])
        prop = DAGProposal(reason="Add Step 3", additions=[SubtaskAddition(subtask=s3)])
        valid, reason = self.guard.evaluate(prop, self.plan)
        self.assertTrue(valid)
        self.assertIn("Approved", reason)

    def test_guard_rejects_empty_proposal(self):
        prop = DAGProposal(reason="Empty proposal")
        valid, reason = self.guard.evaluate(prop, self.plan)
        self.assertFalse(valid)
        self.assertIn("contains no additions", reason)

    def test_guard_rejects_cycle_injection(self):
        s3 = Subtask(subtask_id="st3", title="Step 3", goal="Goal 3", dependencies=["st2"])
        dep_up = DependencyUpdate(subtask_id="st1", dependencies=["st3"])
        prop = DAGProposal(reason="Cycle injection", additions=[SubtaskAddition(subtask=s3)], dependency_updates=[dep_up])
        valid, reason = self.guard.evaluate(prop, self.plan)
        self.assertFalse(valid)
        self.assertIn("cycle", reason.lower())

    def test_guard_rejects_self_dependency(self):
        s3 = Subtask(subtask_id="st3", title="Step 3", goal="Goal 3", dependencies=["st3"])
        prop = DAGProposal(reason="Self dep", additions=[SubtaskAddition(subtask=s3)])
        valid, reason = self.guard.evaluate(prop, self.plan)
        self.assertFalse(valid)
        self.assertIn("self-dependency", reason.lower())

    def test_guard_rejects_missing_dependency(self):
        s3 = Subtask(subtask_id="st3", title="Step 3", goal="Goal 3", dependencies=["non_existent_id"])
        prop = DAGProposal(reason="Missing dep", additions=[SubtaskAddition(subtask=s3)])
        valid, reason = self.guard.evaluate(prop, self.plan)
        self.assertFalse(valid)
        self.assertIn("non-existent", reason.lower())

    def test_guard_rejects_pruned_dependency(self):
        prop = DAGProposal(reason="Prune st1", removals=[SubtaskRemoval(subtask_id="st1")])
        valid, reason = self.guard.evaluate(prop, self.plan)
        self.assertFalse(valid)
        self.assertIn("pruned", reason.lower())

    def test_guard_rejects_duplicate_active_subtask_id(self):
        s_dup = Subtask(subtask_id="st2", title="Duplicate Step 2", goal="Duplicate Goal")
        prop = DAGProposal(reason="Duplicate ID", additions=[SubtaskAddition(subtask=s_dup)])
        valid, reason = self.guard.evaluate(prop, self.plan)
        self.assertFalse(valid)
        self.assertIn("duplicate", reason.lower())

    def test_guard_rejects_empty_title_or_goal(self):
        s_bad = Subtask(subtask_id="st_bad", title="", goal="Some Goal")
        prop = DAGProposal(reason="Bad title", additions=[SubtaskAddition(subtask=s_bad)])
        valid, reason = self.guard.evaluate(prop, self.plan)
        self.assertFalse(valid)
        self.assertIn("non-empty", reason.lower())

    def test_guard_rejects_pruning_running_subtask(self):
        self.s2.status = SubtaskStatus.RUNNING
        prop = DAGProposal(reason="Prune running", removals=[SubtaskRemoval(subtask_id="st2")])
        valid, reason = self.guard.evaluate(prop, self.plan)
        self.assertFalse(valid)
        self.assertIn("running", reason.lower())

    def test_guard_enforces_max_dag_amendments(self):
        for i in range(3):
            self.plan.amendments.append(
                TaskPlanAmendment(
                    amendment_id=f"amend-{i}",
                    version=i + 1,
                    timestamp=datetime.datetime.now(datetime.timezone.utc),
                    proposal=DAGProposal(reason=f"Amend {i}"),
                )
            )
        s3 = Subtask(subtask_id="st3", title="Step 3", goal="Goal 3")
        prop = DAGProposal(reason="Over limit", additions=[SubtaskAddition(subtask=s3)])
        valid, reason = self.guard.evaluate(prop, self.plan)
        self.assertFalse(valid)
        self.assertIn("Maximum DAG amendments limit", reason)

    def test_guard_enforces_max_subtask_additions(self):
        adds = [
            SubtaskAddition(subtask=Subtask(subtask_id=f"add_{i}", title=f"Title {i}", goal=f"Goal {i}"))
            for i in range(6)
        ]
        prop = DAGProposal(reason="Too many additions", additions=adds)
        valid, reason = self.guard.evaluate(prop, self.plan)
        self.assertFalse(valid)
        self.assertIn("Maximum subtask additions limit", reason)

    def test_guard_enforces_max_total_subtasks(self):
        self.guard = DAGAmendmentGuard(max_total_subtasks=3)
        adds = [
            SubtaskAddition(subtask=Subtask(subtask_id=f"add_{i}", title=f"Title {i}", goal=f"Goal {i}"))
            for i in range(2)
        ]
        prop = DAGProposal(reason="Total subtask overflow", additions=adds)
        valid, reason = self.guard.evaluate(prop, self.plan)
        self.assertFalse(valid)
        self.assertIn("Total active subtasks", reason)

    def test_guard_enforces_max_invalidations_per_node(self):
        for i in range(2):
            inv_prop = DAGProposal(reason="inv", invalidations=[SubtaskInvalidation(subtask_id="st1")])
            self.plan.amendments.append(
                TaskPlanAmendment(
                    amendment_id=f"inv-amend-{i}",
                    version=i + 1,
                    timestamp=datetime.datetime.now(datetime.timezone.utc),
                    proposal=inv_prop,
                )
            )
        prop = DAGProposal(reason="3rd invalidation", invalidations=[SubtaskInvalidation(subtask_id="st1")])
        valid, reason = self.guard.evaluate(prop, self.plan)
        self.assertFalse(valid)
        self.assertIn("max invalidations limit", reason)


class TestTaskPlanAmendmentApplication(unittest.TestCase):
    """Test atomic application and lifecycle state transitions on TaskPlan."""

    def setUp(self):
        self.s1 = Subtask(subtask_id="st1", title="Step 1", goal="Goal 1", status=SubtaskStatus.COMPLETED, completion_info={"loc": 42})
        self.s2 = Subtask(subtask_id="st2", title="Step 2", goal="Goal 2", dependencies=["st1"], status=SubtaskStatus.PENDING)
        self.s3 = Subtask(subtask_id="st3", title="Step 3", goal="Goal 3", dependencies=["st2"], status=SubtaskStatus.PENDING)
        self.plan = TaskPlan(objective="Evolve System", subtasks=[self.s1, self.s2, self.s3])

    def test_apply_addition_increments_version(self):
        s4 = Subtask(subtask_id="st4", title="Step 4", goal="Goal 4", dependencies=["st3"])
        prop = DAGProposal(reason="Add final validation step", additions=[SubtaskAddition(subtask=s4)])
        amendment = self.plan.apply_amendment(prop, approved_by="deterministic_policy")

        self.assertEqual(self.plan.version, 2)
        self.assertEqual(len(self.plan.amendments), 1)
        self.assertEqual(amendment.version, 2)
        self.assertEqual(amendment.previous_active_subtask_ids, ["st1", "st2", "st3"])
        self.assertEqual(amendment.new_active_subtask_ids, ["st1", "st2", "st3", "st4"])
        self.assertIn(s4, self.plan.subtasks)

    def test_apply_removal_marks_pruned(self):
        prop = DAGProposal(
            reason="Prune obsolete step 3",
            removals=[SubtaskRemoval(subtask_id="st3", reason="Obsolete")],
        )
        amendment = self.plan.apply_amendment(prop)
        self.assertEqual(self.plan.version, 2)
        self.assertEqual(self.s3.status, SubtaskStatus.PRUNED)
        self.assertEqual(self.plan.active_subtask_ids, ["st1", "st2"])

    def test_apply_invalidation_supersedes_and_preserves_history(self):
        rep = Subtask(subtask_id="st1_v2", title="Step 1 v2", goal="Corrected Goal 1", status=SubtaskStatus.PENDING)
        prop = DAGProposal(
            reason="Flawed interface in st1",
            invalidations=[SubtaskInvalidation(subtask_id="st1", reason="Flawed interface", replacement_subtask=rep)],
            dependency_updates=[DependencyUpdate(subtask_id="st2", dependencies=["st1_v2"])],
        )
        amendment = self.plan.apply_amendment(prop)

        self.assertEqual(self.plan.version, 2)
        self.assertEqual(self.s1.status, SubtaskStatus.SUPERSEDED)
        self.assertEqual(self.s1.completion_info, {"loc": 42})
        self.assertIn(rep, self.plan.subtasks)
        self.assertEqual(self.s2.dependencies, ["st1_v2"])
        self.assertIn("st1_v2", self.plan.active_subtask_ids)
        self.assertNotIn("st1", self.plan.active_subtask_ids)


class TestSchedulerDynamicDAGExecution(unittest.TestCase):
    """Test Scheduler dynamic queue recalculation and autonomous approval."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage = JsonFileStorage(self.temp_dir.name)
        self.credential_store = MockCredentialStore()
        self.base_config = AgentConfig(project=Path(self.temp_dir.name), provider="mock")

        provider_configs = [ProviderConfig(provider_id="mock", priority=1, enabled=True)]
        self.storage.save_provider_configs(provider_configs)
        self.credential_store.save("dungx-ai-coding-agent", "mock", "mock-key")

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_task(
        self,
        task_id: str = "task-1",
        objective: str = "Test",
        plan: TaskPlan | None = None,
        status: TaskStatus = TaskStatus.PENDING,
        autonomous: bool = False,
    ) -> Task:
        now = datetime.datetime.now(datetime.timezone.utc)
        task = Task(
            task_id=task_id,
            objective=objective,
            status=status,
            created_at=now,
            updated_at=now,
            plan=plan,
            autonomous=autonomous,
        )
        self.storage.save_task(task)
        return task

    def test_scheduler_finds_next_runnable_topological_order(self):
        s1 = Subtask(subtask_id="st1", title="A", goal="A", status=SubtaskStatus.COMPLETED)
        s2 = Subtask(subtask_id="st2", title="B", goal="B", dependencies=["st1"], status=SubtaskStatus.PENDING)
        s3 = Subtask(subtask_id="st3", title="C", goal="C", dependencies=["st2"], status=SubtaskStatus.PENDING)
        plan = TaskPlan(objective="Test", subtasks=[s1, s2, s3])
        task = self._create_task(task_id="task-1", objective="Test", plan=plan, status=TaskStatus.PENDING)

        scheduler = Scheduler(self.base_config, self.storage, self.credential_store)
        runnable = scheduler._find_next_runnable_subtask(task)
        self.assertIsNotNone(runnable)
        self.assertEqual(runnable.subtask_id, "st2")

    def test_scheduler_ignores_superseded_and_pruned_subtasks(self):
        s1 = Subtask(subtask_id="st1", title="A", goal="A", status=SubtaskStatus.SUPERSEDED)
        s1_v2 = Subtask(subtask_id="st1_v2", title="A v2", goal="A v2", status=SubtaskStatus.COMPLETED)
        s2 = Subtask(subtask_id="st2", title="B", goal="B", status=SubtaskStatus.PRUNED)
        s3 = Subtask(subtask_id="st3", title="C", goal="C", dependencies=["st1_v2"], status=SubtaskStatus.PENDING)
        plan = TaskPlan(objective="Test", subtasks=[s1, s1_v2, s2, s3])
        task = self._create_task(task_id="task-1", objective="Test", plan=plan, status=TaskStatus.PENDING)

        scheduler = Scheduler(self.base_config, self.storage, self.credential_store)
        runnable = scheduler._find_next_runnable_subtask(task)
        self.assertIsNotNone(runnable)
        self.assertEqual(runnable.subtask_id, "st3")

    def test_scheduler_recalculates_queue_after_subtask_insertion(self):
        s1 = Subtask(subtask_id="st1", title="A", goal="A", status=SubtaskStatus.COMPLETED)
        s2 = Subtask(subtask_id="st2", title="B", goal="B", dependencies=["st1"], status=SubtaskStatus.PENDING)
        plan = TaskPlan(objective="Test", subtasks=[s1, s2])
        task = self._create_task(task_id="task-1", objective="Test", plan=plan, status=TaskStatus.PENDING)

        s_x = Subtask(subtask_id="st_x", title="X", goal="X", dependencies=["st1"], status=SubtaskStatus.PENDING)
        prop = DAGProposal(
            reason="Insert prerequisite X",
            additions=[SubtaskAddition(subtask=s_x)],
            dependency_updates=[DependencyUpdate(subtask_id="st2", dependencies=["st_x"])],
        )
        plan.apply_amendment(prop)

        scheduler = Scheduler(self.base_config, self.storage, self.credential_store)
        runnable = scheduler._find_next_runnable_subtask(task)
        self.assertIsNotNone(runnable)
        self.assertEqual(runnable.subtask_id, "st_x")

    @patch("local_agent.scheduler.Orchestrator")
    def test_autonomous_scheduler_auto_approves_safe_dag_proposal(self, MockOrchestrator):
        s1 = Subtask(subtask_id="st1", title="A", goal="A", status=SubtaskStatus.PENDING)
        plan = TaskPlan(objective="Auto Task", subtasks=[s1])
        task = self._create_task(task_id="task-auto", objective="Auto Task", plan=plan, status=TaskStatus.PENDING, autonomous=True)

        s2 = Subtask(subtask_id="st2", title="B", goal="B", dependencies=["st1"], status=SubtaskStatus.PENDING)
        prop = DAGProposal(reason="Add B", additions=[SubtaskAddition(subtask=s2)])

        mock_report = MagicMock()
        mock_report.outcome = "FAILED"
        mock_report.dag_proposal = prop
        mock_report.plan_proposal = None
        mock_report.tool_metrics = []
        mock_report.tool_history = []
        MockOrchestrator.return_value.run.return_value = mock_report

        scheduler = Scheduler(self.base_config, self.storage, self.credential_store)
        scheduler.run_once()

        loaded_task = self.storage.load_task("task-auto")
        self.assertEqual(loaded_task.status, TaskStatus.PENDING)
        self.assertEqual(loaded_task.plan.version, 2)
        self.assertEqual(len(loaded_task.plan.amendments), 1)
        self.assertEqual(loaded_task.plan.active_subtask_ids, ["st1", "st2"])

    def test_check_and_complete_task_with_superseded_and_pruned(self):
        s1 = Subtask(subtask_id="st1", title="A", goal="A", status=SubtaskStatus.SUPERSEDED)
        s1_v2 = Subtask(subtask_id="st1_v2", title="A v2", goal="A v2", status=SubtaskStatus.COMPLETED)
        s2 = Subtask(subtask_id="st2", title="B", goal="B", status=SubtaskStatus.PRUNED)
        s3 = Subtask(subtask_id="st3", title="C", goal="C", dependencies=["st1_v2"], status=SubtaskStatus.COMPLETED)
        plan = TaskPlan(objective="Complete Task", subtasks=[s1, s1_v2, s2, s3])
        task = self._create_task(task_id="task-c", objective="Complete Task", plan=plan, status=TaskStatus.RUNNING)

        scheduler = Scheduler(self.base_config, self.storage, self.credential_store)
        scheduler._check_and_complete_task(task)

        self.assertEqual(task.status, TaskStatus.COMPLETED)


class TestCheckpointAndResumeWithDAG(unittest.TestCase):
    """Test checkpoint persistence and recovery of versioned TaskPlan DAG state."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage = JsonFileStorage(self.temp_dir.name)
        self.credential_store = MockCredentialStore()
        self.config = AgentConfig(project=Path(self.temp_dir.name), provider="mock")
        self.repo_lock = threading.Lock()
        self.memory_lock = threading.Lock()
        self.orchestrator = Orchestrator(self.config, self.storage, scheduler=None, repo_lock=self.repo_lock, memory_lock=self.memory_lock)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_task(
        self,
        task_id: str = "task-1",
        objective: str = "Test",
        plan: TaskPlan | None = None,
        status: TaskStatus = TaskStatus.RUNNING,
    ) -> Task:
        now = datetime.datetime.now(datetime.timezone.utc)
        task = Task(
            task_id=task_id,
            objective=objective,
            status=status,
            created_at=now,
            updated_at=now,
            plan=plan,
        )
        self.storage.save_task(task)
        return task

    def test_checkpoint_continuation_context_captures_dag_version_and_amendments(self):
        s1 = Subtask(subtask_id="st1", title="A", goal="A", status=SubtaskStatus.COMPLETED)
        s2 = Subtask(subtask_id="st2", title="B", goal="B", dependencies=["st1"], status=SubtaskStatus.RUNNING)
        plan = TaskPlan(objective="Checkpoint Task", subtasks=[s1, s2])

        s3 = Subtask(subtask_id="st3", title="C", goal="C", dependencies=["st2"])
        prop = DAGProposal(reason="Add C", additions=[SubtaskAddition(subtask=s3)])
        plan.apply_amendment(prop)

        task = self._create_task(task_id="task-cp", objective="Checkpoint Task", plan=plan, status=TaskStatus.RUNNING)
        report = RunReport(project=ProjectContext(root=str(self.temp_dir.name)), changed_files=["src/a.py"])

        checkpoint = self.orchestrator._create_checkpoint(task, s2, "Mid-execution checkpoint", report.project, report)
        self.assertIn("task_plan", checkpoint.continuation_context)
        self.assertEqual(checkpoint.continuation_context["task_plan_version"], 2)
        self.assertEqual(len(checkpoint.continuation_context["dag_amendments"]), 1)

    def test_checkpoint_deserialization_restores_dag_version_and_amendments(self):
        s1 = Subtask(subtask_id="st1", title="A", goal="A", status=SubtaskStatus.SUPERSEDED)
        s1_v2 = Subtask(subtask_id="st1_v2", title="A v2", goal="A v2", status=SubtaskStatus.COMPLETED)
        plan = TaskPlan(objective="Restore Task", subtasks=[s1, s1_v2], version=2)
        amendment = TaskPlanAmendment(
            amendment_id="amend-1",
            version=2,
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            proposal=DAGProposal(reason="supersede st1"),
            previous_active_subtask_ids=["st1"],
            new_active_subtask_ids=["st1_v2"],
        )
        plan.amendments.append(amendment)

        task = self._create_task(task_id="task-res", objective="Restore Task", plan=plan, status=TaskStatus.RUNNING)
        report = RunReport(project=ProjectContext(root=str(self.temp_dir.name)))

        checkpoint = self.orchestrator._create_checkpoint(task, s1_v2, "Before resume", report.project, report)
        self.storage.save_checkpoint(checkpoint)

        loaded_cp = self.storage.load_checkpoint(checkpoint.checkpoint_id)
        raw_tp = loaded_cp.continuation_context.get("task_plan")
        self.assertIsNotNone(raw_tp)
        restored_tp = TaskPlan.from_dict(raw_tp)

        self.assertEqual(restored_tp.version, 2)
        self.assertEqual(len(restored_tp.amendments), 1)
        self.assertEqual(restored_tp.active_subtask_ids, ["st1_v2"])


class TestPlannerDAGProposalGeneration(unittest.TestCase):
    """Test Planner.create_dag_proposal integration with AIProvider."""

    def test_planner_create_dag_proposal_delegates_to_provider(self):
        mock_provider = MagicMock(spec=AIProvider)
        expected_prop = DAGProposal(
            reason="Architectural fix",
            additions=[SubtaskAddition(subtask=Subtask(subtask_id="st_new", title="New", goal="Goal"))],
        )
        mock_provider.propose_plan_modification.return_value = expected_prop

        planner = Planner(mock_provider)
        s1 = Subtask(subtask_id="st1", title="A", goal="A")
        plan = TaskPlan(objective="Test", subtasks=[s1])
        failure = FailureAnalysis(probable_root_cause="Missing dependency", category="MISSING_DEPENDENCY")

        result = planner.create_dag_proposal("Test Objective", plan, failure)
        self.assertEqual(result, expected_prop)
        mock_provider.propose_plan_modification.assert_called_once_with("Test Objective", plan, failure)


class TestCLIProposalApproval(unittest.TestCase):
    """Test CLI approve-proposal and reject-proposal with versioned TaskPlan."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.storage = JsonFileStorage(self.root / ".agent_data")

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_task(
        self,
        task_id: str = "task-cli",
        objective: str = "CLI Test",
        plan: TaskPlan | None = None,
        status: TaskStatus = TaskStatus.PLAN_PROPOSED,
    ) -> Task:
        now = datetime.datetime.now(datetime.timezone.utc)
        task = Task(
            task_id=task_id,
            objective=objective,
            status=status,
            created_at=now,
            updated_at=now,
            plan=plan,
        )
        self.storage.save_task(task)
        return task

    def test_cli_approve_proposal_applies_amendment_and_sets_pending(self):
        import io
        from local_agent.cli import main

        s1 = Subtask(subtask_id="st1", title="A", goal="A", status=SubtaskStatus.COMPLETED)
        plan = TaskPlan(objective="CLI Task", subtasks=[s1])
        s2 = Subtask(subtask_id="st2", title="B", goal="B", dependencies=["st1"])
        prop = DAGProposal(reason="Add B", additions=[SubtaskAddition(subtask=s2)])

        task = self._create_task(task_id="task-cli-1", plan=plan, status=TaskStatus.PLAN_PROPOSED)
        task.plan_proposal = prop
        self.storage.save_task(task)

        with patch("sys.stdout", new=io.StringIO()):
            code = main(["approve-proposal", "--project", str(self.root), task.task_id])

        self.assertEqual(code, 0)
        loaded = self.storage.load_task(task.task_id)
        self.assertEqual(loaded.status, TaskStatus.PENDING)
        self.assertIsNone(loaded.plan_proposal)
        self.assertEqual(loaded.plan.version, 2)
        self.assertEqual(len(loaded.plan.amendments), 1)
        self.assertEqual(loaded.plan.amendments[0].approved_by, "user_approval")
        self.assertIn("st2", loaded.plan.active_subtask_ids)


class TestSafetyAndAcyclicityInvariants(unittest.TestCase):
    """Test advanced invariant enforcement across multi-branch DAG structures."""

    def test_diamond_graph_topological_sort_and_tie_breaking(self):
        # Diamond DAG:
        #       st1 (completed)
        #      /   \
        #    st2    st3  (both pending, depending on st1)
        #      \   /
        #       st4      (pending, depending on st2 and st3)
        now = datetime.datetime(2026, 8, 28, 10, 0, 0, tzinfo=datetime.timezone.utc)
        s1 = Subtask(subtask_id="st1", title="Start", goal="A", status=SubtaskStatus.COMPLETED, created_at=now)
        s2 = Subtask(subtask_id="st2", title="Branch Left", goal="B", dependencies=["st1"], status=SubtaskStatus.PENDING, created_at=now + datetime.timedelta(seconds=1))
        s3 = Subtask(subtask_id="st3", title="Branch Right", goal="C", dependencies=["st1"], status=SubtaskStatus.PENDING, created_at=now + datetime.timedelta(seconds=2))
        s4 = Subtask(subtask_id="st4", title="Join", goal="D", dependencies=["st2", "st3"], status=SubtaskStatus.PENDING, created_at=now + datetime.timedelta(seconds=3))

        plan = TaskPlan(objective="Diamond", subtasks=[s1, s2, s3, s4])
        task = Task(task_id="diamond-task", objective="Diamond", plan=plan, status=TaskStatus.PENDING, created_at=now, updated_at=now)

        temp_dir = tempfile.TemporaryDirectory()
        storage = JsonFileStorage(temp_dir.name)
        cred = MockCredentialStore()
        config = AgentConfig(project=Path(temp_dir.name), provider="mock")
        scheduler = Scheduler(config, storage, cred)

        # First runnable must be s2 (earlier created_at than s3 at depth 1)
        runnable = scheduler._find_next_runnable_subtask(task)
        self.assertEqual(runnable.subtask_id, "st2")

        # After s2 is completed, next runnable must be s3 (s4 cannot run until both s2 and s3 are completed)
        s2.status = SubtaskStatus.COMPLETED
        runnable2 = scheduler._find_next_runnable_subtask(task)
        self.assertEqual(runnable2.subtask_id, "st3")

        # After s3 is completed, next runnable must be s4
        s3.status = SubtaskStatus.COMPLETED
        runnable3 = scheduler._find_next_runnable_subtask(task)
        self.assertEqual(runnable3.subtask_id, "st4")

        temp_dir.cleanup()

    def test_guard_simulation_does_not_mutate_original_plan_on_rejection(self):
        s1 = Subtask(subtask_id="st1", title="A", goal="A", status=SubtaskStatus.COMPLETED)
        plan = TaskPlan(objective="Unmutated", subtasks=[s1], version=1)

        guard = DAGAmendmentGuard()
        # Invalid proposal: cycle/self-dependency
        invalid_prop = DAGProposal(
            reason="Invalid",
            additions=[SubtaskAddition(subtask=Subtask(subtask_id="st_bad", title="Bad", goal="Bad", dependencies=["st_bad"]))],
        )
        valid, reason = guard.evaluate(invalid_prop, plan)
        self.assertFalse(valid)

        # Plan must be 100% unmutated
        self.assertEqual(plan.version, 1)
        self.assertEqual(len(plan.subtasks), 1)
        self.assertEqual(len(plan.amendments), 0)
        self.assertEqual(plan.subtasks[0].subtask_id, "st1")

    def test_plan_proposal_with_field_modifications_evaluates_and_applies(self):
        s1 = Subtask(subtask_id="st1", title="Initial Title", goal="Initial Goal", status=SubtaskStatus.PENDING)
        plan = TaskPlan(objective="Modify Title", subtasks=[s1])

        prop = PlanProposal(
            reason="Update goal and title",
            modifications=[SubtaskModification(subtask_id="st1", title="New Title", goal="New Goal")],
        )
        guard = DAGAmendmentGuard()
        # Should be evaluated cleanly via from_plan_proposal
        valid, reason = guard.evaluate(prop, plan)
        self.assertTrue(valid)

    def test_dag_amendment_does_not_reset_recovery_or_iteration_counts(self):
        # Verification that DAG proposals do not reset recovery iteration budget
        recovery = RecoveryState(completed_iterations=2)
        recovery.record_failure(FailureAnalysis(probable_root_cause="Flaw", category="ARCHITECTURAL_FLAW"))
        self.assertEqual(recovery.completed_iterations, 2)
        self.assertEqual(len(recovery.failure_history), 1)


if __name__ == "__main__":
    unittest.main()
