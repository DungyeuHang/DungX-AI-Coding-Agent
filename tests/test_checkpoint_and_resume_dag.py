from __future__ import annotations

import datetime
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from local_agent.config import AgentConfig
from local_agent.coordinator import ParallelExecutionCoordinator
from local_agent.git import GitIntegration
from local_agent.models import (
    Checkpoint,
    DAGExecutionStage,
    ExportedSymbol,
    RepositoryKnowledgeGraph,
    RunReport,
    Subtask,
    SubtaskContract,
    SubtaskStatus,
    Task,
    TaskPlan,
    TaskStatus,
    WorktreeSession,
)
from local_agent.scheduler import Scheduler
from local_agent.storage import JsonFileStorage
from local_agent.worktree import WorktreeManager


def _init_git_repo(path: Path) -> GitIntegration:
    git = GitIntegration(path)
    git._run("init")
    git._run("config", "user.name", "Test Agent")
    git._run("config", "user.email", "agent@dungx.local")
    git._run("config", "commit.gpgsign", "false")
    (path / "README.md").write_text("# Main Workspace\n", encoding="utf-8")
    git.add(["README.md"])
    git.commit("initial commit")
    return git


def _make_test_task(task_id: str, plan: TaskPlan) -> Task:
    now = datetime.datetime.now(datetime.timezone.utc)
    return Task(
        task_id=task_id,
        objective=plan.objective if plan else "Test task",
        created_at=now,
        updated_at=now,
        status=TaskStatus.PENDING,
        plan=plan,
    )


class TestCheckpointSchemaAndStorageAtomicity(unittest.TestCase):
    """Tests for Checkpoint schema versioning, fields, and storage atomicity."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.storage_dir = Path(self.temp_dir) / ".agent_data"
        self.storage = JsonFileStorage(self.storage_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_checkpoint_schema_version_and_dag_stage(self):
        cp = Checkpoint(
            checkpoint_id="cp-test-1",
            task_id="task-1",
            subtask_id="sub-1",
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            current_state_description="Testing schema",
            schema_version="4.15.0",
            dag_stage=DAGExecutionStage.WORKER_COMMITTED.value,
            subtask_states={"sub-1": "completed"},
            integrated_subtasks=["sub-1"],
            verified_subtasks=["sub-1"],
            promoted_subtasks=["sub-1"],
            base_commit="abc1234",
            integration_commit="def5678",
        )
        self.storage.save_checkpoint(cp)

        loaded = self.storage.load_checkpoint("cp-test-1")
        self.assertEqual(loaded.schema_version, "4.15.0")
        self.assertEqual(loaded.dag_stage, DAGExecutionStage.WORKER_COMMITTED.value)
        self.assertEqual(loaded.subtask_states, {"sub-1": "completed"})
        self.assertEqual(loaded.integrated_subtasks, ["sub-1"])
        self.assertEqual(loaded.verified_subtasks, ["sub-1"])
        self.assertEqual(loaded.promoted_subtasks, ["sub-1"])
        self.assertEqual(loaded.base_commit, "abc1234")
        self.assertEqual(loaded.integration_commit, "def5678")

    def test_load_latest_checkpoint_and_list_checkpoints(self):
        now = datetime.datetime.now(datetime.timezone.utc)
        cp1 = Checkpoint(
            checkpoint_id="cp-1",
            task_id="task-1",
            subtask_id="sub-1",
            timestamp=now - datetime.timedelta(seconds=10),
            current_state_description="first",
        )
        cp2 = Checkpoint(
            checkpoint_id="cp-2",
            task_id="task-1",
            subtask_id="sub-1",
            timestamp=now,
            current_state_description="second",
        )
        cp_other = Checkpoint(
            checkpoint_id="cp-3",
            task_id="task-other",
            subtask_id="sub-x",
            timestamp=now,
            current_state_description="other task",
        )
        self.storage.save_checkpoint(cp1)
        self.storage.save_checkpoint(cp2)
        self.storage.save_checkpoint(cp_other)

        cps = self.storage.list_checkpoints_for_task("task-1")
        self.assertEqual(len(cps), 2)
        self.assertEqual(cps[0].checkpoint_id, "cp-1")
        self.assertEqual(cps[1].checkpoint_id, "cp-2")

        latest = self.storage.load_latest_checkpoint("task-1")
        self.assertIsNotNone(latest)
        self.assertEqual(latest.checkpoint_id, "cp-2")

    def test_checkpoint_backward_compatibility_with_missing_fields(self):
        legacy_data = {
            "checkpoint_id": "cp-legacy",
            "task_id": "task-legacy",
            "subtask_id": "sub-legacy",
            "timestamp": "2026-08-30T12:00:00+00:00",
            "current_state_description": "legacy checkpoint",
        }
        cp = Checkpoint.from_dict(legacy_data)
        self.assertEqual(cp.schema_version, "4.15.0")
        self.assertEqual(cp.dag_stage, "init")
        self.assertEqual(cp.subtask_states, {})
        self.assertEqual(cp.integrated_subtasks, [])
        self.assertEqual(cp.verified_subtasks, [])


class TestCrashBoundaryRecovery(unittest.TestCase):
    """
    Exhaustive validation of all crash boundaries (Case A through Case J).
    Uses real Git repositories and isolated worktrees.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(self.temp_dir) / "repo"
        self.project_root.mkdir(parents=True, exist_ok=True)
        self.git = _init_git_repo(self.project_root)

        self.storage_dir = Path(self.temp_dir) / "storage"
        self.storage = JsonFileStorage(self.storage_dir)

        self.config = AgentConfig(
            project=self.project_root,
            provider="mock",
            parallel_worktree_execution=True,
            max_parallel_subtasks=2,
            knowledge_graph_enabled=True,
        )
        self.worktree_manager = WorktreeManager(self.project_root, self.git)
        self.coordinator = ParallelExecutionCoordinator(
            self.config,
            self.storage,
            worktree_manager=self.worktree_manager,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_case_a_crash_before_worker_execution(self):
        """Case A: Process crashed before any worker ran. Subtasks remain PENDING and execute cleanly."""
        s1 = Subtask(subtask_id="sub-a1", title="Sub A1", status=SubtaskStatus.PENDING)
        s2 = Subtask(subtask_id="sub-a2", title="Sub A2", status=SubtaskStatus.PENDING)
        task = _make_test_task("task-case-a", TaskPlan(objective="Case A Goal", subtasks=[s1, s2]))
        self.storage.save_task(task)

        # Simulate crash before worker execution: task status was marked RUNNING
        task.status = TaskStatus.RUNNING
        s1.status = SubtaskStatus.RUNNING
        self.storage.save_task(task)

        # Reconcile and resume
        reconciled_task, report = self.coordinator.reconcile_dag_state(task)
        self.assertEqual(s1.status, SubtaskStatus.PENDING)
        self.assertIn("subtask_sub-a1_stale_running_reset", report["actions"])

    def test_case_b_crash_during_worker_execution_cleans_incomplete_worktree(self):
        """Case B: Worker was executing in worktree, crashed midway without commits."""
        s1 = Subtask(subtask_id="sub-b1", title="Sub B1", status=SubtaskStatus.PENDING)
        task = _make_test_task("task-case-b", TaskPlan(objective="Case B Goal", subtasks=[s1]))
        self.storage.save_task(task)

        # Create worktree representing in-flight worker
        session = self.worktree_manager.create_worktree(task.task_id, s1.subtask_id)
        s1.worktree_session = session
        s1.status = SubtaskStatus.RUNNING
        self.storage.save_task(task)

        # Worktree exists on disk
        self.assertTrue(Path(session.worktree_path).exists())

        # Reconcile and resume
        reconciled_task, report = self.coordinator.reconcile_dag_state(task)
        self.assertEqual(s1.status, SubtaskStatus.PENDING)
        self.assertIn("subtask_sub-b1_incomplete_worker_reset", report["actions"])
        # Worktree directory was cleaned up
        self.assertFalse(Path(session.worktree_path).exists())

    def test_case_c_worker_completed_dirty_worktree_uncommitted(self):
        """Case C: Worker modified tracked files, but crashed before committing."""
        s1 = Subtask(subtask_id="sub-c1", title="Feature C1", status=SubtaskStatus.RUNNING)
        task = _make_test_task("task-case-c", TaskPlan(objective="Case C Goal", subtasks=[s1]))
        self.storage.save_task(task)

        session = self.worktree_manager.create_worktree(task.task_id, s1.subtask_id)
        s1.worktree_session = session

        # Worker modified a tracked file in the worktree (README.md exists from init)
        wt_path = Path(session.worktree_path)
        (wt_path / "README.md").write_text("# Modified by worker\ndef feature_c1(): return 42\n", encoding="utf-8")

        # Reconcile: coordinator should detect dirty worktree, commit tracked changes, advance to COMPLETED
        reconciled_task, report = self.coordinator.reconcile_dag_state(task)
        self.assertEqual(s1.status, SubtaskStatus.COMPLETED)
        self.assertIn("subtask_sub-c1_dirty_worktree_committed", report["actions"])

        # Subtask branch has the commit
        branch_sha = self.git.get_branch_commit(session.branch_name)
        self.assertIsNotNone(branch_sha)

    def test_case_d_crash_after_worker_commit_before_checkpoint(self):
        """Case D: Worker committed to branch, but process crashed before task save/checkpoint."""
        s1 = Subtask(subtask_id="sub-d1", title="Feature D1", status=SubtaskStatus.PENDING)
        task = _make_test_task("task-case-d", TaskPlan(objective="Case D Goal", subtasks=[s1]))
        self.storage.save_task(task)

        # Worker created worktree and committed
        session = self.worktree_manager.create_worktree(task.task_id, s1.subtask_id)
        wt_path = Path(session.worktree_path)
        (wt_path / "module_d1.py").write_text("def module_d(): pass\n", encoding="utf-8")
        wt_git = GitIntegration(wt_path)
        wt_git.add(["module_d1.py"])
        wt_git.commit("feat(sub-d1): Module D1")

        # Stale state on disk: subtask still marked PENDING
        s1.status = SubtaskStatus.PENDING
        s1.worktree_session = session
        self.storage.save_task(task)

        # Reconcile: Git oracle discovers commit on branch ahead of base_commit
        reconciled_task, report = self.coordinator.reconcile_dag_state(task)
        self.assertEqual(s1.status, SubtaskStatus.COMPLETED)
        self.assertIn("subtask_sub-d1_worker_commit_recovered", report["actions"])

    def test_case_e_crash_during_integration_active_merge_aborts_cleanly(self):
        """Case E: Process crashed during git merge, leaving active merge in progress."""
        s1 = Subtask(subtask_id="sub-e1", title="Feature E1", status=SubtaskStatus.PENDING)
        task = _make_test_task("task-case-e", TaskPlan(objective="Case E Goal", subtasks=[s1]))
        self.storage.save_task(task)

        # Create worktree session from initial base commit
        session = self.worktree_manager.create_worktree(task.task_id, s1.subtask_id)

        # Create divergent change on integration branch
        integration_branch = f"agent-task/{task.task_id}"
        self.git.ensure_branch(integration_branch)
        (self.project_root / "shared.txt").write_text("Version Main\n", encoding="utf-8")
        self.git.add(["shared.txt"])
        self.git.commit("change on main")

        # Create conflicting change on worktree branch
        wt_path = Path(session.worktree_path)
        (wt_path / "shared.txt").write_text("Version Branch E1\n", encoding="utf-8")
        wt_git = GitIntegration(wt_path)
        wt_git.add(["shared.txt"])
        wt_git.commit("conflicting change")

        # Trigger merge conflict to simulate in-flight merge crash
        self.git.checkout(integration_branch)
        success, out = self.git.merge_branch(session.branch_name)
        self.assertFalse(success)
        # Main repo now has merge in progress
        self.assertTrue(self.git.is_merge_in_progress())

        # Reconcile: coordinator detects active merge and safely aborts it
        reconciled_task, report = self.coordinator.reconcile_dag_state(task)
        self.assertFalse(self.git.is_merge_in_progress())
        self.assertIn("aborted_active_merge", report["actions"])

    def test_case_f_crash_after_integration_before_verification(self):
        """Case F: Subtask branch is merged, but verification didn't run. Resume detects merge and verifies."""
        sym = ExportedSymbol(symbol_id="fn_f", name="fn_f", kind="function", file_path="mod_f.py", verified=True)
        contract = SubtaskContract(subtask_id="sub-f1", title="Feature F", modified_files=["mod_f.py"], exported_symbols=[sym])
        s1 = Subtask(subtask_id="sub-f1", title="Feature F", contract=contract, status=SubtaskStatus.PENDING)
        task = _make_test_task("task-case-f", TaskPlan(objective="Case F Goal", subtasks=[s1]))
        self.storage.save_task(task)

        integration_branch = f"agent-task/{task.task_id}"
        self.git.ensure_branch(integration_branch)

        session = self.worktree_manager.create_worktree(task.task_id, s1.subtask_id)
        wt_path = Path(session.worktree_path)
        (wt_path / "mod_f.py").write_text("def fn_f(): pass\n", encoding="utf-8")
        wt_git = GitIntegration(wt_path)
        wt_git.add(["mod_f.py"])
        wt_git.commit("feat(sub-f1): add mod_f")

        # Merge branch into integration_branch
        self.git.checkout(integration_branch)
        self.git.merge_branch(session.branch_name, message="merge sub-f1")

        # Reconcile: detects branch already integrated (is_ancestor=True)
        reconciled_task, report = self.coordinator.reconcile_dag_state(task)
        self.assertEqual(s1.status, SubtaskStatus.COMPLETED)
        self.assertIsNotNone(s1.integration_commit)
        self.assertIn("subtask_sub-f1_already_integrated", report["actions"])

    def test_case_g_verification_passed_before_kg_promotion(self):
        """Case G: Tier-2 verification passed, promote contracts to Knowledge Graph idempotently."""
        sym = ExportedSymbol(symbol_id="calc_total", name="calc_total", kind="function", file_path="calc.py", verified=True)
        contract = SubtaskContract(subtask_id="sub-g1", title="Calc", modified_files=["calc.py"], exported_symbols=[sym])
        s1 = Subtask(subtask_id="sub-g1", title="Calc", contract=contract, status=SubtaskStatus.COMPLETED)
        task = _make_test_task("task-case-g", TaskPlan(objective="Calc", subtasks=[s1]))
        self.storage.save_task(task)

        # Run verify_integration
        v_ok = self.coordinator.verify_integration(task, [s1])
        self.assertTrue(v_ok)

        # Authoritative graph has the symbol
        kg = self.storage.load_knowledge_graph()
        self.assertIn("calc_total", kg.symbols)

        # Promote again (idempotent)
        v_ok2 = self.coordinator.verify_integration(task, [s1])
        self.assertTrue(v_ok2)
        kg2 = self.storage.load_knowledge_graph()
        self.assertEqual(len(kg2.symbols), len(kg.symbols))

    def test_case_h_and_i_cleanup_is_safe_and_idempotent(self):
        """Case H & I: Worktree cleanup is safe and idempotent across repeated calls."""
        s1 = Subtask(subtask_id="sub-h1", title="Cleanup H", status=SubtaskStatus.COMPLETED)
        session = self.worktree_manager.create_worktree("task-h", s1.subtask_id)
        s1.worktree_session = session
        task = _make_test_task("task-h", TaskPlan(objective="Cleanup", subtasks=[s1]))

        self.assertTrue(Path(session.worktree_path).exists())

        # First cleanup
        self.coordinator.worktree_manager.remove_worktree(session, force=True)
        self.assertFalse(Path(session.worktree_path).exists())

        # Second cleanup (idempotent, no error)
        self.coordinator.worktree_manager.remove_worktree(session, force=True)

    def test_case_j_completed_task_is_terminal_and_never_reexecuted(self):
        """Case J: Fully completed task remains COMPLETED and is never re-executed."""
        s1 = Subtask(subtask_id="sub-j1", title="Done J", status=SubtaskStatus.COMPLETED, integration_commit="sha123")
        task = _make_test_task("task-j", TaskPlan(objective="Done", subtasks=[s1]))
        task.status = TaskStatus.COMPLETED
        self.storage.save_task(task)

        with patch.object(self.coordinator, "execute_parallel_batch") as mock_exec:
            resumed = self.coordinator.reconcile_and_resume(task)
            self.assertEqual(resumed.status, TaskStatus.COMPLETED)
            mock_exec.assert_not_called()


class TestAdversarialAndMutationResilience(unittest.TestCase):
    """
    Adversarial red-team scenarios, mutation testing, and multi-call idempotency.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(self.temp_dir) / "repo"
        self.project_root.mkdir(parents=True, exist_ok=True)
        self.git = _init_git_repo(self.project_root)

        self.storage_dir = Path(self.temp_dir) / "storage"
        self.storage = JsonFileStorage(self.storage_dir)

        self.config = AgentConfig(
            project=self.project_root,
            provider="mock",
            parallel_worktree_execution=True,
            max_parallel_subtasks=2,
            knowledge_graph_enabled=True,
        )
        self.worktree_manager = WorktreeManager(self.project_root, self.git)
        self.coordinator = ParallelExecutionCoordinator(
            self.config,
            self.storage,
            worktree_manager=self.worktree_manager,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_resume_idempotency_triple_call(self):
        """Calling reconcile_and_resume 3 consecutive times produces identical state without duplicate merges."""
        s1 = Subtask(subtask_id="sub-idemp-1", title="Idemp 1", status=SubtaskStatus.PENDING)
        s2 = Subtask(subtask_id="sub-idemp-2", title="Idemp 2", status=SubtaskStatus.PENDING)
        task = _make_test_task("task-idemp", TaskPlan(objective="Idempotency Test", subtasks=[s1, s2]))
        self.storage.save_task(task)

        # Worker mock completes both
        with patch.object(self.coordinator, "execute_parallel_batch") as mock_exec, \
             patch.object(self.coordinator, "integrate_branches") as mock_merge, \
             patch.object(self.coordinator, "verify_integration", return_value=True):
            
            s1_done = Subtask(subtask_id="sub-idemp-1", status=SubtaskStatus.COMPLETED, integration_commit="commit-1")
            s2_done = Subtask(subtask_id="sub-idemp-2", status=SubtaskStatus.COMPLETED, integration_commit="commit-2")
            mock_exec.return_value = [
                (s1_done, MagicMock(completed=True), None),
                (s2_done, MagicMock(completed=True), None),
            ]
            mock_merge.return_value = ([s1_done, s2_done], [])

            res1 = self.coordinator.reconcile_and_resume(task)
            self.assertEqual(res1.status, TaskStatus.COMPLETED)

            # Call 2
            res2 = self.coordinator.reconcile_and_resume(res1)
            self.assertEqual(res2.status, TaskStatus.COMPLETED)

            # Call 3
            res3 = self.coordinator.reconcile_and_resume(res2)
            self.assertEqual(res3.status, TaskStatus.COMPLETED)

            # Execution was only called on the first pass
            self.assertEqual(mock_exec.call_count, 1)

    def test_git_oracle_overrides_stale_checkpoint(self):
        """When checkpoint on disk disagrees with Git commit reality, Git reality wins."""
        s1 = Subtask(subtask_id="sub-stale", title="Stale Sub", status=SubtaskStatus.PENDING)
        task = _make_test_task("task-stale", TaskPlan(objective="Stale Checkpoint", subtasks=[s1]))
        self.storage.save_task(task)

        session = self.worktree_manager.create_worktree(task.task_id, s1.subtask_id)
        wt_path = Path(session.worktree_path)
        (wt_path / "valid_code.py").write_text("x = 100\n", encoding="utf-8")
        wt_git = GitIntegration(wt_path)
        wt_git.add(["valid_code.py"])
        wt_git.commit("feat(sub-stale): add valid code")

        # Fake a stale checkpoint stating subtask is PENDING
        cp = Checkpoint(
            checkpoint_id="cp-stale",
            task_id=task.task_id,
            subtask_id="sub-stale",
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            current_state_description="stale pending state",
            subtask_states={"sub-stale": "pending"},
        )
        self.storage.save_checkpoint(cp)

        # Reconcile: Git oracle discovers commit and sets COMPLETED
        reconciled_task, report = self.coordinator.reconcile_dag_state(task)
        self.assertEqual(s1.status, SubtaskStatus.COMPLETED)
        self.assertIn("subtask_sub-stale_worker_commit_recovered", report["actions"])

    def test_protected_files_invariance(self):
        """Protected files must remain strictly unmodified."""
        engine_path = Path("local_agent/tool_engine.py")
        approval_path = Path("local_agent/approval.py")

        self.assertTrue(engine_path.exists())
        self.assertTrue(approval_path.exists())

        # Verify git status of protected files is clean
        repo_git = GitIntegration(".")
        if repo_git.is_repository():
            diff = repo_git._run("diff", "--", "local_agent/tool_engine.py", "local_agent/approval.py")
            self.assertEqual(diff, "", "Protected files must have 0 diff!")


class TestDAGTopologyCrashAndResume(unittest.TestCase):
    """
    Multi-subtask Diamond DAG execution, crash injection at intermediate stages,
    and verification that recovery preserves dependency order and parallel concurrency.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(self.temp_dir) / "repo"
        self.project_root.mkdir(parents=True, exist_ok=True)
        self.git = _init_git_repo(self.project_root)

        self.storage_dir = Path(self.temp_dir) / "storage"
        self.storage = JsonFileStorage(self.storage_dir)

        self.config = AgentConfig(
            project=self.project_root,
            provider="mock",
            parallel_worktree_execution=True,
            max_parallel_subtasks=2,
            knowledge_graph_enabled=True,
        )
        self.worktree_manager = WorktreeManager(self.project_root, self.git)
        self.coordinator = ParallelExecutionCoordinator(
            self.config,
            self.storage,
            worktree_manager=self.worktree_manager,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_diamond_dag_crash_after_first_tier_resumes_second_tier_in_parallel(self):
        """
        Diamond DAG: A -> B, A -> C, B -> D, C -> D.
        A completes and integrates. Crash occurs.
        On resume: B and C run in parallel, then D runs and completes in a single resume loop.
        """
        sa = Subtask(subtask_id="sa", title="Root A", status=SubtaskStatus.COMPLETED, integration_commit="commit-a")
        sb = Subtask(subtask_id="sb", title="Branch B", dependencies=["sa"], status=SubtaskStatus.PENDING)
        sc = Subtask(subtask_id="sc", title="Branch C", dependencies=["sa"], status=SubtaskStatus.PENDING)
        sd = Subtask(subtask_id="sd", title="Join D", dependencies=["sb", "sc"], status=SubtaskStatus.PENDING)

        task = _make_test_task("task-diamond", TaskPlan(objective="Diamond Goal", subtasks=[sa, sb, sc, sd]))
        self.storage.save_task(task)

        with patch.object(self.coordinator, "execute_parallel_batch") as mock_exec, \
             patch.object(self.coordinator, "integrate_branches") as mock_merge, \
             patch.object(self.coordinator, "verify_integration", return_value=True):

            sb_done = Subtask(subtask_id="sb", status=SubtaskStatus.COMPLETED, integration_commit="commit-b")
            sc_done = Subtask(subtask_id="sc", status=SubtaskStatus.COMPLETED, integration_commit="commit-c")
            sd_done = Subtask(subtask_id="sd", status=SubtaskStatus.COMPLETED, integration_commit="commit-d")

            mock_exec.side_effect = [
                [
                    (sb_done, MagicMock(completed=True), None),
                    (sc_done, MagicMock(completed=True), None),
                ],
                [
                    (sd_done, MagicMock(completed=True), None),
                ],
            ]
            mock_merge.side_effect = [
                ([sb_done, sc_done], []),
                ([sd_done], []),
            ]

            res = self.coordinator.reconcile_and_resume(task)
            self.assertEqual(res.status, TaskStatus.COMPLETED)
            self.assertEqual(mock_exec.call_count, 2)
            first_batch = mock_exec.call_args_list[0][0][1]
            self.assertEqual({s.subtask_id for s in first_batch}, {"sb", "sc"})
            second_batch = mock_exec.call_args_list[1][0][1]
            self.assertEqual([s.subtask_id for s in second_batch], ["sd"])

    def test_diamond_dag_crash_with_one_unintegrated_branch_recovers_and_integrates(self):
        """
        Subtask B is committed to branch but not merged when crash happened.
        Subtask C is unexecuted.
        On resume: B is integrated without re-running worker, C is executed, then D is executed.
        """
        sa = Subtask(subtask_id="sa", title="Root A", status=SubtaskStatus.COMPLETED, integration_commit="commit-a")
        sb = Subtask(subtask_id="sb", title="Branch B", dependencies=["sa"], status=SubtaskStatus.PENDING)
        sc = Subtask(subtask_id="sc", title="Branch C", dependencies=["sa"], status=SubtaskStatus.PENDING)

        task = _make_test_task("task-unintegrated", TaskPlan(objective="Unintegrated Recovery", subtasks=[sa, sb, sc]))
        self.storage.save_task(task)

        # Worker B created worktree and committed
        session_b = self.worktree_manager.create_worktree(task.task_id, sb.subtask_id)
        wt_path_b = Path(session_b.worktree_path)
        (wt_path_b / "mod_b.py").write_text("def b(): pass\n", encoding="utf-8")
        wt_git_b = GitIntegration(wt_path_b)
        wt_git_b.add(["mod_b.py"])
        wt_git_b.commit("feat(sb): module b")

        # Checkpoint says sb is completed, but integration_commit is None
        sb.status = SubtaskStatus.COMPLETED
        sb.worktree_session = session_b
        self.storage.save_task(task)

        # When reconcile runs: sb is discovered, merged into integration_branch, and sc executes
        with patch.object(self.coordinator, "execute_parallel_batch") as mock_exec_c, \
             patch.object(self.coordinator, "verify_integration", return_value=True):

            sc_done = Subtask(subtask_id="sc", status=SubtaskStatus.COMPLETED)
            mock_exec_c.return_value = [(sc_done, MagicMock(completed=True), None)]

            res = self.coordinator.reconcile_and_resume(task)
            # Both sb and sc were processed
            loaded = self.storage.load_task("task-unintegrated")
            sb_final = next(s for s in loaded.plan.subtasks if s.subtask_id == "sb")
            self.assertEqual(sb_final.status, SubtaskStatus.COMPLETED)
            self.assertIsNotNone(sb_final.integration_commit)


class TestStorageAtomicWriteFailureAndRecovery(unittest.TestCase):
    """Verifies atomic write failure safety and cleanup."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.storage_dir = Path(self.temp_dir) / ".agent_data"
        self.storage = JsonFileStorage(self.storage_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_atomic_write_leaves_target_intact_if_write_fails(self):
        target = self.storage._task_path("task-atomic")
        target.write_text('{"initial": true}', encoding="utf-8")

        # Attempt atomic write with unserializable object
        class Unserializable:
            pass

        with self.assertRaises(Exception):
            self.storage._atomic_write(target, {"bad": Unserializable()})

        # Target remains intact
        content = json.loads(target.read_text(encoding="utf-8"))
        self.assertTrue(content.get("initial"))

        # No .tmp file left behind
        tmp_file = target.with_suffix(".json.tmp")
        self.assertFalse(tmp_file.exists())

class TestForensicAuditP0SafetyFixes(unittest.TestCase):
    """
    Forensic audit tests for P0 safety defects discovered during
    Phase 4.15 independent review.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(self.temp_dir) / "repo"
        self.project_root.mkdir(parents=True, exist_ok=True)
        self.git = _init_git_repo(self.project_root)

        self.storage_dir = Path(self.temp_dir) / "storage"
        self.storage = JsonFileStorage(self.storage_dir)

        self.config = AgentConfig(
            project=self.project_root,
            provider="mock",
            parallel_worktree_execution=True,
            max_parallel_subtasks=2,
            knowledge_graph_enabled=True,
        )
        self.worktree_manager = WorktreeManager(self.project_root, self.git)
        self.coordinator = ParallelExecutionCoordinator(
            self.config,
            self.storage,
            worktree_manager=self.worktree_manager,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_dirty_worktree_with_only_untracked_files_must_not_auto_commit(self):
        """
        P0-1: If a worktree contains ONLY untracked files (not tracked-dirty),
        recovery must NOT auto-commit them. Untracked files could be user-owned
        or foreign artifacts.
        """
        s1 = Subtask(subtask_id="sub-untracked", title="Untracked Test", status=SubtaskStatus.RUNNING)
        task = _make_test_task("task-untracked", TaskPlan(objective="Untracked Test", subtasks=[s1]))
        self.storage.save_task(task)

        session = self.worktree_manager.create_worktree(task.task_id, s1.subtask_id)
        s1.worktree_session = session

        # Drop an untracked file into the worktree — but don't modify any tracked file
        wt_path = Path(session.worktree_path)
        (wt_path / "foreign_artifact.dat").write_text("foreign data\n", encoding="utf-8")

        # The worktree has untracked files but no tracked-dirty files,
        # so git diff --name-only will be empty. Recovery should NOT auto-commit.
        reconciled_task, report = self.coordinator.reconcile_dag_state(task)

        # Subtask must NOT be marked COMPLETED since only untracked files exist
        self.assertNotEqual(s1.status, SubtaskStatus.COMPLETED)
        self.assertNotIn("subtask_sub-untracked_dirty_worktree_committed", report["actions"])

    def test_dirty_worktree_with_protected_file_changes_must_be_refused(self):
        """
        P0-1: If a worktree has dirty changes touching protected files
        (tool_engine.py, approval.py), recovery MUST refuse auto-commit.
        """
        # Create protected file in base repo first so it is tracked in base_commit
        (self.project_root / "tool_engine.py").write_text("# base protected\n", encoding="utf-8")
        self.git.add(["tool_engine.py"])
        self.git.commit("add base tool_engine")

        s1 = Subtask(subtask_id="sub-protected", title="Protected Test", status=SubtaskStatus.RUNNING)
        task = _make_test_task("task-protected", TaskPlan(objective="Protected Test", subtasks=[s1]))
        self.storage.save_task(task)

        session = self.worktree_manager.create_worktree(task.task_id, s1.subtask_id)
        s1.worktree_session = session

        # Modify the protected file in the worktree without committing
        wt_path = Path(session.worktree_path)
        (wt_path / "tool_engine.py").write_text("# MODIFIED BY WORKER\n", encoding="utf-8")

        reconciled_task, report = self.coordinator.reconcile_dag_state(task)

        # Recovery must REFUSE to auto-commit
        self.assertNotEqual(s1.status, SubtaskStatus.COMPLETED)
        self.assertIn("subtask_sub-protected_dirty_worktree_protected_files_refused", report["actions"])

    def test_foreign_merge_on_user_branch_must_not_be_aborted(self):
        """
        P0-2: If the user has an active merge on a non-DungX branch,
        recovery must NOT abort it.
        """
        base_head = self.git.get_head_commit()
        s1 = Subtask(subtask_id="sub-fm", title="Foreign Merge", status=SubtaskStatus.PENDING)
        task = _make_test_task("task-fm", TaskPlan(objective="Foreign Merge Test", subtasks=[s1]))
        self.storage.save_task(task)

        # Create user branch with change from base_head
        self.git.ensure_branch("user-feature", start_point=base_head)
        (self.project_root / "user_file.txt").write_text("user version\n", encoding="utf-8")
        self.git.add(["user_file.txt"])
        self.git.commit("user commit")

        # Create another branch from base_head with conflicting change
        self.git.ensure_branch("user-other", start_point=base_head)
        (self.project_root / "user_file.txt").write_text("other version\n", encoding="utf-8")
        self.git.add(["user_file.txt"])
        self.git.commit("other commit")

        # Start merge on user branch (conflict)
        self.git.checkout("user-feature")
        success, _ = self.git.merge_branch("user-other")
        self.assertFalse(success)
        self.assertTrue(self.git.is_merge_in_progress())

        # DungX reconcile must NOT abort the user's merge
        reconciled_task, report = self.coordinator.reconcile_dag_state(task)
        self.assertTrue(self.git.is_merge_in_progress())
        self.assertIn("skipped_foreign_active_merge", report["actions"])
        self.assertNotIn("aborted_active_merge", report["actions"])

        # Clean up the merge for tearDown
        self.git.merge_abort()

    def test_checkpoint_says_completed_but_git_branch_missing_must_not_fabricate(self):
        """
        Git Oracle Case 2: Checkpoint says worker committed, but Git branch does not exist.
        Recovery must NOT fabricate success.
        """
        s1 = Subtask(subtask_id="sub-ghost", title="Ghost Branch", status=SubtaskStatus.COMPLETED)
        s1.worktree_session = WorktreeSession(
            session_id="wt-ghost",
            subtask_id="sub-ghost",
            worktree_path=str(self.worktree_manager.allocate_worktree_path("task-ghost", "sub-ghost")),
            branch_name="agent/task_ghost/sub_ghost",
            base_commit="nonexistent",
            status="active",
        )
        task = _make_test_task("task-ghost", TaskPlan(objective="Ghost Branch", subtasks=[s1]))
        self.storage.save_task(task)

        # The branch does NOT exist in Git
        branch_commit = self.git.get_branch_commit("agent/task_ghost/sub_ghost")
        self.assertIsNone(branch_commit)

        # Reconcile: subtask was COMPLETED but has no Git evidence
        reconciled_task, report = self.coordinator.reconcile_dag_state(task)

        # Subtask must NOT gain an integration_commit fabricated from nothing
        self.assertIsNone(s1.integration_commit)

    def test_checkpoint_says_not_integrated_but_branch_is_ancestor(self):
        """
        Git Oracle Case 3: Checkpoint says not integrated, but the branch
        commit is already an ancestor of the integration branch.
        Expected: recognize as already integrated without merging again.
        """
        s1 = Subtask(subtask_id="sub-already", title="Already Merged", status=SubtaskStatus.PENDING)
        task = _make_test_task("task-already", TaskPlan(objective="Already Integrated", subtasks=[s1]))
        self.storage.save_task(task)

        integration_branch = f"agent-task/{task.task_id}"
        self.git.ensure_branch(integration_branch)

        session = self.worktree_manager.create_worktree(task.task_id, s1.subtask_id)
        wt_path = Path(session.worktree_path)
        (wt_path / "already.py").write_text("x = 1\n", encoding="utf-8")
        wt_git = GitIntegration(wt_path)
        wt_git.add(["already.py"])
        wt_git.commit("already merged work")

        # Merge into integration branch
        self.git.checkout(integration_branch)
        self.git.merge_branch(session.branch_name, message="merge already")

        # Subtask is still PENDING in checkpoint
        s1.worktree_session = session
        self.storage.save_task(task)

        reconciled_task, report = self.coordinator.reconcile_dag_state(task)
        self.assertEqual(s1.status, SubtaskStatus.COMPLETED)
        self.assertIsNotNone(s1.integration_commit)
        self.assertIn("subtask_sub-already_already_integrated", report["actions"])

    def test_multi_tier_dag_completes_in_single_resume_call(self):
        """
        P1-1: Diamond DAG A→B,C→D must complete all tiers in a single
        reconcile_and_resume call (the while-loop fix).
        """
        sa = Subtask(subtask_id="sa", title="Root A", status=SubtaskStatus.COMPLETED, integration_commit="commit-a")
        sb = Subtask(subtask_id="sb", title="Branch B", dependencies=["sa"], status=SubtaskStatus.PENDING)
        sc = Subtask(subtask_id="sc", title="Branch C", dependencies=["sa"], status=SubtaskStatus.PENDING)
        sd = Subtask(subtask_id="sd", title="Join D", dependencies=["sb", "sc"], status=SubtaskStatus.PENDING)

        task = _make_test_task("task-multi-tier", TaskPlan(objective="Multi-tier", subtasks=[sa, sb, sc, sd]))
        self.storage.save_task(task)

        call_count = [0]

        with patch.object(self.coordinator, "execute_parallel_batch") as mock_exec, \
             patch.object(self.coordinator, "integrate_branches") as mock_merge, \
             patch.object(self.coordinator, "verify_integration", return_value=True):

            def side_effect_exec(task_obj, batch, **kwargs):
                call_count[0] += 1
                results = []
                for s in batch:
                    s_done = Subtask(subtask_id=s.subtask_id, status=SubtaskStatus.COMPLETED, integration_commit=f"commit-{s.subtask_id}")
                    results.append((s_done, MagicMock(completed=True), None))
                return results

            def side_effect_merge(task_obj, subtasks, branch):
                for s in subtasks:
                    s.integration_commit = f"commit-{s.subtask_id}"
                return (subtasks, [])

            mock_exec.side_effect = side_effect_exec
            mock_merge.side_effect = side_effect_merge

            result = self.coordinator.reconcile_and_resume(task)

            # Must have completed ALL subtasks including D
            self.assertEqual(result.status, TaskStatus.COMPLETED)
            # execute_parallel_batch must be called at least twice:
            # once for [B,C] and once for [D]
            self.assertGreaterEqual(mock_exec.call_count, 2)


if __name__ == "__main__":
    unittest.main()
