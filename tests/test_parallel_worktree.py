from __future__ import annotations

import datetime
import shutil
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from local_agent.config import AgentConfig
from local_agent.coordinator import ParallelExecutionCoordinator
from local_agent.credentials import MockCredentialStore
from local_agent.git import GitIntegration
from local_agent.knowledge import KnowledgeGraphManager
from local_agent.models import (
    Checkpoint,
    ExportedSymbol,
    Plan,
    ProjectContext,
    ProviderConfig,
    RepositoryKnowledgeGraph,
    Subtask,
    SubtaskContract,
    SubtaskStatus,
    Task,
    TaskPlan,
    TaskStatus,
    WorktreeSession,
)
from local_agent.scheduler import Scheduler
from local_agent.storage import JsonFileStorage, TaskStorage
from local_agent.worktree import WorktreeManager


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def _init_git_repo(root: Path) -> str:
    """Creates a real single-commit Git repository and returns its HEAD sha."""
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.email", "agent@test.local")
    _git(root, "config", "user.name", "agent")
    _git(root, "config", "commit.gpgsign", "false")
    (root / "shared.py").write_text("BASE = 1\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-m", "base")
    return _git(root, "rev-parse", "HEAD").stdout.strip()


def _make_test_task(task_id: str, plan: TaskPlan) -> Task:
    now = datetime.datetime.now(datetime.timezone.utc)
    return Task(
        task_id=task_id,
        objective=plan.objective,
        status=TaskStatus.PENDING,
        created_at=now,
        updated_at=now,
        plan=plan,
    )


class TestWorktreeLifecycle(unittest.TestCase):
    """Unit tests for WorktreeManager path allocation, creation, and cleanup."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.repo_root = Path(self.temp_dir) / "repo"
        self.repo_root.mkdir()
        self.git = GitIntegration(self.repo_root)
        self.manager = WorktreeManager(self.repo_root, self.git)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_allocate_worktree_path_deterministic(self):
        p1 = self.manager.allocate_worktree_path("task-123", "sub-abc")
        p2 = self.manager.allocate_worktree_path("task-123", "sub-abc")
        self.assertEqual(p1, p2)
        self.assertTrue(p1.as_posix().endswith(".agent_worktrees/task-task-123/sub-sub-abc"))

    def test_allocate_worktree_path_prevents_escape(self):
        with self.assertRaises(ValueError):
            self.manager.allocate_worktree_path("../../../etc", "passwd")

    def test_branch_name_for_subtask_sanitization(self):
        branch = self.manager.branch_name_for_subtask("task:100", "sub/50")
        self.assertEqual(branch, "agent/task_100/sub_50")

    def test_create_and_remove_worktree_cleanly(self):
        session = self.manager.create_worktree("task-1", "sub-1")
        self.assertIsInstance(session, WorktreeSession)
        self.assertEqual(session.status, "active")
        self.assertTrue(Path(session.worktree_path).exists())

        # Removal
        removed = self.manager.remove_worktree(session, force=True)
        self.assertTrue(removed)
        self.assertFalse(Path(session.worktree_path).exists())
        self.assertEqual(session.status, "cleaned")

    def test_prune_and_cleanup_stale_worktrees(self):
        s1 = self.manager.create_worktree("task-1", "sub-1")
        s2 = self.manager.create_worktree("task-1", "sub-2")

        # Keep s1 active, s2 stale
        cleaned = self.manager.cleanup_stale_worktrees(active_session_paths={s1.worktree_path})
        self.assertTrue(Path(s1.worktree_path).exists())

    def test_cleanup_all_worktrees(self):
        s1 = self.manager.create_worktree("task-1", "sub-1")
        s2 = self.manager.create_worktree("task-1", "sub-2")
        self.manager.cleanup_all()
        self.assertFalse(Path(s1.worktree_path).exists())
        self.assertFalse(Path(s2.worktree_path).exists())

    def test_remove_worktree_idempotent(self):
        session = self.manager.create_worktree("task-1", "sub-1")
        self.assertTrue(self.manager.remove_worktree(session))
        # Second call on already cleaned session should be harmless True
        self.assertTrue(self.manager.remove_worktree(session))


class TestWorktreeSessionModel(unittest.TestCase):
    """Unit tests for WorktreeSession data structure and serialization."""

    def test_worktree_session_serialization_roundtrip(self):
        now = datetime.datetime.now(datetime.timezone.utc)
        session = WorktreeSession(
            session_id="wt-123",
            subtask_id="sub-1",
            worktree_path="/repo/.agent_worktrees/task-1/sub-1",
            branch_name="agent/task-1/sub-1",
            base_commit="abc12345",
            created_at=now,
            status="active",
        )
        d = session.to_dict()
        self.assertEqual(d["session_id"], "wt-123")
        self.assertEqual(d["subtask_id"], "sub-1")
        self.assertEqual(d["status"], "active")

        loaded = WorktreeSession.from_dict(d)
        self.assertEqual(loaded.session_id, "wt-123")
        self.assertEqual(loaded.subtask_id, "sub-1")
        self.assertEqual(loaded.worktree_path, "/repo/.agent_worktrees/task-1/sub-1")
        self.assertEqual(loaded.branch_name, "agent/task-1/sub-1")
        self.assertEqual(loaded.base_commit, "abc12345")
        self.assertEqual(loaded.status, "active")

    def test_subtask_with_worktree_session_serialization(self):
        session = WorktreeSession(
            session_id="wt-456",
            subtask_id="sub-2",
            worktree_path="/path",
            branch_name="agent/t/s",
            base_commit="c123",
            status="merged",
        )
        subtask = Subtask(
            subtask_id="sub-2",
            title="Sub 2",
            worktree_session=session,
            integration_commit="commit-789",
        )
        d = subtask.to_dict()
        self.assertIn("worktree_session", d)
        self.assertEqual(d["worktree_session"]["session_id"], "wt-456")
        self.assertEqual(d["integration_commit"], "commit-789")

        loaded = Subtask.from_dict(d)
        self.assertIsNotNone(loaded.worktree_session)
        self.assertEqual(loaded.worktree_session.session_id, "wt-456")
        self.assertEqual(loaded.integration_commit, "commit-789")


class TestGitWorktreePrimitives(unittest.TestCase):
    """Unit tests for GitIntegration worktree and merge primitives."""

    def test_worktree_list_porcelain_parsing(self):
        git = GitIntegration(".")
        sample_porcelain = (
            "worktree /repo/main\n"
            "HEAD 1234567890abcdef\n"
            "branch refs/heads/main\n\n"
            "worktree /repo/.agent_worktrees/task-1/sub-1\n"
            "HEAD fedcba0987654321\n"
            "branch refs/heads/agent/task-1/sub-1\n"
            "locked\n"
        )
        with patch.object(git, "_run", return_value=sample_porcelain):
            parsed = git.worktree_list()
            self.assertEqual(len(parsed), 2)
            self.assertEqual(parsed[0]["worktree"], "/repo/main")
            self.assertEqual(parsed[0]["branch"], "refs/heads/main")
            self.assertEqual(parsed[1]["worktree"], "/repo/.agent_worktrees/task-1/sub-1")
            self.assertEqual(parsed[1]["locked"], "true")

    def test_merge_branch_and_abort(self):
        git = GitIntegration(".")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="Merge made by 'ort' strategy.", stderr="")
            success, output = git.merge_branch("agent/task-1/sub-1", message="merge message")
            self.assertTrue(success)
            self.assertIn("Merge made", output)

        with patch.object(git, "_run_for_exit_code", return_value=0):
            self.assertTrue(git.merge_abort())

    def test_rebase_branch(self):
        git = GitIntegration(".")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="Successfully rebased.", stderr="")
            success, output = git.rebase_branch("main")
            self.assertTrue(success)
            self.assertIn("rebased", output)


class TestDAGSchedulingAndReadiness(unittest.TestCase):
    """Unit tests for runnable subtask discovery across DAG dependencies."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = AgentConfig(project=Path(self.temp_dir), provider="mock")
        self.storage = JsonFileStorage(Path(self.temp_dir) / "storage")
        self.coordinator = ParallelExecutionCoordinator(self.config, self.storage)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_identify_runnable_subtasks_single_ready_root(self):
        s0 = Subtask(subtask_id="sub-0", title="Root", dependencies=[], status=SubtaskStatus.PENDING)
        s1 = Subtask(subtask_id="sub-1", title="Child", dependencies=["sub-0"], status=SubtaskStatus.PENDING)
        task = _make_test_task("t1", TaskPlan(objective="test", subtasks=[s0, s1]))

        runnable = self.coordinator.identify_runnable_subtasks(task)
        self.assertEqual(len(runnable), 1)
        self.assertEqual(runnable[0].subtask_id, "sub-0")

    def test_identify_runnable_subtasks_multiple_independent_ready(self):
        s0 = Subtask(subtask_id="sub-0", title="Root", dependencies=[], status=SubtaskStatus.COMPLETED)
        s1 = Subtask(subtask_id="sub-1", title="Branch A", dependencies=["sub-0"], status=SubtaskStatus.PENDING)
        s2 = Subtask(subtask_id="sub-2", title="Branch B", dependencies=["sub-0"], status=SubtaskStatus.PENDING)
        s3 = Subtask(subtask_id="sub-3", title="Join", dependencies=["sub-1", "sub-2"], status=SubtaskStatus.PENDING)
        task = _make_test_task("t1", TaskPlan(objective="test", subtasks=[s0, s1, s2, s3]))

        runnable = self.coordinator.identify_runnable_subtasks(task)
        self.assertEqual(len(runnable), 2)
        runnable_ids = {s.subtask_id for s in runnable}
        self.assertEqual(runnable_ids, {"sub-1", "sub-2"})

    def test_identify_runnable_subtasks_ignores_superseded_and_pruned(self):
        s0 = Subtask(subtask_id="sub-0", title="Old", dependencies=[], status=SubtaskStatus.SUPERSEDED)
        s1 = Subtask(subtask_id="sub-1", title="Pruned", dependencies=[], status=SubtaskStatus.PRUNED)
        s2 = Subtask(subtask_id="sub-2", title="Valid", dependencies=[], status=SubtaskStatus.PENDING)
        task = _make_test_task("t1", TaskPlan(objective="test", subtasks=[s0, s1, s2]))

        runnable = self.coordinator.identify_runnable_subtasks(task)
        self.assertEqual(len(runnable), 1)
        self.assertEqual(runnable[0].subtask_id, "sub-2")


class TestConflictPredictionAndPartitioning(unittest.TestCase):
    """Unit tests for conflict prediction and batch partitioning."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = AgentConfig(project=Path(self.temp_dir), provider="mock", max_parallel_subtasks=3)
        self.storage = JsonFileStorage(Path(self.temp_dir) / "storage")
        self.coordinator = ParallelExecutionCoordinator(self.config, self.storage)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_predict_file_conflicts_from_subtasks(self):
        s1 = Subtask(subtask_id="s1", title="Update auth", acceptance_criteria=["Modify auth.py"])
        s2 = Subtask(subtask_id="s2", title="Update ui", acceptance_criteria=["Modify ui.py"])
        predicted = self.coordinator.predict_file_conflicts([s1, s2])
        self.assertIn("auth.py", predicted["s1"])
        self.assertIn("ui.py", predicted["s2"])

    def test_partition_disjoint_subtasks_into_single_parallel_batch(self):
        s1 = Subtask(subtask_id="s1", title="Update auth", acceptance_criteria=["Modify auth.py"])
        s2 = Subtask(subtask_id="s2", title="Update billing", acceptance_criteria=["Modify billing.py"])
        s3 = Subtask(subtask_id="s3", title="Update ui", acceptance_criteria=["Modify ui.py"])

        predicted = {
            "s1": {"auth.py"},
            "s2": {"billing.py"},
            "s3": {"ui.py"},
        }
        batches = self.coordinator.partition_parallel_batches([s1, s2, s3], predicted, serialize_overlapping=True)
        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0]), 3)

    def test_partition_overlapping_subtasks_into_serialized_batches(self):
        s1 = Subtask(subtask_id="s1", title="Feature A", acceptance_criteria=["Modify models.py"])
        s2 = Subtask(subtask_id="s2", title="Feature B", acceptance_criteria=["Modify models.py and utils.py"])
        s3 = Subtask(subtask_id="s3", title="Feature C", acceptance_criteria=["Modify helper.py"])

        predicted = {
            "s1": {"models.py"},
            "s2": {"models.py", "utils.py"},
            "s3": {"helper.py"},
        }
        batches = self.coordinator.partition_parallel_batches([s1, s2, s3], predicted, serialize_overlapping=True)
        self.assertEqual(len(batches), 2)
        # Batch 1 contains s1 and s3 (disjoint)
        self.assertEqual({s.subtask_id for s in batches[0]}, {"s1", "s3"})
        # Batch 2 contains s2
        self.assertEqual({s.subtask_id for s in batches[1]}, {"s2"})

    def test_partition_respects_max_workers_bound(self):
        self.coordinator.max_workers = 2
        s1 = Subtask(subtask_id="s1", title="A")
        s2 = Subtask(subtask_id="s2", title="B")
        s3 = Subtask(subtask_id="s3", title="C")

        predicted = {"s1": {"a.py"}, "s2": {"b.py"}, "s3": {"c.py"}}
        batches = self.coordinator.partition_parallel_batches([s1, s2, s3], predicted, serialize_overlapping=True)
        self.assertEqual(len(batches), 2)
        self.assertEqual(len(batches[0]), 2)
        self.assertEqual(len(batches[1]), 1)

    def test_partition_without_serialization_when_disabled(self):
        s1 = Subtask(subtask_id="s1", title="A")
        s2 = Subtask(subtask_id="s2", title="B")
        s3 = Subtask(subtask_id="s3", title="C")
        predicted = {"s1": {"shared.py"}, "s2": {"shared.py"}, "s3": {"shared.py"}}
        batches = self.coordinator.partition_parallel_batches([s1, s2, s3], predicted, serialize_overlapping=False)
        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0]), 3)


class TestParallelExecutionAndIntegration(unittest.TestCase):
    """Integration tests for parallel worktree execution, merging, and verification."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = AgentConfig(
            project=Path(self.temp_dir),
            provider="mock",
            parallel_worktree_execution=True,
            max_parallel_subtasks=2,
        )
        self.storage = JsonFileStorage(Path(self.temp_dir) / "storage")
        self.coordinator = ParallelExecutionCoordinator(self.config, self.storage)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_execute_parallel_batch_concurrent_subtasks(self):
        s1 = Subtask(subtask_id="sub-1", title="Subtask 1", goal="Goal 1")
        s2 = Subtask(subtask_id="sub-2", title="Subtask 2", goal="Goal 2")
        task = _make_test_task("task-123", TaskPlan(objective="test", subtasks=[s1, s2]))

        with patch.object(self.coordinator, "execute_subtask_in_worktree") as mock_exec:
            mock_exec.side_effect = [
                (Subtask(subtask_id="sub-1", status=SubtaskStatus.COMPLETED), MagicMock(completed=True), None),
                (Subtask(subtask_id="sub-2", status=SubtaskStatus.COMPLETED), MagicMock(completed=True), None),
            ]
            results = self.coordinator.execute_parallel_batch(task, [s1, s2])
            self.assertEqual(len(results), 2)
            self.assertEqual(mock_exec.call_count, 2)

    def test_single_worker_failure_does_not_halt_other_workers(self):
        s1 = Subtask(subtask_id="sub-1", title="Subtask 1")
        s2 = Subtask(subtask_id="sub-2", title="Subtask 2")
        task = _make_test_task("task-123", TaskPlan(objective="test", subtasks=[s1, s2]))

        with patch.object(self.coordinator, "execute_subtask_in_worktree") as mock_exec:
            mock_exec.side_effect = [
                (Subtask(subtask_id="sub-1", status=SubtaskStatus.FAILED), None, RuntimeError("Worker failed")),
                (Subtask(subtask_id="sub-2", status=SubtaskStatus.COMPLETED), MagicMock(completed=True), None),
            ]
            results = self.coordinator.execute_parallel_batch(task, [s1, s2])
            self.assertEqual(len(results), 2)
            statuses = {s.status for s, rep, err in results}
            self.assertIn(SubtaskStatus.FAILED, statuses)
            self.assertIn(SubtaskStatus.COMPLETED, statuses)

    def test_tier2_integration_verification_and_knowledge_promotion(self):
        contract = SubtaskContract(
            subtask_id="sub-1",
            title="Sub 1",
            modified_files=["app.py"],
            exported_symbols=[ExportedSymbol(symbol_id="sym1", name="hello", kind="function", file_path="app.py", verified=True)],
        )
        s1 = Subtask(subtask_id="sub-1", title="Sub 1", contract=contract)
        task = _make_test_task("task-1", TaskPlan(objective="test", subtasks=[s1]))

        # Run verify_integration
        verified = self.coordinator.verify_integration(task, [s1])
        self.assertTrue(verified)

        # Check knowledge graph updated
        kg = self.storage.load_knowledge_graph()
        self.assertIn("sym1", kg.symbols)

    def test_failed_merge_aborts_cleanly_without_corrupting_tree(self):
        s1 = Subtask(
            subtask_id="sub-1",
            title="Sub 1",
            worktree_session=WorktreeSession(
                session_id="wt-1",
                subtask_id="sub-1",
                worktree_path="/tmp/path",
                branch_name="agent/t/s",
                base_commit="abc",
            ),
        )
        task = _make_test_task("task-1", TaskPlan(objective="test", subtasks=[s1]))

        with patch.object(self.coordinator.git, "is_repository", return_value=True), \
             patch.object(self.coordinator.git, "ensure_branch", return_value=True), \
             patch.object(self.coordinator.git, "merge_branch", return_value=(False, "CONFLICT")), \
             patch.object(self.coordinator.git, "merge_abort", return_value=True) as mock_abort:
            merged, failed = self.coordinator.integrate_branches(task, [s1], "agent-task/task-1")
            self.assertEqual(len(failed), 1)
            self.assertEqual(mock_abort.call_count, 1)


class TestSchedulerParallelIntegration(unittest.TestCase):
    """Integration tests for Scheduler interaction with parallel worktree execution."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(self.temp_dir) / "proj"
        self.project_root.mkdir()
        # Parallel execution is only offered in a real Git repository, so the
        # scheduler integration fixture must be one.
        _init_git_repo(self.project_root)
        self.storage = JsonFileStorage(Path(self.temp_dir) / "storage")
        self.storage.save_provider_configs([ProviderConfig(provider_id="mock", priority=10, enabled=True)])
        self.credential_store = MockCredentialStore()
        self.credential_store.save("dungx-ai-coding-agent", "mock", "mock_key")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_scheduler_runs_parallel_worktree_execution_when_enabled(self):
        config = AgentConfig(
            project=self.project_root,
            provider="mock",
            parallel_worktree_execution=True,
            max_parallel_subtasks=2,
        )
        scheduler = Scheduler(config, self.storage, self.credential_store)

        s1 = Subtask(subtask_id="s1", title="Sub 1", status=SubtaskStatus.PENDING)
        s2 = Subtask(subtask_id="s2", title="Sub 2", status=SubtaskStatus.PENDING)
        task = _make_test_task("task-parallel", TaskPlan(objective="Parallel goal", subtasks=[s1, s2]))
        self.storage.save_task(task)

        s1_done = Subtask(subtask_id="s1", status=SubtaskStatus.COMPLETED, integration_commit="commit-1")
        s2_done = Subtask(subtask_id="s2", status=SubtaskStatus.COMPLETED, integration_commit="commit-2")

        with patch.object(scheduler.coordinator, "execute_parallel_batch") as mock_exec, \
             patch.object(scheduler.coordinator, "integrate_branches", return_value=([s1_done, s2_done], [])), \
             patch.object(scheduler.coordinator, "verify_integration", return_value=True):
            mock_exec.return_value = [
                (s1_done, MagicMock(completed=True), None),
                (s2_done, MagicMock(completed=True), None),
            ]
            scheduler.run_once()
            self.assertEqual(mock_exec.call_count, 1)

    def test_scheduler_runs_serial_when_parallel_disabled(self):
        config = AgentConfig(
            project=self.project_root,
            provider="mock",
            parallel_worktree_execution=False,
            max_parallel_subtasks=1,
        )
        scheduler = Scheduler(config, self.storage, self.credential_store)

        s1 = Subtask(subtask_id="s1", title="Sub 1", status=SubtaskStatus.PENDING)
        task = _make_test_task("task-serial", TaskPlan(objective="Serial goal", subtasks=[s1]))
        self.storage.save_task(task)

        with patch("local_agent.scheduler.Orchestrator") as mock_orch_cls:
            mock_orch = MagicMock()
            mock_orch.run.return_value = MagicMock(outcome="SUCCESS")
            mock_orch_cls.return_value = mock_orch

            scheduler.run_once()
            self.assertEqual(mock_orch.run.call_count, 1)


class TestKnowledgeGraphParallelIsolation(unittest.TestCase):
    """Integration tests for Knowledge Graph safety and isolation during parallel execution."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(self.temp_dir) / "proj"
        self.project_root.mkdir()
        self.storage = JsonFileStorage(Path(self.temp_dir) / "storage")
        self.config = AgentConfig(project=self.project_root, provider="mock", parallel_worktree_execution=True)
        self.coordinator = ParallelExecutionCoordinator(self.config, self.storage)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_worktree_knowledge_reads_do_not_corrupt_central_graph(self):
        # Save initial clean graph
        initial_graph = RepositoryKnowledgeGraph(repo_id="repo-main")
        self.storage.save_knowledge_graph(initial_graph)

        # Worker reads graph
        loaded = self.storage.load_knowledge_graph()
        self.assertEqual(loaded.repo_id, "repo-main")

    def test_integration_sync_authoritative_hashes(self):
        # Add contract with symbol
        sym = ExportedSymbol(symbol_id="auth_login", name="login", kind="function", file_path="auth.py", verified=True)
        contract = SubtaskContract(subtask_id="sub-auth", title="Auth", modified_files=["auth.py"], exported_symbols=[sym])
        sub = Subtask(subtask_id="sub-auth", title="Auth", contract=contract)
        task = _make_test_task("task-kg", TaskPlan(objective="Auth", subtasks=[sub]))

        self.coordinator.verify_integration(task, [sub])
        kg = self.storage.load_knowledge_graph()
        self.assertIn("auth_login", kg.symbols)


class TestCheckpointAndResumeWithWorktrees(unittest.TestCase):
    """Unit tests for Checkpoint active worktrees persistence and recovery."""

    def test_checkpoint_roundtrip_with_active_worktrees(self):
        cp = Checkpoint(
            checkpoint_id="cp-1",
            task_id="task-1",
            subtask_id="sub-1",
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            current_state_description="Parallel checkpoint",
            active_worktrees=[{"subtask_id": "sub-1", "worktree_path": ".agent_worktrees/task-1/sub-1"}],
            integration_branch="agent-task/task-1",
        )
        d = cp.to_dict()
        self.assertEqual(len(d["active_worktrees"]), 1)
        self.assertEqual(d["integration_branch"], "agent-task/task-1")

        loaded = Checkpoint.from_dict(d)
        self.assertEqual(loaded.checkpoint_id, "cp-1")
        self.assertEqual(len(loaded.active_worktrees), 1)
        self.assertEqual(loaded.integration_branch, "agent-task/task-1")


class TestConfigurationAndBackwardCompatibility(unittest.TestCase):
    """Unit tests for Phase 4.14 configuration and serial execution compatibility."""

    def test_default_config_parallel_worktrees_is_false(self):
        config = AgentConfig(project=Path("."))
        self.assertFalse(config.parallel_worktree_execution)
        self.assertEqual(config.max_parallel_subtasks, 1)
        self.assertTrue(config.serialize_overlapping_subtasks)

    def test_from_environment_overrides_parallel_settings(self):
        config = AgentConfig.from_environment(
            project=Path("."),
            parallel_worktree_execution=True,
            max_parallel_subtasks=3,
            serialize_overlapping_subtasks=False,
        )
        self.assertTrue(config.parallel_worktree_execution)
        self.assertEqual(config.max_parallel_subtasks, 3)
        self.assertFalse(config.serialize_overlapping_subtasks)

    def test_validate_max_parallel_subtasks_bounds(self):
        config = AgentConfig(project=Path("."), max_parallel_subtasks=0)
        with self.assertRaises(ValueError):
            config.validate()

        config2 = AgentConfig(project=Path("."), max_parallel_subtasks=5)
        with self.assertRaises(ValueError):
            config2.validate()

        config3 = AgentConfig(project=Path("."), max_parallel_subtasks=4)
        config3.validate()  # valid

    def test_legacy_task_storage_subclass_compatibility(self):
        class LegacyStorage(TaskStorage):
            def save_task(self, task): pass
            def load_task(self, task_id): pass
            def list_tasks(self): return []
            def save_checkpoint(self, checkpoint): pass
            def load_checkpoint(self, checkpoint_id): pass
            def save_scheduler_state(self, state): pass
            def load_scheduler_state(self): pass
            def save_provider_configs(self, configs): pass
            def load_provider_configs(self): return []
            def save_semantic_index(self, index): pass
            def load_semantic_index(self): pass
            def save_project_memory(self, memory): pass
            def load_project_memory(self): pass

        storage = LegacyStorage()
        kg = storage.load_knowledge_graph()
        self.assertIsInstance(kg, RepositoryKnowledgeGraph)


class TestRealGitWorktreeIsolation(unittest.TestCase):
    """Phase 4.14 audit: worktree isolation against a real Git repository."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.repo = Path(self.temp_dir) / "repo"
        self.repo.mkdir()
        self.base_sha = _init_git_repo(self.repo)
        self.git = GitIntegration(self.repo)
        self.manager = WorktreeManager(self.repo, self.git)

    def tearDown(self):
        try:
            self.manager.cleanup_all()
        except Exception:
            pass
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_worktrees_are_real_git_checkouts_on_distinct_branches(self):
        s1 = self.manager.create_worktree("task-1", "sub-1")
        s2 = self.manager.create_worktree("task-1", "sub-2")

        for session in (s1, s2):
            path = Path(session.worktree_path)
            # A linked worktree has a .git *file* pointing at the main repo.
            self.assertTrue((path / ".git").is_file())
            self.assertEqual(
                _git(path, "branch", "--show-current").stdout.strip(),
                session.branch_name,
            )
            self.assertEqual(_git(path, "rev-parse", "HEAD").stdout.strip(), self.base_sha)
            self.assertEqual(session.base_commit, self.base_sha)
            # The worktree is a full checkout of the base commit, not an empty dir.
            self.assertTrue((path / "shared.py").exists())

        self.assertNotEqual(s1.branch_name, s2.branch_name)

    def test_writes_in_one_worktree_are_invisible_to_others_and_to_main(self):
        s1 = self.manager.create_worktree("task-1", "sub-1")
        s2 = self.manager.create_worktree("task-1", "sub-2")
        (Path(s1.worktree_path) / "shared.py").write_text("MUTATED = 1\n", encoding="utf-8")

        self.assertEqual((self.repo / "shared.py").read_text(encoding="utf-8"), "BASE = 1\n")
        self.assertEqual(
            (Path(s2.worktree_path) / "shared.py").read_text(encoding="utf-8"), "BASE = 1\n"
        )


class TestIntegrationBranchTopology(unittest.TestCase):
    """Phase 4.14 audit: subtask branches must never land on the user's branch."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.repo = Path(self.temp_dir) / "repo"
        self.repo.mkdir()
        self.base_sha = _init_git_repo(self.repo)
        self.storage = JsonFileStorage(Path(self.temp_dir) / "storage")
        self.config = AgentConfig(
            project=self.repo,
            provider="mock",
            parallel_worktree_execution=True,
            max_parallel_subtasks=2,
        )
        self.coordinator = ParallelExecutionCoordinator(self.config, self.storage)

    def tearDown(self):
        try:
            self.coordinator.worktree_manager.cleanup_all()
        except Exception:
            pass
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _worked_subtask(self, sid: str, filename: str, content: str) -> Subtask:
        session = self.coordinator.worktree_manager.create_worktree("task-1", sid)
        wt = Path(session.worktree_path)
        (wt / filename).write_text(content, encoding="utf-8")
        _git(wt, "add", "-A")
        _git(wt, "commit", "-m", f"work {sid}")
        sub = Subtask(subtask_id=sid, title=sid, status=SubtaskStatus.COMPLETED)
        sub.worktree_session = session
        return sub

    def test_merges_land_on_integration_branch_not_on_main(self):
        sub = self._worked_subtask("sub-1", "feature_a.py", "A = 1\n")
        task = _make_test_task("task-1", TaskPlan(objective="t", subtasks=[sub]))
        main_before = _git(self.repo, "rev-parse", "main").stdout.strip()

        merged, failed = self.coordinator.integrate_branches(task, [sub], "agent-task/task-1")

        self.assertEqual([s.subtask_id for s in merged], ["sub-1"])
        self.assertEqual(failed, [])
        # HEAD moved onto the integration branch, and main is untouched.
        self.assertEqual(_git(self.repo, "branch", "--show-current").stdout.strip(), "agent-task/task-1")
        self.assertEqual(_git(self.repo, "rev-parse", "main").stdout.strip(), main_before)
        self.assertIsNotNone(merged[0].integration_commit)

    def test_integration_is_refused_when_branch_cannot_be_checked_out(self):
        sub = self._worked_subtask("sub-1", "feature_a.py", "A = 1\n")
        task = _make_test_task("task-1", TaskPlan(objective="t", subtasks=[sub]))

        with patch.object(self.coordinator.git, "ensure_branch", return_value=False), \
             patch.object(self.coordinator.git, "merge_branch") as mock_merge:
            merged, failed = self.coordinator.integrate_branches(task, [sub], "agent-task/task-1")

        mock_merge.assert_not_called()
        self.assertEqual(merged, [])
        self.assertEqual([s.subtask_id for s in failed], ["sub-1"])

    def test_real_merge_conflict_aborts_and_leaves_repository_usable(self):
        a = self._worked_subtask("sub-a", "shared.py", "VERSION = 'a'\n")
        b = self._worked_subtask("sub-b", "shared.py", "VERSION = 'b'\n")
        task = _make_test_task("task-1", TaskPlan(objective="t", subtasks=[a, b]))

        merged, failed = self.coordinator.integrate_branches(task, [a, b], "agent-task/task-1")

        self.assertEqual([s.subtask_id for s in merged], ["sub-a"])
        self.assertEqual([s.subtask_id for s in failed], ["sub-b"])
        self.assertEqual(failed[0].status, SubtaskStatus.PAUSED)
        # No merge left in progress, and the tree is clean and readable.
        self.assertFalse((self.repo / ".git" / "MERGE_HEAD").exists())
        # No conflict markers and no half-merged index left behind (the only
        # untracked entry is the worktrees directory itself).
        residue = [
            line for line in _git(self.repo, "status", "--porcelain").stdout.splitlines()
            if ".agent_worktrees" not in line
        ]
        self.assertEqual(residue, [])
        self.assertNotEqual(_git(self.repo, "rev-parse", "HEAD").stdout.strip(), "")
        self.assertNotIn("<<<<<<<", (self.repo / "shared.py").read_text(encoding="utf-8"))

    def test_non_git_workspace_reports_no_integration_rather_than_false_success(self):
        plain_dir = Path(self.temp_dir) / "plain"
        plain_dir.mkdir()
        config = AgentConfig(project=plain_dir, provider="mock", parallel_worktree_execution=True)
        coordinator = ParallelExecutionCoordinator(config, self.storage)
        sub = Subtask(subtask_id="s1", title="s1", status=SubtaskStatus.COMPLETED)
        task = _make_test_task("task-x", TaskPlan(objective="t", subtasks=[sub]))

        merged, failed = coordinator.integrate_branches(task, [sub], "agent-task/task-x")

        self.assertEqual(merged, [])
        self.assertEqual([s.subtask_id for s in failed], ["s1"])


class TestIntegrationDeterminismAndPersistence(unittest.TestCase):
    """Phase 4.14 audit: ordering must follow the DAG, and outcomes must persist."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.repo = Path(self.temp_dir) / "repo"
        self.repo.mkdir()
        _init_git_repo(self.repo)
        self.storage = JsonFileStorage(Path(self.temp_dir) / "storage")
        self.config = AgentConfig(project=self.repo, provider="mock", max_parallel_subtasks=2)
        self.coordinator = ParallelExecutionCoordinator(self.config, self.storage)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _subtask(self, sid: str, finished_at: datetime.datetime, deps=None) -> Subtask:
        sub = Subtask(
            subtask_id=sid,
            title=sid,
            status=SubtaskStatus.COMPLETED,
            dependencies=list(deps or []),
        )
        sub.created_at = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)
        sub.completed_at = finished_at
        sub.worktree_session = WorktreeSession(
            session_id=f"wt-{sid}", subtask_id=sid, worktree_path="/x",
            branch_name=f"agent/task-1/{sid}", base_commit="c",
        )
        return sub

    def _merge_order_for(self, alpha_done, beta_done) -> list[str]:
        alpha = self._subtask("alpha", alpha_done)
        beta = self._subtask("beta", beta_done)
        task = _make_test_task("task-1", TaskPlan(objective="t", subtasks=[alpha, beta]))
        order: list[str] = []

        def record(branch, message=""):
            order.append(branch.rsplit("/", 1)[-1])
            return True, "ok"

        with patch.object(self.coordinator.git, "is_repository", return_value=True), \
             patch.object(self.coordinator.git, "ensure_branch", return_value=True), \
             patch.object(self.coordinator.git, "get_head_commit", return_value="deadbeef"), \
             patch.object(self.coordinator.git, "merge_branch", side_effect=record):
            self.coordinator.integrate_branches(task, [alpha, beta], "agent-task/task-1")
        return order

    def test_integration_order_is_independent_of_worker_finish_time(self):
        t0 = datetime.datetime(2024, 6, 1, tzinfo=datetime.timezone.utc)
        late = t0 + datetime.timedelta(seconds=90)
        # Run 1: alpha finishes first. Run 2: beta finishes first.
        run1 = self._merge_order_for(t0, late)
        run2 = self._merge_order_for(late, t0)
        self.assertEqual(run1, run2)
        self.assertEqual(run1, ["alpha", "beta"])

    def test_integration_order_follows_dependency_depth(self):
        root = self._subtask("z-root", datetime.datetime(2024, 6, 1, tzinfo=datetime.timezone.utc))
        child = self._subtask(
            "a-child", datetime.datetime(2024, 5, 1, tzinfo=datetime.timezone.utc), deps=["z-root"]
        )
        task = _make_test_task("task-1", TaskPlan(objective="t", subtasks=[root, child]))
        order: list[str] = []

        with patch.object(self.coordinator.git, "is_repository", return_value=True), \
             patch.object(self.coordinator.git, "ensure_branch", return_value=True), \
             patch.object(self.coordinator.git, "get_head_commit", return_value="deadbeef"), \
             patch.object(
                 self.coordinator.git, "merge_branch",
                 side_effect=lambda branch, message="": (order.append(branch.rsplit("/", 1)[-1]), (True, "ok"))[1],
             ):
            self.coordinator.integrate_branches(task, [child, root], "agent-task/task-1")

        # The dependency root integrates before its dependent, despite finishing later.
        self.assertEqual(order, ["z-root", "a-child"])

    def test_merge_conflict_outcome_is_persisted_so_work_is_not_silently_lost(self):
        good = self._subtask("sub-ok", datetime.datetime(2024, 6, 1, tzinfo=datetime.timezone.utc))
        bad = self._subtask("sub-bad", datetime.datetime(2024, 6, 2, tzinfo=datetime.timezone.utc))
        task = _make_test_task("task-1", TaskPlan(objective="t", subtasks=[good, bad]))
        self.storage.save_task(task)

        good.integration_commit = "abc123"
        bad.status = SubtaskStatus.PAUSED
        self.coordinator.persist_integration_state("task-1", [good], [bad])

        reloaded = self.storage.load_task("task-1")
        by_id = {s.subtask_id: s for s in reloaded.plan.subtasks}
        self.assertEqual(by_id["sub-ok"].status, SubtaskStatus.COMPLETED)
        self.assertEqual(by_id["sub-ok"].integration_commit, "abc123")
        # The conflicted subtask is NOT left as COMPLETED on disk.
        self.assertEqual(by_id["sub-bad"].status, SubtaskStatus.PAUSED)

    def test_task_integration_branch_survives_serialization(self):
        task = _make_test_task("task-1", TaskPlan(objective="t", subtasks=[]))
        task.integration_branch = "agent-task/task-1"
        reloaded = Task.from_dict(task.to_dict())
        self.assertEqual(reloaded.integration_branch, "agent-task/task-1")


class TestWorkerConcurrencyAndStateIsolation(unittest.TestCase):
    """Phase 4.14 audit: real overlap, and no lost updates between workers."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.repo = Path(self.temp_dir) / "repo"
        self.repo.mkdir()
        _init_git_repo(self.repo)
        self.storage = JsonFileStorage(Path(self.temp_dir) / "storage")
        self.config = AgentConfig(
            project=self.repo, provider="mock",
            parallel_worktree_execution=True, max_parallel_subtasks=2,
        )
        self.coordinator = ParallelExecutionCoordinator(self.config, self.storage)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_workers_actually_overlap(self):
        """Both workers must be inside the executor at once or the barrier times out."""
        barrier = threading.Barrier(2, timeout=5)
        overlapped = []

        def fake_exec(task, subtask, base_commit="HEAD", progress=None):
            barrier.wait()  # raises BrokenBarrierError on timeout -> test fails
            overlapped.append(subtask.subtask_id)
            return (subtask, None, None)

        a = Subtask(subtask_id="a", title="A")
        b = Subtask(subtask_id="b", title="B")
        task = _make_test_task("t", TaskPlan(objective="t", subtasks=[a, b]))
        with patch.object(self.coordinator, "execute_subtask_in_worktree", side_effect=fake_exec):
            results = self.coordinator.execute_parallel_batch(task, [a, b])

        self.assertEqual(len(results), 2)
        self.assertEqual(sorted(overlapped), ["a", "b"])

    def test_concurrent_worker_results_do_not_clobber_each_other(self):
        s1 = Subtask(subtask_id="s1", title="S1", status=SubtaskStatus.PENDING)
        s2 = Subtask(subtask_id="s2", title="S2", status=SubtaskStatus.PENDING)
        task = _make_test_task("task-1", TaskPlan(objective="t", subtasks=[s1, s2]))
        self.storage.save_task(task)

        start = threading.Barrier(2, timeout=5)

        def merge_one(sid: str):
            done = Subtask(subtask_id=sid, title=sid, status=SubtaskStatus.COMPLETED)
            # Force maximum interleaving of the read-modify-write cycles.
            start.wait()
            self.coordinator._merge_subtask_into_canonical_task("task-1", done)

        threads = [threading.Thread(target=merge_one, args=(sid,)) for sid in ("s1", "s2")]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        reloaded = self.storage.load_task("task-1")
        statuses = {s.subtask_id: s.status for s in reloaded.plan.subtasks}
        # Neither worker's completion may be erased by the other.
        self.assertEqual(statuses["s1"], SubtaskStatus.COMPLETED)
        self.assertEqual(statuses["s2"], SubtaskStatus.COMPLETED)

    def test_worker_scoped_storage_blocks_premature_knowledge_promotion(self):
        from local_agent.coordinator import _WorkerScopedStorage

        task = _make_test_task("task-1", TaskPlan(objective="t", subtasks=[]))
        self.storage.save_knowledge_graph(RepositoryKnowledgeGraph(repo_id="authoritative"))
        scoped = _WorkerScopedStorage(self.storage, task)

        # A worker "discovers" something and saves it.
        branch_local = RepositoryKnowledgeGraph(repo_id="branch-local")
        scoped.save_knowledge_graph(branch_local)

        # The authoritative graph is untouched; only the worker sees its own view.
        self.assertEqual(self.storage.load_knowledge_graph().repo_id, "authoritative")
        self.assertEqual(scoped.load_knowledge_graph().repo_id, "branch-local")

    def test_worker_scoped_storage_buffers_task_writes(self):
        from local_agent.coordinator import _WorkerScopedStorage

        sub = Subtask(subtask_id="s1", title="S1", status=SubtaskStatus.PENDING)
        task = _make_test_task("task-1", TaskPlan(objective="t", subtasks=[sub]))
        self.storage.save_task(task)

        worker_task = Task.from_dict(task.to_dict())
        scoped = _WorkerScopedStorage(self.storage, worker_task)
        worker_task.plan.subtasks[0].status = SubtaskStatus.COMPLETED
        scoped.save_task(worker_task)

        # Shared record is unchanged until the coordinator merges the result back.
        self.assertEqual(self.storage.load_task("task-1").plan.subtasks[0].status, SubtaskStatus.PENDING)
        self.assertEqual(scoped.load_task("task-1").plan.subtasks[0].status, SubtaskStatus.COMPLETED)
        # Unrelated storage APIs still delegate.
        self.assertEqual(scoped.list_tasks()[0].task_id, "task-1")


class TestSchedulerNonGitFallback(unittest.TestCase):
    """Phase 4.14 audit: parallelism must not run without Git isolation."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(self.temp_dir) / "plain"
        self.project_root.mkdir()
        self.storage = JsonFileStorage(Path(self.temp_dir) / "storage")
        self.storage.save_provider_configs([ProviderConfig(provider_id="mock", priority=10, enabled=True)])
        self.credential_store = MockCredentialStore()
        self.credential_store.save("dungx-ai-coding-agent", "mock", "mock_key")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_non_git_workspace_falls_back_to_serial_execution(self):
        config = AgentConfig(
            project=self.project_root, provider="mock",
            parallel_worktree_execution=True, max_parallel_subtasks=2,
        )
        scheduler = Scheduler(config, self.storage, self.credential_store)
        s1 = Subtask(subtask_id="s1", title="Sub 1", status=SubtaskStatus.PENDING)
        s2 = Subtask(subtask_id="s2", title="Sub 2", status=SubtaskStatus.PENDING)
        task = _make_test_task("task-nogit", TaskPlan(objective="goal", subtasks=[s1, s2]))
        self.storage.save_task(task)

        with patch.object(scheduler.coordinator, "execute_parallel_batch") as mock_parallel, \
             patch("local_agent.scheduler.Orchestrator") as mock_orch_cls:
            mock_orch = MagicMock()
            mock_orch.run.return_value = MagicMock(outcome="SUCCESS")
            mock_orch_cls.return_value = mock_orch
            scheduler.run_once()

        mock_parallel.assert_not_called()
        self.assertEqual(mock_orch.run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
