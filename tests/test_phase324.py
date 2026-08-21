from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_agent.cli import main as cli_main
from local_agent.config import AgentConfig
from local_agent.models import Task, TaskStatus
from local_agent.orchestrator import Orchestrator
from local_agent.storage import JsonFileStorage


class Phase324_GitOperationsTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.storage = JsonFileStorage(self.root / ".agent_data")

        # Initialize a git repository for testing
        subprocess.run(["git", "init"], cwd=self.root, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=self.root, check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=self.root, check=True)
        (self.root / "initial_file.txt").write_text("initial content")
        subprocess.run(["git", "add", "."], cwd=self.root, check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=self.root, check=True, capture_output=True)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root, ignore_errors=True)

    def _create_completed_task(self, changed_files: list[str]) -> Task:
        now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
        task = Task(
            task_id="test-task-12345678",
            objective="Test Git Commit",
            status=TaskStatus.COMPLETED,
            created_at=now,
            updated_at=now,
            changed_files=changed_files,
        )
        self.storage.save_task(task)
        for file_path in changed_files:
            p = self.root / file_path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(f"Content for {file_path}")
        return task

    def test_commit_task_success(self):
        # Arrange
        task = self._create_completed_task(["new_file.txt", "src/another.py"])

        # Act
        with mock.patch("sys.stdout", new=__import__("io").StringIO()) as fake_out:
            result = cli_main(["commit-task", "--project", str(self.root), task.task_id])

        # Assert
        self.assertEqual(result, 0)
        self.assertIn("Successfully committed changes", fake_out.getvalue())

        # Verify git state
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=self.root, text=True).strip()
        self.assertEqual(branch, "agent-task/test-tas")

        log = subprocess.check_output(["git", "log", "-1", "--pretty=%B"], cwd=self.root, text=True).strip()
        self.assertIn("feat: Complete task 'Test Git Commit'", log)
        self.assertIn("Task-ID: test-task-12345678", log)

        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=self.root, text=True).strip()
        status_lines = [line for line in status.splitlines() if not line[3:].startswith(".agent_data")]
        self.assertEqual(status_lines, []) # Working tree should be clean

    def test_commit_task_fails_if_working_tree_is_dirty(self):
        # Arrange
        task = self._create_completed_task(["new_file.txt"])
        (self.root / "unrelated_change.txt").write_text("dirty")

        # Act
        with mock.patch("sys.stderr", new=__import__("io").StringIO()) as fake_err:
            result = cli_main(["commit-task", "--project", str(self.root), task.task_id])

        # Assert
        self.assertEqual(result, 1)
        self.assertIn("Working tree contains unexpected changes", fake_err.getvalue())

    def test_commit_task_fails_if_task_not_completed(self):
        task = self._create_completed_task(["new_file.txt"])
        task.status = TaskStatus.PENDING
        self.storage.save_task(task)

        with mock.patch("sys.stderr", new=__import__("io").StringIO()) as fake_err:
            result = cli_main(["commit-task", "--project", str(self.root), task.task_id])

        self.assertEqual(result, 1)
        self.assertIn("is not in COMPLETED state", fake_err.getvalue())

    @mock.patch("local_agent.orchestrator.Orchestrator._perform_git_commit")
    def test_autonomous_mode_triggers_commit_on_completion(self, mock_perform_commit):
        # Arrange
        config = AgentConfig.from_environment(self.root, autonomous_mode=True, git_commit_on_completion=True)
        task = self._create_completed_task([])
        task.autonomous = True
        report = mock.MagicMock(completed=True, changed_files=["a.py"])
        
        orchestrator = Orchestrator(config, self.storage, mock.MagicMock(), mock.MagicMock(), mock.MagicMock())

        # Act
        # Directly call the final part of the orchestrator's run method
        orchestrator._create_memories_from_run(task, report)

        # Assert
        mock_perform_commit.assert_called_once_with(task)

    @mock.patch("local_agent.orchestrator.Orchestrator._perform_git_commit")
    def test_autonomous_mode_does_not_commit_if_disabled(self, mock_perform_commit):
        # Arrange
        config = AgentConfig.from_environment(self.root, autonomous_mode=True, git_commit_on_completion=False) # Disabled
        task = self._create_completed_task([])
        task.autonomous = True
        report = mock.MagicMock(completed=True, changed_files=["a.py"])
        
        orchestrator = Orchestrator(config, self.storage, mock.MagicMock(), mock.MagicMock(), mock.MagicMock())

        # Act
        orchestrator._create_memories_from_run(task, report)

        # Assert
        mock_perform_commit.assert_not_called()

    def test_push_task_success(self):
        # Arrange
        task = self._create_completed_task(["pushed_file.txt"])
        cli_main(["commit-task", "--project", str(self.root), task.task_id])

        # Create a bare repo to act as a remote
        remote_path = Path(tempfile.mkdtemp()) / "remote.git"
        subprocess.run(["git", "init", "--bare", str(remote_path)], check=True)
        subprocess.run(["git", "remote", "add", "origin", str(remote_path)], cwd=self.root, check=True)

        # Act
        with mock.patch("sys.stdout", new=__import__("io").StringIO()) as fake_out:
            result = cli_main(["push-task", "--project", str(self.root), task.task_id])

        # Assert
        self.assertEqual(result, 0)
        self.assertIn("Successfully pushed branch", fake_out.getvalue())

        # Verify the branch exists on the "remote"
        remote_branches = subprocess.check_output(["git", "branch", "-a"], cwd=remote_path, text=True).strip()
        self.assertIn("agent-task/test-tas", remote_branches)