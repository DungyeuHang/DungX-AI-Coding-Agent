from __future__ import annotations

import datetime
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_agent.cli import main as cli_main
from local_agent.models import PullRequestInfo, Task, TaskStatus
from local_agent.storage import JsonFileStorage


class MockAPIResponse:
    def __init__(self, data, status_code=200):
        self.data = json.dumps(data).encode("utf-8")
        self.status = status_code

    def read(self):
        return self.data

    def getcode(self):
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class Phase325_PullRequestTests(unittest.TestCase):
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
        subprocess.run(["git", "remote", "add", "origin", "https://github.com/test-owner/test-repo.git"], cwd=self.root, check=True)

        self.mock_urlopen_patcher = mock.patch("urllib.request.urlopen")
        self.mock_urlopen = self.mock_urlopen_patcher.start()

    def tearDown(self):
        self.mock_urlopen_patcher.stop()
        import shutil
        shutil.rmtree(self.root, ignore_errors=True)

    def _create_completed_task(self) -> Task:
        now = datetime.datetime.now(datetime.timezone.utc)
        task = Task(
            task_id="test-task-pr12345",
            objective="Test PR Creation",
            status=TaskStatus.COMPLETED,
            created_at=now,
            updated_at=now,
            changed_files=["new_file.txt"],
        )
        self.storage.save_task(task)
        return task

    @mock.patch.dict( "os.environ", {"GITHUB_TOKEN": "test-token"})
    def test_create_pr_command_success(self):
        # Arrange
        task = self._create_completed_task()
        branch_name = f"agent-task/{task.task_id[:8]}"
        subprocess.run(["git", "checkout", "-b", branch_name], cwd=self.root, check=True)

        # Mock API responses: first find returns empty, second create returns a PR
        self.mock_urlopen.side_effect = [
            MockAPIResponse([]), # No existing PR
            MockAPIResponse({
                "number": 101,
                "html_url": "https://github.com/test-owner/test-repo/pull/101",
                "state": "open",
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z",
            }, status_code=201),
        ]

        # Act
        with mock.patch("sys.stdout", new=__import__("io").StringIO()) as fake_out:
            result = cli_main(["create-pr", "--project", str(self.root), task.task_id])

        # Assert
        self.assertEqual(result, 0)
        self.assertIn("Successfully created pull request", fake_out.getvalue())

        loaded_task = self.storage.load_task(task.task_id)
        self.assertIsInstance(loaded_task.pull_request, PullRequestInfo)
        self.assertEqual(loaded_task.pull_request.pr_id, "101")
        self.assertEqual(loaded_task.pull_request.url, "https://github.com/test-owner/test-repo/pull/101")

        # Verify API calls
        self.assertEqual(self.mock_urlopen.call_count, 2)
        create_pr_request = self.mock_urlopen.call_args_list[1].args[0]
        self.assertEqual(create_pr_request.method, "POST")
        body = json.loads(create_pr_request.data)
        self.assertEqual(body["title"], "feat: Test PR Creation")
        self.assertEqual(body["head"], branch_name)

    @mock.patch.dict("os.environ", {"GITHUB_TOKEN": "test-token"})
    def test_create_pr_is_idempotent(self):
        # Arrange
        task = self._create_completed_task()
        pr_info = PullRequestInfo("github", "101", "http://example.com/pr/101", "open", datetime.datetime.now(datetime.timezone.utc))
        task.pull_request = pr_info
        self.storage.save_task(task)

        # Act
        with mock.patch("sys.stdout", new=__import__("io").StringIO()) as fake_out:
            result = cli_main(["create-pr", "--project", str(self.root), task.task_id])

        # Assert
        self.assertEqual(result, 0)
        self.assertIn("already exists: http://example.com/pr/101", fake_out.getvalue())
        self.mock_urlopen.assert_not_called() # Should not make any API calls