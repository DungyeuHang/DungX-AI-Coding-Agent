from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_agent.analyzer import RepositoryAnalyzer
from local_agent.commands import CommandRunner, UnsafeCommandError
from local_agent.config import AgentConfig
from local_agent.filesystem import ProjectFilesystem, ProtectedPathError, SandboxViolation
from local_agent.git import GitIntegration
from local_agent.models import CommandSpec, ExecutionResult
from local_agent.planner import Planner
from local_agent.providers import MockProvider


class CoreTests(unittest.TestCase):
    def test_configuration_reads_environment_and_validates(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"AGENT_PROVIDER": "mock", "AGENT_MAX_ITERATIONS": "3"}, clear=False):
            config = AgentConfig.from_environment(directory, provider="mock", max_iterations=3)
            config.validate()
            self.assertEqual(config.max_iterations, 3)
            self.assertEqual(config.provider, "mock")

    def test_filesystem_is_sandboxed_and_protects_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            filesystem = ProjectFilesystem(directory)
            filesystem.write_file("src/example.txt", "hello")
            self.assertEqual(filesystem.read_file("src/example.txt"), "hello")
            with self.assertRaises(SandboxViolation):
                filesystem.read_file("../outside.txt")
            Path(directory, ".env").write_text("TOKEN=hidden", encoding="utf-8")
            with self.assertRaises(ProtectedPathError):
                filesystem.read_file(".env")
            with self.assertRaises(ProtectedPathError):
                filesystem.delete_file(".git/config")

    def test_analyzer_detects_python_metadata_and_tests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pyproject.toml").write_text("[project]\nname='sample'\nrequires-python='>=3.11'\n", encoding="utf-8")
            (root / "tests").mkdir()
            (root / "tests" / "test_sample.py").write_text("def test_ok(): pass\n", encoding="utf-8")
            context = RepositoryAnalyzer(root).analyze()
            self.assertIn("Python", context.metadata["stacks"])
            self.assertIn("tests/test_sample.py", context.test_files)
            self.assertTrue(any(command.name == "tests" for command in context.validation_commands))

    def test_command_runner_captures_success_and_blocks_destructive_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = CommandRunner(directory)
            result = runner.run(CommandSpec("test", ("python", "-c", "print('ok')")))
            self.assertTrue(result.succeeded)
            self.assertIn("ok", result.stdout)
            with self.assertRaises(UnsafeCommandError):
                runner.run(CommandSpec("bad", ("git", "reset", "--hard")))

    def test_planner_and_mock_provider_are_real_structured_operations(self):
        with tempfile.TemporaryDirectory() as directory:
            context = RepositoryAnalyzer(directory).analyze()
            task_plan = Planner(MockProvider()).create_task_plan("Add a feature", context)
            self.assertEqual(task_plan.objective, "Add a feature")
            self.assertEqual(MockProvider().generate_code("Add a feature", task_plan, context), [])

    def test_failure_result_is_not_marked_success(self):
        result = ExecutionResult("python -c fail", 1, stderr="boom")
        self.assertFalse(result.succeeded)

    def test_git_adapter_handles_non_repository_without_remote_operations(self):
        with tempfile.TemporaryDirectory() as directory:
            git = GitIntegration(directory)
            self.assertEqual(git.status(), "")
            self.assertEqual(git.diff(), "")
            self.assertEqual(git.branch(), "")


if __name__ == "__main__":
    unittest.main()
