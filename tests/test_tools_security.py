from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from local_agent.models import ToolCall
from local_agent.tools import ToolRegistry


class ToolsSecurityTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name).resolve()

        # Setup test files
        (self.root / "src").mkdir(parents=True, exist_ok=True)
        (self.root / "src" / "app.py").write_text("print('hello world')\n", encoding="utf-8")

        # Setup protected directories and files
        (self.root / ".git").mkdir(parents=True, exist_ok=True)
        (self.root / ".git" / "config").write_text("[core]\nrepositoryformatversion = 0\n", encoding="utf-8")

        (self.root / ".agent_data").mkdir(parents=True, exist_ok=True)
        (self.root / ".agent_data" / "tasks.json").write_text("{}", encoding="utf-8")

        (self.root / ".env").write_text("SECRET_KEY=super_secret\n", encoding="utf-8")
        (self.root / "credentials.json").write_text("{\"api_key\": \"12345\"}\n", encoding="utf-8")

        self.registry = ToolRegistry(self.root)

    def tearDown(self):
        self.temp_dir.cleanup()

    # -------------------------------------------------------------------------
    # read_file_range security
    # -------------------------------------------------------------------------

    def test_read_file_range_rejects_parent_traversal(self):
        call = ToolCall("call-1", "read_file_range", {"path": "../../etc/passwd"})
        result = self.registry.execute(call)
        self.assertTrue(result.is_error)
        self.assertIn("Access denied", result.output)

    def test_read_file_range_rejects_absolute_path_outside_project(self):
        call = ToolCall("call-2", "read_file_range", {"path": "C:\\Windows\\System32\\drivers\\etc\\hosts"})
        result = self.registry.execute(call)
        self.assertTrue(result.is_error)
        self.assertIn("Access denied", result.output)

    def test_read_file_range_rejects_git_directory(self):
        call = ToolCall("call-3", "read_file_range", {"path": ".git/config"})
        result = self.registry.execute(call)
        self.assertTrue(result.is_error)
        self.assertIn("Access denied", result.output)
        self.assertIn(".git", result.output)

    def test_read_file_range_rejects_agent_data_directory(self):
        call = ToolCall("call-4", "read_file_range", {"path": ".agent_data/tasks.json"})
        result = self.registry.execute(call)
        self.assertTrue(result.is_error)
        self.assertIn("Access denied", result.output)
        self.assertIn(".agent_data", result.output)

    def test_read_file_range_rejects_secret_files(self):
        call = ToolCall("call-5", "read_file_range", {"path": ".env"})
        result = self.registry.execute(call)
        self.assertTrue(result.is_error)
        self.assertIn("Access denied", result.output)
        self.assertIn("secret-like file", result.output)

        call2 = ToolCall("call-6", "read_file_range", {"path": "credentials.json"})
        result2 = self.registry.execute(call2)
        self.assertTrue(result2.is_error)
        self.assertIn("Access denied", result2.output)

    # -------------------------------------------------------------------------
    # grep_code security
    # -------------------------------------------------------------------------

    def test_grep_code_cannot_escape_project_root(self):
        call = ToolCall("call-7", "grep_code", {"pattern": "secret", "glob": "../../*"})
        result = self.registry.execute(call)
        self.assertTrue(result.is_error)
        self.assertIn("Access denied", result.output)

    def test_grep_code_ignores_protected_directories(self):
        call = ToolCall("call-8", "grep_code", {"pattern": "repositoryformatversion"})
        result = self.registry.execute(call)
        self.assertFalse(result.is_error)
        self.assertNotIn(".git", result.output)
        self.assertIn("No matches found", result.output)

    def test_grep_code_ignores_secret_files(self):
        call = ToolCall("call-9", "grep_code", {"pattern": "SECRET_KEY"})
        result = self.registry.execute(call)
        self.assertFalse(result.is_error)
        self.assertNotIn(".env", result.output)
        self.assertIn("No matches found", result.output)

    def test_grep_code_handles_invalid_regex_safely(self):
        call = ToolCall("call-10", "grep_code", {"pattern": "[unclosed bracket"})
        result = self.registry.execute(call)
        self.assertTrue(result.is_error)
        self.assertIn("Invalid regex pattern", result.output)

    # -------------------------------------------------------------------------
    # find_files security
    # -------------------------------------------------------------------------

    def test_find_files_cannot_escape_project_root(self):
        call = ToolCall("call-11", "find_files", {"pattern": "../../*"})
        result = self.registry.execute(call)
        self.assertTrue(result.is_error)
        self.assertIn("Access denied", result.output)

    def test_find_files_ignores_protected_directories(self):
        call = ToolCall("call-12", "find_files", {"pattern": "*"})
        result = self.registry.execute(call)
        self.assertFalse(result.is_error)
        self.assertNotIn(".git", result.output)
        self.assertNotIn(".agent_data", result.output)
        self.assertIn("src/app.py", result.output)

    def test_find_files_does_not_expose_secrets(self):
        call = ToolCall("call-13", "find_files", {"pattern": "*.env*"})
        result = self.registry.execute(call)
        self.assertFalse(result.is_error)
        self.assertIn("No files found", result.output)

    # -------------------------------------------------------------------------
    # run_command_sandbox security
    # -------------------------------------------------------------------------

    def test_run_command_sandbox_rejects_destructive_commands(self):
        for cmd in [["rm", "-rf", "src"], ["del", "src"], ["format", "c:"], ["shutdown", "/s"]]:
            call = ToolCall("call-14", "run_command_sandbox", {"command": cmd})
            result = self.registry.execute(call)
            self.assertTrue(result.is_error)
            self.assertIn("Security violation", result.output)

    def test_run_command_sandbox_rejects_shell_operators(self):
        for op in ["|", "&&", ";", ">", "<"]:
            call = ToolCall("call-15", "run_command_sandbox", {"command": ["echo", "test", op, "other"]})
            result = self.registry.execute(call)
            self.assertTrue(result.is_error)
            self.assertIn("Security violation", result.output)

    def test_run_command_sandbox_rejects_git_push_and_reset(self):
        for cmd in [["git", "push", "origin", "main"], ["git", "reset", "--hard"], ["git", "clean", "-fd"]]:
            call = ToolCall("call-16", "run_command_sandbox", {"command": cmd})
            result = self.registry.execute(call)
            self.assertTrue(result.is_error)
            self.assertIn("Security violation", result.output)

    def test_run_command_sandbox_rejects_empty_or_non_string_tokens(self):
        call = ToolCall("call-17", "run_command_sandbox", {"command": []})
        result = self.registry.execute(call)
        self.assertTrue(result.is_error)

        call2 = ToolCall("call-18", "run_command_sandbox", {"command": ["python", 123]})
        result2 = self.registry.execute(call2)
        self.assertTrue(result2.is_error)


if __name__ == "__main__":
    unittest.main()
