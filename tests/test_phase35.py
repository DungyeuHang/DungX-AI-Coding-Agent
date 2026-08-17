from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from local_agent.analyzer import RepositoryAnalyzer
from local_agent.cli import main
from local_agent.config import AgentConfig
from local_agent.context import ContextSelector
from local_agent.models import FailureAnalysis, Plan
from local_agent.providers import GeminiProvider


class Phase35Tests(unittest.TestCase):
    def _repository(self) -> Path:
        directory = Path(tempfile.mkdtemp())
        (directory / ".gitignore").write_text("ignored/\n*.ignored\n", encoding="utf-8")
        (directory / ".git").mkdir()
        (directory / ".git" / "config").write_text("private git data", encoding="utf-8")
        (directory / "node_modules").mkdir()
        (directory / "node_modules" / "package.js").write_text("ignored", encoding="utf-8")
        (directory / "ignored").mkdir()
        (directory / "ignored" / "ignored.py").write_text("ignored", encoding="utf-8")
        (directory / "generated_file.py").write_text("generated", encoding="utf-8")
        (directory / "scratch.ignored").write_text("ignored", encoding="utf-8")
        (directory / ".env").write_text("API_KEY=do-not-expose", encoding="utf-8")
        (directory / "private.pem").write_text("PRIVATE KEY", encoding="utf-8")
        (directory / "pyproject.toml").write_text("[project]\nname='map-test'\n", encoding="utf-8")
        (directory / "main.py").write_text("from src.teacher_dashboard import TeacherDashboard\n", encoding="utf-8")
        (directory / "src" / "services").mkdir(parents=True)
        (directory / "tests").mkdir()
        (directory / "src" / "teacher_dashboard.py").write_text(
            "from src.services.class_service import ClassService\n\nclass TeacherDashboard:\n    service = ClassService()\n", encoding="utf-8"
        )
        (directory / "src" / "services" / "class_service.py").write_text(
            "from src.api import request\n\nclass ClassService:\n    pass\n", encoding="utf-8"
        )
        (directory / "src" / "api.py").write_text("def request():\n    return None\n", encoding="utf-8")
        (directory / "src" / "settings.py").write_text("UNRELATED = True\n", encoding="utf-8")
        (directory / "tests" / "test_teacher_dashboard.py").write_text(
            "from src.teacher_dashboard import TeacherDashboard\n", encoding="utf-8"
        )
        return directory

    def test_scan_builds_deterministic_map_and_protects_ignored_files(self):
        root = self._repository()
        context = RepositoryAnalyzer(root).analyze()
        paths = {item.path for item in context.repository_map.files}
        self.assertIn("src/teacher_dashboard.py", paths)
        self.assertIn("src/services/class_service.py", paths)
        self.assertNotIn(".env", paths)
        self.assertNotIn("private.pem", paths)
        self.assertNotIn("generated_file.py", paths)
        self.assertNotIn("ignored/ignored.py", paths)
        self.assertNotIn("node_modules/package.js", paths)
        self.assertIn(".env", {item["path"] for item in context.repository_map.protected_paths})
        ignored = {item["path"]: item["reason"] for item in context.repository_map.ignored_paths}
        self.assertEqual(ignored["node_modules"], "ignored directory")
        self.assertEqual(ignored["generated_file.py"], "generated file")
        self.assertIn("main.py", context.repository_map.entry_points)
        relationships = {(item.source, item.target, item.kind) for item in context.repository_map.relationships}
        self.assertIn(("src/teacher_dashboard.py", "src/services/class_service.py", "imports"), relationships)
        self.assertIn(("src/services/class_service.py", "src/api.py", "imports"), relationships)
        self.assertIn(("src/teacher_dashboard.py", "tests/test_teacher_dashboard.py", "tested_by"), relationships)
        self.assertEqual(context.repository_map.compact(), RepositoryAnalyzer(root).analyze().repository_map.compact())

    def test_relevance_ranking_expands_dependencies_and_respects_depth(self):
        root = self._repository()
        context = RepositoryAnalyzer(root).analyze()
        selector = ContextSelector(root, max_files=5, max_chars=10000, max_file_chars=2000, max_tokens=2500, dependency_depth=1)
        selector.select("Add class management to the teacher dashboard", context)
        selected = context.metadata["selected_files"]
        self.assertEqual(selected[0], "src/teacher_dashboard.py")
        self.assertIn("src/services/class_service.py", selected)
        self.assertIn("tests/test_teacher_dashboard.py", selected)
        self.assertNotIn("src/api.py", selected)
        items = context.metadata["context_selection"]["selected_items"]
        self.assertGreater(items[0]["score"], 0.5)
        excluded = {item["path"]: item["reason"] for item in context.metadata["context_excluded"]}
        self.assertIn("src/settings.py", excluded)

        deep_context = RepositoryAnalyzer(root).analyze()
        ContextSelector(root, max_files=6, max_chars=10000, max_file_chars=2000, max_tokens=2500, dependency_depth=2).select("teacher dashboard", deep_context)
        self.assertIn("src/api.py", deep_context.metadata["selected_files"])

    def test_context_budget_limits_bytes_and_estimated_tokens(self):
        root = self._repository()
        context = RepositoryAnalyzer(root).analyze()
        ContextSelector(root, max_files=2, max_chars=240, max_file_chars=120, max_tokens=60, dependency_depth=1).select("teacher dashboard", context)
        previews = context.metadata["selected_file_previews"]
        self.assertLessEqual(len(previews), 2)
        self.assertLessEqual(sum(len(value.encode("utf-8")) for value in previews.values()), 240)
        self.assertLessEqual(context.metadata["context_selection"]["estimated_tokens"], 60)
        self.assertTrue(context.metadata["context_excluded"])

    def test_context_limits_are_configurable(self):
        root = self._repository()
        with mock.patch.dict("os.environ", {
            "AGENT_MAX_CONTEXT_FILES": "3",
            "AGENT_MAX_CONTEXT_TOKENS": "900",
            "AGENT_DEPENDENCY_DEPTH": "2",
        }, clear=True):
            config = AgentConfig.from_environment(root)
        self.assertEqual(config.max_context_files, 3)
        self.assertEqual(config.max_context_tokens, 900)
        self.assertEqual(config.dependency_depth, 2)

    def test_secret_values_do_not_enter_provider_context(self):
        root = self._repository()
        context = RepositoryAnalyzer(root).analyze()
        ContextSelector(root, max_files=5).select("teacher dashboard", context)
        config = AgentConfig.from_environment(root, provider="gemini")
        provider = GeminiProvider.__new__(GeminiProvider)
        provider.config = config
        payload = provider._context(context, "planning")
        self.assertNotIn("do-not-expose", payload)
        self.assertNotIn("private key", payload.lower())
        self.assertNotIn(".env", payload)

    def test_context_cli_does_not_call_provider(self):
        root = self._repository()
        output = io.StringIO()
        with redirect_stdout(output):
            result = main(["context", "--project", str(root), "teacher dashboard"])
        self.assertEqual(result, 0)
        self.assertIn("Selected context:", output.getvalue())
        self.assertIn("teacher_dashboard.py", output.getvalue())


if __name__ == "__main__":
    unittest.main()
