from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from local_agent.context import ContextSelector
from local_agent.models import (
    FileIndex,
    ProjectContext,
    SemanticIndex,
    SymbolDefinition,
    SymbolLocation,
)


class Phase3156_QualifiedSymbolResolutionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.repo_root = self.temp_dir / "repo"
        self.repo_root.mkdir()

        # Create dummy files
        (self.repo_root / "user_service.py").write_text("class UserService:\n  def save(self, user):\n    pass\n")
        (self.repo_root / "auth_service.py").write_text("class AuthService:\n  def save(self, user):\n    pass\n")
        (self.repo_root / "unrelated_saver.py").write_text("def save():\n  pass\n")
        (self.repo_root / "analytics_service.js").write_text("class AnalyticsService {\n  track() {}\n}\n")
        (self.repo_root / "other.py").write_text("class Other:\n  def other_method(self):\n    pass\n")

        # Manually build the semantic index for testing
        def symbol(name: str, kind: str, line: int, parent: str | None = None) -> SymbolDefinition:
            return SymbolDefinition(
                name=name,
                kind=kind,
                location=SymbolLocation(start_line=line, end_line=line + 1),
                parent=parent,
            )

        self.semantic_index = SemanticIndex(
            files={
                "user_service.py": FileIndex(
                    path="user_service.py", language="Python", content_hash="h1",
                    symbols=[symbol("UserService", "class", 1), symbol("save", "method", 2, "UserService")]
                ),
                "auth_service.py": FileIndex(
                    path="auth_service.py", language="Python", content_hash="h2",
                    symbols=[symbol("AuthService", "class", 1), symbol("save", "method", 2, "AuthService")]
                ),
                "unrelated_saver.py": FileIndex(
                    path="unrelated_saver.py", language="Python", content_hash="h3",
                    symbols=[symbol("save", "function", 1)]
                ),
                "analytics_service.js": FileIndex(
                    path="analytics_service.js", language="JavaScript", content_hash="h4",
                    symbols=[symbol("AnalyticsService", "class", 1), symbol("track", "method", 2, "AnalyticsService")]
                ),
                "other.py": FileIndex(
                    path="other.py", language="Python", content_hash="h5",
                    symbols=[symbol("Other", "class", 1), symbol("other_method", "method", 2, "Other")]
                ),
            }
        )
        self.project_context = ProjectContext(
            root=str(self.repo_root),
            source_files=list(self.semantic_index.files.keys()),
            metadata={"semantic_index": self.semantic_index},
        )
        # Mock repository_map for ContextSelector
        self.project_context.repository_map = unittest.mock.MagicMock()
        self.project_context.repository_map.files = [unittest.mock.MagicMock(path=p) for p in self.semantic_index.files.keys()]
        self.project_context.repository_map.relationships = []

        self.context_selector = ContextSelector(self.repo_root)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def _get_file_score_and_reason(self, task: str, file_path: str) -> tuple[float, list[str]]:
        selected_context = self.context_selector.select(task, self.project_context)
        for item in selected_context.metadata["context_selection"]["selected_items"]:
            if item["path"] == file_path:
                return item["score"], item["reason"]
        return 0.0, []

    def test_exact_qualified_python_match(self):
        task = "Fix `UserService.save` method"
        score_user, reasons_user = self._get_file_score_and_reason(task, "user_service.py")
        score_auth, _ = self._get_file_score_and_reason(task, "auth_service.py")
        score_unrelated, _ = self._get_file_score_and_reason(task, "unrelated_saver.py")

        self.assertGreater(score_user, 0.7, "UserService.save should get a high qualified boost")
        self.assertTrue(any("semantic qualified symbol match: UserService.save" in r for r in reasons_user))
        
        # Other files with 'save' should get a lower, unqualified boost
        self.assertLess(score_auth, 0.6, "AuthService.save should not get qualified boost")
        self.assertGreater(score_auth, 0.4, "AuthService.save should get unqualified boost for 'save'")
        self.assertLess(score_unrelated, 0.6, "unrelated_saver.py should not get qualified boost")
        self.assertGreater(score_unrelated, 0.4, "unrelated_saver.py should get unqualified boost for 'save'")
        
        self.assertGreater(score_user, score_auth)
        self.assertGreater(score_user, score_unrelated)

    def test_exact_qualified_javascript_match(self):
        task = "Debug `AnalyticsService.track`"
        score_analytics, reasons_analytics = self._get_file_score_and_reason(task, "analytics_service.js")
        score_other, _ = self._get_file_score_and_reason(task, "other.py")

        self.assertGreater(score_analytics, 0.7, "AnalyticsService.track should get a high qualified boost")
        self.assertTrue(any("semantic qualified symbol match: AnalyticsService.track" in r for r in reasons_analytics))
        self.assertLess(score_other, 0.2, "other.py should have a very low score")

    def test_parent_container_mismatch(self):
        task = "Fix `AdminService.save`"
        score_user, reasons_user = self._get_file_score_and_reason(task, "user_service.py")
        score_auth, reasons_auth = self._get_file_score_and_reason(task, "auth_service.py")

        # No file should get the high qualified boost
        self.assertFalse(any("semantic qualified symbol match" in r for r in reasons_user))
        self.assertFalse(any("semantic qualified symbol match" in r for r in reasons_auth))

        # Both should get the lower unqualified boost for 'save'
        self.assertGreater(score_user, 0.4, "user_service.py should get unqualified boost for 'save'")
        self.assertLess(score_user, 0.6)
        self.assertGreater(score_auth, 0.4, "auth_service.py should get unqualified boost for 'save'")
        self.assertLess(score_auth, 0.6)

    def test_unqualified_symbol_regression(self):
        task = "Fix the `save` function"
        score_user, reasons_user = self._get_file_score_and_reason(task, "user_service.py")
        score_auth, reasons_auth = self._get_file_score_and_reason(task, "auth_service.py")
        score_unrelated, reasons_unrelated = self._get_file_score_and_reason(task, "unrelated_saver.py")

        # All should get the same unqualified boost
        self.assertTrue(any("semantic symbol definition match: save" in r for r in reasons_user))
        self.assertTrue(any("semantic symbol definition match: save" in r for r in reasons_auth))
        self.assertTrue(any("semantic symbol definition match: save" in r for r in reasons_unrelated))

        self.assertGreater(score_user, 0.4)
        self.assertLess(score_user, 0.6)
        self.assertGreater(score_auth, 0.4)
        self.assertLess(score_auth, 0.6)
        self.assertGreater(score_unrelated, 0.4)
        self.assertLess(score_unrelated, 0.6)

    def test_class_only_symbol_regression(self):
        task = "Update the `UserService`"
        score_user, reasons_user = self._get_file_score_and_reason(task, "user_service.py")
        score_auth, _ = self._get_file_score_and_reason(task, "auth_service.py")

        self.assertGreater(score_user, 0.4, "user_service.py should get a boost for the class name")
        self.assertLess(score_user, 0.6)
        self.assertTrue(any("semantic symbol definition match: UserService" in r for r in reasons_user))
        self.assertLess(score_auth, 0.2, "auth_service.py should not get a boost")

    def test_multiple_valid_qualified_definitions(self):
        # Add a duplicate
        self.semantic_index.files["user_service_v2.py"] = FileIndex(
            path="user_service_v2.py", language="Python", content_hash="h6",
            symbols=[SymbolDefinition(name="UserService", kind="class", location=SymbolLocation(1,1)), SymbolDefinition(name="save", kind="method", location=SymbolLocation(2,2), parent="UserService")]
        )
        (self.repo_root / "user_service_v2.py").write_text("class UserService:\n  def save(self):\n    pass")
        self.project_context.source_files.append("user_service_v2.py")
        self.project_context.repository_map.files.append(unittest.mock.MagicMock(path="user_service_v2.py"))

        task = "Fix `UserService.save`"
        score_v1, reasons_v1 = self._get_file_score_and_reason(task, "user_service.py")
        score_v2, reasons_v2 = self._get_file_score_and_reason(task, "user_service_v2.py")

        self.assertGreater(score_v1, 0.7)
        self.assertTrue(any("semantic qualified symbol match: UserService.save" in r for r in reasons_v1))
        self.assertGreater(score_v2, 0.7)
        self.assertTrue(any("semantic qualified symbol match: UserService.save" in r for r in reasons_v2))

    def test_missing_semantic_index_fallback(self):
        self.project_context.metadata["semantic_index"] = None
        task = "Fix `UserService.save`"
        score_user, reasons_user = self._get_file_score_and_reason(task, "user_service.py")

        self.assertFalse(any("semantic" in r for r in reasons_user))
        # Score should be based only on path/content keywords, which will be lower
        self.assertLess(score_user, 0.5)