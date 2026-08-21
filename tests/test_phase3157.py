from __future__ import annotations

import tempfile
import unittest
from unittest import mock
from pathlib import Path

from local_agent.context import ContextSelector
from local_agent.models import (
    FileIndex,
    FileRelationship,
    ProjectContext,
    RepositoryMap,
    SemanticIndex,
    SymbolDefinition,
    SymbolLocation,
)


class Phase3157_DependencyAwareRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.repo_root = self.temp_dir / "repo"
        self.repo_root.mkdir()

        # Create dummy files
        (self.repo_root / "services/user_service.py").parent.mkdir(parents=True)
        (self.repo_root / "services/user_service.py").write_text("class UserService:\n  def save(self, user):\n    pass\n")
        (self.repo_root / "handlers/user_handler.py").parent.mkdir(parents=True)
        (self.repo_root / "handlers/user_handler.py").write_text("from services.user_service import UserService\n\nclass UserHandler:\n  pass\n")
        (self.repo_root / "utils/logger.py").parent.mkdir(parents=True)
        (self.repo_root / "utils/logger.py").write_text("def log_message(msg):\n  pass\n")
        (self.repo_root / "main.py").write_text("from handlers.user_handler import UserHandler\n")

        def symbol(name: str, kind: str, line: int, parent: str | None = None) -> SymbolDefinition:
            return SymbolDefinition(name=name, kind=kind, location=SymbolLocation(start_line=line, end_line=line + 1), parent=parent)

        self.semantic_index = SemanticIndex(
            files={
                "services/user_service.py": FileIndex(path="services/user_service.py", language="Python", content_hash="h1", symbols=[symbol("UserService", "class", 1), symbol("save", "method", 2, "UserService")]),
                "handlers/user_handler.py": FileIndex(path="handlers/user_handler.py", language="Python", content_hash="h2", symbols=[symbol("UserHandler", "class", 3)]),
                "utils/logger.py": FileIndex(path="utils/logger.py", language="Python", content_hash="h3", symbols=[symbol("log_message", "function", 1)]),
            }
        )

        self.repository_map = RepositoryMap(
            root=str(self.repo_root), project_metadata={}, languages=["Python"], frameworks=[],
            files=[],
            directories=["services", "handlers", "utils"], tests=[], configuration_files=[], entry_points=["main.py"],
            relationships=[
                FileRelationship(source="handlers/user_handler.py", target="services/user_service.py", kind="imports"),
                FileRelationship(source="main.py", target="handlers/user_handler.py", kind="imports"),
            ],
            ignored_paths=[], protected_paths=[],
        )

        self.project_context = ProjectContext(
            root=str(self.repo_root),
            source_files=list(self.semantic_index.files.keys()) + ["main.py"],
            metadata={"semantic_index": self.semantic_index, "repository_map": self.repository_map},
            repository_map=self.repository_map,
        )

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

    def test_direct_semantic_match_gets_highest_boost(self):
        task = "Fix the `UserService` class"
        score_service, reasons_service = self._get_file_score_and_reason(task, "services/user_service.py")
        self.assertGreater(score_service, 0.5, "Direct semantic match should get a high boost")
        self.assertTrue(any("semantic symbol definition match: UserService" in r for r in reasons_service))

    def test_related_file_gets_secondary_boost(self):
        task = "Fix the `UserService` class"
        score_handler, reasons_handler = self._get_file_score_and_reason(task, "handlers/user_handler.py")
        score_logger, _ = self._get_file_score_and_reason(task, "utils/logger.py")
        self.assertGreater(score_handler, 0.15, "Related file should get a secondary boost")
        self.assertLess(score_handler, 0.25, "Related file boost should be less than direct substring match")
        self.assertTrue(any("related to semantic match in services/user_service.py" in r for r in reasons_handler))
        self.assertLess(score_logger, 0.15, "Unrelated file should not get a semantic boost")

    def test_secondary_boost_does_not_outrank_direct_match(self):
        task = "Fix `UserService` and also `log_message`"
        score_service, _ = self._get_file_score_and_reason(task, "services/user_service.py")
        score_handler, _ = self._get_file_score_and_reason(task, "handlers/user_handler.py")
        score_logger, _ = self._get_file_score_and_reason(task, "utils/logger.py")
        self.assertGreater(score_service, score_handler, "Direct match on UserService should outrank its dependency")
        self.assertGreater(score_logger, score_handler, "Direct match on log_message should outrank a dependency of another match")

    def test_multiple_semantic_seeds_do_not_duplicate_boost(self):
        self.repository_map.relationships.append(FileRelationship(source="main.py", target="services/user_service.py", kind="imports"))
        self.semantic_index.files["main.py"] = FileIndex(path="main.py", language="Python", content_hash="h-main", symbols=[SymbolDefinition(name="main_func", kind="function", location=SymbolLocation(1,1))])
        task = "Fix `main_func` and `UserHandler`"
        score_service, reasons_service = self._get_file_score_and_reason(task, "services/user_service.py")
        self.assertGreater(score_service, 0.15)
        self.assertLess(score_service, 0.25, "Score should not be additive for multiple relations")
        relation_reasons = [r for r in reasons_service if "related to semantic match" in r]
        self.assertEqual(len(relation_reasons), 1, "Should only have one relationship boost reason")

    def test_missing_relationship_data_falls_back_safely(self):
        self.project_context.repository_map.relationships = []
        task = "Fix the `UserService` class"
        score_handler, reasons_handler = self._get_file_score_and_reason(task, "handlers/user_handler.py")
        self.assertLess(score_handler, 0.15, "Handler should not get a boost without relationship data")
        self.assertFalse(any("related to semantic match" in r for r in reasons_handler))

    def test_missing_semantic_index_falls_back_safely(self):
        self.project_context.metadata["semantic_index"] = None
        task = "Fix the `UserService` class"
        score_service, reasons_service = self._get_file_score_and_reason(task, "services/user_service.py")
        score_handler, reasons_handler = self._get_file_score_and_reason(task, "handlers/user_handler.py")
        self.assertFalse(any("semantic" in r for r in reasons_service))
        self.assertFalse(any("semantic" in r for r in reasons_handler))
        self.assertLess(score_service, 0.5)
        self.assertLess(score_handler, 0.15)

    def test_qualified_symbol_regression_with_dependency(self):
        task = "Fix `UserService.save`"
        score_service, reasons_service = self._get_file_score_and_reason(task, "services/user_service.py")
        score_handler, reasons_handler = self._get_file_score_and_reason(task, "handlers/user_handler.py")
        self.assertGreater(score_service, 0.7)
        self.assertTrue(any("semantic qualified symbol match: UserService.save" in r for r in reasons_service))
        self.assertGreater(score_handler, 0.15)
        self.assertLess(score_handler, 0.25)
        self.assertTrue(any("related to semantic match in services/user_service.py" in r for r in reasons_handler))
        self.assertGreater(score_service, score_handler)