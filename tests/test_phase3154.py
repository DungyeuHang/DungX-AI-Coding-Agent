from __future__ import annotations

import json
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


class TestPhase3154ContextSelectorIntegration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.repo_root = self.temp_dir / "repo"
        self.repo_root.mkdir()

        # Create dummy Python files for semantic indexing
        (self.repo_root / "my_module.py").write_text(
            "def my_function():\n    pass\n\nclass MyClass:\n    def __init__(self):\n        pass\n    def my_method(self):\n        pass\n"
        )
        (self.repo_root / "user_service.py").write_text(
            "class UserService:\n    def __init__(self):\n        pass\n    def save(self, user):\n        pass\n"
        )
        (self.repo_root / "another_module.py").write_text(
            "def another_function():\n    pass\n"
        )
        (self.repo_root / "duplicate_function.py").write_text(
            "def my_function():\n    # This is a duplicate\n    pass\n"
        )
        (self.repo_root / "unrelated.txt").write_text("some text content")
        (self.repo_root / "config.json").write_text('{"key": "value"}')

        # Build the semantic data directly: these tests verify ContextSelector
        # consumption, independently of the Tree-sitter indexing pipeline.
        def symbol(name: str, kind: str, line: int, parent: str | None = None) -> SymbolDefinition:
            return SymbolDefinition(
                name=name,
                kind=kind,
                location=SymbolLocation(start_line=line, end_line=line + 1),
                parent=parent,
            )

        self.semantic_index = SemanticIndex(
            files={
                "my_module.py": FileIndex(
                    path="my_module.py",
                    language="Python",
                    content_hash="my-module",
                    symbols=[
                        symbol("my_function", "function", 1),
                        symbol("MyClass", "class", 4),
                        symbol("__init__", "method", 5, "MyClass"),
                        symbol("my_method", "method", 7, "MyClass"),
                    ],
                ),
                "user_service.py": FileIndex(
                    path="user_service.py",
                    language="Python",
                    content_hash="user-service",
                    symbols=[
                        symbol("UserService", "class", 1),
                        symbol("__init__", "method", 2, "UserService"),
                        symbol("save", "method", 4, "UserService"),
                    ],
                ),
                "another_module.py": FileIndex(
                    path="another_module.py",
                    language="Python",
                    content_hash="another-module",
                    symbols=[symbol("another_function", "function", 1)],
                ),
                "duplicate_function.py": FileIndex(
                    path="duplicate_function.py",
                    language="Python",
                    content_hash="duplicate-function",
                    symbols=[symbol("my_function", "function", 1)],
                ),
            }
        )
        self.project_context = ProjectContext(
            root=str(self.repo_root),
            source_files=[
                "my_module.py",
                "user_service.py",
                "another_module.py",
                "duplicate_function.py",
            ],
            config_files=["config.json"],
            documentation_files=["unrelated.txt"],
            metadata={"semantic_index": self.semantic_index},
        )

        self.context_selector = ContextSelector(self.repo_root)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir)

    def _get_ranked_paths(self, task: str) -> list[str]:
        selected_context = self.context_selector.select(task, self.project_context)
        # The selected_items are already sorted by score descending
        return [item["path"] for item in selected_context.metadata["context_selection"]["selected_items"]]

    def _get_file_score(self, task: str, file_path: str) -> float:
        selected_context = self.context_selector.select(task, self.project_context)
        for item in selected_context.metadata["context_selection"]["selected_items"]:
            if item["path"] == file_path:
                return item["score"]
        return 0.0

    def test_1_function_definition_priority(self):
        task = "Fix `my_function` in my_module.py"
        ranked_paths = self._get_ranked_paths(task)
        self.assertIn("my_module.py", ranked_paths)
        self.assertGreater(
            self._get_file_score(task, "my_module.py"),
            self._get_file_score(task, "another_module.py"),
            "my_module.py should be prioritized due to semantic match",
        )
        reasons = self.context_selector.select(task, self.project_context).metadata["context_selection"]["selected_items"][0]["reason"]
        self.assertTrue(any(reason.startswith("semantic symbol definition match:") for reason in reasons))

    def test_2_class_definition_priority(self):
        task = "Implement feature for `UserService`"
        ranked_paths = self._get_ranked_paths(task)
        self.assertIn("user_service.py", ranked_paths)
        self.assertGreater(
            self._get_file_score(task, "user_service.py"),
            self._get_file_score(task, "my_module.py"),
            "user_service.py should be prioritized due to semantic match",
        )
        reasons = self.context_selector.select(task, self.project_context).metadata["context_selection"]["selected_items"][0]["reason"]
        self.assertTrue(any(reason.startswith("semantic symbol definition match:") for reason in reasons))

    def test_3_multiple_definitions(self):
        task = "Refactor `my_function`"
        ranked_paths = self._get_ranked_paths(task)
        self.assertIn("my_module.py", ranked_paths)
        self.assertIn("duplicate_function.py", ranked_paths)

        score_my_module = self._get_file_score(task, "my_module.py")
        score_duplicate_function = self._get_file_score(task, "duplicate_function.py")
        self.assertGreater(score_my_module, 0.5, "my_module.py should receive semantic boost")
        self.assertGreater(
            score_duplicate_function, 0.5, "duplicate_function.py should receive semantic boost"
        )

    def test_4_no_semantic_match(self):
        task = "Update `non_existent_symbol`"
        selected_context = self.context_selector.select(task, self.project_context)
        selected_items = selected_context.metadata["context_selection"]["selected_items"]
        self.assertTrue(selected_items)
        self.assertNotIn(
            "semantic symbol definition",
            [
                reason
                for item in selected_items
                for reason in item["reason"]
            ],
        )
        # A non-matching symbol must not receive a semantic score boost.
        self.assertLess(self._get_file_score(task, "my_module.py"), 0.5)

        self.project_context.metadata["semantic_index"] = None
        no_index_items = self.context_selector.select(task, self.project_context).metadata["context_selection"]["selected_items"]
        self.assertEqual(
            [item["path"] for item in selected_items],
            [item["path"] for item in no_index_items],
        )

    def test_5_empty_or_missing_semantic_index(self):
        # Test missing semantic index
        self.project_context.metadata["semantic_index"] = None
        task = "Fix `my_function`"
        ranked_paths_no_index = self._get_ranked_paths(task)
        score_my_module_no_index = self._get_file_score(task, "my_module.py")
        self.assertLess(
            score_my_module_no_index, 0.5, "Score should not have semantic boost without index"
        )

        # Test empty semantic index
        self.project_context.metadata["semantic_index"] = SemanticIndex(files={})
        ranked_paths_empty_index = self._get_ranked_paths(task)
        score_my_module_empty_index = self._get_file_score(task, "my_module.py")
        self.assertLess(
            score_my_module_empty_index, 0.5, "Score should not have semantic boost with empty index"
        )

        # Ensure behavior is consistent with no semantic boost
        self.assertEqual(ranked_paths_no_index, ranked_paths_empty_index)

    def test_6_backtick_symbol_matching(self):
        task = "Implement `MyClass` functionality"
        ranked_paths = self._get_ranked_paths(task)
        self.assertIn("my_module.py", ranked_paths)
        self.assertGreater(
            self._get_file_score(task, "my_module.py"),
            self._get_file_score(task, "user_service.py"),
            "my_module.py should be prioritized for `MyClass`",
        )

    def test_7_qualified_symbol_matching(self):
        task = "Debug `UserService.save` method"
        ranked_paths = self._get_ranked_paths(task)
        self.assertIn("user_service.py", ranked_paths)
        self.assertGreater(
            self._get_file_score(task, "user_service.py"),
            self._get_file_score(task, "my_module.py"),
            "user_service.py should be prioritized for `UserService.save`",
        )
        reasons = self.context_selector.select(task, self.project_context).metadata["context_selection"]["selected_items"][0]["reason"]
        self.assertTrue(any(reason.startswith("semantic symbol definition match:") for reason in reasons))

    def test_8_regression_existing_context_selector_behavior(self):
        # Test a generic task to ensure existing heuristics still work
        task = "Refactor the application configuration"
        selected_context = self.context_selector.select(task, self.project_context)
        selected_items = selected_context.metadata["context_selection"]["selected_items"]
        self.assertTrue(selected_items)
        self.assertGreater(
            self._get_file_score(task, "another_module.py"),
            self._get_file_score(task, "unrelated.txt"),
            "existing source-file ranking should remain active",
        )
        # Generic tasks must not receive semantic symbol reasons or boosts.
        self.assertNotIn(
            "semantic symbol definition",
            [
                reason
                for item in selected_items
                for reason in item["reason"]
            ],
        )

    def test_extract_candidate_symbols_logic(self):
        # Test backtick-wrapped
        self.assertEqual(
            self.context_selector._extract_candidate_symbols("Fix `my_function`"),
            {"my_function"},
        )
        # Test plain identifier
        self.assertEqual(
            self.context_selector._extract_candidate_symbols("Update UserService"),
            {"UserService"},
        )
        # Test qualified name
        self.assertEqual(
            self.context_selector._extract_candidate_symbols("Debug UserService.save"),
            {"UserService.save", "save"},
        )
        # Test mixed
        self.assertEqual(
            self.context_selector._extract_candidate_symbols("Fix `my_function` and UserService.save"),
            {"my_function", "UserService.save", "save"},
        )
        # Test with stop words and short words
        self.assertEqual(
            self.context_selector._extract_candidate_symbols("A fix for `the_bug` in my_app"),
            {"the_bug", "my_app"},
        )
        # Test with numbers
        self.assertEqual(
            self.context_selector._extract_candidate_symbols("Handle API_V2_Client"),
            {"API_V2_Client"},
        )
