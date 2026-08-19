from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_agent.models import SemanticIndex
from local_agent.repository import RepositoryIntelligence

# Check if tree-sitter is available to decide if tests should be skipped.
try:
    # This is a proxy for checking if the whole tree-sitter setup is working
    from local_agent.indexing.parser import TreeSitterParser
    # We also need to check if the language library is built
    if not TreeSitterParser.language_library_exists():
        raise ImportError("Language library not found")
    TREE_SITTER_AVAILABLE = True
except (ImportError, FileNotFoundError):
    TREE_SITTER_AVAILABLE = False


@unittest.skipUnless(TREE_SITTER_AVAILABLE, "Tree-sitter is not installed or grammars not built, skipping semantic indexing integration tests.")
class Phase3158_RepositoryIntelligenceIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.root = self.temp_dir / "repo"
        self.root.mkdir()
        self.files = {
            "src/app.py": "def main_func():\n    pass\n",
            "src/component.js": "export function MyComponent() {}\n",
            "README.md": "Test repo",
        }
        for relative, content in self.files.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_initial_scan_creates_correct_index(self):
        # Arrange
        ri = RepositoryIntelligence(self.root)

        # Act
        context = ri.scan()
        index = context.metadata.get("semantic_index")

        # Assert
        self.assertIsInstance(index, SemanticIndex)
        self.assertIn("src/app.py", index.files)
        self.assertIn("src/component.js", index.files)

        py_file_index = index.files["src/app.py"]
        self.assertEqual(len(py_file_index.symbols), 1)
        self.assertEqual(py_file_index.symbols[0].name, "main_func")
        self.assertEqual(py_file_index.symbols[0].kind, "function")

        js_file_index = index.files["src/component.js"]
        self.assertEqual(len(js_file_index.symbols), 1)
        self.assertEqual(js_file_index.symbols[0].name, "MyComponent")
        self.assertEqual(js_file_index.symbols[0].kind, "function")

    def test_incremental_scan_updates_modified_file_only(self):
        # Arrange
        ri = RepositoryIntelligence(self.root)

        # Act 1: Initial scan
        ri.scan()
        index1 = ri.storage.load_semantic_index()
        hash1_py = index1.files["src/app.py"].content_hash
        hash1_js = index1.files["src/component.js"].content_hash

        # Modify one file
        (self.root / "src/app.py").write_text("def main_func():\n    print('updated')\nclass NewClass:\n    pass\n", encoding="utf-8")

        # Act 2: Incremental scan
        ri.scan()
        index2 = ri.storage.load_semantic_index()
        hash2_py = index2.files["src/app.py"].content_hash
        hash2_js = index2.files["src/component.js"].content_hash

        # Assert
        self.assertNotEqual(hash1_py, hash2_py, "Hash of modified file should change")
        self.assertEqual(hash1_js, hash2_js, "Hash of unmodified file should not change")

        py_file_index_2 = index2.files["src/app.py"]
        self.assertEqual(len(py_file_index_2.symbols), 2)
        symbol_names = {s.name for s in py_file_index_2.symbols}
        self.assertIn("main_func", symbol_names)
        self.assertIn("NewClass", symbol_names)

    def test_incremental_scan_removes_deleted_file(self):
        # Arrange
        ri = RepositoryIntelligence(self.root)

        # Act 1: Initial scan
        ri.scan()
        index1 = ri.storage.load_semantic_index()
        self.assertIn("src/app.py", index1.files)

        # Delete one file
        (self.root / "src/app.py").unlink()

        # Act 2: Incremental scan
        ri.scan()
        index2 = ri.storage.load_semantic_index()

        # Assert
        self.assertNotIn("src/app.py", index2.files, "Deleted file should be removed from index")
        self.assertIn("src/component.js", index2.files, "Other files should remain in index")