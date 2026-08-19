from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_agent.models import SemanticIndex
from local_agent.repository import RepositoryIntelligence


class RepositoryIntelligenceFallbackTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.root = self.temp_dir / "repo"
        self.root.mkdir()
        (self.root / "src/app.py").write_text("def main_func():\n    pass\n", encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @mock.patch("local_agent.repository.TreeSitterParser")
    def test_scan_with_treesitter_unavailable_falls_back_gracefully(self, MockTreeSitterParser):
        # Arrange
        MockTreeSitterParser.side_effect = ImportError("No tree-sitter today")
        ri = RepositoryIntelligence(self.root)

        # Act
        context = ri.scan()
        index = context.metadata.get("semantic_index")

        # Assert
        self.assertIsInstance(index, SemanticIndex)
        self.assertIn("src/app.py", index.files)
        py_file_index = index.files["src/app.py"]
        self.assertEqual(len(py_file_index.symbols), 0, "Symbols list should be empty on fallback")
        self.assertNotEqual(py_file_index.content_hash, "", "Content hash should still be calculated")