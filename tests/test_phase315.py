import datetime
import json
import tempfile
import unittest
from pathlib import Path

from local_agent.models import (
    ExecutionResult, SymbolLocation, SymbolDefinition, FileIndex, SemanticIndex,
)
from local_agent.storage import JsonFileStorage

class Phase315_1Tests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.storage = JsonFileStorage(self.temp_dir / ".agent_data")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_symbol_location_round_trip(self):
        loc = SymbolLocation(start_line=10, end_line=20)
        data = loc.to_dict()
        reconstructed = SymbolLocation.from_dict(data)
        self.assertEqual(loc, reconstructed)
        self.assertIsInstance(reconstructed, SymbolLocation)

    def test_symbol_definition_round_trip(self):
        loc = SymbolLocation(start_line=10, end_line=20)
        sym = SymbolDefinition(name="my_function", kind="function", location=loc, parent=None)
        data = sym.to_dict()
        reconstructed = SymbolDefinition.from_dict(data)
        self.assertEqual(sym, reconstructed)
        self.assertIsInstance(reconstructed, SymbolDefinition)
        self.assertIsInstance(reconstructed.location, SymbolLocation)

    def test_file_index_round_trip(self):
        loc1 = SymbolLocation(start_line=1, end_line=5)
        sym1 = SymbolDefinition(name="MyClass", kind="class", location=loc1)
        loc2 = SymbolLocation(start_line=7, end_line=10)
        sym2 = SymbolDefinition(name="my_method", kind="method", location=loc2, parent="MyClass")
        
        file_idx = FileIndex(
            path="src/module.py",
            language="Python",
            content_hash="abc123def456",
            symbols=[sym1, sym2],
            imports=["math", "os"]
        )
        data = file_idx.to_dict()
        reconstructed = FileIndex.from_dict(data)
        self.assertEqual(file_idx, reconstructed)
        self.assertIsInstance(reconstructed, FileIndex)
        self.assertIsInstance(reconstructed.symbols[0], SymbolDefinition)
        self.assertIsInstance(reconstructed.symbols[0].location, SymbolLocation)

    def test_semantic_index_round_trip(self):
        loc1 = SymbolLocation(start_line=1, end_line=5)
        sym1 = SymbolDefinition(name="MyClass", kind="class", location=loc1)
        file_idx1 = FileIndex(path="src/module.py", language="Python", content_hash="hash1", symbols=[sym1])

        loc2 = SymbolLocation(start_line=10, end_line=15)
        sym2 = SymbolDefinition(name="another_func", kind="function", location=loc2)
        file_idx2 = FileIndex(path="src/other.js", language="JavaScript", content_hash="hash2", symbols=[sym2])

        sem_idx = SemanticIndex(files={"src/module.py": file_idx1, "src/other.js": file_idx2})
        data = sem_idx.to_dict()
        reconstructed = SemanticIndex.from_dict(data)
        self.assertEqual(sem_idx, reconstructed)
        self.assertIsInstance(reconstructed, SemanticIndex)
        self.assertIsInstance(reconstructed.files["src/module.py"], FileIndex)
        self.assertIsInstance(reconstructed.files["src/module.py"].symbols[0], SymbolDefinition)
        self.assertIsInstance(reconstructed.files["src/module.py"].symbols[0].location, SymbolLocation)

    def test_semantic_index_find_symbol(self):
        loc1 = SymbolLocation(start_line=1, end_line=5)
        sym1 = SymbolDefinition(name="common_func", kind="function", location=loc1)
        file_idx1 = FileIndex(path="src/module.py", language="Python", content_hash="hash1", symbols=[sym1])

        loc2 = SymbolLocation(start_line=10, end_line=15)
        sym2 = SymbolDefinition(name="another_func", kind="function", location=loc2)
        loc3 = SymbolLocation(start_line=20, end_line=25)
        sym3 = SymbolDefinition(name="common_func", kind="function", location=loc3)
        file_idx2 = FileIndex(path="src/other.js", language="JavaScript", content_hash="hash2", symbols=[sym2, sym3])

        sem_idx = SemanticIndex(files={"src/module.py": file_idx1, "src/other.js": file_idx2})

        found_unique = sem_idx.find_symbol("another_func")
        self.assertEqual(len(found_unique), 1)
        self.assertEqual(found_unique[0].name, "another_func")

        found_common = sem_idx.find_symbol("common_func")
        self.assertEqual(len(found_common), 2)
        self.assertIn(sym1, found_common)
        self.assertIn(sym3, found_common)

        found_none = sem_idx.find_symbol("non_existent_func")
        self.assertEqual(len(found_none), 0)

    def test_semantic_index_real_disk_persistence(self):
        loc1 = SymbolLocation(start_line=1, end_line=5)
        sym1 = SymbolDefinition(name="MyClass", kind="class", location=loc1)
        file_idx1 = FileIndex(path="src/module.py", language="Python", content_hash="hash1", symbols=[sym1])

        sem_idx = SemanticIndex(files={"src/module.py": file_idx1})
        self.storage.save_semantic_index(sem_idx)

        reloaded_storage = JsonFileStorage(self.temp_dir / ".agent_data")
        reconstructed = reloaded_storage.load_semantic_index()

        self.assertEqual(sem_idx, reconstructed)
        self.assertIsInstance(reconstructed, SemanticIndex)
        self.assertIsInstance(reconstructed.files["src/module.py"], FileIndex)
        self.assertIsInstance(reconstructed.files["src/module.py"].symbols[0], SymbolDefinition)
        self.assertIsInstance(reconstructed.files["src/module.py"].symbols[0].location, SymbolLocation)
        self.assertEqual(reconstructed.files["src/module.py"].symbols[0].name, "MyClass")

    def test_load_empty_semantic_index_file(self):
        reconstructed = self.storage.load_semantic_index()
        self.assertIsInstance(reconstructed, SemanticIndex)
        self.assertEqual(len(reconstructed.files), 0)

    def test_load_corrupted_semantic_index_file(self):
        (self.storage.base_dir / "semantic_index.json").write_text("this is not json")
        reconstructed = self.storage.load_semantic_index()
        self.assertIsInstance(reconstructed, SemanticIndex)
        self.assertEqual(len(reconstructed.files), 0)