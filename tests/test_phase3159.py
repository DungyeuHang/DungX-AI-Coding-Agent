import unittest
from pathlib import Path

class Phase3159_DocumentationTests(unittest.TestCase):
    def test_roadmap_updated_for_semantic_indexing(self):
        readme_path = Path(__file__).parent.parent / "README.md"
        self.assertTrue(readme_path.exists(), "README.md not found")
        content = readme_path.read_text(encoding="utf-8")
        
        self.assertIn("1. Add semantic code search and incremental project indexing. (Completed in Phase 3.15)", content)
        self.assertIn("semantic code indexing", content)
        self.assertIn("qualified symbol matches (e.g., `UserService.save`)", content)
        self.assertIn("dependency-aware retrieval", content)
        self.assertIn("maintains an incremental semantic code index", content)