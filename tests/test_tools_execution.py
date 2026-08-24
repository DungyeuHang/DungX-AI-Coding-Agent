from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from local_agent.models import (
    FileIndex,
    SemanticIndex,
    SymbolDefinition,
    SymbolLocation,
    ToolCall,
)
from local_agent.tools import ToolRegistry


class ToolsExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name).resolve()

        # Create structured code repository files
        src_dir = self.root / "src"
        src_dir.mkdir(parents=True, exist_ok=True)

        self.sample_code = (
            "def add(a: int, b: int) -> int:\n"
            "    \"\"\"Return sum of two numbers.\"\"\"\n"
            "    return a + b\n"
            "\n"
            "def multiply(a: int, b: int) -> int:\n"
            "    \"\"\"Return product.\"\"\"\n"
            "    return a * b\n"
            "\n"
            "class Calculator:\n"
            "    def calculate(self, expr: str) -> int:\n"
            "        return eval(expr)\n"
        )
        (src_dir / "math_utils.py").write_text(self.sample_code, encoding="utf-8")

        # Large file for truncation testing
        large_content = "\n".join(f"Line {i}: some repetitive test data here" for i in range(1, 300))
        (src_dir / "large.txt").write_text(large_content, encoding="utf-8")

        # Setup Semantic Index double
        sym1 = SymbolDefinition(name="add", kind="function", location=SymbolLocation(start_line=1, end_line=3))
        sym2 = SymbolDefinition(name="Calculator", kind="class", location=SymbolLocation(start_line=9, end_line=11))
        self.semantic_index = SemanticIndex(
            files={
                "src/math_utils.py": FileIndex(
                    path="src/math_utils.py",
                    language="Python",
                    content_hash="abc",
                    symbols=[sym1, sym2],
                )
            }
        )

        self.registry = ToolRegistry(
            self.root,
            semantic_index=self.semantic_index,
            max_output_bytes=500,
            max_results=5,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    # -------------------------------------------------------------------------
    # read_file_range tests
    # -------------------------------------------------------------------------

    def test_read_file_range_returns_numbered_lines(self):
        call = ToolCall("c1", "read_file_range", {"path": "src/math_utils.py", "start_line": 1, "end_line": 3})
        result = self.registry.execute(call)
        self.assertFalse(result.is_error)
        self.assertFalse(result.truncated)
        lines = result.output.splitlines()
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[0].startswith("1: def add"))
        self.assertTrue(lines[2].startswith("3:     return a + b"))

    def test_read_file_range_default_arguments(self):
        call = ToolCall("c2", "read_file_range", {"path": "src/math_utils.py"})
        result = self.registry.execute(call)
        self.assertFalse(result.is_error)
        self.assertIn("1: def add", result.output)
        self.assertIn("9: class Calculator", result.output)

    def test_read_file_range_invalid_ranges(self):
        call1 = ToolCall("c3", "read_file_range", {"path": "src/math_utils.py", "start_line": 0})
        res1 = self.registry.execute(call1)
        self.assertTrue(res1.is_error)
        self.assertIn("Invalid 'start_line'", res1.output)

        call2 = ToolCall("c4", "read_file_range", {"path": "src/math_utils.py", "start_line": 10, "end_line": 5})
        res2 = self.registry.execute(call2)
        self.assertTrue(res2.is_error)
        self.assertIn("Invalid line range", res2.output)

    def test_read_file_range_truncation_on_large_slice(self):
        call = ToolCall("c5", "read_file_range", {"path": "src/large.txt", "start_line": 1, "end_line": 200})
        result = self.registry.execute(call)
        self.assertFalse(result.is_error)
        self.assertTrue(result.truncated)
        self.assertIn("[output truncated: exceeded maximum output limit]", result.output)

    # -------------------------------------------------------------------------
    # search_symbols tests
    # -------------------------------------------------------------------------

    def test_search_symbols_found(self):
        call = ToolCall("c6", "search_symbols", {"symbol_name": "calc"})
        result = self.registry.execute(call)
        self.assertFalse(result.is_error)
        self.assertIn("Calculator (class) in src/math_utils.py:9", result.output)

    def test_search_symbols_not_found(self):
        call = ToolCall("c7", "search_symbols", {"symbol_name": "non_existent_symbol"})
        result = self.registry.execute(call)
        self.assertFalse(result.is_error)
        self.assertIn("No symbols found", result.output)

    def test_search_symbols_without_semantic_index(self):
        reg = ToolRegistry(self.root, semantic_index=None)
        call = ToolCall("c8", "search_symbols", {"symbol_name": "add"})
        result = reg.execute(call)
        self.assertFalse(result.is_error)
        self.assertIn("No semantic index available", result.output)

    # -------------------------------------------------------------------------
    # grep_code tests
    # -------------------------------------------------------------------------

    def test_grep_code_matches(self):
        call = ToolCall("c9", "grep_code", {"pattern": "def multiply"})
        result = self.registry.execute(call)
        self.assertFalse(result.is_error)
        self.assertIn("src/math_utils.py:5: def multiply", result.output)

    def test_grep_code_case_insensitive(self):
        call = ToolCall("c10", "grep_code", {"pattern": "CALCULATOR", "case_sensitive": False})
        result = self.registry.execute(call)
        self.assertFalse(result.is_error)
        self.assertIn("src/math_utils.py:9: class Calculator", result.output)

    def test_grep_code_glob_filtering(self):
        call = ToolCall("c11", "grep_code", {"pattern": "Line 1:", "glob": "*.py"})
        result = self.registry.execute(call)
        self.assertFalse(result.is_error)
        self.assertIn("No matches found", result.output)

        call2 = ToolCall("c12", "grep_code", {"pattern": "Line 1:", "glob": "*.txt"})
        result2 = self.registry.execute(call2)
        self.assertFalse(result2.is_error)
        self.assertIn("src/large.txt:1: Line 1:", result2.output)

    # -------------------------------------------------------------------------
    # find_files tests
    # -------------------------------------------------------------------------

    def test_find_files_matching(self):
        call = ToolCall("c13", "find_files", {"pattern": "*.py"})
        result = self.registry.execute(call)
        self.assertFalse(result.is_error)
        self.assertIn("src/math_utils.py", result.output)

    def test_find_files_no_match(self):
        call = ToolCall("c14", "find_files", {"pattern": "*.rs"})
        result = self.registry.execute(call)
        self.assertFalse(result.is_error)
        self.assertIn("No files found matching pattern", result.output)

    # -------------------------------------------------------------------------
    # run_command_sandbox tests
    # -------------------------------------------------------------------------

    def test_run_command_sandbox_safe_execution(self):
        import sys
        call = ToolCall("c15", "run_command_sandbox", {"command": [sys.executable, "-c", "print('sandbox test output')"]})
        result = self.registry.execute(call)
        self.assertFalse(result.is_error)
        self.assertIn("sandbox test output", result.output)

    def test_run_command_sandbox_nonzero_exit_code(self):
        import sys
        call = ToolCall("c16", "run_command_sandbox", {"command": [sys.executable, "-c", "import sys; sys.exit(42)"]})
        result = self.registry.execute(call)
        self.assertTrue(result.is_error)
        self.assertIn("Command failed with exit code 42", result.output)

    # -------------------------------------------------------------------------
    # Tool definitions API
    # -------------------------------------------------------------------------

    def test_definitions_exposes_all_five_tools(self):
        defs = self.registry.definitions()
        names = {d.name for d in defs}
        expected = {
            "read_file_range",
            "search_symbols",
            "grep_code",
            "find_files",
            "run_command_sandbox",
        }
        self.assertEqual(names, expected)


if __name__ == "__main__":
    unittest.main()
