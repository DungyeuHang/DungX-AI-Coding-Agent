from __future__ import annotations

import dataclasses
import json
import unittest

from local_agent.models import (
    ProviderCapability,
    ToolCall,
    ToolDefinition,
    ToolResult,
)


class ToolModelsTests(unittest.TestCase):
    def test_provider_capability_tool_use_exists(self):
        self.assertTrue(hasattr(ProviderCapability, "TOOL_USE"))
        self.assertEqual(ProviderCapability.TOOL_USE.value, "tool_use")
        self.assertEqual(ProviderCapability("tool_use"), ProviderCapability.TOOL_USE)

    def test_tool_definition_construction_and_defaults(self):
        tool_def = ToolDefinition(
            name="read_file_range",
            description="Reads a slice of lines from a file.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                },
                "required": ["path"],
            },
        )
        self.assertEqual(tool_def.name, "read_file_range")
        self.assertEqual(tool_def.description, "Reads a slice of lines from a file.")
        self.assertIn("properties", tool_def.parameters)
        self.assertEqual(tool_def.parameters["required"], ["path"])

    def test_tool_definition_immutability(self):
        tool_def = ToolDefinition(
            name="grep_code",
            description="Search regex in files",
            parameters={"type": "object"},
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            tool_def.name = "other_name"  # type: ignore

    def test_tool_definition_validation(self):
        with self.assertRaises(ValueError):
            ToolDefinition(name="", description="valid", parameters={})
        with self.assertRaises(ValueError):
            ToolDefinition(name="valid", description="", parameters={})
        with self.assertRaises(ValueError):
            ToolDefinition(name="valid", description="valid", parameters="invalid")  # type: ignore

    def test_tool_definition_serialization_round_trip(self):
        original = ToolDefinition(
            name="search_symbols",
            description="Find symbols in project",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        )
        d = original.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["name"], "search_symbols")
        self.assertEqual(d["description"], "Find symbols in project")
        # JSON serializable
        json_str = json.dumps(d)
        self.assertIn('"name": "search_symbols"', json_str)
        # Restore
        restored = ToolDefinition.from_dict(d)
        self.assertEqual(restored, original)

    def test_tool_call_construction_and_immutability(self):
        call = ToolCall(
            call_id="call-123",
            tool_name="read_file_range",
            arguments={"path": "src/app.py", "start_line": 1, "end_line": 20},
        )
        self.assertEqual(call.call_id, "call-123")
        self.assertEqual(call.tool_name, "read_file_range")
        self.assertEqual(call.arguments["path"], "src/app.py")

        with self.assertRaises(dataclasses.FrozenInstanceError):
            call.call_id = "call-456"  # type: ignore

    def test_tool_call_validation(self):
        with self.assertRaises(ValueError):
            ToolCall(call_id="", tool_name="tool", arguments={})
        with self.assertRaises(ValueError):
            ToolCall(call_id="id", tool_name="", arguments={})
        with self.assertRaises(ValueError):
            ToolCall(call_id="id", tool_name="tool", arguments="invalid")  # type: ignore

    def test_tool_call_serialization_round_trip(self):
        original = ToolCall(
            call_id="call-abc",
            tool_name="grep_code",
            arguments={"pattern": "class UserService", "glob": "*.py"},
        )
        d = original.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["call_id"], "call-abc")
        self.assertEqual(d["tool_name"], "grep_code")
        self.assertEqual(d["arguments"]["pattern"], "class UserService")
        # JSON serializable
        json_str = json.dumps(d)
        self.assertIn('"call_id": "call-abc"', json_str)
        # Restore
        restored = ToolCall.from_dict(d)
        self.assertEqual(restored, original)

    def test_tool_result_construction_and_defaults(self):
        result = ToolResult(
            call_id="call-123",
            tool_name="read_file_range",
            output="def hello(): return 'world'\n",
        )
        self.assertEqual(result.call_id, "call-123")
        self.assertEqual(result.tool_name, "read_file_range")
        self.assertEqual(result.output, "def hello(): return 'world'\n")
        self.assertFalse(result.is_error)
        self.assertFalse(result.truncated)

    def test_tool_result_custom_flags_and_immutability(self):
        result = ToolResult(
            call_id="call-456",
            tool_name="read_file_range",
            output="File too large... [truncated]",
            is_error=False,
            truncated=True,
        )
        self.assertTrue(result.truncated)
        self.assertFalse(result.is_error)

        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.is_error = True  # type: ignore

    def test_tool_result_error_flag(self):
        result = ToolResult(
            call_id="call-789",
            tool_name="run_command_sandbox",
            output="Permission denied: rm command is blocked",
            is_error=True,
            truncated=False,
        )
        self.assertTrue(result.is_error)
        self.assertFalse(result.truncated)

    def test_tool_result_validation(self):
        with self.assertRaises(ValueError):
            ToolResult(call_id="", tool_name="tool", output="ok")
        with self.assertRaises(ValueError):
            ToolResult(call_id="id", tool_name="", output="ok")
        with self.assertRaises(ValueError):
            ToolResult(call_id="id", tool_name="tool", output=123)  # type: ignore

    def test_tool_result_serialization_round_trip(self):
        original = ToolResult(
            call_id="call-xyz",
            tool_name="find_files",
            output="src/app.py\nsrc/models.py",
            is_error=False,
            truncated=True,
        )
        d = original.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["call_id"], "call-xyz")
        self.assertEqual(d["tool_name"], "find_files")
        self.assertEqual(d["output"], "src/app.py\nsrc/models.py")
        self.assertFalse(d["is_error"])
        self.assertTrue(d["truncated"])
        # JSON serializable
        json_str = json.dumps(d)
        self.assertIn('"call_id": "call-xyz"', json_str)
        # Restore
        restored = ToolResult.from_dict(d)
        self.assertEqual(restored, original)


if __name__ == "__main__":
    unittest.main()

