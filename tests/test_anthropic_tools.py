from __future__ import annotations

import io
import json
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch
import urllib.error

from local_agent.config import AgentConfig
from local_agent.models import (
    AuthenticationError,
    FileOperation,
    InvalidRequestError,
    ModelUnavailableError,
    Plan,
    ProjectContext,
    ProviderCapability,
    ProviderError,
    QuotaExceededError,
    RateLimitError,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from local_agent.providers import (
    AnthropicProvider,
    _format_anthropic_tools,
)
from local_agent.tool_engine import (
    ToolContextCompactor,
    ToolEngine,
)
from local_agent.tools import ToolRegistry


class AnthropicToolsTests(unittest.TestCase):
    def setUp(self):
        self.config_anthropic = AgentConfig(
            project=Path("."),
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            api_key="test-key-anthropic",
        )
        self.context = ProjectContext(root="/test/project")
        self.plan = Plan(objective="Implement Anthropic tool support", files_likely_to_change=["src/main.py"])
        self.test_tool = ToolDefinition(
            name="read_file_range",
            description="Read lines from file",
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
        self.search_tool = ToolDefinition(
            name="find_files",
            description="Find files matching pattern",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                },
                "required": ["pattern"],
            },
        )

    # -------------------------------------------------------------------------
    # 1. Capability Declaration
    # -------------------------------------------------------------------------

    def test_anthropic_capability_advertising(self):
        """AnthropicProvider advertises TOOL_USE in its capabilities."""
        provider = AnthropicProvider(self.config_anthropic)
        self.assertIn(ProviderCapability.PLANNING, provider.capabilities)
        self.assertIn(ProviderCapability.IMPLEMENTATION, provider.capabilities)
        self.assertIn(ProviderCapability.REPAIR, provider.capabilities)
        self.assertIn(ProviderCapability.REVIEW, provider.capabilities)
        self.assertIn(ProviderCapability.TOOL_USE, provider.capabilities)

    # -------------------------------------------------------------------------
    # 2. Tool Schema Serialization
    # -------------------------------------------------------------------------

    def test_format_anthropic_tools(self):
        """ToolDefinition parameters are serialized under 'input_schema' for Anthropic."""
        formatted = _format_anthropic_tools([self.test_tool, self.search_tool])
        self.assertEqual(len(formatted), 2)
        self.assertEqual(formatted[0]["name"], "read_file_range")
        self.assertEqual(formatted[0]["description"], "Read lines from file")
        self.assertEqual(formatted[0]["input_schema"], self.test_tool.parameters)
        self.assertEqual(formatted[1]["name"], "find_files")
        self.assertEqual(formatted[1]["input_schema"], self.search_tool.parameters)

    # -------------------------------------------------------------------------
    # 3. Single Tool Use Response Parsing
    # -------------------------------------------------------------------------

    @patch("urllib.request.urlopen")
    def test_anthropic_single_tool_use_parsing(self, mock_urlopen):
        """Anthropic tool_use content block parses into ToolCall."""
        mock_payload = {
            "id": "msg_01",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-sonnet-20241022",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_01A",
                    "name": "read_file_range",
                    "input": {"path": "src/main.py", "start_line": 1, "end_line": 20},
                }
            ],
            "stop_reason": "tool_use",
        }

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(mock_payload).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        provider = AnthropicProvider(self.config_anthropic)
        result = provider.generate_code_with_tools(
            "Inspect file", self.plan, self.context, [self.test_tool]
        )

        self.assertIsInstance(result, ToolCall)
        self.assertEqual(result.call_id, "toolu_01A")
        self.assertEqual(result.tool_name, "read_file_range")
        self.assertEqual(result.arguments, {"path": "src/main.py", "start_line": 1, "end_line": 20})

    # -------------------------------------------------------------------------
    # 4. Multiple Tool Use Blocks
    # -------------------------------------------------------------------------

    @patch("urllib.request.urlopen")
    def test_anthropic_multiple_tool_use_blocks(self, mock_urlopen):
        """First valid tool_use block is returned when multiple blocks exist."""
        mock_payload = {
            "id": "msg_02",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_01_first",
                    "name": "read_file_range",
                    "input": {"path": "src/first.py"},
                },
                {
                    "type": "tool_use",
                    "id": "toolu_02_second",
                    "name": "find_files",
                    "input": {"pattern": "*.py"},
                },
            ],
            "stop_reason": "tool_use",
        }

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(mock_payload).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        provider = AnthropicProvider(self.config_anthropic)
        result = provider.generate_code_with_tools(
            "Multi tool call", self.plan, self.context, [self.test_tool, self.search_tool]
        )

        self.assertIsInstance(result, ToolCall)
        self.assertEqual(result.call_id, "toolu_01_first")
        self.assertEqual(result.tool_name, "read_file_range")
        self.assertEqual(result.arguments, {"path": "src/first.py"})

    # -------------------------------------------------------------------------
    # 5. Tool Result Serialization & Multi-turn History
    # -------------------------------------------------------------------------

    @patch("urllib.request.urlopen")
    def test_anthropic_multi_turn_history_serialization(self, mock_urlopen):
        """Tool history is serialized into alternating assistant tool_use and user tool_result blocks."""
        captured_body: dict | None = None

        def fake_urlopen(req, *args, **kwargs):
            nonlocal captured_body
            captured_body = json.loads(req.data.decode("utf-8"))
            mock_payload = {
                "id": "msg_03",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_03",
                        "name": "find_files",
                        "input": {"pattern": "*.ts"},
                    }
                ],
                "stop_reason": "tool_use",
            }
            resp = MagicMock()
            resp.read.return_value = json.dumps(mock_payload).encode("utf-8")
            resp.__enter__.return_value = resp
            return resp

        mock_urlopen.side_effect = fake_urlopen

        provider = AnthropicProvider(self.config_anthropic)
        call1 = ToolCall("call_1", "read_file_range", {"path": "src/main.py"})
        res1 = ToolResult("call_1", "read_file_range", "def hello(): pass\n", False, False)
        call2 = ToolCall("call_2", "run_command_sandbox", {"command": ["pytest"]})
        res2 = ToolResult("call_2", "run_command_sandbox", "Error: failed", True, False)

        history = [(call1, res1), (call2, res2)]
        result = provider.generate_code_with_tools(
            "Task with history", self.plan, self.context, [self.test_tool, self.search_tool], tool_history=history
        )

        self.assertIsInstance(result, ToolCall)
        self.assertIsNotNone(captured_body)
        messages = captured_body["messages"]

        # Initial user turn + 2 tool turns (each has assistant tool_use and user tool_result) = 5 messages total
        self.assertEqual(len(messages), 5)

        # Message 0: Initial user message
        self.assertEqual(messages[0]["role"], "user")

        # Message 1: Assistant tool_use for call 1
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[1]["content"][0]["type"], "tool_use")
        self.assertEqual(messages[1]["content"][0]["id"], "call_1")
        self.assertEqual(messages[1]["content"][0]["name"], "read_file_range")
        self.assertEqual(messages[1]["content"][0]["input"], {"path": "src/main.py"})

        # Message 2: User tool_result for call 1
        self.assertEqual(messages[2]["role"], "user")
        self.assertEqual(messages[2]["content"][0]["type"], "tool_result")
        self.assertEqual(messages[2]["content"][0]["tool_use_id"], "call_1")
        self.assertEqual(messages[2]["content"][0]["content"], "def hello(): pass\n")
        self.assertFalse(messages[2]["content"][0]["is_error"])

        # Message 3: Assistant tool_use for call 2
        self.assertEqual(messages[3]["role"], "assistant")
        self.assertEqual(messages[3]["content"][0]["type"], "tool_use")
        self.assertEqual(messages[3]["content"][0]["id"], "call_2")

        # Message 4: User tool_result for call 2 (error)
        self.assertEqual(messages[4]["role"], "user")
        self.assertEqual(messages[4]["content"][0]["type"], "tool_result")
        self.assertEqual(messages[4]["content"][0]["tool_use_id"], "call_2")
        self.assertEqual(messages[4]["content"][0]["content"], "Error: failed")
        self.assertTrue(messages[4]["content"][0]["is_error"])

    # -------------------------------------------------------------------------
    # 6. Final Code Generation
    # -------------------------------------------------------------------------

    @patch("urllib.request.urlopen")
    def test_anthropic_final_response_parsing(self, mock_urlopen):
        """Final text response containing structured JSON changes parses into list[FileOperation]."""
        mock_payload = {
            "id": "msg_04",
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({
                        "changes": [
                            {
                                "operation": "modify",
                                "path": "src/main.py",
                                "patch": "--- a/src/main.py\n+++ b/src/main.py\n@@ -1 +1 @@\n-old\n+new\n",
                                "reason": "Update main function",
                            }
                        ]
                    }),
                }
            ],
            "stop_reason": "end_turn",
        }

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(mock_payload).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        provider = AnthropicProvider(self.config_anthropic)
        result = provider.generate_code_with_tools(
            "Final changes", self.plan, self.context, [self.test_tool]
        )

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].path, "src/main.py")
        self.assertEqual(result[0].action, "modify")
        self.assertIn("+new", result[0].patch or "")

    # -------------------------------------------------------------------------
    # 7. Malformed Inputs and Edge Cases
    # -------------------------------------------------------------------------

    @patch("urllib.request.urlopen")
    def test_anthropic_malformed_json_tool_input(self, mock_urlopen):
        """Malformed JSON string in tool_use input raises ProviderError."""
        mock_payload = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_bad",
                    "name": "read_file_range",
                    "input": "{bad_json",
                }
            ]
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(mock_payload).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        provider = AnthropicProvider(self.config_anthropic)
        with self.assertRaises(ProviderError):
            provider.generate_code_with_tools("task", self.plan, self.context, [self.test_tool])

    @patch("urllib.request.urlopen")
    def test_anthropic_missing_tool_name(self, mock_urlopen):
        """Tool use block missing a name raises ProviderError."""
        mock_payload = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_bad",
                    "input": {},
                }
            ]
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(mock_payload).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        provider = AnthropicProvider(self.config_anthropic)
        with self.assertRaises(ProviderError):
            provider.generate_code_with_tools("task", self.plan, self.context, [self.test_tool])

    @patch("urllib.request.urlopen")
    def test_anthropic_empty_response(self, mock_urlopen):
        """Empty content blocks raise ProviderError."""
        mock_payload = {"content": []}
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(mock_payload).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        provider = AnthropicProvider(self.config_anthropic)
        with self.assertRaises(ProviderError):
            provider.generate_code_with_tools("task", self.plan, self.context, [self.test_tool])

    # -------------------------------------------------------------------------
    # 8. HTTP Error Classification
    # -------------------------------------------------------------------------

    @patch("urllib.request.urlopen")
    def test_anthropic_http_error_handling(self, mock_urlopen):
        """HTTP error codes (401, 429, 404, 400) map to appropriate ProviderError classes."""
        provider = AnthropicProvider(self.config_anthropic)

        # 401 Unauthorized
        err_401 = urllib.error.HTTPError("https://api.anthropic.com/v1/messages", 401, "Unauthorized", {}, io.BytesIO(b'{"error":{"message":"invalid key"}}'))
        mock_urlopen.side_effect = err_401
        with self.assertRaises(AuthenticationError):
            provider.generate_code_with_tools("task", self.plan, self.context, [self.test_tool])

        # 429 Rate Limit
        err_429 = urllib.error.HTTPError("https://api.anthropic.com/v1/messages", 429, "Too Many Requests", {"Retry-After": "10"}, io.BytesIO(b'{"error":{"message":"rate limit exceeded"}}'))
        mock_urlopen.side_effect = err_429
        with self.assertRaises(RateLimitError):
            provider.generate_code_with_tools("task", self.plan, self.context, [self.test_tool])

        # 404 Model Unavailable
        err_404 = urllib.error.HTTPError("https://api.anthropic.com/v1/messages", 404, "Not Found", {}, io.BytesIO(b'{"error":{"message":"model not found"}}'))
        mock_urlopen.side_effect = err_404
        with self.assertRaises(ModelUnavailableError):
            provider.generate_code_with_tools("task", self.plan, self.context, [self.test_tool])

        # 400 Invalid Request
        err_400 = urllib.error.HTTPError("https://api.anthropic.com/v1/messages", 400, "Bad Request", {}, io.BytesIO(b'{"error":{"message":"invalid payload"}}'))
        mock_urlopen.side_effect = err_400
        with self.assertRaises(InvalidRequestError):
            provider.generate_code_with_tools("task", self.plan, self.context, [self.test_tool])

    # -------------------------------------------------------------------------
    # 9. ToolEngine Integration with AnthropicProvider
    # -------------------------------------------------------------------------

    @patch("urllib.request.urlopen")
    def test_anthropic_tool_engine_exploration_loop(self, mock_urlopen):
        """ToolEngine executes exploration loop with AnthropicProvider, recording telemetry."""
        # 1st response: tool call to find_files
        resp1 = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_explore_1",
                    "name": "find_files",
                    "input": {"pattern": "*.py"},
                }
            ],
            "stop_reason": "tool_use",
        }
        # 2nd response: final code operations
        resp2 = {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({
                        "changes": [
                            {
                                "operation": "create",
                                "path": "src/output.py",
                                "content": "print('anthropic tool engine')\n",
                                "reason": "Created via Anthropic ToolEngine loop",
                            }
                        ]
                    }),
                }
            ],
            "stop_reason": "end_turn",
        }

        mock_responses = [
            json.dumps(resp1).encode("utf-8"),
            json.dumps(resp2).encode("utf-8"),
        ]

        def fake_urlopen(req, *args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.read.return_value = mock_responses.pop(0)
            mock_resp.__enter__.return_value = mock_resp
            return mock_resp

        mock_urlopen.side_effect = fake_urlopen

        provider = AnthropicProvider(self.config_anthropic)
        registry = ToolRegistry(Path("."))
        engine = ToolEngine(provider, registry)

        result = engine.run("Explore and create output.py", self.plan, self.context)

        self.assertTrue(result.completed)
        self.assertEqual(len(result.tool_history), 1)
        self.assertEqual(result.metrics.total_calls, 1)
        self.assertEqual(result.file_operations[0].path, "src/output.py")


if __name__ == "__main__":
    unittest.main()
