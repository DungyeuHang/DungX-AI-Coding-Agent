from __future__ import annotations

import io
import json
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch
import urllib.error

from local_agent.config import AgentConfig
from local_agent.models import (
    FileOperation,
    Plan,
    ProjectContext,
    ProviderCapability,
    ProviderError,
    RateLimitError,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from local_agent.providers import (
    AIProvider,
    AntigravityProvider,
    DeepSeekProvider,
    GeminiProvider,
    MockProvider,
    OpenAIProvider,
    _format_gemini_tools,
    _format_openai_tools,
)


class CustomStubProvider(AIProvider):
    """Stub provider implementing only generate_code to test base fallback."""

    def generate_code(self, task: str, plan: Plan, context: ProjectContext, failure=None, review=None) -> list[FileOperation]:
        return [FileOperation(action="modify", path="test.py", content="# stub\n")]


class ProviderToolsTests(unittest.TestCase):
    def setUp(self):
        self.config_openai = AgentConfig(
            project=Path("."),
            provider="openai",
            model="gpt-4o",
            api_key="test-key-openai",
            api_base_url="https://api.openai.com/v1",
        )
        self.config_deepseek = AgentConfig(
            project=Path("."),
            provider="deepseek",
            model="deepseek-coder",
            api_key="test-key-deepseek",
            deepseek_base_url="https://api.deepseek.com/v1",
        )
        self.config_gemini = AgentConfig(
            project=Path("."),
            provider="gemini",
            model="gemini-2.5-flash",
            api_key="test-key-gemini",
            gemini_base_url="https://generativelanguage.googleapis.com/v1beta",
        )
        self.config_antigravity = AgentConfig(
            project=Path("."),
            provider="antigravity",
            model="antigravity-preview-05-2026",
            api_key="test-key-gemini",
            gemini_base_url="https://generativelanguage.googleapis.com/v1beta",
        )
        self.context = ProjectContext(root="/test/project")
        self.plan = Plan(objective="Implement tools support", files_likely_to_change=["test.py"])
        self.test_tool = ToolDefinition(
            name="read_file_range",
            description="Read lines from file",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                },
                "required": ["path"],
            },
        )

    # -------------------------------------------------------------------------
    # A. Base fallback & B. Capability Declaration
    # -------------------------------------------------------------------------

    def test_base_provider_fallback_to_generate_code(self):
        provider = CustomStubProvider()
        result = provider.generate_code_with_tools("task", self.plan, self.context, [self.test_tool])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].path, "test.py")

    def test_mock_provider_fallback_compatibility(self):
        provider = MockProvider()
        result = provider.generate_code_with_tools("task", self.plan, self.context, [self.test_tool])
        self.assertIsInstance(result, list)
        # MockProvider returns empty file ops
        self.assertEqual(len(result), 0)

    def test_capability_advertising(self):
        openai_p = OpenAIProvider(self.config_openai)
        deepseek_p = DeepSeekProvider(self.config_deepseek)
        gemini_p = GeminiProvider(self.config_gemini)
        antigravity_p = AntigravityProvider(self.config_antigravity)
        mock_p = MockProvider()

        self.assertIn(ProviderCapability.TOOL_USE, openai_p.capabilities)
        self.assertIn(ProviderCapability.TOOL_USE, deepseek_p.capabilities)
        self.assertIn(ProviderCapability.TOOL_USE, gemini_p.capabilities)
        self.assertIn(ProviderCapability.TOOL_USE, antigravity_p.capabilities)
        self.assertNotIn(ProviderCapability.TOOL_USE, mock_p.capabilities)

    # -------------------------------------------------------------------------
    # C. OpenAI Serialization & D. OpenAI Parsing
    # -------------------------------------------------------------------------

    def test_format_openai_tools(self):
        formatted = _format_openai_tools([self.test_tool])
        self.assertEqual(len(formatted), 1)
        self.assertEqual(formatted[0]["type"], "function")
        self.assertEqual(formatted[0]["function"]["name"], "read_file_range")
        self.assertEqual(formatted[0]["function"]["parameters"]["type"], "object")

    @patch("urllib.request.urlopen")
    def test_openai_tool_call_response_parsing(self, mock_urlopen):
        mock_response_payload = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_abc123",
                                "type": "function",
                                "function": {
                                    "name": "read_file_range",
                                    "arguments": json.dumps({"path": "src/app.py", "start_line": 1}),
                                },
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 15, "completion_tokens": 8},
        }

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(mock_response_payload).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        provider = OpenAIProvider(self.config_openai)
        result = provider.generate_code_with_tools("Inspect file", self.plan, self.context, [self.test_tool])

        self.assertIsInstance(result, ToolCall)
        self.assertEqual(result.call_id, "call_abc123")
        self.assertEqual(result.tool_name, "read_file_range")
        self.assertEqual(result.arguments, {"path": "src/app.py", "start_line": 1})

    @patch("urllib.request.urlopen")
    def test_openai_final_response_parsing(self, mock_urlopen):
        mock_response_payload = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({
                            "changes": [
                                {
                                    "operation": "modify",
                                    "path": "src/app.py",
                                    "patch": "--- a\n+++ b\n@@ -1 +1 @@\n-old\n+new\n",
                                    "reason": "Update app",
                                }
                            ]
                        }),
                    }
                }
            ],
            "usage": {"prompt_tokens": 25, "completion_tokens": 12},
        }

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(mock_response_payload).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        provider = OpenAIProvider(self.config_openai)
        result = provider.generate_code_with_tools("Final code", self.plan, self.context, [self.test_tool])

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].path, "src/app.py")
        self.assertEqual(result[0].action, "modify")

    @patch("urllib.request.urlopen")
    def test_openai_malformed_arguments_raises_provider_error(self, mock_urlopen):
        mock_response_payload = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_bad",
                                "type": "function",
                                "function": {
                                    "name": "read_file_range",
                                    "arguments": "{not_valid_json",
                                },
                            }
                        ],
                    }
                }
            ]
        }

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(mock_response_payload).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        provider = OpenAIProvider(self.config_openai)
        with self.assertRaises(ProviderError):
            provider.generate_code_with_tools("Bad args", self.plan, self.context, [self.test_tool])

    # -------------------------------------------------------------------------
    # E. DeepSeek Integration
    # -------------------------------------------------------------------------

    @patch("urllib.request.urlopen")
    def test_deepseek_uses_openai_compatible_format(self, mock_urlopen):
        mock_response_payload = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_ds_1",
                                "type": "function",
                                "function": {
                                    "name": "read_file_range",
                                    "arguments": json.dumps({"path": "main.py"}),
                                },
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(mock_response_payload).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        provider = DeepSeekProvider(self.config_deepseek)
        result = provider.generate_code_with_tools("Deepseek tool call", self.plan, self.context, [self.test_tool])

        self.assertIsInstance(result, ToolCall)
        self.assertEqual(result.call_id, "call_ds_1")

        # Verify request URL targeted deepseek_base_url
        req = mock_urlopen.call_args[0][0]
        self.assertTrue(req.full_url.startswith("https://api.deepseek.com/v1"))

    # -------------------------------------------------------------------------
    # F. Gemini Serialization & G. Gemini Parsing
    # -------------------------------------------------------------------------

    def test_format_gemini_tools(self):
        formatted = _format_gemini_tools([self.test_tool])
        self.assertEqual(len(formatted), 1)
        self.assertIn("functionDeclarations", formatted[0])
        self.assertEqual(formatted[0]["functionDeclarations"][0]["name"], "read_file_range")

    @patch("urllib.request.urlopen")
    def test_gemini_function_call_parsing(self, mock_urlopen):
        mock_response_payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "read_file_range",
                                    "args": {"path": "src/gemini.py", "start_line": 10},
                                }
                            }
                        ]
                    }
                }
            ],
            "usageMetadata": {"promptTokenCount": 50, "candidatesTokenCount": 20},
        }

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(mock_response_payload).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        provider = GeminiProvider(self.config_gemini)
        provider._available_models = [{"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]}]

        result = provider.generate_code_with_tools("Gemini call", self.plan, self.context, [self.test_tool])

        self.assertIsInstance(result, ToolCall)
        self.assertEqual(result.tool_name, "read_file_range")
        self.assertEqual(result.arguments, {"path": "src/gemini.py", "start_line": 10})

    @patch("urllib.request.urlopen")
    def test_gemini_final_response_parsing(self, mock_urlopen):
        mock_response_payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps({
                                    "changes": [
                                        {
                                            "operation": "create",
                                            "path": "src/new_gemini.py",
                                            "content": "# gemini\n",
                                            "reason": "Created file",
                                        }
                                    ]
                                })
                            }
                        ]
                    }
                }
            ],
            "usageMetadata": {"promptTokenCount": 60, "candidatesTokenCount": 30},
        }

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(mock_response_payload).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        provider = GeminiProvider(self.config_gemini)
        provider._available_models = [{"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]}]

        result = provider.generate_code_with_tools("Gemini final", self.plan, self.context, [self.test_tool])

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].path, "src/new_gemini.py")
        self.assertEqual(result[0].action, "create")

    # -------------------------------------------------------------------------
    # H. Tool History Serialization in Requests
    # -------------------------------------------------------------------------

    @patch("urllib.request.urlopen")
    def test_openai_tool_history_included_in_request(self, mock_urlopen):
        history = [
            (
                ToolCall("call_h1", "read_file_range", {"path": "a.py"}),
                ToolResult("call_h1", "read_file_range", "line 1\nline 2"),
            )
        ]

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "choices": [{"message": {"role": "assistant", "content": json.dumps({"changes": []})}}]
        }).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        provider = OpenAIProvider(self.config_openai)
        provider.generate_code_with_tools("Task with history", self.plan, self.context, [self.test_tool], tool_history=history)

        req = mock_urlopen.call_args[0][0]
        req_body = json.loads(req.data.decode("utf-8"))

        messages = req_body["messages"]
        # system (0), user (1), assistant tool_calls (2), tool (3)
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertEqual(messages[2]["tool_calls"][0]["id"], "call_h1")
        self.assertEqual(messages[3]["role"], "tool")
        self.assertEqual(messages[3]["tool_call_id"], "call_h1")
        self.assertEqual(messages[3]["content"], "line 1\nline 2")

    @patch("urllib.request.urlopen")
    def test_gemini_tool_history_included_in_request(self, mock_urlopen):
        history = [
            (
                ToolCall("call_gh1", "read_file_range", {"path": "b.py"}),
                ToolResult("call_gh1", "read_file_range", "1: def foo(): pass"),
            )
        ]

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "candidates": [{"content": {"parts": [{"text": json.dumps({"changes": []})}]}}]
        }).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        provider = GeminiProvider(self.config_gemini)
        provider._available_models = [{"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]}]

        provider.generate_code_with_tools("Task with history", self.plan, self.context, [self.test_tool], tool_history=history)

        req = mock_urlopen.call_args[0][0]
        req_body = json.loads(req.data.decode("utf-8"))

        contents = req_body["contents"]
        # user (0), model functionCall (1), user functionResponse (2)
        self.assertEqual(len(contents), 3)
        self.assertEqual(contents[1]["role"], "model")
        self.assertIn("functionCall", contents[1]["parts"][0])
        self.assertEqual(contents[1]["parts"][0]["functionCall"]["name"], "read_file_range")
        self.assertEqual(contents[2]["role"], "user")
        self.assertIn("functionResponse", contents[2]["parts"][0])

    # -------------------------------------------------------------------------
    # I. Token Metric Recording
    # -------------------------------------------------------------------------

    @patch("urllib.request.urlopen")
    def test_openai_token_metric_recorded(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "choices": [{"message": {"role": "assistant", "content": json.dumps({"changes": []})}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        provider = OpenAIProvider(self.config_openai)
        provider.metrics_enabled = True
        provider.generate_code_with_tools("Metric test", self.plan, self.context, [self.test_tool])

        self.assertEqual(len(provider.provider_metrics), 1)
        metric = provider.provider_metrics[0]
        self.assertEqual(metric.actual_input_tokens, 100)
        self.assertEqual(metric.actual_output_tokens, 50)
        self.assertTrue(metric.succeeded)

    # -------------------------------------------------------------------------
    # J. Error Handling & Rate Limiting
    # -------------------------------------------------------------------------

    @patch("urllib.request.urlopen")
    def test_openai_rate_limit_error_handling(self, mock_urlopen):
        http_err = urllib.error.HTTPError(
            url="https://api.openai.com/v1/chat/completions",
            code=429,
            msg="Too Many Requests",
            hdrs={"Retry-After": "30"},
            fp=io.BytesIO(b'{"error": {"message": "Rate limit exceeded"}}'),
        )
        mock_urlopen.side_effect = http_err

        provider = OpenAIProvider(self.config_openai)
        with self.assertRaises(RateLimitError):
            provider.generate_code_with_tools("Rate limit test", self.plan, self.context, [self.test_tool])


if __name__ == "__main__":
    unittest.main()
