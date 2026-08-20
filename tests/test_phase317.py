from __future__ import annotations

import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from local_agent.config import AgentConfig
from local_agent.providers import (
    AnthropicProvider,
    AuthenticationError,
    InvalidRequestError,
    ModelUnavailableError,
    ProviderError,
    RateLimitError,
    build_provider,
)


class MockHTTPResponse:
    def __init__(self, data, status_code, headers=None):
        self.data = data
        self.status = status_code
        self.headers = headers or {}

    def read(self):
        return self.data

    def getcode(self):
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class Phase317AnthropicProviderTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.config = AgentConfig.from_environment(
            self.root,
            provider="anthropic",
            model="claude-3-haiku-20240307",
            api_key="test-key",
        )
        # Add assumed new config attribute for testing
        self.config.anthropic_base_url = "https://api.anthropic.com/v1"

    def test_build_provider(self):
        provider = build_provider(self.config)
        self.assertIsInstance(provider, AnthropicProvider)

    def test_init_requires_api_key(self):
        config_no_key = AgentConfig.from_environment(
            self.root, provider="anthropic", model="claude-3-haiku-20240307"
        )
        config_no_key.api_key = None
        with self.assertRaisesRegex(ProviderError, "ANTHROPIC_API_KEY is required"):
            AnthropicProvider(config_no_key)

    @mock.patch("urllib.request.urlopen")
    def test_json_call_success(self, mock_urlopen):
        response_payload = {
            "content": [{"type": "text", "text": '{"status": "ok"}'}],
            "model": "claude-3-haiku-20240307",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }
        mock_urlopen.return_value = MockHTTPResponse(json.dumps(response_payload).encode(), 200)

        provider = AnthropicProvider(self.config)
        result = provider._json_call("system prompt", "user prompt")

        self.assertEqual(result, {"status": "ok"})
        mock_urlopen.assert_called_once()
        request = mock_urlopen.call_args[0][0]
        self.assertEqual(request.full_url, "https://api.anthropic.com/v1/messages")
        self.assertEqual(request.headers["X-api-key"], "test-key")
        self.assertEqual(request.headers["Anthropic-version"], "2023-06-01")
        body = json.loads(request.data)
        self.assertEqual(body["model"], "claude-3-haiku-20240307")
        self.assertEqual(body["system"], "system prompt")
        self.assertEqual(body["messages"][0]["content"], "user prompt")

    @mock.patch("urllib.request.urlopen")
    def test_http_error_mapping(self, mock_urlopen):
        error_map = {
            400: InvalidRequestError,
            401: AuthenticationError,
            403: AuthenticationError,
            404: ModelUnavailableError,
            429: RateLimitError,
            500: ProviderError,
        }
        for status, error_class in error_map.items():
            with self.subTest(status=status):
                error_response = urllib.error.HTTPError("url", status, "msg", {}, io.BytesIO(b'{}'))
                mock_urlopen.side_effect = error_response
                provider = AnthropicProvider(self.config)
                with self.assertRaises(error_class):
                    provider._json_call("system", "user")