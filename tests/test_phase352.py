from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from local_agent.config import AgentConfig
from local_agent.models import ProjectContext, RunReport
from local_agent.providers import AntigravityProvider, ProviderError
from local_agent.ui import AgentUI


class _HTTPResponse:
    status = 200

    def __init__(self, payload: object):
        self.body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.body


class _Value:
    def __init__(self, value: object):
        self.value = value

    def get(self):
        return self.value


class Phase352Tests(unittest.TestCase):
    def _root(self) -> Path:
        return Path(tempfile.mkdtemp())

    def _config(self, root: Path, **overrides: object) -> AgentConfig:
        return AgentConfig.from_environment(root, provider="antigravity", model="test-model", **overrides)

    def test_explicit_runtime_key_works_without_environment(self):
        secret = "runtime-secret"
        with mock.patch.dict(os.environ, {}, clear=True):
            provider = AntigravityProvider(self._config(self._root(), api_key=secret))
            captured = {}

            def fake_urlopen(request, timeout):
                captured["request"] = request
                captured["timeout"] = timeout
                return _HTTPResponse({"ok": True})

            with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
                self.assertEqual(provider._request_json("https://example.test/interactions", b"{}", "interactions.create", "test-model", 1), {"ok": True})

        self.assertEqual(captured["request"].headers["X-goog-api-key"], secret)

    def test_environment_fallback_works_when_runtime_key_is_absent(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "environment-secret"}, clear=True):
            config = self._config(self._root())
            provider = AntigravityProvider(config)
        self.assertEqual(config.api_key, "environment-secret")
        self.assertEqual(provider.api_key, "environment-secret")

    def test_explicit_runtime_key_takes_precedence(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "environment-secret"}, clear=True):
            provider = AntigravityProvider(self._config(self._root(), api_key="runtime-secret"))
        self.assertEqual(provider.api_key, "runtime-secret")

    def test_missing_both_produces_safe_credential_error(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ProviderError) as raised:
                AntigravityProvider(self._config(self._root()))
        self.assertIn("GEMINI_API_KEY is required", str(raised.exception))
        self.assertNotIn("secret", str(raised.exception).lower())

    def test_secret_never_appears_in_errors_metrics_urls_or_config_repr(self):
        secret = "runtime-secret-value"
        with mock.patch.dict(os.environ, {}, clear=True):
            config = self._config(self._root(), api_key=secret, metrics_enabled=True)
            provider = AntigravityProvider(config)
            error = urllib.error.HTTPError(
                "https://example.test/interactions",
                401,
                "Unauthorized",
                {},
                io.BytesIO(json.dumps({"error": {"message": f"invalid {secret}"}}).encode()),
            )
            with mock.patch("urllib.request.urlopen", side_effect=error) as opener:
                with self.assertRaises(ProviderError) as raised:
                    provider._request_json("https://example.test/interactions", b"{}", "interactions.create", "test-model", 1)

        self.assertNotIn(secret, str(raised.exception))
        self.assertNotIn(secret, json.dumps([metric.__dict__ for metric in provider.provider_metrics]))
        self.assertNotIn(secret, opener.call_args.args[0].full_url)
        self.assertNotIn(secret, opener.call_args.args[0].data.decode("utf-8"))
        self.assertNotIn(secret, repr(config))
        self.assertNotIn(secret, repr(RunReport(project=ProjectContext(str(self._root()), metadata={"status": "failed"}))))

    def test_ui_config_passes_runtime_key_without_mutating_environment(self):
        secret = "ui-runtime-secret"
        root = self._root()
        ui = AgentUI.__new__(AgentUI)
        ui.provider_var = _Value("antigravity")
        ui.key_var = _Value(secret)
        ui.model_var = _Value("test-model")
        ui.project_var = _Value(str(root))
        ui.dry_run_var = _Value(False)
        ui.approval_var = _Value(False)
        with mock.patch.dict(os.environ, {}, clear=True):
            config = ui._config()
            self.assertNotIn("GEMINI_API_KEY", os.environ)
        self.assertEqual(config.api_key, secret)
        self.assertEqual(AntigravityProvider(config).api_key, secret)

    def test_connection_and_run_share_ui_runtime_configuration_builder(self):
        source_connection = __import__("inspect").getsource(AgentUI.test_connection)
        source_run = __import__("inspect").getsource(AgentUI.run_agent)
        self.assertIn("self._config()", source_connection)
        self.assertIn("self._config()", source_run)


if __name__ == "__main__":
    unittest.main()
