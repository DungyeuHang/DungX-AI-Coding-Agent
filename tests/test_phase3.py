from __future__ import annotations

import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from local_agent.config import AgentConfig
from local_agent.context import ContextSelector
from local_agent.models import FailureAnalysis, FileOperation, Plan, ProjectContext, ReviewResult
from local_agent.orchestrator import Orchestrator
from local_agent.providers import (
    AIProvider,
    GeminiProvider,
    QuotaExceededError,
    RateLimitError,
)


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


class _PatchThenQuotaProvider(AIProvider):
    def __init__(self):
        self.calls = 0
        self.repairs: list[FailureAnalysis | None] = []

    def generate_plan(self, task: str, context: ProjectContext) -> Plan:
        return Plan(task, files_likely_to_change=["calculator.py"])

    def generate_code(self, task: str, plan: Plan, context: ProjectContext, failure=None, review=None):
        self.calls += 1
        self.repairs.append(failure)
        if failure is None:
            return [FileOperation(
                "modify",
                "calculator.py",
                patch="--- a/calculator.py\n+++ b/calculator.py\n@@ -99,1 +99,1 @@\n-return a + b\n+return a - b\n",
                reason="intentionally invalid first patch",
            )]
        raise QuotaExceededError("HTTP 429 quota exceeded")

    def analyze_failure(self, execution, diff, context, plan):
        raise AssertionError("validation failure analysis should not run")

    def review_changes(self, task, plan, diff, context):
        return ReviewResult("APPROVED", "ok")


def _make_orchestrator(config, provider):
    import threading
    from local_agent.storage import JsonFileStorage
    storage = JsonFileStorage(config.project / ".agent_data")
    orch = Orchestrator(config, storage, None, threading.Lock(), threading.Lock())
    orig_run = orch.run
    def run_wrapped(*args, **kwargs):
        with mock.patch("local_agent.orchestrator.build_provider", return_value=provider):
            return orig_run(*args, **kwargs)
    orch.run = run_wrapped
    return orch


class Phase3Tests(unittest.TestCase):
    def test_invalid_patch_then_quota_preserves_file_and_stops_repair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = "def add(a, b):\n    return a + b\n"
            target = root / "calculator.py"
            target.write_text(original, encoding="utf-8")
            provider = _PatchThenQuotaProvider()
            config = AgentConfig.from_environment(root, provider_max_retries=0, max_iterations=5)

            report = _make_orchestrator(config, provider).run("Keep addition correct")

            self.assertEqual(target.read_text(encoding="utf-8"), original)
            self.assertEqual(provider.calls, 2)
            self.assertEqual(report.outcome, "QUOTA_EXCEEDED")
            self.assertFalse(report.completed)
            self.assertEqual(report.failures[0].category, "PATCH_VALIDATION")
            self.assertEqual(report.failures[0].details["path"], "calculator.py")
            self.assertIn("hunk location", report.failures[0].details["validation_error"])
            self.assertEqual(report.failures[1].category, "QUOTA_EXCEEDED")
            self.assertEqual(report.changed_files, [])

    def test_repair_payload_contains_only_relevant_patch_failure(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=True):
            config = AgentConfig.from_environment(directory, provider="gemini", repair_context_bytes=600)
            provider = GeminiProvider(config)
            failure = FailureAnalysis(
                "patch failed",
                ["calculator.py"],
                "repair it",
                category="PATCH_VALIDATION",
                details={
                    "path": "calculator.py",
                    "original_file": "x" * 2000,
                    "generated_patch": "y" * 2000,
                    "validation_error": "hunk location is outside the target file",
                },
            )
            payload = provider._failure_payload(failure, "Fix addition", Plan("Fix", files_likely_to_change=["calculator.py"]))
            serialized = json.dumps(payload)
            self.assertLessEqual(len(serialized), 1200)
            self.assertEqual(payload["failure"]["path"], "calculator.py")
            self.assertIn("generated_patch", payload["failure"])
            self.assertNotIn("file_inventory", serialized)
            self.assertNotIn("repository", serialized.lower())

    def test_context_selector_limits_files_and_file_previews_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.py").write_text("a" * 100, encoding="utf-8")
            (root / "b.py").write_text("b" * 100, encoding="utf-8")
            context = ProjectContext(str(root), source_files=["a.py", "b.py"])
            selected = ContextSelector(root, max_files=1, max_chars=30, max_file_chars=10).select("a", context)
            previews = selected.metadata["selected_file_previews"]
            self.assertLessEqual(len(previews), 1)
            self.assertTrue(all(len(value) <= 10 for value in previews.values()))
            self.assertEqual(selected.metadata["context_selection"]["max_file_chars"], 10)

    def test_quota_429_without_retry_hint_is_classified_and_not_retried(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=True):
            config = AgentConfig.from_environment(directory, provider="gemini", provider_max_retries=3, metrics_enabled=True)
            provider = GeminiProvider(config)
            body = io.BytesIO(b'{"error":{"message":"Quota exceeded"}}')
            error = urllib.error.HTTPError("https://example.test/v1beta/interactions", 429, "Too Many Requests", {}, body)
            with mock.patch("urllib.request.urlopen", side_effect=error) as opener:
                with self.assertRaises(QuotaExceededError) as raised:
                    provider._request_json("https://example.test/v1beta/interactions", b"{}", "interactions.create", "test-model", 1)
            self.assertEqual(opener.call_count, 1)
            self.assertEqual(raised.exception.category, "QUOTA_EXCEEDED")
            self.assertEqual(len(provider.provider_metrics), 1)
            self.assertFalse(provider.provider_metrics[0].succeeded)
            self.assertEqual(provider.provider_metrics[0].error_category, "QUOTA_EXCEEDED")

    def test_retry_after_is_respected_once_and_metrics_are_safe(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=True):
            config = AgentConfig.from_environment(directory, provider="gemini", provider_max_retries=1, max_retry_wait_seconds=5, metrics_enabled=True)
            provider = GeminiProvider(config)
            body = io.BytesIO(b'{"error":{"message":"Too many requests"}}')
            error = urllib.error.HTTPError("https://example.test/v1beta/interactions", 429, "Too Many Requests", {"Retry-After": "0"}, body)
            with mock.patch("urllib.request.urlopen", side_effect=[error, _HTTPResponse({"ok": True})]) as opener:
                with mock.patch("time.sleep") as sleeper:
                    result = provider._request_json("https://example.test/v1beta/interactions", b"{}", "interactions.create", "test-model", 1)
            self.assertEqual(result, {"ok": True})
            self.assertEqual(opener.call_count, 2)
            sleeper.assert_called_once_with(0.0)
            self.assertEqual(len(provider.provider_metrics), 2)
            self.assertFalse(provider.provider_metrics[0].succeeded)
            self.assertTrue(provider.provider_metrics[1].succeeded)


if __name__ == "__main__":
    unittest.main()
