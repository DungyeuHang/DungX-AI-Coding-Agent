from __future__ import annotations

import io
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from local_agent.analyzer import RepositoryAnalyzer
from local_agent.coding_agent import CodingAgent, UnsafeModificationError
from local_agent.config import AgentConfig
from local_agent.context import ContextSelector
from local_agent.filesystem import ProjectFilesystem, SandboxViolation
from local_agent.models import FailureAnalysis, FileOperation, Plan, ProjectContext, ReviewResult
from local_agent.orchestrator import Orchestrator
from local_agent.providers import AIProvider, AntigravityProvider, GeminiProvider, ProviderError, build_provider


class _HTTPResponse:
    def __init__(self, payload: object):
        import json
        self.body = json.dumps(payload).encode("utf-8")
        self.status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.body


class DeterministicProvider(AIProvider):
    def __init__(self, repair: bool = False):
        self.repair = repair
        self.failure_seen = False

    def generate_plan(self, task: str, context: ProjectContext) -> Plan:
        return Plan(
            objective=task,
            files_likely_to_change=["calculator.py"],
            files_likely_to_create=["test_calculator.py"],
            steps=["Implement multiply", "Run tests"],
            validation_strategy=["python -m unittest discover"],
        )

    def generate_code(self, task: str, plan: Plan, context: ProjectContext, failure: FailureAnalysis | None = None, review: ReviewResult | None = None) -> list[FileOperation]:
        if self.repair and failure is not None:
            patch = """--- a/calculator.py
+++ b/calculator.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a - b
+    return (a + b)
"""
            return [FileOperation("modify", "calculator.py", patch=patch, reason="repair the failing addition")]
        if self.repair:
            patch = """--- a/calculator.py
+++ b/calculator.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a + b
+    return a - b
"""
            return [FileOperation("modify", "calculator.py", patch=patch, reason="intentional first-pass defect")]
        patch = """--- a/calculator.py
+++ b/calculator.py
@@ -1,2 +1,5 @@
 def add(a, b):
     return a + b
+
+def multiply(a, b):
+    return a * b
"""
        test_content = "import unittest\nfrom calculator import add, multiply\n\nclass CalculatorTests(unittest.TestCase):\n    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n\n    def test_multiply(self):\n        self.assertEqual(multiply(2, 3), 6)\n"
        return [
            FileOperation("modify", "calculator.py", patch=patch, reason="add multiplication"),
            FileOperation("create", "test_calculator.py", content=test_content, reason="cover the new function"),
        ]

    def analyze_failure(self, execution, diff, context, plan):
        self.failure_seen = True
        return FailureAnalysis("The addition implementation subtracts values", ["calculator.py"], "restore addition")

    def review_changes(self, task, plan, diff, context):
        return ReviewResult("APPROVED", "The deterministic implementation satisfies the task.", [])


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


class Phase2Tests(unittest.TestCase):
    def test_antigravity_configuration_and_documented_interactions_request(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=True):
            config = AgentConfig.from_environment(directory, provider="antigravity")
            self.assertEqual(config.model, "gemini-3.7-flash")
            provider = build_provider(config)
            self.assertIsInstance(provider, AntigravityProvider)
            response = _HTTPResponse({
                "status": "completed",
                "steps": [{"type": "model_output", "content": [{"type": "text", "text": '{"ok": true}'}]}],
            })
            with mock.patch("urllib.request.urlopen", return_value=response) as opener:
                self.assertEqual(provider._json_call("system", "user"), {"ok": True})
            request = opener.call_args.args[0]
            self.assertEqual(request.full_url, "https://generativelanguage.googleapis.com/v1beta/interactions")
            headers = {key.lower(): value for key, value in request.header_items()}
            self.assertEqual(headers.get("x-goog-api-key"), "test-key")
            self.assertNotIn("key=", request.full_url)
            import json
            request_body = json.loads(request.data.decode("utf-8"))
            self.assertEqual(request_body["agent"], "antigravity-preview-05-2026")
            self.assertEqual(request_body["environment"], "remote")
            self.assertEqual(request_body["agent_config"], {"type": "antigravity", "model": "gemini-3.7-flash"})

    def test_antigravity_response_parsing_and_safe_error(self):
        payload = {"steps": [{"type": "model_output", "content": [{"type": "text", "text": "GEMINI_LIVE_OK"}]}]}
        self.assertEqual(AntigravityProvider._parse_interaction_text(payload), "GEMINI_LIVE_OK")
        with self.assertRaisesRegex(ProviderError, "model text"):
            AntigravityProvider._parse_interaction_text({"status": "completed", "steps": []})
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=True):
            config = AgentConfig.from_environment(directory, provider="antigravity")
            provider = AntigravityProvider(config)
            body = io.BytesIO(b'{"error":{"message":"interaction unavailable"}}')
            error = urllib.error.HTTPError("https://example.test/v1beta/interactions", 404, "Not Found", {}, body)
            with mock.patch("urllib.request.urlopen", side_effect=error):
                with self.assertRaisesRegex(ProviderError, "HTTP 404") as raised:
                    provider.test_connection()
            self.assertIn("interactions.create", str(raised.exception))
            self.assertNotIn("test-key", str(raised.exception))

    def test_patch_modify_create_delete_and_dry_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "calculator.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
            filesystem = ProjectFilesystem(root)
            agent = CodingAgent(filesystem)
            plan = Plan("change", files_likely_to_change=["calculator.py"], files_likely_to_create=["new.py", "patch.py"])
            patch = """--- a/calculator.py
+++ b/calculator.py
@@ -1,2 +1,3 @@
 def add(a, b):
     return a + b
+
"""
            changes = agent.prepare([FileOperation("modify", "calculator.py", patch=patch)], plan)
            self.assertEqual((root / "calculator.py").read_text(encoding="utf-8"), "def add(a, b):\n    return a + b\n")
            agent.apply_prepared(changes)
            self.assertTrue((root / "calculator.py").read_text(encoding="utf-8").endswith("\n\n"))
            agent.apply([FileOperation("create", "new.py", content="value = 1\n")], plan)
            agent.apply([FileOperation("delete", "new.py")], plan)
            create_patch = """--- /dev/null
+++ b/patch.py
@@ -0,0 +1 @@
+value = 2
"""
            agent.apply([FileOperation("create", "patch.py", patch=create_patch)], plan)
            with self.assertRaises(UnsafeModificationError):
                agent.prepare([FileOperation("modify", "calculator.py", patch=create_patch)], plan)
            with self.assertRaises(UnsafeModificationError):
                agent.prepare([FileOperation("modify", "calculator.py", patch="not a patch")], plan)
            with self.assertRaises(SandboxViolation):
                agent.prepare([FileOperation("create", "../escape.py", content="bad")], plan)

    def test_unrelated_existing_change_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "user.py").write_text("user change\n", encoding="utf-8")
            agent = CodingAgent(ProjectFilesystem(root), protected_paths={"user.py"})
            with self.assertRaises(UnsafeModificationError):
                agent.apply([FileOperation("write", "user.py", content="overwrite\n")])

    def test_context_selector_prefers_task_related_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "student_dashboard.py").write_text("class StudentDashboard:\n    enrollment = []\n", encoding="utf-8")
            (root / "database.py").write_text("class Database:\n    pass\n", encoding="utf-8")
            (root / "test_student_dashboard.py").write_text("from student_dashboard import StudentDashboard\n", encoding="utf-8")
            context = RepositoryAnalyzer(root).analyze()
            ContextSelector(root, max_files=2).select("add student enrollment", context)
            selected = context.metadata["selected_files"]
            self.assertIn("student_dashboard.py", selected)
            self.assertNotIn("database.py", selected)

    def test_gemini_configuration_missing_key_and_response_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            config = AgentConfig.from_environment(directory, provider="gemini", model="gemini-test")
            self.assertEqual(config.model, "gemini-test")
            with mock.patch.dict("os.environ", {}, clear=True):
                with self.assertRaises(ProviderError):
                    build_provider(config)
            payload = {"candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}]}
            self.assertEqual(GeminiProvider._parse_response(payload)["ok"], True)
            text_payload = {"candidates": [{"content": {"parts": [{"text": "GEMINI_LIVE_OK"}]}}]}
            self.assertEqual(GeminiProvider._parse_text_response(text_payload), "GEMINI_LIVE_OK")
            with self.assertRaises(ProviderError):
                GeminiProvider._parse_response({})

    def test_gemini_connection_discovers_model_and_uses_header_auth(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=True):
            config = AgentConfig.from_environment(directory, provider="gemini", model="gemini-2.0-flash", gemini_base_url="https://example.test/v1beta")
            provider = GeminiProvider(config)
            responses = [
                _HTTPResponse({"models": [{"name": "models/gemini-3.5-flash", "supportedGenerationMethods": ["generateContent"]}]}),
                _HTTPResponse({"candidates": [{"content": {"parts": [{"text": "GEMINI_LIVE_OK"}]}}]}),
            ]
            with mock.patch("urllib.request.urlopen", side_effect=responses) as opener:
                self.assertEqual(provider.test_connection(), "GEMINI_LIVE_OK")
            discovery_request = opener.call_args_list[0].args[0]
            generation_request = opener.call_args_list[1].args[0]
            self.assertIn("/models?pageSize=1000", discovery_request.full_url)
            self.assertEqual(generation_request.full_url, "https://example.test/v1beta/models/gemini-3.5-flash:generateContent")
            headers = {key.lower(): value for key, value in generation_request.header_items()}
            self.assertEqual(headers.get("x-goog-api-key"), "test-key")
            self.assertNotIn("key=", generation_request.full_url)

    def test_gemini_404_has_safe_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=True):
            config = AgentConfig.from_environment(directory, provider="gemini", gemini_base_url="https://example.test/v1beta")
            provider = GeminiProvider(config)
            http_error = urllib.error.HTTPError("https://example.test", 404, "Not Found", {}, io.BytesIO(b"{\"error\": {\"message\": \"models/gemini-2.5-flash is not found\"}}"))
            with mock.patch("urllib.request.urlopen", side_effect=http_error):
                with self.assertRaises(ProviderError) as context:
                    provider.test_connection()
                self.assertIn("models/gemini-2.5-flash is not found", str(context.exception))

    def test_gemini_malformed_generation_response_is_reported(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=True):
            config = AgentConfig.from_environment(directory, provider="gemini", model="gemini-2.5-flash", gemini_base_url="https://example.test/v1beta")
            provider = GeminiProvider(config)
            with mock.patch("urllib.request.urlopen", side_effect=[_HTTPResponse({"models": [{"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]}]}), _HTTPResponse({})]):
                with self.assertRaisesRegex(ProviderError, "candidate text"):
                    provider.test_connection()

    def test_sandbox_integration_performs_real_deterministic_coding_workflow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "calculator.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
            config = AgentConfig.from_environment(root, max_iterations=2, validation_commands=["python -m unittest discover"])
            provider = DeterministicProvider()
            report = _make_orchestrator(config, provider).run("Add multiply(a, b) and tests")
            self.assertTrue(report.completed)
            self.assertEqual(set(report.changed_files), {"calculator.py", "test_calculator.py"})
            self.assertEqual(report.executions[-1].exit_code, 0)
            self.assertIn("multiply", (root / "calculator.py").read_text(encoding="utf-8"))

    def test_dry_run_and_approval_do_not_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = "def add(a, b):\n    return a + b\n"
            (root / "calculator.py").write_text(original, encoding="utf-8")
            dry_config = AgentConfig.from_environment(root, dry_run=True, validation_commands=["python -m unittest discover"])
            dry_report = _make_orchestrator(dry_config, DeterministicProvider()).run("Add multiply")
            self.assertTrue(dry_report.dry_run)
            self.assertEqual((root / "calculator.py").read_text(encoding="utf-8"), original)
            approval_config = AgentConfig.from_environment(root, approval="always", validation_commands=["python -m unittest discover"])
            approval_report = _make_orchestrator(approval_config, DeterministicProvider()).run("Add multiply", approval_callback=lambda changes: False)
            self.assertTrue(approval_report.approval_required)
            self.assertEqual(approval_report.changed_files, [])
            self.assertEqual((root / "calculator.py").read_text(encoding="utf-8"), original)

    def test_repair_cycle_uses_failure_and_respects_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "calculator.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
            (root / "test_calculator.py").write_text("import unittest\nfrom calculator import add\n\nclass CalculatorTests(unittest.TestCase):\n    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n", encoding="utf-8")
            config = AgentConfig.from_environment(root, max_iterations=2, validation_commands=["python -m unittest discover"])
            provider = DeterministicProvider(repair=True)
            report = _make_orchestrator(config, provider).run("Keep addition correct")
            self.assertTrue(report.completed)
            self.assertEqual(report.iterations, 2)
            self.assertEqual(len(report.failures), 1)
            self.assertTrue(provider.failure_seen)


if __name__ == "__main__":
    unittest.main()
