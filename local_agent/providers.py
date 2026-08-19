from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, field
from typing import Any

from .config import AgentConfig
from .models import CommandSpec, ExecutionResult, FailureAnalysis, FileOperation, Plan, PlanProposal, ProjectContext, ProviderCapability, ProviderMetric, ReviewResult, TaskPlan


class ProviderError(RuntimeError):
    """Raised when an AI provider cannot complete a structured operation."""

    category = "UNKNOWN_PROVIDER_ERROR"

    def __init__(self, message: str, *, retry_after_seconds: float | None = None):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class AuthenticationError(ProviderError):
    category = "AUTHENTICATION_ERROR"


class RateLimitError(ProviderError):
    category = "RATE_LIMIT"


class QuotaExceededError(RateLimitError):
    category = "QUOTA_EXCEEDED"


class InvalidRequestError(ProviderError):
    category = "INVALID_REQUEST"


class ModelUnavailableError(ProviderError):
    category = "MODEL_UNAVAILABLE"


class NetworkError(ProviderError):
    category = "NETWORK_ERROR"


class UnknownProviderError(ProviderError):
    category = "UNKNOWN_PROVIDER_ERROR"


class AIProvider:
    @property
    def provider_metrics(self) -> list[ProviderMetric]:
        return getattr(self, "_provider_metrics", [])
    capabilities: set[ProviderCapability] = field(default_factory=set, init=False)

    def _record_metric(self, request_type: str, input_size: int, output_size: int, duration_seconds: float, model: str, succeeded: bool, error_category: str = "") -> None:
        if not getattr(self, "metrics_enabled", False):
            return
        self._provider_metrics = getattr(self, "_provider_metrics", [])
        self._provider_metrics.append(ProviderMetric(
            request_type=request_type,
            input_size=input_size,
            output_size=output_size,
            model=model,
            duration_seconds=round(duration_seconds, 6),
            succeeded=succeeded,
            error_category=error_category,
            approximate_input_tokens=math.ceil(input_size / 4),
            approximate_output_tokens=math.ceil(output_size / 4),
        ))

    def _context(self, context: ProjectContext, stage: str = "planning") -> str:
        """Serialize project context for provider prompts.

        Uses the provider's ``config`` when available; falls back to the
        ``AgentConfig`` defaults so config-less providers (e.g. mock/test
        providers) also satisfy the contract.
        """
        config = getattr(self, "config", None)
        if config is None:
            budget = 30000
            max_files = 24
            max_file_bytes = 5000
        else:
            budget = getattr(config, f"{stage}_context_bytes", 30000)
            max_files = config.max_context_files
            max_file_bytes = config.max_context_file_bytes
        return json.dumps(_bounded_context(context, budget, max_files, max_file_bytes), ensure_ascii=False, indent=2)

    def generate_plan(self, task: str, context: ProjectContext) -> Plan:
        raise NotImplementedError

    def generate_code(self, task: str, plan: Plan, context: ProjectContext, failure: FailureAnalysis | None = None, review: ReviewResult | None = None) -> list[FileOperation]:
        raise NotImplementedError

    def analyze_failure(self, execution: ExecutionResult, diff: str, context: ProjectContext, plan: Plan) -> FailureAnalysis:
        raise NotImplementedError

    def review_changes(self, task: str, plan: Plan, diff: str, context: ProjectContext) -> ReviewResult:
        raise NotImplementedError

    def select_diagnostic_command(self, task: str, plan: Plan, context: ProjectContext, primary_failure: ExecutionResult, available_commands: list[CommandSpec]) -> CommandSpec | None:
        """Select a secondary command to run for more diagnostic info."""
        return None # Default implementation does nothing

    def propose_plan_modification(self, task: str, plan: TaskPlan, failure: FailureAnalysis) -> PlanProposal | None:
        """Propose adding or modifying subtasks to fix a planning-level issue."""
        return None # Default implementation does nothing


class MockProvider(AIProvider):
    """Offline provider used for tests and safe workflow dry runs.

    It creates a real structured plan but intentionally never invents source
    changes. This makes the offline limitation visible instead of pretending a
    task was implemented.
    """

    def __init__(self):
        self.capabilities = {
            ProviderCapability.PLANNING, ProviderCapability.IMPLEMENTATION,
            ProviderCapability.REPAIR, ProviderCapability.REVIEW,
        }

    def generate_plan(self, task: str, context: ProjectContext) -> Plan:
        inspect = (context.documentation_files[:2] + context.config_files[:8] + context.test_files[:8] + context.source_files[:8])
        likely = context.source_files[:5]
        return Plan(
            objective=task,
            files_to_inspect=inspect,
            files_likely_to_change=likely,
            files_likely_to_create=[],
            steps=["Inspect the relevant project files", "Implement the smallest change satisfying the task", "Run the detected validation commands", "Review the final diff"],
            validation_strategy=[command.display() for command in context.validation_commands] or ["No project validation command detected"],
            risks=["Offline provider cannot generate source changes; configure an AI provider for autonomous implementation"],
        )

    def generate_code(self, task: str, plan: Plan, context: ProjectContext, failure: FailureAnalysis | None = None, review: ReviewResult | None = None) -> list[FileOperation]:
        return []

    def analyze_failure(self, execution: ExecutionResult, diff: str, context: ProjectContext, plan: Plan) -> FailureAnalysis:
        return FailureAnalysis(
            probable_root_cause=f"Validation command failed with exit code {execution.exit_code}: {execution.command}",
            affected_files=[],
            recommended_fix="Use an AI provider with code-generation capability to analyze and repair the failure.",
        )

    def review_changes(self, task: str, plan: Plan, diff: str, context: ProjectContext) -> ReviewResult:
        if not diff:
            return ReviewResult("CHANGES_REQUIRED", "No implementation diff was produced by the offline provider.", ["Configure a real provider and provide its API key to generate code."])
        return ReviewResult("CHANGES_REQUIRED", "The offline provider cannot verify generated changes.", ["Run a model-backed review before accepting the implementation."])

    def select_diagnostic_command(self, task: str, plan: Plan, context: ProjectContext, primary_failure: ExecutionResult, available_commands: list[CommandSpec]) -> CommandSpec | None:
        # The mock provider makes a simple, deterministic choice if available.
        return available_commands[0] if available_commands else None

    def propose_plan_modification(self, task: str, plan: TaskPlan, failure: FailureAnalysis) -> PlanProposal | None:
        # The mock provider can propose a simple addition for testing.
        return None


class OpenAIProvider(AIProvider):
    def __init__(self, config: AgentConfig):
        self.config = config
        self.capabilities = {
            ProviderCapability.PLANNING, ProviderCapability.IMPLEMENTATION,
            ProviderCapability.REPAIR, ProviderCapability.REVIEW,
        }
        self.metrics_enabled = config.metrics_enabled
        self.api_key = config.api_key
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ProviderError("OPENAI_API_KEY is required when provider=openai")
        self.model = config.model
        self.base_url = config.api_base_url.rstrip("/")

    def _json_call(self, system: str, user: str) -> dict[str, Any]:
        body = json.dumps({
            "model": self.model,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}, method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                raw = response.read()
                payload = json.loads(raw.decode("utf-8"))
            self._record_metric("structured_request", len(body), len(raw), time.perf_counter() - started, self.model, True)
        except urllib.error.HTTPError as exc:
            reason = _http_error_reason(exc, self.api_key)
            error = _classify_http_error(exc.code, "OpenAI", self.model, "chat.completions", reason, _retry_after_seconds(getattr(exc, "headers", None), {}))
            self._record_metric("structured_request", len(body), 0, time.perf_counter() - started, self.model, False, error.category)
            raise error from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            self._record_metric("structured_request", len(body), 0, time.perf_counter() - started, self.model, False, _error_category(exc))
            if isinstance(exc, (urllib.error.URLError, TimeoutError)):
                raise NetworkError(f"OpenAI request failed: {exc}") from exc
            raise UnknownProviderError(f"OpenAI request failed: {exc}") from exc
        try:
            content = payload["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            return self._parse_json(str(content))
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("OpenAI response did not contain a message") from exc

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if len(lines) < 3:
                raise ProviderError("provider returned an incomplete JSON code fence")
            cleaned = "\n".join(lines[1:-1])
        try:
            value = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"provider returned invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ProviderError("provider JSON must be an object")
        return value

    def _context(self, context: ProjectContext, stage: str = "planning") -> str:
        return json.dumps(_bounded_context(context, getattr(self.config, f"{stage}_context_bytes", 30000), self.config.max_context_files, self.config.max_context_file_bytes), ensure_ascii=False, indent=2)

    def _failure_payload(self, failure: FailureAnalysis | None, task: str, plan: Plan) -> dict[str, Any]:
        if failure is None:
            return {}
        if failure.category == "PATCH_VALIDATION":
            return {"task": task, "plan": _repair_plan(plan), "failure": _bounded_repair_failure(failure, self.config.repair_context_bytes)}
        return {"task": task, "plan": _repair_plan(plan), "failure": asdict(failure)}

    def generate_plan(self, task: str, context: ProjectContext) -> Plan:
        data = self._json_call(
            "You are a careful software architect. Return only valid JSON with keys objective, files_to_inspect, files_to_modify or files_likely_to_change, files_to_create or files_likely_to_create, steps, validation_commands or validation_strategy, risks. Never propose changes outside the project root.",
            f"Task:\n{task}\n\nProject context:\n{self._context(context)}",
        )
        return Plan(objective=str(data.get("objective", task)), files_to_inspect=_strings(data.get("files_to_inspect")), files_likely_to_change=_strings(data.get("files_likely_to_change", data.get("files_to_modify"))), files_likely_to_create=_strings(data.get("files_likely_to_create", data.get("files_to_create"))), steps=_strings(data.get("steps")), validation_strategy=_strings(data.get("validation_strategy", data.get("validation_commands"))), risks=_strings(data.get("risks")))

    def generate_code(self, task: str, plan: Plan, context: ProjectContext, failure: FailureAnalysis | None = None, review: ReviewResult | None = None) -> list[FileOperation]:
        data = self._json_call(
            "You are a careful coding agent. Return only valid JSON: {\"changes\":[{\"operation\":\"modify|create|delete\",\"path\":\"relative/path\",\"patch\":\"unified diff for modify/create/delete\",\"content\":\"optional complete content fallback\",\"reason\":\"...\"}]}. Prefer precise unified patches over full-file replacement. Use only relative paths inside the project. Never modify secrets or .git.",
            f"Task:\n{task}\nPlan:\n{json.dumps(asdict(plan), indent=2)}\nContext:\n{self._context(context, 'repair' if failure else 'implementation') if not failure or failure.category != 'PATCH_VALIDATION' else '{}'}\nFailure:\n{json.dumps(self._failure_payload(failure, task, plan), ensure_ascii=False) if failure else 'none'}\nReview:\n{asdict(review) if review else 'none'}",
        )
        operations: list[FileOperation] = []
        for item in data.get("changes", data.get("operations", [])):
            if isinstance(item, dict):
                operations.append(FileOperation(str(item.get("operation", item.get("action", ""))), str(item.get("path", "")), item.get("content"), str(item.get("reason", "")), item.get("patch")))
        return operations

    def analyze_failure(self, execution: ExecutionResult, diff: str, context: ProjectContext, plan: Plan) -> FailureAnalysis:
        data = self._json_call("Return only JSON with probable_root_cause, affected_files, recommended_fix.", f"Failed command: {execution.command}\nExit code: {execution.exit_code}\nstdout:\n{execution.stdout[-8000:]}\nstderr:\n{execution.stderr[-8000:]}\nDiff:\n{diff[-12000:]}\nPlan:\n{json.dumps(asdict(plan))}\nContext:\n{self._context(context, 'repair')}")
        return FailureAnalysis(str(data.get("probable_root_cause", "Unknown failure")), _strings(data.get("affected_files")), str(data.get("recommended_fix", "")))

    def review_changes(self, task: str, plan: Plan, diff: str, context: ProjectContext) -> ReviewResult:
        data = self._json_call("Return only JSON with verdict (APPROVED or CHANGES_REQUIRED), summary, and findings.", f"Task:\n{task}\nPlan:\n{json.dumps(asdict(plan))}\nDiff:\n{diff[-20000:]}\nContext:\n{self._context(context, 'review')}")
        verdict = str(data.get("verdict", "CHANGES_REQUIRED"))
        if verdict not in {"APPROVED", "CHANGES_REQUIRED"}:
            verdict = "CHANGES_REQUIRED"
        return ReviewResult(verdict, str(data.get("summary", "")), _strings(data.get("findings")))

    def select_diagnostic_command(self, task: str, plan: Plan, context: ProjectContext, primary_failure: ExecutionResult, available_commands: list[CommandSpec]) -> CommandSpec | None:
        if not available_commands:
            return None
        command_list = "\n".join([f"- {cmd.name}: {cmd.display()}" for cmd in available_commands])
        data = self._json_call(
            "You are a debugging expert. Return only JSON with a single key 'command_name' whose value is the name of the best command to run for more diagnostic information, or null if none are suitable.",
            f"Task:\n{task}\nPlan:\n{json.dumps(asdict(plan))}\nPrimary Failure:\n{json.dumps(primary_failure.to_dict())}\n\nAvailable diagnostic commands:\n{command_list}\n\nSelect one command name from the list to help diagnose the failure."
        )
        selected_name = data.get("command_name")
        return next((cmd for cmd in available_commands if cmd.name == selected_name), None)

    def propose_plan_modification(self, task: str, plan: TaskPlan, failure: FailureAnalysis) -> PlanProposal | None:
        # For now, the real providers do not implement this.
        return None


GEMINI_STABLE_MODEL_PREFERENCE = (
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
)

ANTIGRAVITY_AGENT = "antigravity-preview-05-2026"


class GeminiProvider(AIProvider):
    """Gemini REST provider using the standard library to avoid a heavy SDK."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.metrics_enabled = config.metrics_enabled
        self.capabilities = {
            ProviderCapability.PLANNING, ProviderCapability.IMPLEMENTATION,
            ProviderCapability.REPAIR, ProviderCapability.REVIEW,
        }
        # Prefer the explicit runtime credential; retain environment fallback
        # for direct CLI/provider construction.
        self.api_key = config.api_key
        if self.api_key is None:
            self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ProviderError(f"GEMINI_API_KEY is required when provider={config.provider}")
        self.provider_label = "Antigravity" if config.provider == "antigravity" else "Gemini"
        self.model = config.model
        self.base_url = config.gemini_base_url.rstrip("/")
        self._available_models: list[dict[str, Any]] | None = None

    def _json_call(self, system: str, user: str) -> dict[str, Any]:
        self._ensure_model()
        body = json.dumps({
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"},
        }).encode("utf-8")
        payload = self._request_json(self._generation_url(), body, "models.generateContent", self.model, 120)
        return self._parse_response(payload)

    def test_connection(self) -> str:
        """Make one minimal non-mutating request for local UI connectivity checks."""
        self._ensure_model()
        body = json.dumps({
            "contents": [{"role": "user", "parts": [{"text": "Reply with exactly: GEMINI_LIVE_OK"}]}],
            "generationConfig": {"temperature": 0},
        }).encode("utf-8")
        payload = self._request_json(self._generation_url(), body, "models.generateContent", self.model, 60)
        return self._parse_text_response(payload).strip()

    def list_models(self) -> list[dict[str, Any]]:
        """List models exposed by this key, including generation capabilities."""
        models: list[dict[str, Any]] = []
        page_token: str | None = None
        while True:
            query = {"pageSize": "1000"}
            if page_token:
                query["pageToken"] = page_token
            url = f"{self.base_url}/models?{urllib.parse.urlencode(query)}"
            payload = self._request_json(url, None, "models.list", None, 60, method="GET")
            page_models = payload.get("models", [])
            if isinstance(page_models, list):
                models.extend(item for item in page_models if isinstance(item, dict))
            page_token = payload.get("nextPageToken")
            if not isinstance(page_token, str) or not page_token:
                break
        self._available_models = models
        return models

    def generation_models(self) -> list[str]:
        models = self._available_models if self._available_models is not None else self.list_models()
        result: list[str] = []
        for item in models:
            methods = item.get("supportedGenerationMethods", [])
            name = item.get("name")
            if isinstance(name, str) and isinstance(methods, list) and "generateContent" in methods:
                result.append(self._model_id(name))
        return result

    def _ensure_model(self) -> None:
        available = self.generation_models()
        if not available:
            raise ProviderError("Gemini model discovery returned no models supporting generateContent")
        current = self._model_id(self.model)
        if current in available:
            self.model = current
            return
        for preferred in GEMINI_STABLE_MODEL_PREFERENCE:
            if preferred in available:
                self.model = preferred
                return
        self.model = available[0]

    def _generation_url(self) -> str:
        return f"{self.base_url}/models/{urllib.parse.quote(self._model_id(self.model), safe='')}:generateContent"

    def _request_json(self, url: str, body: bytes | None, endpoint: str, model: str | None, timeout: int, method: str = "POST") -> dict[str, Any]:
        attempts = 0
        while True:
            request = urllib.request.Request(
                url,
                data=body,
                headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
                method=method,
            )
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    raw = response.read()
                    payload = json.loads(raw.decode("utf-8"))
                self._record_metric(endpoint, len(body or b""), len(raw), time.perf_counter() - started, model or "", True)
            except urllib.error.HTTPError as exc:
                reason, retry_after = self._http_reason_and_retry(exc)
                error = self._classified_http_error(exc.code, model, endpoint, reason, retry_after)
                self._record_metric(endpoint, len(body or b""), 0, time.perf_counter() - started, model or "", False, error.category)
                if isinstance(error, (RateLimitError, QuotaExceededError)) and retry_after is not None and attempts < self.config.provider_max_retries and retry_after <= self.config.max_retry_wait_seconds:
                    attempts += 1
                    time.sleep(max(0.0, retry_after))
                    continue
                raise error from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                self._record_metric(endpoint, len(body or b""), 0, time.perf_counter() - started, model or "", False, NetworkError.category)
                reason = getattr(exc, "reason", type(exc).__name__)
                raise NetworkError(f"{self.provider_label} request failed:\nHTTP <network>\nModel: {model or '<model discovery>'}\nEndpoint: {endpoint}\nReason: {reason}") from exc
            except json.JSONDecodeError as exc:
                self._record_metric(endpoint, len(body or b""), 0, time.perf_counter() - started, model or "", False, UnknownProviderError.category)
                raise UnknownProviderError(f"{self.provider_label} request failed:\nHTTP {getattr(response, 'status', '<unknown>')}\nModel: {model or '<model discovery>'}\nEndpoint: {endpoint}\nReason: malformed JSON response") from exc
            if not isinstance(payload, dict):
                raise UnknownProviderError(f"{self.provider_label} request failed:\nHTTP <unknown>\nModel: {model or '<model discovery>'}\nEndpoint: {endpoint}\nReason: response was not a JSON object")
            return payload

    def _http_reason_and_retry(self, error: urllib.error.HTTPError) -> tuple[str, float | None]:
        try:
            raw = error.read().decode("utf-8", errors="replace")
            payload = json.loads(raw)
            reason = payload.get("error", {}).get("message", raw) if isinstance(payload, dict) else raw
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            payload = {}
            reason = error.reason or "HTTP error"
        retry_after = _retry_after_seconds(getattr(error, "headers", None), payload)
        return str(reason).replace(self.api_key, "[REDACTED]")[:1000], retry_after

    def _classified_http_error(self, status: int, model: str | None, endpoint: str, reason: str, retry_after: float | None) -> ProviderError:
        return _classify_http_error(status, self.provider_label, model, endpoint, reason, retry_after)

    @staticmethod
    def _model_id(name: str) -> str:
        return name.removeprefix("models/")

    @staticmethod
    def _parse_response(payload: object) -> dict[str, Any]:
        return OpenAIProvider._parse_json(GeminiProvider._parse_text_response(payload))

    @staticmethod
    def _parse_text_response(payload: object) -> str:
        try:
            parts = payload["candidates"][0]["content"]["parts"]  # type: ignore[index]
            content = "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict))
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("Gemini response did not contain candidate text") from exc
        if not content:
            raise ProviderError("Gemini response contained no text")
        return content

    def _context(self, context: ProjectContext, stage: str = "planning") -> str:
        return json.dumps(_bounded_context(context, getattr(self.config, f"{stage}_context_bytes", 30000), self.config.max_context_files, self.config.max_context_file_bytes), ensure_ascii=False, indent=2)

    def _failure_payload(self, failure: FailureAnalysis | None, task: str, plan: Plan) -> dict[str, Any]:
        if failure is None:
            return {}
        if failure.category == "PATCH_VALIDATION":
            return {"task": task, "plan": _repair_plan(plan), "failure": _bounded_repair_failure(failure, self.config.repair_context_bytes)}
        return {"task": task, "plan": _repair_plan(plan), "failure": asdict(failure)}

    def generate_plan(self, task: str, context: ProjectContext) -> Plan:
        data = self._json_call(
            "You are a careful software architect. Return only JSON with objective, files_to_inspect, files_to_modify, files_to_create, steps, validation_commands, and risks. Never propose paths outside the project.",
            f"Task:\n{task}\n\nProject context:\n{self._context(context)}",
        )
        return Plan(objective=str(data.get("objective", task)), files_to_inspect=_strings(data.get("files_to_inspect")), files_likely_to_change=_strings(data.get("files_to_modify", data.get("files_likely_to_change"))), files_likely_to_create=_strings(data.get("files_to_create", data.get("files_likely_to_create"))), steps=_strings(data.get("steps")), validation_strategy=_strings(data.get("validation_commands", data.get("validation_strategy"))), risks=_strings(data.get("risks")))

    def generate_code(self, task: str, plan: Plan, context: ProjectContext, failure: FailureAnalysis | None = None, review: ReviewResult | None = None) -> list[FileOperation]:
        data = self._json_call(
            "You are a careful coding agent. Return only JSON with changes, each containing operation modify/create/delete, relative path, a precise unified patch, optional complete content fallback, and reason. Treat all paths as untrusted and never touch secrets or .git.",
            f"Task:\n{task}\nPlan:\n{json.dumps(asdict(plan), indent=2)}\nContext:\n{self._context(context, 'repair' if failure else 'implementation') if not failure or failure.category != 'PATCH_VALIDATION' else '{}'}\nFailure:\n{json.dumps(self._failure_payload(failure, task, plan), ensure_ascii=False) if failure else 'none'}\nReview:\n{asdict(review) if review else 'none'}",
        )
        return _operations(data)

    def analyze_failure(self, execution: ExecutionResult, diff: str, context: ProjectContext, plan: Plan) -> FailureAnalysis:
        data = self._json_call("Return only JSON with probable_root_cause, affected_files, and recommended_fix.", f"Failed command: {execution.command}\nExit code: {execution.exit_code}\nstdout:\n{execution.stdout[-8000:]}\nstderr:\n{execution.stderr[-8000:]}\nDiff:\n{diff[-12000:]}\nPlan:\n{json.dumps(asdict(plan))}\nContext:\n{self._context(context, 'repair')}")
        return FailureAnalysis(str(data.get("probable_root_cause", "Unknown failure")), _strings(data.get("affected_files")), str(data.get("recommended_fix", "")))

    def review_changes(self, task: str, plan: Plan, diff: str, context: ProjectContext) -> ReviewResult:
        data = self._json_call("Return only JSON with verdict APPROVED or CHANGES_REQUIRED, summary, and findings.", f"Task:\n{task}\nPlan:\n{json.dumps(asdict(plan))}\nDiff:\n{diff[-20000:]}\nContext:\n{self._context(context, 'review')}")
        verdict = str(data.get("verdict", "CHANGES_REQUIRED"))
        if verdict not in {"APPROVED", "CHANGES_REQUIRED"}:
            verdict = "CHANGES_REQUIRED"
        return ReviewResult(verdict, str(data.get("summary", "")), _strings(data.get("findings")))

    def select_diagnostic_command(self, task: str, plan: Plan, context: ProjectContext, primary_failure: ExecutionResult, available_commands: list[CommandSpec]) -> CommandSpec | None:
        if not available_commands:
            return None
        command_list = "\n".join([f"- {cmd.name}: {cmd.display()}" for cmd in available_commands])
        data = self._json_call(
            "You are a debugging expert. Return only JSON with a single key 'command_name' whose value is the name of the best command to run for more diagnostic information, or null if none are suitable.",
            f"Task:\n{task}\nPlan:\n{json.dumps(asdict(plan))}\nPrimary Failure:\n{json.dumps(primary_failure.to_dict())}\n\nAvailable diagnostic commands:\n{command_list}\n\nSelect one command name from the list to help diagnose the failure."
        )
        selected_name = data.get("command_name")
        return next((cmd for cmd in available_commands if cmd.name == selected_name), None)

    def propose_plan_modification(self, task: str, plan: TaskPlan, failure: FailureAnalysis) -> PlanProposal | None:
        # For now, the real providers do not implement this.
        return None


class AntigravityProvider(GeminiProvider):
    """Official Antigravity managed-agent adapter over the Interactions API.

    This intentionally remains a separate provider. The documented API runs the
    agent in Google's remote environment; the local orchestrator still owns the
    project context and applies only the validated operations returned here.
    """

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.agent = os.environ.get("ANTIGRAVITY_AGENT", ANTIGRAVITY_AGENT)
        self.capabilities = {
            ProviderCapability.PLANNING, ProviderCapability.IMPLEMENTATION,
            ProviderCapability.REPAIR, ProviderCapability.REVIEW,
        }

    def _json_call(self, system: str, user: str) -> dict[str, Any]:
        payload = self._interaction(system, user)
        return OpenAIProvider._parse_json(self._parse_interaction_text(payload))

    def test_connection(self) -> str:
        payload = self._interaction("Reply with the requested text and nothing else.", "Reply with exactly: GEMINI_LIVE_OK")
        return self._parse_interaction_text(payload).strip()

    def _interaction(self, system: str, user: str) -> dict[str, Any]:
        body = json.dumps({
            "agent": self.agent,
            "input": user,
            "system_instruction": system,
            "environment": "remote",
            "agent_config": {"type": "antigravity", "model": self.model},
        }).encode("utf-8")
        return self._request_json(f"{self.base_url}/interactions", body, "interactions.create", self.model, 300)

    @staticmethod
    def _parse_interaction_text(payload: object) -> str:
        if not isinstance(payload, dict):
            raise ProviderError("Antigravity response was not a JSON object")
        output_text = payload.get("output_text")
        if isinstance(output_text, str) and output_text:
            return output_text
        steps = payload.get("steps", [])
        if isinstance(steps, list):
            text_blocks: list[str] = []
            for step in steps:
                if not isinstance(step, dict) or step.get("type") != "model_output":
                    continue
                content = step.get("content", [])
                if isinstance(content, list):
                    text_blocks.extend(
                        item.get("text", "")
                        for item in content
                        if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str)
                    )
            if text_blocks:
                return "".join(text_blocks)
        status = payload.get("status")
        if status and status != "completed":
            raise ProviderError(f"Antigravity interaction did not complete: {status}")
        raise ProviderError("Antigravity response did not contain model text")


class DeepSeekProvider(OpenAIProvider):
    """Provider for DeepSeek, using the OpenAI-compatible API."""
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.base_url = config.deepseek_base_url.rstrip("/")


def _strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _operations(data: dict[str, Any]) -> list[FileOperation]:
    operations: list[FileOperation] = []
    for item in data.get("changes", data.get("operations", [])):
        if isinstance(item, dict):
            operations.append(FileOperation(str(item.get("operation", item.get("action", ""))), str(item.get("path", "")), item.get("content"), str(item.get("reason", "")), item.get("patch")))
    return operations


def _bounded_text(value: str, limit: int) -> tuple[str, bool]:
    if len(value) <= limit:
        return value, False
    marker = "\n...[truncated for context budget]...\n"
    available = max(0, limit - len(marker))
    head = available * 2 // 3
    tail = available - head
    return value[:head] + marker + (value[-tail:] if tail else ""), True


def _bounded_context(context: ProjectContext, budget: int, max_files: int, max_file_bytes: int) -> dict[str, Any]:
    compact = context.compact()
    metadata = compact.get("metadata", {}) if isinstance(compact.get("metadata"), dict) else {}
    previews = metadata.get("file_previews", {}) if isinstance(metadata.get("file_previews"), dict) else {}
    selected = list(previews)[:max_files]
    payload: dict[str, Any] = {
        "root": compact.get("root", ""),
        "file_inventory": {
            "source_files": list(compact.get("source_files", []))[:max_files],
            "test_files": list(compact.get("test_files", []))[:max_files],
            "config_files": list(compact.get("config_files", []))[:max_files],
            "documentation_files": list(compact.get("documentation_files", []))[:max_files],
        },
        "selected_files": [],
        "file_previews": {},
        "validation_commands": compact.get("validation_commands", []),
        "git_status": compact.get("git_status", ""),
        "metadata": {
            "context_selection": metadata.get("context_selection", {}),
            "change_impact": metadata.get("change_impact", {}),
            "validation_plan": metadata.get("validation_plan", {}),
            "continuation_context": metadata.get("continuation_context", {}), # Added for Phase 3.9
        },
        "truncated_files": [],
    }
    repository_summary = compact.get("repository_map", {})
    if isinstance(repository_summary, dict):
        payload["repository_map"] = {
            "languages": repository_summary.get("languages", []),
            "frameworks": repository_summary.get("frameworks", []),
            "entry_points": repository_summary.get("entry_points", []),
            "relationships": repository_summary.get("relationships", []),
        }
    for relative in selected:
        preview, truncated = _bounded_text(str(previews[relative]), max_file_bytes)
        candidate = dict(payload)
        candidate["selected_files"] = payload["selected_files"] + [relative]
        candidate["file_previews"] = {**payload["file_previews"], relative: preview}
        if truncated:
            candidate["truncated_files"] = payload["truncated_files"] + [relative]
        if len(json.dumps(candidate, ensure_ascii=False)) > budget and payload["selected_files"]:
            break
        payload = candidate
    if len(json.dumps(payload, ensure_ascii=False)) > budget:
        payload["file_previews"] = {}
        payload["selected_files"] = []
        payload["truncated_files"] = selected
    return payload


def _repair_plan(plan: Plan) -> dict[str, Any]:
    return {
        "objective": plan.objective,
        "files_likely_to_change": plan.files_likely_to_change,
        "files_likely_to_create": plan.files_likely_to_create,
        "steps": plan.steps,
        "validation_strategy": plan.validation_strategy,
    }


def _bounded_repair_failure(failure: FailureAnalysis, budget: int) -> dict[str, Any]:
    details = failure.details
    original, original_truncated = _bounded_text(str(details.get("original_file", "")), max(1, budget // 3))
    patch, patch_truncated = _bounded_text(str(details.get("generated_patch", "")), max(1, budget // 3))
    payload = {
        "category": failure.category,
        "path": details.get("path", ""),
        "validation_error": details.get("validation_error", failure.recommended_fix),
        "original_file": original,
        "generated_patch": patch,
        "truncated": {"original_file": original_truncated, "generated_patch": patch_truncated},
    }
    if len(json.dumps(payload, ensure_ascii=False)) > budget:
        payload["original_file"], _ = _bounded_text(payload["original_file"], max(1, budget // 5))
        payload["generated_patch"], _ = _bounded_text(payload["generated_patch"], max(1, budget // 5))
        payload["truncated"]["budget"] = True
    return payload


def _retry_after_seconds(headers: Any, payload: object) -> float | None:
    value = headers.get("Retry-After") if headers is not None and hasattr(headers, "get") else None
    if value is not None:
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            pass

    def find_retry(value: object) -> float | None:
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key).lower() in {"retryafter", "retry_after", "retrydelay"}:
                    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(item))
                    if match:
                        return float(match.group(1))
                found = find_retry(item)
                if found is not None:
                    return found
        elif isinstance(value, list):
            for item in value:
                found = find_retry(item)
                if found is not None:
                    return found
        return None

    return find_retry(payload)


def _http_error_reason(error: urllib.error.HTTPError, secret: str = "") -> str:
    try:
        raw = error.read().decode("utf-8", errors="replace")
        payload = json.loads(raw)
        reason = payload.get("error", {}).get("message", raw) if isinstance(payload, dict) else raw
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        reason = error.reason or "HTTP error"
    return str(reason).replace(secret, "[REDACTED]")[:1000]


def _classify_http_error(status: int, provider_label: str, model: str | None, endpoint: str, reason: str, retry_after: float | None) -> ProviderError:
    message = f"{provider_label} request failed:\nHTTP {status}\nModel: {model or '<model discovery>'}\nEndpoint: {endpoint}\nReason: {reason}"
    if status in {401, 403}:
        return AuthenticationError(message, retry_after_seconds=retry_after)
    if status == 404:
        return ModelUnavailableError(message, retry_after_seconds=retry_after)
    if status == 400:
        return InvalidRequestError(message, retry_after_seconds=retry_after)
    if status == 429:
        lowered = reason.lower()
        error_type = QuotaExceededError if any(token in lowered for token in ("quota", "exceeded", "free_tier", "resource_exhausted")) else RateLimitError
        return error_type(message, retry_after_seconds=retry_after)
    return UnknownProviderError(message, retry_after_seconds=retry_after)


def _error_category(error: BaseException) -> str:
    if isinstance(error, ProviderError):
        return error.category
    if isinstance(error, urllib.error.HTTPError):
        return {401: AuthenticationError.category, 403: AuthenticationError.category, 404: ModelUnavailableError.category, 429: RateLimitError.category}.get(error.code, UnknownProviderError.category)
    if isinstance(error, (urllib.error.URLError, TimeoutError)):
        return NetworkError.category
    return UnknownProviderError.category


def build_provider(config: AgentConfig, api_key: str | None = None) -> AIProvider:
    # If an explicit key is passed, use it. Otherwise, use the one from config.
    if api_key:
        config.api_key = api_key

    if config.provider == "mock":
        return MockProvider()
    if config.provider == "openai":
        return OpenAIProvider(config)
    if config.provider == "gemini":
        return GeminiProvider(config)
    if config.provider == "antigravity":
        return AntigravityProvider(config)
    if config.provider == "deepseek":
        return DeepSeekProvider(config)
    raise ProviderError(f"unsupported provider: {config.provider}")
