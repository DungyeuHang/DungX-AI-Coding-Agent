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
from .models import (
    AuthenticationError,
    CommandSpec,
    ExecutionResult,
    FailureAnalysis,
    FileOperation,
    InvalidRequestError,
    ModelUnavailableError,
    NetworkError,
    Plan,
    PlanProposal,
    ProjectContext,
    ProviderCapability,
    ProviderError,
    ProviderMetric,
    QuotaExceededError,
    RateLimitError,
    ReviewResult,
    TaskPlan,
    ToolCall,
    ToolDefinition,
    ToolResult,
    UnknownProviderError,
)


class AIProvider:
    @property
    def provider_metrics(self) -> list[ProviderMetric]:
        return getattr(self, "_provider_metrics", [])
    @property
    def capabilities(self) -> set[ProviderCapability]:
        return {
            ProviderCapability.PLANNING,
            ProviderCapability.IMPLEMENTATION,
            ProviderCapability.REPAIR,
            ProviderCapability.REVIEW,
        }

    def _record_metric(self, request_type: str, input_size: int, output_size: int, duration_seconds: float, model: str, succeeded: bool, error_category: str = "", actual_input_tokens: int | None = None, actual_output_tokens: int | None = None) -> None:
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
            actual_input_tokens=actual_input_tokens,
            actual_output_tokens=actual_output_tokens,
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

    def generate_plan_with_tools(
        self,
        task: str,
        context: ProjectContext,
        tools: list[ToolDefinition],
        tool_history: list[tuple[ToolCall, ToolResult]] | None = None,
    ) -> ToolCall | Plan:
        """Generate plan using available tools, or fall back to 1-shot generate_plan."""
        return self.generate_plan(task, context)

    def generate_code(self, task: str, plan: Plan, context: ProjectContext, failure: FailureAnalysis | None = None, review: ReviewResult | None = None) -> list[FileOperation]:
        raise NotImplementedError

    def generate_code_with_tools(
        self,
        task: str,
        plan: Plan,
        context: ProjectContext,
        tools: list[ToolDefinition],
        tool_history: list[tuple[ToolCall, ToolResult]] | None = None,
        failure: FailureAnalysis | None = None,
        review: ReviewResult | None = None,
    ) -> ToolCall | list[FileOperation]:
        """Generate code using available tools, or fall back to 1-shot generate_code."""
        return self.generate_code(task, plan, context, failure=failure, review=review)

    def analyze_failure_with_tools(
        self,
        execution: ExecutionResult,
        diff: str,
        context: ProjectContext,
        plan: Plan,
        tools: list[ToolDefinition],
        tool_history: list[tuple[ToolCall, ToolResult]] | None = None,
    ) -> ToolCall | FailureAnalysis:
        """Analyze failure using available tools, or fall back to 1-shot analyze_failure."""
        return self.analyze_failure(execution, diff, context, plan)

    def verify_changes_with_tools(
        self,
        task: str,
        plan: Plan,
        context: ProjectContext,
        diff: str,
        changed_files: list[str],
        tools: list[ToolDefinition],
        tool_history: list[tuple[ToolCall, ToolResult]] | None = None,
    ) -> ToolCall | dict[str, Any]:
        """Verify changes using available tools, or fall back to default verification."""
        return {"verified": True, "notes": "One-shot verification fallback"}

    def review_changes_with_tools(
        self,
        task: str,
        plan: Plan,
        diff: str,
        context: ProjectContext,
        tools: list[ToolDefinition],
        tool_history: list[tuple[ToolCall, ToolResult]] | None = None,
    ) -> ToolCall | ReviewResult:
        """Review changes using available tools, or fall back to 1-shot review_changes."""
        return self.review_changes(task, plan, diff, context)

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


class BaseHTTPProvider(AIProvider):
    """Base class for AI providers that communicate via HTTP/JSON APIs."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.metrics_enabled = config.metrics_enabled
        self.api_key = config.api_key
        self.model = config.model
        # Subclasses must set self.base_url

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

    def _json_call(self, system: str, user: str) -> dict[str, Any]:
        return self._call_api(system, user)

    @staticmethod
    def _extract_token_counts(payload: dict) -> tuple[int | None, int | None]:
        """Extract actual token counts from a provider API response payload.

        Returns ``(input_tokens, output_tokens)`` or ``(None, None)`` when the
        payload does not contain usage information.

        Handles OpenAI (``usage.prompt_tokens`` / ``usage.completion_tokens``),
        Anthropic (``usage.input_tokens`` / ``usage.output_tokens``), and
        Gemini (``usageMetadata.promptTokenCount`` /
        ``usageMetadata.candidatesTokenCount``) response shapes.
        """
        if not isinstance(payload, dict):
            return None, None
        # OpenAI / Anthropic format
        usage = payload.get("usage")
        if isinstance(usage, dict):
            try:
                inp_val = usage.get("prompt_tokens") if "prompt_tokens" in usage else usage.get("input_tokens")
                out_val = usage.get("completion_tokens") if "completion_tokens" in usage else usage.get("output_tokens")
                if inp_val is not None and out_val is not None:
                    return int(inp_val), int(out_val)
            except (KeyError, TypeError, ValueError):
                pass
        # Gemini / Antigravity format
        usage_meta = payload.get("usageMetadata")
        if isinstance(usage_meta, dict):
            try:
                inp = int(usage_meta["promptTokenCount"])
                out = int(usage_meta["candidatesTokenCount"])
                return inp, out
            except (KeyError, TypeError, ValueError):
                pass
        return None, None

    def _request_json_api(self, url: str, body: bytes | None, headers: dict[str, str], endpoint: str, model: str | None, timeout: int, method: str = "POST") -> dict[str, Any]:
        attempts = 0
        while True:
            request = urllib.request.Request(url, data=body, headers=headers, method=method)
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    raw = response.read()
                    payload = json.loads(raw.decode("utf-8"))
                actual_in, actual_out = self._extract_token_counts(payload) if isinstance(payload, dict) else (None, None)
                self._record_metric(endpoint, len(body or b""), len(raw), time.perf_counter() - started, model or "", True, actual_input_tokens=actual_in, actual_output_tokens=actual_out)
                return payload
            except urllib.error.HTTPError as exc:
                reason, retry_after = _http_error_reason(exc, self.api_key), _retry_after_seconds(getattr(exc, "headers", None), {})
                error = _classify_http_error(exc.code, self.__class__.__name__, model, endpoint, reason, retry_after)
                self._record_metric(endpoint, len(body or b""), 0, time.perf_counter() - started, model or "", False, error.category)
                if isinstance(error, (RateLimitError, QuotaExceededError)) and retry_after is not None and attempts < self.config.provider_max_retries and retry_after <= self.config.max_retry_wait_seconds:
                    attempts += 1
                    time.sleep(max(0.0, retry_after))
                    continue
                raise error from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                self._record_metric(endpoint, len(body or b""), 0, time.perf_counter() - started, model or "", False, NetworkError.category)
                raise NetworkError(f"{self.__class__.__name__} request failed: {exc}") from exc
            except json.JSONDecodeError as exc:
                self._record_metric(endpoint, len(body or b""), 0, time.perf_counter() - started, model or "", False, UnknownProviderError.category)
                raise UnknownProviderError(f"{self.__class__.__name__} request failed: malformed JSON response: {exc}") from exc
            if not isinstance(payload, dict):
                raise UnknownProviderError(f"{self.__class__.__name__} request failed: response was not a JSON object")
            return payload

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
        return json.dumps(_bounded_context(context, budget, max_files, max_file_bytes), ensure_ascii=False, indent=2, default=str)

    def _failure_payload(self, failure: FailureAnalysis | None, task: str, plan: Plan) -> dict[str, Any]:
        if failure is None:
            return {}
        if failure.category == "PATCH_VALIDATION":
            return {"task": task, "plan": _repair_plan(plan), "failure": _bounded_repair_failure(failure, self.config.repair_context_bytes)}
        return {"task": task, "plan": _repair_plan(plan), "failure": asdict(failure)}





class MockProvider(AIProvider):
    """Offline provider used for tests and safe workflow dry runs.

    It creates a real structured plan but intentionally never invents source
    changes. This makes the offline limitation visible instead of pretending a
    task was implemented.
    """

    # Deterministic token values used in tests.
    _MOCK_INPUT_TOKENS = 10
    _MOCK_OUTPUT_TOKENS = 5

    @property
    def capabilities(self) -> set[ProviderCapability]:
        return {
            ProviderCapability.PLANNING, ProviderCapability.IMPLEMENTATION,
            ProviderCapability.REPAIR, ProviderCapability.REVIEW,
        }

    def generate_plan(self, task: str | Task, context: ProjectContext) -> Plan:
        task_text = task.objective if hasattr(task, "objective") else str(task)
        inspect = (context.documentation_files[:2] + context.config_files[:8] + context.test_files[:8] + context.source_files[:8])
        likely = context.source_files[:5]
        self._record_metric("generate_plan", len(task_text), 100, 0.0, "mock", True,
                            actual_input_tokens=self._MOCK_INPUT_TOKENS,
                            actual_output_tokens=self._MOCK_OUTPUT_TOKENS)
        return Plan(
            objective=task_text,
            files_to_inspect=inspect,
            files_likely_to_change=likely,
            files_likely_to_create=[],
            steps=["Inspect the relevant project files", "Implement the smallest change satisfying the task", "Run the detected validation commands", "Review the final diff"],
            validation_strategy=[command.display() for command in context.validation_commands] or ["No project validation command detected"],
            risks=["Offline provider cannot generate source changes; configure an AI provider for autonomous implementation"],
        )

    def generate_code(self, task: str | Task, plan: Plan, context: ProjectContext, failure: FailureAnalysis | None = None, review: ReviewResult | None = None) -> list[FileOperation]:
        task_text = task.objective if hasattr(task, "objective") else str(task)
        self._record_metric("generate_code", len(task_text), 50, 0.0, "mock", True,
                            actual_input_tokens=self._MOCK_INPUT_TOKENS,
                            actual_output_tokens=self._MOCK_OUTPUT_TOKENS)
        return []

    def analyze_failure(self, execution: ExecutionResult, diff: str, context: ProjectContext, plan: Plan) -> FailureAnalysis:
        self._record_metric("analyze_failure", len(execution.stderr), 80, 0.0, "mock", True,
                            actual_input_tokens=self._MOCK_INPUT_TOKENS,
                            actual_output_tokens=self._MOCK_OUTPUT_TOKENS)
        return FailureAnalysis(
            probable_root_cause=f"Validation command failed with exit code {execution.exit_code}: {execution.command}",
            affected_files=[],
            recommended_fix="Use an AI provider with code-generation capability to analyze and repair the failure.",
        )

    def review_changes(self, task: str, plan: Plan, diff: str, context: ProjectContext) -> ReviewResult:
        self._record_metric("review_changes", len(diff), 60, 0.0, "mock", True,
                            actual_input_tokens=self._MOCK_INPUT_TOKENS,
                            actual_output_tokens=self._MOCK_OUTPUT_TOKENS)
        if not diff:
            return ReviewResult("CHANGES_REQUIRED", "No implementation diff was produced by the offline provider.", ["Configure a real provider and provide its API key to generate code."])
        return ReviewResult("CHANGES_REQUIRED", "The offline provider cannot verify generated changes.", ["Run a model-backed review before accepting the implementation."])

    def select_diagnostic_command(self, task: str, plan: Plan, context: ProjectContext, primary_failure: ExecutionResult, available_commands: list[CommandSpec]) -> CommandSpec | None:
        # The mock provider makes a simple, deterministic choice if available.
        return available_commands[0] if available_commands else None

    def propose_plan_modification(self, task: str, plan: TaskPlan, failure: FailureAnalysis) -> PlanProposal | None:
        # The mock provider can propose a simple addition for testing.
        return None


class OpenAIProvider(BaseHTTPProvider):
    """OpenAI chat-completions provider."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        # api_key: prefer config value, then fall back to environment variable.
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ProviderError("OPENAI_API_KEY is required when provider=openai")
        self.model = config.model
        self.base_url = config.api_base_url.rstrip("/")

    @property
    def capabilities(self) -> set[ProviderCapability]:
        return {
            ProviderCapability.PLANNING, ProviderCapability.IMPLEMENTATION,
            ProviderCapability.REPAIR, ProviderCapability.REVIEW,
            ProviderCapability.TOOL_USE,
        }


    def _call_api(self, system: str, user: str) -> dict[str, Any]:
        body = json.dumps({
            "model": self.model,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }).encode("utf-8")
        payload = self._request_json_api(f"{self.base_url}/chat/completions", body, {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}, "chat.completions", self.model, 120)
        try:
            content = payload["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            return self._parse_json(str(content))
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("OpenAI response did not contain a message") from exc

    def generate_plan(self, task: str, context: ProjectContext) -> Plan:
        data = self._json_call(
            "You are a careful software architect. Return only valid JSON with keys objective, files_to_inspect, files_to_modify or files_likely_to_change, files_to_create or files_likely_to_create, steps, validation_commands or validation_strategy, risks. Never propose changes outside the project root.",
            f"Task:\n{task}\n\nProject context:\n{self._context(context)}",
        )
        return Plan(objective=str(data.get("objective", task)), files_to_inspect=_strings(data.get("files_to_inspect")), files_likely_to_change=_strings(data.get("files_likely_to_change", data.get("files_to_modify"))), files_likely_to_create=_strings(data.get("files_likely_to_create", data.get("files_to_create"))), steps=_strings(data.get("steps")), validation_strategy=_strings(data.get("validation_strategy", data.get("validation_commands"))), risks=_strings(data.get("risks")))

    def generate_plan_with_tools(
        self,
        task: str,
        context: ProjectContext,
        tools: list[ToolDefinition],
        tool_history: list[tuple[ToolCall, ToolResult]] | None = None,
    ) -> ToolCall | Plan:
        system_msg = (
            "You are an expert software architect. Inspect existing modules, interfaces, symbols, tests, configuration, and dependencies using available tools. "
            "Explore the repository to understand architecture and conventions before generating the plan.\n"
            "When planning is complete, return only valid JSON with keys: objective, files_to_inspect, files_to_modify or files_likely_to_change, files_to_create or files_likely_to_create, steps, validation_commands or validation_strategy, risks. Never propose changes outside the project root."
        )
        user_msg = f"Task:\n{task}\n\nProject context:\n{self._context(context, 'planning')}"
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        if tool_history:
            for call, result in tool_history:
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call.call_id,
                            "type": "function",
                            "function": {
                                "name": call.tool_name,
                                "arguments": json.dumps(call.arguments),
                            },
                        }
                    ],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.call_id,
                    "content": result.output,
                })

        req_body: dict[str, Any] = {
            "model": self.model,
            "temperature": 0.1,
            "messages": messages,
        }
        if tools:
            req_body["tools"] = _format_openai_tools(tools)
            req_body["tool_choice"] = "auto"

        body = json.dumps(req_body).encode("utf-8")
        payload = self._request_json_api(
            f"{self.base_url}/chat/completions",
            body,
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            "chat.completions",
            self.model,
            120,
        )

        try:
            choice_msg = payload["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("OpenAI response did not contain a message choice") from exc

        tool_calls_raw = choice_msg.get("tool_calls")
        if tool_calls_raw and isinstance(tool_calls_raw, list) and len(tool_calls_raw) > 0:
            first_call = tool_calls_raw[0]
            fn = first_call.get("function", {})
            call_id = first_call.get("id") or f"call_{int(time.time() * 1000)}"
            fn_name = fn.get("name", "")
            raw_args = fn.get("arguments", "{}")
            if not fn_name:
                raise ProviderError("OpenAI tool call missing function name")
            try:
                parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError as exc:
                raise ProviderError(f"OpenAI tool call '{fn_name}' returned malformed JSON arguments: {raw_args}") from exc
            return ToolCall(call_id=str(call_id), tool_name=str(fn_name), arguments=parsed_args if isinstance(parsed_args, dict) else {})

        content = choice_msg.get("content", "")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        data = self._parse_json(str(content))
        return Plan(
            objective=str(data.get("objective", task)),
            files_to_inspect=_strings(data.get("files_to_inspect")),
            files_likely_to_change=_strings(data.get("files_likely_to_change", data.get("files_to_modify"))),
            files_likely_to_create=_strings(data.get("files_likely_to_create", data.get("files_to_create"))),
            steps=_strings(data.get("steps")),
            validation_strategy=_strings(data.get("validation_strategy", data.get("validation_commands"))),
            risks=_strings(data.get("risks")),
        )

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

    def generate_code_with_tools(
        self,
        task: str,
        plan: Plan,
        context: ProjectContext,
        tools: list[ToolDefinition],
        tool_history: list[tuple[ToolCall, ToolResult]] | None = None,
        failure: FailureAnalysis | None = None,
        review: ReviewResult | None = None,
    ) -> ToolCall | list[FileOperation]:
        system_msg = (
            "You are a careful coding agent. You can use available tools to inspect the codebase before making changes. "
            "When you are ready to apply changes, return only JSON with changes: "
            '{"changes":[{"operation":"modify|create|delete","path":"relative/path","patch":"unified diff","content":"optional fallback","reason":"..."}]}. '
            "Prefer precise unified patches over full-file replacement. Use only relative paths inside the project. Never modify secrets or .git."
        )
        user_msg = (
            f"Task:\n{task}\nPlan:\n{json.dumps(asdict(plan), indent=2)}\n"
            f"Context:\n{self._context(context, 'repair' if failure else 'implementation') if not failure or failure.category != 'PATCH_VALIDATION' else '{}'}\n"
            f"Failure:\n{json.dumps(self._failure_payload(failure, task, plan), ensure_ascii=False) if failure else 'none'}\n"
            f"Review:\n{asdict(review) if review else 'none'}"
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        if tool_history:
            for call, result in tool_history:
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call.call_id,
                            "type": "function",
                            "function": {
                                "name": call.tool_name,
                                "arguments": json.dumps(call.arguments),
                            },
                        }
                    ],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.call_id,
                    "content": result.output,
                })

        req_body: dict[str, Any] = {
            "model": self.model,
            "temperature": 0.1,
            "messages": messages,
        }
        if tools:
            req_body["tools"] = _format_openai_tools(tools)
            req_body["tool_choice"] = "auto"

        body = json.dumps(req_body).encode("utf-8")
        payload = self._request_json_api(
            f"{self.base_url}/chat/completions",
            body,
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            "chat.completions",
            self.model,
            120,
        )

        try:
            choice_msg = payload["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("OpenAI response did not contain a message choice") from exc

        tool_calls_raw = choice_msg.get("tool_calls")
        if tool_calls_raw and isinstance(tool_calls_raw, list) and len(tool_calls_raw) > 0:
            first_call = tool_calls_raw[0]
            fn = first_call.get("function", {})
            call_id = first_call.get("id") or f"call_{int(time.time() * 1000)}"
            fn_name = fn.get("name", "")
            raw_args = fn.get("arguments", "{}")
            if not fn_name:
                raise ProviderError("OpenAI tool call missing function name")
            try:
                parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError as exc:
                raise ProviderError(f"OpenAI tool call '{fn_name}' returned malformed JSON arguments: {raw_args}") from exc
            return ToolCall(call_id=str(call_id), tool_name=str(fn_name), arguments=parsed_args if isinstance(parsed_args, dict) else {})

        content = choice_msg.get("content", "")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        data = self._parse_json(str(content))
        return _operations(data)

    def analyze_failure_with_tools(
        self,
        execution: ExecutionResult,
        diff: str,
        context: ProjectContext,
        plan: Plan,
        tools: list[ToolDefinition],
        tool_history: list[tuple[ToolCall, ToolResult]] | None = None,
    ) -> ToolCall | FailureAnalysis:
        system_msg = (
            "You are a debugging expert. Inspect the codebase using available tools to determine the root cause of the failure. "
            "When you have diagnosed the issue, return only JSON: "
            '{"probable_root_cause":"...","affected_files":["..."],"recommended_fix":"..."}.'
        )
        user_msg = (
            f"Failed command: {execution.command}\n"
            f"Exit code: {execution.exit_code}\n"
            f"stdout:\n{execution.stdout[-8000:]}\n"
            f"stderr:\n{execution.stderr[-8000:]}\n"
            f"Diff:\n{diff[-12000:]}\n"
            f"Plan:\n{json.dumps(asdict(plan))}\n"
            f"Context:\n{self._context(context, 'repair')}"
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        if tool_history:
            for call, result in tool_history:
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call.call_id,
                            "type": "function",
                            "function": {
                                "name": call.tool_name,
                                "arguments": json.dumps(call.arguments),
                            },
                        }
                    ],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.call_id,
                    "content": result.output,
                })

        req_body: dict[str, Any] = {
            "model": self.model,
            "temperature": 0.1,
            "messages": messages,
        }
        if tools:
            req_body["tools"] = _format_openai_tools(tools)
            req_body["tool_choice"] = "auto"

        body = json.dumps(req_body).encode("utf-8")
        payload = self._request_json_api(
            f"{self.base_url}/chat/completions",
            body,
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            "chat.completions",
            self.model,
            120,
        )

        try:
            choice_msg = payload["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("OpenAI response did not contain a message choice") from exc

        tool_calls_raw = choice_msg.get("tool_calls")
        if tool_calls_raw and isinstance(tool_calls_raw, list) and len(tool_calls_raw) > 0:
            first_call = tool_calls_raw[0]
            fn = first_call.get("function", {})
            call_id = first_call.get("id") or f"call_{int(time.time() * 1000)}"
            fn_name = fn.get("name", "")
            raw_args = fn.get("arguments", "{}")
            if not fn_name:
                raise ProviderError("OpenAI tool call missing function name")
            try:
                parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError as exc:
                raise ProviderError(f"OpenAI tool call '{fn_name}' returned malformed JSON arguments: {raw_args}") from exc
            return ToolCall(call_id=str(call_id), tool_name=str(fn_name), arguments=parsed_args if isinstance(parsed_args, dict) else {})

        content = choice_msg.get("content", "")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        data = self._parse_json(str(content))
        return FailureAnalysis(
            probable_root_cause=str(data.get("probable_root_cause", "Unknown failure")),
            affected_files=_strings(data.get("affected_files")),
            recommended_fix=str(data.get("recommended_fix", "")),
        )

    def verify_changes_with_tools(
        self,
        task: str,
        plan: Plan,
        context: ProjectContext,
        diff: str,
        changed_files: list[str],
        tools: list[ToolDefinition],
        tool_history: list[tuple[ToolCall, ToolResult]] | None = None,
    ) -> ToolCall | dict[str, Any]:
        system_msg = (
            "You are a verification expert. Inspect modified files and related test assertions using available tools. "
            "When verification is complete, return only JSON: {\"verified\": true/false, \"notes\": \"...\", \"targeted_commands\": [\"...\"]}."
        )
        user_msg = (
            f"Task: {task}\n"
            f"Changed files: {json.dumps(changed_files)}\n"
            f"Diff:\n{diff[-12000:]}\n"
            f"Plan:\n{json.dumps(asdict(plan))}\n"
            f"Context:\n{self._context(context, 'repair')}"
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        if tool_history:
            for call, result in tool_history:
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call.call_id,
                            "type": "function",
                            "function": {
                                "name": call.tool_name,
                                "arguments": json.dumps(call.arguments),
                            },
                        }
                    ],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.call_id,
                    "content": result.output,
                })

        req_body: dict[str, Any] = {
            "model": self.model,
            "temperature": 0.1,
            "messages": messages,
        }
        if tools:
            req_body["tools"] = _format_openai_tools(tools)
            req_body["tool_choice"] = "auto"

        body = json.dumps(req_body).encode("utf-8")
        payload = self._request_json_api(
            f"{self.base_url}/chat/completions",
            body,
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            "chat.completions",
            self.model,
            120,
        )

        try:
            choice_msg = payload["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("OpenAI response did not contain a message choice") from exc

        tool_calls_raw = choice_msg.get("tool_calls")
        if tool_calls_raw and isinstance(tool_calls_raw, list) and len(tool_calls_raw) > 0:
            first_call = tool_calls_raw[0]
            fn = first_call.get("function", {})
            call_id = first_call.get("id") or f"call_{int(time.time() * 1000)}"
            fn_name = fn.get("name", "")
            raw_args = fn.get("arguments", "{}")
            if not fn_name:
                raise ProviderError("OpenAI tool call missing function name")
            try:
                parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError as exc:
                raise ProviderError(f"OpenAI tool call '{fn_name}' returned malformed JSON arguments: {raw_args}") from exc
            return ToolCall(call_id=str(call_id), tool_name=str(fn_name), arguments=parsed_args if isinstance(parsed_args, dict) else {})

        content = choice_msg.get("content", "")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return self._parse_json(str(content))

    def analyze_failure(self, execution: ExecutionResult, diff: str, context: ProjectContext, plan: Plan) -> FailureAnalysis:
        data = self._json_call("Return only JSON with probable_root_cause, affected_files, recommended_fix.", f"Failed command: {execution.command}\nExit code: {execution.exit_code}\nstdout:\n{execution.stdout[-8000:]}\nstderr:\n{execution.stderr[-8000:]}\nDiff:\n{diff[-12000:]}\nPlan:\n{json.dumps(asdict(plan))}\nContext:\n{self._context(context, 'repair')}")
        return FailureAnalysis(str(data.get("probable_root_cause", "Unknown failure")), _strings(data.get("affected_files")), str(data.get("recommended_fix", "")))

    def review_changes_with_tools(
        self,
        task: str,
        plan: Plan,
        diff: str,
        context: ProjectContext,
        tools: list[ToolDefinition],
        tool_history: list[tuple[ToolCall, ToolResult]] | None = None,
    ) -> ToolCall | ReviewResult:
        system_msg = (
            "You are an expert code reviewer. Inspect modified files, related callers, callees, symbol references, configuration, and tests using available tools. "
            "Verify that changes are correct, complete, safe, adhere to the plan, and introduce no regressions.\n"
            "When review is complete, return only JSON: {\"verdict\": \"APPROVED\" or \"CHANGES_REQUIRED\", \"summary\": \"...\", \"findings\": [\"...\"]}."
        )
        user_msg = (
            f"Task:\n{task}\n"
            f"Plan:\n{json.dumps(asdict(plan))}\n"
            f"Diff:\n{diff[-20000:]}\n"
            f"Context:\n{self._context(context, 'review')}"
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        if tool_history:
            for call, result in tool_history:
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call.call_id,
                            "type": "function",
                            "function": {
                                "name": call.tool_name,
                                "arguments": json.dumps(call.arguments),
                            },
                        }
                    ],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.call_id,
                    "content": result.output,
                })

        req_body: dict[str, Any] = {
            "model": self.model,
            "temperature": 0.1,
            "messages": messages,
        }
        if tools:
            req_body["tools"] = _format_openai_tools(tools)
            req_body["tool_choice"] = "auto"

        body = json.dumps(req_body).encode("utf-8")
        payload = self._request_json_api(
            f"{self.base_url}/chat/completions",
            body,
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            "chat.completions",
            self.model,
            120,
        )

        try:
            choice_msg = payload["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("OpenAI response did not contain a message choice") from exc

        tool_calls_raw = choice_msg.get("tool_calls")
        if tool_calls_raw and isinstance(tool_calls_raw, list) and len(tool_calls_raw) > 0:
            first_call = tool_calls_raw[0]
            fn = first_call.get("function", {})
            call_id = first_call.get("id") or f"call_{int(time.time() * 1000)}"
            fn_name = fn.get("name", "")
            raw_args = fn.get("arguments", "{}")
            if not fn_name:
                raise ProviderError("OpenAI tool call missing function name")
            try:
                parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError as exc:
                raise ProviderError(f"OpenAI tool call '{fn_name}' returned malformed JSON arguments: {raw_args}") from exc
            return ToolCall(call_id=str(call_id), tool_name=str(fn_name), arguments=parsed_args if isinstance(parsed_args, dict) else {})

        content = choice_msg.get("content", "")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        data = self._parse_json(str(content))
        verdict = str(data.get("verdict", "CHANGES_REQUIRED"))
        if verdict not in {"APPROVED", "CHANGES_REQUIRED", "CHANGES_REQUESTED"}:
            verdict = "CHANGES_REQUIRED"
        return ReviewResult(verdict, str(data.get("summary", "")), _strings(data.get("findings")))

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


class GeminiProvider(BaseHTTPProvider):
    """Gemini REST provider using the standard library to avoid a heavy SDK."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.api_key = config.api_key
        if self.api_key is None:
            self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ProviderError(f"GEMINI_API_KEY is required when provider={config.provider}")
        self.provider_label = "Antigravity" if config.provider == "antigravity" else "Gemini"
        self.model = config.model
        self.base_url = config.gemini_base_url.rstrip("/")
        self._available_models: list[dict[str, Any]] | None = None

    @property
    def capabilities(self) -> set[ProviderCapability]:
        return {
            ProviderCapability.PLANNING, ProviderCapability.IMPLEMENTATION,
            ProviderCapability.REPAIR, ProviderCapability.REVIEW,
            ProviderCapability.TOOL_USE,
        }

    def _call_api(self, system: str, user: str) -> dict[str, Any]:
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
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        payload = self._request_json_api(url, body, headers, endpoint, model, timeout, method)
        if not isinstance(payload, dict):
            raise UnknownProviderError(f"{self.provider_label} request failed: response was not a JSON object")
        return payload

    def _json_call(self, system: str, user: str) -> dict[str, Any]:
        # This is a wrapper for the Gemini-specific _call_api
        return self._call_api(system, user)

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

    def generate_plan(self, task: str, context: ProjectContext) -> Plan:
        data = self._json_call(
            "You are a careful software architect. Return only JSON with objective, files_to_inspect, files_to_modify, files_to_create, steps, validation_commands, and risks. Never propose paths outside the project.",
            f"Task:\n{task}\n\nProject context:\n{self._context(context)}",
        )
        return Plan(objective=str(data.get("objective", task)), files_to_inspect=_strings(data.get("files_to_inspect")), files_likely_to_change=_strings(data.get("files_to_modify", data.get("files_likely_to_change"))), files_likely_to_create=_strings(data.get("files_to_create", data.get("files_likely_to_create"))), steps=_strings(data.get("steps")), validation_strategy=_strings(data.get("validation_commands", data.get("validation_strategy"))), risks=_strings(data.get("risks")))

    def generate_plan_with_tools(
        self,
        task: str,
        context: ProjectContext,
        tools: list[ToolDefinition],
        tool_history: list[tuple[ToolCall, ToolResult]] | None = None,
    ) -> ToolCall | Plan:
        self._ensure_model()
        system_msg = (
            "You are a careful software architect. Inspect existing modules, interfaces, symbols, tests, configuration, and dependencies using available tools. "
            "Explore the repository to understand architecture and conventions before generating the plan.\n"
            "When planning is complete, return only JSON with objective, files_to_inspect, files_to_modify, files_to_create, steps, validation_commands, and risks. Never propose paths outside the project."
        )
        user_content = f"Task:\n{task}\n\nProject context:\n{self._context(context, 'planning')}"
        contents: list[dict[str, Any]] = [
            {"role": "user", "parts": [{"text": user_content}]},
        ]
        if tool_history:
            for call, result in tool_history:
                contents.append({
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": call.tool_name,
                                "args": call.arguments,
                            }
                        }
                    ],
                })
                contents.append({
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": call.tool_name,
                                "response": {"output": result.output},
                            }
                        }
                    ],
                })

        body_dict: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system_msg}]},
            "contents": contents,
            "generationConfig": {"temperature": 0.1},
        }
        if tools:
            body_dict["tools"] = _format_gemini_tools(tools)
        body = json.dumps(body_dict).encode("utf-8")
        payload = self._request_json(self._generation_url(), body, "models.generateContent", self.model, 120)

        try:
            parts = payload["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("Gemini response did not contain candidate content") from exc

        for part in parts:
            if isinstance(part, dict) and "functionCall" in part:
                fc = part["functionCall"]
                fn_name = fc.get("name", "")
                fn_args = fc.get("args", {})
                if not fn_name:
                    raise ProviderError("Gemini functionCall missing name")
                if not isinstance(fn_args, dict):
                    raise ProviderError("Gemini functionCall args must be a dict")
                call_id = f"call_{int(time.time() * 1000)}"
                return ToolCall(call_id=call_id, tool_name=str(fn_name), arguments=fn_args)

        content = "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict))
        if not content:
            raise ProviderError("Gemini response contained neither functionCall nor text")
        data = OpenAIProvider._parse_json(content)
        return Plan(
            objective=str(data.get("objective", task)),
            files_to_inspect=_strings(data.get("files_to_inspect")),
            files_likely_to_change=_strings(data.get("files_to_modify", data.get("files_likely_to_change"))),
            files_likely_to_create=_strings(data.get("files_to_create", data.get("files_likely_to_create"))),
            steps=_strings(data.get("steps")),
            validation_strategy=_strings(data.get("validation_commands", data.get("validation_strategy"))),
            risks=_strings(data.get("risks")),
        )

    def generate_code(self, task: str, plan: Plan, context: ProjectContext, failure: FailureAnalysis | None = None, review: ReviewResult | None = None) -> list[FileOperation]:
        data = self._json_call(
            "You are a careful coding agent. Return only JSON with changes, each containing operation modify/create/delete, relative path, a precise unified patch, optional complete content fallback, and reason. Treat all paths as untrusted and never touch secrets or .git.",
            f"Task:\n{task}\nPlan:\n{json.dumps(asdict(plan), indent=2)}\nContext:\n{self._context(context, 'repair' if failure else 'implementation') if not failure or failure.category != 'PATCH_VALIDATION' else '{}'}\nFailure:\n{json.dumps(self._failure_payload(failure, task, plan), ensure_ascii=False) if failure else 'none'}\nReview:\n{asdict(review) if review else 'none'}",
        )
        return _operations(data)

    def generate_code_with_tools(
        self,
        task: str,
        plan: Plan,
        context: ProjectContext,
        tools: list[ToolDefinition],
        tool_history: list[tuple[ToolCall, ToolResult]] | None = None,
        failure: FailureAnalysis | None = None,
        review: ReviewResult | None = None,
    ) -> ToolCall | list[FileOperation]:
        self._ensure_model()
        system_msg = (
            "You are a careful coding agent. You can use available tools to inspect the codebase before making changes. "
            "When you are ready to apply changes, return only JSON with changes, each containing operation modify/create/delete, relative path, a precise unified patch, optional complete content fallback, and reason. Treat all paths as untrusted and never touch secrets or .git."
        )
        user_content = (
            f"Task:\n{task}\nPlan:\n{json.dumps(asdict(plan), indent=2)}\n"
            f"Context:\n{self._context(context, 'repair' if failure else 'implementation') if not failure or failure.category != 'PATCH_VALIDATION' else '{}'}\n"
            f"Failure:\n{json.dumps(self._failure_payload(failure, task, plan), ensure_ascii=False) if failure else 'none'}\n"
            f"Review:\n{asdict(review) if review else 'none'}"
        )
        contents: list[dict[str, Any]] = [
            {"role": "user", "parts": [{"text": user_content}]},
        ]
        if tool_history:
            for call, result in tool_history:
                contents.append({
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": call.tool_name,
                                "args": call.arguments,
                            }
                        }
                    ],
                })
                contents.append({
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": call.tool_name,
                                "response": {"output": result.output},
                            }
                        }
                    ],
                })

        body_dict: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system_msg}]},
            "contents": contents,
            "generationConfig": {"temperature": 0.1},
        }
        if tools:
            body_dict["tools"] = _format_gemini_tools(tools)
        body = json.dumps(body_dict).encode("utf-8")
        payload = self._request_json(self._generation_url(), body, "models.generateContent", self.model, 120)

        try:
            parts = payload["candidates"][0]["content"]["parts"]  # type: ignore[index]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("Gemini response did not contain candidate content") from exc

        for part in parts:
            if isinstance(part, dict) and "functionCall" in part:
                fc = part["functionCall"]
                fn_name = fc.get("name", "")
                fn_args = fc.get("args", {})
                if not fn_name:
                    raise ProviderError("Gemini functionCall missing name")
                if not isinstance(fn_args, dict):
                    raise ProviderError("Gemini functionCall args must be a dict")
                call_id = f"call_{int(time.time() * 1000)}"
                return ToolCall(call_id=call_id, tool_name=str(fn_name), arguments=fn_args)

        content = "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict))
        if not content:
            raise ProviderError("Gemini response contained neither functionCall nor text")
        data = OpenAIProvider._parse_json(content)
        return _operations(data)

    def analyze_failure_with_tools(
        self,
        execution: ExecutionResult,
        diff: str,
        context: ProjectContext,
        plan: Plan,
        tools: list[ToolDefinition],
        tool_history: list[tuple[ToolCall, ToolResult]] | None = None,
    ) -> ToolCall | FailureAnalysis:
        self._ensure_model()
        system_msg = (
            "You are a debugging expert. Inspect the codebase using available tools to determine the root cause of the failure. "
            "When you have diagnosed the issue, return only JSON: "
            '{"probable_root_cause":"...","affected_files":["..."],"recommended_fix":"..."}.'
        )
        user_content = (
            f"Failed command: {execution.command}\n"
            f"Exit code: {execution.exit_code}\n"
            f"stdout:\n{execution.stdout[-8000:]}\n"
            f"stderr:\n{execution.stderr[-8000:]}\n"
            f"Diff:\n{diff[-12000:]}\n"
            f"Plan:\n{json.dumps(asdict(plan))}\n"
            f"Context:\n{self._context(context, 'repair')}"
        )
        contents: list[dict[str, Any]] = [
            {"role": "user", "parts": [{"text": user_content}]},
        ]
        if tool_history:
            for call, result in tool_history:
                contents.append({
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": call.tool_name,
                                "args": call.arguments,
                            }
                        }
                    ],
                })
                contents.append({
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": call.tool_name,
                                "response": {"output": result.output},
                            }
                        }
                    ],
                })

        body_dict: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system_msg}]},
            "contents": contents,
            "generationConfig": {"temperature": 0.1},
        }
        if tools:
            body_dict["tools"] = _format_gemini_tools(tools)
        body = json.dumps(body_dict).encode("utf-8")
        payload = self._request_json(self._generation_url(), body, "models.generateContent", self.model, 120)

        try:
            parts = payload["candidates"][0]["content"]["parts"]  # type: ignore[index]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("Gemini response did not contain candidate content") from exc

        for part in parts:
            if isinstance(part, dict) and "functionCall" in part:
                fc = part["functionCall"]
                fn_name = fc.get("name", "")
                fn_args = fc.get("args", {})
                if not fn_name:
                    raise ProviderError("Gemini functionCall missing name")
                if not isinstance(fn_args, dict):
                    raise ProviderError("Gemini functionCall args must be a dict")
                call_id = f"call_{int(time.time() * 1000)}"
                return ToolCall(call_id=call_id, tool_name=str(fn_name), arguments=fn_args)

        content = "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict))
        if not content:
            raise ProviderError("Gemini response contained neither functionCall nor text")
        data = OpenAIProvider._parse_json(content)
        return FailureAnalysis(
            probable_root_cause=str(data.get("probable_root_cause", "Unknown failure")),
            affected_files=_strings(data.get("affected_files")),
            recommended_fix=str(data.get("recommended_fix", "")),
        )

    def verify_changes_with_tools(
        self,
        task: str,
        plan: Plan,
        context: ProjectContext,
        diff: str,
        changed_files: list[str],
        tools: list[ToolDefinition],
        tool_history: list[tuple[ToolCall, ToolResult]] | None = None,
    ) -> ToolCall | dict[str, Any]:
        self._ensure_model()
        system_msg = (
            "You are a verification expert. Inspect modified files and related test assertions using available tools. "
            "When verification is complete, return only JSON: {\"verified\": true/false, \"notes\": \"...\", \"targeted_commands\": [\"...\"]}."
        )
        user_content = (
            f"Task: {task}\n"
            f"Changed files: {json.dumps(changed_files)}\n"
            f"Diff:\n{diff[-12000:]}\n"
            f"Plan:\n{json.dumps(asdict(plan))}\n"
            f"Context:\n{self._context(context, 'repair')}"
        )
        contents: list[dict[str, Any]] = [
            {"role": "user", "parts": [{"text": user_content}]},
        ]
        if tool_history:
            for call, result in tool_history:
                contents.append({
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": call.tool_name,
                                "args": call.arguments,
                            }
                        }
                    ],
                })
                contents.append({
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": call.tool_name,
                                "response": {"output": result.output},
                            }
                        }
                    ],
                })

        body_dict: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system_msg}]},
            "contents": contents,
            "generationConfig": {"temperature": 0.1},
        }
        if tools:
            body_dict["tools"] = _format_gemini_tools(tools)
        body = json.dumps(body_dict).encode("utf-8")
        payload = self._request_json(self._generation_url(), body, "models.generateContent", self.model, 120)

        try:
            parts = payload["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("Gemini response did not contain candidate content") from exc

        for part in parts:
            if isinstance(part, dict) and "functionCall" in part:
                fc = part["functionCall"]
                fn_name = fc.get("name", "")
                fn_args = fc.get("args", {})
                if not fn_name:
                    raise ProviderError("Gemini functionCall missing name")
                if not isinstance(fn_args, dict):
                    raise ProviderError("Gemini functionCall args must be a dict")
                call_id = f"call_{int(time.time() * 1000)}"
                return ToolCall(call_id=call_id, tool_name=str(fn_name), arguments=fn_args)

        content = "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict))
        if not content:
            raise ProviderError("Gemini response contained neither functionCall nor text")
        return OpenAIProvider._parse_json(content)

    def analyze_failure(self, execution: ExecutionResult, diff: str, context: ProjectContext, plan: Plan) -> FailureAnalysis:
        data = self._json_call("Return only JSON with probable_root_cause, affected_files, and recommended_fix.", f"Failed command: {execution.command}\nExit code: {execution.exit_code}\nstdout:\n{execution.stdout[-8000:]}\nstderr:\n{execution.stderr[-8000:]}\nDiff:\n{diff[-12000:]}\nPlan:\n{json.dumps(asdict(plan))}\nContext:\n{self._context(context, 'repair')}")
        return FailureAnalysis(str(data.get("probable_root_cause", "Unknown failure")), _strings(data.get("affected_files")), str(data.get("recommended_fix", "")))

    def review_changes_with_tools(
        self,
        task: str,
        plan: Plan,
        diff: str,
        context: ProjectContext,
        tools: list[ToolDefinition],
        tool_history: list[tuple[ToolCall, ToolResult]] | None = None,
    ) -> ToolCall | ReviewResult:
        self._ensure_model()
        system_msg = (
            "You are an expert code reviewer. Inspect modified files, related callers, callees, symbol references, configuration, and tests using available tools. "
            "Verify that changes are correct, complete, safe, adhere to the plan, and introduce no regressions.\n"
            "When review is complete, return only JSON: {\"verdict\": \"APPROVED\" or \"CHANGES_REQUIRED\", \"summary\": \"...\", \"findings\": [\"...\"]}."
        )
        user_content = (
            f"Task:\n{task}\n"
            f"Plan:\n{json.dumps(asdict(plan))}\n"
            f"Diff:\n{diff[-20000:]}\n"
            f"Context:\n{self._context(context, 'review')}"
        )
        contents: list[dict[str, Any]] = [
            {"role": "user", "parts": [{"text": user_content}]},
        ]
        if tool_history:
            for call, result in tool_history:
                contents.append({
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": call.tool_name,
                                "args": call.arguments,
                            }
                        }
                    ],
                })
                contents.append({
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": call.tool_name,
                                "response": {"output": result.output},
                            }
                        }
                    ],
                })

        body_dict: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system_msg}]},
            "contents": contents,
            "generationConfig": {"temperature": 0.1},
        }
        if tools:
            body_dict["tools"] = _format_gemini_tools(tools)
        body = json.dumps(body_dict).encode("utf-8")
        payload = self._request_json(self._generation_url(), body, "models.generateContent", self.model, 120)

        try:
            parts = payload["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("Gemini response did not contain candidate content") from exc

        for part in parts:
            if isinstance(part, dict) and "functionCall" in part:
                fc = part["functionCall"]
                fn_name = fc.get("name", "")
                fn_args = fc.get("args", {})
                if not fn_name:
                    raise ProviderError("Gemini functionCall missing name")
                if not isinstance(fn_args, dict):
                    raise ProviderError("Gemini functionCall args must be a dict")
                call_id = f"call_{int(time.time() * 1000)}"
                return ToolCall(call_id=call_id, tool_name=str(fn_name), arguments=fn_args)

        content = "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict))
        if not content:
            raise ProviderError("Gemini response contained neither functionCall nor text")
        data = OpenAIProvider._parse_json(content)
        verdict = str(data.get("verdict", "CHANGES_REQUIRED"))
        if verdict not in {"APPROVED", "CHANGES_REQUIRED", "CHANGES_REQUESTED"}:
            verdict = "CHANGES_REQUIRED"
        return ReviewResult(verdict, str(data.get("summary", "")), _strings(data.get("findings")))

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

    @property
    def capabilities(self) -> set[ProviderCapability]:
        return {
            ProviderCapability.PLANNING, ProviderCapability.IMPLEMENTATION,
            ProviderCapability.REPAIR, ProviderCapability.REVIEW,
            ProviderCapability.TOOL_USE,
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


class AnthropicProvider(BaseHTTPProvider):
    """Anthropic Claude provider."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        if self.api_key is None:
            self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ProviderError("ANTHROPIC_API_KEY is required when provider=anthropic")
        self.model = config.model
        anthropic_base = getattr(config, "anthropic_base_url", "https://api.anthropic.com/v1")
        self.base_url = str(anthropic_base).rstrip("/")

    @property
    def capabilities(self) -> set[ProviderCapability]:
        return {
            ProviderCapability.PLANNING,
            ProviderCapability.IMPLEMENTATION,
            ProviderCapability.REPAIR,
            ProviderCapability.REVIEW,
            ProviderCapability.TOOL_USE,
        }

    def _call_api(self, system: str, user: str) -> dict[str, Any]:
        body = json.dumps({
            "model": self.model,
            "max_tokens": 8192,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }).encode("utf-8")
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = self._request_json_api(
            f"{self.base_url}/messages",
            body,
            headers,
            "messages",
            self.model,
            120,
        )
        try:
            content = payload["content"][0]["text"]
            return self._parse_json(str(content))
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("Anthropic response did not contain message text") from exc

    def _json_call(self, system: str, user: str) -> dict[str, Any]:
        return self._call_api(system, user)

    def generate_plan(self, task: str, context: ProjectContext) -> Plan:
        data = self._json_call(
            "You are a careful software architect. Return only valid JSON with keys objective, files_to_inspect, files_to_modify or files_likely_to_change, files_to_create or files_likely_to_create, steps, validation_commands or validation_strategy, risks. Never propose changes outside the project root.",
            f"Task:\n{task}\n\nProject context:\n{self._context(context)}",
        )
        return Plan(
            objective=str(data.get("objective", task)),
            files_to_inspect=_strings(data.get("files_to_inspect")),
            files_likely_to_change=_strings(data.get("files_likely_to_change", data.get("files_to_modify"))),
            files_likely_to_create=_strings(data.get("files_likely_to_create", data.get("files_to_create"))),
            steps=_strings(data.get("steps")),
            validation_strategy=_strings(data.get("validation_strategy", data.get("validation_commands"))),
            risks=_strings(data.get("risks")),
        )

    def generate_plan_with_tools(
        self,
        task: str,
        context: ProjectContext,
        tools: list[ToolDefinition],
        tool_history: list[tuple[ToolCall, ToolResult]] | None = None,
    ) -> ToolCall | Plan:
        system_msg = (
            "You are a careful software architect. Inspect existing modules, interfaces, symbols, tests, configuration, and dependencies using available tools. "
            "Explore the repository to understand architecture and conventions before generating the plan.\n"
            "When planning is complete, return only valid JSON with keys: objective, files_to_inspect, files_to_modify or files_likely_to_change, files_to_create or files_likely_to_create, steps, validation_commands or validation_strategy, risks. Never propose changes outside the project root."
        )
        user_content = f"Task:\n{task}\n\nProject context:\n{self._context(context, 'planning')}"
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_content},
        ]
        if tool_history:
            for call, result in tool_history:
                messages.append({
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": call.call_id,
                            "name": call.tool_name,
                            "input": call.arguments,
                        }
                    ],
                })
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": call.call_id,
                            "content": result.output,
                            "is_error": result.is_error,
                        }
                    ],
                })

        req_body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 8192,
            "temperature": 0.1,
            "system": system_msg,
            "messages": messages,
        }
        if tools:
            req_body["tools"] = _format_anthropic_tools(tools)

        body = json.dumps(req_body).encode("utf-8")
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = self._request_json_api(
            f"{self.base_url}/messages",
            body,
            headers,
            "messages",
            self.model,
            120,
        )

        try:
            content_blocks = payload.get("content", [])
            if not isinstance(content_blocks, list):
                raise ProviderError("Anthropic response content must be a list")
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("Anthropic response did not contain content blocks") from exc

        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                call_id = block.get("id") or f"anthropic_{int(time.time() * 1000)}"
                tool_name = block.get("name", "")
                arguments = block.get("input", {})
                if not tool_name:
                    raise ProviderError("Anthropic tool_use missing name")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError as exc:
                        raise ProviderError(f"Anthropic tool '{tool_name}' returned malformed JSON input: {arguments}") from exc
                return ToolCall(call_id=str(call_id), tool_name=str(tool_name), arguments=arguments if isinstance(arguments, dict) else {})

        text_parts = [
            str(block.get("text", ""))
            for block in content_blocks
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        combined_text = "".join(text_parts).strip()
        if not combined_text:
            raise ProviderError("Anthropic response contained neither tool_use nor text content")

        data = self._parse_json(combined_text)
        return Plan(
            objective=str(data.get("objective", task)),
            files_to_inspect=_strings(data.get("files_to_inspect")),
            files_likely_to_change=_strings(data.get("files_likely_to_change", data.get("files_to_modify"))),
            files_likely_to_create=_strings(data.get("files_likely_to_create", data.get("files_to_create"))),
            steps=_strings(data.get("steps")),
            validation_strategy=_strings(data.get("validation_strategy", data.get("validation_commands"))),
            risks=_strings(data.get("risks")),
        )

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

    def generate_code_with_tools(
        self,
        task: str,
        plan: Plan,
        context: ProjectContext,
        tools: list[ToolDefinition],
        tool_history: list[tuple[ToolCall, ToolResult]] | None = None,
        failure: FailureAnalysis | None = None,
        review: ReviewResult | None = None,
    ) -> ToolCall | list[FileOperation]:
        system_msg = (
            "You are a careful coding agent. You can use available tools to inspect the codebase before making changes. "
            "When you are ready to apply changes, return only JSON with changes: "
            '{"changes":[{"operation":"modify|create|delete","path":"relative/path","patch":"unified diff","content":"optional fallback","reason":"..."}]}. '
            "Prefer precise unified patches over full-file replacement. Use only relative paths inside the project. Never modify secrets or .git."
        )
        user_content = (
            f"Task:\n{task}\nPlan:\n{json.dumps(asdict(plan), indent=2)}\n"
            f"Context:\n{self._context(context, 'repair' if failure else 'implementation') if not failure or failure.category != 'PATCH_VALIDATION' else '{}'}\n"
            f"Failure:\n{json.dumps(self._failure_payload(failure, task, plan), ensure_ascii=False) if failure else 'none'}\n"
            f"Review:\n{asdict(review) if review else 'none'}"
        )
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_content},
        ]
        if tool_history:
            for call, result in tool_history:
                messages.append({
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": call.call_id,
                            "name": call.tool_name,
                            "input": call.arguments,
                        }
                    ],
                })
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": call.call_id,
                            "content": result.output,
                            "is_error": result.is_error,
                        }
                    ],
                })

        req_body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 8192,
            "temperature": 0.1,
            "system": system_msg,
            "messages": messages,
        }
        if tools:
            req_body["tools"] = _format_anthropic_tools(tools)

        body = json.dumps(req_body).encode("utf-8")
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = self._request_json_api(
            f"{self.base_url}/messages",
            body,
            headers,
            "messages",
            self.model,
            120,
        )

        try:
            content_blocks = payload.get("content", [])
            if not isinstance(content_blocks, list):
                raise ProviderError("Anthropic response content must be a list")
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("Anthropic response did not contain content blocks") from exc

        # 1. Check for tool_use content blocks
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                call_id = block.get("id") or f"call_{int(time.time() * 1000)}"
                fn_name = block.get("name", "")
                raw_input = block.get("input", {})
                if not fn_name:
                    raise ProviderError("Anthropic tool_use block missing name")
                if isinstance(raw_input, str):
                    try:
                        raw_input = json.loads(raw_input)
                    except json.JSONDecodeError as exc:
                        raise ProviderError(f"Anthropic tool '{fn_name}' returned malformed JSON input: {raw_input}") from exc
                if not isinstance(raw_input, dict):
                    raise ProviderError(f"Anthropic tool '{fn_name}' input must be a dictionary, got {type(raw_input).__name__}")
                return ToolCall(call_id=str(call_id), tool_name=str(fn_name), arguments=raw_input)

        # 2. Extract text blocks for final code operations
        text_parts = [
            str(block.get("text", ""))
            for block in content_blocks
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        combined_text = "".join(text_parts).strip()
        if not combined_text:
            raise ProviderError("Anthropic response contained neither tool_use nor text content")

        data = self._parse_json(combined_text)
        return _operations(data)

    def analyze_failure_with_tools(
        self,
        execution: ExecutionResult,
        diff: str,
        context: ProjectContext,
        plan: Plan,
        tools: list[ToolDefinition],
        tool_history: list[tuple[ToolCall, ToolResult]] | None = None,
    ) -> ToolCall | FailureAnalysis:
        system_msg = (
            "You are a debugging expert. Inspect the codebase using available tools to determine the root cause of the failure. "
            "When you have diagnosed the issue, return only JSON: "
            '{"probable_root_cause":"...","affected_files":["..."],"recommended_fix":"..."}.'
        )
        user_content = (
            f"Failed command: {execution.command}\n"
            f"Exit code: {execution.exit_code}\n"
            f"stdout:\n{execution.stdout[-8000:]}\n"
            f"stderr:\n{execution.stderr[-8000:]}\n"
            f"Diff:\n{diff[-12000:]}\n"
            f"Plan:\n{json.dumps(asdict(plan))}\n"
            f"Context:\n{self._context(context, 'repair')}"
        )
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_content},
        ]
        if tool_history:
            for call, result in tool_history:
                messages.append({
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": call.call_id,
                            "name": call.tool_name,
                            "input": call.arguments,
                        }
                    ],
                })
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": call.call_id,
                            "content": result.output,
                            "is_error": result.is_error,
                        }
                    ],
                })

        req_body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 8192,
            "temperature": 0.1,
            "system": system_msg,
            "messages": messages,
        }
        if tools:
            req_body["tools"] = _format_anthropic_tools(tools)

        body = json.dumps(req_body).encode("utf-8")
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = self._request_json_api(
            f"{self.base_url}/messages",
            body,
            headers,
            "messages",
            self.model,
            120,
        )

        try:
            content_blocks = payload.get("content", [])
            if not isinstance(content_blocks, list):
                raise ProviderError("Anthropic response content must be a list")
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("Anthropic response did not contain content blocks") from exc

        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                call_id = block.get("id") or f"call_{int(time.time() * 1000)}"
                fn_name = block.get("name", "")
                raw_input = block.get("input", {})
                if not fn_name:
                    raise ProviderError("Anthropic tool_use block missing name")
                if isinstance(raw_input, str):
                    try:
                        raw_input = json.loads(raw_input)
                    except json.JSONDecodeError as exc:
                        raise ProviderError(f"Anthropic tool '{fn_name}' returned malformed JSON input: {raw_input}") from exc
                if not isinstance(raw_input, dict):
                    raise ProviderError(f"Anthropic tool '{fn_name}' input must be a dictionary, got {type(raw_input).__name__}")
                return ToolCall(call_id=str(call_id), tool_name=str(fn_name), arguments=raw_input)

        text_parts = [
            str(block.get("text", ""))
            for block in content_blocks
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        combined_text = "".join(text_parts).strip()
        if not combined_text:
            raise ProviderError("Anthropic response contained neither tool_use nor text content")

        data = self._parse_json(combined_text)
        return FailureAnalysis(
            probable_root_cause=str(data.get("probable_root_cause", "Unknown failure")),
            affected_files=_strings(data.get("affected_files")),
            recommended_fix=str(data.get("recommended_fix", "")),
        )

    def verify_changes_with_tools(
        self,
        task: str,
        plan: Plan,
        context: ProjectContext,
        diff: str,
        changed_files: list[str],
        tools: list[ToolDefinition],
        tool_history: list[tuple[ToolCall, ToolResult]] | None = None,
    ) -> ToolCall | dict[str, Any]:
        system_msg = (
            "You are a verification expert. Inspect modified files and related test assertions using available tools. "
            "When verification is complete, return only JSON: {\"verified\": true/false, \"notes\": \"...\", \"targeted_commands\": [\"...\"]}."
        )
        user_content = (
            f"Task: {task}\n"
            f"Changed files: {json.dumps(changed_files)}\n"
            f"Diff:\n{diff[-12000:]}\n"
            f"Plan:\n{json.dumps(asdict(plan))}\n"
            f"Context:\n{self._context(context, 'repair')}"
        )
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_content},
        ]
        if tool_history:
            for call, result in tool_history:
                messages.append({
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": call.call_id,
                            "name": call.tool_name,
                            "input": call.arguments,
                        }
                    ],
                })
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": call.call_id,
                            "content": result.output,
                            "is_error": result.is_error,
                        }
                    ],
                })

        req_body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 8192,
            "temperature": 0.1,
            "system": system_msg,
            "messages": messages,
        }
        if tools:
            req_body["tools"] = _format_anthropic_tools(tools)

        body = json.dumps(req_body).encode("utf-8")
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = self._request_json_api(
            f"{self.base_url}/messages",
            body,
            headers,
            "messages",
            self.model,
            120,
        )

        try:
            content_blocks = payload.get("content", [])
            if not isinstance(content_blocks, list):
                raise ProviderError("Anthropic response content must be a list")
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("Anthropic response did not contain content blocks") from exc

        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                call_id = block.get("id") or f"call_{int(time.time() * 1000)}"
                fn_name = block.get("name", "")
                raw_input = block.get("input", {})
                if not fn_name:
                    raise ProviderError("Anthropic tool_use block missing name")
                if isinstance(raw_input, str):
                    try:
                        raw_input = json.loads(raw_input)
                    except json.JSONDecodeError as exc:
                        raise ProviderError(f"Anthropic tool '{fn_name}' returned malformed JSON input: {raw_input}") from exc
                if not isinstance(raw_input, dict):
                    raise ProviderError(f"Anthropic tool '{fn_name}' input must be a dictionary, got {type(raw_input).__name__}")
                return ToolCall(call_id=str(call_id), tool_name=str(fn_name), arguments=raw_input)

        text_parts = [
            str(block.get("text", ""))
            for block in content_blocks
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        combined_text = "".join(text_parts).strip()
        if not combined_text:
            raise ProviderError("Anthropic response contained neither tool_use nor text content")

        return self._parse_json(combined_text)

    def analyze_failure(self, execution: ExecutionResult, diff: str, context: ProjectContext, plan: Plan) -> FailureAnalysis:
        data = self._json_call("Return only JSON with probable_root_cause, affected_files, recommended_fix.", f"Failed command: {execution.command}\nExit code: {execution.exit_code}\nstdout:\n{execution.stdout[-8000:]}\nstderr:\n{execution.stderr[-8000:]}\nDiff:\n{diff[-12000:]}\nPlan:\n{json.dumps(asdict(plan))}\nContext:\n{self._context(context, 'repair')}")
        return FailureAnalysis(str(data.get("probable_root_cause", "Unknown failure")), _strings(data.get("affected_files")), str(data.get("recommended_fix", "")))

    def review_changes_with_tools(
        self,
        task: str,
        plan: Plan,
        diff: str,
        context: ProjectContext,
        tools: list[ToolDefinition],
        tool_history: list[tuple[ToolCall, ToolResult]] | None = None,
    ) -> ToolCall | ReviewResult:
        system_msg = (
            "You are an expert code reviewer. Inspect modified files, related callers, callees, symbol references, configuration, and tests using available tools. "
            "Verify that changes are correct, complete, safe, adhere to the plan, and introduce no regressions.\n"
            "When review is complete, return only JSON: {\"verdict\": \"APPROVED\" or \"CHANGES_REQUIRED\", \"summary\": \"...\", \"findings\": [\"...\"]}."
        )
        user_content = (
            f"Task:\n{task}\n"
            f"Plan:\n{json.dumps(asdict(plan))}\n"
            f"Diff:\n{diff[-20000:]}\n"
            f"Context:\n{self._context(context, 'review')}"
        )
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_content},
        ]
        if tool_history:
            for call, result in tool_history:
                messages.append({
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": call.call_id,
                            "name": call.tool_name,
                            "input": call.arguments,
                        }
                    ],
                })
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": call.call_id,
                            "content": result.output,
                            "is_error": result.is_error,
                        }
                    ],
                })

        req_body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 8192,
            "temperature": 0.1,
            "system": system_msg,
            "messages": messages,
        }
        if tools:
            req_body["tools"] = _format_anthropic_tools(tools)

        body = json.dumps(req_body).encode("utf-8")
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = self._request_json_api(
            f"{self.base_url}/messages",
            body,
            headers,
            "messages",
            self.model,
            120,
        )

        try:
            content_blocks = payload.get("content", [])
            if not isinstance(content_blocks, list):
                raise ProviderError("Anthropic response content must be a list")
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("Anthropic response did not contain content blocks") from exc

        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                call_id = block.get("id") or f"anthropic_{int(time.time() * 1000)}"
                tool_name = block.get("name", "")
                arguments = block.get("input", {})
                if not tool_name:
                    raise ProviderError("Anthropic tool_use missing name")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError as exc:
                        raise ProviderError(f"Anthropic tool '{tool_name}' returned malformed JSON input: {arguments}") from exc
                return ToolCall(call_id=str(call_id), tool_name=str(tool_name), arguments=arguments if isinstance(arguments, dict) else {})

        text_parts = [
            str(block.get("text", ""))
            for block in content_blocks
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        combined_text = "".join(text_parts).strip()
        if not combined_text:
            raise ProviderError("Anthropic response contained neither tool_use nor text content")

        data = self._parse_json(combined_text)
        verdict = str(data.get("verdict", "CHANGES_REQUIRED"))
        if verdict not in {"APPROVED", "CHANGES_REQUIRED", "CHANGES_REQUESTED"}:
            verdict = "CHANGES_REQUIRED"
        return ReviewResult(verdict=verdict, summary=str(data.get("summary", "")), findings=_strings(data.get("findings")))

    def review_changes(self, task: str, plan: Plan, diff: str, context: ProjectContext) -> ReviewResult:
        data = self._json_call("Return only JSON with verdict (APPROVED or CHANGES_REQUIRED), summary, and findings.", f"Task:\n{task}\nPlan:\n{json.dumps(asdict(plan))}\nDiff:\n{diff[-20000:]}\nContext:\n{self._context(context, 'review')}")
        verdict = str(data.get("verdict", "CHANGES_REQUIRED"))
        if verdict not in {"APPROVED", "CHANGES_REQUIRED"}:
            verdict = "CHANGES_REQUIRED"
        return ReviewResult(verdict=verdict, summary=str(data.get("summary", "")), findings=_strings(data.get("findings")))


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


def _format_openai_tools(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": td.name,
                "description": td.description,
                "parameters": td.parameters,
            },
        }
        for td in tools
    ]


def _format_gemini_tools(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    return [
        {
            "functionDeclarations": [
                {
                    "name": td.name,
                    "description": td.description,
                    "parameters": td.parameters,
                }
                for td in tools
            ]
        }
    ]


def _format_anthropic_tools(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    return [
        {
            "name": td.name,
            "description": td.description,
            "input_schema": td.parameters,
        }
        for td in tools
    ]


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
        if len(json.dumps(candidate, ensure_ascii=False, default=str)) > budget and payload["selected_files"]:
            break
        payload = candidate
    if len(json.dumps(payload, ensure_ascii=False, default=str)) > budget:
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

    if config.provider == "mock" or config.provider.startswith("mock") or config.provider.startswith("provider_"):
        return MockProvider()
    if config.provider == "openai":
        return OpenAIProvider(config)
    if config.provider == "gemini":
        return GeminiProvider(config)
    if config.provider == "antigravity":
        return AntigravityProvider(config)
    if config.provider == "deepseek":
        return DeepSeekProvider(config)
    if config.provider == "anthropic":
        return AnthropicProvider(config)
    raise ProviderError(f"unsupported provider: {config.provider}")
