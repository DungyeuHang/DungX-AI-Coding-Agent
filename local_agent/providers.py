from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict
from typing import Any

from .config import AgentConfig
from .models import CommandSpec, ExecutionResult, FailureAnalysis, FileOperation, Plan, ProjectContext, ReviewResult


class ProviderError(RuntimeError):
    """Raised when an AI provider cannot complete a structured operation."""


class AIProvider:
    def generate_plan(self, task: str, context: ProjectContext) -> Plan:
        raise NotImplementedError

    def generate_code(self, task: str, plan: Plan, context: ProjectContext, failure: FailureAnalysis | None = None, review: ReviewResult | None = None) -> list[FileOperation]:
        raise NotImplementedError

    def analyze_failure(self, execution: ExecutionResult, diff: str, context: ProjectContext, plan: Plan) -> FailureAnalysis:
        raise NotImplementedError

    def review_changes(self, task: str, plan: Plan, diff: str, context: ProjectContext) -> ReviewResult:
        raise NotImplementedError


class MockProvider(AIProvider):
    """Offline provider used for tests and safe workflow dry runs.

    It creates a real structured plan but intentionally never invents source
    changes. This makes the offline limitation visible instead of pretending a
    task was implemented.
    """

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
            return ReviewResult("CHANGES_REQUIRED", "No implementation diff was produced by the offline provider.", ["Configure AGENT_PROVIDER=openai and provide an API key to generate code."])
        return ReviewResult("CHANGES_REQUIRED", "The offline provider cannot verify generated changes.", ["Run a model-backed review before accepting the implementation."])


class OpenAIProvider(AIProvider):
    def __init__(self, config: AgentConfig):
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
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            raise ProviderError(f"OpenAI request failed: {exc}") from exc
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
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
        try:
            value = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"provider returned invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ProviderError("provider JSON must be an object")
        return value

    @staticmethod
    def _context(context: ProjectContext) -> str:
        return json.dumps(context.compact(), ensure_ascii=False, indent=2)

    def generate_plan(self, task: str, context: ProjectContext) -> Plan:
        data = self._json_call(
            "You are a careful software architect. Return only valid JSON with keys objective, files_to_inspect, files_likely_to_change, files_likely_to_create, steps, validation_strategy, risks. Never propose changes outside the project root.",
            f"Task:\n{task}\n\nProject context:\n{self._context(context)}",
        )
        return Plan(objective=str(data.get("objective", task)), files_to_inspect=_strings(data.get("files_to_inspect")), files_likely_to_change=_strings(data.get("files_likely_to_change")), files_likely_to_create=_strings(data.get("files_likely_to_create")), steps=_strings(data.get("steps")), validation_strategy=_strings(data.get("validation_strategy")), risks=_strings(data.get("risks")))

    def generate_code(self, task: str, plan: Plan, context: ProjectContext, failure: FailureAnalysis | None = None, review: ReviewResult | None = None) -> list[FileOperation]:
        data = self._json_call(
            "You are a careful coding agent. Return only valid JSON: {\"operations\":[{\"action\":\"write|create|delete\",\"path\":\"relative/path\",\"content\":\"full file content for write/create\",\"reason\":\"...\"}]}. Use complete file contents, minimal files, and only relative paths inside the project. Never modify secrets or .git.",
            f"Task:\n{task}\nPlan:\n{json.dumps(asdict(plan), indent=2)}\nContext:\n{self._context(context)}\nFailure:\n{asdict(failure) if failure else 'none'}\nReview:\n{asdict(review) if review else 'none'}",
        )
        operations: list[FileOperation] = []
        for item in data.get("operations", []):
            if isinstance(item, dict):
                operations.append(FileOperation(str(item.get("action", "")), str(item.get("path", "")), item.get("content"), str(item.get("reason", ""))))
        return operations

    def analyze_failure(self, execution: ExecutionResult, diff: str, context: ProjectContext, plan: Plan) -> FailureAnalysis:
        data = self._json_call("Return only JSON with probable_root_cause, affected_files, recommended_fix.", f"Failed command: {execution.command}\nExit code: {execution.exit_code}\nstdout:\n{execution.stdout[-8000:]}\nstderr:\n{execution.stderr[-8000:]}\nDiff:\n{diff[-12000:]}\nPlan:\n{json.dumps(asdict(plan))}\nContext:\n{self._context(context)}")
        return FailureAnalysis(str(data.get("probable_root_cause", "Unknown failure")), _strings(data.get("affected_files")), str(data.get("recommended_fix", "")))

    def review_changes(self, task: str, plan: Plan, diff: str, context: ProjectContext) -> ReviewResult:
        data = self._json_call("Return only JSON with verdict (APPROVED or CHANGES_REQUIRED), summary, and findings.", f"Task:\n{task}\nPlan:\n{json.dumps(asdict(plan))}\nDiff:\n{diff[-20000:]}\nContext:\n{self._context(context)}")
        verdict = str(data.get("verdict", "CHANGES_REQUIRED"))
        if verdict not in {"APPROVED", "CHANGES_REQUIRED"}:
            verdict = "CHANGES_REQUIRED"
        return ReviewResult(verdict, str(data.get("summary", "")), _strings(data.get("findings")))


def _strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def build_provider(config: AgentConfig) -> AIProvider:
    if config.provider == "mock":
        return MockProvider()
    if config.provider == "openai":
        return OpenAIProvider(config)
    raise ProviderError(f"unsupported provider: {config.provider}")
