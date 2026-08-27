from __future__ import annotations

from typing import Any

from .models import (
    Plan,
    ProjectContext,
    ProviderCapability,
    ProviderError,
    ReviewResult,
    RunReport,
    ToolCall,
    ToolDefinition,
    ToolExecutionPolicy,
    ToolResult,
)
from .providers import AIProvider
from .tool_engine import ToolEngine
from .tools import ToolRegistry


class Reviewer:
    def __init__(
        self,
        provider: AIProvider,
        registry: ToolRegistry | None = None,
        policy: ToolExecutionPolicy | None = None,
    ):
        self.provider = provider
        self.registry = registry
        self.policy = policy

    def review(
        self,
        task: str,
        plan: Plan,
        diff: str,
        context: ProjectContext,
        initial_history: list[tuple[ToolCall, ToolResult]] | None = None,
        report: RunReport | None = None,
    ) -> ReviewResult:
        provider_caps = getattr(self.provider, "capabilities", None)
        if (
            isinstance(provider_caps, (set, frozenset))
            and ProviderCapability.TOOL_USE in provider_caps
            and self.registry is not None
            and hasattr(self.provider, "review_changes_with_tools")
        ):
            captured_review: list[ReviewResult] = []

            def _review_step(
                task: str,
                plan: Plan,
                context: ProjectContext,
                tools: list[ToolDefinition],
                tool_history: list[tuple[ToolCall, ToolResult]],
                failure: Any = None,
                review: ReviewResult | None = None,
            ) -> Any:
                resp = self.provider.review_changes_with_tools(
                    task=task,
                    plan=plan,
                    diff=diff,
                    context=context,
                    tools=tools,
                    tool_history=tool_history,
                )
                if isinstance(resp, ToolCall):
                    return resp
                if isinstance(resp, ReviewResult):
                    captured_review.append(resp)
                    return [resp]
                if isinstance(resp, dict):
                    verdict = str(resp.get("verdict", "CHANGES_REQUIRED"))
                    if verdict not in {"APPROVED", "CHANGES_REQUIRED", "CHANGES_REQUESTED"}:
                        verdict = "CHANGES_REQUIRED"
                    findings = resp.get("findings", [])
                    findings_list = [str(f) for f in findings] if isinstance(findings, list) else []
                    res = ReviewResult(
                        verdict=verdict,
                        summary=str(resp.get("summary", "")),
                        findings=findings_list,
                    )
                    captured_review.append(res)
                    return [res]
                raise ProviderError(f"Unexpected response from review_changes_with_tools: {type(resp)}")

            engine = ToolEngine(provider=_review_step, registry=self.registry, policy=self.policy)
            result = engine.run(
                task=f"Review changes for: {task}",
                plan=plan,
                context=context,
                initial_history=initial_history,
            )

            if report is not None and result.metrics.total_calls > 0:
                report.tool_metrics.append(result.metrics)
                report.tool_history = list(result.tool_history)

            if captured_review:
                return captured_review[0]

            # Fallback if ToolEngine terminated without returning ReviewResult (e.g. step limit reached)
            return self.provider.review_changes(task, plan, diff, context)

        return self.provider.review_changes(task, plan, diff, context)
