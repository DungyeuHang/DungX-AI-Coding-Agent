from __future__ import annotations

from typing import Any

from .models import (
    ExecutionResult,
    FailureAnalysis,
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


class FailureAnalyzer:
    def __init__(
        self,
        provider: AIProvider,
        registry: ToolRegistry | None = None,
        policy: ToolExecutionPolicy | None = None,
    ):
        self.provider = provider
        self.registry = registry
        self.policy = policy

    def analyze(
        self,
        execution: ExecutionResult,
        diff: str,
        context: ProjectContext,
        plan: Plan,
        initial_history: list[tuple[ToolCall, ToolResult]] | None = None,
        report: RunReport | None = None,
    ) -> FailureAnalysis:
        provider_caps = getattr(self.provider, "capabilities", None)
        if (
            isinstance(provider_caps, (set, frozenset))
            and ProviderCapability.TOOL_USE in provider_caps
            and self.registry is not None
            and hasattr(self.provider, "analyze_failure_with_tools")
        ):
            captured_analysis: list[FailureAnalysis] = []

            def _diagnostic_step(
                task: str,
                plan: Plan,
                context: ProjectContext,
                tools: list[ToolDefinition],
                tool_history: list[tuple[ToolCall, ToolResult]],
                failure: FailureAnalysis | None = None,
                review: ReviewResult | None = None,
            ) -> Any:
                resp = self.provider.analyze_failure_with_tools(
                    execution=execution,
                    diff=diff,
                    context=context,
                    plan=plan,
                    tools=tools,
                    tool_history=tool_history,
                )
                if isinstance(resp, ToolCall):
                    return resp
                if isinstance(resp, FailureAnalysis):
                    captured_analysis.append(resp)
                    return [resp]  # Return list to signal completion to ToolEngine
                raise ProviderError(f"Unexpected response from analyze_failure_with_tools: {type(resp)}")

            engine = ToolEngine(provider=_diagnostic_step, registry=self.registry, policy=self.policy)
            result = engine.run(
                task=f"Diagnose failure for: {execution.command}",
                plan=plan,
                context=context,
                initial_history=initial_history,
            )

            if report is not None and result.metrics.total_calls > 0:
                report.tool_metrics.append(result.metrics)
                report.tool_history = list(result.tool_history)

            if captured_analysis:
                return captured_analysis[0]

            # Fallback if ToolEngine terminated without returning FailureAnalysis (e.g. step limit)
            return self.provider.analyze_failure(execution, diff, context, plan)

        return self.provider.analyze_failure(execution, diff, context, plan)
