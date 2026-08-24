from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from .models import (
    FailureAnalysis,
    FileOperation,
    Plan,
    ProjectContext,
    ReviewResult,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from .tools import DEFAULT_MAX_OUTPUT_BYTES, ToolRegistry

DEFAULT_MAX_TOOL_STEPS = 8
DEFAULT_TOTAL_TOOL_BUDGET_BYTES = 32000
CONSECUTIVE_REPEAT_LIMIT = 3


def canonicalize_arguments(args: dict[str, Any]) -> str:
    """Deterministically serialize arguments dictionary regardless of key ordering."""
    try:
        return json.dumps(args, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(sorted(args.items()))


def history_to_dict(history: list[tuple[ToolCall, ToolResult]]) -> list[dict[str, Any]]:
    """Serialize tool history list into JSON-compatible dictionaries."""
    return [
        {
            "call": call.to_dict(),
            "result": result.to_dict(),
        }
        for call, result in history
    ]


def history_from_dict(data: list[dict[str, Any]]) -> list[tuple[ToolCall, ToolResult]]:
    """Deserialize JSON-compatible dictionaries back into tool history list."""
    history: list[tuple[ToolCall, ToolResult]] = []
    for item in data:
        call = ToolCall.from_dict(item["call"])
        result = ToolResult.from_dict(item["result"])
        history.append((call, result))
    return history


@dataclass
class ToolEngineResult:
    """Structured result produced by the ToolEngine execution loop."""
    file_operations: list[FileOperation] | None = None
    tool_history: list[tuple[ToolCall, ToolResult]] = field(default_factory=list)
    steps_used: int = 0
    total_tool_output_bytes: int = 0
    completed: bool = False
    termination_reason: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_operations": [op.__dict__ if hasattr(op, "__dict__") else op for op in (self.file_operations or [])] if self.file_operations is not None else None,
            "tool_history": history_to_dict(self.tool_history),
            "steps_used": self.steps_used,
            "total_tool_output_bytes": self.total_tool_output_bytes,
            "completed": self.completed,
            "termination_reason": self.termination_reason,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolEngineResult:
        file_ops_data = data.get("file_operations")
        file_ops = [FileOperation(**op) if isinstance(op, dict) else op for op in file_ops_data] if file_ops_data is not None else None
        return cls(
            file_operations=file_ops,
            tool_history=history_from_dict(data.get("tool_history", [])),
            steps_used=data.get("steps_used", 0),
            total_tool_output_bytes=data.get("total_tool_output_bytes", 0),
            completed=data.get("completed", False),
            termination_reason=data.get("termination_reason"),
            error_message=data.get("error_message"),
        )


class ToolEngine:
    """Coordinates a bounded, interactive tool-use loop with a provider and ToolRegistry."""

    def __init__(
        self,
        provider: Any,
        registry: ToolRegistry,
        max_tool_steps: int = DEFAULT_MAX_TOOL_STEPS,
        max_tool_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        total_tool_budget_bytes: int = DEFAULT_TOTAL_TOOL_BUDGET_BYTES,
    ):
        if max_tool_steps <= 0:
            raise ValueError(f"max_tool_steps must be greater than 0, got {max_tool_steps}")
        if total_tool_budget_bytes <= 0:
            raise ValueError(f"total_tool_budget_bytes must be greater than 0, got {total_tool_budget_bytes}")

        self.provider = provider
        self.registry = registry
        self.max_tool_steps = max_tool_steps
        self.max_tool_output_bytes = max_tool_output_bytes
        self.total_tool_budget_bytes = total_tool_budget_bytes

    def _call_provider_step(
        self,
        task: str,
        plan: Plan,
        context: ProjectContext,
        tools: list[ToolDefinition],
        history: list[tuple[ToolCall, ToolResult]],
        failure: FailureAnalysis | None = None,
        review: ReviewResult | None = None,
    ) -> Any:
        """Invoke provider step via generate_code_with_tools, step, or direct callable."""
        if hasattr(self.provider, "generate_code_with_tools"):
            return self.provider.generate_code_with_tools(
                task, plan, context, tools, tool_history=history, failure=failure, review=review
            )
        if hasattr(self.provider, "step"):
            return self.provider.step(
                task, plan, context, tools, tool_history=history, failure=failure, review=review
            )
        if callable(self.provider):
            return self.provider(
                task=task, plan=plan, context=context, tools=tools, tool_history=history, failure=failure, review=review
            )
        if hasattr(self.provider, "generate_code"):
            # Fallback 1-shot provider
            return self.provider.generate_code(task, plan, context, failure=failure, review=review)

        raise TypeError(f"Provider {type(self.provider)} does not support code generation or tool calling.")

    def run(
        self,
        task: str,
        plan: Plan,
        context: ProjectContext,
        initial_history: list[tuple[ToolCall, ToolResult]] | None = None,
        failure: FailureAnalysis | None = None,
        review: ReviewResult | None = None,
    ) -> ToolEngineResult:
        """Run the bounded tool loop until code changes are generated or bounds are reached."""
        history: list[tuple[ToolCall, ToolResult]] = initial_history if initial_history is not None else []

        # Calculate initial bytes and steps from restored history
        total_output_bytes = sum(len(result.output.encode("utf-8")) for _, result in history)
        steps_used = len(history)

        last_call_key: tuple[str, str] | None = None
        consecutive_repeat_count = 0

        while True:
            # 1. Ask provider for next action
            provider_response = self._call_provider_step(
                task, plan, context, tools, history, failure=failure, review=review
            )

            # 2. Check if provider returned final FileOperations list
            if isinstance(provider_response, list):
                # Validate items are FileOperations or compatible
                return ToolEngineResult(
                    file_operations=provider_response,
                    tool_history=history,
                    steps_used=steps_used,
                    total_tool_output_bytes=total_output_bytes,
                    completed=True,
                    termination_reason="completed",
                )

            # 3. Check if provider returned a ToolCall
            if not isinstance(provider_response, ToolCall):
                return ToolEngineResult(
                    file_operations=None,
                    tool_history=history,
                    steps_used=steps_used,
                    total_tool_output_bytes=total_output_bytes,
                    completed=False,
                    termination_reason="invalid_provider_response",
                    error_message=f"Provider returned invalid response type: {type(provider_response)}",
                )

            tool_call = provider_response

            # 4. Check step limit before executing next tool
            if steps_used >= self.max_tool_steps:
                return ToolEngineResult(
                    file_operations=None,
                    tool_history=history,
                    steps_used=steps_used,
                    total_tool_output_bytes=total_output_bytes,
                    completed=False,
                    termination_reason="max_steps_exceeded",
                    error_message=f"Reached maximum allowed tool steps ({self.max_tool_steps}).",
                )

            # 5. Check total byte budget before executing next tool
            if total_output_bytes >= self.total_tool_budget_bytes:
                return ToolEngineResult(
                    file_operations=None,
                    tool_history=history,
                    steps_used=steps_used,
                    total_tool_output_bytes=total_output_bytes,
                    completed=False,
                    termination_reason="budget_exhausted",
                    error_message=f"Exceeded total tool output budget ({self.total_tool_budget_bytes} bytes).",
                )

            # 6. Check repeated call circuit breaker
            call_key = (tool_call.tool_name, canonicalize_arguments(tool_call.arguments))
            if call_key == last_call_key:
                consecutive_repeat_count += 1
            else:
                last_call_key = call_key
                consecutive_repeat_count = 1

            if consecutive_repeat_count >= CONSECUTIVE_REPEAT_LIMIT:
                breaker_result = ToolResult(
                    call_id=tool_call.call_id,
                    tool_name=tool_call.tool_name,
                    output=f"Circuit breaker triggered: repeated identical tool call '{tool_call.tool_name}' detected {CONSECUTIVE_REPEAT_LIMIT} times consecutively. Please proceed to generate code changes or try a different action.",
                    is_error=True,
                )
                history.append((tool_call, breaker_result))
                steps_used += 1
                total_output_bytes += len(breaker_result.output.encode("utf-8"))

                # Reset repeat counter so provider gets one chance to recover or terminate
                consecutive_repeat_count = 0
                continue

            # 7. Execute tool securely via ToolRegistry
            tool_result = self.registry.execute(tool_call)
            history.append((tool_call, tool_result))
            steps_used += 1
            total_output_bytes += len(tool_result.output.encode("utf-8"))

