from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from .models import (
    FailureAnalysis,
    FileOperation,
    Plan,
    PolicyAction,
    PolicyDecision,
    ProjectContext,
    RecoveryState,
    ReviewResult,
    RunReport,
    ToolCall,
    ToolDefinition,
    ToolExecutionMetrics,
    ToolExecutionPolicy,
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
    metrics: ToolExecutionMetrics = field(default_factory=ToolExecutionMetrics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_operations": [op.__dict__ if hasattr(op, "__dict__") else op for op in (self.file_operations or [])] if self.file_operations is not None else None,
            "tool_history": history_to_dict(self.tool_history),
            "steps_used": self.steps_used,
            "total_tool_output_bytes": self.total_tool_output_bytes,
            "completed": self.completed,
            "termination_reason": self.termination_reason,
            "error_message": self.error_message,
            "metrics": self.metrics.to_dict() if hasattr(self.metrics, "to_dict") else self.metrics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolEngineResult:
        file_ops_data = data.get("file_operations")
        file_ops = [FileOperation(**op) if isinstance(op, dict) else op for op in file_ops_data] if file_ops_data is not None else None
        metrics_data = data.get("metrics")
        metrics = ToolExecutionMetrics.from_dict(metrics_data) if isinstance(metrics_data, dict) else ToolExecutionMetrics()
        return cls(
            file_operations=file_ops,
            tool_history=history_from_dict(data.get("tool_history", [])),
            steps_used=data.get("steps_used", 0),
            total_tool_output_bytes=data.get("total_tool_output_bytes", 0),
            completed=data.get("completed", False),
            termination_reason=data.get("termination_reason"),
            error_message=data.get("error_message"),
            metrics=metrics,
        )


class ToolContextCompactor:
    """Deterministic, provider-agnostic compactor for model-facing exploration context."""

    def __init__(self, window: int = 2, max_context_bytes: int = 8000):
        self.window = max(1, window)
        self.max_context_bytes = max(500, max_context_bytes)

    def compact(
        self,
        history: list[tuple[ToolCall, ToolResult]],
    ) -> tuple[list[tuple[ToolCall, ToolResult]], int, int]:
        """Produce a separate, compacted model-facing exploration history from canonical history.

        Returns:
            (model_facing_history, compacted_entries_count, model_context_bytes)
        """
        if not history:
            return [], 0, 0

        model_history: list[tuple[ToolCall, ToolResult]] = []
        compacted_count = 0
        total_len = len(history)

        for idx, (call, result) in enumerate(history):
            # 1. Highest priority: Errors and circuit-breaker information are ALWAYS full fidelity
            if result.is_error or "Circuit breaker" in result.output:
                model_history.append((call, result))
                continue

            # 2. Recent window turns (last self.window items) are ALWAYS full fidelity
            if idx >= total_len - self.window:
                model_history.append((call, result))
                continue

            # 3. Older successful turns: apply deterministic structural reduction
            compacted_res = self._compact_result(call, result)
            if compacted_res.output != result.output:
                compacted_count += 1
            model_history.append((call, compacted_res))

        model_bytes = sum(len(res.output.encode("utf-8")) for _, res in model_history)
        return model_history, compacted_count, model_bytes

    def _compact_result(self, call: ToolCall, result: ToolResult) -> ToolResult:
        """Deterministically compact older tool output preserving useful semantic information."""
        output = result.output
        if len(output) <= 250:
            return result

        tool_name = call.tool_name
        if tool_name == "read_file_range":
            lines = output.splitlines()
            if len(lines) > 8:
                head = lines[:3]
                tail = lines[-3:]

                def _is_def(l: str) -> bool:
                    stripped = l.lstrip()
                    if ":" in stripped:
                        prefix, rest = stripped.split(":", 1)
                        if prefix.isdigit():
                            stripped = rest.lstrip()
                    return stripped.startswith(("def ", "class ", "async def "))

                defs = [l for l in lines[3:-3] if _is_def(l)]
                omitted_count = max(0, len(lines) - len(head) - len(tail) - len(defs))
                mid_parts = []
                if defs:
                    mid_parts.extend(defs)
                if omitted_count > 0:
                    mid_parts.append(f"... [{omitted_count} body lines omitted for compaction] ...")
                compacted_lines = head + mid_parts + tail
                compacted_output = "\n".join(compacted_lines)
                return ToolResult(
                    call_id=result.call_id,
                    tool_name=result.tool_name,
                    output=compacted_output,
                    is_error=result.is_error,
                    truncated=result.truncated,
                )

        elif tool_name in {"find_files", "grep_code", "search_symbols"}:
            lines = output.splitlines()
            if len(lines) > 10:
                head = lines[:10]
                compacted_lines = head + [f"... [{len(lines) - 10} additional matches omitted for compaction]"]
                return ToolResult(
                    call_id=result.call_id,
                    tool_name=result.tool_name,
                    output="\n".join(compacted_lines),
                    is_error=result.is_error,
                    truncated=result.truncated,
                )

        elif tool_name == "run_command_sandbox":
            lines = output.splitlines()
            if len(lines) > 10:
                head = lines[:5]
                tail = lines[-5:]
                compacted_lines = head + [f"... [{len(lines) - 10} stdout lines omitted]"] + tail
                return ToolResult(
                    call_id=result.call_id,
                    tool_name=result.tool_name,
                    output="\n".join(compacted_lines),
                    is_error=result.is_error,
                    truncated=result.truncated,
                )

        # Fallback for generic long text
        if len(output) > 500:
            compacted_output = output[:250] + "\n... [omitted for context efficiency] ...\n" + output[-200:]
            return ToolResult(
                call_id=result.call_id,
                tool_name=result.tool_name,
                output=compacted_output,
                is_error=result.is_error,
                truncated=result.truncated,
            )

        return result


class ToolEngine:
    """Coordinates a bounded, interactive tool-use loop with a provider and ToolRegistry."""

    def __init__(
        self,
        provider: Any,
        registry: ToolRegistry,
        policy: ToolExecutionPolicy | None = None,
        max_tool_steps: int | None = None,
        max_tool_output_bytes: int | None = None,
        total_tool_budget_bytes: int | None = None,
    ):
        self.provider = provider
        self.registry = registry

        if policy is not None:
            self.policy = policy
        else:
            steps = max_tool_steps if max_tool_steps is not None else DEFAULT_MAX_TOOL_STEPS
            output_bytes = max_tool_output_bytes if max_tool_output_bytes is not None else DEFAULT_MAX_OUTPUT_BYTES
            total_budget = total_tool_budget_bytes if total_tool_budget_bytes is not None else DEFAULT_TOTAL_TOOL_BUDGET_BYTES
            self.policy = ToolExecutionPolicy(
                max_tool_steps=steps,
                max_tool_output_bytes=output_bytes,
                total_tool_budget_bytes=total_budget,
            )

        self.max_tool_steps = self.policy.max_tool_steps
        self.max_tool_output_bytes = self.policy.max_tool_output_bytes
        self.total_tool_budget_bytes = self.policy.total_tool_budget_bytes

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
        start_time = time.perf_counter()
        history: list[tuple[ToolCall, ToolResult]] = initial_history if initial_history is not None else []
        tools = self.registry.definitions()

        compactor = ToolContextCompactor(
            window=self.policy.compaction_window,
            max_context_bytes=self.policy.max_context_bytes,
        )

        # Telemetry tracking structures
        seen_call_signatures: list[tuple[str, str]] = []
        calls_by_tool: dict[str, int] = {}
        output_bytes_by_tool: dict[str, int] = {}
        truncated_results: int = 0
        tool_errors: int = 0
        circuit_breaker_events: int = 0

        # Calculate initial bytes, steps, and telemetry from restored history
        for call, result in history:
            sig = (call.tool_name, canonicalize_arguments(call.arguments))
            seen_call_signatures.append(sig)
            calls_by_tool[call.tool_name] = calls_by_tool.get(call.tool_name, 0) + 1
            byte_len = len(result.output.encode("utf-8"))
            output_bytes_by_tool[call.tool_name] = output_bytes_by_tool.get(call.tool_name, 0) + byte_len
            if result.truncated:
                truncated_results += 1
            if result.is_error:
                tool_errors += 1
            if "Circuit breaker triggered" in result.output:
                circuit_breaker_events += 1

        total_output_bytes = sum(len(result.output.encode("utf-8")) for _, result in history)
        steps_used = len(history)

        last_call_key: tuple[str, str] | None = None
        consecutive_repeat_count = 0

        compacted_entries = 0
        model_context_bytes = total_output_bytes

        def _build_result(
            file_operations: list[FileOperation] | None,
            completed: bool,
            termination_reason: str,
            error_message: str | None = None,
        ) -> ToolEngineResult:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            total_calls = len(history)
            unique_calls = len(set(seen_call_signatures))
            repeated_calls = max(0, total_calls - unique_calls)

            metrics = ToolExecutionMetrics(
                total_calls=total_calls,
                unique_calls=unique_calls,
                repeated_calls=repeated_calls,
                calls_by_tool=dict(calls_by_tool),
                total_output_bytes=total_output_bytes,
                output_bytes_by_tool=dict(output_bytes_by_tool),
                truncated_results=truncated_results,
                tool_errors=tool_errors,
                circuit_breaker_events=circuit_breaker_events,
                steps_used=steps_used,
                history_entries=len(history),
                termination_reason=termination_reason,
                completed=completed,
                elapsed_ms=elapsed_ms,
                compacted_entries=compacted_entries,
                model_context_bytes=model_context_bytes,
            )
            return ToolEngineResult(
                file_operations=file_operations,
                tool_history=history,
                steps_used=steps_used,
                total_tool_output_bytes=total_output_bytes,
                completed=completed,
                termination_reason=termination_reason,
                error_message=error_message,
                metrics=metrics,
            )

        while True:
            # Derive model-facing context from canonical history
            model_history, compacted_entries, model_context_bytes = compactor.compact(history)

            # 1. Ask provider for next action with model-facing history
            provider_response = self._call_provider_step(
                task, plan, context, tools, model_history, failure=failure, review=review
            )

            # 2. Check if provider returned final FileOperations list
            if isinstance(provider_response, list):
                return _build_result(
                    file_operations=provider_response,
                    completed=True,
                    termination_reason="completed",
                )

            # 3. Check if provider returned a ToolCall
            if not isinstance(provider_response, ToolCall):
                return _build_result(
                    file_operations=None,
                    completed=False,
                    termination_reason="invalid_provider_response",
                    error_message=f"Provider returned invalid response type: {type(provider_response)}",
                )

            tool_call = provider_response
            sig = (tool_call.tool_name, canonicalize_arguments(tool_call.arguments))
            seen_call_signatures.append(sig)
            calls_by_tool[tool_call.tool_name] = calls_by_tool.get(tool_call.tool_name, 0) + 1

            # Check repeated call key for consecutive repetition tracking
            if sig == last_call_key:
                consecutive_repeat_count += 1
            else:
                last_call_key = sig
                consecutive_repeat_count = 1

            # 4. Evaluate policy pre-execution
            decision = self.policy.evaluate_call(
                tool_call=tool_call,
                steps_used=steps_used,
                total_output_bytes=total_output_bytes,
                calls_by_tool=calls_by_tool,
                consecutive_repeat_count=consecutive_repeat_count,
            )

            if decision.action == PolicyAction.TERMINATE:
                return _build_result(
                    file_operations=None,
                    completed=False,
                    termination_reason=decision.reason or "policy_terminated",
                    error_message=decision.message or "Execution terminated by policy.",
                )

            if decision.action == PolicyAction.REJECT:
                tool_errors += 1
                reject_result = ToolResult(
                    call_id=tool_call.call_id,
                    tool_name=tool_call.tool_name,
                    output=decision.message or f"Tool call '{tool_call.tool_name}' was rejected by execution policy.",
                    is_error=True,
                )
                reject_bytes = len(reject_result.output.encode("utf-8"))
                output_bytes_by_tool[tool_call.tool_name] = output_bytes_by_tool.get(tool_call.tool_name, 0) + reject_bytes
                history.append((tool_call, reject_result))
                steps_used += 1
                total_output_bytes += reject_bytes
                continue

            if decision.action == PolicyAction.CIRCUIT_BREAKER:
                circuit_breaker_events += 1
                tool_errors += 1
                breaker_result = ToolResult(
                    call_id=tool_call.call_id,
                    tool_name=tool_call.tool_name,
                    output=decision.message or f"Circuit breaker triggered: repeated identical tool call '{tool_call.tool_name}' detected.",
                    is_error=True,
                )
                breaker_bytes = len(breaker_result.output.encode("utf-8"))
                output_bytes_by_tool[tool_call.tool_name] = output_bytes_by_tool.get(tool_call.tool_name, 0) + breaker_bytes
                history.append((tool_call, breaker_result))
                steps_used += 1
                total_output_bytes += breaker_bytes

                # Reset repeat counter so provider gets one chance to recover or terminate
                consecutive_repeat_count = 0
                continue

            # 5. Execute tool securely via ToolRegistry
            tool_result = self.registry.execute(tool_call)
            res_bytes = len(tool_result.output.encode("utf-8"))
            output_bytes_by_tool[tool_call.tool_name] = output_bytes_by_tool.get(tool_call.tool_name, 0) + res_bytes
            if tool_result.truncated:
                truncated_results += 1
            if tool_result.is_error:
                tool_errors += 1

            history.append((tool_call, tool_result))
            steps_used += 1
            total_output_bytes += res_bytes


class IterationHistoryCompactor:
    """Summarizes prior iterations for subsequent repair attempts with strict byte limits."""

    @staticmethod
    def build_cross_iteration_context(
        recovery_state: RecoveryState | None,
        plan: Plan | None,
        report: RunReport | None = None,
        max_bytes: int = 4000,
    ) -> str:
        """Construct a bounded summary of previous iterations, active plan version, amendments, and failure telemetry."""
        lines: list[str] = []

        if plan is not None:
            lines.append("=== Active Plan State ===")
            lines.append(f"Plan Version: v{getattr(plan, 'version', 1)}")
            lines.append(f"Allowed Modify Paths: {getattr(plan, 'files_likely_to_change', [])}")
            lines.append(f"Allowed Create Paths: {getattr(plan, 'files_likely_to_create', [])}")
            amendments = getattr(plan, "amendments", [])
            if amendments:
                lines.append("Plan Amendments:")
                for a in amendments:
                    prop = getattr(a, "proposal", None)
                    path = getattr(prop, "path", "") if prop else ""
                    reason = getattr(prop, "reason", "") if prop else ""
                    lines.append(f"  - v{a.version}: Added '{path}' ({reason[:80]})")

        if recovery_state is not None:
            lines.append("\n=== Prior Iteration Telemetry ===")
            if recovery_state.repair_signatures:
                lines.append("Previous Attempts:")
                for sig in recovery_state.repair_signatures[-3:]:
                    files = ", ".join(sig.affected_files) if sig.affected_files else "none"
                    lines.append(f"  - Iteration {sig.iteration}: Modified [{files}] -> Status: Failed ({sig.failure_category})")

            if recovery_state.failure_history:
                last_f = recovery_state.failure_history[-1]
                lines.append(f"\nLatest Root Cause: {last_f.probable_root_cause[:250]}")
                if last_f.recommended_fix:
                    lines.append(f"Recommended Fix: {last_f.recommended_fix[:250]}")
                if last_f.diagnostic_evidence:
                    first_ev = last_f.diagnostic_evidence[0]
                    lines.append(f"Diagnostic Evidence: {first_ev.command} (exit {first_ev.exit_code})")

            if recovery_state.review_history:
                last_r = recovery_state.review_history[-1]
                if last_r.verdict != "APPROVED":
                    findings = "; ".join(last_r.findings[:3])
                    lines.append(f"Latest Review Feedback: {last_r.summary[:150]} (Findings: {findings[:150]})")

        summary = "\n".join(lines).strip()
        encoded = summary.encode("utf-8")
        if len(encoded) > max_bytes:
            suffix = "\n...[cross-iteration context truncated]"
            suffix_bytes = suffix.encode("utf-8")
            if max_bytes >= len(suffix_bytes):
                target_bytes = max_bytes - len(suffix_bytes)
                truncated_text = encoded[:target_bytes].decode("utf-8", errors="ignore")
                summary = truncated_text + suffix
            else:
                summary = encoded[:max_bytes].decode("utf-8", errors="ignore")
        return summary

