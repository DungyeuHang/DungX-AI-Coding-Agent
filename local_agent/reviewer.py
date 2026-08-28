from __future__ import annotations

import re
from typing import Any

from .models import (
    Plan,
    ProjectContext,
    ProviderCapability,
    ProviderError,
    ReviewConsensusRecord,
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

_HIGH_RISK_PATTERNS = [
    re.compile(r"(auth|security|crypto|permission|secret|token|credential|jwt)", re.I),
    re.compile(r"(schema|migration|prisma|alembic|models\.py)", re.I),
    re.compile(r"(\.github/workflows|Dockerfile|docker-compose|package\.json|pyproject\.toml)", re.I),
]


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


class DeliberativeReviewConsensus:
    """
    Coordinates independent dual-model review consensus for high-risk changes
    and records audit telemetry.
    """

    def __init__(
        self,
        primary_reviewer: Reviewer,
        secondary_reviewer: Reviewer | None = None,
        dual_review_enabled: bool = False,
        high_risk_dual_review: bool = True,
    ):
        self.primary_reviewer = primary_reviewer
        self.secondary_reviewer = secondary_reviewer
        self.dual_review_enabled = dual_review_enabled
        self.high_risk_dual_review = high_risk_dual_review

    def _check_high_risk(self, changed_files: list[str], plan: Plan | None) -> tuple[bool, str]:
        for f in changed_files:
            for pat in _HIGH_RISK_PATTERNS:
                if pat.search(f):
                    return True, f"Changed file '{f}' matches high-risk pattern '{pat.pattern}'"

        if plan and hasattr(plan, "amendments") and len(plan.amendments) >= 2:
            return True, f"Plan underwent {len(plan.amendments)} dynamic scope amendments"

        return False, ""

    def review(
        self,
        task: str,
        plan: Plan,
        diff: str,
        context: ProjectContext,
        changed_files: list[str] | None = None,
        initial_history: list[tuple[ToolCall, ToolResult]] | None = None,
        report: RunReport | None = None,
    ) -> ReviewResult:
        primary_result = self.primary_reviewer.review(
            task=task,
            plan=plan,
            diff=diff,
            context=context,
            initial_history=initial_history,
            report=report,
        )

        files = list(changed_files or (report.changed_files if report else []))
        is_hr, hr_reason = self._check_high_risk(files, plan)
        should_dual = (self.dual_review_enabled or (self.high_risk_dual_review and is_hr)) and (
            self.secondary_reviewer is not None and self.secondary_reviewer != self.primary_reviewer
        )

        if primary_result.verdict != "APPROVED" or not should_dual:
            record = ReviewConsensusRecord(
                role="review",
                primary_provider=getattr(self.primary_reviewer.provider, "__class__", "").__name__,
                primary_model=getattr(self.primary_reviewer.provider, "model", ""),
                primary_verdict=primary_result.verdict,
                final_consensus_verdict=primary_result.verdict,
                is_high_risk=is_hr,
                high_risk_reason=hr_reason,
            )
            primary_result.consensus_records = [record]
            if report is not None:
                report.review_consensus.append(record)
            return primary_result

        # Run secondary reviewer for high-risk consensus
        secondary_result = self.secondary_reviewer.review(
            task=task,
            plan=plan,
            diff=diff,
            context=context,
            initial_history=None,
            report=report,
        )

        if secondary_result.verdict == "APPROVED":
            final_verdict = "APPROVED"
            final_summary = f"{primary_result.summary} (Dual review verified: {secondary_result.summary})"
            final_findings = list(primary_result.findings)
            for f in secondary_result.findings:
                if f not in final_findings:
                    final_findings.append(f)
        else:
            final_verdict = "CHANGES_REQUIRED"
            final_summary = f"Dual review veto: secondary reviewer requested changes ({secondary_result.summary})"
            final_findings = list(primary_result.findings) + list(secondary_result.findings)

        record = ReviewConsensusRecord(
            role="review",
            primary_provider=getattr(self.primary_reviewer.provider, "__class__", "").__name__,
            primary_model=getattr(self.primary_reviewer.provider, "model", ""),
            secondary_provider=getattr(self.secondary_reviewer.provider, "__class__", "").__name__,
            secondary_model=getattr(self.secondary_reviewer.provider, "model", ""),
            primary_verdict=primary_result.verdict,
            secondary_verdict=secondary_result.verdict,
            final_consensus_verdict=final_verdict,
            is_high_risk=is_hr,
            high_risk_reason=hr_reason,
        )
        combined_result = ReviewResult(
            verdict=final_verdict,
            summary=final_summary,
            findings=final_findings,
            consensus_records=[record],
        )
        if report is not None:
            report.review_consensus.append(record)
        return combined_result
