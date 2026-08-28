from __future__ import annotations

import datetime
import json
from typing import Any
import uuid

from .models import (
    DAGProposal,
    FailureAnalysis,
    Plan,
    PlanProposal,
    ProjectContext,
    ProviderCapability,
    ProviderError,
    RunReport,
    Subtask,
    SubtaskStatus,
    Task,
    TaskPlan,
    TaskPlanAmendment,
    TaskStatus,
    ToolCall,
    ToolDefinition,
    ToolExecutionPolicy,
    ToolResult,
)
from .providers import AIProvider, build_provider
from .tool_engine import ToolEngine
from .tools import ToolRegistry


class GraphValidator:
    def __init__(self, subtasks: list[Subtask]):
        self.subtasks = subtasks
        self.subtask_map = {s.subtask_id: s for s in subtasks}

    def validate(self) -> list[str]:
        errors = []
        errors.extend(self._validate_ids())
        errors.extend(self._validate_content())
        if not errors: # Only check for cycles if basic validation passes
            errors.extend(self._validate_cycles())
        return errors

    def _validate_ids(self) -> list[str]:
        errors = []
        if len(self.subtask_map) != len(self.subtasks):
            errors.append("Duplicate subtask IDs found.")
        for subtask in self.subtasks:
            for dep_id in subtask.dependencies:
                if dep_id not in self.subtask_map:
                    errors.append(f"Subtask '{subtask.subtask_id}' has a missing dependency: '{dep_id}'.")
                if dep_id == subtask.subtask_id:
                    errors.append(f"Subtask '{subtask.subtask_id}' has a self-dependency.")
        return errors

    def _validate_content(self) -> list[str]:
        errors = []
        for subtask in self.subtasks:
            if not subtask.title.strip():
                errors.append(f"Subtask '{subtask.subtask_id}' has an empty title.")
            if not subtask.goal.strip():
                errors.append(f"Subtask '{subtask.subtask_id}' has an empty goal.")
        return errors

    def _validate_cycles(self) -> list[str]:
        path = set()
        visited = set()
        for subtask_id in self.subtask_map:
            if subtask_id not in visited:
                if self._is_cyclic_util(subtask_id, visited, path):
                    return ["Dependency cycle detected in the task plan."]
        return []

    def _is_cyclic_util(self, subtask_id: str, visited: set[str], path: set[str]) -> bool:
        visited.add(subtask_id)
        path.add(subtask_id)

        for neighbour_id in self.subtask_map[subtask_id].dependencies:
            if neighbour_id not in visited:
                if self._is_cyclic_util(neighbour_id, visited, path):
                    return True
            elif neighbour_id in path:
                return True

        path.remove(subtask_id)
        return False


class Planner:
    def __init__(
        self,
        provider: AIProvider,
        registry: ToolRegistry | None = None,
        policy: ToolExecutionPolicy | None = None,
    ):
        self.provider = provider
        self.registry = registry
        self.policy = policy

    @staticmethod
    def _normalize_plan_dict(data: dict[str, Any], default_objective: str) -> Plan:
        return Plan(
            objective=str(data.get("objective", default_objective)),
            files_to_inspect=[str(f) for f in data.get("files_to_inspect", [])] if isinstance(data.get("files_to_inspect"), list) else [],
            files_likely_to_change=[str(f) for f in (data.get("files_likely_to_change") or data.get("files_to_modify") or [])] if isinstance(data.get("files_likely_to_change") or data.get("files_to_modify"), list) else [],
            files_likely_to_create=[str(f) for f in (data.get("files_likely_to_create") or data.get("files_to_create") or [])] if isinstance(data.get("files_likely_to_create") or data.get("files_to_create"), list) else [],
            steps=[str(s) for s in data.get("steps", [default_objective])] if isinstance(data.get("steps"), list) else [default_objective],
            validation_strategy=[str(v) for v in (data.get("validation_strategy") or data.get("validation_commands") or [])] if isinstance(data.get("validation_strategy") or data.get("validation_commands"), list) else [],
            risks=[str(r) for r in data.get("risks", [])] if isinstance(data.get("risks"), list) else [],
        )

    def _run_tool_assisted_planning(
        self,
        task: str,
        context: ProjectContext,
        initial_history: list[tuple[ToolCall, ToolResult]] | None = None,
        report: RunReport | None = None,
    ) -> Plan | TaskPlan | None:
        provider_caps = getattr(self.provider, "capabilities", None)
        if not (
            isinstance(provider_caps, (set, frozenset))
            and ProviderCapability.TOOL_USE in provider_caps
            and self.registry is not None
            and hasattr(self.provider, "generate_plan_with_tools")
        ):
            return None

        captured_plan: list[Plan | TaskPlan] = []

        def _planning_step(
            task: str,
            plan: Plan | None,
            context: ProjectContext,
            tools: list[ToolDefinition],
            tool_history: list[tuple[ToolCall, ToolResult]],
            failure: Any = None,
            review: Any = None,
        ) -> Any:
            resp = self.provider.generate_plan_with_tools(
                task=task,
                context=context,
                tools=tools,
                tool_history=tool_history,
            )
            if isinstance(resp, ToolCall):
                return resp
            if isinstance(resp, (Plan, TaskPlan)):
                captured_plan.append(resp)
                return [resp]
            if isinstance(resp, dict):
                if "subtasks" in resp:
                    subtasks = [Subtask.from_dict(s) if isinstance(s, dict) else s for s in resp["subtasks"]]
                    tp = TaskPlan(
                        objective=resp.get("objective", task),
                        subtasks=subtasks,
                        risks=resp.get("risks", []),
                    )
                    captured_plan.append(tp)
                    return [tp]
                p = self._normalize_plan_dict(resp, task)
                captured_plan.append(p)
                return [p]
            raise ProviderError(f"Unexpected response from generate_plan_with_tools: {type(resp)}")

        dummy_plan = Plan(objective=task, steps=[task])
        engine = ToolEngine(provider=_planning_step, registry=self.registry, policy=self.policy)
        result = engine.run(
            task=f"Plan architecture for: {task}",
            plan=dummy_plan,
            context=context,
            initial_history=initial_history,
        )

        if report is not None and result.metrics.total_calls > 0:
            report.tool_metrics.append(result.metrics)
            report.tool_history = list(result.tool_history)

        if captured_plan:
            return captured_plan[0]
        return None

    def create_subtask_plan(
        self,
        subtask: Subtask,
        context: ProjectContext,
        initial_history: list[tuple[ToolCall, ToolResult]] | None = None,
        report: RunReport | None = None,
    ) -> Plan:
        """Creates a simple, single-step plan for executing one subtask, using tools if available."""
        plan = self._run_tool_assisted_planning(
            task=subtask.goal,
            context=context,
            initial_history=initial_history,
            report=report,
        )
        if isinstance(plan, Plan):
            return plan
        if isinstance(plan, TaskPlan):
            return Plan(
                objective=plan.objective or subtask.goal,
                steps=[s.goal or s.title for s in plan.subtasks] if plan.subtasks else [subtask.goal],
                risks=plan.risks,
            )

        # Single-shot fallback
        raw_plan = self.provider.generate_plan(subtask.goal, context)
        if isinstance(raw_plan, Plan):
            return raw_plan
        if isinstance(raw_plan, dict):
            return self._normalize_plan_dict(raw_plan, subtask.goal)
        return Plan(objective=subtask.goal, steps=[subtask.goal])

    def create_plan_for_task(
        self,
        task: str,
        context: ProjectContext,
        initial_history: list[tuple[ToolCall, ToolResult]] | None = None,
        report: RunReport | None = None,
    ) -> Plan:
        """Creates a Plan for a direct single-task objective, using tools if available."""
        plan = self._run_tool_assisted_planning(
            task=task,
            context=context,
            initial_history=initial_history,
            report=report,
        )
        if isinstance(plan, Plan):
            return plan
        if isinstance(plan, TaskPlan):
            return Plan(
                objective=plan.objective or task,
                steps=[s.goal or s.title for s in plan.subtasks] if plan.subtasks else [task],
                risks=plan.risks,
            )

        # Single-shot fallback
        raw_plan = self.provider.generate_plan(task, context)
        if isinstance(raw_plan, Plan):
            return raw_plan
        if isinstance(raw_plan, dict):
            return self._normalize_plan_dict(raw_plan, task)
        return Plan(objective=task, steps=[task])

    def create_task_plan(
        self,
        task: str,
        context: ProjectContext,
        initial_history: list[tuple[ToolCall, ToolResult]] | None = None,
        report: RunReport | None = None,
    ) -> TaskPlan:
        """Decomposes a high-level task into a TaskPlan with a subtask graph, using tools if available."""
        if not task.strip():
            raise ValueError("task cannot be empty")

        plan = self._run_tool_assisted_planning(
            task=task,
            context=context,
            initial_history=initial_history,
            report=report,
        )
        if isinstance(plan, TaskPlan):
            return plan
        if plan is None:
            plan = self.provider.generate_plan(task, context)
            if isinstance(plan, TaskPlan):
                return plan
            if isinstance(plan, dict) and "subtasks" in plan:
                subtasks = [Subtask.from_dict(s) if isinstance(s, dict) else s for s in plan["subtasks"]]
                return TaskPlan(objective=plan.get("objective", task), subtasks=subtasks, risks=plan.get("risks", []))

        now = datetime.datetime.now(datetime.timezone.utc)
        steps = getattr(plan, "steps", None) or (plan.get("steps") if isinstance(plan, dict) else None) or [task]
        subtasks: list[Subtask] = []
        for step in steps:
            step_text = step.strip() if isinstance(step, str) else str(step)
            if not step_text:
                continue
            subtasks.append(
                Subtask(
                    subtask_id=str(uuid.uuid4()),
                    title=step_text,
                    goal=step_text,
                    description=step_text,
                    dependencies=[subtasks[-1].subtask_id] if subtasks else [],
                    acceptance_criteria=[],
                    status=SubtaskStatus.PENDING,
                    created_at=now,
                    updated_at=now,
                )
            )

        errors = GraphValidator(subtasks).validate()
        if errors:
            raise ValueError(f"AI-generated task plan is invalid: {'; '.join(errors)}")

        risks = getattr(plan, "risks", None) or (plan.get("risks", []) if isinstance(plan, dict) else [])
        return TaskPlan(
            objective=task,
            subtasks=subtasks,
            risks=risks,
            assumptions=[],
        )

    def replan_with_context(
        self,
        task: str | Task,
        current_plan: Plan,
        context: ProjectContext,
        failure: FailureAnalysis | None = None,
        discovery_evidence: list[str] | None = None,
        initial_history: list[tuple[ToolCall, ToolResult]] | None = None,
        report: RunReport | None = None,
    ) -> Plan:
        """
        Produce a revised plan incorporating failure feedback or newly discovered scope,
        while preserving plan version history and active amendments.
        """
        task_text = task.objective if hasattr(task, "objective") else str(task)
        new_plan = self._run_tool_assisted_planning(
            task=task_text,
            context=context,
            initial_history=initial_history,
            report=report,
        )
        if new_plan is None:
            new_plan = self.provider.generate_plan(task_text, context)

        if not isinstance(new_plan, Plan):
            if isinstance(new_plan, dict):
                new_plan = self._normalize_plan_dict(new_plan, task_text)
            else:
                new_plan = current_plan

        # Preserve version and historical amendments
        new_plan.version = getattr(current_plan, "version", 1)
        new_plan.amendments = list(getattr(current_plan, "amendments", []))

        # Merge allowed paths so historical amendments are not dropped
        for p in current_plan.files_likely_to_change:
            if p not in new_plan.files_likely_to_change:
                new_plan.files_likely_to_change.append(p)
        for p in current_plan.files_likely_to_create:
            if p not in new_plan.files_likely_to_create:
                new_plan.files_likely_to_create.append(p)

        return new_plan

    def create_dag_proposal(
        self,
        task: str | Task,
        current_task_plan: TaskPlan,
        failure: FailureAnalysis | None = None,
        context: ProjectContext | None = None,
    ) -> DAGProposal | None:
        """Propose structured additions, removals, dependency updates, or invalidations for a TaskPlan."""
        task_text = task.objective if hasattr(task, "objective") else str(task)
        if hasattr(self.provider, "propose_plan_modification"):
            prop = self.provider.propose_plan_modification(task_text, current_task_plan, failure)
            if prop:
                if isinstance(prop, DAGProposal):
                    return prop
                if isinstance(prop, PlanProposal):
                    return DAGProposal.from_plan_proposal(prop)
                if isinstance(prop, dict):
                    return DAGProposal.from_dict(prop)
        return None
