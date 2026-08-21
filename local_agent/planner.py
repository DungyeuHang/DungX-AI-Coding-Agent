from __future__ import annotations

import json
import uuid
import datetime

from .models import Plan, ProjectContext, TaskPlan, Subtask, SubtaskStatus, TaskStatus
from .providers import AIProvider, build_provider


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
    def __init__(self, provider: AIProvider):
        self.provider = provider

    def create_subtask_plan(self, subtask: Subtask, context: ProjectContext) -> Plan:
        """Creates a simple, single-step plan for executing one subtask."""
        # This is a simplified plan for the orchestrator's internal loop.
        # The AI gets the broader context from the continuation_context.
        plan = self.provider.generate_plan(subtask.goal, context)
        if isinstance(plan, Plan):
            return plan
        if isinstance(plan, dict):
            return Plan(
                objective=plan.get("objective", subtask.goal),
                steps=plan.get("steps", [subtask.goal]),
                files_likely_to_change=plan.get("files_likely_to_change", plan.get("files_to_modify", [])),
                files_likely_to_create=plan.get("files_likely_to_create", plan.get("files_to_create", [])),
                risks=plan.get("risks", []),
            )
        return Plan(objective=subtask.goal, steps=[subtask.goal])

    def create_task_plan(self, task: str, context: ProjectContext) -> TaskPlan:
        """Decomposes a high-level task into a TaskPlan with a subtask graph."""
        if not task.strip():
            raise ValueError("task cannot be empty")

        # The public AIProvider contract is `generate_plan(...) -> Plan`. We
        # normalize that provider-level Plan into the structured TaskPlan the
        # scheduler executes: each Plan step becomes a Subtask in a dependency
        # chain, and the Plan risks become the TaskPlan risks.
        plan = self.provider.generate_plan(task, context)
        if isinstance(plan, TaskPlan):
            return plan
        if isinstance(plan, dict) and "subtasks" in plan:
            subtasks = [Subtask.from_dict(s) if isinstance(s, dict) else s for s in plan["subtasks"]]
            return TaskPlan(objective=plan.get("objective", task), subtasks=subtasks, risks=plan.get("risks", []))

        now = datetime.datetime.now(datetime.timezone.utc)
        steps = getattr(plan, "steps", None) or [task]
        subtasks: list[Subtask] = []
        for step in steps:
            step_text = step.strip()
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

        return TaskPlan(
            objective=task,
            subtasks=subtasks,
            risks=plan.risks,
            assumptions=[],
        )
