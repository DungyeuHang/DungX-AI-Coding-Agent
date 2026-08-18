from __future__ import annotations

import json
import uuid
import datetime

from .models import Plan, ProjectContext, TaskPlan, Subtask, SubtaskStatus, TaskStatus
from .providers import AIProvider


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
        return self.provider.generate_plan(subtask.goal, context)

    def create_task_plan(self, task: str, context: ProjectContext) -> TaskPlan:
        """Decomposes a high-level task into a TaskPlan with a subtask graph."""
        if not task.strip():
            raise ValueError("task cannot be empty")

        # This is a conceptual implementation. The actual prompt would be more complex.
        # The provider's `generate_plan` is being repurposed to return a more complex structure.
        # In a real system, this might be a different provider method, e.g., `decompose_task`.
        system_prompt = (
            "You are an expert software architect. Decompose the user's task into a dependency graph of smaller, concrete engineering subtasks. "
            "Return only valid JSON with a 'plan' key. The value should be an object with 'subtasks', 'risks', and 'assumptions'. "
            "Each subtask must have 'id', 'title', 'goal', 'dependencies' (a list of other subtask ids), and 'acceptance_criteria' (a list of strings)."
        )
        user_prompt = f"Task:\n{task}\n\nProject context:\n{self.provider._context(context)}"

        # We assume the provider's `generate_plan` can be adapted or a new method is used
        # to get this structured JSON output. For this phase, we'll simulate it.
        # The `generate_plan` in `providers.py` would need to be updated to handle this.
        # For now, we'll assume it returns the structured data.
        decomposed_plan_data = self.provider.generate_plan(user_prompt, context) # This is a conceptual call

        now = datetime.datetime.now(datetime.timezone.utc)
        subtasks = [
            Subtask(
                subtask_id=s.get("id", str(uuid.uuid4())),
                title=s.get("title", "Untitled Subtask"),
                goal=s.get("goal", "No goal specified."),
                description=s.get("goal", "No goal specified."), # Use goal as description
                dependencies=s.get("dependencies", []),
                acceptance_criteria=s.get("acceptance_criteria", []),
                status=SubtaskStatus.PENDING,
                created_at=now,
                updated_at=now,
            ) for s in decomposed_plan_data.get("subtasks", [])
        ]

        errors = GraphValidator(subtasks).validate()
        if errors:
            raise ValueError(f"AI-generated task plan is invalid: {'; '.join(errors)}")

        return TaskPlan(
            objective=task,
            subtasks=subtasks,
            risks=decomposed_plan_data.get("risks", []),
            assumptions=decomposed_plan_data.get("assumptions", []),
        )
