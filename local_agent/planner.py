from __future__ import annotations

from .models import Plan, ProjectContext
from .providers import AIProvider


class Planner:
    def __init__(self, provider: AIProvider):
        self.provider = provider

    def create_plan(self, task: str, context: ProjectContext) -> Plan:
        if not task.strip():
            raise ValueError("task cannot be empty")
        return self.provider.generate_plan(task.strip(), context)
