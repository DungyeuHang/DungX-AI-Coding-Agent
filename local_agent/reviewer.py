from __future__ import annotations

from .models import Plan, ProjectContext, ReviewResult
from .providers import AIProvider


class Reviewer:
    def __init__(self, provider: AIProvider):
        self.provider = provider

    def review(self, task: str, plan: Plan, diff: str, context: ProjectContext) -> ReviewResult:
        return self.provider.review_changes(task, plan, diff, context)
