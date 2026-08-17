from __future__ import annotations

from .models import ExecutionResult, FailureAnalysis, Plan, ProjectContext
from .providers import AIProvider


class FailureAnalyzer:
    def __init__(self, provider: AIProvider):
        self.provider = provider

    def analyze(self, execution: ExecutionResult, diff: str, context: ProjectContext, plan: Plan) -> FailureAnalysis:
        return self.provider.analyze_failure(execution, diff, context, plan)
