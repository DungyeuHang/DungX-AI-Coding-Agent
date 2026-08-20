from __future__ import annotations

import datetime
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

from local_agent.config import AgentConfig
from local_agent.context import ContextSelector
from local_agent.impact import ChangeImpactAnalyzer
from local_agent.models import (ChangeImpact, Checkpoint, CommandSpec,
                                ExecutionResult, FailureAnalysis,
                                FileOperation, Plan, ProjectContext,
                                ProviderMetric, ReviewResult, Subtask,
                                SubtaskStatus, Task, TaskPlan, TaskStatus,
                                ValidationCommand, ValidationPlan)
from local_agent.orchestrator import Orchestrator
from local_agent.providers import AIProvider, ProviderError, QuotaExceededError, _bounded_context
from local_agent.repository import RepositoryIntelligence
from local_agent.storage import JsonFileStorage, TaskStorage
from local_agent.validation import ValidationIntelligence


class MockTaskStorage(TaskStorage):
    def __init__(self):
        self.tasks: dict[str, Task] = {}
        self.checkpoints: dict[str, Checkpoint] = {}

    def save_task(self, task: Task) -> None:
        self.tasks[task.task_id] = task

    def load_task(self, task_id: str) -> Task:
        if task_id not in self.tasks:
            raise FileNotFoundError(f"Task with ID {task_id} not found.")
        return self.tasks[task_id]

    def list_tasks(self) -> list[Task]:
        return list(self.tasks.values())

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        self.checkpoints[checkpoint.checkpoint_id] = checkpoint

    def load_checkpoint(self, checkpoint_id: str) -> Checkpoint:
        if checkpoint_id not in self.checkpoints:
            raise FileNotFoundError(f"Checkpoint with ID {checkpoint_id} not found.")
        return self.checkpoints[checkpoint_id]

    def save_scheduler_state(self, state) -> None: pass
    def load_scheduler_state(self): return __import__("local_agent.models").SchedulerState()
    def save_provider_configs(self, configs) -> None: pass
    def load_provider_configs(self) -> list: return []
    def save_semantic_index(self, index) -> None: pass
    def load_semantic_index(self): return __import__("local_agent.models").SemanticIndex()




class CapturingProvider(AIProvider):
    def __init__(self, fail_on_code_gen: bool = False, fail_on_plan: bool = False, fail_on_analyze_failure: bool = False, fail_on_review: bool = False, quota_exceeded_on_attempt: int | None = None):
        self.plan_context = None
        self.serialized_context = None
        self.fail_on_code_gen = fail_on_code_gen
        self.fail_on_plan = fail_on_plan
        self.fail_on_analyze_failure = fail_on_analyze_failure
        self.fail_on_review = fail_on_review
        self.quota_exceeded_on_attempt = quota_exceeded_on_attempt
        self.code_gen_attempts = 0
        self.plan_attempts = 0
        self.analyze_failure_attempts = 0
        self.review_attempts = 0
        self._provider_metrics = [] # Initialize metrics list

    def generate_plan(self, task: str, context: ProjectContext) -> Plan:
        self.plan_attempts += 1
        if self.fail_on_plan:
            raise ProviderError("Planning failed")
        if self.quota_exceeded_on_attempt == self.plan_attempts:
            raise QuotaExceededError("Quota exceeded during planning")
        self.plan_context = context
        self.serialized_context = _bounded_context(context, 30000, 20, 10000)
        return Plan(objective=task, steps=["Use the impact analysis and validation plan to guide the change."])

    def generate_code(self, task: str, plan: Plan, context: ProjectContext, failure: FailureAnalysis | None = None, review: ReviewResult | None = None) -> list[FileOperation]:
        self.code_gen_attempts += 1
        if self.fail_on_code_gen:
            raise ProviderError("Code generation failed")
        if self.quota_exceeded_on_attempt == self.code_gen_attempts:
            raise QuotaExceededError("Quota exceeded during code generation")
        return [FileOperation("modify", "src/app.py", content="def hello(): return 'world v2'\n", reason="Updated app")]

    def analyze_failure(self, execution, diff, context, plan):
        self.analyze_failure_attempts += 1
        if self.fail_on_analyze_failure:
            raise ProviderError("Failure analysis failed")
        if self.quota_exceeded_on_attempt == self.analyze_failure_attempts:
            raise QuotaExceededError("Quota exceeded during failure analysis")
        return FailureAnalysis("deterministic test failure", [], "inspect the failing test")

    def review_changes(self, task, plan, diff, context):
        self.review_attempts += 1
        if self.fail_on_review:
            raise ProviderError("Review failed")
        if self.quota_exceeded_on_attempt == self.review_attempts:
            raise QuotaExceededError("Quota exceeded during review")
        return ReviewResult("APPROVED", "No changes were generated by the fake provider.")


class Phase39Tests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.storage_dir = self.root / ".agent_data"
        self.storage = JsonFileStorage(self.storage_dir)
        self.config = AgentConfig.from_environment(self.root, max_iterations=1)
        self._setup_repository()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root)

    def _setup_repository(self):
        (self.root / "src" / "app.py").mkdir(parents=True)
        (self.root / "src" / "app.py").write_text("def hello(): return 'world'\n", encoding="utf-8")
        (self.root / "tests" / "test_app.py").mkdir(parents=True)
        (self.root / "tests" / "test_app.py").write_text("import pytest\nfrom src.app import hello\ndef test_hello(): assert hello() == 'world'\n", encoding="utf-8")
        (self.root / "package.json").write_text(json.dumps({"name": "test-app", "scripts": {"test": "pytest"}}), encoding="utf-8")

    def _create_task_with_plan(self, task_objective: str) -> Task:
        now = datetime.datetime.now(datetime.timezone.utc)
        subtask = Subtask(subtask_id="sub1", title=task_objective, goal=task_objective, created_at=now)
        plan = TaskPlan(objective=task_objective, subtasks=[subtask])
        task = Task(task_id=str(uuid.uuid4()), objective=task_objective, status=TaskStatus.PENDING, created_at=now, updated_at=now, plan=plan)
        self.storage.save_task(task)
        return task

    def test_task_creation_and_persistence(self):
        task_objective = "Implement a new feature"
        provider = CapturingProvider()
        orchestrator = Orchestrator(self.config, provider, self.storage)
        task = self._create_task_with_plan(task_objective)

        report = orchestrator.run(task, "sub1")

        self.assertIsNotNone(report.task_id)
        loaded_task = self.storage.load_task(report.task_id)
        self.assertEqual(loaded_task.objective, task_objective)
        self.assertEqual(loaded_task.status, TaskStatus.COMPLETED)
        self.assertGreater(len(loaded_task.plan.subtasks), 0)
        self.assertEqual(loaded_task.plan.subtasks[0].status, SubtaskStatus.COMPLETED)

    def test_task_reload_after_process_restart_simulation(self):
        task_objective = "Implement a new feature"
        provider = CapturingProvider()
        orchestrator1 = Orchestrator(self.config, provider, self.storage)
        task = self._create_task_with_plan(task_objective)
        report1 = orchestrator1.run(task, "sub1")
        task_id = report1.task_id

        # Simulate restart by creating a new orchestrator instance
        orchestrator2 = Orchestrator(self.config, provider, self.storage)
        loaded_task = orchestrator2.storage.load_task(task_id)
        self.assertEqual(loaded_task.objective, task_objective)
        self.assertEqual(loaded_task.status, TaskStatus.COMPLETED)

    def test_subtask_state_transitions(self):
        task_objective = "Implement a new feature"
        provider = CapturingProvider()
        orchestrator = Orchestrator(self.config, provider, self.storage)
        task = self._create_task_with_plan(task_objective)
        report = orchestrator.run(task, "sub1")
        task = self.storage.load_task(report.task_id)

        self.assertEqual(task.plan.subtasks[0].status, SubtaskStatus.COMPLETED)
        self.assertEqual(task.status, TaskStatus.COMPLETED)

    def test_checkpoint_persistence_and_reload(self):
        task_objective = "Implement a new feature"
        provider = CapturingProvider(quota_exceeded_on_attempt=1) # Fail on first code gen attempt
        orchestrator = Orchestrator(self.config, provider, self.storage)
        task = self._create_task_with_plan(task_objective)
        report = orchestrator.run(task, "sub1")

        self.assertEqual(report.outcome, "QUOTA_EXCEEDED")
        task_after_run = self.storage.load_task(report.task_id)
        self.assertEqual(task_after_run.status, TaskStatus.PAUSED)
        self.assertIsNotNone(task_after_run.plan.subtasks[0].latest_checkpoint_id)

        checkpoint_id = task_after_run.plan.subtasks[0].latest_checkpoint_id
        loaded_checkpoint = self.storage.load_checkpoint(checkpoint_id)
        self.assertEqual(loaded_checkpoint.task_id, report.task_id)
        self.assertIn("After code generation", loaded_checkpoint.current_state_description)
        self.assertIn("Paused due to provider error", loaded_checkpoint.continuation_context["current_progress_description"])
        self.assertIn("Continue the current subtask", loaded_checkpoint.next_recommended_action)
        self.assertIn("task_objective", loaded_checkpoint.continuation_context)

    def test_quota_exhaustion_leads_to_paused_state_and_resume(self):
        task_objective = "Implement a new feature"
        provider1 = CapturingProvider(quota_exceeded_on_attempt=1) # Fail on first code gen attempt
        orchestrator1 = Orchestrator(self.config, provider1, self.storage)
        task = self._create_task_with_plan(task_objective)
        report1 = orchestrator1.run(task, "sub1")

        self.assertEqual(report1.outcome, "QUOTA_EXCEEDED")
        task_after_run1 = self.storage.load_task(report1.task_id)
        self.assertEqual(task_after_run1.status, TaskStatus.PAUSED)
        self.assertEqual(task_after_run1.plan.subtasks[0].status, SubtaskStatus.PAUSED)

        # Resume the task
        provider2 = CapturingProvider() # This provider will succeed
        orchestrator2 = Orchestrator(self.config, provider2, self.storage)
        report2 = orchestrator2.run(task_after_run1, "sub1")

        self.assertEqual(report2.task_id, report1.task_id)
        self.assertEqual(self.storage.load_task(report2.task_id).status, TaskStatus.COMPLETED)
        self.assertEqual(self.storage.load_task(report2.task_id).plan.subtasks[0].status, SubtaskStatus.COMPLETED)
        self.assertEqual(provider2.code_gen_attempts, 1) # Should have only one successful attempt after resume

    def test_temporary_provider_error_leads_to_paused_state(self):
        task_objective = "Implement a new feature"
        provider = CapturingProvider(fail_on_code_gen=True)
        orchestrator = Orchestrator(self.config, provider, self.storage)
        
        task = self._create_task_with_plan(task_objective)
        # Mock ProviderError to be temporary (e.g., RateLimitError)
        provider.generate_code = mock.Mock(side_effect=QuotaExceededError("quota fail"))
        report = orchestrator.run(task, "sub1")

        self.assertEqual(report.outcome, "QUOTA_EXCEEDED")
        task_after_run = self.storage.load_task(report.task_id)
        self.assertEqual(task_after_run.status, TaskStatus.PAUSED)
        self.assertEqual(task_after_run.plan.subtasks[0].status, SubtaskStatus.PAUSED)

    def test_permanent_failure_leads_to_failed_state(self):
        task_objective = "Implement a new feature"
        provider = CapturingProvider(fail_on_code_gen=True)
        orchestrator = Orchestrator(self.config, provider, self.storage)
        task = self._create_task_with_plan(task_objective)
        # Ensure it's a non-retryable ProviderError
        report = orchestrator.run(task, "sub1")

        self.assertEqual(report.outcome, "UNKNOWN_PROVIDER_ERROR")
        task_after_run = self.storage.load_task(report.task_id)
        self.assertEqual(task_after_run.status, TaskStatus.FAILED)
        self.assertEqual(task_after_run.plan.subtasks[0].status, SubtaskStatus.FAILED)

    def test_completed_task_persistence(self):
        task_objective = "Implement a new feature"
        orchestrator = Orchestrator(self.config, CapturingProvider(), self.storage)
        task = self._create_task_with_plan(task_objective)
        report = orchestrator.run(task, "sub1")
        task = self.storage.load_task(report.task_id)

        self.assertEqual(task.status, TaskStatus.COMPLETED)
        self.assertEqual(task.plan.subtasks[0].status, SubtaskStatus.COMPLETED)

    def test_no_api_secrets_stored_in_checkpoints(self):
        task_objective = "Implement a new feature"
        provider = CapturingProvider(quota_exceeded_on_attempt=1)
        orchestrator = Orchestrator(self.config, provider, self.storage)
        task = self._create_task_with_plan(task_objective)
        report = orchestrator.run(task, "sub1")

        checkpoint_id = self.storage.load_task(report.task_id).latest_checkpoint_id
        loaded_checkpoint = self.storage.load_checkpoint(checkpoint_id)
        
        # Check for common secret patterns in the serialized checkpoint
        serialized_checkpoint = json.dumps(loaded_checkpoint.to_dict())
        self.assertNotIn("API_KEY", serialized_checkpoint)
        self.assertNotIn("GEMINI_API_KEY", serialized_checkpoint)
        self.assertNotIn("OPENAI_API_KEY", serialized_checkpoint)
        self.assertNotIn("secret", serialized_checkpoint.lower())

    def test_provider_agnostic_continuation_context(self):
        task_objective = "Implement a new feature"
        provider = CapturingProvider(quota_exceeded_on_attempt=1)
        orchestrator = Orchestrator(self.config, provider, self.storage)
        task = self._create_task_with_plan(task_objective)
        report = orchestrator.run(task, "sub1")

        checkpoint_id = self.storage.load_task(report.task_id).plan.subtasks[0].latest_checkpoint_id
        loaded_checkpoint = self.storage.load_checkpoint(checkpoint_id)
        
        continuation_context = loaded_checkpoint.continuation_context
        self.assertIn("task_objective", continuation_context)
        self.assertIn("current_subtask_goal", continuation_context)
        self.assertIn("completed_subtasks_summary", continuation_context)
        self.assertIn("current_progress_description", continuation_context)
        self.assertIn("modified_files_summary", continuation_context)
        self.assertIn("repository_diff", continuation_context)
        self.assertIn("validation_state", continuation_context)
        self.assertIn("last_provider_event", continuation_context)
        self.assertIn("next_recommended_action", continuation_context)
        
        # Ensure no provider-specific details like model names or API keys
        self.assertNotIn("gemini", json.dumps(continuation_context).lower())
        self.assertNotIn("openai", json.dumps(continuation_context).lower())