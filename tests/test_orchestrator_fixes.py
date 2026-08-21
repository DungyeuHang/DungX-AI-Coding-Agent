from __future__ import annotations

import datetime
import inspect
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock

from local_agent.config import AgentConfig
from local_agent.models import (
    CIFailureContext,
    Plan,
    Subtask,
    Task,
    TaskPlan,
    TaskStatus,
)
from local_agent.orchestrator import Orchestrator
from local_agent.storage import JsonFileStorage


class OrchestratorFixesTests(unittest.TestCase):
    def test_orchestrator_can_be_constructed(self):
        config = AgentConfig(project=Path("."))
        storage = Mock()
        scheduler = Mock()
        repo_lock = threading.Lock()
        memory_lock = threading.Lock()
        orchestrator = Orchestrator(config, storage, scheduler, repo_lock, memory_lock)
        self.assertIsInstance(orchestrator, Orchestrator)
        self.assertEqual(orchestrator.config, config)
        self.assertEqual(orchestrator.storage, storage)
        self.assertEqual(orchestrator.scheduler, scheduler)
        self.assertEqual(orchestrator.repo_lock, repo_lock)
        self.assertEqual(orchestrator.memory_lock, memory_lock)

    def test_ci_failure_seeds_on_first_iteration_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            storage = JsonFileStorage(root / ".agent")
            config = AgentConfig(project=root, max_iterations=2)
            scheduler = Mock()
            repo_lock = threading.Lock()
            memory_lock = threading.Lock()
            orch = Orchestrator(config, storage, scheduler, repo_lock, memory_lock)

            now = datetime.datetime.now(datetime.timezone.utc)
            ci_context = CIFailureContext(
                failed_command="pytest",
                exit_code=1,
                stdout="out",
                stderr="err",
            )
            task = Task(
                task_id="task-1",
                objective="fix bug",
                status=TaskStatus.PENDING,
                created_at=now,
                updated_at=now,
                initial_failure_context=ci_context,
            )
            subtask = Subtask(
                subtask_id="st-1",
                title="repair",
                description="desc",
                dependencies=[],
            )
            task.plan = TaskPlan(
                objective="fix bug",
                subtasks=[subtask],
            )
            storage.save_task(task)

            mock_provider = Mock()
            mock_provider.create_subtask_plan.return_value = Plan(
                objective="fix bug",
                files_to_inspect=[],
                files_likely_to_change=[],
                files_likely_to_create=[],
                steps=[],
                validation_strategy=[],
                risks=[],
            )

            failures_passed = []

            def fake_generate_code(task_arg, plan_arg, context_arg, failure=None, review=None):
                failures_passed.append(failure)
                return []

            mock_provider.generate_code.side_effect = fake_generate_code

            scheduler._select_providers.return_value = [Mock(provider="mock")]
            scheduler._build_provider_instance.return_value = mock_provider
            scheduler.registry.providers = {}

            orch.run(task, "st-1")

            self.assertIsNone(task.initial_failure_context)
            self.assertTrue(len(failures_passed) >= 1)
            self.assertIsNotNone(failures_passed[0])
            self.assertEqual(
                failures_passed[0].probable_root_cause,
                "Initial failure provided by CI environment.",
            )

    def test_no_duplicate_git_changed_paths_method(self):
        source = inspect.getsource(Orchestrator)
        count = source.count("def _git_changed_paths")
        self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()

