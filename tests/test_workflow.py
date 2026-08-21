from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import threading
from unittest import mock

from local_agent.config import AgentConfig
from local_agent.orchestrator import Orchestrator
from local_agent.storage import JsonFileStorage


class WorkflowTests(unittest.TestCase):
    def test_iteration_limit_is_bounded_when_validation_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = AgentConfig.from_environment(root, provider="mock", max_iterations=2, validation_commands=["python -c \"import sys; sys.exit(1)\""])
            storage = JsonFileStorage(root / ".agent_data")
            orchestrator = Orchestrator(config, storage, None, threading.Lock(), threading.Lock())
            report = orchestrator.run("Fix the failing task")
            self.assertEqual(report.iterations, 2)
            self.assertEqual(len(report.failures), 2)
            self.assertFalse(report.completed)

    def test_mock_workflow_reports_incomplete_instead_of_faking_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = AgentConfig.from_environment(root, provider="mock", max_iterations=1)
            storage = JsonFileStorage(root / ".agent_data")
            orchestrator = Orchestrator(config, storage, None, threading.Lock(), threading.Lock())
            report = orchestrator.run("Implement a feature")
            self.assertFalse(report.completed)
            self.assertEqual(report.changed_files, [])
            self.assertEqual(report.review.verdict, "CHANGES_REQUIRED")


if __name__ == "__main__":
    unittest.main()
