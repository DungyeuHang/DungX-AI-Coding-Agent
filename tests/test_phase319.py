from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_agent.approval import ApprovalPolicyEngine
from local_agent.config import AgentConfig
from local_agent.models import (
    ApprovalPolicy,
    ChangeImpact,
    ChangeTarget,
    FileOperation,
    PreparedChange,
    ReviewResult,
)
from local_agent.orchestrator import Orchestrator
from local_agent.providers import MockProvider
from local_agent.storage import TaskStorage


class MockStorage(TaskStorage):
    def save_task(self, task): pass
    def load_task(self, task_id): pass
    def list_tasks(self): return []
    def save_checkpoint(self, checkpoint): pass
    def load_checkpoint(self, checkpoint_id): pass
    def save_scheduler_state(self, state): pass
    def load_scheduler_state(self): from local_agent.models import SchedulerState; return SchedulerState()
    def save_provider_configs(self, configs): pass
    def load_provider_configs(self): return []
    def save_semantic_index(self, index): pass
    def load_semantic_index(self): from local_agent.models import SemanticIndex; return SemanticIndex()
    def save_project_memory(self, memory): pass
    def load_project_memory(self): from local_agent.models import ProjectMemory; return ProjectMemory()


class Phase319_ApprovalPolicyTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.mock_storage = MockStorage()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root)

    def test_auto_approve_test_only_change(self):
        policies = [
            ApprovalPolicy.from_dict({
                "name": "Auto-approve test changes",
                "action": "auto_approve",
                "if_path_matches": ["tests/**", "*.test.py"],
                "if_path_does_not_match": ["src/**"],
                "if_max_lines_changed": 50,
                "if_risk_is_at_most": "low",
            })
        ]
        engine = ApprovalPolicyEngine(policies)
        changes = [PreparedChange("modify", "tests/test_app.py", None, None, "...\n" * 10, "reason")]
        impact = ChangeImpact("summary", [ChangeTarget("tests/test_app.py", "test", 0.9, "reason", risk="low")])
        self.assertFalse(engine.is_manual_approval_required(changes, impact))

    def test_require_approval_for_prod_change(self):
        policies = [ApprovalPolicy.from_dict({"name": "Auto-approve test changes", "action": "auto_approve", "if_path_matches": ["tests/**"]})]
        engine = ApprovalPolicyEngine(policies)
        changes = [PreparedChange("modify", "src/app.py", None, None, "...", "reason")]
        impact = ChangeImpact("summary", [ChangeTarget("src/app.py", "modify", 0.9, "reason", risk="medium")])
        self.assertTrue(engine.is_manual_approval_required(changes, impact))

    def test_require_approval_on_explicit_policy(self):
        policies = [ApprovalPolicy.from_dict({"name": "Require approval for critical files", "action": "require_approval", "if_path_matches": ["src/critical.py"]})]
        engine = ApprovalPolicyEngine(policies)
        changes = [PreparedChange("modify", "src/critical.py", None, None, "...", "reason")]
        impact = ChangeImpact("summary", [])
        self.assertTrue(engine.is_manual_approval_required(changes, impact))

    def test_mixed_change_requires_approval(self):
        policies = [ApprovalPolicy.from_dict({"name": "Auto-approve test changes", "action": "auto_approve", "if_path_matches": ["tests/**"], "if_path_does_not_match": ["src/**"]})]
        engine = ApprovalPolicyEngine(policies)
        changes = [
            PreparedChange("modify", "tests/test_app.py", None, None, "...", "reason"),
            PreparedChange("modify", "src/app.py", None, None, "...", "reason"),
        ]
        impact = ChangeImpact("summary", [])
        self.assertTrue(engine.is_manual_approval_required(changes, impact))

    def test_change_size_limit(self):
        policies = [ApprovalPolicy.from_dict({"name": "Auto-approve small changes", "action": "auto_approve", "if_max_lines_changed": 10})]
        engine = ApprovalPolicyEngine(policies)
        small_changes = [PreparedChange("modify", "a.py", None, None, "...\n" * 5, "reason")]
        large_changes = [PreparedChange("modify", "a.py", None, None, "...\n" * 15, "reason")]
        impact = ChangeImpact("summary", [])
        self.assertFalse(engine.is_manual_approval_required(small_changes, impact))
        self.assertTrue(engine.is_manual_approval_required(large_changes, impact))

    def test_risk_level_limit(self):
        policies = [ApprovalPolicy.from_dict({"name": "Auto-approve low risk", "action": "auto_approve", "if_risk_is_at_most": "low"})]
        engine = ApprovalPolicyEngine(policies)
        low_risk_impact = ChangeImpact("summary", [ChangeTarget("a.py", "modify", 0.9, "r", risk="low")])
        medium_risk_impact = ChangeImpact("summary", [ChangeTarget("a.py", "modify", 0.9, "r", risk="medium")])
        changes = [PreparedChange("modify", "a.py", None, None, "...", "reason")]
        self.assertFalse(engine.is_manual_approval_required(changes, low_risk_impact))
        self.assertTrue(engine.is_manual_approval_required(changes, medium_risk_impact))

    @mock.patch("local_agent.orchestrator.CodingAgent")
    def test_orchestrator_integration_auto_approves(self, MockCodingAgent):
        policies = [{"name": "Auto-approve all", "action": "auto_approve"}]
        config = AgentConfig.from_environment(self.root, approval="policy", approval_policies=policies)
        
        provider = MockProvider()
        provider.generate_code = mock.Mock(return_value=[FileOperation("modify", "a.py", content="new")])
        mock_coding_agent_instance = MockCodingAgent.return_value
        prepared_changes = [PreparedChange("modify", "a.py", "old", "new", "diff")]
        mock_coding_agent_instance.prepare.return_value = prepared_changes
        
        mock_scheduler = mock.MagicMock()
        mock_scheduler.registry.providers = {}
        mock_scheduler._select_providers.return_value = [config]
        mock_scheduler._build_provider_instance.return_value = provider
        import threading
        orchestrator = Orchestrator(config, self.mock_storage, mock_scheduler, threading.Lock(), threading.Lock())
        approval_callback = mock.Mock(return_value=True)
        
        with mock.patch.object(orchestrator, '_validate', return_value=[]), \
             mock.patch('local_agent.orchestrator.Reviewer') as MockReviewer:
            MockReviewer.return_value.review.return_value = ReviewResult("APPROVED", "LGTM", [])
            orchestrator.run(mock.MagicMock(objective="test", plan=mock.MagicMock(subtasks=[mock.MagicMock(subtask_id="sub1")])), subtask_id="sub1", approval_callback=approval_callback)

        approval_callback.assert_not_called()
        mock_coding_agent_instance.apply_prepared.assert_called_with(prepared_changes)

    @mock.patch("local_agent.orchestrator.CodingAgent")
    def test_orchestrator_integration_requires_approval(self, MockCodingAgent):
        policies = [{"name": "Require approval for src", "action": "require_approval", "if_path_matches": ["src/**"]}]
        config = AgentConfig.from_environment(self.root, approval="policy", approval_policies=policies)
        
        provider = MockProvider()
        provider.generate_code = mock.Mock(return_value=[FileOperation("modify", "src/a.py", content="new")])
        mock_coding_agent_instance = MockCodingAgent.return_value
        prepared_changes = [PreparedChange("modify", "src/a.py", "old", "new", "diff")]
        mock_coding_agent_instance.prepare.return_value = prepared_changes
        
        mock_scheduler = mock.MagicMock()
        mock_scheduler.registry.providers = {}
        mock_scheduler._select_providers.return_value = [config]
        mock_scheduler._build_provider_instance.return_value = provider
        import threading
        orchestrator = Orchestrator(config, self.mock_storage, mock_scheduler, threading.Lock(), threading.Lock())
        approval_callback = mock.Mock(return_value=False)
        
        with mock.patch.object(orchestrator, '_validate', return_value=[]):
            report = orchestrator.run(mock.MagicMock(objective="test", plan=mock.MagicMock(subtasks=[mock.MagicMock(subtask_id="sub1")])), subtask_id="sub1", approval_callback=approval_callback)

        approval_callback.assert_called_once_with(prepared_changes)
        mock_coding_agent_instance.apply_prepared.assert_not_called()
        self.assertTrue(report.approval_required)
        self.assertFalse(report.completed)