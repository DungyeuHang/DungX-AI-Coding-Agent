from __future__ import annotations

import datetime
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from local_agent.config import AgentConfig
from local_agent.filesystem import ProjectFilesystem
from local_agent.models import (
    ApprovalPolicy,
    Checkpoint,
    ExecutionResult,
    FailureAnalysis,
    FileOperation,
    Plan,
    PreflightCheckCategory,
    PreflightCheckItem,
    PreflightCheckStatus,
    PreflightReport,
    ProviderCapability,
    ProviderError,
    ProviderHealthStatus,
    RecoveryState,
    RepairSignature,
    RoleHealthReport,
    RunReport,
    SpecialistRole,
    Subtask,
    SubtaskStatus,
    Task,
    TaskPlan,
    TaskStatus,
)
from local_agent.orchestrator import Orchestrator
from local_agent.preflight import PreflightChecker
from local_agent.providers import MockProvider, OpenAIProvider, SpecialistModelRouter
from local_agent.storage import JsonFileStorage


class TestPhase425ProviderDegradationVisibility(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_explicit_mock_provider_health_status(self):
        config = AgentConfig(project=self.project_path, provider="mock")
        router = SpecialistModelRouter(config)

        health = router.get_role_health(SpecialistRole.IMPLEMENTATION)
        self.assertEqual(health.status, ProviderHealthStatus.EXPLICIT_OFFLINE_MOCK.value)
        self.assertEqual(health.active_provider, "mock")
        self.assertFalse(health.is_real_provider)
        self.assertIsNone(health.degradation_reason)

        all_health = router.get_all_roles_health()
        for role_name, rhealth in all_health.items():
            self.assertEqual(rhealth.status, ProviderHealthStatus.EXPLICIT_OFFLINE_MOCK.value)
        self.assertFalse(router.has_any_degradation())

    def test_degraded_fallback_when_real_provider_fails(self):
        # Configure a real provider with invalid / missing keys
        config = AgentConfig(
            project=self.project_path,
            provider="openai",
            api_key=None,
        )
        # Clear env to guarantee OpenAIProvider cannot initialize
        with patch.dict(os.environ, {}, clear=True):
            router = SpecialistModelRouter(config)
            health = router.get_role_health(SpecialistRole.IMPLEMENTATION)
            self.assertEqual(health.status, ProviderHealthStatus.DEGRADED_FALLBACK.value)
            self.assertEqual(health.configured_provider, "openai")
            self.assertEqual(health.active_provider, "mock")
            self.assertIsNotNone(health.degradation_reason)
            self.assertTrue(router.is_role_degraded(SpecialistRole.IMPLEMENTATION))
            self.assertTrue(router.has_any_degradation())

            chain = router.get_provider_chain(SpecialistRole.IMPLEMENTATION)
            self.assertEqual(len(chain), 1)
            self.assertIsInstance(chain[0], MockProvider)
            self.assertTrue(getattr(chain[0], "_is_degraded_fallback", False))

    def test_role_health_report_serialization(self):
        report = RoleHealthReport(
            role="coder",
            status=ProviderHealthStatus.HEALTHY_REAL_PROVIDER.value,
            configured_provider="openai",
            configured_model="gpt-4o",
            active_provider="openai",
            active_model="gpt-4o",
            fallback_chain=["gemini"],
            degradation_reason=None,
            is_real_provider=True,
        )
        d = report.to_dict()
        reconstructed = RoleHealthReport.from_dict(d)
        self.assertEqual(reconstructed.role, "coder")
        self.assertEqual(reconstructed.status, ProviderHealthStatus.HEALTHY_REAL_PROVIDER.value)
        self.assertTrue(reconstructed.is_real_provider)


class TestPhase425ProductionPreflightValidator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_path = Path(self.temp_dir.name)
        self.storage_path = self.project_path / ".agent_data"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.storage = JsonFileStorage(self.storage_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_preflight_passes_for_clean_mock_environment(self):
        config = AgentConfig(
            project=self.project_path,
            provider="mock",
            max_iterations=5,
            command_timeout_seconds=30,
        )
        checker = PreflightChecker(config, storage=self.storage)
        report = checker.check(require_real_providers=False)

        self.assertTrue(report.is_ready)
        self.assertIn(report.overall_status, ("READY", "READY_WITH_WARNINGS"))
        self.assertEqual(len(report.roles_health), len(SpecialistRole))

    def test_preflight_blocks_when_real_providers_required_but_mock_configured(self):
        config = AgentConfig(
            project=self.project_path,
            provider="mock",
        )
        checker = PreflightChecker(config, storage=self.storage)
        report = checker.check(require_real_providers=True)

        self.assertFalse(report.is_ready)
        self.assertEqual(report.overall_status, "BLOCKED")
        self.assertIn("real providers are required", report.blocker_summary)

    def test_preflight_blocks_on_invalid_repository_path(self):
        invalid_path = self.project_path / "non_existent_subdir_12345"
        config = AgentConfig(
            project=invalid_path,
            provider="mock",
        )
        checker = PreflightChecker(config, storage=self.storage)
        report = checker.check(require_real_providers=False)

        self.assertFalse(report.is_ready)
        self.assertEqual(report.overall_status, "BLOCKED")
        self.assertIn("Project path does not exist", report.blocker_summary)

    def test_preflight_report_serialization(self):
        item = PreflightCheckItem(
            name="TEST_CHECK",
            category=PreflightCheckCategory.PROVIDERS.value,
            status=PreflightCheckStatus.PASSED.value,
            message="All good",
        )
        report = PreflightReport(
            overall_status="READY",
            is_ready=True,
            checks=[item],
            roles_health={"planning": {"status": "healthy_real_provider"}},
            summary="1 passed",
        )
        d = report.to_dict()
        reconstructed = PreflightReport.from_dict(d)
        self.assertTrue(reconstructed.is_ready)
        self.assertEqual(len(reconstructed.checks), 1)
        self.assertEqual(reconstructed.checks[0].name, "TEST_CHECK")


class TestPhase425OrchestratorPreflightAndCheckpointResilience(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_path = Path(self.temp_dir.name)
        self.storage_path = self.project_path / ".agent_data"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.storage = JsonFileStorage(self.storage_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_orchestrator_blocks_when_preflight_fails(self):
        config = AgentConfig(project=self.project_path, provider="mock", require_real_providers=True)
        orch = Orchestrator(config, self.storage, None, MagicMock(), MagicMock())

        report = orch.run("Implement feature X")
        self.assertFalse(report.completed)
        self.assertIn("PREFLIGHT_BLOCKED", report.outcome)
        self.assertIsNotNone(report.preflight_report)
        self.assertFalse(report.preflight_report["is_ready"])

    def test_orchestrator_scenario_a_resume_preserves_planning_state(self):
        # Scenario A: Process stopped after planning; resume preserves plan
        config = AgentConfig(project=self.project_path, provider="mock", max_iterations=2)
        orch = Orchestrator(config, self.storage, None, MagicMock(), MagicMock())

        task = orch._create_new_task("Build authentication module")
        plan = Plan(objective="Build authentication module", steps=["step 1", "step 2"])
        task.plan = plan

        cp = Checkpoint(
            checkpoint_id="cp_plan_test",
            task_id=task.task_id,
            subtask_id="main",
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            current_state_description="Plan created",
            files_changed=[],
            continuation_context={"plan": plan.to_dict()},
        )
        self.storage.save_checkpoint(cp)
        task.latest_checkpoint_id = "cp_plan_test"
        self.storage.save_task(task)

        report = orch.run(task)
        self.assertIsNotNone(report.plan)
        self.assertEqual(report.plan.objective, "Build authentication module")

    def test_orchestrator_scenario_b_resume_preserves_partial_implementation(self):
        # Scenario B: Partial implementation changed files preserved
        config = AgentConfig(project=self.project_path, provider="mock", max_iterations=2)
        orch = Orchestrator(config, self.storage, None, MagicMock(), MagicMock())

        task = orch._create_new_task("Build auth")
        cp = Checkpoint(
            checkpoint_id="cp_partial_impl",
            task_id=task.task_id,
            subtask_id="main",
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            current_state_description="Partial implementation applied",
            files_changed=["auth.py"],
            continuation_context={
                "recovery_state": RecoveryState(completed_iterations=1).to_dict(),
            },
        )
        self.storage.save_checkpoint(cp)
        task.latest_checkpoint_id = "cp_partial_impl"

        report = orch.run(task)
        self.assertIsNotNone(report.recovery_state)
        self.assertEqual(report.iterations, 2)

    def test_orchestrator_scenario_c_resume_preserves_failed_verification_repair_state(self):
        # Scenario C: Process stopped after failed verification
        config = AgentConfig(project=self.project_path, provider="mock", max_iterations=2)
        orch = Orchestrator(config, self.storage, None, MagicMock(), MagicMock())

        task = orch._create_new_task("Build auth")
        failure_analysis = FailureAnalysis(
            probable_root_cause="SyntaxError on line 12",
            recommended_fix="Fix missing colon",
        )
        cp = Checkpoint(
            checkpoint_id="cp_failed_verif",
            task_id=task.task_id,
            subtask_id="main",
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            current_state_description="Verification failed",
            files_changed=[],
            continuation_context={
                "validation_state": {"last_failures": [failure_analysis.to_dict()]},
                "recovery_state": RecoveryState(consecutive_same_failure_count=1).to_dict(),
            },
        )
        self.storage.save_checkpoint(cp)
        task.latest_checkpoint_id = "cp_failed_verif"

        report = orch.run(task)
        self.assertIsNotNone(report.recovery_state)
        self.assertEqual(report.recovery_state.consecutive_same_failure_count, 1)

    def test_orchestrator_scenario_e_foreign_checkpoint_rejection(self):
        # Scenario E: Checkpoint belonging to another subtask does not contaminate
        config = AgentConfig(project=self.project_path, provider="mock", max_iterations=2)
        orch = Orchestrator(config, self.storage, None, MagicMock(), MagicMock())

        task = orch._create_new_task("Multi-subtask task")
        sub1 = Subtask(subtask_id="sub1", title="Subtask 1", goal="Goal 1")
        sub2 = Subtask(subtask_id="sub2", title="Subtask 2", goal="Goal 2")
        task.plan = TaskPlan(objective="Task", subtasks=[sub1, sub2])

        foreign_cp = Checkpoint(
            checkpoint_id="cp_sub1_only",
            task_id=task.task_id,
            subtask_id="sub1",
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            current_state_description="Subtask 1 state",
            files_changed=["sub1_file.py"],
            continuation_context={"recovery_state": RecoveryState(completed_iterations=5).to_dict()},
        )
        self.storage.save_checkpoint(foreign_cp)
        task.latest_checkpoint_id = "cp_sub1_only"

        # Run sub2 -> must NOT restore sub1's checkpoint
        report = orch.run(task, subtask_id="sub2")
        self.assertNotEqual(report.recovery_state.completed_iterations, 5)

    def test_orchestrator_scenario_d_resume_preserves_repair_anti_repeat_state(self):
        # Scenario D: Process stopped after repair; anti-repeat state survives
        config = AgentConfig(project=self.project_path, provider="mock", max_iterations=2)
        orch = Orchestrator(config, self.storage, None, MagicMock(), MagicMock())

        task = orch._create_new_task("Repair feature")
        sig = RepairSignature(
            iteration=1,
            failure_category="syntax_error",
            root_cause_hash="root_cause_1",
            patch_hash="patch_sig_12345",
            affected_files=["fix.py"],
        )
        rs = RecoveryState(
            consecutive_same_failure_count=2,
            repair_signatures=[sig],
        )
        cp = Checkpoint(
            checkpoint_id="cp_repair_state",
            task_id=task.task_id,
            subtask_id="main",
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            current_state_description="Repair executed",
            files_changed=["fix.py"],
            continuation_context={"recovery_state": rs.to_dict()},
        )
        self.storage.save_checkpoint(cp)
        task.latest_checkpoint_id = "cp_repair_state"

        report = orch.run(task)
        self.assertIsNotNone(report.recovery_state)
        self.assertTrue(any(s.patch_hash == "patch_sig_12345" for s in report.recovery_state.repair_signatures))

    def test_orchestrator_scenario_f_corrupted_checkpoint_fails_safe(self):
        # Scenario F: Corrupted JSON file in checkpoint store does not crash orchestrator
        config = AgentConfig(project=self.project_path, provider="mock", max_iterations=1)
        orch = Orchestrator(config, self.storage, None, MagicMock(), MagicMock())

        task = orch._create_new_task("Handle corrupt checkpoint")
        corrupt_cp_file = self.storage.checkpoints_dir / "corrupt_cp.json"
        corrupt_cp_file.write_text("{ invalid json corrupt! ", encoding="utf-8")
        task.latest_checkpoint_id = "corrupt_cp"

        # Should execute safely without unhandled JSONDecodeError / ValueError crash
        report = orch.run(task)
        self.assertIsNotNone(report)

    def test_orchestrator_scenario_g_disk_state_precedence_on_resume(self):
        # Scenario G: Repository changed between checkpoint and resume
        config = AgentConfig(project=self.project_path, provider="mock", max_iterations=1)
        orch = Orchestrator(config, self.storage, None, MagicMock(), MagicMock())

        task = orch._create_new_task("Disk state test")
        target_file = self.project_path / "module.py"
        target_file.write_text("def hello(): return 'v1'\n", encoding="utf-8")

        # Checkpoint recorded file changed
        cp = Checkpoint(
            checkpoint_id="cp_disk_test",
            task_id=task.task_id,
            subtask_id="main",
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            current_state_description="v1 applied",
            files_changed=["module.py"],
            continuation_context={"recovery_state": RecoveryState(completed_iterations=0).to_dict()},
        )
        self.storage.save_checkpoint(cp)
        task.latest_checkpoint_id = "cp_disk_test"

        # External modification occurs before resume
        target_file.write_text("def hello(): return 'v2_modified'\n", encoding="utf-8")

        report = orch.run(task)
        self.assertIsNotNone(report)
        # Disk content on disk should be v2_modified
        self.assertIn("v2_modified", target_file.read_text(encoding="utf-8"))


class TestPhase425CLIPrefilght(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_cli_preflight_command(self):
        from local_agent.cli import build_parser, main
        parser = build_parser()
        args = parser.parse_args(["preflight", "--project", str(self.project_path), "--provider", "mock"])
        self.assertEqual(args.command, "preflight")
        self.assertEqual(args.provider, "mock")
        self.assertFalse(args.require_real_providers)


class TestPhase425SpecialistFallbackExecution(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_router_fallback_execution_on_transient_error(self):
        config = AgentConfig(
            project=self.project_path,
            provider="mock",
            implementation_provider="mock_primary",
            implementation_fallbacks=["mock_fallback"],
        )
        router = SpecialistModelRouter(config)

        call_count = [0]
        def flaky_action(provider):
            call_count[0] += 1
            if call_count[0] == 1:
                from local_agent.models import QuotaExceededError
                raise QuotaExceededError("Rate limit hit on primary provider")
            return "SUCCESS_FROM_FALLBACK"

        result = router.execute_with_fallback(
            SpecialistRole.IMPLEMENTATION,
            flaky_action,
            stage_name="test_stage",
        )
        self.assertEqual(result, "SUCCESS_FROM_FALLBACK")
        self.assertEqual(call_count[0], 2)


class TestPhase425RealProviderConfiguration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_path = Path(self.temp_dir.name)
        self.storage = JsonFileStorage(self.project_path / ".agent_data")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_anthropic_single_provider_production_configuration(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test-key-valid"}):
            config = AgentConfig.from_environment(self.project_path, provider="anthropic")
            self.assertEqual(config.provider, "anthropic")
            self.assertEqual(config.model, "claude-3-7-sonnet-20250219")
            self.assertEqual(config.api_key, "sk-ant-test-key-valid")

            router = SpecialistModelRouter(config)
            all_health = router.get_all_roles_health()
            for role in SpecialistRole:
                self.assertEqual(
                    all_health[role.value].status,
                    ProviderHealthStatus.HEALTHY_REAL_PROVIDER.value,
                    f"Role {role.value} should be healthy real provider",
                )

            checker = PreflightChecker(config, storage=self.storage, router=router)
            report = checker.check(require_real_providers=True)
            self.assertTrue(report.is_ready)
            self.assertEqual(report.overall_status, "READY")

    def test_openai_single_provider_production_configuration(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-openai-test-key-valid"}):
            config = AgentConfig.from_environment(self.project_path, provider="openai")
            self.assertEqual(config.provider, "openai")
            self.assertEqual(config.model, "gpt-4.1-mini")
            self.assertEqual(config.api_key, "sk-openai-test-key-valid")

            router = SpecialistModelRouter(config)
            all_health = router.get_all_roles_health()
            for role in SpecialistRole:
                self.assertEqual(
                    all_health[role.value].status,
                    ProviderHealthStatus.HEALTHY_REAL_PROVIDER.value,
                )

            checker = PreflightChecker(config, storage=self.storage, router=router)
            report = checker.check(require_real_providers=True)
            self.assertTrue(report.is_ready)
            self.assertEqual(report.overall_status, "READY")

    def test_heterogeneous_specialist_configuration(self):
        with patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "sk-ant-test-key-valid",
            "OPENAI_API_KEY": "sk-openai-test-key-valid",
        }):
            config = AgentConfig.from_environment(
                self.project_path,
                provider="anthropic",
                planning_provider="anthropic",
                planning_model="claude-3-7-sonnet-20250219",
                implementation_provider="openai",
                implementation_model="gpt-4.1-mini",
                repair_provider="anthropic",
                repair_model="claude-3-7-sonnet-20250219",
                review_provider="anthropic",
                review_model="claude-3-7-sonnet-20250219",
                verification_provider="openai",
                verification_model="gpt-4.1-mini",
            )
            router = SpecialistModelRouter(config)
            all_health = router.get_all_roles_health()
            self.assertEqual(all_health["planning"].active_provider, "anthropic")
            self.assertEqual(all_health["implementation"].active_provider, "openai")
            self.assertEqual(all_health["repair"].active_provider, "anthropic")
            self.assertEqual(all_health["review"].active_provider, "anthropic")
            self.assertEqual(all_health["verification"].active_provider, "openai")

            for r in all_health.values():
                self.assertEqual(r.status, ProviderHealthStatus.HEALTHY_REAL_PROVIDER.value)

            checker = PreflightChecker(config, storage=self.storage, router=router)
            report = checker.check(require_real_providers=True)
            self.assertTrue(report.is_ready)
            self.assertEqual(report.overall_status, "READY")


if __name__ == "__main__":
    unittest.main()
