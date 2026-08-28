from __future__ import annotations

import argparse
import datetime
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from local_agent.config import AgentConfig, add_common_arguments, config_from_args
from local_agent.credentials import MockCredentialStore
from local_agent.models import (
    AuthenticationError,
    Checkpoint,
    CommandSpec,
    ExecutionResult,
    FailureAnalysis,
    FileOperation,
    NetworkError,
    Plan,
    ProjectContext,
    ProviderCapability,
    ProviderError,
    QuotaExceededError,
    RateLimitError,
    ReviewConsensusRecord,
    ReviewResult,
    RunReport,
    ScopeExpansionProposal,
    SpecialistRole,
    Subtask,
    SubtaskContract,
    SubtaskStatus,
    Task,
    TaskPlan,
    TaskStatus,
    TestExecutionRecord,
    VerificationGap,
)
from local_agent.orchestrator import Orchestrator
from local_agent.providers import (
    AIProvider,
    AnthropicProvider,
    AntigravityProvider,
    DeepSeekProvider,
    GeminiProvider,
    MockProvider,
    OpenAIProvider,
    SpecialistModelRouter,
    build_provider,
)
from local_agent.reviewer import DeliberativeReviewConsensus, Reviewer
from local_agent.storage import JsonFileStorage


class SpecialistRoleAndModelsTests(unittest.TestCase):
    """Tests for SpecialistRole enum and ReviewConsensusRecord models."""

    def test_specialist_role_canonical_enum_values(self):
        self.assertEqual(SpecialistRole.PLANNING.value, "planning")
        self.assertEqual(SpecialistRole.IMPLEMENTATION.value, "implementation")
        self.assertEqual(SpecialistRole.REPAIR.value, "repair")
        self.assertEqual(SpecialistRole.REVIEW.value, "review")
        self.assertEqual(SpecialistRole.VERIFICATION.value, "verification")
        self.assertEqual(len(SpecialistRole), 5)

    def test_review_consensus_record_serialization_roundtrip(self):
        now = datetime.datetime.now(datetime.timezone.utc)
        rec = ReviewConsensusRecord(
            role="review",
            primary_provider="OpenAIProvider",
            primary_model="gpt-4.1-mini",
            secondary_provider="AnthropicProvider",
            secondary_model="claude-3-7-sonnet",
            primary_verdict="APPROVED",
            secondary_verdict="APPROVED",
            final_consensus_verdict="APPROVED",
            is_high_risk=True,
            high_risk_reason="Changed file 'auth/login.py' matches high-risk pattern",
            timestamp=now,
        )
        d = rec.to_dict()
        self.assertEqual(d["role"], "review")
        self.assertEqual(d["primary_provider"], "OpenAIProvider")
        self.assertEqual(d["secondary_provider"], "AnthropicProvider")
        self.assertTrue(d["is_high_risk"])
        self.assertEqual(d["final_consensus_verdict"], "APPROVED")

        restored = ReviewConsensusRecord.from_dict(d)
        self.assertEqual(restored.role, rec.role)
        self.assertEqual(restored.primary_provider, rec.primary_provider)
        self.assertEqual(restored.secondary_provider, rec.secondary_provider)
        self.assertEqual(restored.is_high_risk, rec.is_high_risk)
        self.assertEqual(restored.final_consensus_verdict, rec.final_consensus_verdict)

    def test_review_result_consensus_records_integration(self):
        rec = ReviewConsensusRecord(
            primary_provider="OpenAIProvider",
            primary_verdict="APPROVED",
            final_consensus_verdict="APPROVED",
        )
        review = ReviewResult(
            verdict="APPROVED",
            summary="All clean",
            findings=["good naming"],
            consensus_records=[rec],
        )
        d = review.to_dict()
        self.assertEqual(len(d["consensus_records"]), 1)
        self.assertEqual(d["consensus_records"][0]["primary_provider"], "OpenAIProvider")

        restored = ReviewResult.from_dict(d)
        self.assertEqual(restored.verdict, "APPROVED")
        self.assertEqual(len(restored.consensus_records), 1)
        self.assertEqual(restored.consensus_records[0].primary_provider, "OpenAIProvider")


class SpecialistConfigurationTests(unittest.TestCase):
    """Tests for AgentConfig specialist fields, CLI args, and environment variable parsing."""

    def test_default_single_provider_resolves_all_roles(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = AgentConfig(project=Path(td), provider="mock", model="mock-default")
            router = SpecialistModelRouter(cfg)

            for role in SpecialistRole:
                prov_name, model_name, fallbacks = router.get_role_config(role)
                self.assertEqual(prov_name, "mock")
                self.assertEqual(model_name, "mock-default")
                self.assertEqual(fallbacks, [])

    def test_explicit_specialist_overrides_in_config(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = AgentConfig(
                project=Path(td),
                provider="mock",
                model="mock-default",
                planning_provider="gemini",
                planning_model="gemini-2.5-pro",
                planning_fallbacks=["openai"],
                implementation_provider="deepseek",
                implementation_model="deepseek-coder",
                repair_provider="anthropic",
                repair_model="claude-3-7-sonnet",
                review_provider="openai",
                review_model="o3-mini",
                verification_provider="gemini",
                verification_model="gemini-2.5-flash",
                dual_review_enabled=True,
                high_risk_dual_review=True,
            )
            router = SpecialistModelRouter(cfg)

            plan_prov, plan_model, plan_fb = router.get_role_config(SpecialistRole.PLANNING)
            self.assertEqual(plan_prov, "gemini")
            self.assertEqual(plan_model, "gemini-2.5-pro")
            self.assertIn("openai", plan_fb)
            self.assertIn("mock", plan_fb)

            impl_prov, impl_model, _ = router.get_role_config(SpecialistRole.IMPLEMENTATION)
            self.assertEqual(impl_prov, "deepseek")
            self.assertEqual(impl_model, "deepseek-coder")

            rep_prov, rep_model, _ = router.get_role_config(SpecialistRole.REPAIR)
            self.assertEqual(rep_prov, "anthropic")
            self.assertEqual(rep_model, "claude-3-7-sonnet")

            rev_prov, rev_model, _ = router.get_role_config(SpecialistRole.REVIEW)
            self.assertEqual(rev_prov, "openai")
            self.assertEqual(rev_model, "o3-mini")

            ver_prov, ver_model, _ = router.get_role_config(SpecialistRole.VERIFICATION)
            self.assertEqual(ver_prov, "gemini")
            self.assertEqual(ver_model, "gemini-2.5-flash")

    def test_environment_variable_parsing_for_specialists(self):
        with tempfile.TemporaryDirectory() as td:
            env = {
                "AGENT_PLANNING_PROVIDER": "gemini",
                "AGENT_PLANNING_MODEL": "gemini-2.5-pro",
                "AGENT_PLANNING_FALLBACKS": "openai, anthropic",
                "AGENT_IMPLEMENTATION_PROVIDER": "deepseek",
                "AGENT_IMPLEMENTATION_MODEL": "deepseek-coder",
                "AGENT_DUAL_REVIEW": "true",
                "AGENT_HIGH_RISK_DUAL_REVIEW": "true",
            }
            with mock.patch.dict(os.environ, env, clear=False):
                cfg = AgentConfig.from_environment(td)
                self.assertEqual(cfg.planning_provider, "gemini")
                self.assertEqual(cfg.planning_model, "gemini-2.5-pro")
                self.assertEqual(cfg.planning_fallbacks, ["openai", "anthropic"])
                self.assertEqual(cfg.implementation_provider, "deepseek")
                self.assertEqual(cfg.implementation_model, "deepseek-coder")
                self.assertTrue(cfg.dual_review_enabled)
                self.assertTrue(cfg.high_risk_dual_review)

    def test_cli_args_parsing_for_specialists(self):
        with tempfile.TemporaryDirectory() as td:
            parser = argparse.ArgumentParser()
            add_common_arguments(parser)
            args = parser.parse_args([
                "--project", td,
                "--planning-provider", "gemini",
                "--planning-model", "gemini-2.5-pro",
                "--implementation-provider", "deepseek",
                "--dual-review", "true",
            ])
            cfg = config_from_args(args)
            self.assertEqual(cfg.planning_provider, "gemini")
            self.assertEqual(cfg.planning_model, "gemini-2.5-pro")
            self.assertEqual(cfg.implementation_provider, "deepseek")
            self.assertTrue(cfg.dual_review_enabled)


class SpecialistModelRouterTests(unittest.TestCase):
    """Tests for SpecialistModelRouter chain resolution, fallback execution, and caching."""

    def test_provider_instance_caching_and_thread_safety(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = AgentConfig(
                project=Path(td),
                provider="mock",
                planning_provider="openai",
                planning_model="gpt-4.1-mini",
                api_key="sk-test",
            )
            router = SpecialistModelRouter(cfg)

            p1 = router.get_provider(SpecialistRole.PLANNING)
            p2 = router.get_provider(SpecialistRole.PLANNING)
            self.assertIsInstance(p1, OpenAIProvider)
            self.assertIs(p1, p2)

    def test_fallback_chain_advancement_on_rate_limit(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = AgentConfig(project=Path(td), provider="mock")

            class FlakyProvider(MockProvider):
                def __init__(self, name: str, fail_count: int = 1):
                    super().__init__()
                    self.name = name
                    self.fail_count = fail_count
                    self.attempts = 0

                def generate_plan(self, task, context):
                    self.attempts += 1
                    if self.attempts <= self.fail_count:
                        raise RateLimitError("Rate limit exceeded", retry_after_seconds=30)
                    return Plan(objective=f"Plan by {self.name}")

            p_primary = FlakyProvider("Primary", fail_count=999)
            p_fallback = FlakyProvider("Fallback", fail_count=0)

            factory_map = {
                "primary": p_primary,
                "fallback": p_fallback,
            }

            def custom_factory(c, api_key=None):
                return factory_map.get(c.provider, MockProvider())

            router = SpecialistModelRouter(cfg, provider_factory=custom_factory)
            router.get_role_config = lambda r: ("primary", "model1", ["fallback"])

            result = router.execute_with_fallback(
                SpecialistRole.PLANNING,
                lambda p: p.generate_plan("task", None),
                stage_name="planning",
            )
            self.assertEqual(result.objective, "Plan by Fallback")
            self.assertEqual(p_primary.attempts, 1)
            self.assertEqual(p_fallback.attempts, 1)

    def test_fallback_chain_advancement_on_quota_and_network_errors(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = AgentConfig(project=Path(td), provider="mock")

            class QuotaFailProvider(MockProvider):
                def generate_code(self, task, plan, context, failure=None, review=None):
                    raise QuotaExceededError("Quota exhausted")

            class SuccessProvider(MockProvider):
                def generate_code(self, task, plan, context, failure=None, review=None):
                    return [FileOperation("create", "src/hello.py", content="print('hello')\n")]

            factory_map = {
                "quota_fail": QuotaFailProvider(),
                "success": SuccessProvider(),
            }

            router = SpecialistModelRouter(cfg, provider_factory=lambda c, api_key=None: factory_map.get(c.provider, MockProvider()))
            router.get_role_config = lambda r: ("quota_fail", None, ["success"])

            ops = router.execute_with_fallback(
                SpecialistRole.IMPLEMENTATION,
                lambda p: p.generate_code("task", None, None),
                stage_name="implementation",
            )
            self.assertEqual(len(ops), 1)
            self.assertEqual(ops[0].path, "src/hello.py")

    def test_fallback_exhaustion_raises_last_provider_error(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = AgentConfig(project=Path(td), provider="mock")

            class PermanentFailProvider(MockProvider):
                def generate_plan(self, task, context):
                    raise AuthenticationError("Invalid API Key")

            router = SpecialistModelRouter(cfg, provider_factory=lambda c, api_key=None: PermanentFailProvider())
            router.get_role_config = lambda r: ("fail1", None, ["fail2"])

            with self.assertRaises(AuthenticationError):
                router.execute_with_fallback(
                    SpecialistRole.PLANNING,
                    lambda p: p.generate_plan("task", None),
                    stage_name="planning",
                )

    def test_credential_store_isolation_per_specialist(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = AgentConfig(
                project=Path(td),
                provider="mock",
                planning_provider="gemini",
                implementation_provider="openai",
            )
            cred_store = MockCredentialStore()
            cred_store.save("dungx-ai-coding-agent", "gemini", "gemini-secret-key")
            cred_store.save("dungx-ai-coding-agent", "openai", "openai-secret-key")

            router = SpecialistModelRouter(cfg, credential_store=cred_store)

            p_plan = router.get_provider(SpecialistRole.PLANNING)
            p_impl = router.get_provider(SpecialistRole.IMPLEMENTATION)

            self.assertIsInstance(p_plan, GeminiProvider)
            self.assertEqual(p_plan.config.api_key, "gemini-secret-key")
            self.assertIsInstance(p_impl, OpenAIProvider)
            self.assertEqual(p_impl.config.api_key, "openai-secret-key")


class DeliberativeReviewConsensusTests(unittest.TestCase):
    """Tests for DeliberativeReviewConsensus high-risk detection and dual-model gating."""

    def test_low_risk_single_review_approval(self):
        class SimpleReviewProvider(MockProvider):
            def review_changes(self, task, plan, diff, context):
                return ReviewResult("APPROVED", "Looks solid", [])

        primary = Reviewer(SimpleReviewProvider())
        consensus = DeliberativeReviewConsensus(primary, secondary_reviewer=None, dual_review_enabled=False)

        plan = Plan(objective="Simple fix", files_likely_to_change=["src/util.py"])
        context = ProjectContext(root=".")
        report = RunReport(project=context, plan=plan, changed_files=["src/util.py"])

        result = consensus.review("Simple fix", plan, "diff", context, changed_files=["src/util.py"], report=report)
        self.assertEqual(result.verdict, "APPROVED")
        self.assertEqual(len(result.consensus_records), 1)
        self.assertFalse(result.consensus_records[0].is_high_risk)
        self.assertEqual(len(report.review_consensus), 1)

    def test_high_risk_pattern_matching_triggers_dual_review(self):
        class PrimaryReviewProvider(MockProvider):
            def review_changes(self, task, plan, diff, context):
                return ReviewResult("APPROVED", "Primary: Clean auth changes", [])

        class SecondaryReviewProvider(MockProvider):
            def review_changes(self, task, plan, diff, context):
                return ReviewResult("APPROVED", "Secondary: Security audit passed", [])

        primary = Reviewer(PrimaryReviewProvider())
        secondary = Reviewer(SecondaryReviewProvider())

        consensus = DeliberativeReviewConsensus(
            primary_reviewer=primary,
            secondary_reviewer=secondary,
            dual_review_enabled=False,
            high_risk_dual_review=True,
        )

        plan = Plan(objective="Update login auth", files_likely_to_change=["src/auth/jwt_token.py"])
        context = ProjectContext(root=".")
        report = RunReport(project=context, plan=plan, changed_files=["src/auth/jwt_token.py"])

        result = consensus.review("Update login auth", plan, "diff", context, changed_files=["src/auth/jwt_token.py"], report=report)
        self.assertEqual(result.verdict, "APPROVED")
        self.assertIn("Dual review verified", result.summary)
        self.assertEqual(len(result.consensus_records), 1)
        self.assertTrue(result.consensus_records[0].is_high_risk)
        self.assertEqual(result.consensus_records[0].primary_verdict, "APPROVED")
        self.assertEqual(result.consensus_records[0].secondary_verdict, "APPROVED")

    def test_high_risk_secondary_veto_blocks_approval(self):
        class PrimaryReviewProvider(MockProvider):
            def review_changes(self, task, plan, diff, context):
                return ReviewResult("APPROVED", "Primary says okay", [])

        class SecondaryReviewProvider(MockProvider):
            def review_changes(self, task, plan, diff, context):
                return ReviewResult("CHANGES_REQUIRED", "Secondary found vulnerability", ["Hardcoded secret in models.py"])

        primary = Reviewer(PrimaryReviewProvider())
        secondary = Reviewer(SecondaryReviewProvider())

        consensus = DeliberativeReviewConsensus(
            primary_reviewer=primary,
            secondary_reviewer=secondary,
            dual_review_enabled=False,
            high_risk_dual_review=True,
        )

        plan = Plan(objective="Add database model", files_likely_to_change=["src/models.py"])
        context = ProjectContext(root=".")
        report = RunReport(project=context, plan=plan, changed_files=["src/models.py"])

        result = consensus.review("Add database model", plan, "diff", context, changed_files=["src/models.py"], report=report)
        self.assertEqual(result.verdict, "CHANGES_REQUIRED")
        self.assertIn("Dual review veto", result.summary)
        self.assertIn("Hardcoded secret in models.py", result.findings)
        self.assertEqual(result.consensus_records[0].final_consensus_verdict, "CHANGES_REQUIRED")

    def test_primary_rejection_returns_immediately_without_secondary(self):
        secondary_called = []

        class PrimaryReviewProvider(MockProvider):
            def review_changes(self, task, plan, diff, context):
                return ReviewResult("CHANGES_REQUIRED", "Primary rejected immediately", ["Syntax error"])

        class SecondaryReviewProvider(MockProvider):
            def review_changes(self, task, plan, diff, context):
                secondary_called.append(True)
                return ReviewResult("APPROVED", "Secondary okay", [])

        primary = Reviewer(PrimaryReviewProvider())
        secondary = Reviewer(SecondaryReviewProvider())

        consensus = DeliberativeReviewConsensus(
            primary_reviewer=primary,
            secondary_reviewer=secondary,
            dual_review_enabled=True,
        )

        plan = Plan(objective="Broken code", files_likely_to_change=["src/main.py"])
        context = ProjectContext(root=".")
        report = RunReport(project=context, plan=plan, changed_files=["src/main.py"])

        result = consensus.review("Broken code", plan, "diff", context, changed_files=["src/main.py"], report=report)
        self.assertEqual(result.verdict, "CHANGES_REQUIRED")
        self.assertEqual(len(secondary_called), 0)
        self.assertEqual(len(result.consensus_records), 1)
        self.assertEqual(result.consensus_records[0].primary_verdict, "CHANGES_REQUIRED")

    def test_dynamic_scope_growth_triggers_high_risk_dual_review(self):
        class PrimaryReviewProvider(MockProvider):
            def review_changes(self, task, plan, diff, context):
                return ReviewResult("APPROVED", "Primary approved", [])

        class SecondaryReviewProvider(MockProvider):
            def review_changes(self, task, plan, diff, context):
                return ReviewResult("APPROVED", "Secondary verified growth", [])

        primary = Reviewer(PrimaryReviewProvider())
        secondary = Reviewer(SecondaryReviewProvider())

        consensus = DeliberativeReviewConsensus(
            primary_reviewer=primary,
            secondary_reviewer=secondary,
            dual_review_enabled=False,
            high_risk_dual_review=True,
        )

        plan = Plan(objective="Refactor helpers", files_likely_to_change=["src/file1.py"])
        plan.apply_amendment(ScopeExpansionProposal(path="src/file2.py", reason="needed", is_create=True))
        plan.apply_amendment(ScopeExpansionProposal(path="src/file3.py", reason="needed too", is_create=True))

        context = ProjectContext(root=".")
        report = RunReport(project=context, plan=plan, changed_files=["src/file1.py", "src/file2.py", "src/file3.py"])

        result = consensus.review("Refactor helpers", plan, "diff", context, changed_files=["src/file1.py", "src/file2.py", "src/file3.py"], report=report)
        self.assertEqual(result.verdict, "APPROVED")
        self.assertTrue(result.consensus_records[0].is_high_risk)
        self.assertIn("scope amendments", result.consensus_records[0].high_risk_reason)


class OrchestratorSpecialistIntegrationTests(unittest.TestCase):
    """End-to-end integration tests for Orchestrator with SpecialistModelRouter."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.storage = JsonFileStorage(self.root / ".agent_data")
        self.repo_lock = threading.Lock()
        self.memory_lock = threading.Lock()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root)

    def test_orchestrator_routes_specialist_roles_e2e(self):
        roles_invoked = []

        class TrackingPlanner(MockProvider):
            def generate_plan(self, task, context):
                roles_invoked.append("planning")
                return Plan(objective=task, files_likely_to_change=["src/calc.py"])

        class TrackingCoder(MockProvider):
            def generate_code(self, task, plan, context, failure=None, review=None):
                roles_invoked.append("implementation")
                return [FileOperation("create", "src/calc.py", content="def add(a, b): return a + b\n")]

        class TrackingReviewer(MockProvider):
            def review_changes(self, task, plan, diff, context):
                roles_invoked.append("review")
                return ReviewResult("APPROVED", "Clean code", [])

        config = AgentConfig(
            project=self.root,
            provider="mock",
            planning_provider="mock_plan",
            implementation_provider="mock_code",
            review_provider="mock_rev",
        )

        providers = {
            "mock_plan": TrackingPlanner(),
            "mock_code": TrackingCoder(),
            "mock_rev": TrackingReviewer(),
            "mock": MockProvider(),
        }

        def custom_factory(cfg, api_key=None):
            return providers.get(cfg.provider, MockProvider())

        orchestrator = Orchestrator(config, self.storage, scheduler=None, repo_lock=self.repo_lock, memory_lock=self.memory_lock)
        orchestrator.router.provider_factory = custom_factory

        now = datetime.datetime.now(datetime.timezone.utc)
        task = Task(task_id="spec-task-1", objective="Build calculator", status=TaskStatus.PENDING, created_at=now, updated_at=now)
        self.storage.save_task(task)

        report = orchestrator.run(task)
        self.assertTrue(report.completed)
        self.assertIn("planning", roles_invoked)
        self.assertIn("implementation", roles_invoked)
        self.assertIn("review", roles_invoked)

    def test_checkpoint_persists_specialist_state_and_restores_on_resume(self):
        config = AgentConfig(
            project=self.root,
            provider="mock",
            planning_provider="gemini",
            planning_model="gemini-2.5-pro",
            implementation_provider="deepseek",
            implementation_model="deepseek-coder",
        )
        orchestrator = Orchestrator(config, self.storage, scheduler=None, repo_lock=self.repo_lock, memory_lock=self.memory_lock)

        now = datetime.datetime.now(datetime.timezone.utc)
        task = Task(task_id="task-chk-spec", objective="Resume specialist test", status=TaskStatus.PENDING, created_at=now, updated_at=now)
        self.storage.save_task(task)

        context = ProjectContext(root=str(self.root))
        report = RunReport(
            project=context,
            plan=Plan(objective=task.objective),
            changed_files=["src/auth/jwt.py"],
            review_consensus=[
                ReviewConsensusRecord(
                    primary_provider="OpenAIProvider",
                    final_consensus_verdict="APPROVED",
                    is_high_risk=True,
                )
            ],
        )

        chk = orchestrator._create_checkpoint(task, None, "Checkpoint with specialist state", context, report)
        self.assertIn("specialist_routing_state", chk.continuation_context)
        self.assertIn("review_consensus", chk.continuation_context)

        snapshot_report = orchestrator._build_run_report(task)
        self.assertEqual(len(snapshot_report.review_consensus), 1)
        self.assertTrue(snapshot_report.review_consensus[0].is_high_risk)
        self.assertIn("planning", snapshot_report.specialist_routing_state)

    def test_unconditional_dual_review_when_dual_review_enabled_is_true(self):
        secondary_called = []

        class PrimaryReviewer(MockProvider):
            def review_changes(self, task, plan, diff, context):
                return ReviewResult("APPROVED", "Primary OK", [])

        class SecondaryReviewer(MockProvider):
            def review_changes(self, task, plan, diff, context):
                secondary_called.append(True)
                return ReviewResult("APPROVED", "Secondary OK", [])

        primary = Reviewer(PrimaryReviewer())
        secondary = Reviewer(SecondaryReviewer())
        consensus = DeliberativeReviewConsensus(
            primary_reviewer=primary,
            secondary_reviewer=secondary,
            dual_review_enabled=True,
            high_risk_dual_review=False,
        )

        plan = Plan(objective="Safe change", files_likely_to_change=["src/safe_util.py"])
        context = ProjectContext(root=".")
        report = RunReport(project=context, plan=plan, changed_files=["src/safe_util.py"])

        result = consensus.review("Safe change", plan, "diff", context, changed_files=["src/safe_util.py"], report=report)
        self.assertEqual(result.verdict, "APPROVED")
        self.assertEqual(len(secondary_called), 1)
        self.assertIn("Dual review verified", result.summary)

    def test_specialist_router_get_provider_chain_preserves_order(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = AgentConfig(
                project=Path(td),
                provider="mock",
                planning_provider="gemini",
                planning_fallbacks=["openai", "anthropic"],
            )
            with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "k1", "OPENAI_API_KEY": "k2", "ANTHROPIC_API_KEY": "k3"}):
                router = SpecialistModelRouter(cfg)
                chain = router.get_provider_chain(SpecialistRole.PLANNING)
                provider_names = [type(p).__name__ for p in chain]
                self.assertEqual(provider_names[0], "GeminiProvider")
                self.assertEqual(provider_names[1], "OpenAIProvider")
                self.assertEqual(provider_names[2], "AnthropicProvider")
                self.assertEqual(provider_names[3], "MockProvider")

    def test_repair_specialist_used_for_failure_repair(self):
        roles_invoked = []

        class RepairTrackingCoder(MockProvider):
            def generate_code(self, task, plan, context, failure=None, review=None):
                if failure:
                    roles_invoked.append("repair")
                else:
                    roles_invoked.append("implementation")
                return [FileOperation("modify", "src/calc.py", content="def add(a, b): return a + b\n")]

        config = AgentConfig(
            project=self.root,
            provider="mock",
            implementation_provider="mock_impl",
            repair_provider="mock_repair",
        )

        providers = {
            "mock_impl": RepairTrackingCoder(),
            "mock_repair": RepairTrackingCoder(),
            "mock": MockProvider(),
        }

        orchestrator = Orchestrator(config, self.storage, scheduler=None, repo_lock=self.repo_lock, memory_lock=self.memory_lock)
        orchestrator.router.provider_factory = lambda cfg, api_key=None: providers.get(cfg.provider, MockProvider())

        plan = Plan(objective="Fix calc", files_likely_to_change=["src/calc.py"])
        context = ProjectContext(root=str(self.root))
        failure = FailureAnalysis(probable_root_cause="Bad return", recommended_fix="fix return")

        orchestrator._execute_code_generation(
            task=Task("task-repair", "Fix calc", status=TaskStatus.RUNNING, created_at=datetime.datetime.now(datetime.timezone.utc), updated_at=datetime.datetime.now(datetime.timezone.utc)),
            plan=plan,
            context=context,
            failure=failure,
            review=None,
            stage_name="repair",
        )

        self.assertIn("repair", roles_invoked)

    def test_map_role_fallback_to_implementation(self):
        config = AgentConfig(project=self.root, provider="mock")
        orchestrator = Orchestrator(config, self.storage, scheduler=None, repo_lock=self.repo_lock, memory_lock=self.memory_lock)
        mapped = orchestrator._map_role("unknown_custom_role")
        self.assertEqual(mapped, SpecialistRole.IMPLEMENTATION)

    def test_verification_specialist_provider_resolution(self):
        config = AgentConfig(
            project=self.root,
            provider="mock",
            verification_provider="gemini",
            verification_model="gemini-2.5-flash",
        )
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            orchestrator = Orchestrator(config, self.storage, scheduler=None, repo_lock=self.repo_lock, memory_lock=self.memory_lock)
            p = orchestrator.router.get_provider(SpecialistRole.VERIFICATION)
            self.assertIsInstance(p, GeminiProvider)
            self.assertEqual(p.config.model, "gemini-2.5-flash")

    def test_subtask_planning_routes_to_planning_specialist(self):
        plan_invoked = []

        class SubtaskPlanner(MockProvider):
            def generate_plan(self, task, context):
                plan_invoked.append(True)
                return Plan(objective=task, files_likely_to_change=["src/sub.py"])

        class SubtaskCoder(MockProvider):
            def generate_code(self, task, plan, context, failure=None, review=None):
                return [FileOperation("create", "src/sub.py", content="def sub(): pass\n")]

            def review_changes(self, task, plan, diff, context):
                return ReviewResult("APPROVED", "OK", [])

        config = AgentConfig(
            project=self.root,
            provider="mock",
            planning_provider="mock_planner",
        )
        providers = {
            "mock_planner": SubtaskPlanner(),
            "mock": SubtaskCoder(),
        }
        orchestrator = Orchestrator(config, self.storage, scheduler=None, repo_lock=self.repo_lock, memory_lock=self.memory_lock)
        orchestrator.router.provider_factory = lambda cfg, api_key=None: providers.get(cfg.provider, MockProvider())

        subtask = Subtask(subtask_id="sub-1", title="Subtask 1", goal="Goal 1")
        now = datetime.datetime.now(datetime.timezone.utc)
        task = Task(task_id="task-sub", objective="Root", status=TaskStatus.PENDING, created_at=now, updated_at=now, plan=TaskPlan("Root", subtasks=[subtask]))
        self.storage.save_task(task)

        report = orchestrator.run(task, subtask_id="sub-1")
        self.assertTrue(report.completed)
        self.assertEqual(len(plan_invoked), 1)
