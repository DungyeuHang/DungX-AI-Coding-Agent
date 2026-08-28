from __future__ import annotations

import datetime
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

from local_agent.commands import CommandRunner
from local_agent.contract_extractor import ContractExtractor
from local_agent.filesystem import ProjectFilesystem
from local_agent.models import (
    Checkpoint,
    CommandSpec,
    ExecutionResult,
    ExportedSymbol,
    Plan,
    ProjectContext,
    RunReport,
    Subtask,
    SubtaskContract,
    SubtaskStatus,
    Task,
    TaskPlan,
    TestExecutionRecord,
    ValidationPlan,
    VerificationGap,
)
from local_agent.providers import AIProvider
from local_agent.test_synthesizer import (
    BehavioralVerifier,
    TestSynthesizer,
    VerificationGapAnalyzer,
)
from local_agent.validation import ValidationIntelligence


class TestBehavioralVerificationModels(unittest.TestCase):
    def test_test_execution_record_creation_and_bounds(self):
        rec = TestExecutionRecord(
            test_id="test-1",
            command="pytest tests/test_foo.py",
            status="passed",
            exit_code=0,
            duration_seconds=0.125,
            stdout_summary="A" * 600,
            stderr_summary="B" * 600,
            failure_classification="C" * 300,
            exercised_symbols=[f"sym_{i}" for i in range(30)],
        )
        self.assertEqual(rec.status, "passed")
        self.assertEqual(rec.exit_code, 0)
        self.assertLessEqual(len(rec.stdout_summary), 500)
        self.assertLessEqual(len(rec.stderr_summary), 500)
        self.assertLessEqual(len(rec.failure_classification), 200)
        self.assertLessEqual(len(rec.exercised_symbols), 20)
        self.assertTrue(rec.stdout_summary.endswith("..."))

    def test_test_execution_record_serialization_roundtrip(self):
        now = datetime.datetime.now(datetime.timezone.utc)
        rec = TestExecutionRecord(
            test_id="rec-100",
            command="pytest -q tests/test_bar.py",
            status="failed",
            exit_code=1,
            duration_seconds=0.45,
            stdout_summary="AssertionError: expected 5 got 4",
            stderr_summary="",
            synthesized=True,
            exercised_symbols=["BarService", "compute"],
            timestamp=now,
            failure_classification="ASSERTION_ERROR",
        )
        data = rec.to_dict()
        self.assertEqual(data["test_id"], "rec-100")
        self.assertEqual(data["status"], "failed")
        self.assertEqual(data["exit_code"], 1)

        restored = TestExecutionRecord.from_dict(data)
        self.assertEqual(restored.test_id, "rec-100")
        self.assertEqual(restored.status, "failed")
        self.assertEqual(restored.exit_code, 1)
        self.assertEqual(restored.exercised_symbols, ["BarService", "compute"])
        self.assertEqual(restored.failure_classification, "ASSERTION_ERROR")

    def test_verification_gap_creation_and_bounds(self):
        syms = [ExportedSymbol(f"s_{i}", f"Name{i}", "function", "foo.py") for i in range(30)]
        gap = VerificationGap(
            missing_test_symbols=syms,
            untested_files=[f"file_{i}.py" for i in range(30)],
            reasons=[f"reason {i}" for i in range(30)],
            severity="high",
        )
        self.assertLessEqual(len(gap.missing_test_symbols), 20)
        self.assertLessEqual(len(gap.untested_files), 20)
        self.assertLessEqual(len(gap.reasons), 20)
        self.assertEqual(gap.severity, "high")

    def test_verification_gap_serialization_roundtrip(self):
        sym = ExportedSymbol("mod.py::Helper", "Helper", "class", "mod.py")
        gap = VerificationGap(
            missing_test_symbols=[sym],
            untested_files=["mod.py"],
            reasons=["Untested class Helper"],
            severity="medium",
        )
        data = gap.to_dict()
        restored = VerificationGap.from_dict(data)
        self.assertEqual(len(restored.missing_test_symbols), 1)
        self.assertEqual(restored.missing_test_symbols[0].name, "Helper")
        self.assertEqual(restored.untested_files, ["mod.py"])
        self.assertEqual(restored.severity, "medium")

    def test_exported_symbol_verified_field_and_backward_compatibility(self):
        # Legacy dictionary without verified fields
        legacy_data = {
            "symbol_id": "api.py::get_user",
            "name": "get_user",
            "kind": "function",
            "file_path": "api.py",
            "signature": "def get_user(uid: str) -> dict:",
            "description": "Fetch user",
        }
        sym = ExportedSymbol.from_dict(legacy_data)
        self.assertFalse(sym.verified)
        self.assertEqual(sym.verification_source, "")

        # Updated symbol with verified = True
        sym.verified = True
        sym.verification_source = "synthetic_test_1"
        sym_dict = sym.to_dict()
        self.assertTrue(sym_dict["verified"])
        self.assertEqual(sym_dict["verification_source"], "synthetic_test_1")

    def test_subtask_contract_behavioral_evidence_and_prompt_formatting(self):
        record = TestExecutionRecord(
            test_id="t-1",
            command="pytest tests/_synthetic_test.py",
            status="passed",
            exit_code=0,
            duration_seconds=0.1,
            exercised_symbols=["RateLimiter"],
        )
        sym_verified = ExportedSymbol(
            symbol_id="limiter.py::RateLimiter",
            name="RateLimiter",
            kind="class",
            file_path="limiter.py",
            verified=True,
        )
        sym_unverified = ExportedSymbol(
            symbol_id="limiter.py::reset_all",
            name="reset_all",
            kind="function",
            file_path="limiter.py",
            verified=False,
        )
        contract = SubtaskContract(
            subtask_id="sub-1",
            title="Implement Rate Limiter",
            exported_symbols=[sym_verified, sym_unverified],
            created_files=["limiter.py"],
            behavioral_evidence=[record],
        )

        formatted = contract.format_for_prompt()
        self.assertIn("[class] `RateLimiter` in `limiter.py` [VERIFIED]", formatted)
        self.assertIn("[function] `reset_all` in `limiter.py` [UNVERIFIED]", formatted)
        self.assertIn("Behavioral Verification Evidence:", formatted)
        self.assertIn("[PASSED] `pytest tests/_synthetic_test.py`", formatted)


class TestVerificationGapAnalyzer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.filesystem = ProjectFilesystem(self.root)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_gap_analysis_detects_new_module_without_tests(self):
        self.filesystem.create_file("service.py", "class AuthService:\n    def login(self): pass\n")
        analyzer = VerificationGapAnalyzer(self.root, filesystem=self.filesystem)
        context = ProjectContext(root=str(self.root), source_files=["service.py"], test_files=[])

        sym = ExportedSymbol("service.py::AuthService", "AuthService", "class", "service.py")
        gap = analyzer.analyze(changed_files=["service.py"], exported_symbols=[sym], context=context)

        self.assertIsNotNone(gap)
        self.assertEqual(len(gap.missing_test_symbols), 1)
        self.assertEqual(gap.missing_test_symbols[0].name, "AuthService")
        self.assertIn("service.py", gap.untested_files)

    def test_gap_analysis_no_gap_when_symbol_referenced_in_test_file(self):
        self.filesystem.create_file("service.py", "class AuthService:\n    def login(self): pass\n")
        self.filesystem.create_file("tests/test_service.py", "from service import AuthService\ndef test_auth(): AuthService()\n")
        analyzer = VerificationGapAnalyzer(self.root, filesystem=self.filesystem)
        context = ProjectContext(
            root=str(self.root),
            source_files=["service.py"],
            test_files=["tests/test_service.py"],
        )

        sym = ExportedSymbol("service.py::AuthService", "AuthService", "class", "service.py")
        gap = analyzer.analyze(changed_files=["service.py"], exported_symbols=[sym], context=context)

        self.assertIsNone(gap)

    def test_gap_analysis_partial_symbol_coverage(self):
        self.filesystem.create_file("math_tools.py", "def add(a, b): return a + b\ndef multiply(a, b): return a * b\n")
        # Test file only mentions add, not multiply
        self.filesystem.create_file("tests/test_math.py", "from math_tools import add\ndef test_add(): assert add(1, 2) == 3\n")
        analyzer = VerificationGapAnalyzer(self.root, filesystem=self.filesystem)
        context = ProjectContext(
            root=str(self.root),
            source_files=["math_tools.py"],
            test_files=["tests/test_math.py"],
        )

        sym_add = ExportedSymbol("math_tools.py::add", "add", "function", "math_tools.py")
        sym_mult = ExportedSymbol("math_tools.py::multiply", "multiply", "function", "math_tools.py")

        gap = analyzer.analyze(changed_files=["math_tools.py"], exported_symbols=[sym_add, sym_mult], context=context)
        self.assertIsNotNone(gap)
        self.assertEqual(len(gap.missing_test_symbols), 1)
        self.assertEqual(gap.missing_test_symbols[0].name, "multiply")

    def test_gap_analysis_empty_changed_files_returns_none(self):
        analyzer = VerificationGapAnalyzer(self.root, filesystem=self.filesystem)
        context = ProjectContext(root=str(self.root))
        gap = analyzer.analyze(changed_files=[], exported_symbols=[], context=context)
        self.assertIsNone(gap)

    def test_validation_intelligence_analyze_verification_gap_helper(self):
        self.filesystem.create_file("untested.py", "def new_feature(): pass\n")
        val_intel = ValidationIntelligence(self.root)
        context = ProjectContext(root=str(self.root), source_files=["untested.py"], test_files=[])
        sym = ExportedSymbol("untested.py::new_feature", "new_feature", "function", "untested.py")

        gap = val_intel.analyze_verification_gap(["untested.py"], [sym], context)
        self.assertIsNotNone(gap)
        self.assertEqual(len(gap.missing_test_symbols), 1)


class TestTestSynthesizer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.filesystem = ProjectFilesystem(self.root)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_synthesizer_deterministic_template_generation(self):
        sym_class = ExportedSymbol("worker.py::TaskWorker", "TaskWorker", "class", "worker.py")
        sym_func = ExportedSymbol("worker.py::run_worker", "run_worker", "function", "worker.py")
        gap = VerificationGap(missing_test_symbols=[sym_class, sym_func], untested_files=["worker.py"])

        synthesizer = TestSynthesizer(self.root, filesystem=self.filesystem)
        subtask = Subtask(subtask_id="sub-worker", title="Worker Subtask")
        context = ProjectContext(root=str(self.root))

        test_code = synthesizer.synthesize_test(subtask, gap, context)
        self.assertIsNotNone(test_code)
        self.assertIn("import pytest", test_code)
        self.assertIn("from worker import TaskWorker, run_worker", test_code)
        self.assertIn("def test_behavior_taskworker():", test_code)
        self.assertIn("def test_behavior_run_worker():", test_code)

    def test_synthesizer_validates_and_rejects_syntax_errors(self):
        synthesizer = TestSynthesizer(self.root, filesystem=self.filesystem)
        invalid_code = "def broken_test(: assert True\n"
        self.assertFalse(synthesizer._validate_test_code(invalid_code))

    def test_synthesizer_rejects_dangerous_patterns(self):
        synthesizer = TestSynthesizer(self.root, filesystem=self.filesystem)
        dangerous_code = "import os\ndef test_bad(): os.system('rm -rf /')\n"
        self.assertFalse(synthesizer._validate_test_code(dangerous_code))

        dangerous_git = "def test_bad(): import subprocess; subprocess.call('git reset --hard', shell=True)\n"
        self.assertFalse(synthesizer._validate_test_code(dangerous_git))

    def test_synthesizer_with_mock_provider(self):
        provider = MagicMock(spec=AIProvider)
        provider.generate_plan.return_value = Plan(
            objective="Generated Test",
            steps=["```python\ndef test_custom():\n    assert 1 + 1 == 2\n```"]
        )
        synthesizer = TestSynthesizer(self.root, provider=provider, filesystem=self.filesystem)
        subtask = Subtask(subtask_id="sub-ai", title="AI Test")
        sym = ExportedSymbol("calc.py::add", "add", "function", "calc.py")
        gap = VerificationGap(missing_test_symbols=[sym], untested_files=["calc.py"])
        context = ProjectContext(root=str(self.root))

        test_code = synthesizer.synthesize_test(subtask, gap, context)
        self.assertIsNotNone(test_code)
        self.assertIn("def test_custom():", test_code)


class TestBehavioralVerifier(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.filesystem = ProjectFilesystem(self.root)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_verifier_executes_passing_fixture_and_records_evidence(self):
        # Create valid source file in root
        self.filesystem.create_file("calculator.py", "def add(a: int, b: int) -> int:\n    return a + b\n")

        test_code = "from calculator import add\ndef test_add(): assert add(2, 3) == 5\n"
        verifier = BehavioralVerifier(self.root, filesystem=self.filesystem)
        sym = ExportedSymbol("calculator.py::add", "add", "function", "calculator.py")

        record = verifier.verify(test_code, "sub-calc", exercised_symbols=[sym])
        self.assertEqual(record.status, "passed")
        self.assertEqual(record.exit_code, 0)
        self.assertEqual(record.exercised_symbols, ["add"])
        self.assertTrue(record.synthesized)

    def test_verifier_executes_failing_fixture_and_records_failure(self):
        self.filesystem.create_file("broken_calc.py", "def add(a: int, b: int) -> int:\n    return a - b\n")

        test_code = "from broken_calc import add\ndef test_add(): assert add(2, 3) == 5\n"
        verifier = BehavioralVerifier(self.root, filesystem=self.filesystem)
        sym = ExportedSymbol("broken_calc.py::add", "add", "function", "broken_calc.py")

        record = verifier.verify(test_code, "sub-broken-calc", exercised_symbols=[sym])
        self.assertEqual(record.status, "failed")
        self.assertNotEqual(record.exit_code, 0)
        self.assertEqual(record.failure_classification, "SYNTHETIC_TEST_FAILURE")

    def test_verifier_cleans_up_ephemeral_fixture_file(self):
        test_code = "def test_noop(): assert True\n"
        verifier = BehavioralVerifier(self.root, filesystem=self.filesystem)
        record = verifier.verify(test_code, "sub-ephemeral")

        temp_test_file = self.root / "tests" / "_synthetic_test_sub_ephemeral.py"
        self.assertFalse(temp_test_file.exists())


class TestContractCertificationIntegration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.filesystem = ProjectFilesystem(self.root)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_contract_extractor_certifies_only_exercised_symbols(self):
        code = '''class PaymentGateway:
    def charge(self, amount: float) -> bool:
        return True

def refund(tx_id: str) -> bool:
    return True
'''
        self.filesystem.create_file("billing.py", code)
        extractor = ContractExtractor(filesystem=self.filesystem, project_root=self.root)
        subtask = Subtask(subtask_id="sub-bill", title="Billing Engine")
        report = RunReport(project=ProjectContext(root=str(self.root)), changed_files=["billing.py"])

        # Record indicates PaymentGateway was tested and passed, but refund was not
        record = TestExecutionRecord(
            test_id="synth-billing",
            command="pytest tests/_synthetic_test.py",
            status="passed",
            exit_code=0,
            exercised_symbols=["PaymentGateway"],
        )
        contract = extractor.extract_contract(subtask, report, behavioral_records=[record])

        sym_gateway = next((s for s in contract.exported_symbols if s.name == "PaymentGateway"), None)
        sym_refund = next((s for s in contract.exported_symbols if s.name == "refund"), None)

        self.assertIsNotNone(sym_gateway)
        self.assertTrue(sym_gateway.verified)
        self.assertEqual(sym_gateway.verification_source, "synth-billing")

        self.assertIsNotNone(sym_refund)
        self.assertFalse(sym_refund.verified)
        self.assertEqual(sym_refund.verification_source, "")

        self.assertEqual(len(contract.behavioral_evidence), 1)
        self.assertTrue(any("Behavioral verification: 1/1" in note for note in contract.architectural_notes))


class TestCheckpointAndSecurityInvariants(unittest.TestCase):
    def test_checkpoint_persists_and_restores_behavioral_evidence(self):
        record = TestExecutionRecord(
            test_id="synth-1",
            command="pytest -q test.py",
            status="passed",
            exit_code=0,
            exercised_symbols=["Worker"],
        )
        sym = ExportedSymbol("worker.py::Worker", "Worker", "class", "worker.py", verified=True, verification_source="synth-1")
        contract = SubtaskContract(
            subtask_id="sub-1",
            title="Step 1",
            exported_symbols=[sym],
            behavioral_evidence=[record],
        )
        subtask = Subtask(subtask_id="sub-1", title="Step 1", status=SubtaskStatus.COMPLETED, contract=contract)
        plan = TaskPlan(objective="Task", subtasks=[subtask])

        now = datetime.datetime.now(datetime.timezone.utc)
        checkpoint = Checkpoint(
            checkpoint_id="chk-100",
            task_id="task-100",
            subtask_id="sub-1",
            timestamp=now,
            current_state_description="Completed with behavioral evidence",
            continuation_context={"task_plan": plan.to_dict()},
        )

        chk_dict = checkpoint.to_dict()
        restored_chk = Checkpoint.from_dict(chk_dict)
        restored_plan = TaskPlan.from_dict(restored_chk.continuation_context["task_plan"])

        restored_sub = restored_plan.subtasks[0]
        self.assertIsNotNone(restored_sub.contract)
        self.assertEqual(len(restored_sub.contract.behavioral_evidence), 1)
        self.assertEqual(restored_sub.contract.behavioral_evidence[0].status, "passed")
        self.assertTrue(restored_sub.contract.exported_symbols[0].verified)

    def test_legacy_checkpoint_without_behavioral_evidence_deserializes_safely(self):
        legacy_data = {
            "subtask_id": "sub-old",
            "title": "Old Contract",
            "exported_symbols": [
                {"symbol_id": "old.py::fn", "name": "fn", "kind": "function", "file_path": "old.py"}
            ],
            "modified_files": ["old.py"],
            "created_files": [],
            "validation_commands": ["pytest"],
            "architectural_notes": ["Note"],
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        contract = SubtaskContract.from_dict(legacy_data)
        self.assertEqual(contract.behavioral_evidence, [])
        self.assertFalse(contract.exported_symbols[0].verified)

    def test_run_report_verification_gap_and_evidence_serialization(self):
        record = TestExecutionRecord(
            test_id="rec-1",
            command="pytest test.py",
            status="passed",
            exit_code=0,
        )
        gap = VerificationGap(untested_files=["foo.py"], severity="low")
        report = RunReport(
            project=ProjectContext(root="/tmp"),
            changed_files=["foo.py"],
            behavioral_evidence=[record],
            verification_gap=gap,
        )
        self.assertEqual(len(report.behavioral_evidence), 1)
        self.assertIsNotNone(report.verification_gap)
        self.assertEqual(report.verification_gap.untested_files, ["foo.py"])


class TestAdvancedGapAnalysisAndSynthesis(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.filesystem = ProjectFilesystem(self.root)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_gap_analysis_global_suite_pass_does_not_mask_untested_new_symbol(self):
        # Scenario: A new symbol exists, global tests pass (e.g. tests/test_legacy.py exists),
        # but zero tests mention the new symbol.
        self.filesystem.create_file("auth_v2.py", "class TokenManager:\n    def issue_token(self): pass\n")
        self.filesystem.create_file("tests/test_legacy.py", "def test_old_feature(): assert True\n")
        analyzer = VerificationGapAnalyzer(self.root, filesystem=self.filesystem)
        context = ProjectContext(
            root=str(self.root),
            source_files=["auth_v2.py"],
            test_files=["tests/test_legacy.py"],
        )

        sym = ExportedSymbol("auth_v2.py::TokenManager", "TokenManager", "class", "auth_v2.py")
        gap = analyzer.analyze(changed_files=["auth_v2.py"], exported_symbols=[sym], context=context)

        self.assertIsNotNone(gap)
        self.assertEqual(len(gap.missing_test_symbols), 1)
        self.assertEqual(gap.missing_test_symbols[0].name, "TokenManager")

    def test_synthesizer_empty_provider_response_falls_back_to_template(self):
        provider = MagicMock(spec=AIProvider)
        provider.generate_plan.return_value = None  # Provider returns empty/None
        synthesizer = TestSynthesizer(self.root, provider=provider, filesystem=self.filesystem)
        subtask = Subtask(subtask_id="sub-fallback", title="Fallback Test")
        sym = ExportedSymbol("engine.py::start_engine", "start_engine", "function", "engine.py")
        gap = VerificationGap(missing_test_symbols=[sym], untested_files=["engine.py"])
        context = ProjectContext(root=str(self.root))

        test_code = synthesizer.synthesize_test(subtask, gap, context)
        self.assertIsNotNone(test_code)
        self.assertIn("def test_behavior_start_engine():", test_code)

    def test_synthesizer_enforces_char_limit(self):
        synthesizer = TestSynthesizer(self.root, max_synthetic_test_chars=50, filesystem=self.filesystem)
        subtask = Subtask(subtask_id="sub-bound", title="Bounded Test")
        sym = ExportedSymbol("long_mod.py::HugeWorker", "HugeWorker", "class", "long_mod.py")
        gap = VerificationGap(missing_test_symbols=[sym], untested_files=["long_mod.py"])
        context = ProjectContext(root=str(self.root))

        test_code = synthesizer.synthesize_test(subtask, gap, context)
        if test_code is not None:
            self.assertLessEqual(len(test_code), 50)

    def test_verifier_handles_import_error_fixture(self):
        # Test imports non-existent module
        test_code = "from nonexistent_module_xyz import something\ndef test_bad(): assert True\n"
        verifier = BehavioralVerifier(self.root, filesystem=self.filesystem)
        sym = ExportedSymbol("nonexistent_module_xyz.py::something", "something", "function", "nonexistent_module_xyz.py")

        record = verifier.verify(test_code, "sub-import-err", exercised_symbols=[sym])
        self.assertEqual(record.status, "failed")
        self.assertNotEqual(record.exit_code, 0)

    def test_contract_extractor_failed_records_do_not_certify_symbols(self):
        code = "def dangerous_action(): return False\n"
        self.filesystem.create_file("actions.py", code)
        extractor = ContractExtractor(filesystem=self.filesystem, project_root=self.root)
        subtask = Subtask(subtask_id="sub-act", title="Actions")
        report = RunReport(project=ProjectContext(root=str(self.root)), changed_files=["actions.py"])

        failing_record = TestExecutionRecord(
            test_id="synth-act",
            command="pytest tests/_synthetic_test.py",
            status="failed",
            exit_code=1,
            exercised_symbols=["dangerous_action"],
        )
        contract = extractor.extract_contract(subtask, report, behavioral_records=[failing_record])

        sym_act = next((s for s in contract.exported_symbols if s.name == "dangerous_action"), None)
        self.assertIsNotNone(sym_act)
        # Crucial check: failing test does NOT certify symbol as verified!
        self.assertFalse(sym_act.verified)
        self.assertEqual(sym_act.verification_source, "")


class TestOrchestratorAndSecurityInvariants(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.filesystem = ProjectFilesystem(self.root)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_synthesizer_returns_none_when_no_python_symbols_or_files(self):
        gap = VerificationGap(
            missing_test_symbols=[ExportedSymbol("styles.css::body", "body", "style", "styles.css")],
            untested_files=["styles.css"],
        )
        synthesizer = TestSynthesizer(self.root, filesystem=self.filesystem)
        subtask = Subtask(subtask_id="sub-css")
        context = ProjectContext(root=str(self.root))
        result = synthesizer.synthesize_test(subtask, gap, context)
        self.assertIsNone(result)

    def test_gap_analysis_handles_ts_js_files(self):
        self.filesystem.create_file("src/utils.ts", "export function formatName(name: string) { return name.trim(); }\n")
        self.filesystem.create_file("tests/utils.test.ts", "import { formatName } from '../src/utils';\n")
        analyzer = VerificationGapAnalyzer(self.root, filesystem=self.filesystem)
        context = ProjectContext(
            root=str(self.root),
            source_files=["src/utils.ts"],
            test_files=["tests/utils.test.ts"],
        )
        sym = ExportedSymbol("src/utils.ts::formatName", "formatName", "function", "src/utils.ts")
        gap = analyzer.analyze(changed_files=["src/utils.ts"], exported_symbols=[sym], context=context)
        self.assertIsNone(gap)

    def test_behavioral_verifier_exception_safety(self):
        # Even if runner throws an exception, verifier returns a valid failed TestExecutionRecord without crashing
        mock_runner = MagicMock(spec=CommandRunner)
        mock_runner.run.side_effect = RuntimeError("Process execution crashed")

        verifier = BehavioralVerifier(self.root, runner=mock_runner, filesystem=self.filesystem)
        rec = verifier.verify("assert True", "sub-crash")
        self.assertEqual(rec.status, "failed")
        self.assertEqual(rec.exit_code, 1)
        self.assertIn("Process execution crashed", rec.stderr_summary)

    def test_verification_gap_analyzer_with_targeted_commands(self):
        self.filesystem.create_file("core.py", "def run_core(): pass\n")
        analyzer = VerificationGapAnalyzer(self.root, filesystem=self.filesystem)
        context = ProjectContext(root=str(self.root), source_files=["core.py"], test_files=[])
        sym = ExportedSymbol("core.py::run_core", "run_core", "function", "core.py")

        targeted = [CommandSpec("targeted_pytest", ("pytest", "tests/test_core.py"), "Targeted test", "unit_test")]
        gap = analyzer.analyze(
            changed_files=["core.py"],
            exported_symbols=[sym],
            context=context,
            targeted_commands=targeted,
        )
        self.assertIsNotNone(gap)
        self.assertEqual(len(gap.missing_test_symbols), 1)


if __name__ == "__main__":
    unittest.main()

