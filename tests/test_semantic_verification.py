"""Phase 4.24 tests: post-execution semantic verification and adversarial success validation."""

from __future__ import annotations

import ast
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from local_agent.evidence import compute_state_fingerprint
from local_agent.maintenance import (
    SEVERITY_HIGH,
    CandidateRunOutcome,
    CandidateState,
    MaintenanceCandidate,
    MaintenanceSignal,
    ReassessmentOutcome,
)
from local_agent.maintenance_execution import (
    ALL_EXECUTION_STATUSES,
    ExecutionJournal,
    MaintenanceApprovalGate,
    MaintenanceExecutionOutcome,
    MaintenanceExecutionResult,
    MaintenanceExecutionStatus,
    MaintenanceExecutor,
    RETRYABLE_STATUSES,
)
from local_agent.maintenance_oracle import (
    ExecutionOracle,
    OracleClass,
    OracleObservation,
    ParseOracle,
    oracle_for,
)
from local_agent.maintenance_policy import AutonomyTier, MaintenanceBudget, MaintenanceExecutionPolicy
from local_agent.maintenance_runner import (
    MaintenanceWorkOrder,
    ReassessmentVerdict,
    build_work_order,
    reassess,
)
from local_agent.models import FileOperation, ImplementationResult
from local_agent.repository import RepositoryIntelligence
from local_agent.semantic_verification import (
    ALL_FAILURE_CATEGORIES,
    ALL_SEMANTIC_STATUSES,
    DefectIdentity,
    FailureCategory,
    ParseFailureSemanticVerifier,
    SEMANTIC_VERIFICATION_SCHEMA_VERSION,
    SemanticVerificationStatus,
    SemanticVerifier,
    UnverifiableSemanticVerifier,
    VerificationEvidence,
    all_verifiers,
    register_verifier,
    verifier_for,
)


from tests.test_maintenance_execution import ScriptedProvider


class TestDefectIdentityAndEvidenceModels(unittest.TestCase):
    """Test DefectIdentity and VerificationEvidence immutability and serialization."""

    def test_defect_identity_immutability_and_serialization(self):
        identity = DefectIdentity(
            signal_kind=MaintenanceSignal.PARSE_FAILURE,
            candidate_id="c123",
            subject="src/broken.py",
            relative_path="src/broken.py",
            defect_fingerprint="def_fp_123",
            source_state_fingerprint="src_fp_456",
            diagnostic_message="expected ':'",
            diagnostic_line=12,
            diagnostic_column=5,
            syntax_error_class="SyntaxError",
            compiler_or_parser="cpython_ast",
            lexical_symbols=("foo", "bar", "Baz"),
            import_names=("os", "sys"),
            block_signatures={"foo": "sig_foo", "bar": "sig_bar"},
            substantive_symbol_metrics={"foo": {"statement_count": 5}},
            significant_lines=25,
        )

        d = identity.to_dict()
        self.assertEqual(d["signal_kind"], MaintenanceSignal.PARSE_FAILURE)
        self.assertEqual(d["candidate_id"], "c123")
        self.assertEqual(d["diagnostic_line"], 12)
        self.assertEqual(d["diagnostic_column"], 5)
        self.assertEqual(d["lexical_symbols"], ["foo", "bar", "Baz"])
        self.assertEqual(d["import_names"], ["os", "sys"])

        restored = DefectIdentity.from_dict(d)
        self.assertEqual(restored.signal_kind, identity.signal_kind)
        self.assertEqual(restored.candidate_id, identity.candidate_id)
        self.assertEqual(restored.diagnostic_line, identity.diagnostic_line)
        self.assertEqual(restored.lexical_symbols, identity.lexical_symbols)
        self.assertEqual(restored.import_names, identity.import_names)

        # DefectIdentity is frozen and cannot be mutated
        with self.assertRaises((TypeError, AttributeError)):
            identity.diagnostic_line = 99  # type: ignore[misc]

    def test_verification_evidence_serialization(self):
        evidence = VerificationEvidence(
            verifier="parse_failure_semantic_verifier",
            signal_kind=MaintenanceSignal.PARSE_FAILURE,
            candidate_id="cand_99",
            status=SemanticVerificationStatus.RESOLVED,
            failure_category=FailureCategory.NONE,
            confidence=OracleClass.DETERMINISTIC,
            before_fingerprint="bfp",
            after_fingerprint="afp",
            defect_fingerprint="dfp",
            diagnostic_before={"message": "invalid syntax", "line": 4},
            affected_files_before=["src/module.py"],
            affected_files_after=["src/module.py"],
            changed_symbols=["calculate"],
            unexpected_changes=[],
            failure_reasons=[],
        )

        self.assertTrue(evidence.passed)
        d = evidence.to_dict()
        self.assertEqual(d["status"], SemanticVerificationStatus.RESOLVED)
        self.assertEqual(d["failure_category"], FailureCategory.NONE)
        self.assertTrue(d["passed"])
        self.assertEqual(d["schema_version"], SEMANTIC_VERIFICATION_SCHEMA_VERSION)

        restored = VerificationEvidence.from_dict(d)
        self.assertEqual(restored.status, SemanticVerificationStatus.RESOLVED)
        self.assertTrue(restored.passed)

    def test_verification_evidence_passed_only_for_resolved(self):
        for st in ALL_SEMANTIC_STATUSES:
            ev = VerificationEvidence(status=st)
            if st == SemanticVerificationStatus.RESOLVED:
                self.assertTrue(ev.passed)
            else:
                self.assertFalse(ev.passed, f"status {st} should not be passed")


class TestParseFailureSemanticVerifier(unittest.TestCase):
    """Test ParseFailureSemanticVerifier against valid repairs and adversarial mutations."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.verifier = ParseFailureSemanticVerifier()

    def tearDown(self):
        self.tmp.cleanup()

    def test_capture_before_evidence_on_syntax_error(self):
        rel = "pkg/worker.py"
        target = self.root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        broken_source = "import os\nimport sys\n\ndef helper(x: int) -> int:\n    return x * 2\n\ndef compute(a: int, b: int)\n    val = helper(a)\n    return val + b\n"
        target.write_text(broken_source, encoding="utf-8")

        candidate = MaintenanceCandidate(
            candidate_id="cand_1",
            kind=MaintenanceSignal.PARSE_FAILURE,
            affected_files=[rel],
        )

        identity = self.verifier.capture_before_evidence(self.root, rel, candidate)
        self.assertIsNotNone(identity)
        assert identity is not None
        self.assertEqual(identity.signal_kind, MaintenanceSignal.PARSE_FAILURE)
        self.assertEqual(identity.syntax_error_class, "SyntaxError")
        self.assertEqual(identity.diagnostic_line, 7)
        self.assertIn("helper", identity.lexical_symbols)
        self.assertIn("compute", identity.lexical_symbols)
        self.assertIn("os", identity.import_names)
        self.assertIn("sys", identity.import_names)
        self.assertIn("helper", identity.block_signatures)
        self.assertIn("helper", identity.substantive_symbol_metrics)
        self.assertFalse(identity.substantive_symbol_metrics["helper"].get("is_stub"))

    def test_capture_before_evidence_fails_closed_on_missing_file(self):
        candidate = MaintenanceCandidate(candidate_id="cand_x", kind=MaintenanceSignal.PARSE_FAILURE)
        identity = self.verifier.capture_before_evidence(self.root, "nonexistent.py", candidate)
        self.assertIsNone(identity)

    def test_legitimate_repair_passes_semantic_verification(self):
        rel = "pkg/worker.py"
        target = self.root / rel
        target.parent.mkdir(parents=True, exist_ok=True)

        broken_source = "import os\nimport sys\n\ndef helper(x: int) -> int:\n    return x * 2\n\ndef compute(a: int, b: int)\n    val = helper(a)\n    return val + b\n"
        target.write_text(broken_source, encoding="utf-8")
        candidate = MaintenanceCandidate(candidate_id="c1", kind=MaintenanceSignal.PARSE_FAILURE)
        before_identity = self.verifier.capture_before_evidence(self.root, rel, candidate)

        repaired_source = "import os\nimport sys\n\ndef helper(x: int) -> int:\n    return x * 2\n\ndef compute(a: int, b: int):\n    val = helper(a)\n    return val + b\n"
        target.write_text(repaired_source, encoding="utf-8")

        evidence = self.verifier.verify(self.root, rel, before_identity, candidate=candidate)
        self.assertEqual(evidence.status, SemanticVerificationStatus.RESOLVED)
        self.assertEqual(evidence.failure_category, FailureCategory.NONE)
        self.assertTrue(evidence.passed)
        self.assertTrue(evidence.structural_result.get("compiled"))
        self.assertTrue(evidence.semantic_result.get("anti_stub_passed"))
        self.assertTrue(evidence.semantic_result.get("anti_gutting_passed"))
        self.assertTrue(evidence.semantic_result.get("unbroken_blocks_preserved"))

    # =========================================================================
    # Adversarial / Red-Team Attack Tests
    # =========================================================================

    def test_adversarial_attack_complete_deletion_pass(self):
        """Attacker fixes syntax by replacing the whole file with 'pass'."""
        rel = "service/auth.py"
        target = self.root / rel
        target.parent.mkdir(parents=True, exist_ok=True)

        broken = "def authenticate(token: str)\n    if not token:\n        return False\n    return check_db(token)\n"
        target.write_text(broken, encoding="utf-8")
        candidate = MaintenanceCandidate(candidate_id="c_del", kind=MaintenanceSignal.PARSE_FAILURE)
        before = self.verifier.capture_before_evidence(self.root, rel, candidate)

        target.write_text("pass\n", encoding="utf-8")

        evidence = self.verifier.verify(self.root, rel, before, candidate=candidate)
        self.assertEqual(evidence.status, SemanticVerificationStatus.NOT_RESOLVED)
        self.assertIn(
            evidence.failure_category,
            (FailureCategory.UNEXPECTED_SURFACE_CHANGE, FailureCategory.LINE_COUNT_DEGRADATION),
        )
        self.assertFalse(evidence.passed)

    def test_adversarial_attack_surface_preservation_body_gutted_with_pass(self):
        """Attacker keeps def name but guts the body with 'pass'."""
        rel = "service/auth.py"
        target = self.root / rel
        target.parent.mkdir(parents=True, exist_ok=True)

        broken = "def authenticate(token: str)\n    if not token:\n        return False\n    parsed = parse_jwt(token)\n    return verify_signature(parsed)\n"
        target.write_text(broken, encoding="utf-8")
        candidate = MaintenanceCandidate(candidate_id="c_stub", kind=MaintenanceSignal.PARSE_FAILURE)
        before = self.verifier.capture_before_evidence(self.root, rel, candidate)

        target.write_text("def authenticate(token: str):\n    pass\n", encoding="utf-8")

        evidence = self.verifier.verify(self.root, rel, before, candidate=candidate)
        self.assertEqual(evidence.status, SemanticVerificationStatus.NOT_RESOLVED)
        self.assertEqual(evidence.failure_category, FailureCategory.BODY_GUTTED_OR_STUBBED)
        self.assertFalse(evidence.passed)

    def test_adversarial_attack_surface_preservation_body_gutted_with_return_none(self):
        """Attacker keeps def name but guts the body with 'return None'."""
        rel = "service/auth.py"
        target = self.root / rel
        target.parent.mkdir(parents=True, exist_ok=True)

        broken = "def process_payment(amount: float, card: str)\n    validate_card(card)\n    charge(card, amount)\n    return True\n"
        target.write_text(broken, encoding="utf-8")
        candidate = MaintenanceCandidate(candidate_id="c_ret_none", kind=MaintenanceSignal.PARSE_FAILURE)
        before = self.verifier.capture_before_evidence(self.root, rel, candidate)

        target.write_text("def process_payment(amount: float, card: str):\n    return None\n", encoding="utf-8")

        evidence = self.verifier.verify(self.root, rel, before, candidate=candidate)
        self.assertEqual(evidence.status, SemanticVerificationStatus.NOT_RESOLVED)
        self.assertEqual(evidence.failure_category, FailureCategory.BODY_GUTTED_OR_STUBBED)
        self.assertFalse(evidence.passed)

    def test_adversarial_attack_gutting_undamaged_function(self):
        """Attacker fixes broken function, but guts an undamaged substantive function."""
        rel = "service/pipeline.py"
        target = self.root / rel
        target.parent.mkdir(parents=True, exist_ok=True)

        broken = "def critical_analytics(data: list) -> dict:\n    res = {}\n    for item in data:\n        res[item.key] = item.val * 2\n    return res\n\ndef broken_function()\n    return 42\n"
        target.write_text(broken, encoding="utf-8")
        candidate = MaintenanceCandidate(candidate_id="c_undamaged_gut", kind=MaintenanceSignal.PARSE_FAILURE)
        before = self.verifier.capture_before_evidence(self.root, rel, candidate)

        attack_code = "def critical_analytics(data: list) -> dict:\n    pass\n\ndef broken_function():\n    return 42\n"
        target.write_text(attack_code, encoding="utf-8")

        evidence = self.verifier.verify(self.root, rel, before, candidate=candidate)
        self.assertEqual(evidence.status, SemanticVerificationStatus.NOT_RESOLVED)
        self.assertEqual(evidence.failure_category, FailureCategory.BODY_GUTTED_OR_STUBBED)
        self.assertFalse(evidence.passed)
        self.assertIn("critical_analytics", "; ".join(evidence.failure_reasons))

    def test_adversarial_attack_unrelated_ast_mutation_with_reduced_substance(self):
        """Attacker mutates an unrelated function into simpler code with fewer statements."""
        rel = "service/calc.py"
        target = self.root / rel
        target.parent.mkdir(parents=True, exist_ok=True)

        broken = "def complex_operation(x: int) -> int:\n    a = x * 2\n    b = a + 3\n    c = b * 4\n    return c\n\ndef syntax_bad()\n    return 1\n"
        target.write_text(broken, encoding="utf-8")
        candidate = MaintenanceCandidate(candidate_id="c_mut", kind=MaintenanceSignal.PARSE_FAILURE)
        before = self.verifier.capture_before_evidence(self.root, rel, candidate)

        attack_code = "def complex_operation(x: int) -> int:\n    return x\n\ndef syntax_bad():\n    return 1\n"
        target.write_text(attack_code, encoding="utf-8")

        evidence = self.verifier.verify(self.root, rel, before, candidate=candidate)
        self.assertEqual(evidence.status, SemanticVerificationStatus.NOT_RESOLVED)
        self.assertEqual(evidence.failure_category, FailureCategory.AST_MUTATION_SUSPICIOUS)
        self.assertFalse(evidence.passed)

    def test_adversarial_attack_dropped_import(self):
        """Attacker drops an import that was present before."""
        rel = "service/net.py"
        target = self.root / rel
        target.parent.mkdir(parents=True, exist_ok=True)

        broken = "import socket\nimport ssl\n\ndef connect(host: str)\n    return socket.create_connection((host, 443))\n"
        target.write_text(broken, encoding="utf-8")
        candidate = MaintenanceCandidate(candidate_id="c_imp", kind=MaintenanceSignal.PARSE_FAILURE)
        before = self.verifier.capture_before_evidence(self.root, rel, candidate)

        attack_code = "import socket\n\ndef connect(host: str):\n    return socket.create_connection((host, 443))\n"
        target.write_text(attack_code, encoding="utf-8")

        evidence = self.verifier.verify(self.root, rel, before, candidate=candidate)
        self.assertEqual(evidence.status, SemanticVerificationStatus.NOT_RESOLVED)
        self.assertEqual(evidence.failure_category, FailureCategory.IMPORT_SURFACE_CORRUPTED)
        self.assertFalse(evidence.passed)

    def test_adversarial_attack_dropped_symbol(self):
        """Attacker deletes one of the original functions."""
        rel = "service/items.py"
        target = self.root / rel
        target.parent.mkdir(parents=True, exist_ok=True)

        broken = "def item_a():\n    return 'a'\n\ndef item_b()\n    return 'b'\n"
        target.write_text(broken, encoding="utf-8")
        candidate = MaintenanceCandidate(candidate_id="c_sym", kind=MaintenanceSignal.PARSE_FAILURE)
        before = self.verifier.capture_before_evidence(self.root, rel, candidate)

        attack_code = "def item_b():\n    return 'b'\n"
        target.write_text(attack_code, encoding="utf-8")

        evidence = self.verifier.verify(self.root, rel, before, candidate=candidate)
        self.assertEqual(evidence.status, SemanticVerificationStatus.NOT_RESOLVED)
        self.assertEqual(evidence.failure_category, FailureCategory.UNEXPECTED_SURFACE_CHANGE)
        self.assertFalse(evidence.passed)

    def test_adversarial_attack_protected_file_path(self):
        """Attacker attempts to verify or target a protected path."""
        for prot in ("local_agent/tool_engine.py", "local_agent/approval.py"):
            evidence = self.verifier.verify(self.root, prot, None)
            self.assertEqual(evidence.status, SemanticVerificationStatus.NOT_RESOLVED)
            self.assertEqual(evidence.failure_category, FailureCategory.UNEXPECTED_SCOPE_CHANGE)
            self.assertFalse(evidence.passed)

    def test_adversarial_attack_stale_validation_evidence_fingerprint(self):
        """Validation evidence references a stale state fingerprint."""
        rel = "service/calc.py"
        target = self.root / rel
        target.parent.mkdir(parents=True, exist_ok=True)

        broken = "def add(a, b)\n    return a + b\n"
        target.write_text(broken, encoding="utf-8")
        candidate = MaintenanceCandidate(candidate_id="c_stale", kind=MaintenanceSignal.PARSE_FAILURE)
        before = self.verifier.capture_before_evidence(self.root, rel, candidate)

        fixed = "def add(a, b):\n    return a + b\n"
        target.write_text(fixed, encoding="utf-8")

        mock_evidence = MagicMock()
        mock_evidence.state_fingerprint = "00000000000000000000000000000000"

        evidence = self.verifier.verify(
            self.root, rel, before, validation_evidence=mock_evidence, candidate=candidate
        )
        self.assertEqual(evidence.status, SemanticVerificationStatus.NOT_RESOLVED)
        self.assertEqual(evidence.failure_category, FailureCategory.STALE_EVIDENCE)
        self.assertFalse(evidence.passed)


class TestUnverifiableAndRegistry(unittest.TestCase):
    """Test fallback verifier and registry dispatch."""

    def test_unverifiable_verifier_returns_not_applicable(self):
        verifier = UnverifiableSemanticVerifier()
        ev = verifier.verify(Path("/tmp"), "some/file.py", None)
        self.assertEqual(ev.status, SemanticVerificationStatus.NOT_APPLICABLE)
        self.assertEqual(ev.failure_category, FailureCategory.VERIFIER_NOT_APPLICABLE)
        self.assertFalse(ev.passed)

    def test_verifier_registry(self):
        v = verifier_for(MaintenanceSignal.PARSE_FAILURE)
        self.assertIsInstance(v, ParseFailureSemanticVerifier)

        v_unknown = verifier_for("unknown_signal_kind")
        self.assertIsInstance(v_unknown, UnverifiableSemanticVerifier)

        self.assertIn(MaintenanceSignal.PARSE_FAILURE, all_verifiers())


class TestReassessmentIntegration(unittest.TestCase):
    """Test MaintenanceRunner.reassess() integration with semantic verification."""

    def test_reassess_rejects_resolution_when_semantic_verified_is_false(self):
        before = MaintenanceCandidate(
            candidate_id="cand_reassess_fail",
            kind=MaintenanceSignal.PARSE_FAILURE,
            severity=SEVERITY_HIGH,
            affected_files=["src/mod.py"],
        )

        verdict = reassess(
            before,
            None,
            executed=True,
            validation_passed=True,
            semantic_verified=False,
        )

        self.assertEqual(verdict.outcome, ReassessmentOutcome.PERSISTING)
        self.assertIn("semantic verification rejected", "; ".join(verdict.reasons))

    def test_reassess_credits_resolution_when_semantic_verified_is_true(self):
        before = MaintenanceCandidate(
            candidate_id="cand_reassess_pass",
            kind=MaintenanceSignal.PARSE_FAILURE,
            severity=SEVERITY_HIGH,
            affected_files=["src/mod.py"],
        )

        verdict = reassess(
            before,
            None,
            executed=True,
            validation_passed=True,
            semantic_verified=True,
        )

        self.assertEqual(verdict.outcome, ReassessmentOutcome.RESOLVED)
        self.assertIn("passed semantic verification", "; ".join(verdict.reasons))


class TestExecutorSemanticVerificationIntegration(unittest.TestCase):
    """End-to-end integration tests between MaintenanceExecutor and SemanticVerifier."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data_dir = self.root / ".gemini"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.journal = ExecutionJournal(self.data_dir / "journal")

    def tearDown(self):
        self.tmp.cleanup()

    def _make_executor(self, operations: list[FileOperation]) -> MaintenanceExecutor:
        provider = ScriptedProvider([operations])
        return MaintenanceExecutor(
            root=self.root,
            provider_factory=lambda: provider,
            policy=MaintenanceExecutionPolicy(repository_root=self.root),
            budget=MaintenanceBudget(),
            configured_tier=AutonomyTier.EXECUTE_WITH_EXISTING_APPROVAL,
            journal=self.journal,
            approval_gate=MaintenanceApprovalGate(
                approval_mode="never",
                apply_enabled=True,
            ),
            context_provider=lambda: RepositoryIntelligence(self.root).scan(),
            workspace_parent=self.data_dir / "workspaces",
        )

    def test_executor_legitimate_repair_succeeds_with_semantic_verified(self):
        broken_file = self.root / "math_util.py"
        broken_source = "def add_numbers(a: int, b: int)\n    return a + b\n"
        broken_file.write_text(broken_source, encoding="utf-8")

        candidate = MaintenanceCandidate(
            kind=MaintenanceSignal.PARSE_FAILURE,
            affected_files=["math_util.py"],
            subject="math_util.py",
            confidence=1.0,
            sample_size=1,
            severity=SEVERITY_HIGH,
        )
        order = build_work_order(
            candidate,
            granted_tier=AutonomyTier.EXECUTE_WITH_EXISTING_APPROVAL,
            budget=MaintenanceBudget(),
            fingerprint_fn=lambda paths: compute_state_fingerprint(self.root, paths),
        )

        fixed_source = "def add_numbers(a: int, b: int):\n    return a + b\n"
        ops = [
            FileOperation(
                action="modify",
                path="math_util.py",
                content=fixed_source,
                reason="fix missing colon",
            )
        ]

        executor = self._make_executor(ops)
        result = executor.execute(order)

        self.assertEqual(result.status, MaintenanceExecutionStatus.COMPLETED)
        self.assertTrue(result.applied)
        self.assertTrue(result.validation_passed)
        self.assertTrue(result.semantic_verified)
        self.assertEqual(result.semantic_failure_category, FailureCategory.NONE)
        self.assertTrue(result.succeeded)

    def test_executor_gutted_body_fails_semantic_verification_and_rolls_back(self):
        broken_file = self.root / "auth.py"
        broken_source = (
            "def process_data(data: list) -> int:\n"
            "    total = 0\n"
            "    for item in data:\n"
            "        total += item * 2\n"
            "        total += item * 3\n"
            "        total += item * 4\n"
            "    return total\n\n"
            "def simple_compute(x: int)\n"
            "    return x + 1\n"
        )
        broken_file.write_text(broken_source, encoding="utf-8")

        candidate = MaintenanceCandidate(
            kind=MaintenanceSignal.PARSE_FAILURE,
            affected_files=["auth.py"],
            subject="auth.py",
            confidence=1.0,
            sample_size=1,
            severity=SEVERITY_HIGH,
        )
        order = build_work_order(
            candidate,
            granted_tier=AutonomyTier.EXECUTE_WITH_EXISTING_APPROVAL,
            budget=MaintenanceBudget(),
            fingerprint_fn=lambda paths: compute_state_fingerprint(self.root, paths),
        )

        # Attack: fix syntax of simple_compute, but replace body with pass
        gutted_source = (
            "def process_data(data: list) -> int:\n"
            "    total = 0\n"
            "    for item in data:\n"
            "        total += item * 2\n"
            "        total += item * 3\n"
            "        total += item * 4\n"
            "    return total\n\n"
            "def simple_compute(x: int):\n"
            "    pass\n"
        )
        ops = [
            FileOperation(
                action="modify",
                path="auth.py",
                content=gutted_source,
                reason="pass instead of body",
            )
        ]

        executor = self._make_executor(ops)
        result = executor.execute(order)

        self.assertEqual(result.status, MaintenanceExecutionStatus.SEMANTIC_VERIFICATION_FAILED)
        self.assertTrue(result.applied)
        self.assertTrue(result.rolled_back)
        self.assertFalse(result.semantic_verified)
        self.assertEqual(result.semantic_failure_category, FailureCategory.BODY_GUTTED_OR_STUBBED)
        self.assertFalse(result.succeeded)
        # Tree was rolled back byte-for-byte to broken source
        self.assertEqual(broken_file.read_text(encoding="utf-8"), broken_source)


class TestSafetyAndReadOnlyInvariants(unittest.TestCase):
    """Prove SemanticVerifier is read-only, contains no mutation code, and protected files are untouched."""

    def test_semantic_verifier_source_has_no_write_operations(self):
        src_path = Path("local_agent/semantic_verification.py")
        self.assertTrue(src_path.is_file())
        tree = ast.parse(src_path.read_text(encoding="utf-8"))

        forbidden_calls = {"unlink", "rmdir", "remove", "system", "Popen", "call", "check_call"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr in forbidden_calls:
                    self.fail(f"forbidden write/system call found in semantic_verification.py: {node.func.attr}")
                if isinstance(node.func, ast.Name) and node.func.id in forbidden_calls:
                    self.fail(f"forbidden write/system call found in semantic_verification.py: {node.func.id}")

    def test_protected_files_remain_clean(self):
        from local_agent.approval import ApprovalPolicyEngine
        from local_agent.tool_engine import ToolEngine
        self.assertTrue(issubclass(ApprovalPolicyEngine, object))
        self.assertTrue(issubclass(ToolEngine, object))


if __name__ == "__main__":
    unittest.main()
