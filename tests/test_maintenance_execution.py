"""Phase 4.22 - narrow autonomous maintenance execution.

These tests are deliberately built the way the Phase 4.21 forensic audit said
they should have been:

* **Production-reachable fixtures only.** Every ``parse_failure`` candidate in
  this file is produced by running the *real*
  :class:`~local_agent.semantic_impact.SemanticGraph` and the *real*
  :class:`~local_agent.maintenance_analysis.MaintenanceAnalyzer` over a real
  temporary repository containing a real syntax error. Nothing hand-builds a
  candidate that discovery could never emit. Where a test genuinely needs a
  shape discovery cannot produce (a malformed work order, an unsupported
  signal), it says so and is classified as unit-level.
* **Observable proofs.** Safety invariants are asserted against bytes on disk,
  real subprocess exit codes and real persisted records - never against "was
  this method called".
* **Real mechanics.** The filesystem, the candidate workspace, ``compileall``,
  the git-free tree comparison and the persistence layer are all real. Only the
  LLM call boundary is mocked, because there is no way to make that real in a
  test.
"""

from __future__ import annotations

import ast
import concurrent.futures
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

from local_agent.evidence import compute_state_fingerprint
from local_agent.maintenance import (
    MaintenanceBudget,
    MaintenanceCandidate,
    MaintenanceSignal,
)
from local_agent.maintenance_analysis import MaintenanceAnalyzer
from local_agent.maintenance_execution import (
    MAX_SCOPE_FILES,
    NO_MUTATION_STATUSES,
    RETRYABLE_STATUSES,
    SUPPORTED_SIGNAL_KINDS,
    SUPPORTED_TIERS,
    ExecutionJournal,
    MaintenanceApprovalGate,
    MaintenanceExecutionStatus,
    MaintenanceExecutor,
)
from local_agent.maintenance_policy import (
    EXECUTING_TIERS,
    AutonomyTier,
    MaintenanceExecutionPolicy,
)
from local_agent.maintenance_runner import (
    MaintenanceExecutionOutcome,
    MaintenanceWorkOrder,
    build_work_order,
)
from local_agent.models import (
    FailureAnalysis,
    FileOperation,
    Plan,
    ProjectContext,
    ProviderCapability,
    ProviderError,
    ReviewResult,
)
from local_agent.providers import AIProvider
from local_agent.repository import RepositoryIntelligence
from local_agent.semantic_impact import SemanticGraph
from local_agent.validation_lifecycle import LifecycleState, ValidationLifecycleManager
from local_agent.validation_telemetry import ValidationTelemetryManager

REPO_ROOT = Path(__file__).resolve().parents[1]

# The exact defect the executor is built to repair, and its repair.
BROKEN_SOURCE = "def add(a, b)\n    return a + b\n"
FIXED_SOURCE = "def add(a, b):\n    return a + b\n"
#: A repair that is syntactically valid but semantically wrong - used to prove
#: post-apply validation is real and that a failing verdict rolls back.
WRONG_SOURCE = "def add(a, b):\n    return a - b\n"

MODULE_TEST = (
    "from broken import add\n"
    "\n"
    "\n"
    "def test_add():\n"
    "    assert add(2, 3) == 5\n"
)


# =============================================================================
# Fixture construction - through the real discovery path
# =============================================================================


def make_repo_with_parse_failure(root: Path, *, with_test: bool = True) -> None:
    """A real, tiny repository containing a real Python syntax error."""
    (root / "broken.py").write_text(BROKEN_SOURCE, encoding="utf-8")
    (root / "healthy.py").write_text("VALUE = 1\n", encoding="utf-8")
    if with_test:
        (root / "tests").mkdir(exist_ok=True)
        (root / "tests" / "test_broken.py").write_text(MODULE_TEST, encoding="utf-8")
    (root / "README.md").write_text("fixture\n", encoding="utf-8")


def discover_parse_failure(root: Path) -> MaintenanceCandidate:
    """Run the production discovery path and return its ``parse_failure``.

    This is the whole point of the fixture design: if this function stops
    returning a candidate, the signal genuinely is not production-reachable any
    more and every test resting on it should fail loudly rather than quietly
    testing a shape discovery cannot emit.
    """
    graph = SemanticGraph.build(root)
    analysis = MaintenanceAnalyzer(root).analyze(semantic_graph=graph)
    matches = [
        candidate
        for candidate in analysis.candidates
        if candidate.kind == MaintenanceSignal.PARSE_FAILURE
    ]
    if not matches:
        raise AssertionError(
            "the real analyzer produced no parse_failure candidate for this fixture"
        )
    return matches[0]


def snapshot_tree(root: Path) -> dict[str, bytes]:
    """Byte-for-byte snapshot, skipping caches and agent data so it is stable."""
    skip = {
        "__pycache__", ".pytest_cache", ".git", ".agent_data", ".agent_worktrees",
        ".mypy_cache", ".ruff_cache",
    }
    snapshot: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in skip for part in relative.parts):
            continue
        snapshot[relative.as_posix()] = path.read_bytes()
    return snapshot


def _git_init(root: Path) -> bool:
    """Make ``root`` a real git repository with one real commit.

    Returns False when git is unavailable, so a test can skip rather than
    silently assert something weaker.
    """
    import subprocess  # noqa: PLC0415 - test-only

    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    commands = [
        ["git", "init", "-q"],
        ["git", "config", "user.email", "test@example.invalid"],
        ["git", "config", "user.name", "Phase 4.22 fixture"],
        ["git", "add", "-A"],
        ["git", "-c", "commit.gpgsign=false", "commit", "-q", "-m", "fixture"],
    ]
    for command in commands:
        try:
            done = subprocess.run(
                command, cwd=root, capture_output=True, env=env, check=False
            )
        except OSError:
            return False
        if done.returncode != 0:
            return False
    return True


class ScriptedProvider(AIProvider):
    """Deterministic tool-use provider. The ONLY mocked boundary in this file."""

    def __init__(
        self,
        responses: list[Any] | None = None,
        *,
        supports_tools: bool = True,
        raise_exc: Exception | None = None,
    ):
        super().__init__()
        self.provider_id = "scripted"
        self.model = "scripted-v1"
        self.responses = list(responses or [])
        self.raise_exc = raise_exc
        self.calls = 0
        caps = {
            ProviderCapability.PLANNING,
            ProviderCapability.IMPLEMENTATION,
            ProviderCapability.REPAIR,
            ProviderCapability.REVIEW,
        }
        if supports_tools:
            caps.add(ProviderCapability.TOOL_USE)
        self._capabilities = caps

    @property
    def capabilities(self) -> set[ProviderCapability]:
        return self._capabilities

    def generate_plan(self, task, context) -> Plan:
        return Plan(objective="noop")

    def generate_code(self, task, plan, context, failure=None, review=None):
        return list(self.responses[0]) if self.responses else []

    def generate_code_with_tools(
        self, task, plan, context, tools, tool_history=None, failure=None, review=None
    ):
        self.calls += 1
        if self.raise_exc is not None:
            raise self.raise_exc
        if self.responses:
            return self.responses.pop(0)
        return []

    def review_changes(self, task, plan, diff, context) -> ReviewResult:
        return ReviewResult("APPROVED", "ok", [])

    def analyze_failure(self, execution, diff, context, plan) -> FailureAnalysis:
        return FailureAnalysis("boom", [], "fix it")


class LoopingProvider(ScriptedProvider):
    """Never concludes: asks for one more tool call every turn, forever.

    Only a genuinely enforced step budget can stop it, which is what makes it a
    real probe of the budget rather than a probe of the provider script.
    """

    def generate_code_with_tools(self, *args, **kwargs):
        from local_agent.models import ToolCall

        self.calls += 1
        return ToolCall(
            call_id=f"loop{self.calls}",
            tool_name="read_file_range",
            arguments={"path": "broken.py", "start": 1, "end": 1 + (self.calls % 2)},
        )


def fix_operation(path: str = "broken.py", content: str = FIXED_SOURCE) -> list[FileOperation]:
    return [
        FileOperation(
            action="modify",
            path=path,
            content=content,
            reason="repair the syntax error",
        )
    ]


class MemoryStorage:
    """Real in-memory implementation of the storage protocol the managers use.

    Not a mock: it round-trips through the very same ``to_dict``/``from_dict``
    the JSON backend uses, so a record that would not survive persistence does
    not survive here either.
    """

    def __init__(self) -> None:
        self._lifecycle: dict[str, Any] | None = None
        self._telemetry: dict[str, Any] | None = None

    def save_validation_lifecycle(self, store) -> None:
        self._lifecycle = json.loads(json.dumps(store.to_dict()))

    def load_validation_lifecycle(self):
        from local_agent.validation_lifecycle import ValidationLifecycleStore

        return ValidationLifecycleStore.from_dict(self._lifecycle or {})

    def save_validation_telemetry(self, store) -> None:
        self._telemetry = json.loads(json.dumps(store.to_dict()))

    def load_validation_telemetry(self):
        from local_agent.validation_telemetry import ValidationTelemetryStore

        return ValidationTelemetryStore.from_dict(self._telemetry or {})

    def save_maintenance(self, store) -> None:
        self._maintenance = json.loads(json.dumps(store.to_dict()))

    def load_maintenance(self):
        from local_agent.maintenance import MaintenanceStore

        return MaintenanceStore.from_dict(getattr(self, "_maintenance", None) or {})


class ExecutorCase(unittest.TestCase):
    """Base case: one real disposable repository per test."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="mx_base_")).resolve()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        make_repo_with_parse_failure(self.root)
        self.data_dir = Path(tempfile.mkdtemp(prefix="mx_data_")).resolve()
        self.addCleanup(shutil.rmtree, self.data_dir, ignore_errors=True)
        self.storage = MemoryStorage()
        self.cwd_before = os.getcwd()

    def tearDown(self) -> None:
        self.assertEqual(os.getcwd(), self.cwd_before, "process cwd was mutated")

    # -- builders ---------------------------------------------------------

    def candidate(self) -> MaintenanceCandidate:
        return discover_parse_failure(self.root)

    def order(
        self,
        candidate: MaintenanceCandidate | None = None,
        *,
        tier: str = AutonomyTier.EXECUTE_WITH_EXISTING_APPROVAL,
        budget: MaintenanceBudget | None = None,
        fingerprint: bool = True,
    ) -> MaintenanceWorkOrder:
        candidate = candidate or self.candidate()
        return build_work_order(
            candidate,
            granted_tier=tier,
            budget=budget or MaintenanceBudget(),
            configured_tier=tier,
            fingerprint_fn=(
                (lambda paths: compute_state_fingerprint(self.root, paths))
                if fingerprint
                else None
            ),
        )

    def executor(
        self,
        *,
        provider: Any = None,
        apply_enabled: bool = True,
        approval_mode: str = "never",
        approver=None,
        budget: MaintenanceBudget | None = None,
        tier: str = AutonomyTier.EXECUTE_WITH_EXISTING_APPROVAL,
        journal: ExecutionJournal | None = None,
        lifecycle: bool = True,
        policy_engine: Any = None,
        root: Path | None = None,
    ) -> MaintenanceExecutor:
        provider = provider if provider is not None else ScriptedProvider([fix_operation()])
        target = root or self.root
        return MaintenanceExecutor(
            root=target,
            provider_factory=lambda: provider,
            policy=MaintenanceExecutionPolicy(repository_root=target),
            budget=budget or MaintenanceBudget(),
            configured_tier=tier,
            journal=journal or ExecutionJournal(self.data_dir / "journal"),
            approval_gate=MaintenanceApprovalGate(
                approval_mode=approval_mode,
                policy_engine=policy_engine,
                approver=approver,
                apply_enabled=apply_enabled,
            ),
            context_provider=lambda: RepositoryIntelligence(target).scan(),
            lifecycle_manager=(
                ValidationLifecycleManager(self.storage, target) if lifecycle else None
            ),
            telemetry_manager=(
                ValidationTelemetryManager(self.storage, target) if lifecycle else None
            ),
            workspace_parent=self.data_dir / "workspaces",
        )


# =============================================================================
# A. Signal selection is real and narrow
# =============================================================================


class SignalSelectionTests(ExecutorCase):
    def test_the_supported_signal_is_produced_by_real_discovery(self):
        """PRODUCTION-INTEGRATION: the fixture comes from the real analyzer."""
        candidate = self.candidate()
        self.assertEqual(candidate.kind, MaintenanceSignal.PARSE_FAILURE)
        self.assertEqual(candidate.affected_files, ["broken.py"])
        self.assertEqual(candidate.sample_size, 1)
        self.assertEqual(candidate.confidence, 1.0)

    def test_exactly_one_signal_kind_is_supported(self):
        self.assertEqual(SUPPORTED_SIGNAL_KINDS, frozenset({MaintenanceSignal.PARSE_FAILURE}))

    def test_the_supported_tier_is_the_lowest_executing_one(self):
        self.assertEqual(
            SUPPORTED_TIERS, frozenset({AutonomyTier.EXECUTE_WITH_EXISTING_APPROVAL})
        )
        self.assertTrue(SUPPORTED_TIERS.issubset(EXECUTING_TIERS))
        self.assertNotIn(AutonomyTier.EXECUTE_AUTONOMOUSLY, SUPPORTED_TIERS)

    def test_the_real_policy_grants_the_supported_signal_an_executing_tier(self):
        """PRODUCTION-INTEGRATION: no fixture tuning, the real policy decides."""
        verdict = MaintenanceExecutionPolicy(repository_root=self.root).decide(
            self.candidate(),
            configured_tier=AutonomyTier.EXECUTE_AUTONOMOUSLY,
            budget=MaintenanceBudget(),
        )
        self.assertTrue(verdict.may_execute)
        self.assertEqual(verdict.granted_tier, AutonomyTier.EXECUTE_WITH_EXISTING_APPROVAL)

    def test_widening_the_supported_set_alone_does_not_open_the_gate(self):
        """A future signal cannot silently inherit this executor's authority.

        Two independent gates are proved here, in order, by defeating them one
        at a time. This is the defence-in-depth the specification's question T
        asks about, and Phase 4.23 added the first layer.

        Layer 1 (4.23): even with ``test_gap`` forced into the supported set,
        the executor asks the *oracle registry* whether the signal's oracle is
        deterministic. ``test_gap``'s is not, so it is refused as an
        unsupported signal before any file is read.

        Layer 2 (4.22): defeat layer 1 as well, by binding ``test_gap`` to the
        deterministic parse oracle, and the freshness gate still refuses -
        because ``healthy.py`` parses, so the oracle's failure predicate does
        not reproduce. A test gap says nothing about parseability.
        """
        import unittest.mock as mock

        from local_agent import maintenance_execution as module
        from local_agent.maintenance_oracle import ParseOracle

        candidate = MaintenanceCandidate(
            kind=MaintenanceSignal.TEST_GAP,
            subject="healthy.py",
            affected_files=["healthy.py"],
            confidence=1.0,
            severity="medium",
        )
        before = snapshot_tree(self.root)
        widened = frozenset(SUPPORTED_SIGNAL_KINDS | {MaintenanceSignal.TEST_GAP})

        with mock.patch.object(module, "SUPPORTED_SIGNAL_KINDS", widened):
            layer1 = self.executor().execute(self.order(candidate))
        self.assertEqual(
            layer1.status, MaintenanceExecutionStatus.UNSUPPORTED_SIGNAL, layer1.reasons
        )
        self.assertIn("not deterministic", " ".join(layer1.reasons))
        # Nothing was even observed: the refusal came before the target file
        # was read, which is what "structurally, before any workspace exists"
        # has to mean.
        self.assertIsNone(layer1.oracle_precondition)
        self.assertEqual(snapshot_tree(self.root), before)

        with mock.patch.object(module, "SUPPORTED_SIGNAL_KINDS", widened), \
                mock.patch.object(module, "oracle_for", lambda _kind: ParseOracle()):
            layer2 = self.executor().execute(self.order(candidate))
        self.assertNotEqual(layer2.status, MaintenanceExecutionStatus.COMPLETED)
        self.assertEqual(layer2.status, MaintenanceExecutionStatus.STALE_CANDIDATE)
        self.assertEqual(snapshot_tree(self.root), before)

    def test_the_policy_can_never_grant_the_signal_full_autonomy(self):
        """The unattended bar needs 5 samples; a structural fact has 1, always."""
        for _ in range(20):
            candidate = self.candidate()
            verdict = MaintenanceExecutionPolicy(repository_root=self.root).decide(
                candidate,
                configured_tier=AutonomyTier.EXECUTE_AUTONOMOUSLY,
                budget=MaintenanceBudget(),
            )
            self.assertNotEqual(verdict.granted_tier, AutonomyTier.EXECUTE_AUTONOMOUSLY)


# =============================================================================
# B. The happy path, end to end, against a real repository
# =============================================================================


class SuccessfulExecutionTests(ExecutorCase):
    def test_a_supported_signal_executes_applies_validates_and_resolves(self):
        """PRODUCTION-INTEGRATION: the full chain, real subprocesses included."""
        executor = self.executor()
        result = executor.execute(self.order())

        self.assertEqual(result.status, MaintenanceExecutionStatus.COMPLETED, result.reasons)
        # Observable proof, not a success flag: the real file changed on disk.
        self.assertEqual(
            (self.root / "broken.py").read_text(encoding="utf-8"), FIXED_SOURCE
        )
        # ... and it genuinely parses now.
        ast.parse((self.root / "broken.py").read_text(encoding="utf-8"))
        self.assertTrue(result.applied)
        self.assertFalse(result.rolled_back)
        self.assertIs(result.validation_passed, True)
        self.assertIs(result.prospective_validation_passed, True)
        self.assertIs(result.signal_resolved, True)
        self.assertTrue(result.succeeded)
        # A real post-apply command actually ran.
        self.assertGreaterEqual(result.post_apply_commands_run, 1)
        self.assertIn(
            ["python", "-m", "compileall", "-q", "broken.py"], result.post_apply_commands
        )

    def test_unrelated_files_are_untouched(self):
        before = snapshot_tree(self.root)
        self.executor().execute(self.order())
        after = snapshot_tree(self.root)
        for path, content in before.items():
            if path == "broken.py":
                continue
            self.assertEqual(after.get(path), content, f"{path} was modified")

    def test_the_signal_stops_being_discovered_afterwards(self):
        """RESOLVED is proved by a fresh real scan, not by the apply succeeding."""
        self.executor().execute(self.order())
        graph = SemanticGraph.build(self.root)
        self.assertEqual(graph.parse_failures, {})

    def test_lifecycle_reaches_completed_with_a_real_verdict(self):
        result = self.executor().execute(self.order())
        manager = ValidationLifecycleManager(self.storage, self.root)
        record = manager.get(result.lifecycle_id)
        self.assertIsNotNone(record)
        self.assertEqual(record.state, LifecycleState.COMPLETED)
        states = [entry["state"] for entry in record.state_history]
        for expected in (
            LifecycleState.CANDIDATE_GENERATED,
            LifecycleState.VALIDATED,
            LifecycleState.APPROVED,
            LifecycleState.APPLIED,
            LifecycleState.POST_VALIDATED,
            LifecycleState.COMPLETED,
        ):
            self.assertIn(expected, states)
        self.assertTrue(record.iterations)
        self.assertEqual(record.iterations[-1].validation_result, "passed")
        self.assertEqual(record.iterations[-1].validation_stage, "post_apply")

    def test_evidence_is_recorded_for_a_successful_validation(self):
        result = self.executor().execute(self.order())
        self.assertGreaterEqual(result.evidence_recorded, 1)

    def test_telemetry_records_a_real_decision(self):
        result = self.executor().execute(self.order())
        store = self.storage.load_validation_telemetry()
        self.assertTrue(store.decisions)
        self.assertEqual(result.decision_id, store.decisions[-1].decision_id)

    def test_the_outcome_adapter_reports_the_real_verdict(self):
        outcome = self.executor()(self.order())
        self.assertIsInstance(outcome, MaintenanceExecutionOutcome)
        self.assertTrue(outcome.succeeded)
        self.assertIs(outcome.validation_passed, True)
        self.assertEqual(outcome.changed_files, ["broken.py"])


# =============================================================================
# C. Refusals - each one must leave the tree byte-identical
# =============================================================================


class RefusalTests(ExecutorCase):
    def assert_tree_unchanged(self, before: dict[str, bytes]) -> None:
        self.assertEqual(snapshot_tree(self.root), before)

    def test_an_unsupported_signal_is_rejected(self):
        """UNIT-LEVEL: discovery cannot emit a test_gap for this fixture, so the
        candidate is hand-built - and that is exactly the point of the test."""
        before = snapshot_tree(self.root)
        candidate = MaintenanceCandidate(
            kind=MaintenanceSignal.TEST_GAP,
            subject="broken.py",
            affected_files=["broken.py"],
            confidence=1.0,
            severity="medium",
        )
        result = self.executor().execute(self.order(candidate))
        self.assertEqual(result.status, MaintenanceExecutionStatus.UNSUPPORTED_SIGNAL)
        self.assert_tree_unchanged(before)

    def test_a_protected_path_is_rejected(self):
        """UNIT-LEVEL fixture, real protected-path rule.

        Protected paths live under ``local_agent/`` and cannot appear in this
        temp fixture, so the candidate names one directly. The refusal comes
        from the production ``MaintenanceExecutionPolicy``.
        """
        before = snapshot_tree(self.root)
        (self.root / "local_agent").mkdir(exist_ok=True)
        (self.root / "local_agent" / "tool_engine.py").write_text("x = (\n", encoding="utf-8")
        candidate = MaintenanceCandidate(
            kind=MaintenanceSignal.PARSE_FAILURE,
            subject="local_agent/tool_engine.py",
            affected_files=["local_agent/tool_engine.py"],
            confidence=1.0,
        )
        order = self.order(candidate)
        before = snapshot_tree(self.root)
        result = self.executor().execute(order)
        self.assertEqual(result.status, MaintenanceExecutionStatus.REFUSED_BY_POLICY)
        self.assert_tree_unchanged(before)

    def test_protected_path_matching_ignores_case(self):
        """The exact Phase 4.21 defect: do not reintroduce case sensitivity."""
        candidate = MaintenanceCandidate(
            kind=MaintenanceSignal.PARSE_FAILURE,
            subject="Local_Agent/Tool_Engine.py",
            affected_files=["Local_Agent/Tool_Engine.py"],
            confidence=1.0,
        )
        result = self.executor().execute(self.order(candidate))
        self.assertEqual(result.status, MaintenanceExecutionStatus.REFUSED_BY_POLICY)

    def test_a_stale_candidate_is_rejected_when_the_file_changed(self):
        before_order = self.order()
        # A third party fixes the file between planning and execution.
        (self.root / "broken.py").write_text(FIXED_SOURCE, encoding="utf-8")
        before = snapshot_tree(self.root)
        result = self.executor().execute(before_order)
        self.assertEqual(result.status, MaintenanceExecutionStatus.STALE_CANDIDATE)
        self.assert_tree_unchanged(before)

    def test_a_stale_candidate_is_rejected_even_without_a_fingerprint(self):
        """The signal-level re-check stands alone when no fingerprint exists."""
        order = self.order(fingerprint=False)
        (self.root / "broken.py").write_text(FIXED_SOURCE, encoding="utf-8")
        result = self.executor().execute(order)
        self.assertEqual(result.status, MaintenanceExecutionStatus.STALE_CANDIDATE)
        # Phase 4.23: the re-check is the oracle's failure predicate rather
        # than an inline compile, so the wording changed. The substance that
        # matters - the refusal names the oracle and says the failure was not
        # re-observed - is asserted here, along with the recorded observation
        # itself so this is not merely a string test.
        self.assertTrue(
            any(
                "did not re-observe the failure" in reason for reason in result.reasons
            ),
            result.reasons,
        )
        self.assertIsNotNone(result.oracle_precondition)
        self.assertEqual(result.oracle_precondition["outcome"], "resolved")
        self.assertEqual(result.oracle_precondition["oracle"], "parse_oracle")

    def test_a_vanished_file_is_rejected(self):
        order = self.order()
        (self.root / "broken.py").unlink()
        result = self.executor().execute(order)
        self.assertEqual(result.status, MaintenanceExecutionStatus.STALE_CANDIDATE)

    def test_a_malformed_work_order_is_rejected(self):
        """UNIT-LEVEL: these shapes cannot come from ``build_work_order``."""
        before = snapshot_tree(self.root)
        executor = self.executor()
        for order in (
            MaintenanceWorkOrder(),  # no candidate id
            MaintenanceWorkOrder(candidate_id="abc", scope_files=[]),
            MaintenanceWorkOrder(candidate_id="abc", scope_files=["a.py", "b.py"]),
            MaintenanceWorkOrder(candidate_id="abc", scope_files=["notes.md"]),
            MaintenanceWorkOrder(candidate_id="abc", scope_files=["broken.py"]),
        ):
            result = executor.execute(order)
            self.assertEqual(
                result.status,
                MaintenanceExecutionStatus.MALFORMED_WORK_ORDER,
                f"{order.scope_files!r} should have been refused",
            )
        self.assert_tree_unchanged(before)

    def test_a_snapshot_that_disagrees_with_the_order_is_rejected(self):
        order = self.order()
        order.candidate_snapshot = dict(order.candidate_snapshot)
        order.candidate_snapshot["candidate_id"] = "tampered"
        result = self.executor().execute(order)
        self.assertEqual(result.status, MaintenanceExecutionStatus.MALFORMED_WORK_ORDER)

    def test_a_snapshot_whose_scope_disagrees_is_rejected(self):
        order = self.order()
        order.candidate_snapshot = dict(order.candidate_snapshot)
        order.candidate_snapshot["affected_files"] = ["healthy.py"]
        result = self.executor().execute(order)
        self.assertEqual(result.status, MaintenanceExecutionStatus.MALFORMED_WORK_ORDER)

    def test_corrupted_persisted_state_cannot_produce_a_permissive_decision(self):
        """A hostile snapshot claiming maximum evidence still cannot widen the tier."""
        order = self.order()
        order.candidate_snapshot = dict(order.candidate_snapshot)
        order.candidate_snapshot.update(
            {
                "confidence": 9_999,
                "sample_size": 10_000,
                "occurrence_count": 10_000,
                "severity": "catastrophic",
            }
        )
        order.granted_tier = AutonomyTier.EXECUTE_AUTONOMOUSLY
        result = self.executor().execute(order)
        # The forged claims would grant EXECUTE_AUTONOMOUSLY, which this
        # executor does not act at - so it is refused, not honoured.
        self.assertEqual(result.status, MaintenanceExecutionStatus.REFUSED_BY_POLICY)
        self.assertEqual((self.root / "broken.py").read_text(encoding="utf-8"), BROKEN_SOURCE)

    def test_a_blocked_candidate_is_refused_by_the_policy_recheck(self):
        candidate = self.candidate()
        candidate.failure_count = 5
        order = self.order(candidate)
        result = self.executor().execute(order)
        self.assertEqual(result.status, MaintenanceExecutionStatus.REFUSED_BY_POLICY)

    def test_a_configured_tier_below_execution_is_refused(self):
        order = self.order(tier=AutonomyTier.PLAN_ONLY)
        result = self.executor(tier=AutonomyTier.PLAN_ONLY).execute(order)
        self.assertEqual(result.status, MaintenanceExecutionStatus.REFUSED_BY_POLICY)
        self.assertEqual((self.root / "broken.py").read_text(encoding="utf-8"), BROKEN_SOURCE)


# =============================================================================
# D. Budgets that are actually enforced
# =============================================================================


class BudgetEnforcementTests(ExecutorCase):
    def test_a_zero_tool_step_budget_prevents_execution(self):
        budget = MaintenanceBudget(max_tool_steps_per_subtask=0)
        result = self.executor(budget=budget).execute(self.order(budget=budget))
        self.assertEqual(result.status, MaintenanceExecutionStatus.BUDGET_EXHAUSTED)
        self.assertEqual((self.root / "broken.py").read_text(encoding="utf-8"), BROKEN_SOURCE)

    def test_a_zero_changed_file_budget_prevents_execution(self):
        """The *policy* gets there first, which is the stricter of the two gates.

        With a zero changed-file budget the candidate's single affected file
        already exceeds it, so ``MaintenanceExecutionPolicy`` caps the tier at
        ``plan_only`` before the executor's own budget check runs. Asserting
        ``budget_exhausted`` here would encode the wrong layer; what matters is
        that no execution happens and nothing is written.
        """
        budget = MaintenanceBudget(max_changed_files_per_candidate=0)
        before = snapshot_tree(self.root)
        result = self.executor(budget=budget).execute(self.order(budget=budget))
        self.assertEqual(result.status, MaintenanceExecutionStatus.REFUSED_BY_POLICY)
        self.assertFalse(result.applied)
        self.assertEqual(snapshot_tree(self.root), before)

    def test_a_zero_validation_command_budget_prevents_execution(self):
        budget = MaintenanceBudget(max_validation_commands=0)
        result = self.executor(budget=budget).execute(self.order(budget=budget))
        self.assertEqual(result.status, MaintenanceExecutionStatus.BUDGET_EXHAUSTED)

    def test_a_zero_diff_line_budget_prevents_execution(self):
        budget = MaintenanceBudget(max_changed_lines_per_candidate=0)
        result = self.executor(budget=budget).execute(self.order(budget=budget))
        self.assertEqual(result.status, MaintenanceExecutionStatus.BUDGET_EXHAUSTED)

    def test_the_diff_size_budget_is_enforced_against_a_real_diff(self):
        """A tiny-but-nonzero diff budget refuses a real oversized repair."""
        budget = MaintenanceBudget(max_changed_lines_per_candidate=2)
        # A repair that also appends 200 lines: syntactically valid, far too big.
        oversized = FIXED_SOURCE + "".join(f"CONST_{i} = {i}\n" for i in range(200))
        provider = ScriptedProvider([fix_operation(content=oversized)])
        before = snapshot_tree(self.root)
        result = self.executor(provider=provider, budget=budget).execute(
            self.order(budget=budget)
        )
        self.assertEqual(result.status, MaintenanceExecutionStatus.BUDGET_EXHAUSTED)
        self.assertGreater(result.diff_lines, 2)
        self.assertEqual(snapshot_tree(self.root), before)

    def test_the_tool_step_budget_actually_stops_a_looping_agent(self):
        """The bound is not merely copied: it cuts a real tool loop short.

        The provider here never returns file operations - it asks for another
        tool call every turn, forever. Only a genuinely enforced step budget can
        stop it, and the tree must be untouched when it does.
        """
        before = snapshot_tree(self.root)
        budget = MaintenanceBudget(max_tool_steps_per_subtask=3)
        provider = LoopingProvider()
        result = self.executor(provider=provider, budget=budget).execute(
            self.order(budget=budget)
        )
        self.assertGreater(result.tool_steps_used, 0, "the loop must actually have run")
        self.assertLessEqual(result.tool_steps_used, 3)
        self.assertNotEqual(result.status, MaintenanceExecutionStatus.COMPLETED)
        self.assertEqual(snapshot_tree(self.root), before)

    def test_an_exhausted_deadline_prevents_execution(self):
        executor = self.executor()
        executor.deadline = lambda: True
        result = executor.execute(self.order())
        self.assertEqual(result.status, MaintenanceExecutionStatus.BUDGET_EXHAUSTED)
        self.assertEqual((self.root / "broken.py").read_text(encoding="utf-8"), BROKEN_SOURCE)

    def test_only_the_declared_budgets_are_consumed_by_the_executor(self):
        """Honesty check: a budget this phase does not enforce must not look enforced.

        Proved from the AST rather than from prose, so the claim in the module's
        documentation cannot drift away from the code.

        PHASE 4.23 CHANGE, disclosed deliberately. The 4.22 version of this
        test used "the name does not appear anywhere in the module" as its
        proxy for "not enforced", and asserted that for seven budget fields.
        That proxy became wrong, not because enforcement changed, but because
        4.23 added
        :data:`~local_agent.maintenance_execution.UNENFORCED_BUDGET_FIELDS` - a
        declaration whose entire purpose is to name those fields and say, in
        the module itself, that they are not enforced. Deleting the mention to
        satisfy the old proxy would have removed the documentation the
        specification asks for.

        The contract asserted here is therefore the underlying one, and it is
        strictly stronger than a name search: an unenforced budget may never be
        *read* (no ``budget.max_x`` attribute access, no string key handed to
        the ledger), and where its name appears at all it must appear as a key
        of the declaration that disclaims it.
        """
        import ast as _ast

        from local_agent.maintenance_execution import (
            ENFORCED_BUDGET_FIELDS,
            UNENFORCED_BUDGET_FIELDS,
        )

        source = (REPO_ROOT / "local_agent" / "maintenance_execution.py").read_text(
            encoding="utf-8"
        )
        tree = _ast.parse(source)
        attributes = {
            node.attr
            for node in _ast.walk(tree)
            if isinstance(node, _ast.Attribute) and node.attr.startswith("max_")
        }
        constants = {
            node.value
            for node in _ast.walk(tree)
            if isinstance(node, _ast.Constant)
            and isinstance(node.value, str)
            and node.value.startswith("max_")
        }
        read = attributes | constants

        # Everything the executor claims to enforce, it really does touch.
        # ``max_candidates_executed`` is enforced through the policy verdict
        # rather than here, so it is checked in the policy module instead.
        directly_enforced = set(ENFORCED_BUDGET_FIELDS) - {"max_candidates_executed"}
        self.assertTrue(
            directly_enforced.issubset(read), sorted(directly_enforced - read)
        )
        policy_source = (REPO_ROOT / "local_agent" / "maintenance_policy.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("max_candidates_executed", policy_source)

        # Nothing it disclaims is ever read, and every mention of one is inside
        # the disclaimer itself.
        for not_enforced in sorted(UNENFORCED_BUDGET_FIELDS):
            self.assertNotIn(not_enforced, attributes, not_enforced)
            if not_enforced in constants:
                self.assertIn(not_enforced, UNENFORCED_BUDGET_FIELDS, not_enforced)

    def test_all_skipped_validation_commands_produce_no_verdict(self):
        """UNIT-LEVEL: the only way to make every runner unavailable at once.

        A skipped command is not evidence. When nothing actually executes there
        is no verdict, and "no verdict" must never become COMPLETED.
        """
        import unittest.mock as mock

        from local_agent import maintenance_execution as module
        from local_agent.models import ExecutionResult

        def all_missing(self, spec):
            return ExecutionResult(
                " ".join(spec.command), 127, "", "executable not found: x"
            )

        with mock.patch.object(module.CommandRunner, "run", all_missing):
            result = self.executor().execute(self.order())

        self.assertEqual(result.status, MaintenanceExecutionStatus.POST_VALIDATION_FAILED)
        self.assertIsNone(result.validation_passed)
        self.assertEqual(result.post_apply_commands_run, 0)
        self.assertTrue(result.rolled_back)
        self.assertEqual((self.root / "broken.py").read_text(encoding="utf-8"), BROKEN_SOURCE)

    def test_the_effective_bound_is_the_minimum_never_the_order(self):
        """A work order asking for more than the budget cannot widen it."""
        budget = MaintenanceBudget(max_tool_steps_per_subtask=2)
        order = self.order(budget=budget)
        order.max_tool_steps = 9_999
        result = self.executor(provider=LoopingProvider(), budget=budget).execute(order)
        self.assertGreater(result.tool_steps_used, 0)
        self.assertLessEqual(result.tool_steps_used, 2)

    def test_the_executor_ceiling_binds_even_when_the_budget_is_huge(self):
        """The hard-coded ceiling is a floor on safety, not a formality."""
        from local_agent.maintenance_execution import MAX_TOOL_STEPS_CEILING

        budget = MaintenanceBudget(max_tool_steps_per_subtask=100_000)
        order = self.order(budget=budget)
        order.max_tool_steps = 100_000
        result = self.executor(provider=LoopingProvider(), budget=budget).execute(order)
        self.assertLessEqual(result.tool_steps_used, MAX_TOOL_STEPS_CEILING)


# =============================================================================
# E. Idempotency and concurrency
# =============================================================================


class IdempotencyTests(ExecutorCase):
    def test_the_same_candidate_and_state_does_not_execute_twice(self):
        journal = ExecutionJournal(self.data_dir / "journal")
        order = self.order()
        first = self.executor(journal=journal).execute(order)
        self.assertEqual(first.status, MaintenanceExecutionStatus.COMPLETED)
        applied_bytes = (self.root / "broken.py").read_bytes()

        # Replay the exact same (now stale) order.
        second = self.executor(journal=journal).execute(order)
        self.assertIn(
            second.status,
            {
                MaintenanceExecutionStatus.STALE_CANDIDATE,
                MaintenanceExecutionStatus.DUPLICATE_EXECUTION,
            },
        )
        self.assertEqual((self.root / "broken.py").read_bytes(), applied_bytes)

    def test_a_replayed_order_against_an_unchanged_tree_is_a_duplicate(self):
        """Isolate the journal from the freshness gate: refuse the second run."""
        journal = ExecutionJournal(self.data_dir / "journal")
        order = self.order()
        # A provider that produces nothing leaves the tree untouched, so the
        # freshness gate still passes on the replay and the journal is the only
        # thing that can stop it.
        first = self.executor(journal=journal, provider=ScriptedProvider([[]])).execute(order)
        self.assertEqual(first.status, MaintenanceExecutionStatus.NO_CHANGE)
        second = self.executor(journal=journal, provider=ScriptedProvider([[]])).execute(order)
        self.assertEqual(second.status, MaintenanceExecutionStatus.DUPLICATE_EXECUTION)

    def test_the_same_candidate_after_a_repository_change_gets_a_fresh_key(self):
        journal = ExecutionJournal(self.data_dir / "journal")
        order = self.order()
        self.executor(journal=journal, provider=ScriptedProvider([[]])).execute(order)
        # A different, still-broken content: a new tree state, so a new key.
        (self.root / "broken.py").write_text("def add(a, b)\n    return b + a\n", encoding="utf-8")
        fresh_order = self.order()
        second = self.executor(journal=journal, provider=ScriptedProvider([[]])).execute(
            fresh_order
        )
        self.assertEqual(second.status, MaintenanceExecutionStatus.NO_CHANGE)
        self.assertNotEqual(second.execution_key, order.scope_fingerprint)

    def test_a_crashed_claim_permanently_refuses_that_key(self):
        """Fail closed: a claim with no completion is never silently retried.

        Simulates a crash between apply and persistence by claiming exactly the
        key the executor would claim and never completing it.
        """
        journal = ExecutionJournal(self.data_dir / "journal")
        order = self.order()
        executor = self.executor(journal=journal)
        key = executor._execution_key(
            order,
            MaintenanceCandidate.from_dict(order.candidate_snapshot),
            order.scope_files[0],
        )
        self.assertTrue(journal.claim(key))
        self.assertEqual(journal.status_of(key), "in_progress")

        before = snapshot_tree(self.root)
        result = executor.execute(order)
        self.assertEqual(result.status, MaintenanceExecutionStatus.DUPLICATE_EXECUTION)
        self.assertEqual(snapshot_tree(self.root), before)
        # The claim is still in_progress: a crashed run is never quietly closed.
        self.assertEqual(journal.status_of(key), "in_progress")

    def test_a_retryable_failure_releases_the_claim(self):
        journal = ExecutionJournal(self.data_dir / "journal")
        order = self.order()
        failing = ScriptedProvider(raise_exc=ProviderError("rate limited"))
        first = self.executor(journal=journal, provider=failing).execute(order)
        self.assertEqual(first.status, MaintenanceExecutionStatus.PROVIDER_FAILURE)
        self.assertTrue(first.retryable)
        # The claim was released, so a genuine retry is possible.
        second = self.executor(journal=journal).execute(order)
        self.assertEqual(second.status, MaintenanceExecutionStatus.COMPLETED)

    def test_concurrent_invocation_executes_at_most_once(self):
        journal = ExecutionJournal(self.data_dir / "journal")
        order = self.order()
        executors = [
            self.executor(journal=journal, provider=ScriptedProvider([fix_operation()]))
            for _ in range(4)
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda ex: ex.execute(order), executors))
        completed = [r for r in results if r.status == MaintenanceExecutionStatus.COMPLETED]
        self.assertLessEqual(len(completed), 1, [r.status for r in results])
        self.assertEqual(
            (self.root / "broken.py").read_text(encoding="utf-8"),
            FIXED_SOURCE if completed else BROKEN_SOURCE,
        )

    def test_a_refused_approval_releases_the_claim_so_a_later_run_can_apply(self):
        """Regression: a post-claim refusal must hand the journal key back.

        The first draft of the executor returned early from every post-claim
        refusal without the key, so the claim stayed ``in_progress`` forever and
        an ``--apply`` run after an ``--execute``-only run was refused as a
        duplicate. That would have made the two-step operator workflow the CLI
        advertises impossible.
        """
        journal = ExecutionJournal(self.data_dir / "journal")
        order = self.order()
        first = self.executor(journal=journal, apply_enabled=False).execute(order)
        self.assertEqual(first.status, MaintenanceExecutionStatus.APPROVAL_REQUIRED)
        self.assertEqual(journal.status_of(first.execution_key), "")

        second = self.executor(journal=journal, apply_enabled=True).execute(order)
        self.assertEqual(second.status, MaintenanceExecutionStatus.COMPLETED, second.reasons)
        self.assertEqual((self.root / "broken.py").read_text(encoding="utf-8"), FIXED_SOURCE)

    def test_a_diff_budget_refusal_releases_the_claim(self):
        journal = ExecutionJournal(self.data_dir / "journal")
        budget = MaintenanceBudget(max_changed_lines_per_candidate=2)
        oversized = FIXED_SOURCE + "".join(f"C{i} = {i}\n" for i in range(100))
        result = self.executor(
            journal=journal,
            budget=budget,
            provider=ScriptedProvider([fix_operation(content=oversized)]),
        ).execute(self.order(budget=budget))
        self.assertEqual(result.status, MaintenanceExecutionStatus.BUDGET_EXHAUSTED)
        self.assertEqual(journal.status_of(result.execution_key), "")

    def test_a_completed_execution_keeps_its_claim(self):
        journal = ExecutionJournal(self.data_dir / "journal")
        result = self.executor(journal=journal).execute(self.order())
        self.assertEqual(result.status, MaintenanceExecutionStatus.COMPLETED)
        self.assertEqual(journal.status_of(result.execution_key), "completed")

    def test_an_unusable_journal_fails_closed(self):
        """A journal that cannot record a claim must refuse, never wave it through."""
        blocked = self.data_dir / "blocked-journal"
        blocked.write_text("this is a file, not a directory", encoding="utf-8")
        journal = ExecutionJournal(blocked)
        self.assertFalse(journal.claim("anything"))

        before = snapshot_tree(self.root)
        result = self.executor(journal=journal).execute(self.order())
        self.assertEqual(result.status, MaintenanceExecutionStatus.DUPLICATE_EXECUTION)
        self.assertEqual(snapshot_tree(self.root), before)

    def test_journal_claim_is_atomic(self):
        journal = ExecutionJournal(self.data_dir / "journal")
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            outcomes = list(pool.map(lambda _: journal.claim("same-key"), range(8)))
        self.assertEqual(sum(1 for won in outcomes if won), 1)


# =============================================================================
# F. Failure semantics - the tree must be unchanged or restored
# =============================================================================


class FailureSemanticsTests(ExecutorCase):
    def test_a_provider_failure_leaves_the_tree_unchanged(self):
        before = snapshot_tree(self.root)
        provider = ScriptedProvider(raise_exc=ProviderError("all providers exhausted"))
        result = self.executor(provider=provider).execute(self.order())
        self.assertEqual(result.status, MaintenanceExecutionStatus.PROVIDER_FAILURE)
        self.assertFalse(result.applied)
        self.assertIsNone(result.validation_passed)
        self.assertEqual(snapshot_tree(self.root), before)

    def test_a_rate_limit_is_a_provider_failure_not_a_silent_retry(self):
        from local_agent.models import RateLimitError

        provider = ScriptedProvider(raise_exc=RateLimitError("429"))
        result = self.executor(provider=provider).execute(self.order())
        self.assertEqual(result.status, MaintenanceExecutionStatus.PROVIDER_FAILURE)
        self.assertEqual(provider.calls, 1, "the executor must not retry the provider itself")

    def test_prospective_validation_failure_leaves_the_tree_unchanged(self):
        """A repair that still does not parse fails inside the candidate tree."""
        before = snapshot_tree(self.root)
        still_broken = "def add(a, b)\n    return a + b  # still missing the colon\n"
        provider = ScriptedProvider(
            [fix_operation(content=still_broken)] * 4
        )
        result = self.executor(provider=provider).execute(self.order())
        self.assertIn(
            result.status,
            {
                MaintenanceExecutionStatus.PROSPECTIVE_VALIDATION_FAILED,
                MaintenanceExecutionStatus.IMPLEMENTATION_FAILURE,
            },
        )
        self.assertFalse(result.applied)
        self.assertEqual(snapshot_tree(self.root), before)

    def test_a_behaviourally_wrong_repair_is_caught_before_it_is_applied(self):
        """PRODUCTION-INTEGRATION: real pytest inside the candidate tree.

        The proposed repair parses - so the candidate's syntax tier passes - but
        breaks ``add()``. The repository's real test runs against the *candidate*
        copy, fails for real, and the change never reaches the authoritative
        tree at all.
        """
        before = snapshot_tree(self.root)
        provider = ScriptedProvider([fix_operation(content=WRONG_SOURCE)] * 6)
        result = self.executor(provider=provider).execute(self.order())
        self.assertEqual(
            result.status, MaintenanceExecutionStatus.PROSPECTIVE_VALIDATION_FAILED
        )
        self.assertIs(result.prospective_validation_passed, False)
        self.assertFalse(result.applied)
        self.assertIsNone(result.validation_passed)
        self.assertEqual(snapshot_tree(self.root), before)

    def test_post_apply_validation_failure_rolls_the_change_back(self):
        """PRODUCTION-INTEGRATION: a real failing post-apply pytest reverts a real apply.

        Constructing this needs a defect the candidate stage genuinely *cannot*
        see, otherwise prospective validation catches it first and the rollback
        path is never reached. The one used here is real and is exactly the class
        of failure prospective validation is known not to cover: an
        environmental difference between the candidate mirror and the
        authoritative tree. ``CandidateWorkspace`` deliberately does not mirror
        ``.agent_data``, so a test asserting on its absence passes in the
        candidate and fails for real after the apply.

        Everything here is real: real pytest, real exit codes, a real apply and
        a real rollback verified byte-for-byte.
        """
        (self.root / ".agent_data").mkdir(exist_ok=True)
        (self.root / ".agent_data" / "marker.txt").write_text("present", encoding="utf-8")
        (self.root / "tests" / "test_broken.py").write_text(
            MODULE_TEST
            + "\n\n"
            "def test_environment():\n"
            "    import os\n"
            "    assert not os.path.isdir('.agent_data')\n",
            encoding="utf-8",
        )
        before = snapshot_tree(self.root)
        result = self.executor().execute(self.order())

        self.assertEqual(result.status, MaintenanceExecutionStatus.POST_VALIDATION_FAILED)
        self.assertTrue(result.applied, "the change must genuinely have been applied first")
        self.assertTrue(result.rolled_back)
        self.assertIs(result.validation_passed, False)
        self.assertFalse(result.succeeded)
        # The authoritative tree is byte-identical to its pre-apply state.
        self.assertEqual(snapshot_tree(self.root), before)
        self.assertEqual(
            (self.root / "broken.py").read_text(encoding="utf-8"), BROKEN_SOURCE
        )

    def test_a_failed_post_apply_validation_never_reaches_completed_lifecycle(self):
        """The Phase 4.21 defect, re-proved for the new executor path."""
        (self.root / ".agent_data").mkdir(exist_ok=True)
        (self.root / "tests" / "test_broken.py").write_text(
            MODULE_TEST
            + "\n\n"
            "def test_environment():\n"
            "    import os\n"
            "    assert not os.path.isdir('.agent_data')\n",
            encoding="utf-8",
        )
        result = self.executor().execute(self.order())
        self.assertEqual(result.status, MaintenanceExecutionStatus.POST_VALIDATION_FAILED)
        record = ValidationLifecycleManager(self.storage, self.root).get(result.lifecycle_id)
        self.assertIsNotNone(record)
        self.assertNotEqual(record.state, LifecycleState.COMPLETED)
        self.assertEqual(record.state, LifecycleState.FAILED)
        self.assertEqual(record.iterations[-1].validation_result, "failed")

    def test_a_passing_verdict_with_an_unresolved_signal_is_its_own_status(self):
        """Regression: do not report a healthy repository as a failed validation.

        An earlier draft blended "validation passed" and "the signal is gone"
        into ``validation_passed``, so an ineffective-but-safe repair was
        reported to the runner as a *validation failure* - which would have told
        the operator the repository was broken when it was not.
        """
        from local_agent.maintenance_execution import MaintenanceExecutionResult

        entry = MaintenanceExecutionResult(
            status=MaintenanceExecutionStatus.SIGNAL_NOT_RESOLVED,
            applied=True,
            rolled_back=True,
            validation_passed=True,
            signal_resolved=False,
        )
        self.assertFalse(entry.succeeded)
        outcome = entry.to_outcome()
        self.assertFalse(outcome.succeeded)
        self.assertIs(outcome.validation_passed, True)
        self.assertEqual(outcome.changed_files, [])

    def test_a_result_without_a_verdict_can_never_be_credited_as_success(self):
        """No verdict is not a pass - the invariant the 4.21 audit found broken.

        Exercised across every status the executor can produce, so a future
        status cannot quietly acquire the ability to claim success without one.
        """
        from local_agent.maintenance_execution import (
            ALL_EXECUTION_STATUSES,
            MaintenanceExecutionResult,
        )

        for status in ALL_EXECUTION_STATUSES:
            for verdict in (None, False):
                entry = MaintenanceExecutionResult(
                    status=status, applied=True, validation_passed=verdict
                )
                self.assertFalse(
                    entry.succeeded, f"{status} with verdict {verdict!r} claimed success"
                )
                self.assertIsNot(
                    entry.to_outcome().validation_passed,
                    True,
                    f"{status} leaked a passing verdict",
                )
        # And a completed+applied+passed result is the only success.
        good = MaintenanceExecutionResult(
            status=MaintenanceExecutionStatus.COMPLETED,
            applied=True,
            validation_passed=True,
        )
        self.assertTrue(good.succeeded)

    def test_no_operations_is_reported_as_no_change_not_success(self):
        before = snapshot_tree(self.root)
        result = self.executor(provider=ScriptedProvider([[]])).execute(self.order())
        self.assertEqual(result.status, MaintenanceExecutionStatus.NO_CHANGE)
        self.assertFalse(result.succeeded)
        self.assertIsNone(result.validation_passed)
        self.assertEqual(snapshot_tree(self.root), before)

    def test_an_out_of_scope_edit_never_reaches_the_authoritative_tree(self):
        """The candidate workspace refuses it first; nothing is ever written."""
        before = snapshot_tree(self.root)
        out_of_scope = [
            FileOperation(action="modify", path="healthy.py", content="VALUE = 2\n", reason="x")
        ]
        provider = ScriptedProvider([list(out_of_scope) for _ in range(8)])
        result = self.executor(provider=provider).execute(self.order())
        self.assertNotEqual(result.status, MaintenanceExecutionStatus.COMPLETED)
        self.assertFalse(result.applied)
        self.assertEqual(snapshot_tree(self.root), before)

    def test_the_executor_scope_check_rejects_an_out_of_scope_operation(self):
        """UNIT-LEVEL: the executor's own additional ceiling, in isolation.

        The candidate workspace normally rejects an out-of-scope edit before the
        executor sees it, so this exercises the executor's redundant ceiling
        directly - it exists precisely so a future change to the upstream gate
        cannot silently remove the protection.
        """
        from local_agent.maintenance_execution import MaintenanceExecutionResult

        executor = self.executor()
        plan = Plan(objective="x", files_likely_to_change=["broken.py"])
        result = MaintenanceExecutionResult()
        self.assertFalse(
            executor._check_scope(
                [FileOperation(action="modify", path="healthy.py", content="x", reason="")],
                plan,
                result,
            )
        )
        self.assertEqual(result.status, MaintenanceExecutionStatus.SCOPE_VIOLATION)

        result = MaintenanceExecutionResult()
        self.assertFalse(
            executor._check_scope(
                [FileOperation(action="delete", path="broken.py", reason="")], plan, result
            )
        )
        self.assertEqual(result.status, MaintenanceExecutionStatus.SCOPE_VIOLATION)

    def test_a_file_creation_is_refused(self):
        before = snapshot_tree(self.root)
        provider = ScriptedProvider(
            [
                [
                    FileOperation(
                        action="create", path="broken.py", content=FIXED_SOURCE, reason="x"
                    )
                ]
            ]
        )
        result = self.executor(provider=provider).execute(self.order())
        self.assertNotEqual(result.status, MaintenanceExecutionStatus.COMPLETED)
        self.assertEqual(snapshot_tree(self.root), before)

    def test_every_no_mutation_status_is_declared(self):
        """The declared set must not silently include an applying status."""
        self.assertNotIn(MaintenanceExecutionStatus.COMPLETED, NO_MUTATION_STATUSES)
        self.assertNotIn(MaintenanceExecutionStatus.POST_VALIDATION_FAILED, NO_MUTATION_STATUSES)

    def test_retry_is_bounded_by_the_policy(self):
        """Two failed attempts and the policy blocks the candidate outright."""
        candidate = self.candidate()
        policy = MaintenanceExecutionPolicy(repository_root=self.root)
        candidate.failure_count = 2
        verdict = policy.decide(
            candidate,
            configured_tier=AutonomyTier.EXECUTE_WITH_EXISTING_APPROVAL,
            budget=MaintenanceBudget(),
        )
        self.assertTrue(verdict.blocked)
        self.assertFalse(verdict.may_execute)

    def test_retryable_statuses_never_include_a_completed_apply(self):
        self.assertNotIn(MaintenanceExecutionStatus.COMPLETED, RETRYABLE_STATUSES)
        self.assertNotIn(MaintenanceExecutionStatus.POST_VALIDATION_FAILED, RETRYABLE_STATUSES)
        self.assertNotIn(MaintenanceExecutionStatus.REFUSED_BY_POLICY, RETRYABLE_STATUSES)


# =============================================================================
# G. The approval boundary
# =============================================================================


class ApprovalBoundaryTests(ExecutorCase):
    def test_apply_is_refused_without_the_explicit_opt_in(self):
        before = snapshot_tree(self.root)
        result = self.executor(apply_enabled=False).execute(self.order())
        self.assertEqual(result.status, MaintenanceExecutionStatus.APPROVAL_REQUIRED)
        self.assertFalse(result.applied)
        # The implementation and prospective validation still happened for real.
        self.assertIs(result.prospective_validation_passed, True)
        self.assertEqual(snapshot_tree(self.root), before)

    def test_approval_always_without_an_approver_refuses(self):
        before = snapshot_tree(self.root)
        result = self.executor(approval_mode="always").execute(self.order())
        self.assertEqual(result.status, MaintenanceExecutionStatus.APPROVAL_REQUIRED)
        self.assertEqual(snapshot_tree(self.root), before)

    def test_approval_always_with_a_declining_approver_refuses(self):
        before = snapshot_tree(self.root)
        result = self.executor(
            approval_mode="always", approver=lambda changes: False
        ).execute(self.order())
        self.assertEqual(result.status, MaintenanceExecutionStatus.APPROVAL_REQUIRED)
        self.assertEqual(snapshot_tree(self.root), before)

    def test_approval_always_with_an_approving_approver_applies(self):
        seen: list[Any] = []

        def approve(changes):
            seen.append(changes)
            return True

        result = self.executor(approval_mode="always", approver=approve).execute(self.order())
        self.assertEqual(result.status, MaintenanceExecutionStatus.COMPLETED)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][0].path, "broken.py")

    def test_policy_mode_consults_the_real_approval_engine(self):
        from local_agent.approval import ApprovalPolicyEngine
        from local_agent.models import ApprovalPolicy

        engine = ApprovalPolicyEngine(
            [
                ApprovalPolicy.from_dict(
                    {
                        "name": "python-needs-review",
                        "action": "require_approval",
                        "if_path_matches": ["*.py"],
                    }
                )
            ]
        )
        before = snapshot_tree(self.root)
        result = self.executor(approval_mode="policy", policy_engine=engine).execute(self.order())
        self.assertEqual(result.status, MaintenanceExecutionStatus.APPROVAL_REQUIRED)
        self.assertEqual(snapshot_tree(self.root), before)

    def test_policy_mode_with_no_engine_fails_closed(self):
        result = self.executor(approval_mode="policy", policy_engine=None).execute(self.order())
        self.assertEqual(result.status, MaintenanceExecutionStatus.APPROVAL_REQUIRED)

    def test_an_approver_that_raises_is_treated_as_a_refusal(self):
        def boom(changes):
            raise RuntimeError("approver exploded")

        result = self.executor(approval_mode="always", approver=boom).execute(self.order())
        self.assertEqual(result.status, MaintenanceExecutionStatus.APPROVAL_REQUIRED)

    def test_the_gate_never_approves_an_empty_change_set(self):
        gate = MaintenanceApprovalGate(approval_mode="never", apply_enabled=True)
        self.assertFalse(gate.evaluate([]).approved)


# =============================================================================
# H. Isolation
# =============================================================================


class IsolationTests(ExecutorCase):
    def test_the_process_working_directory_is_never_changed(self):
        before = os.getcwd()
        self.executor().execute(self.order())
        self.assertEqual(os.getcwd(), before)

    def test_a_worktree_style_root_is_the_only_tree_modified(self):
        """The executor writes to the root it was given, never to a parent."""
        parent = Path(tempfile.mkdtemp(prefix="mx_parent_")).resolve()
        self.addCleanup(shutil.rmtree, parent, ignore_errors=True)
        (parent / "broken.py").write_text(BROKEN_SOURCE, encoding="utf-8")
        worktree = parent / "wt"
        worktree.mkdir()
        make_repo_with_parse_failure(worktree)
        parent_before = (parent / "broken.py").read_bytes()

        candidate = discover_parse_failure(worktree)
        order = build_work_order(
            candidate,
            granted_tier=AutonomyTier.EXECUTE_WITH_EXISTING_APPROVAL,
            budget=MaintenanceBudget(),
            configured_tier=AutonomyTier.EXECUTE_WITH_EXISTING_APPROVAL,
            fingerprint_fn=lambda paths: compute_state_fingerprint(worktree, paths),
        )
        result = self.executor(root=worktree).execute(order)
        self.assertEqual(result.status, MaintenanceExecutionStatus.COMPLETED)
        self.assertEqual((worktree / "broken.py").read_text(encoding="utf-8"), FIXED_SOURCE)
        self.assertEqual((parent / "broken.py").read_bytes(), parent_before)

    def test_the_candidate_workspace_is_cleaned_up(self):
        workspaces = self.data_dir / "workspaces"
        self.executor().execute(self.order())
        leftovers = [p for p in workspaces.glob("agentcand_*")] if workspaces.exists() else []
        self.assertEqual(leftovers, [])

    def test_two_executors_do_not_share_result_state(self):
        first = self.executor()
        second = self.executor()
        first.execute(self.order())
        self.assertEqual(len(second.results), 0)


# =============================================================================
# I. Architectural invariants - proved from the AST
# =============================================================================


def _module_ast(dotted: str) -> ast.Module:
    path = REPO_ROOT / Path(*dotted.split(".")).with_suffix(".py")
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _class_node(dotted: str, name: str) -> ast.ClassDef:
    for node in ast.walk(_module_ast(dotted)):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {dotted}")


def code_identifiers(node: ast.AST) -> set[str]:
    """Every identifier referenced in *executable* code under ``node``.

    Same helper contract as ``tests/test_maintenance.py``: string literals,
    comments and docstrings are parsed away, so an assertion built on this is a
    claim about behaviour rather than about prose.
    """
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            names.add(child.id)
        elif isinstance(child, ast.Attribute):
            names.add(child.attr)
        elif isinstance(child, (ast.Import, ast.ImportFrom)):
            for alias in child.names:
                names.add(alias.name)
    return names


def imported_modules(dotted: str) -> set[str]:
    package = dotted.rsplit(".", 1)[0]
    found: set[str] = set()
    for node in ast.walk(_module_ast(dotted)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = f"{package}.{base}" if base else package
            found.add(base)
    return found


EXECUTOR_MODULE = "local_agent.maintenance_execution"
MAINTENANCE_ADVISORY_MODULES = (
    "local_agent.maintenance",
    "local_agent.maintenance_analysis",
    "local_agent.maintenance_policy",
    "local_agent.maintenance_runner",
)


class ArchitecturalInvariantTests(unittest.TestCase):
    def setUp(self) -> None:
        self.executor_class = _class_node(EXECUTOR_MODULE, "MaintenanceExecutor")

    # -- the executor cannot write to source files ------------------------

    def test_the_executor_class_never_writes_to_the_filesystem(self):
        identifiers = code_identifiers(self.executor_class)
        for forbidden in (
            "write_text", "write_bytes", "unlink", "rmtree", "copyfile", "copy2",
            "rename", "replace", "truncate", "mkdir", "open",
        ):
            self.assertNotIn(forbidden, identifiers, forbidden)

    def test_the_executor_class_never_spawns_a_subprocess_itself(self):
        identifiers = code_identifiers(self.executor_class)
        for forbidden in ("subprocess", "Popen", "system", "popen", "spawnv", "execv"):
            self.assertNotIn(forbidden, identifiers, forbidden)

    def test_the_executor_module_does_not_import_subprocess(self):
        self.assertNotIn("subprocess", imported_modules(EXECUTOR_MODULE))

    def test_the_executor_class_never_changes_the_working_directory(self):
        self.assertNotIn("chdir", code_identifiers(self.executor_class))
        self.assertNotIn("getcwd", code_identifiers(self.executor_class))

    def test_the_executor_class_never_evaluates_code(self):
        identifiers = code_identifiers(self.executor_class)
        self.assertNotIn("eval", identifiers)
        self.assertNotIn("exec", identifiers)

    # -- it delegates, rather than duplicating, every authority -----------

    def test_the_executor_applies_changes_only_through_the_coding_agent(self):
        identifiers = code_identifiers(self.executor_class)
        self.assertIn("apply_prepared", identifiers)
        self.assertIn("CodingAgent", identifiers)

    def test_the_executor_uses_the_real_prospective_pipeline(self):
        identifiers = code_identifiers(self.executor_class)
        self.assertIn("InteractiveCodingAgent", identifiers)
        self.assertIn("CandidateWorkspace", identifiers)
        self.assertIn("ProspectiveValidator", identifiers)

    def test_the_executor_defers_to_the_validation_decision_engine(self):
        """It *uses* the engine; it never re-implements a scope decision."""
        identifiers = code_identifiers(self.executor_class)
        self.assertIn("ValidationDecisionEngine", identifiers)
        # No home-grown scope vocabulary: the executor must not compute the
        # recommendation itself.
        self.assertNotIn("recommend_validation_scope", identifiers)
        self.assertNotIn("safest_scope", identifiers)

    def test_the_executor_never_calls_the_approval_engine_directly(self):
        """The approval boundary is reached through the gate, not re-created."""
        identifiers = code_identifiers(self.executor_class)
        self.assertNotIn("ApprovalPolicyEngine", identifiers)
        self.assertNotIn("is_manual_approval_required", identifiers)

    def test_the_executor_module_never_imports_the_tool_engine(self):
        self.assertNotIn("local_agent.tool_engine", imported_modules(EXECUTOR_MODULE))

    def test_the_executor_module_never_imports_the_orchestrator(self):
        self.assertNotIn("local_agent.orchestrator", imported_modules(EXECUTOR_MODULE))

    def test_protected_path_enforcement_is_delegated_not_reimplemented(self):
        identifiers = code_identifiers(self.executor_class)
        # It consults the shared, case-folded helper and the policy...
        self.assertIn("is_protected_relative_path", identifiers)
        self.assertIn("is_protected", identifiers)
        # ...and never hard-codes its own protected-path list.
        module_source = (
            REPO_ROOT / "local_agent" / "maintenance_execution.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(module_source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                self.assertNotIn(
                    "tool_engine.py", node.value,
                    "the executor must not carry its own protected-path literal",
                )

    # -- the advisory layer stays advisory --------------------------------

    def test_the_advisory_modules_still_cannot_reach_the_authorities(self):
        for module in MAINTENANCE_ADVISORY_MODULES:
            imports = imported_modules(module)
            self.assertNotIn("local_agent.validation_decision", imports, module)
            self.assertNotIn("local_agent.approval", imports, module)
            self.assertNotIn("local_agent.tool_engine", imports, module)
            self.assertNotIn("local_agent.coding_agent", imports, module)
            self.assertNotIn("local_agent.maintenance_execution", imports, module)

    def test_the_policy_module_cannot_widen_validation_scope(self):
        identifiers = code_identifiers(_module_ast("local_agent.maintenance_policy"))
        for forbidden in (
            "ValidationDecisionEngine", "ValidationDecision", "recommend_validation_scope",
            "selected_commands", "apply_reuse",
        ):
            self.assertNotIn(forbidden, identifiers, forbidden)

    def test_maintenance_never_relaxes_the_validation_confidence_threshold(self):
        """The executor asks the decision engine for its *strictest* setting.

        ``min_confidence`` is the one engine knob that could narrow validation.
        The executor's default is ``"high"`` - the value that permits narrowing
        least often - and nothing in the maintenance layer supplies a weaker one.
        """
        module = _module_ast(EXECUTOR_MODULE)
        executor_init = next(
            node
            for node in ast.walk(_class_node(EXECUTOR_MODULE, "MaintenanceExecutor"))
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        names = [arg.arg for arg in executor_init.args.kwonlyargs]
        defaults = dict(zip(names, executor_init.args.kw_defaults))
        self.assertIn("min_impact_confidence", defaults)
        self.assertEqual(defaults["min_impact_confidence"].value, "high")

        # And the CLI never overrides it downward.
        cli_source = (REPO_ROOT / "local_agent" / "cli.py").read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(cli_source)):
            if isinstance(node, ast.keyword) and node.arg == "min_impact_confidence":
                self.fail("the CLI must not override the maintenance confidence floor")
        self.assertTrue(module.body)

    def test_only_one_module_bridges_maintenance_to_the_pipeline(self):
        """There is exactly one seam, and this is it."""
        bridges = []
        for path in sorted((REPO_ROOT / "local_agent").glob("maintenance*.py")):
            dotted = f"local_agent.{path.stem}"
            if "local_agent.coding_agent" in imported_modules(dotted):
                bridges.append(dotted)
        self.assertEqual(bridges, [EXECUTOR_MODULE])

    # -- protected files stay byte-identical ------------------------------

    def test_tool_engine_is_untouched_by_this_phase(self):
        self.assertProtectedFileUnchanged("local_agent/tool_engine.py")

    def test_approval_is_untouched_by_this_phase(self):
        self.assertProtectedFileUnchanged("local_agent/approval.py")

    def assertProtectedFileUnchanged(self, relative: str) -> None:
        """The file must match its committed HEAD~N baseline exactly.

        Compared against the working tree's git index rather than a hard-coded
        hash, so this stays true as unrelated phases land.
        """
        import subprocess  # noqa: PLC0415 - test-only, never in production code

        path = REPO_ROOT / relative
        self.assertTrue(path.is_file(), relative)
        try:
            committed = subprocess.run(
                ["git", "show", f"HEAD:{relative}"],
                cwd=REPO_ROOT,
                capture_output=True,
                check=False,
            )
        except OSError:
            self.skipTest("git is not available")
            return
        if committed.returncode != 0:
            self.skipTest(f"{relative} is not tracked at HEAD")
            return
        self.assertEqual(
            path.read_bytes().replace(b"\r\n", b"\n"),
            committed.stdout.replace(b"\r\n", b"\n"),
            f"{relative} differs from its committed content",
        )


# =============================================================================
# J. Work-order contract
# =============================================================================


class WorkOrderContractTests(ExecutorCase):
    def test_a_built_work_order_carries_everything_the_executor_needs(self):
        order = self.order()
        self.assertEqual(order.candidate_snapshot["candidate_id"], order.candidate_id)
        self.assertTrue(order.scope_fingerprint)
        self.assertTrue(order.planned_at)
        self.assertEqual(order.configured_tier, AutonomyTier.EXECUTE_WITH_EXISTING_APPROVAL)
        self.assertEqual(len(order.scope_files), MAX_SCOPE_FILES)

    def test_a_work_order_round_trips_through_its_dict(self):
        order = self.order()
        payload = order.to_dict()
        json.dumps(payload)  # must be serialisable for --enqueue
        self.assertEqual(payload["scope_fingerprint"], order.scope_fingerprint)
        self.assertEqual(payload["candidate_snapshot"]["kind"], MaintenanceSignal.PARSE_FAILURE)

    def test_the_fingerprint_reflects_real_file_content(self):
        first = self.order().scope_fingerprint
        (self.root / "broken.py").write_text(BROKEN_SOURCE + "# changed\n", encoding="utf-8")
        second = self.order().scope_fingerprint
        self.assertNotEqual(first, second)

    def test_a_missing_fingerprint_function_is_tolerated(self):
        order = self.order(fingerprint=False)
        self.assertEqual(order.scope_fingerprint, "")


# =============================================================================
# K. The full runner chain, with the executor wired
# =============================================================================


class RunnerIntegrationTests(ExecutorCase):
    """DISCOVER -> PRIORITIZE -> POLICY -> PLAN -> EXECUTE -> ... -> REASSESS.

    The whole point of Phase 4.22: the seam Phase 4.21 left empty, filled, with
    every stage real.
    """

    def setUp(self) -> None:
        super().setUp()
        # A real git repository, because ``AnalysisResult.degraded`` is true
        # whenever any intelligence source is unavailable - and a degraded scan
        # can never credit RESOLVED. Making the fixture a real repository with
        # real churn is what lets this test observe the *undegraded* path that a
        # real deployment sees.
        self.git_available = _git_init(self.root)

    def build_runner(self, executor=None, *, tier=AutonomyTier.EXECUTE_WITH_EXISTING_APPROVAL):
        from local_agent.knowledge import RepositoryKnowledgeGraph
        from local_agent.maintenance_analysis import collect_churn
        from local_agent.maintenance_runner import (
            MaintenanceManager,
            MaintenanceRunner,
            build_scan_function,
        )
        from local_agent.git import GitIntegration

        analyzer = MaintenanceAnalyzer(self.root)
        git = GitIntegration(self.root)
        scan = build_scan_function(
            analyzer,
            lifecycle_provider=self.storage.load_validation_lifecycle,
            telemetry_provider=self.storage.load_validation_telemetry,
            graph_provider=lambda: SemanticGraph.build(self.root),
            knowledge_provider=lambda: RepositoryKnowledgeGraph(),
            churn_provider=lambda: collect_churn(git),
        )
        return MaintenanceRunner(
            analyzer=analyzer,
            manager=MaintenanceManager(self.storage, self.root),
            scan=scan,
            budget=MaintenanceBudget(),
            policy=MaintenanceExecutionPolicy(repository_root=self.root),
            executor=executor,
            configured_tier=tier,
            fingerprint_fn=lambda paths: compute_state_fingerprint(self.root, paths),
        )

    def test_the_full_chain_discovers_plans_executes_validates_and_resolves(self):
        from local_agent.maintenance import RUN_MODE_EXECUTE

        executor = self.executor()
        result = self.build_runner(executor).run(mode=RUN_MODE_EXECUTE)

        record = result.record
        self.assertGreaterEqual(record.candidates_discovered, 1)
        self.assertEqual(record.candidates_selected, 1)
        self.assertEqual(record.execution_attempts, 1)
        self.assertEqual(record.executions_succeeded, 1)
        self.assertEqual(record.executions_failed, 0)

        # The file really changed.
        self.assertEqual((self.root / "broken.py").read_text(encoding="utf-8"), FIXED_SOURCE)
        # RESOLVED came from the runner's own fresh rescan, not from the apply.
        verdicts = list(result.reassessments.values())
        self.assertEqual(len(verdicts), 1)
        self.assertEqual(verdicts[0].outcome, "resolved")
        self.assertEqual(verdicts[0].after_fingerprint, "")

    def test_nothing_executes_when_no_executor_is_wired(self):
        from local_agent.maintenance import RUN_MODE_EXECUTE

        before = snapshot_tree(self.root)
        result = self.build_runner(None).run(mode=RUN_MODE_EXECUTE)
        self.assertEqual(result.record.execution_attempts, 0)
        self.assertEqual(snapshot_tree(self.root), before)
        self.assertTrue(
            any("no executor is wired" in note for note in result.record.notes),
            result.record.notes,
        )

    def test_a_refused_apply_is_never_reported_as_resolved(self):
        """Without --apply the signal survives, and the runner says so."""
        from local_agent.maintenance import RUN_MODE_EXECUTE

        executor = self.executor(apply_enabled=False)
        result = self.build_runner(executor).run(mode=RUN_MODE_EXECUTE)
        self.assertEqual(result.record.execution_attempts, 1)
        self.assertEqual(result.record.executions_succeeded, 0)
        self.assertEqual((self.root / "broken.py").read_text(encoding="utf-8"), BROKEN_SOURCE)
        verdict = list(result.reassessments.values())[0]
        self.assertNotEqual(verdict.outcome, "resolved")

    def test_a_post_validation_failure_is_never_reported_as_resolved(self):
        """The Phase 4.21 invariant, proved end-to-end on the executor path."""
        from local_agent.maintenance import RUN_MODE_EXECUTE

        (self.root / ".agent_data").mkdir(exist_ok=True)
        (self.root / "tests" / "test_broken.py").write_text(
            MODULE_TEST
            + "\n\n"
            "def test_environment():\n"
            "    import os\n"
            "    assert not os.path.isdir('.agent_data')\n",
            encoding="utf-8",
        )
        result = self.build_runner(self.executor()).run(mode=RUN_MODE_EXECUTE)
        self.assertEqual(result.record.executions_failed, 1)
        verdict = list(result.reassessments.values())[0]
        self.assertNotEqual(verdict.outcome, "resolved")
        # And the rolled-back tree still fails to parse, so the signal persists.
        self.assertEqual(SemanticGraph.build(self.root).parse_failures.keys(), {"broken.py"})

    def test_repeated_failure_is_bounded_across_runs(self):
        """Regression: a persistently failing candidate stops being retried.

        Found by the Phase 4.22 forensic audit. Two linked defects made
        ``PolicyThresholds.max_failures_before_block`` inert across runs:

        * ``MaintenanceCandidate.merge_observation`` did not fold attempt/failure
          counts forward, so the store recorded "failed once" no matter how many
          times a candidate failed; and
        * a run scored freshly-discovered candidates, whose counters always start
          at zero, so the policy never saw any history at all.

        The observable consequence was an unbounded retry loop: a scheduled
        ``maintenance run --execute`` against an unchanged repository with a
        persistently failing provider would attempt the same candidate forever.
        """
        from local_agent.maintenance import RUN_MODE_EXECUTE

        attempts: list[int] = []
        for _ in range(5):
            provider = ScriptedProvider(raise_exc=ProviderError("provider is down"))
            result = self.build_runner(self.executor(provider=provider)).run(
                mode=RUN_MODE_EXECUTE
            )
            attempts.append(result.record.execution_attempts)

        # Exactly two real attempts, then the policy blocks the candidate.
        self.assertEqual(attempts, [1, 1, 0, 0, 0], attempts)
        stored = self.storage.load_maintenance().candidates[0]
        self.assertEqual(stored.failure_count, 2)
        self.assertEqual((self.root / "broken.py").read_text(encoding="utf-8"), BROKEN_SOURCE)

    def test_attempt_history_survives_persistence(self):
        """The counts must round-trip, or the bound above is decoration."""
        from local_agent.maintenance import MaintenanceCandidate, MaintenanceStore

        store = MaintenanceStore()
        first = MaintenanceCandidate(kind=MaintenanceSignal.PARSE_FAILURE, subject="a.py")
        first.attempt_count = 3
        first.failure_count = 2
        store.upsert(first)

        # A fresh observation of the same problem must not reset the history.
        fresh = MaintenanceCandidate(kind=MaintenanceSignal.PARSE_FAILURE, subject="a.py")
        store.upsert(fresh)
        self.assertEqual(store.candidates[0].failure_count, 2)
        self.assertEqual(store.candidates[0].attempt_count, 3)

        # A run that incremented its own copy must fold that increase back in.
        incremented = MaintenanceCandidate(kind=MaintenanceSignal.PARSE_FAILURE, subject="a.py")
        incremented.attempt_count = 4
        incremented.failure_count = 3
        store.upsert(incremented)
        self.assertEqual(store.candidates[0].failure_count, 3)

        reloaded = MaintenanceStore.from_dict(json.loads(json.dumps(store.to_dict())))
        self.assertEqual(reloaded.candidates[0].failure_count, 3)
        self.assertEqual(reloaded.candidates[0].attempt_count, 4)

    def test_a_plan_only_tier_never_reaches_the_executor(self):
        from local_agent.maintenance import RUN_MODE_EXECUTE

        executor = self.executor()
        before = snapshot_tree(self.root)
        result = self.build_runner(executor, tier=AutonomyTier.PLAN_ONLY).run(
            mode=RUN_MODE_EXECUTE
        )
        self.assertEqual(result.record.execution_attempts, 0)
        self.assertEqual(executor.results, [])
        self.assertEqual(snapshot_tree(self.root), before)

    def test_dry_run_mode_never_reaches_the_executor(self):
        from local_agent.maintenance import RUN_MODE_DRY_RUN

        executor = self.executor()
        before = snapshot_tree(self.root)
        result = self.build_runner(executor).run(mode=RUN_MODE_DRY_RUN)
        self.assertTrue(result.work_orders)
        self.assertEqual(result.record.execution_attempts, 0)
        self.assertEqual(executor.results, [])
        self.assertEqual(snapshot_tree(self.root), before)


# =============================================================================
# L. The operator surface
# =============================================================================


class CliTests(ExecutorCase):
    def run_cli(self, *extra: str) -> tuple[int, str]:
        import contextlib
        import io

        from local_agent import cli

        buffer = io.StringIO()
        argv = [
            "maintenance", "run",
            "--project", str(self.root),
            "--provider", "mock",
            "--maintenance", "true",
            "--maintenance-tier", "execute_with_existing_approval",
            *extra,
        ]
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            code = cli.main(argv)
        return code, buffer.getvalue()

    def test_apply_without_execute_is_refused(self):
        code, output = self.run_cli("--apply")
        self.assertEqual(code, 1)
        self.assertIn("--apply requires --execute", output)
        self.assertEqual((self.root / "broken.py").read_text(encoding="utf-8"), BROKEN_SOURCE)

    def test_execute_with_dry_run_is_refused(self):
        code, output = self.run_cli("--execute", "--dry-run")
        self.assertEqual(code, 1)
        self.assertIn("mutually exclusive", output)

    def test_without_execute_the_cli_says_nothing_was_executed(self):
        before = snapshot_tree(self.root)
        code, output = self.run_cli()
        self.assertEqual(code, 0)
        self.assertIn("Nothing was executed", output)
        self.assertIn("--execute", output)
        self.assertEqual(snapshot_tree(self.root), before)

    def test_a_degraded_scan_never_lets_the_cli_claim_resolved(self):
        """PRODUCTION-INTEGRATION: a non-git repository is a degraded scan.

        The change applies and post-validates for real, and the CLI still
        reports ``RESOLVED (by rescan): 0`` - because a scan that could not read
        every intelligence source cannot testify that the signal is gone. This
        is the pre-existing Phase 4.21 conservatism, and Phase 4.22 must not
        quietly override it.
        """
        import unittest.mock as mock

        from local_agent import providers

        provider = ScriptedProvider([fix_operation()])
        with mock.patch.object(providers, "build_provider", lambda *a, **k: provider):
            code, output = self.run_cli("--execute", "--apply")
        self.assertEqual(code, 0, output)
        self.assertIn("APPLIED: 1", output)
        self.assertIn("VALIDATED: 1", output)
        self.assertIn("RESOLVED (by rescan): 0", output)
        self.assertIn("degraded", output)
        self.assertEqual((self.root / "broken.py").read_text(encoding="utf-8"), FIXED_SOURCE)

    def test_the_cli_distinguishes_attempted_applied_validated_and_resolved(self):
        """PRODUCTION-INTEGRATION through the real CLI; only the LLM is mocked."""
        import unittest.mock as mock

        from local_agent import providers

        if not _git_init(self.root):
            self.skipTest("git is unavailable, so churn cannot be read and the scan "
                          "is unavoidably degraded")
        provider = ScriptedProvider([fix_operation()])
        with mock.patch.object(providers, "build_provider", lambda *a, **k: provider):
            code, output = self.run_cli("--execute", "--apply")

        self.assertEqual(code, 0, output)
        self.assertIn("Narrow maintenance execution", output)
        self.assertIn("ATTEMPTED: 1", output)
        self.assertIn("APPLIED: 1", output)
        self.assertIn("VALIDATED: 1", output)
        self.assertIn("RESOLVED (by rescan): 1", output)
        self.assertIn("post-apply validation=PASSED", output)
        self.assertEqual((self.root / "broken.py").read_text(encoding="utf-8"), FIXED_SOURCE)

    def test_the_cli_never_claims_resolved_without_an_apply(self):
        import unittest.mock as mock

        from local_agent import providers

        provider = ScriptedProvider([fix_operation()])
        with mock.patch.object(providers, "build_provider", lambda *a, **k: provider):
            code, output = self.run_cli("--execute")

        self.assertEqual(code, 0, output)
        self.assertIn("ATTEMPTED: 1", output)
        self.assertIn("APPLIED: 0", output)
        self.assertIn("VALIDATED: 0", output)
        self.assertIn("RESOLVED (by rescan): 0", output)
        self.assertIn("approval_required", output)
        self.assertEqual((self.root / "broken.py").read_text(encoding="utf-8"), BROKEN_SOURCE)

    def test_the_json_surface_reports_the_execution_truthfully(self):
        import unittest.mock as mock

        from local_agent import providers

        provider = ScriptedProvider([fix_operation()])
        with mock.patch.object(providers, "build_provider", lambda *a, **k: provider):
            code, output = self.run_cli("--execute", "--apply", "--json")
        self.assertEqual(code, 0, output)
        payload = json.loads(output)
        execution = payload["execution"]
        self.assertTrue(execution["executor_wired"])
        self.assertTrue(execution["apply_permitted"])
        self.assertEqual(len(execution["results"]), 1)
        entry = execution["results"][0]
        self.assertEqual(entry["status"], "completed")
        self.assertTrue(entry["applied"])
        self.assertIs(entry["validation_passed"], True)
        self.assertTrue(entry["succeeded"])

    def test_the_json_surface_reports_no_executor_when_none_is_wired(self):
        code, output = self.run_cli("--json")
        self.assertEqual(code, 0)
        payload = json.loads(output)
        self.assertFalse(payload["execution"]["executor_wired"])
        self.assertEqual(payload["execution"]["results"], [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
