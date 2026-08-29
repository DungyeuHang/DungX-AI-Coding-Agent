"""Phase 4.18 - the centralised validation decision engine and hardened
evidence-reuse identity checks (staleness, policy, environment, analyzer
version).

Layered like the rest of the Phase 4.17/4.18 suites: pure-function/unit tests
for the new primitives, then integration tests against the real
:class:`~local_agent.orchestrator.Orchestrator` wiring - a real temp project,
a real :class:`~local_agent.semantic_impact.SemanticChangeImpactAnalyzer` run,
and a real :class:`~local_agent.evidence.EvidenceLedger` - with only the LLM
provider mocked.
"""

from __future__ import annotations

import datetime
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

from local_agent.config import AgentConfig
from local_agent.evidence import (
    EvidenceLedger,
    REASON_ANALYZER_VERSION_MISMATCH,
    REASON_COMMAND_MISMATCH,
    REASON_ENVIRONMENT_MISMATCH,
    REASON_FILES_CHANGED,
    REASON_NOT_PASSED,
    REASON_OK,
    REASON_POLICY_MISMATCH,
    REASON_STALE,
    STATUS_FAILED,
    STATUS_PASSED,
    STATUS_SKIPPED,
    ValidationEvidence,
    age_seconds,
    compute_executable_fingerprint,
    compute_policy_fingerprint,
    compute_state_fingerprint,
)
from local_agent.models import CommandSpec, ImplementationResult, ProjectContext, RunReport
from local_agent.orchestrator import Orchestrator
from local_agent.semantic_impact import (
    CONFIDENCE_HIGH,
    SCOPE_BROAD,
    SCOPE_TARGETED,
    SEMANTIC_ANALYZER_SCHEMA_VERSION,
    ChangeImpactReport,
    ChangedSymbol,
    SemanticChangeImpactAnalyzer,
)
from local_agent.storage import JsonFileStorage
from local_agent.validation_decision import (
    SAFETY_BROAD,
    SAFETY_NARROW,
    ReuseAttempt,
    ValidationDecision,
    ValidationDecisionEngine,
)


# ===========================================================================
# 1. Identity primitives (pure functions)
# ===========================================================================


class TestPolicyFingerprint(unittest.TestCase):
    def test_same_mapping_is_deterministic(self):
        values = {"max_impact_depth": 3, "threshold": "high"}
        self.assertEqual(compute_policy_fingerprint(values), compute_policy_fingerprint(dict(values)))

    def test_key_order_does_not_matter(self):
        a = compute_policy_fingerprint({"x": 1, "y": 2})
        b = compute_policy_fingerprint({"y": 2, "x": 1})
        self.assertEqual(a, b)

    def test_different_value_changes_fingerprint(self):
        a = compute_policy_fingerprint({"max_impact_depth": 3})
        b = compute_policy_fingerprint({"max_impact_depth": 4})
        self.assertNotEqual(a, b)

    def test_type_of_value_matters_not_just_string_form(self):
        # repr(3) != repr("3"), so these must not collide.
        a = compute_policy_fingerprint({"x": 3})
        b = compute_policy_fingerprint({"x": "3"})
        self.assertNotEqual(a, b)

    def test_empty_mapping_is_stable(self):
        self.assertEqual(compute_policy_fingerprint({}), compute_policy_fingerprint({}))


class TestExecutableFingerprint(unittest.TestCase):
    def test_same_command_is_deterministic(self):
        a = compute_executable_fingerprint(("pytest", "tests/test_x.py"))
        b = compute_executable_fingerprint(("pytest", "tests/test_x.py"))
        self.assertEqual(a, b)

    def test_fingerprint_ignores_trailing_arguments(self):
        # The resolved executable is the same regardless of which test file is
        # named; only the head of argv should influence this digest.
        a = compute_executable_fingerprint(("pytest", "tests/test_x.py"))
        b = compute_executable_fingerprint(("pytest", "tests/test_y.py"))
        self.assertEqual(a, b)

    def test_different_logical_executable_name_differs(self):
        a = compute_executable_fingerprint(("pytest", "-q"))
        b = compute_executable_fingerprint(("mypy", "-q"))
        self.assertNotEqual(a, b)

    def test_empty_command_does_not_raise(self):
        compute_executable_fingerprint(())


class TestAgeSeconds(unittest.TestCase):
    def test_empty_timestamp_is_unknown(self):
        self.assertIsNone(age_seconds(""))

    def test_garbage_timestamp_is_unknown(self):
        self.assertIsNone(age_seconds("not-a-date"))

    def test_recent_timestamp_has_small_age(self):
        now = datetime.datetime.now(datetime.timezone.utc)
        ts = now.isoformat()
        age = age_seconds(ts, now=now)
        self.assertIsNotNone(age)
        self.assertLess(age, 1.0)

    def test_naive_timestamp_is_treated_as_utc(self):
        now = datetime.datetime.now(datetime.timezone.utc)
        naive = now.replace(tzinfo=None).isoformat()
        age = age_seconds(naive, now=now)
        self.assertIsNotNone(age)
        self.assertLess(age, 1.0)

    def test_older_timestamp_has_larger_age(self):
        now = datetime.datetime.now(datetime.timezone.utc)
        earlier = (now - datetime.timedelta(hours=2)).isoformat()
        age = age_seconds(earlier, now=now)
        self.assertGreater(age, 3599)

    def test_age_never_negative(self):
        now = datetime.datetime.now(datetime.timezone.utc)
        future = (now + datetime.timedelta(hours=1)).isoformat()
        self.assertEqual(age_seconds(future, now=now), 0.0)


# ===========================================================================
# 2. EvidenceLedger.find_reusable - the four new opt-in checks
# ===========================================================================


class TestReuseIdentityChecks(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="p418_led_"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        (self.root / "a.py").write_text("x = 1\n", encoding="utf-8")
        self.fingerprint = compute_state_fingerprint(self.root, ["a.py"])

    def record(self, ledger: EvidenceLedger, **overrides: Any) -> ValidationEvidence:
        kwargs = dict(
            command=("pytest", "tests/test_a.py"),
            status=STATUS_PASSED,
            impacted_files=["a.py"],
            confidence=CONFIDENCE_HIGH,
            fingerprint=self.fingerprint,
        )
        kwargs.update(overrides)
        return ledger.record(**kwargs)

    def find(self, ledger: EvidenceLedger, **overrides: Any):
        kwargs = dict(
            command=("pytest", "tests/test_a.py"),
            current_root=self.root,
            relevant_files=["a.py"],
            min_confidence="low",
        )
        kwargs.update(overrides)
        return ledger.find_reusable(**kwargs)

    # -- backward compatibility: unset checks behave exactly like Phase 4.17

    def test_no_new_checks_passed_behaves_like_phase_417(self):
        ledger = EvidenceLedger()
        self.record(ledger)
        decision = self.find(ledger)
        self.assertTrue(decision.reusable)
        self.assertEqual(decision.reason, REASON_OK)

    def test_unset_policy_fingerprint_does_not_block_legacy_evidence(self):
        ledger = EvidenceLedger()
        self.record(ledger, policy_fingerprint="")  # legacy entry
        decision = self.find(ledger, policy_fingerprint=None)
        self.assertTrue(decision.reusable)

    # -- staleness

    def test_stale_evidence_is_rejected(self):
        ledger = EvidenceLedger()
        entry = self.record(ledger)
        entry.timestamp = (
            datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=2)
        ).isoformat()
        decision = self.find(ledger, max_age_seconds=60.0)
        self.assertFalse(decision.reusable)
        self.assertEqual(decision.reason, REASON_STALE)

    def test_fresh_evidence_within_max_age_is_accepted(self):
        ledger = EvidenceLedger()
        self.record(ledger)
        decision = self.find(ledger, max_age_seconds=3600.0)
        self.assertTrue(decision.reusable)

    def test_unparseable_timestamp_fails_closed_as_stale(self):
        ledger = EvidenceLedger()
        entry = self.record(ledger)
        entry.timestamp = ""
        decision = self.find(ledger, max_age_seconds=3600.0)
        self.assertFalse(decision.reusable)
        self.assertEqual(decision.reason, REASON_STALE)

    def test_max_age_none_means_no_staleness_check(self):
        ledger = EvidenceLedger()
        entry = self.record(ledger)
        entry.timestamp = "2000-01-01T00:00:00+00:00"
        decision = self.find(ledger, max_age_seconds=None)
        self.assertTrue(decision.reusable)

    # -- policy fingerprint

    def test_policy_mismatch_is_rejected(self):
        ledger = EvidenceLedger()
        self.record(ledger, policy_fingerprint="policy-v1")
        decision = self.find(ledger, policy_fingerprint="policy-v2")
        self.assertFalse(decision.reusable)
        self.assertEqual(decision.reason, REASON_POLICY_MISMATCH)

    def test_policy_match_is_accepted(self):
        ledger = EvidenceLedger()
        self.record(ledger, policy_fingerprint="policy-v1")
        decision = self.find(ledger, policy_fingerprint="policy-v1")
        self.assertTrue(decision.reusable)

    def test_checking_policy_against_legacy_evidence_without_one_is_rejected(self):
        # The critical fail-closed case: evidence recorded before this feature
        # existed has an empty policy_fingerprint, which must never silently
        # match a caller that has started checking it.
        ledger = EvidenceLedger()
        self.record(ledger)  # policy_fingerprint defaults to ""
        decision = self.find(ledger, policy_fingerprint="policy-v1")
        self.assertFalse(decision.reusable)
        self.assertEqual(decision.reason, REASON_POLICY_MISMATCH)

    # -- executable/environment fingerprint

    def test_environment_mismatch_is_rejected(self):
        ledger = EvidenceLedger()
        self.record(ledger, executable_fingerprint="env-a")
        decision = self.find(ledger, executable_fingerprint="env-b")
        self.assertFalse(decision.reusable)
        self.assertEqual(decision.reason, REASON_ENVIRONMENT_MISMATCH)

    def test_environment_match_is_accepted(self):
        ledger = EvidenceLedger()
        self.record(ledger, executable_fingerprint="env-a")
        decision = self.find(ledger, executable_fingerprint="env-a")
        self.assertTrue(decision.reusable)

    # -- analyzer schema version

    def test_analyzer_version_mismatch_is_rejected(self):
        ledger = EvidenceLedger()
        self.record(ledger, analyzer_version="4.17.0")
        decision = self.find(ledger, analyzer_version=SEMANTIC_ANALYZER_SCHEMA_VERSION)
        self.assertFalse(decision.reusable)
        self.assertEqual(decision.reason, REASON_ANALYZER_VERSION_MISMATCH)

    def test_analyzer_version_match_is_accepted(self):
        ledger = EvidenceLedger()
        self.record(ledger, analyzer_version=SEMANTIC_ANALYZER_SCHEMA_VERSION)
        decision = self.find(ledger, analyzer_version=SEMANTIC_ANALYZER_SCHEMA_VERSION)
        self.assertTrue(decision.reusable)

    # -- combined / precedence

    def test_all_four_checks_can_pass_together(self):
        ledger = EvidenceLedger()
        self.record(
            ledger, policy_fingerprint="p1", executable_fingerprint="e1",
            analyzer_version=SEMANTIC_ANALYZER_SCHEMA_VERSION,
        )
        decision = self.find(
            ledger, max_age_seconds=3600.0, policy_fingerprint="p1",
            executable_fingerprint="e1", analyzer_version=SEMANTIC_ANALYZER_SCHEMA_VERSION,
        )
        self.assertTrue(decision.reusable)

    def test_content_fingerprint_mismatch_still_wins_when_nothing_else_is_wrong(self):
        ledger = EvidenceLedger()
        self.record(ledger, fingerprint="deliberately-wrong")
        decision = self.find(ledger)
        self.assertFalse(decision.reusable)
        self.assertIn(decision.reason, {REASON_FILES_CHANGED, "tree_state_changed"})

    def test_failed_evidence_is_never_reused_even_with_matching_identity(self):
        ledger = EvidenceLedger()
        self.record(ledger, status=STATUS_FAILED, policy_fingerprint="p1")
        decision = self.find(ledger, policy_fingerprint="p1")
        self.assertFalse(decision.reusable)
        self.assertEqual(decision.reason, REASON_NOT_PASSED)

    def test_skipped_evidence_is_never_reused(self):
        ledger = EvidenceLedger()
        self.record(ledger, status=STATUS_SKIPPED)
        decision = self.find(ledger)
        self.assertFalse(decision.reusable)

    def test_reuse_evidence_round_trips_through_serialisation(self):
        ledger = EvidenceLedger()
        self.record(ledger, policy_fingerprint="p1", executable_fingerprint="e1", analyzer_version="v1")
        restored = EvidenceLedger.from_dict(ledger.to_dict())
        entry = restored.entries[0]
        self.assertEqual(entry.policy_fingerprint, "p1")
        self.assertEqual(entry.executable_fingerprint, "e1")
        self.assertEqual(entry.analyzer_version, "v1")

    def test_legacy_serialised_entry_without_new_fields_loads_with_empty_defaults(self):
        legacy = {
            "entries": [{
                "command": ["pytest"], "status": "passed", "impacted_files": ["a.py"],
                "confidence": "high", "fingerprint": self.fingerprint,
            }]
        }
        restored = EvidenceLedger.from_dict(legacy)
        entry = restored.entries[0]
        self.assertEqual(entry.policy_fingerprint, "")
        self.assertEqual(entry.executable_fingerprint, "")
        self.assertEqual(entry.analyzer_version, "")


# ===========================================================================
# 3. ValidationDecisionEngine
# ===========================================================================


class DecisionCase(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="p418_dec_")).resolve()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        pkg = self.root / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        self.core_src = "def f():\n    return 1\n"
        (pkg / "core.py").write_text(self.core_src, encoding="utf-8")
        tests = self.root / "tests"
        tests.mkdir()
        (tests / "test_core.py").write_text(
            "from pkg.core import f\n\n\ndef test_f():\n    assert f() == 2\n", encoding="utf-8"
        )

    def build_impact(self, new_src: str) -> ChangeImpactReport:
        base = {"pkg/core.py": self.core_src}
        (self.root / "pkg" / "core.py").write_text(new_src, encoding="utf-8")
        return SemanticChangeImpactAnalyzer(self.root).analyze(["pkg/core.py"], base_contents=base)


class TestDecideNarrowVsBroad(DecisionCase):
    def test_high_confidence_targeted_selects_only_semantic_commands(self):
        impact = self.build_impact("def f():\n    return 2\n")
        self.assertEqual(impact.recommended_scope, SCOPE_TARGETED)
        engine = ValidationDecisionEngine(min_confidence="high")
        decision = engine.decide(impact, current_root=self.root, lexical_commands=[])
        self.assertEqual(decision.scope, SCOPE_TARGETED)
        self.assertEqual(decision.safety_level, SAFETY_NARROW)
        self.assertEqual(len(decision.selected_commands), 1)

    def test_low_confidence_runs_union_with_lexical_commands(self):
        # No base contents -> every symbol looks "added" -> a degradation ->
        # LOW confidence -> BROAD, which must never narrow away the lexical
        # fallback commands.
        impact = SemanticChangeImpactAnalyzer(self.root).analyze(["pkg/core.py"])
        self.assertEqual(impact.confidence, "low")
        lexical = [CommandSpec(name="lexical", command=("pytest", "tests/test_core.py"))]
        engine = ValidationDecisionEngine(min_confidence="high")
        decision = engine.decide(impact, current_root=self.root, lexical_commands=lexical)
        self.assertEqual(decision.safety_level, SAFETY_BROAD)
        self.assertIn(("pytest", "tests/test_core.py"), [tuple(c.command) for c in decision.selected_commands])

    def test_confidence_score_is_monotonic_with_level(self):
        from local_agent.validation_decision import _confidence_score

        self.assertLess(_confidence_score("low"), _confidence_score("medium"))
        self.assertLess(_confidence_score("medium"), _confidence_score("high"))

    def test_uncertainty_sources_expose_the_degradations(self):
        impact = SemanticChangeImpactAnalyzer(self.root).analyze(["pkg/core.py"])
        engine = ValidationDecisionEngine(min_confidence="high")
        decision = engine.decide(impact, current_root=self.root, lexical_commands=[])
        self.assertTrue(decision.uncertainty_sources)

    def test_decision_round_trips_through_serialisation(self):
        impact = self.build_impact("def f():\n    return 2\n")
        engine = ValidationDecisionEngine(min_confidence="high")
        decision = engine.decide(impact, current_root=self.root, lexical_commands=[])
        restored = ValidationDecision.from_dict(decision.to_dict())
        self.assertEqual(restored.scope, decision.scope)
        self.assertEqual(
            [c.command for c in restored.selected_commands],
            [c.command for c in decision.selected_commands],
        )


class TestApplyReuseStandalone(DecisionCase):
    def test_matching_evidence_is_reused_and_removed_from_selection(self):
        impact = self.build_impact("def f():\n    return 2\n")
        spec = CommandSpec(name="t", command=("pytest", "tests/test_core.py"))
        ledger = EvidenceLedger()
        relevant = sorted(set(impact.changed_files) | set(impact.affected_files) | {"tests/test_core.py"})
        ledger.record(
            command=("pytest", "tests/test_core.py"), status=STATUS_PASSED,
            impacted_files=relevant,
            impacted_symbols=sorted({s.qualified_name for s in impact.changed_symbols}),
            confidence=impact.confidence,
            fingerprint=compute_state_fingerprint(self.root, relevant),
        )
        engine = ValidationDecisionEngine(min_confidence="high", reuse_enabled=True)
        remaining, attempts, saved = engine.apply_reuse([spec], impact, self.root, ledger)
        self.assertEqual(remaining, [])
        self.assertEqual(len(attempts), 1)
        self.assertTrue(attempts[0].reusable)
        self.assertIsNotNone(attempts[0].evidence)

    def test_reuse_disabled_keeps_every_command_and_still_reports_why(self):
        impact = self.build_impact("def f():\n    return 2\n")
        spec = CommandSpec(name="t", command=("pytest", "tests/test_core.py"))
        ledger = EvidenceLedger()
        ledger.record(command=("pytest", "tests/test_core.py"), status=STATUS_PASSED)
        engine = ValidationDecisionEngine(min_confidence="high", reuse_enabled=False)
        remaining, attempts, saved = engine.apply_reuse([spec], impact, self.root, ledger)
        self.assertEqual(remaining, [spec])
        self.assertFalse(attempts[0].reusable)

    def test_no_ledger_returns_selection_unchanged(self):
        impact = self.build_impact("def f():\n    return 2\n")
        spec = CommandSpec(name="t", command=("pytest", "tests/test_core.py"))
        engine = ValidationDecisionEngine(min_confidence="high", reuse_enabled=True)
        remaining, attempts, saved = engine.apply_reuse([spec], impact, self.root, None)
        self.assertEqual(remaining, [spec])
        self.assertEqual(attempts, [])

    def test_tree_change_after_recording_invalidates_reuse(self):
        impact = self.build_impact("def f():\n    return 2\n")
        spec = CommandSpec(name="t", command=("pytest", "tests/test_core.py"))
        relevant = sorted(set(impact.changed_files) | set(impact.affected_files) | {"tests/test_core.py"})
        ledger = EvidenceLedger()
        ledger.record(
            command=("pytest", "tests/test_core.py"), status=STATUS_PASSED,
            impacted_files=relevant,
            impacted_symbols=sorted({s.qualified_name for s in impact.changed_symbols}),
            confidence=impact.confidence,
            fingerprint=compute_state_fingerprint(self.root, relevant),
        )
        # Mutate the tree after recording: the fingerprint must no longer match.
        (self.root / "pkg" / "core.py").write_text("def f():\n    return 999\n", encoding="utf-8")
        engine = ValidationDecisionEngine(min_confidence="high", reuse_enabled=True)
        remaining, attempts, saved = engine.apply_reuse([spec], impact, self.root, ledger)
        self.assertEqual(remaining, [spec])
        self.assertFalse(attempts[0].reusable)


class TestReuseAttemptModel(unittest.TestCase):
    def test_round_trip(self):
        attempt = ReuseAttempt(
            command=("pytest",), reusable=True, reason=REASON_OK, time_saved_seconds=1.5,
            evidence={"command": ["pytest"]},
        )
        restored = ReuseAttempt.from_dict(attempt.to_dict())
        self.assertEqual(restored, attempt)

    def test_from_dict_tolerates_garbage(self):
        for payload in (None, [], "x", 42):
            restored = ReuseAttempt.from_dict(payload)
            self.assertFalse(restored.reusable)


# ===========================================================================
# 4. Monotonicity: reuse can only skip a command, never widen or narrow scope
# ===========================================================================


class TestMonotonicity(DecisionCase):
    def test_reuse_never_adds_a_command_beyond_what_impact_selected(self):
        impact = self.build_impact("def f():\n    return 2\n")
        engine = ValidationDecisionEngine(min_confidence="high", reuse_enabled=True)
        without_reuse = engine.decide(impact, current_root=self.root, lexical_commands=[])
        with_reuse = engine.decide(
            impact, current_root=self.root, lexical_commands=[], ledger=EvidenceLedger()
        )
        without_set = {tuple(c.command) for c in without_reuse.selected_commands}
        with_set = {tuple(c.command) for c in with_reuse.selected_commands}
        self.assertTrue(with_set.issubset(without_set) or with_set == without_set)

    def test_broad_scope_is_never_narrowed_by_reuse_alone(self):
        # No base contents -> LOW confidence -> BROAD. Even with a ledger that
        # has *some* unrelated evidence, the scope classification itself must
        # stay BROAD - only command *execution* can be skipped, never the
        # recorded scope/confidence conclusion.
        impact = SemanticChangeImpactAnalyzer(self.root).analyze(["pkg/core.py"])
        self.assertEqual(impact.recommended_scope, SCOPE_BROAD)
        engine = ValidationDecisionEngine(min_confidence="high", reuse_enabled=True)
        decision = engine.decide(
            impact, current_root=self.root, lexical_commands=[], ledger=EvidenceLedger()
        )
        self.assertEqual(decision.scope, SCOPE_BROAD)

    def test_denied_reuse_leaves_the_original_command_intact(self):
        impact = self.build_impact("def f():\n    return 2\n")
        spec = CommandSpec(name="t", command=("pytest", "tests/test_core.py"))
        ledger = EvidenceLedger()  # empty: nothing to reuse
        engine = ValidationDecisionEngine(min_confidence="high", reuse_enabled=True)
        remaining, attempts, _ = engine.apply_reuse([spec], impact, self.root, ledger)
        self.assertEqual(remaining, [spec])


# ===========================================================================
# 5. Ledger isolation (parallel worktrees) and bounded retention
# ===========================================================================


class TestLedgerIsolationAndRetention(unittest.TestCase):
    def test_two_engines_with_independent_ledgers_do_not_share_state(self):
        ledger_a, ledger_b = EvidenceLedger(), EvidenceLedger()
        ledger_a.record(command=("pytest",), status=STATUS_PASSED)
        self.assertEqual(len(ledger_a), 1)
        self.assertEqual(len(ledger_b), 0)

    def test_engine_instances_hold_no_shared_mutable_state(self):
        engine_a = ValidationDecisionEngine(min_confidence="high")
        engine_b = ValidationDecisionEngine(min_confidence="low")
        self.assertNotEqual(engine_a.min_confidence, engine_b.min_confidence)

    def test_new_fields_do_not_break_bounded_retention(self):
        ledger = EvidenceLedger(max_entries=5)
        for i in range(10):
            ledger.record(
                command=("pytest", f"tests/test_{i}.py"), status=STATUS_PASSED,
                policy_fingerprint=f"p{i}", executable_fingerprint=f"e{i}", analyzer_version="v1",
            )
        self.assertEqual(len(ledger), 5)
        # Oldest-first eviction: only the most recent 5 remain.
        self.assertEqual(
            [e.command[-1] for e in ledger.entries],
            [f"tests/test_{i}.py" for i in range(5, 10)],
        )

    def test_corrupted_entry_in_serialised_ledger_does_not_crash_load(self):
        payload = {"entries": [{"command": "not-a-list"}, None, 42, {"status": "passed"}]}
        ledger = EvidenceLedger.from_dict(payload)
        self.assertIsInstance(ledger, EvidenceLedger)

    def test_partial_entry_missing_most_fields_loads_with_defaults(self):
        ledger = EvidenceLedger.from_dict({"entries": [{"status": "passed"}]})
        entry = ledger.entries[0]
        self.assertEqual(entry.command, ())
        self.assertEqual(entry.confidence, "low")


# ===========================================================================
# 6. Real orchestrator integration: the actual post-apply wiring
# ===========================================================================


class OrchestratorCase(unittest.TestCase):
    """Exercises Orchestrator._semantic_targeted_commands /
    _apply_evidence_reuse directly against a real temp project - the actual
    integration seam Phase 4.18 rewired, not a reimplementation of it."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="p418_orch_")).resolve()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        pkg = self.root / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        self.core_src = "def f():\n    return 1\n"
        (pkg / "core.py").write_text(self.core_src, encoding="utf-8")
        tests_dir = self.root / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_core.py").write_text(
            "from pkg.core import f\n\n\ndef test_f():\n    assert f() == 2\n", encoding="utf-8"
        )
        (self.root / "pkg" / "core.py").write_text("def f():\n    return 2\n", encoding="utf-8")

    def make_orchestrator(self, **config_overrides: Any) -> Orchestrator:
        config = AgentConfig(
            project=self.root,
            semantic_impact_analysis_enabled=True,
            knowledge_graph_enabled=False,
            **config_overrides,
        )
        storage = JsonFileStorage(self.root / ".agent_data")
        return Orchestrator(config, storage, None, threading.Lock(), threading.Lock())

    def candidate_impact_report(self) -> dict[str, Any]:
        """A HIGH-confidence impact dict, matching what the real Phase 4.16
        candidate loop supplies: BASE contents let symbols be classified
        precisely instead of degrading to "everything looks added"."""
        impact = SemanticChangeImpactAnalyzer(self.root).analyze(
            ["pkg/core.py"], base_contents={"pkg/core.py": self.core_src}
        )
        self.assertEqual(impact.confidence, CONFIDENCE_HIGH)
        return impact.to_dict()

    def make_report(self, impact_report: dict[str, Any] | None = None) -> RunReport:
        report = RunReport(project=ProjectContext(root=str(self.root)))
        report.changed_files = ["pkg/core.py"]
        report.implementation_result = ImplementationResult(
            success=True, impact_report=impact_report,
        )
        return report

    def test_semantic_targeted_commands_sets_report_semantic_impact(self):
        orch = self.make_orchestrator()
        report = self.make_report()
        commands = orch._semantic_targeted_commands(
            report, ProjectContext(root=str(self.root)), [], lambda msg: None,
        )
        self.assertTrue(report.semantic_impact)
        self.assertTrue(any("tests/test_core.py" in c.command for c in commands))

    def test_reuse_end_to_end_with_matching_candidate_evidence(self):
        orch = self.make_orchestrator(reuse_candidate_validation_evidence=True)
        context = ProjectContext(root=str(self.root))
        report = self.make_report(self.candidate_impact_report())
        commands = orch._semantic_targeted_commands(report, context, [], lambda msg: None)
        self.assertTrue(commands)

        impact = ChangeImpactReport.from_dict(report.semantic_impact)
        relevant = sorted(
            set(impact.changed_files) | set(impact.affected_files) | {"tests/test_core.py"}
        )
        ledger = EvidenceLedger()
        ledger.record(
            command=("pytest", "tests/test_core.py"), status=STATUS_PASSED,
            impacted_files=relevant,
            impacted_symbols=sorted({s.qualified_name for s in impact.changed_symbols}),
            confidence=impact.confidence,
            fingerprint=compute_state_fingerprint(self.root, relevant),
            policy_fingerprint=orch._decision_policy_fingerprint(),
            executable_fingerprint=compute_executable_fingerprint(("pytest", "tests/test_core.py")),
            analyzer_version=SEMANTIC_ANALYZER_SCHEMA_VERSION,
        )
        report.implementation_result.validation_evidence = ledger.to_dict()["entries"]

        remaining = orch._apply_evidence_reuse(report, commands, lambda msg: None)
        self.assertEqual(remaining, [])
        self.assertEqual(report.implementation_result.validation_evidence_reused, 1)
        self.assertTrue(report.validation_evidence)

    def test_reuse_is_denied_when_policy_changes_between_recording_and_reuse(self):
        orch_v1 = self.make_orchestrator(
            reuse_candidate_validation_evidence=True, max_affected_tests=8,
        )
        context = ProjectContext(root=str(self.root))
        report = self.make_report(self.candidate_impact_report())
        commands = orch_v1._semantic_targeted_commands(report, context, [], lambda msg: None)
        impact = ChangeImpactReport.from_dict(report.semantic_impact)
        relevant = sorted(
            set(impact.changed_files) | set(impact.affected_files) | {"tests/test_core.py"}
        )
        ledger = EvidenceLedger()
        ledger.record(
            command=("pytest", "tests/test_core.py"), status=STATUS_PASSED,
            impacted_files=relevant,
            impacted_symbols=sorted({s.qualified_name for s in impact.changed_symbols}),
            confidence=impact.confidence,
            fingerprint=compute_state_fingerprint(self.root, relevant),
            policy_fingerprint=orch_v1._decision_policy_fingerprint(),
            executable_fingerprint=compute_executable_fingerprint(("pytest", "tests/test_core.py")),
            analyzer_version=SEMANTIC_ANALYZER_SCHEMA_VERSION,
        )
        report.implementation_result.validation_evidence = ledger.to_dict()["entries"]

        # A *different* orchestrator instance, differing only in a
        # decision-relevant setting, must not silently reuse evidence recorded
        # under the old policy.
        orch_v2 = self.make_orchestrator(
            reuse_candidate_validation_evidence=True, max_affected_tests=3,
        )
        remaining = orch_v2._apply_evidence_reuse(report, commands, lambda msg: None)
        self.assertEqual(remaining, commands)
        self.assertEqual(report.implementation_result.validation_evidence_reused, 0)

    def test_disabled_reuse_returns_targeted_commands_unchanged(self):
        orch = self.make_orchestrator(reuse_candidate_validation_evidence=False)
        report = self.make_report()
        commands = [CommandSpec(name="t", command=("pytest", "tests/test_core.py"))]
        remaining = orch._apply_evidence_reuse(report, commands, lambda msg: None)
        self.assertEqual(remaining, commands)

    def test_disabled_semantic_impact_returns_lexical_commands_unchanged(self):
        orch = self.make_orchestrator()
        orch.config.semantic_impact_analysis_enabled = False
        report = self.make_report()
        lexical = [CommandSpec(name="lex", command=("pytest", "tests/test_core.py"))]
        result = orch._semantic_targeted_commands(
            report, ProjectContext(root=str(self.root)), lexical, lambda msg: None,
        )
        self.assertEqual(result, lexical)
        self.assertIsNone(report.semantic_impact)


if __name__ == "__main__":
    unittest.main()
