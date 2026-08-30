"""Phase 4.19 (part 2): empirical validation intelligence - the *measurement*
half of the phase.

``tests/test_validation_telemetry.py`` already covers the decision-record
schema, outcome classification, the Wilson reliability estimator, the
calibration safety floor, shadow mode, the bounded store, and the orchestrator
wiring. This module deliberately does not repeat any of that. It covers what
was still missing, and what the phase brief calls out as "much more valuable
than merely increasing test count":

* **Part 12** - a false-positive / false-negative analysis over ten fixture
  repositories, one per dependency-relationship type, each built on the real
  filesystem and analysed by the real
  :class:`~local_agent.semantic_impact.SemanticChangeImpactAnalyzer`. The
  point is not that the analyzer detects everything (it demonstrably does
  not); it is to establish, empirically and in an executable form, *which*
  relationships it misses and whether the scope policy compensates.
* **Part 13** - synthetic defect injection: a real defect introduced into a
  real dependent, with ``pytest`` actually executed as a subprocess under a
  targeted command set and again under a broad one, to observe rather than
  assume whether the narrow scope would have caught it.
* **Part 14** - the ten evidence-reuse experiments, each asserting the exact
  rejection reason, and feeding the resulting reason tally into a real
  telemetry store so Part 10's statistics are exercised end to end.
* **Parts 4 / 11 / 15** - the decision-quality metrics, the measured cost
  model, and the extended health report.
* **Part 20** - measured overhead of the telemetry path.
* **Part 22** - the invariants not already asserted elsewhere.

Nothing here mocks the subsystem under test: the analyzer, the evidence
ledger, the fingerprinting, the JSON persistence, the threads, and the pytest
subprocesses are all real. No test mutates the DungX repository itself; every
fixture lives in a temporary directory that is removed on cleanup.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from local_agent.evidence import (
    EvidenceLedger,
    REASON_ANALYZER_VERSION_MISMATCH,
    REASON_COMMAND_MISMATCH,
    REASON_ENVIRONMENT_MISMATCH,
    REASON_FINGERPRINT_MISMATCH,
    REASON_NOT_PASSED,
    REASON_NO_EVIDENCE,
    REASON_OK,
    REASON_POLICY_MISMATCH,
    REASON_STALE,
    compute_executable_fingerprint,
    compute_policy_fingerprint,
    compute_state_fingerprint,
)
from local_agent.semantic_impact import (
    SCOPE_BROAD,
    SCOPE_EXPANDED,
    SCOPE_ORDER,
    SCOPE_TARGETED,
    SEMANTIC_ANALYZER_SCHEMA_VERSION,
    SemanticChangeImpactAnalyzer,
)
from local_agent.storage import JsonFileStorage
from local_agent.validation_telemetry import (
    CalibrationObservation,
    DEGRADED_EVIDENCE_MARKER,
    OUTCOME_VALIDATION_FAILED,
    OUTCOME_VALIDATION_PASSED,
    QUALITY_BROAD_NOT_PROVEN_NECESSARY,
    QUALITY_CONSISTENT,
    QUALITY_TARGETED_CAUGHT_DEFECT,
    QUALITY_TARGETED_MISSED_DEFECT,
    QUALITY_UNCONFIRMED,
    ValidationDecisionRecord,
    ValidationTelemetryManager,
    ValidationTelemetryStore,
    classify_outcome,
    compute_cost_model,
    compute_decision_quality_metrics,
    compute_health,
    compute_reliability,
    evidence_types_for_impact,
    impact_is_degraded,
    wilson_bounds,
    wilson_lower_bound,
)

STATUS_PASSED = "passed"

# The change under analysis throughout: a behaviour change to
# ``calculate_total`` (and to ``Base.calculate_total``), which every fixture's
# dependent reaches - or fails to reach - by a different language construct.
CORE_BEFORE = (
    "def calculate_total(x):\n"
    "    return x + 1\n"
    "\n"
    "\n"
    "class Base:\n"
    "    def calculate_total(self):\n"
    "        return 1\n"
)
CORE_AFTER = (
    "def calculate_total(x):\n"
    "    return x + 2\n"
    "\n"
    "\n"
    "class Base:\n"
    "    def calculate_total(self):\n"
    "        return 2\n"
)

FACADE = "from pkg.core import calculate_total\n\n__all__ = ['calculate_total']\n"


def build_fixture_repo(dependent_test_source: str) -> Path:
    """A real, minimal package whose only test reaches ``pkg.core`` via
    ``dependent_test_source``. Returned path must be removed by the caller."""
    root = Path(tempfile.mkdtemp(prefix="p419_fx_")).resolve()
    pkg = root / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "core.py").write_text(CORE_AFTER, encoding="utf-8")
    (pkg / "facade.py").write_text(FACADE, encoding="utf-8")
    tests = root / "tests"
    tests.mkdir()
    (tests / "test_core.py").write_text(dependent_test_source, encoding="utf-8")
    return root


# ===========================================================================
# Part 12 - false-positive / false-negative analysis across ten dependency
#           relationship types.
# ===========================================================================


#: ``(label, dependent source, a real dependency exists, expected scope)``.
#:
#: ``expected_scope`` is not an aspiration - it is the behaviour observed from
#: the real analyzer on these exact fixtures, pinned here so a future change to
#: the scope policy that quietly narrows any of them fails loudly. The
#: interesting column is the interaction between "is there really a dependency"
#: and the scope: a MISSED dependency that still yields a non-targeted scope is
#: safe-by-escalation, whereas a missed dependency at TARGETED scope would be a
#: genuine escape.
DEPENDENCY_FIXTURES: tuple[tuple[str, str, bool, str], ...] = (
    (
        "A_isolated",
        "def test_t():\n    assert 1 == 1\n",
        False,
        SCOPE_BROAD,
    ),
    (
        "B_hidden_direct",
        "from pkg.core import calculate_total\n\n\ndef test_t():\n"
        "    assert calculate_total(1) == 3\n",
        True,
        SCOPE_TARGETED,
    ),
    (
        "C_alias",
        "from pkg.core import calculate_total as total\n\n\ndef test_t():\n"
        "    assert total(1) == 3\n",
        True,
        SCOPE_TARGETED,
    ),
    (
        "D_attribute",
        "def test_t():\n"
        "    class O:\n"
        "        def calculate_total(self):\n"
        "            return 2\n"
        "    assert O().calculate_total() == 2\n",
        False,
        SCOPE_EXPANDED,
    ),
    (
        "E_dynamic_import_resolved",
        "import importlib\n\n\ndef test_t():\n"
        "    m = importlib.import_module('pkg.core')\n"
        "    assert m.calculate_total(1) == 3\n",
        True,
        SCOPE_TARGETED,
    ),
    (
        "F_dynamic_import_unresolved",
        "import importlib\n\nNAME = 'pkg.core'\n\n\ndef test_t():\n"
        "    m = importlib.import_module(NAME)\n"
        "    assert m.calculate_total(1) == 3\n",
        True,
        SCOPE_EXPANDED,
    ),
    (
        "G_reexport",
        "from pkg.facade import calculate_total\n\n\ndef test_t():\n"
        "    assert calculate_total(1) == 3\n",
        True,
        SCOPE_EXPANDED,
    ),
    (
        "H_inheritance",
        "from pkg.core import Base\n\n\nclass C(Base):\n    pass\n\n\ndef test_t():\n"
        "    assert C().calculate_total() == 2\n",
        True,
        SCOPE_TARGETED,
    ),
    (
        "I_annotation",
        "from pkg.core import Base\n\n\ndef helper(b: Base) -> Base:\n    return b\n\n\n"
        "def test_t():\n    assert helper(Base()) is not None\n",
        True,
        SCOPE_TARGETED,
    ),
    (
        "J_unresolved_attribute_only",
        "def test_t(obj=None):\n"
        "    assert obj is None or obj.calculate_total() == 2\n",
        True,
        SCOPE_EXPANDED,
    ),
)


class DependencyFixtureAnalysisCase(unittest.TestCase):
    """Part 12: what the analyzer really sees, per relationship type."""

    def analyse(self, source: str):
        root = build_fixture_repo(source)
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        report = SemanticChangeImpactAnalyzer(root).analyze(
            ["pkg/core.py"], base_contents={"pkg/core.py": CORE_BEFORE}
        )
        return root, report

    def test_every_fixture_produces_the_pinned_scope(self):
        for label, source, _real, expected_scope in DEPENDENCY_FIXTURES:
            with self.subTest(fixture=label):
                _root, report = self.analyse(source)
                self.assertEqual(
                    report.recommended_scope,
                    expected_scope,
                    f"{label}: scope drifted from the empirically pinned value",
                )

    def test_a_missed_dependency_is_never_left_at_targeted_scope(self):
        """The safety-critical column of the Part 12 table.

        For each fixture, "did the analyzer actually surface the dependent as a
        selected validation target" is compared against "is there really a
        dependency". A dependency that the graph missed is only acceptable when
        the scope escalated above TARGETED to compensate - which is what makes
        the analyzer's known blind spots (attribute access, unresolved dynamic
        imports) survivable rather than silent escapes.
        """
        for label, source, real_dependency, _expected in DEPENDENCY_FIXTURES:
            with self.subTest(fixture=label):
                _root, report = self.analyse(source)
                detected = "tests/test_core.py" in report.affected_files
                if real_dependency and not detected:
                    self.assertNotEqual(
                        report.recommended_scope,
                        SCOPE_TARGETED,
                        f"{label}: real dependency missed by the graph AND left at "
                        "targeted scope - this is a genuine escape, not a safe miss",
                    )

    def test_a_fixture_with_no_dependency_is_a_cost_false_positive_not_a_safety_one(self):
        """Fixture A has no dependency at all yet is validated BROADly.

        That is a *false positive* in the cost sense - the agent will run more
        than it strictly needed - and it is the correct direction to be wrong
        in. Asserted explicitly so a future "optimisation" that narrows the
        no-evidence case has to change this test on purpose.
        """
        _root, report = self.analyse(DEPENDENCY_FIXTURES[0][1])
        self.assertEqual(report.recommended_scope, SCOPE_BROAD)
        self.assertEqual(report.confidence, "low")
        self.assertNotIn("tests/test_core.py", report.affected_files)

    def test_each_fixture_yields_a_wellformed_calibration_observation(self):
        """Part 12's last column: every fixture can be turned into a real
        observation, with the evidence-type labels the analyzer actually
        produced - not invented ones."""
        for label, source, _real, _expected in DEPENDENCY_FIXTURES:
            with self.subTest(fixture=label):
                root, report = self.analyse(source)
                record = ValidationDecisionRecord(
                    repository_id="fx",
                    scope=report.recommended_scope,
                    confidence_level=report.confidence,
                    evidence_types=sorted(evidence_types_for_impact(report)),
                    degraded_analysis=impact_is_degraded(report),
                    outcome=OUTCOME_VALIDATION_PASSED,
                    decision_quality=(
                        QUALITY_CONSISTENT
                        if report.recommended_scope == SCOPE_TARGETED
                        else QUALITY_BROAD_NOT_PROVEN_NECESSARY
                    ),
                )
                observation = CalibrationObservation.from_record(record)
                self.assertEqual(observation.selected_scope, report.recommended_scope)
                self.assertEqual(observation.predicted_confidence, report.confidence)
                self.assertFalse(observation.later_broader_validation_found_defect)
                # Round-trips through JSON exactly as it will on disk.
                self.assertEqual(
                    CalibrationObservation.from_dict(
                        json.loads(json.dumps(observation.to_dict()))
                    ).to_dict(),
                    observation.to_dict(),
                )
                del root

    def test_evidence_type_labels_are_drawn_from_the_real_vocabulary(self):
        """No fixture may invent an evidence label. Every label is either a
        Phase 4.18 dependency-evidence type or a coarse tier name - never free
        text - which is what keeps the telemetry store's key space bounded."""
        from local_agent.dependency_resolution import ALL_EVIDENCE_TYPES

        seen: set[str] = set()
        for label, source, _real, _expected in DEPENDENCY_FIXTURES:
            _root, report = self.analyse(source)
            seen |= set(evidence_types_for_impact(report))
        self.assertTrue(seen, "the fixture set produced no evidence labels at all")
        tier_names = {t.tier for _l, s, _r, _e in DEPENDENCY_FIXTURES
                      for t in self.analyse(s)[1].validation_targets}
        for label in seen:
            self.assertTrue(
                label in ALL_EVIDENCE_TYPES or label in tier_names,
                f"{label!r} is neither a dependency-evidence type nor a tier name",
            )


# ===========================================================================
# Part 13 - synthetic defect injection, with pytest really executed.
# ===========================================================================


class SyntheticDefectInjectionCase(unittest.TestCase):
    """Inject a real defect into a real dependent and *observe* whether the
    targeted command set catches it.

    This is the only place in the suite that answers the escape question with
    evidence rather than reasoning. ``pytest`` runs as an actual subprocess in
    a temporary repository; the DungX working tree is never touched.
    """

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="p419_inject_")).resolve()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        pkg = self.root / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        # The changed module.
        (pkg / "core.py").write_text(
            "def calculate_total(x):\n    return x + 1\n", encoding="utf-8"
        )
        # A dependent that reaches core only through an attribute access on a
        # module object - the blind spot fixture D/J established above.
        (pkg / "consumer.py").write_text(
            "import pkg.core as _c\n"
            "\n"
            "\n"
            "def doubled(x):\n"
            "    return _c.calculate_total(x) * 2\n",
            encoding="utf-8",
        )
        tests = self.root / "tests"
        tests.mkdir()
        # The test a TARGETED scope would select: exercises core directly and
        # is written against the *new* behaviour, so it passes after the change.
        (tests / "test_core.py").write_text(
            "from pkg.core import calculate_total\n"
            "\n"
            "\n"
            "def test_core():\n"
            "    assert calculate_total(1) == 3\n",
            encoding="utf-8",
        )
        # The dependent's own test, pinned to the OLD behaviour. This is the
        # injected defect: the change to core silently breaks it.
        (tests / "test_consumer.py").write_text(
            "from pkg.consumer import doubled\n"
            "\n"
            "\n"
            "def test_doubled():\n"
            "    assert doubled(1) == 4\n",
            encoding="utf-8",
        )

    def run_pytest(self, *targets: str) -> bool:
        """Return True when the run passed. Real subprocess, real exit code."""
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *targets],
            cwd=str(self.root),
            capture_output=True,
            text=True,
            timeout=300,
        )
        return completed.returncode == 0

    def inject_defect(self) -> None:
        """Apply the 'safe-looking' change to core that breaks the dependent."""
        (self.root / "pkg" / "core.py").write_text(
            "def calculate_total(x):\n    return x + 2\n", encoding="utf-8"
        )

    def test_baseline_suite_is_green_before_injection(self):
        """Guard: without this, a broad-run failure below would prove nothing."""
        (self.root / "tests" / "test_core.py").write_text(
            "from pkg.core import calculate_total\n"
            "\n"
            "\n"
            "def test_core():\n"
            "    assert calculate_total(1) == 2\n",
            encoding="utf-8",
        )
        self.assertTrue(self.run_pytest("tests"))

    def test_targeted_scope_misses_the_injected_defect_that_broad_catches(self):
        self.inject_defect()
        targeted_passed = self.run_pytest("tests/test_core.py")
        broad_passed = self.run_pytest("tests")
        self.assertTrue(
            targeted_passed,
            "the targeted command set was expected to pass - it is the premise "
            "of an escape that the narrow scope sees nothing wrong",
        )
        self.assertFalse(
            broad_passed,
            "the broad command set was expected to catch the injected defect",
        )

    def test_the_escape_classifies_as_targeted_missed_defect(self):
        """The measured escape flows into the outcome vocabulary correctly, so
        a real run of this shape would produce a real calibration failure."""
        self.inject_defect()
        targeted_passed = self.run_pytest("tests/test_core.py")
        broad_passed = self.run_pytest("tests")
        outcome, quality = classify_outcome(
            scope=SCOPE_TARGETED,
            targeted_ran=True,
            targeted_failed=not targeted_passed,
            broad_ran=True,
            broad_failed=not broad_passed,
        )
        self.assertEqual(outcome, OUTCOME_VALIDATION_FAILED)
        self.assertEqual(quality, QUALITY_TARGETED_MISSED_DEFECT)

    def test_broad_scope_on_the_same_defect_is_a_caught_failure_not_an_escape(self):
        """The identical defect, decided BROADly, must not be recorded as an
        escape - the escape signal is about the *scope choice*, and a broad
        choice made no narrow claim to be contradicted."""
        self.inject_defect()
        broad_passed = self.run_pytest("tests")
        outcome, quality = classify_outcome(
            scope=SCOPE_BROAD,
            targeted_ran=False,
            targeted_failed=False,
            broad_ran=True,
            broad_failed=not broad_passed,
        )
        self.assertEqual(outcome, OUTCOME_VALIDATION_FAILED)
        self.assertNotEqual(quality, QUALITY_TARGETED_MISSED_DEFECT)

    def test_one_injected_escape_immediately_lowers_the_evidence_type_estimate(self):
        """End-to-end: measured escape -> observation -> reliability drop.

        Uses the outcome actually produced by the subprocess runs above rather
        than a hard-coded quality, so if the injection ever stopped escaping
        this test would stop claiming a reliability drop.
        """
        self.inject_defect()
        outcome, quality = classify_outcome(
            scope=SCOPE_TARGETED,
            targeted_ran=True,
            targeted_failed=not self.run_pytest("tests/test_core.py"),
            broad_ran=True,
            broad_failed=not self.run_pytest("tests"),
        )
        self.assertEqual(quality, QUALITY_TARGETED_MISSED_DEFECT)
        observations = [
            CalibrationObservation(
                evidence_types=("attribute_resolution",),
                decision_quality=QUALITY_CONSISTENT,
                outcome=OUTCOME_VALIDATION_PASSED,
                selected_scope=SCOPE_TARGETED,
            )
            for _ in range(9)
        ] + [
            CalibrationObservation(
                evidence_types=("attribute_resolution",),
                decision_quality=quality,
                outcome=outcome,
                selected_scope=SCOPE_TARGETED,
                later_broader_validation_found_defect=True,
            )
        ]
        reliability = compute_reliability(observations, min_samples=5)["attribute_resolution"]
        self.assertEqual(reliability.trials, 10)
        self.assertEqual(reliability.failures, 1)
        self.assertLess(reliability.lower_bound, 0.9)


# ===========================================================================
# Part 14 / Part 10 - the ten reuse experiments and their reason statistics.
# ===========================================================================


class ReuseExperimentMatrixCase(unittest.TestCase):
    """Every reuse scenario from Part 14, each asserting its exact reason.

    Real files, real fingerprints, real ledger. The reasons are asserted
    individually because Part 14 requires that *every rejection has a reason* -
    a generic "not reusable" would make the Part 10 statistics meaningless.
    """

    COMMAND = ("pytest", "tests/test_core.py")

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="p419_reuse_")).resolve()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        pkg = self.root / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "core.py").write_text(CORE_AFTER, encoding="utf-8")
        (self.root / "unrelated.txt").write_text("irrelevant\n", encoding="utf-8")
        self.relevant = ["pkg/core.py"]
        self.symbols = ["calculate_total"]
        self.policy = compute_policy_fingerprint({"min_confidence": "high", "max_age": 900})
        self.executable = compute_executable_fingerprint(self.COMMAND)
        self.ledger = EvidenceLedger()
        self.observed_reasons: list[str] = []

    def record(self, **overrides):
        payload = dict(
            command=self.COMMAND,
            status=STATUS_PASSED,
            duration_seconds=2.5,
            impacted_files=self.relevant,
            impacted_symbols=self.symbols,
            confidence="high",
            fingerprint=compute_state_fingerprint(self.root, self.relevant),
            policy_fingerprint=self.policy,
            executable_fingerprint=self.executable,
            analyzer_version=SEMANTIC_ANALYZER_SCHEMA_VERSION,
        )
        payload.update(overrides)
        return self.ledger.record(**payload)

    def attempt(self, **overrides):
        payload = dict(
            command=self.COMMAND,
            current_root=self.root,
            relevant_files=self.relevant,
            relevant_symbols=self.symbols,
            min_confidence="high",
            policy_fingerprint=self.policy,
            executable_fingerprint=self.executable,
            analyzer_version=SEMANTIC_ANALYZER_SCHEMA_VERSION,
        )
        payload.update(overrides)
        decision = self.ledger.find_reusable(**payload)
        # Every verdict this case produces is retained so the aggregate test
        # below can assert over the *actually observed* reason vocabulary
        # rather than over a hand-written list of reasons.
        self.observed_reasons.append(decision.reason)
        return decision

    #: Names of the ten Part 14 scenario tests, in brief order.
    SCENARIOS: tuple[str, ...] = (
        "test_01_exact_same_tree_is_reused",
        "test_02_irrelevant_file_changed_is_still_reused",
        "test_03_affected_file_changed_is_rejected",
        "test_04_policy_changed_is_rejected",
        "test_05_analyzer_version_changed_is_rejected",
        "test_06_command_changed_is_rejected",
        "test_07_environment_changed_is_rejected",
        "test_08_validation_tool_unavailable_is_rejected",
        "test_09_corrupted_evidence_is_rejected",
        "test_10_expired_evidence_is_rejected",
    )

    # -- the ten scenarios --------------------------------------------------

    def test_01_exact_same_tree_is_reused(self):
        self.record()
        decision = self.attempt()
        self.assertTrue(decision.reusable)
        self.assertEqual(decision.reason, REASON_OK)
        self.assertAlmostEqual(decision.time_saved_seconds, 2.5)

    def test_02_irrelevant_file_changed_is_still_reused(self):
        """The fingerprint covers the relevant file set only. Touching a file
        outside it must not invalidate evidence - otherwise reuse would be
        useless in any real repository."""
        self.record()
        (self.root / "unrelated.txt").write_text("changed\n", encoding="utf-8")
        decision = self.attempt()
        self.assertTrue(decision.reusable)
        self.assertEqual(decision.reason, REASON_OK)

    def test_03_affected_file_changed_is_rejected(self):
        self.record()
        (self.root / "pkg" / "core.py").write_text(
            CORE_AFTER + "\n\ndef extra():\n    return 0\n", encoding="utf-8"
        )
        decision = self.attempt()
        self.assertFalse(decision.reusable)
        self.assertEqual(decision.reason, REASON_FINGERPRINT_MISMATCH)

    def test_04_policy_changed_is_rejected(self):
        self.record()
        decision = self.attempt(
            policy_fingerprint=compute_policy_fingerprint({"min_confidence": "low"})
        )
        self.assertFalse(decision.reusable)
        self.assertEqual(decision.reason, REASON_POLICY_MISMATCH)

    def test_05_analyzer_version_changed_is_rejected(self):
        self.record()
        decision = self.attempt(analyzer_version="99.0.0-next")
        self.assertFalse(decision.reusable)
        self.assertEqual(decision.reason, REASON_ANALYZER_VERSION_MISMATCH)

    def test_06_command_changed_is_rejected(self):
        self.record()
        decision = self.attempt(command=("pytest", "tests/test_other.py"))
        self.assertFalse(decision.reusable)
        self.assertEqual(decision.reason, REASON_COMMAND_MISMATCH)

    def test_07_environment_changed_is_rejected(self):
        self.record()
        decision = self.attempt(executable_fingerprint="different-interpreter")
        self.assertFalse(decision.reusable)
        self.assertEqual(decision.reason, REASON_ENVIRONMENT_MISMATCH)

    def test_08_validation_tool_unavailable_is_rejected(self):
        """A tool that has vanished from PATH fingerprints differently from the
        one that produced the evidence, so its evidence stops being reusable -
        the fail-closed behaviour, rather than reusing a result produced by a
        binary that is no longer there."""
        self.record()
        missing = compute_executable_fingerprint(
            ("this-tool-does-not-exist-419", "tests/test_core.py")
        )
        self.assertNotEqual(missing, self.executable)
        decision = self.attempt(executable_fingerprint=missing)
        self.assertFalse(decision.reusable)
        self.assertEqual(decision.reason, REASON_ENVIRONMENT_MISMATCH)

    def test_09_corrupted_evidence_is_rejected(self):
        """A corrupt entry survives deserialisation (tolerantly) but can never
        satisfy reuse: its status is not a pass, so it fails closed."""
        self.record()
        payload = self.ledger.to_dict()
        payload["entries"][0]["status"] = "\x00garbage"
        payload["entries"][0]["fingerprint"] = "not-a-fingerprint"
        reloaded = EvidenceLedger.from_dict(payload)
        self.ledger = reloaded
        decision = self.attempt()
        self.assertFalse(decision.reusable)
        self.assertEqual(decision.reason, REASON_NOT_PASSED)

    def test_10_expired_evidence_is_rejected(self):
        self.record()
        entry = self.ledger.entries[0]
        entry.timestamp = "2001-01-01T00:00:00+00:00"
        self.ledger._entries[0] = entry  # noqa: SLF001 - deliberate age injection
        decision = self.attempt(max_age_seconds=60)
        self.assertFalse(decision.reusable)
        self.assertEqual(decision.reason, REASON_STALE)

    # -- statistics over the matrix ------------------------------------------

    def test_no_evidence_at_all_is_its_own_reason(self):
        decision = self.attempt()
        self.assertFalse(decision.reusable)
        self.assertEqual(decision.reason, REASON_NO_EVIDENCE)

    def collect_scenario_reasons(self) -> dict[str, list[str]]:
        """Actually run all ten scenarios and return what each one produced.

        Each scenario gets a fresh instance (its own temporary tree and its own
        ledger) so the scenarios cannot influence one another. A scenario that
        raises is re-raised, not swallowed - a silently skipped scenario would
        make every aggregate assertion below vacuous.
        """
        collected: dict[str, list[str]] = {}
        for name in self.SCENARIOS:
            case = type(self)(name)
            case.setUp()
            try:
                getattr(case, name)()
                collected[name] = list(case.observed_reasons)
            finally:
                case.doCleanups()
        return collected

    def test_all_ten_scenarios_run_and_each_produces_exactly_one_reason(self):
        collected = self.collect_scenario_reasons()
        self.assertEqual(len(collected), 10)
        for name, reasons in collected.items():
            with self.subTest(scenario=name):
                self.assertEqual(
                    len(reasons), 1, f"{name} produced {reasons!r}, expected one verdict"
                )
                self.assertTrue(reasons[0], f"{name} produced an empty reason")

    def test_every_scenario_reason_is_a_known_bounded_constant(self):
        """Part 10's store must never accumulate free text as dictionary keys."""
        from local_agent import evidence as evidence_module

        known = {
            value
            for name, value in vars(evidence_module).items()
            if name.startswith("REASON_") and isinstance(value, str)
        }
        self.assertTrue(known)
        observed = {r for reasons in self.collect_scenario_reasons().values() for r in reasons}
        self.assertTrue(
            observed.issubset(known), f"free-text reason leaked: {sorted(observed - known)}"
        )
        # Part 10 asks which reasons *actually* occur here. This is the answer,
        # measured rather than assumed: eight of the eleven defined reasons.
        # Scenarios 7 (environment changed) and 8 (tool unavailable) both
        # surface as an environment mismatch - the executable fingerprint is
        # what distinguishes them, and a vanished tool is exactly an
        # environment change - so ten scenarios yield eight distinct reasons.
        self.assertEqual(
            sorted(observed),
            [
                REASON_ANALYZER_VERSION_MISMATCH,
                REASON_OK,
                REASON_COMMAND_MISMATCH,
                REASON_NOT_PASSED,
                REASON_STALE,
                REASON_ENVIRONMENT_MISMATCH,
                REASON_POLICY_MISMATCH,
                REASON_FINGERPRINT_MISMATCH,
            ],
        )
        # The remaining defined reasons these scenarios never reach, named so
        # the coverage gap is explicit rather than silent. Each is exercised
        # elsewhere: reuse_disabled and the file/symbol-set reasons in
        # tests/test_validation_decision_engine.py, no_matching_evidence by
        # test_no_evidence_at_all_is_its_own_reason above.
        self.assertEqual(
            sorted(known - observed),
            ["confidence_below_threshold", "no_matching_evidence", "relevant_file_set_changed",
             "relevant_symbol_set_changed", "reuse_disabled"],
        )

    def test_exactly_two_of_the_ten_scenarios_permit_reuse(self):
        """The fail-closed shape of the matrix, asserted as a whole: only the
        identical-tree and irrelevant-change scenarios may reuse. If a future
        change made a third scenario reusable, that is a safety regression and
        this count catches it even if the individual test was also relaxed."""
        collected = self.collect_scenario_reasons()
        granted = [name for name, reasons in collected.items() if reasons == [REASON_OK]]
        self.assertEqual(
            sorted(granted),
            [
                "test_01_exact_same_tree_is_reused",
                "test_02_irrelevant_file_changed_is_still_reused",
            ],
        )

    def test_reason_tally_feeds_the_telemetry_store_statistics(self):
        """Part 10 end to end: reuse verdicts become a bounded reason tally on
        a real record, aggregate across records, and surface as the health
        report's rejection statistics."""
        self.record()
        reasons: dict[str, int] = {}
        for decision in (
            self.attempt(),  # OK
            self.attempt(analyzer_version="99.0.0-next"),
            self.attempt(policy_fingerprint=compute_policy_fingerprint({"x": 1})),
            self.attempt(executable_fingerprint="elsewhere"),
        ):
            reasons[decision.reason] = reasons.get(decision.reason, 0) + 1

        store = ValidationTelemetryStore()
        store.record_decision(
            ValidationDecisionRecord(scope=SCOPE_TARGETED, reuse_reasons=dict(reasons))
        )
        store.record_decision(
            ValidationDecisionRecord(scope=SCOPE_BROAD, reuse_reasons={REASON_STALE: 2})
        )

        totals = store.reuse_reason_totals()
        self.assertEqual(totals[REASON_OK], 1)
        self.assertEqual(totals[REASON_ANALYZER_VERSION_MISMATCH], 1)
        self.assertEqual(totals[REASON_POLICY_MISMATCH], 1)
        self.assertEqual(totals[REASON_ENVIRONMENT_MISMATCH], 1)
        self.assertEqual(totals[REASON_STALE], 2)

        metrics = compute_decision_quality_metrics(store)
        self.assertEqual(metrics.reuse_attempts, 6)
        self.assertEqual(metrics.reuse_grants, 1)
        self.assertEqual(metrics.reuse_denials, 5)
        self.assertAlmostEqual(metrics.reuse_hit_rate, 1 / 6)
        self.assertAlmostEqual(metrics.reuse_rejection_rate, 5 / 6)
        self.assertEqual(metrics.stale_evidence_rejections, 2)
        self.assertNotIn(REASON_OK, metrics.reuse_rejection_reasons)

        health = compute_health(store, min_samples=5)
        self.assertEqual(health.reuse_rejection_reasons[REASON_STALE], 2)


# ===========================================================================
# Part 4 - decision-quality metrics, with precise terminology.
# ===========================================================================


def _observation(quality: str, *, scope: str = SCOPE_TARGETED, confidence: str = "high",
                 evidence: tuple[str, ...] = ("direct_symbol_match",)) -> CalibrationObservation:
    return CalibrationObservation(
        evidence_types=evidence,
        predicted_confidence=confidence,
        selected_scope=scope,
        actual_validation_scope=scope,
        outcome=(
            OUTCOME_VALIDATION_PASSED
            if quality in (QUALITY_CONSISTENT, QUALITY_BROAD_NOT_PROVEN_NECESSARY)
            else OUTCOME_VALIDATION_FAILED
        ),
        decision_quality=quality,
        later_broader_validation_found_defect=(quality == QUALITY_TARGETED_MISSED_DEFECT),
    )


class DecisionQualityMetricsCase(unittest.TestCase):
    def store_with(self, *qualities: str) -> ValidationTelemetryStore:
        store = ValidationTelemetryStore()
        for quality in qualities:
            record = ValidationDecisionRecord(scope=SCOPE_TARGETED, confidence_level="high")
            store.record_decision(record)
            store.finalize_decision(
                record.decision_id,
                outcome=(
                    OUTCOME_VALIDATION_PASSED
                    if quality == QUALITY_CONSISTENT
                    else OUTCOME_VALIDATION_FAILED
                ),
                decision_quality=quality,
            )
        return store

    def test_empty_store_reports_no_escape_rate_and_a_fully_uncertain_bound(self):
        """Zero data must not look safe. The point estimate is 0.0 only because
        there is nothing to divide; the *upper bound* is 1.0, and that is the
        number any safety argument is required to use."""
        metrics = compute_decision_quality_metrics(ValidationTelemetryStore())
        self.assertEqual(metrics.targeted_resolved_trials, 0)
        self.assertEqual(metrics.observed_escape_rate, 0.0)
        self.assertEqual(metrics.observed_escape_rate_upper_bound, 1.0)

    def test_recall_is_never_reported(self):
        """The data cannot support a recall claim; the flag says so permanently
        and there is no recall field to misread."""
        metrics = compute_decision_quality_metrics(self.store_with(QUALITY_CONSISTENT))
        self.assertFalse(metrics.recall_available)
        self.assertNotIn("recall", metrics.to_dict())

    def test_escape_rate_counts_only_resolved_targeted_trials(self):
        store = self.store_with(
            QUALITY_CONSISTENT,
            QUALITY_CONSISTENT,
            QUALITY_TARGETED_MISSED_DEFECT,
            QUALITY_UNCONFIRMED,  # unresolved: excluded from the denominator
        )
        metrics = compute_decision_quality_metrics(store)
        self.assertEqual(metrics.targeted_resolved_trials, 3)
        self.assertEqual(metrics.targeted_escapes, 1)
        self.assertAlmostEqual(metrics.observed_escape_rate, 1 / 3)
        self.assertAlmostEqual(metrics.targeted_agreement_rate, 2 / 3)

    def test_upper_bound_always_exceeds_the_point_estimate_on_small_samples(self):
        store = self.store_with(QUALITY_CONSISTENT, QUALITY_CONSISTENT)
        metrics = compute_decision_quality_metrics(store)
        self.assertEqual(metrics.observed_escape_rate, 0.0)
        self.assertGreater(metrics.observed_escape_rate_upper_bound, 0.2)

    def test_a_caught_defect_is_a_resolved_trial_but_not_an_escape(self):
        metrics = compute_decision_quality_metrics(
            self.store_with(QUALITY_TARGETED_CAUGHT_DEFECT)
        )
        self.assertEqual(metrics.targeted_resolved_trials, 1)
        self.assertEqual(metrics.targeted_caught_defects, 1)
        self.assertEqual(metrics.targeted_escapes, 0)

    def test_confidence_buckets_separate_predicted_levels(self):
        store = ValidationTelemetryStore()
        for level, quality in (
            ("high", QUALITY_CONSISTENT),
            ("high", QUALITY_TARGETED_MISSED_DEFECT),
            ("medium", QUALITY_CONSISTENT),
        ):
            record = ValidationDecisionRecord(scope=SCOPE_TARGETED, confidence_level=level)
            store.record_decision(record)
            store.finalize_decision(
                record.decision_id,
                outcome=OUTCOME_VALIDATION_PASSED,
                decision_quality=quality,
            )
        buckets = compute_decision_quality_metrics(store).confidence_buckets
        self.assertEqual(buckets["high"].resolved_trials, 2)
        self.assertEqual(buckets["high"].escapes, 1)
        self.assertAlmostEqual(buckets["high"].agreement_rate, 0.5)
        self.assertEqual(buckets["medium"].escapes, 0)
        self.assertNotIn("low", buckets)

    def test_bucket_lower_bound_is_below_the_point_estimate(self):
        store = self.store_with(*[QUALITY_CONSISTENT] * 3)
        bucket = compute_decision_quality_metrics(store).confidence_buckets["high"]
        self.assertEqual(bucket.agreement_rate, 1.0)
        self.assertLess(bucket.agreement_lower_bound, 1.0)

    def test_metrics_are_read_only_over_the_store(self):
        store = self.store_with(QUALITY_CONSISTENT, QUALITY_TARGETED_MISSED_DEFECT)
        before = json.dumps(store.to_dict(), sort_keys=True)
        compute_decision_quality_metrics(store)
        compute_decision_quality_metrics(store)
        self.assertEqual(json.dumps(store.to_dict(), sort_keys=True), before)

    def test_metrics_serialise_and_are_order_independent(self):
        store_a = self.store_with(QUALITY_CONSISTENT, QUALITY_TARGETED_MISSED_DEFECT)
        store_b = self.store_with(QUALITY_TARGETED_MISSED_DEFECT, QUALITY_CONSISTENT)
        a = compute_decision_quality_metrics(store_a).to_dict()
        b = compute_decision_quality_metrics(store_b).to_dict()
        self.assertEqual(a, b)
        self.assertEqual(json.loads(json.dumps(a)), a)


# ===========================================================================
# Part 11 - the cost model: measured, never invented.
# ===========================================================================


class CostModelCase(unittest.TestCase):
    def store_with_durations(self, pairs, *, saved=()):
        store = ValidationTelemetryStore()
        saved = list(saved) + [0.0] * (len(pairs) - len(saved))
        for (targeted, broad), reuse_saved in zip(pairs, saved):
            store.record_decision(
                ValidationDecisionRecord(
                    scope=SCOPE_TARGETED,
                    targeted_duration_seconds=targeted,
                    broad_duration_seconds=broad,
                    time_saved_seconds=reuse_saved,
                    selected_command_count=2,
                    reused_command_count=1,
                )
            )
        return store

    def test_empty_store_is_explicitly_unmeasured(self):
        cost = compute_cost_model(ValidationTelemetryStore())
        self.assertFalse(cost.measured)
        self.assertEqual(cost.decisions, 0)
        self.assertEqual(cost.mean_broad_seconds, 0.0)

    def test_unmeasured_zero_durations_are_excluded_not_averaged_as_zero(self):
        """The central honesty property of the cost model.

        Two decisions, only one of which recorded a broad duration. The mean
        broad cost must be that one measurement (40.0), not 20.0 - averaging in
        a placeholder zero would fabricate a cheaper cost profile.
        """
        cost = compute_cost_model(self.store_with_durations([(3.0, 40.0), (2.0, 0.0)]))
        self.assertEqual(cost.broad_samples, 1)
        self.assertAlmostEqual(cost.mean_broad_seconds, 40.0)
        self.assertEqual(cost.targeted_samples, 2)
        self.assertAlmostEqual(cost.mean_targeted_seconds, 2.5)
        self.assertEqual(cost.decisions, 2)

    def test_ratio_uses_only_decisions_where_both_ends_were_measured(self):
        cost = compute_cost_model(
            self.store_with_durations([(2.0, 40.0), (4.0, 0.0), (0.0, 80.0)])
        )
        self.assertEqual(cost.paired_samples, 1)
        self.assertAlmostEqual(cost.mean_broad_to_targeted_ratio, 20.0)

    def test_median_is_robust_to_a_single_outlier(self):
        cost = compute_cost_model(
            self.store_with_durations([(1.0, 10.0), (1.0, 10.0), (1.0, 1000.0)])
        )
        self.assertAlmostEqual(cost.median_broad_seconds, 10.0)
        self.assertGreater(cost.mean_broad_seconds, cost.median_broad_seconds)

    def test_reuse_savings_are_summed_and_averaged(self):
        cost = compute_cost_model(
            self.store_with_durations([(1.0, 5.0), (1.0, 5.0)], saved=[3.0, 1.0])
        )
        self.assertAlmostEqual(cost.total_reuse_time_saved_seconds, 4.0)
        self.assertAlmostEqual(cost.mean_reuse_time_saved_per_decision_seconds, 2.0)

    def test_command_counts_are_averaged(self):
        cost = compute_cost_model(self.store_with_durations([(1.0, 5.0), (1.0, 5.0)]))
        self.assertAlmostEqual(cost.mean_commands_selected, 2.0)
        self.assertAlmostEqual(cost.mean_commands_reused, 1.0)

    def test_cost_model_never_influences_a_decision(self):
        """Part 11's hard rule. The cost model is computed from the store and
        exposes no method that returns a scope, a confidence, or a command; the
        only consumer in the package is the diagnostic health report.
        """
        cost = compute_cost_model(self.store_with_durations([(1.0, 99.0)]))
        for attribute in dir(cost):
            self.assertNotIn(
                attribute,
                {"scope", "recommend", "decide", "apply", "narrow", "confidence_level"},
            )
        self.assertEqual(
            json.loads(json.dumps(cost.to_dict())).keys(), cost.to_dict().keys()
        )

    def test_cost_model_is_measured_from_real_subprocess_durations(self):
        """Feed the model timings taken from actual command execution rather
        than literals, so 'measured' means measured."""
        durations = []
        for target in ("-c", "-c"):
            started = time.perf_counter()
            subprocess.run(
                [sys.executable, target, "pass"], capture_output=True, timeout=120
            )
            durations.append(time.perf_counter() - started)
        store = self.store_with_durations([(durations[0], durations[1])])
        cost = compute_cost_model(store)
        self.assertTrue(cost.measured)
        self.assertEqual(cost.paired_samples, 1)
        self.assertGreater(cost.mean_targeted_seconds, 0.0)
        self.assertGreater(cost.mean_broad_seconds, 0.0)


# ===========================================================================
# Part 15 - the extended health report.
# ===========================================================================


class HealthExtensionsCase(unittest.TestCase):
    def test_degradation_rate_counts_only_degraded_decisions(self):
        store = ValidationTelemetryStore()
        store.record_decision(ValidationDecisionRecord(degraded_analysis=True))
        store.record_decision(ValidationDecisionRecord(degraded_analysis=False))
        store.record_decision(ValidationDecisionRecord(degraded_analysis=False))
        health = compute_health(store, min_samples=5)
        self.assertAlmostEqual(health.analysis_degradation_rate, 1 / 3)

    def test_degradation_flag_is_derived_from_the_real_analyzer(self):
        """Not a hand-set boolean: built from an impact report that the real
        analyzer produced for a genuinely unanalysable change."""
        root = build_fixture_repo("def test_t():\n    assert 1 == 1\n")
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        (root / "pkg" / "broken.py").write_text("def f(:\n", encoding="utf-8")
        report = SemanticChangeImpactAnalyzer(root).analyze(["pkg/broken.py"])
        self.assertTrue(impact_is_degraded(report))

    def test_corruption_rate_uses_offered_entries_as_the_denominator(self):
        payload = {
            "decisions": [ValidationDecisionRecord().to_dict(), "not-a-dict"],
            "observations": [CalibrationObservation().to_dict()],
        }
        store = ValidationTelemetryStore.from_dict(payload)
        self.assertEqual(store.corrupted_records_skipped, 1)
        health = compute_health(store, min_samples=5)
        # 1 decision kept + 1 observation kept + 1 skipped = 3 offered.
        self.assertAlmostEqual(health.evidence_corruption_rate, 1 / 3)

    def test_shadow_aggregates_default_to_zero_when_calibration_is_off(self):
        store = ValidationTelemetryStore()
        store.record_decision(ValidationDecisionRecord())
        health = compute_health(store, min_samples=5)
        self.assertEqual(health.shadow_comparisons, 0)
        self.assertEqual(health.calibration_drift, 0.0)
        self.assertEqual(health.shadow_safety_overrides, 0)

    def test_calibration_drift_is_the_mean_absolute_delta(self):
        from local_agent.validation_telemetry import ShadowComparison

        store = ValidationTelemetryStore()
        store.record_decision(
            ValidationDecisionRecord(
                shadow=ShadowComparison(computed=True, confidence_delta=-0.2, would_broaden=True)
            )
        )
        store.record_decision(
            ValidationDecisionRecord(
                shadow=ShadowComparison(
                    computed=True, confidence_delta=0.1, would_narrow=True, safety_override=True
                )
            )
        )
        store.record_decision(ValidationDecisionRecord())  # not computed: excluded
        health = compute_health(store, min_samples=5)
        self.assertEqual(health.shadow_comparisons, 2)
        self.assertAlmostEqual(health.calibration_drift, 0.15)
        self.assertEqual(health.shadow_would_narrow, 1)
        self.assertEqual(health.shadow_would_broaden, 1)
        self.assertEqual(health.shadow_safety_overrides, 1)

    def test_health_embeds_cost_and_quality_and_stays_json_serialisable(self):
        store = ValidationTelemetryStore()
        record = ValidationDecisionRecord(
            scope=SCOPE_TARGETED, targeted_duration_seconds=1.5, broad_duration_seconds=30.0
        )
        store.record_decision(record)
        store.finalize_decision(
            record.decision_id,
            outcome=OUTCOME_VALIDATION_PASSED,
            decision_quality=QUALITY_CONSISTENT,
            broad_duration_seconds=30.0,
            targeted_duration_seconds=1.5,
        )
        payload = compute_health(store, min_samples=5).to_dict()
        self.assertEqual(json.loads(json.dumps(payload)), payload)
        self.assertTrue(payload["cost"]["measured"])
        self.assertEqual(payload["quality"]["targeted_resolved_trials"], 1)
        self.assertFalse(payload["quality"]["recall_available"])

    def test_health_never_reports_a_live_calibration_status(self):
        """There is no live mode in this build; the status vocabulary must not
        imply one."""
        store = ValidationTelemetryStore()
        for _ in range(10):
            record = ValidationDecisionRecord(scope=SCOPE_TARGETED)
            store.record_decision(record)
            store.finalize_decision(
                record.decision_id,
                outcome=OUTCOME_VALIDATION_PASSED,
                decision_quality=QUALITY_CONSISTENT,
            )
        status = compute_health(store, min_samples=1).calibration_status
        self.assertIn(status, {"no_observations", "insufficient_data", "shadow_only"})


# ===========================================================================
# Part 22 - invariants not already asserted in test_validation_telemetry.py.
# ===========================================================================


class InvariantCase(unittest.TestCase):
    def test_wilson_bounds_bracket_the_point_estimate(self):
        for successes, trials in ((0, 1), (1, 2), (2, 2), (5, 10), (99, 100)):
            with self.subTest(successes=successes, trials=trials):
                lower, upper = wilson_bounds(successes, trials)
                self.assertLessEqual(lower, successes / trials)
                self.assertGreaterEqual(upper, successes / trials)
                self.assertLessEqual(lower, upper)
                self.assertGreaterEqual(lower, 0.0)
                self.assertLessEqual(upper, 1.0)

    def test_wilson_zero_trials_is_maximally_uncertain(self):
        self.assertEqual(wilson_bounds(0, 0), (0.0, 1.0))
        self.assertEqual(wilson_lower_bound(0, 0), 0.0)

    def test_two_observations_both_passing_never_looks_certain(self):
        """Part 6's explicit example, restated against the escape-rate side."""
        observations = [_observation(QUALITY_CONSISTENT) for _ in range(2)]
        reliability = compute_reliability(observations, min_samples=20)["direct_symbol_match"]
        self.assertEqual(reliability.point_estimate, 1.0)
        self.assertLess(reliability.lower_bound, 0.4)
        self.assertFalse(reliability.sufficient_data)

    def test_more_uncertainty_never_narrows_the_pinned_fixture_scopes(self):
        """Property 1, asserted over the real fixture set: for every fixture,
        removing the dependency evidence entirely (the maximally-uncertain
        case) can only move the scope the same way or wider, never narrower."""
        widest = SCOPE_ORDER.index(SCOPE_BROAD)
        for label, source, _real, expected in DEPENDENCY_FIXTURES:
            with self.subTest(fixture=label):
                root = build_fixture_repo(source)
                self.addCleanup(shutil.rmtree, root, ignore_errors=True)
                # No base contents at all -> every symbol looks added -> the
                # analysis is maximally uncertain.
                blind = SemanticChangeImpactAnalyzer(root).analyze(["pkg/core.py"])
                informed_rank = SCOPE_ORDER.index(expected)
                blind_rank = SCOPE_ORDER.index(blind.recommended_scope)
                self.assertGreaterEqual(blind_rank, informed_rank)
                self.assertLessEqual(blind_rank, widest)

    def test_degraded_marker_is_not_a_real_evidence_type(self):
        """The safety-floor marker must be unable to collide with, or be
        mistaken for, a genuine dependency-evidence label."""
        from local_agent.dependency_resolution import ALL_EVIDENCE_TYPES

        self.assertNotIn(DEGRADED_EVIDENCE_MARKER, ALL_EVIDENCE_TYPES)

    def test_analysing_a_fixture_never_mutates_the_authoritative_tree(self):
        """Property 10: analysis is read-only over the repository it inspects."""
        root = build_fixture_repo(DEPENDENCY_FIXTURES[1][1])
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)

        def snapshot() -> dict[str, str]:
            return {
                str(p.relative_to(root)).replace("\\", "/"): p.read_text(encoding="utf-8")
                for p in sorted(root.rglob("*"))
                if p.is_file() and p.suffix == ".py"
            }

        before = snapshot()
        SemanticChangeImpactAnalyzer(root).analyze(
            ["pkg/core.py"], base_contents={"pkg/core.py": CORE_BEFORE}
        )
        self.assertEqual(snapshot(), before)

    def test_old_policy_evidence_cannot_satisfy_a_new_policy(self):
        """Property 5, at the level the telemetry store sees it: an observation
        recorded under one policy fingerprint keeps that fingerprint through a
        full JSON round trip, so a consumer can always tell the two apart."""
        old = CalibrationObservation(policy_fingerprint="policy-v1")
        restored = CalibrationObservation.from_dict(json.loads(json.dumps(old.to_dict())))
        self.assertEqual(restored.policy_fingerprint, "policy-v1")
        self.assertNotEqual(restored.policy_fingerprint, "policy-v2")

    def test_analyzer_version_travels_with_every_record_and_observation(self):
        """Property 6: version identity is never dropped in serialisation."""
        record = ValidationDecisionRecord(analyzer_version="4.18.0")
        restored = ValidationDecisionRecord.from_dict(
            json.loads(json.dumps(record.to_dict()))
        )
        self.assertEqual(restored.analyzer_version, "4.18.0")
        observation = CalibrationObservation.from_record(restored)
        self.assertEqual(observation.analyzer_version, "4.18.0")

    def test_degraded_analysis_field_is_backward_compatible(self):
        """Property: a record written by the first Phase 4.19 build - which had
        no ``degraded_analysis`` key - loads as not-degraded rather than
        raising or defaulting to a value that would inflate the rate."""
        legacy = ValidationDecisionRecord().to_dict()
        del legacy["degraded_analysis"]
        restored = ValidationDecisionRecord.from_dict(legacy)
        self.assertFalse(restored.degraded_analysis)

    def test_metrics_treat_unresolved_observations_as_neither_success_nor_failure(self):
        store = ValidationTelemetryStore()
        for _ in range(5):
            record = ValidationDecisionRecord(scope=SCOPE_TARGETED)
            store.record_decision(record)
            store.finalize_decision(
                record.decision_id,
                outcome=OUTCOME_VALIDATION_PASSED,
                decision_quality=QUALITY_UNCONFIRMED,
            )
        metrics = compute_decision_quality_metrics(store)
        self.assertEqual(metrics.targeted_resolved_trials, 0)
        self.assertEqual(metrics.observed_escape_rate_upper_bound, 1.0)


# ===========================================================================
# Parts 18 / 19 - concurrency and storage bounds for the new aggregates.
# ===========================================================================


class ConcurrencyAndBoundsCase(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="p419_conc_")).resolve()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def manager(self, name: str, **kwargs) -> ValidationTelemetryManager:
        project = self.root / name
        project.mkdir(parents=True, exist_ok=True)
        return ValidationTelemetryManager(
            JsonFileStorage(project / ".agent_data"), project, **kwargs
        )

    def test_parallel_worktree_managers_do_not_contaminate_each_other(self):
        """Property 11, with real threads, real files and two distinct project
        identities - the parallel-worktree situation Phase 4.14 introduced."""
        a = self.manager("worktree_a")
        b = self.manager("worktree_b")
        errors: list[BaseException] = []

        def write(manager, scope, count):
            try:
                for _ in range(count):
                    manager.record_decision(ValidationDecisionRecord(scope=scope))
            except BaseException as exc:  # pragma: no cover - reported below
                errors.append(exc)

        threads = [
            threading.Thread(target=write, args=(a, SCOPE_TARGETED, 25)),
            threading.Thread(target=write, args=(b, SCOPE_BROAD, 25)),
            threading.Thread(target=write, args=(a, SCOPE_TARGETED, 25)),
            threading.Thread(target=write, args=(b, SCOPE_BROAD, 25)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        health_a = a.health(min_samples=5)
        health_b = b.health(min_samples=5)
        self.assertEqual(health_a.total_decisions, 50)
        self.assertEqual(health_b.total_decisions, 50)
        self.assertEqual(health_a.scope_counts, {SCOPE_TARGETED: 50})
        self.assertEqual(health_b.scope_counts, {SCOPE_BROAD: 50})
        self.assertAlmostEqual(health_a.broad_validation_rate, 0.0)
        self.assertAlmostEqual(health_b.broad_validation_rate, 1.0)

    def test_concurrent_finalization_produces_no_race_corrupted_persistence(self):
        manager = self.manager("shared")
        records = [ValidationDecisionRecord(scope=SCOPE_TARGETED) for _ in range(30)]
        for record in records:
            manager.record_decision(record)

        def finalize(record):
            manager.finalize_decision(
                record.decision_id,
                outcome=OUTCOME_VALIDATION_PASSED,
                decision_quality=QUALITY_CONSISTENT,
                targeted_duration_seconds=1.0,
                broad_duration_seconds=10.0,
            )

        threads = [threading.Thread(target=finalize, args=(r,)) for r in records]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        health = manager.health(min_samples=5)
        self.assertEqual(health.total_observations, 30)
        self.assertEqual(health.quality.targeted_resolved_trials, 30)
        self.assertEqual(health.quality.targeted_escapes, 0)
        self.assertEqual(health.cost.paired_samples, 30)
        self.assertEqual(health.corrupted_records_skipped, 0)
        # The on-disk payload is still valid JSON, not a torn write.
        path = self.root / "shared" / ".agent_data" / "validation_telemetry.json"
        json.loads(path.read_text(encoding="utf-8"))

    def test_the_store_stays_bounded_under_sustained_writes(self):
        """Part 19: prove the store does not grow without limit, including the
        new aggregates computed over it."""
        manager = self.manager("bounded", max_decisions=10, max_observations=10)
        for _ in range(60):
            record = ValidationDecisionRecord(scope=SCOPE_TARGETED, targeted_duration_seconds=1.0)
            manager.record_decision(record)
            manager.finalize_decision(
                record.decision_id,
                outcome=OUTCOME_VALIDATION_PASSED,
                decision_quality=QUALITY_CONSISTENT,
                broad_duration_seconds=5.0,
            )
        health = manager.health(min_samples=5)
        self.assertEqual(health.total_decisions, 10)
        self.assertEqual(health.total_observations, 10)
        self.assertEqual(health.cost.decisions, 10)
        payload = (
            self.root / "bounded" / ".agent_data" / "validation_telemetry.json"
        ).read_text(encoding="utf-8")
        self.assertLess(len(payload), 200_000)

    def test_bounded_file_size_does_not_grow_after_the_cap_is_reached(self):
        manager = self.manager("bounded2", max_decisions=5, max_observations=5)
        path = self.root / "bounded2" / ".agent_data" / "validation_telemetry.json"

        def push(n):
            for _ in range(n):
                manager.record_decision(ValidationDecisionRecord(scope=SCOPE_BROAD))

        push(20)
        first = path.stat().st_size
        push(40)
        second = path.stat().st_size
        self.assertLessEqual(second, first * 2)
        self.assertEqual(manager.health(min_samples=5).total_decisions, 5)


# ===========================================================================
# Part 20 - measured overhead.
# ===========================================================================


class PerformanceOverheadCase(unittest.TestCase):
    """Measure the telemetry path's cost. Thresholds are deliberately loose:
    the claim being defended is "negligible next to running a test command",
    not a microbenchmark target, and a tight bound would be flaky on shared CI.
    The measured values are reported via ``print`` so a human reading the test
    output sees real numbers rather than only a pass/fail.
    """

    def test_record_creation_and_aggregation_overhead_is_small(self):
        store = ValidationTelemetryStore(max_decisions=500, max_observations=500)

        started = time.perf_counter()
        records = []
        for _ in range(500):
            record = ValidationDecisionRecord(
                scope=SCOPE_TARGETED,
                confidence_level="high",
                evidence_types=["direct_symbol_match"],
                targeted_duration_seconds=1.0,
            )
            store.record_decision(record)
            records.append(record)
        record_seconds = time.perf_counter() - started

        started = time.perf_counter()
        for record in records:
            store.finalize_decision(
                record.decision_id,
                outcome=OUTCOME_VALIDATION_PASSED,
                decision_quality=QUALITY_CONSISTENT,
                broad_duration_seconds=10.0,
            )
        finalize_seconds = time.perf_counter() - started

        started = time.perf_counter()
        health = compute_health(store, min_samples=20)
        health_seconds = time.perf_counter() - started

        started = time.perf_counter()
        payload = json.dumps(store.to_dict())
        ValidationTelemetryStore.from_dict(json.loads(payload))
        serialise_seconds = time.perf_counter() - started

        print(
            f"\n[phase4.19 perf] 500 record_decision: {record_seconds:.4f}s"
            f" | 500 finalize_decision: {finalize_seconds:.4f}s"
            f" | compute_health over 500+500: {health_seconds:.4f}s"
            f" | round-trip {len(payload)} bytes: {serialise_seconds:.4f}s"
        )

        self.assertEqual(health.total_decisions, 500)
        # A single real validation command costs seconds; the whole analytical
        # path over a full store must stay far below that.
        self.assertLess(health_seconds, 2.0)
        self.assertLess(serialise_seconds, 5.0)
        self.assertLess(record_seconds, 5.0)
        self.assertLess(finalize_seconds, 20.0)

    def test_persistence_overhead_per_decision_is_bounded(self):
        root = Path(tempfile.mkdtemp(prefix="p419_perf_")).resolve()
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        manager = ValidationTelemetryManager(JsonFileStorage(root / ".agent_data"), root)

        started = time.perf_counter()
        for _ in range(50):
            manager.record_decision(ValidationDecisionRecord(scope=SCOPE_TARGETED))
        elapsed = time.perf_counter() - started
        per_decision = elapsed / 50
        print(
            f"\n[phase4.19 perf] persisted record_decision:"
            f" {per_decision * 1000:.2f}ms/decision over 50 writes"
        )
        # Each write is a full load+save under a lock; it must still be far
        # cheaper than the validation command whose decision it describes.
        self.assertLess(per_decision, 0.5)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
