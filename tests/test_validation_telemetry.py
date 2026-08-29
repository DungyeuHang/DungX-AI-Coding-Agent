"""Phase 4.19 - empirical validation intelligence telemetry, outcome linking,
and shadow calibration.

Layered like the Phase 4.17/4.18 suites: pure-function tests for the
statistics and classification primitives, then real-object tests against
:class:`~local_agent.semantic_impact.ChangeImpactReport` /
:class:`~local_agent.validation_decision.ValidationDecision`, then integration
tests against the real :class:`~local_agent.orchestrator.Orchestrator` wiring
and a real :class:`~local_agent.storage.JsonFileStorage` - only the LLM
provider is ever mocked.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

from local_agent.config import AgentConfig
from local_agent.dependency_resolution import (
    DependencyEvidence,
    DIRECT_SYMBOL,
    DYNAMIC_IMPORT_UNRESOLVED,
)

#: A tier name (see local_agent.semantic_impact.TIER_CALL_GRAPH), used here as
#: a plain fallback label - not a member of dependency_resolution's evidence
#: vocabulary at all, which is exactly the case these tests exercise.
CALL_GRAPH_MATCH = "call_graph_match"
from local_agent.evidence import compute_state_fingerprint
from local_agent.models import ImplementationResult, ProjectContext, RunReport
from local_agent.orchestrator import Orchestrator
from local_agent.semantic_impact import (
    ChangeImpactReport,
    ImpactEvidence,
    SemanticChangeImpactAnalyzer,
    ValidationTarget,
    CONFIDENCE_HIGH,
)
from local_agent.storage import JsonFileStorage, TaskStorage
from local_agent.validation_decision import ReuseAttempt, ValidationDecision
from local_agent import validation_telemetry as vt


# -- Wilson lower bound ---------------------------------------------------------


class WilsonLowerBoundCase(unittest.TestCase):
    def test_zero_trials_is_zero(self):
        self.assertEqual(vt.wilson_lower_bound(0, 0), 0.0)

    def test_small_sample_never_reaches_certainty(self):
        # Part 6's explicit example: 2/2 must not look 100% reliable.
        self.assertLess(vt.wilson_lower_bound(2, 2), 1.0)
        self.assertLess(vt.wilson_lower_bound(2, 2), 0.9)

    def test_more_trials_at_same_rate_increases_the_lower_bound(self):
        small = vt.wilson_lower_bound(9, 10)
        large = vt.wilson_lower_bound(900, 1000)
        self.assertLess(small, large)

    def test_bounded_to_unit_interval(self):
        for successes, trials in [(0, 1), (1, 1), (5, 5), (0, 100), (100, 100)]:
            value = vt.wilson_lower_bound(successes, trials)
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_successes_greater_than_trials_is_clamped_not_raised(self):
        # Defensive: a corrupted/impossible input must fail closed, not crash.
        value = vt.wilson_lower_bound(999, 10)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)


# -- outcome / decision-quality classification -----------------------------------


class ClassifyOutcomeCase(unittest.TestCase):
    def test_targeted_passed_then_broad_failed_is_the_escape_signal(self):
        outcome, quality = vt.classify_outcome(
            scope="targeted", targeted_ran=True, targeted_failed=False,
            broad_ran=True, broad_failed=True,
        )
        self.assertEqual(outcome, vt.OUTCOME_VALIDATION_FAILED)
        self.assertEqual(quality, vt.QUALITY_TARGETED_MISSED_DEFECT)

    def test_targeted_failed_is_a_caught_defect_not_a_missed_one(self):
        outcome, quality = vt.classify_outcome(
            scope="targeted", targeted_ran=True, targeted_failed=True,
            broad_ran=False, broad_failed=False,
        )
        self.assertEqual(outcome, vt.OUTCOME_VALIDATION_FAILED)
        self.assertEqual(quality, vt.QUALITY_TARGETED_CAUGHT_DEFECT)

    def test_broad_passing_does_not_prove_broad_was_necessary(self):
        outcome, quality = vt.classify_outcome(
            scope="broad", targeted_ran=False, targeted_failed=False,
            broad_ran=True, broad_failed=False,
        )
        self.assertEqual(outcome, vt.OUTCOME_VALIDATION_PASSED)
        self.assertEqual(quality, vt.QUALITY_BROAD_NOT_PROVEN_NECESSARY)

    def test_targeted_and_broad_both_passing_is_merely_consistent(self):
        outcome, quality = vt.classify_outcome(
            scope="targeted", targeted_ran=True, targeted_failed=False,
            broad_ran=True, broad_failed=False,
        )
        self.assertEqual(outcome, vt.OUTCOME_VALIDATION_PASSED)
        self.assertEqual(quality, vt.QUALITY_CONSISTENT)
        self.assertNotEqual(quality, "proven_sufficient")  # never overclaimed

    def test_expanded_scope_failure_carries_no_scope_judgement(self):
        outcome, quality = vt.classify_outcome(
            scope="expanded", targeted_ran=False, targeted_failed=False,
            broad_ran=True, broad_failed=True,
        )
        self.assertEqual(outcome, vt.OUTCOME_VALIDATION_FAILED)
        self.assertEqual(quality, vt.QUALITY_VALIDATION_FAILED)

    def test_no_targeted_commands_and_broad_never_ran_is_unconfirmed_pending(self):
        outcome, quality = vt.classify_outcome(
            scope="targeted", targeted_ran=False, targeted_failed=False,
            broad_ran=False, broad_failed=False,
        )
        self.assertEqual(outcome, vt.OUTCOME_PENDING)
        self.assertEqual(quality, vt.QUALITY_UNCONFIRMED)

    def test_targeted_scope_with_zero_targeted_commands_but_broad_passed(self):
        outcome, quality = vt.classify_outcome(
            scope="targeted", targeted_ran=False, targeted_failed=False,
            broad_ran=True, broad_failed=False,
        )
        self.assertEqual(outcome, vt.OUTCOME_VALIDATION_PASSED)
        self.assertEqual(quality, vt.QUALITY_UNCONFIRMED)


# -- evidence-type extraction / degradation gate ---------------------------------


class EvidenceExtractionCase(unittest.TestCase):
    def test_falls_back_to_tier_name_when_no_fine_grained_evidence(self):
        impact = ChangeImpactReport(changed_files=["a.py"])
        impact.validation_targets = [
            ValidationTarget(path="t.py", command=("pytest", "t.py"), tier=CALL_GRAPH_MATCH)
        ]
        self.assertEqual(vt.evidence_types_for_impact(impact), frozenset({CALL_GRAPH_MATCH}))

    def test_prefers_fine_grained_dependency_evidence_over_tier(self):
        evidence = DependencyEvidence(source_file="t.py", target_file="a.py", evidence_type=DIRECT_SYMBOL)
        impact = ChangeImpactReport(changed_files=["a.py"])
        impact.validation_targets = [
            ValidationTarget(
                path="t.py", command=("pytest", "t.py"), tier=CALL_GRAPH_MATCH,
                dependency_evidence=(evidence,),
            )
        ]
        self.assertEqual(vt.evidence_types_for_impact(impact), frozenset({DIRECT_SYMBOL}))

    def test_tier_only_evidence_is_not_treated_as_degraded(self):
        """Regression test: a target with only coarse tier evidence (no
        dependency_resolution label at all) must NOT be flagged degraded just
        because ``confidence_for`` fails closed to 0.0 for any unrecognised
        label. Caught during development - see module comments in
        ``impact_is_degraded`` for the full explanation."""
        impact = ChangeImpactReport(changed_files=["a.py"])
        impact.validation_targets = [
            ValidationTarget(path="t.py", command=("pytest", "t.py"), tier=CALL_GRAPH_MATCH)
        ]
        self.assertFalse(vt.impact_is_degraded(impact))

    def test_unresolved_dynamic_import_evidence_is_degraded(self):
        evidence = DependencyEvidence(
            source_file="t.py", target_file="a.py", evidence_type=DYNAMIC_IMPORT_UNRESOLVED
        )
        impact = ChangeImpactReport(changed_files=["a.py"])
        impact.validation_targets = [
            ValidationTarget(
                path="t.py", command=("pytest", "t.py"), tier="reverse_dependency_match",
                dependency_evidence=(evidence,),
            )
        ]
        self.assertTrue(vt.impact_is_degraded(impact))

    def test_recorded_degradation_reason_is_degraded(self):
        impact = ChangeImpactReport(changed_files=["a.py"])
        impact.evidence = ImpactEvidence(degradations=["semantic graph unavailable"])
        self.assertTrue(vt.impact_is_degraded(impact))

    def test_unresolved_symbol_is_degraded(self):
        impact = ChangeImpactReport(changed_files=["a.py"], unresolved_symbols=["Thing"])
        self.assertTrue(vt.impact_is_degraded(impact))

    def test_clean_high_confidence_direct_symbol_is_not_degraded(self):
        evidence = DependencyEvidence(source_file="t.py", target_file="a.py", evidence_type=DIRECT_SYMBOL)
        impact = ChangeImpactReport(changed_files=["a.py"])
        impact.validation_targets = [
            ValidationTarget(
                path="t.py", command=("pytest", "t.py"), tier="direct_symbol_match",
                dependency_evidence=(evidence,),
            )
        ]
        self.assertFalse(vt.impact_is_degraded(impact))


# -- reliability estimation -------------------------------------------------------


class ComputeReliabilityCase(unittest.TestCase):
    def _obs(self, evidence_types, quality, outcome=vt.OUTCOME_VALIDATION_PASSED):
        return vt.CalibrationObservation(
            evidence_types=tuple(evidence_types), decision_quality=quality, outcome=outcome
        )

    def test_zero_observations_yields_empty_map(self):
        self.assertEqual(vt.compute_reliability([], min_samples=5), {})

    def test_unresolved_observations_do_not_count_as_trials(self):
        obs = [self._obs(["x"], vt.QUALITY_UNCONFIRMED, vt.OUTCOME_PENDING)]
        result = vt.compute_reliability(obs, min_samples=1)
        self.assertEqual(result, {})

    def test_one_success_one_failure_gives_a_50pct_point_estimate(self):
        obs = [
            self._obs(["x"], vt.QUALITY_CONSISTENT),
            self._obs(["x"], vt.QUALITY_TARGETED_MISSED_DEFECT, vt.OUTCOME_VALIDATION_FAILED),
        ]
        result = vt.compute_reliability(obs, min_samples=1)
        self.assertEqual(result["x"].trials, 2)
        self.assertEqual(result["x"].successes, 1)
        self.assertEqual(result["x"].failures, 1)
        self.assertAlmostEqual(result["x"].point_estimate, 0.5)
        self.assertLess(result["x"].lower_bound, 0.5)

    def test_sufficient_data_flag_respects_min_samples(self):
        obs = [self._obs(["x"], vt.QUALITY_CONSISTENT) for _ in range(3)]
        low_bar = vt.compute_reliability(obs, min_samples=2)
        high_bar = vt.compute_reliability(obs, min_samples=10)
        self.assertTrue(low_bar["x"].sufficient_data)
        self.assertFalse(high_bar["x"].sufficient_data)

    def test_order_independence(self):
        """Property 9: observation ordering must not change the result."""
        obs = [
            self._obs(["x"], vt.QUALITY_CONSISTENT),
            self._obs(["x", "y"], vt.QUALITY_TARGETED_MISSED_DEFECT, vt.OUTCOME_VALIDATION_FAILED),
            self._obs(["y"], vt.QUALITY_CONSISTENT),
        ]
        forward = vt.compute_reliability(obs, min_samples=1)
        backward = vt.compute_reliability(list(reversed(obs)), min_samples=1)
        self.assertEqual(
            {k: (v.trials, v.successes, v.failures) for k, v in forward.items()},
            {k: (v.trials, v.successes, v.failures) for k, v in backward.items()},
        )

    def test_deterministic_for_identical_input(self):
        """Property 8: calibration must be deterministic for the same
        observation set."""
        obs = [self._obs(["x"], vt.QUALITY_CONSISTENT) for _ in range(5)]
        first = vt.compute_reliability(obs, min_samples=2)
        second = vt.compute_reliability(obs, min_samples=2)
        self.assertEqual(first["x"].to_dict(), second["x"].to_dict())

    def test_multiple_evidence_types_on_one_observation_each_get_credit(self):
        obs = [self._obs(["x", "y", "z"], vt.QUALITY_CONSISTENT)]
        result = vt.compute_reliability(obs, min_samples=1)
        self.assertEqual(set(result), {"x", "y", "z"})
        for label in ("x", "y", "z"):
            self.assertEqual(result[label].trials, 1)


# -- calibration signal / safety floor --------------------------------------------


class ComputeCalibrationSignalCase(unittest.TestCase):
    def _reliable(self, evidence_type, trials, successes):
        failures = trials - successes
        return {
            evidence_type: vt.EvidenceTypeReliability(
                evidence_type=evidence_type,
                trials=trials,
                successes=successes,
                failures=failures,
                point_estimate=successes / trials if trials else 0.0,
                lower_bound=vt.wilson_lower_bound(successes, trials),
                sufficient_data=trials >= 20,
            )
        }

    def test_no_evidence_types_yields_no_signal(self):
        signal = vt.compute_calibration_signal(
            frozenset(), {}, 0.5, min_samples=20, max_adjustment=0.15, degraded=False
        )
        self.assertEqual(signal.direction, "none")

    def test_no_reliability_data_for_present_types_yields_no_signal(self):
        signal = vt.compute_calibration_signal(
            frozenset({"unknown_type"}), {}, 0.5, min_samples=20, max_adjustment=0.15, degraded=False
        )
        self.assertEqual(signal.direction, "none")

    def test_insufficient_samples_blocks_upward_even_with_100pct_success(self):
        # Part 6's explicit example: 2 observations, both passed.
        reliability = self._reliable("x", 2, 2)
        signal = vt.compute_calibration_signal(
            frozenset({"x"}), reliability, 0.2, min_samples=20, max_adjustment=0.5, degraded=False
        )
        self.assertEqual(signal.direction, "none")

    def test_sufficient_samples_with_zero_failures_signals_up(self):
        reliability = self._reliable("x", 50, 50)
        signal = vt.compute_calibration_signal(
            frozenset({"x"}), reliability, 0.2, min_samples=20, max_adjustment=0.3, degraded=False
        )
        self.assertEqual(signal.direction, "up")
        self.assertGreater(signal.calibrated_confidence_score, 0.2)

    def test_any_recorded_failure_signals_down_regardless_of_sample_size(self):
        reliability = self._reliable("x", 3, 2)  # only 3 trials, well under min_samples
        signal = vt.compute_calibration_signal(
            frozenset({"x"}), reliability, 0.6, min_samples=20, max_adjustment=0.3, degraded=False
        )
        self.assertEqual(signal.direction, "down")
        self.assertLess(signal.calibrated_confidence_score, 0.6)

    def test_degraded_blocks_upward_even_with_perfect_reliability(self):
        reliability = self._reliable("x", 1000, 1000)
        signal = vt.compute_calibration_signal(
            frozenset({"x"}), reliability, 0.2, min_samples=20, max_adjustment=0.3, degraded=True
        )
        self.assertEqual(signal.direction, "none")
        self.assertTrue(signal.suppressed_by_safety_floor)

    def test_degraded_does_not_block_downward(self):
        reliability = self._reliable("x", 5, 2)
        signal = vt.compute_calibration_signal(
            frozenset({"x"}), reliability, 0.6, min_samples=20, max_adjustment=0.3, degraded=True
        )
        self.assertEqual(signal.direction, "down")

    def test_adjustment_is_capped_by_max_adjustment(self):
        reliability = self._reliable("x", 1000, 1000)
        signal = vt.compute_calibration_signal(
            frozenset({"x"}), reliability, 0.1, min_samples=20, max_adjustment=0.05, degraded=False
        )
        self.assertEqual(signal.direction, "up")
        self.assertLessEqual(signal.calibrated_confidence_score, 0.1 + 0.05 + 1e-9)

    def test_result_is_always_clamped_to_unit_interval(self):
        reliability = self._reliable("x", 1000, 1000)
        signal = vt.compute_calibration_signal(
            frozenset({"x"}), reliability, 0.95, min_samples=20, max_adjustment=0.9, degraded=False
        )
        self.assertLessEqual(signal.calibrated_confidence_score, 1.0)

    def test_weakest_of_several_evidence_types_governs_upward_signal(self):
        reliability = {**self._reliable("x", 1000, 1000), **self._reliable("y", 20, 20)}
        signal = vt.compute_calibration_signal(
            frozenset({"x", "y"}), reliability, 0.1, min_samples=20, max_adjustment=0.9, degraded=False
        )
        self.assertEqual(signal.direction, "up")
        # y's lower bound (fewer trials) must be the binding constraint.
        self.assertLess(signal.calibrated_confidence_score, reliability["x"].lower_bound)


class ShadowCalibrationEngineCase(unittest.TestCase):
    def _impact_and_decision(self, tier="call_graph_match", confidence_level="medium"):
        impact = ChangeImpactReport(changed_files=["a.py"], confidence=confidence_level)
        impact.validation_targets = [
            ValidationTarget(path="t.py", command=("pytest", "t.py"), tier=tier)
        ]
        decision = ValidationDecision(scope="expanded", confidence_level=confidence_level, confidence_score=0.5)
        return impact, decision

    def _passing_observations(self, evidence_type, count):
        out = []
        for _ in range(count):
            r = vt.ValidationDecisionRecord(scope="expanded", confidence_level="medium", evidence_types=[evidence_type])
            r.outcome = vt.OUTCOME_VALIDATION_PASSED
            r.decision_quality = vt.QUALITY_CONSISTENT
            out.append(vt.CalibrationObservation.from_record(r))
        return out

    def test_shadow_comparison_exposes_no_mutator_beyond_serialisation(self):
        """Architectural invariant: ShadowComparison is a plain, inert data
        holder. Its only callables are serialisation helpers - nothing on it
        can be invoked to change a real decision."""
        import inspect
        callables = {
            name for name in dir(vt.ShadowComparison)
            if not name.startswith("_") and callable(getattr(vt.ShadowComparison, name, None))
        }
        self.assertEqual(callables, {"to_dict", "from_dict"})

    def test_no_observations_produces_no_signal(self):
        impact, decision = self._impact_and_decision()
        engine = vt.ShadowCalibrationEngine(min_samples=20, max_adjustment=0.3)
        result = engine.evaluate(impact, decision, [])
        self.assertTrue(result.computed)
        self.assertFalse(result.would_narrow)
        self.assertFalse(result.would_broaden)
        self.assertEqual(result.shadow_scope, decision.scope)

    def test_strong_reliable_history_can_signal_narrowing(self):
        impact, decision = self._impact_and_decision()
        engine = vt.ShadowCalibrationEngine(min_samples=20, max_adjustment=0.5)
        observations = self._passing_observations("call_graph_match", 200)
        result = engine.evaluate(impact, decision, observations)
        self.assertTrue(result.would_narrow)
        self.assertEqual(result.shadow_scope, "targeted")
        self.assertFalse(result.safety_override)

    def test_degraded_impact_never_narrows_regardless_of_history(self):
        impact, decision = self._impact_and_decision()
        impact.evidence = ImpactEvidence(degradations=["unparseable file"])
        engine = vt.ShadowCalibrationEngine(min_samples=20, max_adjustment=0.5)
        observations = self._passing_observations("call_graph_match", 200)
        result = engine.evaluate(impact, decision, observations)
        self.assertFalse(result.would_narrow)
        self.assertTrue(result.safety_override)

    def test_history_of_escapes_can_signal_broadening(self):
        impact, decision = self._impact_and_decision(confidence_level="high")
        decision.confidence_score = 1.0
        engine = vt.ShadowCalibrationEngine(min_samples=20, max_adjustment=0.9)
        observations = []
        for _ in range(3):
            r = vt.ValidationDecisionRecord(scope="targeted", confidence_level="high", evidence_types=["call_graph_match"])
            r.outcome = vt.OUTCOME_VALIDATION_FAILED
            r.decision_quality = vt.QUALITY_TARGETED_MISSED_DEFECT
            observations.append(vt.CalibrationObservation.from_record(r))
        result = engine.evaluate(impact, decision, observations)
        self.assertTrue(result.would_broaden)


# -- decision record / observation model ------------------------------------------


class BuildDecisionRecordCase(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="p419_build_"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        (self.root / "a.py").write_text("x = 1\n", encoding="utf-8")

    def test_build_decision_record_captures_impact_and_decision(self):
        impact = ChangeImpactReport(changed_files=["a.py"], confidence="high", recommended_scope="targeted")
        impact.validation_targets = [ValidationTarget(path="t.py", command=("pytest", "t.py"), tier="direct_symbol_match")]
        decision = ValidationDecision(
            scope="targeted", confidence_level="high", confidence_score=1.0,
            selected_commands=[], reuse_attempts=[ReuseAttempt(command=("pytest",), reusable=True, reason="assumptions_still_hold", time_saved_seconds=2.0)],
        )
        record = vt.build_decision_record(impact, decision, root=self.root, reuse_reasons={"assumptions_still_hold": 1})
        self.assertEqual(record.scope, "targeted")
        self.assertEqual(record.confidence_level, "high")
        self.assertEqual(record.reused_command_count, 1)
        self.assertEqual(record.reuse_reasons, {"assumptions_still_hold": 1})
        self.assertEqual(record.changed_files, ["a.py"])
        self.assertTrue(record.tree_fingerprint)
        # Round trip.
        restored = vt.ValidationDecisionRecord.from_dict(record.to_dict())
        self.assertEqual(restored.to_dict(), record.to_dict())

    def test_no_raw_source_or_output_is_ever_stored(self):
        impact = ChangeImpactReport(changed_files=["a.py"], confidence="high", recommended_scope="targeted")
        decision = ValidationDecision(scope="targeted", confidence_level="high", confidence_score=1.0)
        record = vt.build_decision_record(impact, decision, root=self.root)
        payload = json.dumps(record.to_dict())
        self.assertNotIn("x = 1", payload)  # the file's actual content text

    def test_list_fields_are_bounded(self):
        impact = ChangeImpactReport(
            changed_files=[f"m{i}.py" for i in range(500)], confidence="low", recommended_scope="broad"
        )
        decision = ValidationDecision(scope="broad", confidence_level="low", confidence_score=0.0)
        record = vt.build_decision_record(impact, decision, root=self.root)
        self.assertLessEqual(len(record.changed_files), vt._MAX_LIST_FIELD_ENTRIES)


class FromDictToleranceCase(unittest.TestCase):
    """Every model here must tolerate garbage/partial/future-shaped input,
    the same convention Phase 4.17/4.18 established."""

    def test_decision_record_from_garbage(self):
        for payload in ({}, None, [], "x", {"changed_files": "not-a-list"}):
            record = vt.ValidationDecisionRecord.from_dict(payload)
            self.assertEqual(record.outcome, vt.OUTCOME_PENDING)

    def test_observation_from_garbage(self):
        for payload in ({}, None, [], 42):
            obs = vt.CalibrationObservation.from_dict(payload)
            self.assertEqual(obs.evidence_types, ())

    def test_shadow_comparison_from_garbage(self):
        for payload in ({}, None, "nope"):
            shadow = vt.ShadowComparison.from_dict(payload)
            self.assertFalse(shadow.computed)

    def test_store_from_garbage_top_level(self):
        for payload in ({}, None, [], "x"):
            store = vt.ValidationTelemetryStore.from_dict(payload)
            self.assertEqual(len(store.decisions), 0)

    def test_store_from_dict_skips_non_dict_entries_and_counts_them_corrupted(self):
        store = vt.ValidationTelemetryStore.from_dict(
            {"decisions": [{"scope": "targeted"}, "garbage", 42], "observations": ["also garbage"]}
        )
        self.assertEqual(len(store.decisions), 1)
        self.assertEqual(len(store.observations), 0)
        self.assertEqual(store.corrupted_records_skipped, 3)

    def test_store_from_dict_with_a_future_unknown_key_does_not_raise(self):
        store = vt.ValidationTelemetryStore.from_dict(
            {"decisions": [], "observations": [], "a_field_from_phase_4_25": {"nested": True}}
        )
        self.assertEqual(len(store.decisions), 0)


# -- bounded store ------------------------------------------------------------------


class ValidationTelemetryStoreCase(unittest.TestCase):
    def test_decisions_are_bounded_oldest_evicted_first(self):
        store = vt.ValidationTelemetryStore(max_decisions=3, max_observations=3)
        ids = []
        for i in range(5):
            record = vt.ValidationDecisionRecord(scope="targeted")
            ids.append(record.decision_id)
            store.record_decision(record)
        self.assertEqual(len(store.decisions), 3)
        remaining_ids = {r.decision_id for r in store.decisions}
        self.assertEqual(remaining_ids, set(ids[-3:]))

    def test_observations_are_bounded(self):
        store = vt.ValidationTelemetryStore(max_decisions=100, max_observations=2)
        for _ in range(5):
            record = vt.ValidationDecisionRecord(scope="targeted")
            store.record_decision(record)
            store.finalize_decision(
                record.decision_id, outcome=vt.OUTCOME_VALIDATION_PASSED, decision_quality=vt.QUALITY_CONSISTENT
            )
        self.assertEqual(len(store.observations), 2)

    def test_finalize_unknown_decision_id_returns_none_and_records_nothing(self):
        store = vt.ValidationTelemetryStore()
        result = store.finalize_decision(
            "does-not-exist", outcome=vt.OUTCOME_VALIDATION_PASSED, decision_quality=vt.QUALITY_CONSISTENT
        )
        self.assertIsNone(result)
        self.assertEqual(len(store.observations), 0)

    def test_finalize_is_idempotent_safe_to_call_twice(self):
        store = vt.ValidationTelemetryStore()
        record = vt.ValidationDecisionRecord(scope="targeted")
        store.record_decision(record)
        store.finalize_decision(record.decision_id, outcome=vt.OUTCOME_VALIDATION_PASSED, decision_quality=vt.QUALITY_CONSISTENT)
        store.finalize_decision(record.decision_id, outcome=vt.OUTCOME_VALIDATION_FAILED, decision_quality=vt.QUALITY_TARGETED_CAUGHT_DEFECT)
        self.assertEqual(len(store.decisions), 1)
        self.assertEqual(store.find_decision(record.decision_id).outcome, vt.OUTCOME_VALIDATION_FAILED)
        self.assertEqual(len(store.observations), 2)

    def test_reuse_reason_totals_aggregate_across_decisions(self):
        store = vt.ValidationTelemetryStore()
        r1 = vt.ValidationDecisionRecord(scope="targeted", reuse_reasons={"tree_state_changed": 2})
        r2 = vt.ValidationDecisionRecord(scope="targeted", reuse_reasons={"tree_state_changed": 1, "assumptions_still_hold": 1})
        store.record_decision(r1)
        store.record_decision(r2)
        self.assertEqual(store.reuse_reason_totals(), {"tree_state_changed": 3, "assumptions_still_hold": 1})

    def test_full_round_trip_preserves_everything(self):
        store = vt.ValidationTelemetryStore(max_decisions=10, max_observations=10)
        record = vt.ValidationDecisionRecord(scope="expanded", evidence_types=["x"])
        store.record_decision(record)
        store.finalize_decision(record.decision_id, outcome=vt.OUTCOME_VALIDATION_PASSED, decision_quality=vt.QUALITY_CONSISTENT, broad_duration_seconds=1.5)
        restored = vt.ValidationTelemetryStore.from_dict(store.to_dict())
        self.assertEqual(restored.to_dict(), store.to_dict())


class ComputeHealthCase(unittest.TestCase):
    def test_empty_store_reports_no_observations(self):
        health = vt.compute_health(vt.ValidationTelemetryStore(), min_samples=20)
        self.assertEqual(health.calibration_status, "no_observations")
        self.assertEqual(health.total_decisions, 0)

    def test_health_tallies_scope_counts_and_false_confidence(self):
        store = vt.ValidationTelemetryStore()
        for scope in ("targeted", "targeted", "broad"):
            record = vt.ValidationDecisionRecord(scope=scope, evidence_types=["call_graph_match"])
            store.record_decision(record)
            quality = vt.QUALITY_TARGETED_MISSED_DEFECT if scope == "targeted" else vt.QUALITY_BROAD_NOT_PROVEN_NECESSARY
            outcome = vt.OUTCOME_VALIDATION_FAILED if scope == "targeted" else vt.OUTCOME_VALIDATION_PASSED
            store.finalize_decision(record.decision_id, outcome=outcome, decision_quality=quality)
        health = vt.compute_health(store, min_samples=20)
        self.assertEqual(health.scope_counts, {"targeted": 2, "broad": 1})
        self.assertAlmostEqual(health.broad_validation_rate, 1 / 3)
        self.assertEqual(health.false_confidence_incidents, 2)

    def test_health_is_diagnostic_only_never_mutates_the_store(self):
        store = vt.ValidationTelemetryStore()
        record = vt.ValidationDecisionRecord(scope="targeted")
        store.record_decision(record)
        before = store.to_dict()
        vt.compute_health(store, min_samples=20)
        self.assertEqual(store.to_dict(), before)


# -- persistence manager: real storage, backward compatibility, concurrency -------


class ValidationTelemetryManagerCase(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="p419_manager_"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.storage = JsonFileStorage(self.root / ".agent_data")

    def test_pre_phase_4_19_repository_has_no_telemetry_file_and_loads_empty(self):
        self.assertFalse((self.root / ".agent_data" / "validation_telemetry.json").exists())
        store = self.storage.load_validation_telemetry()
        self.assertEqual(len(store.decisions), 0)

    def test_legacy_storage_backend_without_the_new_methods_is_a_safe_no_op(self):
        """Mirrors the existing knowledge-graph backward-compatibility test:
        a storage backend written before Phase 4.19 (implementing only the
        original abstract methods) must not break callers of the new ones,
        via the base class's non-abstract defaults."""

        class LegacyStorage(TaskStorage):
            def save_task(self, task): pass
            def load_task(self, task_id): raise FileNotFoundError()
            def list_tasks(self): return []
            def save_checkpoint(self, checkpoint): pass
            def load_checkpoint(self, checkpoint_id): raise FileNotFoundError()
            def save_scheduler_state(self, state): pass
            def load_scheduler_state(self): return None
            def save_provider_configs(self, configs): pass
            def load_provider_configs(self): return []
            def save_semantic_index(self, semantic_index): pass
            def load_semantic_index(self): return None
            def save_project_memory(self, memory): pass
            def load_project_memory(self): return None

        manager = vt.ValidationTelemetryManager(LegacyStorage(), self.root)
        record = vt.ValidationDecisionRecord(scope="targeted")
        manager.record_decision(record)  # must not raise
        health = manager.health(min_samples=5)
        self.assertEqual(health.total_decisions, 0)  # legacy backend never persisted it

    def test_record_and_finalize_round_trip_through_real_json_file(self):
        manager = vt.ValidationTelemetryManager(self.storage, self.root)
        record = vt.ValidationDecisionRecord(scope="targeted", evidence_types=["direct_symbol_match"])
        manager.record_decision(record)
        path = self.root / ".agent_data" / "validation_telemetry.json"
        self.assertTrue(path.exists())
        on_disk = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(on_disk["decisions"]), 1)

        manager.finalize_decision(
            record.decision_id, outcome=vt.OUTCOME_VALIDATION_PASSED, decision_quality=vt.QUALITY_CONSISTENT
        )
        on_disk_after = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk_after["decisions"][0]["outcome"], vt.OUTCOME_VALIDATION_PASSED)
        self.assertEqual(len(on_disk_after["observations"]), 1)

    def test_finalize_missing_decision_id_is_a_safe_no_op(self):
        manager = vt.ValidationTelemetryManager(self.storage, self.root)
        result = manager.finalize_decision("", outcome=vt.OUTCOME_VALIDATION_PASSED, decision_quality=vt.QUALITY_CONSISTENT)
        self.assertIsNone(result)
        self.assertFalse((self.root / ".agent_data" / "validation_telemetry.json").exists())

    def test_corrupted_telemetry_file_is_quarantined_not_fatal(self):
        data_dir = self.root / ".agent_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "validation_telemetry.json").write_text("{not valid json", encoding="utf-8")
        store = self.storage.load_validation_telemetry()
        self.assertEqual(len(store.decisions), 0)
        quarantined = list(data_dir.glob("validation_telemetry.json.corrupt.*"))
        self.assertEqual(len(quarantined), 1)

    def test_concurrent_record_decision_from_multiple_threads_loses_nothing(self):
        """Part 18/19: no race-corrupted persistence. Each
        ValidationTelemetryManager instance mirrors the per-worktree-manager
        pattern already used for the knowledge graph (a fresh instance per
        thread, same underlying storage/root) - the per-repository lock must
        still serialise their read-modify-write cycles."""
        errors: list[BaseException] = []

        def worker(n: int) -> None:
            try:
                manager = vt.ValidationTelemetryManager(self.storage, self.root, max_decisions=1000)
                for i in range(10):
                    manager.record_decision(
                        vt.ValidationDecisionRecord(scope="targeted", changed_files=[f"w{n}_{i}.py"])
                    )
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        self.assertEqual(errors, [])
        final_store = self.storage.load_validation_telemetry()
        self.assertEqual(len(final_store.decisions), 80)
        # The file itself must still be valid JSON (no interleaved/partial write).
        path = self.root / ".agent_data" / "validation_telemetry.json"
        json.loads(path.read_text(encoding="utf-8"))

    def test_separate_project_roots_use_independent_locks_and_stores(self):
        other_root = Path(tempfile.mkdtemp(prefix="p419_other_"))
        self.addCleanup(shutil.rmtree, other_root, ignore_errors=True)
        other_storage = JsonFileStorage(other_root / ".agent_data")
        manager_a = vt.ValidationTelemetryManager(self.storage, self.root)
        manager_b = vt.ValidationTelemetryManager(other_storage, other_root)
        manager_a.record_decision(vt.ValidationDecisionRecord(scope="targeted"))
        self.assertEqual(len(other_storage.load_validation_telemetry().decisions), 0)
        self.assertEqual(len(self.storage.load_validation_telemetry().decisions), 1)


# -- orchestrator integration -------------------------------------------------------


class OrchestratorTelemetryCase(unittest.TestCase):
    """Exercises the actual orchestrator wiring
    (_record_validation_decision / _finalize_validation_decision), the real
    integration seam Phase 4.19 added - not a reimplementation of it."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="p419_orch_")).resolve()
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

    def make_orchestrator(self, **overrides: Any) -> Orchestrator:
        overrides.setdefault("validation_telemetry_enabled", True)
        config = AgentConfig(
            project=self.root,
            semantic_impact_analysis_enabled=True,
            knowledge_graph_enabled=False,
            **overrides,
        )
        storage = JsonFileStorage(self.root / ".agent_data")
        return Orchestrator(config, storage, None, threading.Lock(), threading.Lock())

    def candidate_impact_report(self) -> dict[str, Any]:
        impact = SemanticChangeImpactAnalyzer(self.root).analyze(
            ["pkg/core.py"], base_contents={"pkg/core.py": self.core_src}
        )
        self.assertEqual(impact.confidence, CONFIDENCE_HIGH)
        return impact.to_dict()

    def make_report(self, impact_report: dict[str, Any] | None = None) -> RunReport:
        report = RunReport(project=ProjectContext(root=str(self.root)))
        report.changed_files = ["pkg/core.py"]
        report.implementation_result = ImplementationResult(success=True, impact_report=impact_report)
        return report

    def test_disabled_telemetry_never_creates_a_store_or_a_decision_id(self):
        orch = self.make_orchestrator(validation_telemetry_enabled=False)
        self.assertIsNone(orch.telemetry_manager)
        report = self.make_report(self.candidate_impact_report())
        commands = orch._semantic_targeted_commands(report, ProjectContext(root=str(self.root)), [], lambda m: None)
        commands = orch._apply_evidence_reuse(report, commands, lambda m: None)
        decision_id = orch._record_validation_decision(report, commands, report.validation_reuse_attempts)
        self.assertEqual(decision_id, "")
        self.assertFalse((self.root / ".agent_data" / "validation_telemetry.json").exists())

    def test_enabled_telemetry_records_a_decision_and_links_the_missed_defect_outcome(self):
        orch = self.make_orchestrator()
        context = ProjectContext(root=str(self.root))
        report = self.make_report(self.candidate_impact_report())
        commands = orch._semantic_targeted_commands(report, context, [], lambda m: None)
        self.assertTrue(commands)
        commands = orch._apply_evidence_reuse(report, commands, lambda m: None)
        decision_id = orch._record_validation_decision(report, commands, report.validation_reuse_attempts)
        self.assertTrue(decision_id)
        report.validation_decision_id = decision_id

        stored = orch.telemetry_manager.observations()
        self.assertEqual(stored, [])  # not finalized yet

        orch._finalize_validation_decision(
            report, targeted_ran=True, targeted_failed=False,
            broad_ran=True, broad_failed=True,
            targeted_duration=0.1, broad_duration=4.2,
        )
        health = orch.telemetry_manager.health(min_samples=5)
        self.assertEqual(health.total_observations, 1)
        self.assertEqual(health.false_confidence_incidents, 1)

    def test_enabled_telemetry_records_consistent_outcome_when_everything_passes(self):
        orch = self.make_orchestrator()
        context = ProjectContext(root=str(self.root))
        report = self.make_report(self.candidate_impact_report())
        commands = orch._semantic_targeted_commands(report, context, [], lambda m: None)
        commands = orch._apply_evidence_reuse(report, commands, lambda m: None)
        report.validation_decision_id = orch._record_validation_decision(
            report, commands, report.validation_reuse_attempts
        )
        orch._finalize_validation_decision(
            report, targeted_ran=True, targeted_failed=False,
            broad_ran=True, broad_failed=False,
            targeted_duration=0.1, broad_duration=1.0,
        )
        health = orch.telemetry_manager.health(min_samples=5)
        self.assertEqual(health.false_confidence_incidents, 0)

    def test_disabled_semantic_impact_never_records_telemetry_either(self):
        orch = self.make_orchestrator()
        orch.config.semantic_impact_analysis_enabled = False
        report = self.make_report()  # no impact report
        decision_id = orch._record_validation_decision(report, [], [])
        self.assertEqual(decision_id, "")

    def test_telemetry_recording_failure_never_raises(self):
        """A telemetry defect must never interrupt a real validation run."""
        orch = self.make_orchestrator()
        report = self.make_report(self.candidate_impact_report())
        orch._semantic_targeted_commands(report, ProjectContext(root=str(self.root)), [], lambda m: None)
        self.assertTrue(report.semantic_impact)
        report.semantic_impact["changed_files"] = object()  # deliberately unparseable
        decision_id = orch._record_validation_decision(report, [], [])
        self.assertEqual(decision_id, "")  # failed closed, no exception raised

    def test_finalize_with_unknown_decision_id_never_raises(self):
        orch = self.make_orchestrator()
        report = self.make_report()
        report.validation_decision_id = "not-a-real-id"
        orch._finalize_validation_decision(
            report, targeted_ran=True, targeted_failed=False, broad_ran=True,
            broad_failed=False, targeted_duration=0.0, broad_duration=0.0,
        )  # must not raise

    def test_shadow_calibration_is_recorded_only_when_explicitly_enabled(self):
        orch_off = self.make_orchestrator(validation_calibration_enabled=False)
        report_off = self.make_report(self.candidate_impact_report())
        commands = orch_off._semantic_targeted_commands(report_off, ProjectContext(root=str(self.root)), [], lambda m: None)
        commands = orch_off._apply_evidence_reuse(report_off, commands, lambda m: None)
        decision_id = orch_off._record_validation_decision(report_off, commands, report_off.validation_reuse_attempts)
        record = orch_off.telemetry_manager._load().find_decision(decision_id)
        self.assertFalse(record.shadow.computed)

        orch_on = self.make_orchestrator(validation_calibration_enabled=True)
        report_on = self.make_report(self.candidate_impact_report())
        commands2 = orch_on._semantic_targeted_commands(report_on, ProjectContext(root=str(self.root)), [], lambda m: None)
        commands2 = orch_on._apply_evidence_reuse(report_on, commands2, lambda m: None)
        decision_id2 = orch_on._record_validation_decision(report_on, commands2, report_on.validation_reuse_attempts)
        record2 = orch_on.telemetry_manager._load().find_decision(decision_id2)
        self.assertTrue(record2.shadow.computed)

    def test_shadow_calibration_never_alters_the_actually_selected_commands(self):
        """Architectural invariant, asserted end to end: enabling shadow
        calibration must not change which commands `_semantic_targeted_commands`
        / `_apply_evidence_reuse` actually selected."""
        orch_off = self.make_orchestrator(validation_calibration_enabled=False)
        report_off = self.make_report(self.candidate_impact_report())
        commands_off = orch_off._semantic_targeted_commands(report_off, ProjectContext(root=str(self.root)), [], lambda m: None)
        commands_off = orch_off._apply_evidence_reuse(report_off, commands_off, lambda m: None)
        orch_off._record_validation_decision(report_off, commands_off, report_off.validation_reuse_attempts)

        orch_on = self.make_orchestrator(validation_calibration_enabled=True)
        report_on = self.make_report(self.candidate_impact_report())
        commands_on = orch_on._semantic_targeted_commands(report_on, ProjectContext(root=str(self.root)), [], lambda m: None)
        commands_on = orch_on._apply_evidence_reuse(report_on, commands_on, lambda m: None)
        orch_on._record_validation_decision(report_on, commands_on, report_on.validation_reuse_attempts)

        self.assertEqual(
            [c.command for c in commands_off], [c.command for c in commands_on]
        )


if __name__ == "__main__":
    unittest.main()
