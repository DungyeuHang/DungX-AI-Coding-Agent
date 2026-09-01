"""Phase 4.24 (continued) -- Evidence Semantics & Completion False-Positive
Hardening.

Phase 4.23 disclosed several residual limitations without treating them as
proven-safe: diff-correlation uses raw substring matching over the whole
diff text; path-scoping strength depends on planner path prediction and is
a no-op when target_paths are empty; lexical ambiguity remains for a
symbol whose name genuinely contains multiple obligations' words. This
module treats each as a hypothesis to reproduce against live code, not a
fact to trust.

Two of the three investigated limitations turned out to conceal real,
independently reproduced false-positive-completion vulnerabilities. The
third (CONSTRAINT_ABSENCE case-sensitivity) was found during systematic
attack-matrix construction, not disclosed by Phase 4.23, but is fixed here
as part of the same "textual matching promotes a false conclusion" family.

VULN-4.24B-01 -- Diff-correlation counted a token appearing ONLY on a
REMOVED (``-``) diff line, an unchanged context line, or a file/hunk
header as evidence that new code addressing it was written. Reproduced: a
diff that only *deleted* a stale "# TODO: json export" comment (and
implemented something unrelated) satisfied a "JSON export" obligation.
Fix: ``_added_diff_content()`` restricts every diff-text token-correlation
haystack (DIFF_PRESENCE, DIFF_CORRELATION, and the Phase 4.22 "from A to B"
value-colocation check) to genuinely ADDED (``+``) line content only.

VULN-4.24B-02 -- When neither an obligation nor its parent requirement
names a target path (a real, anticipated case for a minimal/offline
planner -- see derive_task_contract's own docstring), behavioral-evidence
path scoping becomes a no-op, leaving component-token matching as the only
signal. Reproduced: a stray passing test for an unrelated, pre-existing
"csv config parsing" helper satisfied "Add CSV export" at the strongest
(EXECUTABLE_BEHAVIORAL) tier purely by sharing the "csv" identifier
component -- with no sibling obligation to trigger the Phase 4.23
cross-obligation ambiguity exclusion (there is nothing to be ambiguous
WITH). Fix: when no path anchor exists, a PASSING behavioral match must
also have its symbol name appear in this task's own tracked diff before it
can grant EXECUTABLE_BEHAVIORAL -- proof it is something this task's
changeset actually concerns, not merely something that happens to exist
and pass elsewhere in the repository. FAILING matches are explicitly never
subject to this extra check: failure dominance must not depend on whether
a failure can be independently corroborated.

VULN-4.24B-03 -- CONSTRAINT_ABSENCE ("do not modify X") path matching was
case-sensitive even though X is free text typed by the user, compared
against real (possibly differently-cased) file paths -- on a
case-insensitive filesystem (the Windows/macOS default) this could
silently miss a genuine constraint violation. Fix: case-folded comparison,
original wording preserved in the reported violation message.
"""

from __future__ import annotations

import datetime
import inspect
import time

import pytest

from local_agent.completion import (
    CompletionEvidenceStore,
    EvidenceTrustTier,
    EvidenceType,
)
from local_agent.filesystem import ProjectFilesystem
from local_agent.models import FileOperation, ReviewResult, Task, TaskStatus
from local_agent.task_contract import (
    AcceptanceMethod,
    AcceptanceObligation,
    Requirement,
    RequirementAssessmentEngine,
    RequirementState,
    RequirementType,
    TaskContract,
    VerificationStrategy,
    _added_diff_content,
    derive_task_contract,
)


def make_task(task_id: str = "t1", objective: str = "Implement feature") -> Task:
    now = datetime.datetime.now(datetime.timezone.utc)
    return Task(task_id=task_id, objective=objective, status=TaskStatus.PENDING, created_at=now, updated_at=now)


def record_behavioral(
    store: CompletionEvidenceStore,
    task_id: str,
    symbols: list[str],
    exit_code: int,
    trivial: bool,
    target_paths: list[str] | None = None,
) -> None:
    store.record(
        task_id=task_id, subtask_id="main", turn_number=1, stage="verifying",
        evidence_type=EvidenceType.TEST_EXECUTION, source="behavioral_verification_synthesizer",
        trust_tier=EvidenceTrustTier.AUTHORITATIVE_EXECUTION,
        target_paths=target_paths or [], target_symbols=symbols,
        command=["pytest"], exit_code=exit_code,
        payload={"synthesized": True, "trivial": trivial},
    )


def two_obligation_contract(objective: str = "Add CSV and JSON export"):
    task = make_task(objective=objective)
    contract = derive_task_contract(task, None)
    req = next(r for r in contract.requirements if r.requirement_type == RequirementType.FUNCTIONAL.value)
    assert len(req.acceptance_obligations) >= 2
    return task, contract, req


# =============================================================================
# Section 1: reproduced vulnerabilities (regression-locked)
# =============================================================================

class TestReproducedVulnerabilities:
    def test_vuln01_removed_line_mention_does_not_satisfy_obligation(self, tmp_path):
        """A diff that only DELETES a stray mention of the obligation's
        token, and implements something unrelated, must not satisfy it."""
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        _, contract, req = two_obligation_contract()
        diff_text = (
            "--- a/exporter.py\n+++ b/exporter.py\n@@ -1,2 +1,2 @@\n"
            "-# TODO: add json export\n"
            "+def export_csv(data):\n"
            "+    return ','.join(data)\n"
        )
        engine = RequirementAssessmentEngine(fs)
        result = engine.assess(contract, store, [FileOperation(action="modify", path="exporter.py")], diff_text, last_review=None)
        req_result = next(r for r in result.requirements if r.requirement_id == req.requirement_id)
        json_ob = next(o for o in req_result.acceptance_obligations if "JSON" in o.description)
        csv_ob = next(o for o in req_result.acceptance_obligations if "CSV" in o.description)
        assert json_ob.state != RequirementState.SATISFIED.value
        assert csv_ob.state == RequirementState.SATISFIED.value  # genuinely implemented, unaffected

    def test_vuln01_removed_line_diff_presence_regression(self, tmp_path):
        """Store-level regression lock for the DIFF_PRESENCE (non-obligation)
        path, using a compound objective to force strict (non-lenient)
        correlation rather than the always-True single-vague-requirement
        shortcut."""
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        task = make_task(objective="Change the retry limit from 3 to 9 attempts")
        contract = derive_task_contract(task, None)
        engine = RequirementAssessmentEngine(fs)
        # "9" only ever appears on a REMOVED line; the actual added code sets
        # something unrelated ("max_retries = 3" stays untouched).
        diff_text = "-# old cap was 9, revisit later\n+max_retries = 3\n"
        ops = [FileOperation(action="modify", path="config.py")]
        result = engine.assess(contract, store, ops, diff_text, last_review=None)
        req = next(r for r in result.requirements if r.requirement_type == RequirementType.FUNCTIONAL.value)
        assert req.state != RequirementState.SATISFIED.value

    def test_vuln02_unrelated_pre_existing_symbol_no_longer_grants_behavioral_proof(self, tmp_path):
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        ob = AcceptanceObligation(obligation_id="OB1", description="CSV export", target_tokens=["csv"])
        req = Requirement(requirement_id="REQ-001", statement="Add CSV export", acceptance_obligations=[ob])
        contract = TaskContract(task_id="t1", objective="Add CSV export", requirements=[req])
        record_behavioral(store, "t1", symbols=["parse_csv_config_line"], exit_code=0, trivial=False,
                          target_paths=["config/csv_config_reader.py"])
        engine = RequirementAssessmentEngine(fs)
        result = engine.assess(contract, store, [FileOperation(action="modify", path="settings_ui.py")], "+// still just a stub\n", last_review=None)
        ob_result = result.requirements[0].acceptance_obligations[0]
        assert ob_result.method != AcceptanceMethod.EXECUTABLE_BEHAVIORAL.value
        assert ob_result.state != RequirementState.SATISFIED.value

    def test_vuln02_legitimate_no_path_match_still_works_when_diff_mentions_symbol(self, tmp_path):
        """Sanity check: the fix must not make the ordinary, legitimate
        no-plan case (Phase 4.23's own dominant test shape) impossible --
        when the diff genuinely mentions the symbol, behavioral proof is
        still granted."""
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        ob = AcceptanceObligation(obligation_id="OB1", description="CSV export", target_tokens=["csv"])
        req = Requirement(requirement_id="REQ-001", statement="Add CSV export", acceptance_obligations=[ob])
        contract = TaskContract(task_id="t1", objective="Add CSV export", requirements=[req])
        record_behavioral(store, "t1", symbols=["export_csv"], exit_code=0, trivial=False)
        engine = RequirementAssessmentEngine(fs)
        result = engine.assess(contract, store, [FileOperation(action="modify", path="exporter.py")], "+def export_csv(): ...", last_review=None)
        ob_result = result.requirements[0].acceptance_obligations[0]
        assert ob_result.method == AcceptanceMethod.EXECUTABLE_BEHAVIORAL.value
        assert ob_result.state == RequirementState.SATISFIED.value

    def test_vuln02_failure_dominance_unaffected_by_missing_diff_anchor(self, tmp_path):
        """Critical regression guard: a genuine FAILING behavioral match for
        an obligation with no path anchor must still dominate, even though
        its symbol is not literally present in the (unrelated) diff text --
        the diff-anchor requirement applies ONLY to promoting a PASSING
        match to EXECUTABLE_BEHAVIORAL, never to suppressing a failure."""
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        ob = AcceptanceObligation(obligation_id="OB1", description="authenticate", target_tokens=["authenticate"])
        req = Requirement(requirement_id="REQ-001", statement="Add authentication", acceptance_obligations=[ob])
        contract = TaskContract(task_id="t1", objective="Add authentication", requirements=[req])
        record_behavioral(store, "t1", symbols=["authenticate"], exit_code=1, trivial=False)
        engine = RequirementAssessmentEngine(fs)
        review = ReviewResult(verdict="APPROVED", summary="looks great, ships auth")
        result = engine.assess(
            contract, store,
            [FileOperation(action="modify", path="auth.py")],
            "+def authenticate(): return True  # always true, never checks",
            last_review=review,
        )
        ob_result = result.requirements[0].acceptance_obligations[0]
        assert ob_result.state == RequirementState.FAILED.value
        assert ob_result.method == AcceptanceMethod.EXECUTABLE_BEHAVIORAL.value

    def test_vuln03_case_insensitive_constraint_violation_detected(self, tmp_path):
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        task = make_task(objective="Do not modify Tool_Engine.py")
        contract = derive_task_contract(task, None)
        engine = RequirementAssessmentEngine(fs)
        result = engine.assess(contract, store, [FileOperation(action="modify", path="tool_engine.py")], "+bad", last_review=None)
        constraint = next(r for r in result.requirements if r.requirement_type == RequirementType.CONSTRAINT.value)
        assert constraint.state == RequirementState.FAILED.value
        assert result.satisfied is False

    def test_vuln03_case_insensitive_constraint_still_passes_when_untouched(self, tmp_path):
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        task = make_task(objective="Do not modify Tool_Engine.py")
        contract = derive_task_contract(task, None)
        engine = RequirementAssessmentEngine(fs)
        result = engine.assess(contract, store, [FileOperation(action="modify", path="unrelated.py")], "+ok", last_review=None)
        constraint = next(r for r in result.requirements if r.requirement_type == RequirementType.CONSTRAINT.value)
        assert constraint.state == RequirementState.SATISFIED.value


# =============================================================================
# Section 2: _added_diff_content unit properties
# =============================================================================

class TestAddedDiffContent:
    def test_excludes_removed_lines(self):
        assert "old" not in _added_diff_content("-old line\n+new line\n")

    def test_includes_added_lines(self):
        assert "new line" in _added_diff_content("-old line\n+new line\n")

    def test_excludes_file_header(self):
        result = _added_diff_content("+++ b/secret_marker.py\n+real content\n")
        assert "secret_marker" not in result
        assert "real content" in result

    def test_excludes_context_lines(self):
        result = _added_diff_content(" def unchanged():\n+    return 1\n")
        assert "unchanged" not in result

    def test_empty_diff_yields_empty(self):
        assert _added_diff_content("") == ""


# =============================================================================
# Section 3: Attack Family A -- diff-correlation false-positive matrix
# =============================================================================

class TestDiffCorrelationMatrix:
    """Each case: an obligation token appears SOMEWHERE in the diff/paths,
    but never in a way that represents genuinely new code addressing it."""

    def _single_obligation_assess(self, tmp_path, token, description, diff_text, changed_path="impl.py"):
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        ob = AcceptanceObligation(obligation_id="OB1", description=description, target_tokens=[token])
        req = Requirement(requirement_id="REQ-001", statement=description, acceptance_obligations=[ob])
        contract = TaskContract(task_id="t1", objective=description, requirements=[req])
        engine = RequirementAssessmentEngine(fs)
        result = engine.assess(contract, store, [FileOperation(action="modify", path=changed_path)], diff_text, last_review=None)
        return result.requirements[0].acceptance_obligations[0]

    def test_a01_removed_line_only(self, tmp_path):
        ob = self._single_obligation_assess(tmp_path, "webhook", "Add webhook support",
                                            "-def register_webhook(): pass\n+def unrelated(): pass\n")
        assert ob.state != RequirementState.SATISFIED.value

    def test_a02_context_line_only(self, tmp_path):
        ob = self._single_obligation_assess(tmp_path, "webhook", "Add webhook support",
                                            " def register_webhook():  # unchanged\n+    pass  # nothing added\n")
        assert ob.state != RequirementState.SATISFIED.value

    def test_a03_file_header_only(self, tmp_path):
        ob = self._single_obligation_assess(tmp_path, "webhook", "Add webhook support",
                                            "--- a/webhook_unrelated.py\n+++ b/webhook_unrelated.py\n+x = 1\n")
        assert ob.state != RequirementState.SATISFIED.value

    def test_a04_legitimate_addition_still_satisfied(self, tmp_path):
        """Positive control: the matrix must not become so strict that a
        real implementation stops being recognized."""
        ob = self._single_obligation_assess(tmp_path, "webhook", "Add webhook support",
                                            "+def register_webhook(url):\n+    subscribers.append(url)\n")
        assert ob.state == RequirementState.SATISFIED.value

    def test_a05_removed_and_added_but_added_is_unrelated(self, tmp_path):
        ob = self._single_obligation_assess(tmp_path, "ratelimit", "Add rate limiting",
                                            "-# ratelimit: not implemented\n+def log_request(): pass\n")
        assert ob.state != RequirementState.SATISFIED.value

    def test_a06_value_colocation_ignores_removed_line(self, tmp_path):
        """The Phase 4.22 'from A to B' value-colocation defense must also
        respect the added-lines-only restriction."""
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        task = make_task(objective="Change the retry limit from 3 to 9 attempts")
        contract = derive_task_contract(task, None)
        engine = RequirementAssessmentEngine(fs)
        # "9" and "retry"/"limit" co-occur only on a REMOVED line.
        diff_text = "-retry_limit = 9  # was already correct, being reverted\n+retry_limit = 3\n"
        result = engine.assess(contract, store, [FileOperation(action="modify", path="config.py")], diff_text, last_review=None)
        req = next(r for r in result.requirements if r.requirement_type == RequirementType.FUNCTIONAL.value)
        assert req.state != RequirementState.SATISFIED.value

    def test_a07_value_colocation_added_line_still_satisfied(self, tmp_path):
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        task = make_task(objective="Change the retry limit from 3 to 9 attempts")
        contract = derive_task_contract(task, None)
        engine = RequirementAssessmentEngine(fs)
        # Must repeat every one of the statement's own content words
        # ("retry", "limit", "attempts") on the added line -- DIFF_PRESENCE
        # correlation (not just value-colocation) requires all of them when
        # there are <= 4 distinct content tokens.
        diff_text = "-retry_limit = 3  # old attempts cap\n+retry_limit = 9  # new attempts cap\n"
        result = engine.assess(contract, store, [FileOperation(action="modify", path="config.py")], diff_text, last_review=None)
        req = next(r for r in result.requirements if r.requirement_type == RequirementType.FUNCTIONAL.value)
        assert req.state == RequirementState.SATISFIED.value


# =============================================================================
# Section 4: Attack Family B -- planner/path absence matrix
# =============================================================================

class TestPathAbsenceMatrix:
    def test_b01_obligation_path_empty_requirement_path_populated_uses_requirement_scope(self, tmp_path):
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        ob = AcceptanceObligation(obligation_id="OB1", description="CSV export", target_tokens=["csv"])
        req = Requirement(requirement_id="REQ-001", statement="Add CSV export", acceptance_obligations=[ob], target_paths=["exporter.py"])
        contract = TaskContract(task_id="t1", objective="Add CSV export", requirements=[req])
        # Evidence in an unrelated file -- requirement-level path scoping
        # (still populated) correctly excludes it regardless of the new
        # diff-anchor rule.
        record_behavioral(store, "t1", symbols=["parse_csv_config_line"], exit_code=0, trivial=False,
                          target_paths=["config/csv_config_reader.py"])
        engine = RequirementAssessmentEngine(fs)
        result = engine.assess(contract, store, [FileOperation(action="modify", path="exporter.py")], "+// stub\n", last_review=None)
        assert result.requirements[0].acceptance_obligations[0].state != RequirementState.SATISFIED.value

    def test_b02_obligation_path_populated_requirement_path_empty_uses_obligation_scope(self, tmp_path):
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        ob = AcceptanceObligation(obligation_id="OB1", description="CSV export", target_tokens=["csv"], target_paths=["exporter.py"])
        req = Requirement(requirement_id="REQ-001", statement="Add CSV export", acceptance_obligations=[ob])
        contract = TaskContract(task_id="t1", objective="Add CSV export", requirements=[req])
        store.record(task_id="t1", subtask_id="main", turn_number=1, stage="verifying",
                     evidence_type=EvidenceType.TEST_EXECUTION, source="behavioral_verification_synthesizer",
                     trust_tier=EvidenceTrustTier.AUTHORITATIVE_EXECUTION,
                     target_paths=["exporter.py"], target_symbols=["export_csv"],
                     command=["pytest"], exit_code=0, payload={"synthesized": True, "trivial": False})
        engine = RequirementAssessmentEngine(fs)
        result = engine.assess(contract, store, [FileOperation(action="modify", path="exporter.py")], "+def export_csv(): ...", last_review=None)
        assert result.requirements[0].acceptance_obligations[0].state == RequirementState.SATISFIED.value

    def test_b03_sibling_obligations_both_path_empty_still_cross_excluded(self, tmp_path):
        """Interaction check: Phase 4.23's cross-obligation ambiguity
        exclusion and the Phase 4.24 diff-anchor rule are independent and
        compose correctly. "convert_csv_to_jsonlike" contains "csv" as a
        real, sole-matching identifier component (legitimate, not
        ambiguous -- json is not a component of "jsonlike"), so CSV
        correctly still gets EXECUTABLE_BEHAVIORAL; JSON must not (this is
        the exact Phase 4.23 regression lock, re-verified after this
        phase's changes -- see test_phase423_evidence_provenance.py's
        identical scenario)."""
        _, contract, req = two_obligation_contract()
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        record_behavioral(store, "t1", symbols=["convert_csv_to_jsonlike"], exit_code=0, trivial=False, target_paths=[])
        engine = RequirementAssessmentEngine(fs)
        result = engine.assess(contract, store, [], "+def convert_csv_to_jsonlike(): ...", last_review=None)
        req_result = next(r for r in result.requirements if r.requirement_id == req.requirement_id)
        by_desc = {o.description: o for o in req_result.acceptance_obligations}
        assert by_desc["JSON export"].method != AcceptanceMethod.EXECUTABLE_BEHAVIORAL.value

    def test_b04_same_symbol_different_modules_no_path_anchor_requires_diff_mention(self, tmp_path):
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        ob = AcceptanceObligation(obligation_id="OB1", description="cache layer", target_tokens=["cache"])
        req = Requirement(requirement_id="REQ-001", statement="Add cache layer", acceptance_obligations=[ob])
        contract = TaskContract(task_id="t1", objective="Add cache layer", requirements=[req])
        # "cache" module exists for an unrelated purpose (HTTP response
        # caching) elsewhere in the same task.
        record_behavioral(store, "t1", symbols=["http_response_cache_get"], exit_code=0, trivial=False, target_paths=["http/cache.py"])
        engine = RequirementAssessmentEngine(fs)
        result = engine.assess(contract, store, [FileOperation(action="modify", path="db/layer.py")], "+// TODO db cache layer\n", last_review=None)
        assert result.requirements[0].acceptance_obligations[0].method != AcceptanceMethod.EXECUTABLE_BEHAVIORAL.value


# =============================================================================
# Section 5: integration (real, unmocked completion path)
# =============================================================================

class TestIntegration:
    def test_end_to_end_orchestrator_shape_diff_correlation_and_path_absence_both_hold(self, tmp_path):
        """One real, unmocked pass through derive_task_contract ->
        RequirementAssessmentEngine.assess() exactly as Orchestrator.run()
        and MultiTurnImplementationAgent.execute() call it, combining both
        fixed vulnerabilities in one scenario: a compound objective with no
        Plan (path-absent), where CSV is genuinely implemented and JSON is
        faked via a removed-line mention plus an unrelated passing test."""
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        task = make_task(objective="Add CSV and JSON export")
        contract = derive_task_contract(task, None)  # plan=None: no target_paths anywhere
        record_behavioral(store, task.task_id, symbols=["parse_json_config"], exit_code=0, trivial=False, target_paths=[])
        diff_text = (
            "-# json export: not implemented yet\n"
            "+def export_csv(data):\n"
            "+    return ','.join(data)\n"
        )
        engine = RequirementAssessmentEngine(fs)
        review = ReviewResult(verdict="APPROVED", summary="looks complete")
        result = engine.assess(
            contract, store, [FileOperation(action="modify", path="exporter.py")], diff_text,
            last_review=review,
        )
        req = next(r for r in result.requirements if r.requirement_type == RequirementType.FUNCTIONAL.value)
        by_desc = {o.description: o for o in req.acceptance_obligations}
        csv_ob = next(o for d, o in by_desc.items() if "CSV" in d)
        json_ob = next(o for d, o in by_desc.items() if "JSON" in d)
        assert csv_ob.state == RequirementState.SATISFIED.value
        assert json_ob.state != RequirementState.SATISFIED.value
        assert result.satisfied is False  # partial compound completion correctly blocked


# =============================================================================
# Section 6: invariants (P24-1 .. P24-6)
# =============================================================================

class TestInvariants:
    def test_p24_1_removed_lines_never_grant_diff_correlation(self, tmp_path):
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        ob = AcceptanceObligation(obligation_id="OB1", description="x feature", target_tokens=["xfeature"])
        req = Requirement(requirement_id="REQ-001", statement="Add x feature", acceptance_obligations=[ob])
        contract = TaskContract(task_id="t1", objective="Add x feature", requirements=[req])
        engine = RequirementAssessmentEngine(fs)
        result = engine.assess(contract, store, [FileOperation(action="modify", path="a.py")], "-xfeature removed\n", last_review=None)
        assert result.requirements[0].acceptance_obligations[0].state != RequirementState.SATISFIED.value

    def test_p24_2_failing_behavioral_evidence_always_dominates_regardless_of_path_or_diff_anchor(self, tmp_path):
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        ob = AcceptanceObligation(obligation_id="OB1", description="y feature", target_tokens=["yfeature"])
        req = Requirement(requirement_id="REQ-001", statement="Add y feature", acceptance_obligations=[ob])
        contract = TaskContract(task_id="t1", objective="Add y feature", requirements=[req])
        record_behavioral(store, "t1", symbols=["yfeature_impl"], exit_code=1, trivial=False, target_paths=[])
        engine = RequirementAssessmentEngine(fs)
        result = engine.assess(contract, store, [FileOperation(action="modify", path="unrelated.py")], "+// nothing about yfeature\n", last_review=None)
        assert result.requirements[0].acceptance_obligations[0].state == RequirementState.FAILED.value

    def test_p24_3_no_path_anchor_passing_match_requires_diff_mention(self, tmp_path):
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        ob = AcceptanceObligation(obligation_id="OB1", description="z feature", target_tokens=["zfeature"])
        req = Requirement(requirement_id="REQ-001", statement="Add z feature", acceptance_obligations=[ob])
        contract = TaskContract(task_id="t1", objective="Add z feature", requirements=[req])
        record_behavioral(store, "t1", symbols=["zfeature_helper"], exit_code=0, trivial=False, target_paths=[])
        engine = RequirementAssessmentEngine(fs)
        result = engine.assess(contract, store, [FileOperation(action="modify", path="unrelated.py")], "+// nothing here\n", last_review=None)
        assert result.requirements[0].acceptance_obligations[0].method != AcceptanceMethod.EXECUTABLE_BEHAVIORAL.value

    def test_p24_4_path_anchored_match_unaffected_by_diff_anchor_rule(self, tmp_path):
        """When a real path anchor exists, the (unrelated) new diff-anchor
        rule must not apply at all -- only the no-path case is restricted."""
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        ob = AcceptanceObligation(obligation_id="OB1", description="w feature", target_tokens=["wfeature"], target_paths=["w.py"])
        req = Requirement(requirement_id="REQ-001", statement="Add w feature", acceptance_obligations=[ob], target_paths=["w.py"])
        contract = TaskContract(task_id="t1", objective="Add w feature", requirements=[req])
        record_behavioral(store, "t1", symbols=["wfeature_impl"], exit_code=0, trivial=False, target_paths=["w.py"])
        engine = RequirementAssessmentEngine(fs)
        # Diff text deliberately does NOT mention "wfeature_impl" -- the
        # real path anchor is sufficient corroboration on its own.
        result = engine.assess(contract, store, [FileOperation(action="modify", path="w.py")], "+# implemented, see tests\n", last_review=None)
        assert result.requirements[0].acceptance_obligations[0].method == AcceptanceMethod.EXECUTABLE_BEHAVIORAL.value

    def test_p24_5_constraint_absence_case_insensitive_both_directions(self, tmp_path):
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        task = make_task(objective="Do not modify config.py")
        contract = derive_task_contract(task, None)
        engine = RequirementAssessmentEngine(fs)
        result = engine.assess(contract, store, [FileOperation(action="modify", path="Config.py")], "+x", last_review=None)
        constraint = next(r for r in result.requirements if r.requirement_type == RequirementType.CONSTRAINT.value)
        assert constraint.state == RequirementState.FAILED.value

    def test_p24_6_persisted_forged_obligation_state_still_recomputed_fresh(self, tmp_path):
        """Phase 4.20-4.23 guarantee, re-verified after this phase's
        changes: a forged persisted SATISFIED state is never trusted."""
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        _, contract, req = two_obligation_contract()
        forged_obligations = [
            o.__class__(**{**o.to_dict(), "state": RequirementState.SATISFIED.value, "method": AcceptanceMethod.EXECUTABLE_BEHAVIORAL.value})
            for o in req.acceptance_obligations
        ]
        forged_req = req.__class__(**{**req.to_dict(), "acceptance_obligations": forged_obligations, "state": RequirementState.SATISFIED.value})
        contract.requirements = [forged_req if r.requirement_id == req.requirement_id else r for r in contract.requirements]
        engine = RequirementAssessmentEngine(fs)
        result = engine.assess(contract, store, [], "", last_review=None)
        assert result.satisfied is False


# =============================================================================
# Section 7: false-positive campaign self-audit
# =============================================================================

class TestFalsePositiveSelfAudit:
    """Attacks the new tests/logic themselves for weak assertions or
    accidental self-repair, per the forensic self-audit questions."""

    def test_multiturn_orchestrator_still_share_identical_trust_model(self):
        """Q15-style guard: confirm no existing Phase 4.23 invariant was
        weakened by this phase's changes."""
        from local_agent.multi_turn import MultiTurnImplementationAgent
        from local_agent.orchestrator import Orchestrator
        orch_src = inspect.getsource(Orchestrator.run)
        mt_src = inspect.getsource(MultiTurnImplementationAgent.execute)
        assert "req_assessment.satisfied" in orch_src
        assert "req_assessment.satisfied" in mt_src

    def test_diff_anchor_check_uses_lowercased_comparison_not_accidentally_case_sensitive(self, tmp_path):
        """The diff-anchor corroboration itself must not silently fail on a
        case mismatch between the recorded symbol and the diff text (which
        would make the fix over-strict, a false-positive-on-the-fix-itself
        in the other direction)."""
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        ob = AcceptanceObligation(obligation_id="OB1", description="csv export", target_tokens=["csv"])
        req = Requirement(requirement_id="REQ-001", statement="Add csv export", acceptance_obligations=[ob])
        contract = TaskContract(task_id="t1", objective="Add csv export", requirements=[req])
        record_behavioral(store, "t1", symbols=["Export_CSV"], exit_code=0, trivial=False, target_paths=[])
        engine = RequirementAssessmentEngine(fs)
        result = engine.assess(contract, store, [FileOperation(action="modify", path="exporter.py")], "+def Export_CSV(): ...", last_review=None)
        assert result.requirements[0].acceptance_obligations[0].method == AcceptanceMethod.EXECUTABLE_BEHAVIORAL.value


# =============================================================================
# Section 8: performance (measured)
# =============================================================================

class TestPerformance:
    def test_added_diff_content_scales_linearly(self):
        big_diff = "".join(f"+line {i}\n-old {i}\n context {i}\n" for i in range(20000))
        start = time.perf_counter()
        result = _added_diff_content(big_diff)
        elapsed = time.perf_counter() - start
        assert "line 19999" in result
        assert "old 19999" not in result
        assert elapsed < 2.0, f"_added_diff_content over 60k lines took {elapsed:.3f}s"

    def test_assess_with_many_obligations_and_no_path_anchor_stays_bounded(self, tmp_path):
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        obs = [AcceptanceObligation(obligation_id=f"OB{i}", description=f"feature {i}", target_tokens=[f"feat{i}"]) for i in range(8)]
        req = Requirement(requirement_id="REQ-001", statement="Add many features", acceptance_obligations=obs)
        contract = TaskContract(task_id="t1", objective="Add many features", requirements=[req])
        for i in range(8):
            record_behavioral(store, "t1", symbols=[f"feat{i}_impl"], exit_code=0, trivial=False, target_paths=[])
        diff_text = "".join(f"+def feat{i}_impl(): ...\n" for i in range(8))
        engine = RequirementAssessmentEngine(fs)
        start = time.perf_counter()
        result = engine.assess(contract, store, [FileOperation(action="modify", path="a.py")], diff_text, last_review=None)
        elapsed = time.perf_counter() - start
        assert result.satisfied is True
        assert elapsed < 2.0


# =============================================================================
# Section 9: backward compatibility
# =============================================================================

class TestBackwardCompatibility:
    def test_old_style_obligation_dict_without_new_behavior_still_loads(self):
        from local_agent.task_contract import AcceptanceObligation as AO
        old_dict = {
            "obligation_id": "OB1", "description": "x", "method": "diff_correlation",
            "provenance": "system_derived", "state": "unverified", "target_tokens": ["x"],
            "target_paths": [], "target_symbols": [], "evidence_ids": [], "unsatisfied_reason": "",
        }
        ob = AO.from_dict(old_dict)
        assert ob.obligation_id == "OB1"

    def test_assess_signature_unchanged(self, tmp_path):
        fs = ProjectFilesystem(tmp_path)
        store = CompletionEvidenceStore(tmp_path)
        task = make_task(objective="Implement a new feature")
        contract = derive_task_contract(task, None)
        engine = RequirementAssessmentEngine(fs)
        # Positional-call shape exactly as Orchestrator.run() and
        # MultiTurnImplementationAgent.execute() invoke it.
        result = engine.assess(contract, store, [FileOperation(action="modify", path="a.py")], "+x=1", last_review=None)
        assert result.satisfied is True
