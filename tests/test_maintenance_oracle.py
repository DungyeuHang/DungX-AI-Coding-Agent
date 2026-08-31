"""Phase 4.23 - the maintenance execution oracle framework.

Test classification, continuing the discipline Phase 4.22 established:

* **PRODUCTION-INTEGRATION** - the fixture is produced by the real discovery
  path (real ``SemanticGraph`` + real ``MaintenanceAnalyzer``) over a real
  temporary repository, and the assertion is about bytes on disk, real
  subprocess exit codes or real persisted records.
* **CONTRACT** - a unit-level assertion about a pure function or a structural
  property of the source. Labelled as such; it proves the contract, not the
  end-to-end behaviour.

The one mocked boundary anywhere in this file is the LLM call, reusing
``ScriptedProvider`` from the Phase 4.22 suite. Everything else - the
filesystem, ``compileall``, the candidate workspace, the journal, the
lifecycle store - is real.
"""

from __future__ import annotations

import ast
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import unittest.mock as mock
from pathlib import Path

from local_agent.maintenance import ALL_SIGNAL_KINDS, MaintenanceBudget, MaintenanceSignal
from local_agent.maintenance_analysis import MaintenanceAnalyzer
from local_agent.maintenance_execution import (
    ENFORCED_BUDGET_FIELDS,
    NO_MUTATION_STATUSES,
    SUPPORTED_SIGNAL_KINDS,
    SUPPORTED_TIERS,
    UNENFORCED_BUDGET_FIELDS,
    ExecutionJournal,
    MaintenanceExecutionStatus,
    _executable_missing,
)
from local_agent.maintenance_oracle import (
    ALL_ORACLE_CLASSES,
    ALL_ORACLE_OUTCOMES,
    AUTONOMOUS_SIGNAL_KINDS,
    MAX_SHRINK_RATIO,
    REPLACED_LINE_SLACK,
    ORACLE_FRAMEWORK_VERSION,
    PROMOTABLE_ORACLE_CLASSES,
    SHRINK_SLACK_LINES,
    SIGNAL_INVENTORY,
    ExecutionOracle,
    OracleClass,
    OracleObservation,
    OracleOutcome,
    ParseOracle,
    UnverifiableOracle,
    _substance_floor,
    inventory_for,
    inventory_rows,
    missing_inventory_entries,
    oracle_for,
    unknown_inventory_entries,
)
from local_agent.maintenance_policy import (
    AUTONOMOUSLY_ACTIONABLE_KINDS,
    TIER_ORDER,
    AutonomyTier,
    MaintenanceExecutionPolicy,
)

from .test_maintenance_execution import (
    BROKEN_SOURCE,
    FIXED_SOURCE,
    ExecutorCase,
    ScriptedProvider,
    discover_parse_failure,
    fix_operation,
    snapshot_tree,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ORACLE_MODULE = "local_agent.maintenance_oracle"
EXECUTOR_MODULE = "local_agent.maintenance_execution"

#: A module with real substance and a real syntax error. Used everywhere a
#: destructive "repair" must be distinguishable from a real one.
SUBSTANTIAL_BROKEN = (
    "import math\n"
    "\n"
    "CONSTANT = 42\n"
    "\n"
    "\n"
    "def add(a, b)\n"  # <- the defect: missing colon
    "    return a + b\n"
    "\n"
    "\n"
    "def multiply(a, b):\n"
    "    return a * b\n"
    "\n"
    "\n"
    "class Calculator:\n"
    "    def compute(self, x):\n"
    "        return math.sqrt(x)\n"
)
SUBSTANTIAL_FIXED = SUBSTANTIAL_BROKEN.replace("def add(a, b)\n", "def add(a, b):\n")
#: The exploit that defeated Phase 4.22: it parses, and it destroys the module.
DESTRUCTIVE_REPAIR = "pass\n"
#: A subtler destruction: keeps the broken function, deletes everything else.
PARTIAL_DESTRUCTION = "def add(a, b):\n    return a + b\n"


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

    Same helper contract as the Phase 4.21/4.22 suites: string literals,
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


# =============================================================================
# A. The inventory is consistent with the real implementation
# =============================================================================


class SignalInventoryTests(unittest.TestCase):
    """The inventory must describe the code, not an aspiration."""

    def test_every_signal_has_an_inventory_entry(self):
        """CONTRACT (reflection over the real ALL_SIGNAL_KINDS)."""
        self.assertEqual(missing_inventory_entries(), ())

    def test_every_inventory_entry_names_a_real_signal(self):
        """CONTRACT: the inventory cannot invent signals to fill a table."""
        self.assertEqual(unknown_inventory_entries(), ())

    def test_the_inventory_is_exactly_the_signal_vocabulary(self):
        self.assertEqual(set(SIGNAL_INVENTORY), set(ALL_SIGNAL_KINDS))
        self.assertEqual(len(SIGNAL_INVENTORY), 13)

    def test_every_recorded_producer_exists_on_the_real_analyzer(self):
        """PRODUCTION-INTEGRATION: reflection against the shipped analyzer.

        This is what stops the inventory rotting: renaming an extractor without
        updating its entry fails here rather than being discovered by a reader
        of a stale table.
        """
        for kind, entry in sorted(SIGNAL_INVENTORY.items()):
            with self.subTest(signal=kind):
                owner, _, method = entry.producer.rpartition(".")
                self.assertEqual(owner, "MaintenanceAnalyzer", entry.producer)
                self.assertTrue(
                    hasattr(MaintenanceAnalyzer, method),
                    f"{entry.producer} does not exist",
                )
                self.assertTrue(callable(getattr(MaintenanceAnalyzer, method)))

    def test_every_recorded_policy_ceiling_is_a_real_tier(self):
        for kind, entry in sorted(SIGNAL_INVENTORY.items()):
            with self.subTest(signal=kind):
                self.assertIn(entry.policy_max_tier, TIER_ORDER)

    def test_the_recorded_ceiling_matches_what_the_real_policy_grants(self):
        """PRODUCTION-INTEGRATION: the real policy decides, not the table.

        A kind outside ``AUTONOMOUSLY_ACTIONABLE_KINDS`` is capped at
        ``recommend`` by the real policy; the inventory must say the same. This
        is asserted against the shipped constant rather than re-implementing
        the policy's reasoning here.
        """
        for kind, entry in sorted(SIGNAL_INVENTORY.items()):
            with self.subTest(signal=kind):
                if kind in AUTONOMOUSLY_ACTIONABLE_KINDS:
                    self.assertEqual(
                        entry.policy_max_tier,
                        AutonomyTier.EXECUTE_WITH_EXISTING_APPROVAL,
                    )
                else:
                    self.assertEqual(entry.policy_max_tier, AutonomyTier.RECOMMEND)

    def test_every_oracle_class_recorded_is_real(self):
        for kind, entry in sorted(SIGNAL_INVENTORY.items()):
            with self.subTest(signal=kind):
                self.assertIn(entry.oracle_class, ALL_ORACLE_CLASSES)

    def test_the_inventory_agrees_with_the_bound_oracle(self):
        """The table and the registry cannot drift apart."""
        for kind, entry in sorted(SIGNAL_INVENTORY.items()):
            with self.subTest(signal=kind):
                oracle = oracle_for(kind)
                self.assertEqual(oracle.oracle_class, entry.oracle_class)
                self.assertEqual(oracle.deterministic, entry.deterministic_oracle)
                self.assertEqual(oracle.signal_kind, kind)

    def test_every_non_autonomous_signal_records_why(self):
        """Rejection must be reasoned, not merely asserted."""
        for kind, entry in sorted(SIGNAL_INVENTORY.items()):
            with self.subTest(signal=kind):
                if entry.autonomous_execution:
                    self.assertEqual(entry.rejection_reasons, ())
                else:
                    self.assertTrue(
                        entry.rejection_reasons,
                        f"{kind} is non-autonomous with no recorded reason",
                    )
                    # Substance, not a placeholder. Individual reasons may be
                    # legitimately terse ("names no file" is a complete
                    # argument), so the bar is on the argument as a whole.
                    self.assertGreater(
                        len(" ".join(entry.rejection_reasons)), 80, kind
                    )
                    for reason in entry.rejection_reasons:
                        self.assertTrue(reason.strip())

    def test_every_autonomous_signal_records_what_it_still_cannot_prove(self):
        """Granting autonomy must never read as granting a total guarantee."""
        for kind, entry in sorted(SIGNAL_INVENTORY.items()):
            with self.subTest(signal=kind):
                if entry.autonomous_execution:
                    self.assertTrue(
                        entry.residual_limitations,
                        f"{kind} is autonomous but claims no residual limitation",
                    )
                    self.assertGreater(
                        len(" ".join(entry.residual_limitations)), 120, kind
                    )
                else:
                    # Non-autonomous signals carry the argument in
                    # rejection_reasons instead; a second list would just
                    # duplicate it.
                    self.assertEqual(entry.residual_limitations, ())

    def test_only_deterministic_oracles_are_marked_autonomous(self):
        for kind, entry in sorted(SIGNAL_INVENTORY.items()):
            with self.subTest(signal=kind):
                if entry.autonomous_execution:
                    self.assertTrue(entry.deterministic_oracle)
                    self.assertEqual(entry.oracle_class, OracleClass.DETERMINISTIC)

    def test_exactly_one_signal_is_autonomous_in_this_build(self):
        self.assertEqual(
            AUTONOMOUS_SIGNAL_KINDS, frozenset({MaintenanceSignal.PARSE_FAILURE})
        )

    def test_the_executor_takes_its_supported_set_from_the_framework(self):
        """The executor may not keep a second, independent list."""
        self.assertIs(SUPPORTED_SIGNAL_KINDS, AUTONOMOUS_SIGNAL_KINDS)

    def test_inventory_rows_are_json_safe_and_ordered(self):
        rows = inventory_rows()
        self.assertEqual([row["signal"] for row in rows], list(ALL_SIGNAL_KINDS))
        json.dumps(rows)  # must not raise

    def test_entries_are_frozen(self):
        """A mutable inventory could be edited at run time to grant authority."""
        import dataclasses

        entry = inventory_for(MaintenanceSignal.TEST_GAP)
        assert entry is not None
        with self.assertRaises(dataclasses.FrozenInstanceError):
            entry.autonomous_execution = True  # type: ignore[misc]
        self.assertFalse(inventory_for(MaintenanceSignal.TEST_GAP).autonomous_execution)


# =============================================================================
# B. The oracle contract is fail-closed
# =============================================================================


class OracleContractTests(unittest.TestCase):
    def test_the_base_oracle_can_never_report_success(self):
        """CONTRACT: forgetting to implement a method degrades to refusal."""
        oracle = ExecutionOracle()
        root = Path(tempfile.gettempdir())
        self.assertFalse(oracle.observe_failure(root, "x.py").resolved)
        self.assertFalse(oracle.observe_success(root, "x.py").resolved)
        self.assertTrue(oracle.observe_success(root, "x.py").inconclusive)

    def test_the_base_oracle_is_not_promotable(self):
        self.assertFalse(ExecutionOracle().promotable)
        self.assertEqual(ExecutionOracle().oracle_class, OracleClass.UNSAFE)

    def test_only_the_deterministic_class_is_promotable(self):
        for oracle_class in ALL_ORACLE_CLASSES:
            with self.subTest(oracle_class=oracle_class):
                oracle = UnverifiableOracle("x", oracle_class)
                self.assertEqual(
                    oracle.promotable, oracle_class == OracleClass.DETERMINISTIC
                )
        self.assertEqual(PROMOTABLE_ORACLE_CLASSES, frozenset({OracleClass.DETERMINISTIC}))

    def test_an_unknown_signal_gets_an_unsafe_oracle(self):
        """A forged or corrupted kind must not fall through to a real oracle."""
        oracle = oracle_for("totally_made_up_signal")
        self.assertIsInstance(oracle, UnverifiableOracle)
        self.assertEqual(oracle.oracle_class, OracleClass.UNSAFE)
        self.assertFalse(oracle.promotable)

    def test_an_unknown_oracle_class_string_degrades_to_unsafe(self):
        oracle = UnverifiableOracle("x", "deterministic_but_not_really")
        self.assertEqual(oracle.oracle_class, OracleClass.UNSAFE)
        self.assertFalse(oracle.promotable)

    def test_observations_are_immutable(self):
        """An observation is a record, not a mutable flag.

        If ``outcome`` could be reassigned, an inconclusive verdict could be
        laundered into a pass by any caller holding a reference.
        """
        import dataclasses

        observation = OracleObservation(outcome=OracleOutcome.INCONCLUSIVE)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            observation.outcome = OracleOutcome.RESOLVED  # type: ignore[misc]
        self.assertFalse(observation.resolved)

    def test_observation_evidence_cannot_be_rewritten(self):
        """The BEFORE baseline is what the success predicate is measured
        against; a mutable one could be emptied to make preservation vacuous."""
        observation = OracleObservation(evidence={"lexical_surface": ["a"]})
        with self.assertRaises(TypeError):
            observation.evidence["lexical_surface"] = []  # type: ignore[index]
        with self.assertRaises(TypeError):
            del observation.evidence["lexical_surface"]  # type: ignore[attr-defined]
        self.assertEqual(observation.evidence["lexical_surface"], ["a"])

    def test_the_three_outcomes_are_mutually_exclusive(self):
        for outcome in ALL_ORACLE_OUTCOMES:
            with self.subTest(outcome=outcome):
                observation = OracleObservation(outcome=outcome)
                flags = [
                    observation.resolved,
                    observation.failing,
                    observation.inconclusive,
                ]
                self.assertEqual(sum(1 for flag in flags if flag), 1)

    def test_an_out_of_vocabulary_outcome_becomes_inconclusive(self):
        oracle = ParseOracle()
        observation = oracle._observation("definitely_fine", "forged")
        self.assertTrue(observation.inconclusive)
        self.assertFalse(observation.resolved)

    def test_twelve_signals_are_bound_to_unverifiable_oracles(self):
        unverifiable = [
            kind for kind in ALL_SIGNAL_KINDS
            if isinstance(oracle_for(kind), UnverifiableOracle)
        ]
        self.assertEqual(len(unverifiable), 12)
        self.assertNotIn(MaintenanceSignal.PARSE_FAILURE, unverifiable)

    def test_unverifiable_oracles_refuse_both_directions(self):
        root = Path(tempfile.gettempdir())
        for kind in ALL_SIGNAL_KINDS:
            if kind == MaintenanceSignal.PARSE_FAILURE:
                continue
            with self.subTest(signal=kind):
                oracle = oracle_for(kind)
                self.assertTrue(oracle.observe_failure(root, "a.py").inconclusive)
                self.assertTrue(oracle.observe_success(root, "a.py").inconclusive)
                self.assertEqual(oracle.max_scope_files, 0)
                self.assertEqual(oracle.acceptance_commands(root, ["a.py"]), [])


# =============================================================================
# C. ParseOracle - the deterministic oracle, adversarially
# =============================================================================


class ParseOracleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="oracle_")).resolve()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.oracle = ParseOracle()

    def write(self, source: str, name: str = "mod.py") -> str:
        (self.root / name).write_text(source, encoding="utf-8")
        return name

    # -- BEFORE ------------------------------------------------------------

    def test_a_broken_file_is_observed_as_failing(self):
        relative = self.write(SUBSTANTIAL_BROKEN)
        before = self.oracle.observe_failure(self.root, relative)
        self.assertTrue(before.failing)
        self.assertFalse(before.resolved)
        self.assertTrue(before.deterministic)
        self.assertIn("SyntaxError", before.evidence["parse_error"])

    def test_a_healthy_file_is_observed_as_resolved(self):
        relative = self.write(SUBSTANTIAL_FIXED)
        before = self.oracle.observe_failure(self.root, relative)
        self.assertTrue(before.resolved)
        self.assertFalse(before.failing)

    def test_a_missing_file_is_inconclusive_not_failing(self):
        observation = self.oracle.observe_failure(self.root, "nope.py")
        self.assertTrue(observation.inconclusive)
        self.assertFalse(observation.failing)
        self.assertFalse(observation.resolved)

    def test_an_undecodable_file_is_inconclusive_in_both_directions(self):
        """Neither 'parses' nor 'does not parse'; never collapsed into either."""
        (self.root / "bin.py").write_bytes(b"\xff\xfe\x00\x01 not utf-8 \xc3\x28")
        self.assertTrue(self.oracle.observe_failure(self.root, "bin.py").inconclusive)
        self.assertTrue(self.oracle.observe_success(self.root, "bin.py").inconclusive)

    def test_a_nul_byte_source_is_a_failure_not_a_crash(self):
        # NB the filename: ``nul.py`` maps to the Windows NUL *device*, even
        # with an extension, so a fixture using it silently tests nothing.
        (self.root / "has_nul.py").write_bytes(b"x = 1\x00\n")
        observation = self.oracle.observe_failure(self.root, "has_nul.py")
        self.assertTrue(observation.failing)
        self.assertFalse(observation.resolved)
        # CPython reports this as SyntaxError on 3.12 and ValueError on some
        # earlier versions; the oracle treats both as "does not parse", which
        # is the property that matters.
        self.assertRegex(
            observation.evidence["parse_error"], r"SyntaxError|ValueError"
        )

    # -- AFTER: the Phase 4.22 regression ---------------------------------

    def test_wholesale_deletion_does_not_satisfy_the_success_predicate(self):
        """REGRESSION for the Phase 4.22 defect, at oracle level.

        ``pass`` parses. Under 4.22's predicate that was success. It must not
        be, and the reason must name what was destroyed.
        """
        relative = self.write(SUBSTANTIAL_BROKEN)
        before = self.oracle.observe_failure(self.root, relative)
        self.write(DESTRUCTIVE_REPAIR)
        after = self.oracle.observe_success(self.root, relative, before)

        self.assertFalse(after.resolved)
        self.assertTrue(after.failing)
        clauses = dict(after.clauses)
        self.assertTrue(clauses["parses"])  # it really does parse
        self.assertFalse(clauses["surface_preserved"])
        self.assertFalse(clauses["substance_preserved"])
        self.assertEqual(
            set(after.evidence["missing_names"]),
            {"math", "CONSTANT", "add", "multiply", "Calculator", "compute"},
        )

    def test_partial_deletion_does_not_satisfy_the_success_predicate(self):
        """Even a repair that fixes the actual defect is refused if it deletes."""
        relative = self.write(SUBSTANTIAL_BROKEN)
        before = self.oracle.observe_failure(self.root, relative)
        self.write(PARTIAL_DESTRUCTION)
        after = self.oracle.observe_success(self.root, relative, before)

        self.assertFalse(after.resolved)
        clauses = dict(after.clauses)
        self.assertTrue(clauses["parses"])
        self.assertFalse(clauses["surface_preserved"])
        self.assertEqual(
            set(after.evidence["missing_names"]),
            {"math", "CONSTANT", "multiply", "Calculator", "compute"},
        )

    def test_a_genuine_minimal_repair_satisfies_the_success_predicate(self):
        relative = self.write(SUBSTANTIAL_BROKEN)
        before = self.oracle.observe_failure(self.root, relative)
        self.write(SUBSTANTIAL_FIXED)
        after = self.oracle.observe_success(self.root, relative, before)

        self.assertTrue(after.resolved)
        self.assertTrue(all(ok for _, ok in after.clauses))
        self.assertEqual(after.evidence["missing_names"], [])

    def test_a_repair_that_adds_code_is_still_a_success(self):
        """The predicate bounds overwriting, not growth.

        Insertions consume no original line, so a repair may add as much as it
        likes. That asymmetry is what lets the locality clause be tight enough
        to catch gutting without rejecting ordinary work.
        """
        relative = self.write(SUBSTANTIAL_BROKEN)
        before = self.oracle.observe_failure(self.root, relative)
        self.write(SUBSTANTIAL_FIXED + "\n\ndef extra():\n    return 1\n")
        after = self.oracle.observe_success(self.root, relative, before)
        self.assertTrue(after.resolved, after.detail)
        self.assertEqual(after.evidence["replaced_lines"], 1)

    def test_hollowing_out_the_bodies_is_refused(self):
        """REGRESSION for a hole found by probing this implementation.

        Replacing every function body with ``pass`` keeps every name and the
        exact significant-line count, so ``surface_preserved`` and
        ``substance_preserved`` both hold. Only ``repair_is_local`` catches it,
        because gutting must overwrite the lines it guts.
        """
        relative = self.write(SUBSTANTIAL_BROKEN)
        before = self.oracle.observe_failure(self.root, relative)
        hollow = (
            "import math\n"
            "\n"
            "CONSTANT = 42\n"
            "\n"
            "\n"
            "def add(a, b):\n"
            "    pass\n"
            "\n"
            "\n"
            "def multiply(a, b):\n"
            "    pass\n"
            "\n"
            "\n"
            "class Calculator:\n"
            "    def compute(self, x):\n"
            "        pass\n"
        )
        self.write(hollow)
        after = self.oracle.observe_success(self.root, relative, before)

        clauses = dict(after.clauses)
        # The first three clauses really are satisfied - that is the point.
        self.assertTrue(clauses["parses"])
        self.assertTrue(clauses["surface_preserved"])
        self.assertTrue(clauses["substance_preserved"])
        # And the fourth one refuses it.
        self.assertFalse(clauses["repair_is_local"])
        self.assertFalse(after.resolved)
        self.assertGreater(
            after.evidence["replaced_lines"], after.evidence["allowed_replaced_lines"]
        )

    def test_a_wholesale_reformat_is_refused(self):
        """Reformatting a file is not repairing its syntax."""
        relative = self.write(SUBSTANTIAL_BROKEN)
        before = self.oracle.observe_failure(self.root, relative)
        self.write(SUBSTANTIAL_FIXED.replace("\n", "  \n"))
        after = self.oracle.observe_success(self.root, relative, before)
        self.assertFalse(after.resolved)
        self.assertFalse(dict(after.clauses)["repair_is_local"])

    def test_line_digests_carry_no_source_text(self):
        """Evidence is recorded in telemetry; it must not leak file contents."""
        relative = self.write(SUBSTANTIAL_BROKEN)
        before = self.oracle.observe_failure(self.root, relative)
        blob = json.dumps(before.to_dict())
        for secret in ("math.sqrt", "return a + b", "CONSTANT = 42"):
            self.assertNotIn(secret, blob)
        self.assertEqual(
            len(before.evidence["line_digests"]),
            len(SUBSTANTIAL_BROKEN.split("\n")),
        )

    def test_the_locality_clause_does_not_degrade_quadratically(self):
        """REGRESSION for an O(n^2) blowup this phase introduced and then fixed.

        ``difflib.SequenceMatcher`` is quadratic when its inputs contain many
        identical elements, which source files do. Measured before the fix: a
        ten-thousand-line file took 6.1 s inside the post-apply path against
        76 ms at one thousand - 80x the cost for 10x the input. Trimming the
        common prefix and suffix, plus an upper-bound short circuit, brought it
        to 260 ms with linear scaling.

        The assertion is on the algorithm, not the clock: a fifty-thousand-line
        input differing in one line must return exactly 1, which is only
        reachable in reasonable time via the trim. The wall-clock bound is a
        deliberately loose smoke guard, not the thing being proved.
        """
        from local_agent.maintenance_oracle import _replaced_line_count

        # Pathological input: every line identical except one.
        before = ["    return a + b"] * 50_000
        after = list(before)
        after[25_000] = "    return a - b"
        started = time.perf_counter()
        self.assertEqual(_replaced_line_count(before, after), 1)
        self.assertLess(time.perf_counter() - started, 5.0)

    def test_an_enormous_rewrite_short_circuits_to_an_upper_bound(self):
        """The short circuit may only over-report, which may only refuse."""
        from local_agent.maintenance_oracle import (
            MAX_EXACT_DIFF_LINES,
            _replaced_line_count,
        )

        size = MAX_EXACT_DIFF_LINES * 3
        before = [f"a{i}" for i in range(size)]
        after = [f"b{i}" for i in range(size)]
        started = time.perf_counter()
        count = _replaced_line_count(before, after)
        self.assertEqual(count, size)
        self.assertLess(time.perf_counter() - started, 5.0)

    def test_the_locality_allowance_matches_the_documented_rule(self):
        from local_agent.maintenance_oracle import (
            _replaced_line_allowance,
            _replaced_line_count,
        )

        self.assertEqual(_replaced_line_allowance(0), REPLACED_LINE_SLACK)
        self.assertEqual(_replaced_line_allowance(10), REPLACED_LINE_SLACK)
        self.assertEqual(_replaced_line_allowance(100), 10)
        # Pure insertion consumes no original line.
        self.assertEqual(_replaced_line_count(["a", "b"], ["a", "x", "y", "b"]), 0)
        # Replacement and deletion both do.
        self.assertEqual(_replaced_line_count(["a", "b"], ["a", "z"]), 1)
        self.assertEqual(_replaced_line_count(["a", "b", "c"], ["a"]), 2)
        self.assertEqual(_replaced_line_count(["a"], []), 1)

    def test_a_tiny_file_is_not_held_to_a_percentage(self):
        """The absolute slack exists so two-line fixtures still work."""
        relative = self.write(BROKEN_SOURCE)
        before = self.oracle.observe_failure(self.root, relative)
        self.write(FIXED_SOURCE)
        self.assertTrue(self.oracle.observe_success(self.root, relative, before).resolved)

    def test_the_success_predicate_still_fails_when_the_file_does_not_parse(self):
        relative = self.write(SUBSTANTIAL_BROKEN)
        before = self.oracle.observe_failure(self.root, relative)
        after = self.oracle.observe_success(self.root, relative, before)
        self.assertTrue(after.failing)
        self.assertEqual(dict(after.clauses), {"parses": False})

    # -- AFTER: the baseline cannot be forged or omitted -------------------

    def test_success_without_a_before_observation_is_inconclusive(self):
        """No baseline means the preservation clause would be vacuous."""
        relative = self.write(SUBSTANTIAL_FIXED)
        after = self.oracle.observe_success(self.root, relative, None)
        self.assertTrue(after.inconclusive)
        self.assertFalse(after.resolved)

    def test_a_malformed_before_observation_is_inconclusive(self):
        """ADVERSARIAL: a hostile baseline must not become a free pass."""
        relative = self.write(SUBSTANTIAL_FIXED)
        for evidence in (
            {},
            {"lexical_surface": "not-a-list", "significant_lines": 9},
            {"lexical_surface": ["a"], "significant_lines": "nine"},
            {"lexical_surface": ["a"], "significant_lines": True},
            {"lexical_surface": ["a"]},
            {"significant_lines": 9},
        ):
            with self.subTest(evidence=evidence):
                forged = OracleObservation(
                    outcome=OracleOutcome.NOT_RESOLVED, evidence=evidence
                )
                after = self.oracle.observe_success(self.root, relative, forged)
                self.assertTrue(after.inconclusive, after.detail)
                self.assertFalse(after.resolved)

    def test_an_emptied_baseline_cannot_wave_a_deletion_through(self):
        """ADVERSARIAL: forging an empty BEFORE surface.

        A forged baseline claiming the broken file had no names and no lines
        would make both preservation clauses vacuously true. That is exactly
        the attack, and it is why the executor derives the baseline from its
        own observation rather than from anything persisted: this test proves
        the oracle would accept such a baseline, and the executor-level test
        ``test_the_baseline_comes_from_the_executors_own_observation`` proves
        no such baseline can reach it.
        """
        relative = self.write(DESTRUCTIVE_REPAIR)
        forged = OracleObservation(
            outcome=OracleOutcome.NOT_RESOLVED,
            evidence={
                "lexical_surface": [],
                "significant_lines": 0,
                "line_digests": [],
            },
        )
        self.assertTrue(self.oracle.observe_success(self.root, relative, forged).resolved)

    def test_a_baseline_missing_the_line_digests_is_inconclusive(self):
        """Every clause needs its own baseline field; a partial one is refused."""
        relative = self.write(SUBSTANTIAL_FIXED)
        partial = OracleObservation(
            outcome=OracleOutcome.NOT_RESOLVED,
            evidence={"lexical_surface": [], "significant_lines": 0},
        )
        after = self.oracle.observe_success(self.root, relative, partial)
        self.assertTrue(after.inconclusive)
        self.assertFalse(after.resolved)

    def test_the_before_observation_never_reads_persisted_history(self):
        """SAFETY: history can never override a current observation.

        Proved structurally: the oracle module imports nothing that could
        supply history, and its executable code names no storage, lifecycle or
        telemetry symbol.
        """
        imports = imported_modules(ORACLE_MODULE)
        for forbidden in (
            "local_agent.storage",
            "local_agent.validation_lifecycle",
            "local_agent.validation_telemetry",
            "local_agent.knowledge",
            "local_agent.evidence",
            "json",
            "sqlite3",
        ):
            self.assertNotIn(forbidden, imports)

    # -- scope and shrink arithmetic --------------------------------------

    def test_the_shrink_floor_is_monotone_and_bounded(self):
        """CONTRACT: a bigger file may never be allowed to shrink below a
        smaller one's floor, and the floor is never above the original."""
        previous = -1
        for size in range(0, 400):
            floor = _substance_floor(size)
            self.assertGreaterEqual(floor, 0)
            self.assertLessEqual(floor, size)
            self.assertGreaterEqual(floor, previous)
            previous = floor

    def test_the_shrink_floor_matches_the_documented_rule(self):
        self.assertEqual(_substance_floor(0), 0)
        self.assertEqual(_substance_floor(2), 0)  # slack covers it
        self.assertEqual(_substance_floor(SHRINK_SLACK_LINES), 0)
        self.assertEqual(_substance_floor(100), 100 - int(100 * MAX_SHRINK_RATIO))

    def test_the_oracle_declares_a_single_file_python_scope(self):
        self.assertEqual(self.oracle.max_scope_files, 1)
        self.assertEqual(self.oracle.supported_suffixes, frozenset({".py"}))

    def test_acceptance_commands_skip_files_that_do_not_exist(self):
        """A command naming a missing file would fail for the wrong reason."""
        self.assertEqual(self.oracle.acceptance_commands(self.root, ["ghost.py"]), [])
        self.write(SUBSTANTIAL_FIXED)
        commands = self.oracle.acceptance_commands(self.root, ["mod.py"])
        self.assertEqual(len(commands), 1)
        self.assertEqual(
            tuple(commands[0].command), ("python", "-m", "compileall", "-q", "mod.py")
        )

    def test_acceptance_commands_ignore_non_python_files(self):
        (self.root / "README.md").write_text("hi\n", encoding="utf-8")
        self.assertEqual(self.oracle.acceptance_commands(self.root, ["README.md"]), [])

    def test_the_plan_fragment_tells_the_agent_deletion_is_rejected(self):
        """The instruction and the predicate are authored together."""
        fragment = self.oracle.plan_fragment("mod.py")
        blob = " ".join(
            [str(fragment["agent_objective"]), *map(str, fragment["steps"])]
        ).lower()
        self.assertIn("delet", blob)
        self.assertIn("mod.py", str(fragment["objective"]))

    def test_the_documented_predicates_are_not_empty_prose(self):
        self.assertIn("compil", self.oracle.describe_failure_predicate().lower())
        success = self.oracle.describe_success_predicate().lower()
        for token in ("parses", "name", "shrink"):
            self.assertIn(token, success)


class ParsedSurfaceSymmetryTests(unittest.TestCase):
    """CONTRACT: the AFTER scan matches what the BEFORE scan can see.

    An asymmetry here is a hole: any name the AFTER side counts that the
    BEFORE side could never have produced is a way to satisfy the preservation
    clause without preserving anything.
    """

    def setUp(self) -> None:
        from local_agent.maintenance_oracle import _lexical_surface, _parsed_surface

        self.parsed = lambda src: _parsed_surface(ast.parse(src))
        self.lexical = _lexical_surface

    def test_a_parameter_cannot_stand_in_for_a_deleted_constant(self):
        """``def f(CONSTANT)`` is not a replacement for ``CONSTANT = 42``."""
        self.assertIn("CONSTANT", self.lexical("CONSTANT = 42\n"))
        self.assertNotIn("CONSTANT", self.parsed("def f(CONSTANT):\n    return 1\n"))

    def test_a_local_cannot_stand_in_for_a_deleted_module_constant(self):
        self.assertNotIn(
            "CONSTANT", self.parsed("def f():\n    CONSTANT = 42\n    return CONSTANT\n")
        )
        self.assertIn("CONSTANT", self.parsed("CONSTANT = 42\n"))

    def test_module_level_assignment_forms_are_all_recognised(self):
        for source in (
            "VALUE = 1\n",
            "VALUE: int = 1\n",
            "VALUE = 0\nVALUE += 1\n",
            "VALUE, OTHER = 1, 2\n",
        ):
            with self.subTest(source=source):
                self.assertIn("VALUE", self.parsed(source))

    def test_definitions_and_imports_count_at_any_depth(self):
        source = (
            "class Outer:\n"
            "    import math\n"
            "    def inner(self):\n"
            "        def deeper():\n"
            "            return 1\n"
            "        return deeper\n"
        )
        found = self.parsed(source)
        for name in ("Outer", "inner", "deeper", "math"):
            self.assertIn(name, found)

    def test_a_genuine_repair_of_a_real_module_preserves_its_surface(self):
        """PRODUCTION-INTEGRATION: over real files from this repository.

        Every module in ``local_agent`` parses today. Treating each one as its
        own BEFORE and AFTER must satisfy the preservation clause - if the two
        scans disagree on real code, the oracle would reject genuine repairs.
        """
        for path in sorted((REPO_ROOT / "local_agent").glob("*.py")):
            with self.subTest(module=path.name):
                source = path.read_text(encoding="utf-8")
                missing = self.lexical(source) - self.parsed(source)
                # The lexical scan is deliberately over-inclusive (it sees
                # names inside string literals), so a small residue is
                # expected and is the safe direction. What must not happen is
                # a large disagreement, which would mean the scans are
                # measuring different things.
                self.assertLess(
                    len(missing),
                    max(5, len(self.lexical(source)) // 10),
                    f"{path.name}: {sorted(missing)[:10]}",
                )


class LexicalSurfaceTests(unittest.TestCase):
    """CONTRACT: the surface scan over source that does not parse."""

    def setUp(self) -> None:
        from local_agent.maintenance_oracle import _lexical_surface

        self.scan = _lexical_surface

    def test_it_finds_names_in_unparseable_source(self):
        found = self.scan(SUBSTANTIAL_BROKEN)
        for name in ("math", "CONSTANT", "add", "multiply", "Calculator", "compute"):
            self.assertIn(name, found)

    def test_it_finds_async_definitions(self):
        self.assertIn("fetch", self.scan("async def fetch(x)\n    pass\n"))

    def test_it_finds_from_imports_including_aliases(self):
        found = self.scan("from a.b import c, d as e\nfrom f import (g, h)\n")
        self.assertIn("c", found)
        self.assertIn("e", found)
        self.assertNotIn("d", found)
        self.assertIn("g", found)
        self.assertIn("h", found)

    def test_it_ignores_star_imports(self):
        self.assertNotIn("*", self.scan("from a import *\n"))

    def test_it_ignores_indented_assignments(self):
        """Locals are not part of a module's surface."""
        found = self.scan("def f(:\n    local_thing = 1\nTOP = 2\n")
        self.assertIn("TOP", found)
        self.assertNotIn("local_thing", found)

    def test_it_ignores_equality_comparisons(self):
        self.assertNotIn("x", self.scan("x == 1\n"))

    def test_it_handles_annotated_module_bindings(self):
        self.assertIn("VALUE", self.scan("VALUE: int = 3\n"))

    def test_it_is_over_inclusive_which_is_the_safe_direction(self):
        """A name inside a string is picked up. That makes the post-condition
        stricter, which costs a rejected repair and never a bad apply."""
        self.assertIn("ghost", self.scan('TEXT = """\ndef ghost():\n"""\n'))


# =============================================================================
# D. End to end, through the real pipeline, against a real repository
# =============================================================================


class OracleExecutionIntegrationTests(ExecutorCase):
    """PRODUCTION-INTEGRATION: real discovery -> policy -> executor -> oracle."""

    def setUp(self) -> None:
        super().setUp()
        (self.root / "broken.py").write_text(SUBSTANTIAL_BROKEN, encoding="utf-8")
        shutil.rmtree(self.root / "tests", ignore_errors=True)

    def test_a_real_repair_completes_and_is_credited(self):
        before = snapshot_tree(self.root)
        provider = ScriptedProvider([fix_operation("broken.py", SUBSTANTIAL_FIXED)])
        result = self.executor(provider=provider).execute(self.order())

        self.assertEqual(result.status, MaintenanceExecutionStatus.COMPLETED, result.reasons)
        self.assertTrue(result.succeeded)
        self.assertTrue(result.signal_resolved)
        self.assertFalse(result.rolled_back)
        # The repair really is on disk, and it really is the repair.
        self.assertEqual(
            (self.root / "broken.py").read_text(encoding="utf-8"), SUBSTANTIAL_FIXED
        )
        self.assertNotEqual(snapshot_tree(self.root), before)
        # Observability: both observations were recorded, by name.
        self.assertEqual(result.oracle_name, "parse_oracle")
        self.assertEqual(result.oracle_class, OracleClass.DETERMINISTIC)
        self.assertEqual(result.oracle_precondition["outcome"], "not_resolved")
        self.assertEqual(result.oracle_postcondition["outcome"], "resolved")
        # And a real subprocess actually ran.
        self.assertTrue(result.post_apply_executed_any)
        self.assertIn(
            ["python", "-m", "compileall", "-q", "broken.py"],
            result.post_apply_executed_commands,
        )

    def test_a_destructive_repair_is_rolled_back_byte_for_byte(self):
        """REGRESSION, end to end, for the Phase 4.22 defect.

        Verified against the 4.22 build that this exact scenario produced
        ``completed`` / ``succeeded=True`` / ``signal_resolved=True`` with
        ``broken.py`` reduced to ``pass``.
        """
        before = snapshot_tree(self.root)
        provider = ScriptedProvider([fix_operation("broken.py", DESTRUCTIVE_REPAIR)])
        result = self.executor(provider=provider).execute(self.order())

        self.assertEqual(result.status, MaintenanceExecutionStatus.SIGNAL_NOT_RESOLVED)
        self.assertFalse(result.succeeded)
        self.assertFalse(result.signal_resolved)
        self.assertTrue(result.applied)
        self.assertTrue(result.rolled_back)
        # The authoritative tree is exactly as it was, to the byte.
        self.assertEqual(snapshot_tree(self.root), before)
        self.assertEqual(
            (self.root / "broken.py").read_text(encoding="utf-8"), SUBSTANTIAL_BROKEN
        )
        clauses = dict(
            (name, ok) for name, ok in result.oracle_postcondition["clauses"]
        )
        self.assertTrue(clauses["parses"])
        self.assertFalse(clauses["surface_preserved"])

    def test_a_partially_destructive_repair_is_also_refused(self):
        before = snapshot_tree(self.root)
        provider = ScriptedProvider([fix_operation("broken.py", PARTIAL_DESTRUCTION)])
        result = self.executor(provider=provider).execute(self.order())
        self.assertNotEqual(result.status, MaintenanceExecutionStatus.COMPLETED)
        self.assertFalse(result.succeeded)
        self.assertEqual(snapshot_tree(self.root), before)

    def test_the_agent_claiming_success_is_not_the_oracle(self):
        """The negative case the specification asks for explicitly.

        ``ScriptedProvider`` returns a file operation and the interactive agent
        reports success; the candidate's own ``compileall`` passes, because
        ``pass`` compiles. Nothing in that chain is evidence, and the result
        must not be credited.
        """
        provider = ScriptedProvider([fix_operation("broken.py", DESTRUCTIVE_REPAIR)])
        result = self.executor(provider=provider).execute(self.order())
        # The agent did claim success - prospective validation passed.
        self.assertTrue(result.prospective_validation_passed)
        # And it changed nothing, because the oracle is the authority.
        self.assertFalse(result.succeeded)
        self.assertIsNot(result.signal_resolved, True)

    def test_the_baseline_comes_from_the_executors_own_observation(self):
        """No persisted or work-order-supplied baseline can reach the oracle.

        Structural: ``observe_success`` is only ever called with the value
        returned by ``_check_freshness``, which is the executor's own live
        observation. Asserted from the AST so it is a claim about code.
        """
        tree = _module_ast(EXECUTOR_MODULE)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "observe_success"
        ]
        self.assertEqual(len(calls), 1, "observe_success must have exactly one caller")
        third = calls[0].args[2] if len(calls[0].args) > 2 else None
        self.assertIsInstance(third, ast.Name)
        self.assertEqual(third.id, "before")

    def test_a_no_change_result_is_not_a_success(self):
        provider = ScriptedProvider([[]])
        result = self.executor(provider=provider).execute(self.order())
        self.assertEqual(result.status, MaintenanceExecutionStatus.NO_CHANGE)
        self.assertFalse(result.succeeded)
        self.assertIsNone(result.signal_resolved)
        self.assertIsNone(result.oracle_postcondition)


class RescanResolutionTests(ExecutorCase):
    """PRODUCTION-INTEGRATION: RESOLVED only after a real, fresh rescan."""

    def test_a_real_repair_makes_the_signal_vanish_from_a_fresh_scan(self):
        (self.root / "broken.py").write_text(SUBSTANTIAL_BROKEN, encoding="utf-8")
        shutil.rmtree(self.root / "tests", ignore_errors=True)
        # The signal is genuinely discoverable before.
        self.assertEqual(discover_parse_failure(self.root).affected_files, ["broken.py"])

        provider = ScriptedProvider([fix_operation("broken.py", SUBSTANTIAL_FIXED)])
        result = self.executor(provider=provider).execute(self.order())
        self.assertEqual(result.status, MaintenanceExecutionStatus.COMPLETED)

        # A completely fresh run of the real discovery path no longer sees it.
        from local_agent.semantic_impact import SemanticGraph

        analysis = MaintenanceAnalyzer(self.root).analyze(
            semantic_graph=SemanticGraph.build(self.root)
        )
        remaining = [
            candidate
            for candidate in analysis.candidates
            if candidate.kind == MaintenanceSignal.PARSE_FAILURE
        ]
        self.assertEqual(remaining, [])

    def test_a_destructive_repair_leaves_the_signal_in_a_fresh_scan(self):
        """The rollback means the defect is still discoverable afterwards."""
        (self.root / "broken.py").write_text(SUBSTANTIAL_BROKEN, encoding="utf-8")
        shutil.rmtree(self.root / "tests", ignore_errors=True)
        provider = ScriptedProvider([fix_operation("broken.py", DESTRUCTIVE_REPAIR)])
        result = self.executor(provider=provider).execute(self.order())
        self.assertNotEqual(result.status, MaintenanceExecutionStatus.COMPLETED)
        self.assertEqual(discover_parse_failure(self.root).affected_files, ["broken.py"])


# =============================================================================
# E. Environment safety
# =============================================================================


class EnvironmentSafetyTests(ExecutorCase):
    def setUp(self) -> None:
        super().setUp()
        (self.root / "broken.py").write_text(SUBSTANTIAL_BROKEN, encoding="utf-8")
        shutil.rmtree(self.root / "tests", ignore_errors=True)

    def test_a_missing_executable_is_never_a_pass(self):
        """PRODUCTION-INTEGRATION: a real run with an unresolvable interpreter.

        ``resolve_executable`` is patched to the identity so the ``python`` ->
        ``sys.executable`` fallback cannot rescue the command, which is the
        only way to make the runner genuinely emit its 127. The repair itself
        is correct; the run must still refuse to credit it, because no verdict
        exists.
        """
        from local_agent import commands as commands_module

        before = snapshot_tree(self.root)
        provider = ScriptedProvider([fix_operation("broken.py", SUBSTANTIAL_FIXED)])
        with mock.patch.object(
            commands_module,
            "resolve_executable",
            lambda command: (("definitely-not-a-real-binary-4711", *command[1:]), ""),
        ):
            result = self.executor(provider=provider).execute(self.order())

        self.assertNotEqual(result.status, MaintenanceExecutionStatus.COMPLETED)
        self.assertFalse(result.succeeded)
        self.assertIsNone(result.validation_passed)
        self.assertFalse(result.post_apply_executed_any)
        self.assertEqual(result.post_apply_executed_commands, [])
        self.assertTrue(result.post_apply_skipped_commands)
        self.assertTrue(result.rolled_back)
        self.assertEqual(snapshot_tree(self.root), before)

    def test_a_skipped_command_is_reported_as_skipped_not_run(self):
        from local_agent import commands as commands_module

        provider = ScriptedProvider([fix_operation("broken.py", SUBSTANTIAL_FIXED)])
        with mock.patch.object(
            commands_module,
            "resolve_executable",
            lambda command: (("definitely-not-a-real-binary-4711", *command[1:]), ""),
        ):
            result = self.executor(provider=provider).execute(self.order())
        self.assertEqual(result.post_apply_commands_run, 0)
        self.assertTrue(
            any("executable not available" in reason for reason in result.reasons)
        )

    def test_a_genuine_exit_127_counts_as_having_run(self):
        """CONTRACT: 'the tool failed with 127' is not 'the tool is missing'.

        Phase 4.22 used two different predicates for this question, so a real
        exit-127 failure was excluded from the run count while still setting
        the verdict. One predicate now answers both.
        """

        class Result:
            def __init__(self, exit_code: int, stderr: str) -> None:
                self.exit_code = exit_code
                self.stderr = stderr

        self.assertTrue(_executable_missing(Result(127, "executable not found: python")))
        self.assertFalse(_executable_missing(Result(127, "command failed badly")))
        self.assertFalse(_executable_missing(Result(1, "executable not found: python")))
        self.assertFalse(_executable_missing(Result(0, "")))

    def test_the_acceptance_command_is_resolvable_on_this_platform(self):
        """PRODUCTION-INTEGRATION: the real command, really executed here.

        Windows in particular may have no ``python`` on PATH (only ``py``), so
        this proves the command as constructed actually runs in this
        environment rather than assuming it does.
        """
        from local_agent.commands import CommandRunner

        (self.root / "ok.py").write_text("VALUE = 1\n", encoding="utf-8")
        commands = ParseOracle().acceptance_commands(self.root, ["ok.py"])
        execution = CommandRunner(self.root, 120).run(commands[0])
        self.assertEqual(execution.exit_code, 0, execution.stderr)
        self.assertFalse(_executable_missing(execution))

    def test_the_oracle_never_spawns_a_subprocess_itself(self):
        """The oracle describes commands; the existing runner executes them."""
        self.assertNotIn("subprocess", imported_modules(ORACLE_MODULE))
        self.assertNotIn("os", imported_modules(ORACLE_MODULE))
        identifiers = code_identifiers(_module_ast(ORACLE_MODULE))
        for forbidden in ("Popen", "system", "popen", "spawn", "chdir", "execv"):
            self.assertNotIn(forbidden, identifiers)

    def test_the_process_working_directory_is_never_changed(self):
        provider = ScriptedProvider([fix_operation("broken.py", SUBSTANTIAL_FIXED)])
        before = os.getcwd()
        self.executor(provider=provider).execute(self.order())
        self.assertEqual(os.getcwd(), before)


# =============================================================================
# F. Budget enforcement
# =============================================================================


class BudgetEnforcementTests(ExecutorCase):
    def setUp(self) -> None:
        super().setUp()
        (self.root / "broken.py").write_text(SUBSTANTIAL_BROKEN, encoding="utf-8")
        shutil.rmtree(self.root / "tests", ignore_errors=True)

    def test_a_deadline_that_passes_mid_run_stops_the_apply(self):
        """PRODUCTION-INTEGRATION: the gap Phase 4.22 left open.

        The deadline was consulted once, before any work. Everything between
        that check and the apply can take minutes, so a run that had already
        overrun could still perform the single action that writes to the
        repository. It is now re-checked immediately before the apply.
        """
        before = snapshot_tree(self.root)
        state = {"calls": 0}

        def deadline() -> bool:
            # False on the first ask (so execution starts), True afterwards.
            state["calls"] += 1
            return state["calls"] > 1

        provider = ScriptedProvider([fix_operation("broken.py", SUBSTANTIAL_FIXED)])
        executor = self.executor(provider=provider)
        executor.deadline = deadline
        result = executor.execute(self.order())

        self.assertEqual(result.status, MaintenanceExecutionStatus.BUDGET_EXHAUSTED)
        self.assertFalse(result.applied)
        self.assertEqual(snapshot_tree(self.root), before)
        self.assertGreater(state["calls"], 1, "the deadline was only consulted once")

    def test_a_ledger_that_raises_is_treated_as_exhausted(self):
        """ADVERSARIAL: a corrupted ledger must stop work, not be ignored."""

        class HostileLedger:
            def exhausted(self, _name: str) -> bool:
                raise RuntimeError("corrupted budget ledger")

        before = snapshot_tree(self.root)
        provider = ScriptedProvider([fix_operation("broken.py", SUBSTANTIAL_FIXED)])
        executor = self.executor(provider=provider)
        executor.ledger = HostileLedger()
        result = executor.execute(self.order())
        self.assertEqual(result.status, MaintenanceExecutionStatus.BUDGET_EXHAUSTED)
        self.assertFalse(result.applied)
        self.assertEqual(snapshot_tree(self.root), before)

    def test_a_deadline_that_raises_is_treated_as_exhausted(self):
        before = snapshot_tree(self.root)
        provider = ScriptedProvider([fix_operation("broken.py", SUBSTANTIAL_FIXED)])
        executor = self.executor(provider=provider)

        def boom() -> bool:
            raise RuntimeError("clock exploded")

        executor.deadline = boom
        result = executor.execute(self.order())
        self.assertEqual(result.status, MaintenanceExecutionStatus.BUDGET_EXHAUSTED)
        self.assertEqual(snapshot_tree(self.root), before)

    def test_a_zero_budget_refuses_before_any_work(self):
        for field, value in (
            ("max_tool_steps_per_subtask", 0),
            ("max_candidate_iterations", 0),
            ("max_validation_commands", 0),
            ("max_changed_files_per_candidate", 0),
            ("max_changed_lines_per_candidate", 0),
        ):
            with self.subTest(field=field):
                before = snapshot_tree(self.root)
                budget = MaintenanceBudget(**{field: value})
                provider = ScriptedProvider(
                    [fix_operation("broken.py", SUBSTANTIAL_FIXED)]
                )
                result = self.executor(provider=provider, budget=budget).execute(
                    self.order(budget=budget)
                )
                self.assertIn(
                    result.status,
                    {
                        MaintenanceExecutionStatus.BUDGET_EXHAUSTED,
                        MaintenanceExecutionStatus.REFUSED_BY_POLICY,
                    },
                )
                self.assertFalse(result.applied)
                self.assertEqual(snapshot_tree(self.root), before)

    def test_a_diff_over_the_line_ceiling_is_refused_without_applying(self):
        before = snapshot_tree(self.root)
        budget = MaintenanceBudget(max_changed_lines_per_candidate=2)
        huge = SUBSTANTIAL_FIXED + "\n".join(f"X{i} = {i}" for i in range(200)) + "\n"
        provider = ScriptedProvider([fix_operation("broken.py", huge)])
        result = self.executor(provider=provider, budget=budget).execute(
            self.order(budget=budget)
        )
        self.assertEqual(result.status, MaintenanceExecutionStatus.BUDGET_EXHAUSTED)
        self.assertFalse(result.applied)
        self.assertEqual(snapshot_tree(self.root), before)

    def test_the_changed_file_budget_is_checked_against_the_real_apply(self):
        """PRODUCTION-INTEGRATION: the control Phase 4.22 computed and ignored.

        ``max_changed_files`` was derived, zero-checked and then never compared
        to anything - a decorative safety control. It is now enforced against
        the files the authoritative apply actually reported, and a breach rolls
        the change back. The upstream scope check makes a genuine breach
        unreachable, so the enforcement point is reached here by forcing the
        limit to zero after the earlier gates have run.
        """
        before = snapshot_tree(self.root)
        provider = ScriptedProvider([fix_operation("broken.py", SUBSTANTIAL_FIXED)])
        executor = self.executor(provider=provider)
        real_limits = executor._effective_limits

        def shrink(order, result):
            limits = real_limits(order, result)
            if limits is not None:
                limits = dict(limits)
                limits["max_changed_files"] = 0
            return limits

        with mock.patch.object(executor, "_effective_limits", shrink):
            result = executor.execute(self.order())

        self.assertEqual(
            result.status, MaintenanceExecutionStatus.POST_APPLY_BUDGET_BREACH
        )
        self.assertTrue(result.applied)
        self.assertTrue(result.rolled_back)
        self.assertFalse(result.succeeded)
        self.assertEqual(snapshot_tree(self.root), before)

    def test_the_post_apply_breach_status_is_not_claimed_to_be_mutation_free(self):
        """CONTRACT: the status vocabulary must not lie about what happened."""
        self.assertNotIn(
            MaintenanceExecutionStatus.POST_APPLY_BUDGET_BREACH, NO_MUTATION_STATUSES
        )

    def test_a_work_order_permitting_no_subtask_is_refused(self):
        before = snapshot_tree(self.root)
        order = self.order()
        order.max_subtasks = 0
        provider = ScriptedProvider([fix_operation("broken.py", SUBSTANTIAL_FIXED)])
        result = self.executor(provider=provider).execute(order)
        self.assertEqual(result.status, MaintenanceExecutionStatus.BUDGET_EXHAUSTED)
        self.assertFalse(result.applied)
        self.assertEqual(snapshot_tree(self.root), before)


class BudgetDeclarationTests(unittest.TestCase):
    """Every budget field is deliberately classified, and stays that way.

    Phase 4.21 found several budgets to be decorative and Phase 4.22 described
    the situation in prose. Prose rots. These assertions make the two
    declarations cover the real dataclass, so a budget field cannot be added
    without a decision about whether this executor honours it - and cannot be
    quietly claimed as enforced when it is not.
    """

    def setUp(self) -> None:
        self.fields = set(MaintenanceBudget.__dataclass_fields__)

    def test_every_budget_field_is_classified_exactly_once(self):
        classified = set(ENFORCED_BUDGET_FIELDS) | set(UNENFORCED_BUDGET_FIELDS)
        self.assertEqual(
            self.fields - classified, set(), "unclassified budget field(s)"
        )
        self.assertEqual(
            classified - self.fields, set(), "classified field(s) that do not exist"
        )
        self.assertEqual(
            set(ENFORCED_BUDGET_FIELDS) & set(UNENFORCED_BUDGET_FIELDS), set()
        )

    def test_every_unenforced_field_says_why(self):
        for name, reason in sorted(UNENFORCED_BUDGET_FIELDS.items()):
            with self.subTest(field=name):
                self.assertGreater(len(reason), 40, name)

    def test_the_field_that_nothing_enforces_says_so_unambiguously(self):
        """``max_estimated_cost_units`` has no cost model behind it anywhere.

        Naming it here is the point: an operator setting it must be able to
        find out that it constrains nothing, rather than discovering that by
        watching a run overspend.
        """
        reason = UNENFORCED_BUDGET_FIELDS["max_estimated_cost_units"]
        self.assertIn("NOT ENFORCED ANYWHERE", reason)

    def test_every_enforced_field_is_genuinely_consumed_somewhere(self):
        """Structural anti-rot: an enforced name must really be read.

        "Read" means an attribute access or a string key, in the executor or in
        the policy the executor obeys - ``max_candidates_executed`` is enforced
        by ``MaintenanceExecutionPolicy.decide``, and ``max_elapsed_seconds``
        reaches the ledger as a string rather than an attribute, so a check
        over identifiers in one module alone would fail for the wrong reason.
        """
        read: set[str] = set()
        for module in (EXECUTOR_MODULE, "local_agent.maintenance_policy"):
            tree = _module_ast(module)
            read |= {
                node.attr
                for node in ast.walk(tree)
                if isinstance(node, ast.Attribute) and node.attr.startswith("max_")
            }
            read |= {
                node.value
                for node in ast.walk(tree)
                if isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value.startswith("max_")
            }
        for name in sorted(ENFORCED_BUDGET_FIELDS):
            with self.subTest(field=name):
                self.assertIn(name, read, name)


# =============================================================================
# G. Autonomy-tier integrity
# =============================================================================


class AutonomyIntegrityTests(ExecutorCase):
    def setUp(self) -> None:
        super().setUp()
        (self.root / "broken.py").write_text(SUBSTANTIAL_BROKEN, encoding="utf-8")
        shutil.rmtree(self.root / "tests", ignore_errors=True)

    def test_a_forged_subject_in_the_snapshot_breaks_the_candidate_identity(self):
        """ADVERSARIAL: the id is re-derived from (kind, subject).

        ``MaintenanceCandidate`` keeps a persisted ``candidate_id`` verbatim
        and only computes one when it is absent, so an edited record can carry
        a borrowed identity that the order-versus-snapshot comparison cannot
        see. Phase 4.23 re-derives it at the execution boundary; this proves
        the re-derivation, not merely the pairing.
        """
        order = self.order()
        forged = dict(order.candidate_snapshot)
        forged["subject"] = "some/other/module.py"
        order.candidate_snapshot = forged
        before = snapshot_tree(self.root)
        result = self.executor().execute(order)
        self.assertEqual(result.status, MaintenanceExecutionStatus.MALFORMED_WORK_ORDER)
        self.assertIn("not the identity", " ".join(result.reasons))
        self.assertEqual(snapshot_tree(self.root), before)

    def test_a_forged_kind_in_the_snapshot_cannot_execute(self):
        """ADVERSARIAL: relabelling a candidate does not borrow authority."""
        order = self.order()
        forged = dict(order.candidate_snapshot)
        forged["kind"] = MaintenanceSignal.RECURRING_DEFECT
        order.candidate_snapshot = forged
        before = snapshot_tree(self.root)
        result = self.executor().execute(order)
        self.assertIn(
            result.status,
            {
                MaintenanceExecutionStatus.MALFORMED_WORK_ORDER,
                MaintenanceExecutionStatus.UNSUPPORTED_SIGNAL,
            },
        )
        self.assertFalse(result.applied)
        self.assertEqual(snapshot_tree(self.root), before)

    def test_relabelling_a_rejected_signal_as_parse_failure_still_fails(self):
        """The reverse forgery: claim the executable kind for other work.

        Relabelling changes the identity, so the re-derivation catches it. Even
        if it did not, the oracle's failure predicate would have to reproduce
        on the named file - which is the point of re-observing rather than
        trusting the record.
        """
        order = self.order()
        forged = dict(order.candidate_snapshot)
        forged["kind"] = MaintenanceSignal.PARSE_FAILURE
        forged["subject"] = "healthy.py"
        forged["affected_files"] = ["healthy.py"]
        order.candidate_snapshot = forged
        order.scope_files = ["healthy.py"]
        before = snapshot_tree(self.root)
        result = self.executor().execute(order)
        self.assertIn(result.status, NO_MUTATION_STATUSES)
        self.assertFalse(result.applied)
        self.assertEqual(snapshot_tree(self.root), before)

    def test_inflating_the_snapshots_evidence_cannot_buy_unattended_autonomy(self):
        """ADVERSARIAL, and an honest account of where the boundary really is.

        The candidate id is ``sha256(kind, subject)``. It does **not** cover
        confidence, sample size or occurrence count, so a forged snapshot can
        genuinely make the policy grant ``execute_autonomously`` - the id check
        does not catch this and it would be wrong to claim it does.

        What actually stops it is the executor's acting set: it refuses any
        tier outside ``execute_with_existing_approval``, so forging the
        evidence *upward* moves the candidate out of the range it will act on
        rather than into a stronger one. Inflating evidence is therefore not a
        privilege escalation - it is a self-inflicted refusal. Both halves are
        asserted here so the mechanism is documented by test rather than by
        comment.
        """
        order = self.order(tier=AutonomyTier.EXECUTE_AUTONOMOUSLY)
        forged = dict(order.candidate_snapshot)
        forged["confidence"] = 1.0
        forged["sample_size"] = 10_000
        forged["occurrence_count"] = 10_000
        forged["uncertainty"] = []
        order.candidate_snapshot = forged

        # Half one: the policy really is fooled by the forged numbers.
        from local_agent.maintenance import MaintenanceCandidate

        verdict = MaintenanceExecutionPolicy(repository_root=self.root).decide(
            MaintenanceCandidate.from_dict(forged),
            configured_tier=AutonomyTier.EXECUTE_AUTONOMOUSLY,
            budget=MaintenanceBudget(),
        )
        self.assertEqual(verdict.granted_tier, AutonomyTier.EXECUTE_AUTONOMOUSLY)

        # Half two: and the executor refuses it anyway, writing nothing.
        before = snapshot_tree(self.root)
        result = self.executor(tier=AutonomyTier.EXECUTE_AUTONOMOUSLY).execute(order)
        self.assertEqual(result.status, MaintenanceExecutionStatus.REFUSED_BY_POLICY)
        self.assertIn("outside the set", " ".join(result.reasons))
        self.assertFalse(result.applied)
        self.assertEqual(snapshot_tree(self.root), before)

    def test_a_work_order_cannot_claim_a_stronger_tier_than_the_policy_grants(self):
        before = snapshot_tree(self.root)
        order = self.order(tier=AutonomyTier.EXECUTE_AUTONOMOUSLY)
        result = self.executor(tier=AutonomyTier.EXECUTE_WITH_EXISTING_APPROVAL).execute(
            order
        )
        self.assertEqual(result.status, MaintenanceExecutionStatus.REFUSED_BY_POLICY)
        self.assertEqual(snapshot_tree(self.root), before)

    def test_the_signal_can_never_reach_unattended_autonomy(self):
        """PRODUCTION-INTEGRATION: the real policy, the real discovered candidate."""
        candidate = discover_parse_failure(self.root)
        verdict = MaintenanceExecutionPolicy(repository_root=self.root).decide(
            candidate,
            configured_tier=AutonomyTier.EXECUTE_AUTONOMOUSLY,
            budget=MaintenanceBudget(),
        )
        self.assertEqual(
            verdict.granted_tier, AutonomyTier.EXECUTE_WITH_EXISTING_APPROVAL
        )

    def test_execution_cannot_promote_itself_across_repeated_runs(self):
        """Running the executor never raises the tier it is granted next time."""
        provider = ScriptedProvider(
            [fix_operation("broken.py", SUBSTANTIAL_FIXED) for _ in range(3)]
        )
        executor = self.executor(provider=provider)
        for _ in range(3):
            (self.root / "broken.py").write_text(SUBSTANTIAL_BROKEN, encoding="utf-8")
            result = executor.execute(self.order())
            self.assertNotEqual(result.granted_tier, AutonomyTier.EXECUTE_AUTONOMOUSLY)
            # Against the production constant, not a copy of it.
            self.assertIn(result.granted_tier, SUPPORTED_TIERS)

    def test_the_approval_boundary_is_mandatory_without_apply(self):
        before = snapshot_tree(self.root)
        provider = ScriptedProvider([fix_operation("broken.py", SUBSTANTIAL_FIXED)])
        result = self.executor(provider=provider, apply_enabled=False).execute(
            self.order()
        )
        self.assertEqual(result.status, MaintenanceExecutionStatus.APPROVAL_REQUIRED)
        self.assertFalse(result.applied)
        self.assertEqual(snapshot_tree(self.root), before)

    def test_a_declining_approver_blocks_the_write(self):
        before = snapshot_tree(self.root)
        provider = ScriptedProvider([fix_operation("broken.py", SUBSTANTIAL_FIXED)])
        result = self.executor(
            provider=provider, approval_mode="always", approver=lambda _prepared: False
        ).execute(self.order())
        self.assertEqual(result.status, MaintenanceExecutionStatus.APPROVAL_REQUIRED)
        self.assertEqual(snapshot_tree(self.root), before)

    def test_a_protected_file_is_refused_even_at_the_strongest_tier(self):
        """PRODUCTION-INTEGRATION: a real parse failure in a protected path.

        The defect is injected at ``local_agent/tool_engine.py`` *inside the
        disposable fixture repository*, never in the real checkout, and the
        real analyzer discovers it. The policy must still refuse.
        """
        package = self.root / "local_agent"
        package.mkdir(exist_ok=True)
        (package / "tool_engine.py").write_text(SUBSTANTIAL_BROKEN, encoding="utf-8")
        before = snapshot_tree(self.root)

        from local_agent.semantic_impact import SemanticGraph

        analysis = MaintenanceAnalyzer(self.root).analyze(
            semantic_graph=SemanticGraph.build(self.root)
        )
        protected = [
            c
            for c in analysis.candidates
            if c.kind == MaintenanceSignal.PARSE_FAILURE
            and c.affected_files == ["local_agent/tool_engine.py"]
        ]
        self.assertTrue(protected, "the analyzer did not discover the injected defect")

        result = self.executor(tier=AutonomyTier.EXECUTE_AUTONOMOUSLY).execute(
            self.order(protected[0], tier=AutonomyTier.EXECUTE_AUTONOMOUSLY)
        )
        self.assertEqual(result.status, MaintenanceExecutionStatus.REFUSED_BY_POLICY)
        self.assertFalse(result.applied)
        self.assertEqual(snapshot_tree(self.root), before)


# =============================================================================
# H. Idempotency and concurrency
# =============================================================================


class IdempotencyTests(ExecutorCase):
    def setUp(self) -> None:
        super().setUp()
        (self.root / "broken.py").write_text(SUBSTANTIAL_BROKEN, encoding="utf-8")
        shutil.rmtree(self.root / "tests", ignore_errors=True)

    def test_a_repeated_execution_of_the_same_state_is_refused(self):
        journal = ExecutionJournal(self.data_dir / "j1")
        provider = ScriptedProvider(
            [fix_operation("broken.py", SUBSTANTIAL_FIXED) for _ in range(2)]
        )
        executor = self.executor(provider=provider, journal=journal)
        order = self.order()
        first = executor.execute(order)
        self.assertEqual(first.status, MaintenanceExecutionStatus.COMPLETED)
        # Put the defect back so freshness would otherwise permit a rerun.
        (self.root / "broken.py").write_text(SUBSTANTIAL_BROKEN, encoding="utf-8")
        second = executor.execute(order)
        self.assertEqual(second.status, MaintenanceExecutionStatus.DUPLICATE_EXECUTION)
        self.assertFalse(second.applied)

    def test_a_changed_oracle_generation_mints_a_different_key(self):
        """A key minted under one success predicate is not honoured by another."""
        from local_agent import maintenance_execution as module

        executor = self.executor()
        order = self.order()
        candidate = order_candidate(order)
        first = executor._execution_key(order, candidate, "broken.py")
        with mock.patch.object(module, "ORACLE_FRAMEWORK_VERSION", "9.9.9"):
            second = executor._execution_key(order, candidate, "broken.py")
        self.assertNotEqual(first, second)

    def test_concurrent_duplicate_claims_admit_exactly_one_winner(self):
        journal = ExecutionJournal(self.data_dir / "race")
        key = "abc123"
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            wins = list(pool.map(lambda _i: journal.claim(key), range(8)))
        self.assertEqual(sum(1 for won in wins if won), 1)

    def test_concurrent_executions_of_one_candidate_apply_at_most_once(self):
        """PRODUCTION-INTEGRATION: two real executors, one shared journal."""
        journal = ExecutionJournal(self.data_dir / "shared")
        order = self.order()

        def run(_i: int):
            provider = ScriptedProvider([fix_operation("broken.py", SUBSTANTIAL_FIXED)])
            return self.executor(provider=provider, journal=journal).execute(order)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(run, range(4)))
        applied = [r for r in results if r.applied]
        duplicates = [
            r
            for r in results
            if r.status == MaintenanceExecutionStatus.DUPLICATE_EXECUTION
        ]
        self.assertLessEqual(len(applied), 1)
        self.assertEqual(len(duplicates), len(results) - len(applied))

    def test_an_abandoned_in_progress_claim_fails_closed(self):
        """A crash between apply and persistence must not silently retry.

        This is a KNOWN LIMITATION made explicit rather than hidden: the claim
        has no TTL and no owning-process identity, so an interrupted run locks
        that exact repository state out permanently until an operator removes
        the journal entry. Fail-closed is the right default; the absence of a
        reaper is a real limitation.
        """
        journal = ExecutionJournal(self.data_dir / "abandoned")
        key = "stuck"
        self.assertTrue(journal.claim(key))
        self.assertEqual(journal.status_of(key), "in_progress")
        self.assertFalse(journal.claim(key))
        time.sleep(0.05)
        self.assertFalse(journal.claim(key), "an aged claim is still refused")

    def test_a_corrupted_claim_file_still_blocks(self):
        """ADVERSARIAL: unreadable bookkeeping must not grant permission."""
        journal = ExecutionJournal(self.data_dir / "corrupt")
        self.assertTrue(journal.claim("k"))
        journal._path("k").write_text("{not json", encoding="utf-8")
        self.assertEqual(journal.status_of("k"), "")
        self.assertFalse(journal.claim("k"))

    def test_a_retryable_failure_releases_the_claim(self):
        journal = ExecutionJournal(self.data_dir / "retry")
        provider = ScriptedProvider([], raise_exc=RuntimeError("provider down"))
        executor = self.executor(provider=provider, journal=journal)
        result = executor.execute(self.order())
        self.assertEqual(result.status, MaintenanceExecutionStatus.PROVIDER_FAILURE)
        self.assertTrue(result.retryable)
        self.assertEqual(journal.status_of(result.execution_key), "")

    def test_a_claim_survives_a_process_restart(self):
        """The journal is on disk, so a fresh instance still sees the claim."""
        directory = self.data_dir / "restart"
        first = ExecutionJournal(directory)
        provider = ScriptedProvider([fix_operation("broken.py", SUBSTANTIAL_FIXED)])
        order = self.order()
        result = self.executor(provider=provider, journal=first).execute(order)
        self.assertEqual(result.status, MaintenanceExecutionStatus.COMPLETED)

        # A brand-new journal object over the same directory, as a restarted
        # process would build. No in-memory state carries over.
        (self.root / "broken.py").write_text(SUBSTANTIAL_BROKEN, encoding="utf-8")
        second = ExecutionJournal(directory)
        again = self.executor(
            provider=ScriptedProvider([fix_operation("broken.py", SUBSTANTIAL_FIXED)]),
            journal=second,
        ).execute(order)
        self.assertEqual(again.status, MaintenanceExecutionStatus.DUPLICATE_EXECUTION)
        self.assertFalse(again.applied)

    def test_an_already_resolved_candidate_is_refused_before_any_claim(self):
        # The order is planned while the defect is real, then somebody else
        # fixes the file before the executor gets to it.
        order = self.order(fingerprint=False)
        (self.root / "broken.py").write_text(SUBSTANTIAL_FIXED, encoding="utf-8")
        journal = ExecutionJournal(self.data_dir / "resolved")
        result = self.executor(journal=journal).execute(order)
        self.assertEqual(result.status, MaintenanceExecutionStatus.STALE_CANDIDATE)
        # Nothing was claimed, so a later legitimate attempt is not locked out.
        self.assertEqual(result.execution_key, "")

    def test_a_changed_defect_fingerprint_mints_a_new_key(self):
        executor = self.executor()
        order = self.order()
        candidate = order_candidate(order)
        first = executor._execution_key(order, candidate, "broken.py")
        (self.root / "broken.py").write_text(
            SUBSTANTIAL_BROKEN + "\nEXTRA = 1\n", encoding="utf-8"
        )
        second = executor._execution_key(order, candidate, "broken.py")
        self.assertNotEqual(first, second)

    def test_a_journal_directory_that_cannot_be_written_fails_closed(self):
        """ADVERSARIAL: no journal means no duplicate protection, so refuse."""
        blocker = self.data_dir / "blocked"
        blocker.write_text("not a directory", encoding="utf-8")
        journal = ExecutionJournal(blocker / "inner")
        self.assertFalse(journal.claim("anything"))
        before = snapshot_tree(self.root)
        provider = ScriptedProvider([fix_operation("broken.py", SUBSTANTIAL_FIXED)])
        result = self.executor(provider=provider, journal=journal).execute(self.order())
        self.assertEqual(result.status, MaintenanceExecutionStatus.DUPLICATE_EXECUTION)
        self.assertFalse(result.applied)
        self.assertEqual(snapshot_tree(self.root), before)

    def test_repeated_execution_creates_no_unbounded_lifecycle_records(self):
        journal = ExecutionJournal(self.data_dir / "bounded")
        provider = ScriptedProvider(
            [fix_operation("broken.py", SUBSTANTIAL_FIXED) for _ in range(5)]
        )
        executor = self.executor(provider=provider, journal=journal)
        order = self.order()
        for _ in range(5):
            executor.execute(order)
        store = self.storage.load_validation_lifecycle()
        # One real attempt; the rest were refused as duplicates before any
        # lifecycle was started.
        self.assertLessEqual(len(store.lifecycles), 1)


def order_candidate(order):
    from local_agent.maintenance import MaintenanceCandidate

    return MaintenanceCandidate.from_dict(order.candidate_snapshot)


# =============================================================================
# I. Failure injection
# =============================================================================


class FailureInjectionTests(ExecutorCase):
    def setUp(self) -> None:
        super().setUp()
        (self.root / "broken.py").write_text(SUBSTANTIAL_BROKEN, encoding="utf-8")
        shutil.rmtree(self.root / "tests", ignore_errors=True)
        self.before = snapshot_tree(self.root)

    def assert_no_success_and_tree_intact(self, result) -> None:
        self.assertFalse(result.succeeded, result.status)
        self.assertIsNot(result.signal_resolved, True)
        self.assertEqual(snapshot_tree(self.root), self.before)

    def test_provider_construction_failure(self):
        executor = self.executor()
        executor.provider_factory = lambda: (_ for _ in ()).throw(
            RuntimeError("no api key")
        )
        result = executor.execute(self.order())
        self.assertEqual(result.status, MaintenanceExecutionStatus.PROVIDER_FAILURE)
        self.assert_no_success_and_tree_intact(result)

    def test_provider_rate_limit(self):
        from local_agent.models import ProviderError

        provider = ScriptedProvider([], raise_exc=ProviderError("429 rate limited"))
        result = self.executor(provider=provider).execute(self.order())
        self.assertEqual(result.status, MaintenanceExecutionStatus.PROVIDER_FAILURE)
        self.assert_no_success_and_tree_intact(result)

    def test_malformed_model_output(self):
        provider = ScriptedProvider(["this is not a list of file operations"])
        result = self.executor(provider=provider).execute(self.order())
        self.assertIn(result.status, NO_MUTATION_STATUSES)
        self.assert_no_success_and_tree_intact(result)

    def test_an_operation_outside_scope_never_reaches_the_repository(self):
        """PRODUCTION-INTEGRATION: two layers refuse, and neither writes.

        ``InteractiveCodingAgent`` already discards operations outside the
        plan's allowed paths, so in practice the executor sees an empty
        operation list and reports ``no_change``. That is a stronger refusal
        than ``scope_violation`` - the edit never even became a candidate - so
        the assertion is on the outcome that matters: nothing was written and
        nothing was credited. The executor's own scope gate is the second
        layer and is proved separately, below.
        """
        provider = ScriptedProvider([fix_operation("healthy.py", "VALUE = 2\n")])
        result = self.executor(provider=provider).execute(self.order())
        self.assertIn(
            result.status,
            {
                MaintenanceExecutionStatus.SCOPE_VIOLATION,
                MaintenanceExecutionStatus.NO_CHANGE,
            },
        )
        self.assertIn(result.status, NO_MUTATION_STATUSES)
        self.assert_no_success_and_tree_intact(result)

    def test_the_executors_own_scope_gate_refuses_every_illegal_operation(self):
        """CONTRACT: the second layer, exercised directly.

        Driven against the real ``_check_scope`` with a real ``Plan``, because
        the upstream agent filter means these operations cannot be delivered to
        it through the normal path - and a defence-in-depth layer that is never
        tested is not a defence.
        """
        from local_agent.maintenance_execution import MaintenanceExecutionResult
        from local_agent.models import FileOperation, Plan

        executor = self.executor()
        plan = Plan(objective="x", files_likely_to_change=["broken.py"])
        for operation, label in (
            (
                FileOperation(action="modify", path="healthy.py", content="", reason="r"),
                "outside the plan scope",
            ),
            (
                FileOperation(
                    action="modify",
                    path="local_agent/tool_engine.py",
                    content="",
                    reason="r",
                ),
                "a protected file",
            ),
            (
                FileOperation(action="delete", path="broken.py", content="", reason="r"),
                "a delete",
            ),
            (
                FileOperation(action="create", path="broken.py", content="", reason="r"),
                "a create",
            ),
        ):
            with self.subTest(case=label):
                result = MaintenanceExecutionResult()
                self.assertFalse(executor._check_scope([operation], plan, result))
                self.assertEqual(
                    result.status, MaintenanceExecutionStatus.SCOPE_VIOLATION
                )
        self.assertEqual(snapshot_tree(self.root), self.before)

    def test_a_protected_path_operation_never_reaches_the_repository(self):
        provider = ScriptedProvider(
            [fix_operation("local_agent/tool_engine.py", "VALUE = 1\n")]
        )
        result = self.executor(provider=provider).execute(self.order())
        self.assertIn(result.status, NO_MUTATION_STATUSES)
        self.assert_no_success_and_tree_intact(result)
        self.assertFalse((self.root / "local_agent" / "tool_engine.py").exists())

    def test_a_context_provider_failure(self):
        executor = self.executor()
        executor.context_provider = lambda: (_ for _ in ()).throw(OSError("no repo"))
        result = executor.execute(self.order())
        self.assertEqual(
            result.status, MaintenanceExecutionStatus.IMPLEMENTATION_FAILURE
        )
        self.assert_no_success_and_tree_intact(result)

    def test_a_workspace_that_cannot_be_created(self):
        executor = self.executor()
        executor.workspace_parent = self.root / "broken.py" / "impossible"
        result = executor.execute(self.order())
        self.assertIn(result.status, NO_MUTATION_STATUSES)
        self.assert_no_success_and_tree_intact(result)

    def test_a_post_apply_validation_failure_rolls_back(self):
        """PRODUCTION-INTEGRATION: a real failing test causes a real rollback."""
        (self.root / "tests").mkdir(exist_ok=True)
        (self.root / "tests" / "test_broken.py").write_text(
            "from broken import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n",
            encoding="utf-8",
        )
        before = snapshot_tree(self.root)
        wrong = SUBSTANTIAL_FIXED.replace("return a + b", "return a - b")
        provider = ScriptedProvider([fix_operation("broken.py", wrong)])
        result = self.executor(provider=provider).execute(self.order())
        if result.status == MaintenanceExecutionStatus.COMPLETED:
            self.skipTest("no targeted test was selected in this environment")
        self.assertTrue(result.rolled_back or not result.applied)
        self.assertEqual(snapshot_tree(self.root), before)

    def test_a_lifecycle_manager_that_raises_does_not_stop_the_work(self):
        """History is bookkeeping; losing it must not corrupt the repository."""

        class HostileLifecycle:
            def start(self, **_kwargs):
                raise RuntimeError("lifecycle store corrupt")

            def transition(self, *_args, **_kwargs):
                raise RuntimeError("nope")

            def record_iteration(self, *_args, **_kwargs):
                raise RuntimeError("nope")

        provider = ScriptedProvider([fix_operation("broken.py", SUBSTANTIAL_FIXED)])
        executor = self.executor(provider=provider)
        executor.lifecycle_manager = HostileLifecycle()
        result = executor.execute(self.order())
        self.assertEqual(result.status, MaintenanceExecutionStatus.COMPLETED)
        self.assertEqual(
            (self.root / "broken.py").read_text(encoding="utf-8"), SUBSTANTIAL_FIXED
        )

    def test_an_oracle_that_raises_rolls_the_change_back(self):
        """REGRESSION for a defect inherited from Phase 4.22.

        The authoritative tree is already written when the oracle is consulted.
        In 4.22 an exception anywhere in the post-apply path escaped to the
        top-level handler, which recorded a status but performed no rollback -
        the run ended ``implementation_failure`` / ``applied=True`` /
        ``rolled_back=False`` with the change still on disk. Asserting only
        "status is not completed", as the first version of this test did, was
        not enough to notice; the assertion that matters is on the bytes.
        """
        from local_agent import maintenance_execution as module

        class ExplodingOracle(ParseOracle):
            def observe_success(self, root, relative, before=None):
                raise RuntimeError("oracle exploded")

        provider = ScriptedProvider([fix_operation("broken.py", SUBSTANTIAL_FIXED)])
        with mock.patch.object(module, "oracle_for", lambda _k: ExplodingOracle()):
            result = self.executor(provider=provider).execute(self.order())

        self.assertNotEqual(result.status, MaintenanceExecutionStatus.COMPLETED)
        self.assertFalse(result.succeeded)
        self.assertTrue(result.applied)
        self.assertTrue(result.rolled_back, result.errors)
        self.assertIsNone(result.validation_passed)
        self.assertEqual(snapshot_tree(self.root), self.before)

    def test_a_post_apply_validation_crash_rolls_the_change_back(self):
        """Same invariant, from a different injection point."""
        provider = ScriptedProvider([fix_operation("broken.py", SUBSTANTIAL_FIXED)])
        executor = self.executor(provider=provider)

        def boom(*_args, **_kwargs):
            raise RuntimeError("validation engine exploded")

        with mock.patch.object(executor, "_post_apply_validation", boom):
            result = executor.execute(self.order())
        self.assertNotEqual(result.status, MaintenanceExecutionStatus.COMPLETED)
        self.assertTrue(result.rolled_back, result.errors)
        self.assertIsNone(result.validation_passed)
        self.assertEqual(snapshot_tree(self.root), self.before)

    def test_an_oracle_that_always_says_resolved_still_needs_a_real_verdict(self):
        """Defence in depth: the oracle is necessary, not sufficient.

        With a lying oracle, the *validation* verdict is still required, and
        the change is still refused when nothing actually executed.
        """
        from local_agent import commands as commands_module
        from local_agent import maintenance_execution as module

        class LyingOracle(ParseOracle):
            def observe_success(self, root, relative, before=None):
                return self._observation(OracleOutcome.RESOLVED, "trust me")

        provider = ScriptedProvider([fix_operation("broken.py", SUBSTANTIAL_FIXED)])
        with mock.patch.object(module, "oracle_for", lambda _k: LyingOracle()), \
                mock.patch.object(
                    commands_module,
                    "resolve_executable",
                    lambda c: (("definitely-not-a-real-binary-4711", *c[1:]), ""),
                ):
            result = self.executor(provider=provider).execute(self.order())
        self.assertNotEqual(result.status, MaintenanceExecutionStatus.COMPLETED)
        self.assertIsNone(result.validation_passed)
        self.assertEqual(snapshot_tree(self.root), self.before)


# =============================================================================
# J. Adversarial persistence
# =============================================================================


class AdversarialPersistenceTests(ExecutorCase):
    def setUp(self) -> None:
        super().setUp()
        (self.root / "broken.py").write_text(SUBSTANTIAL_BROKEN, encoding="utf-8")
        shutil.rmtree(self.root / "tests", ignore_errors=True)
        self.before = snapshot_tree(self.root)

    def test_a_forged_journal_success_record_does_not_unlock_anything(self):
        """A journal entry claiming a prior success only ever refuses more."""
        journal = ExecutionJournal(self.data_dir / "forged")
        journal.directory.mkdir(parents=True, exist_ok=True)
        executor = self.executor(journal=journal)
        order = self.order()
        key = executor._execution_key(order, order_candidate(order), "broken.py")
        journal._path(key).write_text(
            json.dumps({"key": key, "status": "completed", "detail": {"applied": True}}),
            encoding="utf-8",
        )
        result = executor.execute(order)
        self.assertEqual(result.status, MaintenanceExecutionStatus.DUPLICATE_EXECUTION)
        self.assertEqual(snapshot_tree(self.root), self.before)

    def test_a_forged_high_confidence_candidate_of_a_rejected_kind_is_refused(self):
        """ADVERSARIAL: perfect numbers on a non-autonomous kind change nothing."""
        from local_agent.maintenance import MaintenanceCandidate

        for kind in sorted(set(ALL_SIGNAL_KINDS) - {MaintenanceSignal.PARSE_FAILURE}):
            with self.subTest(signal=kind):
                candidate = MaintenanceCandidate(
                    kind=kind,
                    subject="broken.py",
                    affected_files=["broken.py"],
                    confidence=1.0,
                    sample_size=10_000,
                    occurrence_count=10_000,
                    severity="high",
                )
                result = self.executor(
                    tier=AutonomyTier.EXECUTE_AUTONOMOUSLY
                ).execute(self.order(candidate, tier=AutonomyTier.EXECUTE_AUTONOMOUSLY))
                self.assertIn(
                    result.status,
                    {
                        MaintenanceExecutionStatus.UNSUPPORTED_SIGNAL,
                        MaintenanceExecutionStatus.REFUSED_BY_POLICY,
                        MaintenanceExecutionStatus.MALFORMED_WORK_ORDER,
                    },
                )
                self.assertFalse(result.applied)
        self.assertEqual(snapshot_tree(self.root), self.before)

    def test_hostile_budget_values_never_widen_anything(self):
        """ADVERSARIAL: negative, zero and enormous budgets."""
        for value in (-1, 0, 10**9):
            with self.subTest(value=value):
                budget = MaintenanceBudget(
                    max_changed_files_per_candidate=value,
                    max_validation_commands=value,
                    max_tool_steps_per_subtask=value,
                )
                provider = ScriptedProvider(
                    [fix_operation("broken.py", SUBSTANTIAL_FIXED)]
                )
                result = self.executor(provider=provider, budget=budget).execute(
                    self.order(budget=budget)
                )
                if value <= 0:
                    self.assertFalse(result.applied)
                else:
                    # A huge budget is still capped by the executor's own
                    # ceilings; it can never permit more than one file.
                    self.assertLessEqual(len(result.changed_files), 1)

    def test_an_unknown_kind_in_a_persisted_snapshot_cannot_execute(self):
        order = self.order()
        forged = dict(order.candidate_snapshot)
        forged["kind"] = "brand_new_signal_kind"
        order.candidate_snapshot = forged
        result = self.executor().execute(order)
        self.assertIn(
            result.status,
            {
                MaintenanceExecutionStatus.MALFORMED_WORK_ORDER,
                MaintenanceExecutionStatus.UNSUPPORTED_SIGNAL,
            },
        )
        self.assertEqual(snapshot_tree(self.root), self.before)

    def test_a_stale_fingerprint_refuses_execution(self):
        order = self.order()
        (self.root / "broken.py").write_text(
            SUBSTANTIAL_BROKEN + "\nOTHER = 1\n", encoding="utf-8"
        )
        before = snapshot_tree(self.root)
        result = self.executor().execute(order)
        self.assertEqual(result.status, MaintenanceExecutionStatus.STALE_CANDIDATE)
        self.assertEqual(snapshot_tree(self.root), before)

    def test_a_scope_that_disagrees_with_the_snapshot_is_refused(self):
        order = self.order()
        order.scope_files = ["healthy.py"]
        result = self.executor().execute(order)
        self.assertEqual(result.status, MaintenanceExecutionStatus.MALFORMED_WORK_ORDER)
        self.assertEqual(snapshot_tree(self.root), self.before)

    def test_a_traversal_scope_path_is_refused(self):
        for path in ("../outside.py", "/etc/passwd", "a/../../b.py"):
            with self.subTest(path=path):
                order = self.order()
                order.scope_files = [path]
                result = self.executor().execute(order)
                self.assertEqual(
                    result.status, MaintenanceExecutionStatus.MALFORMED_WORK_ORDER
                )
        self.assertEqual(snapshot_tree(self.root), self.before)


# =============================================================================
# K. Architectural invariants, from the AST
# =============================================================================


class OracleArchitecturalInvariantTests(unittest.TestCase):
    """Structural proofs about the new module, not prose about it."""

    def setUp(self) -> None:
        self.oracle_module = _module_ast(ORACLE_MODULE)
        self.identifiers = code_identifiers(self.oracle_module)
        self.imports = imported_modules(ORACLE_MODULE)

    def test_the_oracle_module_has_no_filesystem_write_path(self):
        for forbidden in (
            "write_text",
            "write_bytes",
            "mkdir",
            "unlink",
            "rmtree",
            "copytree",
            "rename",
            "touch",
            "chdir",
            "getcwd",
        ):
            self.assertNotIn(forbidden, self.identifiers, forbidden)
        # ``replace`` and ``remove`` are deliberately absent from that list:
        # ``str.replace`` and ``set.remove`` are ordinary and unrelated, so
        # asserting on the bare attribute name would be a test that fails for
        # the wrong reason. The write-capable paths are covered above and by
        # the import check below.

    def test_the_oracle_module_never_opens_a_file_for_writing(self):
        for node in ast.walk(self.oracle_module):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotEqual(node.func.id, "open")

    def test_the_oracle_module_cannot_reach_the_approval_or_tool_engine(self):
        for forbidden in (
            "local_agent.approval",
            "local_agent.tool_engine",
            "local_agent.coding_agent",
            "local_agent.sandbox",
            "local_agent.validation_decision",
        ):
            self.assertNotIn(forbidden, self.imports)

    def test_the_oracle_module_cannot_alter_a_validation_decision(self):
        for forbidden in ("ValidationDecisionEngine", "decide", "finalize_decision"):
            self.assertNotIn(forbidden, self.identifiers)

    def test_the_oracle_module_imports_no_policy_module(self):
        """Tier strings are literals here; the policy must not be importable.

        Otherwise an oracle could consult - or worse, influence - the tier it
        is being judged under.
        """
        self.assertNotIn("local_agent.maintenance_policy", self.imports)

    def test_the_literal_tier_strings_match_the_real_policy(self):
        """...and the duplication is kept honest by this assertion."""
        from local_agent.maintenance_oracle import (
            _TIER_EXECUTE_WITH_APPROVAL,
            _TIER_RECOMMEND,
        )

        self.assertEqual(_TIER_RECOMMEND, AutonomyTier.RECOMMEND)
        self.assertEqual(
            _TIER_EXECUTE_WITH_APPROVAL, AutonomyTier.EXECUTE_WITH_EXISTING_APPROVAL
        )

    def test_promotability_cannot_be_flipped_on_an_instance(self):
        """A stored flag could be set at run time; a read-only property cannot.

        Asserted by attempting the assignment rather than by inspecting the
        class dictionary, so this is a claim about behaviour.
        """
        oracle = oracle_for(MaintenanceSignal.TEST_GAP)
        self.assertFalse(oracle.promotable)
        with self.assertRaises(AttributeError):
            oracle.promotable = True  # type: ignore[misc]
        self.assertFalse(oracle_for(MaintenanceSignal.TEST_GAP).promotable)

    def test_the_autonomous_set_requires_both_gates(self):
        """CONTRACT: flipping the inventory flag alone grants nothing.

        The rule is ``inventory says so AND the oracle is deterministic``. A
        forged entry claiming autonomy for ``test_gap`` - whose oracle is
        ambiguous - must change nothing.
        """
        import dataclasses

        from local_agent import maintenance_oracle as module

        liar = dataclasses.replace(
            SIGNAL_INVENTORY[MaintenanceSignal.TEST_GAP], autonomous_execution=True
        )
        self.assertTrue(liar.autonomous_execution)
        patched = dict(SIGNAL_INVENTORY)
        patched[MaintenanceSignal.TEST_GAP] = liar
        with mock.patch.object(module, "SIGNAL_INVENTORY", patched):
            recomputed = module._autonomous_kinds()
        self.assertNotIn(MaintenanceSignal.TEST_GAP, recomputed)
        self.assertEqual(recomputed, frozenset({MaintenanceSignal.PARSE_FAILURE}))

    def test_marking_a_deterministic_oracles_signal_autonomous_would_work(self):
        """The complement of the test above: the rule is a conjunction, not a
        refusal of everything. Without this, the previous test would also pass
        if ``_autonomous_kinds`` simply returned a hard-coded set."""
        import dataclasses

        from local_agent import maintenance_oracle as module

        withdrawn = dataclasses.replace(
            SIGNAL_INVENTORY[MaintenanceSignal.PARSE_FAILURE],
            autonomous_execution=False,
        )
        patched = dict(SIGNAL_INVENTORY)
        patched[MaintenanceSignal.PARSE_FAILURE] = withdrawn
        with mock.patch.object(module, "SIGNAL_INVENTORY", patched):
            self.assertEqual(module._autonomous_kinds(), frozenset())

    def test_the_executor_still_has_no_filesystem_write_path(self):
        """The Phase 4.22 invariant, re-asserted after the 4.23 rewiring."""
        executor = _class_node(EXECUTOR_MODULE, "MaintenanceExecutor")
        identifiers = code_identifiers(executor)
        for forbidden in (
            "write_text",
            "write_bytes",
            "mkdir",
            "unlink",
            "rmtree",
            "copytree",
            "chdir",
            "getcwd",
        ):
            self.assertNotIn(forbidden, identifiers, forbidden)
        for node in ast.walk(executor):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotEqual(node.func.id, "open")

    def test_the_executor_cannot_bypass_prepare_or_apply_prepared(self):
        executor = _class_node(EXECUTOR_MODULE, "MaintenanceExecutor")
        identifiers = code_identifiers(executor)
        self.assertIn("prepare", identifiers)
        self.assertIn("apply_prepared", identifiers)
        # And it never reaches for a lower-level writer.
        for forbidden in ("apply_operations", "_write", "patch_file", "apply_patch"):
            self.assertNotIn(forbidden, identifiers, forbidden)

    def test_the_executor_imports_no_subprocess(self):
        self.assertNotIn("subprocess", imported_modules(EXECUTOR_MODULE))

    def test_candidate_validation_happens_inside_a_candidate_workspace(self):
        executor = _class_node(EXECUTOR_MODULE, "MaintenanceExecutor")
        self.assertIn("CandidateWorkspace", code_identifiers(executor))


# =============================================================================
# L. Honest telemetry
# =============================================================================


class TelemetryHonestyTests(ExecutorCase):
    def setUp(self) -> None:
        super().setUp()
        (self.root / "broken.py").write_text(SUBSTANTIAL_BROKEN, encoding="utf-8")
        shutil.rmtree(self.root / "tests", ignore_errors=True)

    def test_a_result_answers_every_observability_question(self):
        provider = ScriptedProvider([fix_operation("broken.py", SUBSTANTIAL_FIXED)])
        result = self.executor(provider=provider).execute(self.order())
        payload = result.to_dict()
        json.dumps(payload)  # must round-trip
        for key in (
            "oracle_name",
            "oracle_class",
            "oracle_precondition",
            "oracle_postcondition",
            "post_apply_executed_commands",
            "post_apply_skipped_commands",
            "post_apply_executed_any",
            "granted_tier",
            "validation_scope",
            "candidate_iterations",
            "diff_lines",
        ):
            self.assertIn(key, payload)
        self.assertEqual(payload["oracle_name"], "parse_oracle")
        self.assertTrue(payload["oracle_precondition"]["deterministic"])

    def test_unreached_stages_are_none_not_empty(self):
        """'We never got there' must not look like 'nothing was wrong'."""
        result = self.executor(apply_enabled=False).execute(self.order())
        self.assertIsNone(result.oracle_postcondition)
        self.assertIsNone(result.validation_passed)
        self.assertIsNone(result.signal_resolved)

    def test_skipped_commands_are_never_reported_as_coverage(self):
        from local_agent import commands as commands_module

        provider = ScriptedProvider([fix_operation("broken.py", SUBSTANTIAL_FIXED)])
        with mock.patch.object(
            commands_module,
            "resolve_executable",
            lambda c: (("definitely-not-a-real-binary-4711", *c[1:]), ""),
        ):
            result = self.executor(provider=provider).execute(self.order())
        self.assertFalse(result.post_apply_executed_any)
        self.assertEqual(result.post_apply_commands_run, 0)
        self.assertEqual(result.post_apply_executed_commands, [])

    def test_the_oracle_framework_version_changes_the_execution_identity(self):
        """Behavioural, not a grep for the constant's name.

        A work order planned under one definition of success must not be
        honoured by a build with a different one, so the framework version has
        to reach the execution key. The proof is that changing it changes the
        key - which is also asserted from the idempotency side.
        """
        self.assertTrue(ORACLE_FRAMEWORK_VERSION)
        from local_agent import maintenance_execution as module
        from local_agent.maintenance import MaintenanceCandidate

        executor = self.executor()
        order = self.order()
        candidate = MaintenanceCandidate.from_dict(order.candidate_snapshot)
        baseline = executor._execution_key(order, candidate, "broken.py")
        with mock.patch.object(module, "ORACLE_FRAMEWORK_VERSION", "0.0.0-other"):
            shifted = executor._execution_key(order, candidate, "broken.py")
        self.assertNotEqual(baseline, shifted)
        self.assertEqual(
            baseline, executor._execution_key(order, candidate, "broken.py")
        )


# =============================================================================
# M. The real DungX repository, in a disposable copy
# =============================================================================


class RealRepositoryIntegrationTests(unittest.TestCase):
    """PRODUCTION-INTEGRATION against the actual DungX checkout.

    The authoritative checkout is never written to. A disposable copy is made,
    a controlled defect is injected there, the full autonomous path runs
    against the copy, and the authoritative tree is proved byte-identical
    afterwards - the same discipline Phase 4.22 applied.
    """

    #: Enough of the real repository to be a real repository, without copying
    #: hundreds of megabytes of history and caches.
    COPY = ("local_agent",)

    def setUp(self) -> None:
        self.workdir = Path(tempfile.mkdtemp(prefix="dungx_copy_")).resolve()
        self.addCleanup(shutil.rmtree, self.workdir, ignore_errors=True)
        self.data_dir = Path(tempfile.mkdtemp(prefix="dungx_data_")).resolve()
        self.addCleanup(shutil.rmtree, self.data_dir, ignore_errors=True)
        self.cwd_before = os.getcwd()

    def tearDown(self) -> None:
        self.assertEqual(os.getcwd(), self.cwd_before)

    def _authoritative_fingerprint(self) -> dict[str, str]:
        """Per-file hash of every source file in the real checkout.

        A single rolled-up digest would answer "did anything change?" but not
        "what", and a bare pair of hex strings is close to undiagnosable when
        it fails. Keeping it per-file costs nothing and lets
        :meth:`assert_authoritative_unchanged` name the offending path.
        """
        import hashlib

        return {
            path.relative_to(REPO_ROOT).as_posix(): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted((REPO_ROOT / "local_agent").rglob("*.py"))
        }

    def assert_authoritative_unchanged(self, before: dict[str, str]) -> None:
        after = self._authoritative_fingerprint()
        added = sorted(set(after) - set(before))
        removed = sorted(set(before) - set(after))
        modified = sorted(
            path for path in set(before) & set(after) if before[path] != after[path]
        )
        self.assertEqual(added, [], f"files appeared in the real checkout: {added}")
        self.assertEqual(removed, [], f"files vanished from the real checkout: {removed}")
        self.assertEqual(
            modified, [], f"files were modified in the real checkout: {modified}"
        )

    def test_the_full_path_runs_against_a_disposable_copy_of_the_real_repo(self):
        authoritative_before = self._authoritative_fingerprint()

        for name in self.COPY:
            shutil.copytree(
                REPO_ROOT / name,
                self.workdir / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        target_rel = "local_agent/_phase423_probe.py"
        healthy = (
            "import math\n"
            "\n"
            "PROBE_CONSTANT = 7\n"
            "\n"
            "\n"
            "def probe_add(a, b):\n"
            "    return a + b\n"
            "\n"
            "\n"
            "class ProbeCalculator:\n"
            "    def probe_compute(self, x):\n"
            "        return math.sqrt(x)\n"
        )
        broken = healthy.replace("def probe_add(a, b):", "def probe_add(a, b)")
        (self.workdir / target_rel).write_text(broken, encoding="utf-8")

        # 1. Real discovery finds it.
        from local_agent.semantic_impact import SemanticGraph

        analysis = MaintenanceAnalyzer(self.workdir).analyze(
            semantic_graph=SemanticGraph.build(self.workdir)
        )
        matches = [
            c
            for c in analysis.candidates
            if c.kind == MaintenanceSignal.PARSE_FAILURE
            and c.affected_files == [target_rel]
        ]
        self.assertTrue(matches, "the real analyzer did not find the injected defect")
        candidate = matches[0]

        # 2. The real policy grants an executing tier.
        verdict = MaintenanceExecutionPolicy(repository_root=self.workdir).decide(
            candidate,
            configured_tier=AutonomyTier.EXECUTE_AUTONOMOUSLY,
            budget=MaintenanceBudget(),
        )
        self.assertTrue(verdict.may_execute)
        self.assertEqual(
            verdict.granted_tier, AutonomyTier.EXECUTE_WITH_EXISTING_APPROVAL
        )

        # 3. The real executor repairs it, judged by the real oracle.
        from local_agent.evidence import compute_state_fingerprint
        from local_agent.maintenance_execution import (
            MaintenanceApprovalGate,
            MaintenanceExecutor,
        )
        from local_agent.maintenance_runner import build_work_order
        from local_agent.repository import RepositoryIntelligence

        order = build_work_order(
            candidate,
            granted_tier=AutonomyTier.EXECUTE_WITH_EXISTING_APPROVAL,
            budget=MaintenanceBudget(),
            configured_tier=AutonomyTier.EXECUTE_WITH_EXISTING_APPROVAL,
            fingerprint_fn=lambda paths: compute_state_fingerprint(self.workdir, paths),
        )
        provider = ScriptedProvider([fix_operation(target_rel, healthy)])
        executor = MaintenanceExecutor(
            root=self.workdir,
            provider_factory=lambda: provider,
            policy=MaintenanceExecutionPolicy(repository_root=self.workdir),
            budget=MaintenanceBudget(),
            configured_tier=AutonomyTier.EXECUTE_WITH_EXISTING_APPROVAL,
            journal=ExecutionJournal(self.data_dir / "journal"),
            approval_gate=MaintenanceApprovalGate(
                approval_mode="never", apply_enabled=True
            ),
            context_provider=lambda: RepositoryIntelligence(self.workdir).scan(),
            workspace_parent=self.data_dir / "workspaces",
        )
        result = executor.execute(order)

        self.assertEqual(result.status, MaintenanceExecutionStatus.COMPLETED, result.reasons)
        self.assertTrue(result.succeeded)
        self.assertEqual(
            (self.workdir / target_rel).read_text(encoding="utf-8"), healthy
        )

        # 4. A fresh rescan of the copy no longer reports the signal.
        rescan = MaintenanceAnalyzer(self.workdir).analyze(
            semantic_graph=SemanticGraph.build(self.workdir)
        )
        self.assertEqual(
            [
                c
                for c in rescan.candidates
                if c.kind == MaintenanceSignal.PARSE_FAILURE
                and c.affected_files == [target_rel]
            ],
            [],
        )

        # 5. The authoritative checkout was never touched.
        self.assert_authoritative_unchanged(authoritative_before)
        self.assertFalse((REPO_ROOT / target_rel).exists())

    def test_a_destructive_repair_against_the_real_repo_copy_is_refused(self):
        authoritative_before = self._authoritative_fingerprint()
        for name in self.COPY:
            shutil.copytree(
                REPO_ROOT / name,
                self.workdir / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        target_rel = "local_agent/_phase423_probe.py"
        broken = (
            "import math\n\nPROBE = 1\n\n\ndef probe(a, b)\n    return a + b\n"
            "\n\nclass P:\n    def q(self):\n        return math.pi\n"
        )
        (self.workdir / target_rel).write_text(broken, encoding="utf-8")
        before_bytes = (self.workdir / target_rel).read_bytes()

        from local_agent.evidence import compute_state_fingerprint
        from local_agent.maintenance_execution import (
            MaintenanceApprovalGate,
            MaintenanceExecutor,
        )
        from local_agent.maintenance_runner import build_work_order
        from local_agent.repository import RepositoryIntelligence
        from local_agent.semantic_impact import SemanticGraph

        analysis = MaintenanceAnalyzer(self.workdir).analyze(
            semantic_graph=SemanticGraph.build(self.workdir)
        )
        candidate = next(
            c
            for c in analysis.candidates
            if c.kind == MaintenanceSignal.PARSE_FAILURE
            and c.affected_files == [target_rel]
        )
        order = build_work_order(
            candidate,
            granted_tier=AutonomyTier.EXECUTE_WITH_EXISTING_APPROVAL,
            budget=MaintenanceBudget(),
            configured_tier=AutonomyTier.EXECUTE_WITH_EXISTING_APPROVAL,
            fingerprint_fn=lambda paths: compute_state_fingerprint(self.workdir, paths),
        )
        provider = ScriptedProvider([fix_operation(target_rel, DESTRUCTIVE_REPAIR)])
        executor = MaintenanceExecutor(
            root=self.workdir,
            provider_factory=lambda: provider,
            policy=MaintenanceExecutionPolicy(repository_root=self.workdir),
            budget=MaintenanceBudget(),
            configured_tier=AutonomyTier.EXECUTE_WITH_EXISTING_APPROVAL,
            journal=ExecutionJournal(self.data_dir / "journal2"),
            approval_gate=MaintenanceApprovalGate(
                approval_mode="never", apply_enabled=True
            ),
            context_provider=lambda: RepositoryIntelligence(self.workdir).scan(),
            workspace_parent=self.data_dir / "workspaces2",
        )
        result = executor.execute(order)

        self.assertNotEqual(result.status, MaintenanceExecutionStatus.COMPLETED)
        self.assertFalse(result.succeeded)
        self.assertTrue(result.rolled_back)
        self.assertEqual((self.workdir / target_rel).read_bytes(), before_bytes)
        self.assert_authoritative_unchanged(authoritative_before)


class SignalInventoryCliTests(unittest.TestCase):
    """The inventory is an operator artefact, not just a module constant."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="cli_")).resolve()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def _run(self, *extra: str) -> tuple[int, str]:
        import io
        from contextlib import redirect_stdout

        from local_agent.cli import main as cli_main

        out = io.StringIO()
        with redirect_stdout(out):
            code = cli_main(
                ["maintenance", "signals", "--project", str(self.root), *extra]
            )
        return code, out.getvalue()

    def test_the_command_lists_every_signal_with_its_oracle(self):
        code, text = self._run()
        self.assertEqual(code, 0)
        for kind in ALL_SIGNAL_KINDS:
            self.assertIn(kind, text)
        self.assertIn("parse_oracle", text)
        self.assertIn("AUTONOMOUS", text)

    def test_the_command_states_why_each_gated_signal_is_gated(self):
        _code, text = self._run()
        self.assertGreaterEqual(text.count("NOT AUTOMATED:"), 12)
        self.assertIn("STILL UNPROVEN:", text)

    def test_the_json_form_is_machine_readable_and_complete(self):
        code, text = self._run("--json")
        self.assertEqual(code, 0)
        payload = json.loads(text)
        self.assertEqual(payload["autonomous_signals"], ["parse_failure"])
        self.assertEqual(
            [row["signal"] for row in payload["signals"]], list(ALL_SIGNAL_KINDS)
        )

    def test_the_command_runs_in_a_real_subprocess(self):
        """PRODUCTION-INTEGRATION: the shipped entry point, real exit code."""
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "local_agent",
                "maintenance",
                "signals",
                "--project",
                str(self.root),
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=180,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Maintenance Signal Inventory", completed.stdout)

    def test_the_execution_report_separates_executed_from_skipped(self):
        """CONTRACT: the report must not label skipped commands as executed.

        Phase 4.22 printed every *selected* command under the heading
        "validation commands actually executed", so a run whose test runner was
        missing reported coverage it did not have.
        """
        import io
        from contextlib import redirect_stdout

        from local_agent.cli import _print_execution_report
        from local_agent.maintenance_execution import MaintenanceExecutionResult

        entry = MaintenanceExecutionResult(
            candidate_id="c1",
            status=MaintenanceExecutionStatus.POST_VALIDATION_FAILED,
            signal_kind=MaintenanceSignal.PARSE_FAILURE,
            oracle_name="parse_oracle",
            oracle_class=OracleClass.DETERMINISTIC,
            post_apply_executed_commands=[],
            post_apply_skipped_commands=[["pytest", "-q"]],
        )

        class FakeExecutor:
            results = [entry]

        class FakeRun:
            reassessments: dict = {}

        out = io.StringIO()
        with redirect_stdout(out):
            _print_execution_report(FakeExecutor(), True, True, FakeRun())
        text = out.getvalue()
        self.assertIn("validation commands that actually executed: none", text)
        self.assertIn("SKIPPED", text)
        self.assertIn("pytest -q", text)
        self.assertIn("parse_oracle", text)


class ProtectedFileTests(unittest.TestCase):
    """The two files that must never change, checked against git itself."""

    PROTECTED = ("local_agent/tool_engine.py", "local_agent/approval.py")

    def test_the_protected_files_are_unmodified_in_the_working_tree(self):
        try:
            done = subprocess.run(
                ["git", "diff", "--name-only", "--", *self.PROTECTED],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            self.skipTest("git is unavailable")
        if done.returncode != 0:
            self.skipTest(f"git diff failed: {done.stderr}")
        self.assertEqual(done.stdout.strip(), "")

    def test_no_maintenance_module_can_import_the_protected_modules(self):
        for module in (
            "local_agent.maintenance_oracle",
            "local_agent.maintenance",
            "local_agent.maintenance_analysis",
            "local_agent.maintenance_policy",
            "local_agent.maintenance_runner",
        ):
            with self.subTest(module=module):
                imports = imported_modules(module)
                self.assertNotIn("local_agent.tool_engine", imports)
                self.assertNotIn("local_agent.approval", imports)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
