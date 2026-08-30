"""Phase 4.20 Part F: attribute-receiver and dynamic-import resolution.

Phase 4.19 recorded two named blind spots in the dependency analyzer and
deferred them, noting they were only survivable because the scope policy
escalates the affected cases above TARGETED:

1. an attribute access ``obj.method()`` could never say what ``obj`` was, so
   the association rested on the attribute *name* alone;
2. ``importlib.import_module('.sub', package='pkg')`` was declined outright.

This file covers the narrowed versions of both. The organising principle
throughout: every new resolution rule needs a positive case (it resolves what
it claims to), a negative case (it declines what it cannot know), and an
adversarial case (something constructed to look resolvable that must still be
declined). The safety asymmetry is asserted directly in
``ScopeEscalationStillHoldsCase``: an improvement here must never turn a case
that previously escalated into one left at TARGETED without genuinely
resolving it.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from local_agent.dependency_resolution import (
    ATTRIBUTE_RECEIVER_RESOLVED,
    ATTRIBUTE_RESOLUTION,
    ALL_EVIDENCE_TYPES,
    ALL_RECEIVER_FAILURES,
    CONFIDENCE_BY_EVIDENCE_TYPE,
    DYNAMIC_IMPORT_RESOLVED,
    RECEIVER_AMBIGUOUS,
    RECEIVER_ATTRIBUTE_NOT_CHANGED,
    RECEIVER_LOCAL,
    RECEIVER_TYPE_UNRESOLVED,
    RECEIVER_UNBOUND,
    confidence_for,
    resolve_attribute_receiver,
)
from local_agent.indexing.ast_python_indexer import AstPythonIndexer
from local_agent.semantic_impact import (
    SCOPE_TARGETED,
    SemanticChangeImpactAnalyzer,
)

CORE_BEFORE = (
    "class Widget:\n"
    "    def render(self):\n"
    "        return 'old'\n"
    "\n"
    "\n"
    "class Unrelated:\n"
    "    def render(self):\n"
    "        return 'other'\n"
)
CORE_AFTER = (
    "class Widget:\n"
    "    def render(self):\n"
    "        return 'new'\n"
    "\n"
    "\n"
    "class Unrelated:\n"
    "    def render(self):\n"
    "        return 'other'\n"
)


def build_repo(test_source: str, *, core: str = CORE_AFTER) -> str:
    """A minimal two-package repo whose only test file is ``test_source``."""
    root = tempfile.mkdtemp(prefix="dungx_attr_")
    (Path(root) / "pkg").mkdir()
    (Path(root) / "tests").mkdir()
    (Path(root) / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (Path(root) / "tests" / "__init__.py").write_text("", encoding="utf-8")
    (Path(root) / "pkg" / "core.py").write_text(core, encoding="utf-8")
    (Path(root) / "tests" / "test_core.py").write_text(test_source, encoding="utf-8")
    return root


class IndexerAttributeFactsCase(unittest.TestCase):
    """What the indexer now records about attribute receivers."""

    def setUp(self):
        self.indexer = AstPythonIndexer()

    def test_plain_name_receiver_is_recorded(self):
        facts = self.indexer.analyze("obj.render()\n")
        self.assertIn(("obj", "render"), facts.attribute_accesses)

    def test_constructor_receiver_is_recorded_as_the_class(self):
        facts = self.indexer.analyze("Widget().render()\n")
        self.assertIn(("Widget", "render"), facts.attribute_accesses)

    def test_complex_receiver_is_not_recorded(self):
        # ``things[0].render()`` - the receiver is a subscript; recording
        # ``things`` would be wrong and recording nothing is correct.
        facts = self.indexer.analyze("things[0].render()\n")
        self.assertEqual(
            [pair for pair in facts.attribute_accesses if pair[1] == "render"], []
        )

    def test_attribute_chain_receiver_is_not_recorded(self):
        facts = self.indexer.analyze("a.b.render()\n")
        self.assertNotIn(("b", "render"), facts.attribute_accesses)
        self.assertNotIn(("a", "render"), facts.attribute_accesses)

    def test_attribute_name_evidence_is_still_collected(self):
        """The weaker pre-existing signal must survive unchanged."""
        facts = self.indexer.analyze("things[0].render()\n")
        self.assertIn("render", facts.attribute_references)

    def test_constructor_assignment_binds_the_type(self):
        facts = self.indexer.analyze("w = Widget()\nw.render()\n")
        self.assertEqual(facts.local_type_bindings.get("w"), "Widget")

    def test_module_qualified_constructor_binds_the_class_name(self):
        facts = self.indexer.analyze("w = mod.Widget()\n")
        self.assertEqual(facts.local_type_bindings.get("w"), "Widget")

    def test_annotated_assignment_binds_the_type(self):
        facts = self.indexer.analyze("w: Widget = make()\n")
        self.assertEqual(facts.local_type_bindings.get("w"), "Widget")

    def test_bare_annotation_binds_the_type(self):
        facts = self.indexer.analyze("w: Widget\n")
        self.assertEqual(facts.local_type_bindings.get("w"), "Widget")

    def test_parameter_annotation_binds_the_type(self):
        facts = self.indexer.analyze("def f(w: Widget):\n    return w\n")
        self.assertEqual(facts.local_type_bindings.get("w"), "Widget")

    def test_subscripted_annotation_does_not_bind(self):
        # ``ws: list[Widget]`` - ``ws`` is a list, not a Widget.
        facts = self.indexer.analyze("ws: list[Widget] = []\n")
        self.assertNotIn("ws", facts.local_type_bindings)

    def test_factory_call_result_does_not_bind_to_the_factory(self):
        # ``w = make_widget()`` binds ``w`` to ``make_widget``, which is a
        # function, not a class. The receiver resolver refuses it later because
        # the name does not resolve to a changed *class*; what matters here is
        # that nothing pretends ``w`` is a ``Widget``.
        facts = self.indexer.analyze("w = make_widget()\n")
        self.assertNotEqual(facts.local_type_bindings.get("w"), "Widget")

    def test_conflicting_rebind_is_ambiguous_not_last_write_wins(self):
        facts = self.indexer.analyze("w = Widget()\nw = Unrelated()\n")
        self.assertIn("w", facts.ambiguous_bindings)
        self.assertNotIn("w", facts.local_type_bindings)

    def test_rebind_to_an_untypeable_value_makes_the_name_ambiguous(self):
        """The adversarial case: a constructor followed by an opaque rebind.

        Without recording the second assignment at all, ``w`` would still look
        unambiguously like a ``Widget`` even though it demonstrably is not by
        the time the attribute is accessed.
        """
        facts = self.indexer.analyze("w = Widget()\nw = compute()[0]\nw.render()\n")
        self.assertIn("w", facts.ambiguous_bindings)
        self.assertNotIn("w", facts.local_type_bindings)

    def test_class_bases_are_recorded(self):
        facts = self.indexer.analyze("class C(Widget):\n    pass\n")
        self.assertEqual(facts.class_bases.get("C"), ("Widget",))

    def test_locally_defined_names_are_recorded(self):
        facts = self.indexer.analyze("class Widget:\n    def render(self):\n        return 1\n")
        self.assertIn("Widget", facts.locally_defined_names)
        self.assertIn("render", facts.locally_defined_names)

    def test_parse_failure_yields_no_receiver_facts(self):
        facts = self.indexer.analyze("def broken(:\n")
        self.assertTrue(facts.parse_error)
        self.assertEqual(facts.attribute_accesses, frozenset())
        self.assertEqual(facts.local_type_bindings, {})


class ResolveAttributeReceiverCase(unittest.TestCase):
    """The pure resolver, exercised directly with each refusal reason."""

    def resolve(self, **overrides):
        kwargs = dict(
            source_file="tests/test_core.py",
            receiver="w",
            attribute="render",
            local_type_bindings={"w": "Widget"},
            ambiguous_bindings=frozenset(),
            class_bases={},
            locally_defined_names=frozenset(),
            imported_symbol_origins={
                "tests/test_core.py": {"Widget": ("pkg/core.py", "Widget")}
            },
            changed_files=frozenset({"pkg/core.py"}),
            changed_symbol_names=frozenset({"render", "Widget"}),
            symbols_by_file={"pkg/core.py": frozenset({"Widget", "render"})},
        )
        kwargs.update(overrides)
        return resolve_attribute_receiver(**kwargs)

    def test_positive_resolution(self):
        outcome = self.resolve()
        self.assertTrue(outcome.resolved)
        self.assertEqual(outcome.evidence.evidence_type, ATTRIBUTE_RECEIVER_RESOLVED)
        self.assertEqual(outcome.evidence.target_file, "pkg/core.py")
        self.assertEqual(outcome.evidence.target_symbol, "render")
        self.assertEqual(outcome.reason, "")

    def test_resolution_carries_provenance_and_notes(self):
        evidence = self.resolve().evidence
        self.assertIn("w.render", evidence.provenance)
        self.assertIn("Widget", evidence.provenance)
        self.assertTrue(evidence.resolution_notes)

    def test_resolution_carries_the_table_confidence(self):
        evidence = self.resolve().evidence
        self.assertEqual(
            evidence.confidence, CONFIDENCE_BY_EVIDENCE_TYPE[ATTRIBUTE_RECEIVER_RESOLVED]
        )

    def test_unbound_receiver_is_declined(self):
        outcome = self.resolve(local_type_bindings={})
        self.assertFalse(outcome.resolved)
        self.assertEqual(outcome.reason, RECEIVER_UNBOUND)

    def test_ambiguous_receiver_is_declined_not_guessed(self):
        outcome = self.resolve(ambiguous_bindings=frozenset({"w"}))
        self.assertFalse(outcome.resolved)
        self.assertEqual(outcome.reason, RECEIVER_AMBIGUOUS)

    def test_locally_defined_type_is_a_resolved_negative(self):
        outcome = self.resolve(locally_defined_names=frozenset({"Widget"}))
        self.assertFalse(outcome.resolved)
        self.assertEqual(outcome.reason, RECEIVER_LOCAL)

    def test_type_not_traceable_to_a_changed_file_is_declined(self):
        outcome = self.resolve(imported_symbol_origins={})
        self.assertFalse(outcome.resolved)
        self.assertEqual(outcome.reason, RECEIVER_TYPE_UNRESOLVED)

    def test_type_from_an_unchanged_file_is_declined(self):
        outcome = self.resolve(
            imported_symbol_origins={
                "tests/test_core.py": {"Widget": ("pkg/other.py", "Widget")}
            }
        )
        self.assertFalse(outcome.resolved)
        self.assertEqual(outcome.reason, RECEIVER_TYPE_UNRESOLVED)

    def test_attribute_that_did_not_change_is_declined(self):
        outcome = self.resolve(changed_symbol_names=frozenset({"Widget"}))
        self.assertFalse(outcome.resolved)
        self.assertEqual(outcome.reason, RECEIVER_ATTRIBUTE_NOT_CHANGED)

    def test_adversarial_attribute_absent_from_the_changed_file_is_declined(self):
        """The class is right, the attribute changed *somewhere*, but not here.

        Without the ``symbols_by_file`` cross-check this would resolve, and the
        resulting edge would confidently point at a file that has no such
        method - a false confident edge, which is the exact failure mode this
        rule exists to prevent.
        """
        outcome = self.resolve(symbols_by_file={"pkg/core.py": frozenset({"Widget"})})
        self.assertFalse(outcome.resolved)
        self.assertEqual(outcome.reason, RECEIVER_TYPE_UNRESOLVED)

    def test_self_receiver_resolves_through_a_single_base_class(self):
        outcome = self.resolve(
            receiver="self",
            local_type_bindings={},
            class_bases={"C": ("Widget",)},
        )
        self.assertTrue(outcome.resolved)
        self.assertIn("inherits", outcome.evidence.provenance)

    def test_self_receiver_with_multiple_candidate_bases_is_declined(self):
        outcome = self.resolve(
            receiver="self",
            local_type_bindings={},
            class_bases={"C": ("Widget",), "D": ("Other",)},
        )
        self.assertFalse(outcome.resolved)
        self.assertEqual(outcome.reason, RECEIVER_AMBIGUOUS)

    def test_self_receiver_with_no_bases_is_declined(self):
        outcome = self.resolve(receiver="self", local_type_bindings={}, class_bases={})
        self.assertFalse(outcome.resolved)
        self.assertEqual(outcome.reason, RECEIVER_UNBOUND)

    def test_constructor_receiver_needs_no_binding(self):
        outcome = self.resolve(receiver="Widget", local_type_bindings={})
        self.assertTrue(outcome.resolved)

    def test_every_refusal_reason_is_in_the_declared_vocabulary(self):
        for reason in (
            RECEIVER_UNBOUND,
            RECEIVER_AMBIGUOUS,
            RECEIVER_LOCAL,
            RECEIVER_TYPE_UNRESOLVED,
            RECEIVER_ATTRIBUTE_NOT_CHANGED,
        ):
            self.assertIn(reason, ALL_RECEIVER_FAILURES)

    def test_resolution_is_deterministic(self):
        first = self.resolve().evidence.to_dict()
        second = self.resolve().evidence.to_dict()
        self.assertEqual(first, second)


class EvidenceVocabularyCase(unittest.TestCase):
    def test_new_evidence_type_is_in_the_vocabulary(self):
        self.assertIn(ATTRIBUTE_RECEIVER_RESOLVED, ALL_EVIDENCE_TYPES)

    def test_receiver_resolution_outranks_bare_attribute_matching(self):
        self.assertGreater(
            confidence_for(ATTRIBUTE_RECEIVER_RESOLVED), confidence_for(ATTRIBUTE_RESOLUTION)
        )

    def test_receiver_resolution_does_not_outrank_a_direct_symbol_match(self):
        self.assertLess(
            confidence_for(ATTRIBUTE_RECEIVER_RESOLVED), confidence_for("direct_symbol_match")
        )

    def test_unknown_evidence_type_still_fails_closed(self):
        self.assertEqual(confidence_for("something_from_the_future"), 0.0)


class EndToEndReceiverEvidenceCase(unittest.TestCase):
    """The rule as it actually behaves through the full analyzer."""

    def analyse(self, source: str, *, core: str = CORE_AFTER):
        root = build_repo(source, core=core)
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        report = SemanticChangeImpactAnalyzer(root).analyze(
            ["pkg/core.py"], base_contents={"pkg/core.py": CORE_BEFORE}
        )
        return report

    def evidence_types(self, report) -> set[str]:
        return {
            evidence.evidence_type
            for target in report.validation_targets
            for evidence in target.dependency_evidence
        }

    def test_constructor_bound_receiver_produces_resolved_evidence(self):
        report = self.analyse(
            "from pkg.core import Widget\n\n\ndef test_t():\n"
            "    w = Widget()\n    assert w.render() == 'new'\n"
        )
        self.assertIn(ATTRIBUTE_RECEIVER_RESOLVED, self.evidence_types(report))

    def test_weaker_attribute_evidence_is_still_emitted_alongside(self):
        """Additive, never substitutive: removing the weak edge could change a
        tier, so the improvement deliberately leaves it in place."""
        report = self.analyse(
            "from pkg.core import Widget\n\n\ndef test_t():\n"
            "    w = Widget()\n    assert w.render() == 'new'\n"
        )
        types = self.evidence_types(report)
        self.assertIn(ATTRIBUTE_RESOLUTION, types)
        self.assertIn(ATTRIBUTE_RECEIVER_RESOLVED, types)

    def test_annotation_bound_receiver_produces_resolved_evidence(self):
        report = self.analyse(
            "from pkg.core import Widget\n\n\ndef helper(w: Widget):\n"
            "    return w.render()\n\n\ndef test_t():\n"
            "    assert helper(Widget()) == 'new'\n"
        )
        self.assertIn(ATTRIBUTE_RECEIVER_RESOLVED, self.evidence_types(report))

    def test_locally_defined_class_produces_no_resolved_receiver_evidence(self):
        """The Phase 4.19 ``D_attribute`` shape: a same-named method on a class
        defined right there in the test. Resolving this would be a false
        positive, and it must stay unresolved."""
        report = self.analyse(
            "def test_t():\n"
            "    class O:\n"
            "        def render(self):\n"
            "            return 2\n"
            "    assert O().render() == 2\n"
        )
        self.assertNotIn(ATTRIBUTE_RECEIVER_RESOLVED, self.evidence_types(report))

    def test_unbound_receiver_produces_no_resolved_receiver_evidence(self):
        """The Phase 4.19 ``J_unresolved_attribute_only`` shape."""
        report = self.analyse(
            "def test_t(obj=None):\n"
            "    assert obj is None or obj.render() == 2\n"
        )
        self.assertNotIn(ATTRIBUTE_RECEIVER_RESOLVED, self.evidence_types(report))

    def test_ambiguous_receiver_produces_no_resolved_receiver_evidence(self):
        report = self.analyse(
            "from pkg.core import Widget, Unrelated\n\n\ndef test_t():\n"
            "    w = Widget()\n    w = Unrelated()\n    assert w.render()\n"
        )
        self.assertNotIn(ATTRIBUTE_RECEIVER_RESOLVED, self.evidence_types(report))

    def test_analysis_is_deterministic_across_runs(self):
        source = (
            "from pkg.core import Widget\n\n\ndef test_t():\n"
            "    w = Widget()\n    assert w.render() == 'new'\n"
        )
        first = self.analyse(source)
        second = self.analyse(source)
        self.assertEqual(
            sorted(self.evidence_types(first)), sorted(self.evidence_types(second))
        )

    def test_analysis_does_not_mutate_the_analysed_tree(self):
        root = build_repo(
            "from pkg.core import Widget\n\n\ndef test_t():\n"
            "    w = Widget()\n    assert w.render() == 'new'\n"
        )
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        before = {
            path: (Path(root) / path).read_bytes()
            for path in ("pkg/core.py", "tests/test_core.py", "pkg/__init__.py")
        }
        listing_before = sorted(
            str(Path(dirpath).relative_to(root) / name)
            for dirpath, _dirs, names in os.walk(root)
            for name in names
        )
        SemanticChangeImpactAnalyzer(root).analyze(
            ["pkg/core.py"], base_contents={"pkg/core.py": CORE_BEFORE}
        )
        for path, content in before.items():
            self.assertEqual((Path(root) / path).read_bytes(), content, path)
        listing_after = sorted(
            str(Path(dirpath).relative_to(root) / name)
            for dirpath, _dirs, names in os.walk(root)
            for name in names
        )
        self.assertEqual(listing_before, listing_after)


class DynamicImportResolutionCase(unittest.TestCase):
    """Package-relative dynamic imports: positive, negative and adversarial."""

    def setUp(self):
        self.indexer = AstPythonIndexer()

    def dynamic_modules(self, source: str) -> list[str]:
        facts = self.indexer.analyze(source)
        return [record.module for record in facts.imports if record.dynamic]

    def test_single_dot_with_literal_package_resolves(self):
        self.assertEqual(
            self.dynamic_modules(
                "import importlib\nimportlib.import_module('.sub', package='pkg')\n"
            ),
            ["pkg.sub"],
        )

    def test_dotted_package_resolves(self):
        self.assertEqual(
            self.dynamic_modules(
                "import importlib\nimportlib.import_module('.leaf', package='a.b.c')\n"
            ),
            ["a.b.c.leaf"],
        )

    def test_double_dot_walks_up_one_level(self):
        self.assertEqual(
            self.dynamic_modules(
                "import importlib\nimportlib.import_module('..sib', package='a.b')\n"
            ),
            ["a.sib"],
        )

    def test_bare_dot_resolves_to_the_package_itself(self):
        self.assertEqual(
            self.dynamic_modules(
                "import importlib\nimportlib.import_module('.', package='a.b')\n"
            ),
            ["a.b"],
        )

    def test_resolution_clears_the_degradation_flag(self):
        facts = self.indexer.analyze(
            "import importlib\nimportlib.import_module('.sub', package='pkg')\n"
        )
        self.assertFalse(facts.has_dynamic_imports)

    def test_dunder_import_form_also_resolves(self):
        self.assertEqual(
            self.dynamic_modules("__import__('.sub', package='pkg')\n"), ["pkg.sub"]
        )

    # -- negatives ---------------------------------------------------------

    def test_missing_package_keyword_is_declined(self):
        source = "import importlib\nimportlib.import_module('.sub')\n"
        self.assertEqual(self.dynamic_modules(source), [])
        self.assertTrue(self.indexer.analyze(source).has_dynamic_imports)

    def test_non_literal_package_is_declined(self):
        source = "import importlib\np = 'a'\nimportlib.import_module('.sub', package=p)\n"
        self.assertEqual(self.dynamic_modules(source), [])
        self.assertTrue(self.indexer.analyze(source).has_dynamic_imports)

    def test_fstring_package_is_declined(self):
        source = "import importlib\nimportlib.import_module('.sub', package=f'a{x}')\n"
        self.assertEqual(self.dynamic_modules(source), [])
        self.assertTrue(self.indexer.analyze(source).has_dynamic_imports)

    def test_non_literal_module_argument_is_declined(self):
        source = "import importlib\nn = '.sub'\nimportlib.import_module(n, package='pkg')\n"
        self.assertEqual(self.dynamic_modules(source), [])
        self.assertTrue(self.indexer.analyze(source).has_dynamic_imports)

    # -- adversarial -------------------------------------------------------

    def test_escaping_the_package_root_is_declined(self):
        source = "import importlib\nimportlib.import_module('...sub', package='pkg')\n"
        self.assertEqual(self.dynamic_modules(source), [])
        self.assertTrue(self.indexer.analyze(source).has_dynamic_imports)

    def test_non_identifier_package_is_declined(self):
        source = "import importlib\nimportlib.import_module('.sub', package='not a pkg')\n"
        self.assertEqual(self.dynamic_modules(source), [])
        self.assertTrue(self.indexer.analyze(source).has_dynamic_imports)

    def test_non_identifier_remainder_is_declined(self):
        source = "import importlib\nimportlib.import_module('.a-b', package='pkg')\n"
        self.assertEqual(self.dynamic_modules(source), [])
        self.assertTrue(self.indexer.analyze(source).has_dynamic_imports)

    def test_empty_package_string_is_declined(self):
        source = "import importlib\nimportlib.import_module('.sub', package='')\n"
        self.assertEqual(self.dynamic_modules(source), [])
        self.assertTrue(self.indexer.analyze(source).has_dynamic_imports)

    def test_exec_is_still_opaque_even_with_a_literal(self):
        facts = self.indexer.analyze("exec('import pkg.core')\n")
        self.assertTrue(facts.has_dynamic_imports)
        self.assertEqual([r for r in facts.imports if r.dynamic], [])

    def test_package_keyword_on_a_non_relative_literal_is_ignored(self):
        # ``import_module('pkg.core', package='other')`` - the absolute name
        # wins at runtime, and it must win here too.
        self.assertEqual(
            self.dynamic_modules(
                "import importlib\nimportlib.import_module('pkg.core', package='other')\n"
            ),
            ["pkg.core"],
        )


class DynamicImportEndToEndCase(unittest.TestCase):
    def test_relative_dynamic_import_produces_a_real_graph_edge(self):
        root = tempfile.mkdtemp(prefix="dungx_dyn_")
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        (Path(root) / "pkg").mkdir()
        (Path(root) / "tests").mkdir()
        (Path(root) / "pkg" / "__init__.py").write_text("", encoding="utf-8")
        (Path(root) / "tests" / "__init__.py").write_text("", encoding="utf-8")
        (Path(root) / "pkg" / "core.py").write_text(CORE_AFTER, encoding="utf-8")
        (Path(root) / "tests" / "test_core.py").write_text(
            "import importlib\n\n\ndef test_t():\n"
            "    m = importlib.import_module('.core', package='pkg')\n"
            "    assert m.Widget().render() == 'new'\n",
            encoding="utf-8",
        )
        report = SemanticChangeImpactAnalyzer(root).analyze(
            ["pkg/core.py"], base_contents={"pkg/core.py": CORE_BEFORE}
        )
        self.assertIn("tests/test_core.py", report.affected_files)
        types = {
            evidence.evidence_type
            for target in report.validation_targets
            for evidence in target.dependency_evidence
        }
        self.assertIn(DYNAMIC_IMPORT_RESOLVED, types)


class ScopeEscalationStillHoldsCase(unittest.TestCase):
    """Phase 4.19's guarantee, re-verified against the improved analyzer.

    The rule it protects: a real dependency the graph *missed* must never be
    left at TARGETED scope. Improving resolution changes which cases are
    missed, so this has to be re-checked rather than assumed - a partial
    improvement that resolved a case only *nearly* well enough would otherwise
    be free to narrow validation.
    """

    def analyse(self, source: str):
        root = build_repo(source)
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        return SemanticChangeImpactAnalyzer(root).analyze(
            ["pkg/core.py"], base_contents={"pkg/core.py": CORE_BEFORE}
        )

    #: ``(label, source, there_is_really_a_dependency)``.
    CASES = (
        (
            "unbound_receiver",
            "def test_t(obj=None):\n    assert obj is None or obj.render() == 2\n",
            True,
        ),
        (
            "ambiguous_receiver",
            "from pkg.core import Widget, Unrelated\n\n\ndef test_t():\n"
            "    w = Widget()\n    w = Unrelated()\n    assert w.render()\n",
            True,
        ),
        (
            "dynamic_variable_module",
            "import importlib\n\nNAME = 'pkg.core'\n\n\ndef test_t():\n"
            "    m = importlib.import_module(NAME)\n    assert m.Widget()\n",
            True,
        ),
        (
            "dynamic_relative_no_package",
            "import importlib\n\n\ndef test_t():\n"
            "    m = importlib.import_module('.core')\n    assert m\n",
            True,
        ),
        (
            "no_dependency_at_all",
            "def test_t():\n    assert 1 == 1\n",
            False,
        ),
    )

    def test_a_missed_dependency_is_never_left_at_targeted_scope(self):
        for label, source, real in self.CASES:
            with self.subTest(case=label):
                report = self.analyse(source)
                detected = "tests/test_core.py" in report.affected_files
                if real and not detected:
                    self.assertNotEqual(
                        report.recommended_scope,
                        SCOPE_TARGETED,
                        f"{label}: real dependency missed AND left at targeted scope",
                    )

    def test_resolved_cases_are_genuinely_detected_not_merely_narrowed(self):
        """The other half of the same guarantee.

        A case is only allowed to sit at TARGETED if the dependent actually
        shows up as an affected file. "Narrowed because the analyzer stopped
        noticing the problem" and "narrowed because the analyzer solved it" look
        identical from the scope alone, so the detection is asserted directly.
        """
        for label, source in (
            (
                "constructor_bound",
                "from pkg.core import Widget\n\n\ndef test_t():\n"
                "    w = Widget()\n    assert w.render() == 'new'\n",
            ),
            (
                "relative_dynamic_with_package",
                "import importlib\n\n\ndef test_t():\n"
                "    m = importlib.import_module('.core', package='pkg')\n"
                "    assert m.Widget().render() == 'new'\n",
            ),
        ):
            with self.subTest(case=label):
                report = self.analyse(source)
                if report.recommended_scope == SCOPE_TARGETED:
                    self.assertIn(
                        "tests/test_core.py",
                        report.affected_files,
                        f"{label}: targeted scope without the dependent being detected",
                    )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
