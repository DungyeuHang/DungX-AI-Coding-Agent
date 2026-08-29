"""Phase 4.18 - explainable, provenance-typed dependency resolution.

These tests exercise the real ``ast`` indexer and the real
:class:`~local_agent.semantic_impact.SemanticGraph`/
:class:`~local_agent.semantic_impact.SemanticChangeImpactAnalyzer` pipeline
against real files on disk - only the LLM/provider layer, which nothing here
touches, would ever be mocked.
"""

from __future__ import annotations

import shutil
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Any

from local_agent.dependency_resolution import (
    ALL_EVIDENCE_TYPES,
    ANNOTATION,
    ATTRIBUTE_RESOLUTION,
    CONFIDENCE_BY_EVIDENCE_TYPE,
    DECORATOR,
    DYNAMIC_IMPORT_RESOLVED,
    INHERITANCE,
    IMPORT_ALIAS,
    LEXICAL_FALLBACK,
    REEXPORT,
    DependencyEvidence,
    confidence_for,
    make_evidence,
    resolve_alias_reference,
)
from local_agent.indexing.ast_python_indexer import AstPythonIndexer, ImportRecord
from local_agent.semantic_impact import (
    CONFIDENCE_HIGH,
    SCOPE_BROAD,
    SCOPE_TARGETED,
    SemanticChangeImpactAnalyzer,
    SemanticGraph,
    TIER_CALL_GRAPH,
    TIER_DIRECT_IMPORT,
    TIER_DIRECT_SYMBOL,
    TIER_REVERSE_DEPENDENCY,
)


# ===========================================================================
# 1. Pure evidence model
# ===========================================================================


class TestDependencyEvidenceModel(unittest.TestCase):
    def test_confidence_table_covers_every_declared_type(self):
        for evidence_type in ALL_EVIDENCE_TYPES:
            self.assertIn(evidence_type, CONFIDENCE_BY_EVIDENCE_TYPE)

    def test_confidence_values_are_in_unit_range(self):
        for value in CONFIDENCE_BY_EVIDENCE_TYPE.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_unknown_evidence_type_fails_closed_to_zero(self):
        self.assertEqual(confidence_for("something_invented"), 0.0)

    def test_direct_symbol_is_the_strongest_declared_type(self):
        self.assertEqual(max(CONFIDENCE_BY_EVIDENCE_TYPE.values()), confidence_for("direct_symbol_match"))

    def test_dynamic_import_unresolved_is_zero_confidence(self):
        # Genuine uncertainty must never be dressed up as partial evidence.
        self.assertEqual(confidence_for("dynamic_import_unresolved"), 0.0)

    def test_make_evidence_assigns_table_confidence(self):
        ev = make_evidence(source_file="a.py", target_file="b.py", evidence_type=IMPORT_ALIAS)
        self.assertEqual(ev.confidence, confidence_for(IMPORT_ALIAS))

    def test_to_dict_from_dict_round_trip(self):
        ev = make_evidence(
            source_file="a.py", target_file="b.py", evidence_type=INHERITANCE,
            target_symbol="Base", source_reference="Base", provenance="x inherits from Base",
            resolution_notes=("note",),
        )
        restored = DependencyEvidence.from_dict(ev.to_dict())
        self.assertEqual(restored, ev)

    def test_from_dict_tolerates_garbage(self):
        for payload in (None, [], "x", 42, {}):
            restored = DependencyEvidence.from_dict(payload)
            self.assertIsInstance(restored, DependencyEvidence)

    def test_from_dict_unknown_evidence_type_is_preserved_not_rejected(self):
        # A payload from a newer schema version should round-trip its type
        # string even if this build does not recognise it as strong evidence.
        restored = DependencyEvidence.from_dict({"evidence_type": "future_type_1"})
        self.assertEqual(restored.evidence_type, "future_type_1")

    def test_evidence_is_hashable_and_deduplicates(self):
        a = make_evidence(source_file="a.py", target_file="b.py", evidence_type=DECORATOR)
        b = make_evidence(source_file="a.py", target_file="b.py", evidence_type=DECORATOR)
        self.assertEqual(len({a, b}), 1)


# ===========================================================================
# 2. Alias resolution (pure function)
# ===========================================================================


class TestAliasResolution(unittest.TestCase):
    def test_simple_alias_resolves(self):
        origins = {"test_x.py": {"total": ("module_a.py", "calculate_total")}}
        ev = resolve_alias_reference(
            source_file="test_x.py", reference="total", imported_symbol_origins=origins,
            changed_files=frozenset({"module_a.py"}), changed_symbol_names=frozenset({"calculate_total"}),
        )
        self.assertIsNotNone(ev)
        self.assertEqual(ev.evidence_type, IMPORT_ALIAS)
        self.assertEqual(ev.target_file, "module_a.py")
        self.assertEqual(ev.target_symbol, "calculate_total")
        self.assertIn("total", ev.provenance)
        self.assertIn("calculate_total", ev.provenance)

    def test_unaliased_import_still_resolves(self):
        # ``from x import y`` with no ``as``: local_name == symbol name.
        origins = {"test_x.py": {"calculate_total": ("module_a.py", "calculate_total")}}
        ev = resolve_alias_reference(
            source_file="test_x.py", reference="calculate_total", imported_symbol_origins=origins,
            changed_files=frozenset({"module_a.py"}), changed_symbol_names=frozenset({"calculate_total"}),
        )
        self.assertIsNotNone(ev)
        self.assertIn("imported directly", ev.provenance)

    def test_non_alias_reference_returns_none(self):
        origins = {"test_x.py": {"total": ("module_a.py", "calculate_total")}}
        ev = resolve_alias_reference(
            source_file="test_x.py", reference="unrelated_name", imported_symbol_origins=origins,
            changed_files=frozenset({"module_a.py"}), changed_symbol_names=frozenset({"calculate_total"}),
        )
        self.assertIsNone(ev)

    def test_alias_to_unchanged_file_returns_none(self):
        origins = {"test_x.py": {"total": ("module_a.py", "calculate_total")}}
        ev = resolve_alias_reference(
            source_file="test_x.py", reference="total", imported_symbol_origins=origins,
            changed_files=frozenset({"other.py"}), changed_symbol_names=frozenset({"calculate_total"}),
        )
        self.assertIsNone(ev)

    def test_alias_to_unchanged_symbol_in_changed_file_returns_none(self):
        # The file changed, but a *different* symbol in it did - the alias
        # must not be credited as evidence for a symbol it does not name.
        origins = {"test_x.py": {"total": ("module_a.py", "calculate_total")}}
        ev = resolve_alias_reference(
            source_file="test_x.py", reference="total", imported_symbol_origins=origins,
            changed_files=frozenset({"module_a.py"}), changed_symbol_names=frozenset({"unrelated_symbol"}),
        )
        self.assertIsNone(ev)

    def test_two_hop_reexport_chain_resolves(self):
        origins = {
            "c.py": {"calculate_total": ("a.py", "calculate_total")},
            "a.py": {"calculate_total": ("b.py", "calculate_total")},
        }
        ev = resolve_alias_reference(
            source_file="c.py", reference="calculate_total", imported_symbol_origins=origins,
            changed_files=frozenset({"b.py"}), changed_symbol_names=frozenset({"calculate_total"}),
        )
        self.assertIsNotNone(ev)
        self.assertEqual(ev.target_file, "b.py")

    def test_chain_exceeding_max_hops_returns_none(self):
        # Five real hops with a max_chain of 2 must give up rather than guess.
        origins = {
            "f0.py": {"x": ("f1.py", "x")},
            "f1.py": {"x": ("f2.py", "x")},
            "f2.py": {"x": ("f3.py", "x")},
            "f3.py": {"x": ("f4.py", "x")},
        }
        ev = resolve_alias_reference(
            source_file="f0.py", reference="x", imported_symbol_origins=origins,
            changed_files=frozenset({"f4.py"}), changed_symbol_names=frozenset({"x"}),
            max_chain=2,
        )
        self.assertIsNone(ev)

    def test_cyclic_reexport_terminates(self):
        cyclic = {"a.py": {"x": ("b.py", "x")}, "b.py": {"x": ("a.py", "x")}}
        ev = resolve_alias_reference(
            source_file="a.py", reference="x", imported_symbol_origins=cyclic,
            changed_files=frozenset({"c.py"}), changed_symbol_names=frozenset({"x"}),
        )
        self.assertIsNone(ev)

    def test_self_referential_single_entry_terminates(self):
        # Degenerate but should not hang: an origin pointing at itself.
        origins = {"a.py": {"x": ("a.py", "x")}}
        ev = resolve_alias_reference(
            source_file="a.py", reference="x", imported_symbol_origins=origins,
            changed_files=frozenset({"c.py"}), changed_symbol_names=frozenset({"x"}),
        )
        self.assertIsNone(ev)

    def test_missing_source_file_in_origins_returns_none(self):
        ev = resolve_alias_reference(
            source_file="nowhere.py", reference="x", imported_symbol_origins={},
            changed_files=frozenset({"a.py"}), changed_symbol_names=frozenset({"x"}),
        )
        self.assertIsNone(ev)

    def test_result_is_deterministic(self):
        origins = {"test_x.py": {"total": ("module_a.py", "calculate_total")}}
        kwargs: dict[str, Any] = dict(
            source_file="test_x.py", reference="total", imported_symbol_origins=origins,
            changed_files=frozenset({"module_a.py"}), changed_symbol_names=frozenset({"calculate_total"}),
        )
        first = resolve_alias_reference(**kwargs)
        second = resolve_alias_reference(**kwargs)
        self.assertEqual(first, second)


# ===========================================================================
# 3. End-to-end: real files, real graph, real target selection
# ===========================================================================


class ProjectCase(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="p418_dep_")).resolve()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def write(self, relative: str, content: str) -> None:
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")

    def analyze(self, path: str, new_source: str, *, base_source: str | None = None):
        base = {path: base_source}
        self.write(path, new_source)
        return SemanticChangeImpactAnalyzer(self.root).analyze([path], base_contents=base)

    def target_for(self, report, path: str):
        return next((t for t in report.validation_targets if t.path == path), None)


class TestImportAliasUpgradesTier(ProjectCase):
    def setUp(self) -> None:
        super().setUp()
        self.write("pkg/__init__.py", "")
        self.core = "def calculate_total(values):\n    return sum(values)\n"
        self.write("pkg/core.py", self.core)
        self.write(
            "tests/test_core.py",
            """
            from pkg.core import calculate_total as total


            def test_total():
                assert total([1, 2]) == 3
            """,
        )

    def test_aliased_reference_is_upgraded_to_direct_symbol_match(self):
        new = "def calculate_total(values):\n    return sum(values) + 0\n"
        report = self.analyze("pkg/core.py", new, base_source=self.core)
        target = self.target_for(report, "tests/test_core.py")
        self.assertIsNotNone(target)
        self.assertEqual(target.tier, TIER_DIRECT_SYMBOL)

    def test_alias_evidence_is_attached_with_correct_provenance(self):
        new = "def calculate_total(values):\n    return sum(values) + 0\n"
        report = self.analyze("pkg/core.py", new, base_source=self.core)
        target = self.target_for(report, "tests/test_core.py")
        alias_hits = [e for e in target.dependency_evidence if e.evidence_type == IMPORT_ALIAS]
        self.assertEqual(len(alias_hits), 1)
        self.assertEqual(alias_hits[0].target_symbol, "calculate_total")
        self.assertEqual(alias_hits[0].source_reference, "total")

    def test_evidence_survives_report_serialisation(self):
        new = "def calculate_total(values):\n    return sum(values) + 0\n"
        report = self.analyze("pkg/core.py", new, base_source=self.core)
        from local_agent.semantic_impact import ChangeImpactReport

        restored = ChangeImpactReport.from_dict(report.to_dict())
        target = next(t for t in restored.validation_targets if t.path == "tests/test_core.py")
        self.assertTrue(any(e.evidence_type == IMPORT_ALIAS for e in target.dependency_evidence))

    def test_unrelated_alias_to_a_different_symbol_does_not_upgrade(self):
        # Change ``calculate_total``'s neighbour ``other``; the alias to
        # ``calculate_total`` must not spuriously count as evidence for it.
        self.write(
            "pkg/core.py",
            self.core.rstrip() + "\n\n\ndef other():\n    return 1\n",
        )
        base = self.core
        new = self.core.rstrip() + "\n\n\ndef other():\n    return 2\n"
        report = self.analyze("pkg/core.py", new, base_source=base)
        target = self.target_for(report, "tests/test_core.py")
        # ``other`` is not imported by the test at all, but ``pkg/core.py`` is
        # (via the alias import), so the file-level edge still gives
        # direct_import_match; it must NOT be upgraded to direct_symbol_match
        # since the referenced alias names a different, unchanged symbol.
        self.assertIsNotNone(target)
        self.assertEqual(target.tier, TIER_DIRECT_IMPORT)


class TestModuleAliasAlreadyWorks(ProjectCase):
    """Regression-lock: a module alias needs no special handling because the
    attribute name itself (not the alias) is what gets captured as a reference."""

    def test_module_alias_attribute_call_reaches_direct_symbol_match(self):
        self.write("pkg/__init__.py", "")
        core = "def calculate_total(values):\n    return sum(values)\n"
        self.write("pkg/core.py", core)
        self.write(
            "tests/test_core.py",
            """
            import pkg.core as m


            def test_total():
                assert m.calculate_total([1, 2]) == 3
            """,
        )
        new = "def calculate_total(values):\n    return sum(values) + 0\n"
        report = self.analyze("pkg/core.py", new, base_source=core)
        target = self.target_for(report, "tests/test_core.py")
        self.assertIsNotNone(target)
        self.assertEqual(target.tier, TIER_DIRECT_SYMBOL)


class TestInheritanceDecoratorAnnotationEvidence(ProjectCase):
    def setUp(self) -> None:
        super().setUp()
        self.write("pkg/__init__.py", "")
        self.base_src = "class Base:\n    def m(self):\n        return 1\n"
        self.write("pkg/base.py", self.base_src)

    def test_inheritance_reference_is_tagged(self):
        self.write(
            "tests/test_sub.py",
            """
            from pkg.base import Base


            class Sub(Base):
                pass


            def test_it():
                assert Sub().m() == 1
            """,
        )
        # Change a class-level attribute (not the nested method), so the
        # *class* "Base" itself - not "m" - is the changed symbol name; only
        # then can a base-class reference to "Base" be evidence for it.
        new = "class Base:\n    LIMIT = 5\n\n    def m(self):\n        return 1\n"
        report = self.analyze("pkg/base.py", new, base_source=self.base_src)
        self.assertIn("Base", report.changed_symbol_names)
        target = self.target_for(report, "tests/test_sub.py")
        self.assertIsNotNone(target)
        # This test both imports Base (direct edge) and inherits from it, so
        # inheritance evidence should be present *in addition to* the tier.
        kinds = {e.evidence_type for e in target.dependency_evidence}
        self.assertIn(INHERITANCE, kinds)

    def test_decorator_reference_is_tagged(self):
        deco_src = "def deco(f):\n    return f\n"
        self.write("pkg/deco.py", deco_src)
        self.write(
            "tests/test_deco.py",
            """
            from pkg.deco import deco


            @deco
            def test_it():
                assert True
            """,
        )
        # Change "deco" itself (not merely add an unrelated sibling function),
        # so "deco" - the name actually used as a decorator - is what changed.
        new = "def deco(f):\n    print('wrapped')\n    return f\n"
        report = self.analyze("pkg/deco.py", new, base_source=deco_src)
        self.assertIn("deco", report.changed_symbol_names)
        target = self.target_for(report, "tests/test_deco.py")
        self.assertIsNotNone(target)
        kinds = {e.evidence_type for e in target.dependency_evidence}
        self.assertIn(DECORATOR, kinds)

    def test_annotation_reference_is_tagged(self):
        types_src = "class Config:\n    pass\n"
        self.write("pkg/types_mod.py", types_src)
        self.write(
            "tests/test_types.py",
            """
            from pkg.types_mod import Config


            def helper(x: Config) -> Config:
                return x


            def test_it():
                assert helper(Config()) is not None
            """,
        )
        new = "class Config:\n    value = 1\n"
        report = self.analyze("pkg/types_mod.py", new, base_source=types_src)
        target = self.target_for(report, "tests/test_types.py")
        self.assertIsNotNone(target)
        kinds = {e.evidence_type for e in target.dependency_evidence}
        self.assertIn(ANNOTATION, kinds)


class TestReexportEvidence(ProjectCase):
    def test_intermediate_reexport_via_all_is_tagged(self):
        self.write("pkg/__init__.py", "")
        internal_src = "def calculate_total(values):\n    return sum(values)\n"
        self.write("pkg/internal.py", internal_src)
        # ``facade.py`` re-exports ``calculate_total`` via __all__, and also
        # defines its own ``use()`` that the test actually names - so the test
        # is NOT a bare-name/import match for "calculate_total" itself (which
        # would win a stronger tier before reverse_dependency_match is ever
        # reached); it reaches facade.py only through the reverse-dependency
        # graph, which is exactly the path this evidence type explains.
        self.write(
            "pkg/facade.py",
            """
            from pkg.internal import calculate_total

            __all__ = ["calculate_total"]


            def use():
                return calculate_total([1, 2])
            """,
        )
        self.write(
            "tests/test_facade.py",
            """
            from pkg.facade import use


            def test_it():
                assert use() == 3
            """,
        )
        new = "def calculate_total(values):\n    return sum(values) + 0\n"
        report = self.analyze("pkg/internal.py", new, base_source=internal_src)
        target = self.target_for(report, "tests/test_facade.py")
        self.assertIsNotNone(target)
        self.assertEqual(target.tier, TIER_REVERSE_DEPENDENCY)
        kinds = {e.evidence_type for e in target.dependency_evidence}
        self.assertIn(REEXPORT, kinds)


class TestProportionalDynamicImportDegradation(ProjectCase):
    """Part D: resolvability, not file location, decides how much a dynamic
    import degrades confidence."""

    def setUp(self) -> None:
        super().setUp()
        self.write("pkg/__init__.py", "")
        self.core = "def calculate_total(values):\n    return sum(values)\n"
        self.write("pkg/core.py", self.core)

    def test_literal_dynamic_import_in_a_test_does_not_degrade(self):
        self.write(
            "tests/test_dynamic.py",
            """
            import importlib


            def test_it():
                module = importlib.import_module("pkg.core")
                assert module.calculate_total([1, 2]) == 3
            """,
        )
        new = "def calculate_total(values):\n    return sum(values) + 0\n"
        report = self.analyze("pkg/core.py", new, base_source=self.core)
        # A resolvable literal is not uncertainty: no dynamic-import
        # degradation note should be recorded for this file.
        self.assertFalse(
            any("dynamic" in note for note in report.evidence.degradations)
        )
        target = self.target_for(report, "tests/test_dynamic.py")
        self.assertIsNotNone(target)
        self.assertEqual(target.tier, TIER_DIRECT_SYMBOL)

    def test_computed_dynamic_import_in_production_degrades(self):
        # A *production* module whose dependents cannot be enumerated because
        # of a genuinely unresolvable dynamic import must still degrade.
        self.write(
            "pkg/loader.py",
            """
            def load(name):
                import importlib
                return importlib.import_module(name)


            def uses_core():
                return load("pkg.core").calculate_total([1, 2])
            """,
        )
        new = "def calculate_total(values):\n    return sum(values) + 0\n"
        report = self.analyze("pkg/core.py", new, base_source=self.core)
        # loader.py is not statically known to depend on pkg.core (the import
        # is computed), so it correctly does NOT appear as affected - but if
        # loader.py itself were the *changed* file, its own dynamic call must
        # still be recorded as a degradation. Verify that half directly:
        analyzer_report = self.analyze(
            "pkg/loader.py",
            "def load(name):\n    import importlib\n    return importlib.import_module(name)\n",
            base_source="def load(name):\n    return None\n",
        )
        self.assertTrue(
            any("dynamic" in note for note in analyzer_report.evidence.degradations)
        )

    def test_computed_dynamic_import_in_a_test_never_earns_a_stronger_tier(self):
        # The earlier fixture proves a *resolvable* literal reaches
        # direct_symbol_match (it becomes a real, if dynamic, import edge).
        # A computed argument must not receive that same credit just because
        # the test also happens to name the changed symbol by bare reference -
        # it may still land on the weaker call_graph_match name-only tier, but
        # never a stronger one manufactured from an edge that isn't real.
        self.write(
            "tests/test_dynamic_unresolved.py",
            """
            import importlib

            _NAME = "pkg.core"


            def test_it():
                module = importlib.import_module(_NAME)
                assert module.calculate_total([1, 2]) == 3
            """,
        )
        new = "def calculate_total(values):\n    return sum(values) + 0\n"
        report = self.analyze("pkg/core.py", new, base_source=self.core)
        target = self.target_for(report, "tests/test_dynamic_unresolved.py")
        self.assertIsNotNone(target)
        self.assertEqual(target.tier, TIER_CALL_GRAPH)
        kinds = {e.evidence_type for e in target.dependency_evidence}
        self.assertNotIn(DYNAMIC_IMPORT_RESOLVED, kinds)

    def test_dynamic_import_resolved_evidence_is_attached(self):
        self.write(
            "tests/test_dynamic.py",
            """
            import importlib


            def test_it():
                module = importlib.import_module("pkg.core")
                assert module.calculate_total([1, 2]) == 3
            """,
        )
        new = "def calculate_total(values):\n    return sum(values) + 0\n"
        report = self.analyze("pkg/core.py", new, base_source=self.core)
        target = self.target_for(report, "tests/test_dynamic.py")
        self.assertIsNotNone(target)
        kinds = {e.evidence_type for e in target.dependency_evidence}
        self.assertIn(DYNAMIC_IMPORT_RESOLVED, kinds)

    def test_dynamic_import_to_a_module_outside_the_repo_is_just_unresolved(self):
        # A literal, resolvable-in-principle string that names a third-party
        # package (not present in this repo) must behave exactly like a
        # static import of that same package: an ordinary unresolved import,
        # not a crash and not a spurious edge.
        self.write(
            "tests/test_external.py",
            """
            import importlib


            def test_it():
                json_mod = importlib.import_module("json")
                assert json_mod is not None
            """,
        )
        graph = SemanticGraph.build(self.root)
        self.assertIn("json", graph.unresolved_imports.get("tests/test_external.py", set()))
        self.assertFalse(graph.files["tests/test_external.py"].has_dynamic_imports)


class TestAttributeResolutionEvidence(ProjectCase):
    def test_attribute_access_without_import_is_tagged_and_weak(self):
        self.write("pkg/__init__.py", "")
        core = "def render_widget():\n    return 'w1'\n"
        self.write("pkg/core.py", core)
        # A distinctive attribute access with no import at all: call_graph_match.
        self.write(
            "tests/test_attr.py",
            """
            def test_it():
                service = get_service()
                assert service.render_widget() == 'w2'
            """,
        )
        new = "def render_widget():\n    return 'w2'\n"
        report = self.analyze("pkg/core.py", new, base_source=core)
        target = self.target_for(report, "tests/test_attr.py")
        self.assertIsNotNone(target)
        self.assertEqual(target.tier, TIER_CALL_GRAPH)
        kinds = {e.evidence_type for e in target.dependency_evidence}
        self.assertIn(ATTRIBUTE_RESOLUTION, kinds)
        attr_ev = next(e for e in target.dependency_evidence if e.evidence_type == ATTRIBUTE_RESOLUTION)
        self.assertLess(attr_ev.confidence, confidence_for(IMPORT_ALIAS))


class TestFalsePositiveAndFalseNegativeGuards(ProjectCase):
    def test_similarly_named_unrelated_symbol_is_not_conflated(self):
        # Two unrelated symbols with similar/identical short names in
        # different, unrelated files must not cross-contaminate.
        self.write("pkg/__init__.py", "")
        widget_src = "def render():\n    return 'widget'\n"
        self.write("pkg/widget.py", widget_src)
        self.write("pkg/other.py", "def render():\n    return 'other'\n")
        self.write(
            "tests/test_other.py",
            "from pkg.other import render\n\n\ndef test_o():\n    assert render() == 'other'\n",
        )
        new = "def render():\n    return 'widget2'\n"
        report = self.analyze("pkg/widget.py", new, base_source=widget_src)
        target = self.target_for(report, "tests/test_other.py")
        self.assertTrue(target is None or target.tier not in {TIER_DIRECT_SYMBOL, TIER_DIRECT_IMPORT})

    def test_dynamic_resolution_never_reaches_outside_the_analysed_root(self):
        # A literal dynamic import naming an absolute path component must
        # never be treated as resolved to something outside the repo tree.
        self.write("pkg/__init__.py", "")
        core = "def f():\n    return 1\n"
        self.write("pkg/core.py", core)
        facts = AstPythonIndexer().analyze(
            "import importlib\nimportlib.import_module('/etc/passwd')\n"
        )
        # Not a valid dotted identifier path, so it must be declined, not
        # resolved into a bogus edge.
        self.assertTrue(facts.has_dynamic_imports)
        self.assertEqual([r for r in facts.imports if r.dynamic], [])


if __name__ == "__main__":
    unittest.main()
