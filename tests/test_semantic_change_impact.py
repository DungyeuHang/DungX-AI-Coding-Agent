"""Phase 4.17 - semantic change-impact validation & evidence-aware execution.

These tests exercise the real machinery, not a mock of it:

* real files on a real disk are parsed by the real ``ast`` indexer,
* the real import/reference graph is built and traversed,
* the real :class:`ProspectiveValidator` runs real ``pytest``/``compileall``
  subprocesses against a real :class:`CandidateWorkspace`,
* evidence fingerprints are recomputed from real file bytes.

Only the LLM/provider layer is mocked. The filesystem, subprocesses and graph
analysis are never mocked away.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import os
import shutil
import sys
import tempfile
import textwrap
import threading
import unittest
from pathlib import Path
from typing import Any

from local_agent.coding_agent import CodingAgent, InteractiveCodingAgent
from local_agent.config import AgentConfig, add_common_arguments, config_from_args
from local_agent.evidence import (
    DEFAULT_MAX_EVIDENCE_ENTRIES,
    REASON_COMMAND_MISMATCH,
    REASON_CONFIDENCE_TOO_LOW,
    REASON_FILES_CHANGED,
    REASON_FINGERPRINT_MISMATCH,
    REASON_NO_EVIDENCE,
    REASON_NOT_PASSED,
    REASON_OK,
    REASON_REUSE_DISABLED,
    REASON_SYMBOLS_CHANGED,
    STATUS_FAILED,
    STATUS_PASSED,
    STATUS_SKIPPED,
    EvidenceLedger,
    ValidationEvidence,
    compute_state_fingerprint,
)
from local_agent.filesystem import ProjectFilesystem
from local_agent.indexing.ast_python_indexer import (
    AstPythonIndexer,
    ImportRecord,
    is_public_symbol,
    qualified_name,
)
from local_agent.models import (
    FileOperation,
    ImplementationResult,
    Plan,
    ProjectContext,
    ProviderCapability,
    ReviewResult,
    RunReport,
    SemanticIndex,
)
from local_agent.providers import AIProvider
from local_agent.sandbox import CandidateWorkspace, ProspectiveValidator
from local_agent.semantic_impact import (
    AMBIGUOUS_SYMBOL_DEFINITION_FILES,
    CHANGE_ADDED,
    CHANGE_MODIFIED,
    CHANGE_REMOVED,
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    SCOPE_BROAD,
    SCOPE_EXPANDED,
    SCOPE_ORDER,
    SCOPE_TARGETED,
    TIER_BROAD,
    TIER_CALL_GRAPH,
    TIER_DIRECT_IMPORT,
    TIER_DIRECT_SYMBOL,
    TIER_FILENAME,
    TIER_MODULE,
    TIER_REVERSE_DEPENDENCY,
    TIER_WEIGHTS,
    AffectedSymbol,
    ChangeImpactReport,
    ChangedSymbol,
    ImpactEvidence,
    SemanticChangeImpactAnalyzer,
    SemanticGraph,
    ValidationTarget,
    apply_knowledge_support,
    confidence_at_least,
    diff_python_symbols,
    escalate_scope,
    looks_like_test_path,
    module_name_for,
    recommend_validation_scope,
)
from local_agent.storage import JsonFileStorage
from local_agent.tools import ToolRegistry

# ---------------------------------------------------------------------------
# Fixture project: a small package with genuine import relationships that
# exercise every association tier.
# ---------------------------------------------------------------------------

CORE_PY = textwrap.dedent(
    '''
    """Core engine."""

    __all__ = ["Engine", "compute_widget_total"]


    class Engine:
        def start(self):
            return _bootstrap() + 1

        def _internal(self):
            return 2


    def compute_widget_total(values):
        return sum(values)


    def _bootstrap():
        return 41
    '''
).lstrip()

HELPERS_PY = textwrap.dedent(
    """
    from .core import Engine


    def assist():
        return Engine().start()
    """
).lstrip()

CONSUMER_PY = textwrap.dedent(
    """
    from .helpers import assist


    def use():
        return assist() * 2
    """
).lstrip()

INIT_PY = textwrap.dedent(
    """
    from .core import Engine

    VERSION = "1.0"
    """
).lstrip()

CYCLE_A_PY = textwrap.dedent(
    """
    from .cycle_b import beta


    def alpha():
        return beta()
    """
).lstrip()

CYCLE_B_PY = textwrap.dedent(
    """
    from .cycle_a import alpha


    def beta():
        return 1
    """
).lstrip()

DYNAMIC_PY = textwrap.dedent(
    """
    import importlib

    from .core import Engine


    def load(name):
        return importlib.import_module(name)
    """
).lstrip()

STAR_PY = "from .core import *\n"

WIDGET_PY = "def render():\n    return 'widget'\n"

TEST_CORE_PY = textwrap.dedent(
    """
    from pkg.core import Engine


    def test_start():
        assert Engine().start() == 42
    """
).lstrip()

TEST_HELPERS_PY = textwrap.dedent(
    """
    from pkg.helpers import assist


    def test_assist():
        assert assist() == 42
    """
).lstrip()

TEST_CONSUMER_PY = textwrap.dedent(
    """
    from pkg.consumer import use


    def test_use():
        assert use() == 84
    """
).lstrip()

TEST_WIDGET_PY = textwrap.dedent(
    """
    def test_placeholder():
        assert True
    """
).lstrip()

TEST_UNRELATED_PY = textwrap.dedent(
    """
    def test_nothing():
        assert 1 == 1
    """
).lstrip()

# References ``compute_widget_total`` by bare name without importing pkg.core,
# which is exactly the ``call_graph_match`` situation.
TEST_REFERENCE_ONLY_PY = textwrap.dedent(
    """
    def compute_widget_total_stub(values):
        return sum(values)


    def test_reference():
        assert compute_widget_total_stub([1, 2]) == 3
    """
).lstrip()


def make_project(root: Path) -> None:
    """A real, importable package with a real dependency chain."""
    pkg = root / "pkg"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text(INIT_PY, encoding="utf-8")
    (pkg / "core.py").write_text(CORE_PY, encoding="utf-8")
    (pkg / "helpers.py").write_text(HELPERS_PY, encoding="utf-8")
    (pkg / "consumer.py").write_text(CONSUMER_PY, encoding="utf-8")
    (pkg / "cycle_a.py").write_text(CYCLE_A_PY, encoding="utf-8")
    (pkg / "cycle_b.py").write_text(CYCLE_B_PY, encoding="utf-8")
    (pkg / "dynamic.py").write_text(DYNAMIC_PY, encoding="utf-8")
    (pkg / "star.py").write_text(STAR_PY, encoding="utf-8")
    (pkg / "widget.py").write_text(WIDGET_PY, encoding="utf-8")

    tests = root / "tests"
    tests.mkdir(parents=True, exist_ok=True)
    (tests / "test_core.py").write_text(TEST_CORE_PY, encoding="utf-8")
    (tests / "test_helpers.py").write_text(TEST_HELPERS_PY, encoding="utf-8")
    (tests / "test_consumer.py").write_text(TEST_CONSUMER_PY, encoding="utf-8")
    (tests / "test_widget.py").write_text(TEST_WIDGET_PY, encoding="utf-8")
    (tests / "test_unrelated.py").write_text(TEST_UNRELATED_PY, encoding="utf-8")
    (tests / "test_reference_only.py").write_text(TEST_REFERENCE_ONLY_PY, encoding="utf-8")
    (root / "README.md").write_text("fixture\n", encoding="utf-8")


class ProjectCase(unittest.TestCase):
    """Base case owning a disposable project tree."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="p417_")).resolve()
        make_project(self.root)
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def analyzer(self, **kwargs: Any) -> SemanticChangeImpactAnalyzer:
        return SemanticChangeImpactAnalyzer(self.root, **kwargs)

    def base_contents(self, *paths: str) -> dict[str, str | None]:
        """Exact BASE text for the given paths, as a CandidateWorkspace supplies."""
        out: dict[str, str | None] = {}
        for path in paths:
            candidate = self.root / path
            out[path] = candidate.read_text(encoding="utf-8") if candidate.is_file() else None
        return out

    def write(self, relative: str, content: str) -> None:
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def snapshot(self) -> dict[str, bytes]:
        out: dict[str, bytes] = {}
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                out[path.relative_to(self.root).as_posix()] = path.read_bytes()
        return out


# ===========================================================================
# 1. AST fact extraction
# ===========================================================================


class TestAstPythonIndexer(unittest.TestCase):
    def setUp(self) -> None:
        self.indexer = AstPythonIndexer()

    def test_extracts_module_level_functions_and_classes(self):
        facts = self.indexer.analyze(CORE_PY)
        names = {qualified_name(s) for s in facts.symbols}
        self.assertIn("Engine", names)
        self.assertIn("compute_widget_total", names)
        self.assertIn("_bootstrap", names)

    def test_records_methods_with_their_parent_class(self):
        facts = self.indexer.analyze(CORE_PY)
        by_qname = {qualified_name(s): s for s in facts.symbols}
        self.assertIn("Engine.start", by_qname)
        self.assertEqual(by_qname["Engine.start"].kind, "method")
        self.assertEqual(by_qname["Engine.start"].parent, "Engine")

    def test_module_level_function_kind_is_function(self):
        facts = self.indexer.analyze(CORE_PY)
        by_qname = {qualified_name(s): s for s in facts.symbols}
        self.assertEqual(by_qname["compute_widget_total"].kind, "function")
        self.assertIsNone(by_qname["compute_widget_total"].parent)

    def test_symbol_locations_are_real_line_numbers(self):
        facts = self.indexer.analyze(CORE_PY)
        engine = next(s for s in facts.symbols if s.name == "Engine")
        lines = CORE_PY.splitlines()
        self.assertTrue(lines[engine.location.start_line - 1].startswith("class Engine"))
        self.assertGreater(engine.location.end_line, engine.location.start_line)

    def test_async_functions_are_indexed(self):
        facts = self.indexer.analyze("async def fetch():\n    return 1\n")
        self.assertEqual([s.name for s in facts.symbols], ["fetch"])

    def test_definitions_inside_if_type_checking_are_indexed(self):
        source = "import typing\nif typing.TYPE_CHECKING:\n    class Shim:\n        pass\n"
        facts = self.indexer.analyze(source)
        self.assertIn("Shim", {s.name for s in facts.symbols})

    def test_definitions_inside_try_except_import_are_indexed(self):
        source = "try:\n    import missing\nexcept ImportError:\n    def missing():\n        return 1\n"
        facts = self.indexer.analyze(source)
        self.assertIn("missing", {s.name for s in facts.symbols})

    def test_functions_nested_in_functions_are_not_indexed(self):
        # A closure is not part of any importable surface, so indexing it would
        # add noise to impact analysis without adding signal.
        facts = self.indexer.analyze("def outer():\n    def inner():\n        return 1\n    return inner\n")
        self.assertEqual([s.name for s in facts.symbols], ["outer"])

    def test_plain_import_record(self):
        facts = self.indexer.analyze("import os.path\n")
        self.assertEqual(facts.imports, [ImportRecord(module="os.path", level=0, name=None, asname=None)])

    def test_aliased_import_binds_alias_name(self):
        facts = self.indexer.analyze("import numpy as np\n")
        self.assertEqual(facts.imports[0].local_name, "np")

    def test_plain_import_binds_top_level_package(self):
        facts = self.indexer.analyze("import os.path\n")
        self.assertEqual(facts.imports[0].local_name, "os")

    def test_from_import_records_module_and_name(self):
        facts = self.indexer.analyze("from a.b import c\n")
        record = facts.imports[0]
        self.assertEqual((record.module, record.level, record.name), ("a.b", 0, "c"))

    def test_relative_import_records_level(self):
        facts = self.indexer.analyze("from ..x import y\n")
        record = facts.imports[0]
        self.assertEqual((record.module, record.level, record.name), ("x", 2, "y"))

    def test_bare_relative_import_has_empty_module(self):
        facts = self.indexer.analyze("from . import sibling\n")
        record = facts.imports[0]
        self.assertEqual((record.module, record.level, record.name), ("", 1, "sibling"))

    def test_star_import_flags_dynamic(self):
        facts = self.indexer.analyze("from .core import *\n")
        self.assertTrue(facts.has_dynamic_imports)

    def test_importlib_call_flags_dynamic(self):
        facts = self.indexer.analyze("import importlib\nimportlib.import_module('x')\n")
        self.assertTrue(facts.has_dynamic_imports)

    def test_dunder_import_flags_dynamic(self):
        facts = self.indexer.analyze("__import__('os')\n")
        self.assertTrue(facts.has_dynamic_imports)

    def test_plain_module_has_no_dynamic_flag(self):
        facts = self.indexer.analyze(HELPERS_PY)
        self.assertFalse(facts.has_dynamic_imports)

    def test_references_include_names_and_attributes(self):
        facts = self.indexer.analyze("import mod\n\n\ndef f():\n    return mod.helper(VALUE)\n")
        self.assertIn("helper", facts.references)
        self.assertIn("VALUE", facts.references)

    def test_references_reach_inside_nested_function_bodies(self):
        facts = self.indexer.analyze("def test_x():\n    def inner():\n        return Engine()\n    return inner\n")
        self.assertIn("Engine", facts.references)

    def test_dunder_all_is_captured(self):
        facts = self.indexer.analyze(CORE_PY)
        self.assertEqual(facts.exported_names, ("Engine", "compute_widget_total"))

    def test_syntax_error_yields_parse_error_not_exception(self):
        facts = self.indexer.analyze("def broken(:\n    pass\n")
        self.assertTrue(facts.parse_error)
        self.assertEqual(facts.symbols, [])

    def test_null_byte_source_yields_parse_error(self):
        facts = self.indexer.analyze("x = 1\x00\n")
        self.assertTrue(facts.parse_error)

    def test_undecodable_bytes_yield_parse_error(self):
        facts = self.indexer.analyze(b"\xff\xfe\x00invalid")
        self.assertTrue(facts.parse_error)

    def test_oversized_source_is_refused_not_parsed(self):
        indexer = AstPythonIndexer()
        indexer.max_source_bytes = 10
        facts = indexer.analyze("x = 1\ny = 2\nz = 3\n")
        self.assertIn("maximum indexable size", facts.parse_error)

    def test_index_method_is_signature_compatible_with_treesitter_indexer(self):
        symbols, imports = self.indexer.index(HELPERS_PY.encode("utf-8"))
        self.assertTrue(any(s.name == "assist" for s in symbols))
        self.assertIn(".core", imports)

    def test_symbol_hash_ignores_pure_reformatting(self):
        a = self.indexer.analyze("def f():\n    return 1\n")
        b = self.indexer.analyze("def f():   \n\n    return 1\n\n")
        self.assertEqual(a.symbol_hashes["f"], b.symbol_hashes["f"])

    def test_symbol_hash_changes_for_real_body_edit(self):
        a = self.indexer.analyze("def f():\n    return 1\n")
        b = self.indexer.analyze("def f():\n    return 2\n")
        self.assertNotEqual(a.symbol_hashes["f"], b.symbol_hashes["f"])

    def test_is_public_symbol_rules(self):
        facts = self.indexer.analyze(CORE_PY)
        by_qname = {qualified_name(s): s for s in facts.symbols}
        self.assertTrue(is_public_symbol(by_qname["Engine"], facts.exported_names))
        self.assertFalse(is_public_symbol(by_qname["_bootstrap"], facts.exported_names))
        self.assertFalse(is_public_symbol(by_qname["Engine._internal"], facts.exported_names))

    def test_dunder_all_overrides_underscore_convention(self):
        source = '__all__ = ["_special"]\n\n\ndef _special():\n    return 1\n'
        facts = self.indexer.analyze(source)
        symbol = facts.symbols[0]
        self.assertTrue(is_public_symbol(symbol, facts.exported_names))


# ===========================================================================
# 2. Path / module helpers
# ===========================================================================


class TestModuleResolutionHelpers(unittest.TestCase):
    def test_module_name_for_plain_module(self):
        self.assertEqual(module_name_for("a/b/c.py"), "a.b.c")

    def test_module_name_for_package_init(self):
        self.assertEqual(module_name_for("a/b/__init__.py"), "a.b")

    def test_module_name_for_windows_separators(self):
        self.assertEqual(module_name_for("a\\b\\c.py"), "a.b.c")

    def test_module_name_for_non_python_is_none(self):
        self.assertIsNone(module_name_for("a/b/c.ts"))

    def test_module_name_for_invalid_identifier_is_none(self):
        self.assertIsNone(module_name_for("a/my-dir/c.py"))

    def test_module_name_for_top_level_init_is_none(self):
        self.assertIsNone(module_name_for("__init__.py"))

    def test_looks_like_test_path_variants(self):
        for path in (
            "tests/test_a.py", "test/test_a.py", "a/test_b.py",
            "a/b_test.py", "src/__tests__/c.js", "a/b.spec.ts",
        ):
            self.assertTrue(looks_like_test_path(path), path)

    def test_looks_like_test_path_rejects_ordinary_modules(self):
        for path in ("pkg/core.py", "src/latest.py", "attestation.py"):
            self.assertFalse(looks_like_test_path(path), path)


# ===========================================================================
# 3. Graph construction
# ===========================================================================


class TestSemanticGraph(ProjectCase):
    def setUp(self) -> None:
        super().setUp()
        self.graph = SemanticGraph.build(self.root)

    def test_discovers_every_python_file(self):
        self.assertIn("pkg/core.py", self.graph.files)
        self.assertIn("tests/test_core.py", self.graph.files)
        self.assertNotIn("README.md", self.graph.files)

    def test_module_map_maps_package_init_to_package_name(self):
        self.assertEqual(self.graph.module_to_file["pkg"], "pkg/__init__.py")

    def test_module_map_maps_submodules(self):
        self.assertEqual(self.graph.module_to_file["pkg.core"], "pkg/core.py")

    def test_relative_import_resolves_to_sibling_module(self):
        self.assertIn("pkg/core.py", self.graph.file_imports["pkg/helpers.py"])

    def test_absolute_import_from_test_resolves(self):
        self.assertIn("pkg/core.py", self.graph.file_imports["tests/test_core.py"])

    def test_reverse_dependencies_are_recorded(self):
        self.assertIn("pkg/helpers.py", self.graph.reverse_deps["pkg/core.py"])
        self.assertIn("tests/test_core.py", self.graph.reverse_deps["pkg/core.py"])

    def test_imported_symbol_origin_is_tracked(self):
        origins = self.graph.imported_symbol_origins["pkg/helpers.py"]
        self.assertEqual(origins["Engine"], ("pkg/core.py", "Engine"))

    def test_unresolved_third_party_imports_are_counted_not_invented(self):
        self.assertIn("importlib", self.graph.unresolved_imports["pkg/dynamic.py"])
        self.assertNotIn("importlib", self.graph.module_to_file)

    def test_import_resolution_ratio_is_between_zero_and_one(self):
        self.assertGreater(self.graph.total_imports, 0)
        ratio = self.graph.resolved_import_count / self.graph.total_imports
        self.assertTrue(0.0 < ratio <= 1.0)

    def test_populates_the_shared_semantic_index_model(self):
        index = SemanticIndex()
        graph = SemanticGraph.build(self.root, semantic_index=index)
        self.assertIs(graph.semantic_index, index)
        entry = index.files["pkg/core.py"]
        self.assertEqual(entry.language, "Python")
        self.assertTrue(any(s.name == "Engine" for s in entry.symbols))
        self.assertEqual(
            entry.content_hash,
            hashlib.sha256((self.root / "pkg/core.py").read_bytes()).hexdigest(),
        )

    def test_test_files_helper_lists_only_tests(self):
        tests = self.graph.test_files()
        self.assertIn("tests/test_core.py", tests)
        self.assertNotIn("pkg/core.py", tests)
        self.assertEqual(tests, sorted(tests))

    def test_unparseable_file_is_recorded_but_module_still_registered(self):
        self.write("pkg/broken.py", "def broken(:\n")
        graph = SemanticGraph.build(self.root)
        self.assertIn("pkg/broken.py", graph.parse_failures)
        self.assertEqual(graph.module_to_file["pkg.broken"], "pkg/broken.py")

    def test_max_files_bound_marks_graph_truncated(self):
        graph = SemanticGraph.build(self.root, max_files=3)
        self.assertTrue(graph.truncated_at_max_files)
        self.assertLessEqual(len(graph.files), 3)

    def test_excluded_directories_are_skipped(self):
        (self.root / "node_modules").mkdir()
        (self.root / "node_modules" / "junk.py").write_text("x = 1\n", encoding="utf-8")
        graph = SemanticGraph.build(self.root)
        self.assertNotIn("node_modules/junk.py", graph.files)

    def test_dot_git_as_a_FILE_does_not_break_discovery(self):
        # Inside a Git worktree ``.git`` is a file pointing at the real repo.
        # Phase 4.16 was bitten by this; the graph walker must survive it too.
        (self.root / ".git").write_text("gitdir: /elsewhere/.git/worktrees/w\n", encoding="utf-8")
        graph = SemanticGraph.build(self.root)
        self.assertIn("pkg/core.py", graph.files)
        self.assertNotIn(".git", graph.files)

    def test_self_import_creates_no_edge(self):
        self.write("pkg/selfish.py", "from pkg import selfish\n")
        graph = SemanticGraph.build(self.root)
        self.assertNotIn("pkg/selfish.py", graph.file_imports.get("pkg/selfish.py", set()))

    def test_circular_imports_are_both_recorded(self):
        self.assertIn("pkg/cycle_b.py", self.graph.file_imports["pkg/cycle_a.py"])
        self.assertIn("pkg/cycle_a.py", self.graph.file_imports["pkg/cycle_b.py"])

    def test_ambiguous_symbol_detection(self):
        for index in range(AMBIGUOUS_SYMBOL_DEFINITION_FILES + 2):
            self.write(f"pkg/dup{index}.py", "def shared():\n    return 1\n")
        graph = SemanticGraph.build(self.root)
        self.assertTrue(graph.is_ambiguous_symbol("shared"))
        self.assertFalse(graph.is_ambiguous_symbol("compute_widget_total"))

    def test_duplicate_symbol_names_across_files_are_all_indexed(self):
        self.write("pkg/dup_one.py", "def shared():\n    return 1\n")
        self.write("pkg/dup_two.py", "def shared():\n    return 2\n")
        graph = SemanticGraph.build(self.root)
        self.assertEqual(graph.definition_file_counts["shared"], 2)

    def test_graph_build_is_deterministic(self):
        first = SemanticGraph.build(self.root)
        second = SemanticGraph.build(self.root)
        self.assertEqual(sorted(first.files), sorted(second.files))
        self.assertEqual(first.module_to_file, second.module_to_file)
        self.assertEqual(
            {k: sorted(v) for k, v in first.file_imports.items()},
            {k: sorted(v) for k, v in second.file_imports.items()},
        )


# ===========================================================================
# 4. Reverse-dependency traversal
# ===========================================================================


class TestReverseDependencyTraversal(ProjectCase):
    def setUp(self) -> None:
        super().setUp()
        self.graph = SemanticGraph.build(self.root)

    def test_direct_dependents_at_depth_one(self):
        found, bounds = self.graph.reverse_dependents(["pkg/core.py"], max_depth=1)
        self.assertEqual(found["pkg/helpers.py"][0], 1)
        self.assertIn("pkg/__init__.py", found)

    def test_transitive_dependents_at_depth_two(self):
        found, _ = self.graph.reverse_dependents(["pkg/core.py"], max_depth=3)
        self.assertEqual(found["pkg/consumer.py"][0], 2)

    def test_depth_bound_excludes_deeper_nodes(self):
        found, bounds = self.graph.reverse_dependents(["pkg/core.py"], max_depth=1)
        self.assertNotIn("pkg/consumer.py", found)
        self.assertIn("max_impact_depth", bounds)

    def test_no_bound_reported_when_traversal_completes(self):
        found, bounds = self.graph.reverse_dependents(["pkg/widget.py"], max_depth=5)
        self.assertEqual(bounds, [])

    def test_node_bound_truncates_and_reports(self):
        found, bounds = self.graph.reverse_dependents(
            ["pkg/core.py"], max_depth=5, max_nodes=1
        )
        self.assertEqual(len(found), 1)
        self.assertIn("max_affected_symbols", bounds)

    def test_via_explains_each_edge(self):
        found, _ = self.graph.reverse_dependents(["pkg/core.py"], max_depth=1)
        self.assertEqual(found["pkg/helpers.py"][1], "pkg/helpers.py imports pkg/core.py")

    def test_seed_is_never_reported_as_its_own_dependent(self):
        found, _ = self.graph.reverse_dependents(["pkg/core.py"], max_depth=3)
        self.assertNotIn("pkg/core.py", found)

    def test_circular_imports_terminate(self):
        found, _ = self.graph.reverse_dependents(["pkg/cycle_a.py"], max_depth=10)
        self.assertIn("pkg/cycle_b.py", found)
        self.assertNotIn("pkg/cycle_a.py", found)

    def test_traversal_is_deterministic_across_repeated_runs(self):
        a, bounds_a = self.graph.reverse_dependents(["pkg/core.py"], max_depth=3, max_nodes=4)
        b, bounds_b = self.graph.reverse_dependents(["pkg/core.py"], max_depth=3, max_nodes=4)
        self.assertEqual(a, b)
        self.assertEqual(bounds_a, bounds_b)

    def test_truncation_point_is_deterministic(self):
        results = [
            sorted(self.graph.reverse_dependents(["pkg/core.py"], max_depth=5, max_nodes=2)[0])
            for _ in range(5)
        ]
        self.assertEqual(len(set(map(tuple, results))), 1)

    def test_zero_depth_returns_nothing(self):
        found, _ = self.graph.reverse_dependents(["pkg/core.py"], max_depth=0)
        self.assertEqual(found, {})

    def test_high_fanout_is_actually_capped(self):
        for index in range(60):
            self.write(f"pkg/fan{index}.py", "from .widget import render\n")
        graph = SemanticGraph.build(self.root)
        found, bounds = graph.reverse_dependents(
            ["pkg/widget.py"], max_depth=3, max_nodes=10
        )
        self.assertEqual(len(found), 10)
        self.assertIn("max_affected_symbols", bounds)

    def test_windows_style_seed_paths_are_normalised(self):
        found, _ = self.graph.reverse_dependents(["pkg\\core.py"], max_depth=1)
        self.assertIn("pkg/helpers.py", found)


# ===========================================================================
# 5. Symbol-level diffing
# ===========================================================================


class TestSymbolDiff(unittest.TestCase):
    def test_added_symbol_detected(self):
        changes, error = diff_python_symbols(
            "def a():\n    return 1\n", "def a():\n    return 1\n\n\ndef b():\n    return 2\n", "m.py"
        )
        self.assertEqual(error, "")
        self.assertEqual([(c.name, c.change) for c in changes], [("b", CHANGE_ADDED)])

    def test_removed_symbol_detected(self):
        changes, _ = diff_python_symbols(
            "def a():\n    return 1\n\n\ndef b():\n    return 2\n", "def a():\n    return 1\n", "m.py"
        )
        self.assertEqual([(c.name, c.change) for c in changes], [("b", CHANGE_REMOVED)])

    def test_modified_symbol_detected(self):
        changes, _ = diff_python_symbols(
            "def a():\n    return 1\n", "def a():\n    return 99\n", "m.py"
        )
        self.assertEqual([(c.name, c.change) for c in changes], [("a", CHANGE_MODIFIED)])

    def test_reformatting_is_not_a_semantic_change(self):
        changes, _ = diff_python_symbols(
            "def a():\n    return 1\n", "def a():\n\n    return 1\n   \n", "m.py"
        )
        self.assertEqual(changes, [])

    def test_created_file_reports_every_symbol_as_added(self):
        changes, _ = diff_python_symbols(None, CORE_PY, "pkg/core.py")
        self.assertTrue(changes)
        self.assertTrue(all(c.change == CHANGE_ADDED for c in changes))

    def test_deleted_file_reports_every_symbol_as_removed(self):
        changes, _ = diff_python_symbols(CORE_PY, None, "pkg/core.py")
        self.assertTrue(changes)
        self.assertTrue(all(c.change == CHANGE_REMOVED for c in changes))

    def test_method_changes_use_qualified_names(self):
        base = "class A:\n    def m(self):\n        return 1\n"
        new = "class A:\n    def m(self):\n        return 2\n"
        changes, _ = diff_python_symbols(base, new, "m.py")
        self.assertEqual(changes[0].qualified_name, "A.m")
        self.assertEqual(changes[0].kind, "method")

    def test_publicity_is_recorded(self):
        changes, _ = diff_python_symbols(None, CORE_PY, "pkg/core.py")
        by_name = {c.qualified_name: c for c in changes}
        self.assertTrue(by_name["Engine"].is_public)
        self.assertFalse(by_name["_bootstrap"].is_public)

    def test_renamed_symbol_appears_as_removed_plus_added(self):
        changes, _ = diff_python_symbols(
            "def old_name():\n    return 1\n", "def new_name():\n    return 1\n", "m.py"
        )
        kinds = {(c.name, c.change) for c in changes}
        self.assertEqual(kinds, {("old_name", CHANGE_REMOVED), ("new_name", CHANGE_ADDED)})

    def test_unparseable_base_reports_error_not_empty_success(self):
        changes, error = diff_python_symbols("def broken(:\n", "def a():\n    pass\n", "m.py")
        self.assertEqual(changes, [])
        self.assertIn("base revision unparseable", error)

    def test_unparseable_new_reports_error(self):
        changes, error = diff_python_symbols("def a():\n    pass\n", "def broken(:\n", "m.py")
        self.assertEqual(changes, [])
        self.assertIn("new revision unparseable", error)

    def test_both_revisions_missing_is_an_error(self):
        changes, error = diff_python_symbols(None, None, "m.py")
        self.assertEqual(changes, [])
        self.assertTrue(error)

    def test_changes_are_sorted_deterministically(self):
        base = "def z():\n    return 1\n\n\ndef a():\n    return 1\n"
        new = "def z():\n    return 2\n\n\ndef a():\n    return 2\n"
        first = [c.qualified_name for c in diff_python_symbols(base, new, "m.py")[0]]
        second = [c.qualified_name for c in diff_python_symbols(base, new, "m.py")[0]]
        self.assertEqual(first, second)
        self.assertEqual(first, sorted(first))

    def test_windows_path_is_normalised_in_result(self):
        changes, _ = diff_python_symbols(None, "def a():\n    pass\n", "pkg\\m.py")
        self.assertEqual(changes[0].file, "pkg/m.py")


# ===========================================================================
# 6. Confidence and scope policy
# ===========================================================================


class TestConfidenceModel(unittest.TestCase):
    def test_high_requires_direct_evidence_and_coverage(self):
        evidence = ImpactEvidence(direct_import_matches=1, graph_coverage=1.0)
        level, reason = evidence.assess()
        self.assertEqual(level, CONFIDENCE_HIGH)
        self.assertIn("direct import/symbol evidence", reason)

    def test_direct_evidence_with_low_coverage_is_not_high(self):
        evidence = ImpactEvidence(direct_import_matches=1, graph_coverage=0.4)
        self.assertEqual(evidence.assess()[0], CONFIDENCE_MEDIUM)

    def test_medium_for_weaker_graph_evidence(self):
        evidence = ImpactEvidence(reverse_dependency_matches=2, graph_coverage=0.9)
        level, reason = evidence.assess()
        self.assertEqual(level, CONFIDENCE_MEDIUM)
        self.assertIn("reverse_dependency_match", reason)

    def test_any_degradation_forces_low(self):
        evidence = ImpactEvidence(
            direct_symbol_matches=5, graph_coverage=1.0, degradations=["parse failed"]
        )
        level, reason = evidence.assess()
        self.assertEqual(level, CONFIDENCE_LOW)
        self.assertIn("parse failed", reason)

    def test_lexical_only_evidence_is_low(self):
        evidence = ImpactEvidence(filename_matches=3, graph_coverage=1.0)
        level, reason = evidence.assess()
        self.assertEqual(level, CONFIDENCE_LOW)
        self.assertIn("filename heuristics", reason)

    def test_no_evidence_at_all_is_low(self):
        level, reason = ImpactEvidence().assess()
        self.assertEqual(level, CONFIDENCE_LOW)
        self.assertIn("no association", reason)

    def test_best_tier_prefers_strongest_present(self):
        evidence = ImpactEvidence(direct_import_matches=1, filename_matches=9)
        self.assertEqual(evidence.best_tier, TIER_DIRECT_IMPORT)

    def test_best_tier_is_broad_when_empty(self):
        self.assertEqual(ImpactEvidence().best_tier, TIER_BROAD)

    def test_tier_weights_are_strictly_ordered(self):
        ordered = [
            TIER_DIRECT_SYMBOL, TIER_DIRECT_IMPORT, TIER_CALL_GRAPH,
            TIER_REVERSE_DEPENDENCY, TIER_MODULE, TIER_FILENAME, TIER_BROAD,
        ]
        weights = [TIER_WEIGHTS[t] for t in ordered]
        self.assertEqual(weights, sorted(weights, reverse=True))
        self.assertEqual(len(set(weights)), len(weights))

    def test_confidence_at_least_ordering(self):
        self.assertTrue(confidence_at_least(CONFIDENCE_HIGH, CONFIDENCE_LOW))
        self.assertTrue(confidence_at_least(CONFIDENCE_MEDIUM, CONFIDENCE_MEDIUM))
        self.assertFalse(confidence_at_least(CONFIDENCE_LOW, CONFIDENCE_MEDIUM))

    def test_confidence_at_least_rejects_unknown_values(self):
        self.assertFalse(confidence_at_least("bogus", CONFIDENCE_LOW))
        self.assertFalse(confidence_at_least(CONFIDENCE_HIGH, "bogus"))


class TestScopePolicy(unittest.TestCase):
    def test_escalate_returns_broader_scope(self):
        self.assertEqual(escalate_scope(SCOPE_TARGETED, SCOPE_BROAD), SCOPE_BROAD)
        self.assertEqual(escalate_scope(SCOPE_BROAD, SCOPE_TARGETED), SCOPE_BROAD)
        self.assertEqual(escalate_scope(SCOPE_TARGETED, SCOPE_EXPANDED), SCOPE_EXPANDED)

    def test_escalate_is_idempotent(self):
        for scope in SCOPE_ORDER:
            self.assertEqual(escalate_scope(scope, scope), scope)

    def test_escalate_never_narrows_for_any_pair(self):
        for a in SCOPE_ORDER:
            for b in SCOPE_ORDER:
                result = escalate_scope(a, b)
                self.assertGreaterEqual(SCOPE_ORDER.index(result), SCOPE_ORDER.index(a))
                self.assertGreaterEqual(SCOPE_ORDER.index(result), SCOPE_ORDER.index(b))

    def test_unknown_scope_is_treated_as_broad(self):
        self.assertEqual(escalate_scope("nonsense", SCOPE_TARGETED), SCOPE_BROAD)
        self.assertEqual(escalate_scope(SCOPE_TARGETED, "nonsense"), SCOPE_BROAD)

    def _scope(self, **overrides: Any) -> tuple[str, list[str]]:
        kwargs: dict[str, Any] = dict(
            confidence=CONFIDENCE_HIGH,
            has_targets=True,
            removed_public_symbols=0,
            affected_file_count=1,
            bounds_hit=[],
            unsupported_files=[],
            unparseable_files={},
        )
        kwargs.update(overrides)
        return recommend_validation_scope(**kwargs)

    def test_high_confidence_narrow_change_is_targeted(self):
        self.assertEqual(self._scope()[0], SCOPE_TARGETED)

    def test_medium_confidence_is_expanded(self):
        self.assertEqual(self._scope(confidence=CONFIDENCE_MEDIUM)[0], SCOPE_EXPANDED)

    def test_low_confidence_is_broad(self):
        self.assertEqual(self._scope(confidence=CONFIDENCE_LOW)[0], SCOPE_BROAD)

    def test_no_targets_forces_broad_even_at_high_confidence(self):
        scope, reasons = self._scope(has_targets=False)
        self.assertEqual(scope, SCOPE_BROAD)
        self.assertTrue(any("no validation target" in r for r in reasons))

    def test_removed_public_symbol_escalates(self):
        scope, reasons = self._scope(removed_public_symbols=2)
        self.assertEqual(scope, SCOPE_EXPANDED)
        self.assertTrue(any("public symbol" in r for r in reasons))

    def test_high_fanout_escalates(self):
        scope, reasons = self._scope(affected_file_count=100)
        self.assertEqual(scope, SCOPE_EXPANDED)
        self.assertTrue(any("fan-out" in r for r in reasons))

    def test_bounds_hit_forces_broad(self):
        scope, reasons = self._scope(bounds_hit=["max_impact_depth"])
        self.assertEqual(scope, SCOPE_BROAD)
        self.assertTrue(any("bound" in r for r in reasons))

    def test_unsupported_language_forces_broad(self):
        scope, _ = self._scope(unsupported_files=["app.ts"])
        self.assertEqual(scope, SCOPE_BROAD)

    def test_unparseable_file_forces_broad(self):
        scope, _ = self._scope(unparseable_files={"a.py": "SyntaxError"})
        self.assertEqual(scope, SCOPE_BROAD)

    def test_scope_is_never_skip_for_any_input_combination(self):
        for confidence in (CONFIDENCE_LOW, CONFIDENCE_MEDIUM, CONFIDENCE_HIGH):
            for has_targets in (True, False):
                for bounds in ([], ["max_impact_depth"]):
                    scope, _ = self._scope(
                        confidence=confidence, has_targets=has_targets, bounds_hit=bounds
                    )
                    self.assertIn(scope, SCOPE_ORDER)

    def test_combined_degradations_take_the_broadest_minimum(self):
        scope, reasons = self._scope(
            confidence=CONFIDENCE_HIGH,
            removed_public_symbols=1,
            bounds_hit=["max_affected_symbols"],
        )
        self.assertEqual(scope, SCOPE_BROAD)
        self.assertGreaterEqual(len(reasons), 3)


# ===========================================================================
# 7. End-to-end target selection on a real tree
# ===========================================================================


class TestTargetSelection(ProjectCase):
    def analyze_change(self, path: str, new_source: str | None = None, **kwargs: Any):
        base = self.base_contents(path)
        if new_source is not None:
            self.write(path, new_source)
        analyzer = self.analyzer(**kwargs)
        return analyzer.analyze([path], base_contents=base)

    def targets_by_path(self, report: ChangeImpactReport) -> dict[str, ValidationTarget]:
        return {t.path: t for t in report.validation_targets}

    def test_direct_symbol_match_when_test_imports_and_references(self):
        report = self.analyze_change(
            "pkg/core.py", CORE_PY.replace("return _bootstrap() + 1", "return _bootstrap() + 2")
        )
        target = self.targets_by_path(report)["tests/test_core.py"]
        self.assertEqual(target.tier, TIER_DIRECT_SYMBOL)
        self.assertIn("tests/test_core.py", target.selected_because)
        self.assertIn("pkg/core.py", target.selected_because)

    def test_direct_import_match_when_symbol_not_referenced(self):
        # Change only ``_bootstrap``, which tests/test_core.py never names.
        report = self.analyze_change(
            "pkg/core.py", CORE_PY.replace("return 41", "return 40")
        )
        target = self.targets_by_path(report)["tests/test_core.py"]
        self.assertEqual(target.tier, TIER_DIRECT_IMPORT)
        self.assertIn("directly imports", target.selected_because)

    def test_reverse_dependency_match_for_indirect_test(self):
        report = self.analyze_change("pkg/core.py", CORE_PY.replace("return 41", "return 40"))
        target = self.targets_by_path(report)["tests/test_helpers.py"]
        self.assertEqual(target.tier, TIER_REVERSE_DEPENDENCY)
        self.assertEqual(target.depth, 1)
        self.assertIn("transitively depend", target.selected_because)

    def test_reverse_dependency_depth_two_is_reported(self):
        report = self.analyze_change("pkg/core.py", CORE_PY.replace("return 41", "return 40"))
        target = self.targets_by_path(report)["tests/test_consumer.py"]
        self.assertEqual(target.tier, TIER_REVERSE_DEPENDENCY)
        self.assertEqual(target.depth, 2)

    def test_call_graph_match_for_distinctive_name_without_import(self):
        report = self.analyze_change(
            "pkg/core.py", CORE_PY.replace("return sum(values)", "return sum(values) + 0")
        )
        target = self.targets_by_path(report).get("tests/test_reference_only.py")
        self.assertIsNotNone(target)
        self.assertEqual(target.tier, TIER_CALL_GRAPH)
        self.assertIn("name match only", target.selected_because)

    def test_module_match_for_naming_convention_only(self):
        report = self.analyze_change("pkg/widget.py", "def render():\n    return 'w2'\n")
        target = self.targets_by_path(report)["tests/test_widget.py"]
        self.assertEqual(target.tier, TIER_MODULE)
        self.assertIn("named after changed module", target.selected_because)

    def test_changed_test_file_selects_itself(self):
        report = self.analyze_change(
            "tests/test_widget.py", "def test_placeholder():\n    assert 2 == 2\n"
        )
        target = self.targets_by_path(report)["tests/test_widget.py"]
        self.assertEqual(target.tier, TIER_DIRECT_SYMBOL)
        self.assertIn("was itself changed", target.selected_because)

    def test_unrelated_test_is_not_selected(self):
        report = self.analyze_change("pkg/widget.py", "def render():\n    return 'w2'\n")
        self.assertNotIn("tests/test_unrelated.py", self.targets_by_path(report))

    def test_targets_are_ranked_strongest_first(self):
        report = self.analyze_change(
            "pkg/core.py", CORE_PY.replace("return _bootstrap() + 1", "return _bootstrap() + 3")
        )
        weights = [t.weight for t in report.validation_targets]
        self.assertEqual(weights, sorted(weights, reverse=True))

    def test_every_target_carries_an_explanation(self):
        report = self.analyze_change("pkg/core.py", CORE_PY.replace("return 41", "return 40"))
        for target in report.validation_targets:
            self.assertTrue(target.selected_because.strip(), target.path)

    def test_ambiguous_symbol_names_do_not_create_call_graph_targets(self):
        # ``render`` becomes defined in many modules, so a bare reference to it
        # must stop counting as evidence.
        for index in range(AMBIGUOUS_SYMBOL_DEFINITION_FILES + 2):
            self.write(f"pkg/many{index}.py", "def render():\n    return 1\n")
        self.write("tests/test_bare_reference.py", "def test_x():\n    render = 1\n    assert render\n")
        report = self.analyze_change("pkg/widget.py", "def render():\n    return 'w2'\n")
        target = self.targets_by_path(report).get("tests/test_bare_reference.py")
        self.assertTrue(target is None or target.tier != TIER_CALL_GRAPH)
        self.assertTrue(any("ambiguous" in note for note in report.knowledge_notes))

    def test_one_test_covering_multiple_changed_symbols_lists_them_all(self):
        new = CORE_PY.replace("return _bootstrap() + 1", "return _bootstrap() + 5")
        new = new.replace("return sum(values)", "return sum(values) + 0")
        report = self.analyze_change("pkg/core.py", new)
        target = self.targets_by_path(report)["tests/test_core.py"]
        self.assertEqual(target.tier, TIER_DIRECT_SYMBOL)
        self.assertGreaterEqual(len(target.matched_symbols), 1)

    def test_private_helper_change_still_reaches_public_api_dependents(self):
        report = self.analyze_change("pkg/core.py", CORE_PY.replace("return 41", "return 40"))
        self.assertTrue(any(s.name == "_bootstrap" for s in report.modified_symbols))
        self.assertFalse(any(s.name == "_bootstrap" and s.is_public for s in report.modified_symbols))
        # Its public dependents are still surfaced as affected modules.
        self.assertIn("pkg/helpers.py", report.affected_files)

    def test_max_affected_tests_bound_truncates_and_degrades(self):
        report = self.analyze_change(
            "pkg/core.py", CORE_PY.replace("return 41", "return 40"), max_affected_tests=1
        )
        self.assertEqual(len(report.validation_targets), 1)
        self.assertIn("max_affected_tests", report.bounds_hit)
        self.assertEqual(report.confidence, CONFIDENCE_LOW)
        self.assertEqual(report.recommended_scope, SCOPE_BROAD)

    def test_broad_fallback_target_when_nothing_associates(self):
        self.write("lonely.py", "def solitary_unique_function():\n    return 1\n")
        report = self.analyzer().analyze(
            ["lonely.py"], base_contents={"lonely.py": None}
        )
        self.assertEqual(report.validation_targets[0].tier, TIER_BROAD)
        self.assertEqual(report.validation_targets[0].command, ("pytest",))
        self.assertEqual(report.recommended_scope, SCOPE_BROAD)

    def test_lexical_tier_is_used_when_graph_finds_nothing(self):
        # ``notes.py`` has a matching ``tests/test_notes.py`` but no import edge
        # and no distinctive symbol reference, so the module tier catches it.
        self.write("notes.py", "def jot():\n    return 1\n")
        self.write("tests/test_notes.py", "def test_jot():\n    assert True\n")
        report = self.analyzer().analyze(["notes.py"], base_contents={"notes.py": None})
        target = self.targets_by_path(report).get("tests/test_notes.py")
        self.assertIsNotNone(target)
        self.assertIn(target.tier, {TIER_MODULE, TIER_FILENAME})

    def test_tests_considered_counts_the_whole_candidate_pool(self):
        report = self.analyze_change("pkg/core.py", CORE_PY.replace("return 41", "return 40"))
        self.assertEqual(report.tests_considered, len(SemanticGraph.build(self.root).test_files()))


# ===========================================================================
# 8. Uncertainty always widens, never narrows
# ===========================================================================


class TestUncertaintyHandling(ProjectCase):
    def test_non_python_change_is_unsupported_and_broadens(self):
        self.write("app.ts", "export const x = 1;\n")
        report = self.analyzer().analyze(["app.ts"], base_contents={"app.ts": None})
        self.assertIn("app.ts", report.unsupported_files)
        self.assertEqual(report.confidence, CONFIDENCE_LOW)
        self.assertEqual(report.recommended_scope, SCOPE_BROAD)

    def test_unparseable_changed_file_broadens(self):
        base = self.base_contents("pkg/core.py")
        self.write("pkg/core.py", "class Engine(:\n")
        report = self.analyzer().analyze(["pkg/core.py"], base_contents=base)
        self.assertIn("pkg/core.py", report.unparseable_files)
        self.assertEqual(report.recommended_scope, SCOPE_BROAD)

    def test_missing_base_snapshot_is_a_degradation(self):
        report = self.analyzer().analyze(["pkg/core.py"])
        self.assertTrue(any("no base snapshot" in d for d in report.evidence.degradations))
        self.assertEqual(report.confidence, CONFIDENCE_LOW)

    def test_dynamic_import_in_changed_module_degrades(self):
        base = self.base_contents("pkg/dynamic.py")
        self.write("pkg/dynamic.py", DYNAMIC_PY.replace("return importlib", "return  importlib"))
        report = self.analyzer().analyze(["pkg/dynamic.py"], base_contents=base)
        self.assertTrue(any("dynamic or star" in d for d in report.evidence.degradations))
        self.assertEqual(report.confidence, CONFIDENCE_LOW)

    def test_star_import_dependent_degrades_confidence(self):
        base = self.base_contents("pkg/core.py")
        self.write("pkg/core.py", CORE_PY.replace("return 41", "return 40"))
        report = self.analyzer().analyze(["pkg/core.py"], base_contents=base)
        # pkg/star.py does ``from .core import *`` and is a dependent.
        self.assertIn("pkg/star.py", report.affected_files)
        self.assertTrue(any("dynamic or star" in d for d in report.evidence.degradations))

    def test_deleted_file_is_analysed_as_removals_not_ignored(self):
        base = self.base_contents("pkg/widget.py")
        (self.root / "pkg" / "widget.py").unlink()
        report = self.analyzer().analyze(["pkg/widget.py"], base_contents=base)
        self.assertTrue(report.removed_symbols)
        self.assertEqual(report.removed_symbols[0].name, "render")

    def test_missing_file_with_no_base_is_reported_unreadable(self):
        report = self.analyzer().analyze(
            ["pkg/gone.py"], base_contents={"pkg/gone.py": None}
        )
        self.assertIn("pkg/gone.py", report.unparseable_files)
        self.assertEqual(report.recommended_scope, SCOPE_BROAD)

    def test_empty_change_set_needs_no_validation_but_is_not_an_error(self):
        report = self.analyzer().analyze([])
        self.assertEqual(report.changed_files, [])
        self.assertEqual(report.recommended_scope, SCOPE_TARGETED)
        self.assertIn("nothing changed", report.scope_reasons[0])

    def test_graph_file_bound_degrades_confidence(self):
        base = self.base_contents("pkg/core.py")
        self.write("pkg/core.py", CORE_PY.replace("return 41", "return 40"))
        report = self.analyzer(max_graph_files=2).analyze(
            ["pkg/core.py"], base_contents=base
        )
        self.assertIn("max_graph_files", report.bounds_hit)
        self.assertEqual(report.confidence, CONFIDENCE_LOW)
        self.assertEqual(report.recommended_scope, SCOPE_BROAD)

    def test_high_confidence_is_achievable_on_a_clean_narrow_change(self):
        # Guards against the policy being so conservative it never narrows.
        base = self.base_contents("pkg/widget.py")
        self.write("pkg/widget.py", "def render():\n    return 'w2'\n")
        self.write("tests/test_widget.py", "from pkg.widget import render\n\n\ndef test_r():\n    assert render()\n")
        report = self.analyzer().analyze(["pkg/widget.py"], base_contents=base)
        self.assertEqual(report.confidence, CONFIDENCE_HIGH)
        self.assertEqual(report.recommended_scope, SCOPE_TARGETED)

    def test_analysis_is_deterministic_run_twice(self):
        base = self.base_contents("pkg/core.py")
        self.write("pkg/core.py", CORE_PY.replace("return 41", "return 40"))
        first = self.analyzer().analyze(["pkg/core.py"], base_contents=base)
        second = self.analyzer().analyze(["pkg/core.py"], base_contents=base)
        self.assertEqual(first.fingerprint(), second.fingerprint())

    def test_analysis_is_deterministic_with_a_shared_graph(self):
        graph = SemanticGraph.build(self.root)
        base = self.base_contents("pkg/core.py")
        a = SemanticChangeImpactAnalyzer(self.root, graph=graph).analyze(
            ["pkg/core.py"], base_contents=base
        )
        b = SemanticChangeImpactAnalyzer(self.root, graph=graph).analyze(
            ["pkg/core.py"], base_contents=base
        )
        self.assertEqual(a.to_dict()["validation_targets"], b.to_dict()["validation_targets"])


# ===========================================================================
# 9. Serialisation & backward compatibility
# ===========================================================================


class TestSerialisation(ProjectCase):
    def build_report(self) -> ChangeImpactReport:
        base = self.base_contents("pkg/core.py")
        self.write("pkg/core.py", CORE_PY.replace("return 41", "return 40"))
        return self.analyzer().analyze(["pkg/core.py"], base_contents=base)

    def test_report_round_trips(self):
        report = self.build_report()
        restored = ChangeImpactReport.from_dict(report.to_dict())
        self.assertEqual(restored.to_dict(), report.to_dict())

    def test_report_from_empty_dict_uses_safe_defaults(self):
        restored = ChangeImpactReport.from_dict({})
        self.assertEqual(restored.confidence, CONFIDENCE_LOW)
        self.assertEqual(restored.recommended_scope, SCOPE_BROAD)

    def test_report_from_none_is_safe(self):
        self.assertEqual(ChangeImpactReport.from_dict(None).changed_files, [])

    def test_report_ignores_unknown_future_keys(self):
        payload = self.build_report().to_dict()
        payload["a_field_from_phase_500"] = {"x": 1}
        restored = ChangeImpactReport.from_dict(payload)
        self.assertEqual(restored.confidence, payload["confidence"])

    def test_report_tolerates_legacy_partial_payload(self):
        restored = ChangeImpactReport.from_dict(
            {"changed_files": ["a.py"], "confidence": "medium"}
        )
        self.assertEqual(restored.changed_files, ["a.py"])
        self.assertEqual(restored.confidence, CONFIDENCE_MEDIUM)
        self.assertEqual(restored.validation_targets, [])

    def test_changed_symbol_round_trip(self):
        symbol = ChangedSymbol("A.m", "m", "a.py", "method", CHANGE_MODIFIED, True)
        self.assertEqual(ChangedSymbol.from_dict(symbol.to_dict()), symbol)

    def test_affected_symbol_round_trip(self):
        affected = AffectedSymbol("pkg.core", "pkg/core.py", 2, "x imports y")
        self.assertEqual(AffectedSymbol.from_dict(affected.to_dict()), affected)

    def test_validation_target_round_trip(self):
        target = ValidationTarget("t.py", ("pytest", "t.py"), TIER_DIRECT_IMPORT, "because", ["S"], ["f.py"], 1)
        self.assertEqual(ValidationTarget.from_dict(target.to_dict()), target)

    def test_validation_target_command_becomes_tuple(self):
        restored = ValidationTarget.from_dict({"path": "t.py", "command": ["pytest", "t.py"]})
        self.assertIsInstance(restored.command, tuple)

    def test_impact_evidence_round_trip(self):
        evidence = ImpactEvidence(direct_symbol_matches=2, graph_coverage=0.5, degradations=["x"])
        self.assertEqual(ImpactEvidence.from_dict(evidence.to_dict()), evidence)

    def test_fingerprint_ignores_timing(self):
        report = self.build_report()
        payload = report.to_dict()
        other = ChangeImpactReport.from_dict(payload)
        other.analysis_seconds = report.analysis_seconds + 99.0
        self.assertEqual(other.fingerprint(), report.fingerprint())

    def test_summary_is_human_readable(self):
        summary = self.build_report().summary()
        self.assertIn("Confidence:", summary)
        self.assertIn("Recommended validation scope:", summary)

    def test_implementation_result_carries_new_fields_and_stays_backward_compatible(self):
        result = ImplementationResult(
            semantic_impact_used=True,
            impact_confidence=CONFIDENCE_HIGH,
            impact_recommended_scope=SCOPE_TARGETED,
            impact_tests_selected=3,
        )
        restored = ImplementationResult.from_dict(result.to_dict())
        self.assertTrue(restored.semantic_impact_used)
        self.assertEqual(restored.impact_confidence, CONFIDENCE_HIGH)
        self.assertEqual(restored.impact_tests_selected, 3)

    def test_implementation_result_loads_pre_phase_417_payload(self):
        legacy = {"success": True, "summary": "old", "candidate_iterations": 2}
        restored = ImplementationResult.from_dict(legacy)
        self.assertTrue(restored.success)
        self.assertEqual(restored.candidate_iterations, 2)
        self.assertFalse(restored.semantic_impact_used)
        self.assertEqual(restored.impact_confidence, "")
        self.assertEqual(restored.validation_evidence, [])

    def test_run_report_has_additive_phase_417_fields(self):
        report = RunReport(project=ProjectContext(root=str(self.root)))
        self.assertIsNone(report.semantic_impact)
        self.assertEqual(report.validation_evidence, [])


# ===========================================================================
# 10. Evidence ledger
# ===========================================================================


class TestEvidenceLedger(ProjectCase):
    def setUp(self) -> None:
        super().setUp()
        self.ledger = EvidenceLedger()

    def record(self, **kwargs: Any) -> ValidationEvidence:
        defaults: dict[str, Any] = dict(
            command=("pytest", "tests/test_core.py"),
            status=STATUS_PASSED,
            duration_seconds=1.5,
            impacted_files=["pkg/core.py"],
            impacted_symbols=["Engine"],
            confidence=CONFIDENCE_HIGH,
            candidate_iteration=1,
            environment_root="candidate",
        )
        defaults.update(kwargs)
        defaults.setdefault(
            "fingerprint",
            compute_state_fingerprint(self.root, defaults["impacted_files"]),
        )
        return self.ledger.record(**defaults)

    def test_record_returns_the_stored_entry(self):
        entry = self.record()
        self.assertEqual(self.ledger.entries, [entry])

    def test_entries_are_bounded_oldest_first(self):
        ledger = EvidenceLedger(max_entries=3)
        for index in range(6):
            ledger.record(command=("pytest", f"t{index}.py"), status=STATUS_PASSED)
        self.assertEqual(len(ledger), 3)
        self.assertEqual(ledger.entries[0].command, ("pytest", "t3.py"))

    def test_default_bound_is_applied(self):
        self.assertEqual(EvidenceLedger().max_entries, DEFAULT_MAX_EVIDENCE_ENTRIES)

    def test_output_summaries_are_trimmed(self):
        ledger = EvidenceLedger(max_summary_chars=40)
        entry = ledger.record(command=("x",), status=STATUS_FAILED, stdout="A" * 500)
        self.assertLessEqual(len(entry.stdout_summary), 80)
        self.assertIn("trimmed", entry.stdout_summary)

    def test_short_output_is_not_trimmed(self):
        entry = self.record(stdout="all good")
        self.assertEqual(entry.stdout_summary, "all good")

    def test_impacted_files_are_normalised_and_sorted(self):
        entry = self.record(impacted_files=["b\\x.py", "a.py", "a.py"])
        self.assertEqual(entry.impacted_files, ["a.py", "b/x.py"])

    def test_iteration_history_is_retained(self):
        self.record(candidate_iteration=1)
        self.record(candidate_iteration=2, command=("pytest", "tests/test_helpers.py"))
        self.assertEqual(self.ledger.iterations, [1, 2])
        self.assertEqual(len(self.ledger.entries_for_iteration(2)), 1)

    def test_status_helpers(self):
        self.assertTrue(self.record(status=STATUS_PASSED).passed)
        self.assertFalse(self.record(status=STATUS_FAILED).passed)
        self.assertFalse(self.record(status=STATUS_SKIPPED).passed)

    def test_ledger_round_trips(self):
        self.record()
        self.record(candidate_iteration=2)
        restored = EvidenceLedger.from_dict(self.ledger.to_dict())
        self.assertEqual(
            [e.to_dict() for e in restored.entries],
            [e.to_dict() for e in self.ledger.entries],
        )

    def test_ledger_from_empty_or_none(self):
        self.assertEqual(len(EvidenceLedger.from_dict({})), 0)
        self.assertEqual(len(EvidenceLedger.from_dict(None)), 0)

    def test_ledger_from_legacy_payload_with_only_entries(self):
        self.record()
        restored = EvidenceLedger.from_dict({"entries": [e.to_dict() for e in self.ledger.entries]})
        self.assertEqual(len(restored), 1)

    def test_ledger_deserialisation_respects_max_entries(self):
        for index in range(10):
            self.record(command=("pytest", f"t{index}.py"))
        payload = self.ledger.to_dict()
        payload["max_entries"] = 2
        restored = EvidenceLedger.from_dict(payload)
        self.assertEqual(len(restored), 2)

    def test_validation_evidence_from_unknown_shape_is_safe(self):
        self.assertEqual(ValidationEvidence.from_dict("nonsense").command, ())

    def test_evidence_ignores_unknown_future_keys(self):
        payload = self.record().to_dict()
        payload["future_field"] = 1
        self.assertEqual(ValidationEvidence.from_dict(payload).confidence, CONFIDENCE_HIGH)


# ===========================================================================
# 11. Evidence reuse & invalidation
# ===========================================================================


class TestEvidenceReuse(ProjectCase):
    def setUp(self) -> None:
        super().setUp()
        self.ledger = EvidenceLedger()
        self.files = ["pkg/core.py", "tests/test_core.py"]
        self.symbols = ["Engine.start"]
        self.command = ("pytest", "tests/test_core.py")
        self.ledger.record(
            command=self.command,
            status=STATUS_PASSED,
            duration_seconds=2.5,
            impacted_files=self.files,
            impacted_symbols=self.symbols,
            confidence=CONFIDENCE_HIGH,
            candidate_iteration=1,
            fingerprint=compute_state_fingerprint(self.root, self.files),
        )

    def decide(self, **overrides: Any):
        kwargs: dict[str, Any] = dict(
            command=self.command,
            current_root=self.root,
            relevant_files=self.files,
            relevant_symbols=self.symbols,
            min_confidence=CONFIDENCE_HIGH,
        )
        kwargs.update(overrides)
        return self.ledger.find_reusable(**kwargs)

    def test_reuse_granted_when_every_assumption_holds(self):
        decision = self.decide()
        self.assertTrue(decision.reusable)
        self.assertEqual(decision.reason, REASON_OK)
        self.assertEqual(decision.time_saved_seconds, 2.5)

    def test_reuse_updates_counters(self):
        self.decide()
        self.assertEqual(self.ledger.reuse_grants, 1)
        self.assertEqual(self.ledger.time_saved_seconds, 2.5)

    def test_tampering_with_file_content_invalidates(self):
        (self.root / "pkg" / "core.py").write_text(CORE_PY + "\n# tampered\n", encoding="utf-8")
        decision = self.decide()
        self.assertFalse(decision.reusable)
        self.assertEqual(decision.reason, REASON_FINGERPRINT_MISMATCH)

    def test_deleting_a_relevant_file_invalidates(self):
        (self.root / "tests" / "test_core.py").unlink()
        self.assertEqual(self.decide().reason, REASON_FINGERPRINT_MISMATCH)

    def test_adding_a_relevant_file_invalidates(self):
        decision = self.decide(relevant_files=self.files + ["pkg/helpers.py"])
        self.assertFalse(decision.reusable)
        self.assertEqual(decision.reason, REASON_FILES_CHANGED)

    def test_different_symbol_set_invalidates(self):
        decision = self.decide(relevant_symbols=["Engine.start", "compute_widget_total"])
        self.assertEqual(decision.reason, REASON_SYMBOLS_CHANGED)

    def test_different_command_invalidates(self):
        decision = self.decide(command=("pytest", "tests/test_helpers.py"))
        self.assertEqual(decision.reason, REASON_COMMAND_MISMATCH)

    def test_command_argument_order_matters(self):
        decision = self.decide(command=("tests/test_core.py", "pytest"))
        self.assertFalse(decision.reusable)

    def test_failed_evidence_is_never_reused(self):
        ledger = EvidenceLedger()
        ledger.record(
            command=self.command, status=STATUS_FAILED, exit_code=1,
            impacted_files=self.files, impacted_symbols=self.symbols,
            confidence=CONFIDENCE_HIGH,
            fingerprint=compute_state_fingerprint(self.root, self.files),
        )
        self.ledger = ledger
        self.assertEqual(self.decide().reason, REASON_NOT_PASSED)

    def test_skipped_evidence_is_never_reused(self):
        ledger = EvidenceLedger()
        ledger.record(
            command=self.command, status=STATUS_SKIPPED,
            impacted_files=self.files, impacted_symbols=self.symbols,
            confidence=CONFIDENCE_HIGH,
            fingerprint=compute_state_fingerprint(self.root, self.files),
        )
        self.ledger = ledger
        self.assertFalse(self.decide().reusable)

    def test_confidence_below_threshold_invalidates(self):
        ledger = EvidenceLedger()
        ledger.record(
            command=self.command, status=STATUS_PASSED,
            impacted_files=self.files, impacted_symbols=self.symbols,
            confidence=CONFIDENCE_LOW,
            fingerprint=compute_state_fingerprint(self.root, self.files),
        )
        self.ledger = ledger
        self.assertEqual(self.decide().reason, REASON_CONFIDENCE_TOO_LOW)

    def test_lower_threshold_allows_medium_confidence_reuse(self):
        ledger = EvidenceLedger()
        ledger.record(
            command=self.command, status=STATUS_PASSED,
            impacted_files=self.files, impacted_symbols=self.symbols,
            confidence=CONFIDENCE_MEDIUM,
            fingerprint=compute_state_fingerprint(self.root, self.files),
        )
        self.ledger = ledger
        self.assertTrue(self.decide(min_confidence=CONFIDENCE_MEDIUM).reusable)

    def test_empty_ledger_reports_no_evidence(self):
        self.ledger = EvidenceLedger()
        self.assertEqual(self.decide().reason, REASON_NO_EVIDENCE)

    def test_disabled_reuse_short_circuits(self):
        decision = self.decide(enabled=False)
        self.assertFalse(decision.reusable)
        self.assertEqual(decision.reason, REASON_REUSE_DISABLED)

    def test_latest_iteration_supersedes_earlier_one(self):
        self.ledger.record(
            command=self.command, status=STATUS_PASSED, duration_seconds=9.0,
            impacted_files=self.files, impacted_symbols=self.symbols,
            confidence=CONFIDENCE_HIGH, candidate_iteration=2,
            fingerprint=compute_state_fingerprint(self.root, self.files),
        )
        decision = self.decide()
        self.assertTrue(decision.reusable)
        self.assertEqual(decision.evidence.candidate_iteration, 2)

    def test_stale_earlier_iteration_does_not_block_valid_later_one(self):
        ledger = EvidenceLedger()
        ledger.record(
            command=self.command, status=STATUS_FAILED,
            impacted_files=self.files, impacted_symbols=self.symbols,
            confidence=CONFIDENCE_HIGH, candidate_iteration=1, fingerprint="stale",
        )
        ledger.record(
            command=self.command, status=STATUS_PASSED, duration_seconds=1.0,
            impacted_files=self.files, impacted_symbols=self.symbols,
            confidence=CONFIDENCE_HIGH, candidate_iteration=2,
            fingerprint=compute_state_fingerprint(self.root, self.files),
        )
        self.ledger = ledger
        self.assertTrue(self.decide().reusable)

    def test_denials_are_counted(self):
        self.decide(command=("nope",))
        self.assertEqual(self.ledger.reuse_denials, 1)


class TestStateFingerprint(ProjectCase):
    def test_fingerprint_is_stable_for_unchanged_content(self):
        a = compute_state_fingerprint(self.root, ["pkg/core.py"])
        b = compute_state_fingerprint(self.root, ["pkg/core.py"])
        self.assertEqual(a, b)

    def test_fingerprint_changes_when_content_changes(self):
        before = compute_state_fingerprint(self.root, ["pkg/core.py"])
        (self.root / "pkg" / "core.py").write_text(CORE_PY + "# x\n", encoding="utf-8")
        self.assertNotEqual(before, compute_state_fingerprint(self.root, ["pkg/core.py"]))

    def test_fingerprint_is_root_independent_for_identical_content(self):
        other = Path(tempfile.mkdtemp(prefix="p417b_")).resolve()
        self.addCleanup(shutil.rmtree, other, ignore_errors=True)
        make_project(other)
        self.assertEqual(
            compute_state_fingerprint(self.root, ["pkg/core.py"]),
            compute_state_fingerprint(other, ["pkg/core.py"]),
        )

    def test_fingerprint_ignores_path_order_and_duplicates(self):
        a = compute_state_fingerprint(self.root, ["pkg/core.py", "pkg/helpers.py"])
        b = compute_state_fingerprint(self.root, ["pkg/helpers.py", "pkg/core.py", "pkg/core.py"])
        self.assertEqual(a, b)

    def test_fingerprint_normalises_windows_separators(self):
        self.assertEqual(
            compute_state_fingerprint(self.root, ["pkg/core.py"]),
            compute_state_fingerprint(self.root, ["pkg\\core.py"]),
        )

    def test_missing_file_participates_rather_than_being_skipped(self):
        with_missing = compute_state_fingerprint(self.root, ["pkg/core.py", "pkg/gone.py"])
        without = compute_state_fingerprint(self.root, ["pkg/core.py"])
        self.assertNotEqual(with_missing, without)

    def test_a_file_appearing_later_invalidates_the_fingerprint(self):
        before = compute_state_fingerprint(self.root, ["pkg/new.py"])
        self.write("pkg/new.py", "x = 1\n")
        self.assertNotEqual(before, compute_state_fingerprint(self.root, ["pkg/new.py"]))

    def test_empty_path_set_is_stable(self):
        self.assertEqual(
            compute_state_fingerprint(self.root, []),
            compute_state_fingerprint(self.root, []),
        )

    def test_path_traversal_attempt_does_not_escape_or_crash(self):
        # A traversal path simply hashes as missing/whatever is there; the point
        # is that it neither raises nor silently matches the in-tree fingerprint.
        traversal = compute_state_fingerprint(self.root, ["../../../etc/passwd"])
        self.assertIsInstance(traversal, str)
        self.assertNotEqual(traversal, compute_state_fingerprint(self.root, ["pkg/core.py"]))


# ===========================================================================
# 12. Knowledge graph interaction
# ===========================================================================


class _FakeFileNode:
    def __init__(self, content_hash: str, dependents: list[str], validation_commands: list[str] | None = None):
        self.content_hash = content_hash
        self.dependents = dependents
        self.validation_commands = validation_commands or []


class _FakeFailurePattern:
    def __init__(self, pattern_id: str, signature: str, affected: list[str]):
        self.pattern_id = pattern_id
        self.error_signature = signature
        self.affected_files = affected


class _FakeGraph:
    def __init__(self, files: dict[str, Any], failure_patterns: list[Any] | None = None):
        self.files = files
        self.failure_patterns = failure_patterns or []


class _FakeKnowledgeManager:
    def __init__(self, graph: Any):
        self._graph = graph

    def get_graph(self):
        if self._graph is None:
            raise OSError("knowledge store unreadable")
        return self._graph


class TestKnowledgeIntegration(ProjectCase):
    def base_report(self) -> ChangeImpactReport:
        base = self.base_contents("pkg/core.py")
        self.write("pkg/core.py", CORE_PY.replace("return 41", "return 40"))
        return self.analyzer().analyze(["pkg/core.py"], base_contents=base)

    def current_hash(self, relative: str) -> str:
        return hashlib.sha256((self.root / relative).read_bytes()).hexdigest()

    def test_no_manager_leaves_report_untouched(self):
        report = self.base_report()
        before = report.to_dict()
        apply_knowledge_support(report, None, root=self.root)
        self.assertEqual(report.to_dict(), before)

    def test_fresh_knowledge_adds_supporting_notes(self):
        report = self.base_report()
        graph = _FakeGraph({"pkg/core.py": _FakeFileNode(self.current_hash("pkg/core.py"), ["pkg/helpers.py"])})
        apply_knowledge_support(report, _FakeKnowledgeManager(graph), root=self.root)
        self.assertTrue(any("historical dependent" in n for n in report.knowledge_notes))

    def test_stale_knowledge_is_ignored_and_escalates_scope(self):
        report = self.base_report()
        report.recommended_scope = SCOPE_TARGETED
        graph = _FakeGraph({"pkg/core.py": _FakeFileNode("deadbeef", ["pkg/helpers.py"])})
        apply_knowledge_support(report, _FakeKnowledgeManager(graph), root=self.root)
        self.assertTrue(any("stale knowledge" in n for n in report.knowledge_notes))
        self.assertEqual(report.recommended_scope, SCOPE_EXPANDED)

    def test_stale_knowledge_does_not_contribute_dependents(self):
        report = self.base_report()
        graph = _FakeGraph({"pkg/core.py": _FakeFileNode("deadbeef", ["pkg/ghost.py"])})
        apply_knowledge_support(report, _FakeKnowledgeManager(graph), root=self.root)
        self.assertFalse(any("pkg/ghost.py" in n and "historical" in n for n in report.knowledge_notes))

    def test_knowledge_never_raises_confidence(self):
        report = self.base_report()
        before = report.confidence
        graph = _FakeGraph({"pkg/core.py": _FakeFileNode(self.current_hash("pkg/core.py"), ["pkg/helpers.py"])})
        apply_knowledge_support(report, _FakeKnowledgeManager(graph), root=self.root)
        self.assertEqual(report.confidence, before)

    def test_knowledge_never_narrows_scope(self):
        report = self.base_report()
        report.recommended_scope = SCOPE_BROAD
        graph = _FakeGraph({"pkg/core.py": _FakeFileNode("stale", [])})
        apply_knowledge_support(report, _FakeKnowledgeManager(graph), root=self.root)
        self.assertEqual(report.recommended_scope, SCOPE_BROAD)

    def test_failure_patterns_overlapping_the_change_are_surfaced(self):
        report = self.base_report()
        graph = _FakeGraph({}, [_FakeFailurePattern("p1", "ImportError: pkg", ["pkg/core.py"])])
        apply_knowledge_support(report, _FakeKnowledgeManager(graph), root=self.root)
        self.assertTrue(any("recurring failure pattern" in n for n in report.knowledge_notes))

    def test_failure_patterns_not_overlapping_are_ignored(self):
        report = self.base_report()
        graph = _FakeGraph({}, [_FakeFailurePattern("p1", "boom", ["other/file.py"])])
        apply_knowledge_support(report, _FakeKnowledgeManager(graph), root=self.root)
        self.assertFalse(any("recurring failure pattern" in n for n in report.knowledge_notes))

    def test_unreadable_knowledge_store_degrades_to_a_note(self):
        report = self.base_report()
        apply_knowledge_support(report, _FakeKnowledgeManager(None), root=self.root)
        self.assertTrue(any("unavailable" in n for n in report.knowledge_notes))

    def test_notes_are_bounded(self):
        report = self.base_report()
        graph = _FakeGraph(
            {"pkg/core.py": _FakeFileNode(self.current_hash("pkg/core.py"), [f"d{i}.py" for i in range(50)])}
        )
        before = len(report.knowledge_notes)
        apply_knowledge_support(report, _FakeKnowledgeManager(graph), root=self.root, max_notes=2)
        self.assertLessEqual(len(report.knowledge_notes) - before, 2)


# ===========================================================================
# 13. Configuration
# ===========================================================================


class TestConfiguration(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="p417cfg_")).resolve()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def config(self, **overrides: Any) -> AgentConfig:
        return AgentConfig.from_environment(self.root, **overrides)

    def test_defaults_keep_phase_417_off(self):
        config = self.config()
        self.assertFalse(config.semantic_impact_analysis_enabled)
        self.assertFalse(config.reuse_candidate_validation_evidence)

    def test_default_bounds(self):
        config = self.config()
        self.assertEqual(config.max_impact_depth, 3)
        self.assertEqual(config.max_affected_symbols, 200)
        self.assertEqual(config.max_affected_tests, 8)
        self.assertEqual(config.validation_confidence_threshold, "high")

    def test_overrides_are_honoured(self):
        config = self.config(
            semantic_impact_analysis_enabled=True,
            max_impact_depth=5,
            max_affected_tests=2,
            validation_confidence_threshold="medium",
            reuse_candidate_validation_evidence=True,
        )
        config.validate()
        self.assertTrue(config.semantic_impact_analysis_enabled)
        self.assertEqual(config.max_impact_depth, 5)
        self.assertEqual(config.max_affected_tests, 2)
        self.assertEqual(config.validation_confidence_threshold, "medium")
        self.assertTrue(config.reuse_candidate_validation_evidence)

    def test_environment_variables_are_read(self):
        os.environ["AGENT_SEMANTIC_IMPACT_ANALYSIS"] = "true"
        os.environ["AGENT_MAX_IMPACT_DEPTH"] = "7"
        self.addCleanup(os.environ.pop, "AGENT_SEMANTIC_IMPACT_ANALYSIS", None)
        self.addCleanup(os.environ.pop, "AGENT_MAX_IMPACT_DEPTH", None)
        config = self.config()
        self.assertTrue(config.semantic_impact_analysis_enabled)
        self.assertEqual(config.max_impact_depth, 7)

    def test_zero_bound_is_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            self.config(max_impact_depth=0)

    def test_negative_bound_is_rejected(self):
        with self.assertRaises(ValueError):
            self.config(max_affected_tests=-1)

    def test_invalid_confidence_threshold_is_rejected_by_validate(self):
        config = self.config()
        config.validation_confidence_threshold = "certain"
        with self.assertRaises(ValueError):
            config.validate()

    def test_validate_rejects_non_positive_new_bounds(self):
        for name in ("max_impact_depth", "max_affected_symbols", "max_affected_tests"):
            config = self.config()
            setattr(config, name, 0)
            with self.assertRaises(ValueError, msg=name):
                config.validate()

    def test_validate_regression_guard_default_config_is_valid(self):
        # Phase 4.15's WIP once deleted the positive-int guard, which made
        # validate() raise for every configuration. This asserts the happy path
        # stays happy so that class of regression cannot return silently.
        self.config().validate()

    def test_validate_regression_guard_all_bounds_positive_by_default(self):
        config = self.config()
        for name in (
            "max_context_files", "max_context_file_bytes", "max_context_tokens",
            "max_tool_steps", "max_implementation_tool_steps", "max_candidate_iterations",
            "candidate_validation_timeout_seconds", "max_impact_depth",
            "max_affected_symbols", "max_affected_tests",
        ):
            self.assertGreaterEqual(getattr(config, name), 1, name)

    def test_cli_flags_are_registered_and_parsed(self):
        parser = argparse.ArgumentParser()
        add_common_arguments(parser)
        args = parser.parse_args([
            "--project", str(self.root),
            "--semantic-impact-analysis", "true",
            "--max-impact-depth", "4",
            "--max-affected-symbols", "50",
            "--max-affected-tests", "3",
            "--validation-confidence-threshold", "medium",
            "--reuse-candidate-evidence", "true",
        ])
        config = config_from_args(args)
        self.assertTrue(config.semantic_impact_analysis_enabled)
        self.assertEqual(config.max_impact_depth, 4)
        self.assertEqual(config.max_affected_symbols, 50)
        self.assertEqual(config.max_affected_tests, 3)
        self.assertEqual(config.validation_confidence_threshold, "medium")
        self.assertTrue(config.reuse_candidate_validation_evidence)

    def test_cli_defaults_leave_the_feature_off(self):
        parser = argparse.ArgumentParser()
        add_common_arguments(parser)
        config = config_from_args(parser.parse_args(["--project", str(self.root)]))
        self.assertFalse(config.semantic_impact_analysis_enabled)
        self.assertFalse(config.reuse_candidate_validation_evidence)

    def test_invalid_threshold_choice_is_rejected_by_argparse(self):
        parser = argparse.ArgumentParser()
        add_common_arguments(parser)
        with self.assertRaises(SystemExit):
            parser.parse_args(["--validation-confidence-threshold", "certain"])


# ===========================================================================
# 14. Candidate-sandbox integration (real subprocesses)
# ===========================================================================


class TestProspectiveValidatorIntegration(ProjectCase):
    def setUp(self) -> None:
        super().setUp()
        self.workspaces: list[CandidateWorkspace] = []

    def tearDown(self) -> None:
        for workspace in self.workspaces:
            workspace.cleanup()

    def workspace(self, **kwargs: Any) -> CandidateWorkspace:
        workspace = CandidateWorkspace(self.root, **kwargs)
        self.workspaces.append(workspace)
        return workspace

    def plan(self, *paths: str) -> Plan:
        return Plan(
            objective="edit",
            files_to_inspect=list(paths),
            files_likely_to_change=list(paths),
            files_likely_to_create=[],
            steps=["edit"],
            validation_strategy=["pytest"],
            risks=[],
        )

    def test_base_contents_returns_pre_change_text(self):
        ws = self.workspace().setup()
        ws.rebuild([FileOperation("update", "pkg/widget.py", "def render():\n    return 'x'\n")], self.plan("pkg/widget.py"))
        contents = ws.base_contents(["pkg/widget.py"])
        self.assertEqual(contents["pkg/widget.py"], WIDGET_PY)

    def test_base_contents_marks_created_files_as_none(self):
        ws = self.workspace().setup()
        ws.rebuild([FileOperation("create", "pkg/brand_new.py", "x = 1\n")], self.plan("pkg/brand_new.py"))
        self.assertIsNone(ws.base_contents(["pkg/brand_new.py"])["pkg/brand_new.py"])

    def test_base_contents_omits_unknown_paths(self):
        ws = self.workspace().setup()
        self.assertEqual(ws.base_contents(["pkg/never_touched.py"]), {})

    def test_semantic_targeting_produces_explained_commands(self):
        ws = self.workspace().setup()
        changed = ws.rebuild(
            [FileOperation("update", "pkg/core.py", CORE_PY.replace("return 41", "return 40"))],
            self.plan("pkg/core.py"),
        )
        validator = ProspectiveValidator(semantic_impact_enabled=True)
        report = validator.validate(ws, changed, None)
        self.assertTrue(validator.last_impact_report is not None)
        self.assertIn(report.impact_confidence, {CONFIDENCE_LOW, CONFIDENCE_MEDIUM, CONFIDENCE_HIGH})
        self.assertTrue(report.impact_summary)

    def test_semantic_targeting_runs_real_subprocesses(self):
        ws = self.workspace().setup()
        changed = ws.rebuild(
            [FileOperation("update", "pkg/core.py", CORE_PY.replace("return _bootstrap() + 1", "return _bootstrap() + 2"))],
            self.plan("pkg/core.py"),
        )
        validator = ProspectiveValidator(semantic_impact_enabled=True, enable_static_analysis=False)
        report = validator.validate(ws, changed, None)
        # ``Engine().start()`` now returns 43, so the real pytest run must fail.
        self.assertFalse(report.passed)
        self.assertEqual(report.failed_tier, "targeted_tests")
        self.assertTrue(any("pytest" in " ".join(r.command) for r in report.executed_results))

    def test_correct_change_passes_real_validation(self):
        ws = self.workspace().setup()
        changed = ws.rebuild(
            [FileOperation("update", "pkg/widget.py", "def render():\n    return 'widget2'\n")],
            self.plan("pkg/widget.py"),
        )
        validator = ProspectiveValidator(semantic_impact_enabled=True, enable_static_analysis=False)
        report = validator.validate(ws, changed, None)
        self.assertTrue(report.passed, report.render_feedback())

    def test_disabled_semantic_analysis_keeps_phase_416_behaviour(self):
        ws = self.workspace().setup()
        changed = ws.rebuild(
            [FileOperation("update", "pkg/core.py", CORE_PY.replace("return 41", "return 40"))],
            self.plan("pkg/core.py"),
        )
        validator = ProspectiveValidator(semantic_impact_enabled=False)
        report = validator.validate(ws, changed, None)
        self.assertIsNone(validator.last_impact_report)
        self.assertEqual(report.impact_confidence, "")

    def test_scope_budget_multiplier_widens_with_uncertainty(self):
        validator = ProspectiveValidator(max_targeted_commands=2, semantic_impact_enabled=True)
        self.assertEqual(validator._SCOPE_BUDGET_MULTIPLIER[SCOPE_TARGETED], 1)
        self.assertEqual(validator._SCOPE_BUDGET_MULTIPLIER[SCOPE_EXPANDED], 2)
        self.assertEqual(validator._SCOPE_BUDGET_MULTIPLIER[SCOPE_BROAD], 3)

    def test_candidate_validation_never_touches_the_authoritative_tree(self):
        before = self.snapshot()
        ws = self.workspace().setup()
        changed = ws.rebuild(
            [FileOperation("update", "pkg/core.py", CORE_PY.replace("return 41", "return 40"))],
            self.plan("pkg/core.py"),
        )
        ProspectiveValidator(semantic_impact_enabled=True).validate(ws, changed, None)
        self.assertEqual(self.snapshot(), before)

    def test_analysis_runs_against_candidate_root_not_base(self):
        ws = self.workspace().setup()
        changed = ws.rebuild(
            [FileOperation("create", "pkg/candidate_only.py", "def only_here():\n    return 1\n")],
            self.plan(*[]) if False else Plan(
                objective="create", files_to_inspect=[], files_likely_to_change=[],
                files_likely_to_create=["pkg/candidate_only.py"], steps=["x"],
                validation_strategy=[], risks=[],
            ),
        )
        validator = ProspectiveValidator(semantic_impact_enabled=True)
        validator.validate(ws, changed, None)
        impact = validator.last_impact_report
        self.assertIsNotNone(impact)
        self.assertEqual(impact.changed_files, ["pkg/candidate_only.py"])
        self.assertTrue(any(s.name == "only_here" for s in impact.added_symbols))
        self.assertFalse((self.root / "pkg" / "candidate_only.py").exists())

    def test_impact_analysis_failure_falls_back_to_lexical_not_to_nothing(self):
        ws = self.workspace().setup()
        changed = ws.rebuild(
            [FileOperation("update", "pkg/widget.py", "def render():\n    return 'w2'\n")],
            self.plan("pkg/widget.py"),
        )
        validator = ProspectiveValidator(semantic_impact_enabled=True, enable_static_analysis=False)
        validator.analyze_impact = lambda *a, **k: None  # simulate total analysis failure
        report = validator.validate(ws, changed, None)
        self.assertIn("syntax", report.tiers_run)


# ===========================================================================
# 15. Concurrency & isolation
# ===========================================================================


class TestConcurrentIsolation(unittest.TestCase):
    def setUp(self) -> None:
        self.roots = []
        for _ in range(3):
            root = Path(tempfile.mkdtemp(prefix="p417c_")).resolve()
            make_project(root)
            self.roots.append(root)
            self.addCleanup(shutil.rmtree, root, ignore_errors=True)

    def test_analyzers_in_threads_do_not_interfere(self):
        # Give each tree a distinct extra module so a leak between analyzers
        # would show up as the wrong module name in the wrong result.
        for index, root in enumerate(self.roots):
            (root / "pkg" / f"unique_{index}.py").write_text(
                f"def marker_{index}():\n    return {index}\n", encoding="utf-8"
            )

        def run(index: int) -> set[str]:
            analyzer = SemanticChangeImpactAnalyzer(self.roots[index])
            report = analyzer.analyze(
                [f"pkg/unique_{index}.py"],
                base_contents={f"pkg/unique_{index}.py": None},
            )
            return {s.name for s in report.added_symbols}

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(run, range(3)))
        for index, names in enumerate(results):
            self.assertEqual(names, {f"marker_{index}"})

    def test_graphs_built_concurrently_stay_rooted_at_their_own_tree(self):
        def build(root: Path) -> str:
            return str(SemanticGraph.build(root).root)

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            roots = list(pool.map(build, self.roots))
        self.assertEqual(roots, [str(r) for r in self.roots])

    def test_ledgers_are_instance_scoped(self):
        a, b = EvidenceLedger(), EvidenceLedger()
        a.record(command=("pytest",), status=STATUS_PASSED)
        self.assertEqual(len(a), 1)
        self.assertEqual(len(b), 0)

    def test_process_working_directory_is_never_changed(self):
        before = os.getcwd()
        for root in self.roots:
            SemanticChangeImpactAnalyzer(root).analyze(
                ["pkg/core.py"], base_contents={"pkg/core.py": CORE_PY}
            )
        self.assertEqual(os.getcwd(), before)


# ===========================================================================
# 16. Adversarial inputs
# ===========================================================================


class TestAdversarialInputs(ProjectCase):
    def test_path_traversal_in_changed_files_does_not_escape(self):
        report = self.analyzer().analyze(
            ["../../../etc/passwd"], base_contents={"../../../etc/passwd": None}
        )
        self.assertEqual(report.unsupported_files, ["../../../etc/passwd"])
        self.assertEqual(report.recommended_scope, SCOPE_BROAD)

    def test_traversal_path_ending_in_py_is_reported_unreadable_not_analysed(self):
        report = self.analyzer().analyze(
            ["../outside.py"], base_contents={"../outside.py": None}
        )
        self.assertEqual(report.recommended_scope, SCOPE_BROAD)
        self.assertEqual(report.added_symbols, [])

    def test_windows_style_changed_paths_are_normalised(self):
        base = {"pkg/core.py": CORE_PY}
        report = self.analyzer().analyze(["pkg\\core.py"], base_contents=base)
        self.assertEqual(report.changed_files, ["pkg/core.py"])

    def test_binary_file_named_py_is_reported_not_crashed(self):
        (self.root / "pkg" / "blob.py").write_bytes(b"\x00\x01\x02\xff")
        report = self.analyzer().analyze(["pkg/blob.py"], base_contents={"pkg/blob.py": None})
        self.assertIn("pkg/blob.py", report.unparseable_files)

    def test_empty_python_file_is_valid_with_no_symbols(self):
        self.write("pkg/empty.py", "")
        report = self.analyzer().analyze(["pkg/empty.py"], base_contents={"pkg/empty.py": None})
        self.assertEqual(report.added_symbols, [])
        self.assertEqual(report.unparseable_files, {})

    def test_init_py_change_propagates_to_package_importers(self):
        self.write("pkg/uses_package.py", "import pkg\n\n\ndef f():\n    return pkg.VERSION\n")
        base = self.base_contents("pkg/__init__.py")
        self.write("pkg/__init__.py", INIT_PY.replace('"1.0"', '"2.0"'))
        report = self.analyzer().analyze(["pkg/__init__.py"], base_contents=base)
        self.assertIn("pkg/uses_package.py", report.affected_files)

    def test_duplicate_symbol_names_do_not_cross_contaminate_direct_matches(self):
        self.write("pkg/other.py", "def render():\n    return 'other'\n")
        self.write("tests/test_other.py", "from pkg.other import render\n\n\ndef test_o():\n    assert render()\n")
        base = self.base_contents("pkg/widget.py")
        self.write("pkg/widget.py", "def render():\n    return 'w2'\n")
        report = self.analyzer().analyze(["pkg/widget.py"], base_contents=base)
        by_path = {t.path: t for t in report.validation_targets}
        # tests/test_other.py imports pkg/other.py, not the changed pkg/widget.py,
        # so it must not be a direct match.
        other = by_path.get("tests/test_other.py")
        self.assertTrue(other is None or other.tier not in {TIER_DIRECT_SYMBOL, TIER_DIRECT_IMPORT})

    def test_symlink_is_skipped_rather_than_followed(self):
        link = self.root / "pkg" / "linked.py"
        try:
            link.symlink_to(self.root / "pkg" / "core.py")
        except (OSError, NotImplementedError):
            self.skipTest("symlink creation not permitted on this platform")
        graph = SemanticGraph.build(self.root)
        self.assertNotIn("pkg/linked.py", graph.files)

    def test_deeply_nested_literal_does_not_crash_the_indexer(self):
        self.write("pkg/deep.py", "x = " + ("[" * 120) + ("]" * 120) + "\n")
        facts = AstPythonIndexer().analyze((self.root / "pkg" / "deep.py").read_bytes())
        self.assertIsInstance(facts.parse_error, str)

    def test_unicode_identifiers_are_handled(self):
        self.write("pkg/unicode_mod.py", "def caf\u00e9():\n    return 1\n")
        report = self.analyzer().analyze(
            ["pkg/unicode_mod.py"], base_contents={"pkg/unicode_mod.py": None}
        )
        self.assertEqual([s.name for s in report.added_symbols], ["caf\u00e9"])

    def test_generated_file_with_no_test_is_broad_scoped(self):
        self.write("pkg/generated_pb2.py", "def _serialized():\n    return 1\n")
        report = self.analyzer().analyze(
            ["pkg/generated_pb2.py"], base_contents={"pkg/generated_pb2.py": None}
        )
        self.assertEqual(report.recommended_scope, SCOPE_BROAD)

    def test_mixed_supported_and_unsupported_change_degrades(self):
        self.write("app.ts", "export const x = 1;\n")
        base = self.base_contents("pkg/widget.py")
        base["app.ts"] = None
        self.write("pkg/widget.py", "def render():\n    return 'w2'\n")
        report = self.analyzer().analyze(["pkg/widget.py", "app.ts"], base_contents=base)
        self.assertEqual(report.unsupported_files, ["app.ts"])
        self.assertLess(report.graph_coverage, 1.0)
        self.assertEqual(report.recommended_scope, SCOPE_BROAD)

    def test_many_changed_files_stay_within_bounds(self):
        base: dict[str, str | None] = {}
        changed = []
        for index in range(40):
            path = f"pkg/bulk{index}.py"
            self.write(path, f"def bulk_{index}():\n    return {index}\n")
            base[path] = None
            changed.append(path)
        report = self.analyzer(max_affected_tests=5).analyze(changed, base_contents=base)
        self.assertLessEqual(len(report.validation_targets), 5)
        self.assertEqual(report.recommended_scope, SCOPE_BROAD)


# ===========================================================================
# 17. True end-to-end pipeline (only the provider is mocked)
# ===========================================================================


class _E2EProvider(AIProvider):
    """Scripted tool-use provider: proposes a bad edit, then a good one."""

    def __init__(self, responses: list[Any]):
        super().__init__()
        self.provider_id = "e2e"
        self.model = "e2e-v1"
        self.responses = list(responses)
        self.calls = 0
        self.seen_feedback: list[str] = []

    @property
    def capabilities(self) -> set[ProviderCapability]:
        return {
            ProviderCapability.PLANNING,
            ProviderCapability.IMPLEMENTATION,
            ProviderCapability.REPAIR,
            ProviderCapability.REVIEW,
            ProviderCapability.TOOL_USE,
        }

    def generate_plan(self, task, context):  # pragma: no cover - unused here
        raise NotImplementedError

    def generate_code(self, task, plan, context, failure=None, review=None):
        return self.responses[-1]

    def generate_code_with_tools(self, task, plan, context, tools, tool_history=None, failure=None, review=None):
        self.calls += 1
        for _, result in list(tool_history or []):
            if result.tool_name == "candidate_validation":
                self.seen_feedback.append(result.output)
        return self.responses.pop(0) if self.responses else []

    def review_changes(self, task, plan, diff, context):
        return ReviewResult("APPROVED", "ok", [])

    def analyze_failure(self, execution, diff, context, plan):  # pragma: no cover
        raise NotImplementedError


class TestEndToEndPipeline(ProjectCase):
    """candidate change -> real impact analysis -> real target selection ->
    real subprocess validation -> real evidence -> apply -> reuse/invalidate."""

    def setUp(self) -> None:
        super().setUp()
        self.workspaces: list[CandidateWorkspace] = []

    def tearDown(self) -> None:
        for workspace in self.workspaces:
            workspace.cleanup()

    def make_agent(self, responses: list[Any], **kwargs: Any):
        filesystem = ProjectFilesystem(self.root)
        registry = ToolRegistry(self.root, filesystem=filesystem)
        workspace = CandidateWorkspace(self.root)
        self.workspaces.append(workspace)
        validator = ProspectiveValidator(
            semantic_impact_enabled=True, enable_static_analysis=False
        )
        agent = InteractiveCodingAgent(
            filesystem=filesystem,
            registry=registry,
            sandbox=workspace,
            validator=validator,
            cleanup_sandbox=False,
            **kwargs,
        )
        return agent, _E2EProvider(responses), filesystem

    def plan(self, *paths: str) -> Plan:
        return Plan(
            objective="change widget",
            files_to_inspect=list(paths),
            files_likely_to_change=list(paths),
            files_likely_to_create=[],
            steps=["edit"],
            validation_strategy=["pytest"],
            risks=[],
        )

    def context(self) -> ProjectContext:
        return ProjectContext(root=str(self.root))

    def test_full_loop_bad_candidate_then_good_candidate(self):
        # tests/test_core.py asserts Engine().start() == 42.
        bad = [FileOperation("update", "pkg/core.py", CORE_PY.replace("return 41", "return 1"))]
        good = [FileOperation("update", "pkg/core.py", CORE_PY.replace("return 41", "return 41"))]
        agent, provider, _ = self.make_agent([bad, good], max_candidate_iterations=2)
        before = self.snapshot()

        result = agent.execute(
            provider=provider,
            task_objective="keep start() == 42",
            plan=self.plan("pkg/core.py"),
            context=self.context(),
        )

        # 1. The loop genuinely observed a real failing subprocess and refined.
        self.assertGreaterEqual(provider.calls, 2)
        self.assertTrue(provider.seen_feedback)
        self.assertIn("Candidate validation FAILED", provider.seen_feedback[0])

        # 2. The impact analysis really ran and really picked semantic targets.
        self.assertTrue(result.semantic_impact_used)
        self.assertGreater(result.impact_tests_considered, 0)
        self.assertGreater(result.impact_semantic_targets, 0)
        self.assertIn(result.impact_confidence, {CONFIDENCE_LOW, CONFIDENCE_MEDIUM, CONFIDENCE_HIGH})
        self.assertTrue(result.impact_report)

        # 3. Real evidence was recorded across both candidate iterations.
        self.assertGreaterEqual(len(agent.evidence_ledger), 2)
        self.assertEqual(agent.evidence_ledger.iterations, [1, 2])
        self.assertTrue(any(e.status == STATUS_FAILED for e in agent.evidence_ledger.entries))
        self.assertTrue(any(e.status == STATUS_PASSED for e in agent.evidence_ledger.entries))

        # 4. Every evidence entry explains itself.
        for entry in agent.evidence_ledger.entries:
            self.assertTrue(entry.selected_because.strip())
            self.assertTrue(entry.fingerprint)

        # 5. The authoritative tree is still untouched: only FileOperations were
        #    produced; nothing was written outside the candidate.
        self.assertEqual(self.snapshot(), before)
        self.assertTrue(result.success)
        self.assertIsNotNone(result.file_operations)

    def test_evidence_is_reusable_after_applying_the_validated_operations(self):
        good = [FileOperation("update", "pkg/widget.py", "def render():\n    return 'widget2'\n")]
        agent, provider, filesystem = self.make_agent([good])
        result = agent.execute(
            provider=provider,
            task_objective="change widget",
            plan=self.plan("pkg/widget.py"),
            context=self.context(),
        )
        self.assertTrue(result.success)
        passing = [
            entry for entry in agent.evidence_ledger.entries
            if entry.status == STATUS_PASSED and entry.command[0] == "pytest"
        ]
        if not passing:
            self.skipTest("no passing pytest evidence was produced in this environment")
        entry = passing[0]

        # Apply for real through the unmodified pipeline.
        coding_agent = CodingAgent(filesystem)
        changed = coding_agent.apply(result.file_operations, self.plan("pkg/widget.py"))
        self.assertEqual(changed, ["pkg/widget.py"])

        decision = agent.evidence_ledger.find_reusable(
            command=entry.command,
            current_root=self.root,
            relevant_files=entry.impacted_files,
            relevant_symbols=entry.impacted_symbols,
            min_confidence=entry.confidence,
        )
        self.assertTrue(decision.reusable, decision.reason)
        self.assertEqual(decision.reason, REASON_OK)

    def test_evidence_is_invalidated_when_the_applied_tree_diverges(self):
        good = [FileOperation("update", "pkg/widget.py", "def render():\n    return 'widget2'\n")]
        agent, provider, filesystem = self.make_agent([good])
        result = agent.execute(
            provider=provider,
            task_objective="change widget",
            plan=self.plan("pkg/widget.py"),
            context=self.context(),
        )
        passing = [
            entry for entry in agent.evidence_ledger.entries
            if entry.status == STATUS_PASSED and entry.command[0] == "pytest"
        ]
        if not passing:
            self.skipTest("no passing pytest evidence was produced in this environment")
        entry = passing[0]

        CodingAgent(filesystem).apply(result.file_operations, self.plan("pkg/widget.py"))
        # Something else edits the tree after the candidate was validated.
        (self.root / "pkg" / "widget.py").write_text(
            "def render():\n    return 'tampered'\n", encoding="utf-8"
        )

        decision = agent.evidence_ledger.find_reusable(
            command=entry.command,
            current_root=self.root,
            relevant_files=entry.impacted_files,
            relevant_symbols=entry.impacted_symbols,
            min_confidence=entry.confidence,
        )
        self.assertFalse(decision.reusable)
        self.assertEqual(decision.reason, REASON_FINGERPRINT_MISMATCH)

    def test_mode_c_without_semantic_analysis_still_works(self):
        """Phase 4.16 mode C must be unaffected by Phase 4.17."""
        good = [FileOperation("update", "pkg/widget.py", "def render():\n    return 'widget2'\n")]
        filesystem = ProjectFilesystem(self.root)
        workspace = CandidateWorkspace(self.root)
        self.workspaces.append(workspace)
        agent = InteractiveCodingAgent(
            filesystem=filesystem,
            registry=ToolRegistry(self.root, filesystem=filesystem),
            sandbox=workspace,
            validator=ProspectiveValidator(semantic_impact_enabled=False),
            cleanup_sandbox=False,
        )
        result = agent.execute(
            provider=_E2EProvider([good]),
            task_objective="change widget",
            plan=self.plan("pkg/widget.py"),
            context=self.context(),
        )
        self.assertTrue(result.success)
        self.assertFalse(result.semantic_impact_used)
        self.assertEqual(result.impact_confidence, "")

    def test_mode_b_without_prospective_validation_still_works(self):
        """Phase 4.15 mode B: no sandbox at all, so no impact analysis."""
        good = [FileOperation("update", "pkg/widget.py", "def render():\n    return 'widget2'\n")]
        filesystem = ProjectFilesystem(self.root)
        agent = InteractiveCodingAgent(
            filesystem=filesystem,
            registry=ToolRegistry(self.root, filesystem=filesystem),
        )
        result = agent.execute(
            provider=_E2EProvider([good]),
            task_objective="change widget",
            plan=self.plan("pkg/widget.py"),
            context=self.context(),
        )
        self.assertTrue(result.success)
        self.assertFalse(result.prospective_validation_used)
        self.assertFalse(result.semantic_impact_used)
        self.assertEqual(len(agent.evidence_ledger), 0)


if __name__ == "__main__":
    unittest.main()
