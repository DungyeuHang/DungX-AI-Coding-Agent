"""Phase 4.17: semantic, dependency-aware change-impact analysis.

Relationship to :mod:`local_agent.impact`
-----------------------------------------
:class:`local_agent.impact.ChangeImpactAnalyzer` is *predictive*: it reads the
natural-language task description before any code exists and guesses which
files a change will touch. The machinery here is *retrospective and
structural*: given files that actually changed - normally inside a Phase 4.16
:class:`~local_agent.sandbox.CandidateWorkspace`, before anything is written to
the authoritative tree - it computes

    changed files -> changed symbols -> import/reference graph
                  -> affected symbols/modules -> relevant tests

with an explicit, deterministic confidence level and an adaptive validation
scope. The two share no inputs, outputs or failure modes, so they are separate
classes; this module is the single home for all diff-driven impact reasoning.

THE CENTRAL SAFETY RULE
-----------------------
Uncertainty may only ever *widen* validation. A parse failure, an unsupported
language, a missing index, a deleted or renamed file, a star/dynamic import or
a graph traversal that hit its bounds all lower confidence, and lower
confidence always maps to a broader scope. :func:`escalate_scope` is the only
way a scope is ever assigned after the initial value, and it is monotone: there
is no code path from "analysis failed" to "run less" or "skip validation".

Determinism
-----------
Every collection this module iterates is sorted before use and every traversal
is depth- and node-bounded, so the same repository state plus the same changed
file list always produces a byte-identical :class:`ChangeImpactReport` (modulo
the wall-clock ``analysis_seconds`` field, which is excluded from equality
comparisons by :meth:`ChangeImpactReport.fingerprint`).

Language support
----------------
Genuine graph analysis is Python-only, because that is the only language for
which this repository has a dependable parser (the Tree-sitter grammars are an
optional, usually-absent build artefact). Non-Python changes are recorded as
``unsupported_files``, which is a degradation: they force LOW confidence and a
BROAD scope rather than being silently treated as "no impact".
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .dependency_resolution import (
    ANNOTATION,
    ATTRIBUTE_RESOLUTION,
    DECORATOR,
    DYNAMIC_IMPORT_RESOLVED,
    INHERITANCE,
    REEXPORT,
    DependencyEvidence,
    make_evidence,
    resolve_alias_reference,
)
from .indexing.ast_python_indexer import (
    AstPythonIndexer,
    ImportRecord,
    PythonFileFacts,
    is_public_symbol,
    qualified_name,
)
from .models import FileIndex, SemanticIndex, SymbolDefinition
from .sandbox import EXCLUDED_DIRECTORY_NAMES

LOGGER = logging.getLogger(__name__)

#: Phase 4.18: bumped whenever a change here could alter a *previously
#: recorded* confidence/tier/scope conclusion for the same inputs (a new
#: evidence type, a changed confidence table, a changed scope policy). Not
#: bumped for additive fields that do not change existing conclusions (a new
#: dataclass field with a safe default, a new optional evidence tag). Recorded
#: on every :class:`~local_agent.evidence.ValidationEvidence` entry and
#: consulted, opt-in, by :meth:`~local_agent.evidence.EvidenceLedger.find_reusable`
#: so a build that changed how confidence is computed cannot silently reuse a
#: conclusion an older build reached under different rules.
SEMANTIC_ANALYZER_SCHEMA_VERSION = "4.18.0"

# -- vocabulary ---------------------------------------------------------------

#: Confidence levels, weakest first. Order is meaningful.
CONFIDENCE_LOW = "low"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_HIGH = "high"
CONFIDENCE_ORDER: tuple[str, ...] = (CONFIDENCE_LOW, CONFIDENCE_MEDIUM, CONFIDENCE_HIGH)

#: Validation scopes, narrowest first. Order is meaningful.
SCOPE_TARGETED = "targeted"
SCOPE_EXPANDED = "expanded"
SCOPE_BROAD = "broad"
SCOPE_ORDER: tuple[str, ...] = (SCOPE_TARGETED, SCOPE_EXPANDED, SCOPE_BROAD)

#: Evidence tiers for the test<->change association, strongest first. Weights
#: are small integers on purpose: every selection must be explainable to a human
#: in one sentence, so there is no opaque weighting or learned scoring here.
TIER_DIRECT_SYMBOL = "direct_symbol_match"
TIER_DIRECT_IMPORT = "direct_import_match"
TIER_CALL_GRAPH = "call_graph_match"
TIER_REVERSE_DEPENDENCY = "reverse_dependency_match"
TIER_MODULE = "module_match"
TIER_FILENAME = "filename_match"
TIER_BROAD = "broad_fallback"

TIER_WEIGHTS: dict[str, int] = {
    TIER_DIRECT_SYMBOL: 6,
    TIER_DIRECT_IMPORT: 5,
    TIER_CALL_GRAPH: 4,
    TIER_REVERSE_DEPENDENCY: 3,
    TIER_MODULE: 2,
    TIER_FILENAME: 1,
    TIER_BROAD: 0,
}

#: Tiers derived from the import/reference graph rather than from file naming.
SEMANTIC_TIERS: frozenset[str] = frozenset({
    TIER_DIRECT_SYMBOL, TIER_DIRECT_IMPORT, TIER_CALL_GRAPH, TIER_REVERSE_DEPENDENCY,
})

CHANGE_ADDED = "added"
CHANGE_REMOVED = "removed"
CHANGE_MODIFIED = "modified"

DEFAULT_MAX_IMPACT_DEPTH = 3
DEFAULT_MAX_AFFECTED_SYMBOLS = 200
DEFAULT_MAX_AFFECTED_TESTS = 8
#: Hard cap on files parsed for one graph build. Hitting it marks the graph
#: incomplete, which lowers confidence and widens scope.
DEFAULT_MAX_GRAPH_FILES = 4000
#: A change touching more modules than this counts as high fan-out and escalates
#: the scope even when confidence is HIGH.
HIGH_FANOUT_AFFECTED_FILES = 12
#: A symbol name defined in more than this many files is *ambiguous*: a bare
#: textual reference to it (``analyze``, ``run``, ``to_dict``) is not evidence
#: that the referring file exercises the changed definition. Ambiguous names are
#: therefore excluded from the reference-only ``call_graph_match`` tier. They
#: are still honoured at ``direct_symbol_match``, where a resolved import edge
#: corroborates the name, so this only ever removes *spurious* associations.
AMBIGUOUS_SYMBOL_DEFINITION_FILES = 3


def confidence_at_least(level: str, minimum: str) -> bool:
    """Whether ``level`` is at least as strong as ``minimum``."""
    if level not in CONFIDENCE_ORDER or minimum not in CONFIDENCE_ORDER:
        return False
    return CONFIDENCE_ORDER.index(level) >= CONFIDENCE_ORDER.index(minimum)


def escalate_scope(current: str, candidate: str) -> str:
    """Return whichever of the two scopes is broader.

    This is the *only* way a scope is reassigned after its initial value, which
    makes the "uncertainty never reduces validation" rule a property of the code
    rather than a convention reviewers have to police. Unknown values are
    treated as :data:`SCOPE_BROAD` - an unrecognised scope is itself a form of
    uncertainty and must not be allowed to narrow anything.
    """
    current_rank = SCOPE_ORDER.index(current) if current in SCOPE_ORDER else len(SCOPE_ORDER) - 1
    candidate_rank = (
        SCOPE_ORDER.index(candidate) if candidate in SCOPE_ORDER else len(SCOPE_ORDER) - 1
    )
    return SCOPE_ORDER[max(current_rank, candidate_rank)]


def is_python_path(path: str) -> bool:
    return str(path).lower().endswith(".py")


def looks_like_test_path(path: str) -> bool:
    """Filename-level test detection, matching the repository scanner's notion."""
    posix = Path(str(path)).as_posix().lower()
    stem = Path(posix).stem
    if any(part in {"tests", "test", "__tests__"} for part in posix.split("/")[:-1]):
        return True
    return (
        stem.startswith("test_")
        or stem.endswith("_test")
        or stem.endswith(".test")
        or stem.endswith(".spec")
    )


def module_name_for(relative_path: str) -> str | None:
    """Dotted module name for a repo-relative Python file path.

    ``a/b/c.py`` -> ``a.b.c``; ``a/b/__init__.py`` -> ``a.b``. Returns ``None``
    for a non-Python path or one whose components are not valid Python
    identifiers, since such a file cannot participate in a static import graph.
    """
    posix = Path(str(relative_path)).as_posix()
    if not posix.lower().endswith(".py"):
        return None
    parts = [part for part in posix[:-3].split("/") if part not in ("", ".")]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts:
        return None
    if not all(part.isidentifier() for part in parts):
        return None
    return ".".join(parts)


def normalize_relative(path: str) -> str:
    """Repo-relative POSIX form, with Windows separators and ``./`` folded away.

    Traversal (``..``) and absolute components are deliberately **preserved**.
    An escaping path must stay recognisable so :func:`escapes_root` can reject
    it; silently rewriting ``../../../etc/passwd`` into the innocuous-looking
    ``etc/passwd`` would turn an out-of-tree read into an in-tree one. Leading
    dots are likewise preserved, so ``.github/workflows/ci.yml`` keeps its
    directory instead of becoming ``github/workflows/ci.yml``.
    """
    text = str(path).replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    if not text:
        return str(path)
    return Path(text).as_posix()


def escapes_root(relative: str) -> bool:
    """Whether ``relative`` could resolve outside the tree it is joined to.

    Absolute paths, drive-qualified paths and any ``..`` component all qualify.
    Callers must treat such a path as *unsupported* rather than analysing it -
    the analyzer only ever reads inside the root it was given.
    """
    text = str(relative).replace("\\", "/")
    if not text or text.startswith("/"):
        return True
    if len(text) > 1 and text[1] == ":":
        return True
    return ".." in [part for part in text.split("/") if part]


# -- structured results -------------------------------------------------------


@dataclass
class ChangedSymbol:
    """One symbol added, removed or modified by a change."""

    qualified_name: str
    name: str
    file: str
    kind: str = "function"
    change: str = CHANGE_MODIFIED
    is_public: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "qualified_name": self.qualified_name,
            "name": self.name,
            "file": self.file,
            "kind": self.kind,
            "change": self.change,
            "is_public": self.is_public,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "ChangedSymbol":
        if not isinstance(data, dict):
            return cls(qualified_name="", name="", file="")
        return cls(
            qualified_name=str(data.get("qualified_name", "")),
            name=str(data.get("name", "")),
            file=str(data.get("file", "")),
            kind=str(data.get("kind", "function")),
            change=str(data.get("change", CHANGE_MODIFIED)),
            is_public=bool(data.get("is_public", False)),
        )


@dataclass
class AffectedSymbol:
    """A module/symbol reachable from a changed file through reverse dependencies."""

    qualified_name: str
    file: str
    depth: int = 1
    #: Human-readable dependency edge, e.g. ``b.py imports a.py``.
    via: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "qualified_name": self.qualified_name,
            "file": self.file,
            "depth": self.depth,
            "via": self.via,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "AffectedSymbol":
        if not isinstance(data, dict):
            return cls(qualified_name="", file="")
        return cls(
            qualified_name=str(data.get("qualified_name", "")),
            file=str(data.get("file", "")),
            depth=int(data.get("depth", 1) or 1),
            via=str(data.get("via", "")),
        )


@dataclass
class ValidationTarget:
    """A validation command selected because of specific, stated evidence."""

    path: str
    command: tuple[str, ...] = ()
    tier: str = TIER_BROAD
    #: One sentence a human can read to understand why this command ran.
    selected_because: str = ""
    matched_symbols: list[str] = field(default_factory=list)
    matched_files: list[str] = field(default_factory=list)
    depth: int = 0
    #: Phase 4.18: fine-grained, provenance-typed evidence explaining *why* this
    #: target was associated, layered on top of (never replacing) ``tier``. Can
    #: be empty even for a semantic tier - e.g. a plain bare-name match records
    #: no additional evidence beyond the tier itself, which already explains it.
    dependency_evidence: tuple["DependencyEvidence", ...] = ()

    @property
    def weight(self) -> int:
        return TIER_WEIGHTS.get(self.tier, 0)

    @property
    def is_semantic(self) -> bool:
        """True when this target came from the graph, not from file naming."""
        return self.tier in SEMANTIC_TIERS

    def sort_key(self) -> tuple[int, int, int, str]:
        """Total order used for ranking; deterministic for identical input."""
        return (-self.weight, self.depth, -len(self.matched_symbols), self.path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "command": list(self.command),
            "tier": self.tier,
            "selected_because": self.selected_because,
            "matched_symbols": list(self.matched_symbols),
            "matched_files": list(self.matched_files),
            "depth": self.depth,
            "dependency_evidence": [e.to_dict() for e in self.dependency_evidence],
        }

    @classmethod
    def from_dict(cls, data: Any) -> "ValidationTarget":
        if not isinstance(data, dict):
            return cls(path="")
        return cls(
            path=str(data.get("path", "")),
            command=tuple(str(token) for token in (data.get("command") or [])),
            tier=str(data.get("tier", TIER_BROAD)),
            selected_because=str(data.get("selected_because", "")),
            matched_symbols=[str(s) for s in (data.get("matched_symbols") or [])],
            matched_files=[str(s) for s in (data.get("matched_files") or [])],
            depth=int(data.get("depth", 0) or 0),
            dependency_evidence=tuple(
                DependencyEvidence.from_dict(e) for e in (data.get("dependency_evidence") or [])
            ),
        )


@dataclass
class ImpactEvidence:
    """Transparent, countable dimensions behind a confidence level.

    Every field is an integer count or a plain ratio produced by deterministic
    code in this module. Nothing here is learned, probabilistic, or weighted by
    opaque coefficients: given identical repository state and identical changed
    files the values are reproducible bit for bit.
    """

    direct_symbol_matches: int = 0
    direct_import_matches: int = 0
    call_graph_matches: int = 0
    reverse_dependency_matches: int = 0
    module_matches: int = 0
    filename_matches: int = 0
    #: Fraction (0..1) of changed files the graph could actually analyse.
    graph_coverage: float = 0.0
    #: Fraction (0..1) of import statements resolved to a file inside this repo.
    import_resolution: float = 0.0
    #: Reasons the analysis is known to be incomplete. Any entry forces LOW.
    degradations: list[str] = field(default_factory=list)

    @property
    def best_tier(self) -> str:
        for tier, count in (
            (TIER_DIRECT_SYMBOL, self.direct_symbol_matches),
            (TIER_DIRECT_IMPORT, self.direct_import_matches),
            (TIER_CALL_GRAPH, self.call_graph_matches),
            (TIER_REVERSE_DEPENDENCY, self.reverse_dependency_matches),
            (TIER_MODULE, self.module_matches),
            (TIER_FILENAME, self.filename_matches),
        ):
            if count > 0:
                return tier
        return TIER_BROAD

    @property
    def semantic_matches(self) -> int:
        return (
            self.direct_symbol_matches
            + self.direct_import_matches
            + self.call_graph_matches
            + self.reverse_dependency_matches
        )

    def assess(self) -> tuple[str, str]:
        """Return ``(confidence_level, human_readable_reason)``.

        Rules, in order; the first match wins:

        ``HIGH``
            No known degradation, at least 60% of the changed files analysed,
            and at least one *direct* association - a test that imports the
            changed module (optionally also naming a changed symbol).
        ``MEDIUM``
            No known degradation, at least 30% of the changed files analysed,
            and at least one association drawn from the graph at any tier.
        ``LOW``
            Everything else, and unconditionally whenever any degradation was
            recorded.

        LOW is the safe default on purpose: it maps to the broadest validation
        scope, so being unsure can only ever cost time, never coverage.
        """
        if self.degradations:
            unique = sorted(set(self.degradations))
            shown = "; ".join(unique[:4])
            if len(unique) > 4:
                shown += f"; (+{len(unique) - 4} more)"
            return CONFIDENCE_LOW, f"semantic analysis incomplete: {shown}"
        direct = self.direct_symbol_matches + self.direct_import_matches
        if direct > 0 and self.graph_coverage >= 0.6:
            return CONFIDENCE_HIGH, (
                f"direct import/symbol evidence at {self.graph_coverage:.0%} graph coverage "
                f"({self.direct_symbol_matches} test(s) import a changed module and reference a "
                f"changed symbol, {self.direct_import_matches} import a changed module)"
            )
        if self.semantic_matches > 0 and self.graph_coverage >= 0.3:
            return CONFIDENCE_MEDIUM, (
                f"{self.semantic_matches} graph-derived association(s), best tier "
                f"'{self.best_tier}', at {self.graph_coverage:.0%} graph coverage"
            )
        if self.semantic_matches > 0:
            return CONFIDENCE_LOW, (
                f"graph-derived associations exist but only {self.graph_coverage:.0%} of the "
                "changed files could be analysed"
            )
        if self.module_matches or self.filename_matches:
            return CONFIDENCE_LOW, (
                "no graph-derived association; falling back to module/filename heuristics"
            )
        return CONFIDENCE_LOW, "no association between the change and any known test was found"

    def to_dict(self) -> dict[str, Any]:
        return {
            "direct_symbol_matches": self.direct_symbol_matches,
            "direct_import_matches": self.direct_import_matches,
            "call_graph_matches": self.call_graph_matches,
            "reverse_dependency_matches": self.reverse_dependency_matches,
            "module_matches": self.module_matches,
            "filename_matches": self.filename_matches,
            "graph_coverage": round(self.graph_coverage, 4),
            "import_resolution": round(self.import_resolution, 4),
            "degradations": list(self.degradations),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "ImpactEvidence":
        if not isinstance(data, dict):
            return cls()
        return cls(
            direct_symbol_matches=int(data.get("direct_symbol_matches", 0) or 0),
            direct_import_matches=int(data.get("direct_import_matches", 0) or 0),
            call_graph_matches=int(data.get("call_graph_matches", 0) or 0),
            reverse_dependency_matches=int(data.get("reverse_dependency_matches", 0) or 0),
            module_matches=int(data.get("module_matches", 0) or 0),
            filename_matches=int(data.get("filename_matches", 0) or 0),
            graph_coverage=float(data.get("graph_coverage", 0.0) or 0.0),
            import_resolution=float(data.get("import_resolution", 0.0) or 0.0),
            degradations=[str(item) for item in (data.get("degradations") or [])],
        )


@dataclass
class ChangeImpactReport:
    """Structured, serialisable outcome of one semantic impact analysis.

    Backward compatibility: :meth:`from_dict` reads any subset of these keys and
    supplies defaults for the rest, so a checkpoint serialised before this phase
    (or by a future phase that adds fields) deserialises without error. All
    fields added here are additive; none is required.
    """

    changed_files: list[str] = field(default_factory=list)
    added_symbols: list[ChangedSymbol] = field(default_factory=list)
    removed_symbols: list[ChangedSymbol] = field(default_factory=list)
    modified_symbols: list[ChangedSymbol] = field(default_factory=list)
    affected_symbols: list[AffectedSymbol] = field(default_factory=list)
    affected_files: list[str] = field(default_factory=list)
    public_api_symbols: list[str] = field(default_factory=list)
    validation_targets: list[ValidationTarget] = field(default_factory=list)
    evidence: ImpactEvidence = field(default_factory=ImpactEvidence)
    confidence: str = CONFIDENCE_LOW
    confidence_reason: str = ""
    recommended_scope: str = SCOPE_BROAD
    scope_reasons: list[str] = field(default_factory=list)
    #: Symbols referenced by dependents that could not be resolved to a definition.
    unresolved_symbols: list[str] = field(default_factory=list)
    #: Changed files whose language this analysis cannot reason about.
    unsupported_files: list[str] = field(default_factory=list)
    #: Changed files that exist but could not be parsed, with the parser message.
    unparseable_files: dict[str, str] = field(default_factory=dict)
    graph_coverage: float = 0.0
    #: Which bound stopped the analysis short, if any (depth/symbols/tests/files).
    bounds_hit: list[str] = field(default_factory=list)
    tests_considered: int = 0
    analysis_seconds: float = 0.0
    #: Supporting-only observations sourced from the persistent knowledge graph.
    knowledge_notes: list[str] = field(default_factory=list)

    @property
    def changed_symbols(self) -> list[ChangedSymbol]:
        return list(self.added_symbols) + list(self.removed_symbols) + list(self.modified_symbols)

    @property
    def changed_symbol_names(self) -> set[str]:
        return {symbol.name for symbol in self.changed_symbols if symbol.name}

    @property
    def selected_targets(self) -> list[ValidationTarget]:
        return list(self.validation_targets)

    @property
    def is_degraded(self) -> bool:
        return bool(self.evidence.degradations)

    def fingerprint(self) -> str:
        """Stable digest of everything except wall-clock timing.

        Two analyses of the same repository state and the same changed files
        must produce the same fingerprint; the determinism tests assert exactly
        that by running the analyzer twice and comparing this value.
        """
        payload = self.to_dict()
        payload.pop("analysis_seconds", None)
        return hashlib.sha256(
            repr(sorted(payload.items(), key=lambda item: item[0])).encode("utf-8", "replace")
        ).hexdigest()

    def summary(self) -> str:
        """Compact, model- and human-facing description of the impact."""
        lines = [
            f"Semantic impact: {len(self.changed_files)} changed file(s), "
            f"{len(self.changed_symbols)} changed symbol(s), "
            f"{len(self.affected_files)} affected module(s).",
            f"Confidence: {self.confidence.upper()} - {self.confidence_reason}",
            f"Recommended validation scope: {self.recommended_scope.upper()}"
            + (f" ({'; '.join(self.scope_reasons)})" if self.scope_reasons else ""),
        ]
        if self.public_api_symbols:
            lines.append(
                "Public API affected: " + ", ".join(sorted(self.public_api_symbols)[:10])
            )
        for target in self.validation_targets[:6]:
            lines.append(f"  - {' '.join(target.command)} :: {target.selected_because}")
        if self.unsupported_files:
            lines.append(
                "Unsupported (not semantically analysed): "
                + ", ".join(sorted(self.unsupported_files)[:6])
            )
        for note in self.knowledge_notes[:3]:
            lines.append(f"  (knowledge) {note}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "changed_files": list(self.changed_files),
            "added_symbols": [s.to_dict() for s in self.added_symbols],
            "removed_symbols": [s.to_dict() for s in self.removed_symbols],
            "modified_symbols": [s.to_dict() for s in self.modified_symbols],
            "affected_symbols": [s.to_dict() for s in self.affected_symbols],
            "affected_files": list(self.affected_files),
            "public_api_symbols": list(self.public_api_symbols),
            "validation_targets": [t.to_dict() for t in self.validation_targets],
            "evidence": self.evidence.to_dict(),
            "confidence": self.confidence,
            "confidence_reason": self.confidence_reason,
            "recommended_scope": self.recommended_scope,
            "scope_reasons": list(self.scope_reasons),
            "unresolved_symbols": list(self.unresolved_symbols),
            "unsupported_files": list(self.unsupported_files),
            "unparseable_files": dict(self.unparseable_files),
            "graph_coverage": round(self.graph_coverage, 4),
            "bounds_hit": list(self.bounds_hit),
            "tests_considered": self.tests_considered,
            "analysis_seconds": round(self.analysis_seconds, 4),
            "knowledge_notes": list(self.knowledge_notes),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "ChangeImpactReport":
        """Tolerant deserialisation.

        Unknown keys are ignored and missing keys take their default, so state
        written by an older phase (which had none of these keys) or a newer one
        (which may add more) both load cleanly.
        """
        if not isinstance(data, dict):
            return cls()
        unparseable = data.get("unparseable_files") or {}
        if not isinstance(unparseable, dict):
            unparseable = {}
        return cls(
            changed_files=[str(p) for p in (data.get("changed_files") or [])],
            added_symbols=[ChangedSymbol.from_dict(s) for s in (data.get("added_symbols") or [])],
            removed_symbols=[
                ChangedSymbol.from_dict(s) for s in (data.get("removed_symbols") or [])
            ],
            modified_symbols=[
                ChangedSymbol.from_dict(s) for s in (data.get("modified_symbols") or [])
            ],
            affected_symbols=[
                AffectedSymbol.from_dict(s) for s in (data.get("affected_symbols") or [])
            ],
            affected_files=[str(p) for p in (data.get("affected_files") or [])],
            public_api_symbols=[str(s) for s in (data.get("public_api_symbols") or [])],
            validation_targets=[
                ValidationTarget.from_dict(t) for t in (data.get("validation_targets") or [])
            ],
            evidence=ImpactEvidence.from_dict(data.get("evidence")),
            confidence=str(data.get("confidence", CONFIDENCE_LOW)),
            confidence_reason=str(data.get("confidence_reason", "")),
            recommended_scope=str(data.get("recommended_scope", SCOPE_BROAD)),
            scope_reasons=[str(r) for r in (data.get("scope_reasons") or [])],
            unresolved_symbols=[str(s) for s in (data.get("unresolved_symbols") or [])],
            unsupported_files=[str(s) for s in (data.get("unsupported_files") or [])],
            unparseable_files={str(k): str(v) for k, v in unparseable.items()},
            graph_coverage=float(data.get("graph_coverage", 0.0) or 0.0),
            bounds_hit=[str(b) for b in (data.get("bounds_hit") or [])],
            tests_considered=int(data.get("tests_considered", 0) or 0),
            analysis_seconds=float(data.get("analysis_seconds", 0.0) or 0.0),
            knowledge_notes=[str(n) for n in (data.get("knowledge_notes") or [])],
        )


# -- the graph ----------------------------------------------------------------


class SemanticGraph:
    """Python import/reference graph for one tree, built from ``ast`` facts.

    This is not a new symbol index: it populates and reuses the project's
    existing :class:`~local_agent.models.SemanticIndex` /
    :class:`~local_agent.models.FileIndex` /
    :class:`~local_agent.models.SymbolDefinition` models. What it adds on top -
    resolved import edges, reverse dependencies and free-name references - is
    graph structure the index deliberately does not store.

    The graph is immutable once built and holds no global state, so two
    concurrently-executing worktrees each build their own instance rooted at
    their own directory with no interference.
    """

    def __init__(self, root: str | Path, *, max_files: int = DEFAULT_MAX_GRAPH_FILES):
        self.root = Path(root)
        self.max_files = max(1, int(max_files))
        self.files: dict[str, PythonFileFacts] = {}
        self.module_to_file: dict[str, str] = {}
        #: file -> repo-relative files it imports (resolved edges only).
        self.file_imports: dict[str, set[str]] = {}
        #: file -> files that import it (the reverse dependency graph).
        self.reverse_deps: dict[str, set[str]] = {}
        #: file -> local alias -> (defining file, original symbol name).
        self.imported_symbol_origins: dict[str, dict[str, tuple[str, str]]] = {}
        #: file -> dotted import strings that resolved to nothing in this repo.
        self.unresolved_imports: dict[str, set[str]] = {}
        #: file -> target files reached only through a resolved *dynamic* call
        #: (``importlib.import_module("literal")``), never a static import
        #: statement. Subset of ``file_imports[file]``; used purely for
        #: provenance (Phase 4.18) - it changes no traversal or scope decision.
        self.dynamic_resolved_edges: dict[str, set[str]] = {}
        self.parse_failures: dict[str, str] = {}
        self.semantic_index: SemanticIndex = SemanticIndex()
        self.truncated_at_max_files = False
        self.total_imports = 0
        self.resolved_import_count = 0
        self.build_seconds = 0.0
        #: symbol name -> number of distinct files defining a symbol with that
        #: name. Used to detect ambiguous names; see
        #: :data:`AMBIGUOUS_SYMBOL_DEFINITION_FILES`.
        self.definition_file_counts: dict[str, int] = {}

    # -- construction ------------------------------------------------------

    @classmethod
    def build(
        cls,
        root: str | Path,
        *,
        semantic_index: SemanticIndex | None = None,
        max_files: int = DEFAULT_MAX_GRAPH_FILES,
        excluded_directories: Iterable[str] | None = None,
    ) -> "SemanticGraph":
        """Parse every Python file under ``root`` and wire up the edges.

        ``semantic_index``, when given, is *enriched in place* with the parsed
        symbols and imports rather than replaced, so a caller that already holds
        the project's index gets the benefit of this parse for free (the
        Tree-sitter path leaves those lists empty when the optional grammar
        bundle is not built, which is the normal case).
        """
        graph = cls(root, max_files=max_files)
        started = time.perf_counter()
        if semantic_index is not None:
            graph.semantic_index = semantic_index
        excluded = {
            name.lower() for name in (excluded_directories or EXCLUDED_DIRECTORY_NAMES)
        }
        indexer = AstPythonIndexer()

        for relative in graph._discover_python_files(excluded):
            absolute = graph.root / relative
            try:
                raw = absolute.read_bytes()
            except OSError as exc:
                graph.parse_failures[relative] = f"unreadable: {exc}"
                continue
            facts = indexer.analyze(raw)
            if facts.parse_error:
                graph.parse_failures[relative] = facts.parse_error
                # Still register the module name: a file that fails to parse is
                # a real module other files may import, and pretending it does
                # not exist would understate the impact.
                graph.files[relative] = facts
            else:
                graph.files[relative] = facts
            graph._record_index_entry(relative, raw, facts)

        graph._build_module_map()
        graph._resolve_edges()
        graph._count_definitions()
        graph.build_seconds = time.perf_counter() - started
        return graph

    def _count_definitions(self) -> None:
        for relative in sorted(self.files):
            for name in sorted({symbol.name for symbol in self.files[relative].symbols}):
                self.definition_file_counts[name] = self.definition_file_counts.get(name, 0) + 1

    def is_ambiguous_symbol(self, name: str) -> bool:
        """Whether ``name`` is defined in enough files to make a bare reference
        to it meaningless as evidence."""
        return self.definition_file_counts.get(name, 0) > AMBIGUOUS_SYMBOL_DEFINITION_FILES

    def _discover_python_files(self, excluded: set[str]) -> list[str]:
        """Deterministically enumerate repo-relative ``.py`` paths under the root.

        Symlinks are skipped (matching the repository scanner and the candidate
        mirror) so a link cannot pull in content from outside the tree or cause
        infinite recursion. Excluded names are skipped whatever their type,
        which specifically covers a Git *worktree*, where ``.git`` is a FILE
        rather than a directory.
        """
        found: list[str] = []
        root = self.root
        for current, dirnames, filenames in os.walk(root, topdown=True):
            current_path = Path(current)
            dirnames[:] = sorted(
                name for name in dirnames
                if name.lower() not in excluded and not (current_path / name).is_symlink()
            )
            for name in sorted(filenames):
                if name.lower() in excluded:
                    continue
                if not name.lower().endswith(".py"):
                    continue
                candidate = current_path / name
                if candidate.is_symlink():
                    continue
                try:
                    relative = candidate.relative_to(root).as_posix()
                except ValueError:
                    continue
                found.append(relative)
                if len(found) >= self.max_files:
                    self.truncated_at_max_files = True
                    return found
        return found

    def _record_index_entry(self, relative: str, raw: bytes, facts: PythonFileFacts) -> None:
        """Populate the shared :class:`SemanticIndex` with what we just parsed."""
        content_hash = hashlib.sha256(raw).hexdigest()
        existing = self.semantic_index.files.get(relative)
        if existing is not None and existing.content_hash == content_hash and existing.symbols:
            # Already indexed at this exact content by someone else (e.g. the
            # Tree-sitter path); do not clobber it.
            return
        self.semantic_index.files[relative] = FileIndex(
            path=relative,
            language="Python",
            content_hash=content_hash,
            symbols=list(facts.symbols),
            imports=list(facts.module_import_names),
        )

    def _build_module_map(self) -> None:
        for relative in sorted(self.files):
            module = module_name_for(relative)
            if module is None:
                continue
            # A package ``__init__.py`` and a sibling module can never collide
            # (``a/__init__.py`` -> ``a``, ``a.py`` -> ``a``); prefer the
            # package, matching Python's own import resolution.
            existing = self.module_to_file.get(module)
            if existing is None or relative.endswith("/__init__.py"):
                self.module_to_file[module] = relative

    def _resolve_edges(self) -> None:
        for relative in sorted(self.files):
            facts = self.files[relative]
            imports: set[str] = set()
            unresolved: set[str] = set()
            origins: dict[str, tuple[str, str]] = {}
            dynamic_targets: set[str] = set()
            for record in facts.imports:
                self.total_imports += 1
                target, symbol = self._resolve_import(relative, record)
                if target is None:
                    dotted = ("." * record.level) + record.module if record.level else record.module
                    if dotted:
                        unresolved.add(dotted)
                    continue
                self.resolved_import_count += 1
                if target == relative:
                    # A module importing itself contributes no useful edge and
                    # would create a trivial cycle in the traversal.
                    continue
                imports.add(target)
                if record.dynamic:
                    dynamic_targets.add(target)
                if symbol:
                    origins[record.local_name or symbol] = (target, symbol)
            self.file_imports[relative] = imports
            if unresolved:
                self.unresolved_imports[relative] = unresolved
            if origins:
                self.imported_symbol_origins[relative] = origins
            if dynamic_targets:
                self.dynamic_resolved_edges[relative] = dynamic_targets
            for target in imports:
                self.reverse_deps.setdefault(target, set()).add(relative)

    def _resolve_import(
        self, importing_file: str, record: ImportRecord
    ) -> tuple[str | None, str | None]:
        """Resolve one import to ``(defining file, imported symbol name)``.

        Returns ``(None, None)`` for anything outside this repository (standard
        library, third-party packages, or a relative import that escapes the
        tree). Those are counted as unresolved, which lowers
        ``import_resolution`` and can only ever widen the eventual scope.
        """
        dotted = record.module
        if record.level:
            parts = Path(importing_file).as_posix().split("/")[:-1]
            drop = record.level - 1
            if drop > len(parts):
                return None, None
            base = parts[: len(parts) - drop] if drop else parts
            pieces = [piece for piece in base if piece]
            if record.module:
                pieces.append(record.module)
            dotted = ".".join(pieces)
        if not dotted:
            return None, None

        # ``from pkg.mod import thing``: prefer ``pkg.mod.thing`` when that is
        # itself a module (a submodule import), otherwise fall back to
        # ``pkg.mod`` and treat ``thing`` as a symbol defined there.
        if record.name:
            submodule = self.module_to_file.get(f"{dotted}.{record.name}")
            if submodule is not None:
                return submodule, None
        target = self.module_to_file.get(dotted)
        if target is not None:
            return target, (record.name or None)
        return None, None

    # -- queries -----------------------------------------------------------

    def references_of(self, relative: str) -> frozenset[str]:
        facts = self.files.get(relative)
        return facts.references if facts is not None else frozenset()

    def symbols_of(self, relative: str) -> list[SymbolDefinition]:
        facts = self.files.get(relative)
        return list(facts.symbols) if facts is not None else []

    def test_files(self) -> list[str]:
        """Every parsed file whose path looks like a test, sorted."""
        return sorted(path for path in self.files if looks_like_test_path(path))

    def reverse_dependents(
        self,
        seeds: Iterable[str],
        *,
        max_depth: int = DEFAULT_MAX_IMPACT_DEPTH,
        max_nodes: int = DEFAULT_MAX_AFFECTED_SYMBOLS,
    ) -> tuple[dict[str, tuple[int, str]], list[str]]:
        """Bounded BFS over the reverse dependency graph.

        Returns ``({file: (depth, via)}, bounds_hit)``. The traversal is:

        * **deterministic** - every frontier and adjacency set is sorted before
          iteration, so identical input yields an identical result and an
          identical truncation point;
        * **cycle-safe** - a visited set is carried across levels, so circular
          imports terminate instead of looping forever;
        * **bounded twice over** - by ``max_depth`` and by ``max_nodes``. Hitting
          either records an entry in ``bounds_hit``, which the caller turns into
          a degradation, hence LOW confidence, hence a broader scope.
        """
        max_depth = max(0, int(max_depth))
        max_nodes = max(0, int(max_nodes))
        seed_list = sorted({normalize_relative(seed) for seed in seeds if seed})
        result: dict[str, tuple[int, str]] = {}
        bounds_hit: list[str] = []
        seen: set[str] = set(seed_list)
        frontier: list[str] = seed_list

        depth = 0
        while frontier and depth < max_depth:
            depth += 1
            next_frontier: list[str] = []
            for node in frontier:
                for dependent in sorted(self.reverse_deps.get(node, ())):
                    if dependent in seen:
                        continue
                    if len(result) >= max_nodes:
                        bounds_hit.append("max_affected_symbols")
                        return result, bounds_hit
                    seen.add(dependent)
                    result[dependent] = (depth, f"{dependent} imports {node}")
                    next_frontier.append(dependent)
            frontier = sorted(next_frontier)

        if frontier and any(
            dependent not in seen
            for node in frontier
            for dependent in self.reverse_deps.get(node, ())
        ):
            bounds_hit.append("max_impact_depth")
        return result, bounds_hit


# -- symbol diffing -----------------------------------------------------------


def diff_python_symbols(
    base_text: str | None,
    new_text: str | None,
    path: str,
    *,
    indexer: AstPythonIndexer | None = None,
) -> tuple[list[ChangedSymbol], str]:
    """Compare two revisions of one Python file at the symbol level.

    ``base_text is None`` means the file was created; ``new_text is None`` means
    it was deleted. Returns ``(changes, error)`` where a non-empty ``error``
    means the comparison could not be trusted - the caller must record that as a
    degradation rather than concluding "no symbols changed".

    Modification is detected by comparing a whitespace-normalised hash of each
    symbol's source body, so a pure reformat is not reported as a semantic
    change while any real edit is.
    """
    indexer = indexer or AstPythonIndexer()
    if base_text is None and new_text is None:
        return [], "neither revision of the file is available"

    base_facts = indexer.analyze(base_text) if base_text is not None else None
    new_facts = indexer.analyze(new_text) if new_text is not None else None

    errors: list[str] = []
    if base_facts is not None and base_facts.parse_error:
        errors.append(f"base revision unparseable ({base_facts.parse_error})")
    if new_facts is not None and new_facts.parse_error:
        errors.append(f"new revision unparseable ({new_facts.parse_error})")
    if errors:
        return [], "; ".join(errors)

    def symbol_map(facts: PythonFileFacts | None) -> dict[str, SymbolDefinition]:
        if facts is None:
            return {}
        return {qualified_name(symbol): symbol for symbol in facts.symbols}

    base_symbols = symbol_map(base_facts)
    new_symbols = symbol_map(new_facts)
    base_hashes = base_facts.symbol_hashes if base_facts is not None else {}
    new_hashes = new_facts.symbol_hashes if new_facts is not None else {}
    exported = new_facts.exported_names if new_facts is not None else (
        base_facts.exported_names if base_facts is not None else ()
    )

    changes: list[ChangedSymbol] = []
    for qname in sorted(set(base_symbols) | set(new_symbols)):
        in_base = qname in base_symbols
        in_new = qname in new_symbols
        symbol = new_symbols.get(qname) or base_symbols[qname]
        if in_new and not in_base:
            change = CHANGE_ADDED
        elif in_base and not in_new:
            change = CHANGE_REMOVED
        elif base_hashes.get(qname) != new_hashes.get(qname):
            change = CHANGE_MODIFIED
        else:
            continue
        changes.append(
            ChangedSymbol(
                qualified_name=qname,
                name=symbol.name,
                file=normalize_relative(path),
                kind=symbol.kind,
                change=change,
                is_public=is_public_symbol(symbol, exported),
            )
        )
    return changes, ""


# -- scope policy -------------------------------------------------------------


def recommend_validation_scope(
    *,
    confidence: str,
    has_targets: bool,
    removed_public_symbols: int,
    affected_file_count: int,
    bounds_hit: Sequence[str],
    unsupported_files: Sequence[str],
    unparseable_files: Mapping[str, str],
) -> tuple[str, list[str]]:
    """Map evidence onto an adaptive validation scope. Escalation-only.

    The policy, stated once so it can be reviewed as a whole:

    ===============================================  ====================
    Situation                                        Minimum scope
    ===============================================  ====================
    HIGH confidence, narrow impact                   ``targeted``
    MEDIUM confidence                                ``expanded``
    LOW confidence (incl. every degraded analysis)   ``broad``
    No validation target could be associated         ``broad``
    A public symbol was removed or renamed           ``expanded``
    High fan-out (many affected modules)             ``expanded``
    A traversal bound was hit                        ``broad``
    A changed file's language is unsupported         ``broad``
    A changed file could not be parsed               ``broad``
    ===============================================  ====================

    Rows combine by taking the *broadest* applicable minimum, never the
    narrowest, and the result is never allowed to be "skip". Analysis failure
    means "run more", which is why every branch below calls
    :func:`escalate_scope` instead of assigning.
    """
    reasons: list[str] = []
    if confidence == CONFIDENCE_HIGH:
        scope = SCOPE_TARGETED
        reasons.append("high-confidence direct evidence permits targeted validation")
    elif confidence == CONFIDENCE_MEDIUM:
        scope = SCOPE_EXPANDED
        reasons.append("medium-confidence evidence requires expanded validation")
    else:
        scope = SCOPE_BROAD
        reasons.append("low-confidence evidence requires broad validation")

    if not has_targets:
        scope = escalate_scope(scope, SCOPE_BROAD)
        reasons.append("no validation target could be associated with the change")
    if removed_public_symbols:
        scope = escalate_scope(scope, SCOPE_EXPANDED)
        reasons.append(
            f"{removed_public_symbols} public symbol(s) removed or renamed; unseen callers may break"
        )
    if affected_file_count > HIGH_FANOUT_AFFECTED_FILES:
        scope = escalate_scope(scope, SCOPE_EXPANDED)
        reasons.append(
            f"high fan-out: {affected_file_count} modules depend on the change"
        )
    if bounds_hit:
        scope = escalate_scope(scope, SCOPE_BROAD)
        reasons.append(
            "impact traversal hit its bound(s) (" + ", ".join(sorted(set(bounds_hit))) + ")"
        )
    if unsupported_files:
        scope = escalate_scope(scope, SCOPE_BROAD)
        reasons.append(
            f"{len(unsupported_files)} changed file(s) are in a language this analysis "
            "cannot reason about"
        )
    if unparseable_files:
        scope = escalate_scope(scope, SCOPE_BROAD)
        reasons.append(f"{len(unparseable_files)} changed file(s) could not be parsed")
    return scope, reasons


# -- the analyzer -------------------------------------------------------------


class SemanticChangeImpactAnalyzer:
    """Turns a set of changed files into a :class:`ChangeImpactReport`.

    Holds no mutable global state: the root, the bounds and the optional
    collaborators are all instance-scoped, so parallel worktree workers each own
    an independent analyzer and cannot interfere with one another. The analyzer
    only ever *reads* the tree it is pointed at - it never writes, and it never
    changes the process working directory.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        semantic_index: SemanticIndex | None = None,
        graph: SemanticGraph | None = None,
        max_impact_depth: int = DEFAULT_MAX_IMPACT_DEPTH,
        max_affected_symbols: int = DEFAULT_MAX_AFFECTED_SYMBOLS,
        max_affected_tests: int = DEFAULT_MAX_AFFECTED_TESTS,
        max_graph_files: int = DEFAULT_MAX_GRAPH_FILES,
    ):
        self.root = Path(root)
        self.semantic_index = semantic_index
        self.max_impact_depth = max(0, int(max_impact_depth))
        self.max_affected_symbols = max(0, int(max_affected_symbols))
        self.max_affected_tests = max(0, int(max_affected_tests))
        self.max_graph_files = max(1, int(max_graph_files))
        self._graph = graph

    @property
    def graph(self) -> SemanticGraph:
        """The import/reference graph, built lazily and then cached."""
        if self._graph is None:
            self._graph = SemanticGraph.build(
                self.root,
                semantic_index=self.semantic_index,
                max_files=self.max_graph_files,
            )
        return self._graph

    def invalidate_graph(self) -> None:
        """Force the next analysis to re-read the tree from disk."""
        self._graph = None

    # -- main entry point --------------------------------------------------

    def analyze(
        self,
        changed_files: Iterable[str],
        *,
        base_contents: Mapping[str, str | None] | None = None,
        validation_intelligence: Any | None = None,
    ) -> ChangeImpactReport:
        """Produce the impact report for ``changed_files``.

        ``base_contents`` maps a repo-relative path to the file's *pre-change*
        text (``None`` when the file did not exist before). A
        :class:`~local_agent.sandbox.CandidateWorkspace` supplies this from its
        frozen BASE snapshot, which is what makes candidate-time symbol diffing
        exact rather than guessed. When it is omitted the analysis still runs,
        but every changed file's symbols are treated as *added* and that is
        recorded as a degradation.

        ``validation_intelligence`` is the existing
        :class:`~local_agent.validation.ValidationIntelligence`, reused as the
        bottom (lexical) association tier rather than reimplemented.
        """
        started = time.perf_counter()
        report = ChangeImpactReport()
        normalized = sorted({normalize_relative(path) for path in changed_files if path})
        report.changed_files = list(normalized)
        if not normalized:
            report.confidence = CONFIDENCE_LOW
            report.confidence_reason = "no files changed"
            report.recommended_scope = SCOPE_TARGETED
            report.scope_reasons = ["nothing changed, so there is nothing to validate"]
            report.analysis_seconds = time.perf_counter() - started
            return report

        evidence = ImpactEvidence()
        try:
            graph = self.graph
        except OSError as exc:
            # A tree we cannot walk is a total analysis failure: report it as
            # such and let the scope policy force BROAD validation.
            LOGGER.warning("Could not build semantic graph at %s: %s", self.root, exc)
            evidence.degradations.append(f"semantic graph unavailable ({exc})")
            report.evidence = evidence
            report.confidence, report.confidence_reason = evidence.assess()
            report.recommended_scope, report.scope_reasons = recommend_validation_scope(
                confidence=report.confidence,
                has_targets=False,
                removed_public_symbols=0,
                affected_file_count=0,
                bounds_hit=[],
                unsupported_files=list(normalized),
                unparseable_files={},
            )
            report.analysis_seconds = time.perf_counter() - started
            return report

        if graph.truncated_at_max_files:
            evidence.degradations.append("repository larger than the graph file bound")
            report.bounds_hit.append("max_graph_files")

        self._collect_symbol_changes(report, evidence, normalized, base_contents)
        self._collect_affected(report, evidence, graph, normalized)
        self._select_targets(report, evidence, graph, normalized, validation_intelligence)

        analyzable = [
            path for path in normalized
            if is_python_path(path)
            and path not in report.unparseable_files
            and path not in report.unsupported_files
        ]
        evidence.graph_coverage = len(analyzable) / len(normalized)
        evidence.import_resolution = (
            graph.resolved_import_count / graph.total_imports if graph.total_imports else 0.0
        )
        report.graph_coverage = evidence.graph_coverage

        report.evidence = evidence
        report.confidence, report.confidence_reason = evidence.assess()
        removed_public = sum(
            1 for symbol in report.removed_symbols if symbol.is_public
        )
        report.recommended_scope, report.scope_reasons = recommend_validation_scope(
            confidence=report.confidence,
            has_targets=any(
                target.tier != TIER_BROAD for target in report.validation_targets
            ),
            removed_public_symbols=removed_public,
            affected_file_count=len(report.affected_files),
            bounds_hit=report.bounds_hit,
            unsupported_files=report.unsupported_files,
            unparseable_files=report.unparseable_files,
        )
        report.analysis_seconds = time.perf_counter() - started
        return report

    # -- stage 1: changed files -> changed symbols -------------------------

    def _collect_symbol_changes(
        self,
        report: ChangeImpactReport,
        evidence: ImpactEvidence,
        changed_files: list[str],
        base_contents: Mapping[str, str | None] | None,
    ) -> None:
        indexer = AstPythonIndexer()
        for relative in changed_files:
            if escapes_root(relative):
                # Never join an escaping path onto the root: record it as
                # unsupported (which forces LOW confidence and a BROAD scope)
                # and read nothing.
                report.unsupported_files.append(relative)
                evidence.degradations.append(
                    f"'{relative}' points outside the analysed tree and was not read"
                )
                continue
            if not is_python_path(relative):
                report.unsupported_files.append(relative)
                evidence.degradations.append(
                    f"'{relative}' is not Python and was not semantically analysed"
                )
                continue

            new_text = self._read(relative)
            if base_contents is not None and relative in base_contents:
                # A present key with a ``None`` value means "did not exist in
                # BASE", i.e. the file was created. That is exact information.
                base_text: str | None = base_contents[relative]
            else:
                # No base revision available. We cannot distinguish added from
                # modified, so treat every symbol as added (the conservative
                # reading: it maximises the changed-symbol set) and record the
                # gap, which forces LOW confidence and a broader scope.
                base_text = None
                evidence.degradations.append(
                    f"no base snapshot for '{relative}'; symbol changes are approximate"
                )

            if new_text is None and base_text is None:
                report.unparseable_files[relative] = "file missing in both revisions"
                evidence.degradations.append(f"'{relative}' could not be read")
                continue

            changes, error = diff_python_symbols(
                base_text, new_text, relative, indexer=indexer
            )
            if error:
                report.unparseable_files[relative] = error
                evidence.degradations.append(f"'{relative}': {error}")
                continue
            for change in changes:
                if change.change == CHANGE_ADDED:
                    report.added_symbols.append(change)
                elif change.change == CHANGE_REMOVED:
                    report.removed_symbols.append(change)
                else:
                    report.modified_symbols.append(change)
                if change.is_public:
                    report.public_api_symbols.append(f"{relative}::{change.qualified_name}")

        report.added_symbols.sort(key=lambda s: (s.file, s.qualified_name))
        report.removed_symbols.sort(key=lambda s: (s.file, s.qualified_name))
        report.modified_symbols.sort(key=lambda s: (s.file, s.qualified_name))
        report.public_api_symbols = sorted(set(report.public_api_symbols))

    def _read(self, relative: str) -> str | None:
        # Defence in depth: callers already filter escaping paths, but this is
        # the only place the analyzer touches the filesystem, so the guard lives
        # here too rather than relying on every caller remembering it.
        if escapes_root(relative):
            return None
        candidate = self.root / relative
        try:
            if not candidate.is_file():
                return None
            return candidate.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError, ValueError):
            return None

    # -- stage 2: changed files -> affected modules ------------------------

    def _collect_affected(
        self,
        report: ChangeImpactReport,
        evidence: ImpactEvidence,
        graph: SemanticGraph,
        changed_files: list[str],
    ) -> None:
        seeds = [
            path for path in changed_files
            if is_python_path(path) and not escapes_root(path)
        ]
        dependents, bounds_hit = graph.reverse_dependents(
            seeds,
            max_depth=self.max_impact_depth,
            max_nodes=self.max_affected_symbols,
        )
        report.bounds_hit.extend(bounds_hit)
        if bounds_hit:
            evidence.degradations.append(
                "reverse dependency traversal was truncated ("
                + ", ".join(sorted(set(bounds_hit)))
                + ")"
            )

        report.affected_files = sorted(dependents)
        for path in report.affected_files:
            depth, via = dependents[path]
            module = module_name_for(path) or path
            report.affected_symbols.append(
                AffectedSymbol(qualified_name=module, file=path, depth=depth, via=via)
            )
        report.affected_symbols.sort(key=lambda s: (s.depth, s.file))

        # A dependent that references a changed symbol name it cannot resolve to
        # any definition in this repository is worth surfacing: it usually means
        # a rename, a dynamic attribute, or a symbol coming from a dependency.
        changed_names = report.changed_symbol_names
        if changed_names:
            defined: set[str] = set()
            for path in sorted(graph.files):
                for symbol in graph.symbols_of(path):
                    defined.add(symbol.name)
            report.unresolved_symbols = sorted(changed_names - defined)

        dynamic = sorted(
            path for path in report.affected_files
            if graph.files.get(path) is not None and graph.files[path].has_dynamic_imports
        )
        if dynamic:
            evidence.degradations.append(
                f"{len(dynamic)} affected module(s) use dynamic or star imports, so their "
                "dependency edges are incomplete"
            )
        if any(
            graph.files.get(path) is not None and graph.files[path].has_dynamic_imports
            for path in changed_files
        ):
            evidence.degradations.append(
                "a changed module uses dynamic or star imports, so its dependents "
                "cannot be fully enumerated"
            )

    # -- stage 3: affected modules -> validation targets -------------------

    def _select_targets(
        self,
        report: ChangeImpactReport,
        evidence: ImpactEvidence,
        graph: SemanticGraph,
        changed_files: list[str],
        validation_intelligence: Any | None,
    ) -> None:
        """Associate tests with the change, strongest evidence tier first.

        Tier ladder, strongest first. Each test is assigned to exactly one tier
        - its strongest - and carries a sentence explaining that choice:

        1. ``direct_symbol_match``  - the test imports a changed module *and*
           references a changed symbol by name (or the test file itself changed);
        2. ``direct_import_match``  - the test imports a changed module;
        3. ``call_graph_match``     - the test references a changed symbol name
           but no import edge was resolved (weaker: names can collide);
        4. ``reverse_dependency_match`` - the test imports a module that
           transitively depends on a changed module;
        5. ``module_match``         - the test's module name derives from a
           changed module's name (``impact.py`` <-> ``test_impact.py``);
        6. ``filename_match``       - whatever the pre-existing
           :class:`~local_agent.validation.ValidationIntelligence` lexical
           heuristics find, reused rather than reimplemented;
        7. ``broad_fallback``       - nothing associated; run the whole suite.
        """
        changed_set = {
            path for path in changed_files
            if is_python_path(path) and not escapes_root(path)
        }
        changed_names = report.changed_symbol_names
        affected_set = set(report.affected_files)
        affected_depths = {s.file: s.depth for s in report.affected_symbols}
        # Phase 4.18: precomputed once so per-test evidence construction below
        # can attribute an inheritance/decorator/annotation/dynamic-import hit
        # to the specific changed file(s) that actually define the name.
        name_to_changed_files: dict[str, set[str]] = {}
        for symbol in report.changed_symbols:
            name_to_changed_files.setdefault(symbol.name, set()).add(symbol.file)

        candidates = graph.test_files()
        report.tests_considered = len(candidates)
        targets: dict[str, ValidationTarget] = {}

        # Names common enough to be meaningless on their own (``run``, ``analyze``,
        # ``to_dict``) are excluded from the *reference-only* tier. They still
        # count wherever a resolved import edge corroborates them, so this
        # removes spurious associations without ever removing a real one.
        distinctive_names = {
            name for name in changed_names if not graph.is_ambiguous_symbol(name)
        }
        ambiguous_names = sorted(changed_names - distinctive_names)
        if ambiguous_names:
            report.knowledge_notes.append(
                "ambiguous changed symbol name(s) "
                + ", ".join(ambiguous_names[:6])
                + " were not used as standalone reference evidence "
                f"(each is defined in more than {AMBIGUOUS_SYMBOL_DEFINITION_FILES} files)"
            )

        for test_path in candidates:
            imports = graph.file_imports.get(test_path, set())
            imported_changed = sorted(imports & changed_set)
            references = graph.references_of(test_path)

            # Phase 4.18: resolve local aliases (``from x import y as z``) back
            # to the definition they actually name, so an aliased reference
            # counts as evidence for the *original* symbol it refers to instead
            # of being invisible to the bare name-match below. Every reference
            # not already a literal changed-symbol name is a candidate; most
            # resolve to nothing (not an alias at all) and cost one dict lookup.
            alias_evidence: dict[str, DependencyEvidence] = {}
            for name in sorted(references - changed_names):
                hit = resolve_alias_reference(
                    source_file=test_path,
                    reference=name,
                    imported_symbol_origins=graph.imported_symbol_origins,
                    changed_files=frozenset(changed_set),
                    changed_symbol_names=frozenset(changed_names),
                )
                if hit is not None:
                    alias_evidence[hit.target_symbol] = hit

            referenced = sorted((changed_names & references) | set(alias_evidence))
            distinctive_referenced = sorted(
                (distinctive_names & references) | set(alias_evidence)
            )
            extra_evidence = self._construct_evidence(
                graph, test_path, distinctive_names, name_to_changed_files,
            ) + tuple(alias_evidence.values())

            if test_path in changed_set:
                targets[test_path] = ValidationTarget(
                    path=test_path,
                    command=("pytest", test_path),
                    tier=TIER_DIRECT_SYMBOL,
                    selected_because=(
                        f"the test file '{test_path}' was itself changed by this edit"
                    ),
                    matched_symbols=sorted(
                        symbol.qualified_name
                        for symbol in report.changed_symbols
                        if symbol.file == test_path
                    ),
                    matched_files=[test_path],
                )
                continue
            if imported_changed and referenced:
                targets[test_path] = ValidationTarget(
                    path=test_path,
                    command=("pytest", test_path),
                    tier=TIER_DIRECT_SYMBOL,
                    selected_because=(
                        f"'{test_path}' imports changed module(s) {', '.join(imported_changed)} "
                        f"and references changed symbol(s) {', '.join(referenced)}"
                    ),
                    matched_symbols=referenced,
                    matched_files=imported_changed,
                    dependency_evidence=extra_evidence,
                )
                continue
            if imported_changed:
                targets[test_path] = ValidationTarget(
                    path=test_path,
                    command=("pytest", test_path),
                    tier=TIER_DIRECT_IMPORT,
                    selected_because=(
                        f"'{test_path}' directly imports changed module(s) "
                        f"{', '.join(imported_changed)}"
                    ),
                    matched_files=imported_changed,
                    dependency_evidence=extra_evidence,
                )
                continue
            if distinctive_referenced:
                targets[test_path] = ValidationTarget(
                    path=test_path,
                    command=("pytest", test_path),
                    tier=TIER_CALL_GRAPH,
                    selected_because=(
                        f"'{test_path}' references distinctive changed symbol(s) "
                        f"{', '.join(distinctive_referenced)} without a resolved import edge "
                        "(name match only)"
                    ),
                    matched_symbols=distinctive_referenced,
                    dependency_evidence=extra_evidence,
                )
                continue
            imported_affected = sorted(imports & affected_set)
            if imported_affected:
                depth = min(
                    (affected_depths.get(path, 1) for path in imported_affected), default=1
                )
                targets[test_path] = ValidationTarget(
                    path=test_path,
                    command=("pytest", test_path),
                    tier=TIER_REVERSE_DEPENDENCY,
                    selected_because=(
                        f"'{test_path}' imports {', '.join(imported_affected)}, which "
                        f"transitively depend(s) on the change at depth {depth}"
                    ),
                    matched_files=imported_affected,
                    depth=depth,
                    dependency_evidence=self._reexport_evidence(
                        graph, test_path, imported_affected, changed_set, changed_names
                    ),
                )
                continue
            module_match = self._module_match(test_path, changed_set)
            if module_match:
                targets[test_path] = ValidationTarget(
                    path=test_path,
                    command=("pytest", test_path),
                    tier=TIER_MODULE,
                    selected_because=(
                        f"'{test_path}' is named after changed module '{module_match}' "
                        "(naming convention only, no import edge found)"
                    ),
                    matched_files=[module_match],
                )

        self._add_lexical_targets(targets, changed_files, validation_intelligence)

        ranked = sorted(targets.values(), key=lambda target: target.sort_key())
        if self.max_affected_tests and len(ranked) > self.max_affected_tests:
            report.bounds_hit.append("max_affected_tests")
            evidence.degradations.append(
                f"{len(ranked)} associated test(s) exceeded the max_affected_tests bound "
                f"of {self.max_affected_tests}"
            )
            ranked = ranked[: self.max_affected_tests]

        for target in ranked:
            if target.tier == TIER_DIRECT_SYMBOL:
                evidence.direct_symbol_matches += 1
            elif target.tier == TIER_DIRECT_IMPORT:
                evidence.direct_import_matches += 1
            elif target.tier == TIER_CALL_GRAPH:
                evidence.call_graph_matches += 1
            elif target.tier == TIER_REVERSE_DEPENDENCY:
                evidence.reverse_dependency_matches += 1
            elif target.tier == TIER_MODULE:
                evidence.module_matches += 1
            elif target.tier == TIER_FILENAME:
                evidence.filename_matches += 1

        if not ranked:
            ranked = [
                ValidationTarget(
                    path="",
                    command=("pytest",),
                    tier=TIER_BROAD,
                    selected_because=(
                        "no test could be associated with the change by import, reference, "
                        "dependency or naming evidence, so the whole suite is the only "
                        "safe validation"
                    ),
                )
            ]
        report.validation_targets = ranked

    def _construct_evidence(
        self,
        graph: "SemanticGraph",
        test_path: str,
        distinctive_names: set[str],
        name_to_changed_files: dict[str, set[str]],
    ) -> tuple[DependencyEvidence, ...]:
        """Phase 4.18: label *why* a distinctive reference is trustworthy.

        Purely explanatory: it never changes which tier a target lands in
        (that ladder is unchanged), only attaches a more specific reason when
        one of these constructs - rather than an ordinary statement - is what
        connects ``test_path`` to the change.
        """
        facts = graph.files.get(test_path)
        if facts is None:
            return ()
        out: list[DependencyEvidence] = []

        def _tag(names: frozenset[str], evidence_type: str, phrase: str) -> None:
            for name in sorted(distinctive_names & names):
                for target_file in sorted(name_to_changed_files.get(name, ())):
                    out.append(
                        make_evidence(
                            source_file=test_path,
                            target_file=target_file,
                            evidence_type=evidence_type,
                            target_symbol=name,
                            source_reference=name,
                            provenance=f"'{test_path}' {phrase} '{name}', changed in {target_file}",
                        )
                    )

        _tag(facts.base_class_references, INHERITANCE, "inherits from")
        _tag(facts.decorator_references, DECORATOR, "is decorated with")
        _tag(facts.annotation_references, ANNOTATION, "type-annotates with")
        _tag(facts.attribute_references, ATTRIBUTE_RESOLUTION, "accesses an attribute named")

        for target_file in sorted(graph.dynamic_resolved_edges.get(test_path, ())):
            out.append(
                make_evidence(
                    source_file=test_path,
                    target_file=target_file,
                    evidence_type=DYNAMIC_IMPORT_RESOLVED,
                    provenance=(
                        f"'{test_path}' loads {target_file} through a dynamic "
                        "import call whose module string was a resolvable literal"
                    ),
                )
            )
        return tuple(out)

    def _reexport_evidence(
        self,
        graph: "SemanticGraph",
        test_path: str,
        imported_affected: list[str],
        changed_set: set[str],
        changed_names: set[str],
    ) -> tuple[DependencyEvidence, ...]:
        """Label a ``reverse_dependency_match`` that passes through a re-export.

        ``imported_affected`` are files ``test_path`` imports that are
        themselves downstream of the change. When one of them lists the
        changed symbol in its own ``__all__``, the edge is a deliberate
        re-export rather than an incidental transitive import, which is worth
        saying explicitly - it is materially stronger evidence than "some file
        two hops away happens to import something".
        """
        out: list[DependencyEvidence] = []
        for intermediate in imported_affected:
            facts = graph.files.get(intermediate)
            if facts is None:
                continue
            reexported = set(facts.exported_names) & changed_names
            if not reexported:
                continue
            origin = next(
                (source for source in sorted(changed_set) if source != intermediate), intermediate
            )
            for name in sorted(reexported):
                out.append(
                    make_evidence(
                        source_file=test_path,
                        target_file=intermediate,
                        evidence_type=REEXPORT,
                        target_symbol=name,
                        provenance=(
                            f"'{intermediate}' re-exports '{name}' via __all__, and "
                            f"'{test_path}' imports {intermediate}"
                        ),
                        resolution_notes=(f"originating change traced toward {origin}",),
                    )
                )
        return tuple(out)

    def _module_match(self, test_path: str, changed_files: set[str]) -> str | None:
        """``local_agent/impact.py`` <-> ``tests/test_impact.py`` naming link."""
        stem = Path(test_path).stem
        bases: set[str] = set()
        if stem.startswith("test_"):
            bases.add(stem[5:])
        if stem.endswith("_test"):
            bases.add(stem[:-5])
        if not bases:
            return None
        for changed in sorted(changed_files):
            if Path(changed).stem in bases:
                return changed
        return None

    def _add_lexical_targets(
        self,
        targets: dict[str, ValidationTarget],
        changed_files: list[str],
        validation_intelligence: Any | None,
    ) -> None:
        """Bottom tier: reuse the existing lexical heuristics, do not rebuild them."""
        if validation_intelligence is None:
            from .validation import ValidationIntelligence

            validation_intelligence = ValidationIntelligence(self.root)
        try:
            discovered = validation_intelligence.discover_targeted_commands(
                list(changed_files), None
            )
        except (OSError, ValueError, AttributeError) as exc:
            LOGGER.debug("Lexical target discovery unavailable: %s", exc)
            return
        for spec in discovered:
            command = tuple(str(token) for token in getattr(spec, "command", ()))
            if len(command) < 2:
                continue
            path = normalize_relative(command[-1])
            if escapes_root(path):
                # A lexical heuristic should never hand back an out-of-tree
                # path, but a target is a command that will really be executed,
                # so it is checked rather than assumed.
                LOGGER.debug("Ignoring out-of-tree lexical validation target %r", path)
                continue
            if path in targets:
                # Already associated by stronger, semantic evidence.
                continue
            targets[path] = ValidationTarget(
                path=path,
                command=command,
                tier=TIER_FILENAME,
                selected_because=(
                    f"'{path}' matches a changed file by filename convention "
                    f"({getattr(spec, 'reason', 'lexical heuristic')})"
                ),
            )


# -- knowledge graph integration ---------------------------------------------


def apply_knowledge_support(
    report: ChangeImpactReport,
    knowledge_manager: Any | None,
    *,
    root: str | Path | None = None,
    max_notes: int = 5,
) -> ChangeImpactReport:
    """Fold persistent knowledge into a report as *supporting evidence only*.

    Hard rules, enforced here rather than trusted to callers:

    * Knowledge may add a ``filename_match``-tier target and explanatory notes.
      It may **never** raise ``confidence`` and may **never** narrow
      ``recommended_scope`` - the report's scope is recomputed only through
      :func:`escalate_scope`, so the result can only stay the same or widen.
    * A knowledge node whose recorded ``content_hash`` no longer matches the
      file's current bytes is *stale*. Stale knowledge is discarded with an
      explicit note; it is never applied on the assumption that it is probably
      still right.
    * Any failure to read the graph is swallowed into a note. Knowledge is an
      optimisation; its absence must never break or weaken the analysis.
    """
    if knowledge_manager is None:
        return report
    try:
        graph = knowledge_manager.get_graph()
    except (OSError, ValueError, AttributeError) as exc:
        report.knowledge_notes.append(f"knowledge graph unavailable ({exc})")
        return report
    if graph is None:
        return report

    base = Path(root) if root is not None else None
    notes: list[str] = []
    stale = 0
    for relative in report.changed_files:
        node = getattr(graph, "files", {}).get(relative)
        if node is None:
            continue
        recorded_hash = str(getattr(node, "content_hash", "") or "")
        current_hash = _current_hash(base, relative) if base is not None else None
        if recorded_hash and current_hash is not None and recorded_hash != current_hash:
            stale += 1
            notes.append(
                f"ignored stale knowledge for '{relative}' (recorded hash no longer matches)"
            )
            continue
        for dependent in sorted(getattr(node, "dependents", []) or [])[:3]:
            notes.append(
                f"knowledge graph records '{dependent}' as a historical dependent of '{relative}'"
            )

    for pattern in sorted(
        getattr(graph, "failure_patterns", []) or [],
        key=lambda item: str(getattr(item, "pattern_id", "")),
    ):
        affected = set(str(path) for path in getattr(pattern, "affected_files", []) or [])
        overlap = sorted(affected & set(report.changed_files))
        if not overlap:
            continue
        notes.append(
            f"recurring failure pattern '{getattr(pattern, 'error_signature', '')}' previously "
            f"affected {', '.join(overlap)}"
        )

    if stale:
        # Stale knowledge is itself a small signal that the picture is
        # incomplete, so it widens - never narrows - the scope.
        report.recommended_scope = escalate_scope(report.recommended_scope, SCOPE_EXPANDED)
        report.scope_reasons.append(
            f"{stale} knowledge entr(y/ies) for changed files were stale and ignored"
        )
    report.knowledge_notes.extend(notes[:max_notes])
    return report


def _current_hash(root: Path, relative: str) -> str | None:
    try:
        return hashlib.sha256((root / relative).read_bytes()).hexdigest()
    except OSError:
        return None
