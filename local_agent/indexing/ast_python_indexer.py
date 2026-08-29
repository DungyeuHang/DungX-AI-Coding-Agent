"""Phase 4.17: stdlib-``ast`` Python fact extraction.

Why this exists
---------------
The project already ships a Tree-sitter backed :class:`PythonIndexer`, but it
only runs when a compiled ``build/languages.so`` grammar bundle is present. On
a plain checkout that file does not exist, so :class:`SemanticIndex` entries are
created with *empty* ``symbols``/``imports`` lists and every downstream
consumer silently degrades to lexical matching.

Semantic change-impact analysis cannot be built on an index that is empty in
practice, so this module extracts the same facts using the standard library's
``ast`` module - the approach ``contract_extractor.py`` already uses in this
codebase. It deliberately emits the **existing** :class:`SymbolDefinition` /
:class:`SymbolLocation` models rather than introducing a parallel symbol model,
and exposes an ``index()`` method with the same signature as
:class:`PythonIndexer` so it is a drop-in fallback.

It additionally exposes :meth:`AstPythonIndexer.analyze`, which returns the
richer facts the impact graph needs and the Tree-sitter indexer does not
currently produce: structured import records (including relative-import level),
free name references, per-symbol body hashes (for modified-symbol detection)
and a dynamic-import flag.

Scope: Python only. Anything else must be reported as *unsupported*, never
guessed at - see :mod:`local_agent.impact` for how that degrades confidence.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field
from typing import Sequence

from ..models import SymbolDefinition, SymbolLocation

#: Callables whose presence means imports cannot be resolved statically.
_DYNAMIC_IMPORT_NAMES = frozenset({"__import__", "import_module", "load_module", "exec", "eval"})


@dataclass(frozen=True)
class ImportRecord:
    """One imported thing, preserving enough structure to resolve it later.

    ``import a.b``            -> ImportRecord("a.b", 0, None, None)
    ``import a.b as x``       -> ImportRecord("a.b", 0, None, "x")
    ``from a.b import c``     -> ImportRecord("a.b", 0, "c", None)
    ``from . import c``       -> ImportRecord("", 1, "c", None)
    ``from ..x import c as d``-> ImportRecord("x", 2, "c", "d")
    """

    module: str
    level: int = 0
    name: str | None = None
    asname: str | None = None

    @property
    def local_name(self) -> str:
        """The name this import binds in the importing module's namespace."""
        if self.asname:
            return self.asname
        if self.name is not None:
            return self.name
        # ``import a.b`` binds the top-level package name ``a``.
        return self.module.split(".", 1)[0] if self.module else ""


@dataclass
class PythonFileFacts:
    """Everything :mod:`local_agent.impact` needs from one Python source file."""

    symbols: list[SymbolDefinition] = field(default_factory=list)
    #: Qualified name (``Class.method`` / ``function``) -> hash of its source body.
    symbol_hashes: dict[str, str] = field(default_factory=dict)
    imports: list[ImportRecord] = field(default_factory=list)
    #: Every identifier referenced in a load context anywhere in the module.
    references: frozenset[str] = frozenset()
    #: True when the module performs imports that cannot be statically resolved.
    has_dynamic_imports: bool = False
    #: Non-empty when the file could not be parsed; symbols/imports are then empty.
    parse_error: str = ""
    #: Module-level ``__all__`` entries, when declared as a literal list/tuple.
    exported_names: tuple[str, ...] = ()

    @property
    def module_import_names(self) -> list[str]:
        """Dotted module strings, matching ``FileIndex.imports``' existing shape."""
        seen: dict[str, None] = {}
        for record in self.imports:
            dotted = record.module
            if record.level:
                dotted = ("." * record.level) + record.module
            if dotted:
                seen.setdefault(dotted, None)
        return list(seen)


def _definition_extent(node: ast.AST) -> tuple[int, int]:
    """Full source span a definition owns, decorators included.

    ``ast`` puts ``lineno`` on the ``def``/``class`` keyword, not on the first
    decorator, so a decorator-only edit would otherwise be attributed to the
    *enclosing* scope instead of to the decorated symbol.
    """
    start = int(getattr(node, "lineno", 1))
    for decorator in getattr(node, "decorator_list", None) or []:
        decorator_line = getattr(decorator, "lineno", None)
        if decorator_line:
            start = min(start, int(decorator_line))
    end = int(getattr(node, "end_lineno", None) or getattr(node, "lineno", start))
    return start, max(start, end)


def qualified_name(symbol: SymbolDefinition) -> str:
    """``Class.method`` for nested symbols, plain ``name`` for module-level ones."""
    return f"{symbol.parent}.{symbol.name}" if symbol.parent else symbol.name


def is_public_symbol(symbol: SymbolDefinition, exported_names: tuple[str, ...] = ()) -> bool:
    """Public-API heuristic: no leading underscore anywhere in the qualified path.

    An explicit ``__all__`` entry wins over the underscore convention, because a
    project that exports ``_compat`` deliberately means it.
    """
    if symbol.name in exported_names:
        return True
    if symbol.name.startswith("_") and not (
        symbol.name.startswith("__") and symbol.name.endswith("__")
    ):
        return False
    if symbol.parent and symbol.parent.startswith("_"):
        return False
    return True


class AstPythonIndexer:
    """Extracts symbols, imports and references from Python source via ``ast``.

    Stateless and therefore safe to share across threads / parallel worktrees.
    """

    #: Guard against pathological inputs; a source file larger than this is not
    #: something a semantic diff needs to reason about symbol-by-symbol.
    max_source_bytes: int = 2 * 1024 * 1024

    def index(self, content: bytes | str) -> tuple[list[SymbolDefinition], list[str]]:
        """Signature-compatible with :class:`PythonIndexer.index`."""
        facts = self.analyze(content)
        return facts.symbols, facts.module_import_names

    def analyze(self, content: bytes | str) -> PythonFileFacts:
        """Parse ``content`` and return structured facts.

        Never raises for bad input: a syntax error, a null byte or an undecodable
        blob yields ``PythonFileFacts(parse_error=...)`` so callers can *lower
        confidence* rather than crash or, worse, silently claim a clean analysis.
        """
        if isinstance(content, bytes):
            if len(content) > self.max_source_bytes:
                return PythonFileFacts(parse_error="source exceeds maximum indexable size")
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError as exc:
                return PythonFileFacts(parse_error=f"undecodable source: {exc}")
        else:
            text = content
            if len(text) > self.max_source_bytes:
                return PythonFileFacts(parse_error="source exceeds maximum indexable size")

        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError, MemoryError, RecursionError) as exc:
            # ValueError covers embedded null bytes; RecursionError covers
            # adversarially deeply-nested literals.
            return PythonFileFacts(parse_error=f"{type(exc).__name__}: {exc}")

        lines = text.splitlines()
        collector = _FactCollector(lines)
        collector.visit_module(tree)
        return PythonFileFacts(
            symbols=collector.symbols,
            symbol_hashes=collector.symbol_hashes,
            imports=collector.imports,
            references=frozenset(collector.references),
            has_dynamic_imports=collector.has_dynamic_imports,
            exported_names=tuple(collector.exported_names),
        )


class _FactCollector:
    """Single-pass AST walk collecting symbols, imports and references."""

    def __init__(self, lines: list[str]):
        self._lines = lines
        self.symbols: list[SymbolDefinition] = []
        self.symbol_hashes: dict[str, str] = {}
        self.imports: list[ImportRecord] = []
        self.references: set[str] = set()
        self.has_dynamic_imports = False
        self.exported_names: list[str] = []

    # -- entry point -------------------------------------------------------

    def visit_module(self, tree: ast.Module) -> None:
        self._walk_body(tree.body, parent=None)
        # References and imports are collected over the *whole* tree, including
        # nested function bodies, because a test exercising a changed symbol
        # usually calls it from inside a test method.
        for node in ast.walk(tree):
            self._collect_reference(node)
            self._collect_import(node)
            self._collect_dunder_all(node)

    # -- definitions -------------------------------------------------------

    def _walk_body(self, body: list[ast.stmt], parent: str | None) -> None:
        """Record class/function/method definitions with their parent scope.

        Only one level of nesting is recorded (matching the existing Tree-sitter
        indexer's ``parent`` field, which is a single name and not a full path).
        Definitions nested inside functions are intentionally not indexed: they
        are not part of any importable surface.
        """
        for node in body:
            if isinstance(node, ast.ClassDef):
                # A class's hash covers only the lines it owns *directly*: the
                # spans of its nested definitions are subtracted, so editing one
                # method body reports ``A.m`` alone rather than both ``A.m`` and
                # ``A``. Editing the class's decorators, bases or class-level
                # attributes still reports ``A``, because those lines are its own.
                self._record(
                    node,
                    kind="class",
                    parent=parent,
                    excluded_spans=self._definition_spans(node.body),
                )
                self._walk_body(node.body, parent=node.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "method" if parent else "function"
                self._record(node, kind=kind, parent=parent)
            elif isinstance(node, (ast.If, ast.Try)):
                # ``if TYPE_CHECKING:`` / ``try: ... except ImportError:`` blocks
                # routinely hold real top-level definitions and imports.
                self._walk_body(node.body, parent)
                self._walk_body(getattr(node, "orelse", []), parent)
                for handler in getattr(node, "handlers", []):
                    self._walk_body(handler.body, parent)
                self._walk_body(getattr(node, "finalbody", []), parent)

    def _definition_spans(self, body: list[ast.stmt]) -> list[tuple[int, int]]:
        """Extents of the definitions :meth:`_walk_body` would record from ``body``.

        Mirrors :meth:`_walk_body`'s traversal exactly - including its descent
        into ``if``/``try`` blocks - so the set of subtracted spans is precisely
        the set of separately-hashed child symbols. The ``if``/``try`` header
        lines themselves are *not* subtracted: they belong to the enclosing
        scope, so changing a ``if TYPE_CHECKING:`` guard does register as a
        change to the class that contains it.
        """
        spans: list[tuple[int, int]] = []
        for node in body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                spans.append(_definition_extent(node))
            elif isinstance(node, (ast.If, ast.Try)):
                spans.extend(self._definition_spans(node.body))
                spans.extend(self._definition_spans(getattr(node, "orelse", [])))
                for handler in getattr(node, "handlers", []):
                    spans.extend(self._definition_spans(handler.body))
                spans.extend(self._definition_spans(getattr(node, "finalbody", [])))
        return spans

    def _record(
        self,
        node: ast.AST,
        kind: str,
        parent: str | None,
        excluded_spans: Sequence[tuple[int, int]] = (),
    ) -> None:
        # ``location`` deliberately keeps the ``def``/``class`` keyword line, which
        # is what every existing consumer of SymbolLocation expects; the wider,
        # decorator-inclusive extent is used only for hashing.
        start = int(getattr(node, "lineno", 1))
        end = int(getattr(node, "end_lineno", None) or start)
        symbol = SymbolDefinition(
            name=node.name,  # type: ignore[attr-defined]
            kind=kind,  # type: ignore[arg-type]
            location=SymbolLocation(start_line=start, end_line=end),
            parent=parent,
        )
        self.symbols.append(symbol)
        hash_start, hash_end = _definition_extent(node)
        self.symbol_hashes[qualified_name(symbol)] = self._body_hash(
            hash_start, hash_end, excluded_spans
        )

    def _body_hash(
        self,
        start_line: int,
        end_line: int,
        excluded_spans: Sequence[tuple[int, int]] = (),
    ) -> str:
        """Hash of the symbol's own source text, whitespace-normalised per line.

        Normalising trailing whitespace and blank lines means a pure
        reformatting edit does not register as a semantic symbol change, while
        any real edit to the body does. Leading indentation is preserved because
        in Python it is semantically significant.

        ``excluded_spans`` are inclusive 1-based line ranges belonging to nested
        symbols that are hashed separately; subtracting them is what gives the
        symbol diff method-level rather than class-level granularity.
        """
        excluded: set[int] = set()
        for span_start, span_end in excluded_spans:
            excluded.update(range(span_start, span_end + 1))
        first = max(1, int(start_line))
        segment = self._lines[first - 1:end_line]
        normalised = "\n".join(
            line.rstrip()
            for number, line in enumerate(segment, start=first)
            if number not in excluded and line.strip()
        )
        return hashlib.sha256(normalised.encode("utf-8", "replace")).hexdigest()[:32]

    # -- references / imports ---------------------------------------------

    def _collect_reference(self, node: ast.AST) -> None:
        if isinstance(node, ast.Name):
            self.references.add(node.id)
        elif isinstance(node, ast.Attribute):
            # ``mod.symbol`` -> record ``symbol`` so a qualified call still
            # counts as a reference to the changed symbol.
            self.references.add(node.attr)

    def _collect_import(self, node: ast.AST) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                self.imports.append(
                    ImportRecord(module=alias.name, level=0, name=None, asname=alias.asname)
                )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            level = int(node.level or 0)
            for alias in node.names:
                if alias.name == "*":
                    # A star import binds an unknown set of names; treat the
                    # module edge as real but flag the file as not statically
                    # resolvable at the symbol level.
                    self.has_dynamic_imports = True
                    self.imports.append(ImportRecord(module=module, level=level, name=None))
                    continue
                self.imports.append(
                    ImportRecord(module=module, level=level, name=alias.name, asname=alias.asname)
                )
        elif isinstance(node, ast.Call):
            func = node.func
            called = ""
            if isinstance(func, ast.Name):
                called = func.id
            elif isinstance(func, ast.Attribute):
                called = func.attr
            if called in _DYNAMIC_IMPORT_NAMES:
                self.has_dynamic_imports = True

    def _collect_dunder_all(self, node: ast.AST) -> None:
        if not isinstance(node, ast.Assign):
            return
        for target in node.targets:
            if not (isinstance(target, ast.Name) and target.id == "__all__"):
                continue
            if isinstance(node.value, (ast.List, ast.Tuple)):
                for element in node.value.elts:
                    if isinstance(element, ast.Constant) and isinstance(element.value, str):
                        self.exported_names.append(element.value)
