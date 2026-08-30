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

#: Callables that name-resolve a module by string; a *literal* string argument
#: can be resolved just like a static import (see ``_try_resolve_dynamic_call``).
#: ``load_module`` is the legacy ``imp`` equivalent and is treated the same way.
_MODULE_RESOLVING_CALLS = frozenset({"__import__", "import_module", "load_module"})
#: Calls whose argument can be an arbitrary program, not just a module name -
#: even a literal string argument tells us nothing about *what modules it uses*,
#: so these can never be resolved and always degrade confidence.
_OPAQUE_EXECUTION_CALLS = frozenset({"exec", "eval"})
_DYNAMIC_IMPORT_NAMES = _MODULE_RESOLVING_CALLS | _OPAQUE_EXECUTION_CALLS


@dataclass(frozen=True)
class ImportRecord:
    """One imported thing, preserving enough structure to resolve it later.

    ``import a.b``            -> ImportRecord("a.b", 0, None, None)
    ``import a.b as x``       -> ImportRecord("a.b", 0, None, "x")
    ``from a.b import c``     -> ImportRecord("a.b", 0, "c", None)
    ``from . import c``       -> ImportRecord("", 1, "c", None)
    ``from ..x import c as d``-> ImportRecord("x", 2, "c", "d")
    ``importlib.import_module("a.b")`` -> ImportRecord("a.b", 0, None, None, dynamic=True)
    """

    module: str
    level: int = 0
    name: str | None = None
    asname: str | None = None
    #: True when this edge came from a resolved *dynamic* call
    #: (``importlib.import_module("literal")``, ``__import__("literal")``)
    #: rather than a static ``import``/``from`` statement. Carried through so
    #: downstream provenance can say "resolved from a dynamic import call"
    #: instead of claiming the same certainty as a plain import statement.
    dynamic: bool = False

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
    #: True when the module performs imports that cannot be statically resolved:
    #: a star import, ``exec``/``eval`` at all, or a dynamic module-loading call
    #: (``importlib.import_module``, ``__import__``, ``imp.load_module``) whose
    #: argument is not a literal string. A dynamic call *with* a literal string
    #: argument is resolved into a normal (dynamic-flagged) entry in ``imports``
    #: instead - see :data:`_MODULE_RESOLVING_CALLS` - so it does NOT set this.
    has_dynamic_imports: bool = False
    #: Non-empty when the file could not be parsed; symbols/imports are then empty.
    parse_error: str = ""
    #: Module-level ``__all__`` entries, when declared as a literal list/tuple.
    exported_names: tuple[str, ...] = ()
    #: Identifiers referenced specifically as a base class in a ``class X(Y):``
    #: statement. A subset of ``references`` kept separately so a dependency on
    #: ``Y`` can be explained as *inheritance*, not a generic name match.
    base_class_references: frozenset[str] = frozenset()
    #: Identifiers referenced specifically as a decorator (``@deco``).
    decorator_references: frozenset[str] = frozenset()
    #: Identifiers referenced specifically in a type annotation (parameter,
    #: return type, or ``x: SomeType`` variable annotation).
    annotation_references: frozenset[str] = frozenset()
    #: Identifiers referenced through ``obj.name`` attribute access - a subset
    #: of ``references``, kept separately because it is weaker evidence: it
    #: proves nothing about what ``obj`` actually is (see ``ATTRIBUTE_RESOLUTION``
    #: in :mod:`local_agent.dependency_resolution`).
    attribute_references: frozenset[str] = frozenset()
    # --- Phase 4.20: attribute *receiver* facts -------------------------------
    #: ``(receiver, attribute)`` pairs for attribute accesses whose receiver is
    #: a plain name (``obj.method()`` -> ``("obj", "method")``) or an immediate
    #: constructor call (``Widget().method()`` -> ``("Widget", "method")``).
    #: Recorded so :func:`~local_agent.dependency_resolution.resolve_attribute_receiver`
    #: can *sometimes* say which class ``obj`` is, rather than always giving up.
    #: An access whose receiver is any more complex expression is deliberately
    #: not recorded at all - see that function's docstring.
    attribute_accesses: frozenset[tuple[str, str]] = frozenset()
    #: Local name -> the type name it is confidently bound to, from a
    #: constructor assignment (``x = Widget()``), an annotated assignment
    #: (``x: Widget = ...``) or a parameter/variable annotation. A name bound to
    #: two *different* type names anywhere in the module is omitted entirely
    #: rather than resolved to either - ambiguity must not become a guess.
    local_type_bindings: dict[str, str] = field(default_factory=dict)
    #: Names bound more than once to conflicting types, kept so a caller can
    #: explain *why* a receiver was left unresolved instead of it looking like
    #: the name was simply never seen.
    ambiguous_bindings: frozenset[str] = frozenset()
    #: Class name -> its base-class names, for ``self.method()`` resolution
    #: inside a subclass of a changed class.
    class_bases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: Every class/function name defined in *this* module. A receiver bound to
    #: a locally-defined class is a resolved *negative*: it proves the attribute
    #: belongs to local code, not to the changed module.
    locally_defined_names: frozenset[str] = frozenset()

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
            base_class_references=frozenset(collector.base_class_references),
            decorator_references=frozenset(collector.decorator_references),
            annotation_references=frozenset(collector.annotation_references),
            attribute_references=frozenset(collector.attribute_references),
            attribute_accesses=frozenset(collector.attribute_accesses),
            local_type_bindings=dict(collector.resolved_bindings()),
            ambiguous_bindings=frozenset(collector.ambiguous_bindings),
            class_bases=dict(collector.class_bases),
            locally_defined_names=frozenset(
                symbol.name for symbol in collector.symbols
            ),
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
        self.base_class_references: set[str] = set()
        self.decorator_references: set[str] = set()
        self.annotation_references: set[str] = set()
        self.attribute_references: set[str] = set()
        # --- Phase 4.20 ---
        self.attribute_accesses: set[tuple[str, str]] = set()
        #: name -> every distinct type name it was ever bound to in this module.
        #: Kept as a set per name (rather than last-write-wins) precisely so a
        #: conflicting rebind is *detectable*; see :meth:`resolved_bindings`.
        self._bindings: dict[str, set[str]] = {}
        self.class_bases: dict[str, tuple[str, ...]] = {}

    @property
    def ambiguous_bindings(self) -> set[str]:
        return {name for name, types in self._bindings.items() if len(types) > 1}

    def resolved_bindings(self) -> dict[str, str]:
        """Only the names bound to exactly one type. Ambiguity yields nothing."""
        return {
            name: next(iter(types))
            for name, types in self._bindings.items()
            if len(types) == 1 and not next(iter(types)).startswith("\x00")
        }

    def _bind(self, name: str, type_name: str) -> None:
        if not name or not type_name or name == type_name:
            return
        self._bindings.setdefault(name, set()).add(type_name)

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
            self._collect_reference_kind(node)
            self._collect_binding(node)

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
            self.attribute_references.add(node.attr)
            # Phase 4.20: also remember *what the receiver was written as*, but
            # only for the two shapes that can be resolved without inventing a
            # type inference engine. Anything else (a subscript, another
            # attribute chain, a comprehension variable, a function return) is
            # left unrecorded rather than guessed at.
            receiver = node.value
            if isinstance(receiver, ast.Name):
                self.attribute_accesses.add((receiver.id, node.attr))
            elif isinstance(receiver, ast.Call) and isinstance(receiver.func, ast.Name):
                # ``Widget().method()`` - the constructor is right there, so the
                # receiver's type is known exactly, with no inference at all.
                self.attribute_accesses.add((receiver.func.id, node.attr))

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
            if called in _MODULE_RESOLVING_CALLS:
                literal = _literal_module_argument(node)
                if literal is not None:
                    # A resolvable literal is exactly as knowable as a static
                    # import: reuse the same edge machinery, just labelled
                    # ``dynamic`` so provenance can say where it came from.
                    self.imports.append(
                        ImportRecord(module=literal, level=0, name=None, asname=None, dynamic=True)
                    )
                    return
                # Phase 4.20: a *package-relative* literal
                # (``import_module(".sub", package="pkg")``) is equally knowable
                # when the anchoring package is itself a literal, because that
                # is precisely the information ``importlib`` itself uses. Only
                # that exact shape is resolved; see ``_relative_module_argument``.
                relative = _relative_module_argument(node)
                if relative is not None:
                    self.imports.append(
                        ImportRecord(module=relative, level=0, name=None, asname=None, dynamic=True)
                    )
                    return
                # A computed/variable argument, or a relative import with no
                # statically-known anchoring package: genuinely cannot know the
                # target without running the program, so this stays a real
                # degradation.
                self.has_dynamic_imports = True
            elif called in _OPAQUE_EXECUTION_CALLS:
                # ``exec``/``eval`` can do anything a literal string can't tell
                # us about, even when the argument itself is a literal.
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

    def _collect_reference_kind(self, node: ast.AST) -> None:
        """Tag a subset of references by *where* they were used.

        These are strict subsets of the generic ``references`` set collected by
        :meth:`_collect_reference` above; nothing here changes which names are
        "referenced", only which ones can be explained as an inheritance,
        decorator, or annotation dependency rather than an ordinary name use.
        """
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                _collect_names(base, self.base_class_references)
            for decorator in node.decorator_list:
                _collect_names(decorator, self.decorator_references)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                _collect_names(decorator, self.decorator_references)
            args = node.args
            all_args = [*args.posonlyargs, *args.args, *args.kwonlyargs]
            if args.vararg is not None:
                all_args.append(args.vararg)
            if args.kwarg is not None:
                all_args.append(args.kwarg)
            for argument in all_args:
                if argument.annotation is not None:
                    _collect_names(argument.annotation, self.annotation_references)
            if node.returns is not None:
                _collect_names(node.returns, self.annotation_references)
        elif isinstance(node, ast.AnnAssign) and node.annotation is not None:
            _collect_names(node.annotation, self.annotation_references)

    def _collect_binding(self, node: ast.AST) -> None:
        """Phase 4.20: record the few name->type bindings that are *certain*.

        Four shapes, all of which state the type explicitly in the source with
        no inference required:

        1. ``x = Widget()``            - a direct constructor call;
        2. ``x: Widget = ...`` / ``x: Widget`` - an annotated assignment;
        3. ``def f(x: Widget)``        - an annotated parameter;
        4. ``class C(Base)``           - recorded into :attr:`class_bases` so
           ``self.method()`` inside ``C`` can be attributed to ``Base``.

        Everything else - a factory function's return value, a container
        element, a conditional rebind, ``*args`` - is deliberately not bound.
        Binding a name is a *claim*, and an unbound name simply falls through
        to the pre-existing (weaker, safe) attribute-name evidence.

        A bare ``Name`` annotation is used; a subscripted one (``list[Widget]``)
        is declined, because the receiver is then the container, not the widget.
        """
        if isinstance(node, ast.Assign):
            value = node.value
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self._bind(target.id, value.func.id)
            elif isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
                # ``x = module.Widget()`` - the attribute name is the class.
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self._bind(target.id, value.func.attr)
            else:
                # A rebind to anything we cannot type still has to be *seen*, or
                # ``x = Widget()`` followed by ``x = make_thing()`` would look
                # unambiguous. Recording a sentinel makes it ambiguous instead.
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self._bind(target.id, "\x00unknown")
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and isinstance(node.annotation, ast.Name):
                self._bind(node.target.id, node.annotation.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            for argument in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
                if isinstance(argument.annotation, ast.Name):
                    self._bind(argument.arg, argument.annotation.id)
        elif isinstance(node, ast.ClassDef):
            bases: list[str] = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(base.attr)
            if bases:
                self.class_bases[node.name] = tuple(bases)


def _collect_names(expr: ast.expr, into: set[str]) -> None:
    """Every ``Name``/``Attribute`` identifier within one expression subtree."""
    for node in ast.walk(expr):
        if isinstance(node, ast.Name):
            into.add(node.id)
        elif isinstance(node, ast.Attribute):
            into.add(node.attr)


def _literal_module_argument(call: ast.Call) -> str | None:
    """The dotted module string a dynamic-import call resolves to, if knowable.

    Returns ``None`` - meaning "cannot be resolved statically" - for anything
    other than a plain string constant: an f-string, a variable, string
    concatenation, or a leading-dot (package-relative) string, which would need
    the call's ``package=`` keyword resolved too and is deliberately not
    attempted. A non-identifier-shaped string is also declined, since it is
    almost certainly runtime-computed content rather than a module path.
    """
    if not call.args:
        return None
    first = call.args[0]
    if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
        return None
    value = first.value.strip()
    if not value or value.startswith("."):
        return None
    parts = value.split(".")
    if not all(part.isidentifier() for part in parts):
        return None
    return value


def _relative_module_argument(call: ast.Call) -> str | None:
    """Phase 4.20: absolute module name for a package-relative dynamic import.

    ``importlib.import_module(".sub", package="pkg.core")`` is not a guess: it
    resolves by exactly the documented rule ``importlib`` itself applies - the
    leading dots count levels up from ``package``, and the remainder is
    appended. When both the module string and ``package`` are literal
    constants, the answer is fully determined at parse time and is therefore
    every bit as knowable as ``from pkg import sub``.

    Resolution is refused, returning ``None`` (which leaves the file flagged as
    having unresolvable dynamic imports, exactly as before), whenever any part
    of that rule cannot be evaluated statically:

    * the first argument is not a literal string, or does not start with a dot;
    * there is no ``package=`` keyword, or its value is not a literal string -
      ``importlib`` would fall back to the *caller's* ``__package__``, which is
      a runtime property this indexer must not assume;
    * ``package`` is not a dotted identifier path;
    * the dot count walks above the root of ``package`` (which is a
      ``ValueError`` at runtime, not a module);
    * the remainder after the dots is not a dotted identifier path.
    """
    if not call.args:
        return None
    first = call.args[0]
    if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
        return None
    raw = first.value.strip()
    if not raw.startswith("."):
        return None

    package: str | None = None
    for keyword in call.keywords:
        if keyword.arg == "package":
            if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                package = keyword.value.value.strip()
            break
    if not package:
        return None
    package_parts = package.split(".")
    if not all(part.isidentifier() for part in package_parts):
        return None

    level = len(raw) - len(raw.lstrip("."))
    remainder = raw[level:]
    # ``import_module(".sub", package="pkg")`` -> level 1 -> anchored at ``pkg``
    # itself; ``..sub`` -> level 2 -> anchored at ``pkg``'s parent.
    drop = level - 1
    if drop > len(package_parts) - 1:
        # Walks above the root: at runtime this raises, so there is no module
        # to point an edge at.
        return None
    base = package_parts[: len(package_parts) - drop] if drop else package_parts
    pieces = [piece for piece in base if piece]
    if remainder:
        remainder_parts = remainder.split(".")
        if not all(part.isidentifier() for part in remainder_parts):
            return None
        pieces.extend(remainder_parts)
    if not pieces:
        return None
    return ".".join(pieces)
