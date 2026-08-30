"""Phase 4.18: explainable, provenance-typed dependency evidence.

:mod:`local_agent.semantic_impact` already builds a real import/reference graph
and ranks validation targets by evidence tier (``direct_symbol_match``,
``direct_import_match``, ``call_graph_match``, ...). What it could not do is
say *which specific language construct* produced a given piece of evidence: a
bare name reference, an import alias, a base class, a decorator, or a type
annotation were all folded into one undifferentiated "reference" set.

This module adds that layer. It does not replace the tier ladder or the scope
policy in :mod:`local_agent.semantic_impact` - those already enforce "more
uncertainty never means less validation" and are left untouched - it only
explains, for one candidate association, *why* it holds:

    "'total' is a local alias for 'calculate_total', imported from pkg.core"

rather than the coarser "references a changed symbol name".

Everything here is a pure function or a frozen, serialisable value: no I/O, no
mutable shared state, and every function's output depends only on its
arguments, so two calls with the same inputs always agree (a requirement of
:meth:`~local_agent.semantic_impact.ChangeImpactReport.fingerprint`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# -- evidence vocabulary -------------------------------------------------------

#: A bare reference to a symbol name that is also a changed symbol, backed by a
#: resolved import edge to the file that defines it.
DIRECT_SYMBOL = "direct_symbol_match"
#: A local name that resolves - through ``from x import y as z`` - back to a
#: changed symbol in a changed file. Strictly more precise than a bare name
#: match: the resolved import edge proves *which* definition is meant.
IMPORT_ALIAS = "import_alias_match"
#: ``import module_a as m`` followed by ``m.symbol()``. Handled for free by the
#: existing attribute-name reference capture (the alias never appears in the
#: attribute name), so this label exists for explainability, not new logic.
MODULE_ALIAS = "module_alias_match"
#: An attribute access (``obj.method()``) whose attribute name matches a
#: changed symbol name, without proof that ``obj`` is an instance of the
#: defining class. Deliberately lower confidence than :data:`DIRECT_SYMBOL`.
ATTRIBUTE_RESOLUTION = "attribute_resolution"
#: Phase 4.20. An attribute access whose *receiver* was resolved to a specific
#: class that the change actually touched - ``w = Widget(); w.render()`` where
#: ``Widget`` is imported from a changed file. Materially stronger than
#: :data:`ATTRIBUTE_RESOLUTION` because it no longer relies on the attribute
#: name alone being distinctive: the receiver's type is stated in the source.
#: Still below :data:`DIRECT_SYMBOL`, because a bare name reference to a
#: resolved import is more direct evidence than a two-step receiver walk.
ATTRIBUTE_RECEIVER_RESOLVED = "attribute_receiver_resolved"
#: The changed name appears as a base class in ``class X(Changed):``.
INHERITANCE = "inheritance_match"
#: The changed name appears as a parameter/return/variable type annotation.
ANNOTATION = "annotation_match"
#: The changed name appears as a decorator.
DECORATOR = "decorator_match"
#: Reached through a file that re-exports the changed symbol via ``__all__``.
REEXPORT = "reexport_match"
#: ``importlib.import_module("literal")`` / ``__import__("literal")`` whose
#: argument was a statically-resolvable literal string.
DYNAMIC_IMPORT_RESOLVED = "dynamic_import_resolved"
#: The same call family, but the argument could not be resolved statically
#: (a variable, an f-string, or a package-relative literal).
DYNAMIC_IMPORT_UNRESOLVED = "dynamic_import_unresolved"
#: Filename/module-naming convention only; no graph evidence at all.
LEXICAL_FALLBACK = "lexical_fallback"

#: Deterministic confidence per evidence type, in ``[0.0, 1.0]``. This is a
#: fixed table, not a learned or adjustable score - Part B is explicit that
#: opaque scoring is not wanted, and a fixed table is what stays reviewable and
#: reproducible. Ordering mirrors (but does not replace) the tier weights in
#: :mod:`local_agent.semantic_impact`: this is finer-grained *explanation*
#: layered on top of, not a substitute for, that already-audited ranking.
CONFIDENCE_BY_EVIDENCE_TYPE: dict[str, float] = {
    DIRECT_SYMBOL: 1.0,
    IMPORT_ALIAS: 0.95,
    MODULE_ALIAS: 0.95,
    INHERITANCE: 0.85,
    DECORATOR: 0.8,
    ANNOTATION: 0.7,
    REEXPORT: 0.75,
    ATTRIBUTE_RECEIVER_RESOLVED: 0.75,
    DYNAMIC_IMPORT_RESOLVED: 0.6,
    ATTRIBUTE_RESOLUTION: 0.5,
    DYNAMIC_IMPORT_UNRESOLVED: 0.0,
    LEXICAL_FALLBACK: 0.2,
}

#: Every evidence type this module can emit, for validation and iteration.
ALL_EVIDENCE_TYPES: frozenset[str] = frozenset(CONFIDENCE_BY_EVIDENCE_TYPE)


def confidence_for(evidence_type: str) -> float:
    """Deterministic confidence for ``evidence_type``; ``0.0`` if unknown.

    An unrecognised type - e.g. evidence recorded by a newer schema version and
    loaded by an older build - fails closed to the *lowest* confidence rather
    than raising or defaulting to something moderate.
    """
    return CONFIDENCE_BY_EVIDENCE_TYPE.get(evidence_type, 0.0)


@dataclass(frozen=True)
class DependencyEvidence:
    """One explained edge: why ``source`` is believed to depend on ``target``.

    Frozen and hashable so a set of these can be deduplicated cheaply. Every
    field is a plain string/float, so serialisation is a direct dict, and a
    payload from an unknown future schema deserialises tolerantly (unknown
    fields dropped, missing fields defaulted) rather than raising.
    """

    source_file: str
    target_file: str
    evidence_type: str
    #: The symbol believed responsible, when there is a single specific one
    #: (empty for file-level evidence such as a plain module import).
    target_symbol: str = ""
    #: The name actually written in ``source_file`` that led here - may differ
    #: from ``target_symbol`` (a local alias, a decorator name, ...).
    source_reference: str = ""
    confidence: float = 0.0
    #: One human-readable sentence explaining this specific edge.
    provenance: str = ""
    resolution_notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_file": self.source_file,
            "target_file": self.target_file,
            "evidence_type": self.evidence_type,
            "target_symbol": self.target_symbol,
            "source_reference": self.source_reference,
            "confidence": round(self.confidence, 4),
            "provenance": self.provenance,
            "resolution_notes": list(self.resolution_notes),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "DependencyEvidence":
        if not isinstance(data, dict):
            return cls(source_file="", target_file="", evidence_type=LEXICAL_FALLBACK)
        return cls(
            source_file=str(data.get("source_file", "")),
            target_file=str(data.get("target_file", "")),
            evidence_type=str(data.get("evidence_type", LEXICAL_FALLBACK)),
            target_symbol=str(data.get("target_symbol", "")),
            source_reference=str(data.get("source_reference", "")),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            provenance=str(data.get("provenance", "")),
            resolution_notes=tuple(str(n) for n in (data.get("resolution_notes") or ())),
        )


def make_evidence(
    *,
    source_file: str,
    target_file: str,
    evidence_type: str,
    target_symbol: str = "",
    source_reference: str = "",
    provenance: str = "",
    resolution_notes: tuple[str, ...] = (),
) -> DependencyEvidence:
    """Build one :class:`DependencyEvidence` with its table-driven confidence."""
    return DependencyEvidence(
        source_file=source_file,
        target_file=target_file,
        evidence_type=evidence_type,
        target_symbol=target_symbol,
        source_reference=source_reference,
        confidence=confidence_for(evidence_type),
        provenance=provenance,
        resolution_notes=resolution_notes,
    )


# -- alias resolution -----------------------------------------------------------


def resolve_alias_reference(
    *,
    source_file: str,
    reference: str,
    imported_symbol_origins: dict[str, dict[str, tuple[str, str]]],
    changed_files: frozenset[str],
    changed_symbol_names: frozenset[str],
    max_chain: int = 4,
) -> DependencyEvidence | None:
    """Resolve one local name in ``source_file`` back to a changed definition.

    ``imported_symbol_origins`` is
    :attr:`~local_agent.semantic_impact.SemanticGraph.imported_symbol_origins`:
    ``file -> local_alias -> (defining_file, original_symbol_name)``. It already
    holds exactly what is needed to answer "does this local name actually refer
    to something that changed?" - the graph just never asked the question, so a
    ``from module_a import calculate_total as total`` edge could only ever
    justify the weaker ``direct_import_match`` tier even when ``total`` really
    was the changed symbol.

    A re-export chain (``a`` imports ``calculate_total`` from ``b``, and a
    third file imports it from ``a``) is followed for up to ``max_chain`` hops,
    with a visited-set so a cyclic re-export cannot loop forever; exceeding the
    bound or hitting a cycle yields ``None`` (no evidence), never a guess.

    Returns ``None`` when ``reference`` is not a known alias, or resolves to a
    file/symbol that did not change - in both cases there is nothing to explain
    here and the caller's existing (safe) fallback tiers apply unchanged.
    """
    origins = imported_symbol_origins.get(source_file)
    if not origins or reference not in origins:
        return None

    visited: set[tuple[str, str]] = set()
    current_file, current_symbol = origins[reference]
    hops = 0
    while True:
        key = (current_file, current_symbol)
        if key in visited:
            return None
        visited.add(key)
        if current_file in changed_files and current_symbol in changed_symbol_names:
            note = (
                f"'{reference}' is a local alias for '{current_symbol}', imported "
                f"from {current_file}"
                if reference != current_symbol
                else f"'{reference}' is imported directly from {current_file}"
            )
            return make_evidence(
                source_file=source_file,
                target_file=current_file,
                evidence_type=IMPORT_ALIAS,
                target_symbol=current_symbol,
                source_reference=reference,
                provenance=note,
                resolution_notes=(f"resolved through {hops + 1} import hop(s)",) if hops else (),
            )
        hops += 1
        if hops >= max_chain:
            return None
        next_origins = imported_symbol_origins.get(current_file)
        if not next_origins or current_symbol not in next_origins:
            return None
        current_file, current_symbol = next_origins[current_symbol]


# -- attribute receiver resolution (Phase 4.20) ---------------------------------


#: Why a receiver could not be resolved. A vocabulary, not free text, so a
#: caller can count blind-spot causes without parsing sentences - and so the
#: telemetry key space stays bounded.
RECEIVER_UNBOUND = "receiver_not_bound_to_any_type"
RECEIVER_AMBIGUOUS = "receiver_bound_to_conflicting_types"
RECEIVER_LOCAL = "receiver_type_defined_locally"
RECEIVER_TYPE_UNRESOLVED = "receiver_type_not_traceable_to_a_changed_file"
RECEIVER_ATTRIBUTE_NOT_CHANGED = "attribute_is_not_a_changed_symbol"

ALL_RECEIVER_FAILURES: frozenset[str] = frozenset({
    RECEIVER_UNBOUND,
    RECEIVER_AMBIGUOUS,
    RECEIVER_LOCAL,
    RECEIVER_TYPE_UNRESOLVED,
    RECEIVER_ATTRIBUTE_NOT_CHANGED,
})


@dataclass(frozen=True)
class ReceiverResolution:
    """Outcome of one attribute-receiver resolution attempt.

    Carries the *failure reason* as well as the success case, because "we could
    not tell what ``obj`` is" and "we could tell, and it is not related to the
    change" are completely different facts about the analysis, and only the
    first is a blind spot.
    """

    evidence: DependencyEvidence | None = None
    reason: str = ""

    @property
    def resolved(self) -> bool:
        return self.evidence is not None


def resolve_attribute_receiver(
    *,
    source_file: str,
    receiver: str,
    attribute: str,
    local_type_bindings: dict[str, str],
    ambiguous_bindings: frozenset[str],
    class_bases: dict[str, tuple[str, ...]],
    locally_defined_names: frozenset[str],
    imported_symbol_origins: dict[str, dict[str, tuple[str, str]]],
    changed_files: frozenset[str],
    changed_symbol_names: frozenset[str],
    symbols_by_file: dict[str, frozenset[str]] | None = None,
) -> ReceiverResolution:
    """Resolve ``receiver.attribute`` to a changed definition, or explain why not.

    **What this deliberately is not.** It is not type inference. There is no
    dataflow, no call-graph return-type propagation, no cross-module
    resolution of what a factory function hands back, and no unification. Every
    rule below reads a type that the source *states*, in one step, and stops.
    That ceiling is the point: a speculative resolution that is wrong produces
    a *false confident edge*, and a false confident edge is exactly the input
    that could justify narrowing validation. A refusal, by contrast, leaves the
    pre-existing (weaker, safe) attribute-name evidence in place and can only
    ever make validation broader.

    The rules, all of which require the receiver's type to be written down:

    1. ``receiver`` is bound to exactly one type name in this file, via a
       constructor call, an annotated assignment, or a parameter annotation
       (:attr:`~local_agent.indexing.ast_python_indexer.PythonFileFacts.local_type_bindings`);
    2. that type name resolves - through the already-audited import-origin map,
       following re-export chains via :func:`resolve_alias_reference` - to a
       symbol defined in a *changed* file;
    3. ``attribute`` is itself one of the changed symbol names.

    Plus one inheritance rule: ``self.attribute`` inside a class whose base
    class resolves to a changed file, which is the same three steps with the
    base class standing in for the binding.

    Refusals, each with its own reason code:

    * the receiver was never bound to a type -> :data:`RECEIVER_UNBOUND`;
    * it was bound to two conflicting types -> :data:`RECEIVER_AMBIGUOUS`
      (ambiguity is never broken by picking one);
    * the type is defined in *this* file -> :data:`RECEIVER_LOCAL`, a resolved
      *negative*: the attribute provably belongs to local code;
    * the type does not trace to a changed file -> :data:`RECEIVER_TYPE_UNRESOLVED`;
    * the attribute is not a changed symbol -> :data:`RECEIVER_ATTRIBUTE_NOT_CHANGED`.
    """
    if attribute not in changed_symbol_names:
        return ReceiverResolution(reason=RECEIVER_ATTRIBUTE_NOT_CHANGED)

    type_name = ""
    via = ""
    if receiver == "self":
        # Inheritance rule: ``self.attribute`` is attributable only when the
        # enclosing module declares exactly one class whose bases we know, and
        # one of those bases resolves to the change. More than one candidate
        # base class means we cannot say which ``self`` is, so we decline.
        candidates = sorted({
            base for bases in class_bases.values() for base in bases
        })
        if len(candidates) != 1:
            return ReceiverResolution(
                reason=RECEIVER_UNBOUND if not candidates else RECEIVER_AMBIGUOUS
            )
        type_name = candidates[0]
        via = "inherited"
    else:
        if receiver in ambiguous_bindings:
            return ReceiverResolution(reason=RECEIVER_AMBIGUOUS)
        # A receiver written as a constructor call (``Widget().render()``) is
        # recorded with the class name in the receiver slot itself, so a name
        # that is a known type needs no binding lookup.
        type_name = local_type_bindings.get(receiver, "")
        if not type_name and (
            receiver in locally_defined_names
            or receiver in (imported_symbol_origins.get(source_file) or {})
        ):
            type_name = receiver
        if not type_name:
            return ReceiverResolution(reason=RECEIVER_UNBOUND)
        via = "bound"

    if type_name in locally_defined_names:
        # Defined right here: a resolved negative, not a blind spot.
        return ReceiverResolution(reason=RECEIVER_LOCAL)

    alias = resolve_alias_reference(
        source_file=source_file,
        reference=type_name,
        imported_symbol_origins=imported_symbol_origins,
        changed_files=changed_files,
        changed_symbol_names=frozenset(changed_symbol_names | {type_name}),
    )
    if alias is None:
        return ReceiverResolution(reason=RECEIVER_TYPE_UNRESOLVED)

    target_file = alias.target_file
    if target_file not in changed_files:
        return ReceiverResolution(reason=RECEIVER_TYPE_UNRESOLVED)
    if symbols_by_file is not None:
        defined = symbols_by_file.get(target_file, frozenset())
        # The resolved class must actually live in the changed file, and the
        # attribute must be a name that file defines. Without this the rule
        # would happily attribute ``widget.render()`` to a changed file that
        # defines ``Widget`` but has no ``render`` at all.
        if alias.target_symbol not in defined or attribute not in defined:
            return ReceiverResolution(reason=RECEIVER_TYPE_UNRESOLVED)

    phrase = (
        f"'{source_file}' calls '{receiver}.{attribute}' where '{receiver}' inherits from "
        f"'{alias.target_symbol}'"
        if via == "inherited"
        else f"'{source_file}' calls '{receiver}.{attribute}' where '{receiver}' is a "
        f"'{alias.target_symbol}'"
    )
    return ReceiverResolution(
        evidence=make_evidence(
            source_file=source_file,
            target_file=target_file,
            evidence_type=ATTRIBUTE_RECEIVER_RESOLVED,
            target_symbol=attribute,
            source_reference=f"{receiver}.{attribute}",
            provenance=f"{phrase}, defined in {target_file} and changed by this edit",
            resolution_notes=(
                f"receiver type '{type_name}' resolved via {via} declaration",
            ),
        ),
        reason="",
    )
