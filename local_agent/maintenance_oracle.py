"""Phase 4.23: the maintenance execution oracle framework.

Phase 4.22 wired exactly one signal - ``parse_failure`` - through the real
execution pipeline, and it decided "did the repair work?" with two inline calls
to a private helper. That worked, but it hid the question that actually governs
autonomy:

    *Does this signal have a success criterion a machine can check, such that
    satisfying it implies the defect is genuinely gone?*

This module makes that question first-class. Every maintenance signal is bound
to an :class:`ExecutionOracle`, the oracle declares which class of evidence it
can produce, and **only a deterministic oracle may unlock autonomous
execution**. Everything else is refused structurally, before any workspace
exists.

Why an oracle rather than a flag
--------------------------------
A boolean "this signal is safe" on a config table is a claim. An oracle is an
*obligation*: to be registered as promotable it must be able to answer, in
code, all six questions the framework asks -

``BEFORE``
    What exactly constitutes failure? :meth:`ExecutionOracle.observe_failure`.
``AFTER``
    What exact evidence constitutes successful remediation?
    :meth:`ExecutionOracle.observe_success`.
``SCOPE``
    Which files and commands may be examined?
    :attr:`ExecutionOracle.max_scope_files`, :attr:`supported_suffixes`,
    :meth:`acceptance_commands`.
``CONFIDENCE``
    Deterministic or heuristic? :attr:`ExecutionOracle.oracle_class`.
``INCONCLUSIVE``
    What happens when success cannot be established?
    :data:`OracleOutcome.INCONCLUSIVE` - which is never a pass.
``SAFETY``
    Can history override a current safety concern? **No.** No oracle in this
    module reads persisted history at all; every verdict is computed from the
    bytes on disk at the moment it is asked. The absence of any such input is
    asserted structurally by the test-suite from the AST.

Fail-closed by construction
---------------------------
:class:`OracleObservation` has no "assume success" path. ``resolved`` is a
property that is true only for :data:`OracleOutcome.RESOLVED`, and the base
class refuses to produce that outcome; a subclass must compute it from
observed evidence. "Nothing ran", "the tool was missing", "the file could not
be read" and "the evidence was malformed" all map to
:data:`OracleOutcome.INCONCLUSIVE`, which callers must treat exactly as they
treat failure.

The post-condition is stronger than the negation of the pre-condition
---------------------------------------------------------------------
This is the correction Phase 4.23 makes to Phase 4.22. That build's success
predicate for ``parse_failure`` was literally "the file now parses", and a file
whose entire contents have been replaced by ``pass`` parses. A destructive
"repair" was therefore credited as a completed, validated, resolved
maintenance action - verified by experiment against the 4.22 build, not
inferred. :class:`ParseOracle` now additionally requires that the repair
*preserved the module*: every definition, class and import name that was
lexically present in the broken source must still be present in the parsed
result, and the file may not shrink beyond a bounded tolerance. See
:class:`ParseOracle` for the full predicate and its failure modes.
"""

from __future__ import annotations

import ast
import difflib
import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .maintenance import (
    ALL_SIGNAL_KINDS,
    MaintenanceSignal,
    sanitize_text,
)

__all__ = [
    "ALL_ORACLE_CLASSES",
    "ALL_ORACLE_OUTCOMES",
    "AUTONOMOUS_SIGNAL_KINDS",
    "ExecutionOracle",
    "OracleClass",
    "OracleObservation",
    "OracleOutcome",
    "ORACLE_FRAMEWORK_VERSION",
    "ParseOracle",
    "SIGNAL_INVENTORY",
    "SignalInventoryEntry",
    "UnverifiableOracle",
    "inventory_for",
    "oracle_for",
]

#: Version of the oracle contract. Folded into the executor's policy
#: fingerprint so a work order planned under one oracle generation is never
#: silently executed by another.
ORACLE_FRAMEWORK_VERSION = "4.23.0"


# =============================================================================
# Vocabulary
# =============================================================================


class OracleClass:
    """How strong the evidence an oracle can produce actually is.

    These are the four categories Phase 4.23 is required to distinguish, and
    the ordering is meaningful: only :data:`DETERMINISTIC` may ever unlock
    autonomous execution.
    """

    #: The verdict is a structural fact, recomputed from the current bytes, and
    #: is reproducible on any machine with no model, no history and no rate.
    DETERMINISTIC = "deterministic"
    #: The verdict rests on a rate, a bound or a sample. Real evidence, but it
    #: cannot prove that *this* defect is gone.
    STATISTICAL = "statistical"
    #: The condition is observable but what counts as "fixed" is a judgement.
    AMBIGUOUS = "ambiguous"
    #: Automating a remediation here would be unsafe regardless of evidence.
    UNSAFE = "unsafe"


ALL_ORACLE_CLASSES: tuple[str, ...] = (
    OracleClass.DETERMINISTIC,
    OracleClass.STATISTICAL,
    OracleClass.AMBIGUOUS,
    OracleClass.UNSAFE,
)

#: The only class from which autonomy may be granted. Referenced rather than
#: inlined so the rule has exactly one definition.
PROMOTABLE_ORACLE_CLASSES: frozenset[str] = frozenset({OracleClass.DETERMINISTIC})


class OracleOutcome:
    """The three answers an oracle may give. There is no fourth."""

    #: The success predicate holds, on evidence observed now.
    RESOLVED = "resolved"
    #: The success predicate demonstrably does not hold.
    NOT_RESOLVED = "not_resolved"
    #: The oracle could not establish either. Treated exactly as failure by
    #: every caller; never convertible to success.
    INCONCLUSIVE = "inconclusive"


ALL_ORACLE_OUTCOMES: tuple[str, ...] = (
    OracleOutcome.RESOLVED,
    OracleOutcome.NOT_RESOLVED,
    OracleOutcome.INCONCLUSIVE,
)


@dataclass(frozen=True)
class OracleObservation:
    """One observation, with the evidence that produced it.

    Frozen on purpose: an observation is a record of what was true when it was
    taken. A caller that could mutate ``outcome`` after the fact would be able
    to launder an inconclusive result into a pass, which is precisely the
    failure mode the framework exists to prevent.
    """

    outcome: str = OracleOutcome.INCONCLUSIVE
    oracle_name: str = ""
    signal_kind: str = ""
    #: True only when the outcome was computed by a deterministic oracle.
    deterministic: bool = False
    detail: str = ""
    #: Structured, JSON-safe evidence. Never contains file contents.
    evidence: Mapping[str, Any] = field(default_factory=dict)
    #: Predicate clauses that were evaluated, each ``(name, satisfied)``.
    clauses: tuple[tuple[str, bool], ...] = ()

    def __post_init__(self) -> None:
        # ``frozen`` protects the fields, not the objects they point at, and
        # ``evidence`` is the BEFORE baseline the success predicate is measured
        # against. A plain dict would let any code holding the observation
        # rewrite that baseline between the two observations - emptying
        # ``lexical_surface`` makes the preservation clause vacuously true,
        # which is exactly the destructive-repair hole this framework exists to
        # close. A read-only view removes the possibility rather than relying
        # on nobody trying.
        from types import MappingProxyType

        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))

    @property
    def resolved(self) -> bool:
        """True *only* for an explicit RESOLVED verdict."""
        return self.outcome == OracleOutcome.RESOLVED

    @property
    def failing(self) -> bool:
        """True only when failure was positively observed.

        Deliberately not ``not self.resolved``: an inconclusive observation is
        neither, and a pre-condition gate that treated it as "the defect is
        present" would execute against a repository it could not read.
        """
        return self.outcome == OracleOutcome.NOT_RESOLVED

    @property
    def inconclusive(self) -> bool:
        return self.outcome == OracleOutcome.INCONCLUSIVE

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "oracle": self.oracle_name,
            "signal_kind": self.signal_kind,
            "deterministic": self.deterministic,
            "detail": self.detail,
            "evidence": dict(self.evidence),
            "clauses": [[name, bool(ok)] for name, ok in self.clauses],
        }


# =============================================================================
# The oracle contract
# =============================================================================


class ExecutionOracle:
    """Base contract. Refuses everything; subclasses earn their verdicts.

    The base implementation returns :data:`OracleOutcome.INCONCLUSIVE` from
    both observation methods. That is not a placeholder - it is the safe
    default, and it means a subclass that forgets to implement a method
    degrades to "cannot establish success" rather than to "success".
    """

    #: Which signal this oracle answers for.
    signal_kind: str = ""
    #: Stable identifier used in telemetry and evidence.
    name: str = "unverifiable"
    #: See :class:`OracleClass`.
    oracle_class: str = OracleClass.UNSAFE
    #: Maximum number of files a remediation for this signal may touch.
    max_scope_files: int = 0
    #: File suffixes the oracle understands. Empty means "none".
    supported_suffixes: frozenset[str] = frozenset()

    # -- declared contract -------------------------------------------------

    @property
    def deterministic(self) -> bool:
        return self.oracle_class == OracleClass.DETERMINISTIC

    @property
    def promotable(self) -> bool:
        """Whether this oracle may unlock autonomous execution at all.

        A single expression with a single definition. Nothing else in the
        codebase decides this, and no configuration, persisted record or
        historical success can change it: it is a property of the oracle
        class, computed fresh on every access.
        """
        return self.oracle_class in PROMOTABLE_ORACLE_CLASSES

    def describe_failure_predicate(self) -> str:
        """BEFORE: what exactly constitutes failure."""
        return "no machine-checkable failure predicate is defined for this signal"

    def describe_success_predicate(self) -> str:
        """AFTER: what exact evidence constitutes successful remediation."""
        return "no machine-checkable success predicate is defined for this signal"

    # -- observation -------------------------------------------------------

    def observe_failure(self, root: Path, relative: str) -> OracleObservation:
        """BEFORE. ``failing`` must be true for execution to be permitted."""
        return self._inconclusive(
            "this signal has no deterministic failure predicate, so its presence "
            "cannot be re-established at execution time"
        )

    def observe_success(
        self,
        root: Path,
        relative: str,
        before: OracleObservation | None = None,
    ) -> OracleObservation:
        """AFTER. ``resolved`` must be true for the change to be credited."""
        return self._inconclusive(
            "this signal has no deterministic success predicate, so a repair "
            "cannot be mechanically verified"
        )

    # -- what the executor needs to drive a repair -------------------------

    def acceptance_commands(self, root: Path, changed: Sequence[str]) -> list[Any]:
        """Mandatory commands this signal's acceptance requires.

        Additive only: the executor prepends these to whatever the validation
        authority selected, so an oracle can cause *more* validation to run and
        never less. Returning an empty list is always safe.
        """
        return []

    def plan_fragment(self, relative: str) -> Mapping[str, Any]:
        """Objective, steps and validation strategy for the coding agent.

        Kept on the oracle rather than in the executor so that the description
        of the repair and the definition of its success are written in one
        place and cannot drift apart.
        """
        return {
            "objective": f"Investigate {relative}",
            "steps": [],
            "validation_strategy": [],
            "risks": [],
        }

    # -- helpers -----------------------------------------------------------

    def _inconclusive(self, detail: str, **evidence: Any) -> OracleObservation:
        return OracleObservation(
            outcome=OracleOutcome.INCONCLUSIVE,
            oracle_name=self.name,
            signal_kind=self.signal_kind,
            deterministic=self.deterministic,
            detail=sanitize_text(detail, limit=400),
            evidence=dict(evidence),
        )

    def _observation(
        self,
        outcome: str,
        detail: str,
        *,
        clauses: Sequence[tuple[str, bool]] = (),
        **evidence: Any,
    ) -> OracleObservation:
        if outcome not in ALL_ORACLE_OUTCOMES:
            outcome = OracleOutcome.INCONCLUSIVE
        return OracleObservation(
            outcome=outcome,
            oracle_name=self.name,
            signal_kind=self.signal_kind,
            deterministic=self.deterministic,
            detail=sanitize_text(detail, limit=400),
            evidence=dict(evidence),
            clauses=tuple((str(name), bool(ok)) for name, ok in clauses),
        )


class UnverifiableOracle(ExecutionOracle):
    """The oracle for every signal that has no machine-checkable success test.

    It is a real object rather than ``None`` so that the executor's lookup can
    never produce a missing-oracle branch, and so that "we deliberately cannot
    verify this" is a value that can be inspected, printed and asserted on.

    Twelve of the thirteen maintenance signals are bound to one of these. The
    reason each is unverifiable is carried by its
    :class:`SignalInventoryEntry`, not invented here.
    """

    def __init__(self, signal_kind: str, oracle_class: str = OracleClass.AMBIGUOUS):
        self.signal_kind = str(signal_kind)
        self.oracle_class = (
            str(oracle_class) if oracle_class in ALL_ORACLE_CLASSES else OracleClass.UNSAFE
        )
        self.name = f"unverifiable:{self.signal_kind}"


# =============================================================================
# ParseOracle - the one deterministic oracle in this build
# =============================================================================

#: Definition-like names visible even in source that does not parse.
_DEF_PATTERN = re.compile(r"^[ \t]*(?:async[ \t]+)?def[ \t]+([A-Za-z_]\w*)", re.MULTILINE)
_CLASS_PATTERN = re.compile(r"^[ \t]*class[ \t]+([A-Za-z_]\w*)", re.MULTILINE)
_IMPORT_PATTERN = re.compile(
    r"^[ \t]*import[ \t]+([A-Za-z_][\w.]*)"
    r"|^[ \t]*from[ \t]+[.\w]+[ \t]+import[ \t]+(.+)$",
    re.MULTILINE,
)
#: A module-level binding: an identifier at column zero followed by ``=`` that
#: is not ``==``. Column zero matters - indented assignments are locals and are
#: not part of the module's surface.
_MODULE_BINDING_PATTERN = re.compile(
    r"^([A-Za-z_]\w*)[ \t]*(?::[^=\n]+)?=(?!=)", re.MULTILINE
)

#: How much of the broken file's substance a legitimate syntax repair may
#: remove. A syntax error is a local defect; a repair that deletes a quarter of
#: the significant lines is not repairing it, it is deleting the problem.
MAX_SHRINK_RATIO = 0.25
#: Absolute slack, so a three-line file is not held to a percentage.
SHRINK_SLACK_LINES = 3

#: How many of the *original* lines a repair may overwrite or drop.
#:
#: This clause exists because the shrink and surface clauses share a blind
#: spot, found by probing the implementation rather than by reading it: a
#: change that keeps every name and every line *count* while replacing each
#: function body with ``pass`` satisfies both. Line counts cannot separate that
#: from a legitimate repair - but the *direction* of the edit can. A real
#: syntax repair overwrites almost nothing; it inserts a colon, a bracket or a
#: quote. Gutting a module necessarily overwrites the lines it guts. So
#: insertions are unbounded here and overwrites are not.
MAX_REPLACED_LINE_RATIO = 0.10
#: Absolute slack for the same reason as above. Deliberately small: at three,
#: hollowing out the three bodies of a sixteen-line module fits inside the
#: allowance, which would make the clause decorative.
REPLACED_LINE_SLACK = 2


def _significant_lines(source: str) -> int:
    """Non-blank, non-comment lines. A crude but honest measure of substance."""
    count = 0
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        count += 1
    return count


def _line_digests(source: str) -> list[str]:
    """A content-free fingerprint of each line, in order.

    Digests rather than the lines themselves so the BEFORE observation - which
    is recorded in telemetry and in the evidence ledger - never carries source
    text. ``difflib`` only needs the sequence elements to be hashable and
    comparable, so a digest sequence diffs exactly as the real lines would.
    """
    normalised = source.replace("\r\n", "\n").replace("\r", "\n")
    return [
        hashlib.sha1(line.encode("utf-8", errors="replace")).hexdigest()[:12]
        for line in normalised.split("\n")
    ]


#: Above this many non-matching lines the exact diff is not computed.
#:
#: ``difflib.SequenceMatcher`` is quadratic in the presence of many identical
#: elements, and source files are full of them. Measured on this
#: implementation before the guard existed: a ten-thousand-line file took
#: 6.1 seconds inside the post-apply path, against 76 ms at one thousand lines
#: - an 80x cost for a 10x input. Since any input this large has already
#: blown through every allowance the clause could grant, the exact number is
#: not worth computing: an upper bound refuses just as correctly and does it in
#: linear time.
MAX_EXACT_DIFF_LINES = 2000


def _replaced_line_count(before: Sequence[str], after: Sequence[str]) -> int:
    """How many BEFORE lines a change overwrote or dropped.

    Pure insertions contribute zero by construction: an ``insert`` opcode
    consumes no BEFORE line. That asymmetry is the whole point of the clause -
    adding code to a file is not evidence of destruction, overwriting what was
    already there can be.

    The common prefix and suffix are trimmed first. A genuine syntax repair
    changes one line in the middle of an otherwise identical file, so trimming
    reduces the diff to a handful of lines and the quadratic matcher never sees
    a large input. When the remaining span is still enormous the count is
    returned as its own upper bound rather than computed exactly - see
    :data:`MAX_EXACT_DIFF_LINES`. That can only *over*-report, which can only
    make the clause refuse, which is the safe direction.
    """
    left = list(before)
    right = list(after)

    start = 0
    limit = min(len(left), len(right))
    while start < limit and left[start] == right[start]:
        start += 1
    end = 0
    while (
        end < limit - start
        and left[len(left) - 1 - end] == right[len(right) - 1 - end]
    ):
        end += 1

    middle_before = left[start : len(left) - end]
    middle_after = right[start : len(right) - end]
    if not middle_before:
        # Everything that survived the trim is an insertion.
        return 0
    if len(middle_before) > MAX_EXACT_DIFF_LINES:
        return len(middle_before)

    matcher = difflib.SequenceMatcher(a=middle_before, b=middle_after, autojunk=False)
    consumed = 0
    for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
        if tag != "equal":
            consumed += i2 - i1
    return consumed


def _replaced_line_allowance(before_lines: int) -> int:
    """How many original lines a legitimate repair may overwrite."""
    if before_lines <= 0:
        return REPLACED_LINE_SLACK
    return max(
        REPLACED_LINE_SLACK, math.ceil(before_lines * MAX_REPLACED_LINE_RATIO)
    )


def _lexical_surface(source: str) -> set[str]:
    """Definition/class/import/module-binding names visible in *unparsed* text.

    The broken file does not parse, so an AST is not available and a lexical
    scan is the only instrument there is. It is deliberately over-inclusive:
    a name that appears inside a string literal or a comment-like region will
    be picked up, which can only make the post-condition *stricter*. A stricter
    post-condition costs a rejected repair (which is rolled back), never a bad
    apply - so the error is biased in the safe direction by construction.
    """
    names: set[str] = set()
    names.update(_DEF_PATTERN.findall(source))
    names.update(_CLASS_PATTERN.findall(source))
    names.update(_MODULE_BINDING_PATTERN.findall(source))
    for plain, from_clause in _IMPORT_PATTERN.findall(source):
        if plain:
            names.add(plain.split(".")[0])
        if from_clause:
            for chunk in from_clause.split(","):
                token = chunk.strip()
                if not token or token.startswith("("):
                    token = token.lstrip("(").strip()
                if " as " in token:
                    token = token.split(" as ")[-1].strip()
                token = token.strip(") \t")
                if token and token != "*" and token.isidentifier():
                    names.add(token)
    names.discard("")
    return names


def _parsed_surface(tree: ast.Module) -> set[str]:
    """Every name a parsed module binds, matched to what the lexical scan sees.

    The two sides must be *symmetric*, or the comparison has holes:

    * ``def``/``class``/``import`` are collected at **any** nesting depth,
      because :func:`_lexical_surface` matches them at any indentation. A
      repair that moves a helper inside a class has not deleted it, and
      refusing that would be a false negative for no gain.
    * Plain assignments are collected at **module level only**, because
      :data:`_MODULE_BINDING_PATTERN` is anchored at column zero. Accepting a
      nested assignment here would let a change delete a module constant and
      "restore" it as a local inside some function.
    * Function *parameters* are deliberately not collected at all. They are
      never in the BEFORE set, so including them could only ever satisfy a
      missing name by coincidence - ``def f(CONSTANT)`` is not a replacement
      for a deleted ``CONSTANT = 42``.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)

    for statement in getattr(tree, "body", []):
        targets: list[ast.expr] = []
        if isinstance(statement, ast.Assign):
            targets = list(statement.targets)
        elif isinstance(statement, (ast.AnnAssign, ast.AugAssign)):
            targets = [statement.target]
        for target in targets:
            for node in ast.walk(target):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    names.add(node.id)

    names.discard("")
    return names


def _read_source(path: Path) -> tuple[str | None, str]:
    try:
        return path.read_text(encoding="utf-8"), ""
    except (OSError, UnicodeDecodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _parse_error(source: str, path: Path) -> str:
    """``""`` when the source parses; a short description when it does not."""
    try:
        compile(source, str(path), "exec")
    except SyntaxError as exc:
        return sanitize_text(f"SyntaxError: {exc.msg} (line {exc.lineno})", limit=200)
    except ValueError as exc:  # e.g. a NUL byte in the source
        return sanitize_text(f"ValueError: {exc}", limit=200)
    return ""


class ParseOracle(ExecutionOracle):
    """Deterministic oracle for ``parse_failure``.

    **Failure predicate F.** ``root/relative`` is a readable ``.py`` file and
    CPython's own compiler rejects it. This is a structural fact: no model, no
    rate, no history. It cannot be a false positive, and it is re-observed at
    execution time rather than trusted from the plan.

    **Success predicate S.** All four clauses must hold:

    ``parses``
        CPython's compiler accepts the file.
    ``surface_preserved``
        Every definition, class, import and module-level binding name that was
        lexically visible in the *broken* source is still bound in the parsed
        result.
    ``substance_preserved``
        The file's significant (non-blank, non-comment) line count did not drop
        by more than :data:`MAX_SHRINK_RATIO`, with :data:`SHRINK_SLACK_LINES`
        of absolute slack for very small files.
    ``repair_is_local``
        The change overwrote or dropped no more than
        :func:`_replaced_line_allowance` of the *original* lines. Insertions are
        unbounded; overwrites are not. See :data:`MAX_REPLACED_LINE_RATIO` for
        why this clause exists and what the first three missed.

    **Why S must be more than "it parses".** Phase 4.22's predicate was exactly
    that clause alone, and it is satisfiable by deleting the file's contents:
    ``pass`` parses. Verified by experiment against that build - a sixteen-line
    module containing a constant, two functions and a class was replaced
    wholesale with ``pass``, applied to the authoritative tree, and reported as
    ``completed`` with ``signal_resolved=True``. ``parses`` alone gives
    S ⇒ ¬F but *not* S ⇒ "the defect was repaired". The three additional
    clauses narrow that gap: a change satisfying all four has kept the module's
    entire named surface, substantially all of its body, and has overwritten
    almost none of the lines that were already there - so the edits that can
    satisfy S are, in the main, edits that changed syntax rather than content.

    **What S still does not prove - stated plainly.** S bounds *destruction*.
    It does not and cannot prove *semantic preservation*, because the BEFORE
    file does not parse, so there is no parsed baseline to compare against. Two
    concrete consequences, both established by probing this implementation:

    * A syntactically valid change of ``+`` to ``-`` satisfies every clause.
      Behaviour is the post-apply validation's job, not the oracle's, and the
      change is rolled back when that validation fails.
    * A change that keeps every name and every line count while replacing
      function bodies satisfies the first three clauses. That is what
      ``repair_is_local`` was added for, and it catches the realistic shapes of
      it - but a sufficiently patient adversarial rewrite that overwrites only
      one or two original lines at a time, across repeated runs, is not
      excluded by any clause here. What stands between that and a bad outcome
      is the post-apply validation and the mandatory approval boundary, neither
      of which this oracle replaces. This is a real residual limitation, not a
      hypothetical one, and it is recorded on the inventory entry too.

    **Inconclusive.** An unreadable or undecodable file yields
    :data:`OracleOutcome.INCONCLUSIVE` from both methods. It is neither "parses"
    nor "does not parse", and collapsing it into either would be a lie - into
    success, a dangerous one.
    """

    signal_kind = MaintenanceSignal.PARSE_FAILURE
    name = "parse_oracle"
    oracle_class = OracleClass.DETERMINISTIC
    max_scope_files = 1
    supported_suffixes = frozenset({".py"})

    def describe_failure_predicate(self) -> str:
        return (
            "the named .py file is readable and CPython's compiler rejects it "
            "(SyntaxError or ValueError)"
        )

    def describe_success_predicate(self) -> str:
        return (
            "the file parses AND every def/class/import/module-binding name "
            "lexically present in the broken source is still bound AND the "
            f"significant line count did not shrink by more than "
            f"{int(MAX_SHRINK_RATIO * 100)}% (slack {SHRINK_SLACK_LINES} lines) "
            f"AND the change overwrote at most "
            f"{int(MAX_REPLACED_LINE_RATIO * 100)}% of the original lines "
            f"(slack {REPLACED_LINE_SLACK} lines); insertions are unbounded"
        )

    # -- BEFORE ------------------------------------------------------------

    def observe_failure(self, root: Path, relative: str) -> OracleObservation:
        path = Path(root) / relative
        if not path.is_file():
            return self._inconclusive(
                f"'{relative}' does not exist, so the parse failure cannot be "
                "re-observed",
                path=str(relative),
            )
        source, read_error = _read_source(path)
        if source is None:
            return self._inconclusive(
                f"'{relative}' could not be read for the failure re-check: {read_error}",
                path=str(relative),
            )
        error = _parse_error(source, path)
        surface = sorted(_lexical_surface(source))
        substance = _significant_lines(source)
        digests = _line_digests(source)
        if not error:
            return self._observation(
                OracleOutcome.RESOLVED,
                f"'{relative}' parses cleanly; the parse-failure signal does not "
                "reproduce",
                clauses=(("parses", True),),
                path=str(relative),
                lexical_surface=surface,
                significant_lines=substance,
                line_digests=digests,
            )
        return self._observation(
            OracleOutcome.NOT_RESOLVED,
            f"'{relative}' does not parse: {error}",
            clauses=(("parses", False),),
            path=str(relative),
            parse_error=error,
            lexical_surface=surface,
            significant_lines=substance,
            line_digests=digests,
        )

    # -- AFTER -------------------------------------------------------------

    def observe_success(
        self,
        root: Path,
        relative: str,
        before: OracleObservation | None = None,
    ) -> OracleObservation:
        path = Path(root) / relative
        if not path.is_file():
            return self._inconclusive(
                f"'{relative}' does not exist after the change, so the repair "
                "cannot be verified",
                path=str(relative),
            )
        source, read_error = _read_source(path)
        if source is None:
            return self._inconclusive(
                f"'{relative}' could not be read after the change: {read_error}",
                path=str(relative),
            )
        error = _parse_error(source, path)
        if error:
            return self._observation(
                OracleOutcome.NOT_RESOLVED,
                f"the target file still does not parse after the change: {error}",
                clauses=(("parses", False),),
                path=str(relative),
                parse_error=error,
            )

        try:
            tree = ast.parse(source, filename=str(path))
        except (SyntaxError, ValueError) as exc:  # pragma: no cover - defensive
            # ``compile`` accepted it, so this should be unreachable. If the two
            # ever disagree, refusing is the only safe answer.
            return self._inconclusive(
                f"'{relative}' compiled but could not be parsed into an AST: {exc}",
                path=str(relative),
            )
        after_surface = _parsed_surface(tree)
        after_substance = _significant_lines(source)

        # Without a BEFORE observation there is nothing to compare against, and
        # inventing a baseline from the current file would make the clause
        # vacuously true. Refuse instead.
        if before is None or not isinstance(before.evidence, Mapping):
            return self._inconclusive(
                f"'{relative}' parses, but no pre-change observation is available "
                "to prove the repair preserved the module",
                path=str(relative),
            )
        raw_surface = before.evidence.get("lexical_surface")
        raw_substance = before.evidence.get("significant_lines")
        raw_digests = before.evidence.get("line_digests")
        if (
            not isinstance(raw_surface, (list, tuple, set, frozenset))
            or not isinstance(raw_substance, int)
            or isinstance(raw_substance, bool)
            or not isinstance(raw_digests, (list, tuple))
        ):
            return self._inconclusive(
                f"'{relative}' parses, but the pre-change observation is malformed "
                "and cannot support a preservation check",
                path=str(relative),
            )
        before_surface = {str(name) for name in raw_surface}
        before_substance = int(raw_substance)
        before_digests = [str(item) for item in raw_digests]

        missing = sorted(before_surface - after_surface)
        surface_ok = not missing

        floor = _substance_floor(before_substance)
        substance_ok = after_substance >= floor

        replaced = _replaced_line_count(before_digests, _line_digests(source))
        allowance = _replaced_line_allowance(len(before_digests))
        locality_ok = replaced <= allowance

        clauses = (
            ("parses", True),
            ("surface_preserved", surface_ok),
            ("substance_preserved", substance_ok),
            ("repair_is_local", locality_ok),
        )
        evidence = {
            "path": str(relative),
            "missing_names": missing[:20],
            "before_significant_lines": before_substance,
            "after_significant_lines": after_substance,
            "required_significant_lines": floor,
            "replaced_lines": replaced,
            "allowed_replaced_lines": allowance,
        }
        if surface_ok and substance_ok and locality_ok:
            return self._observation(
                OracleOutcome.RESOLVED,
                f"'{relative}' parses and the repair preserved the module's "
                f"{len(before_surface)} named binding(s) and "
                f"{after_substance}/{before_substance} significant line(s), "
                f"overwriting {replaced} original line(s)",
                clauses=clauses,
                **evidence,
            )
        problems: list[str] = []
        if not surface_ok:
            problems.append(
                "the repair removed name(s) that were present in the broken "
                f"source: {', '.join(missing[:8])}"
            )
        if not substance_ok:
            problems.append(
                f"the file shrank from {before_substance} to {after_substance} "
                f"significant line(s), below the {floor}-line floor"
            )
        if not locality_ok:
            problems.append(
                f"the change overwrote {replaced} original line(s), over the "
                f"{allowance}-line allowance for a local syntax repair"
            )
        return self._observation(
            OracleOutcome.NOT_RESOLVED,
            f"'{relative}' parses but the change is a rewrite rather than a "
            "repair: " + "; ".join(problems),
            clauses=clauses,
            **evidence,
        )

    # -- driving the repair ------------------------------------------------

    def acceptance_commands(self, root: Path, changed: Sequence[str]) -> list[Any]:
        """One ``compileall`` over the changed Python files that exist.

        This is the signal's own acceptance test expressed as a real
        subprocess, so the verdict does not rest solely on the in-process
        ``compile`` above. It is additive to whatever the validation authority
        chose.
        """
        from .models import CommandSpec

        python_files = sorted(
            str(path)
            for path in changed
            if str(path).lower().endswith(".py") and (Path(root) / str(path)).is_file()
        )
        if not python_files:
            return []
        return [
            CommandSpec(
                name="maintenance_compileall",
                command=("python", "-m", "compileall", "-q", *python_files),
                reason="mandatory syntax acceptance check for a parse-failure repair",
                category="type_check",
                risk="low",
            )
        ]

    def plan_fragment(self, relative: str) -> Mapping[str, Any]:
        return {
            "objective": f"Repair the parse failure in {relative}",
            "agent_objective": (
                f"Repair the Python syntax error in {relative} so the file parses. "
                "Make the smallest possible edit. Preserve every function, class, "
                "import and module-level name that is already in the file. Do not "
                "change behaviour, public names, imports or any other file. A change "
                "that deletes code instead of repairing the syntax will be rejected."
            ),
            "steps": [
                f"Read {relative} and locate the syntax error reported by the parser.",
                "Correct the syntax with the smallest possible edit.",
                "Do not change the module's behaviour, public names or imports.",
                "Do not delete any existing definition, class, import or constant.",
            ],
            "validation_strategy": [
                f"{relative} must compile as valid Python.",
                "Every name present before the repair must still be present.",
                "Existing validation must continue to pass.",
            ],
            "risks": [
                "A syntax repair that changes behaviour would be worse than the defect.",
                "Deleting the file's contents would satisfy a naive 'it parses' check "
                "while destroying the module.",
            ],
        }


def _substance_floor(before_significant: int) -> int:
    """Fewest significant lines a legitimate repair may leave behind."""
    if before_significant <= 0:
        return 0
    allowance = max(SHRINK_SLACK_LINES, math.ceil(before_significant * MAX_SHRINK_RATIO))
    return max(0, before_significant - allowance)


# =============================================================================
# The authoritative signal inventory
# =============================================================================


@dataclass(frozen=True)
class SignalInventoryEntry:
    """Everything Phase 4.23 must record about one maintenance signal.

    This is a *description of the implementation*, not an aspiration. The
    test-suite walks :data:`ALL_SIGNAL_KINDS` and asserts that every signal has
    an entry and every entry names a real signal, that the recorded producer
    method exists on the real analyzer, and that the recorded policy ceiling
    matches what the real policy actually grants.
    """

    signal: str
    producer: str
    detection_mechanism: str
    evidence_source: str
    confidence_source: str
    affected_files_rule: str
    remediation_type: str

    oracle_class: str
    #: Does a deterministic success oracle exist for this signal?
    deterministic_oracle: bool
    #: Can validation prove the signal is resolved, as opposed to merely not
    #: contradicting resolution?
    validation_can_prove_resolution: bool
    #: Could remediating a false positive cause damage?
    false_positive_damage: bool
    #: Can remediation expand beyond the named files?
    remediation_can_expand_scope: bool
    #: Might remediation need dependency or environment changes?
    needs_environment_change: bool
    #: Does remediation require semantic judgement?
    needs_semantic_judgement: bool
    #: Can remediation be safely bounded to a small, checkable diff?
    remediation_boundable: bool

    #: The strongest tier the *real* policy can grant this kind.
    policy_max_tier: str
    #: Whether this build executes the signal autonomously.
    autonomous_execution: bool
    #: Current automated test coverage, described honestly.
    test_coverage: str
    #: Why the signal is not autonomous. Empty exactly when it is.
    rejection_reasons: tuple[str, ...] = ()
    #: What the oracle still cannot prove, for a signal that *is* autonomous.
    #: Empty for non-autonomous signals, where the rejection reasons carry the
    #: argument instead. Recorded so that granting autonomy is never mistaken
    #: for granting a total guarantee.
    residual_limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal": self.signal,
            "producer": self.producer,
            "detection_mechanism": self.detection_mechanism,
            "evidence_source": self.evidence_source,
            "confidence_source": self.confidence_source,
            "affected_files_rule": self.affected_files_rule,
            "remediation_type": self.remediation_type,
            "oracle_class": self.oracle_class,
            "deterministic_oracle": self.deterministic_oracle,
            "validation_can_prove_resolution": self.validation_can_prove_resolution,
            "false_positive_damage": self.false_positive_damage,
            "remediation_can_expand_scope": self.remediation_can_expand_scope,
            "needs_environment_change": self.needs_environment_change,
            "needs_semantic_judgement": self.needs_semantic_judgement,
            "remediation_boundable": self.remediation_boundable,
            "policy_max_tier": self.policy_max_tier,
            "autonomous_execution": self.autonomous_execution,
            "test_coverage": self.test_coverage,
            "rejection_reasons": list(self.rejection_reasons),
            "residual_limitations": list(self.residual_limitations),
        }


# Tier strings are duplicated as literals rather than imported from
# ``maintenance_policy`` on purpose: importing it here would create a cycle
# (the policy imports ``maintenance``, this module is imported by the
# executor), and the test-suite asserts these strings against the real
# ``AutonomyTier`` members, so a drift fails loudly rather than silently.
_TIER_RECOMMEND = "recommend"
_TIER_EXECUTE_WITH_APPROVAL = "execute_with_existing_approval"


SIGNAL_INVENTORY: Mapping[str, SignalInventoryEntry] = {
    entry.signal: entry
    for entry in (
        SignalInventoryEntry(
            signal=MaintenanceSignal.PARSE_FAILURE,
            producer="MaintenanceAnalyzer._parse_failures",
            detection_mechanism=(
                "SemanticGraph.parse_failures - CPython's own parser rejected the "
                "file during indexing"
            ),
            evidence_source="semantic_graph",
            confidence_source="structural observation, fixed 1.0, sample_size 1",
            affected_files_rule="exactly one file, the one that failed to parse",
            remediation_type="in-place syntax repair of a single .py file",
            oracle_class=OracleClass.DETERMINISTIC,
            deterministic_oracle=True,
            validation_can_prove_resolution=True,
            false_positive_damage=False,
            remediation_can_expand_scope=False,
            needs_environment_change=False,
            needs_semantic_judgement=False,
            remediation_boundable=True,
            policy_max_tier=_TIER_EXECUTE_WITH_APPROVAL,
            autonomous_execution=True,
            test_coverage=(
                "end-to-end through discovery, policy, executor, workspace, oracle, "
                "approval, apply, post-apply validation and rescan"
            ),
            residual_limitations=(
                "the oracle bounds destruction, not semantics: a syntactically "
                "valid change of behaviour (say '+' to '-') satisfies every "
                "clause, and only the post-apply validation can catch it",
                "the BEFORE file does not parse, so the preservation clauses rest "
                "on a lexical scan rather than a parsed baseline; the scan is "
                "over-inclusive, which biases it toward refusing good repairs "
                "rather than accepting bad ones",
                "an adversarial rewrite that overwrites only one or two original "
                "lines per run is inside the locality allowance; the approval "
                "boundary and post-apply validation, not the oracle, are what "
                "contain that case",
            ),
        ),
        SignalInventoryEntry(
            signal=MaintenanceSignal.TEST_GAP,
            producer="MaintenanceAnalyzer._test_gaps",
            detection_mechanism=(
                "SemanticGraph.reverse_deps - a module with >= 3 dependents that no "
                "test file imports directly"
            ),
            evidence_source="semantic_graph",
            confidence_source="structural observation, fixed 1.0, sample_size 1",
            affected_files_rule="exactly one file, the untested module",
            remediation_type="create a new test file",
            oracle_class=OracleClass.AMBIGUOUS,
            deterministic_oracle=False,
            validation_can_prove_resolution=False,
            false_positive_damage=True,
            remediation_can_expand_scope=True,
            needs_environment_change=True,
            needs_semantic_judgement=True,
            remediation_boundable=False,
            policy_max_tier=_TIER_EXECUTE_WITH_APPROVAL,
            autonomous_execution=False,
            test_coverage=(
                "discovery and policy capping; no executor path exists, and an "
                "adversarial test proves the executor refuses this kind even when "
                "its evidence is forged perfect"
            ),
            rejection_reasons=(
                "the success predicate is 'a useful test now exists', which no "
                "machine here can check: a test asserting True passes",
                "a vacuous test is worse than the gap, because the next scan sees a "
                "test importing the module and stops reporting the signal - the "
                "remediation silently suppresses its own detector",
                "remediation creates a file rather than repairing one, so there is "
                "no before-state to compare a post-condition against",
                "the signal always carries an explicit uncertainty caveat "
                "('direct-import evidence only'), which is an admission that the "
                "failure predicate itself is approximate",
            ),
        ),
        SignalInventoryEntry(
            signal=MaintenanceSignal.ANALYZER_BLIND_SPOT,
            producer="MaintenanceAnalyzer._analyzer_blind_spots",
            detection_mechanism=(
                "SemanticGraph.unresolved_imports - >= 3 local-looking imports the "
                "dependency resolver could not resolve"
            ),
            evidence_source="semantic_graph",
            confidence_source="structural observation, fixed 1.0, sample_size 1",
            affected_files_rule="exactly one file, the one with unresolved imports",
            remediation_type="rewrite imports across modules, or extend the resolver",
            oracle_class=OracleClass.AMBIGUOUS,
            deterministic_oracle=False,
            validation_can_prove_resolution=False,
            false_positive_damage=True,
            remediation_can_expand_scope=True,
            needs_environment_change=True,
            needs_semantic_judgement=True,
            remediation_boundable=False,
            policy_max_tier=_TIER_RECOMMEND,
            autonomous_execution=False,
            test_coverage=(
                "discovery and policy capping; no executor path exists, and an "
                "adversarial test proves the executor refuses this kind even when "
                "its evidence is forged perfect"
            ),
            rejection_reasons=(
                "the obvious mechanical post-condition - 'the unresolved-import count "
                "dropped' - is satisfied perfectly by DELETING the imports, which is "
                "catastrophic and indistinguishable from a real fix to the oracle",
                "a genuine fix is usually in the resolver, i.e. in the agent's own "
                "code, not in the file the signal names",
                "the third-party-versus-local heuristic (_looks_local) means the "
                "failure predicate is itself approximate",
            ),
        ),
        SignalInventoryEntry(
            signal=MaintenanceSignal.ARCHITECTURAL_RISK,
            producer="MaintenanceAnalyzer._architectural_risk",
            detection_mechanism=(
                "SemanticGraph.reverse_deps fan-in >= 8 AND git churn >= 4 commits"
            ),
            evidence_source="semantic_graph + git churn",
            confidence_source="structural observation, fixed 1.0, sample_size 1",
            affected_files_rule="exactly one file, the high-fan-in module",
            remediation_type="architectural redesign",
            oracle_class=OracleClass.AMBIGUOUS,
            deterministic_oracle=False,
            validation_can_prove_resolution=False,
            false_positive_damage=True,
            remediation_can_expand_scope=True,
            needs_environment_change=False,
            needs_semantic_judgement=True,
            remediation_boundable=False,
            policy_max_tier=_TIER_RECOMMEND,
            autonomous_execution=False,
            test_coverage=(
                "discovery and policy capping; no executor path exists, and an "
                "adversarial test proves the executor refuses this kind even when "
                "its evidence is forged perfect"
            ),
            rejection_reasons=(
                "'risky' is not a defect; there is no failure to reproduce and "
                "therefore no post-condition to satisfy",
                "the signal carries its own uncertainty caveat saying the risk is "
                "inferred from coupling and churn, not from an observed defect",
                "any real remediation is a multi-file design change",
            ),
        ),
        SignalInventoryEntry(
            signal=MaintenanceSignal.RECURRING_DEFECT,
            producer="MaintenanceAnalyzer._recurring_defects",
            detection_mechanism=(
                "the same defect fingerprint appears in >= 2 distinct validation "
                "lifecycles"
            ),
            evidence_source="validation_lifecycle",
            confidence_source="Wilson lower bound over lifecycle count",
            affected_files_rule="union of the files named by every matching signature",
            remediation_type="fix an underlying defect",
            oracle_class=OracleClass.STATISTICAL,
            deterministic_oracle=False,
            validation_can_prove_resolution=False,
            false_positive_damage=True,
            remediation_can_expand_scope=True,
            needs_environment_change=True,
            needs_semantic_judgement=True,
            remediation_boundable=False,
            policy_max_tier=_TIER_EXECUTE_WITH_APPROVAL,
            autonomous_execution=False,
            test_coverage=(
                "discovery, confidence and policy capping; no executor path exists, "
                "and an adversarial test proves the executor refuses this kind even "
                "when its evidence is forged perfect"
            ),
            rejection_reasons=(
                "the failure predicate is a historical rate, so the defect may not be "
                "present in the working tree at all - there is nothing to re-observe "
                "at execution time",
                "the success predicate is 'the underlying defect is gone', which "
                "cannot be checked without reproducing the defect first",
                "the affected-file set is a union over history and is unbounded",
                "absence of the signal on the next scan is also produced by simply "
                "not running the failing work again, so a rescan cannot distinguish "
                "a repair from inactivity",
            ),
        ),
        SignalInventoryEntry(
            signal=MaintenanceSignal.REPEATED_REPAIR,
            producer="MaintenanceAnalyzer._repeated_repairs",
            detection_mechanism="lifecycles whose repair_count >= 2, grouped by file",
            evidence_source="validation_lifecycle",
            confidence_source="Wilson lower bound over lifecycle count",
            affected_files_rule="one file per candidate, from the defect signature",
            remediation_type="fix whatever keeps needing repair",
            oracle_class=OracleClass.STATISTICAL,
            deterministic_oracle=False,
            validation_can_prove_resolution=False,
            false_positive_damage=True,
            remediation_can_expand_scope=True,
            needs_environment_change=True,
            needs_semantic_judgement=True,
            remediation_boundable=False,
            policy_max_tier=_TIER_EXECUTE_WITH_APPROVAL,
            autonomous_execution=False,
            test_coverage=(
                "discovery, confidence and policy capping; no executor path exists, "
                "and an adversarial test proves the executor refuses this kind even "
                "when its evidence is forged perfect"
            ),
            rejection_reasons=(
                "identifies a file that has been hard to change, not a defect in it; "
                "there is no failure predicate to reproduce",
                "the success predicate would be 'future changes here need fewer "
                "repairs', which is unobservable at execution time",
            ),
        ),
        SignalInventoryEntry(
            signal=MaintenanceSignal.KNOWN_FAILURE_PATTERN,
            producer="MaintenanceAnalyzer._failure_patterns",
            detection_mechanism=(
                "RepositoryKnowledgeGraph.failure_patterns with >= 3 occurrences"
            ),
            evidence_source="knowledge_graph",
            confidence_source="stored confidence clamped by a Wilson bound",
            affected_files_rule="whatever the stored pattern recorded, unbounded",
            remediation_type="apply a remembered repair",
            oracle_class=OracleClass.STATISTICAL,
            deterministic_oracle=False,
            validation_can_prove_resolution=False,
            false_positive_damage=True,
            remediation_can_expand_scope=True,
            needs_environment_change=True,
            needs_semantic_judgement=True,
            remediation_boundable=False,
            policy_max_tier=_TIER_EXECUTE_WITH_APPROVAL,
            autonomous_execution=False,
            test_coverage=(
                "discovery, confidence clamping and policy capping; no executor "
                "path exists, and an adversarial test proves the executor refuses "
                "this kind even when its evidence is forged perfect"
            ),
            rejection_reasons=(
                "the subject is an error-signature string, not a path, so there is no "
                "file whose state can be observed before or after",
                "the affected-file list comes from PERSISTED state and is therefore "
                "attacker-controlled input; granting it execution authority would let "
                "a forged record choose the scope of an autonomous change",
                "the success predicate is 'this failure will not recur', which is a "
                "prediction, not an observation",
            ),
        ),
        SignalInventoryEntry(
            signal=MaintenanceSignal.ABANDONED_WORK,
            producer="MaintenanceAnalyzer._abandoned_work",
            detection_mechanism=(
                ">= 30% of terminal lifecycles ended abandoned or failed, over >= 5 "
                "lifecycles"
            ),
            evidence_source="validation_lifecycle",
            confidence_source="Wilson lower bound over terminal lifecycles",
            affected_files_rule="none; subject is 'repository'",
            remediation_type="process change, not a code change",
            oracle_class=OracleClass.UNSAFE,
            deterministic_oracle=False,
            validation_can_prove_resolution=False,
            false_positive_damage=True,
            remediation_can_expand_scope=True,
            needs_environment_change=False,
            needs_semantic_judgement=True,
            remediation_boundable=False,
            policy_max_tier=_TIER_RECOMMEND,
            autonomous_execution=False,
            test_coverage=(
                "discovery and policy capping; no executor path exists, and an "
                "adversarial test proves the executor refuses this kind even when "
                "its evidence is forged perfect"
            ),
            rejection_reasons=(
                "names no file, so there is literally nothing to change",
                "describes the agent's own success rate; the only 'fix' an automated "
                "system could apply to it is to stop recording failures",
            ),
        ),
        SignalInventoryEntry(
            signal=MaintenanceSignal.CANDIDATE_INSTABILITY,
            producer="MaintenanceAnalyzer._candidate_instability",
            detection_mechanism=(
                ">= 3 failed candidate-stage validation iterations for one file"
            ),
            evidence_source="validation_lifecycle",
            confidence_source="Wilson lower bound with a self-referential denominator",
            affected_files_rule="one file, from the defect signature",
            remediation_type="unclear; the file is merely hard to change",
            oracle_class=OracleClass.AMBIGUOUS,
            deterministic_oracle=False,
            validation_can_prove_resolution=False,
            false_positive_damage=True,
            remediation_can_expand_scope=True,
            needs_environment_change=True,
            needs_semantic_judgement=True,
            remediation_boundable=False,
            policy_max_tier=_TIER_RECOMMEND,
            autonomous_execution=False,
            test_coverage=(
                "discovery and policy capping; no executor path exists, and an "
                "adversarial test proves the executor refuses this kind even when "
                "its evidence is forged perfect"
            ),
            rejection_reasons=(
                "records that the agent kept failing here, which is a fact about the "
                "agent rather than a defect in the file",
                "its confidence uses wilson_lower_bound(count, count + 1), a "
                "self-referential denominator that is not a real trial count - the "
                "number is not calibrated and must not gate authority",
                "acting on it means the component that already failed three times "
                "tries a fourth time unattended",
            ),
        ),
        SignalInventoryEntry(
            signal=MaintenanceSignal.BROAD_VALIDATION_PRESSURE,
            producer="MaintenanceAnalyzer._validation_pressure",
            detection_mechanism=">= 60% of >= 10 validation decisions chose broad scope",
            evidence_source="validation_telemetry",
            confidence_source="Wilson lower bound over decisions",
            affected_files_rule="none; subject is 'validation_scope'",
            remediation_type="tune the agent's own impact analysis",
            oracle_class=OracleClass.UNSAFE,
            deterministic_oracle=False,
            validation_can_prove_resolution=False,
            false_positive_damage=True,
            remediation_can_expand_scope=True,
            needs_environment_change=False,
            needs_semantic_judgement=True,
            remediation_boundable=False,
            policy_max_tier=_TIER_RECOMMEND,
            autonomous_execution=False,
            test_coverage=(
                "discovery and policy capping; no executor path exists, and an "
                "adversarial test proves the executor refuses this kind even when "
                "its evidence is forged perfect"
            ),
            rejection_reasons=(
                "names no file",
                "the remediation is to make the agent validate LESS, so a successful "
                "autonomous fix would weaken the very safety machinery that is "
                "supposed to contain it - the incentive points the wrong way",
            ),
        ),
        SignalInventoryEntry(
            signal=MaintenanceSignal.EVIDENCE_REUSE_FAILURE,
            producer="MaintenanceAnalyzer._evidence_reuse",
            detection_mechanism=(
                "one reuse-denial reason accounts for >= 70% of denials, over >= 10 "
                "decisions"
            ),
            evidence_source="validation_telemetry",
            confidence_source="Wilson lower bound over denial totals",
            affected_files_rule="none; subject is 'reuse:<reason>'",
            remediation_type="tune the agent's own evidence-reuse rules",
            oracle_class=OracleClass.UNSAFE,
            deterministic_oracle=False,
            validation_can_prove_resolution=False,
            false_positive_damage=True,
            remediation_can_expand_scope=True,
            needs_environment_change=False,
            needs_semantic_judgement=True,
            remediation_boundable=False,
            policy_max_tier=_TIER_RECOMMEND,
            autonomous_execution=False,
            test_coverage=(
                "discovery and policy capping; no executor path exists, and an "
                "adversarial test proves the executor refuses this kind even when "
                "its evidence is forged perfect"
            ),
            rejection_reasons=(
                "names no file",
                "the remediation is to accept MORE stale evidence, which directly "
                "weakens a safety control",
            ),
        ),
        SignalInventoryEntry(
            signal=MaintenanceSignal.FALSE_CONFIDENCE,
            producer="MaintenanceAnalyzer._false_confidence",
            detection_mechanism=(
                ">= 1 telemetry decision whose recorded quality is 'false_confidence'"
            ),
            evidence_source="validation_telemetry",
            confidence_source="fixed 1.0; sample_size is the incident count",
            affected_files_rule="union of changed_files across the incidents",
            remediation_type="human review of a safety-analysis failure",
            oracle_class=OracleClass.UNSAFE,
            deterministic_oracle=False,
            validation_can_prove_resolution=False,
            false_positive_damage=True,
            remediation_can_expand_scope=True,
            needs_environment_change=True,
            needs_semantic_judgement=True,
            remediation_boundable=False,
            policy_max_tier=_TIER_RECOMMEND,
            autonomous_execution=False,
            test_coverage=(
                "discovery, CRITICAL severity and policy capping; no executor path "
                "exists, and an adversarial test proves the executor refuses this "
                "kind even when its evidence is forged perfect"
            ),
            rejection_reasons=(
                "this signal means the agent's own safety analysis was wrong; the "
                "correct response is a human reading it, not the same analysis "
                "trying again",
                "it is reported at confidence 1.0 and CRITICAL severity, so it would "
                "pass every numeric gate - it is held back by kind alone, which is "
                "exactly why the kind gate must never be made configurable",
            ),
        ),
        SignalInventoryEntry(
            signal=MaintenanceSignal.ANALYSIS_DEGRADATION,
            producer="MaintenanceAnalyzer._analysis_degradation",
            detection_mechanism=">= 30% of >= 10 decisions ran with degraded analysis",
            evidence_source="validation_telemetry",
            confidence_source="Wilson lower bound over decisions",
            affected_files_rule="none; subject is 'impact_analysis'",
            remediation_type="fix the agent's own analysis pipeline",
            oracle_class=OracleClass.UNSAFE,
            deterministic_oracle=False,
            validation_can_prove_resolution=False,
            false_positive_damage=True,
            remediation_can_expand_scope=True,
            needs_environment_change=True,
            needs_semantic_judgement=True,
            remediation_boundable=False,
            policy_max_tier=_TIER_RECOMMEND,
            autonomous_execution=False,
            test_coverage=(
                "discovery and policy capping; no executor path exists, and an "
                "adversarial test proves the executor refuses this kind even when "
                "its evidence is forged perfect"
            ),
            rejection_reasons=(
                "names no file",
                "it says the agent's analysis is unreliable right now, which is the "
                "worst possible moment to let that same analysis authorise an "
                "unattended change",
            ),
        ),
    )
}


#: Signal kind -> oracle instance. Constructed once; the oracles are stateless.
_ORACLES: Mapping[str, ExecutionOracle] = {
    MaintenanceSignal.PARSE_FAILURE: ParseOracle(),
    **{
        entry.signal: UnverifiableOracle(entry.signal, entry.oracle_class)
        for entry in SIGNAL_INVENTORY.values()
        if entry.signal != MaintenanceSignal.PARSE_FAILURE
    },
}


def oracle_for(signal_kind: Any) -> ExecutionOracle:
    """The oracle for ``signal_kind``. Never ``None``.

    An unknown kind gets an :class:`UnverifiableOracle` marked
    :data:`OracleClass.UNSAFE`, so a corrupted or forged signal name cannot
    reach a promotable oracle by being unrecognised.
    """
    key = str(signal_kind)
    found = _ORACLES.get(key)
    if found is not None:
        return found
    return UnverifiableOracle(key, OracleClass.UNSAFE)


def inventory_for(signal_kind: Any) -> SignalInventoryEntry | None:
    return SIGNAL_INVENTORY.get(str(signal_kind))


def _autonomous_kinds() -> frozenset[str]:
    """Signals this build executes autonomously.

    Both conditions must hold, and both are recomputed from live objects rather
    than written down: the inventory must say so, *and* the bound oracle must be
    promotable. A future editor who flips ``autonomous_execution`` on an entry
    whose oracle is not deterministic changes nothing.
    """
    return frozenset(
        entry.signal
        for entry in SIGNAL_INVENTORY.values()
        if entry.autonomous_execution and oracle_for(entry.signal).promotable
    )


AUTONOMOUS_SIGNAL_KINDS: frozenset[str] = _autonomous_kinds()


def missing_inventory_entries() -> tuple[str, ...]:
    """Signals in :data:`ALL_SIGNAL_KINDS` with no inventory entry."""
    return tuple(kind for kind in ALL_SIGNAL_KINDS if kind not in SIGNAL_INVENTORY)


def unknown_inventory_entries() -> tuple[str, ...]:
    """Inventory entries naming a signal the codebase does not define."""
    return tuple(
        signal for signal in sorted(SIGNAL_INVENTORY) if signal not in ALL_SIGNAL_KINDS
    )


def inventory_rows() -> list[dict[str, Any]]:
    """The whole inventory, ordered as :data:`ALL_SIGNAL_KINDS` is."""
    rows: list[dict[str, Any]] = []
    for kind in ALL_SIGNAL_KINDS:
        entry = SIGNAL_INVENTORY.get(kind)
        if entry is not None:
            rows.append(entry.to_dict())
    return rows
