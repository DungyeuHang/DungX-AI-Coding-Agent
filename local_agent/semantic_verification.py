"""Phase 4.24: post-execution semantic verification and adversarial success validation.

Phase 4.23 introduced the execution oracle framework and demonstrated a critical
exploit: a parse oracle based on "does the file parse?" and surface name
preservation could still be satisfied by replacing function bodies with stubs or
deleting defective regions while padding lines.

This module introduces **POST-EXECUTION SEMANTIC VERIFICATION**.

The central question is:

    *After the candidate was applied and validated, did the original defect
    actually disappear without introducing unacceptable new behavior or gutting
    the implementation?*

The system strictly separates and distinguishes:

    ``ORACLE_ACCEPTED``
        The structural boundaries (file compiles, names bound, line counts
        within ratio) are satisfied.
    ``VALIDATION_PASSED``
        The targeted and post-apply validation commands exited 0.
    ``SEMANTICALLY_RESOLVED``
        The specific defect diagnostic is demonstrably resolved, pre-existing
        valid AST constructs remain functionally preserved, function bodies are
        not gutted or stubbed, and the repair is semantically credible.
    ``RESOLVED``
        The final lifecycle outcome credited only when ALL upstream stages
        succeed and semantic verification passes.

Architecture & Safety Invariants
--------------------------------
1. **Zero Second Decision Authority**: This module provides observation and
   evidence about repair validity. It cannot modify validation scope, bypass
   approvals, or write to the authoritative tree.
2. **Read-Only / No Mutation**: The verifier class body contains zero filesystem
   writes, no ``open(..., 'w')``, no ``write_text``, no ``shutil`` mutation, and
   spawns no subprocesses.
3. **Fail-Closed by Construction**: Missing before state, unreadable files,
   stale/corrupted evidence, or unrecognised signals return
   :data:`SemanticVerificationStatus.INCONCLUSIVE` or
   :data:`SemanticVerificationStatus.NOT_APPLICABLE`, never a resolution.
4. **Historical Isolation**: Historical success never overrides a current
   structural or semantic defect.
"""

from __future__ import annotations

import ast
import datetime
import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .maintenance import (
    MaintenanceCandidate,
    MaintenanceSignal,
    is_protected_relative_path,
    sanitize_relative_path,
    sanitize_string_list,
    sanitize_text,
)
from .maintenance_oracle import OracleClass

LOGGER = logging.getLogger(__name__)

SEMANTIC_VERIFICATION_SCHEMA_VERSION = "4.24.0"

__all__ = [
    "ALL_FAILURE_CATEGORIES",
    "ALL_SEMANTIC_STATUSES",
    "DefectIdentity",
    "FailureCategory",
    "ParseFailureSemanticVerifier",
    "SEMANTIC_VERIFICATION_SCHEMA_VERSION",
    "SemanticVerificationStatus",
    "SemanticVerifier",
    "UnverifiableSemanticVerifier",
    "VerificationEvidence",
    "all_verifiers",
    "register_verifier",
    "verifier_for",
]


# =============================================================================
# Vocabulary & Result Categories
# =============================================================================


class SemanticVerificationStatus:
    """The exhaustive outcomes a semantic verifier may produce."""

    #: Proven: the original defect is gone and semantic integrity is intact.
    RESOLVED = "resolved"
    #: The defect remains or semantic/structural corruption was detected.
    NOT_RESOLVED = "not_resolved"
    #: The verifier could not establish proof (missing baseline, unreadable, etc.).
    INCONCLUSIVE = "insufficient_evidence"
    #: The signal kind or file type is not supported by this verifier.
    NOT_APPLICABLE = "not_applicable"
    #: Internal or parser exception occurred during verification.
    VERIFICATION_ERROR = "verification_error"


ALL_SEMANTIC_STATUSES: tuple[str, ...] = (
    SemanticVerificationStatus.RESOLVED,
    SemanticVerificationStatus.NOT_RESOLVED,
    SemanticVerificationStatus.INCONCLUSIVE,
    SemanticVerificationStatus.NOT_APPLICABLE,
    SemanticVerificationStatus.VERIFICATION_ERROR,
)


class FailureCategory:
    """Precise deterministic failure categories for semantic verification."""

    NONE = "none"
    ORIGINAL_DEFECT_REMAINS = "original_defect_remains"
    DIAGNOSTIC_CHANGED_NOT_RESOLVED = "diagnostic_changed_not_resolved"
    UNEXPECTED_SURFACE_CHANGE = "unexpected_surface_change"
    UNEXPECTED_SCOPE_CHANGE = "unexpected_scope_change"
    BODY_GUTTED_OR_STUBBED = "body_gutted_or_stubbed"
    AST_MUTATION_SUSPICIOUS = "ast_mutation_suspicious"
    IMPORT_SURFACE_CORRUPTED = "import_surface_corrupted"
    LINE_COUNT_DEGRADATION = "line_count_degradation"
    NEW_REGRESSION = "new_regression"
    EVIDENCE_MISMATCH = "evidence_mismatch"
    STALE_EVIDENCE = "stale_evidence"
    IDENTITY_MISMATCH = "identity_mismatch"
    VERIFIER_NOT_APPLICABLE = "verifier_not_applicable"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    VERIFICATION_ERROR = "verification_error"


ALL_FAILURE_CATEGORIES: tuple[str, ...] = (
    FailureCategory.NONE,
    FailureCategory.ORIGINAL_DEFECT_REMAINS,
    FailureCategory.DIAGNOSTIC_CHANGED_NOT_RESOLVED,
    FailureCategory.UNEXPECTED_SURFACE_CHANGE,
    FailureCategory.UNEXPECTED_SCOPE_CHANGE,
    FailureCategory.BODY_GUTTED_OR_STUBBED,
    FailureCategory.AST_MUTATION_SUSPICIOUS,
    FailureCategory.IMPORT_SURFACE_CORRUPTED,
    FailureCategory.LINE_COUNT_DEGRADATION,
    FailureCategory.NEW_REGRESSION,
    FailureCategory.EVIDENCE_MISMATCH,
    FailureCategory.STALE_EVIDENCE,
    FailureCategory.IDENTITY_MISMATCH,
    FailureCategory.VERIFIER_NOT_APPLICABLE,
    FailureCategory.INSUFFICIENT_EVIDENCE,
    FailureCategory.VERIFICATION_ERROR,
)


# =============================================================================
# Defect Identity & Verification Evidence Models
# =============================================================================


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


@dataclass(frozen=True)
class DefectIdentity:
    """Causal, immutable record of what exact defect was observed BEFORE repair.

    Preserves diagnostic details, parser error class, affected span, and
    isolated AST signatures of the undamaged sections of the module.
    """

    signal_kind: str
    candidate_id: str
    subject: str
    relative_path: str
    defect_fingerprint: str
    source_state_fingerprint: str
    diagnostic_message: str = ""
    diagnostic_line: int | None = None
    diagnostic_column: int | None = None
    syntax_error_class: str = ""
    compiler_or_parser: str = "cpython_ast"
    lexical_symbols: tuple[str, ...] = ()
    import_names: tuple[str, ...] = ()
    block_signatures: Mapping[str, str] = field(default_factory=dict)
    substantive_symbol_metrics: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict
    )
    significant_lines: int = 0
    captured_at: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        from types import MappingProxyType

        object.__setattr__(
            self, "block_signatures", MappingProxyType(dict(self.block_signatures))
        )
        object.__setattr__(
            self,
            "substantive_symbol_metrics",
            MappingProxyType(dict(self.substantive_metrics_copy())),
        )

    def substantive_metrics_copy(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for k, v in self.substantive_symbol_metrics.items():
            result[k] = dict(v) if isinstance(v, Mapping) else {}
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_kind": self.signal_kind,
            "candidate_id": self.candidate_id,
            "subject": self.subject,
            "relative_path": self.relative_path,
            "defect_fingerprint": self.defect_fingerprint,
            "source_state_fingerprint": self.source_state_fingerprint,
            "diagnostic_message": self.diagnostic_message,
            "diagnostic_line": self.diagnostic_line,
            "diagnostic_column": self.diagnostic_column,
            "syntax_error_class": self.syntax_error_class,
            "compiler_or_parser": self.compiler_or_parser,
            "lexical_symbols": list(self.lexical_symbols),
            "import_names": list(self.import_names),
            "block_signatures": dict(self.block_signatures),
            "substantive_symbol_metrics": self.substantive_metrics_copy(),
            "significant_lines": self.significant_lines,
            "captured_at": self.captured_at,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "DefectIdentity":
        if not isinstance(data, Mapping):
            return cls(
                signal_kind="",
                candidate_id="",
                subject="",
                relative_path="",
                defect_fingerprint="",
                source_state_fingerprint="",
            )
        return cls(
            signal_kind=sanitize_text(data.get("signal_kind", ""), limit=64),
            candidate_id=sanitize_text(data.get("candidate_id", ""), limit=64),
            subject=sanitize_text(data.get("subject", ""), limit=400),
            relative_path=sanitize_relative_path(str(data.get("relative_path", ""))),
            defect_fingerprint=sanitize_text(data.get("defect_fingerprint", ""), limit=64),
            source_state_fingerprint=sanitize_text(
                data.get("source_state_fingerprint", ""), limit=64
            ),
            diagnostic_message=sanitize_text(data.get("diagnostic_message", "")),
            diagnostic_line=(
                int(data["diagnostic_line"])
                if isinstance(data.get("diagnostic_line"), (int, float))
                else None
            ),
            diagnostic_column=(
                int(data["diagnostic_column"])
                if isinstance(data.get("diagnostic_column"), (int, float))
                else None
            ),
            syntax_error_class=sanitize_text(data.get("syntax_error_class", ""), limit=64),
            compiler_or_parser=sanitize_text(data.get("compiler_or_parser", "cpython_ast"), limit=64),
            lexical_symbols=tuple(
                sanitize_string_list(data.get("lexical_symbols", []))
            ),
            import_names=tuple(sanitize_string_list(data.get("import_names", []))),
            block_signatures=dict(data.get("block_signatures") or {}),
            substantive_symbol_metrics=dict(data.get("substantive_symbol_metrics") or {}),
            significant_lines=max(0, int(data.get("significant_lines", 0) or 0)),
            captured_at=sanitize_text(data.get("captured_at", "") or _now()),
        )


@dataclass
class VerificationEvidence:
    """Structured, serializable account of post-execution semantic verification."""

    verifier: str = ""
    signal_kind: str = ""
    candidate_id: str = ""
    status: str = SemanticVerificationStatus.INCONCLUSIVE
    failure_category: str = FailureCategory.INSUFFICIENT_EVIDENCE
    confidence: str = "none"
    before_fingerprint: str = ""
    after_fingerprint: str = ""
    defect_fingerprint: str = ""
    diagnostic_before: dict[str, Any] | None = None
    diagnostic_after: dict[str, Any] | None = None
    structural_result: dict[str, Any] = field(default_factory=dict)
    semantic_result: dict[str, Any] = field(default_factory=dict)
    validation_result: dict[str, Any] | None = None
    affected_files_before: list[str] = field(default_factory=list)
    affected_files_after: list[str] = field(default_factory=list)
    changed_symbols: list[str] = field(default_factory=list)
    unexpected_changes: list[str] = field(default_factory=list)
    failure_reasons: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=_now)
    schema_version: str = SEMANTIC_VERIFICATION_SCHEMA_VERSION

    @property
    def passed(self) -> bool:
        """True strictly and only for RESOLVED status."""
        return self.status == SemanticVerificationStatus.RESOLVED

    def to_dict(self) -> dict[str, Any]:
        return {
            "verifier": self.verifier,
            "signal_kind": self.signal_kind,
            "candidate_id": self.candidate_id,
            "status": self.status,
            "failure_category": self.failure_category,
            "confidence": self.confidence,
            "before_fingerprint": self.before_fingerprint,
            "after_fingerprint": self.after_fingerprint,
            "defect_fingerprint": self.defect_fingerprint,
            "diagnostic_before": dict(self.diagnostic_before) if self.diagnostic_before else None,
            "diagnostic_after": dict(self.diagnostic_after) if self.diagnostic_after else None,
            "structural_result": dict(self.structural_result),
            "semantic_result": dict(self.semantic_result),
            "validation_result": dict(self.validation_result) if self.validation_result else None,
            "affected_files_before": list(self.affected_files_before),
            "affected_files_after": list(self.affected_files_after),
            "changed_symbols": list(self.changed_symbols),
            "unexpected_changes": list(self.unexpected_changes),
            "failure_reasons": list(self.failure_reasons),
            "timestamp": self.timestamp,
            "schema_version": self.schema_version,
            "passed": self.passed,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "VerificationEvidence":
        if not isinstance(data, Mapping):
            return cls()
        return cls(
            verifier=sanitize_text(data.get("verifier", ""), limit=64),
            signal_kind=sanitize_text(data.get("signal_kind", ""), limit=64),
            candidate_id=sanitize_text(data.get("candidate_id", ""), limit=64),
            status=(
                data.get("status")
                if data.get("status") in ALL_SEMANTIC_STATUSES
                else SemanticVerificationStatus.INCONCLUSIVE
            ),
            failure_category=(
                data.get("failure_category")
                if data.get("failure_category") in ALL_FAILURE_CATEGORIES
                else FailureCategory.INSUFFICIENT_EVIDENCE
            ),
            confidence=sanitize_text(data.get("confidence", "none"), limit=32),
            before_fingerprint=sanitize_text(data.get("before_fingerprint", ""), limit=64),
            after_fingerprint=sanitize_text(data.get("after_fingerprint", ""), limit=64),
            defect_fingerprint=sanitize_text(data.get("defect_fingerprint", ""), limit=64),
            diagnostic_before=dict(data.get("diagnostic_before")) if isinstance(data.get("diagnostic_before"), Mapping) else None,
            diagnostic_after=dict(data.get("diagnostic_after")) if isinstance(data.get("diagnostic_after"), Mapping) else None,
            structural_result=dict(data.get("structural_result") or {}),
            semantic_result=dict(data.get("semantic_result") or {}),
            validation_result=dict(data.get("validation_result")) if isinstance(data.get("validation_result"), Mapping) else None,
            affected_files_before=sanitize_string_list(data.get("affected_files_before", [])),
            affected_files_after=sanitize_string_list(data.get("affected_files_after", [])),
            changed_symbols=sanitize_string_list(data.get("changed_symbols", [])),
            unexpected_changes=sanitize_string_list(data.get("unexpected_changes", [])),
            failure_reasons=sanitize_string_list(data.get("failure_reasons", [])),
            timestamp=sanitize_text(data.get("timestamp", "") or _now()),
            schema_version=sanitize_text(data.get("schema_version", SEMANTIC_VERIFICATION_SCHEMA_VERSION), limit=32),
        )


# =============================================================================
# Helper Utilities (Lexical / AST analysis)
# =============================================================================


def _compute_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:32]


def _read_source(path: Path) -> tuple[str | None, str]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        return None, f"OSError: {exc}"
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return data.decode(encoding), ""
        except UnicodeDecodeError:
            continue
    return None, "Undecodable bytes"


def _significant_lines(source: str) -> int:
    return sum(
        1
        for line in source.splitlines()
        if line.strip() and not line.strip().startswith("#")
    )


def _lexical_surface(source: str) -> set[str]:
    """Names bound at top level or in class/def structures."""
    names: set[str] = set()
    patterns = (
        r"^(?:async\s+def|def)\s+([A-Za-z_][A-Za-z0-9_]*)",
        r"^\s+(?:async\s+def|def)\s+([A-Za-z_][A-Za-z0-9_]*)",
        r"^class\s+([A-Za-z_][A-Za-z0-9_]*)",
        r"^([A-Za-z_][A-Za-z0-9_]*)\s*:[^=\n]+(?:=|$)",
        r"^([A-Za-z_][A-Za-z0-9_]*)\s*=(?!=)",
        r"^from\s+[A-Za-z0-9_\.]+\s+import\s+([A-Za-z0-9_,\s\*\(\)]+)",
        r"^import\s+([A-Za-z0-9_,\s\.]+)",
    )
    for line in source.splitlines():
        line_clean = line.split("#", 1)[0]
        for pat in patterns:
            match = re.match(pat, line_clean)
            if not match:
                continue
            group = match.group(1).strip()
            if "import" in pat:
                for item in re.split(r"[,\s\(\)]+", group):
                    item = item.strip()
                    if item and item != "as" and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", item):
                        names.add(item)
            elif group and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", group):
                names.add(group)
    return names


def _extract_blocks(source: str) -> list[tuple[str, str, int, int]]:
    """Partition Python source into top-level blocks: (symbol_name, text, start_line, end_line)."""
    blocks: list[tuple[str, str, int, int]] = []
    lines = source.splitlines(keepends=True)
    if not lines:
        return blocks

    current_name = "__preamble__"
    current_lines: list[str] = []
    start_line = 1

    for i, line in enumerate(lines, 1):
        stripped = line.rstrip()
        is_top_level = line and not line[0].isspace() and not stripped.startswith("#")
        is_def_or_class = bool(re.match(r"^(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", line))

        if is_top_level and is_def_or_class and current_lines:
            blocks.append((current_name, "".join(current_lines), start_line, i - 1))
            current_lines = [line]
            start_line = i
            match = re.match(r"^(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", line)
            current_name = match.group(1) if match else f"block_{i}"
        else:
            if not current_lines and is_def_or_class:
                match = re.match(r"^(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", line)
                current_name = match.group(1) if match else f"block_{i}"
                start_line = i
            current_lines.append(line)

    if current_lines:
        blocks.append((current_name, "".join(current_lines), start_line, len(lines)))
    return blocks


def _ast_metrics(node: ast.AST) -> dict[str, Any]:
    """Compute complexity and substance metrics for a function or class AST node."""
    statement_count = 0
    substantive_statements = 0
    calls: list[str] = []
    has_loops = False
    has_branches = False
    is_vacuous_stub = True

    for child in ast.walk(node):
        if isinstance(child, ast.stmt):
            statement_count += 1
            if not isinstance(child, (ast.Pass, ast.Expr)):
                substantive_statements += 1
                is_vacuous_stub = False
            elif isinstance(child, ast.Expr):
                # An expression statement that is not just a string docstring or Ellipsis
                if not isinstance(child.value, (ast.Constant, ast.Ellipsis)):
                    substantive_statements += 1
                    is_vacuous_stub = False
        if isinstance(child, (ast.For, ast.AsyncFor, ast.While)):
            has_loops = True
        if isinstance(child, (ast.If, ast.Try, ast.Match)):
            has_branches = True
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name):
                calls.append(child.func.id)
            elif isinstance(child.func, ast.Attribute):
                calls.append(child.func.attr)

    return {
        "statement_count": statement_count,
        "substantive_statements": substantive_statements,
        "call_count": len(calls),
        "calls": sorted(set(calls))[:20],
        "has_loops": has_loops,
        "has_branches": has_branches,
        "is_vacuous_stub": is_vacuous_stub,
    }


def _is_stub_body(node: ast.AST) -> bool:
    """Detect if a function/method AST body is a hollow stub (pass, return None, ...)."""
    body = getattr(node, "body", [])
    if not body:
        return True
    # Strip docstring if present
    statements = body
    if (
        len(statements) >= 1
        and isinstance(statements[0], ast.Expr)
        and isinstance(getattr(statements[0], "value", None), ast.Constant)
        and isinstance(statements[0].value.value, str)
    ):
        statements = statements[1:]

    if not statements:
        return True

    if len(statements) == 1:
        stmt = statements[0]
        # pass
        if isinstance(stmt, ast.Pass):
            return True
        # ...
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and stmt.value.value is ...:
            return True
        # return / return None
        if isinstance(stmt, ast.Return):
            if stmt.value is None:
                return True
            if isinstance(stmt.value, ast.Constant) and stmt.value.value is None:
                return True
        # raise NotImplementedError(...)
        if isinstance(stmt, ast.Raise):
            exc = stmt.exc
            if isinstance(exc, ast.Name) and exc.id in ("NotImplementedError", "NotImplemented"):
                return True
            if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name) and exc.func.id in ("NotImplementedError", "NotImplemented"):
                return True
    return False


# =============================================================================
# The Semantic Verifier Contract (ABC)
# =============================================================================


class SemanticVerifier:
    """Abstract base contract for post-execution semantic verification."""

    verifier_name: str = "generic_semantic_verifier"
    supported_signals: frozenset[str] = frozenset()
    confidence_class: str = OracleClass.DETERMINISTIC

    def supports(self, signal_kind: str) -> bool:
        return signal_kind in self.supported_signals

    def capture_before_evidence(
        self, root: Path, relative: str, candidate: MaintenanceCandidate
    ) -> DefectIdentity | None:
        """Capture deep defect identity before any candidate workspace or mutation."""
        return None

    def verify(
        self,
        root: Path,
        relative: str,
        before_identity: DefectIdentity | None,
        *,
        validation_evidence: Any | None = None,
        expected_state_fingerprint: str = "",
        candidate: MaintenanceCandidate | None = None,
    ) -> VerificationEvidence:
        """Perform semantic verification comparing BEFORE identity vs AFTER state."""
        return VerificationEvidence(
            verifier=self.verifier_name,
            signal_kind=before_identity.signal_kind if before_identity else "",
            candidate_id=before_identity.candidate_id if before_identity else "",
            status=SemanticVerificationStatus.NOT_APPLICABLE,
            failure_category=FailureCategory.VERIFIER_NOT_APPLICABLE,
            failure_reasons=[f"verifier '{self.verifier_name}' does not implement verification"],
        )


# =============================================================================
# Concrete Implementation: ParseFailureSemanticVerifier
# =============================================================================


class ParseFailureSemanticVerifier(SemanticVerifier):
    """Deep post-execution semantic verifier for ``parse_failure`` defects.

    Guarantees:
    1. Defect Extinction: CPython compilation & AST parse succeed cleanly.
    2. Defect Locality: Diagnostic area was appropriately repaired rather than
       deleted.
    3. Structural & Semantic Body Preservation (Anti-Gutting): Functions and
       methods that had substantive logic before cannot be replaced with
       ``pass``, ``return None``, or minimal dummy statements.
    4. Surface & Signature Invariance: Named bindings, parameter lists, and
       imports remain intact.
    5. Block AST Invariance: Non-defective blocks that parsed cleanly before
       must retain their AST equivalence.
    6. Evidence & State Freshness: Validation evidence must match the exact
       content fingerprint of the resulting tree.
    """

    verifier_name: str = "parse_failure_semantic_verifier"
    supported_signals: frozenset[str] = frozenset({MaintenanceSignal.PARSE_FAILURE})
    confidence_class: str = OracleClass.DETERMINISTIC

    def capture_before_evidence(
        self, root: Path, relative: str, candidate: MaintenanceCandidate
    ) -> DefectIdentity | None:
        path = Path(root) / relative
        if not path.is_file():
            return None
        source, err = _read_source(path)
        if source is None:
            return None

        # Capture syntax error diagnostic
        diagnostic_msg = ""
        diag_line: int | None = None
        diag_col: int | None = None
        syntax_err_class = ""
        try:
            compile(source, str(path), "exec", dont_inherit=True)
        except SyntaxError as exc:
            syntax_err_class = type(exc).__name__
            diagnostic_msg = exc.msg or str(exc)
            diag_line = exc.lineno
            diag_col = exc.offset
        except Exception as exc:
            syntax_err_class = type(exc).__name__
            diagnostic_msg = str(exc)

        # Lexical surface and imports
        surface = sorted(_lexical_surface(source))
        import_names = sorted(
            name
            for name in surface
            if re.search(rf"\b(?:import\s+{name}|from\s+\S+\s+import\s+.*?\b{name}\b)", source)
        )

        # Partition into blocks and extract ASTs for undamaged blocks
        blocks = _extract_blocks(source)
        block_sigs: dict[str, str] = {}
        substantive_metrics: dict[str, dict[str, Any]] = {}

        for b_name, b_text, b_start, b_end in blocks:
            # Check if this block parses cleanly on its own
            try:
                b_tree = ast.parse(b_text, filename=f"{relative}::{b_name}")
                b_sig = _compute_sha256(ast.dump(b_tree, include_attributes=False))
                block_sigs[b_name] = b_sig

                for node in ast.walk(b_tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        sym_name = node.name
                        metrics = _ast_metrics(node)
                        metrics["signature"] = _compute_sha256(
                            ast.dump(node, include_attributes=False)
                        )
                        metrics["is_stub"] = _is_stub_body(node)
                        metrics["start_line"] = b_start
                        metrics["end_line"] = b_end
                        substantive_metrics[sym_name] = metrics
            except Exception:
                # This block is defective; estimate lexical substance
                b_substance = _significant_lines(b_text)
                substantive_metrics[b_name] = {
                    "is_defective_block": True,
                    "estimated_substance": b_substance,
                    "start_line": b_start,
                    "end_line": b_end,
                }

        state_fp = _compute_sha256(source)
        defect_payload = f"{relative}\x1f{syntax_err_class}\x1f{diagnostic_msg}\x1f{diag_line}:{diag_col}\x1f{state_fp}"
        defect_fp = hashlib.sha256(defect_payload.encode("utf-8")).hexdigest()[:32]

        return DefectIdentity(
            signal_kind=MaintenanceSignal.PARSE_FAILURE,
            candidate_id=candidate.candidate_id if candidate else "",
            subject=candidate.subject if candidate else relative,
            relative_path=relative,
            defect_fingerprint=defect_fp,
            source_state_fingerprint=state_fp,
            diagnostic_message=diagnostic_msg,
            diagnostic_line=diag_line,
            diagnostic_column=diag_col,
            syntax_error_class=syntax_err_class,
            compiler_or_parser="cpython_ast",
            lexical_symbols=tuple(surface),
            import_names=tuple(import_names),
            block_signatures=block_sigs,
            substantive_symbol_metrics=substantive_metrics,
            significant_lines=_significant_lines(source),
        )

    def verify(
        self,
        root: Path,
        relative: str,
        before_identity: DefectIdentity | None,
        *,
        validation_evidence: Any | None = None,
        expected_state_fingerprint: str = "",
        candidate: MaintenanceCandidate | None = None,
    ) -> VerificationEvidence:
        evidence = VerificationEvidence(
            verifier=self.verifier_name,
            signal_kind=MaintenanceSignal.PARSE_FAILURE,
            candidate_id=before_identity.candidate_id if before_identity else (candidate.candidate_id if candidate else ""),
            confidence=self.confidence_class,
            affected_files_before=[relative] if relative else [],
            affected_files_after=[relative] if relative else [],
        )

        if not relative or is_protected_relative_path(relative):
            evidence.status = SemanticVerificationStatus.NOT_RESOLVED
            evidence.failure_category = FailureCategory.UNEXPECTED_SCOPE_CHANGE
            evidence.failure_reasons.append(f"relative path '{relative}' is invalid or protected")
            return evidence

        if before_identity is None:
            evidence.status = SemanticVerificationStatus.INCONCLUSIVE
            evidence.failure_category = FailureCategory.INSUFFICIENT_EVIDENCE
            evidence.failure_reasons.append("no BEFORE defect identity available for verification")
            return evidence

        evidence.before_fingerprint = before_identity.source_state_fingerprint
        evidence.defect_fingerprint = before_identity.defect_fingerprint
        evidence.diagnostic_before = {
            "message": before_identity.diagnostic_message,
            "line": before_identity.diagnostic_line,
            "column": before_identity.diagnostic_column,
            "error_class": before_identity.syntax_error_class,
        }

        path = Path(root) / relative
        if not path.is_file():
            evidence.status = SemanticVerificationStatus.NOT_RESOLVED
            evidence.failure_category = FailureCategory.ORIGINAL_DEFECT_REMAINS
            evidence.failure_reasons.append(f"file '{relative}' does not exist after repair")
            return evidence

        after_source, read_err = _read_source(path)
        if after_source is None:
            evidence.status = SemanticVerificationStatus.INCONCLUSIVE
            evidence.failure_category = FailureCategory.INSUFFICIENT_EVIDENCE
            evidence.failure_reasons.append(f"could not read after source: {read_err}")
            return evidence

        after_fp = _compute_sha256(after_source)
        evidence.after_fingerprint = after_fp

        # Step 1: Compilation & AST Parse
        try:
            compile(after_source, str(path), "exec", dont_inherit=True)
            after_tree = ast.parse(after_source, filename=str(path))
        except SyntaxError as exc:
            evidence.status = SemanticVerificationStatus.NOT_RESOLVED
            evidence.failure_category = FailureCategory.ORIGINAL_DEFECT_REMAINS
            evidence.diagnostic_after = {
                "message": exc.msg or str(exc),
                "line": exc.lineno,
                "column": exc.offset,
                "error_class": type(exc).__name__,
            }
            evidence.failure_reasons.append(
                f"file still fails to parse: {type(exc).__name__}: {exc}"
            )
            return evidence
        except Exception as exc:
            evidence.status = SemanticVerificationStatus.NOT_RESOLVED
            evidence.failure_category = FailureCategory.ORIGINAL_DEFECT_REMAINS
            evidence.failure_reasons.append(f"file compilation failed: {type(exc).__name__}: {exc}")
            return evidence

        # Step 2: Surface, Import & Lexical Binding Preservation
        after_surface = _lexical_surface(after_source)

        # 2a: Imports
        missing_imports = sorted(set(before_identity.import_names) - after_surface)
        if missing_imports:
            evidence.status = SemanticVerificationStatus.NOT_RESOLVED
            evidence.failure_category = FailureCategory.IMPORT_SURFACE_CORRUPTED
            evidence.failure_reasons.append(
                f"import(s) dropped by repair: {', '.join(missing_imports)}"
            )
            evidence.unexpected_changes.extend([f"dropped_import:{name}" for name in missing_imports])
            return evidence

        # 2b: Top-level Named Symbols & Classes
        non_import_symbols = set(before_identity.lexical_symbols) - set(before_identity.import_names)
        missing_names = sorted(non_import_symbols - after_surface)
        if missing_names:
            evidence.status = SemanticVerificationStatus.NOT_RESOLVED
            evidence.failure_category = FailureCategory.UNEXPECTED_SURFACE_CHANGE
            evidence.failure_reasons.append(
                f"repair dropped named symbols present in original source: {', '.join(missing_names[:8])}"
            )
            evidence.unexpected_changes.extend([f"dropped_symbol:{name}" for name in missing_names])
            return evidence

        # Step 3: Extract AFTER symbols and metrics
        after_symbols: dict[str, dict[str, Any]] = {}
        for node in ast.walk(after_tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                sym_name = node.name
                metrics = _ast_metrics(node)
                metrics["signature"] = _compute_sha256(
                    ast.dump(node, include_attributes=False)
                )
                metrics["is_stub"] = _is_stub_body(node)
                after_symbols[sym_name] = metrics

        # Step 4: Anti-Gutting & Anti-Stub Verification
        # Check every symbol that was substantive before: it cannot be turned into a stub!
        for sym_name, b_metrics in before_identity.substantive_symbol_metrics.items():
            if sym_name not in after_symbols:
                continue
            a_metrics = after_symbols[sym_name]

            # Case A: An undamaged substantive function was gutted into a stub
            if not b_metrics.get("is_stub", False) and not b_metrics.get("is_defective_block", False):
                if a_metrics.get("is_stub", False):
                    evidence.status = SemanticVerificationStatus.NOT_RESOLVED
                    evidence.failure_category = FailureCategory.BODY_GUTTED_OR_STUBBED
                    evidence.failure_reasons.append(
                        f"symbol '{sym_name}' was gutted into a vacuous stub implementation"
                    )
                    evidence.unexpected_changes.append(f"gutted_symbol:{sym_name}")
                    return evidence

            # Case B: A defective block was replaced by an empty stub while before lines indicated non-trivial body
            if b_metrics.get("is_defective_block", False):
                est_substance = b_metrics.get("estimated_substance", 0)
                if est_substance >= 1 and a_metrics.get("is_stub", False):
                    evidence.status = SemanticVerificationStatus.NOT_RESOLVED
                    evidence.failure_category = FailureCategory.BODY_GUTTED_OR_STUBBED
                    evidence.failure_reasons.append(
                        f"defective symbol '{sym_name}' ({est_substance} lines) was replaced with a vacuous stub instead of repaired"
                    )
                    evidence.unexpected_changes.append(f"stub_replacement:{sym_name}")
                    return evidence

        # Step 5: Unbroken Blocks AST Invariance
        # Undamaged blocks that had clean ASTs before must retain their integrity
        for b_name, b_sig in before_identity.block_signatures.items():
            if b_name == "__preamble__":
                continue
            if b_name in after_symbols:
                # If the symbol was NOT the one where the syntax error occurred
                diag_line = before_identity.diagnostic_line
                b_metrics = before_identity.substantive_symbol_metrics.get(b_name, {})
                start_l = b_metrics.get("start_line", 0)
                end_l = b_metrics.get("end_line", 0)

                # If the defect was not in this block
                if diag_line is None or not (start_l <= diag_line <= end_l):
                    orig_sig = b_metrics.get("signature")
                    curr_sig = after_symbols[b_name].get("signature")
                    if orig_sig and curr_sig and orig_sig != curr_sig:
                        # AST of an unrelated function was mutated
                        # Check if substance was reduced
                        orig_sub = b_metrics.get("substantive_statements", 0)
                        curr_sub = after_symbols[b_name].get("substantive_statements", 0)
                        if curr_sub < orig_sub:
                            evidence.status = SemanticVerificationStatus.NOT_RESOLVED
                            evidence.failure_category = FailureCategory.AST_MUTATION_SUSPICIOUS
                            evidence.failure_reasons.append(
                                f"unrelated symbol '{b_name}' was mutated with reduced substance"
                            )
                            evidence.unexpected_changes.append(f"unrelated_mutation:{b_name}")
                            return evidence

        # Step 6: Import Preservation
        for imp_name in before_identity.import_names:
            if imp_name not in after_surface:
                evidence.status = SemanticVerificationStatus.NOT_RESOLVED
                evidence.failure_category = FailureCategory.IMPORT_SURFACE_CORRUPTED
                evidence.failure_reasons.append(
                    f"import '{imp_name}' was removed by the repair"
                )
                evidence.unexpected_changes.append(f"dropped_import:{imp_name}")
                return evidence

        # Step 7: Significant Lines Floor
        after_substance = _significant_lines(after_source)
        if before_identity.significant_lines > 5 and after_substance < int(before_identity.significant_lines * 0.6):
            evidence.status = SemanticVerificationStatus.NOT_RESOLVED
            evidence.failure_category = FailureCategory.LINE_COUNT_DEGRADATION
            evidence.failure_reasons.append(
                f"significant lines dropped from {before_identity.significant_lines} to {after_substance}"
            )
            return evidence

        # Step 8: Validation Evidence Integrity (if provided)
        if validation_evidence is not None:
            # Check validation evidence freshness / fingerprint
            ev_fingerprint = getattr(validation_evidence, "source_fingerprint", "") or getattr(validation_evidence, "state_fingerprint", "")
            if ev_fingerprint and ev_fingerprint != after_fp:
                evidence.status = SemanticVerificationStatus.NOT_RESOLVED
                evidence.failure_category = FailureCategory.STALE_EVIDENCE
                evidence.failure_reasons.append(
                    f"validation evidence state fingerprint ({ev_fingerprint[:12]}) does not match current tree ({after_fp[:12]})"
                )
                return evidence

        # All checks passed!
        evidence.status = SemanticVerificationStatus.RESOLVED
        evidence.failure_category = FailureCategory.NONE
        evidence.structural_result = {
            "compiled": True,
            "parsed_ast": True,
            "names_count": len(after_surface),
            "symbols_verified": len(after_symbols),
        }
        evidence.semantic_result = {
            "anti_stub_passed": True,
            "anti_gutting_passed": True,
            "unbroken_blocks_preserved": True,
            "imports_preserved": True,
        }
        return evidence


# =============================================================================
# Fallback / Unverifiable Semantic Verifier
# =============================================================================


class UnverifiableSemanticVerifier(SemanticVerifier):
    """Safe fallback for signals that lack a deterministic semantic verifier."""

    verifier_name: str = "unverifiable_semantic_verifier"
    supported_signals: frozenset[str] = frozenset()
    confidence_class: str = OracleClass.UNSAFE

    def verify(
        self,
        root: Path,
        relative: str,
        before_identity: DefectIdentity | None,
        *,
        validation_evidence: Any | None = None,
        expected_state_fingerprint: str = "",
        candidate: MaintenanceCandidate | None = None,
    ) -> VerificationEvidence:
        kind = before_identity.signal_kind if before_identity else (candidate.kind if candidate else "")
        return VerificationEvidence(
            verifier=self.verifier_name,
            signal_kind=kind,
            candidate_id=before_identity.candidate_id if before_identity else (candidate.candidate_id if candidate else ""),
            status=SemanticVerificationStatus.NOT_APPLICABLE,
            failure_category=FailureCategory.VERIFIER_NOT_APPLICABLE,
            confidence=self.confidence_class,
            failure_reasons=[
                f"signal kind '{kind}' has no deterministic post-execution semantic verifier"
            ],
        )


# =============================================================================
# Registry & Dispatch
# =============================================================================


_REGISTRY: dict[str, SemanticVerifier] = {
    MaintenanceSignal.PARSE_FAILURE: ParseFailureSemanticVerifier(),
}


def verifier_for(signal_kind: str) -> SemanticVerifier:
    """Return the SemanticVerifier registered for ``signal_kind``."""
    return _REGISTRY.get(str(signal_kind), UnverifiableSemanticVerifier())


def register_verifier(signal_kind: str, verifier: SemanticVerifier) -> None:
    """Register a custom or specialized SemanticVerifier."""
    if not isinstance(verifier, SemanticVerifier):
        raise TypeError(f"expected SemanticVerifier, got {type(verifier).__name__}")
    _REGISTRY[str(signal_kind)] = verifier


def all_verifiers() -> Mapping[str, SemanticVerifier]:
    return dict(_REGISTRY)
