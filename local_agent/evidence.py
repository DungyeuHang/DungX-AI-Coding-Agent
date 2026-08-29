"""Phase 4.17: structured validation evidence, and safe reuse of it.

Phase 4.16 runs real validation commands against an isolated candidate tree
before anything is written to the authoritative tree. When that candidate is
then applied for real, the orchestrator has historically re-run the same
commands against the now-identical authoritative tree - correct, but pure
duplicated work.

This module makes that duplication avoidable *safely*. Every candidate command
execution is recorded as a :class:`ValidationEvidence` entry carrying a content
fingerprint of the files the result depended on. Post-apply, a reuse request is
granted only when :meth:`EvidenceLedger.find_reusable` can prove that every
assumption still holds:

* the command (and its argument vector) is identical,
* the evidence recorded a *pass* - a failure is never reused to skip a rerun,
* the relevant file set is identical,
* the relevant symbol set is identical,
* the content fingerprint over those files is byte-identical, and
* the impact confidence attached to the evidence meets the caller's threshold.

If any one of those fails, the decision is ``reusable=False`` with a specific
machine-readable ``reason`` and the caller reruns. There is deliberately no
"probably fine" path: a false reuse would silently skip validation, which is
the single worst failure this system can have.

Why a *content* fingerprint and not a path or mtime
---------------------------------------------------
The candidate tree and the authoritative tree are different directories, so any
root-relative identity would never match. Hashing file *content* keyed by
repo-relative path makes the fingerprint root-independent, which is exactly the
property needed: candidate evidence is reusable precisely when the applied files
are byte-identical to the candidate's. mtime and size are deliberately not used
- both routinely collide between two revisions written in the same second.

Bounds
------
The ledger is bounded (entry count, per-entry output summary length) so it stays
compact enough to serialise into a checkpoint alongside the existing telemetry,
and it keeps history across candidate rebuild/revalidate iterations rather than
only the last one.
"""

from __future__ import annotations

import datetime
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

#: Default number of evidence entries retained. Old entries are dropped
#: oldest-first, so history is bounded rather than unbounded.
DEFAULT_MAX_EVIDENCE_ENTRIES = 40
#: Default per-stream output budget for one entry. Raw logs are never stored.
DEFAULT_MAX_SUMMARY_CHARS = 400
#: Sentinel recorded in a fingerprint for a path that does not exist.
MISSING_MARKER = "<missing>"

STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"

# Machine-readable invalidation reasons.
REASON_NO_EVIDENCE = "no_matching_evidence"
REASON_COMMAND_MISMATCH = "command_mismatch"
REASON_NOT_PASSED = "evidence_did_not_pass"
REASON_FINGERPRINT_MISMATCH = "tree_state_changed"
REASON_FILES_CHANGED = "relevant_file_set_changed"
REASON_SYMBOLS_CHANGED = "relevant_symbol_set_changed"
REASON_CONFIDENCE_TOO_LOW = "confidence_below_threshold"
REASON_REUSE_DISABLED = "reuse_disabled"
REASON_OK = "assumptions_still_hold"


def _trim(text: str, limit: int) -> str:
    """Head/tail trim, so both the first error and the final summary survive."""
    text = (text or "").strip()
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    return f"{text[:head]}\n... [trimmed] ...\n{text[-tail:]}"


def compute_state_fingerprint(root: str | Path, paths: Iterable[str]) -> str:
    """Root-independent digest of the *content* of ``paths`` under ``root``.

    Deterministic: paths are sorted and each contributes ``path:sha256`` (or
    ``path:<missing>``). A file that cannot be read is recorded as missing
    rather than skipped, so an unreadable file invalidates reuse instead of
    silently matching.
    """
    base = Path(root)
    digest = hashlib.sha256()
    for relative in sorted({str(p).replace("\\", "/") for p in paths if p}):
        candidate = base / relative
        try:
            marker = hashlib.sha256(candidate.read_bytes()).hexdigest() if candidate.is_file() \
                else MISSING_MARKER
        except OSError:
            marker = MISSING_MARKER
        digest.update(relative.encode("utf-8", "replace"))
        digest.update(b"\0")
        digest.update(marker.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass
class ValidationEvidence:
    """One recorded validation command execution and why it was run."""

    command: tuple[str, ...] = ()
    category: str = "unit_test"
    exit_code: int = 0
    duration_seconds: float = 0.0
    status: str = STATUS_PASSED
    #: The impact analyzer's one-sentence explanation for selecting this target.
    selected_because: str = ""
    #: Evidence tier that selected it (see :mod:`local_agent.semantic_impact`).
    tier: str = ""
    impacted_symbols: list[str] = field(default_factory=list)
    impacted_files: list[str] = field(default_factory=list)
    confidence: str = "low"
    stdout_summary: str = ""
    stderr_summary: str = ""
    timestamp: str = ""
    candidate_iteration: int = 0
    #: Where this ran ("candidate" or "authoritative"), for diagnostics only -
    #: never part of the reuse decision, since reuse is content-based.
    environment_root: str = ""
    skipped_reason: str = ""
    #: Content fingerprint of ``impacted_files`` at execution time.
    fingerprint: str = ""

    @property
    def passed(self) -> bool:
        return self.status == STATUS_PASSED

    def display(self) -> str:
        return " ".join(self.command)

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": list(self.command),
            "category": self.category,
            "exit_code": self.exit_code,
            "duration_seconds": round(self.duration_seconds, 4),
            "status": self.status,
            "selected_because": self.selected_because,
            "tier": self.tier,
            "impacted_symbols": list(self.impacted_symbols),
            "impacted_files": list(self.impacted_files),
            "confidence": self.confidence,
            "stdout_summary": self.stdout_summary,
            "stderr_summary": self.stderr_summary,
            "timestamp": self.timestamp,
            "candidate_iteration": self.candidate_iteration,
            "environment_root": self.environment_root,
            "skipped_reason": self.skipped_reason,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "ValidationEvidence":
        """Tolerant: unknown keys ignored, missing keys defaulted."""
        if not isinstance(data, dict):
            return cls()
        return cls(
            command=tuple(str(token) for token in (data.get("command") or [])),
            category=str(data.get("category", "unit_test")),
            exit_code=int(data.get("exit_code", 0) or 0),
            duration_seconds=float(data.get("duration_seconds", 0.0) or 0.0),
            status=str(data.get("status", STATUS_PASSED)),
            selected_because=str(data.get("selected_because", "")),
            tier=str(data.get("tier", "")),
            impacted_symbols=[str(s) for s in (data.get("impacted_symbols") or [])],
            impacted_files=[str(s) for s in (data.get("impacted_files") or [])],
            confidence=str(data.get("confidence", "low")),
            stdout_summary=str(data.get("stdout_summary", "")),
            stderr_summary=str(data.get("stderr_summary", "")),
            timestamp=str(data.get("timestamp", "")),
            candidate_iteration=int(data.get("candidate_iteration", 0) or 0),
            environment_root=str(data.get("environment_root", "")),
            skipped_reason=str(data.get("skipped_reason", "")),
            fingerprint=str(data.get("fingerprint", "")),
        )


@dataclass
class EvidenceReuseDecision:
    """Outcome of a reuse request. Always carries a specific reason."""

    reusable: bool
    reason: str
    evidence: ValidationEvidence | None = None
    #: Seconds of command runtime avoided when ``reusable`` is True.
    time_saved_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "reusable": self.reusable,
            "reason": self.reason,
            "time_saved_seconds": round(self.time_saved_seconds, 4),
            "evidence": self.evidence.to_dict() if self.evidence is not None else None,
        }


class EvidenceLedger:
    """Bounded, iteration-aware history of validation evidence.

    Instance-scoped with no module-level state, so two concurrently-running
    worktree sessions each keep their own ledger and cannot read or corrupt each
    other's evidence.
    """

    def __init__(
        self,
        *,
        max_entries: int = DEFAULT_MAX_EVIDENCE_ENTRIES,
        max_summary_chars: int = DEFAULT_MAX_SUMMARY_CHARS,
    ):
        self.max_entries = max(1, int(max_entries))
        self.max_summary_chars = max(0, int(max_summary_chars))
        self._entries: list[ValidationEvidence] = []
        #: Counters exposed as telemetry; see ``ImplementationResult``.
        self.reuse_grants = 0
        self.reuse_denials = 0
        self.time_saved_seconds = 0.0

    # -- recording ---------------------------------------------------------

    def record(
        self,
        *,
        command: Sequence[str],
        status: str,
        exit_code: int = 0,
        duration_seconds: float = 0.0,
        category: str = "unit_test",
        selected_because: str = "",
        tier: str = "",
        impacted_files: Iterable[str] = (),
        impacted_symbols: Iterable[str] = (),
        confidence: str = "low",
        stdout: str = "",
        stderr: str = "",
        candidate_iteration: int = 0,
        environment_root: str = "",
        skipped_reason: str = "",
        fingerprint: str = "",
    ) -> ValidationEvidence:
        """Append one entry, trimming outputs and evicting the oldest if needed."""
        entry = ValidationEvidence(
            command=tuple(str(token) for token in command),
            category=category,
            exit_code=int(exit_code),
            duration_seconds=float(duration_seconds or 0.0),
            status=status,
            selected_because=selected_because,
            tier=tier,
            impacted_symbols=sorted({str(s) for s in impacted_symbols if s}),
            impacted_files=sorted({str(f).replace("\\", "/") for f in impacted_files if f}),
            confidence=confidence,
            stdout_summary=_trim(stdout, self.max_summary_chars),
            stderr_summary=_trim(stderr, self.max_summary_chars),
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            candidate_iteration=int(candidate_iteration),
            environment_root=str(environment_root),
            skipped_reason=skipped_reason,
            fingerprint=fingerprint,
        )
        self._entries.append(entry)
        if len(self._entries) > self.max_entries:
            # Drop oldest-first: recent iterations are the ones reuse can match.
            del self._entries[: len(self._entries) - self.max_entries]
        return entry

    @property
    def entries(self) -> list[ValidationEvidence]:
        return list(self._entries)

    def entries_for_iteration(self, iteration: int) -> list[ValidationEvidence]:
        return [entry for entry in self._entries if entry.candidate_iteration == iteration]

    @property
    def iterations(self) -> list[int]:
        return sorted({entry.candidate_iteration for entry in self._entries})

    def __len__(self) -> int:
        return len(self._entries)

    # -- reuse -------------------------------------------------------------

    def find_reusable(
        self,
        *,
        command: Sequence[str],
        current_root: str | Path,
        relevant_files: Iterable[str],
        relevant_symbols: Iterable[str] = (),
        min_confidence: str = "medium",
        enabled: bool = True,
    ) -> EvidenceReuseDecision:
        """Decide whether a previously recorded result may stand in for a rerun.

        Every assumption is checked explicitly against *current* state; nothing
        is inferred from the fact that a candidate previously passed. The most
        recent matching entry is preferred, so a later iteration's evidence
        supersedes an earlier one for the same command.
        """
        from .semantic_impact import confidence_at_least

        if not enabled:
            self.reuse_denials += 1
            return EvidenceReuseDecision(False, REASON_REUSE_DISABLED)

        wanted = tuple(str(token) for token in command)
        files = sorted({str(f).replace("\\", "/") for f in relevant_files if f})
        symbols = sorted({str(s) for s in relevant_symbols if s})

        matching = [entry for entry in self._entries if entry.command == wanted]
        if not matching:
            self.reuse_denials += 1
            return EvidenceReuseDecision(False, REASON_COMMAND_MISMATCH
                                         if self._entries else REASON_NO_EVIDENCE)

        # Newest first: the latest candidate iteration is the authoritative one.
        for entry in reversed(matching):
            if not entry.passed:
                continue
            if sorted(entry.impacted_files) != files:
                continue
            if sorted(entry.impacted_symbols) != symbols:
                continue
            if not confidence_at_least(entry.confidence, min_confidence):
                continue
            current = compute_state_fingerprint(current_root, files)
            if current != entry.fingerprint:
                continue
            self.reuse_grants += 1
            self.time_saved_seconds += entry.duration_seconds
            return EvidenceReuseDecision(
                True, REASON_OK, entry, time_saved_seconds=entry.duration_seconds
            )

        # Nothing was reusable: report the most specific reason, checked against
        # the newest matching entry so the message describes the real blocker.
        newest = matching[-1]
        self.reuse_denials += 1
        if not newest.passed:
            return EvidenceReuseDecision(False, REASON_NOT_PASSED, newest)
        if sorted(newest.impacted_files) != files:
            return EvidenceReuseDecision(False, REASON_FILES_CHANGED, newest)
        if sorted(newest.impacted_symbols) != symbols:
            return EvidenceReuseDecision(False, REASON_SYMBOLS_CHANGED, newest)
        if not confidence_at_least(newest.confidence, min_confidence):
            return EvidenceReuseDecision(False, REASON_CONFIDENCE_TOO_LOW, newest)
        return EvidenceReuseDecision(False, REASON_FINGERPRINT_MISMATCH, newest)

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_entries": self.max_entries,
            "max_summary_chars": self.max_summary_chars,
            "reuse_grants": self.reuse_grants,
            "reuse_denials": self.reuse_denials,
            "time_saved_seconds": round(self.time_saved_seconds, 4),
            "entries": [entry.to_dict() for entry in self._entries],
        }

    @classmethod
    def from_dict(cls, data: Any) -> "EvidenceLedger":
        """Tolerant deserialisation; unknown/missing keys use defaults."""
        if not isinstance(data, dict):
            return cls()
        ledger = cls(
            max_entries=int(data.get("max_entries", DEFAULT_MAX_EVIDENCE_ENTRIES) or
                            DEFAULT_MAX_EVIDENCE_ENTRIES),
            max_summary_chars=int(
                data.get("max_summary_chars", DEFAULT_MAX_SUMMARY_CHARS)
                if data.get("max_summary_chars") is not None else DEFAULT_MAX_SUMMARY_CHARS
            ),
        )
        ledger.reuse_grants = int(data.get("reuse_grants", 0) or 0)
        ledger.reuse_denials = int(data.get("reuse_denials", 0) or 0)
        ledger.time_saved_seconds = float(data.get("time_saved_seconds", 0.0) or 0.0)
        entries = [ValidationEvidence.from_dict(item) for item in (data.get("entries") or [])]
        ledger._entries = entries[-ledger.max_entries:]
        return ledger
