from __future__ import annotations

import datetime
import hashlib
import logging
import re
import uuid
from pathlib import Path
from typing import Any

from .models import (
    ArchitecturalInvariant,
    BehavioralAssertion,
    FailurePatternRecord,
    KnowledgeFileNode,
    KnowledgeSymbolNode,
    ProjectContext,
    RepositoryKnowledgeGraph,
    RunReport,
    SubtaskContract,
    Task,
)
from .storage import TaskStorage

LOGGER = logging.getLogger(__name__)

# Secret sanitization patterns
_SECRET_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9_\-]{20,}", re.IGNORECASE),
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}", re.IGNORECASE),
    re.compile(r"ghp_[a-zA-Z0-9]{20,}", re.IGNORECASE),
    re.compile(r"Bearer\s+[a-zA-Z0-9._\-\+/=]{10,}", re.IGNORECASE),
    re.compile(r"(?:api[_-]?key|secret|password|token)\s*[:=]\s*[\"']?([a-zA-Z0-9_\-\.]{6,})[\"']?", re.IGNORECASE),
]


def sanitize_text(text: str) -> str:
    """Sanitizes sensitive credentials, tokens, and keys from persistent text."""
    if not text:
        return ""
    sanitized = text
    for pattern in _SECRET_PATTERNS:
        sanitized = pattern.sub("[REDACTED_SECRET]", sanitized)
    return sanitized


def compute_file_hash(content_bytes: bytes) -> str:
    """Computes a deterministic SHA-256 hash for file content."""
    return hashlib.sha256(content_bytes).hexdigest()


class KnowledgeGraphManager:
    """
    Manages the persistent, hash-aware repository knowledge graph.
    Encapsulates synchronization with filesystem scans, promotion of verified contracts,
    failure pattern indexing, invalidation on file modifications, and bounded context queries.
    """

    def __init__(self, storage: TaskStorage, project_root: str | Path):
        self.storage = storage
        self.project_root = Path(project_root).expanduser().resolve()
        self._graph: RepositoryKnowledgeGraph | None = None

    def get_graph(self) -> RepositoryKnowledgeGraph:
        """Returns the active knowledge graph, loading it lazily from storage if needed."""
        if self._graph is None:
            self._graph = self.storage.load_knowledge_graph()
            if not self._graph.repo_id:
                self._graph.repo_id = self.project_root.name
        return self._graph

    def save(self) -> None:
        """Flushes the knowledge graph to storage after compaction."""
        graph = self.get_graph()
        self.compact()
        graph.updated_at = datetime.datetime.now(datetime.timezone.utc)
        self.storage.save_knowledge_graph(graph)

    def calculate_file_hash(self, relative_path: str) -> str | None:
        """Calculates the SHA-256 hash of a file relative to project root."""
        full_path = self.project_root / relative_path
        if not full_path.is_file():
            return None
        try:
            return compute_file_hash(full_path.read_bytes())
        except OSError:
            return None

    def sync_with_scan(self, project_context: ProjectContext) -> None:
        """
        Synchronizes the knowledge graph with a fresh repository scan.
        Detects deleted files, modified files (hash mismatch), and new files.
        Demotes stale behavioral assertions when file content hashes differ.
        """
        graph = self.get_graph()
        scanned_files = set(project_context.source_files + project_context.test_files + project_context.config_files)

        # 1. Handle deleted files
        existing_paths = list(graph.files.keys())
        for path in existing_paths:
            if path not in scanned_files and not (self.project_root / path).is_file():
                # Remove file node and orphan symbols
                file_node = graph.files.pop(path, None)
                if file_node:
                    for sym_id in file_node.exported_symbol_ids:
                        graph.symbols.pop(sym_id, None)

        # Also purge any symbols whose file_path is no longer on disk
        orphan_symbols = [
            sym_id for sym_id, sym in graph.symbols.items()
            if sym.file_path and sym.file_path not in scanned_files and not (self.project_root / sym.file_path).is_file()
        ]
        for sym_id in orphan_symbols:
            graph.symbols.pop(sym_id, None)

        # 2. Check hash validity for all scanned files
        for rel_path in scanned_files:
            current_hash = self.calculate_file_hash(rel_path)
            if not current_hash:
                continue

            if rel_path in graph.files:
                file_node = graph.files[rel_path]
                if file_node.content_hash != current_hash:
                    # Content changed on disk: update hash and invalidate behavioral assertions
                    file_node.content_hash = current_hash
                    file_node.last_modified_at = datetime.datetime.now(datetime.timezone.utc)
                    for sym_id in file_node.exported_symbol_ids:
                        if sym_id in graph.symbols:
                            sym = graph.symbols[sym_id]
                            sym.content_hash = current_hash
                            # Demote verified behaviors since underlying code changed
                            sym.verified_behaviors = [
                                b for b in sym.verified_behaviors
                                if b.status == "passed" and getattr(b, "commit_sha", None) == current_hash
                            ]
                            if not sym.verified_behaviors:
                                sym.confidence = max(0.5, sym.confidence * 0.7)
            else:
                # New file node discovered
                graph.files[rel_path] = KnowledgeFileNode(
                    path=rel_path,
                    content_hash=current_hash,
                    language=Path(rel_path).suffix,
                    last_modified_at=datetime.datetime.now(datetime.timezone.utc),
                )

    def invalidate_paths(self, modified_paths: set[str]) -> None:
        """Explicitly invalidates cached knowledge for paths modified during execution."""
        graph = self.get_graph()
        for path in modified_paths:
            new_hash = self.calculate_file_hash(path) or ""
            if path in graph.files:
                fn = graph.files[path]
                fn.content_hash = new_hash
                fn.last_modified_at = datetime.datetime.now(datetime.timezone.utc)
                for sym_id in fn.exported_symbol_ids:
                    if sym_id in graph.symbols:
                        sym = graph.symbols[sym_id]
                        sym.content_hash = new_hash
                        # Demote stale assertions
                        sym.verified_behaviors = []
                        sym.confidence = 0.6

    def promote_subtask_contract(self, contract: SubtaskContract, file_hashes: dict[str, str] | None = None, task_id: str | None = None) -> None:
        """
        Promotes verified SubtaskContract knowledge (exported symbols, validation commands,
        architectural invariants, behavioral evidence) into the persistent knowledge graph.
        """
        if not contract:
            return
        graph = self.get_graph()
        hashes = file_hashes or {}
        now = datetime.datetime.now(datetime.timezone.utc)

        # 1. Promote created and modified file nodes
        all_contract_files = list(contract.created_files) + list(contract.modified_files)
        for path in all_contract_files:
            file_hash = hashes.get(path) or self.calculate_file_hash(path) or ""
            if path not in graph.files:
                graph.files[path] = KnowledgeFileNode(
                    path=path,
                    content_hash=file_hash,
                    language=Path(path).suffix,
                    last_modified_task_id=task_id or contract.subtask_id,
                    last_modified_at=now,
                )
            else:
                fn = graph.files[path]
                if file_hash:
                    fn.content_hash = file_hash
                fn.last_modified_task_id = task_id or contract.subtask_id
                fn.last_modified_at = now

            # Record validation commands against file
            if contract.validation_commands:
                existing_cmds = set(graph.files[path].validation_commands)
                for cmd in contract.validation_commands:
                    clean_cmd = sanitize_text(cmd)
                    if clean_cmd and clean_cmd not in existing_cmds:
                        graph.files[path].validation_commands.append(clean_cmd)

        # 2. Promote Exported Symbols & Behavioral Evidence
        evidence_by_symbol: dict[str, list[BehavioralAssertion]] = {}
        if contract.behavioral_evidence:
            for rec in contract.behavioral_evidence:
                if rec.status == "passed":
                    assertion = BehavioralAssertion(
                        assertion_id=str(uuid.uuid4()),
                        description=sanitize_text(f"Passes command: {rec.command}"),
                        test_command=sanitize_text(rec.command),
                        status="passed",
                        commit_sha=rec.test_id,
                        verified_at=now,
                    )
                    for sym_name in rec.exercised_symbols:
                        evidence_by_symbol.setdefault(sym_name, []).append(assertion)

        for sym in contract.exported_symbols:
            sym_id = sym.symbol_id or f"{sym.file_path}::{sym.name}"
            sym_file_hash = hashes.get(sym.file_path) or self.calculate_file_hash(sym.file_path) or ""

            # Check if we have passing behavioral evidence for this symbol
            behaviors = evidence_by_symbol.get(sym.name, [])
            provenance = "behavioral_test" if (sym.verified or behaviors) else "subtask_contract"
            confidence = 1.0 if (sym.verified or behaviors) else 0.85

            if sym_id in graph.symbols:
                existing_node = graph.symbols[sym_id]
                existing_node.signature = sym.signature or existing_node.signature
                existing_node.docstring = sanitize_text(sym.description) or existing_node.docstring
                existing_node.content_hash = sym_file_hash or existing_node.content_hash
                existing_node.provenance = provenance
                existing_node.confidence = max(existing_node.confidence, confidence)
                existing_node.last_verified_at = now
                if behaviors:
                    existing_node.verified_behaviors.extend(behaviors)
                    # Deduplicate assertions
                    seen_cmds = set()
                    unique_b = []
                    for b in existing_node.verified_behaviors:
                        if b.test_command not in seen_cmds:
                            seen_cmds.add(b.test_command)
                            unique_b.append(b)
                    existing_node.verified_behaviors = unique_b[:10]
            else:
                new_node = KnowledgeSymbolNode(
                    symbol_id=sym_id,
                    name=sym.name,
                    kind=sym.kind if sym.kind in ("class", "function", "method", "type", "variable") else "function",
                    file_path=sym.file_path,
                    signature=sym.signature,
                    docstring=sanitize_text(sym.description),
                    content_hash=sym_file_hash,
                    verified_behaviors=behaviors[:10],
                    confidence=confidence,
                    provenance=provenance,
                    last_verified_at=now if (sym.verified or behaviors) else None,
                )
                graph.symbols[sym_id] = new_node

            # Link symbol to file node
            if sym.file_path in graph.files:
                if sym_id not in graph.files[sym.file_path].exported_symbol_ids:
                    graph.files[sym.file_path].exported_symbol_ids.append(sym_id)

        # 3. Promote Architectural Notes as Invariants
        if contract.architectural_notes:
            for note in contract.architectural_notes:
                clean_note = sanitize_text(note)
                if not clean_note:
                    continue
                # Avoid duplicate invariants
                if any(inv.rule_text == clean_note for inv in graph.invariants):
                    continue
                invariant = ArchitecturalInvariant(
                    invariant_id=str(uuid.uuid4()),
                    scope="module" if contract.modified_files else "repository",
                    target_path=contract.modified_files[0] if contract.modified_files else "*",
                    rule_text=clean_note,
                    enforcement_type="contract",
                    source_task_id=task_id or contract.subtask_id,
                    confidence=0.9,
                    created_at=now,
                )
                graph.invariants.append(invariant)

    def promote_run_report(self, task: Task, report: RunReport, file_hashes: dict[str, str] | None = None) -> None:
        """
        Promotes knowledge from a completed task RunReport, including subtask contracts,
        failure-to-repair recipes, and review consensus.
        """
        if not report or not report.completed:
            return
        hashes = file_hashes or {}

        # 1. Promote subtask contracts if plan has subtasks
        if task.plan and task.plan.subtasks:
            for subtask in task.plan.subtasks:
                if subtask.contract:
                    self.promote_subtask_contract(subtask.contract, hashes, task_id=task.task_id)

        # 2. Record failure/repair pattern if repair succeeded
        if report.plan and hasattr(report.plan, "repair_history") and report.plan.repair_history:
            for repair in report.plan.repair_history:
                if isinstance(repair, dict) and repair.get("status") == "success":
                    failing_cmd = repair.get("command", "")
                    error_sig = repair.get("error", "")
                    fix_summary = repair.get("summary", "Successfully repaired test failure")
                    if failing_cmd or error_sig:
                        self.record_failure_pattern(
                            failing_command=failing_cmd,
                            error_text=error_sig,
                            repair_summary=fix_summary,
                            affected_files=list(report.changed_files),
                        )

    def record_failure_pattern(
        self,
        failing_command: str,
        error_text: str,
        repair_summary: str,
        affected_files: list[str] | None = None,
    ) -> None:
        """Records a recurring failure pattern and its verified successful repair."""
        graph = self.get_graph()
        clean_err = sanitize_text(error_text)[:500].strip()
        clean_cmd = sanitize_text(failing_command).strip()
        clean_repair = sanitize_text(repair_summary)[:500].strip()

        if not clean_err and not clean_cmd:
            return

        # Check for existing matching pattern
        for pat in graph.failure_patterns:
            if (pat.failing_command == clean_cmd and clean_cmd) or (clean_err and pat.error_signature == clean_err):
                pat.occurrence_count += 1
                pat.last_seen_at = datetime.datetime.now(datetime.timezone.utc)
                pat.confidence = min(1.0, pat.confidence + 0.05)
                if clean_repair:
                    pat.successful_repair_summary = clean_repair
                if affected_files:
                    merged_files = set(pat.affected_files) | set(affected_files)
                    pat.affected_files = list(merged_files)[:10]
                return

        new_pat = FailurePatternRecord(
            pattern_id=str(uuid.uuid4()),
            error_signature=clean_err,
            failing_command=clean_cmd,
            root_cause_summary="Automated repair pattern",
            successful_repair_summary=clean_repair,
            affected_files=list(affected_files or [])[:10],
            occurrence_count=1,
            confidence=0.8,
            last_seen_at=datetime.datetime.now(datetime.timezone.utc),
        )
        graph.failure_patterns.append(new_pat)

    def query_context_knowledge(
        self,
        task_objective: str,
        changed_files: list[str] | None = None,
        max_chars: int = 2000,
    ) -> tuple[str, list[KnowledgeSymbolNode], list[ArchitecturalInvariant], list[FailurePatternRecord]]:
        """
        Queries the knowledge graph for hash-valid, relevant knowledge for a given task objective.
        Returns:
            (formatted_prompt_text, matching_symbols, matching_invariants, matching_failure_patterns)
        Enforces a strict max_chars budget cap (default 2000 characters).
        """
        graph = self.get_graph()
        obj_lower = task_objective.lower()
        obj_words = set(re.findall(r"\b[a-zA-Z0-9_]{3,}\b", obj_lower))

        # 1. Match Symbols (only include hash-valid symbols whose content hash matches current disk content)
        matching_symbols: list[KnowledgeSymbolNode] = []
        for sym in graph.symbols.values():
            if not sym.file_path:
                continue

            # Verify hash validity against disk
            current_hash = self.calculate_file_hash(sym.file_path)
            if not current_hash or (sym.content_hash and sym.content_hash != current_hash):
                # Stale symbol; skip from authoritative injection
                continue

            # Check relevance: symbol name in objective or file path in changed_files
            is_name_match = sym.name.lower() in obj_words or sym.name.lower() in obj_lower
            is_file_match = bool(changed_files and sym.file_path in changed_files)
            if is_name_match or is_file_match:
                matching_symbols.append(sym)

        # Sort symbols by confidence and verified status
        matching_symbols.sort(
            key=lambda s: (
                bool(s.verified_behaviors),
                s.provenance == "behavioral_test",
                s.confidence,
            ),
            reverse=True,
        )
        matching_symbols = matching_symbols[:10]

        # 2. Match Architectural Invariants
        matched_file_paths = {s.file_path.lower() for s in matching_symbols if s.file_path}
        matching_invariants: list[ArchitecturalInvariant] = []
        for inv in graph.invariants:
            inv_text_lower = inv.rule_text.lower()
            target_lower = inv.target_path.lower()
            is_rule_match = any(w in inv_text_lower for w in obj_words)
            is_path_match = bool(changed_files and any(f.lower() in target_lower or target_lower in f.lower() for f in changed_files))
            is_target_in_obj = (inv.target_path != "*" and (target_lower in obj_lower or Path(inv.target_path).stem.lower() in obj_words))
            is_symbol_file_match = bool(target_lower in matched_file_paths or any(mf in target_lower for mf in matched_file_paths))
            if is_rule_match or is_path_match or is_target_in_obj or is_symbol_file_match or inv.scope == "repository":
                matching_invariants.append(inv)
        matching_invariants.sort(key=lambda i: i.confidence, reverse=True)
        matching_invariants = matching_invariants[:5]

        # 3. Match Failure Patterns
        matching_patterns: list[FailurePatternRecord] = []
        for pat in graph.failure_patterns:
            pat_text = (pat.error_signature + " " + pat.failing_command).lower()
            if any(w in pat_text for w in obj_words):
                matching_patterns.append(pat)
        matching_patterns.sort(key=lambda p: (p.confidence, p.occurrence_count), reverse=True)
        matching_patterns = matching_patterns[:3]

        # 4. Format Prompt Text with Hard Budget Cap
        sections: list[str] = []
        if matching_symbols:
            sym_lines = ["VERIFIED REPOSITORY INTERFACES & SYMBOLS:"]
            for s in matching_symbols:
                sig = s.signature or s.name
                verified_tag = " [BEHAVIORALLY_VERIFIED]" if s.verified_behaviors else ""
                sym_lines.append(f"  - [{s.kind}] `{sig}` in `{s.file_path}`{verified_tag}")
                for b in s.verified_behaviors[:2]:
                    sym_lines.append(f"    * Proven: {b.description}")
            sections.append("\n".join(sym_lines))

        if matching_invariants:
            inv_lines = ["ARCHITECTURAL INVARIANTS:"]
            for inv in matching_invariants:
                inv_lines.append(f"  - [{inv.enforcement_type.upper()}] {inv.rule_text} ({inv.target_path})")
            sections.append("\n".join(inv_lines))

        if matching_patterns:
            pat_lines = ["HISTORICAL REPAIR PATTERNS:"]
            for pat in matching_patterns:
                pat_lines.append(f"  - On `{pat.failing_command or pat.error_signature[:40]}`: {pat.successful_repair_summary}")
            sections.append("\n".join(pat_lines))

        formatted_text = "\n\n".join(sections)
        if len(formatted_text) > max_chars:
            formatted_text = formatted_text[: max_chars - 3] + "..."

        return formatted_text, matching_symbols, matching_invariants, matching_patterns

    def compact(
        self,
        max_symbols: int = 1000,
        max_invariants: int = 100,
        max_patterns: int = 50,
        max_files: int = 500,
    ) -> None:
        """
        Enforces bounded storage limits. Deterministically prunes oldest, lowest-confidence,
        and unverified entries first while preserving verified contracts.
        """
        graph = self.get_graph()

        # Compaction for symbols
        if len(graph.symbols) > max_symbols:
            sorted_symbols = sorted(
                graph.symbols.items(),
                key=lambda item: (
                    bool(item[1].verified_behaviors),
                    item[1].provenance == "behavioral_test",
                    item[1].confidence,
                    item[1].last_verified_at or datetime.datetime.min.replace(tzinfo=datetime.timezone.utc),
                ),
                reverse=True,
            )
            graph.symbols = dict(sorted_symbols[:max_symbols])

        # Compaction for invariants
        if len(graph.invariants) > max_invariants:
            graph.invariants.sort(
                key=lambda inv: (inv.confidence, inv.created_at),
                reverse=True,
            )
            graph.invariants = graph.invariants[:max_invariants]

        # Compaction for failure patterns (decay stale patterns > 90 days)
        now = datetime.datetime.now(datetime.timezone.utc)
        active_patterns = []
        for pat in graph.failure_patterns:
            age_days = (now - pat.last_seen_at).days
            if age_days < 90 or pat.occurrence_count >= 3:
                active_patterns.append(pat)
        active_patterns.sort(
            key=lambda p: (p.confidence, p.occurrence_count, p.last_seen_at),
            reverse=True,
        )
        graph.failure_patterns = active_patterns[:max_patterns]

        # Compaction for files
        if len(graph.files) > max_files:
            sorted_files = sorted(
                graph.files.items(),
                key=lambda item: (
                    bool(item[1].exported_symbol_ids),
                    item[1].last_modified_at or datetime.datetime.min.replace(tzinfo=datetime.timezone.utc),
                ),
                reverse=True,
            )
            graph.files = dict(sorted_files[:max_files])
