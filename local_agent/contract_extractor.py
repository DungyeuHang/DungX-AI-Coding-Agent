from __future__ import annotations

import ast
import datetime
import logging
import re
import uuid
from pathlib import Path
from typing import Any

from .filesystem import ProjectFilesystem
from .models import ExportedSymbol, RunReport, Subtask, SubtaskContract

LOGGER = logging.getLogger(__name__)


class ContractExtractor:
    """
    Deterministically synthesizes a bounded SubtaskContract from verified subtask execution evidence:
    - AST parsing of modified/created Python modules for exported classes, functions, and types
    - Regex pattern extraction for TypeScript/JavaScript exports
    - Verified validation commands executed during the run
    - Concise, deterministic architectural invariants and file deltas
    """

    def __init__(self, filesystem: ProjectFilesystem | None = None, project_root: Path | None = None):
        self.filesystem = filesystem
        self.project_root = project_root or (filesystem.root if filesystem else None)

    def extract_contract(
        self,
        subtask: Subtask,
        report: RunReport,
        preexisting_files: set[str] | None = None,
    ) -> SubtaskContract:
        """Extracts an authoritative, bounded SubtaskContract for a completed subtask."""
        all_changed = list(report.changed_files) if hasattr(report, "changed_files") and report.changed_files else []
        if not all_changed and hasattr(report, "plan") and report.plan:
            all_changed = list(set(report.plan.files_likely_to_change + report.plan.files_likely_to_create))

        # Classify created vs modified files
        created_files: list[str] = []
        modified_files: list[str] = []

        if preexisting_files is not None:
            for f in all_changed:
                if f in preexisting_files:
                    modified_files.append(f)
                else:
                    created_files.append(f)
        else:
            if hasattr(report, "plan") and report.plan:
                created_set = set(report.plan.files_likely_to_create)
                for f in all_changed:
                    if f in created_set:
                        created_files.append(f)
                    else:
                        modified_files.append(f)
            else:
                modified_files = list(all_changed)

        # Extract exported symbols from changed files
        exported_symbols: list[ExportedSymbol] = []
        for file_path in all_changed:
            if len(exported_symbols) >= 10:
                break
            symbols = self._extract_symbols_from_file(file_path)
            for sym in symbols:
                if len(exported_symbols) >= 10:
                    break
                exported_symbols.append(sym)

        # Extract verified validation commands
        validation_commands: list[str] = []
        if hasattr(report, "validation_plan") and report.validation_plan:
            vp = report.validation_plan
            for cmd_spec in (getattr(vp, "targeted_commands", []) + getattr(vp, "primary_commands", []) + getattr(vp, "secondary_commands", [])):
                cmd_str = getattr(cmd_spec, "command_string", None) or " ".join(getattr(cmd_spec, "command", []))
                if cmd_str and cmd_str not in validation_commands:
                    validation_commands.append(cmd_str)
                    if len(validation_commands) >= 10:
                        break

        # Synthesize concise architectural notes
        architectural_notes: list[str] = []
        if created_files:
            architectural_notes.append(f"Introduced new modules: {', '.join(created_files[:5])}")
        if exported_symbols:
            top_classes = [s.name for s in exported_symbols if s.kind == "class"]
            if top_classes:
                architectural_notes.append(f"Exported primary classes: {', '.join(top_classes[:4])}")
            top_funcs = [s.name for s in exported_symbols if s.kind == "function"]
            if top_funcs:
                architectural_notes.append(f"Exported core functions: {', '.join(top_funcs[:4])}")
        if validation_commands:
            architectural_notes.append(f"Verified against suite: {validation_commands[0][:80]}")

        return SubtaskContract(
            subtask_id=subtask.subtask_id,
            title=subtask.title or subtask.goal or subtask.subtask_id,
            exported_symbols=exported_symbols[:10],
            modified_files=modified_files[:20],
            created_files=created_files[:20],
            validation_commands=validation_commands[:10],
            architectural_notes=architectural_notes[:10],
            created_at=datetime.datetime.now(datetime.timezone.utc),
        )

    def _extract_symbols_from_file(self, file_path: str) -> list[ExportedSymbol]:
        """Extracts exported / public symbols from a source file."""
        content = self._read_file_content(file_path)
        if not content:
            return []

        if file_path.endswith(".py"):
            return self._extract_python_symbols(file_path, content)
        elif file_path.endswith((".ts", ".tsx", ".js", ".jsx")):
            return self._extract_js_ts_symbols(file_path, content)
        return []

    def _read_file_content(self, file_path: str) -> str | None:
        try:
            if self.filesystem:
                return self.filesystem.read_file(file_path)
            elif self.project_root:
                full_path = self.project_root / file_path
                if full_path.exists():
                    return full_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            LOGGER.debug("Could not read file %s for contract extraction: %s", file_path, e)
        return None

    def _extract_python_symbols(self, file_path: str, content: str) -> list[ExportedSymbol]:
        symbols: list[ExportedSymbol] = []
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return []

        for node in tree.body:
            if len(symbols) >= 10:
                break

            # Classes
            if isinstance(node, ast.ClassDef):
                if node.name.startswith("_") and not node.name.startswith("__"):
                    continue
                bases_str = f"({', '.join(ast.unparse(b) for b in node.bases)})" if node.bases else ""
                sig = f"class {node.name}{bases_str}:"
                doc = ast.get_docstring(node) or ""
                first_doc_line = doc.strip().split("\n")[0] if doc else ""
                symbols.append(
                    ExportedSymbol(
                        symbol_id=f"{file_path}::{node.name}",
                        name=node.name,
                        kind="class",
                        file_path=file_path,
                        signature=sig,
                        description=first_doc_line,
                    )
                )

            # Functions & Async Functions
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("_") and not node.name.startswith("__"):
                    continue
                prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
                args_str = ast.unparse(node.args) if hasattr(ast, "unparse") else "..."
                ret_str = f" -> {ast.unparse(node.returns)}" if getattr(node, "returns", None) and hasattr(ast, "unparse") else ""
                sig = f"{prefix}{node.name}({args_str}){ret_str}:"
                doc = ast.get_docstring(node) or ""
                first_doc_line = doc.strip().split("\n")[0] if doc else ""
                symbols.append(
                    ExportedSymbol(
                        symbol_id=f"{file_path}::{node.name}",
                        name=node.name,
                        kind="function",
                        file_path=file_path,
                        signature=sig,
                        description=first_doc_line,
                    )
                )

            # Top-level Type Aliases & Constants (UPPERCASE or AnnAssign)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                name = node.target.id
                if not name.startswith("_"):
                    type_str = ast.unparse(node.annotation) if hasattr(ast, "unparse") else ""
                    symbols.append(
                        ExportedSymbol(
                            symbol_id=f"{file_path}::{name}",
                            name=name,
                            kind="type" if "Type" in name or name.isupper() else "variable",
                            file_path=file_path,
                            signature=f"{name}: {type_str}" if type_str else name,
                            description="Module-level type/variable",
                        )
                    )

        return symbols

    def _extract_js_ts_symbols(self, file_path: str, content: str) -> list[ExportedSymbol]:
        symbols: list[ExportedSymbol] = []
        export_pattern = re.compile(
            r"^export\s+(?:async\s+)?(class|function|interface|type|const|let|var)\s+([a-zA-Z0-9_]+)(.*?)(?:\{|=|;|\n)",
            re.MULTILINE,
        )

        for match in export_pattern.finditer(content):
            if len(symbols) >= 10:
                break
            raw_kind, name, rest = match.groups()
            kind_map = {
                "class": "class",
                "function": "function",
                "interface": "type",
                "type": "type",
                "const": "variable",
                "let": "variable",
                "var": "variable",
            }
            kind = kind_map.get(raw_kind, "variable")
            sig = f"export {raw_kind} {name}{rest.strip()}"
            symbols.append(
                ExportedSymbol(
                    symbol_id=f"{file_path}::{name}",
                    name=name,
                    kind=kind,
                    file_path=file_path,
                    signature=sig.strip(),
                    description=f"Exported {kind} from {file_path}",
                )
            )

        return symbols
