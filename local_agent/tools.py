from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path
from typing import Any

from .commands import CommandRunner, CommandSpec, UnsafeCommandError
from .filesystem import ProjectFilesystem, ProtectedPathError, SandboxViolation, SECRET_NAMES
from .models import SemanticIndex, ToolCall, ToolDefinition, ToolResult
from .repository import IGNORED_DIRECTORIES

DEFAULT_MAX_OUTPUT_BYTES = 4000
DEFAULT_MAX_RESULTS = 50
DEFAULT_MAX_LINES = 500

PROTECTED_DIRS = {".git", ".hg", ".svn", ".agent_data", ".agent_worktrees"}


def _is_binary_file(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return b"\x00" in handle.read(4096)
    except OSError:
        return True


def _truncate_output(text: str, max_bytes: int = DEFAULT_MAX_OUTPUT_BYTES) -> tuple[str, bool]:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text, False
    
    # Truncate to max_bytes on valid UTF-8 boundary
    truncated_bytes = encoded[:max_bytes]
    truncated_text = truncated_bytes.decode("utf-8", errors="ignore")
    suffix = "\n... [output truncated: exceeded maximum output limit]"
    return truncated_text + suffix, True


class ToolRegistry:
    """Secure registry for discovering and executing sandboxed agent exploration tools."""

    def __init__(
        self,
        project_root: str | Path,
        filesystem: ProjectFilesystem | None = None,
        command_runner: CommandRunner | None = None,
        semantic_index: SemanticIndex | None = None,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        max_results: int = DEFAULT_MAX_RESULTS,
    ):
        self.root = Path(project_root).expanduser().resolve()
        if not self.root.is_dir():
            raise ValueError(f"project directory does not exist: {self.root}")
        self.filesystem = filesystem or ProjectFilesystem(self.root)
        self.runner = command_runner or CommandRunner(self.root)
        self.semantic_index = semantic_index
        self.max_output_bytes = max_output_bytes
        self.max_results = max_results

        self._definitions: dict[str, ToolDefinition] = {
            "read_file_range": ToolDefinition(
                name="read_file_range",
                description="Reads a numbered slice of lines from a project file.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative or project path to the file to read.",
                        },
                        "start_line": {
                            "type": "integer",
                            "description": "1-based starting line number (inclusive). Default is 1.",
                            "default": 1,
                        },
                        "end_line": {
                            "type": "integer",
                            "description": "1-based ending line number (inclusive). Default is 100.",
                            "default": 100,
                        },
                    },
                    "required": ["path"],
                },
            ),
            "search_symbols": ToolDefinition(
                name="search_symbols",
                description="Searches the semantic AST index for symbol definitions (classes, functions, methods).",
                parameters={
                    "type": "object",
                    "properties": {
                        "symbol_name": {
                            "type": "string",
                            "description": "The symbol or substring to look up across the repository.",
                        },
                    },
                    "required": ["symbol_name"],
                },
            ),
            "grep_code": ToolDefinition(
                name="grep_code",
                description="Performs a regex or literal text search across project source files.",
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Regular expression or literal pattern to search for.",
                        },
                        "glob": {
                            "type": "string",
                            "description": "File glob pattern to filter target files (e.g. '*.py', '*.ts'). Default is '*'.",
                            "default": "*",
                        },
                        "case_sensitive": {
                            "type": "boolean",
                            "description": "Whether to perform case-sensitive search. Default is True.",
                            "default": True,
                        },
                    },
                    "required": ["pattern"],
                },
            ),
            "find_files": ToolDefinition(
                name="find_files",
                description="Finds files matching a glob pattern within the project tree.",
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Glob pattern for matching filenames or paths (e.g. '*.json', 'src/**/*.tsx'). Default is '*'.",
                            "default": "*",
                        },
                    },
                },
            ),
            "run_command_sandbox": ToolDefinition(
                name="run_command_sandbox",
                description="Runs a sandboxed, read-only diagnostic or build command without shell invocation.",
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of command tokens to execute (e.g. ['pytest', 'tests/test_app.py']).",
                        },
                    },
                    "required": ["command"],
                },
            ),
        }

    def definitions(self) -> list[ToolDefinition]:
        """Return the list of tool definitions available in the registry."""
        return list(self._definitions.values())

    def get_definition(self, name: str) -> ToolDefinition | None:
        """Lookup a specific tool definition by name."""
        return self._definitions.get(name)

    def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call securely and return a structured ToolResult."""
        if not isinstance(tool_call, ToolCall):
            return ToolResult(
                call_id=getattr(tool_call, "call_id", "unknown"),
                tool_name=getattr(tool_call, "tool_name", "unknown"),
                output="Invalid tool call object provided.",
                is_error=True,
            )

        handler = getattr(self, f"_tool_{tool_call.tool_name}", None)
        if handler is None:
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                output=f"Unknown tool '{tool_call.tool_name}'. Available tools: {', '.join(self._definitions.keys())}",
                is_error=True,
            )

        try:
            return handler(tool_call)
        except Exception as exc:
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                output=f"Unexpected error executing tool '{tool_call.tool_name}': {exc}",
                is_error=True,
            )

    # -------------------------------------------------------------------------
    # Internal Security Helpers
    # -------------------------------------------------------------------------

    def _guard_path(self, raw_path: str) -> tuple[Path | None, str | None]:
        """Resolve and validate a file path against sandbox and protection rules."""
        try:
            resolved = self.filesystem.resolve(raw_path)
        except (SandboxViolation, ValueError) as exc:
            return None, f"Access denied: path outside project directory: {raw_path}"

        rel_parts = resolved.relative_to(self.root).parts
        for part in rel_parts:
            if part.lower() in PROTECTED_DIRS:
                return None, f"Access denied: protected directory in path: {part}"

        if resolved.name.lower() in {name.lower() for name in SECRET_NAMES}:
            return None, f"Access denied: secret-like file is protected: {resolved.name}"

        return resolved, None

    # -------------------------------------------------------------------------
    # Tool 1: read_file_range
    # -------------------------------------------------------------------------

    def _tool_read_file_range(self, tool_call: ToolCall) -> ToolResult:
        args = tool_call.arguments
        path_str = args.get("path")
        if not path_str or not isinstance(path_str, str):
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                output="Parameter 'path' must be a non-empty string.",
                is_error=True,
            )

        start_line = args.get("start_line", 1)
        end_line = args.get("end_line", 100)

        if not isinstance(start_line, int) or start_line < 1:
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                output=f"Invalid 'start_line': {start_line}. Line numbers must be integers >= 1.",
                is_error=True,
            )
        if not isinstance(end_line, int) or end_line < start_line:
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                output=f"Invalid line range: start_line ({start_line}) cannot be greater than end_line ({end_line}).",
                is_error=True,
            )
        if end_line - start_line + 1 > DEFAULT_MAX_LINES:
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                output=f"Line range too large ({end_line - start_line + 1} lines). Maximum allowed range is {DEFAULT_MAX_LINES} lines.",
                is_error=True,
            )

        resolved, error_msg = self._guard_path(path_str)
        if error_msg is not None:
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                output=error_msg,
                is_error=True,
            )

        if not resolved.exists():
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                output=f"File not found: {path_str}",
                is_error=True,
            )
        if resolved.is_dir():
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                output=f"Path is a directory, not a file: {path_str}",
                is_error=True,
            )
        if _is_binary_file(resolved):
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                output=f"Cannot read binary file: {path_str}",
                is_error=True,
            )

        lines: list[str] = []
        try:
            with resolved.open("r", encoding="utf-8", errors="replace") as f:
                for current_idx, line in enumerate(f, start=1):
                    if current_idx > end_line:
                        break
                    if current_idx >= start_line:
                        lines.append(f"{current_idx}: {line.rstrip('\r\n')}")
        except OSError as exc:
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                output=f"Failed to read file {path_str}: {exc}",
                is_error=True,
            )

        if not lines:
            raw_output = f"File {path_str} has fewer than {start_line} lines (empty slice)."
        else:
            raw_output = "\n".join(lines)

        output, truncated = _truncate_output(raw_output, self.max_output_bytes)
        return ToolResult(
            call_id=tool_call.call_id,
            tool_name=tool_call.tool_name,
            output=output,
            is_error=False,
            truncated=truncated,
        )

    # -------------------------------------------------------------------------
    # Tool 2: search_symbols
    # -------------------------------------------------------------------------

    def _tool_search_symbols(self, tool_call: ToolCall) -> ToolResult:
        args = tool_call.arguments
        symbol_name = args.get("symbol_name")
        if not symbol_name or not isinstance(symbol_name, str) or not symbol_name.strip():
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                output="Parameter 'symbol_name' must be a non-empty string.",
                is_error=True,
            )

        query = symbol_name.strip()
        if self.semantic_index is None:
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                output=f"No semantic index available for repository. Cannot search symbol '{query}'.",
                is_error=False,
            )

        matches = self.semantic_index.search_symbols(query)
        if not matches:
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                output=f"No symbols found matching '{query}'.",
                is_error=False,
            )

        lines: list[str] = []
        truncated = False
        for idx, (path, symbol) in enumerate(matches):
            if idx >= self.max_results:
                truncated = True
                break
            lines.append(f"- {symbol.name} ({symbol.kind}) in {path}:{symbol.location.start_line}")

        if truncated:
            lines.append(f"... [truncated: showing first {self.max_results} of {len(matches)} matches]")

        raw_output = "\n".join(lines)
        output, byte_truncated = _truncate_output(raw_output, self.max_output_bytes)
        return ToolResult(
            call_id=tool_call.call_id,
            tool_name=tool_call.tool_name,
            output=output,
            is_error=False,
            truncated=truncated or byte_truncated,
        )

    # -------------------------------------------------------------------------
    # Tool 3: grep_code
    # -------------------------------------------------------------------------

    def _tool_grep_code(self, tool_call: ToolCall) -> ToolResult:
        args = tool_call.arguments
        pattern_str = args.get("pattern")
        if not pattern_str or not isinstance(pattern_str, str):
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                output="Parameter 'pattern' must be a non-empty string.",
                is_error=True,
            )

        glob_pattern = args.get("glob", "*")
        if not isinstance(glob_pattern, str):
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                output="Parameter 'glob' must be a string.",
                is_error=True,
            )

        # Check for path traversal attempts in glob pattern
        if ".." in glob_pattern or glob_pattern.startswith(("/", "\\")):
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                output=f"Access denied: path traversal not allowed in glob pattern: {glob_pattern}",
                is_error=True,
            )

        case_sensitive = args.get("case_sensitive", True)
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(pattern_str, flags)
        except re.error as exc:
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                output=f"Invalid regex pattern '{pattern_str}': {exc}",
                is_error=True,
            )

        results: list[str] = []
        truncated = False

        for root_dir, dirs, files in os.walk(self.root):
            # Prune protected and ignored directories
            dirs[:] = [
                d for d in dirs
                if d.lower() not in PROTECTED_DIRS and d.lower() not in IGNORED_DIRECTORIES
            ]

            for file_name in sorted(files):
                if file_name.lower() in {name.lower() for name in SECRET_NAMES}:
                    continue
                if not fnmatch.fnmatch(file_name, glob_pattern):
                    continue

                full_path = Path(root_dir) / file_name
                rel_path = full_path.relative_to(self.root).as_posix()

                if _is_binary_file(full_path):
                    continue

                try:
                    with full_path.open("r", encoding="utf-8", errors="replace") as f:
                        for line_num, line in enumerate(f, start=1):
                            if regex.search(line):
                                results.append(f"{rel_path}:{line_num}: {line.rstrip('\r\n')}")
                                if len(results) >= self.max_results:
                                    truncated = True
                                    break
                except OSError:
                    continue

                if truncated:
                    break
            if truncated:
                break

        if not results:
            raw_output = f"No matches found for pattern '{pattern_str}'."
        else:
            if truncated:
                results.append(f"... [truncated: showing first {self.max_results} matches]")
            raw_output = "\n".join(results)

        output, byte_truncated = _truncate_output(raw_output, self.max_output_bytes)
        return ToolResult(
            call_id=tool_call.call_id,
            tool_name=tool_call.tool_name,
            output=output,
            is_error=False,
            truncated=truncated or byte_truncated,
        )

    # -------------------------------------------------------------------------
    # Tool 4: find_files
    # -------------------------------------------------------------------------

    def _tool_find_files(self, tool_call: ToolCall) -> ToolResult:
        args = tool_call.arguments
        pattern = args.get("pattern", "*")
        if not isinstance(pattern, str):
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                output="Parameter 'pattern' must be a string.",
                is_error=True,
            )

        if ".." in pattern or pattern.startswith(("/", "\\")):
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                output=f"Access denied: path traversal not allowed in pattern: {pattern}",
                is_error=True,
            )

        matches: list[str] = []
        truncated = False

        for root_dir, dirs, files in os.walk(self.root):
            # Prune protected and ignored directories
            dirs[:] = [
                d for d in dirs
                if d.lower() not in PROTECTED_DIRS and d.lower() not in IGNORED_DIRECTORIES
            ]

            for file_name in sorted(files):
                if file_name.lower() in {name.lower() for name in SECRET_NAMES}:
                    continue

                full_path = Path(root_dir) / file_name
                rel_path = full_path.relative_to(self.root).as_posix()

                if fnmatch.fnmatch(file_name, pattern) or fnmatch.fnmatch(rel_path, pattern):
                    matches.append(rel_path)
                    if len(matches) >= self.max_results:
                        truncated = True
                        break
            if truncated:
                break

        if not matches:
            raw_output = f"No files found matching pattern '{pattern}'."
        else:
            if truncated:
                matches.append(f"... [truncated: showing first {self.max_results} matches]")
            raw_output = "\n".join(sorted(matches))

        output, byte_truncated = _truncate_output(raw_output, self.max_output_bytes)
        return ToolResult(
            call_id=tool_call.call_id,
            tool_name=tool_call.tool_name,
            output=output,
            is_error=False,
            truncated=truncated or byte_truncated,
        )

    # -------------------------------------------------------------------------
    # Tool 5: run_command_sandbox
    # -------------------------------------------------------------------------

    def _tool_run_command_sandbox(self, tool_call: ToolCall) -> ToolResult:
        args = tool_call.arguments
        cmd_list = args.get("command")
        if not isinstance(cmd_list, (list, tuple)) or not cmd_list:
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                output="Parameter 'command' must be a non-empty list of string tokens.",
                is_error=True,
            )

        if not all(isinstance(token, str) for token in cmd_list):
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                output="All tokens in 'command' parameter must be strings.",
                is_error=True,
            )

        cmd_tuple = tuple(str(token) for token in cmd_list)
        try:
            self.runner.validate(cmd_tuple)
        except (UnsafeCommandError, PermissionError) as exc:
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                output=f"Security violation: {exc}",
                is_error=True,
            )

        spec = CommandSpec(name=" ".join(cmd_tuple), command=cmd_tuple)
        result = self.runner.run(spec)

        if result.exit_code != 0:
            raw_output = f"Command failed with exit code {result.exit_code}.\n"
            if result.stdout:
                raw_output += f"STDOUT:\n{result.stdout}\n"
            if result.stderr:
                raw_output += f"STDERR:\n{result.stderr}"
            is_error = True
        else:
            raw_output = result.stdout or result.stderr or "Command executed successfully with no output."
            is_error = False

        output, truncated = _truncate_output(raw_output, self.max_output_bytes)
        return ToolResult(
            call_id=tool_call.call_id,
            tool_name=tool_call.tool_name,
            output=output,
            is_error=is_error,
            truncated=truncated,
        )
