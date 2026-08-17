from __future__ import annotations

import re
from pathlib import Path


class PatchApplicationError(ValueError):
    """Raised when an AI-generated unified patch cannot be applied exactly."""


_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class UnifiedPatchApplier:
    """Small, strict unified-diff applier with no shell or external dependency."""

    def apply(self, original: str, patch: str, expected_path: str | None = None) -> str:
        if not isinstance(patch, str) or not patch.strip():
            raise PatchApplicationError("patch must be a non-empty string")
        patch_lines = patch.splitlines(keepends=True)
        if expected_path is not None:
            self._validate_headers(patch_lines, expected_path)
        hunks = self._hunks(patch_lines)
        if not hunks:
            raise PatchApplicationError("patch contains no unified-diff hunks")

        original_lines = original.splitlines(keepends=True)
        output: list[str] = []
        cursor = 0
        for start, hunk_lines in hunks:
            target_index = max(start - 1, 0)
            if target_index < cursor or target_index > len(original_lines):
                raise PatchApplicationError("hunk location is outside the target file")
            output.extend(original_lines[cursor:target_index])
            for line in hunk_lines:
                if line.startswith("\\"):
                    continue
                if not line:
                    raise PatchApplicationError("malformed empty hunk line")
                marker, body = line[0], line[1:]
                if marker == " ":
                    self._expect(original_lines, cursor, body, "context")
                    output.append(original_lines[cursor])
                    cursor += 1
                elif marker == "-":
                    self._expect(original_lines, cursor, body, "removal")
                    cursor += 1
                elif marker == "+":
                    output.append(body)
                else:
                    raise PatchApplicationError(f"unsupported patch line marker: {marker!r}")
        output.extend(original_lines[cursor:])
        return "".join(output)

    @staticmethod
    def _validate_headers(lines: list[str], expected_path: str) -> None:
        headers = [line[4:].split("\t", 1)[0].strip() for line in lines if line.startswith("--- ") or line.startswith("+++ ")]
        if not headers:
            return
        normalized_expected = Path(expected_path.replace("\\", "/")).as_posix()
        for header in headers:
            if header == "/dev/null":
                continue
            normalized = header[2:] if header.startswith(("a/", "b/")) else header
            if Path(normalized).as_posix() != normalized_expected:
                raise PatchApplicationError(f"patch header targets {normalized!r}, expected {normalized_expected!r}")

    @staticmethod
    def _hunks(lines: list[str]) -> list[tuple[int, list[str]]]:
        hunks: list[tuple[int, list[str]]] = []
        index = 0
        while index < len(lines) and not lines[index].startswith("@@ "):
            index += 1
        while index < len(lines):
            match = _HUNK_RE.match(lines[index].rstrip("\r\n"))
            if not match:
                raise PatchApplicationError("malformed unified-diff hunk header")
            start = int(match.group(1))
            index += 1
            hunk: list[str] = []
            while index < len(lines) and not lines[index].startswith("@@ "):
                hunk.append(lines[index])
                index += 1
            if not hunk:
                raise PatchApplicationError("unified-diff hunk is empty")
            hunks.append((start, hunk))
        return hunks

    @staticmethod
    def _expect(lines: list[str], index: int, expected: str, kind: str) -> None:
        if index >= len(lines) or lines[index].rstrip("\r\n") != expected.rstrip("\r\n"):
            actual = lines[index].rstrip("\r\n") if index < len(lines) else "<end of file>"
            raise PatchApplicationError(f"{kind} does not match target at line {index + 1}: {actual!r}")
