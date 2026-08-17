from __future__ import annotations

import difflib
import os
from pathlib import Path


class SandboxViolation(PermissionError):
    """Raised when an operation would leave the target project."""


class ProtectedPathError(PermissionError):
    """Raised when an operation targets Git metadata or an obvious secret."""


SECRET_NAMES = {
    ".env", ".env.local", ".env.production", "credentials.json", "secrets.json",
    "id_rsa", "id_ed25519", "token.json", ".npmrc", ".pypirc",
}


class ProjectFilesystem:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise ValueError(f"project directory does not exist: {self.root}")

    def resolve(self, path: str | Path) -> Path:
        candidate = Path(path)
        resolved = (self.root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise SandboxViolation(f"path is outside project: {path}") from exc
        return resolved

    def _guard(self, path: str | Path, *, read: bool = False, delete: bool = False) -> Path:
        resolved = self.resolve(path)
        relative_parts = resolved.relative_to(self.root).parts
        if ".git" in relative_parts:
            raise ProtectedPathError(".git is protected")
        if resolved.name.lower() in {name.lower() for name in SECRET_NAMES} and (read or delete):
            raise ProtectedPathError(f"secret-like file is protected: {resolved.name}")
        if delete and resolved == self.root:
            raise ProtectedPathError("cannot delete project root")
        return resolved

    def read_file(self, path: str | Path, encoding: str = "utf-8") -> str:
        return self._guard(path, read=True).read_text(encoding=encoding)

    def write_file(self, path: str | Path, content: str, encoding: str = "utf-8") -> None:
        resolved = self._guard(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding=encoding)

    def create_file(self, path: str | Path, content: str = "", encoding: str = "utf-8") -> None:
        resolved = self._guard(path)
        if resolved.exists():
            raise FileExistsError(str(path))
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding=encoding)

    def delete_file(self, path: str | Path) -> None:
        resolved = self._guard(path, delete=True)
        if resolved.is_dir():
            raise IsADirectoryError(str(path))
        resolved.unlink(missing_ok=False)

    def list_directory(self, path: str | Path = ".") -> list[str]:
        directory = self._guard(path, read=True)
        if not directory.is_dir():
            raise NotADirectoryError(str(path))
        return sorted(item.name for item in directory.iterdir())

    def search_files(self, pattern: str = "*", path: str | Path = ".") -> list[str]:
        base = self._guard(path, read=True)
        if not base.is_dir():
            return []
        results: list[str] = []
        for item in base.rglob(pattern):
            if item.is_file() and ".git" not in item.relative_to(self.root).parts:
                results.append(item.relative_to(self.root).as_posix())
        return sorted(results)

    def file_exists(self, path: str | Path) -> bool:
        return self._guard(path).exists()

    def get_diff(self, before: dict[str, str] | None = None) -> str:
        if before is None:
            try:
                from .git import GitIntegration
                return GitIntegration(self.root).diff()
            except OSError:
                return ""
        diff: list[str] = []
        for relative, old_content in before.items():
            path = self.resolve(relative)
            new_content = path.read_text(encoding="utf-8") if path.exists() else ""
            diff.extend(difflib.unified_diff(
                old_content.splitlines(keepends=True), new_content.splitlines(keepends=True),
                fromfile=f"a/{relative}", tofile=f"b/{relative}",
            ))
        return "".join(diff)

    def snapshot(self, paths: list[str]) -> dict[str, str]:
        result: dict[str, str] = {}
        for path in paths:
            try:
                result[path] = self.read_file(path)
            except (FileNotFoundError, UnicodeDecodeError, ProtectedPathError):
                continue
        return result
