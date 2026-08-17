from __future__ import annotations

import subprocess
import difflib
from pathlib import Path


class GitIntegration:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def _run(self, *args: str) -> str:
        try:
            result = subprocess.run(["git", *args], cwd=self.root, capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            return ""
        if result.returncode != 0:
            return ""
        return result.stdout.strip()

    def is_repository(self) -> bool:
        return bool(self._run("rev-parse", "--is-inside-work-tree"))

    def status(self) -> str:
        return self._run("status", "--short", "--branch")

    def diff(self) -> str:
        tracked_diff = self._run("diff", "--no-ext-diff", "--unified=3", "--", ".")
        untracked_diff: list[str] = []
        for relative in self._untracked_paths():
            path = self.root / relative
            candidates = path.rglob("*") if path.is_dir() else [path]
            for candidate in candidates:
                if not candidate.is_file() or ".git" in candidate.relative_to(self.root).parts or candidate.name.lower() in {".env", ".env.local", ".env.production", "credentials.json", "secrets.json", "id_rsa", "id_ed25519"}:
                    continue
                try:
                    content = candidate.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                relative_file = candidate.relative_to(self.root).as_posix()
                untracked_diff.extend(difflib.unified_diff([], content.splitlines(keepends=True), fromfile="/dev/null", tofile=f"b/{relative_file}"))
        return tracked_diff + ("\n" if tracked_diff and untracked_diff else "") + "".join(untracked_diff)

    def _untracked_paths(self) -> list[str]:
        status = self._run("status", "--short")
        paths: list[str] = []
        for line in status.splitlines():
            if line.startswith("?? "):
                paths.append(line[3:].strip().strip('"'))
        return paths

    def branch(self) -> str:
        return self._run("branch", "--show-current")

    def log(self, limit: int = 5) -> str:
        return self._run("log", f"-{max(1, min(limit, 50))}", "--oneline", "--decorate")
