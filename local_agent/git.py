from __future__ import annotations

import re
import subprocess
import difflib
from pathlib import Path


class GitIntegration:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def _run(self, *args: str) -> str:
        try: # Added timeout for safety
            result = subprocess.run(["git", *args], cwd=self.root, capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            return ""
        if result.returncode != 0:
            return ""
        return result.stdout.strip()

    def _run_for_exit_code(self, *args: str) -> int:
        """Runs a git command and returns the exit code."""
        try:
            result = subprocess.run(
                ["git", *args], cwd=self.root, capture_output=True, text=True, timeout=30
            )
            return result.returncode
        except (OSError, subprocess.TimeoutExpired):
            return -1

    def is_repository(self) -> bool:
        # Use a more reliable check that works even in subdirectories of a git repo
        # and returns a non-zero exit code if not in a repo.
        return self._run("rev-parse", "--git-dir") != ""

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

    def is_dirty(self, expected_changes: list[str] | None = None) -> bool:
        """Checks if the working tree is dirty with unexpected changes."""
        status_output = self._run("status", "--porcelain", "-uall")
        if not status_output:
            return False # Clean

        expected_set = {Path(p).as_posix() for p in expected_changes or []}
        
        for line in status_output.splitlines():
            # Porcelain format: XY PATH
            # For renames: R  ORIG -> NEW
            path_str = line[3:].strip().strip('"')
            if " -> " in path_str:
                path_str = path_str.split(" -> ")[1].strip().strip('"')
            
            path = Path(path_str.strip()).as_posix()
            if path == ".agent_data" or path.startswith(".agent_data/"):
                continue
            if path not in expected_set:
                return True # Found an unexpected change
        
        return False

    def create_branch(self, branch_name: str) -> bool:
        """Creates a new branch from the current HEAD."""
        return self._run_for_exit_code("checkout", "-b", branch_name) == 0

    def add(self, paths: list[str]) -> bool:
        """Stages the specified paths."""
        if not paths:
            return True
        return self._run_for_exit_code("add", "--", *paths) == 0

    def commit(self, message: str) -> bool:
        """Creates a commit with the given message."""
        return self._run_for_exit_code("commit", "-m", message) == 0

    def push(self, remote: str, branch: str, set_upstream: bool = False) -> bool:
        """Pushes the specified branch to the remote."""
        args = ["push", remote, branch]
        if set_upstream:
            args.insert(1, "--set-upstream")
        return self._run_for_exit_code(*args) == 0

    def get_current_branch(self) -> str:
        return self._run("rev-parse", "--abbrev-ref", "HEAD")

    def get_remote_url(self, remote_name: str) -> str | None:
        return self._run("config", "--get", f"remote.{remote_name}.url")
