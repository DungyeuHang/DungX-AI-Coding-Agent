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

    def file_change_counts(self, limit: int = 200, max_files: int = 2000) -> dict[str, int]:
        """Repo-relative path -> number of recent commits that touched it.

        Additive helper for Phase 4.21's churn signal. Deliberately bounded on
        both axes (commits inspected, distinct paths returned) so that running
        it against a repository with a hundred thousand files cannot turn a
        maintenance scan into an unbounded memory event. Returns ``{}`` on any
        git failure - churn is an *enrichment* signal, and a repository without
        usable history should simply produce fewer maintenance candidates
        rather than fail the scan.
        """
        limit = max(1, min(int(limit), 1000))
        output = self._run("log", f"-{limit}", "--name-only", "--pretty=format:", "--no-merges")
        if not output:
            return {}
        counts: dict[str, int] = {}
        max_files = max(1, int(max_files))
        for line in output.splitlines():
            relative = line.strip().strip('"')
            if not relative:
                continue
            if relative not in counts and len(counts) >= max_files:
                continue
            counts[relative] = counts.get(relative, 0) + 1
        return counts

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

    def get_head_commit(self, ref: str = "HEAD") -> str:
        """Returns the full commit SHA of the specified ref."""
        return self._run("rev-parse", ref)

    def checkout(self, target: str) -> bool:
        """Checks out a branch, tag, or commit."""
        return self._run_for_exit_code("checkout", target) == 0

    def branch_exists(self, branch_name: str) -> bool:
        """Returns True when the named local branch resolves to a commit."""
        if not branch_name:
            return False
        return self._run_for_exit_code(
            "rev-parse", "--verify", "--quiet", f"refs/heads/{branch_name}"
        ) == 0

    def ensure_branch(self, branch_name: str, start_point: str = "HEAD") -> bool:
        """
        Checks out ``branch_name``, creating it from ``start_point`` when it does
        not yet exist. Returns True only when HEAD ends up on that branch.

        Phase 4.14 integration depends on this: subtask branches must be merged
        into a dedicated integration branch, never into whichever branch the
        user happened to have checked out.
        """
        if not branch_name:
            return False
        if self.branch_exists(branch_name):
            return self.checkout(branch_name)
        return self._run_for_exit_code("checkout", "-b", branch_name, start_point) == 0

    def get_remote_url(self, remote_name: str) -> str | None:
        return self._run("config", "--get", f"remote.{remote_name}.url")

    def worktree_add(
        self,
        path: Path | str,
        branch: str,
        start_point: str = "HEAD",
        create_branch: bool = True,
    ) -> bool:
        """
        Creates a new Git worktree at path with the specified branch.
        If create_branch is True, creates a new branch (-b branch start_point).
        """
        target_path = Path(path).resolve()
        # Ensure parent directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if create_branch:
            code = self._run_for_exit_code("worktree", "add", "-b", branch, str(target_path), start_point)
            if code != 0:
                # If branch already exists, attempt adding worktree tracking existing branch
                code = self._run_for_exit_code("worktree", "add", str(target_path), branch)
        else:
            code = self._run_for_exit_code("worktree", "add", str(target_path), branch)
        return code == 0

    def worktree_remove(self, path: Path | str, force: bool = True) -> bool:
        """Removes a Git worktree at the specified path."""
        target_path = Path(path).resolve()
        args = ["worktree", "remove"]
        if force:
            args.append("--force")
        args.append(str(target_path))
        return self._run_for_exit_code(*args) == 0

    def worktree_prune(self) -> bool:
        """Prunes stale Git worktree metadata."""
        return self._run_for_exit_code("worktree", "prune") == 0

    def worktree_list(self) -> list[dict[str, str]]:
        """
        Lists all worktrees using porcelain format.
        Returns a list of dicts with keys: worktree, HEAD, branch, bare, locked, etc.
        """
        output = self._run("worktree", "list", "--porcelain")
        if not output:
            return []
        worktrees: list[dict[str, str]] = []
        current: dict[str, str] = {}
        for line in output.splitlines():
            line = line.strip()
            if not line:
                if current and "worktree" in current:
                    worktrees.append(current)
                    current = {}
                continue
            if " " in line:
                key, val = line.split(" ", 1)
                current[key] = val.strip()
            else:
                current[line] = "true"
        if current and "worktree" in current:
            worktrees.append(current)
        return worktrees

    def merge_branch(self, branch: str, message: str = "") -> tuple[bool, str]:
        """
        Merges the specified branch into the current HEAD using --no-ff.
        Returns (success, output_or_error).
        """
        args = ["git", "merge", "--no-ff", branch]
        if message:
            args.extend(["-m", message])
        try:
            result = subprocess.run(
                args, cwd=self.root, capture_output=True, text=True, timeout=60
            )
            out = (result.stdout + "\n" + result.stderr).strip()
            return result.returncode == 0, out
        except (OSError, subprocess.TimeoutExpired) as e:
            return False, str(e)

    def merge_abort(self) -> bool:
        """Aborts an in-progress merge conflict."""
        return self._run_for_exit_code("merge", "--abort") == 0

    def rebase_branch(self, upstream: str) -> tuple[bool, str]:
        """Rebases current HEAD onto the specified upstream ref."""
        try:
            result = subprocess.run(
                ["git", "rebase", upstream],
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=60,
            )
            out = (result.stdout + "\n" + result.stderr).strip()
            return result.returncode == 0, out
        except (OSError, subprocess.TimeoutExpired) as e:
            return False, str(e)

    def is_ancestor(self, ancestor_ref: str, descendant_ref: str) -> bool:
        """Returns True if ancestor_ref is an ancestor of descendant_ref in git commit graph."""
        if not ancestor_ref or not descendant_ref:
            return False
        return self._run_for_exit_code("merge-base", "--is-ancestor", ancestor_ref, descendant_ref) == 0

    def is_merge_in_progress(self) -> bool:
        """Checks if a git merge is currently in progress (e.g. MERGE_HEAD exists)."""
        merge_head = self.root / ".git" / "MERGE_HEAD"
        if merge_head.exists():
            return True
        return self._run_for_exit_code("rev-parse", "-q", "--verify", "MERGE_HEAD") == 0

    def get_branch_commit(self, branch_name: str) -> str | None:
        """Returns the commit SHA of the named local branch, or None if not found."""
        if not branch_name:
            return None
        sha = self._run("rev-parse", "--verify", "--quiet", f"refs/heads/{branch_name}")
        return sha if sha else None

    def count_commits_between(self, base_ref: str, target_ref: str) -> int:
        """Counts the number of commits in target_ref that are not in base_ref."""
        if not base_ref or not target_ref:
            return 0
        out = self._run("rev-list", "--count", f"{base_ref}..{target_ref}")
        try:
            return int(out.strip()) if out else 0
        except ValueError:
            return 0

