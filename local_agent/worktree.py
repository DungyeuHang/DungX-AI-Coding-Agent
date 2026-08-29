from __future__ import annotations

import logging
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from .git import GitIntegration
from .models import WorktreeSession

LOGGER = logging.getLogger(__name__)


def _sanitize_id(identifier: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_-]", "_", str(identifier).strip())
    return clean or "default"


class WorktreeManager:
    """
    Manages Git worktree allocations, branch creation, lifecycle, and cleanup
    for bounded parallel DAG execution.
    """

    def __init__(self, repo_root: str | Path, git: GitIntegration | None = None):
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.worktrees_root = (self.repo_root / ".agent_worktrees").resolve()
        self.git = git or GitIntegration(self.repo_root)
        self._active_sessions: dict[str, WorktreeSession] = {}

    def allocate_worktree_path(self, task_id: str, subtask_id: str) -> Path:
        """
        Allocates a deterministic, compact worktree path inside .agent_worktrees.
        Guards against directory traversal.
        """
        if ".." in str(task_id) or ".." in str(subtask_id) or "/" in str(task_id) or "\\" in str(task_id) or "/" in str(subtask_id) or "\\" in str(subtask_id):
            raise ValueError(f"Invalid worktree path escape attempted: {task_id}, {subtask_id}")
        clean_task = _sanitize_id(task_id)
        clean_subtask = _sanitize_id(subtask_id)
        path = (self.worktrees_root / f"task-{clean_task}" / f"sub-{clean_subtask}").resolve()

        # Security check: Ensure target path is strictly within self.worktrees_root
        try:
            path.relative_to(self.worktrees_root)
        except ValueError as exc:
            raise ValueError(f"Invalid worktree path escape attempted: {path}") from exc

        return path

    def branch_name_for_subtask(self, task_id: str, subtask_id: str) -> str:
        clean_task = _sanitize_id(task_id)
        clean_subtask = _sanitize_id(subtask_id)
        return f"agent/{clean_task}/{clean_subtask}"

    def create_worktree(
        self,
        task_id: str,
        subtask_id: str,
        base_commit: str = "HEAD",
    ) -> WorktreeSession:
        worktree_path = self.allocate_worktree_path(task_id, subtask_id)
        branch_name = self.branch_name_for_subtask(task_id, subtask_id)
        session_id = f"wt-{uuid.uuid4().hex[:12]}"

        if worktree_path.exists():
            self.remove_worktree(worktree_path, force=True)

        worktree_path.parent.mkdir(parents=True, exist_ok=True)

        if self.git.is_repository():
            head_sha = self.git.get_head_commit(base_commit) or "HEAD"
            success = self.git.worktree_add(
                worktree_path,
                branch=branch_name,
                start_point=base_commit,
                create_branch=True,
            )
            if not success:
                LOGGER.warning(
                    "git.worktree_add failed for %s. Attempting fallback directory creation.",
                    worktree_path,
                )
                worktree_path.mkdir(parents=True, exist_ok=True)
        else:
            head_sha = "non_git_head"
            worktree_path.mkdir(parents=True, exist_ok=True)

        session = WorktreeSession(
            session_id=session_id,
            subtask_id=subtask_id,
            worktree_path=str(worktree_path),
            branch_name=branch_name,
            base_commit=head_sha,
            status="active",
        )
        self._active_sessions[session_id] = session
        return session

    def remove_worktree(
        self,
        session_or_path: WorktreeSession | Path | str,
        force: bool = True,
    ) -> bool:
        if isinstance(session_or_path, WorktreeSession):
            target_path = Path(session_or_path.worktree_path).resolve()
            session_or_path.status = "cleaned"
            self._active_sessions.pop(session_or_path.session_id, None)
        else:
            target_path = Path(session_or_path).resolve()
            to_remove = [
                s_id
                for s_id, s in self._active_sessions.items()
                if Path(s.worktree_path).resolve() == target_path
            ]
            for s_id in to_remove:
                self._active_sessions[s_id].status = "cleaned"
                self._active_sessions.pop(s_id, None)

        if self.git.is_repository():
            self.git.worktree_remove(target_path, force=force)
            self.git.worktree_prune()

        if target_path.exists():
            for attempt in range(3):
                try:
                    shutil.rmtree(target_path, ignore_errors=True)
                    if not target_path.exists():
                        break
                except OSError:
                    pass
                time.sleep(0.05 * (attempt + 1))

        if self.git.is_repository():
            self.git.worktree_prune()

        return not target_path.exists()

    def prune(self) -> bool:
        if self.git.is_repository():
            return self.git.worktree_prune()
        return True

    def list_active_sessions(self) -> list[WorktreeSession]:
        return list(self._active_sessions.values())

    def cleanup_stale_worktrees(self, active_session_paths: set[str] | None = None) -> int:
        cleaned_count = 0
        active_set = {
            Path(p).resolve().as_posix() for p in (active_session_paths or set())
        }

        if self.git.is_repository():
            raw_worktrees = self.git.worktree_list()
            for entry in raw_worktrees:
                wt_str = entry.get("worktree", "")
                if not wt_str:
                    continue
                wt_path = Path(wt_str).resolve()
                try:
                    wt_path.relative_to(self.worktrees_root)
                    if wt_path.as_posix() not in active_set:
                        LOGGER.info("Cleaning stale Git worktree: %s", wt_path)
                        self.remove_worktree(wt_path, force=True)
                        cleaned_count += 1
                except ValueError:
                    continue

        if self.worktrees_root.exists():
            for task_dir in self.worktrees_root.iterdir():
                if task_dir.is_dir() and not any(task_dir.iterdir()):
                    try:
                        task_dir.rmdir()
                    except OSError:
                        pass

        return cleaned_count

    def cleanup_all(self) -> int:
        """Cleans all allocated worktrees under .agent_worktrees."""
        count = self.cleanup_stale_worktrees(active_session_paths=set())
        if self.worktrees_root.exists():
            for child in list(self.worktrees_root.iterdir()):
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                    count += 1
        return count
