from __future__ import annotations

import concurrent.futures
import datetime
import logging
import threading
from pathlib import Path
from typing import Any, Callable

from .approval import ApprovalPolicyEngine
from .config import AgentConfig
from .git import GitIntegration
from .impact import ChangeImpactAnalyzer
from .knowledge import KnowledgeGraphManager
from .models import (
    ApprovalPolicy,
    Checkpoint,
    ProjectContext,
    ProjectMemory,
    RunReport,
    SpecialistRole,
    Subtask,
    SubtaskContract,
    SubtaskStatus,
    Task,
    TaskPlan,
    TaskStatus,
    ValidationPlan,
    WorktreeSession,
)
from .orchestrator import Orchestrator
from .providers import build_provider
from .repository import RepositoryIntelligence
from .storage import TaskStorage
from .worktree import WorktreeManager

LOGGER = logging.getLogger(__name__)


class ParallelExecutionCoordinator:
    """
    Coordinates bounded parallel DAG subtask execution in isolated Git worktrees,
    file-conflict prediction, deterministic branch merging, Tier-2 integration validation,
    and Phase 4.13 Knowledge Graph synchronization.
    """

    def __init__(
        self,
        base_config: AgentConfig,
        storage: TaskStorage,
        scheduler: Any = None,
        repo_lock: threading.Lock | None = None,
        memory_lock: threading.Lock | None = None,
        worktree_manager: WorktreeManager | None = None,
    ):
        self.base_config = base_config
        self.storage = storage
        self.scheduler = scheduler
        self.repo_lock = repo_lock or threading.Lock()
        self.memory_lock = memory_lock or threading.Lock()
        self.git = GitIntegration(base_config.project)
        self.worktree_manager = worktree_manager or WorktreeManager(base_config.project, self.git)
        self.max_workers = max(1, min(getattr(base_config, "max_parallel_subtasks", 1), 4))
        self.impact_analyzer = ChangeImpactAnalyzer(base_config.project)

    def identify_runnable_subtasks(self, task: Task) -> list[Subtask]:
        """
        Identifies all active subtasks whose dependencies are completed in the DAG.
        Returns a deterministically sorted list of runnable subtasks.
        """
        if not task.plan:
            return []

        active_subtasks = getattr(task.plan, "active_subtasks", [
            s for s in task.plan.subtasks
            if s.status not in {SubtaskStatus.SUPERSEDED, SubtaskStatus.PRUNED}
        ])
        if not active_subtasks:
            return []

        completed_ids = {s.subtask_id for s in active_subtasks if s.status == SubtaskStatus.COMPLETED}
        active_map = {s.subtask_id: s for s in active_subtasks}

        # Calculate topological depth for deterministic tie-breaking
        depth_memo: dict[str, int] = {}

        def _get_depth(sub_id: str, visiting: set[str]) -> int:
            if sub_id in depth_memo:
                return depth_memo[sub_id]
            if sub_id not in active_map or sub_id in visiting:
                return 0
            visiting.add(sub_id)
            sub = active_map[sub_id]
            if not sub.dependencies:
                depth = 0
            else:
                depth = 1 + max((_get_depth(d, visiting) for d in sub.dependencies if d in active_map), default=0)
            visiting.remove(sub_id)
            depth_memo[sub_id] = depth
            return depth

        for sub in active_subtasks:
            _get_depth(sub.subtask_id, set())

        def _sort_key(s: Subtask):
            created_ts = s.created_at or datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
            return (depth_memo.get(s.subtask_id, 0), created_ts, s.subtask_id)

        sorted_subtasks = sorted(active_subtasks, key=_sort_key)
        runnable: list[Subtask] = []
        for subtask in sorted_subtasks:
            if subtask.status in {SubtaskStatus.PENDING, SubtaskStatus.PAUSED}:
                if all(dep_id in completed_ids for dep_id in subtask.dependencies):
                    runnable.append(subtask)

        return runnable

    def predict_file_conflicts(
        self, subtasks: list[Subtask], project_context: ProjectContext | None = None
    ) -> dict[str, set[str]]:
        """
        Conservatively predicts the set of files each subtask is expected to touch.
        """
        ctx = project_context or ProjectContext(root=str(self.base_config.project))
        predicted_map: dict[str, set[str]] = {}

        for subtask in subtasks:
            predicted_files: set[str] = set()
            # 1. Check title and goal with ChangeImpactAnalyzer
            text_query = f"{subtask.title} {subtask.goal} {' '.join(subtask.acceptance_criteria)}"
            try:
                impact = self.impact_analyzer.analyze(text_query, ctx)
                for target in impact.targets:
                    predicted_files.add(Path(target.path).as_posix())
            except Exception:
                pass

            # 2. Check explicitly named source files in acceptance criteria
            for criterion in subtask.acceptance_criteria:
                for word in criterion.split():
                    if ("." in word or "/" in word or "\\" in word) and not word.startswith("http"):
                        cleaned = word.strip("`'\"(),;:")
                        if cleaned.endswith((".py", ".js", ".ts", ".jsx", ".tsx", ".json", ".md", ".toml", ".yaml")):
                            predicted_files.add(Path(cleaned).as_posix())

            # Fallback: if no files predicted, assign a dummy generic file to be safe
            if not predicted_files:
                predicted_files.add(f"__subtask_{subtask.subtask_id}__")

            predicted_map[subtask.subtask_id] = predicted_files

        return predicted_map

    def partition_parallel_batches(
        self,
        subtasks: list[Subtask],
        predicted_files_map: dict[str, set[str]],
        serialize_overlapping: bool = True,
    ) -> list[list[Subtask]]:
        """
        Partitions runnable subtasks into execution batches with deterministic ordering.
        If serialize_overlapping is True, subtasks in the same batch must have disjoint predicted files.
        """
        if not subtasks:
            return []

        if not serialize_overlapping:
            # Batch strictly by max_workers capacity
            batches: list[list[Subtask]] = []
            for i in range(0, len(subtasks), self.max_workers):
                batches.append(subtasks[i : i + self.max_workers])
            return batches

        batches = []
        remaining = list(subtasks)

        while remaining:
            current_batch: list[Subtask] = []
            batch_files: set[str] = set()
            still_remaining: list[Subtask] = []

            for subtask in remaining:
                if len(current_batch) >= self.max_workers:
                    still_remaining.append(subtask)
                    continue

                subtask_files = predicted_files_map.get(subtask.subtask_id, set())
                if not (subtask_files & batch_files):
                    current_batch.append(subtask)
                    batch_files.update(subtask_files)
                else:
                    still_remaining.append(subtask)

            batches.append(current_batch)
            remaining = still_remaining

        return batches

    def execute_subtask_in_worktree(
        self,
        task: Task,
        subtask: Subtask,
        base_commit: str = "HEAD",
        progress: Callable[[str], None] | None = None,
    ) -> tuple[Subtask, RunReport | None, Exception | None]:
        """
        Executes a single subtask inside its isolated Git worktree.
        """
        session: WorktreeSession | None = None
        try:
            # Allocate worktree
            session = self.worktree_manager.create_worktree(
                task.task_id, subtask.subtask_id, base_commit=base_commit
            )
            subtask.worktree_session = session
            subtask.status = SubtaskStatus.RUNNING
            subtask.started_at = datetime.datetime.now(datetime.timezone.utc)

            # Build worktree configuration pointing to the isolated worktree path
            worktree_config = self.base_config.from_environment(
                project=session.worktree_path,
                provider=self.base_config.provider,
                model=self.base_config.model,
                parallel_worktree_execution=False,
                knowledge_graph_enabled=self.base_config.knowledge_graph_enabled,
            )

            # Instantiate worker orchestrator
            worker_orchestrator = Orchestrator(
                worktree_config,
                self.storage,
                self.scheduler,
                self.repo_lock,
                self.memory_lock,
            )

            # Execute full lifecycle inside worktree
            report = worker_orchestrator.run(
                task=task,
                subtask_id=subtask.subtask_id,
                progress=progress,
            )

            # Check if subtask completed successfully in the worker
            updated_task = self.storage.load_task(task.task_id)
            updated_sub = next(
                (s for s in (updated_task.plan.subtasks if updated_task.plan else []) if s.subtask_id == subtask.subtask_id),
                subtask,
            )

            # Commit worktree changes to subtask branch if modified
            worktree_git = GitIntegration(session.worktree_path)
            if worktree_git.is_repository() and worktree_git.is_dirty():
                worktree_git.add(["."])
                commit_msg = f"feat({subtask.subtask_id}): {subtask.title}"
                worktree_git.commit(commit_msg)

            return updated_sub, report, None

        except Exception as exc:
            LOGGER.exception("Worker execution failed for subtask %s: %s", subtask.subtask_id, exc)
            subtask.status = SubtaskStatus.FAILED
            return subtask, None, exc

    def execute_parallel_batch(
        self,
        task: Task,
        batch: list[Subtask],
        base_commit: str = "HEAD",
        progress: Callable[[str], None] | None = None,
    ) -> list[tuple[Subtask, RunReport | None, Exception | None]]:
        """
        Executes a batch of independent subtasks concurrently via ThreadPoolExecutor.
        """
        if not batch:
            return []

        if len(batch) == 1:
            return [self.execute_subtask_in_worktree(task, batch[0], base_commit=base_commit, progress=progress)]

        results: list[tuple[Subtask, RunReport | None, Exception | None]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(batch), self.max_workers)) as executor:
            future_to_subtask = {
                executor.submit(
                    self.execute_subtask_in_worktree,
                    task,
                    subtask,
                    base_commit,
                    progress,
                ): subtask
                for subtask in batch
            }
            for future in concurrent.futures.as_completed(future_to_subtask):
                sub = future_to_subtask[future]
                try:
                    res = future.result()
                    results.append(res)
                except Exception as exc:
                    results.append((sub, None, exc))

        return results

    def integrate_branches(
        self,
        task: Task,
        completed_subtasks: list[Subtask],
        integration_branch: str,
    ) -> tuple[list[Subtask], list[Subtask]]:
        """
        Deterministically integrates verified subtask branches into the integration branch.
        Returns (successfully_merged_subtasks, failed_subtasks).
        """
        if not completed_subtasks:
            return [], []

        merged: list[Subtask] = []
        failed: list[Subtask] = []

        if not self.git.is_repository():
            # In non-git environment, treat as merged directly
            return completed_subtasks, []

        # Checkout integration branch
        self.git.checkout(integration_branch)

        # Deterministic sort: (completed_at, subtask_id)
        def _merge_sort_key(s: Subtask):
            ts = s.completed_at or datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
            return (ts, s.subtask_id)

        sorted_subtasks = sorted(completed_subtasks, key=_merge_sort_key)

        for subtask in sorted_subtasks:
            if not subtask.worktree_session or not subtask.worktree_session.branch_name:
                merged.append(subtask)
                continue

            branch = subtask.worktree_session.branch_name
            merge_msg = f"merge({subtask.subtask_id}): {subtask.title}"
            success, output = self.git.merge_branch(branch, message=merge_msg)

            if success:
                subtask.integration_commit = self.git.get_head_commit()
                if subtask.worktree_session:
                    subtask.worktree_session.status = "merged"
                merged.append(subtask)
                LOGGER.info("Merged subtask branch %s into %s", branch, integration_branch)
            else:
                LOGGER.warning("Merge conflict on branch %s: %s. Attempting abort.", branch, output)
                self.git.merge_abort()
                subtask.status = SubtaskStatus.PAUSED
                failed.append(subtask)

        return merged, failed

    def verify_integration(
        self,
        task: Task,
        merged_subtasks: list[Subtask],
    ) -> bool:
        """
        Performs Tier-2 Integration Validation across the integrated repository tree.
        If validation succeeds, promotes SubtaskContracts to Phase 4.13 Knowledge Graph under memory_lock.
        """
        if not merged_subtasks:
            return True

        # 1. Run primary validation commands on the integrated tree
        validator = Orchestrator(
            self.base_config,
            self.storage,
            self.scheduler,
            self.repo_lock,
            self.memory_lock,
        )
        context = validator.analyzer.scan()

        # If explicit validation commands are configured, run them
        if self.base_config.validation_commands:
            for cmd_str in self.base_config.validation_commands:
                res = validator.runner.run(["python", "-c", "import sys; sys.exit(0)"] if cmd_str == "pass" else cmd_str.split())
                if not res.succeeded:
                    LOGGER.warning("Tier-2 Integration validation failed for command: %s", cmd_str)
                    return False

        # 2. Promote contracts and update authoritative Phase 4.13 Knowledge Graph under memory_lock
        if getattr(self.base_config, "knowledge_graph_enabled", True):
            with self.memory_lock:
                kg_manager = KnowledgeGraphManager(self.storage, self.base_config.project)
                kg_manager.sync_with_scan(context)
                for subtask in merged_subtasks:
                    if subtask.contract is not None:
                        kg_manager.promote_subtask_contract(subtask.contract)
                kg_manager.compact()
                self.storage.save_knowledge_graph(kg_manager.get_graph())

        # 3. Clean up worktrees for merged subtasks
        for subtask in merged_subtasks:
            if subtask.worktree_session:
                self.worktree_manager.remove_worktree(subtask.worktree_session, force=True)

        return True
