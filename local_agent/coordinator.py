from __future__ import annotations

import concurrent.futures
import datetime
import logging
import threading
import uuid
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
    DAGExecutionStage,
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


class _WorkerScopedStorage:
    """
    Storage facade handed to a worker Orchestrator running inside an isolated
    worktree.

    Two Phase 4.14 invariants depend on this indirection:

    * **No lost updates.** Concurrent workers each run a full
      read-modify-write cycle over the *same* task record. Letting both write
      the shared task file means the slower writer silently erases the faster
      writer's completed subtask. Writes to the worker's own task are therefore
      buffered in memory; the coordinator merges only that worker's subtask
      back into the canonical record, under a lock, once the worker returns.
    * **No premature knowledge promotion.** A worker's findings are
      branch-local evidence until its branch has been merged and the
      integrated tree has passed Tier-2 validation. Knowledge-graph writes are
      buffered here and discarded; ``verify_integration`` performs the
      authoritative promotion after integration succeeds.
    """

    def __init__(self, inner: TaskStorage, task: Task):
        self._inner = inner
        self._task_id = task.task_id
        self._task = task
        self._graph: Any = None

    @property
    def buffered_task(self) -> Task:
        return self._task

    def save_task(self, task: Task) -> None:
        if getattr(task, "task_id", None) == self._task_id:
            self._task = task
            return
        self._inner.save_task(task)

    def load_task(self, task_id: str) -> Task:
        if task_id == self._task_id:
            return self._task
        return self._inner.load_task(task_id)

    def save_knowledge_graph(self, graph: Any) -> None:
        # Branch-local only: deliberately not forwarded to the shared store.
        self._graph = graph

    def load_knowledge_graph(self) -> Any:
        if self._graph is not None:
            return self._graph
        return self._inner.load_knowledge_graph()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


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
        # Serialises merge-back of worker results into the canonical task record.
        self.task_lock = threading.Lock()
        self.git = GitIntegration(base_config.project)
        self.worktree_manager = worktree_manager or WorktreeManager(base_config.project, self.git)
        self.max_workers = max(1, min(getattr(base_config, "max_parallel_subtasks", 1), 4))
        self.impact_analyzer = ChangeImpactAnalyzer(base_config.project)

    def _active_subtasks(self, task: Task) -> list[Subtask]:
        if not task.plan:
            return []
        return getattr(task.plan, "active_subtasks", [
            s for s in task.plan.subtasks
            if s.status not in {SubtaskStatus.SUPERSEDED, SubtaskStatus.PRUNED}
        ])

    def dag_order_keys(self, task: Task) -> dict[str, tuple]:
        """
        Deterministic ordering key per subtask id: (topological depth, declared
        creation timestamp, subtask id).

        Every component is a property of the *plan* - its shape and its task
        identities - so the same DAG always yields the same order regardless of
        how long any worker took. This is the single source of ordering truth
        for both readiness dispatch and branch integration.
        """
        active_subtasks = self._active_subtasks(task)
        active_map = {s.subtask_id: s for s in active_subtasks}
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

        epoch = datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
        keys: dict[str, tuple] = {}
        for sub in active_subtasks:
            depth = _get_depth(sub.subtask_id, set())
            keys[sub.subtask_id] = (depth, sub.created_at or epoch, sub.subtask_id)
        return keys

    def identify_runnable_subtasks(self, task: Task) -> list[Subtask]:
        """
        Identifies all active subtasks whose dependencies are completed in the DAG.
        Returns a deterministically sorted list of runnable subtasks.
        """
        if not task.plan:
            return []

        active_subtasks = self._active_subtasks(task)
        if not active_subtasks:
            return []

        completed_ids = {s.subtask_id for s in active_subtasks if s.status == SubtaskStatus.COMPLETED}
        order_keys = self.dag_order_keys(task)
        epoch = datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
        sorted_subtasks = sorted(
            active_subtasks,
            key=lambda s: order_keys.get(s.subtask_id, (0, epoch, s.subtask_id)),
        )
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

            # Each worker mutates its own private copy of the task record. The
            # shared Task object and the shared task file are never touched by
            # worker threads, so two workers cannot clobber one another.
            worker_task = Task.from_dict(task.to_dict())
            worker_storage = _WorkerScopedStorage(self.storage, worker_task)

            # Instantiate worker orchestrator
            worker_orchestrator = Orchestrator(
                worktree_config,
                worker_storage,
                self.scheduler,
                self.repo_lock,
                self.memory_lock,
            )

            # Execute full lifecycle inside worktree
            report = worker_orchestrator.run(
                task=worker_task,
                subtask_id=subtask.subtask_id,
                progress=progress,
            )

            # Read the worker's own view of its subtask, then merge exactly that
            # subtask back into the canonical record under a lock.
            updated_task = worker_storage.buffered_task
            updated_sub = next(
                (s for s in (updated_task.plan.subtasks if updated_task.plan else []) if s.subtask_id == subtask.subtask_id),
                subtask,
            )
            updated_sub.worktree_session = session
            self._merge_subtask_into_canonical_task(task.task_id, updated_sub)

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

    def _merge_subtask_into_canonical_task(self, task_id: str, updated_sub: Subtask) -> None:
        """
        Copies one worker's subtask result into the stored task record.

        Held under ``task_lock`` so that concurrent workers serialise their
        read-modify-write cycles instead of overwriting each other. Only the
        worker's own subtask is copied - never the whole task - so a worker can
        never revert a sibling's progress.
        """
        with self.task_lock:
            try:
                canonical = self.storage.load_task(task_id)
            except Exception:
                return
            if canonical is None or not getattr(canonical, "plan", None):
                return
            for index, existing in enumerate(canonical.plan.subtasks):
                if existing.subtask_id == updated_sub.subtask_id:
                    canonical.plan.subtasks[index] = updated_sub
                    break
            else:
                return
            canonical.updated_at = datetime.datetime.now(datetime.timezone.utc)
            try:
                self.storage.save_task(canonical)
            except Exception:
                LOGGER.exception("Failed to persist worker result for %s", updated_sub.subtask_id)

    def persist_integration_state(
        self,
        task_id: str,
        merged: list[Subtask],
        failed: list[Subtask],
    ) -> None:
        """
        Durably records the outcome of an integration pass.

        Without this, a merge conflict leaves the subtask marked COMPLETED on
        disk while its branch was never merged - the task would then be reported
        as finished and the work silently lost. Persisting both outcomes is what
        makes resume able to tell integrated work from work still owed.
        """
        by_id = {s.subtask_id: s for s in merged}
        by_id.update({s.subtask_id: s for s in failed})
        if not by_id:
            return
        with self.task_lock:
            try:
                canonical = self.storage.load_task(task_id)
            except Exception:
                return
            if canonical is None or not getattr(canonical, "plan", None):
                return
            changed = False
            for existing in canonical.plan.subtasks:
                source = by_id.get(existing.subtask_id)
                if source is None:
                    continue
                existing.status = source.status
                existing.integration_commit = source.integration_commit
                if source.worktree_session is not None:
                    existing.worktree_session = source.worktree_session
                changed = True
            if changed:
                canonical.updated_at = datetime.datetime.now(datetime.timezone.utc)
                try:
                    self.storage.save_task(canonical)
                except Exception:
                    LOGGER.exception("Failed to persist integration state for task %s", task_id)

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
            # Nothing was branched, so nothing can be integrated. Reporting these
            # as merged would be a false success.
            LOGGER.warning(
                "Refusing to integrate in a non-Git workspace; no branches exist to merge."
            )
            return [], list(completed_subtasks)

        # Move HEAD onto the integration branch, creating it if needed. If this
        # fails we must NOT merge, or subtask branches land on whatever branch
        # the user happened to have checked out.
        if not self.git.ensure_branch(integration_branch):
            LOGGER.error(
                "Could not check out integration branch %s; aborting integration.",
                integration_branch,
            )
            return [], list(completed_subtasks)

        # Deterministic order derived from the DAG, never from worker timing.
        order_keys = self.dag_order_keys(task)
        epoch = datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
        sorted_subtasks = sorted(
            completed_subtasks,
            key=lambda s: order_keys.get(s.subtask_id, (0, epoch, s.subtask_id)),
        )

        for subtask in sorted_subtasks:
            if not subtask.worktree_session or not subtask.worktree_session.branch_name:
                merged.append(subtask)
                continue

            branch = subtask.worktree_session.branch_name
            branch_commit = self.git.get_branch_commit(branch)

            # Check if branch is already merged into HEAD / integration branch
            if branch_commit and self.git.is_ancestor(branch_commit, "HEAD"):
                subtask.integration_commit = self.git.get_head_commit()
                if subtask.worktree_session:
                    subtask.worktree_session.status = "merged"
                merged.append(subtask)
                LOGGER.info("Subtask branch %s is already an ancestor of %s; skipping duplicate merge.", branch, integration_branch)
                continue

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

    def create_dag_checkpoint(
        self,
        task: Task,
        stage: DAGExecutionStage | str,
        description: str,
        *,
        subtask_id: str = "",
        integration_branch: str | None = None,
        extra_context: dict[str, Any] | None = None,
    ) -> Checkpoint:
        """
        Persists a durable, versioned DAG execution checkpoint at a critical state boundary.
        """
        checkpoint_id = f"cp-dag-{uuid.uuid4().hex[:12]}"
        now = datetime.datetime.now(datetime.timezone.utc)
        stage_str = stage.value if isinstance(stage, DAGExecutionStage) else str(stage)

        active_wts = []
        subtask_states = {}
        integrated_subs = []
        if task.plan:
            for s in task.plan.subtasks:
                subtask_states[s.subtask_id] = s.status.value if isinstance(s.status, SubtaskStatus) else str(s.status)
                if s.worktree_session and s.worktree_session.status == "active":
                    active_wts.append(s.worktree_session.to_dict())
                if getattr(s, "integration_commit", None):
                    integrated_subs.append(s.subtask_id)

        branch = integration_branch or getattr(task, "integration_branch", None) or f"agent-task/{task.task_id}"
        head_commit = self.git.get_head_commit() if self.git.is_repository() else ""

        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            task_id=task.task_id,
            subtask_id=subtask_id or (task.plan.subtasks[0].subtask_id if task.plan and task.plan.subtasks else ""),
            timestamp=now,
            current_state_description=description,
            files_changed=[],
            repository_diff="",
            validation_state={},
            last_provider_result=None,
            next_recommended_action=f"stage:{stage_str}",
            continuation_context=extra_context or {},
            active_worktrees=active_wts,
            integration_branch=branch,
            schema_version="4.15.0",
            dag_stage=stage_str,
            subtask_states=subtask_states,
            integrated_subtasks=integrated_subs,
            verified_subtasks=list(getattr(task, "verified_subtasks", []) or []),
            promoted_subtasks=list(getattr(task, "promoted_subtasks", []) or []),
            cleaned_worktrees=list(getattr(task, "cleaned_worktrees", []) or []),
            base_commit=head_commit,
            integration_commit=head_commit if stage_str in {"integrated", "tier2_verified", "completed"} else None,
        )

        self.storage.save_checkpoint(checkpoint)
        with self.task_lock:
            try:
                canonical = self.storage.load_task(task.task_id)
                canonical.latest_checkpoint_id = checkpoint_id
                self.storage.save_task(canonical)
                task.latest_checkpoint_id = checkpoint_id
            except Exception:
                task.latest_checkpoint_id = checkpoint_id
                self.storage.save_task(task)
        return checkpoint

    def reconcile_dag_state(self, task: Task) -> tuple[Task, dict[str, Any]]:
        """
        Inspects Git reality, branches, worktrees, and commit ancestry to reconcile
        durable task state against disk.
        """
        report: dict[str, Any] = {
            "task_id": task.task_id,
            "reconciled_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "actions": [],
        }

        if not task.plan:
            return task, report

        if not self.git.is_repository():
            # Non-git workspace fallback
            return task, report

        integration_branch = getattr(task, "integration_branch", None) or f"agent-task/{task.task_id}"
        integration_head = self.git.get_branch_commit(integration_branch)

        # 1. Detect and safely abort stray agent merges in the main repo
        if self.git.is_merge_in_progress():
            LOGGER.warning("Active merge in progress detected during DAG reconciliation. Aborting.")
            self.git.merge_abort()
            report["actions"].append("aborted_active_merge")

        # 2. Inspect each subtask
        for subtask in task.plan.subtasks:
            sub_branch = self.worktree_manager.branch_name_for_subtask(task.task_id, subtask.subtask_id)
            branch_commit = self.git.get_branch_commit(sub_branch)
            worktree_path = self.worktree_manager.allocate_worktree_path(task.task_id, subtask.subtask_id)
            base_commit = subtask.worktree_session.base_commit if subtask.worktree_session else (self.git.get_head_commit() or "HEAD")

            # Case A/E/F: Check if branch is already merged into integration_branch
            if integration_head and branch_commit and self.git.is_ancestor(branch_commit, integration_head):
                subtask.status = SubtaskStatus.COMPLETED
                subtask.integration_commit = integration_head
                if subtask.worktree_session:
                    subtask.worktree_session.status = "merged"
                report["actions"].append(f"subtask_{subtask.subtask_id}_already_integrated")
                continue

            # Case C/D: Check if branch exists with a new worker commit ahead of base_commit
            if branch_commit and branch_commit != base_commit:
                commits_ahead = self.git.count_commits_between(base_commit, branch_commit)
                if commits_ahead > 0:
                    subtask.status = SubtaskStatus.COMPLETED
                    if not subtask.worktree_session:
                        subtask.worktree_session = WorktreeSession(
                            session_id=f"wt-recov-{subtask.subtask_id}",
                            subtask_id=subtask.subtask_id,
                            worktree_path=str(worktree_path),
                            branch_name=sub_branch,
                            base_commit=base_commit,
                            status="active",
                        )
                    report["actions"].append(f"subtask_{subtask.subtask_id}_worker_commit_recovered")
                    continue

            # Case C: Check if worktree directory exists and is dirty
            if worktree_path.exists():
                wt_git = GitIntegration(worktree_path)
                if wt_git.is_repository() and wt_git.is_dirty():
                    wt_git.add(["."])
                    commit_msg = f"feat({subtask.subtask_id}): {subtask.title} (recovered)"
                    if wt_git.commit(commit_msg):
                        recovered_sha = wt_git.get_head_commit()
                        subtask.status = SubtaskStatus.COMPLETED
                        if not subtask.worktree_session:
                            subtask.worktree_session = WorktreeSession(
                                session_id=f"wt-recov-{subtask.subtask_id}",
                                subtask_id=subtask.subtask_id,
                                worktree_path=str(worktree_path),
                                branch_name=sub_branch,
                                base_commit=base_commit,
                                status="active",
                            )
                        report["actions"].append(f"subtask_{subtask.subtask_id}_dirty_worktree_committed")
                        continue

                # Case B: Incomplete running worker without commits or changes -> reset to PENDING
                if subtask.status == SubtaskStatus.RUNNING:
                    self.worktree_manager.remove_worktree(worktree_path, force=True)
                    subtask.status = SubtaskStatus.PENDING
                    report["actions"].append(f"subtask_{subtask.subtask_id}_incomplete_worker_reset")
                    continue

            # Case A/B: Stale RUNNING marker without active worktree or branch
            if subtask.status == SubtaskStatus.RUNNING:
                subtask.status = SubtaskStatus.PENDING
                report["actions"].append(f"subtask_{subtask.subtask_id}_stale_running_reset")

        with self.task_lock:
            self.storage.save_task(task)

        return task, report

    def reconcile_and_resume(
        self,
        task: Task,
        progress: Callable[[str], None] | None = None,
    ) -> Task:
        """
        Recovers and resumes a parallel DAG execution from durable checkpoint and Git state.
        Executes ready subtasks concurrently in worktrees, checkpoints at every state boundary,
        integrates branches deterministically, and validates Tier-2 integration.
        """
        def emit(msg: str) -> None:
            if progress:
                progress(msg)
            LOGGER.info("[DAG Resume] %s", msg)

        emit(f"Reconciling DAG execution state for task {task.task_id}...")
        task, report = self.reconcile_dag_state(task)
        integration_branch = getattr(task, "integration_branch", None) or f"agent-task/{task.task_id}"

        if self.git.is_repository() and not self.git.branch_exists(integration_branch):
            self.git.ensure_branch(integration_branch)

        self.create_dag_checkpoint(
            task,
            DAGExecutionStage.READY,
            "Reconciled DAG state from durable Git truth and checkpoints",
            integration_branch=integration_branch,
            extra_context={"reconciliation_report": report},
        )

        # Step 1: Check for any already-completed subtasks that need integration
        completed_unmerged = [
            s for s in (task.plan.subtasks if task.plan else [])
            if s.status == SubtaskStatus.COMPLETED and not getattr(s, "integration_commit", None)
        ]
        if completed_unmerged:
            emit(f"Integrating {len(completed_unmerged)} recovered completed subtask(s)...")
            self.create_dag_checkpoint(task, DAGExecutionStage.INTEGRATING, "Integrating completed subtasks")
            merged, failed = self.integrate_branches(task, completed_unmerged, integration_branch)
            self.persist_integration_state(task.task_id, merged, failed)
            if merged:
                v_ok = self.verify_integration(task, merged)
                self.create_dag_checkpoint(
                    task,
                    DAGExecutionStage.TIER2_VERIFIED if v_ok else DAGExecutionStage.FAILED,
                    f"Tier-2 integration verification {'passed' if v_ok else 'failed'}",
                )

        # Step 2: Execute runnable parallel batches
        task = self.storage.load_task(task.task_id)
        runnable_subtasks = self.identify_runnable_subtasks(task)
        if runnable_subtasks:
            emit(f"Found {len(runnable_subtasks)} ready subtask(s) to execute.")
            predicted_map = self.predict_file_conflicts(runnable_subtasks)
            batches = self.partition_parallel_batches(
                runnable_subtasks,
                predicted_map,
                serialize_overlapping=getattr(self.base_config, "serialize_overlapping_subtasks", True),
            )

            for batch in batches:
                emit(f"Executing batch of {len(batch)} parallel subtask(s)...")
                self.create_dag_checkpoint(
                    task,
                    DAGExecutionStage.RUNNING,
                    f"Executing parallel batch of {len(batch)} subtasks: {[s.subtask_id for s in batch]}",
                )
                results = self.execute_parallel_batch(task, batch, progress=progress)

                # Checkpoint after worker execution
                self.create_dag_checkpoint(
                    task,
                    DAGExecutionStage.WORKER_COMMITTED,
                    f"Completed worker execution for batch: {[s.subtask_id for s, _, _ in results]}",
                )

                completed_in_batch = [
                    s for s, rep, err in results
                    if s.status == SubtaskStatus.COMPLETED or (rep and getattr(rep, "completed", False))
                ]
                if completed_in_batch:
                    self.create_dag_checkpoint(
                        task,
                        DAGExecutionStage.INTEGRATING,
                        f"Integrating batch of {len(completed_in_batch)} completed subtasks",
                    )
                    merged, failed = self.integrate_branches(task, completed_in_batch, integration_branch)
                    if not self.verify_integration(task, merged):
                        emit("Tier-2 integration validation failed; halting further integration.")
                        for subtask in merged:
                            subtask.status = SubtaskStatus.PAUSED
                        self.persist_integration_state(task.task_id, [], merged + failed)
                        self.create_dag_checkpoint(task, DAGExecutionStage.FAILED, "Tier-2 integration verification failed")
                        return self.storage.load_task(task.task_id)
                    self.persist_integration_state(task.task_id, merged, failed)
                    task = self.storage.load_task(task.task_id)
                    self.create_dag_checkpoint(
                        task,
                        DAGExecutionStage.TIER2_VERIFIED,
                        "Tier-2 integration verification passed for batch",
                    )

        # Step 3: Finalize task status
        task = self.storage.load_task(task.task_id)
        if task.plan:
            active_subtasks = self._active_subtasks(task)
            all_done = all(s.status == SubtaskStatus.COMPLETED and getattr(s, "integration_commit", None) for s in active_subtasks)
            if all_done and active_subtasks:
                task.status = TaskStatus.COMPLETED
                with self.task_lock:
                    self.storage.save_task(task)
                self.create_dag_checkpoint(task, DAGExecutionStage.COMPLETED, "DAG execution completed and verified")
            elif any(s.status == SubtaskStatus.FAILED for s in active_subtasks):
                task.status = TaskStatus.FAILED
                with self.task_lock:
                    self.storage.save_task(task)
                self.create_dag_checkpoint(task, DAGExecutionStage.FAILED, "DAG execution failed")

        return self.storage.load_task(task.task_id)

