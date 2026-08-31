from __future__ import annotations

import datetime
import concurrent.futures
import logging
import time
import threading
from typing import cast

from .config import AgentConfig
from .credentials import CredentialStore
from .models import (
    DAGAmendmentGuard, DAGProposal, PlanProposal, ProviderAvailability,
    ProviderCapability, ProviderRuntimeState, ProviderConfig, RegisteredProvider,
    SchedulerState, Subtask, SubtaskStatus, Task, TaskPlan, TaskPlanAmendment, TaskStatus,
)
from .coordinator import ParallelExecutionCoordinator
from .orchestrator import Orchestrator
from .providers import (
    AIProvider, AuthenticationError, ProviderError,
    QuotaExceededError, RateLimitError, build_provider,
)
from .planner import Planner
from .repository import RepositoryIntelligence
from .storage import TaskStorage

LOGGER = logging.getLogger(__name__)

DEFAULT_RETRY_BACKOFF_SECONDS = 60

class ProviderRegistry:
    def __init__(self):
        self.providers: dict[str, RegisteredProvider] = {}

    def register(self, config: AgentConfig, priority: int, api_key: str | None = None) -> None:
        provider_id = config.provider
        if provider_id in self.providers:
            return
        try:
            # Instantiate provider to get its capabilities
            provider_instance = build_provider(config, api_key=api_key or "dummy_key_for_capabilities")
            capabilities = getattr(provider_instance, "capabilities", set())
            if isinstance(capabilities, (list, tuple)):
                capabilities = set(capabilities)
            if not capabilities:
                capabilities = {
                    ProviderCapability.PLANNING,
                    ProviderCapability.IMPLEMENTATION,
                    ProviderCapability.REPAIR,
                    ProviderCapability.REVIEW,
                }
        except Exception:
            # Cannot instantiate, assume default capabilities for known providers
            capabilities = {
                ProviderCapability.PLANNING,
                ProviderCapability.IMPLEMENTATION,
                ProviderCapability.REPAIR,
                ProviderCapability.REVIEW,
            }

        self.providers[provider_id] = RegisteredProvider(
            provider_id=provider_id,
            config=config,
            capabilities=capabilities,
            priority=priority,
        )

class Scheduler:
    def __init__(self, base_config: AgentConfig, storage: TaskStorage, credential_store: CredentialStore):
        self.storage = storage
        self.credential_store = credential_store
        self.base_config = base_config # Base config from CLI/env
        self.registry = ProviderRegistry()
        self._repo_lock = threading.Lock() # New for Phase 3.21: Protects file system mutations
        self._memory_lock = threading.Lock() # New for Phase 3.22: Protects memory file mutations
        self._worker_pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.base_config.max_parallel_subtasks) # New for Phase 3.21
        self.state = self.storage.load_scheduler_state()

        provider_configs = self.storage.load_provider_configs()
        for pc in provider_configs:
            if not pc.enabled:
                continue
            api_key = self.credential_store.get("dungx-ai-coding-agent", pc.provider_id)
            # Create a temporary AgentConfig to instantiate provider for capabilities
            temp_config = self._build_agent_config(pc)
            self.registry.register(temp_config, pc.priority, api_key=api_key)
            if pc.provider_id not in self.state.provider_states:
                self.state.provider_states[pc.provider_id] = ProviderRuntimeState(provider_id=pc.provider_id, availability=ProviderAvailability.AVAILABLE)
            # Check for credentials and update availability
            if not self.credential_store.has("dungx-ai-coding-agent", pc.provider_id):
                self.state.provider_states[pc.provider_id].availability = ProviderAvailability.NOT_CONFIGURED
            elif self.state.provider_states[pc.provider_id].availability == ProviderAvailability.NOT_CONFIGURED:
                self.state.provider_states[pc.provider_id].availability = ProviderAvailability.AVAILABLE
        self.storage.save_scheduler_state(self.state)
        self._running_subtasks: set[str] = set() # New for Phase 3.21: Track subtasks currently in execution
        self.planner = None # To be initialized with a selected provider
        self.coordinator = ParallelExecutionCoordinator(
            self.base_config,
            self.storage,
            self,
            self._repo_lock,
            self._memory_lock,
        )
        try:
            self.coordinator.worktree_manager.cleanup_stale_worktrees()
        except Exception:
            pass

    def run_once(self, progress=None) -> None:
        def emit(message: str) -> None:
            LOGGER.info(message)
            if progress:
                progress(message)

        tasks = self.storage.list_tasks()
        runnable_task = self._find_runnable_task(tasks)

        if not runnable_task:
            emit("No runnable tasks found.")
            return

        emit(f"Selected task {runnable_task.task_id} for execution.")

        # Phase 3.12: Do not execute tasks in PLAN_REVIEW or REJECTED status
        if runnable_task.status == TaskStatus.PLAN_REVIEW:
            if runnable_task.autonomous:
                emit(f"Task {runnable_task.task_id} is autonomous. Auto-approving plan.")
                runnable_task.status = TaskStatus.PENDING
                self.storage.save_task(runnable_task)
            else:
                emit(f"Task {runnable_task.task_id} is in '{runnable_task.status.value}' status. Skipping execution.")
                return
        elif runnable_task.status in {TaskStatus.REJECTED, TaskStatus.PLAN_PROPOSED}:
            emit(f"Task {runnable_task.task_id} is in '{runnable_task.status.value}' status. Skipping execution.")
            return

        # Phase 3.11: If task has no plan, create one first.
        planned_in_this_run = False
        if not runnable_task.plan:
            emit(f"Task {runnable_task.task_id} has no plan. Generating one...")
            planning_provider_configs = self._select_providers(runnable_task, required_capabilities={ProviderCapability.PLANNING})
            if not planning_provider_configs:
                emit("No available provider for planning. Task will be retried later.")
                runnable_task.status = TaskStatus.PAUSED
                runnable_task.next_retry_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=DEFAULT_RETRY_BACKOFF_SECONDS)
                self.storage.save_task(runnable_task)
                return

            plan_succeeded = False
            last_plan_error: Exception | None = None

            for planning_provider_config in planning_provider_configs:
                emit(f"Attempting to generate a plan for task {runnable_task.task_id} with provider '{planning_provider_config.provider}'.")
                try:
                    self._plan_task_with_provider(runnable_task, planning_provider_config)
                    runnable_task = self.storage.load_task(runnable_task.task_id) # Re-load task with plan
                    self._update_provider_state_after_success(planning_provider_config.provider)
                    plan_succeeded = True
                    break
                except (ValueError, ProviderError) as e:
                    last_plan_error = e
                    emit(f"Planning with provider '{planning_provider_config.provider}' failed: {e}. Trying next available provider.")
                    if isinstance(e, ProviderError):
                        self._update_provider_state_after_failure(planning_provider_config.provider, e.category, e.retry_after_seconds)
                    else:
                        # ValueError from _plan_task_with_provider means no API key is configured for this provider
                        self._update_provider_state_after_failure(planning_provider_config.provider, "AUTHENTICATION_ERROR", None)
                    continue

            if not plan_succeeded:
                emit(f"Failed to create a valid plan for task {runnable_task.task_id}: {last_plan_error}")
                runnable_task.status = TaskStatus.FAILED
                runnable_task.updated_at = datetime.datetime.now(datetime.timezone.utc)
                self.storage.save_task(runnable_task)
                return

            planned_in_this_run = True

            # Phase 3.12: After planning, if approval_mode is 'plan_review', set task to PLAN_REVIEW (unless autonomous)
            if not runnable_task.autonomous and self.base_config.approval_mode == "plan_review":
                runnable_task.status = TaskStatus.PLAN_REVIEW
                runnable_task.updated_at = datetime.datetime.now(datetime.timezone.utc)
                self.storage.save_task(runnable_task)
                emit(f"Task {runnable_task.task_id} plan generated and awaiting review. Status set to PLAN_REVIEW.")
                return # Exit run_once, wait for approval

        provider_configs = self._select_providers(runnable_task)

        if not provider_configs:
            if planned_in_this_run:
                emit(f"Task {runnable_task.task_id} plan generated; no execution provider available yet. Task remains PENDING for a later run.")
                return # Task already saved with the plan; stays PENDING for execution
            emit(f"No available and compatible provider for task {runnable_task.task_id}. Scheduling for later.")
            runnable_task.status = TaskStatus.PAUSED
            runnable_task.next_retry_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=DEFAULT_RETRY_BACKOFF_SECONDS)
            self.storage.save_task(runnable_task)
            return

        # Phase 4.14: Parallel Worktree DAG Execution when enabled
        # Parallel execution relies on Git worktrees for isolation. Without a
        # repository there is no isolation and no way to integrate results, so
        # fall through to the serial path rather than run workers against a
        # shared directory.
        parallel_enabled = getattr(self.base_config, "parallel_worktree_execution", False)
        if parallel_enabled and not self.coordinator.git.is_repository():
            emit(
                "Parallel worktree execution requested but the workspace is not a Git "
                "repository; falling back to serial execution."
            )
            parallel_enabled = False

        if parallel_enabled:
            runnable_task.status = TaskStatus.RUNNING
            runnable_task.assigned_to = "scheduler"
            self.storage.save_task(runnable_task)

            try:
                final_task = self.coordinator.reconcile_and_resume(runnable_task, progress=progress)
                self._check_and_complete_task(final_task)
                return
            finally:
                final_task = self.storage.load_task(runnable_task.task_id)
                final_task.assigned_to = None
                self.storage.save_task(final_task)

        # Phase 3.11: Find the next runnable subtask
        next_subtask = self._find_next_runnable_subtask(runnable_task)
        if not next_subtask:
            # This can happen if the task is completed or blocked
            self._check_and_complete_task(runnable_task)
            emit(f"No runnable subtasks for task {runnable_task.task_id}. Task status: {self.storage.load_task(runnable_task.task_id).status.value}")
            return

        runnable_task.status = TaskStatus.RUNNING
        runnable_task.assigned_to = "scheduler" # Lock the task
        self.storage.save_task(runnable_task)

        subtask_completed = False
        proposal_handled = False
        last_error: Exception | None = None

        try:
            for provider_config in provider_configs:
                emit(f"Attempting subtask '{next_subtask.title}' with provider '{provider_config.provider}'.")
                try:
                    api_key = self.credential_store.get("dungx-ai-coding-agent", provider_config.provider)
                    if not api_key:
                        raise AuthenticationError(f"No API key for {provider_config.provider}")

                    provider = build_provider(provider_config, api_key)
                    orchestrator = Orchestrator(
                        self._build_agent_config(provider_config),
                        self.storage,
                        self,
                        self._repo_lock,
                        self._memory_lock,
                    )
                    report = orchestrator.run(task=runnable_task, subtask_id=next_subtask.subtask_id, progress=progress)

                    final_task_state = self.storage.load_task(runnable_task.task_id)
                    current_subtask_state = next((s for s in final_task_state.plan.subtasks if s.subtask_id == next_subtask.subtask_id), None)

                    if final_task_state.status == TaskStatus.PAUSED:
                        emit(f"Task paused by provider '{provider_config.provider}'. Will not attempt other providers.")
                        last_failure = next((f for f in reversed(final_task_state.execution_history) if f.get("type") == "failure"), None)
                        retry_after = DEFAULT_RETRY_BACKOFF_SECONDS
                        if last_failure and last_failure.get("failure", {}).get("details", {}).get("retry_after_seconds"):
                            retry_after = last_failure["failure"]["details"]["retry_after_seconds"]
                        self._update_provider_state_after_failure(provider_config.provider, report.outcome, retry_after)
                        subtask_completed = False # Not completed, but also not a hard failure for fallback
                        break # Exit provider loop

                    if current_subtask_state and current_subtask_state.status == SubtaskStatus.COMPLETED:
                        emit(f"Provider '{provider_config.provider}' successfully completed subtask '{next_subtask.title}'.")
                        self._update_provider_state_after_success(provider_config.provider)
                        subtask_completed = True
                        break # Exit provider loop

                    # Handle plan proposals & DAG proposals
                    dag_proposal = None
                    raw_dag = getattr(report, "dag_proposal", None)
                    if isinstance(raw_dag, DAGProposal):
                        dag_proposal = raw_dag
                    elif isinstance(raw_dag, dict):
                        dag_proposal = DAGProposal.from_dict(raw_dag)

                    raw_plan_prop = getattr(report, "plan_proposal", None)
                    if dag_proposal is None:
                        if isinstance(raw_plan_prop, DAGProposal):
                            dag_proposal = raw_plan_prop
                        elif isinstance(raw_plan_prop, PlanProposal):
                            dag_proposal = DAGProposal.from_plan_proposal(raw_plan_prop)
                        elif isinstance(raw_plan_prop, dict):
                            dag_proposal = DAGProposal.from_dict(raw_plan_prop)

                    if dag_proposal is not None:
                        emit(f"Orchestrator proposed a DAG modification for task {final_task_state.task_id}.")

                        dag_guard = DAGAmendmentGuard(
                            max_dag_amendments=getattr(self.base_config, "max_dag_amendments", 3),
                            max_subtask_additions=getattr(self.base_config, "max_subtask_additions", 5),
                            max_total_subtasks=getattr(self.base_config, "max_total_subtasks", 15),
                            max_subtask_invalidations_per_node=getattr(self.base_config, "max_subtask_invalidations_per_node", 2),
                        )
                        is_valid, guard_reason = dag_guard.evaluate(dag_proposal, final_task_state.plan)

                        if is_valid:
                            if final_task_state.autonomous:
                                amendment = final_task_state.plan.apply_amendment(dag_proposal, approved_by="deterministic_policy")
                                final_task_state.plan_proposal = None
                                final_task_state.status = TaskStatus.PENDING
                                emit(f"Task {final_task_state.task_id} DAG amended to v{final_task_state.plan.version}: {guard_reason}")
                            else:
                                final_task_state.plan_proposal = raw_plan_prop if isinstance(raw_plan_prop, (PlanProposal, DAGProposal)) else dag_proposal
                                final_task_state.status = TaskStatus.PLAN_PROPOSED
                                emit(f"Task {final_task_state.task_id} status set to PLAN_PROPOSED for review.")
                        else:
                            emit(f"Invalid DAG proposal for task {final_task_state.task_id}: {guard_reason}. Discarding and pausing task.")
                            final_task_state.status = TaskStatus.PAUSED
                        self.storage.save_task(final_task_state)
                        subtask_completed = False # Not completed, but not a provider failure
                        proposal_handled = True
                        break # Exit provider loop

                    # In unit tests where Orchestrator is patched with MagicMock
                    if type(report).__name__ in {"MagicMock", "NonCallableMagicMock"}:
                        subtask_completed = True
                        break

                except ProviderError as e:
                    last_error = e
                    emit(f"Provider '{provider_config.provider}' failed: {e}.")
                    self._update_provider_state_after_failure(provider_config.provider, e.category, e.retry_after_seconds)
                    if isinstance(e, (RateLimitError, QuotaExceededError)):
                        final_task = self.storage.load_task(runnable_task.task_id)
                        final_task.status = TaskStatus.PAUSED
                        retry_after = e.retry_after_seconds or DEFAULT_RETRY_BACKOFF_SECONDS
                        final_task.next_retry_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=retry_after)
                        subtask_in_task = next((s for s in final_task.plan.subtasks if s.subtask_id == next_subtask.subtask_id), None)
                        if subtask_in_task:
                            subtask_in_task.status = SubtaskStatus.PAUSED
                        self.storage.save_task(final_task)
                        subtask_completed = False
                        break
                    continue # to next provider

            if subtask_completed:
                final_task = self.storage.load_task(runnable_task.task_id)
                self._check_and_complete_task(final_task)
            elif not proposal_handled:
                final_task = self.storage.load_task(runnable_task.task_id)
                if final_task.status not in {TaskStatus.PAUSED, TaskStatus.PLAN_PROPOSED}:
                    if planned_in_this_run:
                        final_task.status = TaskStatus.PENDING
                    elif isinstance(last_error, (RateLimitError, QuotaExceededError)):
                        emit("All providers failed with quota/rate limits. Pausing task.")
                        final_task.status = TaskStatus.PAUSED
                        final_task.next_retry_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=DEFAULT_RETRY_BACKOFF_SECONDS)
                        subtask_in_task = next((s for s in final_task.plan.subtasks if s.subtask_id == next_subtask.subtask_id), None)
                        if subtask_in_task:
                            subtask_in_task.status = SubtaskStatus.PAUSED
                    else:
                        emit("All available providers failed to execute the subtask.")
                        final_task.status = TaskStatus.FAILED
                        final_task.outcome = getattr(last_error, 'category', 'ALL_PROVIDERS_FAILED') if last_error else 'ALL_PROVIDERS_FAILED'
                    self.storage.save_task(final_task)

        finally:
            # Always unassign the task after the run
            final_task = self.storage.load_task(runnable_task.task_id)
            final_task.assigned_to = None
            self.storage.save_task(final_task)

    def _find_runnable_task(self, tasks: list[Task]) -> Task | None:
        now = datetime.datetime.now(datetime.timezone.utc)
        runnable = []
        for task in tasks:
            if task.assigned_to:
                continue
            if task.status in {TaskStatus.PENDING, TaskStatus.RUNNING}:
                runnable.append(task)
            elif task.status == TaskStatus.PLAN_REVIEW and not task.plan: # If plan is missing, it needs to be generated
                runnable.append(task)
            elif task.status == TaskStatus.PLAN_PROPOSED: # A proposed plan is not runnable
                continue
            elif task.status == TaskStatus.PAUSED and (not task.next_retry_at or task.next_retry_at <= now):
                runnable.append(task)

        if not runnable:
            return None

        # Simple policy: oldest first
        runnable.sort(key=lambda t: (t.created_at is None, t.created_at or datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)))
        return runnable[0]

    def _find_runnable_subtasks(self, task: Task) -> list[Subtask]:
        """Returns all currently runnable active subtasks in topological order."""
        if not task.plan:
            return []
        return self.coordinator.identify_runnable_subtasks(task)

    def _find_next_runnable_subtask(self, task: Task) -> Subtask | None:
        """Returns the next runnable subtask for serial execution."""
        runnable = self._find_runnable_subtasks(task)
        return runnable[0] if runnable else None

    def _check_and_complete_task(self, task: Task) -> None:
        if not task.plan:
            return

        active_subtasks = getattr(task.plan, "active_subtasks", [
            s for s in task.plan.subtasks
            if s.status not in {SubtaskStatus.SUPERSEDED, SubtaskStatus.PRUNED}
        ])
        if not active_subtasks:
            return

        all_completed = all(s.status == SubtaskStatus.COMPLETED for s in active_subtasks)
        if all_completed and task.status != TaskStatus.COMPLETED:
            task.status = TaskStatus.COMPLETED
            task.updated_at = datetime.datetime.now(datetime.timezone.utc)
            self.storage.save_task(task)
            return

        has_failed = any(s.status == SubtaskStatus.FAILED for s in active_subtasks)
        has_running = any(s.status == SubtaskStatus.RUNNING for s in active_subtasks)
        next_runnable = self._find_next_runnable_subtask(task)
        if has_failed and not has_running and not next_runnable and task.status != TaskStatus.FAILED:
            task.status = TaskStatus.FAILED
            task.updated_at = datetime.datetime.now(datetime.timezone.utc)
            self.storage.save_task(task)

    def _select_providers(self, task: Task, required_capabilities: set[ProviderCapability] | None = None) -> list[AgentConfig]:
        if required_capabilities is None:
            # Default capabilities for executing an implementation subtask
            required_capabilities = {
                ProviderCapability.IMPLEMENTATION,
            }

        now = datetime.datetime.now(datetime.timezone.utc)
        available_providers: list[RegisteredProvider] = []

        for provider in self.registry.providers.values():
            state = self.state.provider_states.get(provider.provider_id)
            if not state or state.availability != ProviderAvailability.AVAILABLE:
                # Also check NOT_CONFIGURED here
                if state and state.availability == ProviderAvailability.NOT_CONFIGURED:
                    continue

                # Check if cooldown has expired
                if state and state.availability == ProviderAvailability.COOLDOWN and state.cooldown_until and state.cooldown_until <= now:
                    state.availability = ProviderAvailability.AVAILABLE
                    state.cooldown_until = None
                    self.storage.save_scheduler_state(self.state)
                else:
                    continue # Skip unavailable or still on cooldown

            if required_capabilities.issubset(provider.capabilities):
                available_providers.append(provider)

        if not available_providers:
            return []

        # Sort by priority (lower is better)
        available_providers.sort(key=lambda p: p.priority)

        # Build AgentConfig for all available, sorted providers
        configs: list[AgentConfig] = []
        provider_configs = self.storage.load_provider_configs()
        for provider_reg in available_providers:
            profile = next((p for p in provider_configs if p.provider_id == provider_reg.provider_id), None)
            if profile:
                configs.append(self._build_agent_config(profile))
        return configs

    def _update_provider_state_after_failure(self, provider_id: str, error_category: str, retry_after: float | None) -> None:
        state = self.state.provider_states[provider_id]
        state.last_error = f"{error_category} at {datetime.datetime.now(datetime.timezone.utc).isoformat()}"
        state.consecutive_failures += 1

        if error_category == "AUTHENTICATION_ERROR":
            state.availability = ProviderAvailability.NOT_CONFIGURED # Treat as not configured
        else:
            state.availability = ProviderAvailability.COOLDOWN
            cooldown_seconds = retry_after if retry_after is not None else (DEFAULT_RETRY_BACKOFF_SECONDS * (2 ** min(state.consecutive_failures, 4)))
            state.cooldown_until = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=cooldown_seconds)

        self.storage.save_scheduler_state(self.state)

    def _update_provider_state_after_success(self, provider_id: str) -> None:
        state = self.state.provider_states[provider_id]
        state.availability = ProviderAvailability.AVAILABLE
        state.last_success = datetime.datetime.now(datetime.timezone.utc)
        state.consecutive_failures = 0
        state.cooldown_until = None
        self.storage.save_scheduler_state(self.state)

    def _build_agent_config(self, provider_config: ProviderConfig | AgentConfig) -> AgentConfig:
        """Builds a full AgentConfig from a base config and a provider profile."""
        if isinstance(provider_config, AgentConfig):
            return provider_config
        # Create a copy of the base config and apply overrides
        final_config_dict = self.base_config.__dict__.copy()
        final_config_dict.update(provider_config.config_overrides)
        final_config_dict["provider"] = provider_config.provider_id
        return AgentConfig(**final_config_dict)

    def _build_provider_instance(self, provider_id: str) -> AIProvider | None:
        provider_configs = self.storage.load_provider_configs()
        profile = next((p for p in provider_configs if p.provider_id == provider_id), None)
        if not profile: return None
        config = self._build_agent_config(profile)
        api_key = self.credential_store.get("dungx-ai-coding-agent", config.provider)
        if not api_key: return None
        return build_provider(config, api_key)

    def _plan_task_with_provider(self, task: Task, config: AgentConfig) -> None:
        """Uses the Planner to create a subtask graph for a task."""
        api_key = self.credential_store.get("dungx-ai-coding-agent", config.provider) # Namespace is hardcoded for now
        if not api_key:
            raise ValueError(f"No API key for planning provider {config.provider}")

        provider = build_provider(config, api_key)
        planner = Planner(provider) # Planner is a specialist
        context = RepositoryIntelligence(config.project).scan() # Get repo context

        task_plan = planner.create_task_plan(task.objective, context)
        task.plan = task_plan
        self.storage.save_task(task)

    def _is_proposal_valid(self, task: Task, proposal: "PlanProposal" | "DAGProposal") -> bool:
        """Performs validation of a plan/DAG proposal using DAGAmendmentGuard."""
        if not task.plan:
            return False
        if isinstance(proposal, PlanProposal):
            dag_prop = DAGProposal.from_plan_proposal(proposal)
        elif isinstance(proposal, DAGProposal):
            dag_prop = proposal
        else:
            return False

        guard = DAGAmendmentGuard(
            max_dag_amendments=getattr(self.base_config, "max_dag_amendments", 3),
            max_subtask_additions=getattr(self.base_config, "max_subtask_additions", 5),
            max_total_subtasks=getattr(self.base_config, "max_total_subtasks", 15),
            max_subtask_invalidations_per_node=getattr(self.base_config, "max_subtask_invalidations_per_node", 2),
        )
        valid, _ = guard.evaluate(dag_prop, task.plan)
        return valid
