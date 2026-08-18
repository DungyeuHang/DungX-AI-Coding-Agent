from __future__ import annotations

import datetime
import logging
import time
from typing import cast

from .config import AgentConfig
from .credentials import CredentialStore
from .models import (
    ProviderAvailability, ProviderCapability, ProviderRuntimeState,
    ProviderConfig, RegisteredProvider, SchedulerState, Subtask, SubtaskStatus,
    Task, TaskStatus,
)
from .orchestrator import Orchestrator
from .providers import AIProvider, ProviderError, build_provider
from .planner import Planner
from .storage import TaskStorage

LOGGER = logging.getLogger(__name__)

DEFAULT_RETRY_BACKOFF_SECONDS = 60

class ProviderRegistry:
    def __init__(self):
        self.providers: dict[str, RegisteredProvider] = {}

    def register(self, config: AgentConfig, priority: int) -> None:
        provider_id = config.provider
        if provider_id in self.providers:
            return
        try:
            # Instantiate provider to get its capabilities
            provider_instance = build_provider(config)
            capabilities = provider_instance.capabilities
        except ProviderError:
            # Cannot instantiate, assume no capabilities for now
            capabilities = set()

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
        self.state = self.storage.load_scheduler_state()

        provider_configs = self.storage.load_provider_configs()
        for pc in provider_configs:
            if not pc.enabled:
                continue
            # Create a temporary AgentConfig to instantiate provider for capabilities
            temp_config = self._build_agent_config(pc)
            self.registry.register(temp_config, pc.priority)
            if pc.provider_id not in self.state.provider_states:
                self.state.provider_states[pc.provider_id] = ProviderRuntimeState(provider_id=pc.provider_id, availability=ProviderAvailability.AVAILABLE)
            # Check for credentials and update availability
            if not self.credential_store.has("dungx-ai-coding-agent", pc.provider_id):
                self.state.provider_states[pc.provider_id].availability = ProviderAvailability.NOT_CONFIGURED
        self.storage.save_scheduler_state(self.state)
        self.planner = None # To be initialized with a selected provider

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
        if runnable_task.status in {TaskStatus.PLAN_REVIEW, TaskStatus.REJECTED, TaskStatus.PLAN_PROPOSED}:
            emit(f"Task {runnable_task.task_id} is in {runnable_task.status.value} status. Skipping execution.")
            return

        # Phase 3.11: If task has no plan, create one first.
        if not runnable_task.plan:
            emit(f"Task {runnable_task.task_id} has no plan. Generating one...")
            planning_provider_config = self._select_provider(runnable_task, required_capabilities={ProviderCapability.PLANNING})
            if not planning_provider_config:
                emit("No available provider for planning. Task will be retried later.")
                runnable_task.status = TaskStatus.PAUSED
                runnable_task.next_retry_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=DEFAULT_RETRY_BACKOFF_SECONDS)
                self.storage.save_task(runnable_task)
                return
            
            try:
                self._plan_task(runnable_task, planning_provider_config)
                runnable_task = self.storage.load_task(runnable_task.task_id) # Re-load task with plan
            except (ValueError, ProviderError) as e:
                emit(f"Failed to create a valid plan for task {runnable_task.task_id}: {e}")
                runnable_task.status = TaskStatus.FAILED
                runnable_task.updated_at = datetime.datetime.now(datetime.timezone.utc)
                self.storage.save_task(runnable_task)
                return
            
            # Phase 3.12: After planning, if approval_mode is 'plan_review', set task to PLAN_REVIEW
            if self.base_config.approval_mode == "plan_review":
                runnable_task.status = TaskStatus.PLAN_REVIEW
                runnable_task.updated_at = datetime.datetime.now(datetime.timezone.utc)
                self.storage.save_task(runnable_task)
                emit(f"Task {runnable_task.task_id} plan generated and awaiting review. Status set to PLAN_REVIEW.")
                return # Exit run_once, wait for approval

        provider_config = self._select_provider(runnable_task)

        if not provider_config:
            emit(f"No available and compatible provider for task {runnable_task.task_id}. Scheduling for later.")
            runnable_task.status = TaskStatus.PAUSED
            runnable_task.next_retry_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=DEFAULT_RETRY_BACKOFF_SECONDS)
            self.storage.save_task(runnable_task)
            return

        # Phase 3.11: Find the next runnable subtask
        next_subtask = self._find_next_runnable_subtask(runnable_task)
        if not next_subtask:
            # This can happen if the task is completed or blocked
            self._check_and_complete_task(runnable_task)
            emit(f"No runnable subtasks for task {runnable_task.task_id}. Task status: {self.storage.load_task(runnable_task.task_id).status.value}")
            return

        emit(f"Selected provider '{provider_config.provider}' for task {runnable_task.task_id}.")
        runnable_task.status = TaskStatus.RUNNING
        runnable_task.assigned_to = provider_config.provider
        self.storage.save_task(runnable_task)

        api_key = self.credential_store.get("dungx-ai-coding-agent", provider_config.provider)
        if not api_key:
             # This should not happen if _select_provider is correct, but as a safeguard:
            emit(f"FATAL: No API key found for selected provider '{provider_config.provider}' after selection.")
            self._update_provider_state_after_failure(provider_config.provider, "AUTHENTICATION_ERROR", None)
            return
        try:
            provider = build_provider(provider_config, api_key)
            orchestrator = Orchestrator(provider_config, provider, self.storage) # Orchestrator now takes a subtask
            report = orchestrator.run(task=runnable_task, subtask_id=next_subtask.subtask_id, progress=progress)

            final_task_state = self.storage.load_task(runnable_task.task_id)
            if final_task_state.status == TaskStatus.PAUSED:
                emit(f"Task {final_task_state.task_id} was paused by provider. Updating provider state.")
                last_failure = next((f for f in reversed(final_task_state.execution_history) if f.get("type") == "failure"), None)
                retry_after = DEFAULT_RETRY_BACKOFF_SECONDS
                if last_failure and last_failure.get("failure", {}).get("details", {}).get("retry_after_seconds"):
                    retry_after = last_failure["failure"]["details"]["retry_after_seconds"]
                self._update_provider_state_after_failure(provider_config.provider, report.outcome, retry_after)
            elif final_task_state.status == TaskStatus.COMPLETED:
                emit(f"Task {final_task_state.task_id} completed successfully.")
                self._update_provider_state_after_success(provider_config.provider)
            elif final_task_state.status == TaskStatus.RUNNING: # Subtask finished, but task is not
                emit(f"Subtask {next_subtask.subtask_id} completed. Task continues.")
            else: # FAILED
                # Phase 3.14: Check if the orchestrator proposed a plan modification
                if report.plan_proposal:
                    emit(f"Orchestrator proposed a plan modification for task {final_task_state.task_id}.")
                    # Basic validation of the proposal
                    if self._is_proposal_valid(final_task_state, report.plan_proposal):
                        final_task_state.plan_proposal = report.plan_proposal
                        final_task_state.status = TaskStatus.PLAN_PROPOSED
                        emit(f"Task {final_task_state.task_id} status set to PLAN_PROPOSED for review.")
                    else:
                        emit(f"Invalid plan proposal for task {final_task_state.task_id}. Discarding and pausing task.")
                        final_task_state.status = TaskStatus.PAUSED
                        final_task_state.next_retry_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=DEFAULT_RETRY_BACKOFF_SECONDS)
                    self.storage.save_task(final_task_state)
                    return # Stop here, wait for human review or retry

                emit(f"Task {final_task_state.task_id} failed.")
                self._update_provider_state_after_failure(provider_config.provider, report.outcome, None) # None retry_after for permanent failure

        except Exception as e:
            LOGGER.exception("Scheduler caught unhandled exception during orchestration: %s", e)
            self._update_provider_state_after_failure(provider_config.provider, "SCHEDULER_ERROR", None)
            runnable_task.status = TaskStatus.FAILED
            runnable_task.outcome = "SCHEDULER_ERROR"
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
            if task.status == TaskStatus.PENDING:
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
        runnable.sort(key=lambda t: t.created_at)
        return runnable[0]

    def _find_next_runnable_subtask(self, task: Task) -> Subtask | None:
        if not task.plan:
            return None

        completed_ids = {s.subtask_id for s in task.plan.subtasks if s.status == SubtaskStatus.COMPLETED}
        
        for subtask in sorted(task.plan.subtasks, key=lambda s: s.created_at):
            if subtask.status in {SubtaskStatus.PENDING, SubtaskStatus.PAUSED}:
                if all(dep_id in completed_ids for dep_id in subtask.dependencies):
                    return subtask
        return None

    def _check_and_complete_task(self, task: Task) -> None:
        if not task.plan:
            return

        all_completed = all(s.status == SubtaskStatus.COMPLETED for s in task.plan.subtasks)
        if all_completed and task.status != TaskStatus.COMPLETED:
            task.status = TaskStatus.COMPLETED
            task.updated_at = datetime.datetime.now(datetime.timezone.utc)
            self.storage.save_task(task)

    def _select_provider(self, task: Task, required_capabilities: set[ProviderCapability] | None = None) -> AgentConfig | None:
        if required_capabilities is None:
            # Default capabilities for a standard implementation subtask
            required_capabilities = {
                ProviderCapability.PLANNING, # For sub-plan
                ProviderCapability.IMPLEMENTATION,
                ProviderCapability.REPAIR,
                ProviderCapability.REVIEW,
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
            return None

        # Sort by priority (lower is better)
        available_providers.sort(key=lambda p: p.priority)
        if not available_providers:
            return None
        
        # Now, build the final AgentConfig for the selected provider
        selected_provider_reg = available_providers[0]
        provider_configs = self.storage.load_provider_configs()
        selected_provider_profile = next((p for p in provider_configs if p.provider_id == selected_provider_reg.provider_id), None)
        if not selected_provider_profile:
            return None # Should not happen
        return self._build_agent_config(selected_provider_profile)

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

    def _build_agent_config(self, provider_config: ProviderConfig) -> AgentConfig:
        """Builds a full AgentConfig from a base config and a provider profile."""
        # Create a copy of the base config and apply overrides
        final_config_dict = self.base_config.__dict__.copy()
        final_config_dict.update(provider_config.config_overrides)
        final_config_dict["provider"] = provider_config.provider_id
        return AgentConfig(**final_config_dict)

    def _plan_task(self, task: Task, config: AgentConfig) -> None:
        """Uses the Planner to create a subtask graph for a task."""
        api_key = self.credential_store.get("dungx-ai-coding-agent", config.provider)
        if not api_key:
            raise ValueError(f"No API key for planning provider {config.provider}")
        
        provider = build_provider(config, api_key)
        planner = Planner(provider)
        context = Orchestrator(config, provider, self.storage).analyze() # Get repo context
        
        task_plan = planner.create_task_plan(task.objective, context)
        task.plan = task_plan
        self.storage.save_task(task)

    def _is_proposal_valid(self, task: Task, proposal: "PlanProposal") -> bool:
        """Performs a basic pre-validation of a plan proposal."""
        if not proposal.additions and not proposal.modifications:
            return False # Empty proposal is not valid
        
        completed_subtask_ids = {s.subtask_id for s in task.plan.subtasks if s.status == SubtaskStatus.COMPLETED}
        for mod in proposal.modifications:
            if mod.subtask_id in completed_subtask_ids:
                return False # Cannot modify a completed subtask
        return True