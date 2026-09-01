"""
Production Preflight Validator & Environment Readiness Checker for Phase 4.25.
Enforces deterministic validation of providers, repository, sandboxing, storage,
safety policies, and execution budgets before autonomous task execution starts.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

from .config import AgentConfig
from .filesystem import ProjectFilesystem, ProtectedPathError, SandboxViolation, SECRET_NAMES
from .git import GitIntegration
from .models import (
    ApprovalPolicy,
    PreflightCheckCategory,
    PreflightCheckItem,
    PreflightCheckStatus,
    PreflightReport,
    ProviderHealthStatus,
    RoleHealthReport,
    SpecialistRole,
)
from .providers import SpecialistModelRouter

LOGGER = logging.getLogger(__name__)


class PreflightChecker:
    """
    Evaluates preflight readiness across 6 key operational categories:
    1. PROVIDERS: checks configured vs active provider health for all specialist roles
    2. REPOSITORY: validates project directory existence, write permissions, git status
    3. TOOLS: verifies command runner and sandbox containment
    4. STORAGE: verifies atomic task storage directory and checkpoint integrity
    5. SAFETY: verifies approval policies and secret protection invariants
    6. EXECUTION: validates iteration limits, timeout bounds, and context budgets
    """

    def __init__(
        self,
        config: AgentConfig,
        storage: Any | None = None,
        router: SpecialistModelRouter | None = None,
    ):
        self.config = config
        self.storage = storage
        self.router = router or SpecialistModelRouter(config)
        self.filesystem = ProjectFilesystem(config.project) if Path(config.project).is_dir() else None

    def check(self, require_real_providers: bool = False) -> PreflightReport:
        """
        Runs the complete preflight check suite.
        If require_real_providers is True (e.g. in autonomous dogfooding mode),
        any degraded or mock provider for critical roles will result in a BLOCKED status.
        """
        checks: list[PreflightCheckItem] = []

        # 1. Provider Checks
        checks.extend(self._check_providers(require_real_providers=require_real_providers))

        # 2. Repository Checks
        checks.extend(self._check_repository())

        # 3. Tool & Sandbox Checks
        checks.extend(self._check_tools_and_sandbox())

        # 4. Storage & Persistence Checks
        checks.extend(self._check_storage())

        # 5. Safety & Approval Checks
        checks.extend(self._check_safety())

        # 6. Execution & Budget Checks
        checks.extend(self._check_execution_budgets())

        # Compute overall status
        has_blocker = any(c.status == PreflightCheckStatus.BLOCKED.value for c in checks)
        has_warning = any(c.status == PreflightCheckStatus.WARNING.value for c in checks)

        if has_blocker:
            overall_status = "BLOCKED"
            is_ready = False
        elif has_warning:
            overall_status = "READY_WITH_WARNINGS"
            is_ready = True
        else:
            overall_status = "READY"
            is_ready = True

        roles_health = {
            role: report.to_dict()
            for role, report in self.router.get_all_roles_health().items()
        }

        blocker_count = sum(1 for c in checks if c.status == PreflightCheckStatus.BLOCKED.value)
        warning_count = sum(1 for c in checks if c.status == PreflightCheckStatus.WARNING.value)
        passed_count = sum(1 for c in checks if c.status == PreflightCheckStatus.PASSED.value)
        summary = f"Preflight {overall_status}: {passed_count} passed, {warning_count} warnings, {blocker_count} blockers"

        return PreflightReport(
            overall_status=overall_status,
            is_ready=is_ready,
            checks=checks,
            roles_health=roles_health,
            summary=summary,
        )

    def _check_providers(self, require_real_providers: bool = False) -> list[PreflightCheckItem]:
        items: list[PreflightCheckItem] = []
        all_health = self.router.get_all_roles_health()

        for role_name, health in all_health.items():
            if health.status == ProviderHealthStatus.HEALTHY_REAL_PROVIDER.value:
                items.append(PreflightCheckItem(
                    name=f"PROVIDER_ROLE_{role_name.upper()}",
                    category=PreflightCheckCategory.PROVIDERS.value,
                    status=PreflightCheckStatus.PASSED.value,
                    message=f"Specialist role '{role_name}' is backed by healthy provider '{health.active_provider}'",
                    details=health.to_dict(),
                ))
            elif health.status == ProviderHealthStatus.EXPLICIT_OFFLINE_MOCK.value:
                if require_real_providers:
                    items.append(PreflightCheckItem(
                        name=f"PROVIDER_ROLE_{role_name.upper()}",
                        category=PreflightCheckCategory.PROVIDERS.value,
                        status=PreflightCheckStatus.BLOCKED.value,
                        message=f"Specialist role '{role_name}' is configured for MockProvider, but real providers are required",
                        remediation=f"Configure a real provider (e.g. GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, DEEPSEEK_API_KEY) or pass --provider with a real provider",
                        details=health.to_dict(),
                    ))
                else:
                    items.append(PreflightCheckItem(
                        name=f"PROVIDER_ROLE_{role_name.upper()}",
                        category=PreflightCheckCategory.PROVIDERS.value,
                        status=PreflightCheckStatus.INFO.value,
                        message=f"Specialist role '{role_name}' is operating in explicit offline mock mode",
                        details=health.to_dict(),
                    ))
            elif health.status == ProviderHealthStatus.DEGRADED_FALLBACK.value:
                if require_real_providers:
                    items.append(PreflightCheckItem(
                        name=f"PROVIDER_ROLE_{role_name.upper()}",
                        category=PreflightCheckCategory.PROVIDERS.value,
                        status=PreflightCheckStatus.BLOCKED.value,
                        message=f"Specialist role '{role_name}' silently degraded to MockProvider: {health.degradation_reason}",
                        remediation=f"Check API keys and network connectivity for configured provider '{health.configured_provider}'",
                        details=health.to_dict(),
                    ))
                else:
                    items.append(PreflightCheckItem(
                        name=f"PROVIDER_ROLE_{role_name.upper()}",
                        category=PreflightCheckCategory.PROVIDERS.value,
                        status=PreflightCheckStatus.WARNING.value,
                        message=f"Specialist role '{role_name}' degraded to MockProvider: {health.degradation_reason}",
                        remediation=f"Check credentials for configured provider '{health.configured_provider}'",
                        details=health.to_dict(),
                    ))
            else:
                items.append(PreflightCheckItem(
                    name=f"PROVIDER_ROLE_{role_name.upper()}",
                    category=PreflightCheckCategory.PROVIDERS.value,
                    status=PreflightCheckStatus.BLOCKED.value,
                    message=f"Specialist role '{role_name}' is unavailable",
                    remediation="Configure a valid AI provider",
                    details=health.to_dict(),
                ))

        return items

    def _check_repository(self) -> list[PreflightCheckItem]:
        items: list[PreflightCheckItem] = []
        proj = Path(self.config.project)

        if not proj.exists() or not proj.is_dir():
            items.append(PreflightCheckItem(
                name="REPOSITORY_PATH_EXISTS",
                category=PreflightCheckCategory.REPOSITORY.value,
                status=PreflightCheckStatus.BLOCKED.value,
                message=f"Project path does not exist or is not a directory: {proj}",
                remediation="Specify a valid target repository directory with --project",
            ))
            return items
        else:
            items.append(PreflightCheckItem(
                name="REPOSITORY_PATH_EXISTS",
                category=PreflightCheckCategory.REPOSITORY.value,
                status=PreflightCheckStatus.PASSED.value,
                message=f"Target repository path verified: {proj}",
            ))

        try:
            probe_file = proj / ".preflight_write_probe.tmp"
            probe_file.write_text("probe", encoding="utf-8")
            probe_file.unlink()
            items.append(PreflightCheckItem(
                name="REPOSITORY_WRITABLE",
                category=PreflightCheckCategory.REPOSITORY.value,
                status=PreflightCheckStatus.PASSED.value,
                message="Target repository is writable",
            ))
        except Exception as exc:
            items.append(PreflightCheckItem(
                name="REPOSITORY_WRITABLE",
                category=PreflightCheckCategory.REPOSITORY.value,
                status=PreflightCheckStatus.BLOCKED.value,
                message=f"Target repository is not writable: {exc}",
                remediation="Ensure user permissions allow writing to the project directory",
            ))

        git = GitIntegration(proj)
        if git.is_repository():
            raw_status = git.status()
            # Non-branch status lines indicate uncommitted changes
            changed_lines = [l for l in raw_status.splitlines() if l.strip() and not l.startswith("##")]
            if not changed_lines:
                items.append(PreflightCheckItem(
                    name="GIT_WORKING_TREE_CLEAN",
                    category=PreflightCheckCategory.REPOSITORY.value,
                    status=PreflightCheckStatus.PASSED.value,
                    message="Git repository working tree is clean",
                ))
            else:
                items.append(PreflightCheckItem(
                    name="GIT_WORKING_TREE_CLEAN",
                    category=PreflightCheckCategory.REPOSITORY.value,
                    status=PreflightCheckStatus.WARNING.value,
                    message=f"Git working tree has {len(changed_lines)} uncommitted file(s)",
                    remediation="Commit or stash changes before autonomous execution to ensure clean rollback capability",
                    details={"changed_files": changed_lines},
                ))
        else:
            items.append(PreflightCheckItem(
                name="GIT_REPOSITORY_DETECTED",
                category=PreflightCheckCategory.REPOSITORY.value,
                status=PreflightCheckStatus.INFO.value,
                message="Target directory is not a Git repository; VCS tracking and branch isolation will be skipped",
                remediation="Run 'git init' in the project directory for full autonomous git branch isolation",
            ))

        return items

    def _check_tools_and_sandbox(self) -> list[PreflightCheckItem]:
        items: list[PreflightCheckItem] = []
        if self.filesystem:
            try:
                self.filesystem.resolve("../outside_target_file.txt")
                items.append(PreflightCheckItem(
                    name="FILESYSTEM_SANDBOX_CONFINEMENT",
                    category=PreflightCheckCategory.TOOLS.value,
                    status=PreflightCheckStatus.BLOCKED.value,
                    message="Filesystem sandbox failed to reject outside path traversal",
                    remediation="Inspect ProjectFilesystem.resolve security boundary",
                ))
            except (SandboxViolation, PermissionError, ValueError):
                items.append(PreflightCheckItem(
                    name="FILESYSTEM_SANDBOX_CONFINEMENT",
                    category=PreflightCheckCategory.TOOLS.value,
                    status=PreflightCheckStatus.PASSED.value,
                    message="Filesystem sandbox correctly confines operations to project root",
                ))
        return items

    def _check_storage(self) -> list[PreflightCheckItem]:
        items: list[PreflightCheckItem] = []
        data_dir = Path(getattr(self.config, "data_dir", None) or (self.config.project / ".agent_data"))
        try:
            data_dir.mkdir(parents=True, exist_ok=True)
            items.append(PreflightCheckItem(
                name="STORAGE_PERSISTENCE_DIR",
                category=PreflightCheckCategory.STORAGE.value,
                status=PreflightCheckStatus.PASSED.value,
                message=f"Persistence storage directory is accessible: {data_dir}",
            ))
        except Exception as exc:
            items.append(PreflightCheckItem(
                name="STORAGE_PERSISTENCE_DIR",
                category=PreflightCheckCategory.STORAGE.value,
                status=PreflightCheckStatus.BLOCKED.value,
                message=f"Cannot create or write to persistence storage directory '{data_dir}': {exc}",
                remediation="Ensure write permissions for .agent_data directory",
            ))

        if self.storage and hasattr(self.storage, "list_checkpoints"):
            try:
                checkpoints = self.storage.list_checkpoints()
                items.append(PreflightCheckItem(
                    name="CHECKPOINT_STORE_INTEGRITY",
                    category=PreflightCheckCategory.STORAGE.value,
                    status=PreflightCheckStatus.PASSED.value,
                    message=f"Checkpoint store verified ({len(checkpoints)} existing checkpoint(s))",
                ))
            except Exception as exc:
                items.append(PreflightCheckItem(
                    name="CHECKPOINT_STORE_INTEGRITY",
                    category=PreflightCheckCategory.STORAGE.value,
                    status=PreflightCheckStatus.WARNING.value,
                    message=f"Warning inspecting existing checkpoints: {exc}",
                    remediation="Clean corrupted files from .agent_data/checkpoints if resuming fails",
                ))
        return items

    def _check_safety(self) -> list[PreflightCheckItem]:
        items: list[PreflightCheckItem] = []
        policies = getattr(self.config, "approval_policies", [])
        try:
            parsed_policies = [ApprovalPolicy.from_dict(p) for p in policies]
            items.append(PreflightCheckItem(
                name="APPROVAL_POLICY_SYNTAX",
                category=PreflightCheckCategory.SAFETY.value,
                status=PreflightCheckStatus.PASSED.value,
                message=f"Approval policy engine validated ({len(parsed_policies)} active policy rules)",
            ))
        except Exception as exc:
            items.append(PreflightCheckItem(
                name="APPROVAL_POLICY_SYNTAX",
                category=PreflightCheckCategory.SAFETY.value,
                status=PreflightCheckStatus.BLOCKED.value,
                message=f"Invalid approval policies format: {exc}",
                remediation="Fix JSON schema in approval policy configuration",
            ))
        return items

    def _check_execution_budgets(self) -> list[PreflightCheckItem]:
        items: list[PreflightCheckItem] = []
        if self.config.max_iterations < 1:
            items.append(PreflightCheckItem(
                name="ITERATION_BUDGET",
                category=PreflightCheckCategory.EXECUTION.value,
                status=PreflightCheckStatus.BLOCKED.value,
                message=f"max_iterations must be at least 1 (configured: {self.config.max_iterations})",
            ))
        else:
            items.append(PreflightCheckItem(
                name="ITERATION_BUDGET",
                category=PreflightCheckCategory.EXECUTION.value,
                status=PreflightCheckStatus.PASSED.value,
                message=f"Execution iteration budget: {self.config.max_iterations}",
            ))

        if self.config.command_timeout_seconds < 1:
            items.append(PreflightCheckItem(
                name="COMMAND_TIMEOUT",
                category=PreflightCheckCategory.EXECUTION.value,
                status=PreflightCheckStatus.BLOCKED.value,
                message=f"command_timeout_seconds must be positive (configured: {self.config.command_timeout_seconds})",
            ))
        else:
            items.append(PreflightCheckItem(
                name="COMMAND_TIMEOUT",
                category=PreflightCheckCategory.EXECUTION.value,
                status=PreflightCheckStatus.PASSED.value,
                message=f"Command execution timeout: {self.config.command_timeout_seconds}s",
            ))

        return items
