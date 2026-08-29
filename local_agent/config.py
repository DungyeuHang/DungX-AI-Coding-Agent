from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on"}


def _positive_int(value: object, name: str, minimum: int = 1) -> int:
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


@dataclass
class AgentConfig:
    project: Path
    provider: str = "mock"
    model: str = "gpt-4.1-mini"
    # Process-local runtime credential. repr=False prevents accidental display.
    api_key: str | None = field(default=None, repr=False)
    max_iterations: int = 5
    command_timeout_seconds: int = 120
    validation_commands: list[str] = field(default_factory=list)
    log_level: str = "INFO"
    api_base_url: str = "https://api.openai.com/v1"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    dry_run: bool = False
    approval: str = "never"
    git_commit_on_completion: bool = False # New for Phase 3.24
    git_default_remote: str = "origin" # New for Phase 3.24
    git_protected_branches: list[str] = field(default_factory=lambda: ["main", "master", "develop"]) # New for Phase 3.24
    git_pr_on_completion: bool = False # New for Phase 3.25
    git_hosting_provider: str = "github" # New for Phase 3.25
    max_parallel_subtasks: int = 1 # New for Phase 3.21
    autonomous_mode: bool = False # New for autonomous mode
    approval_policies: list[dict[str, Any]] = field(default_factory=list)
    max_context_files: int = 24
    max_context_file_bytes: int = 5000
    max_context_tokens: int = 7500
    dependency_depth: int = 1
    planning_context_bytes: int = 30000
    implementation_context_bytes: int = 22000
    repair_context_bytes: int = 12000
    review_context_bytes: int = 16000
    provider_max_retries: int = 1
    approval_mode: Literal["never", "plan_review", "always"] = "never" # Added for Phase 3.12
    max_secondary_validations_per_iteration: int = 1 # Added for Phase 3.13
    max_diagnostic_output_bytes: int = 4000 # Added for Phase 3.13
    max_retry_wait_seconds: int = 60
    max_plan_amendments: int = 5
    max_scope_growth_factor: float = 2.0
    memory_enabled: bool = True # New for Phase 3.22
    metrics_enabled: bool = False
    max_tool_steps: int = 8
    max_tool_output_bytes: int = 4000
    total_tool_budget_bytes: int = 32000
    max_consecutive_repeats: int = 3
    per_tool_limits: dict[str, int] = field(default_factory=dict)
    disallowed_tools: list[str] = field(default_factory=list)
    tool_history_compaction_window: int = 2
    max_tool_history_context_bytes: int = 8000
    planning_provider: str | None = None
    planning_model: str | None = None
    planning_fallbacks: list[str] = field(default_factory=list)
    implementation_provider: str | None = None
    implementation_model: str | None = None
    implementation_fallbacks: list[str] = field(default_factory=list)
    repair_provider: str | None = None
    repair_model: str | None = None
    repair_fallbacks: list[str] = field(default_factory=list)
    review_provider: str | None = None
    review_model: str | None = None
    review_fallbacks: list[str] = field(default_factory=list)
    verification_provider: str | None = None
    verification_model: str | None = None
    verification_fallbacks: list[str] = field(default_factory=list)
    dual_review_enabled: bool = False
    high_risk_dual_review: bool = True
    knowledge_graph_enabled: bool = True
    max_knowledge_context_chars: int = 2000
    max_knowledge_symbols: int = 1000
    parallel_worktree_execution: bool = False
    serialize_overlapping_subtasks: bool = True

    @property
    def tool_policy(self) -> Any:
        from .models import ToolExecutionPolicy
        return ToolExecutionPolicy(
            max_tool_steps=self.max_tool_steps,
            max_tool_output_bytes=self.max_tool_output_bytes,
            total_tool_budget_bytes=self.total_tool_budget_bytes,
            max_consecutive_repeats=self.max_consecutive_repeats,
            per_tool_limits=dict(self.per_tool_limits),
            disallowed_tools=set(self.disallowed_tools),
            compaction_window=self.tool_history_compaction_window,
            max_context_bytes=self.max_tool_history_context_bytes,
        )

    @classmethod
    def from_environment(cls, project: str | Path, **overrides: object) -> "AgentConfig":
        environment_names = {
            "provider": "AGENT_PROVIDER",
            "model": "OPENAI_MODEL",
            "max_iterations": "AGENT_MAX_ITERATIONS",
            "command_timeout_seconds": "AGENT_COMMAND_TIMEOUT",
            "validation_commands": "AGENT_VALIDATION_COMMANDS",
            "log_level": "AGENT_LOG_LEVEL",
            "api_base_url": "OPENAI_BASE_URL",
            "gemini_base_url": "GEMINI_BASE_URL",
            "deepseek_base_url": "DEEPSEEK_BASE_URL",
            "dry_run": "AGENT_DRY_RUN",
            "approval": "AGENT_APPROVAL",
            "git_commit_on_completion": "AGENT_GIT_COMMIT_ON_COMPLETION",
            "git_default_remote": "AGENT_GIT_DEFAULT_REMOTE",
            "git_protected_branches": "AGENT_GIT_PROTECTED_BRANCHES",
            "git_pr_on_completion": "AGENT_GIT_PR_ON_COMPLETION",
            "git_hosting_provider": "AGENT_GIT_HOSTING_PROVIDER",
            "max_parallel_subtasks": "AGENT_MAX_PARALLEL_SUBTASKS",
            "autonomous_mode": "AGENT_AUTONOMOUS",
            "approval_policies": "AGENT_APPROVAL_POLICIES",
            "max_context_files": "AGENT_MAX_CONTEXT_FILES",
            "max_context_file_bytes": "AGENT_MAX_CONTEXT_FILE_BYTES",
            "max_context_tokens": "AGENT_MAX_CONTEXT_TOKENS",
            "dependency_depth": "AGENT_DEPENDENCY_DEPTH",
            "planning_context_bytes": "AGENT_PLANNING_CONTEXT_BYTES",
            "implementation_context_bytes": "AGENT_IMPLEMENTATION_CONTEXT_BYTES",
            "repair_context_bytes": "AGENT_REPAIR_CONTEXT_BYTES",
            "review_context_bytes": "AGENT_REVIEW_CONTEXT_BYTES",
            "provider_max_retries": "AGENT_PROVIDER_MAX_RETRIES",
            "approval_mode": "AGENT_APPROVAL_MODE", # Added for Phase 3.12
            "max_secondary_validations_per_iteration": "AGENT_MAX_SECONDARY_VALIDATIONS", # Added for Phase 3.13
            "max_diagnostic_output_bytes": "AGENT_MAX_DIAGNOSTIC_OUTPUT_BYTES", # Added for Phase 3.13
            "max_retry_wait_seconds": "AGENT_MAX_RETRY_WAIT_SECONDS",
            "memory_enabled": "AGENT_MEMORY_ENABLED",
            "metrics_enabled": "AGENT_METRICS",
            "max_tool_steps": "AGENT_MAX_TOOL_STEPS",
            "max_tool_output_bytes": "AGENT_MAX_TOOL_OUTPUT_BYTES",
            "total_tool_budget_bytes": "AGENT_TOTAL_TOOL_BUDGET_BYTES",
            "max_consecutive_repeats": "AGENT_MAX_CONSECUTIVE_REPEATS",
            "per_tool_limits": "AGENT_PER_TOOL_LIMITS",
            "disallowed_tools": "AGENT_DISALLOWED_TOOLS",
            "tool_history_compaction_window": "AGENT_TOOL_COMPACTION_WINDOW",
            "max_tool_history_context_bytes": "AGENT_MAX_TOOL_HISTORY_CONTEXT_BYTES",
            "max_plan_amendments": "AGENT_MAX_PLAN_AMENDMENTS",
            "max_scope_growth_factor": "AGENT_MAX_SCOPE_GROWTH_FACTOR",
            "planning_provider": "AGENT_PLANNING_PROVIDER",
            "planning_model": "AGENT_PLANNING_MODEL",
            "planning_fallbacks": "AGENT_PLANNING_FALLBACKS",
            "implementation_provider": "AGENT_IMPLEMENTATION_PROVIDER",
            "implementation_model": "AGENT_IMPLEMENTATION_MODEL",
            "implementation_fallbacks": "AGENT_IMPLEMENTATION_FALLBACKS",
            "repair_provider": "AGENT_REPAIR_PROVIDER",
            "repair_model": "AGENT_REPAIR_MODEL",
            "repair_fallbacks": "AGENT_REPAIR_FALLBACKS",
            "review_provider": "AGENT_REVIEW_PROVIDER",
            "review_model": "AGENT_REVIEW_MODEL",
            "review_fallbacks": "AGENT_REVIEW_FALLBACKS",
            "verification_provider": "AGENT_VERIFICATION_PROVIDER",
            "verification_model": "AGENT_VERIFICATION_MODEL",
            "verification_fallbacks": "AGENT_VERIFICATION_FALLBACKS",
            "dual_review_enabled": "AGENT_DUAL_REVIEW",
            "high_risk_dual_review": "AGENT_HIGH_RISK_DUAL_REVIEW",
            "knowledge_graph_enabled": "AGENT_KNOWLEDGE_GRAPH_ENABLED",
            "max_knowledge_context_chars": "AGENT_MAX_KNOWLEDGE_CONTEXT_CHARS",
            "max_knowledge_symbols": "AGENT_MAX_KNOWLEDGE_SYMBOLS",
            "parallel_worktree_execution": "AGENT_PARALLEL_WORKTREES",
            "serialize_overlapping_subtasks": "AGENT_SERIALIZE_OVERLAPPING_SUBTASKS",
        }

        def value(name: str, default: object) -> object:
            return overrides.get(name, os.environ.get(environment_names.get(name, name), default))

        iterations = int(value("max_iterations", 5))
        if iterations < 1 or iterations > 20:
            raise ValueError("max_iterations must be between 1 and 20")
        commands = value("validation_commands", "")
        if isinstance(commands, str):
            commands = [item.strip() for item in commands.split("||") if item.strip()]
        provider_name = str(value("provider", "mock")).lower()
        approval_policies_str = value("approval_policies", "[]")
        try:
            approval_policies_val = json.loads(approval_policies_str) if isinstance(approval_policies_str, str) else approval_policies_str
            if not isinstance(approval_policies_val, list):
                raise ValueError("AGENT_APPROVAL_POLICIES must be a JSON list of objects")
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Invalid format for AGENT_APPROVAL_POLICIES: {e}") from e

        per_tool_limits_val = value("per_tool_limits", {})
        if isinstance(per_tool_limits_val, str) and per_tool_limits_val.strip():
            try:
                parsed_limits = json.loads(per_tool_limits_val)
                if isinstance(parsed_limits, dict):
                    per_tool_limits_val = {str(k): int(v) for k, v in parsed_limits.items()}
                else:
                    per_tool_limits_val = {}
            except (json.JSONDecodeError, ValueError):
                per_tool_limits_val = {}
        elif not isinstance(per_tool_limits_val, dict):
            per_tool_limits_val = {}

        disallowed_tools_val = value("disallowed_tools", [])
        if isinstance(disallowed_tools_val, str) and disallowed_tools_val.strip():
            try:
                parsed_disallowed = json.loads(disallowed_tools_val)
                if isinstance(parsed_disallowed, list):
                    disallowed_tools_val = [str(x).strip() for x in parsed_disallowed if str(x).strip()]
                else:
                    disallowed_tools_val = [t.strip() for t in disallowed_tools_val.split(",") if t.strip()]
            except (json.JSONDecodeError, ValueError):
                disallowed_tools_val = [t.strip() for t in disallowed_tools_val.split(",") if t.strip()]
        elif isinstance(disallowed_tools_val, (set, list, tuple)):
            disallowed_tools_val = [str(x) for x in disallowed_tools_val]
        else:
            disallowed_tools_val = []

        explicit_api_key = overrides.get("api_key")
        if explicit_api_key is None:
            credential_name = "DEEPSEEK_API_KEY" if provider_name == "deepseek" else \
                              "GEMINI_API_KEY" if provider_name in {"gemini", "antigravity"} else \
                              "OPENAI_API_KEY"
            configured_api_key = os.environ.get(credential_name)
        else:
            configured_api_key = str(explicit_api_key) or None
        configured_model = overrides.get("model")
        if configured_model is None:
            model_env = "GEMINI_MODEL" if provider_name in {"gemini", "antigravity"} else "OPENAI_MODEL"
            configured_model = os.environ.get(model_env, "gemini-2.5-flash" if provider_name == "gemini" else "gpt-4.1-mini")
            if provider_name == "antigravity" and model_env not in os.environ:
                configured_model = "gemini-3.7-flash"

        def _parse_fallbacks(val: object) -> list[str]:
            if isinstance(val, str) and val.strip():
                try:
                    parsed = json.loads(val)
                    if isinstance(parsed, list):
                        return [str(x).strip() for x in parsed if str(x).strip()]
                except Exception:
                    pass
                return [x.strip() for x in val.split(",") if x.strip()]
            elif isinstance(val, (list, tuple, set)):
                return [str(x).strip() for x in val if str(x).strip()]
            return []

        def _opt_str(val: object) -> str | None:
            if val is None:
                return None
            s = str(val).strip()
            return s if s else None

        config = cls(
            project=Path(project).expanduser().resolve(),
            provider=provider_name,
            model=str(configured_model),
            api_key=configured_api_key,
            max_iterations=iterations,
            command_timeout_seconds=int(value("command_timeout_seconds", 120)),
            validation_commands=list(commands),
            log_level=str(value("log_level", "INFO")).upper(),
            api_base_url=str(value("api_base_url", "https://api.openai.com/v1")),
            gemini_base_url=str(value("gemini_base_url", "https://generativelanguage.googleapis.com/v1beta")),
            deepseek_base_url=str(value("deepseek_base_url", "https://api.deepseek.com/v1")),
            dry_run=_bool(value("dry_run", False)),
            approval=str(value("approval", "never")).lower(),
            git_commit_on_completion=_bool(value("git_commit_on_completion", False)),
            git_default_remote=str(value("git_default_remote", "origin")),
            git_protected_branches=[b.strip() for b in str(value("git_protected_branches", "main,master,develop")).split(',')],
            git_pr_on_completion=_bool(value("git_pr_on_completion", False)),
            git_hosting_provider=str(value("git_hosting_provider", "github")).lower(),
            max_parallel_subtasks=_positive_int(value("max_parallel_subtasks", 1), "max_parallel_subtasks"),
            autonomous_mode=_bool(value("autonomous_mode", False)),
            approval_policies=approval_policies_val,
            max_context_files=_positive_int(value("max_context_files", 24), "max_context_files"),
            max_context_file_bytes=_positive_int(value("max_context_file_bytes", 5000), "max_context_file_bytes"),
            max_context_tokens=_positive_int(value("max_context_tokens", 7500), "max_context_tokens"),
            dependency_depth=_positive_int(value("dependency_depth", 1), "dependency_depth", minimum=0),
            planning_context_bytes=_positive_int(value("planning_context_bytes", 30000), "planning_context_bytes"),
            implementation_context_bytes=_positive_int(value("implementation_context_bytes", 22000), "implementation_context_bytes"),
            repair_context_bytes=_positive_int(value("repair_context_bytes", 12000), "repair_context_bytes"),
            review_context_bytes=_positive_int(value("review_context_bytes", 16000), "review_context_bytes"),
            # For Phase 3.12, only 'never' and 'plan_review' are fully implemented.
            # 'always' is a placeholder for future work.
            approval_mode=str(value("approval_mode", "never")).lower(), # Added for Phase 3.12
            max_secondary_validations_per_iteration=_positive_int(value("max_secondary_validations_per_iteration", 1), "max_secondary_validations_per_iteration", minimum=0), # Added for Phase 3.13
            max_diagnostic_output_bytes=_positive_int(value("max_diagnostic_output_bytes", 4000), "max_diagnostic_output_bytes"), # Added for Phase 3.13
            provider_max_retries=_positive_int(value("provider_max_retries", 1), "provider_max_retries", minimum=0),
            memory_enabled=_bool(value("memory_enabled", True)),
            max_retry_wait_seconds=_positive_int(value("max_retry_wait_seconds", 60), "max_retry_wait_seconds"),
            metrics_enabled=_bool(value("metrics_enabled", False)),
            max_tool_steps=_positive_int(value("max_tool_steps", 8), "max_tool_steps"),
            max_tool_output_bytes=_positive_int(value("max_tool_output_bytes", 4000), "max_tool_output_bytes"),
            total_tool_budget_bytes=_positive_int(value("total_tool_budget_bytes", 32000), "total_tool_budget_bytes"),
            max_consecutive_repeats=_positive_int(value("max_consecutive_repeats", 3), "max_consecutive_repeats"),
            per_tool_limits=per_tool_limits_val,
            disallowed_tools=disallowed_tools_val,
            tool_history_compaction_window=_positive_int(value("tool_history_compaction_window", 2), "tool_history_compaction_window"),
            max_tool_history_context_bytes=_positive_int(value("max_tool_history_context_bytes", 8000), "max_tool_history_context_bytes"),
            max_plan_amendments=_positive_int(value("max_plan_amendments", 5), "max_plan_amendments"),
            max_scope_growth_factor=float(value("max_scope_growth_factor", 2.0)),
            planning_provider=_opt_str(value("planning_provider", None)),
            planning_model=_opt_str(value("planning_model", None)),
            planning_fallbacks=_parse_fallbacks(value("planning_fallbacks", [])),
            implementation_provider=_opt_str(value("implementation_provider", None)),
            implementation_model=_opt_str(value("implementation_model", None)),
            implementation_fallbacks=_parse_fallbacks(value("implementation_fallbacks", [])),
            repair_provider=_opt_str(value("repair_provider", None)),
            repair_model=_opt_str(value("repair_model", None)),
            repair_fallbacks=_parse_fallbacks(value("repair_fallbacks", [])),
            review_provider=_opt_str(value("review_provider", None)),
            review_model=_opt_str(value("review_model", None)),
            review_fallbacks=_parse_fallbacks(value("review_fallbacks", [])),
            verification_provider=_opt_str(value("verification_provider", None)),
            verification_model=_opt_str(value("verification_model", None)),
            verification_fallbacks=_parse_fallbacks(value("verification_fallbacks", [])),
            dual_review_enabled=_bool(value("dual_review_enabled", False)),
            high_risk_dual_review=_bool(value("high_risk_dual_review", True)),
            knowledge_graph_enabled=_bool(value("knowledge_graph_enabled", True)),
            max_knowledge_context_chars=_positive_int(value("max_knowledge_context_chars", 2000), "max_knowledge_context_chars"),
            max_knowledge_symbols=_positive_int(value("max_knowledge_symbols", 1000), "max_knowledge_symbols"),
            parallel_worktree_execution=_bool(value("parallel_worktree_execution", False)),
            serialize_overlapping_subtasks=_bool(value("serialize_overlapping_subtasks", True)),
        )
        if config.approval_mode not in {"never", "plan_review", "always"}:
            raise ValueError("approval_mode must be 'never', 'plan_review', or 'always'")
        return config

    def validate(self) -> None:
        if not self.project.is_dir():
            raise ValueError(f"project directory does not exist: {self.project}")
        valid_providers = {"mock", "openai", "gemini", "antigravity", "deepseek", "anthropic"}
        if self.provider not in valid_providers:
            raise ValueError("provider must be one of 'mock', 'openai', 'gemini', 'antigravity', 'deepseek', 'anthropic'")
        for role_name, prov in (
            ("planning_provider", self.planning_provider),
            ("implementation_provider", self.implementation_provider),
            ("repair_provider", self.repair_provider),
            ("review_provider", self.review_provider),
            ("verification_provider", self.verification_provider),
        ):
            if prov is not None and prov not in valid_providers:
                raise ValueError(f"{role_name} must be one of 'mock', 'openai', 'gemini', 'antigravity', 'deepseek', 'anthropic'")
        if self.approval not in {"never", "always", "policy"}:
            raise ValueError("approval must be 'never', 'always', or 'policy'")
        if self.approval_mode not in {"never", "plan_review", "always"}:
            raise ValueError("approval_mode must be 'never', 'plan_review', or 'always'")
        if self.command_timeout_seconds < 1:
            raise ValueError("command_timeout_seconds must be positive")
        if self.max_parallel_subtasks < 1:
            raise ValueError("max_parallel_subtasks must be at least 1")
        if self.max_parallel_subtasks > 4:
            raise ValueError("max_parallel_subtasks cannot exceed 4")
        for name in ("max_context_files", "max_context_file_bytes", "max_context_tokens", "planning_context_bytes", "implementation_context_bytes", "repair_context_bytes", "review_context_bytes", "max_retry_wait_seconds", "max_diagnostic_output_bytes", "max_tool_steps", "max_tool_output_bytes", "total_tool_budget_bytes", "max_consecutive_repeats", "max_knowledge_context_chars", "max_knowledge_symbols"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.provider_max_retries < 0:
            raise ValueError("provider_max_retries cannot be negative")
        if self.dependency_depth < 0:
            raise ValueError("dependency_depth cannot be negative")
        if self.max_plan_amendments < 0:
            raise ValueError("max_plan_amendments cannot be negative")
        if self.max_scope_growth_factor < 1.0:
            raise ValueError("max_scope_growth_factor must be at least 1.0")
        # Validate tool policy creation
        _ = self.tool_policy

def add_common_arguments(parser: argparse.ArgumentParser, include_provider_args: bool = True) -> None:
    parser.add_argument("--project", default=".", help="local project directory")
    parser.add_argument("--provider", choices=("mock", "openai", "gemini", "antigravity", "deepseek", "anthropic"), default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--planning-provider", choices=("mock", "openai", "gemini", "antigravity", "deepseek", "anthropic"), default=None)
    parser.add_argument("--planning-model", default=None)
    parser.add_argument("--implementation-provider", choices=("mock", "openai", "gemini", "antigravity", "deepseek", "anthropic"), default=None)
    parser.add_argument("--implementation-model", default=None)
    parser.add_argument("--repair-provider", choices=("mock", "openai", "gemini", "antigravity", "deepseek", "anthropic"), default=None)
    parser.add_argument("--repair-model", default=None)
    parser.add_argument("--review-provider", choices=("mock", "openai", "gemini", "antigravity", "deepseek", "anthropic"), default=None)
    parser.add_argument("--review-model", default=None)
    parser.add_argument("--verification-provider", choices=("mock", "openai", "gemini", "antigravity", "deepseek", "anthropic"), default=None)
    parser.add_argument("--verification-model", default=None)
    parser.add_argument("--dual-review", type=_bool, default=None, help="enable deliberative dual-model review")
    parser.add_argument("--knowledge-graph", type=_bool, default=None, help="enable persistent repository knowledge graph (true/false)")
    parser.add_argument("--parallel-worktrees", type=_bool, default=None, help="enable parallel DAG execution in isolated Git worktrees (true/false)")
    parser.add_argument("--serialize-overlapping-subtasks", type=_bool, default=None, help="serialize execution of subtasks with predicted file overlap (true/false)")
    parser.add_argument("--validation", action="append", default=None, help="explicit validation command")
    parser.add_argument("--log-level", default=None, choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"))
    parser.add_argument("--dry-run", action="store_true", help="generate and display changes without writing files")
    parser.add_argument("--approval", choices=("never", "always", "policy"), default=None, help="control code change approval behavior")
    parser.add_argument("--git-pr-on-completion", type=_bool, default=None, help="[Autonomous] create a pull request on successful push")
    parser.add_argument("--git-commit-on-completion", type=_bool, default=None, help="[Autonomous] create a local commit on successful completion")
    parser.add_argument("--max-parallel-subtasks", type=int, default=None, help="maximum number of subtasks to run in parallel")
    parser.add_argument("--autonomous", action="store_true", help="enable experimental autonomous execution mode")
    parser.add_argument("--approval-policy-file", default=None, help="path to a JSON file containing approval policies")
    parser.add_argument("--memory", type=_bool, default=None, help="enable or disable project memory (true/false)")
    parser.add_argument("--approval-mode", choices=("never", "plan_review", "always"), default=None, help="control plan approval behavior") # Added for Phase 3.12
    if include_provider_args:
        parser.add_argument("--api-key", default=None, help="provider API key (defaults to environment variable)")
        parser.add_argument("--api-base-url", default=None, help="override the provider API base URL")
        parser.add_argument("--anthropic-base-url", default=None, help="override the Anthropic API base URL")
        parser.add_argument("--gemini-base-url", default=None, help="override the Gemini API base URL")
        parser.add_argument("--deepseek-base-url", default=None, help="override the DeepSeek API base URL")


def config_from_args(args: argparse.Namespace) -> AgentConfig:
    overrides = {
        key: value for key, value in {
            "provider": args.provider,
            "model": args.model,
            "planning_provider": getattr(args, "planning_provider", None),
            "planning_model": getattr(args, "planning_model", None),
            "implementation_provider": getattr(args, "implementation_provider", None),
            "implementation_model": getattr(args, "implementation_model", None),
            "repair_provider": getattr(args, "repair_provider", None),
            "repair_model": getattr(args, "repair_model", None),
            "review_provider": getattr(args, "review_provider", None),
            "review_model": getattr(args, "review_model", None),
            "verification_provider": getattr(args, "verification_provider", None),
            "verification_model": getattr(args, "verification_model", None),
            "dual_review_enabled": getattr(args, "dual_review", None),
            "knowledge_graph_enabled": getattr(args, "knowledge_graph", None),
            "parallel_worktree_execution": getattr(args, "parallel_worktrees", None),
            "serialize_overlapping_subtasks": getattr(args, "serialize_overlapping_subtasks", None),
            "validation_commands": args.validation,
            "log_level": args.log_level,
            "dry_run": args.dry_run if args.dry_run else None,
            "approval": args.approval,
            "git_pr_on_completion": args.git_pr_on_completion,
            "git_commit_on_completion": args.git_commit_on_completion,
            "max_parallel_subtasks": args.max_parallel_subtasks,
            "autonomous_mode": args.autonomous if args.autonomous else None,
            "memory_enabled": args.memory,
            "approval_mode": args.approval_mode, # Added for Phase 3.12
            "api_key": getattr(args, "api_key", None),
            "api_base_url": getattr(args, "api_base_url", None),
            "anthropic_base_url": getattr(args, "anthropic_base_url", None),
            "gemini_base_url": getattr(args, "gemini_base_url", None),
            "deepseek_base_url": getattr(args, "deepseek_base_url", None),
        }.items() if value is not None
    }
    if getattr(args, "approval_policy_file", None):
        policy_path = Path(args.approval_policy_file)
        if not policy_path.is_file():
            raise FileNotFoundError(f"Approval policy file not found: {policy_path}")
        with policy_path.open('r', encoding='utf-8') as f:
            overrides["approval_policies"] = json.load(f)

    config = AgentConfig.from_environment(args.project, **overrides)
    config.validate()
    return config
