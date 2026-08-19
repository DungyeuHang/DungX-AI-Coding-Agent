from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path


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
    metrics_enabled: bool = False

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
            "metrics_enabled": "AGENT_METRICS",
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
        return cls(
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
            max_retry_wait_seconds=_positive_int(value("max_retry_wait_seconds", 60), "max_retry_wait_seconds"),
            metrics_enabled=_bool(value("metrics_enabled", False)),
        )

    def validate(self) -> None:
        if not self.project.is_dir():
            raise ValueError(f"project directory does not exist: {self.project}")
        if self.provider not in {"mock", "openai", "gemini", "antigravity", "deepseek"}:
            raise ValueError("provider must be 'mock', 'openai', 'gemini', 'antigravity', or 'deepseek'")
        if self.approval not in {"never", "always"}: # This is for code changes approval, not plan approval.
            raise ValueError("approval must be 'never' or 'always'")
        if self.approval_mode not in {"never", "plan_review", "always"}:
            raise ValueError("approval_mode must be 'never', 'plan_review', or 'always'")
        if self.command_timeout_seconds < 1:
            raise ValueError("command_timeout_seconds must be positive")
        for name in ("max_context_files", "max_context_file_bytes", "max_context_tokens", "planning_context_bytes", "implementation_context_bytes", "repair_context_bytes", "review_context_bytes", "max_retry_wait_seconds", "max_diagnostic_output_bytes"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.provider_max_retries < 0:
            raise ValueError("provider_max_retries cannot be negative")
        if self.dependency_depth < 0:
            raise ValueError("dependency_depth cannot be negative")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project", default=".", help="local project directory")
    parser.add_argument("--provider", choices=("mock", "openai", "gemini", "antigravity"), default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--validation", action="append", default=None, help="explicit validation command")
    parser.add_argument("--log-level", default=None, choices=("DEBUG", "INFO", "WARNING"))
    parser.add_argument("--dry-run", action="store_true", help="generate and display changes without writing files")
    parser.add_argument("--approval", choices=("never", "always"), default=None, help="pause for approval before applying changes")
    parser.add_argument("--approval-mode", choices=("never", "plan_review", "always"), default=None, help="control plan approval behavior") # Added for Phase 3.12


def config_from_args(args: argparse.Namespace) -> AgentConfig:
    overrides = {
        key: value for key, value in {
            "provider": args.provider,
            "model": args.model,
            "max_iterations": args.max_iterations,
            "validation_commands": args.validation,
            "log_level": args.log_level,
            "dry_run": args.dry_run if args.dry_run else None,
            "approval": args.approval,
            "approval_mode": args.approval_mode, # Added for Phase 3.12
        }.items() if value is not None
    }
    config = AgentConfig.from_environment(args.project, **overrides)
    config.validate()
    return config
