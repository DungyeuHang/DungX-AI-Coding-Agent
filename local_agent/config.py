from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AgentConfig:
    project: Path
    provider: str = "mock"
    model: str = "gpt-4.1-mini"
    max_iterations: int = 5
    command_timeout_seconds: int = 120
    validation_commands: list[str] = field(default_factory=list)
    log_level: str = "INFO"
    api_base_url: str = "https://api.openai.com/v1"

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
        }

        def value(name: str, default: object) -> object:
            return overrides.get(name, os.environ.get(environment_names.get(name, name), default))

        iterations = int(value("max_iterations", 5))
        if iterations < 1 or iterations > 20:
            raise ValueError("max_iterations must be between 1 and 20")
        commands = value("validation_commands", "")
        if isinstance(commands, str):
            commands = [item.strip() for item in commands.split("||") if item.strip()]
        return cls(
            project=Path(project).expanduser().resolve(),
            provider=str(value("provider", "mock")).lower(),
            model=str(value("model", "gpt-4.1-mini")),
            max_iterations=iterations,
            command_timeout_seconds=int(value("command_timeout_seconds", 120)),
            validation_commands=list(commands),
            log_level=str(value("log_level", "INFO")).upper(),
            api_base_url=str(value("api_base_url", "https://api.openai.com/v1")),
        )

    def validate(self) -> None:
        if not self.project.is_dir():
            raise ValueError(f"project directory does not exist: {self.project}")
        if self.provider not in {"mock", "openai"}:
            raise ValueError("provider must be 'mock' or 'openai'")
        if self.command_timeout_seconds < 1:
            raise ValueError("command_timeout_seconds must be positive")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project", default=".", help="local project directory")
    parser.add_argument("--provider", choices=("mock", "openai"), default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--validation", action="append", default=None, help="explicit validation command")
    parser.add_argument("--log-level", default=None, choices=("DEBUG", "INFO", "WARNING"))


def config_from_args(args: argparse.Namespace) -> AgentConfig:
    overrides = {
        key: value for key, value in {
            "provider": args.provider,
            "model": args.model,
            "max_iterations": args.max_iterations,
            "validation_commands": args.validation,
            "log_level": args.log_level,
        }.items() if value is not None
    }
    config = AgentConfig.from_environment(args.project, **overrides)
    config.validate()
    return config
