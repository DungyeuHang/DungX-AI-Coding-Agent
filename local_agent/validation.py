from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from .models import ChangeImpact, CommandSpec, ProjectContext, RepositoryMap, ValidationCommand, ValidationPlan

_INTENT_KEYWORDS = {
    "add": re.compile(r"\b(add|create|implement|introduce|build)\b", re.I),
    "modify": re.compile(r"\b(modify|change|update|alter|adjust|set)\b", re.I),
    "fix": re.compile(r"\b(fix|repair|resolve|correct|debug)\b", re.I),
    "remove": re.compile(r"\b(remove|delete|drop)\b", re.I),
    "refactor": re.compile(r"\b(refactor|restructure|reorganize|cleanup)\b", re.I),
    "test": re.compile(r"\b(test|tests|testing)\b", re.I),
}

_ENTITY_KEYWORDS = {
    "page": {"page", "screen"},
    "component": {"component", "widget", "element"},
    "route": {"route", "routing", "path"},
    "navigation": {"navigation", "nav", "menu", "link"},
    "api": {"api", "endpoint", "interface"},
    "backend": {"backend", "server", "firebase", "service", "database", "db"},
    "auth": {"auth", "authentication", "login", "signin", "signup"},
    "test": {"test", "tests", "testing"},
    "config": {"config", "configuration", "env"},
    "dependency": {"dependency", "package", "library"},
    "build": {"build", "compile"},
    "lint": {"lint", "format", "check"},
}

_DANGEROUS_COMMAND_PATTERNS = [
    re.compile(r"\b(rm|del)\s+.*(--force|-f)\b", re.I),
    re.compile(r"\b(git\s+(reset|clean|push|rebase|merge))\b", re.I),
    re.compile(r"\b(db:migrate:reset|db:drop|migrate:rollback)\b", re.I), # Rails-like
    re.compile(r"\b(deploy|publish)\b", re.I),
]

def _name_tokens(value: str) -> set[str]:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    value = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", value)
    return {token.lower() for token in re.findall(r"[A-Za-z0-9]+", value)}

class ValidationIntelligence:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def discover_commands(self, repo_map: RepositoryMap) -> list[ValidationCommand]:
        discovered: list[ValidationCommand] = []

        # JavaScript/TypeScript projects
        package_json_path = self.root / "package.json"
        if package_json_path.exists():
            try:
                package_data = json.loads(package_json_path.read_text(encoding="utf-8"))
                scripts = package_data.get("scripts", {})
                for script_name, script_cmd in scripts.items():
                    cmd_parts = tuple(script_cmd.split())
                    category, risk, destructive = self._classify_js_command(script_name, script_cmd, repo_map)
                    discovered.append(ValidationCommand(
                        name=script_name,
                        command=cmd_parts,
                        category=category,
                        confidence=0.9,
                        reason=f"package.json script '{script_name}'",
                        working_directory=".",
                        destructive=destructive,
                        risk=risk,
                    ))
            except (OSError, json.JSONDecodeError):
                pass

        # Python projects
        if any(f.extension == ".py" for f in repo_map.files):
            # unittest
            if any("unittest" in f.path for f in repo_map.test_files):
                discovered.append(ValidationCommand(
                    name="unittest",
                    command=("python", "-m", "unittest", "discover"),
                    category="unit_test",
                    confidence=0.8,
                    reason="Python unittest files detected",
                    working_directory=".",
                    risk="low",
                ))
            # pytest
            if any("pytest" in f.path for f in repo_map.test_files) or (self.root / "pytest.ini").exists():
                discovered.append(ValidationCommand(
                    name="pytest",
                    command=("pytest",),
                    category="unit_test",
                    confidence=0.8,
                    reason="pytest files or config detected",
                    working_directory=".",
                    risk="low",
                ))
            # mypy
            if (self.root / "mypy.ini").exists() or any("mypy" in f.path for f in repo_map.configuration_files):
                discovered.append(ValidationCommand(
                    name="mypy",
                    command=("mypy", "."),
                    category="type_check",
                    confidence=0.7,
                    reason="mypy config detected",
                    working_directory=".",
                    risk="low",
                ))
            # ruff
            if (self.root / "pyproject.toml").exists() and "ruff" in (self.root / "pyproject.toml").read_text(encoding="utf-8").lower():
                discovered.append(ValidationCommand(
                    name="ruff",
                    command=("ruff", "check", "."),
                    category="lint",
                    confidence=0.7,
                    reason="ruff configured in pyproject.toml",
                    working_directory=".",
                    risk="low",
                ))
            # compileall (basic syntax check)
            if any(f.extension == ".py" for f in repo_map.files):
                 discovered.append(ValidationCommand(
                    name="compileall",
                    command=("python", "-m", "compileall", "-q", "."),
                    category="type_check", # Can be considered a basic syntax/compile check
                    confidence=0.6,
                    reason="Python files detected, basic syntax check",
                    working_directory=".",
                    risk="low",
                ))

        # Other common build/check tools
        if (self.root / "tsconfig.json").exists():
            discovered.append(ValidationCommand(
                name="tsc",
                command=("tsc", "--noEmit"),
                category="type_check",
                confidence=0.9,
                reason="TypeScript project detected",
                working_directory=".",
                risk="low",
            ))

        return discovered

    def _classify_js_command(self, script_name: str, script_cmd: str, repo_map: RepositoryMap) -> tuple[Literal["unit_test", "integration_test", "e2e_test", "type_check", "lint", "build", "format", "destructive", "other"], Literal["low", "medium", "high"], bool]:
        lower_name = script_name.lower()
        lower_cmd = script_cmd.lower()
        category: Literal["unit_test", "integration_test", "e2e_test", "type_check", "lint", "build", "format", "destructive", "other"] = "other"
        risk: Literal["low", "medium", "high"] = "low"
        destructive = False

        # Check for destructive commands first
        if any(pattern.search(lower_cmd) for pattern in _DANGEROUS_COMMAND_PATTERNS):
            category = "destructive"
            risk = "high"
            destructive = True
            return category, risk, destructive

        # Classify by name
        if "test" in lower_name:
            if "e2e" in lower_name or "cypress" in lower_cmd or "playwright" in lower_cmd:
                category = "e2e_test"
                risk = "medium"
            elif "integration" in lower_name:
                category = "integration_test"
                risk = "medium"
            else:
                category = "unit_test"
                risk = "low"
        elif "build" in lower_name or "webpack" in lower_cmd or "vite" in lower_cmd:
            category = "build"
            risk = "medium"
        elif "lint" in lower_name or "eslint" in lower_cmd or "prettier" in lower_cmd:
            category = "lint"
            risk = "low"
        elif "typecheck" in lower_name or "tsc" in lower_cmd:
            category = "type_check"
            risk = "low"
        elif "format" in lower_name:
            category = "format"
            risk = "low"

        # Refine risk based on content
        if "jest" in lower_cmd or "vitest" in lower_cmd:
            if category == "other": category = "unit_test"
            risk = "low" # Unit tests are generally low risk
        if "cypress" in lower_cmd or "playwright" in lower_cmd:
            if category == "other": category = "e2e_test"
            risk = "medium" # E2E tests can interact with external systems

        return category, risk, destructive

    def select_commands(self, task: str, change_impact: ChangeImpact, discovered_commands: list[ValidationCommand]) -> ValidationPlan:
        task_keywords = _name_tokens(task)
        intents = {name for name, pattern in _INTENT_KEYWORDS.items() if pattern.search(task)}
        entities = {name for name, pattern in _ENTITY_KEYWORDS.items() if pattern.search(task)}

        primary_commands: list[CommandSpec] = []
        secondary_commands: list[CommandSpec] = []
        skipped_commands: list[CommandSpec] = []
        reasons: list[str] = []
        overall_risk: Literal["low", "medium", "high"] = "low"

        # Prioritize commands based on task intent and entities
        if "add" in intents or "modify" in intents or "refactor" in intents:
            reasons.append("Task involves adding, modifying, or refactoring code.")
            # Always include type checks and linting for code changes
            for cmd in discovered_commands:
                if cmd.category in {"type_check", "lint", "format"} and not cmd.destructive:
                    primary_commands.append(CommandSpec(cmd.name, cmd.command, cmd.reason, cmd.category, cmd.risk, cmd.destructive))
                    reasons.append(f"Included {cmd.name} ({cmd.category}) for code quality.")
                    if cmd.risk == "medium": overall_risk = "medium"

            # Unit tests for any code change
            for cmd in discovered_commands:
                if cmd.category == "unit_test" and not cmd.destructive:
                    primary_commands.append(CommandSpec(cmd.name, cmd.command, cmd.reason, cmd.category, cmd.risk, cmd.destructive))
                    reasons.append(f"Included {cmd.name} (unit_test) for code changes.")

            # Build commands if relevant
            if "build" in entities or any(t.role in {"create", "modify"} for t in change_impact.targets if t.relationship == "build"):
                for cmd in discovered_commands:
                    if cmd.category == "build" and not cmd.destructive:
                        primary_commands.append(CommandSpec(cmd.name, cmd.command, cmd.reason, cmd.category, cmd.risk, cmd.destructive))
                        reasons.append(f"Included {cmd.name} (build) due to task entities or change impact.")
                        if cmd.risk == "medium": overall_risk = "medium"

            # E2E/Integration tests if navigation/routing/API is affected
            if "navigation" in entities or "route" in entities or "api" in entities or any(t.relationship in {"router", "navigation", "api"} for t in change_impact.targets):
                for cmd in discovered_commands:
                    if cmd.category in {"e2e_test", "integration_test"} and not cmd.destructive:
                        secondary_commands.append(CommandSpec(cmd.name, cmd.command, cmd.reason, cmd.category, cmd.risk, cmd.destructive))
                        reasons.append(f"Included {cmd.name} (e2e/integration_test) due to navigation/routing/API changes.")
                        if cmd.risk == "medium": overall_risk = "medium"

        elif "fix" in intents:
            reasons.append("Task involves fixing an issue.")
            # Prioritize tests related to affected files
            affected_paths = {t.path for t in change_impact.targets if t.role in {"modify", "create"}}
            for cmd in discovered_commands:
                if cmd.category in {"unit_test", "integration_test", "e2e_test"} and not cmd.destructive:
                    # Heuristic: if test command name or script contains keywords from affected paths
                    cmd_keywords = _name_tokens(" ".join(cmd.command))
                    if any(pk in cmd_keywords for path in affected_paths for pk in _name_tokens(path)):
                        primary_commands.append(CommandSpec(cmd.name, cmd.command, cmd.reason, cmd.category, cmd.risk, cmd.destructive))
                        reasons.append(f"Included {cmd.name} (test) as it seems related to affected files.")
                        if cmd.risk == "medium": overall_risk = "medium"
                    else:
                        secondary_commands.append(CommandSpec(cmd.name, cmd.command, cmd.reason, cmd.category, cmd.risk, cmd.destructive)) # Include other tests as secondary

            # Type checks and linting
            for cmd in discovered_commands:
                if cmd.category in {"type_check", "lint", "format"} and not cmd.destructive:
                    primary_commands.append(CommandSpec(cmd.name, cmd.command, cmd.reason, cmd.category, cmd.risk, cmd.destructive))
                    reasons.append(f"Included {cmd.name} ({cmd.category}) for code quality during fix.")
                    if cmd.risk == "medium": overall_risk = "medium"

            # Backend-specific fixes
            if "backend" in entities or any(t.relationship == "backend" for t in change_impact.targets):
                for cmd in discovered_commands:
                    if cmd.category in {"unit_test", "integration_test"} and "backend" in cmd.name.lower() and not cmd.destructive:
                        primary_commands.append(CommandSpec(cmd.name, cmd.command, cmd.reason, cmd.category, cmd.risk, cmd.destructive))
                        reasons.append(f"Included {cmd.name} (backend test) for backend fix.")
                        if cmd.risk == "medium": overall_risk = "medium"

        # General rules for all tasks
        for cmd in discovered_commands:
            if cmd.destructive or cmd.risk == "high":
                skipped_commands.append(CommandSpec(cmd.name, cmd.command, cmd.reason, cmd.category, cmd.risk, cmd.destructive))
                reasons.append(f"Skipped {cmd.name} due to high risk or destructive nature.")
                overall_risk = "high" # If any high risk command is discovered, overall risk is high
            elif cmd.risk == "medium" and overall_risk == "low":
                overall_risk = "medium"

        # Remove duplicates and ensure order
        seen_primary = set()
        unique_primary = []
        for cmd in primary_commands:
            cmd_tuple = (cmd.name, cmd.display())
            if cmd_tuple not in seen_primary:
                unique_primary.append(cmd)
                seen_primary.add(cmd_tuple)
        primary_commands = unique_primary

        seen_secondary = set()
        unique_secondary = []
        for cmd in secondary_commands:
            cmd_tuple = (cmd.name, cmd.display())
            if cmd_tuple not in seen_secondary and cmd_tuple not in seen_primary:
                unique_secondary.append(cmd)
                seen_secondary.add(cmd_tuple)
        secondary_commands = unique_secondary

        return ValidationPlan(
            commands=discovered_commands,
            primary_commands=primary_commands,
            secondary_commands=secondary_commands,
            skipped_commands=skipped_commands,
            reasons=list(set(reasons)), # Deduplicate reasons
            risk_level=overall_risk,
        )