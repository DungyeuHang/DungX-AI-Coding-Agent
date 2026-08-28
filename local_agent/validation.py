from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Literal

from .models import (
    ChangeImpact,
    CommandSpec,
    FailureAnalysis,
    Plan,
    ProjectContext,
    ProviderCapability,
    ProviderError,
    RepositoryMap,
    ReviewResult,
    RunReport,
    ToolCall,
    ToolDefinition,
    ToolExecutionPolicy,
    ToolResult,
    ValidationCommand,
    ValidationPlan,
)
from .providers import AIProvider
from .tool_engine import ToolEngine
from .tools import ToolRegistry

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
    re.compile(r"\b(db:migrate:reset|db:drop|migrate:rollback|db:reset|prisma\s+migrate\s+reset|migrate\s+reset)\b", re.I), # Rails/Prisma-like
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
            if any("pytest" in path for path in repo_map.tests) or (self.root / "pytest.ini").exists() or (self.root / "requirements.txt").exists() and "pytest" in (self.root / "requirements.txt").read_text(encoding="utf-8").lower() or any("test" in path for path in repo_map.tests):
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
            if (self.root / "mypy.ini").exists() or any("mypy" in (f.path if hasattr(f, "path") else str(f)) for f in repo_map.configuration_files):
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
        if any(pattern.search(lower_cmd) or pattern.search(lower_name) for pattern in _DANGEROUS_COMMAND_PATTERNS):
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
        entities = {name for name, keywords in _ENTITY_KEYWORDS.items() if keywords & task_keywords}

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
            if "build" in entities or "page" in entities or "component" in entities or any(t.role in {"create", "modify"} for t in change_impact.targets if t.relationship in {"build", "page", "component", "router", "navigation"}):
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
            # Type checks and linting
            for cmd in discovered_commands:
                if cmd.category in {"type_check", "lint", "format"} and not cmd.destructive:
                    primary_commands.append(CommandSpec(cmd.name, cmd.command, cmd.reason, cmd.category, cmd.risk, cmd.destructive))
                    reasons.append(f"Included {cmd.name} ({cmd.category}) for code quality during fix.")
                    if cmd.risk == "medium": overall_risk = "medium"

            # Unit tests for fix
            for cmd in discovered_commands:
                if cmd.category == "unit_test" and not cmd.destructive:
                    primary_commands.append(CommandSpec(cmd.name, cmd.command, cmd.reason, cmd.category, cmd.risk, cmd.destructive))
                    reasons.append(f"Included {cmd.name} (unit_test) for fix.")

            # Build commands
            for cmd in discovered_commands:
                if cmd.category == "build" and not cmd.destructive:
                    primary_commands.append(CommandSpec(cmd.name, cmd.command, cmd.reason, cmd.category, cmd.risk, cmd.destructive))

            # E2E/Integration tests if navigation/routing/API/auth is affected
            if "navigation" in entities or "route" in entities or "api" in entities or "auth" in entities:
                for cmd in discovered_commands:
                    if cmd.category in {"e2e_test", "integration_test"} and not cmd.destructive:
                        secondary_commands.append(CommandSpec(cmd.name, cmd.command, cmd.reason, cmd.category, cmd.risk, cmd.destructive))

            # Backend-specific fixes
            if "backend" in entities or any(t.relationship == "backend" for t in change_impact.targets):
                for cmd in discovered_commands:
                    if cmd.category in {"unit_test", "integration_test"} and "backend" in cmd.name.lower() and not cmd.destructive:
                        primary_commands.append(CommandSpec(cmd.name, cmd.command, cmd.reason, cmd.category, cmd.risk, cmd.destructive))
                        reasons.append(f"Included {cmd.name} (backend test) for backend fix.")
                        if cmd.risk == "medium": overall_risk = "medium"

        # General rules for all tasks
        if not primary_commands and discovered_commands:
            for cmd in discovered_commands:
                if not getattr(cmd, "destructive", False) and getattr(cmd, "risk", "low") != "high":
                    primary_commands.append(CommandSpec(cmd.name, cmd.command, getattr(cmd, "reason", "configured validation command"), getattr(cmd, "category", "unit_test"), getattr(cmd, "risk", "low"), getattr(cmd, "destructive", False)))

        for cmd in discovered_commands:
            if getattr(cmd, "destructive", False) or getattr(cmd, "risk", "low") == "high":
                skipped_commands.append(CommandSpec(cmd.name, cmd.command, getattr(cmd, "reason", ""), getattr(cmd, "category", "other"), getattr(cmd, "risk", "high"), getattr(cmd, "destructive", True)))
                reasons.append(f"Skipped {cmd.name} due to high risk or destructive nature.")
                overall_risk = "high" # If any high risk command is discovered, overall risk is high
            elif getattr(cmd, "risk", "low") == "medium" and overall_risk == "low":
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

    def discover_targeted_commands(
        self,
        changed_files: list[str],
        repo_map: RepositoryMap | None = None,
    ) -> list[CommandSpec]:
        """Discover targeted test commands for the specific files modified during implementation."""
        if not changed_files:
            return []

        targeted: list[CommandSpec] = []
        seen_cmds: set[tuple[str, ...]] = set()

        for changed in changed_files:
            p = Path(changed)
            posix_path = p.as_posix()
            stem = p.stem
            ext = p.suffix.lower()

            is_test_file = (
                "test" in posix_path.lower()
                or stem.startswith("test_")
                or stem.endswith("_test")
                or stem.endswith(".test")
                or stem.endswith(".spec")
            )
            candidate_test_paths: list[str] = []
            if is_test_file:
                candidate_test_paths.append(posix_path)
            else:
                if ext == ".py":
                    candidate_test_paths.extend([
                        f"tests/test_{stem}.py",
                        f"test/test_{stem}.py",
                        f"tests/{stem}_test.py",
                        f"tests/unit/test_{stem}.py",
                        f"test_{stem}.py",
                        f"{p.parent.as_posix()}/test_{stem}.py" if p.parent.as_posix() != "." else f"test_{stem}.py",
                    ])
                elif ext in {".ts", ".tsx", ".js", ".jsx"}:
                    candidate_test_paths.extend([
                        f"{p.parent.as_posix()}/{stem}.test{ext}" if p.parent.as_posix() != "." else f"{stem}.test{ext}",
                        f"{p.parent.as_posix()}/{stem}.spec{ext}" if p.parent.as_posix() != "." else f"{stem}.spec{ext}",
                        f"tests/{stem}.test{ext}",
                        f"tests/{stem}.test.ts",
                        f"tests/{stem}.test.js",
                        f"test/{stem}.test.ts",
                        f"test/{stem}.test.js",
                    ])

            for candidate in candidate_test_paths:
                candidate_path = self.root / candidate
                if candidate_path.exists() and candidate_path.is_file():
                    try:
                        rel_candidate = candidate_path.relative_to(self.root).as_posix()
                    except ValueError:
                        rel_candidate = Path(candidate).as_posix()

                    clean_stem = Path(rel_candidate).stem
                    if clean_stem.endswith(".test") or clean_stem.endswith(".spec"):
                        clean_stem = Path(clean_stem).stem

                    if ext == ".py" or rel_candidate.endswith(".py"):
                        cmd_tuple = ("pytest", rel_candidate)
                        if cmd_tuple not in seen_cmds:
                            seen_cmds.add(cmd_tuple)
                            targeted.append(
                                CommandSpec(
                                    name=f"targeted_pytest_{clean_stem}",
                                    command=cmd_tuple,
                                    reason=f"Targeted test for {changed}",
                                    category="unit_test",
                                    risk="low",
                                    destructive=False,
                                )
                            )
                    elif ext in {".ts", ".tsx", ".js", ".jsx"} or any(rel_candidate.endswith(x) for x in [".ts", ".tsx", ".js", ".jsx"]):
                        cmd_tuple = ("npm", "test", "--", rel_candidate)
                        if cmd_tuple not in seen_cmds:
                            seen_cmds.add(cmd_tuple)
                            targeted.append(
                                CommandSpec(
                                    name=f"targeted_test_{clean_stem}",
                                    command=cmd_tuple,
                                    reason=f"Targeted test for {changed}",
                                    category="unit_test",
                                    risk="low",
                                    destructive=False,
                                )
                            )

        return targeted

    def analyze_verification_gap(
        self,
        changed_files: list[str],
        exported_symbols: list[Any],
        context: ProjectContext,
        targeted_commands: list[CommandSpec] | None = None,
    ) -> Any | None:
        """Analyzes whether changed files / exported symbols have targeted verification coverage."""
        from .test_synthesizer import VerificationGapAnalyzer
        return VerificationGapAnalyzer(self.root).analyze(
            changed_files,
            exported_symbols,
            context,
            targeted_commands=targeted_commands,
        )


@dataclass
class VerificationResult:
    verified: bool = True
    notes: str = ""
    targeted_commands: list[str] = field(default_factory=list)


class VerificationIntelligence:
    """Coordinates tool-assisted verification and assertion inspection prior to full validation."""

    def __init__(
        self,
        provider: AIProvider,
        registry: ToolRegistry | None = None,
        policy: ToolExecutionPolicy | None = None,
    ):
        self.provider = provider
        self.registry = registry
        self.policy = policy

    def verify(
        self,
        task: str,
        plan: Plan,
        context: ProjectContext,
        diff: str,
        changed_files: list[str],
        initial_history: list[tuple[ToolCall, ToolResult]] | None = None,
        report: RunReport | None = None,
    ) -> VerificationResult:
        provider_caps = getattr(self.provider, "capabilities", None)
        if (
            isinstance(provider_caps, (set, frozenset))
            and ProviderCapability.TOOL_USE in provider_caps
            and self.registry is not None
            and hasattr(self.provider, "verify_changes_with_tools")
        ):
            captured_result: list[VerificationResult] = []

            def _verify_step(
                task: str,
                plan: Plan,
                context: ProjectContext,
                tools: list[ToolDefinition],
                tool_history: list[tuple[ToolCall, ToolResult]],
                failure: FailureAnalysis | None = None,
                review: ReviewResult | None = None,
            ) -> Any:
                resp = self.provider.verify_changes_with_tools(
                    task=task,
                    plan=plan,
                    context=context,
                    diff=diff,
                    changed_files=changed_files,
                    tools=tools,
                    tool_history=tool_history,
                )
                if isinstance(resp, ToolCall):
                    return resp
                if isinstance(resp, (dict, VerificationResult)):
                    if isinstance(resp, dict):
                        res = VerificationResult(
                            verified=bool(resp.get("verified", True)),
                            notes=str(resp.get("notes", "")),
                            targeted_commands=[str(c) for c in resp.get("targeted_commands", []) if isinstance(c, str)],
                        )
                    else:
                        res = resp
                    captured_result.append(res)
                    return [res]  # Signal completion to ToolEngine
                raise ProviderError(f"Unexpected response from verify_changes_with_tools: {type(resp)}")

            engine = ToolEngine(provider=_verify_step, registry=self.registry, policy=self.policy)
            result = engine.run(
                task=f"Verify changes for: {task}",
                plan=plan,
                context=context,
                initial_history=initial_history,
            )

            if report is not None and result.metrics.total_calls > 0:
                report.tool_metrics.append(result.metrics)
                report.tool_history = list(result.tool_history)

            if captured_result:
                return captured_result[0]

            return VerificationResult(verified=True, notes="Tool verification concluded")

        return VerificationResult(verified=True, notes="Standard verification")
