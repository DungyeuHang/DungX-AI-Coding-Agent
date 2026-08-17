from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path

from .git import GitIntegration
from .models import CommandSpec, ProjectContext
from .repository import RepositoryIntelligence


IGNORED_DIRECTORIES = {
    ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "env", "__pycache__",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build", "target", ".tox",
}
SECRET_NAMES = {".env", ".env.local", ".env.production", "credentials.json", "secrets.json", "id_rsa"}
SOURCE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".kt", ".go", ".rs", ".cs", ".cpp", ".c",
    ".h", ".hpp", ".rb", ".php", ".swift", ".scala", ".dart", ".vue", ".svelte",
}
DOC_NAMES = {"readme", "contributing", "changelog", "license"}


class RepositoryAnalyzer:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise ValueError(f"project directory does not exist: {self.root}")
        self.git = GitIntegration(self.root)

    def analyze(self) -> ProjectContext:
        return RepositoryIntelligence(self.root).scan()

    def _file_previews(self, context: ProjectContext) -> dict[str, str]:
        previews: dict[str, str] = {}
        total = 0
        candidates = context.documentation_files[:2] + context.config_files[:8] + context.dependency_files[:8] + context.test_files[:8] + context.source_files[:12]
        for relative in candidates:
            if relative in previews:
                continue
            try:
                content = (self.root / relative).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            content = content[:3000]
            if total + len(content) > 24000:
                break
            previews[relative] = content
            total += len(content)
        return previews

    def _classify(self, relative: str, context: ProjectContext) -> None:
        path = Path(relative)
        name = path.name.lower()
        suffix = path.suffix.lower()
        if suffix in SOURCE_EXTENSIONS:
            context.source_files.append(relative)
        if name.startswith("test_") or name.endswith(("_test.py", ".test.js", ".test.ts", ".spec.js", ".spec.ts")) or "tests" in path.parts:
            context.test_files.append(relative)
        if name.startswith("readme") or name.split(".")[0] in DOC_NAMES:
            context.documentation_files.append(relative)
        if name in {"pyproject.toml", "setup.cfg", "tox.ini", ".editorconfig", ".pre-commit-config.yaml", ".pre-commit-config.yml"}:
            context.config_files.append(relative)
        if name in {"package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "requirements.txt", "poetry.lock", "pipfile", "pipfile.lock", "go.mod", "go.sum", "cargo.toml", "cargo.lock", "pom.xml", "build.gradle", "build.gradle.kts", "gemfile", "composer.json"}:
            context.dependency_files.append(relative)
        if name in {"dockerfile", "makefile", "justfile", "pyproject.toml", "package.json", "vite.config.js", "vite.config.ts", "webpack.config.js", "tsconfig.json", "cargo.toml", "pom.xml", "build.gradle", "build.gradle.kts"}:
            context.build_files.append(relative)
        if any(token in name for token in ("lint", "eslint", "ruff", "flake8", "pylint", "prettier")) or name in {".pre-commit-config.yaml", ".pre-commit-config.yml"}:
            context.lint_files.append(relative)
        if name in {"mypy.ini", "pyrightconfig.json", "tsconfig.json"} or "typecheck" in name or "type-check" in name:
            context.typecheck_files.append(relative)

    def _detect_metadata(self, files: list[Path]) -> dict[str, object]:
        names = {path.name.lower() for path in files}
        stacks: list[str] = []
        if "pyproject.toml" in names or "requirements.txt" in names or "setup.py" in names:
            stacks.append("Python")
        if "package.json" in names:
            stacks.append("JavaScript/TypeScript")
        if "go.mod" in names:
            stacks.append("Go")
        if "cargo.toml" in names:
            stacks.append("Rust")
        if "pom.xml" in names or "build.gradle" in names or "build.gradle.kts" in names:
            stacks.append("JVM")
        if any(path.suffix.lower() == ".csproj" for path in files):
            stacks.append(".NET")
        metadata: dict[str, object] = {"stacks": stacks or ["Unknown"], "file_count": len(files)}
        pyproject = next((path for path in files if path.name == "pyproject.toml"), None)
        if pyproject:
            try:
                with pyproject.open("rb") as handle:
                    data = tomllib.load(handle)
                project = data.get("project", {})
                metadata["project_name"] = project.get("name")
                metadata["python_requires"] = project.get("requires-python")
            except (OSError, tomllib.TOMLDecodeError):
                metadata["pyproject_parse_error"] = True
        package = next((path for path in files if path.name == "package.json"), None)
        if package:
            try:
                data = json.loads(package.read_text(encoding="utf-8"))
                metadata["package_name"] = data.get("name")
                metadata["package_scripts"] = sorted(data.get("scripts", {}).keys())
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                metadata["package_json_parse_error"] = True
        return metadata

    def _validation_commands(self, context: ProjectContext) -> list[CommandSpec]:
        names = {Path(path).name.lower() for path in context.dependency_files + context.config_files + context.build_files}
        commands: list[CommandSpec] = []
        if "pyproject.toml" in names or "requirements.txt" in names or "setup.py" in names:
            if context.test_files:
                commands.append(CommandSpec("tests", ("python", "-m", "unittest", "discover"), "Python tests detected"))
            if context.typecheck_files:
                commands.append(CommandSpec("typecheck", ("python", "-m", "mypy", "."), "Python type-check configuration detected"))
            if context.lint_files:
                commands.append(CommandSpec("lint", ("python", "-m", "ruff", "check", "."), "Python lint configuration detected"))
        package_path = self.root / "package.json"
        if package_path.exists():
            try:
                scripts = json.loads(package_path.read_text(encoding="utf-8")).get("scripts", {})
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                scripts = {}
            for key in ("test", "lint", "typecheck", "type-check", "build"):
                if key in scripts:
                    commands.append(CommandSpec(key, ("npm", "run", key), f"package.json script '{key}'"))
        if "go.mod" in names:
            commands.append(CommandSpec("tests", ("go", "test", "./..."), "Go module detected"))
        if "cargo.toml" in names:
            commands.append(CommandSpec("tests", ("cargo", "test"), "Cargo manifest detected"))
        if "pom.xml" in names:
            commands.append(CommandSpec("tests", ("mvn", "test"), "Maven project detected"))
        return commands
