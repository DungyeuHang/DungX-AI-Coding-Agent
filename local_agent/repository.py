from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import os
import posixpath
import re
import tomllib
from pathlib import Path, PurePosixPath

from .git import GitIntegration
from .indexing.javascript_indexer import JavaScriptIndexer
from .indexing.parser import TreeSitterParser
from .indexing.python_indexer import PythonIndexer
from .models import (
    CommandSpec,
    FileIndex,
    FileRelationship,
    ProjectContext,
    RepositoryFile,
    RepositoryMap,
    SemanticIndex,
)
from .storage import JsonFileStorage


IGNORED_DIRECTORIES = {
    ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "env", "__pycache__",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".cache", "dist", "build", "coverage", "target", ".tox",
    ".agent_data", # Agent's own persistence dir; must not pollute the scanned tree
}
SECRET_NAMES = {".env", ".env.local", ".env.production", "credentials.json", "secrets.json", "id_rsa", "id_ed25519", "token.json"}
LOGGER = logging.getLogger(__name__)
LANGUAGES = {
    ".py": "Python", ".js": "JavaScript", ".jsx": "JavaScript", ".mjs": "JavaScript", ".cjs": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript", ".java": "Java", ".kt": "Kotlin", ".go": "Go", ".rs": "Rust",
    ".cs": "C#", ".cpp": "C++", ".c": "C", ".h": "C/C++", ".hpp": "C++", ".rb": "Ruby", ".php": "PHP",
    ".swift": "Swift", ".scala": "Scala", ".dart": "Dart", ".vue": "Vue", ".svelte": "Svelte",
}
SOURCE_EXTENSIONS = set(LANGUAGES)
TEMP_SUFFIXES = {".tmp", ".temp", ".bak", ".swp", ".log"}
GENERATED_MARKERS = ("generated", ".min.", ".bundle.", ".map")
DOC_NAMES = {"readme", "contributing", "changelog", "license"}
_PY_IMPORT_RE = re.compile(r"^\s*(?:from\s+([.\w]+)\s+import|import\s+([\w.]+))", re.MULTILINE)
_JS_IMPORT_RE = re.compile(r"(?:import\s+(?:[^\n;]*?\s+from\s+)?|require\s*\()\s*[\"']([^\"']+)[\"']", re.MULTILINE)
_JS_ROUTE_PATH_RE = re.compile(r"\bpath\s*:\s*[\"']([^\"']+)[\"']")


class GitIgnoreMatcher:
    def __init__(self, root: Path):
        self.patterns: list[tuple[str, bool]] = []
        try:
            lines = (root / ".gitignore").read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            lines = []
        for line in lines:
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            negated = value.startswith("!")
            self.patterns.append((value[1:] if negated else value, negated))

    def matches(self, relative: str, is_directory: bool = False) -> bool:
        value = relative.strip("/").replace("\\", "/")
        ignored = False
        for pattern, negated in self.patterns:
            pattern = pattern.rstrip("/")
            matched = fnmatch.fnmatch(value, pattern) or fnmatch.fnmatch(value + ("/" if is_directory else ""), pattern)
            if "/" not in pattern:
                matched = matched or any(fnmatch.fnmatch(part, pattern) for part in PurePosixPath(value).parts)
            if matched:
                ignored = not negated
        return ignored


class RepositoryIntelligence:
    """Deterministic repository map and relationship scanner; never calls a provider."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise ValueError(f"project directory does not exist: {self.root}")
        self.git = GitIntegration(self.root)
        self.ignore = GitIgnoreMatcher(self.root)
        self.storage = JsonFileStorage(self.root / ".agent_data")
        self.semantic_index_path = "semantic_index.json"
        self._semantic_index: SemanticIndex | None = None
        self._ts_available = False
        try:
            self._parser = TreeSitterParser()
            self._python_indexer = PythonIndexer(self._parser)
            self._javascript_indexer = JavaScriptIndexer(self._parser)
            self._ts_available = True
            LOGGER.info("Tree-sitter parser for Python is available.")
        except (ImportError, OSError) as e:
            LOGGER.warning("Tree-sitter parser for Python not available. Semantic indexing will be disabled. Error: %s", e)

    @property
    def semantic_index(self) -> SemanticIndex | None:
        """
        Returns the current semantic index. Loads from disk if not already in memory.
        Does not trigger a full rescan.
        """
        if self._semantic_index is None:
            self._semantic_index = self.storage.load_semantic_index()
        return self._semantic_index

    def scan(self) -> ProjectContext:
        context = ProjectContext(root=str(self.root))
        records: list[RepositoryFile] = []
        ignored: list[dict[str, str]] = []
        protected: list[dict[str, str]] = []
        directories: set[str] = set()
        existing_semantic_index = self.storage.load_semantic_index()
        current_semantic_index_files: dict[str, FileIndex] = {}
        scanned_paths = set()

        for current, dirnames, names in os.walk(self.root, topdown=True):
            current_path = Path(current)
            current_relative = "" if current_path == self.root else current_path.relative_to(self.root).as_posix()
            kept: list[str] = []
            for name in sorted(dirnames):
                relative = f"{current_relative}/{name}".strip("/")
                reason = self._directory_ignore_reason(name, relative)
                if reason:
                    ignored.append({"path": relative, "reason": reason})
                else:
                    kept.append(name)
            dirnames[:] = kept
            if current_relative:
                directories.add(current_relative)
            for name in sorted(names):
                path = current_path / name
                relative = path.relative_to(self.root).as_posix()
                scanned_paths.add(relative)
                if self._is_secret(name):
                    protected.append({"path": relative, "reason": "protected secret-like file"})
                    continue
                reason = self._file_ignore_reason(path, relative)
                if reason:
                    ignored.append({"path": relative, "reason": reason})
                    continue
                record = self._record(path, relative)
                if record is None:
                    ignored.append({"path": relative, "reason": "binary or unreadable file"})
                    continue
                records.append(record)
                self._classify(relative, context)
                try:
                    content_hash = self._calculate_sha256(path)
                    existing_file_index = existing_semantic_index.files.get(relative)
                    if existing_file_index and existing_file_index.content_hash == content_hash:
                        current_semantic_index_files[relative] = existing_file_index
                    else:
                        file_index = FileIndex(path=relative, language=record.language, content_hash=content_hash)
                        if record.language == "Python" and self._ts_available:
                            file_index = self._index_python_file(path, relative, content_hash)
                        elif record.language in {"JavaScript", "TypeScript"} and self._ts_available:
                            file_index = self._index_javascript_file(path, relative, content_hash)
                        current_semantic_index_files[relative] = file_index
                except OSError as e:
                    LOGGER.warning("Could not process file for semantic index %s: %s", relative, e)

        indexed_paths = set(existing_semantic_index.files.keys())
        deleted_paths = indexed_paths - scanned_paths
        for path in deleted_paths:
            LOGGER.debug("Removing deleted file from semantic index: %s", path)
        self._semantic_index = SemanticIndex(files=current_semantic_index_files)
        self.storage.save_semantic_index(self._semantic_index)
        context.directories = sorted(directories)
        context.git_status = self.git.status()
        metadata = self._metadata(records)
        context.metadata.update(metadata)
        context.metadata.update({"is_git_repository": self.git.is_repository(), "git_branch": self.git.branch(), "git_log": self.git.log()})
        context.metadata["file_previews"] = self._previews(context)
        context.validation_commands = self._validation_commands(context)
        relationships = self._relationships(records)
        context.repository_map = RepositoryMap(
            root=str(self.root), project_metadata={key: value for key, value in metadata.items() if key != "file_previews"},
            languages=sorted({item.language for item in records if item.language}), frameworks=sorted(metadata.get("frameworks", [])),
            files=sorted(records, key=lambda item: item.path), directories=context.directories, tests=sorted(context.test_files),
            configuration_files=sorted(set(context.config_files + context.dependency_files + context.build_files)),
            entry_points=sorted(item.path for item in records if item.is_entry_point), relationships=relationships,
            ignored_paths=sorted(ignored, key=lambda item: item["path"]), protected_paths=sorted(protected, key=lambda item: item["path"]),
        )
        context.metadata["semantic_index"] = self._semantic_index
        context.metadata["scan_statistics"] = {"scanned_files": len(records), "ignored_paths": len(ignored), "protected_paths": len(protected), "relationships": len(relationships)}
        return context

    def _calculate_sha256(self, path: Path) -> str:
        h = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            return h.hexdigest()
        except OSError as e:
            LOGGER.warning("Could not calculate hash for %s: %s", path, e)
            return ""

    def _directory_ignore_reason(self, name: str, relative: str) -> str | None:
        if name in IGNORED_DIRECTORIES:
            return "ignored directory"
        if self.ignore.matches(relative, is_directory=True):
            return ".gitignore rule"
        return None

    def _file_ignore_reason(self, path: Path, relative: str) -> str | None:
        name = path.name.lower()
        if path.is_symlink():
            return "symbolic link not scanned"
        if self.ignore.matches(relative):
            return ".gitignore rule"
        if name.startswith("~") or path.suffix.lower() in TEMP_SUFFIXES:
            return "temporary file"
        if self._is_generated(name):
            return "generated file"
        try:
            with path.open("rb") as handle:
                if b"\x00" in handle.read(4096):
                    return "binary file"
        except OSError:
            return "unreadable file"
        return None

    def _record(self, path: Path, relative: str) -> RepositoryFile | None:
        try:
            size = path.stat().st_size
        except OSError:
            return None
        suffix = path.suffix.lower()
        name = path.name.lower()
        return RepositoryFile(
            path=relative, extension=suffix, size_bytes=size, line_count=self._line_count(path), language=LANGUAGES.get(suffix, ""),
            is_test=self._is_test(relative), is_configuration=self._is_configuration(name),
            is_documentation=name.startswith("readme") or name.split(".")[0] in DOC_NAMES or suffix in {".md", ".rst", ".txt"},
            is_generated=self._is_generated(name), is_entry_point=self._is_entry_point(name),
        )

    def _index_python_file(self, path: Path, relative_path: str, content_hash: str) -> FileIndex:
        """Use PythonIndexer to create a FileIndex for a Python file."""
        file_index = FileIndex(path=relative_path, language="Python", content_hash=content_hash)
        try:
            content = path.read_bytes()
            symbols, imports = self._python_indexer.index(content)
            file_index.symbols = symbols
            file_index.imports = imports
        except Exception as e:
            LOGGER.warning("Failed to index Python file %s: %s", relative_path, e)
        return file_index

    def _index_javascript_file(self, path: Path, relative_path: str, content_hash: str) -> FileIndex:
        """Use JavaScriptIndexer for JavaScript, JSX, TypeScript, or TSX files."""
        file_index = FileIndex(path=relative_path, language=LANGUAGES.get(path.suffix.lower(), "JavaScript"), content_hash=content_hash)
        try:
            content = path.read_bytes()
            parser_language = "tsx" if path.suffix.lower() == ".tsx" else "typescript" if path.suffix.lower() == ".ts" else "javascript"
            symbols, imports = self._javascript_indexer.index(content, parser_language)
            file_index.symbols = symbols
            file_index.imports = imports
        except Exception as e:
            LOGGER.warning("Failed to index JavaScript/TypeScript file %s: %s", relative_path, e)
        return file_index

    @staticmethod
    def _line_count(path: Path) -> int | None:
        try:
            count = 0
            last = b""
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(65536), b""):
                    count += chunk.count(b"\n")
                    last = chunk[-1:]
            return count + (1 if last and last != b"\n" else 0)
        except OSError:
            return None

    @staticmethod
    def _is_secret(name: str) -> bool:
        lower = name.lower()
        return lower in SECRET_NAMES or lower == ".env" or lower.startswith(".env.") or lower.endswith((".pem", ".key", ".p12", ".pfx")) or lower in {"credentials", "credentials.txt", "client_secret.json"}

    @staticmethod
    def _is_generated(name: str) -> bool:
        return any(marker in name for marker in GENERATED_MARKERS) or name.endswith((".generated.py", ".generated.ts", ".generated.js"))

    @staticmethod
    def _is_test(relative: str) -> bool:
        path = PurePosixPath(relative)
        name = path.name.lower()
        return name.startswith("test_") or name.endswith(("_test.py", ".test.js", ".test.ts", ".test.tsx", ".spec.js", ".spec.ts", ".spec.tsx")) or "tests" in path.parts

    @staticmethod
    def _is_entry_point(name: str) -> bool:
        return name in {"main.py", "__main__.py", "app.py", "cli.py", "index.js", "index.ts", "main.js", "main.ts", "server.js", "server.ts", "manage.py", "program.cs"}

    @staticmethod
    def _is_configuration(name: str) -> bool:
        return name in {".gitignore", ".editorconfig", "pyproject.toml", "setup.cfg", "tox.ini", "package.json", "tsconfig.json", "vite.config.js", "vite.config.ts", "requirements.txt", "go.mod", "cargo.toml"}

    def _classify(self, relative: str, context: ProjectContext) -> None:
        path = Path(relative)
        name = path.name.lower()
        if path.suffix.lower() in SOURCE_EXTENSIONS:
            context.source_files.append(relative)
        if self._is_test(relative):
            context.test_files.append(relative)
        if name.startswith("readme") or name.split(".")[0] in DOC_NAMES or path.suffix.lower() in {".md", ".rst"}:
            context.documentation_files.append(relative)
        if self._is_configuration(name) or name in {".pre-commit-config.yaml", ".pre-commit-config.yml", "mypy.ini", "pyrightconfig.json"}:
            context.config_files.append(relative)
        if name in {"package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "requirements.txt", "poetry.lock", "pipfile", "pipfile.lock", "go.mod", "go.sum", "cargo.toml", "cargo.lock", "pom.xml", "build.gradle", "build.gradle.kts", "gemfile", "composer.json"}:
            context.dependency_files.append(relative)
        if name in {"dockerfile", "makefile", "justfile", "pyproject.toml", "package.json", "vite.config.js", "vite.config.ts", "webpack.config.js", "tsconfig.json", "cargo.toml", "pom.xml", "build.gradle", "build.gradle.kts"}:
            context.build_files.append(relative)
        if any(token in name for token in ("lint", "eslint", "ruff", "flake8", "pylint", "prettier")):
            context.lint_files.append(relative)
        if name in {"mypy.ini", "pyrightconfig.json", "tsconfig.json"} or "typecheck" in name or "type-check" in name:
            context.typecheck_files.append(relative)

    def _metadata(self, records: list[RepositoryFile]) -> dict[str, object]:
        names = {item.path.lower() for item in records}
        stacks: list[str] = []
        frameworks: set[str] = set()
        if any(path.endswith(("pyproject.toml", "requirements.txt", "setup.py")) for path in names):
            stacks.append("Python")
        if "package.json" in {PurePosixPath(path).name for path in names}:
            stacks.append("JavaScript/TypeScript")
        if any(path.endswith("go.mod") for path in names):
            stacks.append("Go")
        if any(path.endswith("cargo.toml") for path in names):
            stacks.append("Rust")
        if any(path.endswith(("pom.xml", "build.gradle", "build.gradle.kts")) for path in names):
            stacks.append("JVM")
        if any(path.endswith(".csproj") for path in names):
            stacks.append(".NET")
        pyproject = next((self.root / item.path for item in records if item.path == "pyproject.toml"), None)
        if pyproject:
            try:
                data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
                project = data.get("project", {})
                dependencies = json.dumps(data).lower()
                frameworks.update(item for item in ("django", "flask", "fastapi", "pytest") if item in dependencies)
                metadata: dict[str, object] = {"project_name": project.get("name"), "python_requires": project.get("requires-python")}
            except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
                metadata = {"pyproject_parse_error": True}
        else:
            metadata = {}
        package = next((self.root / item.path for item in records if item.path == "package.json"), None)
        if package:
            try:
                data = json.loads(package.read_text(encoding="utf-8"))
                metadata.update({"package_name": data.get("name"), "package_scripts": sorted(data.get("scripts", {}).keys())})
                metadata["package_dependencies"] = sorted(set(data.get("dependencies", {})) | set(data.get("devDependencies", {})))
                dependencies = json.dumps(data).lower()
                frameworks.update(item for item in ("react", "next", "vite", "express", "vue") if item in dependencies)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                metadata["package_json_parse_error"] = True
        metadata.update({"stacks": stacks or ["Unknown"], "frameworks": sorted(frameworks), "file_count": len(records)})
        return metadata

    def _previews(self, context: ProjectContext) -> dict[str, str]:
        result: dict[str, str] = {}
        total = 0
        candidates = context.documentation_files[:2] + context.config_files[:8] + context.dependency_files[:8] + context.test_files[:8] + context.source_files[:12]
        for relative in candidates:
            if relative in result:
                continue
            try:
                content = (self.root / relative).read_text(encoding="utf-8")[:3000]
            except (OSError, UnicodeDecodeError):
                continue
            if total + len(content) > 24000:
                break
            result[relative] = content
            total += len(content)
        return result

    def _relationships(self, records: list[RepositoryFile]) -> list[FileRelationship]:
        paths = {item.path for item in records}
        relationships: set[tuple[str, str, str]] = set()
        contents: dict[str, str] = {}
        for record in records:
            try:
                content = (self.root / record.path).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            contents[record.path] = content
            imports: list[str] = []
            if record.extension == ".py":
                imports = [match.group(1) or match.group(2) for match in _PY_IMPORT_RE.finditer(content)]
                for module in imports:
                    target = self._resolve_python(record.path, module, paths)
                    if target and target != record.path:
                        relationships.add((record.path, target, "imports"))
            elif record.extension in {".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"}:
                imports = [match.group(1) for match in _JS_IMPORT_RE.finditer(content)]
                for module in imports:
                    target = self._resolve_javascript(record.path, module, paths)
                    if target and target != record.path:
                        relationships.add((record.path, target, "imports"))
                        source_content = content
                        target_record = next((item for item in records if item.path == target), None)
                        target_name = PurePosixPath(target).stem
                        if self._is_router(record.path, source_content):
                            relationships.add((record.path, target, "router_route" if self._is_page(target, target_record) else "router_uses"))
                        if self._is_layout(record.path, source_content):
                            relationships.add((record.path, target, "layout_uses"))
                        if self._is_page(record.path, target_record=None) and self._is_shared_component(target):
                            relationships.add((record.path, target, "page_uses"))
                        if self._is_navigation(record.path, source_content):
                            relationships.add((record.path, target, "navigation_uses"))
                        if target_name and target_name in source_content and self._is_router(record.path, source_content):
                            relationships.add((record.path, target, "router_route"))
        sources = [item.path for item in records if not item.is_test]
        tests = [item.path for item in records if item.is_test]
        for source in sources:
            stem = PurePosixPath(source).stem.lower()
            for test in tests:
                name = PurePosixPath(test).name.lower()
                if stem in name or name.removeprefix("test_").startswith(stem) or name.startswith(stem + "."):
                    relationships.add((source, test, "tested_by"))
        router_paths = [path for path, content in contents.items() if self._is_router(path, content)]
        navigation_paths = [path for path, content in contents.items() if self._is_navigation(path, content)]
        for navigation in navigation_paths:
            for router in router_paths:
                if navigation != router:
                    relationships.add((navigation, router, "navigation_routes"))
        return [FileRelationship(*item) for item in sorted(relationships)]

    @staticmethod
    def _is_router(relative: str, content: str) -> bool:
        lower = relative.lower()
        return "router" in PurePosixPath(relative).stem.lower() or any(token in content for token in ("createBrowserRouter", "createHashRouter", "useRoutes", "<Route"))

    @staticmethod
    def _is_layout(relative: str, content: str) -> bool:
        return "layout" in relative.lower() or "<Outlet" in content or "Outlet" in content

    @staticmethod
    def _is_navigation(relative: str, content: str) -> bool:
        lower = relative.lower()
        return any(token in lower for token in ("navigation", "navbar", "sidebar", "nav")) or any(token in content for token in ("<NavLink", "useNavigate", "navItems"))

    @staticmethod
    def _is_page(relative: str, target_record: RepositoryFile | None) -> bool:
        lower = relative.lower()
        return "/pages/" in f"/{lower}/" or lower.endswith("page.tsx") or lower.endswith("page.ts") or (target_record is not None and target_record.is_test is False and "page" in PurePosixPath(relative).stem.lower())

    @staticmethod
    def _is_shared_component(relative: str) -> bool:
        lower = relative.lower()
        return "/shared/" in f"/{lower}/" or "/components/" in f"/{lower}/"

    @staticmethod
    def _resolve_python(source: str, module: str, paths: set[str]) -> str | None:
        if module.startswith("."):
            dots = len(module) - len(module.lstrip("."))
            base = list(PurePosixPath(source).parent.parts)
            base = base[:max(0, len(base) - dots + 1)]
            module = module[dots:]
            parts = base + ([part for part in module.split(".") if part] if module else [])
        else:
            parts = module.split(".")
        stem = "/".join(parts)
        return next((candidate for candidate in (stem + ".py", stem + "/__init__.py") if candidate in paths), None)

    @staticmethod
    def _resolve_javascript(source: str, module: str, paths: set[str]) -> str | None:
        if not module.startswith("."):
            return None
        raw = posixpath.normpath((PurePosixPath(source).parent / module).as_posix())
        if raw.startswith("../"):
            return None
        candidates = [raw] + [raw + extension for extension in (".js", ".jsx", ".ts", ".tsx")] + [raw + "/index.js", raw + "/index.ts"]
        return next((candidate for candidate in candidates if candidate in paths), None)

    def _validation_commands(self, context: ProjectContext) -> list[CommandSpec]:
        names = {Path(path).name.lower() for path in context.dependency_files + context.config_files + context.build_files}
        commands: list[CommandSpec] = []
        if {"pyproject.toml", "requirements.txt", "setup.py"} & names:
            if context.test_files:
                commands.append(CommandSpec("tests", ("python", "-m", "unittest", "discover"), "Python tests detected"))
            if context.typecheck_files:
                commands.append(CommandSpec("typecheck", ("python", "-m", "mypy", "."), "Python type-check configuration detected"))
            if context.lint_files:
                commands.append(CommandSpec("lint", ("python", "-m", "ruff", "check", "."), "Python lint configuration detected"))
        package = self.root / "package.json"
        if package.exists():
            try:
                scripts = json.loads(package.read_text(encoding="utf-8")).get("scripts", {})
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
