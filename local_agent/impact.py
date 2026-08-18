from __future__ import annotations

import re
from pathlib import Path

from .models import ChangeImpact, ChangeTarget, ProjectContext

_INTENTS = {
    "add": re.compile(r"\b(add|create|implement|introduce|build)\b", re.I),
    "modify": re.compile(r"\b(modify|change|update|alter|adjust|set)\b", re.I),
    "fix": re.compile(r"\b(fix|repair|resolve|correct|debug)\b", re.I),
    "remove": re.compile(r"\b(remove|delete|drop)\b", re.I),
    "refactor": re.compile(r"\b(refactor|restructure|reorganize|cleanup)\b", re.I),
    "test": re.compile(r"\b(test|tests|testing)\b", re.I),
}
_ENTITIES = {
    "page": re.compile(r"\b(page|screen)\b", re.I),
    "component": re.compile(r"\b(component|widget|element)\b", re.I),
    "route": re.compile(r"\b(route|routing|path)\b", re.I),
    "navigation": re.compile(r"\b(navigation|nav|menu|link)\b", re.I),
    "backend": re.compile(r"\b(backend|server|firebase|service)\b", re.I),
    "auth": re.compile(r"\b(auth|authentication|login|signin|signup)\b", re.I),
    "dependency": re.compile(r"\b(dependency|package|library)\b", re.I),
}
_RISK_PATTERNS = {
    "high": re.compile(r"package\.json|auth|security|database|schema|build|config", re.I),
    "medium": re.compile(r"router|layout|shared|provider", re.I),
}
_PAGE_NAME_RE = re.compile(r"\b(?:add|create|implement|introduce|build)\s+(?:an?\s+)?([A-Za-z][A-Za-z0-9_-]*)\s+(?:page|screen)\b", re.I)
_STOPWORDS = {"a", "an", "and", "the", "in", "it", "page", "screen", "fix", "the"}


def _name_tokens(value: str) -> set[str]:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    value = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", value)
    return {token.lower() for token in re.findall(r"[A-Za-z0-9]+", value)}


class ChangeImpactAnalyzer:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def analyze(self, task: str, context: ProjectContext) -> ChangeImpact:
        intents = {name for name, pattern in _INTENTS.items() if pattern.search(task)}
        entities = {name for name, pattern in _ENTITIES.items() if pattern.search(task)}
        targets: dict[str, ChangeTarget] = {}

        if "add" in intents and "page" in entities:
            self._apply_new_page_rules(task, context, targets, entities)
        elif "fix" in intents and "backend" in entities:
            self._apply_backend_fix_rules(task, context, targets)
        elif "fix" in intents:
            self._apply_generic_fix_rules(task, context, targets)
        if "auth" in entities:
            self._apply_auth_rules(context, targets)
        if "dependency" in entities:
            self._apply_dependency_rules(context, targets)

        for path in context.metadata.get("selected_files", []):
            if path not in targets:
                targets[path] = ChangeTarget(path, "modify", 0.5, "Selected by context retriever as relevant to the task.", risk=self._get_risk(path))

        summary = self._summarize(intents, entities, targets)
        return ChangeImpact(summary=summary, targets=list(targets.values()))

    def _apply_new_page_rules(self, task: str, context: ProjectContext, targets: dict[str, ChangeTarget], entities: set[str]):
        repo_map = context.repository_map
        page_dirs = sorted({Path(file.path).parent for file in repo_map.files if "page" in file.path.lower() and "src" in file.path.lower()}, key=lambda path: path.as_posix())
        page_dir = "src/pages" if Path("src/pages") in page_dirs else (page_dirs[0] if page_dirs else Path("src/features/unnamed")).as_posix()

        match = _PAGE_NAME_RE.search(task)
        page_name = match.group(1) if match else "New"
        page_name = "".join(part.capitalize() for part in re.split(r"[-_]", page_name))
        new_page_path = f"{page_dir}/{page_name}Page.tsx"
        targets[new_page_path] = ChangeTarget(new_page_path, "create", 1.0, f"Proposed new page file based on task entity 'page' and existing project structure.", "page", "low")
        new_test_path = f"{page_dir}/__tests__/{page_name}Page.test.tsx"
        targets[new_test_path] = ChangeTarget(new_test_path, "create", 0.8, f"Proposed new test file for {page_name}Page.", "test", "low")

        for file in repo_map.files:
            if "router" in file.path.lower():
                targets[file.path] = ChangeTarget(file.path, "modify", 0.95, "The router needs to be modified to include the new page route.", "router", "medium")
            if "navigation" in entities and "nav" in file.path.lower():
                targets[file.path] = ChangeTarget(file.path, "modify", 0.9, "The navigation component likely needs a new link to the new page.", "navigation", "medium")
            if "layout" in file.path.lower():
                targets[file.path] = ChangeTarget(file.path, "architecture", 0.7, "Layout component is part of the UI architecture for new pages.", "layout", "medium")

    def _apply_backend_fix_rules(self, task: str, context: ProjectContext, targets: dict[str, ChangeTarget]):
        repo_map = context.repository_map
        task_keywords = _name_tokens(task) - _STOPWORDS
        for file in repo_map.files:
            is_backend = "functions" in file.path or "backend" in file.path
            if not is_backend or file.is_test:
                continue
            path_keywords = _name_tokens(file.path)
            if task_keywords & path_keywords:
                targets[file.path] = ChangeTarget(file.path, "modify", 0.9, "File path matches keywords from backend fix task.", "backend", "medium")
                # Find related test
                for rel in repo_map.relationships:
                    if rel.source == file.path and rel.kind == "tested_by":
                        targets[rel.target] = ChangeTarget(rel.target, "test", 0.85, f"Test file for {file.path}.", "test", "low")

    def _apply_generic_fix_rules(self, task: str, context: ProjectContext, targets: dict[str, ChangeTarget]):
        repo_map = context.repository_map
        task_keywords = _name_tokens(task) - _STOPWORDS
        for file in repo_map.files:
            if file.is_test:
                continue
            path_keywords = _name_tokens(file.path)
            if task_keywords & path_keywords:
                targets[file.path] = ChangeTarget(file.path, "modify", 0.8, "File path matches keywords from fix task.", "primary", self._get_risk(file.path))
                # Find dependencies and tests
                for rel in repo_map.relationships:
                    if rel.source == file.path and rel.kind == "imports":
                        targets[rel.target] = ChangeTarget(rel.target, "architecture", 0.6, f"Direct dependency of {file.path}.", "dependency", self._get_risk(rel.target))
                    if rel.source == file.path and rel.kind == "tested_by":
                        targets[rel.target] = ChangeTarget(rel.target, "test", 0.85, f"Test file for {file.path}.", "test", "low")

    def _apply_auth_rules(self, context: ProjectContext, targets: dict[str, ChangeTarget]):
        for file in context.repository_map.files:
            if "auth" in file.path.lower():
                targets[file.path] = ChangeTarget(file.path, "modify", 0.9, "Authentication-related file, relevant to auth task.", "auth", "high")

    def _apply_dependency_rules(self, context: ProjectContext, targets: dict[str, ChangeTarget]):
        for file in context.repository_map.files:
            if "package.json" in file.path:
                targets[file.path] = ChangeTarget(file.path, "modify", 0.95, "Package manifest file for dependency changes.", "dependency", "high")

    def _get_risk(self, path: str) -> "Literal['low', 'medium', 'high']":
        if _RISK_PATTERNS["high"].search(path):
            return "high"
        if _RISK_PATTERNS["medium"].search(path):
            return "medium"
        return "low"

    def _summarize(self, intents: set[str], entities: set[str], targets: dict[str, ChangeTarget]) -> str:
        intent_str = ", ".join(intents) if intents else "address"
        entity_str = ", ".join(entities) if entities else "the issue"
        creates = [t for t in targets.values() if t.role == "create"]
        modifies = [t for t in targets.values() if t.role == "modify"]

        summary = f"The task is to {intent_str} {entity_str}. "
        if creates:
            summary += f"This will likely involve creating {len(creates)} file(s), including {creates[0].path}. "
        if modifies:
            summary += f"It will also require modifying {len(modifies)} existing file(s) such as {modifies[0].path}."
        if not creates and not modifies:
            summary += "The exact files to change are not yet determined."
        return summary.strip()
